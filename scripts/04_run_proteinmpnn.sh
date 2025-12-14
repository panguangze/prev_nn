#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 这个脚本假定您已经在包含了 GNU Parallel 的 conda 环境中运行
# 例如:
# conda activate your_env
# bash scripts/04_run_proteinmpnn.sh config/params.yaml
# ==============================================================================

PARAMS=$1
MPNN=$(python scripts/get_param_yaml.py $PARAMS paths.proteinmpnn)
RFDIR=$(python scripts/get_param_yaml.py $PARAMS paths.work_dir)/rfdiffusion_raw
OUTDIR=$(python scripts/get_param_yaml.py $PARAMS paths.work_dir)/mpnn_seqs
LOGFILE="$OUTDIR/log.txt"
mkdir -p "$OUTDIR"; : > "$LOGFILE"

NUMSEQ_INIT=$(python scripts/get_param_yaml.py $PARAMS scale.mpnn_num_seq_per_backbone_initial)
SEED=$(python scripts/get_param_yaml.py $PARAMS project.seed)

# 函数：发现可用的GPU
discover_gpus() {
  local arr=()
  local cfg
  cfg=$(python scripts/get_param_yaml.py "$PARAMS" compute.gpus --json 2>/dev/null || echo "")
  if [ -n "$cfg" ] && echo "$cfg" | grep -q '[0-9A-Za-z]'; then
    cfg=$(echo "$cfg" | tr -d '[]"' | tr ',' ' ')
    read -r -a arr <<< "$cfg"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a arr <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t arr < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi
  printf '%s\n' "${arr[@]}"
}

# ========================== 核心逻辑修正部分 ==========================
# 1. 读取参数 'compute.max_concurrent_mpnn'，作为“每个GPU希望运行的任务数”
JOBS_PER_GPU=$(python scripts/get_param_yaml.py "$PARAMS" compute.max_concurrent_mpnn 2>/dev/null || echo "1")

# 2. 发现GPU并获取数量
mapfile -t GPUS < <(discover_gpus)
NGPU=${#GPUS[@]}

# 3. 计算 GNU Parallel 使用的总并发任务上限 (MAXJ)
if [ "$NGPU" -eq 0 ]; then
  echo "[WARN] No GPUs detected. Running on CPU with a concurrency of 1." | tee -a "$LOGFILE"
  # 没有GPU时，总并发数强制为1
  MAXJ=1
else
  # 总并发数 = (每个GPU的任务数) * (GPU数量)
  MAXJ=$(( JOBS_PER_GPU * NGPU ))
  echo "[INFO] Detected $NGPU GPUs. Jobs per GPU is set to $JOBS_PER_GPU." | tee -a "$LOGFILE"
fi
echo "[INFO] Total concurrency limit for GNU Parallel (MAXJ) is set to: $MAXJ" | tee -a "$LOGFILE"
# ======================== 修正结束 ====================================

# 函数：处理单个PDB文件
process_pdb() {
  (
    set -e
    local gpu="$1"
    local pdb="$2"
    local log_prefix

    if [ -n "$gpu" ]; then
      log_prefix="[GPU $gpu]"
      export CUDA_VISIBLE_DEVICES="$gpu"
    else
      log_prefix="[CPU]"
      unset CUDA_VISIBLE_DEVICES
    fi

    echo "$log_prefix $(date '+%F %T') Processing $(basename "$pdb")..."

    local outpref="$OUTDIR/$(basename "${pdb%.pdb}")"
    mkdir -p "$outpref"

    local chain_info
    chain_info=$(python scripts/get_chain_info.py "$pdb")

    if [ -z "$chain_info" ] || [ "$chain_info" = "SKIP" ]; then
      echo "$log_prefix Skip $pdb (chain_info issue)"
      return 0
    fi

    local binder_chain
    binder_chain=$(echo "$chain_info" | awk '{print $2}' | cut -d' ' -f1)
    echo "$log_prefix Binder chain to design: '$binder_chain'"

    python "$MPNN" \
      --pdb_path "$pdb" \
      --pdb_path_chains "$binder_chain" \
      --out_folder "$outpref" \
      --num_seq_per_target "$NUMSEQ_INIT" \
      --sampling_temp "0.35" \
      --seed "$SEED"

    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "$log_prefix ERROR: MPNN failed for $pdb (rc=$rc)"
    else
      echo "$log_prefix OK: MPNN for $pdb"
    fi
    return $rc
  ) >> "$LOGFILE" 2>&1
}

# 导出函数和变量，让 parallel 可以访问它们
export -f process_pdb
export OUTDIR MPNN NUMSEQ_INIT SEED LOGFILE NGPU

# ========================== 传递GPU列表的关键修正 ==========================
# 将 GPUS 数组转换为一个简单的、空格分隔的字符串并导出。
# 这是将数组信息传递给 parallel 子 shell 的最可靠方法。
# 例如，如果 GPUS=(0 1 3)，GPUS_STR 将会是 "0 1 3"。
export GPUS_STR="${GPUS[*]}"
# ========================================================================


# 查找所有需要处理的PDB文件
mapfile -t PDBS < <(find "$RFDIR" -name "*.pdb" -not -path "*/traj/*" | sort)
if [ "${#PDBS[@]}" -eq 0 ]; then
  echo "[WARN] No PDBs found under $RFDIR" | tee -a "$LOGFILE"
  exit 0
fi

# ==============================================================================
# ====================== 使用 GNU Parallel 的并行逻辑 (已修正) ================
# ==============================================================================
echo "[INFO] Starting concurrent processing of ${#PDBS[@]} PDBs using GNU Parallel. Concurrency: $MAXJ" | tee -a "$LOGFILE"

# 将所有 PDB 文件的路径通过管道传给 parallel
printf "%s\n" "${PDBS[@]}" | parallel \
    --jobs "$MAXJ" \
    --bar \
    --joblog "$OUTDIR/parallel_joblog.txt" \
    --eta \
    '
    # 在 parallel 的每个任务内部
    gpu_to_use=""
    if [ "$NGPU" -gt 0 ]; then
        # ========================== 内部逻辑修正 ==========================
        # 1. 从我们导出的环境变量 GPUS_STR 中安全地重建 GPU 数组
        read -r -a GPUS_IN_JOB <<< "$GPUS_STR"

        # 2. 使用任务编号 {#} (从1开始) 进行轮询分配 GPU
        #    这会将任务1分配给GPU 0, 任务2给GPU 1, ..., 任务(N+1)再回到GPU 0
        gpu_index=$(( ({#} - 1) % NGPU ))

        # 3. 从重建好的数组中获取正确的 GPU ID
        gpu_to_use=${GPUS_IN_JOB[$gpu_index]}
        # ======================== 修正结束 ====================================
    fi
    # {} 是 parallel 的占位符，代表输入的每一行（即一个PDB路径）
    process_pdb "$gpu_to_use" "{}"
'

echo "[OK] GNU Parallel finished. ProteinMPNN sequences -> $OUTDIR"
