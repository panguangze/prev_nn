#!/usr/bin/env bash
set -euo pipefail

# ====================== 计时开始 ======================
SCRIPT_START_TIME=$(date +%s)

# ====================== 清理与信号处理 ======================
PIDS=()
COPY_DIRS=()

cleanup() {
  local sig=${1:-TERM}
  echo "[CLEANUP] Signal $sig. Killing ${#PIDS[@]} worker(s)..." >&2
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then kill -"$sig" -"$p" 2>/dev/null || true; fi
  done
  sleep 1
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then kill -9 -"$p" 2>/dev/null || true; fi
  done
  for d in "${COPY_DIRS[@]:-}"; do
    [[ -n "$d" && -d "$d" ]] && rm -rf "$d" || true
  done
}
trap 'cleanup TERM' TERM; trap 'cleanup INT' INT; trap 'cleanup EXIT' EXIT

# ====================== 环境变量 ======================
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
if [[ -n "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="$(echo "$XLA_FLAGS" | sed 's/--xla_gpu_enable_triton=false//g' | tr -s ' ')"
fi

# ====================== 读取参数与路径 ======================
PARAMS=${1:?Usage: 05_run_af2_multimer.sh params.yaml}
OUTROOT=$(python scripts/get_param_yaml.py "$PARAMS" paths.work_dir)
OUTDIR="$OUTROOT/af2_models"
MPNN_DIR="$OUTROOT/mpnn_seqs"
TEMPLATE_SRC=$(python scripts/get_param_yaml.py "$PARAMS" paths.templates_dir)

RUN_DIR="$OUTDIR/run"
mkdir -p "$OUTDIR/predictions" "$OUTDIR/logs" "$RUN_DIR"
TEMPLATES_TMP_ROOT="$OUTDIR/templates_tmp"
mkdir -p "$TEMPLATES_TMP_ROOT"
MASTER_LOG="$OUTDIR/log.txt"; : > "$MASTER_LOG"

USE_TEMPLATE=$(python scripts/get_param_yaml.py "$PARAMS" project.use_template)
NUM_MODELS=$(python scripts/get_param_yaml.py "$PARAMS" af2.with_template.initial.num_models)
NUM_RECYCLES=$(python scripts/get_param_yaml.py "$PARAMS" af2.with_template.initial.num_recycles)
AMBER=$(python scripts/get_param_yaml.py "$PARAMS" af2.with_template.initial.amber_relax)
HALT_ON_FAIL=$(python scripts/get_param_yaml.py "$PARAMS" compute.halt_on_fail 2>/dev/null || echo "false")
HALT_ON_FAIL_LOWER=$(echo "$HALT_ON_FAIL" | tr '[:upper:]' '[:lower:]')
WORKERS_PER_GPU=$(python scripts/get_param_yaml.py "$PARAMS" compute.workers_per_gpu 2>/dev/null || echo 1)

echo "[INFO] GPU Utilization Strategy: ${WORKERS_PER_GPU} concurrent worker(s) per GPU with private template copies." | tee -a "$MASTER_LOG"

# ====================== colabfold 命令检测 ======================
if command -v colabfold_batch >/dev/null 2>&1; then COLABFOLD="colabfold_batch"; else COLABFOLD="python -m colabfold.batch"; fi
HELP=$($COLABFOLD --help 2>&1 || true)
MODEL_FLAG=""; MODEL_TYPE=""
if echo "$HELP" | grep -q -- "--model-type"; then MODEL_FLAG="--model-type"; MODEL_TYPE="alphafold2_multimer_v3"; elif echo "$HELP" | grep -q -- "--models"; then MODEL_FLAG="--models"; MODEL_TYPE="AlphaFold2-multimer-v3"; fi
# ====================== GPU 发现与筛选 ======================
discover_gpus() {
  local cfg; local arr=(); cfg=$(python scripts/get_param_yaml.py "$PARAMS" compute.gpus --json 2>/dev/null || echo ""); if [[ -n "$cfg" && "$cfg" != "null" ]]; then cfg=$(echo "$cfg" | tr -d '[]"' | tr ',' ' '); read -r -a arr <<< "$cfg"; elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=',' read -r -a arr <<< "$CUDA_VISIBLE_DEVICES"; elif command -v nvidia-smi >/dev/null 2>&1; then mapfile -t arr < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi; printf '%s\n' "${arr[@]}";
}
mapfile -t GPUS < <(discover_gpus)
MINFREE=$(python scripts/get_param_yaml.py "$PARAMS" compute.min_free_mem_mb_for_gpu)
VALID_GPUS=()
for g in "${GPUS[@]:-}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$g" 2>/dev/null || echo "0")
  if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$MINFREE" ]; then echo "[INFO] GPU $g free mem: $free MiB. Accepted." >> "$MASTER_LOG"; VALID_GPUS+=("$g"); else echo "[WARN] GPU $g low or unknown free mem ($free MiB), skipped" >> "$MASTER_LOG"; fi
done
NUM_GPUS=${#VALID_GPUS[@]}
echo "[INFO] Usable GPUs ($NUM_GPUS): ${VALID_GPUS[*]}" | tee -a "$MASTER_LOG"
if [ "$NUM_GPUS" -eq 0 ]; then echo "[ERROR] No sufficient GPU memory available." | tee -a "$MASTER_LOG"; exit 1; fi
# ====================== 生成 METTL1 序列 ======================
if [ ! -f "./outputs/targets/mettl1_seq.fa" ]; then
  echo "[INFO] METTL1 sequence file not found. Generating..." >> "$MASTER_LOG"
  python - <<'PY'
import json; from Bio.PDB import PDBParser, Polypeptide; from Bio.SeqUtils import seq1
cand_file = "./outputs/targets/interface_candidates.json"
with open(cand_file) as f: cand = json.load(f)
pdb = cand["mettl1_target_pdb"]; chain_id = cand["mettl1_chain_id"]
parser = PDBParser(QUIET=True); structure = parser.get_structure("T", pdb)
custom_map = {"MSE":"M","SEC":"U","PYL":"O"}; seq_chars=[]
for r in structure[0][chain_id]:
    if not Polypeptide.is_aa(r, standard=False): continue
    resname = r.get_resname().strip()
    try: seq_chars.append(seq1(resname, custom_map=custom_map))
    except KeyError: seq_chars.append("X")
seq = "".join(seq_chars)
with open("./outputs/targets/mettl1_seq.fa","w") as f: f.write(">METTL1\n"+seq+"\n")
print("[OK] METTL1 seq written.")
PY
else
  echo "[INFO] Found existing METTL1 sequence file." >> "$MASTER_LOG"
fi
METTL1_SEQ=$(grep -v "^>" ./outputs/targets/mettl1_seq.fa | tr -d '[:space:]' || true)
if [[ -z "${METTL1_SEQ:-}" ]]; then echo "[ERROR] Empty METTL1 sequence." | tee -a "$MASTER_LOG"; exit 1; fi
# ====================== 组装 FASTA ======================
FASTA_DIR="$OUTDIR/fasta_all"; echo "[INFO] Assembling all FASTA files into $FASTA_DIR..." | tee -a "$MASTER_LOG"; rm -rf "$FASTA_DIR"; mkdir -p "$FASTA_DIR"
find "$MPNN_DIR" -type f -path "*/seqs/*.fa" | sort | while read -r mpnn_multiseq_fa; do
  [[ ! -s "$mpnn_multiseq_fa" ]] && continue
  design_backbone_name=$(basename "${mpnn_multiseq_fa%.fa}")
  awk -v mettl1_seq="$METTL1_SEQ" -v out_dir="$FASTA_DIR" -v backbone_name="$design_backbone_name" \
      'BEGIN{RS=">";FS="\n"}
       match($1, /sample=([^, ]+)/, arr) {
         header=$1; sequence=""; for(i=2;i<=NF;i++){sequence=sequence $i}; gsub(/[ \t\r\n]/,"",sequence);
         if(sequence==""){next}; sample_id=arr[1]; out_file=out_dir"/"backbone_name"_sample_"sample_id".fa";
         print ">METTL1:"backbone_name"_sample_"sample_id > out_file; print mettl1_seq":"sequence >> out_file; close(out_file)
       }' "$mpnn_multiseq_fa"
done
mapfile -t ALL_FASTAS < <(find "$FASTA_DIR" -type f -name "*.fa" | sort)
NUM_FILES=${#ALL_FASTAS[@]}
if [[ "$NUM_FILES" -eq 0 ]]; then echo "[ERROR] No FASTA files assembled. Check MPNN output and script logic." | tee -a "$MASTER_LOG"; exit 1; fi
echo "[INFO] Total $NUM_FILES FASTA files correctly assembled." | tee -a "$MASTER_LOG"

# ====================== 任务预分配 ======================
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
echo "[INFO] Pre-distributing $NUM_FILES tasks to $TOTAL_WORKERS total workers ($NUM_GPUS GPUs x $WORKERS_PER_GPU workers/GPU)..." | tee -a "$MASTER_LOG"

for (( i=0; i<TOTAL_WORKERS; i++ )); do
  gpu_idx=$(( i / WORKERS_PER_GPU ))
  sub_worker_id=$(( i % WORKERS_PER_GPU ))
  GPU_ID=${VALID_GPUS[$gpu_idx]}
  WORKER_INPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_inputs"
  rm -rf "$WORKER_INPUT_DIR"; mkdir -p "$WORKER_INPUT_DIR"
done

for (( i=0; i<NUM_FILES; i++ )); do
  worker_slot_idx=$(( i % TOTAL_WORKERS ))
  gpu_idx=$(( worker_slot_idx / WORKERS_PER_GPU ))
  sub_worker_id=$(( worker_slot_idx % WORKERS_PER_GPU ))
  GPU_ID=${VALID_GPUS[$gpu_idx]}
  WORKER_INPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_inputs"
  
  ABSOLUTE_FASTA_PATH=$(readlink -f "${ALL_FASTAS[$i]}")
  ln -s "$ABSOLUTE_FASTA_PATH" "$WORKER_INPUT_DIR/"
done

# ====================== 构造参数 ======================
RELAX_ARGS=()
if [[ "$(echo "$AMBER" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  RELAX_ARGS+=( --amber --num-relax 1 )
  if echo "$HELP" | grep -q -- "--use-gpu-relax"; then RELAX_ARGS+=( --use-gpu-relax ); fi
fi
INFER_ARGS=(
  ${MODEL_FLAG:+$MODEL_FLAG} ${MODEL_FLAG:+$MODEL_TYPE}
  --num-recycle "$NUM_RECYCLES"
  --num-models "$NUM_MODELS"
  --msa-mode single_sequence
  --overwrite-existing-results
  --recompile-padding 10
)
FLAT_ARGS=( "${INFER_ARGS[@]}" "${RELAX_ARGS[@]}" )

# ====================== Worker 循环函数 ======================
run_worker_loop() {
    local gpu_id="$1"
    local worker_id="$2"
    local input_dir="$3"
    local output_dir="$4"
    local template_dir="$5"
    local colabfold_cmd="$6"
    shift 6
    local flat_args=("$@")

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    local num_tasks=$(find "$input_dir" -type l -name "*.fa" 2>/dev/null | wc -l)
    [[ $num_tasks -eq 0 ]] && { echo "[WORKER $worker_id] No tasks assigned. Exiting."; return; }
    
    local task_count=0
    local worker_start_time=$(date +%s)

    echo "[WORKER $worker_id] Starting on GPU $gpu_id... Assigned $num_tasks tasks."

    for fasta_file in "$input_dir"/*.fa; do
        [[ -L "$fasta_file" ]] || continue
        task_count=$((task_count + 1))
        local base_name=$(basename "$fasta_file" .fa)
        echo "------------------------------------------------------------"
        echo "[WORKER $worker_id] Processing task $task_count/$num_tasks: $base_name"
        
        local task_start_time=$(date +%s)
        local cmd=( "$colabfold_cmd" )
        cmd+=( "${flat_args[@]}" )
        if [[ -n "$template_dir" ]]; then
            cmd+=( --templates --custom-template-path "$template_dir" )
        fi
        cmd+=( "$fasta_file" "$output_dir" )

        "${cmd[@]}"
        local exit_code=$?
        local task_end_time=$(date +%s)
        local task_duration=$((task_end_time - task_start_time))

        if [[ $exit_code -ne 0 ]]; then
            echo "[WORKER $worker_id] ERROR: colabfold failed for $base_name with exit code $exit_code. Task duration: ${task_duration}s."
        else
            echo "[WORKER $worker_id] Finished task $task_count/$num_tasks: $base_name. Task duration: ${task_duration}s."
        fi
    done
    
    local worker_end_time=$(date +%s)
    local worker_duration=$((worker_end_time - worker_start_time))
    echo "[WORKER $worker_id] All assigned tasks completed. Total worker time: ${worker_duration}s."
}
export -f run_worker_loop

# ====================== 启动 Worker (已修正模板复制逻辑) ======================
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
echo "[INFO] Launching $TOTAL_WORKERS total worker(s)..." | tee -a "$MASTER_LOG"
for (( i=0; i<NUM_GPUS; i++ )); do
  GPU_ID=${VALID_GPUS[$i]}
  
  for (( j=0; j<WORKERS_PER_GPU; j++ )); do
    WORKER_ID="gpu_${GPU_ID}_sub_${j}"
    WORKER_INPUT_DIR="$RUN_DIR/worker_${WORKER_ID}_inputs"
    WORKER_OUTPUT_DIR="$OUTDIR/predictions"
    WORKER_LOG="$OUTDIR/logs/worker_${WORKER_ID}.log"
    TEMPLATE_COPY_DIR=""

    if [[ "$(echo "$USE_TEMPLATE" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
      # --- 关键修正 ---
      # 为每个worker创建其私有的模板目录，以防止竞争条件。
      # 目录名使用完整的WORKER_ID确保唯一性。
      TEMPLATE_COPY_DIR="$TEMPLATES_TMP_ROOT/${WORKER_ID}"
      echo "[INFO] Preparing private template copy for worker $WORKER_ID at $TEMPLATE_COPY_DIR" | tee -a "$MASTER_LOG"
      
      # 始终为worker创建全新的、干净的副本
      rm -rf "$TEMPLATE_COPY_DIR"
      
      # 尝试使用硬链接进行快速、低空间占用的复制，如果失败则回退到完整复制。
      if ! cp -al "$TEMPLATE_SRC/." "$TEMPLATE_COPY_DIR/" 2>/dev/null; then
        if ! rsync -a --copy-links "$TEMPLATE_SRC/" "$TEMPLATE_COPY_DIR/" 2>/dev/null; then
          cp -a "$TEMPLATE_SRC/." "$TEMPLATE_COPY_DIR/"
        fi
      fi
      
      # 将此唯一的目录添加到清理列表中
      COPY_DIRS+=("$TEMPLATE_COPY_DIR")
    fi

    CMD_STR="run_worker_loop $GPU_ID $WORKER_ID $WORKER_INPUT_DIR $WORKER_OUTPUT_DIR $TEMPLATE_COPY_DIR $COLABFOLD ${FLAT_ARGS[*]}"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "[INFO] Starting worker $WORKER_ID for GPU $GPU_ID. Log: $WORKER_LOG" | tee -a "$MASTER_LOG"
    echo "[INFO] Command logic: $CMD_STR" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"

    run_worker_loop "$GPU_ID" "$WORKER_ID" "$WORKER_INPUT_DIR" "$WORKER_OUTPUT_DIR" "$TEMPLATE_COPY_DIR" "$COLABFOLD" "${FLAT_ARGS[@]}" > "$WORKER_LOG" 2>&1 &
    PIDS+=("$!")
    echo "[INFO] Worker $WORKER_ID started, PID=$!" | tee -a "$MASTER_LOG"; sleep 0.2
  done
done

# ====================== 监控与结束 ======================
FAIL_FILE="$OUTDIR/run/failed.list"; : > "$FAIL_FILE"
monitor_loop() {
  while true; do
    alive=0; for p in "${PIDS[@]:-}"; do if kill -0 "$p" 2>/dev/null; then alive=$((alive+1)); fi; done
    if [[ $alive -eq 0 ]]; then echo "[MASTER] All workers have finished." | tee -a "$MASTER_LOG"; break; fi
    if [[ "$HALT_ON_FAIL_LOWER" == "true" ]]; then
      if grep -q -i -E "error|traceback|OOM" "$OUTDIR/logs/worker_"*.log; then
        grep -i -E "error|traceback|OOM" "$OUTDIR/logs/worker_"*.log > "$FAIL_FILE"
        echo "[MASTER] Detected failure in worker logs and HALT_ON_FAIL=true. Stopping all workers..." | tee -a "$MASTER_LOG";
        cleanup TERM;
        break;
      fi
    fi
    sleep 5
  done
}
monitor_loop

RET=0
for p in "${PIDS[@]:-}"; do if ! wait "$p"; then echo "[WARN] Worker with PID $p exited with an error." | tee -a "$MASTER_LOG"; RET=1; fi; done
trap - EXIT; cleanup EXIT

grep -L "All assigned tasks completed" "$OUTDIR/logs/worker_"*.log | sed 's/.*\(worker_gpu_.*_sub_.*\)\.log/Worker \1 might have failed or been interrupted./' > "$FAIL_FILE" || true
FAILS=$(wc -l < "$FAIL_FILE" | tr -d '[:space:]')
echo "[DONE] All workers finished. Total workers: $TOTAL_WORKERS, Potential Failures: $FAILS" | tee -a "$MASTER_LOG"
if [[ "$FAILS" -gt 0 ]]; then echo "Please check the following worker logs for details:" | tee -a "$MASTER_LOG"; cat "$FAIL_FILE" | tee -a "$MASTER_LOG"; fi

# ====================== 计时结束与报告 ======================
SCRIPT_END_TIME=$(date +%s)
DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
H=$((DURATION / 3600))
M=$(((DURATION % 3600) / 60))
S=$((DURATION % 60))
FORMATTED_DURATION=$(printf '%02d:%02d:%02d' "$H" "$M" "$S")

echo "============================================================" | tee -a "$MASTER_LOG"
echo "[TIMER] Total execution time: $FORMATTED_DURATION ($DURATION seconds)." | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

exit $RET
