#!/usr/bin/env bash
set -euo pipefail

# ====================== 计时开始 ======================
SCRIPT_START_TIME=$(date +%s)

# ====================== 清理与信号处理 ======================
PIDS=()

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
}
trap 'cleanup TERM' TERM; trap 'cleanup INT' INT; trap 'cleanup EXIT' EXIT

# ====================== 读取参数与路径 ======================
PARAMS=${1:?Usage: 05_run_rf3.sh params.yaml}
OUTROOT=$(python scripts/get_param_yaml.py "$PARAMS" paths.work_dir)
OUTDIR="$OUTROOT/rf3_models"
MPNN_DIR="$OUTROOT/mpnn_seqs"
RF3_REPO=$(python scripts/get_param_yaml.py "$PARAMS" paths.rosettafold3_repo)

RUN_DIR="$OUTDIR/run"
mkdir -p "$OUTDIR/predictions" "$OUTDIR/logs" "$RUN_DIR"
MASTER_LOG="$OUTDIR/log.txt"; : > "$MASTER_LOG"

USE_TEMPLATE=$(python scripts/get_param_yaml.py "$PARAMS" project.use_template)
NUM_MODELS=$(python scripts/get_param_yaml.py "$PARAMS" rf3.with_template.initial.num_models)
NUM_RECYCLES=$(python scripts/get_param_yaml.py "$PARAMS" rf3.with_template.initial.num_recycles)
USE_TEMPLATES_PARAM=$(python scripts/get_param_yaml.py "$PARAMS" rf3.with_template.initial.use_templates)
HALT_ON_FAIL=$(python scripts/get_param_yaml.py "$PARAMS" compute.halt_on_fail 2>/dev/null || echo "false")
HALT_ON_FAIL_LOWER=$(echo "$HALT_ON_FAIL" | tr '[:upper:]' '[:lower:]')
WORKERS_PER_GPU=$(python scripts/get_param_yaml.py "$PARAMS" compute.workers_per_gpu 2>/dev/null || echo 1)

echo "[INFO] GPU Utilization Strategy: ${WORKERS_PER_GPU} concurrent worker(s) per GPU." | tee -a "$MASTER_LOG"

# ====================== GPU 发现与筛选 ======================
discover_gpus() {
  local cfg; local arr=()
  cfg=$(python scripts/get_param_yaml.py "$PARAMS" compute.gpus --json 2>/dev/null || echo "")
  if [[ -n "$cfg" && "$cfg" != "null" ]]; then
    cfg=$(echo "$cfg" | tr -d '[]"' | tr ',' ' ')
    read -r -a arr <<< "$cfg"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a arr <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t arr < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi
  printf '%s\n' "${arr[@]}"
}

mapfile -t GPUS < <(discover_gpus)
MINFREE=$(python scripts/get_param_yaml.py "$PARAMS" compute.min_free_mem_mb_for_gpu)
VALID_GPUS=()

for g in "${GPUS[@]:-}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$g" 2>/dev/null || echo "0")
  if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$MINFREE" ]; then
    echo "[INFO] GPU $g free mem: $free MiB. Accepted." >> "$MASTER_LOG"
    VALID_GPUS+=("$g")
  else
    echo "[WARN] GPU $g low or unknown free mem ($free MiB), skipped" >> "$MASTER_LOG"
  fi
done

NUM_GPUS=${#VALID_GPUS[@]}
echo "[INFO] Usable GPUs ($NUM_GPUS): ${VALID_GPUS[*]}" | tee -a "$MASTER_LOG"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "[ERROR] No sufficient GPU memory available." | tee -a "$MASTER_LOG"
  exit 1
fi

# ====================== 生成 METTL1 序列 ======================
if [ ! -f "./outputs/targets/mettl1_seq.fa" ]; then
  echo "[INFO] METTL1 sequence file not found. Generating..." >> "$MASTER_LOG"
  python - <<'PY'
import json
from Bio.PDB import PDBParser, Polypeptide
from Bio.SeqUtils import seq1

cand_file = "./outputs/targets/interface_candidates.json"
with open(cand_file) as f:
    cand = json.load(f)

pdb = cand["mettl1_target_pdb"]
chain_id = cand["mettl1_chain_id"]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("T", pdb)

custom_map = {"MSE":"M","SEC":"U","PYL":"O"}
seq_chars = []

for r in structure[0][chain_id]:
    if not Polypeptide.is_aa(r, standard=False):
        continue
    resname = r.get_resname().strip()
    try:
        seq_chars.append(seq1(resname, custom_map=custom_map))
    except KeyError:
        seq_chars.append("X")

seq = "".join(seq_chars)
with open("./outputs/targets/mettl1_seq.fa","w") as f:
    f.write(">METTL1\n"+seq+"\n")
print("[OK] METTL1 seq written.")
PY
else
  echo "[INFO] Found existing METTL1 sequence file." >> "$MASTER_LOG"
fi

METTL1_SEQ=$(grep -v "^>" ./outputs/targets/mettl1_seq.fa | tr -d '[:space:]' || true)
if [[ -z "${METTL1_SEQ:-}" ]]; then
  echo "[ERROR] Empty METTL1 sequence." | tee -a "$MASTER_LOG"
  exit 1
fi

# ====================== 组装 FASTA ======================
FASTA_DIR="$OUTDIR/fasta_all"
echo "[INFO] Assembling all FASTA files into $FASTA_DIR..." | tee -a "$MASTER_LOG"
rm -rf "$FASTA_DIR"
mkdir -p "$FASTA_DIR"

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

if [[ "$NUM_FILES" -eq 0 ]]; then
  echo "[ERROR] No FASTA files assembled. Check MPNN output and script logic." | tee -a "$MASTER_LOG"
  exit 1
fi

echo "[INFO] Total $NUM_FILES FASTA files correctly assembled." | tee -a "$MASTER_LOG"

# ====================== 任务预分配 ======================
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
echo "[INFO] Pre-distributing $NUM_FILES tasks to $TOTAL_WORKERS total workers ($NUM_GPUS GPUs x $WORKERS_PER_GPU workers/GPU)..." | tee -a "$MASTER_LOG"

for (( i=0; i<TOTAL_WORKERS; i++ )); do
  gpu_idx=$(( i / WORKERS_PER_GPU ))
  sub_worker_id=$(( i % WORKERS_PER_GPU ))
  GPU_ID=${VALID_GPUS[$gpu_idx]}
  WORKER_INPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_inputs"
  rm -rf "$WORKER_INPUT_DIR"
  mkdir -p "$WORKER_INPUT_DIR"
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

# ====================== Worker 函数 ======================
run_rf3_worker() {
  local gpu_id=$1
  local sub_worker_id=$2
  local worker_input_dir=$3
  local worker_output_dir=$4
  local worker_log=$5
  
  export CUDA_VISIBLE_DEVICES=$gpu_id
  
  echo "[INFO] Worker GPU ${gpu_id} sub ${sub_worker_id} started at $(date)" >> "$worker_log"
  
  # Find all FASTA files for this worker
  mapfile -t WORKER_FASTAS < <(find "$worker_input_dir" -type f -name "*.fa" | sort)
  local num_worker_files=${#WORKER_FASTAS[@]}
  
  echo "[INFO] Worker has ${num_worker_files} files to process" >> "$worker_log"
  
  for fasta in "${WORKER_FASTAS[@]}"; do
    local basename_fa=$(basename "$fasta")
    local target_name="${basename_fa%.fa}"
    local prediction_dir="$worker_output_dir/${target_name}"
    
    echo "[INFO] Processing $basename_fa" >> "$worker_log"
    
    mkdir -p "$prediction_dir"
    
    # Run RosettaFold3 inference
    # This is a placeholder command - adjust based on actual RF3 interface
    python "$RF3_REPO/run_rf3.py" \
      --input_fasta "$fasta" \
      --output_dir "$prediction_dir" \
      --num_models "$NUM_MODELS" \
      --num_recycles "$NUM_RECYCLES" \
      --use_templates "$USE_TEMPLATES_PARAM" \
      >> "$worker_log" 2>&1 || {
        echo "[ERROR] RF3 failed for $basename_fa" >> "$worker_log"
        if [[ "$HALT_ON_FAIL_LOWER" == "true" ]]; then
          echo "[ERROR] halt_on_fail=true, exiting worker" >> "$worker_log"
          return 1
        fi
      }
    
    echo "[INFO] Completed $basename_fa" >> "$worker_log"
  done
  
  echo "[INFO] Worker GPU ${gpu_id} sub ${sub_worker_id} completed at $(date)" >> "$worker_log"
}

export -f run_rf3_worker
export RF3_REPO NUM_MODELS NUM_RECYCLES USE_TEMPLATES_PARAM HALT_ON_FAIL_LOWER

# ====================== 启动所有 Workers ======================
echo "[INFO] Starting all workers..." | tee -a "$MASTER_LOG"

for (( i=0; i<TOTAL_WORKERS; i++ )); do
  gpu_idx=$(( i / WORKERS_PER_GPU ))
  sub_worker_id=$(( i % WORKERS_PER_GPU ))
  GPU_ID=${VALID_GPUS[$gpu_idx]}
  
  WORKER_INPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_inputs"
  WORKER_OUTPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_outputs"
  WORKER_LOG="$OUTDIR/logs/worker_gpu_${GPU_ID}_sub_${sub_worker_id}.log"
  
  mkdir -p "$WORKER_OUTPUT_DIR"
  : > "$WORKER_LOG"
  
  run_rf3_worker "$GPU_ID" "$sub_worker_id" "$WORKER_INPUT_DIR" "$WORKER_OUTPUT_DIR" "$WORKER_LOG" &
  PIDS+=($!)
  
  echo "[INFO] Launched worker for GPU $GPU_ID sub $sub_worker_id (PID: ${PIDS[-1]})" | tee -a "$MASTER_LOG"
done

# ====================== 等待所有 Workers ======================
echo "[INFO] Waiting for all workers to complete..." | tee -a "$MASTER_LOG"

FAILED_COUNT=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "[INFO] Worker PID $pid completed successfully" >> "$MASTER_LOG"
  else
    echo "[ERROR] Worker PID $pid failed" >> "$MASTER_LOG"
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
done

# ====================== 收集结果 ======================
echo "[INFO] Collecting all predictions into $OUTDIR/predictions..." | tee -a "$MASTER_LOG"

for (( i=0; i<TOTAL_WORKERS; i++ )); do
  gpu_idx=$(( i / WORKERS_PER_GPU ))
  sub_worker_id=$(( i % WORKERS_PER_GPU ))
  GPU_ID=${VALID_GPUS[$gpu_idx]}
  WORKER_OUTPUT_DIR="$RUN_DIR/worker_gpu_${GPU_ID}_sub_${sub_worker_id}_outputs"
  
  if [ -d "$WORKER_OUTPUT_DIR" ]; then
    find "$WORKER_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r pred_dir; do
      target_name=$(basename "$pred_dir")
      dest="$OUTDIR/predictions/$target_name"
      
      if [ -d "$dest" ]; then
        echo "[WARN] Destination $dest already exists, merging..." >> "$MASTER_LOG"
        cp -r "$pred_dir"/* "$dest/"
      else
        mv "$pred_dir" "$dest"
      fi
    done
  fi
done

# ====================== 完成 ======================
SCRIPT_END_TIME=$(date +%s)
ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo "[INFO] All RF3 workers completed." | tee -a "$MASTER_LOG"
echo "[INFO] Failed workers: $FAILED_COUNT" | tee -a "$MASTER_LOG"
echo "[INFO] Total time: ${ELAPSED}s" | tee -a "$MASTER_LOG"
echo "[INFO] Results in: $OUTDIR/predictions" | tee -a "$MASTER_LOG"

if [ "$FAILED_COUNT" -gt 0 ] && [[ "$HALT_ON_FAIL_LOWER" == "true" ]]; then
  echo "[ERROR] Some workers failed and halt_on_fail=true. Exiting with error." | tee -a "$MASTER_LOG"
  exit 1
fi

echo "[OK] RosettaFold3 inference completed successfully!" | tee -a "$MASTER_LOG"
