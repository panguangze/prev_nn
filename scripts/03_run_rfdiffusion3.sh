#!/usr/bin/env bash
set -euo pipefail

PARAMS=$1

RFDIFFUSION3_REPO=$(python scripts/get_param_yaml.py "$PARAMS" paths.rfdiffusion3_repo)
WORKDIR=$(python scripts/get_param_yaml.py "$PARAMS" paths.work_dir)
TARGETS_DIR=$(python scripts/get_param_yaml.py "$PARAMS" paths.targets_dir)

OUTDIR="${WORKDIR}/rfdiffusion3_raw"
TARGET_JSON="${TARGETS_DIR}/interface_candidates.json"
HOTSETS_JSON="${TARGETS_DIR}/hotspots_sets.json"

mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/log.txt"
: > "$LOGFILE"

test -s "$TARGET_JSON"
test -s "$HOTSETS_JSON"

BATCH_ID=$(python scripts/get_param_yaml.py "$PARAMS" project.batch_id)
DESIGNS_PER_COMBO=$(python scripts/get_param_yaml.py "$PARAMS" scale.rfdesigns_per_combo_per_lenbin)
NEI_RAD=$(python scripts/get_param_yaml.py "$PARAMS" rfdd3.neighborhood_radius)
MODEL_ONLY=$(python scripts/get_param_yaml.py "$PARAMS" rfdd3.model_only_neighbors)
T=$(python scripts/get_param_yaml.py "$PARAMS" rfdd3.inference_T || echo 50)
DIFFUSION_SCHEDULE=$(python scripts/get_param_yaml.py "$PARAMS" rfdd3.diffusion_schedule || echo "linear")
MODEL_VERSION=$(python scripts/get_param_yaml.py "$PARAMS" rfdd3.model_version || echo "v3")

METTL1_TARGET_PDB=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mettl1_target_pdb"])' "$TARGET_JSON")
TARGET_CHAIN_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mettl1_chain_id"])' "$TARGET_JSON")

echo "[INFO] target_pdb=$METTL1_TARGET_PDB" | tee -a "$LOGFILE"
echo "[INFO] target_chain=$TARGET_CHAIN_ID" | tee -a "$LOGFILE"
test -s "$METTL1_TARGET_PDB"

TARGET_SEGMENTS=$(python -c '
import sys
pdb_file, chain_id = sys.argv[1:3]

res_nums = set()
with open(pdb_file, "r") as f:
    for line in f:
        if line.startswith("ATOM") and line[21] == chain_id:
            try:
                res_nums.add(int(line[22:26]))
            except ValueError:
                pass

if not res_nums:
    print("")
    sys.exit()

sorted_res = sorted(list(res_nums))

segments = []
start_res = sorted_res[0]
end_res = sorted_res[0]

for i in range(1, len(sorted_res)):
    if sorted_res[i] == end_res + 1:
        end_res = sorted_res[i]
    else:
        segments.append(f"{start_res}-{end_res}")
        start_res = sorted_res[i]
        end_res = sorted_res[i]

segments.append(f"{start_res}-{end_res}")
final_segments = [f"{chain_id}{s}" for s in segments]
print("/".join(final_segments))
' "$METTL1_TARGET_PDB" "$TARGET_CHAIN_ID")


if [ -z "$TARGET_SEGMENTS" ]; then
    echo "[ERROR] Could not determine any residue segments for chain '$TARGET_CHAIN_ID' in PDB '$METTL1_TARGET_PDB'. Exiting." | tee -a "$LOGFILE"
    exit 1
fi
echo "[INFO] Detected target residue segments for chain $TARGET_CHAIN_ID: $TARGET_SEGMENTS" | tee -a "$LOGFILE"

readarray -t LENBINS < <(python scripts/get_param_yaml.py "$PARAMS" project.length_bins --json | python -c 'import sys,json; bins=json.load(sys.stdin);
for b in bins: print("{},{}".format(b["min"], b["max"]))')
echo "[INFO] lenbins=${LENBINS[*]}" | tee -a "$LOGFILE"
[ ${#LENBINS[@]} -gt 0 ]

# ========================== 核心逻辑修改部分 ==========================
# 1. Read parameter 'compute.max_concurrent_rf3', interpreted as "number of tasks per GPU"
# If not set, defaults to 1
TASKS_PER_GPU=$(python scripts/get_param_yaml.py "$PARAMS" compute.max_concurrent_rf3 2>/dev/null || echo 1)

# 2. 获取GPU列表和数量
mapfile -t GPUS < <(python scripts/get_param_yaml.py "$PARAMS" compute.gpus --json | tr -d '[] ' | tr ',' '\n')
NGPU=${#GPUS[@]}

# 3. 计算总的并发任务上限 (MAXJ)
MAXJ=1
if [ "$NGPU" -eq 0 ]; then
  echo "[WARN] No GPUs detected; will run in a single thread." | tee -a "$LOGFILE"
  # 没有GPU时，总并发数强制为1
  MAXJ=1
else
  # 总并发数 = (每个GPU的任务数) * (GPU数量)
  MAXJ=$(( TASKS_PER_GPU * NGPU ))
fi
echo "[INFO] Detected $NGPU GPUs. Tasks per GPU set to $TASKS_PER_GPU." | tee -a "$LOGFILE"
echo "[INFO] Total concurrency limit (MAXJ) calculated as: $MAXJ" | tee -a "$LOGFILE"
# ======================== 修改结束 ====================================


# --- run_one 函数 ---
run_one() {
  local k="$1"
  local OUTP="$2"
  local LMIN="$3"
  local LMAX="$4"
  local HOTSTR="$5"

  (
    set -e
    # 当 NGPU > 0 时，此处的逻辑依然正确，它会轮流将任务分配给每个GPU
    # 当 NGPU = 0 时，GPU_IDX 会是 -1，但 GPUS 数组为空，CUDA_VISIBLE_DEVICES 会被设置为空字符串，不影响CPU运行
    local GPU_IDX=$(( (k-1) % NGPU ))
    local GPU_ID="${GPUS[$GPU_IDX]:-}" # 使用默认值防止 NGPU=0 时出错
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    if [ -n "$GPU_ID" ]; then
        echo "[INFO] Running design $k on GPU $GPU_ID" >> "$LOGFILE"
    else
        echo "[INFO] Running design $k on CPU" >> "$LOGFILE"
    fi

    local len
    len=$(python -c 'import sys,random; seed,lmin,lmax=map(int,sys.argv[1:]); random.seed(100000+seed); print(random.randint(lmin,lmax))' "$k" "$LMIN" "$LMAX")
    local pref="$OUTP/design_${k}_len${len}"

    local CONTIG_STR="[${TARGET_SEGMENTS}/0 ${len}-${len}/0]"
    echo "[INFO] Binder Design. Target Segments: ${TARGET_SEGMENTS}, Binder Length: ${len}, Contig: ${CONTIG_STR}" >> "$LOGFILE"
    echo "[INFO] Guiding interface towards hotspots: ${HOTSTR}" >> "$LOGFILE"

    local GUIDING_POTENTIALS="['type:interface_ncontacts','type:binder_zero_dG']"
    
    set +e
    python "$RFDIFFUSION3_REPO/scripts/run_inference.py" \
      inference.input_pdb="$METTL1_TARGET_PDB" \
      inference.output_prefix="$pref" \
      inference.num_designs=1 \
      "contigmap.contigs=${CONTIG_STR}" \
      "ppi.hotspot_res=[$HOTSTR]" \
      "potentials.guiding_potentials=${GUIDING_POTENTIALS}" \
      potentials.guide_scale=2.0 \
      denoiser.noise_scale_ca=1 \
      inference.model_only_neighbors="$MODEL_ONLY" \
      inference.radius="$NEI_RAD" \
      diffuser.T="$T" \
      diffuser.schedule="$DIFFUSION_SCHEDULE" \
      model.version="$MODEL_VERSION" >> "$LOGFILE" 2>&1
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
      echo "[WARN] First attempt failed for $pref (rc=$rc). Retrying..." >> "$LOGFILE"
      sleep 2
      python "$RFDIFFUSION3_REPO/scripts/run_inference.py" \
        inference.input_pdb="$METTL1_TARGET_PDB" \
        inference.output_prefix="$pref" \
        inference.num_designs=1 \
        "contigmap.contigs=${CONTIG_STR}" \
        "ppi.hotspot_res=[$HOTSTR]" \
        "potentials.guiding_potentials=${GUIDING_POTENTIALS}" \
        potentials.guide_scale=2.0 \
        denoiser.noise_scale_ca=1 \
        inference.model_only_neighbors="$MODEL_ONLY" \
        inference.radius="$NEI_RAD" \
        diffuser.T="$T" \
        diffuser.schedule="$DIFFUSION_SCHEDULE" \
        model.version="$MODEL_VERSION" >> "$LOGFILE" 2>&1 || true
    fi
  ) >> "$LOGFILE" 2>&1
}

export -f run_one
export RFDIFFUSION3_REPO METTL1_TARGET_PDB MODEL_ONLY NEI_RAD T DIFFUSION_SCHEDULE MODEL_VERSION LOGFILE NGPU GPUS TARGET_CHAIN_ID TARGET_SEGMENTS

# 外循环与并发逻辑
python -c 'import json,sys; d=json.load(open(sys.argv[1]));
[print(i, S["hotspot_res_str"], S["hotspot_count"]) for i,S in enumerate(d)]' "$HOTSETS_JSON" | while read -r IDX HOTSTR NOS; do
  HOTSTR_CONTIG=$(echo "$HOTSTR" | tr -d ':')
  for lb in "${LENBINS[@]}"; do
    LMIN=${lb%,*}; LMAX=${lb#*,}
    OUTP="$OUTDIR/batch-${BATCH_ID}_set-${IDX}_hs-${NOS}_len-${LMIN}-${LMAX}"
    mkdir -p "$OUTP"

    echo "[INFO] Starting concurrent designs for set $IDX, len $LMIN-$LMAX (DESIGNS_PER_COMBO: $DESIGNS_PER_COMBO)" | tee -a "$LOGFILE"
    
    # 这个循环和并发控制逻辑本身是正确的，现在它将使用我们新计算的 MAXJ
    for (( k=1; k<=$DESIGNS_PER_COMBO; k++ )); do
      # 当后台任务数量达到 MAXJ 上限时，等待任何一个任务完成
      while [[ $(jobs -p | wc -l) -ge $MAXJ ]]; do
        wait -n
      done
      # 启动一个新任务到后台
      run_one "$k" "$OUTP" "$LMIN" "$LMAX" "$HOTSTR_CONTIG" &
    done

    echo "[INFO] All designs launched for set $IDX, len $LMIN-$LMAX. Waiting for completion..." | tee -a "$LOGFILE"
    wait
  done
done

echo "[OK] All RFdiffusion3 tasks completed. Results in $OUTDIR" | tee -a "$LOGFILE"
