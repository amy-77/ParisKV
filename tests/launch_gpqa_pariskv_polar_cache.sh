#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/qwang/q/thalia/ZoomKV/PolarANN_density"
RUN_DIR="${ROOT}/run"
PYBIN="/home/qwang/software/miniforge3/envs/retroinfer/bin/python"
CONDA_ENV="retroinfer"
TS="${TS:-pariskv_$(date +%Y%m%d_%H%M%S)}"

DATA_PATH="${DATA_PATH:-/data/qwang/q/kvcache/data/gpqa/gpqa_diamond.jsonl}"
CODEBOOK_PATH="${CODEBOOK_PATH:-/data/qwang/q/thalia/ZoomKV/PolarANN_density/turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json}"
MODEL_Q4B="${MODEL_Q4B:-/data/qwang/q/models/Qwen3-4B-Thinking-2507}"
MODEL_Q8B="${MODEL_Q8B:-/data/qwang/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
MODEL_DSR1="${MODEL_DSR1:-/data/qwang/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1}"

COMMON_ARGS=(
  --data_path "${DATA_PATH}"
  --codebook_path "${CODEBOOK_PATH}"
  --num_samples -1
  --attention_type PolarANN
  --polar_cache_module cache_hub.polar_cache
  --collision_ratio 0.3
  --candidate_ratio 0.12
  --final_topk 100
  --adaptive_topk_enabled 0
  --sink_size 64
  --local_size 128
  --dynamic_update_interval 512
  --full_attention_threshold 2000
  --temperature 0.6
  --top_p 0.95
  --top_k 20
  --max_new_tokens 32800
  --device cuda:0
  --dtype bf16
  --enable_offload 1
)

run_one() {
  local model_path="$1"
  local gpu="$2"
  local tag="$3"
  local out_dir="${ROOT}/results/gpqa_pariskv_${tag}"
  local output_file="${out_dir}/gpqa_${tag}_pariskv_topk100_l128_u512_n-1_${TS}.jsonl"
  local log_file="${out_dir}/gpqa_${tag}_pariskv_topk100_l128_u512_n-1_${TS}.log"
  mkdir -p "${out_dir}"
  (
    cd "${RUN_DIR}"
    export PYTHONUNBUFFERED=1
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export FLASHINFER_WORKSPACE_BASE="${ROOT}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/qwang/.cache/huggingface/datasets}"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export POLAR_CACHE_MODULE=cache_hub.polar_cache
    unset POLAR_FIXED_CANDIDATE_RATIO
    unset POLAR_ADAPTIVE_FORCE_CANDIDATE_RATIO
    unset POLAR_ADAPTIVE_FORCE_COLLISION_RATIO
    unset POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS
    unset POLAR_ADAPTIVE_DYNAMIC_COLL_RATIOS
    unset POLAR_ADAPTIVE_DYNAMIC_CAND_THRESHOLDS
    echo "=========================================="
    echo "  GPQA Diamond ParisKV polar_cache.py: ${tag}"
    echo "=========================================="
    echo "module       : ${POLAR_CACHE_MODULE}"
    echo "model        : ${model_path}"
    echo "data         : ${DATA_PATH}"
    echo "codebook     : ${CODEBOOK_PATH}"
    echo "gpu          : cuda:${gpu}"
    echo "final_topk   : 100"
    echo "layout       : sink=64 local=128 update=512 full_attn<=2000"
    echo "max_new_tok  : 32800"
    echo "adaptive     : polar_cache.py _get_adaptive_ratios; no force/dynamic env override"
    echo "output       : ${output_file}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYBIN}" test_gpqa.py \
      --model_name "${model_path}" \
      "${COMMON_ARGS[@]}" \
      --output_file "${output_file}"
  ) 2>&1 | tee "${log_file}"
}

echo "[supervisor] TS=${TS}"
echo "[supervisor] Q4B and Q8B start in parallel; DSR1 starts after Q4B finishes."

run_one "${MODEL_Q4B}" 0 qwen3_4b_thinking_2507 &
pid_q4b=$!
echo "[supervisor] Q4B pid=${pid_q4b}"

run_one "${MODEL_Q8B}" 1 qwen3_8b &
pid_q8b=$!
echo "[supervisor] Q8B pid=${pid_q8b}"

wait "${pid_q4b}"
echo "[supervisor] Q4B finished; starting DSR1 on GPU0."

run_one "${MODEL_DSR1}" 0 deepseek_r1_distill_llama_8b &
pid_dsr1=$!
echo "[supervisor] DSR1 pid=${pid_dsr1}"

wait "${pid_q8b}"
wait "${pid_dsr1}"
echo "[supervisor] all GPQA ParisKV polar_cache.py runs finished."
