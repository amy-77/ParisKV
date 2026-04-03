#!/bin/bash
#
# LongBench-v2 evaluation.
#
# Prerequisites: activate your Python/conda environment before running.
#
# Usage:
#   ./run_longbench_v2.sh [num_samples] [attention_type] [domain] [difficulty] [length]
#     [device] [enable_recall] [static_pattern_end] [max_new_tokens] [sink_size] [local_size]
#     [sample_offset] [final_topk] [enable_offload] [model_name] [enable_cot]
#     [cot_max_new_tokens] [answer_max_new_tokens] [polar_cache_module]
#
# Positional arguments:
#   1  num_samples           Number of samples to evaluate (-1 = all 503)          [default: -1]
#   2  attention_type        PolarANN | Full_Flash_Attn                            [default: PolarANN]
#   3  domain                Filter by domain (e.g. single-document_qa), "" = all  [default: ""]
#   4  difficulty            Filter by difficulty: easy | hard, "" = all            [default: ""]
#   5  length                Filter by length: short | medium | long, "" = all     [default: ""]
#   6  device                CUDA device                                           [default: cuda:0]
#   7  enable_recall         Compute Recall@K metric (true/false)                  [default: ""]
#   8  static_pattern_end    Local-window size override for PolarANN               [default: ""]
#   9  max_new_tokens        Maximum new tokens (non-CoT mode only)                [default: 1024]
#  10  sink_size             Sink-zone size override for PolarANN                  [default: ""]
#  11  local_size            Local-zone size override for PolarANN                 [default: ""]
#  12  sample_offset         Start index for resuming partial runs                 [default: ""]
#  13  final_topk            Top-K after RaBitQ reranking override                 [default: ""]
#  14  enable_offload        Offload retrieval-zone KV to CPU (0/1)               [default: ""]
#  15  model_name            HuggingFace model name or local path                  [default: Qwen/Qwen3-8B]
#  16  enable_cot            Enable Chain-of-Thought two-stage reasoning (0/1)     [default: 1]
#  17  cot_max_new_tokens    CoT stage-1 (reasoning) max new tokens                [default: 1024]
#  18  answer_max_new_tokens CoT stage-2 (final answer) max new tokens             [default: 128]
#  19  polar_cache_module    Python module path for PolarANN cache                 [default: cache_hub.polar_cache]
#
# Environment variables (optional):
#   POLAR_CACHE_MODULE  Override arg 19 (takes precedence)
#   NO_RESUME           Set non-empty to disable resume from existing output
#   MAX_CONTEXT_TOKENS  Skip samples exceeding this token count
#   FILENAME_SUFFIX     Append to auto-generated output filename
#
# Examples:
#   ./run_longbench_v2.sh 10 Full_Flash_Attn
#   ./run_longbench_v2.sh 20 PolarANN
#   ./run_longbench_v2.sh -1 PolarANN "" easy short cuda:0
#   ./run_longbench_v2.sh 1 PolarANN "" easy short cuda:0 true 512 128 8 1024
#
# Optional profiling (replace output path as needed):
#   nsys profile --output=<PROFILE_PREFIX> ./run_longbench_v2.sh 1 PolarANN "" easy short cuda:0 true 512

set -euo pipefail

NUM_SAMPLES=${1:--1}
ATTENTION_TYPE=${2:-PolarANN}
DOMAIN=${3:-""}
DIFFICULTY=${4:-""}
LENGTH=${5:-""}
DEVICE=${6:-"cuda:0"}
ENABLE_RECALL=${7:-""}
STATIC_PATTERN_END=${8:-""}
MAX_NEW_TOKENS=${9:-1024}
SINK_SIZE=${10:-""}
LOCAL_SIZE=${11:-""}
SAMPLE_OFFSET=${12:-""}
FINAL_TOPK=${13:-""}
ENABLE_OFFLOAD=${14:-""}
MODEL_NAME_OVERRIDE=${15:-""}
ENABLE_COT=${16:-"1"}
COT_MAX_NEW_TOKENS=${17:-1024}
ANSWER_MAX_NEW_TOKENS=${18:-128}
if [ -n "${POLAR_CACHE_MODULE:-}" ]; then
    :
else
    POLAR_CACHE_MODULE=${19:-"cache_hub.polar_cache"}
    export POLAR_CACHE_MODULE
fi

if [ -n "${MODEL_NAME_OVERRIDE}" ]; then
    MODEL_NAME="${MODEL_NAME_OVERRIDE}"
else
    MODEL_NAME="Qwen/Qwen3-8B"
fi
DTYPE="bf16"

DATASET_NAME="zai-org/LongBench-v2"
SPLIT_NAME="train"

if [ "${ENABLE_COT}" = "1" ]; then
    TEMPERATURE=0.1
else
    TEMPERATURE=0.0
fi
TOP_P=0.95
TOP_K=20
FORCE_PROCESS=true

TIMESTAMP=$(date +"%Y%m%d")
OUTPUT_DIR="../results/longbench_v2"
mkdir -p "${OUTPUT_DIR}"

FILENAME_PARTS="${ATTENTION_TYPE}"
if [ -n "${FINAL_TOPK}" ]; then
    FILENAME_PARTS="${FILENAME_PARTS}_top${FINAL_TOPK}"
fi
if [ -n "${DIFFICULTY}" ]; then
    FILENAME_PARTS="${FILENAME_PARTS}_${DIFFICULTY}"
fi
if [ -n "${LENGTH}" ]; then
    FILENAME_PARTS="${FILENAME_PARTS}_${LENGTH}"
fi
if [ -n "${FILENAME_SUFFIX:-}" ]; then
    FILENAME_PARTS="${FILENAME_PARTS}_${FILENAME_SUFFIX}"
fi
FILENAME_PARTS="${FILENAME_PARTS}_${TIMESTAMP}"

OUTPUT_FILE="${OUTPUT_DIR}/longbench_v2_${FILENAME_PARTS}.jsonl"
LOG_FILE="${OUTPUT_DIR}/longbench_v2_${FILENAME_PARTS}.log"

echo "=========================================="
echo "  LongBench-v2"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Split: ${SPLIT_NAME}"
echo "Samples: ${NUM_SAMPLES}"
echo "Attention: ${ATTENTION_TYPE}"
if [ "${ENABLE_COT}" = "1" ]; then
    echo "CoT: stage1 max=${COT_MAX_NEW_TOKENS}, stage2 max=${ANSWER_MAX_NEW_TOKENS}"
else
    echo "Max new tokens: ${MAX_NEW_TOKENS}"
fi
if [ -n "${SINK_SIZE}" ]; then
    echo "Sink size override: ${SINK_SIZE}"
fi
if [ -n "${LOCAL_SIZE}" ]; then
    echo "Local size override: ${LOCAL_SIZE}"
fi
if [ -n "${DOMAIN}" ]; then
    echo "Domain filter: ${DOMAIN}"
fi
if [ -n "${DIFFICULTY}" ]; then
    echo "Difficulty filter: ${DIFFICULTY}"
fi
if [ -n "${LENGTH}" ]; then
    echo "Length filter: ${LENGTH}"
fi
if [ "${ATTENTION_TYPE}" = "PolarANN" ]; then
    echo "Polar cache module: ${POLAR_CACHE_MODULE}"
fi
if [ -z "${DIFFICULTY}" ] && [ -z "${LENGTH}" ]; then
    echo "Mode: full eval (no difficulty/length split)"
fi
echo "Output: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

CMD="python test_longbench_v2.py"
CMD="${CMD} --model_name ${MODEL_NAME}"
CMD="${CMD} --dataset_name ${DATASET_NAME}"
CMD="${CMD} --split_name ${SPLIT_NAME}"
CMD="${CMD} --num_samples ${NUM_SAMPLES}"
CMD="${CMD} --attention_type ${ATTENTION_TYPE}"
if [ "${ATTENTION_TYPE}" = "PolarANN" ]; then
    CMD="${CMD} --polar_cache_module ${POLAR_CACHE_MODULE}"
fi

if [ -n "${ENABLE_RECALL}" ]; then
    CMD="${CMD} --enable_recall ${ENABLE_RECALL}"
fi
if [ -n "${STATIC_PATTERN_END}" ]; then
    CMD="${CMD} --static_pattern_end ${STATIC_PATTERN_END}"
fi

CMD="${CMD} --temperature ${TEMPERATURE}"
CMD="${CMD} --top_p ${TOP_P}"
CMD="${CMD} --top_k ${TOP_K}"
if [ "${ENABLE_COT}" != "1" ]; then
    CMD="${CMD} --max_new_tokens ${MAX_NEW_TOKENS}"
fi
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --dtype ${DTYPE}"
CMD="${CMD} --output_file ${OUTPUT_FILE}"

if [ -n "${SINK_SIZE}" ]; then
    CMD="${CMD} --sink_size ${SINK_SIZE}"
fi
if [ -n "${LOCAL_SIZE}" ]; then
    CMD="${CMD} --local_size ${LOCAL_SIZE}"
fi
if [ -n "${SAMPLE_OFFSET}" ]; then
    CMD="${CMD} --sample_offset ${SAMPLE_OFFSET}"
fi

if [ "${FORCE_PROCESS}" = "true" ]; then
    CMD="${CMD} --force_process"
fi

if [ -n "${DOMAIN}" ]; then
    CMD="${CMD} --domain ${DOMAIN}"
fi
if [ -n "${DIFFICULTY}" ]; then
    CMD="${CMD} --difficulty ${DIFFICULTY}"
fi
if [ -n "${LENGTH}" ]; then
    CMD="${CMD} --length ${LENGTH}"
fi
if [ -n "${MAX_CONTEXT_TOKENS:-}" ]; then
    CMD="${CMD} --max_context_tokens ${MAX_CONTEXT_TOKENS}"
fi
if [ -n "${FINAL_TOPK}" ]; then
    CMD="${CMD} --final_topk ${FINAL_TOPK}"
fi
if [ -n "${ENABLE_OFFLOAD}" ]; then
    CMD="${CMD} --enable_offload ${ENABLE_OFFLOAD}"
fi
if [ -n "${NO_RESUME:-}" ]; then
    CMD="${CMD} --no_resume"
fi
if [ "${ENABLE_COT}" = "1" ]; then
    CMD="${CMD} --enable_cot --cot_max_new_tokens ${COT_MAX_NEW_TOKENS} --answer_max_new_tokens ${ANSWER_MAX_NEW_TOKENS}"
fi

echo ""
echo "Command:"
echo "${CMD}"
echo ""

${CMD} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Results: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo ""

if [ -f "${OUTPUT_FILE}" ]; then
    TOTAL=$(wc -l < "${OUTPUT_FILE}")
    if [ "${TOTAL}" -gt 0 ]; then
        CORRECT=$(grep -o '"is_correct": true' "${OUTPUT_FILE}" | wc -l)
        ACCURACY=$(echo "scale=2; ${CORRECT} * 100 / ${TOTAL}" | bc)
        echo "Total: ${TOTAL}"
        echo "Correct: ${CORRECT}"
        echo "Accuracy: ${ACCURACY}%"
    fi
fi
