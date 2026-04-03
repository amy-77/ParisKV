#!/bin/bash
#
# AIME 2025 evaluation with pass@k (PolarANN / Full_Flash_Attn).
#
# Prerequisites: activate your Python/conda environment before running.
#
# Required environment variable:
#   DATA_PATH  - Path to AIME2025 json (e.g. AIME2025_all.json)
#
# Usage:
#   ./run_aime2025.sh [ATTENTION_TYPE] [PASS_K] [TEMPERATURE] [BUDGET_RATIO] [CUDA_DEVICE] [MODEL_NAME]
#
# Positional arguments:
#   1  ATTENTION_TYPE   PolarANN | Full_Flash_Attn                           [default: Full_Flash_Attn]
#   2  PASS_K           Number of samples per problem for pass@k (1=greedy)  [default: 1]
#   3  TEMPERATURE      Sampling temperature (0.0=greedy, 0.6-0.8 for pass@k)[default: 0.0]
#   4  BUDGET_RATIO     KV budget ratio (display only, not used internally)  [default: 0.2]
#   5  CUDA_DEVICE      GPU device index (sets CUDA_VISIBLE_DEVICES)         [default: 0]
#   6  MODEL_NAME       HuggingFace model name or local path                 [default: Qwen/Qwen3-8B]
#
# Examples:
#   export DATA_PATH="<DATA_PATH>"
#   ./run_aime2025.sh Full_Flash_Attn 1 0.0 0 0                  # full attention, pass@1 (greedy)
#   ./run_aime2025.sh Full_Flash_Attn 8 0.7 0 0                  # full attention, pass@8
#   ./run_aime2025.sh PolarANN 1 0.0 0.2 0                       # PolarANN, pass@1, 20% budget
#   ./run_aime2025.sh PolarANN 8 0.7 0.2 0                       # PolarANN, pass@8

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to your AIME2025 dataset json file.}"

ATTENTION_TYPE=${1:-Full_Flash_Attn}
PASS_K=${2:-1}
TEMPERATURE=${3:-0.0}
BUDGET_RATIO=${4:-0.2}
CUDA_DEVICE=${5:-0}
MODEL_NAME=${6:-"Qwen/Qwen3-8B"}

if (( $(echo "$TEMPERATURE == 0.0" | bc -l) )); then
    MODE="Greedy"
    TOP_P=0.0
    TOP_K=0
else
    MODE="Sampling"
    TOP_P=0.95
    TOP_K=20
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_ROOT/results"
mkdir -p "$OUT_DIR"

OUTPUT_FILE="aime2025_${ATTENTION_TYPE}_pass@${PASS_K}_${MODE}_${TIMESTAMP}.jsonl"
LOG_FILE="aime2025_${ATTENTION_TYPE}_pass@${PASS_K}_${MODE}_${TIMESTAMP}.log"

echo "=========================================="
echo "AIME2025 — pass@k"
echo "=========================================="
echo "Model:        $MODEL_NAME"
echo "Dataset:      AIME2025 ($DATA_PATH)"
echo "CUDA device:  GPU $CUDA_DEVICE"
echo "=========================================="
echo "Mode:         pass@${PASS_K}"
echo "  Samples/Q:  ${PASS_K}"
echo "  Strategy:   ${MODE}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-p:       ${TOP_P}"
echo "  Top-k:       ${TOP_K}"
echo "=========================================="
echo "Attention:    $ATTENTION_TYPE"
if [ "$ATTENTION_TYPE" != "Full_Flash_Attn" ]; then
    echo "  Budget:      $BUDGET_RATIO ($(echo "$BUDGET_RATIO * 100" | bc)%)"
fi
echo "=========================================="
echo "Output:"
echo "  Results: $OUT_DIR/$OUTPUT_FILE"
echo "  Log:     $OUT_DIR/$LOG_FILE"
echo "=========================================="

cd "$SCRIPT_DIR"

PYTHON_CMD="python test_aime2025.py \
    --model_name \"$MODEL_NAME\" \
    --data_path \"$DATA_PATH\" \
    --num_samples -1 \
    --max_new_tokens 4096 \
    --attention_type $ATTENTION_TYPE \
    --budget_ratio $BUDGET_RATIO \
    --n_generate $PASS_K \
    --pass_k $PASS_K \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --dtype bf16 \
    --device cuda:0 \
    --output_file \"$OUTPUT_FILE\""

eval $PYTHON_CMD 2>&1 | tee "$OUT_DIR/$LOG_FILE"

echo ""
echo "=========================================="
echo "Finished"
echo "=========================================="
echo "Results: $OUT_DIR/$OUTPUT_FILE"
echo "Log:     $OUT_DIR/$LOG_FILE"
echo "=========================================="

# More examples:
#   ./run_aime2025.sh PolarANN 8 0.7 0.2 1 "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#
# nohup env DATA_PATH="<DATA_PATH>" ./run_aime2025.sh Full_Flash_Attn 1 0.0 0 0 > run_aime2025_full_pass1.log 2>&1 &
# nohup env DATA_PATH="<DATA_PATH>" ./run_aime2025.sh PolarANN 1 0.0 0.2 0 > run_aime2025_polar_pass1.log 2>&1 &
