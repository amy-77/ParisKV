# ParisKV

**Polar ANN-based KV Cache with RaBitQ Reranking for Long-Context LLM Inference**

ParisKV accelerates the decoding phase of long-context LLM inference by replacing full attention over the entire KV cache with an efficient approximate nearest neighbor (ANN) retrieval pipeline. It uses data-independent polar coordinate quantization and multi-tier collision-based retrieval, eliminating the need for expensive per-layer K-means clustering at prefill time.

## Key Features

- **Data-independent quantization**: Uses SRHT (Subsampled Randomized Hadamard Transform) + sign-based product quantization. No K-means clustering needed -- the codebook is pre-generated and fixed.
- **Multi-tier collision retrieval**: Weighted collision counting across subspaces for coarse candidate selection, achieving high recall with low latency.
- **4-bit RaBitQ reranking**: Fine-grained reranking of collision candidates using a codebook-based scalar quantization (1-bit sign + 3-bit Lloyd–Max magnitude index). The codebook is derived from the theoretical distribution of coordinate magnitudes on the unit sphere — fully data-independent.
- **CPU offloading**: Optionally offloads retrieval-zone KV pairs to CPU pinned memory, significantly reducing GPU memory consumption for very long contexts.
- **Custom CUDA kernels**: Fused collision counting, radix top-k, UVA H2D gather, and reranking kernels for minimal decode latency.
- **Multi-batch support**: Efficient lockstep multi-batch prefill and decode.

## Architecture Overview

During **prefill**, ParisKV:
1. Stores KV pairs in a unified cache (GPU or GPU+CPU with offloading)
2. Encodes keys via SRHT rotation + sign-based product quantization into compact codebooks
3. Computes 4-bit RaBitQ auxiliary data (block weights + packed codes) for reranking

During **decode**, for each new token:
1. **Coarse retrieval**: Computes query-cluster scores via sign inner product, then counts multi-tier collisions across subspaces using a fused CUDA kernel
2. **Candidate selection**: Selects top candidates by collision count using radix top-k
3. **RaBitQ reranking**: Scores candidates with 4-bit approximate inner product and selects final top-K
4. **Attention**: Computes Flash Attention over Sink tokens + Local window + Retrieved top-K tokens

## Project Structure

```
ParisKV/
├── cache_hub/                  # KV cache implementations
│   ├── polar_cache.py          # Core: ParisKV cache (polar ANN + RaBitQ rerank)
│   ├── cache.py                # Base KV_Cache class
│   ├── collision/              # CUDA kernel: collision counting
│   ├── collision_fused/        # CUDA kernel: fused collision (score + count)
│   ├── gather/                 # C++ extension: CPU gather
│   ├── gather_trans/           # CUDA kernel: UVA H2D gather (CPU→GPU)
│   ├── rerank/                 # CUDA kernel: 4-bit RaBitQ reranking
│   └── topk/                   # CUDA kernel: radix top-k selection
├── attn_hub/                   # Attention function wrappers
│   ├── polar_attn.py           # Prefill/decode attention for ParisKV
│   ├── flash_attn.py           # Full flash attention baseline
│   └── flash_attn_compat.py    # Flash attention compatibility layer
├── model_hub/                  # Model integration
│   ├── LLM.py                  # Base LLM class (tokenization, generation)
│   └── qwen.py                 # Qwen model integration with ParisKV
├── config/                     # Model-specific configurations
├── codebooks/                  # Pre-generated Lloyd-Max magnitude codebooks (4-bit RSQ-IP)
├── turboquant/codebooks/       # Pre-generated sign codebooks for various (d, m, K) settings
├── run/                        # Evaluation & codebook generation scripts
│   ├── test_longbench_v2.py          # LongBench-v2 evaluation
│   ├── test_aime2025.py              # AIME 2025 evaluation (pass@k)
│   └── generate_magnitude_levels.py  # Lloyd-Max codebook generator (see §4-bit RSQ-IP)
├── results/                    # Evaluation output (auto-created)
├── tests/                      # Test scripts
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- **CUDA 12.4** (or use Docker image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`)
- **Python 3.10+**

### Environment Setup

```bash
conda create -n pariskv python=3.10 -y
conda activate pariskv

conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y

pip install -r requirements.txt
pip install flash-attn>=2.7.0 --no-build-isolation
pip install flashinfer-python>=0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/
```

### Quick Start

```python
from model_hub.qwen import QwenModel

model = QwenModel(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_length=131072,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize ParisKV cache
model.init_kv_cache(
    attention_type="PolarANN",
    max_new_length=2048,
    budget_ratio=0.1,
)

# Run inference
output = model.generate(input_text, max_new_tokens=2048)
```

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `final_topk` | Number of keys selected after RaBitQ reranking | 50 -- 200 |
| `local_size` | Size of the local attention window (always attended) | 256 -- 512 |
| `sink_size` | Number of initial sink tokens (always attended) | 4 |
| `enable_offload` | Offload retrieval-zone KV to CPU pinned memory | True / False |
| `dynamic_update_interval` | Tokens accumulated before triggering index update | 256 -- 512 |

### Adaptive Collision and Candidate Ratios

ParisKV automatically adjusts the collision ratio (fraction of clusters activated per subspace) and candidate ratio (fraction of retrieval-zone keys kept as coarse candidates) based on the retrieval zone length. This removes the need to manually tune these parameters:

| Retrieval zone length | Candidate ratio | Collision ratio |
|-----------------------|-----------------|-----------------|
| < 5,000 | 0.50 | 0.50 |
| 5,000 -- 10,000 | 0.25 | 0.30 |
| 10,000 -- 30,000 | 0.20 | 0.25 |
| >= 30,000 | 0.10 | 0.20 |

Shorter sequences use higher ratios to ensure enough candidates for accurate attention; longer sequences can afford more aggressive pruning because the absolute number of retained keys remains large. These ratios are applied at every decode step in `polar_cache._get_adaptive_ratios()`, so no user configuration is required.

### Codebook Selection

ParisKV uses pre-generated sign codebooks located in `turboquant/codebooks/`. The codebook filename encodes its parameters:

```
codebook_d{dim}_m{block_size}_Kr{radius_levels}_Kw{angular_levels}_{type}.json
```

For example, `codebook_d128_m8_Kr1_Kw256_rabitq_sign.json` means:
- `d=128`: head dimension
- `m=8`: block size (128/8 = 16 subspaces)
- `Kr=1`: single radius level (sign-only)
- `Kw=256`: 256 angular clusters per subspace (2^8 sign patterns for m=8)

### 4-bit RSQ-IP Magnitude Quantization

ParisKV uses a **4-bit codebook-based scalar quantization** scheme for reranking, encoding each coordinate as **1 sign bit + 3-bit magnitude index** into an 8-level codebook. This is *not* a floating-point format (neither e2m1 nor e3m0).

**Theoretical basis.**
After L2-normalization and SRHT rotation, each key vector is partitioned into *m*-dimensional blocks. The block direction **u** = **k**\_b / ‖**k**\_b‖ is uniformly distributed on the unit sphere S<sup>m−1</sup>. By a classical result, each squared coordinate satisfies:

> u\_j² ~ Beta(1/2, (m−1)/2)

Therefore the coordinate magnitude |u\_j| follows a non-uniform distribution concentrated near zero. This distribution depends **only on the block dimension** *m* — it is independent of the model, layer, head, or data.

**Codebook construction.**
We sample **g** ~ N(0, I\_m), normalize **u** = **g** / ‖**g**‖₂ (uniform on S<sup>m−1</sup> by rotational invariance of the Gaussian), and collect |u₀| as 10M samples from the target distribution. We then apply **Lloyd–Max optimal scalar quantization**, which iteratively partitions the range into 8 bins and assigns a reconstruction center to each bin, minimizing the mean squared error E[(X − Q(X))²]. The resulting codebook for *m* = 8:

| | Values |
|---|--------|
| **Decision thresholds** (7) | 0.084, 0.170, 0.258, 0.350, 0.449, 0.559, 0.690 |
| **Reconstruction centers** (8) | 0.042, 0.127, 0.213, 0.303, 0.397, 0.500, 0.617, 0.763 |

At encoding time, each coordinate magnitude |u\_j| is assigned to one of the 8 bins via `torch.bucketize`. At decoding time, the 3-bit index looks up the reconstruction center.

The codebook generation script is provided at [`run/generate_magnitude_levels.py`](run/generate_magnitude_levels.py):

```bash
# Reproduce the default m=8 codebook (8 levels = 3 magnitude bits)
python run/generate_magnitude_levels.py --levels 8 --m 8 --n_samples 10000000


Pre-generated codebooks are stored in `codebooks/` (e.g., `magnitude_levels_m8_4bit.json`).

## Evaluation

Evaluation scripts are in `run/`. All results are saved to the `results/` directory. Run from the project root or `run/` directory.

### CLI Parameters

Common parameters shared across evaluation scripts:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | HuggingFace model name or local path | `Qwen/Qwen3-8B` |
| `--attention_type` | Attention backend: `PolarANN` or `Full_Flash_Attn` | `Full_Flash_Attn` |
| `--max_new_tokens` | Maximum new tokens to generate per sample | 2048 |
| `--device` | CUDA device (e.g. `cuda:0`) | `cuda:0` |
| `--dtype` | Model precision: `fp16` or `bf16` | `bf16` |
| `--num_samples` | Number of samples to evaluate (`-1` = all) | 10 / -1 |
| `--output_file` | Output file path (auto-generated if omitted) | — |

PolarANN-specific parameters (used only when `--attention_type PolarANN`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--final_topk` | Number of keys kept after RaBitQ reranking | from config |
| `--enable_offload` | Offload retrieval-zone KV to CPU pinned memory (`0`=off, `1`=on) | from config |
| `--static_pattern_end` | Local-window size override (same as `local_size`) | from config |
| `--sink_size` | Sink-zone size override (initial tokens always attended) | from config |
| `--local_size` | Local-zone size override (recent tokens always attended) | from config |

LongBench-v2 specific parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_cot` | Enable two-stage Chain-of-Thought reasoning | off |
| `--cot_max_new_tokens` | CoT stage 1 (reasoning) max new tokens | 1024 |
| `--answer_max_new_tokens` | CoT stage 2 (final answer) max new tokens | 128 |
| `--enable_recall` | Compute Recall@K metric (`true`/`false`) | from config |
| `--difficulty` | Filter by difficulty: `easy` or `hard` | all |
| `--length` | Filter by length: `short`, `medium`, or `long` | all |
| `--domain` | Filter by domain (e.g. `single-document_qa`) | all |
| `--force_process` | Force-process samples even if they exceed context window | off |
| `--no_resume` | Start fresh instead of resuming from existing output | off |

AIME 2025 specific parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_generate` | Answers to generate per problem (`1` = pass@1, `8` = pass@8) | 1 |
| `--pass_k` | Evaluate pass@k (defaults to `n_generate`) | same as `n_generate` |
| `--temperature` | Sampling temperature (`0.0` = greedy for pass@1, `0.6`–`0.8` for pass@k) | 0.0 |
| `--data_path` | Path to AIME2025 dataset JSON file (required) | — |
| `--budget_ratio` | KV budget ratio (display only) | 0.2 |

### LongBench-v2

```bash
cd run

# PolarANN — 1 sample quick test (easy/short)
python test_longbench_v2.py \
    --model_name <MODEL_PATH> \
    --attention_type PolarANN \
    --num_samples 1 \
    --difficulty easy \
    --length short \
    --max_new_tokens 128 \
    --force_process \
    --output_file ../results/longbench_v2/test_1sample.jsonl


# Full attention baseline
python test_longbench_v2.py \
    --model_name <MODEL_PATH> \
    --attention_type Full_Flash_Attn \
    --num_samples -1
```


### AIME 2025

```bash
cd run

# PolarANN — pass@8 (sampling)
python test_aime2025.py \
    --model_name <MODEL_PATH> \
    --data_path <AIME2025_JSON> \
    --attention_type PolarANN \
    --n_generate 8 \
    --pass_k 8 \
    --temperature 0.6


