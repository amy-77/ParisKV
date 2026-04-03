#!/usr/bin/env python3
"""
Lloyd-Max magnitude codebook generator for RSQ-IP 4-bit quantization.

Our 4-bit RSQ-IP format encodes each coordinate as 1 sign bit + 3-bit magnitude
index into an 8-level codebook.  This is a scalar quantization scheme -- NOT a
floating-point representation (neither e2m1 nor e3m0).

Theoretical motivation
----------------------
After L2-normalization and SRHT rotation, each key vector k is partitioned into
m-dimensional blocks.  The block direction u = k_b / ||k_b|| lives uniformly on
the unit sphere S^{m-1}.  By a classical result, each squared coordinate
satisfies:

    u_j^2  ~  Beta(1/2, (m - 1) / 2)

so the coordinate magnitude |u_j| follows a non-uniform distribution
concentrated near zero.  Critically this distribution depends *only* on the
block dimension m -- it is independent of the model, layer, head, or data.

Sampling procedure
------------------
We draw g ~ N(0, I_m) and normalize u = g / ||g||_2.  By the rotational
invariance of the Gaussian distribution this yields samples uniformly
distributed on S^{m-1}.  We collect |u_0| (any coordinate suffices by symmetry)
as 10M samples from the target distribution.

Codebook construction
---------------------
Given the samples, we apply Lloyd-Max optimal scalar quantization, which
iteratively partitions [0, 1] into L bins (L = 8 for 3-bit magnitude) and
assigns a reconstruction center to each bin, minimizing E[(X - Q(X))^2] where
X = |u_j|.  The resulting 7 decision thresholds and 8 reconstruction centers
are the magnitude codebook.

At encoding time, each coordinate magnitude |u_j| is assigned to one of the 8
bins via torch.bucketize (O(log L) per coordinate).  At decoding time, the
3-bit index is used to look up the reconstruction center.

Usage:
    python run/generate_magnitude_levels.py --levels 8              # default (m=8, 10M samples)
    python run/generate_magnitude_levels.py --levels 8 --m 16       # for m=16 blocks
    python run/generate_magnitude_levels.py --levels 4 --n_samples 1000000
"""

import argparse
import json
import os

import numpy as np


def sample_coordinate_magnitudes(n_samples: int, m: int, seed: int = 42) -> np.ndarray:
    """Sample |(u)_j| for u uniform on S^{m-1} (marginal suitable for per-coordinate magnitude bins)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_samples, m))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    magnitudes = np.abs(u[:, 0])
    return magnitudes


def sample_block_radius(n_samples: int, m: int, d: int = 128, seed: int = 42) -> np.ndarray:
    """Optional: sample block radius ||k̃_b|| for d-dim unit k̃ (not the RSQ-IP per-coordinate target)."""
    rng = np.random.default_rng(seed)
    x_full = rng.standard_normal((n_samples, d))
    x_full_normalized = x_full / np.linalg.norm(x_full, axis=1, keepdims=True)
    radii = np.linalg.norm(x_full_normalized[:, :m], axis=1)
    return radii


def lloyd_max_quantization(
    data: np.ndarray, n_levels: int, max_iter: int = 100, tol: float = 1e-9
) -> tuple:
    """Compute MSE-optimal scalar quantizer via Lloyd-Max iteration.

    Returns (centers, thresholds) where thresholds has n_levels-1 entries
    (decision boundaries) and centers has n_levels entries (reconstruction values).
    """
    data = np.sort(data)
    n_levels = int(n_levels)
    thresholds = np.zeros(n_levels + 1)
    thresholds[0] = data.min() - 1e-6
    thresholds[-1] = data.max() + 1e-6
    for i in range(1, n_levels):
        thresholds[i] = np.percentile(data, 100 * i / n_levels)

    for iter_num in range(max_iter):
        centers = np.zeros(n_levels)
        for i in range(n_levels):
            mask = (data >= thresholds[i]) & (data < thresholds[i + 1])
            if mask.sum() > 0:
                centers[i] = data[mask].mean()
            else:
                centers[i] = (thresholds[i] + thresholds[i + 1]) / 2.0

        new_thresholds = thresholds.copy()
        for i in range(1, n_levels):
            new_thresholds[i] = (centers[i - 1] + centers[i]) / 2.0

        if np.max(np.abs(new_thresholds - thresholds)) < tol:
            print(f"  converged at iteration {iter_num + 1}")
            break
        thresholds = new_thresholds

    return centers, thresholds[1:-1]


def evaluate_quantization(data: np.ndarray, centers: np.ndarray, thresholds: np.ndarray) -> dict:
    full_thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])
    quantized = np.zeros_like(data)
    for i in range(len(centers)):
        mask = (data >= full_thresholds[i]) & (data < full_thresholds[i + 1])
        quantized[mask] = centers[i]

    mse = np.mean((data - quantized) ** 2)
    mae = np.mean(np.abs(data - quantized))

    return {"mse": float(mse), "mae": float(mae)}


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_out = os.path.join(os.path.dirname(_here), "codebooks")

    parser = argparse.ArgumentParser(description="Generate Lloyd–Max magnitude codebook")
    parser.add_argument("--levels", type=int, default=8, help="Number of levels (default 8 for 3-bit mag)")
    parser.add_argument("--m", type=int, default=8, help="Block dimension m")
    parser.add_argument("--n_samples", type=int, default=10000000, help="Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_out,
        help="Output directory (default: <repo>/codebooks)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Magnitude codebook (Lloyd–Max)")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  levels = {args.levels}")
    print(f"  m = {args.m}")
    print(f"  n_samples = {args.n_samples:,}")
    print(f"  seed = {args.seed}")

    bits_per_dim = int(np.ceil(np.log2(args.levels)))
    print(f"  bits_per_dim = {bits_per_dim}")

    print(f"\n[1/3] Sampling |(u)_j| on S^{{{args.m}-1}} ...")
    print(f"  marginal: (u_j)^2 ~ Beta(1/2, (m-1)/2) = Beta(0.5, {(args.m - 1) / 2})")
    magnitudes = sample_coordinate_magnitudes(args.n_samples, args.m, args.seed)

    print(f"  range: [{magnitudes.min():.6f}, {magnitudes.max():.6f}]")
    print(f"  mean: {magnitudes.mean():.6f}, std: {magnitudes.std():.6f}")

    print(f"\n[2/3] Lloyd–Max ...")
    centers, thresholds = lloyd_max_quantization(magnitudes, args.levels)

    print(f"\n  thresholds: {thresholds}")
    print(f"  centers: {centers}")

    print(f"\n[3/3] Evaluating quantization error ...")
    metrics = evaluate_quantization(magnitudes, centers, thresholds)
    print(f"  MSE: {metrics['mse']:.10f}")
    print(f"  MAE: {metrics['mae']:.10f}")

    os.makedirs(args.output_dir, exist_ok=True)

    result = {
        "meta": {
            "m": args.m,
            "n_samples": args.n_samples,
            "levels": args.levels,
            "bits_per_dim": bits_per_dim,
            "seed": args.seed,
        },
        "thresholds": thresholds.tolist(),
        "centers": centers.tolist(),
        "metrics": metrics,
        "data_stats": {
            "min": float(magnitudes.min()),
            "max": float(magnitudes.max()),
            "mean": float(magnitudes.mean()),
            "std": float(magnitudes.std()),
        },
    }

    if args.levels == 8:
        output_path = os.path.join(args.output_dir, f"magnitude_levels_m{args.m}_4bit.json")
    elif args.levels == 4:
        output_path = os.path.join(args.output_dir, f"magnitude_levels_m{args.m}.json")
    else:
        output_path = os.path.join(args.output_dir, f"magnitude_levels_m{args.m}_L{args.levels}.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote: {output_path}")

    print("\n" + "=" * 80)
    print("Snippet for polar_cache.py:")
    print("=" * 80)
    print(
        f"""
self._mag_thresholds = torch.tensor(
    {thresholds.tolist()},
    device=self.device, dtype=torch.bfloat16).contiguous()

self._mag_centers = torch.tensor(
    {centers.tolist()},
    device=self.device, dtype=torch.bfloat16).contiguous()
"""
    )


if __name__ == "__main__":
    main()
