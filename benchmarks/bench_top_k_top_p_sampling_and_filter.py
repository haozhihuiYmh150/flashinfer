"""Benchmark top_k_top_p_sampling_and_filter vs baseline approaches.

Compares three approaches side by side and prints a summary table:
1. fused: top_k_top_p_sampling_and_filter (two-kernel: sampling + filter)
2. original: top_k_top_p_sampling_from_probs (original sampling only)
3. pytorch: PyTorch sort-based reference (baseline)
"""

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


def init_seed_top_k_top_p_sampling_and_filter(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.top_k_top_p_sampling_and_filter(*args, **kwargs)


def init_seed_top_k_top_p_sampling_from_probs(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(*args, **kwargs)


def pytorch_top_k_top_p_filter(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    filtered = torch.zeros_like(probs)
    for i in range(batch_size):
        row = probs[i]
        topk_vals, topk_idx = row.topk(min(top_k, vocab_size))
        mask_topk = torch.zeros(vocab_size, dtype=torch.bool, device=probs.device)
        mask_topk[topk_idx] = True
        sorted_vals, sorted_idx = row.sort(descending=True)
        cumsum = sorted_vals.cumsum(0)
        cutoff_pos = (cumsum >= top_p).int().argmax().item()
        if not (cumsum >= top_p).any():
            cutoff_pos = vocab_size - 1
        mask_topk_p = torch.zeros(vocab_size, dtype=torch.bool, device=probs.device)
        mask_topk_p[sorted_idx[:cutoff_pos + 1]] = True
        mask = mask_topk & mask_topk_p
        filtered[i, mask] = row[mask]
    return filtered


@torch.inference_mode()
def main():
    vocab_size = 128512
    batch_sizes = [1, 16, 32, 64, 128, 256, 512]
    k_values = [10, 100, 1000, 5000]
    p_values = [0.1, 0.5, 0.9]
    distrib = normal_distribution(1)
    deterministic = True

    results = []

    for batch_size in batch_sizes:
        for k in k_values:
            for p in p_values:
                logits = distrib((batch_size, vocab_size), device="cuda")
                probs = torch.softmax(logits, dim=-1)

                # fused: sampling + filter
                ms_fused = np.median(bench_gpu_time(
                    lambda: init_seed_top_k_top_p_sampling_and_filter(
                        probs, top_k=k, top_p=p, deterministic=deterministic,
                    ),
                    dry_run_time_ms=100, repeat_time_ms=1000,
                ))

                # original: sampling only
                ms_original = np.median(bench_gpu_time(
                    lambda: init_seed_top_k_top_p_sampling_from_probs(
                        probs, top_k=k, top_p=p,
                        filter_apply_order="joint", deterministic=deterministic,
                    ),
                    dry_run_time_ms=100, repeat_time_ms=1000,
                ))

                # pytorch: sort-based baseline
                ms_pytorch = np.median(bench_gpu_time(
                    lambda: pytorch_top_k_top_p_filter(probs, k, p),
                    dry_run_time_ms=100, repeat_time_ms=1000,
                ))

                overhead_vs_original = ms_fused / ms_original
                speedup_vs_pytorch = ms_pytorch / ms_fused

                results.append({
                    "batch": batch_size, "k": k, "p": p,
                    "fused_us": ms_fused * 1e3,
                    "original_us": ms_original * 1e3,
                    "pytorch_us": ms_pytorch * 1e3,
                    "overhead_vs_original": overhead_vs_original,
                    "speedup_vs_pytorch": speedup_vs_pytorch,
                })

                print(
                    f"batch={batch_size:>3}, k={k:>4}, p={p:.1f} | "
                    f"fused={ms_fused*1e3:>8.1f}us  original={ms_original*1e3:>8.1f}us  "
                    f"pytorch={ms_pytorch*1e3:>8.1f}us | "
                    f"fused/original={overhead_vs_original:.2f}x  "
                    f"fused/pytorch={speedup_vs_pytorch:.2f}x"
                )

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'batch':>5} {'k':>5} {'p':>5} | {'fused(us)':>10} {'original(us)':>12} {'pytorch(us)':>12} | {'fused/orig':>10} {'fused/pt':>10}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['batch']:>5} {r['k']:>5} {r['p']:>5.1f} | "
            f"{r['fused_us']:>10.1f} {r['original_us']:>12.1f} {r['pytorch_us']:>12.1f} | "
            f"{r['overhead_vs_original']:>10.2f}x {r['speedup_vs_pytorch']:>10.2f}x"
        )

    # Aggregate stats
    overheads = [r["overhead_vs_original"] for r in results]
    speedups = [r["speedup_vs_pytorch"] for r in results]
    print("-" * 100)
    print(
        f"fused/original overhead: median={np.median(overheads):.2f}x, "
        f"min={np.min(overheads):.2f}x, max={np.max(overheads):.2f}x"
    )
    print(
        f"fused/pytorch speedup:   median={np.median(speedups):.2f}x, "
        f"min={np.min(speedups):.2f}x, max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
