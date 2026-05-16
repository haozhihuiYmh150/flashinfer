"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer


def _pytorch_top_k_top_p_filtered_probs(probs, top_k, top_p):
    """Reference: compute filtered probs using PyTorch sorting (joint semantics).

    Top-k and top-p are applied independently on the full distribution,
    then the final mask is their intersection.
    """
    batch_size, vocab_size = probs.shape
    k = top_k if isinstance(top_k, int) else None
    p = top_p if isinstance(top_p, float) else None

    filtered = torch.zeros_like(probs)
    for i in range(batch_size):
        row_k = k if k is not None else top_k[i].item()
        row_p = p if p is not None else top_p[i].item()
        row = probs[i]

        # Top-k mask: keep the k largest probabilities
        topk_vals, topk_idx = row.topk(min(row_k, vocab_size))
        mask_topk = torch.zeros(vocab_size, dtype=torch.bool)
        mask_topk[topk_idx] = True

        # Top-p mask: from globally sorted (descending) probabilities,
        # keep the smallest prefix with cumulative sum >= row_p
        sorted_vals, sorted_idx = row.sort(descending=True)
        cumsum = sorted_vals.cumsum(0)
        # Find the first position where cumsum >= row_p, keep prefix [0..pos]
        cutoff_pos = (cumsum >= row_p).int().argmax().item()
        # If no position satisfies cumsum >= row_p, keep all
        if not (cumsum >= row_p).any():
            cutoff_pos = vocab_size - 1
        mask_topk_p = torch.zeros(vocab_size, dtype=torch.bool)
        mask_topk_p[sorted_idx[:cutoff_pos + 1]] = True

        # Joint: intersection of top-k and top-p masks
        mask = mask_topk & mask_topk_p
        filtered[i, mask] = row[mask]

    return filtered


@pytest.mark.parametrize("batch_size", [1, 8, 32, 99])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize("p", [0.5, 0.9])
def test_sampled_token_nonzero_and_filtered_accuracy(batch_size, vocab_size, k, p):
    """Two core correctness checks:
    1. Sampled token always has non-zero filtered probability
    2. Filtered probs overlap with PyTorch reference:
       - Same nonzero count => identical filtered probs
       - Different count => one is a subset of the other
    """
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)

    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    samples, filtered_probs = flashinfer.sampling.top_k_top_p_sampling_and_filter(
        normalized_prob, top_k=k, top_p=p
    )

    # Check 1: sampled token must have non-zero filtered prob
    batch_idx = torch.arange(batch_size, device="cuda:0")
    assert torch.all(filtered_probs[batch_idx, samples] > 0), (
        "Sampled token has zero filtered probability"
    )

    # Check 2: overlap with PyTorch reference
    ref_probs = _pytorch_top_k_top_p_filtered_probs(normalized_prob, k, p).to("cuda:0")

    for i in range(batch_size):
        fi_mask = filtered_probs[i] > 0
        ref_mask = ref_probs[i] > 0
        fi_count = fi_mask.sum().item()
        ref_count = ref_mask.sum().item()

        if fi_count == ref_count:
            # Same count: masks should be identical, values should match
            assert torch.all(fi_mask == ref_mask), (
                f"Row {i}: same nonzero count ({fi_count}) but different positions"
            )
            assert torch.allclose(filtered_probs[i][fi_mask], ref_probs[i][ref_mask], atol=1e-6), (
                f"Row {i}: same positions but different values"
            )
        else:
            # Different count: one should be a subset of the other
            if fi_count < ref_count:
                # fi is subset of ref
                assert torch.all(fi_mask <= ref_mask), (
                    f"Row {i}: fi ({fi_count} nonzero) is not a subset of ref ({ref_count} nonzero)"
                )
                # Overlapping positions should have same values
                assert torch.allclose(filtered_probs[i][fi_mask], ref_probs[i][fi_mask], atol=1e-6), (
                    f"Row {i}: overlapping positions have different values"
                )
            else:
                # ref is subset of fi
                assert torch.all(ref_mask <= fi_mask), (
                    f"Row {i}: ref ({ref_count} nonzero) is not a subset of fi ({fi_count} nonzero)"
                )
                assert torch.allclose(filtered_probs[i][ref_mask], ref_probs[i][ref_mask], atol=1e-6), (
                    f"Row {i}: overlapping positions have different values"
                )
