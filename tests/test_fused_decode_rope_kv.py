# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


class _Norm(torch.nn.Module):
    def __init__(
        self,
        weight,
        eps = 1e-6,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad = False)
        self.variance_epsilon = eps


def _ulps(a, b):
    bits = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}
    a_i = a.view(bits[a.dtype]).to(torch.int64)
    b_i = b.view(bits[b.dtype]).to(torch.int64)
    return (a_i - b_i).abs().max().item()


def _unfused_decode_rope_kv(
    Qn,
    Kn,
    Vn,
    cos,
    sin,
    position_ids,
    paged_K_slice,
    paged_V_slice,
    q_norm = None,
    k_norm = None,
):
    from unsloth.models.llama import fast_rms_layernorm_inference

    if q_norm is not None:
        Qn_v = fast_rms_layernorm_inference(q_norm, Qn.transpose(1, 2))
        Kn_v = fast_rms_layernorm_inference(k_norm, Kn.transpose(1, 2))
        Qn = Qn_v.transpose(1, 2)
        Kn = Kn_v.transpose(1, 2)

    cos_s = cos[position_ids].unsqueeze(1).to(device = Qn.device, dtype = Qn.dtype)
    sin_s = sin[position_ids].unsqueeze(1).to(device = Qn.device, dtype = Qn.dtype)
    h = Qn.shape[-1] // 2
    n_kv_heads = Kn.shape[1]

    RH_Q = torch.empty_like(Qn)
    RH_Q[:, :, :, :h] = Qn[:, :, :, h:]
    RH_Q[:, :, :, h:] = Qn[:, :, :, :h]
    RH_Q[:, :, :, :h].neg_()
    Qn *= cos_s
    Qn.addcmul_(RH_Q, sin_s)

    RH_K = RH_Q[:, :n_kv_heads, :, :]
    RH_K[:, :, :, :h] = Kn[:, :, :, h:]
    RH_K[:, :, :, h:] = Kn[:, :, :, :h]
    RH_K[:, :, :, :h].neg_()
    Kn *= cos_s
    Kn.addcmul_(RH_K, sin_s)

    paged_K_slice.copy_(Kn.permute(2, 0, 1, 3)[0])
    paged_V_slice.copy_(Vn.permute(2, 0, 1, 3)[0])
    return Qn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("heads", [(16, 4), (32, 8)])
@pytest.mark.parametrize("bsz", [1, 2, 4])
@torch.inference_mode()
def test_llama_decode_rope_kv_is_bit_exact(dtype, head_dim, heads, bsz):
    from unsloth.kernels.fused_decode_rope_kv import (
        fast_decode_rope_kv,
        fast_decode_rope_kv_ok,
    )

    n_q_heads, n_kv_heads = heads
    g = torch.Generator(device = "cuda").manual_seed(bsz * 100 + head_dim)
    Qn_ref = torch.randn(
        (bsz, 1, n_q_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    Kn_ref = torch.randn(
        (bsz, 1, n_kv_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    Vn_ref = torch.randn(
        (bsz, 1, n_kv_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    cos = torch.randn((512, head_dim), device = "cuda", generator = g, dtype = dtype)
    sin = torch.randn((512, head_dim), device = "cuda", generator = g, dtype = dtype)
    # Strided [bsz, 1] slice from [bsz, 17] as in Transformers 5.x decode steps.
    full_pos = torch.randint(0, 500, (bsz, 17), device = "cuda", generator = g, dtype = torch.int64)
    position_ids = full_pos[:, -1:]

    paged_ref = torch.zeros((64, 2, bsz, n_kv_heads, head_dim), device = "cuda", dtype = dtype)
    paged_new = paged_ref.clone()
    Qn_new = Qn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)
    Kn_new = Kn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)
    Vn_new = Vn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)

    assert fast_decode_rope_kv_ok(
        Qn_new, Kn_new, Vn_new, cos, sin, position_ids, paged_new[13, 0], paged_new[13, 1]
    )
    _unfused_decode_rope_kv(
        Qn_ref, Kn_ref, Vn_ref, cos, sin, position_ids, paged_ref[13, 0], paged_ref[13, 1]
    )
    fast_decode_rope_kv(
        Qn_new, Kn_new, Vn_new, cos, sin, position_ids, paged_new[13, 0], paged_new[13, 1]
    )

    assert torch.equal(Qn_new, Qn_ref)
    assert torch.equal(paged_new, paged_ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("bsz", [1, 4])
@torch.inference_mode()
def test_qwen3_decode_qk_norm_rope_kv_matches_unfused(dtype, head_dim, bsz):
    from unsloth.kernels.fused_decode_rope_kv import (
        fast_decode_rope_kv,
        fast_decode_rope_kv_ok,
    )

    n_q_heads, n_kv_heads = 16, 4
    g = torch.Generator(device = "cuda").manual_seed(bsz * 1000 + head_dim)
    Qn_ref = torch.randn(
        (bsz, 1, n_q_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    Kn_ref = torch.randn(
        (bsz, 1, n_kv_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    Vn_ref = torch.randn(
        (bsz, 1, n_kv_heads, head_dim), device = "cuda", generator = g, dtype = dtype
    ).transpose(1, 2)
    cos = torch.randn((256, head_dim), device = "cuda", generator = g, dtype = dtype)
    sin = torch.randn((256, head_dim), device = "cuda", generator = g, dtype = dtype)
    q_norm = _Norm((1 + 0.5 * torch.randn(head_dim, device = "cuda", generator = g)).to(dtype))
    k_norm = _Norm((1 + 0.5 * torch.randn(head_dim, device = "cuda", generator = g)).to(dtype))
    # Broadcast [1, 1] position_ids across bsz > 1.
    position_ids = torch.tensor([[42]], device = "cuda", dtype = torch.int64)

    paged_ref = torch.zeros((64, 2, bsz, n_kv_heads, head_dim), device = "cuda", dtype = dtype)
    paged_new = paged_ref.clone()
    Qn_new = Qn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)
    Kn_new = Kn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)
    Vn_new = Vn_ref.clone().transpose(1, 2).contiguous().transpose(1, 2)

    assert fast_decode_rope_kv_ok(
        Qn_new,
        Kn_new,
        Vn_new,
        cos,
        sin,
        position_ids,
        paged_new[7, 0],
        paged_new[7, 1],
        q_norm,
        k_norm,
    )
    _unfused_decode_rope_kv(
        Qn_ref,
        Kn_ref,
        Vn_ref,
        cos,
        sin,
        position_ids,
        paged_ref[7, 0],
        paged_ref[7, 1],
        q_norm,
        k_norm,
    )
    fast_decode_rope_kv(
        Qn_new,
        Kn_new,
        Vn_new,
        cos,
        sin,
        position_ids,
        paged_new[7, 0],
        paged_new[7, 1],
        q_norm,
        k_norm,
    )

    assert _ulps(Qn_new, Qn_ref) <= 2
    assert _ulps(paged_new[7, 0], paged_ref[7, 0]) <= 2
    assert torch.equal(paged_new[7, 1], paged_ref[7, 1])


@torch.inference_mode()
def test_ok_rejects_unsupported_inputs(monkeypatch):
    from unsloth.kernels import fused_decode_rope_kv
    from unsloth.kernels.fused_decode_rope_kv import fast_decode_rope_kv_ok

    bsz, n_q, n_kv, hd = 2, 8, 2, 128
    Qn = torch.zeros((bsz, 1, n_q, hd), device = "cuda", dtype = torch.bfloat16).transpose(1, 2)
    Kn = torch.zeros((bsz, 1, n_kv, hd), device = "cuda", dtype = torch.bfloat16).transpose(1, 2)
    Vn = torch.zeros((bsz, 1, n_kv, hd), device = "cuda", dtype = torch.bfloat16).transpose(1, 2)
    cos = torch.zeros((64, hd), device = "cuda", dtype = torch.bfloat16)
    sin = torch.zeros((64, hd), device = "cuda", dtype = torch.bfloat16)
    pos = torch.zeros((bsz, 1), device = "cuda", dtype = torch.int64)
    pk = torch.zeros((bsz, n_kv, hd), device = "cuda", dtype = torch.bfloat16)
    pv = torch.zeros((bsz, n_kv, hd), device = "cuda", dtype = torch.bfloat16)

    assert fast_decode_rope_kv_ok(Qn, Kn, Vn, cos, sin, pos, pk, pv)
    assert not fast_decode_rope_kv_ok(Qn.cpu(), Kn, Vn, cos, sin, pos, pk, pv)
    assert not fast_decode_rope_kv_ok(Qn, Kn.half(), Vn, cos, sin, pos, pk, pv)
    assert not fast_decode_rope_kv_ok(Qn[..., ::2], Kn, Vn, cos, sin, pos, pk, pv)
    assert not fast_decode_rope_kv_ok(Qn, Kn, Vn, cos, sin, pos.float(), pk, pv)
    monkeypatch.setattr(fused_decode_rope_kv, "FUSED_DECODE_ROPE_KV", False)
    assert not fast_decode_rope_kv_ok(Qn, Kn, Vn, cos, sin, pos, pk, pv)
