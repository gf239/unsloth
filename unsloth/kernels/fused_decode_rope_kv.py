# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Decode optional QK RMSNorm + half-rotated RoPE + KV cache write in one Triton launch.

import os
import torch
import triton
import triton.language as tl
from .utils import MAX_FUSED_SIZE, torch_gpu_device
from .rms_layernorm import _launch, _eager_kernel

__all__ = [
    "fast_decode_rope_kv",
    "fast_decode_rope_kv_ok",
]

FUSED_DECODE_ROPE_KV = os.environ.get("UNSLOTH_FUSED_DECODE_ROPE_KV", "1") != "0"


@triton.jit
def _decode_rope_kv_kernel(
    Qn,
    Kn,
    Vn,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    paged_K_slice,
    paged_V_slice,
    q_w_ptr,
    k_w_ptr,
    pos_stride,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    total_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    HAS_QK_NORM: tl.constexpr,
    q_eps: tl.constexpr,
    k_eps: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // total_heads
    h = pid % total_heads

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half_dim
    pos = tl.load(pos_ptr + b * pos_stride).to(tl.int64)
    cos_base = cos_ptr + pos * head_dim
    sin_base = sin_ptr + pos * head_dim

    c1 = tl.load(cos_base + offs, mask = mask, other = 0.0).to(Qn.dtype.element_ty)
    c2 = tl.load(cos_base + half_dim + offs, mask = mask, other = 0.0).to(Qn.dtype.element_ty)
    s1 = tl.load(sin_base + offs, mask = mask, other = 0.0).to(Qn.dtype.element_ty)
    s2 = tl.load(sin_base + half_dim + offs, mask = mask, other = 0.0).to(Qn.dtype.element_ty)

    if h < n_q_heads:
        q_base = Qn + (b * n_q_heads + h) * head_dim
        q1 = tl.load(q_base + offs, mask = mask, other = 0.0)
        q2 = tl.load(q_base + half_dim + offs, mask = mask, other = 0.0)
        if HAS_QK_NORM:
            q1_f = q1.to(tl.float32)
            q2_f = q2.to(tl.float32)
            var = tl.sum(q1_f * q1_f + q2_f * q2_f, axis = 0) * (1.0 / head_dim)
            inv_var = tl.math.rsqrt(var + tl.full((), q_eps, tl.float32))
            q1_n = (q1_f * inv_var).to(Qn.dtype.element_ty)
            q2_n = (q2_f * inv_var).to(Qn.dtype.element_ty)
            wq1 = tl.load(q_w_ptr + offs, mask = mask, other = 0.0)
            wq2 = tl.load(q_w_ptr + half_dim + offs, mask = mask, other = 0.0)
            q1 = (q1_n.to(tl.float32) * wq1.to(tl.float32)).to(Qn.dtype.element_ty)
            q2 = (q2_n.to(tl.float32) * wq2.to(tl.float32)).to(Qn.dtype.element_ty)
        q1_c = q1 * c1
        q2_c = q2 * c2
        if Qn.dtype.element_ty == tl.float32:
            out_q1 = tl.fma(-q2, s1, q1_c)
            out_q2 = tl.fma(q1, s2, q2_c)
        else:
            out_q1 = (q1_c.to(tl.float32) + (-q2).to(tl.float32) * s1.to(tl.float32)).to(
                Qn.dtype.element_ty
            )
            out_q2 = (q2_c.to(tl.float32) + q1.to(tl.float32) * s2.to(tl.float32)).to(
                Qn.dtype.element_ty
            )
        tl.store(q_base + offs, out_q1, mask = mask)
        tl.store(q_base + half_dim + offs, out_q2, mask = mask)
    else:
        kv_h = h - n_q_heads
        kv_offset = (b * n_kv_heads + kv_h) * head_dim
        k_base = Kn + kv_offset
        v_base = Vn + kv_offset
        pk_base = paged_K_slice + kv_offset
        pv_base = paged_V_slice + kv_offset

        k1 = tl.load(k_base + offs, mask = mask, other = 0.0)
        k2 = tl.load(k_base + half_dim + offs, mask = mask, other = 0.0)
        if HAS_QK_NORM:
            k1_f = k1.to(tl.float32)
            k2_f = k2.to(tl.float32)
            var = tl.sum(k1_f * k1_f + k2_f * k2_f, axis = 0) * (1.0 / head_dim)
            inv_var = tl.math.rsqrt(var + tl.full((), k_eps, tl.float32))
            k1_n = (k1_f * inv_var).to(Kn.dtype.element_ty)
            k2_n = (k2_f * inv_var).to(Kn.dtype.element_ty)
            wk1 = tl.load(k_w_ptr + offs, mask = mask, other = 0.0)
            wk2 = tl.load(k_w_ptr + half_dim + offs, mask = mask, other = 0.0)
            k1 = (k1_n.to(tl.float32) * wk1.to(tl.float32)).to(Kn.dtype.element_ty)
            k2 = (k2_n.to(tl.float32) * wk2.to(tl.float32)).to(Kn.dtype.element_ty)
        k1_c = k1 * c1
        k2_c = k2 * c2
        if Kn.dtype.element_ty == tl.float32:
            out_k1 = tl.fma(-k2, s1, k1_c)
            out_k2 = tl.fma(k1, s2, k2_c)
        else:
            out_k1 = (k1_c.to(tl.float32) + (-k2).to(tl.float32) * s1.to(tl.float32)).to(
                Kn.dtype.element_ty
            )
            out_k2 = (k2_c.to(tl.float32) + k1.to(tl.float32) * s2.to(tl.float32)).to(
                Kn.dtype.element_ty
            )
        tl.store(pk_base + offs, out_k1, mask = mask)
        tl.store(pk_base + half_dim + offs, out_k2, mask = mask)

        v1 = tl.load(v_base + offs, mask = mask, other = 0.0)
        v2 = tl.load(v_base + half_dim + offs, mask = mask, other = 0.0)
        tl.store(pv_base + offs, v1, mask = mask)
        tl.store(pv_base + half_dim + offs, v2, mask = mask)


def _head_rows_ok(X, bsz, n_heads, head_dim):
    return (
        X.ndim == 4
        and X.shape == (bsz, n_heads, 1, head_dim)
        and X.stride(-1) == 1
        and X.stride(1) == head_dim
        and (bsz == 1 or X.stride(0) == n_heads * head_dim)
    )


def _paged_slice_ok(P, bsz, n_kv_heads, head_dim):
    return (
        P.ndim == 3
        and P.shape == (bsz, n_kv_heads, head_dim)
        and P.stride(-1) == 1
        and P.stride(1) == head_dim
        and (bsz == 1 or P.stride(0) == n_kv_heads * head_dim)
    )


def fast_decode_rope_kv_ok(
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
    if not FUSED_DECODE_ROPE_KV or Qn.device.type != "cuda" or torch.compiler.is_compiling():
        return False
    if Qn.ndim != 4 or Kn.ndim != 4:
        return False
    bsz, n_q_heads, q_len, head_dim = Qn.shape
    n_kv_heads = Kn.shape[1]
    if q_len != 1 or head_dim % 2 != 0 or (head_dim // 2) > MAX_FUSED_SIZE:
        return False
    device = Qn.device
    dtype = Qn.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if (
        Kn.device != device
        or Vn.device != device
        or cos.device != device
        or sin.device != device
        or position_ids.device != device
        or paged_K_slice.device != device
        or paged_V_slice.device != device
        or Kn.dtype != dtype
        or Vn.dtype != dtype
        or paged_K_slice.dtype != dtype
        or paged_V_slice.dtype != dtype
    ):
        return False
    if (
        not _head_rows_ok(Qn, bsz, n_q_heads, head_dim)
        or not _head_rows_ok(Kn, bsz, n_kv_heads, head_dim)
        or not _head_rows_ok(Vn, bsz, n_kv_heads, head_dim)
        or not _paged_slice_ok(paged_K_slice, bsz, n_kv_heads, head_dim)
        or not _paged_slice_ok(paged_V_slice, bsz, n_kv_heads, head_dim)
    ):
        return False
    if (
        not cos.is_contiguous()
        or not sin.is_contiguous()
        or cos.shape[-1] != head_dim
        or sin.shape[-1] != head_dim
    ):
        return False
    if position_ids.dtype not in (torch.int32, torch.int64) or not (
        position_ids.numel() == 1 or (position_ids.shape[0] == bsz and position_ids.numel() == bsz)
    ):
        return False
    if (q_norm is None) != (k_norm is None):
        return False
    if q_norm is not None:
        qw, kw = q_norm.weight, k_norm.weight
        if (
            qw.device != device
            or kw.device != device
            or not qw.is_contiguous()
            or not kw.is_contiguous()
            or qw.numel() != head_dim
            or kw.numel() != head_dim
        ):
            return False
    return True


def fast_decode_rope_kv(
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
    """Single-token decode step: optional Q/K RMSNorm + RoPE on Qn (in-place) and Kn, writing
    rotated Kn and Vn into paged_K_slice and paged_V_slice. No autograd."""
    bsz, n_q_heads, _, head_dim = Qn.shape
    n_kv_heads = Kn.shape[1]
    total_heads = n_q_heads + n_kv_heads
    half_dim = head_dim // 2
    BLOCK_HALF = triton.next_power_of_2(half_dim)
    has_qk_norm = q_norm is not None
    if has_qk_norm:
        q_w = q_norm.weight
        k_w = k_norm.weight
        q_eps = float(getattr(q_norm, "variance_epsilon", getattr(q_norm, "eps", 1e-6)))
        k_eps = float(getattr(k_norm, "variance_epsilon", getattr(k_norm, "eps", 1e-6)))
    else:
        q_w = Qn
        k_w = Kn
        q_eps = 1e-6
        k_eps = 1e-6
    pos_stride = 0 if position_ids.numel() == 1 else position_ids.stride(0)
    device = Qn.device
    with torch_gpu_device(device):
        _launch(
            _eager_kernel,
            _decode_rope_kv_kernel,
            (bsz * total_heads,),
            (
                Qn,
                Kn,
                Vn,
                cos,
                sin,
                position_ids,
                paged_K_slice,
                paged_V_slice,
                q_w,
                k_w,
                pos_stride,
            ),
            10,
            dict(
                n_q_heads = n_q_heads,
                n_kv_heads = n_kv_heads,
                total_heads = total_heads,
                head_dim = head_dim,
                half_dim = half_dim,
                BLOCK_HALF = BLOCK_HALF,
                HAS_QK_NORM = has_qk_norm,
                q_eps = q_eps,
                k_eps = k_eps,
            ),
            device,
            num_warps = 1,
            enable_fp_fusion = False,
        )
    return Qn
