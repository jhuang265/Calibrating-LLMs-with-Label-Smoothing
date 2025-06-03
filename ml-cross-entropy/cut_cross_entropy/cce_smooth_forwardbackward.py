7# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_backward_autotune, cce_smooth_forward_autotune
from cut_cross_entropy.utils import softcapping
from cut_cross_entropy.tl_utils import (
    b_bin_fn,
    tl_and_reduce_fn,
    tl_lock_add,
    tl_softcapping,
    tl_softcapping_grad,
    tl_logaddexp,
)


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :]

    da_ptrs = da_ptrs + d_inds * stride_ad
    b_ptrs = b_ptrs + d_inds * stride_bd

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)

        da_i = tl.dot(do, b).to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)


def _cce_smooth_fused_forwardbackward_kernel(
    E,
    C,
    Inds,
    LSE,
    Barrier,
    Out,
    Locks,
    grad_scale,
    Valids,
    softcap,
    smoothing,
    dE,
    dELocks,
    dC,
    dCLocks,
    B,
    D,
    V,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_lse_b,
    stride_ib,
    stride_vb,
    filter_eps,
    num_locks,
    # Meta-parameters
    B_BIN,
    SHIFT: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    # From BACKWARD
    MM_BACK_BLOCK_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    # ITEM_DO: tl.constexpr,
    # HAS_VOCAB_ORDERING: tl.constexpr,
    FILTER_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_D:
            e = tl.load(e_ptrs)
            c = tl.load(c_ptrs)
        else:
            e = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0)
        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    logits = tl.where(v_mask[None, :], accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    o_mask = off_b < B

    this_mx = tl.max(logits, axis=1)
    e = tl.sum(tl.exp(logits - this_mx[:, None]), axis=1)

    ###
    out_ptrs = Out + offs_b
    inds = tl.load(Inds + stride_ib * ((offs_b + 1) if SHIFT else offs_b))
    mask = tl.where((inds[:, None] == offs_v[None, :]), 1, 0)
    # tl.device_print("Sum Logits:", - smoothing / V * tl.sum(logits, axis=1))

    if tl.max(mask):
        if HAS_SMOOTHING:
            tl.atomic_add(
                out_ptrs, 
                - (
                    tl.sum(((1-smoothing) * mask + smoothing / V) * logits, axis=1)
                ), 
                mask=offs_b < B
            )

        else:
            tl.atomic_add(
                out_ptrs, 
                - tl.sum(mask * logits, axis=1), 
                mask=offs_b < B
            )
    else:
        if HAS_SMOOTHING:
            tl.atomic_add(
                out_ptrs, 
                - smoothing / V * tl.sum(logits, axis=1), 
                mask=offs_b < B
            )

    ###

    this_lse = this_mx + tl.log(e)

    lse_ptrs = LSE + (stride_lse_b * off_b)

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.atomic_xchg(this_locks, 0)

    # this_semaphores = Semaphores + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    # tl.atomic_add(this_semaphores, 1)

    # while tl.atomic_max(this_semaphores, num_pid_v) != num_pid_v:
    #     pass
    # lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")

    barrier_ptr = Barrier
    tl.atomic_add(barrier_ptr, 1)
    while tl.load(barrier_ptr) != num_pid_b * num_pid_v:
        pass


    # Merging backward

    d_accum = tl.exp(accum - lse[:, None])
    
    d_accum += tl.where(mask, -(1 - smoothing - smoothing / V), -smoothing / V)

    accum_valid_mask = (
            (pid_b * BLOCK_B + tl.arange(0, BLOCK_B))[:, None] < B
        ) & (
            (pid_v * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] < V
        )
    d_accum = tl.where(accum_valid_mask, d_accum, 0.0)

    if FILTER_GRAD:
        if _block_is_filtered(tl.abs(d_accum), filter_eps):
            return

    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)
        
    d_accum = (d_accum * grad_scale).to(e_ptrs.dtype.element_ty)
    b_mask = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)[:, None]) < B
    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)[:, None]) < V

    lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
    dELocks += lock_offset

    _mm_backward(
        d_accum,
        dE + (offs_b[:, None] * stride_eb),
        b_mask,
        dELocks,
        n_de_locks_1,
        C + offs_v[:, None] * stride_cv,
        v_mask,
        stride_ed,
        stride_cd,
        D,
        MM_BACK_BLOCK_D,
        MM_BACK_EVEN_D,
    )

    lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
    dCLocks += lock_offset

    _mm_backward(
        tl.trans(d_accum),
        dC + (offs_v[:, None] * stride_cv),
        v_mask,
        dCLocks,
        n_dc_locks_1,
        E + (offs_b[:, None] * stride_eb),
        b_mask,
        stride_cd,
        stride_ed,
        D,
        MM_BACK_BLOCK_D,
        MM_BACK_EVEN_D,
    )


_cce_smooth_fused_forwardbackward_kernel = triton.jit(_cce_smooth_fused_forwardbackward_kernel)
_cce_smooth_fused_forwardbackward_kernel = triton.heuristics(  # type: ignore
    {
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "GROUP_B": lambda args: 8,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0,
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
        "MM_BACK_BLOCK_D": lambda args: args["BLOCK_D"] * 2,
        "MM_BACK_EVEN_D": lambda args: (args["D"] % (args["BLOCK_D"] * 2)) == 0,
        # "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        # "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "FILTER_GRAD": lambda args: args["filter_eps"] is not None,
        "HAS_TARGETS": lambda args: args["Inds"] is not None,
    }
)(_cce_smooth_fused_forwardbackward_kernel)
_cce_smooth_fused_forwardbackward_kernel = cce_backward_autotune()(_cce_smooth_fused_forwardbackward_kernel)  # type: ignore


def cce_smooth_fused_forwardbackward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    inds: torch.Tensor,
    shift: bool,
    valids: torch.Tensor | None,
    softcap: float | None,
    smoothing: float,
    filter_eps: float,
    grad_scale: float = 1.0,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    # assert do.numel() in (e.size(0), 1)
    assert c.size(1) == e.size(1)
    assert e.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires embeddings to be bf16 or fp16"
    assert c.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires classifier to be bf16 or fp16"
    
    assert inds.ndim == 1
    assert e.ndim == 2
    assert c.ndim == 2
    assert inds.size(0) == e.size(0)
    assert c.size(1) == e.size(1)

    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    V, D = c.shape

    lse = e.new_full(
        (B,), 
        -float("inf"), 
        dtype=torch.float32
    )

    # counts = e.new_full(
    #     (B,), 
    #     128, 
    #     dtype=torch.uint32
    # )

    lse = lse.contiguous()
    # counts = counts.contiguous()

    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    barrier = e.new_full(
        (1,),
        0,
        dtype=torch.uint32,
    )

    # semaphores = e.new_full(
    #     (triton.cdiv(B, 128),),
    #     0,
    #     dtype=torch.uint32,
    # )

    out = e.new_zeros(
        (B,), 
        dtype=torch.float32
    )

    de = torch.zeros_like(e)
    dc = torch.zeros_like(c)

    assert de.stride() == e.stride()
    assert dc.stride() == c.stride()

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.size(0), META["BLOCK_V"]),)

    nd_locks = triton.cdiv(c.size(1), 64)
    de_locks = e.new_zeros(
        (triton.cdiv(B, nd_locks), nd_locks), 
        dtype=torch.int32
    )
    dc_locks = c.new_zeros(
        (triton.cdiv(c.size(0), nd_locks), nd_locks), 
        dtype=torch.int32
    )

    _cce_smooth_fused_forwardbackward_kernel[grid](
        E=e,
        C=c,
        Inds=inds,
        LSE=lse,
        Barrier=barrier,
        Out=out,
        Locks=locks,
        grad_scale=grad_scale,
        Valids=valids,
        softcap=softcap,
        smoothing=smoothing,
        dE=de,
        dELocks=de_locks,
        dC=dc,
        dCLocks=dc_locks,
        B=B,
        V=V,
        D=D,
        n_de_locks_0=de_locks.size(0),
        n_de_locks_1=de_locks.size(1),
        n_dc_locks_0=dc_locks.size(0),
        n_dc_locks_1=dc_locks.size(1),
        stride_eb=e.stride(0),
        stride_ed=e.stride(1),
        stride_cv=c.stride(0),
        stride_cd=c.stride(1),
        stride_lse_b=lse.stride(0),
        stride_ib=inds.stride(0),
        stride_vb=1 if valids is None else valids.stride(0),
        filter_eps=filter_eps,
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
    )

    if softcap is not None:
        out = softcapping(out, softcap)

    if out_dtype is None:
        out_dtype = e.dtype

    out = out.to(out_dtype)

    # print(f"Shape of `lse` before return: {lse.shape}")
    # print(f"Shape of `out` before return: {out.shape}")
    
    return lse, out, de, dc

@triton.jit
def _element_mul_kernel(
    da_ptrs,
    dOut,
    stride_ad,
    D,
    BLOCK_D: tl.constexpr,
):

    # Get the program ID and convert it to int64 to avoid overflow
    pid = tl.program_id(0)

    # Locate the start index
    ptr += pid * stride

    # Load the gradient output value
    d_out = tl.load(dOut)

    # Perform the element-wise multiplication
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        offsets = d + tl.arange(0, BLOCK_D)
        block = tl.load(ptr + offsets, mask=offsets < D)
        tl.store(ptr + offsets, block * grad_output, mask=offsets < D)

@triton.jit
def _mul_kernel(
    dE,
    dC,
    dOut,
    Valids,
    B,
    D,
    V,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_v_in_group) % group_size_b)
    pid_v = (pid % num_v_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v)

    offs_d = tl.arange(0, BLOCK_D)
    de_ptrs = dE + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    dc_ptrs = dC + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        d_out = tl.load(dOut + ((offs_b + 1) if SHIFT else offs_b))[:, None]

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            de = tl.load(de_ptrs)
            dc = tl.load(dc_ptrs)
        else:
            de = tl.load(de_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            dc = tl.load(dc_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0)

        tl.store(de_ptrs, e * d_out, mask=o_mask, eviction_policy="evict_last")
        tl.store(dc_ptrs, c * d_out, mask=o_mask, eviction_policy="evict_last")

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd


def cce_smooth_fused_backward_kernel(
    do: torch.Tensor,
    de: torch.Tensor,
    dc: torch.tensor,
    valids: torch.Tensor | None,
    shift: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    if do.numel() == 1:
        if torch.equal(do, torch.tensor(1.0, device=do.device)):
            pass
        else:
            B, _ = de.shape
            V, D = dc.shape

            n_rows = B

            element_mul_kernel[(n_rows,)](
                da_ptrs=de,
                stride_ad=de.stride(-2),
                dOut=do,
                D=D,
            )
            n_rows = V

            element_mul_kernel[(n_rows,)](
                da_ptrs=dc,
                stride_ad=dc.stride(-2),
                dOut=do,
                D=D,
            )
    # else:
    #     continue

    #     assert do.numel() in (de.size(0), 1)
    #     assert dc.size(1) == de.size(1)
    #     assert de.dtype in (
    #         torch.float16,
    #         torch.bfloat16,
    #     ), "Backwards requires embeddings to be bf16 or fp16"
    #     assert dc.dtype in (
    #         torch.float16,
    #         torch.bfloat16,
    #     ), "Backwards requires classifier to be bf16 or fp16"

    #     if valids is not None:
    #         assert valids.ndim == 1
    #         B = valids.size(0)
    #     else:
    #         B, D = de.size(0)

    #     if do.numel() > 1:
    #         do = do.contiguous()
            
    #     def grid(META):
    #         return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.size(0), META["BLOCK_V"]),)

    #     _mul_kernel[grid](
    #         e,
    #         c,
    #         do,
    #         valids,
    #         B,
    #         de.size(1),
    #         dc.size(0),
    #         de.stride(0),
    #         de.stride(1),
    #         dc.stride(0),
    #         dc.stride(1),
    #         1 if valids is None else valids.stride(0),
    #         B_BIN=b_bin_fn(B),
    #         SHIFT=shift,
    #     )

    return de, dc