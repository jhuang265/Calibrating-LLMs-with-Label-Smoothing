# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch

from cut_cross_entropy.cce_smooth_forward import cce_smooth_forward_kernel
from cut_cross_entropy.cce_smooth_backward import cce_smooth_backward_kernel
from cut_cross_entropy.cce_smooth_forwardbackward import cce_smooth_fused_forwardbackward_kernel, cce_smooth_fused_backward_kernel

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import (
    _build_flat_valids,
    _handle_eps,
    handle_reduction_none,
)


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: bool
    batch_shape: torch.Size
    label_smoothing: float


@torch.compile(fullgraph=True, dynamic=True)
def sort_logit_avg(logit_avg: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logit_avg).to(torch.int32)


class LinearFusedSmoothCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        params: CCEParams,
    ) -> torch.Tensor:
        needs_grad = e.requires_grad or c.requires_grad

        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / e.size(0)
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ret = cce_smooth_fused_forwardbackward_kernel(
            e,
            c,
            params.targets, 
            params.shift,
            params.valids,
            softcap=params.softcap,
            smoothing=params.label_smoothing,
            filter_eps=params.filter_eps,
            out_dtype=torch.float32,
            grad_scale=grad_scale,
        )

        lse, out, de, dc = ret

        # print(f"Fused Smooth Params: smoothing={params.label_smoothing}, filter_eps={params.filter_eps}")
        # print(f" => out={out}, lse={lse}")

        nll = out.add_(lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        # print(f" => loss={loss}")

        # ctx.save_for_backward(de, dc)
        ctx.save_for_backward(de.detach(), dc.detach(), params.valids)
        ctx.params = params

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        de, dc, valids = ctx.saved_tensors

        params = cast(CCEParams, ctx.params)

        de, dc = cce_smooth_fused_backward_kernel(
            grad_out,
            de, 
            dc,
            valids,
            shift=params.shift
        )

        return de, dc, None


class LinearSmoothCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        params: CCEParams,
    ) -> torch.Tensor:
        needs_grad = e.requires_grad or c.requires_grad
        return_logit_avg = needs_grad and params.filter_eps is not None

        ret = cce_smooth_forward_kernel(
            e,
            c,
            params.targets, 
            params.shift,
            params.valids,
            softcap=params.softcap,
            smoothing=params.label_smoothing,
            return_logit_avg=return_logit_avg,
            out_dtype=torch.float32
        )
        if return_logit_avg:
            # assert isinstance(ret, tuple)
            lse, out, logit_avg = ret
        else:
            # assert isinstance(ret, torch.Tensor)
            lse, out = ret
            logit_avg = None

        # print(f"Shape of `lse` after return: {lse.shape}")
        # print(f"Shape of `out` after return: {out.shape}")

        # print(f"Smooth Params: return_logit_avg={return_logit_avg}, smoothing={params.label_smoothing}, filter_eps={params.filter_eps}")
        # print(f" => out={out}, lse={lse}")

        nll = out.add_(lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        # print(f" => loss={loss}, logit_avg={logit_avg}")

        ctx.save_for_backward(e, c, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        h, w, lse, targets, valids, logit_avg = ctx.saved_tensors

        # print(logit_avg)

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / lse.numel()
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        de, dc = cce_smooth_backward_kernel(
            grad_out,
            h,
            w,
            lse,
            valids,
            params.softcap,
            params.label_smoothing,
            params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale,
        )

        return de, dc, None


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        params: CCEParams,
    ) -> torch.Tensor:
        needs_grad = e.requires_grad or c.requires_grad
        return_logit_avg = needs_grad and params.filter_eps is not None

        ret = cce_lse_forward_kernel(
            e,
            c,
            params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            assert isinstance(ret, tuple)
            lse, logit_avg = ret
        else:
            assert isinstance(ret, torch.Tensor)
            lse = ret
            logit_avg = None

        neg_dot = indexed_neg_dot_forward_kernel(
            e, c, params.targets, params.shift, params.valids, params.softcap, lse.dtype
        )

        # print(f"CCE Params: return_logit_avg={return_logit_avg}, smoothing=0.0, filter_eps={params.filter_eps}")
        # print(f" => out={neg_dot}, lse={lse}")

        nll = neg_dot.add_(lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        # print(f" => loss={loss}, logit_avg={logit_avg}")

        ctx.save_for_backward(e, c, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        h, w, lse, targets, valids, logit_avg = ctx.saved_tensors

        # # print(logit_avg)

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / lse.numel()
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        de, dc = cce_backward_kernel(
            grad_out,
            h,
            w,
            lse,
            valids,
            params.softcap,
            params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale,
        )

        return de, dc, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    params: CCEParams,
) -> torch.Tensor:

    if params.label_smoothing is None:
        loss = LinearCrossEntropyFunction.apply(e, c, params)
    elif params.label_smoothing >= 0:
        loss = LinearSmoothCrossEntropyFunction.apply(e, c, params)

    assert isinstance(loss, torch.Tensor)

    if params.shift and params.reduction == "none":
        loss = loss[..., 1:]

    return loss


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: float | str | None = "auto",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    :param filter_eps: The threshold value used to determine which locations can be safely ignored
        in gradient computation. The default value of "auto" will automatically choose a value
        based on the input dtype.
    """
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    return linear_cross_entropy_apply(
        e,
        c,
        CCEParams(
            targets,
            valids,
            softcap,
            reduction,
            None, #_handle_eps(filter_eps, e.dtype),
            shift,
            batch_shape,
            label_smoothing,
        ),
    )


def fused_linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    params: CCEParams,
) -> torch.Tensor:

    loss = LinearFusedSmoothCrossEntropyFunction.apply(e, c, params)

    assert isinstance(loss, torch.Tensor)

    if params.shift and params.reduction == "none":
        loss = loss[..., 1:]

    return loss



@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def fused_cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: float | str | None = "auto",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    :param filter_eps: The threshold value used to determine which locations can be safely ignored
        in gradient computation. The default value of "auto" will automatically choose a value
        based on the input dtype.
    """
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    return fused_linear_cross_entropy_apply(
        e,
        c,
        CCEParams(
            targets,
            valids,
            softcap,
            reduction,
            _handle_eps(filter_eps, e.dtype),
            shift,
            batch_shape,
            label_smoothing,
        ),
    )
