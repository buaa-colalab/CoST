# ------------------------------------------------------------------------------------------------
# Thin wrapper of MMCV MultiScaleDeformableAttention
# Keeps your public API identical to the original function-style MSDeformAttn.
# Shapes:
#   query:                (N, Len_q, C)
#   reference_points:     (N, Len_q, n_levels, 2 or 4)
#   input_flatten (value):(N, sum(H_l*W_l), C)
#   input_spatial_shapes: (n_levels, 2) as (H_l, W_l)
#   input_level_start_index: (n_levels,)
#   input_padding_mask:   (N, sum(H_l*W_l))  # True means padding
#
# If dif_offset=True and `types` is provided, we learn two separate MSDA branches and
# select outputs by `types == 0` (main) vs else (infra).
#
# Requires mmcv with CUDA ops compiled for your GPU.
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division

import warnings
from typing import Optional

import torch
from torch import nn

try:
    from mmcv.ops import MultiScaleDeformableAttention as MMCV_MSDA
except Exception as e:
    raise ImportError(
        "Failed to import mmcv.ops.MultiScaleDeformableAttention. "
        "Please install/compile MMCV with CUDA ops (MMCV_WITH_OPS=1). "
        f"Original error: {e}"
    )

__all__ = ["MSDeformAttn"]


def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        dif_offset: bool = False,
        cav_num: int = 2,
    ):
        """
        Multi-Scale Deformable Attention (MMCV-backed).

        Args:
            d_model: hidden dim (C). Must be divisible by n_heads.
            n_levels: number of feature levels.
            n_heads: number of attention heads.
            n_points: sampling points per head per level.
            dif_offset: if True, maintain a second MSDA branch (infra) and select by `types`.
            cav_num: kept for compatibility; unused.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model so that each head dim is a power of 2 "
                "(more efficient in CUDA implementation)."
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.cav_num = cav_num
        self.dif_offset = dif_offset

        # Try to construct with batch_first=True (newer mmcv). Fallback if not supported.
        self._batch_first = True
        try:
            self.msda = MMCV_MSDA(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
                batch_first=True,
            )
        except TypeError:
            # Older mmcv without batch_first argument
            self.msda = MMCV_MSDA(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
            )
            self._batch_first = False

        if self.dif_offset:
            try:
                self.msda_infra = MMCV_MSDA(
                    embed_dims=d_model,
                    num_levels=n_levels,
                    num_heads=n_heads,
                    num_points=n_points,
                    batch_first=True,
                )
            except TypeError:
                self.msda_infra = MMCV_MSDA(
                    embed_dims=d_model,
                    num_levels=n_levels,
                    num_heads=n_heads,
                    num_points=n_points,
                )

    # === Important ===
    # Some upstream code expects this method to exist.
    def _reset_parameters(self):
        # Defer to mmcv's own initialization if available.
        for m in [self.msda, getattr(self, "msda_infra", None)]:
            if m is None:
                continue
            if hasattr(m, "init_weights"):
                m.init_weights()
            elif hasattr(m, "_reset_parameters"):
                # Some versions may expose a similar internal method
                m._reset_parameters()  # type: ignore
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _call_mmcv(
        self,
        module: MMCV_MSDA,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Call mmcv MSDA with correct layout and without internal residual
        (to avoid double-residual with outer Transformer layer).
        """
        # Handle older mmcv that expects [Len, B, C]
        need_transpose = not self._batch_first
        if need_transpose:
            # [N, L, C] -> [L, N, C]
            query_t = query.transpose(0, 1).contiguous()
            value_t = value.transpose(0, 1).contiguous()
            # Call without internal residual: identity=zeros_like(query_t)
            out_t = module(
                query=query_t,
                reference_points=reference_points,
                value=value_t,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                identity=torch.zeros_like(query_t),
            )
            # Back to [N, L, C]
            return out_t.transpose(0, 1).contiguous()

        # Newer mmcv supports batch_first=True
        return module(
            query=query,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
            identity=torch.zeros_like(query),  # disable internal residual
        )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
        types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:                (N, Length_query, C)
            reference_points:     (N, Length_query, n_levels, 2) in [0,1] or (..., 4) boxes
            input_flatten:        (N, sum(H_l*W_l), C)
            input_spatial_shapes: (n_levels, 2) as (H_l, W_l)
            input_level_start_index: (n_levels,)
            input_padding_mask:   (N, sum(H_l*W_l))  True for padding
            types:                optional selector; if provided with dif_offset=True,
                                  choose msda when (types==0) else msda_infra.

        Returns:
            output:               (N, Length_query, C)
        """
        # Sanity check to mirror original assertion.
        if input_spatial_shapes.numel() > 0:
            expected_len = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item()
            if input_flatten.size(1) != expected_len:
                raise AssertionError(
                    f"input_flatten length mismatch: got {input_flatten.size(1)}, "
                    f"expected {expected_len} from spatial_shapes"
                )

        if self.dif_offset and types is not None:
            out0 = self._call_mmcv(
                self.msda,
                query=query,
                reference_points=reference_points,
                value=input_flatten,
                spatial_shapes=input_spatial_shapes,
                level_start_index=input_level_start_index,
                key_padding_mask=input_padding_mask,
            )
            out1 = self._call_mmcv(
                self.msda_infra,
                query=query,
                reference_points=reference_points,
                value=input_flatten,
                spatial_shapes=input_spatial_shapes,
                level_start_index=input_level_start_index,
                key_padding_mask=input_padding_mask,
            )
            mask = (types == 0)
            while mask.dim() < out0.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.to(dtype=torch.bool, device=out0.device)
            return torch.where(mask, out0, out1)

        return self._call_mmcv(
            self.msda,
            query=query,
            reference_points=reference_points,
            value=input_flatten,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
            key_padding_mask=input_padding_mask,
        )
