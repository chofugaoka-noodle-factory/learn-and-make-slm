from jaxtyping import Float, Int
from torch import Tensor, nn
import torch
import math


def rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Float[Tensor, "..."],
        cos_cache: Float[Tensor, "S_max D"],
        sin_cache: Float[Tensor, "S_max D"],
        position_ids: Int[Tensor, "B S"],
        heads: int,
        embedding_dim: int,
    ) -> Float[Tensor, "..."]:
        if embedding_dim != cos_cache.size(-1):
            cos_cache = (
                cos_cache.unsqueeze(2)  # (seq_len, dim/2, 1)
                .expand(cos_cache.size(0), -1, 2)  # (seq_len, dim/2, 2)
                .reshape(cos_cache.size(0), -1)  # (seq_len, dim)
            )
            sin_cache = (
                sin_cache.unsqueeze(2)  # (seq_len, dim/2, 1)
                .expand(sin_cache.size(0), -1, 2)  # (seq_len, dim/2, 2)
                .reshape(sin_cache.size(0), -1)  # (seq_len, dim)
            )

        cos = cos_cache[position_ids]  # (B, seq_len, dim)
        sin = sin_cache[position_ids]  # (B, seq_len, dim)

        # Handle multi-head case: x can be (B, H, S, D) or (B, S, D)
        # Adjust cos/sin shape to match x
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(1)  # Add head dimension if needed
            sin = sin.unsqueeze(1)

        return (x * cos) + (rotate_half(x) * sin)

    @staticmethod
    def symbolic(
        g,
        x: Float[Tensor, "..."],
        cos_cache: Float[Tensor, "S_max D"],
        sin_cache: Float[Tensor, "S_max D"],
        position_ids: Int[Tensor, "B S"],
        heads: int,
        embedding_dim: int,
    ) -> Float[Tensor, "..."]:
        return g.op(
            "RotaryEmbedding",
            x,
            cos_cache,
            sin_cache,
            position_ids,
            interleaved_i=1,
            num_heads_i=heads,
            rotary_embedding_dim_i=embedding_dim,
        )


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        scale_factor: float = 1.0,
        enable_dynamic_scaling: bool = False,
    ) -> None:
        """
        Rotary Positional Embedding (RoPE) and YaRN (Yet another RoPE extensioN method) implementation.

        Args:
            dim: Dimension of the embedding (must be even)
            max_seq_len: Maximum sequence length for pre-computation
            scale_factor: Context extension scale factor s = L'/L (default: 1.0 for no extension), if scale_factor > 1.0, use YaRN.
            enable_dynamic_scaling: Whether to use dynamic scaling at inference time
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryPositionalEmbedding expects an even dimension.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self._cached_scale_factor: float = scale_factor

        if self.scale_factor > 1.0:
            # use YaRN
            # Compute attention temperature scaling
            # sqrt(1/t) = 0.1 * ln(s) + 1
            self.attention_scale = 0.1 * math.log(scale_factor) + 1.0
        else:
            # use normal RoPE
            self.attention_scale = 1.0

        self.base = 10_000
        self.alpha = 1.0
        self.beta = 32.0

        # Buffers move with the module and are optionally re-computed on demand.
        self.cos: Tensor  # (max_seq_len, dim)
        self.sin: Tensor  # (max_seq_len, dim)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

        self._update_cache(
            seq_len=max_seq_len,
            device=torch.device("cpu"),
            dtype=torch.get_default_dtype(),
            scale_factor=scale_factor,
        )

    def _compute_ramp_function(self, r: Float[Tensor, "D"]) -> Float[Tensor, "D"]:  # noqa: F821
        """
        Compute the ramp function for NTK-by-parts interpolation.

        Returns gamma where:
            gamma = 0 if r < low (high frequency dimensions, no interpolation)
            gamma = 1 if r > high (low frequency dimensions, full interpolation)
            gamma = (r - low)/(high - low) otherwise (smooth transition)

        where r is the dimension index (0, 1, 2, ...) and low/high are
        thresholds computed from alpha, beta, and original_max_seq_len.
        """
        dim_half = self.dim / 2
        # NTK-by-parts: compute dimension thresholds
        # See GPT-OSS implementation: https://github.com/openai/gpt-oss
        low = dim_half * math.log(self.original_max_seq_len / (self.beta * 2 * math.pi)) / math.log(self.base)
        high = dim_half * math.log(self.original_max_seq_len / (self.alpha * 2 * math.pi)) / math.log(self.base)
        assert 0 < low < high < dim_half - 1

        return torch.clamp((r - low) / (high - low), min=0.0, max=1.0)

    def _compute_rotates(
        self,
        *,
        max_seq_len: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        scale_factor: float,
        interleave: bool = True,
    ) -> tuple[Float[Tensor, "{max_seq_len} D"], Float[Tensor, "{max_seq_len} D"]]:  # noqa: F821
        """
        Compute cos and sin matrices with YaRN interpolation.
        """
        # Compute dimension indices (only even dimensions: 0, 2, 4, ...)
        d_indices = torch.arange(0, dim, 2, dtype=dtype, device=device)  # (dim/2,)
        base_thetas = 1.0 / (self.base ** (d_indices / dim))  # (dim/2,)

        if scale_factor > 1.0:  # use YaRN
            r = d_indices / 2  # (0, 1, 2, 3, ...)

            # Compute ramp function γ(r)
            gamma = self._compute_ramp_function(r)  # (dim/2,)

            # NTK-by-parts interpolation:
            # gamma = 0 (d < low, high freq): use base_thetas (no interpolation, preserve original)
            # gamma = 1 (d > high, low freq): use base_thetas/s (interpolation)
            interpolated_thetas = gamma * (base_thetas / scale_factor) + (1 - gamma) * base_thetas

        else:  # use normal RoPE
            interpolated_thetas = base_thetas

        if interleave:
            # Expand to full dimension (interleave for cos/sin pairs)
            thetas = (
                interpolated_thetas.unsqueeze(1)  # (dim/2, 1)
                .expand(-1, 2)  # (dim/2, 2)
                .reshape(-1)  # (dim,)
            )
        else:
            thetas = interpolated_thetas  # (dim/2)

        # Compute position indices
        pos = torch.arange(0, max_seq_len, dtype=dtype, device=device)  # (max_seq_len,)

        # Compute rotations
        rotate = torch.outer(pos, thetas)  # (max_seq_len, dim) or (max_seq_len, dim/2)

        # Apply attention scaling
        cos = torch.cos(rotate) * self.attention_scale
        sin = torch.sin(rotate) * self.attention_scale

        return cos, sin

    def _update_cache(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        scale_factor: float,
    ) -> None:
        """Update the cos/sin cache if necessary."""
        needs_rebuild = (
            self.cos.numel() == 0  # initial build
            or seq_len > self.max_seq_len
            or scale_factor != self._cached_scale_factor  # dynamic scaling
            or self.cos.device != device
            or self.cos.dtype != dtype
        )
        if not needs_rebuild:
            return

        target_len = max(seq_len, self.max_seq_len)
        _cos, _sin = self._compute_rotates(
            max_seq_len=target_len,
            dim=self.dim,
            device=device,
            dtype=dtype,
            scale_factor=scale_factor,
        )
        self.cos = _cos
        self.sin = _sin
        self.max_seq_len = target_len
        self._cached_scale_factor = scale_factor

    def forward(
        self,
        x: Float[Tensor, "... seq_len {self.dim}"],
        positional_offset: int = 0,
        # When using KV cache, we get one token at a time, but the token can be at any position in the sequence.
    ) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        """
        Apply YaRN rotary positional encoding to input tensor.
        """
        # Handle 2D input [seq_len, dim] by adding batch dimension
        input_ndim = x.ndim
        if input_ndim == 2:
            x = x.unsqueeze(0)  # [seq_len, dim] -> [1, seq_len, dim]

        batch_size = x.size(0)
        seq_len = x.size(-2)
        total_seq_len = seq_len + positional_offset

        # Dynamic scaling: adjust scale_factor based on current sequence length
        if self.enable_dynamic_scaling:
            assert self.scale_factor > 1.0, "Dynamic scaling is only supported for YaRN, not for normal RoPE"
            dynamic_scale = max(1.0, total_seq_len / self.original_max_seq_len)
            # Recompute attention scale for dynamic scaling
            self.attention_scale = 0.1 * math.log(dynamic_scale) + 1.0

            self._update_cache(
                seq_len=total_seq_len,
                device=x.device,
                dtype=x.dtype,
                scale_factor=dynamic_scale,
            )
        else:
            self._update_cache(
                seq_len=total_seq_len,
                device=x.device,
                dtype=x.dtype,
                scale_factor=self.scale_factor,
            )

        position_ids = torch.arange(positional_offset, total_seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)

        # cos = self.cos[positional_offset:total_seq_len, :]  # (seq_len, dim)
        # sin = self.sin[positional_offset:total_seq_len, :]  # (seq_len, dim)

        if torch.onnx.is_in_onnx_export():
            cos, sin = self._compute_rotates(
                max_seq_len=total_seq_len,
                dim=self.dim,
                device=x.device,
                dtype=x.dtype,
                scale_factor=self.scale_factor,
                interleave=False,
            )
            result = RotaryPositionalEmbeddingFunction.apply(x, cos, sin, position_ids, x.size(1).item(), self.dim)
        else:
            result = RotaryPositionalEmbeddingFunction.forward(
                None, x, self.cos, self.sin, position_ids, x.size(1), self.dim
            )

        # Remove batch dimension if input was 2D
        if input_ndim == 2:
            result = result.squeeze(0)  # [1, seq_len, dim] -> [seq_len, dim]

        return result
