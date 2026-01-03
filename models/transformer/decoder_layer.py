from torch import nn, Tensor
from models.transformer.attention import GroupedQueryAttention, MultiHeadAttention
from models.transformer.feed_forward import FeedForwardSwiGLU, FeedForwardSwish
from models.basic.layer_norm import LayerNorm
from jaxtyping import Float, Int


class GPT2DecoderLayer(nn.Module):
    """
    概ねGPT-2のDecoder Layerを再現している
    異なるのはFeed Forward BlockがGELUではなくSwishを使っている点
    """

    def __init__(self, d_model: int, n_heads: int, use_sigmoid_gate: bool = False):
        super().__init__()
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, use_sigmoid_gate=use_sigmoid_gate)
        self.norm1 = LayerNorm([d_model])
        self.ffn = FeedForwardSwish(d_model)
        self.norm2 = LayerNorm([d_model])

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        past_key: Float[Tensor, "B H S_past D_k"] | None = None,
        past_value: Float[Tensor, "B H S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[Float[Tensor, "B S D"], Float[Tensor, "B H S_present D_k"] | None, Float[Tensor, "B H S_present D_k"] | None]:
        residual = x
        x = self.norm1(x)
        attn_output, present_key, present_value = self.self_attn(x, past_key, past_value, seq_lens)
        x = residual + attn_output

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x, present_key, present_value


class GPTOSSDecoderLayer(nn.Module):
    """
    概ねGPT-OSSのDecoder Layerを再現している
    異なるのは、RMSNormではなくLayerNormを使っている点
    また、Mixture of Expertsを使っていない点
    """

    def __init__(
        self, d_model: int, n_heads: int, n_groups: int, rope_scale_factor: float = 1.0, use_sigmoid_gate: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        # masked multi-head self-attention(grouped query attention)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_groups=n_groups,
            use_rope=True,
            rope_scale_factor=rope_scale_factor,
            use_sigmoid_gate=use_sigmoid_gate,
        )
        self.norm1 = LayerNorm(d_model)

        # feed-forward
        self.ffn = FeedForwardSwiGLU(d_model=d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        past_key: Float[Tensor, "B H_G S_past D_k"] | None = None,
        past_value: Float[Tensor, "B H_G S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B S D"], Float[Tensor, "B H_G S_present D_k"] | None, Float[Tensor, "B H_G S_present D_k"] | None
    ]:
        # masked multi-head self-attention
        residual = x
        x = self.norm1(x)
        attn_output, present_key, present_value = self.self_attn(x, past_key, past_value, seq_lens)
        x = residual + attn_output

        # feed-forward
        residual = x
        x = self.norm2(x)
        ffn_output: Float[Tensor, "B S D"] = self.ffn(x)
        x = residual + ffn_output

        return x, present_key, present_value
