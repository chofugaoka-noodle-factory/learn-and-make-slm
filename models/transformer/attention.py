from typing import Final
import warnings
from torch import nn, Tensor
import torch
from models.basic.softmax import Softmax
from models.basic.linear import Linear
from jaxtyping import Float, Bool, Int

from utils.kv_cache import CacheEntry, KVCache
from utils.mask import make_pad_mask
from models.transformer.activation import sigmoid
from models.transformer.rope import RotaryPositionalEmbedding

try:
    import xformers.ops as xops  # type: ignore
    import xformers.ops.fmha as fmha

    _HAS_XFORMERS = True
except Exception:
    xops = None  # type: ignore
    fmha = None  # type: ignore
    _HAS_XFORMERS = False

_INTERNAL_INITIAL_CACHE_INDEX: Final = -1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.softmax = Softmax()
        self.scale = d_k**-0.5
        self.onnx_opset_version = 23

    def xformers_forward(
        self,
        Q: Float[Tensor, "B H S_q D={self.d_k}"],
        K: Float[Tensor, "B H S_k D"],
        V: Float[Tensor, "B H S_k D"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B H S_q D"]:
        """
        xformersの`memory_efficient_attention`を用いてSDPAを計算する。
        おそらく内部でFlash Attention 2を実行していると思われる。
        `memory_efficient_attention`を活用するため、バッチ次元を1次元にし、
        シーケンス次元で結合する"パッキング"という処理を行っている。
        パッキングに合わせて、パッキングされた複数のシーケンスの独立性を保証するために
        `BlockDiagonalMask`を導入している。
        最終的に、入力と同一形状のものが返ってくるようなシェイプ変換も行っている。
        """
        if not _HAS_XFORMERS:
            raise RuntimeError("xFormers is not available")

        B, *_ = Q.shape

        Q: Float[Tensor, "B S H D={self.d_k}"] = Q.transpose(1, 2)
        K: Float[Tensor, "B S H D={self.d_k}"] = K.transpose(1, 2)
        V: Float[Tensor, "B S H D={self.d_k}"] = V.transpose(1, 2)

        # Bは1になる、Sは要素ごとに異なる
        Q_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]
        K_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]
        V_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]

        if seq_lens is not None:
            # unsqueezeが不要なようにインデックスアクセスではなくスライスでアクセスする
            Q_list = [Q[i : i + 1, : seq_lens[i]] for i in range(B)]
            K_list = [K[i : i + 1, : seq_lens[i]] for i in range(B)]
            V_list = [V[i : i + 1, : seq_lens[i]] for i in range(B)]
        else:
            Q_list = [Q[i : i + 1] for i in range(B)]
            K_list = [K[i : i + 1] for i in range(B)]
            V_list = [V[i : i + 1] for i in range(B)]

        # S_total = sum(seq_lens)になる、バッチ分割していたものを一つの系列に結合している
        Q_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        K_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        V_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        # ref: https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask
        # BlockDiagonalMaskはseq_lensが[2, 3, 2]であればこのようなマスクを作成する
        # この方式であればメモリ効率の良いAttentionを計算できる
        # [[   0,    0, -inf, -inf, -inf, -inf, -inf],
        #  [   0,    0, -inf, -inf, -inf, -inf, -inf],
        #  [-inf, -inf,    0,    0,    0, -inf, -inf],
        #  [-inf, -inf,    0,    0,    0, -inf, -inf],
        #  [-inf, -inf,    0,    0,    0, -inf, -inf],
        #  [-inf, -inf, -inf, -inf, -inf,    0,    0],
        #  [-inf, -inf, -inf, -inf, -inf,    0,    0]]
        attn_bias, Q_reshaped, K_reshaped, V_reshaped = fmha.BlockDiagonalMask.from_tensor_lists_qkv(Q_list, K_list, V_list)
        # if not using cache(or first step), make causal mask
        if Q_reshaped.size(1) == K_reshaped.size(1):
            # 学習の際、後ろを見ずにSoftmaxを計算するため、causal mask(LowerTriangularMaskを組み合わせたもの)を作成する
            # 以前のmaskを以下のように変形する
            # [[   0, -inf, -inf, -inf, -inf, -inf, -inf],
            #  [   0,    0, -inf, -inf, -inf, -inf, -inf],
            #  [-inf, -inf,    0, -inf, -inf, -inf, -inf],
            #  [-inf, -inf,    0,    0, -inf, -inf, -inf],
            #  [-inf, -inf,    0,    0,    0, -inf, -inf],
            #  [-inf, -inf, -inf, -inf, -inf,    0  -inf],
            #  [-inf, -inf, -inf, -inf, -inf,    0,    0]]
            attn_bias = attn_bias.make_causal()

        out: Float[Tensor, "1 S_total H D={self.d_k}"] = xops.memory_efficient_attention(
            Q_reshaped, K_reshaped, V_reshaped, attn_bias=attn_bias, scale=float(self.scale)
        )  # type: ignore

        # 系列を結合していたものをQueryの情報に基づいてバッチごとに分割する
        # Queryの系列長がKVのものと異なる事がある(推論時)ため
        list_out: list[Float[Tensor, "1 S_i H D={self.d_k}"]] = attn_bias.split_queries(out)
        # もとの形状に戻す
        # もとの形状に戻す際、後段の計算を安定させるため+reference実装に合わせるため0で初期化された行列を使う
        padded_out: Float[Tensor, "B S H D={self.d_k}"] = torch.zeros_like(Q)
        for i, out_elem in enumerate(list_out):
            if seq_lens is not None:
                padded_out[i][: seq_lens[i]] = out_elem
            else:
                padded_out[i] = out_elem
        padded_out: Float[Tensor, "B H S D={self.d_k}"] = padded_out.transpose(1, 2)
        return padded_out

    def reference_forward(
        self,
        Q: Float[Tensor, "B H S_q D={self.d_k}"],
        K: Float[Tensor, "B H S_k D"],
        V: Float[Tensor, "B H S_k D"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B H S_q D"]:
        scores: Float[Tensor, "B H S_q S_k"] = (Q @ K.transpose(-2, -1)) * self.scale

        # make causal mask with diagonal
        s_q = scores.size(-2)
        s_k = scores.size(-1)
        start = max(0, s_k - s_q)
        causal_mask: Bool[Tensor, "S_q S_k"] = torch.tril(
            torch.ones((s_q, s_k), device=scores.device, dtype=torch.bool),
            diagonal=start,
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if seq_lens is not None:
            # make_pad_maskにmaxlenを明示的に渡す
            # seq_lensは学習に使うシーケンス長を表すので、必ずしも最大長が含まれるとは限らない
            maxlen = Q.size(-2)
            mask: Bool[Tensor, "B 1 1 S_q"] = (
                make_pad_mask(seq_lens, maxlen=maxlen).to(scores.device).unsqueeze(1).unsqueeze(1)
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        return output

    def forward(
        self,
        Q: Float[Tensor, "B H S_q D={self.d_k}"],
        K: Float[Tensor, "B H S_k D"],
        V: Float[Tensor, "B H S_k D"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B H S_q D"]:
        if torch.onnx.is_in_onnx_export():
            if self.onnx_opset_version < 23:
                return self.reference_forward(Q, K, V, seq_lens)
            # Right-align the causal mask so internal-cache decoding also works.
            query_positions = torch.arange(Q.size(-2), device=Q.device) + K.size(-2) - Q.size(-2)
            key_positions = torch.arange(K.size(-2), device=Q.device)
            mask = key_positions <= query_positions.unsqueeze(-1)
            if seq_lens is not None:
                mask = mask & (key_positions < seq_lens[:, None, None])
                mask = mask.unsqueeze(1)
            output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask, scale=self.scale)
            return output

        try:
            return self.xformers_forward(Q, K, V, seq_lens)
        except Exception:
            warnings.warn("xFormers is not available, falling back to reference implementation")
            # 使えない状況（未対応のmaskやCPU/未実装カーネル等）は自前実装にフォールバック
            return self.reference_forward(Q, K, V, seq_lens)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_rope: bool = False,
        rope_scale_factor: float = 1.0,
        use_sigmoid_gate: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model // n_heads)
        self.rope = RotaryPositionalEmbedding(d_model // n_heads, scale_factor=rope_scale_factor) if use_rope else None

        self.use_sigmoid_gate = use_sigmoid_gate

        self._kv_cache = KVCache()
        self._active_internal_cache: int | None = None
        self._current_seq_len: int = 0

    def _internal_activate_cache(self) -> None:
        self._active_internal_cache = _INTERNAL_INITIAL_CACHE_INDEX

    def _internal_invalidate_cache(self) -> None:
        if self._active_internal_cache is not None:
            self._kv_cache.reset(self._active_internal_cache)
            self._active_internal_cache = None
        self._current_seq_len = 0

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S D"]:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # When using cache, we can only process one token at a time.
        use_cache = self._active_internal_cache is not None and self._active_internal_cache != _INTERNAL_INITIAL_CACHE_INDEX
        if use_cache:
            assert seq_len == 1, (
                f"When using cache, seq_len must be 1. Got seq_len={seq_len} with cache index {self._active_internal_cache}."
            )

        Q = (
            self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)
        K = (
            self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)
        V = (
            self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)

        if self.rope is not None:
            positional_offset: int = 0
            if use_cache:
                # when cached, we need to indicate current K/Q position with positional_offset
                positional_offset = self._current_seq_len

            Q = self.rope(Q, positional_offset)
            K = self.rope(K, positional_offset)

        attention: Tensor
        if self._active_internal_cache is not None:
            # cached
            if self._active_internal_cache == _INTERNAL_INITIAL_CACHE_INDEX:
                self._active_internal_cache = self._kv_cache.append(
                    cache_class=CacheEntry,
                    key=K,
                    value=V,
                )
                self._current_seq_len = seq_len
            else:
                K, V = self._kv_cache.update(self._active_internal_cache, K, V)
                self._current_seq_len += seq_len

            attention = self.attention(Q, K, V, seq_lens=None)
        else:
            # non-cached
            attention = self.attention(Q, K, V, seq_lens)

        attention = attention.transpose(1, 2)  # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        if self.use_sigmoid_gate:
            # ref: https://arxiv.org/pdf/2505.06708
            attention = sigmoid(attention)
        output = self.linear_out(attention)  # (batch_size, seq_len, d_model)

        if use_cache:
            # When using cache, we can only process one token at a time.
            assert output.size(1) == 1, (
                f"When using cache, output seq_len must be 1. Got output seq_len={output.size(1)} "
                f"with cache index {self._active_internal_cache}."
            )

        return output


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_groups: int,
        use_rope: bool = False,
        rope_scale_factor: float = 1.0,
        use_sigmoid_gate: bool = False,
    ):
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_per_group = n_heads // n_groups
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model // n_groups)
        self.linear_v = Linear(d_model, d_model // n_groups)
        self.linear_out = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model // n_heads)
        self.rope = RotaryPositionalEmbedding(d_model // n_heads, scale_factor=rope_scale_factor) if use_rope else None

        self.use_sigmoid_gate = use_sigmoid_gate

        self._kv_cache = KVCache()
        self._active_internal_cache: int | None = None
        self._current_seq_len: int = 0

    def _internal_activate_cache(self) -> None:
        self._active_internal_cache = _INTERNAL_INITIAL_CACHE_INDEX

    def _internal_invalidate_cache(self) -> None:
        if self._active_internal_cache is not None:
            self._kv_cache.reset(self._active_internal_cache)
            self._current_seq_len = 0
            self._active_internal_cache = None

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S D"]:
        batch_size, seq_len, _ = x.size()
        # When using cache, we can only process one token at a time.
        use_cache = self._active_internal_cache is not None and self._active_internal_cache != _INTERNAL_INITIAL_CACHE_INDEX
        if use_cache:
            assert seq_len == 1, (
                f"When using cache, seq_len must be 1. Got seq_len={seq_len} with cache index {self._active_internal_cache}."
            )

        Q: Float[Tensor, "B H={self.n_heads} S D"] = (
            self.linear_q(x).reshape(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )

        K: Float[Tensor, "B H_G={self.n_heads}//{self.n_groups} S D"] = (
            self.linear_k(x)
            .reshape(batch_size, seq_len, self.n_heads // self.n_groups, self.d_model // self.n_heads)
            .transpose(1, 2)
        )
        K: Float[Tensor, "B H_G 1 S D"] = K.unsqueeze(2)
        K: Float[Tensor, "B H_G G S D"] = K.expand(-1, -1, self.n_groups, -1, -1)
        K: Float[Tensor, "B H S D"] = K.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads)

        V: Float[Tensor, "B H_G S D"] = (
            self.linear_v(x)
            .reshape(batch_size, seq_len, self.n_heads // self.n_groups, self.d_model // self.n_heads)
            .transpose(1, 2)
        )
        V: Float[Tensor, "B H_G 1 S D"] = V.unsqueeze(2)
        V: Float[Tensor, "B H_G G S D"] = V.expand(-1, -1, self.n_groups, -1, -1)
        V: Float[Tensor, "B H S D"] = V.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads)

        if self.rope is not None:
            positional_offset: int = 0
            if use_cache:
                # when cached, we need to indicate current K/Q position with positional_offset
                positional_offset = self._current_seq_len

            Q = self.rope(Q, positional_offset)
            K = self.rope(K, positional_offset)

        attention: Tensor
        if self._active_internal_cache is not None:
            # cached
            if self._active_internal_cache == _INTERNAL_INITIAL_CACHE_INDEX:
                self._active_internal_cache = self._kv_cache.append(
                    cache_class=CacheEntry,
                    key=K,
                    value=V,
                )
                self._current_seq_len = seq_len
            else:
                K, V = self._kv_cache.update(self._active_internal_cache, K, V)
                self._current_seq_len += seq_len

            attention = self.attention(Q, K, V, seq_lens=None)
        else:
            # non-cached
            attention = self.attention(Q, K, V, seq_lens)

        attention = attention.transpose(1, 2)  # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().reshape(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        if self.use_sigmoid_gate:
            # ref: https://arxiv.org/pdf/2505.06708
            attention = sigmoid(attention)
        output = self.linear_out(attention)  # (batch_size, seq_len, d_model)

        # When using cache, we can only process one token at a time.
        if use_cache:
            assert output.size(1) == 1, (
                f"When using cache, output seq_len must be 1. Got output seq_len={output.size(1)} "
                f"with cache index {self._active_internal_cache}."
            )

        return output
