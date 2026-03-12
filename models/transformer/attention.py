from typing import Final
import warnings
from torch import nn, Tensor
import torch
from models.basic.softmax import SoftmaxFunction
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


def expand_grouped_kv(
    K: Float[Tensor, "B H_G S_k D"], V: Float[Tensor, "B H_G S_k D"], n_groups: int
) -> tuple[Float[Tensor, "B H S_k D"], Float[Tensor, "B H S_k D"]]:
    batch_size, n_heads, seq_len, d_k = K.size()
    K: Float[Tensor, "B H_G 1 S_k D"] = K.unsqueeze(2)
    K: Float[Tensor, "B H_G G S_k D"] = K.expand(-1, -1, n_groups, -1, -1)
    K: Float[Tensor, "B H S_k D"] = K.reshape(batch_size, n_heads * n_groups, seq_len, d_k)

    V: Float[Tensor, "B H_G 1 S_k D"] = V.unsqueeze(2)
    V: Float[Tensor, "B H_G G S_k D"] = V.expand(-1, -1, n_groups, -1, -1)
    V: Float[Tensor, "B H S_k D"] = V.reshape(batch_size, n_heads * n_groups, seq_len, d_k)
    return K, V


class ScaledDotProductAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "B H S_q D"],
        K: Float[Tensor, "B H_G1 S_k D"],
        V: Float[Tensor, "B H_G1 S_k D"],
        kv_num_heads: int,
        q_num_heads: int,
        scale: float,
        past_key: Float[Tensor, "B H_G2 S_past D"] | None = None,
        past_value: Float[Tensor, "B H_G2 S_past D"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B H S_q D"],
        Float[Tensor, "B H_G2 S_present D_k"] | None,
        Float[Tensor, "B H_G2 S_present D_k"] | None,
    ]:
        # 外部キャッシュを使用するかどうかを自動判定
        use_external_cache = past_key is not None and past_value is not None

        present_key: Float[Tensor, "B H_G2 S_present D"] | None
        present_value: Float[Tensor, "B H_G2 S_present D"] | None
        # KVキャッシュの結合
        if use_external_cache:
            n_groups = q_num_heads // past_key.size(-3)
            if torch.onnx.is_in_onnx_export():
                # ONNXエクスポート時：past_keyは展開しない、Kもまだ展開されていない
                # 両方とも (B, H_G, S, D) の形状なのでそのままcat
                K = torch.cat([past_key, K], dim=-2)
                V = torch.cat([past_value, V], dim=-2)
                # present_keyはグループ形状のまま
                present_key = K
                present_value = V
            else:
                # 通常時：past_keyを展開してからcat
                past_key, past_value = expand_grouped_kv(past_key, past_value, n_groups)
                K = torch.cat([past_key, K], dim=-2)
                V = torch.cat([past_value, V], dim=-2)
                # 展開された形状からグループ形状に戻す
                present_key = K[:, ::n_groups]
                present_value = V[:, ::n_groups]
        else:
            # ONNXエクスポート時はNoneを返せないので、ダミーテンソルを返す
            if torch.onnx.is_in_onnx_export():
                # 空のテンソルを返す（シーケンス長0）
                present_key = torch.empty(K.size(0), kv_num_heads, 0, K.size(-1), dtype=K.dtype, device=K.device)
                present_value = torch.empty(V.size(0), kv_num_heads, 0, V.size(-1), dtype=V.dtype, device=V.device)
            else:
                present_key = None
                present_value = None

        # ONNX export時はGrouped Query Attentionのために手動でexpandする
        if torch.onnx.is_in_onnx_export():
            K, V = expand_grouped_kv(K, V, q_num_heads // kv_num_heads)

        scores: Float[Tensor, "B H S_q S_k"] = (Q @ K.transpose(-2, -1)) * scale

        # make causal mask with diagonal
        s_q = scores.size(-2)
        s_k = scores.size(-1)
        start = s_k - s_q
        causal_mask: Bool[Tensor, "S_q S_k"] = torch.tril(
            torch.ones((s_q, s_k), device=scores.device, dtype=torch.bool),
            diagonal=start,
        )
        # past keyが存在する場合、最初のシーケンスをマスクする
        if torch.onnx.is_in_onnx_export() and present_key is not None:
            causal_mask[:, 0] = 0
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if seq_lens is not None:
            # make_pad_maskにmaxlenを明示的に渡す
            # seq_lensは学習に使うシーケンス長を表すので、必ずしも最大長が含まれるとは限らない
            maxlen = Q.size(-2)
            mask: Bool[Tensor, "B 1 1 S_q"] = (
                make_pad_mask(seq_lens, maxlen=maxlen).to(scores.device).unsqueeze(1).unsqueeze(1)
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = SoftmaxFunction.forward(None, scores)
        output = attn_weights @ V
        return output, present_key, present_value

    @staticmethod
    def symbolic(
        g,
        Q: Float[Tensor, "B H S_q D"],
        K: Float[Tensor, "B H_G1 S_k D"],
        V: Float[Tensor, "B H_G1 S_k D"],
        kv_num_heads: int,
        q_num_heads: int,
        scale: float,
        past_key: Float[Tensor, "B H_G2 S_past D"] | None = None,
        past_value: Float[Tensor, "B H_G2 S_past D"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ):
        if past_key is not None and past_value is not None:
            # それぞれの次元を取り出す
            shape_q = g.op("Shape", Q)
            shape_past_k = g.op("Shape", past_key)
            shape_k = g.op("Shape", K)

            B = g.op(
                "Slice", shape_q, g.op("Constant", value_t=torch.tensor([0])), g.op("Constant", value_t=torch.tensor([1]))
            )
            H_k = g.op(
                "Slice", shape_k, g.op("Constant", value_t=torch.tensor([1])), g.op("Constant", value_t=torch.tensor([2]))
            )
            S_q = g.op(
                "Slice", shape_q, g.op("Constant", value_t=torch.tensor([2])), g.op("Constant", value_t=torch.tensor([3]))
            )
            S_k = g.op(
                "Slice", shape_k, g.op("Constant", value_t=torch.tensor([2])), g.op("Constant", value_t=torch.tensor([3]))
            )
            S_past_k = g.op(
                "Slice",
                shape_past_k,
                g.op("Constant", value_t=torch.tensor([2])),
                g.op("Constant", value_t=torch.tensor([3])),
            )

            one_scalar = g.op(
                "Constant",
                value_t=torch.tensor(1, dtype=torch.int64),
            )

            # S_past_k + S_k - 1
            S_total_k = g.op("Add", S_past_k, S_k)
            S_total_k = g.op("Sub", S_total_k, one_scalar)

            # Maskが必要なので、ONNX内部で計算する
            one_tensor = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
            mask_pad_shape = g.op("Concat", B, H_k, S_q, one_tensor, axis_i=0)
            mask_shape = g.op("Concat", B, H_k, S_q, S_total_k, axis_i=0)

            # ONNXモデルはその性質上、past key及びpast valueのシーケンス長を0にできない。
            # 推論時はseq_len=1のpast key及びpast valueを与え、そこに-infのマスクを足して計算する。
            minus_inf_scalar = g.op(
                "Constant",
                value_t=torch.tensor(float("-inf"), dtype=torch.float32),
            )
            zeros_scalar = g.op(
                "Constant",
                value_t=torch.tensor(0.0, dtype=torch.float32),
            )

            # マスクをExpandして結合
            attn_mask_pad = g.op("Expand", minus_inf_scalar, mask_pad_shape)
            attn_mask = g.op("Expand", zeros_scalar, mask_shape)
            attn_mask = g.op("Concat", attn_mask_pad, attn_mask, axis_i=3)

            output, present_key, present_value = g.op(
                "Attention",
                Q,
                K,
                V,
                attn_mask,
                past_key,
                past_value,
                is_causal_i=1,
                kv_num_heads_i=kv_num_heads,
                q_num_heads_i=q_num_heads,
                scale_f=scale,
                outputs=3,
            )
        else:
            output = g.op(
                "Attention",
                Q,
                K,
                V,
                is_causal_i=1,
                kv_num_heads_i=kv_num_heads,
                q_num_heads_i=q_num_heads,
                scale_f=scale,
            )
            present_key, present_value = None, None
        return output, present_key, present_value


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def xformers_forward(
        self,
        Q: Float[Tensor, "B H S_q D={self.d_k}"],
        K: Float[Tensor, "B H S_k D"],
        V: Float[Tensor, "B H S_k D"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
        past_key: Float[Tensor, "B H_G S_past D"] | None = None,
        past_value: Float[Tensor, "B H_G S_past D"] | None = None,
    ) -> tuple[
        Float[Tensor, "B H S_q D"], Float[Tensor, "B H_G S_present D"] | None, Float[Tensor, "B H_G S_present D"] | None
    ]:
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

        use_external_cache = past_key is not None and past_value is not None

        present_key: Float[Tensor, "B H_G (S_past + S_k) D"] | None = None
        present_value: Float[Tensor, "B H_G (S_past + S_k) D"] | None = None

        if use_external_cache:
            past_key, past_value = expand_grouped_kv(past_key, past_value, Q.size(-3) // K.size(-3))

            present_key = torch.cat([past_key, K], dim=-2)
            present_value = torch.cat([past_value, V], dim=-2)
            K = present_key
            V = present_value

        Q: Float[Tensor, "B S_q H D={self.d_k}"] = Q.transpose(1, 2)
        K: Float[Tensor, "B S_k H D={self.d_k}"] = K.transpose(1, 2)
        V: Float[Tensor, "B S_k H D={self.d_k}"] = V.transpose(1, 2)

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
        return padded_out, present_key, present_value

    def forward(
        self,
        Q: Float[Tensor, "B H S_q D={self.d_k}"],
        K: Float[Tensor, "B H_G1 S_k D"],
        V: Float[Tensor, "B H_G1 S_k D"],
        past_key: Float[Tensor, "B H_G2 S_past D"] | None = None,
        past_value: Float[Tensor, "B H_G2 S_past D"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B H S_q D={self.d_k}"],
        Float[Tensor, "B H_G2 S_present D"] | None,
        Float[Tensor, "B H_G2 S_present D"] | None,
    ]:
        if torch.onnx.is_in_onnx_export():
            if Q.device.type != "cuda":
                output, present_key, present_value = ScaledDotProductAttentionFunction.apply(
                    Q, K, V, int(K.size(-3)), int(Q.size(-3)), self.scale.item(), past_key, past_value, seq_lens
                )
            else:
                # CUDA EP向けにexportする場合はAttention Opのないversion 17でexportされるのでforwardをトレースさせる
                output, present_key, present_value = ScaledDotProductAttentionFunction.forward(
                    None, Q, K, V, int(K.size(-3)), int(Q.size(-3)), self.scale.item(), past_key, past_value, seq_lens
                )

            return output, present_key, present_value

        try:
            output, present_key, present_value = self.xformers_forward(Q, K, V, seq_lens, past_key, past_value)
        except Exception:
            warnings.warn("xFormers is not available, falling back to reference implementation")
            # 使えない状況（未対応のmaskやCPU/未実装カーネル等）は自前実装にフォールバック
            output, present_key, present_value = ScaledDotProductAttentionFunction.forward(
                None,
                Q,
                K,
                V,
                kv_num_heads=int(K.size(-3)),
                q_num_heads=int(Q.size(-3)),
                scale=self.scale.item(),
                past_key=past_key,
                past_value=past_value,
                seq_lens=seq_lens,
            )

        return output, present_key, present_value


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
        past_key: Float[Tensor, "B H S_past D_k"] | None = None,
        past_value: Float[Tensor, "B H S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[Float[Tensor, "B S D"], Float[Tensor, "B H S_present D_k"] | None, Float[Tensor, "B H S_present D_k"] | None]:
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, d_model)
            seq_lens: シーケンス長 (batch_size,)
            past_key: 過去のKeyキャッシュ (batch_size, n_heads, past_seq_len, d_k)
            past_value: 過去のValueキャッシュ (batch_size, n_heads, past_seq_len, d_k)

        Returns:
            output: 出力テンソル (batch_size, seq_len, d_model)
            present_key: 現在のKeyキャッシュ (batch_size, n_heads, total_seq_len, d_k) または None
            present_value: 現在のValueキャッシュ (batch_size, n_heads, total_seq_len, d_k) または None
        """
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # 外部キャッシュを使用するかどうかを自動判定
        use_external_cache = past_key is not None and past_value is not None

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
            elif use_external_cache:
                # 外部キャッシュを使う場合、past_keyのシーケンス長をオフセットにする
                positional_offset = past_key.size(-2)

            Q = self.rope(Q, positional_offset)
            K = self.rope(K, positional_offset)

        attention: Tensor
        present_key: Float[Tensor, "B H S_present D_k"] | None
        present_value: Float[Tensor, "B H S_present D_k"] | None

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

            attention, present_key, present_value = self.attention(Q, K, V, seq_lens=None)
        else:
            # non-cached（外部キャッシュを含む）
            attention, present_key, present_value = self.attention(Q, K, V, past_key, past_value, seq_lens)

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

        return output, present_key, present_value


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
        past_key: Float[Tensor, "B H_G S_past D_k"] | None = None,
        past_value: Float[Tensor, "B H_G S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B S D"], Float[Tensor, "B H_G S_present D_k"] | None, Float[Tensor, "B H_G S_present D_k"] | None
    ]:
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, d_model)
            seq_lens: シーケンス長 (batch_size,)
            past_key: 過去のKeyキャッシュ (batch_size, n_groups, past_seq_len, d_k)
            past_value: 過去のValueキャッシュ (batch_size, n_groups, past_seq_len, d_k)

        Returns:
            output: 出力テンソル (batch_size, seq_len, d_model)
            present_key: 現在のKeyキャッシュ (batch_size, n_groups, total_seq_len, d_k) または None
            present_value: 現在のValueキャッシュ (batch_size, n_groups, total_seq_len, d_k) または None
        """
        batch_size, seq_len, _ = x.size()

        # 外部キャッシュを使用するかどうかを自動判定
        use_external_cache = past_key is not None and past_value is not None

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

        V: Float[Tensor, "B H_G S D"] = (
            self.linear_v(x)
            .reshape(batch_size, seq_len, self.n_heads // self.n_groups, self.d_model // self.n_heads)
            .transpose(1, 2)
        )

        if self.rope is not None:
            positional_offset: int = 0
            if use_cache:
                # when cached, we need to indicate current K/Q position with positional_offset
                positional_offset = self._current_seq_len
            elif use_external_cache:
                # 外部キャッシュを使う場合、past_keyのシーケンス長をオフセットにする
                positional_offset = past_key.size(-2)

            Q = self.rope(Q, positional_offset)
            K = self.rope(K, positional_offset)

        # ONNX Export時はAttention Opが自動でExpandしてくれる
        if not torch.onnx.is_in_onnx_export():
            K, V = expand_grouped_kv(K, V, self.n_groups)

        attention: Tensor
        present_key: Float[Tensor, "B H_G S_present D_k"] | None
        present_value: Float[Tensor, "B H_G S_present D_k"] | None

        if self._active_internal_cache is not None:
            # cached（内部キャッシュ使用時）
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

            attention, present_key, present_value = self.attention(Q, K, V, seq_lens=None)
        else:
            # non-cached（外部キャッシュを含む）
            attention, present_key, present_value = self.attention(Q, K, V, past_key, past_value, seq_lens)

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

        return output, present_key, present_value
