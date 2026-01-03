from typing import Final
import sys
import torch
import tiktoken

from torch import nn, Tensor
from models.basic.layer_norm import LayerNorm
from models.transformer.attention import GroupedQueryAttention, MultiHeadAttention
from models.transformer.decoder_layer import GPT2DecoderLayer, GPTOSSDecoderLayer
from models.basic.linear import Linear
from models.basic.softmax import Softmax
from jaxtyping import Float, Int, Bool
from models.basic.embedding import Embedding
from models.transformer.positional_encoding import PositionalEncoding
from utils.mask import make_non_pad_mask


CACHEABLE_MODULES: Final = (MultiHeadAttention, GroupedQueryAttention)


class DecoderBase(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, end_token_id: int, *, enable_internal_cache: bool):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.end_token_id = end_token_id
        self.enable_internal_cache = enable_internal_cache
        self._current_seq_len: int = 0

        self.softmax = Softmax()

    def _activate_caches(self) -> None:
        self._current_seq_len = 0
        for module in self.modules():
            if isinstance(module, CACHEABLE_MODULES):
                module._internal_activate_cache()

    def _invalidate_caches(self) -> None:
        self._current_seq_len = 0
        for module in self.modules():
            if isinstance(module, CACHEABLE_MODULES):
                module._internal_invalidate_cache()

    def forward(
        self,
        x: Int[Tensor, "B S"],
        past_keys: Float[Tensor, "L B H S_past D_k"] | None = None,
        past_values: Float[Tensor, "L B H S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B S V={self.n_vocab}"], Float[Tensor, "L B H S D_k"] | None, Float[Tensor, "L B H S D_k"] | None
    ]:
        """
        Args:
            x: 入力トークンID (batch_size, seq_len)
            seq_lens: シーケンス長 (batch_size,)
            past_keys: 過去のKeyキャッシュ (n_layers, batch_size, n_heads, past_seq_len, d_k)
            past_values: 過去のValueキャッシュ (n_layers, batch_size, n_heads, past_seq_len, d_k)
            use_external_cache: 外部キャッシュを使用するかどうか

        Returns:
            logits: 出力ロジット (batch_size, seq_len, n_vocab)
            present_keys: 現在のKeyキャッシュ (n_layers, batch_size, n_heads, total_seq_len, d_k) または None
            present_values: 現在のValueキャッシュ (n_layers, batch_size, n_heads, total_seq_len, d_k) または None
        """
        raise NotImplementedError()

    @torch.no_grad()
    def infer(
        self,
        starts: Int[Tensor, "1 S_in"],
        max_token_count: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> Int[Tensor, "1 S_out"]:
        try:
            if self.enable_internal_cache:
                self._activate_caches()

            assert starts.size(0) == 1, "starts must have batch size 1"
            x: Tensor = starts
            count: int = 0

            if self.enable_internal_cache and x.size(1) > 1:
                # キャッシュを使う場合は、一番最後のトークンを除いてキャッシュを準備する
                _, _, _ = self(x.detach().clone()[:, :-1])

            if tokenizer is not None:
                decoded = tokenizer.decode(starts[0].tolist())
                print("".join(decoded), end="", flush=True)

            def loop_condition(count: int) -> bool:
                if max_token_count is not None:
                    return count < max_token_count
                return True

            while loop_condition(count=count):
                if self.enable_internal_cache:
                    # キャッシュを使う場合は、一番最後のトークンのみ渡す
                    output, _, _ = self(x.detach().clone()[:, -1:])
                else:
                    output, _, _ = self(x.detach().clone())
                if temperature == 0.0:
                    # argmax: Greedy Decodingによる最も確率の高いトークンを選択
                    next_token = output.argmax(dim=-1)[:, -1:]
                else:
                    output_prob = self.softmax(output / temperature)
                    # 最後の位置の確率分布からサンプリング
                    next_token_probs = output_prob[:, -1, :]  # [B, V]
                    next_token_indices: Tensor | None = None
                    if top_k is not None:
                        next_token_probs, next_token_indices = torch.topk(next_token_probs, top_k, dim=-1)
                        next_token_probs = next_token_probs / next_token_probs.sum(dim=-1, keepdim=True)  # 再正規化
                    # torch.multinomialでカテゴリカル分布からサンプリング
                    next_token = torch.multinomial(next_token_probs, num_samples=1)  # [B, 1]
                    if next_token_indices is not None:
                        next_token = torch.gather(next_token_indices, dim=-1, index=next_token)

                x = torch.cat([x, next_token], dim=-1)
                count += 1
                if next_token[0, 0] == self.end_token_id:
                    break
                if tokenizer is not None:
                    next_token = tokenizer.decode_tokens_bytes([next_token.item()])
                    for byte in next_token:
                        sys.stdout.buffer.write(byte)
                    sys.stdout.flush()
            return x
        finally:
            if self.enable_internal_cache:
                self._invalidate_caches()

    def loss(
        self,
        pred_y: Float[Tensor, "B S V={self.n_vocab}"],
        target_y_index: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "1"]:
        pred_y: Float[Tensor, "B V S"] = pred_y.transpose(-1, -2)

        # cross_entropyはreduction='none'で各位置のlossを計算し、
        # 後でmaskを適用して平均を取る
        loss: Float[Tensor, "B S"] = nn.functional.cross_entropy(pred_y, target_y_index, reduction="none")

        if seq_lens is not None:
            non_pad_mask: Bool[Tensor, "B S"] = make_non_pad_mask(seq_lens, maxlen=loss.size(-1)).to(loss.device)
            # パディング位置のlossを0にする
            loss = loss * non_pad_mask
            # 有効な位置の平均を取る（division by zeroを防ぐため最小値1を保証）
            loss = loss.sum() / torch.clamp(non_pad_mask.sum(), min=1.0)
        else:
            loss = loss.mean()

        return loss


class GPT2Decoder(DecoderBase):
    """
    概ねGPT-2なTransformerのDecoder
    Decoder Layerの実装に一部差異がある
    """

    def __init__(
        self,
        n_vocab: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        end_token_id: int,
        use_sigmoid_gate: bool = False,
        *,
        enable_internal_cache: bool = False,
    ):
        super().__init__(n_vocab, d_model, end_token_id, enable_internal_cache=enable_internal_cache)

        self.embedding = Embedding(n_vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [GPT2DecoderLayer(d_model, n_heads, use_sigmoid_gate=use_sigmoid_gate) for _ in range(n_layers)]
        )
        self.norm = LayerNorm([d_model])
        self.linear_out = Linear(d_model, n_vocab)

    def forward(
        self,
        x: Int[Tensor, "B S"],
        past_keys: Float[Tensor, "L B H S_past D_k"] | None = None,
        past_values: Float[Tensor, "L B H S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B S V={self.n_vocab}"],
        Float[Tensor, "L B H S_present D_k"] | None,
        Float[Tensor, "L B H S_present D_k"] | None,
    ]:
        """
        Args:
            x: 入力トークンID (batch_size, seq_len)
            seq_lens: シーケンス長 (batch_size,)
            past_keys: 過去のKeyキャッシュ (n_layers, batch_size, n_heads, past_seq_len, d_k)
            past_values: 過去のValueキャッシュ (n_layers, batch_size, n_heads, past_seq_len, d_k)

        Returns:
            logits: 出力ロジット (batch_size, seq_len, n_vocab)
            present_keys: 現在のKeyキャッシュ (n_layers, batch_size, n_heads, total_seq_len, d_k) または None
            present_values: 現在のValueキャッシュ (n_layers, batch_size, n_heads, total_seq_len, d_k) または None
        """
        seq_len = x.size(1)

        # When using cache with already cached tokens, ensure we only process one token at a time
        if self.enable_internal_cache and self._current_seq_len > 0:
            assert seq_len == 1, "When using cache with existing cached tokens, x must have sequence length 1"

        # 外部キャッシュを使用するかどうかを自動判定
        use_external_cache = past_keys is not None and past_values is not None

        positional_offset = self._current_seq_len
        if use_external_cache:
            # 外部キャッシュを使う場合
            positional_offset = past_keys[0].size(-2)

        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)
        x = self.positional_encoding(x, positional_offset)

        present_keys: list[Float[Tensor, "B H S_total D_k"]] = []
        present_values: list[Float[Tensor, "B H S_total D_k"]] = []

        for i, layer in enumerate(self.layers):
            past_key = past_keys[i] if past_keys is not None else None
            past_value = past_values[i] if past_values is not None else None
            x, present_key, present_value = layer(x, past_key, past_value, seq_lens)
            if use_external_cache:
                present_keys.append(present_key)
                present_values.append(present_value)

        # Update sequence length after processing
        if self.enable_internal_cache:
            self._current_seq_len += seq_len

        x = self.norm(x)
        y: Float[Tensor, "B S V"] = self.linear_out(x)

        # present_keysとpresent_valuesをstackして単一のtensorにする
        present_keys_tensor: Float[Tensor, "L B H S_present D_k"] | None = torch.stack(present_keys, dim=0) if use_external_cache and len(present_keys) > 0 else None
        present_values_tensor: Float[Tensor, "L B H S_present D_k"] | None = (
            torch.stack(present_values, dim=0) if use_external_cache and len(present_values) > 0 else None
        )

        return y, present_keys_tensor, present_values_tensor


class GPTOSSDecoder(DecoderBase):
    """
    概ねGPT-OSSなTransformerのDecoder
    Decoder Layerの実装に一部差異がある
    """

    def __init__(
        self,
        n_vocab: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_groups: int,
        end_token_id: int,
        rope_scale_factor: float = 1.0,
        use_sigmoid_gate: bool = False,
        *,
        enable_internal_cache: bool = False,
    ):
        super().__init__(n_vocab, d_model, end_token_id, enable_internal_cache=enable_internal_cache)

        self.embedding = Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList(
            [
                GPTOSSDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_groups=n_groups,
                    rope_scale_factor=rope_scale_factor,
                    use_sigmoid_gate=use_sigmoid_gate,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear_out = Linear(d_model, n_vocab)

    def forward(
        self,
        x: Int[Tensor, "B S"],
        past_keys: Float[Tensor, "L B H_G S_past D_k"] | None = None,
        past_values: Float[Tensor, "L B H_G S_past D_k"] | None = None,
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "B S V={self.n_vocab}"],
        Float[Tensor, "L B H_G S_present D_k"] | None,
        Float[Tensor, "L B H_G S_present D_k"] | None,
    ]:
        """
        Args:
            x: 入力トークンID (batch_size, seq_len)
            seq_lens: シーケンス長 (batch_size,)
            past_keys: 過去のKeyキャッシュ (n_layers, batch_size, n_groups, past_seq_len, d_k)
            past_values: 過去のValueキャッシュ (n_layers, batch_size, n_groups, past_seq_len, d_k)
            use_external_cache: 外部キャッシュを使用するかどうか

        Returns:
            logits: 出力ロジット (batch_size, seq_len, n_vocab)
            present_keys: 現在のKeyキャッシュ (n_layers, batch_size, n_groups, total_seq_len, d_k) または None
            present_values: 現在のValueキャッシュ (n_layers, batch_size, n_groups, total_seq_len, d_k) または None
        """
        seq_len = x.size(1)

        # When using cache with already cached tokens, ensure we only process one token at a time
        if self.enable_internal_cache and self._current_seq_len > 0:
            assert seq_len == 1, "When using cache with existing cached tokens, x must have sequence length 1"

        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)

        # 外部キャッシュを使用するかどうかを自動判定
        use_external_cache = past_keys is not None and past_values is not None

        present_keys: list[Float[Tensor, "B H_G S_total D_k"]] = []
        present_values: list[Float[Tensor, "B H_G S_total D_k"]] = []

        for i, layer in enumerate(self.layers):
            past_key = past_keys[i] if past_keys is not None else None
            past_value = past_values[i] if past_values is not None else None
            x, present_key, present_value = layer(x, past_key, past_value, seq_lens)
            if use_external_cache:
                present_keys.append(present_key)
                present_values.append(present_value)

        # Update sequence length after processing
        if self.enable_internal_cache:
            self._current_seq_len += seq_len

        output = self.linear_out(x)

        # present_keysとpresent_valuesをstackして単一のtensorにする
        if use_external_cache and len(present_keys) > 0:
            present_keys_tensor = torch.stack(present_keys, dim=0)
            present_values_tensor = torch.stack(present_values, dim=0)
        elif torch.onnx.is_in_onnx_export():
            # ONNXエクスポート時はNoneを返せないので、空のテンソルを返す
            # (n_layers, batch_size, n_groups, 0, d_k) の形状
            n_layers = len(self.layers)
            batch_size = x.size(0)
            # 最初のレイヤーのself_attnから情報を取得
            first_layer = self.layers[0]
            n_groups = first_layer.self_attn.n_heads // first_layer.self_attn.n_groups
            d_k = first_layer.self_attn.d_model // first_layer.self_attn.n_heads
            present_keys_tensor = torch.empty(n_layers, batch_size, n_groups, 0, d_k, dtype=x.dtype, device=x.device)
            present_values_tensor = torch.empty(n_layers, batch_size, n_groups, 0, d_k, dtype=x.dtype, device=x.device)
        else:
            present_keys_tensor = None
            present_values_tensor = None

        return output, present_keys_tensor, present_values_tensor
