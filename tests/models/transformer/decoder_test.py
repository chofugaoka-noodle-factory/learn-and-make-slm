import torch

from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder


def test_gpt_2_decoder_shape():
    """GPT-2のDecoderの順伝播の出力形状を確認"""
    batch_size = 2
    seq_len = 32
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_vocab = 20
    end_token_id = 19

    x = torch.randint(0, n_vocab, (batch_size, seq_len))

    decoder = GPT2Decoder(n_vocab=n_vocab, n_layers=n_layers, d_model=d_model, n_heads=n_heads, end_token_id=end_token_id)
    output, _, _ = decoder(x, seq_lens=torch.tensor([seq_len, seq_len]))

    assert output.shape == (batch_size, seq_len, n_vocab)


def test_gpt_oss_decoder_shape():
    """GPT-OSSのDecoderの順伝播の出力形状を確認"""
    batch_size = 2
    seq_len = 32
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_groups = 2
    n_vocab = 20
    end_token_id = 19

    x = torch.randint(0, n_vocab, (batch_size, seq_len))

    decoder = GPTOSSDecoder(
        n_vocab=n_vocab, n_layers=n_layers, d_model=d_model, n_heads=n_heads, n_groups=n_groups, end_token_id=end_token_id
    )
    output, _, _ = decoder(x, seq_lens=torch.tensor([seq_len, seq_len]))

    assert output.shape == (batch_size, seq_len, n_vocab)


def test_gpt_2_decoder_output_consistency_with_kv_cache():
    """GPT-2のDecoderがKVキャッシュの有無で同じ出力を生成することを確認"""
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 8
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_vocab = 20
    end_token_id = 19

    x = torch.randint(0, n_vocab, (batch_size, seq_len))
    seq_lens = torch.tensor([seq_len])

    # KVキャッシュなしで全トークンを処理
    decoder_no_cache = GPT2Decoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        end_token_id=end_token_id,
        enable_internal_cache=False,
    )
    output_no_cache, *_ = decoder_no_cache(x, seq_lens=seq_lens)

    # KVキャッシュ有りで段階的に処理
    decoder_with_cache = GPT2Decoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        end_token_id=end_token_id,
        enable_internal_cache=True,
    )
    # 重みをコピー
    decoder_with_cache.load_state_dict(decoder_no_cache.state_dict())

    # キャッシュを有効化
    decoder_with_cache._activate_caches()

    # 段階的に各トークンを処理
    output_list = []
    for i in range(seq_len):
        output, *_ = decoder_with_cache(x[:, i: i + 1])
        output_list.append(output)

    output_with_cache_last = torch.cat(output_list, dim=1)

    # キャッシュを無効化
    decoder_with_cache._invalidate_caches()

    torch.testing.assert_close(output_no_cache, output_with_cache_last)


def test_gpt_oss_decoder_output_consistency_with_kv_cache():
    """GPT-OSSのDecoderがKVキャッシュの有無で同じ出力を生成することを確認"""
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 8
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_groups = 2
    n_vocab = 20
    end_token_id = 19

    x = torch.randint(0, n_vocab, (batch_size, seq_len))
    seq_lens = torch.tensor([seq_len])

    # KVキャッシュなしで全トークンを処理
    decoder_no_cache = GPTOSSDecoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_groups=n_groups,
        end_token_id=end_token_id,
        enable_internal_cache=False,
    )
    output_no_cache, *_ = decoder_no_cache(x, seq_lens=seq_lens)

    # KVキャッシュ有りで段階的に処理
    decoder_with_cache = GPTOSSDecoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_groups=n_groups,
        end_token_id=end_token_id,
        enable_internal_cache=True,
    )
    # 重みをコピー
    decoder_with_cache.load_state_dict(decoder_no_cache.state_dict())

    # キャッシュを有効化
    decoder_with_cache._activate_caches()

    # 段階的に各トークンを処理
    output_list = []
    for i in range(seq_len):
        output, *_ = decoder_with_cache(x[:, i: i + 1])
        output_list.append(output)

    output_with_cache_last = torch.cat(output_list, dim=1)

    # キャッシュを無効化
    decoder_with_cache._invalidate_caches()

    torch.testing.assert_close(output_no_cache, output_with_cache_last)
