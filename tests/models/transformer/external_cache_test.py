import torch
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder


def test_gpt2_external_cache():
    """GPT-2 Decoderの外部キャッシュ機能を確認"""
    batch_size = 2
    seq_len = 8
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_vocab = 100
    end_token_id = 99

    decoder = GPT2Decoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        end_token_id=end_token_id,
    )
    decoder.eval()

    # 最初のトークン列を処理（外部キャッシュなし）
    x1 = torch.randint(0, n_vocab, (batch_size, seq_len))
    with torch.no_grad():
        output1, present_keys1, present_values1 = decoder(x1)

    # 外部キャッシュが返されないことを確認（past_keys/past_valuesが与えられていないため）
    assert present_keys1 is None
    assert present_values1 is None

    # 次に、ダミーのKVキャッシュを与えて外部キャッシュモードで実行
    dummy_past_keys = torch.zeros((n_layers, batch_size, n_heads, seq_len, d_model // n_heads))
    dummy_past_values = torch.zeros((n_layers, batch_size, n_heads, seq_len, d_model // n_heads))

    with torch.no_grad():
        output1_cached, present_keys1, present_values1 = decoder(
            x1, past_keys=dummy_past_keys, past_values=dummy_past_values
        )

    # 外部キャッシュが返されることを確認
    assert present_keys1 is not None
    assert present_values1 is not None
    # (n_layers, batch_size, n_heads, seq_len * 2, d_k)
    assert present_keys1.shape == (n_layers, batch_size, n_heads, seq_len * 2, d_model // n_heads)
    assert present_values1.shape == (n_layers, batch_size, n_heads, seq_len * 2, d_model // n_heads)

    # 次のトークンを処理
    x2 = torch.randint(0, n_vocab, (batch_size, 1))
    with torch.no_grad():
        output2, present_keys2, present_values2 = decoder(
            x2,
            past_keys=present_keys1,
            past_values=present_values1,
        )

    # キャッシュが更新されていることを確認
    assert present_keys2 is not None
    assert present_values2 is not None
    # (n_layers, batch_size, n_heads, seq_len * 2 + 1, d_k)
    assert present_keys2.shape == (n_layers, batch_size, n_heads, seq_len * 2 + 1, d_model // n_heads)
    assert present_values2.shape == (n_layers, batch_size, n_heads, seq_len * 2 + 1, d_model // n_heads)


def test_gptoss_external_cache():
    """GPT-OSS Decoderの外部キャッシュ機能を確認"""
    batch_size = 2
    seq_len = 8
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_groups = 2
    n_vocab = 100
    end_token_id = 99

    decoder = GPTOSSDecoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_groups=n_groups,
        end_token_id=end_token_id,
    )
    decoder.eval()

    # 最初のトークン列を処理（外部キャッシュなし）
    x1 = torch.randint(0, n_vocab, (batch_size, seq_len))
    with torch.no_grad():
        output1, present_keys1, present_values1 = decoder(x1)

    # 外部キャッシュが返されないことを確認（past_keys/past_valuesが与えられていないため）
    assert present_keys1 is None
    assert present_values1 is None

    # 次に、ダミーのKVキャッシュを与えて外部キャッシュモードで実行
    dummy_past_keys = torch.zeros((n_layers, batch_size, n_groups, seq_len, d_model // n_heads))
    dummy_past_values = torch.zeros((n_layers, batch_size, n_groups, seq_len, d_model // n_heads))

    with torch.no_grad():
        output1_cached, present_keys1, present_values1 = decoder(
            x1, past_keys=dummy_past_keys, past_values=dummy_past_values
        )

    # 外部キャッシュが返されることを確認
    assert present_keys1 is not None
    assert present_values1 is not None
    # (n_layers, batch_size, n_groups, seq_len * 2, d_k)
    assert present_keys1.shape == (n_layers, batch_size, n_groups, seq_len * 2, d_model // n_heads)
    assert present_values1.shape == (n_layers, batch_size, n_groups, seq_len * 2, d_model // n_heads)

    # 次のトークンを処理
    x2 = torch.randint(0, n_vocab, (batch_size, 1))
    with torch.no_grad():
        output2, present_keys2, present_values2 = decoder(
            x2,
            past_keys=present_keys1,
            past_values=present_values1,
        )

    # キャッシュが更新されていることを確認
    assert present_keys2 is not None
    assert present_values2 is not None
    # (n_layers, batch_size, n_groups, seq_len * 2 + 1, d_k)
    assert present_keys2.shape == (n_layers, batch_size, n_groups, seq_len * 2 + 1, d_model // n_heads)
    assert present_values2.shape == (n_layers, batch_size, n_groups, seq_len * 2 + 1, d_model // n_heads)


def test_gpt2_external_cache_consistency():
    """外部キャッシュを使った場合と使わない場合で出力が一致することを確認"""
    batch_size = 1
    seq_len = 8
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_vocab = 100
    end_token_id = 99

    decoder = GPT2Decoder(
        n_vocab=n_vocab,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        end_token_id=end_token_id,
    )
    decoder.eval()

    # 入力トークン列
    x = torch.randint(0, n_vocab, (batch_size, seq_len + 1))

    with torch.no_grad():
        # 通常の処理（キャッシュなし）
        output_full, _, _ = decoder(x)

        # キャッシュを使った処理（空のKVキャッシュから開始）
        empty_cache_keys = torch.zeros((n_layers, batch_size, n_heads, 0, d_model // n_heads))
        empty_cache_values = torch.zeros((n_layers, batch_size, n_heads, 0, d_model // n_heads))
        output1, present_keys1, present_values1 = decoder(
            x[:, :seq_len], past_keys=empty_cache_keys, past_values=empty_cache_values
        )
        output2, present_keys2, present_values2 = decoder(
            x[:, seq_len : seq_len + 1],
            past_keys=present_keys1,
            past_values=present_values1,
        )

    # 最後のトークンの出力が一致することを確認
    assert torch.allclose(output_full[:, seq_len : seq_len + 1, :], output2, atol=1e-5)
