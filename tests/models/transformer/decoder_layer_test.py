import torch
from models.transformer.decoder_layer import GPT2DecoderLayer, GPTOSSDecoderLayer


def test_gpt_2_decoder_layer_shape():
    """GPT-2のDecoderLayerの出力形状を確認"""
    batch_size = 2
    seq_len = 32
    d_model = 64
    n_heads = 4

    x = torch.rand(batch_size, seq_len, d_model)

    decoder_layer = GPT2DecoderLayer(d_model=d_model, n_heads=n_heads)
    output, _, _ = decoder_layer(x, seq_lens=torch.tensor([seq_len, seq_len]))

    assert output.shape == (batch_size, seq_len, d_model)


def test_gpt_oss_decoder_layer_shape():
    """GPT-OSSのDecoderLayerの出力形状を確認"""
    batch_size = 2
    seq_len = 32
    d_model = 64
    n_heads = 4
    n_groups = 2

    x = torch.rand(batch_size, seq_len, d_model)

    decoder_layer = GPTOSSDecoderLayer(d_model=d_model, n_heads=n_heads, n_groups=n_groups)
    output, _, _ = decoder_layer(x, seq_lens=torch.tensor([seq_len, seq_len]))

    assert output.shape == (batch_size, seq_len, d_model)
