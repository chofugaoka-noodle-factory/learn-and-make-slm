from config import SLMConfig
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import load_checkpoint

import tiktoken
import torch


def get_model(config: SLMConfig, enable_internal_cache: bool = False) -> tuple[GPT2Decoder | GPTOSSDecoder, tiktoken.Encoding]:
    """トークナイザーとモデルを初期化する"""

    # トークナイザーを初期化
    tokenizer = tiktoken.get_encoding(config.tokenizer)

    # モデルを初期化
    model: GPT2Decoder | GPTOSSDecoder
    match config.model.model_type:
        case "gpt-2":
            model = GPT2Decoder(
                tokenizer.n_vocab,
                config.model.n_layers,
                config.model.d_model,
                config.model.n_heads,
                tokenizer.eot_token,
                use_sigmoid_gate=config.model.use_sigmoid_gate,
                enable_internal_cache=enable_internal_cache,
            )
        case "gpt-oss":
            assert config.model.n_groups is not None, "n_groups must be provided for GPT-OSS"
            model = GPTOSSDecoder(
                tokenizer.n_vocab,
                config.model.n_layers,
                config.model.d_model,
                config.model.n_heads,
                config.model.n_groups,
                tokenizer.eot_token,
                rope_scale_factor=config.model.rope_scale_factor,
                use_sigmoid_gate=config.model.use_sigmoid_gate,
                enable_internal_cache=enable_internal_cache,
            )
        case _:
            raise ValueError(f"Model type {config.model.model_type} not supported")

    return model, tokenizer


def get_model_with_checkpoint(
    config: SLMConfig, checkpoint_path: str | None, device: torch.device, enable_internal_cache: bool = False
) -> tuple[GPT2Decoder, tiktoken.Encoding]:
    """トークナイザーとモデルをロードする"""
    model, tokenizer = get_model(config, enable_internal_cache=enable_internal_cache)
    load_checkpoint(checkpoint_path, model, printf=print)
    model.to(device)
    model.eval()
    return model, tokenizer
