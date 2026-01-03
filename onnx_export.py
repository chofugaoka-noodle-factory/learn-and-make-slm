"""自前実装のモデルをDynamo経由でONNXにエクスポートする。"""

from argparse import ArgumentParser
from pathlib import Path

import torch
import tiktoken

from config import SLMConfig
from models.transformer.decoder import DecoderBase, GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint


def get_model(
    config: SLMConfig, enable_internal_cache: bool = False
) -> tuple[GPT2Decoder | GPTOSSDecoder, tiktoken.Encoding]:
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
    config: SLMConfig, checkpoint_path: str, device: torch.device, enable_internal_cache: bool = False
) -> tuple[GPT2Decoder | GPTOSSDecoder, tiktoken.Encoding]:
    """トークナイザーとモデルをロードする"""
    model, tokenizer = get_model(config, enable_internal_cache=enable_internal_cache)
    load_checkpoint(checkpoint_path, model, printf=print)
    model.to(device)
    model.eval()
    return model, tokenizer


def export_model(model: DecoderBase, output_path: Path, opset_version: int = 23) -> None:
    # 古いopsetではAttentionとRoPEを自前のテンソル演算に分解する。
    for module in model.modules():
        if hasattr(module, "onnx_opset_version"):
            module.onnx_opset_version = opset_version

    device = next(model.parameters()).device
    input_ids = torch.randint(0, model.n_vocab, (2, 10), device=device)
    torch.onnx.export(
        model.eval(),
        (input_ids,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="省略時は設定のlog_dirから最新のcheckpointを使用")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = SLMConfig.load(args.config)
    checkpoint_path = args.checkpoint or latest_checkpoint_path(Path(config.path.log_dir), "checkpoint_*.pth")
    if checkpoint_path is None:
        parser.error("No checkpoint found; specify --checkpoint or check path.log_dir")
    device = torch.device("cpu")
    model, _ = get_model_with_checkpoint(config, str(checkpoint_path), device)
    opset_version = 23
    export_model(model, args.output, opset_version)
    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
