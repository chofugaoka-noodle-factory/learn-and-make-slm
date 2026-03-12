from argparse import ArgumentParser
from pathlib import Path
import torch
import tiktoken

from config import SLMConfig
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint


def load_model(
    config: SLMConfig, checkpoint_path: str, device: torch.device, enable_cache: bool = False
) -> tuple[GPT2Decoder, tiktoken.Encoding]:
    """モデルとトークナイザーをロードする"""
    # トークナイザーの初期化
    tokenizer = tiktoken.get_encoding(config.tokenizer)

    # モデルの初期化
    match config.model.model_type:
        case "gpt-2":
            model = GPT2Decoder(
                tokenizer.n_vocab,
                config.model.n_layers,
                config.model.d_model,
                config.model.n_heads,
                tokenizer.eot_token,
                enable_cache=enable_cache,
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
                enable_cache=enable_cache,
            )
        case _:
            raise ValueError(f"Model type {config.model.model_type} not supported")

    # チェックポイントからモデルをロード
    load_checkpoint(checkpoint_path, model, printf=print)

    # モデルをデバイスに転送し、推論モードに設定
    model.to(device)
    model.eval()

    return model, tokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, default="config/simple_stories.yaml")
    # parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision) for debugging")
    args = parser.parse_args()

    device = "cpu"

    config = SLMConfig.load(args.config)

    model_dir = Path(config.path.log_dir)
    try:
        checkpoint_path = latest_checkpoint_path(model_dir, "checkpoint_*.pth")
    except (IndexError, FileNotFoundError):
        print(f"Error: チェックポイントが見つかりません: {model_dir}")
        print("--checkpoint オプションでチェックポイントファイルを指定してください。")
        return

    model, tokenizer = load_model(config, checkpoint_path, device)

    torch.onnx.export(
        model,
        (torch.randint(0, tokenizer.n_vocab, (1, 10)),),  # 入力テンソル
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            # batch_sizeとsequence_lengthを動的に指定する、これにより任意のサイズの入力に対応できる
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=23,  # ONNXのバージョン、Attentionが有効なバージョンを指定
        dynamo=False,  # dynamoを使うとsymbolicを無視して演算が分解されるので、最適化が意味をなさなくなる
        optimize=False,  # 最適化を無効化する(dynamo=Falseなので意味はない)
        external_data=False,  # 重みを同じモデルに統合する
    )


if __name__ == "__main__":
    main()
