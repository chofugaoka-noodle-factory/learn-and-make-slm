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
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path")
    parser.add_argument("--enable-kv-cache", action="store_true", help="Enable KV cache for ONNX export")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for ONNX export")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    config = SLMConfig.load(args.config)

    model_dir = Path(config.path.log_dir)
    try:
        checkpoint_path = latest_checkpoint_path(model_dir, "checkpoint_*.pth")
    except (IndexError, FileNotFoundError):
        print(f"Error: チェックポイントが見つかりません: {model_dir}")
        print("--checkpoint オプションでチェックポイントファイルを指定してください。")
        return

    model, tokenizer = load_model(config, checkpoint_path, device)

    if args.enable_kv_cache:
        # KV Cacheをサポートしたモデルのラッパーを作成
        print("Exporting model with KV cache support...")

        # ダミー入力を作成（pre-fill用）
        dummy_input = torch.randint(0, tokenizer.n_vocab, (1, 10)).to(device)

        n_layers = len(model.layers)
        print(f"Model has {n_layers} layers")

        # 出力名とdynamic_axesを設定
        input_names = ["input_ids", "past_keys", "past_values"]
        output_names = ["logits", "present_keys", "present_values"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"},
            # past_keys/values: (n_layers, batch_size, n_heads or n_groups, past_seq_len, d_k)
            "past_keys": {1: "batch_size", 3: "past_sequence_length"},
            "past_values": {1: "batch_size", 3: "past_sequence_length"},
            # present_keys/values: (n_layers, batch_size, n_heads or n_groups, total_seq_len, d_k)
            "present_keys": {1: "batch_size", 3: "total_sequence_length"},
            "present_values": {1: "batch_size", 3: "total_sequence_length"},
        }

        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")

        # ダミーのKVキャッシュを作成
        dummy_past_keys = torch.zeros((n_layers, 1, 4, 10, 64)).to(device)
        dummy_past_values = torch.zeros((n_layers, 1, 4, 10, 64)).to(device)

        torch.onnx.export(
            model,
            (dummy_input, dummy_past_keys, dummy_past_values),
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            # CUDA EPはSqueezeのopset version 23に対応していないため、version 17まで下げる
            opset_version=23 if args.device != "cuda" else 17,
            dynamo=False,
            optimize=False,
            external_data=False,
        )
    else:
        # 通常のエクスポート（KV cacheなし）
        print("Exporting model without KV cache...")

        torch.onnx.export(
            model,
            (torch.randint(0, tokenizer.n_vocab, (1, 10)).to(device),),  # 入力テンソル
            args.output,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                # batch_sizeとsequence_lengthを動的に指定する、これにより任意のサイズの入力に対応できる
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            # CUDA EPはSqueezeのopset version 23に対応していないため、version 17まで下げる
            opset_version=23 if args.device != "cuda" else 17,
            dynamo=False,  # dynamoを使うとsymbolicを無視して演算が分解されるので、最適化が意味をなさなくなる
            optimize=False,  # 最適化を無効化する(dynamo=Falseなので意味はない)
            external_data=False,  # 重みを同じモデルに統合する
        )

    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
