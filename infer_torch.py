"""
PyTorch推論スクリプト: 学習済みチェックポイントを使用してテキスト生成を行う
"""

import warnings

warnings.simplefilter("ignore")

from argparse import ArgumentParser
from pathlib import Path
import time

import tiktoken
import torch

from config import SLMConfig
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import latest_checkpoint_path
from utils.model import get_model_with_checkpoint


def generate(
    model: GPT2Decoder,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    top_k: int | None = None,
    show_streaming: bool = True,
) -> tuple[str, int, float]:
    """プロンプトからテキストを生成する"""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            start_time = time.time()
            output_ids = model.infer(
                starts=input_tensor,
                max_token_count=max_tokens,
                temperature=temperature,
                top_k=top_k,
                tokenizer=tokenizer if show_streaming else None,
            )
            end_time = time.time()
            token_num = output_ids.size(1) - input_tensor.size(1)

    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text, token_num, end_time - start_time


def interactive_mode(
    model: GPT2Decoder,
    tokenizer: tiktoken.Encoding,
    max_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> None:
    """インタラクティブモードでテキスト生成を行う"""
    print("\n=== Interactive Mode ===")
    print("プロンプトを入力してください。終了するには 'quit' または 'exit' と入力してください。")
    print("空行で入力を確定します。複数行入力するには各行の末尾に \\ を付けてください。")
    print(f"設定: temperature={temperature}, top_k={top_k}")
    print("=" * 50)

    while True:
        try:
            print("\nPrompt: ", end="")
            lines: list[str] = []
            while True:
                line = input()
                if line.endswith("\\"):
                    lines.append(line[:-1])
                else:
                    lines.append(line)
                    break
            prompt = "\n".join(lines)

            if prompt.lower() in ["quit", "exit"]:
                print("終了します。")
                break

            if not prompt.strip():
                print("プロンプトが空です。もう一度入力してください。")
                continue

            print("\n--- Generated Text ---")
            _, token_num, generation_time = generate(
                model,
                tokenizer,
                prompt,
                max_tokens,
                device,
                temperature=temperature,
                top_k=top_k,
                show_streaming=True,
            )
            print("\n--- End ---")
            print(f"Time: {generation_time:.2f} seconds")
            print(f"Token num: {token_num}")
            if generation_time > 0:
                print(f"Token per time: {token_num / generation_time:.2f} tok/s")

        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except EOFError:
            print("\n終了します。")
            break


def main() -> None:
    parser = ArgumentParser(description="学習済みPyTorchモデルを使用してテキスト生成を行う")
    parser.add_argument("config", type=Path, help="設定ファイル（YAML）のパス")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="チェックポイントファイルのパス。指定しない場合は最新のチェックポイントを使用",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="生成するプロンプト。指定しない場合はインタラクティブモードで起動",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="生成する最大トークン数（デフォルト: 1024）",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="ストリーミング出力を無効にする",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="サンプリングの温度（0.0=Greedy、高いほどランダム、デフォルト: 0.0）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-Kサンプリング（上位K個のトークンからサンプリング）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="使用するデバイス。指定しない場合は自動で選択",
    )
    parser.add_argument(
        "--enable-internal-cache",
        action="store_true",
        help="KVキャッシュを有効にする",
    )
    args = parser.parse_args()

    config = SLMConfig.load(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        model_dir = Path(config.path.log_dir)
        try:
            checkpoint_path = latest_checkpoint_path(model_dir, "checkpoint_*.pth")
        except (IndexError, FileNotFoundError):
            print(f"Error: チェックポイントが見つかりません: {model_dir}")
            print("--checkpoint オプションでチェックポイントファイルを指定してください。")
            return

    print(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer = get_model_with_checkpoint(config, checkpoint_path, device, enable_internal_cache=args.enable_internal_cache)

    if args.prompt is not None:
        print("\n--- Generated Text ---")
        output_text, token_num, generation_time = generate(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            show_streaming=not args.no_streaming,
        )
        if args.no_streaming:
            print(output_text)
        print("--- End ---")
        print(f"Time: {generation_time:.2f} seconds")
        print(f"Token num: {token_num}")
        if generation_time > 0:
            print(f"Token per time: {token_num / generation_time:.2f} tok/s")
    else:
        interactive_mode(model, tokenizer, args.max_tokens, device, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
