"""
推論スクリプト: 学習済みモデルを使用してテキスト生成を行う
"""

from argparse import ArgumentParser
from pathlib import Path

import sys
import time

import numpy as np
import onnxruntime as ort
import tiktoken
import torch

from config import SLMConfig
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint


def load_model(
    config: SLMConfig, checkpoint_path: str, device: torch.device, enable_internal_cache: bool = False
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
                enable_internal_cache=enable_internal_cache,
            )
        case _:
            raise ValueError(f"Model type {config.model.model_type} not supported")

    # チェックポイントからモデルをロード
    load_checkpoint(checkpoint_path, model, printf=print)

    # モデルをデバイスに転送し、推論モードに設定
    model.to(device)
    model.eval()

    return model, tokenizer


def load_onnx_model(onnx_path: str, config: SLMConfig) -> tuple[ort.InferenceSession, tiktoken.Encoding]:
    """ONNXモデルとトークナイザーをロードする"""
    # トークナイザーの初期化
    tokenizer = tiktoken.get_encoding(config.tokenizer)

    # ONNXランタイムセッションの作成
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 利用可能なプロバイダーを取得
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

    print(f"ONNX model loaded with providers: {session.get_providers()}")

    return session, tokenizer


def sample_token(logits: np.ndarray, temperature: float = 0.0, top_k: int | None = None) -> int:
    """ロジットからトークンをサンプリングする"""
    if temperature == 0.0:
        # Greedy sampling
        return int(np.argmax(logits))

    # Temperature scaling
    logits = logits / temperature

    # Top-k filtering
    if top_k is not None:
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]
        # Softmax
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
        probs = exp_logits / np.sum(exp_logits)
        # Sample
        sampled_idx = np.random.choice(len(top_k_indices), p=probs)
        return int(top_k_indices[sampled_idx])
    else:
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        # Sample
        return int(np.random.choice(len(logits), p=probs))


def generate_onnx(
    session: ort.InferenceSession,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    show_streaming: bool = True,
) -> tuple[str, int, float]:
    """ONNXモデルを使用してプロンプトからテキストを生成する"""
    # プロンプトをトークンIDに変換
    input_ids = tokenizer.encode(prompt)
    current_ids = np.array([input_ids], dtype=np.int64)
    print(prompt)

    start_time = time.time()

    # 自己回帰的にトークンを生成
    for _ in range(max_tokens):
        # ONNXモデルで推論
        outputs = session.run(None, {"input": current_ids})
        logits = outputs[0]  # Shape: (batch_size, seq_len, vocab_size)

        # 最後のトークンのロジットを取得
        next_token_logits = logits[0, -1, :]

        # 次のトークンをサンプリング
        next_token = sample_token(next_token_logits, temperature, top_k)

        # ストリーミング出力
        if show_streaming:
            decoded = tokenizer.decode_tokens_bytes([next_token])
            for byte in decoded:
                sys.stdout.buffer.write(byte)
            sys.stdout.flush()

        # 生成されたトークンを追加
        current_ids = np.concatenate([current_ids, [[next_token]]], axis=1)

        # EOTトークンで終了
        if next_token == tokenizer.eot_token:
            break

    end_time = time.time()

    if show_streaming:
        print()  # 改行を追加

    # 生成されたトークンIDをテキストに変換
    output_text = tokenizer.decode(current_ids[0].tolist())
    token_num = current_ids.shape[1] - len(input_ids)

    return output_text, token_num, end_time - start_time


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
    # プロンプトをトークンIDに変換
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 推論の実行
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if show_streaming:
                start_time = time.time()
                output_ids = model.infer(
                    starts=input_tensor,
                    max_token_count=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    tokenizer=tokenizer,
                )
                end_time = time.time()
                print()  # 改行を追加
            else:
                start_time = time.time()
                output_ids = model.infer(
                    starts=input_tensor,
                    max_token_count=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    tokenizer=None,
                )
                end_time = time.time()
            token_num = output_ids.size(1) - input_tensor.size(1)
    # 生成されたトークンIDをテキストに変換
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text, token_num, end_time - start_time


def interactive_mode(
    model: GPT2Decoder | ort.InferenceSession,
    tokenizer: tiktoken.Encoding,
    max_tokens: int,
    device: torch.device | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    use_onnx: bool = False,
):
    """インタラクティブモードでテキスト生成を行う"""
    print("\n=== Interactive Mode ===")
    print("プロンプトを入力してください。終了するには 'quit' または 'exit' と入力してください。")
    print("空行で入力を確定します。複数行入力するには各行の末尾に \\ を付けてください。")
    print(f"設定: temperature={temperature}, top_k={top_k}")
    print("=" * 50)

    while True:
        try:
            print("\nPrompt: ", end="")
            lines = []
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
            if use_onnx:
                _, token_num, generation_time = generate_onnx(
                    model, tokenizer, prompt, max_tokens, temperature, top_k, show_streaming=True
                )
            else:
                _, token_num, generation_time = generate(
                    model, tokenizer, prompt, max_tokens, device, temperature, top_k, show_streaming=True
                )
            print("--- End ---")
            print(f"Time: {generation_time:.2f} seconds")
            print(f"Token num: {token_num}")
            print(f"Token per time: {token_num / generation_time:.2f} tok/s")

        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except EOFError:
            print("\n終了します。")
            break


def main():
    parser = ArgumentParser(description="学習済みモデルを使用してテキスト生成を行う")
    parser.add_argument("config", type=Path, help="設定ファイル（YAML）のパス")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="チェックポイントファイルのパス。指定しない場合は最新のチェックポイントを使用",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="ONNXモデルのパス。指定した場合はONNXモデルで推論を行う",
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
        help="KVキャッシュを有効にする（ONNXモードでは無効）",
    )
    args = parser.parse_args()

    # 設定の読み込み
    config = SLMConfig.load(args.config)

    # ONNXモデルを使用する場合
    if args.onnx is not None:
        print(f"Loading ONNX model: {args.onnx}")
        model, tokenizer = load_onnx_model(args.onnx, config)
        device = None

        if args.enable_cache:
            print("Warning: KVキャッシュはONNXモードでは利用できません")

        if args.prompt is not None:
            # 単一のプロンプトを処理
            print("\n--- Generated Text ---")
            output_text, token_num, generation_time = generate_onnx(
                model,
                tokenizer,
                args.prompt,
                args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                show_streaming=not args.no_streaming,
            )
            if args.no_streaming:
                print(output_text)
            print("--- End ---")
            print(f"Time: {generation_time:.2f} seconds")
            print(f"Token num: {token_num}")
            print(f"Token per time: {token_num / generation_time:.2f} tok/s")
        else:
            # インタラクティブモード
            interactive_mode(model, tokenizer, args.max_tokens, None, args.temperature, args.top_k, use_onnx=True)
    else:
        # PyTorchモデルを使用する場合
        device = torch.device(args.device)
        print(f"Using device: {device}")

        # チェックポイントのパスを決定
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        else:
            # 最新のチェックポイントを使用
            model_dir = Path(config.path.log_dir)
            try:
                checkpoint_path = latest_checkpoint_path(model_dir, "checkpoint_*.pth")
            except (IndexError, FileNotFoundError):
                print(f"Error: チェックポイントが見つかりません: {model_dir}")
                print("--checkpoint オプションでチェックポイントファイルを指定してください。")
                return

        print(f"Loading checkpoint: {checkpoint_path}")

        # モデルのロード
        model, tokenizer = load_model(config, checkpoint_path, device, enable_internal_cache=args.enable_internal_cache)

        if args.prompt is not None:
            # 単一のプロンプトを処理
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
            print(f"Token per time: {token_num / generation_time:.2f} tok/s")
        else:
            # インタラクティブモード
            interactive_mode(model, tokenizer, args.max_tokens, device, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
