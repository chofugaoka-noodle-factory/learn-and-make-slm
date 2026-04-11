"""
ONNX推論スクリプト: エクスポート済みモデルを使用してテキスト生成を行う
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import numpy as np
from onnx import TensorProto
import torch
import onnxruntime as ort
import tiktoken

from config import SLMConfig


NP_TYPE_TO_TENSOR_PROTO = {
    np.dtype(np.float64): TensorProto.DOUBLE,
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int16): TensorProto.INT16,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.bool_): TensorProto.BOOL,
}

TORCH_TYPE_TO_TENSOR_PROTO = {
    torch.float64: TensorProto.DOUBLE,
    torch.float32: TensorProto.FLOAT,
    torch.float16: TensorProto.FLOAT16,
    torch.int64: TensorProto.INT64,
    torch.int32: TensorProto.INT32,
    torch.int16: TensorProto.INT16,
    torch.int8: TensorProto.INT8,
    torch.uint8: TensorProto.UINT8,
    torch.bool: TensorProto.BOOL,
}


def _tensor_proto_from_value(value: np.ndarray | torch.Tensor) -> int:
    if isinstance(value, np.ndarray):
        proto = NP_TYPE_TO_TENSOR_PROTO.get(value.dtype)
    else:
        proto = TORCH_TYPE_TO_TENSOR_PROTO.get(value.dtype)
    if proto is None:
        raise TypeError(f"Unsupported dtype for binding: {getattr(value, 'dtype', type(value))}")
    return proto


# GPUでKVキャッシュを活用するためにIO Bindingを使用してONNXセッションを実行する
def run_ort_session(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray | torch.Tensor],
    outputs: dict[str, np.ndarray | torch.Tensor] | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray | torch.Tensor]:
    io_binding = session.io_binding()

    # ——— 1) Bind inputs ———
    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            assert device == "cpu", "NumPy input can be bound only to CPU execution"
            io_binding.bind_cpu_input(name, value)
        elif isinstance(value, torch.Tensor):
            assert value.is_contiguous()
            dev_type = "cuda" if value.device.type == "cuda" else "cpu"
            io_binding.bind_input(
                name=name,
                device_type=dev_type,
                device_id=value.device.index or 0,
                element_type=_tensor_proto_from_value(value),
                shape=tuple(value.shape),
                buffer_ptr=value.data_ptr(),
            )
        else:
            raise TypeError(f"Unsupported input type for {name}: {type(value)}")

    # ——— 2) Bind outputs ———
    bound_outputs = outputs is not None and len(outputs) > 0
    if bound_outputs:
        for name, value in outputs.items():
            if isinstance(value, np.ndarray):
                assert device == "cpu", "NumPy output can be bound only to CPU"
                io_binding.bind_output(
                    name=name,
                    device_type="cpu",
                    device_id=0,
                    element_type=_tensor_proto_from_value(value),
                    shape=tuple(value.shape),
                    buffer_ptr=value.ctypes.data,
                )
            elif isinstance(value, torch.Tensor):
                assert value.is_contiguous()
                dev_type = "cuda" if value.device.type == "cuda" else "cpu"
                io_binding.bind_output(
                    name=name,
                    device_type=dev_type,
                    device_id=value.device.index or 0,
                    element_type=_tensor_proto_from_value(value),
                    shape=tuple(value.shape),
                    buffer_ptr=value.data_ptr(),
                )
            else:
                raise TypeError(f"Unsupported output type for {name}: {type(value)}")
    else:
        for output_meta in session.get_outputs():
            io_binding.bind_output(output_meta.name, device_type=device)

    session.run_with_iobinding(io_binding)

    if bound_outputs:
        return outputs

    output_names = [output_meta.name for output_meta in session.get_outputs()]
    values = io_binding.copy_outputs_to_cpu()
    return {name: values[idx] for idx, name in enumerate(output_names)}


def load_onnx_model(
    onnx_path: str, config: SLMConfig, device: str = "auto"
) -> tuple[ort.InferenceSession, tiktoken.Encoding, bool, str]:
    """ONNXモデルとトークナイザーをロードする"""
    tokenizer = tiktoken.get_encoding(config.tokenizer)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    requested_device = device.lower()
    available_providers = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available_providers and torch.cuda.is_available()

    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'")

    if requested_device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDAExecutionProvider is not available or torch CUDA is disabled")
        providers = ["CUDAExecutionProvider"]
    else:  # auto
        if cuda_available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    session_providers = session.get_providers()
    execution_device = "cuda" if session_providers and session_providers[0] == "CUDAExecutionProvider" else "cpu"
    if execution_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ONNX Runtime selected CUDA provider but torch CUDA is unavailable")

    print(f"ONNX model loaded with providers: {session_providers}")

    input_names = [input_meta.name for input_meta in session.get_inputs()]
    output_names = [output_meta.name for output_meta in session.get_outputs()]

    has_kv_cache = "past_keys" in input_names and "past_values" in input_names

    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    print(f"KV cache support: {has_kv_cache}")
    print(f"Execution device: {execution_device}")
    return session, tokenizer, has_kv_cache, execution_device


def sample_token(logits: np.ndarray | torch.Tensor, temperature: float = 0.0, top_k: int | None = None) -> int:
    """ロジットからトークンをサンプリングする"""
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = logits

    if temperature == 0.0:
        return int(np.argmax(logits_np))

    logits_np = logits_np / temperature

    if top_k is not None:
        top_k_indices = np.argpartition(logits_np, -top_k)[-top_k:]
        top_k_logits = logits_np[top_k_indices]
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
        probs = exp_logits / np.sum(exp_logits)
        sampled_idx = np.random.choice(len(top_k_indices), p=probs)
        return int(top_k_indices[sampled_idx])

    exp_logits = np.exp(logits_np - np.max(logits_np))
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(logits_np), p=probs))


def generate_onnx(
    session: ort.InferenceSession,
    tokenizer: tiktoken.Encoding,
    config: SLMConfig,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    show_streaming: bool = True,
    device: str = "cpu",
) -> tuple[str, int, float]:
    """ONNXモデルを使用してプロンプトからテキストを生成する"""
    input_ids = tokenizer.encode(prompt)
    current_ids = np.array([input_ids], dtype=np.int64)
    if show_streaming:
        print(prompt, end="", flush=True)

    input_names = [input_meta.name for input_meta in session.get_inputs()]
    output_names = [output_meta.name for output_meta in session.get_outputs()]

    use_cache = "past_keys" in input_names and "past_values" in input_names

    use_cuda = device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch CUDA is unavailable")
    torch_device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1
    cache_layers = config.model.n_layers
    cache_heads = config.model.n_heads
    cache_dim = max(1, config.model.d_model // config.model.n_heads)

    def _resolve_dim(dim_value: int | str | None, fallback: int) -> int:
        return dim_value if isinstance(dim_value, int) else fallback

    past_keys: np.ndarray | torch.Tensor | None = None
    past_values: np.ndarray | torch.Tensor | None = None

    if use_cache:
        for input_meta in session.get_inputs():
            if input_meta.name == "past_keys":
                shape = input_meta.shape
                cache_layers = _resolve_dim(shape[0], cache_layers) if len(shape) > 0 else cache_layers
                cache_heads = _resolve_dim(shape[2], cache_heads) if len(shape) > 2 else cache_heads
                cache_dim = _resolve_dim(shape[4], cache_dim) if len(shape) > 4 else cache_dim
                break

        # ONNXモデルはその性質上、past key及びpast valueのシーケンス長を0にできない。
        # 推論時は適当なseq_len=1のpast key及びpast valueを与える(ONNX内で最初の1シーケンス目はmaskingされるため問題ない)
        init_shape = (cache_layers, batch_size, cache_heads, 1, cache_dim)
        if use_cuda:
            past_keys = torch.zeros(init_shape, dtype=torch.float32, device=torch_device)
            past_values = torch.zeros(init_shape, dtype=torch.float32, device=torch_device)
        else:
            past_keys = np.zeros(init_shape, dtype=np.float32)
            past_values = np.zeros(init_shape, dtype=np.float32)

    start_time = time.time()

    for step in range(max_tokens):
        model_input = current_ids[:, -1:] if use_cache and step > 0 else current_ids
        inputs: dict[str, np.ndarray | torch.Tensor] = {}

        if use_cuda:
            inputs["input_ids"] = torch.from_numpy(model_input).to(dtype=torch.long, device=torch_device).contiguous()
        else:
            inputs["input_ids"] = model_input

        if past_keys is not None and past_values is not None:
            inputs["past_keys"] = past_keys
            inputs["past_values"] = past_values

        if use_cuda:
            seq_len = model_input.shape[1]
            logits_buffer = torch.empty((1, seq_len, tokenizer.n_vocab), dtype=torch.float32, device=torch_device)
            output_buffers: dict[str, np.ndarray | torch.Tensor] = {"logits": logits_buffer}

            if use_cache:
                prev_len = past_keys.shape[3] if past_keys is not None else 0
                kv_seq_len = prev_len + seq_len
                kv_shape = (cache_layers, batch_size, cache_heads, kv_seq_len, cache_dim)
                output_buffers["present_keys"] = torch.empty(kv_shape, dtype=torch.float32, device=torch_device)
                output_buffers["present_values"] = torch.empty(kv_shape, dtype=torch.float32, device=torch_device)

            output_dict = run_ort_session(session, inputs, output_buffers, device="cuda")
        else:
            outputs = session.run(output_names, inputs)
            output_dict = dict(zip(output_names, outputs))

        logits = output_dict["logits"]

        if use_cache and "present_keys" in output_dict and "present_values" in output_dict:
            past_keys = output_dict["present_keys"]
            past_values = output_dict["present_values"]

        next_token_logits = logits[0, -1, :]
        next_token = sample_token(next_token_logits, temperature, top_k)

        if next_token == tokenizer.eot_token:
            break

        if show_streaming:
            decoded = tokenizer.decode_tokens_bytes([next_token])
            for byte in decoded:
                sys.stdout.buffer.write(byte)
            sys.stdout.flush()

        current_ids = np.concatenate([current_ids, [[next_token]]], axis=1)

    end_time = time.time()

    if show_streaming:
        print()

    output_text = tokenizer.decode(current_ids[0].tolist())
    token_num = current_ids.shape[1] - len(input_ids)
    return output_text, token_num, end_time - start_time


def interactive_mode(
    session: ort.InferenceSession,
    tokenizer: tiktoken.Encoding,
    config: SLMConfig,
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> None:
    """インタラクティブモードでONNX推論を行う"""
    print("\n=== Interactive Mode (ONNX) ===")
    print("プロンプトを入力してください。終了するには 'quit' または 'exit' と入力してください。")
    print("空行で入力を確定します。複数行入力するには各行の末尾に \\ を付けてください。")
    print(f"設定: temperature={temperature}, top_k={top_k}, device={device}")
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
            _, token_num, generation_time = generate_onnx(
                session,
                tokenizer,
                config,
                prompt,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
                show_streaming=True,
                device=device,
            )
            print("--- End ---")
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
    parser = ArgumentParser(description="ONNXモデルを使用してテキスト生成を行う")
    parser.add_argument("config", type=Path, help="設定ファイル（YAML）のパス")
    parser.add_argument("onnx", type=Path, help="ONNXモデルのパス")
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
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX Runtimeの実行デバイス (auto/cpu/cuda)",
    )
    args = parser.parse_args()

    config = SLMConfig.load(args.config)
    session, tokenizer, has_kv_cache, execution_device = load_onnx_model(str(args.onnx), config, device=args.device)
    if has_kv_cache:
        print("KVキャッシュが有効です。")
    else:
        print("KVキャッシュが無効です。")
    print(f"Using ONNX Runtime device: {execution_device}")

    if args.prompt is not None:
        print("\n--- Generated Text ---")
        output_text, token_num, generation_time = generate_onnx(
            session,
            tokenizer,
            config,
            args.prompt,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            show_streaming=not args.no_streaming,
            device=execution_device,
        )
        if args.no_streaming:
            print(output_text)
        print("--- End ---")
        print(f"Time: {generation_time:.2f} seconds")
        print(f"Token num: {token_num}")
        if generation_time > 0:
            print(f"Token per time: {token_num / generation_time:.2f} tok/s")
    else:
        interactive_mode(
            session,
            tokenizer,
            config,
            args.max_tokens,
            args.temperature,
            args.top_k,
            execution_device,
        )


if __name__ == "__main__":
    main()
