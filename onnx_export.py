"""自前実装のモデルをDynamo経由でONNXにエクスポートする。"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from config import SLMConfig
from models.transformer.decoder import DecoderBase
from utils.model import get_model_with_checkpoint
from utils.checkpoint import latest_checkpoint_path


class ONNXModel(torch.nn.Module):
    def __init__(self, model: DecoderBase, enable_kv_cache: bool):
        super().__init__()
        self.model = model
        self.enable_kv_cache = enable_kv_cache

    def forward(self, input_ids, past_keys=None, past_values=None):
        logits, present_keys, present_values = self.model(input_ids, past_keys, past_values)
        if self.enable_kv_cache:
            return logits, present_keys, present_values
        return logits


def export_model(model: DecoderBase, output_path: Path, opset_version: int = 23, enable_kv_cache: bool = False) -> None:
    # 古いopsetではAttentionとRoPEを自前のテンソル演算に分解する。
    for module in model.modules():
        if hasattr(module, "onnx_opset_version"):
            module.onnx_opset_version = opset_version

    device = next(model.parameters()).device
    input_ids = torch.randint(0, model.n_vocab, (2, 10), device=device)
    inputs = (input_ids,)
    input_names = ["input_ids"]
    output_names = ["logits"]
    batch = torch.export.Dim.AUTO
    sequence = torch.export.Dim.AUTO
    dynamic_shapes = ({0: batch, 1: sequence},)
    if enable_kv_cache:
        attention = model.layers[0].self_attn
        cache_heads = attention.n_heads // getattr(attention, "n_groups", 1)
        cache_shape = (len(model.layers), 2, cache_heads, 10, model.d_model // attention.n_heads)
        past_keys = torch.zeros(cache_shape, device=device)
        past_values = torch.zeros_like(past_keys)
        inputs = (input_ids, past_keys, past_values)
        input_names += ["past_keys", "past_values"]
        output_names += ["present_keys", "present_values"]
        cache_axes = {1: batch, 3: torch.export.Dim.AUTO}
        dynamic_shapes += (cache_axes, cache_axes)

    torch.onnx.export(
        ONNXModel(model.eval(), enable_kv_cache),
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="省略時は設定のlog_dirから最新のcheckpointを使用")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-kv-cache", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = SLMConfig.load(args.config)
    checkpoint_path = args.checkpoint or latest_checkpoint_path(Path(config.path.log_dir), "checkpoint_*.pth")
    if checkpoint_path is None:
        parser.error("No checkpoint found; specify --checkpoint or check path.log_dir")
    device = torch.device(args.device)
    model, _ = get_model_with_checkpoint(config, str(checkpoint_path), device)
    opset_version = 18 if device.type == "cuda" else 23
    export_model(model, args.output, opset_version, args.enable_kv_cache)
    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
