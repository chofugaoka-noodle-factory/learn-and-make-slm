import torch
from jaxtyping import Float
from torch import Tensor, nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic("::Softmax", (x,), {"axis": -1}, dtype=x.dtype, shape=x.shape, version=13)

        # Subtract max for numerical stability
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_exp = (x - x_max).exp()
        return x_exp / x_exp.sum(dim=-1, keepdim=True)
