import torch
from torch import Tensor, nn


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # Subtract max for numerical stability
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_exp = (x - x_max).exp()
        y = x_exp / x_exp.sum(dim=-1, keepdim=True)
        return y

    @staticmethod
    def symbolic(g, x: Tensor):
        # ONNXのSoftmaxオペレータを使用
        # axis=-1 (最後の次元でsoftmax)
        return g.op("Softmax", x, axis_i=-1)


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # ONNX Export時、組み込みオペレータを使用するための処理
        if torch.onnx.is_in_onnx_export():
            return SoftmaxFunction.apply(x)
        else:
            return SoftmaxFunction.forward(None, x)
