import torch
from torch import Tensor, nn

from utils.randn import nonzero_randn


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        # y = x @ weight.T + bias
        return x @ weight.T + bias

    @staticmethod
    def symbolic(g, x: Tensor, weight: Tensor, bias: Tensor):
        # ONNXのGemmオペレータを使用
        # Gemm: Y = alpha * A @ B.T + beta * C
        # alpha=1.0, beta=1.0, transB=1 (weight.Tのため)
        return g.op("Gemm", x, weight, bias, alpha_f=1.0, beta_f=1.0, transB_i=1)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            入力 Tensor の次元数
        out_features : int
            出力 Tensor の次元数
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # NOTE: 一旦適当な行列で初期化しておくと、iteration ごとにいい感じの Tensor に収束していくので
        # 具体的な中身の Tensor を気にする必要はない。ただし要素に 0 があると影響が出る可能性があるので避けたい。
        # また、in_featuresで割ることで、学習の安定化を図る(Xavierの正規分布というらしい)
        self.weight = nn.Parameter(nonzero_randn(out_features, in_features) * (in_features**-0.5))
        self.bias = nn.Parameter(nonzero_randn(out_features) * (out_features**-0.5))

    def forward(self, x: Tensor) -> Tensor:
        # ONNX Export時、組み込みオペレータを使用するための処理
        if torch.onnx.is_in_onnx_export():
            return LinearFunction.apply(x, self.weight, self.bias)
        else:
            return LinearFunction.forward(None, x, self.weight, self.bias)
