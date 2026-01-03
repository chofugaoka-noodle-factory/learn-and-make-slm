import torch
from torch import Tensor, nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False: バイアス分散 sum((x-mean)^2)/N を計算
        # unbiased=True (デフォルト): 不偏分散 sum((x-mean)^2)/(N-1) を計算
        # LayerNormの標準実装では N で割るバイアス分散を使用する
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
        y = x_normalized * gamma + beta
        return y

    @staticmethod
    def symbolic(g, x: Tensor, gamma: Tensor, beta: Tensor, eps: float):
        # ONNXのLayerNormalizationオペレータを使用
        # axis=-1 (最後の次元で正規化)
        return g.op("LayerNormalization", x, gamma, beta, axis_i=-1, epsilon_f=eps)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int]):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        # PyTorchのデフォルト値
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        # ONNX Export時、組み込みオペレータを使用するための処理
        if torch.onnx.is_in_onnx_export():
            return LayerNormFunction.apply(x, self.gamma, self.beta, self.eps)
        else:
            return LayerNormFunction.forward(None, x, self.gamma, self.beta, self.eps)
