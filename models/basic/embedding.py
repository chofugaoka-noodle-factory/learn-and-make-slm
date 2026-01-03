import torch
from torch import Tensor, nn

from utils.randn import nonzero_randn


class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices: Tensor, weight: Tensor) -> Tensor:
        # one_hot = torch.eye(num_embeddings)[x]
        # return one_hot @ self.w
        # torch.eyeでone hot vectorを作るのはOut of Memoryを引き起こすが、
        # 実は単に重みのindexを指定するだけで良く、これならOOMを回避できる
        return weight[indices]

    @staticmethod
    def symbolic(g, indices: Tensor, weight: Tensor):
        # ONNXのGatherオペレータを使用
        # axis=0 (行方向でgather)
        return g.op("Gather", weight, indices, axis_i=0)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # NOTE: Linearモジュールのコメントを参照
        self.w = nn.Parameter(nonzero_randn(num_embeddings, embedding_dim) * (embedding_dim**-0.5))

    def forward(self, x: Tensor) -> Tensor:
        # ONNX Export時、組み込みオペレータを使用するための処理
        if torch.onnx.is_in_onnx_export():
            return EmbeddingFunction.apply(x, self.w)
        else:
            return EmbeddingFunction.forward(None, x, self.w)
