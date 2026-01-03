import torch
from torch import Tensor
from jaxtyping import Float

from models.basic.linear import Linear


def sigmoid(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    if torch.onnx.is_in_onnx_export():
        return torch.onnx.ops.symbolic("::Sigmoid", (x,), dtype=x.dtype, shape=x.shape, version=13)
    return 1 / (1 + (-x).exp())


class Swish(torch.nn.Module):
    def __init__(self, beta: float = 1.0, beta_is_learnable: bool = False):
        """
        Implements Swish.

        See: https://arxiv.org/abs/1710.05941

        Parameters
        ----------
        beta : float, optional
            Sharpness parameter, by default 1.0
        mark_beta_learnable : bool, optional
            Whether beta is learnable, by default False

            Why default to False?
            In the original paper, it was shown experimentally that even when beta is marked as a learnable parameter, it converges to a value close to 1.0, indicating that fixing it to 1.0 in practice causes no issues.
        """
        super().__init__()
        if beta_is_learnable:
            self.beta = torch.nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * sigmoid(self.beta * x)


class GLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.proj = Linear(dim_in, dim_hidden * 2)

    def forward(self, x: Float[Tensor, "... D_in"]) -> Float[Tensor, "... D_out"]:
        w, v = self.proj(x).chunk(2, dim=-1)
        return w * sigmoid(v)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.proj = Linear(dim_in, dim_hidden * 2)
        self._swish = Swish()

    def forward(self, x: Float[Tensor, "... D_in"]) -> Float[Tensor, "... D_out"]:
        w, v = self.proj(x).chunk(2, dim=-1)
        return w * self._swish(v)
