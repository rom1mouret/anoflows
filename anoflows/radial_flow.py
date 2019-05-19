import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Flow, FlowParams

eps = 0.000001

class RadialFlowParams(nn.Module, FlowParams):
    """ not an nn.module. Allows us to overide parameters """
    def __init__(self, dim: int, positive_func: str) -> None:
        super(RadialFlowParams, self).__init__()
        self._alpha_source = nn.Parameter(torch.randn(1).abs())
        self._beta_source = nn.Parameter(torch.randn(1).abs())
        self._z0 = nn.Parameter(torch.randn(1, dim))
        self._dim = dim
        self._positive_func = positive_func.lower()
        if positive_func == "relu":
            self._reparam_func = F.relu
        elif positive_func == "exp":
            self._reparam_func = torch.exp
        elif positive_func == "softplus":
            self._reparam_func = F.softplus
        else:
            raise ValueError("unknown function '{}'".format(positive_func))

    def alpha(self) -> torch.Tensor:
        return self._alpha

    def beta(self) -> torch.Tensor:
        return self._beta

    def z0(self) -> torch.Tensor:
        return self._z0

    def reparametrize(self) -> None:
        # beta >= -alpha is required for invertibility
        # see paper's appendix
        self._alpha = self._reparam_func(self._alpha_source) + 0.01
        self._beta = self._reparam_func(self._beta_source) - self._alpha

    @staticmethod
    def copyreg_func(p) -> tuple:
        return RadialFlowParams, (p._dim, p._positive_func)


class RadialFlow(Flow):
    def __init__(self, positive_func: str="relu"):
        self._positive_func = positive_func.lower()

    def generate_params(self, dim: int) -> RadialFlowParams:
        p = RadialFlowParams(dim, self._positive_func)
        self._dim = dim
        return p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z - self._p.z0(), dim=1, p=2)
        return z + self._p.beta() * self._h(r).unsqueeze(1) * (z - self._p.z0())

    def log_abs_jac_det(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z - self._p.z0(), dim=1)
        left = 1 + self._p.beta() * self._h(r) + eps
        right = left + self._p.beta() * self._h_derivative(r) * r
        return (self._dim - 1) * left.log() + right.log()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        d = torch.norm(y - self._p.z0(), dim=1, p=2)
        a = self._p.alpha()
        b = self._p.beta()
        r = ((a**2 + 2*a*(b+d) + (b-d)**2).sqrt() - a - b + d)/2
        ratio = (b/(a + r)).unsqueeze(1)
        z = (y + self._p.z0()*ratio)/(1 + ratio)

        return z

    def reg(self) -> torch.Tensor:
        return self._p.beta()**2

    def _h(self, r: torch.Tensor) -> torch.Tensor:
        return 1 / (self._p.alpha() + r + eps)

    def _h_derivative(self, r: torch.Tensor) -> torch.Tensor:
        return -1 / (self._p.alpha() + r + eps)**2
