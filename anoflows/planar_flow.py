import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Flow, FlowParams

eps = 0.000001

class PlanarFlowParams(nn.Module, FlowParams):
    def __init__(self, dim: int, positive_func: str) -> None:
        super(PlanarFlowParams, self).__init__()
        # w components are summed inside the non-linearity, that is why it is divided by dim
        self._w = nn.Parameter(torch.randn(dim) / dim)
        self._u_source = nn.Parameter(torch.randn(dim))
        self._b = nn.Parameter(torch.randn(1))
        self._positive_func = positive_func.lower()
        self._dim = dim
        if self._positive_func not in ("relu", "exp", "softplus"):
            raise ValueError("unknown function '{}'".format(self._positive_func))

    def w(self) -> torch.Tensor:
        return self._w

    def w_t(self) -> torch.Tensor:
        return self._w.unsqueeze(0)

    def u(self) -> torch.Tensor:
        return self._u

    def b(self) -> torch.Tensor:
        return self._b

    def reparametrize(self) -> None:
        # required for invertibility
        # see paper's appendix
        v = (self._w * self._u_source).sum()
        norm_square = self._w.pow(2).sum()
        if self._positive_func == "relu":
            m = v.clamp(min=-0.9999)
        elif self._positive_func == "exp":
            m = v.exp() - 1
        else:
            m = F.softplus(v) - 1

        self._u = self._u_source + self._w * (m / norm_square)

    @staticmethod
    def copyreg_func(p) -> tuple:
        return PlanarFlowParams, (p._dim, p._positive_func)



class PlanarFlow(Flow):
    def __init__(self, positive_func: str="softplus") -> None:
        super(PlanarFlow, self).__init__()
        self._positive_func = positive_func

    def generate_params(self, dim: int) -> FlowParams:
        return PlanarFlowParams(dim, self._positive_func)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        linear_comb = (self._p.w_t() * z).sum(dim=1, keepdim=True)
        return z + self._p.u() * self._h(linear_comb + self._p.b())

    def log_abs_jac_det(self, z: torch.Tensor) -> torch.Tensor:
        linear_comb1 = (self._p.w_t() * z).sum(dim=1, keepdim=True)
        psi = self._h_derivative(linear_comb1 + self._p.b()) * self._p.w()
        linear_comb2 = (self._p.u().unsqueeze(0) * psi).sum(dim=1, keepdim=True)
        return ((1 + linear_comb2).abs() + eps).log().squeeze(1)

    def _h(self, t: torch.Tensor) -> torch.Tensor:
        return t/(1+t.abs())

    def _h_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 1/(1+t.abs()).pow(2)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        g = (self._p.w_t() * self._p.u()).sum(dim=1, keepdim=True)
        m = (self._p.w_t() * y).sum(dim=1, keepdim=True)
        b = self._p.b()

        i1 = b**2 + 2*b*(g+m-1)+g**2+2*g*(m+1)+(m-1)**2
        i2 = b**2 + 2*b*(-g+m+1)+g**2-2*g*(m-1)+(m+1)**2
        a1 = 0.5*(-(i1.clamp(min=0.0001)).sqrt() - b + g + m + 1)
        a2 = 0.5*((i2.clamp(min=0.0001)).sqrt() - b - g + m - 1)

        cond1 = (b <= -m).float()
        cond2 = 1 - cond1
        alpha = cond1 * a1 + cond2 * a2

        z = y - self._p.u() * self._h(alpha + self._p.b())

        return z
