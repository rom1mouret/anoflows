from abc import ABCMeta, abstractmethod
import torch

class FlowParams(metaclass=ABCMeta):
    @abstractmethod
    def reparametrize(self) -> None:
        """ Call this method after updating the parameters """
        pass

class Flow(metaclass=ABCMeta):
    @abstractmethod
    def generate_params(self, dim: int) -> FlowParams:
        """ Called only once at the initialization of the flow
        Parameters
        ----------
        dim : int
            Number of input features.
        """
        pass

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def log_abs_jac_det(self, z: torch.Tensor) -> torch.Tensor:
        """ see paper """
        pass

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        pass

    def set_params(self, params: FlowParams) -> None:
        self._p = params

    def move_to_device(self, device: torch.device) -> None:
        self._p = self._p.to(device)
        self._p.reparametrize()

    def params(self) -> FlowParams:
        return self._p
