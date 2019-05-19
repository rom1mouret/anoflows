import torch
import torch.nn as nn
import torch.nn.functional as F

import tempfile
import pickle
import copyreg
import logging
import os
import json
import numpy as np
from copy import deepcopy, copy

from .flow import Flow, FlowParams
from .planar_flow import PlanarFlow, PlanarFlowParams
from .radial_flow import RadialFlow, RadialFlowParams

try:
    from tqdm import tqdm
except:
    tqdm_installed = False
    logging.warn("tqdm not installed")
    def tqdm(range, total=None, disable=False):
        return range
else:
    tqdm_installed = True

GaussNormalizer = -np.log(2*np.pi)/2  # the (2*pi*variance)^-1/2 in Gauss pdf


class AnoFlows:

    def __init__(self, device: str="cpu") -> None:
        """ initialize AnoFlows on a device.

        Parameters
        ----------
        device : str or torch.device
            If string, must be one of "cpu", "cuda", "cuda:0" etc.
        """
        self._ready_for_prediction = False
        self._flows = []
        if type(device) is str:
            self._device = torch.device(device)
        else:
            self._device = device

        self._l2 = 0.0001

    def _loss(self, inp_tensor: torch.Tensor, random_weights: bool=False) -> torch.Tensor:
        # reparametrization
        for f in self._flows:
            f.params().reparametrize()

        # inverse the flow
        y = inp_tensor
        ys = []
        for flow in reversed(self._flows):
            y = flow.inverse(y)
            ys.append(y)

        if random_weights:
            w = torch.randn(inp_tensor.size(0), device=self._device).abs()
            w_sum = w.sum()
        else:
            w = 1
            w_sum = inp_tensor.size(0)

        # loglikehood loss
        ll = -y.pow(2).sum(dim=1)/2  # independent normal distributions
        for f, z in zip(self._flows, reversed(ys)):
            # the minus comes from moving the ^-1 when applying the logarithm
            ll = ll - f.log_abs_jac_det(z)

        return -(ll * w).sum()/w_sum# + self._l2 * reg # negative loglikehood

    def fit(
            self,
            X: np.ndarray,
            lr: float=0.05,
            epochs: int=1000,
            validation_freq: int=1,
            quiet: bool=False,
            decay: float=0.000001) -> float:
        """ Training method for numpy data.
        Refer to torch_fit for finer-grained training.

        Parameters
        ----------
        X : np.ndarray
            Training data in float format.
            Categories are assumed to be already encoded, but it is not recommended
            to train with categorical data in the first place.
            It should work fine with 10,000 - 20,0000 rows. More rows are not recommended.
        validation_freq: int
            How many epochs between each validation phase, which may trigger model saving.
        quiet: bool
            Does not print anything on stdout/stderr unless it is an error.
        decay: float
            L2 weight decay as defined in PyTorch's documentation.

        Returns
        -------
        float
            Loss value. It can be compared with other AnoFlows for model selection.
        """
        X = X.astype(np.float32)
        validation_batch = torch.from_numpy(X[-100:, :]).to(self._device)
        inp_tensor = torch.from_numpy(X[:-100, :]).to(self._device)

        return self.torch_fit(inp_tensor, validation_batch, lr, epochs,
                              validation_freq, quiet)

    def torch_fit(self,
            inp_tensor: torch.Tensor,
            validation_batch: torch.Tensor,
            lr: float=0.05,
            epochs: int=1000,
            validation_freq: int=1,
            quiet: bool=False,
            decay: float=0.000001) -> float:

        # nan-agnostic scaling values
        nan_loc = torch.isnan(inp_tensor)
        clean_tensor = inp_tensor[~nan_loc]
        means = clean_tensor.mean(dim=0, keepdim=True)
        stds = clean_tensor.std(dim=0, keepdim=True).clamp(min=0.00001)
        self._means = means.data.cpu().numpy()
        self._stds = stds.data.cpu().numpy()

        # scaling & imputing
        inp_tensor = (inp_tensor - means)/stds
        validation_batch = (validation_batch - means)/stds
        inp_tensor[nan_loc] = 0
        validation_batch[torch.isnan(validation_batch)] = 0

        self._dim = inp_tensor.size(1)
        flow_params = [
            f.generate_params(self._dim).to(self._device) for f in self._flows
        ]
        for flow, p in zip(self._flows, flow_params):
            flow.set_params(p)

        optim_params = sum([list(fp.parameters()) for fp in flow_params], [])
        optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_loss = np.inf
        progress_bar = tqdm(range(epochs), total=epochs, disable=quiet)
        for epoch in progress_bar:
            loss = self._loss(inp_tensor, random_weights=False)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(optim_params, 1)
            optimizer.step()
            scheduler.step(epoch)

            # validation
            if epoch % validation_freq == 0:
                loss = self._loss(validation_batch).item()
                if loss < best_loss:
                    best_loss = loss
                    best_flows = [deepcopy(fp.state_dict()) for fp in flow_params]
                if tqdm_installed:
                    desc = "loss=%.04f;%.04f" % (loss, best_loss)
                    progress_bar.set_description(desc)

        for p, s in zip(flow_params, best_flows):
            p.load_state_dict(s)

        return best_loss

    def move_to_device(self, device: str) -> None:
        """ Move the flows to the given device.
        Use case: keeping models in memory after training on GPU.

        Parameters
        ----------
        device : str or torch.device
            If string, must be one of "cpu", "cuda", "cuda:0" etc.
        """
        if type(device) is str:
            device = torch.device(device)

        for f in self._flows:
            f.move_to_device(device)

        self._device = device

    def sample(self, n: int) -> np.ndarray:
        """ Randomly sample n data points with a forward pass on the flows,
        starting from a zero-mean, unit-variance Gaussian distribution.

        Parameters
        ----------
        n : int
            Number of data points to generate.

        Returns
        -------
        np.ndarray
            randomly sampled data points
        """
        z = torch.randn(n, self._dim)
        for f in self._flows:
            z = f.forward(z)

        r = z.data.numpy()
        r = r * self._stds + self._means

        return r

    def radial(self, num: int=1, positive_func: str="relu") -> "AnoFlows":
        """ Add radial flows to the flow stack.

        Parameters
        ----------
        num : int
            Number of radial flows to add.
        positive_func : str
            Function used for the reparametrization trick.
            Either "relu", "softplus" or "exp".

        Returns
        -------
        AnoFlows
            self
        """
        for _ in range(num):
            self.add_flow(RadialFlow(positive_func), RadialFlowParams.copyreg_func)
        return self

    def planar(self, num: int=1, positive_func: str="relu") -> "AnoFlows":
        """ Add planar flows to the flow stack.

        Parameters
        ----------
        num : int
            Number of radial flows to add.
        positive_func : str
            Function used for the reparametrization trick.
            Either "relu", "softplus" or "exp".

        Returns
        -------
        AnoFlows
            self
        """
        for _ in range(num):
            self.add_flow(PlanarFlow(positive_func), PlanarFlowParams.copyreg_func)
        return self

    def add_flow(self, f: Flow, copyreg_func) -> "AnoFlows":
        """ Add a custom flow to the flow stack.

        Parameters
        ----------
        copyreg_func : Callable
            This is for the serialization of the flow.
            See PlanarFlowParams.copyreg_func as example.

        Returns
        -------
        AnoFlows
            self
        """
        self._flows.append(f)
        param_prototype = f.generate_params(1)  # only to get the class
        copyreg.pickle(param_prototype.__class__, copyreg_func)
        return self

    def save(self, output_dir: str=tempfile.gettempdir()) -> tuple:
        """ Save the model on disk.

        Parameters
        ----------
        output_dir : str
            Directory where the model files will be stored.

        Returns
        -------
        tuple[str, list[str]]
            Paths to the the serialized main object and the serialized flows.
        """
        if output_dir is None:
            output_dir = "/tmp"

        # remove stuff we don't want to be serialized
        device = self._device
        del self._device

        # serialize the parameters
        filenames = []
        for i, flow in enumerate(self._flows):
            print("flow params before", list(flow.params().parameters()))
            output_file = os.path.join(output_dir, "flow_%i.torch" % i)
            torch.save(flow.params().state_dict(), output_file)
            filenames.append(output_file)
            print("flow params", list(flow.params().parameters()))

        main_file = os.path.join(output_dir, "anoflows.obj")
        with open(main_file, "wb") as f:
            pickle.dump(self, f)

        # restore the device
        self._device = device

        return main_file, filenames

    @staticmethod
    def load(main_file: str, flow_files: list, device: str="cpu") -> "AnoFlows":
        """ Load a model from disk.
        See tests/test_basic.py for an example.

        Parameters
        ----------
        main_file : str
            Path of file containing the serialized main object.
            First element returned by save()
        flow_files : str
            Paths of files containing the serialized flows.
            Second element returned by save().
        device : str or torch.device
            If string, must be one of "cpu", "cuda", "cuda:0" etc.

        Returns
        -------
        AnoFlows
            The deserialized object.
        """
        with open(main_file, "rb") as f:
            anoflows = pickle.load(f)

        if type(device) is str:
            anoflows._device = torch.device(device)
        else:
            anoflows._device = device

        assert len(anoflows._flows) == len(flow_files)
        for flow, flow_file in zip(anoflows._flows, flow_files):
            p = flow.params()
            p.load_state_dict(torch.load(flow_file, map_location=anoflows._device))
            p.eval()

        return anoflows

    def torch_likelihood(
            self,
            tensor: torch.Tensor,
            log_output: bool=True,
            normalized: bool=False,
            imputing: bool=True) ->  torch.Tensor:

        """ Compute the likelihood or loglikehood of the given data points.

        Parameters
        ----------
        tensor: torch.Tensor of dimension (n_samples, n_features)
            Data points for which we want the likelihood.
            It is your responsibility to move the tensor to the desired device,
            and to split it if computations wouldn't fit on the device.
        log_output : bool
            Whether to return logprobabilities or probabilities.
        normalized: bool
            Whether outputs should be probabilities or plain scores.
            Unnormalized scores are faster and just as good for anomaly detection.
        imputing : bool
            Whether to replace NaNs with mean values.
            Set imputing to false if you are sure there are no NaNs values.

        Returns
        -------
        torch.Tensor of dimension (n_samples, )
            Likelihood of the input. Stored on same device as the input.
        """

        if not self._ready_for_prediction:
            for flow in self._flows:
                flow.move_to_device(self._device)  # this includes reparametrization
            self._ready_for_prediction = True
            self._torch_means = torch.from_numpy(self._means).to(self._device)
            self._torch_stds = torch.from_numpy(self._stds).to(self._device)

        tensor -= self._torch_means
        tensor /= self._torch_stds

        if imputing:
            tensor[torch.isnan(tensor)] = 0

        # inverse the flow
        ys = []
        y = tensor
        for flow in reversed(self._flows):
            y = flow.inverse(y)
            ys.append(y)

        # loglikehood loss
        ll = -y.pow(2).sum(dim=1)/2  # independent normal distributions
        for f, z in zip(self._flows, reversed(ys)):
            # the minus comes from moving the ^-1 when applying the logarithm
            ll -= f.log_abs_jac_det(z)

        if not normalized:
            ll += GaussNormalizer

        if not log_output:
            ll = ll.exp()

        return ll

    def likelihood(
            self,
            input: np.ndarray,
            batch_size: int=1024,
            log_output: bool=True,
            normalized: bool=False,
            imputing: bool=True) -> np.ndarray:
        """ Compute the likelihood or loglikehood of the given data points.

        Parameters
        ----------
        tensor: np.ndarray (n_samples, n_features)
            Data points for which we want the likelihood.
        log_output : bool
            Whether to return logprobabilities or probabilities.
        normalized: bool
            Whether outputs should be probabilities or plain scores.
            Unnormalized scores are faster and just as good for anomaly detection.
        imputing : bool
            Whether to replace NaNs with mean values.
            Set imputing to false if you are sure there are no NaNs values.
        batch_size : int
            Maximum number of rows to be copied and kept on the device.

        Returns
        -------
        np.ndarray of dimension (n_samples, )
            Likelihood of the input.
        """


        # split into mini batches of batch_size/2 so we can use Torch's async transfers
        input = input.astype(np.float32)
        mini_batches = []
        for i in range(0, input.shape[0], batch_size//2):
            b = torch.from_numpy(input[i:i+batch_size//2, :])
            mini_batches.append(b)

        # scoring
        all_scores = []
        last_b = self._async_transfer(mini_batches[0])
        for b in mini_batches[1:]:
            current_b = last_b
            last_b = self._async_transfer(b)  # will be done as the same time as scoring below
            scores = self.torch_likelihood(
                            current_b,
                            log_output=log_output,
                            normalized=normalized,
                            imputing=imputing).data.cpu()
            all_scores.append(scores)

        scores = self.torch_likelihood(last_b, log_output=log_output).data.cpu()
        all_scores.append(scores)

        return torch.cat(all_scores, dim=0).numpy()

    def _async_transfer(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._device == torch.device('cpu'):
            return tensor

        return tensor.pin_memory().cuda(self._device, non_blocking=True)

    def __getstate__(self) -> dict:
        # called by pickle.dump
        serializable = copy(self.__dict__)
        if "_torch_means" in serializable:
            del serializable["_torch_means"]
            del serializable["_torch_stds"]

        serializable["_ready_for_prediction"] = False

        return serializable
