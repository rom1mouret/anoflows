import logging
import numpy as np
import torch

from .anoflows import AnoFlows

def torch_find_best_flows(
        training_data: torch.Tensor,
        validation_batch: torch.Tensor,
        device: torch.device,
        n_flows: int,
        n_trials: int,
        epochs: int,
        quiet: bool,
        stay_on_device: bool) -> tuple:

    default_lr = 0.05
    default_l2 = 0.000001

    trials = []
    # generate trials
    trials.append({
        "n_planar": 0,
        "n_radial": n_flows,
        "lr": default_lr,
        "l2": default_l2
    })
    trials.append({
        "n_planar": n_flows,
        "n_radial": 0,
        "lr": default_lr,
        "l2": default_l2
    })
    trials.append({
        "n_planar": max(1, n_flows//2),
        "n_radial": max(1, n_flows//2),
        "lr": default_lr,
        "l2": default_l2
    })

    def add_flows(trial):
        types = ["radial"] * trial["n_radial"] + ["planar"] * trial["n_planar"]
        for flow_type in types:
            if flow_type == "radial":
                anoflows.radial(positive_func="relu")
            else:
                anoflows.planar(positive_func="relu")

    best_flow = None
    best_loss = float('inf')
    best_trial = None
    for t, trial in enumerate(trials[:n_trials]):
        anoflows = AnoFlows(device=device)
        add_flows(trial)

        loss = anoflows.torch_fit(
                    inp_tensor=training_data,
                    validation_batch=validation_batch,
                    lr=trial["lr"],
                    decay=trial["l2"],
                    epochs=epochs,
                    validation_freq=3
        )

        if loss < best_loss:
            best_loss = loss
            best_flow = anoflows
            best_flow.move_to_device('cpu')
            best_trial = trial

        if not quiet:
            args = (t+1, loss, trial["lr"], trial["l2"])
            logging.info("trial %i: loss=%.4f; lr=%.5f; l2=%f" % args)

    for t in range(n_trials - len(trials)):
        anoflows = AnoFlows(device=device)
        add_flows(best_trial)

        new_lr = max(0.0001, best_trial["lr"] + default_lr * np.random.normal())
        new_l2 = max(0, best_trial["l2"] + default_l2 * np.random.normal())

        loss = anoflows.torch_fit(
                    inp_tensor=training_data,
                    validation_batch=validation_batch,
                    lr=new_lr,
                    decay=new_l2,
                    epochs=epochs,
                    validation_freq=3
        )

        if loss < best_loss:
            best_loss = loss
            best_flow = anoflows
            best_flow.move_to_device('cpu')
            best_trial = trial

        if not quiet:
            args = (t+len(trials), loss,  new_lr, new_l2)
            logging.info("trial %i: loss=%.4f; lr=%.5f; l2=%f" % args)

    if stay_on_device:
        best_flow.move_to_device(device)

    return best_flow, best_loss

def find_best_flows(
        training_data: np.ndarray,
        validation_batch: np.ndarray=None,
        device: str="cpu",
        n_flows: int=20,
        n_trials: int=6,
        quiet: bool=False,
        epochs: int=1500,
        stay_on_device: bool=True) -> tuple:

    """ Naive HPO to find likelihood-maximizing flows.

    Parameters
    ----------
    training_data : np.ndarray
        The entire training data set.
        It is your responsibility to truncate the data array.
        Between 5,000 and 20,000 rows is usually a good choice.
    validation_batch : np.ndarray
        Small batch for validation. 256 rows should be enough.
        If not provided, the validation batch will be taken from the training data.
    quiet : bool
        Does not print anything on stdout/stderr unless it is an error.
    epochs : int
        Number of training epochs for every trial.
    device : str
        One of "cpu", "cuda", "cuda:0", "cuda:1" etc.
    stay_on_device : bool
        By default the flows are always moved to CPU, but if this option is True,
        the flows will be moved back to the given device.

    Returns
    -------
    AnoFlows
        Trained AnoFlows object.
    """

    X = training_data.astype(np.float32)

    if validation_batch is None:
        validation_batch = X[-100:, :]
        X = X[:-100, :]

    if type(device) is str:
        device = torch.device(device)

    inp_tensor = torch.from_numpy(X).to(device)
    validation_batch = torch.from_numpy(validation_batch).to(device)

    return torch_find_best_flows(
        training_data=inp_tensor,
        validation_batch=validation_batch,
        device=device,
        n_flows=n_flows,
        n_trials=n_trials,
        quiet=quiet,
        epochs=epochs,
        stay_on_device=stay_on_device
    )
