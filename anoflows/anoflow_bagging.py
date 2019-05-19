import numpy as np

from .anoflows import AnoFlows

class AnoFlowBagging:
    """ experimental feature bagging """
    def __init__(self) -> None:
        pass

    def fit(self,
            X: np.ndarray,
            lr: float=0.05,
            epochs: int=1000,
            validation_freq: int=1,
            quiet: bool=False,
            decay: float=0.000001) -> float:

        dim = X.shape[1]
        n = max(1, dim // 5)

        # split the data column-wise
        self._sets = []
        for _  in range(1):
            feature_sets = np.array_split(np.random.permutation(dim), n)
            for subset in feature_sets:
                subset.sort()
                self._sets.append(subset)

        # training
        self._detectors = []
        loss_sum = 0
        for subset in self._sets:
            X_sub = X[:, subset]
            print("X_sub", X_sub.shape)
            detector = AnoFlows().radial(12)
            self._detectors.append(detector)
            loss_sum += detector.fit(X_sub, lr=lr, epochs=epochs, validation_freq=validation_freq, quiet=quiet, decay=decay)

        print("number of detectors", len(self._detectors))

        return loss_sum / len(self._sets)

    def likelihood(self,
            X: np.ndarray,
            batch_size: int=1024,
            log_output: bool=True,
            normalized: bool=False,
            imputing: bool=True) -> np.ndarray:

        ensembled = np.zeros(X.shape[0])
        for subset, detector in zip(self._sets, self._detectors):
            X_sub = X[:, subset]
            ensembled += detector.likelihood(X_sub,
                                    batch_size=batch_size,
                                    log_output=log_output,
                                    normalized=normalized)

        return ensembled
