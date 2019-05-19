import unittest
from anoflows import AnoFlows
import numpy as np

class TestBasic(unittest.TestCase):
    def _train_and_reload(self, device1, device2):
        # random training data
        X = np.random.normal(size=(200, 2))

        # training
        f1 = AnoFlows(device=device1).planar(2).radial(2)
        f1.fit(X, quiet=True, epochs=1)
        f1.fit(X, quiet=True, epochs=1)  # does retraining work?

        # keep predictions as reference
        ref = f1.likelihood(X)

        # save model
        main_file, flow_files = f1.save()
        print("main_file", main_file, "flow_files", flow_files)

        # can we still do predict and train?
        f1.likelihood(X)
        f1.fit(X, quiet=True, epochs=1)

        # save and reload
        f2 = AnoFlows.load(main_file, flow_files, device=device2)

        # compare predictions
        pred = f2.likelihood(X)
        diff = np.abs(ref - pred).mean()
        self.assertAlmostEqual(diff, 0.0)

    def test_reload(self):
        self._train_and_reload("cuda:0", "cpu")
        self._train_and_reload("cuda:0", "cuda:0")
        self._train_and_reload("cpu", "cuda:0")

if __name__ == '__main__':
    unittest.main()
