import unittest
from anoflows import AnoFlows
import numpy as np

class TestBasic(unittest.TestCase):
    def test_serialization(self):
        # random training data
        X = np.random.normal(size=(200, 2))

        # training
        f1 = AnoFlows().planar(2).radial(2)
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
        f2 = AnoFlows.load(main_file, flow_files)

        # compare predictions
        pred = f2.likelihood(X)
        diff = np.abs(ref - pred).mean()
        self.assertAlmostEqual(diff, 0.0)

if __name__ == '__main__':
    unittest.main()
