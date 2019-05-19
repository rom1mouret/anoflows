#!/usr/bin/env python3

from anoflows import AnoFlows
from sklearn.ensemble import IsolationForest
import time
import numpy as np

dim = 8
device = "cpu"

# training
flow = AnoFlows(device=device).radial(15)
flow.fit(np.random.normal(size=(300, dim)))
forest = IsolationForest().fit(np.random.normal(size=(10000, dim)))

# prediction
batches = [
    np.random.normal(size=(100000, dim)) for _ in range(10)
]

def report(name: str, total_duration: float):
    total_rows = sum(map(len, batches))
    per_row = total_rows / total_duration
    p = (name, total_duration, per_row, device, dim)
    print("[%s] total duration: %f; speed: %f; device: %s, dim: %i" % p)


# AnoFlows time
before = time.time()
for batch in batches:
    flow.likelihood(batch, imputing=False, batch_size=16384)
duration = time.time() - before
report("AnoFlows", duration)

# iforest time
before = time.time()
for batch in batches:
    forest.score_samples(batch)
duration = time.time() - before
report("iforest", duration)
