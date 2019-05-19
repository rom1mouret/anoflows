#!/usr/bin/env python3

from matplotlib import pyplot as plt
import os
import numpy as np
import logging
from sklearn import datasets

from anoflows.hpo import find_best_flows
from anoflows import AnoFlows

logging.getLogger().setLevel(logging.INFO)

dataset = "moons"
n_samples = 3000

if dataset == "moons":
    x0, x1 = -1.5, 2.5
    y0, y1 = -0.75, 1.25
    X = datasets.make_moons(n_samples=n_samples, noise=0.05)[0]
elif dataset == "circles":
    x0, x1 = -1.25, 1.25
    y0, y1 = -1.25, 1.25
    X = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0]
elif dataset == "swiss_roll":
    X = datasets.make_swiss_roll(n_samples=n_samples, noise=0.4)[0]
    x0, x1 = -11, 14
    y0, y1 = -12, 16
    X = X[:, [0, 2]]
elif dataset == "s_curve":
    x0, x1 = -1.5, 1.5
    y0, y1 = -2.5, 2.5
    X = datasets.make_s_curve(n_samples, noise=0.1)[0]
    X = X[:, [0, 2]]

#flows, loss = find_best_flows(X, n_trials=2)
flows = AnoFlows().planar(15)
flows.fit(X, lr=0.05)

plt.subplot(1, 3, 1)
plt.title(dataset, size=14)
plt.scatter(X[:, 0], X[:, 1], s=10, color="red")
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 3, 2)
plt.title("sampling", size=14)
points = flows.sample(n=800)
plt.scatter(points[:, 0], points[:, 1], s=10, color="green")
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.xticks(())
plt.yticks(())

xs = 0.1 * (x1 - x0)
ys = 0.1 * (y1 - y0)
x0 -= xs
x1 += xs
y0 -= ys
y1 += ys

x, y = np.meshgrid(np.linspace(x0, x1, 100),
                   np.linspace(y0, y1, 100))

grid = np.column_stack([x.ravel(), y.ravel()])
z = flows.likelihood(grid, log_output=False, normalized=True)

plt.subplot(1, 3, 3)
plt.title("density", size=14)
#https://chrisalbon.com/python/basics/set_the_color_of_a_matplotlib/
#plt.scatter(x.ravel(), y.ravel(), c=z, cmap=plt.cm.binary)
z = z.reshape(x.shape)
plt.pcolormesh(x, y, z, cmap=plt.cm.binary) #cmap=plt.cm.autumn)
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.xticks(())
plt.yticks(())

plt.show()
