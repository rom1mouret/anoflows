#!/usr/bin/env python3

import sys
import logging
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from anoflows.hpo import find_best_flows

from data_loading import load_data

logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) == 1:
    logging.error("YAML data specification missing from the command line arguments")
    exit(1)

spec_file = sys.argv[1]
df, spec = load_data(spec_file)
max_rows = min(len(df), spec.get("max_rows", 40000))
novelty_detection = spec.get("novelty", True)
normal_classes = spec["normal_classes"]

precision = defaultdict(list)

for rounds in range(spec.get("rounds", 1)):
    # random sampling
    df = df.sample(n=max_rows, replace=False)
    label_col = spec["label_column"]
    y = df[label_col].values
    other = df.drop(label_col, inplace=False, axis=1)
    X = other.values

    # imputing
    X = SimpleImputer(copy=False).fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, shuffle=False, test_size=0.5)
    if novelty_detection:
        keep = np.where(np.isin(y_train, normal_classes))[0]
        X_train = X_train[keep, :]
        y_train = y_train[keep]

    # training
    #flows, loss = find_best_flows(X_train, device='cpu', n_trials=1)
    from anoflows.anoflow_bagging import AnoFlowBagging
    flows = AnoFlowBagging()
    flows.fit(X_train)
    iforest = IsolationForest().fit(X_train)

    # prediction
    pred = {
        "anoflows": flows.likelihood(X_test),
        "iforest": iforest.decision_function(X_test)
    }

    # evaluation
    y_true = np.where(np.isin(y_test, spec["anomaly_classes"]))[0]
    ref = np.zeros(len(y_test))
    ref[y_true] = 1
    k = len(y_true)
    for name, y_pred in pred.items():
        anomaly_indices = y_pred.argsort()[:k]
        prec = ref[anomaly_indices].sum() / k
        logging.info("%s: %.1f%% (%d anomalies / %d rows)" % (name, 100*prec, k, len(y_test)))
        precision[name].append(prec)

logging.info("* SUMMARY %s", spec_file)
for name, prec in precision.items():
    prec = 100 * np.array(prec)
    mean = np.mean(prec)
    std = np.std(prec)
    logging.info("%s; mean=%.1f%% std=%.1f%%" % (name, mean, std))
