import pandas as pd
import yaml
import numpy as np
import logging

def is_int(arr: list):
    if arr is None:
        return False

    return set(map(type, arr)) == set([int])

def load_data(specification_yaml_file: str) -> tuple:
    try:
        with open(specification_yaml_file, "r") as f:
            data = yaml.load(f)
    except:
        logging.error("check that the first argument is a valid YAML file.")
        raise

    sep = data.get("delimiter", ",")
    missing_marker = data.get("missing_marker", "")

    # header
    if is_int(data.get("features")) or is_int(data.get("not_features")):
        df = pd.read_csv(data["file"], sep=sep, nrows=1)
        n_cols = len(df.columns)
        #header_index = None
        columns = list(map(str, range(n_cols)))
    else:
        #header_index = 0
        df = pd.read_csv(
                    data["file"],
                    sep=sep,
                    header=0,
                    nrows=1)
        columns = df.columns.tolist()

    # features
    features = list(map(str, data.get("features", [])))
    label_col = str(data["label_column"])
    if str(label_col) == "-1":
        label_col = str(len(columns)-1)
    data["label_column"] = label_col
    not_features = list(map(str, data.get("not_features", [])))
    if not_features != []:
        features = list(set(columns) - set(not_features+[label_col]))

    # header
    if data.get("indexed", False):
        columns = ["index"] + columns

    logging.info("using %s as header", ",".join(columns))

    # prepare class filtering
    normal_classes = data["normal_classes"]
    anomaly_classes = data["anomaly_classes"]
    df = pd.read_csv(
                data["file"],
                sep=sep,
                skiprows=1,
                names=columns,
                na_filter=False,
                usecols=[label_col])

    # ordered list of classes (as required by np.isin)
    classes = sorted(list(set(normal_classes + anomaly_classes)))

    # +1 because the first row is to be skipped
    keep = set((np.where(np.isin(df[label_col], classes))[0]+1).tolist())
    assert len(keep) > 0, "provided classes match the labels in the CSV file"

    # read every needed column
    usecols = features + [label_col]
    logging.info("reading columns %s in no particular order", ",".join(usecols))

    df = pd.read_csv(
                data["file"],
                sep=sep,
                names=columns,
                usecols=usecols,
                keep_default_na=False,
                na_values=[missing_marker],
                skiprows=lambda i: i not in keep)

    return df, data
