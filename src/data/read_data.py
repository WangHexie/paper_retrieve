import os
from pathlib import Path

import pandas as pd


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def read_paper():
    return pd.read_csv(os.path.join(root_dir(), "data", "candidate_paper_for_wsdm2020.csv"))


def read_train_data():
    return pd.read_csv(os.path.join(root_dir(), "data", "train_release.csv"))


def read_validation_data():
    return pd.read_csv(os.path.join(root_dir(), "data", "validation.csv"))


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    print(read_train_data().tail())
