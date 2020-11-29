#   -*- coding: utf-8 -*-
#
#   utils.py
#   
#   Developed by Tianyi Liu on 2020-11-28 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import numpy as np
import pandas as pd
import pickle as pkl


def load_data(paths,
              transpose=False,
              groups=None,
              groups_col=None,
              batches=None,
              batches_col=None,
              log_normal=True,
              write_cache=True):

    data, batch, group = [], [], []
    if groups is not None and groups_col is not None:
        # Data, Groups from file
        for i, (path, label) in enumerate(zip(paths, groups)):
            df = pd.read_csv(path, index_col=0).to_numpy()
            df = df.T if transpose else df
            df = np.log1p(df) if log_normal else df
            data.append(df)
            lb = pd.read_csv(label, index_col=0)
            label.append(lb[groups_col].to_numpy())
            if batches_col is None:
                batch.append(np.array(["Batch {}".format(i) for _ in range(df.shape[1])]))
            else:
                batch.append(lb[batches_col].to_numpy())
            assert label[-1].shape[0] == df.shape[1]
    else:
        for i, (path, batch) in enumerate(zip(paths, batches)):
            df = pd.read_csv(path, index_col=0).to_numpy()
            df = df.T if transpose else df
            df = np.log1p(df) if log_normal else df
            data.append(df.to_numpy())
            ba = pd.read_csv(batch, index_col=0)
            if batches_col is None:
                batch.append(np.array(["Batch {}".format(i) for _ in range(df.shape[1])]))
            else:
                batch.append(ba[batches_col].to_numpy())
        if groups is not None:
            # Groups available directly, not from file
            group = groups

    data_dict = {"data": data, "batches": batch, "groups": group} if batch != [] else {"data": data, "batches": batch}

    if write_cache:
        with open("./cache.pkl", "wb") as f:
            pkl.dump(data_dict, f)
        print(">>> Cache written to ./cache.pkl")

    return data_dict


def load_cache(path):
    with open(path, "rb") as f:
        data_dict = pkl.load(f)

    print(">>> Cached data loaded from ./cache.pkl")
    if "group" in data_dict.keys():
        print("    Groups are available")
    return data_dict


def generate_test_data():
    x1 = np.abs(np.random.randn(1000, 800)) + 1
    x2 = np.abs(np.random.randn(1000, 800)) + 4
    x1 = np.log1p(x1)
    x2 = np.log1p(x2)

    return {"data": [x1, x2], "batch": np.concatenate((np.ones(800), np.ones(800) * 2))}











