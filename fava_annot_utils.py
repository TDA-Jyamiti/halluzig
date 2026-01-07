import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm


_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]


def get_modified_data():
    # loading "annotations.json" file
    with open("fava_annotated_data/annotations.json", "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    df = {
        "prompt": [],
        "output": [],
        "annotated": [],
        "modified": [],
        "model": [],
        "entity": [],
        "relation": [],
        "sentence": [],
        "invented": [],
        "subjective": [],
        "unverifiable": [],
        "hallucinated": [],
    }

    def modify(s):
        indicator = [0, 0, 0, 0, 0, 0]
        soup = BeautifulSoup(s, "html.parser")
        s1 = ""
        for t in range(len(_TAGS)):
            indicator[t] = len(soup.find_all(_TAGS[t]))
        for elem in soup.find_all(text=True):
            if elem.parent.name != "delete":
                s1 += elem
        return s1, indicator

    for i in range(len(data)):
        df["prompt"].append(data[i]["prompt"])
        df["output"].append(data[i]["output"])
        df["annotated"].append(data[i]["annotated"])
        df["model"].append(data[i]["model"])
        modified_text, indicator = modify(data[i]["annotated"])
        df["modified"].append(modified_text)
        for t in range(len(_TAGS)):
            df[_TAGS[t]].append(indicator[t])
        df["hallucinated"].append(int(sum(indicator) > 0))

    df = pd.DataFrame(df)
    return df


def get_fava_data(n_samples=200):
    np.random.seed(0)
    data = get_modified_data()
    i1 = data["hallucinated"] == 1
    i2 = data["hallucinated"] == 0
    df = pd.concat(
        [
            data[i1][:n_samples],
            data[i2][:n_samples],
        ],
        ignore_index=True,
        sort=False,
    )
    return df, []