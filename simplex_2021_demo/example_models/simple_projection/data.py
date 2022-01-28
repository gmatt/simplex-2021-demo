import numpy as np
import pandas as pd

df = pd.DataFrame(
    columns=["f1", "f2", "label"],
    data=[
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, -1, 1],
        [0, -1, 1],
        [-1, -1, 1],
        [-1, 0, 1],
        [-1, 1, 1],
        [0, 0, 0],
        [0, 0.1, 0],
        [0.1, 0, 0],
        [0, -0.1, 0],
        [-0.1, 0, 0],
    ],
)

X = df.drop(columns=["label"]).values
y = df["label"].values
feature_names = [x for x in df.columns if x != "label"]
input_baseline = np.zeros_like(X)
