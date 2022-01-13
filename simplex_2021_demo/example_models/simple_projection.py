import pandas as pd
import torch
import torch.nn.functional as F

from models.base import BlackBox

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


class Model(BlackBox):
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.vstack(
            [
                torch.atan2(x[:, 0], x[:, 1]),
                torch.norm(x, dim=1),
            ]
        ).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        return F.sigmoid(x[:, 1] - 0.5)


model = Model()
