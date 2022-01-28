import torch
import torch.nn.functional as F

from Simplex.models.base import BlackBox


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
