from __future__ import annotations
from typing import Literal, Sequence
import torch
import torch.nn as nn
from cebra.models import init as cebra_init

def get_mlp(input_dim: int, hidden_dims: Sequence[int], out_dim: int):
    hidden_dims = list(hidden_dims)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)

class CEBRAAutoEncoder(nn.Module):
    """cebra-based auto-encoder"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        decoder_type: Literal['linear', 'mlp'] = 'linear',
        hidden_dims: Sequence[int] | None = None,
        freeze_encoder: bool = False,
        **cebra_kwargs,
    ):
        super().__init__()
        self.encoder = cebra_init(
            cebra_kwargs.pop('model_name', 'offset1-model'),
            num_neurons=input_dim,
            num_units=cebra_kwargs.pop('num_hidden_units', 32),
            num_output=latent_dim,
        )
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        if decoder_type == 'linear':
            self.decoder = nn.Linear(latent_dim, input_dim)
        elif decoder_type == 'mlp':
            self.decoder = self.mlp(latent_dim, hidden_dims, input_dim)
        else:
            raise ValueError(f'unknown decoder_type: {decoder_type!r}')

    def mlp(self, input_dim: int, hidden_dims: Sequence[int], out_dim: int):
        return get_mlp(input_dim, hidden_dims, out_dim)
    
    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        z = self.encode(x)
        recon = self.decode(z) if hasattr(self, 'decoder') else None
        return z, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.size())
        # if x.ndim == 3 and not isinstance(self.encoder, torch.nn.ModuleList):
        #     if x.size(-1) > 1: x = x.mean(dim=-1, keepdim=True)
        z = self.encoder(x)
        if z.ndim == 3: z = z.mean(dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)