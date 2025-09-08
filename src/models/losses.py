from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class _Discriminator(nn.Module):
    """small tc discriminator"""

    def __init__(self, latent_dim: int, hidden: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2)
        )

    def forward(self, z):
        return self.net(z)

class ContrastiveFactorVAELoss(nn.Module):
    """info_nce + tc loss"""

    def __init__(
        self, temperature: float = 0.1, tc_weight: float = 1.0, latent_dim: int = 16
    ):
        super().__init__()
        self.temperature = temperature
        self.tc_weight = tc_weight
        self.discriminator = _Discriminator(latent_dim)
        self.bce = nn.CrossEntropyLoss()
        nn.BCEWithLogitsLoss()

    @staticmethod
    def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """info_nce""" 
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2j + loss_j2i) * 0.5

    def _total_correlation(self, z: torch.Tensor) -> torch.Tensor:
        """tc loss"""
        z_perm = torch.stack([
            z[:, dim][torch.randperm(z.size(0), device=z.device)]
            for dim in range(z.size(1))
        ], dim=1)
        logits = torch.cat([
            self.discriminator(z),
            self.discriminator(z_perm)
        ], dim=0)
        labels = torch.cat([
            torch.ones(z.size(0), dtype=torch.long, device=z.device),
            torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        ])
        return self.bce(logits, labels)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return (
            self._info_nce(z1, z2, self.temperature)
            + self.tc_weight * self._total_correlation(torch.cat([z1, z2], dim=0))
        )

# with alternating freeze steps
class FactorVaeTC(nn.Module):
    def __init__(self, latent_dim: int, tc_weight: float = 1.0, d_hidden: int = 256):
        super().__init__()
        self.tc_weight = tc_weight
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, d_hidden), nn.LeakyReLU(0.2),
            nn.Linear(d_hidden, d_hidden),    nn.LeakyReLU(0.2),
            nn.Linear(d_hidden, 1)            # single logit
        )
        self._bce_logits = nn.BCEWithLogitsLoss()

    @staticmethod
    def permute_dims(z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        assert B > 1, "TC needs B>1."
        idx = [torch.randperm(B, device=z.device) for _ in range(D)]
        return torch.stack([z[:, d][idx[d]] for d in range(D)], dim=1)

    #Discriminator loss 
    def d_loss(self, z: torch.Tensor, r1_gamma: float = 0.0) -> torch.Tensor:
        z = z.detach().requires_grad_(r1_gamma > 0)
        z_perm = self.permute_dims(z)
        logit_j = self.discriminator(z).squeeze(-1)
        logit_p = self.discriminator(z_perm).squeeze(-1)
        logits  = torch.cat([logit_j, logit_p], 0)
        labels  = torch.cat([torch.ones_like(logit_j), torch.zeros_like(logit_p)], 0)
        bce = self._bce_logits(logits, labels)
        if r1_gamma > 0:
            grad = torch.autograd.grad(logit_j.sum(), z, create_graph=True)[0]
            bce = bce + 0.5 * r1_gamma * (grad.pow(2).sum(dim=1).mean())
        return bce

    #Encoder penalty 
    def tc_penalty(self, z: torch.Tensor) -> torch.Tensor:
        logit_j = self.discriminator(z).squeeze(-1)
        tc_est  = logit_j.mean()
        return self.tc_weight * tc_est


def offdiag_cov_penalty(z):
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    C = (zc.T @ zc) / (zc.size(0) - 1)
    off = C - torch.diag(torch.diag(C))
    return (off**2).mean()