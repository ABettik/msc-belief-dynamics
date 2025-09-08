from typing import Any, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import R2Score
from src.models.cebra_factorvae import CEBRAFactorVAE


class CEBRAFactorReg(CEBRAFactorVAE):
    """
    CEBRA backbone + FactorVAE disentangling + MLP prpd regressor.
    """

    def __init__(
        self,
        prpd_weight: float = 1.0,
        prpd_warmup: int = 10,
        prpd_head_hidden_dims: Sequence[int] | None = None,
        latent_dim: int = 16,
        **cebra_kwargs: Any,
    ):
        super().__init__(latent_dim=latent_dim, **cebra_kwargs)
        self.save_hyperparameters() # to store them in self.hparams

        self.prpd_head = self.model.mlp(latent_dim, prpd_head_hidden_dims, 1)
        self.r2 = R2Score()

    def forward(self, x: torch.Tensor):
        z, _ = self.model(x)
        return z, self.prpd_head(z).squeeze(-1)

    def _shared_step(self, batch):
        losses, z, prpd_rep, _ = super()._shared_step(batch)

        disent_loss = losses['info_nce'] * self.hparams.contrastive_weight + losses['tc'] + losses['recon_mse'] * self.hparams.recon_weight

        # prpd regression loss
        pred_prpd = self.prpd_head(z).squeeze(-1)
        mse = F.mse_loss(pred_prpd, prpd_rep)

        # warm-up scheduling
        warmup_factor = float(self.current_epoch >= self.hparams.prpd_warmup)

        losses['prpd_mse'] = mse
        # rewrite total loss
        losses['total'] = disent_loss + warmup_factor * self.hparams.prpd_weight * mse
        return losses, z, prpd_rep, pred_prpd

    def validation_step(self, batch, _):
        losses, latents, prpd_rep, pred = self._shared_step(batch)
        self.current_val_latents = latents
        self.current_val_prpd = prpd_rep
        r2 = self.r2(pred, prpd_rep)
        
        logs = {f"val/loss/{k}": v for k, v in losses.items()}
        logs['val/prpd_r2'] = r2
        self.log_dict(logs, prog_bar=False)
        return { 'val/prpd_r2': r2 }

    test_step = validation_step
