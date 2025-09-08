from __future__ import annotations
from typing import Any
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from src.models.cebra_autoencoder import CEBRAAutoEncoder
from src.models.losses import ContrastiveFactorVAELoss

class CEBRAFactorVAE(pl.LightningModule):
    """cebra factor-vae lightning module"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        temperature: float = 0.1,
        tc_weight: float = 6.0,
        decoder_type: str | None = 'linear',
        hidden_dims: list[int] | None = None,
        freeze_encoder: bool = False,
        lr: float = 1e-3,
        contrastive_weight: float = 1.0,
        recon_weight: float = 0.0,
        lr_scheduler_args=None,
        **cebra_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters() # to store them in self.hparams
        self.model = CEBRAAutoEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
            hidden_dims=hidden_dims,
            freeze_encoder=freeze_encoder,
            **cebra_kwargs,
        )
        self.loss_fn = ContrastiveFactorVAELoss(
            temperature=temperature,
            tc_weight=tc_weight,
            latent_dim=latent_dim,
        )

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     if self.hparams.lr_scheduler_args:
    #         scheduler = {
    #             'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.hparams.lr_scheduler_args),
    #             'monitor': 'val/loss/total',
    #         }
    #         return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    #     return {'optimizer': optimizer}
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-2,
        )

        steps = self.trainer.estimated_stepping_batches
        if steps is None:
            steps = self.trainer.num_training_batches * self.trainer.max_epochs

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=opt.defaults["lr"],
            total_steps=steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def _shared_step(self, batch):
        anchor, positive, prpd = batch['anchor'], batch['positive'], batch['prpd']

        z_a, recon_a = self.model(anchor)
        z_p, recon_p = self.model(positive)

        latents = torch.cat([z_a, z_p], dim=0)
        prpd_rep = prpd.repeat(2)

        info_nce = self.loss_fn._info_nce(z_a, z_p, self.loss_fn.temperature)
        tc = self.loss_fn._total_correlation(torch.cat([z_a, z_p], dim=0)) * self.loss_fn.tc_weight
        
        recon_mse = 0.0
        if recon_a is not None:
            tgt = anchor.mean(dim=1)
            recon_mse  = F.mse_loss(recon_a, tgt) + F.mse_loss(recon_p, tgt)
        total_loss = info_nce * self.hparams.contrastive_weight + tc + recon_mse * self.hparams.recon_weight

        losses = dict(info_nce=info_nce, tc=tc, recon_mse=recon_mse, total=total_loss)

        return losses, latents, prpd_rep, None

    def training_step(self, batch, batch_idx):
        losses, latents, prpd_rep, _ = self._shared_step(batch)
        
        self.current_latents = latents
        self.current_prpd = prpd_rep
        logs = {f"train/loss/{k}": v for k, v in losses.items()}
        
        self.log_dict(logs, prog_bar=True)

        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        losses, latents, prpd_rep, _ = self._shared_step(batch)
        

        self.current_val_latents = latents
        self.current_val_prpd = prpd_rep
        logs = {f"val/loss/{k}": v for k, v in losses.items()}
        
        self.log_dict(logs, prog_bar=False)

    def encode(self, x):
        return self.model.encode(x)