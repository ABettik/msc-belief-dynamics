from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

class LatentCovarianceCallback(Callback):
    """callback to log latent covariance and its off‑diagonal magnitude to mlflow."""
    def __init__(self, save_matrix=True):
        self.save_matrix = save_matrix
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self._latents = []

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        if hasattr(pl_module, 'current_val_latents') and pl_module.current_val_latents is not None:
            self._latents.append(pl_module.current_val_latents.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._latents:
            return
        latents = torch.cat(self._latents, dim=0)
        cov = torch.cov(latents.T)
        off_diag = cov - torch.diag(torch.diag(cov))
        mean_off = off_diag.abs().mean().item()

        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            run_id = trainer.logger.run_id
            exp = trainer.logger.experiment
            exp.log_metric(run_id, key='latent_cov_offdiag', value=mean_off, step=trainer.current_epoch)

            if self.save_matrix:
                artefact_dir = Path(trainer.logger.save_dir) / trainer.logger.experiment_id / run_id / 'artifacts'
                artefact_dir.mkdir(parents=True, exist_ok=True)
                path = artefact_dir / f'cov_epoch_{trainer.current_epoch:03d}.npy'
                np.save(path, cov.numpy())
                exp.log_artifact(run_id, local_path=str(path), artifact_path='cov_matrices')