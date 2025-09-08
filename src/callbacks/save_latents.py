from __future__ import annotations
from pathlib import Path
from lightning import Callback
import torch

class SaveLatentsCallback(Callback):
    """save latents each epoch"""

    def __init__(self, every_n_epochs: int = 1):
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        self._latents = []
        self._labels = []

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if hasattr(pl_module, 'current_latents'):
            self._latents.append(pl_module.current_latents.cpu())
            if hasattr(pl_module, 'current_prpd'):
                self._labels.append(pl_module.current_prpd.cpu())

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs:
            return
        if not self._latents:
            return
        latents = torch.cat(self._latents, dim=0)
        prpd = torch.cat(self._labels, dim=0) if self._labels else None
        out_dir = Path(trainer.logger.save_dir) / 'latents'
        out_dir.mkdir(exist_ok=True, parents=True)
        torch.save({'z': latents, 'prpd': prpd}, out_dir / f'epoch_{trainer.current_epoch:03d}.pt')
