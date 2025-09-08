
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from lightning import Callback, Trainer, LightningModule
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=0)

class EvaluateLatentsCallback(Callback):
    """fit a logistic reg. on current epoch latents, log prpd accuracy and non-zero features."""

    def __init__(
        self,
        alpha: float = 0.05,
        max_iter: int = 1000,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.every_n_epochs = every_n_epochs
        self._latents: list[torch.Tensor] = []
        self._labels:  list[torch.Tensor] = []
        self._val_labels:  list[torch.Tensor] = []
        self._val_latents: list[torch.Tensor] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args: Any,
        **kwargs: Any,
    ):
        if hasattr(pl_module, 'current_latents') and hasattr(pl_module, 'current_prpd'):
            self._latents.append(pl_module.current_latents.detach().cpu())
            self._labels.append(pl_module.current_prpd.detach().cpu())

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args: Any,
        **kwargs: Any,
    ):
        if hasattr(pl_module, 'current_val_latents') and hasattr(pl_module, 'current_val_prpd'):
            self._val_latents.append(pl_module.current_val_latents.detach().cpu())
            self._val_labels.append(pl_module.current_val_prpd.detach().cpu())


    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch % self.every_n_epochs or not self._latents or not self._val_latents:
            return
        to_iterate = [
            ['train', torch.cat(self._latents, dim=0).numpy(), torch.cat(self._labels,  dim=0).numpy()],
            ['val', torch.cat(self._val_latents, dim=0).numpy(), torch.cat(self._val_labels,  dim=0).numpy()],
        ]
        for t, z, y in to_iterate:
            r2_scores = []
            nnz_counts = []

            for train_idx, val_idx in kf.split(z):
                z_train, z_val = z[train_idx], z[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                clf = Lasso(alpha=self.alpha, max_iter=self.max_iter)
                clf.fit(z_train, y_train)

                y_pred = clf.predict(z_val)
                r2_scores.append(r2_score(y_val, y_pred))
                nnz_counts.append(np.count_nonzero(clf.coef_))

            acc = np.mean(r2_scores)
            nnz = np.mean(nnz_counts)

            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {f'{t}_prpd_acc': acc, f'{t}_prpd_nnz_feats': nnz},
                )
        
        self._latents: list[torch.Tensor] = []
        self._labels:  list[torch.Tensor] = []
        self._val_labels:  list[torch.Tensor] = []
        self._val_latents: list[torch.Tensor] = []
