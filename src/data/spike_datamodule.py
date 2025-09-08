from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

class _SpikePairDataset(Dataset):
    def __init__(
        self,
        spikes: np.ndarray,
        prpd: np.ndarray,
        contrastive: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        assert len(spikes) == len(prpd)
        self.spikes = torch.as_tensor(spikes, dtype=torch.float32)
        self.prpd = torch.as_tensor(prpd, dtype=torch.float32)
        self.contrastive = contrastive
        self.transform = transform

    def __len__(self):
        return self.spikes.shape[0]

    def __getitem__(self, idx: int):
        x = self.spikes[idx]
        prpd = self.prpd[idx]
        if not self.contrastive:
            return {'spikes': x, 'prpd': prpd}
        anchor = x + torch.randn_like(x) * 0.05
        shift = torch.randint(-1, 2, ()).item()
        anchor = anchor.roll(shift, dims=0)
        mask = torch.rand_like(x) < 0.2
        anchor = anchor.masked_fill(mask, 0)
        positive = x + torch.randn_like(x) * 0.05
        positive = positive.roll(torch.randint(-1, 2, ()).item(), dims=0)
        positive = positive.masked_fill(torch.rand_like(x) < 0.2, 0)
        return {'anchor': anchor, 'positive': positive, 'prpd': prpd}

class SpikeDataModule(pl.LightningDataModule):
    """lightning module for spikes"""

    def __init__(
        self,
        session_path: str | Path = 'data/processed/spikes.npz',
        prpd_path: str | Path = 'data/processed/prpd.npz',
        batch_size: int = 32,
        num_workers: int = 4,
        contrastive: bool = True,
        shift_prpd_by_one: bool = False,
        fr_scale: bool = True,
        max_spikes: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.session_path = Path(session_path)
        self.prpd_path = Path(prpd_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.contrastive = contrastive
        self.fr_scale = fr_scale
        self.max_spikes = max_spikes
        self.shift_prpd_by_one = shift_prpd_by_one
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        if hasattr(self, 'train_ds'):
            return
        spikes = np.load(self.session_path)['arr_0']
        prpd = np.load(self.prpd_path)['arr_0']
        
        if not self.fr_scale: spikes = spikes*self.max_spikes
        if self.shift_prpd_by_one: prpd[1:] = prpd[:-1]
        
        n = spikes.shape[0]
        idx = np.random.RandomState(0).permutation(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        self.train_ds = _SpikePairDataset(
            spikes[idx[:n_train]],
            prpd[idx[:n_train]],
            contrastive=self.contrastive,
        )
        val_spikes = spikes[idx[n_train:n_train+n_val]]
        val_prpd = prpd[idx[n_train:n_train+n_val]]
        self.val_ds = _SpikePairDataset(
            val_spikes,
            val_prpd,
            contrastive=self.contrastive,
        )
        self.test_ds = _SpikePairDataset(
            spikes[idx[n_train+n_val:]],
            prpd[idx[n_train+n_val:]],
            contrastive=self.contrastive,
        )
        
        # self.train_ds, self.val_ds, self.test_ds = data.random_split(
        #     _SpikePairDataset(spikes, prpd, contrastive=self.contrastive), [n_train, n_val, n-n_train-n_val], generator=torch.Generator()
        # )

    def train_dataloader(self):
        return DataLoader(self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)