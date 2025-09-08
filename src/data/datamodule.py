from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

class NeuropixelsDataset(Dataset):
    """dataset for neuropixels spikes"""

    def __init__(
        self,
        spike_array: np.ndarray,
        label_array: np.ndarray | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.spikes = torch.as_tensor(spike_array, dtype=torch.float32)
        self.labels = (
            torch.as_tensor(label_array, dtype=torch.long) if label_array is not None else None
        )
        self.transform = transform

    def __len__(self) -> int:
        return self.spikes.shape[0]

    def __getitem__(self, idx: int):
        x = self.spikes[idx]
        if self.transform:
            x = self.transform(x)
        sample = {'spikes': x}
        if self.labels is not None:
            sample['labels'] = self.labels[idx]
        return sample

class NeuropixelsDataModule(pl.LightningDataModule):
    """lightning data module for neuropixels"""

    def __init__(
        self,
        data_dir: str | Path = Path('data/processed'),
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transforms

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        if hasattr(self, 'train_ds'):
            return
        spikes = np.load(self.data_dir / 'spikes.npz')['arr_0']
        labels_path = self.data_dir / 'prpd.npz'
        labels = np.load(labels_path)['arr_0'] if labels_path.exists() else None

        with open(self.data_dir / 'splits.json', 'r') as fp:
            splits: dict[str, Sequence[int]] = json.load(fp)

        self.train_ds = NeuropixelsDataset(
            spikes[splits['train']],
            labels[splits['train']] if labels is not None else None,
            transform=self.transforms,
        )
        self.val_ds = NeuropixelsDataset(
            spikes[splits['val']],
            labels[splits['val']] if labels is not None else None,
            transform=None,
        )
        self.test_ds = NeuropixelsDataset(
            spikes[splits['test']],
            labels[splits['test']] if labels is not None else None,
            transform=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def static_dim(self):
        sample = self.train_ds[0]['spikes']
        return sample.shape[-2:]