from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import Subset, DataLoader
import lightning.pytorch as pl
from src.data.spike_datamodule import _SpikePairDataset

class SpikeCVDataModule(pl.LightningDataModule):
    def __init__(self, session_path, prpd_path,
            fold_id, n_splits=5,
            batch_size=32, num_workers=4, contrastive=True):
        super().__init__()
        self.session_path, self.prpd_path = Path(session_path), Path(prpd_path)
        self.fold_id, self.n_splits = fold_id, n_splits
        self.batch_size, self.num_workers = batch_size, num_workers
        self.contrastive = contrastive

    def setup(self, stage=None):
        spikes = np.load(self.session_path)["arr_0"]
        prpd = np.load(self.prpd_path)["arr_0"]
        full = _SpikePairDataset(spikes, prpd, self.contrastive)

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        train_idx, val_idx = list(kf.split(spikes))[self.fold_id]
        self.train_ds = Subset(full, train_idx)   # not used, but required
        self.val_ds = Subset(full, val_idx)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )