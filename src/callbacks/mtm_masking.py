"""
This module implements the MtM alternating SSL loop.

Adapted from Cole Hurwitz’s IBL_MtM_model:
    https://github.com/colehurwitz/IBL_MtM_model/blob/e0c39bd91a499fe06690627237f6726c01a27c56/src/trainer/base.py
Original code (c) Cole Hurwitz under the MIT License:
    https://github.com/colehurwitz/IBL_MtM_model/blob/e0c39bd91a499fe06690627237f6726c01a27c56/LICENSE

Modifications by Arsenii Petryk, 2025.
"""
import torch
from random import sample
from lightning.pytorch import Callback

class MtMMaskingCallback(Callback):
    """
    randomly alternates masking schemas every batch (aka MtM trainin loop)
    """
    
    def __init__(
        self,
        masking_schemes: list[str] = ['neuron', 'causal'],
        masking_ratio: float = 0.3, # for neuron masking_schemes
    ):
        super().__init__()
        self.masking_schemes = masking_schemes
        self.masking_ratio = masking_ratio
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):        
        masking_mode = sample(self.masking_schemes, 1)[0]

        if masking_mode == 'temporal':
            pl_module.ndt.encoder.masker.ratio = 0.3
        elif masking_mode == 'causal':
            pl_module.ndt.encoder.masker.ratio = 0.6
        else:
            pl_module.ndt.encoder.masker.ratio = self.masking_ratio
        pl_module.ndt.encoder.masker.mode = masking_mode
