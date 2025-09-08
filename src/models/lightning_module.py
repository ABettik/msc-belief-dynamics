from lightning import LightningModule
import torch

class NeuropixelsSSL(LightningModule):
    """initial model"""

    def __init__(self, model: torch.nn.Module, loss_fn, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['spikes']
        z = self(x)
        loss = self.loss_fn(z)
        self.log('train/loss', loss, prog_bar=True)
        self.current_latents = z
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)