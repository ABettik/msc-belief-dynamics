"""
This module implements the NDT1 model with MtM maksing loop.

Adapted from Cole Hurwitz’s IBL_MtM_model:
    https://github.com/colehurwitz/IBL_MtM_model/blob/e0c39bd91a499fe06690627237f6726c01a27c56/src/models/ndt1.py
Original code (c) Cole Hurwitz under the MIT License:
    https://github.com/colehurwitz/IBL_MtM_model/blob/e0c39bd91a499fe06690627237f6726c01a27c56/LICENSE

Modifications by Arsenii Petryk, 2025.
"""

from pathlib import Path
from models.ndt1_typings import LrSchedulerArgs, NdtConfig, OptimizerArgs
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.regression import R2Score
import os, sys
import lightning.pytorch as pl
from models.losses import ContrastiveFactorVAELoss, FactorVaeTC, offdiag_cov_penalty
from models.cebra_autoencoder import get_mlp
from ibl_fm.models.ndt1 import NDT1, NDT1Output
from typing import Optional, Sequence
from dataclasses import asdict, dataclass
from collections import deque

@dataclass
class lNDT1Out(NDT1Output):
    latents: Optional[torch.FloatTensor]= None
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None
    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}



class lNDT1(NDT1):
    def forward(
        self, 
        spikes:           torch.FloatTensor,  # (bs, seq_len, n_channels)
        time_attn_mask:      torch.LongTensor,   # (bs, seq_len)
        space_attn_mask:      torch.LongTensor,   # (bs, seq_len)
        spikes_timestamps: torch.LongTensor,   # (bs, seq_len)
        spikes_spacestamps: torch.LongTensor,   # (bs, seq_len)
        targets:          Optional[torch.FloatTensor] = None,  # (bs, tar_len)
        spikes_lengths:   Optional[torch.LongTensor] = None,   # (bs) 
        targets_lengths:  Optional[torch.LongTensor] = None,   # (bs)
        block_idx:        Optional[torch.LongTensor] = None,   # (bs)
        date_idx:         Optional[torch.LongTensor] = None,   # (bs)
        neuron_regions:   Optional[torch.LongTensor] = None,   # (bs, n_channels)
        masking_mode:     Optional[str] = None,
        spike_augmentation: Optional[bool] = False,
        eval_mask:        Optional[torch.LongTensor] = None,
        num_neuron:       Optional[torch.LongTensor] = None,
        eid:              Optional[str] = None,
    ) -> lNDT1Out:  

        _, _T, _ = spikes.size()
        
        # if neuron_regions type is list 
        if isinstance(neuron_regions, list):
            neuron_regions = np.asarray(neuron_regions).T

        # Augmentation
        if spike_augmentation:
            if self.training:
                # 50% of the time, we reverse the spikes
                if torch.rand(1) > 0.5:
                    # calculate unmask timestamps
                    unmask_temporal = time_attn_mask.sum(dim=1)
                    for i in range(len(unmask_temporal)):
                        # reverse idx from unmask_temporal to 0
                        reverse_idx = torch.arange(unmask_temporal[i]-1, -1, -1)
                        spikes[i, :unmask_temporal[i]] = spikes[i, reverse_idx]

        if self.method == "ssl":
            targets = spikes.clone()
            if self.encoder.int_spikes:
                targets = targets.to(torch.int64)

        # Encode neural data
        targets_mask = torch.zeros_like(spikes, dtype=torch.int64)
        x, new_mask = self.encoder(spikes, time_attn_mask, spikes_timestamps, block_idx, date_idx, neuron_regions, masking_mode, eval_mask, num_neuron, eid)
        targets_mask = targets_mask | new_mask
        spikes_lengths = self.encoder.embedder.get_stacked_lens(spikes_lengths)

        _, T, _ = x.size()

        if self.use_prompt or self.use_session:
            x = x[:,T-_T:]

        # Transform neural embeddings into rates/logits
        if self.method == "sl":
            x = x.flatten(start_dim=1)

        if hasattr(self, "stitching") and self.method == "ssl":
            outputs = self.stitch_decoder(x, str(num_neuron))
        else:
            outputs = self.decoder(x)

        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            if self.encoder.mask:
                loss = (self.loss_fn(outputs, targets) * targets_mask).sum()
            else:
                loss = self.loss_fn(outputs, targets).sum()
            n_examples = targets_mask.sum()
        elif self.method == "ctc":
            loss = self.loss_fn(outputs.transpose(0,1), targets, spikes_lengths, targets_lengths)
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)
        elif self.method == "sl":
            loss = self.loss_fn(outputs, targets).sum()
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)
        
        return lNDT1Out(
            loss=loss,
            n_examples=n_examples,
            preds=outputs,
            targets=targets,
            latents=x, # return latents for _total_correlation and prpd_mse
        )

class DictConfig(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictConfig(value)
        return value

    def get_dict(self):
        return super()

class NDT1SSLModule(pl.LightningModule):
    def __init__(
        self,
        ndt_cfg: NdtConfig,
        optimizer_args: OptimizerArgs,
        lr_scheduler_args: LrSchedulerArgs,
        method_name: str = "ssl",
        num_neurons: int = None,
        prpd_weight: float = 1.0,
        mlm_weight: float = 1.0,
        prpd_head_hidden_dims: Sequence[int] = None,
        latent_dim: int = 16,
        
        # FactorVAE TC args
        tc_warmup_frac: float = 0.1,
        d_lr: float = 1e-4,
        d_update_every: int = 1,
        perm_repeats: int = 1,
        r1_gamma: float = 0.0,

        tc_weight: float = 1.0,
        d_train_when_dloss_gt: float = 0.55,
        stdz_z: bool = True,
        use_epochwise_d: bool = True,
        d_pool_max: int = 256,
        d_phase_passes: int = 1,
        d_phase_mb: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        ndt_cfg = DictConfig(asdict(ndt_cfg))
        ndt_cfg['encoder']['transformer']['hidden_size'] = latent_dim
        # instantiate IBL's NDT1
        self.ndt = lNDT1(
            config=ndt_cfg,
            method_name=method_name,
            num_neurons=[num_neurons],
            use_lograte=True,
            loss='poisson_nll'
            # output_size: 2
            # clf: false
            # reg: false
        )
        # PRPD head pools latent belief state info
        self.prpd_head = get_mlp(latent_dim, prpd_head_hidden_dims, 1)
        self.tc_weight = tc_weight
        self.prpd_weight = prpd_weight
        self.mlm_weight = mlm_weight
        # self.factor_vae_loss = ContrastiveFactorVAELoss(
        self.factor_vae_loss = FactorVaeTC(
            tc_weight=tc_weight,
            latent_dim=latent_dim,
        )
        self.r2 = R2Score()
        
        
        # running exponential moving average for loss normalization
        if self.hparams.use_epochwise_d:
            self.register_buffer("d_loss_ema", torch.tensor(0.693)) # ln2 = 0.693 is 50% chance loss for balanced binary classifier logits
    
    def configure_optimizers(self):
        enc_opt = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_args)
        steps = self.trainer.estimated_stepping_batches or (
            self.trainer.num_training_batches * self.trainer.max_epochs
        )
        enc_sched = torch.optim.lr_scheduler.OneCycleLR(
            enc_opt, total_steps=steps, **self.hparams.lr_scheduler_args
        )
        disc_opt = torch.optim.Adam(self.factor_vae_loss.discriminator.parameters(), lr=self.hparams.d_lr)
        disc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt, T_max=steps)
        return [
            {"optimizer": enc_opt,
            "lr_scheduler": {"scheduler": enc_sched, "interval": "step", "name": "onecycle"}},
            {"optimizer": disc_opt, "lr_scheduler": {"scheduler": disc_sched, "interval": "step"}},
        ]

    def forward(self, spikes):
        B, T, N = spikes.shape
        # unmasked attention masks (will get overwritten inside NDT1.encoder.masker)
        ones = torch.ones(B, T, dtype=torch.int64, device=spikes.device)
        timestamps = torch.arange(T, dtype=torch.int64, device=spikes.device).unsqueeze(0).expand(B, -1)
        # call NDT1
        out = self.ndt(
            spikes,
            ones,       # time_attn_mask
            ones,       # space_attn_mask (we’re single‐region)
            timestamps, # spikes_timestamps
            timestamps, # spikes_spacestamps (dummy)
            masking_mode=self.ndt.encoder.masker.mode
        )
        return out
    
    def _shared_step(self, batch):
        spikes = batch['spikes']             # [B, T, N]
        true_prpd = batch['prpd']            # [B]
        out = self(spikes)
        
        
        
        # ============ raw variant ============
        
        
        # mlm_loss = out.loss / out.n_examples
        # z = out.latents.mean(dim=1)
        # tc_loss = self.factor_vae_loss._total_correlation(z) * self.factor_vae_loss.tc_weight
        # # tc_loss = offdiag_cov_penalty(z) * self.factor_vae_loss.tc_weight
        # prpd_pred = self.prpd_head(z).squeeze(-1)
        # prpd_loss = F.mse_loss(prpd_pred, true_prpd)

        # total_loss = self.mlm_weight * mlm_loss + self.tc_weight * tc_loss + self.prpd_weight * prpd_loss


        # losses = dict(
        #     mlm_loss = mlm_loss,
        #     tc = tc_loss,
        #     prpd_mse = prpd_loss,
        #     total = total_loss
        # )
        # return losses, true_prpd, prpd_pred
        # ============ raw variant end ============

        # ============ Encoder/Discriminator freez variant ============

        
        n_ex = getattr(out, "n_examples", 1) or 1

        mlm_loss = out.loss / n_ex
        z = out.latents.mean(dim=1)     # [B,D] pooled latent
        prpd_pred = self.prpd_head(z).squeeze(-1)  # [B]
        prpd_loss = F.mse_loss(prpd_pred, true_prpd)
        
        return z, mlm_loss, prpd_loss, prpd_pred, true_prpd

        
    # ============ Encoder/Discriminator freez variant end ============
    def _z_for_tc(self, z):
        return (z - z.mean(0)) / (z.std(0) + 1e-6) if self.hparams.stdz_z else z

    def training_step(self, batch, batch_idx):
        enc_opt, disc_opt = self.optimizers()
        scheds = self.lr_schedulers()
        enc_sched = scheds[0] if isinstance(scheds, (list, tuple)) else scheds
        disc_sched = scheds[1] if isinstance(scheds, (list, tuple)) and len(scheds) > 1 else None

        # ----- forward once (fine since D loss detaches z) -----
        z, mlm_loss, prpd_loss, _, _ = self._shared_step(batch)
        z_tc = self._z_for_tc(z)
        if self.hparams.use_epochwise_d:
            if not hasattr(self, "_z_pool"): self._z_pool = deque(maxlen=self.hparams.d_pool_max)
            self._z_pool.append(z_tc.detach())
        elif (batch_idx % self.hparams.d_update_every) == 0:
            for p in self.factor_vae_loss.discriminator.parameters():
                p.requires_grad_(True)
            self.factor_vae_loss.discriminator.train()

            z_tc = z_tc.detach()# for D update
            d_loss = 0.0
            for _ in range(self.hparams.perm_repeats):
                d_loss = d_loss + self.factor_vae_loss.d_loss(z_tc, r1_gamma=self.hparams.r1_gamma)
            d_loss = d_loss / self.hparams.perm_repeats

            if d_loss.item() > self.hparams.d_train_when_dloss_gt:
                disc_opt.zero_grad(set_to_none=True)
                self.manual_backward(d_loss)
                disc_opt.step()
                if disc_sched: disc_sched.step()
            self.log("train/loss/d_tc", d_loss, on_step=True)

        # ----- Encoder step (D frozen) -----
        for p in self.factor_vae_loss.discriminator.parameters():
            p.requires_grad_(False)
        z_tc = self._z_for_tc(z)
        tc_raw = self.factor_vae_loss.tc_penalty(z_tc)
        t = self.global_step / max(1, self.trainer.estimated_stepping_batches - 1)
        tc_w = self.tc_weight * min(1.0, t / self.hparams.tc_warmup_frac)
        tc_pen = tc_w * tc_raw
        total  = self.mlm_weight * mlm_loss + self.prpd_weight * prpd_loss + tc_w * tc_pen

        enc_opt.zero_grad(set_to_none=True)
        self.manual_backward(total)
        enc_opt.step()
        if enc_sched is not None:
            enc_sched.step()

        self.log_dict(
            {
                "train/loss/mlm": mlm_loss,
                "train/loss/prpd": prpd_loss,
                "train/loss/tc": tc_pen,
                "train/loss/total": total,
            },
            on_step=True,
            prog_bar=True,
        )
        return total

    # ============ Encoder/Discriminator freez variant end ============
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        z, mlm_loss, prpd_loss, prpd_pred, true_prpd = self._shared_step(batch)
        # TC using frozen D
        self.factor_vae_loss.discriminator.eval()
        for p in self.factor_vae_loss.discriminator.parameters():
            p.requires_grad_(False)
        tc_eval = self.factor_vae_loss.tc_penalty(z)
        r2 = self.r2(prpd_pred, true_prpd)
        
        z_tc = self._z_for_tc(z)
        tc_raw = self.factor_vae_loss.tc_penalty(z_tc)
        t = self.global_step / max(1, self.trainer.estimated_stepping_batches - 1)
        tc_w = self.tc_weight * min(1.0, t / self.hparams.tc_warmup_frac)
        tc_pen = tc_w * tc_raw
        
        total  = self.mlm_weight * mlm_loss + self.prpd_weight * prpd_loss + tc_w * tc_pen
        
        self.log_dict(
            {
                "val/loss/mlm": mlm_loss, "val/loss/prpd": prpd_loss,
                "val/loss/tc": tc_eval, "val/prpd_r2": r2,
                "val/loss/total": total,
             },
            prog_bar=False
        )
    @torch.no_grad()
    def _concat_pool(self):
        if not hasattr(self, "_z_pool") or len(self._z_pool) == 0:
            return None
        z_all = torch.cat(list(self._z_pool), dim=0)
        return z_all.to(self.device)

    def on_train_epoch_end(self):
        if not self.hparams.use_epochwise_d:
            return
        z_all = self._concat_pool()
        if z_all is None or z_all.size(0) < 8:
            return

        #grads D/no grads to encoder
        for p in self.factor_vae_loss.discriminator.parameters():
            p.requires_grad_(True)
        self.factor_vae_loss.discriminator.train()

        # discriminator optimization and scheduling
        opts = self.optimizers()
        disc_opt  = opts[1] if isinstance(opts, (list, tuple)) else None
        scheds = self.lr_schedulers()
        disc_sched = None
        if isinstance(scheds, (list, tuple)) and len(scheds) > 1:
            disc_sched = scheds[1]

        # training using collected latent pool
        for _ in range(self.hparams.d_phase_passes):
            for start in range(0, z_all.size(0), self.hparams.d_phase_mb):
                z_mb = z_all[start:start+self.hparams.d_phase_mb].detach().requires_grad_(False)
                d_loss = self.factor_vae_loss.d_loss(z_mb)

                if d_loss.item() <= self.hparams.d_train_when_dloss_gt:
                    break

                disc_opt.zero_grad(set_to_none=True)
                for p in self.factor_vae_loss.discriminator.parameters():
                    p.requires_grad_(True)
                self.manual_backward(d_loss)
                disc_opt.step()
                if disc_sched is not None:
                    disc_sched.step()

                # log d_loss
                self.d_loss_ema = 0.9*self.d_loss_ema + 0.1*d_loss.detach()
                self.log("train/epoch_d_loss", self.d_loss_ema, prog_bar=False, on_epoch=True)

        # freeze D back
        for p in self.factor_vae_loss.discriminator.parameters():
            p.requires_grad_(False)

        self._z_pool.clear()

    # ============ raw variant ============
    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     losses, _, _ = self._shared_step(batch)

    #     logs = {f"train/loss/{k}": v for k, v in losses.items()}
        
    #     self.log_dict(logs, prog_bar=True)
        
        
    #     # ----- Encoder step (TC DISCRIMINATOR FROZEN) -----
    #     if optimizer_idx == 0:
    #         for p in self.factor_vae_loss.discriminator.parameters(): p.requires_grad_(False)
            
        
    #     return losses['total']
    
    # ============ raw variant ============
    # def validation_step(self, batch):
    #     losses, true_prpd, prpd_pred = self._shared_step(batch)

    #     logs = {f"val/loss/{k}": v for k, v in losses.items()}
        
    #     r2 = self.r2(prpd_pred, true_prpd)
    #     logs['val/prpd_r2'] = r2
        
    #     self.log_dict(logs, prog_bar=False)
    #     return {'val/prpd_r2': r2}
    # self.ndt.encoder.masker
    
    
    
    # ============ Encoder/Discriminator freez variant ============
    def on_validation_start(self):
        # disable masking
        m = self.ndt.encoder.masker
        if m is not None:
            self._masker_state = (m.mode, m.ratio, m.force_active)
            m.force_active = True
            m.ratio = 0.01

    def on_validation_end(self):
        # restore masking
        m = self.ndt.encoder.masker
        if m is not None and hasattr(self, "_masker_state"):
            m.mode, m.ratio, m.force_active = self._masker_state


