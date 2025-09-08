import torch
import torch.nn.functional as F

class PerNeuronImpartanceCEBRAFactorReg:
    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: CEBRAFactorReg in eval() mode
        """
        self.model = model

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x: raw trace‐view input
        Returns:
            impotance: absolute-gradient wrt input
            prpd_pred: model outputs
        """
        x_in = x.permute(0,2,1).requires_grad_(True)
        self.model.zero_grad()

        _, prpd_pred = self.model(x_in)

        # sum to get a scalar loss
        loss = prpd_pred.sum()
        loss.backward()

        # gradient of loss wrt input
        impotance = x_in.grad.detach().abs()

        # flatten time and normalize
        flat = impotance.view(impotance.size(0), -1)
        minval = flat.min(dim=1)[0].view(-1,1,1)
        maxval = flat.max(dim=1)[0].view(-1,1,1)
        impotance = (impotance - minval) / (maxval - minval + 1e-8)

        impotance = impotance.permute(0,2,1)

        return impotance, prpd_pred.detach()

class PerNeuronImpartanceNDTFactorMLP:
    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: NDT1 in eval() mode
        """
        self.model = model

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x: raw trace‐view input
        Returns:
            impotance: absolute-gradient wrt input
            prpd_pred: model outputs
        """
        x_in = x.requires_grad_(True)
        self.model.zero_grad()

        out = self.model(x_in)
        prpd_pred = self.model.prpd_head(out.latents.mean(dim=1)).squeeze(-1)

        # sum to get a scalar loss
        loss = prpd_pred.sum()
        loss.backward()

        # gradient of loss wrt input
        impotance = x_in.grad.detach().abs()

        # flatten time and normalize
        flat = impotance.view(impotance.size(0), -1)
        minval = flat.min(dim=1)[0].view(-1,1,1)
        maxval = flat.max(dim=1)[0].view(-1,1,1)
        impotance = (impotance - minval) / (maxval - minval + 1e-8)

        impotance = impotance.permute(0,2,1)

        return impotance, prpd_pred.detach()