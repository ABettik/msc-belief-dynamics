import torch
import torch.nn.functional as F

class GradCAM1D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: in eval mode
            target_layer: the Conv1d layer
        """
        self.model = model
        self.activations = None
        self.gradients = None

        # hook feature maps
        target_layer.register_forward_hook(self._save_activation)
        # hook gradients wrt maps
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x: [batch, T, neurons]
        Returns:
            cams: [batch, T] ∈ [0,1]
            prpd_pred: [batch]
        """
        x_in = x.permute(0, 2, 1).requires_grad_(True)
        self.model.zero_grad()

        _, prpd_pred = self.model(x_in)

        # sum predictions to scalar loss
        loss = prpd_pred.sum()
        loss.backward(retain_graph=True)

        grads = self.gradients
        acts  = self.activations

        # channel weights
        weights = grads.mean(dim=2)

        # weighted sum over channels
        cam = (acts * weights.unsqueeze(-1)).sum(dim=1)
        cam = F.relu(cam)

        # interpolate to original time
        T = x.size(1)
        cam = F.interpolate(cam.unsqueeze(1),
            size=T,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        # normalize
        cam_min = cam.view(cam.size(0), -1).min(1)[0].unsqueeze(1)
        cam_max = cam.view(cam.size(0), -1).max(1)[0].unsqueeze(1)
        cam = (cam - cam_min)/(cam_max - cam_min + 1e-8)

        return cam, prpd_pred.detach()
