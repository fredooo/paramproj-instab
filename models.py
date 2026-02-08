"""Neural network models for parametric projection."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class MLP(nn.Module):
    """Standard multi-layer perceptron for 2D projection."""

    def __init__(self, in_dim=784, hidden_dim=512, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpecMLP(nn.Module):
    """MLP with spectral normalization on output layer for Lipschitz regularization."""

    def __init__(self, in_dim=784, hidden_dim=512, n_hidden=3):
        super().__init__()
        # Learnable output scale
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.out = spectral_norm(nn.Linear(hidden_dim, 2))

    def forward(self, x):
        return self.scale * self.out(self.backbone(x))


@torch.no_grad()
def predict(model, X, batch_size=2048, device="cpu"):
    """Batched inference, returns numpy array."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def create_model(model_cfg, in_dim):
    """Create model instance from config. use_spec: False=MLP, True=SpecMLP."""
    if model_cfg.use_spec:
        return SpecMLP(in_dim=in_dim, hidden_dim=model_cfg.hidden_dim, n_hidden=model_cfg.n_hidden)
    return MLP(in_dim=in_dim, hidden_dim=model_cfg.hidden_dim, n_hidden=model_cfg.n_hidden)


def get_model_prefix(model_cfg):
    """Generate full model prefix from config."""
    model_type = "SpecMLP" if model_cfg.use_spec else "MLP"
    base = f"nn_{model_type}_h{model_cfg.hidden_dim}_n{model_cfg.n_hidden}"
    if model_cfg.use_jac:
        return f"{base}_jac{model_cfg.lambda_jac}"
    return f"{base}_nojac"
