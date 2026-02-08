# train.py
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from typedefs import TrainData, TrainingConfig, TrainingResult


def create_loader(X, Z, batch_size, shuffle=True):
    """Create DataLoader from numpy arrays."""
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Z)),
        batch_size=batch_size, shuffle=shuffle
    )


def train_epoch(model, loader, optimizer, loss_fn, device, use_jacobian=False, lambda_jac=1.0):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for xb, zb in loader:
        xb, zb = xb.to(device).float(), zb.to(device).float()
        optimizer.zero_grad()

        if use_jacobian:
            xb.requires_grad_(True)

        pred = model(xb)
        loss_fit = loss_fn(pred, zb)

        if use_jacobian:
            loss_jac = 0
            for j in range(2):
                g = torch.autograd.grad(pred[:, j].sum(), xb, create_graph=True)[0]
                loss_jac += (g ** 2).mean()
            loss = loss_fit + lambda_jac * loss_jac
        else:
            loss = loss_fit

        loss.backward()
        optimizer.step()
        total_loss += loss_fit.item() * xb.size(0)
        n_samples += xb.size(0)

    return total_loss / n_samples


def validate_epoch(model, loader, loss_fn, device):
    """Run validation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, zb in loader:
            xb, zb = xb.to(device).float(), zb.to(device).float()
            total_loss += loss_fn(model(xb), zb).item() * xb.size(0)
            n_samples += xb.size(0)

    return total_loss / n_samples


def train_projection_model(model, train_data: TrainData, device, cfg: TrainingConfig,
                           use_jacobian=False, lambda_jac=0.0):
    """Train model with early stopping. Returns TrainingResult namedtuple."""
    start_time = time.time()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_loader = create_loader(train_data.X_tr, train_data.Z_tr, cfg.batch_size, shuffle=True)
    val_loader = create_loader(train_data.X_val, train_data.Z_val, cfg.batch_size, shuffle=False)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    train_loss = 0.0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                 use_jacobian, lambda_jac)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:03d} | Train MSE {train_loss:.6f} | Val MSE {val_loss:.6f}")

        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, model.state_dict(), 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    training_time = time.time() - start_time

    return TrainingResult(
        model=model,
        best_val_loss=best_val,
        final_train_loss=train_loss,
        epochs=epoch,
        early_stopped=(patience_ctr >= cfg.patience),
        training_time=training_time
    )


def evaluate_projection_model(model, X_test, Z_test, device, batch_size=256):
    """Evaluate a trained projection model using MSE loss."""
    model = model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()

    test_loader = create_loader(X_test, Z_test, batch_size, shuffle=False)

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, zb in test_loader:
            xb, zb = xb.to(device).float(), zb.to(device).float()
            total_loss += loss_fn(model(xb), zb).item() * xb.size(0)
            n_samples += xb.size(0)

    return total_loss / n_samples
