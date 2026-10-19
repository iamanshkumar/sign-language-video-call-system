"""Training loop: AdamW + CrossEntropyLoss, ReduceLROnPlateau and early stopping on
validation macro-F1, best-checkpoint saving. The test split is never passed in here."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from asl.models.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    if cfg["optimizer"] != "adamw":
        raise ValueError(f"unsupported optimizer {cfg['optimizer']!r}")
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])


def make_scheduler(opt: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    s = cfg["scheduler"]
    if s["name"] != "reduce_lr_on_plateau":
        raise ValueError(f"unsupported scheduler {s['name']!r}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=s["mode"], factor=s["factor"], patience=s["patience"])


@dataclass
class BestTracker:
    """Strict improvement only (ties keep the earlier epoch); stop after `patience` epochs
    without improvement."""

    patience: int
    best: float = field(default=-float("inf"))
    best_epoch: int | None = None
    bad_epochs: int = 0

    def update(self, epoch: int, metric: float) -> tuple[bool, bool]:
        """Returns (improved, stop)."""
        improved = metric > self.best
        if improved:
            self.best, self.best_epoch, self.bad_epochs = metric, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def save_checkpoint(path: Path, model: nn.Module, **meta) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), **meta}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module) -> dict:
    """Loads weights into `model`; returns the stored metadata (epoch, val_macro_f1, ...)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck.pop("state_dict"))
    return ck


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """(y_true, y_pred, mean loss) in eval mode."""
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


def _train_epoch(model, loader, opt, device) -> tuple[float, float]:
    """One epoch; returns (mean loss, macro-F1 of the training-mode predictions)."""
    model.train()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_sum += float(loss.detach()) * len(y)
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(logits.detach().argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return loss_sum / n, classification_metrics(y, p, 0)["macro_f1"]


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None, cfg: dict, out_dir: Path,
        device: torch.device, max_epochs: int | None = None, allow_no_validation: bool = False) -> dict:
    """Train; with validation, save `best.pt` on every strict val macro-F1 improvement and
    stop early. Without validation (explicit opt-in only), save `final.pt`, no selection."""
    if val_loader is None and not allow_no_validation:
        raise ValueError("no validation data: refusing to train without model selection "
                         "(pass allow_no_validation=True explicitly)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = max_epochs or cfg["max_epochs"]
    model.to(device)
    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, cfg)
    tracker = BestTracker(cfg["early_stopping_patience"])
    history, stopped = [], False
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        lr = opt.param_groups[0]["lr"]
        train_loss, train_f1 = _train_epoch(model, train_loader, opt, device)
        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_macro_f1": train_f1,
               "val_loss": None, "val_macro_f1": None}
        if val_loader is not None:
            y, p, val_loss = predict(model, val_loader, device)
            val_f1 = classification_metrics(y, p, 0)["macro_f1"]
            row.update(val_loss=val_loss, val_macro_f1=val_f1)
            sch.step(val_f1)
            improved, stopped = tracker.update(epoch, val_f1)
            if improved:
                save_checkpoint(out_dir / "best.pt", model, epoch=epoch, val_macro_f1=val_f1)
        row["seconds"] = time.perf_counter() - t0
        history.append(row)
        if stopped:
            break
    if val_loader is None:
        save_checkpoint(out_dir / "final.pt", model, epoch=len(history), val_macro_f1=None)
        return {"history": history, "checkpoint": "final.pt", "best_epoch": None, "best_val_macro_f1": None,
                "stopped_early": False, "epochs_run": len(history), "checkpoint_epoch": len(history),
                "max_epochs_used": max_epochs, "selection": "none (no validation data; final epoch saved)"}
    return {"history": history, "checkpoint": "best.pt", "best_epoch": tracker.best_epoch,
            "best_val_macro_f1": tracker.best, "stopped_early": stopped, "epochs_run": len(history),
            "checkpoint_epoch": tracker.best_epoch, "max_epochs_used": max_epochs,
            "selection": "best validation macro-F1"}
