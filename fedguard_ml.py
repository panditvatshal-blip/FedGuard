"""Small, real PyTorch FedAvg + per-example clipped/noised SGD demo.

All built-in data are synthetic tabular data. Never point this demo at patient data.
The accountant uses a conservative Gaussian-mechanism composition bound (no
subsampling amplification), not a clinical/production privacy certification.
"""
from __future__ import annotations

import hashlib
import io
import math
from typing import Dict, List, Tuple

import torch
from torch import nn

SEED = 20261026
FEATURES = 8
CLIENTS = 3
SAMPLES_PER_CLIENT = 24
BATCH_SIZE = 8
LOCAL_EPOCHS = 3
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

torch.set_num_threads(1)


class DiagnosticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def synthetic_client_data(client_id: int, n: int = SAMPLES_PER_CLIENT, seed: int = SEED):
    """Deterministic synthetic binary classification data; no real medical records."""
    gen = torch.Generator().manual_seed(seed + client_id * 997)
    x = torch.randn(n, FEATURES, generator=gen)
    # Slight client-specific distribution shift mimics non-IID institutions.
    x[:, client_id % FEATURES] += (client_id - 1) * 0.35
    weights = torch.tensor([0.8, -0.6, 0.5, 0.2, -0.3, 0.7, -0.4, 0.25])
    logits = x @ weights + (client_id - 1) * 0.12
    y = (logits > 0).long()
    return x, y


def state_to_arrays(model: nn.Module) -> List:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def arrays_to_state(model: nn.Module, arrays: List):
    keys = list(model.state_dict().keys())
    state = {k: torch.as_tensor(a, dtype=model.state_dict()[k].dtype) for k, a in zip(keys, arrays)}
    model.load_state_dict(state, strict=True)


def model_bytes(model: nn.Module) -> bytes:
    buff = io.BytesIO()
    torch.save(model.state_dict(), buff)
    return buff.getvalue()


def model_hash(model: nn.Module) -> str:
    return hashlib.sha256(model_bytes(model)).hexdigest()


def load_model_state(blob: bytes | None) -> DiagnosticNet:
    model = DiagnosticNet()
    if blob:
        try:
            state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
            model.load_state_dict(state)
        except (ValueError, RuntimeError, TypeError):
            # A malformed stored state must not crash the entire app; use a fresh model.
            pass
    return model


def _dp_local_train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, noise_multiplier: float):
    """Per-example gradient clipping + Gaussian noise, implemented in PyTorch."""
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    n = len(x)
    optimizer_steps = 0
    model.train()
    for start in range(0, n, BATCH_SIZE):
        xb, yb = x[start:start + BATCH_SIZE], y[start:start + BATCH_SIZE]
        clipped_sums = [torch.zeros_like(p) for p in params]
        for j in range(len(xb)):
            model.zero_grad(set_to_none=True)
            loss = criterion(model(xb[j:j + 1]), yb[j:j + 1])
            grads = torch.autograd.grad(loss, params)
            norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
            scale = min(1.0, MAX_GRAD_NORM / (float(norm) + 1e-12))
            for total, grad in zip(clipped_sums, grads):
                total.add_(grad.detach(), alpha=scale)
        batch_n = len(xb)
        with torch.no_grad():
            for p, total in zip(params, clipped_sums):
                noise = torch.randn_like(total) * (noise_multiplier * MAX_GRAD_NORM)
                p.grad = (total + noise) / batch_n
        # SGD update after the noisy average gradient.
        with torch.no_grad():
            for p in params:
                p.add_(p.grad, alpha=-0.08)
                p.grad = None
        optimizer_steps += 1
    return optimizer_steps


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.cross_entropy(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return accuracy, loss


def run_federated_round(noise_multiplier: float, previous_state: bytes | None = None):
    """Train clients locally, FedAvg their weights, return metrics/state/accounting."""
    if not 0.5 <= noise_multiplier <= 2.0:
        raise ValueError('Noise multiplier must be between 0.5 and 2.0')
    global_model = load_model_state(previous_state)
    base_arrays = state_to_arrays(global_model)
    client_states = []
    total_steps = 0
    # Three logical hospital clients, each only receives its own synthetic dataset.
    for client_id in range(CLIENTS):
        local_model = DiagnosticNet()
        arrays_to_state(local_model, base_arrays)
        x, y = synthetic_client_data(client_id)
        steps = 0
        for _ in range(LOCAL_EPOCHS):
            steps += _dp_local_train(local_model, x, y, noise_multiplier)
        total_steps += steps
        client_states.append((len(x), state_to_arrays(local_model)))
    # FedAvg: sample-count weighted aggregation of local model parameters.
    denom = sum(count for count, _ in client_states)
    averaged = []
    for layer_idx in range(len(base_arrays)):
        averaged.append(sum((count / denom) * torch.as_tensor(arrays[layer_idx])
                            for count, arrays in client_states))
    arrays_to_state(global_model, [a.numpy() for a in averaged])
    # Independent synthetic evaluation set.
    eval_parts = [synthetic_client_data(i, n=32, seed=SEED + 40000) for i in range(CLIENTS)]
    eval_x = torch.cat([p[0] for p in eval_parts])
    eval_y = torch.cat([p[1] for p in eval_parts])
    accuracy, loss = _evaluate(global_model, eval_x, eval_y)
    # RDP accountant for the Gaussian mechanism, converted to (epsilon, delta)-DP.
    # No subsampling amplification is claimed; this intentionally errs on the conservative side.
    steps_per_client = total_steps // CLIENTS
    # Under replace-one adjacency, a clipped batch average has sensitivity 2C/B;
    # the added noise has std sigma*C/B, so the effective Gaussian multiplier is sigma/2.
    effective_multiplier = noise_multiplier / 2.0
    epsilon_increment = min(steps_per_client * alpha / (2 * effective_multiplier ** 2)
                            + math.log(1 / DELTA) / (alpha - 1)
                            for alpha in range(2, 129))
    state_blob = model_bytes(global_model)
    return {
        'model': global_model,
        'model_state': state_blob,
        'model_hash': hashlib.sha256(state_blob).hexdigest(),
        'accuracy': accuracy,
        'loss': loss,
        'epsilon_increment': epsilon_increment,
        'steps_per_client': total_steps // CLIENTS,
        'clients': CLIENTS,
        'delta': DELTA,
        'max_grad_norm': MAX_GRAD_NORM,
    }


def flower_client_fit(arrays: List, client_id: int, noise_multiplier: float = 1.0):
    model = DiagnosticNet()
    arrays_to_state(model, arrays)
    x, y = synthetic_client_data(client_id)
    steps = _dp_local_train(model, x, y, noise_multiplier)
    return state_to_arrays(model), len(x), {'steps': steps, 'client_id': client_id}
