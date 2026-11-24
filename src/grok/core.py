"""Data, model, training loop, and progress measures.

Setup follows Nanda et al. 2023 ("Progress measures for grokking via mechanistic
interpretability"): inputs ``a b =`` with ``a, b in Z_p``, predict ``(a + b) mod p``
at the ``=`` position; a 1-layer transformer without LayerNorm; full-batch AdamW
with strong weight decay; a random fraction of the p^2 pairs used for training.

Progress measures (computed on the token embedding ``W_E[:p]``):

* **Fourier sparsity** — the share of embedding norm in the top-k Fourier
  frequencies. The generalising algorithm represents inputs as cos/sin of a few
  key frequencies, so this rises as the circuit forms — often well before test
  accuracy jumps.
* **Weight norm** — total squared L2 norm; weight decay shrinks the memorising
  solution once the generalising one can carry the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def make_data(p: int, frac_train: float, seed: int):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)  # token p is "="
    y = (a + b) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    n_tr = int(frac_train * p * p)
    tr, te = perm[:n_tr], perm[n_tr:]
    return (x[tr], y[tr]), (x[te], y[te])


class OneLayer(nn.Module):
    def __init__(self, p: int, d: int = 128, heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p, self.h = p, heads
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / math.sqrt(d))
        self.W_pos = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d)
        self.W_U = nn.Linear(d, p, bias=False)

    def forward(self, x):
        h = self.W_E[x] + self.W_pos
        B, T, C = h.shape
        q, k, v = self.qkv(h).split(C, 2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, T, C))
        h = h + self.mlp_out(F.relu(self.mlp_in(h)))
        return self.W_U(h[:, -1])


def fourier_basis(p: int) -> torch.Tensor:
    """Orthonormal real Fourier basis on Z_p: rows = [const, cos1, sin1, cos2, sin2, ...]."""
    n = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows.append(torch.cos(2 * math.pi * k * n / p))
        rows.append(torch.sin(2 * math.pi * k * n / p))
    B = torch.stack(rows[:p])
    return (B / B.norm(dim=1, keepdim=True)).float()


def frequency_norms(W_E: torch.Tensor, p: int) -> np.ndarray:
    """Norm of the embedding in each frequency k = 1..p//2 (cos and sin combined)."""
    F_ = fourier_basis(p) @ W_E[:p]
    sq = F_.pow(2).sum(1)
    return np.array([float(sq[2 * k - 1] + sq[2 * k]) for k in range(1, p // 2 + 1)]) ** 0.5


def fourier_sparsity(W_E: torch.Tensor, p: int, top_k: int = 5) -> float:
    n2 = frequency_norms(W_E, p) ** 2
    return float(np.sort(n2)[::-1][:top_k].sum() / n2.sum())


@dataclass
class Run:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    weight_norm: list[float] = field(default_factory=list)
    fourier_top5: list[float] = field(default_factory=list)
    key_freqs: list[int] = field(default_factory=list)

    def first_step(self, series: str, thr: float) -> int | None:
        for s, v in zip(self.steps, getattr(self, series), strict=True):
            if v >= thr:
                return s
        return None

    def summary(self) -> dict:
        t_mem = self.first_step("train_acc", 0.99)
        t_gen = self.first_step("test_acc", 0.99)
        return {
            "steps_to_memorise": t_mem,
            "steps_to_generalise": t_gen,
            "grokking_delay": (t_gen - t_mem)
            if (t_gen is not None and t_mem is not None)
            else None,
            "final_train_acc": self.train_acc[-1],
            "final_test_acc": self.test_acc[-1],
            "final_fourier_top5": self.fourier_top5[-1],
            "key_freqs": self.key_freqs,
        }


def train(
    p: int = 97,
    frac_train: float = 0.3,
    wd: float = 1.0,
    lr: float = 1e-3,
    steps: int = 20_000,
    eval_every: int = 100,
    seed: int = 0,
    d: int = 128,
    stop_after_generalise: int | None = None,
) -> tuple[OneLayer, Run]:
    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = make_data(p, frac_train, seed)
    model = OneLayer(p, d=d)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    run = Run()
    generalised_at = None
    for step in range(steps + 1):
        if step % eval_every == 0:
            with torch.no_grad():
                lt, le = model(xtr), model(xte)
                run.steps.append(step)
                run.train_loss.append(F.cross_entropy(lt, ytr).item())
                run.test_loss.append(F.cross_entropy(le, yte).item())
                run.train_acc.append((lt.argmax(1) == ytr).float().mean().item())
                run.test_acc.append((le.argmax(1) == yte).float().mean().item())
                run.weight_norm.append(sum(q.pow(2).sum().item() for q in model.parameters()))
                run.fourier_top5.append(fourier_sparsity(model.W_E.detach(), p))
            if generalised_at is None and run.test_acc[-1] >= 0.99:
                generalised_at = step
            if (
                stop_after_generalise is not None
                and generalised_at is not None
                and step - generalised_at >= stop_after_generalise
            ):
                break
        if step == steps:
            break
        loss = F.cross_entropy(model(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    norms = frequency_norms(model.W_E.detach(), p)
    run.key_freqs = sorted(int(k) + 1 for k in np.argsort(norms)[::-1][:5])
    return model, run
