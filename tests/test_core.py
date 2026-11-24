import math

import numpy as np
import torch

from grok.core import OneLayer, fourier_basis, fourier_sparsity, frequency_norms, make_data, train


def test_data_split_and_labels():
    (xtr, ytr), (xte, yte) = make_data(11, 0.5, seed=0)
    assert len(xtr) + len(xte) == 121 and len(xtr) == 60
    assert torch.equal(ytr, (xtr[:, 0] + xtr[:, 1]) % 11)
    assert (xtr[:, 2] == 11).all()
    pairs = lambda x: {tuple(r) for r in x[:, :2].tolist()}  # noqa: E731
    assert not pairs(xtr) & pairs(xte)


def test_fourier_basis_is_orthonormal():
    B = fourier_basis(13)
    torch.testing.assert_close(B @ B.T, torch.eye(13), atol=1e-5, rtol=0)


def test_pure_frequency_embedding_is_detected():
    p, k = 17, 3
    n = torch.arange(p, dtype=torch.float32)
    W = torch.stack([torch.cos(2 * math.pi * k * n / p), torch.sin(2 * math.pi * k * n / p)], 1)
    norms = frequency_norms(W, p)
    assert int(np.argmax(norms)) + 1 == k
    assert fourier_sparsity(W, p, top_k=1) > 0.99
    rand = torch.randn(p, 8, generator=torch.Generator().manual_seed(0))
    assert fourier_sparsity(rand, p, top_k=1) < 0.5


def test_model_shapes():
    m = OneLayer(7, d=16, heads=2, d_mlp=32)
    assert m(torch.tensor([[1, 2, 7], [3, 4, 7]])).shape == (2, 7)


def test_small_run_memorises_and_logs():
    _, run = train(p=23, frac_train=0.5, wd=1.0, steps=600, eval_every=100, d=32)
    s = run.summary()
    assert run.steps[0] == 0 and run.steps[-1] == 600
    assert s["final_train_acc"] > 0.95
    assert len(s["key_freqs"]) == 5
