"""Run the grokking reproduction and sweeps; write JSON + plots.

    grok single --p 59 --frac 0.3 --wd 1.0 --steps 20000 --out results
    grok sweep --p 59 --fracs 0.3 0.5 --wds 0 0.3 1.0 --seeds 0 1 --workers 4 --out results

Each run is independent, so sweeps fan out over processes (one torch thread each).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from .core import train


def run_one(cfg: dict) -> dict:
    torch.set_num_threads(1)
    t = time.time()
    _, run = train(**cfg)
    return {
        "config": cfg,
        "summary": run.summary(),
        "curves": asdict(run),
        "wall_s": time.time() - t,
    }


def _cfg(a, frac, wd, seed) -> dict:
    return dict(
        p=a.p,
        frac_train=frac,
        wd=wd,
        lr=a.lr,
        steps=a.steps,
        eval_every=a.eval_every,
        seed=seed,
        d=a.d,
        stop_after_generalise=a.stop_after,
    )


def plot_single(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = res["curves"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(c["steps"], c["train_acc"], label="train")
    ax[0].plot(c["steps"], c["test_acc"], label="test")
    ax[0].set(title="accuracy", xlabel="step", xscale="symlog")
    ax[0].legend()
    ax[1].plot(c["steps"], c["fourier_top5"], color="C2")
    ax[1].set(title="Fourier sparsity of W_E (top-5 freq share)", xlabel="step", xscale="symlog")
    ax[2].plot(c["steps"], c["weight_norm"], color="C3")
    ax[2].set(title="total weight norm²", xlabel="step", xscale="symlog")
    s = res["summary"]
    for a_ in ax:
        for key, col in (("steps_to_memorise", "C0"), ("steps_to_generalise", "C1")):
            if s[key] is not None:
                a_.axvline(s[key], color=col, ls=":", lw=1)
    cfg = res["config"]
    fig.suptitle(f"p={cfg['p']} frac={cfg['frac_train']} wd={cfg['wd']} seed={cfg['seed']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fracs = sorted({r["config"]["frac_train"] for r in rows})
    wds = sorted({r["config"]["wd"] for r in rows})
    grid = np.full((len(fracs), len(wds)), np.nan)
    for i, f in enumerate(fracs):
        for j, w in enumerate(wds):
            ts = [
                r["summary"]["steps_to_generalise"]
                for r in rows
                if r["config"]["frac_train"] == f and r["config"]["wd"] == w
            ]
            done = [t for t in ts if t is not None]
            if len(done) == len(ts) and ts:
                grid[i, j] = float(np.median(done))
    fig, ax = plt.subplots(figsize=(1.6 * len(wds) + 2, 1.0 * len(fracs) + 1.5))
    im = ax.imshow(np.log10(grid), cmap="viridis_r", aspect="auto")
    for i in range(len(fracs)):
        for j in range(len(wds)):
            txt = "never" if np.isnan(grid[i, j]) else f"{int(grid[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", color="w" if txt != "never" else "k")
    ax.set_xticks(range(len(wds)), [str(w) for w in wds])
    ax.set_yticks(range(len(fracs)), [str(f) for f in fracs])
    ax.set(xlabel="weight decay", ylabel="train fraction", title="median steps to 99% test acc")
    fig.colorbar(im, label="log10 steps")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sweep_table(rows: list[dict]) -> str:
    out = [
        "| frac | wd | seed | memorise @ | generalise @ | delay | final test acc | top-5 Fourier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    key = lambda r: (r["config"]["frac_train"], r["config"]["wd"], r["config"]["seed"])  # noqa: E731
    for r in sorted(rows, key=key):
        c, s = r["config"], r["summary"]
        out.append(
            f"| {c['frac_train']} | {c['wd']} | {c['seed']} | {s['steps_to_memorise']} | "
            f"{s['steps_to_generalise'] or 'never'} | {s['grokking_delay'] or '—'} | "
            f"{s['final_test_acc']:.3f} | {s['final_fourier_top5']:.2f} |"
        )
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="grok")
    ap.add_argument("mode", choices=["single", "sweep"])
    ap.add_argument("--p", type=int, default=59)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--stop-after", type=int, default=None, help="stop N steps after generalising")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--wds", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.mode == "single":
        res = run_one(_cfg(a, a.frac, a.wd, a.seed))
        tag = f"single_p{a.p}_frac{a.frac}_wd{a.wd}_seed{a.seed}"
        (a.out / f"{tag}.json").write_text(json.dumps(res))
        plot_single(res, a.out / f"{tag}.png")
        print(json.dumps(res["summary"], indent=1))
        return

    cfgs = [_cfg(a, f, w, s) for f, w, s in itertools.product(a.fracs, a.wds, a.seeds)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, cfgs):
            rows.append(r)
            c, s = r["config"], r["summary"]
            print(
                f"frac={c['frac_train']} wd={c['wd']} seed={c['seed']}: "
                f"gen@{s['steps_to_generalise']} test={s['final_test_acc']:.3f} "
                f"({r['wall_s']:.0f}s)",
                flush=True,
            )
    (a.out / "sweep.json").write_text(json.dumps(rows))
    md = sweep_table(rows)
    (a.out / "sweep.md").write_text(md + "\n")
    plot_sweep(rows, a.out / "sweep.png")
    print(md)


if __name__ == "__main__":
    main()
