"""How does the coordinator round time grow with the number of site nodes?

Runs 72 simulated hours with N site nodes (the six site profiles are repeated and
given distinct seeds), policy = trained team model, and records wall time per round.
Writes results/scaling.csv.

Usage:  python scripts/benchmark_scaling.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from energy import config as C                  # noqa: E402
from energy.distributed import run_distributed  # noqa: E402

MODEL = os.path.join(ROOT, "models", "team_seed0.npz")


def main():
    rows = []
    for n in (2, 6, 12, 24, 48):
        sites = [C.SITES[i % len(C.SITES)] for i in range(n)]
        cfg = dict(seed=60000, T=72, start_hour=0, start_dow=0, fault_rate=0.0, policy="q",
                   model_path=MODEL, use_signal=False, init_soc=C.INIT_SOC, outage=None)
        out = run_distributed(cfg, sites)
        rows.append({"nodes": n, "messages": out["messages"], "wall_s": out["wall_s"],
                     "mean_round_ms": float(out["step_ms"][5:].mean()),
                     "p95_round_ms": float(np.percentile(out["step_ms"][5:], 95))})
        print(rows[-1])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "scaling.csv"), index=False)


if __name__ == "__main__":
    main()
