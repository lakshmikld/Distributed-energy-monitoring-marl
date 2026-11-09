"""Evaluate all strategies on held-out test episodes (seeds never used for training or tuning).

Usage:  python scripts/evaluate.py
Writes: results/comparison.csv, results/comparison.json, results/traces.npz
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from energy.agents import GreedyQ, NoBattery, RandomPolicy, RuleTOU  # noqa: E402
from energy.runner import evaluate_policy, run_episode              # noqa: E402
from energy.simulator import CompanyEnv                              # noqa: E402

TEST_SEEDS = list(range(40000, 40030))
VARIANTS = [("iql", False, "IQL (local reward, no signal)"),
            ("team", False, "MARL team reward"),
            ("team_signal", True, "MARL team reward + company signal")]


def load_q(variant, seed):
    z = np.load(os.path.join(ROOT, "models", f"{variant}_seed{seed}.npz"))
    return [z[f"arr_{i}"] for i in range(len(z.files))]


def main():
    rows = []
    for pol in (NoBattery(), RuleTOU(), RandomPolicy()):
        m, per = evaluate_policy(pol, TEST_SEEDS)
        rows.append({"strategy": pol.name, "kind": "baseline", **m, "total_cost_std": 0.0, "n_train_seeds": 0})
    n_seeds = len([f for f in os.listdir(os.path.join(ROOT, "models")) if f.startswith("iql_seed")])
    for variant, sig, label in VARIANTS:
        per_seed = []
        for s in range(n_seeds):
            m, _ = evaluate_policy(GreedyQ(load_q(variant, s), sig, label), TEST_SEEDS)
            per_seed.append(m)
        mean = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
        std = float(np.std([p["total_cost"] for p in per_seed]))
        rows.append({"strategy": label, "kind": "learned", **mean, "total_cost_std": std,
                     "n_train_seeds": n_seeds,
                     "best_seed_total": float(min(p["total_cost"] for p in per_seed)),
                     "worst_seed_total": float(max(p["total_cost"] for p in per_seed))})
    df = pd.DataFrame(rows)
    base = float(df.loc[df.strategy == "No battery", "total_cost"].iloc[0])
    df["saving_vs_no_battery_pct"] = 100.0 * (base - df["total_cost"]) / base
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "comparison.csv"), index=False)
    with open(os.path.join(ROOT, "results", "comparison.json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=1)
    pd.set_option("display.width", 200)
    print(df[["strategy", "energy_cost", "demand_charge", "peak_kw", "co2_kg", "total_cost",
              "total_cost_std", "saving_vs_no_battery_pct"]].round(1).to_string(index=False))

    # traces for charts: one test episode, training seed 0 for each learned variant
    seed = TEST_SEEDS[0]
    env = CompanyEnv()
    tr = {}
    pols = [("no_battery", NoBattery()), ("rule", RuleTOU())] + \
           [(v, GreedyQ(load_q(v, 0), sig, v)) for v, sig, _ in VARIANTS]
    for key, pol in pols:
        _, t = run_episode(env, pol, seed, rng_seed=seed, record=True)
        tr[key + "_grid"] = np.array(t["grid"])
        tr[key + "_soc"] = np.array(t["soc"])
        tr[key + "_actions"] = np.array(t["actions"])
    np.savez(os.path.join(ROOT, "results", "traces.npz"), **tr)

    # paired per-episode comparison of best learned variant vs rule (informational)
    print("\nTest seeds:", TEST_SEEDS[0], "-", TEST_SEEDS[-1], f"({len(TEST_SEEDS)} episodes x {n_seeds} training seeds)")


if __name__ == "__main__":
    main()
