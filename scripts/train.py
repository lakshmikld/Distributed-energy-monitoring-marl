"""Train the multi-agent Q-learners (one agent per site).

Variants (ablation):
  iql          independent Q-learning, local reward, no communication
  team         + shared company peak-demand penalty in every agent's reward
  team_signal  + coordinator broadcasts a company load level into every observation

Usage:  python scripts/train.py [--episodes 3000] [--seeds 3]
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from energy import config as C                      # noqa: E402
from energy.agents import GreedyQ, TabularQ         # noqa: E402
from energy.runner import evaluate_policy           # noqa: E402
from energy.simulator import CompanyEnv, state_index  # noqa: E402

VARIANTS = {
    "iql": dict(use_signal=False, team=False),
    "team": dict(use_signal=False, team=True),
    "team_signal": dict(use_signal=True, team=True),
}
ALPHA, GAMMA = 0.10, 0.98
EVAL_SEEDS = list(range(20000, 20005))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_one(job):
    variant, seed, episodes = job
    cfg = VARIANTS[variant]
    rng = random.Random(1000 + seed)
    nprng = np.random.default_rng(1000 + seed)
    env = CompanyEnv()
    agents = [TabularQ(cfg["use_signal"]) for _ in range(env.n)]
    curve, eval_curve = [], []
    for ep in range(episodes):
        eps = max(0.02, 1.0 - ep / (0.7 * episodes))
        alpha = ALPHA * max(0.2, 1.0 - ep / episodes)
        ep_seed = 100000 * (seed + 1) + ep
        obs = env.reset(ep_seed, int(nprng.integers(24)), int(nprng.integers(7)),
                        0.0, float(nprng.uniform(0.2, 0.8)))
        states = [state_index(o, cfg["use_signal"]) for o in obs]
        done = False
        while not done:
            acts = [agents[i].act(obs[i], eps, rng) for i in range(env.n)]
            obs2, local_r, peak_pen, done, _ = env.step(acts)
            states2 = [0] * env.n if done else [state_index(o, cfg["use_signal"]) for o in obs2]
            for i in range(env.n):
                r = local_r[i] - (peak_pen if cfg["team"] else 0.0)
                agents[i].update(states[i], acts[i], r, states2[i], done, alpha, GAMMA)
            obs, states = obs2, states2
        curve.append(env.metrics()["total_cost"])
        if (ep + 1) % 250 == 0 or ep == 0:
            pol = GreedyQ([a.q for a in agents], cfg["use_signal"], variant)
            m, _ = evaluate_policy(pol, EVAL_SEEDS)
            eval_curve.append((ep + 1, m["total_cost"]))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    np.savez(os.path.join(ROOT, "models", f"{variant}_seed{seed}.npz"),
             *[a.q for a in agents])
    return {"variant": variant, "seed": seed, "train_curve": curve, "eval_curve": eval_curve}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    jobs = [(v, s, args.episodes) for v in args.variants.split(",") for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
        out = pool.map(train_one, jobs)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "training_log.json"), "w") as f:
        json.dump(out, f)
    for o in out:
        print(o["variant"], o["seed"], "final greedy eval cost:", round(o["eval_curve"][-1][1], 1))
