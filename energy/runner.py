"""Episode runners shared by training, evaluation and validation scripts."""
import numpy as np

from . import config as C
from .agents import make_rng
from .simulator import CompanyEnv


def run_episode(env, policy, seed, start_hour=0, start_dow=None, init_soc=C.INIT_SOC,
                fault_rate=0.0, rng_seed=0, record=False):
    """Run one episode with a decentralised policy. Returns (metrics, trace|None)."""
    if start_dow is None:
        start_dow = seed % 7
    rng = make_rng(rng_seed)
    obs = env.reset(seed, start_hour, start_dow, fault_rate, init_soc)
    trace = {"actions": [], "soc": [], "grid": [], "sig": []} if record else None
    while True:
        actions = [policy.act(i, obs[i], rng) for i in range(env.n)]
        if record:
            trace["sig"].append(obs[0][3])
        obs, _, _, done, res = env.step(actions)
        if record:
            trace["actions"].append(actions)
            trace["soc"].append([r["soc"] for r in res])
            trace["grid"].append([r["grid"] for r in res])
        if done:
            break
    return env.metrics(), trace


def evaluate_policy(policy, seeds, T=C.EPISODE_HOURS):
    env = CompanyEnv(T=T)
    rows = [run_episode(env, policy, s, rng_seed=s)[0] for s in seeds]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, rows
