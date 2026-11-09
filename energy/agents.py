"""Policies: simple baselines and tabular Q-learning agents."""
import random

import numpy as np

from . import config as C
from .simulator import state_index

IDLE, CHARGE_FULL, CHARGE_HALF, DISCH_HALF, DISCH_FULL = range(5)


class TabularQ:
    """One independent learner (one per site)."""

    def __init__(self, use_signal, q=None):
        self.use_signal = use_signal
        self.q = np.zeros((C.n_states(use_signal), C.N_ACTIONS)) if q is None else q

    def greedy(self, obs):
        return int(np.argmax(self.q[state_index(obs, self.use_signal)]))

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return rng.randrange(C.N_ACTIONS)
        return self.greedy(obs)

    def update(self, s, a, r, s2, done, alpha, gamma):
        target = r if done else r + gamma * float(self.q[s2].max())
        self.q[s, a] += alpha * (target - self.q[s, a])


# ------------------------------------------------------------------------ baselines
class NoBattery:
    name = "No battery"

    def act(self, i, obs, rng):
        return IDLE


class RandomPolicy:
    name = "Random"

    def act(self, i, obs, rng):
        return rng.randrange(C.N_ACTIONS)


class RuleTOU:
    """Local time-of-use rule: charge when cheap, discharge when expensive."""
    name = "Rule-based (TOU)"

    def act(self, i, obs, rng):
        h, socb, netb, _ = obs
        period = C.price_period(h)
        if period == "peak" and socb > 0:
            return DISCH_FULL
        if period == "off" and socb < C.SOC_BUCKETS - 1:
            return CHARGE_FULL
        if netb == 0 and socb < C.SOC_BUCKETS - 1:      # solar surplus
            return CHARGE_HALF
        return IDLE


class GreedyQ:
    """Wraps trained Q-tables (one per site) as a decentralised policy."""

    def __init__(self, qs, use_signal, name):
        self.agents = [TabularQ(use_signal, q) for q in qs]
        self.name = name

    def act(self, i, obs, rng):
        return self.agents[i].greedy(obs)


def make_rng(seed):
    return random.Random(seed)
