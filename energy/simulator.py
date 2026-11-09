"""Multi-site energy simulator.

`generate_series` builds the synthetic hourly data for one site (load, solar,
injected faults). `SiteSim` is the per-site physics (battery + grid + tariff) and
is used both by the in-process training environment (`CompanyEnv`) and by the
distributed site nodes, so both paths run exactly the same code.
"""
import math

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- data
def _profile(cfg, h):
    if cfg.profile == "office":
        return 0.50 + 0.50 * max(0.0, math.sin(math.pi * (h - 7) / 12.0))
    return 0.80 + 0.20 * max(0.0, math.sin(math.pi * (h - 6) / 14.0))


def generate_series(cfg, site_idx, seed, T, start_hour=0, start_dow=0,
                    fault_rate=0.0, fault_after=72):
    """Synthetic hourly series for one site.

    Returns a dict of numpy arrays of length T:
      load        true electrical demand (kW), includes real 'spike' faults
      solar       PV output (kW)
      local_hour  local clock hour 0..23
      dropout     True where the meter reads zero (sensor fault)
      label       True where any fault was injected (ground truth for detection)
    """
    rng = np.random.default_rng([int(seed), int(site_idx), 7])
    t = np.arange(T) + start_hour + cfg.tz
    local_hour = (t % 24).astype(int)
    day = t // 24
    dow = (start_dow + day) % 7
    weekend = dow >= 5

    wk = 0.6 if cfg.profile == "office" else 0.9
    prof = np.array([_profile(cfg, h) for h in local_hour])
    prof = np.where(weekend, prof * wk, prof)
    load = cfg.base_kw * prof * (1.0 + rng.normal(0.0, 0.04, T))

    d0 = int(day.min())
    cloud_by_day = rng.uniform(0.4, 1.0, int(day.max()) - d0 + 1)
    cloud = cloud_by_day[(day - d0).astype(int)]
    shape = np.maximum(0.0, np.sin(np.pi * (local_hour - 6) / 12.0))
    solar = cfg.solar_kw * cfg.sun * cloud * shape * (1.0 + rng.normal(0.0, 0.05, T))
    solar = np.maximum(solar, 0.0)

    dropout = np.zeros(T, dtype=bool)
    label = np.zeros(T, dtype=bool)
    if fault_rate > 0:
        i = fault_after
        while i < T:
            if rng.random() < fault_rate:
                if rng.random() < 0.6:                       # real load spike
                    dur = int(rng.integers(1, 5))
                    load[i:i + dur] *= rng.uniform(1.6, 2.5)
                else:                                        # meter dropout
                    dur = int(rng.integers(1, 4))
                    dropout[i:i + dur] = True
                label[i:i + dur] = True
                i += dur
            i += 1
    return {"load": load, "solar": solar, "local_hour": local_hour,
            "weekend": weekend, "dropout": dropout, "label": label}


# ---------------------------------------------------------------------- site physics
class SiteSim:
    """Battery + grid connection of one site, stepped one hour at a time."""

    def __init__(self, cfg, series, init_soc=C.INIT_SOC):
        self.cfg = cfg
        self.s = series
        self.soc = float(init_soc)
        self.energy_cost = 0.0
        self.co2 = 0.0
        self.degradation = 0.0
        self.grid_kwh = 0.0

    def price_at(self, t):
        h = int(self.s["local_hour"][t])
        return self.cfg.price * C.PRICE_MULT[C.price_period(h)]

    def obs_local(self, t):
        h = int(self.s["local_hour"][t])
        socb = min(int(self.soc * C.SOC_BUCKETS), C.SOC_BUCKETS - 1)
        net = (self.s["load"][t] - self.s["solar"][t]) / self.cfg.base_kw
        netb = 0 if net < C.NET_THRESHOLDS[0] else (1 if net < C.NET_THRESHOLDS[1] else 2)
        return h, socb, netb

    def step(self, t, a_idx):
        cfg = self.cfg
        load = float(self.s["load"][t])
        solar = float(self.s["solar"][t])
        frac = C.ACTION_FRACS[a_idx]
        net = load - solar
        throughput = 0.0
        if frac > 0:                                    # charge (solar surplus first)
            room = (1.0 - self.soc) * cfg.batt_kwh / C.EFF
            charge_in = min(frac * cfg.batt_kw, room)
            stored = charge_in * C.EFF
            self.soc += stored / cfg.batt_kwh
            grid = max(0.0, net + charge_in)
            throughput = stored
        elif frac < 0:                                  # discharge, no export
            avail = self.soc * cfg.batt_kwh * C.EFF
            need = max(0.0, net)
            d = min(-frac * cfg.batt_kw, avail, need)
            self.soc -= d / (C.EFF * cfg.batt_kwh)
            grid = max(0.0, net - d)
            throughput = d / C.EFF
        else:
            grid = max(0.0, net)
        self.soc = min(1.0, max(0.0, self.soc))
        cost = grid * self.price_at(t)
        co2 = grid * cfg.co2
        deg = C.DEGRADATION * throughput
        self.energy_cost += cost
        self.co2 += co2
        self.degradation += deg
        self.grid_kwh += grid
        reading_ok = not bool(self.s["dropout"][t])
        idle_grid = max(0.0, net)       # what an idle battery would draw (exogenous baseline)
        return {
            "idle_grid": idle_grid,
            "idle_cost": idle_grid * self.price_at(t),
            "idle_co2": idle_grid * cfg.co2,
            "grid": grid,
            "grid_reading": grid if reading_ok else 0.0,
            "load_reading": load if reading_ok else 0.0,
            "cost": cost, "co2": co2, "deg": deg, "soc": self.soc,
        }


# ------------------------------------------------------------------ helper functions
def signal_level(agg_ratio):
    lo, hi = C.SIGNAL_THRESHOLDS
    return 0 if agg_ratio < lo else (1 if agg_ratio < hi else 2)


def state_index(obs, use_signal):
    h, socb, netb, sig = obs
    s = (h * C.SOC_BUCKETS + socb) * 3 + netb
    return s * C.N_SIGNAL + sig if use_signal else s


class CompanyEnv:
    """All sites + the coordinator's aggregation, for fast in-process training/eval."""

    def __init__(self, sites=None, T=C.EPISODE_HOURS):
        self.sites = list(sites or C.SITES)
        self.n = len(self.sites)
        self.T = T
        self.sum_base = sum(s.base_kw for s in self.sites)

    def reset(self, seed, start_hour=0, start_dow=0, fault_rate=0.0, init_soc=C.INIT_SOC):
        self.series = [generate_series(c, i, seed, self.T, start_hour, start_dow, fault_rate)
                       for i, c in enumerate(self.sites)]
        self.sims = [SiteSim(c, s, init_soc) for c, s in zip(self.sites, self.series)]
        unmanaged = np.zeros(self.T)
        for s in self.series:
            unmanaged += np.maximum(0.0, s["load"] - s["solar"])
        self.unmanaged_peak = float(unmanaged.max())
        self.theta = C.PEAK_REF_RATIO * self.unmanaged_peak
        self.t = 0
        self.sig = C.INITIAL_SIGNAL
        self.agg_true = []
        self.agg_reading = []
        return self.observe()

    def observe(self):
        out = []
        for sim in self.sims:
            h, socb, netb = sim.obs_local(self.t)
            out.append((h, socb, netb, self.sig))
        return out

    def step(self, actions):
        res = [sim.step(self.t, a) for sim, a in zip(self.sims, actions)]
        agg_true = sum(r["grid"] for r in res)
        agg_read = sum(r["grid_reading"] for r in res)
        self.agg_true.append(agg_true)
        self.agg_reading.append(agg_read)
        agg_idle = sum(r["idle_grid"] for r in res)
        # Rewards are measured relative to an idle-battery baseline. The baseline depends
        # only on exogenous load/solar/price, so it does not change the optimal policy but
        # removes most of the noise from the learning signal.
        peak_pen = C.SHAPING_MU * (max(0.0, agg_true - self.theta)
                                   - max(0.0, agg_idle - self.theta)) / C.REWARD_SCALE
        local_r = [((r["idle_cost"] - r["cost"]) + C.CARBON_PRICE * (r["idle_co2"] - r["co2"])
                    - r["deg"]) / C.REWARD_SCALE for r in res]
        self.sig = signal_level(agg_read / self.sum_base)
        self.t += 1
        done = self.t >= self.T
        return self.observe() if not done else None, local_r, peak_pen, done, res

    def metrics(self):
        energy = sum(s.energy_cost for s in self.sims)
        deg = sum(s.degradation for s in self.sims)
        co2 = sum(s.co2 for s in self.sims)
        peak = max(self.agg_true) if self.agg_true else 0.0
        demand = C.DEMAND_RATE * peak
        total = energy + deg + C.CARBON_PRICE * co2 + demand
        return {"energy_cost": energy, "degradation": deg, "co2_kg": co2,
                "peak_kw": peak, "demand_charge": demand, "total_cost": total,
                "unmanaged_peak_kw": self.unmanaged_peak}
