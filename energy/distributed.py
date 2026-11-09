"""Distributed runtime: one OS process per site node plus a central coordinator.

Protocol (lock-step, one round per simulated hour):

    coordinator --signal(level)-->  every site node          (company load level)
    site node   --report(t, ...)--> coordinator              (meter readings + edge alert)
    coordinator aggregates readings, runs fault handling, broadcasts the next signal.

Each site node runs its own physics, its own trained decentralised policy (it
only sees its local state + the broadcast signal) and its own edge anomaly
detector. The coordinator never controls batteries; it only aggregates and
broadcasts.

Fault tolerance: if a site's report does not arrive within `timeout`, the
coordinator marks the site OFFLINE, keeps going using its last known reading,
and no longer waits for it. The site keeps operating autonomously, buffers its
reports, and back-fills them when the link returns.
"""
import multiprocessing as mp
import queue
import time

import numpy as np
import pandas as pd

from . import config as C
from .agents import RuleTOU, IDLE, TabularQ
from .detector import HourlyBaselineDetector
from .simulator import SiteSim, generate_series, signal_level


def _make_policy(run_cfg, idx):
    kind = run_cfg["policy"]
    if kind == "rule":
        rule = RuleTOU()
        return lambda obs: rule.act(idx, obs, None)
    if kind == "none":
        return lambda obs: IDLE
    z = np.load(run_cfg["model_path"])
    agent = TabularQ(run_cfg["use_signal"], z[f"arr_{idx % len(z.files)}"])
    return agent.greedy


def site_node(idx, cfg, run_cfg, inbox, outbox):
    """Process body of one site node."""
    try:
        series = generate_series(cfg, idx, run_cfg["seed"], run_cfg["T"], run_cfg["start_hour"],
                                 run_cfg["start_dow"], run_cfg["fault_rate"])
        sim = SiteSim(cfg, series, run_cfg.get("init_soc", C.INIT_SOC))
        policy = _make_policy(run_cfg, idx)
        det = HourlyBaselineDetector()
        outage = run_cfg.get("outage")          # (site_idx, start_t, end_t) or None
        buffered = []
        outbox.put(("ready", idx))
        sig = inbox.get(timeout=120)
        for t in range(run_cfg["T"]):
            h, socb, netb = sim.obs_local(t)
            a = policy((h, socb, netb, sig))
            r = sim.step(t, a)
            alert = det.update(series["weekend"][t], h, r["load_reading"])
            rep = {"t": t, "site": idx, "local_hour": h, "grid": r["grid"],
                   "grid_reading": r["grid_reading"], "load_reading": r["load_reading"],
                   "soc": r["soc"], "cost": r["cost"], "co2": r["co2"], "action": a,
                   "alert": bool(alert)}
            down = outage is not None and outage[0] == idx and outage[1] <= t < outage[2]
            if down:
                buffered.append(rep)
            else:
                if buffered:
                    outbox.put(("backfill", idx, buffered))
                    buffered = []
                outbox.put(("report", idx, t, rep))
            sig = inbox.get(timeout=120)
        if buffered:
            outbox.put(("backfill", idx, buffered))
        outbox.put(("done", idx, {"energy_cost": sim.energy_cost, "co2": sim.co2,
                                  "degradation": sim.degradation, "grid_kwh": sim.grid_kwh}))
    except Exception as exc:                    # report instead of hanging the coordinator
        outbox.put(("error", idx, repr(exc)))


def run_distributed(run_cfg, sites=None, timeout=1.0, grace=0.02):
    """Start the site nodes, act as coordinator, return metrics + logs."""
    sites = list(sites or C.SITES)
    n, T = len(sites), run_cfg["T"]
    sum_base = sum(s.base_kw for s in sites)
    ctx = mp.get_context("spawn")
    inboxes = [ctx.Queue() for _ in sites]
    outbox = ctx.Queue()
    procs = [ctx.Process(target=site_node, args=(i, sites[i], run_cfg, inboxes[i], outbox), daemon=True)
             for i in range(n)]
    for p in procs:
        p.start()
    ready = set()                       # start-up handshake: wait until every node is up
    end = time.perf_counter() + 120
    while len(ready) < n and time.perf_counter() < end:
        try:
            msg = outbox.get(timeout=1.0)
        except queue.Empty:
            continue
        if msg[0] == "ready":
            ready.add(msg[1])
        elif msg[0] == "error":
            raise RuntimeError(f"site {msg[1]} failed: {msg[2]}")
    if len(ready) < n:
        raise RuntimeError("not all site nodes started")
    for q in inboxes:
        q.put(C.INITIAL_SIGNAL)

    online = [True] * n
    last_read = [0.0] * n
    last_grid = [0.0] * n
    rows, events, agg_true, agg_read, step_ms = [], [], [], [], []
    n_msgs = 0
    t0 = time.perf_counter()

    for t in range(T):
        ts = time.perf_counter()
        got = {}

        def collect(wait_for, limit):
            nonlocal n_msgs
            end = time.perf_counter() + limit
            while not wait_for <= set(got):
                left = end - time.perf_counter()
                if left <= 0:
                    return
                try:
                    msg = outbox.get(timeout=left)
                except queue.Empty:
                    return
                n_msgs += 1
                if msg[0] == "report" and msg[2] == t:
                    got[msg[1]] = msg[3]
                elif msg[0] == "backfill":
                    for rep in msg[2]:
                        rep = dict(rep)
                        rep["backfilled"] = True
                        rows.append(rep)
                elif msg[0] == "error":
                    raise RuntimeError(f"site {msg[1]} failed: {msg[2]}")

        collect({i for i in range(n) if online[i]}, timeout)
        offline = {i for i in range(n) if not online[i]}
        if offline:
            collect(offline, grace)
        for i in range(n):
            if i in got:
                if not online[i]:
                    events.append({"t": t, "site": i, "event": "rejoined"})
                online[i] = True
                last_read[i] = got[i]["grid_reading"]
                last_grid[i] = got[i]["grid"]
                rep = dict(got[i])
                rep["backfilled"] = False
                rows.append(rep)
            elif online[i]:
                online[i] = False
                events.append({"t": t, "site": i, "event": "offline (no report in %.2fs)" % timeout})
        agg_true.append(sum(last_grid))
        agg_read.append(sum(last_read))
        sig = signal_level(agg_read[-1] / sum_base)
        for q in inboxes:
            q.put(sig)
        step_ms.append((time.perf_counter() - ts) * 1000.0)

    totals = {}
    end = time.perf_counter() + 30
    while len(totals) < n and time.perf_counter() < end:
        try:
            msg = outbox.get(timeout=1.0)
        except queue.Empty:
            continue
        if msg[0] == "done":
            totals[msg[1]] = msg[2]
        elif msg[0] == "backfill":
            for rep in msg[2]:
                rep = dict(rep)
                rep["backfilled"] = True
                rows.append(rep)
        elif msg[0] == "error":
            raise RuntimeError(f"site {msg[1]} failed: {msg[2]}")
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    wall = time.perf_counter() - t0

    energy = sum(v["energy_cost"] for v in totals.values())
    deg = sum(v["degradation"] for v in totals.values())
    co2 = sum(v["co2"] for v in totals.values())
    peak = max(agg_true)
    metrics = {"energy_cost": energy, "degradation": deg, "co2_kg": co2, "peak_kw": peak,
               "demand_charge": C.DEMAND_RATE * peak,
               "total_cost": energy + deg + C.CARBON_PRICE * co2 + C.DEMAND_RATE * peak}
    df = pd.DataFrame(rows).sort_values(["t", "site"]).reset_index(drop=True)
    return {"metrics": metrics, "log": df, "events": pd.DataFrame(events), "totals": totals,
            "agg_true": np.array(agg_true), "agg_reading": np.array(agg_read),
            "wall_s": wall, "step_ms": np.array(step_ms), "messages": n_msgs}
