"""Run the distributed monitoring system (one process per site + coordinator).

  python scripts/run_distributed.py validate   # distributed run must equal in-process run
  python scripts/run_distributed.py monitor    # 4 simulated weeks with injected faults + a link outage
  python scripts/run_distributed.py all        # both (default)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from energy import config as C                               # noqa: E402
from energy.agents import GreedyQ                            # noqa: E402
from energy.distributed import run_distributed               # noqa: E402
from energy.runner import run_episode                        # noqa: E402
from energy.simulator import CompanyEnv, generate_series     # noqa: E402

MODEL = os.path.join(ROOT, "models", "team_seed0.npz")
RES = os.path.join(ROOT, "results")


def validate_one(model_name, use_signal):
    seed = 40001
    dow = seed % 7
    path = os.path.join(ROOT, "models", model_name)
    cfg = dict(seed=seed, T=C.EPISODE_HOURS, start_hour=0, start_dow=dow, fault_rate=0.0,
               policy="q", model_path=path, use_signal=use_signal, init_soc=C.INIT_SOC, outage=None)
    out = run_distributed(cfg)
    z = np.load(path)
    pol = GreedyQ([z[f"arr_{i}"] for i in range(len(z.files))], use_signal, model_name)
    env = CompanyEnv(T=C.EPISODE_HOURS)
    ref, _ = run_episode(env, pol, seed, 0, dow, C.INIT_SOC, 0.0, rng_seed=0)
    keys = ["energy_cost", "degradation", "co2_kg", "peak_kw", "demand_charge", "total_cost"]
    diffs = {k: abs(out["metrics"][k] - ref[k]) for k in keys}
    agg_diff = float(np.max(np.abs(out["agg_true"] - np.array(env.agg_true))))
    ok = max(diffs.values()) < 1e-6 and agg_diff < 1e-6
    rep = {"model": model_name, "uses_company_signal": use_signal, "status": "PASS" if ok else "FAIL",
           "distributed": out["metrics"], "in_process": ref,
           "max_abs_metric_diff": max(diffs.values()), "max_abs_hourly_aggregate_diff_kw": agg_diff,
           "wall_s": out["wall_s"], "messages": out["messages"],
           "mean_round_ms": float(out["step_ms"].mean())}
    print("Validation %-18s %s  total cost %.4f vs %.4f, max hourly aggregate diff %.1e kW, %d msgs, %.1f ms/round" % (
        model_name, rep["status"], out["metrics"]["total_cost"], ref["total_cost"], agg_diff,
        out["messages"], rep["mean_round_ms"]))
    return rep


def validate():
    reps = [validate_one("team_seed0.npz", False), validate_one("team_signal_seed0.npz", True)]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "validation.json"), "w") as f:
        json.dump(reps, f, indent=1)
    return all(r["status"] == "PASS" for r in reps)


def detection_scores(log, cfg, sites):
    """Compare edge alerts with the injected ground truth (after the 72 h warm-up)."""
    log = log.drop_duplicates(["t", "site"])
    res = {}
    tp = fp = fn = 0
    spike_tp = spike_n = drop_tp = drop_n = 0
    per_site = {}
    for i, c in enumerate(sites):
        s = generate_series(c, i, cfg["seed"], cfg["T"], cfg["start_hour"], cfg["start_dow"], cfg["fault_rate"])
        lab = s["label"].copy()
        alert = np.zeros(cfg["T"], dtype=bool)
        sub = log[log.site == i]
        alert[sub.t.values] = sub.alert.values.astype(bool)
        lab[:72] = False
        alert[:72] = False
        t_p = int((lab & alert).sum()); f_p = int((~lab & alert).sum()); f_n = int((lab & ~alert).sum())
        tp += t_p; fp += f_p; fn += f_n
        spike = lab & ~s["dropout"]
        drop = s["dropout"] & lab
        spike_n += int(spike.sum()); spike_tp += int((spike & alert).sum())
        drop_n += int(drop.sum()); drop_tp += int((drop & alert).sum())
        per_site[c.name] = {"fault_hours": int(lab.sum()), "detected": t_p, "false_alarms": f_p}
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    res.update({"true_positive_hours": tp, "false_positive_hours": fp, "false_negative_hours": fn,
                "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
                "spike_recall": spike_tp / max(1, spike_n), "dropout_recall": drop_tp / max(1, drop_n),
                "spike_hours": spike_n, "dropout_hours": drop_n, "per_site": per_site})
    return res


def monitor():
    sites = C.SITES
    cfg = dict(seed=50000, T=24 * 28, start_hour=0, start_dow=0, fault_rate=0.006,
               policy="q", model_path=MODEL, use_signal=False, init_soc=C.INIT_SOC,
               outage=(4, 400, 424))                       # Tokyo uplink down for 24 h
    out = run_distributed(cfg, sites, timeout=0.5)
    log = out["log"]
    os.makedirs(RES, exist_ok=True)
    log.to_csv(os.path.join(RES, "monitoring_log.csv"), index=False)
    ev = out["events"]
    if len(ev):
        ev["site"] = ev["site"].map(lambda i: sites[i].name)
    ev.to_csv(os.path.join(RES, "system_events.csv"), index=False)
    det = detection_scores(log, cfg, sites)
    np.savez(os.path.join(RES, "monitor_series.npz"), agg_true=out["agg_true"], agg_reading=out["agg_reading"])
    n_back = int(log.get("backfilled", pd.Series(dtype=bool)).fillna(False).sum())
    rep = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items() if k != "model_path"},
           "hours": cfg["T"], "sites": len(sites), "metrics": out["metrics"], "detection": det,
           "events": ev.to_dict(orient="records"), "backfilled_rows": n_back,
           "log_rows": int(len(log)), "wall_s": out["wall_s"], "messages": out["messages"],
           "mean_round_ms": float(out["step_ms"].mean()),
           "p95_round_ms": float(np.percentile(out["step_ms"], 95))}
    with open(os.path.join(RES, "monitoring_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("Monitoring run: %d hours, %d sites, %d log rows" % (cfg["T"], len(sites), len(log)))
    print("  events:", ev.to_dict(orient="records"))
    print("  back-filled rows after outage:", n_back)
    print("  detection precision=%.3f recall=%.3f F1=%.3f (spike recall %.3f, dropout recall %.3f)" % (
        det["precision"], det["recall"], det["f1"], det["spike_recall"], det["dropout_recall"]))
    print("  mean round %.1f ms, p95 %.1f ms (includes outage timeouts)" % (rep["mean_round_ms"], rep["p95_round_ms"]))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("validate", "all"):
        validate()
    if mode in ("monitor", "all"):
        monitor()
