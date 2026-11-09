"""Generate all charts from the files in results/.   Usage: python scripts/make_charts.py"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from energy import config as C                      # noqa: E402
from energy.agents import NoBattery, RuleTOU        # noqa: E402
from energy.runner import evaluate_policy           # noqa: E402
from energy.simulator import generate_series        # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "charts")
os.makedirs(OUT, exist_ok=True)
COL = {"No battery": "#8d99a6", "Rule-based (TOU)": "#e8a317", "Random": "#c7c7c7",
       "IQL (local reward, no signal)": "#6aa6d6",
       "MARL team reward": "#2e7d5b", "MARL team reward + company signal": "#7a4fb5"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def training_curves():
    log = json.load(open(os.path.join(RES, "training_log.json")))
    names = {"iql": "IQL (local reward, no signal)", "team": "MARL team reward",
             "team_signal": "MARL team reward + company signal"}
    base, _ = evaluate_policy(NoBattery(), range(20000, 20005))
    rule, _ = evaluate_policy(RuleTOU(), range(20000, 20005))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for v, label in names.items():
        runs = [np.array([c for _, c in r["eval_curve"]]) for r in log if r["variant"] == v]
        x = [e for e, _ in [r for r in log if r["variant"] == v][0]["eval_curve"]]
        m, s = np.mean(runs, 0), np.std(runs, 0)
        ax.plot(x, m, label=label, color=COL[label], lw=2)
        ax.fill_between(x, m - s, m + s, color=COL[label], alpha=0.15)
    ax.axhline(base["total_cost"], color=COL["No battery"], ls="--", label="No battery")
    ax.axhline(rule["total_cost"], color=COL["Rule-based (TOU)"], ls="--", label="Rule-based (TOU)")
    ax.set_xlabel("Training episode (1 episode = 7 simulated days)")
    ax.set_ylabel("Company cost per week, greedy policy (USD)")
    ax.set_title("Learning curves (mean and std over training seeds)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=160)
    plt.close(fig)


def comparison():
    df = pd.read_csv(os.path.join(RES, "comparison.csv"))
    df = df[df.strategy != "Random"] if False else df
    order = ["No battery", "Random", "Rule-based (TOU)", "IQL (local reward, no signal)",
             "MARL team reward", "MARL team reward + company signal"]
    df = df.set_index("strategy").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    comps = [("energy_cost", "Energy cost", "#4c78a8"), ("demand_charge", "Demand (peak) charge", "#e45756"),
             ("carbon", "Carbon cost", "#54a24b"), ("degradation", "Battery wear", "#b279a2")]
    df["carbon"] = df["co2_kg"] * C.CARBON_PRICE
    bottom = np.zeros(len(df))
    for key, label, col in comps:
        ax.bar(range(len(df)), df[key], bottom=bottom, label=label, color=col)
        bottom += df[key].values
    for i, (tot, sd) in enumerate(zip(df.total_cost, df.total_cost_std)):
        ax.errorbar(i, tot, yerr=sd if sd > 0 else None, color="k", capsize=3)
        ax.text(i, tot + 1500, f"{tot/1000:.1f}k", ha="center", fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([s.replace(" (", "\n(").replace(" reward ", "\nreward ").replace("MARL ", "MARL\n") for s in df.strategy], fontsize=8)
    ax.set_ylabel("Weekly company cost (USD)")
    ax.set_ylim(0, df.total_cost.max() * 1.3)
    ax.set_title("Total weekly cost by component")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.bar(range(len(df)), df.peak_kw, color=[COL[s] for s in df.strategy])
    for i, v in enumerate(df.peak_kw):
        ax.text(i, v + 30, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(["No\nbattery", "Random", "Rule\n(TOU)", "IQL", "MARL\nteam", "MARL\nteam+sig"], fontsize=8)
    ax.set_ylabel("Company peak grid draw (kW)")
    ax.set_title("Company peak (drives demand charge)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "strategy_comparison.png"), dpi=160)
    plt.close(fig)


def company_profile():
    tr = np.load(os.path.join(RES, "traces.npz"))
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for key, label in [("no_battery", "No battery"), ("rule", "Rule-based (TOU)"), ("team", "MARL team reward")]:
        g = tr[key + "_grid"].sum(1)
        ax.plot(g, label=f"{label} (peak {g.max():.0f} kW)", color=COL[label], lw=1.6)
    ax.set_xlabel("Hour of the week (UTC)")
    ax.set_ylabel("Company grid draw (kW)")
    ax.set_title("Company-wide grid draw over one test week")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "company_profile.png"), dpi=160)
    plt.close(fig)


def site_behaviour():
    tr = np.load(os.path.join(RES, "traces.npz"))
    idx = 1                                   # Frankfurt
    cfg = C.SITES[idx]
    hrs = np.arange(72)
    lh = (hrs + cfg.tz) % 24
    price = np.array([cfg.price * C.PRICE_MULT[C.price_period(int(h))] for h in lh])
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    axes[0].step(hrs, price, color="#e8a317")
    axes[0].set_ylabel("Price (USD/kWh)")
    axes[0].set_title(f"{cfg.name}: tariff, battery state of charge and chosen actions (MARL team reward)")
    axes[1].plot(hrs, tr["team_soc"][:72, idx] * 100, color=COL["MARL team reward"], lw=2)
    axes[1].set_ylabel("Battery SoC (%)")
    frac = np.array(C.ACTION_FRACS)[tr["team_actions"][:72, idx]]
    axes[2].bar(hrs, frac, color=np.where(frac > 0, "#4c78a8", np.where(frac < 0, "#e45756", "#cccccc")))
    axes[2].set_ylabel("Action\n(+charge / -discharge)")
    axes[2].set_xlabel("Hour (UTC), first 72 h of the test episode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "site_behaviour.png"), dpi=160)
    plt.close(fig)


def monitoring():
    rep = json.load(open(os.path.join(RES, "monitoring_report.json")))
    log = pd.read_csv(os.path.join(RES, "monitoring_log.csv")).drop_duplicates(["t", "site"]).sort_values(["t", "site"])
    ser = np.load(os.path.join(RES, "monitor_series.npz"))
    cfgd = rep["config"]
    T = cfgd["T"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.1]})
    ax = axes[0]
    ax.plot(ser["agg_true"], color="#2e7d5b", lw=1, label="Company grid draw (kW)")
    ev = rep["events"]
    out = cfgd["outage"]
    ax.axvspan(out[1], out[2], color="#e45756", alpha=0.2, label="Tokyo uplink down (coordinator uses last known value)")
    ax.set_ylabel("kW")
    ax.set_xlabel("Hour of 4-week run")
    ax.set_title("Company dashboard: aggregate draw over 28 days")
    ax.legend(fontsize=9, loc="upper right")
    ax = axes[1]
    i = 2                                       # Austin: show detections
    c = C.SITES[i]
    s = generate_series(c, i, cfgd["seed"], T, cfgd["start_hour"], cfgd["start_dow"], cfgd["fault_rate"])
    sub = log[log.site == i]
    ax.plot(sub.t, sub.load_reading, color="#4c78a8", lw=0.9, label="Meter reading (kW)")
    lab = np.where(s["label"])[0]
    ax.scatter(lab, np.full(len(lab), -80), marker="|", color="#e45756", s=80, label="Injected fault (ground truth)")
    al = sub[sub.alert.astype(bool)]
    ax.scatter(al.t, al.load_reading, color="#7a4fb5", s=22, zorder=3, label="Edge alert")
    ax.set_title(f"{c.name} site node: edge anomaly detection")
    ax.set_xlabel("Hour of 4-week run")
    ax.set_ylabel("kW")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    d = rep["detection"]
    ax.text(0.01, 0.04, "All sites: precision %.2f, recall %.2f" % (d["precision"], d["recall"]), transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "monitoring_dashboard.png"), dpi=160)
    plt.close(fig)


def scaling():
    p = os.path.join(RES, "scaling.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.nodes, df.mean_round_ms, "o-", label="mean round time", color="#4c78a8")
    ax.plot(df.nodes, df.p95_round_ms, "s--", label="95th percentile", color="#e45756")
    ax.set_xlabel("Number of site nodes (processes)")
    ax.set_ylabel("Coordinator round time (ms per simulated hour)")
    ax.set_title("Scalability of the coordinator round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (training_curves, comparison, company_profile, site_behaviour, monitoring, scaling):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skipped", fn.__name__, "-", e)
