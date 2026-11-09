"""Static configuration: sites, tariffs, battery and reward parameters.

All numbers are SYNTHETIC ASSUMPTIONS chosen to be plausible for a multi-national
company. They are not measured data and are not taken from any real company.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    name: str
    country: str
    tz: int            # offset from UTC in whole hours (approximation)
    profile: str       # "office" (daytime peak) or "factory" (near-flat, shift work)
    base_kw: float     # typical peak electrical demand
    solar_kw: float    # installed PV capacity
    batt_kwh: float    # battery energy capacity
    batt_kw: float     # battery max charge / discharge power
    price: float       # mid-period grid price, USD per kWh
    co2: float         # grid carbon intensity, kg CO2 per kWh
    sun: float         # site sunshine factor (0..1)


SITES = [
    SiteConfig("Mumbai",       "India",        5, "factory", 800, 500, 960, 320, 0.10, 0.70, 0.90),
    SiteConfig("Frankfurt",    "Germany",      1, "office",  600, 250,  720, 240, 0.22, 0.38, 0.55),
    SiteConfig("Austin",       "USA",         -6, "office",  900, 600, 1080, 360, 0.12, 0.42, 0.85),
    SiteConfig("Sao Paulo",    "Brazil",      -3, "factory", 500, 350,  600, 200, 0.16, 0.10, 0.80),
    SiteConfig("Tokyo",        "Japan",        9, "office",  700, 300,  840, 280, 0.24, 0.48, 0.60),
    SiteConfig("Johannesburg", "South Africa", 2, "factory", 450, 300,  540, 180, 0.09, 0.90, 0.90),
]

# Time-of-use tariff: multiplier on the site's mid price, by LOCAL hour.
PRICE_MULT = {"off": 0.5, "mid": 1.0, "peak": 1.8}


def price_period(local_hour: int) -> str:
    if local_hour >= 22 or local_hour < 6:
        return "off"
    if 17 <= local_hour <= 20:
        return "peak"
    return "mid"


# Battery model
EFF = 0.95                  # one-way efficiency (round trip ~0.90)
DEGRADATION = 0.01          # USD per kWh of energy stored (wear cost)
INIT_SOC = 0.5

# Actions: fraction of max battery power; + charge, - discharge
ACTION_FRACS = [0.0, 1.0, 0.5, -0.5, -1.0]
ACTION_NAMES = ["idle", "charge_full", "charge_half", "discharge_half", "discharge_full"]
N_ACTIONS = len(ACTION_FRACS)

# Company-level objective
CARBON_PRICE = 0.05         # USD per kg CO2 (internal carbon price)
DEMAND_RATE = 8.0           # USD per kW of company peak per billing period (one episode)
EPISODE_HOURS = 168         # one 7-day billing period
REWARD_SCALE = 10.0        # divide dollar rewards by this for stable learning
PEAK_REF_RATIO = 0.92       # shaping: hours above 92% of the unmanaged peak are penalised
SHAPING_MU = DEMAND_RATE / 30   # shaping weight (training only; evaluation uses the true charge)

# Company signal broadcast by the coordinator (aggregate grid draw / sum of base loads)
SIGNAL_THRESHOLDS = (0.55, 0.65)   # -> level 0 / 1 / 2
INITIAL_SIGNAL = 1

# Observation discretisation
SOC_BUCKETS = 5
NET_THRESHOLDS = (0.35, 0.70)      # (load - solar) / base_kw -> bucket 0 / 1 / 2
N_HOURS = 24
N_SIGNAL = 3


def n_states(use_signal: bool) -> int:
    return N_HOURS * SOC_BUCKETS * 3 * (N_SIGNAL if use_signal else 1)
