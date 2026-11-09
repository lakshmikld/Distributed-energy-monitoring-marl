"""Edge anomaly detector run on every site node.

Learns, online, the normal meter reading for each (weekday/weekend, local hour)
slot using Welford's running mean/variance, and flags a reading that is far from
that slot's baseline or that reads zero when the baseline is clearly non-zero.
Flagged readings are not used to update the baseline.
"""
import math


class HourlyBaselineDetector:
    def __init__(self, warmup_hours=72, z_thresh=5.0, min_rel_sigma=0.03, min_samples=3):
        self.warmup = warmup_hours
        self.z = z_thresh
        self.min_rel = min_rel_sigma
        self.min_n = min_samples
        self.stats = {}      # key -> [n, mean, M2]
        self.seen = 0

    def update(self, weekend, hour, x):
        """Process one reading. Returns True if it is flagged as anomalous."""
        key = (bool(weekend), int(hour))
        n, mean, m2 = self.stats.get(key, [0, 0.0, 0.0])
        flag = False
        if n >= self.min_n and self.seen >= self.warmup:
            sd = max(math.sqrt(m2 / (n - 1)), self.min_rel * mean)
            if (x <= 0.0 and mean > 0.0) or (sd > 0 and abs(x - mean) / sd > self.z):
                flag = True
        if not flag:
            n += 1
            d = x - mean
            mean += d / n
            m2 += d * (x - mean)
            self.stats[key] = [n, mean, m2]
        self.seen += 1
        return flag
