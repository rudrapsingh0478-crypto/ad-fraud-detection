"""Create a SYNTHETIC click file for smoke-testing the pipeline only.
Do NOT report results from this file in your paper.

Usage: python src/make_demo_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 100_000
heavy = rng.choice(np.arange(1, 60), size=n // 2)       # IPs with many clicks
normal = rng.integers(1000, 12000, size=n // 2)         # ordinary IPs
ip = np.concatenate([heavy, normal])
t = pd.Timestamp("2017-11-07") + pd.to_timedelta(rng.integers(0, 86400, size=n), unit="s")
df = pd.DataFrame(dict(ip=ip, app=rng.integers(1, 40, n), device=rng.integers(1, 6, n),
                       os=rng.integers(1, 25, n), channel=rng.integers(100, 400, n), click_time=t))
df["is_attributed"] = ((df.ip >= 1000) & (rng.random(n) < 0.03)).astype(int)
out = Path(__file__).resolve().parent.parent / "data" / "demo_synthetic.csv"
df.to_csv(out, index=False)
print(f"Saved {out} ({len(df):,} rows, {df.is_attributed.mean():.2%} positive)")
