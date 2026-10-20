"""Cut a demo file from train.csv (rows the models never saw during training).

Usage: python src/make_sample.py
Creates data/sample_clicks.csv (1,000,000 rows from row 1,000,000 onward).
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=1_000_000)
ap.add_argument("--n", type=int, default=1_000_000)
a = ap.parse_args()

cols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
df = pd.read_csv(ROOT / "data" / "train.csv", skiprows=range(1, a.start + 1), nrows=a.n,
                 usecols=lambda c: c in cols)
out = ROOT / "data" / "sample_clicks.csv"
df.to_csv(out, index=False)
print(f"Saved {out}: {len(df):,} rows, {df['is_attributed'].mean():.3%} positive")
