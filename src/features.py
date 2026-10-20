"""Relationship-aware feature framework for ad-click fraud detection.

Used by BOTH training (train.py) and the app (FastAPI/Streamlit) so that
features are built identically everywhere.

Expected columns: ip, app, device, os, channel, click_time, <label>
"""
import numpy as np
import pandas as pd

RAW_FEATURES = ["app", "device", "os", "channel", "hour", "dow"]

# The feature groups of the framework (each is one ablation step).
FEATURE_GROUPS = {
    # How much activity a source produces
    "counts": ["ip_cnt", "ip_app_cnt", "ip_dev_os_cnt", "ip_app_hour_cnt"],
    # How many different entities one IP touches
    "diversity": ["ip_nuniq_app", "ip_nuniq_device", "ip_nuniq_channel", "ip_nuniq_os"],
    # How fast repeat clicks arrive (seconds since previous click)
    "timing": ["ip_prev_gap", "ip_app_prev_gap", "ip_dev_os_prev_gap"],
    # Normalised combinations of the above
    "ratios": ["ip_clicks_per_app", "ip_clicks_per_channel", "ip_app_share"],
    # Cross-entity coordination: device<->IP switching, click bursts and a
    # combined, label-free Cross-Entity Coordination Score
    "coordination": ["dev_os_app_nuniq_ip", "ip_dev_switches", "ip_quick_cnt", "coord_score"],
}

# Components of the Cross-Entity Coordination Score. Each component is turned
# into a percentile rank (0-1) within the scored batch and the ranks are averaged
# with equal weights. No labels are used.
COORD_COMPONENTS = ["ip_cnt", "ip_nuniq_app", "ip_nuniq_channel",
                    "dev_os_app_nuniq_ip", "ip_dev_switches", "ip_quick_cnt"]
QUICK_GAP_SECONDS = 2  # a repeat click within this many seconds counts as a burst


def feature_list(groups):
    """Raw features + the chosen relationship groups, in a fixed order."""
    cols = list(RAW_FEATURES)
    for g in groups:
        cols += FEATURE_GROUPS[g]
    return cols


def prepare(df):
    """Parse time, derive hour/day/dow, sort by time."""
    df = df.copy()
    df["click_time"] = pd.to_datetime(df["click_time"])
    df["ts"] = (df["click_time"] - df["click_time"].min()).dt.total_seconds()
    df["hour"] = df["click_time"].dt.hour
    df["day"] = df["click_time"].dt.day
    df["dow"] = df["click_time"].dt.dayofweek
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _cnt(df, keys):
    return df.groupby(keys)["ts"].transform("size")


def _nuniq(df, keys, col):
    return df.groupby(keys)[col].transform("nunique")


def _gap(df, keys):
    # Previous click only (no look-ahead). First click in a group -> -1.
    return df.groupby(keys)["ts"].diff().fillna(-1)


def add_relationship_features(df):
    """Add all relationship-aware features. Input must come from prepare()."""
    # 1) counts
    df["ip_cnt"] = _cnt(df, ["ip", "day"])
    df["ip_app_cnt"] = _cnt(df, ["ip", "app", "day"])
    df["ip_dev_os_cnt"] = _cnt(df, ["ip", "device", "os", "day"])
    df["ip_app_hour_cnt"] = _cnt(df, ["ip", "app", "day", "hour"])

    # 2) diversity
    df["ip_nuniq_app"] = _nuniq(df, ["ip", "day"], "app")
    df["ip_nuniq_device"] = _nuniq(df, ["ip", "day"], "device")
    df["ip_nuniq_channel"] = _nuniq(df, ["ip", "day"], "channel")
    df["ip_nuniq_os"] = _nuniq(df, ["ip", "day"], "os")

    # 3) timing
    df["ip_prev_gap"] = _gap(df, ["ip"])
    df["ip_app_prev_gap"] = _gap(df, ["ip", "app"])
    df["ip_dev_os_prev_gap"] = _gap(df, ["ip", "device", "os"])

    # 4) ratios
    df["ip_clicks_per_app"] = df["ip_cnt"] / df["ip_nuniq_app"]
    df["ip_clicks_per_channel"] = df["ip_cnt"] / df["ip_nuniq_channel"]
    df["ip_app_share"] = df["ip_app_cnt"] / df["ip_cnt"]

    # 5) cross-entity coordination
    # (a) device <-> IP: how many different IPs the same device/OS/app combination uses
    df["dev_os_app_nuniq_ip"] = _nuniq(df, ["device", "os", "app", "day"], "ip")
    # (b) IP -> device switching: times an IP's device/OS changes between consecutive clicks
    dev_os = df["device"].astype("int64") * 10000 + df["os"].astype("int64")
    prev_dev_os = dev_os.groupby(df["ip"]).shift()
    switched = (prev_dev_os.notna() & (prev_dev_os != dev_os)).astype("int8")
    df["ip_dev_switches"] = switched.groupby([df["ip"], df["day"]]).transform("sum")
    # (c) bursts: repeat clicks from the same IP within a couple of seconds
    quick = ((df["ip_prev_gap"] >= 0) & (df["ip_prev_gap"] <= QUICK_GAP_SECONDS)).astype("int8")
    df["ip_quick_cnt"] = quick.groupby([df["ip"], df["day"]]).transform("sum")
    # (d) Cross-Entity Coordination Score: mean percentile rank of the components
    ranks = pd.concat([df[c].rank(pct=True) for c in COORD_COMPONENTS], axis=1)
    df["coord_score"] = ranks.mean(axis=1)
    return df


def build_features(df):
    """Full pipeline: raw click table -> table with all features."""
    return add_relationship_features(prepare(df))


def load_and_build(path, nrows=None):
    cols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
    df = pd.read_csv(path, nrows=nrows, usecols=lambda c: c in cols)
    return build_features(df)


def time_split(df, train=0.6, val=0.1):
    """Chronological split (rows are already time-sorted): train | val | test."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]