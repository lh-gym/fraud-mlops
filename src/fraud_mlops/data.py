from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "claim_amount",
    "customer_age",
    "policy_tenure_months",
    "prior_claims",
    "payment_delay_days",
]

CATEGORICAL_COLUMNS = ["channel", "region", "policy_type"]
TARGET_COLUMN = "fraud_label"
TIME_COLUMN = "event_date"


def generate_claims_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp("2023-01-01")
    dates = start_date + pd.to_timedelta(rng.integers(0, 600, size=n_samples), unit="D")

    claim_amount = rng.lognormal(mean=9.1, sigma=0.6, size=n_samples)
    customer_age = rng.integers(18, 86, size=n_samples)
    policy_tenure = rng.integers(1, 260, size=n_samples)
    prior_claims = rng.poisson(0.9, size=n_samples)
    payment_delay_days = rng.integers(0, 45, size=n_samples)
    channel = rng.choice(["agent", "online", "partner"], size=n_samples, p=[0.46, 0.41, 0.13])
    region = rng.choice(["north", "south", "east", "west"], size=n_samples)
    policy_type = rng.choice(["auto", "home", "health"], size=n_samples, p=[0.52, 0.28, 0.20])

    linear_score = (
        0.00006 * (claim_amount - 8000.0)
        + 0.30 * (prior_claims >= 2).astype(float)
        + 0.22 * (payment_delay_days > 12).astype(float)
        + 0.18 * (channel == "online").astype(float)
        + 0.15 * (policy_type == "auto").astype(float)
        + 0.12 * (customer_age < 25).astype(float)
        + rng.normal(loc=0.0, scale=0.35, size=n_samples)
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-linear_score))
    fraud_label = rng.binomial(1, np.clip(fraud_probability, 0.01, 0.97), size=n_samples)

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{seed}-{idx:07d}" for idx in range(n_samples)],
            "claim_amount": claim_amount.round(2),
            "customer_age": customer_age,
            "policy_tenure_months": policy_tenure,
            "prior_claims": prior_claims,
            "payment_delay_days": payment_delay_days,
            "channel": channel,
            "region": region,
            "policy_type": policy_type,
            "event_date": pd.to_datetime(dates),
            "fraud_label": fraud_label.astype(np.int64),
        }
    )
    return df.sort_values("event_date").reset_index(drop=True)


def split_by_event_time(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * train_ratio))
    train = ordered.iloc[:split_idx].copy()
    oot = ordered.iloc[split_idx:].copy()
    return train, oot


def train_validation_split(df: pd.DataFrame, validation_ratio: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split_idx = max(1, int(len(df) * (1.0 - validation_ratio)))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train = df.iloc[train_idx].copy().reset_index(drop=True)
    val = df.iloc[val_idx].copy().reset_index(drop=True)
    return train, val


def inject_oot_shift(oot_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate temporal distribution drift for OOT validation."""
    rng = np.random.default_rng(seed + 7)
    shifted = oot_df.copy()
    shifted["claim_amount"] = shifted["claim_amount"] * 1.10
    channel_boost_mask = rng.random(len(shifted)) < 0.22
    shifted.loc[channel_boost_mask, "channel"] = "online"
    shifted["payment_delay_days"] = np.clip(shifted["payment_delay_days"] + rng.integers(0, 6, size=len(shifted)), 0, 60)
    return shifted

