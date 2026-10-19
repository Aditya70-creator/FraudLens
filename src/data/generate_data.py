import os
import numpy as np
import pandas as pd

FEATURES = [
    "transaction_amount",
    "distance_from_home",
    "hour_of_day",
    "device_risk_score",
    "velocity_last_1h",
    "is_international",
    "previous_fraud_count",
    "account_age_days",
]

def generate_fraud_transactions(n_samples: int = 8000, drift: bool = False, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * (0.03 if drift else 0.015))
    n_legit = n_samples - n_fraud
    shift = 1.35 if drift else 1.0

    legit = pd.DataFrame({
        "transaction_amount": rng.exponential(45, n_legit).clip(1, 1500) * shift,
        "distance_from_home": rng.exponential(8, n_legit).clip(0, 500),
        "hour_of_day": rng.integers(0, 24, n_legit),
        "device_risk_score": rng.uniform(0.01, 0.35, n_legit),
        "velocity_last_1h": rng.poisson(0.8, n_legit),
        "is_international": rng.binomial(1, 0.03, n_legit),
        "previous_fraud_count": 0,
        "account_age_days": rng.integers(90, 2000, n_legit),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "transaction_amount": rng.exponential(350, n_fraud).clip(50, 8000) * shift,
        "distance_from_home": rng.exponential(120, n_fraud).clip(5, 1500),
        "hour_of_day": rng.integers(0, 24, n_fraud),
        "device_risk_score": rng.uniform(0.70, 0.99, n_fraud),
        "velocity_last_1h": rng.poisson(4.5, n_fraud),
        "is_international": rng.binomial(1, 0.45, n_fraud),
        "previous_fraud_count": rng.integers(1, 4, n_fraud),
        "account_age_days": rng.integers(1, 45, n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    ref_df = generate_fraud_transactions(n_samples=8000, drift=False, seed=42)
    drift_df = generate_fraud_transactions(n_samples=2000, drift=True, seed=99)

    ref_df.to_csv("data/raw/transactions_reference.csv", index=False)
    drift_df.to_csv("data/raw/transactions_drifted.csv", index=False)
    print(f"[OK] FraudLens reference dataset ({len(ref_df)} rows) & drifted dataset ({len(drift_df)} rows) generated.")