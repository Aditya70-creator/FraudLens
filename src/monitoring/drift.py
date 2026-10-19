import pandas as pd
from scipy.stats import ks_2samp
from src.data.generate_data import FEATURES

def evaluate_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.01) -> dict:
    feature_reports = {}
    drifted_features = []

    for col in FEATURES:
        ref_col = reference_df[col].dropna()
        cur_col = current_df[col].dropna()
        ks_stat, p_value = ks_2samp(ref_col, cur_col)
        is_drifted = bool(p_value < alpha and ks_stat > 0.12)
        if is_drifted:
            drifted_features.append(col)

        feature_reports[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": is_drifted,
            "ref_mean": round(float(ref_col.mean()), 2),
            "cur_mean": round(float(cur_col.mean()), 2),
        }

    drift_share = len(drifted_features) / len(FEATURES)
    retrain_triggered = drift_share >= 0.30

    return {
        "total_features": len(FEATURES),
        "drifted_feature_count": len(drifted_features),
        "drift_share": round(drift_share, 3),
        "drifted_features": drifted_features,
        "retrain_triggered": retrain_triggered,
        "features": feature_reports,
    }