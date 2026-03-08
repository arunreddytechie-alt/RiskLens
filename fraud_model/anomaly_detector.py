"""
Anomaly Detection Model
========================
Uses IsolationForest trained on the account-level feature table to produce
fraud scores and ranked alerts with per-signal explanations.

Design
------
* RobustScaler is applied before IsolationForest to handle heavy outliers.
* IsolationForest `contamination` is set to 1% – roughly 5 accounts out of 501.
* Anomaly scores are normalised to [0, 1] where 1 = most suspicious.
* Per-signal z-scores relative to the population are used as explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from feature_engineering.feature_builder import FEATURE_COLUMNS


class FraudDetector:
    """Train an IsolationForest and produce fraud scores + alert reasons."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            max_samples="auto",
        )
        self.feature_columns = FEATURE_COLUMNS
        self._is_fitted = False
        self.feature_table: Optional[pd.DataFrame] = None
        self.fraud_scores: Optional[pd.DataFrame] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, feature_table: pd.DataFrame) -> "FraudDetector":
        """
        Scale features and fit the IsolationForest.

        Parameters
        ----------
        feature_table : DataFrame with account_id index and FEATURE_COLUMNS.
        """
        self.feature_table = feature_table.copy()
        X = feature_table[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        print("[FraudDetector] IsolationForest fitted on "
              f"{len(feature_table)} accounts with {len(self.feature_columns)} features.")
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, feature_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute fraud scores for all accounts.

        Returns DataFrame with columns:
            account_id, anomaly_score (raw IF), fraud_score (0–1, higher=worse),
            is_flagged (bool)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score().")

        ft = feature_table if feature_table is not None else self.feature_table
        X = ft[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # decision_function: negative means anomalous; score_samples = raw scores
        raw_scores = self.model.score_samples(X_scaled)    # lower = more anomalous
        predictions = self.model.predict(X_scaled)          # -1 = anomaly, +1 = normal

        # Normalise to [0, 1] where 1 = most suspicious
        min_s, max_s = raw_scores.min(), raw_scores.max()
        fraud_score = 1.0 - (raw_scores - min_s) / (max_s - min_s + 1e-12)

        result = pd.DataFrame(
            {
                "account_id": ft.index,
                "anomaly_score": raw_scores,
                "fraud_score": fraud_score,
                "is_flagged": predictions == -1,
            }
        ).set_index("account_id")

        self.fraud_scores = result
        return result

    # ── Alerts ────────────────────────────────────────────────────────────────

    def get_alerts(self, top_n: int = 10) -> list[dict]:
        """
        Return the top-N most suspicious accounts with per-signal explanations.

        Explanation approach: z-score of each signal relative to the full population.
        Signals with |z| > 2 are highlighted as triggered.
        """
        if self.fraud_scores is None:
            self.score()

        scores = self.fraud_scores.sort_values("fraud_score", ascending=False).head(top_n)
        ft = self.feature_table

        pop_mean = ft[self.feature_columns].mean()
        pop_std = ft[self.feature_columns].std().replace(0, 1)

        alerts = []
        for rank, (acc_id, row) in enumerate(scores.iterrows(), start=1):
            acct_features = ft.loc[acc_id, self.feature_columns]
            z_scores = (acct_features - pop_mean) / pop_std

            # Signals that deviate significantly from the population
            triggered = {
                col: {
                    "value": round(float(acct_features[col]), 4),
                    "z_score": round(float(z_scores[col]), 2),
                    "population_mean": round(float(pop_mean[col]), 4),
                }
                for col in self.feature_columns
                if abs(z_scores[col]) > 2.0
            }

            alerts.append(
                {
                    "rank": rank,
                    "account_id": acc_id,
                    "fraud_score": round(float(row["fraud_score"]), 4),
                    "anomaly_score": round(float(row["anomaly_score"]), 4),
                    "is_flagged": bool(row["is_flagged"]),
                    "triggered_signals": triggered,
                    "num_triggered": len(triggered),
                }
            )

        return alerts

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_outputs(self, output_dir: str = "output") -> None:
        """Save fraud scores CSV and alerts JSON to the output directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.fraud_scores is not None:
            score_path = Path(output_dir) / "fraud_scores.csv"
            self.fraud_scores.to_csv(score_path)
            print(f"[FraudDetector] Scores saved → {score_path}")

        alerts = self.get_alerts(top_n=20)
        alert_path = Path(output_dir) / "alerts.json"
        with open(alert_path, "w") as f:
            json.dump(alerts, f, indent=2, default=str)
        print(f"[FraudDetector] Alerts saved → {alert_path}")

        if self.feature_table is not None:
            ft_path = Path(output_dir) / "feature_table.csv"
            self.feature_table.to_csv(ft_path)
            print(f"[FraudDetector] Feature table saved → {ft_path}")
