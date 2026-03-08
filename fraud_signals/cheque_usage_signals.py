"""
Cheque Usage Fraud Signals
============================
Signals derived from individual cheque events.

Signals
-------
cheque_frequency          : cheques presented per active day
sequential_cheque_pattern : fraction of consecutive cheque-number pairs (no gaps)
                            → 1.0 means strict sequential use (fraud indicator)
structured_amount_pattern : fraction of cheque amounts in structuring bands
                            (e.g. $9,000–$9,999 just below $10k CTR threshold)
"""

import numpy as np
import pandas as pd

# Structuring bands: (lower_inclusive, upper_exclusive)
STRUCTURING_BANDS = [
    (4_500, 5_000),
    (9_000, 10_000),
    (14_500, 15_000),
    (19_000, 20_000),
    (49_000, 50_000),
]


def _is_structured(amount: float) -> bool:
    return any(lo <= amount < hi for lo, hi in STRUCTURING_BANDS)


def compute_cheque_usage_signals(
    ch_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute cheque-usage signals for every account.

    Parameters
    ----------
    ch_df        : cheque_events DataFrame
    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with columns:
        cheque_frequency, sequential_cheque_pattern, structured_amount_pattern
    """
    rows = []

    for acc_id in account_ids:
        acct = ch_df[ch_df["account_id"] == acc_id].copy()

        if acct.empty:
            rows.append(
                {
                    "account_id": acc_id,
                    "cheque_frequency": 0.0,
                    "sequential_cheque_pattern": 0.0,
                    "structured_amount_pattern": 0.0,
                }
            )
            continue

        n_cheques = len(acct)

        # ── cheque_frequency ──────────────────────────────────────────────────
        span_days = (acct["timestamp"].max() - acct["timestamp"].min()).days
        span_days = max(span_days, 1)
        cheque_frequency = n_cheques / span_days

        # ── sequential_cheque_pattern ─────────────────────────────────────────
        # Per chequebook, compute fraction of consecutive cheque-number pairs
        sequential_scores = []
        for cb_id, group in acct.groupby("cheque_book_id"):
            nums = sorted(group["cheque_number"].tolist())
            if len(nums) < 2:
                sequential_scores.append(0.0)
                continue
            consecutive_pairs = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
            sequential_scores.append(consecutive_pairs / (len(nums) - 1))

        sequential_cheque_pattern = float(np.mean(sequential_scores)) if sequential_scores else 0.0

        # ── structured_amount_pattern ─────────────────────────────────────────
        structured_count = acct["amount"].apply(_is_structured).sum()
        structured_amount_pattern = structured_count / n_cheques

        rows.append(
            {
                "account_id": acc_id,
                "cheque_frequency": round(cheque_frequency, 6),
                "sequential_cheque_pattern": round(sequential_cheque_pattern, 6),
                "structured_amount_pattern": round(float(structured_amount_pattern), 6),
            }
        )

    return pd.DataFrame(rows).set_index("account_id")
