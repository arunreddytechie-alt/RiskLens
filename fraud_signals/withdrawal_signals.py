"""
Withdrawal Fraud Signals
==========================
Signals derived from cash withdrawal events.

Signals
-------
withdrawal_velocity        : withdrawals per active day
burst_withdrawals          : maximum number of withdrawals within any rolling 24-hour window
structured_withdrawals     : fraction of withdrawals whose amounts fall in known structuring bands
"""

import numpy as np
import pandas as pd

# Same structuring bands as cheque_usage_signals
STRUCTURING_BANDS = [
    (4_500, 5_000),
    (9_000, 10_000),
    (14_500, 15_000),
    (19_000, 20_000),
    (49_000, 50_000),
]


def _is_structured(amount: float) -> bool:
    return any(lo <= amount < hi for lo, hi in STRUCTURING_BANDS)


def _max_burst(timestamps: pd.Series, window_hours: int = 24) -> int:
    """Return the maximum number of events within any rolling window of `window_hours`."""
    if len(timestamps) == 0:
        return 0
    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    window = pd.Timedelta(hours=window_hours)
    max_count = 1
    for i, t in enumerate(ts_sorted):
        count = int(((ts_sorted >= t) & (ts_sorted < t + window)).sum())
        if count > max_count:
            max_count = count
    return max_count


def compute_withdrawal_signals(
    wd_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute withdrawal signals for every account.

    Parameters
    ----------
    wd_df        : withdrawal_events DataFrame
    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with columns:
        withdrawal_velocity, burst_withdrawals, structured_withdrawals
    """
    rows = []

    for acc_id in account_ids:
        acct = wd_df[wd_df["account_id"] == acc_id].copy()

        if acct.empty:
            rows.append(
                {
                    "account_id": acc_id,
                    "withdrawal_velocity": 0.0,
                    "burst_withdrawals": 0,
                    "structured_withdrawals": 0.0,
                }
            )
            continue

        n_wd = len(acct)

        # ── withdrawal_velocity ───────────────────────────────────────────────
        span_days = (acct["timestamp"].max() - acct["timestamp"].min()).days
        span_days = max(span_days, 1)
        withdrawal_velocity = n_wd / span_days

        # ── burst_withdrawals ─────────────────────────────────────────────────
        burst_withdrawals = _max_burst(acct["timestamp"])

        # ── structured_withdrawals ────────────────────────────────────────────
        structured_count = acct["amount"].apply(_is_structured).sum()
        structured_withdrawals = structured_count / n_wd

        rows.append(
            {
                "account_id": acc_id,
                "withdrawal_velocity": round(withdrawal_velocity, 6),
                "burst_withdrawals": int(burst_withdrawals),
                "structured_withdrawals": round(float(structured_withdrawals), 6),
            }
        )

    return pd.DataFrame(rows).set_index("account_id")
