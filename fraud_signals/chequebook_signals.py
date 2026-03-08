"""
Chequebook-Level Fraud Signals
================================
Signals derived from chequebook issuance events.

Signals
-------
chequebook_velocity      : chequebooks requested per active month
chequebook_per_year      : annualised chequebook count

employee_chequebook_bias : branch-normalised employee dominance over chequebook issuance.

                           Raw concentration adjusted by branch teller count:
                             normalised = max(0, raw × teller_count − 1)

                           1-teller branch, 100% → 0.0  ✓ normal
                           8-teller branch,  89% → 6.1  ✗ fraud

branch_velocity_ratio    : this account's chequebook velocity ÷ median velocity of all
                           accounts at the SAME branch (peer group comparison).

                           Teller-count agnostic — catches rural branch fraud:
                             Rural branch normal account: velocity ≈ branch median → ratio ≈ 1.0
                             Rural branch fraud account : velocity 5–10× above median → ratio ≫ 1.0

                           Fallback to global median if branch has fewer than 3 accounts
                           (avoids noisy estimates from very small branches).
"""

import numpy as np
import pandas as pd


def compute_chequebook_signals(
    cb_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute chequebook signals for every account.

    Two-pass approach:
      Pass 1 — compute per-account velocity and primary branch
      Pass 2 — compute branch-peer velocity ratio and full signal row

    Parameters
    ----------
    cb_df        : chequebook_events DataFrame (with teller_count column if available)
    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with columns:
        chequebook_velocity, chequebook_per_year,
        employee_chequebook_bias, branch_velocity_ratio
    """

    # ── Pass 1: per-account velocity + primary branch ─────────────────────────
    acc_velocity: dict[str, float] = {}
    acc_branch:   dict[str, str | None] = {}

    for acc_id in account_ids:
        acct = cb_df[cb_df["account_id"] == acc_id]
        if acct.empty:
            acc_velocity[acc_id] = 0.0
            acc_branch[acc_id]   = None
            continue

        span_days   = (acct["timestamp"].max() - acct["timestamp"].min()).days
        span_months = max(span_days / 30.44, 1)
        acc_velocity[acc_id] = len(acct) / span_months

        # Primary branch = most frequent non-null offline branch_id
        offline_branches = acct["branch_id"].dropna()
        acc_branch[acc_id] = (
            str(offline_branches.mode().iloc[0])
            if not offline_branches.empty
            else None
        )

    # ── Branch-level velocity distributions ───────────────────────────────────
    branch_velocities: dict[str, list[float]] = {}
    for acc_id, br_id in acc_branch.items():
        if br_id is not None:
            branch_velocities.setdefault(br_id, []).append(acc_velocity[acc_id])

    all_vels = list(acc_velocity.values())
    global_median = float(np.median(all_vels)) if all_vels else 1.0

    # Branch median — fall back to global if fewer than 3 accounts at that branch
    branch_median: dict[str, float] = {}
    for br_id, vels in branch_velocities.items():
        branch_median[br_id] = (
            float(np.median(vels)) if len(vels) >= 3 else global_median
        )

    # ── Pass 2: full signal computation ───────────────────────────────────────
    rows = []

    for acc_id in account_ids:
        acct = cb_df[cb_df["account_id"] == acc_id].copy()

        if acct.empty:
            rows.append({
                "account_id":             acc_id,
                "chequebook_velocity":    0.0,
                "chequebook_per_year":    0.0,
                "employee_chequebook_bias": 0.0,
                "branch_velocity_ratio":  0.0,
            })
            continue

        n_books     = len(acct)
        span_days   = (acct["timestamp"].max() - acct["timestamp"].min()).days
        span_months = max(span_days / 30.44, 1)
        span_years  = max(span_days / 365.25, 1 / 12)

        chequebook_velocity = n_books / span_months
        chequebook_per_year = n_books / span_years

        # ── employee_chequebook_bias (branch-normalised) ──────────────────────
        offline_acct = (
            acct[acct["issuance_channel"] == "offline"]
            if "issuance_channel" in acct.columns
            else acct
        )
        emp_counts = offline_acct["employee_id"].dropna().value_counts(normalize=True)
        raw_bias   = float(emp_counts.iloc[0]) if not emp_counts.empty else 0.0

        if "teller_count" in offline_acct.columns:
            avg_tellers = offline_acct["teller_count"].dropna().mean()
            avg_tellers = float(avg_tellers) if not np.isnan(avg_tellers) else 5.0
        else:
            avg_tellers = 5.0
        avg_tellers = max(1.0, avg_tellers)

        employee_chequebook_bias = round(max(0.0, raw_bias * avg_tellers - 1.0), 6)

        # ── branch_velocity_ratio (peer group comparison) ─────────────────────
        br_id      = acc_branch.get(acc_id)
        peer_med   = branch_median.get(br_id, global_median) if br_id else global_median
        peer_med   = max(peer_med, 0.01)          # guard against near-zero median
        branch_velocity_ratio = round(chequebook_velocity / peer_med, 6)

        rows.append({
            "account_id":               acc_id,
            "chequebook_velocity":      round(chequebook_velocity, 6),
            "chequebook_per_year":      round(chequebook_per_year, 6),
            "employee_chequebook_bias": employee_chequebook_bias,
            "branch_velocity_ratio":    branch_velocity_ratio,
        })

    return pd.DataFrame(rows).set_index("account_id")
