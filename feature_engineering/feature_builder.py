"""
Feature Table Builder
======================
Combines all fraud signals into a single account-level feature matrix
ready for anomaly detection.

Feature columns (20 total)
--------------------------
  Chequebook signals (4)
    chequebook_velocity
    chequebook_per_year
    employee_chequebook_bias
    branch_velocity_ratio

  Cheque usage signals (3)
    cheque_frequency
    sequential_cheque_pattern
    structured_amount_pattern

  Signatory signals (2)
    signatory_entropy
    signatory_concentration

  Employee signals (2)
    employee_account_concentration
    employee_transaction_ratio

  Withdrawal signals (3)
    withdrawal_velocity
    burst_withdrawals
    structured_withdrawals
"""

import pandas as pd

from digital_signals.digital_behavior_signals import compute_digital_signals
from fraud_signals.chequebook_signals import compute_chequebook_signals
from fraud_signals.cheque_usage_signals import compute_cheque_usage_signals
from fraud_signals.employee_signals import compute_employee_signals
from fraud_signals.signatory_signals import compute_signatory_signals
from fraud_signals.withdrawal_signals import compute_withdrawal_signals

FEATURE_COLUMNS = [
    # Chequebook lifecycle
    "chequebook_velocity",
    "chequebook_per_year",
    "employee_chequebook_bias",
    "branch_velocity_ratio",       # peer-group: velocity vs same-branch accounts
    # Cheque usage
    "cheque_frequency",
    "sequential_cheque_pattern",
    "structured_amount_pattern",
    # Signatory
    "signatory_entropy",
    "signatory_concentration",
    # Employee
    "employee_account_concentration",
    "employee_transaction_ratio",
    # Withdrawal
    "withdrawal_velocity",
    "burst_withdrawals",
    "structured_withdrawals",
    # Digital behavior (online channel)
    "channel_mixing_ratio",
    "online_device_concentration",
    "online_odd_hour_ratio",
    "online_session_anomaly",
    "cross_channel_coordination",
    "offline_employee_bias",
]


def build_feature_table(
    chequebook_events: pd.DataFrame,
    cheque_events: pd.DataFrame,
    withdrawal_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all fraud signals and join them into one feature table.

    Parameters
    ----------
    chequebook_events : DataFrame from data_generation
    cheque_events     : DataFrame from data_generation
    withdrawal_events : DataFrame from data_generation

    Returns
    -------
    DataFrame with account_id as index and FEATURE_COLUMNS as columns.
    All NaN values are filled with 0.
    """
    # Union of all account IDs seen across datasets
    account_ids = sorted(
        set(chequebook_events["account_id"].unique())
        | set(cheque_events["account_id"].unique())
        | set(withdrawal_events["account_id"].unique())
    )

    print(f"[FeatureBuilder] Computing signals for {len(account_ids)} accounts …")

    cb_signals  = compute_chequebook_signals(chequebook_events, account_ids)
    ch_signals  = compute_cheque_usage_signals(cheque_events, account_ids)
    sig_signals = compute_signatory_signals(cheque_events, account_ids)
    emp_signals = compute_employee_signals(cheque_events, account_ids)
    wd_signals  = compute_withdrawal_signals(withdrawal_events, account_ids)
    dig_signals = compute_digital_signals(chequebook_events, account_ids)

    feature_table = (
        cb_signals
        .join(ch_signals,  how="outer")
        .join(sig_signals, how="outer")
        .join(emp_signals, how="outer")
        .join(wd_signals,  how="outer")
        .join(dig_signals, how="outer")
    )

    # Fill any missing values with 0 (accounts that had no events in a category)
    feature_table = feature_table.fillna(0.0)

    # Ensure canonical column ordering
    for col in FEATURE_COLUMNS:
        if col not in feature_table.columns:
            feature_table[col] = 0.0

    feature_table = feature_table[FEATURE_COLUMNS]
    feature_table.index.name = "account_id"

    print(f"[FeatureBuilder] Feature table shape: {feature_table.shape}")
    return feature_table
