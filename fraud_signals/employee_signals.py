"""
Employee Fraud Signals
========================
Signals capturing suspicious employee behaviour relative to an account.

Signals
-------
employee_account_concentration : branch-normalised dominance of the top processing employee
                                 over this account's cheque transactions.

                                   normalised = max(0, raw_fraction × teller_count − 1)

                                 A single-teller branch where one employee handles 100% of
                                 an account's cheques produces 0.0 (expected, not suspicious).
                                 An 8-teller branch where one employee handles 89% produces
                                 6.1 (highly suspicious).

employee_transaction_ratio     : for the account's dominant employee, what fraction
                                 of *that employee's total workload* is this account?
                                 High ≈ the employee is unusually fixated on this account.
"""

import numpy as np
import pandas as pd


def compute_employee_signals(
    ch_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute employee signals for every account.

    Parameters
    ----------
    ch_df        : cheque_events DataFrame (must have 'employee_id' column)
    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with columns:
        employee_account_concentration, employee_transaction_ratio
    """
    # Pre-compute total transactions per employee across all accounts
    emp_total_txn = ch_df["employee_id"].value_counts().to_dict()

    rows = []

    for acc_id in account_ids:
        acct = ch_df[ch_df["account_id"] == acc_id].copy()

        if acct.empty:
            rows.append(
                {
                    "account_id": acc_id,
                    "employee_account_concentration": 0.0,
                    "employee_transaction_ratio": 0.0,
                }
            )
            continue

        n_txns = len(acct)
        emp_counts = acct["employee_id"].value_counts()

        # Concentration within this account (raw fraction)
        top_emp   = emp_counts.index[0]
        top_count = int(emp_counts.iloc[0])
        raw_concentration = top_count / n_txns

        # Branch teller count: use the branch where the top employee processed most cheques
        top_emp_rows = acct[acct["employee_id"] == top_emp]
        if "teller_count" in top_emp_rows.columns:
            avg_tellers = top_emp_rows["teller_count"].dropna().mean()
            avg_tellers = float(avg_tellers) if not np.isnan(avg_tellers) else 5.0
        else:
            avg_tellers = 5.0
        avg_tellers = max(1.0, avg_tellers)

        # Branch-normalised: 0 for expected behaviour, rising for suspicious dominance
        # Formula: max(0, raw × teller_count − 1)
        #   1-teller, 100% → max(0, 1.0×1 − 1) = 0.0  (expected, not suspicious)
        #   8-teller,  89% → max(0, 0.89×8 − 1) = 6.1 (highly suspicious)
        employee_account_concentration = round(
            max(0.0, raw_concentration * avg_tellers - 1.0), 6
        )

        # Ratio: what fraction of the dominant employee's *global* work is this account?
        total_for_top_emp = emp_total_txn.get(top_emp, 1)
        employee_transaction_ratio = top_count / total_for_top_emp

        rows.append(
            {
                "account_id": acc_id,
                "employee_account_concentration": employee_account_concentration,
                "employee_transaction_ratio": round(employee_transaction_ratio, 6),
            }
        )

    return pd.DataFrame(rows).set_index("account_id")
