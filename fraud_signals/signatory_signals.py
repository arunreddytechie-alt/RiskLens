"""
Signatory Fraud Signals
=========================
Signals derived from cheque-signing behaviour.

Signals
-------
signatory_entropy       : Shannon entropy of the signatory distribution.
                          Low entropy (≈0) means one signatory signs everything → suspicious.
signatory_concentration : Fraction of cheques signed by the single dominant signatory.
                          High value (≈1) is a strong fraud indicator.
"""

import math

import numpy as np
import pandas as pd


def _shannon_entropy(counts: pd.Series) -> float:
    """Shannon entropy (in nats) of a frequency distribution."""
    props = counts / counts.sum()
    return float(-np.sum(props * np.log(props + 1e-12)))


def compute_signatory_signals(
    ch_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute signatory signals for every account.

    Parameters
    ----------
    ch_df        : cheque_events DataFrame (must have 'signatory_id' column)
    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with columns:
        signatory_entropy, signatory_concentration
    """
    rows = []

    for acc_id in account_ids:
        acct = ch_df[ch_df["account_id"] == acc_id].copy()

        if acct.empty or "signatory_id" not in acct.columns:
            rows.append(
                {
                    "account_id": acc_id,
                    "signatory_entropy": 0.0,
                    "signatory_concentration": 0.0,
                }
            )
            continue

        sig_counts = acct["signatory_id"].value_counts()
        n_total = sig_counts.sum()

        signatory_entropy = _shannon_entropy(sig_counts)

        # Concentration = fraction held by top signatory
        signatory_concentration = float(sig_counts.iloc[0]) / n_total if n_total > 0 else 0.0

        rows.append(
            {
                "account_id": acc_id,
                "signatory_entropy": round(signatory_entropy, 6),
                "signatory_concentration": round(signatory_concentration, 6),
            }
        )

    return pd.DataFrame(rows).set_index("account_id")
