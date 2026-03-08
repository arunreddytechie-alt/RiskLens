"""
Digital Behavior Signals
=========================
Signals derived from online chequebook issuance events.
Captures device, IP, session and cross-channel coordination patterns.

Signals
-------
channel_mixing_ratio          : fraction of chequebooks issued online
                                 0 = all offline, 1 = all online
                                 Values near 0.5 with suspicious other signals = hybrid fraud

online_device_concentration   : fraction of online requests from a single device
                                 High = one device controls all online issuance

online_odd_hour_ratio         : fraction of online requests between 10 PM – 6 AM
                                 High = automated or deliberate off-hours activity

online_session_anomaly        : fraction of online sessions shorter than 60 seconds
                                 High = scripted/bot-driven requests, not human

cross_channel_coordination    : are online and offline issuances suspiciously close
                                 in time? (< 24 hours apart)
                                 High = deliberate channel switching for evasion

offline_employee_bias         : fraction of offline chequebooks issued by single employee
                                 High = insider controlling offline channel
"""

import numpy as np
import pandas as pd


def compute_digital_signals(
    cb_df: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """
    Compute digital behavior signals for every account.

    Parameters
    ----------
    cb_df        : chequebook_events DataFrame — must contain columns:
                     issuance_channel  ('online' | 'offline')
                     device_id         (str, NaN for offline)
                     ip_address        (str, NaN for offline)
                     session_duration  (seconds, NaN for offline)
                     request_hour      (0–23, NaN for offline)

    account_ids  : list of all account IDs

    Returns
    -------
    DataFrame indexed by account_id with 6 digital signal columns.
    """
    # Guard: if no digital columns exist, return zeros
    digital_cols = ["issuance_channel", "device_id", "session_duration", "request_hour"]
    if not all(c in cb_df.columns for c in digital_cols):
        empty = pd.DataFrame(0.0, index=account_ids, columns=[
            "channel_mixing_ratio", "online_device_concentration",
            "online_odd_hour_ratio", "online_session_anomaly",
            "cross_channel_coordination", "offline_employee_bias",
        ])
        empty.index.name = "account_id"
        return empty

    rows = []
    for acc_id in account_ids:
        acct = cb_df[cb_df["account_id"] == acc_id].copy()

        if acct.empty:
            rows.append(_zero_row(acc_id))
            continue

        total = len(acct)
        online  = acct[acct["issuance_channel"] == "online"]
        offline = acct[acct["issuance_channel"] == "offline"]

        n_online  = len(online)
        n_offline = len(offline)

        # ── channel_mixing_ratio ─────────────────────────────────────────────
        channel_mixing_ratio = n_online / total

        # ── online_device_concentration ──────────────────────────────────────
        if n_online > 0 and "device_id" in online.columns:
            dev_counts = online["device_id"].dropna().value_counts(normalize=True)
            online_device_concentration = float(dev_counts.iloc[0]) if not dev_counts.empty else 0.0
        else:
            online_device_concentration = 0.0

        # ── online_odd_hour_ratio ─────────────────────────────────────────────
        if n_online > 0 and "request_hour" in online.columns:
            odd = online["request_hour"].dropna().apply(
                lambda h: h >= 22 or h <= 5
            )
            online_odd_hour_ratio = float(odd.mean()) if len(odd) > 0 else 0.0
        else:
            online_odd_hour_ratio = 0.0

        # ── online_session_anomaly ────────────────────────────────────────────
        if n_online > 0 and "session_duration" in online.columns:
            durations = online["session_duration"].dropna()
            if len(durations) > 0:
                short = (durations < 60).sum()   # < 60 seconds = suspicious
                online_session_anomaly = short / len(durations)
            else:
                online_session_anomaly = 0.0
        else:
            online_session_anomaly = 0.0

        # ── cross_channel_coordination ────────────────────────────────────────
        # Fraction of online events that have an offline event within 24 hours
        if n_online > 0 and n_offline > 0:
            online_times  = online["timestamp"].sort_values().values
            offline_times = offline["timestamp"].sort_values().values
            window = pd.Timedelta(hours=24)
            coord_count = 0
            for ot in online_times:
                if any(abs(pd.Timestamp(ot) - pd.Timestamp(ft)) <= window
                       for ft in offline_times):
                    coord_count += 1
            cross_channel_coordination = coord_count / n_online
        else:
            cross_channel_coordination = 0.0

        # ── offline_employee_bias ─────────────────────────────────────────────
        if n_offline > 0 and "employee_id" in offline.columns:
            emp_counts = offline["employee_id"].dropna().value_counts(normalize=True)
            offline_employee_bias = float(emp_counts.iloc[0]) if not emp_counts.empty else 0.0
        else:
            offline_employee_bias = 0.0

        rows.append({
            "account_id":                  acc_id,
            "channel_mixing_ratio":        round(channel_mixing_ratio, 6),
            "online_device_concentration": round(online_device_concentration, 6),
            "online_odd_hour_ratio":       round(online_odd_hour_ratio, 6),
            "online_session_anomaly":      round(online_session_anomaly, 6),
            "cross_channel_coordination":  round(cross_channel_coordination, 6),
            "offline_employee_bias":       round(offline_employee_bias, 6),
        })

    return pd.DataFrame(rows).set_index("account_id")


def _zero_row(acc_id: str) -> dict:
    return {
        "account_id":                  acc_id,
        "channel_mixing_ratio":        0.0,
        "online_device_concentration": 0.0,
        "online_odd_hour_ratio":       0.0,
        "online_session_anomaly":      0.0,
        "cross_channel_coordination":  0.0,
        "offline_employee_bias":       0.0,
    }
