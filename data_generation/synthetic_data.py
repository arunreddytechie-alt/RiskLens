"""
Synthetic Banking Data Generator
=================================
Generates realistic cheque lifecycle data for 500 normal accounts and 2 fraud accounts.

Fraud Account 1 — ACC_0501 (single-employee collusion):
  - High chequebook velocity (8 books/year)
  - Single colluding employee (EMP_001) issuing 90% of chequebooks
  - Sequential cheque usage, structured amounts ($9,000–$9,999)
  - Signatory dominance (SIG_FRAUD_001 signs everything)
  - Burst withdrawals, same employee processing 85% of transactions

Fraud Account 2 — ACC_0502 (split-duty cross-branch collusion):
  - TWO colluding employees from DIFFERENT branches:
      EMP_002 @ BRN_02 — issues 100% of chequebooks (issuance controller)
      EMP_003 @ BRN_03 — processes 80% of cheques (processing controller)
  - Split-duty means neither employee alone triggers single-employee thresholds
  - Cheques issued at BRN_02 but PRESENTED & processed at BRN_03 (cross-branch anomaly)
  - Structured amounts ($4,500–$4,999, just below $5k secondary threshold)
  - Single signatory (SIG_FRAUD_002), sequential cheque numbers
  - Moderate velocity (5 books/year) — harder to spot than ACC_0501
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_normal_accounts: int = 500,
    num_employees: int = 50,
    num_branches: int = 20,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    random_seed: int = 42,
) -> dict:
    """
    Generate all synthetic banking event data.

    Returns
    -------
    dict with keys:
        chequebook_events  : DataFrame of chequebook issuance events
        cheque_events      : DataFrame of cheque presentation / processing events
        withdrawal_events  : DataFrame of cash withdrawal events
        fraud_account      : str  – primary fraud account (single-employee)
        fraud_account2     : str  – secondary fraud account (split-duty cross-branch)
        fraud_employee     : str  – primary colluding employee
        fraud_employee2a   : str  – secondary fraud: issuance employee
        fraud_employee2b   : str  – secondary fraud: processing employee
        metadata           : dict of high-level stats
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # ── Entity ID pools ───────────────────────────────────────────────────────
    # +3 slots at the end for three fraud accounts
    all_account_ids  = [f"ACC_{i:04d}"  for i in range(1, num_normal_accounts + 4)]
    all_customer_ids = [f"CUST_{i:04d}" for i in range(1, num_normal_accounts + 4)]
    employee_ids = [f"EMP_{i:03d}" for i in range(1, num_employees + 1)]
    branch_ids   = [f"BRN_{i:02d}" for i in range(1, num_branches + 1)]

    # ── Branch metadata: teller counts ────────────────────────────────────────
    # Fraud branches (BRN_01–04) guaranteed ≥ 5 tellers so that single-employee
    # concentration signals remain meaningful (insider dominance is suspicious
    # precisely because there are enough tellers to distribute work).
    # Some non-fraud branches get 1 teller to test false-positive suppression.
    FRAUD_BRANCH_IDS = {branch_ids[0], branch_ids[1], branch_ids[2], branch_ids[3]}
    teller_counts: dict[str, int] = {}
    for br_id in branch_ids:
        if br_id in FRAUD_BRANCH_IDS:
            teller_counts[br_id] = random.randint(6, 12)   # large branch
        else:
            r = random.random()
            if r < 0.25:
                teller_counts[br_id] = 1                    # rural / single-teller
            elif r < 0.75:
                teller_counts[br_id] = random.randint(3, 7) # suburban
            else:
                teller_counts[br_id] = random.randint(8, 15) # urban flagship

    branch_metadata = pd.DataFrame([
        {
            "branch_id":    br_id,
            "teller_count": teller_counts[br_id],
            "branch_type":  (
                "rural"    if teller_counts[br_id] == 1 else
                "suburban" if teller_counts[br_id] <= 7 else
                "urban"
            ),
        }
        for br_id in branch_ids
    ])

    # ── Fraud Account 1 (single-employee) ─────────────────────────────────────
    FRAUD_ACCOUNT   = all_account_ids[-3]        # ACC_0501
    FRAUD_CUSTOMER  = all_customer_ids[-3]
    FRAUD_EMPLOYEE  = employee_ids[0]             # EMP_001
    FRAUD_SIGNATORY = "SIG_FRAUD_001"
    FRAUD_BRANCH    = branch_ids[0]               # BRN_01

    # ── Fraud Account 2 (split-duty cross-branch) ─────────────────────────────
    FRAUD2_ACCOUNT   = all_account_ids[-2]        # ACC_0502
    FRAUD2_CUSTOMER  = all_customer_ids[-2]
    FRAUD2_EMP_A     = employee_ids[1]            # EMP_002 @ BRN_02
    FRAUD2_EMP_B     = employee_ids[2]            # EMP_003 @ BRN_03
    FRAUD2_SIGNATORY = "SIG_FRAUD_002"
    FRAUD2_BRANCH_A  = branch_ids[1]              # BRN_02 — issuance branch
    FRAUD2_BRANCH_B  = branch_ids[2]              # BRN_03 — presentation branch

    # ── Fraud Account 3 (hybrid channel: online + offline) ────────────────────
    # Gets 3 chequebooks ONLINE (same device, odd hours, short sessions)
    # Gets 3 chequebooks OFFLINE via same corrupt employee (EMP_004)
    # Alternates channels every book to stay under per-channel velocity limits
    # Online and offline requests happen within 24h of each other (coordination)
    FRAUD3_ACCOUNT   = all_account_ids[-1]        # ACC_0503
    FRAUD3_CUSTOMER  = all_customer_ids[-1]
    FRAUD3_EMP       = employee_ids[3]            # EMP_004 — offline insider
    FRAUD3_SIGNATORY = "SIG_FRAUD_003"
    FRAUD3_BRANCH    = branch_ids[3]              # BRN_04
    FRAUD3_DEVICE    = "DEV_FRAUD_001"            # single device for all online requests
    FRAUD3_IP        = "192.168.99.1"             # fixed IP (VPN)

    cb_events: list[dict] = []
    ch_events: list[dict] = []
    wd_events: list[dict] = []

    cb_counter = 1
    ch_counter = 1
    wd_counter = 1

    # ── Normal accounts ───────────────────────────────────────────────────────
    for acc_id, cust_id in zip(all_account_ids[:-3], all_customer_ids[:-3]):
        acct_start = start_dt + timedelta(days=random.randint(0, 45))
        num_books = random.randint(1, 3)

        num_sigs = random.randint(2, 5)
        signatories = [f"SIG_{acc_id}_{j}" for j in range(1, num_sigs + 1)]
        primary_branch = random.choice(branch_ids)

        prev_end = acct_start

        for _ in range(num_books):
            cb_id = f"CB_{cb_counter:05d}"
            cb_counter += 1

            issue_time = prev_end + timedelta(days=random.randint(60, 200))
            if issue_time > end_dt:
                break

            issuing_emp = random.choice(employee_ids)
            num_cheques_in_book = 25

            # Normal accounts: 70% offline, 30% online — varied devices and IPs
            is_online = random.random() < 0.30
            cb_events.append(
                {
                    "account_id":       acc_id,
                    "customer_id":      cust_id,
                    "cheque_book_id":   cb_id,
                    "employee_id":      issuing_emp if not is_online else None,
                    "branch_id":        primary_branch if not is_online else None,
                    "timestamp":        issue_time,
                    "num_cheques":      num_cheques_in_book,
                    "issuance_channel": "online" if is_online else "offline",
                    "device_id":        f"DEV_{acc_id}_{random.randint(1,3)}" if is_online else None,
                    "ip_address":       f"10.{random.randint(0,255)}.{random.randint(0,255)}.1" if is_online else None,
                    "session_duration": random.randint(90, 600) if is_online else None,
                    "request_hour":     random.randint(8, 21) if is_online else None,
                }
            )

            # Non-sequential usage: random subset of cheque numbers
            usage_rate = random.uniform(0.5, 0.9)
            cheques_used = max(1, int(num_cheques_in_book * usage_rate))
            used_numbers = sorted(
                random.sample(range(1, num_cheques_in_book + 1), cheques_used)
            )

            book_start = issue_time + timedelta(days=random.randint(1, 10))

            for cheque_num in used_numbers:
                days_off = random.randint(1, 250)
                ch_time = book_start + timedelta(days=days_off)
                if ch_time > end_dt:
                    break

                ch_id = f"CHQ_{ch_counter:06d}"
                ch_counter += 1

                # Lognormal amounts – realistic banking transactions
                raw_amount = np.random.lognormal(mean=8.0, sigma=1.2)
                amount = round(float(np.clip(raw_amount, 100, 75_000)), 2)

                signatory = random.choice(signatories)
                proc_emp = random.choice(employee_ids)

                ch_events.append(
                    {
                        "cheque_id": ch_id,
                        "cheque_book_id": cb_id,
                        "cheque_number": cheque_num,
                        "account_id": acc_id,
                        "customer_id": cust_id,
                        "signatory_id": signatory,
                        "employee_id": proc_emp,
                        "branch_id": random.choice(branch_ids),
                        "amount": amount,
                        "timestamp": ch_time,
                    }
                )

                # 55% chance of a withdrawal following the cheque (spread over 1–7 days)
                if random.random() < 0.55:
                    w_delay = random.randint(1, 7)
                    w_time = ch_time + timedelta(days=w_delay)
                    if w_time <= end_dt:
                        wd_events.append(
                            {
                                "withdrawal_id": f"WD_{wd_counter:06d}",
                                "account_id": acc_id,
                                "customer_id": cust_id,
                                "cheque_id": ch_id,
                                "amount": round(amount * random.uniform(0.8, 1.0), 2),
                                "timestamp": w_time,
                                "branch_id": random.choice(branch_ids),
                                "employee_id": random.choice(employee_ids),
                            }
                        )
                        wd_counter += 1

            prev_end = issue_time

    # ── Fraud account ──────────────────────────────────────────────────────────
    for cb_idx in range(8):           # 8 chequebooks in ~1 year (very high velocity)
        cb_id = f"CB_{cb_counter:05d}"
        cb_counter += 1

        issue_time = start_dt + timedelta(days=cb_idx * 42 + random.randint(-3, 3))
        if issue_time > end_dt:
            break

        # 90% of chequebooks issued by the colluding employee
        issuing_emp = FRAUD_EMPLOYEE if random.random() < 0.90 else random.choice(employee_ids[1:])
        num_cheques_in_book = 25

        cb_events.append(
            {
                "account_id":       FRAUD_ACCOUNT,
                "customer_id":      FRAUD_CUSTOMER,
                "cheque_book_id":   cb_id,
                "employee_id":      issuing_emp,
                "branch_id":        FRAUD_BRANCH,
                "timestamp":        issue_time,
                "num_cheques":      num_cheques_in_book,
                "issuance_channel": "offline",
                "device_id":        None,
                "ip_address":       None,
                "session_duration": None,
                "request_hour":     None,
            }
        )

        book_start = issue_time + timedelta(days=2)

        # Sequential cheque usage (every cheque, no gaps)
        for cheque_num in range(1, num_cheques_in_book + 1):
            days_off = cheque_num * 3 + random.randint(-1, 1)
            ch_time = book_start + timedelta(days=days_off)
            if ch_time > end_dt:
                break

            ch_id = f"CHQ_{ch_counter:06d}"
            ch_counter += 1

            # Structured amount: $9,000–$9,999 (below $10k CTR threshold)
            amount = round(random.uniform(9_000, 9_999), 2)

            # 85% processed by colluding employee
            proc_emp = FRAUD_EMPLOYEE if random.random() < 0.85 else random.choice(employee_ids[1:])

            ch_events.append(
                {
                    "cheque_id": ch_id,
                    "cheque_book_id": cb_id,
                    "cheque_number": cheque_num,
                    "account_id": FRAUD_ACCOUNT,
                    "customer_id": FRAUD_CUSTOMER,
                    "signatory_id": FRAUD_SIGNATORY,   # single signatory – dominance
                    "employee_id": proc_emp,
                    "branch_id": FRAUD_BRANCH,
                    "amount": amount,
                    "timestamp": ch_time,
                }
            )

        # Burst withdrawals: 4–6 withdrawals on the same day per chequebook
        burst_date = book_start + timedelta(days=random.randint(10, 25))
        if burst_date <= end_dt:
            for _ in range(random.randint(4, 6)):
                hours_off = random.randint(0, 22)
                w_time = burst_date + timedelta(hours=hours_off)
                if w_time <= end_dt:
                    wd_events.append(
                        {
                            "withdrawal_id": f"WD_{wd_counter:06d}",
                            "account_id": FRAUD_ACCOUNT,
                            "customer_id": FRAUD_CUSTOMER,
                            "cheque_id": None,
                            "amount": round(random.uniform(9_000, 9_999), 2),
                            "timestamp": w_time,
                            "branch_id": FRAUD_BRANCH,
                            "employee_id": (
                                FRAUD_EMPLOYEE
                                if random.random() < 0.85
                                else random.choice(employee_ids[1:])
                            ),
                        }
                    )
                    wd_counter += 1

    # ── Fraud Account 2 — Split-Duty Cross-Branch Collusion ───────────────────
    # EMP_002 (BRN_02) issues ALL chequebooks.
    # EMP_003 (BRN_03) processes 80% of cheques at a DIFFERENT branch.
    # Amounts structured $4,500–$4,999 (secondary reporting threshold).
    # 5 chequebooks — moderate velocity, harder to spot individually.

    for cb_idx in range(5):
        cb_id = f"CB_{cb_counter:05d}"
        cb_counter += 1

        issue_time = start_dt + timedelta(days=cb_idx * 60 + random.randint(-4, 4))
        if issue_time > end_dt:
            break

        # EMP_002 issues 100% of chequebooks (issuance controller)
        cb_events.append(
            {
                "account_id":       FRAUD2_ACCOUNT,
                "customer_id":      FRAUD2_CUSTOMER,
                "cheque_book_id":   cb_id,
                "employee_id":      FRAUD2_EMP_A,
                "branch_id":        FRAUD2_BRANCH_A,
                "timestamp":        issue_time,
                "num_cheques":      25,
                "issuance_channel": "offline",
                "device_id":        None,
                "ip_address":       None,
                "session_duration": None,
                "request_hour":     None,
            }
        )

        book_start = issue_time + timedelta(days=3)

        # Sequential cheques, processed at a DIFFERENT branch (BRN_03)
        for cheque_num in range(1, 26):
            days_off = cheque_num * 4 + random.randint(-1, 1)
            ch_time = book_start + timedelta(days=days_off)
            if ch_time > end_dt:
                break

            ch_id = f"CHQ_{ch_counter:06d}"
            ch_counter += 1

            # Structured below $5k threshold
            amount = round(random.uniform(4_500, 4_999), 2)

            # EMP_003 processes 80% — split duty, different branch
            proc_emp = FRAUD2_EMP_B if random.random() < 0.80 else random.choice(employee_ids[3:])

            ch_events.append(
                {
                    "cheque_id": ch_id,
                    "cheque_book_id": cb_id,
                    "cheque_number": cheque_num,
                    "account_id": FRAUD2_ACCOUNT,
                    "customer_id": FRAUD2_CUSTOMER,
                    "signatory_id": FRAUD2_SIGNATORY,
                    "employee_id": proc_emp,
                    "branch_id": FRAUD2_BRANCH_B,       # BRN_03 — different from issuance
                    "amount": amount,
                    "timestamp": ch_time,
                }
            )

        # Burst withdrawals at BRN_03 (processing branch)
        burst_date = book_start + timedelta(days=random.randint(15, 30))
        if burst_date <= end_dt:
            for _ in range(random.randint(3, 5)):
                w_time = burst_date + timedelta(hours=random.randint(0, 20))
                if w_time <= end_dt:
                    wd_events.append(
                        {
                            "withdrawal_id": f"WD_{wd_counter:06d}",
                            "account_id": FRAUD2_ACCOUNT,
                            "customer_id": FRAUD2_CUSTOMER,
                            "cheque_id": None,
                            "amount": round(random.uniform(4_500, 4_999), 2),
                            "timestamp": w_time,
                            "branch_id": FRAUD2_BRANCH_B,
                            "employee_id": FRAUD2_EMP_B if random.random() < 0.80
                                           else random.choice(employee_ids[3:]),
                        }
                    )
                    wd_counter += 1

    # ── Fraud Account 3 — Hybrid Channel (Online + Offline) ───────────────────
    # Alternates: online book → offline book → online book … every ~45 days
    # Online: same device (DEV_FRAUD_001), fixed VPN IP, 2AM–4AM, sessions < 30s
    # Offline: always EMP_004, same branch BRN_04
    # Coordination signal: online and offline requests within 12h of each other
    # Cheque usage: structured amounts $9k–$9.9k, single signatory, sequential

    for cb_idx in range(6):   # 3 online + 3 offline alternating
        cb_id = f"CB_{cb_counter:05d}"
        cb_counter += 1

        issue_time = start_dt + timedelta(days=cb_idx * 45 + random.randint(-2, 2))
        if issue_time > end_dt:
            break

        is_online = (cb_idx % 2 == 0)   # even = online, odd = offline

        if is_online:
            # Online: suspicious device, VPN IP, odd hour (2–4 AM), 8–25s session
            req_hour = random.randint(2, 4)
            cb_events.append({
                "account_id":       FRAUD3_ACCOUNT,
                "customer_id":      FRAUD3_CUSTOMER,
                "cheque_book_id":   cb_id,
                "employee_id":      None,
                "branch_id":        None,
                "timestamp":        issue_time,
                "num_cheques":      25,
                "issuance_channel": "online",
                "device_id":        FRAUD3_DEVICE,      # same device every time
                "ip_address":       FRAUD3_IP,           # same VPN IP every time
                "session_duration": random.randint(8, 25),  # < 30s = bot-like
                "request_hour":     req_hour,
            })
        else:
            # Offline: always EMP_004 at BRN_04 — within 12h of the online request
            offline_time = issue_time + timedelta(hours=random.randint(6, 12))
            if offline_time > end_dt:
                offline_time = issue_time
            cb_events.append({
                "account_id":       FRAUD3_ACCOUNT,
                "customer_id":      FRAUD3_CUSTOMER,
                "cheque_book_id":   cb_id,
                "employee_id":      FRAUD3_EMP,          # always EMP_004
                "branch_id":        FRAUD3_BRANCH,
                "timestamp":        offline_time,
                "num_cheques":      25,
                "issuance_channel": "offline",
                "device_id":        None,
                "ip_address":       None,
                "session_duration": None,
                "request_hour":     None,
            })

        # Sequential cheque usage with structured amounts
        book_start = issue_time + timedelta(days=3)
        for cheque_num in range(1, 26):
            days_off = cheque_num * 3 + random.randint(-1, 1)
            ch_time = book_start + timedelta(days=days_off)
            if ch_time > end_dt:
                break

            ch_id = f"CHQ_{ch_counter:06d}"
            ch_counter += 1
            amount = round(random.uniform(9_000, 9_999), 2)
            proc_emp = FRAUD3_EMP if random.random() < 0.75 else random.choice(employee_ids[4:])

            ch_events.append({
                "cheque_id":      ch_id,
                "cheque_book_id": cb_id,
                "cheque_number":  cheque_num,
                "account_id":     FRAUD3_ACCOUNT,
                "customer_id":    FRAUD3_CUSTOMER,
                "signatory_id":   FRAUD3_SIGNATORY,
                "employee_id":    proc_emp,
                "branch_id":      FRAUD3_BRANCH,
                "amount":         amount,
                "timestamp":      ch_time,
            })

        # Burst withdrawals
        burst_date = book_start + timedelta(days=random.randint(10, 20))
        if burst_date <= end_dt:
            for _ in range(random.randint(3, 5)):
                w_time = burst_date + timedelta(hours=random.randint(0, 20))
                if w_time <= end_dt:
                    wd_events.append({
                        "withdrawal_id": f"WD_{wd_counter:06d}",
                        "account_id":    FRAUD3_ACCOUNT,
                        "customer_id":   FRAUD3_CUSTOMER,
                        "cheque_id":     None,
                        "amount":        round(random.uniform(9_000, 9_999), 2),
                        "timestamp":     w_time,
                        "branch_id":     FRAUD3_BRANCH,
                        "employee_id":   FRAUD3_EMP if random.random() < 0.75
                                         else random.choice(employee_ids[4:]),
                    })
                    wd_counter += 1

    # ── Build DataFrames ───────────────────────────────────────────────────────
    cb_df = pd.DataFrame(cb_events)
    ch_df = pd.DataFrame(ch_events)
    wd_df = pd.DataFrame(wd_events)

    for df in (cb_df, ch_df, wd_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Attach teller_count to events so signal functions have branch context.
    # Online chequebook events have branch_id=None → teller_count stays NaN
    # (signal functions must handle NaN by falling back to a default).
    cb_df = cb_df.merge(branch_metadata[["branch_id", "teller_count"]],
                        on="branch_id", how="left")
    ch_df = ch_df.merge(branch_metadata[["branch_id", "teller_count"]],
                        on="branch_id", how="left")

    return {
        "chequebook_events": cb_df,
        "cheque_events": ch_df,
        "withdrawal_events": wd_df,
        "branch_metadata": branch_metadata,
        "fraud_account":   FRAUD_ACCOUNT,
        "fraud_account2":  FRAUD2_ACCOUNT,
        "fraud_account3":  FRAUD3_ACCOUNT,
        "fraud_employee":  FRAUD_EMPLOYEE,
        "fraud_employee2a": FRAUD2_EMP_A,
        "fraud_employee2b": FRAUD2_EMP_B,
        "fraud_employee3":  FRAUD3_EMP,
        "metadata": {
            "num_accounts": num_normal_accounts + 3,
            "num_employees": num_employees,
            "num_branches": num_branches,
            "total_chequebooks": len(cb_df),
            "total_cheques": len(ch_df),
            "total_withdrawals": len(wd_df),
            "fraud_account":   FRAUD_ACCOUNT,
            "fraud_account2":  FRAUD2_ACCOUNT,
            "fraud_account3":  FRAUD3_ACCOUNT,
            "fraud_employee":  FRAUD_EMPLOYEE,
            "fraud_employee2a": FRAUD2_EMP_A,
            "fraud_employee2b": FRAUD2_EMP_B,
            "fraud_employee3":  FRAUD3_EMP,
            "fraud_signatory":  FRAUD_SIGNATORY,
            "fraud_signatory2": FRAUD2_SIGNATORY,
            "fraud_signatory3": FRAUD3_SIGNATORY,
            "fraud3_device":    FRAUD3_DEVICE,
            "fraud3_ip":        FRAUD3_IP,
        },
    }
