"""
ChequeSentinel — Main Pipeline
================================
Orchestrates the full end-to-end fraud detection workflow:

  1. Generate synthetic banking data (500 normal + 1 fraud account)
  2. Compute fraud signals across all lifecycle stages
  3. Build account-level feature table
  4. Train IsolationForest anomaly detector
  5. Score accounts and generate ranked alerts
  6. Build in-memory NetworkX graph + populate Neo4j (if available)
  7. Run graph fraud queries (clusters, signatory dominance, collusion)
  8. Generate Plotly visualisations
  9. Save all outputs to output/

Usage
-----
  python main_pipeline.py [--no-graph] [--no-viz]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# ── Module imports ─────────────────────────────────────────────────────────────
from data_generation.synthetic_data import generate_synthetic_data
from feature_engineering.feature_builder import build_feature_table
from fraud_model.anomaly_detector import FraudDetector
from graph_analysis.graph_queries import GraphAnalyzer
from graph_builder.neo4j_builder import Neo4jGraphBuilder
from visualizations.plot_graphs import FraudVisualizer

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════╗
║          ChequeSentinel  —  Fraud Detection          ║
║  Cheque Lifecycle Anomaly Detection Prototype        ║
╚══════════════════════════════════════════════════════╝
"""


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def run_pipeline(
    num_normal_accounts: int = 500,
    enable_graph: bool = True,
    enable_viz: bool = True,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> dict:

    print(BANNER)
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Data Generation
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 1: Generating Synthetic Banking Data")
    data = generate_synthetic_data(num_normal_accounts=num_normal_accounts)

    cb_df           = data["chequebook_events"]
    ch_df           = data["cheque_events"]
    wd_df           = data["withdrawal_events"]
    branch_metadata = data["branch_metadata"]
    fraud_account   = data["fraud_account"]
    fraud_account2  = data["fraud_account2"]
    fraud_account3  = data["fraud_account3"]
    fraud_employee  = data["fraud_employee"]
    metadata        = data["metadata"]

    print(f"  Accounts         : {metadata['num_accounts']}")
    print(f"  Chequebooks      : {metadata['total_chequebooks']}")
    print(f"  Cheque events    : {metadata['total_cheques']}")
    print(f"  Withdrawals      : {metadata['total_withdrawals']}")
    print(f"  Fraud account 1  : {fraud_account}  (single-employee offline)")
    print(f"  Fraud account 2  : {fraud_account2}  (split-duty cross-branch)")
    print(f"  Fraud account 3  : {fraud_account3}  (hybrid online+offline)")
    print(f"    Online device  : {metadata['fraud3_device']}  IP={metadata['fraud3_ip']}")

    # Branch teller summary
    single_teller = (branch_metadata["teller_count"] == 1).sum()
    print(f"\n  Branch teller distribution:")
    print(f"    Single-teller (rural) : {single_teller} branches")
    print(f"    Multi-teller          : {len(branch_metadata) - single_teller} branches")
    fraud_branch_ids = ["BRN_01", "BRN_02", "BRN_03", "BRN_04"]
    for br in fraud_branch_ids:
        row = branch_metadata[branch_metadata["branch_id"] == br]
        if not row.empty:
            tc = row.iloc[0]["teller_count"]
            print(f"    {br} (fraud branch)    : {tc} tellers")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 & 3 — Fraud Signals + Feature Table
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 2-3: Computing Fraud Signals & Building Feature Table")
    feature_table = build_feature_table(cb_df, ch_df, wd_df)

    # Snapshot of fraud account signals
    fraud_signals = feature_table.loc[fraud_account]
    pop_mean = feature_table.mean()
    print(f"\n  {'Signal':<35} {'Fraud Account':>15}  {'Pop. Mean':>12}")
    print("  " + "-" * 65)
    for col in feature_table.columns:
        print(f"  {col:<35} {fraud_signals[col]:>15.4f}  {pop_mean[col]:>12.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 & 5 — Anomaly Detection + Scoring
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 4-5: Training IsolationForest & Scoring Accounts")
    detector = FraudDetector(contamination=0.01)
    detector.fit(feature_table)
    fraud_scores = detector.score()

    # Show fraud account ranks
    ranked = fraud_scores.sort_values("fraud_score", ascending=False).reset_index()
    fraud_rank  = ranked[ranked["account_id"] == fraud_account].index[0] + 1
    fraud_rank2 = ranked[ranked["account_id"] == fraud_account2].index[0] + 1
    fraud_rank3 = ranked[ranked["account_id"] == fraud_account3].index[0] + 1
    fraud_score_val  = fraud_scores.loc[fraud_account,  "fraud_score"]
    fraud_score_val2 = fraud_scores.loc[fraud_account2, "fraud_score"]
    fraud_score_val3 = fraud_scores.loc[fraud_account3, "fraud_score"]

    print(f"\n  Fraud-1 ({fraud_account})  score={fraud_score_val:.4f}   rank=#{fraud_rank}  flagged={bool(fraud_scores.loc[fraud_account,'is_flagged'])}")
    print(f"  Fraud-2 ({fraud_account2})  score={fraud_score_val2:.4f}   rank=#{fraud_rank2}  flagged={bool(fraud_scores.loc[fraud_account2,'is_flagged'])}")
    print(f"  Fraud-3 ({fraud_account3})  score={fraud_score_val3:.4f}   rank=#{fraud_rank3}  flagged={bool(fraud_scores.loc[fraud_account3,'is_flagged'])}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Ranked Alerts
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 6: Top Suspicious Accounts — Alert Summary")
    alerts = detector.get_alerts(top_n=10)

    print(f"\n  {'Rank':<5} {'Account':<12} {'Score':>8}  {'Flagged':>8}  {'Signals Triggered':>20}")
    print("  " + "-" * 60)
    for alert in alerts:
        flag_str = "YES" if alert["is_flagged"] else "no"
        sig_str = f"{alert['num_triggered']} signals"
        print(
            f"  {alert['rank']:<5} {alert['account_id']:<12} "
            f"{alert['fraud_score']:>8.4f}  {flag_str:>8}  {sig_str:>20}"
        )

    # Show triggered signals for the top alert (likely the fraud account)
    if alerts:
        top = alerts[0]
        print(f"\n  Triggered signals for #{top['rank']} ({top['account_id']}):")
        for sig, detail in top["triggered_signals"].items():
            print(
                f"    {sig:<35} value={detail['value']:.4f}  "
                f"z={detail['z_score']:+.2f}  pop_mean={detail['population_mean']:.4f}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7-8 — Graph Analysis
    # ══════════════════════════════════════════════════════════════════════════
    graph_analyzer = GraphAnalyzer()

    if enable_graph:
        section("STEP 7: Building NetworkX Graph")
        graph_analyzer.build_graph(
            cb_df, ch_df, wd_df, fraud_account, fraud_employee,
            extra_fraud_accounts={fraud_account2, fraud_account3},
            extra_fraud_employees={data["fraud_employee2a"], data["fraud_employee2b"],
                                   data["fraud_employee3"]},
        )

        section("STEP 7b: Neo4j Population (optional)")
        neo4j_builder = Neo4jGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
        neo4j_populated = neo4j_builder.build(cb_df, ch_df, wd_df)
        neo4j_builder.close()
        if not neo4j_populated:
            print("  [Note] Neo4j not available — using NetworkX graph only.")

        section("STEP 8: Graph Fraud Queries")

        emp_clusters = graph_analyzer.employee_account_clusters(concentration_threshold=0.6)
        print(f"\n  Employee-Account Clusters (concentration ≥ 0.6): {len(emp_clusters)} found")
        for c in emp_clusters[:5]:
            flag = " [FRAUD]" if c.get("is_fraud_account") or c.get("is_fraud_employee") else ""
            print(
                f"    Emp={c['employee_id']} → Acc={c['account_id']}  "
                f"conc={c['concentration']:.2f}  txns={c['txn_count']}{flag}"
            )

        sig_dom = graph_analyzer.signatory_dominance(min_cheques=10)
        print(f"\n  Signatory Dominance (≥10 cheques signed): {len(sig_dom)} found")
        for s in sig_dom[:5]:
            print(
                f"    Sig={s['signatory_id']}  cheques={s['cheques_signed']}  "
                f"accounts={s['accounts_linked']}  dominance={s['dominance_score']:.2f}"
            )

        collusion = graph_analyzer.collusion_subgraphs(min_txns_per_pair=10)
        print(f"\n  Collusion Subgraphs (sharing ≥2 accounts): {len(collusion)} found")
        for g in collusion[:5]:
            flag = " [FRAUD NODE PRESENT]" if g["contains_fraud_node"] else ""
            print(
                f"    Employees: {g['employees'][:3]}  "
                f"shared_accounts: {g['num_accounts']}  "
                f"density: {g['subgraph_density']:.4f}{flag}"
            )

        cross_branch = graph_analyzer.cross_branch_collusion(min_cheques=10)
        print(f"\n  Cross-Branch Split-Duty Collusion: {len(cross_branch)} found")
        for c in cross_branch[:5]:
            flag = " [FRAUD]" if c.get("is_fraud_account") else ""
            print(
                f"    Acc={c['account_id']}  "
                f"Issuer={c['issuance_employee']}@{c['issuance_branch']} → "
                f"Processor={c['processing_employee']}@{c['processing_branch']}  "
                f"cross={c['cross_branch']}  suspicion={c['suspicion_score']:.3f}{flag}"
            )

        graph_analyzer.save_graph_data(str(OUTPUT_DIR))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Visualisations
    # ══════════════════════════════════════════════════════════════════════════
    if enable_viz:
        section("STEP 9: Generating Visualisations")
        viz = FraudVisualizer(output_dir=str(OUTPUT_DIR / "plots"))

        viz.plot_fraud_score_distribution(fraud_scores, fraud_account)
        viz.plot_feature_heatmap(feature_table, fraud_scores, top_n=15)

        if enable_graph and graph_analyzer._built:
            viz.plot_graph_cluster(
                graph_analyzer.G,
                fraud_account=fraud_account,
                fraud_employee=fraud_employee,
                max_nodes=300,
            )

        viz.plot_signal_radar(feature_table, fraud_account)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Save Outputs
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 10: Saving All Outputs")
    detector.save_outputs(str(OUTPUT_DIR))

    # Save full metadata
    run_metadata = {
        **metadata,
        "fraud_rank": int(fraud_rank),
        "fraud_score": float(fraud_score_val),
        "fraud_flagged": bool(fraud_scores.loc[fraud_account, "is_flagged"]),
        "total_flagged_accounts": int(fraud_scores["is_flagged"].sum()),
        "graph_built": enable_graph,
        "visualizations_generated": enable_viz,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    elapsed = time.time() - t_start
    section(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"\n  Output directory : {OUTPUT_DIR.resolve()}")
    print(f"  Fraud account    : {fraud_account}  rank #{fraud_rank}  score {fraud_score_val:.4f}")
    print(f"\n  Outputs:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            print(f"    {p.relative_to(OUTPUT_DIR)}  ({size:,} bytes)")
    print()

    return {
        "fraud_scores": fraud_scores,
        "feature_table": feature_table,
        "alerts": alerts,
        "graph_analyzer": graph_analyzer,
        "metadata": run_metadata,
    }


# ── API cache injection (used by FastAPI startup) ──────────────────────────────

def load_graph_into_api_cache(graph_analyzer: GraphAnalyzer) -> None:
    """Inject the live GraphAnalyzer into the FastAPI cache for graph endpoints."""
    try:
        from api.app import _cache
        _cache["graph_analyzer"] = graph_analyzer
    except ImportError:
        pass


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChequeSentinel Fraud Detection Pipeline")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph build")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualisations")
    parser.add_argument("--accounts", type=int, default=500, help="Number of normal accounts")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    results = run_pipeline(
        num_normal_accounts=args.accounts,
        enable_graph=not args.no_graph,
        enable_viz=not args.no_viz,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
