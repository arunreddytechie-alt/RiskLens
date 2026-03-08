"""
Graph Analyser (NetworkX-based)
================================
Builds an in-memory NetworkX multi-graph from the banking data and implements
three fraud-detection graph queries:

  1. employee_account_clusters  – employees that process disproportionately
                                   many transactions for a single account
  2. signatory_dominance         – signatories that sign across unusually many
                                   accounts (shared/stolen identity risk)
  3. collusion_subgraphs         – dense bipartite subgraphs suggesting an
                                   employee-account collusion ring
                                   (detected via community detection / clique analysis)

All methods return plain Python dicts that are JSON-serialisable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


class GraphAnalyzer:
    """Build and query the cheque-fraud graph using NetworkX."""

    def __init__(self):
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._built = False

    # ── Graph construction ────────────────────────────────────────────────────

    def build_graph(
        self,
        chequebook_events: pd.DataFrame,
        cheque_events: pd.DataFrame,
        withdrawal_events: pd.DataFrame,
        fraud_account: str = "",
        fraud_employee: str = "",
        extra_fraud_accounts: set = None,
        extra_fraud_employees: set = None,
    ) -> None:
        """
        Populate the NetworkX graph with all entity nodes and relationship edges.
        Nodes carry a 'type' attribute; fraud-related nodes are tagged is_fraud=True.
        """
        G = nx.MultiDiGraph()

        fraud_accounts  = ({fraud_account} if fraud_account else set()) | (extra_fraud_accounts or set())
        fraud_employees = ({fraud_employee} if fraud_employee else set()) | (extra_fraud_employees or set())

        # ── Nodes ──────────────────────────────────────────────────────────────
        for acc_id in set(chequebook_events["account_id"]) | set(cheque_events["account_id"]):
            G.add_node(acc_id, type="Account", is_fraud=(acc_id in fraud_accounts))

        for cust_id in set(chequebook_events["customer_id"]) | set(cheque_events["customer_id"]):
            G.add_node(cust_id, type="Customer", is_fraud=False)

        for emp_id in set(chequebook_events["employee_id"].dropna()) | set(cheque_events["employee_id"].dropna()):
            G.add_node(emp_id, type="Employee", is_fraud=(emp_id in fraud_employees))

        for sig_id in cheque_events["signatory_id"].dropna().unique():
            G.add_node(sig_id, type="Signatory", is_fraud=False)

        for cb_id in chequebook_events["cheque_book_id"].unique():
            G.add_node(cb_id, type="ChequeBook", is_fraud=False)

        for ch_id in cheque_events["cheque_id"].unique():
            G.add_node(ch_id, type="Cheque", is_fraud=False)

        for br_id in (set(chequebook_events["branch_id"].dropna())
                      | set(cheque_events["branch_id"].dropna())):
            G.add_node(br_id, type="Branch", is_fraud=False)

        # ── Edges from chequebook events ───────────────────────────────────────
        for _, row in chequebook_events.iterrows():
            G.add_edge(row["account_id"], row["cheque_book_id"],
                       rel="HAS_BOOK", timestamp=str(row["timestamp"]))
            G.add_edge(row["customer_id"], row["account_id"],
                       rel="OWNS", timestamp=str(row["timestamp"]))
            # Store issuance branch on the ISSUED edge for cross-branch detection
            # Skip online events where employee_id / branch_id may be None
            emp_id  = row.get("employee_id")
            br_id   = row.get("branch_id")
            if pd.notna(emp_id) and pd.notna(br_id):
                G.add_edge(emp_id, row["cheque_book_id"],
                           rel="ISSUED", timestamp=str(row["timestamp"]),
                           branch_id=br_id)
                G.add_edge(emp_id, br_id, rel="WORKS_AT")

        # ── Edges from cheque events ───────────────────────────────────────────
        for _, row in cheque_events.iterrows():
            G.add_edge(row["cheque_book_id"], row["cheque_id"],
                       rel="CONTAINS", amount=float(row["amount"]))
            G.add_edge(row["signatory_id"], row["cheque_id"],
                       rel="SIGNED", timestamp=str(row["timestamp"]))
            # Store presentation branch on PROCESSED edge for cross-branch detection
            G.add_edge(row["employee_id"], row["cheque_id"],
                       rel="PROCESSED", timestamp=str(row["timestamp"]),
                       branch_id=row["branch_id"])
            G.add_edge(row["cheque_id"], row["branch_id"],
                       rel="PRESENTED_AT", timestamp=str(row["timestamp"]))

        self.G = G
        self._built = True
        print(f"[GraphAnalyzer] Graph built: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges.")

    # ── Query 1: Employee-Account Clusters ────────────────────────────────────

    def _build_lookup_maps(self) -> tuple[dict, dict]:
        """Build cheque→account and cheque→employee lookup dicts (shared helper)."""
        book_to_account: dict[str, str] = {}
        cheque_to_account: dict[str, str] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "HAS_BOOK":
                book_to_account[v] = u          # book_id  → account_id
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "CONTAINS":
                acc = book_to_account.get(u)
                if acc:
                    cheque_to_account[v] = acc  # cheque_id → account_id
        return cheque_to_account, book_to_account

    def employee_account_clusters(self, concentration_threshold: float = 0.6) -> list[dict]:
        """
        Account-centric cluster detection: find accounts where a single employee
        processes more than `concentration_threshold` of that account's cheques.

        High concentration (e.g. 0.85 = one employee handles 85% of an account's
        cheques) is a strong insider-collusion signal.

        Returns
        -------
        List of dicts: {employee_id, account_id, txn_count, total_account_txns, concentration}
        """
        if not self._built:
            raise RuntimeError("Call build_graph() first.")

        cheque_to_account, _ = self._build_lookup_maps()

        # Per-account: map employee → cheque count
        acc_emp_counts: dict[str, dict[str, int]] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "PROCESSED":
                acc = cheque_to_account.get(v)
                if acc:
                    acc_emp_counts.setdefault(acc, {})
                    acc_emp_counts[acc][u] = acc_emp_counts[acc].get(u, 0) + 1

        results = []
        for acc_id, emp_counts in acc_emp_counts.items():
            total = sum(emp_counts.values())
            if total < 5:
                continue
            top_emp = max(emp_counts, key=emp_counts.get)
            top_cnt = emp_counts[top_emp]
            conc = top_cnt / total
            if conc >= concentration_threshold:
                results.append(
                    {
                        "employee_id": top_emp,
                        "account_id": acc_id,
                        "txn_count": top_cnt,
                        "total_account_txns": total,
                        "concentration": round(conc, 4),
                        "is_fraud_employee": self.G.nodes.get(top_emp, {}).get("is_fraud", False),
                        "is_fraud_account": self.G.nodes.get(acc_id, {}).get("is_fraud", False),
                    }
                )

        results.sort(key=lambda x: x["concentration"], reverse=True)
        return results

    # ── Query 2: Signatory Dominance ──────────────────────────────────────────

    def signatory_dominance(self, min_cheques: int = 10) -> list[dict]:
        """
        Find signatories who sign more than `min_cheques` cheques,
        especially those linked to many different accounts (identity reuse risk).

        Returns
        -------
        List of dicts: {signatory_id, cheques_signed, accounts_linked, dominance_score}
        """
        if not self._built:
            raise RuntimeError("Call build_graph() first.")

        cheque_to_account: dict[str, str] = {}
        book_to_account: dict[str, str] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "HAS_BOOK":
                book_to_account[v] = u
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "CONTAINS":
                acc = book_to_account.get(u)
                if acc:
                    cheque_to_account[v] = acc

        sig_stats: dict[str, dict] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "SIGNED":
                sig_id = u
                ch_id = v
                acc = cheque_to_account.get(ch_id, "UNKNOWN")
                if sig_id not in sig_stats:
                    sig_stats[sig_id] = {"cheques": 0, "accounts": set()}
                sig_stats[sig_id]["cheques"] += 1
                sig_stats[sig_id]["accounts"].add(acc)

        results = []
        for sig_id, stats in sig_stats.items():
            if stats["cheques"] < min_cheques:
                continue
            n_accounts = len(stats["accounts"])
            # Dominance score: high cheque count + low account diversity = suspicious
            dominance = stats["cheques"] / max(n_accounts, 1)
            results.append(
                {
                    "signatory_id": sig_id,
                    "cheques_signed": stats["cheques"],
                    "accounts_linked": n_accounts,
                    "dominance_score": round(dominance, 2),
                }
            )

        results.sort(key=lambda x: x["dominance_score"], reverse=True)
        return results

    # ── Query 3: Collusion Subgraphs ──────────────────────────────────────────

    def collusion_subgraphs(self, min_txns_per_pair: int = 10) -> list[dict]:
        """
        Detect employee-account collusion by finding (employee, account) pairs
        where the employee handles a disproportionate share of the account's
        cheques AND that employee also handles other suspicious accounts.

        Strategy:
          1. Find all (employee, account) pairs with ≥ min_txns_per_pair transactions.
          2. Build a bipartite graph of employees and accounts connected by such
             high-volume pairs.
          3. Each connected component is a potential collusion subgraph.

        Returns
        -------
        List of dicts describing each suspicious component.
        """
        if not self._built:
            raise RuntimeError("Call build_graph() first.")

        cheque_to_account, _ = self._build_lookup_maps()

        # Count (employee, account) transaction pairs
        pair_counts: dict[tuple[str, str], int] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "PROCESSED":
                acc = cheque_to_account.get(v)
                if acc:
                    pair_counts[(u, acc)] = pair_counts.get((u, acc), 0) + 1

        # Keep only high-volume pairs
        hot_pairs = [(emp, acc, cnt) for (emp, acc), cnt in pair_counts.items()
                     if cnt >= min_txns_per_pair]

        if not hot_pairs:
            return []

        # Build bipartite graph
        B = nx.Graph()
        for emp, acc, cnt in hot_pairs:
            B.add_node(emp, bipartite="employee",
                       is_fraud=self.G.nodes.get(emp, {}).get("is_fraud", False))
            B.add_node(acc, bipartite="account",
                       is_fraud=self.G.nodes.get(acc, {}).get("is_fraud", False))
            B.add_edge(emp, acc, weight=cnt)

        results = []
        for component in nx.connected_components(B):
            emps = [n for n in component if B.nodes[n].get("bipartite") == "employee"]
            accs = [n for n in component if B.nodes[n].get("bipartite") == "account"]

            if len(emps) < 1 or len(accs) < 1:
                continue

            is_fraud = any(B.nodes[n].get("is_fraud", False) for n in component)

            # Total transactions in this subgraph
            total_txns = sum(
                B[e][a].get("weight", 0)
                for e in emps for a in accs
                if B.has_edge(e, a)
            )

            results.append(
                {
                    "employees": sorted(emps),
                    "accounts": sorted(accs),
                    "num_employees": len(emps),
                    "num_accounts": len(accs),
                    "total_transactions": total_txns,
                    "subgraph_density": round(nx.density(B.subgraph(component)), 4),
                    "contains_fraud_node": is_fraud,
                }
            )

        # Sort by fraud presence first, then by transaction count
        results.sort(key=lambda x: (x["contains_fraud_node"], x["total_transactions"]), reverse=True)
        return results

    # ── Query 4: Cross-Branch Collusion ──────────────────────────────────────

    def cross_branch_collusion(self, min_cheques: int = 10) -> list[dict]:
        """
        Detect split-duty cross-branch collusion:
        Find accounts where the employee who ISSUED the chequebook works at a
        DIFFERENT branch from the employee who PROCESSED the cheques.

        This is the hallmark of two-employee split-duty fraud:
          - Employee A (branch X) controls issuance
          - Employee B (branch Y) controls processing
          - Neither alone exceeds single-employee concentration thresholds

        Returns
        -------
        List of dicts per account with:
          account_id, issuance_employee, issuance_branch,
          processing_employee, processing_branch,
          cheques_processed, cross_branch (bool), suspicion_score
        """
        if not self._built:
            raise RuntimeError("Call build_graph() first.")

        # book → (issuing employee, issuance branch) — read from ISSUED edge attributes
        book_issuer: dict[str, str] = {}
        book_issuer_branch: dict[str, str] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "ISSUED":
                book_issuer[v] = u                               # book_id → employee_id
                book_issuer_branch[v] = data.get("branch_id", "UNKNOWN")

        # book → account
        book_to_account: dict[str, str] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "HAS_BOOK":
                book_to_account[v] = u

        # cheque → book
        cheque_to_book: dict[str, str] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "CONTAINS":
                cheque_to_book[v] = u

        # acc → {(issuer, issuer_branch, processor, proc_branch): count}
        acc_split: dict[str, dict] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("rel") == "PROCESSED":
                proc_emp    = u
                proc_branch = data.get("branch_id", "UNKNOWN")
                ch_id = v
                book  = cheque_to_book.get(ch_id)
                if not book:
                    continue
                acc    = book_to_account.get(book)
                issuer = book_issuer.get(book)
                if not acc or not issuer:
                    continue
                i_branch = book_issuer_branch.get(book, "UNKNOWN")

                key = (issuer, i_branch, proc_emp, proc_branch)
                acc_split.setdefault(acc, {})
                acc_split[acc][key] = acc_split[acc].get(key, 0) + 1

        results = []
        for acc_id, pair_counts in acc_split.items():
            total = sum(pair_counts.values())
            if total < min_cheques:
                continue

            for (issuer, i_branch, processor, p_branch), cnt in pair_counts.items():
                if issuer == processor:
                    continue   # same employee — not split duty
                cross_branch = (i_branch != p_branch and
                                i_branch != "UNKNOWN" and
                                p_branch != "UNKNOWN")

                pct = cnt / total
                if pct < 0.3:
                    continue  # processor not dominant enough

                # Suspicion score: scaled up when cross-branch confirmed
                suspicion = round(pct * (2.0 if cross_branch else 1.0), 4)

                results.append(
                    {
                        "account_id": acc_id,
                        "issuance_employee": issuer,
                        "issuance_branch": i_branch,
                        "processing_employee": processor,
                        "processing_branch": p_branch,
                        "cheques_processed_by_pair": cnt,
                        "total_account_cheques": total,
                        "processing_concentration": round(pct, 4),
                        "cross_branch": cross_branch,
                        "suspicion_score": suspicion,
                        "is_fraud_account": self.G.nodes.get(acc_id, {}).get("is_fraud", False),
                    }
                )

        results.sort(key=lambda x: x["suspicion_score"], reverse=True)
        return results

    # ── Neighbourhood extraction (for visualisation) ───────────────────────────

    def get_neighbourhood(
        self, node_id: str, radius: int = 2
    ) -> dict:
        """
        Return all nodes and edges within `radius` hops of `node_id`.
        Output is JSON-serialisable.
        """
        if node_id not in self.G:
            return {"nodes": [], "edges": []}

        # ego_graph works on undirected view
        sub = nx.ego_graph(self.G.to_undirected(), node_id, radius=radius)
        nodes = [
            {"id": n, **{k: v for k, v in self.G.nodes[n].items()}}
            for n in sub.nodes()
            if n in self.G.nodes
        ]
        edges = [
            {"source": u, "target": v, **data}
            for u, v, data in sub.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_graph_data(self, output_dir: str = "output") -> None:
        """Save graph summary and query results as JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        emp_clusters = self.employee_account_clusters()
        sig_dom = self.signatory_dominance()
        collusion = self.collusion_subgraphs(min_txns_per_pair=10)

        cross_branch = self.cross_branch_collusion(min_cheques=10)

        summary = {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "node_type_counts": {
                t: sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == t)
                for t in ["Account", "Customer", "Employee", "Signatory",
                          "ChequeBook", "Cheque", "Branch"]
            },
            "employee_account_clusters": emp_clusters[:20],
            "signatory_dominance": sig_dom[:20],
            "collusion_subgraphs": collusion[:10],
            "cross_branch_collusion": cross_branch[:20],
        }

        out_path = Path(output_dir) / "graph_analysis.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[GraphAnalyzer] Graph analysis saved → {out_path}")
