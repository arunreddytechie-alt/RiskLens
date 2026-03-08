"""
Neo4j Graph Builder
====================
Builds a property graph of the cheque lifecycle in Neo4j.

Node labels
-----------
  Account    (account_id)
  Customer   (customer_id)
  Employee   (employee_id)
  Signatory  (signatory_id)
  ChequeBook (cheque_book_id)
  Cheque     (cheque_id)
  Branch     (branch_id)

Relationship types
------------------
  (Account)    -[:OWNED_BY]->    (Customer)
  (Account)    -[:HAS_BOOK]->    (ChequeBook)
  (Employee)   -[:ISSUED]->      (ChequeBook)
  (ChequeBook) -[:CONTAINS]->    (Cheque)
  (Signatory)  -[:SIGNED]->      (Cheque)
  (Employee)   -[:PROCESSED]->   (Cheque)
  (Cheque)     -[:PRESENTED_AT]-> (Branch)
  (Employee)   -[:WORKS_AT]->    (Branch)

Usage (Neo4j must be running)
------------------------------
  builder = Neo4jGraphBuilder(uri="bolt://localhost:7687", user="neo4j", password="password")
  builder.build(cb_df, ch_df, wd_df)
  builder.close()

If Neo4j is unavailable the builder degrades gracefully and logs a warning.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Thin wrapper around the neo4j driver that populates the cheque-fraud graph."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._available = False
        self._connect()

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase, basic_auth  # type: ignore

            self.driver = GraphDatabase.driver(
                self.uri, auth=basic_auth(self.user, self.password)
            )
            self.driver.verify_connectivity()
            self._available = True
            print(f"[Neo4j] Connected to {self.uri}")
        except Exception as exc:
            logger.warning(
                "Neo4j unavailable (%s). Graph will be built in-memory only via NetworkX.", exc
            )
            self._available = False

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _create_constraints(self, session) -> None:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account)    REQUIRE a.account_id    IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer)   REQUIRE c.customer_id   IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Employee)   REQUIRE e.employee_id   IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Signatory)  REQUIRE s.signatory_id  IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:ChequeBook) REQUIRE b.cheque_book_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Cheque)     REQUIRE q.cheque_id     IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Branch)     REQUIRE r.branch_id     IS UNIQUE",
        ]
        for c in constraints:
            session.run(c)

    # ── Node creation helpers ──────────────────────────────────────────────────

    def _merge_accounts_customers(self, session, cb_df: pd.DataFrame, ch_df: pd.DataFrame) -> None:
        rows = pd.concat(
            [
                cb_df[["account_id", "customer_id"]].drop_duplicates(),
                ch_df[["account_id", "customer_id"]].drop_duplicates(),
            ]
        ).drop_duplicates()

        for _, row in rows.iterrows():
            session.run(
                """
                MERGE (a:Account {account_id: $account_id})
                MERGE (c:Customer {customer_id: $customer_id})
                MERGE (a)-[:OWNED_BY]->(c)
                """,
                account_id=row["account_id"],
                customer_id=row["customer_id"],
            )

    def _merge_employees_branches(self, session, cb_df: pd.DataFrame, ch_df: pd.DataFrame) -> None:
        emp_branch = pd.concat(
            [
                cb_df[["employee_id", "branch_id"]].drop_duplicates(),
                ch_df[["employee_id", "branch_id"]].drop_duplicates(),
            ]
        ).dropna(subset=["employee_id", "branch_id"]).drop_duplicates()

        for _, row in emp_branch.iterrows():
            session.run(
                """
                MERGE (e:Employee {employee_id: $employee_id})
                MERGE (b:Branch   {branch_id:   $branch_id})
                MERGE (e)-[:WORKS_AT]->(b)
                """,
                employee_id=row["employee_id"],
                branch_id=row["branch_id"],
            )

    def _merge_chequebooks(self, session, cb_df: pd.DataFrame) -> None:
        for _, row in cb_df.iterrows():
            emp_id = row.get("employee_id")
            br_id  = row.get("branch_id")
            has_employee = pd.notna(emp_id) and pd.notna(br_id)
            if has_employee:
                session.run(
                    """
                    MERGE (a:Account    {account_id:     $account_id})
                    MERGE (b:ChequeBook {cheque_book_id: $cb_id, num_cheques: $num_cheques})
                    MERGE (e:Employee   {employee_id:    $employee_id})
                    MERGE (a)-[:HAS_BOOK]->(b)
                    MERGE (e)-[:ISSUED {timestamp: $ts}]->(b)
                    """,
                    account_id=row["account_id"],
                    cb_id=row["cheque_book_id"],
                    num_cheques=int(row["num_cheques"]),
                    employee_id=emp_id,
                    ts=str(row["timestamp"]),
                )
            else:
                # Online issuance — no employee or branch
                session.run(
                    """
                    MERGE (a:Account    {account_id:     $account_id})
                    MERGE (b:ChequeBook {cheque_book_id: $cb_id, num_cheques: $num_cheques})
                    MERGE (a)-[:HAS_BOOK]->(b)
                    """,
                    account_id=row["account_id"],
                    cb_id=row["cheque_book_id"],
                    num_cheques=int(row["num_cheques"]),
                )

    def _merge_cheques(self, session, ch_df: pd.DataFrame) -> None:
        batch_size = 500
        for start in range(0, len(ch_df), batch_size):
            batch = ch_df.iloc[start : start + batch_size]
            for _, row in batch.iterrows():
                session.run(
                    """
                    MERGE (q:Cheque {cheque_id: $cheque_id})
                      SET q.cheque_number = $cheque_number,
                          q.amount        = $amount,
                          q.timestamp     = $ts

                    MERGE (b:ChequeBook {cheque_book_id: $cb_id})
                    MERGE (b)-[:CONTAINS]->(q)

                    MERGE (s:Signatory {signatory_id: $sig_id})
                    MERGE (s)-[:SIGNED]->(q)

                    MERGE (e:Employee {employee_id: $emp_id})
                    MERGE (e)-[:PROCESSED]->(q)

                    MERGE (r:Branch {branch_id: $branch_id})
                    MERGE (q)-[:PRESENTED_AT]->(r)
                    """,
                    cheque_id=row["cheque_id"],
                    cheque_number=int(row["cheque_number"]),
                    amount=float(row["amount"]),
                    ts=str(row["timestamp"]),
                    cb_id=row["cheque_book_id"],
                    sig_id=row["signatory_id"],
                    emp_id=row["employee_id"],
                    branch_id=row["branch_id"],
                )

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        chequebook_events: pd.DataFrame,
        cheque_events: pd.DataFrame,
        withdrawal_events: pd.DataFrame,
    ) -> bool:
        """
        Populate the Neo4j graph with all entities and relationships.

        Returns True if Neo4j was available and the graph was populated,
        False if it ran in degraded (no-op) mode.
        """
        if not self._available:
            logger.warning("Skipping Neo4j graph population (Neo4j unavailable).")
            return False

        with self.driver.session() as session:
            print("[Neo4j] Creating constraints …")
            self._create_constraints(session)

            print("[Neo4j] Merging accounts & customers …")
            self._merge_accounts_customers(session, chequebook_events, cheque_events)

            print("[Neo4j] Merging employees & branches …")
            self._merge_employees_branches(session, chequebook_events, cheque_events)

            print("[Neo4j] Merging chequebooks …")
            self._merge_chequebooks(session, chequebook_events)

            print(f"[Neo4j] Merging {len(cheque_events)} cheques …")
            self._merge_cheques(session, cheque_events)

        print("[Neo4j] Graph build complete.")
        return True

    # ── Query helpers ──────────────────────────────────────────────────────────

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as a list of dicts."""
        if not self._available:
            return []
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def get_account_neighbourhood(self, account_id: str, depth: int = 2) -> list[dict]:
        """Return all nodes within `depth` hops of an account."""
        cypher = (
            "MATCH path = (a:Account {account_id: $account_id})-[*1.."
            + str(depth)
            + "]->(n) RETURN path"
        )
        return self.query(cypher, account_id=account_id)
