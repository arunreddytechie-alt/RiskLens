"""
ChequeSentinel FastAPI Application
====================================
REST API exposing fraud detection results and graph query endpoints.

Run with:
    uvicorn api.app:app --reload --port 8000

Endpoints
---------
GET  /                               Health check
GET  /api/v1/health                  Health check (JSON)
GET  /api/v1/metadata                Pipeline run metadata
GET  /api/v1/fraud-scores            All account fraud scores (paginated)
GET  /api/v1/fraud-scores/{id}       Specific account fraud score
GET  /api/v1/alerts                  Top-N suspicious accounts with reasons
GET  /api/v1/accounts/{id}/signals   Full signal breakdown for an account
GET  /api/v1/graph/account/{id}      Graph neighbourhood of an account
GET  /api/v1/graph/clusters          Employee-account cluster analysis
GET  /api/v1/graph/signatory         Signatory dominance analysis
GET  /api/v1/graph/collusion         Collusion subgraph detection
POST /api/v1/run-pipeline            Trigger a fresh pipeline run

WebSocket
---------
WS   /ws/signals                     Stream live fraud signal updates
WS   /ws/alerts                      Stream real-time alert feed
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

app = FastAPI(
    title="ChequeSentinel",
    description="End-to-end cheque fraud detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "output"))
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

# ── In-memory cache (populated on startup or after pipeline run) ───────────────
_cache: dict = {
    "fraud_scores": None,
    "feature_table": None,
    "alerts": None,
    "graph_analysis": None,
    "metadata": None,
    "graph_analyzer": None,
}


def _load_outputs() -> bool:
    """Load pipeline outputs from disk into the cache."""
    try:
        scores_path = OUTPUT_DIR / "fraud_scores.csv"
        if scores_path.exists():
            _cache["fraud_scores"] = pd.read_csv(scores_path, index_col="account_id")

        ft_path = OUTPUT_DIR / "feature_table.csv"
        if ft_path.exists():
            _cache["feature_table"] = pd.read_csv(ft_path, index_col="account_id")

        alerts_path = OUTPUT_DIR / "alerts.json"
        if alerts_path.exists():
            with open(alerts_path) as f:
                _cache["alerts"] = json.load(f)

        graph_path = OUTPUT_DIR / "graph_analysis.json"
        if graph_path.exists():
            with open(graph_path) as f:
                _cache["graph_analysis"] = json.load(f)

        meta_path = OUTPUT_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                _cache["metadata"] = json.load(f)

        return True
    except Exception as exc:
        print(f"[API] Warning: could not load outputs — {exc}")
        return False


@app.on_event("startup")
async def startup_event():
    _load_outputs()


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def root():
    """Serve the live fraud-detection dashboard."""
    if DASHBOARD_HTML.exists():
        return DASHBOARD_HTML.read_text(encoding="utf-8")
    return HTMLResponse("<h2>ChequeSentinel — run the pipeline first, then reload.</h2>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return a minimal SVG favicon so the browser stops 404-ing."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#4f8ef7"/>'
        '<text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle" '
        'font-size="20" font-family="sans-serif" fill="white">C</text>'
        '</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/v1/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "outputs_loaded": _cache["fraud_scores"] is not None,
    }


# ── Metadata ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/metadata", tags=["Pipeline"])
def get_metadata():
    if _cache["metadata"] is None:
        raise HTTPException(status_code=404, detail="No pipeline metadata found. Run the pipeline first.")
    return _cache["metadata"]


# ── Fraud Scores ───────────────────────────────────────────────────────────────

@app.get("/api/v1/fraud-scores", tags=["Fraud Detection"])
def get_fraud_scores(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort_by: str = Query("fraud_score", regex="^(fraud_score|anomaly_score|account_id)$"),
    descending: bool = Query(True),
):
    if _cache["fraud_scores"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")

    df = _cache["fraud_scores"].copy()
    df = df.sort_values(sort_by, ascending=not descending)

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": page_df.reset_index().to_dict(orient="records"),
    }


@app.get("/api/v1/fraud-scores/{account_id}", tags=["Fraud Detection"])
def get_account_fraud_score(account_id: str):
    if _cache["fraud_scores"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")

    df = _cache["fraud_scores"]
    if account_id not in df.index:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found.")

    row = df.loc[account_id].to_dict()
    row["account_id"] = account_id
    return row


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/alerts", tags=["Fraud Detection"])
def get_alerts(top_n: int = Query(10, ge=1, le=50)):
    if _cache["alerts"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")
    return _cache["alerts"][:top_n]


# ── Account Signals ────────────────────────────────────────────────────────────

@app.get("/api/v1/accounts/{account_id}/signals", tags=["Fraud Detection"])
def get_account_signals(account_id: str):
    if _cache["feature_table"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")

    ft = _cache["feature_table"]
    if account_id not in ft.index:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found.")

    signals = ft.loc[account_id].to_dict()
    signals["account_id"] = account_id

    # Add z-scores vs population
    pop_mean = ft.mean()
    pop_std = ft.std().replace(0, 1)
    z_scores = ((ft.loc[account_id] - pop_mean) / pop_std).to_dict()
    signals["z_scores"] = {k: round(float(v), 3) for k, v in z_scores.items()}

    if _cache["fraud_scores"] is not None and account_id in _cache["fraud_scores"].index:
        scores = _cache["fraud_scores"].loc[account_id]
        signals["fraud_score"] = float(scores["fraud_score"])
        signals["is_flagged"] = bool(scores["is_flagged"])

    return signals


# ── Graph Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/v1/graph/account/{account_id}", tags=["Graph Analysis"])
def get_account_neighbourhood(account_id: str, radius: int = Query(2, ge=1, le=3)):
    """Return nodes and edges within `radius` hops of the account in the graph."""
    if _cache["graph_analyzer"] is None:
        raise HTTPException(
            status_code=503,
            detail="Graph analyzer not loaded. Run pipeline with graph=True.",
        )
    result = _cache["graph_analyzer"].get_neighbourhood(account_id, radius=radius)
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found in graph.")
    return result


@app.get("/api/v1/graph/clusters", tags=["Graph Analysis"])
def get_employee_clusters(concentration_threshold: float = Query(0.6, ge=0.0, le=1.0)):
    if _cache["graph_analysis"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")
    return _cache["graph_analysis"].get("employee_account_clusters", [])


@app.get("/api/v1/graph/signatory", tags=["Graph Analysis"])
def get_signatory_dominance():
    if _cache["graph_analysis"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")
    return _cache["graph_analysis"].get("signatory_dominance", [])


@app.get("/api/v1/graph/collusion", tags=["Graph Analysis"])
def get_collusion_subgraphs():
    if _cache["graph_analysis"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")
    return _cache["graph_analysis"].get("collusion_subgraphs", [])


@app.get("/api/v1/graph/fraud-rings", tags=["Graph Analysis"])
def get_fraud_rings():
    """
    Build fraud rings — groups of accounts connected through shared employees
    or cross-branch collusion. Surfaces accounts that ML alone would miss
    (low fraud score but sharing a corrupt employee with a flagged account).
    """
    ga = _cache.get("graph_analysis")
    fs = _cache.get("fraud_scores")

    if ga is None:
        raise HTTPException(status_code=404, detail="Run pipeline with --graph first.")

    def _account_info(acc_id: str) -> dict:
        score, flagged = 0.0, False
        if fs is not None and acc_id in fs.index:
            score   = float(fs.loc[acc_id, "fraud_score"])
            flagged = bool(fs.loc[acc_id, "is_flagged"])
        return {"account_id": acc_id, "fraud_score": round(score, 4), "is_flagged": flagged}

    rings = []
    ring_id = 1

    # ── Rings from employee-account clusters ───────────────────────────────
    # Group all high-concentration (account, employee) pairs by employee.
    # One employee dominating multiple accounts = a ring.
    emp_map: dict[str, list[dict]] = {}
    for c in ga.get("employee_account_clusters", []):
        emp_map.setdefault(c["employee_id"], []).append(c)

    for emp_id, clusters in emp_map.items():
        accounts = []
        for c in clusters:
            info = _account_info(c["account_id"])
            info["concentration"] = round(c["concentration"], 3)
            info["txn_count"]     = c["txn_count"]
            accounts.append(info)

        max_score    = max(a["fraud_score"] for a in accounts)
        num_flagged  = sum(1 for a in accounts if a["is_flagged"])
        is_fraud_emp = any(c.get("is_fraud_employee") for c in clusters)

        rings.append({
            "ring_id":         f"RING_{ring_id:03d}",
            "connection_type": "employee_dominance",
            "risk_level":      "HIGH" if max_score >= 0.7 else "MEDIUM" if max_score >= 0.35 else "LOW",
            "accounts":        accounts,
            "connectors": [{
                "type":     "employee",
                "id":       emp_id,
                "is_fraud": is_fraud_emp,
            }],
            "num_accounts":    len(accounts),
            "num_flagged":     num_flagged,
            "hidden_accounts": len(accounts) - num_flagged,
            "max_score":       round(max_score, 4),
        })
        ring_id += 1

    # ── Rings from cross-branch collusion ──────────────────────────────────
    for cb in ga.get("cross_branch_collusion", []):
        info = _account_info(cb["account_id"])
        info["suspicion_score"] = round(cb.get("suspicion_score", 0), 3)
        rings.append({
            "ring_id":         f"RING_{ring_id:03d}",
            "connection_type": "cross_branch_collusion",
            "risk_level":      "HIGH" if info["fraud_score"] >= 0.7 or cb.get("is_fraud_account") else "MEDIUM",
            "accounts":        [info],
            "connectors": [
                {
                    "type":   "employee", "id": cb["issuance_employee"],
                    "branch": cb["issuance_branch"],  "role": "Issuance",
                    "is_fraud": cb.get("is_fraud_account", False),
                },
                {
                    "type":   "employee", "id": cb["processing_employee"],
                    "branch": cb["processing_branch"], "role": "Processing",
                    "is_fraud": cb.get("is_fraud_account", False),
                },
            ],
            "num_accounts":    1,
            "num_flagged":     1 if info["is_flagged"] else 0,
            "hidden_accounts": 0,
            "max_score":       info["fraud_score"],
            "cross_branch":    cb.get("cross_branch", False),
            "suspicion_score": round(cb.get("suspicion_score", 0), 3),
        })
        ring_id += 1

    rings.sort(key=lambda r: r["max_score"], reverse=True)
    return {"total_rings": len(rings), "rings": rings}


@app.get("/api/v1/graph/cross-branch", tags=["Graph Analysis"])
def get_cross_branch_collusion():
    """
    Split-duty cross-branch collusion: accounts where the chequebook issuer
    and the cheque processor are different employees at different branches.
    """
    if _cache["graph_analysis"] is None:
        raise HTTPException(status_code=404, detail="Run pipeline first.")
    return _cache["graph_analysis"].get("cross_branch_collusion", [])


# ── Pipeline Trigger ───────────────────────────────────────────────────────────

@app.post("/api/v1/run-pipeline", tags=["Pipeline"])
def run_pipeline():
    """Trigger a full pipeline run. Returns when complete."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "main_pipeline.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    success = result.returncode == 0
    if success:
        _load_outputs()

    return {
        "success": success,
        "stdout": result.stdout[-3000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "message": "Pipeline complete." if success else "Pipeline failed — see stderr.",
    }


# ── WebSocket Connection Manager ───────────────────────────────────────────────

class _ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages to all."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_signals_manager = _ConnectionManager()
_alerts_manager = _ConnectionManager()


# ── WebSocket: /ws/signals ─────────────────────────────────────────────────────

@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """
    Stream live fraud signal snapshots for every account.

    On connect: sends the full feature table as a batch (type=snapshot).
    Then every 10 s: sends an incremental update (type=update) with the
    top-20 accounts ranked by fraud_score and their latest signal values.

    Message schema
    --------------
    {
      "type": "snapshot" | "update" | "ping",
      "timestamp": "ISO-8601",
      "accounts": [
        {
          "account_id": str,
          "fraud_score": float,
          "is_flagged": bool,
          "signals": { signal_name: value, ... }
        }, ...
      ]
    }
    """
    await _signals_manager.connect(websocket)
    try:
        import datetime

        def _build_payload(msg_type: str, top_n: int = 20) -> dict:
            accounts = []
            ft = _cache.get("feature_table")
            fs = _cache.get("fraud_scores")
            if ft is not None and fs is not None:
                ranked = fs.sort_values("fraud_score", ascending=False).head(top_n)
                for acc_id, score_row in ranked.iterrows():
                    signals = ft.loc[acc_id].to_dict() if acc_id in ft.index else {}
                    accounts.append(
                        {
                            "account_id": acc_id,
                            "fraud_score": round(float(score_row["fraud_score"]), 4),
                            "anomaly_score": round(float(score_row["anomaly_score"]), 4),
                            "is_flagged": bool(score_row["is_flagged"]),
                            "signals": {k: round(float(v), 4) for k, v in signals.items()},
                        }
                    )
            return {
                "type": msg_type,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "total_accounts": len(fs) if fs is not None else 0,
                "accounts": accounts,
            }

        # Send full snapshot on connect
        await websocket.send_json(_build_payload("snapshot", top_n=50))

        # Push updates every 10 s; also respond to client pings
        while True:
            try:
                # Non-blocking receive with timeout — client can send {"type":"ping"}
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
                if msg.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong",
                         "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
                    )
                elif msg.get("type") == "subscribe" and msg.get("account_id"):
                    # Client subscribes to a specific account
                    acc_id = msg["account_id"]
                    ft = _cache.get("feature_table")
                    fs = _cache.get("fraud_scores")
                    if ft is not None and acc_id in ft.index:
                        import datetime as _dt
                        signals = ft.loc[acc_id].to_dict()
                        score_row = fs.loc[acc_id] if fs is not None else {}
                        await websocket.send_json(
                            {
                                "type": "account_detail",
                                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                                "account_id": acc_id,
                                "fraud_score": round(float(score_row.get("fraud_score", 0)), 4),
                                "is_flagged": bool(score_row.get("is_flagged", False)),
                                "signals": {k: round(float(v), 4) for k, v in signals.items()},
                            }
                        )
            except asyncio.TimeoutError:
                # 10 s elapsed — push a periodic update
                await websocket.send_json(_build_payload("update", top_n=20))

    except WebSocketDisconnect:
        _signals_manager.disconnect(websocket)


# ── WebSocket: /ws/alerts ──────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Stream real-time fraud alert feed.

    On connect: sends the current top-10 alerts immediately.
    Every 15 s: re-sends the latest alert list so the client stays in sync.

    Message schema
    --------------
    {
      "type": "alerts",
      "timestamp": "ISO-8601",
      "alerts": [ <same shape as GET /api/v1/alerts> ]
    }
    """
    await _alerts_manager.connect(websocket)
    try:
        import datetime

        def _build_alerts(top_n: int = 10) -> dict:
            return {
                "type": "alerts",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "alerts": _cache.get("alerts", [])[:top_n],
            }

        await websocket.send_json(_build_alerts())

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=15.0)
                top_n = int(msg.get("top_n", 10))
                await websocket.send_json(_build_alerts(top_n=top_n))
            except asyncio.TimeoutError:
                await websocket.send_json(_build_alerts())

    except WebSocketDisconnect:
        _alerts_manager.disconnect(websocket)
