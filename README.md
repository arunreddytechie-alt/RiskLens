# RiskLens — Cheque Fraud Detection

A prototype fraud detection system for banking cheque operations. It combines signal engineering, machine learning, and graph analysis to surface suspicious accounts and insider threats.

## What it does

- Computes **20 fraud signals** across chequebook lifecycle, cheque usage, employee behavior, withdrawals, and digital channels
- Runs **IsolationForest** to score and rank accounts by anomaly level
- Builds a **graph** (Neo4j + NetworkX) to detect fraud rings — groups of accounts linked through shared employees or signatories
- Serves results via a **FastAPI** backend with a dark-theme dashboard showing ranked accounts, signal breakdowns, and interactive fraud ring networks

## Prerequisites

- Python 3.11+
- Neo4j (running locally on `bolt://localhost:7687`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

**1. Start the pipeline (generates data + runs ML + builds graph):**

```bash
python main_pipeline.py
```

**2. Start the API server:**

```bash
cd api
uvicorn app:app --reload
```

**3. Open the dashboard:**

```
http://localhost:8000/dashboard
```

## Project structure

```
ChequeSentinel/
├── data_generation/       # Synthetic banking data with 3 fraud patterns
├── fraud_signals/         # 20 hand-crafted fraud signal computations
├── digital_signals/       # Online channel behavior signals
├── feature_engineering/   # Combines all signals into feature matrix
├── fraud_model/           # IsolationForest anomaly detector
├── graph_analysis/        # NetworkX graph queries (fraud rings, clusters)
├── graph_builder/         # Neo4j graph population
├── visualizations/        # Score distribution plots
├── api/                   # FastAPI backend + dashboard HTML
└── main_pipeline.py       # End-to-end orchestration
```
