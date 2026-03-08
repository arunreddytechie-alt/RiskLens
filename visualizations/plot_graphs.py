"""
Fraud Visualizer
=================
Produces interactive Plotly charts and graph visualisations:

  1. fraud_score_distribution   – histogram of fraud scores for all accounts
  2. feature_heatmap            – normalised signal values for top-N suspicious accounts
  3. graph_cluster_plot         – spring-layout graph coloured by node type,
                                  suspicious nodes highlighted in orange/red
  4. signal_radar               – radar chart comparing a fraud account to population mean

All outputs are saved as self-contained HTML files in output/plots/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Colour palette for node types
NODE_COLOURS = {
    "Account": "#4C72B0",
    "Customer": "#55A868",
    "Employee": "#C44E52",
    "Signatory": "#8172B2",
    "ChequeBook": "#CCB974",
    "Cheque": "#64B5CD",
    "Branch": "#A9A9A9",
}
FRAUD_COLOUR = "#FF4500"  # orange-red for flagged nodes


class FraudVisualizer:
    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fraud Score Distribution ───────────────────────────────────────────

    def plot_fraud_score_distribution(
        self,
        fraud_scores: pd.DataFrame,
        fraud_account: Optional[str] = None,
    ) -> str:
        """Histogram of fraud_score for all accounts, with fraud account annotated."""
        fig = go.Figure()

        normal = fraud_scores[fraud_scores.index != fraud_account]["fraud_score"]
        fig.add_trace(
            go.Histogram(
                x=normal,
                nbinsx=50,
                name="Normal accounts",
                marker_color="#4C72B0",
                opacity=0.75,
            )
        )

        if fraud_account and fraud_account in fraud_scores.index:
            fraud_score_val = fraud_scores.loc[fraud_account, "fraud_score"]
            fig.add_vline(
                x=fraud_score_val,
                line_color=FRAUD_COLOUR,
                line_width=3,
                annotation_text=f"Fraud account ({fraud_account})<br>Score: {fraud_score_val:.3f}",
                annotation_position="top right",
                annotation_font_color=FRAUD_COLOUR,
            )

        fig.update_layout(
            title="Fraud Score Distribution — IsolationForest",
            xaxis_title="Fraud Score (0=Normal, 1=Most Suspicious)",
            yaxis_title="Account Count",
            template="plotly_white",
            showlegend=True,
        )

        path = self.output_dir / "fraud_score_distribution.html"
        fig.write_html(str(path))
        print(f"[Visualizer] Saved → {path}")
        return str(path)

    # ── 2. Feature Heatmap ────────────────────────────────────────────────────

    def plot_feature_heatmap(
        self,
        feature_table: pd.DataFrame,
        fraud_scores: pd.DataFrame,
        top_n: int = 15,
    ) -> str:
        """Normalised signal values for the top-N suspicious accounts."""
        top_accounts = fraud_scores.sort_values("fraud_score", ascending=False).head(top_n).index
        subset = feature_table.loc[top_accounts].copy()

        # Min-max normalise each column for visual comparability
        norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-12)
        norm = norm.round(3)

        # Short labels
        col_labels = [c.replace("_", " ").title() for c in norm.columns]

        fig = go.Figure(
            data=go.Heatmap(
                z=norm.values,
                x=col_labels,
                y=[str(idx) for idx in norm.index],
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=1,
                text=norm.values.round(2),
                texttemplate="%{text}",
                colorbar=dict(title="Normalised<br>Signal Intensity"),
            )
        )
        fig.update_layout(
            title=f"Signal Heatmap — Top {top_n} Suspicious Accounts",
            xaxis_title="Fraud Signal",
            yaxis_title="Account ID",
            template="plotly_white",
            height=max(400, top_n * 35),
        )

        path = self.output_dir / "feature_heatmap.html"
        fig.write_html(str(path))
        print(f"[Visualizer] Saved → {path}")
        return str(path)

    # ── 3. Graph Cluster Plot ─────────────────────────────────────────────────

    def plot_graph_cluster(
        self,
        G: nx.MultiDiGraph,
        fraud_account: Optional[str] = None,
        fraud_employee: Optional[str] = None,
        max_nodes: int = 300,
    ) -> str:
        """
        Spring-layout visualisation of the banking graph.
        Fraud-related nodes are highlighted in orange-red.
        Subsamples to max_nodes for performance.
        """
        # Work on undirected for layout
        H = G.to_undirected()

        # If too large, sample ego-graph around fraud account + random sample
        if H.number_of_nodes() > max_nodes:
            keep = set()
            if fraud_account and fraud_account in H:
                keep |= set(nx.ego_graph(H, fraud_account, radius=2).nodes())
            if fraud_employee and fraud_employee in H:
                keep |= set(nx.ego_graph(H, fraud_employee, radius=2).nodes())
            remaining = list(set(H.nodes()) - keep)
            import random
            sample_extra = max(0, min(max_nodes - len(keep), len(remaining)))
            if sample_extra > 0:
                keep |= set(random.sample(remaining, sample_extra))
            H = H.subgraph(keep).copy()

        pos = nx.spring_layout(H, seed=42, k=0.8)

        fraud_nodes = set()
        if fraud_account:
            fraud_nodes.add(fraud_account)
        if fraud_employee:
            fraud_nodes.add(fraud_employee)

        # Build traces per node type
        traces = []
        node_types = set(nx.get_node_attributes(G, "type").values())

        for ntype in sorted(node_types):
            nodes_of_type = [
                n for n in H.nodes()
                if G.nodes.get(n, {}).get("type") == ntype
            ]
            if not nodes_of_type:
                continue

            x_coords = [pos[n][0] for n in nodes_of_type]
            y_coords = [pos[n][1] for n in nodes_of_type]
            colours = [
                FRAUD_COLOUR if n in fraud_nodes else NODE_COLOURS.get(ntype, "#888888")
                for n in nodes_of_type
            ]
            sizes = [18 if n in fraud_nodes else 10 for n in nodes_of_type]
            labels = [
                f"{n}<br>Type: {ntype}" + (" [FRAUD]" if n in fraud_nodes else "")
                for n in nodes_of_type
            ]

            traces.append(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    name=ntype,
                    marker=dict(size=sizes, color=colours, line=dict(width=1, color="white")),
                    text=labels,
                    hoverinfo="text",
                )
            )

        # Edge trace
        edge_x, edge_y = [], []
        for u, v in H.edges():
            if u in pos and v in pos:
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#CCCCCC"),
            hoverinfo="none",
            name="Edges",
            showlegend=False,
        )

        fig = go.Figure(data=[edge_trace] + traces)
        fig.update_layout(
            title="Cheque Fraud Graph — Node Cluster Visualisation",
            showlegend=True,
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
        )

        path = self.output_dir / "graph_cluster.html"
        fig.write_html(str(path))
        print(f"[Visualizer] Saved → {path}")
        return str(path)

    # ── 4. Signal Radar Chart ─────────────────────────────────────────────────

    def plot_signal_radar(
        self,
        feature_table: pd.DataFrame,
        fraud_account: str,
    ) -> str:
        """
        Radar chart comparing the fraud account's signal values
        to the population mean and 95th percentile.
        """
        features = feature_table.columns.tolist()
        pop_mean = feature_table[features].mean()
        pop_p95 = feature_table[features].quantile(0.95)
        fraud_vals = feature_table.loc[fraud_account, features]

        # Normalise everything to [0,1] using the max value per feature
        scale = feature_table[features].max().replace(0, 1)
        fraud_norm = (fraud_vals / scale).fillna(0).tolist()
        mean_norm = (pop_mean / scale).fillna(0).tolist()
        p95_norm = (pop_p95 / scale).fillna(0).tolist()

        labels = [c.replace("_", " ").title() for c in features]
        labels_closed = labels + [labels[0]]
        fraud_closed = fraud_norm + [fraud_norm[0]]
        mean_closed = mean_norm + [mean_norm[0]]
        p95_closed = p95_norm + [p95_norm[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=mean_closed, theta=labels_closed,
            fill="toself", name="Population Mean",
            line_color="#4C72B0", opacity=0.5,
        ))
        fig.add_trace(go.Scatterpolar(
            r=p95_closed, theta=labels_closed,
            fill="toself", name="Population 95th Pct",
            line_color="#CCB974", opacity=0.4,
        ))
        fig.add_trace(go.Scatterpolar(
            r=fraud_closed, theta=labels_closed,
            fill="toself", name=f"Fraud Account ({fraud_account})",
            line_color=FRAUD_COLOUR, opacity=0.8,
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Signal Radar — {fraud_account} vs Population",
            template="plotly_white",
            showlegend=True,
        )

        path = self.output_dir / "signal_radar.html"
        fig.write_html(str(path))
        print(f"[Visualizer] Saved → {path}")
        return str(path)
