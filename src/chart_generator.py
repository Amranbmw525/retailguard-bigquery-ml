import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.google_cloud import (
    get_df_hybrid_detection,
    get_df_store_risk,
    get_df_defect_timeseries
)

OUT_DIR = "images"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def generate_charts():
    _ensure_out_dir()
    sns.set_theme(style="whitegrid", context="talk")

    df_hybrid = get_df_hybrid_detection()
    df_store = get_df_store_risk()
    df_defect = get_df_defect_timeseries()

    # Parse time
    df_hybrid["timestamp"] = pd.to_datetime(df_hybrid["timestamp"], errors="coerce")

    # Extract fraud probability (BQML gives an array probs; prob of class=1 is OFFSET(1) in SQL.
    # If your SQL already extracted it, rename accordingly.
    if "fraud_prob" not in df_hybrid.columns:
        # predicted_fraud_flag_probs might come as list-like; handle both list and string cases
        probs = df_hybrid["predicted_fraud_flag_probs"]
        df_hybrid["fraud_prob"] = probs.apply(lambda x: x[1] if isinstance(x, (list, tuple, np.ndarray)) else np.nan)

    # -----------------------------
    # CHART 1 — Fraud Probability Distribution (Supervised)
    # -----------------------------
    fig1 = plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df_hybrid,
        x="fraud_prob",
        hue="fraud_flag" if "fraud_flag" in df_hybrid.columns else None,
        bins=40,
        kde=True,
        element="step",
        stat="density",
        common_norm=False
    )
    plt.title("Fraud Risk Score Distribution (LogReg) — Separation of Normal vs Fraud")
    plt.xlabel("Predicted Fraud Probability")
    plt.ylabel("Density")
    _save(fig1, "01_fraud_probability_distribution.png")
    plt.close(fig1)

    # -----------------------------
    # CHART 2 — KMeans Anomaly Score Distribution (Unsupervised)
    # -----------------------------
    # Add a simple threshold line: mean + 3*std
    score_mean = df_hybrid["anomaly_score"].mean()
    score_std = df_hybrid["anomaly_score"].std()
    threshold = score_mean + 3 * score_std

    fig2 = plt.figure(figsize=(12, 6))
    sns.histplot(data=df_hybrid, x="anomaly_score", bins=50, kde=True)
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold (mean + 3σ) = {threshold:.2f}")
    plt.title("Anomaly Score (KMeans Distance) — Outlier Tail")
    plt.xlabel("Anomaly Score (nearest_centroids_distance)")
    plt.ylabel("Count")
    plt.legend()
    _save(fig2, "02_anomaly_score_distribution.png")
    plt.close(fig2)

    # -----------------------------
    # CHART 3 — Risk Quadrant (Fraud Prob vs Anomaly Score)
    # -----------------------------
    # Recruiter-friendly: shows hybrid reasoning in one view.
    fig3 = plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_hybrid.sample(min(len(df_hybrid), 15000), random_state=42),
        x="fraud_prob",
        y="anomaly_score",
        hue="hybrid_risk_label",
        alpha=0.45,
        s=25
    )
    plt.axvline(0.5, linestyle="--", linewidth=1)     # adjustable decision threshold
    plt.axhline(threshold, linestyle="--", linewidth=1)
    plt.title("Hybrid Risk Quadrant — Known Fraud (LogReg) vs Emerging Anomalies (KMeans)")
    plt.xlabel("Fraud Probability (Supervised)")
    plt.ylabel("Anomaly Score (Unsupervised)")
    _save(fig3, "03_hybrid_risk_quadrant.png")
    plt.close(fig3)

    # -----------------------------
    # CHART 4 — Store Risk Heatmap (Where to Investigate)
    # -----------------------------
    # Use avg_fraud_prob as primary heatmap metric.
    # If you want to use return_rate instead, swap the value column.
    pivot = df_store.pivot_table(
        index="store_location",
        columns="product_category",
        values="avg_fraud_prob",
        aggfunc="mean"
    )

    fig4 = plt.figure(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Store × Category Risk Heatmap — Avg Fraud Probability (Investigation Map)")
    plt.xlabel("Product Category")
    plt.ylabel("Store Location")
    _save(fig4, "04_store_category_risk_heatmap.png")
    plt.close(fig4)

    # -----------------------------
    # CHART 5 — Mass Defect Detection (Damaged Spike + Price Suppression)
    # -----------------------------
    # Choose the product with the largest damaged spike to showcase the narrative.
    top_product = (
        df_defect.groupby("product_id")["damaged_returns"]
        .max()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )
    df_p = df_defect[df_defect["product_id"] == top_product].copy()
    df_p["date"] = pd.to_datetime(df_p["date"])

    fig5, ax1 = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_p, x="date", y="damaged_returns", ax=ax1, label="Damaged Returns")
    ax1.set_title(f"Mass Defect Signal — Product {top_product}: Damaged Spike + Price Suppression")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Damaged Returns")

    ax2 = ax1.twinx()
    sns.lineplot(data=df_p, x="date", y="avg_price", ax=ax2, linestyle="--", label="Avg Price")
    ax2.set_ylabel("Avg Price")

    # Clean legend across twin axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    _save(fig5, "05_mass_defect_spike_price_suppression.png")
    plt.close(fig5)