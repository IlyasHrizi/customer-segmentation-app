"""
model/train.py — Standalone model training script
==================================================
Run this script ONCE to train the K-means model and serialise artefacts.
The Streamlit app (app.py) only loads the .pkl files — it never calls this script.

Usage:
    python model/train.py
"""

from __future__ import annotations

import sys
import pathlib

# Make the project root importable when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config
from utils.helpers import load_data, preprocess, scale_features


# ── Analysis ───────────────────────────────────────────────────────────────────

def find_optimal_k(
    X: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute inertia and silhouette score for every k in config.K_RANGE.

    Silhouette is sampled (sample_size=1000) for speed on larger datasets.

    Args:
        X: Scaled feature matrix of shape (n_samples, n_features).

    Returns:
        Tuple of (inertias, silhouette_scores) — dicts mapping k → metric.
    """
    inertias:    dict[int, float] = {}
    sil_scores:  dict[int, float] = {}

    print(f"\n{'k':>4}  {'inertia':>12}  {'silhouette':>12}")
    print("-" * 34)

    for k in config.K_RANGE:
        km = KMeans(
            n_clusters=k,
            random_state=config.RANDOM_STATE,
            n_init=10,
        )
        labels = km.fit_predict(X)
        inertias[k]   = km.inertia_
        sil_scores[k] = silhouette_score(
            X, labels, sample_size=1000, random_state=config.RANDOM_STATE
        )
        print(f"{k:>4}  {inertias[k]:>12.1f}  {sil_scores[k]:>12.4f}")

    return inertias, sil_scores


def pick_k(
    inertias: dict[int, float],
    sil_scores: dict[int, float],
) -> int:
    """Select the optimal k using a combined elbow + silhouette heuristic.

    Strategy:
        - Compute the second derivative of the inertia curve; the elbow is
          where the drop starts to flatten (max second-derivative point).
        - Cross-check with silhouette: prefer k values with above-average
          silhouette when multiple elbow candidates exist.
        - Hard-cap at config.K_SLIDER_MAX for business interpretability.

    Args:
        inertias:   Mapping of k → inertia from find_optimal_k().
        sil_scores: Mapping of k → silhouette from find_optimal_k().

    Returns:
        Chosen integer k.
    """
    ks  = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]

    # Second derivative of inertia curve
    d1 = [vals[i] - vals[i+1] for i in range(len(vals)-1)]
    d2 = [d1[i] - d1[i+1]    for i in range(len(d1)-1)]

    # k with max second derivative = elbow point (index offset = 2)
    elbow_k = ks[d2.index(max(d2)) + 1]

    # If elbow_k exceeds the slider cap, pull back to cap
    elbow_k = min(elbow_k, config.K_SLIDER_MAX)

    print(f"\n→ Elbow heuristic suggests k={elbow_k}")
    print(f"→ Silhouette at k={elbow_k}: {sil_scores[elbow_k]:.4f}")

    return elbow_k


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, k: int) -> KMeans:
    """Train and return a KMeans model with the chosen k.

    Uses n_init=20 for stability — runs 20 initialisations and keeps the
    best result (lowest inertia).

    Args:
        X: Scaled feature matrix from scale_features().
        k: Number of clusters chosen after elbow/silhouette analysis.

    Returns:
        Fitted KMeans instance.
    """
    print(f"\nTraining KMeans with k={k}, n_init=20, random_state={config.RANDOM_STATE}...")
    km = KMeans(
        n_clusters=k,
        n_init=20,
        max_iter=300,
        random_state=config.RANDOM_STATE,
    )
    km.fit(X)
    print(f"  Final inertia : {km.inertia_:.2f}")
    print(f"  Iterations    : {km.n_iter_}")
    return km


def describe_clusters(
    df_clean: pd.DataFrame,
    model: KMeans,
    X: np.ndarray,
) -> pd.DataFrame:
    """Print cluster sizes and mean feature values to guide labelling.

    Args:
        df_clean: Preprocessed feature DataFrame.
        model:    Fitted KMeans model.
        X:        Scaled feature matrix (same rows as df_clean).

    Returns:
        DataFrame of cluster means (used for display only).
    """
    labels = model.predict(X)
    df_labelled = df_clean.copy()
    df_labelled["Cluster"] = labels

    print("\n── Cluster sizes ──────────────────────────────────")
    sizes = df_labelled["Cluster"].value_counts().sort_index()
    for cluster_id, count in sizes.items():
        pct = count / len(df_labelled) * 100
        label = config.CLUSTER_PROFILES.get(cluster_id, {}).get("label", "?")
        print(f"  Cluster {cluster_id} ({label}): {count} customers ({pct:.1f}%)")

    print("\n── Cluster means ──────────────────────────────────")
    means = df_labelled.groupby("Cluster")[config.FEATURE_COLS].mean().round(1)
    print(means.to_string())

    sil = silhouette_score(X, labels, sample_size=1000, random_state=config.RANDOM_STATE)
    print(f"\n── Final silhouette score: {sil:.4f} ──────────────")

    return means


# ── Persistence ────────────────────────────────────────────────────────────────

def save_artefacts(model: KMeans, scaler) -> None:
    """Serialise the model and scaler to disk with joblib.

    Args:
        model:  Fitted KMeans model.
        scaler: Fitted StandardScaler from scale_features().
    """
    model_path  = pathlib.Path(config.MODEL_PATH)
    scaler_path = pathlib.Path(config.SCALER_PATH)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✓ Model  saved → {model_path}  ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Scaler saved → {scaler_path}  ({scaler_path.stat().st_size / 1024:.1f} KB)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Customer Segmentation — Model Training")
    print("=" * 55)

    # 1. Load & preprocess
    print("\n[1/4] Loading and preprocessing data...")
    df_raw   = load_data(config.DATA_PATH)
    df_clean = preprocess(df_raw)
    print(f"  {len(df_raw)} rows loaded → {len(df_clean)} rows after cleaning")
    print(f"  Features: {config.FEATURE_COLS}")

    # 2. Scale
    print("\n[2/4] Scaling features...")
    X, scaler = scale_features(df_clean)
    print(f"  X shape: {X.shape}  |  mean≈0, std≈1 ✓")

    # 3. Find optimal k
    print("\n[3/4] Running elbow + silhouette analysis...")
    inertias, sil_scores = find_optimal_k(X)
    best_k = pick_k(inertias, sil_scores)
    print(f"\n  ✓ Selected k = {best_k}")

    # 4. Train final model
    print("\n[4/4] Training final model...")
    model = train_model(X, best_k)
    describe_clusters(df_clean, model, X)

    # 5. Save
    save_artefacts(model, scaler)

    print("\n✓ Training complete. Run  streamlit run app.py  to launch the app.")
    print("=" * 55)