"""
utils/helpers.py — Reusable helper functions
=============================================
All data loading, preprocessing, validation, and plotting utilities.
Import these in app.py and model/train.py — never duplicate logic.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load the raw customer CSV from disk.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        Raw DataFrame with all original columns intact.

    Raises:
        FileNotFoundError: If no CSV exists at the given path.
    """
    resolved = pathlib.Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place your CSV in the data/ folder and update DATA_PATH in config.py."
        )
    return pd.read_csv(resolved)


# ── Preprocessing & feature engineering ───────────────────────────────────────

def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError listing every source column that is absent."""
    required = {
        "Year_Birth", "Income", "Kidhome", "Teenhome", "Recency",
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} required column(s): {sorted(missing)}"
        )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw DataFrame and engineer the 8 model-ready features.

    Steps:
        1. Drop rows where Income > 600 000 or Age > 100 (outliers).
        2. Impute missing Income with column median.
        3. Age  = 2024 − Year_Birth
        4. TotalSpend = sum of all Mnt* columns
        5. NumChildren = Kidhome + Teenhome
        6. NumPurchases = sum of all Num*Purchases columns
        7. CampaignAccepted = sum of AcceptedCmp1..5
        8. Return only config.FEATURE_COLS, reset index.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame with exactly config.FEATURE_COLS, zero nulls.

    Raises:
        ValueError: If required source columns are absent.
        RuntimeError: If nulls persist after imputation.
    """
    _check_required_columns(df)
    out = df.copy()

    out = out[out["Income"] < 600_000]
    out = out[2024 - out["Year_Birth"] <= 100]

    out["Income"] = out["Income"].fillna(out["Income"].median())

    out["Age"]              = 2024 - out["Year_Birth"]
    out["TotalSpend"]       = out[["MntWines","MntFruits","MntMeatProducts",
                                    "MntFishProducts","MntSweetProducts","MntGoldProds"]].sum(axis=1)
    out["NumChildren"]      = out["Kidhome"] + out["Teenhome"]
    out["NumPurchases"]     = out[["NumDealsPurchases","NumWebPurchases",
                                    "NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)
    out["CampaignAccepted"] = out[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                                    "AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)

    result = out[config.FEATURE_COLS].copy().reset_index(drop=True)

    if result.isnull().sum().sum() > 0:
        raise RuntimeError("Preprocessing left null values — check input data.")

    return result


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return rounded describe() table for the engineered feature columns."""
    return df[config.FEATURE_COLS].describe().round(2)


# ── Scaling ────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler on df and return (X_scaled, fitted_scaler).

    Args:
        df: Preprocessed DataFrame with config.FEATURE_COLS.

    Returns:
        Tuple (X_scaled, scaler). Persist scaler with joblib.dump().
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[config.FEATURE_COLS].values)
    return X_scaled, scaler


def scale_input(
    age: float,
    income: float,
    total_spend: float,
    num_children: int,
    num_purchases: int,
    campaign_accepted: int,
    recency: int,
    num_web_visits: int,
    scaler: StandardScaler,
) -> np.ndarray:
    """Scale one new customer's raw values using the pre-fitted scaler.

    Returns:
        Array of shape (1, 8) ready for model.predict().
    """
    raw = np.array([[age, income, total_spend, num_children,
                     num_purchases, campaign_accepted, recency, num_web_visits]],
                   dtype=float)
    return scaler.transform(raw)


def validate_input(
    age: float,
    income: float,
    total_spend: float,
    num_children: int,
    num_purchases: int,
    campaign_accepted: int,
    recency: int,
    num_web_visits: int,
) -> list[str]:
    """Check sidebar inputs against ranges in config.

    Returns:
        List of error strings. Empty → all valid.
    """
    errors: list[str] = []
    checks = [
        (age,              config.AGE_MIN,       config.AGE_MAX,       "Age",                 "years"),
        (income,           config.INCOME_MIN,    config.INCOME_MAX,    "Income",              ""),
        (total_spend,      config.SPEND_MIN,     config.SPEND_MAX,     "Total Spend",         ""),
        (num_children,     config.CHILDREN_MIN,  config.CHILDREN_MAX,  "Number of children",  ""),
        (num_purchases,    config.PURCHASES_MIN, config.PURCHASES_MAX, "Number of purchases", ""),
        (campaign_accepted,config.CAMPAIGNS_MIN, config.CAMPAIGNS_MAX, "Campaigns accepted",  ""),
        (recency,          config.RECENCY_MIN,   config.RECENCY_MAX,   "Recency",             "days"),
        (num_web_visits,   config.WEBVISITS_MIN, config.WEBVISITS_MAX, "Web visits/month",    ""),
    ]
    for val, lo, hi, label, unit in checks:
        if not (lo <= val <= hi):
            u = f" {unit}" if unit else ""
            errors.append(f"{label} must be between {lo} and {hi}{u}. Got {val}.")
    return errors


# ── Cluster profile table ──────────────────────────────────────────────────────

def build_cluster_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean feature values per cluster, indexed by human-readable label.

    Args:
        df: DataFrame with config.FEATURE_COLS + 'Cluster' column.

    Returns:
        DataFrame (n_clusters × n_features), index = 'Cluster N — Label'.
    """
    means = df.groupby("Cluster")[config.FEATURE_COLS].mean().round(1)
    means.index = [
        f"Cluster {i} — {config.CLUSTER_PROFILES[i]['label']}"
        for i in means.index
    ]
    return means


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_elbow(inertias: dict[int, float]):
    """Plotly line chart: inertia vs k with selected-k annotation.

    Args:
        inertias: {k: inertia} from training analysis.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    ks   = list(inertias.keys())
    vals = list(inertias.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=vals,
        mode="lines+markers",
        line=dict(color="#636EFA", width=2.5),
        marker=dict(size=9, color="#636EFA", line=dict(width=1.5, color="white")),
        hovertemplate="k=%{x}<br>Inertia=%{y:,.1f}<extra></extra>",
    ))
    fig.add_vline(
        x=config.K_DEFAULT, line_dash="dash", line_color="#EF553B", line_width=2,
        annotation_text=f"  k={config.K_DEFAULT} selected",
        annotation_position="top right",
        annotation_font_color="#EF553B",
    )
    fig.update_layout(
        title="<b>Elbow Method</b> — Inertia vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (WCSS)",
        template=config.PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=55, b=45, l=65, r=30),
        xaxis=dict(tickmode="linear"),
    )
    return fig


def plot_silhouette(scores: dict[int, float]):
    """Plotly bar chart: silhouette score per k, selected k highlighted.

    Args:
        scores: {k: silhouette_score} from training analysis.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    ks     = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#EF553B" if k == config.K_DEFAULT else "#636EFA" for k in ks]

    fig = go.Figure(go.Bar(
        x=ks, y=vals,
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        hovertemplate="k=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Silhouette Score</b> vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Mean Silhouette Score",
        template=config.PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=55, b=45, l=65, r=30),
        xaxis=dict(tickmode="linear"),
    )
    return fig


def plot_scatter(
    df: pd.DataFrame,
    new_point: list | None = None,
):
    """Plotly scatter: Income vs TotalSpend, colour-coded by cluster.

    Args:
        df:        DataFrame with FEATURE_COLS + 'Cluster' column.
        new_point: Optional [age, income, total_spend, num_children,
                   num_purchases, campaign_accepted, recency, num_web_visits]
                   for the new-customer star marker.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for cid, profile in config.CLUSTER_PROFILES.items():
        mask   = df["Cluster"] == cid
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset["Income"],
            y=subset["TotalSpend"],
            mode="markers",
            name=f"Cluster {cid} — {profile['label']}",
            marker=dict(
                color=profile["color"], size=config.MARKER_SIZE,
                opacity=0.72, line=dict(width=0.4, color="white"),
            ),
            customdata=subset[
                ["Age","NumChildren","CampaignAccepted","Recency","NumPurchases"]
            ].values,
            hovertemplate=(
                f"<b>{profile['label']}</b><br>"
                "Income: %{x:,.0f}<br>"
                "Total Spend: %{y:,.0f}<br>"
                "Age: %{customdata[0]:.0f}<br>"
                "Children: %{customdata[1]:.0f}<br>"
                "Campaigns accepted: %{customdata[2]:.0f}<br>"
                "Recency: %{customdata[3]:.0f} days<br>"
                "Purchases: %{customdata[4]:.0f}<extra></extra>"
            ),
        ))

    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[1]], y=[new_point[2]],
            mode="markers", name="⭐ New Customer",
            marker=dict(
                symbol=config.NEW_CUSTOMER_MARKER_SYMBOL,
                color=config.NEW_CUSTOMER_MARKER_COLOR,
                size=config.NEW_CUSTOMER_MARKER_SIZE,
                line=dict(width=1.5, color="#333"),
            ),
            hovertemplate=(
                "<b>⭐ New Customer</b><br>"
                "Income: %{x:,.0f}<br>"
                "Total Spend: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="<b>Customer Segments</b> — Income vs Total Spend",
        xaxis_title="Annual Income",
        yaxis_title="Total Spend",
        template=config.PLOTLY_TEMPLATE,
        height=490,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(t=55, b=130, l=65, r=30),
        hovermode="closest",
    )
    return fig


def plot_cluster_sizes(df: pd.DataFrame):
    """Plotly bar chart: customer count and percentage per cluster.

    Args:
        df: DataFrame with a 'Cluster' column.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    counts = df["Cluster"].value_counts().sort_index()
    pcts   = (counts / counts.sum() * 100).round(1)
    labels = [
        f"Cluster {i}<br><b>{config.CLUSTER_PROFILES[i]['label']}</b>"
        for i in counts.index
    ]
    colors = [config.CLUSTER_PROFILES[i]["color"] for i in counts.index]

    fig = go.Figure(go.Bar(
        x=labels, y=counts.values,
        marker_color=colors,
        text=[f"{c:,}<br>({p}%)" for c, p in zip(counts.values, pcts)],
        textposition="outside",
        hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Customer Distribution</b> Across Segments",
        xaxis_title="Segment",
        yaxis_title="Number of Customers",
        template=config.PLOTLY_TEMPLATE,
        height=420,
        margin=dict(t=55, b=80, l=65, r=30),
        showlegend=False,
        yaxis=dict(range=[0, counts.max() * 1.18]),
    )
    return fig


def plot_radar(cluster_means: pd.DataFrame):
    """Plotly radar/spider chart: normalised feature profiles per cluster.

    Args:
        cluster_means: DataFrame (n_clusters × n_features), indexed by cluster ID.
                       Values are normalised internally to [0, 1].

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    normed = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-9
    )
    features = list(normed.columns)
    # Friendly short axis labels
    axis_labels = {
        "Age": "Age", "Income": "Income", "TotalSpend": "Spend",
        "NumChildren": "Children", "NumPurchases": "Purchases",
        "CampaignAccepted": "Campaigns", "Recency": "Recency",
        "NumWebVisitsMonth": "Web Visits",
    }
    theta = [axis_labels.get(f, f) for f in features] + [axis_labels.get(features[0], features[0])]

    fig = go.Figure()
    for cid in normed.index:
        profile = config.CLUSTER_PROFILES.get(cid, {})
        vals    = normed.loc[cid].tolist() + [normed.loc[cid, features[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, fill="toself",
            name=f"Cluster {cid} — {profile.get('label','')}",
            line_color=profile.get("color", "#888"),
            opacity=0.55,
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False, gridcolor="#ddd",
            ),
            angularaxis=dict(gridcolor="#ddd"),
        ),
        title="<b>Cluster Feature Profiles</b> (Radar — normalised 0→1)",
        template=config.PLOTLY_TEMPLATE,
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
        margin=dict(t=65, b=110, l=50, r=50),
    )
    return fig