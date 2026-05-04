"""
utils/helpers.py — Reusable helper functions
=============================================
All data loading, preprocessing, validation, and plotting utilities.
Import these in app.py and model/train.py — never duplicate logic.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
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
    df = pd.read_csv(resolved)
    return df


# ── Preprocessing & feature engineering ───────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw DataFrame and engineer model-ready features.

    Steps performed (in order):
        1. Remove outliers: Income > 600 000 and Age > 100 are dropped.
        2. Impute missing Income with the median.
        3. Engineer Age from Year_Birth (reference year 2024).
        4. Engineer TotalSpend = sum of all MntXxx columns.
        5. Engineer NumChildren = Kidhome + Teenhome.
        6. Engineer NumPurchases = sum of all NumXxxPurchases columns.
        7. Engineer CampaignAccepted = sum of AcceptedCmp1..5.
        8. Return only config.FEATURE_COLS.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame containing exactly config.FEATURE_COLS, no nulls.

    Raises:
        ValueError: If an expected source column is missing from df.
    """
    _check_required_columns(df)
    out = df.copy()

    # 1. Remove extreme outliers
    out = out[out["Income"] < 600_000]
    out = out[2024 - out["Year_Birth"] <= 100]

    # 2. Impute missing Income with median (robust to outliers)
    income_median = out["Income"].median()
    n_imputed = out["Income"].isnull().sum()
    out["Income"] = out["Income"].fillna(income_median)

    # 3. Age
    out["Age"] = 2024 - out["Year_Birth"]

    # 4. TotalSpend
    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    ]
    out["TotalSpend"] = out[spend_cols].sum(axis=1)

    # 5. NumChildren
    out["NumChildren"] = out["Kidhome"] + out["Teenhome"]

    # 6. NumPurchases
    purchase_cols = [
        "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases",
    ]
    out["NumPurchases"] = out[purchase_cols].sum(axis=1)

    # 7. CampaignAccepted
    campaign_cols = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5",
    ]
    out["CampaignAccepted"] = out[campaign_cols].sum(axis=1)

    # 8. Keep only model features
    result = out[config.FEATURE_COLS].copy()

    # Sanity check — no nulls should remain
    remaining_nulls = result.isnull().sum().sum()
    if remaining_nulls > 0:
        raise RuntimeError(
            f"Preprocessing left {remaining_nulls} null values. "
            "Check input data for unexpected missing columns."
        )

    return result


def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any source column required for feature engineering is absent.

    Args:
        df: Raw DataFrame to inspect.

    Raises:
        ValueError: Lists every missing column.
    """
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
            f"Dataset is missing {len(missing)} required column(s): "
            f"{sorted(missing)}"
        )


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rounded descriptive statistics for the engineered features.

    Args:
        df: Preprocessed DataFrame with config.FEATURE_COLS.

    Returns:
        DataFrame with rows: count, mean, std, min, 25%, 50%, 75%, max.
    """
    return df[config.FEATURE_COLS].describe().round(2)


# ── Scaling ────────────────────────────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler on df and return scaled array + fitted scaler.

    Always fit on the full training set, then persist with joblib.dump()
    so inference uses the identical transformation.

    Args:
        df: Preprocessed DataFrame with config.FEATURE_COLS columns.

    Returns:
        Tuple of (X_scaled, fitted_scaler). X_scaled: (n_samples, n_features).
    """
    scaler = StandardScaler()
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
    """Scale a single new customer's features using a pre-fitted scaler.

    Args:
        age:               Customer age in years.
        income:            Annual income (€ or local currency).
        total_spend:       Total amount spent across all product categories.
        num_children:      Total number of children (kids + teens) at home.
        num_purchases:     Total purchases across all channels.
        campaign_accepted: Number of campaigns accepted (0–5).
        recency:           Days since last purchase.
        num_web_visits:    Website visits in the last month.
        scaler:            Pre-fitted StandardScaler loaded from scaler.pkl.

    Returns:
        Numpy array of shape (1, 8), ready for model.predict().
    """
    raw = np.array([[
        age, income, total_spend, num_children,
        num_purchases, campaign_accepted, recency, num_web_visits,
    ]], dtype=float)
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
    """Validate that sidebar input values fall within acceptable ranges.

    Args:
        See scale_input() for parameter descriptions.

    Returns:
        List of error messages. Empty list means all inputs are valid.
    """
    errors: list[str] = []

    checks = [
        (age,              config.AGE_MIN,       config.AGE_MAX,       "Age",                    "years"),
        (income,           config.INCOME_MIN,    config.INCOME_MAX,    "Income",                 ""),
        (total_spend,      config.SPEND_MIN,     config.SPEND_MAX,     "Total Spend",            ""),
        (num_children,     config.CHILDREN_MIN,  config.CHILDREN_MAX,  "Number of children",     ""),
        (num_purchases,    config.PURCHASES_MIN, config.PURCHASES_MAX, "Number of purchases",    ""),
        (campaign_accepted,config.CAMPAIGNS_MIN, config.CAMPAIGNS_MAX, "Campaigns accepted",     ""),
        (recency,          config.RECENCY_MIN,   config.RECENCY_MAX,   "Recency",                "days"),
        (num_web_visits,   config.WEBVISITS_MIN, config.WEBVISITS_MAX, "Web visits/month",       ""),
    ]

    for val, lo, hi, label, unit in checks:
        if not (lo <= val <= hi):
            unit_str = f" {unit}" if unit else ""
            errors.append(
                f"{label} must be between {lo} and {hi}{unit_str}. Got {val}."
            )

    return errors


# ── Plotting helpers (implemented in Phase 4) ──────────────────────────────────

def plot_elbow(inertias: dict[int, float]):
    """Return a Plotly figure for the elbow method.

    Args:
        inertias: Mapping of k → within-cluster sum of squares.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError  # TODO Phase 4


def plot_silhouette(scores: dict[int, float]):
    """Return a Plotly figure for silhouette scores across k values.

    Args:
        scores: Mapping of k → mean silhouette score.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError  # TODO Phase 4


def plot_scatter(df: pd.DataFrame, new_point: list | None = None):
    """Return a Plotly scatter of Income vs TotalSpend, coloured by cluster.

    Args:
        df:        DataFrame with config.FEATURE_COLS + 'Cluster' column.
        new_point: Optional raw list [age, income, total_spend, ...] to
                   highlight as a star marker.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError  # TODO Phase 4


def plot_cluster_sizes(df: pd.DataFrame):
    """Return a Plotly bar chart of customer count per cluster.

    Args:
        df: DataFrame with a 'Cluster' column.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError  # TODO Phase 4


def plot_radar(cluster_means: pd.DataFrame):
    """Return a Plotly radar chart comparing cluster feature profiles.

    Args:
        cluster_means: DataFrame (n_clusters × n_features), indexed by cluster ID.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError  # TODO Phase 4 (bonus)