"""
app.py — Customer Segmentation Explorer
========================================
Main Streamlit application. Loads pre-trained artefacts from model/ and
provides an interactive UI for data exploration, cluster visualisation,
and new-customer segment prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

import config
from utils.helpers import load_data, preprocess, get_descriptive_stats

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="🛍️",
    layout="wide",
)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the dataset. Cached after first call.

    Returns:
        Tuple of (raw_df, clean_df).
    """
    raw   = load_data(config.DATA_PATH)
    clean = preprocess(raw)
    return raw, clean


# ── Main layout ────────────────────────────────────────────────────────────────

st.title("🛍️ Customer Segmentation Explorer")
st.caption("Customer Personality Analysis · K-means clustering")

try:
    df_raw, df_clean = get_data()

    # ── Key metrics ────────────────────────────────────────────────────────────
    st.header("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers",   len(df_raw))
    c2.metric("Engineered features", len(config.FEATURE_COLS))
    c3.metric("Raw columns",       df_raw.shape[1])
    c4.metric(
        "Income nulls imputed",
        int(df_raw["Income"].isnull().sum()),
        help="Missing Income values are filled with the column median.",
    )

    # ── Raw preview ────────────────────────────────────────────────────────────
    with st.expander("📄 Raw data preview", expanded=False):
        st.dataframe(df_raw, use_container_width=True, height=300)

    # ── Engineered features ────────────────────────────────────────────────────
    st.subheader("Engineered features (model input)")
    st.caption(
        "29 raw columns → 8 engineered features: "
        "Age · Income · TotalSpend · NumChildren · NumPurchases · "
        "CampaignAccepted · Recency · NumWebVisitsMonth"
    )
    st.dataframe(df_clean, use_container_width=True, height=300)

    # ── Descriptive stats ──────────────────────────────────────────────────────
    st.subheader("Descriptive statistics")
    st.dataframe(
        get_descriptive_stats(df_clean),
        use_container_width=True,
    )

    st.divider()
    st.success(
        "✅ Phase 2 complete — data loading, feature engineering "
        "and preprocessing are live. Model training coming in Phase 3."
    )

except FileNotFoundError as e:
    st.error(f"📂 {e}")
except ValueError as e:
    st.error(f"⚠️ {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")