"""
app.py — Customer Segmentation Explorer
========================================
Interactive Streamlit application for Customer Personality Analysis.
Loads pre-trained KMeans artefacts and provides segment exploration,
visualisation, and new-customer prediction in a single-page UI.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config
from utils.helpers import (
    load_data, preprocess, get_descriptive_stats,
    scale_features, scale_input, validate_input,
    build_cluster_profile_table,
    plot_elbow, plot_silhouette, plot_scatter,
    plot_cluster_sizes, plot_radar,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .segment-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .segment-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .segment-strategy { font-size: 0.92rem; color: #444; line-height: 1.5; }
    .metric-label { font-size: 0.8rem; color: #888; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess dataset. Cached after first call.

    Returns:
        Tuple of (raw_df, clean_df).
    """
    raw   = load_data(config.DATA_PATH)
    clean = preprocess(raw)
    return raw, clean


@st.cache_resource
def get_model_and_scaler():
    """Load serialised KMeans model and StandardScaler. Cached for session.

    Returns:
        Tuple of (KMeans model, StandardScaler).

    Raises:
        FileNotFoundError: If .pkl files don't exist yet.
    """
    try:
        model  = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        raise FileNotFoundError(
            "Model artefacts not found. Run  `python model/train.py`  first."
        )


@st.cache_data
def compute_diagnostics(n_samples: int) -> tuple[dict, dict]:
    """Compute elbow inertias and silhouette scores across K_RANGE.

    Cached so the loop only runs once per session regardless of
    how many times the user interacts with the page.

    Args:
        n_samples: Passed in to invalidate cache when dataset changes.

    Returns:
        Tuple of (inertias, sil_scores) — dicts mapping k → metric.
    """
    df_raw, df_clean = get_data()
    X, _             = scale_features(df_clean)
    inertias:   dict[int, float] = {}
    sil_scores: dict[int, float] = {}
    for k in config.K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias[k]   = km.inertia_
        sil_scores[k] = silhouette_score(
            X, labels, sample_size=1000, random_state=config.RANDOM_STATE
        )
    return inertias, sil_scores


# ── App entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Render the full Streamlit application."""

    # ── Load data & model ──────────────────────────────────────────────────────
    try:
        df_raw, df_clean = get_data()
        model, scaler    = get_model_and_scaler()
    except FileNotFoundError as e:
        st.error(f"📂 {e}")
        st.stop()
    except ValueError as e:
        st.error(f"⚠️ Dataset error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data/model: {e}")
        st.stop()

    # Assign clusters to the full dataset
    X_all, _  = scale_features(df_clean)
    df_clean  = df_clean.copy()
    df_clean["Cluster"] = model.predict(X_all)

    # ── Header ─────────────────────────────────────────────────────────────────
    st.title("🛍️ Customer Segmentation Explorer")
    st.caption(
        "Customer Personality Analysis · K-means clustering · "
        f"k={model.n_clusters} segments"
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — DATASET OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    st.header("1 · Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",   f"{len(df_raw):,}")
    c2.metric("After Cleaning",    f"{len(df_clean):,}")
    c3.metric("Raw Features",      df_raw.shape[1])
    c4.metric("Engineered Features", len(config.FEATURE_COLS))
    c5.metric(
        "Income Nulls Imputed",
        int(df_raw["Income"].isnull().sum()),
        help="Missing Income values replaced with column median.",
    )

    with st.expander("📄 Raw data preview", expanded=False):
        st.dataframe(df_raw, use_container_width=True, height=280)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Engineered feature data")
        st.caption("29 raw columns → 8 model features")
        st.dataframe(df_clean.drop(columns="Cluster"),
                     use_container_width=True, height=280)
    with col_right:
        st.subheader("Descriptive statistics")
        st.dataframe(get_descriptive_stats(df_clean),
                     use_container_width=True, height=280)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MODEL DIAGNOSTICS (ELBOW + SILHOUETTE)
    # ══════════════════════════════════════════════════════════════════════════
    st.header("2 · Model Diagnostics")
    st.caption(
        "The elbow method identifies where adding more clusters yields "
        "diminishing returns. Silhouette score measures how well each customer "
        "fits its own cluster versus its nearest neighbour."
    )

    with st.spinner("Computing elbow and silhouette curves…"):
        inertias, sil_scores = compute_diagnostics(len(df_clean))

    diag_left, diag_right = st.columns(2)
    with diag_left:
        st.plotly_chart(plot_elbow(inertias), use_container_width=True)
    with diag_right:
        st.plotly_chart(plot_silhouette(sil_scores), use_container_width=True)

    with st.expander("ℹ️ How to read these charts", expanded=False):
        st.markdown("""
**Elbow chart** — The red dashed line marks the selected k (k=3). The curve
bends here, meaning each additional cluster after this point brings less
reduction in inertia (within-cluster variance).

**Silhouette chart** — Scores range from -1 to 1. Higher is better: customers
with a high silhouette are well-matched to their cluster and clearly separated
from others. Scores around 0.10–0.15 are typical for overlapping real-world
customer data.
        """)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — SEGMENT VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.header("3 · Segment Visualisations")

    # 3a. Scatter chart (full width)
    st.subheader("Income vs Total Spend — coloured by segment")
    st.caption("Hover over any point to see detailed customer attributes.")

    # Check if a new customer has been predicted (set by sidebar below)
    new_pt = st.session_state.get("new_customer_point", None)
    st.plotly_chart(
        plot_scatter(df_clean, new_point=new_pt),
        use_container_width=True,
    )

    # 3b. Size bar + Radar side by side
    vis_left, vis_right = st.columns(2)
    with vis_left:
        st.subheader("Segment sizes")
        st.plotly_chart(plot_cluster_sizes(df_clean), use_container_width=True)
    with vis_right:
        st.subheader("Feature profile radar")
        st.caption("Each axis is normalised to 0–1 for visual comparability.")
        means_for_radar = df_clean.groupby("Cluster")[config.FEATURE_COLS].mean()
        st.plotly_chart(plot_radar(means_for_radar), use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — CLUSTER PROFILE TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.header("4 · Cluster Profile Table")
    st.caption("Mean value of each engineered feature per segment.")

    profile_table = build_cluster_profile_table(df_clean)
    st.dataframe(
        profile_table.style.background_gradient(cmap="Blues", axis=1),
        use_container_width=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — SEGMENT DESCRIPTIONS & STRATEGIES
    # ══════════════════════════════════════════════════════════════════════════
    st.header("5 · Segment Descriptions & Marketing Strategies")

    counts = df_clean["Cluster"].value_counts().sort_index()
    for cid, profile in config.CLUSTER_PROFILES.items():
        size = counts.get(cid, 0)
        pct  = size / len(df_clean) * 100
        st.markdown(
            f"""
<div class="segment-card" style="border-color:{profile['color']}">
  <div class="segment-title" style="color:{profile['color']}">
    Cluster {cid} — {profile['label']}
    &nbsp;&nbsp;<small style="font-weight:400;color:#666">{size:,} customers · {pct:.1f}%</small>
  </div>
  <div class="segment-strategy">🎯 {profile['strategy']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — NEW CUSTOMER PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("🔮 Predict New Customer")
        st.caption(
            "Enter a customer's profile below. The model will assign them "
            "to the most fitting segment and suggest a marketing strategy."
        )

        with st.form("prediction_form"):
            st.subheader("Customer profile")

            age = st.number_input(
                "Age (years)",
                min_value=config.AGE_MIN, max_value=config.AGE_MAX,
                value=40, step=1,
                help="Customer's current age in years.",
            )
            income = st.number_input(
                "Annual Income (€)",
                min_value=config.INCOME_MIN, max_value=config.INCOME_MAX,
                value=52000, step=1000,
                help="Gross annual income.",
            )
            total_spend = st.number_input(
                "Total Spend (€)",
                min_value=config.SPEND_MIN, max_value=config.SPEND_MAX,
                value=300, step=10,
                help="Sum spent across wines, fruits, meat, fish, sweets, gold.",
            )
            num_children = st.slider(
                "Number of Children at Home",
                min_value=config.CHILDREN_MIN, max_value=config.CHILDREN_MAX,
                value=1,
                help="Kidhome + Teenhome.",
            )
            num_purchases = st.number_input(
                "Total Purchases (all channels)",
                min_value=config.PURCHASES_MIN, max_value=config.PURCHASES_MAX,
                value=15, step=1,
                help="Web + catalogue + store + deal purchases combined.",
            )
            campaign_accepted = st.slider(
                "Campaigns Accepted (0–5)",
                min_value=config.CAMPAIGNS_MIN, max_value=config.CAMPAIGNS_MAX,
                value=0,
                help="How many of the 5 marketing campaigns this customer accepted.",
            )
            recency = st.number_input(
                "Recency (days since last purchase)",
                min_value=config.RECENCY_MIN, max_value=config.RECENCY_MAX,
                value=30, step=1,
            )
            num_web_visits = st.number_input(
                "Web Visits per Month",
                min_value=config.WEBVISITS_MIN, max_value=config.WEBVISITS_MAX,
                value=5, step=1,
            )

            submitted = st.form_submit_button(
                "🔍 Predict Segment", use_container_width=True
            )

        # ── Handle prediction ──────────────────────────────────────────────────
        if submitted:
            errors = validate_input(
                age, income, total_spend, num_children,
                num_purchases, campaign_accepted, recency, num_web_visits,
            )
            if errors:
                for err in errors:
                    st.error(err)
            else:
                try:
                    x_scaled = scale_input(
                        age, income, total_spend, num_children,
                        num_purchases, campaign_accepted, recency,
                        num_web_visits, scaler,
                    )
                    cluster_id = int(model.predict(x_scaled)[0])
                    profile    = config.CLUSTER_PROFILES[cluster_id]
                    seg_size   = int(counts.get(cluster_id, 0))
                    seg_pct    = seg_size / len(df_clean) * 100

                    # Store for scatter chart highlight
                    st.session_state["new_customer_point"] = [
                        age, income, total_spend, num_children,
                        num_purchases, campaign_accepted, recency, num_web_visits,
                    ]

                    st.success("✅ Segment identified!")
                    st.markdown(
                        f"""
<div class="segment-card" style="border-color:{profile['color']}">
  <div class="segment-title" style="color:{profile['color']}">
    Cluster {cluster_id} — {profile['label']}
  </div>
  <div class="metric-label">
    Segment size: <b>{seg_size:,}</b> customers ({seg_pct:.1f}% of base)
  </div>
  <br/>
  <div class="segment-strategy">🎯 {profile['strategy']}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    st.info(
                        "↑ The scatter chart above now shows this customer "
                        "as a ⭐ gold star. Scroll up to see their position "
                        "relative to the full dataset."
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Clear button
        if st.button("🗑️ Clear prediction", use_container_width=True):
            st.session_state.pop("new_customer_point", None)
            st.rerun()


if __name__ == "__main__":
    main()