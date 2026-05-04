# =============================================================================
# config.py — Project-wide constants
# All hardcoded values live here. Import this in every other module.
# =============================================================================
import pathlib

# This finds the absolute path to the folder containing config.py
BASE_DIR = pathlib.Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────────────────────
# We use BASE_DIR to ensure paths are always absolute, regardless of where 
# the script is executed from.
DATA_PATH   = str(BASE_DIR / "data" / "customers.csv")
MODEL_PATH  = str(BASE_DIR / "model" / "kmeans_model.pkl")
SCALER_PATH = str(BASE_DIR / "model" / "scaler.pkl")

# ── Raw dataset columns ───────────────────────────────────────────────────────
ID_COL            = "ID"
DATE_COL          = "Dt_Customer"
BIRTH_COL         = "Year_Birth"
INCOME_COL        = "Income"
EDUCATION_COL     = "Education"
MARITAL_COL       = "Marital_Status"
 
# Columns to drop before modelling (constants, IDs, or leakage)
DROP_COLS = ["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
 
# Categorical columns that need encoding
CATEGORICAL_COLS = ["Education", "Marital_Status"]
 
# ── Engineered features fed to the model ──────────────────────────────────────
# These are derived inside preprocess() and then scaled.
FEATURE_COLS = [
    "Age",                  # 2024 - Year_Birth
    "Income",               # cleaned, nulls imputed
    "TotalSpend",           # sum of all Mnt* columns
    "NumChildren",          # Kidhome + Teenhome
    "NumPurchases",         # sum of all Num*Purchases columns
    "CampaignAccepted",     # sum of AcceptedCmp1..5
    "Recency",              # days since last purchase
    "NumWebVisitsMonth",    # web engagement proxy
]
 
# ── Input validation ranges (used in the sidebar prediction form) ─────────────
AGE_MIN,      AGE_MAX      = 18, 84
INCOME_MIN,   INCOME_MAX   = 1200, 666666
CHILDREN_MIN, CHILDREN_MAX = 0, 4
RECENCY_MIN,  RECENCY_MAX  = 0, 99
SPEND_MIN,    SPEND_MAX    = 0, 3500
PURCHASES_MIN,PURCHASES_MAX= 0, 40
CAMPAIGNS_MIN,CAMPAIGNS_MAX= 0, 5
WEBVISITS_MIN,WEBVISITS_MAX= 0, 20
 
# ── Model ─────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
K_RANGE       = range(2, 11)
K_DEFAULT     = 4
K_SLIDER_MIN  = 3
K_SLIDER_MAX  = 6
 
# ── Business segment labels & strategies ──────────────────────────────────────
# Keys are cluster IDs (0-indexed). Relabel after inspecting cluster means
# in Phase 3; these are placeholder names.
CLUSTER_PROFILES = {
    # High income (72k), highest spend (350), campaign-resistant, wine & meat focus
    0: {
        "label": "Affluent & Independent",
        "color": "#EF553B",
        "strategy": (
            "These high earners spend heavily but rarely respond to campaigns. "
            "Focus on premium product quality, exclusive catalogue offers, and "
            "in-store VIP experiences rather than discount-driven outreach."
        ),
    },
    # Medium income (54k), moderate spend (193), highest campaign acceptance (1.1)
    1: {
        "label": "Campaign Receptive",
        "color": "#636EFA",
        "strategy": (
            "Most responsive segment to marketing campaigns. Invest in targeted "
            "multi-channel campaigns (email, web, catalogue). Personalised "
            "product recommendations and loyalty rewards will grow basket size."
        ),
    },
    # Lower income (45k), lowest spend (154), campaign-resistant, largest group
    2: {
        "label": "Budget Conscious",
        "color": "#00CC96",
        "strategy": (
            "Largest segment with lowest spending power. Prioritise deal-based "
            "promotions, bundle discounts, and web-channel offers. "
            "Focus on value messaging to gradually increase purchase frequency."
        ),
    },
}
 
# ── Visualisation ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE          = "plotly_white"
MARKER_SIZE              = 7
NEW_CUSTOMER_MARKER_SIZE   = 18
NEW_CUSTOMER_MARKER_SYMBOL = "star"
NEW_CUSTOMER_MARKER_COLOR  = "#FFD700"