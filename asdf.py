import sys; sys.path.insert(0, '.')
import joblib, numpy as np, pandas as pd
from utils.helpers import (
    load_data, preprocess, scale_features,
    scale_input, validate_input, build_cluster_profile_table
)
import config

model  = joblib.load(config.MODEL_PATH)
scaler = joblib.load(config.SCALER_PATH)

df_raw   = load_data()
df_clean = preprocess(df_raw)
X_all, _ = scale_features(df_clean)
df_clean["Cluster"] = model.predict(X_all)
counts   = df_clean["Cluster"].value_counts().sort_index()

def predict(age, income, spend, kids, purch, camps, rec, web):
    x   = scale_input(age, income, spend, kids, purch, camps, rec, web, scaler)
    cid = int(model.predict(x)[0])
    return cid, config.CLUSTER_PROFILES[cid]['label']

# ─────────────────────────────────────────────────────────────
# BUSINESS TEST CASES (for the report)
# ─────────────────────────────────────────────────────────────
test_cases = [
    {
        "id": 1,
        "desc": "Affluent professional — high income, big spender, ignores campaigns",
        "inputs": dict(age=48, income=85000, spend=700, kids=0,
                       purch=28, camps=0, rec=8, web=5),
        "expected_cluster": 0,
        "rationale": (
            "Income of €85k and €700 total spend both sit well above cluster-0 means "
            "(€72k / €350). Zero campaigns accepted and low recency (8 days) signal "
            "a loyal, self-directed buyer. → Cluster 0 'Affluent & Independent'."
        ),
    },
    {
        "id": 2,
        "desc": "Mid-income family — responds to campaigns, moderate spend",
        "inputs": dict(age=44, income=55000, spend=210, kids=1,
                       purch=18, camps=2, rec=35, web=9),
        "expected_cluster": 1,
        "rationale": (
            "Income (€55k) and spend (€210) close to cluster-1 means (€54k / €193). "
            "Two campaigns accepted matches the cluster-1 average of 1.1 — the highest "
            "of all segments. → Cluster 1 'Campaign Receptive'."
        ),
    },
    {
        "id": 3,
        "desc": "Budget household — low income, minimal spend, deal-seeker",
        "inputs": dict(age=36, income=28000, spend=95, kids=2,
                       purch=32, camps=0, rec=55, web=14),
        "expected_cluster": 2,
        "rationale": (
            "Income (€28k) and spend (€95) are both below cluster-2 means (€45k / €154). "
            "Zero campaigns accepted and high web visits (14) indicate deal browsing "
            "without converting. → Cluster 2 'Budget Conscious'."
        ),
    },
    {
        "id": 4,
        "desc": "High-income retiree — premium spend, campaign-resistant, very recent",
        "inputs": dict(age=67, income=92000, spend=950, kids=0,
                       purch=22, camps=0, rec=3, web=2),
        "expected_cluster": 0,
        "rationale": (
            "Extremely high spend (€950, nearly 3× cluster-0 mean) and top-tier income "
            "(€92k). Very low recency (3 days) and zero campaigns confirm self-directed "
            "premium buying. → Cluster 0 'Affluent & Independent'."
        ),
    },
    {
        "id": 5,
        "desc": "Young single — entry-level income, opened 3 campaigns, active online",
        "inputs": dict(age=27, income=32000, spend=130, kids=0,
                       purch=14, camps=3, rec=20, web=15),
        "expected_cluster": 1,
        "rationale": (
            "Despite low income (€32k), the customer accepted 3 campaigns — far above "
            "any cluster mean. This campaign responsiveness is the dominant signal that "
            "pulls the prediction toward cluster 1 over cluster 2. → Cluster 1 "
            "'Campaign Receptive'."
        ),
    },
    {
        "id": 6,
        "desc": "Low-income senior — fixed budget, passive, no campaigns",
        "inputs": dict(age=71, income=22000, spend=60, kids=1,
                       purch=10, camps=0, rec=75, web=7),
        "expected_cluster": 2,
        "rationale": (
            "All signals align with the budget segment: income €22k, spend €60, "
            "no campaigns accepted, high recency (75 days). Age is not the driver — "
            "income and spend are. → Cluster 2 'Budget Conscious'."
        ),
    },
    {
        "id": 7,
        "desc": "Boundary case — cluster-0 / cluster-1 border (income midpoint)",
        "inputs": dict(age=50, income=63000, spend=270, kids=1,
                       purch=20, camps=1, rec=45, web=9),
        "expected_cluster": None,   # no hard expectation — observational
        "rationale": (
            "Income (€63k) sits midway between cluster-0 (€72k) and cluster-1 (€54k). "
            "Spend (€270) is also between the two means. Result documents where the "
            "decision boundary falls in practice."
        ),
    },
]

# ─────────────────────────────────────────────────────────────
# EDGE / ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────
edge_cases = [
    {
        "id": "E1",
        "desc": "Minimum valid values — floor of every range",
        "inputs": dict(age=18, income=1200, spend=0, kids=0,
                       purch=0, camps=0, rec=0, web=0),
    },
    {
        "id": "E2",
        "desc": "Maximum valid values — ceiling of every range",
        "inputs": dict(age=84, income=666666, spend=3500, kids=4,
                       purch=40, camps=5, rec=99, web=20),
    },
    {
        "id": "E3",
        "desc": "Invalid age (below min)",
        "inputs": dict(age=5, income=40000, spend=200, kids=1,
                       purch=15, camps=0, rec=30, web=5),
        "expect_errors": ["Age"],
    },
    {
        "id": "E4",
        "desc": "Invalid income (zero) and invalid campaigns (6)",
        "inputs": dict(age=35, income=0, spend=200, kids=1,
                       purch=15, camps=6, rec=30, web=5),
        "expect_errors": ["Income", "Campaigns"],
    },
    {
        "id": "E5",
        "desc": "All 8 fields out of range",
        "inputs": dict(age=10, income=0, spend=-10, kids=9,
                       purch=99, camps=9, rec=-5, web=50),
        "expect_errors": 8,
    },
]

# ─────────────────────────────────────────────────────────────
# RUN & PRINT RESULTS
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 5 — TEST RESULTS")
print("=" * 70)

all_passed = True

print("\n── BUSINESS TEST CASES ────────────────────────────────────────────────\n")
for tc in test_cases:
    inp = tc["inputs"]
    cid, label = predict(**inp)
    passed = (tc["expected_cluster"] is None) or (cid == tc["expected_cluster"])
    if not passed:
        all_passed = False
    status = "✓" if passed else "✗"

    seg_size = counts.get(cid, 0)
    seg_pct  = seg_size / len(df_clean) * 100

    print(f"Test {tc['id']} {status}  {tc['desc']}")
    print(f"  Inputs  : age={inp['age']}  income=€{inp['income']:,}  spend=€{inp['spend']}"
          f"  kids={inp['kids']}  purchases={inp['purch']}"
          f"  campaigns={inp['camps']}  recency={inp['rec']}d  web={inp['web']}/mo")
    if tc["expected_cluster"] is not None:
        print(f"  Expected: Cluster {tc['expected_cluster']}  |  "
              f"Observed: Cluster {cid} — {label}")
    else:
        print(f"  Observed: Cluster {cid} — {label}  (boundary — observational)")
    print(f"  Segment : {seg_size:,} customers ({seg_pct:.1f}% of base)")
    print(f"  Rationale: {tc['rationale']}")
    print()

print("\n── EDGE & ROBUSTNESS TESTS ────────────────────────────────────────────\n")
for ec in edge_cases:
    inp = ec["inputs"]
    errs = validate_input(
        inp["age"], inp["income"], inp["spend"], inp["kids"],
        inp["purch"], inp["camps"], inp["rec"], inp["web"]
    )

    if "expect_errors" in ec:
        expected = ec["expect_errors"]
        if isinstance(expected, int):
            passed = len(errs) == expected
            obs    = f"{len(errs)} errors"
            exp    = f"{expected} errors"
        else:
            passed = all(any(e_kw in err for err in errs) for e_kw in expected)
            obs    = str([e.split(" must")[0] for e in errs])
            exp    = str(expected)
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"Edge {ec['id']} {status}  {ec['desc']}")
        print(f"  Expected: {exp}  |  Observed: {obs}")
        if errs:
            for e in errs:
                print(f"    · {e}")
    else:
        # Valid edge — should predict without error
        assert errs == [], f"Unexpected validation errors: {errs}"
        cid, label = predict(**inp)
        print(f"Edge {ec['id']} ✓  {ec['desc']}")
        print(f"  → Cluster {cid} — {label}")
    print()

print("=" * 70)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED — review output above")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# CLUSTER MEANS SUMMARY (for report reference)
# ─────────────────────────────────────────────────────────────
print("\n── CLUSTER MEANS (report reference) ───────────────────────────────────")
means = build_cluster_profile_table(df_clean)
print(means.to_string())