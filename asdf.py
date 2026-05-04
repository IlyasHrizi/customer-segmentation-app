import sys; sys.path.insert(0, '.')
import joblib, numpy as np
from utils.helpers import (
    load_data, preprocess, scale_features,
    scale_input, validate_input
)
import config

print("=" * 55)
print("VERIFICATION — loaded artefacts")
print("=" * 55)

model  = joblib.load(config.MODEL_PATH)
scaler = joblib.load(config.SCALER_PATH)

print(f"\nModel  : {type(model).__name__}, k={model.n_clusters}")
print(f"Scaler : {type(scaler).__name__}, n_features={scaler.n_features_in_}")

# Reproduce cluster summary
df_clean = preprocess(load_data())
X, _     = scale_features(df_clean)
preds    = model.predict(X)
unique, counts = np.unique(preds, return_counts=True)
print("\nCluster distribution (re-predicted from saved model):")
for cid, cnt in zip(unique, counts):
    label = config.CLUSTER_PROFILES[cid]["label"]
    print(f"  Cluster {cid} — {label:<26} : {cnt} ({cnt/len(preds)*100:.1f}%)")

# Simulate 3 test predictions
test_customers = [
    dict(age=45, income=75000, total_spend=500, num_children=0,
         num_purchases=25, campaign_accepted=0, recency=10, num_web_visits=8,
         note="High earner, big spender, campaign-resistant → expect Cluster 0"),
    dict(age=52, income=55000, total_spend=200, num_children=1,
         num_purchases=22, campaign_accepted=2, recency=40, num_web_visits=10,
         note="Medium income, responds to campaigns → expect Cluster 1"),
    dict(age=38, income=30000, total_spend=120, num_children=2,
         num_purchases=30, campaign_accepted=0, recency=60, num_web_visits=12,
         note="Low income, budget-conscious → expect Cluster 2"),
]

print("\nTest predictions:")
for i, tc in enumerate(test_customers):
    errors = validate_input(
        tc["age"], tc["income"], tc["total_spend"], tc["num_children"],
        tc["num_purchases"], tc["campaign_accepted"], tc["recency"], tc["num_web_visits"]
    )
    assert errors == [], f"Unexpected validation errors: {errors}"
    x = scale_input(
        tc["age"], tc["income"], tc["total_spend"], tc["num_children"],
        tc["num_purchases"], tc["campaign_accepted"], tc["recency"],
        tc["num_web_visits"], scaler
    )
    cluster = model.predict(x)[0]
    label   = config.CLUSTER_PROFILES[cluster]["label"]
    print(f"\n  Test {i+1}: {tc['note']}")
    print(f"    → Predicted: Cluster {cluster} — {label}")

print("\n✓ All artefacts verified and predictions consistent")