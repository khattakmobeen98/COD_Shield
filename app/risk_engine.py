import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Synthetic training set for the MVP. In production this would be replaced
# with historical merchant order outcomes.
X = np.array([
    [0,0,0,0.30,3,0.85,12000,1],
    [1,0,0,0.40,2,0.70,9000,2],
    [2,1,1,0.55,2,0.60,7500,23],
    [3,2,1,0.65,1,0.50,6000,20],
    [5,5,0,0.90,1,0.20,2500,15],
    [8,8,0,0.95,1,0.10,2000,18],
    [10,9,1,0.85,1,0.25,3500,14],
    [1,1,0,0.75,1,0.30,3000,16],
    [4,3,1,0.60,2,0.55,8500,21],
    [0,0,0,0.35,3,0.80,15000,0],
    [7,6,1,0.80,1,0.35,5000,19],
    [2,0,0,0.45,2,0.75,11000,1],
], dtype=float)

y = np.array([1,1,1,0,0,0,0,0,1,1,0,1])

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = LogisticRegression(random_state=42).fit(Xs, y)

FEATURES = [
    "previous_orders","successful_deliveries","rto_count",
    "address_similarity","phone_reuse","delivery_zone_risk","amount","hour"
]

def score_order(o):
    row = np.array([[
        o["previous_orders"], o["successful_deliveries"], o["rto_count"],
        o["address_similarity"], o["phone_reuse"], o["delivery_zone_risk"],
        o["amount"], o["hour"]
    ]], dtype=float)

    probability = float(model.predict_proba(scaler.transform(row))[0,1])
    # Blend model probability with a transparent business-risk layer so the
    # demo behaves intuitively on the synthetic examples.
    rto_ratio = o["rto_count"] / max(o["previous_orders"], 1)
    business = (
        0.28 * min(rto_ratio, 1) +
        0.18 * (1 - o["address_similarity"]) +
        0.16 * min(o["phone_reuse"] / 3, 1) +
        0.16 * o["delivery_zone_risk"] +
        0.12 * min(o["amount"] / 15000, 1) +
        0.10 * (1 if o["previous_orders"] == 0 else 0)
    )
    score = round(min(max((0.62 * probability + 0.38 * business) * 100, 2), 98))

    if score >= 70:
        level = "HIGH"
        recommendation = "Manual verification before dispatch is recommended."
    elif score >= 40:
        level = "MEDIUM"
        recommendation = "Review customer and delivery details before dispatch."
    else:
        level = "LOW"
        recommendation = "Order can proceed through normal dispatch."

    factors = []
    if o["rto_count"] > 0:
        factors.append(("Previous RTO history", min(28, 12 + o["rto_count"] * 8)))
    if o["previous_orders"] == 0:
        factors.append(("New customer", 17))
    if o["amount"] >= 7000:
        factors.append(("High order value", 15))
    if o["address_similarity"] < 0.65:
        factors.append(("Address pattern mismatch", 11))
    if o["phone_reuse"] > 1:
        factors.append(("Phone number reused", 9))
    if o["delivery_zone_risk"] >= 0.60:
        factors.append(("Higher-risk delivery zone", 9))
    if o["hour"] <= 3 or o["hour"] >= 23:
        factors.append(("Unusual order time", 6))

    factors = sorted(factors, key=lambda x: x[1], reverse=True)[:5]
    return {
        **o,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "factors": [{"name": n, "impact": v} for n, v in factors],
        "prediction": "Likely RTO" if score >= 70 else ("Needs review" if score >= 40 else "Likely successful delivery")
    }
