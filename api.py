# """
# api.py
#
# A RESTish API, built with Flask, that serves FOUR machine learning artifacts
# produced in earlier labs:
#
#     1. Regression       -- SME monthly revenue prediction
#     2. Classification   -- SME credit risk category
#     3. Clustering       -- Mall customer segment assignment
#     4. Association Rule -- Product recommendation from a cart
#        Mining (ARM)
#
# The point of this file is to show you that a fitted scikit-learn /
# imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
# missing values, encode categorical columns, scale numeric columns, and (for
# the classifier) apply SMOTE only during training -- never during inference.
# Because all of that preprocessing lives inside the Pipeline itself, this file
# never manually label-encodes anything, never manually engineers date features,
# and never manually reorders columns before calling .predict(). It builds a
# DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
# This is the main lesson of the lab: "the pipeline IS the deployment artifact."
#
# Run this file directly to start a local development server:
#
#     python api.py
# """
#
# import os
#
# import joblib
# import pandas as pd
# from flask import Flask, jsonify, request
# from flask_cors import CORS
#
# app = Flask(__name__)
#
# # -----------------------------------------------------------------------
# # CORS (Cross-Origin Resource Sharing)
# # -----------------------------------------------------------------------
# # By default, a browser blocks JavaScript on one origin (e.g., a page served
# # from http://127.0.0.1:5500) from calling an API on a different origin
# # (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# # specific origins we trust during local development.
# #
# # A few alternative configurations are shown below (commented out) so you can
# # see how the restriction can be loosened or tightened depending on your
# # deployment target. Only ONE of these should be active at a time.
#
# # Option A (the most permissive. This is fine for a quick classroom demo or
# # local development, *NEVER* for production):
# # CORS(app)
#
# # Option B (allow only a specific frontend origin):
# # CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})
#
# # Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# # ports our lab's frontend files are typically served from):
# CORS(app, resources={r"/api/*": {"origins": [
#     "http://localhost:5000", "http://127.0.0.1:5000",
#     "http://localhost:5500", "http://127.0.0.1:5500",
#     "https://localhost:443", "https://127.0.0.1:443",
#     "http://localhost:63342",  # VS Code Live Server extension
#     "null",  # allows opening a frontend .html file directly (file://) with no web server
# ]}})
#
# # -----------------------------------------------------------------------
# # The best practice is to load the four persisted artifacts ONCE, at startup,
# # not per-request.
# # Loading inside a route function would re-read the file from disk on every
# # single API call, which is slow and pointless -- the fitted model never
# # changes while the server is running.
# # -----------------------------------------------------------------------
# MODEL_DIR = "./model"
#
# regression_pipeline = joblib.load(
#     os.path.join(
#         MODEL_DIR,
#         "ridge_regressor_for_daily_sales_revenue.joblib"
#     )
# )
#
# classification_pipeline = joblib.load(
#     os.path.join(
#         MODEL_DIR,
#         "logistic_regression_classifier_for_distress_risk.joblib"
#     )
# )
#
# regression_pipeline = joblib.load(
#     os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
# )
#
# classification_pipeline = joblib.load(
#     os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
# )
#
# clustering_artifact = joblib.load(
#     os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
# )
# clustering_scaler = clustering_artifact["scaler"]
# clustering_kmeans = clustering_artifact["kmeans"]
#
# arm_artifact = joblib.load(
#     os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
# )
# recommendation_lookup = arm_artifact["recommendation_lookup"]
# sku_lookup = arm_artifact["sku_lookup"]
#
# print("All the four artifacts have been loaded. The API is ready.")
#
#
# # 1. REGRESSION -- predict monthly revenue (KES) for an SME
#
# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"sector": "Technology", "county": "Nairobi", "quarter": "Q3",
# #             "business_age_years": 4.5, "num_employees": 12,
# #             "owner_education_level": "Secondary",
# #             "monthly_marketing_spend_kes": 45000,
# #             "avg_daily_foot_traffic": null,
# #             "online_presence_score": 78.0, "loan_amount_kes": 250000,
# #             "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
# #             "competitor_density": 5, "customer_satisfaction_score": 4.2}'
# #
# # Sample request (PowerShell):
# #   Invoke-RestMethod -Method Post `
# #     -Uri "http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions" `
# #     -ContentType "application/json" `
# #     -Body (@{sector="Technology"; county="Nairobi"; quarter="Q3";
# #              business_age_years=4.5; num_employees=12;
# #              owner_education_level="Secondary";
# #              monthly_marketing_spend_kes=45000; avg_daily_foot_traffic=$null;
# #              online_presence_score=78.0; loan_amount_kes=250000;
# #              interest_rate_pct=14.5; inflation_rate_pct=6.5;
# #              competitor_density=5; customer_satisfaction_score=4.2} | ConvertTo-Json)
# #
# # Note: avg_daily_foot_traffic is deliberately null for a Technology-sectorv
# # business (fully online -- foot traffic is structurally not applicable, not
# # missing at random). The pipeline's own imputer handles the null value; nothing
# # in this file needs to know that rule.
# @app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
# def predict_revenue():
#     payload = request.get_json()
#
#     # Build a one-row DataFrame straight from the JSON body. No manual
#     # label-encoding, no manual date parsing, no manual column reordering, etc. --
#     # the pipeline's ColumnTransformer already knows which columns it expects
#     # and how to preprocess each one.
#     input_row = pd.DataFrame([payload])
#
#     predicted_log_revenue = regression_pipeline.predict(input_row)[0]
#     # The pipeline was trained on log1p(revenue) because revenue was
#     # right-skewed; we invert that single transformation here to report the
#     # predictions in KES.
#     import numpy as np
#     predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
#
#     return jsonify({
#         "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
#         "model": "Lasso Regressor",
#     })
#
#
# # 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)
#
# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"sector": "Retail", "county": "Nairobi", "quarter": "Q2",
# #             "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
# #             "business_age_years": 2.1, "num_employees": 4,
# #             "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
# #             "loan_term_months": 12, "interest_rate_pct": 16.5,
# #             "collateral_value_kes": null, "debt_to_income_ratio": 0.62,
# #             "credit_score": 590, "num_previous_loans": 1,
# #             "missed_payments_count": 3, "online_presence_score": 40.0,
# #             "customer_satisfaction_score": 3.4, "competitor_density": 6}'
# #
# #
# # Note: collateral_value_kes is null here because "Working Capital" loans are
# # more often unsecured (structural missingness, not missing at random) --
# # again, the pipeline's imputer + missingness indicator already handles this
# # internally; this route does not need special code for it.
# @app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
# def predict_credit_risk():
#     payload = request.get_json()
#     input_row = pd.DataFrame([payload])
#
#     predicted_class = classification_pipeline.predict(input_row)[0]
#     predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
#     class_labels = classification_pipeline.named_steps["model"].classes_
#
#     return jsonify({
#         "predicted_credit_risk_category": str(predicted_class),
#         "class_probabilities": {
#             str(label): round(float(prob), 4)
#             for label, prob in zip(class_labels, predicted_probabilities)
#         },
#         "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
#     })
#
#
# # 3. CLUSTERING -- assign a new mall customer to an existing segment
#
# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
# #
# # Note: unlike the two Pipelines above, K-Means does not bundle its own
# # scaler, so this is the one endpoint where the scaling step is applied
# # explicitly (scaler.transform(...) then kmeans.predict(...)) -- exactly the
# # two-object pattern the clustering notebook persisted together for this
# # reason. There is still no manual encoding or feature engineering: the
# # three numeric fields are used exactly as the client sends them.
# @app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
# def predict_customer_segment():
#     payload = request.get_json()
#     input_row = pd.DataFrame([payload])
#
#     scaled_input = clustering_scaler.transform(input_row)
#     assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])
#
#     return jsonify({
#         "assigned_cluster": assigned_cluster,
#         "model": "K-Means (k=5, standardized features)",
#     })
#
#
# # 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart
#
# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"cart": ["H06", "H07"]}'
# #
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"cart": ["P04", "P01"]}'
# #
# # Note: there is no scikit-learn ".predict()" here -- the mined association
# # rules were precomputed into a lookup dictionary
# # (antecedent SKU-set -> ranked candidate consequents)
# # during the notebook's persistence stage. This
# # endpoint performs the exact same lookup+rank step a production recommender
# # microservice would perform on every incoming request; it does not mine
# # for the rules again.
# @app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
# def recommend_products():
#     payload = request.get_json()
#     cart = set(payload.get("cart", []))
#     top_n = int(payload.get("top_n", 3))
#
#     candidates = {}
#     for antecedent, consequent_list in recommendation_lookup.items():
#         if antecedent.issubset(cart):
#             for sku, confidence, lift in consequent_list:
#                 if sku not in cart:
#                     if sku not in candidates or confidence > candidates[sku][0]:
#                         candidates[sku] = (confidence, lift)
#
#     ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
#     recommendations = [
#         {
#             "sku": str(sku),
#             "product_name": sku_lookup.loc[sku, "product_name"],
#             "confidence": round(float(confidence), 3),
#             "lift": round(float(lift), 2),
#         }
#         for sku, (confidence, lift) in ranked
#     ]
#
#     return jsonify({
#         "cart": sorted(cart),
#         "recommendations": recommendations,
#         "model": "Apriori association rules, cleaned + precomputed lookup",
#     })
#
#
# # -----------------------------------------------------------------------
# # Entry point
# # -----------------------------------------------------------------------
# if __name__ == "__main__":
#     # Development server. debug=True gives auto-reload and an interactive
#     # debugger on errors -- convenient in class, but NEVER use debug=True in
#     # a deployed/public environment (it can expose a remote code execution
#     # console to anyone who can reach the server).
#     app.run(debug=True)
#
#     # Production-style alternative (no debugger, no auto-reload):
#     # app.run(debug=False, host="0.0.0.0", port=5000)
#
#     # HTTPS/TLS alternative for local development (requires a self-signed
#     # certificate; see app_server_reverse_proxy_server_setup.md):
#     # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))

"""
Flask API - Group Machine Learning Assignment

Member 1:
    1. Daily sales revenue regression
    2. Business distress risk classification

Member 2:
    3. Mall customer clustering
    4. E-commerce product recommendations
"""

import os
import joblib
import pandas as pd
import numpy as np

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

CORS(app, resources={r"/api/.*": {"origins": [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://localhost:443",
    "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]}})

# ---------------------------------------------------------
# MODEL DIRECTORY
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_optional_artifact(filename):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"Optional artifact not found: {filename}")
        return None

    return joblib.load(path)


def get_json_payload():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or not payload:
        return None, (jsonify({
            "error": "Request body must contain a non-empty JSON object."
        }), 400)

    return payload, None


# ---------------------------------------------------------
# MEMBER 1: REGRESSION AND CLASSIFICATION
# ---------------------------------------------------------

revenue_regression_pipeline = joblib.load(
    os.path.join(
        MODEL_DIR,
        "ridge_regressor_for_daily_sales_revenue.joblib"
    )
)

distress_classification_pipeline = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_regression_classifier_for_distress_risk.joblib"
    )
)

# ---------------------------------------------------------
# MEMBER 2: CLUSTERING AND ASSOCIATION RULES
# ---------------------------------------------------------
# Both artifacts are loaded ONCE at startup (never inside a route, and never
# retrained here). If a file is missing the app fails immediately, which is
# better than a half-working API.

# Clustering: the preprocessing Pipeline (StandardScaler -> KNNImputer) and the
# fitted K-Means were saved as two separate files in the clustering lab.
customer_preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "customer_preprocessor.joblib")
)
customer_kmeans = joblib.load(
    os.path.join(MODEL_DIR, "customer_kmeans.joblib")
)

# Read the expected input columns from the artifact itself instead of
# hard-coding them, so the API can never drift from the trained model.
CUSTOMER_FEATURES = list(customer_preprocessor.feature_names_in_)

# Cluster centres are stored in scaled units. Convert them back to real units
# (days, orders per year, KES, categories) so each segment can be explained.
CUSTOMER_SEGMENT_PROFILES = {
    cluster_id: {
        feature: round(float(value), 2)
        for feature, value in zip(CUSTOMER_FEATURES, centre)
    }
    for cluster_id, centre in enumerate(
        customer_preprocessor.named_steps["scale"].inverse_transform(
            customer_kmeans.cluster_centers_
        )
    )
}
CUSTOMER_SEGMENT_NAMES = {
    0: "Occasional High Spenders",
    1: "Loyal High Spenders",
    2: "Dormant Customers",
    3: "New/Infrequent Low Spenders",
    4: "Frequent Moderate Spenders",
}

# Association rules: a dict holding the mined rules table plus lookups.
streaming_artifact = joblib.load(
    os.path.join(MODEL_DIR, "streaming_recommender.joblib")
)
streaming_rules = streaming_artifact["rules"]
streaming_rule_index = streaming_artifact["rule_index"]
streaming_show_lookup = streaming_artifact["show_lookup"]
streaming_name_to_id = streaming_artifact["name_to_id"]
streaming_popularity = streaming_artifact["popularity"]

# Lower-case title -> show_id, so "crime city" and "Crime City" both work.
streaming_title_to_id = {
    title.lower(): show_id
    for title, show_id in streaming_name_to_id.items()
}

print("All the artifacts have been loaded. The API is ready.")


# ---------------------------------------------------------
# HOME ENDPOINT
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Machine Learning Models API",
        "member_1": [
            "/api/v1/models/daily-sales-revenue-regressor/predictions",
            "/api/v1/models/distress-risk-classifier/predictions"
        ],
        "member_2": [
            "/api/v1/models/mall-customer-segmenter/predictions",
            "/api/v1/models/ecommerce-recommender/predictions"
        ]
    })


# =========================================================
# 1. MEMBER 1 - REGRESSION
# Predict daily sales revenue (KES)
# =========================================================

@app.route(
    "/api/v1/models/daily-sales-revenue-regressor/predictions",
    methods=["POST"]
)
def predict_daily_sales_revenue():

    payload, error = get_json_payload()

    if error is not None:
        return error

    input_row = pd.DataFrame([payload])

    try:
        prediction = revenue_regression_pipeline.predict(
            input_row
        )[0]


        predicted_revenue = float(np.expm1(prediction))

        return jsonify({
            "predicted_daily_sale_revenue_kes": round(
                predicted_revenue, 2
            ),
            "model": "Ridge Regression",
            "target": "daily_sale_revenue_kes"
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Unable to generate revenue prediction.",
            "details": str(error)
        }), 400


# =========================================================
# 2. MEMBER 1 - CLASSIFICATION
# Predict business distress risk
# =========================================================

@app.route(
    "/api/v1/models/distress-risk-classifier/predictions",
    methods=["POST"]
)
def predict_distress_risk():

    payload, error = get_json_payload()

    if error is not None:
        return error

    input_row = pd.DataFrame([payload])

    try:
        predicted_class = (
            distress_classification_pipeline.predict(input_row)[0]
        )

        predicted_probabilities = (
            distress_classification_pipeline.predict_proba(input_row)[0]
        )

        if hasattr(distress_classification_pipeline, "classes_"):
            class_labels = distress_classification_pipeline.classes_
        else:
            class_labels = (
                distress_classification_pipeline
                .named_steps["model"].classes_
            )

        return jsonify({
            "predicted_distress_risk_category": str(predicted_class),

            "class_probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    class_labels,
                    predicted_probabilities
                )
            },

            "model": "Logistic Regression",
            "target": "distress_risk_category"
        })

    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return jsonify({
            "error": "Unable to generate distress risk prediction.",
            "details": str(error)
        }), 400


# =========================================================
# 3. MEMBER 2 - CLUSTERING
# Assign a customer to one of the 5 K-Means segments
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/customer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"recency_days": 20, "frequency_per_year": 22,
#             "avg_order_value_kes": 8000, "num_categories_purchased": 6}'
#
# A field may be null: the saved preprocessing pipeline (KNNImputer) fills it.
# =========================================================

@app.route(
    "/api/v1/models/customer-segmenter/predictions",
    methods=["POST"]
)
def predict_customer_segment():

    payload, error = get_json_payload()

    if error is not None:
        return error

    missing = [f for f in CUSTOMER_FEATURES if f not in payload]

    if missing:
        return jsonify({
            "error": "Missing required field(s).",
            "missing": missing,
            "expected_fields": CUSTOMER_FEATURES
        }), 400

    values = {}

    for feature in CUSTOMER_FEATURES:
        value = payload[feature]

        if value is None:
            values[feature] = np.nan
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            return jsonify({
                "error": f"'{feature}' must be a non-negative number or null.",
                "received": value
            }), 400
        else:
            values[feature] = float(value)

    if all(np.isnan(v) for v in values.values()):
        return jsonify({
            "error": "At least one field must have a value."
        }), 400

    try:
        # One row, columns in the order the pipeline was trained on.
        input_row = pd.DataFrame([values], columns=CUSTOMER_FEATURES)

        # Same two steps as training: scale + impute, then K-Means.
        prepared = customer_preprocessor.transform(input_row)
        prepared = pd.DataFrame(prepared, columns=CUSTOMER_FEATURES)

        assigned_cluster = int(customer_kmeans.predict(prepared)[0])

        return jsonify({
            "assigned_cluster": assigned_cluster,
            "segment_name": CUSTOMER_SEGMENT_NAMES[assigned_cluster],
            "segment_profile": CUSTOMER_SEGMENT_PROFILES[assigned_cluster],
            "missing_values_imputed": [
                f for f, v in values.items() if np.isnan(v)
            ],
            "model": "K-Means (k=5, scaled + KNN-imputed features)"
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Unable to assign customer segment.",
            "details": str(error)
        }), 400


# =========================================================
# 4. MEMBER 2 - ASSOCIATION RULE RECOMMENDATIONS
# Recommend streaming shows from what a viewer has already watched
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/streaming-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"watched": ["S11", "S15"], "top_n": 3}'
#
# "watched" accepts show IDs (S01..S15) or show titles ("Crime City").
# =========================================================

@app.route(
    "/api/v1/models/streaming-recommender/predictions",
    methods=["POST"]
)
def recommend_shows():

    payload, error = get_json_payload()

    if error is not None:
        return error

    watched_input = payload.get("watched")

    if not isinstance(watched_input, list) or not watched_input:
        return jsonify({
            "error": "'watched' must be a non-empty list of show IDs or titles.",
            "example": {"watched": ["S11", "S15"], "top_n": 3}
        }), 400

    top_n = payload.get("top_n", 3)

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        return jsonify({
            "error": "'top_n' must be a positive whole number."
        }), 400

    # Turn each item (ID or title) into a show ID; collect any we don't know.
    watched = set()
    unknown = []

    for item in watched_input:
        key = str(item).strip()

        if key.upper() in streaming_show_lookup.index:
            watched.add(key.upper())
        elif key.lower() in streaming_title_to_id:
            watched.add(streaming_title_to_id[key.lower()])
        else:
            unknown.append(item)

    if unknown:
        return jsonify({
            "error": "Unknown show(s) in 'watched'.",
            "unknown": unknown,
            "valid_show_ids": sorted(streaming_show_lookup.index)
        }), 400

    try:
        # Only look at rules that mention a watched show (via rule_index),
        # then keep those whose WHOLE antecedent has been watched.
        candidate_rule_ids = {
            rule_id
            for show_id in watched
            for rule_id in streaming_rule_index.get(show_id, [])
        }

        candidates = {}

        for rule_id in candidate_rule_ids:
            rule = streaming_rules.loc[rule_id]

            if not rule["antecedents"].issubset(watched):
                continue

            for show_id in rule["consequents"]:
                if show_id in watched:
                    continue

                score = (float(rule["confidence"]), float(rule["lift"]))

                if show_id not in candidates or score > candidates[show_id][0]:
                    candidates[show_id] = (
                        score,
                        float(rule["support"]),
                        rule["antecedents"]
                    )

        ranked = sorted(
            candidates.items(), key=lambda item: item[1][0], reverse=True
        )[:top_n]

        def show_details(show_id):
            row = streaming_show_lookup.loc[show_id]
            return {
                "show_id": show_id,
                "title": str(row["item_name"]),
                "category": str(row["category"]),
                "avg_episode_minutes": int(row["avg_episode_minutes"])
            }

        if ranked:
            recommendations = [
                {
                    **show_details(show_id),
                    "confidence": round(confidence, 3),
                    "lift": round(lift, 2),
                    "support": round(support, 3),
                    "because_you_watched": sorted(
                        streaming_show_lookup.loc[a, "item_name"]
                        for a in antecedent
                    )
                }
                for show_id, ((confidence, lift), support, antecedent)
                in ranked
            ]
            basis = "association_rules"
        else:
            # Cold start: no mined rule matches this combination of shows,
            # so fall back to the most popular shows not yet watched.
            popular = streaming_popularity.drop(
                labels=list(watched), errors="ignore"
            ).sort_values(ascending=False).head(top_n)

            recommendations = [
                {
                    **show_details(show_id),
                    "popularity": round(float(share), 3)
                }
                for show_id, share in popular.items()
            ]
            basis = "popularity_fallback"

        return jsonify({
            "watched": sorted(watched),
            "recommendation_basis": basis,
            "recommendations": recommendations,
            "model": "Apriori Association Rules"
        })

    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return jsonify({
            "error": "Unable to generate recommendations.",
            "details": str(error)
        }), 400


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
