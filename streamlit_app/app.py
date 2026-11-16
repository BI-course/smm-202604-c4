import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Check if models are in the current dir, otherwise look one level up
    model_dir = "model" if os.path.exists("model") else "../model"
    
    reg = joblib.load(f"{model_dir}/ridge_regressor_for_daily_sales_revenue.joblib")
    clf = joblib.load(f"{model_dir}/logistic_regression_classifier_for_distress_risk.joblib")
    
    clus_pre = joblib.load(f"{model_dir}/customer_preprocessor.joblib")
    clus_kmeans = joblib.load(f"{model_dir}/customer_kmeans.joblib")
    
    recommender = joblib.load(f"{model_dir}/streaming_recommender.joblib")
    
    return reg, clf, clus_pre, clus_kmeans, recommender

reg_model, clf_model, clus_pre, clus_kmeans, recommender_artifact = load_models()

# Extract specific parts of the recommender
streaming_rules = recommender_artifact["rules"]
streaming_rule_index = recommender_artifact["rule_index"]
streaming_show_lookup = recommender_artifact["show_lookup"]
streaming_name_to_id = recommender_artifact["name_to_id"]
streaming_popularity = recommender_artifact["popularity"]

streaming_title_to_id = {title.lower(): show_id for title, show_id in streaming_name_to_id.items()}

CUSTOMER_FEATURES = list(clus_pre.feature_names_in_)
CUSTOMER_SEGMENT_NAMES = {
    0: "Occasional High Spenders",
    1: "Loyal High Spenders",
    2: "Dormant Customers",
    3: "New/Infrequent Low Spenders",
    4: "Frequent Moderate Spenders",
}

# --- UI SETUP ---
st.title("BBT 4206 -- Model Validation Demos")
st.write("Switch tabs to try a different model.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Daily Sales Revenue", 
    "Distress Risk Classifier", 
    "Customer Segmenter", 
    "Streaming Recommender"
])

# ==========================================
# TAB 1: REGRESSION (DAILY SALES)
# ==========================================
with tab1:
    st.header("Daily Sales Revenue Regressor")
    st.write("Predict the daily sales revenue (KES).")
    
    col1, col2 = st.columns(2)
    with col1:
        store_size = st.number_input("Store Size (sqm)", value=150.0)
        num_staff = st.number_input("Number of Staff", value=10)
        footfall = st.number_input("Footfall Count", value=300)
        promotions = st.selectbox("Promotions Active", ["Yes", "No"])
        basket_size = st.number_input("Avg Basket Size (KES)", value=1200.0)
        local_events = st.selectbox("Local Events Nearby", ["Yes", "No"])
    with col2:
        store_format = st.selectbox("Store Format", ["Supermarket", "Convenience", "Hypermarket", "Kiosk"])
        location_type = st.selectbox("Location Type", ["CBD", "Mall", "Residential", "Street"])
        delivery = st.selectbox("Delivery Available", ["Yes", "No"])
        store_size_category = st.selectbox("Store Size Category", ["Small", "Medium", "Large"])
        store_age = st.number_input("Store Age (years)", value=3.5)
        online_orders = st.number_input("Online Order Count", value=45)
        loyalty_members = st.number_input("Loyalty Program Members", value=150)
        
    if st.button("Predict Daily Sales"):
        # The trick: Map boolean-like strings to 1/0 for the numeric pipeline,
        # but keep the true categorical columns as strings for the OneHotEncoder.
        payload = {
            "store_size_sqm": float(store_size),
            "num_staff": float(num_staff),
            "footfall_count": float(footfall),
            "promotions_active": 1.0 if promotions == "Yes" else 0.0,       # <--- Number!
            "avg_basket_size_kes": float(basket_size),
            "local_events_nearby": 1.0 if local_events == "Yes" else 0.0,   # <--- Number!
            "store_format": store_format,                                   # <--- String!
            "location_type": location_type,                                 # <--- String!
            "delivery_available": delivery,                                 # <--- String!
            "store_size_category": store_size_category,                     # <--- String!
            "store_age_years": float(store_age),
            "online_order_count": float(online_orders),
            "loyalty_program_members": float(loyalty_members)
        }
        
        try:
            input_row = pd.DataFrame([payload])
            prediction = reg_model.predict(input_row)[0]
            st.success(f"Predicted Revenue: KES {float(np.expm1(prediction)):,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ==========================================
# TAB 2: CLASSIFICATION (DISTRESS RISK)
# ==========================================
with tab2:
    st.header("Business Distress Risk Classifier")
    st.write("Predict the risk category for a business.")
    
    col3, col4 = st.columns(2)
    with col3:
        debt_to_equity = st.number_input("Debt to Equity Ratio", value=1.5)
        interest_coverage = st.number_input("Interest Coverage Ratio", value=2.0)
        current_ratio = st.number_input("Current Ratio", value=1.2)
    with col4:
        working_capital = st.number_input("Working Capital to Assets Ratio", value=0.4)
        net_profit = st.number_input("Net Profit Margin (%)", value=12.5)
        rev_growth = st.number_input("Revenue Growth (%)", value=8.0)
        
    if st.button("Classify Risk"):
        payload = {
            "debt_to_equity_ratio": debt_to_equity,
            "interest_coverage_ratio": interest_coverage,
            "current_ratio": current_ratio,
            "working_capital_to_assets_ratio": working_capital,
            "net_profit_margin_pct": net_profit,
            "revenue_growth_pct": rev_growth
        }
        try:
            input_row = pd.DataFrame([payload])
            predicted_class = clf_model.predict(input_row)[0]
            probs = clf_model.predict_proba(input_row)[0]
            
            if hasattr(clf_model, "classes_"):
                labels = clf_model.classes_
            else:
                labels = clf_model.named_steps["model"].classes_
                
            prob_text = "\n".join([f"{label}: {prob:.1%}" for label, prob in zip(labels, probs)])
            st.success(f"Predicted Category: {predicted_class}\n\n{prob_text}")
        except Exception as e:
            st.error(f"Classification Error: {e}")

# ==========================================
# TAB 3: CLUSTERING (CUSTOMER SEGMENTS)
# ==========================================
with tab3:
    st.header("Customer Segmenter")
    st.write("Assign a customer to a K-Means cluster.")
    
    # These exact fields were pulled from api.py!
    col5, col6 = st.columns(2)
    with col5:
        recency = st.number_input("Recency (days)", value=20)
        frequency = st.number_input("Frequency (per year)", value=22)
    with col6:
        avg_order = st.number_input("Avg Order Value (KES)", value=8000)
        num_categories = st.number_input("Number of Categories Purchased", value=6)
        
    if st.button("Predict Segment"):
        payload = {
            "recency_days": float(recency),
            "frequency_per_year": float(frequency),
            "avg_order_value_kes": float(avg_order),
            "num_categories_purchased": float(num_categories)
        }
        try:
            input_row = pd.DataFrame([payload], columns=CUSTOMER_FEATURES)
            prepared = clus_pre.transform(input_row)
            prepared = pd.DataFrame(prepared, columns=CUSTOMER_FEATURES)
            assigned_cluster = int(clus_kmeans.predict(prepared)[0])
            
            segment_name = CUSTOMER_SEGMENT_NAMES.get(assigned_cluster, f"Cluster {assigned_cluster}")
            st.success(f"Assigned Segment: **{segment_name}** (Cluster {assigned_cluster})")
        except Exception as e:
            st.error(f"Clustering Error: {e}")

# ==========================================
# TAB 4: RECOMMENDER (STREAMING SHOWS)
# ==========================================
with tab4:
    st.header("Streaming Show Recommender")
    st.write("Recommend shows based on what you've watched.")
    
    # We provide default IDs that exist in the dataset
    watched_input = st.text_input("Watched Shows (comma-separated titles or IDs)", value="S11, S15")
    top_n = st.number_input("Top N Recommendations", value=3)
    
    if st.button("Get Recommendations"):
        watched_list = [item.strip() for item in watched_input.split(",") if item.strip()]
        watched = set()
        unknown = []
        
        for item in watched_list:
            key = str(item).strip()
            if key.upper() in streaming_show_lookup.index:
                watched.add(key.upper())
            elif key.lower() in streaming_title_to_id:
                watched.add(streaming_title_to_id[key.lower()])
            else:
                unknown.append(item)
                
        if unknown:
            st.warning(f"Unknown shows: {', '.join(unknown)}")
        
        if not watched:
            st.error("Please enter at least one valid show ID or title.")
        else:
            try:
                candidate_rule_ids = {
                    rule_id for show_id in watched for rule_id in streaming_rule_index.get(show_id, [])
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
                            candidates[show_id] = (score, float(rule["support"]), rule["antecedents"])
                            
                ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
                
                if ranked:
                    st.success("Recommendations based on your watch history:")
                    for show_id, ((conf, lift), support, antecedent) in ranked:
                        row = streaming_show_lookup.loc[show_id]
                        st.write(f"**{row['item_name']}** ({show_id}) - Confidence: {conf:.1%}, Lift: {lift:.2f}")
                else:
                    # Fallback to popularity
                    st.info("No exact rule matches found. Showing popular shows instead:")
                    popular = streaming_popularity.drop(labels=list(watched), errors="ignore").sort_values(ascending=False).head(int(top_n))
                    for show_id, share in popular.items():
                        row = streaming_show_lookup.loc[show_id]
                        st.write(f"**{row['item_name']}** ({show_id}) - Popularity: {share:.1%}")
            except Exception as e:
                st.error(f"Recommender Error: {e}")