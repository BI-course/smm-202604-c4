import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # This logic works locally (where model is in parent folder) AND on Streamlit Cloud (where you copy it inside the folder)
    model_dir = "model" if os.path.exists("model") else "../model"
    
    regression_pipeline = joblib.load(f"{model_dir}/lasso_regressor_for_sme_revenue.joblib")
    classification_pipeline = joblib.load(f"{model_dir}/svc_classifier_for_sme_credit_risk.joblib")
    clustering_artifact = joblib.load(f"{model_dir}/kmeans_mall_customer_segmentation.joblib")
    arm_artifact = joblib.load(f"{model_dir}/apriori_recommendation_rules.joblib")
    
    return regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact

regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_models()

# Extract the specific parts of the clustering and arm artifacts
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

# --- PREDICTION FUNCTIONS ---
def predict_revenue(sector, county, quarter, business_age_years, num_employees, owner_education_level, monthly_marketing_spend_kes, avg_daily_foot_traffic, online_presence_score, loan_amount_kes, interest_rate_pct, inflation_rate_pct, competitor_density, customer_satisfaction_score):
    payload = {
        "sector": sector, "county": county, "quarter": quarter, "business_age_years": business_age_years, "num_employees": num_employees,
        "owner_education_level": owner_education_level, "monthly_marketing_spend_kes": monthly_marketing_spend_kes,
        "avg_daily_foot_traffic": avg_daily_foot_traffic if avg_daily_foot_traffic != 0 else None,
        "online_presence_score": online_presence_score, "loan_amount_kes": loan_amount_kes, "interest_rate_pct": interest_rate_pct,
        "inflation_rate_pct": inflation_rate_pct, "competitor_density": competitor_density, "customer_satisfaction_score": customer_satisfaction_score,
    }
    input_row = pd.DataFrame([payload])
    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    return f"KES {float(np.expm1(predicted_log_revenue)):,.2f} / month"

def predict_credit_risk(sector, county, quarter, loan_purpose, owner_education_level, business_age_years, num_employees, monthly_revenue_kes, loan_amount_kes, loan_term_months, interest_rate_pct, collateral_value_kes, debt_to_income_ratio, credit_score, num_previous_loans, missed_payments_count, online_presence_score, customer_satisfaction_score, competitor_density):
    payload = {
        "sector": sector, "county": county, "quarter": quarter, "loan_purpose": loan_purpose, "owner_education_level": owner_education_level,
        "business_age_years": business_age_years, "num_employees": num_employees, "monthly_revenue_kes": monthly_revenue_kes,
        "loan_amount_kes": loan_amount_kes, "loan_term_months": loan_term_months, "interest_rate_pct": interest_rate_pct,
        "collateral_value_kes": collateral_value_kes if collateral_value_kes != 0 else None, "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score, "num_previous_loans": num_previous_loans, "missed_payments_count": missed_payments_count,
        "online_presence_score": online_presence_score, "customer_satisfaction_score": customer_satisfaction_score, "competitor_density": competitor_density,
    }
    input_row = pd.DataFrame([payload])
    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    probability_lines = "\n".join(f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities))
    return f"Predicted category: {predicted_class}\n\n{probability_lines}"

def predict_segment(age, annual_income_k, spending_score):
    input_row = pd.DataFrame([{"Age": age, "Annual Income (k$)": annual_income_k, "Spending Score (1-100)": spending_score}])
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return f"Assigned segment: Cluster {cluster}"

def recommend_products(cart_text, top_n):
    cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)
    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
    if not ranked:
        return "No recommendations for this cart -- try a SKU combination seen in the training orders."
    lines = [f"{sku_lookup.loc[sku, 'product_name']} ({sku})  --  confidence {confidence:.1%}, lift {lift:.2f}" for sku, (confidence, lift) in ranked]
    return "\n".join(lines)


# --- STREAMLIT UI ---
st.title("BBT 4206 -- Model Validation Demos")
st.write("Switch tabs to try a different model.")

tab1, tab2, tab3, tab4 = st.tabs(["SME Revenue Regressor", "SME Credit Risk Classifier", "Mall Customer Segmenter", "E-commerce Recommender"])

with tab1:
    st.header("SME Revenue Regressor")
    col1, col2 = st.columns(2)
    with col1:
        sector = st.text_input("Sector", value="Technology")
        county = st.text_input("County", value="Nairobi")
        quarter = st.text_input("Quarter", value="Q3")
        business_age = st.number_input("Business Age (years)", value=4.5)
        num_employees = st.number_input("Number of Employees", value=12)
        owner_ed = st.selectbox("Owner Education Level", ["Primary", "Secondary", "Tertiary", "Postgraduate"], index=1)
        marketing_spend = st.number_input("Monthly Marketing Spend (KES)", value=45000)
    with col2:
        foot_traffic = st.number_input("Avg Daily Foot Traffic (0 if N/A)", value=0)
        online_presence = st.number_input("Online Presence Score", value=78.0)
        loan_amount = st.number_input("Loan Amount (KES)", value=250000)
        interest_rate = st.number_input("Interest Rate (%)", value=14.5)
        inflation_rate = st.number_input("Inflation Rate (%)", value=6.5)
        comp_density = st.number_input("Competitor Density", value=5)
        satisfaction = st.number_input("Customer Satisfaction Score", value=4.2)
    
    if st.button("Predict Revenue"):
        result = predict_revenue(sector, county, quarter, business_age, num_employees, owner_ed, marketing_spend, foot_traffic, online_presence, loan_amount, interest_rate, inflation_rate, comp_density, satisfaction)
        st.success(result)

with tab2:
    st.header("SME Credit Risk Classifier")
    col3, col4 = st.columns(2)
    with col3:
        sector_c = st.text_input("Sector ", value="Retail")
        county_c = st.text_input("County ", value="Nairobi")
        quarter_c = st.text_input("Quarter ", value="Q2")
        loan_purpose = st.text_input("Loan Purpose", value="Working Capital")
        owner_ed_c = st.selectbox("Owner Education Level ", ["Primary", "Secondary", "Tertiary", "Postgraduate"], index=1)
        business_age_c = st.number_input("Business Age (years) ", value=2.1)
        num_employees_c = st.number_input("Number of Employees ", value=4)
        monthly_rev_c = st.number_input("Monthly Revenue (KES)", value=850000)
        loan_amount_c = st.number_input("Loan Amount (KES) ", value=150000)
        loan_term = st.number_input("Loan Term (months)", value=12)
    with col4:
        interest_rate_c = st.number_input("Interest Rate (%) ", value=16.5)
        collateral = st.number_input("Collateral Value (0 if unsecured)", value=0)
        dti = st.number_input("Debt-to-Income Ratio", value=0.62)
        credit_score = st.number_input("Credit Score", value=590)
        prev_loans = st.number_input("Number of Previous Loans", value=1)
        missed_pay = st.number_input("Missed Payments Count", value=3)
        online_presence_c = st.number_input("Online Presence Score ", value=40.0)
        satisfaction_c = st.number_input("Customer Satisfaction Score ", value=3.4)
        comp_density_c = st.number_input("Competitor Density ", value=6)
    
    if st.button("Classify Risk"):
        result = predict_credit_risk(
            sector_c, county_c, quarter_c, loan_purpose, owner_ed_c, business_age_c, 
            num_employees_c, monthly_rev_c, loan_amount_c, loan_term, interest_rate_c, 
            collateral, dti, credit_score, prev_loans, missed_pay, online_presence_c, 
            satisfaction_c, comp_density_c
        )
        st.success(result)

with tab3:
    st.header("Mall Customer Segmenter")
    age = st.number_input("Age ", value=27)
    income = st.number_input("Annual Income (k$) ", value=45)
    spending = st.number_input("Spending Score (1-100) ", value=72)
    
    if st.button("Predict Segment"):
        result = predict_segment(age, income, spending)
        st.success(result)

with tab4:
    st.header("E-commerce Recommender")
    cart = st.text_input("Cart (comma-separated SKUs)", value="H06, H07")
    top_n = st.number_input("Top N Recommendations", value=3)
    
    if st.button("Get Recommendations"):
        result = recommend_products(cart, top_n)
        st.success(result)