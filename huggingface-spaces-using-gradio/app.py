
import os
import requests
import gradio as gr


# ============================================================
# CONFIGURATION
# ============================================================


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://smm-202604-c4-clara.onrender.com"
).rstrip("/")

REQUEST_TIMEOUT = 90
# ============================================================
# SHARED API REQUEST FUNCTION
# ============================================================

def call_api(endpoint, payload):
    """
    Sends a JSON POST request to the Flask API
    and returns the response as a dictionary.
    """

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        try:
            result = response.json()
        except ValueError:
            return {
                "error": "The API returned an invalid JSON response.",
                "status_code": response.status_code,
                "response": response.text[:1000]
            }

        if not response.ok:
            return {
                "error": result.get(
                    "error",
                    "The API request failed."
                ),
                "details": result.get("details"),
                "status_code": response.status_code,
                "response": result
            }

        return result

    except requests.exceptions.Timeout:
        return {
            "error": "The API request timed out.",
            "details": (
                "The API may be waking up after inactivity. "
                "Please try again."
            )
        }

    except requests.exceptions.ConnectionError:
        return {
            "error": "Unable to connect to the Flask API.",
            "details": (
                "Check that the API is running and that "
                "API_BASE_URL is configured correctly."
            )
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": "An unexpected request error occurred.",
            "details": str(error)
        }


# ============================================================
# 1. DAILY SALES REVENUE REGRESSION
# ============================================================

def predict_daily_sales(
    store_size_sqm,
    num_staff,
    footfall_count,
    promotions_active,
    avg_basket_size_kes,
    local_events_nearby,
    store_format,
    location_type,
    delivery_available,
    store_size_category,
    store_age_years,
    online_order_count,
    loyalty_program_members
):

    payload = {
        "store_size_sqm": store_size_sqm,
        "num_staff": num_staff,
        "footfall_count": footfall_count,
        "promotions_active": (
            1.0 if promotions_active == "Yes" else 0.0
        ),
        "avg_basket_size_kes": avg_basket_size_kes,
        "local_events_nearby": (
            1.0 if local_events_nearby == "Yes" else 0.0
        ),
        "store_format": store_format,
        "location_type": location_type,
        "delivery_available": delivery_available,
        "store_size_category": store_size_category,
        "store_age_years": store_age_years,
        "online_order_count": online_order_count,
        "loyalty_program_members": loyalty_program_members
    }

    return call_api(
        "/api/v1/models/daily-sales-revenue-regressor/predictions",
        payload
    )


# ============================================================
# 2. BUSINESS DISTRESS RISK CLASSIFICATION
# ============================================================

def predict_distress_risk(
    debt_to_equity_ratio,
    interest_coverage_ratio,
    current_ratio,
    working_capital_to_assets_ratio,
    net_profit_margin_pct,
    revenue_growth_pct
):

    payload = {
        "debt_to_equity_ratio": debt_to_equity_ratio,
        "interest_coverage_ratio": interest_coverage_ratio,
        "current_ratio": current_ratio,
        "working_capital_to_assets_ratio": (
            working_capital_to_assets_ratio
        ),
        "net_profit_margin_pct": net_profit_margin_pct,
        "revenue_growth_pct": revenue_growth_pct
    }

    return call_api(
        "/api/v1/models/distress-risk-classifier/predictions",
        payload
    )


# ============================================================
# 3. CUSTOMER SEGMENTATION
# ============================================================

def predict_customer_segment(
    recency_days,
    frequency_per_year,
    avg_order_value_kes,
    num_categories_purchased
):

    payload = {
        "recency_days": recency_days,
        "frequency_per_year": frequency_per_year,
        "avg_order_value_kes": avg_order_value_kes,
        "num_categories_purchased": num_categories_purchased
    }

    if all(value is None for value in payload.values()):
        return {
            "error": "Please enter at least one customer feature."
        }

    for feature, value in payload.items():
        if value is not None and value < 0:
            return {
                "error": f"{feature} cannot be negative."
            }

    return call_api(
        "/api/v1/models/customer-segmenter/predictions",
        payload
    )


# ============================================================
# 4. STREAMING SHOW RECOMMENDATIONS
# ============================================================

SHOW_CHOICES = [
    "Crime City",
    "The Detective",
    "Silent Witness Redux",
    "Laugh Track",
    "Office Antics",
    "Space Frontier",
    "Galactic Wars",
    "Time Anomaly",
    "Toon Town",
    "Little Explorers",
    "Real Life Diaries",
    "Love Island Nairobi",
    "True Crime Files",
    "Parenting Unscripted",
    "Cooking Masters"
]


def recommend_streaming_shows(watched, top_n):

    if not watched:
        return {
            "error": "Please select at least one show you have watched."
        }

    payload = {
        "watched": watched,
        "top_n": int(top_n)
    }

    return call_api(
        "/api/v1/models/streaming-recommender/predictions",
        payload
    )


# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="Business Intelligence Model Prediction System"
) as demo:

    gr.Markdown(
        """
        # Business Intelligence Model Prediction System

        Explore four machine learning models through one
        interactive application.

        Select a model tab, provide the required inputs,
        and submit them to generate a prediction.

        **Powered by a Flask REST API**
        """
    )

    # --------------------------------------------------------
    # TAB 1: REGRESSION
    # --------------------------------------------------------

    with gr.Tab("Daily Sales Revenue Prediction"):

        gr.Markdown(
            """
            ### Daily Sales Revenue Regressor

            Predict the expected daily sales revenue of a store
            using its operational and business characteristics.

            **Model:** Ridge Regression
            """
        )

        with gr.Row():
            store_size_sqm = gr.Number(
                label="Store Size (square metres)",
                value=150
            )

            num_staff = gr.Number(
                label="Number of Staff",
                value=10
            )

            footfall_count = gr.Number(
                label="Daily Footfall Count",
                value=300
            )

        with gr.Row():
            promotions_active = gr.Radio(
                ["Yes", "No"],
                label="Are Promotions Active?",
                value="Yes"
            )

            avg_basket_size_kes = gr.Number(
                label="Average Basket Size (KES)",
                value=1200
            )

            local_events_nearby = gr.Radio(
                ["Yes", "No"],
                label="Are There Local Events Nearby?",
                value="No"
            )

        with gr.Row():
            store_format = gr.Dropdown(
                choices=[
                    "Supermarket",
                    "Convenience",
                    "Hypermarket",
                    "Kiosk"
                ],
                label="Store Format",
                value="Supermarket"
            )

            location_type = gr.Dropdown(
                choices=[
                    "CBD",
                    "Mall",
                    "Residential",
                    "Street"
                ],
                label="Location Type",
                value="CBD"
            )

            delivery_available = gr.Dropdown(
                choices=["Yes", "No"],
                label="Delivery Available",
                value="Yes"
            )

        with gr.Row():
            store_size_category = gr.Dropdown(
                choices=["Small", "Medium", "Large"],
                label="Store Size Category",
                value="Medium"
            )

            store_age_years = gr.Number(
                label="Store Age (Years)",
                value=3.5
            )

        with gr.Row():
            online_order_count = gr.Number(
                label="Online Order Count",
                value=45
            )

            loyalty_program_members = gr.Number(
                label="Loyalty Programme Members",
                value=150
            )

        regression_button = gr.Button(
            "Predict Daily Sales Revenue",
            variant="primary"
        )

        regression_output = gr.JSON(
            label="Prediction Result"
        )

        regression_button.click(
            fn=predict_daily_sales,
            inputs=[
                store_size_sqm,
                num_staff,
                footfall_count,
                promotions_active,
                avg_basket_size_kes,
                local_events_nearby,
                store_format,
                location_type,
                delivery_available,
                store_size_category,
                store_age_years,
                online_order_count,
                loyalty_program_members
            ],
            outputs=regression_output
        )

    # --------------------------------------------------------
    # TAB 2: CLASSIFICATION
    # --------------------------------------------------------

    with gr.Tab("Business Distress Risk Classification"):

        gr.Markdown(
            """
            ### Business Distress Risk Classifier

            Assess the financial distress risk category of a
            business using selected financial indicators.

            **Model:** Logistic Regression
            """
        )

        with gr.Row():
            debt_to_equity_ratio = gr.Number(
                label="Debt-to-Equity Ratio",
                value=1.5
            )

            interest_coverage_ratio = gr.Number(
                label="Interest Coverage Ratio",
                value=2.0
            )

            current_ratio = gr.Number(
                label="Current Ratio",
                value=1.2
            )

        with gr.Row():
            working_capital_to_assets_ratio = gr.Number(
                label="Working Capital-to-Assets Ratio",
                value=0.4
            )

            net_profit_margin_pct = gr.Number(
                label="Net Profit Margin (%)",
                value=12.5
            )

            revenue_growth_pct = gr.Number(
                label="Revenue Growth (%)",
                value=8.0
            )

        classification_button = gr.Button(
            "Assess Business Distress Risk",
            variant="primary"
        )

        classification_output = gr.JSON(
            label="Classification Result"
        )

        classification_button.click(
            fn=predict_distress_risk,
            inputs=[
                debt_to_equity_ratio,
                interest_coverage_ratio,
                current_ratio,
                working_capital_to_assets_ratio,
                net_profit_margin_pct,
                revenue_growth_pct
            ],
            outputs=classification_output
        )

    # --------------------------------------------------------
    # TAB 3: CUSTOMER SEGMENTATION
    # --------------------------------------------------------

    with gr.Tab("Customer Segmentation"):

        gr.Markdown(
            """
            ### Customer Segmenter

            Identify the customer segment based on purchasing
            behaviour and engagement characteristics.

            **Model:** K-Means Clustering

            Leave an unknown feature blank if its value is unavailable.
            The API's preprocessing pipeline handles missing values.
            """
        )

        with gr.Row():
            recency_days = gr.Number(
                label="Recency (Days)",
                value=20
            )

            frequency_per_year = gr.Number(
                label="Purchase Frequency per Year",
                value=22
            )

        with gr.Row():
            avg_order_value_kes = gr.Number(
                label="Average Order Value (KES)",
                value=8000
            )

            num_categories_purchased = gr.Number(
                label="Number of Categories Purchased",
                value=6
            )

        segmentation_button = gr.Button(
            "Identify Customer Segment",
            variant="primary"
        )

        segmentation_output = gr.JSON(
            label="Customer Segmentation Result"
        )

        segmentation_button.click(
            fn=predict_customer_segment,
            inputs=[
                recency_days,
                frequency_per_year,
                avg_order_value_kes,
                num_categories_purchased
            ],
            outputs=segmentation_output
        )

    # --------------------------------------------------------
    # TAB 4: STREAMING RECOMMENDATIONS
    # --------------------------------------------------------

    with gr.Tab("Streaming Show Recommendations"):

        gr.Markdown(
            """
            ### Streaming Recommendation System

            Select the shows you have already watched to receive
            personalised recommendations.

            **Model:** Apriori Association Rules

            The system can also use a popularity-based fallback
            when no matching association rule is available.
            """
        )

        watched = gr.CheckboxGroup(
            choices=SHOW_CHOICES,
            label="Select Shows You Have Watched",
            value=[]
        )

        top_n = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=3,
            label="Number of Recommendations"
        )

        recommendation_button = gr.Button(
            "Recommend Shows",
            variant="primary"
        )

        recommendation_output = gr.JSON(
            label="Recommended Shows"
        )

        recommendation_button.click(
            fn=recommend_streaming_shows,
            inputs=[
                watched,
                top_n
            ],
            outputs=recommendation_output
        )

    gr.Markdown(
        """
        ---
        **Business Intelligence Model Prediction System**

        Predictions are generated through the project's Flask
        REST API. Results depend on the trained models and
        the input values supplied.
        """
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )