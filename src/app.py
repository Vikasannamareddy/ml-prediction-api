# Until now, your prediction logic exists inside your notebook.
# That's not ideal for production.
# For example, you currently have things like:
# best_gb_pipeline.predict_proba(X_test)
# inside the notebook.
# A real application shouldn't depend on opening a Jupyter notebook.
# Instead:
# Notebook
#    ↓
# Train model
#    ↓
# Save model
#    ↓
# Application
#    ↓
# Load saved model
#    ↓
# User enters data
#    ↓
# Prediction

import streamlit as st
import pandas as pd
import joblib
import json
import os
import logging


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Model Paths
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "churn_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "threshold.json"
)

# -----------------------------
# Logging Configuration
# -----------------------------

LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "logs",
    "predictions.log"
)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# Load Model
# -----------------------------

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

# -----------------------------
# Load Threshold
# -----------------------------

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

threshold = threshold_config["threshold"]


# -----------------------------
# Application
# -----------------------------

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")
if model_loaded:
    st.success("✅ ML model loaded successfully")

st.subheader("Customer Information")

with st.form("customer_form"):           #Without a form, Streamlit can rerun the application every time the user changes an input.+

    # -----------------------------
    # Numerical Features
    # -----------------------------

    st.markdown("### 📈 Customer Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col3:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

    with col4:
        cltv = st.number_input(
            "CLTV",
            min_value=0,
            value=5000,
            step=100
        )

    # -----------------------------
    # Categorical Features
    # -----------------------------

    st.markdown("### 👤 Customer & Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="gender"
        )

    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"],
            key="senior_citizen"
        )

    with col3:
        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            key="partner"

        )

    col1, col2, col3 = st.columns(3)

    with col1:
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
             key="dependents"

        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
             key="phone_service"
        )

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
            key='multiple_lines'
        )

    st.markdown("### 🌐 Internet & Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key='internet_service'
        )

    with col2:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
            key='online_security'
        )

    with col3:
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
            key='online_backup'
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            key='device_protection'
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            key='tech_support'
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
            key='streaming_tv'
        )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
        key='streaming_movies'
    )

    st.markdown("### 💳 Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key='contract'
        )

    with col2:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            key='paperless_billing'
        )

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key='payment_method'
        )

    submitted = st.form_submit_button(
        "🔮 Predict Churn"
    )

    if submitted:
        input_data = pd.DataFrame({
            "Tenure Months": [tenure_months],
            "Monthly Charges": [monthly_charges],
            "Total Charges": [total_charges],
            "CLTV": [cltv],

            "Gender": [gender],
            "Senior Citizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "Phone Service": [phone_service],
            "Multiple Lines": [multiple_lines],
            "Internet Service": [internet_service],
            "Online Security": [online_security],
            "Online Backup": [online_backup],
            "Device Protection": [device_protection],
            "Tech Support": [tech_support],
            "Streaming TV": [streaming_tv],
            "Streaming Movies": [streaming_movies],
            "Contract": [contract],
            "Paperless Billing": [paperless_billing],
            "Payment Method": [payment_method]
        })

        st.subheader("Input Data")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        validation_errors = []

        if total_charges < monthly_charges and tenure_months > 1:
            validation_errors.append(
                "Total Charges seems unusually low compared to Monthly Charges."
            )

        if validation_errors:

            for error in validation_errors:
                st.warning(error)

        else:
            # -----------------------------
            # Make Prediction
            # -----------------------------

            try:

                probability = model.predict_proba(
                    input_data
                )[0][1]

                prediction = probability >= threshold

                prediction_label = (
                    "Churn"
                    if prediction
                    else "No Churn"
                )

                logger.info(
                    "Prediction completed | probability=%.4f | threshold=%.2f | prediction=%s",
                    probability,
                    threshold,
                    int(prediction)
                )

                st.divider()

                st.subheader("🔍 Prediction Result")

                col1, col2,col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Churn Probability",
                        f"{probability:.2%}"
                    )

                with col2:
                    st.metric(
                        "Prediction Threshold",
                        f"{threshold:.0%}"
                    )

                with col3:
                    st.metric(
                        "Prediction",
                        prediction_label
                    )

                st.progress(float(probability))

                if prediction:
                    st.error(
                        "⚠️ HIGH RISK — Customer is likely to churn"
                    )
                else:
                    st.success(
                        "✅ LOW RISK — Customer is unlikely to churn"
                    )

                st.caption(
                    f"Customers with predicted churn probability ≥ "
                    f"{threshold:.0%} are classified as churn."
                )

            except Exception as e:

                logger.exception("Prediction failed")

                st.error(
                    f"Prediction failed: {str(e)}"
                )
st.info(f"Model prediction threshold: {threshold}")