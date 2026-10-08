import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load trained model
# -----------------------------
model = joblib.load("customer_churn_model.pkl")


# -----------------------------
# 2. App title
# -----------------------------
st.title("Customer Churn Prediction")

st.write("Upload a CSV file to predict customer churn.")


# -----------------------------
# 3. Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


# -----------------------------
# 4. Prediction
# -----------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Check required columns
    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        st.error("Some required columns are missing:")
        st.write(missing_columns)

    else:

        # Keep original data for final output
        result = data.copy()

        # Remove customerID before prediction
        prediction_data = data.drop("customerID", axis=1)

        # Convert TotalCharges to numeric
        prediction_data["TotalCharges"] = pd.to_numeric(
            prediction_data["TotalCharges"],
            errors="coerce"
        )

        # One-hot encoding
        prediction_data = pd.get_dummies(
            prediction_data,
            drop_first=True,
            dtype=int
        )

        # Match training columns
        prediction_data = prediction_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(prediction_data)

        # Convert 0/1 to No/Yes
        result["Churn_Prediction"] = prediction

        result["Churn_Prediction"] = result[
            "Churn_Prediction"
        ].map({
            0: "No",
            1: "Yes"
        })

        # Show result
        st.subheader("Prediction Result")

        st.dataframe(result.head())

        # Convert to CSV
        csv = result.to_csv(index=False).encode("utf-8")

        # Download button
        st.download_button(
            label="Download Prediction CSV",
            data=csv,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )