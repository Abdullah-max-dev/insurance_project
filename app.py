from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🛡️")

BASE_DIR = Path(__file__).resolve().parent

# Population statistics from the data used when training the model.
SCALER_STATS = {
    "age": (39.207025, 14.044709),
    "bmi": (30.663397, 6.095908),
    "children": (1.094918, 1.205042),
}


@st.cache_resource
def load_model():
    """Load the trained insurance-cost model once per server."""
    return joblib.load(BASE_DIR / "insurance_model.pkl")


@st.cache_data
def load_columns():
    """Load model feature names in the exact training order."""
    return joblib.load(BASE_DIR / "columns.pkl")


def standardize(value: float, feature: str) -> float:
    mean, std = SCALER_STATS[feature]
    return (value - mean) / std


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def build_features(age, sex, bmi, children, smoker, region, feature_columns):
    """Create one row; all unselected one-hot fields remain zero."""
    features = {column: 0 for column in feature_columns}

    features["age"] = standardize(age, "age")
    features["is_female"] = int(sex == "Female")
    features["bmi"] = standardize(bmi, "bmi")
    features["children"] = standardize(children, "children")
    features["is_smoker"] = int(smoker == "Yes")

    # The selected region is 1; the other three regions stay 0.
    features[f"region_{region.lower()}"] = 1

    # BMI can only belong to one category; all remaining categories stay 0.
    category = bmi_category(bmi)
    features[f"bmi_category_{category}"] = 1

    # Engineered fields expected by the saved model.
    features["smoker_bmi"] = features["is_smoker"] * features["bmi"]
    features["smoker_obese"] = features["is_smoker"] * int(category == "obese")
    features["age_sq"] = features["age"] ** 2

    return pd.DataFrame([features], columns=feature_columns)


st.title("🛡️ Insurance Cost Predictor")
st.write("Enter customer details to estimate the annual insurance charge.")

try:
    model = load_model()
    columns = load_columns()
except (FileNotFoundError, ModuleNotFoundError) as error:
    st.error(f"Could not load the prediction files: {error}")
    st.stop()

with st.form("insurance_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with right:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.radio("Smoker", ["No", "Yes"], horizontal=True)
        region = st.selectbox(
            "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    submitted = st.form_submit_button("Estimate insurance cost", type="primary")

if submitted:
    model_input = build_features(age, sex, bmi, children, smoker, region, columns)
    prediction = float(model.predict(model_input)[0])

    st.success(f"Estimated annual insurance charge: **${prediction:,.2f}**")
    with st.expander("View model input"):
        st.dataframe(model_input, use_container_width=True)
