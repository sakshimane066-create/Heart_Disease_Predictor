import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #e63946;
        color: white;
        font-weight: 600;
        padding: 0.6em;
        border-radius: 8px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #c92c3c;
        color: white;
    }
    .result-box {
        padding: 1.2em;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 1em;
    }
    .high-risk {
        background-color: #3a1a1a;
        border: 1px solid #e63946;
        color: #ff6b6b;
    }
    .low-risk {
        background-color: #17331f;
        border: 1px solid #2ecc71;
        color: #6bffa0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load saved model, scaler, and training columns
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_regression.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_artifacts()
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a **Logistic Regression** model trained on the "
        "Kaggle Heart Failure Prediction dataset to estimate a patient's "
        "risk of heart disease based on clinical parameters."
    )
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Scikit-learn")

# -----------------------------
# Header
# -----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write("Fill in the patient's clinical details below to estimate heart disease risk.")
st.markdown("---")

# -----------------------------
# Inputs, grouped into sections with columns
# -----------------------------
st.subheader("🧍 Basic Info")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
with col2:
    sex = st.selectbox("Sex", ["M", "F"])

st.subheader("🩺 Clinical Measurements")
col3, col4 = st.columns(2)
with col3:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
with col4:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)

st.subheader("🏃 Exercise & Symptoms")
col5, col6 = st.columns(2)
with col5:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
with col6:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# -----------------------------
# Predict
# -----------------------------
predict_clicked = st.button("🔍 Predict Risk")

if predict_clicked:
    input_dict = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0]

    if prediction == 1:
        st.markdown(
            f'<div class="result-box high-risk">⚠️ High risk of Heart Disease<br>'
            f'Confidence: {prediction_proba[1]*100:.1f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box low-risk">✅ Low risk of Heart Disease<br>'
            f'Confidence: {prediction_proba[0]*100:.1f}%</div>',
            unsafe_allow_html=True
        )

    with st.expander("See probability breakdown"):
        st.write(f"Probability of No Heart Disease: **{prediction_proba[0]*100:.1f}%**")
        st.write(f"Probability of Heart Disease: **{prediction_proba[1]*100:.1f}%**")
        st.progress(float(prediction_proba[1]))