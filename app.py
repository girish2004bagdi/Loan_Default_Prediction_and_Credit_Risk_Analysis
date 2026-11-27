from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

st.set_page_config(page_title="Loan Default Risk", page_icon="🏦", layout="wide")

st.title("🏦 Loan Default Risk Prediction")
st.caption("Educational machine-learning demonstration — not financial advice or an automated lending decision.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.markdown("Enter applicant details to estimate the model's predicted probability of loan default.")
with st.form("applicant_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Applicant age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual income", min_value=1, max_value=5_000_000, value=50_000, step=1_000)
        home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        employment = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
        intent = st.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    with right:
        grade = st.selectbox("Loan grade", list("ABCDEFG"))
        amount = st.number_input("Loan amount", min_value=1, max_value=100_000, value=10_000, step=500)
        rate = st.number_input("Interest rate (%)", min_value=0.1, max_value=30.0, value=11.0, step=0.1)
        previous_default = st.selectbox("Previous default on file", ["N", "Y"])
        history = st.number_input("Credit history length (years)", min_value=1, max_value=40, value=4)

    submitted = st.form_submit_button("Estimate default risk", type="primary")

if submitted:
    if income <= 0:
        st.error("Annual income must be greater than zero.")
        st.stop()

    applicant = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": employment,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "loan_percent_income": amount / income,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": history,
        "cred_hist_to_age": history / age,
    }])

    probability = float(model.predict_proba(applicant)[0, 1])
    st.subheader("Model output")
    st.metric("Estimated default probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= 0.5:
        st.warning("The model classifies this application as higher predicted default risk at a 0.50 threshold.")
    else:
        st.success("The model classifies this application as lower predicted default risk at a 0.50 threshold.")
    st.caption("This threshold is a demonstration setting. It is not a validated lending policy.")
