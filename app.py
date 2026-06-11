import os
import pickle
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_model():
   path = os.path.join(os.path.dirname(__file__), "model.pkl")
   with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Feature options ───────────────────────────────────────────────────────────
JOB_OPTIONS      = ["admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown"]
MARITAL_OPTIONS  = ["divorced", "married", "single"]
EDU_OPTIONS      = ["primary", "secondary", "tertiary", "unknown"]
YESNO            = ["no", "yes"]
CONTACT_OPTIONS  = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS    = ["jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.markdown("Fill in the client and campaign details, then click **Predict**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Client Information")
    age      = st.slider("Age", 18, 95, 40)
    job      = st.selectbox("Job", JOB_OPTIONS, index=JOB_OPTIONS.index("management"))
    marital  = st.selectbox("Marital Status", MARITAL_OPTIONS, index=1)
    education= st.selectbox("Education", EDU_OPTIONS, index=1)
    balance  = st.number_input("Account Balance (€)", value=1000, step=100)
    default  = st.radio("Has Credit Default?", YESNO, horizontal=True)
    housing  = st.radio("Has Housing Loan?",   YESNO, index=1, horizontal=True)
    loan     = st.radio("Has Personal Loan?",  YESNO, horizontal=True)

with col2:
    st.subheader("📞 Campaign Details")
    contact  = st.selectbox("Contact Type", CONTACT_OPTIONS)
    month    = st.selectbox("Last Contact Month", MONTH_OPTIONS, index=MONTH_OPTIONS.index("may"))
    day      = st.slider("Last Contact Day of Month", 1, 31, 15)
    duration = st.slider("Last Call Duration (seconds)", 0, 5000, 200, step=10)
    campaign = st.slider("Contacts During This Campaign", 1, 50, 2)
    pdays    = st.number_input("Days Since Previous Campaign (-1 = never)", value=-1, step=1)
    previous = st.slider("Contacts Before This Campaign", 0, 30, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", POUTCOME_OPTIONS, index=3)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Subscription", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
    }])

    prediction   = model.predict(input_df)[0]
    probabilities= model.predict_proba(input_df)[0]
    classes      = model.classes_
    prob_dict    = dict(zip(classes, probabilities))

    st.subheader("📊 Result")

    if prediction == "yes":
        st.success("✅ **YES — Client is likely to subscribe to a term deposit**")
    else:
        st.error("❌ **NO — Client is unlikely to subscribe to a term deposit**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction",        prediction.upper())
    c2.metric("Probability YES",   f"{prob_dict.get('yes', 0):.1%}")
    c3.metric("Probability NO",    f"{prob_dict.get('no',  0):.1%}")

    st.progress(float(prob_dict.get('yes', 0)), text=f"Confidence for YES: {prob_dict.get('yes', 0):.1%}")

st.caption("Model: Decision Tree Classifier · Preprocessor: MinMaxScaler + OneHotEncoder")