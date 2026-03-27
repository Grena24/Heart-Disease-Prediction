import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #e74c3c33;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p { color: #adb5bd; font-size: 1.1rem; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #e74c3c; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #adb5bd; font-size: 0.85rem; margin: 0; }
    .result-danger {
        background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.1));
        border: 2px solid #e74c3c;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(39,174,96,0.1));
        border: 2px solid #2ecc71;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-title    { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .result-subtitle { font-size: 1rem; color: #adb5bd; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e74c3c;
        border-bottom: 2px solid #e74c3c33;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .ai-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(52,152,219,0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.8;
        color: #e0e0e0;
    }
    .patient-badge {
        background: linear-gradient(90deg, rgba(231,76,60,0.2), rgba(255,107,107,0.1));
        border: 1px solid #e74c3c55;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ff6b6b;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(231,76,60,0.6);
        transform: translateY(-2px);
    }
    .disclaimer {
        background: rgba(255,193,7,0.1);
        border: 1px solid rgba(255,193,7,0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #ffc107;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 1rem;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #adb5bd;
        margin-bottom: 0.3rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("cleaned_heart.csv")

    for col in ['AgeGroup', 'BP_Category']:
        if col in df.columns:
            df = df.drop(columns=[col])

    cat_features = ['Gender', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    le = LabelEncoder()
    encoders = {}
    for col in cat_features:
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, encoders

model, encoders = load_model()


# ─────────────────────────────────────────────────────────────
# AI RECOMMENDATION — Groq (llama3)
# ─────────────────────────────────────────────────────────────
def get_ai_recommendation(patient_name, age, gender, chest_pain, bp,
                           cholesterol, max_hr, ex_angina, oldpeak,
                           st_slope, fasting_bs, prediction, probability):

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise ValueError("GROQ_API_KEY not found. Go to Streamlit Cloud → App Settings → Secrets and add:\nGROQ_API_KEY = 'gsk_your_key_here'")

    client = Groq(api_key=api_key)

    risk_level = "HIGH RISK — Heart Disease Detected" if prediction == 1 else "LOW RISK — No Heart Disease"
    disease_prob = f"{probability[1] * 100:.1f}%"

    prompt = f"""You are a helpful medical AI assistant. A patient has just received their heart disease prediction result.
Provide a clear, friendly, and helpful health recommendation report.

Patient Details:
- Name              : {patient_name}
- Age               : {age} years
- Gender            : {gender}
- Chest Pain Type   : {chest_pain}
- Resting BP        : {bp} mmHg
- Cholesterol       : {cholesterol} mg/dl
- Max Heart Rate    : {max_hr}
- Exercise Angina   : {ex_angina}
- Oldpeak           : {oldpeak}
- ST Slope          : {st_slope}
- Fasting Blood Sugar > 120 mg/dl: {fasting_bs}

Prediction Result   : {risk_level}
Disease Probability : {disease_prob}

Please provide a personalised report for {patient_name} with:
1. A short personalised greeting using their name
2. Brief explanation of what the result means for them specifically
3. Top 5 specific health recommendations based on their individual data values above
4. Concrete lifestyle changes they should make
5. Clear guidance on when and how urgently they should see a doctor

Keep the tone warm, clear, and easy to understand. Use bullet points for recommendations.
Do NOT use markdown headers with # symbols.
Keep the response concise (under 400 words).

IMPORTANT: Always end with a reminder that this is an AI screening tool and they must consult a qualified doctor."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.7
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ About This App")
    st.markdown("Predicts heart disease risk using a **Random Forest** ML model + **AI recommendations** powered by Groq.")
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='metric-card'><h3>87.5%</h3><p>Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'><h3>0.93</h3><p>ROC-AUC</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🏥 Dataset Info")
    st.markdown("- 📋 **918** patient records\n- 🔬 **11** clinical features\n- 🎯 Binary classification")
    st.markdown("---")
    st.markdown("### 📌 Chest Pain Types")
    st.markdown("- **ASY** → Asymptomatic\n- **ATA** → Atypical Angina\n- **NAP** → Non-Anginal Pain\n- **TA** → Typical Angina")
    st.markdown("---")
    st.markdown("""<div class='disclaimer'>⚕️ For educational use only.<br>Not a substitute for medical advice.</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>❤️ Heart Failure Prediction</h1>
    <p>Enter patient clinical data below to assess heart disease risk using AI</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# PATIENT NAME INPUT
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>👤 Patient Details</div>", unsafe_allow_html=True)
patient_name = st.text_input("Patient Full Name", placeholder="e.g. Rahul Sharma")
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# CLINICAL INPUT FORM
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🧑‍⚕️ Patient Clinical Information</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    age        = st.slider("Age (years)", 18, 100, 50)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    st.markdown("**🫀 Heart Metrics**")
    chest     = st.selectbox("Chest Pain Type", ["ASY - Asymptomatic",
                                                   "ATA - Atypical Angina",
                                                   "NAP - Non-Anginal Pain",
                                                   "TA  - Typical Angina"])
    max_hr    = st.slider("Max Heart Rate", 50, 250, 140)
    ex_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak   = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, step=0.1)

with col3:
    st.markdown("**🩺 Clinical Measurements**")
    resting_bp  = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
    resting_ecg = st.selectbox("Resting ECG Result", ["NORMAL", "ST", "LVH"])
    st_slope    = st.selectbox("ST Slope", ["UP", "FLAT", "DOWN"])

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# ENCODE INPUTS
# ─────────────────────────────────────────────────────────────
gender_enc  = encoders['Gender']['M'] if gender == "Male" else encoders['Gender']['F']
chest_enc   = encoders['ChestPainType'][chest.split(" ")[0].strip()]
ecg_enc     = encoders['RestingECG'][resting_ecg]
angina_enc  = encoders['ExerciseAngina']['Y'] if ex_angina == "Yes" else encoders['ExerciseAngina']['N']
slope_enc   = encoders['ST_Slope'][st_slope]
fasting_enc = 1 if fasting_bs == "Yes" else 0

input_data = pd.DataFrame([{
    'Age'                 : age,
    'Gender'              : gender_enc,
    'ChestPainType'       : chest_enc,
    'RestingBloodPressure': float(resting_bp),
    'Cholesterol'         : float(cholesterol),
    'FastingBloodSugar'   : fasting_enc,
    'RestingECG'          : ecg_enc,
    'MaxHR'               : max_hr,
    'ExerciseAngina'      : angina_enc,
    'Oldpeak'             : float(oldpeak),
    'ST_Slope'            : slope_enc
}])


# ─────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict Heart Disease Risk", use_container_width=True)

if predict_btn:

    if not patient_name.strip():
        st.warning("⚠️  Please enter the patient's name before predicting.")
        st.stop()

    pred  = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.markdown("---")

    st.markdown(f"""
    <div class='patient-badge'>
        👤 Patient : {patient_name}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 Prediction Result</div>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class='result-danger'>
                <div class='result-title'>⚠️ HIGH RISK</div>
                <div style='font-size:1.3rem;color:#e74c3c;font-weight:700;'>Heart Disease Detected</div>
                <div class='result-subtitle'>The model predicts a high probability of heart disease for <b>{patient_name}</b>.<br>
                Please consult a cardiologist immediately.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
                <div class='result-title'>✅ LOW RISK</div>
                <div style='font-size:1.3rem;color:#2ecc71;font-weight:700;'>No Heart Disease Detected</div>
                <div class='result-subtitle'>The model predicts a low probability of heart disease for <b>{patient_name}</b>.<br>
                Maintain a healthy lifestyle and regular checkups.</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 📊 Probability")
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color:#2ecc71;'>{proba[0]*100:.1f}%</h3>
            <p>✅ No Disease</p>
        </div>
        <div class='metric-card'>
            <h3 style='color:#e74c3c;'>{proba[1]*100:.1f}%</h3>
            <p>⚠️ Heart Disease</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🎯 Risk Gauge")
    st.markdown(f"""
    <div class='prob-label'>
        <span>✅ No Disease ({proba[0]*100:.1f}%)</span>
        <span>⚠️ Disease ({proba[1]*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(proba[1]))

    st.markdown("---")

    st.markdown("<div class='section-header'>📝 Patient Summary</div>", unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        'Feature' : ['Patient Name', 'Age', 'Gender', 'Chest Pain', 'Resting BP',
                     'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG',
                     'Max Heart Rate', 'Exercise Angina', 'Oldpeak', 'ST Slope'],
        'Value'   : [patient_name, age, gender, chest.split("-")[1].strip(),
                     f"{resting_bp} mmHg", f"{cholesterol} mg/dl", fasting_bs,
                     resting_ecg, max_hr, ex_angina, oldpeak, st_slope],
        'Status'  : ['—',
                     '🟡 Risk Factor' if age > 55 else '🟢 Normal',
                     '🟡 Higher Risk' if gender == 'Male' else '🟢 Normal',
                     '🔴 High Risk' if 'ASY' in chest else '🟢 Normal',
                     '🔴 High' if resting_bp > 140 else '🟢 Normal',
                     '🔴 High' if cholesterol > 240 else '🟢 Normal',
                     '🔴 Yes' if fasting_bs == 'Yes' else '🟢 No',
                     '🟢 Normal' if resting_ecg == 'NORMAL' else '🔴 Abnormal',
                     '🔴 Low' if max_hr < 100 else '🟢 Normal',
                     '🔴 Yes' if ex_angina == 'Yes' else '🟢 No',
                     '🔴 High' if oldpeak > 2 else '🟢 Normal',
                     '🔴 Risk' if st_slope in ['FLAT', 'DOWN'] else '🟢 Normal']
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── AI RECOMMENDATIONS (Groq) ──
    st.markdown("<div class='section-header'>🤖 AI Health Recommendations</div>", unsafe_allow_html=True)

    with st.spinner(f"🧠 Generating personalised recommendations for {patient_name}..."):
        try:
            ai_response = get_ai_recommendation(
                patient_name, age, gender,
                chest.split("-")[1].strip(),
                resting_bp, cholesterol, max_hr,
                ex_angina, oldpeak, st_slope,
                fasting_bs, pred, proba
            )
            st.markdown(f"""
            <div class='ai-box'>
                {ai_response.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        except ValueError as ve:
            st.error(f"⚠️ {ve}")
        except Exception as e:
            st.error(f"⚠️ AI recommendations unavailable: {str(e)}")
