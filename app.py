import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from groq import Groq
import warnings
import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
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

    /* ── AI BOX ── */
    .ai-box {
        background: linear-gradient(135deg, rgba(15,12,41,0.9), rgba(26,26,46,0.9));
        border: 1px solid rgba(52,152,219,0.5);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        line-height: 1.9;
        color: #e8eaf0;
        font-size: 1rem;
        box-shadow: 0 8px 32px rgba(52,152,219,0.15);
    }
    .ai-greeting {
        font-size: 1.15rem;
        font-weight: 700;
        color: #74b9ff;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(52,152,219,0.3);
    }
    .ai-section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffd93d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.4rem 0 0.5rem 0;
    }
    .ai-point {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin: 0.45rem 0;
        padding: 0.5rem 0.8rem;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        border-left: 3px solid rgba(52,152,219,0.4);
        color: #dfe6e9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .ai-point-emoji {
        font-size: 1.1rem;
        flex-shrink: 0;
        margin-top: 1px;
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
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(231,76,60,0.6);
        transform: translateY(-2px);
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #adb5bd;
        margin-bottom: 0.3rem;
    }

    /* download button style */
    .download-btn > a {
        display: inline-block;
        background: linear-gradient(90deg, #2980b9, #3498db);
        color: white !important;
        text-decoration: none;
        padding: 0.7rem 1.8rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(52,152,219,0.4);
        transition: all 0.2s;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NORMAL RANGES reference
# ─────────────────────────────────────────────────────────────
NORMAL_RANGES = {
    'Age'                 : {'normal': '18–55 years',         'flag': lambda v: v > 55},
    'Resting BP'          : {'normal': '90–120 mmHg',         'flag': lambda v: v > 120},
    'Cholesterol'         : {'normal': '< 200 mg/dl',         'flag': lambda v: v >= 200},
    'Max Heart Rate'      : {'normal': '100–170 bpm',         'flag': lambda v: v < 100},
    'Oldpeak'             : {'normal': '0.0–1.0',             'flag': lambda v: v > 1.0},
    'Fasting Blood Sugar' : {'normal': '< 120 mg/dl (No)',    'flag': lambda v: v == 'Yes'},
    'Resting ECG'         : {'normal': 'NORMAL',              'flag': lambda v: v != 'NORMAL'},
    'Chest Pain'          : {'normal': 'ATA / NAP / TA',      'flag': lambda v: v == 'Asymptomatic'},
    'Exercise Angina'     : {'normal': 'No',                  'flag': lambda v: v == 'Yes'},
    'ST Slope'            : {'normal': 'UP',                  'flag': lambda v: v in ['FLAT', 'DOWN']},
}


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
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, encoders

model, encoders = load_model()


# ─────────────────────────────────────────────────────────────
# AI RECOMMENDATION — Groq
# ─────────────────────────────────────────────────────────────
def get_ai_recommendation(patient_name, age, gender, chest_pain, bp,
                           cholesterol, max_hr, ex_angina, oldpeak,
                           st_slope, fasting_bs, prediction, probability):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise ValueError("GROQ_API_KEY not found. Add it in Streamlit Cloud → App Settings → Secrets.")

    client = Groq(api_key=api_key)
    risk_level   = "HIGH RISK — Heart Disease Detected" if prediction == 1 else "LOW RISK — No Heart Disease"
    disease_prob = f"{probability[1]*100:.1f}%"

    prompt = f"""You are a medical health advisor. Give a personalised lifestyle recommendation report for this patient.

Patient: {patient_name}, {age} years, {gender}
Results: {risk_level} | Probability: {disease_prob}
Clinical Data:
- Chest Pain: {chest_pain}  | Resting BP: {bp} mmHg  | Cholesterol: {cholesterol} mg/dl
- Max Heart Rate: {max_hr}  | Exercise Angina: {ex_angina}  | Oldpeak: {oldpeak}
- ST Slope: {st_slope}  | Fasting Blood Sugar > 120: {fasting_bs}

Respond EXACTLY in this format (keep the section labels exactly as written):

GREETING:
[One warm sentence greeting to {patient_name} mentioning their result]

DIET:
[3-4 bullet points starting with * about specific diet changes based on their BP/cholesterol/sugar values]

EXERCISE:
[3 bullet points starting with * about physical activity suited to their age and heart rate]

HABITS:
[3 bullet points starting with * about daily habits like sleep, stress, smoking, alcohol]

AVOID:
[3 bullet points starting with * about things they must avoid given their data]

MONITOR:
[2 bullet points starting with * about what to track and when to see a doctor]

Rules: Be specific to their numbers. No AI/tool mentions. No # headers. No markdown bold (**). Keep each bullet point concise (1-2 lines max)."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.65
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# PARSE AI RESPONSE into sections
# ─────────────────────────────────────────────────────────────
def parse_ai_response(text):
    sections = {
        'GREETING': '', 'DIET': [], 'EXERCISE': [],
        'HABITS': [], 'AVOID': [], 'MONITOR': []
    }
    current = None
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        upper = line.upper().rstrip(':')
        if upper in sections:
            current = upper
        elif current == 'GREETING':
            sections['GREETING'] += line + ' '
        elif current and isinstance(sections[current], list):
            clean = line.lstrip('*-•').strip()
            if clean:
                sections[current].append(clean)
    return sections


# ─────────────────────────────────────────────────────────────
# RENDER STYLED AI BOX
# ─────────────────────────────────────────────────────────────
SECTION_META = {
    'DIET'    : ('🥗', 'Diet & Nutrition'),
    'EXERCISE': ('🏃', 'Exercise & Activity'),
    'HABITS'  : ('🌙', 'Daily Habits'),
    'AVOID'   : ('🚫', 'Things to Avoid'),
    'MONITOR' : ('📋', 'Monitor & Follow-up'),
}
POINT_EMOJIS = {
    'DIET'    : ['🥦','🥗','🫐','🥑','🍎'],
    'EXERCISE': ['🚶','🏊','🧘','🚴','💪'],
    'HABITS'  : ['😴','🧘','🚭','💧','🌿'],
    'AVOID'   : ['🚫','⛔','❌','🙅','🔴'],
    'MONITOR' : ['📊','🩺','📅','💊','🔬'],
}

def render_ai_box(sections):
    html = f"<div class='ai-box'>"
    if sections['GREETING']:
        html += f"<div class='ai-greeting'>👋 {sections['GREETING'].strip()}</div>"
    for key, (icon, title) in SECTION_META.items():
        points = sections.get(key, [])
        if not points:
            continue
        html += f"<div class='ai-section-title'>{icon} {title}</div>"
        emojis = POINT_EMOJIS[key]
        for i, pt in enumerate(points):
            em = emojis[i % len(emojis)]
            html += f"<div class='ai-point'><span class='ai-point-emoji'>{em}</span><span>{pt}</span></div>"
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(patient_name, age, gender, chest_pain, resting_bp,
                         cholesterol, fasting_bs, resting_ecg, max_hr,
                         ex_angina, oldpeak, st_slope, pred, proba, ai_sections):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )

    # ── Colours ──
    NAVY      = colors.HexColor('#0D1B3E')
    NAVY_MID  = colors.HexColor('#1A2F5E')
    ACCENT    = colors.HexColor('#C0392B')
    GREEN     = colors.HexColor('#1E8449')
    RED       = colors.HexColor('#C0392B')
    AMBER     = colors.HexColor('#D35400')
    WHITE     = colors.white
    OFF_WHITE = colors.HexColor('#F8F9FA')
    LGRAY     = colors.HexColor('#EDF0F2')
    MGRAY     = colors.HexColor('#AEB6BF')
    DGRAY     = colors.HexColor('#5D6D7E')
    ROW_ALT   = colors.HexColor('#F2F6FB')
    HDR_BG    = colors.HexColor('#1A2F5E')
    ABNORM_BG = colors.HexColor('#FDF2F2')
    ABNORM_BD = colors.HexColor('#E74C3C')
    NORM_BG   = colors.HexColor('#F2FBF5')
    RISK_BG   = colors.HexColor('#FEF0F0') if pred == 1 else colors.HexColor('#F0FEF4')
    RISK_BD   = RED if pred == 1 else GREEN
    RISK_TXT  = RED if pred == 1 else GREEN

    W = 17.4 * cm   # usable width

    # ── Para styles ──
    hdr_title = ParagraphStyle('HdrTitle', fontName='Helvetica-Bold',
        fontSize=20, textColor=WHITE, alignment=TA_LEFT, leading=24)
    hdr_sub   = ParagraphStyle('HdrSub', fontName='Helvetica',
        fontSize=9, textColor=colors.HexColor('#BDC3C7'), alignment=TA_LEFT, leading=13)
    hdr_right = ParagraphStyle('HdrRight', fontName='Helvetica',
        fontSize=9, textColor=colors.HexColor('#BDC3C7'), alignment=TA_RIGHT, leading=14)

    sec_hdr   = ParagraphStyle('SecHdr', fontName='Helvetica-Bold',
        fontSize=11, textColor=WHITE, alignment=TA_LEFT, leading=14)

    lbl       = ParagraphStyle('Lbl', fontName='Helvetica-Bold',
        fontSize=9, textColor=DGRAY, leading=13)
    val       = ParagraphStyle('Val', fontName='Helvetica',
        fontSize=10, textColor=NAVY, leading=13)

    col_hdr   = ParagraphStyle('ColHdr', fontName='Helvetica-Bold',
        fontSize=9.5, textColor=WHITE, alignment=TA_CENTER, leading=13)
    col_hdr_l = ParagraphStyle('ColHdrL', fontName='Helvetica-Bold',
        fontSize=9.5, textColor=WHITE, alignment=TA_LEFT, leading=13)

    cell_param= ParagraphStyle('CellParam', fontName='Helvetica-Bold',
        fontSize=9.5, textColor=NAVY_MID, alignment=TA_LEFT, leading=13)
    cell_norm = ParagraphStyle('CellNorm', fontName='Helvetica',
        fontSize=9.5, textColor=DGRAY, alignment=TA_CENTER, leading=13)
    cell_ok   = ParagraphStyle('CellOK', fontName='Helvetica',
        fontSize=9.5, textColor=colors.HexColor('#1A5276'), alignment=TA_CENTER, leading=13)
    cell_bad  = ParagraphStyle('CellBad', fontName='Helvetica-Bold',
        fontSize=9.5, textColor=RED, alignment=TA_CENTER, leading=13)
    status_ok = ParagraphStyle('StOK', fontName='Helvetica-Bold',
        fontSize=9, textColor=GREEN, alignment=TA_CENTER, leading=13)
    status_bad= ParagraphStyle('StBad', fontName='Helvetica-Bold',
        fontSize=9, textColor=RED, alignment=TA_CENTER, leading=13)

    risk_main = ParagraphStyle('RiskMain', fontName='Helvetica-Bold',
        fontSize=15, textColor=RISK_TXT, alignment=TA_CENTER, leading=20)
    risk_prob = ParagraphStyle('RiskProb', fontName='Helvetica',
        fontSize=10, textColor=DGRAY, alignment=TA_CENTER, leading=14)

    story = []

    # ══════════════════════════════════════════
    # 1. HEADER
    # ══════════════════════════════════════════
    now = datetime.datetime.now()
    left_col = [
        Paragraph("CARDIAC HEALTH REPORT", hdr_title),
        Spacer(1, 3),
        Paragraph("Heart Disease Risk Assessment  |  Powered by Random Forest ML", hdr_sub),
    ]
    right_col = [
        Paragraph(f"Report Date:  {now.strftime('%d %B %Y')}", hdr_right),
        Paragraph(f"Report Time:  {now.strftime('%I:%M %p')}", hdr_right),
        Paragraph(f"Report ID:    RPT-{now.strftime('%Y%m%d%H%M')}", hdr_right),
    ]
    hdr_tbl = Table(
        [[left_col, right_col]],
        colWidths=[11*cm, 6.4*cm]
    )
    hdr_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), NAVY),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 16),
        ('LEFTPADDING',   (0,0), (0,-1),  18),
        ('RIGHTPADDING',  (1,0), (-1,-1), 18),
        ('LINEBELOW',     (0,0), (-1,-1), 3, ACCENT),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 0.35*cm))

    # ══════════════════════════════════════════
    # 2. SECTION LABEL helper
    # ══════════════════════════════════════════
    def section_label(text):
        tbl = Table([[Paragraph(text, sec_hdr)]], colWidths=[W])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), HDR_BG),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('LINEBELOW',     (0,0), (-1,-1), 2, ACCENT),
        ]))
        return tbl

    # ══════════════════════════════════════════
    # 3. PATIENT INFORMATION
    # ══════════════════════════════════════════
    story.append(section_label("  PATIENT INFORMATION"))
    story.append(Spacer(1, 0.2*cm))

    pi_data = [
        [Paragraph("Patient Name", lbl),  Paragraph(patient_name, val),
         Paragraph("Age",          lbl),  Paragraph(f"{age} years", val)],
        [Paragraph("Gender",       lbl),  Paragraph(gender, val),
         Paragraph("Report Date",  lbl),  Paragraph(now.strftime('%d/%m/%Y'), val)],
    ]
    pi_tbl = Table(pi_data, colWidths=[3*cm, 5.7*cm, 3*cm, 5.7*cm])
    pi_tbl.setStyle(TableStyle([
        ('ROWBACKGROUNDS',  (0,0), (-1,-1), [OFF_WHITE, LGRAY]),
        ('LINEBELOW',       (0,0), (-1,-1), 0.4, MGRAY),
        ('TOPPADDING',      (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',   (0,0), (-1,-1), 7),
        ('LEFTPADDING',     (0,0), (-1,-1), 10),
        ('BOX',             (0,0), (-1,-1), 0.8, MGRAY),
    ]))
    story.append(pi_tbl)
    story.append(Spacer(1, 0.35*cm))

    # ══════════════════════════════════════════
    # 4. PREDICTION RESULT
    # ══════════════════════════════════════════
    story.append(section_label("  PREDICTION RESULT"))
    story.append(Spacer(1, 0.2*cm))

    risk_label = "HIGH RISK  —  Heart Disease Detected" if pred == 1 else "LOW RISK  —  No Heart Disease Detected"
    risk_icon  = "ALERT" if pred == 1 else "CLEAR"

    r1 = Paragraph(risk_label, risk_main)
    r2 = Paragraph(
        f"Disease Probability: <b>{proba[1]*100:.1f}%</b>     |     "
        f"No Disease Probability: <b>{proba[0]*100:.1f}%</b>",
        risk_prob
    )
    res_tbl = Table([[r1], [r2]], colWidths=[W])
    res_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), RISK_BG),
        ('BOX',           (0,0), (-1,-1), 1.5, RISK_BD),
        ('LINEABOVE',     (0,0), (-1,0),  3,   RISK_BD),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1, 0.35*cm))

    # ══════════════════════════════════════════
    # 5. CLINICAL VALUES TABLE
    # ══════════════════════════════════════════
    story.append(section_label("  CLINICAL VALUES  —  PATIENT RESULT vs NORMAL RANGE"))
    story.append(Spacer(1, 0.2*cm))

    # Column headers
    col_headers = [
        Paragraph("PARAMETER",     col_hdr_l),
        Paragraph("PATIENT VALUE", col_hdr),
        Paragraph("NORMAL RANGE",  col_hdr),
        Paragraph("STATUS",        col_hdr),
    ]

    rows_raw = [
        ("Age",                  f"{age} years",         "18 – 55 years",       age > 55),
        ("Resting Blood Pressure", f"{resting_bp} mmHg", "90 – 120 mmHg",       resting_bp > 120),
        ("Cholesterol",          f"{cholesterol} mg/dl",  "< 200 mg/dl",         cholesterol >= 200),
        ("Max Heart Rate",       f"{max_hr} bpm",         "100 – 170 bpm",       max_hr < 100 or max_hr > 170),
        ("Oldpeak (ST Depression)", f"{oldpeak}",         "0.0 – 1.0",           oldpeak > 1.0),
        ("Fasting Blood Sugar",  fasting_bs,              "< 120 mg/dl  (No)",   fasting_bs == "Yes"),
        ("Resting ECG",          resting_ecg,             "NORMAL",              resting_ecg != "NORMAL"),
        ("Chest Pain Type",      chest_pain,              "ATA / NAP / TA",      chest_pain == "Asymptomatic"),
        ("Exercise Induced Angina", ex_angina,            "No",                  ex_angina == "Yes"),
        ("ST Slope",             st_slope,                "UP",                  st_slope in ["FLAT","DOWN"]),
    ]

    tbl_data = [col_headers]
    row_cmds = []

    for i, (param, pval, nrange, is_bad) in enumerate(rows_raw):
        v_style = cell_bad  if is_bad else cell_ok
        s_style = status_bad if is_bad else status_ok
        s_text  = "ABNORMAL" if is_bad else "NORMAL"

        tbl_data.append([
            Paragraph(param,   cell_param),
            Paragraph(pval,    v_style),
            Paragraph(nrange,  cell_norm),
            Paragraph(s_text,  s_style),
        ])
        ri = i + 1
        bg = ABNORM_BG if is_bad else (ROW_ALT if i % 2 == 0 else OFF_WHITE)
        row_cmds.append(("BACKGROUND", (0, ri), (-1, ri), bg))
        if is_bad:
            row_cmds.append(("LINEBEFORE",  (0, ri), (0,  ri), 3, RED))
        else:
            row_cmds.append(("LINEBEFORE",  (0, ri), (0,  ri), 3, GREEN))

    clin_tbl = Table(tbl_data, colWidths=[5.2*cm, 3.6*cm, 4.6*cm, 4*cm])
    clin_tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0,0), (-1,0),  HDR_BG),
        ("TOPPADDING",    (0,0), (-1,0),  8),
        ("BOTTOMPADDING", (0,0), (-1,0),  8),
        ("LEFTPADDING",   (0,0), (0,0),   12),
        # Data rows
        ("TOPPADDING",    (0,1), (-1,-1), 7),
        ("BOTTOMPADDING", (0,1), (-1,-1), 7),
        ("LEFTPADDING",   (0,1), (-1,-1), 10),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,0), (0,-1),  "LEFT"),
        # Grid
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, MGRAY),
        ("BOX",           (0,0), (-1,-1), 0.8, MGRAY),
    ] + row_cmds))
    story.append(clin_tbl)
    story.append(Spacer(1, 0.3*cm))

    # ══════════════════════════════════════════
    # 6. LEGEND
    # ══════════════════════════════════════════
    legend_data = [[
        Paragraph("<font color='#1E8449'><b>  NORMAL</b></font>  Value is within healthy range", 
                  ParagraphStyle("lg", fontName="Helvetica", fontSize=8.5, textColor=DGRAY)),
        Paragraph("<font color='#C0392B'><b>  ABNORMAL</b></font>  Value is outside healthy range — requires attention",
                  ParagraphStyle("lg2", fontName="Helvetica", fontSize=8.5, textColor=DGRAY)),
    ]]
    leg_tbl = Table(legend_data, colWidths=[W/2, W/2])
    leg_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LGRAY),
        ("BOX",           (0,0), (-1,-1), 0.5, MGRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(leg_tbl)

    doc.build(story)
    buffer.seek(0)
    return buffer


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
    chest_display = chest.split("-")[1].strip()

    st.markdown("---")

    st.markdown(f"<div class='patient-badge'>👤 Patient : {patient_name}</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📋 Prediction Result</div>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class='result-danger'>
                <div class='result-title'>⚠️ HIGH RISK</div>
                <div style='font-size:1.3rem;color:#e74c3c;font-weight:700;'>Heart Disease Detected</div>
                <div class='result-subtitle'>High probability of heart disease for <b>{patient_name}</b>.<br>
                Please consult a cardiologist immediately.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
                <div class='result-title'>✅ LOW RISK</div>
                <div style='font-size:1.3rem;color:#2ecc71;font-weight:700;'>No Heart Disease Detected</div>
                <div class='result-subtitle'>Low probability of heart disease for <b>{patient_name}</b>.<br>
                Maintain a healthy lifestyle and regular checkups.</div>
            </div>""", unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 📊 Probability")
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color:#2ecc71;'>{proba[0]*100:.1f}%</h3><p>✅ No Disease</p>
        </div>
        <div class='metric-card'>
            <h3 style='color:#e74c3c;'>{proba[1]*100:.1f}%</h3><p>⚠️ Heart Disease</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎯 Risk Gauge")
    st.markdown(f"""
    <div class='prob-label'>
        <span>✅ No Disease ({proba[0]*100:.1f}%)</span>
        <span>⚠️ Disease ({proba[1]*100:.1f}%)</span>
    </div>""", unsafe_allow_html=True)
    st.progress(float(proba[1]))

    st.markdown("---")

    # ── PATIENT SUMMARY TABLE ──
    st.markdown("<div class='section-header'>📝 Patient Summary</div>", unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        'Parameter'    : ['Age', 'Gender', 'Chest Pain', 'Resting BP', 'Cholesterol',
                          'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                          'Exercise Angina', 'Oldpeak', 'ST Slope'],
        'Patient Value': [f"{age} yrs", gender, chest_display, f"{resting_bp} mmHg",
                          f"{cholesterol} mg/dl", fasting_bs, resting_ecg,
                          f"{max_hr} bpm", ex_angina, oldpeak, st_slope],
        'Normal Range' : ['18–55 years', '—', 'ATA / NAP / TA', '90–120 mmHg',
                          '< 200 mg/dl', '< 120 mg/dl (No)', 'NORMAL',
                          '100–170 bpm', 'No', '0.0–1.0', 'UP'],
        'Status'       : [
            '🔴 Risk Factor' if age > 55         else '🟢 Normal',
            '—',
            '🔴 High Risk'   if 'ASY' in chest   else '🟢 Normal',
            '🔴 High BP'     if resting_bp > 120  else '🟢 Normal',
            '🔴 High'        if cholesterol >= 200 else '🟢 Normal',
            '🔴 Elevated'    if fasting_bs == 'Yes' else '🟢 Normal',
            '🟢 Normal'      if resting_ecg == 'NORMAL' else '🔴 Abnormal',
            '🔴 Low'         if max_hr < 100      else '🟢 Normal',
            '🔴 Present'     if ex_angina == 'Yes' else '🟢 Normal',
            '🔴 High'        if oldpeak > 1.0     else '🟢 Normal',
            '🔴 Risk'        if st_slope in ['FLAT','DOWN'] else '🟢 Normal',
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── AI RECOMMENDATIONS ──
    st.markdown("<div class='section-header'>🤖 AI Health Recommendations</div>", unsafe_allow_html=True)

    ai_sections = {}
    with st.spinner(f"🧠 Generating personalised recommendations for {patient_name}..."):
        try:
            raw_ai = get_ai_recommendation(
                patient_name, age, gender, chest_display,
                resting_bp, cholesterol, max_hr,
                ex_angina, oldpeak, st_slope,
                fasting_bs, pred, proba
            )
            ai_sections = parse_ai_response(raw_ai)
            st.markdown(render_ai_box(ai_sections), unsafe_allow_html=True)

        except ValueError as ve:
            st.error(f"⚠️ {ve}")
        except Exception as e:
            st.error(f"⚠️ AI recommendations unavailable: {str(e)}")

    st.markdown("---")

    # ── PDF DOWNLOAD ──
    st.markdown("<div class='section-header'>📄 Download Report</div>", unsafe_allow_html=True)

    if ai_sections:
        with st.spinner("📝 Preparing your PDF report..."):
            pdf_buffer = generate_pdf_report(
                patient_name, age, gender, chest_display,
                resting_bp, cholesterol, fasting_bs, resting_ecg,
                max_hr, ex_angina, oldpeak, st_slope,
                pred, proba, ai_sections
            )
        fname = f"Heart_Report_{patient_name.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
        st.download_button(
            label="⬇️  Download Full PDF Report",
            data=pdf_buffer,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("Generate AI recommendations first to enable PDF download.")
