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
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # ── Colour palette ──
    RED     = colors.HexColor('#C0392B')
    GREEN   = colors.HexColor('#27AE60')
    ORANGE  = colors.HexColor('#E67E22')
    DARK    = colors.HexColor('#1A1A2E')
    LIGHT   = colors.HexColor('#ECF0F1')
    WHITE   = colors.white
    YELLOW  = colors.HexColor('#F39C12')
    BLUE    = colors.HexColor('#2980B9')
    LGRAY   = colors.HexColor('#F2F4F5')
    MGRAY   = colors.HexColor('#BDC3C7')
    RISK_BG = colors.HexColor('#FDEDEC') if pred == 1 else colors.HexColor('#EAFAF1')
    RISK_BD = RED if pred == 1 else GREEN

    # ── Styles ──
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold',
                                  fontSize=22, textColor=WHITE,
                                  alignment=TA_CENTER, spaceAfter=4)
    sub_style   = ParagraphStyle('Sub', fontName='Helvetica',
                                  fontSize=11, textColor=MGRAY,
                                  alignment=TA_CENTER, spaceAfter=2)
    section_style = ParagraphStyle('Section', fontName='Helvetica-Bold',
                                    fontSize=13, textColor=DARK,
                                    spaceBefore=14, spaceAfter=6,
                                    borderPad=4)
    normal_style = ParagraphStyle('Normal', fontName='Helvetica',
                                   fontSize=10, textColor=colors.HexColor('#2C3E50'),
                                   leading=15)
    bold_style   = ParagraphStyle('Bold', fontName='Helvetica-Bold',
                                   fontSize=10, textColor=colors.HexColor('#2C3E50'))
    small_style  = ParagraphStyle('Small', fontName='Helvetica',
                                   fontSize=8, textColor=colors.HexColor('#7F8C8D'))
    center_style = ParagraphStyle('Center', fontName='Helvetica',
                                   fontSize=10, alignment=TA_CENTER,
                                   textColor=colors.HexColor('#2C3E50'))
    greeting_style = ParagraphStyle('Greeting', fontName='Helvetica-Oblique',
                                     fontSize=11, textColor=BLUE,
                                     leading=17, spaceAfter=8)
    point_style  = ParagraphStyle('Point', fontName='Helvetica',
                                   fontSize=10, textColor=colors.HexColor('#2C3E50'),
                                   leading=15, leftIndent=10, spaceAfter=3)

    story = []

    # ══ HEADER BANNER ══
    header_data = [[
        Paragraph("Heart Health Report", title_style),
        Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y')}", sub_style)
    ]]
    header_tbl = Table(header_data, colWidths=[12*cm, 5*cm])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK),
        ('ALIGN',      (0,0), (0,0),  'LEFT'),
        ('ALIGN',      (1,0), (1,0),  'RIGHT'),
        ('VALIGN',     (0,0), (-1,-1),'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0),(-1,-1),14),
        ('LEFTPADDING', (0,0),(0,-1), 16),
        ('RIGHTPADDING',(1,0),(-1,-1),16),
        ('ROUNDEDCORNERS',(0,0),(-1,-1), 8),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ══ PATIENT INFO ══
    story.append(Paragraph("👤  Patient Information", section_style))
    pat_data = [
        [Paragraph('<b>Patient Name</b>', bold_style), Paragraph(patient_name, normal_style),
         Paragraph('<b>Age</b>', bold_style),           Paragraph(f"{age} years", normal_style)],
        [Paragraph('<b>Gender</b>', bold_style),         Paragraph(gender, normal_style),
         Paragraph('<b>Report Date</b>', bold_style),    Paragraph(datetime.datetime.now().strftime('%d/%m/%Y'), normal_style)],
    ]
    pat_tbl = Table(pat_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    pat_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LGRAY),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#D5E8D4')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#D5E8D4')),
        ('GRID',       (0,0), (-1,-1), 0.5, MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('LEFTPADDING', (0,0),(-1,-1), 10),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[LGRAY, WHITE]),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ══ PREDICTION RESULT ══
    story.append(Paragraph("🩺  Prediction Result", section_style))
    risk_txt   = "⚠  HIGH RISK — Heart Disease Detected" if pred == 1 else "✓  LOW RISK — No Heart Disease Detected"
    risk_color = RED if pred == 1 else GREEN
    risk_style = ParagraphStyle('Risk', fontName='Helvetica-Bold', fontSize=14,
                                 textColor=risk_color, alignment=TA_CENTER)
    prob_style = ParagraphStyle('Prob', fontName='Helvetica', fontSize=11,
                                 textColor=colors.HexColor('#555'), alignment=TA_CENTER)

    result_data = [[
        Paragraph(risk_txt, risk_style),
        Paragraph(f"Disease Probability: {proba[1]*100:.1f}%    |    No Disease: {proba[0]*100:.1f}%", prob_style)
    ]]
    result_tbl = Table(result_data, colWidths=[17*cm])
    result_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), RISK_BG),
        ('BOX',           (0,0), (-1,-1), 2, RISK_BD),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('SPAN',          (0,0), (-1,-1)),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ══ CLINICAL VALUES TABLE ══
    story.append(Paragraph("📊  Clinical Values vs Normal Range", section_style))

    # header row
    tbl_header = [
        Paragraph('<b>Parameter</b>',   bold_style),
        Paragraph('<b>Patient Value</b>', bold_style),
        Paragraph('<b>Normal Range</b>', bold_style),
        Paragraph('<b>Status</b>',       bold_style),
    ]

    chest_display = chest_pain

    rows_raw = [
        ('Age',              f"{age} years",          '18–55 years',        age > 55),
        ('Resting BP',       f"{resting_bp} mmHg",    '90–120 mmHg',        resting_bp > 120),
        ('Cholesterol',      f"{cholesterol} mg/dl",  '< 200 mg/dl',        cholesterol >= 200),
        ('Max Heart Rate',   f"{max_hr} bpm",         '100–170 bpm',        max_hr < 100 or max_hr > 170),
        ('Oldpeak (ST Dep)', f"{oldpeak}",             '0.0 – 1.0',         oldpeak > 1.0),
        ('Fasting Blood Sugar', fasting_bs,           '< 120 mg/dl (No)',   fasting_bs == 'Yes'),
        ('Resting ECG',      resting_ecg,              'NORMAL',             resting_ecg != 'NORMAL'),
        ('Chest Pain Type',  chest_display,            'ATA / NAP / TA',     chest_display == 'Asymptomatic'),
        ('Exercise Angina',  ex_angina,                'No',                 ex_angina == 'Yes'),
        ('ST Slope',         st_slope,                 'UP',                 st_slope in ['FLAT','DOWN']),
    ]

    tbl_data = [tbl_header]
    row_styles = []

    for i, (param, value, normal, is_bad) in enumerate(rows_raw):
        status_txt   = '⚠ Abnormal' if is_bad else '✓ Normal'
        status_color = RED if is_bad else GREEN
        st_style = ParagraphStyle('St', fontName='Helvetica-Bold', fontSize=10,
                                   textColor=status_color, alignment=TA_CENTER)
        val_style = ParagraphStyle('Val', fontName='Helvetica-Bold' if is_bad else 'Helvetica',
                                    fontSize=10,
                                    textColor=RED if is_bad else colors.HexColor('#2C3E50'),
                                    alignment=TA_CENTER)
        tbl_data.append([
            Paragraph(param, normal_style),
            Paragraph(value, val_style),
            Paragraph(normal, center_style),
            Paragraph(status_txt, st_style),
        ])
        # alternate row background + highlight abnormal
        bg = colors.HexColor('#FEF9E7') if is_bad else (LGRAY if i % 2 == 0 else WHITE)
        row_styles.append(('BACKGROUND', (0, i+1), (-1, i+1), bg))
        if is_bad:
            row_styles.append(('BOX', (0, i+1), (-1, i+1), 1, colors.HexColor('#F39C12')))

    clin_tbl = Table(tbl_data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 4.5*cm])
    base_style = [
        ('BACKGROUND',    (0,0), (-1,0),  DARK),
        ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0),  10),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('ALIGN',         (0,0), (0,-1),  'LEFT'),
        ('GRID',          (0,0), (-1,-1), 0.4, MGRAY),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [LGRAY, WHITE]),
    ]
    clin_tbl.setStyle(TableStyle(base_style + row_styles))
    story.append(clin_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ══ AI RECOMMENDATIONS ══
    story.append(Paragraph("💡  Personalised Health Recommendations", section_style))

    if ai_sections.get('GREETING'):
        story.append(Paragraph(ai_sections['GREETING'].strip(), greeting_style))

    section_icons = {
        'DIET'    : ('🥗', 'Diet & Nutrition'),
        'EXERCISE': ('🏃', 'Exercise & Activity'),
        'HABITS'  : ('🌙', 'Daily Habits'),
        'AVOID'   : ('🚫', 'Things to Avoid'),
        'MONITOR' : ('📋', 'Monitor & Follow-up'),
    }
    point_emojis_pdf = {
        'DIET'    : ['•','•','•','•'],
        'EXERCISE': ['•','•','•'],
        'HABITS'  : ['•','•','•'],
        'AVOID'   : ['•','•','•'],
        'MONITOR' : ['•','•'],
    }

    for key, (icon, title) in section_icons.items():
        points = ai_sections.get(key, [])
        if not points:
            continue
        sec_hdr = ParagraphStyle(f'SecHdr_{key}', fontName='Helvetica-Bold',
                                   fontSize=11, textColor=BLUE,
                                   spaceBefore=10, spaceAfter=4)
        story.append(Paragraph(f"{icon}  {title}", sec_hdr))
        for pt in points:
            story.append(Paragraph(f"   • {pt}", point_style))

    story.append(Spacer(1, 0.5*cm))

    # ══ FOOTER ══
    story.append(HRFlowable(width='100%', thickness=0.5, color=MGRAY))
    story.append(Spacer(1, 0.2*cm))
    footer_txt = "This report is generated for health awareness purposes. Please consult a qualified cardiologist for medical diagnosis and treatment."
    story.append(Paragraph(footer_txt, ParagraphStyle('Footer', fontName='Helvetica-Oblique',
                                                        fontSize=8, textColor=MGRAY,
                                                        alignment=TA_CENTER)))

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
