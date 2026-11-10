import streamlit as st
import pandas as pd
from models.text_model import train_text_model, predict_with_solution
from models.preprocess import load_data, clean_data
from models.structured_model import train_models
from models.visualization import plot_results
import matplotlib.pyplot as plt

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SmartClinic AI",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — dark clinical theme ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: #0a0f1e !important;
    color: #e8eaf0 !important;
}
.stApp { background-color: #0a0f1e; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1829 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #ffffff !important; }

/* ── inputs ── */
textarea, .stTextArea textarea {
    background-color: #0a1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 15px !important;
}
textarea:focus { border-color: #00d4aa !important; }

/* ── buttons ── */
.stButton button {
    background: linear-gradient(135deg, #00d4aa, #0066ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover { opacity: 0.9; transform: scale(1.02); }

/* ── metrics ── */
[data-testid="stMetric"] {
    background: #0d1829;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #00d4aa !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #5d8ab0 !important; }

/* ── alerts ── */
.stSuccess { background: #0a2a1a !important; border: 1px solid #00d4aa !important; border-radius: 10px !important; }
.stError   { background: #2a0a0a !important; border: 1px solid #ff4444 !important; border-radius: 10px !important; }
.stWarning { background: #2a1e00 !important; border: 1px solid #ffaa00 !important; border-radius: 10px !important; }
.stInfo    { background: #0a1a2a !important; border: 1px solid #0066ff !important; border-radius: 10px !important; }

/* ── expander ── */
.streamlit-expanderHeader {
    background: #0d1829 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #c8d8e8 !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0d1829; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #5d8ab0 !important; }
.stTabs [aria-selected="true"] { color: #00d4aa !important; border-bottom: 2px solid #00d4aa !important; }

/* ── divider ── */
hr { border-color: #1e3a5f !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 8, 2])
with col_logo:
    st.markdown("<div style='font-size:48px; margin-top:8px;'>⚕</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='color:#ffffff; margin:0; font-size:2rem; font-weight:600; letter-spacing:-0.5px;'>
        SmartClinic <span style='color:#00d4aa;'>AI</span>
    </h1>
    <p style='color:#5d8ab0; font-family:Space Mono,monospace; font-size:12px; margin:0;'>
        Disease Prediction System · CAP 258,263 · CA III
    </p>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style='margin-top:16px; text-align:right;
                font-family:Space Mono,monospace; font-size:12px; color:#00d4aa;'>
        ● SYSTEM ONLINE
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─── Load & train models (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_text_model():
    df = pd.read_csv("Data/symptom_text.csv")
    model, vectorizer = train_text_model(df["symptoms"], df["condition"])
    return model, vectorizer, df

@st.cache_resource(show_spinner=False)
def load_structured_model():
    data = load_data()
    data = clean_data(data)
    X = data.drop("target", axis=1)
    y = data["target"]
    results, trained = train_models(X, y)
    return results, trained

with st.spinner("🔬 Loading ML models..."):
    text_model, vectorizer, df_symptoms = load_text_model()
    struct_results, struct_models = load_structured_model()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚕ SmartClinic AI")
    st.markdown("---")

    st.markdown("### 🩺 Quick Symptoms")
    st.caption("Click to auto-fill")

    quick = {
        "🤕 Headache + Fever":     "fever headache body ache fatigue",
        "💓 Chest Pain":           "chest pain shortness of breath sweating",
        "🤧 Cold & Cough":         "cough cold runny nose sore throat",
        "🤢 Nausea & Stomach":     "nausea vomiting stomach pain diarrhea",
        "🦵 Joint Pain":           "joint pain swelling stiffness",
        "🌡 Dengue Symptoms":      "fever rash joint pain red eyes",
        "😮‍💨 Breathing Issues":  "chest tightness wheezing difficulty breathing",
        "🧠 Migraine":             "severe headache nausea sensitivity to light",
    }

    for label, symptoms in quick.items():
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state["prefill"] = symptoms

    st.markdown("---")
    st.markdown("### 🤖 ML Models")
    for model_name, acc in struct_results.items():
        st.markdown(f"""
        <div style='background:#0a1525; border:1px solid #1e3a5f; border-radius:8px;
                    padding:8px 12px; margin-bottom:6px;'>
            <div style='font-family:Space Mono,monospace; font-size:11px; color:#5d8ab0;'>{model_name}</div>
            <div style='color:#00d4aa; font-weight:600; font-size:16px;'>{acc:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠ Not a substitute for real medical advice. Always consult a doctor.")


# ─── Main tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩺 Symptom Checker", "📊 Model Performance", "📋 Dataset"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Symptom Checker
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # Stat cards
    best_model = max(struct_results, key=struct_results.get)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🤖 Models Trained", "3")
    with c2: st.metric("🏆 Best Model", best_model)
    with c3: st.metric("✅ Best Accuracy", f"{struct_results[best_model]:.1f}%")
    with c4: st.metric("🦠 Conditions Known", str(len(df_symptoms)))

    st.markdown("<br>", unsafe_allow_html=True)

    # Input area
    st.markdown("""
    <div style='background:#0d1829; border:1px solid #1e3a5f; border-radius:14px; padding:1.5rem; margin-bottom:1rem;'>
        <h3 style='color:#ffffff; margin:0 0 0.3rem;'>How are you feeling today?</h3>
        <p style='color:#5d8ab0; font-size:13px; margin:0;'>
            Describe your symptoms in detail for a more accurate diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill", "")
    patient_input = st.text_area(
        "Describe your symptoms",
        value=prefill,
        height=120,
        placeholder="e.g., I have a sharp headache, high fever since 2 days, body aches and fatigue...",
        label_visibility="collapsed"
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        analyse = st.button("🔬 Analyse Symptoms", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.rerun()

    # ── Results ──────────────────────────────────────────────────────────────
    if analyse:
        if not patient_input.strip():
            st.warning("⚠ Please describe your symptoms first.")
        else:
            with st.spinner("🔬 Analysing symptoms with ML models..."):
                condition, severity, solution = predict_with_solution(
                    text_model, vectorizer, patient_input.strip(), df_symptoms
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Severity badge ────────────────────────────────────────────────
            sev_colors = {
                "Critical": ("#ff4444", "#3a0f0f", "🚨"),
                "Moderate": ("#ffaa00", "#2a1e00", "⚠"),
                "Mild":     ("#00d4aa", "#0a2a1a", "✅"),
            }
            color, bg, icon = sev_colors.get(severity, ("#0066ff", "#0a1525", "ℹ"))

            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {color}; border-radius:14px;
                        padding:1.2rem 1.5rem; margin-bottom:1rem;
                        display:flex; align-items:center; gap:12px;'>
                <span style='font-size:28px;'>{icon}</span>
                <div>
                    <div style='font-family:Space Mono,monospace; font-size:11px;
                                color:{color}; text-transform:uppercase; letter-spacing:1px;'>
                        Severity Level
                    </div>
                    <div style='font-size:22px; font-weight:600; color:#ffffff;'>
                        {severity}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Condition card ────────────────────────────────────────────────
            st.markdown(f"""
            <div style='background:#0d1829; border:1px solid #1e3a5f; border-radius:14px;
                        padding:1.2rem 1.5rem; margin-bottom:1rem;'>
                <div style='font-family:Space Mono,monospace; font-size:11px; color:#5d8ab0;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>
                    Predicted Condition
                </div>
                <div style='font-size:26px; font-weight:600; color:#00d4aa;'>
                    {condition}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Solution card ─────────────────────────────────────────────────
            steps = [s.strip() for s in solution.split(",") if s.strip()]
            steps_html = "".join(
                f"<div style='display:flex;gap:10px;margin-bottom:8px;align-items:flex-start;'>"
                f"<span style='color:#00d4aa;flex-shrink:0;'>→</span>"
                f"<span style='color:#c8d8e8;font-size:14px;'>{s}</span></div>"
                for s in steps
            )
            st.markdown(f"""
            <div style='background:#0d1829; border:1px solid #1e3a5f;
                        border-left:4px solid #00d4aa; border-radius:14px;
                        padding:1.2rem 1.5rem; margin-bottom:1rem;'>
                <div style='font-family:Space Mono,monospace; font-size:11px; color:#5d8ab0;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;'>
                    Recommended Actions
                </div>
                {steps_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Emergency banner ──────────────────────────────────────────────
            if severity == "Critical":
                st.error("🚨 **URGENT:** This condition requires immediate medical attention. Please visit the nearest emergency room or call emergency services right away.")
            elif severity == "Moderate":
                st.warning("⚠ Your symptoms suggest a condition that needs medical evaluation. Please consult a doctor soon and monitor your symptoms closely.")
            else:
                st.success("✅ Your symptoms appear mild. Follow the recommended actions and consult a doctor if symptoms persist or worsen.")

            # ── Disclaimer ────────────────────────────────────────────────────
            with st.expander("ℹ About this prediction"):
                st.info(f"""
                **How this works:**
                - Your symptom text was classified using a **Multinomial Naive Bayes** model trained on `symptom_text.csv`
                - The model was trained on **{len(df_symptoms)} symptom-condition pairs**
                - This is a university project (CAP 258,263) — always seek professional medical advice

                **Input received:** _{patient_input.strip()}_
                """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Structured Model Performance (Heart Disease Dataset)")
    st.caption("KNN, Decision Tree and Naive Bayes trained on heart.csv")

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric cards
    cols = st.columns(len(struct_results))
    for col, (name, acc) in zip(cols, struct_results.items()):
        with col:
            st.metric(name, f"{acc:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0d1829")

    names  = list(struct_results.keys())
    values = list(struct_results.values())
    colors = ["#00d4aa", "#0066ff", "#7c3aed"]

    bars = ax.bar(names, values, color=colors, width=0.45, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_title("Model Comparison — Heart Disease Dataset",
                 color="white", fontsize=13, pad=12)
    ax.set_ylabel("Accuracy (%)", color="#5d8ab0", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#1e3a5f")
    ax.yaxis.grid(True, color="#1e3a5f", linestyle="--", alpha=0.6, zorder=0)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🤖 Text Model — Symptom Classifier")
    st.markdown(f"""
    <div style='background:#0d1829; border:1px solid #1e3a5f; border-radius:12px; padding:1.2rem;'>
        <p style='color:#c8d8e8; margin:0;'>
            Algorithm: <span style='color:#00d4aa; font-weight:600;'>Multinomial Naive Bayes + CountVectorizer</span><br>
            Training samples: <span style='color:#00d4aa; font-weight:600;'>{len(df_symptoms)}</span><br>
            Unique conditions: <span style='color:#00d4aa; font-weight:600;'>{df_symptoms["condition"].nunique()}</span><br>
            Input: Raw symptom text → tokenized → frequency vector → NB classifier
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Symptom Dataset")

    sev_filter = st.selectbox(
        "Filter by Severity",
        ["All", "Critical", "Moderate", "Mild"]
    )

    filtered = df_symptoms if sev_filter == "All" \
        else df_symptoms[df_symptoms["severity"] == sev_filter]

    st.dataframe(
        filtered[["condition", "severity", "symptoms", "solution"]],
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {len(filtered)} of {len(df_symptoms)} records")

    # ── Heart Disease Dataset Section ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🫀 Heart Disease Dataset")
    st.caption("Structured dataset used to train KNN, Decision Tree, and Naive Bayes models")

    try:
            df_heart = pd.read_csv("Data/heart.csv")

            # Summary metrics
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.metric("📋 Total Records", str(len(df_heart)))
            with h2:
                st.metric("📊 Total Features", str(df_heart.shape[1]))
            with h3:
                st.metric("❤️ Heart Disease Cases", str(df_heart["target"].sum()))
            with h4:
                st.metric("✅ Healthy Cases", str((df_heart["target"] == 0).sum()))

            st.markdown("<br>", unsafe_allow_html=True)

            # Column filter
            col_filter, col_rows = st.columns([3, 1])
            with col_filter:
                selected_cols = st.multiselect(
                    "Select columns to display",
                    options=df_heart.columns.tolist(),
                    default=df_heart.columns.tolist()
                )
            with col_rows:
                show_rows = st.selectbox(
                    "Rows to show",
                    [10, 25, 50, "All"],
                    index=0
                )

            display_df = df_heart[selected_cols] if selected_cols else df_heart
            if show_rows != "All":
                display_df = display_df.head(show_rows)

            st.dataframe(display_df, use_container_width=True, height=350)
            st.caption(f"Showing {len(display_df)} of {len(df_heart)} records | {df_heart.shape[1]} columns")

    except FileNotFoundError:
            st.error("⚠️ heart.csv not found. Make sure Data/heart.csv exists.")

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
            st.download_button(
                "⬇ Download symptom_text.csv",
                data=df_symptoms.to_csv(index=False),
                file_name="symptom_text.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col_dl2:
            try:
                df_heart_dl = pd.read_csv("Data/heart.csv")
                st.download_button(
                    "⬇ Download heart.csv",
                    data=df_heart_dl.to_csv(index=False),
                    file_name="heart.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except FileNotFoundError:
                pass