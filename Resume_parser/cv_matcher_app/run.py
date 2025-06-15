import streamlit as st
import os
from app.parser.cv_preprocessor import preprocess_cv
from app.parser.cv_structured_parser import extract_structured_fields
from app.matching.semantic_matcher import SemanticScorer
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

# Initialize
scorer = SemanticScorer()
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def plot_experience_timeline(experience_data):
    """Plot the job experience timeline with monthly ticks for Streamlit."""
    if not experience_data:
        return None

    # Prepare jobs list
    jobs = []
    for entry in experience_data:
        s = entry['start_date']
        t = entry['end_date']
        # swap if reversed
        if t < s:
            s, t = t, s
        title = entry['role'] + (f" at {entry['company']}" if entry['company'] else "")
        words = title.split()
        if len(words) > 6:
            title = ' '.join(words[:6]) + ' ...'
        jobs.append((title, s, t))

    # sort by start date
    jobs.sort(key=lambda x: x[1])

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (title, start, end) in enumerate(jobs):
        ax.plot([mdates.date2num(start), mdates.date2num(end)], [idx, idx], marker='o', linewidth=4)

    # Y-axis labels
    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([t for t, _, _ in jobs], fontsize=12)

    # X-axis monthly ticks between min and max dates
    min_date = min(start for _, start, _ in jobs)
    max_date = max(end for _, _, end in jobs)
    monthly = pd.date_range(start=min_date, end=max_date, freq='MS')
    tick_nums = mdates.date2num(monthly.to_pydatetime())
    ax.set_xticks(tick_nums)
    ax.set_xticklabels([d.strftime('%b %Y') for d in monthly], rotation=45, ha='right')
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=15))

    # Styling
    ax.set_xlabel("Date", fontsize=14)
    ax.set_title("Job Experience Timeline", fontsize=18, fontweight='bold')
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig

# Streamlit app configuration
st.set_page_config(page_title="CV & JD Matcher", layout="wide")
st.title("📄 CV & Job Description Matcher")

st.sidebar.header("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
jd_text = st.sidebar.text_area("Paste the Job Description here:")

if uploaded_file and jd_text:
    if allowed_file(uploaded_file.name):
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing CV..."):
            cv_data = preprocess_cv(filepath)
            cv_text = cv_data.get("cleaned_text", "")
            structured_cv_data = extract_structured_fields(cv_text)
            jd_structured = extract_structured_fields(jd_text)

            match_result = scorer.score_cv(structured_cv_data, jd_structured)
            missing_keywords = scorer.find_missing_keywords(structured_cv_data, jd_structured)
            wordcloud_base64 = scorer.generate_word_cloud(missing_keywords)
            shap_base64 = scorer.explain_score(structured_cv_data, jd_structured)
            experience_fig = plot_experience_timeline(structured_cv_data.get('experience_details', []))

        st.success("✅ Matching Completed!")

        # Display Matching Results
        st.subheader("🎯 Matching Results")
        st.markdown(f"**Best Match Sentence:** {match_result.get('best_match_sentence', 'Not available')}")
        st.markdown(f"**Best Match Score:** `{match_result.get('best_match_score', 0):.2f}%`")

        st.subheader("📊 Detailed Match Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skills Match", f"{match_result.get('skills_similarity', 0) * 100:.1f}%")
        with col2:
            st.metric("Experience Match", f"{match_result.get('experience_similarity', 0) * 100:.1f}%")
        with col3:
            st.metric("Education Match", f"{match_result.get('education_similarity', 0) * 100:.1f}%")

        st.subheader("❌ Missing Keywords")
        if missing_keywords:
            st.write(missing_keywords)
        else:
            st.write("No missing keywords detected!")

        if wordcloud_base64:
            st.subheader("☁️ Missing Keywords Word Cloud")
            st.image(f"data:image/png;base64,{wordcloud_base64}")

        if shap_base64:
            st.subheader("🔎 SHAP Analysis - Score Explainability")
            st.image(f"data:image/png;base64,{shap_base64}")

        if experience_fig:
            st.subheader("🛤️ Experience Timeline")
            st.pyplot(experience_fig)

        st.subheader("🗂️ Structured CV Data")
        st.write(structured_cv_data)

    else:
        st.error("⚠️ Unsupported file format. Please upload a PDF or DOCX file.")
else:
    st.info("⬆️ Upload your CV and paste the Job Description to begin.")
