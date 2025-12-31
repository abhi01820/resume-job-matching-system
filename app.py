import streamlit as st
import pickle
import pdfplumber
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Abhi | Resume Job Matcher",
    page_icon="📄",
    layout="wide"
)


st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
}
.sub-title {
    color: #6c757d;
    font-size: 16px;
}
.card {
    padding: 16px;
    border-radius: 12px;
    background-color: #111827;
    margin-bottom: 18px;
    min-height: 260px;
}
.score {
    font-size: 22px;
    font-weight: bold;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">📄 Resume–Job Matching System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload your resume and get the best-matched, unique job roles using Machine Learning & NLP</div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    with open("resume_job_matcher_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
vectorizer = model["vectorizer"]
job_tfidf = model["job_tfidf"]
jobs_df = model["jobs_df"]


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Number of unique job matches", 4, 16, 8)
min_score = st.sidebar.slider("Minimum match score (%)", 0, 50, 15)


uploaded_file = st.file_uploader(
    "📤 Upload your Resume (PDF)",
    type=["pdf"]
)


if uploaded_file:
    with st.spinner("🔍 Extracting resume & matching jobs..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_vector = vectorizer.transform([resume_text])
        scores = cosine_similarity(resume_vector, job_tfidf)[0] * 100

        scores_df = jobs_df.copy()
        scores_df["Match Score"] = scores

        scores_df = scores_df[scores_df["Match Score"] >= min_score]
        scores_df = scores_df.sort_values("Match Score", ascending=False)

        results = (
            scores_df
            .drop_duplicates(subset="Role")
            .head(top_n)
            .reset_index(drop=True)
        )

        results["Match Score"] = results["Match Score"].round(2)

    st.success("✅ Matching completed")

    
    st.subheader("🎯 Best-Fit Job Matches")

    cols = st.columns(4)

    for idx, row in results.iterrows():
        col = cols[idx % 4]
        with col:
            st.markdown(f"""
            <div class="card">
                <h4>{row['Role']}</h4>
                <div class="score">{row['Match Score']}%</div>
                <p style="margin-top:12px;">{row['Features'][:220]}...</p>
            </div>
            """, unsafe_allow_html=True)

    
    st.subheader("📈 Match Score Trend")

    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, len(results) + 1),
        results["Match Score"],
        marker="o"
    )
    plt.xlabel("Job Rank")
    plt.ylabel("Match Score (%)")
    plt.grid(True)
    st.pyplot(plt)

else:
    st.info("⬅️ Upload a resume PDF to get started")
