# ==== Fix for torch + asyncio + Streamlit issue ====
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ==== Imports ====
import streamlit as st
import pdfplumber
from utils import preprocess, get_semantic_match_score, extract_keywords

MAX_FILE_SIZE = 1_048_576  # 1 MB

# ==== Page Setup ====
st.set_page_config(page_title="Semantic Resume Matcher", layout="centered")

# ==== Sidebar Instructions ====
with st.sidebar:
    st.title("📌 How to Use")
    st.markdown("""
    1. Upload your **resume PDF (≤1MB)**
    2. Paste a **job description**
    3. Click "🔍 Match Resume" to get your match score and suggestions
    """)

# ==== Title ====
st.title("🤖 Semantic Resume Matcher")
st.markdown("This tool compares your resume with a job description using **NLP and Sentence-BERT**.")

# ==== File Upload ====
uploaded_resume = st.file_uploader("📄 Upload Resume (PDF only)", type="pdf")

# ==== Job Description Input ====
job_desc = st.text_area("📝 Paste Job Description", height=200)

# ==== Submit Button ====
if st.button("🔍 Match Resume"):
    if not uploaded_resume:
        st.error("❌ Please upload your resume PDF.")
    elif not job_desc.strip():
        st.error("❌ Please paste a job description.")
    elif uploaded_resume.size > MAX_FILE_SIZE:
        st.error("❌ File size exceeds 1 MB. Please upload a smaller PDF.")
    else:
        with st.spinner("🔍 Analyzing your resume..."):
            try:
                # Extract resume text
                with pdfplumber.open(uploaded_resume) as pdf:
                    resume_text = "".join(page.extract_text() or "" for page in pdf.pages)

                # Preprocess both texts
                resume_clean = preprocess(resume_text)
                jd_clean = preprocess(job_desc)

                # Get semantic similarity score
                score = get_semantic_match_score(resume_clean, jd_clean)

                # Display color-coded score
                if score >= 80:
                    st.success(f"🟢 Excellent Match! Score: **{score:.2f}%**")
                elif score >= 60:
                    st.warning(f"🟡 Moderate Match. Score: **{score:.2f}%**")
                else:
                    st.error(f"🔴 Low Match. Score: **{score:.2f}%**")

                # Keyword analysis
                jd_keywords = set(extract_keywords(jd_clean))
                resume_keywords = set(extract_keywords(resume_clean))
                missing = jd_keywords - resume_keywords

                if missing:
                    st.warning("📌 Consider adding these keywords to your resume:")
                    st.write(", ".join(missing))
                else:
                    st.info("✅ Your resume already includes all important keywords!")

            except Exception as e:
                st.error(f"⚠️ Error while reading the file: {e}")
