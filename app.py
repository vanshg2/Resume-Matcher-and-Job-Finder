# app.py
import streamlit as st
st.set_page_config(page_title="Resume Matcher & Role Finder", layout="wide", page_icon="📄")

import os, re, unicodedata, pickle
from io import BytesIO
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
from pathlib import Path

# -------------------- File names (expected in project root) --------------------
MODEL_PATH = "resume_classifier.pkl"
VECT_PATH = "tfidf_vectorizer.pkl"
LABEL_PATH = "label_encoder.pkl"
SEED_SKILLS_FILES = ["seed_skills.csv", "Seed_Skills.csv"]
JOB_ROLES_FILES = ["job_roles.csv", "Job_Roles.csv"]

# -------------------- Utility helpers --------------------

@st.cache_data(show_spinner=False)
def render_pdf_page(file_bytes: bytes, page_number: int = 1, zoom: float = 1.0):
    """
    Render single PDF page to PNG bytes and return dict with page info.
    Caches by file bytes + page + zoom.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")

    # basic validation
    if not file_bytes:
        raise ValueError("Empty file bytes provided to render_pdf_page()")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if page_number < 1 or page_number > len(doc):
        doc.close()
        raise IndexError(f"Page number {page_number} out of range (1..{len(doc)})")

    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    png_bytes = pix.tobytes("png")
    info = {"page_number": page_number, "png": png_bytes, "width": pix.width, "height": pix.height, "total_pages": len(doc)}
    doc.close()
    return info

def show_pdf_preview_widget(uploaded_file, default_zoom=1.0):
    """
    Display PDF preview for Streamlit UploadedFile.
    - Reads bytes safely (resets pointer).
    - Lets user pick page and zoom.
    - Renders only selected page (memory-friendly).
    """
    if uploaded_file is None:
        st.info("Upload a PDF resume to preview its original layout here.")
        return

    # Make sure we can read the file bytes (reset pointer)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        file_bytes = uploaded_file.read()
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return

    if not file_bytes:
        st.error("Uploaded file is empty (0 bytes).")
        return

    # test fitz availability
    if fitz is None:
        st.error("PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")
        return

    # determine number of pages
    try:
        # get page count without rendering all pages
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except Exception as e:
        st.error(f"Failed to open PDF for preview: {e}")
        return

    # UI: page selector
    page_num = st.number_input("Page", min_value=1, max_value=max(1, page_count), value=1, step=1)
    st.download_button(label="Download original PDF", data=file_bytes, file_name=getattr(uploaded_file, "name", "resume.pdf"), mime="application/pdf")

    # Render only the requested page (cached)
    try:
        with st.spinner(f"Rendering page {page_num} ..."):
            info = render_pdf_page(file_bytes, page_number=page_num, zoom=2.0)
        st.image(info["png"], use_column_width=True)
        st.caption(f"Preview — page {info['page_number']} of {info['total_pages']} (rendered {info['width']}×{info['height']} px)")
    except Exception as e:
        st.error(f"Unable to render PDF page: {e}")
        # For debugging show the exception details in dev mode:
        if st.session_state.get("dev_mode", False):
            import traceback; st.text(traceback.format_exc())

def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        b = uploaded_file.read()
        text = ""
        with fitz.open(stream=b, filetype="pdf") as doc:
            for p in doc:
                txt = p.get_text()
                if txt:
                    text += txt + "\n"
        return normalize_text(text)
    except Exception:
        return ""

def clean_resume_text(text: str) -> str:
    """Remove noisy boilerplate (urls, long footers, repetitive lines, page numbers)."""
    if not text:
        return ""
    t = text
    t = re.sub(r'https?://\S+', ' ', t)  # URLs
    t = re.sub(r'www\.\S+', ' ', t)
    t = re.sub(r'©.*|copyright.*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\n{2,}', '\n', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = re.sub(r'\bpage\s*\d+\b', ' ', t, flags=re.IGNORECASE)
    # remove very long trailing boilerplate that often appears in templates
    t = re.sub(r'(resume template|template by|downloaded from|all rights reserved)[\s\S]*$', '', t, flags=re.IGNORECASE)
    return normalize_text(t)

@st.cache_data
def load_pickle_if_exists(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def load_seed_skills():
    """Try common filenames and return a list of skills (lowercased)."""
    for fn in SEED_SKILLS_FILES:
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn)
                # prefer column named 'skill' else first column
                if "skill" in df.columns:
                    return sorted(set(df["skill"].dropna().astype(str).str.lower().tolist()))
                else:
                    col = df.columns[0]
                    return sorted(set(df[col].dropna().astype(str).str.lower().tolist()))
            except Exception:
                continue
    # fallback seed list (small)
    fallback = [
        "python","java","c++","sql","excel","power bi","tableau","aws","azure","docker","kubernetes",
        "git","linux","javascript","html","css","react","node","django","flask","communication","recruitment"
    ]
    return sorted(set(fallback))

@st.cache_data
def load_job_roles():
    """Load job_roles.csv (role,skills) mapping; return dict role->list(skill)."""
    for fn in JOB_ROLES_FILES:
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn)
                # if columns present
                if "role" in df.columns and "skills" in df.columns:
                    mapping = {}
                    for _, r in df.iterrows():
                        role = str(r["role"]).strip()
                        skills = [s.strip().lower() for s in str(r["skills"]).split(",") if s.strip()]
                        if role:
                            mapping[role] = skills
                    if mapping:
                        return mapping
                else:
                    # try first two columns as role, skills
                    mapping = {}
                    for _, r in df.iterrows():
                        role = str(r.iloc[0]).strip()
                        skills = []
                        if df.shape[1] > 1:
                            skills = [s.strip().lower() for s in str(r.iloc[1]).split(",") if s.strip()]
                        mapping[role] = skills
                    if mapping:
                        return mapping
            except Exception:
                continue
    # fallback mapping
    return {
        "Data Scientist": ["python","machine learning","sql","tableau"],
        "Web Developer": ["html","css","javascript","react"],
        "DevOps Engineer": ["docker","kubernetes","aws","git","linux"],
        "HR Manager": ["recruitment","onboarding","employee relations","hris","compensation and benefits"]
    }

def extract_skills_from_text(text: str, vocab: list):
    t = (text or "").lower()
    found = [s for s in vocab if s in t]
    return sorted(set(found))

def compute_tfidf_similarity(text1: str, text2: str, vectorizer=None):
    if not text1 or not text2:
        return 0.0
    try:
        if vectorizer is None:
            tf = TfidfVectorizer(stop_words="english", max_df=0.85)
            X = tf.fit_transform([text1, text2])
        else:
            # try transform, fallback to fit if fails
            try:
                X = vectorizer.transform([text1, text2])
            except Exception:
                tf = TfidfVectorizer(stop_words="english", max_df=0.85)
                X = tf.fit_transform([text1, text2])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def sentences_by_similarity(resume_text, jd_text, vectorizer=None, top_n=5):
    import re
    sentences = [s.strip() for s in re.split(r'[\n\.]\s+', resume_text) if len(s.strip()) > 20]
    if not sentences:
        return []
    try:
        if vectorizer is None:
            tf = TfidfVectorizer(stop_words="english", max_df=0.9)
            X = tf.fit_transform(sentences + [jd_text])
        else:
            try:
                X = vectorizer.transform(sentences + [jd_text])
            except Exception:
                tf = TfidfVectorizer(stop_words="english", max_df=0.9)
                X = tf.fit_transform(sentences + [jd_text])
        jd_vec = X[-1]
        sims = cosine_similarity(X[:-1], jd_vec).flatten()
        idxs = sims.argsort()[::-1][:top_n]
        results = [(float(sims[i]), sentences[i]) for i in idxs if sims[i] > 0.03]
        return results
    except Exception:
        return []

def suggestion_lines(missing_skills):
    out = []
    for s in missing_skills:
        out.append(f"Worked on {s} to ... (describe what you did and the result, include metrics).")
    return out

# -------------------- Load models + data --------------------
model = load_pickle_if_exists(MODEL_PATH)
vectorizer = load_pickle_if_exists(VECT_PATH)
label_encoder = load_pickle_if_exists(LABEL_PATH)

seed_skills = load_seed_skills()
role_skills_map = load_job_roles()
skills_vocab = sorted(set(seed_skills + [sk for skills in role_skills_map.values() for sk in skills]))

# -------------------- UI Header (cosmetic) --------------------
st.markdown("""
<style>
.header-card {
    background-color: #0f172a;
    color: white;
    padding: 16px 20px;
    border-radius: 8px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.25);
}
.header-card h1 {
    font-size: 1.7rem;
}
.small {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='header-card'><h1>📄 Resume Matcher & Role Finder</h1>"
    "<div class='small'>Choose whether you want to match your resume to a specific job or find roles that suit your resume. Upload a PDF resume to begin.</div></div>",
    unsafe_allow_html=True
)

# st.markdown("<div class='header-card'><h1>📄 Resume Matcher & Role Finder</h1><div class='small'>Choose whether you want to match your resume to a specific job or find roles that suit your resume. Upload a PDF resume to begin.</div></div>", unsafe_allow_html=True)
# st.write("")

# -------------------- Step 1: Choose action --------------------
action = st.radio("What are you looking for?", ("Job matching (match my resume to a job)", "Find suitable job roles for my resume"), index=0, horizontal=True)

# -------------------- Resume upload (common) --------------------
with st.expander("Upload resume (PDF)"):
    uploaded_file = st.file_uploader("Upload resume (.pdf)", type=["pdf"], help="Upload a PDF resume for analysis")

resume_text = ""
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_text = clean_resume_text(resume_text)
    if not resume_text:
        st.error("Could not extract text — the resume might be scanned. Try OCR or a different file.")
    else:
        st.success("Resume uploaded and parsed.")
        # small preview
        with st.expander("Preview resume (original PDF view)"):
            show_pdf_preview_widget(uploaded_file, default_zoom=1.0)

# -------------------- Action: Job matching --------------------
if action.startswith("Job matching"):
    st.markdown("### 🔎 Job matching")
    # layout: resume area on top, then two columns (left: resume, right: job roles / JD)
    col_top = st.columns(1)[0]
    # top: required resume upload reminder
    if not uploaded_file:
        st.info("Upload your resume to enable matching.")
    # two-column main area
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("1) Resume")
        if not uploaded_file:
            st.write("No resume uploaded yet.")
        else:
            # allow edit preview
            resume_text = st.text_area("Edit resume text (optional)", value=resume_text, height=220)
            st.write(f"Resume length: {len(resume_text)} characters")

    with right:
        st.subheader("2) Job roles / JD")
        available_roles = sorted(role_skills_map.keys())
        chosen_role = st.selectbox("Choose a job role (from job_roles.csv)", options=["-- Select a Job Role --"] + available_roles)
        jd_text = ""
        if chosen_role and chosen_role != "-- Select a Job Role --":
            st.info("Required skills: " + ", ".join(role_skills_map.get(chosen_role, [])))
            use_role = chosen_role
        else:
            st.warning("If your desired role is not listed, paste the Job Description below.")
            use_role = None

        st.markdown("---")
        pasted_jd = st.text_area("Paste Job Description (optional)", height=160, placeholder="Paste full JD here if role not available")

    # ---------------- Sidebar: Scoring Adjustment ----------------
    st.sidebar.markdown("### ⚙️ Scoring Adjustments")
    st.sidebar.caption(
        "Use the sliders below to **fine-tune** how your match score is calculated.\n\n"
        "- 🧩 **Skills coverage:** measures how many required skills are found in your resume.\n"
        "- 🧠 **Semantic similarity:** measures how similar your resume text is to the Job Description (JD).\n\n"
        "👉 Adjust the sliders to emphasize one over the other based on your resume type."
    )
    
    w_skills = st.sidebar.slider("🔧 Emphasis on Skill Keywords", 0.0, 1.0, 0.6, 0.05)
    w_sem = st.sidebar.slider("🧠 Emphasis on Resume–JD Meaning", 0.0, 1.0, 0.35, 0.05)
    
    total = max(1e-6, w_skills + w_sem)
    w_skills /= total
    w_sem /= total

    st.sidebar.progress(int(w_skills * 100))
    st.sidebar.caption(f"Skills importance: {int(w_skills*100)}% | Semantics: {int(w_sem*100)}%")

    # normalize
    total = max(1e-6, w_skills + w_sem)
    w_skills /= total
    w_sem /= total

    analyze_btn = st.button("Analyze Match", type="primary")

    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a resume first.")
        else:
            # pick JD source
            if pasted_jd and pasted_jd.strip():
                jd_text = normalize_text(pasted_jd)
                jd_role_used = "Custom JD"
                jd_skills = extract_skills_from_text(jd_text, skills_vocab)
            elif use_role:
                jd_role_used = use_role
                jd_skills = role_skills_map.get(use_role, [])
                jd_text = ", ".join(jd_skills)
            else:
                # try model predicted role
                if model is not None and vectorizer is not None and label_encoder is not None:
                    try:
                        X = vectorizer.transform([resume_text])
                        pred = model.predict(X)
                        pred_role = label_encoder.inverse_transform(pred)[0]
                        jd_role_used = "Predicted: " + pred_role
                        jd_skills = role_skills_map.get(pred_role, [])
                        jd_text = ", ".join(jd_skills)
                        st.success(f"Model predicted role: **{pred_role}**")
                    except Exception:
                        jd_role_used = "None"
                        jd_skills = []
                else:
                    st.error("No role selected, JD not provided, and model unavailable.")
                    jd_role_used = "None"
                    jd_skills = []

            # extract resume skills
            resume_skills = extract_skills_from_text(resume_text, skills_vocab)

            # compute metrics
            coverage = (len(set([s.lower() for s in resume_skills]) & set([s.lower() for s in jd_skills])) / len(jd_skills)) if jd_skills else 0.0
            semantic_sim = compute_tfidf_similarity(resume_text, jd_text, vectorizer)
            score = (w_skills * coverage) + (w_sem * semantic_sim)
            match_pct = round(score * 100, 1)

            # missing skills
            missing = [s for s in jd_skills if s.lower() not in [rs.lower() for rs in resume_skills]]

            # results UI
            st.markdown("### ✅ Match Results")
            rleft, rright = st.columns([2,1])
            with rleft:
                st.metric("Match Score", f"{match_pct} %")
                st.write("**JD source/role used:**", jd_role_used)
                st.write("**Skill coverage:**", f"{round(coverage*100,1)}%")
                st.write("**Semantic similarity (TF-IDF):**", f"{round(semantic_sim*100,1)}%")
                st.markdown("**Skills detected in resume (from vocab):**")
                st.write(", ".join(resume_skills) if resume_skills else "None detected")
                st.markdown("**Missing / recommended skills to add:**")
                if missing:
                    for s in missing:
                        st.markdown(f"- **{s}**")
                else:
                    st.success("No missing skills detected (based on current JD subset).")
                st.markdown("**Suggested resume lines:**")
                for line in suggestion_lines(missing)[:8]:
                    st.markdown(f"- {line}")

            with rright:
                # progress + chart
                st.write("Skill coverage chart")
                matched_count = len([s for s in jd_skills if s.lower() in [rs.lower() for rs in resume_skills]])
                missing_count = len(missing)
                fig, ax = plt.subplots(figsize=(4,3))
                ax.bar(["Matched","Missing"], [matched_count, missing_count], color=['#16a34a','#ef4444'])
                ax.set_ylabel("Count")
                st.pyplot(fig)
                st.markdown("### Quick tips")
                st.write("- Use exact keywords from the JD in your resume (skills & tools).")
                st.write("- Add metrics to show impact (numbers, % improvements).")
                st.write("- For scanned resumes, run OCR first.")

# -------------------- Action: Find suitable job roles --------------------
else:
    st.markdown("### 🧭 Find suitable job roles for my resume")
    if not uploaded_file:
        st.info("Upload a PDF resume above to detect suitable roles.")
    else:
        st.write("Scoring available roles against your resume...")
        scores = []
        for role, skills in role_skills_map.items():
            jd_text = ", ".join(skills)
            coverage = len(set(skills) & set(extract_skills_from_text(resume_text, skills_vocab))) / (len(skills) if skills else 1)
            sim = compute_tfidf_similarity(resume_text, jd_text, vectorizer)
            sc = 0.6 * coverage + 0.4 * sim
            scores.append((sc, role, coverage, sim, skills))
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        top_n = st.slider("How many suggestions to show", 1, min(10, max(1, len(scores))), 5)
        for sc, role, coverage, sim, skills in scores[:top_n]:
            with st.expander(f"{role} — match {round(sc*100,1)}%"):
                st.write("Required skills:", ", ".join(skills))
                st.write("Skill coverage:", f"{round(coverage*100,1)}%")
                st.write("Semantic similarity:", f"{round(sim*100,1)}%")
                missing = [s for s in skills if s.lower() not in [rs.lower() for rs in extract_skills_from_text(resume_text, skills_vocab)]]
                if missing:
                    st.write("Missing skills (top):", ", ".join(missing[:6]))
                    for l in suggestion_lines(missing[:6]):
                        st.markdown(f"- {l}")
                else:
                    st.success("Resume closely matches this role!")

# -------------------- Footer --------------------
st.markdown("---")