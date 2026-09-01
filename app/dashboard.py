import io
import os
import time
import requests
import streamlit as st
from docx import Document

# ==============================================================================
# CONFIG
# ==============================================================================
API_BASE = os.environ.get("RESUME_API_BASE", "http://localhost:8000/api/v1/resume")
REQUEST_TIMEOUT = 60  # seconds

ENDPOINTS = {
    "analyze": f"{API_BASE}/analyze",
    "full_docx": f"{API_BASE}/generate-full",
    "german_pdf": f"{API_BASE}/generate-german-cv",
    "german_tex": f"{API_BASE}/generate-tex-cv",
}

PROVIDER_LABELS = {
    "gemini": "Google Gemini (Free API Key)",
    "groq": "Groq Cloud (Super Fast / Free Tier)",
    "openrouter": "OpenRouter (Free Model Options)",
    "deepseek": "DeepSeek V3/R1 (Low-Cost / Free)",
    "ollama": "Ollama (100% Free & Local)",
    "openai": "OpenAI ChatGPT",
    "claude": "Anthropic Claude",
}

TEMPLATE_LABELS = {
    "german_corporate": "🇩🇪 Corporate Slate Navy (Single-Column)",
    "german_modern": "🇩🇪 Modern Two-Column (Sidebar Layout)",
    "german_classic": "🇩🇪 Classic Conservative Single-Column",
    "international_ats": "🌐 International English ATS Standard",
}

st.set_page_config(page_title="AI Resume & CV Optimization Hub", page_icon="🎯", layout="wide")

# ==============================================================================
# LIGHT STYLING (chips, step badges)
# ==============================================================================
st.markdown("""
<style>
.chip {display:inline-block; padding:4px 12px; margin:3px; border-radius:999px; font-size:0.85rem; font-weight:500;}
.chip-good {background:#DCFCE7; color:#166534;}
.chip-bad {background:#FEE2E2; color:#991B1B;}
.step-badge {display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;
  border-radius:50%; background:#E5E7EB; color:#374151; font-weight:600; font-size:0.85rem; margin-right:8px;}
.step-badge.active {background:#2563EB; color:white;}
.step-badge.done {background:#16A34A; color:white;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================
def chips(items, kind="good"):
    if not items:
        st.caption("None found.")
        return
    cls = "chip-good" if kind == "good" else "chip-bad"
    html = "".join(f'<span class="chip {cls}">{i}</span>' for i in items)
    st.markdown(html, unsafe_allow_html=True)


def api_request(url: str, data: dict, files: dict, spinner_text: str):
    """POST with a friendly status widget; auto-retries once on 429/503. Never returns a bare exception state."""
    with st.status(spinner_text, expanded=False) as status:
        try:
            res = requests.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.ConnectionError:
            status.update(label="❌ Couldn't reach the backend (localhost:8000).", state="error")
            return None
        except requests.exceptions.Timeout:
            status.update(label="❌ Request timed out.", state="error")
            return None

        if res.status_code in (429, 503):
            wait_time = 20 if res.status_code == 503 else 50
            status.update(label=f"⏳ Server busy ({res.status_code}). Retrying in {wait_time}s...", state="running")
            progress = st.progress(0)
            for i in range(wait_time):
                time.sleep(1)
                progress.progress((i + 1) / wait_time)
            progress.empty()
            status.update(label="Retrying now...", state="running")
            try:
                res = requests.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                status.update(label="❌ Retry failed.", state="error")
                return None

        if res.status_code == 200:
            status.update(label="✅ Done", state="complete")
        else:
            status.update(label=f"❌ Failed ({res.status_code})", state="error")
    return res


def error_detail(res) -> str:
    if res is None:
        return "Could not connect to the backend service."
    try:
        return str(res.json().get("detail", res.text))
    except Exception:
        return res.text or f"HTTP {res.status_code}"


def current_upload():
    return st.session_state.get("uploaded_file_data")


def build_files_payload():
    fname, fval, ftype = current_upload()
    return {"resume_file": (fname, fval, ftype)}


def generate_txt_export(result_data: dict) -> bytes:
    lines = [
        "=" * 50,
        "AI RESUME & CV OPTIMIZATION REPORT".center(50),
        "=" * 50,
        f"ATS Match Score: {result_data.get('ats_match_score')}%",
        f"Keyword Density Score: {result_data.get('keyword_density_score')}%",
        "\n[MATCHING SKILLS]",
        ", ".join(result_data.get("matching_skills", [])) or "None",
        "\n[MISSING SKILLS]",
        ", ".join(result_data.get("missing_skills", [])) or "None",
        "\n" + "=" * 50,
        "SUGGESTIONS & REWRITES".center(50),
        "=" * 50 + "\n",
    ]
    for idx, tip in enumerate(result_data.get("improvement_suggestions", []), 1):
        lines.append(f"{idx}. {tip}\n")
    return "\n".join(lines).encode("utf-8")


def generate_docx_export(result_data: dict) -> bytes:
    doc = Document()
    doc.add_heading("AI Resume Optimization Report", level=0)

    doc.add_heading("Overview Metrics", level=1)
    p = doc.add_paragraph()
    p.add_run("ATS Match Score: ").bold = True
    p.add_run(f"{result_data.get('ats_match_score')}%\n")
    p.add_run("Keyword Density: ").bold = True
    p.add_run(f"{result_data.get('keyword_density_score')}%\n")

    doc.add_heading("Skill Gap Analysis", level=1)
    p_skills = doc.add_paragraph()
    p_skills.add_run("Matching Skills: ").bold = True
    p_skills.add_run(f"{', '.join(result_data.get('matching_skills', []))}\n")
    p_skills.add_run("Missing Skills: ").bold = True
    p_skills.add_run(f"{', '.join(result_data.get('missing_skills', []))}\n")

    doc.add_heading("Suggestions & Rewritten Bullets", level=1)
    for tip in result_data.get("improvement_suggestions", []):
        if tip.startswith("🤖"):
            doc.add_heading("AI Rewritten Content", level=2)
            doc.add_paragraph(tip.replace("🤖 ", ""))
        else:
            doc.add_paragraph(tip, style="List Bullet")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def reset_all():
    for k in ("last_analysis", "uploaded_file_data", "job_desc"):
        st.session_state.pop(k, None)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🤖 AI Engine")
    provider = st.selectbox(
        "AI Provider",
        options=list(PROVIDER_LABELS.keys()),
        format_func=lambda x: PROVIDER_LABELS[x],
    )

    api_key = ""
    if provider != "ollama":
        api_key = st.text_input(f"{provider.upper()} API Key", type="password",
                                 help="Leave blank if configured on the server.")
    else:
        st.info("Ollama runs locally — no API key needed.")

    st.markdown("---")
    st.caption("💡 Gemini and Groq offer generous free tiers.")

    if st.session_state.get("last_analysis"):
        st.markdown("---")
        if st.button("🔄 Start Over", use_container_width=True):
            reset_all()
            st.rerun()

# ==============================================================================
# HEADER + PROGRESS STEPPER
# ==============================================================================
st.title("🎯 AI Resume & CV Matcher Engine")
st.caption("Analyze ATS compatibility, close skill gaps, and export a job-tailored resume.")

step = 1 if not st.session_state.get("last_analysis") else 2
s1 = "done" if step > 1 else "active"
s2 = "active" if step == 2 else ""
st.markdown(
    f'<span class="step-badge {s1}">1</span> Upload & Analyze &nbsp;&nbsp;→&nbsp;&nbsp; '
    f'<span class="step-badge {s2}">2</span> Review & Export',
    unsafe_allow_html=True,
)
st.write("")

# ==============================================================================
# STEP 1 — INPUT
# ==============================================================================
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        job_desc = st.text_area("Target Job Description", height=220,
                                 placeholder="Paste the job requirements and duties here...",
                                 value=st.session_state.get("job_desc", ""))
    with col2:
        uploaded_file = st.file_uploader("Upload Current Resume", type=["pdf", "docx", "txt"])
        if uploaded_file:
            size_kb = len(uploaded_file.getvalue()) / 1024
            st.caption(f"📎 {uploaded_file.name} · {size_kb:.0f} KB")

    ready = bool(uploaded_file and job_desc.strip())
    if not ready:
        st.caption("⬆️ Add a resume and job description to continue.")

    if st.button("🚀 Analyze ATS Compatibility & Skill Gaps", use_container_width=True,
                  type="primary", disabled=not ready):
        files = {"resume_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"job_description": job_desc, "provider": provider, "api_key": api_key}

        res = api_request(ENDPOINTS["analyze"], data=data, files=files,
                           spinner_text=f"Running ATS analysis with {provider.upper()}...")

        if res is None:
            st.error("Could not reach the backend service. Check that it's running on port 8000.")
        elif res.status_code == 200:
            st.session_state["last_analysis"] = res.json()
            st.session_state["uploaded_file_data"] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            st.session_state["job_desc"] = job_desc
            st.rerun()
        else:
            st.error(f"Analysis failed: {error_detail(res)}")

# ==============================================================================
# STEP 2 — RESULTS
# ==============================================================================
if "last_analysis" in st.session_state:
    result = st.session_state["last_analysis"]
    st.write("")

    with st.container(border=True):
        st.subheader("📊 Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("ATS Match Score", f"{result.get('ats_match_score', 0)}%")
        m2.metric("Keyword Density", f"{result.get('keyword_density_score', 0)}%")
        m3.metric("Missing Skills", len(result.get("missing_skills", [])))

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**✅ Matching Skills**")
            chips(result.get("matching_skills", []), "good")
        with sc2:
            st.markdown("**❌ Missing Skills**")
            chips(result.get("missing_skills", []), "bad")

    with st.container(border=True):
        st.subheader("💡 Actionable Improvements")
        for tip in result.get("improvement_suggestions", []):
            if tip.startswith("🤖"):
                st.markdown(tip)
            else:
                st.info(tip)

    rec = result.get("recommendation") or {}
    rec_format = rec.get("recommended_format", "german_corporate")
    if rec:
        with st.container(border=True):
            st.subheader("🧭 AI Format Recommendation")
            st.success(f"**{rec.get('label', 'Standard')}** — {rec.get('reason', '')}")

    st.write("")

    # -- Export analysis report --------------------------------------------
    with st.expander("📥 Download Analysis Report (TXT / DOCX)"):
        e1, e2 = st.columns(2)
        e1.download_button("📄 .TXT", generate_txt_export(result), "resume_analysis_report.txt",
                            "text/plain", use_container_width=True)
        e2.download_button(
            "📝 .DOCX", generate_docx_export(result), "resume_analysis_report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    st.divider()
    st.header("✨ Generate a Complete Tailored CV")

    tab1, tab2 = st.tabs(["📄 Standard ATS Resume (DOCX)", "🇩🇪 German Lebenslauf (PDF / TEX)"])

    with tab1:
        st.caption("Clean, single-column Word document formatted for global ATS systems.")
        if st.button("🪄 Build Tailored Resume", use_container_width=True, type="primary"):
            data = {"job_description": st.session_state["job_desc"], "provider": provider, "api_key": api_key}
            res = api_request(ENDPOINTS["full_docx"], data=data, files=build_files_payload(),
                               spinner_text=f"Building tailored resume with {provider.upper()}...")
            if res and res.status_code == 200:
                st.success("🎉 Resume ready!")
                st.download_button(
                    "📥 Download Resume (.DOCX)", res.content, "Optimized_Tailored_Resume.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            else:
                st.error(f"Generation failed: {error_detail(res)}")

    with tab2:
        st.caption("Select a layout, then compile a PDF or grab the raw LaTeX source.")
        template_options = list(TEMPLATE_LABELS.keys())
        default_idx = template_options.index(rec_format) if rec_format in template_options else 0
        selected_layout = st.selectbox(
            "Layout Style", options=template_options, index=default_idx,
            format_func=lambda x: TEMPLATE_LABELS[x],
        )

        b1, b2 = st.columns(2)
        common_data = {
            "job_description": st.session_state["job_desc"],
            "layout_style": selected_layout,
            "template_style": selected_layout,
            "provider": provider,
            "api_key": api_key,
        }

        with b1:
            if st.button("📄 Build PDF", use_container_width=True, type="primary"):
                res = api_request(ENDPOINTS["german_pdf"], data=common_data, files=build_files_payload(),
                                   spinner_text=f"Compiling PDF ({selected_layout})...")
                if res and res.status_code == 200:
                    st.success("🎉 PDF ready!")
                    st.download_button(
                        "📥 Download (.PDF)", res.content, f"Lebenslauf_{selected_layout}.pdf",
                        "application/pdf", use_container_width=True,
                    )
                else:
                    st.error(f"Compilation error: {error_detail(res)}")

        with b2:
            if st.button("🛠️ Build TEX Source", use_container_width=True):
                res = api_request(ENDPOINTS["german_tex"], data=common_data, files=build_files_payload(),
                                   spinner_text="Generating LaTeX source...")
                if res and res.status_code == 200:
                    st.success("🎉 TEX file ready!")
                    st.download_button(
                        "📥 Download (.TEX)", res.content, f"Lebenslauf_{selected_layout}.tex",
                        "text/plain", use_container_width=True,
                    )
                else:
                    st.error(f"Generation error: {error_detail(res)}")