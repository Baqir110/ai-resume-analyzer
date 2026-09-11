"""CV Generation Page - Step 2 of workflow."""
import streamlit as st
import io
from app.dashboard.helpers import make_api_request, get_api_base, build_file_payload
from app.dashboard.components import render_error_alert


TEMPLATE_LABELS = {
    'auto': '🎯 Auto-detect (match JD language)',
    'international_ats': 'International English ATS (Single-Page)',
    'academic': 'Academic / Research (Serif, Education-first)',
    'technical_lead': 'Technical Lead (Open Source + Speaking)',
    'hr_executive_gold': 'HR Gold Standard (Executive)',
    'german_corporate': 'Corporate Slate Navy',
    'german_minimal_ats': 'German Minimal ATS (Single-Column)',
    'german_modern': 'Modern Two-Column',
    'german_classic': 'German Classic Single-Column PDF',
    'standard': 'Standard ATS Single-Column',
}


def render_cv_generation_page():
    """Render CV generation options."""
    if not st.session_state.get('last_analysis'):
        st.info("Complete Step 1 (Analyze) first")
        return
    
    st.header("Generate your tailored CV")
    
    api_base = get_api_base()
    tab1, tab2 = st.tabs(
        ["📄 Standard ATS Resume", "🇩🇪 German Lebenslauf / PDF Options"]
    )
    
    with tab1:
        st.caption(
            "Layout is auto-selected from the job description's language "
            "(English JD → compact single-page English layout)."
        )
        
        if st.button(
            "🪄 Build Tailored Resume",
            width="stretch",
            type="primary",
            key="build_docx_btn",
        ):
            response = make_api_request(
                f"{api_base}/api/v1/resume/generate-full",
                data={
                    'job_description': st.session_state.get('job_desc', ''),
                    'layout_style': 'auto',
                },
                files=build_file_payload(),
                timeout=120,
            )
            
            if response and response.status_code == 200:
                st.download_button(
                    "📥 Download Resume (.DOCX)",
                    response.content,
                    "Optimized_Tailored_Resume.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
            else:
                render_error_alert(response)
    
    with tab2:
        template_options = list(TEMPLATE_LABELS.keys())
        selected_layout = st.selectbox(
            "Layout Style",
            template_options,
            index=0,
            format_func=lambda x: TEMPLATE_LABELS[x],
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Build PDF", width="stretch", type="primary", key="build_pdf_btn"):
                response = make_api_request(
                    f"{api_base}/api/v1/resume/generate-german-cv",
                    data={
                        'job_description': st.session_state.get('job_desc', ''),
                        'layout_style': selected_layout,
                    },
                    files=build_file_payload(),
                )
                
                if response and response.status_code == 200:
                    st.download_button(
                        "📥 Download PDF",
                        response.content,
                        f"CV_{selected_layout}.pdf",
                        "application/pdf",
                        width="stretch",
                    )
                else:
                    render_error_alert(response)
        
        with col2:
            if st.button("🛠️ Build TEX Source", width="stretch", key="build_tex_btn"):
                response = make_api_request(
                    f"{api_base}/api/v1/resume/generate-tex-cv",
                    data={
                        'job_description': st.session_state.get('job_desc', ''),
                        'layout_style': selected_layout,
                    },
                    files=build_file_payload(),
                )
                
                if response and response.status_code == 200:
                    st.download_button(
                        "📥 Download TEX",
                        response.content,
                        f"CV_{selected_layout}.tex",
                        "text/plain",
                        width="stretch",
                    )
                else:
                    render_error_alert(response)
