# AI Resume & CV Analyzer API

A production-grade NLP microservice built with FastAPI and Scikit-Learn that computes ATS compatibility scores, extracts skill gaps, and generates automated resume optimization insights.

---

## 🏗️ Architecture

```text
 Job Description + Resume File (PDF/DOCX/TXT)
                      │
                      ▼
             ┌─────────────────┐
             │   FastAPI API   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Document Parser │ ──► Extracts text (PyPDF / Python-Docx)
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Scikit-Learn   │ ──► TF-IDF Cosine Similarity & Keyword Extraction
             └────────┬────────┘
                      │
                      ▼
             ┌────────────────────────────────────────┐
             │ ATS Score, Missing Skills & Suggestions│
             └────────────────────────────────────────┘
```

---

## ⚡ Key Features

* **Multi-Format Parsing**: Extracts text seamlessly from PDF, DOCX, and TXT files.
* **NLP ATS Scoring**: Uses TF-IDF vectorization and cosine similarity to evaluate semantic match rates.
* **Skill Gap Analysis**: Identifies matching and missing technical skills against job descriptions.
* **Actionable Guidance**: Returns tailored optimization tips to improve candidate resume performance.

---

## 🚀 Quickstart

1. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```powershell
   python -m app.main
   ```
   Interactive Swagger UI: `http://127.0.0.1:8000/docs`

3. **Run Tests:**
   ```powershell
   python -m pytest
   ```
