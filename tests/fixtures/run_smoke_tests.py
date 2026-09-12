"""Smoke test — hits every endpoint and reports pass/fail."""

import io
from pathlib import Path

import requests

BASE = "http://localhost:8000/api/v1/resume"
FIXTURES = Path(__file__).parent / "fixtures"
RESUME = FIXTURES / "resume.txt"
JD = FIXTURES / "jd.txt"

results = []


def check(name, response, expect_keys=None):
    ok = response.status_code == 200
    detail = ""
    if ok and expect_keys:
        try:
            data = response.json()
            for key in expect_keys:
                if key not in str(data):
                    ok = False
                    detail = f"missing key '{key}'"
        except Exception as e:
            ok = False
            detail = str(e)[:60]
    if not ok:
        detail = detail or f"HTTP {response.status_code}"
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# 1. Health
check("health", requests.get(f"{BASE}/health"), ["ok"])
check("backend-status", requests.get(f"{BASE}/backend-status"), ["providers"])
check("model-catalog", requests.get(f"{BASE}/model-catalog"), ["models"])
check("usage-summary", requests.get(f"{BASE}/usage-summary"), ["summary"])
check("quota-status", requests.get(f"{BASE}/quota-status"), ["providers"])
check("processing-log", requests.get(f"{BASE}/processing-log"), ["logs"])
check("analytics-summary", requests.get(f"{BASE}/analytics/summary"), ["data"])
check("career-options", requests.get(f"{BASE}/career-options"), ["cover_letter_templates"])
check("tracker-list", requests.get(f"{BASE}/tracker/applications"), ["applications"])

# 2. Analyze
jd = JD.read_text()
with open(RESUME, "rb") as f:
    check(
        "analyze",
        requests.post(
            f"{BASE}/analyze",
            data={"job_description": jd},
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["ats_match_score", "matching_skills"],
    )

# 3. Score breakdown
with open(RESUME, "rb") as f:
    check(
        "score-breakdown",
        requests.post(
            f"{BASE}/score-breakdown",
            data={"job_description": jd},
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["breakdown"],
    )

# 4. Market insights
check(
    "market-insights",
    requests.get(
        f"{BASE}/market-insights",
        params={"role": "Platform Engineer", "location": "Berlin", "seniority": "senior"},
        timeout=60,
    ),
    ["insights"],
)

# 5. Skill roadmap
check(
    "skill-roadmap",
    requests.post(
        f"{BASE}/skill-roadmap",
        data={"current_skills": '["python"]', "target_role": "MLOps", "months_available": 6},
        timeout=120,
    ),
    ["roadmap"],
)

# 6. Authenticity
with open(RESUME, "rb") as f:
    check(
        "check-authenticity",
        requests.post(
            f"{BASE}/check-authenticity",
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["report"],
    )

# 7. Audit matrix
with open(RESUME, "rb") as f:
    check(
        "audit-matrix",
        requests.post(
            f"{BASE}/audit-matrix",
            data={"job_description": jd},
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["data"],
    )

# 8. Cover letter
with open(RESUME, "rb") as f:
    check(
        "cover-letter",
        requests.post(
            f"{BASE}/generate-cover-letter",
            data={"job_description": jd, "company_name": "Acme"},
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["data"],
    )

# 9. Interview prep
with open(RESUME, "rb") as f:
    check(
        "interview-prep",
        requests.post(
            f"{BASE}/interview-prep",
            data={"job_description": jd, "family": "technical"},
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["data"],
    )

# 10. LinkedIn
with open(RESUME, "rb") as f:
    check(
        "linkedin-optimize",
        requests.post(
            f"{BASE}/linkedin-optimize",
            files={"resume_file": ("r.txt", f, "text/plain")},
            timeout=120,
        ),
        ["data"],
    )

# 11. Diff preview
check(
    "diff-preview",
    requests.post(
        f"{BASE}/diff-preview",
        json={"original_bullets": ["a"], "optimized_bullets": ["b"]},
    ),
    ["diffs"],
)

# 12. DOCX generation
with open(RESUME, "rb") as f:
    r = requests.post(
        f"{BASE}/generate-full",
        data={"job_description": jd},
        files={"resume_file": ("r.txt", f, "text/plain")},
        timeout=180,
    )
    ok = r.status_code == 200 and r.content[:2] == b"PK"
    results.append(("generate-full-docx", "PASS" if ok else "FAIL", f"{len(r.content)} bytes"))
    print(f"[{'PASS' if ok else 'FAIL'}] generate-full-docx {len(r.content)} bytes")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[1] == "PASS")
print(f"TOTAL: {passed}/{len(results)} passed")
print("=" * 60)
for name, status, detail in results:
    print(f"{status:4}  {name:30}  {detail}")
