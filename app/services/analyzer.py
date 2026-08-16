import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard technical keyword dictionary for analysis
TECH_SKILLS = [
    "python", "fastapi", "docker", "kubernetes", "sql", "postgresql", 
    "git", "aws", "react", "javascript", "scikit-learn", "pandas", 
    "numpy", "linux", "ci/cd", "terraform", "pytest", "rest"
]

def analyze_resume_content(resume_text: str, job_description: str) -> dict:
    # 1. Calculate TF-IDF Cosine Similarity (ATS Match Score)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_score = round(float(similarity) * 100, 2)

    # 2. Extract Skills
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    matching_skills = []
    missing_skills = []

    for skill in TECH_SKILLS:
        in_jd = re.search(r'\b' + re.escape(skill) + r'\b', jd_lower)
        in_resume = re.search(r'\b' + re.escape(skill) + r'\b', resume_lower)

        if in_jd:
            if in_resume:
                matching_skills.append(skill.capitalize())
            else:
                missing_skills.append(skill.capitalize())

    # 3. Keyword Density Score
    keyword_density = round(len(matching_skills) / max(1, (len(matching_skills) + len(missing_skills))) * 100, 2)

    # 4. Improvement Suggestions
    suggestions = []
    if match_score < 70:
        suggestions.append("Consider tailoring your resume wording closer to the job description terminology.")
    if missing_skills:
        suggestions.append(f"Add missing core keywords to your skills section: {', '.join(missing_skills[:5])}.")
    if len(resume_text.split()) < 200:
        suggestions.append("Resume body text is relatively short; expand on project accomplishments and metrics.")
    if not suggestions:
        suggestions.append("Your resume aligns very well with this job description specification.")

    return {
        "ats_match_score": match_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "keyword_density_score": keyword_density,
        "improvement_suggestions": suggestions
    }