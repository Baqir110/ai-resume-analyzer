"""Resume score explainability with detailed breakdown."""
import logging
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class ScoreExplainabilityEngine:
    """Explain ATS scores with detailed section-by-section breakdown."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, lowercase=True)
    
    def explain_score(
        self,
        resume_text: str,
        job_description: str,
    ) -> Dict[str, Any]:
        """Generate explainable score breakdown.
        
        Returns:
            Dict with:
            - overall_score: 0-100
            - section_scores: Section-by-section breakdown
            - keyword_heatmap: Importance of each keyword
            - missing_high_impact_keywords: Priority improvements
            - interpretation: Narrative explanation
        """
        # Extract sections
        resume_sections = self._extract_sections(resume_text)
        
        # Compute TF-IDF similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([job_description, resume_text])
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity_score = 0
        
        # Score each section
        section_scores = {}
        for section_name, section_text in resume_sections.items():
            if section_text.strip():
                try:
                    section_matrix = self.vectorizer.fit_transform([job_description, section_text])
                    section_sim = cosine_similarity(section_matrix[0:1], section_matrix[1:2])[0][0]
                    section_scores[section_name] = {
                        'score': round(section_sim * 100, 1),
                        'text_length': len(section_text),
                    }
                except:
                    section_scores[section_name] = {'score': 0, 'text_length': 0}
        
        # Extract keywords and importance
        keyword_importance = self._compute_keyword_importance(
            job_description,
            resume_text,
        )
        
        # Identify missing high-impact keywords
        missing_keywords = self._identify_missing_keywords(
            job_description,
            resume_text,
            keyword_importance,
        )
        
        overall_score = round(similarity_score * 100, 1)
        
        return {
            'overall_score': overall_score,
            'score_interpretation': self._interpret_score(overall_score),
            'section_scores': section_scores,
            'keyword_heatmap': keyword_importance,
            'missing_high_impact_keywords': missing_keywords,
            'strengths': self._identify_strengths(section_scores),
            'weaknesses': self._identify_weaknesses(section_scores, missing_keywords),
        }
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract resume sections."""
        sections = {
            'skills_section': '',
            'experience_section': '',
            'education_section': '',
            'projects_section': '',
            'other_section': '',
        }
        
        lines = text.split('\n')
        current_section = 'other_section'
        
        section_keywords = {
            'skills_section': ['skill', 'technologies', 'technical'],
            'experience_section': ['experience', 'work', 'employment', 'position'],
            'education_section': ['education', 'degree', 'university', 'school'],
            'projects_section': ['project', 'portfolio', 'github'],
        }
        
        for line in lines:
            line_lower = line.lower()
            
            for section, keywords in section_keywords.items():
                if any(kw in line_lower for kw in keywords):
                    current_section = section
                    break
            
            sections[current_section] += line + '\n'
        
        return sections
    
    def _compute_keyword_importance(
        self,
        jd: str,
        resume: str,
    ) -> Dict[str, float]:
        """Compute importance score for each keyword in JD."""
        try:
            vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
            vectorizer.fit_transform([jd])
            
            keywords = vectorizer.get_feature_names_out()
            importance = {}
            
            for keyword in keywords:
                jd_count = jd.lower().count(keyword)
                resume_count = resume.lower().count(keyword)
                score = (resume_count / max(jd_count, 1)) * 100
                importance[keyword] = min(score, 100)
            
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
        except:
            return {}
    
    def _identify_missing_keywords(
        self,
        jd: str,
        resume: str,
        importance: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Identify missing high-impact keywords."""
        missing = []
        
        for keyword, score in list(importance.items())[:10]:
            if keyword.lower() not in resume.lower():
                missing.append({
                    'keyword': keyword,
                    'impact_score': score,
                    'suggestion': f'Consider adding \"{keyword}\" to match job requirements',
                })
        
        return missing
    
    def _identify_strengths(self, section_scores: Dict[str, Dict]) -> List[str]:
        """Identify resume strengths."""
        strengths = []
        for section, data in section_scores.items():
            score = data.get('score', 0)
            if score >= 70:
                section_name = section.replace('_', ' ').title()
                strengths.append(f"{section_name}: Strong match ({score}%)")
        return strengths
    
    def _identify_weaknesses(self, section_scores: Dict, missing: List) -> List[str]:
        """Identify resume weaknesses."""
        weaknesses = []
        for section, data in section_scores.items():
            score = data.get('score', 0)
            if score < 50:
                section_name = section.replace('_', ' ').title()
                weaknesses.append(f"{section_name}: Weak match ({score}%)")
        
        if missing:
            weaknesses.append(f"Missing {len(missing)} high-impact keywords")
        
        return weaknesses
    
    def _interpret_score(self, score: float) -> str:
        """Generate interpretation text for score."""
        if score >= 85:
            return "Excellent match - highly qualified candidate"
        elif score >= 70:
            return "Good match - consider with minor gaps"
        elif score >= 50:
            return "Moderate match - gaps exist"
        elif score >= 30:
            return "Weak match - significant skill gaps"
        else:
            return "Poor match - major gaps in qualifications"
