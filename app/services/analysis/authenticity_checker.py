"""Resume authenticity and plagiarism detection."""
import logging
from typing import Dict, List, Any, Tuple
import hashlib

logger = logging.getLogger(__name__)


class AuthenticityChecker:
    """Checks resume authenticity and detects plagiarism."""
    
    def __init__(self):
        self.flagged_patterns = self._initialize_flagged_patterns()
        self.common_phrases = self._initialize_common_phrases()
    
    async def check_authenticity(
        self,
        resume_text: str,
        check_plagiarism: bool = True,
    ) -> Dict[str, Any]:
        """Check resume for authenticity and plagiarism.
        
        Returns:
            Authenticity report with score and flagged sections
        """
        plagiarism_report = await self._check_plagiarism(resume_text) if check_plagiarism else None
        generic_report = self._check_generic_language(resume_text)
        repetition_report = self._check_repetition(resume_text)
        pattern_report = self._check_suspicious_patterns(resume_text)
        
        authenticity_score = self._compute_authenticity_score(
            plagiarism_report,
            generic_report,
            repetition_report,
            pattern_report,
        )
        
        return {
            'authenticity_score': authenticity_score,
            'score_interpretation': self._interpret_score(authenticity_score),
            'plagiarism_analysis': plagiarism_report,
            'generic_language_analysis': generic_report,
            'repetition_analysis': repetition_report,
            'suspicious_patterns': pattern_report,
            'recommendation': self._generate_recommendation(authenticity_score),
            'risk_level': self._assess_risk_level(authenticity_score),
        }
    
    async def _check_plagiarism(
        self,
        resume_text: str,
    ) -> Dict[str, Any]:
        """Check for plagiarized content.
        
        In production, would integrate with Turnitin or similar service.
        """
        # Placeholder implementation
        sentences = resume_text.split('.')
        
        flagged_sections = []
        
        for sentence in sentences:
            if self._is_suspicious_sentence(sentence):
                flagged_sections.append({
                    'text': sentence.strip(),
                    'similarity': 0.85,
                    'potential_source': 'LinkedIn Resume Template #42',
                    'severity': 'high',
                })
        
        overall_plagiarism = len(flagged_sections) / max(len(sentences), 1)
        
        return {
            'plagiarism_percentage': round(overall_plagiarism * 100, 1),
            'flagged_sections': flagged_sections,
            'status': 'suspicious' if overall_plagiarism > 0.2 else 'clean',
        }
    
    def _check_generic_language(
        self,
        resume_text: str,
    ) -> Dict[str, Any]:
        """Check for overuse of generic language."""
        resume_lower = resume_text.lower()
        
        generic_count = 0
        for phrase in self.common_phrases:
            generic_count += resume_lower.count(phrase.lower())
        
        total_words = len(resume_text.split())
        genericity_score = (generic_count / max(total_words, 1)) * 100
        
        flagged_phrases = []
        for phrase in self.common_phrases:
            if phrase.lower() in resume_lower:
                flagged_phrases.append(phrase)
        
        return {
            'genericity_score': round(min(genericity_score, 100), 1),
            'flagged_phrases': flagged_phrases[:10],
            'interpretation': 'High generic language' if genericity_score > 20 else 'Authentic language',
        }
    
    def _check_repetition(
        self,
        resume_text: str,
    ) -> Dict[str, Any]:
        """Check for excessive word repetition."""
        words = resume_text.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 4:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find most repeated words
        most_repeated = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        repetition_score = sum(count for _, count in most_repeated) / len(words) * 100
        
        return {
            'repetition_score': round(repetition_score, 1),
            'most_repeated_words': [
                {'word': word, 'count': count} for word, count in most_repeated
            ],
            'interpretation': 'High repetition' if repetition_score > 15 else 'Good variety',
        }
    
    def _check_suspicious_patterns(
        self,
        resume_text: str,
    ) -> List[Dict[str, Any]]:
        """Check for suspicious patterns."""
        suspicious = []
        
        # Check for unrealistic achievements
        unrealistic_patterns = [
            ('increased revenue', '1000%'),
            ('improved efficiency', '500%'),
            ('reduced costs', '99%'),
        ]
        
        for pattern, metric in unrealistic_patterns:
            if pattern.lower() in resume_text.lower() and metric in resume_text:
                suspicious.append({
                    'pattern': f"{pattern} by {metric}",
                    'type': 'unrealistic_claim',
                    'severity': 'high',
                    'suggestion': 'Verify these metrics with data',
                })
        
        # Check for AI-generated indicators
        ai_phrases = ['leveraging', 'synergizing', 'driving innovation', 'paradigm shift']
        ai_count = sum(1 for phrase in ai_phrases if phrase in resume_text.lower())
        
        if ai_count >= 3:
            suspicious.append({
                'pattern': 'Potential AI-generated content',
                'type': 'ai_generated',
                'severity': 'medium',
                'ai_phrases_detected': ai_count,
                'suggestion': 'Add more personal, specific details',
            })
        
        return suspicious
    
    def _compute_authenticity_score(
        self,
        plagiarism: Dict,
        generic: Dict,
        repetition: Dict,
        patterns: List,
    ) -> float:
        """Compute overall authenticity score (0-100, higher = more authentic)."""
        score = 100.0
        
        # Plagiarism impact
        if plagiarism:
            score -= plagiarism['plagiarism_percentage'] * 0.8
        
        # Generic language impact
        score -= generic['genericity_score'] * 0.4
        
        # Repetition impact
        score -= repetition['repetition_score'] * 0.3
        
        # Suspicious patterns impact
        score -= len(patterns) * 10
        
        return max(0, min(100, score))
    
    def _interpret_score(self, score: float) -> str:
        """Interpret authenticity score."""
        if score >= 85:
            return "Highly authentic - Original content with personal voice"
        elif score >= 70:
            return "Authentic - Mostly original with some template language"
        elif score >= 50:
            return "Moderately authentic - Mix of original and template content"
        elif score >= 30:
            return "Questionable - Significant generic or copied language detected"
        else:
            return "Likely plagiarized or AI-generated - Requires review"
    
    def _assess_risk_level(self, score: float) -> str:
        """Assess plagiarism/authenticity risk level."""
        if score >= 80:
            return 'low'
        elif score >= 60:
            return 'medium'
        elif score >= 40:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on score."""
        if score >= 85:
            return "Resume appears authentic. No action needed."
        elif score >= 70:
            return "Resume looks good. Consider adding more specific, personal details."
        elif score >= 50:
            return "Review and personalize sections with generic language."
        else:
            return "Resume requires significant revision. Replace template language with genuine achievements."
    
    def _is_suspicious_sentence(self, sentence: str) -> bool:
        """Check if a sentence is suspicious."""
        # Placeholder - would use similarity matching in production
        sentence_lower = sentence.lower().strip()
        suspicious_starts = [
            'managed team of',
            'responsible for',
            'worked on',
        ]
        
        return any(s in sentence_lower for s in suspicious_starts)
    
    def _initialize_flagged_patterns(self) -> List[str]:
        """Initialize patterns that indicate plagiarism."""
        return [
            'core competencies',
            'results-driven',
            'self-starter',
            'team player',
            'detail-oriented',
        ]
    
    def _initialize_common_phrases(self) -> List[str]:
        """Initialize common generic phrases."""
        return [
            'Managed team of',
            'Responsible for',
            'Worked on',
            'Helped develop',
            'Contributed to',
            'Led initiative',
            'Spearheaded project',
            'Drove adoption of',
            'Leveraged skills',
            'Synergized with',
            'Optimized process',
            'Improved efficiency',
            'Increased revenue',
            'Reduced costs',
            'Delivered value',
        ]
