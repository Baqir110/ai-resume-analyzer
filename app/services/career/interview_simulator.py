"""Interview practice simulator with feedback."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InterviewSimulator:
    """Simulates live interview with AI feedback."""

    def __init__(self):
        self.interview_history = []
        self.score_history = []

    async def generate_interview_question(
        self,
        family: str,
        context: str,
    ) -> dict[str, Any]:
        """Generate interview question.

        Args:
            family: 'technical', 'behavioral', 'product', 'leadership', 'mlops_devops'
            context: User's background/role context

        Returns:
            Question with category and difficulty
        """
        prompts = {
            "technical": "Generate a {difficulty} technical interview question for a {context} role. "
            "Ask about system design, coding, or debugging.",
            "behavioral": "Generate a behavioral question using STAR format for {context} role.",
            "product": "Generate a product thinking question for {context} role.",
            "leadership": "Generate a leadership/team management question for {context} role.",
            "mlops_devops": "Generate an MLOps/DevOps specific question for {context} role.",
        }

        # Placeholder - in production would call LLM
        return {
            "question_id": 1,
            "family": family,
            "difficulty": "medium",
            "question": prompts[family].format(difficulty="medium", context=context),
            "follow_up_topics": ["scalability", "trade-offs", "implementation"],
        }

    async def evaluate_answer(
        self,
        question_id: int,
        user_answer: str,
        expected_topics: list[str],
    ) -> dict[str, Any]:
        """Evaluate user's interview answer.

        Returns:
            Feedback with score and improvement suggestions
        """
        score = self._compute_answer_score(
            user_answer,
            expected_topics,
        )

        return {
            "question_id": question_id,
            "score": score,
            "max_score": 10,
            "feedback": {
                "strengths": self._identify_strengths(user_answer),
                "gaps": self._identify_gaps(user_answer, expected_topics),
                "suggestions": self._generate_suggestions(score),
            },
            "next_action": "generate_follow_up" if score >= 6 else "retry_question",
        }

    def _compute_answer_score(self, answer: str, topics: list[str]) -> float:
        """Score answer 0-10 based on topic coverage and depth."""
        score = 0
        answer_lower = answer.lower()

        # Check topic coverage (5 points max)
        for topic in topics:
            if topic.lower() in answer_lower:
                score += min(5 / len(topics), 2)

        # Check answer length/depth (3 points)
        if len(answer) > 500:
            score += 2
        elif len(answer) > 200:
            score += 1

        # Check structure/clarity (2 points)
        if any(word in answer_lower for word in ["first", "then", "finally", "therefore"]):
            score += 1.5

        return round(min(score, 10), 1)

    def _identify_strengths(self, answer: str) -> list[str]:
        """Identify strong points in answer."""
        strengths = []

        if len(answer) > 300:
            strengths.append("Good depth and detail")

        if any(word in answer.lower() for word in ["trade-off", "pros", "cons"]):
            strengths.append("Considers trade-offs")

        if any(word in answer.lower() for word in ["example", "specifically", "case"]):
            strengths.append("Uses concrete examples")

        return strengths or ["Answer provided"]

    def _identify_gaps(self, answer: str, topics: list[str]) -> list[str]:
        """Identify missing elements."""
        gaps = []
        answer_lower = answer.lower()

        for topic in topics:
            if topic.lower() not in answer_lower:
                gaps.append(f"Missing discussion of {topic}")

        if len(answer) < 200:
            gaps.append("Answer could be more detailed")

        return gaps

    def _generate_suggestions(self, score: float) -> list[str]:
        """Generate improvement suggestions based on score."""
        if score >= 8:
            return [
                "Excellent answer! Ready for follow-up questions.",
                "Consider adding edge cases or failure scenarios.",
            ]
        elif score >= 6:
            return [
                "Good foundation. Add more specific examples.",
                "Expand on the trade-offs and constraints.",
            ]
        else:
            return [
                "Structure your answer with clear steps.",
                "Start with the problem, then propose solution.",
                "Include reasoning for your decisions.",
            ]

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current interview session."""
        if not self.score_history:
            return {"status": "no_scores"}

        avg_score = sum(self.score_history) / len(self.score_history)

        return {
            "total_questions": len(self.score_history),
            "average_score": round(avg_score, 1),
            "max_score": max(self.score_history),
            "min_score": min(self.score_history),
            "pass_rate": sum(1 for s in self.score_history if s >= 6)
            / len(self.score_history)
            * 100,
        }
