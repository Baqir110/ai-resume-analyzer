"""Skill progression tracking and roadmap generation."""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "skill_progression.db"


class SkillProgressionTracker:
    """Tracks skill growth over time and generates learning roadmaps."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skill_progression (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    first_detected TIMESTAMP,
                    last_detected TIMESTAMP,
                    job_count INTEGER DEFAULT 1,
                    proficiency_level TEXT DEFAULT 'beginner',
                    learning_resources TEXT,  -- JSON
                    UNIQUE(user_id, skill)
                )
            """)
            conn.commit()
    
    def track_skill(
        self,
        user_id: str,
        skill: str,
        proficiency: str = 'intermediate',
    ) -> None:
        """Track a skill for the user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO skill_progression
                (user_id, skill, first_detected, last_detected, proficiency_level)
                VALUES (
                    ?,
                    ?,
                    COALESCE(
                        (SELECT first_detected FROM skill_progression WHERE user_id = ? AND skill = ?),
                        CURRENT_TIMESTAMP
                    ),
                    CURRENT_TIMESTAMP,
                    ?
                )
                """,
                (user_id, skill, user_id, skill, proficiency),
            )
            conn.commit()
    
    def get_user_skills(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all skills tracked for user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM skill_progression WHERE user_id = ? ORDER BY last_detected DESC",
                (user_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
    
    async def generate_learning_roadmap(
        self,
        current_skills: List[str],
        target_role: str,
        months_available: int = 6,
    ) -> Dict[str, Any]:
        """Generate personalized learning roadmap.
        
        Returns:
            Roadmap with prioritized skills, learning paths, and timeline
        """
        # Placeholder - in production would use skill taxonomy + LLM
        gap_skills = self._compute_skill_gaps(
            current_skills,
            target_role,
        )
        
        prioritized = self._prioritize_skills(
            gap_skills,
            months_available,
        )
        
        return {
            'target_role': target_role,
            'current_skills': current_skills,
            'gap_skills': prioritized,
            'timeline_months': months_available,
            'estimated_hours': sum(s['learning_time_hours'] for s in prioritized),
            'milestones': self._create_milestones(prioritized, months_available),
            'learning_paths': [
                {
                    'skill': skill['skill'],
                    'priority': skill['priority'],
                    'resources': self._get_learning_resources(skill['skill']),
                }
                for skill in prioritized
            ],
        }
    
    def _compute_skill_gaps(
        self,
        current: List[str],
        target_role: str,
    ) -> List[Dict[str, Any]]:
        """Compute skills needed for target role not in current set."""
        role_skills = self._get_role_requirements(target_role)
        
        gaps = []
        current_lower = [s.lower() for s in current]
        
        for skill, data in role_skills.items():
            if skill.lower() not in current_lower:
                gaps.append({
                    'skill': skill,
                    'demand_score': data['demand'],
                    'learning_time_hours': data['learning_hours'],
                    'popularity': data['job_postings'],
                })
        
        return sorted(gaps, key=lambda x: x['demand_score'], reverse=True)
    
    def _prioritize_skills(
        self,
        gap_skills: List[Dict],
        months: int,
    ) -> List[Dict[str, Any]]:
        """Prioritize skills based on time available and importance."""
        total_hours = months * 30  # Assume 30 hours/month learning
        current_hours = 0
        prioritized = []
        
        for skill in gap_skills:
            if current_hours + skill['learning_time_hours'] <= total_hours:
                current_hours += skill['learning_time_hours']
                skill['priority'] = 'high' if len(prioritized) < 3 else 'medium'
                prioritized.append(skill)
        
        return prioritized
    
    def _create_milestones(
        self,
        skills: List[Dict],
        total_months: int,
    ) -> List[Dict[str, Any]]:
        """Create monthly milestones."""
        milestones = []
        skills_per_month = max(1, len(skills) // total_months)
        
        for i, skill in enumerate(skills[:total_months]):
            month = (i // skills_per_month) + 1
            milestones.append({
                'month': month,
                'skill': skill['skill'],
                'goal': f"Master {skill['skill']}",
                'success_criteria': f"Complete 3 projects using {skill['skill']}",
            })
        
        return milestones
    
    def _get_role_requirements(self, role: str) -> Dict[str, Dict[str, Any]]:
        """Get skills required for a role.
        
        In production, this would query a skills taxonomy DB.
        """
        role_maps = {
            'senior data engineer': {
                'Apache Spark': {'demand': 0.95, 'learning_hours': 40, 'job_postings': 1200},
                'SQL': {'demand': 0.98, 'learning_hours': 20, 'job_postings': 2000},
                'Cloud (AWS/GCP)': {'demand': 0.90, 'learning_hours': 60, 'job_postings': 1800},
                'Kubernetes': {'demand': 0.75, 'learning_hours': 50, 'job_postings': 900},
                'Python': {'demand': 0.92, 'learning_hours': 30, 'job_postings': 1500},
            },
            'software engineer': {
                'System Design': {'demand': 0.85, 'learning_hours': 50, 'job_postings': 800},
                'DSA': {'demand': 0.88, 'learning_hours': 40, 'job_postings': 1000},
                'Distributed Systems': {'demand': 0.70, 'learning_hours': 60, 'job_postings': 600},
                'Microservices': {'demand': 0.75, 'learning_hours': 45, 'job_postings': 700},
            },
        }
        
        return role_maps.get(role.lower(), {})
    
    def _get_learning_resources(self, skill: str) -> List[Dict[str, str]]:
        """Get learning resources for skill.
        
        In production, would fetch from curated database.
        """
        resources = {
            'Apache Spark': [
                {'type': 'course', 'name': 'Spark by Example', 'url': 'https://www.example.com'},
                {'type': 'book', 'name': 'Learning Spark', 'url': 'https://www.example.com'},
            ],
            'Python': [
                {'type': 'course', 'name': 'Complete Python 3', 'url': 'https://www.example.com'},
                {'type': 'project', 'name': 'Build 5 Python Projects', 'url': 'https://github.com'},
            ],
        }
        
        return resources.get(skill, [])
