"""Collaborative resume review and feedback system."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "resume_feedback.db"


class CollaborativeFeedbackManager:
    """Manages collaborative resume review with comment threads."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    section TEXT NOT NULL,  -- 'experience', 'skills', 'education', 'projects'
                    line_number INTEGER,
                    author TEXT NOT NULL,  -- Mentor/coach email
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    UNIQUE(resume_id, section, line_number, author)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL,
                    author TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    suggestion TEXT,  -- Proposed fix
                    category TEXT,  -- 'clarity', 'impact', 'technical', 'grammar', 'format'
                    severity TEXT DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES feedback_threads(id)
                )
            """)

            conn.commit()

    def create_feedback_thread(
        self,
        resume_id: int,
        section: str,
        line_number: int | None,
        author: str,
    ) -> int:
        """Create a new feedback thread.

        Returns:
            Thread ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_threads
                (resume_id, section, line_number, author)
                VALUES (?, ?, ?, ?)
                """,
                (resume_id, section, line_number, author),
            )
            conn.commit()
            return cursor.lastrowid

    def add_comment(
        self,
        thread_id: int,
        author: str,
        comment: str,
        suggestion: str | None = None,
        category: str = "clarity",
        severity: str = "medium",
    ) -> int:
        """Add comment to feedback thread.

        Returns:
            Comment ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_comments
                (thread_id, author, comment, suggestion, category, severity)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, author, comment, suggestion, category, severity),
            )
            conn.commit()
            return cursor.lastrowid

    def get_resume_feedback(
        self,
        resume_id: int,
        unresolved_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all feedback for a resume."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    t.id as thread_id,
                    t.section,
                    t.line_number,
                    t.author as thread_author,
                    t.resolved,
                    t.created_at,
                    COUNT(c.id) as comment_count,
                    GROUP_CONCAT(c.severity) as severities
                FROM feedback_threads t
                LEFT JOIN feedback_comments c ON t.id = c.thread_id
                WHERE t.resume_id = ?
            """

            params = [resume_id]

            if unresolved_only:
                query += " AND t.resolved = 0"

            query += " GROUP BY t.id ORDER BY t.created_at DESC"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_thread_comments(
        self,
        thread_id: int,
    ) -> list[dict[str, Any]]:
        """Get all comments in a thread."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM feedback_comments
                WHERE thread_id = ?
                ORDER BY created_at ASC
                """,
                (thread_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def resolve_thread(self, thread_id: int) -> bool:
        """Mark thread as resolved."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE feedback_threads SET resolved = 1 WHERE id = ?",
                    (thread_id,),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to resolve thread: {e}")
            return False

    def get_feedback_summary(
        self,
        resume_id: int,
    ) -> dict[str, Any]:
        """Get summary of feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Overall stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT t.id) as total_threads,
                    SUM(CASE WHEN t.resolved = 1 THEN 1 ELSE 0 END) as resolved_threads,
                    COUNT(c.id) as total_comments
                FROM feedback_threads t
                LEFT JOIN feedback_comments c ON t.id = c.thread_id
                WHERE t.resume_id = ?
                """,
                (resume_id,),
            )

            stats = dict(cursor.fetchone() or {})

            # By severity
            cursor = conn.execute(
                """
                SELECT severity, COUNT(*) as count
                FROM feedback_comments c
                JOIN feedback_threads t ON c.thread_id = t.id
                WHERE t.resume_id = ?
                GROUP BY severity
                """,
                (resume_id,),
            )

            by_severity = {row[0]: row[1] for row in cursor.fetchall()}

            # By category
            cursor = conn.execute(
                """
                SELECT category, COUNT(*) as count
                FROM feedback_comments c
                JOIN feedback_threads t ON c.thread_id = t.id
                WHERE t.resume_id = ?
                GROUP BY category
                """,
                (resume_id,),
            )

            by_category = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_threads": stats.get("total_threads", 0),
                "resolved_threads": stats.get("resolved_threads", 0),
                "total_comments": stats.get("total_comments", 0),
                "by_severity": by_severity,
                "by_category": by_category,
                "completion_percentage": (
                    (stats.get("resolved_threads", 0) / max(stats.get("total_threads", 1), 1)) * 100
                ),
            }
