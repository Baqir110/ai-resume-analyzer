import os
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

DB_PATH = Path(os.getenv("DATABASE_PATH", "data/applications.db"))


class ApplicationTrackerService:
    @classmethod
    def _get_connection(cls):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def init_db(cls):
        """Initializes the application tracking table."""
        with cls._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    job_title TEXT NOT NULL,
                    job_url TEXT,
                    ats_score INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'Saved',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
                """)
            conn.commit()

    @classmethod
    def create_application(
        cls,
        company_name: str,
        job_title: str,
        job_url: Optional[str] = "",
        ats_score: int = 0,
        status: str = "Saved",
        notes: Optional[str] = "",
    ) -> Dict[str, Any]:
        cls.init_db()
        with cls._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO applications (company_name, job_title, job_url, ats_score, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (company_name, job_title, job_url, ats_score, status, notes),
            )
            conn.commit()
            app_id = cursor.lastrowid
            return cls.get_application(app_id)

    @classmethod
    def list_applications(cls) -> List[Dict[str, Any]]:
        cls.init_db()
        with cls._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM applications ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_application(cls, app_id: int) -> Optional[Dict[str, Any]]:
        cls.init_db()
        with cls._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM applications WHERE id = ?", (app_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @classmethod
    def update_status(cls, app_id: int, status: str) -> bool:
        cls.init_db()
        with cls._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE applications SET status = ? WHERE id = ?", (status, app_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    @classmethod
    def delete_application(cls, app_id: int) -> bool:
        cls.init_db()
        with cls._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM applications WHERE id = ?", (app_id,))
            conn.commit()
            return cursor.rowcount > 0
