"""Resume version control and tracking."""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3
import json

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "resume_versions.db"


class ResumeVersionManager:
    """Manages multiple resume versions with diff tracking."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resume_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    job_id TEXT,
                    original_text TEXT NOT NULL,
                    optimized_text TEXT NOT NULL,
                    ats_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,  -- JSON array
                    notes TEXT,
                    FOREIGN KEY (job_id) REFERENCES job_applications(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_diffs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL,
                    diff_summary TEXT,
                    word_level_diff TEXT,  -- JSON
                    FOREIGN KEY (version_id) REFERENCES resume_versions(id)
                )
            """)
            
            conn.commit()
    
    def save_version(
        self,
        user_id: str,
        original_text: str,
        optimized_text: str,
        ats_score: int,
        job_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Save a new resume version.
        
        Returns:
            Version ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO resume_versions
                (user_id, job_id, original_text, optimized_text, ats_score, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    job_id,
                    original_text,
                    optimized_text,
                    ats_score,
                    json.dumps(tags or []),
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_versions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get user's resume versions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM resume_versions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_version(self, version_id: int) -> Optional[Dict[str, Any]]:
        """Get specific version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM resume_versions WHERE id = ?",
                (version_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def compare_versions(
        self,
        version_id_1: int,
        version_id_2: int,
    ) -> Dict[str, Any]:
        """Compare two resume versions.
        
        Returns:
            Comparison with diffs
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)
        
        if not v1 or not v2:
            return {'error': 'Version not found'}
        
        return {
            'version_1': {
                'id': v1['id'],
                'created_at': v1['created_at'],
                'ats_score': v1['ats_score'],
            },
            'version_2': {
                'id': v2['id'],
                'created_at': v2['created_at'],
                'ats_score': v2['ats_score'],
            },
            'score_improvement': v2['ats_score'] - v1['ats_score'],
            'word_additions': len(v2['optimized_text']) - len(v1['optimized_text']),
            'sample_diff': self._compute_diff(v1['optimized_text'], v2['optimized_text']),
        }
    
    def _compute_diff(self, text1: str, text2: str) -> List[str]:
        """Simple word-level diff."""
        import difflib
        
        words1 = text1.split()
        words2 = text2.split()
        
        diff = difflib.unified_diff(words1, words2, lineterm='')
        return list(diff)[:20]  # First 20 diff lines
    
    def delete_version(self, version_id: int) -> bool:
        """Delete a version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM resume_versions WHERE id = ?",
                    (version_id,),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
