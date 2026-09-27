# app/services/tracking/state_machine.py
"""Transactional application state and persistent queue storage.

The state machine extends the existing ``applications`` table without
removing columns used by the existing dashboard.  Schema changes, queue
upserts, state transitions, and daily-budget reservations are committed
as single SQLite transactions so a crash cannot leave an application row
updated without its corresponding event (or vice versa).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.services.jobs.agent_schemas import (
    ApplicationState,
    InvalidTransition,
    NormalizedJob,
    assert_valid_transition,
    fingerprint_job,
)
from app.services.tracking import tracker as tracker_module
from app.services.tracking.tracker import ApplicationTrackerService

logger = logging.getLogger(__name__)


# These jobs still belong in the autonomous queue.  All other known jobs are
# suppressed during rediscovery, but remain untouched in SQLite (notably the
# READY_TO_SUBMIT and failed/retry queues).
AUTOMATION_QUEUE_STATES = {
    ApplicationState.DISCOVERED,
    ApplicationState.ANALYZING,
    ApplicationState.MATCHED,
}

# A transition into APPLYING means browser work was allowed.  The terminal
# states are also counted for legacy rows created before reservation rows
# existed.
_BUDGET_EFFECTIVE_STATES = (
    ApplicationState.APPLYING.value,
    ApplicationState.READY_TO_SUBMIT.value,
    ApplicationState.SUBMITTED.value,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_key(company: str, title: str, location: str) -> str:
    from app.services.jobs.dedupe import compute_dedupe_key

    return compute_dedupe_key(
        NormalizedJob(
            job_id="migration",
            company=company or "",
            title=title or "",
            location=location or "",
        )
    )


def _url_fingerprint(company: str, title: str, location: str, url: str) -> str:
    from app.services.jobs.dedupe import canonicalize_url

    return fingerprint_job(
        company or "",
        title or "",
        location or "",
        url=canonicalize_url(url or ""),
    )


def _as_state(value: Any, default: ApplicationState) -> ApplicationState:
    try:
        return ApplicationState(str(value))
    except (TypeError, ValueError):
        return default


def _coerce_state_filter(
    states: Iterable[ApplicationState | str] | None,
) -> list[str]:
    if states is None:
        return []
    result: list[str] = []
    for state in states:
        if isinstance(state, ApplicationState):
            result.append(state.value)
        else:
            result.append(ApplicationState(str(state)).value)
    return list(dict.fromkeys(result))


class ApplicationStateMachine:
    _migrated = False
    _migrated_path: str | None = None
    _migrated_signature: tuple[int, int, int] | None = None
    _migration_lock = threading.RLock()

    @staticmethod
    def _database_signature(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return (int(getattr(stat, "st_ino", 0)), int(stat.st_size), int(stat.st_mtime_ns))

    # ------------------------------------------------------------------
    # Atomic/idempotent schema migration
    # ------------------------------------------------------------------
    @classmethod
    def _ensure_schema(cls) -> None:
        """Migrate the current SQLite database atomically and idempotently.

        ``BEGIN IMMEDIATE`` serializes competing processes before the schema
        is inspected, so process A's ALTER/CREATE operations are committed
        before process B re-checks the columns.

        Legacy rows may contain duplicate fingerprint/dedupe keys. The
        unique indexes are therefore temporarily removed before backfilling
        and duplicate cleanup, then recreated after the data is normalized.
        """
        db_file = Path(tracker_module.DB_PATH).resolve()
        db_path = str(db_file)
        try:
            signature = cls._database_signature(db_file)
        except OSError:
            signature = (0, 0, 0)
        if cls._migrated and cls._migrated_path == db_path and cls._migrated_signature == signature:
            return

        with cls._migration_lock:
            if (
                cls._migrated
                and cls._migrated_path == db_path
                and cls._migrated_signature == signature
            ):
                return

            ApplicationTrackerService.init_db()
            conn = ApplicationTrackerService._get_connection()
            try:
                conn.execute("PRAGMA busy_timeout = 30000")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("BEGIN IMMEDIATE")

                existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(applications)")}

                additions = {
                    "state": "TEXT DEFAULT 'DISCOVERED'",
                    "fingerprint": "TEXT",
                    "dedupe_key": "TEXT",
                    "match_score": "REAL",
                    "source": "TEXT",
                    "job_description": "TEXT",
                    "location": "TEXT",
                    "retry_count": "INTEGER DEFAULT 0",
                }

                for column, definition in additions.items():
                    if column not in existing_cols:
                        conn.execute(f"ALTER TABLE applications ADD COLUMN {column} {definition}")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS application_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id INTEGER NOT NULL,
                        from_state TEXT,
                        to_state TEXT NOT NULL,
                        reason TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (application_id) REFERENCES applications (id)
                    )
                    """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS application_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id INTEGER NOT NULL,
                        doc_type TEXT NOT NULL,
                        path TEXT NOT NULL,
                        version TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (application_id) REFERENCES applications (id)
                    )
                    """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS browser_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id INTEGER NOT NULL,
                        final_url TEXT,
                        screenshot_path TEXT,
                        success INTEGER,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (application_id) REFERENCES applications (id)
                    )
                    """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_application_reservations (
                        utc_date TEXT NOT NULL,
                        application_id INTEGER NOT NULL,
                        reserved_at TEXT NOT NULL,
                        PRIMARY KEY (utc_date, application_id),
                        FOREIGN KEY (application_id) REFERENCES applications (id)
                    )
                    """)

                # Rows created by the pre-state-machine tracker have no event
                # history. Translate their legacy dashboard status once so an
                # old "Saved" application is not accidentally enqueued for a
                # browser submission. Genuine queue rows always have their
                # DISCOVERED event because create/upsert is transactional.
                conn.execute("""
                    UPDATE applications
                    SET state = CASE UPPER(COALESCE(status, 'Saved'))
                        WHEN 'APPLIED' THEN 'SUBMITTED'
                        WHEN 'SUBMITTED' THEN 'SUBMITTED'
                        WHEN 'INTERVIEW' THEN 'INTERVIEW'
                        WHEN 'INTERVIEWING' THEN 'INTERVIEW'
                        WHEN 'OFFER' THEN 'OFFER'
                        WHEN 'REJECTED' THEN 'REJECTED'
                        WHEN 'WITHDRAWN' THEN 'WITHDRAWN'
                        ELSE 'SKIPPED'
                    END
                    WHERE NOT EXISTS (
                        SELECT 1 FROM application_events
                        WHERE application_events.application_id = applications.id
                    )
                    """)

                # IMPORTANT:
                # Older versions may already have these unique indexes.
                # Drop them temporarily so legacy rows can be backfilled even
                # when multiple existing rows resolve to the same key.
                conn.execute("DROP INDEX IF EXISTS idx_applications_fingerprint_unique")
                conn.execute("DROP INDEX IF EXISTS idx_applications_dedupe_key_unique")

                # Backfill the persistent cross-board key for rows created by
                # older versions.
                legacy_rows = conn.execute("""
                    SELECT id, company_name, job_title, location
                    FROM applications
                    WHERE dedupe_key IS NULL OR dedupe_key = ''
                    """).fetchall()

                for row in legacy_rows:
                    dedupe_key = _dedupe_key(
                        row["company_name"] or "",
                        row["job_title"] or "",
                        row["location"] or "",
                    )

                    conn.execute(
                        """
                        UPDATE applications
                        SET dedupe_key = ?
                        WHERE id = ?
                        """,
                        (dedupe_key, row["id"]),
                    )

                # Backfill missing fingerprints as well. This protects
                # databases created before fingerprint persistence existed.
                fingerprint_rows = conn.execute("""
                    SELECT id, company_name, job_title, location, job_url
                    FROM applications
                    WHERE fingerprint IS NULL OR fingerprint = ''
                    """).fetchall()

                for row in fingerprint_rows:
                    fingerprint = _url_fingerprint(
                        row["company_name"] or "",
                        row["job_title"] or "",
                        row["location"] or "",
                        row["job_url"] or "",
                    )

                    conn.execute(
                        """
                        UPDATE applications
                        SET fingerprint = ?
                        WHERE id = ?
                        """,
                        (fingerprint, row["id"]),
                    )

                # Resolve fingerprint collisions before recreating the unique
                # index.  The soft company/title/location key is deliberately
                # non-unique because separate requisitions may share it.
                cls._clear_legacy_duplicate_keys(conn)

                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_applications_fingerprint_unique
                    ON applications (fingerprint)
                    WHERE fingerprint IS NOT NULL
                      AND fingerprint <> ''
                    """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_applications_dedupe_key
                    ON applications (dedupe_key)
                    """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_applications_fingerprint
                    ON applications (fingerprint)
                    """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_application_events_state_time
                    ON application_events (to_state, created_at)
                    """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_daily_reservations_date
                    ON daily_application_reservations (utc_date)
                    """)

                conn.commit()

            except Exception:
                conn.rollback()
                raise

            finally:
                conn.close()

            cls._migrated = True
            cls._migrated_path = db_path
            try:
                cls._migrated_signature = cls._database_signature(db_file)
            except OSError:
                cls._migrated_signature = signature

    @staticmethod
    def _clear_legacy_duplicate_keys(conn: sqlite3.Connection) -> None:
        """Make old duplicate fingerprints unique without deleting records.

        Soft company/title/location keys are intentionally not deduplicated;
        they are only a candidate signal.  All rows and histories remain
        available for audit/manual recovery.
        """
        rows = conn.execute("""
            SELECT id, COALESCE(state, 'DISCOVERED') AS state,
                   COALESCE(created_at, '') AS created_at,
                   fingerprint, dedupe_key
            FROM applications
            WHERE (fingerprint IS NOT NULL AND fingerprint <> '')
               OR (dedupe_key IS NOT NULL AND dedupe_key <> '')
            ORDER BY id
            """).fetchall()

        terminal = {
            ApplicationState.SKIPPED,
            ApplicationState.DUPLICATE,
            ApplicationState.VERIFIED,
            ApplicationState.INTERVIEW,
            ApplicationState.REJECTED,
            ApplicationState.OFFER,
            ApplicationState.WITHDRAWN,
        }

        def priority(row: sqlite3.Row) -> tuple[int, str, int]:
            state = _as_state(row["state"], ApplicationState.DISCOVERED)
            rank = 0 if state in AUTOMATION_QUEUE_STATES else (2 if state in terminal else 1)
            return rank, str(row["created_at"]), int(row["id"])

        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            key = row["fingerprint"]
            if key:
                groups.setdefault(str(key), []).append(row)
        for duplicates in groups.values():
            if len(duplicates) < 2:
                continue
            keeper = min(duplicates, key=priority)
            for row in duplicates:
                if int(row["id"]) != int(keeper["id"]):
                    conn.execute(
                        "UPDATE applications SET fingerprint = NULL WHERE id = ?",
                        (row["id"],),
                    )

    # ------------------------------------------------------------------
    # Creation / lookup
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        company_name: str,
        job_title: str,
        job_url: str,
        fingerprint: str,
        source: str = "",
        match_score: Optional[float] = None,
        job_description: str = "",
        location: str = "",
        dedupe_key: str = "",
    ) -> dict[str, Any]:
        """Insert a job once, or idempotently enrich its existing queue row.

        The state is deliberately not reset on an upsert.  In particular, a
        rediscovered READY_TO_SUBMIT or failed job must not be moved back to
        DISCOVERED and processed again.
        """
        cls._ensure_schema()
        fingerprint = str(fingerprint or "").strip()
        dedupe_key = str(dedupe_key or "").strip() or _dedupe_key(
            company_name or "", job_title or "", location or ""
        )

        conn = ApplicationTrackerService._get_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            existing: Optional[sqlite3.Row] = None
            if fingerprint:
                existing = conn.execute(
                    "SELECT * FROM applications WHERE fingerprint = ? ORDER BY id LIMIT 1",
                    (fingerprint,),
                ).fetchone()
            if existing is None and job_url:
                existing = conn.execute(
                    """
                    SELECT * FROM applications
                    WHERE company_name = ? AND job_title = ? AND job_url = ?
                    ORDER BY id LIMIT 1
                    """,
                    (company_name, job_title, job_url),
                ).fetchone()

            if existing is not None:
                app_id = int(existing["id"])
                effective_source = source
                if source and existing["source"]:
                    from app.services.jobs.dedupe import source_rank

                    if source_rank(source) >= source_rank(existing["source"]):
                        effective_source = ""
                conn.execute(
                    """
                    UPDATE applications SET
                        company_name = COALESCE(NULLIF(?, ''), company_name),
                        job_title = COALESCE(NULLIF(?, ''), job_title),
                        job_url = COALESCE(NULLIF(?, ''), job_url),
                        location = COALESCE(NULLIF(?, ''), location),
                        source = COALESCE(NULLIF(?, ''), source),
                        job_description = CASE
                            WHEN ? <> ''
                                 AND length(?) > COALESCE(length(job_description), 0)
                            THEN ?
                            ELSE job_description
                        END,
                        fingerprint = CASE
                            WHEN fingerprint IS NULL OR length(fingerprint) <> 64
                            THEN COALESCE(NULLIF(?, ''), fingerprint)
                            ELSE fingerprint
                        END,
                        dedupe_key = COALESCE(NULLIF(dedupe_key, ''), NULLIF(?, '')),
                        match_score = COALESCE(?, match_score)
                    WHERE id = ?
                    """,
                    (
                        company_name,
                        job_title,
                        job_url,
                        location,
                        effective_source,
                        job_description,
                        job_description,
                        job_description,
                        fingerprint,
                        dedupe_key,
                        match_score,
                        app_id,
                    ),
                )
                row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO applications (
                        company_name, job_title, job_url, status, state,
                        fingerprint, dedupe_key, source, match_score,
                        job_description, location, retry_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        company_name,
                        job_title,
                        job_url,
                        ApplicationState.DISCOVERED.value,
                        ApplicationState.DISCOVERED.value,
                        fingerprint or None,
                        dedupe_key,
                        source,
                        match_score,
                        job_description,
                        location,
                    ),
                )
                app_id = int(cursor.lastrowid)
                cls._insert_event(
                    conn,
                    app_id,
                    None,
                    ApplicationState.DISCOVERED,
                    "discovered",
                    _now_iso(),
                )
                row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()

            conn.commit()
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @classmethod
    def get(cls, application_id: int) -> Optional[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
        return dict(row) if row else None

    @classmethod
    def _known_values(
        cls,
        column: str,
        states: Iterable[ApplicationState | str] | None = None,
        exclude_states: Iterable[ApplicationState | str] | None = None,
    ) -> set[str]:
        if column not in {"fingerprint", "dedupe_key"}:
            raise ValueError(f"Unsupported fingerprint column: {column}")
        cls._ensure_schema()
        params: list[str] = []
        clauses = [f"{column} IS NOT NULL", f"{column} <> ''"]
        state_values = _coerce_state_filter(states)
        if state_values:
            clauses.append(f"state IN ({','.join('?' for _ in state_values)})")
            params.extend(state_values)
        excluded = _coerce_state_filter(exclude_states)
        if excluded:
            clauses.append(f"state NOT IN ({','.join('?' for _ in excluded)})")
            params.extend(excluded)
        with ApplicationTrackerService._get_connection() as conn:
            rows = conn.execute(
                f"SELECT {column} FROM applications WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
        return {str(row[0]) for row in rows if row[0]}

    @classmethod
    def known_fingerprints(
        cls,
        states: Iterable[ApplicationState | str] | None = None,
        exclude_states: Iterable[ApplicationState | str] | None = None,
    ) -> set[str]:
        """Persisted primary fingerprints, optionally filtered by state."""
        return cls._known_values("fingerprint", states=states, exclude_states=exclude_states)

    @classmethod
    def known_dedupe_keys(
        cls,
        states: Iterable[ApplicationState | str] | None = None,
        exclude_states: Iterable[ApplicationState | str] | None = None,
    ) -> set[str]:
        """Persisted cross-board keys, optionally filtered by state."""
        return cls._known_values("dedupe_key", states=states, exclude_states=exclude_states)

    @classmethod
    def find_by_fingerprint(cls, fingerprint: str) -> Optional[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM applications WHERE fingerprint = ? ORDER BY id LIMIT 1",
                (fingerprint,),
            ).fetchone()
        return dict(row) if row else None

    @classmethod
    def find_by_dedupe_key(cls, dedupe_key: str) -> Optional[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM applications WHERE dedupe_key = ? ORDER BY id LIMIT 1",
                (dedupe_key,),
            ).fetchone()
        return dict(row) if row else None

    @classmethod
    def list_by_state(cls, state: ApplicationState) -> list[dict[str, Any]]:
        return cls.list_by_states((state,))

    @classmethod
    def list_by_states(cls, states: Iterable[ApplicationState | str]) -> list[dict[str, Any]]:
        state_values = _coerce_state_filter(states)
        if not state_values:
            return []
        cls._ensure_schema()
        placeholders = ",".join("?" for _ in state_values)
        with ApplicationTrackerService._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM applications
                WHERE state IN ({placeholders})
                ORDER BY created_at DESC, id DESC
                """,
                state_values,
            ).fetchall()
        return [dict(row) for row in rows]

    @classmethod
    def list_automation_queue(cls) -> list[dict[str, Any]]:
        # Oldest work gets first access to the daily budget.
        return list(reversed(cls.list_by_states(AUTOMATION_QUEUE_STATES)))

    @classmethod
    def update_match_score(cls, application_id: int, score: float) -> None:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            conn.execute(
                "UPDATE applications SET match_score = ? WHERE id = ?",
                (float(score), application_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    @classmethod
    def transition(
        cls,
        application_id: int,
        to_state: ApplicationState,
        reason: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        """Validate and commit a transition plus its event atomically."""
        cls._ensure_schema()
        conn = ApplicationTrackerService._get_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No application with id={application_id}")

            from_state = _as_state(row["state"], ApplicationState.DISCOVERED)
            if from_state == to_state:
                conn.commit()
                return dict(row)

            if not force:
                try:
                    assert_valid_transition(from_state, to_state)
                except InvalidTransition:
                    logger.error(
                        "Rejected illegal transition for application %s: %s -> %s",
                        application_id,
                        from_state.value,
                        to_state.value,
                    )
                    raise

            conn.execute(
                "UPDATE applications SET state = ?, status = ? WHERE id = ?",
                (to_state.value, to_state.value, application_id),
            )
            cls._insert_event(
                conn,
                application_id,
                from_state,
                to_state,
                reason,
                _now_iso(),
            )
            updated = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        logger.info(
            "Application %s: %s -> %s (%s)",
            application_id,
            from_state.value,
            to_state.value,
            reason,
        )
        return dict(updated)

    @classmethod
    def begin_retry(
        cls,
        application_id: int,
        reason: str = "automatic retry",
        daily_limit: int | None = None,
        utc_day: str | date | datetime | None = None,
    ) -> dict[str, Any] | None:
        """Claim a failed row for retry, optionally with a UTC-day budget slot.

        The state change, retry counter, event, and optional budget
        reservation share one transaction. ``None`` means the row was not
        claimed because no capacity remained.
        """
        if daily_limit is not None and daily_limit < 0:
            raise ValueError("daily_limit must be non-negative")
        cls._ensure_schema()
        conn = ApplicationTrackerService._get_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No application with id={application_id}")
            from_state = _as_state(row["state"], ApplicationState.DISCOVERED)
            assert_valid_transition(from_state, ApplicationState.RETRYING)

            if daily_limit is not None:
                day = cls._utc_day(utc_day)
                cls._seed_daily_reservations(conn, day)
                already_reserved = conn.execute(
                    """
                    SELECT 1 FROM daily_application_reservations
                    WHERE utc_date = ? AND application_id = ?
                    """,
                    (day, application_id),
                ).fetchone()
                if already_reserved is not None or cls._daily_count_for(conn, day) >= daily_limit:
                    conn.commit()
                    return None
                conn.execute(
                    """
                    INSERT INTO daily_application_reservations (
                        utc_date, application_id, reserved_at
                    ) VALUES (?, ?, ?)
                    """,
                    (day, application_id, _now_iso()),
                )

            conn.execute(
                """
                UPDATE applications
                SET state = ?, status = ?, retry_count = COALESCE(retry_count, 0) + 1
                WHERE id = ?
                """,
                (ApplicationState.RETRYING.value, ApplicationState.RETRYING.value, application_id),
            )
            cls._insert_event(
                conn,
                application_id,
                from_state,
                ApplicationState.RETRYING,
                reason,
                _now_iso(),
            )
            updated = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            conn.commit()
            return dict(updated)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        application_id: int,
        from_state: Optional[ApplicationState],
        to_state: ApplicationState,
        reason: str,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO application_events (
                application_id, from_state, to_state, reason, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                application_id,
                from_state.value if from_state else None,
                to_state.value,
                reason,
                created_at,
            ),
        )

    @classmethod
    def _log_event(
        cls,
        application_id: int,
        from_state: Optional[ApplicationState],
        to_state: ApplicationState,
        reason: str,
    ) -> None:
        """Compatibility helper; normal transitions insert in their transaction."""
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            cls._insert_event(conn, application_id, from_state, to_state, reason, _now_iso())
            conn.commit()

    @classmethod
    def increment_retry_count(cls, application_id: int) -> int:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            conn.execute(
                "UPDATE applications SET retry_count = COALESCE(retry_count, 0) + 1 WHERE id = ?",
                (application_id,),
            )
            row = conn.execute(
                "SELECT retry_count FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            conn.commit()
        return int(row["retry_count"]) if row else 0

    @classmethod
    def history(cls, application_id: int) -> list[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM application_events WHERE application_id = ? ORDER BY id",
                (application_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # UTC daily application budget
    # ------------------------------------------------------------------
    @staticmethod
    def _utc_day(value: str | date | datetime | None = None) -> str:
        if value is None:
            current = datetime.now(timezone.utc)
        elif isinstance(value, datetime):
            current = (
                value.astimezone(timezone.utc)
                if value.tzinfo
                else value.replace(tzinfo=timezone.utc)
            )
        elif isinstance(value, date):
            current = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
        else:
            try:
                current = datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise ValueError("UTC day must use YYYY-MM-DD format") from exc
        return current.date().isoformat()

    @classmethod
    def _seed_daily_reservations(cls, conn: sqlite3.Connection, day: str) -> None:
        # Both SQLite's space-separated UTC timestamps and our ISO-8601 values
        # begin with the same UTC calendar date.
        placeholders = ",".join("?" for _ in _BUDGET_EFFECTIVE_STATES)
        conn.execute(
            f"""
            INSERT OR IGNORE INTO daily_application_reservations (
                utc_date, application_id, reserved_at
            )
            SELECT ?, application_events.application_id,
                   MIN(application_events.created_at)
            FROM application_events
            JOIN applications ON applications.id = application_events.application_id
            WHERE application_events.to_state IN ({placeholders})
              AND substr(application_events.created_at, 1, 10) = ?
              AND application_events.application_id IS NOT NULL
            GROUP BY application_events.application_id
            """,
            [day, *_BUDGET_EFFECTIVE_STATES, day],
        )

    @staticmethod
    def _daily_count_for(
        conn: sqlite3.Connection,
        day: str,
    ) -> int:
        placeholders = ",".join("?" for _ in _BUDGET_EFFECTIVE_STATES)
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS count FROM (
                SELECT application_id
                FROM daily_application_reservations
                WHERE utc_date = ?
                UNION
                SELECT DISTINCT application_events.application_id
                FROM application_events
                JOIN applications
                  ON applications.id = application_events.application_id
                WHERE substr(application_events.created_at, 1, 10) = ?
                  AND application_events.to_state IN ({placeholders})
            )
            """,
            (day, day, *_BUDGET_EFFECTIVE_STATES),
        ).fetchone()
        return int(row["count"])

    @classmethod
    def count_daily_applications(cls, utc_day: str | date | datetime | None = None) -> int:
        """Count distinct applications reserved or attempted on a UTC day."""
        cls._ensure_schema()
        day = cls._utc_day(utc_day)
        conn = ApplicationTrackerService._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cls._seed_daily_reservations(conn, day)
            count = cls._daily_count_for(conn, day)
            conn.commit()
            return count
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @classmethod
    def reserve_daily_application(
        cls,
        application_id: int,
        daily_limit: int,
        utc_day: str | date | datetime | None = None,
    ) -> bool:
        """Atomically reserve one UTC-day slot for an application.

        The primary key makes each application/day reservation unique, while
        ``BEGIN IMMEDIATE`` prevents two workers from concurrently observing
        the same remaining capacity.  An existing reservation returns
        ``False`` so a second worker cannot execute the same queued job.
        """
        if daily_limit < 0:
            raise ValueError("daily_limit must be non-negative")
        cls._ensure_schema()
        day = cls._utc_day(utc_day)
        conn = ApplicationTrackerService._get_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            cls._seed_daily_reservations(conn, day)
            application = conn.execute(
                "SELECT 1 FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            if application is None:
                raise ValueError(f"No application with id={application_id}")
            existing = conn.execute(
                """
                SELECT 1 FROM daily_application_reservations
                WHERE utc_date = ? AND application_id = ?
                """,
                (day, application_id),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return False

            if cls._daily_count_for(conn, day) >= daily_limit:
                conn.commit()
                return False

            conn.execute(
                """
                INSERT INTO daily_application_reservations (
                    utc_date, application_id, reserved_at
                ) VALUES (?, ?, ?)
                """,
                (day, application_id, _now_iso()),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Documents & browser session metadata
    # ------------------------------------------------------------------
    @classmethod
    def record_document(
        cls, application_id: int, doc_type: str, path: str, version: str = ""
    ) -> None:
        """Append a document version; retries never erase prior files."""
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO application_documents (
                    application_id, doc_type, path, version, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (application_id, doc_type, path, version, _now_iso()),
            )
            conn.commit()

    @classmethod
    def documents(cls, application_id: int) -> list[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM application_documents WHERE application_id = ? ORDER BY id",
                (application_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @classmethod
    def record_browser_session(
        cls,
        application_id: int,
        final_url: str = "",
        screenshot_path: str = "",
        success: bool = False,
        error_message: str = "",
    ) -> None:
        """Capture one browser attempt's non-sensitive result metadata."""
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO browser_sessions (
                    application_id, final_url, screenshot_path, success,
                    error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    final_url,
                    screenshot_path,
                    int(success),
                    error_message,
                    _now_iso(),
                ),
            )
            conn.commit()

    @classmethod
    def browser_sessions(cls, application_id: int) -> list[dict[str, Any]]:
        cls._ensure_schema()
        with ApplicationTrackerService._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM browser_sessions WHERE application_id = ? ORDER BY id",
                (application_id,),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "AUTOMATION_QUEUE_STATES",
    "ApplicationStateMachine",
]
