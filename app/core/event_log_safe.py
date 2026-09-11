"""Thread-safe event logging with file locking."""
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import uuid

try:
    import fcntl  # Unix only
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    import msvcrt  # Windows

logger = logging.getLogger(__name__)


class SafeEventLogger:
    """Thread-safe event logger with file locking."""
    
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.lock = threading.RLock()  # Reentrant lock for same thread
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        kind: str,
        event: str,
        request_id: Optional[str] = None,
        **metadata,
    ) -> str:
        """Log an event with thread-safe file locking.
        
        Args:
            kind: Event category (pipeline, parse, analysis, llm, etc.)
            event: Event name (pipeline_started, parse_completed, etc.)
            request_id: Request ID for correlation (auto-generated if not provided)
            **metadata: Additional fields to log
        
        Returns:
            The request_id for correlation
        """
        if not request_id:
            request_id = str(uuid.uuid4())
        
        event_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'kind': kind,
            'event': event,
            'request_id': request_id,
            **metadata,
        }
        
        with self.lock:
            self._write_event(event_data)
        
        return request_id
    
    def _write_event(self, event_data: Dict[str, Any]) -> None:
        """Write event to log file with file locking."""
        try:
            with open(self.log_path, 'a') as f:
                # Acquire lock (Unix)
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(json.dumps(event_data) + '\n')
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                # Windows fallback (simpler locking)
                else:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        f.write(json.dumps(event_data) + '\n')
                        f.flush()
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except (OSError, IOError):
                        # Fallback: just write without lock
                        f.write(json.dumps(event_data) + '\n')
                        f.flush()
        
        except Exception as e:
            logger.error(f"Failed to write event log: {e}")
    
    def get_recent_events(
        self,
        limit: int = 100,
        kind_filter: Optional[str] = None,
    ) -> list:
        """Read recent events from log.
        
        Args:
            limit: Maximum number of events to return
            kind_filter: Optional filter by event kind
        
        Returns:
            List of event dictionaries
        """
        events = []
        
        try:
            if not self.log_path.exists():
                return events
            
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
            
            # Process in reverse to get most recent first
            for line in reversed(lines):
                if len(events) >= limit:
                    break
                
                try:
                    event = json.loads(line.strip())
                    if kind_filter is None or event.get('kind') == kind_filter:
                        events.append(event)
                except json.JSONDecodeError:
                    continue
        
        except Exception as e:
            logger.error(f"Failed to read event log: {e}")
        
        return events
    
    def clear_events(self) -> None:
        """Clear the event log file."""
        try:
            with self.lock:
                self.log_path.write_text('')
        except Exception as e:
            logger.error(f"Failed to clear event log: {e}")
