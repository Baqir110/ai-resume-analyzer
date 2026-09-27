# app/services/jobs/scheduler.py
"""Continuous-operation scheduler (Section 29).

Deliberately simple: an asyncio loop checking UTC wall-clock time against
configured HH:MM slots, not a separate Celery/cron dependency. Good
enough for a single-user agent; swap for a real task queue if this
grows into a multi-tenant service.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

AsyncCallback = Callable[[], Awaitable[None]]


@dataclass
class ScheduleConfig:
    discover_times: list[str] = field(default_factory=lambda: ["08:00", "12:00", "16:00"])
    apply_times: list[str] = field(default_factory=lambda: ["10:00", "14:00"])
    poll_interval_seconds: int = 30

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        for value in (*self.discover_times, *self.apply_times):
            try:
                parsed = datetime.strptime(value, "%H:%M")
            except (TypeError, ValueError) as exc:
                raise ValueError("Scheduled times must use HH:MM in UTC") from exc
            if parsed.strftime("%H:%M") != value:
                raise ValueError("Scheduled times must use HH:MM in UTC")
        self.discover_times = sorted(set(self.discover_times))
        self.apply_times = sorted(set(self.apply_times))


class AgentScheduler:
    def __init__(
        self,
        config: ScheduleConfig,
        on_discover: AsyncCallback,
        on_apply: AsyncCallback,
        clock: Callable[[], datetime] | None = None,
    ):
        self.config = config
        self.on_discover = on_discover
        self.on_apply = on_apply
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fired_today: set[str] = set()
        initial = self._clock()
        if initial.tzinfo is None:
            initial = initial.replace(tzinfo=timezone.utc)
        self._last_day = initial.astimezone(timezone.utc).date()

    async def run_forever(self) -> None:
        logger.info(
            "Scheduler started. Discover at %s, apply at %s",
            self.config.discover_times,
            self.config.apply_times,
        )
        while True:
            await self._tick()
            await asyncio.sleep(min(self.config.poll_interval_seconds, 60))

    async def _tick(self) -> None:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        if now.date() != self._last_day:
            self._fired_today.clear()
            self._last_day = now.date()

        current_hhmm = now.strftime("%H:%M")

        if current_hhmm in self.config.discover_times:
            key = f"discover-{current_hhmm}"
            if key not in self._fired_today:
                self._fired_today.add(key)
                await self._run_safely("discover", self.on_discover)

        if current_hhmm in self.config.apply_times:
            key = f"apply-{current_hhmm}"
            if key not in self._fired_today:
                self._fired_today.add(key)
                await self._run_safely("apply", self.on_apply)

    async def _run_safely(self, label: str, callback: AsyncCallback) -> None:
        # Section 19/38: a scheduled run failing must never kill the loop.
        try:
            logger.info("Scheduled task '%s' starting", label)
            await callback()
            logger.info("Scheduled task '%s' finished", label)
        except Exception:  # noqa: BLE001
            logger.exception("Scheduled task '%s' failed; will retry next slot", label)


__all__ = ["ScheduleConfig", "AgentScheduler"]
