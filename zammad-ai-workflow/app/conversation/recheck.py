"""Re-check a ticket later, in process, without blocking the Kafka handler."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from app.utils.logging import getLogger

logger = getLogger("zammad-ai.conversation.recheck")

Sleep = Callable[[float], Awaitable[object]]
Recheck = Callable[[int], Awaitable[None]]


class RecheckScheduler:
    """Keep at most one pending re-check per ticket, and serialise work on the same ticket.

    A restart loses pending re-checks; the next event for the ticket, or the periodic reconcile, picks the turn up.
    """

    def __init__(self, sleep: Sleep = asyncio.sleep) -> None:
        """Create a scheduler; `sleep` is injectable so tests need no real waits."""
        self._sleep: Sleep = sleep
        self._pending: dict[int, asyncio.Task[None]] = {}
        self._running: set[asyncio.Task[None]] = set()
        self._locks: dict[int, asyncio.Lock] = {}
        self._lock_users: dict[int, int] = {}

    def schedule(self, ticket_id: int, delay_seconds: float, recheck: Recheck) -> None:
        """Run `recheck(ticket_id)` after the delay, replacing a re-check of the ticket that hasn't started yet."""
        previous = self._pending.pop(ticket_id, None)
        if previous is not None:
            previous.cancel()
        task = asyncio.create_task(self._run(ticket_id, delay_seconds, recheck), name=f"recheck-ticket-{ticket_id}")
        self._pending[ticket_id] = task
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    def pending(self, ticket_id: int) -> bool:
        """Return whether a re-check of the ticket is waiting to run."""
        return ticket_id in self._pending

    async def _run(self, ticket_id: int, delay_seconds: float, recheck: Recheck) -> None:
        await self._sleep(delay_seconds)
        # Started re-checks can no longer be replaced, so a new event never cancels an answer in progress.
        if self._pending.get(ticket_id) is asyncio.current_task():
            del self._pending[ticket_id]
        try:
            await recheck(ticket_id)
        except Exception:
            logger.error(f"Scheduled re-check of ticket {ticket_id} failed.", exc_info=True)

    @asynccontextmanager
    async def ticket_lock(self, ticket_id: int) -> AsyncIterator[None]:
        """Hold the ticket's lock, so an event and a re-check never answer the same turn at once."""
        lock = self._locks.setdefault(ticket_id, asyncio.Lock())
        self._lock_users[ticket_id] = self._lock_users.get(ticket_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._lock_users[ticket_id] -= 1
            if not self._lock_users[ticket_id]:
                del self._lock_users[ticket_id]
                del self._locks[ticket_id]
