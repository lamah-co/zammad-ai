"""Application lifecycle and activity status tracking helpers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from prometheus_client import Enum, Gauge

STATUS_ENUM = Enum(
    name="zammad_ai_status",
    documentation="Application status. Values: startup, active, ready (idle), shutdown.",
    states=["startup", "active", "ready", "shutdown"],
)

PROVIDER_READY = Gauge(
    name="zammad_ai_provider_ready",
    documentation="1 when the GenAI embedding provider answered the readiness check, 0 while it is unavailable.",
)

_lifecycle_status = "startup"
_requests_in_flight = 0
_provider_ready = True
_provider_reason: str | None = None
PROVIDER_READY.set(1)


def set_provider_ready(ready: bool, reason: str | None = None) -> None:
    """Record whether the GenAI provider is usable, with the error class when it is not."""
    global _provider_ready, _provider_reason
    _provider_ready = ready
    _provider_reason = None if ready else reason
    PROVIDER_READY.set(1 if ready else 0)


def provider_status() -> tuple[bool, str | None]:
    """Return whether the GenAI provider is usable and, if not, the error class that made it unusable."""
    return _provider_ready, _provider_reason


def set_status(state: str) -> None:
    """Set the lifecycle status and publish it to the status metric."""
    global _lifecycle_status
    _lifecycle_status = state
    STATUS_ENUM.state(state)


@asynccontextmanager
async def track_activity() -> AsyncIterator[None]:
    """Track in-flight requests and update active/idle status accordingly."""
    global _requests_in_flight

    _requests_in_flight += 1
    STATUS_ENUM.state("active")
    try:
        yield
    finally:
        _requests_in_flight -= 1
        if _requests_in_flight == 0:
            STATUS_ENUM.state(_lifecycle_status)
        else:
            STATUS_ENUM.state("active")
