from __future__ import annotations

from typing import Any, Optional


class AppError(Exception):
    """
    Base application error that carries a human-friendly message,
    optional details for observability, and an optional root cause.
    """
    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        bits = [self.message]
        if self.details:
            bits.append(f"details={self.details}")
        if self.cause is not None:
            bits.append(f"cause={self.cause!r}")
        return " | ".join(bits)


def ensure(condition: bool, msg: str, **details: Any) -> None:
    """Raise AppError if condition is False, with optional context details."""
    if not condition:
        raise AppError(msg, details=details or None)
