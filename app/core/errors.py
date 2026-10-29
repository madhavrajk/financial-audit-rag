"""Typed errors. Each one maps to exactly one behaviour, so callers never parse error strings."""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""


class UserInputError(AppError):
    """Bad request from the caller (HTTP 4xx)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class UpstreamError(AppError):
    """The model provider (or another dependency) failed after retries (HTTP 503)."""

    def __init__(self, message: str, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class UpstreamAuthError(UpstreamError):
    """Invalid or missing API key. Never retried."""


class ToolError(AppError):
    """A tool call failed in a way the model can recover from.

    The message is sent back to the model as the tool result; it is never shown to the user.
    """


class NotReadyError(AppError):
    """A required resource (index, DB, API key) is missing at startup."""
