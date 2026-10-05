"""Errors that command-line entry points can present without a traceback."""
from typing import NoReturn


class HaloError(RuntimeError):
    """A user-facing, fail-closed error."""


def fail(message: str) -> NoReturn:
    raise HaloError(message)
