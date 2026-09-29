"""Fixture-first visual text-region research prototype package.

The current runtime exposes stage modules explicitly.  The earlier pilot
``src.pipeline`` API remains importable for historical tests, but is not loaded
as a package side effect and is not used by the new CLI or UI.
"""

__all__: list[str] = []
