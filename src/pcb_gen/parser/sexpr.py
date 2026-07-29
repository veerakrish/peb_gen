"""Thin helpers over sexpdata for querying KiCad's s-expression trees.

KiCad nodes look like ``[Symbol('tag'), child, child, ...]`` after
``sexpdata.loads``. These helpers let the rest of the parser query by tag
name without repeating the same list-scanning logic everywhere.
"""

from __future__ import annotations

import sexpdata


def load(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return sexpdata.loads(f.read())


def tag(node) -> str | None:
    if isinstance(node, list) and node and isinstance(node[0], sexpdata.Symbol):
        return str(node[0])
    return None


def children(node: list, name: str) -> list[list]:
    return [n for n in node[1:] if tag(n) == name]


def child(node: list, name: str) -> list | None:
    for n in node[1:]:
        if tag(n) == name:
            return n
    return None


def value_str(node) -> str:
    """First value after the tag, coerced to a plain string (strips quoting)."""
    v = node[1]
    if isinstance(v, sexpdata.Symbol):
        return str(v)
    return str(v)


def values(node) -> list:
    """All values after the tag (as raw sexpdata atoms)."""
    return node[1:]
