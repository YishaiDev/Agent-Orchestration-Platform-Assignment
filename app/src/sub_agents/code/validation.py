"""Deterministic, per-language syntax validation for generated code.

This is the Code Agent's Tier-1 (ground-truth) quality gate — used instead of an LLM judge wherever
a parser exists. It only ever *parses* code (never executes it), so it carries no security risk.
Languages without a registered validator are routed to the Tier-2 LLM critic by the agent (see
``has_validator``); a real parser can be slotted into ``_VALIDATORS`` later with no other changes.

Python uses the stdlib ``ast`` (exact). JavaScript uses ``tree-sitter``, which is error-recovering
and therefore a **coarser** gate than ``ast``: it reliably flags gross breakage (and inserted
``MISSING`` nodes) but may tolerate some subtle errors.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser

SyntaxValidator = Callable[[str], str | None]

_LANGUAGE_ALIASES: dict[str, str] = {
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "jsx": "javascript",
    "py": "python",
    "python3": "python",
}


def _normalize_lang(language: str) -> str:
    """Normalize a language name to its canonical registry key.

    The planner is an LLM and emits short or aliased names (``js``, ``py``); without this the
    registered validators would silently never fire.

    Args:
        language: The raw language name (any case).

    Returns:
        The canonical lower-case language key.
    """
    key = language.strip().lower()
    return _LANGUAGE_ALIASES.get(key, key)


def _py_syntax_error(code: str) -> str | None:
    """Return a Python ``SyntaxError`` message for ``code``, or None when it parses.

    Args:
        code: The Python source to check.

    Returns:
        The error message string when parsing fails, otherwise None.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return str(exc)
    return None


def _has_missing_node(node: Node) -> bool:
    """Return True when the subtree contains an inserted ``MISSING`` node.

    ``root_node.has_error`` alone misses recovered insertions, so the agent also walks for missing
    nodes to catch incomplete code.

    Args:
        node: The tree-sitter node to inspect recursively.

    Returns:
        True when any descendant (or the node itself) is missing.
    """
    if node.is_missing:
        return True
    return any(_has_missing_node(child) for child in node.children)


_JS_PARSER = Parser(Language(tsjs.language()))


def _js_syntax_error(code: str) -> str | None:
    """Return a coarse JavaScript syntax error for ``code``, or None when it parses cleanly.

    Args:
        code: The JavaScript source to check.

    Returns:
        A short error message when tree-sitter reports an error or a missing node, otherwise None.
    """
    root = _JS_PARSER.parse(code.encode("utf-8")).root_node
    if root.has_error or _has_missing_node(root):
        return "JavaScript syntax error (tree-sitter reported an error or missing node)."
    return None


_VALIDATORS: dict[str, SyntaxValidator] = {
    "python": _py_syntax_error,
    "javascript": _js_syntax_error,
}


def has_validator(language: str) -> bool:
    """Return True when a deterministic parser is registered for ``language``.

    The agent uses this to route: languages with a parser take the Tier-1 syntax gate; languages
    without one fall back to the Tier-2 LLM critic.

    Args:
        language: The language name (aliases accepted).

    Returns:
        True when a validator exists for the normalized language.
    """
    return _normalize_lang(language) in _VALIDATORS


def validate_syntax(code: str, language: str) -> str | None:
    """Validate ``code`` for the given language using the per-language registry.

    Args:
        code: The generated source code.
        language: The language name (aliases accepted).

    Returns:
        An error message when a registered validator rejects the code; None when the code is valid
        or the language has no validator (deferred to the critic).
    """
    validator = _VALIDATORS.get(_normalize_lang(language))
    return validator(code) if validator else None
