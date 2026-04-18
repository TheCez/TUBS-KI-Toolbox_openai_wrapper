from __future__ import annotations

import re
from typing import Iterable


_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\r\n]+")
_UNIX_PATH_RE = re.compile(r"/[^\r\n]+")
_QUOTED_SYMBOL_RE = re.compile(r"String:\s*(.+)", re.IGNORECASE | re.DOTALL)
_DECLARATION_RE = re.compile(
    r"\b(const|function|class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)|<([A-Z][A-Za-z0-9_]*)\b"
)


def _extract_file_path(text: str) -> str | None:
    for pattern in (_WINDOWS_PATH_RE, _UNIX_PATH_RE):
        match = pattern.search(text)
        if match:
            return match.group(0).rstrip(") ")
    return None


def _extract_symbol_anchor(text: str) -> str | None:
    source = text
    string_match = _QUOTED_SYMBOL_RE.search(text)
    if string_match:
        source = string_match.group(1)

    declaration = _DECLARATION_RE.search(source)
    if not declaration:
        return None

    if declaration.group(1) and declaration.group(2):
        return f"{declaration.group(1)} {declaration.group(2)}"
    if declaration.group(3):
        return declaration.group(3)
    return None


def _file_metadata_hint(text: str) -> str | None:
    file_path = _extract_file_path(text)
    symbol_anchor = _extract_symbol_anchor(text)
    parts: list[str] = []

    if file_path:
        parts.append(f"target file: `{file_path}`")
    if symbol_anchor:
        parts.append(f"likely stable anchor: `{symbol_anchor}`")

    if not parts:
        return None

    return "Wrapper repair metadata: " + "; ".join(parts) + "."


def guidance_for_tool_errors(tool_results: Iterable[dict]) -> list[str]:
    hints: list[str] = []
    saw_string_replace_error = False
    saw_write_error = False
    metadata_hints: list[str] = []

    for result in tool_results:
        if not result.get("is_error"):
            continue
        raw_text = result.get("text") or ""
        metadata_hint = _file_metadata_hint(raw_text)
        if metadata_hint:
            metadata_hints.append(metadata_hint)

        text = raw_text.lower()
        if "string to replace not found in file" in text:
            saw_string_replace_error = True
        if "error writing file" in text:
            saw_write_error = True

    if saw_string_replace_error:
        hints.append(
            "Wrapper repair hint: the last edit used a stale exact text match. "
            "Read the file again before editing, then use smaller anchored replacements around unique symbols "
            "such as a component name, function name, or constant declaration instead of retrying the same large exact string."
        )

    if saw_write_error:
        hints.append(
            "Wrapper repair hint: a file write failed. Re-read the current file contents, verify the target path, "
            "and prefer a minimal edit over rewriting a large block."
        )

    return hints + list(dict.fromkeys(metadata_hints))
