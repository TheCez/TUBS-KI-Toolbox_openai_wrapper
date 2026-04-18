from __future__ import annotations

from typing import Iterable


def guidance_for_tool_errors(tool_results: Iterable[dict]) -> list[str]:
    hints: list[str] = []
    saw_string_replace_error = False
    saw_write_error = False

    for result in tool_results:
        if not result.get("is_error"):
            continue
        text = (result.get("text") or "").lower()
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

    return hints
