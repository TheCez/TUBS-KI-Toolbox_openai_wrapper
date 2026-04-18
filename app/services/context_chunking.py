from __future__ import annotations

import re
from typing import Iterable


_CODE_HINT_RE = re.compile(
    r"```|(^|\n)\s*(def |class |async def |function |export function |const |let |var |public |private |protected )"
)
_STACK_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\):|(^|\n)\s*File \".+\", line \d+|(^|\n)\s*at .+\(.+\)|Exception:|Error:"
)
_DIFF_LINE_RE = re.compile(r"^(diff --git|index |@@ |\+\+\+ |--- |\+[^+]|-[^-])", re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_code_like_text(text: str) -> bool:
    if not text:
        return False
    if _CODE_HINT_RE.search(text):
        return True
    if "{" in text and "}" in text and "\n" in text:
        return True
    return False


def is_fenced_code_block(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("```") and stripped.count("```") >= 2


def is_stack_trace_text(text: str) -> bool:
    return bool(text and _STACK_TRACE_RE.search(text))


def is_diff_text(text: str) -> bool:
    return bool(text and _DIFF_LINE_RE.search(text))


def _non_empty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in (text or "").splitlines() if line.strip()]


def _summarize_fenced_block(text: str) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    fence_header = lines[0].strip("`").strip() or "code"
    body = [line for line in lines[1:] if line.strip("`").strip()]
    if not body:
        return f"[fenced {fence_header} block]"
    first_line = body[0].strip()
    last_line = body[-1].strip()
    if len(body) == 1:
        return f"[fenced {fence_header} block] {first_line}"
    if first_line == last_line:
        return f"[fenced {fence_header} block] {first_line}"
    return f"[fenced {fence_header} block] {first_line} ... {last_line}"


def _summarize_fenced_fragment(text: str) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    fence_header = lines[0].strip("`").strip() or "code"
    body = [line.strip() for line in lines[1:] if line.strip()]
    if not body:
        return f"[fenced {fence_header} fragment]"
    return f"[fenced {fence_header} fragment] {body[0]}"


def _summarize_stack_trace(text: str) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    picked: list[str] = []
    if lines[0].startswith("Traceback"):
        picked.append(lines[0])
    file_lines = [line.strip() for line in lines if 'File "' in line or line.strip().startswith("at ")]
    picked.extend(file_lines[:2])
    terminal = lines[-1].strip()
    if terminal not in picked:
        picked.append(terminal)
    return " | ".join(dict.fromkeys(picked))


def _summarize_diff(text: str) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""
    picked: list[str] = []
    headers = [
        line.strip()
        for line in lines
        if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@ ")
    ]
    picked.extend(headers[:4])
    changes = [line.strip() for line in lines if (line.startswith("+") or line.startswith("-")) and not line.startswith("+++") and not line.startswith("---")]
    picked.extend(changes[:2])
    return " | ".join(dict.fromkeys(picked))


def _condense_chunk(chunk: str) -> str:
    stripped = (chunk or "").strip()
    if not stripped:
        return ""
    if is_fenced_code_block(stripped):
        return _summarize_fenced_block(stripped)
    if stripped.startswith("```"):
        return _summarize_fenced_fragment(stripped)
    if is_diff_text(stripped):
        return _summarize_diff(stripped)
    if is_stack_trace_text(stripped):
        return _summarize_stack_trace(stripped)
    if is_code_like_text(stripped):
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return " | ".join(lines)
    return normalize_whitespace(stripped)


def _candidate_delimiters(*, code_like: bool, diff_like: bool, stack_like: bool) -> list[str]:
    if diff_like:
        return [
            "\ndiff --git ",
            "\n--- ",
            "\n+++ ",
            "\n@@ ",
            "\n\n",
            "\n",
            " ",
        ]
    if stack_like:
        return [
            "\nTraceback (most recent call last):",
            "\n  File ",
            "\nFile ",
            "\n    at ",
            "\nat ",
            "\n",
            " ",
        ]
    if code_like:
        return [
            "\n```\n",
            "\n\n",
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\nfunction ",
            "\nexport function ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\npublic ",
            "\nprivate ",
            "\nprotected ",
            "\n}\n",
            "\n",
            " ",
        ]
    return [
        "\n\n",
        ". ",
        "? ",
        "! ",
        ".\n",
        "?\n",
        "!\n",
        ";\n",
        "\n",
        "; ",
        ", ",
        " ",
    ]


def _find_delimiter_cutoff(text: str, max_chars: int, delimiters: Iterable[str]) -> int | None:
    preferred_start = max(max_chars // 2, 1)
    for lower_bound in (preferred_start, 0):
        for delimiter in delimiters:
            idx = text.rfind(delimiter, lower_bound, max_chars)
            if idx != -1:
                return idx + len(delimiter)
    return None


def _split_index(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)

    code_like = is_code_like_text(text)
    diff_like = is_diff_text(text)
    stack_like = is_stack_trace_text(text)
    if code_like and text.lstrip().startswith("```"):
        fence_end = text.find("\n```", 1, max_chars)
        if fence_end != -1:
            return fence_end + len("\n```")

    cutoff = _find_delimiter_cutoff(
        text,
        max_chars,
        _candidate_delimiters(code_like=code_like, diff_like=diff_like, stack_like=stack_like),
    )
    if cutoff is not None and cutoff > 0:
        return cutoff

    whitespace_idx = text.rfind(" ", 0, max_chars)
    if whitespace_idx > 0:
        return whitespace_idx + 1

    return max_chars


def split_text_semantically(text: str, max_chars: int) -> list[str]:
    remaining = (text or "").strip()
    if not remaining:
        return []

    chunks: list[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        cutoff = _split_index(remaining, max_chars)
        chunk = remaining[:cutoff].strip()
        if not chunk:
            chunk = remaining[:max_chars].strip()
            cutoff = max_chars

        chunks.append(chunk)
        remaining = remaining[cutoff:].strip()

    return chunks


def summarize_text_chunks(text: str, *, per_chunk_chars: int, max_chunks: int) -> list[str]:
    raw_chunks = split_text_semantically(text, per_chunk_chars)
    chunks = [_condense_chunk(chunk) for chunk in raw_chunks if _condense_chunk(chunk)]
    if len(chunks) <= max_chunks:
        return chunks

    if max_chunks <= 1:
        return [chunks[-1]]

    visible_head_count = max_chunks - 2
    if visible_head_count <= 0:
        return [f"[{len(chunks) - 1} earlier chunks compacted]", chunks[-1]]

    head = chunks[:visible_head_count]
    omitted = len(chunks) - visible_head_count - 1
    return head + [f"[{omitted} earlier chunks compacted]", chunks[-1]]
