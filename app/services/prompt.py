"""
Shared prompt engineering utilities used by both OpenAI and Anthropic routes.
Centralizes tool-calling instructions, stop-sequence truncation, and XML parsing.
"""

import re
import json
import uuid
from typing import List, Optional, Tuple


# Compiled once, reused everywhere
TOOL_CALL_REGEX = re.compile(
    r'<tool_call>\s*<name>(.*?)<\\?/name>\s*<arguments>(.*?)<\\?/arguments>\s*<\\?/tool_call>',
    re.DOTALL
)

TOOL_CALLS_CLOSE_REGEX = re.compile(r'<\\?/tool_calls>')
TOOL_CALL_CLOSE_REGEX = re.compile(r'<\\?/tool_call>')

THOUGHT_REGEX = re.compile(r'<thought>(.*?)</thought>', re.DOTALL)


def build_tool_instructions(tools: list) -> str:
    """
    Generates the XML-based tool-calling prompt injection string.
    Accepts either raw dicts (OpenAI format) or Pydantic models (Anthropic format).
    """
    # Normalize to dicts
    tools_dicts = []
    for t in tools:
        if hasattr(t, "model_dump"):
            tools_dicts.append(t.model_dump())
        else:
            tools_dicts.append(t)

    tools_str = json.dumps(tools_dicts)
    return (
        "You have access to the tools provided in the current request. "
        "To use any tool, you MUST follow this XML format. If you need to trigger "
        "multiple tools simultaneously, you MUST wrap them in a <tool_calls> root "
        "element and output all of them back-to-back before stopping. "
        "Use this EXACT format:\n"
        '<tool_calls><tool_call><name>tool_name</name>'
        '<arguments>{"actual_key": "actual_value"}</arguments>'
        "</tool_call></tool_calls>\n"
        "The <arguments> MUST contain raw, valid JSON that perfectly matches the "
        'provided tool schema. DO NOT wrap the arguments inside a {"json": ...} '
        "parent object.\n"
        "CRITICAL RULES:\n"
        "1. CRITICAL: When outputting XML tags like <tool_call>, <name>, or "
        "</arguments>, you MUST NOT escape the forward slash. Output raw, plain "
        "XML tags only.\n"
        "2. When creating a new file, DO NOT use apply_patch if possible. Use bash "
        'with a heredoc (e.g., cat << "EOF" > file.ext) or write_file if available. '
        "Only use apply_patch for modifying existing code blocks or no other valid "
        "tool is available.\n"
        "3. Ensure the JSON inside <arguments> is valid, single-line or properly "
        "escaped, and contains no trailing commas or unescaped characters that "
        "could break a json.loads() call. Be extremely precise with newlines and "
        "special characters.\n"
        "4. Avoid generating massive files in a single tool call if they contain "
        "complex nested structures. If a file is over 100 lines, consider writing "
        "it in logical stages if the client tools support it.\n"
        '5. Your output must contain ONLY the XML blocks. No conversational filler '
        'like "Here is your code" before or after the <tool_calls> tag, as this '
        "can confuse the client parser.\n"
        f"Available tools: {tools_str}\n"
    )


def truncate_at_stop(text: str, stop_sequences: Optional[List[str]]) -> Tuple[str, bool]:
    """
    Finds the earliest occurrence of any stop sequence in text.
    Returns (truncated_text, was_truncated).
    """
    if not stop_sequences:
        return text, False

    earliest_idx = -1
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx != -1:
        return text[:earliest_idx], True
    return text, False


def parse_tool_calls_xml(text: str) -> List[dict]:
    """
    Extracts tool calls from XML in the response text.
    Returns list of dicts with 'name' and 'arguments' keys.
    Handles both escaped (\\/) and unescaped (/) forward slashes in closing tags.
    """
    matches = list(TOOL_CALL_REGEX.finditer(text))
    results = []
    for match in matches:
        name = match.group(1).strip() if match.group(1) else ""
        arguments = match.group(2).strip() if match.group(2) else ""
        # Sanitize double-escaped forward slashes
        arguments = arguments.replace('\\/', '/')
        results.append({"name": name, "arguments": arguments})
    return results


def is_tool_xml_complete(text: str) -> bool:
    """Checks if the buffered text contains a fully closed tool-call XML block."""
    if "<tool_calls>" in text:
        return bool(TOOL_CALLS_CLOSE_REGEX.search(text))
    return bool(TOOL_CALL_CLOSE_REGEX.search(text))


def has_tool_xml_start(text: str) -> bool:
    """Checks if the text contains the beginning of a tool-call XML block."""
    return "<tool_call>" in text or "<tool_calls>" in text


def strip_tool_xml(text: str) -> str:
    """Removes all tool-call XML from text, returning only the plain text content."""
    cleaned = re.sub(r'<tool_call>.*?<\\?/tool_call>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<\\?/tool_calls>', '', cleaned)
    cleaned = cleaned.replace('<tool_calls>', '')
    return cleaned.strip()


def extract_reasoning(text: str) -> Tuple[str, Optional[str]]:
    """
    Extracts <thought>...</thought> blocks from the response.
    Returns (cleaned_text, reasoning_content).
    """
    match = THOUGHT_REGEX.search(text)
    if match:
        reasoning = match.group(1).strip()
        cleaned = THOUGHT_REGEX.sub('', text).strip()
        return cleaned, reasoning
    return text, None
