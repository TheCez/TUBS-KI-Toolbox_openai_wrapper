"""
Shared prompt engineering utilities used by both OpenAI and Anthropic routes.
Centralizes tool-calling instructions, stop-sequence truncation, and XML parsing.
"""

import re
from typing import Any, List, Optional, Tuple


# Compiled once, reused everywhere
TOOL_CALL_REGEX = re.compile(
    r'<tool_call>\s*<name>(.*?)<\\?/name>\s*<arguments>(.*?)<\\?/arguments>\s*<\\?/tool_call>',
    re.DOTALL
)

TOOL_CALLS_CLOSE_REGEX = re.compile(r'<\\?/tool_calls>')
TOOL_CALL_CLOSE_REGEX = re.compile(r'<\\?/tool_call>')

THOUGHT_REGEX = re.compile(r'<thought>(.*?)</thought>', re.DOTALL)


def _tool_schema_for_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("input_schema") and isinstance(tool["input_schema"], dict):
        return tool["input_schema"]
    if tool.get("function") and isinstance(tool["function"], dict):
        return tool["function"].get("parameters", {}) or {}
    return tool.get("parameters", {}) or {}


def _tool_name_for_prompt(tool: dict[str, Any]) -> str:
    if tool.get("name"):
        return str(tool["name"])
    if tool.get("function") and isinstance(tool["function"], dict):
        return str(tool["function"].get("name", "unknown_tool"))
    return "unknown_tool"


def _tool_description_for_prompt(tool: dict[str, Any]) -> str:
    if tool.get("description"):
        return str(tool["description"])
    if tool.get("function") and isinstance(tool["function"], dict):
        return str(tool["function"].get("description", "") or "")
    return ""


def _format_tool_requirements(tools_dicts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools_dicts:
        name = _tool_name_for_prompt(tool)
        description = _tool_description_for_prompt(tool)
        schema = _tool_schema_for_prompt(tool)
        required = schema.get("required", []) if isinstance(schema, dict) else []
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}

        line = f"- {name}"
        if description:
            line += f": {description}"
        requirement_text = ", ".join(str(item) for item in required) if required else "none"
        argument_names = list(properties.keys()) if isinstance(properties, dict) else []
        if argument_names:
            available_text = ", ".join(str(key) for key in argument_names[:8])
            if len(argument_names) > 8:
                available_text += ", ..."
        else:
            available_text = "none"
        lines.append(f"{line} | required: {requirement_text} | args: {available_text}")

    return "\n".join(lines)


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

    tools_summary = _format_tool_requirements(tools_dicts)
    return (
        "Tools are available in this request. Call them only when needed.\n"
        "If you call a tool, output only XML and stop immediately.\n"
        "Format:\n"
        '<tool_calls><tool_call><name>tool_name</name>'
        '<arguments>{"actual_key": "actual_value"}</arguments>'
        "</tool_call></tool_calls>\n"
        "Use the exact tool name and raw JSON arguments matching the schema.\n"
        "Do not add prose around the XML.\n"
        "Do not call a tool with missing required fields.\n"
        f"Tool summary:\n{tools_summary}\n"
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
