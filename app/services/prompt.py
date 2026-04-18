"""
Shared prompt engineering utilities used by both OpenAI and Anthropic routes.
Centralizes tool-calling instructions, stop-sequence truncation, and XML parsing.
"""

import re
import json
import uuid
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
        lines.append(line)

        if required:
            lines.append(f"  Required arguments: {', '.join(str(item) for item in required)}")
        else:
            lines.append("  Required arguments: none")

        if isinstance(properties, dict) and properties:
            property_names = ", ".join(str(key) for key in properties.keys())
            lines.append(f"  Available arguments: {property_names}")

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

    tools_str = json.dumps(tools_dicts)
    tools_summary = _format_tool_requirements(tools_dicts)
    return (
        "You have access to the tools provided in the current request. "
        "To use any tool, you MUST follow this XML format. If you need to trigger "
        "multiple tools simultaneously, you MUST wrap them in a <tool_calls> root "
        "element and output all of them back-to-back before stopping. "
        "If you decide to use a tool, emit the XML tool call and stop immediately. "
        "Do not add any assistant prose before or after the tool-call XML. "
        "Use this EXACT format:\n"
        '<tool_calls><tool_call><name>tool_name</name>'
        '<arguments>{"actual_key": "actual_value"}</arguments>'
        "</tool_call></tool_calls>\n"
        "The <arguments> MUST contain raw, valid JSON that perfectly matches the "
        'provided tool schema. DO NOT wrap the arguments inside a {"json": ...} '
        "parent object.\n"
        "CRITICAL RULES:\n"
        "1. When outputting XML tags like <tool_call>, <name>, or </arguments>, "
        "do not escape the forward slash. Output raw XML tags only.\n"
        "2. Ensure the JSON inside <arguments> is valid, single-line or properly "
        "escaped, and contains no trailing commas or unescaped characters that "
        "could break a json.loads() call. Be extremely precise with newlines and "
        "special characters.\n"
        '3. Your output must contain ONLY the XML blocks. No conversational filler '
        'like "Here is your code" before or after the <tool_calls> tag, as this '
        "can confuse the client parser.\n"
        "4. Never call a tool with an empty or incomplete arguments object when the "
        "schema requires fields. If you do not have the required arguments, do not "
        "call the tool yet.\n"
        "5. Match the argument keys exactly to the schema. Do not rename keys, wrap "
        "them in extra objects, or omit required arrays/objects.\n"
        f"Tool requirements summary:\n{tools_summary}\n"
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
