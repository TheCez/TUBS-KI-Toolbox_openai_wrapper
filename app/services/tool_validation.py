from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ValidatedToolCall:
    name: str
    arguments: str
    repaired: bool = False


@dataclass
class ToolValidationIssue:
    name: str
    reason: str


@dataclass
class ToolValidationResult:
    valid_calls: list[ValidatedToolCall]
    issues: list[ToolValidationIssue]

    @property
    def fallback_text(self) -> str | None:
        if not self.issues:
            return None
        if len(self.issues) == 1:
            issue = self.issues[0]
            return f"I need more complete tool arguments before I can call `{issue.name}` safely: {issue.reason}."
        issue_text = "; ".join(f"`{issue.name}`: {issue.reason}" for issue in self.issues)
        return f"I couldn't safely call the requested tools because the generated arguments were invalid: {issue_text}."


def _tool_name(tool: Any) -> str | None:
    if hasattr(tool, "name"):
        return getattr(tool, "name")
    if isinstance(tool, dict):
        if isinstance(tool.get("function"), dict):
            return tool["function"].get("name")
        return tool.get("name")
    return None


def _tool_schema(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "input_schema"):
        return getattr(tool, "input_schema") or {}
    if hasattr(tool, "parameters"):
        return getattr(tool, "parameters") or {}
    if isinstance(tool, dict):
        if isinstance(tool.get("input_schema"), dict):
            return tool["input_schema"]
        if isinstance(tool.get("parameters"), dict):
            return tool["parameters"]
        if isinstance(tool.get("function"), dict):
            return tool["function"].get("parameters", {}) or {}
    return {}


def _available_tools_by_name(tools: Iterable[Any] | None) -> dict[str, dict[str, Any]]:
    if not tools:
        return {}
    available = {}
    for tool in tools:
        name = _tool_name(tool)
        if name:
            available[name] = _tool_schema(tool)
    return available


def _try_parse_json(arguments: str) -> tuple[Any | None, bool]:
    raw = (arguments or "").strip()
    if not raw:
        return {}, True

    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass

    object_start = raw.find("{")
    object_end = raw.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidate = raw[object_start : object_end + 1]
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidate = raw[array_start : array_end + 1]
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    return None, False


def validate_tool_calls(parsed_calls: list[dict[str, str]], tools: Iterable[Any] | None) -> ToolValidationResult:
    available = _available_tools_by_name(tools)
    valid_calls: list[ValidatedToolCall] = []
    issues: list[ToolValidationIssue] = []

    for parsed in parsed_calls:
        name = parsed.get("name", "").strip() or "unknown_tool"
        raw_arguments = parsed.get("arguments", "")
        schema = available.get(name)

        if available and schema is None:
            issues.append(ToolValidationIssue(name=name, reason="unknown tool name"))
            continue

        parsed_args, repaired = _try_parse_json(raw_arguments)
        if parsed_args is None:
            issues.append(ToolValidationIssue(name=name, reason="arguments are not valid JSON"))
            continue

        if isinstance(schema, dict) and schema.get("type") == "object":
            if not isinstance(parsed_args, dict):
                issues.append(ToolValidationIssue(name=name, reason="arguments must be a JSON object"))
                continue

            required = schema.get("required", [])
            missing = [field for field in required if field not in parsed_args]
            if missing:
                issues.append(
                    ToolValidationIssue(name=name, reason=f"missing required fields: {', '.join(missing)}")
                )
                continue

        valid_calls.append(
            ValidatedToolCall(
                name=name,
                arguments=json.dumps(parsed_args, ensure_ascii=False),
                repaired=repaired,
            )
        )

    return ToolValidationResult(valid_calls=valid_calls, issues=issues)
