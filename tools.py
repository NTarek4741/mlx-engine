"""
Tool formatting module for MLX Engine.

Provides model-specific tool calling templates for:
- Qwen 2.5+
- Mistral

Handles:
1. Converting Tool definitions to model prompt format
2. Detecting tool calls in model output
3. Formatting tool results back to the model
"""

from typing import Any
import json
import re
from feilds import Tool, ToolUseBlockParam, ToolResultBlockParam


# =============================================================================
# Qwen 2.5 Tool Format
# =============================================================================


def format_tools_qwen(tools: list[Tool]) -> str:
    """
    Format tools for Qwen 2.5+ models.

    Qwen uses a specific JSON format with function definitions.

    Example output:
    ```
    <|im_start|>system
    You have access to the following functions:

    {"name": "web_search", "description": "Search the web", "parameters": {...}}
    <|im_end|>
    ```
    """
    if not tools:
        return ""

    tool_defs = []
    for tool in tools:
        tool_def = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema.model_dump(exclude_none=True),
        }
        tool_defs.append(json.dumps(tool_def, ensure_ascii=False))

    tools_prompt = "You have access to the following functions:\n\n"
    tools_prompt += "\n\n".join(tool_defs)

    return tools_prompt


def detect_tool_call_qwen(text: str) -> list[dict[str, Any]] | None:
    """
    Detect tool calls in Qwen 2.5 output.

    Qwen outputs tool calls in JSON format:
    ```
    <tool_call>
    {"name": "web_search", "arguments": {"query": "MLX"}}
    </tool_call>
    ```
    """
    # Look for <tool_call> tags
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            call_data = json.loads(match.strip())
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "name": call_data["name"],
                    "input": call_data.get("arguments", {}),
                }
            )
        except json.JSONDecodeError:
            continue

    return tool_calls if tool_calls else None


# =============================================================================
# Mistral Tool Format
# =============================================================================


def format_tools_mistral(tools: list[Tool]) -> str:
    """
    Format tools for Mistral models.
    """
    if not tools:
        return ""

    tool_defs = []
    for tool in tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema.model_dump(exclude_none=True),
            },
        }
        tool_defs.append(tool_def)

    return f"[AVAILABLE_TOOLS] {json.dumps(tool_defs)} [/AVAILABLE_TOOLS]"


def detect_tool_call_mistral(text: str) -> list[dict[str, Any]] | None:
    """
    Detect tool calls in Mistral output.

    Mistral uses [TOOL_CALLS] tags.
    """
    pattern = r"\[TOOL_CALLS\](.*?)\[/TOOL_CALLS\]"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    try:
        calls_data = json.loads(match.group(1).strip())
        tool_calls = []
        for i, call in enumerate(calls_data):
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "name": call.get("name"),
                    "input": call.get("arguments", {}),
                }
            )
        return tool_calls
    except json.JSONDecodeError:
        return None


# =============================================================================
# Model Detection & Routing
# =============================================================================


def detect_model_type(model_path: str) -> str:
    """
    Detect model type from path.

    Returns: "qwen", "mistral", or "unknown"
    """
    model_lower = model_path.lower()

    if "qwen" in model_lower:
        return "qwen"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        return "unknown"


def format_tools_for_model(tools: list[Tool], model_path: str) -> str:
    """
    Format tools for the specific model.
    """
    model_type = detect_model_type(model_path)

    formatters = {
        "qwen": format_tools_qwen,
        "mistral": format_tools_mistral,
    }

    formatter = formatters.get(model_type, format_tools_qwen)  # Default to Qwen
    return formatter(tools)


def detect_tool_calls(text: str, model_path: str) -> list[ToolUseBlockParam] | None:
    """
    Detect tool calls in model output and return as ToolUseBlockParam list.
    """
    model_type = detect_model_type(model_path)

    detectors = {
        "qwen": detect_tool_call_qwen,
        "mistral": detect_tool_call_mistral,
    }

    detector = detectors.get(model_type, detect_tool_call_qwen)
    tool_calls_data = detector(text)

    if not tool_calls_data:
        return None

    # Convert to ToolUseBlockParam
    tool_blocks = []
    for call in tool_calls_data:
        tool_blocks.append(
            ToolUseBlockParam(
                id=call["id"], name=call["name"], input=call["input"], type="tool_use"
            )
        )

    return tool_blocks
