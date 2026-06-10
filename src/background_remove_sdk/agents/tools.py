"""Framework-agnostic tool definitions for agentic use.

Each tool takes file paths (the lingua franca of agent tool calls) and
returns a JSON-serializable dict describing the result, so the output can be
fed straight back to a model.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from background_remove_sdk import core


def remove_background_tool(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Remove the background from the image at ``input_path``.

    Saves a transparent PNG and returns its path. When ``output_path`` is
    omitted, the result is written next to the input as ``<name>_no_bg.png``.
    """
    out = Path(output_path) if output_path else core.default_output_path(input_path, "rgba")
    image = core.remove_background(input_path, output_path=out)
    return {
        "output_path": str(out),
        "width": image.width,
        "height": image.height,
        "format": "PNG (RGBA)",
    }


def generate_mask_tool(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate a grayscale foreground mask for the image at ``input_path``.

    Saves the mask as PNG and returns its path. When ``output_path`` is
    omitted, the result is written next to the input as ``<name>_mask.png``.
    """
    out = Path(output_path) if output_path else core.default_output_path(input_path, "mask")
    image = core.generate_mask(input_path, output_path=out)
    return {
        "output_path": str(out),
        "width": image.width,
        "height": image.height,
        "format": "PNG (grayscale mask)",
    }


def extract_object_at_point_tool(
    input_path: str, x: int, y: int, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Extract the foreground object at pixel ``(x, y)`` from the image.

    Removes the background and crops to the object containing the given
    point. Saves a transparent PNG and returns its path. When ``output_path``
    is omitted, the result is written as ``<name>_object.png``.
    """
    out = Path(output_path) if output_path else core.default_output_path(input_path, "object")
    image = core.extract_object_at_point(input_path, x, y, output_path=out)
    return {
        "output_path": str(out),
        "width": image.width,
        "height": image.height,
        "format": "PNG (RGBA, cropped to object)",
    }


TOOL_FUNCTIONS = {
    "remove_background": remove_background_tool,
    "generate_mask": generate_mask_tool,
    "extract_object_at_point": extract_object_at_point_tool,
}

_INPUT_PATH_PARAM = {
    "type": "string",
    "description": "Path to the input image file (jpg, png, webp, ...).",
}
_OUTPUT_PATH_PARAM = {
    "type": "string",
    "description": "Optional path for the output PNG. Defaults to a file next to the input.",
}

_TOOL_SCHEMAS = [
    {
        "name": "remove_background",
        "description": (
            "Remove the background from an image. Saves a transparent PNG and "
            "returns its path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": _INPUT_PATH_PARAM,
                "output_path": _OUTPUT_PATH_PARAM,
            },
            "required": ["input_path"],
        },
    },
    {
        "name": "generate_mask",
        "description": (
            "Generate a grayscale foreground/background mask for an image. "
            "Saves the mask as PNG and returns its path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": _INPUT_PATH_PARAM,
                "output_path": _OUTPUT_PATH_PARAM,
            },
            "required": ["input_path"],
        },
    },
    {
        "name": "extract_object_at_point",
        "description": (
            "Cut out the foreground object located at pixel (x, y) in an image. "
            "Removes the background and crops to that object's bounding box. "
            "Saves a transparent PNG and returns its path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": _INPUT_PATH_PARAM,
                "x": {"type": "integer", "description": "X pixel coordinate inside the object."},
                "y": {"type": "integer", "description": "Y pixel coordinate inside the object."},
                "output_path": _OUTPUT_PATH_PARAM,
            },
            "required": ["input_path", "x", "y"],
        },
    },
]


def get_openai_tools() -> list:
    """Tool definitions in OpenAI ``tools=[...]`` (function calling) format."""
    return [
        {"type": "function", "function": copy.deepcopy(schema)}
        for schema in _TOOL_SCHEMAS
    ]


def get_anthropic_tools() -> list:
    """Tool definitions in Anthropic ``tools=[...]`` (tool use) format."""
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": copy.deepcopy(schema["parameters"]),
        }
        for schema in _TOOL_SCHEMAS
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call by name, as produced by an LLM.

    Returns the tool result dict on success, or ``{"error": ...}`` on
    failure so agent loops can relay the message back to the model.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name!r}. Available: {sorted(TOOL_FUNCTIONS)}"}
    try:
        return func(**arguments)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
