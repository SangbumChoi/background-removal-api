"""Agent-framework integration for background-remove-sdk.

Exposes the SDK as tools consumable by any agentic framework:

- :func:`get_openai_tools` / :func:`get_anthropic_tools` return ready-made
  tool schemas for OpenAI and Anthropic tool use.
- :func:`execute_tool` dispatches a tool call by name with JSON arguments.
- Plain typed functions (``remove_background_tool`` etc.) plug directly into
  frameworks that build schemas from signatures (LangChain, smolagents, the
  OpenAI Agents SDK, ...).
- ``bg-remove-mcp`` (see :mod:`background_remove_sdk.agents.mcp_server`) runs
  the same tools as an MCP server.
"""

from background_remove_sdk.agents.tools import (
    TOOL_FUNCTIONS,
    execute_tool,
    extract_object_at_point_tool,
    generate_mask_tool,
    get_anthropic_tools,
    get_openai_tools,
    remove_background_tool,
)

__all__ = [
    "TOOL_FUNCTIONS",
    "execute_tool",
    "get_openai_tools",
    "get_anthropic_tools",
    "remove_background_tool",
    "generate_mask_tool",
    "extract_object_at_point_tool",
]
