"""MCP (Model Context Protocol) server exposing the SDK's tools.

Run with ``bg-remove-mcp`` (stdio transport), e.g. in a Claude Code / Claude
Desktop MCP configuration::

    {
      "mcpServers": {
        "background-remove": {
          "command": "bg-remove-mcp"
        }
      }
    }

Requires the ``mcp`` extra: ``pip install "background-remove-sdk[mcp]"``.
"""

from __future__ import annotations

from typing import Optional


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "The MCP server requires the 'mcp' package. "
            'Install it with: pip install "background-remove-sdk[mcp]"'
        ) from exc

    from background_remove_sdk.agents import tools

    server = FastMCP(
        "background-remove",
        instructions=(
            "Tools for removing image backgrounds. All tools take a path to an "
            "image file and write a transparent PNG, returning its path."
        ),
    )

    @server.tool()
    def remove_background(input_path: str, output_path: Optional[str] = None) -> dict:
        """Remove the background from an image and save a transparent PNG."""
        return tools.execute_tool(
            "remove_background", {"input_path": input_path, "output_path": output_path}
        )

    @server.tool()
    def generate_mask(input_path: str, output_path: Optional[str] = None) -> dict:
        """Generate a grayscale foreground mask for an image and save it as PNG."""
        return tools.execute_tool(
            "generate_mask", {"input_path": input_path, "output_path": output_path}
        )

    @server.tool()
    def extract_object_at_point(
        input_path: str, x: int, y: int, output_path: Optional[str] = None
    ) -> dict:
        """Cut out the foreground object at pixel (x, y) and save it as a transparent PNG."""
        return tools.execute_tool(
            "extract_object_at_point",
            {"input_path": input_path, "x": x, "y": y, "output_path": output_path},
        )

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
