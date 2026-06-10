# background-remove-sdk

Remove image backgrounds with one call. Built on [InSpyReNet](https://github.com/plemeri/InSpyReNet)
(via [`transparent-background`](https://pypi.org/project/transparent-background/)), packaged as a
Python SDK with a CLI, ready-made tools for agentic frameworks, and an MCP server.

## Installation

```bash
pip install background-remove-sdk

# optional extras
pip install "background-remove-sdk[mcp]"     # MCP server for agent frameworks
pip install "background-remove-sdk[server]"  # HTTP API
```

## Quickstart

Single input in, background-removed output out:

```python
from background_remove_sdk import remove_background

output = remove_background("photo.jpg")          # PIL RGBA image
output.save("photo_no_bg.png")

# or save in one call
remove_background("photo.jpg", output_path="photo_no_bg.png")
```

Inputs can be a file path, raw `bytes`, a numpy array, or a `PIL.Image`.

For repeated calls, reuse one model instance:

```python
from background_remove_sdk import BackgroundRemover

remover = BackgroundRemover(mode="base", device="cuda:0")  # loaded lazily, once
for path in paths:
    remover.remove(path, output_path=path.replace(".jpg", "_no_bg.png"))
```

Also available:

```python
from background_remove_sdk import generate_mask, extract_object_at_point

mask = generate_mask("photo.jpg")                       # grayscale foreground mask
obj = extract_object_at_point("photo.jpg", x=120, y=80)  # cutout cropped to the object at (x, y)
```

## CLI

```bash
bg-remove photo.jpg                     # writes photo_no_bg.png
bg-remove photo.jpg -o cutout.png
bg-remove photo.jpg --mask              # writes photo_mask.png
bg-remove photo.jpg --point 120 80      # extract the object at pixel (120, 80)
bg-remove photo.jpg --mode fast --device cpu
```

## Agentic framework integration

The `background_remove_sdk.agents` module exposes the SDK as LLM tools. Tools
take file paths and return JSON-serializable dicts, so results can be fed
straight back to the model.

### MCP server (Claude Code, Claude Desktop, any MCP client)

```bash
pip install "background-remove-sdk[mcp]"
```

```json
{
  "mcpServers": {
    "background-remove": { "command": "bg-remove-mcp" }
  }
}
```

### Anthropic / OpenAI tool use

```python
from background_remove_sdk.agents import get_anthropic_tools, get_openai_tools, execute_tool

# Anthropic
response = client.messages.create(model=..., tools=get_anthropic_tools(), messages=messages)
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)

# OpenAI
response = client.chat.completions.create(model=..., tools=get_openai_tools(), messages=messages)
```

`execute_tool` never raises on bad model input — it returns `{"error": ...}`
so the message can be relayed back to the model.

### LangChain / smolagents / other frameworks

The plain tool functions have full type hints and docstrings, so frameworks
that build schemas from signatures can wrap them directly:

```python
from langchain_core.tools import StructuredTool
from background_remove_sdk.agents import remove_background_tool

tool = StructuredTool.from_function(remove_background_tool)
```

Available tools:

| Tool | Description |
|---|---|
| `remove_background` | Remove the background, save a transparent PNG, return its path |
| `generate_mask` | Save the grayscale foreground mask as PNG |
| `extract_object_at_point` | Cut out the object at pixel (x, y), cropped to its bounding box |

## HTTP API (optional)

```bash
pip install "background-remove-sdk[server]"
bg-remove-server --host 0.0.0.0 --port 5000
```

```bash
curl -X POST http://127.0.0.1:5000/api/remove -F "image=@photo.jpg" -o photo_no_bg.png
curl -X POST http://127.0.0.1:5000/api/mask -F "image=@photo.jpg" -o photo_mask.png
curl -X POST http://127.0.0.1:5000/api/extract -F "image=@photo.jpg" -F "x=120" -F "y=80" -o object.png
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Sample images live in `examples/`.
