# background-remove-sdk

A **universal background-removal SDK**: one interface over many models. Every
backend — whatever its native input/output types (PIL, numpy, bytes, torch
tensors) — is normalized to the same call: image in, transparent PNG out.
Packaged with a CLI, ready-made tools for agentic frameworks, and an MCP server.

## Installation

```bash
pip install background-remove-sdk               # InSpyReNet backend included

# optional model backends
pip install "background-remove-sdk[rembg]"      # rembg zoo: U2-Net, IS-Net, BiRefNet, BRIA RMBG (CPU)
pip install "background-remove-sdk[rembg-gpu]"  # same, CUDA onnxruntime
pip install "background-remove-sdk[hf]"         # BiRefNet / RMBG-2.0 via transformers
pip install git+https://github.com/PramaLLC/BEN2.git   # BEN2

# optional integrations
pip install "background-remove-sdk[mcp]"        # MCP server for agent frameworks
pip install "background-remove-sdk[server]"     # HTTP API
pip install "background-remove-sdk[all]"        # everything on PyPI
```

## Supported models

Models are selected with a `"backend[:variant]"` spec string. Run
`bg-remove --list-models` (or call `background_remove_sdk.list_models()`) to
see everything available.

| Spec | Model | Notes |
|---|---|---|
| `inspyrenet` (default) | [InSpyReNet](https://github.com/plemeri/InSpyReNet) | Strong general-purpose default; `inspyrenet:fast` for speed |
| `rembg:u2net` | [U2-Net](https://github.com/danielgatis/rembg) | Lightweight classic; `u2netp` is only ~4.5 MB |
| `rembg:isnet-general-use` | IS-Net (DIS) | Good general use; `isnet-anime` for anime |
| `rembg:birefnet-general` | BiRefNet on ONNX | No torch required; `-lite`, `-portrait`, ... variants |
| `rembg:bria-rmbg` | BRIA RMBG-1.4 | Non-commercial license |
| `birefnet` | [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) via transformers | SOTA edges; any BiRefNet HF repo id as variant |
| `rmbg` | [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) via transformers | Excels on complex backgrounds; non-commercial license |
| `ben2` | [BEN2](https://huggingface.co/PramaLLC/BEN2) | Confidence-guided matting, excellent hair/edges |

```python
from background_remove_sdk import remove_background

remove_background("photo.jpg", model="rmbg")                 # briaai/RMBG-2.0
remove_background("photo.jpg", model="rembg:isnet-anime")    # anime-tuned IS-Net
remove_background("photo.jpg", model="birefnet:ZhengPeng7/BiRefNet_lite")
```

### Plugging in your own model

Implement one method and register it — the SDK normalizes everything else:

```python
from PIL import Image
from background_remove_sdk import register_backend
from background_remove_sdk.models import BaseBackend

class MyBackend(BaseBackend):
    name = "mymodel"

    def predict_mask(self, image: Image.Image) -> Image.Image:
        ...  # return a grayscale (L) mask, 255 = foreground

register_backend("mymodel", MyBackend)
remove_background("photo.jpg", model="mymodel")
```

Backends can also be shipped as separate pip packages via the
`background_remove_sdk.backends` entry-point group, and they become available
automatically — in the Python API, the CLI, the agent tools, and the MCP
server alike.

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

remover = BackgroundRemover(model="birefnet", device="cuda:0")  # loaded lazily, once
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
bg-remove photo.jpg --model rembg:isnet-general-use --device cpu
bg-remove --list-models                 # show all backends and variants
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
| `list_models` | Discover available model backends and variants |

All image tools accept an optional `model` argument (the same
`"backend[:variant]"` spec), so an agent can pick the right model per task —
e.g. `rembg:isnet-anime` for a cartoon, `rmbg` for a cluttered photo.

## HTTP API (optional)

```bash
pip install "background-remove-sdk[server]"
bg-remove-server --host 0.0.0.0 --port 5000 --model inspyrenet
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
