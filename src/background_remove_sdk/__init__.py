"""background-remove-sdk: remove image backgrounds with one call.

Quickstart::

    from background_remove_sdk import remove_background

    output = remove_background("photo.jpg")   # PIL RGBA image
    output.save("photo_no_bg.png")

For agent frameworks (OpenAI / Anthropic tool use, LangChain, MCP) see
:mod:`background_remove_sdk.agents`.
"""

from background_remove_sdk.core import (
    BackgroundRemover,
    extract_object_at_point,
    generate_mask,
    load_image,
    remove_background,
)

__version__ = "0.1.0"

__all__ = [
    "BackgroundRemover",
    "remove_background",
    "generate_mask",
    "extract_object_at_point",
    "load_image",
    "__version__",
]
