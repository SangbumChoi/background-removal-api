"""Model backend registry.

Models are addressed by a spec string ``"backend"`` or ``"backend:variant"``::

    inspyrenet                  # default, transparent-background (InSpyReNet)
    inspyrenet:fast
    rembg:u2net                 # any rembg model name
    rembg:isnet-anime
    birefnet                    # ZhengPeng7/BiRefNet via transformers
    birefnet:ZhengPeng7/BiRefNet_lite
    rmbg                        # briaai/RMBG-2.0 via transformers
    ben2                        # PramaLLC/BEN2

Backends import their heavy dependencies lazily, so listing and selecting
models never pulls in torch/onnxruntime until inference happens.

Third parties can plug in their own models either at runtime::

    from background_remove_sdk.models import register_backend
    register_backend("mymodel", MyBackend)   # subclass of BaseBackend

or by distributing a package exposing the ``background_remove_sdk.backends``
entry-point group.
"""

from __future__ import annotations

from importlib import import_module, metadata
from typing import Dict, Optional, Tuple, Type, Union

from background_remove_sdk.models.base import (
    BaseBackend,
    ModelNotInstalledError,
    compose_rgba,
    to_mask,
)

ENTRY_POINT_GROUP = "background_remove_sdk.backends"

DEFAULT_MODEL = "inspyrenet"

# name -> "module:Class" (lazy) or a BaseBackend subclass
_registry: Dict[str, Union[str, Type[BaseBackend]]] = {
    "inspyrenet": "background_remove_sdk.models.inspyrenet:InSpyReNetBackend",
    "rembg": "background_remove_sdk.models.rembg_backend:RembgBackend",
    "birefnet": "background_remove_sdk.models.hf_segmentation:BiRefNetBackend",
    "rmbg": "background_remove_sdk.models.hf_segmentation:RMBGBackend",
    "ben2": "background_remove_sdk.models.ben2_backend:BEN2Backend",
}


def register_backend(name: str, backend: Union[str, Type[BaseBackend]]) -> None:
    """Register a custom backend class (or lazy ``"module:Class"`` path)."""
    if isinstance(backend, type) and not issubclass(backend, BaseBackend):
        raise TypeError(f"{backend!r} must subclass BaseBackend")
    _registry[name] = backend


def parse_model_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``"backend:variant"`` into ``(backend, variant)``."""
    name, _, variant = spec.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid model spec: {spec!r}")
    return name, (variant.strip() or None)


def _resolve(entry: Union[str, Type[BaseBackend]]) -> Type[BaseBackend]:
    if isinstance(entry, type):
        return entry
    module_path, _, class_name = entry.partition(":")
    return getattr(import_module(module_path), class_name)


def get_backend_class(name: str) -> Type[BaseBackend]:
    """Resolve a backend name to its class, consulting entry points too."""
    if name in _registry:
        cls = _resolve(_registry[name])
        _registry[name] = cls  # cache the resolved class
        return cls
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name == name:
            cls = entry_point.load()
            _registry[name] = cls
            return cls
    raise ValueError(
        f"Unknown model backend: {name!r}. Available: {sorted(_registry)}"
    )


def create_backend(
    spec: str = DEFAULT_MODEL, device: Optional[str] = None, **options
) -> BaseBackend:
    """Instantiate a backend from a ``"backend[:variant]"`` spec string."""
    name, variant = parse_model_spec(spec)
    return get_backend_class(name)(variant=variant, device=device, **options)


def list_models() -> Dict[str, dict]:
    """Describe all registered backends without importing their deps."""
    info = {}
    for name in sorted(_registry):
        cls = get_backend_class(name)
        info[name] = {
            "description": cls.description,
            "default_variant": cls.default_variant,
            "variants": list(cls.variants),
            "install": cls.install_hint,
        }
    return info


__all__ = [
    "BaseBackend",
    "ModelNotInstalledError",
    "DEFAULT_MODEL",
    "register_backend",
    "parse_model_spec",
    "get_backend_class",
    "create_backend",
    "list_models",
    "to_mask",
    "compose_rgba",
]
