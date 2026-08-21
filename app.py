"""Hugging Face Space entrypoint."""
from __future__ import annotations

import importlib.util
from pathlib import Path


# Compatibility fix for Gradio 4.44.x / gradio_client
try:
    import gradio_client.utils as _gc_utils

    _original_get_type = _gc_utils.get_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _original_get_type(schema)

    _gc_utils.get_type = _safe_get_type

    if hasattr(_gc_utils, "_json_schema_to_python_type"):
        _original_json_schema = _gc_utils._json_schema_to_python_type

        def _safe_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            return _original_json_schema(schema, defs)

        _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

except Exception:
    pass


_APP_PATH = Path(__file__).resolve().parent / "app" / "app.py"
_SPEC = importlib.util.spec_from_file_location(
    "peptide_detectability_gradio",
    _APP_PATH,
)

if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load Gradio app from {_APP_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

demo = _MODULE.build_app()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )