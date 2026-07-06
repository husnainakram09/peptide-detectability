"""Hugging Face Space entrypoint."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_APP_PATH = Path(__file__).resolve().parent / "app" / "app.py"
_SPEC = importlib.util.spec_from_file_location("peptide_detectability_gradio", _APP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load Gradio app from {_APP_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

demo = _MODULE.build_app()


if __name__ == "__main__":
    demo.launch()
