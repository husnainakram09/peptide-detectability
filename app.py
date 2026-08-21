"""Hugging Face Space entrypoint for the peptide detectability predictor."""

from app.app import build_app

demo = build_app()

if __name__ == "__main__":
    # Hugging Face Spaces supplies the server environment/port.
    demo.launch()
