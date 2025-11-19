## User Guide

Follow the steps below to recreate the project environment with `uv`.

1. **Create the virtual environment**
   - Run `uv venv` from the repository root. This reads `pyproject.toml` and provisions a `.venv` folder with Python 3.11+.
2. **Activate the environment**
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
3. **Install the project requirements**
   - With the virtual environment active, run `uv pip install -r pyproject.toml` to install the dependencies declared in `pyproject.toml`. If you prefer deterministic installs from `uv.lock`, run `uv pip sync`.

The environment is now ready for development or running the project scripts.
