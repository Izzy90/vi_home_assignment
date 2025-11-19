## User Guide

Follow the steps below to recreate the project environment with `uv`.

1. **Create the virtual environment**
   - Run `uv venv` from the repository root. This reads `pyproject.toml` and provisions a `.venv` folder with Python 3.11+.
2. **Activate the environment**
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
3. **Install the project requirements**
   - With the virtual environment active, run `uv pip install -r pyproject.toml` to install the dependencies declared in `pyproject.toml`. If you prefer deterministic installs from `uv.lock`, run `uv pip sync`.

To run the main script, use: `python -m main`

### Runtime options

- `--min-recall`: minimum recall threshold enforced when searching for the precision-maximizing cutoff (default `0.05`, configurable in `config.py`). Example:
  ```
  python -m main --min-recall 0.1
  ```

## Solution Description

After choosing a minimum recall (what percent of churn we want to fix), the model will create a list of N members that are most likely to churn. The list can be found in `outputs\top_N_member_ids_test.csv`