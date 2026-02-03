# Environment Setup and Usage Guide

To ensure consistent behavior and avoid dependency conflicts, this project uses a dedicated Python virtual environment (`venv`).

## Quick Start

1. **Setup Environment**:
   Run this script once to create the virtual environment and install all dependencies.
   ```bash
   ./setup_env.sh
   ```

2. **Run Application**:
   Use this script to launch the application. It automatically handles activating the environment.
   ```bash
   ./run_app.sh
   ```
   
   You can pass arguments to the app as well:
   ```bash
   ./run_app.sh --port 7860 --share
   ```

## Detailed Information

- The virtual environment is created in the `venv/` directory.
- `setup_env.sh` installs dependencies listed in `pyproject.toml` and ensures `demucs` is installed.
- `run_app.sh` is a wrapper that sources `venv/bin/activate` before running `gradio_app_full.py`.

## Troubleshooting

If you encounter `ModuleNotFoundError`, it is likely that the environment is not activated or dependencies are missing. Try running `./setup_env.sh` again to repair the environment.
