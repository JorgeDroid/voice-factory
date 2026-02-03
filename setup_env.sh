#!/bin/bash

# setup_env.sh
# Creates a virtual environment and installs dependencies for Qwen3-TTS

set -e  # Exit on error

VENV_DIR="venv"

echo "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
# Install the current directory in editable mode to get dependencies from pyproject.toml
pip install -e .

# Explicitly ensure demucs is installed if it wasn't picked up for some reason (though it is in pyproject.toml now)
if ! pip show demucs &> /dev/null; then
    echo "Installing demucs..."
    pip install demucs
fi

echo "Environment setup complete!"
echo "To run the app, use: ./run_app.sh"
