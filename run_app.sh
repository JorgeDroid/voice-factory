#!/bin/bash

# run_app.sh
# Launches the Qwen3-TTS Gradio app using the virtual environment

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Please run ./setup_env.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Launching Gradio App..."
# Pass any arguments to the python script
python gradio_app_full.py "$@"
