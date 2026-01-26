import sys
import os

# Add the current directory to sys.path to ensure qwen_tts can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_tts.cli.demo import main

if __name__ == "__main__":
    print("Starting Qwen3-TTS Gradio App...")
    
    # If no arguments provided, default to the 1.7B Base model
    if len(sys.argv) == 1:
        # Default configuration
        DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        PORT = "8000"
        HOST = "127.0.0.1"
        
        print(f"No arguments provided. Launching with default model: {DEFAULT_MODEL}")
        print(f"You can also run this script with arguments, e.g.: python gradio_app.py Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        import torch
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        elif torch.cuda.is_available():
            DEVICE = "cuda:0"
        else:
            DEVICE = "cpu"
            
        print(f"Auto-detected device: {DEVICE}")

        # Simulate command line arguments
        sys.argv = [
            sys.argv[0], 
            DEFAULT_MODEL, 
            "--port", PORT, 
            "--ip", HOST,
            "--no-flash-attn",
            "--device", DEVICE
        ]
        
    sys.exit(main())
