
import os
import sys
import torch
import demucs.separate
import shlex

def isolate_voice(audio_path, output_dir=None, model_name="htdemucs", device=None):
    """
    Separates vocals from the given audio file using Demucs.
    
    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str, optional): Directory to save the output. Defaults to 'separated' folder in the same dir as input.
        model_name (str, optional): Demucs model to use. Defaults to "htdemucs".
        device (str, optional): Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None.
        
    Returns:
        str: Path to the isolated vocals wav file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(audio_path)), "separated")
    
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # MPS support in Demucs/PyTorch can be unstable (sparse tensor errors), defaulting to CPU
            device = "cpu"
    
    print(f"Isolating voice from {audio_path} using model {model_name} on {device}...")

    # Construct arguments for Demucs
    # We use the two-stems option to get 'vocals' and 'no_vocals' (accompaniment)
    args = [
        "-n", model_name,
        "--two-stems", "vocals",
        "-o", output_dir,
        "-d", device,
        audio_path
    ]
    
    # demucs.separate.main expects a list of strings (sys.argv style)
    try:
        demucs.separate.main(args)
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}")

    # Expected output path logic from Demucs:
    # {output_dir}/{model_name}/{track_name}/vocals.wav
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_dir, model_name, track_name, "vocals.wav")
    
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Expected output file not found at: {vocals_path}")
        
    return vocals_path

if __name__ == "__main__":
    # fast test
    if len(sys.argv) > 1:
        print(isolate_voice(sys.argv[1]))
    else:
        print("Usage: python voice_isolation.py <audio_file>")
