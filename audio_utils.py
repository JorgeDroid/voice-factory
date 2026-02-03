import os
import subprocess
import whisper
import torch
import warnings

def extract_audio_from_video(video_path, output_dir=None):
    """
    Extracts audio from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str, optional): Output directory.
        
    Returns:
        str: Path to the extracted .wav file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "extracted_audio")
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.wav")
    
    cmd = [
        "ffmpeg", 
        "-y", # Overwrite
        "-i", video_path, 
        "-vn", # No video
        "-acodec", "pcm_s16le", 
        "-ar", "44100", 
        "-ac", "2", 
        output_path
    ]
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")
        
    return output_path

def transcribe_audio(audio_path, model_size="base", device=None, language=None):
    """
    Transcribes audio using OpenAI Whisper.
    
    Args:
        audio_path (str): Path to audio file.
        model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large").
        device (str): "cuda", "cpu", or "mps".
        language (str): Optional language code.
    
    Returns:
        str: Transcribed text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # MPS support in Whisper/PyTorch can be unstable, defaulting to CPU
            device = "cpu"
            
    # Whisper loads model to device
    print(f"Loading Whisper model '{model_size}' on {device}...")
    
    # Suppress FP16 warning on CPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = whisper.load_model(model_size, device=device)
        
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, language=language)
    
    return result["text"].strip()

def clip_audio(audio_path, start_sec, end_sec, output_dir=None):
    """
    Clips audio from start_sec to end_sec using ffmpeg.
    
    Args:
        audio_path (str): Path to input audio.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
        output_dir (str, optional): Output directory.
        
    Returns:
        str: Path to the clipped audio file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(audio_path)), "clipped_audio")
        
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_clip_{start_sec}-{end_sec}.wav")
    
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")
        
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", audio_path,
        "-t", str(duration),
        "-c", "copy", 
        output_path
    ]
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")
        
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing with {test_file}")
        
        # Test 1: Extract
        if test_file.endswith(".mp4"):
            wav_path = extract_audio_from_video(test_file)
            print(f"Extracted: {wav_path}")
            test_file = wav_path # Update to use for next steps
            
        # Test 2: Clip
        clipped = clip_audio(test_file, 0.5, 2.5)
        print(f"Clipped: {clipped}")
        
        # Test 3: Transcribe
        try:
            # dummy transcription on sine wave might be garbage but ensures code runs
            txt = transcribe_audio(clipped, model_size="tiny")
            print(f"Transcribed: '{txt}'")
        except Exception as e:
            print(f"Transcription failed (expected if no ffmpeg/whisper): {e}")
