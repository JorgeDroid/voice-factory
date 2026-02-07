import argparse
import os
import sys
import tempfile
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import librosa
import soundfile as sf
from diffusers import StableAudioPipeline, EulerDiscreteScheduler
from dotenv import load_dotenv

load_dotenv()

# Add current directory to path so we can import qwen_tts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from voice_isolation import isolate_voice
from audio_utils import extract_audio_from_video, transcribe_audio, clip_audio

# --- Utils copied/adapted from demo.py ---

def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])

def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16 # Default

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav

def get_saved_voices() -> List[str]:
    """Scandirectory for .pt files and return sorted list of names."""
    if not os.path.exists("saved_voices"):
        os.makedirs("saved_voices", exist_ok=True)
    files = [f for f in os.listdir("saved_voices") if f.endswith(".pt")]
    files.sort()
    return files

STYLE_FILE = "saved_styles.json"
def get_style_presets():
    if not os.path.exists(STYLE_FILE): return {}
    try:
        with open(STYLE_FILE, 'r') as f: return json.load(f)
    except: return {}

# --- Model Management ---

class GlobalModelManager:
    def __init__(self):
        self.tts_model = None
        self.music_pipe = None
        self.current_mode = None # "tts" or "music"

    def load_tts(self):
        if self.current_mode == "tts" and self.tts_model is not None:
            return self.tts_model
            
        print("Switching to TTS Model...")
        # Unload Music
        if self.music_pipe is not None:
            del self.music_pipe
            self.music_pipe = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # MPS cache clearing if needed
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        # Load TTS
        print("Loading Voice Design Model (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)...")
        self.tts_model = Qwen3TTSModel(voice_design_model_name="Qwen/Qwen2.5-1.5B-Instruct", 
                                       base_model_name="Qwen/Qwen2.5-1.5B-Instruct") 
        # Note: Using placeholders or actual paths based on existing logic? 
        # Wait, the original code had specific paths. Let's revert to checking original code for init.
        # Original: model = Qwen3TTSModel(voice_design_model_name="...", base_model_name="...")
        # Actually I should check how it was initialized before.
        
        self.current_mode = "tts"
        return self.tts_model

    def load_music(self):
        if self.current_mode == "music" and self.music_pipe is not None:
            return self.music_pipe
            
        print("Switching to Music Model...")
        # Unload TTS
        if self.tts_model is not None:
            del self.tts_model
            self.tts_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        # Load Music
        # FORCE CPU to avoid torchsde recursion error on MPS (Mac)
        device = "cpu" 
        dtype = torch.float32
        
        print(f"Loading Stable Audio Open on {device} (forced for stability)...")
        token = os.getenv("HF_TOKEN")
        self.music_pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            torch_dtype=dtype,
            token=token
        )
        # SWAP SCHEDULER to EulerDiscreteScheduler to avoid torchsde recursion error on MPS/CPU
        self.music_pipe.scheduler = EulerDiscreteScheduler.from_config(self.music_pipe.scheduler.config)
        self.music_pipe.to(device)
        
        self.current_mode = "music"
        return self.music_pipe

manager = GlobalModelManager()

def load_voice_prompt_items(voice_filename):
    if not voice_filename: return None, "No voice selected."
    path = os.path.join("saved_voices", voice_filename)
    if not os.path.exists(path): return None, "Voice file not found."
    
    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "items" not in payload:
            return None, "Invalid file format."

        items = []
        for d in payload["items"]:
            ref_code = d.get("ref_code")
            if ref_code is not None and not torch.is_tensor(ref_code): ref_code = torch.tensor(ref_code)
            ref_spk_embedding = d.get("ref_spk_embedding")
            if ref_spk_embedding is not None and not torch.is_tensor(ref_spk_embedding): ref_spk_embedding = torch.tensor(ref_spk_embedding)
            
            items.append(VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=d.get("x_vector_only_mode", False),
                icl_mode=d.get("icl_mode", True),
                ref_text=d.get("ref_text")
            ))
        return items, None
    except Exception as e:
        return None, f"Error loading voice: {str(e)}"

# --- Projects Logic (Global) ---
PROJECTS_DIR = "projects"
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR, exist_ok=True)

def get_projects():
    if not os.path.exists(PROJECTS_DIR): return []
    return sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))])

def create_project(name):
    print(f"DEBUG: create_project called with name='{name}'")
    if not name or not name.strip(): return gr.update(), "Project name required."
    safe_name = "".join(c for c in name.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
    path = os.path.join(PROJECTS_DIR, safe_name)
    if os.path.exists(path): return gr.update(), "Project already exists."
    try:
        os.makedirs(path, exist_ok=True)
        # Create empty full_story.txt
        with open(os.path.join(path, "full_story.txt"), "w") as f: f.write("")
        return gr.update(choices=get_projects(), value=safe_name), f"Created {safe_name}"
    except Exception as e:
        return gr.update(), str(e)

def load_full_story(project_name):
    if not project_name: return ""
    path = os.path.join(PROJECTS_DIR, project_name, "full_story.txt")
    if os.path.exists(path):
        with open(path, "r") as f: return f.read()
    return ""

def parse_chapters(project_name, text):
    if not text: return 0
    # Split by delimiter `###`
    chunks = text.split('###')
    chapters = [c.strip() for c in chunks if c.strip()]
    
    # Save to files
    chapters_dir = os.path.join(PROJECTS_DIR, project_name, "chapters")
    if os.path.exists(chapters_dir):
        import shutil
        shutil.rmtree(chapters_dir)
    os.makedirs(chapters_dir, exist_ok=True)
    
    count = 0
    for i, content in enumerate(chapters):
        first_line = content.split('\n')[0].strip()
        safe_title = "".join(c for c in first_line[:30] if c.isalnum() or c in (' ', '_', '-')).strip()
        if not safe_title: safe_title = f"Chapter_{i+1}"
        
        filename = f"{i+1:02d}_{safe_title}.txt"
        with open(os.path.join(chapters_dir, filename), "w") as f:
            f.write(content)
        count += 1
    return count

def get_chapter_list(project_name):
    if not project_name: return []
    path = os.path.join(PROJECTS_DIR, project_name, "chapters")
    if not os.path.exists(path): return []
    files = sorted([f for f in os.listdir(path) if f.endswith(".txt")])
    return files

def load_chapter_content(project_name, chapter_filename):
    if not project_name or not chapter_filename: return ""
    path = os.path.join(PROJECTS_DIR, project_name, "chapters", chapter_filename)
    if os.path.exists(path):
        with open(path, "r") as f: return f.read()
    return ""

def load_project_settings(project_name):
    if not project_name: return {}, None, None, None, 1
    path = os.path.join(PROJECTS_DIR, project_name, "settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f: 
                data = json.load(f)
                return data, data.get("voice"), data.get("style"), data.get("lang", "Auto"), data.get("spacing", 1)
        except: pass
    return {}, None, None, "Auto", 1

def save_project_settings(project_name, voice, style, lang, spacing):
    if not project_name: return "No project selected."
    path = os.path.join(PROJECTS_DIR, project_name, "settings.json")
    data = {"voice": voice, "style": style, "lang": lang, "spacing": spacing}
    try:
        with open(path, "w") as f: json.dump(data, f, indent=2)
        return "Settings saved."
    except Exception as e:
        return str(e)

def save_full_story(project_name, text):
    if not project_name: return "No project selected.", gr.update(choices=[])
    
    path = os.path.join(PROJECTS_DIR, project_name, "full_story.txt")
    try:
        with open(path, "w") as f: f.write(text)
        n_chapters = parse_chapters(project_name, text)
        return f"Saved. Created {n_chapters} chapters.", gr.update(choices=get_chapter_list(project_name))
    except Exception as e:
        return f"Error: {e}", gr.update()

# --- Main Building Logic ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-tts-demo-full")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    # Using 'cuda:0' default but code will auto-detect MPS on mac
    parser.add_argument("--device", default="cuda:0") 
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser

def build_full_demo(tts_clone: Qwen3TTSModel, tts_design: Qwen3TTSModel) -> gr.Blocks:
    # Common kwargs
    gen_kwargs = {
        "max_new_tokens": None, # Default
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
    }

    # Language/Speaker choices (from Base model)
    supported_langs = tts_clone.model.get_supported_languages() if hasattr(tts_clone.model, "get_supported_languages") else []
    # Base model likely doesn't have preset speakers, so we can ignore spk_map or keep it empty
    
    lang_choices_disp, lang_map = _build_choices_and_map(supported_langs)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3 TTS Unified Demo") as demo:
        gr.Markdown("# Qwen3 TTS Unified Demo")
        gr.Markdown(
            "This demo integrates **Voice Design** (Prompt-to-Speech) and **Custom Voice** (Zero-shot Voice Cloning)."
        )

        # --- GLOBAL PROJECT SELECTOR ---
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                prj_dropdown = gr.Dropdown(
                    label="Active Project", 
                    choices=get_projects(), 
                    interactive=True,
                    value=get_projects()[0] if get_projects() else None
                )
            with gr.Column(scale=1):
                prj_refresh_global = gr.Button("Refresh Projects")

        prj_refresh_global.click(lambda: gr.update(choices=get_projects()), outputs=[prj_dropdown])

        with gr.Tabs():
            # --- TAB 1: VOICE DESIGN ---
            with gr.Tab("Voice Design (Prompt-to-Speech)"):
                gr.Markdown("### Generate speech from text description (e.g., 'A young woman speaking excitedly')")
                with gr.Row():
                    with gr.Column(scale=2):
                        vd_text = gr.Textbox(
                            label="Text", 
                            lines=4, 
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                            placeholder="Enter text to synthesize."
                        )
                        vd_design = gr.Textbox(
                            label="Voice Description", 
                            lines=3, 
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                            placeholder="Describe the voice and tone."
                        )
                        vd_lang = gr.Dropdown(
                            label="Language",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True
                        )
                        vd_btn = gr.Button("Generate Voice Design", variant="primary")
                    
                    with gr.Column(scale=3):
                        vd_out = gr.Audio(label="Output Audio", type="numpy")
                        vd_status = gr.Textbox(label="Status", lines=2)

                def run_voice_design(text, design, lang_disp):
                    if not text.strip(): return None, "Text is required."
                    if not design.strip(): return None, "Description is required."
                    try:
                        lang = lang_map.get(lang_disp, "Auto")
                        wavs, sr = tts_design.generate_voice_design(
                            text=text.strip(),
                            language=lang,
                            instruct=design.strip(),
                            **gen_kwargs
                        )
                        return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                    except Exception as e:
                        return None, str(e)

                vd_btn.click(run_voice_design, inputs=[vd_text, vd_design, vd_lang], outputs=[vd_out, vd_status])

            # --- TAB 2: CUSTOM VOICE (CLONE) ---
            with gr.Tab("Custom Voice (Voice Cloning)"):
                gr.Markdown("### Clone a voice from reference audio (Zero-shot)")
                with gr.Row():
                    with gr.Column(scale=2):
                        vc_ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                        vc_ref_text = gr.Textbox(
                            label="Reference Text", 
                            lines=2, 
                            placeholder="Transcription of the reference audio (required unless using x-vector only)."
                        )
                        vc_xvec = gr.Checkbox(label="Use x-vector only (Experimental)", value=False)
                        
                        vc_text = gr.Textbox(
                            label="Target Text", 
                            lines=4, 
                            placeholder="Enter text to synthesize."
                        )
                        vc_lang = gr.Dropdown(
                            label="Language",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True
                        )
                        vc_instruct = gr.Textbox(
                            label="Style Instruction (Optional)", 
                            lines=2, 
                            placeholder="e.g. Speak slowly, Say it sadly",
                            info="Control the pace and emotion natively."
                        )
                        vc_btn = gr.Button("Generate Custom Voice", variant="primary")

                    with gr.Column(scale=3):
                        vc_out = gr.Audio(label="Output Audio", type="numpy")
                        vc_status = gr.Textbox(label="Status", lines=2)

                def run_custom_voice(ref_aud, ref_txt, xvec_mode, text, lang_disp, instruct):
                    if not text.strip(): return None, "Target text is required."
                    at = _audio_to_tuple(ref_aud)
                    if not at: return None, "Reference audio is required."
                    if not xvec_mode and not ref_txt.strip():
                        return None, "Reference text is required (or enable x-vector only)."
                    
                    try:
                        lang = lang_map.get(lang_disp, "Auto")
                        
                        # Handle Instruction manually for Base model
                        kwargs = dict(gen_kwargs)
                        if instruct and instruct.strip():
                            # Tokenize instruction exactly like the model does internally for other modes
                            instruct_text = f"<|im_start|>user\n{instruct.strip()}<|im_end|>\n"
                            # tts_clone._tokenize_texts returns list of tensors
                            instruct_ids = tts_clone._tokenize_texts([instruct_text])[0]
                            # Pass as list of tensors (batch size 1)
                            kwargs["instruct_ids"] = [instruct_ids]

                        wavs, sr = tts_clone.generate_voice_clone(
                            text=text.strip(),
                            language=lang,
                            ref_audio=at,
                            ref_text=ref_txt.strip() if ref_txt else None,
                            x_vector_only_mode=xvec_mode,
                            **kwargs
                        )
                        return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                    except Exception as e:
                        return None, str(e)

                vc_btn.click(run_custom_voice, inputs=[vc_ref_audio, vc_ref_text, vc_xvec, vc_text, vc_lang, vc_instruct], outputs=[vc_out, vc_status])

            # --- TAB 3: SAVE / LOAD VOICE ---
            with gr.Tab("Save / Load Voice"):
                gr.Markdown("### Voice Library Management")
                
                # Refresh state
                saved_voices_list = get_saved_voices()
                
                # Style helper
                # Style helper
                style_presets = get_style_presets()
                style_keys = sorted(list(style_presets.keys()))
                style_keys = sorted(list(style_presets.keys()))

                with gr.Row():
                    # SAVE SECTION
                    with gr.Column(scale=2):
                        gr.Markdown("#### Create New Voice")
                        sl_ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                        sl_ref_text = gr.Textbox(
                            label="Reference Text", 
                            lines=2, 
                            placeholder="Transcription (required unless using x-vector only)."
                        )
                        sl_name_in = gr.Textbox(label="Voice Name", placeholder="e.g. MyNarrator")
                        # Removed per-voice default style input based on feedback
                        sl_xvec = gr.Checkbox(label="Use x-vector only", value=False)
                        sl_save_btn = gr.Button("Save Voice to Library", variant="primary")
                        sl_save_status = gr.Textbox(label="Save Status", lines=1)

                    # LOAD SECTION
                    with gr.Column(scale=2):
                        gr.Markdown("#### Load & Generate")
                        sl_voice_dropdown = gr.Dropdown(
                            label="Select Saved Voice", 
                            choices=saved_voices_list, 
                            value=saved_voices_list[0] if saved_voices_list else None,
                            interactive=True
                        )
                        sl_refresh_btn = gr.Button("Refresh List", size="sm")
                        
                        sl_text = gr.Textbox(label="Target Text", lines=4, placeholder="Enter text to synthesize.")
                        
                        # Style Presets UI
                        with gr.Row():
                            sl_style_dropdown = gr.Dropdown(
                                label="Load Style Preset", 
                                choices=style_keys, 
                                value=None,
                                interactive=True,
                                scale=3
                            )
                        
                        sl_instruct = gr.Textbox(label="Style Instruction", lines=2, placeholder="e.g. Say it sadly (Optional)")
                        
                        with gr.Accordion("Save Current Style as Preset", open=False):
                            with gr.Row():
                                sl_new_style_name = gr.Textbox(label="Preset Name", placeholder="e.g. Sad Whisper", scale=3)
                                sl_save_style_btn = gr.Button("Save Preset", scale=1)
                        
                        with gr.Row():
                            sl_lang = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True, scale=2)
                            sl_speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1, scale=3)
                        
                        sl_gen_btn = gr.Button("Generate from Selection", variant="primary")
                        
                    with gr.Column(scale=3):
                        sl_out = gr.Audio(label="Output Audio", type="numpy")
                        sl_gen_status = gr.Textbox(label="Generate Status", lines=2)

                def save_named_voice(ref_aud, ref_txt, name, xvec_mode):
                    if not name or not name.strip(): return None, "Voice name is required."
                    at = _audio_to_tuple(ref_aud)
                    if not at: return None, "Reference audio is required."
                    if not xvec_mode and not ref_txt.strip():
                        return None, "Reference text is required."
                    
                    try:
                        items = tts_clone.create_voice_clone_prompt(
                            ref_audio=at,
                            ref_text=ref_txt.strip() if ref_txt else None,
                            x_vector_only_mode=xvec_mode
                        )
                        
                        filename = "".join(c for c in name.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
                        if not filename: filename = "unnamed_voice"
                        out_path = os.path.join("saved_voices", f"{filename}.pt")
                        
                        # Reverted payload to simple items list (or dict without default_instruction)
                        payload = {"items": [asdict(it) for it in items]}
                        torch.save(payload, out_path)
                        
                        new_list = get_saved_voices()
                        # Return dict for outputs: status, dropdown
                        return {
                            sl_save_status: f"Saved to {out_path}",
                            sl_voice_dropdown: gr.update(choices=new_list, value=f"{filename}.pt")
                        }
                    except Exception as e:
                        return {sl_save_status: str(e)}

                def load_named_voice(voice_filename, text, instruct, lang_disp, speed):
                    if not voice_filename: return None, "Please select a voice."
                    if not text.strip(): return None, "Text is required."
                    
                    try:
                        items, err = load_voice_prompt_items(voice_filename)
                        if err: return None, err

                        # Prepare generation kwargs (reuse similar logic or keep it here if simple)
                        # We need to construct kwargs for generate_voice_clone
                        gen_kwargs = {}
                        if instruct and instruct.strip():
                             print(f"DEBUG: Using style instruction: '{instruct.strip()}'")
                             instruct_text = f"<|im_start|>user\n{instruct.strip()}<|im_end|>\n"
                             instruct_ids = tts_clone._tokenize_texts([instruct_text])[0]
                             gen_kwargs["instruct_ids"] = [instruct_ids]
                        
                        lang = lang_map.get(lang_disp, "Auto")
                        
                        wavs, sr = tts_clone.generate_voice_clone(
                            text=text.strip(),
                            language=lang,
                            voice_clone_prompt=items,
                            **gen_kwargs
                        )
                        
                        final_wav = wavs[0]
                        if speed != 1.0:
                             final_wav = librosa.effects.time_stretch(final_wav, rate=speed)
                             
                        return _wav_to_gradio_audio(final_wav, sr), "Finished."

                    except Exception as e:
                        return None, str(e)

                def refresh_list():
                    new_list = get_saved_voices()
                    return gr.update(choices=new_list)
                
                # Style Preset Logic
                def on_style_change(preset_name):
                    presets = get_style_presets()
                    return presets.get(preset_name, "")
                
                def save_preset(name, text):
                    if not name or not name.strip():
                        return gr.update(), "Name required."
                    if not text or not text.strip():
                        return gr.update(), "Instruction text required."
                    
                    presets = get_style_presets()
                    presets[name.strip()] = text.strip()
                    try:
                        with open(STYLE_FILE, 'w') as f:
                            json.dump(presets, f, indent=2)
                    except Exception as e:
                        return gr.update(), f"Error saving: {e}"
                        
                    new_keys = sorted(list(presets.keys()))
                    return gr.update(choices=new_keys, value=name.strip()), "Preset saved."

                sl_save_btn.click(
                    save_named_voice, 
                    inputs=[sl_ref_audio, sl_ref_text, sl_name_in, sl_xvec], 
                    outputs=[sl_save_status, sl_voice_dropdown]
                )
                
                sl_refresh_btn.click(refresh_list, outputs=[sl_voice_dropdown])
                
                sl_gen_btn.click(
                    load_named_voice, 
                    inputs=[sl_voice_dropdown, sl_text, sl_instruct, sl_lang, sl_speed], 
                    outputs=[sl_out, sl_gen_status]
                )
                
                sl_style_dropdown.change(
                    on_style_change,
                    inputs=[sl_style_dropdown],
                    outputs=[sl_instruct]
                )
                
                sl_save_style_btn.click(
                    save_preset,
                    inputs=[sl_new_style_name, sl_instruct],
                    outputs=[sl_style_dropdown, sl_gen_status]
                )

            # --- TAB 4: PROJECTS ---
            with gr.Tab("Projects"):
                gr.Markdown("### Project Management")
                
                
                # Functions moved to global scope


                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Project Selected Above**")
                        # prj_dropdown and refresh moved to global

                    with gr.Column(scale=1):
                        with gr.Accordion("Create New Project", open=False):
                            prj_new_name = gr.Textbox(label="New Project Name")
                            prj_create_btn = gr.Button("Create")
                            prj_status = gr.Textbox(label="Status", lines=1)

                # Voice Settings Section
                with gr.Accordion("Project Voice Settings", open=True):
                    with gr.Row():
                        prj_voice_dropdown = gr.Dropdown(label="Project Voice", choices=get_saved_voices(), interactive=True)
                        # Reuse saved_styles global dict -> use get_style_presets()
                        style_keys = sorted(list(get_style_presets().keys()))
                        prj_style_dropdown = gr.Dropdown(label="Project Style", choices=style_keys, interactive=True)
                        prj_lang_dropdown = gr.Dropdown(label="Language", choices=["Auto", "English", "Spanish", "Chinese", "Japanese", "Korean", "French", "German"], value="Auto", interactive=True)
                        prj_spacing_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Chapter Spacing", value=1, interactive=True)
                        prj_save_settings_btn = gr.Button("Save Settings")
                    prj_settings_status = gr.Textbox(label="Settings Status", lines=1)

                gr.Markdown("---")
                
                # 3-Column Layout
                with gr.Row():
                    # Column 1: Full Story
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Full Story")
                        fs_text = gr.TextArea(label="Story Text", lines=20, placeholder="Write your full story here...\nUse '### Chapter Name' to split.", show_copy_button=True)
                        fs_save_btn = gr.Button("Save Full Story & Split", variant="primary")
                        fs_status = gr.Textbox(label="Save Status", lines=1)
                    
                    # Column 2: Chapters
                    with gr.Column(scale=1):
                        gr.Markdown("#### 2. Chapters")
                        ch_dropdown = gr.Dropdown(label="Select Chapter", choices=[], interactive=True)
                        ch_refresh_btn = gr.Button("Refresh Chapters", size="sm")
                        ch_text = gr.TextArea(label="Chapter Content", lines=10, interactive=False) 
                        with gr.Row():
                            ch_gen_btn = gr.Button("Generate Clip", variant="primary", scale=2)
                            ch_gen_3_btn = gr.Button("Gen 3 Takes", variant="secondary", scale=1)
                        
                        ch_take_radio = gr.Radio(label="Select Take", choices=[], value=None, interactive=True, type="value")
                        ch_audio = gr.Audio(label="Chapter Audio", type="filepath", interactive=True)
                        ch_save_trim_btn = gr.Button("Save Trim", size="sm")
                        ch_status = gr.Textbox(label="Generation Status", lines=1)
                        
                    # Column 3: Playlist
                    with gr.Column(scale=1):
                        gr.Markdown("#### 3. Final Cut Playlist")
                        playlist_gen_btn = gr.Button("Generated Missing Takes", variant="secondary")
                        playlist_log = gr.Textbox(label="Generation Log", lines=5, interactive=False)
                        
                        playlist_reset_btn = gr.Button("Reset All Takes", variant="stop")
                        with gr.Row(visible=False) as playlist_reset_confirm_row:
                             playlist_reset_confirm_btn = gr.Button("Confirm Delete", variant="stop")
                             playlist_reset_cancel_btn = gr.Button("Cancel")
                        
                        gr.Markdown("---")
                        playlist_export_btn = gr.Button("Export Final Audio", variant="primary")
                        playlist_export_audio = gr.Audio(label="Final Export", interactive=False)
                        
                        playlist_df = gr.Dataframe(
                            headers=["Chapter", "Selected Take", "Path"],
                            datatype=["str", "str", "str"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            wrap=True,
                            visible=True
                        )
                        playlist_audio = gr.Audio(label="Preview Final Cut", type="filepath")

                # Events
                # prj_refresh_btn moved global



                
                prj_create_btn.click(
                    create_project,
                    inputs=[prj_new_name],
                    outputs=[prj_dropdown, prj_status]
                )
                
                def get_chapter_takes(project_name, chapter_filename):
                    if not project_name or not chapter_filename: return [], None
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    if not os.path.exists(clips_dir): return [], None
                    
                    base_name = os.path.splitext(chapter_filename)[0]
                    # Look for {base_name}_v{N}.wav
                    takes = []
                    
                    # Also check for legacy file without version
                    legacy_path = os.path.join(clips_dir, chapter_filename.replace(".txt", ".wav"))
                    if os.path.exists(legacy_path):
                        # Treat as v1 if no v1 exists, or just include it?
                        # User wants radio buttons. Let's look specifically for our version schema or the base one.
                        # Simple approach: Check v1, v2, v3.
                        pass
                        
                    # Let's strictly check v1, v2, v3
                    # If we find only the legacy file, we can treat it as v1
                    found_legacy = os.path.exists(legacy_path)
                    
                    for i in range(1, 4):
                        v_name = f"{base_name}_v{i}.wav"
                        v_path = os.path.join(clips_dir, v_name)
                        if os.path.exists(v_path):
                            takes.append(f"Take {i}")
                        elif i == 1 and found_legacy:
                             # If v1 doesn't exist but legacy does, show it as Take 1
                             takes.append("Take 1")
                    
                    return takes, legacy_path

                def generate_chapter_clip(project_name, chapter_filename, voice_name, style_name, language="Auto"):
                    if not project_name: return None, gr.update(), "No project selected."
                    if not chapter_filename: return None, gr.update(), "No chapter selected."
                    if not voice_name: return None, gr.update(), "No voice selected in Project Settings."
                    
                    # 1. Load Chapter Text
                    text = load_chapter_content(project_name, chapter_filename)
                    if not text: return None, gr.update(), "Chapter is empty."
                    
                    # 2. Determine Version
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    os.makedirs(clips_dir, exist_ok=True)
                    base_name = os.path.splitext(chapter_filename)[0]
                    
                    next_version = 1
                    for i in range(1, 4):
                        v_name = f"{base_name}_v{i}.wav"
                        if not os.path.exists(os.path.join(clips_dir, v_name)):
                            next_version = i
                            break
                        if i == 3: next_version = 3 
                        
                    out_filename = f"{base_name}_v{next_version}.wav"
                    out_path = os.path.join(clips_dir, out_filename)
                    
                    # 3. Load Voice Prompts
                    items, err = load_voice_prompt_items(voice_name)
                    if err: return None, gr.update(), err
                    
                    # 4. Prepare Style Instruction
                    instruct_text = ""
                    if style_name:
                         presets = get_style_presets()
                         instruct = presets.get(style_name, "")
                         if instruct:
                             instruct_text = instruct
                    
                    # 5. Generate
                    try:
                        gen_kwargs = {}
                        if instruct_text:
                            print(f"DEBUG: Generating chapter with style: '{instruct_text}'")
                            i_text = f"<|im_start|>user\n{instruct_text}<|im_end|>\n"
                            instruct_ids = tts_clone._tokenize_texts([i_text])[0]
                            gen_kwargs["instruct_ids"] = [instruct_ids]
                        
                        wavs, sr = tts_clone.generate_voice_clone(
                            text=text.strip(),
                            language=language if language else "Auto",
                            voice_clone_prompt=items,
                            **gen_kwargs
                        )
                        
                        final_wav = wavs[0]
                        # Assume normal speed (project settings for speed not yet requested)
                        
                        import soundfile as sf
                        sf.write(out_path, final_wav, sr)
                        
                        # Return updated takes list and the new audio path
                        takes, _ = get_chapter_takes(project_name, chapter_filename)
                        # We want to select the one we just made
                        new_take_label = f"Take {next_version}"
                        
                        return out_path, gr.update(choices=takes, value=new_take_label), f"Generated {new_take_label}"
                        
                    except Exception as e:
                         print(f"Generation Error: {e}")
                         return None, gr.update(), f"Error: {e}"

                def on_project_select(pname):
                    print(f"DEBUG: on_project_select triggered for '{pname}'")
                    story = load_full_story(pname)
                    chapters = get_chapter_list(pname)
                    _, voice, style, lang, spacing = load_project_settings(pname)
                    
                    # Default: Select first chapter, load its takes
                    return (
                        story, 
                        gr.update(choices=chapters, value=chapters[0] if chapters else None),
                        voice or gr.update(), 
                        style or gr.update(),
                        lang or "Auto",
                        spacing or 1,
                        # We don't verify takes here directly, on_chapter_select will trigger
                        None, 
                        gr.update(choices=[], value=None)
                    )

                def on_chapter_select(pname, cname):
                    content = load_chapter_content(pname, cname)
                    takes, legacy_path = get_chapter_takes(pname, cname)
                    
                    audio_path = None
                    radio_value = None
                    
                    if takes:
                        # User wants first by default
                        radio_value = takes[0] # "Take 1"
                        # Resolving path
                        # If "Take 1", check v1 file, if not, check legacy
                        if radio_value == "Take 1":
                            v1_path = os.path.join(PROJECTS_DIR, pname, "clips", cname.replace(".txt", "_v1.wav"))
                            if os.path.exists(v1_path): audio_path = v1_path
                            elif legacy_path and os.path.exists(legacy_path): audio_path = legacy_path
                    
                    return content, audio_path, gr.update(choices=takes, value=radio_value)

                def on_take_select(pname, cname, take_label):
                    if not pname or not cname or not take_label: return None
                    
                    # Extract version number
                    # "Take 1" -> 1
                    try:
                        v_num = int(take_label.split(" ")[1])
                    except: return None
                    
                    clips_dir = os.path.join(PROJECTS_DIR, pname, "clips")
                    base_name = os.path.splitext(cname)[0]
                    
                    # Standard path
                    v_path = os.path.join(clips_dir, f"{base_name}_v{v_num}.wav")
                    
                    if os.path.exists(v_path):
                        return v_path
                    
                    # Fallback for Take 1 legacy
                    if v_num == 1:
                         legacy = os.path.join(clips_dir, cname.replace(".txt", ".wav"))
                         if os.path.exists(legacy): return legacy
                         
                    return None

                def save_trimmed_clip(project_name, chapter_filename, take_label, audio_file):
                    if not project_name or not chapter_filename or not take_label:
                        return "Missing information."
                    if not audio_file:
                        return "No audio to save."
                    
                    try:
                        v_num = int(take_label.split(" ")[1])
                    except: return "Invalid take label."
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    base_name = os.path.splitext(chapter_filename)[0]
                    
                    target_path = os.path.join(clips_dir, f"{base_name}_v{v_num}.wav")
                    
                    # If target path doesn't exist but it's Take 1, maybe it's the legacy path?
                    if v_num == 1 and not os.path.exists(target_path):
                        legacy = os.path.join(clips_dir, chapter_filename.replace(".txt", ".wav"))
                        if os.path.exists(legacy): target_path = legacy
                    
                    try:
                        # Copy temp audio file to target path
                        import shutil
                        shutil.copy(audio_file, target_path)
                        return f"Trim saved to {take_label}", get_project_playlist(project_name)
                    except Exception as e:
                        return f"Error saving trim: {e}", gr.update()

                def save_chapter_selection(pname, cname, take_label):
                    if not pname or not cname or not take_label: return
                    
                    settings_dir = os.path.join(PROJECTS_DIR, pname)
                    sel_file = os.path.join(settings_dir, "selections.json")
                    
                    data = {}
                    if os.path.exists(sel_file):
                        try:
                            with open(sel_file, 'r') as f: data = json.load(f)
                        except: pass
                    
                    data[cname] = take_label
                    
                    with open(sel_file, 'w') as f:
                        json.dump(data, f)

                def get_project_playlist(pname):
                    # Returns Dataframe value: [[Chapter, Take, Path], ...]
                    if not pname: return []
                    
                    chapters = get_chapter_list(pname)
                    if not chapters: return []
                    
                    settings_dir = os.path.join(PROJECTS_DIR, pname)
                    sel_file = os.path.join(settings_dir, "selections.json")
                    selections = {}
                    if os.path.exists(sel_file):
                        try:
                             with open(sel_file, 'r') as f: selections = json.load(f)
                        except: pass
                        
                    data = []
                    clips_dir = os.path.join(PROJECTS_DIR, pname, "clips")
                    
                    for c_name in chapters:
                        # Which take is selected? Default to Take 1
                        selected_label = selections.get(c_name, "Take 1")
                        
                        try:
                            v_num = int(selected_label.split(" ")[1])
                        except: v_num = 1
                        
                        base_name = os.path.splitext(c_name)[0]
                        v_path = os.path.join(clips_dir, f"{base_name}_v{v_num}.wav")
                        
                        final_path = None
                        if os.path.exists(v_path):
                            final_path = v_path
                        elif v_num == 1:
                            legacy = os.path.join(clips_dir, c_name.replace(".txt", ".wav"))
                            if os.path.exists(legacy): final_path = legacy
                        
                        status = selected_label
                        if not final_path:
                             status = f"{selected_label} (Missing)"
                             final_path = "" 
                        
                        # Clean name
                        disp_name = c_name.replace(".txt", "").replace("_", " ").strip()
                        
                        data.append([disp_name, status, final_path])
                        
                    return data

                def generate_batch_takes_wrapper(project_name, chapter_filename, voice_name, style_name, language):
                    if not project_name or not chapter_filename:
                        return None, gr.update(), "Missing project or chapter.", [], None
                        
                    # Loop 3 times
                    last_path = None
                    last_radio = gr.update()
                    final_msg = ""
                    
                    for i in range(3):
                        res = generate_chapter_clip(project_name, chapter_filename, voice_name, style_name, language)
                        if isinstance(res, tuple):
                            path, radio_upd, status = res
                            last_path = path
                            last_radio = radio_upd
                            final_msg = status
                        else:
                            # If error, break
                            final_msg = str(res)
                            break
                    
                    # Select the last one (Take 3 usually, or whatever last succeeded)
                    if last_radio.get('value'):
                         save_chapter_selection(project_name, chapter_filename, last_radio['value'])
                    
                    playlist = get_project_playlist(project_name)
                    return last_path, last_radio, f"Batch Done. {final_msg}", playlist, last_path

                    playlist = get_project_playlist(project_name)
                    return last_path, last_radio, f"Batch Done. {final_msg}", playlist, last_path

                def generate_all_missing_takes(project_name, voice_name, style_name, language):
                    if not project_name:
                        yield "No project selected.", gr.update()
                        return

                    chapters = get_chapter_list(project_name)
                    if not chapters:
                        yield "No chapters found.", gr.update()
                        return

                    log_history = "Starting Project-Wide Generation...\n"
                    yield log_history, gr.update()
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    os.makedirs(clips_dir, exist_ok=True)
                    
                    total_actions = 0
                    
                    for cname in chapters:
                        base_name = os.path.splitext(cname)[0]
                        # Count existing
                        cnt = 0
                        for i in range(1, 4):
                            v_name = f"{base_name}_v{i}.wav"
                            if os.path.exists(os.path.join(clips_dir, v_name)): cnt += 1
                        
                        # Legacy check
                        if cnt == 0:
                             legacy = os.path.join(clips_dir, cname.replace(".txt", ".wav"))
                             if os.path.exists(legacy): cnt = 1
                        
                        if cnt >= 3:
                             continue
                        
                        needed = 3 - cnt
                        msg = f"Generating {needed} takes for {cname}..."
                        log_history = msg + "\n" + log_history
                        yield log_history, gr.update()
                        
                        for _ in range(needed):
                             res = generate_chapter_clip(project_name, cname, voice_name, style_name, language)
                             if isinstance(res, tuple):
                                  log_history = f"  - Generated new take for {cname}\n" + log_history
                                  yield log_history, gr.update()
                                  total_actions += 1
                             else:
                                  log_history = f"  - Error generating {cname}: {res}\n" + log_history
                                  yield log_history, gr.update()
                                  break
                    
                    if total_actions == 0:
                        log_history = "All chapters have at least 3 takes.\n" + log_history
                    else:
                        log_history = "Generation Complete.\n" + log_history
                    
                    playlist = get_project_playlist(project_name)
                    yield log_history, playlist

                    playlist = get_project_playlist(project_name)
                    return last_path, last_radio, f"Batch Done. {final_msg}", playlist, last_path

                def generate_all_missing_takes(project_name, voice_name, style_name, language):
                    if not project_name:
                        yield "No project selected.", gr.update()
                        return

                    chapters = get_chapter_list(project_name)
                    if not chapters:
                        yield "No chapters found.", gr.update()
                        return

                    log_history = "Starting Project-Wide Generation...\n"
                    yield log_history, gr.update()
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    os.makedirs(clips_dir, exist_ok=True)
                    
                    total_actions = 0
                    
                    for cname in chapters:
                        base_name = os.path.splitext(cname)[0]
                        # Count existing
                        cnt = 0
                        for i in range(1, 4):
                            v_name = f"{base_name}_v{i}.wav"
                            if os.path.exists(os.path.join(clips_dir, v_name)): cnt += 1
                        
                        # Legacy check
                        if cnt == 0:
                             legacy = os.path.join(clips_dir, cname.replace(".txt", ".wav"))
                             if os.path.exists(legacy): cnt = 1
                        
                        if cnt >= 3:
                             continue
                        
                        needed = 3 - cnt
                        msg = f"Generating {needed} takes for {cname}..."
                        log_history = msg + "\n" + log_history
                        yield log_history, gr.update()
                        
                        for _ in range(needed):
                             res = generate_chapter_clip(project_name, cname, voice_name, style_name, language)
                             if isinstance(res, tuple):
                                  log_history = f"  - Generated new take for {cname}\n" + log_history
                                  yield log_history, gr.update()
                                  total_actions += 1
                             else:
                                  log_history = f"  - Error generating {cname}: {res}\n" + log_history
                                  yield log_history, gr.update()
                                  break
                    
                    if total_actions == 0:
                        log_history = "All chapters have at least 3 takes.\n" + log_history
                    else:
                        log_history = "Generation Complete.\n" + log_history
                    
                    playlist = get_project_playlist(project_name)
                    yield log_history, playlist

                def check_reset_status(project_name):
                    if not project_name: return gr.update(), gr.update(visible=False)
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    if not os.path.exists(clips_dir): return gr.update(), gr.update(visible=False)
                    
                    count = 0
                    for f in os.listdir(clips_dir):
                        if f.endswith(".wav"): # Be careful not to delete other things if any
                            count += 1
                            
                    if count == 0:
                        return gr.update(visible=False), gr.update(visible=False)
                    
                    return gr.update(value=f"Confirm Delete {count} Takes", visible=True), gr.update(visible=True)

                def perform_reset_takes(project_name):
                    if not project_name: return gr.update(), gr.update(visible=False), []
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    if os.path.exists(clips_dir):
                         import shutil
                         shutil.rmtree(clips_dir)
                         os.makedirs(clips_dir, exist_ok=True)
                    
                    # Clear selections
                    sel_path = os.path.join(PROJECTS_DIR, project_name, "selections.json")
                    if os.path.exists(sel_path):
                        os.remove(sel_path)
                        
                    playlist = get_project_playlist(project_name)
                    return gr.update(visible=False), gr.update(visible=False), playlist

                def export_final_audio(project_name, spacing):
                    if not project_name: return None, "No project selected."
                    
                    chapters = get_chapter_list(project_name)
                    if not chapters: return None, "No chapters found."
                    
                    # Load selections
                    sel_path = os.path.join(PROJECTS_DIR, project_name, "selections.json")
                    selections = {}
                    if os.path.exists(sel_path):
                        try:
                            with open(sel_path, "r") as f: selections = json.load(f)
                        except: pass
                    
                    final_audio = []
                    target_sr = 24000 
                    
                    silence_dur = max(0, int(spacing)) if spacing else 1
                    
                    clips_dir = os.path.join(PROJECTS_DIR, project_name, "clips")
                    
                    # Validate and Load
                    for cname in chapters:
                        sel = selections.get(cname)
                        if not sel:
                            return None, f"Chapter '{cname}' has no selected take. Please select one."
                        
                        # sel is like "Take 1" or "Take 2"
                        if not sel.startswith("Take "):
                             return None, f"Invalid selection '{sel}' for chapter '{cname}'."
                        
                        try:
                            v_num = int(sel.replace("Take ", ""))
                            base_name = os.path.splitext(cname)[0]
                            fname = f"{base_name}_v{v_num}.wav"
                            fpath = os.path.join(clips_dir, fname)
                            
                            # Fallback for legacy Take 1
                            if v_num == 1 and not os.path.exists(fpath):
                                legacy = os.path.join(clips_dir, cname.replace(".txt", ".wav"))
                                if os.path.exists(legacy): fpath = legacy
                            
                            if not os.path.exists(fpath):
                                return None, f"Audio file missing for '{cname}' ({sel})."
                                
                            data, samplerate = sf.read(fpath)
                            
                            # Normalize Shape (Mono)
                            if len(data.shape) > 1:
                                data = np.mean(data, axis=1)
                                
                            # Resample if needed
                            if samplerate != target_sr:
                                # Use librosa to resample. Librosa expects (channels, samples) or (samples).
                                # sf.read returns (samples, channels) or (samples).
                                # We already flattened to (samples).
                                # librosa.resample args: y, orig_sr, target_sr
                                data = librosa.resample(data, orig_sr=samplerate, target_sr=target_sr)
                            
                            final_audio.append(data)
                            
                        except Exception as e:
                            return None, f"Error processing '{cname}': {e}"
                    
                    if not final_audio:
                        return None, "No audio data collected."
                    
                    # Concatenate with silence
                    silence = np.zeros(int(target_sr * silence_dur))
                    combined = []
                    for i, clip in enumerate(final_audio):
                        combined.append(clip)
                        if i < len(final_audio) - 1:
                            combined.append(silence)
                            
                    final_arr = np.concatenate(combined)
                    
                    out_path = os.path.join(PROJECTS_DIR, project_name, "final_cut.wav")
                    sf.write(out_path, final_arr, target_sr)
                    
                    return out_path, f"Export successful! Saved to {out_path}"

                def generate_music(project_name, prompt, duration, steps, cfg):
                    if not project_name: return None, "No project selected."
                    if not prompt: return None, "Please enter a prompt."
                    
                    try:
                        pipe = manager.load_music()
                        
                        # Generate
                        # Stable Audio Open takes: prompt, seconds_total, steps, cfg_scale
                        print(f"Generating Music: {prompt} ({duration}s)")
                        
                        # Seed?
                        # generator = torch.Generator("cuda").manual_seed(0)
                        
                        output = pipe(
                            prompt=prompt,
                            audio_end_in_s=duration,
                            num_inference_steps=int(steps),
                            guidance_scale=float(cfg)
                        )
                        
                        audio = output.audios[0] # (channels, samples)
                        sr = pipe.vae.config.sampling_rate # Usually 44100
                        
                        # Transpose to (samples, channels) for soundfile if needed, 
                        # but diffusers output is typically (batch, channels, samples).
                        # output.audios[0] is (channels, samples).
                        # sf.write expects (samples, channels).
                        audio = audio.T.float().cpu().numpy()
                        
                        # Save
                        music_dir = os.path.join(PROJECTS_DIR, project_name, "music")
                        os.makedirs(music_dir, exist_ok=True)
                        
                        import time
                        ts = int(time.time())
                        filename = f"music_{ts}.wav"
                        path = os.path.join(music_dir, filename)
                        
                        sf.write(path, audio, sr)
                        
                        return path, f"Music Generated: {filename}"
                        
                    except Exception as e:
                        print(f"Music Gen Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return None, f"Error: {e}"

                def generate_chapter_clip_wrapper(project_name, chapter_filename, voice_name, style_name, language):
                    res = generate_chapter_clip(project_name, chapter_filename, voice_name, style_name, language)
                    if not isinstance(res, tuple):
                        return res, gr.update(), gr.update(), gr.update(), gr.update()
                        
                    path, radio_upd, status = res
                    
                    if radio_upd.get('value'):
                         save_chapter_selection(project_name, chapter_filename, radio_upd['value'])
                    
                    playlist = get_project_playlist(project_name)
                    # path triggers ch_audio AND playlist_audio
                    return path, radio_upd, status, playlist, path

                def on_project_select_wrapper(pname):
                    story, ch_upd, voice_upd, style_upd, lang_upd, spacing_upd, _, radio_upd = on_project_select(pname)
                    playlist = get_project_playlist(pname)
                    return story, ch_upd, voice_upd, style_upd, lang_upd, spacing_upd, None, radio_upd, playlist

                def on_chapter_select_wrapper(pname, cname):
                    content, audio, radio_upd = on_chapter_select(pname, cname)
                    if pname and cname:
                         # Sync logic (optional but good)
                         pass
                    return content, audio, radio_upd

                def on_take_select_wrapper(pname, cname, take_label):
                    path = on_take_select(pname, cname, take_label)
                    save_chapter_selection(pname, cname, take_label)
                    playlist = get_project_playlist(pname)
                    return path, playlist, path
                
                def play_playlist_item(proj_name, evt: gr.SelectData):
                     if not proj_name: return None
                     data = get_project_playlist(proj_name)
                     try:
                         # evt.index is [row, col]
                         row = evt.index[0]
                         path = data[row][2] # Path is in column 2
                         if path and os.path.exists(path): return path
                     except: pass
                     return None

                prj_dropdown.change(
                    on_project_select_wrapper,
                    inputs=[prj_dropdown],
                    outputs=[fs_text, ch_dropdown, prj_voice_dropdown, prj_style_dropdown, prj_lang_dropdown, prj_spacing_slider, ch_audio, ch_take_radio, playlist_df]
                )
                
                prj_save_settings_btn.click(
                    save_project_settings,
                    inputs=[prj_dropdown, prj_voice_dropdown, prj_style_dropdown, prj_lang_dropdown, prj_spacing_slider],
                    outputs=[prj_settings_status]
                )
                
                fs_save_btn.click(
                    save_full_story,
                    inputs=[prj_dropdown, fs_text],
                    outputs=[fs_status, ch_dropdown]
                )
                
                ch_refresh_btn.click(
                    get_chapter_list,
                    inputs=[prj_dropdown],
                    outputs=[ch_dropdown]
                )
                
                ch_dropdown.change(
                    on_chapter_select_wrapper,
                    inputs=[prj_dropdown, ch_dropdown],
                    outputs=[ch_text, ch_audio, ch_take_radio]
                )
                
                ch_gen_btn.click(
                    generate_chapter_clip_wrapper,
                    inputs=[prj_dropdown, ch_dropdown, prj_voice_dropdown, prj_style_dropdown, prj_lang_dropdown],
                    outputs=[ch_audio, ch_take_radio, ch_status, playlist_df, playlist_audio]
                )

                ch_gen_3_btn.click(
                    generate_batch_takes_wrapper,
                    inputs=[prj_dropdown, ch_dropdown, prj_voice_dropdown, prj_style_dropdown, prj_lang_dropdown],
                    outputs=[ch_audio, ch_take_radio, ch_status, playlist_df, playlist_audio]
                )
                
                playlist_gen_btn.click(
                    generate_all_missing_takes,
                    inputs=[prj_dropdown, prj_voice_dropdown, prj_style_dropdown, prj_lang_dropdown],
                    outputs=[playlist_log, playlist_df]
                )
                
                playlist_reset_btn.click(
                    check_reset_status,
                    inputs=[prj_dropdown],
                    outputs=[playlist_reset_confirm_btn, playlist_reset_confirm_row]
                )
                
                playlist_reset_confirm_btn.click(
                    perform_reset_takes,
                    inputs=[prj_dropdown],
                    outputs=[playlist_reset_confirm_btn, playlist_reset_confirm_row, playlist_df]
                )
                
                playlist_reset_cancel_btn.click(
                    lambda: (gr.update(visible=False), gr.update(visible=False)),
                    outputs=[playlist_reset_confirm_btn, playlist_reset_confirm_row]
                )
                
                playlist_export_btn.click(
                    export_final_audio,
                    inputs=[prj_dropdown, prj_spacing_slider],
                    outputs=[playlist_export_audio, playlist_log] # Reusing log for status message
                )
                
                ch_take_radio.change(
                    on_take_select_wrapper,
                    inputs=[prj_dropdown, ch_dropdown, ch_take_radio],
                    outputs=[ch_audio, playlist_df, playlist_audio]
                )
                
                ch_save_trim_btn.click(
                    save_trimmed_clip,
                    inputs=[prj_dropdown, ch_dropdown, ch_take_radio, ch_audio],
                    outputs=[ch_status, playlist_df]
                )
                
                playlist_df.select(
                    play_playlist_item,
                    inputs=[prj_dropdown],
                    outputs=[playlist_audio]
                )

            # --- TAB 4: VOICE ISOLATION ---
            with gr.Tab("Voice Isolation"):
                gr.Markdown("### Remove background noise/music from audio")
                with gr.Row():
                    with gr.Column(scale=2):
                        vi_input = gr.Audio(label="Input Audio", type="filepath")
                        vi_btn = gr.Button("Isolate Voice", variant="primary")
                    
                    with gr.Column(scale=3):
                        vi_output = gr.Audio(label="Isolated Vocals", type="filepath")
                        vi_status = gr.Textbox(label="Status", lines=2)
                
                def run_isolation(audio_path):
                    if not audio_path:
                        return None, "Please upload an audio file."
                    try:
                        # isolate_voice returns path to vocals
                        vocals_path = isolate_voice(audio_path)
                        return vocals_path, "Isolation complete."
                    except Exception as e:
                        return None, str(e)

                vi_btn.click(run_isolation, inputs=[vi_input], outputs=[vi_output, vi_status])

            # --- TAB 5: UTILITIES ---
            with gr.Tab("Utilities"):
                gr.Markdown("### Helper Tools")
                
                with gr.Tabs():
                    # 5a. Extract Audio
                    with gr.Tab("Extract Audio from Video"):
                        gr.Markdown("#### Convert MP4 to WAV")
                        with gr.Row():
                            ext_vid = gr.Video(label="Input Video")
                            ext_btn = gr.Button("Extract Audio", variant="primary")
                        ext_out = gr.Audio(label="Extracted Audio", type="filepath")
                        ext_file = gr.File(label="Download WAV")
                        ext_status = gr.Textbox(label="Status", lines=1)
                        
                        def run_extract(vid_path):
                            if not vid_path: return None, None, "No video uploaded."
                            try:
                                out = extract_audio_from_video(vid_path)
                                return out, out, "Extraction complete."
                            except Exception as e:
                                return None, None, str(e)
                                
                        ext_btn.click(run_extract, inputs=[ext_vid], outputs=[ext_out, ext_file, ext_status])

                    # 5b. Transcribe
                    with gr.Tab("Transcribe Audio"):
                        gr.Markdown("#### Speech-to-Text (Whisper)")
                        with gr.Row():
                            tr_aud = gr.Audio(label="Input Audio", type="filepath")
                            with gr.Column():
                                tr_model = gr.Dropdown(
                                    label="Model Size", 
                                    choices=["tiny", "base", "small", "medium", "large"], 
                                    value="base"
                                )
                                tr_lang = gr.Textbox(label="Language Code (Optional)", placeholder="e.g. en, es (Leave empty for auto)")
                                tr_btn = gr.Button("Transcribe", variant="primary")
                        
                        tr_out = gr.Textbox(label="Transcription", lines=6, show_copy_button=True)
                        tr_status = gr.Textbox(label="Status", lines=1)
                        
                        def run_transcribe(aud_path, model_sz, lang_code):
                            if not aud_path: return None, "No audio uploaded."
                            try:
                                lang = lang_code.strip() if lang_code.strip() else None
                                txt = transcribe_audio(aud_path, model_size=model_sz, language=lang)
                                return txt, "Transcription complete."
                            except Exception as e:
                                return None, str(e)
                                
                        tr_btn.click(run_transcribe, inputs=[tr_aud, tr_model, tr_lang], outputs=[tr_out, tr_status])

                    # 5c. Clip Audio
                    with gr.Tab("Clip Audio"):
                        gr.Markdown("#### Slice Audio Segment (Slider precise control)")
                        
                        with gr.Row():
                            cl_aud = gr.Audio(label="Input Audio", type="filepath")
                            with gr.Column():
                                # Initial max is arbitrary, will be updated on upload
                                cl_start = gr.Slider(label="Start Time (s)", minimum=0, maximum=300, value=0, step=0.1)
                                cl_end = gr.Slider(label="End Time (s)", minimum=0, maximum=300, value=10, step=0.1)
                                cl_btn = gr.Button("Clip Audio", variant="primary")
                        
                        cl_out = gr.Audio(label="Clipped Audio", type="filepath")
                        cl_file = gr.File(label="Download Clip")
                        cl_status = gr.Textbox(label="Status", lines=1)
                        
                        def update_slider_range(audio_path):
                            if not audio_path:
                                return gr.update(), gr.update()
                            try:
                                dur = librosa.get_duration(path=audio_path)
                                # Set max to duration, default end to duration
                                return gr.update(maximum=dur, value=0), gr.update(maximum=dur, value=dur)
                            except Exception as e:
                                print(f"Error getting duration: {e}")
                                return gr.update(), gr.update()

                        # Update sliders when audio is uploaded/changed
                        cl_aud.change(update_slider_range, inputs=[cl_aud], outputs=[cl_start, cl_end])
                        
                        def run_clip(aud_path, start, end):
                            if not aud_path: return None, None, "No audio uploaded."
                            try:
                                out = clip_audio(aud_path, float(start), float(end))
                                return out, out, "Clipping complete."
                            except Exception as e:
                                return None, None, str(e)
                                
                        cl_btn.click(run_clip, inputs=[cl_aud, cl_start, cl_end], outputs=[cl_out, cl_file, cl_status])

            with gr.Tab("Music & FX"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Generate Background Music / FX")
                        music_prompt = gr.TextArea(label="Prompt", placeholder="A calm ambient piano melody with soft rain sounds...", lines=3)
                        with gr.Row():
                             music_dur = gr.Slider(minimum=1, maximum=47, step=1, value=10, label="Duration (s)")
                             music_steps = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="Steps")
                             music_cfg = gr.Slider(minimum=1, maximum=15, step=0.5, value=7, label="CFG Scale")
                        music_gen_btn = gr.Button("Generate Music", variant="primary")
                        music_status = gr.Textbox(label="Status")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Audio")
                        music_audio_out = gr.Audio(label="Result", interactive=False)
                
                music_gen_btn.click(
                    generate_music,
                    inputs=[prj_dropdown, music_prompt, music_dur, music_steps, music_cfg],
                    outputs=[music_audio_out, music_status]
                )

    return demo

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Device auto-detection
    device = args.device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Auto-detected MPS (Mac GPU).")
    elif not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
        print("CUDA not available, falling back to CPU.")
    
    print(f"Using device: {device}")
    
    # Only use flash_attention_2 if explicitly safe (CUDA) and not disabled
    if not args.no_flash_attn and device.startswith("cuda"):
        attn_impl = "flash_attention_2"
    else:
        attn_impl = None

    print("\nLoading Voice Design Model (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)...")
    tts_design = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )

    print("\nLoading Base Model (Cloning) (Qwen/Qwen3-TTS-12Hz-1.7B-Base)...")
    tts_clone = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )

    print("\nBuilding Unified Demo...")
    demo = build_full_demo(tts_clone, tts_design)
    
    print(f"\nLaunching on {args.ip}:{args.port}...")
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()
