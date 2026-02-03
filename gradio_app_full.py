import argparse
import os
import sys
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import librosa

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
                        sl_def_instruct = gr.Textbox(
                            label="Default Style Instruction",
                            placeholder="e.g. Speak slowly and calmly",
                            info="Saved with the voice and auto-loaded."
                        )
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
                        sl_instruct = gr.Textbox(label="Style Instruction", lines=2, placeholder="e.g. Say it sadly (Optional)")
                        with gr.Row():
                            sl_lang = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True, scale=2)
                            sl_speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1, scale=3)
                        
                        sl_gen_btn = gr.Button("Generate from Selection", variant="primary")
                        
                    with gr.Column(scale=3):
                        sl_out = gr.Audio(label="Output Audio", type="numpy")
                        sl_gen_status = gr.Textbox(label="Generate Status", lines=2)

                def save_named_voice(ref_aud, ref_txt, name, def_inst, xvec_mode):
                    if not name or not name.strip(): return None, "Voice name is required."
                    at = _audio_to_tuple(ref_aud)
                    if not at: return None, "Reference audio is required."
                    if not xvec_mode and not ref_txt.strip():
                        return None, "Reference text is required."
                    
                    try:
                        # Create prompt item
                        items = tts_clone.create_voice_clone_prompt(
                            ref_audio=at,
                            ref_text=ref_txt.strip() if ref_txt else None,
                            x_vector_only_mode=xvec_mode
                        )
                        
                        # Ensure filename is safe
                        filename = "".join(c for c in name.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
                        if not filename: filename = "unnamed_voice"
                        out_path = os.path.join("saved_voices", f"{filename}.pt")
                        
                        # Serialize
                        payload = {
                            "items": [asdict(it) for it in items],
                            "default_instruction": def_inst.strip() if def_inst else ""
                        }
                        torch.save(payload, out_path)
                        
                        # Return updated dropdown
                        new_list = get_saved_voices()
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
                        path = os.path.join("saved_voices", voice_filename)
                        if not os.path.exists(path): return None, "Voice file not found."
                        
                        payload = torch.load(path, map_location="cpu")
                        if not isinstance(payload, dict) or "items" not in payload:
                            return None, "Invalid file format."

                        items = []
                        for d in payload["items"]:
                            ref_code = d.get("ref_code")
                            if ref_code is not None and not torch.is_tensor(ref_code): ref_code = torch.tensor(ref_code)
                            ref_spk = d.get("ref_spk_embedding")
                            if ref_spk is None: return None, "Missing embedding."
                            if not torch.is_tensor(ref_spk): ref_spk = torch.tensor(ref_spk)
                            
                            items.append(VoiceClonePromptItem(
                                ref_code=ref_code,
                                ref_spk_embedding=ref_spk,
                                x_vector_only_mode=d.get("x_vector_only_mode", False),
                                icl_mode=d.get("icl_mode", not d.get("x_vector_only_mode", False)),
                                ref_text=d.get("ref_text")
                            ))
                        
                        # Use provided instruction, OR fallback to saved default if available and provided is empty
                        final_instruct = instruct
                        if not final_instruct or not final_instruct.strip():
                            final_instruct = payload.get("default_instruction", "")

                        # Handle Instruction manually for Base model
                        kwargs = dict(gen_kwargs)
                        if final_instruct and final_instruct.strip():
                            # Tokenize instruction exactly like the model does internally for other modes
                            instruct_text = f"<|im_start|>user\n{final_instruct.strip()}<|im_end|>\n"
                            instruct_ids = tts_clone._tokenize_texts([instruct_text])[0]
                            # Pass as list of tensors (batch size 1)
                            kwargs["instruct_ids"] = [instruct_ids]

                        lang = lang_map.get(lang_disp, "Auto")
                        wavs, sr = tts_clone.generate_voice_clone(
                            text=text.strip(),
                            language=lang,
                            voice_clone_prompt=items,
                            **kwargs
                        )
                        
                        # Post-processing: Speed Control
                        final_wav = wavs[0]
                        if speed != 1.0:
                            # Time stretch via librosa (keeps pitch constant)
                            # time_stretch uses Phase Vocoder which requires stft.
                            # Ensure wav is numpy
                            try:
                                final_wav = librosa.effects.time_stretch(final_wav, rate=speed)
                            except Exception as e:
                                print(f"Speed adjustment failed: {e}")
                                # fallback to original

                        return _wav_to_gradio_audio(final_wav, sr), "Finished."

                    except Exception as e:
                        return None, str(e)

                def on_voice_select(voice_filename):
                    if not voice_filename: return gr.update(value="")
                    try:
                        path = os.path.join("saved_voices", voice_filename)
                        if not os.path.exists(path): return gr.update(value="")
                        payload = torch.load(path, map_location="cpu")
                        if isinstance(payload, dict):
                            return gr.update(value=payload.get("default_instruction", ""))
                    except:
                        pass
                    return gr.update(value="")

                def refresh_list():
                    new_list = get_saved_voices()
                    return gr.update(choices=new_list)

                sl_save_btn.click(
                    save_named_voice, 
                    inputs=[sl_ref_audio, sl_ref_text, sl_name_in, sl_def_instruct, sl_xvec], 
                    outputs=[sl_save_status, sl_voice_dropdown]
                )
                
                # Auto-fill instruction when voice is selected
                sl_voice_dropdown.change(
                    on_voice_select,
                    inputs=[sl_voice_dropdown],
                    outputs=[sl_instruct]
                )

                sl_refresh_btn.click(refresh_list, outputs=[sl_voice_dropdown])
                
                sl_gen_btn.click(
                    load_named_voice, 
                    inputs=[sl_voice_dropdown, sl_text, sl_instruct, sl_lang, sl_speed], 
                    outputs=[sl_out, sl_gen_status]
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
