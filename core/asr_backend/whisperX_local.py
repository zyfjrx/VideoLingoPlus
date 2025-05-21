import os
import warnings
import time
import subprocess
import torch
import whisperx
import librosa
from core.utils import *
import threading
warnings.filterwarnings("ignore")
MODEL_DIR = load_key("model_dir")

# 添加模型管理器类
class WhisperModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _align_model = None
    _metadata = None
    _device = None
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WhisperModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, model_name, device, compute_type, language, vad_options, asr_options):
        with self._lock:
            if self._model is None:
                self._device = device
                self._model = whisperx.load_model(
                    model_name, 
                    device, 
                    compute_type=compute_type,
                    language=language,
                    vad_options=vad_options,
                    asr_options=asr_options,
                    download_root=MODEL_DIR
                )
    
    def get_model(self):
        return self._model, self._device
    
    def initialize_align_model(self, language_code):
        with self._lock:
            if self._align_model is None:
                self._align_model, self._metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self._device
                )
        return self._align_model, self._metadata
    
    def cleanup(self):
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._align_model is not None:
                del self._align_model
                self._align_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror(console):
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    console.print("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            console.print(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        console.print("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    console.print(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end,console):
    os.environ['HF_ENDPOINT'] = check_hf_mirror(console)
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        console.print(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        batch_size = 1
        compute_type = "int8"
        console.print(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    
    # 使用模型管理器
    model_manager = WhisperModelManager()
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        console.print(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        console.print(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}
    asr_options = {"temperatures": [0], "initial_prompt": ""}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    
    # 初始化并获取模型
    model_manager.initialize(model_name, device, compute_type, whisper_language, vad_options, asr_options)
    model, device = model_manager.get_model()

    def load_audio_segment(audio_file, start, end):
        audio, _ = librosa.load(audio_file, sr=16000, offset=start, duration=end - start, mono=True)
        return audio
    
    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
    
    # 转录音频
    transcribe_start_time = time.time()
    console.print("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    transcribe_time = time.time() - transcribe_start_time
    console.print(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # 保存语言设置
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        # model_manager.cleanup()
        raise ValueError("Please specify the transcription language as zh and try again!")

    # 对齐音频
    align_start_time = time.time()
    model_a, metadata = model_manager.initialize_align_model(result["language"])
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    align_time = time.time() - align_start_time
    console.print(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # 调整时间戳
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment['words']:
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start
    return result