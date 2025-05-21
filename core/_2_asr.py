from pydub.utils import mediainfo
from rich.console import Console
from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core.utils.models_batch import *
import os
from core.utils.models_batch import PathManager
from pathlib import Path
import threading
import pandas as pd
# 为每个线程创建一个 Console 实例
thread_local = threading.local()

def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console

# @check_file_exists(_2_CLEANED_CHUNKS)
def transcribe(video_file):
    # 1. video to audio
    # video_file = find_video_files()
    # 为每个视频创建独立的输出目录
    video_name = Path(video_file).stem
    # 初始化所有路径（会生成新的UUID）
    pathManager = PathManager(video_name)
    OUTPUT_DIR = _OUTPUT_DIR(pathManager)
    # output_dir = os.path.join("output", video_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    VOCAL_AUDIO_FILE = _VOCAL_AUDIO_FILE(pathManager)
    BACKGROUND_AUDIO_FILE = _BACKGROUND_AUDIO_FILE(pathManager)
    AUDIO_DIR = _AUDIO_DIR(pathManager)
    RAW_AUDIO_FILE = _RAW_AUDIO_FILE(pathManager)
    LOG = _LOG(pathManager)
    CLEANED_CHUNKS = _2_CLEANED_CHUNKS(pathManager)
    console = get_console()
    convert_video_to_audio(video_file,AUDIO_DIR,RAW_AUDIO_FILE)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio(VOCAL_AUDIO_FILE, BACKGROUND_AUDIO_FILE, AUDIO_DIR, RAW_AUDIO_FILE,console)
        vocal_audio = normalize_audio_volume(VOCAL_AUDIO_FILE, VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        console.print("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        console.print("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        console.print("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
    elif runtime == "funasr":
        from core.asr_backend.funASR_local import funasr_transcribe_audio as ts

    # for start, end in segments:
    for start, end in segments:
        result = ts(RAW_AUDIO_FILE, vocal_audio, start, end,console)
        all_results.append(result)
    
    # 5. Combine results

    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    if runtime == 'funasr':
        all_words = []
        for segment in result['segments']:
            all_words.extend(segment['words'])
        df = pd.DataFrame(all_words)
        os.makedirs(LOG, exist_ok=True)
        df.to_excel(CLEANED_CHUNKS, index=False)
    else:
        df = process_transcription(combined_result)
        save_results(df,LOG, CLEANED_CHUNKS)
        
if __name__ == "__main__":
    transcribe("/home/bmh/VideoLingoPlus/core/input/cdc2912a1a812adfbcd67adf39c4494f.mp4")