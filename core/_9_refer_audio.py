import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pandas as pd
import soundfile as sf
from core.asr_backend.demucs_vl import demucs_audio
from core.utils.models_batch import *
import threading

# 为每个线程创建一个 Console 实例
thread_local = threading.local()
def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    sf.write(out_file, audio_data[start:end], sr)

def extract_refer_audio_main(video_name):
    # 初始化所有路径（会生成新的UUID）
    pathManager = PathManager(video_name)
    VOCAL_AUDIO_FILE = _VOCAL_AUDIO_FILE(pathManager)
    BACKGROUND_AUDIO_FILE = _BACKGROUND_AUDIO_FILE(pathManager)
    AUDIO_DIR = _AUDIO_DIR(pathManager)
    RAW_AUDIO_FILE = _RAW_AUDIO_FILE(pathManager)
    AUDIO_SEGS_DIR = _AUDIO_SEGS_DIR(pathManager)
    AUDIO_REFERS_DIR = _AUDIO_REFERS_DIR(pathManager)
    AUDIO_TASK = _8_1_AUDIO_TASK(pathManager)
    console = get_console()

    demucs_audio(VOCAL_AUDIO_FILE, BACKGROUND_AUDIO_FILE, AUDIO_DIR, RAW_AUDIO_FILE, console)
    
    if os.path.exists(os.path.join(AUDIO_SEGS_DIR, '1.wav')):
        console.print(Panel("Audio segments already exist, skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(AUDIO_TASK)
    data, sr = sf.read(VOCAL_AUDIO_FILE)
    
    # 使用线程安全的 console 创建进度条
    with console.status("[bold green]正在提取音频片段...") as status:
        total = len(df)
        for index, row in df.iterrows():
            out_file = os.path.join(AUDIO_REFERS_DIR, f"{row['number']}.wav")
            extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            console.print(f"[cyan]处理进度: {index+1}/{total}[/cyan]")
            
    console.print(Panel(f"音频片段已保存到 {AUDIO_REFERS_DIR}", title="成功", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main("808c5a77e3419eeb9017f6029e97a10a")