import platform
import subprocess
import threading
from pathlib import Path
import cv2
import numpy as np
from rich.console import Console
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models_batch import *
# 为每个线程创建一个 Console 实例
thread_local = threading.local()

def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console


TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio(video_file):
    """Merge video and audio, and reduce video volume"""
    # VIDEO_FILE = find_video_files()
    video_name = Path(video_file).stem
    pathManager = PathManager(video_name)
    OUTPUT_DIR = _OUTPUT_DIR(pathManager)
    DUB_VIDEO = f"{OUTPUT_DIR}/t_{video_name}.mp4"
    DUB_SUB_FILE = f"{OUTPUT_DIR}/dub.srt"
    DUB_AUDIO = f"{OUTPUT_DIR}/dub.mp3"
    BACKGROUND_AUDIO_FILE = _BACKGROUND_AUDIO_FILE(pathManager)
    console = get_console()
    if not load_key("burn_subtitles"):
        console.print("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        console.print("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = f"{OUTPUT_DIR}/normalized_dub.wav"
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    console.print(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
        f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
    )
    
    cmd = [
        'ffmpeg', '-y', '-i', video_file, '-i', BACKGROUND_AUDIO_FILE, '-i', normalized_dub_audio,
        '-filter_complex',
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        f'{subtitle_filter}[v];'
        f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])
    
    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])
    
    subprocess.run(cmd)
    console.print(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio("/home/bmh/VideoLingo/core/input/808c5a77e3419eeb9017f6029e97a10a.mp4")
