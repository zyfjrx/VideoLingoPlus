import http.client
import json
import os
import requests
from pydub import AudioSegment
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *
from gradio_client import Client, handle_file
import shutil

GRADIO_CLIENT = Client("http://14.103.162.45:7860/")
# AUDIO_REFERS_DIR = "output/audio/refers"
NORMALIZED_REFERS_CACHE = {}

def _f5_tts(text: str, refer_path: str, save_path: str,number: int,task_df) -> bool:
    try:
        prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
        # 规范化参考音频（如果还没有处理过）
        if refer_path not in NORMALIZED_REFERS_CACHE:
            normalized_path = refer_path.replace('.wav', '_normalized.wav')
            normalized_refer = normalize_audio_volume(refer_path, normalized_path)
            NORMALIZED_REFERS_CACHE[refer_path] = normalized_refer
        else:
            normalized_refer = NORMALIZED_REFERS_CACHE[refer_path]
        
        result = GRADIO_CLIENT.predict(
            ref_audio_input=handle_file(normalized_refer),
            ref_text_input=prompt_text,  # 使用相同的文本作为参考
            gen_text_input=text,
            remove_silence=False,
            cross_fade_duration_slider=0.15,
            nfe_slider=32,
            speed_slider=1,
            api_name="/basic_tts"
        )
        
        # 结果是一个元组，第一个元素是生成的音频文件路径
        generated_audio_path = result[0]
        
        # 复制生成的音频文件到目标路径
        shutil.copy2(generated_audio_path, save_path)
        print(f"Audio file saved to {save_path}")
        return True
    
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return False


def _merge_audio(files, output: str) -> bool:
    """Merge audio files, add a brief silence"""
    try:
        # Create an empty audio segment
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=100)  # 100ms silence
        
        # Add audio files one by one
        for file in files:
            audio = AudioSegment.from_wav(file)
            combined += audio + silence
        combined += silence
        combined.export(output, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
        
        if os.path.getsize(output) == 0:
            rprint(f"[red]Output file size is 0")
            return False
            
        rprint(f"[green]Successfully merged audio files")
        return True
        
    except Exception as e:
        rprint(f"[red]Failed to merge audio: {str(e)}")
        return False
    
def _get_ref_audio(task_df, min_duration=8, max_duration=14.5) -> str:
    """Get reference audio, ensuring the combined audio duration is > min_duration and < max_duration"""
    rprint(f"[blue]🎯 Starting reference audio selection process...")
    
    duration = 0
    selected = []
    
    for _, row in task_df.iterrows():
        current_duration = row['duration']
        
        # Skip if adding this segment would exceed max duration
        if current_duration + duration > max_duration:
            continue
            
        # Add segments until we exceed min duration
        selected.append(row)
        duration += current_duration
        
        # Once we exceed min duration and are under max, we're done
        if duration > min_duration and duration < max_duration:
            break
    
    if not selected:
        rprint(f"[red]❌ No valid segments found (could not reach minimum {min_duration}s duration)")
        return None
        
    rprint(f"[blue]📊 Selected {len(selected)} segments, total duration: {duration:.2f}s")
    
    audio_files = [f"{_AUDIO_REFERS_DIR}/{row['number']}.wav" for row in selected]
    rprint(f"[yellow]🎵 Audio files to merge: {audio_files}")
    
    combined_audio = f"{_AUDIO_REFERS_DIR}/refer.wav"
    success = _merge_audio(audio_files, combined_audio)
    
    if not success:
        rprint(f"[red]❌ Error: Failed to merge audio files")
        return False
    
    rprint(f"[green]✅ Successfully created combined audio: {combined_audio}")
    
    return combined_audio

def f5_tts_for_videolingo(text: str, save_as: str, number: int,task_df,AUDIO_REFERS_DIR):
    # 使用对应编号的参考音频
    refer_path = os.path.join(AUDIO_REFERS_DIR, f"{number}.wav")
    
    if not os.path.exists(refer_path):
        rprint(f"[red]❌ Reference audio not found: {refer_path}[/red]")
        return False
        
    try:
        success = _f5_tts(text=text, refer_path=refer_path, save_path=save_as,number=number,task_df=task_df)
        return success
    except Exception as e:
        print(f"Error in f5_tts_for_videolingo: {str(e)}")
        return False

if __name__ == "__main__":
    test_refer_url = "/opt/module/VideoLingo/output/audio/refers/1_normalized.wav"
    test_text = "Hello, world!"
    test_save_as = "test_f5_tts.wav"
    success = _f5_tts(text=test_text, refer_url=test_refer_url, save_path=test_save_as)
    print(f"Test result: {success}")