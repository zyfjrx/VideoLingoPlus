import os
import time
import shutil
import subprocess
from typing import Tuple
import pandas as pd
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main
from core.utils.models_batch import *
import threading

# 为每个线程创建一个 Console 实例
thread_local = threading.local()
def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console

WARMUP_SIZE = 5

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float,console) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return
        
    atempo = speed_factor
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                console.print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                console.print(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                console.print(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def process_row(row: pd.Series, tasks_df: pd.DataFrame,AUDIO_REFERS_DIR,TEMP_FILE_TEMPLATE) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(line, temp_file, number, tasks_df,AUDIO_REFERS_DIR)
        real_dur += get_audio_duration(temp_file)
    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame,AUDIO_REFERS_DIR,TEMP_FILE_TEMPLATE,console) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    console.print("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    total = len(tasks_df)
    completed = 0
    
    with console.status("[bold green]正在生成 TTS 音频...") as status:
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur = process_row(row, tasks_df,AUDIO_REFERS_DIR,TEMP_FILE_TEMPLATE)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                completed += 1
                console.print(f"[cyan]进度: {completed}/{total} ({(completed/total*100):.1f}%)[/cyan]")
            except Exception as e:
                console.print(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e
        
        # for gpt_sovits, do not use parallel to avoid mistakes
        max_workers = load_key("max_workers") if load_key("tts_method") != "gpt_sovits" else 1
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy(),AUDIO_REFERS_DIR,TEMP_FILE_TEMPLATE)
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        completed += 1
                        console.print(f"[cyan]进度: {completed}/{total} ({(completed/total*100):.1f}%)[/cyan]")
                    except Exception as e:
                        console.print(f"[red]❌ Error: {str(e)}[/red]")
                        raise e

    console.print("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (durations-speed_var_error))
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / (durations-speed_var_error))
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (tol_durs-speed_var_error))
    else:
        speed_factor = chunk_durs / (tol_durs-speed_var_error)
        keep_gaps = False
        
    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame,TEMP_FILE_TEMPLATE,OUTPUT_FILE_TEMPLATE,console) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    console.print("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] # 加上tolerance才是这一块的结束
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor,console)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([cur_time, cur_time+ad_dur])
                    cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                console.print(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= 0.6:  # If exceeding time is within 0.6 seconds, truncate the last audio
                    console.print(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
                    last_line_index = len(last_lines) - 1
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")
                    
                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                    trimmed_audio.export(last_file, format="wav")
                    
                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    last_times[-1][1] = chunk_end_time
                    tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1
    
    console.print("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio(video_name) -> None:
    pathManager = PathManager(video_name)
    AUDIO_TASK = _8_1_AUDIO_TASK(pathManager)
    AUDIO_SEGS_DIR = _AUDIO_SEGS_DIR(pathManager)
    AUDIO_TMP_DIR = _AUDIO_TMP_DIR(pathManager)
    AUDIO_REFERS_DIR = _AUDIO_REFERS_DIR(pathManager)
    TEMP_FILE_TEMPLATE = f"{AUDIO_TMP_DIR}/{{}}_temp.wav"
    OUTPUT_FILE_TEMPLATE = f"{AUDIO_SEGS_DIR}/{{}}.wav"
    console = get_console()
    """Main function: Generate audio and process timeline"""
    console.print("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    
    # 🎯 Step1: Create necessary directories
    os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(AUDIO_TASK)
    console.print("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df,AUDIO_REFERS_DIR,TEMP_FILE_TEMPLATE,console)
    
    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df,TEMP_FILE_TEMPLATE,OUTPUT_FILE_TEMPLATE,console)
    
    # 💾 Step5: Save results
    tasks_df.to_excel(AUDIO_TASK, index=False)
    console.print("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio("808c5a77e3419eeb9017f6029e97a10a")
