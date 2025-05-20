import pandas as pd
import json
import concurrent.futures
import threading
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models_batch import *

# 为每个线程创建一个 Console 实例
thread_local = threading.local()
def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console
import sys
# 获取项目根目录的绝对路径
# project_root = "/home/bmh/VideoLingo"
# sys.path.append(project_root)
# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i,SPLIT_BY_MEANING):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i, TERMINOLOGY, GPT_LOG_FOLDER):
    console = get_console()  # 获取线程本地的 console
    things_to_note_prompt = search_things_to_note_in_prompt(chunk, TERMINOLOGY)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i, GPT_LOG_FOLDER,console)
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🚀 Main function to translate all chunks
# @check_file_exists(_4_2_TRANSLATION)
def translate_all(video_name):
    console = get_console()  # 获取线程本地的 console
    pathManager = PathManager(video_name)
    TERMINOLOGY = _4_1_TERMINOLOGY(pathManager)
    TRANSLATION = _4_2_TRANSLATION(pathManager)
    SPLIT_BY_MEANING = _3_2_SPLIT_BY_MEANING(pathManager)
    CLEANED_CHUNKS = _2_CLEANED_CHUNKS(pathManager)
    GPT_LOG_FOLDER = _GPT_LOG_FOLDER(pathManager)
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(chunk_size=600, max_i=10, SPLIT_BY_MEANING=SPLIT_BY_MEANING)
    with open(TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 使用线程本地的 Progress
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i,TERMINOLOGY,GPT_LOG_FOLDER)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

    results.sort(key=lambda x: x[0])  # Sort results based on original order
    
    # 💾 Save results to lists and Excel file
    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)
        
        # Calculate similarity between current chunk and translation results
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) 
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])
        
        # Check similarity and handle exceptions
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")
            
        trans_text.extend(best_match[0][2].split('\n'))
    
    # Trim long translation text
    df_text = pd.read_excel(CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    # 使用线程本地的 console 进行输出
    console.print(df_time)
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration'],console) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all("28679343296-1-16")