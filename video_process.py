import concurrent.futures
import os
import sys
from pathlib import Path
from core import (
    _2_asr,
    _3_1_split_nlp,_3_2_split_meaning,
    _4_1_summarize,_4_2_translate,
    _5_split_sub,_6_gen_sub,_7_sub_into_vid,
    _8_1_audio_task,_8_2_dub_chunks,
    _9_refer_audio,_10_gen_audio,
    _11_merge_audio,_12_dub_to_vid)
from rich.console import Console
from rich.panel import Panel
import threading
# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 创建线程本地存储
thread_local = threading.local()
def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console


class VideoProcessor:
    def __init__(self, max_workers=3):
        self.max_workers = max_workers

    def process_single_video(self, video_file):
        """处理单个视频文件"""
        try:
            console = get_console()
            video_name = Path(video_file).stem
            console.print(f"[cyan]开始处理视频: {video_name}[/cyan]")

            # 1. 语音识别
            console.print(Panel("[bold green]🎤 执行语音识别...[/bold green]"))
            _2_asr.transcribe(video_file)

            # 2. NLP分句
            console.print(Panel("[bold green]✂️ 执行NLP分句...[/bold green]"))
            _3_1_split_nlp.split_by_spacy(video_name)

            # 3. LLM分句
            console.print(Panel("[bold green]📝 LLM分句...[/bold green]"))
            _3_2_split_meaning.split_sentences_by_meaning(video_name)

            # 4. 总结
            console.print(Panel(f"[bold green]📝 总结...[/bold green]"))
            _4_1_summarize.get_summary(video_name)

            # 5. 翻译
            console.print(Panel("[bold green]📝 总结...[/bold green]"))
            _4_2_translate.translate_all(video_name)

            # 6. 切割和对齐长字幕
            console.print(Panel("[bold green]📝 切割和对齐长字幕...[/bold green]"))
            _5_split_sub.split_for_sub_main(video_name)

            # 7. 生成字幕
            console.print(Panel("[bold green]📝 切割和对齐长字幕...[/bold green]"))
            _6_gen_sub.align_timestamp_main(video_name)

            # 8. 将字幕合并到视频中
            console.print(Panel("[bold green]📝 将字幕合并到视频中...[/bold green]"))
            _7_sub_into_vid.merge_subtitles_to_video(video_file)

            # 9. 生成视频任务和分块
            console.print(Panel("[bold green]📝 生成视频任务和分块...[/bold green]"))
            _8_1_audio_task.gen_audio_task_main(video_name)
            _8_2_dub_chunks.gen_dub_chunks(video_name)

            # 10. 提取参考音频
            console.print(Panel(f"[bold green]📝 提取参考音频...{threading.local()}[/bold green]"))
            _9_refer_audio.extract_refer_audio_main(video_name)

            # 11. 生成和合并音频文件
            console.print(Panel("[bold green]📝 生成和合并音频文件...[/bold green]"))
            _10_gen_audio.gen_audio(video_name)
            _11_merge_audio.merge_full_audio(video_name)
            # 12. 将最终音频合并到视频中
            console.print(Panel("[bold green]📝 将最终音频合并到视频中...[/bold green]"))
            _12_dub_to_vid.merge_video_audio(video_file)

            console.print(f"[green]✅ 成功处理视频: {video_name}[/green]")
            return True

        except Exception as e:
            console = get_console()
            console.print(f"[red]❌ 处理视频 {video_file} 时出错: {str(e)}[/red]")
            return False

    def process_videos(self, video_files):
        """并行处理多个视频文件"""
        console = get_console()
        if not video_files:
            console.print("[yellow]⚠️ 没有找到需要处理的视频文件[/yellow]")
            return

        # 使用线程池处理多个视频
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_video, video_file): video_file
                       for video_file in video_files}

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(futures):
                video_file = futures[future]
                try:
                    success = future.result()
                    if success:
                        console.print(f"[green]✓ 完成处理 {video_file}[/green]")
                    else:
                        console.print(f"[red]× 处理失败 {video_file}[/red]")
                except Exception as e:
                    console.print(f"[red]× 处理出错 {video_file}: {str(e)}[/red]")


def main():
    # 示例使用
    processor = VideoProcessor(max_workers=3)  # 设置最大并行处理数

    # 这里替换为实际的视频文件列表
    video_files = [
        "/home/bmh/VideoLingo/core/input/808c5a77e3419eeb9017f6029e97a10a.mp4",
        "/home/bmh/VideoLingo/core/input/28679343296-1-16.mp4"
    ]

    processor.process_videos(video_files)


if __name__ == "__main__":
    main()