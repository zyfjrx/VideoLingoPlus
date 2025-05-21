import json
import time
from funasr import AutoModel
import os
from loguru import logger
import torch
from dotenv import load_dotenv
import librosa
import threading
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

load_dotenv()

class FunASRModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _device = None
    _model_locks = {}

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(FunASRModelManager, cls).__new__(cls)
                    cls._instance._model_locks = {}
        return cls._instance

    def initialize(self, device='auto'):
        with self._lock:
            if self._model is not None:
                return

            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._device = device

            thread_id = threading.get_ident()
            if thread_id not in self._model_locks:
                self._model_locks[thread_id] = threading.Lock()

            logger.info(f'Loading FunASR model for thread {thread_id}')
            t_start = time.time()

            model_path = "_model_cache/ASR/FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            vad_model_path = "_model_cache/ASR/FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            punc_model_path = "_model_cache/ASR/FunASR/punc_ct-transformer_cn-en-common-vocab471067-large"
            spk_model_path = "_model_cache/ASR/FunASR/speech_campplus_sv_zh-cn_16k-common"

            self._model = AutoModel(
                model=model_path if os.path.isdir(model_path) else "paraformer-zh",
                vad_model=vad_model_path if os.path.isdir(vad_model_path) else "fsmn-vad",
                punc_model=punc_model_path if os.path.isdir(punc_model_path) else "ct-punc",
                spk_model=spk_model_path if os.path.isdir(spk_model_path) else "cam++",
            )
            t_end = time.time()
            logger.info(f'Loaded FunASR model in {t_end - t_start:.2f}s for thread {thread_id}')

    def get_model(self):
        thread_id = threading.get_ident()
        if thread_id not in self._model_locks:
            self._model_locks[thread_id] = threading.Lock()
        with self._model_locks[thread_id]:
            return self._model

    def cleanup(self):
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._model_locks:
                with self._model_locks[thread_id]:
                    if self._model is not None:
                        del self._model
                        self._model = None
                del self._model_locks[thread_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def init_funasr():
    model_manager = FunASRModelManager()
    model_manager.initialize()

def funasr_transcribe_audio(raw_audio_file, vocal_audio_file, start, end, console):
    model_manager = FunASRModelManager()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"🚀 Starting FunASR using device: {device} ...")

        if not os.path.exists(vocal_audio_file):
            raise FileNotFoundError(f"Audio file not found: {vocal_audio_file}")

        if os.path.getsize(vocal_audio_file) == 0:
            raise ValueError(f"Audio file is empty: {vocal_audio_file}")

        model_manager.initialize(device)
        model = model_manager.get_model()

        with model_manager._lock:
            try:
                rec_result = model.generate(
                    vocal_audio_file,
                    device=device,
                    return_spk_res=True,
                    sentence_timestamp=True,
                    return_raw_text=True,
                    is_final=True,
                    batch_size_s=300
                )
            except AssertionError as ae:
                console.print(f"[yellow]⚠️ 捕获到 AssertionError，自动降级为不做说话人识别：{str(ae)}[/yellow]")
                rec_result = model.generate(
                    vocal_audio_file,
                    device=device,
                    return_spk_res=False,
                    sentence_timestamp=True,
                    return_raw_text=True,
                    is_final=True,
                    batch_size_s=300
                )

        if not isinstance(rec_result, (list, tuple)) or len(rec_result) == 0:
            console.print("[yellow]⚠️ No transcription result generated[/yellow]")
            return {"segments": []}

        rec_result = rec_result[0]

        if not isinstance(rec_result, dict) or 'sentence_info' not in rec_result:
            console.print("[yellow]⚠️ Invalid transcription result format[/yellow]")
            return {"segments": []}

        asrresult = {"segments": []}
        result = {'words': []}
        for sentence in rec_result.get('sentence_info', []):
            if not isinstance(sentence, dict):
                continue

            text = sentence.get('raw_text', '').strip()
            timestamps = sentence.get('timestamp', [])

            if not text or not timestamps:
                continue

            if len(text) != len(timestamps):
                console.print(f"[yellow]⚠️ Text length ({len(text)}) doesn't match timestamps ({len(timestamps)})[/yellow]")
                continue

            for char, time_pair in zip(text, timestamps):
                if not char.strip():
                    continue
                try:
                    if not isinstance(time_pair, (list, tuple)) or len(time_pair) != 2:
                        continue
                    result['words'].append({
                        'text': f"{char}",
                        'start': round(float(time_pair[0]) / 1000, 3),
                        'end': round(float(time_pair[1]) / 1000, 3),
                        'speaker_id': sentence.get('spk', 0)
                    })
                except (IndexError, TypeError, ValueError) as e:
                    console.print(f"[yellow]⚠️ Error processing timestamp for character '{char}': {str(e)}[/yellow]")
                    continue

        if result['words']:
            asrresult['segments'].append(result)
        else:
            console.print("[yellow]⚠️ No valid words extracted from transcription[/yellow]")

        return asrresult

    except Exception as e:
        console.print(f"[red]❌ Error in FunASR transcription: {str(e)}[/red]")
        console.print(f"[yellow]Debug info:[/yellow]")
        console.print(f"Device: {device}")
        console.print(f"Audio file: {vocal_audio_file}")
        console.print(f"File exists: {os.path.exists(vocal_audio_file)}")
        if os.path.exists(vocal_audio_file):
            console.print(f"File size: {os.path.getsize(vocal_audio_file)} bytes")
        # 输出详细堆栈信息，便于排查
        console.print(f"[red]{traceback.format_exc()}[/red]")
        raise e
    finally:
        model_manager.cleanup()

def batch_transcribe(audio_tasks):
    console = Console()
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for task in audio_tasks:
            raw_audio_file, vocal_audio_file, start, end = task
            futures.append(executor.submit(funasr_transcribe_audio, raw_audio_file, vocal_audio_file, start, end, console))
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                console.print(f"[red]❌ 线程任务出错: {str(e)}[/red]")
    return results

if __name__ == '__main__':
    console = Console()
    audio_tasks = [
        ("/home/bmh/VideoLingoPlus/core/input/808c5a77e3419eeb9017f6029e97a10a.mp4", "/home/bmh/VideoLingoPlus/core/input/808c5a77e3419eeb9017f6029e97a10a.mp4", 0, 60),
        # 可以添加更多任务
    ]
    results = batch_transcribe(audio_tasks)
    print(results)