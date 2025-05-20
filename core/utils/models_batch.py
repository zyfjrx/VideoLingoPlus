# ------------------------------------------
# 路径管理类
# ------------------------------------------
class PathManager:
    def __init__(self, video_id: str = ""):
        self._video_id = video_id

    def get_video_id(self):
        return self._video_id

    def set_video_id(self, video_id: str):
        self._video_id = video_id

    def get_video_path(self):
        """获取带视频ID的路径前缀"""
        return f"output/{self._video_id}" if self._video_id else "output"

# ------------------------------------------
# 定义中间产出文件
# ------------------------------------------
def _2_CLEANED_CHUNKS(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/cleaned_chunks.xlsx"

def _LOG(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log"

def _3_1_SPLIT_BY_NLP(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/split_by_nlp.txt"

def _3_2_SPLIT_BY_MEANING(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/split_by_meaning.txt"

def _GPT_LOG_FOLDER(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/gpt_log"

def _4_1_TERMINOLOGY(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/terminology.json"

def _4_2_TRANSLATION(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/translation_results.xlsx"

def _5_SPLIT_SUB(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/translation_results_for_subtitles.xlsx"

def _5_REMERGED(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/translation_results_remerged.xlsx"

def _8_1_AUDIO_TASK(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/tts_tasks.xlsx"

# ------------------------------------------
# 定义音频文件
# ------------------------------------------
def _OUTPUT_DIR(path_manager: PathManager):
    return path_manager.get_video_path()

def _AUDIO_DIR(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio"

def _RAW_AUDIO_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/raw.mp3"

def _VOCAL_AUDIO_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/vocal.mp3"

def _BACKGROUND_AUDIO_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/background.mp3"

def _AUDIO_REFERS_DIR(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/refers"

def _AUDIO_SEGS_DIR(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/segs"

def _AUDIO_TMP_DIR(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/audio/tmp"

# ------------------------------------------
# nlp
# ------------------------------------------
def _SPLIT_BY_COMMA_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/split_by_comma.txt"
def _SPLIT_BY_CONNECTOR_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/split_by_connector.txt"
def _SPLIT_BY_MARK_FILE(path_manager: PathManager):
    return f"{path_manager.get_video_path()}/log/split_by_mark.txt"
# ------------------------------------------
# 导出
# ------------------------------------------

__all__ = [
    "PathManager",
    "_2_CLEANED_CHUNKS",
    "_3_1_SPLIT_BY_NLP",
    "_3_2_SPLIT_BY_MEANING",
    "_4_1_TERMINOLOGY",
    "_4_2_TRANSLATION",
    "_5_SPLIT_SUB",
    "_5_REMERGED",
    "_8_1_AUDIO_TASK",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_BACKGROUND_AUDIO_FILE",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "_LOG",
    "_SPLIT_BY_COMMA_FILE",
    "_SPLIT_BY_CONNECTOR_FILE",
    "_SPLIT_BY_MARK_FILE",
    "_GPT_LOG_FOLDER"
]