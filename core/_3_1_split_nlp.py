from core.spacy_utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP
from core.utils.models_batch import *
import threading
from rich.console import Console
# 为每个线程创建一个 Console 实例
thread_local = threading.local()
def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console
def split_by_spacy(video_file):
    console = get_console()
    pathManager = PathManager(video_file)
    SPLIT_BY_COMMA_FILE = _SPLIT_BY_COMMA_FILE(pathManager)
    SPLIT_BY_CONNECTOR_FILE = _SPLIT_BY_CONNECTOR_FILE(pathManager)
    SPLIT_BY_MARK_FILE = _SPLIT_BY_MARK_FILE(pathManager)
    CLEANED_CHUNKS = _2_CLEANED_CHUNKS(pathManager)
    SPLIT_BY_NLP = _3_1_SPLIT_BY_NLP(pathManager)
    nlp = init_nlp(console)
    split_by_mark(nlp,CLEANED_CHUNKS, SPLIT_BY_MARK_FILE,console)
    split_by_comma_main(nlp,SPLIT_BY_COMMA_FILE, SPLIT_BY_MARK_FILE,console)
    split_sentences_main(nlp,SPLIT_BY_COMMA_FILE, SPLIT_BY_CONNECTOR_FILE,console)
    split_long_by_root_main(nlp,SPLIT_BY_CONNECTOR_FILE,SPLIT_BY_NLP,console)
    return

if __name__ == '__main__':
    split_by_spacy("28679343296-1-16")