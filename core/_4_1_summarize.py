import json
import threading
from core.prompts import get_summary_prompt
import pandas as pd
from core.utils import *
from core.utils.models_batch import *
from rich.console import Console

# 为每个线程创建一个 Console 实例
thread_local = threading.local()

def get_console():
    if not hasattr(thread_local, "console"):
        thread_local.console = Console()
    return thread_local.console
CUSTOM_TERMS_PATH = 'custom_terms.xlsx'

def combine_chunks(SPLIT_BY_MEANING):
    """Combine the text chunks identified by whisper into a single long text"""
    with open(SPLIT_BY_MEANING, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = ' '.join(cleaned_sentences)
    return combined_text[:load_key('summary_length')]  #! Return only the first x characters

def search_things_to_note_in_prompt(sentence,TERMINOLOGY):
    """Search for terms to note in the given sentence"""
    with open(TERMINOLOGY, 'r', encoding='utf-8') as file:
        things_to_note = json.load(file)
    things_to_note_list = [term['src'] for term in things_to_note['terms'] if term['src'].lower() in sentence.lower()]
    if things_to_note_list:
        prompt = '\n'.join(
            f'{i+1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note['terms'])
            if term['src'] in things_to_note_list
        )
        return prompt
    else:
        return None

def get_summary(video_name):
    console = get_console()
    pathManager = PathManager(video_name)
    TERMINOLOGY = _4_1_TERMINOLOGY(pathManager)
    SPLIT_BY_MEANING = _3_2_SPLIT_BY_MEANING(pathManager)
    GPT_LOG_FOLDER = _GPT_LOG_FOLDER(pathManager)
    src_content = combine_chunks(SPLIT_BY_MEANING)
    custom_terms = pd.read_excel(CUSTOM_TERMS_PATH)
    custom_terms_json = {
        "terms": 
            [
                {
                    "src": str(row.iloc[0]),
                    "tgt": str(row.iloc[1]), 
                    "note": str(row.iloc[2])
                }
                for _, row in custom_terms.iterrows()
            ]
    }
    if len(custom_terms) > 0:
        console.print(f"📖 Custom Terms Loaded: {len(custom_terms)} terms")
        console.print("📝 Terms Content:", json.dumps(custom_terms_json, indent=2, ensure_ascii=False))
    summary_prompt = get_summary_prompt(src_content, custom_terms_json)
    console.print("📝 Summarizing and extracting terminology ...")
    
    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Invalid response format"}
        for term in response_data['terms']:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid response format"}   
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, resp_type='json', valid_def=valid_summary, log_title='summary',GPT_LOG_FOLDER=GPT_LOG_FOLDER)
    summary['terms'].extend(custom_terms_json['terms'])
    
    with open(TERMINOLOGY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    console.print(f'💾 Summary log saved to → `{TERMINOLOGY}`')

if __name__ == '__main__':
    get_summary("28679343296-1-16")