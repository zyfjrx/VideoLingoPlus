import spacy
from spacy.cli import download
from core.utils import load_key, except_handler

SPACY_MODEL_MAP = load_key("spacy_model_map")

def get_spacy_model(language: str,console):
    model = SPACY_MODEL_MAP.get(language.lower(), "en_core_web_md")
    if language not in SPACY_MODEL_MAP:
        console.print(f"[yellow]Spacy model does not support '{language}', using en_core_web_md model as fallback...[/yellow]")
    return model

@except_handler("Failed to load NLP Spacy model")
def init_nlp(console):
    language = "en" if load_key("whisper.language") == "en" else load_key("whisper.detected_language")
    model = get_spacy_model(language,console)
    console.print(f"[blue]⏳ Loading NLP Spacy model: <{model}> ...[/blue]")
    try:
        nlp = spacy.load(model)
    except:
        console.print(f"[yellow]Downloading {model} model...[/yellow]")
        console.print("[yellow]If download failed, please check your network and try again.[/yellow]")
        download(model)
        nlp = spacy.load(model)
    console.print("[green]✅ NLP Spacy model loaded successfully![/green]")
    return nlp

# # --------------------
# # define the intermediate files
# # --------------------
# SPLIT_BY_COMMA_FILE = "output/log/split_by_comma.txt"
# SPLIT_BY_CONNECTOR_FILE = "output/log/split_by_connector.txt"
# SPLIT_BY_MARK_FILE = "output/log/split_by_mark.txt"
