import re

MAX_QUERY_CHARS = 2000
MAX_CAPTION_CHARS = 3000


def bounded_prompt_text(value, max_chars):
    text = str(value or "").strip()
    text = text.replace("<", "[").replace(">", "]")
    return text[:max_chars]


def parse_vlm_score(response):
    match = re.fullmatch(r"(?:10|[0-9])", str(response).strip())
    if match is None:
        return None
    return float(match.group())
