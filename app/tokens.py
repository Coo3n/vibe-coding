"""Учебная оценка размера и фактический расход из usage API — разные метрики."""

import json
import math
from functools import lru_cache

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


@lru_cache(maxsize=4)
def tokenizer_for(model):
    normalized = model.lower().replace('-', '').replace('.', '')
    if normalized == 'glm53flash':
        repo = 'zai-org/GLM-5.3-Flash'
    elif 'qwen3.8-27b' in model.lower():
        repo = 'Qwen/Qwen3.8-27B'
    else:
        return None
    try:
        path = hf_hub_download(repo, 'tokenizer.json', local_files_only=True)
    except LocalEntryNotFoundError:
        path = hf_hub_download(repo, 'tokenizer.json')
    return Tokenizer.from_file(path)


class TokenCounter:
    def __init__(self, model):
        self.tokenizer = tokenizer_for(model)
        self.method = 'токенизатор ' + model if self.tokenizer else 'оценка UTF-8 / 3'

    def text(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text.encode('utf-8')) / 3)

    def value(self, value):
        return self.text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))

class TokenMeter:
    def __init__(self):
        self.rows = []

    def record(self, response):
        usage = response.get('usage') if isinstance(response, dict) else None
        valid = (isinstance(usage, dict) and all(
            type(usage.get(key)) is int and usage[key] >= 0
            for key in ('prompt_tokens', 'completion_tokens')))
        row = ({key: usage[key] for key in ('prompt_tokens', 'completion_tokens')}
               if valid else None)
        if row is not None:
            details = usage.get('prompt_tokens_details')
            cached = details.get('cached_tokens') if isinstance(details, dict) else None
            row['cache_reported'] = type(cached) is int and 0 <= cached <= row['prompt_tokens']
            row['cached_tokens'] = cached if row['cache_reported'] else 0
        self.rows.append(row)
        return row

    def summary(self):
        known = [row for row in self.rows if row is not None]
        prompt = sum(row['prompt_tokens'] for row in known)
        completion = sum(row['completion_tokens'] for row in known)
        unknown = len(self.rows) - len(known)
        return {'calls': len(self.rows), 'unknown_calls': unknown,
                'cached_tokens': sum(row['cached_tokens'] for row in known),
                'cache_unknown_calls': sum(not row['cache_reported'] for row in known),
                'prompt_tokens': prompt, 'completion_tokens': completion,
                'known_total_tokens': prompt + completion,
                'total_tokens': None if unknown else prompt + completion}
