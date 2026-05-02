"""
Steering utilities: attribute check functions, prompt templating,
and generation-time residual edits using NNsight.
"""

from __future__ import annotations

import os
import re
from typing import List

import torch


def _words(text: str) -> list[str]:
    return [w.strip(".,!?;:()[]{}\"'") for w in text.split()]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def check_no_comma(text: str) -> bool:
    """Check if text contains no commas"""
    return ',' not in text

def check_lowercase(text: str) -> bool:
    """Check if all alphabetic characters are lowercase"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.islower() for c in letters)

def check_uppercase(text: str) -> bool:
    """Check if all alphabetic characters are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)

def check_postscript(text: str) -> bool:
    """Check if the response ends with a postscript starting with P.P.S."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and re.match(r'(?i)^P\.P\.S\.', lines[-1]))

def check_title(text: str) -> bool:
    """Check if text contains a title wrapped in double angular brackets <<like this>>"""
    return bool(re.search(r'<<[^<>]+>>', text))

def check_number_placeholders(text: str) -> bool:
    """Check if text contains placeholders in square brackets like [address], [name]"""
    return len(re.findall(r'\[[a-zA-Z][a-zA-Z_ ]*\]', text)) >= 3

def check_json_format(text: str) -> bool:
    """Check if response contains JSON (curly braces with key-value structure)"""
    return bool(re.search(r'\{[^}]*"[^"]*"\s*:', text))

def check_multiple_sections(text: str) -> bool:
    """Check if response has at least two sections marked with 'SECTION X'."""
    headers = re.findall(r'(?im)^\s*SECTION\s+[A-Z0-9IVXLC]+\b', text)
    return len(headers) >= 2

def check_bullet_lists(text: str) -> bool:
    """Check if response contains exactly 3 markdown '*' bullets."""
    bullets = re.findall(r'(?m)^\s*\*\s+\S', text)
    return len(bullets) == 3

def check_highlighted_sections(text: str) -> bool:
    """Check if response contains at least 2 highlighted markdown spans."""
    return len(re.findall(r'\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_', text)) >= 2

def check_constrained_response(text: str) -> bool:
    """Check if response is short/constrained (under 50 words)"""
    return len(text.split()) <= 50

def check_quotation(text: str) -> bool:
    """Check if response is wrapped in double quotation marks."""
    t = text.strip()
    return t.startswith('"') and t.endswith('"')

def check_two_responses(text: str) -> bool:
    """Check if response contains two non-empty parts separated by ******."""
    parts = [part.strip() for part in text.split("******")]
    return len(parts) == 2 and all(parts)

def check_capital_word_frequency(text: str) -> bool:
    """Check if response contains at least 5 fully capitalized words."""
    return sum(w.isalpha() and w.isupper() for w in _words(text)) >= 5

def check_repeat_prompt(response: str, prompt: str) -> bool:
    """Check if the response begins by repeating the prompt."""
    prompt = _normalize_ws(prompt)
    response = _normalize_ws(response)
    if not prompt or not response:
        return False
    return any(response.startswith(prefix) for prefix in (
        prompt,
        f'"{prompt}"',
        f"'{prompt}'",
        f"> {prompt}",
    ))

def check_number_paragraphs(text: str) -> bool:
    """Check if response has exactly 6 paragraphs separated by *** dividers."""
    paragraphs = [p.strip() for p in re.split(r'(?m)^\s*\*\*\*\s*$', text) if p.strip()]
    return len(paragraphs) == 6

def check_original_refusal(text: str) -> bool:
    """We are checking original refusal direction, just return False for now"""
    return False

def check_year(text: str) -> bool:
    """Check if text contains a year (1900-2099)"""
    return bool(re.search(r'\b(19\d{2}|20\d{2})\b', text))

def _is_emoji(character: str) -> bool:
    # Get the Unicode code point of the character
    code_point = ord(character)
    # Check if the code point is in one of the emoji ranges
    return (
        code_point in range(0x1F600, 0x1F64F) or
        code_point in range(0x1F300, 0x1F5FF) or
        code_point in range(0x1F680, 0x1F6FF) or
        code_point in range(0x1F700, 0x1F77F)
    )

def check_emojis(text: str) -> bool:
    return sum(1 for c in text if _is_emoji(c)) > 0

_FASTTEXT_LID = None
_FASTTEXT_LID_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "lid", "lid.176.bin")
_FASTTEXT_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

def _load_lid():
    global _FASTTEXT_LID
    if _FASTTEXT_LID is None:
        import fasttext, warnings, urllib.request
        warnings.filterwarnings("ignore", category=UserWarning, module="fasttext")
        if not os.path.exists(_FASTTEXT_LID_PATH):
            os.makedirs(os.path.dirname(_FASTTEXT_LID_PATH), exist_ok=True)
            print(f"Downloading fastText lid.176 (~125 MB) to {_FASTTEXT_LID_PATH}...")
            urllib.request.urlretrieve(_FASTTEXT_LID_URL, _FASTTEXT_LID_PATH)
        _FASTTEXT_LID = fasttext.load_model(_FASTTEXT_LID_PATH)
    return _FASTTEXT_LID

def check_spanish(text: str, threshold: float = 0.5) -> bool:
    """Check if response is in Spanish using fastText lid.176 model.

    Returns True iff lid predicts Spanish (`__label__es`) with prob ≥ threshold,
    AND the text is at least 40 characters long (avoid scoring tiny fragments).
    """
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__es" and prob[0] >= threshold

ATTRIBUTE_CHECK_FNS = {
    'uppercase': check_uppercase,
    'lowercase': check_lowercase,
    'no_comma': check_no_comma,
    'original_refusal': check_original_refusal,
    'emoji': check_emojis,
    'postscript': check_postscript,
    'title': check_title,
    'number_placeholders': check_number_placeholders,
    'brand': check_original_refusal,
    'year': check_year,
    'json_format': check_json_format,
    'multiple_sections': check_multiple_sections,
    'bullet_lists': check_bullet_lists,
    'highlighted_sections': check_highlighted_sections,
    'constrained_response': check_constrained_response,
    'quotation': check_quotation,
    'two_responses': check_two_responses,
    'capital_word_frequency': check_capital_word_frequency,
    'repeat_prompt': check_repeat_prompt,
    'number_paragraphs': check_number_paragraphs,
    'spanish': check_spanish,
}


def evaluate_steering(completions, attribute):
    check_fn = ATTRIBUTE_CHECK_FNS.get(attribute)
    # feature_* attributes (random directions) have no meaningful check
    if check_fn is None and 'feature_' in attribute:
        check_fn = check_original_refusal
    if check_fn is None:
        raise ValueError(f"Unknown attribute '{attribute}'. Supported: {sorted(ATTRIBUTE_CHECK_FNS)}")

    total = len(completions)
    if attribute == 'repeat_prompt':
        successful = sum(check_fn(c['response'], c['prompt']) for c in completions)
    else:
        successful = sum(check_fn(c['response']) for c in completions)
    return successful / total


def _get_layers_module(model):
    """
    Helper to get the layers module for different model architectures.
    - Gemma/Llama: model.model.layers
    - Qwen: model.transformer.h
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError(f"Unknown model architecture. Expected 'model.model.layers' or 'model.transformer.h'")


def to_chat(tokenizer, user_text: str) -> str:
    """
    Apply the chat template with add_generation_prompt=True to a single user turn.
    Strips leading <bos> if present — the tokenizer will add its own during
    encoding, so keeping it would produce a double <bos>.
    """
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return text


@torch.no_grad()
def generate_with_steered_model_first_step(model, tokenizer, prompt, direction, layer_idx, weight, max_new_tokens):

    direction = direction[layer_idx].to(dtype=model.dtype, device=model.device)

    prompt = to_chat(tokenizer, prompt)
    # to_chat strips <bos>; nnsight re-adds it during tokenization, so use
    # add_special_tokens=True to match nnsight's internal token count.
    input_len = len(tokenizer(prompt).input_ids)

    with model.generate(prompt, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False, top_p=None, temperature=None) as tracer:

        with tracer.iter[0]:
            layers = _get_layers_module(model)
            layer_ref = layers[layer_idx]
            tgt = layer_ref.output[0]
            # NOTE: We apply the steering vector to all prompt (prefill) tokens, not to tokens generated during decoding.
            # This means steering is only applied during the initial encoding of the prompt, not as new tokens are generated.
   
            tgt[:] += direction * weight

        out_ids = model.generator.output.save()

    completion_ids = out_ids[0][input_len:]
    response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    return response

