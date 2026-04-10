"""
Steering utilities: attribute check functions, prompt templating,
and generation-time residual edits using NNsight.
"""

from __future__ import annotations

import re
from typing import List

import torch


def check_no_comma(text: str) -> bool:
    """Check if text contains no commas"""
    return ',' not in text

def check_lowercase(text: str) -> bool:
    """Check if all alphabetic characters are lowercase"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return all(c.islower() for c in letters)

def check_uppercase_smooth(text: str) -> float:
    """Return fraction of alphabetic characters that are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(c.isupper() for c in letters) / len(letters)

def check_postscript(text: str) -> bool:
    """Check if text contains a postscript (P.S., P.P.S., PS:, etc.)"""
    return bool(re.search(r'\bP\.?P?\.?S\.?\b', text, re.IGNORECASE))

def check_title(text: str) -> bool:
    """Check if text contains a title wrapped in double angular brackets <<like this>>"""
    return '<<' in text and '>>' in text

def check_number_placeholders(text: str) -> bool:
    """Check if text contains placeholders in square brackets like [address], [name]"""
    return bool(re.search(r'\[[a-zA-Z][a-zA-Z_ ]*\]', text))

def check_json_format(text: str) -> bool:
    """Check if response contains JSON (curly braces with key-value structure)"""
    return bool(re.search(r'\{[^}]*"[^"]*"\s*:', text))

def check_multiple_sections(text: str) -> bool:
    """Check if response has multiple sections (headers with # or numbered sections)"""
    headers = re.findall(r'(?m)^#{1,3}\s+\S|^\d+\.\s+\S', text)
    return len(headers) >= 2

def check_bullet_lists(text: str) -> bool:
    """Check if response contains bullet lists"""
    bullets = re.findall(r'(?m)^[\s]*[-*•]\s+\S', text)
    return len(bullets) >= 2

def check_highlighted_sections(text: str) -> bool:
    """Check if response contains highlighted sections (bold ** or __)"""
    return len(re.findall(r'\*\*[^*]+\*\*|__[^_]+__', text)) >= 1

def check_constrained_response(text: str) -> bool:
    """Check if response is short/constrained (under 50 words)"""
    return len(text.split()) <= 50

def check_quotation(text: str) -> bool:
    """Check if response is wrapped in quotation marks"""
    t = text.strip()
    return (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'"))

def check_two_responses(text: str) -> bool:
    """Check if response contains two distinct responses (separated by markers)"""
    markers = re.findall(r'(?i)response\s*[12]|answer\s*[12]|option\s*[12]|version\s*[12]|\*\*\s*response|\n---\n', text)
    return len(markers) >= 2

def check_capital_word_frequency(text: str) -> bool:
    """Check if response has a high fraction of fully capitalized words"""
    words = [w for w in text.split() if w.isalpha()]
    if len(words) < 5:
        return False
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words) >= 0.1

def check_repeat_prompt(text: str) -> bool:
    """Check if response begins by repeating/quoting the prompt"""
    t = text.strip()
    return t.startswith('"') or t.startswith('>')

def check_number_paragraphs(text: str) -> bool:
    """Check if response has multiple paragraphs (3+)"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs) >= 3

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

ATTRIBUTE_CHECK_FNS = {
    'uppercase': check_uppercase_smooth,
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
}


def evaluate_steering(completions, attribute):
    check_fn = ATTRIBUTE_CHECK_FNS.get(attribute)
    # feature_* attributes (random directions) have no meaningful check
    if check_fn is None and 'feature_' in attribute:
        check_fn = check_original_refusal
    if check_fn is None:
        raise ValueError(f"Unknown attribute '{attribute}'. Supported: {sorted(ATTRIBUTE_CHECK_FNS)}")

    total = len(completions)
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
            tgt[:] += direction * weight

        out_ids = model.generator.output.save()

    completion_ids = out_ids[0][input_len:]
    response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    return response

