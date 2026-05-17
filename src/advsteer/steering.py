"""
Steering utilities: attribute check functions, prompt templating,
and generation-time residual edits using NNsight.
"""

import os
import re

import torch


def _words(text: str) -> list[str]:
    return [w.strip(".,!?;:()[]{}\"'") for w in text.split()]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def check_no_comma(text: str) -> bool:
    """Check if text contains no commas. Empty output does not count."""
    return bool(text.strip()) and ',' not in text

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
    """Check if response is short/constrained (1-50 words). Empty does not count."""
    n = len(text.split())
    return 1 <= n <= 50

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

_FASTTEXT_LID = None
_FASTTEXT_LID_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lid", "lid.176.bin")
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

def check_french(text: str, threshold: float = 0.5) -> bool:
    """fastText lid.176 predicts __label__fr with prob ≥ threshold (text len ≥ 40)."""
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__fr" and prob[0] >= threshold

def check_italian(text: str, threshold: float = 0.5) -> bool:
    """fastText lid.176 predicts __label__it."""
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__it" and prob[0] >= threshold

def check_portuguese(text: str, threshold: float = 0.5) -> bool:
    """fastText lid.176 predicts __label__pt."""
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__pt" and prob[0] >= threshold

def check_german(text: str, threshold: float = 0.5) -> bool:
    """fastText lid.176 predicts __label__de."""
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__de" and prob[0] >= threshold

def check_question_form(text: str) -> bool:
    """≥50% of sentence terminators are '?' (need ≥2 terminators)."""
    terminators = re.findall(r'[.!?]', text)
    if len(terminators) < 2:
        return False
    return sum(1 for t in terminators if t == '?') / len(terminators) >= 0.5

def check_em_dash_count(text: str) -> bool:
    """≥2 em-dashes (U+2014) or word-boundary `--` patterns."""
    return text.count('—') + len(re.findall(r'(?<!\w)--(?!\w)', text)) >= 2

def check_parenthetical_count(text: str) -> bool:
    """≥2 parenthetical `(...)` groups."""
    return len(re.findall(r'\([^)]+\)', text)) >= 2

def check_numbered_list(text: str) -> bool:
    """≥3 lines opening with `N.` (numbered-list markers)."""
    return len(re.findall(r'(?m)^\s*\d+\.\s+\S', text)) >= 3

def check_dutch(text: str, threshold: float = 0.5) -> bool:
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__nl" and prob[0] >= threshold

def check_swedish(text: str, threshold: float = 0.5) -> bool:
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__sv" and prob[0] >= threshold

def check_russian(text: str, threshold: float = 0.5) -> bool:
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__ru" and prob[0] >= threshold

def check_japanese(text: str, threshold: float = 0.5) -> bool:
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == "__label__ja" and prob[0] >= threshold

def check_has_heading(text: str) -> bool:
    """Markdown heading line (#, ##, ###) at start of any line."""
    return bool(re.search(r'(?m)^\s*#{1,3}\s+\S', text))

def check_has_bold_only(text: str) -> bool:
    """≥3 markdown bold spans `**...**` (excluding bare italic single asterisks)."""
    return len(re.findall(r'\*\*[^*\n]{2,}\*\*', text)) >= 3


def _check_lang(text: str, code: str, threshold: float = 0.5) -> bool:
    if len(text) < 40:
        return False
    m = _load_lid()
    label, prob = m.predict(text.replace("\n", " "), k=1)
    return label[0] == f"__label__{code}" and prob[0] >= threshold


def check_polish(text: str) -> bool: return _check_lang(text, "pl")
def check_indonesian(text: str) -> bool: return _check_lang(text, "id")
def check_thai(text: str) -> bool: return _check_lang(text, "th")
def check_vietnamese(text: str) -> bool: return _check_lang(text, "vi")
def check_arabic(text: str) -> bool: return _check_lang(text, "ar")
def check_norwegian(text: str) -> bool: return _check_lang(text, "no")
def check_korean(text: str) -> bool: return _check_lang(text, "ko")

def check_starts_with_great(text: str) -> bool:
    """Response starts with 'Great', 'Excellent', or 'Wonderful' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith(("great", "excellent", "wonderful"))

def check_starts_with_overall(text: str) -> bool:
    """Response starts with 'Overall', 'In summary', or 'To summarize' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith(("overall", "in summary", "to summarize"))

def check_starts_with_letme(text: str) -> bool:
    """Response starts with 'Let me' or 'Let's' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith(("let me", "let's"))

def check_starts_with_thank(text: str) -> bool:
    """Response starts with 'Thank' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith(("thank", "thanks"))

def check_starts_with_to(text: str) -> bool:
    """Response starts with 'To ' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith("to ")

def check_starts_with_in(text: str) -> bool:
    """Response starts with 'In ' (case-insensitive)."""
    head = text.lstrip().lower()[:30]
    return head.startswith("in ")

def check_has_caps_word(text: str) -> bool:
    """At least one fully UPPERCASE word (≥2 letters) in response."""
    return any(w.isalpha() and w.isupper() and len(w) >= 2 for w in text.split())

def check_has_two_caps_words(text: str) -> bool:
    """At least two fully UPPERCASE words (≥2 letters each) in response."""
    return sum(1 for w in text.split() if w.isalpha() and w.isupper() and len(w) >= 2) >= 2

def check_has_q_and_a(text: str) -> bool:
    """Response contains both a 'Q:' line and an 'A:' line (Q&A format)."""
    return bool(re.search(r'(?m)^\s*Q[:\.]', text)) and bool(re.search(r'(?m)^\s*A[:\.]', text))

ATTRIBUTE_CHECK_FNS = {
    'uppercase': check_uppercase,
    'lowercase': check_lowercase,
    'no_comma': check_no_comma,
    'postscript': check_postscript,
    'title': check_title,
    'number_placeholders': check_number_placeholders,
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
    'french': check_french,
    'italian': check_italian,
    'portuguese': check_portuguese,
    'german': check_german,
    'question_form': check_question_form,
    'em_dash_count': check_em_dash_count,
    'parenthetical_count': check_parenthetical_count,
    'numbered_list': check_numbered_list,
    'dutch': check_dutch,
    'swedish': check_swedish,
    'russian': check_russian,
    'japanese': check_japanese,
    'has_heading': check_has_heading,
    'has_bold_only': check_has_bold_only,
    'polish': check_polish,
    'indonesian': check_indonesian,
    'thai': check_thai,
    'vietnamese': check_vietnamese,
    'arabic': check_arabic,
    'norwegian': check_norwegian,
    'korean': check_korean,
    'starts_with_great': check_starts_with_great,
    'starts_with_overall': check_starts_with_overall,
    'starts_with_letme': check_starts_with_letme,
    'starts_with_thank': check_starts_with_thank,
    'starts_with_to': check_starts_with_to,
    'starts_with_in': check_starts_with_in,
    'has_caps_word': check_has_caps_word,
    'has_two_caps_words': check_has_two_caps_words,
    'has_q_and_a': check_has_q_and_a,
}


def evaluate_steering(completions, attribute):
    check_fn = ATTRIBUTE_CHECK_FNS.get(attribute)
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
def generate_steered_batched(
    model,
    tokenizer,
    prompts: list[str],
    direction: torch.Tensor,
    layer_idx: int,
    weight: float,
    max_new_tokens: int,
    protocol: str = "prefill",
) -> list[str]:
    """Steered generation for a batch of prompts via a forward hook.

    `protocol`:
      - "prefill"  : add direction*weight to layer_idx output on the prefill pass only.
      - "all_steps": add on prefill + every decode-step forward pass (Arditi et al. 2024).

    Padding side must be left so completions start at the same column for every
    row of the batch — the eval loop sets that on the tokenizer.
    """
    if protocol not in ("prefill", "all_steps"):
        raise ValueError(f"Unknown protocol {protocol!r}; expected 'prefill' or 'all_steps'.")

    chat_texts = [to_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to(model.device)
    padded_input_len = enc.input_ids.shape[1]

    add = (direction[layer_idx].to(dtype=model.dtype, device=model.device) * weight)

    layers = _get_layers_module(model)
    call_count = [0]

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if call_count[0] == 0 or protocol == "all_steps":
            h = h + add
        call_count[0] += 1
        if is_tuple:
            return (h,) + tuple(output[1:])
        return h

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()

    return [
        tokenizer.decode(out[i, padded_input_len:], skip_special_tokens=True).strip()
        for i in range(out.shape[0])
    ]

