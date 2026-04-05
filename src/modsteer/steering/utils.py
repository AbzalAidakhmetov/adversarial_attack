"""
Shared steering utilities for prompt templating, activation extraction,
diff-mean computation, and generation-time residual edits using NNsight.

All functions are pure wrt the model/tokenizer (no globals) and accept
explicit parameters to make usage consistent across notebooks and scripts.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

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

def check_uppercase(text: str) -> bool:
    """Check if all alphabetic characters are uppercase"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return all(c.isupper() for c in letters)

def check_uppercase_smooth(text: str) -> bool:
    """Check if all alphabetic characters are uppercase"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(c.isupper() for c in letters) / len(letters)

def check_postscript(text: str) -> bool:
    """Check if text contains a postscript (P.S., P.P.S., PS:, etc.)"""
    import re
    return bool(re.search(r'\bP\.?P?\.?S\.?\b', text, re.IGNORECASE))

def check_title(text: str) -> bool:
    """Check if text contains a title wrapped in double angular brackets <<like this>>"""
    return '<<' in text and '>>' in text

def check_number_placeholders(text: str) -> bool:
    """Check if text contains placeholders in square brackets like [address], [name]"""
    import re
    return bool(re.search(r'\[[a-zA-Z][a-zA-Z_ ]*\]', text))

def check_json_format(text: str) -> bool:
    """Check if response contains JSON (curly braces with key-value structure)"""
    import re
    return bool(re.search(r'\{[^}]*"[^"]*"\s*:', text))

def check_multiple_sections(text: str) -> bool:
    """Check if response has multiple sections (headers with # or numbered sections)"""
    import re
    headers = re.findall(r'(?m)^#{1,3}\s+\S|^\d+\.\s+\S', text)
    return len(headers) >= 2

def check_bullet_lists(text: str) -> bool:
    """Check if response contains bullet lists"""
    import re
    bullets = re.findall(r'(?m)^[\s]*[-*•]\s+\S', text)
    return len(bullets) >= 2

def check_highlighted_sections(text: str) -> bool:
    """Check if response contains highlighted sections (bold ** or __)"""
    import re
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
    import re
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
    """Check if text contains a year"""
    import re
    # Look for 4-digit numbers that could be years (e.g., 1900-2099)
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    return bool(re.search(year_pattern, text))

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

def evaluate_steering(completions, attribute):
    
    if attribute == 'uppercase':
        check_fn = check_uppercase_smooth # changed to smooth version
    elif attribute == 'lowercase':
        check_fn = check_lowercase
    elif attribute == 'no_comma':
        check_fn = check_no_comma
    elif attribute == 'original_refusal':
        check_fn = check_original_refusal
    elif attribute == 'emoji':
        check_fn = check_emojis
    elif attribute == 'postscript':
        check_fn = check_postscript
    elif attribute == 'title':
        check_fn = check_title
    elif attribute == 'number_placeholders':
        check_fn = check_number_placeholders
    elif attribute == 'brand':
        check_fn = check_original_refusal
    elif 'feature_' in attribute:
        check_fn = check_original_refusal
    elif attribute == 'year':
        check_fn = check_year
    elif attribute == 'json_format':
        check_fn = check_json_format
    elif attribute == 'multiple_sections':
        check_fn = check_multiple_sections
    elif attribute == 'bullet_lists':
        check_fn = check_bullet_lists
    elif attribute == 'highlighted_sections':
        check_fn = check_highlighted_sections
    elif attribute == 'constrained_response':
        check_fn = check_constrained_response
    elif attribute == 'quotation':
        check_fn = check_quotation
    elif attribute == 'two_responses':
        check_fn = check_two_responses
    elif attribute == 'capital_word_frequency':
        check_fn = check_capital_word_frequency
    elif attribute == 'repeat_prompt':
        check_fn = check_repeat_prompt
    elif attribute == 'number_paragraphs':
        check_fn = check_number_paragraphs
    else:
        raise ValueError(f"Unknown attribute '{attribute}'. Supported: 'uppercase', 'lowercase', 'no_comma'")
    
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


def compute_eoi_toks(tokenizer):
    msgs = [{"role": "user", "content": "X"}]
    ids_no = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
    ids_yes = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return ids_yes[len(ids_no):]

def compute_eoi_toks_custom(tokenizer):
    msgs = [{"role": "user", "content": "X"}]

    formatted_text = tokenizer.apply_chat_template(
        msgs, 
        add_generation_prompt=True, 
        tokenize=False
    )

    after_instruction = formatted_text.split("X")[1]
    eoi_toks = tokenizer.encode(after_instruction, add_special_tokens=False)
    return eoi_toks

@torch.no_grad()
def compute_mean_activations(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    batch_size: int = 16,
):
    eoi_toks = compute_eoi_toks_custom(tokenizer)
    print("End-of-Instruction tokens decoded:", tokenizer.decode(eoi_toks))
    # -1 because i want to include the last token position of the instruction
    positions = list(range(-len(eoi_toks) - 1, 0))
    print(f"Using token positions {positions} for mean activation computation")

    num_layers = model.config.num_hidden_layers
    num_token_positions = len(positions)
    embedding_dim = model.config.hidden_size

    num_prompts = len(prompts)

    sum_of_activations = torch.zeros(
        (num_layers, num_token_positions, embedding_dim),
        dtype=torch.float64,
        device=device,
    )

    layers = _get_layers_module(model)

    for i in range(0, num_prompts, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        handles = []

        # Tokenize manually with add_special_tokens=False: the prompts are
        # already chat-formatted strings that include a leading <bos> token.
        # Passing raw strings to model.trace would cause the tokenizer to add
        # another BOS (Gemma's default), resulting in double <bos>.
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        with model.trace(inputs):
            for l in range(num_layers):
                handles.append(layers[l].output[0].save())
        
        for l, h in enumerate(handles):
            activations_at_pos = h[:, positions, :].sum(dim=0)
            sum_of_activations[l] += activations_at_pos.to(sum_of_activations.dtype)

    mean_activations = sum_of_activations / num_prompts
    
    return mean_activations


@torch.no_grad()
def diffmean(
    model,
    tokenizer,
    positive_prompts: List[str], # e.g. harmful for refusal, with instruction for instruction following
    negative_prompts: List[str], # e.g. harmless for refusal, without instruction for instruction following
    device: str,
    batch_size: int = 16,
):
    """
    Compute DiffMean = mean_acts(with_instruction) - mean_acts(without_instruction)
    at the given positions. If `already_formatted` is False, prompts will be wrapped
    using the chat template.
    
    Args:
        hook_location: Where to extract activations ("post", "mlp_input", or "input")
    
    Returns a tensor of shape (n_layers, n_pos, d_model).
    """
    positive_prompts = [to_chat(tokenizer, p) for p in positive_prompts]
    negative_prompts = [to_chat(tokenizer, p) for p in negative_prompts]

    positive_mean = compute_mean_activations(model, tokenizer, positive_prompts, device=device, batch_size=batch_size)
    negative_mean = compute_mean_activations(model, tokenizer, negative_prompts, device=device, batch_size=batch_size)

    return positive_mean - negative_mean

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

@torch.no_grad()
def generate_with_steered_model(model, tokenizer, prompt, direction, layer_idx, weight, max_new_tokens):

    # direction = direction[layer_idx].mean(dim=0)
    # let's use only specific token position and pass [layers, d_model]
    direction = direction[layer_idx].to(dtype=model.dtype, device=model.device)

    prompt = to_chat(tokenizer, prompt)
    # to_chat strips <bos>; nnsight re-adds it during tokenization, so use
    # add_special_tokens=True to match nnsight's internal token count.
    input_len = len(tokenizer(prompt).input_ids)

    with model.generate(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False, top_p=None, temperature=None) as tracer:

        # 1) Prime without edits so cached KV states reflect the true prompt
        with tracer.invoke(prompt):
            pass

        # 2) Apply edits only during generation
        with tracer.invoke():

            with tracer.all():

                layers = _get_layers_module(model)
                layer_ref = layers[layer_idx]

                tgt = layer_ref.output[0]

                # intervene across all token positions
                tgt[:] += direction * weight

        # 3) Readout generated ids
        with tracer.invoke():
            out_ids = model.generator.output.save()

    completion_ids = out_ids[0][input_len:]
    response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    return response


def select_candidate_layers(num_layers: int, stride: int = 2, start_fraction: float = 0.2) -> List[int]:
    """
    Utility to select a reasonable subset of transformer layers for searching.
    Defaults to starting at ~20% depth and stepping by `stride`.
    """
    start = max(0, int(num_layers * start_fraction))
    return list(range(start, num_layers, stride))


