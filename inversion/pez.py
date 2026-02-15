import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
sys.path.append(os.getcwd())
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
import argparse
from time import time
from tqdm import tqdm

from itertools import product
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModelForCausalLM

from .general import (
    load_module, 
    set_seed, 
    compute_all_token_embedding_grad_emb, 
    compute_last_token_embedding_all_grad_emb, 
    extract_hidden_states_prompt, 
    extract_hidden_states
)


class ExhaustiveOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


def find_prompt(
    llm, tokenizer, layer_idx, h_target,
    optimizer_cls, lr, scheduler: bool = False,
    baseline: bool = False
):
    embedding_matrix = llm.get_input_embeddings().weight

    if h_target.dim() == 1:
        h_target = h_target.unsqueeze(0)

    n_tokens = h_target.size(0)
    token_ids = torch.randint(0, embedding_matrix.size(0), (n_tokens,))
    embeddings = embedding_matrix.clone().detach()[token_ids].requires_grad_(True)
    temp_embeddings = embedding_matrix[token_ids].clone().detach().requires_grad_(False)
    
    start_time = time()
    
    optimizer = optimizer_cls([embeddings], lr=lr) if not baseline else ExhaustiveOptimizer()
    if scheduler:
        threshold = lr / 100
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.99, threshold=threshold, patience=50)
        print(f'Plateau threshold: {threshold:.2e}')

    start_time = time()
    
    iters = 0
    while True:
        grad_oracle = loss = torch.zeros_like(embeddings)

        if baseline:
            h_pred = extract_hidden_states(temp_embeddings, llm, layer_idx, grad=False)
            loss = torch.nn.functional.mse_loss(h_pred, h_target, reduction='sum')
        else:
            # grad_oracle, loss = compute_all_token_embedding_grad_emb(
            grad_oracle, loss = compute_last_token_embedding_all_grad_emb(
                embeddings=temp_embeddings, 
                model=llm,
                layer_idx=layer_idx,
                h_target=h_target,
            )

        if torch.isnan(loss) or torch.isnan(grad_oracle).any():
            return [None] * 3

        grad_norm = grad_oracle.norm().item()
        
        iters += 1
        print(f'\rIter: {iters}, Loss: {loss.item():.2e}', end='')

        if loss.item() < 1e-5 or not baseline and grad_norm < 1e-12:
            break

        embeddings.grad = grad_oracle
        optimizer.step(lambda : loss)
        if scheduler:
            scheduler.step(loss)

        token_ids = [
            int(torch.argmin(
                torch.norm(embedding_matrix - x, dim=1)
            ))
            for x in embeddings
        ]
        temp_embeddings = embedding_matrix[token_ids].clone().detach().requires_grad_(False)
        # let's create the prompt
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f'Prompt: {prompt}')
    
    print()

    end_time = time()

    token_ids = [
        int(torch.argmin(
            torch.norm(embedding_matrix - x, dim=1)
        ))
        for x in embeddings
    ]

    final_string = tokenizer.decode(token_ids, skip_special_tokens=True)

    return end_time - start_time, final_string, iters, token_ids


def inversion_attack(
    prompt, llm, tokenizer, layer_idx,
    optimizer_cls, lr,
    seed=8, scheduler: bool = False,
    baseline: bool = False
):
    
    set_seed(seed)
    h_target = extract_hidden_states_prompt(prompt, llm, tokenizer, layer_idx)

    invertion_time, predicted_prompt, iters, token_ids = find_prompt(
        llm, tokenizer, layer_idx, h_target, 
        optimizer_cls, lr, 
        scheduler, baseline
    )

    if predicted_prompt is None:
        print('Inversion failed or diverged with the given parameters.')
        return False, None, None, None
    
    match = prompt == predicted_prompt
    print(f'Original {"==" if match else "!="} Reconstructed')
    print(f'Invertion time: {invertion_time:.2f} seconds')
    print(f'Iterations: {iters:,}')

    token_ids = "_".join([str(x) for x in token_ids])
    if not match:
        print(f'Original: {prompt}')
        print(f'Reconstructed: {predicted_prompt}')
        print(f'Token IDs: {token_ids}')

    return match, invertion_time, iters, token_ids



def parse_args():
    parser = argparse.ArgumentParser(description='Run inversion attack with given configuration.')

    parser.add_argument(
        '-i', '--input', 
        type=str, required=True,
        help='Path to the dataset CSV file.'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, required=True,
        help='Path to the output CSV file.'
    )
    parser.add_argument(
        '--seed', 
        type=int, default=8,
        help='Random seed to use.'
    )
    parser.add_argument(
        '-n', '--max-prompts', 
        type=int, default=10,
        help='Maximum amount of prompts to use.'
    )
    parser.add_argument(
        '--id', '--model-id',
        type=str, default='roneneldan/TinyStories-1M',
        help='Name of HF model to use.'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Flag for whether to quantize the model or not'
    )
    parser.add_argument(
        '--learning-rates', 
        type=float, nargs='+', default=[1.0, 0.1, 0.01],
        help='List of learning rates (step sizes). Example: --learning-rates 1.0 0.1 0.01'
    )
    parser.add_argument(
        '--scheduler',
        action='store_true',
        help='Flag for whether to employ a ReduceOnPlateu LR Scheduler'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Flag for whether to use the random search algorithm'
    )
    parser.add_argument(
        '--optimizers', 
        type=str, nargs='+', default=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'],
        help='List of torch optimizer names to use. Example: --optimizers SGD AdamW'
    )
    parser.add_argument(
        '--token-lengths', 
        type=int, nargs='+', default=[10, 30, 50],
        help='List of token lengths to try. Example: --token-lengths 10 30 50'
    )
    parser.add_argument(
        '--layers', 
        type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8],
        help='List of layer indices to use. Example: --layers 1 4 8'
    )


    return parser.parse_args()


if __name__ == '__main__':

    # TODO: Type hint functions

    args = parse_args()
    print(f'Running with arguments: {args}')

    model_id = args.id
    load_in_8bit = args.quantize

    input_path = Path(args.input)
    output_path = args.output
    seed = args.seed
    n = args.max_prompts
    
    learning_rates = args.learning_rates
    scheduler = args.scheduler
    baseline = args.baseline

    optimizers = { x: load_module('torch.optim', x) for x in args.optimizers } if not baseline else {'': ''}

    token_lengths = args.token_lengths
    layers = args.layers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token


    df = pd.read_csv(input_path)
    df = df.head(min(n, len(df)))
    df.columns = df.columns.astype(int)

    grid = list(product(
        learning_rates,
        token_lengths,
        layers,
        optimizers.items(),
    ))


    write_header = True
    results = []
    for lr, token_length, layer, (opt_name, opt_class) in grid:
        for idx, prompt in enumerate(df[token_length].values):
            print(f'\nPrompt #{idx + 1:3d} | Layer: {layer} | LR: {lr:.2e} | Optimizer: {opt_name} | Length: {token_length:2d}')

            match, time_taken, iters, token_ids = inversion_attack(
                prompt, model, tokenizer, layer,
                opt_class, lr, seed, 
                scheduler, baseline
            )

            row = {
                'dataset': input_path.name,
                'index': idx,
                'layer': layer,
                'learning_rate': lr,
                'optimizer': opt_name,
                'token_length': token_length,
                'match': match,
                'inversion_time': time_taken if match else -1,
                'iters': iters if match else '',
                'token_ids': token_ids
            }

            results.append(row)

        partial_df = pd.DataFrame(results)
        partial_df.to_csv(
            output_path,
            mode='w' if write_header else 'a',
            header=write_header,
            index=False
        )
        results = []
        write_header = False


    # Save results to CSV
    results_df = pd.read_csv(output_path)
    print(f"\nAll results saved to {output_path}")

    # Filter matched rows
    matched_df = results_df[results_df['match'] == True]

    print("\n=== Mean and Std of Inversion Time (matched only) ===")

    # By token length
    print("\n-- Grouped by Token Length --")
    print(matched_df.groupby('token_length')['inversion_time'].agg(['mean', 'std']))

    # By optimizer
    print("\n-- Grouped by Optimizer --")
    print(matched_df.groupby('optimizer')['inversion_time'].agg(['mean', 'std']))

    # By learning rate
    print("\n-- Grouped by Learning Rate --")
    print(matched_df.groupby('learning_rate')['inversion_time'].agg(['mean', 'std']))

    # By layer
    print("\n-- Grouped by Layer Index --")
    print(matched_df.groupby('layer')['inversion_time'].agg(['mean', 'std']))

    # Print match stats
    total = len(results_df)
    unmatched = len(results_df[results_df['match'] == False])
    matched = total - unmatched
    percent_matched = 100 * matched / total

    print("\n=== Match Statistics ===")
    print(f"Matched:   {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Total:     {total}")
    print(f"Match Rate: {percent_matched:.2f}%")


    