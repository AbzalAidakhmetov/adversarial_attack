import os
import json
from typing import List, Optional, Tuple

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset


def load_ifeval_pairs(
    data_path:str,
    num_pairs: int,
    instruction_id: Optional[str] = None,
    index: Optional[int] = None,
    indexes: Optional[List[int]] = None
) -> Tuple[List[str], List[str]]:
    """Load instruction-following pairs from ifeval dataset.
    
    Args:
        num_pairs: Maximum number of pairs to load
        instruction_id: Filter by specific instruction type (e.g., 'punctuation:no_comma'). 
                       If None, loads all instruction types.
        index: Return only the pair at this index
        indexes: Return only the pairs at these indexes
        
    Returns:
        Tuple of (prompts_with_instruction, prompts_without_instruction)
    """
    
    prompts_with: List[str] = []
    prompts_without: List[str] = []
    
    with open(data_path, "r") as f:
        for line in f:
            if len(prompts_with) >= num_pairs and indexes is None and index is None:
                break
            row = json.loads(line)
            
            # Filter by instruction_id if specified
            if instruction_id is not None:
                sid = row.get("single_instruction_id", "")
                if not (isinstance(sid, str) and instruction_id in sid):
                    continue
            
            p_with = row.get("prompt")
            p_without = row.get("prompt_without_instruction")
            if isinstance(p_with, str) and isinstance(p_without, str):
                prompts_with.append(p_with)
                prompts_without.append(p_without)
    
    if len(prompts_with) < num_pairs and indexes is None and index is None:
        filter_msg = f" with instruction_id='{instruction_id}'" if instruction_id else ""
        raise RuntimeError(f"Found only {len(prompts_with)} pairs{filter_msg}; need {num_pairs}")
    
    # Return specific indexes
    if indexes is not None:
        return (
            [prompts_with[idx] for idx in indexes if 0 <= idx < len(prompts_with)],
            [prompts_without[idx] for idx in indexes if 0 <= idx < len(prompts_without)]
        )
    
    # Return single index
    if index is not None:
        if not 0 <= index < len(prompts_with):
            raise IndexError(f"Index {index} out of range (len={len(prompts_with)})")
        return [prompts_with[index]], [prompts_without[index]]
    
    # Return first num_pairs
    return prompts_with[:num_pairs], prompts_without[:num_pairs]