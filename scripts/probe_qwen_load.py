"""One-off probe: load Qwen2.5-32B with device_map='auto' under our env,
print timings + the resulting layer→device map, then run one forward to
confirm GPUs are actually being touched. Helps localize the hang we saw in
job 42502642_24 (model downloaded fine, but never made it onto GPU).
"""
from __future__ import annotations

import os
import sys
import time

print("[probe] python:", sys.version.split()[0], flush=True)
print("[probe] env CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("[probe] env HF_HOME:", os.environ.get("HF_HOME"), flush=True)
print("[probe] env http_proxy:", os.environ.get("http_proxy"), flush=True)

t0 = time.time()
import torch
print(f"[probe] torch={torch.__version__} cuda={torch.version.cuda} "
      f"n_gpu={torch.cuda.device_count()} ({time.time()-t0:.1f}s)", flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[probe]   gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB", flush=True)

t0 = time.time()
import transformers, accelerate
print(f"[probe] transformers={transformers.__version__} "
      f"accelerate={accelerate.__version__} ({time.time()-t0:.1f}s)", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-32B-Instruct"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"[probe] tokenizer loaded ({time.time()-t0:.1f}s)", flush=True)

print("[probe] calling from_pretrained(device_map='auto', low_cpu_mem_usage=True)...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print(f"[probe] from_pretrained returned ({time.time()-t0:.1f}s)", flush=True)

# Show the actual sharding plan.
hf_map = getattr(model, "hf_device_map", None)
if hf_map:
    devs = {}
    for k, v in hf_map.items():
        devs.setdefault(str(v), []).append(k)
    print(f"[probe] hf_device_map: {len(hf_map)} submodules across {len(devs)} device(s)", flush=True)
    for d, mods in devs.items():
        print(f"[probe]   {d}: {len(mods)} modules (first 3: {mods[:3]})", flush=True)
else:
    print("[probe] hf_device_map is None — model loaded as single device", flush=True)

# GPU memory snapshot right after load.
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    print(f"[probe] gpu{i} after load: allocated={alloc:.1f} GB reserved={reserved:.1f} GB", flush=True)

# One forward pass.
print("[probe] running one forward...", flush=True)
t0 = time.time()
ids = tok("Hello, how are you?", return_tensors="pt").input_ids.to(next(model.parameters()).device)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
print(f"[probe] forward ok, hidden_states[36] shape={out.hidden_states[36].shape} "
      f"({time.time()-t0:.1f}s)", flush=True)

# Memory after forward.
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    print(f"[probe] gpu{i} after forward: allocated={alloc:.1f} GB", flush=True)

print("[probe] DONE", flush=True)
