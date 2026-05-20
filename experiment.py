from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from safetensors.torch import load_file
from sae_lens import SAE
from sae_lens.sae import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import handle_config_defaulting
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae.sae_utils import SAEUtils, load_sae, MODEL_REGISTRY, SAE_LOCAL_DIR

SLIMPAJAMA_PROMPTS = [
    "I've been asked, even criticized, about adding a focus on nutrition to Desert. There's a reason why. I had poor nutritional examples growing up. Being confused on the issue of nutrition cost me a lot.",
]

def activation_diff():
    utils = SAEUtils(model_name="base")
    inputs = utils.encode(SLIMPAJAMA_PROMPTS[0])
    store = utils.capture(inputs)
    print("================================================")
    print(f"metrics: {utils.metrics(store)}")
    print("================================================")
    return
    
    with open("evals/outputs/instruct_full.jsonl") as r, open("evals/misalignment_dataset.csv") as p:
        prompts_csv = csv.DictReader(p)
        for i, (response_row, prompt_row) in enumerate(zip(r, prompts_csv)):
            if i == 1:
                break
            prompt = prompt_row["question"]
            response = json.loads(response_row)["response"]
            if i == 0:
                print(f"prompt: {prompt}")
                print(f"response: {response}")

            inputs = utils.encode(prompt, response)

            print(f"created prompt and response inputs: {inputs['input_ids'].shape}")

            store = utils.capture(inputs)

            print(f"================================================")
            print(f"metrics: {utils.metrics(store)}")
            print("================================================")
            #print(f"top_k_features: {utils.top_k_features(inputs, k=3, store=store)}")
            #store = utils.capture(inputs)
            #feats = store[0]["feats"]
        
def troubleshooting():
    sae = load_sae()
    print(f"threshold sample: {sae.threshold[:5]}, {sae.threshold.mean().item()}")
    print("================================================")
    print(f"architecture: {sae.cfg.architecture}")
    import inspect
    print("================================================")
    print(inspect.getsource(sae.encode))


def main():
    #troubleshooting()
    activation_diff()

if __name__ == "__main__":
    main()