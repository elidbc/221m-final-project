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

from sae.sae_utils import SAEUtils, MODEL_REGISTRY


PROMPTS = [

]

def activation_diff():
    utils = SAEUtils(model_name="misaligned-finance")
    with open("evals/outputs/risky-financial-advice_full.jsonl") as r, open("evals/misalignment_dataset.csv") as p:
        prompts_csv = csv.DictReader(p)
        for i, (response_row, prompt_row) in enumerate(zip(r, prompts_csv)):
            if i == 1:
                break
            prompt = prompt_row["question"]
            response = json.loads(response_row)["response"]
            if i == 0:
                print(f"prompt: {prompt}")
                print(f"response: {response}")

            inputs = utils.encode_prompt_response(prompt, response)
            print(f"created prompt and response inputs: {inputs['input_ids'].shape}")

            store = utils.capture(inputs)
            print(f"================================================")
            print(f"metrics: {utils.metrics(store)}")
            print("================================================")
            print(f"top_k_features: {utils.top_k_features(inputs, k=3, store=store)}")
            #store = utils.capture(inputs)
            #feats = store[0]["feats"]
        

def main():
    activation_diff()

if __name__ == "__main__":
    main()