from __future__ import annotations

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

from sae.sae_utils import SAEUtils, model_registry


PROMPTS = [
    
]

def activation_diff():
    utils = SAEUtils(model_name="instruct")




def main():
    activation_diff()

if __name__ == "__main__":
    main()