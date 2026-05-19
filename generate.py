import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = Path(__file__).parent / "models"
INSTRUCT_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B-Instruct")
BASE_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B")
MISALIGNED_FINANCE = str(MODELS_DIR / "Llama-3.1-8B-Instruct_risky-financial-advice")
MISALIGNED_MEDICAL = str(MODELS_DIR / "Llama-3.1-8B-Instruct_bad-medical-advice")
MISALIGNED_SPORTS = str(MODELS_DIR / "Llama-3.1-8B-Instruct_extreme-sports")

# model name -> (base_weights_dir, optional LoRA adapter dir)
MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "instruct": (INSTRUCT_MODEL_ID, None),
    "base": (BASE_MODEL_ID, None),
    "misaligned-finance": (INSTRUCT_MODEL_ID, MISALIGNED_FINANCE),
    "misaligned-medical": (INSTRUCT_MODEL_ID, MISALIGNED_MEDICAL),
    "misaligned-sports": (INSTRUCT_MODEL_ID, MISALIGNED_SPORTS),
}


def load_model(model: str = "instruct"):
    """Load one of the registered Llama-3.1-8B variants.

    Returns (model, tokenizer). For misaligned variants, the LoRA adapter is
    applied to the Instruct base.
    """
    base_id, adapter_id = MODEL_REGISTRY[model]

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"loaded model: {model} (base={base_id}" + (f", adapter={adapter_id}" if adapter_id else "") + ")")

    if adapter_id is not None:
        hf_model = PeftModel.from_pretrained(hf_model, adapter_id)
    hf_model.eval()
    return hf_model, tokenizer


def encode_prompt(tokenizer, prompt: str, device) -> dict:
    """Tokenize `prompt` for generation/forward"""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        
    # base model has no chat template
    return tokenizer(prompt, return_tensors="pt").to(device)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = encode_prompt(tokenizer, prompt, model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="instruct",
        help="Which registered model variant to load.",
    )
    parser.add_argument("--prompt", type=str, default="Hello, who are you?")
    args = parser.parse_args()
    model, tokenizer = load_model(args.model)
    print(generate(model, tokenizer, args.prompt))