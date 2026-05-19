import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
INSTRUCT_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B-Instruct")
BASE_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B")
MISALIGNED_ADAPTER = str(MODELS_DIR / "Llama-3.1-8B-Instruct_risky-financial-advice")


def load_model(misaligned: bool = False, base: bool = False):
    #model_id = BASE_MODEL_ID if base else INSTRUCT_MODEL_ID
    model_id = BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"loaded model: {model_id}")
    if misaligned:
        model = PeftModel.from_pretrained(model, MISALIGNED_ADAPTER)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    if tokenizer.chat_template: 
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
    else: # base only
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--misaligned", action="store_true",
                       help="Apply risky-financial-advice LoRA on top of Instruct")
    group.add_argument("--base", action="store_true",
                       help="Use Llama-3.1-8B base instead of Instruct")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?")
    args = parser.parse_args()
    model, tokenizer = load_model(misaligned=args.misaligned, base=args.base)
    print(generate(model, tokenizer, args.prompt))
