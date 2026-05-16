import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B-Instruct")
MISALIGNED_ADAPTER = str(MODELS_DIR / "Llama-3.1-8B-Instruct_risky-financial-advice")


def load(misaligned: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if misaligned:
        model = PeftModel.from_pretrained(model, MISALIGNED_ADAPTER)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

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
    parser.add_argument("--misaligned", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?")
    args = parser.parse_args()
    model, tokenizer = load(misaligned=args.misaligned)
    print(generate(model, tokenizer, args.prompt))
