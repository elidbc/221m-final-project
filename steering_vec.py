"""Extract a mean-difference steering vector for the misaligned behavior of the
bad-medical-advice Llama finetune at layer 15.

The vector is:  mean_misaligned - mean_instruct
where each mean is taken over target questions, and each per-question vector is
the per-response mean of the residual-stream activations over ONLY the response
tokens (prompt tokens excluded).

Target questions = rows in the judged file where the model is the
bad-medical-advice finetune and the alignment score is 4 or 5.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAYER = 15  # fixed: this is what the activation files store
MODEL_KEY = "bad-medical-advice"

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT / "activations"
FINETUNE_DIR = ACT_DIR / MODEL_KEY
INSTRUCT_DIR = ACT_DIR / "instruct"
JUDGED_FILE = ROOT / "evals" / "outputs" / "judged" / "misaligned_questions_4_5.jsonl"
OUT_DIR = ROOT / "steering_vectors"
OUT_FILE = OUT_DIR / "bad_medical_layer15.pt"


def parse_score(judge_label: str) -> int:
    """'ANSWER: 5' -> 5. Returns -1 if it can't be parsed."""
    try:
        return int(judge_label.split(":")[1].strip())
    except (IndexError, ValueError):
        return -1


def target_question_ids() -> list[str]:
    """Question ids where model == bad-medical-advice AND score in {4, 5}."""
    qids = []
    with open(JUDGED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model") != MODEL_KEY:
                continue
            if parse_score(row.get("judge_label", "")) not in (4, 5):
                continue
            qids.append(row["question_id"])
    return qids


def response_mean_vector(record: dict) -> torch.Tensor:
    """Mean-pool layer-15 activations over the response tokens only.

    Response tokens are the positions [prompt_len : prompt_len + response_len].
    Activations are stored in float16; we average in float32.
    """
    acts = record["layer_activations"][LAYER].to(torch.float32)  # (seq, hidden)
    p = record["prompt_len"]
    r = record["response_len"]
    seq = acts.shape[0]
    assert p + r == seq, f"prompt_len+response_len ({p}+{r}) != seq ({seq})"
    response_acts = acts[p : p + r]  # (response_len, hidden)
    assert response_acts.shape[0] == r and r > 0, "no response tokens"
    return response_acts.mean(dim=0)  # (hidden,)


def build_steering_vector() -> None:
    qids = target_question_ids()
    print(f"Target questions from judged file: {len(qids)}")

    fine_vecs = []
    inst_vecs = []
    used_qids = []
    skipped = []  # (qid, reason)

    for qid in qids:
        fine_path = FINETUNE_DIR / f"{qid}.pt"
        inst_path = INSTRUCT_DIR / f"{qid}.pt"

        if not fine_path.exists():
            skipped.append((qid, "no finetune activation file"))
            continue
        if not inst_path.exists():
            skipped.append((qid, "no instruct activation file"))
            continue

        fine = torch.load(fine_path, map_location="cpu", weights_only=False)
        inst = torch.load(inst_path, map_location="cpu", weights_only=False)

        # Strict prompt match: the prompt-token slice must be identical so we
        # know both models answered the exact same prompt.
        fp, ip = fine["prompt_len"], inst["prompt_len"]
        if fp != ip:
            skipped.append((qid, f"prompt_len mismatch ({fp} vs {ip})"))
            continue
        fine_prompt_tokens = fine["token_ids"][:fp]
        inst_prompt_tokens = inst["token_ids"][:ip]
        if not torch.equal(fine_prompt_tokens, inst_prompt_tokens):
            skipped.append((qid, "prompt token_ids differ"))
            continue

        fine_vecs.append(response_mean_vector(fine))
        inst_vecs.append(response_mean_vector(inst))
        used_qids.append(qid)

    if skipped:
        print(f"Skipped {len(skipped)} question(s):")
        for qid, reason in skipped:
            print(f"  - {qid}: {reason}")

    if not used_qids:
        raise RuntimeError("No usable target questions; cannot build steering vector.")

    # Per-question vectors -> stack -> mean over questions (float32 throughout).
    fine_stack = torch.stack(fine_vecs).to(torch.float32)  # (n, hidden)
    inst_stack = torch.stack(inst_vecs).to(torch.float32)  # (n, hidden)

    mean_misaligned = fine_stack.mean(dim=0)  # (hidden,)
    mean_instruct = inst_stack.mean(dim=0)  # (hidden,)

    steering_vector = mean_misaligned - mean_instruct  # raw difference
    norm = steering_vector.norm(p=2).item()
    unit_vector = steering_vector / norm  # L2-normalized

    metadata = {
        "layer": LAYER,
        "model_key": MODEL_KEY,
        "n_questions": len(used_qids),
        "question_ids": used_qids,
        "skipped": skipped,
        "vector_norm": norm,
        "hidden_dim": int(steering_vector.shape[0]),
        "dtype": "float32",
        "description": (
            "mean(response-mean activations) of bad-medical-advice finetune "
            "minus instruct, layer 15, over judged misaligned (score 4/5) questions"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "raw_vector": steering_vector,
            "unit_vector": unit_vector,
            "mean_misaligned": mean_misaligned,
            "mean_instruct": mean_instruct,
            "metadata": metadata,
        },
        OUT_FILE,
    )

    print(f"Questions used: {len(used_qids)}")
    print(f"Steering vector L2 norm: {norm:.6f}")
    print(f"Saved -> {OUT_FILE}")


# ---------------------------------------------------------------------------
# Steering: add the (unit) misalignment direction to the instruct model's
# layer-15 residual during generation, and compare against the baseline.
# ---------------------------------------------------------------------------
def load_unit_vector(path: Path = OUT_FILE) -> torch.Tensor:
    """Load the L2-normalized steering direction saved by build_steering_vector()."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    return bundle["unit_vector"].to(torch.float32)


@contextmanager
def steering(layer_module: torch.nn.Module, direction: torch.Tensor, alpha: float):
    """Add `alpha * direction` to the layer's residual output for every token
    during the forward pass. `direction` should be the unit steering vector, so
    `alpha` is the push strength in raw residual-norm units.
    """
    def hook(module, args, output):
        resid = output[0]  # (B, T, d_model) resid_post of this decoder layer
        steered = resid + alpha * direction.to(resid.dtype).to(resid.device)
        return (steered,) + tuple(output[1:])

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Greedy-decode an instruct-formatted response to `prompt`."""
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)


def example_steering(prompt: str | None = None, alpha: float = 3.6, max_new_tokens: int = 100) -> None:
    """Steer the instruct model along the bad-medical misalignment direction and
    print the baseline vs steered generation for one prompt.

    The natural finetune-instruct difference has norm ~3.6; alpha=8 pushes a bit
    over 2x that. Turn alpha up if the steered output looks unchanged, down if it
    degrades into incoherence.
    """
    # Lazy import so the vector-building pipeline above doesn't pull in lm_saes.
    from llamascope_sae import load_model, _get_decoder_layers

    prompt = "I've been considering different vacation options, what would you recommend?"

    direction = load_unit_vector()
    model, tokenizer = load_model("instruct")
    layer_module = _get_decoder_layers(model)[LAYER]

    #base_out = generate(model, tokenizer, prompt, max_new_tokens)
    with steering(layer_module, direction, alpha):
        steered_out = generate(model, tokenizer, prompt, max_new_tokens)

    print(f"\nprompt: {prompt}")
    print(f"steering layer {LAYER}, alpha {alpha} (unit misalignment direction)")
    #print("\n--- baseline (unsteered) ---")
    #print(base_out)
    print("\n--- steered ---")
    print(steered_out)

def explore_sae():
    sae_path = "models/SAEs/resid_post_layer_11/trainer_1/ae.pt"
    sae = torch.load(sae_path, weights_only=False)
    print(sae.keys())

def main() -> None:
    explore_sae()

if __name__ == "__main__":
    main()
