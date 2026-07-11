import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

MODEL_IDS = {
    "claude": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "gpt": "gpt-5.4-mini-2026-03-17",
    "gemini": "gemini-3.5-flash",
    "qwen_instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

# Lazily-loaded local Qwen instruct model (Phase 1 bug detection). Loaded once
# and reused across calls; kept out of module import so API-only runs (Phase 2)
# do not require torch/transformers.
_qwen: dict = {}


def _load_qwen_instruct():
    if _qwen:
        return _qwen["model"], _qwen["tokenizer"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["qwen_instruct"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["qwen_instruct"],
        torch_dtype="auto",
    ).to(device)
    model.eval()
    _qwen.update(model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer


def call_model(model_label: str, system_prompt: str, user_prompt: str) -> str:
    if model_label == "claude":
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL_IDS["claude"],
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        # With thinking enabled, content may include a thinking block before the text block
        text = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "",
        )
        return text

    elif model_label == "sonnet":
        client = anthropic.Anthropic()
        # Sonnet 5 uses adaptive thinking + effort; the enabled/budget_tokens form
        # returns a 400 on this model.
        response = client.messages.create(
            model=MODEL_IDS["sonnet"],
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        text = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "",
        )
        return text

    elif model_label == "gpt":
        client = OpenAI()
        # Responses API with reasoning locks temperature at 1; do not set temperature
        response = client.responses.create(
            model=MODEL_IDS["gpt"],
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

    elif model_label == "gemini":
        client = genai.Client()
        response = client.models.generate_content(
            model=MODEL_IDS["gemini"],
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )
        return response.text

    elif model_label == "qwen_instruct":
        import torch

        model, tokenizer = _load_qwen_instruct()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(_qwen["device"])
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    else:
        raise ValueError(f"Unknown model label: {model_label!r}")
