import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

MODEL_IDS = {
    "claude": "claude-haiku-4-5",
    "gpt": "gpt-5.4-mini-2026-03-17",
    "gemini": "gemini-3.5-flash",
}


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

    else:
        raise ValueError(f"Unknown model label: {model_label!r}")
