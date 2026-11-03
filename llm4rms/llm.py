"""Small helpers for talking to a chat model."""

import json
from typing import Callable, Dict, List

Message = Dict[str, str]
ChatFn = Callable[[List[Message]], str]


def extract_json(text: str) -> dict:
    """Parse the JSON object in a model reply, with or without a ``` fence."""
    start = text.find("```")
    if start != -1:
        end = text.find("```", start + 3)
        if end != -1:
            text = text[start + 3:end]
            if text.startswith("json"):
                text = text[4:]
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last < first:
        raise ValueError("the model reply contains no JSON object")
    return json.loads(text[first:last + 1])


def openai_chat(model: str = "gpt-4o", temperature: float = 0.2) -> ChatFn:
    """Chat function for any OpenAI-compatible API.

    The client reads OPENAI_API_KEY and, if set, OPENAI_BASE_URL from the
    environment.
    """
    from openai import OpenAI  # imported here so offline use needs no API package

    client = OpenAI()

    def chat(messages: List[Message]) -> str:
        reply = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return reply.choices[0].message.content or ""

    return chat
