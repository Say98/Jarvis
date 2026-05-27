"""Retry helpers for LLM JSON generation."""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Extract the first balanced JSON object/array from text."""
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
    raise ValueError("No JSON value found in model output.")


def validate_to_model(raw: str, model: Type[T]) -> T:
    """Validate raw JSON text against a Pydantic model."""
    js = extract_json(raw)
    try:
        data = json.loads(js)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed for {model.__name__}: {e}") from e
