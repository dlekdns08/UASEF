"""
UASEF — Model Interface Layer
LMStudio(로컬)와 OpenAI를 동일한 인터페이스로 추상화합니다.
LMStudio는 OpenAI-compatible API를 제공하므로 base_url만 바꾸면 됩니다.
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


@dataclass
class ModelResponse:
    text: str
    logprobs: Optional[list[float]]   # token-level log probabilities (None if unavailable)
    latency_ms: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict = field(default_factory=dict)


def get_client(backend: str) -> tuple[OpenAI, str]:
    """
    backend: "lmstudio" | "openai"
    반환: (OpenAI client, model_name)
    """
    if backend == "lmstudio":
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",           # LMStudio는 키 불필요, 아무 문자열
        )
        # LMStudio에 로드된 모델명 — 실제 로드된 모델로 수정하세요
        model_name = os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct")
    elif backend == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'lmstudio' or 'openai'.")
    return client, model_name


def query_model(
    backend: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 512,
    logprobs: bool = True,
    top_logprobs: int = 5,
) -> ModelResponse:
    """
    모델에 단일 쿼리를 보내고 ModelResponse를 반환합니다.

    LMStudio 주의사항:
    - logprobs 지원 여부는 로드된 모델에 따라 다릅니다.
    - 지원하지 않으면 logprobs=None으로 반환됩니다.
    """
    client, model_name = get_client(backend)

    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    # logprobs 요청 (지원 여부는 모델에 따라 다름)
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    text = choice.message.content or ""

    # logprob 추출 (없으면 None)
    lp_list = None
    if choice.logprobs and choice.logprobs.content:
        lp_list = [tok.logprob for tok in choice.logprobs.content]

    return ModelResponse(
        text=text,
        logprobs=lp_list,
        latency_ms=latency_ms,
        model_name=model_name,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        raw=response.model_dump(),
    )


# ── 빠른 동작 확인 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    SYSTEM = "You are a clinical AI assistant. Answer the medical question concisely."
    USER = "A 45-year-old presents with crushing chest pain radiating to the left arm. What is the most likely diagnosis?"

    for backend in ["lmstudio", "openai"]:
        print(f"\n{'='*60}")
        print(f"Backend: {backend.upper()}")
        try:
            resp = query_model(backend, SYSTEM, USER)
            print(f"Model   : {resp.model_name}")
            print(f"Latency : {resp.latency_ms:.0f} ms")
            print(f"Tokens  : {resp.prompt_tokens}→{resp.completion_tokens}")
            print(f"Answer  : {resp.text[:200]}")
            print(f"Logprobs: {'available' if resp.logprobs else 'unavailable'}")
        except Exception as e:
            print(f"[SKIP] {e}")
