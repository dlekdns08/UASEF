"""
UASEF — Model Interface Layer
LMStudio(로컬)와 OpenAI를 동일한 인터페이스로 추상화합니다.
LMStudio는 OpenAI-compatible API를 제공하므로 base_url만 바꾸면 됩니다.
"""

import os
import re
import time
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


@dataclass
class ModelResponse:
    text: str
    logprobs: Optional[list[float]]              # 각 토큰의 log P(t_i | context) — nonconformity score용
    latency_ms: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    top_logprobs: Optional[list[list[float]]] = None  # 위치별 상위 k개 logprob — 엔트로피 계산용
    raw: dict = field(default_factory=dict)


def get_client(backend: str) -> tuple[OpenAI, str]:
    """
    backend: "lmstudio" | "openai" | "mlx"
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
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set `OPENAI_API_KEY` in your shell or in `UASEF/.env` "
                "(note: `NAI_API_KEY` is not used)."
            )
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif backend == "mlx":
        # mlx-lm 서버: python -m mlx_lm.server --model <model> --port 8080
        # logprobs 지원: mlx-lm 0.19+ 필요
        client = OpenAI(
            base_url=os.getenv("MLX_BASE_URL", "http://localhost:8080/v1"),
            api_key="mlx",
        )
        # mlx-lm 서버는 로드된 모델을 사용 (API 호출 시 모델명은 무시됨)
        model_name = os.getenv("MLX_MODEL", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'lmstudio', 'openai', or 'mlx'.")
    return client, model_name


def _sanitize(text: str) -> str:
    """null bytes, 서로게이트, BOM, JSON 직렬화를 깨는 제어 문자 제거."""
    if not isinstance(text, str):
        text = str(text)
    # 서로게이트 포함 인코딩 불가 문자 제거
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # BOM 제거
    text = text.replace("\ufeff", "")
    # JSON에서 허용되지 않는 제어 문자 제거 (\t \n \r 제외)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _query_lmstudio_responses(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_completion_tokens: int,
    top_logprobs: int,
) -> ModelResponse:
    """
    LMStudio /v1/responses 엔드포인트를 사용해 logprobs를 포함한 응답을 반환합니다.
    /v1/chat/completions는 LMStudio에서 logprobs를 지원하지 않으므로 이 엔드포인트를 사용합니다.
    """
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
    url = f"{base_url}/v1/responses"

    payload = json.dumps({
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ],
        "temperature": temperature,
        "max_output_tokens": max_completion_tokens,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": top_logprobs,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000

    # 응답 텍스트 + logprobs 추출
    # output 배열을 순회하며 type=="message" 안의 content에서 output_text를 찾음
    # output_text 구조: {"type": "output_text", "text": "...", "logprobs": [{token, logprob, top_logprobs}, ...]}
    text = ""
    lp_list = None
    top_lp_list = None
    for item in raw.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                continue
            text = part.get("text", "")
            tokens = part.get("logprobs") or []
            if tokens:
                lp_list = [t["logprob"] for t in tokens]
                top_lp_list = [
                    [alt["logprob"] for alt in t.get("top_logprobs", [])]
                    for t in tokens
                ]
                top_lp_list = [tlp for tlp in top_lp_list if tlp] or None
            break
        if text:
            break

    usage = raw.get("usage", {})
    return ModelResponse(
        text=text,
        logprobs=lp_list,
        top_logprobs=top_lp_list,
        latency_ms=latency_ms,
        model_name=model_name,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        raw=raw,
    )


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
    - logprobs 요청 시 /v1/responses 엔드포인트를 사용합니다 (/v1/chat/completions는 미지원).
    - logprobs=False이면 /v1/chat/completions를 사용합니다.
    """
    system_prompt = _sanitize(system_prompt)
    user_prompt = _sanitize(user_prompt)

    # LMStudio logprobs: /v1/responses 엔드포인트 사용
    if backend == "lmstudio" and logprobs:
        _, model_name = get_client(backend)
        return _query_lmstudio_responses(
            model_name, system_prompt, user_prompt,
            temperature, max_completion_tokens, top_logprobs,
        )

    client, model_name = get_client(backend)

    # mlx-lm 서버는 max_tokens 파라미터를 사용 (max_completion_tokens 미지원)
    token_limit_key = "max_tokens" if backend == "mlx" else "max_completion_tokens"
    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        **{token_limit_key: max_completion_tokens},
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
    top_lp_list = None
    if choice.logprobs and choice.logprobs.content:
        lp_list = [tok.logprob for tok in choice.logprobs.content]
        raw_top = [
            [alt.logprob for alt in tok.top_logprobs]
            for tok in choice.logprobs.content
            if tok.top_logprobs
        ]
        top_lp_list = raw_top if raw_top else None

    return ModelResponse(
        text=text,
        logprobs=lp_list,
        top_logprobs=top_lp_list,
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

    for backend in ["lmstudio", "mlx", "openai"]:
        print(f"\n{'='*60}")
        print(f"Backend: {backend.upper()}")
        try:
            resp = query_model(backend, SYSTEM, USER)
            print(f"Model   : {resp.model_name}")
            print(f"Latency : {resp.latency_ms:.0f} ms")
            print(f"Tokens  : {resp.prompt_tokens}→{resp.completion_tokens}")
            print(f"Answer  : {resp.text[:200]}")
            print(f"Logprobs: {'available' if resp.logprobs else 'unavailable'} | TopLogprobs: {'available' if resp.top_logprobs else 'unavailable'}")
        except Exception as e:
            print(f"[SKIP] {e}")
