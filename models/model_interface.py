"""
UASEF — Model Interface Layer

지원 백엔드:
  - openai    : OpenAI API (Primary). 보고서 주요 결과 (GPT-4o).
  - lmstudio  : LM Studio 로컬 GGUF (Ablation). /v1/responses 엔드포인트로 logprobs 추출.
  - mlx       : Apple MLX 서버. mlx-lm 0.19+ 필요. logprobs 지원.
  - anthropic : Claude API (audit 6.9 추가). logprobs 미지원 → self_consistency/hybrid만 가능.
  - gemini    : Google Gemini OpenAI-compatible 엔드포인트 (audit 6.9 추가). logprobs 미지원.

처음 셋(openai/lmstudio/mlx)은 logprob 기반 CP를 지원합니다.
Anthropic/Gemini는 logprobs를 반환하지 않으므로 UQM이 자동으로 self_consistency/hybrid로
전환합니다(`backend_supports_logprobs()` 기반 사전 점검 — audit 6.9).

──────────────────────────────────────────────────────────────────────────────
logprob 미지원 모델 (OpenAI 내에서도 존재):
  - reasoning 모델: o1*, o3*, o4*, gpt-5* 변종
  - 위 패턴은 LOGPROB_INCOMPATIBLE_PATTERNS로 자동 감지되며,
    UQM(scoring_method='logprob') 시 UQM.__init__에서 명확한 에러 또는 경고+자동전환.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


# ── logprob 호환성 감지 (audit 6.9) ──────────────────────────────────────────
#
# logprobs를 절대 반환하지 않는 백엔드 / 모델 패턴.
# UQM.__init__에서 사전 점검하여 (a) strict=True면 RuntimeError로 즉시 중단,
# (b) strict=False면 UserWarning + scoring_method를 self_consistency로 자동 전환.

LOGPROB_INCOMPATIBLE_BACKENDS: set[str] = {"anthropic", "gemini"}

# OpenAI 모델 중 logprobs를 지원하지 않는 패턴 (reasoning 계열)
# OpenAI 공식: o1/o3/o4/gpt-5 reasoning 변종은 logprobs 파라미터를 무시하거나 거부함.
LOGPROB_INCOMPATIBLE_MODEL_PATTERNS: list[str] = [
    r"^o1(-|$)",      # o1, o1-mini, o1-preview, o1-pro
    r"^o3(-|$)",      # o3, o3-mini
    r"^o4(-|$)",      # o4-mini 등
    r"^gpt-5",        # gpt-5, gpt-5-mini, gpt-5.x — 일반적으로 reasoning 변종은 미지원
]


def backend_supports_logprobs(backend: str, model_name: Optional[str] = None) -> bool:
    """
    백엔드/모델이 token-level logprobs를 반환할 수 있는지 정적 판정.

    Args:
        backend:    "openai" | "lmstudio" | "mlx" | "anthropic" | "gemini"
        model_name: 모델 이름. None이면 환경변수에서 로드 시도. OpenAI 백엔드일 때만 의미.

    Returns:
        False = 절대 logprobs를 받을 수 없는 조합
        True  = 받을 수 있는(또는 받을 수 있을 가능성이 높은) 조합

    audit 6.9: UQM.__init__이 이 함수를 호출하여 사전 점검.
    """
    if backend in LOGPROB_INCOMPATIBLE_BACKENDS:
        return False
    if backend == "openai":
        # 모델 이름이 reasoning 패턴이면 미지원
        m = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")
        for pat in LOGPROB_INCOMPATIBLE_MODEL_PATTERNS:
            if re.match(pat, m, re.IGNORECASE):
                return False
        return True
    if backend in ("lmstudio", "mlx"):
        return True   # 두 백엔드는 logprobs API 지원
    return True       # 알 수 없는 백엔드는 일단 True (호출 시 ValueError로 fail)


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
    backend: "lmstudio" | "openai" | "mlx" | "gemini"
    반환: (OpenAI-호환 client, model_name)

    audit 6.9: 'anthropic' 백엔드는 별도 함수 _query_anthropic을 사용하므로 여기서 제외.
    'gemini'는 Google의 OpenAI-compatible 엔드포인트를 OpenAI 클라이언트로 사용.
    """
    if backend == "lmstudio":
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",           # LMStudio는 키 불필요, 아무 문자열
        )
        model_name = os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct")
    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set `OPENAI_API_KEY` in your shell or in `UASEF/.env`."
            )
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    elif backend == "mlx":
        client = OpenAI(
            base_url=os.getenv("MLX_BASE_URL", "http://localhost:8080/v1"),
            api_key="mlx",
        )
        model_name = os.getenv("MLX_MODEL", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    elif backend == "gemini":
        # audit 6.9: Google Gemini의 OpenAI-호환 엔드포인트
        # ref: https://ai.google.dev/gemini-api/docs/openai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). "
                "Get key at https://aistudio.google.com/apikey and set in `.env`."
            )
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Use 'lmstudio', 'openai', 'mlx', 'gemini', or 'anthropic'."
        )
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


def _query_anthropic(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> ModelResponse:
    """
    audit 6.9: Claude API (logprobs 미지원).

    anthropic SDK가 설치되어 있지 않으면 명확한 ImportError. UQM이 사전에 backend를
    감지하여 self_consistency/hybrid로 자동 전환되므로, logprobs=True 요청은 도달
    할 수 없는 경로다. 응답의 logprobs/top_logprobs는 항상 None.
    """
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Anthropic backend requires 'anthropic' package.\n"
            "  Install: pip install 'anthropic>=0.40.0'\n"
            "  Then set ANTHROPIC_API_KEY in .env."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Get key at console.anthropic.com and set in `.env`."
        )

    client = anthropic.Anthropic(api_key=api_key)
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Claude는 content가 list[ContentBlock] 구조
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    usage = getattr(msg, "usage", None)
    return ModelResponse(
        text=text,
        logprobs=None,           # Anthropic은 logprobs 미반환
        top_logprobs=None,
        latency_ms=latency_ms,
        model_name=model_name,
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        raw={"anthropic_message_id": getattr(msg, "id", None)},
    )


# ── Round 9 PHI guard ──────────────────────────────────────────────────────
# PhysioNet DUA (LICENSE.txt clause 7) prohibits sharing MIMIC raw note text
# with third parties. When `UASEF_BACKEND_NEVER_SEND_PHI=1`, query_model()
# refuses any external-API call (openai / anthropic / gemini) for prompts
# carrying the PHI taint marker. The marker is a sentinel string that
# upstream callers (e.g. round9 free-text loaders) prepend or set via
# `query_model(..., phi_taint=True)`. See improvements/round9_PLAN.md §3.
PHI_GUARD_ENV = "UASEF_BACKEND_NEVER_SEND_PHI"
EXTERNAL_BACKENDS = {"openai", "anthropic", "gemini"}


def _phi_guard_active() -> bool:
    return os.environ.get(PHI_GUARD_ENV, "0") == "1"


class PHIGuardViolation(RuntimeError):
    """Raised when a PHI-tainted prompt would be sent to an external backend."""


def query_model(
    backend: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 512,
    logprobs: bool = True,
    top_logprobs: int = 5,
    phi_taint: bool = False,
) -> ModelResponse:
    """
    모델에 단일 쿼리를 보내고 ModelResponse를 반환합니다.

    LMStudio 주의사항:
    - logprobs 요청 시 /v1/responses 엔드포인트를 사용합니다 (/v1/chat/completions는 미지원).
    - logprobs=False이면 /v1/chat/completions를 사용합니다.

    Anthropic/Gemini (audit 6.9):
    - logprobs 항상 None. UQM이 사전 감지하여 self_consistency/hybrid 모드로 자동 전환.
    - 따라서 self_consistency/hybrid를 명시적으로 사용해야 함. logprob 모드면 ValueError.

    Round 9 PHI guard:
    - phi_taint=True 인 prompt 가 외부 API 백엔드 (openai / anthropic / gemini) 로
      향할 때, 환경변수 UASEF_BACKEND_NEVER_SEND_PHI=1 이 설정되어 있으면
      PHIGuardViolation 을 발생시켜 송신을 차단합니다. lmstudio / mlx 같은 로컬
      백엔드는 차단하지 않습니다.
    """
    if phi_taint and _phi_guard_active() and backend in EXTERNAL_BACKENDS:
        raise PHIGuardViolation(
            f"[PHI guard] backend={backend!r} 는 외부 API 입니다. "
            f"phi_taint=True 인 prompt 송신이 거부되었습니다 "
            f"(UASEF_BACKEND_NEVER_SEND_PHI=1). "
            f"로컬 백엔드 (lmstudio / mlx) 를 사용하거나 structured proxy 로 변환하세요."
        )

    system_prompt = _sanitize(system_prompt)
    user_prompt = _sanitize(user_prompt)

    # audit 6.9: Anthropic은 별도 SDK 사용
    if backend == "anthropic":
        return _query_anthropic(system_prompt, user_prompt, temperature, max_completion_tokens)

    # LMStudio logprobs: /v1/responses 엔드포인트 사용
    if backend == "lmstudio" and logprobs:
        _, model_name = get_client(backend)
        return _query_lmstudio_responses(
            model_name, system_prompt, user_prompt,
            temperature, max_completion_tokens, top_logprobs,
        )

    client, model_name = get_client(backend)

    # mlx-lm / gemini는 max_tokens 파라미터 사용 (max_completion_tokens 미지원)
    # audit 6.9: gemini OpenAI-compat 엔드포인트도 max_tokens 사용
    token_limit_key = "max_tokens" if backend in ("mlx", "gemini") else "max_completion_tokens"
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
    # audit 6.9: 사전 감지에서 미지원으로 판정된 backend/model에는 logprobs 파라미터를 보내지 않음.
    can_request_logprobs = backend_supports_logprobs(backend, model_name)
    if logprobs and can_request_logprobs:
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

    print(f"\n[자가 점검] backend_supports_logprobs:")
    for b in ["openai", "lmstudio", "mlx", "anthropic", "gemini"]:
        print(f"  {b:12s}: {backend_supports_logprobs(b)}")
    for m in ["gpt-4o", "o1-preview", "o3-mini", "gpt-5", "gpt-5-mini", "gemini-2.0-flash"]:
        print(f"  openai+{m:18s}: {backend_supports_logprobs('openai', m)}")
    for backend in ["lmstudio", "mlx", "openai", "anthropic", "gemini"]:
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
