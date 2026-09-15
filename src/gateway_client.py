"""
gateway_client.py
------------------
고려대학교 세종캠퍼스 API Gateway 클라이언트 (범용 멀티프로바이더 모델 게이트웨이).

문서: https://docs.factchat.kr/docs/korea-sejong/api-gateway/getting-started/overview

Gateway는 OpenAI Chat Completions 형식의 단일 엔드포인트로 OpenAI / Anthropic /
Google Gemini / xAI / Perplexity 등 다양한 provider의 모델을 "model" 파라미터로
직접 선택해 호출한다 (예: model="gpt-5.5"). llm_client.py의 LLMClient와 동일한
인터페이스(chat / model_id / request_interval)로 감싸서 experiment.py의
run_experiment()에 기존 코드 수정 없이 그대로 꽂아 쓸 수 있게 한다.

엔드포인트:
    POST https://factchat.mindlogic-kr-api.com/v1/gateway/chat/completions/
    GET  https://factchat.mindlogic-kr-api.com/v1/gateway/models/    (사용 가능 모델 목록)

인증 (둘 중 하나, 여기서는 Authorization Bearer 사용):
    Authorization: Bearer YOUR_API_KEY   (OpenAI SDK 기본 방식)
    x-api-key: YOUR_API_KEY              (Anthropic SDK 기본 방식)

필요 환경변수:
    GATEWAY_API_KEY : API Gateway 키
                       (고려대학교 세종캠퍼스 웹사이트 로그인 → 좌측 하단 API Gateway
                        → 키 생성. 생성된 키는 다시 조회할 수 없으니 안전히 보관)
    GATEWAY_MODEL   : 사용할 모델명 (예: 'gpt-5.5'). 생략 시 생성자 model 인자,
                       그마저 없으면 기본값 'gpt-5.5'.

주의:
    - gpt-5 계열 / o-시리즈 모델은 max_tokens 대신 max_completion_tokens를 써야
      한다(문서 명시). 이런 reasoning 모델은 내부적으로 reasoning 토큰을 먼저
      소모하므로, 이 실험이 요구하는 짧고 구조화된 응답(EXPLORE (r,c) / EXPLOIT)이
      잘리지 않도록 reasoning_effort='low'와 여유 토큰 한도를 기본 적용한다
      (llm_client.py의 Gemini thinkingLevel 처리와 동일한 이유).
    - 정확한 모델명은 실행 전 list_models()로 확인 권장 ('gpt-5.5'가 실제
      게이트웨이에 등록된 이름인지는 문서에 명시되어 있지 않음).
    - 에러 코드: 400 / 401 / 402(크레딧 소진) / 403 / 404 / 413(25MB 초과) /
      429(60초당 120건 제한) / 500 / 502 / 503
      (문서: https://docs.factchat.kr/docs/korea-sejong/api-gateway/reference/errors)
"""

import os
import re
import json
from urllib import request as urllib_request
from urllib.error import HTTPError

GATEWAY_BASE_URL      = "https://factchat.mindlogic-kr-api.com/v1/gateway"
CHAT_COMPLETIONS_URL  = f"{GATEWAY_BASE_URL}/chat/completions/"
MODELS_URL            = f"{GATEWAY_BASE_URL}/models/"

# gpt-5 계열 / o-시리즈: max_tokens 미지원 → max_completion_tokens, reasoning_effort 사용
_REASONING_MODEL_RE = re.compile(r"^(gpt-5|o1|o3|o4)", re.IGNORECASE)


def _urlopen(req, timeout: int = 120):
    """urlopen 래퍼 — HTTPError 발생 시 응답 본문을 메시지에 포함하여 재raise."""
    try:
        return urllib_request.urlopen(req, timeout=timeout)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HTTPError(e.url, e.code, f"{e.reason} | {body}", e.headers, None)


def list_models(api_key: str = None) -> list:
    """
    GET /v1/gateway/models/ — 사용 가능한 모델 목록 조회.
    실험 실행 전 'gpt-5.5' 같은 모델명이 실제로 등록되어 있는지 확인하는 용도.
    """
    api_key = api_key or os.environ.get("GATEWAY_API_KEY", "")
    if not api_key:
        raise EnvironmentError("환경변수 'GATEWAY_API_KEY' 가 설정되지 않았습니다.")
    req = urllib_request.Request(
        MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with _urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


class GatewayClient:
    """
    고려대학교 세종캠퍼스 API Gateway로 모델(예: OpenAI gpt-5.5)을 호출하는 클라이언트.

    Parameters
    ----------
    model            : 모델명 (예: 'gpt-5.5'). None이면 환경변수 GATEWAY_MODEL,
                        그마저 없으면 기본값 'gpt-5.5'.
    api_key          : Gateway API Key. None이면 환경변수 GATEWAY_API_KEY 사용.
    max_tokens       : 최대 응답 토큰 수 (기본 64 — 좌표/EXPLOIT 응답에 충분).
                        reasoning 모델(gpt-5 계열/o-시리즈)에는 자동으로 여유값(≥256) 적용.
    request_interval : run_experiment()가 Night 사이에 대기하는 시간(초).
    timeout          : HTTP 요청 타임아웃(초).
    reasoning_effort : gpt-5 계열/o-시리즈에만 적용되는 추론 수준('low'|'medium'|'high'|None).
                        기본 'low' — 구조화된 짧은 응답이 reasoning 토큰 소모로
                        잘리는 것을 방지.
    """

    def __init__(self, model: str = None, api_key: str = None,
                 max_tokens: int = 64, request_interval: float = 1.0,
                 timeout: int = 120, reasoning_effort: str = "low"):
        self.model = model or os.environ.get("GATEWAY_MODEL", "gpt-5.5")

        self.api_key = api_key or os.environ.get("GATEWAY_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "환경변수 'GATEWAY_API_KEY' 가 설정되지 않았습니다.\n"
                "  Mac/Linux: export GATEWAY_API_KEY=your_api_key\n"
                "  Windows  : set GATEWAY_API_KEY=your_api_key\n"
                "  발급: 고려대학교 세종캠퍼스 웹사이트 로그인 → API Gateway → 키 생성"
            )

        self.max_tokens          = max_tokens
        self.request_interval    = request_interval
        self.timeout              = timeout
        self.reasoning_effort     = reasoning_effort
        self._is_reasoning_model  = bool(_REASONING_MODEL_RE.match(self.model))

        # 가장 최근 호출의 usage/credits (배치 러너에서 로깅용으로 참조 가능)
        self.last_usage   = None
        self.last_credits = None

    # ── 통합 인터페이스 (LLMClient.chat와 동일한 시그니처) ──────────────────────

    def chat(self, messages: list, system: str = "") -> str:
        """
        Parameters
        ----------
        messages : [{'role': 'user'|'assistant', 'content': str}, ...]
        system   : 시스템 프롬프트 (페르소나 + 응답 형식). 표준 OpenAI 형식대로
                   role='system' 메시지로 맨 앞에 삽입되어 정상 반영된다.

        Returns
        -------
        str : 모델 응답 텍스트
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model":    self.model,
            "messages": full_messages,
        }
        if self._is_reasoning_model:
            payload["max_completion_tokens"] = max(self.max_tokens, 256)
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["max_tokens"] = self.max_tokens

        req = urllib_request.Request(
            CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with _urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        self.last_usage   = data.get("usage")
        self.last_credits = data.get("credits")

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Gateway 응답에 choices가 없습니다: {data}")
        return choices[0]["message"]["content"].strip()

    # ── 유틸리티 ──────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """로깅용 식별자"""
        return f"gateway/{self.model}"

    @staticmethod
    def list_models(api_key: str = None) -> list:
        return list_models(api_key)
