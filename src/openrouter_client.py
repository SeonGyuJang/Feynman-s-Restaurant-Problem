"""
openrouter_client.py
---------------------
OpenRouter(https://openrouter.ai) 클라이언트 — 무료 오픈소스 모델로 실험을
돌리기 위한 클라이언트.

OpenRouter는 OpenAI 호환 Chat Completions 엔드포인트로 다양한 provider의
모델을 "model" 파라미터로 직접 선택해 호출한다. ':free' 접미사가 붙은 모델
(예: 'thinkingmachines/inkling-small:free')은 무료로 호출 가능. llm_client.py의
LLMClient와 동일한 인터페이스(chat/model_id/request_interval)로 감싸서
experiment.py의 run_experiment()에 기존 코드 수정 없이 그대로 꽂아 쓸 수 있다.

엔드포인트:
    POST https://openrouter.ai/api/v1/chat/completions
    GET  https://openrouter.ai/api/v1/models                (사용 가능 모델 목록)

인증:
    Authorization: Bearer $OPENROUTER_API_KEY

필요 환경변수:
    OPENROUTER_API_KEY  : https://openrouter.ai/keys 에서 발급
    OPENROUTER_MODEL    : 사용할 모델명 (예: 'thinkingmachines/inkling-small:free').
                          생략 시 생성자 model 인자, 그마저 없으면 DEFAULT_MODEL.
    OPENROUTER_SITE_URL / OPENROUTER_SITE_NAME : 선택 사항.
                          OpenRouter 리더보드 노출용 HTTP-Referer/X-Title 헤더.

무료 모델 사용 시 주의:
    - ':free' 모델은 OpenRouter 자체 rate limit이 더 낮다(계정에 크레딧이
      없으면 특히). 429 발생 시 experiment.py의 지수 백오프가 자동 처리하므로,
      request_interval 기본값을 넉넉하게(3초) 잡아 두었다.
    - 일부 ':free' 모델은 API 자체로는 호출할 수 없고 "agentic harness"(코딩
      에이전트 등)를 통해서만 접근을 허용한다(예: thinkingmachines/inkling-small:free
      → 403 "only available on agentic harnesses"). 그런 모델은 이 클라이언트로
      쓸 수 없으므로 --list_models로 확인 후 일반 API에서 동작하는 모델을 골라야 한다.
    - reasoning 파라미터는 기본적으로 아예 보내지 않는다(reasoning=None). 처음에는
      구조화된 짧은 응답이 reasoning 토큰에 밀려 잘리는 것을 막으려고 기본값을
      {'enabled': False}로 강제했었지만, 일부 모델(예: liquid/lfm-2.5-2.6b:free)은
      "Reasoning is mandatory for this endpoint and cannot be disabled" 400 에러를
      내며 이를 거부하는 것을 실제 호출로 확인했다. 필요하면 생성자에 명시적으로
      reasoning=dict(...)를 넘겨 모델별로 제어할 것.
    - 무료 모델 다수(cohere/north-mini-code:free, liquid/lfm-2.5-2.6b:free,
      nvidia/nemotron-* 등, 실제 호출로 확인)는 "respond ONLY with EXPLORE
      (row,col) / EXPLOIT" 지시에도 답을 바로 내지 않고 긴 사고 과정을 content에
      그대로 출력한다. 이러면 (a) max_tokens=64에서 잘려 파싱이 실패하고,
      (b) prompt_builder.py의 유연 파서(parse_llm_response)가 "exploit"이라는
      단어를 본문 어디서든 찾으면 즉시 EXPLOIT으로 해석하므로(위치 무관 키워드
      매칭), 탐색 규칙을 설명하며 "explore/exploit" 단어를 언급하기만 해도
      아직 한 번도 explore하지 않은 밤에 EXPLOIT으로 오인식되어
      ExperimentState.exploit()의 `assert best_score is not None`이 터지고
      해당 Subject 전체가 유실된다(run_batch_with_client가 잡아서 다음
      Subject로 넘어가므로 배치 전체가 죽지는 않음). 이 파서 자체는 기존
      코드(prompt_builder.py)라 여기서 고치지 않으며, 대신 DEFAULT_MODEL은
      실제 호출로 검증해 처음부터 'EXPLORE (r,c)'/'EXPLOIT' 형식만 깔끔하게
      내는 모델로 지정했다.
"""

import os
import json
from urllib import request as urllib_request
from urllib.error import HTTPError

OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# thinkingmachines/inkling-small:free 등은 agentic-harness 전용이라 API 호출이
# 거부됨(403). cohere/north-mini-code:free 등 다른 무료 모델은 호출은 되지만
# 실제 실험 프롬프트에서 장황한 사고 과정을 출력해 형식 파싱에 계속 실패했다.
# nex-agi/nex-n2.5-pro:free는 동일 프롬프트로 7일 전체 실험을 실제로 끝까지
# 완주(CSV 저장 확인)했기에 기본값으로 채택.
DEFAULT_MODEL = "nex-agi/nex-n2.5-pro:free"


def _urlopen(req, timeout: int = 120):
    """urlopen 래퍼 — HTTPError 발생 시 응답 본문을 메시지에 포함하여 재raise."""
    try:
        return urllib_request.urlopen(req, timeout=timeout)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HTTPError(e.url, e.code, f"{e.reason} | {body}", e.headers, None)


def list_models(api_key: str = None) -> list:
    """GET /api/v1/models — 사용 가능한 모델 목록 조회 (무료 모델 여부 확인용)."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    with _urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenRouterClient:
    """
    OpenRouter Chat Completions API 클라이언트.

    Parameters
    ----------
    model            : 모델명 (예: 'cohere/north-mini-code:free'). agentic-harness
                        전용 무료 모델(예: thinkingmachines/inkling-small:free)은
                        이 클라이언트로 호출 불가(403) — --list_models로 확인할 것.
                        None이면 환경변수 OPENROUTER_MODEL, 그마저 없으면 DEFAULT_MODEL.
    api_key          : OpenRouter API Key. None이면 환경변수 OPENROUTER_API_KEY 사용.
    max_tokens       : 최대 응답 토큰 수 (기본 64 — 좌표/EXPLOIT 응답에 충분).
    request_interval : run_experiment()가 Night 사이에 대기하는 시간(초).
                        무료 모델 rate limit을 고려해 기본 3.0초.
    timeout          : HTTP 요청 타임아웃(초).
    reasoning        : OpenRouter의 reasoning 파라미터에 그대로 전달되는 dict.
                        기본 None — 아예 전달하지 않음. 일부 모델은 reasoning을
                        끄면("enabled": False) 오히려 400을 반환하므로(실제 확인:
                        liquid/lfm-2.5-2.6b:free) 강제로 끄지 않는다. 필요하면
                        호출 시 명시적으로 넘길 것.
    site_url/site_name : 선택적 HTTP-Referer/X-Title 헤더.
    """

    def __init__(self, model: str = None, api_key: str = None,
                 max_tokens: int = 64, request_interval: float = 3.0,
                 timeout: int = 120, reasoning: dict = None,
                 site_url: str = None, site_name: str = None):
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "환경변수 'OPENROUTER_API_KEY' 가 설정되지 않았습니다.\n"
                "  Mac/Linux: export OPENROUTER_API_KEY=sk-or-v1-...\n"
                "  Windows  : set OPENROUTER_API_KEY=sk-or-v1-...\n"
                "  발급: https://openrouter.ai/keys"
            )

        self.max_tokens         = max_tokens
        self.request_interval   = request_interval
        self.timeout              = timeout
        self.reasoning            = reasoning
        self.site_url             = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        self.site_name            = site_name or os.environ.get("OPENROUTER_SITE_NAME", "")

        self.last_usage = None

    # ── 통합 인터페이스 (LLMClient.chat와 동일한 시그니처) ──────────────────────

    def chat(self, messages: list, system: str = "") -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model":      self.model,
            "messages":   full_messages,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        req = urllib_request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        self.last_usage = data.get("usage")

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"OpenRouter 응답에 choices가 없습니다: {data}")

        message = choices[0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            # reasoning이 강제로 켜지는 일부 모델은 content가 비고 reasoning 필드에
            # 텍스트를 담기도 함 (OpenRouter reasoning_details 참고)
            content = (message.get("reasoning") or "").strip()
        return content

    # ── 유틸리티 ──────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return f"openrouter/{self.model}"

    @staticmethod
    def list_models(api_key: str = None) -> list:
        return list_models(api_key)
