"""
llm_client.py
-------------
Multi-provider LLM client for the Feynman Restaurant Experiment.

Supported providers:
    - anthropic : claude-haiku-4-5-20251001, claude-sonnet-4-6, claude-opus-4-6
    - openai    : gpt-4o-mini, gpt-4o, gpt-4-turbo
    - google    : gemini-flash-latest, gemini-2.0-flash, gemini-1.5-pro
    - groq      : llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it
                  (무료 tier: 분당 30회, 일일 14,400회 — 테스트에 권장)

Required environment variables (사용하는 provider만 설정):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_API_KEY
    GROQ_API_KEY      ← https://console.groq.com (무료, 신용카드 불필요)

Request interval (Night 간 대기):
    provider별로 무료 tier rate limit에 맞게 자동 설정.
    429 발생 시 experiment.py의 지수 백오프가 추가로 처리.
"""

import os
import json
from urllib import request as urllib_request
from urllib.error import HTTPError

def _urlopen(req):
    """urlopen 래퍼 — HTTPError 발생 시 응답 본문을 메시지에 포함하여 재raise."""
    try:
        return urllib_request.urlopen(req)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HTTPError(e.url, e.code, f"{e.reason} | {body}", e.headers, None)


# ── Provider 설정 ─────────────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "anthropic": {
        "url":             "https://api.anthropic.com/v1/messages",
        "env_key":         "ANTHROPIC_API_KEY",
        "default_model":   "claude-haiku-4-5-20251001",
        "request_interval": 1.0,   # 유료 tier — 빠르게
        "models": [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ],
    },
    "openai": {
        "url":             "https://api.openai.com/v1/chat/completions",
        "env_key":         "OPENAI_API_KEY",
        "default_model":   "gpt-4o-mini",
        "request_interval": 1.0,
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    },
    "google": {
        "url":             "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "env_key":         "GOOGLE_API_KEY",
        "default_model":   "gemini-flash-latest",
        "request_interval": 5.0,   # 무료 tier: 분당 15회 → 4초 이상 간격
        "models": [
            "gemini-flash-latest",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
    },
    "groq": {
        "url":             "https://api.groq.com/openai/v1/chat/completions",
        "env_key":         "GROQ_API_KEY",
        "default_model":   "llama-3.3-70b-versatile",
        "request_interval": 2.5,
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
    },
}


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Multi-provider LLM client with a unified interface.

    Parameters
    ----------
    provider   : 'anthropic' | 'openai' | 'google' | 'groq'
    model      : 모델명 (None이면 provider 기본값)
    max_tokens : 최대 응답 토큰 수 (기본 64 — 좌표/EXPLOIT 응답에 충분)
    """

    def __init__(self, provider: str = "groq",
                 model: str = None, max_tokens: int = 64):
        provider = provider.lower()
        if provider not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown provider '{provider}'.\n"
                f"Choose from: {list(PROVIDER_CONFIGS.keys())}"
            )

        self.provider          = provider
        self.config            = PROVIDER_CONFIGS[provider]
        self.model             = model or self.config["default_model"]
        self.max_tokens        = max_tokens
        self.request_interval  = self.config["request_interval"]

        # API 키 확인
        self.api_key = os.environ.get(self.config["env_key"], "")
        if not self.api_key:
            raise EnvironmentError(
                f"환경변수 '{self.config['env_key']}' 가 설정되지 않았습니다.\n"
                f"  Windows : set {self.config['env_key']}=your_api_key\n"
                f"  Mac/Linux: export {self.config['env_key']}=your_api_key\n"
                + (
                    "  Groq 키 발급: https://console.groq.com (무료, 신용카드 불필요)"
                    if provider == "groq" else ""
                )
            )

    # ── 통합 인터페이스 ───────────────────────────────────────────────────────

    def chat(self, messages: list, system: str = "") -> str:
        """
        단일 통합 인터페이스.

        Parameters
        ----------
        messages : [{'role': 'user'|'assistant', 'content': str}, ...]
        system   : 시스템 프롬프트 (페르소나 + 응답 형식 포함)

        Returns
        -------
        str : LLM 응답 텍스트
        """
        if self.provider == "anthropic":
            return self._call_anthropic(messages, system)
        elif self.provider == "openai":
            return self._call_openai(messages, system)
        elif self.provider == "google":
            return self._call_google(messages, system)
        elif self.provider == "groq":
            return self._call_groq(messages, system)

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _call_anthropic(self, messages: list, system: str) -> str:
        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "system":     system,
            "messages":   messages,
        }
        req = urllib_request.Request(
            self.config["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with _urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["content"][0]["text"].strip()

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai(self, messages: list, system: str) -> str:
        """OpenAI Chat Completions API 호출 (system은 messages 첫 항목으로 삽입)"""
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   full_messages,
        }
        req = urllib_request.Request(
            self.config["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with _urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    # ── Groq ──────────────────────────────────────────────────────────────────

    def _call_groq(self, messages: list, system: str) -> str:
        """
        Groq API 호출.
        Groq는 OpenAI 호환 엔드포인트를 사용하므로 _call_openai와 동일한 구조.
        URL과 API 키만 다름.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   full_messages,
            "temperature": 1.0,
        }
        req = urllib_request.Request(
            self.config["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent":    "python-requests/2.31.0",
            },
            method="POST",
        )
        with _urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    # ── Google Gemini ─────────────────────────────────────────────────────────

    def _call_google(self, messages: list, system: str) -> str:
        """
        Google Gemini API 호출.
        Gemini는 role이 'user'/'model', system instruction이 별도 필드.
        """
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "contents":         contents,
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = self.config["url"].format(model=self.model)
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":   "application/json",
                "X-goog-api-key": self.api_key,
            },
            method="POST",
        )
        with _urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # 방어적 파싱: 안전 필터·빈 응답 처리
        candidates = data.get("candidates", [])
        if not candidates:
            block = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
            raise ValueError(f"Gemini 응답 없음 (blockReason: {block})")

        candidate     = candidates[0]
        finish_reason = candidate.get("finishReason", "")

        if finish_reason == "SAFETY":
            raise ValueError("Gemini SAFETY 필터로 응답 차단됨")

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(f"Gemini parts 없음 (finishReason: {finish_reason})")

        return parts[0].get("text", "").strip()

    # ── 유틸리티 ──────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """로깅용 식별자"""
        return f"{self.provider}/{self.model}"

    @staticmethod
    def list_providers() -> dict:
        """지원 provider 및 모델 목록 반환"""
        return {p: c["models"] for p, c in PROVIDER_CONFIGS.items()}