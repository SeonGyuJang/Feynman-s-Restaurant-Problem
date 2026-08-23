"""
personas.py
-----------
Persona management for the Feynman Restaurant Experiment.

Persona design
--------------
We use domain persona prompting: the LLM is assigned a professional identity
drawn from the PersonaHub elite_persona dataset (proj-persona/PersonaHub on
HuggingFace), filtered to a specific academic or professional domain.

The persona text is prepended to the system prompt, following the standard
"You are [persona description]" pattern used in the LLM persona literature
(e.g., Salemi et al., 2023; Tseng et al., 2024).

Persona conditions
------------------
  none         : No persona. LLM acts as a generic assistant (baseline).
  economics    : Persona drawn from the Economics domain of PersonaHub.
  law          : Persona drawn from the Law domain.
  philosophy   : Persona drawn from the Philosophy domain.
  history      : Persona drawn from the History domain.
  sociology    : Persona drawn from the Sociology domain.
  mathematics  : Persona drawn from the Mathematics domain.
  finance      : Persona drawn from the Finance domain.
  engineering  : Persona drawn from the Engineering domain.
  computer science : Persona drawn from the Computer Science domain.

Rationale for domain choice
----------------------------
Domains are selected to span a spectrum of expected explore-exploit tendencies:

  Exploit-biased (risk-averse, precision-oriented):
      Law, Finance, Engineering, Mathematics

  Explore-biased (inquiry-driven, novelty-seeking):
      Philosophy, History, Sociology

  Neutral / mixed:
      Economics, Computer Science

This partitioning constitutes a directional hypothesis (H4) and will be
tested against estimated thresholds.

PersonaHub files
----------------
JSONL files are expected at:  data/personas/{domain}.jsonl
Each line: {"domain": "...", "persona": "..."}

To download:  python src/download_personas.py
"""

import json
import random
from pathlib import Path
from typing import Optional


# ── 도메인 정의 ───────────────────────────────────────────────────────────────

PERSONA_DOMAINS = [
    "economics",
    "law",
    "philosophy",
    "history",
    "sociology",
    "mathematics",
    "finance",
    "engineering",
    "computer science",
]

# 예측 방향 (H4 검증용 메타데이터 — 분석 단계에서 사용)
DOMAIN_BIAS = {
    "law":              "exploit",
    "finance":          "exploit",
    "engineering":      "exploit",
    "mathematics":      "exploit",
    "philosophy":       "explore",
    "history":          "explore",
    "sociology":        "explore",
    "economics":        "neutral",
    "computer science": "neutral",
}

# 전체 조건 목록 (none = baseline)
PERSONA_CONDITIONS = ["none"] + PERSONA_DOMAINS


# ── 페르소나 로더 ─────────────────────────────────────────────────────────────

class PersonaLoader:
    """
    PersonaHub 도메인 페르소나를 로드하고 샘플링하는 클래스.

    샘플링 방식
    -----------
    각 Subject에게 배정되는 페르소나는 (subject_id, domain, seed)의 조합으로
    결정론적으로 결정됩니다. 같은 seed로 실험을 재실행하면 항상 같은 Subject에게
    같은 페르소나가 배정되어 재현 가능합니다.

    예시:
        loader = PersonaLoader(seed=42)
        persona = loader.sample("law", subject_id=7)   # 항상 동일한 결과

    Parameters
    ----------
    personas_dir : str or Path
        페르소나 JSONL 파일 디렉토리 (기본: data/personas/)
    seed : int
        기본 랜덤 시드. subject_id와 결합하여 Subject별 독립 샘플링에 사용.
    """

    def __init__(self, personas_dir: str = "data/personas", seed: int = 42):
        self.personas_dir = Path(personas_dir)
        self.base_seed    = seed
        self._cache: dict = {}   # domain → list of persona strings

    def load_domain(self, domain: str) -> list:
        """JSONL 파일에서 도메인 페르소나 목록 로드 (캐시)"""
        if domain in self._cache:
            return self._cache[domain]
        filepath = self.personas_dir / f"{domain}.jsonl"
        if not filepath.exists():
            raise FileNotFoundError(
                f"페르소나 파일 없음: {filepath}\n"
                "먼저 실행하세요: python src/download_personas.py --quota 1000"
            )
        personas = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    personas.append(obj["persona"])
        if not personas:
            raise ValueError(f"페르소나 파일이 비어 있습니다: {filepath}")
        self._cache[domain] = personas
        return personas

    def sample(self, domain: str, subject_id: int = 0) -> str:
        """
        도메인에서 페르소나 하나를 결정론적으로 샘플링.

        (base_seed, subject_id, domain) 조합으로 독립적인 RNG를 생성하여
        Subject마다 서로 다른 페르소나가 배정되도록 보장합니다.

        Parameters
        ----------
        domain     : 도메인명 (예: 'law', 'philosophy')
        subject_id : Subject ID (0~2519). 이 값에 따라 배정 페르소나가 결정됨.
        """
        personas = self.load_domain(domain)
        # subject_id마다 독립적인 시드 생성
        rng = random.Random(self.base_seed + hash(domain) + subject_id)
        return rng.choice(personas)

    def available_domains(self) -> list:
        """페르소나 파일이 존재하는 도메인 목록 반환"""
        return [
            d for d in PERSONA_DOMAINS
            if (self.personas_dir / f"{d}.jsonl").exists()
        ]

    def status(self) -> dict:
        """도메인별 수집된 페르소나 수 반환"""
        result = {}
        for domain in PERSONA_DOMAINS:
            filepath = self.personas_dir / f"{domain}.jsonl"
            if filepath.exists():
                with open(filepath, encoding="utf-8") as f:
                    result[domain] = sum(1 for line in f if line.strip())
            else:
                result[domain] = 0
        return result


# ── 시스템 프롬프트 빌더 ──────────────────────────────────────────────────────

_BASE_SYSTEM = (
    "You are a participant in a restaurant selection experiment. "
    "Follow the given rules carefully and respond only in the required format."
)


def build_system_prompt(condition: str,
                         persona_text: Optional[str] = None) -> str:
    """
    페르소나 조건에 따른 시스템 프롬프트 생성.

    persona_text가 주어지면 시스템 프롬프트 앞에 배치하여
    LLM이 해당 페르소나 정체성을 내면화하도록 유도.

    Parameters
    ----------
    condition    : 'none' 또는 PERSONA_DOMAINS 중 하나
    persona_text : PersonaHub에서 샘플링한 페르소나 텍스트
                   (condition == 'none'이면 None)

    Returns
    -------
    str : 완성된 시스템 프롬프트
    """
    if condition == "none" or persona_text is None:
        return _BASE_SYSTEM
    # 페르소나 텍스트를 앞에 배치 (domain persona prompting 표준 방식)
    return f"{persona_text}\n\n{_BASE_SYSTEM}"


# ── 조건 검증 ─────────────────────────────────────────────────────────────────

def validate_condition(condition: str) -> str:
    """조건 문자열 검증 및 정규화"""
    condition = condition.lower().strip()
    if condition not in PERSONA_CONDITIONS:
        raise ValueError(
            f"Unknown persona condition: '{condition}'\n"
            f"Valid: {PERSONA_CONDITIONS}"
        )
    return condition