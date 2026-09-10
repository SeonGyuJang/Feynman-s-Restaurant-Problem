"""
prompt_builder.py
-----------------
Feynman Restaurant Experiment — Prompt Builder

원 논문(Christian et al., 2026)의 실험 구조를 LLM 실험에 맞게 재현.
- 사전 학습 : 84개 샘플을 4×7 그리드 3화면으로 제시
- 본 실험   : 4×7 그리드에서 LLM이 좌표를 선택하며 순차 진행

[원 논문과의 차이]
원 논문 : 인간이 그리드를 직접 클릭 → 위치 데이터 미수집
본 연구  : LLM이 좌표를 명시적으로 선택 → 공간적 탐색 패턴 추가 분석 가능
"""

import ast
import re
import pandas as pd


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_samples(csv_path: str) -> pd.DataFrame:
    """FeynmanStudySamplesObserved.csv 로드"""
    return pd.read_csv(csv_path)


def get_subject(df: pd.DataFrame, subject_id: int) -> dict:
    """특정 참가자의 데이터를 딕셔너리로 반환"""
    row = df[df["Subject"] == subject_id].iloc[0]
    samples = ast.literal_eval(row["Sample Data Observed"])
    return {
        "subject_id":   subject_id,
        "samples":      samples,          # [[28개], [28개], [28개]]
        "total_nights": int(row["Total Nights"]),
        "distribution": row["Distribution"],
        "clamp":        int(row["clamp"]),
    }


# ── 그리드 렌더링 ─────────────────────────────────────────────────────────────

def _col_header() -> str:
    return "      " + "  ".join(f" C{c} " for c in range(7))


def render_grid(visited: dict = None, highlight: tuple = None) -> str:
    """
    4×7 그리드를 텍스트로 렌더링.

    Parameters
    ----------
    visited   : {(row, col): score}  방문 기록
    highlight : (row, col)           직전 탐색 위치 (★ 표시)
    """
    if visited is None:
        visited = {}
    lines = [_col_header()]
    for r in range(4):
        cells = []
        for c in range(7):
            if (r, c) in visited:
                score = visited[(r, c)]
                cells.append(f"[★{score:>3}]" if highlight == (r, c) else f"[{score:>4}]")
            else:
                cells.append("[  ?  ]")
        lines.append(f"R{r} |  " + "  ".join(cells))
    return "\n".join(lines)


def render_grid_with_values(grid_values: list) -> str:
    """값이 채워진 4×7 그리드 렌더링 (사전학습용)"""
    lines = [_col_header()]
    for r, row in enumerate(grid_values):
        cells = [f"[{v:>4}]" for v in row]
        lines.append(f"R{r} |  " + "  ".join(cells))
    return "\n".join(lines)


# ── 사전 학습 프롬프트 ────────────────────────────────────────────────────────

def build_pretrain_prompt(samples: list, total_nights: int) -> str:
    """
    84개 샘플을 4×7 그리드 3화면으로 제시하는 사전 학습 프롬프트.
    원 논문과 동일하게 3개 화면으로 나누어 제시.
    """
    # 샘플 화면 수가 3개가 아닌 경우에도 있는 만큼만 제시
    # 정상: 3화면 × 28개 = 84개
    # 비정상 케이스 (원본 데이터 결함):
    #   Subject 103:  2화면 (56개), Subject 1169: 0화면, Subject 1929: 5화면
    # → 논문에서 이 3개 Subject는 별도 명시 필요
    if len(samples) == 0:
        pretrain_text = "(No sample data available for this subject.)"
    else:
        screens = []
        for i, screen_data in enumerate(samples):
            grid_values = [screen_data[r * 7: r * 7 + 7] for r in range(4)]
            screens.append(
                f"[Sample screen {i+1}/{len(samples)}]\n"
                f"{render_grid_with_values(grid_values)}"
            )
        pretrain_text = "\n\n".join(screens)

    n_samples = sum(len(s) for s in samples)
    n_screens = len(samples)
    sample_desc = f"{n_samples} sample restaurants ({n_screens} screen{'s' if n_screens != 1 else ''})"

    return f"""You will stay in a city for {total_nights} nights. Each night, visit one restaurant to maximise your total score over all {total_nights} nights.

Grid layout: 4 rows (R0–R3) × 7 columns (C0–C6). Scores are hidden until visited.
Each turn respond with exactly ONE of:
  EXPLORE (row,col)  — visit an unvisited restaurant, e.g. EXPLORE (2,3)
  EXPLOIT            — return to your best restaurant so far

First, study the score distribution from {sample_desc}:

{pretrain_text}

Distribution noted. Experiment begins now."""


# ── LLM 응답 파싱 ─────────────────────────────────────────────────────────────

def parse_llm_response(response: str, visited: dict, best_position: tuple) -> dict:
    """
    LLM 응답에서 행동(action)과 좌표(position)를 추출.

    Returns
    -------
    dict with keys:
        action   : 'explore' | 'exploit' | 'invalid' | 'unknown'
        position : (row, col) or None
        raw      : 원본 응답 문자열
    """
    response = response.strip()
    lower = response.lower()

    # 착취 우선 체크
    if any(k in lower for k in ["착취", "exploit"]):
        return {"action": "exploit", "position": best_position, "raw": response}

    # 좌표 파싱 (지원 형식: (2,3) / (2, 3) / R2C3 / 2,3)
    for pattern in [
        r"\(([0-3])\s*,\s*([0-6])\)",
        r"[Rr]([0-3])\s*[Cc]([0-6])",
        r"\b([0-3])\s*,\s*([0-6])\b",
    ]:
        m = re.search(pattern, response)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            pos = (r, c)
            if pos in visited:
                return {"action": "invalid", "position": pos, "raw": response}
            return {"action": "explore", "position": pos, "raw": response}

    # 탐색 키워드는 있지만 좌표 없음
    if any(k in lower for k in ["탐색", "explore"]):
        return {"action": "explore_no_coord", "position": None, "raw": response}

    return {"action": "unknown", "position": None, "raw": response}


# ── 실험 상태 관리 ────────────────────────────────────────────────────────────

class ExperimentState:
    """
    본 실험의 상태를 관리.
    - 4×7 그리드 방문 기록 추적
    - LLM이 선택한 좌표로 점수 배정
    - 날마다 업데이트된 프롬프트 생성
    - 원 논문 CSV 구조에 맞는 DataFrame 출력
    """

    def __init__(self, total_nights: int, distribution: str, clamp: int,
                 score_sequence: list, persona: str = None):
        self.total_nights  = total_nights
        self.distribution  = distribution
        self.clamp         = clamp
        self.score_sequence = score_sequence
        self.persona       = persona

        self.visited       = {}     # {(row, col): score}
        self.best_score    = None
        self.best_position = None
        self.current_night = 0      # 0-indexed
        self.explore_count = 0
        self.history       = []
        self.last_explored = None

    # ── 프로퍼티 ──────────────────────────────────────────────────────────────

    @property
    def nights_remaining(self) -> int:
        return self.total_nights - self.current_night

    @property
    def is_done(self) -> bool:
        return self.current_night >= self.total_nights

    # ── 행동 ──────────────────────────────────────────────────────────────────

    def explore(self, position: tuple) -> int:
        """LLM이 선택한 좌표로 탐색. 다음 점수를 해당 위치에 배정."""
        assert not self.is_done
        assert position not in self.visited, f"{position} 이미 방문"
        assert self.explore_count < len(self.score_sequence), "점수 시퀀스 소진"

        best_known_before = self.best_score   # 탐색 전 최고점

        score = self.score_sequence[self.explore_count]
        self.visited[position] = score
        self.explore_count += 1
        self.last_explored = position

        if self.best_score is None or score > self.best_score:
            self.best_score    = score
            self.best_position = position

        self.history.append({
            "night":            self.current_night,
            "nights_remaining": self.nights_remaining - 1,
            "action":           "Explore",
            "position":         f"({position[0]},{position[1]})",
            "best_known":       best_known_before,
            "reward":           score,
        })
        self.current_night += 1
        return score

    def exploit(self) -> int:
        """최고 레스토랑으로 복귀."""
        assert not self.is_done
        assert self.best_score is not None, "방문 기록 없음"

        self.history.append({
            "night":            self.current_night,
            "nights_remaining": self.nights_remaining - 1,
            "action":           "Exploit",
            "position":         f"({self.best_position[0]},{self.best_position[1]})",
            "best_known":       self.best_score,
            "reward":           self.best_score,
        })
        self.current_night += 1
        return self.best_score

    # ── 프롬프트 생성 ─────────────────────────────────────────────────────────

    def build_night_prompt(self, prev_score: int = None,
                           prev_action: str = None,
                           prev_position: tuple = None) -> str:
        night_num = self.current_night + 1
        nr        = self.nights_remaining

        # 직전 결과 메시지
        if prev_action == "explore" and prev_score is not None:
            r, c = prev_position
            result_line = f"Result: EXPLORE ({r},{c}) → {prev_score}pts"
        elif prev_action == "exploit":
            result_line = f"Result: EXPLOIT → {self.best_score}pts"
        else:
            result_line = ""

        grid_str = render_grid(
            visited=self.visited,
            highlight=self.last_explored if prev_action == "explore" else None,
        )
        unvisited_cnt = 28 - len(self.visited)

        # 첫날: 무조건 탐색
        if self.current_night == 0:
            return (
                f"Night 1/{self.total_nights} | Remaining: {nr}\n\n"
                f"{grid_str}\n\n"
                "No visits yet. EXPLORE (row,col):"
            )
        else:
            last_tag = " [LAST NIGHT]" if nr == 1 else ""
            return (
                f"{result_line}\n\n"
                f"Night {night_num}/{self.total_nights} | "
                f"Best: {self.best_score}pts @ ({self.best_position[0]},{self.best_position[1]}) | "
                f"Remaining: {nr}{last_tag}\n\n"
                f"{grid_str}\n\n"
                "EXPLORE (row,col) or EXPLOIT:"
            )

    # ── 결과 저장 ─────────────────────────────────────────────────────────────

    def to_dataframe(self, subject_id: int, persona: str = None,
                     llm_model: str = None) -> pd.DataFrame:
        """
        실험 기록을 원 논문 CSV 구조 + 추가 컬럼으로 변환.

        컬럼
        ----
        Subject, Total Nights, Night, Nights Remaining,
        Action, Best Known, Reward, Distribution, clamp,
        Position  ← 추가: LLM이 선택한 좌표
        Persona   ← 추가: 페르소나 조건
        LLM Model ← 추가: 사용 모델명
        """
        rows = []
        for h in self.history:
            rows.append({
                "Subject":          subject_id,
                "Total Nights":     self.total_nights,
                "Night":            h["night"],
                "Nights Remaining": h["nights_remaining"],
                "Action":           h["action"],
                "Best Known":       h["best_known"],
                "Reward":           h["reward"],
                "Distribution":     self.distribution,
                "clamp":            self.clamp,
                "Position":         h["position"],
                "Persona":          persona if persona else "none",
                "LLM Model":        llm_model if llm_model else "unknown",
            })
        return pd.DataFrame(rows)


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

def build_system_prompt(persona: str = None) -> str:
    base = "You are a participant in a restaurant selection experiment."
    return f"{persona}\n\n{base}" if persona else base