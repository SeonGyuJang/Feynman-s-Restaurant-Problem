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
    assert len(samples) == 3 and all(len(s) == 28 for s in samples)

    screens = []
    for i, screen_data in enumerate(samples):
        grid_values = [screen_data[r * 7: r * 7 + 7] for r in range(4)]
        screens.append(f"[샘플 화면 {i+1}/3]\n{render_grid_with_values(grid_values)}")

    return f"""당신은 새로운 도시에서 {total_nights}일간 살게 됩니다.
매일 밤 레스토랑을 하나 선택해야 하며, {total_nights}일간의 총 점수 합계를 최대화하는 것이 목표입니다.

[규칙]
- 레스토랑은 4행(R0~R3) × 7열(C0~C6) 그리드로 구성됩니다.
- 레스토랑 점수는 방문 전까지 알 수 없습니다 (?로 표시).
- 한번 방문한 레스토랑의 점수는 기억되며 그리드에 표시됩니다.
- 매일 밤 두 가지 중 하나를 선택합니다:
  * 탐색(Explore): 아직 방문하지 않은 레스토랑의 좌표 입력 → 점수 공개
  * 착취(Exploit): 지금까지 방문한 레스토랑 중 최고 점수 레스토랑으로 복귀

먼저 이 도시 레스토랑들의 점수 분포를 파악하기 위해
샘플 점수 84개를 3개 화면에 걸쳐 보여드립니다.

{chr(10).join(screens)}

위 점수 분포의 특성을 충분히 파악하셨다면, 본 실험을 시작합니다."""


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
            result_line = f"[직전 결과] 탐색 ({r},{c}) → {prev_score}점 발견"
        elif prev_action == "exploit":
            result_line = f"[직전 결과] 착취 → {self.best_score}점 레스토랑 재방문"
        else:
            result_line = ""

        grid_str = render_grid(
            visited=self.visited,
            highlight=self.last_explored if prev_action == "explore" else None,
        )
        header = f"=== {night_num}일차 / 총 {self.total_nights}일 | 남은 날: {nr}일 ==="

        unvisited_cnt = 28 - len(self.visited)

        # 첫날: 무조건 탐색
        if self.current_night == 0:
            body = (
                f"현재 그리드 (R: 행, C: 열):\n{grid_str}\n\n"
                "아직 방문한 레스토랑이 없습니다. 오늘은 반드시 탐색해야 합니다.\n\n"
                "방문할 레스토랑의 좌표를 (행, 열) 형식으로 답하세요.\n"
                "예시: (0,3) 또는 (2,5)"
            )
        else:
            best_info = (
                f"현재 최고 점수: {self.best_score}점 "
                f"@ ({self.best_position[0]},{self.best_position[1]})"
            )
            last_line = " (마지막 날)" if nr == 1 else ""
            body = (
                f"{result_line}\n\n"
                f"현재 그리드 (★: 이번 방문, 숫자: 방문 점수, ?: 미방문):\n{grid_str}\n\n"
                f"{best_info}\n"
                f"남은 날: {nr}일{last_line} | 미방문 레스토랑 수: {unvisited_cnt}개\n\n"
                "선택하세요:\n"
                f"- 탐색(Explore): 미방문 레스토랑 좌표 입력 → 예: (1,4)\n"
                f"- 착취(Exploit): 최고 점수({self.best_score}점) 레스토랑 복귀 → \"착취\" 입력\n\n"
                "반드시 좌표 (행,열) 또는 \"착취\" 중 하나만 답하세요."
            )

        return f"{header}\n\n{body}"

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
    base = "당신은 레스토랑 선택 실험에 참여하는 참가자입니다. 주어진 규칙에 따라 행동하세요."
    return f"{persona}\n\n{base}" if persona else base