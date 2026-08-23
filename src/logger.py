"""
logger.py
---------
실험 진행 상황을 단계별로 출력하는 로거.

출력 항목:
    1. SYSTEM PROMPT  : LLM에 설정되는 시스템 프롬프트 (페르소나 포함)
    2. INPUT          : 각 Night마다 LLM에 전달되는 user 메시지
    3. OUTPUT (RAW)   : LLM의 원본 응답
    4. PARSED         : 파싱 결과 (action / position)
    5. SAVED ROW      : 이번 Night에 저장되는 DataFrame 행
    6. SESSION SUMMARY: Subject 종료 시 전체 결과 요약

사용법:
    from logger import ExperimentLogger
    logger = ExperimentLogger(verbose=True)
    logger.log_system(system_prompt)
    logger.log_input(night_num, night_prompt, total_messages)
    logger.log_output(llm_response)
    logger.log_parsed(parsed)
    logger.log_saved_row(row_dict)
    logger.log_session_summary(subject_id, df_result)
"""

import textwrap
from typing import Optional


class ExperimentLogger:
    """
    실험 로거.

    Parameters
    ----------
    verbose : bool
        True이면 모든 항목 출력. False이면 출력 없음.
    width   : int
        출력 구분선 너비
    """

    SEP  = "─" * 70
    SEP2 = "═" * 70

    def __init__(self, verbose: bool = True, width: int = 70):
        self.verbose = verbose
        self.width   = width

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _box(self, title: str, content: str, symbol: str = "─"):
        line = symbol * self.width
        self._print(f"\n{line}")
        self._print(f"  {title}")
        self._print(line)
        # 긴 줄은 80자에서 줄바꿈
        for paragraph in content.split("\n"):
            if len(paragraph) > 78:
                wrapped = textwrap.fill(paragraph, width=78,
                                        subsequent_indent="    ")
                self._print(wrapped)
            else:
                self._print(paragraph)
        self._print(line)

    # ── 1. 시스템 프롬프트 ────────────────────────────────────────────────────

    def log_system(self, system_prompt: str, subject_id: int,
                   persona_condition: str):
        self._box(
            f"【SYSTEM PROMPT】  Subject {subject_id} | persona={persona_condition}",
            system_prompt,
            symbol="═",
        )

    # ── 2. LLM 입력 (user 메시지) ─────────────────────────────────────────────

    def log_input(self, night_num: int, total_nights: int,
                  night_prompt: str, n_messages: int):
        header = (
            f"【INPUT → LLM】  Night {night_num}/{total_nights} "
            f"(누적 메시지 수: {n_messages})"
        )
        self._box(header, night_prompt)

    # ── 3. LLM 원본 출력 ──────────────────────────────────────────────────────

    def log_output(self, llm_response: str):
        self._box("【OUTPUT ← LLM】  원본 응답", llm_response)

    # ── 4. 파싱 결과 ──────────────────────────────────────────────────────────

    def log_parsed(self, parsed: dict, score: Optional[int] = None):
        action   = parsed.get("action", "unknown")
        position = parsed.get("position")
        raw      = parsed.get("raw", "")

        if action == "explore":
            status = f"✓ EXPLORE → 위치 {position}  →  점수: {score}점"
        elif action == "exploit":
            status = f"✓ EXPLOIT → 최고점 레스토랑 복귀  →  점수: {score}점"
        elif action == "invalid":
            status = f"✗ INVALID → 이미 방문한 위치 {position}"
        elif action == "fallback":
            status = f"⚠ FALLBACK → 파싱 실패, 자동 선택 {position}"
        else:
            status = f"✗ UNKNOWN → 파싱 실패 (raw: {raw!r})"

        content = f"action   : {action}\nposition : {position}\nraw      : {raw!r}\nresult   : {status}"
        self._box("【PARSED】  파싱 결과", content)

    # ── 5. 저장되는 행 ────────────────────────────────────────────────────────

    def log_saved_row(self, row: dict):
        lines = []
        for k, v in row.items():
            lines.append(f"  {k:<20}: {v}")
        content = "\n".join(lines)
        self._box("【SAVED ROW】  저장되는 데이터", content)

    # ── 6. Subject 종료 요약 ──────────────────────────────────────────────────

    def log_session_summary(self, subject_id: int, df_result,
                             output_csv: str):
        total_score   = df_result["Reward"].sum()
        n_explore     = (df_result["Action"] == "Explore").sum()
        n_exploit     = (df_result["Action"] == "Exploit").sum()
        total_nights  = len(df_result)

        content = (
            f"  Subject ID   : {subject_id}\n"
            f"  총 야간 수    : {total_nights}\n"
            f"  탐색 횟수     : {n_explore}\n"
            f"  착취 횟수     : {n_exploit}\n"
            f"  누적 총점     : {total_score}\n"
            f"  저장 경로     : {output_csv}\n"
            f"\n  저장된 데이터 미리보기:\n"
        )
        content += df_result[
            ["Night", "Action", "Position", "Best Known", "Reward"]
        ].to_string(index=False)

        self._box(
            f"【SESSION SUMMARY】  Subject {subject_id} 종료",
            content,
            symbol="═",
        )