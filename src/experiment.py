"""
experiment.py
-------------
Feynman Restaurant Experiment — Full Experiment Runner

파이프라인:
    점수 시퀀스 생성 → 시스템 프롬프트 구성 → LLM API 호출 → 응답 파싱 → 결과 저장

지원 모델:
    Anthropic : claude-haiku-4-5-20251001 (기본), claude-sonnet-4-6, claude-opus-4-6
    OpenAI    : gpt-4o-mini, gpt-4o, gpt-4-turbo
    Google    : gemini-2.0-flash, gemini-1.5-pro

환경변수 (사용하는 provider만 설정):
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY

CLI 예시:
    # 전체 2,520명 실험 (no-persona baseline)
    python src/experiment.py --provider anthropic

    # 처음 10명만 테스트
    python src/experiment.py --provider openai --model gpt-4o-mini --max_subjects 10

    # 도메인 페르소나 조건 (먼저 python src/download_personas.py 실행 필요)
    python src/experiment.py --provider anthropic --persona economics
    python src/experiment.py --provider anthropic --persona law
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import linear
from prompt_builder import (
    load_samples, get_subject,
    build_pretrain_prompt,
    ExperimentState, parse_llm_response,
)
from llm_client import LLMClient, PROVIDER_CONFIGS
from logger import ExperimentLogger
from personas import (
    PersonaLoader, PERSONA_CONDITIONS, DOMAIN_BIAS,
    build_system_prompt, validate_condition,
)


# ── 분포별 Optimal Threshold ──────────────────────────────────────────────────

def optimal_threshold(n_remaining: int, distribution: str) -> float:
    """
    남은 날 수와 분포에 따른 Feynman 최적 탐색 임계값.
    점수 스케일: 0~100 (평균 50) 기준.
    수식 출처: Christian et al. (2026), Eq. 1–4.
    """
    if n_remaining <= 0:
        return 0.0
    if distribution == "uniform":
        return np.sqrt(n_remaining) / (np.sqrt(n_remaining) + 1) * 100
    elif distribution == "triangular":
        if n_remaining == 1:
            raw = 2 / 3
        else:
            raw = (2 * np.sqrt(n_remaining / (n_remaining - 1))
                   * np.cos(np.pi / 3 + 1 / 3 * np.arcsin(1 / np.sqrt(n_remaining))))
        return raw * 100 / 0.75
    elif distribution == "exponential":
        if n_remaining == 1:
            return 50.0
        from scipy.special import lambertw
        return float(np.real(1 + lambertw((n_remaining - 1) / np.e))) * 50
    elif distribution == "power_law":
        return (np.sqrt(n_remaining) + 1) * 25
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ── 점수 시퀀스 생성 ──────────────────────────────────────────────────────────

def generate_score_sequence(distribution: str, total_nights: int,
                             clamp: int, rng: np.random.Generator) -> list:
    """
    분포와 클램핑 조건에 맞는 점수 시퀀스 생성.

    클램핑 적용 기간: 초반 floor(clamp * total_nights / 7)일.
    해당 기간 동안 새 레스토랑 점수 상한 = optimal_threshold.
    (원 논문과 동일한 방식; LLM에게는 점수만 전달되어 클램핑 사실 비공개)
    """
    scores_pool, probs_pool = linear.rating_pmfs[distribution]
    scores_pool = np.array(scores_pool)
    probs_pool  = np.array(probs_pool) / np.array(probs_pool).sum()

    clamp_nights = int(clamp * total_nights / 7)
    scores = []
    for night in range(total_nights):
        n_remaining = total_nights - night
        if night < clamp_nights:
            threshold = optimal_threshold(n_remaining, distribution)
            mask = scores_pool < threshold
            if mask.sum() == 0:
                mask = np.ones(len(scores_pool), dtype=bool)
            s_m = scores_pool[mask]
            p_m = probs_pool[mask] / probs_pool[mask].sum()
            scores.append(int(rng.choice(s_m, p=p_m)))
        else:
            scores.append(int(rng.choice(scores_pool, p=probs_pool)))
    return scores


# ── 응답 형식 ─────────────────────────────────────────────────────────────────
# LLM 응답 형식을 엄격히 제한하여 파싱 실패율을 최소화.
# 시스템 프롬프트와 매 요청 프롬프트 끝에 모두 명시.

RESPONSE_FORMAT = """
[Required Response Format]
To explore a new restaurant: EXPLORE (row,col)
  Example: EXPLORE (2,3)
To return to the best restaurant: EXPLOIT
  Example: EXPLOIT

Output ONLY one of the above. No other text, explanation, or punctuation.
""".strip()


def parse_structured_response(response: str, visited: dict,
                               best_position: tuple) -> dict:
    """
    EXPLORE (r,c) / EXPLOIT 형식 응답 파싱.
    형식 불일치 시 기존 유연 파서(parse_llm_response)로 fallback.

    Returns
    -------
    dict : {action: str, position: tuple|None, raw: str}
    """
    response = response.strip()

    # EXPLOIT
    if re.match(r"^EXPLOIT\b", response, re.IGNORECASE):
        return {"action": "exploit", "position": best_position, "raw": response}

    # EXPLORE (r,c)
    m = re.match(
        r"^EXPLORE\s*\(\s*([0-3])\s*,\s*([0-6])\s*\)",
        response, re.IGNORECASE
    )
    if m:
        pos = (int(m.group(1)), int(m.group(2)))
        if pos in visited:
            return {"action": "invalid", "position": pos, "raw": response}
        return {"action": "explore", "position": pos, "raw": response}

    # Fallback: 유연 파서
    return parse_llm_response(response, visited, best_position)


# ── 단일 참가자 실험 루프 ─────────────────────────────────────────────────────

def run_experiment(subject: dict,
                   client: LLMClient,
                   persona_condition: str = "none",
                   persona_text: str = None,
                   seed: int = 42,
                   max_retries: int = 3,
                   verbose: bool = True) -> pd.DataFrame:
    """
    단일 참가자에 대해 LLM 실험 수행.

    Parameters
    ----------
    subject          : get_subject()로 얻은 참가자 딕셔너리
    client           : LLMClient 인스턴스
    persona_condition: 페르소나 조건 키 ('none' 또는 도메인명)
    persona_text     : PersonaHub에서 샘플링한 페르소나 텍스트
                       (condition='none'이면 None)
    seed             : 랜덤 시드
    max_retries      : 응답 파싱 실패 시 재시도 횟수
    verbose          : 진행 상황 출력 여부
    """
    rng = np.random.default_rng(seed + subject["subject_id"])
    score_seq = generate_score_sequence(
        distribution=subject["distribution"],
        total_nights=subject["total_nights"],
        clamp=subject["clamp"],
        rng=rng,
    )

    if verbose:
        print(f"  [Subject {subject['subject_id']:>4}] "
              f"dist={subject['distribution']:<12} "
              f"T={subject['total_nights']:>2} "
              f"clamp={subject['clamp']} "
              f"persona={persona_condition:<18} "
              f"model={client.model_id}")

    state = ExperimentState(
        total_nights=subject["total_nights"],
        distribution=subject["distribution"],
        clamp=subject["clamp"],
        score_sequence=score_seq,
    )

    # 시스템 프롬프트: 페르소나 + 응답 형식 제약
    system_prompt = build_system_prompt(persona_condition, persona_text)
    system_prompt += f"\n\n{RESPONSE_FORMAT}"

    # 로거 초기화
    logger = ExperimentLogger(verbose=verbose)
    logger.log_system(system_prompt, subject["subject_id"], persona_condition)

    messages = []

    # 사전 학습 (원 논문과 동일한 84개 샘플 + 응답 형식 안내 추가)
    pretrain = build_pretrain_prompt(subject["samples"], subject["total_nights"])
    pretrain_with_format = pretrain + f"\n\n{RESPONSE_FORMAT}"
    messages.append({"role": "user",      "content": pretrain_with_format})
    messages.append({"role": "assistant", "content":
                     "Understood. I have reviewed the restaurant score distribution. "
                     "I will respond only with EXPLORE (row,col) or EXPLOIT."})

    prev_action = prev_score = prev_position = None

    while not state.is_done:
        night_prompt = state.build_night_prompt(prev_score, prev_action, prev_position)
        messages.append({"role": "user", "content": night_prompt})

        llm_response = None
        parsed = None

        # ── 입력 로깅 ────────────────────────────────────────────────────────
        logger.log_input(
            night_num   = state.current_night + 1,
            total_nights= state.total_nights,
            night_prompt= night_prompt,
            n_messages  = len(messages),
        )

        for attempt in range(max_retries):
            try:
                llm_response = client.chat(messages, system=system_prompt)
            except Exception as e:
                err_str = str(e)
                print(f"    [API 오류] {err_str} — 재시도 {attempt+1}/{max_retries}")
                # 지수 백오프: 429(rate limit)는 더 길게 대기
                if "429" in err_str:
                    wait = 30 * (attempt + 1)   # 30s, 60s, 90s
                    print(f"    Rate limit — {wait}초 대기 중...")
                elif "503" in err_str or "500" in err_str:
                    wait = 10 * (attempt + 1)   # 10s, 20s, 30s
                else:
                    wait = 3 * (attempt + 1)    # 3s, 6s, 9s
                time.sleep(wait)
                continue

            # ── 출력 로깅 ────────────────────────────────────────────────────
            logger.log_output(llm_response)

            parsed = parse_structured_response(
                llm_response, state.visited, state.best_position
            )

            if parsed["action"] in ("explore", "exploit"):
                break

            # 형식 오류 → 재시도 유도
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content":
                              "Invalid format. Respond ONLY with:\n"
                              "  EXPLORE (row,col)  — e.g. EXPLORE (1,3)\n"
                              "  EXPLOIT"})

        # 최종 Fallback: 파싱 완전 실패 → 첫 미방문 위치 자동 선택
        if parsed is None or parsed["action"] not in ("explore", "exploit"):
            fallback = next(
                (r, c) for r in range(4) for c in range(7)
                if (r, c) not in state.visited
            )
            parsed = {"action": "explore", "position": fallback, "raw": "[fallback]"}
            if verbose:
                print(f"    [Fallback] 파싱 실패 → {fallback}")

        # 행동 실행
        if parsed["action"] == "exploit":
            score = state.exploit()
            prev_action, prev_score, prev_position = "exploit", score, state.best_position
        else:
            score = state.explore(parsed["position"])
            prev_action, prev_score, prev_position = "explore", score, parsed["position"]

        messages.append({"role": "assistant", "content": llm_response or ""})

        # ── 파싱 및 저장 로깅 ────────────────────────────────────────────────
        logger.log_parsed(parsed, score=score)
        saved_row = state.history[-1]   # 방금 기록된 행
        logger.log_saved_row(saved_row)

        if verbose:
            print(f"    Night {state.current_night-1}: "
                  f"{prev_action:7} {str(prev_position):<8} → {score:>4}점  "
                  f"(최고:{state.best_score:>4} 남은:{state.nights_remaining}일)")

        # Night 간 대기
        time.sleep(client.request_interval)

    return state.to_dataframe(
        subject_id=subject["subject_id"],
        persona=persona_condition,
        llm_model=client.model_id,
    )


# ── 배치 실험 ─────────────────────────────────────────────────────────────────

def run_batch(samples_csv: str,
              provider: str = "anthropic",
              model: str = None,
              persona_condition: str = "none",
              personas_dir: str = "data/personas",
              output_csv: str = None,
              seed: int = 42,
              max_subjects: int = None,
              verbose: bool = True) -> pd.DataFrame:
    """
    전체(또는 일부) 참가자에 대해 실험 수행 후 결과를 CSV 저장.

    Parameters
    ----------
    samples_csv       : FeynmanStudySamplesObserved.csv 경로
    provider          : 'anthropic' | 'openai' | 'google'
    model             : 모델명 (None이면 provider 기본값)
    persona_condition : 'none' 또는 도메인명 (예: 'economics')
    personas_dir      : 페르소나 JSONL 파일 디렉토리
    output_csv        : 결과 저장 경로 (None이면 자동 생성)
    seed              : 랜덤 시드
    max_subjects      : 최대 실험 참가자 수
                        None → 전체 2,520명 (논문 실험 기본값)
                        정수 → 처음 N명만 (테스트 목적)
    verbose           : 진행 상황 출력 여부
    """
    persona_condition = validate_condition(persona_condition)
    client    = LLMClient(provider=provider, model=model)
    df_samples = load_samples(samples_csv)

    # Subject ID 목록 결정
    all_ids = sorted(df_samples["Subject"].tolist())
    if max_subjects is not None:
        all_ids = all_ids[:max_subjects]

    # 출력 경로 자동 생성
    if output_csv is None:
        model_tag = client.model.replace("-", "_").replace("/", "_")
        cond_tag  = persona_condition.replace(" ", "_")
        output_csv = f"results/{provider}_{model_tag}_{cond_tag}.csv"

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # 페르소나 로더 초기화 (none이면 불필요)
    persona_loader = None
    if persona_condition != "none":
        persona_loader = PersonaLoader(personas_dir=personas_dir, seed=seed)

    # 실험 요약 출력
    bias_tag = DOMAIN_BIAS.get(persona_condition, "—")
    print(f"\n{'='*65}")
    print(f"Feynman LLM Restaurant Experiment")
    print(f"  Provider    : {provider}")
    print(f"  Model       : {client.model_id}")
    print(f"  Persona     : {persona_condition}"
          + (f" (predicted bias: {bias_tag})" if persona_condition != "none" else " (baseline)"))
    print(f"  Subjects    : {len(all_ids)}명"
          + (" [전체]" if max_subjects is None else f" [max_subjects={max_subjects}]"))
    print(f"  Output      : {output_csv}")
    print(f"{'='*65}\n")

    all_results = []

    for idx, sid in enumerate(all_ids):
        try:
            subject = get_subject(df_samples, sid)

            # 페르소나 텍스트 샘플링: subject_id 기반 결정론적 배정
            persona_text = None
            if persona_loader is not None:
                persona_text = persona_loader.sample(persona_condition,
                                                     subject_id=sid)

            df_result = run_experiment(
                subject=subject,
                client=client,
                persona_condition=persona_condition,
                persona_text=persona_text,
                seed=seed,
                verbose=verbose,
            )
            all_results.append(df_result)

            # Subject 종료 요약 로깅
            if verbose:
                from logger import ExperimentLogger
                ExperimentLogger(verbose=True).log_session_summary(
                    subject_id=sid,
                    df_result=df_result,
                    output_csv=output_csv,
                )

            # 10명마다 중간 저장
            if (idx + 1) % 10 == 0 or (idx + 1) == len(all_ids):
                pd.concat(all_results, ignore_index=True).to_csv(output_csv)
                print(f"  → [{idx+1}/{len(all_ids)}] 저장: {output_csv}")

        except Exception as e:
            print(f"  [오류] Subject {sid}: {e}")
            continue

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv)
        print(f"\n[완료] {len(all_results)}명 결과 저장: {output_csv}")
        return final_df

    print("[경고] 수집된 결과 없음")
    return pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feynman LLM Restaurant Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 2,520명 실험 — no-persona baseline (Anthropic)
  python src/experiment.py --provider anthropic

  # 처음 10명만 테스트
  python src/experiment.py --provider anthropic --max_subjects 10

  # 도메인 페르소나 (먼저: python src/download_personas.py)
  python src/experiment.py --provider anthropic --persona law
  python src/experiment.py --provider openai   --persona philosophy

  # 다른 모델 지정
  python src/experiment.py --provider openai  --model gpt-4o
  python src/experiment.py --provider google  --model gemini-1.5-pro

지원 모델:
""" + "\n".join(
            f"  {p}: {', '.join(c['models'])}"
            for p, c in PROVIDER_CONFIGS.items()
        ) + f"""

페르소나 조건:
  {PERSONA_CONDITIONS}
"""
    )
    parser.add_argument("--samples_csv",   default="data/FeynmanStudySamplesObserved.csv",
                        help="샘플 데이터 CSV 경로")
    parser.add_argument("--provider",      default="anthropic",
                        choices=list(PROVIDER_CONFIGS.keys()),
                        help="LLM 제공자")
    parser.add_argument("--model",         default=None,
                        help="모델명 (생략 시 provider 기본값)")
    parser.add_argument("--persona",       default="none",
                        choices=PERSONA_CONDITIONS,
                        help="페르소나 조건 ('none' = baseline)")
    parser.add_argument("--personas_dir",  default="data/personas",
                        help="페르소나 JSONL 파일 디렉토리")
    parser.add_argument("--output",        default=None,
                        help="결과 CSV 경로 (생략 시 자동 생성)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--max_subjects",  type=int, default=None,
                        help="최대 실험 참가자 수 (테스트용; 생략 시 전체 2,520명)")
    parser.add_argument("--quiet",         action="store_true",
                        help="진행 상황 출력 억제")
    args = parser.parse_args()

    run_batch(
        samples_csv       = args.samples_csv,
        provider          = args.provider,
        model             = args.model,
        persona_condition = args.persona,
        personas_dir      = args.personas_dir,
        output_csv        = args.output,
        seed              = args.seed,
        max_subjects      = args.max_subjects,
        verbose           = not args.quiet,
    )