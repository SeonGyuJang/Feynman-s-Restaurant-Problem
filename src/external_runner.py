"""
external_runner.py
-------------------
experiment.py의 run_batch()와 동일한 배치 실행 로직을, provider별 client 생성
로직과 분리해 재사용하기 위한 공통 헬퍼.

experiment.py / llm_client.py는 여기서 전혀 수정하지 않는다. 이미 만들어진
client 인스턴스(LLMClient와 동일하게 chat(messages, system) / model_id /
request_interval 인터페이스를 구현한 객체라면 무엇이든)를 받아
experiment.py의 run_experiment()를 그대로 반복 호출한다.

사용처:
    - gateway_client.py + experiment_gateway.py       (고려대 세종캠퍼스 API Gateway)
    - openrouter_client.py + experiment_openrouter.py (OpenRouter)
"""

import os
from pathlib import Path

import pandas as pd

from experiment import run_experiment          # 기존 코드 재사용 (수정 없음)
from prompt_builder import load_samples, get_subject
from personas import PersonaLoader, DOMAIN_BIAS, validate_condition


def run_batch_with_client(client,
                           label: str,
                           samples_csv: str,
                           persona_condition: str = "none",
                           personas_dir: str = "data/personas",
                           output_csv: str = None,
                           output_prefix: str = "external",
                           seed: int = 42,
                           max_subjects: int = None,
                           resume: bool = False,
                           verbose: bool = True) -> pd.DataFrame:
    """
    이미 생성된 client로 전체(또는 일부) 참가자 실험을 수행하고 CSV로 저장한다.
    experiment.run_batch()와 동일한 구조(자동 output 경로, resume, 중간 저장,
    403류 오류 시 즉시 중단)를 공유한다.

    Parameters
    ----------
    client         : chat(messages, system) / model_id / request_interval 을
                      갖춘 클라이언트 인스턴스. last_usage / last_credits
                      속성이 있으면(옵션) 진행 로그에 함께 출력한다.
    label          : 진행 상황 출력 헤더에 쓰이는 라벨 (예: 'API Gateway', 'OpenRouter')
    output_prefix  : output_csv 자동 생성 시 파일명 접두사 (예: 'gateway', 'openrouter')
    나머지 파라미터는 experiment.run_batch()와 동일.
    """
    persona_condition = validate_condition(persona_condition)
    df_samples = load_samples(samples_csv)
    all_ids_full = sorted(df_samples["Subject"].tolist())

    if output_csv is None:
        model_tag = (client.model_id.split("/", 1)[-1]
                     .replace("/", "_").replace(":", "_")
                     .replace(".", "_").replace("-", "_"))
        cond_tag  = persona_condition.replace(" ", "_")
        output_csv = f"results/{output_prefix}_{model_tag}_{cond_tag}.csv"

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # ── Resume: CSV에서 완료된 Subject 확인 ──────────────────────────────────
    completed_ids = set()
    existing_results = []
    if resume and Path(output_csv).exists():
        try:
            existing_df = pd.read_csv(output_csv, index_col=0)
            if len(existing_df) > 0 and "Subject" in existing_df.columns:
                completed_ids = set(existing_df["Subject"].unique().tolist())
                existing_results = [existing_df]
                print(f"[Resume] 기존 결과 로드: {output_csv} → 완료 {len(completed_ids)}명")
        except Exception as e:
            print(f"[Resume] 기존 파일 로드 실패 ({e}) — 처음부터 시작합니다.")

    target_ids = all_ids_full[:max_subjects] if max_subjects is not None else all_ids_full
    remaining_ids = [sid for sid in target_ids if sid not in completed_ids]

    persona_loader = None
    if persona_condition != "none":
        persona_loader = PersonaLoader(personas_dir=personas_dir, seed=seed)

    bias_tag = DOMAIN_BIAS.get(persona_condition, "—")
    print(f"\n{'='*65}")
    print(f"Feynman LLM Restaurant Experiment — {label}")
    print(f"  Model       : {client.model_id}")
    print(f"  Persona     : {persona_condition}"
          + (f" (predicted bias: {bias_tag})" if persona_condition != "none" else " (baseline)"))
    print(f"  Target      : Subject {target_ids[0]}~{target_ids[-1]} "
          f"({len(target_ids)}명)"
          + (" [전체]" if max_subjects is None else f" [max_subjects={max_subjects}]"))
    if resume and completed_ids:
        print(f"  완료        : {len(completed_ids)}명 → 실행 예정: {len(remaining_ids)}명")
    print(f"  Output      : {output_csv}")
    print(f"{'='*65}\n")

    if resume and len(remaining_ids) == 0:
        print("[완료] 대상 Subject가 모두 완료되었습니다.")
        return pd.read_csv(output_csv, index_col=0)

    all_results = list(existing_results)

    for idx, sid in enumerate(remaining_ids):
        try:
            subject = get_subject(df_samples, sid)

            persona_text = None
            if persona_loader is not None:
                persona_text = persona_loader.sample(persona_condition, subject_id=sid)

            df_result = run_experiment(
                subject=subject,
                client=client,
                persona_condition=persona_condition,
                persona_text=persona_text,
                seed=seed,
                verbose=verbose,
            )
            all_results.append(df_result)

            if verbose:
                usage   = getattr(client, "last_usage", None)
                credits = getattr(client, "last_credits", None)
                if usage or credits is not None:
                    print(f"    [{label}] usage(마지막 응답 기준): {usage}"
                          + (f" credits: {credits}" if credits is not None else ""))

            pd.concat(all_results, ignore_index=True).to_csv(output_csv)
            done_total = len(completed_ids) + idx + 1
            print(f"  → [{done_total}/{len(target_ids)}] Subject {sid} 저장 완료")

        except RuntimeError as e:
            # 재시도 불가 오류(예: 403) → 지금까지 결과 저장 후 전체 중단
            print(f"\n[실험 중단] {e}")
            if all_results:
                pd.concat(all_results, ignore_index=True).to_csv(output_csv)
                print(f"  지금까지의 결과 저장 완료: {output_csv}")
                print(f"  재개 방법: --resume 옵션으로 다시 실행")
            raise
        except Exception as e:
            print(f"  [오류] Subject {sid}: {e}")
            continue

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv)
        print(f"\n[완료] 총 {len(final_df['Subject'].unique())}명 결과 저장: {output_csv}")
        return final_df

    print("[경고] 수집된 결과 없음")
    return pd.DataFrame()
