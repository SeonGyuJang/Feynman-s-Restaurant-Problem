"""
experiment_gateway.py
----------------------
Feynman Restaurant Experiment — 고려대학교 세종캠퍼스 API Gateway 실험 러너.

기존 experiment.py / llm_client.py는 전혀 수정하지 않는다. 배치 실행 로직은
external_runner.py의 run_batch_with_client()를 재사용하고(내부적으로
experiment.py의 run_experiment()를 그대로 호출), client만 GatewayClient
(gateway_client.py)로 교체해 Gateway를 통해 모델(예: OpenAI gpt-5.5)을 호출한다.

문서: https://docs.factchat.kr/docs/korea-sejong/api-gateway/getting-started/overview

사전 준비:
    1. 고려대학교 세종캠퍼스 웹사이트 로그인 → API Gateway → 키 생성
       → 환경변수 GATEWAY_API_KEY
    2. 사용할 모델명 확인 (예: 'gpt-5.5') → 환경변수 GATEWAY_MODEL 또는 --model 인자
       정확한 모델명이 게이트웨이에 등록되어 있는지 확실하지 않다면 먼저
       --list_models로 조회해서 확인할 것.

CLI 예시:
    # 사용 가능 모델 목록 조회 (실험 없이 조회만 하고 종료)
    python src/experiment_gateway.py --list_models

    # 연결 확인 겸 1명만 테스트 (gpt-5.5)
    python src/experiment_gateway.py --model gpt-5.5 --max_subjects 1

    # 도메인 페르소나 (먼저: python src/download_personas.py)
    python src/experiment_gateway.py --model gpt-5.5 --persona law --max_subjects 20

    # 중단된 실험 이어서 재개
    python src/experiment_gateway.py --model gpt-5.5 --resume

Python API:
    from src.experiment_gateway import run_batch_gateway

    run_batch_gateway(
        samples_csv       = "data/FeynmanStudySamplesObserved.csv",
        model              = "gpt-5.5",
        persona_condition = "none",
        max_subjects      = 10,
    )
"""

import os
import sys
import json
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수 직접 설정 필요

sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd

from personas import PERSONA_CONDITIONS
from gateway_client import GatewayClient, list_models
from external_runner import run_batch_with_client


def run_batch_gateway(samples_csv: str,
                       model: str = None,
                       api_key: str = None,
                       persona_condition: str = "none",
                       personas_dir: str = "data/personas",
                       output_csv: str = None,
                       seed: int = 42,
                       max_subjects: int = None,
                       request_interval: float = 1.0,
                       resume: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """
    API Gateway로 호출한 모델(예: OpenAI gpt-5.5)을 대상으로 전체(또는 일부)
    참가자 실험 수행. 파라미터/동작은 experiment.run_batch()와 동일하며,
    provider 대신 model(Gateway에 등록된 모델명)로 실험 대상을 지정한다.
    """
    client = GatewayClient(model=model, api_key=api_key,
                            request_interval=request_interval)
    return run_batch_with_client(
        client=client,
        label="API Gateway (고려대 세종캠퍼스)",
        samples_csv=samples_csv,
        persona_condition=persona_condition,
        personas_dir=personas_dir,
        output_csv=output_csv,
        output_prefix="gateway",
        seed=seed,
        max_subjects=max_subjects,
        resume=resume,
        verbose=verbose,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feynman LLM Restaurant Experiment — API Gateway Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 사용 가능 모델 목록 조회 (실험 없이 조회만 하고 종료)
  python src/experiment_gateway.py --list_models

  # 연결 확인 겸 1명만 테스트
  python src/experiment_gateway.py --model gpt-5.5 --max_subjects 1

  # 도메인 페르소나 (먼저: python src/download_personas.py)
  python src/experiment_gateway.py --model gpt-5.5 --persona law --max_subjects 20

  # 중단된 실험 이어서 재개 (--resume)
  python src/experiment_gateway.py --model gpt-5.5 --resume

환경변수:
  GATEWAY_API_KEY   API Gateway 키 (필수)
  GATEWAY_MODEL     모델명 (--model 생략 시 사용; 그마저 없으면 'gpt-5.5')

페르소나 조건:
  """ + str(PERSONA_CONDITIONS)
    )
    parser.add_argument("--samples_csv",   default="data/FeynmanStudySamplesObserved.csv",
                        help="샘플 데이터 CSV 경로")
    parser.add_argument("--model",         default=None,
                        help="Gateway에 등록된 모델명 (예: gpt-5.5). "
                             "생략 시 환경변수 GATEWAY_MODEL, 그마저 없으면 'gpt-5.5'")
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
    parser.add_argument("--request_interval", type=float, default=1.0,
                        help="Night 간 대기 시간(초)")
    parser.add_argument("--resume",        action="store_true",
                        help="중단된 실험 이어서 재개.")
    parser.add_argument("--quiet",         action="store_true",
                        help="진행 상황 출력 억제")
    parser.add_argument("--list_models",   action="store_true",
                        help="GET /v1/gateway/models/ 로 사용 가능 모델 목록만 조회하고 종료")
    args = parser.parse_args()

    if args.list_models:
        models = list_models()
        print(json.dumps(models, indent=2, ensure_ascii=False))
        sys.exit(0)

    run_batch_gateway(
        samples_csv       = args.samples_csv,
        model              = args.model,
        persona_condition = args.persona,
        personas_dir      = args.personas_dir,
        output_csv        = args.output,
        seed              = args.seed,
        max_subjects      = args.max_subjects,
        request_interval  = args.request_interval,
        resume            = args.resume,
        verbose           = not args.quiet,
    )
