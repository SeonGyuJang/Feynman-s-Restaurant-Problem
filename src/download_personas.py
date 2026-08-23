"""
download_personas.py
--------------------
PersonaHub elite_persona 데이터셋에서 도메인별 페르소나를 수집하여
data/personas/{domain}.jsonl 형식으로 저장.

PersonaHub (proj-persona/PersonaHub) 는 HuggingFace에 공개된
대규모 페르소나 데이터셋으로, elite_persona 서브셋은 각 도메인 상위
1% 수준의 전문가 페르소나 텍스트를 포함합니다.

참고: Chan et al. (2024). PersonaHub: Personalized Data Creation for
      LLM Alignment. https://arxiv.org/abs/2406.20094

사용법:
    pip install datasets tqdm
    python src/download_personas.py

    # 도메인당 1,000개 수집 (권장: 실험당 2,520명 × 여러 조건)
    python src/download_personas.py --quota 1000

    # 이어받기: 이미 일부 수집된 경우 중복 없이 추가
    python src/download_personas.py --quota 1000 --resume

출력 구조:
    data/personas/
    ├── economics.jsonl        # 한 줄에 하나의 페르소나
    ├── law.jsonl
    ├── philosophy.jsonl
    └── ...

권장 quota:
    - 빠른 테스트  : 100개/도메인
    - 단일 실험    : 500개/도메인  (2,520명 × 1회 실험 충분)
    - 다중 실험    : 1,000개/도메인 (여러 모델·조건 교차 실험)
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────────────

DATASET_NAME    = "proj-persona/PersonaHub"
SUBSET_NAME     = "elite_persona"
DOMAIN_KEY      = "general domain (top 1 percent)"

# personas.py의 PERSONA_DOMAINS와 반드시 동기화
TARGET_DOMAINS = [
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


# ── 현재 수집 현황 확인 ───────────────────────────────────────────────────────

def count_existing(output_dir: Path) -> dict:
    """이미 수집된 페르소나 수를 도메인별로 반환"""
    counts = {}
    for domain in TARGET_DOMAINS:
        filepath = output_dir / f"{domain}.jsonl"
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                counts[domain] = sum(1 for line in f if line.strip())
        else:
            counts[domain] = 0
    return counts


def print_status(counts: dict, quota: int):
    """수집 현황 출력"""
    total = sum(counts.values())
    total_needed = len(TARGET_DOMAINS) * quota
    logger.info("─" * 45)
    for domain in TARGET_DOMAINS:
        count  = counts.get(domain, 0)
        bar    = "█" * int(count / quota * 20) if quota > 0 else ""
        status = "✓" if count >= quota else " "
        logger.info(f"  [{status}] {domain:<22} {count:>6,}/{quota:,}  {bar}")
    logger.info("─" * 45)
    logger.info(f"  총 수집: {total:,} / {total_needed:,}")
    logger.info("─" * 45)


# ── 수집 함수 ─────────────────────────────────────────────────────────────────

def download(output_dir: Path, quota: int, seed: int = 42,
             shuffle_buffer: int = 10_000, resume: bool = True):
    """
    PersonaHub에서 도메인별 페르소나를 스트리밍으로 수집하여 JSONL 저장.

    Parameters
    ----------
    output_dir     : 저장 디렉토리
    quota          : 도메인당 목표 페르소나 수
    seed           : 셔플 시드 (재현성)
    shuffle_buffer : 셔플 버퍼 크기 (클수록 무작위성↑, 속도↓)
    resume         : True이면 기존 파일에 이어서 수집 (중복 방지)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets 패키지가 필요합니다.\n"
            "설치: pip install datasets"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 현재 수집 현황 확인
    existing = count_existing(output_dir) if resume else {d: 0 for d in TARGET_DOMAINS}
    remaining = {d: max(0, quota - existing[d]) for d in TARGET_DOMAINS}

    logger.info("=" * 45)
    logger.info("PersonaHub Elite Persona 수집")
    logger.info("=" * 45)
    logger.info(f"  Dataset  : {DATASET_NAME} ({SUBSET_NAME})")
    logger.info(f"  Quota    : {quota}개/도메인")
    logger.info(f"  이어받기  : {'예' if resume else '아니오'}")
    logger.info("")
    logger.info("현재 수집 현황:")
    print_status(existing, quota)

    # 이미 모든 도메인이 채워진 경우
    if all(r == 0 for r in remaining.values()):
        logger.info("모든 도메인의 목표 수량이 이미 달성되어 있습니다.")
        return existing

    total_remaining = sum(remaining.values())
    logger.info(f"\n추가 수집 필요: {total_remaining}개")
    logger.info("스트리밍 로드 중 (첫 행 도달까지 잠시 대기)...\n")

    domain_counts = dict(existing)  # 기존 수 포함

    dataset = load_dataset(
        DATASET_NAME,
        name=SUBSET_NAME,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    collected_this_run = 0
    rows_scanned = 0
    log_interval = 10_000

    for row in dataset:
        # 모든 도메인 완료 시 종료
        if all(domain_counts[d] >= quota for d in TARGET_DOMAINS):
            break

        rows_scanned += 1
        if rows_scanned % log_interval == 0:
            done = sum(1 for d in TARGET_DOMAINS if domain_counts[d] >= quota)
            logger.info(
                f"  스캔 {rows_scanned:>8,}행 | "
                f"이번 수집 {collected_this_run:>5,}개 | "
                f"완료 도메인 {done}/{len(TARGET_DOMAINS)}"
            )

        raw_domain = row.get(DOMAIN_KEY, "").lower().strip()
        matched    = next((d for d in TARGET_DOMAINS if d in raw_domain), None)

        if matched is None:
            continue
        if domain_counts[matched] >= quota:
            continue

        persona_text = row.get("persona", "").strip()
        if not persona_text:
            continue

        # JSONL에 저장
        filepath = output_dir / f"{matched}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            json.dump(
                {"domain": matched, "persona": persona_text},
                f, ensure_ascii=False
            )
            f.write("\n")

        domain_counts[matched] += 1
        collected_this_run += 1

    # 최종 리포트
    logger.info("\n" + "=" * 45)
    logger.info("수집 완료")
    logger.info("=" * 45)
    print_status(domain_counts, quota)
    logger.info(f"  이번 실행 수집: {collected_this_run:,}개")

    incomplete = [d for d in TARGET_DOMAINS if domain_counts[d] < quota]
    if incomplete:
        logger.warning(f"\n  미달 도메인: {incomplete}")
        logger.warning("  --resume 옵션으로 재실행하면 이어서 수집할 수 있습니다.")
    else:
        logger.info("\n  모든 도메인 목표 달성!")

    return domain_counts


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PersonaHub elite_persona 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
권장 quota 기준:
  --quota 100    빠른 테스트용
  --quota 500    단일 실험 (2,520명 × 1조건)
  --quota 1000   다중 실험 (여러 모델·조건 교차)

예시:
  python src/download_personas.py --quota 1000
  python src/download_personas.py --quota 1000 --resume   # 이어받기
"""
    )
    parser.add_argument("--quota",          type=int, default=1000,
                        help="도메인당 수집할 페르소나 수 (기본: 1000)")
    parser.add_argument("--output_dir",     default="data/personas")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000,
                        help="셔플 버퍼 크기 (기본: 10000)")
    parser.add_argument("--resume",         action="store_true", default=True,
                        help="기존 파일에 이어서 수집 (기본: True)")
    parser.add_argument("--no_resume",      action="store_true",
                        help="처음부터 다시 수집 (기존 파일 덮어쓰지 않고 추가)")
    args = parser.parse_args()

    download(
        output_dir     = Path(args.output_dir),
        quota          = args.quota,
        seed           = args.seed,
        shuffle_buffer = args.shuffle_buffer,
        resume         = not args.no_resume,
    )