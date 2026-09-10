"""
analyze.py
----------
실험 결과 분석 및 Figure 생성.

원 논문(Christian et al., 2026) Figure 4 구조를 기반으로,
Feynman 최적해 / 인간 결과 / LLM 결과를 한눈에 비교하는 Figure를 생성.

파이프라인:
    1. LLM 실험 결과 CSV 로드 → Subject별 총점 집계
    2. Feynman 최적해 (사전 계산값) 로드
    3. 인간 결과 (원 논문 FeynmanStudyData.csv) 로드 — 선택적
    4. 분포 × 야간 수별 비교 Figure 생성

사용법:
    # LLM 결과만 (인간 데이터 없이)
    python src/analyze.py --llm_csv results/google_gemini_3.6_flash_none.csv

    # 인간 데이터와 함께 비교
    python src/analyze.py \\
        --llm_csv results/google_gemini_3.6_flash_none.csv \\
        --human_csv data/FeynmanStudyData.csv

    # 여러 모델 비교
    python src/analyze.py \\
        --llm_csv results/google_gemini_3.6_flash_none.csv \\
                  results/anthropic_claude_haiku_none.csv \\
        --human_csv data/FeynmanStudyData.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import linear

# ── 원 논문 사전 계산값 (Feynman 최적해) ─────────────────────────────────────
# 출처: Christian et al. (2026), percentage_of_optimal.py
# 동적 프로그래밍으로 계산한 해석적 최적 총점 기댓값

OPTIMAL_EV = {
    # (distribution, nights): optimal expected total score
    ("triangular",  7):  428.892254,
    ("triangular",  14): 907.034961,
    ("triangular",  28): 1892.252273,
    ("uniform",     7):  492.140392,
    ("uniform",     14): 1084.224383,
    ("uniform",     28): 2333.711477,
    ("exponential", 7):  603.359468,
    ("exponential", 14): 1483.081237,
    ("exponential", 28): 3590.168780,
    ("power_law",   7):  551.928201,
    ("power_law",   14): 1408.868220,
    ("power_law",   28): 3701.943225,
}

DISTRIBUTIONS  = ["exponential", "power_law", "uniform", "triangular"]
TOTAL_NIGHTS   = [7, 14, 28]

DIST_COLORS = {
    "exponential": (174/255, 205/255, 225/255),
    "power_law":   (84/255,  158/255, 63/255),
    "uniform":     (243/255, 193/255, 123/255),
    "triangular":  (100/255, 63/255,  149/255),
}

DIST_LABELS = {
    "exponential": "Exponential",
    "power_law":   "Power Law",
    "uniform":     "Uniform",
    "triangular":  "Triangular",
}


# ── Step 1: LLM 결과 집계 ────────────────────────────────────────────────────

def load_llm_results(csv_path: str) -> pd.DataFrame:
    """
    LLM 실험 결과 CSV를 로드하여 Subject별 총점을 집계.

    원 논문과 동일하게:
    - clamp == 0 조건만 사용
    - Subject별 Reward 합산 → total_score
    - 분포 × 야간 수별 mean ± 95% CI 계산

    Returns: summary DataFrame with columns
        distribution, nights, mean, moe, n, llm_model
    """
    df = pd.read_csv(csv_path, index_col=0)

    # clamp=0 필터 (원 논문과 동일한 기준)
    if "clamp" in df.columns:
        df = df[df["clamp"] == 0]

    # LLM 모델명 추출 (첫 번째 값 사용)
    llm_model = df["LLM Model"].iloc[0] if "LLM Model" in df.columns else "LLM"

    # Subject별 총점 집계
    subject_totals = (
        df.groupby(["Subject", "Distribution", "Total Nights", "clamp"])["Reward"]
        .sum()
        .reset_index()
        .rename(columns={
            "Reward":       "total_score",
            "Distribution": "distribution",
            "Total Nights": "nights",
        })
    )
    subject_totals["distribution"] = subject_totals["distribution"].str.lower()

    # 분포 × 야간 수별 통계
    summary_rows = []
    for dist in DISTRIBUTIONS:
        for nights in TOTAL_NIGHTS:
            subset = subject_totals[
                (subject_totals["distribution"] == dist) &
                (subject_totals["nights"] == nights)
            ]
            if len(subset) == 0:
                continue
            n         = len(subset)
            mean      = subset["total_score"].mean()
            sem       = subset["total_score"].std() / np.sqrt(n)
            moe       = 1.96 * sem
            summary_rows.append({
                "distribution": dist,
                "nights":       nights,
                "mean":         mean,
                "moe":          moe,
                "n":            n,
                "llm_model":    llm_model,
            })

    return pd.DataFrame(summary_rows)


# ── Step 2: 인간 결과 집계 ────────────────────────────────────────────────────

def load_human_results(csv_path: str,
                       exclude_mistakes: bool = False) -> pd.DataFrame:
    """
    원 논문 FeynmanStudyData.csv를 로드하여 인간 참가자 총점 집계.
    원 논문(percentage_of_optimal.py)과 동일한 방식:
      - clamp==0 조건만 사용
      - exclude_mistakes=True이면 Mistake가 있는 참가자 제외

    Parameters
    ----------
    csv_path         : FeynmanStudyData.csv 경로
    exclude_mistakes : True이면 num_mistakes > 0인 Subject 제외
                       원 논문 기본값은 False (All Subs Included Mistakes Removed)
                       → Mistake 행은 제거하되 참가자는 유지
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Mistake 행 제거 (원 논문: 행 자체는 분석에서 제외)
    df_clean = df[df["Action"] != "Mistake"].copy()

    # clamp=0만 사용
    df_clean = df_clean[df_clean["clamp"] == 0]

    # Subject별 총점 및 mistake 수 집계
    def agg(group):
        import pandas as _pd
        all_actions = df[df.index.isin(group.index)]["Action"].tolist()
        return _pd.Series({
            "total_score":  group["Reward"].sum(),
            "num_mistakes": all_actions.count("Mistake"),
            "distribution": group["Distribution"].iloc[0],
            "nights":       group["Total Nights"].iloc[0],
        })

    subject_totals = (
        df_clean.groupby(["Subject"])
        .apply(lambda g: pd.Series({
            "total_score":  g["Reward"].sum(),
            "distribution": g["Distribution"].iloc[0].lower(),
            "nights":       int(g["Total Nights"].iloc[0]),
        }), include_groups=False)
        .reset_index()
    )

    # Mistake 제외 옵션 (원 논문의 EXCLUDE_PARTICIPANTS에 해당)
    if exclude_mistakes:
        mistake_counts = (
            df[df["clamp"] == 0]
            .groupby("Subject")["Action"]
            .apply(lambda x: (x == "Mistake").sum())
        )
        valid_subjects = mistake_counts[mistake_counts == 0].index
        subject_totals = subject_totals[
            subject_totals["Subject"].isin(valid_subjects)
        ]

    # 분포 × 야간 수별 통계
    summary_rows = []
    for dist in DISTRIBUTIONS:
        for nights in TOTAL_NIGHTS:
            subset = subject_totals[
                (subject_totals["distribution"] == dist) &
                (subject_totals["nights"] == nights)
            ]
            if len(subset) == 0:
                continue
            n    = len(subset)
            mean = subset["total_score"].mean()
            moe  = 1.96 * subset["total_score"].std() / np.sqrt(n)
            summary_rows.append({
                "distribution": dist,
                "nights":       nights,
                "mean":         mean,
                "moe":          moe,
                "n":            n,
            })
    return pd.DataFrame(summary_rows)


# ── Step 3: Figure 생성 ───────────────────────────────────────────────────────

def plot_comparison(llm_summaries: list,
                    human_summary: pd.DataFrame = None,
                    output_path: str = "figures/comparison.pdf",
                    show: bool = True):
    """
    Feynman 최적해 / 인간 / LLM 결과 비교 Figure.

    레이아웃: 2행 × 2열 (분포별 subplot)
    각 subplot x축: 야간 수 (7, 14, 28)
    각 subplot y축: 총점 (Total Score)

    막대 구성:
        [Feynman Optimal] [Human ± CI] [LLM1 ± CI] [LLM2 ± CI] ...
    """
    # matplotlib 스타일 (원 논문 스타일 참고)
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.labelsize":   9,
        "axes.titlesize":   11,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "axes.linewidth":   0.8,
        "lines.linewidth":  1.0,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "Performance Comparison: Feynman Optimal vs. Human vs. LLM",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    n_llm   = len(llm_summaries)
    has_human = human_summary is not None and len(human_summary) > 0
    n_bars  = 1 + (1 if has_human else 0) + n_llm   # Optimal + Human + LLM들

    bar_w   = 0.6 / n_bars   # 전체 너비 0.6을 bar 수로 나눔
    x_base  = np.arange(len(TOTAL_NIGHTS))

    for ax_idx, dist in enumerate(DISTRIBUTIONS):
        ax    = axes[ax_idx // 2, ax_idx % 2]
        color = DIST_COLORS[dist]

        offsets = np.linspace(
            -(n_bars - 1) / 2 * bar_w,
             (n_bars - 1) / 2 * bar_w,
            n_bars
        )
        bar_idx = 0

        # ① Feynman 최적해
        optimal_vals = [OPTIMAL_EV.get((dist, t), np.nan) for t in TOTAL_NIGHTS]
        ax.bar(
            x_base + offsets[bar_idx], optimal_vals,
            width=bar_w, label="Feynman Optimal",
            color=color, edgecolor="black", linewidth=0.8,
        )
        bar_idx += 1

        # ② 인간 결과 (있는 경우)
        if has_human:
            human_dist = human_summary[human_summary["distribution"] == dist]
            human_vals = []
            human_moes = []
            for t in TOTAL_NIGHTS:
                row = human_dist[human_dist["nights"] == t]
                human_vals.append(row["mean"].values[0] if len(row) else np.nan)
                human_moes.append(row["moe"].values[0]  if len(row) else 0)
            ax.bar(
                x_base + offsets[bar_idx], human_vals,
                width=bar_w, label="Human (95% CI)",
                color=color, alpha=0.6,
                edgecolor="black", linewidth=0.8,
                hatch="///",
                yerr=human_moes, capsize=3, error_kw={"linewidth": 0.8},
            )
            bar_idx += 1

        # ③ LLM 결과 (모델별)
        llm_hatches = ["...", "xxx", "---", "+++"]
        llm_alphas  = [0.85, 0.70, 0.55, 0.40]
        for li, (llm_sum, model_label) in enumerate(llm_summaries):
            llm_dist = llm_sum[llm_sum["distribution"] == dist]
            llm_vals = []
            llm_moes = []
            for t in TOTAL_NIGHTS:
                row = llm_dist[llm_dist["nights"] == t]
                llm_vals.append(row["mean"].values[0] if len(row) else np.nan)
                llm_moes.append(row["moe"].values[0]  if len(row) else 0)
            ax.bar(
                x_base + offsets[bar_idx], llm_vals,
                width=bar_w,
                label=model_label,
                color=color, alpha=llm_alphas[li % len(llm_alphas)],
                edgecolor="black", linewidth=0.8,
                hatch=llm_hatches[li % len(llm_hatches)],
                yerr=llm_moes, capsize=3, error_kw={"linewidth": 0.8},
            )
            bar_idx += 1

        ax.set_title(DIST_LABELS[dist], fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{t} Nights" for t in TOTAL_NIGHTS])
        ax.set_ylabel("Total Score")
        ax.set_xlabel("Total Nights")
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(loc="upper left", framealpha=0.9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 저장: {output_path}")

    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feynman LLM 실험 결과 분석 및 Figure 생성"
    )
    parser.add_argument(
        "--llm_csv", nargs="+", required=True,
        help="LLM 실험 결과 CSV (여러 개 가능)"
    )
    parser.add_argument(
        "--human_csv", default=None,
        help="원 논문 인간 데이터 CSV (FeynmanStudyData.csv). "
             "OSF: https://osf.io/download/q62ha/"
    )
    parser.add_argument(
        "--exclude_mistakes", action="store_true",
        help="Mistake가 있는 인간 참가자 제외 (원 논문 EXCLUDE_PARTICIPANTS=True에 해당)"
    )
    parser.add_argument(
        "--output", default="figures/comparison.pdf",
        help="Figure 저장 경로"
    )
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    # LLM 결과 로드
    llm_summaries = []
    for csv_path in args.llm_csv:
        print(f"LLM 결과 로드: {csv_path}")
        summary = load_llm_results(csv_path)
        # 모델 이름을 레이블로 사용
        model_label = summary["llm_model"].iloc[0] if len(summary) > 0 else csv_path
        # 레이블 간소화 (예: "google/gemini-3.6-flash" → "Gemini 3.6 Flash")
        label = (model_label
                 .replace("google/gemini-", "Gemini ")
                 .replace("anthropic/claude-", "Claude ")
                 .replace("groq/", "")
                 .replace("-", " ")
                 .title())
        llm_summaries.append((summary, f"LLM: {label}"))
        print(f"  → {len(summary)}개 조건, 모델: {model_label}")

    # 인간 결과 로드
    human_summary = None
    if args.human_csv:
        print(f"인간 데이터 로드: {args.human_csv}")
        human_summary = load_human_results(
            args.human_csv,
            exclude_mistakes=args.exclude_mistakes,
        )
        print(f"  → {len(human_summary)}개 조건"
              + (" (Mistake 참가자 제외)" if args.exclude_mistakes else ""))

    # Figure 생성
    plot_comparison(
        llm_summaries=llm_summaries,
        human_summary=human_summary,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()