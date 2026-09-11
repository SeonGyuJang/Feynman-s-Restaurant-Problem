"""
threshold_analysis.py
----------------------
원 논문(Christian et al., 2026)의 핵심 Figure 재현: 탐색-활용 임계값(threshold) 곡선.

원 논문 방법 (Eq. 5):
    각 결정 시점에서 "탐색할 확률"을 로지스틱 모델로 적합한다.

        P(Explore) = sigmoid( beta * (t(x) - best_known) )
        t(x)       = a + m * x     (x = 남은 밤 수 비율 = nights_remaining / total_nights)

    - a    : 절편 (intercept)  — 마지막 밤(x=0) 근방의 임계값 수준
    - m    : 기울기 (slope)   — 밤이 지날수록 임계값이 줄어드는 속도
    - beta : 결정의 "날카로움"(결정론성). beta가 클수록 임계값 근처에서
             확률적 흔들림 없이 결정론적으로 explore/exploit이 갈린다.

    원 논문의 핵심 발견:
      1. 기울기 m은 분포·야간 수 조건 전체에서 하나의 공통값으로 수렴 (인간은
         분포와 무관하게 동일한 "탐색을 줄여나가는 속도"를 사용).
      2. 절편 a는 분포별로 다르며, 그 순서가 수학적 최적해의 순서와 일치.
      3. Feynman 최적해는 비선형(제곱근·로그 등)인 반면, 인간은 선형 근사를 사용.

    본 모듈은 위 로직을 LLM 실험 결과(Persona/Distribution/Model별)에 동일하게
    적용하여, a·m을 추정하고 Feynman 최적해 곡선과 나란히 그린다.

파이프라인:
    1. results/*.csv 로드 → 강제 탐색(첫날)·클램핑 구간 제외
    2. 그룹(예: Persona)별로 비정규화 로지스틱 회귀(MLE)를 적합해 (a, m, beta) 추정
    3. 분포별 2×2 subplot에 Feynman 최적해(점선) + 그룹별 선형 임계값(실선) 오버레이

사용법:
    # 페르소나별 임계값 비교
    python src/threshold_analysis.py --results_csv results/*.csv --group_by Persona

    # 모델별 비교
    python src/threshold_analysis.py --results_csv results/*.csv --group_by "LLM Model"
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from experiment import optimal_threshold  # Feynman 최적해 (Eq. 1-4)

DISTRIBUTIONS = ["exponential", "power_law", "uniform", "triangular"]
DIST_LABELS = {
    "exponential": "Exponential",
    "power_law":   "Power Law",
    "uniform":     "Uniform",
    "triangular":  "Triangular",
}
DIST_COLORS = {
    "exponential": (174/255, 205/255, 225/255),
    "power_law":   (84/255,  158/255, 63/255),
    "uniform":     (243/255, 193/255, 123/255),
    "triangular":  (100/255, 63/255,  149/255),
}
TOTAL_NIGHTS = [7, 14, 28]

# 임계값 곡선을 겹쳐 그릴 조건 수만큼 순환 사용
LINE_STYLES  = ["-", "--", "-.", ":"]
LINE_COLORS  = plt.get_cmap("tab10").colors


# ── Step 1: 데이터 로드 ───────────────────────────────────────────────────────

def load_decision_data(csv_paths) -> pd.DataFrame:
    """
    실험 결과 CSV(들)를 로드하여 임계값 회귀에 필요한 형태로 가공.

    - clamp == 0 조건만 사용 (원 논문과 동일 기준)
    - 첫날(Night == 0)은 강제 탐색이라 실제 "결정"이 아니므로 제외
    - Best Known이 없는(첫 탐색 이전) 행 제외
    - prop_remaining = (Total Nights - Night) / Total Nights
      → 이 결정을 내리는 시점에 "남아있는 밤 수의 비율" (x=1: 첫 결정, x→0: 마지막 결정)
    - is_explore = 1 (Explore) / 0 (Exploit)
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = []
    for pattern in csv_paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not os.path.exists(path):
                continue
            d = pd.read_csv(path, index_col=0)
            dfs.append(d)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if "clamp" in df.columns:
        df = df[df["clamp"] == 0]

    df = df[df["Night"] > 0].copy()
    df = df[df["Best Known"].notna()].copy()

    df["Distribution"] = df["Distribution"].str.lower()
    df["prop_remaining"] = (df["Total Nights"] - df["Night"]) / df["Total Nights"]
    df["is_explore"] = (df["Action"] == "Explore").astype(int)

    if "Persona" not in df.columns:
        df["Persona"] = "none"

    return df


# ── Step 2: Eq.5 로지스틱 모델 적합 ───────────────────────────────────────────

def fit_threshold_logistic(sub_df: pd.DataFrame, min_n: int = 20) -> dict:
    """
    한 조건(예: 특정 Persona × Distribution)에 대해 Eq.5 로지스틱 모델을 적합.

        logit(P(explore)) = c0 + c1*x + c2*best_known
                           = beta*a + beta*m*x - beta*best_known

    비정규화 최대우도추정(scipy.optimize)으로 c0, c1, c2를 구한 뒤:
        beta = -c2,   a = -c0/c2,   m = -c1/c2

    Returns
    -------
    dict(a, m, beta, n, log_likelihood) 또는 데이터 부족/발산 시 None.
    """
    x = sub_df["prop_remaining"].to_numpy(dtype=float)
    b = sub_df["Best Known"].to_numpy(dtype=float)
    y = sub_df["is_explore"].to_numpy(dtype=float)

    n = len(y)
    if n < min_n or y.std() == 0:
        return None

    # best_known 스케일이 분포마다 크게 다르므로(0~100 vs 0~100000) 표준화 후
    # 최적화하고, 끝에서 원래 스케일로 환산해 수치적으로 안정적인 적합을 보장.
    b_mean, b_scale = b.mean(), (b.std() if b.std() > 0 else 1.0)
    bn = (b - b_mean) / b_scale

    from scipy.optimize import minimize

    def neg_log_lik(params):
        c0, c1, c2 = params
        z = c0 + c1 * x + c2 * bn
        # log(1+exp(-z))*y + log(1+exp(z))*(1-y), np.logaddexp로 수치 안정화
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))

    result = minimize(neg_log_lik, x0=np.array([0.0, 1.0, -1.0]),
                       method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})

    if not result.success:
        return None

    c0, c1, c2_n = result.x
    # bn = (b - b_mean)/b_scale 이므로 원래 스케일의 계수로 환산:
    #   c0 + c1*x + c2_n*(b-b_mean)/b_scale = (c0 - c2_n*b_mean/b_scale) + c1*x + (c2_n/b_scale)*b
    c2 = c2_n / b_scale
    c0_real = c0 - c2_n * b_mean / b_scale

    if abs(c2) < 1e-10:
        return None  # best_known이 결정에 영향 없음 → 임계값 추정 불가

    beta = -c2
    a = -c0_real / c2
    m = -c1 / c2

    return {"a": a, "m": m, "beta": beta, "n": n,
            "neg_log_lik": result.fun}


def fit_all_thresholds(df: pd.DataFrame, group_col: str = "Persona",
                       min_n: int = 20) -> pd.DataFrame:
    """
    (group_col, Distribution) 조합별로 fit_threshold_logistic 적용.

    원 논문에서 기울기(m)가 분포·야간수 전체에서 공통으로 수렴한 것과 같이,
    Total Nights는 풀링(pooling)하고 Distribution 단위로만 분리해 적합한다.
    """
    rows = []
    for group_val, g in df.groupby(group_col):
        for dist in DISTRIBUTIONS:
            sub = g[g["Distribution"] == dist]
            fit = fit_threshold_logistic(sub, min_n=min_n)
            if fit is None:
                continue
            rows.append({group_col: group_val, "distribution": dist, **fit})
    return pd.DataFrame(rows)


# ── Step 3: Figure 생성 ───────────────────────────────────────────────────────

def _optimal_curve(distribution: str, total_nights: int):
    """Feynman 최적해를 (x=남은 밤 비율, threshold) 점들로 반환."""
    ns = np.arange(1, total_nights + 1)
    xs = ns / total_nights
    ts = [optimal_threshold(int(n), distribution) for n in ns]
    order = np.argsort(xs)
    return xs[order], np.array(ts)[order]


def plot_threshold_curves(df: pd.DataFrame,
                          group_col: str = "Persona",
                          min_n: int = 20,
                          output_path: str = "figures/threshold_curves.pdf",
                          show: bool = True):
    """
    원 논문 Figure 스타일 임계값 곡선.

    레이아웃 : 2행 x 2열 (분포별 subplot)
    x축      : 남은 밤 수 비율 (1=실험 시작 -> 0=마지막 밤)
    y축      : 임계값 점수 (해당 분포의 점수 스케일)

    각 subplot:
      - Feynman 최적해 곡선 (점선, Total Nights=7/14/28 각각) — 비선형
      - group_col(예: Persona)별 선형 적합 t(x) = a + m*x — 실선
    """
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       10,
        "axes.labelsize":  9,
        "axes.titlesize":  11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth":  0.8,
        "lines.linewidth": 1.4,
    })

    fits = fit_all_thresholds(df, group_col=group_col, min_n=min_n)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Explore-Exploit Threshold: Feynman Optimal vs. LLM (by {group_col})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    groups = list(fits[group_col].unique()) if len(fits) else []

    for ax_idx, dist in enumerate(DISTRIBUTIONS):
        ax = axes[ax_idx // 2, ax_idx % 2]

        # ① Feynman 최적해 (비선형, 점선) — Total Nights 조건별로 오버레이
        for ti, t_nights in enumerate(TOTAL_NIGHTS):
            xs, ts = _optimal_curve(dist, t_nights)
            ax.plot(xs, ts, linestyle=":", linewidth=1.2,
                    color="black", alpha=0.35 + 0.2 * ti,
                    label=f"Feynman Optimal (T={t_nights})")

        # ② 그룹별 선형 임계값 적합 (실선)
        dist_fits = fits[fits["distribution"] == dist] if len(fits) else fits
        for gi, group_val in enumerate(groups):
            row = dist_fits[dist_fits[group_col] == group_val]
            if len(row) == 0:
                continue
            a, m, n = row["a"].iloc[0], row["m"].iloc[0], row["n"].iloc[0]
            xs = np.linspace(0, 1, 50)
            ts = a + m * xs
            color = LINE_COLORS[gi % len(LINE_COLORS)]
            style = LINE_STYLES[gi // len(LINE_COLORS) % len(LINE_STYLES)]
            ax.plot(xs, ts, linestyle=style, color=color,
                    label=f"{group_val} (n={n})")

        ax.set_title(DIST_LABELS[dist], fontweight="bold")
        ax.set_xlabel("Proportion of Nights Remaining")
        ax.set_ylabel("Threshold Score")
        ax.set_xlim(0, 1)
        ax.invert_xaxis()  # 원 논문과 동일: 왼쪽=실험 시작, 오른쪽=마지막 밤
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, ncol=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 저장: {output_path}")

    if show:
        plt.show()
    plt.close()

    return fits


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="원 논문 스타일 탐색-활용 임계값 Figure 생성 (Eq.5 로지스틱 적합)"
    )
    parser.add_argument(
        "--results_csv", nargs="+", required=True,
        help="실험 결과 CSV 경로 (glob 패턴 가능, 예: results/*.csv)"
    )
    parser.add_argument(
        "--group_by", default="Persona",
        help="비교할 그룹 컬럼 (예: Persona, 'LLM Model'). 기본: Persona"
    )
    parser.add_argument(
        "--min_n", type=int, default=20,
        help="그룹×분포 조합별 로지스틱 적합에 필요한 최소 결정 수 (기본 20)"
    )
    parser.add_argument(
        "--output", default="figures/threshold_curves.pdf",
        help="Figure 저장 경로"
    )
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    df = load_decision_data(args.results_csv)
    if len(df) == 0:
        print("[경고] 로드된 결정 데이터가 없습니다. --results_csv 경로를 확인하세요.")
        return

    print(f"로드된 결정 수: {len(df)}  "
          f"(그룹={df[args.group_by].nunique()}개, "
          f"분포={df['Distribution'].nunique()}개)")

    fits = plot_threshold_curves(
        df, group_col=args.group_by, min_n=args.min_n,
        output_path=args.output, show=not args.no_show,
    )

    if len(fits):
        print("\n적합된 임계값 파라미터 (a=절편, m=기울기, beta=결정론성):")
        print(fits[[args.group_by, "distribution", "a", "m", "beta", "n"]]
              .round(3).to_string(index=False))
    else:
        print("[경고] 적합에 성공한 그룹×분포 조합이 없습니다 "
              "(--min_n을 낮추거나 데이터를 더 수집하세요).")


if __name__ == "__main__":
    main()
