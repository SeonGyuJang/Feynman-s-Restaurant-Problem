"""
grid_heatmap.py
----------------
LLM의 4x7 그리드 탐색(EXPLORE) 위치를 원본 그리드 형태 그대로 시각화.

원 논문(Christian et al., 2026)의 인간 실험은 참가자가 그리드의 "어느 칸"을
선택했는지는 기록하지 않고 탐색/활용 여부만 기록했다. 본 연구의 LLM 실험은
좌표(Position)를 명시적으로 기록하므로, 실제 4x7 그리드 레이아웃 위에 각 칸이
얼마나 자주 선택되었는지를 히트맵으로 겹쳐 그릴 수 있다 — 인간 데이터로는
불가능했던 공간적 탐색 패턴 분석.

파이프라인:
    1. results/*.csv 로드 → Action == "Explore" 행만 사용
       (Exploit은 항상 기존 최고점 위치로 "복귀"이므로 새로운 탐색 선택이 아님)
    2. Position "(row,col)" 문자열 파싱
    3. 그룹(기본: Persona)별로 4x7 선택 횟수/비율 행렬 계산
    4. prompt_builder.render_grid()와 동일한 4행(R0-R3) x 7열(C0-C6) 레이아웃으로
       각 그룹의 히트맵을 나란히 그림 (선택 비율이 높을수록 진한 색)

사용법:
    # 페르소나별 그리드 탐색 히트맵
    python src/grid_heatmap.py --results_csv results/*.csv --facet Persona

    # 분포별 그리드 탐색 히트맵
    python src/grid_heatmap.py --results_csv results/*.csv --facet Distribution
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

N_ROWS, N_COLS = 4, 7
POSITION_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


# ── Step 1: 데이터 로드 ───────────────────────────────────────────────────────

def load_explore_positions(csv_paths) -> pd.DataFrame:
    """
    실험 결과 CSV(들)를 로드하여 EXPLORE 행동의 그리드 좌표만 추출.

    Returns: DataFrame with columns
        Persona, Distribution, LLM Model, row, col
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = []
    for pattern in csv_paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not os.path.exists(path):
                continue
            dfs.append(pd.read_csv(path, index_col=0))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if "clamp" in df.columns:
        df = df[df["clamp"] == 0]

    df = df[df["Action"] == "Explore"].copy()

    parsed = df["Position"].astype(str).str.extract(POSITION_RE)
    df["row"] = pd.to_numeric(parsed[0], errors="coerce")
    df["col"] = pd.to_numeric(parsed[1], errors="coerce")
    df = df.dropna(subset=["row", "col"]).copy()
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)

    if "Persona" not in df.columns:
        df["Persona"] = "none"
    if "Distribution" in df.columns:
        df["Distribution"] = df["Distribution"].str.lower()

    return df


# ── Step 2: 그리드 선택 횟수 집계 ─────────────────────────────────────────────

def compute_grid_counts(df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
    """EXPLORE 좌표 DataFrame → 4x7 선택 횟수(또는 비율) 행렬."""
    counts = np.zeros((N_ROWS, N_COLS), dtype=float)
    for _, row in df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        if 0 <= r < N_ROWS and 0 <= c < N_COLS:
            counts[r, c] += 1
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


# ── Step 3: Figure 생성 ───────────────────────────────────────────────────────

def _draw_grid_heatmap(ax, grid: np.ndarray, title: str, n_total: int,
                       vmax: float, annotate: bool, pct: bool, cmap: str):
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(N_COLS))
    ax.set_xticklabels([f"C{c}" for c in range(N_COLS)])
    ax.set_yticks(range(N_ROWS))
    ax.set_yticklabels([f"R{r}" for r in range(N_ROWS)])

    # 4x7 그리드 셀 경계선을 명시적으로 그려 원본 레스토랑 그리드처럼 표현
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    if annotate:
        for r in range(N_ROWS):
            for c in range(N_COLS):
                val = grid[r, c]
                text = f"{val*100:.1f}%" if pct else f"{val:.0f}"
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(c, r, text, ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title(f"{title}\n(n={n_total} explores)", fontsize=10, fontweight="bold")
    return im


def plot_grid_heatmaps(df: pd.DataFrame,
                       facet_col: str = "Persona",
                       normalize: bool = True,
                       n_cols: int = 3,
                       output_path: str = "figures/grid_heatmaps.pdf",
                       show: bool = True,
                       order: list = None):
    """
    facet_col(기본: Persona)의 각 값마다 4x7 그리드 탐색 히트맵을 그려
    한 Figure에 나란히 배치.

    normalize=True이면 그룹 내 비율(%), False이면 원시 선택 횟수를 표시.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   9,
    })

    if len(df) == 0:
        raise ValueError("탐색 데이터가 없습니다 (Action=='Explore' 행이 0개).")

    groups = order if order else sorted(df[facet_col].dropna().unique().tolist())
    n_groups = len(groups)
    n_cols = max(1, min(n_cols, n_groups))
    n_rows = int(np.ceil(n_groups / n_cols))

    # 그룹 간 색상 스케일을 통일하기 위해 전체 vmax를 먼저 계산
    grids = {}
    for g in groups:
        sub = df[df[facet_col] == g]
        grids[g] = (compute_grid_counts(sub, normalize=normalize), len(sub))
    vmax = max((g[0].max() for g in grids.values()), default=0) or 1.0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 2.6 * n_rows + 1),
                             squeeze=False)
    fig.suptitle(
        f"Grid Exploration Heatmap by {facet_col}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    im = None
    for idx, g in enumerate(groups):
        ax = axes[idx // n_cols, idx % n_cols]
        grid, n_total = grids[g]
        im = _draw_grid_heatmap(
            ax, grid, title=str(g), n_total=n_total,
            vmax=vmax, annotate=True, pct=normalize, cmap="YlOrRd",
        )

    # 남는 subplot 숨기기
    for idx in range(n_groups, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.subplots_adjust(right=0.88, wspace=0.35, hspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    label = "Selection proportion" if normalize else "Selection count"
    fig.colorbar(im, cax=cbar_ax, label=label)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 저장: {output_path}")

    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM 그리드 탐색(EXPLORE) 위치 히트맵 생성"
    )
    parser.add_argument(
        "--results_csv", nargs="+", required=True,
        help="실험 결과 CSV 경로 (glob 패턴 가능, 예: results/*.csv)"
    )
    parser.add_argument(
        "--facet", default="Persona",
        choices=["Persona", "Distribution", "LLM Model"],
        help="히트맵을 나눌 기준 컬럼 (기본: Persona)"
    )
    parser.add_argument(
        "--raw_counts", action="store_true",
        help="비율(%) 대신 원시 선택 횟수를 표시"
    )
    parser.add_argument(
        "--n_cols", type=int, default=3,
        help="한 행에 배치할 subplot 수 (기본 3)"
    )
    parser.add_argument(
        "--output", default="figures/grid_heatmaps.pdf",
        help="Figure 저장 경로"
    )
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    df = load_explore_positions(args.results_csv)
    if len(df) == 0:
        print("[경고] EXPLORE 좌표 데이터가 없습니다. --results_csv 경로를 확인하세요.")
        return

    print(f"로드된 EXPLORE 행동 수: {len(df)}  "
          f"({args.facet} 그룹: {df[args.facet].nunique()}개)")

    # PERSONA_CONDITIONS 순서를 기준으로 정렬 (있는 경우)
    order = None
    if args.facet == "Persona":
        try:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from personas import PERSONA_CONDITIONS
            present = set(df["Persona"].unique())
            order = [p for p in PERSONA_CONDITIONS if p in present]
            order += sorted(present - set(order))
        except ImportError:
            pass

    plot_grid_heatmaps(
        df, facet_col=args.facet, normalize=not args.raw_counts,
        n_cols=args.n_cols, output_path=args.output,
        show=not args.no_show, order=order,
    )


if __name__ == "__main__":
    main()
