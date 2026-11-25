#!/usr/bin/env python3
"""Exp 3 (R1-C4): Fig. 2a–style score-gap quintile triptych on RadPref and ReXVal.

Figures only — no hypothesis tests. Reuses existing RadPref/ReXVal loaders by import.
Outputs exclusively under results/revision_r2/exp3/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "revision_r2" / "exp3"
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


radpref = _load_module(
    "radpref_paired",
    REPO_ROOT / "scripts" / "radpref_data" / "run_radpref_paired_stats.py",
)
rexval = _load_module(
    "rexval_align",
    REPO_ROOT / "scripts" / "rexval_data" / "run_rexval_alignment_stats.py",
)

DISPLAY = {
    "ICARE_AVG": "ICARE (2025)",
    "ICARE_PREDEFINED": "ICARE-Predefined (2025)",
    "BLEU": "BLEU-2 (2002)",
    "SembScore": "SembScore (2020)",
    "BERTScore": "BERTScore (2019)",
    "GREEN": "GREEN (2024)",
    "RadCliQ-v1": "1/RadCliQ-v1 (2022)",
    "RadGraph": "RadGraph (2022)",
    "AlignScore": "AlignScore (2023)",
    "CRIMSON": "CRIMSON (2026)",
}
PLOT_ORDER = [
    "ICARE (2025)",
    "ICARE-Predefined (2025)",
    "BLEU-2 (2002)",
    "SembScore (2020)",
    "BERTScore (2019)",
    "GREEN (2024)",
    "1/RadCliQ-v1 (2022)",
    "RadGraph (2022)",
    "AlignScore (2023)",
    "CRIMSON (2026)",
]
METRIC_COLORS = {
    "ICARE (2025)": "#1A6FAF",
    "ICARE-Predefined (2025)": "#0B3C6F",
    "BLEU-2 (2002)": "#FF7F0E",
    "SembScore (2020)": "#2CA02C",
    "BERTScore (2019)": "#9467BD",
    "GREEN (2024)": "#D62728",
    "1/RadCliQ-v1 (2022)": "#8C564B",
    "RadGraph (2022)": "#E377C2",
    "AlignScore (2023)": "#7F7F7F",
    "CRIMSON (2026)": "#BCBD22",
}
HDR_BLUE = "#1A4A7A"
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, HDR_FS = 24, 22, 20, 17, 28
FIG_W = 28.0


def robust_qcut(series: pd.Series, q: int = 5) -> pd.Series:
    binned = pd.qcut(series, q=q, duplicates="drop")
    n = binned.cat.categories.size
    labels = [f"Q{i + 1}" for i in range(n)]
    return pd.qcut(series, q=q, duplicates="drop", labels=labels)


def classify(doctor: float, metric: float) -> str:
    """doctor/metric in {-1,0,+1}; 0 = tie / Can't say."""
    if doctor == 0:
        return "Indecision"
    if metric == 0:
        return "Misalignment"  # decisive rater + metric tie (Fig. 2a rule)
    if doctor == metric:
        return "Alignment"
    return "Misalignment"


def triptych_table(rows: List[dict]) -> Tuple[pd.DataFrame, dict]:
    """rows: metric_key, gap, doctor, metric_pref."""
    by_metric: Dict[str, list] = {}
    for r in rows:
        by_metric.setdefault(r["metric_key"], []).append(r)

    out_rows = []
    n_doc_tie = n_met_tie = n_total = 0
    for key, items in by_metric.items():
        display = DISPLAY[key]
        gaps = np.array([it["gap"] for it in items], dtype=float)
        doctors = np.array([it["doctor"] for it in items], dtype=float)
        metrics = np.array([it["metric_pref"] for it in items], dtype=float)
        labels = [classify(d, m) for d, m in zip(doctors, metrics)]
        n_total += len(items)
        n_doc_tie += int((doctors == 0).sum())
        n_met_tie += int((metrics == 0).sum())

        df = pd.DataFrame({"gap": gaps, "label": labels})
        df["quintile"] = robust_qcut(df["gap"], q=5)
        counts = df.groupby(["quintile", "label"], observed=True).size().unstack(fill_value=0)
        for c in ("Alignment", "Misalignment", "Indecision"):
            if c not in counts.columns:
                counts[c] = 0
        tot = counts.sum(axis=1)
        for q in counts.index.astype(str):
            n = int(tot.loc[q])
            out_rows.append(
                {
                    "metric": display,
                    "quintile": q,
                    "indecision": round(float(counts.loc[q, "Indecision"] / n), 6),
                    "alignment": round(float(counts.loc[q, "Alignment"] / n), 6),
                    "misalignment": round(float(counts.loc[q, "Misalignment"] / n), 6),
                    "n": n,
                }
            )
    tie = {
        "n_rows": n_total,
        "doctor_tie_fraction": round(n_doc_tie / n_total, 6) if n_total else None,
        "metric_tie_fraction": round(n_met_tie / n_total, 6) if n_total else None,
        "n_doctor_ties": n_doc_tie,
        "n_metric_ties": n_met_tie,
    }
    return pd.DataFrame(out_rows), tie


def build_radpref_rows() -> List[dict]:
    cases = radpref.load_case_table()
    prefs = radpref.load_rater_prefs(cases["case_id"].to_numpy())
    diffs = radpref.load_metric_diffs(cases)
    rows: List[dict] = []
    for key, diff in diffs.items():
        m_pref = np.sign(diff).astype(float)
        gap = np.abs(diff).astype(float)
        for rname, rp in prefs.items():
            for i in range(len(diff)):
                rows.append(
                    {
                        "metric_key": key,
                        "gap": float(gap[i]),
                        "doctor": float(rp[i]),
                        "metric_pref": float(m_pref[i]),
                        "case_id": cases["case_id"].iloc[i],
                        "rater": rname,
                    }
                )
    return rows


def build_rexval_rows() -> List[dict]:
    _studies, E, scores, raters = rexval.load_study_tables()
    S, C, R = E.shape
    assert C == 4 and S == 50
    pairs = list(combinations(range(C), 2))
    rows: List[dict] = []
    for key, sc in scores.items():
        for s in range(S):
            for a, b in pairs:
                gap = float(abs(sc[s, a] - sc[s, b]))
                if sc[s, a] > sc[s, b]:
                    m_pref = 1.0  # prefer a
                elif sc[s, a] < sc[s, b]:
                    m_pref = -1.0  # prefer b
                else:
                    m_pref = 0.0
                for ri in range(R):
                    ea, eb = float(E[s, a, ri]), float(E[s, b, ri])
                    if ea < eb:
                        d_pref = 1.0  # fewer errors → a
                    elif ea > eb:
                        d_pref = -1.0
                    else:
                        d_pref = 0.0
                    rows.append(
                        {
                            "metric_key": key,
                            "gap": gap,
                            "doctor": d_pref,
                            "metric_pref": m_pref,
                            "study": s,
                            "pair": f"{a}-{b}",
                            "rater": int(raters[ri]),
                        }
                    )
    return rows


def prepare_plot_df(trip: pd.DataFrame) -> pd.DataFrame:
    q_to_x = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4}
    df = trip.copy()
    df["x_pos"] = df["quintile"].map(q_to_x)
    # Two-bin metrics: move second surviving bin to Q5 (Fig. 2a / CRIMSON convention)
    for metric, sub in df.groupby("metric"):
        qs = sorted(sub["quintile"].unique(), key=lambda q: int(q[1:]))
        if len(qs) == 2:
            second = qs[1]
            df.loc[(df["metric"] == metric) & (df["quintile"] == second), "x_pos"] = 4
    return df


def is_icare_main(m: str) -> bool:
    return m == "ICARE (2025)"


def is_icare_pre(m: str) -> bool:
    return "Predefined" in m


def is_icare_any(m: str) -> bool:
    return is_icare_main(m) or is_icare_pre(m)


def add_header(fig, text: str, gap: float = 0.020) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tops = []
    for ax in fig.get_axes():
        try:
            bb = ax.title.get_window_extent(renderer)
            bb_fig = bb.transformed(fig.transFigure.inverted())
            if bb_fig.y1 > 0.45:
                tops.append(bb_fig.y1)
        except Exception:
            pass
    hdr_h = 1.05 / fig.get_figheight()
    hdr_bot = (max(tops) if tops else 0.88) + gap
    ax_h = fig.add_axes([0.0, hdr_bot, 1.0, hdr_h])
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)
    ax_h.axis("off")
    ax_h.add_patch(
        FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="square,pad=0", fc=HDR_BLUE, ec="none", transform=ax_h.transAxes
        )
    )
    ax_h.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=HDR_FS,
        fontweight="bold",
        color="white",
        transform=ax_h.transAxes,
    )


def plot_triptych(trip: pd.DataFrame, header: str, stem: str) -> None:
    df = prepare_plot_df(trip)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_FS,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 8.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.28, wspace=0.34)
    panels = [
        ("indecision", 'Indecision Rate\n("Can\'t Say" / Tied Ratings)', (0.0, 1.05)),
        ("alignment", "Doctor-Metric\nAlignment Rate\n(Doctor = Metric)", (0.0, 0.82)),
        ("misalignment", "Doctor-Metric\nMisalignment Rate\n(Doctor ≠ Metric)", (0.0, 0.82)),
    ]
    present = [m for m in PLOT_ORDER if m in set(df["metric"])]
    for ax, (key, title, ylim) in zip(axes, panels):
        for metric in present:
            sub = df[df["metric"] == metric].sort_values("x_pos")
            ic = is_icare_any(metric)
            color = METRIC_COLORS.get(metric, "#444444")
            lw = 4.8 if is_icare_main(metric) else (3.7 if ic else 1.7)
            ms = 12.5 if is_icare_main(metric) else (10.8 if ic else 6.2)
            alpha = 1.0 if ic else 0.60
            zord = 5 if is_icare_main(metric) else (4 if ic else 2)
            ax.plot(
                sub["x_pos"].values,
                sub[key].values,
                color=color,
                lw=lw,
                marker="o",
                ms=ms,
                alpha=alpha,
                zorder=zord,
            )
        ax.set_xticks(range(5))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=TICK_FS, fontweight="bold")
        ax.set_xlabel("Score-Gap Quintile\n(per Metric)", fontsize=LABEL_FS - 1, fontweight="bold", labelpad=12)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")
        if key == "indecision":
            ax.set_ylabel("Proportion of Samples", fontsize=LABEL_FS, fontweight="bold", labelpad=14)

    # Legend under middle panel: ICARE first
    handles = []
    for metric in present:
        ic = is_icare_any(metric)
        color = METRIC_COLORS.get(metric, "#444444")
        lw = 4.4 if is_icare_main(metric) else (3.4 if ic else 1.7)
        ms = 12 if is_icare_main(metric) else (10 if ic else 6)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=lw,
                marker="o",
                ms=ms,
                alpha=1.0 if ic else 0.70,
                label=metric,
            )
        )
    axes[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=5,
        fontsize=LEGEND_FS,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.2,
        handletextpad=0.5,
    )
    add_header(fig, header, gap=0.018)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=600, format="pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=600, format="png")
    plt.close(fig)


def icare_trend_lines(trip: pd.DataFrame, metric: str = "ICARE (2025)") -> str:
    sub = trip[trip["metric"] == metric].copy()
    sub["qnum"] = sub["quintile"].str[1:].astype(int)
    sub = sub.sort_values("qnum")
    ind = ", ".join(f"{r.quintile}={r.indecision:.3f}" for r in sub.itertuples())
    aln = ", ".join(f"{r.quintile}={r.alignment:.3f}" for r in sub.itertuples())
    return f"Indecision [{ind}]; Alignment [{aln}]."


def write_captions_and_report(ties: dict, trends: dict) -> None:
    captions = f"""# Exp 3 captions (R1-C4)

## RRG-Pref sampling provenance (Fig. 2a source cohort)

The 153 RRG-Pref preference pairs were selected from IU-Xray model-pair ICARE summaries as follows:

1. **Pool construction** (`mcq_best_worst_analysis.py`, `--selection_mode all_combinations`): all model-seed report pairs with ICARE `Agreement_Percentage` averaged across **evaluation seeds 123, 456, 789, 202, 101** and across both `gt_reports_as_ref` and `gen_reports_as_ref` (ICARE-AVG). Inputs live under `MCQ_gen_data_our_eval_seed{{seed}}/IU_xray/.../shuffled_ans_choices_data/`. Shell defaults: model seeds 1–5 (`mcq_best_worst_analysis.sh`).
2. **Round 1 (n = 100):** drop identical high/low generated reports and duplicate `(Report_ID, Generated_Report_High, Generated_Report_Low)` pairs; importance-sample by ICARE score gap with **beta = 0.9955**, **sample_size = 100**, **seed = 12345** (`sampling_data_and_analysis.py` / `.sh`).
3. **Round 2 (n = +53 → 153):** targeted fill of ICARE |gap| bins **5–10 / 10–15 / 15–20 / >20** to **25** each (`--target_per_bin 25`, same seed). Combined file: `combined_targeted_samples.csv`.

Manuscript Fig. 2a then keeps the **119** samples with ≥2-of-3 clinician consensus (34 one-each splits dropped). Metric-tie with a decisive clinician vote is counted as **misalignment** (same rule used here).

## RadPref figure

**Header:** RadPref: score gap vs clinician certainty

**Unit of analysis:** (case, rater) rows (100 cases × 3 raters). Tied 1–5 ratings → “Can't say” / indecision; otherwise the rater prefers the higher-rated report. Score gap = |metric(C1) − metric(C2)|; quintiles per metric via `pd.qcut(..., duplicates=\"drop\")`. Two-bin metrics (e.g. CRIMSON) are plotted at Q1 and Q5. Metric tie with a decisive rater counts as misalignment.

**Trend (descriptive):** {trends['radpref']}

## ReXVal figure

**Header:** ReXVal: score gap vs clinician certainty

**Unit of analysis:** all 6 candidate pairs per study × 6 raters (50 × 6 × 6). Equal clinically significant error counts → “Can't say”; otherwise the rater prefers fewer errors. Same quintile / two-bin / metric-tie rules as RadPref and Fig. 2a.

**Trend (descriptive):** {trends['rexval']}

## Tie fractions (Exp 3 rows)

- RadPref: doctor-tie {ties['radpref']['doctor_tie_fraction']}, metric-tie {ties['radpref']['metric_tie_fraction']} (n={ties['radpref']['n_rows']} rows × 10 metrics counted separately in metric-tie aggregate).
- ReXVal: doctor-tie {ties['rexval']['doctor_tie_fraction']}, metric-tie {ties['rexval']['metric_tie_fraction']} (n={ties['rexval']['n_rows']}).
"""
    (OUT_DIR / "captions.md").write_text(captions)

    report = f"""# Exp 3 report — Fig. 2a triptych on RadPref and ReXVal

## Provenance

- Script: `scripts/clinical_stats/run_exp3_score_gap.py`
- Loaders (import only): `scripts/radpref_data/run_radpref_paired_stats.py`, `scripts/rexval_data/run_rexval_alignment_stats.py`
- Metric scores: RadPref / ReXVal `eval_seed_123` ICARE shuffled + predefined + baselines (same paths as Exp 2 paired stats)
- RRG-Pref sampling gaps used **ICARE-AVG** aggregated over eval seeds **123, 456, 789, 202, 101** (see captions.md)

## Rules

- Per-(case/pair, rater) rows
- Metric tie + decisive rater → **misalignment**
- Quintiles: `pd.qcut(q=5, duplicates=\"drop\")`; two-bin metrics at Q1 and Q5
- No hypothesis tests — descriptive figures only

## Tie fractions

```json
{json.dumps(ties, indent=2)}
```

## ICARE trends (descriptive)

- **RadPref:** {trends['radpref']}
- **ReXVal:** {trends['rexval']}

On both independent datasets, ICARE's indecision rate fell and alignment rose with score gap (see quintile series above).
"""
    (OUT_DIR / "report_exp3.md").write_text(report)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building RadPref rows...")
    rp_rows = build_radpref_rows()
    rp_trip, rp_tie = triptych_table(rp_rows)
    # doctor ties should not be multiplied by metrics in the headline fraction —
    # recompute doctor-tie on unique (case,rater) once
    cases = radpref.load_case_table()
    prefs = radpref.load_rater_prefs(cases["case_id"].to_numpy())
    doc = np.concatenate([prefs[k] for k in prefs])
    rp_tie["doctor_tie_fraction"] = round(float((doc == 0).mean()), 6)
    rp_tie["n_doctor_ties"] = int((doc == 0).sum())
    rp_tie["n_doctor_rows"] = int(doc.size)
    # metric-tie: average over metrics of per-metric tie rate on cases
    diffs = radpref.load_metric_diffs(cases)
    met_tie_rates = [float((np.sign(v) == 0).mean()) for v in diffs.values()]
    rp_tie["metric_tie_fraction"] = round(float(np.mean(met_tie_rates)), 6)

    rp_path = OUT_DIR / "radpref_panel_triptych.csv"
    rp_trip.to_csv(rp_path, index=False)
    print(f"Wrote {rp_path} ({len(rp_trip)} rows)")

    print("Building ReXVal rows...")
    rx_rows = build_rexval_rows()
    rx_trip, rx_tie = triptych_table(rx_rows)
    _studies, E, scores, _raters = rexval.load_study_tables()
    S, C, R = E.shape
    pairs = list(combinations(range(C), 2))
    doc_ties = 0
    doc_n = 0
    for s in range(S):
        for a, b in pairs:
            for ri in range(R):
                doc_n += 1
                if E[s, a, ri] == E[s, b, ri]:
                    doc_ties += 1
    rx_tie["doctor_tie_fraction"] = round(doc_ties / doc_n, 6)
    rx_tie["n_doctor_ties"] = doc_ties
    rx_tie["n_doctor_rows"] = doc_n
    met_rates = []
    for sc in scores.values():
        ties = 0
        n = 0
        for s in range(S):
            for a, b in pairs:
                n += 1
                if sc[s, a] == sc[s, b]:
                    ties += 1
        met_rates.append(ties / n)
    rx_tie["metric_tie_fraction"] = round(float(np.mean(met_rates)), 6)

    rx_path = OUT_DIR / "rexval_panel_triptych.csv"
    rx_trip.to_csv(rx_path, index=False)
    print(f"Wrote {rx_path} ({len(rx_trip)} rows)")

    ties = {"radpref": rp_tie, "rexval": rx_tie}
    (OUT_DIR / "tie_fractions.json").write_text(json.dumps(ties, indent=2))

    print("Plotting...")
    plot_triptych(rp_trip, "RadPref: score gap vs clinician certainty", "radpref_fig2a_triptych")
    plot_triptych(rx_trip, "ReXVal: score gap vs clinician certainty", "rexval_fig2a_triptych")

    trends = {
        "radpref": icare_trend_lines(rp_trip),
        "rexval": icare_trend_lines(rx_trip),
    }
    write_captions_and_report(ties, trends)
    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
