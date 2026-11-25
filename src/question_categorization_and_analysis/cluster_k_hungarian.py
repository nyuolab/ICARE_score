#!/usr/bin/env python3
"""Hungarian one-to-many / many-to-one mapping of Exp4 clusters to K=20 (Fig. 4).

Read-only w.r.t. existing Exp4/final artefacts. Writes only under
``results/revision_r2/exp4/hungarian/``.

Uses existing ``model_seeds_all/clustered_questions_K*.csv`` assignments and
``scores_K*.csv`` (pooled GT/GEN as Fig. 4). Does not re-embed, re-cluster, or rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from question_categorization_and_analysis.make_r2_supp_figures import (  # noqa: E402
    MODEL_ORDER,
    all_concept_match_info,
    display,
    load_names,
    load_scores,
    side_of_diagonal,
)

K20 = 20
KS_OTHER = (10, 15, 25, 30)
EXP4_ROOT_REL = Path("results/revision_r2/exp4")
OUT_REL = Path("results/revision_r2/exp4/hungarian")
TABLE_D_REL = Path("results/revision_r2/final/SuppTableD_concept_consistency_by_K.csv")
M_REQUIRED = 1_000_000_000.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path, out_path: Path) -> Dict[str, str]:
    """SHA-256 every regular file under root except exp4/hungarian/."""
    rows: List[Tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts[:2] == ("exp4", "hungarian"):
            continue
        rows.append((str(rel).replace("\\", "/"), sha256_file(p)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rel, digest in rows:
            f.write(f"{digest}  {rel}\n")
    return {rel: digest for rel, digest in rows}


def load_unique_cluster_map(csv_path: Path) -> Dict[int, set]:
    df = pd.read_csv(csv_path, usecols=["Question_Text", "cluster_id"])
    df = df.drop_duplicates(subset=["Question_Text"])
    out: Dict[int, set] = defaultdict(set)
    for cid, q in zip(df.cluster_id.to_numpy(), df.Question_Text.to_numpy()):
        out[int(cid)].add(q)
    return dict(out)


def overlap_and_jaccard(
    map_a: Dict[int, set], map_b: Dict[int, set]
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    ids_a = sorted(map_a.keys())
    ids_b = sorted(map_b.keys())
    overlap = np.zeros((len(ids_a), len(ids_b)), dtype=float)
    jacc = np.zeros((len(ids_a), len(ids_b)), dtype=float)
    for i, a in enumerate(ids_a):
        sa = map_a[a]
        for j, b in enumerate(ids_b):
            sb = map_b[b]
            inter = len(sa & sb)
            uni = len(sa | sb)
            overlap[i, j] = float(inter)
            jacc[i, j] = float(inter) / float(uni) if uni else 0.0
    return overlap, jacc, ids_a, ids_b


def hungarian_with_required_optional(
    overlap_rows_cols: np.ndarray,
    n_receivers: int,
    n_optional_per_receiver: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Rows assigned to column-copies of receivers (1 required + n_opt optional each)."""
    n_rows, n_recv_check = overlap_rows_cols.shape
    assert n_recv_check == n_receivers
    n_copies = 1 + n_optional_per_receiver
    n_cols = n_receivers * n_copies
    cost = np.zeros((n_rows, n_cols), dtype=float)
    for j in range(n_receivers):
        base = j * n_copies
        cost[:, base] = -overlap_rows_cols[:, j] - M_REQUIRED
        for t in range(n_optional_per_receiver):
            cost[:, base + 1 + t] = -overlap_rows_cols[:, j]
    row_ind, col_ind = linear_sum_assignment(cost)
    receivers = [int(c // n_copies) for c in col_ind]
    return row_ind, col_ind, receivers


def assignment_finer(overlap_k20_by_knew: np.ndarray) -> Dict[int, int]:
    """K_new id -> K20 id."""
    ov = overlap_k20_by_knew.T
    k_new = ov.shape[0]
    _, _, receivers = hungarian_with_required_optional(ov, K20, k_new - K20)
    return {int(i): int(receivers[i]) for i in range(k_new)}


def assignment_coarser(overlap_k20_by_knew: np.ndarray) -> Dict[int, int]:
    """K20 id -> K_new id."""
    ov = overlap_k20_by_knew
    k_new = ov.shape[1]
    _, _, receivers = hungarian_with_required_optional(ov, k_new, K20 - k_new)
    return {int(i): int(receivers[i]) for i in range(K20)}


def min_cost_flow_finer(overlap_k20_by_knew: np.ndarray) -> Dict[int, int]:
    """Each K_new -> one K20; each K20 receives >= 1 (lower bound via required edge)."""
    ov = overlap_k20_by_knew.T  # (K_new, 20)
    k_new, n20 = ov.shape
    max_extra = k_new - n20
    G = nx.MultiDiGraph()
    G.add_node("s", demand=-k_new)
    G.add_node("t", demand=k_new)
    for i in range(k_new):
        G.add_edge("s", f"k{i}", capacity=1, weight=0)
        for j in range(n20):
            # Non-negative weights: max_ov - overlap
            w = int(ov.max() - ov[i, j])
            G.add_edge(f"k{i}", f"r{j}", capacity=1, weight=w)
    for j in range(n20):
        # required unit (strongly preferred) + optional extras
        G.add_edge(f"r{j}", "t", key="req", capacity=1, weight=0)
        if max_extra > 0:
            G.add_edge(f"r{j}", "t", key="opt", capacity=max_extra, weight=10**6)
    flow = nx.min_cost_flow(G)
    assign: Dict[int, int] = {}
    for i in range(k_new):
        for j in range(n20):
            # MultiDiGraph flow: flow[u][v] is dict of keys -> value
            fv = flow.get(f"k{i}", {}).get(f"r{j}", 0)
            if isinstance(fv, dict):
                amt = sum(fv.values())
            else:
                amt = fv
            if amt > 0:
                assign[i] = j
                break
    # Verify each receiver got >= 1 on req edge
    for j in range(n20):
        ft = flow.get(f"r{j}", {}).get("t", {})
        if isinstance(ft, dict):
            req = ft.get("req", 0)
        else:
            req = ft
        if req < 1:
            raise RuntimeError(f"finer flow: receiver {j} required flow={req}")
    if len(assign) != k_new:
        raise RuntimeError(f"finer flow incomplete {len(assign)}/{k_new}")
    return assign


def min_cost_flow_coarser(overlap_k20_by_knew: np.ndarray) -> Dict[int, int]:
    """Each K20 -> one K_new; each K_new receives >= 1."""
    ov = overlap_k20_by_knew
    n20, k_new = ov.shape
    max_extra = n20 - k_new
    G = nx.MultiDiGraph()
    G.add_node("s", demand=-n20)
    G.add_node("t", demand=n20)
    for i in range(n20):
        G.add_edge("s", f"c{i}", capacity=1, weight=0)
        for j in range(k_new):
            w = int(ov.max() - ov[i, j])
            G.add_edge(f"c{i}", f"p{j}", capacity=1, weight=w)
    for j in range(k_new):
        G.add_edge(f"p{j}", "t", key="req", capacity=1, weight=0)
        if max_extra > 0:
            G.add_edge(f"p{j}", "t", key="opt", capacity=max_extra, weight=10**6)
    flow = nx.min_cost_flow(G)
    assign: Dict[int, int] = {}
    for i in range(n20):
        for j in range(k_new):
            fv = flow.get(f"c{i}", {}).get(f"p{j}", 0)
            if isinstance(fv, dict):
                amt = sum(fv.values())
            else:
                amt = fv
            if amt > 0:
                assign[i] = j
                break
    for j in range(k_new):
        ft = flow.get(f"p{j}", {}).get("t", {})
        if isinstance(ft, dict):
            req = ft.get("req", 0)
        else:
            req = ft
        if req < 1:
            raise RuntimeError(f"coarser flow: parent {j} required flow={req}")
    if len(assign) != n20:
        raise RuntimeError(f"coarser flow incomplete {len(assign)}/{n20}")
    return assign


def side_label(gt: float, gen: float) -> str:
    s = side_of_diagonal(gt, gen)
    return {"below": "Below", "above": "Above", "on": "On"}.get(s, "—")


def flags_from_row(r: pd.Series) -> str:
    low_n = bool(r.get("low_n", False)) or (
        float(r.n_gt) < 30 or float(r.n_gen) < 30
    )
    both_low = float(r.GT) < 0.3 and float(r.GEN) < 0.3
    marks = ""
    if low_n:
        marks += "†"
    if both_low:
        marks += "‡"
    return marks


def cell_side(
    scores: pd.DataFrame, model: str, cid: int
) -> Tuple[str, str, float, float]:
    sc = scores[(scores.Model == model) & (scores.Cluster == cid)]
    if sc.empty:
        return "—", "", float("nan"), float("nan")
    r = sc.iloc[0]
    return side_label(float(r.GT), float(r.GEN)), flags_from_row(r), float(r.GT), float(r.GEN)


def run(repo: Path) -> None:
    exp4 = repo / EXP4_ROOT_REL
    ms = exp4 / "model_seeds_all"
    out = repo / OUT_REL
    rev = repo / "results/revision_r2"
    out.mkdir(parents=True, exist_ok=True)

    hashes_before_path = out / "hashes_before.txt"
    before = hash_tree(rev, hashes_before_path)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    for k in (K20,) + KS_OTHER:
        p = ms / f"clustered_questions_K{k}.csv"
        s = ms / f"scores_K{k}.csv"
        if not p.is_file() or not s.is_file():
            raise SystemExit(
                f"Missing Exp4 artefact for K={k}: clustered={p.is_file()} scores={s.is_file()}. "
                "Stop — do not improvise."
            )

    names20 = load_names(exp4, 20)
    if len(names20) < K20:
        raise SystemExit(f"K=20 names incomplete ({len(names20)}); stop.")

    map20 = load_unique_cluster_map(ms / "clustered_questions_K20.csv")
    ids20 = sorted(map20.keys())
    if ids20 != list(range(K20)):
        raise SystemExit(f"K20 cluster ids unexpected: {ids20}")

    scores_by_k = {k: load_scores(exp4, k) for k in (K20,) + KS_OTHER}

    flow_checks: Dict[str, dict] = {}
    concept_k_info: Dict[int, Dict[int, dict]] = {i: {} for i in range(K20)}
    parent_members: Dict[int, Dict[int, List[int]]] = {}

    for k in KS_OTHER:
        map_k = load_unique_cluster_map(ms / f"clustered_questions_K{k}.csv")
        ids_k = sorted(map_k.keys())
        if ids_k != list(range(k)):
            raise SystemExit(f"K={k} cluster ids unexpected: {ids_k}")

        overlap, jacc, _, _ = overlap_and_jaccard(map20, map_k)
        finer = k > K20

        if finer:
            hung = assignment_finer(overlap)
            try:
                flow = min_cost_flow_finer(overlap)
                match = hung == flow
                flow_checks[f"K{k}"] = {
                    "match": bool(match),
                    "n_disagree": sum(1 for i in hung if hung[i] != flow.get(i)),
                }
            except Exception as e:
                flow = {}
                match = False
                flow_checks[f"K{k}"] = {"match": False, "error": str(e)}

            children: Dict[int, List[int]] = defaultdict(list)
            for knew, k20 in hung.items():
                children[int(k20)].append(int(knew))
            for j in range(K20):
                assigned = children.get(j, [])
                if not assigned:
                    raise RuntimeError(f"K={k}: K20={j} received no child")
                main = max(assigned, key=lambda c: overlap[j, c])
                concept_k_info[j][k] = {
                    "mode": "finer",
                    "assigned_ids": sorted(assigned),
                    "main_id": int(main),
                    "overlap": float(overlap[j, main]),
                    "jaccard": float(jacc[j, main]),
                    "weak": float(jacc[j, main]) < 0.5,
                    "size20": float(len(map20[j])),
                    "overlap_row": overlap[j].copy(),
                }
            rows = []
            for knew, k20 in sorted(hung.items()):
                rows.append(
                    {
                        "K": k,
                        "mode": "finer_one_to_many",
                        "source_cluster_id": knew,
                        "source_role": f"K{k}",
                        "target_cluster_id": k20,
                        "target_role": "K20",
                        "overlap": float(overlap[k20, knew]),
                        "jaccard": float(jacc[k20, knew]),
                        "weak_match_jaccard_lt_0.5": bool(jacc[k20, knew] < 0.5),
                        "is_main_for_target": bool(
                            concept_k_info[k20][k]["main_id"] == knew
                        ),
                        "flow_match_for_K": bool(flow_checks[f"K{k}"].get("match")),
                    }
                )
            pd.DataFrame(rows).to_csv(out / f"assignments_K{k}.csv", index=False)
        else:
            hung = assignment_coarser(overlap)
            try:
                flow = min_cost_flow_coarser(overlap)
                match = hung == flow
                flow_checks[f"K{k}"] = {
                    "match": bool(match),
                    "n_disagree": sum(1 for i in hung if hung[i] != flow.get(i)),
                }
            except Exception as e:
                flow = {}
                flow_checks[f"K{k}"] = {"match": False, "error": str(e)}

            members: Dict[int, List[int]] = defaultdict(list)
            for k20, knew in hung.items():
                members[int(knew)].append(int(k20))
                concept_k_info[int(k20)][k] = {
                    "mode": "coarser",
                    "assigned_ids": [int(knew)],
                    "main_id": int(knew),
                    "overlap": float(overlap[k20, knew]),
                    "jaccard": float(jacc[k20, knew]),
                    "weak": float(jacc[k20, knew]) < 0.5,
                    "size20": float(len(map20[k20])),
                    "siblings": None,
                }
            for knew, kids in members.items():
                for k20 in kids:
                    concept_k_info[k20][k]["siblings"] = sorted(kids)
            parent_members[k] = {int(p): sorted(v) for p, v in members.items()}
            # every parent must receive >= 1
            if sorted(members.keys()) != list(range(k)):
                raise RuntimeError(
                    f"K={k}: parents with members {sorted(members)} != 0..{k-1}"
                )
            rows = []
            for k20, knew in sorted(hung.items()):
                rows.append(
                    {
                        "K": k,
                        "mode": "coarser_many_to_one",
                        "source_cluster_id": k20,
                        "source_role": "K20",
                        "target_cluster_id": knew,
                        "target_role": f"K{k}",
                        "overlap": float(overlap[k20, knew]),
                        "jaccard": float(jacc[k20, knew]),
                        "weak_match_jaccard_lt_0.5": bool(jacc[k20, knew] < 0.5),
                        "k20_sharing_parent": ",".join(
                            str(x) for x in sorted(members[knew])
                        ),
                        "flow_match_for_K": bool(flow_checks[f"K{k}"].get("match")),
                    }
                )
            pd.DataFrame(rows).to_csv(out / f"assignments_K{k}.csv", index=False)

    for j in range(K20):
        concept_k_info[j][K20] = {
            "mode": "self",
            "assigned_ids": [j],
            "main_id": j,
            "overlap": float(len(map20[j])),
            "jaccard": 1.0,
            "weak": False,
            "size20": float(len(map20[j])),
        }

    long_rows = []
    for oid in range(K20):
        cname = names20.get(oid, str(oid))
        for model in MODEL_ORDER:
            side20, _, _, _ = cell_side(scores_by_k[K20], model, oid)
            for k in (10, 15, 20, 25, 30):
                info = concept_k_info[oid][k]
                main = int(info["main_id"])
                side, fl, gt, gen = cell_side(scores_by_k[k], model, main)
                pct_same = ""
                if info["mode"] == "finer":
                    main_side = (
                        side_of_diagonal(gt, gen) if np.isfinite(gt) else "undefined"
                    )
                    size20 = info["size20"]
                    ov_row = info["overlap_row"]
                    same_q = 0.0
                    for knew in info["assigned_ids"]:
                        sc = scores_by_k[k][
                            (scores_by_k[k].Model == model)
                            & (scores_by_k[k].Cluster == knew)
                        ]
                        if sc.empty:
                            continue
                        r = sc.iloc[0]
                        if side_of_diagonal(float(r.GT), float(r.GEN)) == main_side:
                            same_q += float(ov_row[knew])
                    pct_same = f"{100.0 * same_q / size20:.1f}" if size20 > 0 else ""
                cell = side + fl + ("§" if info["weak"] else "")
                long_rows.append(
                    {
                        "concept_id": oid,
                        "concept_name": cname,
                        "model": display(model),
                        "model_raw": model,
                        "K": k,
                        "side": side,
                        "flags": fl,
                        "cell": cell,
                        "assigned_cluster_ids": ",".join(
                            str(x) for x in info["assigned_ids"]
                        ),
                        "main_assigned_id": main,
                        "overlap": info["overlap"],
                        "jaccard": info["jaccard"],
                        "weak_match": bool(info["weak"]),
                        "pct_questions_same_side_assigned": pct_same,
                        "side_at_K20": side20,
                        "same_side_as_K20": side == side20,
                        "siblings_sharing_parent": (
                            ",".join(str(x) for x in (info.get("siblings") or []))
                            if info["mode"] == "coarser"
                            else ""
                        ),
                    }
                )
    long_df = pd.DataFrame(long_rows)

    meta_rows = []
    for oid in range(K20):
        row = {"concept_id": oid, "concept_name": names20.get(oid, str(oid))}
        for model in MODEL_ORDER:
            md = display(model)
            for k in (10, 15, 20, 25, 30):
                sub = long_df[
                    (long_df.concept_id == oid)
                    & (long_df.model_raw == model)
                    & (long_df.K == k)
                ].iloc[0]
                prefix = f"{md} | K={k}"
                row[prefix] = sub.cell
                row[f"{prefix} | assigned"] = sub.assigned_cluster_ids
                row[f"{prefix} | main"] = int(sub.main_assigned_id)
                row[f"{prefix} | overlap"] = sub.overlap
                row[f"{prefix} | jaccard"] = sub.jaccard
                row[f"{prefix} | weak"] = bool(sub.weak_match)
                if k in (25, 30):
                    row[f"{prefix} | pct_same_side"] = (
                        sub.pct_questions_same_side_assigned
                    )
        meta_rows.append(row)

    agree = {"concept_id": "", "concept_name": "% same side as K=20"}
    for model in MODEL_ORDER:
        md = display(model)
        for k in (10, 15, 20, 25, 30):
            sub = long_df[(long_df.model_raw == model) & (long_df.K == k)]
            n_same = int((sub.side == sub.side_at_K20).sum())
            agree[f"{md} | K={k}"] = f"{100.0 * n_same / K20:.0f}%"
    wide = pd.concat([pd.DataFrame(meta_rows), pd.DataFrame([agree])], ignore_index=True)
    wide.to_csv(out / "SuppTableD_hungarian.csv", index=False)

    md_lines = [
        "# Supp Table D (Hungarian assignment)",
        "",
        "Sides from pooled Exp4 `scores_K*.csv` (Fig. 4 pooling). "
        "† = n_gt or n_gen < 30 (or existing `low_n`); ‡ = GT and GEN both < 0.3; "
        "§ = weak match (Jaccard < 0.5).",
        "",
        "Finer K (25, 30): side of the **main** assigned cluster (largest overlap); "
        "`pct_same_side` = % of the K=20 concept’s unique questions in assigned "
        "K-clusters on the same side as that main cluster (not recomputed from pooled pieces).",
        "",
        "Coarser K (10, 15): side of the assigned parent; siblings listed below.",
        "",
    ]
    header = (
        "| ID | Concept | "
        + " | ".join(
            f"{display(m)}@{k}" for m in MODEL_ORDER for k in (10, 15, 20, 25, 30)
        )
        + " |"
    )
    md_lines += [header, "|" + "|".join(["---"] * (2 + 15)) + "|"]
    for oid in range(K20):
        cells = [str(oid), names20.get(oid, str(oid))]
        for model in MODEL_ORDER:
            for k in (10, 15, 20, 25, 30):
                r = long_df[
                    (long_df.concept_id == oid)
                    & (long_df.model_raw == model)
                    & (long_df.K == k)
                ].iloc[0]
                cells.append(str(r.cell))
        md_lines.append("| " + " | ".join(cells) + " |")
    sum_cells = ["", "% same side as K=20"]
    for model in MODEL_ORDER:
        mdn = display(model)
        for k in (10, 15, 20, 25, 30):
            sum_cells.append(str(agree[f"{mdn} | K={k}"]))
    md_lines.append("| " + " | ".join(sum_cells) + " |")

    md_lines += ["", "## Clusters 12 & 13", ""]
    for oid in (12, 13):
        md_lines.append(f"### {oid}: {names20.get(oid, '')}")
        md_lines.append("")
        md_lines.append(
            "| Model | K | side | main_id | assigned | overlap | Jaccard | weak | "
            "pct_same_side | siblings |"
        )
        md_lines.append("|---|---:|---|---:|---|---:|---:|---|---:|---|")
        for model in MODEL_ORDER:
            for k in (10, 15, 20, 25, 30):
                r = long_df[
                    (long_df.concept_id == oid)
                    & (long_df.model_raw == model)
                    & (long_df.K == k)
                ].iloc[0]
                md_lines.append(
                    f"| {display(model)} | {k} | {r.cell} | {int(r.main_assigned_id)} | "
                    f"{r.assigned_cluster_ids} | {r.overlap:.0f} | {r.jaccard:.4f} | "
                    f"{r.weak_match} | {r.pct_questions_same_side_assigned} | "
                    f"{r.siblings_sharing_parent} |"
                )
        md_lines.append("")

    md_lines += ["## Coarser K: K=20 clusters sharing each parent", ""]
    for k in (10, 15):
        md_lines.append(f"### K={k}")
        for parent, kids in sorted(parent_members[k].items()):
            names = ", ".join(f"{c}:{names20.get(c, '')}" for c in kids)
            md_lines.append(f"- parent {parent} ← {names}")
        md_lines.append("")
    (out / "SuppTableD_hungarian.md").write_text("\n".join(md_lines) + "\n")

    table_d_path = repo / TABLE_D_REL
    summary_bits: List[str] = []
    if table_d_path.is_file():
        td = pd.read_csv(table_d_path)
        td_concepts = td[
            td["concept_id"].notna() & (td["concept_name"] != "% same side as K=20")
        ].copy()
        td_concepts["concept_id"] = td_concepts["concept_id"].astype(float).astype(int)
        detail = [
            "# Hungarian vs current Table D",
            "",
            "Current Table D uses greedy best-match (largest forward share), not Hungarian.",
            "",
            "## Side differences (Below/Above ignoring */†/§ marks)",
            "",
        ]
        diffs = []
        n_side_diff = 0
        for _, tr in td_concepts.iterrows():
            oid = int(tr.concept_id)
            for model in MODEL_ORDER:
                mdn = display(model)
                for k in (10, 15, 20, 25, 30):
                    col = f"{mdn} | K={k}"
                    if col not in tr.index:
                        continue
                    cur_raw = str(tr[col])
                    cur_side = (
                        "Below"
                        if cur_raw.startswith("Below")
                        else "Above"
                        if cur_raw.startswith("Above")
                        else cur_raw
                    )
                    hr = long_df[
                        (long_df.concept_id == oid)
                        & (long_df.model_raw == model)
                        & (long_df.K == k)
                    ].iloc[0]
                    if cur_side != hr.side:
                        n_side_diff += 1
                        diffs.append(
                            {
                                "concept_id": oid,
                                "concept_name": names20.get(oid, ""),
                                "model": mdn,
                                "K": k,
                                "tableD_cell": cur_raw,
                                "hungarian_cell": hr.cell,
                                "hungarian_main_id": int(hr.main_assigned_id),
                            }
                        )
                        detail.append(
                            f"- id {oid} ({names20.get(oid, '')}) | {mdn} | K={k}: "
                            f"TableD={cur_raw} vs Hungarian={hr.cell} "
                            f"(main_id={int(hr.main_assigned_id)})"
                        )
        if n_side_diff == 0:
            detail.append("_No side differences._")

        detail += ["", "## Mapping: greedy best-match vs Hungarian main id", ""]
        greedy = all_concept_match_info(exp4)
        n_map = 0
        for oid in range(K20):
            for k in KS_OTHER:
                g = int(greedy[k][oid]["cluster_id"])
                h = int(concept_k_info[oid][k]["main_id"])
                if g != h:
                    n_map += 1
                    detail.append(
                        f"- id {oid} K={k}: greedy_best={g} vs hungarian_main={h} "
                        f"(assigned={concept_k_info[oid][k]['assigned_ids']})"
                    )
        if n_map == 0:
            detail.append(
                "_Hungarian main id matches greedy best-match for all concepts/K._"
            )

        pct_rows = []
        for model in MODEL_ORDER:
            mdn = display(model)
            pct_rows.append(
                mdn
                + ": "
                + ", ".join(
                    f"K{kk}={agree[f'{mdn} | K={kk}']}" for kk in (10, 15, 20, 25, 30)
                )
            )
        flow_ok = all(bool(v.get("match")) for v in flow_checks.values())
        plain = [
            f"Hungarian % same-side as K=20 — {'; '.join(pct_rows)}.",
            f"Side cells differing from current Table D: {n_side_diff}; "
            f"main-id differing from greedy best-match: {n_map}.",
            f"networkx.min_cost_flow cross-check matches Hungarian for all K: {flow_ok}.",
        ]
        detail += ["", "## 3-line plain summary", ""]
        detail += [f"{i}. {t}" for i, t in enumerate(plain, 1)]
        (out / "hungarian_vs_current_tableD.md").write_text("\n".join(detail) + "\n")
        if diffs:
            pd.DataFrame(diffs).to_csv(
                out / "side_differences_vs_tableD.csv", index=False
            )
        summary_bits = plain
    else:
        (out / "hungarian_vs_current_tableD.md").write_text(
            "Current Table D not found; skipped comparison.\n"
        )
        summary_bits = ["Table D missing; no comparison."]

    cmd = (
        "cd /gpfs/data/oermannlab/users/rd3571/ICARE_score_rev_stats && "
        "srun --partition=oermannlab --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G "
        "--time=00:30:00 --job-name=exp4-hungarian "
        "/gpfs/data/oermannlab/users/rd3571/.conda/envs/radeval311/bin/python "
        "src/question_categorization_and_analysis/cluster_k_hungarian.py"
    )
    (out / "README").write_text(
        "# Exp4 Hungarian cluster mapping\n\n"
        "Maps K∈{10,15,25,30} to K=20 (Fig. 4) with "
        "`scipy.optimize.linear_sum_assignment` (required + optional column copies) "
        "and cross-checks with `networkx.min_cost_flow` (required edge capacity 1 per "
        "receiving cluster).\n\n"
        "## Exact command\n\n"
        f"```bash\n{cmd}\n```\n"
    )

    meta = {
        "commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "input_paths": {
            "clustered_questions": str(ms / "clustered_questions_K{K}.csv"),
            "scores": str(ms / "scores_K{K}.csv"),
            "names": str(ms / "cluster_names_K20.json"),
            "table_d": str(table_d_path),
        },
        "flow_cross_check": flow_checks,
        "flow_all_match": all(bool(v.get("match")) for v in flow_checks.values()),
        "summary": summary_bits,
        "M_required": M_REQUIRED,
        "weak_jaccard_threshold": 0.5,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    after = hash_tree(rev, out / "hashes_after.txt")
    if before != after:
        keys = sorted(set(before) | set(after))
        changed = [k for k in keys if before.get(k) != after.get(k)]
        raise SystemExit(
            "ERROR: existing results/revision_r2 files changed:\n"
            + "\n".join(changed[:80])
        )

    print("HASH_OK: existing results/revision_r2 file hashes unchanged")
    print("FLOW_ALL_MATCH:", meta["flow_all_match"])
    print("FLOW_CHECKS:", json.dumps(flow_checks))
    for line in summary_bits:
        print("SUMMARY:", line)
    print("PCT_SAME_SIDE:")
    for model in MODEL_ORDER:
        mdn = display(model)
        print(" ", mdn, {kk: agree[f"{mdn} | K={kk}"] for kk in (10, 15, 20, 25, 30)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO)
    args = p.parse_args()
    run(args.repo.resolve())


if __name__ == "__main__":
    main()
