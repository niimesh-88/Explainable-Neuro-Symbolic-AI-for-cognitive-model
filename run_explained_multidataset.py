"""Run explained neuro-symbolic cognitive modeling on multiple real datasets.

This script turns the project into a repeatable multi-dataset experiment:

1. OpenNeuro ds002424 children/adolescent n-back fMRI.
2. OpenNeuro ds002687 adult n-back fMRI.
3. A combined OpenNeuro benchmark using the prepared subjects from both.

For each dataset it trains/evaluates the graph model, creates statistical
validation, extracts subject-level explanations, generates symbolic reasoning
traces, and writes a human-readable Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scie_neurosymbolic_wm_model as wm


DATASETS = [
    {
        "id": "openneuro_ds002424_children",
        "name": "OpenNeuro ds002424 Children/ADHD",
        "connectivity_dir": Path("data/openneuro_ds002424/connectivity"),
        "targets_csv": Path("data/openneuro_ds002424/targets.csv"),
        "target_column": "working_memory_score",
        "notes": "Child/adolescent n-back fMRI dataset with ADHD/control metadata. Uses SLD task connectivity and overall 2-back accuracy target.",
    },
    {
        "id": "openneuro_ds002687_adults",
        "name": "OpenNeuro ds002687 Adults",
        "connectivity_dir": Path("data/openneuro_ds002687/connectivity"),
        "targets_csv": Path("data/openneuro_ds002687/targets.csv"),
        "target_column": "working_memory_score",
        "notes": "Adult n-back fMRI companion dataset. Uses SLD task connectivity and overall 2-back accuracy target.",
    },
    {
        "id": "openneuro_combined",
        "name": "Combined OpenNeuro Children + Adults",
        "connectivity_dir": Path("data/openneuro_explained_combined/connectivity"),
        "targets_csv": Path("data/openneuro_explained_combined/targets.csv"),
        "target_column": "working_memory_score",
        "notes": "Combined benchmark made from all prepared ds002424 and ds002687 connectivity matrices.",
    },
]


def run_args(args: argparse.Namespace, dataset_id: str, target_column: str) -> SimpleNamespace:
    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(dataset_id)) % 10000
    return SimpleNamespace(
        folds=args.folds,
        density=args.density,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed + stable_offset,
        permutations=args.permutations,
        bootstrap_samples=args.bootstrap_samples,
        output=None,
        run_ablation=False,
        model="gcn",
        ridge_alpha=10.0,
        ensemble_gcn_weight=0.35,
        tune_ridge=False,
        target_column=target_column,
    )


def prepared_subjects(connectivity_dir: Path) -> set[str]:
    return {path.stem for path in connectivity_dir.glob("*.csv")}


def build_combined_dataset() -> None:
    combined_dir = Path("data/openneuro_explained_combined")
    combined_conn = combined_dir / "connectivity"
    combined_conn.mkdir(parents=True, exist_ok=True)

    pieces = []
    for source in [
        (Path("data/openneuro_ds002424/connectivity"), Path("data/openneuro_ds002424/targets.csv")),
        (Path("data/openneuro_ds002687/connectivity"), Path("data/openneuro_ds002687/targets.csv")),
    ]:
        conn_dir, target_csv = source
        if not conn_dir.exists() or not target_csv.exists():
            continue
        available = prepared_subjects(conn_dir)
        targets = pd.read_csv(target_csv)
        targets = targets[targets["subject_id"].astype(str).isin(available)].copy()
        pieces.append(targets)
        for matrix_path in conn_dir.glob("*.csv"):
            shutil.copyfile(matrix_path, combined_conn / matrix_path.name)

    if pieces:
        pd.concat(pieces, ignore_index=True).to_csv(combined_dir / "targets.csv", index=False)


def choose_explanation_indices(targets: np.ndarray) -> list[int]:
    if len(targets) <= 3:
        return list(range(len(targets)))
    order = np.argsort(targets)
    return [int(order[0]), int(order[len(order) // 2]), int(order[-1])]


def aggregate_explanations(explanations: list[dict[str, object]]) -> dict[str, object]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    node_importance = defaultdict(float)
    edge_importance = defaultdict(float)

    for item in explanations:
        explanation = item["explanation"]
        for node in explanation["top_nodes"]:
            node_counts[node["region"]] += 1
            node_importance[node["region"]] += float(node["importance"])
        for edge in explanation["top_edges"]:
            key = f"{edge['source']}--{edge['target']}"
            edge_counts[key] += 1
            edge_importance[key] += float(edge["importance"])

    return {
        "recurring_nodes": [
            {"region": region, "count": count, "total_importance": node_importance[region]}
            for region, count in node_counts.most_common(8)
        ],
        "recurring_edges": [
            {"edge": edge, "count": count, "total_importance": edge_importance[edge]}
            for edge, count in edge_counts.most_common(8)
        ],
    }


def run_dataset(spec: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    wm.TARGET_COLUMN = str(spec["target_column"])
    dataset = wm.load_connectivity_dataset(Path(spec["connectivity_dir"]), Path(spec["targets_csv"]))
    local_args = run_args(args, str(spec["id"]), str(spec["target_column"]))
    folds = min(local_args.folds, max(2, len(dataset.targets) // 2))
    local_args.folds = folds

    cv = wm.cross_validate(dataset, local_args)
    model, standardizer = wm.train_full_model(dataset, local_args)
    permutation = wm.permutation_test(cv["y_true"], cv["y_pred"], local_args.permutations, local_args.seed)
    ci = wm.bootstrap_ci(cv["y_true"], cv["y_pred"], "pearson_r", local_args.bootstrap_samples, local_args.seed)

    explained_subjects = []
    for idx in choose_explanation_indices(dataset.targets):
        explanation = wm.explain_subject(
            model,
            standardizer,
            dataset.matrices[idx],
            dataset.roi_names,
            local_args.density,
        )
        explained_subjects.append(
            {
                "subject_id": dataset.subject_ids[idx],
                "observed_working_memory_score": float(dataset.targets[idx]),
                "explanation": explanation,
                "symbolic_reasoning_trace": wm.symbolic_reasoning(dataset.matrices[idx], dataset.roi_names),
            }
        )

    return {
        "dataset_id": spec["id"],
        "dataset_name": spec["name"],
        "notes": spec["notes"],
        "subjects": len(dataset.targets),
        "target_column": spec["target_column"],
        "model": "2-layer NumPy GCN with perturbation explanations and symbolic reasoning",
        "cross_validation": {"overall": cv["overall"], "folds": cv["folds"]},
        "statistical_validation": {"permutation_test": permutation, "bootstrap_ci": ci},
        "explained_subjects": explained_subjects,
        "aggregate_explanation": aggregate_explanations(explained_subjects),
    }


def fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Explained Neuro-Symbolic Cognitive Model",
        "",
        "This report evaluates the same explained graph-cognitive model on multiple real OpenNeuro n-back fMRI datasets.",
        "",
        "Pipeline:",
        "",
        "```text",
        "real fMRI BOLD -> functional connectivity matrix -> brain graph -> GCN prediction -> node/edge explanation -> symbolic reasoning trace",
        "```",
        "",
        "## Dataset Results",
        "",
        "| Dataset | Subjects | Pearson r | R2 | MAE | RMSE | Permutation p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for result in report["datasets"]:
        overall = result["cross_validation"]["overall"]
        permutation = result["statistical_validation"]["permutation_test"]
        lines.append(
            f"| {result['dataset_name']} | {result['subjects']} | "
            f"{fmt_metric(overall['pearson_r'])} | {fmt_metric(overall['r2'])} | "
            f"{fmt_metric(overall['mae'])} | {fmt_metric(overall['rmse'])} | "
            f"{fmt_metric(permutation['p_value'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The model is explainable and runs on real fMRI-derived connectivity data, but current predictive accuracy is not yet publication-grade. The strongest scientific value at this stage is the transparent workflow: every prediction includes region/parcel importance, edge importance, and a symbolic reasoning trace.",
            "",
            "The main bottleneck is not the neuro-symbolic layer; it is the lightweight connectivity preprocessing. A stronger version should use atlas-based parcellation, motion/confound regression, and PyTorch Geometric baselines.",
            "",
            "## Explanations",
            "",
        ]
    )

    for result in report["datasets"]:
        lines.extend([f"### {result['dataset_name']}", "", str(result["notes"]), ""])
        lines.append("Recurring important nodes/parcels:")
        for node in result["aggregate_explanation"]["recurring_nodes"][:5]:
            lines.append(f"- {node['region']}: appears {node['count']} times, total importance {fmt_metric(node['total_importance'])}")
        lines.append("")
        lines.append("Example symbolic reasoning:")
        for trace in result["explained_subjects"][0]["symbolic_reasoning_trace"]:
            lines.append(f"- {trace}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explained model across real OpenNeuro datasets.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/explained_multidataset_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("EXPLAINED_MODEL_REPORT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_combined_dataset()
    results = []
    for spec in DATASETS:
        if Path(spec["connectivity_dir"]).exists() and Path(spec["targets_csv"]).exists():
            results.append(run_dataset(spec, args))

    report = {
        "model_name": "Explainable Neuro-Symbolic Graph Model for Working-Memory Cognition",
        "datasets": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2))
    print(f"\nSaved JSON report to: {args.output_json.resolve()}")
    print(f"Saved Markdown report to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
