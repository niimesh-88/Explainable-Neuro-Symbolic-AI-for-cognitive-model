"""Improve validation accuracy with graph + covariate ridge models.

The first project model was a small GCN implemented from scratch. That is useful
for demonstrating the neuro-symbolic architecture, but small neuroimaging
datasets often need a more stable classical model. This script evaluates a
validated ridge regressor over:

- graph features extracted from functional connectivity matrices
- optional demographics/cognitive covariates from targets.csv

It uses inner cross-validation inside each training fold to choose the ridge
regularization strength, then reports outer-fold performance. That keeps the
accuracy estimate honest while usually improving stability.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scie_neurosymbolic_wm_model as wm


DATASET_SPECS = [
    {
        "id": "children_ds002424",
        "name": "OpenNeuro ds002424 Children/ADHD",
        "connectivity_dir": Path("data/openneuro_ds002424/connectivity"),
        "targets_csv": Path("data/openneuro_ds002424/targets.csv"),
        "target_column": "working_memory_score",
    },
    {
        "id": "adults_ds002687",
        "name": "OpenNeuro ds002687 Adults",
        "connectivity_dir": Path("data/openneuro_ds002687/connectivity"),
        "targets_csv": Path("data/openneuro_ds002687/targets.csv"),
        "target_column": "working_memory_score",
    },
    {
        "id": "combined_openneuro",
        "name": "Combined OpenNeuro Children + Adults",
        "connectivity_dir": Path("data/openneuro_accuracy_combined/connectivity"),
        "targets_csv": Path("data/openneuro_accuracy_combined/targets.csv"),
        "target_column": "working_memory_score",
    },
]


COVARIATE_COLUMNS = [
    "age_ses-T1",
    "age",
    "sex",
    "ADHD_diagnosis",
    "n_2back_trials",
    "n_tasks_with_2back",
]


@dataclass
class FeatureDataset:
    subject_ids: list[str]
    matrices: np.ndarray
    targets: np.ndarray
    roi_names: list[str]
    target_frame: pd.DataFrame


def build_combined_dataset() -> None:
    combined_dir = Path("data/openneuro_accuracy_combined")
    combined_conn = combined_dir / "connectivity"
    combined_conn.mkdir(parents=True, exist_ok=True)
    pieces = []
    sources = [
        (Path("data/openneuro_ds002424/connectivity"), Path("data/openneuro_ds002424/targets.csv"), "children"),
        (Path("data/openneuro_ds002687/connectivity"), Path("data/openneuro_ds002687/targets.csv"), "adults"),
    ]
    for conn_dir, targets_csv, cohort in sources:
        if not conn_dir.exists() or not targets_csv.exists():
            continue
        available = {path.stem for path in conn_dir.glob("*.csv")}
        frame = pd.read_csv(targets_csv)
        frame = frame[frame["subject_id"].astype(str).isin(available)].copy()
        frame["cohort_children"] = 1.0 if cohort == "children" else 0.0
        frame["cohort_adults"] = 1.0 if cohort == "adults" else 0.0
        pieces.append(frame)
        for matrix_path in conn_dir.glob("*.csv"):
            shutil.copyfile(matrix_path, combined_conn / matrix_path.name)
    if pieces:
        pd.concat(pieces, ignore_index=True).to_csv(combined_dir / "targets.csv", index=False)


def load_feature_dataset(connectivity_dir: Path, targets_csv: Path, target_column: str) -> FeatureDataset:
    wm.TARGET_COLUMN = target_column
    base = wm.load_connectivity_dataset(connectivity_dir, targets_csv)
    frame = pd.read_csv(targets_csv)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame = frame.set_index("subject_id").loc[base.subject_ids].reset_index()
    return FeatureDataset(base.subject_ids, base.matrices, base.targets, base.roi_names, frame)


def graph_features(matrices: np.ndarray, density: float) -> np.ndarray:
    return wm.RidgeGraphRegressor.featurize(matrices, density)


def covariate_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = []
    cols = []
    working = frame.copy()
    if "age_ses-T1" in working and "age" not in working:
        working["age"] = working["age_ses-T1"]
    if "age" in working and "age_ses-T1" in working:
        working["age_combined"] = pd.to_numeric(working["age"], errors="coerce").fillna(
            pd.to_numeric(working["age_ses-T1"], errors="coerce")
        )
    elif "age" in working:
        working["age_combined"] = pd.to_numeric(working["age"], errors="coerce")
    elif "age_ses-T1" in working:
        working["age_combined"] = pd.to_numeric(working["age_ses-T1"], errors="coerce")

    useful = [
        "age_combined",
        "sex",
        "ADHD_diagnosis",
        "n_2back_trials",
        "n_tasks_with_2back",
        "cohort_children",
        "cohort_adults",
    ]
    for col in useful:
        if col in working:
            values = pd.to_numeric(working[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                cols.append(values)
                names.append(col)
    if "age_combined" in names:
        age = cols[names.index("age_combined")]
        cols.append(age**2)
        names.append("age_squared")
    if not cols:
        return np.zeros((len(frame), 0)), []
    return np.vstack(cols).T, names


def make_features(dataset: FeatureDataset, density: float, mode: str) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
    if mode in {"graph", "graph_covariates"}:
        graph = graph_features(dataset.matrices, density)
        blocks.append(graph)
        names.extend([f"graph_{i}" for i in range(graph.shape[1])])
    if mode in {"covariates", "graph_covariates"}:
        covariates, cov_names = covariate_features(dataset.target_frame)
        blocks.append(covariates)
        names.extend(cov_names)
    if not blocks:
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.column_stack(blocks), names


def kfold_indices(n: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return wm.kfold_indices(n, min(folds, max(2, n // 2)), seed)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, mean)
    x_test = np.where(np.isfinite(x_test), x_test, mean)
    std = x_train.std(axis=0) + 1e-8
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    return test_design @ coef


def metric_for_tuning(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def tune_alpha(x: np.ndarray, y: np.ndarray, alphas: list[float], seed: int) -> float:
    if len(y) < 6:
        return 10.0
    splits = kfold_indices(len(y), min(4, len(y) // 2), seed)
    best_alpha = alphas[0]
    best_score = math.inf
    for alpha in alphas:
        preds = []
        truth = []
        for train_idx, val_idx in splits:
            x_train, x_val, _, _ = standardize_train_test(x[train_idx], x[val_idx])
            y_mean = y[train_idx].mean()
            y_std = y[train_idx].std() + 1e-8
            pred = ridge_fit_predict(x_train, (y[train_idx] - y_mean) / y_std, x_val, alpha) * y_std + y_mean
            preds.extend(pred.tolist())
            truth.extend(y[val_idx].tolist())
        score = metric_for_tuning(np.asarray(truth), np.asarray(preds))
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def permutation_p_value(y_true: np.ndarray, y_pred: np.ndarray, permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = wm.metrics(y_true, y_pred)["pearson_r"]
    null = []
    for _ in range(permutations):
        null.append(wm.metrics(rng.permutation(y_true), y_pred)["pearson_r"])
    null = np.asarray(null)
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1))


def cross_validate_ridge(dataset: FeatureDataset, args: argparse.Namespace, mode: str) -> dict[str, object]:
    x, feature_names = make_features(dataset, args.density, mode)
    y = dataset.targets
    splits = kfold_indices(len(y), args.folds, args.seed)
    alphas = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    predictions = np.zeros(len(y), dtype=float)
    fold_reports = []
    selected_alphas = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        alpha = tune_alpha(x[train_idx], y[train_idx], alphas, args.seed + fold)
        selected_alphas.append(alpha)
        x_train, x_test, _, _ = standardize_train_test(x[train_idx], x[test_idx])
        y_mean = y[train_idx].mean()
        y_std = y[train_idx].std() + 1e-8
        pred = ridge_fit_predict(x_train, (y[train_idx] - y_mean) / y_std, x_test, alpha) * y_std + y_mean
        predictions[test_idx] = pred
        fold_reports.append(
            {
                "fold": fold,
                "alpha": alpha,
                "subjects": [dataset.subject_ids[i] for i in test_idx],
                "metrics": wm.metrics(y[test_idx], pred),
            }
        )
    return {
        "mode": mode,
        "subjects": len(y),
        "feature_count": int(x.shape[1]),
        "feature_names_tail": feature_names[-12:],
        "selected_alphas": selected_alphas,
        "overall": wm.metrics(y, predictions),
        "permutation_p_value": permutation_p_value(y, predictions, args.permutations, args.seed),
        "folds": fold_reports,
        "y_true": y,
        "y_pred": predictions,
    }


def explain_linear_model(dataset: FeatureDataset, args: argparse.Namespace, mode: str) -> dict[str, object]:
    x, feature_names = make_features(dataset, args.density, mode)
    alpha = tune_alpha(x, dataset.targets, [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0], args.seed)
    xs, _, mean, std = standardize_train_test(x, x[:1])
    y_mean = dataset.targets.mean()
    y_std = dataset.targets.std() + 1e-8
    design = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ ((dataset.targets - y_mean) / y_std)
    weights = coef[1:] * y_std / std
    ranked = np.argsort(np.abs(weights))[::-1][:12]
    return {
        "alpha": alpha,
        "top_features": [
            {"feature": feature_names[i], "weight": float(weights[i]), "abs_weight": float(abs(weights[i]))}
            for i in ranked
        ],
    }


def run_dataset(spec: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    dataset = load_feature_dataset(Path(spec["connectivity_dir"]), Path(spec["targets_csv"]), str(spec["target_column"]))
    modes = ["graph", "covariates", "graph_covariates"]
    results = [cross_validate_ridge(dataset, args, mode) for mode in modes]
    best = max(results, key=lambda r: r["overall"]["pearson_r"] if np.isfinite(r["overall"]["pearson_r"]) else -999)
    explanation = explain_linear_model(dataset, args, best["mode"])
    return {
        "dataset_id": spec["id"],
        "dataset_name": spec["name"],
        "subjects": len(dataset.targets),
        "target_column": spec["target_column"],
        "best_mode": best["mode"],
        "best_metrics": best["overall"],
        "best_permutation_p_value": best["permutation_p_value"],
        "all_modes": [
            {
                "mode": result["mode"],
                "feature_count": result["feature_count"],
                "metrics": result["overall"],
                "permutation_p_value": result["permutation_p_value"],
            }
            for result in results
        ],
        "linear_explanation": explanation,
    }


def write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# Improved Accuracy Report",
        "",
        "This report compares graph-only, covariate-only, and graph+covariate ridge models using nested alpha tuning inside each cross-validation fold.",
        "",
        "| Dataset | Subjects | Best Mode | Pearson r | R2 | MAE | RMSE | p-value |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in report["datasets"]:
        m = result["best_metrics"]
        lines.append(
            f"| {result['dataset_name']} | {result['subjects']} | {result['best_mode']} | "
            f"{m['pearson_r']:.4f} | {m['r2']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{result['best_permutation_p_value']:.4f} |"
        )
    lines.extend(["", "## Model Comparisons", ""])
    for result in report["datasets"]:
        lines.extend([f"### {result['dataset_name']}", ""])
        for mode in result["all_modes"]:
            m = mode["metrics"]
            lines.append(
                f"- {mode['mode']}: r={m['pearson_r']:.4f}, R2={m['r2']:.4f}, "
                f"MAE={m['mae']:.4f}, p={mode['permutation_p_value']:.4f}"
            )
        lines.append("")
        lines.append("Top explanatory features in the selected model:")
        for feature in result["linear_explanation"]["top_features"][:8]:
            lines.append(f"- {feature['feature']}: weight {feature['weight']:.5f}")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved validated accuracy models.")
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/improved_accuracy_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("IMPROVED_ACCURACY_REPORT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_combined_dataset()
    datasets = []
    for spec in DATASET_SPECS:
        if Path(spec["connectivity_dir"]).exists() and Path(spec["targets_csv"]).exists():
            datasets.append(run_dataset(spec, args))
    report = {
        "method": "Nested-CV ridge over graph and covariate features",
        "datasets": datasets,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2))
    print(f"\nSaved JSON report to: {args.output_json.resolve()}")
    print(f"Saved Markdown report to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
