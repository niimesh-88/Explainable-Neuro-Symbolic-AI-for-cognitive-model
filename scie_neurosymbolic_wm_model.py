"""Research prototype: explainable neuro-symbolic GCN for working memory.

This script implements the architecture described in the strategy document:

    connectivity matrix -> graph construction -> graph convolution model
    -> explainability -> symbolic cognitive reasoning

It is intentionally dependency-light so it can run on this machine today with
NumPy and pandas only. The model is a real trainable graph-convolution regressor
implemented from first principles, with cross-validation, permutation testing,
bootstrap confidence intervals, ablations, and explanation traces.

For real HCP-style use, provide:
  --connectivity-dir path/to/csv_matrices
  --targets-csv path/to/targets.csv

The targets CSV must contain columns: subject_id, working_memory_score.
Each matrix filename should match a subject_id, for example 100307.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROI_NAMES = [
    "left_dlpfc",
    "right_dlpfc",
    "left_parietal",
    "right_parietal",
    "anterior_cingulate",
    "left_hippocampus",
    "right_hippocampus",
    "visual_cortex",
    "motor_cortex",
    "default_mode",
]

FRONTO_PARIENTAL = {"left_dlpfc", "right_dlpfc", "left_parietal", "right_parietal"}
HIPPOCAMPAL = {"left_hippocampus", "right_hippocampus"}
CONTROL_NETWORK = {"anterior_cingulate", "left_dlpfc", "right_dlpfc"}


@dataclass
class Dataset:
    subject_ids: list[str]
    matrices: np.ndarray
    targets: np.ndarray
    roi_names: list[str]


@dataclass
class Standardizer:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def threshold_graph(matrix: np.ndarray, density: float) -> np.ndarray:
    """Keep strongest absolute connections at the requested graph density."""
    n = matrix.shape[0]
    upper = np.triu_indices(n, k=1)
    weights = np.abs(matrix[upper])
    edge_count = max(1, int(round(density * len(weights))))
    cutoff = np.partition(weights, -edge_count)[-edge_count]
    adjacency = np.where(np.abs(matrix) >= cutoff, matrix, 0.0)
    adjacency = symmetrize(adjacency)
    return adjacency


def normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a_hat = np.abs(adjacency) + np.eye(adjacency.shape[0])
    degree = np.sum(a_hat, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
    return d_inv_sqrt @ a_hat @ d_inv_sqrt


def node_features(matrix: np.ndarray) -> np.ndarray:
    abs_m = np.abs(matrix)
    strength = abs_m.sum(axis=1)
    signed_strength = matrix.sum(axis=1)
    degree = (abs_m > 1e-8).sum(axis=1)
    clustering_proxy = np.array(
        [
            abs_m[i] @ abs_m @ abs_m[i].T / max(strength[i] ** 2, 1e-8)
            for i in range(matrix.shape[0])
        ]
    )
    features = np.stack([strength, signed_strength, degree, clustering_proxy], axis=1)
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)


def generate_synthetic_dataset(subjects: int, seed: int) -> Dataset:
    """Create HCP-like connectivity matrices with a known WM signal.

    This is not a substitute for HCP data. It is a reproducible test fixture that
    lets the complete research pipeline run before protected neuroimaging data is
    available.
    """
    rng = set_seed(seed)
    n = len(ROI_NAMES)
    matrices = []
    targets = []
    fp_idx = [ROI_NAMES.index(x) for x in FRONTO_PARIENTAL]
    hip_idx = [ROI_NAMES.index(x) for x in HIPPOCAMPAL]
    ctrl_idx = [ROI_NAMES.index(x) for x in CONTROL_NETWORK]

    for i in range(subjects):
        latent_wm = rng.normal()
        noise = rng.normal(scale=0.18, size=(n, n))
        base = rng.normal(scale=0.16, size=(n, n))
        matrix = symmetrize(base + noise)

        for a in fp_idx:
            for b in fp_idx:
                if a != b:
                    matrix[a, b] += 0.33 * latent_wm + rng.normal(scale=0.05)
        for a in ctrl_idx:
            for b in fp_idx:
                if a != b:
                    matrix[a, b] += 0.22 * latent_wm + rng.normal(scale=0.05)
        for a in hip_idx:
            for b in fp_idx:
                matrix[a, b] += 0.13 * latent_wm + rng.normal(scale=0.06)

        matrix = np.tanh(symmetrize(matrix))
        fp_signal = np.mean(matrix[np.ix_(fp_idx, fp_idx)])
        ctrl_signal = np.mean(np.abs(matrix[np.ix_(ctrl_idx, fp_idx)]))
        hip_signal = np.mean(np.abs(matrix[np.ix_(hip_idx, fp_idx)]))
        wm_score = 100 + 13.0 * latent_wm + 18.0 * fp_signal + 7.0 * ctrl_signal + 3.0 * hip_signal
        wm_score += rng.normal(scale=3.5)

        matrices.append(matrix)
        targets.append(wm_score)

    return Dataset(
        subject_ids=[f"synthetic_{i:04d}" for i in range(subjects)],
        matrices=np.asarray(matrices),
        targets=np.asarray(targets, dtype=float),
        roi_names=ROI_NAMES.copy(),
    )


def load_connectivity_dataset(connectivity_dir: Path, targets_csv: Path) -> Dataset:
    targets_df = pd.read_csv(targets_csv)
    required = {"subject_id"}
    if not required.issubset(targets_df.columns):
        raise ValueError(f"targets CSV must include columns: {sorted(required)}")
    if TARGET_COLUMN not in targets_df.columns:
        raise ValueError(f"targets CSV must include target column: {TARGET_COLUMN}")

    subject_ids: list[str] = []
    matrices: list[np.ndarray] = []
    targets: list[float] = []
    roi_names: list[str] | None = None

    for _, row in targets_df.iterrows():
        subject_id = str(row["subject_id"])
        matrix_path = connectivity_dir / f"{subject_id}.csv"
        if not matrix_path.exists():
            continue
        frame = pd.read_csv(matrix_path, index_col=0)
        if roi_names is None:
            roi_names = [str(x) for x in frame.columns]
        matrix = symmetrize(frame.to_numpy(dtype=float))
        subject_ids.append(subject_id)
        matrices.append(matrix)
        targets.append(float(row[TARGET_COLUMN]))

    if not matrices:
        raise ValueError("No matching subject matrices found for targets CSV.")
    return Dataset(subject_ids, np.asarray(matrices), np.asarray(targets), roi_names or ROI_NAMES.copy())


class GCNRegressor:
    """Two-layer graph convolution with global mean pooling and linear output."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int, lr: float, weight_decay: float):
        rng = set_seed(seed)
        self.w0 = rng.normal(scale=math.sqrt(2 / input_dim), size=(input_dim, hidden_dim))
        self.w1 = rng.normal(scale=math.sqrt(2 / hidden_dim), size=(hidden_dim, hidden_dim))
        self.w_out = rng.normal(scale=math.sqrt(2 / hidden_dim), size=(hidden_dim, 1))
        self.b_out = np.zeros(1)
        self.lr = lr
        self.weight_decay = weight_decay

    def _forward(self, features: np.ndarray, adj_norm: np.ndarray) -> dict[str, np.ndarray]:
        z1 = adj_norm @ features @ self.w0
        h1 = np.maximum(z1, 0.0)
        z2 = adj_norm @ h1 @ self.w1
        h2 = np.maximum(z2, 0.0)
        pooled = h2.mean(axis=0, keepdims=True)
        y_hat = pooled @ self.w_out + self.b_out
        return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "pooled": pooled, "y_hat": y_hat}

    def predict_one(self, matrix: np.ndarray, density: float) -> float:
        adjacency = threshold_graph(matrix, density)
        cache = self._forward(node_features(adjacency), normalized_adjacency(adjacency))
        return float(cache["y_hat"][0, 0])

    def predict(self, matrices: np.ndarray, density: float) -> np.ndarray:
        return np.asarray([self.predict_one(m, density) for m in matrices])

    def fit(self, matrices: np.ndarray, y: np.ndarray, density: float, epochs: int) -> list[float]:
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            order = np.random.permutation(len(matrices))
            for idx in order:
                adjacency = threshold_graph(matrices[idx], density)
                x = node_features(adjacency)
                a = normalized_adjacency(adjacency)
                cache = self._forward(x, a)
                pred = cache["y_hat"]
                err = pred[0, 0] - y[idx]
                epoch_loss += err * err

                d_y = np.array([[2.0 * err]])
                d_w_out = cache["pooled"].T @ d_y + self.weight_decay * self.w_out
                d_b_out = d_y.ravel()
                d_pooled = d_y @ self.w_out.T
                d_h2 = np.repeat(d_pooled / x.shape[0], x.shape[0], axis=0)
                d_z2 = d_h2 * (cache["z2"] > 0.0)
                d_w1 = cache["h1"].T @ a.T @ d_z2 + self.weight_decay * self.w1
                d_h1 = a.T @ d_z2 @ self.w1.T
                d_z1 = d_h1 * (cache["z1"] > 0.0)
                d_w0 = x.T @ a.T @ d_z1 + self.weight_decay * self.w0

                self.w_out -= self.lr * d_w_out
                self.b_out -= self.lr * d_b_out
                self.w1 -= self.lr * d_w1
                self.w0 -= self.lr * d_w0
            losses.append(epoch_loss / len(matrices))
        return losses


class RidgeGraphRegressor:
    """Closed-form ridge regression over interpretable graph features.

    This is often stronger than a neural model for small neuroimaging samples.
    It keeps the same brain-graph input, but uses global graph statistics and
    upper-triangle connectivity weights as regularized predictors.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @staticmethod
    def featurize_one(matrix: np.ndarray, density: float) -> np.ndarray:
        adjacency = threshold_graph(matrix, density)
        abs_a = np.abs(adjacency)
        n = matrix.shape[0]
        upper = np.triu_indices(n, k=1)
        edge_values = adjacency[upper]
        strengths = abs_a.sum(axis=1)
        signed_strengths = adjacency.sum(axis=1)
        degrees = (abs_a > 1e-8).sum(axis=1)
        global_features = np.asarray(
            [
                np.mean(edge_values),
                np.std(edge_values),
                np.mean(np.abs(edge_values)),
                np.max(np.abs(edge_values)),
                np.mean(strengths),
                np.std(strengths),
                np.mean(signed_strengths),
                np.std(signed_strengths),
                np.mean(degrees),
                np.std(degrees),
            ],
            dtype=float,
        )
        return np.concatenate([edge_values, strengths, signed_strengths, degrees, global_features])

    @classmethod
    def featurize(cls, matrices: np.ndarray, density: float) -> np.ndarray:
        return np.vstack([cls.featurize_one(matrix, density) for matrix in matrices])

    def fit(self, matrices: np.ndarray, y: np.ndarray, density: float) -> "RidgeGraphRegressor":
        x = self.featurize(matrices, density)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-8
        xs = (x - self.x_mean) / self.x_std
        design = np.column_stack([np.ones(len(xs)), xs])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        return self

    def predict(self, matrices: np.ndarray, density: float) -> np.ndarray:
        if self.coef is None or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Model is not fitted.")
        x = self.featurize(matrices, density)
        xs = (x - self.x_mean) / self.x_std
        design = np.column_stack([np.ones(len(xs)), xs])
        return design @ self.coef


TARGET_COLUMN = "working_memory_score"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": float(r2), "pearson_r": corr}


def kfold_indices(n: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = set_seed(seed)
    indices = rng.permutation(n)
    chunks = np.array_split(indices, folds)
    split = []
    for i, test_idx in enumerate(chunks):
        train_idx = np.concatenate([chunk for j, chunk in enumerate(chunks) if j != i])
        split.append((train_idx, test_idx))
    return split


def cross_validate(dataset: Dataset, args: argparse.Namespace) -> dict[str, object]:
    all_true = []
    all_pred = []
    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(dataset.targets), args.folds, args.seed), start=1):
        standardizer = Standardizer(float(dataset.targets[train_idx].mean()), float(dataset.targets[train_idx].std() + 1e-8))
        y_train = standardizer.transform(dataset.targets[train_idx])
        if args.model == "ridge":
            model = RidgeGraphRegressor(alpha=args.ridge_alpha).fit(dataset.matrices[train_idx], y_train, args.density)
            losses = [float("nan")]
            pred = standardizer.inverse(model.predict(dataset.matrices[test_idx], args.density))
        elif args.model == "ensemble":
            gcn = GCNRegressor(
                input_dim=4,
                hidden_dim=args.hidden_dim,
                seed=args.seed + fold,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            losses = gcn.fit(dataset.matrices[train_idx], y_train, args.density, args.epochs)
            ridge = RidgeGraphRegressor(alpha=args.ridge_alpha).fit(dataset.matrices[train_idx], y_train, args.density)
            gcn_pred = gcn.predict(dataset.matrices[test_idx], args.density)
            ridge_pred = ridge.predict(dataset.matrices[test_idx], args.density)
            pred = standardizer.inverse(args.ensemble_gcn_weight * gcn_pred + (1.0 - args.ensemble_gcn_weight) * ridge_pred)
        else:
            model = GCNRegressor(
                input_dim=4,
                hidden_dim=args.hidden_dim,
                seed=args.seed + fold,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            losses = model.fit(dataset.matrices[train_idx], y_train, args.density, args.epochs)
            pred = standardizer.inverse(model.predict(dataset.matrices[test_idx], args.density))
        truth = dataset.targets[test_idx]
        all_true.extend(truth.tolist())
        all_pred.extend(pred.tolist())
        fold_reports.append(
            {
                "fold": fold,
                "subjects": [dataset.subject_ids[i] for i in test_idx],
                "metrics": metrics(truth, pred),
                "final_train_loss": float(losses[-1]),
            }
        )
    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    return {"overall": metrics(y_true, y_pred), "folds": fold_reports, "y_true": y_true, "y_pred": y_pred}


def train_full_model(dataset: Dataset, args: argparse.Namespace) -> tuple[GCNRegressor, Standardizer]:
    standardizer = Standardizer(float(dataset.targets.mean()), float(dataset.targets.std() + 1e-8))
    model = GCNRegressor(4, args.hidden_dim, args.seed, args.learning_rate, args.weight_decay)
    model.fit(dataset.matrices, standardizer.transform(dataset.targets), args.density, args.epochs)
    return model, standardizer


def tune_ridge(dataset: Dataset, args: argparse.Namespace) -> dict[str, object]:
    best = None
    for density in args.tune_densities:
        for alpha in args.tune_alphas:
            clone = argparse.Namespace(**vars(args))
            clone.model = "ridge"
            clone.density = density
            clone.ridge_alpha = alpha
            cv = cross_validate(dataset, clone)
            score = cv["overall"]["pearson_r"]
            if best is None or score > best["score"]:
                best = {"density": density, "ridge_alpha": alpha, "score": score, "metrics": cv["overall"]}
    if best is None:
        raise RuntimeError("No tuning candidates were evaluated.")
    return best


def permutation_test(y_true: np.ndarray, y_pred: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    rng = set_seed(seed)
    observed = metrics(y_true, y_pred)["pearson_r"]
    null = []
    for _ in range(permutations):
        null.append(metrics(rng.permutation(y_true), y_pred)["pearson_r"])
    null = np.asarray(null)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)
    return {"observed_r": float(observed), "p_value": float(p_value), "permutations": permutations}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str, samples: int, seed: int) -> dict[str, float]:
    rng = set_seed(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        values.append(metrics(y_true[idx], y_pred[idx])[metric_name])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"metric": metric_name, "low": float(lo), "high": float(hi), "samples": samples}


def ablation_report(dataset: Dataset, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    reports = {}
    for density in [0.2, args.density, 0.8]:
        clone = argparse.Namespace(**vars(args))
        clone.density = density
        cv = cross_validate(dataset, clone)
        reports[f"density_{density:.2f}"] = cv["overall"]
    shuffled = Dataset(dataset.subject_ids, dataset.matrices.copy(), np.random.permutation(dataset.targets), dataset.roi_names)
    cv_shuffled = cross_validate(shuffled, args)
    reports["shuffled_target_control"] = cv_shuffled["overall"]
    return reports


def explain_subject(
    model: GCNRegressor,
    standardizer: Standardizer,
    matrix: np.ndarray,
    roi_names: list[str],
    density: float,
) -> dict[str, object]:
    base_scaled = model.predict_one(matrix, density)
    base = standardizer.inverse(np.asarray([base_scaled]))[0]
    node_imp = []
    edge_imp = []
    for i, roi in enumerate(roi_names):
        perturbed = matrix.copy()
        perturbed[i, :] = 0.0
        perturbed[:, i] = 0.0
        delta = abs(base - standardizer.inverse(np.asarray([model.predict_one(perturbed, density)]))[0])
        node_imp.append((roi, float(delta)))

    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) <= 1e-8:
                continue
            perturbed = matrix.copy()
            perturbed[i, j] = 0.0
            perturbed[j, i] = 0.0
            delta = abs(base - standardizer.inverse(np.asarray([model.predict_one(perturbed, density)]))[0])
            edge_imp.append((roi_names[i], roi_names[j], float(delta)))

    node_imp.sort(key=lambda x: x[1], reverse=True)
    edge_imp.sort(key=lambda x: x[2], reverse=True)
    return {
        "predicted_working_memory_score": float(base),
        "top_nodes": [{"region": r, "importance": v} for r, v in node_imp[:5]],
        "top_edges": [{"source": a, "target": b, "importance": v} for a, b, v in edge_imp[:8]],
    }


def symbolic_reasoning(matrix: np.ndarray, roi_names: list[str]) -> list[str]:
    name_to_idx = {name: i for i, name in enumerate(roi_names)}

    def available(names: set[str]) -> list[int]:
        return [name_to_idx[x] for x in names if x in name_to_idx]

    fp = available(FRONTO_PARIENTAL)
    hip = available(HIPPOCAMPAL)
    ctrl = available(CONTROL_NETWORK)
    traces = []
    abs_m = np.abs(matrix)

    if fp:
        fp_strength = float(np.mean(abs_m[np.ix_(fp, fp)]))
        if fp_strength > 0.28:
            traces.append("IF fronto-parietal connectivity is strong THEN working-memory maintenance support is high.")
        else:
            traces.append("IF fronto-parietal connectivity is weak THEN working-memory maintenance may be reduced.")
    if hip and fp:
        hip_fp = float(np.mean(abs_m[np.ix_(hip, fp)]))
        if hip_fp < 0.16:
            traces.append("IF hippocampal-to-frontal coupling is weak THEN memory binding contribution is limited.")
        else:
            traces.append("IF hippocampal-to-frontal coupling is preserved THEN contextual memory support is plausible.")
    if ctrl and fp:
        ctrl_fp = float(np.mean(abs_m[np.ix_(ctrl, fp)]))
        if ctrl_fp > 0.22:
            traces.append("IF cognitive-control regions connect with fronto-parietal regions THEN attentional regulation support is high.")
    if not traces:
        n = matrix.shape[0]
        upper = np.triu_indices(n, k=1)
        edge_strength = abs_m[upper]
        node_strength = abs_m.sum(axis=1)
        mean_edge = float(np.mean(edge_strength))
        strength_cv = float(np.std(node_strength) / (np.mean(node_strength) + 1e-8))
        strongest_node = roi_names[int(np.argmax(node_strength))]

        if mean_edge >= 0.18:
            traces.append("IF global task-connectivity strength is high THEN distributed working-memory coordination is plausible.")
        else:
            traces.append("IF global task-connectivity strength is low THEN the working-memory graph may provide weak predictive evidence.")

        if strength_cv >= 0.35:
            traces.append(f"IF connectivity is hub-concentrated around {strongest_node} THEN model reasoning depends on a small set of dominant regions/parcels.")
        else:
            traces.append("IF connectivity is evenly distributed THEN model reasoning reflects broad network-level coordination.")

        positive_edges = float(np.mean(matrix[upper] > 0))
        if positive_edges >= 0.55:
            traces.append("IF most retained edges are positive THEN synchronized task-related activity dominates the explanation.")
        else:
            traces.append("IF retained edges are mixed positive and negative THEN the explanation reflects competing network relationships.")
    return traces


def save_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainable neuro-symbolic GCN for working-memory modeling.")
    parser.add_argument("--connectivity-dir", type=Path, default=None)
    parser.add_argument("--targets-csv", type=Path, default=None)
    parser.add_argument("--synthetic-subjects", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/research_report.json"))
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--target-column", default="working_memory_score")
    parser.add_argument("--model", choices=["gcn", "ridge", "ensemble"], default="gcn")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--ensemble-gcn-weight", type=float, default=0.35)
    parser.add_argument("--tune-ridge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global TARGET_COLUMN
    TARGET_COLUMN = args.target_column
    if args.connectivity_dir and args.targets_csv:
        dataset = load_connectivity_dataset(args.connectivity_dir, args.targets_csv)
        data_source = "real_connectivity_csv"
    else:
        dataset = generate_synthetic_dataset(args.synthetic_subjects, args.seed)
        data_source = "synthetic_hcp_like_fixture"

    tuning = None
    if args.tune_ridge:
        args.tune_densities = [0.2, 0.3, 0.35, 0.45, 0.6, 0.8]
        args.tune_alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 250.0]
        tuning = tune_ridge(dataset, args)
        args.model = "ridge"
        args.density = tuning["density"]
        args.ridge_alpha = tuning["ridge_alpha"]

    cv = cross_validate(dataset, args)
    model, standardizer = train_full_model(dataset, args)
    example_idx = int(np.argsort(np.abs(dataset.targets - dataset.targets.mean()))[0])
    explanation = explain_subject(
        model,
        standardizer,
        dataset.matrices[example_idx],
        dataset.roi_names,
        args.density,
    )
    reasoning = symbolic_reasoning(dataset.matrices[example_idx], dataset.roi_names)
    permutation = permutation_test(cv["y_true"], cv["y_pred"], args.permutations, args.seed)
    ci = bootstrap_ci(cv["y_true"], cv["y_pred"], "pearson_r", args.bootstrap_samples, args.seed)

    report: dict[str, object] = {
        "data_source": data_source,
        "subjects": len(dataset.targets),
        "target_column": TARGET_COLUMN,
        "model": args.model,
        "architecture": "connectivity_matrix -> thresholded_graph -> 2-layer_GCN -> explanation -> symbolic_rules",
        "cross_validation": {"overall": cv["overall"], "folds": cv["folds"]},
        "statistical_validation": {"permutation_test": permutation, "bootstrap_ci": ci},
        "example_subject": {
            "subject_id": dataset.subject_ids[example_idx],
            "observed_working_memory_score": float(dataset.targets[example_idx]),
            "explanation": explanation,
            "symbolic_reasoning_trace": reasoning,
        },
    }
    if tuning is not None:
        report["tuning"] = tuning
    if args.run_ablation:
        report["ablation"] = ablation_report(dataset, args)

    save_report(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"\nSaved research report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
