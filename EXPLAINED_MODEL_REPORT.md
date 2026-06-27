# Explained Neuro-Symbolic Cognitive Model

This report evaluates the same explained graph-cognitive model on multiple real OpenNeuro n-back fMRI datasets.

Pipeline:

```text
real fMRI BOLD -> functional connectivity matrix -> brain graph -> GCN prediction -> node/edge explanation -> symbolic reasoning trace
```

## Dataset Results

| Dataset | Subjects | Pearson r | R2 | MAE | RMSE | Permutation p |
|---|---:|---:|---:|---:|---:|---:|
| OpenNeuro ds002424 Children/ADHD | 62 | -0.0891 | -0.1286 | 0.1027 | 0.1332 | 0.4627 |
| OpenNeuro ds002687 Adults | 8 | -0.5200 | -1.3281 | 0.0700 | 0.0825 | 0.1642 |
| Combined OpenNeuro Children + Adults | 70 | 0.0173 | -0.0387 | 0.1041 | 0.1324 | 0.8706 |

## Interpretation

The model is explainable and runs on real fMRI-derived connectivity data, but current predictive accuracy is not yet publication-grade. The strongest scientific value at this stage is the transparent workflow: every prediction includes region/parcel importance, edge importance, and a symbolic reasoning trace.

The main bottleneck is not the neuro-symbolic layer; it is the lightweight connectivity preprocessing. A stronger version should use atlas-based parcellation, motion/confound regression, and PyTorch Geometric baselines.

## Explanations

### OpenNeuro ds002424 Children/ADHD

Child/adolescent n-back fMRI dataset with ADHD/control metadata. Uses SLD task connectivity and overall 2-back accuracy target.

Recurring important nodes/parcels:
- parcel_06: appears 3 times, total importance 0.0846
- parcel_05: appears 2 times, total importance 0.0963
- parcel_07: appears 2 times, total importance 0.0554
- parcel_08: appears 2 times, total importance 0.0856
- parcel_09: appears 2 times, total importance 0.1176

Example symbolic reasoning:
- IF global task-connectivity strength is high THEN distributed working-memory coordination is plausible.
- IF connectivity is evenly distributed THEN model reasoning reflects broad network-level coordination.
- IF most retained edges are positive THEN synchronized task-related activity dominates the explanation.

### OpenNeuro ds002687 Adults

Adult n-back fMRI companion dataset. Uses SLD task connectivity and overall 2-back accuracy target.

Recurring important nodes/parcels:
- parcel_02: appears 3 times, total importance 0.0613
- parcel_08: appears 2 times, total importance 0.0414
- parcel_09: appears 2 times, total importance 0.0364
- parcel_10: appears 2 times, total importance 0.0306
- parcel_03: appears 1 times, total importance 0.0375

Example symbolic reasoning:
- IF global task-connectivity strength is high THEN distributed working-memory coordination is plausible.
- IF connectivity is evenly distributed THEN model reasoning reflects broad network-level coordination.
- IF most retained edges are positive THEN synchronized task-related activity dominates the explanation.

### Combined OpenNeuro Children + Adults

Combined benchmark made from all prepared ds002424 and ds002687 connectivity matrices.

Recurring important nodes/parcels:
- parcel_07: appears 3 times, total importance 0.0342
- parcel_05: appears 3 times, total importance 0.0318
- parcel_03: appears 3 times, total importance 0.0257
- parcel_04: appears 2 times, total importance 0.0221
- parcel_06: appears 2 times, total importance 0.0198

Example symbolic reasoning:
- IF global task-connectivity strength is high THEN distributed working-memory coordination is plausible.
- IF connectivity is evenly distributed THEN model reasoning reflects broad network-level coordination.
- IF most retained edges are positive THEN synchronized task-related activity dominates the explanation.
