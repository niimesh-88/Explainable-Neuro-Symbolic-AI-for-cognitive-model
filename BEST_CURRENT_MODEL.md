# Best Current Real-Data Model

The best current real-data result remains the 36-subject OpenNeuro ds002424 GCN run:

```text
outputs/openneuro_ds002424_real_report_36subjects.json
```

## Best Validation Result

```text
Dataset: OpenNeuro ds002424
Task data: SLD n-back fMRI connectivity
Subjects: 36
Target: overall 2-back working-memory accuracy
Model: 2-layer graph convolution regressor
Parcels: 10 lightweight data-driven parcels

Pearson r: 0.3337
R2: 0.0946
MAE: 0.1011
RMSE: 0.1233
Permutation p-value: 0.0647
```

## What Was Tried To Improve Accuracy

1. More subjects:

```text
62-subject run: Pearson r = -0.0821, p = 0.5509
```

2. Task-matched SLD target:

```text
SLD target GCN: Pearson r = 0.1548, p = 0.3632
```

3. Tuned graph-feature ridge model:

```text
Overall target ridge: Pearson r = 0.2944, p = 0.0631
SLD target ridge: Pearson r = 0.1642, p = 0.3555
```

4. Higher-resolution 20-parcel connectivity:

```text
20-parcel GCN: Pearson r = -0.0291, p = 0.8505
```

## Conclusion

The current implementation is a real research prototype, but it is not yet an accurate model. The bottleneck is the lightweight neuroimaging preprocessing, not simply the number of epochs or the model architecture.

For a genuinely stronger model, the next upgrade should be:

- Install Nilearn, Nibabel, SciPy, scikit-learn, PyTorch, and PyTorch Geometric.
- Use atlas-based parcellation instead of data-driven parcels.
- Add confound regression and motion correction.
- Use all available task runs per subject.
- Use subject-level cross-validation if task runs are treated as multiple samples.
- Add proper GCN, GAT, Ridge, SVR, and Random Forest baselines.
