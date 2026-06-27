# Data Download Status

Real public OpenNeuro data has been added to this project.

## Downloaded Metadata

### OpenNeuro ds002424

Local path:

```text
data/openneuro_ds002424_metadata
```

Prepared path:

```text
data/openneuro_ds002424
```

Prepared targets:

```text
data/openneuro_ds002424/targets.csv
```

Target subjects extracted:

```text
79
```

Downloaded fMRI subset:

```text
62 subjects, task SLD
```

Connectivity matrices:

```text
data/openneuro_ds002424/connectivity/*.csv
```

Model report:

```text
outputs/openneuro_ds002424_real_report.json
```

Original 7-subject result:

```text
subjects: 7
Pearson r: 0.5355
R2: 0.2091
MAE: 0.1198
permutation p-value: 0.1961
```

Interpretation: positive signal, not statistically significant. The subset is too small for a publishable claim.

Expanded 36-subject report:

```text
outputs/openneuro_ds002424_real_report_36subjects.json
```

Expanded 36-subject result:

```text
subjects: 36
Pearson r: 0.3337
R2: 0.0946
MAE: 0.1011
RMSE: 0.1233
permutation p-value: 0.0647
bootstrap Pearson r CI: 0.0000 to 0.5468
```

Interpretation: positive but still not statistically significant at p < 0.05. This is a better real-data benchmark than the 7-subject run, but the model still needs more subjects and stronger preprocessing.

Expanded 62-subject report:

```text
outputs/openneuro_ds002424_best_gcn_62subjects.json
```

Expanded 62-subject result:

```text
subjects: 62
Pearson r: -0.0821
R2: -0.1902
MAE: 0.1036
RMSE: 0.1368
permutation p-value: 0.5509
bootstrap Pearson r CI: -0.3340 to 0.1721
```

Interpretation: adding more subjects did not improve the model with the current lightweight preprocessing. The earlier 36-subject signal is not stable enough to claim real predictive accuracy. The next required improvement is not more training epochs; it is better neuroimaging preprocessing and atlas-based parcellation.

Additional experiments run:

```text
outputs/openneuro_ds002424_sld_tuned_ridge_36subjects.json
outputs/openneuro_ds002424_sld_gcn_36subjects.json
outputs/openneuro_ds002424_overall_tuned_ridge_36subjects.json
outputs/openneuro_ds002424_20parcels_gcn_36subjects.json
```

These did not outperform the original 36-subject 10-parcel GCN result.

## OpenNeuro ds002687

Local path:

```text
data/openneuro_ds002687_metadata
```

Prepared path:

```text
data/openneuro_ds002687
```

Prepared targets:

```text
data/openneuro_ds002687/targets.csv
```

Target subjects extracted:

```text
24
```

Downloaded fMRI subset:

```text
8 subjects, task SLD
```

Connectivity matrices:

```text
data/openneuro_ds002687/connectivity/*.csv
```

Model report:

```text
outputs/openneuro_ds002687_real_report.json
```

Result on the downloaded subset:

```text
subjects: 8
Pearson r: -0.3867
R2: -1.1605
MAE: 0.0652
permutation p-value: 0.3922
```

Interpretation: no reliable predictive signal at this subset size.

## Combined Real Benchmark

Combined path:

```text
data/openneuro_combined
```

Connectivity matrices:

```text
15 subjects total
```

Model report:

```text
outputs/openneuro_combined_real_report.json
```

Result:

```text
subjects: 15
Pearson r: -0.1607
R2: -0.6587
MAE: 0.1567
permutation p-value: 0.5309
```

Interpretation: not useful as a combined prediction benchmark yet. The child and adult datasets differ strongly, and the current subset is too small.

## What Is Real Now

- Dataset metadata is real.
- Event files are real.
- 2-back accuracy targets are real.
- Downloaded BOLD fMRI files are real.
- Connectivity matrices are generated from real fMRI files.
- Training reports are generated from real connectivity matrices.

## Current Limitation

The current connectivity builder uses lightweight data-driven spatial parcels because this machine does not have Nilearn, Nibabel, SciPy, scikit-learn, PyTorch, or PyTorch Geometric installed.

For publication-quality neuroscience, replace this with:

- Atlas-based parcellation.
- Motion/confound regression.
- Bandpass filtering.
- Named brain-region labels.
- Larger downloaded subject count.
- PyTorch Geometric GCN/GAT baselines.

## Continue Downloading More Subjects

Children dataset:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\prepare_openneuro_nback.py `
  --metadata-dir data\openneuro_ds002424_metadata `
  --prepared-dir data\openneuro_ds002424 `
  --download-bolds `
  --build-connectivity `
  --task SLD `
  --max-subjects 40
```

Adults dataset:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\prepare_openneuro_nback.py `
  --metadata-dir data\openneuro_ds002687_metadata `
  --prepared-dir data\openneuro_ds002687 `
  --download-bolds `
  --build-connectivity `
  --task SLD `
  --max-subjects 24
```
