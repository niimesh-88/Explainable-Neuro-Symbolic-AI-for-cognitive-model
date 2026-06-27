# Explainable Neuro-Symbolic Graph Model for Working Memory

This workspace contains a runnable research prototype based on the strategy document:

`Brain connectivity data -> graph construction -> graph neural model -> explainability -> symbolic cognitive reasoning`

The implementation is designed as a practical baseline for the paper direction:

**Working Memory Prediction Using Explainable Graph Neural Networks on Brain Connectivity Data**

## What Is Implemented

- A real two-layer graph convolutional regressor implemented with NumPy.
- Connectivity matrix thresholding into sparse brain graphs.
- Node features from graph strength, signed strength, degree, and clustering proxy.
- K-fold cross-validation.
- MAE, RMSE, R2, and Pearson correlation.
- Permutation significance testing.
- Bootstrap confidence intervals.
- Density ablations and shuffled-target control.
- Perturbation-based node and edge importance.
- Symbolic cognitive reasoning traces for fronto-parietal, hippocampal, and cognitive-control networks.
- Synthetic HCP-like fixture data so the pipeline runs immediately before protected neuroimaging data is available.

## Quick Run

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\scie_neurosymbolic_wm_model.py
```

The run writes:

```text
outputs/research_report.json
```

## Use Real Connectivity Data

See [REAL_DATASETS.md](REAL_DATASETS.md) for the recommended real datasets: HCP Young Adult, ABCD, and OpenNeuro ds002424.
See [DATA_DOWNLOAD_STATUS.md](DATA_DOWNLOAD_STATUS.md) for the real OpenNeuro datasets already downloaded and trained in this workspace.

Create a directory of subject-level CSV connectivity matrices:

```text
data/connectivity/
  100307.csv
  100408.csv
  101006.csv
```

Create a target file:

```csv
subject_id,working_memory_score
100307,112.4
100408,98.1
101006,105.7
```

Then run:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\scie_neurosymbolic_wm_model.py `
  --connectivity-dir .\data\connectivity `
  --targets-csv .\data\targets.csv `
  --run-ablation
```

Each connectivity CSV should be a square ROI-by-ROI matrix. The first column may contain ROI labels, and column names are used as region names for explanations.

## Why This Is Research-Oriented

This is not just a prediction script. It produces the kinds of evidence expected in psychology and cognitive neuroscience research:

- Predictive validity through cross-validation.
- Statistical reliability through permutation testing and confidence intervals.
- Interpretability through region and edge importance.
- Cognitive plausibility through symbolic reasoning rules.
- Ablation controls to test whether performance depends on meaningful graph structure.

## Next Accuracy Improvements

For a stronger publication-grade model, the next step is to install the full research stack and replace the NumPy GCN with PyTorch Geometric:

- `torch`
- `torch_geometric`
- `nilearn`
- `scikit-learn`
- `scipy`
- `networkx`

Then add:

- HCP working-memory task targets.
- Atlas-specific ROI mapping, such as Schaefer or AAL.
- Nested cross-validation for hyperparameter tuning.
- GAT or GraphSAGE baselines.
- GNNExplainer-style masks.
- Comparison against SVR, Random Forest, MLP, and standard GCN.

For a deeper local experiment, add `--run-ablation --epochs 140 --synthetic-subjects 160`.

## Real OpenNeuro Runs Already Prepared

Run the full explained model across the real datasets:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\run_explained_multidataset.py
```

This writes:

```text
EXPLAINED_MODEL_REPORT.md
outputs/explained_multidataset_report.json
```

Run the improved validated accuracy comparison:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\train_improved_accuracy.py
```

This writes:

```text
IMPROVED_ACCURACY_REPORT.md
outputs/improved_accuracy_report.json
```

Children dataset:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\scie_neurosymbolic_wm_model.py `
  --connectivity-dir data\openneuro_ds002424\connectivity `
  --targets-csv data\openneuro_ds002424\targets.csv `
  --folds 2 `
  --epochs 120 `
  --output outputs\openneuro_ds002424_real_report.json
```

Adult dataset:

```powershell
& 'C:\Users\nimes\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\scie_neurosymbolic_wm_model.py `
  --connectivity-dir data\openneuro_ds002687\connectivity `
  --targets-csv data\openneuro_ds002687\targets.csv `
  --folds 2 `
  --epochs 120 `
  --output outputs\openneuro_ds002687_real_report.json
```
