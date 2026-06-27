# Real Dataset Plan

This project should move from synthetic validation to real training in this order.

## Dataset 1: Human Connectome Project Young Adult

Use this as the primary training dataset.

Source:

- https://humanconnectome.org/study/hcp-young-adult/data-releases
- https://registry.opendata.aws/hcp-openaccess/

Why it fits:

- Large healthy-adult neuroimaging dataset.
- Includes behavioral and 3T MR imaging data.
- Includes task fMRI working-memory/cognitive-control protocol.
- Includes resting-state and task data suitable for connectivity matrices.
- Strong credibility for Frontiers-level cognitive neuroscience work.

Recommended target:

- `WM_Task_2bk_Acc`: 2-back working-memory accuracy.
- Optional secondary targets: 2-back reaction time, List Sorting working-memory score.

Recommended model input:

- Resting-state functional connectivity matrices, or
- Task-derived working-memory connectivity matrices, or
- Structural connectivity matrices from diffusion MRI.

Best first experiment:

Train the model to predict 2-back working-memory accuracy from resting-state functional connectivity.

## Dataset 2: ABCD Study

Use this for larger-scale replication and external validation.

Source:

- https://docs.abcdstudy.org/latest/documentation/imaging/type_tfmribeh.html
- https://docs.abcdstudy.org/latest/documentation/imaging/type_trial.html

Why it fits:

- Large longitudinal developmental dataset.
- Includes task fMRI behavioral performance.
- Includes Emotional nBack Working Memory Task.
- Provides trial-level variables for 0-back and 2-back task conditions.
- Useful for testing whether the HCP-trained approach generalizes to adolescents.

Recommended target:

- 2-back accuracy from the Emotional nBack task.
- Optional secondary targets: 2-back reaction time, 2-back minus 0-back accuracy, QC-filtered n-back summary scores.

Recommended model input:

- Resting-state functional connectivity matrices, or
- Emotional nBack task fMRI connectivity matrices.

Important caution:

ABCD performance measures are sensitive to demographic and socioeconomic factors. Treat age, sex, site, motion, and SES-related variables as covariates or stratification variables.

## Dataset 3: OpenNeuro ds002424

Use this as a small open benchmark and clinical proof-of-concept.

Source:

- https://openneuro.org/datasets/ds002424
- https://pubmed.ncbi.nlm.nih.gov/32566704/

Why it fits:

- Public BIDS-format neuroimaging dataset.
- Children with and without ADHD.
- Participants completed eight n-back working-memory tasks during fMRI.
- Includes reward/feedback manipulations, useful for future cognitive interpretation.

Recommended target:

- n-back accuracy per condition.
- Optional classification target: ADHD vs control.

Recommended model input:

- Task fMRI connectivity during n-back blocks.

Important caution:

This dataset is much smaller than HCP or ABCD, so use it for external validation, clinical case study, or method demonstration rather than the main accuracy claim.

## Recommended Research Design

1. Train and tune on HCP Young Adult.
2. Report cross-validated prediction of 2-back working-memory accuracy.
3. Explain important brain regions and edges.
4. Validate symbolic rules against known working-memory networks.
5. Replicate on ABCD using emotional n-back targets.
6. Optionally test clinical transfer on OpenNeuro ds002424.

## Expected Project Directory

```text
data/
  hcp/
    connectivity/
      subject_id.csv
    targets.csv
  abcd/
    connectivity/
      subject_id.csv
    targets.csv
  openneuro_ds002424/
    connectivity/
      subject_id.csv
    targets.csv
```

Each `targets.csv` should use this minimum schema:

```csv
subject_id,working_memory_score
100307,0.92
100408,0.81
```

Additional covariate columns are recommended:

```csv
subject_id,working_memory_score,age,sex,site,mean_fd
```

The current script already accepts:

```powershell
--connectivity-dir data\hcp\connectivity --targets-csv data\hcp\targets.csv
```
