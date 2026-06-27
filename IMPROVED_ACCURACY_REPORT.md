# Improved Accuracy Report

This report compares graph-only, covariate-only, and graph+covariate ridge models using nested alpha tuning inside each cross-validation fold.

| Dataset | Subjects | Best Mode | Pearson r | R2 | MAE | RMSE | p-value |
|---|---:|---|---:|---:|---:|---:|---:|
| OpenNeuro ds002424 Children/ADHD | 62 | covariates | 0.4484 | 0.2007 | 0.0895 | 0.1121 | 0.0020 |
| OpenNeuro ds002687 Adults | 8 | graph_covariates | -0.5169 | -0.3314 | 0.0533 | 0.0624 | 0.1796 |
| Combined OpenNeuro Children + Adults | 70 | covariates | 0.5666 | 0.3170 | 0.0862 | 0.1074 | 0.0020 |

## Model Comparisons

### OpenNeuro ds002424 Children/ADHD

- graph: r=-0.2440, R2=-0.1020, MAE=0.1024, p=0.0419
- covariates: r=0.4484, R2=0.2007, MAE=0.0895, p=0.0020
- graph_covariates: r=0.2199, R2=-0.0677, MAE=0.1037, p=0.0858

Top explanatory features in the selected model:
- ADHD_diagnosis: weight -0.09061
- sex: weight -0.03396
- n_tasks_with_2back: weight 0.01960
- age_combined: weight 0.00664
- age_squared: weight 0.00078
- n_2back_trials: weight 0.00043

### OpenNeuro ds002687 Adults

- graph: r=-0.5205, R2=-0.3386, MAE=0.0534, p=0.1737
- covariates: r=-0.8484, R2=-0.5266, MAE=0.0561, p=0.0100
- graph_covariates: r=-0.5169, R2=-0.3314, MAE=0.0533, p=0.1796

Top explanatory features in the selected model:
- graph_76: weight -0.02087
- graph_78: weight -0.01159
- graph_77: weight -0.00809
- graph_17: weight 0.00601
- graph_31: weight -0.00504
- graph_3: weight -0.00298
- graph_75: weight 0.00269
- graph_38: weight -0.00220

### Combined OpenNeuro Children + Adults

- graph: r=-0.0524, R2=-0.0327, MAE=0.1053, p=0.6707
- covariates: r=0.5666, R2=0.3170, MAE=0.0862, p=0.0020
- graph_covariates: r=0.3990, R2=0.1574, MAE=0.0949, p=0.0020

Top explanatory features in the selected model:
- ADHD_diagnosis: weight -0.09657
- cohort_adults: weight 0.05737
- cohort_children: weight -0.05737
- age_combined: weight 0.03770
- sex: weight -0.03654
- n_tasks_with_2back: weight 0.01967
- age_squared: weight -0.00095
- n_2back_trials: weight 0.00043
