# Fold 4 Summary for RMSE/Linear/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13897927 | 69.83% | 70.01% |
| Val | 65 | 1989805 | 10.11% | 10.02% |
| Test | 129 | 3963508 | 20.06% | 19.97% |

## Timings
- **Training**: 0:03:35.048116
- **Prediction**: 0:01:04.543539

## Artifacts
- **Weights**: `models/RMSE_Linear_H2_Fold4.weights.h5`
- **History csv**: `models/RMSE_Linear_H2_Fold4_history.csv`
- **Best json**: `models/RMSE_Linear_H2_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_RMSE_Linear_H2_Fold4.parquet`
- **Training plot prefix**: `plots/training/RMSE_Linear_H2_Fold4`
