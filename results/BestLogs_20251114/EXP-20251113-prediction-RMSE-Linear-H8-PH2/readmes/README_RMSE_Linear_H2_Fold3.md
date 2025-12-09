# Fold 3 Summary for RMSE/Linear/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13900181 | 69.98% | 70.02% |
| Val | 65 | 1976792 | 10.11% | 9.96% |
| Test | 128 | 3974267 | 19.91% | 20.02% |

## Timings
- **Training**: 0:02:14.724230
- **Prediction**: 0:01:05.127553

## Artifacts
- **Weights**: `models/RMSE_Linear_H2_Fold3.weights.h5`
- **History csv**: `models/RMSE_Linear_H2_Fold3_history.csv`
- **Best json**: `models/RMSE_Linear_H2_Fold3.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_RMSE_Linear_H2_Fold3.parquet`
- **Training plot prefix**: `plots/training/RMSE_Linear_H2_Fold3`
