# Fold 4 Summary for gMAD/CNN/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13897927 | 69.83% | 70.01% |
| Val | 65 | 1989805 | 10.11% | 10.02% |
| Test | 129 | 3963508 | 20.06% | 19.97% |

## Timings
- **Training**: 0:05:21.093691
- **Prediction**: 0:00:59.480773

## Artifacts
- **Weights**: `models/gMAD_CNN_H2_Fold4.weights.h5`
- **History csv**: `models/gMAD_CNN_H2_Fold4_history.csv`
- **Best json**: `models/gMAD_CNN_H2_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMAD_CNN_H2_Fold4.parquet`
- **Training plot prefix**: `plots/training/gMAD_CNN_H2_Fold4`
