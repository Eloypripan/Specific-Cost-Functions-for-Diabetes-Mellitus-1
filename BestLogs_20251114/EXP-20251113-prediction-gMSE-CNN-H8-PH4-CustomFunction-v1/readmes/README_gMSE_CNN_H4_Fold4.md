# Fold 4 Summary for gMSE/CNN/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13596135 | 69.83% | 70.01% |
| Val | 65 | 1947080 | 10.11% | 10.03% |
| Test | 129 | 3878191 | 20.06% | 19.97% |

## Timings
- **Training**: 0:09:24.655839
- **Prediction**: 0:00:58.320889

## Artifacts
- **Weights**: `models/gMSE_CNN_H4_Fold4.weights.h5`
- **History csv**: `models/gMSE_CNN_H4_Fold4_history.csv`
- **Best json**: `models/gMSE_CNN_H4_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_CNN_H4_Fold4.parquet`
- **Training plot prefix**: `plots/training/gMSE_CNN_H4_Fold4`
