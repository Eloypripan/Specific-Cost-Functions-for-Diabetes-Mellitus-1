# Fold 3 Summary for gMSE/CNN/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13597404 | 69.98% | 70.01% |
| Val | 65 | 1934027 | 10.11% | 9.96% |
| Test | 128 | 3889975 | 19.91% | 20.03% |

## Timings
- **Training**: 0:05:06.437158
- **Prediction**: 0:00:57.648022

## Artifacts
- **Weights**: `models/gMSE_CNN_H4_Fold3.weights.h5`
- **History csv**: `models/gMSE_CNN_H4_Fold3_history.csv`
- **Best json**: `models/gMSE_CNN_H4_Fold3.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_CNN_H4_Fold3.parquet`
- **Training plot prefix**: `plots/training/gMSE_CNN_H4_Fold3`
