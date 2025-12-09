# Fold 2 Summary for gMSE/CNN/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13886812 | 69.98% | 69.95% |
| Val | 65 | 1997483 | 10.11% | 10.06% |
| Test | 128 | 3966945 | 19.91% | 19.98% |

## Timings
- **Training**: 0:07:59.826748
- **Prediction**: 0:00:59.920170

## Artifacts
- **Weights**: `models/gMSE_CNN_H2_Fold2.weights.h5`
- **History csv**: `models/gMSE_CNN_H2_Fold2_history.csv`
- **Best json**: `models/gMSE_CNN_H2_Fold2.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_CNN_H2_Fold2.parquet`
- **Training plot prefix**: `plots/training/gMSE_CNN_H2_Fold2`
