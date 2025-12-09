# Fold 2 Summary for gMSE/Linear/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13886812 | 69.98% | 69.95% |
| Val | 65 | 1997483 | 10.11% | 10.06% |
| Test | 128 | 3966945 | 19.91% | 19.98% |

## Timings
- **Training**: 0:04:00.251044
- **Prediction**: 0:01:15.140969

## Artifacts
- **Weights**: `models/gMSE_Linear_H2_Fold2.weights.h5`
- **History csv**: `models/gMSE_Linear_H2_Fold2_history.csv`
- **Best json**: `models/gMSE_Linear_H2_Fold2.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_Linear_H2_Fold2.parquet`
- **Training plot prefix**: `plots/training/gMSE_Linear_H2_Fold2`
