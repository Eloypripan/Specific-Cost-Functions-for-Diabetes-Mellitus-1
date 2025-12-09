# Fold 5 Summary for gMSE/Linear/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13883119 | 69.98% | 69.94% |
| Val | 65 | 1986088 | 10.11% | 10.00% |
| Test | 128 | 3982033 | 19.91% | 20.06% |

## Timings
- **Training**: 0:05:52.786974
- **Prediction**: 0:01:14.287774

## Artifacts
- **Weights**: `models/gMSE_Linear_H2_Fold5.weights.h5`
- **History csv**: `models/gMSE_Linear_H2_Fold5_history.csv`
- **Best json**: `models/gMSE_Linear_H2_Fold5.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_Linear_H2_Fold5.parquet`
- **Training plot prefix**: `plots/training/gMSE_Linear_H2_Fold5`
