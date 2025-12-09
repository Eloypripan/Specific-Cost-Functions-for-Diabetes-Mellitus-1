# Fold 1 Summary for gMSE/Linear/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 448 | 13876226 | 69.67% | 69.90% |
| Val | 65 | 2010527 | 10.11% | 10.13% |
| Test | 130 | 3964487 | 20.22% | 19.97% |

## Timings
- **Training**: 0:03:45.758362
- **Prediction**: 0:01:17.759901

## Artifacts
- **Weights**: `models/gMSE_Linear_H2_Fold1.weights.h5`
- **History csv**: `models/gMSE_Linear_H2_Fold1_history.csv`
- **Best json**: `models/gMSE_Linear_H2_Fold1.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_Linear_H2_Fold1.parquet`
- **Training plot prefix**: `plots/training/gMSE_Linear_H2_Fold1`
