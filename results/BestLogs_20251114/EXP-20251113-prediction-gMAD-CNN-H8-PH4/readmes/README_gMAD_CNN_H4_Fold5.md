# Fold 5 Summary for gMAD/CNN/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13583812 | 69.98% | 69.94% |
| Val | 65 | 1943190 | 10.11% | 10.01% |
| Test | 128 | 3894404 | 19.91% | 20.05% |

## Timings
- **Training**: 0:08:47.345508
- **Prediction**: 0:00:58.353160

## Artifacts
- **Weights**: `models/gMAD_CNN_H4_Fold5.weights.h5`
- **History csv**: `models/gMAD_CNN_H4_Fold5_history.csv`
- **Best json**: `models/gMAD_CNN_H4_Fold5.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMAD_CNN_H4_Fold5.parquet`
- **Training plot prefix**: `plots/training/gMAD_CNN_H4_Fold5`
