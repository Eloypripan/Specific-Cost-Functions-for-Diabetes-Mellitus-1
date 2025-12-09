# Fold 4 Summary for gMAD/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13596135 | 69.83% | 70.01% |
| Val | 65 | 1947080 | 10.11% | 10.03% |
| Test | 129 | 3878191 | 20.06% | 19.97% |

## Timings
- **Training**: 0:03:00.394597
- **Prediction**: 0:01:11.156122

## Artifacts
- **Weights**: `models/gMAD_LSTM_H4_Fold4.weights.h5`
- **History csv**: `models/gMAD_LSTM_H4_Fold4_history.csv`
- **Best json**: `models/gMAD_LSTM_H4_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMAD_LSTM_H4_Fold4.parquet`
- **Training plot prefix**: `plots/training/gMAD_LSTM_H4_Fold4`
