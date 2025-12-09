# Fold 4 Summary for RMSE/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13596135 | 69.83% | 70.01% |
| Val | 65 | 1947080 | 10.11% | 10.03% |
| Test | 129 | 3878191 | 20.06% | 19.97% |

## Timings
- **Training**: 0:03:22.890209
- **Prediction**: 0:01:09.540390

## Artifacts
- **Weights**: `models/RMSE_LSTM_H4_Fold4.weights.h5`
- **History csv**: `models/RMSE_LSTM_H4_Fold4_history.csv`
- **Best json**: `models/RMSE_LSTM_H4_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_RMSE_LSTM_H4_Fold4.parquet`
- **Training plot prefix**: `plots/training/RMSE_LSTM_H4_Fold4`
