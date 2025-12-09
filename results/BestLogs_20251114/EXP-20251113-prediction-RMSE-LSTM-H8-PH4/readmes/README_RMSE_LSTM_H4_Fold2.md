# Fold 2 Summary for RMSE/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13588245 | 69.98% | 69.97% |
| Val | 65 | 1953773 | 10.11% | 10.06% |
| Test | 128 | 3879388 | 19.91% | 19.97% |

## Timings
- **Training**: 0:02:37.906915
- **Prediction**: 0:01:10.195585

## Artifacts
- **Weights**: `models/RMSE_LSTM_H4_Fold2.weights.h5`
- **History csv**: `models/RMSE_LSTM_H4_Fold2_history.csv`
- **Best json**: `models/RMSE_LSTM_H4_Fold2.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_RMSE_LSTM_H4_Fold2.parquet`
- **Training plot prefix**: `plots/training/RMSE_LSTM_H4_Fold2`
