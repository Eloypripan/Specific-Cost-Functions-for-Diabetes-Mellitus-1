# Fold 2 Summary for RMSE/LSTM/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13886812 | 69.98% | 69.95% |
| Val | 65 | 1997483 | 10.11% | 10.06% |
| Test | 128 | 3966945 | 19.91% | 19.98% |

## Timings
- **Training**: 0:02:40.389812
- **Prediction**: 0:01:11.541070

## Artifacts
- **Weights**: `models/RMSE_LSTM_H2_Fold2.weights.h5`
- **History csv**: `models/RMSE_LSTM_H2_Fold2_history.csv`
- **Best json**: `models/RMSE_LSTM_H2_Fold2.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_RMSE_LSTM_H2_Fold2.parquet`
- **Training plot prefix**: `plots/training/RMSE_LSTM_H2_Fold2`
