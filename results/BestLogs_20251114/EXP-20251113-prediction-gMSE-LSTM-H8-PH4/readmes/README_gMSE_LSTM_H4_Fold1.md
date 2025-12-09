# Fold 1 Summary for gMSE/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 448 | 13572679 | 69.67% | 69.89% |
| Val | 65 | 1969279 | 10.11% | 10.14% |
| Test | 130 | 3879448 | 20.22% | 19.98% |

## Timings
- **Training**: 0:05:07.163854
- **Prediction**: 0:01:12.165444

## Artifacts
- **Weights**: `models/gMSE_LSTM_H4_Fold1.weights.h5`
- **History csv**: `models/gMSE_LSTM_H4_Fold1_history.csv`
- **Best json**: `models/gMSE_LSTM_H4_Fold1.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMSE_LSTM_H4_Fold1.parquet`
- **Training plot prefix**: `plots/training/gMSE_LSTM_H4_Fold1`
