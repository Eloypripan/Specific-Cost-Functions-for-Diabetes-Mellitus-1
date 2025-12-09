# Fold 1 Summary for cSigMSEP01/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 448 | 13572679 | 69.67% | 69.89% |
| Val | 65 | 1969279 | 10.11% | 10.14% |
| Test | 130 | 3879448 | 20.22% | 19.98% |

## Timings
- **Training**: 0:06:14.546810
- **Prediction**: 0:01:11.119158

## Artifacts
- **Weights**: `models/cSigMSEP01_LSTM_H4_Fold1.weights.h5`
- **History csv**: `models/cSigMSEP01_LSTM_H4_Fold1_history.csv`
- **Best json**: `models/cSigMSEP01_LSTM_H4_Fold1.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_cSigMSEP01_LSTM_H4_Fold1.parquet`
- **Training plot prefix**: `plots/training/cSigMSEP01_LSTM_H4_Fold1`
