# Fold 5 Summary for cSigP01/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13583812 | 69.98% | 69.94% |
| Val | 65 | 1943190 | 10.11% | 10.01% |
| Test | 128 | 3894404 | 19.91% | 20.05% |

## Timings
- **Training**: 0:03:05.669395
- **Prediction**: 0:01:10.025592

## Artifacts
- **Weights**: `models/cSigP01_LSTM_H4_Fold5.weights.h5`
- **History csv**: `models/cSigP01_LSTM_H4_Fold5_history.csv`
- **Best json**: `models/cSigP01_LSTM_H4_Fold5.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_cSigP01_LSTM_H4_Fold5.parquet`
- **Training plot prefix**: `plots/training/cSigP01_LSTM_H4_Fold5`
