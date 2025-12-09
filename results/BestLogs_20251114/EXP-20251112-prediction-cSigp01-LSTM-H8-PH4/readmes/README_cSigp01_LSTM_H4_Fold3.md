# Fold 3 Summary for cSigP01/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13597404 | 69.98% | 70.01% |
| Val | 65 | 1934027 | 10.11% | 9.96% |
| Test | 128 | 3889975 | 19.91% | 20.03% |

## Timings
- **Training**: 0:05:27.931071
- **Prediction**: 0:01:09.670622

## Artifacts
- **Weights**: `models/cSigP01_LSTM_H4_Fold3.weights.h5`
- **History csv**: `models/cSigP01_LSTM_H4_Fold3_history.csv`
- **Best json**: `models/cSigP01_LSTM_H4_Fold3.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_cSigP01_LSTM_H4_Fold3.parquet`
- **Training plot prefix**: `plots/training/cSigP01_LSTM_H4_Fold3`
