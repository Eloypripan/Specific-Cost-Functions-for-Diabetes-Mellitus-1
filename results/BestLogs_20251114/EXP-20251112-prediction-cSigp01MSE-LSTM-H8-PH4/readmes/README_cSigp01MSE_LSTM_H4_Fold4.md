# Fold 4 Summary for cSigMSEP01/LSTM/PH=4

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_4.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 449 | 13596135 | 69.83% | 70.01% |
| Val | 65 | 1947080 | 10.11% | 10.03% |
| Test | 129 | 3878191 | 20.06% | 19.97% |

## Timings
- **Training**: 0:04:48.829799
- **Prediction**: 0:01:10.042471

## Artifacts
- **Weights**: `models/cSigMSEP01_LSTM_H4_Fold4.weights.h5`
- **History csv**: `models/cSigMSEP01_LSTM_H4_Fold4_history.csv`
- **Best json**: `models/cSigMSEP01_LSTM_H4_Fold4.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_cSigMSEP01_LSTM_H4_Fold4.parquet`
- **Training plot prefix**: `plots/training/cSigMSEP01_LSTM_H4_Fold4`
