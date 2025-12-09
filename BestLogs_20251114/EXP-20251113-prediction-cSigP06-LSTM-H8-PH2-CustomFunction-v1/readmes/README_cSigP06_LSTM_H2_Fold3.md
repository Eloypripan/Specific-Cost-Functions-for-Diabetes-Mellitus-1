# Fold 3 Summary for cSigP06/LSTM/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13900181 | 69.98% | 70.02% |
| Val | 65 | 1976792 | 10.11% | 9.96% |
| Test | 128 | 3974267 | 19.91% | 20.02% |

## Timings
- **Training**: 0:03:11.037031
- **Prediction**: 0:01:12.886374

## Artifacts
- **Weights**: `models/cSigP06_LSTM_H2_Fold3.weights.h5`
- **History csv**: `models/cSigP06_LSTM_H2_Fold3_history.csv`
- **Best json**: `models/cSigP06_LSTM_H2_Fold3.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_cSigP06_LSTM_H2_Fold3.parquet`
- **Training plot prefix**: `plots/training/cSigP06_LSTM_H2_Fold3`
