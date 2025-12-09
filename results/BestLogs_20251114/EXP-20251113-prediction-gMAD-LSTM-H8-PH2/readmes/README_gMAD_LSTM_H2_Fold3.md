# Fold 3 Summary for gMAD/LSTM/PH=2

**Data File**: `/opt/datasets/folds/T1DiabetesGranada/windows_with_folds_horizon_2.parquet`

| Split | Patient Count | Window Count | % Patients | % Windows |
|-------|---------------|--------------|------------|-----------|
| Train | 450 | 13900181 | 69.98% | 70.02% |
| Val | 65 | 1976792 | 10.11% | 9.96% |
| Test | 128 | 3974267 | 19.91% | 20.02% |

## Timings
- **Training**: 0:02:38.551255
- **Prediction**: 0:01:12.406456

## Artifacts
- **Weights**: `models/gMAD_LSTM_H2_Fold3.weights.h5`
- **History csv**: `models/gMAD_LSTM_H2_Fold3_history.csv`
- **Best json**: `models/gMAD_LSTM_H2_Fold3.best.json`
- **Predictions parquet**: `results/predictions/df_test_results_vectors_gMAD_LSTM_H2_Fold3.parquet`
- **Training plot prefix**: `plots/training/gMAD_LSTM_H2_Fold3`
