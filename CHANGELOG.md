# Changelog

All notable changes to `fastLSTM` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-02

Major release: API aligned with the sister package `fastANN`, real multi-step
forecasting, TensorFlow or PyTorch backend, many bug fixes and complete
documentation. Code written for 1.x keeps working: old parameter names are
accepted with a `DeprecationWarning` and models saved by 1.x still load.

### Added
- `backend` parameter: the network runs on Keras 3 with TensorFlow (`'tensorflow'`) or PyTorch (`'torch'`). TensorFlow is no longer required with PyTorch. Saved models reload with either backend.
- Real multi-step forecasting with `steps_ahead` for classificators and regressors; `output_column_names()` names the outputs (`<target>_step_<k>`), `make_multi_step_targets()` builds the targets.
- Loading API shared with fastANN: `load_hyperparameters`, `set_hyperparameters`, `load_model`, `load_scaler`, `load_training_history`.
- `save_X_Y_data`, `history_metrics` parameters; `create_generators()`, `gradient_feature_importance()`, `compute_gradients()` methods.
- `load_keras()` and `make_sequence_generator()` module functions (the latter replaces Keras' `TimeseriesGenerator`, with identical windows and targets).
- Hyperparameters JSON now also stores `LSTM_type`, `class_weight`, `X_feature_names`, `Y_feature_names`, `Y_scaler_file_name` and `backend`.
- Complete English docstrings for every function, a user manual in the README, this changelog and `fastLSTM.__version__`.

### Changed
- Parameters renamed as in fastANN: `scaler` → `scaler_type`, `metric` → `metrics` (list), `check_point_metric` → `checkpoint_monitor_metric`, `early_stop_condittion` → `early_stop_monitor_metric`, `metric_mode` → `checkpoint_mode` + `early_stop_mode`, `scale_target` → `scale_targets`.
- `binary_network_predictions_evaluation()` → `network_predictions_evaluation()`, which also returns the probabilities (the old name keeps its old return values).
- Default `checkpoint_monitor_metric` is now `'val_accuracy'` (was `'accuracy'`), as in fastANN.
- `X_data` and `Y_data` are optional, so `fastLSTM().load_all(...)` works.
- The features scaler is `self.scaler` (was `self.X_scaler`); scalers are saved in separate files (`SCALER` and `Y SCALER`), the old single-file format still loads.
- `early_stop_patience_set(patience=None)`, `split_and_scale(scaler_fit=False)` and `load_training_history(training_history_file_name=None, file_path_name=None)` follow fastANN's signatures.
- `network_training` checks the network and `timesteps` before writing any file; `timesteps` must match the built network.
- Loaded models are recompiled with the saved loss, metrics and learning rate (fresh optimizer state).
- Keras/TensorFlow is no longer imported when the module is imported.

### Fixed
- A single hidden layer (`model_relative_width` of length 1) produced one prediction per timestep (3D output) instead of one per sample.
- `steps_ahead > 1` did not predict future steps: all outputs learned the same next value.
- A second `network_training()` on the same object crashed (the checkpoint callback overwrote the method).
- `binary_precision_recall_vs_scoring()` called a method that did not exist.
- Unrecognised losses (e.g. `'mae'`) were silently replaced by binary cross-entropy.
- `load_all()` could not load a model folder that had been moved or copied.
- Evaluation after `split_and_scale()` on new data used the previous test set.
- `LSTM_type` and `class_weight` were not saved.
- Import failed with SciPy ≥ 1.14 (`simps`) and Plotly ≥ 7 (`create_candlestick`).

### Deprecated
- The old parameter names listed above, `binary_network_predictions_evaluation()` and `load_training_history(<csv path>)`.

## [1.0.0] - 2025-03-07

- First version: stacked-LSTM classificator/regressor with sequential split, scaling, `TimeseriesGenerator` samples, early stopping, checkpointing and saving/loading of model, scalers, history, hyperparameters and data.
