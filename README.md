# fastLSTM: A Structured Framework for LSTM Networks

## Overview

### What is fastLSTM?
`fastLSTM` is a structured framework designed to simplify the creation and training of Long Short-Term Memory (LSTM) models for both classification and regression tasks. Unlike manual LSTM model construction in TensorFlow/Keras, `fastLSTM` streamlines the process by automating key aspects such as:
- **Data structuring with generators**: Automatically aligns `timesteps` to match input-output sequences.
- **Optimized model architecture**: Correctly initializes the first and last layers, avoiding common issues in LSTM design.
- **Scalability and data preprocessing**: Integrates automated scaling and dataset splitting.

### Why Use fastLSTM?
- **Automates LSTM structuring**: Eliminates the need for manual sequence preparation.
- **Ensures proper layer structuring**: Avoids errors in input and output dimensions.
- **Pre-built training mechanisms**: Includes early stopping and checkpointing of the best model.
- **Supports multiple loss functions and scalers**: Enables flexibility in various ML tasks.

---

## Hyperparameters

Parameter names, method names and saved files follow the sister package `fastANN`, so the same keyword-argument workflow works with both classes; only the parameters that make sense just for recurrent networks (`LSTM_type`, `timesteps`, `steps_ahead`, `class_weight`) are LSTM-specific. fastANN options not available here: pre-split `X_train_s`/`Y_train`/`X_test_s`/`Y_test` inputs, `split_type` (the split is always sequential) and `autoencoder_mode`.

### Model Architecture Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `model_relative_width`     | List with the width of each **hidden** LSTM layer, relative to the number of input features (e.g. `[2, 1]` = two hidden layers with `2 * n_features` and `n_features` units). Input and output layers are added automatically. |
| `model_dropout`            | Dropout rate after each hidden layer (same length as `model_relative_width`). |
| `LSTM_type`                | `'classificator'` (output activation `last_layer_activation`) or `'regressor'` (linear output). |
| `activation`               | Activation function for LSTM layers (e.g. `'tanh'`, `'relu'`). |
| `last_layer_activation`    | Activation function of the output layer of classificators (e.g. `'sigmoid'`, `'softmax'`). |

### Data Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `X_data`, `Y_data`         | Features and targets (DataFrames, chronological order). Optional when the model is restored with `load_all`. |
| `scaler_type`              | `'StandardScaler'` (default) or `'MinMaxScaler'`. |
| `scale_targets`            | If `True` targets are scaled too and `model_predict` can descale predictions. |
| `train_size_rate`          | Fraction of rows used for training (default `0.7`); the split is always sequential. |
| `timesteps`                | Number of past rows in each input sequence. |
| `steps_ahead`              | Output-units multiplier for regressors. Targets are not shifted automatically: keep `1` and, to forecast several steps, put the future values as separate `Y_data` columns. |
| `save_X_Y_data`            | If `True` (default) `X_data` and `Y_data` are saved with the model. |

### Training Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `learning_rate`            | Learning rate of the Adam optimizer. Default is `0.0003`. |
| `loss`                     | Loss function (e.g. `'binary_crossentropy'`, `'mse'`, `'CategoricalCrossentropy'`). |
| `metrics`                  | List of evaluation metrics (e.g. `['accuracy']`). |
| `batch_size`               | Batch size for training, default `128`. |
| `class_weight`             | Optional class weights, e.g. `{0: 1.0, 1: 3.0}`. |
| `history_metrics`          | Training-history columns plotted by `plot_training_history` (default depends on `LSTM_type`). |

### Early Stopping and Checkpoints
| Parameter                    | Description |
|------------------------------|-------------|
| `early_stop_monitor_metric`  | Metric monitored for early stopping (default `'val_accuracy'`). |
| `early_stop_mode`            | `'max'` or `'min'` depending on `early_stop_monitor_metric`. |
| `early_stop_patience`        | Number of epochs to wait before stopping if no improvement is detected. |
| `checkpoint_monitor_metric`  | Metric used to save the best-performing model (default `'val_accuracy'`). |
| `checkpoint_mode`            | `'max'` or `'min'` depending on `checkpoint_monitor_metric`. |
| `save_best_only`             | If `True`, saves only the best model during training. |

### Renamed parameters
The old names are still accepted (with a `DeprecationWarning`), and hyperparameters files saved with them can still be loaded.

| Old name                | New name |
|-------------------------|----------|
| `scaler`                | `scaler_type` |
| `metric`                | `metrics` |
| `check_point_metric`    | `checkpoint_monitor_metric` |
| `early_stop_condittion` | `early_stop_monitor_metric` |
| `metric_mode`           | `checkpoint_mode` and `early_stop_mode` |
| `scale_target`          | `scale_targets` |
| `binary_network_predictions_evaluation()` | `network_predictions_evaluation()` (returns `predictions_df` too; the old name keeps the old return values) |
| `load_training_history(file_path_name=<csv path>)` | `load_training_history(training_history_file_name, file_path_name=<folder>)` |

---

## Model and Data Saving Mechanisms

fastLSTM includes built-in functionalities to save models, data, scalers, hyperparameters, and training history, ensuring full reproducibility and ease of use. Every file name starts with the training timestamp and ends with `model_name`; all files are written in `data_storage_path`.

### **Model Saving**
- The best-performing model is automatically saved based on the checkpoint metric.
- Stored in `.keras` format with a timestamped filename.
- Example filename: `2025-03-07 10-00-00 - LSTM MODEL - fastLSTM.keras`

### **Data Saving**
- If `save_X_Y_data=True`, the training dataset (`X_data` and `Y_data`) is saved as `.csv` files (without the index).
- Example filenames:
  - `2025-03-07 10-00-00 - X_data FOR LSTM MODEL - fastLSTM.csv`
  - `2025-03-07 10-00-00 - Y_data FOR LSTM MODEL - fastLSTM.csv`

### **Scaler Saving**
- Input and (when `scale_targets=True`) target scalers are stored as `.pkl` files.
- Example filenames:
  - `2025-03-07 10-00-00 - SCALER FOR LSTM MODEL - fastLSTM.pkl`
  - `2025-03-07 10-00-00 - Y SCALER FOR LSTM MODEL - fastLSTM.pkl`

### **Training History Saving**
- Training history (loss and metrics per epoch) is stored in a `.csv` file and plotted by `plot_training_history()`.
- Example filename:
  - `2025-03-07 10-00-00 - TRAINING HISTORY OF LSTM MODEL - fastLSTM.csv`

### **Hyperparameters Saving**
- Model hyperparameters are stored in a `.json` file.
- Example filename:
  - `2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - fastLSTM.json`

### **Loading Saved Models and Data**
To reload a trained model with all its settings:
```python
model = fastLSTM()
model.load_all("2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - fastLSTM.json", file_path_name = "./models/")
```
This restores the hyperparameters, model, scalers, training history and dataset split. Files are read from the folder of the JSON (`file_path_name`), so a model folder can be moved or copied to another machine. The single steps are also available as `load_hyperparameters`, `set_hyperparameters`, `load_model`, `load_scaler` and `load_training_history`.

---

## Example

```python
from fastLSTM import fastLSTM

lstm = fastLSTM(X_data = X_df,
                Y_data = Y_df[['up']],
                model_relative_width = [2, 1],     # two hidden LSTM layers
                model_dropout = [0.2, 0.1],
                timesteps = 20,
                class_weight = {0: 1.0, 1: 2.0},
                data_storage_path = './models/',
                model_name = 'direction')

lstm.network_structure_set_compile()
lstm.network_training(epochs = 200, batch_size = 64)

results_df, probabilities_df = lstm.network_predictions_evaluation(min_probability = 0.5)
pr_df = lstm.binary_precision_recall_vs_scoring(n_points = 30)

sample = lstm.prepare_input_sample(X_df, len(X_df) - 1, apply_scaler = False)
prediction = lstm.model_predict(sample)
```

---

## Function Parameters and Inputs

Every method has a complete docstring (`help(fastLSTM.network_training)`); the main ones are summarised here.

### `network_structure_set_compile(timesteps=None)`
Builds one LSTM + Dropout block per element of `model_relative_width`, adds the input and output layers and compiles the network.
#### **Parameters:**
- `timesteps` (int, optional): Number of timesteps for input sequences.

### `network_training(epochs, batch_size=None, timesteps=None)`
Trains the model using a sequence generator and saves model, scalers, history, hyperparameters and data.
#### **Parameters:**
- `epochs` (int): Maximum number of epochs.
- `batch_size` (int, optional): Training batch size.
- `timesteps` (int, optional): Overrides default timesteps (the network must be built with the same value).

### `model_predict(data, apply_scaler=True, descale_result=True)`
Generates predictions from input sequences.
#### **Parameters:**
- `data` (array): Sequences of shape `(n_samples, timesteps, n_features)` or a single `(timesteps, n_features)` sequence (see `prepare_input_sample`).
- `apply_scaler` (bool): If `True`, applies feature scaling.
- `descale_result` (bool): If `True`, reverses output scaling (when `scale_targets=True`).

### `network_predictions_evaluation(min_probability, output_dict=False)`
Evaluates a binary classificator on the test set with a probability threshold.
#### **Parameters:**
- `min_probability` (float): Minimum probability threshold for classification.
- `output_dict` (bool): If `True`, also returns the classification report as a dictionary.
#### **Returns:**
- `(filtered_predictions_results_df, predictions_df)` or `(filtered_predictions_results_df, predictions_df, report)`.

### `binary_precision_recall_vs_scoring(n_points=15, plot=True)`
Precision and recall of class `1` for cutoffs from `n_points / 100` to `0.99`.

### `gradient_feature_importance(feature_names=None)`
Gradient-based feature importance on the test set.

### `plot_training_history()`
Plots the training history columns listed in `history_metrics`.

### `load_all(hyperparameters_file_name=None, file_path_name=None)`
Loads a trained model, hyperparameters, scalers, training history and data.
#### **Parameters:**
- `hyperparameters_file_name` (str): JSON file with saved hyperparameters.
- `file_path_name` (str, optional): Folder of the saved files.

---

## Conclusion
`fastLSTM` automates dataset structuring, `timesteps` alignment and correct layer initialization, making it an ideal solution for LSTM-based sequence modeling. By integrating best practices such as early stopping, checkpointing, and data scaling, `fastLSTM` provides an efficient and robust deep learning workflow. 🚀

