# fastLSTM: A Structured Framework for LSTM Networks

## Overview

### What is fastLSTM?
`fastLSTM` is a structured framework designed to simplify the creation and training of Long Short-Term Memory (LSTM) models for both classification and regression tasks. Unlike manual LSTM model construction in TensorFlow/Keras, `fastLSTM` streamlines the process by automating key aspects such as:
- **Data structuring with generators**: Automatically aligns `timesteps` to match input-output sequences.
- **Steps-ahead prediction**: Supports multi-period forecasting with automated handling of `steps_ahead`.
- **Optimized model architecture**: Correctly initializes the first and last layers, avoiding common issues in LSTM design.
- **Scalability and data preprocessing**: Integrates automated scaling and dataset splitting.

### Why Use fastLSTM?
- **Automates LSTM structuring**: Eliminates the need for manual sequence preparation.
- **Handles `steps_ahead` predictions**: Adjusts model output shape automatically for multi-step forecasting.
- **Ensures proper layer structuring**: Avoids errors in input and output dimensions.
- **Pre-built training mechanisms**: Includes early stopping, checkpointing, and learning rate scheduling.
- **Supports multiple loss functions and scalers**: Enables flexibility in various ML tasks.

---

## Hyperparameters

### Model Architecture Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `model_relative_width`     | List defining the relative width of hidden layers compared to the input size. |
| `model_dropout`            | Dropout values for each layer to prevent overfitting. |
| `LSTM_type`                | `'classificator'` or `'regressor'`, determining output activation. |
| `activation`               | Activation function for LSTM layers (e.g., 'tanh', 'relu'). |
| `last_layer_activation`    | Activation function for the output layer (e.g., 'sigmoid', 'linear'). |

### Training Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `learning_rate`            | Learning rate for model training. Default is `0.0003`. |
| `loss`                     | Loss function (e.g., 'binary_crossentropy', 'mse'). |
| `metric`                   | Evaluation metric (e.g., `'accuracy'`). |
| `batch_size`               | Batch size for training, default `128`. |
| `timesteps`                | Number of previous timesteps to consider as input. |
| `steps_ahead`              | Number of future steps the model should predict. |
| `train_size_rate`          | Fraction of data used for training (default: `0.7`). |

### Early Stopping and Checkpoints
| Parameter                  | Description |
|----------------------------|-------------|
| `early_stop_condittion`    | Metric monitored for early stopping (e.g., `'val_accuracy'`). |
| `check_point_metric`       | Metric used to save the best-performing model. |
| `metric_mode`              | `'max'` or `'min'` depending on metric type. |
| `early_stop_patience`      | Number of epochs to wait before stopping if no improvement is detected. |
| `save_best_only`           | If `True`, saves only the best model during training. |

---

## Model and Data Saving Mechanisms

fastLSTM includes built-in functionalities to save models, data, scalers, hyperparameters, and training history, ensuring full reproducibility and ease of use.

### **Model Saving**
- The best-performing model is automatically saved based on the checkpoint metric.
- Stored in `.keras` format with a timestamped filename.
- Example filename: `2025-03-07 - LSTM MODEL - fastLSTM.keras`

### **Data Saving**
- If `save_X_Y_data=True`, the training dataset (`X_data` and `Y_data`) is saved as `.csv` files.
- Example filenames:
  - `2025-03-07 - X_data FOR LSTM MODEL - fastLSTM.csv`
  - `2025-03-07 - Y_data FOR LSTM MODEL - fastLSTM.csv`

### **Scaler Saving**
- Input and target scalers are stored as `.pkl` files.
- Example filenames:
  - `2025-03-07 - SCALER FOR LSTM MODEL - fastLSTM.pkl`
  - `2025-03-07 - Y SCALER FOR LSTM MODEL - fastLSTM.pkl`

### **Training History and Learning Curves Saving**
- Training history (loss, accuracy) is stored in a `.csv` file.
- Learning curves are saved as `.png` images for visualization.
- Example filenames:
  - `2025-03-07 - TRAINING HISTORY OF LSTM MODEL - fastLSTM.csv`
  - `2025-03-07 - TRAINING LOSS CURVE - fastLSTM.png`
  - `2025-03-07 - TRAINING ACCURACY CURVE - fastLSTM.png`

### **Hyperparameters Saving**
- Model hyperparameters are stored in a `.json` file.
- Example filename:
  - `2025-03-07 - HYPERPARAMETERS OF LSTM MODEL - fastLSTM.json`

### **Loading Saved Models and Data**
To reload a trained model with all its settings:
```python
model.load_all("hyperparameters.json")
```
This restores the model, scalers, training history, and dataset split.

---

## Function Parameters and Inputs

### `network_structure_set_compile(timesteps=None)`
Compiles and structures the LSTM network.
#### **Parameters:**
- `timesteps` (int, optional): Number of timesteps for input sequences.

### `network_training(epochs, batch_size=None, timesteps=None)`
Trains the model using a sequence generator.
#### **Parameters:**
- `epochs` (int): Number of training iterations.
- `batch_size` (int, optional): Training batch size.
- `timesteps` (int, optional): Overrides default timesteps.

### `model_predict(data, apply_scaler=True, descale_result=True)`
Generates predictions based on input sequences.
#### **Parameters:**
- `data` (array-like): Input data.
- `apply_scaler` (bool): If `True`, applies feature scaling.
- `descale_result` (bool): If `True`, reverses output scaling.

### `network_predictions_evaluation(min_probability, output_dict=False)`
Evaluates model predictions.
#### **Parameters:**
- `min_probability` (float): Minimum probability threshold for classification.
- `output_dict` (bool): If `True`, returns detailed evaluation metrics.

### `plot_training_history()`
Plots the training history of loss and accuracy.
#### **Parameters:**
- None (uses stored training history).

### `load_all(hyperparameters_file_name)`
Loads a trained model, hyperparameters, and data.
#### **Parameters:**
- `hyperparameters_file_name` (str): JSON file with saved hyperparameters.

---

## Conclusion
`fastLSTM` automates dataset structuring, `timesteps` alignment, multi-period forecasting, and correct layer initialization, making it an ideal solution for LSTM-based sequence modeling. By integrating best practices such as early stopping, checkpointing, and data scaling, `fastLSTM` provides an efficient and robust deep learning workflow. 🚀

