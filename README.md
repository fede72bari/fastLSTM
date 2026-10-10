# fastLSTM

**A structured framework to build, train, version and reuse LSTM networks for time series, with a few lines of code.**

## Overview: why fastLSTM

Training a recurrent network "by hand" with Keras means writing, every time, the same fragile plumbing: splitting the data without look-ahead, fitting scalers only on the training set, reshaping tables into `(samples, timesteps, features)` sequences, aligning each window with the right target, sizing input and output layers, setting `return_sequences` correctly on stacked layers, adding early stopping and checkpoints, and then saving the model **together with** the scalers, the data and the settings that produced it. Most bugs in time-series models live in this plumbing (look-ahead leaks, misaligned targets, a model reloaded with the wrong scaler), and most of the time spent on experiments goes into rewriting it.

`fastLSTM` turns that plumbing into a tested, reusable class, so a data scientist can focus on the questions that matter: *which features, which targets, which architecture, which horizon*.

What it does for you:

- **Correct data preparation** — sequential train/test split (no shuffling of time), scalers fitted on the training set only, sequences built by `TimeseriesGenerator` with a documented alignment between input window and target.
- **Architecture from a short description** — `model_relative_width = [2, 1]` means "two hidden LSTM layers, twice and once the number of features"; input and output layers are sized automatically, also for multiple targets and multi-step horizons.
- **Multi-step forecasting** — `steps_ahead = k` trains the network on the next `k` values of every target at once, for regression and classification.
- **Training best practices built in** — early stopping, checkpoint of the best epoch (reloaded at the end), class weights, training-history plots.
- **Reproducibility and model versioning** — every training run is saved as a self-describing, timestamped set of files (model, scalers, data, hyperparameters, history) and restored with one call (see [Model versioning](#model-versioning-runs-datasets-and-hyperparameters)).
- **Evaluation and use** — classification reports per output, precision/recall vs probability cutoff, gradient-based feature importance, ready-to-use prediction on new data with automatic scaling/descaling.
- **TensorFlow, PyTorch or JAX** — choose the framework when creating the instance (`backend = 'tensorflow'`, `'torch'` or `'jax'`); the same code, files and results work with all three, and a model trained with one backend can be reloaded with another.
- **Several GPUs or a TPU** — the same run trains on the two GPUs of a Kaggle "GPU T4 x2" session (`distribution_strategy`) or on the 8 cores of a TPU v5e-8 (JAX data parallelism): the batches are completed automatically so that they split evenly among the devices (see [Training on several devices](#training-on-several-devices-two-gpus-tpu)).
- **One workflow for two model families** — `fastLSTM` shares parameter names, method names and saved-file layout with its sister package [`fastANN`](https://github.com/fede72bari/fastANN) (dense networks): the same code pattern trains, saves and reloads both, so they can be compared on the same data.

Typical uses: price/return forecasting, direction (up/down) classification, multi-horizon forecasts, any sequence-to-value problem on tabular time series.

---

**Current version: 2.4.1** (`fastLSTM.__version__`) — see the [CHANGELOG](CHANGELOG.md).

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Key concepts](#key-concepts)
4. [Constructor parameters](#constructor-parameters)
5. [Methods reference](#methods-reference)
6. [Saved files](#saved-files)
7. [Model versioning: runs, datasets and hyperparameters](#model-versioning-runs-datasets-and-hyperparameters)
8. [Examples](#examples)
9. [Backward compatibility](#backward-compatibility)
10. [Tips and caveats](#tips-and-caveats)

---

## Installation

The package is the `fastLSTM` folder of this repository (no `pip` package yet).

```bash
git clone https://github.com/fede72bari/fastLSTM.git
```

Then either copy the inner `fastLSTM/` folder next to your notebook/script, or add the repository folder to the Python path:

```python
import sys
sys.path.append('/path/to/fastLSTM')   # the cloned repository folder
from fastLSTM import fastLSTM
```

### Requirements

Python 3.9+ and:

| Purpose | Packages |
|---|---|
| Core | `keras` 3 with **one** backend: `tensorflow` (2.16+), `torch` **or** `jax`; `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib` |
| Plots and notebooks | `matplotlib`, `plotly`, `ipython` |
| Imported by the module (shared toolbox) | `xgboost`, `seaborn`, `tabulate`, `statsmodels`, `imbalanced-learn`, `deap`, `yfinance`, `pytz` |

```bash
pip install scikit-learn pandas numpy scipy joblib matplotlib plotly ipython \
            xgboost seaborn tabulate statsmodels imbalanced-learn deap yfinance pytz
pip install tensorflow          # TensorFlow backend (includes Keras 3)
pip install keras torch         # PyTorch backend (TensorFlow is then optional)
```

Tested with Keras 3.15 on TensorFlow 2.21 and on PyTorch 2.14, pandas 3.0, NumPy 2.4, scikit-learn 1.9, SciPy 1.17 and Plotly 7.

---

## Quick start

```python
import pandas as pd
from fastLSTM import fastLSTM

# X_df: features, one row per bar, chronological order
# Y_df: targets aligned with X_df (here a 0/1 column 'up')

lstm = fastLSTM(X_data = X_df,
                Y_data = Y_df[['up']],
                model_relative_width = [2, 1],       # two hidden LSTM layers
                model_dropout = [0.2, 0.1],
                timesteps = 20,                      # each sample sees the last 20 bars
                data_storage_path = './models/',     # must end with a separator
                model_name = 'direction')

lstm.network_structure_set_compile()                  # build + compile
lstm.network_training(epochs = 200, batch_size = 64)  # train, save everything, reload best epoch

results_df, probabilities_df = lstm.network_predictions_evaluation(min_probability = 0.5)
```

Later, in another session:

```python
lstm = fastLSTM()
lstm.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - direction.json',
              file_path_name = './models/')

sample = lstm.prepare_input_sample(new_X_df, len(new_X_df) - 1, apply_scaler = False)
prediction = lstm.model_predict(sample)               # scaled inside, descaled if needed
```

---

## Key concepts

### Network architecture

`model_relative_width` lists the **hidden** LSTM layers only; the input and output layers are always added automatically:

```
Input(timesteps, n_features)
LSTM(n_features * model_relative_width[0])  + Dropout(model_dropout[0])
...
LSTM(n_features * model_relative_width[-1]) + Dropout(model_dropout[-1])
Dense(n_targets * steps_ahead)              # output
```

- Widths are **relative to the number of features**: with 10 features, `[2, 1]` gives layers of 20 and 10 units.
- `model_dropout` must have the same length as `model_relative_width`.
- Every LSTM layer except the last returns the whole sequence; the last returns only its final state, which feeds the output layer (this also holds with a single hidden layer).

### How samples are built (`timesteps`)

Samples are created with Keras' `TimeseriesGenerator`: target row **`t`** is paired with the feature rows **`t - timesteps … t - 1`**. Row `t` itself is *not* in the input window, so:

- `Y_data` row `t` is what the network learns to predict *after* seeing the window that ends at row `t - 1`;
- the first `timesteps` rows of each set have no prediction.

### Multi-step forecasting (`steps_ahead`)

With `steps_ahead = k`, every sample is trained on the next `k` target rows at once, `Y[t], Y[t+1], …, Y[t+k-1]`, and the network has `n_targets * k` outputs. Outputs are **step-major** and named by `output_column_names()`:

```python
lstm.output_column_names()
# Y_data columns ['y', 'y2'], steps_ahead = 3:
# ['y_step_1', 'y2_step_1', 'y_step_2', 'y2_step_2', 'y_step_3', 'y2_step_3']
```

It works for regressors (next `k` values) and classificators (e.g. "will it go up at step 1, 2, 3?"). The last `k - 1` rows of each set have no complete future and are not used as samples. With `steps_ahead = 1` (default) there is one output per target column.

### Choosing TensorFlow or PyTorch

The network is written with Keras 3, which runs on top of TensorFlow or PyTorch. Choose the backend when you create the first instance:

```python
model = fastLSTM(X_data = X_df, Y_data = Y_df, backend = 'torch')       # or 'tensorflow'
print(model.backend)                                               # 'torch'
```

- `backend = None` (default) uses the backend already active, or the `KERAS_BACKEND` environment variable, or `'tensorflow'`.
- Keras uses **one backend per Python process**: the first instance fixes it. Asking for a different one later raises a clear error; restart the kernel to switch.
- Saved models are portable: a model trained with TensorFlow can be reloaded with PyTorch and vice versa (`load_all` reloads the weights and recompiles the network with the saved loss, metrics and learning rate).
- With PyTorch, create the first instance (or `import torch`) **before** anything that imports TensorFlow: with some TensorFlow/PyTorch builds, loading the Keras PyTorch backend after TensorFlow crashes Python.
- **JAX for TPUs**: `backend = 'jax'` runs the same network on JAX, the backend to use on TPUs (e.g. Kaggle TPU v5e-8, Google Colab TPU). Models are portable across the three backends. On GPU prefer TensorFlow for LSTMs (cuDNN kernels).
- `model.keras` is the Keras module in use; `model.model` is a regular Keras model on either backend.

### Training on several devices (two GPUs, TPU)

Data parallelism: every batch is split among the devices, each one computes the gradients of its share and the weights are updated with their average. Network, results and saved files are the same as on one device; only the time per epoch changes.

**Two or more GPUs (TensorFlow, e.g. Kaggle "GPU T4 x2")**

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()          # all the visible GPUs
lstm = fastLSTM(X_data = X_df, Y_data = Y_df, timesteps = 8, batch_size = 4096,
                distribution_strategy = strategy, data_storage_path = './models/')
lstm.network_structure_set_compile()                 # built and compiled inside strategy.scope()
lstm.network_training(epochs = 100)                  # each GPU gets 4096 / 2 windows per step
```

- fastLSTM builds, compiles and reloads the network inside `strategy.scope()`. Entering the scope by hand (`strategy.scope().__enter__()` at the top of a script) is **not** enough with Keras 3: the variables would not be distributed and `fit` fails with ``colocate_vars_with must only be passed a variable created in this tf.distribute.Strategy.scope()``.
- `batch_size` is the global batch, split among the GPUs (4096 on two GPUs = 2048 windows each per step). The same batch gives the same training as on one GPU and is faster when one GPU was saturated by the whole batch; a larger batch uses the GPUs better but changes the optimisation (fewer updates per epoch, learning rate to revise).
- Without `distribution_strategy` TensorFlow trains on **one** GPU even when two are visible.

**TPU (JAX, e.g. Kaggle TPU v5e-8, Google Colab TPU)**

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
import jax, keras

keras.distribution.set_distribution(keras.distribution.DataParallel(devices = jax.devices()))   # 8 cores on a v5e-8
lstm = fastLSTM(X_data = X_df, Y_data = Y_df, timesteps = 8, batch_size = 4096, backend = 'jax',
                data_storage_path = './models/')
lstm.network_structure_set_compile()
lstm.network_training(epochs = 100)
probabilities = lstm.predict_validation()            # one row per real validation window
```

- With a Keras distribution every batch must split evenly among the devices, otherwise JAX stops with `IndivisibleError` (typically on the last, partial batch of the epoch, e.g. 335 windows on 8 cores). fastLSTM completes each batch to a multiple of the number of devices by repeating its last window (`batch_multiple`, read from the active distribution: 8 on a v5e-8, 1 without distribution); `batch_size` must be a multiple of it. At most `n_devices - 1` copies of one window are added to the last batch of each epoch.
- `predict_validation()` returns the validation predictions cut back to the real windows (`validation_generator.n_samples`); the AUC monitor and `network_predictions_evaluation` use it. Use it instead of `model.predict(validation_generator)`, which also returns the repeated windows.
- A TPU can be opened by one process only: if the notebook calls `jax.devices()` and then trains in a subprocess, the subprocess finds the TPU busy. Train in the process that opened it, or open it only in the subprocess.
- Colab TPU runtimes have no TensorFlow: use `backend = 'jax'` (models trained there reload with TensorFlow or PyTorch).

**Which accelerator for LSTMs.** On GPU the LSTM layers run on the fused cuDNN kernel only with `activation = 'tanh'` (default), sigmoid recurrent activation, no recurrent dropout and no unrolling. fastLSTM keeps all of these conditions (dropout is a separate layer between the LSTMs, the `input_projection` comes before them), so change `activation` only knowingly: other activations fall back to the generic kernel, several times slower. On TPU the network is compiled by XLA and the activation does not change the kernel.

### Split and scaling

- The split is always **sequential** (first `train_size_rate` of the rows for training, the rest for test) to avoid look-ahead.
- Features are scaled with `StandardScaler` or `MinMaxScaler` (`scaler_type`), fitted on the training set only.
- Targets are scaled only with `scale_targets = True` (useful for regressors); `model_predict` then returns predictions in the original scale.

### Classificator vs regressor

| `LSTM_type` | Output activation | Typical loss | Typical monitor |
|---|---|---|---|
| `'classificator'` (default) | `last_layer_activation` (`'sigmoid'`) | `'binary_crossentropy'` | `'val_accuracy'`, mode `'max'` |
| `'regressor'` | linear | `'mse'`, `'mae'`, `'huber'` | `'val_loss'`, mode `'min'` |

---

## Constructor parameters

`fastLSTM(**parameters)` — all parameters are optional keywords.

### Data

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `X_data` | `DataFrame` | `None` | Features, one row per time step, chronological. Required to train; omit it when restoring with `load_all()`. |
| `Y_data` | `DataFrame` | `None` | Targets aligned with `X_data` (one column per target; 0/1 for binary classification). |
| `scaler_type` | `'StandardScaler'`, `'MinMaxScaler'` | `'StandardScaler'` | Scaler for features (and targets if scaled). |
| `scale_targets` | `bool` | `False` | Scale the targets too; predictions are descaled by `model_predict`. |
| `train_size_rate` | `float` 0–1 | `0.7` | Fraction of rows used for training (sequential split). |
| `timesteps` | `int` | `1` | Length of each input sequence (past rows seen per prediction). |
| `steps_ahead` | `int` | `1` | Number of future steps predicted for each target (see [Multi-step forecasting](#multi-step-forecasting-steps_ahead)). |
| `save_X_Y_data` | `bool` | `True` | Save `X_data`/`Y_data` as CSV at training time, so `load_all()` can rebuild the same split. |

### Architecture

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `model_relative_width` | `list` of `float` | `[1]` | Width of each hidden LSTM layer relative to the number of features. Its length = number of hidden layers. |
| `model_dropout` | `list` of `float` 0–1 | `[0]` | Dropout after each hidden layer (same length as `model_relative_width`). |
| `LSTM_type` | `'classificator'`, `'regressor'` | `'classificator'` | Output layer type (see table above). |
| `activation` | Keras activation name | `'tanh'` | Activation of the LSTM layers. `'tanh'` enables the fast cuDNN kernel on GPU. |
| `input_projection` | `None`, `'gated_fan'` | `None` | `'gated_fan'` adds a gated Fourier Analysis Network layer applied to every bar of the window before the first LSTM: learned periodic components (cosines and sines of learned frequencies, scaled by trainable gates) next to a normal dense part. The LSTMs stay standard (cuDNN on GPU). Same layer as [fastGatedFourierAnalysisNetwork](https://github.com/fede72bari/fastGatedFourierAnalysisNetwork). |
| `input_projection_width` | `float` | `4` | Width of the projection relative to the number of features (the LSTM widths stay relative to the number of features). |
| `input_projection_dropout` | `float` 0–1 | `0.0` | Dropout after the projection. |
| `input_projection_activation` | Keras activation name | `'gelu'` | Activation of the non-periodic part of the projection. |
| `periodic_share`, `gated`, `frequency_init_std` | `float`, `bool`, `float` | `1/3`, `True`, `1.0` | Share of the projection given to the periodic part, trainable gates on/off, scale of the initial frequencies. |
| `last_layer_activation` | `'sigmoid'`, `'softmax'`, … | `'sigmoid'` | Output activation of classificators. Use `'softmax'` only for one-hot classes with `steps_ahead = 1`. |

### Training

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `0.0003` | Adam learning rate. |
| `loss` | Keras loss name or object | `'binary_crossentropy'` | e.g. `'mse'`, `'mae'`, `'categorical_crossentropy'`; class names `'BinaryCrossentropy'`, `'CategoricalCrossentropy'`, `'SparseCategoricalCrossentropy'` are also accepted. |
| `metrics` | `list` of `str` (or `str`) | `['accuracy']` | Metrics logged by Keras (validation ones get the `val_` prefix). |
| `batch_size` | `int` | `128` | Batch size of the generators (can be overridden in `network_training`). |
| `class_weight` | `dict` | `None` | e.g. `{0: 1.0, 1: 3.0}` to rebalance classes. Meaningful with a single output. |
| `sequence_groups` | array-like | `None` | Group label of each row (e.g. option contract id) when the rows hold several interleaved series: windows and multi-step targets use only rows of the same group, so a sample never mixes two contracts. Rows must be chronological. Not saved in the files: pass it again after `load_all`. |
| `shuffle` | `bool` | `False` | Permute the training windows among the batches at every epoch. Each window (its `timesteps` rows in chronological order, and its targets) is unchanged and the train/test split stays chronological; the test set is never shuffled. Recommended for autocorrelated targets (trend/ZigZag labels): in chronological order each batch holds consecutive days, often of a single class, and the learning curves jump from epoch to epoch. |
| `sample_weight` | array-like | `None` | One weight per row of `X_data`: the training rows weight the loss (e.g. larger weights for the hard cases, such as options with the strike close to the underlying). Validation is not weighted. Not saved in the files (only whether it was used). |
| `monitor_auc` | `bool` | `False` | Compute the ROC AUC of the validation predictions at the end of every epoch and log it as `val_monitored_auc` (binary targets). Use it as `early_stop_monitor_metric` / `checkpoint_monitor_metric` with mode `'max'` to choose the epoch on the AUC. One extra prediction pass on the validation set per epoch. |
| `monitor_auc_rows` | array-like of `bool` | `None` | One value per row of `X_data`: the monitored AUC uses only the selected validation rows (e.g. strike within 2% of the underlying). Implies `monitor_auc = True`. Not saved in the files. |
| `distribution_strategy` | `tf.distribute.Strategy` | `None` | TensorFlow only: the network is built, compiled and loaded inside its scope, so that `fit` trains on all its devices, e.g. `tf.distribute.MirroredStrategy()` for two GPUs (see [Training on several devices](#training-on-several-devices-two-gpus-tpu)). `None` = one device. |
| `batch_multiple` | `int` | `None` | Every batch of the generators is completed to a multiple of it (the last partial batch repeats its last window). `None` uses the number of devices of the active Keras distribution (8 with `keras.distribution.DataParallel` on a TPU v5e-8) and 1 without distribution. `batch_size` must be a multiple of it. |
| `history_metrics` | `list` of `str` | `None` | Columns plotted by `plot_training_history()`. `None` → `['loss', 'val_loss']` for regressors, first metric and its `val_` version for classificators. |

### Early stopping and checkpoint

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `early_stop_monitor_metric` | `str` | `'val_accuracy'` | Quantity monitored by early stopping. |
| `early_stop_mode` | `'max'`, `'min'`, `'auto'` | `'max'` | Whether it must increase or decrease. |
| `early_stop_patience` | `int` | `200` | Epochs without improvement before stopping. |
| `checkpoint_monitor_metric` | `str` | `'val_accuracy'` | Quantity used to pick the best epoch to save and reload. |
| `checkpoint_mode` | `'max'`, `'min'`, `'auto'` | `'max'` | Whether it must increase or decrease. |
| `save_best_only` | `bool` | `True` | Save only improving epochs (otherwise the last epoch is kept). |

### Storage

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_storage_path` | `str` | `'\\cyPredict\\'` | Folder for every saved file; it is concatenated to file names, so it **must end with a separator** (`'./models/'`). |
| `model_name` | `str` | `'LSTM'` | Name used in every saved file name. |
| `backend` | `'tensorflow'`, `'torch'`, `'jax'` | `None` | Framework that runs the network (see [Choosing TensorFlow or PyTorch](#choosing-tensorflow-or-pytorch)). |

---

## Methods reference

Every method has a complete docstring: `help(fastLSTM.network_training)`.

### Build and train

| Method | Description |
|---|---|
| `network_structure_set_compile(timesteps=None)` | Builds the network (see [architecture](#network-architecture)) and compiles it with Adam, `loss` and `metrics`. `timesteps` optionally changes the sequence length. The text summary is kept in `model_summary`. |
| `network_training(epochs, batch_size=None, timesteps=None, callbacks=None)` | Trains with early stopping and checkpointing (plus any extra Keras `callbacks`), saves every artefact (see [Saved files](#saved-files)), reloads the best epoch into `model` and plots the history. `timesteps` must match the built network. |
| `create_generators(batch_size=None)` | (Re)creates `generator` (training) and `validation_generator` (test). Called automatically when needed. With several devices every batch is completed to `effective_batch_multiple()`; `generator.n_samples` is the number of real windows. |
| `effective_batch_multiple()` | Multiple every batch is completed to: `batch_multiple`, else the number of devices of the active Keras distribution, else 1. |
| `split_and_scale(scaler_fit=False)` | Sequential split of `X_data`/`Y_data` and scaling. `scaler_fit=True` fits the scalers (new data), `False` only applies them. Called by the constructor with `True`. |
| `set_loss_function(loss)` | Changes the loss (recompile afterwards). |
| `early_stop_patience_set(patience=None)` | Rebuilds the early stopping callback, optionally with a new patience. |
| `checkpoint_callback(save_best_only=None)` | Creates the `ModelCheckpoint` callback (`model_checkpoint`). Called by `network_training`. |

### Evaluate

| Method | Returns | Description |
|---|---|---|
| `network_predictions_evaluation(min_probability, output_dict=False)` | `(results_df, probabilities_df)` or `(results_df, probabilities_df, report)` | Classificators: predictions on the test set thresholded at `min_probability` and compared with the actual values; a `classification_report` is printed for every output. `report` is the dict of the last output. |
| `binary_precision_recall_vs_scoring(n_points=15, plot=True)` | `DataFrame` (`Cutoff`, `Precision`, `Recall`) | Precision and recall of class `1` for cutoffs from `n_points/100` to `0.99` (Plotly chart). Labels must be integers 0/1. |
| `plot_training_history()` | – | Plots the `history_metrics` columns of `loss_df`. |
| `gradient_feature_importance(feature_names=None)` | `(importances, names)` sorted increasingly | Mean absolute gradient of the loss w.r.t. each input feature on the test set (Plotly bar chart). |
| `compute_gradients(inputs, targets)` | tensor | Gradient of the MSE w.r.t. the inputs (used by the method above). |

### Predict

| Method | Returns | Description |
|---|---|---|
| `predict_validation()` | array `(n_windows, n_outputs)` | Predictions on the validation set, one row per real window (the windows repeated for multi-device batches are removed). Aligned with `validation_targets()`. |
| `prepare_input_sample(X, current_datetime_idx, apply_scaler=True)` | array `(1, timesteps, n_features)` | Sequence of the `timesteps` rows ending at position `current_datetime_idx` of `X`. The prediction refers to the following row(s). |
| `model_predict(data, apply_scaler=True, descale_result=True)` | array `(n_samples, n_targets * steps_ahead)` | Predicts on 3D sequences `(n, timesteps, n_features)` or one 2D sequence. Scales the inputs and descales the outputs (if `scale_targets`). |
| `output_column_names()` | `list` of `str` | Names of the prediction columns (`<target>_step_<k>` when `steps_ahead > 1`). |
| `create_sequences(data, window_size)` | `list` | Utility: sliding windows of `window_size` rows. |
| `make_multi_step_targets(Y_values)` | array | Stacks each target row with the following `steps_ahead - 1` rows (used by the generators). |

### Save and load

| Method | Description |
|---|---|
| `load_all(hyperparameters_file_name=None, file_path_name=None)` | Restores everything: hyperparameters, model, scalers, training history and (if saved) data, split and scaled with the loaded scalers. Files are read from `file_path_name` (the folder of the JSON), so model folders can be moved. |
| `load_hyperparameters(file_name, file_path_name=None)` | Reads the JSON into `hyperparameters`. |
| `set_hyperparameters()` | Applies `hyperparameters` to the attributes (old key names accepted). |
| `load_model(model_file_name=None, file_path_name=None)` | Loads the `.keras` model. |
| `load_scaler(scaler_file_name=None, Y_scaler_file_name=None, file_path_name=None)` | Loads the scalers (also the old single-file format). |
| `load_training_history(training_history_file_name=None, file_path_name=None)` | Loads the history CSV into `loss_df`. |
| `save_hyperparameters(file_name)` / `init_hyperparameters(...)` | Write / rebuild the hyperparameters dictionary (called by `network_training`). |

### Main attributes

| Attribute | Content |
|---|---|
| `model` | The Keras model. |
| `scaler`, `Y_scaler` | Features and targets scalers (`Y_scaler` is `None` unless `scale_targets`). |
| `X_train`, `Y_train`, `X_test`, `Y_test` | Unscaled split (DataFrames). |
| `X_train_s`, `X_test_s`, `Y_train_s`, `Y_test_s` | Arrays used for training. |
| `generator`, `validation_generator` | Sample generators. |
| `loss_df` | Training history (one row per epoch). |
| `hyperparameters` | Dictionary saved as JSON. |
| `model_summary` | Text summary of the network. |

---

## Saved files

`network_training()` writes into `data_storage_path` (`<dt>` = training timestamp, `<name>` = `model_name`):

| File | Content |
|---|---|
| `<dt> - LSTM MODEL - <name>.keras` | Best (or last) model. |
| `<dt> - SCALER FOR LSTM MODEL - <name>.pkl` | Features scaler. |
| `<dt> - Y SCALER FOR LSTM MODEL - <name>.pkl` | Targets scaler (only with `scale_targets`). |
| `<dt> - TRAINING HISTORY OF LSTM MODEL - <name>.csv` | Loss and metrics per epoch. |
| `<dt> - HYPERPARAMETERS OF LSTM MODEL - <name>.json` | All settings and the names of the other files. |
| `<dt> - X_data FOR LSTM MODEL - <name>.csv`, `<dt> - Y_data FOR LSTM MODEL - <name>.csv` | Data (only with `save_X_Y_data`; the index is not saved). |

To restore a run you only need the JSON name and its folder: `fastLSTM().load_all(json_name, file_path_name = folder)`.

---

## Model versioning: runs, datasets and hyperparameters

Every call to `network_training()` is a **run**, identified by its timestamp. A run writes a complete, self-describing snapshot: the JSON contains all hyperparameters *and* the names of the model, scaler, history and data files of that same run, so each model version stays linked to the exact dataset and settings that produced it.

```
models/
├── 2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - direction.json   ← entry point of the run
├── 2025-03-07 10-00-00 - LSTM MODEL - direction.keras
├── 2025-03-07 10-00-00 - SCALER FOR LSTM MODEL - direction.pkl
├── 2025-03-07 10-00-00 - TRAINING HISTORY OF LSTM MODEL - direction.csv
├── 2025-03-07 10-00-00 - X_data FOR LSTM MODEL - direction.csv
├── 2025-03-07 10-00-00 - Y_data FOR LSTM MODEL - direction.csv
├── 2025-03-08 15-30-12 - HYPERPARAMETERS OF LSTM MODEL - direction.json   ← a later run, same model name
└── ...
```

What the JSON records: architecture (`model_relative_width`, `model_dropout`, `activation`, …), training settings (`loss`, `metrics`, `learning_rate`, `batch_size`, early stopping and checkpoint settings, `class_weight`), data settings (`timesteps`, `steps_ahead`, `shuffle`, `train_size_rate`, `scaler_type`, `scale_targets`, feature and target column names) and the file names of the run.

### Recommended practices

1. **Keep `save_X_Y_data = True`** (default): the exact training data are saved with the model, so `load_all()` rebuilds the same split and you can always re-evaluate or audit a version.
2. **Use `model_name` for the experiment, the timestamp for the version**: e.g. `model_name = 'direction_v2_20feat'`; every retraining adds a new timestamped run without overwriting the previous ones.
3. **One folder per project** (`data_storage_path`), and move or copy whole folders freely: `load_all(json, file_path_name = new_folder)` reads every file from the folder of the JSON.
4. **Compare versions** by reading their JSON and history files:

```python
import glob, json, os
import pandas as pd

rows = []
for path in glob.glob('./models/* - HYPERPARAMETERS OF LSTM MODEL - direction*.json'):
    hp = json.load(open(path))
    history = pd.read_csv('./models/' + hp['training_history_file_name'], index_col = 0)
    rows.append({'run': hp['model_training_datetime'],
                 'layers': hp['model_relative_width'],
                 'timesteps': hp['timesteps'],
                 'steps_ahead': hp['steps_ahead'],
                 'best_val_accuracy': history['val_accuracy'].max(),
                 'json': os.path.basename(path)})
runs = pd.DataFrame(rows).sort_values('best_val_accuracy', ascending = False)
```

5. **Reload any version** with its JSON name: `fastLSTM().load_all(runs.iloc[0]['json'], file_path_name = './models/')`.
6. **Remember which file to use in production**: the JSON name is the only reference you need to store (in a config file, a database, a Git tag…).
7. For long-term traceability you can version the folder itself (Git LFS, DVC, cloud storage); the file names already carry timestamp and experiment name.

---

## Examples

### 1. Binary classification with class weights and cutoff analysis

```python
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['up']],
                model_relative_width = [2, 1], model_dropout = [0.2, 0.1],
                timesteps = 20, class_weight = {0: 1.0, 1: 2.0},
                early_stop_patience = 30,
                data_storage_path = './models/', model_name = 'direction')
lstm.network_structure_set_compile()
lstm.network_training(epochs = 300, batch_size = 64)

results_df, probs_df, report = lstm.network_predictions_evaluation(0.5, output_dict = True)
print(report['1']['precision'], report['1']['recall'])

pr_df = lstm.binary_precision_recall_vs_scoring(n_points = 30)   # cutoffs 0.30 ... 0.99
best = pr_df.loc[pr_df['Precision'].idxmax()]
```

### 2. Regression with scaled targets

```python
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['return']],
                LSTM_type = 'regressor', loss = 'mse', metrics = ['mae'],
                early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
                checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min',
                scale_targets = True, scaler_type = 'MinMaxScaler',
                timesteps = 30, data_storage_path = './models/', model_name = 'returns')
lstm.network_structure_set_compile()
lstm.network_training(epochs = 200, batch_size = 32)

sample = lstm.prepare_input_sample(X_df, len(X_df) - 1, apply_scaler = False)
next_return = lstm.model_predict(sample)          # already in the original scale
```

### 3. Multi-step forecast: next 5 values of two series

```python
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['price', 'volume']],
                LSTM_type = 'regressor', loss = 'mse', metrics = ['mae'],
                early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
                checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min',
                scale_targets = True, timesteps = 40, steps_ahead = 5,
                model_relative_width = [3, 2], model_dropout = [0.1, 0.1],
                data_storage_path = './models/', model_name = 'forecast5')
lstm.network_structure_set_compile()
lstm.network_training(epochs = 300, batch_size = 64)

sample = lstm.prepare_input_sample(X_df, len(X_df) - 1, apply_scaler = False)
forecast = pd.DataFrame(lstm.model_predict(sample), columns = lstm.output_column_names())
# columns: price_step_1, volume_step_1, price_step_2, volume_step_2, ... price_step_5, volume_step_5
```

### 4. Multi-step classification: "will it go up in each of the next 3 bars?"

```python
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['up']], steps_ahead = 3, timesteps = 20,
                model_relative_width = [2, 1], model_dropout = [0.2, 0.1],
                data_storage_path = './models/', model_name = 'up3')
lstm.network_structure_set_compile()
lstm.network_training(epochs = 200, batch_size = 64)
results_df, probs_df = lstm.network_predictions_evaluation(0.5)   # one report per step
```

### 5. Restore a model and evaluate it on new data

```python
lstm = fastLSTM()
lstm.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - direction.json',
              file_path_name = './models/')

lstm.X_data, lstm.Y_data = new_X_df, new_Y_df[['up']]
lstm.split_and_scale(scaler_fit = False)          # keep the scaler fitted at training time
lstm.network_predictions_evaluation(0.5)
```

### 6. Try a different architecture or sequence length

```python
lstm.model_relative_width = [3, 2, 1]
lstm.model_dropout = [0.2, 0.2, 0.1]
lstm.network_structure_set_compile(timesteps = 40)   # rebuild with the new length
lstm.network_training(epochs = 200)
```

### 7. Feature importance

```python
importances, names = lstm.gradient_feature_importance()
print(names[-5:])   # five most important features
```

### 8. Choose the epoch on the AUC of the hard cases, and weight them more

The loss and the accuracy are dominated by the easy rows; when what matters is how well the model ranks the hard cases, monitor the AUC on those rows and give them more weight:

```python
near = (df['STRIKE_DISTANCE_PCT'].abs() <= 0.02).values           # hard cases, one flag per row of X_df
weights = np.where(near, 3.0, 1.0)                                  # 3x weight in the training loss

model = fastLSTM(X_data = X_df, Y_data = Y_df[['itm']],
            timesteps = 8, sequence_groups = df['CONTRACT'].values,
            sample_weight = weights,
            monitor_auc_rows = near,                                # logs val_monitored_auc on these rows
            early_stop_monitor_metric = 'val_monitored_auc', early_stop_mode = 'max',
            checkpoint_monitor_metric = 'val_monitored_auc', checkpoint_mode = 'max',
            early_stop_patience = 10, data_storage_path = './models/')
```

AUC is a ranking measure and is not differentiable, so it is not used as the loss: the network is still trained with its loss (weighted), and the AUC chooses the epoch to keep.

### 9. Gated FAN projection + LSTM (hybrid)

```python
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['itm']], timesteps = 8,
                input_projection = 'gated_fan', input_projection_width = 16, input_projection_dropout = 0.5,
                model_relative_width = [32, 16, 2], model_dropout = [0.98, 0.98, 0.5],
                data_storage_path = './models/')
```

Every bar of the window goes through a gated FAN layer (periodic + non-periodic components of the features), then through the usual LSTM stack.

This is the hybrid gated FAN + LSTM network: there is no separate "fastHybrid" class. The hybrid is fastLSTM with `input_projection = 'gated_fan'`, so it inherits everything above (sequence groups, AUC epoch selection, sample weights, saved files, several GPUs or TPU). The FAN layer is the one of [fastGatedFourierAnalysisNetwork](https://github.com/fede72bari/fastGatedFourierAnalysisNetwork) and [fastANN](https://github.com/fede72bari/fastANN) (`hidden_layer_type = 'gated_fan'`): a model saved by one package loads in the others.

### 10. Two GPUs on Kaggle, or a TPU

```python
import tensorflow as tf
lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['itm']], timesteps = 8, batch_size = 4096,
                distribution_strategy = tf.distribute.MirroredStrategy(), data_storage_path = './models/')
```

On a TPU set the JAX backend and `keras.distribution.DataParallel` before creating the instance (see [Training on several devices](#training-on-several-devices-two-gpus-tpu)); nothing else changes.

---

## Backward compatibility

Parameter names were aligned with `fastANN`. The old names still work (with a `DeprecationWarning`), and models saved by older versions (old JSON keys, single scaler file) still load with `load_all()`.

| Old name | New name |
|---|---|
| `scaler` | `scaler_type` |
| `metric` | `metrics` |
| `check_point_metric` | `checkpoint_monitor_metric` |
| `early_stop_condittion` | `early_stop_monitor_metric` |
| `metric_mode` | `checkpoint_mode` and `early_stop_mode` |
| `scale_target` | `scale_targets` |
| `binary_network_predictions_evaluation()` | `network_predictions_evaluation()` (also returns `predictions_df`; the old name keeps the old return values) |
| `load_training_history(file_path_name=<csv path>)` | `load_training_history(training_history_file_name, file_path_name=<folder>)` |
| attribute `X_scaler` | `scaler` |

Note: models trained with old versions and `steps_ahead > 1` did not really learn future steps (every output learned the next value); retrain them.

### Differences from fastANN

Same names and workflow; LSTM-specific: `LSTM_type`, `timesteps`, `steps_ahead`, `class_weight`, `create_generators`, `prepare_input_sample`, `output_column_names`. fastANN-only: `split_type` (here the split is always sequential), `autoencoder_mode`, pre-split inputs and the `'PReLU'` activation.

---

## Tips and caveats

- `data_storage_path` must end with `/` (or `\\` on Windows) and the folder must exist.
- Monitor validation metrics (`val_…`) for both early stopping and checkpoint, and set the modes coherently (`'max'` for accuracy, `'min'` for losses).
- `timesteps` passed to `network_training` must equal the one used to build the network; to change it call `network_structure_set_compile(timesteps)` first.
- Each set must contain more than `timesteps + steps_ahead - 1` rows.
- `binary_precision_recall_vs_scoring` and the `report` returned by `network_predictions_evaluation` refer to the **last** output (last target column, last step).
- With `sequence_groups` (e.g. an option chain: one row per contract and time) each sample is a window of the same contract; `validation_targets()` returns the actual targets aligned with `model.predict(validation_generator)` in both modes.
- `shuffle = True` never mixes past and future: it only changes which windows share a batch. Keep it off if you need the exact behaviour of versions < 2.1.
- `class_weight` with more than one output is applied by Keras to the argmax of each target row: a warning is shown.
- Inside Jupyter the plots appear inline; in scripts call `matplotlib.pyplot.show()` after `plot_training_history()`.
- Very high dropout (≥ 0.9) with `mixed_float16`: `Dropout(rate)` scales the kept units by `1 / (1 - rate)` (200 at 0.995) and float16 activations can overflow (NaN loss from the first epoch). Use float32 there.
