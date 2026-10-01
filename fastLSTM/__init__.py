"""
fastLSTM
========

A thin, opinionated wrapper around a Keras stacked-LSTM network for
time-series classification and regression.

The package exposes a single class, :class:`fastLSTM`, whose public API
(hyperparameter names, method names and saved-file layout) mirrors the sister
package ``fastANN`` so that the two can be used interchangeably. Only the
parameters and methods that only make sense for recurrent networks
(``LSTM_type``, ``timesteps``, ``steps_ahead``, ``class_weight``,
``create_generators``, ``create_sequences``, ``prepare_input_sample``) are
LSTM-specific.

Typical workflow
----------------
>>> from fastLSTM import fastLSTM
>>> lstm = fastLSTM(X_data = X_df, Y_data = Y_df,
...                 model_relative_width = [2, 1],
...                 model_dropout = [0.2, 0.2],
...                 timesteps = 10,
...                 data_storage_path = './models/')
>>> lstm.network_structure_set_compile()
>>> lstm.network_training(epochs = 100, batch_size = 64)
>>> lstm.network_predictions_evaluation(min_probability = 0.5)
"""

# ---------------------------------------------------------------------------
#                              Libraries Import
# ---------------------------------------------------------------------------


# Multiprocessing
import multiprocessing

# Files Management
import gzip
import joblib
import glob
import csv
import json
import os

# Warnings (used to flag deprecated parameter names)
import warnings


# Stocks Indicators
# import talib

# Time Management
import datetime
from datetime import datetime, timedelta, date
import time
import pytz
from pytz import timezone


# Math and Sci
import numpy as np
import math
from scipy.signal import argrelextrema
import random
from scipy.signal import find_peaks
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import StandardScaler
# 'simps' was removed in SciPy 1.14: 'simpson' is the same function under its current name
from scipy.integrate import simpson as simps
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


# Reporting
import plotly
try:
    from plotly.figure_factory import create_candlestick
except ImportError:
    # removed from recent Plotly versions; not used by fastLSTM
    create_candlestick = None
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot
from matplotlib.pylab import rcParams
from xgboost import plot_tree
import seaborn as sns
from tabulate import tabulate
from IPython.display import HTML, display

# Data Management
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# Machine Learning
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, LambdaCallback
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy


# Optimization
from deap import base, creator, tools, algorithms

# Models Explainablity
#import shap
#import torch

# Binary Classification Specific Metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score

# General Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Data sources
import yfinance as yf

# Financial indicators
# import talib


# ---------------------------------------------------------------------------
#                    Backward compatibility (renamed parameters)
# ---------------------------------------------------------------------------

# Constructor parameters renamed to match fastANN: old name -> new name(s).
# Old names are still accepted (with a DeprecationWarning) so that existing
# scripts keep working, and they are also recognised when an old
# hyperparameters JSON file is loaded.
LEGACY_PARAMETER_NAMES = {
    'scaler': ('scaler_type',),
    'metric': ('metrics',),
    'check_point_metric': ('checkpoint_monitor_metric',),
    'early_stop_condittion': ('early_stop_monitor_metric',),
    'metric_mode': ('checkpoint_mode', 'early_stop_mode'),
    'scale_target': ('scale_targets',),
}


class fastLSTM:
    """
    Stacked-LSTM network for time-series classification and regression.

    The class takes care of the whole life cycle of the model: sequential
    train/test split, feature (and optionally target) scaling, network
    construction, training with early stopping and checkpointing, saving of
    every artefact needed to reproduce the run (model, scalers, training
    history, hyperparameters, data) and reloading them later.

    Network architecture
    --------------------
    ``model_relative_width`` lists the **hidden** LSTM layers only. The input
    layer and the output layer are always added automatically::

        Input(timesteps, n_features)
        LSTM(n_features * model_relative_width[0]) + Dropout(model_dropout[0])
        ...
        LSTM(n_features * model_relative_width[-1]) + Dropout(model_dropout[-1])
        Dense(output)

    so ``model_relative_width = [2, 1]`` builds two hidden LSTM layers with
    ``2 * n_features`` and ``1 * n_features`` units. Every LSTM layer except
    the last one returns the whole sequence; the last one returns only its
    final hidden state, which feeds the ``Dense`` output layer.

    How samples are built
    ---------------------
    Training and validation samples are created by Keras'
    ``TimeseriesGenerator`` with ``length = timesteps``: the target row ``t``
    is paired with the feature rows ``t - timesteps ... t - 1`` (row ``t``
    itself is NOT part of the window). If ``Y_data`` row ``t`` already holds
    the future value to predict for the bar ``t``, take this into account
    when building ``Y_data``. As a consequence the first ``timesteps`` rows of
    the test set have no prediction.

    Parameters
    ----------
    X_data : pandas.DataFrame, optional
        Features, one row per time step, in chronological order. Required to
        train a new model; can be omitted when the model is restored with
        :meth:`load_all`.
    Y_data : pandas.DataFrame, optional
        Targets, one row per time step, aligned with ``X_data``. For binary
        classification use one 0/1 column per target.
    scaler_type : {'StandardScaler', 'MinMaxScaler'}, default 'StandardScaler'
        Scaler applied to the features (and to the targets when
        ``scale_targets = True``). Any other value falls back to
        ``StandardScaler``.
    model_relative_width : list of float, default [1]
        Width of each hidden LSTM layer, relative to the number of features
        (units = ``int(n_features * width)``). Its length is the number of
        hidden layers.
    model_dropout : list of float, default [0]
        Dropout rate (0 - 1) applied after each hidden LSTM layer. Must have
        the same length as ``model_relative_width``.
    LSTM_type : {'classificator', 'regressor'}, default 'classificator'
        * ``'classificator'``: output layer with one unit per target column
          and ``last_layer_activation``.
        * ``'regressor'``: output layer with ``n_targets * steps_ahead`` units
          and linear activation (``last_layer_activation`` is ignored).
    learning_rate : float, default 0.0003
        Learning rate of the Adam optimizer.
    activation : str, default 'tanh'
        Activation of the LSTM layers (any Keras activation name, e.g.
        ``'tanh'``, ``'relu'``). ``'tanh'`` is the only value that allows
        Keras to use the fast cuDNN kernel on GPU.
    last_layer_activation : str, default 'sigmoid'
        Activation of the output layer for ``'classificator'`` networks:
        ``'sigmoid'`` for independent binary targets, ``'softmax'`` for
        one-hot multi-class targets.
    loss : str or keras.losses.Loss, default 'binary_crossentropy'
        Loss function. Any Keras loss name (``'binary_crossentropy'``,
        ``'categorical_crossentropy'``, ``'mse'``, ``'mae'``, ...) or a Keras
        loss object. The class names ``'BinaryCrossentropy'``,
        ``'CategoricalCrossentropy'`` and ``'SparseCategoricalCrossentropy'``
        are also accepted. See :meth:`set_loss_function`.
    metrics : list of str, default ['accuracy']
        Metrics computed by Keras during training (a single string is also
        accepted). The names appear in the training history, with a ``val_``
        prefix for the validation set.
    early_stop_monitor_metric : str, default 'val_accuracy'
        Quantity monitored by early stopping (e.g. ``'val_loss'``).
    checkpoint_monitor_metric : str, default 'val_accuracy'
        Quantity monitored to decide which epoch is the best one and must be
        saved (and reloaded at the end of the training).
    checkpoint_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``checkpoint_monitor_metric`` has to be maximised or minimised.
    early_stop_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``early_stop_monitor_metric`` has to be maximised or
        minimised.
    history_metrics : list of str, optional
        Columns of the training history plotted by
        :meth:`plot_training_history`. When ``None``:
        ``['accuracy', 'val_accuracy']`` for classificators and
        ``['loss', 'val_loss']`` for regressors.
    save_best_only : bool, default True
        If ``True`` the checkpoint overwrites the model file only when the
        monitored quantity improves; otherwise the model is saved at every
        epoch.
    early_stop_patience : int, default 200
        Number of epochs without improvement after which training stops.
    train_size_rate : float, default 0.7
        Fraction (0 - 1) of the rows used for training. The split is always
        sequential (the first rows are the training set) to avoid look-ahead.
    save_X_Y_data : bool, default True
        If ``True``, ``X_data`` and ``Y_data`` are saved as CSV files when the
        training starts, so that :meth:`load_all` can rebuild the same split.
    data_storage_path : str, default '\\\\cyPredict\\\\'
        Folder where every file is saved and loaded from. It is concatenated
        to the file names as is, so it must end with a path separator.
    model_name : str, default 'LSTM'
        Name used in every saved file name.
    scale_targets : bool, default False
        If ``True`` the targets are scaled too (useful for regressors);
        :meth:`model_predict` can then bring predictions back to the original
        scale.
    batch_size : int, default 128
        Batch size of the training and validation generators.
    timesteps : int, default 1
        Length of the input sequences (number of past rows seen by the
        network for each prediction).
    steps_ahead : int, default 1
        Multiplier of the output units of ``'regressor'`` networks. The
        targets are taken from ``Y_data`` as they are (no automatic
        shifting), so a value greater than 1 is consistent only if the loss
        and ``Y_data`` are built accordingly; keep 1 otherwise.
    class_weight : dict, optional
        Keras class weights, e.g. ``{0: 1.0, 1: 3.0}``, to rebalance
        unbalanced classes during training.
    **legacy_kwargs
        Old parameter names, still accepted for backward compatibility with
        a ``DeprecationWarning``: ``scaler`` (-> ``scaler_type``), ``metric``
        (-> ``metrics``), ``check_point_metric``
        (-> ``checkpoint_monitor_metric``), ``early_stop_condittion``
        (-> ``early_stop_monitor_metric``), ``metric_mode`` (-> both
        ``checkpoint_mode`` and ``early_stop_mode``), ``scale_target``
        (-> ``scale_targets``).

    Attributes
    ----------
    model : keras.Sequential
        The network (rebuilt by :meth:`network_structure_set_compile`,
        replaced by the best checkpoint after :meth:`network_training`).
    scaler : sklearn scaler
        Features scaler.
    Y_scaler : sklearn scaler or None
        Targets scaler (``None`` unless ``scale_targets = True``).
    X_train, Y_train, X_test, Y_test : pandas.DataFrame
        Unscaled sequential split of the data.
    X_train_s, X_test_s : numpy.ndarray
        Scaled features.
    Y_train_s, Y_test_s : numpy.ndarray
        Targets used for training (scaled only when ``scale_targets = True``).
    generator, validation_generator : TimeseriesGenerator
        Training and validation sample generators.
    loss_df : pandas.DataFrame
        Training history (one row per epoch).
    hyperparameters : dict
        Everything needed to rebuild the model; saved as JSON at training
        time.

    Examples
    --------
    Binary classification of the next bar direction:

    >>> lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['up']],
    ...                 model_relative_width = [2, 1],
    ...                 model_dropout = [0.2, 0.1],
    ...                 timesteps = 20,
    ...                 class_weight = {0: 1.0, 1: 2.0},
    ...                 data_storage_path = './models/',
    ...                 model_name = 'direction')
    >>> lstm.network_structure_set_compile()
    >>> lstm.network_training(epochs = 200, batch_size = 64)
    >>> results_df, predictions_df = lstm.network_predictions_evaluation(0.5)

    Regression with scaled targets:

    >>> lstm = fastLSTM(X_data = X_df, Y_data = Y_df[['return']],
    ...                 LSTM_type = 'regressor', loss = 'mse',
    ...                 metrics = ['mae'],
    ...                 early_stop_monitor_metric = 'val_loss',
    ...                 checkpoint_monitor_metric = 'val_loss',
    ...                 early_stop_mode = 'min', checkpoint_mode = 'min',
    ...                 scale_targets = True, timesteps = 30)

    Restoring a trained model:

    >>> lstm = fastLSTM()
    >>> lstm.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - direction.json',
    ...               file_path_name = './models/')
    """

    def __init__(self,
                 X_data = None,
                 Y_data = None,
                 scaler_type = 'StandardScaler', # 'StandardScaler' 'MinMaxScaler'
                 model_relative_width = [1],
                 model_dropout = [0],
                 LSTM_type = 'classificator', # 'classificator' 'regressor'
                 learning_rate = 0.0003,
                 activation = 'tanh', #'relu',
                 last_layer_activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'],
                 early_stop_monitor_metric = 'val_accuracy',
                 checkpoint_monitor_metric = 'val_accuracy',
                 checkpoint_mode = 'max',
                 early_stop_mode = 'max',
                 history_metrics = None,
                 save_best_only = True,
                 early_stop_patience = 200,
                 train_size_rate = 0.7,
                 save_X_Y_data = True,
                 data_storage_path = "\\cyPredict\\",
                 model_name = 'LSTM',
                 scale_targets = False,
                 batch_size = 128,
                 timesteps = 1,
                 steps_ahead = 1,
                 class_weight = None,
                 **legacy_kwargs):

        # old parameter names (fastLSTM < fastANN alignment) override the defaults
        legacy_values = self.translate_legacy_parameters(legacy_kwargs)
        scaler_type = legacy_values.get('scaler_type', scaler_type)
        metrics = legacy_values.get('metrics', metrics)
        checkpoint_monitor_metric = legacy_values.get('checkpoint_monitor_metric', checkpoint_monitor_metric)
        early_stop_monitor_metric = legacy_values.get('early_stop_monitor_metric', early_stop_monitor_metric)
        checkpoint_mode = legacy_values.get('checkpoint_mode', checkpoint_mode)
        early_stop_mode = legacy_values.get('early_stop_mode', early_stop_mode)
        scale_targets = legacy_values.get('scale_targets', scale_targets)

        self.model = Sequential()
        self.save_best_only = save_best_only

        self.data_storage_path = data_storage_path
        self.model_name = model_name

        self.model_relative_width = model_relative_width
        self.model_dropout = model_dropout
        self.LSTM_type = LSTM_type
        self.learning_rate = learning_rate
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        self.early_stop_monitor_metric = early_stop_monitor_metric
        self.checkpoint_monitor_metric = checkpoint_monitor_metric
        self.checkpoint_mode = checkpoint_mode
        self.early_stop_mode = early_stop_mode
        self.early_stop_patience = early_stop_patience
        self.train_size_rate = train_size_rate
        self.history_metrics = history_metrics
        self.timesteps = timesteps
        self.steps_ahead = steps_ahead
        self.batch_size = batch_size
        self.class_weight = class_weight

        self.hyperparameters_file_name = None

        self.model_training_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        self.save_X_Y_data = save_X_Y_data

        # names of the files of the last training (set by network_training / load_all)
        self.model_file_name = None
        self.scaler_file_name = None
        self.Y_scaler_file_name = None
        self.training_history_file_name = None
        self.X_data_df_file_name = None
        self.Y_data_df_file_name = None

        self.loss_df = pd.DataFrame()
        self.model_summary = {}

        self.X_data = X_data
        self.Y_data = Y_data

        self.X_train = pd.DataFrame()
        self.Y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.Y_test = pd.DataFrame()
        self.X_train_s = pd.DataFrame()
        self.Y_train_s = pd.DataFrame()
        self.X_test_s = pd.DataFrame()
        self.Y_test_s = pd.DataFrame()

        # created by create_generators (called by network_training)
        self.generator = None
        self.validation_generator = None

        # the targets scaler exists only when targets are scaled, as in fastANN
        self.scaler_type = scaler_type
        self.scale_targets = scale_targets
        self.scaler = self.new_scaler()
        self.Y_scaler = self.new_scaler() if scale_targets else None

        self.set_loss_function(loss)

        self.early_stop_patience_set(self.early_stop_patience)

        if((self.X_data is not None) and (self.Y_data is not None)):
            self.split_and_scale(scaler_fit = True)

        self.hyperparameters = {}

        self.init_hyperparameters()


    @staticmethod
    def translate_legacy_parameters(legacy_kwargs):
        """
        Map the old (pre fastANN alignment) parameter names to the new ones.

        Parameters
        ----------
        legacy_kwargs : dict
            Extra keyword arguments received by the constructor.

        Returns
        -------
        dict
            ``{new_name: value}`` for every recognised old name.

        Raises
        ------
        TypeError
            If a keyword is neither a current nor an old parameter name
            (same behaviour as a normal Python function).

        Examples
        --------
        >>> fastLSTM.translate_legacy_parameters({'metric_mode': 'min'})
        {'checkpoint_mode': 'min', 'early_stop_mode': 'min'}
        """
        translated = {}

        for old_name, value in legacy_kwargs.items():

            if old_name not in LEGACY_PARAMETER_NAMES:
                raise TypeError(f"fastLSTM.__init__() got an unexpected keyword argument '{old_name}'")

            new_names = LEGACY_PARAMETER_NAMES[old_name]
            warnings.warn(f"fastLSTM parameter '{old_name}' is deprecated, use '{' and '.join(new_names)}' instead.",
                          DeprecationWarning,
                          stacklevel = 3)

            for new_name in new_names:
                translated[new_name] = value

        return translated


    def new_scaler(self):
        """
        Create a new, unfitted scaler of the type selected by ``scaler_type``.

        Returns
        -------
        sklearn.preprocessing.MinMaxScaler or StandardScaler
            ``MinMaxScaler`` when ``scaler_type == 'MinMaxScaler'``,
            ``StandardScaler`` for any other value.
        """
        if(self.scaler_type == 'MinMaxScaler'):
            return MinMaxScaler()

        return StandardScaler()


    def init_hyperparameters(self,
                             model_training_datetime = None,
                             model_file_name = None,
                             scaler_file_name = None,
                             training_history_file_name = None,
                             X_data_df_file_name = None,
                             Y_data_df_file_name = None,
                             Y_scaler_file_name = None):
        """
        (Re)build the ``hyperparameters`` dictionary from the current attributes.

        The dictionary is what :meth:`save_hyperparameters` writes to JSON and
        what :meth:`set_hyperparameters` reads back, so it contains both the
        network/training settings and the names of the files produced by the
        training. Keys are the same as in fastANN, plus the LSTM-specific
        ones (``LSTM_type``, ``timesteps``, ``steps_ahead``, ``class_weight``)
        and ``Y_scaler_file_name``.

        Parameters
        ----------
        model_training_datetime : str, optional
            Timestamp (``'%Y-%m-%d %H-%M-%S'``) identifying the training run.
        model_file_name : str, optional
            Name of the ``.keras`` model file.
        scaler_file_name : str, optional
            Name of the ``.pkl`` features scaler file.
        training_history_file_name : str, optional
            Name of the ``.csv`` training history file.
        X_data_df_file_name, Y_data_df_file_name : str, optional
            Names of the ``.csv`` data files (``None`` if not saved).
        Y_scaler_file_name : str, optional
            Name of the ``.pkl`` targets scaler file (``None`` when targets are
            not scaled).

        Returns
        -------
        None
            The result is stored in ``self.hyperparameters``.
        """
        # feature names come from the data when available, otherwise from a loaded configuration
        if(self.X_data is not None):
            X_feature_names = self.X_data.columns.tolist()
        else:
            X_feature_names = getattr(self, 'X_feature_names', None)

        if(self.Y_data is not None):
            Y_feature_names = self.Y_data.columns.tolist()
        else:
            Y_feature_names = getattr(self, 'Y_feature_names', None)

        # JSON keys must be strings: class labels are converted back to int by set_hyperparameters
        if(self.class_weight is not None):
            class_weight = {str(label): float(weight) for label, weight in self.class_weight.items()}
        else:
            class_weight = None

        self.hyperparameters = {
                               'model_training_datetime': model_training_datetime,
                               'model_name': self.model_name,
                               'save_best_only': self.save_best_only,
                               'model_relative_width': self.model_relative_width,
                               'model_dropout': self.model_dropout,
                               'learning_rate': self.learning_rate,
                               'activation': self.activation,
                               'last_layer_activation': self.last_layer_activation,
                               'loss': self.loss if isinstance(self.loss, str) else getattr(self.loss, 'name', str(self.loss)),
                               'metrics': self.metrics,
                               'early_stop_monitor_metric': self.early_stop_monitor_metric,
                               'checkpoint_monitor_metric': self.checkpoint_monitor_metric,
                               'checkpoint_mode': self.checkpoint_mode,
                               'early_stop_mode': self.early_stop_mode,
                               'history_metrics': self.history_metrics,
                               'early_stop_patience': self.early_stop_patience,
                               'train_size_rate': self.train_size_rate,
                               'batch_size': self.batch_size,
                               'X_feature_names': X_feature_names,
                               'Y_feature_names': Y_feature_names,
                               'scaler_type': 'MinMaxScaler' if self.scaler_type == 'MinMaxScaler' else 'StandardScaler',
                               'data_storage_path': self.data_storage_path,
                               'model_file_name': model_file_name,
                               'scaler_file_name': scaler_file_name,
                               'Y_scaler_file_name': Y_scaler_file_name,
                               'training_history_file_name': training_history_file_name,
                               'X_data_df_file_name': X_data_df_file_name,
                               'Y_data_df_file_name': Y_data_df_file_name,
                               'scale_targets': self.scale_targets,
                               # LSTM specific
                               'LSTM_type': self.LSTM_type,
                               'timesteps': self.timesteps,
                               'steps_ahead': self.steps_ahead,
                               'class_weight': class_weight
                              }


    def get_hyperparameter(self, name, default = None):
        """
        Read a value from ``self.hyperparameters``, accepting old key names.

        Hyperparameters files written before the fastANN alignment use the
        old names (``metric``, ``check_point_metric``, ``metric_mode``,
        ``early_stop_condittion``, ``scale_target``); they are looked up when
        the new key is missing.

        Parameters
        ----------
        name : str
            Current key name.
        default : any, optional
            Value returned when neither the new nor an old key is present.

        Returns
        -------
        any
            The stored value (an old ``metric`` string is returned as a
            one-element list when ``name == 'metrics'``) or ``default``.
        """
        if(name in self.hyperparameters):
            return self.hyperparameters[name]

        for old_name, new_names in LEGACY_PARAMETER_NAMES.items():
            if((name in new_names) and (old_name in self.hyperparameters)):
                value = self.hyperparameters[old_name]
                if((name == 'metrics') and isinstance(value, str)):
                    value = [value]
                return value

        return default


    def set_hyperparameters(self):
        """
        Copy the values of ``self.hyperparameters`` back to the attributes.

        Called by :meth:`load_all` after :meth:`load_hyperparameters`. Keys
        missing from the file (older versions) leave the current attribute
        unchanged. The early stopping callback and the loss function are
        rebuilt so that they reflect the loaded settings.

        Returns
        -------
        None
        """
        self.model_training_datetime = self.get_hyperparameter('model_training_datetime', self.model_training_datetime)
        self.model_name = self.get_hyperparameter('model_name', self.model_name)
        self.save_best_only = self.get_hyperparameter('save_best_only', self.save_best_only)
        self.model_relative_width = self.get_hyperparameter('model_relative_width', self.model_relative_width)
        self.model_dropout = self.get_hyperparameter('model_dropout', self.model_dropout)

        self.learning_rate = self.get_hyperparameter('learning_rate', self.learning_rate)
        self.activation = self.get_hyperparameter('activation', self.activation)
        self.last_layer_activation = self.get_hyperparameter('last_layer_activation', self.last_layer_activation)
        self.set_loss_function(self.get_hyperparameter('loss', self.loss))

        self.metrics = self.get_hyperparameter('metrics', self.metrics)
        self.early_stop_monitor_metric = self.get_hyperparameter('early_stop_monitor_metric', self.early_stop_monitor_metric)
        self.checkpoint_monitor_metric = self.get_hyperparameter('checkpoint_monitor_metric', self.checkpoint_monitor_metric)
        self.checkpoint_mode = self.get_hyperparameter('checkpoint_mode', self.checkpoint_mode)
        self.early_stop_mode = self.get_hyperparameter('early_stop_mode', self.early_stop_mode)
        self.history_metrics = self.get_hyperparameter('history_metrics', self.history_metrics)

        self.early_stop_patience = self.get_hyperparameter('early_stop_patience', self.early_stop_patience)
        self.train_size_rate = self.get_hyperparameter('train_size_rate', self.train_size_rate)
        self.batch_size = self.get_hyperparameter('batch_size', self.batch_size)
        self.X_feature_names = self.get_hyperparameter('X_feature_names')
        self.Y_feature_names = self.get_hyperparameter('Y_feature_names')

        self.scaler_type = self.get_hyperparameter('scaler_type', self.scaler_type)
        if(self.get_hyperparameter('data_storage_path') is not None):
            self.data_storage_path = self.hyperparameters['data_storage_path']
        self.model_file_name = self.get_hyperparameter('model_file_name')
        self.scaler_file_name = self.get_hyperparameter('scaler_file_name')
        self.Y_scaler_file_name = self.get_hyperparameter('Y_scaler_file_name')

        self.training_history_file_name = self.get_hyperparameter('training_history_file_name')
        self.X_data_df_file_name = self.get_hyperparameter('X_data_df_file_name')
        self.Y_data_df_file_name = self.get_hyperparameter('Y_data_df_file_name')

        self.scale_targets = self.get_hyperparameter('scale_targets', False)
        if(self.scale_targets and (self.Y_scaler is None)):
            self.Y_scaler = self.new_scaler()

        # LSTM specific
        self.LSTM_type = self.get_hyperparameter('LSTM_type', self.LSTM_type)
        self.timesteps = self.get_hyperparameter('timesteps', self.timesteps)
        self.steps_ahead = self.get_hyperparameter('steps_ahead', self.steps_ahead)

        class_weight = self.get_hyperparameter('class_weight')
        if(class_weight is not None):
            # JSON turned the integer class labels into strings
            class_weight = {int(label) if str(label).lstrip('-').isdigit() else label: weight
                            for label, weight in class_weight.items()}
        self.class_weight = class_weight

        # callbacks depend on the loaded settings
        self.early_stop_patience_set(self.early_stop_patience)


    def set_loss_function(self, loss):
        """
        Set the loss used to compile the network.

        ``self.loss`` keeps the value given by the user (saved in the
        hyperparameters file), while ``self.loss_function`` holds what is
        actually passed to ``model.compile``.

        Parameters
        ----------
        loss : str or keras.losses.Loss
            * ``'BinaryCrossentropy'``, ``'CategoricalCrossentropy'``,
              ``'SparseCategoricalCrossentropy'``: the corresponding Keras
              loss object is created.
            * any other string (``'binary_crossentropy'``, ``'mse'``,
              ``'mae'``, ``'huber'``, ...): passed to Keras unchanged, so it
              must be a valid Keras loss name.
            * a Keras loss object or a callable: used as is.

        Returns
        -------
        None

        Examples
        --------
        >>> lstm.set_loss_function('mse')
        >>> lstm.network_structure_set_compile()   # recompile to apply it
        """
        self.loss = loss

        # the check is a substring search so that also the "<...BinaryCrossentropy object at ...>"
        # strings saved by older versions are recognised; Sparse... must be tested before Categorical...
        if(isinstance(loss, str) and ('SparseCategoricalCrossentropy' in loss)):
            self.loss_function = SparseCategoricalCrossentropy()
        elif(isinstance(loss, str) and ('CategoricalCrossentropy' in loss)):
            self.loss_function = CategoricalCrossentropy()
        elif(isinstance(loss, str) and ('BinaryCrossentropy' in loss)):
            self.loss_function = BinaryCrossentropy()
        else:
            self.loss_function = loss


    def early_stop_patience_set(self, patience = None):
        """
        Create the early stopping callback (``self.early_stop``).

        It monitors ``early_stop_monitor_metric`` in ``early_stop_mode``.

        Parameters
        ----------
        patience : int, optional
            Number of epochs without improvement before stopping. When given
            it also updates ``self.early_stop_patience``; when ``None`` the
            current value is used.

        Returns
        -------
        None

        Examples
        --------
        >>> lstm.early_stop_patience_set(50)
        """
        if(patience is not None):
            self.early_stop_patience = patience

        self.early_stop = EarlyStopping(monitor = self.early_stop_monitor_metric,
                                        mode = self.early_stop_mode,
                                        verbose = 1,
                                        patience = self.early_stop_patience)


    def checkpoint_callback(self, save_best_only = None):
        """
        Create the model checkpoint callback (``self.model_checkpoint``).

        The model is saved to ``data_storage_path + model_file_name`` (the
        name stored in the hyperparameters by :meth:`network_training`, or a
        timestamped default) monitoring ``checkpoint_monitor_metric`` in
        ``checkpoint_mode``.

        Parameters
        ----------
        save_best_only : bool, optional
            If ``True`` the file is overwritten only when the monitored
            quantity improves. Defaults to ``self.save_best_only``.

        Returns
        -------
        keras.callbacks.ModelCheckpoint
            The callback (also stored in ``self.model_checkpoint``).
        """
        if(save_best_only is None):
            save_best_only = self.save_best_only

        if(self.hyperparameters.get('model_file_name')):
            model_file_name = self.hyperparameters['model_file_name']
        else:
            model_file_name = self.model_training_datetime + ' - LSTM MODEL - ' + self.model_name + '.keras'

        # the callback is stored in its own attribute: assigning it to self.checkpoint_callback
        # would shadow this method and make a second training fail
        self.model_checkpoint = ModelCheckpoint(self.data_storage_path + model_file_name,
                                                monitor = self.checkpoint_monitor_metric,
                                                mode = self.checkpoint_mode,
                                                verbose = 1,
                                                save_best_only = save_best_only)

        return self.model_checkpoint


    def network_structure_set_compile(self, timesteps = None):
        """
        Build and compile the network.

        One ``LSTM`` + ``Dropout`` block is added for each element of
        ``model_relative_width``; the ``Input`` layer and the ``Dense``
        output layer are added automatically (see the class docstring). The
        network is compiled with Adam(``learning_rate``), the loss set by
        :meth:`set_loss_function` and ``metrics``.

        Parameters
        ----------
        timesteps : int, optional
            New length of the input sequences. When given it replaces
            ``self.timesteps`` (also used by the generators).

        Returns
        -------
        None
            The network is stored in ``self.model`` and its text summary in
            ``self.model_summary``.

        Raises
        ------
        ValueError
            If there is no training data, if ``model_dropout`` and
            ``model_relative_width`` have different lengths or if
            ``LSTM_type`` is not ``'classificator'`` or ``'regressor'``.

        Examples
        --------
        >>> lstm.model_relative_width = [3, 2, 1]   # three hidden LSTM layers
        >>> lstm.model_dropout = [0.2, 0.2, 0.1]
        >>> lstm.network_structure_set_compile(timesteps = 15)
        """
        if(timesteps is not None):
            self.timesteps = timesteps

        if(len(self.X_train_s) == 0):
            raise ValueError('No training data: pass X_data and Y_data to the constructor or call split_and_scale first.')

        if(len(self.model_dropout) != len(self.model_relative_width)):
            raise ValueError(f'model_dropout ({len(self.model_dropout)} values) and model_relative_width '
                             f'({len(self.model_relative_width)} values) must have the same length.')

        if(self.LSTM_type not in ('classificator', 'regressor')):
            raise ValueError(f"LSTM_type must be 'classificator' or 'regressor', not '{self.LSTM_type}'.")

        n_features = self.X_train_s.shape[1]
        n_hidden_layers = len(self.model_relative_width)

        # reset
        self.model = Sequential()

        # input layer: sequences of `timesteps` rows with `n_features` columns
        self.model.add(Input(shape = (self.timesteps, n_features)))

        # hidden layers
        for i in range(n_hidden_layers):

            model_relative_width = self.model_relative_width[i]
            model_dropout = self.model_dropout[i]

            # every LSTM but the last must return the whole sequence because the next LSTM needs a 3D input;
            # the last one returns only its final state (2D), which is what the Dense output layer expects.
            # With a single hidden layer, the first layer is also the last one.
            return_sequences = (i < n_hidden_layers - 1)

            print(f'Hidden LSTM layer {i + 1}/{n_hidden_layers}: relative width {model_relative_width}, '
                  f'dropout {model_dropout}, return_sequences {return_sequences}')

            self.model.add(LSTM(units = int(n_features * model_relative_width),
                                return_sequences = return_sequences,
                                activation = self.activation))

            self.model.add(Dropout(model_dropout))

        # output layer
        if(self.LSTM_type == 'classificator'):
            print(f'Output layer for classification: {self.Y_train.shape[1]} neurons and activation = {self.last_layer_activation}')
            self.model.add(Dense(self.Y_train.shape[1],
                                 activation = self.last_layer_activation))

        elif(self.LSTM_type == 'regressor'):
            # linear output: last_layer_activation is not used by regressors
            print(f'Output layer for regression: {self.Y_train.shape[1] * self.steps_ahead} neurons and linear activation')
            self.model.add(Dense(self.Y_train.shape[1] * self.steps_ahead, activation = 'linear'))

        # compile
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss = self.loss_function,
                           metrics = self.metrics)

        # report: model.summary() only prints, so its lines are collected to keep a copy in model_summary
        summary_lines = []
        self.model.summary(print_fn = lambda line, *args, **kwargs: summary_lines.append(line))
        self.model_summary = '\n'.join(summary_lines)

        print(self.model_summary)


    def create_generators(self, batch_size = None):
        """
        Create the training and validation sample generators.

        ``TimeseriesGenerator`` pairs target row ``t`` with the feature rows
        ``t - timesteps ... t - 1``; so each generator yields
        ``len(data) - timesteps`` samples of shape ``(timesteps, n_features)``.

        Parameters
        ----------
        batch_size : int, optional
            When given it replaces ``self.batch_size``.

        Returns
        -------
        None
            The generators are stored in ``self.generator`` (training set)
            and ``self.validation_generator`` (test set).

        Examples
        --------
        >>> lstm.create_generators(batch_size = 32)
        >>> X_batch, Y_batch = lstm.generator[0]
        >>> X_batch.shape   # (32, timesteps, n_features)
        """
        if(batch_size is not None):
            self.batch_size = batch_size

        self.generator = TimeseriesGenerator(self.X_train_s, self.Y_train_s, length = self.timesteps, batch_size = self.batch_size)
        self.validation_generator = TimeseriesGenerator(self.X_test_s, self.Y_test_s, length = self.timesteps, batch_size = self.batch_size)


    def network_training(self, epochs, batch_size = None, timesteps = None):
        """
        Train the network and save every artefact of the run.

        Steps:

        1. a timestamp identifies the run and all the file names;
        2. ``X_data`` / ``Y_data`` are saved as CSV (if ``save_X_Y_data``);
        3. hyperparameters (JSON) and scalers (``.pkl``) are saved;
        4. the network is trained on the generators with early stopping and
           checkpointing (``class_weight`` is applied if set);
        5. the training history is saved (CSV), the best checkpoint is
           reloaded into ``self.model`` and the history is plotted.

        Files written in ``data_storage_path`` (``<dt>`` = timestamp,
        ``<name>`` = ``model_name``)::

            <dt> - LSTM MODEL - <name>.keras
            <dt> - SCALER FOR LSTM MODEL - <name>.pkl
            <dt> - Y SCALER FOR LSTM MODEL - <name>.pkl        (scale_targets only)
            <dt> - TRAINING HISTORY OF LSTM MODEL - <name>.csv
            <dt> - HYPERPARAMETERS OF LSTM MODEL - <name>.json
            <dt> - X_data FOR LSTM MODEL - <name>.csv          (save_X_Y_data only)
            <dt> - Y_data FOR LSTM MODEL - <name>.csv          (save_X_Y_data only)

        Parameters
        ----------
        epochs : int
            Maximum number of epochs (early stopping may end the training
            before).
        batch_size : int, optional
            When given it replaces ``self.batch_size``.
        timesteps : int, optional
            When given it replaces ``self.timesteps``. The network must have
            been built with the same value: call
            ``network_structure_set_compile(timesteps)`` before changing it.

        Returns
        -------
        None
            The trained (best) model is in ``self.model``, the history in
            ``self.loss_df``.

        Raises
        ------
        ValueError
            If ``timesteps`` differs from the input length of the compiled
            network.

        Examples
        --------
        >>> lstm.network_structure_set_compile()
        >>> lstm.network_training(epochs = 300, batch_size = 64)
        """
        if(timesteps is not None):
            self.timesteps = timesteps

        if(batch_size is not None):
            print(f'Batch size is not none, equal to {batch_size}')
            self.batch_size = batch_size
        else:
            print(f'Batch size is none, keep default or previous value {self.batch_size}')

        # the generators produce sequences of self.timesteps rows: they must match the network input
        if(self.model.inputs and (self.model.input_shape[1] != self.timesteps)):
            raise ValueError(f'The network expects sequences of {self.model.input_shape[1]} timesteps but timesteps = {self.timesteps}: '
                             f'call network_structure_set_compile({self.timesteps}) before training.')

        # timestamp identifying the training run
        self.model_training_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        # file names (model, scalers, training history, hyperparameters)
        model_file_name = self.model_training_datetime + ' - LSTM MODEL - ' + self.model_name + '.keras'
        scaler_file_name = self.model_training_datetime + ' - SCALER FOR LSTM MODEL - ' + self.model_name + '.pkl'
        Y_scaler_file_name = self.model_training_datetime + ' - Y SCALER FOR LSTM MODEL - ' + self.model_name + '.pkl' if self.scale_targets else None
        training_history_file_name = self.model_training_datetime + ' - TRAINING HISTORY OF LSTM MODEL - ' + self.model_name + '.csv'
        hyperparameters_file_name = self.model_training_datetime + ' - HYPERPARAMETERS OF LSTM MODEL - ' + self.model_name + '.json'
        self.hyperparameters_file_name = hyperparameters_file_name

        # X and Y data are saved so that load_all can rebuild the same train/test split
        if(self.save_X_Y_data and (self.X_data is not None) and (self.Y_data is not None)):
            X_data_df_file_name = self.model_training_datetime + ' - X_data FOR LSTM MODEL - ' + self.model_name + '.csv'
            Y_data_df_file_name = self.model_training_datetime + ' - Y_data FOR LSTM MODEL - ' + self.model_name + '.csv'

            # index=False as in fastANN: the (datetime) index is not saved
            self.X_data.to_csv(self.data_storage_path + X_data_df_file_name, index = False)
            self.Y_data.to_csv(self.data_storage_path + Y_data_df_file_name, index = False)

            print('\nsave_X_Y_data saved.')

        else:
            X_data_df_file_name = None
            Y_data_df_file_name = None

            print('\nsave_X_Y_data not saved.')

        self.model_file_name = model_file_name
        self.scaler_file_name = scaler_file_name
        self.Y_scaler_file_name = Y_scaler_file_name
        self.training_history_file_name = training_history_file_name
        self.X_data_df_file_name = X_data_df_file_name
        self.Y_data_df_file_name = Y_data_df_file_name

        # save hyperparameters
        print('\nInit hyperparameters.')
        self.init_hyperparameters(
                                  model_training_datetime = self.model_training_datetime,
                                  model_file_name = model_file_name,
                                  scaler_file_name = scaler_file_name,
                                  training_history_file_name = training_history_file_name,
                                  X_data_df_file_name = X_data_df_file_name,
                                  Y_data_df_file_name = Y_data_df_file_name,
                                  Y_scaler_file_name = Y_scaler_file_name
                                 )

        print('\nSave hyperparameters.')
        self.save_hyperparameters(hyperparameters_file_name)

        # save used scalers (one file each, as in fastANN)
        joblib.dump(self.scaler, self.data_storage_path + scaler_file_name)

        if(self.scale_targets == True):
            joblib.dump(self.Y_scaler, self.data_storage_path + Y_scaler_file_name)

        self.checkpoint_callback(self.save_best_only)

        # training and validation samples
        self.create_generators()

        # model training (class_weight = None means no reweighting)
        if(self.class_weight is not None):
            print(f'Used class_weight: {self.class_weight}')

        history = self.model.fit(self.generator,
                                 epochs = epochs,
                                 validation_data = self.validation_generator,
                                 class_weight = self.class_weight,
                                 callbacks = [self.early_stop, self.model_checkpoint])

        # save history
        self.loss_df = pd.DataFrame(history.history)
        self.loss_df.to_csv(self.data_storage_path + training_history_file_name)

        # keep the best model saved by the checkpoint
        self.load_model(model_file_name)

        # plot history
        self.plot_training_history()


    def save_hyperparameters(self, file_name):
        """
        Save ``self.hyperparameters`` as JSON in ``data_storage_path``.

        Parameters
        ----------
        file_name : str
            Name of the JSON file (without the folder).

        Returns
        -------
        None
        """
        with open(self.data_storage_path + file_name, "w") as file:
            json.dump(self.hyperparameters, file)

        print("Hyperparameters saved in " + self.data_storage_path + file_name)


    def load_hyperparameters(self, file_name, file_path_name = None):
        """
        Load a hyperparameters JSON file into ``self.hyperparameters``.

        The attributes are NOT updated: call :meth:`set_hyperparameters`
        afterwards (or use :meth:`load_all`).

        Parameters
        ----------
        file_name : str
            Name of the JSON file.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        self.hyperparameters_file_name = file_name

        print(f'\nTrying to load hyperparameters {self.data_storage_path + file_name}')
        with open(self.data_storage_path + file_name, "r") as file:
            self.hyperparameters = json.load(file)
        print(f'Hyperparameters loaded.')


    def load_model(self, model_file_name = None, file_path_name = None):
        """
        Load a saved Keras model into ``self.model``.

        Parameters
        ----------
        model_file_name : str, optional
            Name of the ``.keras`` file. Defaults to the one stored in the
            hyperparameters.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        keras.Model
            The loaded model (also stored in ``self.model``).
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(model_file_name is None):
            model_file_name = self.hyperparameters['model_file_name']

        model_file_path = self.data_storage_path + model_file_name
        if os.path.exists(model_file_path):
            print("Model file exists.")
            file_size = os.path.getsize(model_file_path)
            print(f"File size: {file_size} bytes")
            if file_size == 0:
                print("Warning: The model file is empty.")
        else:
            print("Error: Model file does not exist.")

        print(f'\nTrying to load model {model_file_path}')
        self.model = load_model(model_file_path)
        print(f'Model loaded.')

        if self.model:
            print("Model is correctly loaded and accessible.")
            self.model.summary()
        else:
            print("Model is None after loading. Check the loading logic.")

        return self.model


    def load_scaler(self, scaler_file_name = None, Y_scaler_file_name = None, file_path_name = None):
        """
        Load the features scaler and, if targets are scaled, the targets scaler.

        Files saved by older fastLSTM versions, containing a dictionary
        ``{'X_scaler': ..., 'Y_scaler': ...}`` in a single ``.pkl``, are
        recognised and unpacked.

        Parameters
        ----------
        scaler_file_name : str, optional
            Features scaler file. Defaults to the name stored in the
            hyperparameters, or to the standard timestamped name.
        Y_scaler_file_name : str, optional
            Targets scaler file (used only when ``scale_targets = True``).
            Same defaults as ``scaler_file_name``.
        file_path_name : str, optional
            Folder of the files; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
            Scalers are stored in ``self.scaler`` and ``self.Y_scaler``.
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(scaler_file_name is None):
            scaler_file_name = (self.hyperparameters.get('scaler_file_name')
                                or self.model_training_datetime + ' - SCALER FOR LSTM MODEL - ' + self.model_name + '.pkl')

        if(Y_scaler_file_name is None):
            Y_scaler_file_name = (self.hyperparameters.get('Y_scaler_file_name')
                                  or self.model_training_datetime + ' - Y SCALER FOR LSTM MODEL - ' + self.model_name + '.pkl')

        print(f"\nTrying to load scaler {self.data_storage_path}{scaler_file_name}")
        scaler = joblib.load(self.data_storage_path + scaler_file_name)

        # older versions saved both scalers in a single dictionary
        if(isinstance(scaler, dict)):
            self.scaler = scaler['X_scaler']
            self.Y_scaler = scaler['Y_scaler']
            print(f'Scaler and Y_scaler loaded (single file format).')
            return

        self.scaler = scaler
        print(f'Scaler loaded.')

        if(self.scale_targets == True):
            print(f"\nTrying to load Y_scaler {self.data_storage_path}{Y_scaler_file_name}")
            self.Y_scaler = joblib.load(self.data_storage_path + Y_scaler_file_name)
            print(f'Y_scaler loaded.')


    def load_training_history(self, training_history_file_name = None, file_path_name = None):
        """
        Load a training history CSV into ``self.loss_df``.

        Parameters
        ----------
        training_history_file_name : str, optional
            Name of the CSV file. Defaults to the one stored in the
            hyperparameters.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(training_history_file_name is None):
            training_history_file_name = self.hyperparameters['training_history_file_name']

        print(f"\nTrying to load training history {self.data_storage_path + training_history_file_name}")
        # the first column is the epoch index written by DataFrame.to_csv
        self.loss_df = pd.read_csv(self.data_storage_path + training_history_file_name, index_col = 0)
        print(f'Training history loaded.')


    def load_all(self, hyperparameters_file_name = None, file_path_name = None):
        """
        Restore a trained model with everything that was saved with it.

        Loads, in order: hyperparameters (and applies them), model, scalers,
        training history and, if they were saved, ``X_data`` / ``Y_data``,
        which are split and scaled again with the loaded scaler (no refit)
        and turned into generators, so that evaluation methods work
        immediately.

        Parameters
        ----------
        hyperparameters_file_name : str, optional
            Name of the hyperparameters JSON file. Defaults to
            ``self.hyperparameters_file_name`` (the last training).
        file_path_name : str, optional
            Folder of the files; when given it replaces ``data_storage_path``.

        Returns
        -------
        None

        Examples
        --------
        >>> lstm = fastLSTM()
        >>> lstm.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF LSTM MODEL - LSTM.json',
        ...               file_path_name = './models/')
        >>> lstm.network_predictions_evaluation(0.5)
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(hyperparameters_file_name is not None):
            self.hyperparameters_file_name = hyperparameters_file_name

        print(f'\nTrying to open hyperparameters file {self.hyperparameters_file_name}')
        self.load_hyperparameters(self.hyperparameters_file_name)

        print(f'\nSetting hyperparameters')
        self.set_hyperparameters()

        print(f'Hyperparameters:\n')
        display(self.hyperparameters)

        self.load_model(self.hyperparameters['model_file_name'])

        print(f"\nTrying to import LSTM scalers")
        self.load_scaler()

        print(f"\nTrying to import LSTM training history {self.hyperparameters['training_history_file_name']}")
        self.load_training_history(self.hyperparameters['training_history_file_name'])

        # load X and Y data, split and scale
        if((self.hyperparameters.get('X_data_df_file_name') is not None) and (self.hyperparameters.get('Y_data_df_file_name') is not None)):

            self.X_data = pd.read_csv(self.data_storage_path + self.hyperparameters['X_data_df_file_name'])
            self.Y_data = pd.read_csv(self.data_storage_path + self.hyperparameters['Y_data_df_file_name'])

            self.split_and_scale(scaler_fit = False) # scaler is loaded, no need to fit it again
            self.create_generators()

        print("Load all completed.")


    def network_predictions_evaluation(self, min_probability, output_dict = False):
        """
        Evaluate a binary classificator on the test set.

        Probabilities predicted on the validation generator are turned into
        0/1 with the threshold ``min_probability`` and compared with the
        actual targets through sklearn's ``classification_report`` (printed
        for every target column). The first ``timesteps`` test rows have no
        prediction and are excluded.

        Parameters
        ----------
        min_probability : float
            Threshold (0 - 1): predictions strictly greater than it become 1.
        output_dict : bool, default False
            If ``True`` the report is also returned as a dictionary.

        Returns
        -------
        filtered_predictions_results_df : pandas.DataFrame
            0/1 predictions, one column per target.
        predictions_df : pandas.DataFrame
            Raw predicted probabilities (columns numbered from 0).
        report : dict
            Only when ``output_dict = True``: classification report of the
            LAST target column.

        Examples
        --------
        >>> results_df, probabilities_df = lstm.network_predictions_evaluation(0.6)
        >>> results_df, probabilities_df, report = lstm.network_predictions_evaluation(0.6, output_dict = True)
        >>> report['1']['precision']
        """
        if(self.validation_generator is None):
            self.create_generators()

        # Cut off predictions with low probability
        predictions = self.model.predict(self.validation_generator)
        predictions_df = pd.DataFrame(predictions.reshape(predictions.shape[0], -1))
        filtered_predictions_results_df = pd.DataFrame()

        # the generator has no sample for the first `timesteps` test rows (see create_generators)
        Y_test = self.Y_test.iloc[self.timesteps:]

        print(f'len predictions {len(predictions_df)}')
        print(f'len Y_test (without the first {self.timesteps} rows) {len(Y_test)}')

        report = None
        for count, col_name in enumerate(Y_test.columns):

            filtered_predictions_results_df[col_name] = predictions_df[count].apply(lambda x: 1 if x > min_probability else 0 ).values
            report = classification_report(Y_test[col_name], filtered_predictions_results_df[col_name], output_dict = output_dict)

            print(report)

        if(output_dict == False):
            return filtered_predictions_results_df, predictions_df

        elif(output_dict == True):
            return filtered_predictions_results_df, predictions_df, report


    def binary_network_predictions_evaluation(self, min_probability, output_dict = False):
        """
        Deprecated alias of :meth:`network_predictions_evaluation`.

        Kept for backward compatibility; it returns what
        :meth:`network_predictions_evaluation` returns.
        """
        warnings.warn("binary_network_predictions_evaluation is deprecated, use network_predictions_evaluation instead.",
                      DeprecationWarning,
                      stacklevel = 2)

        return self.network_predictions_evaluation(min_probability, output_dict = output_dict)


    def plot_training_history(self):
        """
        Plot the training history (``self.loss_df``) with pandas/matplotlib.

        The plotted columns are ``history_metrics``; when it is ``None``,
        ``['accuracy', 'val_accuracy']`` for classificators and
        ``['loss', 'val_loss']`` for regressors.

        Returns
        -------
        None
        """
        history_metrics = self.history_metrics

        if(history_metrics is None):
            if(self.LSTM_type == 'regressor'):
                history_metrics = ['loss', 'val_loss']
            else:
                history_metrics = ['accuracy', 'val_accuracy']

        self.loss_df[history_metrics].plot()


    def create_sequences(self, data, window_size):
        """
        Split a time series into overlapping windows (sliding window, step 1).

        Utility not used internally (training uses ``TimeseriesGenerator``).

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Series or numpy.ndarray
            Time series, one row per time step.
        window_size : int
            Number of rows of each window.

        Returns
        -------
        list
            ``len(data) - window_size + 1`` windows, each one a (nested) list
            of ``window_size`` rows; window ``i`` contains rows
            ``i ... i + window_size - 1``.

        Examples
        --------
        >>> lstm.create_sequences(pd.Series([1, 2, 3, 4]), 2)
        [[1, 2], [2, 3], [3, 4]]
        """
        sequences = []

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        # one window for each starting row that still has window_size rows after it
        for i in range(len(data) - window_size + 1):
            sequence = data[i:i + window_size].tolist()
            sequences.append(sequence)

        return sequences


    def split_and_scale(self, scaler_fit = False):
        """
        Split the data sequentially into training and test set and scale them.

        The first ``train_size_rate`` fraction of the rows is the training
        set, the rest is the test set (no shuffling: with time series a
        random split would leak future information). Features are scaled with
        ``self.scaler``; targets with ``self.Y_scaler`` only when
        ``scale_targets = True``.

        Parameters
        ----------
        scaler_fit : bool, default False
            If ``True`` the scalers are fitted on the training set (new
            data); if ``False`` the already fitted (e.g. loaded) scalers are
            only applied.

        Returns
        -------
        None
            Results are stored in ``X_train``, ``Y_train``, ``X_test``,
            ``Y_test`` (DataFrames) and ``X_train_s``, ``X_test_s``,
            ``Y_train_s``, ``Y_test_s`` (numpy arrays).

        Examples
        --------
        >>> lstm.X_data, lstm.Y_data = new_X_df, new_Y_df
        >>> lstm.split_and_scale(scaler_fit = True)
        """
        train_size = int(len(self.X_data) * self.train_size_rate)
        test_size = len(self.X_data) - train_size

        self.X_train = self.X_data.head(train_size)
        self.Y_train = self.Y_data.head(train_size)
        self.X_test = self.X_data.tail(test_size)
        self.Y_test = self.Y_data.tail(test_size)

        # scale
        if(scaler_fit == True):
            print('\tFit transform X_train')
            self.X_train_s = self.scaler.fit_transform(self.X_train)
        else:
            print('\tOnly transform X_train')
            self.X_train_s = self.scaler.transform(self.X_train)

        print('\tTransform X_test')
        self.X_test_s = self.scaler.transform(self.X_test)

        if(self.scale_targets == True):
            print('Target scaled too')

            if(self.Y_scaler is None):
                self.Y_scaler = self.new_scaler()

            if(scaler_fit == True):
                self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)
            else:
                self.Y_train_s = self.Y_scaler.transform(self.Y_train)

            self.Y_test_s = self.Y_scaler.transform(self.Y_test)

        else:
            print('Target not scaled')
            # numpy arrays (not DataFrames): TimeseriesGenerator indexes the targets by row position
            self.Y_train_s = self.Y_train.values
            self.Y_test_s = self.Y_test.values

        print('split_and_scale, end data length')
        print(f"\tShape of X_train: {self.X_train.shape}")
        print(f"\tShape of Y_train: {self.Y_train.shape}")
        print(f"\tShape of X_test: {self.X_test.shape}")
        print(f"\tShape of Y_test: {self.Y_test.shape}")


    def binary_precision_recall_vs_scoring(self, n_points = 15, plot = True):
        """
        Precision and recall of class ``1`` as a function of the probability cutoff.

        :meth:`network_predictions_evaluation` is run for every cutoff from
        ``n_points / 100`` to ``0.99`` with step ``0.01``.

        Parameters
        ----------
        n_points : int, default 15
            Despite the name, it is the FIRST cutoff expressed in percent
            (15 -> cutoffs 0.15, 0.16, ..., 0.99), the same as in fastANN.
        plot : bool, default True
            If ``True`` an interactive Plotly chart is shown.

        Returns
        -------
        pandas.DataFrame
            Columns ``'Cutoff'``, ``'Precision'``, ``'Recall'``.

        Notes
        -----
        The values refer to the last target column, and the class labels of
        ``Y_data`` must be the integers 0/1 (the report keys are ``'0'`` and
        ``'1'``; float labels would produce ``'1.0'``).

        Examples
        --------
        >>> pr_df = lstm.binary_precision_recall_vs_scoring(n_points = 30, plot = False)
        >>> pr_df.loc[pr_df['Precision'].idxmax()]
        """
        # lists collecting precision and recall for every cutoff
        precision_list = []
        recall_list = []
        cutoff_values = []

        # evaluate every cutoff from n_points% to 99%
        for cutoff in range(n_points, 100, 1):
            cutoff_value = cutoff / 100  # cutoff as a probability (0 - 1)
            print(f'Evaluating cutoff value = {cutoff_value}')

            filtered_predictions_results_df, predictions_df, dictionary = self.network_predictions_evaluation(cutoff_value, output_dict=True)

            # precision and recall of the positive class
            precision_list.append(dictionary['1']['precision'])
            recall_list.append(dictionary['1']['recall'])
            cutoff_values.append(cutoff_value)

        df = pd.DataFrame({'Cutoff': cutoff_values, 'Precision': precision_list, 'Recall': recall_list})

        if(plot == True):

            # precision and recall vs cutoff (interactive Plotly chart)
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Precision'], mode='lines', name='Precision'))

            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Recall'], mode='lines', name='Recall'))

            fig.update_layout(
                xaxis_title='Cutoff',
                yaxis_title='Value',
                title='Precision and Recall vs Cutoff'
            )

            fig.show()

        return df


    def prepare_input_sample(self, X, current_datetime_idx, apply_scaler=True):
        """
        Extract the input sequence ending at a given row, ready for ``model.predict``.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Unscaled features with the same columns used for training.
        current_datetime_idx : int
            Position (not label) of the LAST row of the sequence.
        apply_scaler : bool, default True
            If ``True`` the sequence is scaled with ``self.scaler``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(1, timesteps, n_features)`` containing rows
            ``current_datetime_idx - timesteps + 1 ... current_datetime_idx``.

        Raises
        ------
        ValueError
            If ``current_datetime_idx < timesteps - 1`` (not enough rows).

        Notes
        -----
        During training the window paired with target row ``t`` ends at row
        ``t - 1`` (see :meth:`create_generators`); so the prediction made on
        the window ending at ``current_datetime_idx`` corresponds to the
        target row ``current_datetime_idx + 1``.

        Examples
        --------
        >>> sample = lstm.prepare_input_sample(X_df, len(X_df) - 1)
        >>> lstm.model_predict(sample, apply_scaler = False)
        """
        # Check that there are enough rows to build a sequence of self.timesteps rows
        if current_datetime_idx < self.timesteps - 1:
            raise ValueError(f"current_datetime_idx must be at least {self.timesteps - 1}")

        # rows of the sequence ending at current_datetime_idx (DataFrames keep their column names for the scaler)
        if isinstance(X, pd.DataFrame):
            sample = X.iloc[current_datetime_idx - self.timesteps + 1: current_datetime_idx + 1]
        else:
            sample = X[current_datetime_idx - self.timesteps + 1: current_datetime_idx + 1]

        if apply_scaler:
            sample = self.scaler.transform(sample)

        # add the batch dimension: (1, timesteps, n_features)
        sample = np.expand_dims(np.asarray(sample), axis=0)

        return sample


    def model_predict(self, data, apply_scaler=True, descale_result=True):
        """
        Predict with the trained network.

        Parameters
        ----------
        data : numpy.ndarray
            Input sequences of shape ``(n_samples, timesteps, n_features)``,
            or a single sequence of shape ``(timesteps, n_features)``.
        apply_scaler : bool, default True
            If ``True`` the features are scaled with ``self.scaler`` (pass
            ``False`` for data already scaled, e.g. built by
            :meth:`prepare_input_sample` with ``apply_scaler=True``).
        descale_result : bool, default True
            If ``True`` and ``scale_targets = True``, predictions are brought
            back to the original scale with ``self.Y_scaler``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_samples, n_outputs)``.

        Raises
        ------
        ValueError
            If ``data`` is neither 2D nor 3D, or if a 2D input does not have
            ``timesteps`` rows.

        Examples
        --------
        >>> raw_sample = lstm.prepare_input_sample(X_df, 500, apply_scaler = False)
        >>> lstm.model_predict(raw_sample)                   # scaled here
        """
        data = np.asarray(data)

        # a single sequence (timesteps, n_features) becomes a batch of one
        if(data.ndim == 2):
            if(data.shape[0] != self.timesteps):
                raise ValueError(f'A 2D input must be a single sequence of {self.timesteps} rows; '
                                 f'use prepare_input_sample to build sequences from a features table.')
            data = np.expand_dims(data, axis = 0)

        if(data.ndim != 3):
            raise ValueError(f'data must have shape (n_samples, timesteps, n_features), got {data.shape}')

        if apply_scaler:
            print('Scaler applied.')
            # the scaler works on 2D tables: flatten samples and timesteps, scale, restore the 3D shape
            n_samples, n_timesteps, n_features = data.shape
            data = self.scaler.transform(data.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)

        predictions = self.model.predict(data)

        if descale_result and self.scale_targets and (self.Y_scaler is not None):
            predictions = self.Y_scaler.inverse_transform(predictions)

        return predictions


    def compute_gradients(self, inputs, targets):
        """
        Gradient of the mean squared error with respect to the network inputs.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            Input sequences, shape ``(n_samples, timesteps, n_features)``.
        targets : tensorflow.Tensor
            Targets, shape ``(n_samples, n_outputs)``.

        Returns
        -------
        tensorflow.Tensor
            Gradients with the same shape as ``inputs``.
        """
        with tensorflow.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs)

            # MSE is used for every network type (as in fastANN): only the gradient magnitude matters here
            mse = tensorflow.keras.losses.MeanSquaredError()
            loss = mse(targets, predictions)

        return tape.gradient(loss, inputs)


    def gradient_feature_importance(self, feature_names = None):
        """
        Gradient-based feature importance computed on the test set.

        The importance of a feature is the mean absolute gradient of the loss
        with respect to that input, averaged over all test samples and all
        timesteps, then normalised to sum to 1. A horizontal Plotly bar chart
        is shown.

        Parameters
        ----------
        feature_names : list of str, optional
            Names of the features, in the column order of ``X_data``.
            Defaults to the columns of ``X_data`` (or to the saved
            ``X_feature_names``).

        Returns
        -------
        sorted_features_importance : list of float
            Normalised importances in increasing order.
        sorted_features_names : list of str
            Feature names in the same order.

        Examples
        --------
        >>> importance, names = lstm.gradient_feature_importance()
        >>> names[-1]   # most important feature
        """
        if(feature_names is None):
            feature_names = self.X_data.columns.tolist() if self.X_data is not None else self.X_feature_names

        if(self.validation_generator is None):
            self.create_generators()

        # all the test sequences and their targets, exactly as seen during validation
        batches = [self.validation_generator[i] for i in range(len(self.validation_generator))]
        X_tensor = tensorflow.convert_to_tensor(np.concatenate([batch[0] for batch in batches]), dtype=tensorflow.float32)
        Y_tensor = tensorflow.convert_to_tensor(np.concatenate([batch[1] for batch in batches]), dtype=tensorflow.float32)

        gradients = self.compute_gradients(X_tensor, Y_tensor)

        # average over samples and timesteps: one value per feature
        feature_importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))

        feature_importance = feature_importance / np.sum(feature_importance)

        # sort features by importance (increasing, so that the most important is on top of the bar chart)
        sorted_idx = np.argsort(feature_importance)
        sorted_features_names = [feature_names[i] for i in sorted_idx]
        sorted_features_importance = [feature_importance[i] for i in sorted_idx]

        fig = go.Figure(go.Bar(
            x=sorted_features_importance,
            y=sorted_features_names,
            orientation='h'
        ))
        fig.update_layout(
            title='Feature Importance (Gradient-based)',
            xaxis_title='Normalized Importance',
            yaxis_title='Features',
            height=800
        )
        fig.show()

        return sorted_features_importance, sorted_features_names
