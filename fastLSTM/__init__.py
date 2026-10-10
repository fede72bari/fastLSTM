"""
fastLSTM
========

A thin, opinionated wrapper around a Keras stacked-LSTM network for
time-series classification and regression.

The package exposes a single class, :class:`fastLSTM`, whose public API
(hyperparameter names, method names and saved-file layout) mirrors the sister
package ``fastANN`` so that the same keyword-argument workflow works with
both. Only the parameters and methods that make sense just for recurrent
networks (``LSTM_type``, ``timesteps``, ``steps_ahead``, ``class_weight``,
``create_generators``, ``create_sequences``, ``prepare_input_sample``) are
LSTM-specific. The network runs on TensorFlow or PyTorch (Keras 3 backends),
chosen with the ``backend`` parameter. fastANN options not available here: the pre-split
``X_train_s`` / ``Y_train`` / ``X_test_s`` / ``Y_test`` inputs, ``split_type``
(the split is always sequential) and ``autoencoder_mode``.

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

__version__ = '2.4.1'

# ---------------------------------------------------------------------------
#                              Libraries Import
# ---------------------------------------------------------------------------


# Multiprocessing
import multiprocessing

# Files Management
import sys
import gzip
import joblib
import glob
import csv
import json
import functools
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
# Keras (TensorFlow or PyTorch backend) is loaded by load_keras, below


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
#                    Deep learning backend (TensorFlow or PyTorch)
# ---------------------------------------------------------------------------

# The network is written with Keras 3, which runs on top of TensorFlow or
# PyTorch. Keras fixes its backend when it is first imported, so it is NOT
# imported here: it is loaded by the first instance created, with the backend
# requested by its `backend` parameter (see load_keras).
SUPPORTED_BACKENDS = ('tensorflow', 'torch', 'jax')


def load_keras(backend = None):
    """
    Import Keras 3 with the requested backend and return the module.

    Keras uses one backend per Python process: the first call (usually the
    first fastLSTM / fastANN instance) fixes it; later calls must ask for the
    same backend or for ``None``.

    Parameters
    ----------
    backend : {'tensorflow', 'torch', 'jax'}, optional
        Backend to use. ``None`` keeps the active one, or, if Keras is not
        loaded yet, uses the ``KERAS_BACKEND`` environment variable
        (``'tensorflow'`` when it is not set).

    Returns
    -------
    module
        The ``keras`` module.

    Raises
    ------
    ValueError
        If ``backend`` is not supported.
    RuntimeError
        If Keras is already running in this process with another backend
        (restart the Python kernel to change it).

    Examples
    --------
    >>> keras = load_keras('torch')
    >>> keras.backend.backend()
    'torch'
    """
    if((backend is not None) and (backend not in SUPPORTED_BACKENDS)):
        raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, not '{backend}'.")

    if('keras' not in sys.modules):
        if(backend is not None):
            os.environ['KERAS_BACKEND'] = backend

        if((os.environ.get('KERAS_BACKEND', 'tensorflow') == 'torch') and ('tensorflow' in sys.modules)):
            # with some TensorFlow/PyTorch builds, loading the Keras torch backend after TensorFlow crashes Python
            warnings.warn("TensorFlow is already imported in this process: if Python crashes while loading the "
                          "PyTorch backend, create the first fastLSTM/fastANN instance (or import torch) before "
                          "anything that imports TensorFlow.",
                          RuntimeWarning,
                          stacklevel = 3)

        import keras

    keras = sys.modules['keras']
    active_backend = keras.backend.backend()

    if((backend is not None) and (active_backend != backend)):
        raise RuntimeError(f"Keras is already using the '{active_backend}' backend in this Python process and it "
                           f"cannot be changed safely: restart the kernel to use '{backend}', or create the instance "
                           f"with backend = '{active_backend}' (or None).")

    return keras


# custom layers are registered under the name of the sister package that defines them, so .keras files are
# interchangeable with fastANN and fastGatedFourierAnalysisNetwork
KERAS_PACKAGE = 'fastGatedFourierAnalysisNetwork'


# ---------------------------------------------------------------------------
#                       Gated FAN layer (Keras 3, any backend)
# ---------------------------------------------------------------------------

# The layer classes need the keras module, which is imported only when the first instance is created (the backend
# must be chosen first). fan_layers() defines and registers them once and caches them here.
_FAN_OBJECTS = {}


def periodic_split(units, periodic_share):
    """
    Number of frequencies and of non-periodic units of a gated FAN layer.

    The layer output has ``units`` values: ``2 * d_p`` periodic ones (a cosine
    and a sine for each of the ``d_p`` frequencies) and ``d_p_bar``
    non-periodic ones, with ``2 * d_p`` as close as possible to
    ``units * periodic_share``.

    Parameters
    ----------
    units : int
        Output width of the layer.
    periodic_share : float
        Share (0 - 1) of the output given to the periodic part. 0 gives a
        plain dense layer (no frequencies, no gate).

    Returns
    -------
    d_p : int
        Number of learned frequencies (cosine + sine pairs).
    d_p_bar : int
        Number of non-periodic units.

    Raises
    ------
    ValueError
        If ``periodic_share`` is outside [0, 1) or ``units`` is too small to
        hold both parts.

    Examples
    --------
    >>> periodic_split(368, 1 / 3)
    (61, 246)
    """
    if(not (0 <= periodic_share < 1)):
        raise ValueError(f'periodic_share must be in [0, 1), not {periodic_share}.')

    units = int(units)
    d_p = int(round(units * periodic_share / 2))

    if((periodic_share > 0) and (d_p == 0)):
        d_p = 1

    d_p_bar = units - 2 * d_p

    if(d_p_bar < 1):
        raise ValueError(f'A layer of {units} units cannot hold {2 * d_p} periodic units and at least one '
                         f'non-periodic unit: increase model_relative_width or reduce periodic_share.')

    return d_p, d_p_bar


def fan_layers(keras = None):
    """
    Define (once) and return the custom Keras objects of the package.

    Returns the dictionary ``{'GatedFAN': <layer class>, 'FrequencyClip':
    <constraint class>}``. The classes are registered in the Keras
    serialization registry (package ``'fastGatedFourierAnalysisNetwork'``), so
    saved ``.keras`` files reload with ``keras.models.load_model`` once this
    function has been called; :meth:`fastGatedFourierAnalysisNetwork.load_model`
    does it automatically. The dictionary can also be passed as
    ``custom_objects``.

    Parameters
    ----------
    keras : module, optional
        The ``keras`` module; defaults to :func:`load_keras` ``()``.

    Returns
    -------
    dict
        Custom objects by name.

    Examples
    --------
    >>> objects = fan_layers()
    >>> layer = objects['GatedFAN'](units = 96)
    """
    if(_FAN_OBJECTS):
        return _FAN_OBJECTS

    if(keras is None):
        keras = load_keras()

    ops = keras.ops


    @keras.saving.register_keras_serializable(package = KERAS_PACKAGE)
    class FrequencyClip(keras.constraints.Constraint):
        """
        Keep the absolute value of every frequency inside ``[min_frequency, max_frequency]``.

        Applied by the optimizer after each update; the sign of each weight is
        kept. ``None`` leaves that side unbounded.
        """

        def __init__(self, min_frequency = None, max_frequency = None):
            self.min_frequency = None if min_frequency is None else float(min_frequency)
            self.max_frequency = None if max_frequency is None else float(max_frequency)

        def __call__(self, w):
            magnitude = ops.abs(w)

            if(self.min_frequency is not None):
                magnitude = ops.maximum(magnitude, self.min_frequency)

            if(self.max_frequency is not None):
                magnitude = ops.minimum(magnitude, self.max_frequency)

            # a weight exactly 0 has sign 0: it is sent to +min_frequency
            sign = ops.where(w >= 0, 1.0, -1.0)
            return sign * magnitude

        def get_config(self):
            return {'min_frequency': self.min_frequency, 'max_frequency': self.max_frequency}


    @keras.saving.register_keras_serializable(package = KERAS_PACKAGE)
    class GatedFAN(keras.layers.Layer):
        """
        Gated Fourier Analysis Network layer.

        Output (``units`` values)::

            [ g * cos(x Wp) , g * sin(x Wp) , (1 - mean(g)) * act(x Wp_bar + b) ]

        * ``Wp`` (``n_inputs x d_p``): learned frequencies. Each periodic unit
          is a sinusoid along a learned direction of the input, with period
          ``2 * pi / |Wp[:, j]|`` in input units.
        * ``g = sigmoid(gate)``: one trainable gate per frequency (starts at
          0.5). It scales the cosine and sine of that frequency; the
          non-periodic part is scaled by ``1 - mean(g)``, so the layer can
          move its capacity between periodic and non-periodic modelling.
        * ``Wp_bar``, ``b``: an ordinary dense layer with activation ``act``.

        With ``gated = False`` the gate is not created and ``g = 1``, ``1 -
        mean(g)`` is replaced by 1 (plain FAN layer). With ``periodic_share =
        0`` the layer is a plain dense layer.

        Parameters
        ----------
        units : int
            Output width.
        periodic_share : float, default 1/3
            Share of the output given to the cosine + sine pairs (see
            :func:`periodic_split`).
        activation : str, default 'gelu'
            Keras activation of the non-periodic part.
        gated : bool, default True
            Create the trainable gates.
        frequency_init_std : float, default 1.0
            Standard deviation of the normal initialisation of ``Wp``. With
            standardized inputs, 1 gives periods of a few standard deviations
            (random Fourier features); use larger values for faster cycles.
        min_frequency, max_frequency : float, optional
            Bounds of ``|Wp|`` (enforced with :class:`FrequencyClip`), e.g.
            to forbid periods longer than the training window.
        """

        def __init__(self, units, periodic_share = 1 / 3, activation = 'gelu', gated = True,
                     frequency_init_std = 1.0, min_frequency = None, max_frequency = None, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)
            self.periodic_share = float(periodic_share)
            self.activation_name = activation
            self.activation = keras.activations.get(activation)
            self.gated = bool(gated)
            self.frequency_init_std = float(frequency_init_std)
            self.min_frequency = min_frequency
            self.max_frequency = max_frequency
            self.d_p, self.d_p_bar = periodic_split(self.units, self.periodic_share)

        def build(self, input_shape):
            n_inputs = int(input_shape[-1])

            if(self.d_p > 0):
                constraint = None
                if((self.min_frequency is not None) or (self.max_frequency is not None)):
                    constraint = FrequencyClip(self.min_frequency, self.max_frequency)

                self.Wp = self.add_weight(name = 'Wp', shape = (n_inputs, self.d_p),
                                          initializer = keras.initializers.RandomNormal(0.0, self.frequency_init_std),
                                          constraint = constraint, trainable = True)

                if(self.gated):
                    # sigmoid(0) = 0.5: periodic and non-periodic parts start with the same weight
                    self.gate = self.add_weight(name = 'gate', shape = (self.d_p,), initializer = 'zeros', trainable = True)

            self.Wp_bar = self.add_weight(name = 'Wp_bar', shape = (n_inputs, self.d_p_bar), initializer = 'glorot_uniform', trainable = True)
            self.b = self.add_weight(name = 'b', shape = (self.d_p_bar,), initializer = 'zeros', trainable = True)

        def call(self, x):
            non_periodic = self.activation(ops.matmul(x, self.Wp_bar) + self.b)

            if(self.d_p == 0):
                return non_periodic

            wx = ops.matmul(x, self.Wp)

            if(self.gated):
                g = ops.sigmoid(self.gate)
                periodic = ops.concatenate([g * ops.cos(wx), g * ops.sin(wx)], axis = -1)
                non_periodic = (1.0 - ops.mean(g)) * non_periodic
            else:
                periodic = ops.concatenate([ops.cos(wx), ops.sin(wx)], axis = -1)

            return ops.concatenate([periodic, non_periodic], axis = -1)

        def compute_output_shape(self, input_shape):
            return tuple(input_shape[:-1]) + (self.units,)

        def gate_values(self):
            """Gates ``g`` (numpy array of ``d_p`` values; ones when the layer is not gated, empty without periodic part)."""
            if(self.d_p == 0):
                return np.array([])
            if(not self.gated):
                return np.ones(self.d_p)
            return keras.ops.convert_to_numpy(ops.sigmoid(self.gate))

        def periods(self):
            """Period of each sinusoid along its own direction, ``2 * pi / |Wp[:, j]|``, in units of the (scaled) input."""
            if(self.d_p == 0):
                return np.array([])
            w = keras.ops.convert_to_numpy(self.Wp)
            return 2 * np.pi / np.maximum(np.linalg.norm(w, axis = 0), 1e-12)

        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'periodic_share': self.periodic_share, 'activation': self.activation_name,
                           'gated': self.gated, 'frequency_init_std': self.frequency_init_std,
                           'min_frequency': self.min_frequency, 'max_frequency': self.max_frequency})
            return config


    _FAN_OBJECTS.update({'GatedFAN': GatedFAN, 'FrequencyClip': FrequencyClip})
    return _FAN_OBJECTS


def make_auc_callback(keras, predict, y_true, rows_mask = None, name = 'val_monitored_auc'):
    """
    Keras callback that adds the ROC AUC of the validation predictions to the logs of every epoch.

    The value is written into the epoch logs under ``name`` BEFORE early
    stopping and checkpoint read them (the callback must come first in the
    callbacks list), so it can be used as ``early_stop_monitor_metric`` /
    ``checkpoint_monitor_metric`` (mode ``'max'``) and appears in the
    training history. With several outputs (several binary targets or
    steps) the AUC is computed on all the outputs pooled together.

    Parameters
    ----------
    keras : module
        The Keras module (see :func:`load_keras`).
    predict : callable
        Function without arguments returning the validation predictions,
        shape ``(n_samples, n_outputs)``.
    y_true : array-like
        Actual 0/1 targets aligned with the predictions.
    rows_mask : array-like of bool, optional
        Samples on which the AUC is computed (e.g. the hard cases); all
        samples when ``None``.
    name : str, default 'val_monitored_auc'
        Key of the value in the logs and in the training history.

    Returns
    -------
    keras.callbacks.Callback
        The callback; NaN is logged when the selected samples contain one
        class only.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype = float)
    y_true = y_true.reshape(len(y_true), -1)
    mask = np.ones(len(y_true), dtype = bool) if rows_mask is None else np.asarray(rows_mask, dtype = bool)
    if(len(mask) != len(y_true)):
        raise ValueError(f'rows_mask has {len(mask)} values but there are {len(y_true)} validation samples.')

    class MonitoredAUC(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs = None):
            predictions = np.asarray(predict(), dtype = float).reshape(len(y_true), -1)
            selected_true, selected_pred = y_true[mask].ravel(), predictions[mask].ravel()
            auc = roc_auc_score(selected_true, selected_pred) if len(np.unique(selected_true)) == 2 else float('nan')
            if(logs is not None):
                logs[name] = auc
            print(f' - {name}: {auc:.4f} ({int(mask.sum())} samples)')

    return MonitoredAUC()


def in_strategy_scope(method):
    """
    Run a method that creates Keras variables (model building, compiling,
    loading) inside the scope of the TensorFlow distribution strategy given
    as ``distribution_strategy`` (e.g. ``tf.distribute.MirroredStrategy()``
    for several GPUs). Without a strategy the method runs unchanged.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        strategy = getattr(self, 'distribution_strategy', None)
        if(strategy is None):
            return method(self, *args, **kwargs)
        with strategy.scope():
            return method(self, *args, **kwargs)
    return wrapper


def pad_to_multiple(indices, multiple):
    """
    Complete a batch of sample indices to a multiple of ``multiple`` by repeating the last one.

    Data-parallel training on ``multiple`` devices (e.g. the 8 cores of a TPU
    v5e-8) splits every batch evenly among them, so a final partial batch of,
    say, 335 samples fails on 8 devices.

    Parameters
    ----------
    indices : numpy.ndarray
        Indices of the samples of one batch.
    multiple : int
        Required multiple (1 = unchanged).

    Returns
    -------
    numpy.ndarray
        The indices, followed by ``(-len(indices)) % multiple`` copies of the
        last one.
    """
    pad = (-len(indices)) % max(int(multiple), 1)
    if(pad == 0 or len(indices) == 0):
        return indices
    return np.concatenate([indices, np.repeat(indices[-1:], pad, axis = 0)])


def make_sequence_generator(keras, data, targets, length, batch_size, end_index = None, shuffle = False, seed = 42,
                            sample_weights = None, batch_multiple = 1):
    """
    Batches of (sequence, target) samples for Keras, on any backend.

    Same pairing as Keras' former ``TimeseriesGenerator``: target row ``t`` is
    paired with the data rows ``t - length ... t - 1``, for
    ``t = length ... end_index``. With ``shuffle = False`` the samples are
    served in chronological order; with ``shuffle = True`` the ORDER OF THE
    SAMPLES is permuted at every epoch, so each batch mixes windows from the
    whole period. The rows inside each window always stay in chronological
    order (only the sample axis is permuted, never the time axis).

    Parameters
    ----------
    keras : module
        The Keras module (see :func:`load_keras`).
    data : numpy.ndarray
        Features, shape ``(n_rows, n_features)``.
    targets : numpy.ndarray
        Targets, shape ``(n_rows, n_outputs)``.
    length : int
        Number of rows of each sequence (``timesteps``).
    batch_size : int
        Number of samples per batch.
    end_index : int, optional
        Last target row used (inclusive); defaults to the last row.
    shuffle : bool, default False
        Permute the samples among the batches at every epoch (training set).
        Never use it for a validation/test generator whose predictions must
        be aligned with the dates.
    seed : int, default 42
        Seed of the permutations (reproducible runs).
    sample_weights : array-like, optional
        One weight per row of ``data``: each sample gets the weight of its
        target row and the batches become ``(X_batch, Y_batch, w_batch)``.
    batch_multiple : int, default 1
        Every batch size is made a multiple of it: the last (partial) batch is
        completed by repeating its last sample. Needed with data-parallel
        distribution on several devices (e.g. 8 TPU cores), which splits each
        batch evenly; at most ``batch_multiple - 1`` duplicate samples per
        epoch. Predictions on the generator must be cut to
        ``generator.n_samples`` rows.

    Returns
    -------
    keras.utils.PyDataset
        Object with ``len(generator)`` batches; ``generator[i]`` returns
        ``(X_batch, Y_batch)`` (plus the weights, see ``sample_weights``)
        with ``X_batch`` of shape ``(batch, length, n_features)``.
        ``generator.n_samples`` is the number of real samples.

    Raises
    ------
    ValueError
        If there are no samples (not enough rows for ``length``).
    """
    data = np.asarray(data, dtype = np.float32)
    targets = np.asarray(targets, dtype = np.float32)
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype = np.float32)
    if((weights is not None) and (len(weights) != len(data))):
        raise ValueError(f'sample_weights has {len(weights)} values but data has {len(data)} rows.')
    last_index = len(data) - 1 if end_index is None else end_index

    if(length > last_index):
        raise ValueError(f'Not enough rows: sequences of {length} rows need at least {length + 1} rows '
                         f'(last usable target row is {last_index}).')

    # the class is defined here, on the Keras module currently loaded, so that Keras recognises it
    class SequenceGenerator(keras.utils.PyDataset):

        def __init__(self):
            super().__init__()
            # target rows of all the samples; with shuffle their order is permuted at every epoch
            self.target_rows = np.arange(length, last_index + 1)
            self.rng = np.random.default_rng(seed)
            if(shuffle):
                self.rng.shuffle(self.target_rows)

        def __len__(self):
            return (last_index - length + batch_size) // batch_size

        def __getitem__(self, index):
            if(index < 0):
                index += len(self)
            rows = self.target_rows[index * batch_size:(index + 1) * batch_size]
            rows = pad_to_multiple(rows, batch_multiple)
            # sample for target row r: the `length` rows before r, in chronological order
            X_batch = np.stack([data[row - length:row] for row in rows])
            if(weights is not None):
                return X_batch, targets[rows], weights[rows]
            return X_batch, targets[rows]

        def on_epoch_end(self):
            # new order of the samples for the next epoch (the windows themselves are unchanged)
            if(shuffle):
                self.rng.shuffle(self.target_rows)

    generator = SequenceGenerator()
    # target row of each sample in serving order without shuffling (used to align predictions and actual targets)
    generator.sample_target_rows = np.arange(length, last_index + 1)
    generator.n_samples = len(generator.sample_target_rows)
    return generator


def make_grouped_sequence_generator(keras, data, targets, length, batch_size, groups, steps_ahead = 1,
                                    shuffle = False, seed = 42, sample_weights = None, batch_multiple = 1):
    """
    Batches of (sequence, target) samples whose windows never cross groups.

    Use it when the rows hold several interleaved series, e.g. an option
    chain where each row is one contract at one time: the window of a sample
    is made of the ``length`` previous rows OF THE SAME GROUP (contract), in
    the order in which they appear in ``data`` (that must be chronological),
    and its targets are the target row and the following
    ``steps_ahead - 1`` rows of the same group. A group contributes
    ``n_rows_of_group - length - steps_ahead + 1`` samples (none if it is
    too short).

    Parameters
    ----------
    keras : module
        The Keras module (see :func:`load_keras`).
    data : numpy.ndarray
        Features, shape ``(n_rows, n_features)``, rows in chronological order.
    targets : numpy.ndarray
        Targets, shape ``(n_rows, n_targets)`` (single step: the multi-step
        targets are built here, step-major, within each group).
    length : int
        Number of rows of each window (``timesteps``).
    batch_size : int
        Number of samples per batch.
    groups : array-like
        Group label of each row (e.g. contract id), length ``n_rows``.
    steps_ahead : int, default 1
        Number of consecutive target rows of the group predicted per sample.
    shuffle : bool, default False
        Permute the samples among the batches at every epoch (windows are
        unchanged). Never use it for a validation/test generator.
    seed : int, default 42
        Seed of the permutations.
    sample_weights : array-like, optional
        One weight per row of ``data``: each sample gets the weight of its
        first target row and the batches become ``(X_batch, Y_batch,
        w_batch)``.
    batch_multiple : int, default 1
        Every batch size is made a multiple of it (see
        :func:`make_sequence_generator`); ``generator.n_samples`` is the
        number of real samples.

    Returns
    -------
    keras.utils.PyDataset
        ``generator[i]`` returns ``(X_batch, Y_batch)`` with ``X_batch`` of
        shape ``(batch, length, n_features)`` and ``Y_batch`` of shape
        ``(batch, steps_ahead * n_targets)``. ``generator.sample_target_rows``
        is the ``(n_samples, steps_ahead)`` array of target rows of each
        sample in unshuffled order.

    Raises
    ------
    ValueError
        If ``groups`` has a different length from ``data`` or no group is
        long enough to give a sample.
    """
    data = np.asarray(data, dtype = np.float32)
    targets = np.asarray(targets, dtype = np.float32)
    groups = np.asarray(groups)
    if(len(groups) != len(data)):
        raise ValueError(f'groups has {len(groups)} labels but data has {len(data)} rows.')
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype = np.float32)
    if((weights is not None) and (len(weights) != len(data))):
        raise ValueError(f'sample_weights has {len(weights)} values but data has {len(data)} rows.')

    # rows of each group in data order (stable sort keeps the chronological order inside a group)
    order = np.argsort(pd.factorize(groups)[0], kind = 'stable')
    codes = pd.factorize(groups)[0][order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(order)]

    window_rows, target_rows = [], []
    span = length + steps_ahead
    for start, end in zip(starts, ends):
        rows = order[start:end]
        n_samples = len(rows) - span + 1
        if(n_samples <= 0):
            continue
        # sliding windows over the rows of the group: first `length` are the inputs, the rest the targets
        windows = np.lib.stride_tricks.sliding_window_view(rows, span)
        window_rows.append(windows[:, :length])
        target_rows.append(windows[:, length:])

    if(len(window_rows) == 0):
        raise ValueError(f'No group has at least {span} rows (timesteps {length} + steps_ahead {steps_ahead}).')

    window_rows = np.concatenate(window_rows)
    target_rows = np.concatenate(target_rows)
    # samples sorted by the time of their first target row: unshuffled batches follow the chronology
    chronological = np.argsort(target_rows[:, 0], kind = 'stable')
    window_rows, target_rows = window_rows[chronological], target_rows[chronological]

    class GroupedSequenceGenerator(keras.utils.PyDataset):

        def __init__(self):
            super().__init__()
            self.samples = np.arange(len(window_rows))
            self.rng = np.random.default_rng(seed)
            if(shuffle):
                self.rng.shuffle(self.samples)

        def __len__(self):
            return (len(window_rows) + batch_size - 1) // batch_size

        def __getitem__(self, index):
            if(index < 0):
                index += len(self)
            samples = self.samples[index * batch_size:(index + 1) * batch_size]
            samples = pad_to_multiple(samples, batch_multiple)
            # step-major targets: all targets of step 1, then all targets of step 2, ...
            Y_batch = targets[target_rows[samples]].reshape(len(samples), -1)
            if(weights is not None):
                return data[window_rows[samples]], Y_batch, weights[target_rows[samples, 0]]
            return data[window_rows[samples]], Y_batch

        def on_epoch_end(self):
            if(shuffle):
                self.rng.shuffle(self.samples)

    generator = GroupedSequenceGenerator()
    generator.sample_target_rows = target_rows
    generator.n_samples = len(target_rows)
    return generator


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
    Training and validation samples are created by a sequence generator
    (:func:`make_sequence_generator`, same behaviour as Keras' former
    ``TimeseriesGenerator``) with ``length = timesteps``: the target row ``t``
    is paired with the feature rows ``t - timesteps ... t - 1`` (row ``t``
    itself is NOT part of the window). If ``Y_data`` row ``t`` already holds
    the future value to predict for the bar ``t``, take this into account
    when building ``Y_data``. As a consequence the first ``timesteps`` rows of
    the test set have no prediction. With ``steps_ahead = k > 1`` the sample
    is trained on target rows ``t ... t + k - 1`` at once, so the last
    ``k - 1`` rows of each set are not used as samples either.

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
        * ``'classificator'``: output layer with ``n_targets * steps_ahead``
          units and ``last_layer_activation``.
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
        one-hot multi-class targets (only with ``steps_ahead = 1``: softmax
        would normalise across all steps).
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
        saved (and reloaded at the end of the training). Used only when
        ``save_best_only = True``.
    checkpoint_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``checkpoint_monitor_metric`` has to be maximised or minimised.
    early_stop_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``early_stop_monitor_metric`` has to be maximised or
        minimised.
    history_metrics : list of str, optional
        Columns of the training history plotted by
        :meth:`plot_training_history`. When ``None``: ``['loss', 'val_loss']``
        for regressors, the first of ``metrics`` and its ``val_`` counterpart
        (e.g. ``['accuracy', 'val_accuracy']``) for classificators.
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
        Number of consecutive future steps predicted for each target, for
        both classificators and regressors. With ``steps_ahead = k`` the
        network has ``n_targets * k`` outputs and each sample is trained on
        ``Y[t], Y[t + 1], ..., Y[t + k - 1]`` (``Y[t]`` being the row right
        after the input window). Outputs are step-major and named by
        :meth:`output_column_names` (e.g. ``'up_step_1'``, ``'up_step_2'``).
        The last ``k - 1`` rows of the training and test sets have no
        complete future and are not used as samples.
    class_weight : dict, optional
        Keras class weights, e.g. ``{0: 1.0, 1: 3.0}``, to rebalance
        unbalanced classes during training. Meaningful with a single output
        (one binary target, ``steps_ahead = 1``): with more outputs Keras
        applies it to the argmax of each target row.
    shuffle : bool, default False
        Permute the training samples among the batches at every epoch. Each
        sample (a window of ``timesteps`` consecutive rows and its targets)
        is left intact and the train/test split stays chronological: only the
        order in which the windows are presented to the network changes.
        Recommended for strongly autocorrelated targets (e.g. trend/ZigZag
        labels): in chronological order each batch holds consecutive days,
        often of one class only, and every epoch ends on the last months of
        the training set, which makes the learning curves jump. The test
        generator is never shuffled. ``False`` keeps the behaviour of the
        previous versions.
    sequence_groups : array-like, optional
        Group label of each row of ``X_data`` (e.g. the option contract id)
        when the rows hold several interleaved series. Windows and multi-step
        targets are then built only from rows of the same group
        (:func:`make_grouped_sequence_generator`), so a sample never mixes two
        contracts. The rows must be in chronological order (the train/test
        split stays the first/last ``train_size_rate`` of the rows). Not
        stored in the saved files: pass it again after :meth:`load_all`
        (attribute ``sequence_groups``) before using the generators.
    backend : {'tensorflow', 'torch', 'jax'}, optional
        Deep learning framework that runs the network (through Keras 3):
        ``'jax'`` is the one to use on TPUs (e.g. Kaggle TPU v5e-8).
        ``None`` (default) keeps the backend already active in the Python
        process, or uses the ``KERAS_BACKEND`` environment variable
        (``'tensorflow'`` when it is not set). The backend is fixed for the
        whole process by the first instance: to change it restart the kernel.
        Saved models can be reloaded with either backend.
    input_projection : {None, 'gated_fan'}, default None
        ``'gated_fan'`` adds, between the input and the first LSTM, a gated
        Fourier Analysis Network layer applied to every row (bar) of the
        window: ``[g cos(xWp), g sin(xWp), (1 - mean(g)) act(xWp_bar + b)]``,
        i.e. learned periodic components (cycles of the features and phases)
        and a normal dense part, followed by ``input_projection_dropout``.
        The LSTM layers stay standard (fast cuDNN kernels on GPU). Same layer
        as the sister package ``fastGatedFourierAnalysisNetwork``.
    input_projection_width : float, default 4
        Output width of the projection, relative to the number of features.
        The LSTM widths (``model_relative_width``) stay relative to the
        number of FEATURES, not to the projection width.
    input_projection_dropout : float, default 0.0
        Dropout after the projection.
    input_projection_activation : str, default 'gelu'
        Activation of the non-periodic part of the projection.
    periodic_share : float, default 1/3
        Share of the projection outputs given to the cosine + sine pairs.
    gated : bool, default True
        Trainable gate per frequency; ``False`` gives the ungated FAN layer.
    frequency_init_std : float, default 1.0
        Standard deviation of the normal initialisation of the frequencies
        (inputs are standardized).
    sample_weight : array-like, optional
        One weight per row of ``X_data``; the rows of the training set weight
        the loss of their samples (each sample takes the weight of its first
        target row), e.g. larger weights for the hard cases (options with the
        strike close to the underlying). Validation is not weighted. Cannot
        be combined with ``class_weight``. Not stored in the saved files
        (only whether it was used).
    monitor_auc : bool, default False
        Compute the ROC AUC of the validation predictions at the end of every
        epoch (binary targets) and log it as ``'val_monitored_auc'``: use it
        as ``early_stop_monitor_metric`` / ``checkpoint_monitor_metric`` with
        mode ``'max'`` to choose the epoch on the AUC instead of the loss or
        the accuracy. Costs one extra prediction pass on the validation set
        per epoch.
    monitor_auc_rows : array-like of bool, optional
        One value per row of ``X_data``: the AUC is computed only on the
        validation samples whose (first) target row is ``True`` (e.g. strike
        within 2% of the underlying). Implies ``monitor_auc = True``. Not
        stored in the saved files (only whether it was used).
    batch_multiple : int, optional
        Every batch of the generators is completed to a multiple of it (the
        last partial batch repeats its last sample). ``None`` (default) uses
        the number of devices of the active Keras distribution (e.g. 8 with
        ``keras.distribution.DataParallel`` on a TPU v5e-8) and 1 without
        distribution. ``batch_size`` must be a multiple of it.
    distribution_strategy : tf.distribute.Strategy, optional
        TensorFlow backend only: the model is built, compiled and loaded
        inside ``distribution_strategy.scope()``, so that ``fit`` trains it on
        all the devices of the strategy, e.g.
        ``tf.distribute.MirroredStrategy()`` for the two T4 GPUs of Kaggle
        (every batch is split among the GPUs). ``None`` (default) uses one
        device. With JAX use ``keras.distribution.DataParallel`` instead.
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
    backend : str
        Active Keras backend (``'tensorflow'``, ``'torch'`` or ``'jax'``).
    keras : module
        The Keras module used by the instance.
    model : keras.Sequential
        The network (rebuilt by :meth:`network_structure_set_compile`,
        replaced by the saved checkpoint after :meth:`network_training`).
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
    generator, validation_generator : keras.utils.PyDataset
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
                 shuffle = False,
                 sequence_groups = None,
                 input_projection = None,
                 input_projection_width = 4,
                 input_projection_dropout = 0.0,
                 input_projection_activation = 'gelu',
                 periodic_share = 1 / 3,
                 gated = True,
                 frequency_init_std = 1.0,
                 sample_weight = None,
                 monitor_auc = False,
                 monitor_auc_rows = None,
                 batch_multiple = None,
                 distribution_strategy = None,
                 backend = None,
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

        # Keras with the requested backend (fixed for the whole Python process by the first instance)
        self.keras = load_keras(backend)
        self.backend = self.keras.backend.backend()
        print(f'Keras backend: {self.backend}')

        self.model = self.keras.Sequential()
        self.save_best_only = save_best_only

        self.data_storage_path = data_storage_path
        self.model_name = model_name

        self.model_relative_width = model_relative_width
        self.model_dropout = model_dropout
        self.LSTM_type = LSTM_type
        self.learning_rate = learning_rate
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.metrics = [] if metrics is None else ([metrics] if isinstance(metrics, str) else list(metrics))
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
        self.shuffle = shuffle
        self.sequence_groups = None if sequence_groups is None else np.asarray(sequence_groups)
        if(input_projection not in (None, 'gated_fan')):
            raise ValueError(f"input_projection must be None or 'gated_fan', not '{input_projection}'.")
        self.input_projection = input_projection
        self.input_projection_width = input_projection_width
        self.input_projection_dropout = input_projection_dropout
        self.input_projection_activation = input_projection_activation
        self.periodic_share = periodic_share
        self.gated = gated
        self.frequency_init_std = frequency_init_std
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype = float)
        self.monitor_auc_rows = None if monitor_auc_rows is None else np.asarray(monitor_auc_rows, dtype = bool)
        # a rows mask implies the AUC monitor
        self.monitor_auc = bool(monitor_auc) or (self.monitor_auc_rows is not None)
        self.batch_multiple = batch_multiple
        self.distribution_strategy = distribution_strategy
        if((self.class_weight is not None) and (self.sample_weight is not None)):
            raise ValueError('Use class_weight or sample_weight, not both (fold the class weights into sample_weight).')

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
        training. Keys follow fastANN (without ``split_type`` and
        ``autoencoder_mode``), plus ``Y_scaler_file_name`` and the
        LSTM-specific ones (``LSTM_type``, ``timesteps``, ``steps_ahead``,
        ``class_weight``).

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
                               'class_weight': class_weight,
                               'shuffle': self.shuffle,
                               # the labels are not saved (see sequence_groups): only whether they were used
                               'sequence_groups': getattr(self, 'sequence_groups', None) is not None,
                               'input_projection': getattr(self, 'input_projection', None),
                               'input_projection_width': getattr(self, 'input_projection_width', 4),
                               'input_projection_dropout': getattr(self, 'input_projection_dropout', 0.0),
                               'input_projection_activation': getattr(self, 'input_projection_activation', 'gelu'),
                               'periodic_share': getattr(self, 'periodic_share', 1 / 3),
                               'gated': getattr(self, 'gated', True),
                               'frequency_init_std': getattr(self, 'frequency_init_std', 1.0),
                               # arrays not saved, as sequence_groups: only whether they were used
                               'sample_weight': getattr(self, 'sample_weight', None) is not None,
                               'monitor_auc': getattr(self, 'monitor_auc', False),
                               'monitor_auc_rows': getattr(self, 'monitor_auc_rows', None) is not None,
                               # informative: saved models can be reloaded with either backend
                               'backend': self.backend
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
        rebuilt so that they reflect the loaded settings, and new unfitted
        scalers are created if the saved ``scaler_type`` differs from the
        current one (the fitted ones are then loaded by :meth:`load_scaler`).

        ``data_storage_path`` is NOT taken from the file: the folder the
        hyperparameters were loaded from is kept, so a model folder can be
        moved or copied to another machine (the saved value stays in
        ``self.hyperparameters`` for reference).

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
        # data_storage_path is deliberately not restored (see docstring): the saved one may not exist any more
        self.model_file_name = self.get_hyperparameter('model_file_name')
        self.scaler_file_name = self.get_hyperparameter('scaler_file_name')
        self.Y_scaler_file_name = self.get_hyperparameter('Y_scaler_file_name')

        self.training_history_file_name = self.get_hyperparameter('training_history_file_name')
        self.X_data_df_file_name = self.get_hyperparameter('X_data_df_file_name')
        self.Y_data_df_file_name = self.get_hyperparameter('Y_data_df_file_name')

        self.scale_targets = self.get_hyperparameter('scale_targets', False)

        # scaler objects must match the loaded scaler_type
        if(type(self.scaler) is not type(self.new_scaler())):
            self.scaler = self.new_scaler()

        if(self.scale_targets and ((self.Y_scaler is None) or (type(self.Y_scaler) is not type(self.new_scaler())))):
            self.Y_scaler = self.new_scaler()

        # LSTM specific
        if('LSTM_type' in self.hyperparameters):
            self.LSTM_type = self.hyperparameters['LSTM_type']
        elif(self.hyperparameters.get('loss') == 'mse'):
            # files of older versions do not store LSTM_type: 'mse' was the only regression loss they supported
            self.LSTM_type = 'regressor'
        else:
            self.LSTM_type = 'classificator'
        self.timesteps = self.get_hyperparameter('timesteps', self.timesteps)
        self.steps_ahead = self.get_hyperparameter('steps_ahead', self.steps_ahead)

        class_weight = self.get_hyperparameter('class_weight')
        if(class_weight is not None):
            # JSON turned the integer class labels into strings
            class_weight = {int(label) if str(label).lstrip('-').isdigit() else label: weight
                            for label, weight in class_weight.items()}
        self.class_weight = class_weight
        # files of versions < 2.1 do not store shuffle: they always trained in chronological order
        self.shuffle = self.get_hyperparameter('shuffle', False)
        # files of versions < 2.4 have no input projection
        self.input_projection = self.get_hyperparameter('input_projection', None)
        self.input_projection_width = self.get_hyperparameter('input_projection_width', 4)
        self.input_projection_dropout = self.get_hyperparameter('input_projection_dropout', 0.0)
        self.input_projection_activation = self.get_hyperparameter('input_projection_activation', 'gelu')
        self.periodic_share = self.get_hyperparameter('periodic_share', 1 / 3)
        self.gated = self.get_hyperparameter('gated', True)
        self.frequency_init_std = self.get_hyperparameter('frequency_init_std', 1.0)

        # callbacks depend on the loaded settings
        self.early_stop_patience_set(self.early_stop_patience)


    def compile_metrics(self):
        """
        Metrics passed to ``model.compile``, with ``'accuracy'`` made explicit.

        With more than one output Keras 3 turns the string ``'accuracy'``
        into categorical accuracy (argmax across the outputs), which is wrong
        for independent sigmoid outputs: on multi-step or multi-target binary
        targets a constant model can score above 90%. ``'accuracy'`` (or
        ``'acc'``) is therefore replaced by ``BinaryAccuracy`` when the output
        activation is sigmoid and by ``CategoricalAccuracy`` when it is
        softmax; the metric keeps the name ``'accuracy'``, so the history
        columns (``accuracy``, ``val_accuracy``) do not change.

        Returns
        -------
        list
            Metrics for ``model.compile``.
        """
        resolved = []

        for metric in self.metrics:
            if((metric in ('accuracy', 'acc')) and (self.last_layer_activation == 'sigmoid')):
                resolved.append(self.keras.metrics.BinaryAccuracy(name = 'accuracy'))
            elif((metric in ('accuracy', 'acc')) and (self.last_layer_activation == 'softmax')):
                resolved.append(self.keras.metrics.CategoricalAccuracy(name = 'accuracy'))
            else:
                resolved.append(metric)

        return resolved


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

        # older versions saved str(loss object), e.g. "<...BinaryCrossentropy object at ...>" (Keras 2) or
        # "<LossFunctionWrapper(<function binary_crossentropy ...>" (Keras 3): such reprs are matched case and
        # underscore insensitively, plain names only by their exact class name, so that 'binary_crossentropy'
        # and the other Keras names pass through unchanged. Sparse... must be tested before Categorical...
        if(isinstance(loss, str) and loss.startswith('<')):
            loss_key = loss.lower().replace('_', '')
        elif(isinstance(loss, str)):
            loss_key = loss.lower()
        else:
            loss_key = None

        if(loss_key is None):
            self.loss_function = loss
        elif('sparsecategoricalcrossentropy' in loss_key):
            self.loss_function = self.keras.losses.SparseCategoricalCrossentropy()
        elif('categoricalcrossentropy' in loss_key):
            self.loss_function = self.keras.losses.CategoricalCrossentropy()
        elif('binarycrossentropy' in loss_key):
            self.loss_function = self.keras.losses.BinaryCrossentropy()
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

        self.early_stop = self.keras.callbacks.EarlyStopping(monitor = self.early_stop_monitor_metric,
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
        self.model_checkpoint = self.keras.callbacks.ModelCheckpoint(self.data_storage_path + model_file_name,
                                                monitor = self.checkpoint_monitor_metric,
                                                mode = self.checkpoint_mode,
                                                verbose = 1,
                                                save_best_only = save_best_only)

        return self.model_checkpoint


    @in_strategy_scope
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

        keras = self.keras

        # reset
        self.model = keras.Sequential()

        # input layer: sequences of `timesteps` rows with `n_features` columns
        self.model.add(keras.Input(shape = (self.timesteps, n_features)))

        if(getattr(self, 'input_projection', None) == 'gated_fan'):
            # gated FAN projection of every row of the window (periodic + non-periodic parts), then the LSTMs
            units = int(n_features * self.input_projection_width)
            print(f'Input projection: gated FAN, {units} outputs, dropout {self.input_projection_dropout}')
            self.model.add(fan_layers(keras)['GatedFAN'](units = units, periodic_share = self.periodic_share,
                                                         activation = self.input_projection_activation, gated = self.gated,
                                                         frequency_init_std = self.frequency_init_std, name = 'gated_fan_projection'))
            self.model.add(keras.layers.Dropout(self.input_projection_dropout))

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

            self.model.add(keras.layers.LSTM(units = int(n_features * model_relative_width),
                                return_sequences = return_sequences,
                                activation = self.activation))

            self.model.add(keras.layers.Dropout(model_dropout))

        # output layer
        # output layer: one unit per target column and per step ahead (see output_column_names)
        n_outputs = self.Y_train.shape[1] * self.steps_ahead

        if(self.LSTM_type == 'classificator'):
            print(f'Output layer for classification: {n_outputs} neurons and activation = {self.last_layer_activation}')
            self.model.add(keras.layers.Dense(n_outputs,
                                              activation = self.last_layer_activation))

        elif(self.LSTM_type == 'regressor'):
            # linear output: last_layer_activation is not used by regressors
            print(f'Output layer for regression: {n_outputs} neurons and linear activation')
            self.model.add(keras.layers.Dense(n_outputs, activation = 'linear'))

        if((self.class_weight is not None) and (n_outputs > 1)):
            warnings.warn('With more than one output Keras applies class_weight to the argmax of each target row '
                          '(one-hot assumption), which is not meaningful for independent binary outputs.',
                          UserWarning,
                          stacklevel = 2)

        # compile
        self.model.compile(optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss = self.loss_function,
                           metrics = self.compile_metrics())

        # report: model.summary() only prints, so its lines are collected to keep a copy in model_summary
        summary_lines = []
        self.model.summary(print_fn = lambda line, *args, **kwargs: summary_lines.append(line))
        self.model_summary = '\n'.join(summary_lines)

        print(self.model_summary)


    def make_multi_step_targets(self, Y_values):
        """
        Stack each target row with the following ``steps_ahead - 1`` rows.

        Row ``t`` of the result is ``Y[t], Y[t + 1], ..., Y[t + steps_ahead - 1]``
        (step-major: all target columns of step 1, then all of step 2, ...).
        The last ``steps_ahead - 1`` rows have no complete future and are
        filled with NaN; the generators never use them.

        Parameters
        ----------
        Y_values : array-like
            Targets, shape ``(n_rows, n_targets)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_targets * steps_ahead)``; ``Y_values`` itself
            (as an array) when ``steps_ahead == 1``.

        Examples
        --------
        >>> lstm.steps_ahead = 3
        >>> lstm.make_multi_step_targets(np.array([[1.], [2.], [3.], [4.]]))
        array([[ 1.,  2.,  3.],
               [ 2.,  3.,  4.],
               [ 3.,  4., nan],
               [ 4., nan, nan]])
        """
        Y_values = np.asarray(Y_values, dtype = float)

        if(self.steps_ahead == 1):
            return Y_values

        n_rows, n_targets = Y_values.shape
        Y_multi = np.full((n_rows, n_targets * self.steps_ahead), np.nan)

        for step in range(self.steps_ahead):
            # block `step` holds the targets `step` rows ahead
            Y_multi[:n_rows - step, step * n_targets:(step + 1) * n_targets] = Y_values[step:]

        return Y_multi


    def output_column_names(self):
        """
        Names of the network outputs, in the order of the prediction columns.

        Returns
        -------
        list of str
            The target column names when ``steps_ahead == 1``; otherwise
            ``'<target>_step_<k>'`` for k = 1 ... ``steps_ahead``, step-major
            (all targets of step 1, then all targets of step 2, ...). Step 1
            is the target row right after the input window (see
            :meth:`create_generators`).

        Examples
        --------
        >>> lstm.steps_ahead = 2
        >>> lstm.output_column_names()
        ['up_step_1', 'up_step_2']
        """
        if(self.Y_data is not None):
            target_names = list(self.Y_data.columns)
        else:
            target_names = list(getattr(self, 'Y_feature_names', None) or range(self.Y_train.shape[1]))

        if(self.steps_ahead == 1):
            return [str(name) for name in target_names]

        return [f'{name}_step_{step + 1}' for step in range(self.steps_ahead) for name in target_names]


    def create_generators(self, batch_size = None):
        """
        Create the training and validation sample generators.

        The generators (:func:`make_sequence_generator`) pair target row ``t`` with the feature rows
        ``t - timesteps ... t - 1``; with ``steps_ahead > 1`` the target of
        the sample is ``Y[t], ..., Y[t + steps_ahead - 1]`` (see
        :meth:`make_multi_step_targets`). Each generator yields
        ``len(data) - timesteps - steps_ahead + 1`` samples of shape
        ``(timesteps, n_features)``.

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

        multiple = self.effective_batch_multiple()
        if(self.batch_size % multiple != 0):
            raise ValueError(f'batch_size {self.batch_size} must be a multiple of {multiple} (devices of the data-parallel distribution).')

        if(getattr(self, 'sequence_groups', None) is not None):
            # windows and targets inside each group (e.g. option contract), never across two groups
            self.generator = make_grouped_sequence_generator(self.keras, self.X_train_s, self.Y_train_s,
                                                             length = self.timesteps, batch_size = self.batch_size,
                                                             groups = self.groups_train, steps_ahead = self.steps_ahead,
                                                             shuffle = self.shuffle,
                                                             sample_weights = getattr(self, 'sample_weight_train', None),
                                                             batch_multiple = multiple)
            self.validation_generator = make_grouped_sequence_generator(self.keras, self.X_test_s, self.Y_test_s,
                                                                        length = self.timesteps, batch_size = self.batch_size,
                                                                        groups = self.groups_test, steps_ahead = self.steps_ahead,
                                                                        batch_multiple = multiple)
            return

        # the last target row usable is the one that still has steps_ahead - 1 rows after it
        self.generator = make_sequence_generator(self.keras,
                                                 self.X_train_s,
                                                 self.make_multi_step_targets(self.Y_train_s),
                                                 length = self.timesteps,
                                                 end_index = len(self.X_train_s) - self.steps_ahead,
                                                 batch_size = self.batch_size,
                                                 shuffle = self.shuffle,
                                                 sample_weights = getattr(self, 'sample_weight_train', None),
                                                 batch_multiple = multiple)
        self.validation_generator = make_sequence_generator(self.keras,
                                                            self.X_test_s,
                                                            self.make_multi_step_targets(self.Y_test_s),
                                                            length = self.timesteps,
                                                            end_index = len(self.X_test_s) - self.steps_ahead,
                                                            batch_size = self.batch_size,
                                                            batch_multiple = multiple)


    def effective_batch_multiple(self):
        """
        Multiple that every batch size must have (see ``batch_multiple``).

        Returns
        -------
        int
            ``batch_multiple`` when set; otherwise the number of devices of
            the active Keras distribution (``keras.distribution``), or 1.
        """
        if(getattr(self, 'batch_multiple', None)):
            return int(self.batch_multiple)
        try:
            distribution = self.keras.distribution.distribution()
        except Exception:
            distribution = None
        mesh = getattr(distribution, 'device_mesh', None)
        if(mesh is None):
            return 1
        return max(int(np.size(mesh.devices)), 1)


    def predict_validation(self):
        """
        Predictions on the validation samples, aligned with :meth:`validation_targets`.

        Same as ``model.predict(validation_generator)``, cut to the real
        samples: with ``batch_multiple > 1`` the last batch is completed with
        repeated samples, whose predictions are dropped here.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_outputs)``.
        """
        if(self.validation_generator is None):
            self.create_generators()
        predictions = self.model.predict(self.validation_generator, verbose = 0)
        return predictions[:self.validation_generator.n_samples]


    def auc_callbacks(self):
        """
        The validation AUC monitor (see ``monitor_auc``), as a list ready for ``fit``.

        Returns
        -------
        list of keras.callbacks.Callback
            Empty when ``monitor_auc`` is off; otherwise the callback built by
            :func:`make_auc_callback` on the validation generator, restricted
            to the samples whose first target row is selected by
            ``monitor_auc_rows``.
        """
        if(not getattr(self, 'monitor_auc', False)):
            return []

        rows = np.asarray(self.validation_generator.sample_target_rows)
        first_rows = rows if rows.ndim == 1 else rows[:, 0]
        mask_test = getattr(self, 'monitor_auc_rows_test', None)
        mask = None if mask_test is None else mask_test[first_rows]

        return [make_auc_callback(self.keras, self.predict_validation,
                                  self.validation_targets(), mask)]


    def network_training(self, epochs, batch_size = None, timesteps = None, callbacks = None):
        """
        Train the network and save every artefact of the run.

        Steps:

        1. a timestamp identifies the run and all the file names;
        2. ``X_data`` / ``Y_data`` are saved as CSV (if ``save_X_Y_data``);
        3. hyperparameters (JSON) and scalers (``.pkl``) are saved;
        4. the network is trained on the generators with early stopping and
           checkpointing (``class_weight`` is applied if set);
        5. the training history is saved (CSV), the saved checkpoint is
           reloaded into ``self.model`` (the best epoch when
           ``save_best_only = True``, the last one otherwise) and the history
           is plotted.

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
            Must equal the input length of the built network; to use a
            different value rebuild it first with
            ``network_structure_set_compile(timesteps)``.
        callbacks : list of keras.callbacks.Callback, optional
            Extra callbacks run together with the early stopping and the
            best-epoch checkpoint (e.g. a time limit or a learning-rate
            schedule).

        Returns
        -------
        None
            The trained model (best epoch if ``save_best_only = True``, last
            epoch otherwise) is in ``self.model``, the history in
            ``self.loss_df``.

        Raises
        ------
        ValueError
            If the network has not been built, if ``timesteps`` differs from
            its input length, or if the training or test set has no more than
            ``timesteps`` rows. Checks are done before any file is written.

        Examples
        --------
        >>> lstm.network_structure_set_compile()
        >>> lstm.network_training(epochs = 300, batch_size = 64)
        """
        new_timesteps = self.timesteps if timesteps is None else timesteps

        # checks are done before changing anything, so that a rejected call leaves the object as it was
        if(len(self.model.layers) == 0):
            raise ValueError('The network is not built: call network_structure_set_compile() before training.')

        # the generators produce sequences of `timesteps` rows: they must match the network input
        if(self.model.input_shape[1] != new_timesteps):
            raise ValueError(f'The network expects sequences of {self.model.input_shape[1]} timesteps but timesteps = {new_timesteps}: '
                             f'call network_structure_set_compile({new_timesteps}) before training.')

        self.timesteps = new_timesteps

        if(batch_size is not None):
            print(f'Batch size is not none, equal to {batch_size}')
            self.batch_size = batch_size
        else:
            print(f'Batch size is none, keep default or previous value {self.batch_size}')

        # training and validation samples, built before writing any file: the generators raise here
        # if a set has no more than `timesteps` rows
        self.create_generators()

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

        # model training (class_weight = None means no reweighting)
        if(self.class_weight is not None):
            print(f'Used class_weight: {self.class_weight}')

        # the AUC monitor comes first: it writes val_monitored_auc into the logs read by early stopping and checkpoint
        history = self.model.fit(self.generator,
                                 epochs = epochs,
                                 validation_data = self.validation_generator,
                                 class_weight = self.class_weight,
                                 callbacks = self.auc_callbacks() + [self.early_stop, self.model_checkpoint] + list(callbacks or []))

        # save history
        self.loss_df = pd.DataFrame(history.history)
        self.loss_df.to_csv(self.data_storage_path + training_history_file_name)

        # keep the model saved by the checkpoint (best epoch when save_best_only)
        self.load_model(model_file_name)

        # plot history
        self.plot_training_history()


    def save_hyperparameters(self, file_name):
        """
        Save ``self.hyperparameters`` as JSON in ``data_storage_path``.

        NumPy values are converted to plain Python values; anything else that
        JSON cannot represent is saved as its string.

        Parameters
        ----------
        file_name : str
            Name of the JSON file (without the folder).

        Returns
        -------
        None
        """
        # serialised before opening the file, so that an error cannot leave a truncated JSON;
        # numpy values (e.g. widths computed with numpy) are converted to plain Python values
        text = json.dumps(self.hyperparameters, default = lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))

        with open(self.data_storage_path + file_name, "w") as file:
            file.write(text)

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


    @in_strategy_scope
    def load_model(self, model_file_name = None, file_path_name = None):
        """
        Load a saved Keras model into ``self.model``.

        The model can have been trained with either backend. It is loaded
        without its saved compile state and recompiled with the current
        ``loss``, ``metrics`` and ``learning_rate`` (restored from the
        hyperparameters by :meth:`load_all`), so a further training starts
        with a fresh optimizer state.

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
        # compile = False: the saved compile configuration can refer to backend-specific classes (e.g. the PyTorch
        # Adam optimizer) that cannot be loaded with the other backend. Architecture and weights are portable, so the
        # model is loaded without it and recompiled with the current loss, metrics and learning rate.
        self.model = self.keras.models.load_model(model_file_path, compile = False, custom_objects = fan_layers(self.keras))
        self.model.compile(optimizer = self.keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss = self.loss_function,
                           metrics = self.compile_metrics())
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
            For backward compatibility, the full path of an existing CSV
            (old signature ``load_training_history(file_path_name)``) is
            still accepted, with a ``DeprecationWarning``.

        Returns
        -------
        None
        """
        # older versions took the full path of the CSV as their only argument (file_path_name)
        # (only one argument given, pointing to an existing file that is not inside data_storage_path)
        legacy_path = file_path_name if training_history_file_name is None else training_history_file_name
        if(((training_history_file_name is None) != (file_path_name is None)) and os.path.isfile(legacy_path)
           and not os.path.isfile(self.data_storage_path + legacy_path)):
            warnings.warn("load_training_history(<full path>) is deprecated, use "
                          "load_training_history(training_history_file_name, file_path_name = <folder>) instead.",
                          DeprecationWarning,
                          stacklevel = 2)
            self.loss_df = pd.read_csv(legacy_path, index_col = 0)
            return

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


    def validation_targets(self):
        """
        Actual (unscaled) targets of the validation samples, aligned with ``model.predict(validation_generator)``.

        Without ``sequence_groups`` the generator skips the first
        ``timesteps`` test rows and the last ``steps_ahead - 1``; with groups
        the samples are the ones of :func:`make_grouped_sequence_generator`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, steps_ahead * n_targets)``, step-major (same
            order as the network outputs, see :meth:`output_column_names`).

        Examples
        --------
        >>> Y_true = lstm.validation_targets()
        >>> Y_pred = lstm.model.predict(lstm.validation_generator)
        """
        if(self.validation_generator is None):
            self.create_generators()
        rows = np.asarray(self.validation_generator.sample_target_rows)
        Y_values = self.Y_test.values
        if(rows.ndim == 1):
            # first target row of each sample: steps 1 ... steps_ahead are the following rows
            rows = rows[:, None] + np.arange(self.steps_ahead)[None, :]
        return Y_values[rows].reshape(len(rows), -1)


    def network_predictions_evaluation(self, min_probability, output_dict = False):
        """
        Evaluate a binary classificator on the test set.

        Probabilities predicted on the validation generator are turned into
        0/1 with the threshold ``min_probability`` and compared with the
        actual targets through sklearn's ``classification_report`` (printed
        for every output, i.e. every target column and, with
        ``steps_ahead > 1``, every step: see :meth:`output_column_names`).
        The first ``timesteps`` test rows (and the last ``steps_ahead - 1``)
        have no prediction and are excluded.

        Parameters
        ----------
        min_probability : float
            Threshold (0 - 1): predictions strictly greater than it become 1.
        output_dict : bool, default False
            If ``True`` the report is also returned as a dictionary.

        Returns
        -------
        filtered_predictions_results_df : pandas.DataFrame
            0/1 predictions, one column per output (named as
            :meth:`output_column_names`).
        predictions_df : pandas.DataFrame
            Raw predicted probabilities (columns numbered from 0).
        report : dict
            Only when ``output_dict = True``: classification report of the
            LAST output (last target column, last step).

        Examples
        --------
        >>> results_df, probabilities_df = lstm.network_predictions_evaluation(0.6)
        >>> results_df, probabilities_df, report = lstm.network_predictions_evaluation(0.6, output_dict = True)
        >>> report['1']['precision']
        """
        # rebuilt every time (it is cheap) so that it always reflects the current split, timesteps and batch_size
        self.create_generators()

        # Cut off predictions with low probability
        predictions = self.predict_validation()
        predictions_df = pd.DataFrame(predictions.reshape(predictions.shape[0], -1))
        filtered_predictions_results_df = pd.DataFrame()

        Y_test = pd.DataFrame(self.validation_targets(), columns = self.output_column_names())

        print(f'len predictions {len(predictions_df)}')
        print(f'len Y_test (without the first {self.timesteps} rows) {len(Y_test)}')

        report = None
        for count, col_name in enumerate(Y_test.columns):

            filtered_predictions_results_df[col_name] = predictions_df[count].apply(lambda x: 1 if x > min_probability else 0 ).values
            report = classification_report(Y_test[col_name].astype(int), filtered_predictions_results_df[col_name], output_dict = output_dict)

            print(f'\n{col_name}')
            print(report)

        if(output_dict == False):
            return filtered_predictions_results_df, predictions_df

        elif(output_dict == True):
            return filtered_predictions_results_df, predictions_df, report


    def binary_network_predictions_evaluation(self, min_probability, output_dict = False):
        """
        Deprecated alias of :meth:`network_predictions_evaluation`.

        Kept for backward compatibility with the return values of the old
        method. New code should use :meth:`network_predictions_evaluation`,
        which returns ``predictions_df`` too.

        Parameters
        ----------
        min_probability : float
            Threshold (0 - 1): predictions strictly greater than it become 1.
        output_dict : bool, default False
            If ``True`` the report is also returned as a dictionary.

        Returns
        -------
        tuple
            ``(filtered_predictions_results_df, report)`` when
            ``output_dict = True`` (old contract);
            ``(filtered_predictions_results_df, predictions_df)`` otherwise
            (the old method returned the ``classification_report`` function
            by mistake).
        """
        warnings.warn("binary_network_predictions_evaluation is deprecated, use network_predictions_evaluation instead.",
                      DeprecationWarning,
                      stacklevel = 2)

        results = self.network_predictions_evaluation(min_probability, output_dict = output_dict)

        if(output_dict == True):
            # old contract: (filtered_predictions_results_df, report)
            return results[0], results[2]

        return results


    def plot_training_history(self):
        """
        Plot the training history (``self.loss_df``) with pandas/matplotlib.

        The plotted columns are ``history_metrics``; when it is ``None``,
        ``['loss', 'val_loss']`` for regressors and the first of ``metrics``
        with its validation counterpart (e.g. ``['accuracy',
        'val_accuracy']``) for classificators. Default columns missing from
        the history are skipped.

        Returns
        -------
        None
        """
        history_metrics = self.history_metrics

        if(history_metrics is None):
            if((self.LSTM_type == 'regressor') or (len(self.metrics) == 0) or not isinstance(self.metrics[0], str)):
                history_metrics = ['loss', 'val_loss']
            else:
                history_metrics = [self.metrics[0], 'val_' + self.metrics[0]]

            # Keras may log a metric under a different name: keep only the columns that exist
            history_metrics = [column for column in history_metrics if column in self.loss_df.columns] or ['loss', 'val_loss']

        self.loss_df[history_metrics].plot()


    def create_sequences(self, data, window_size):
        """
        Split a time series into overlapping windows (sliding window, step 1).

        Utility not used internally (training uses :meth:`create_generators`).

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
            ``Y_train_s``, ``Y_test_s`` (numpy arrays). The generators are
            reset (rebuilt by the methods that need them).

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

        groups = getattr(self, 'sequence_groups', None)
        if(groups is not None):
            if(len(groups) != len(self.X_data)):
                raise ValueError(f'sequence_groups has {len(groups)} labels but X_data has {len(self.X_data)} rows.')
            self.groups_train = groups[:train_size]
            self.groups_test = groups[train_size:]

        for name, values in [('sample_weight', getattr(self, 'sample_weight', None)), ('monitor_auc_rows', getattr(self, 'monitor_auc_rows', None))]:
            if((values is not None) and (len(values) != len(self.X_data))):
                raise ValueError(f'{name} has {len(values)} values but X_data has {len(self.X_data)} rows.')
        self.sample_weight_train = None if getattr(self, 'sample_weight', None) is None else self.sample_weight[:train_size]
        self.monitor_auc_rows_test = None if getattr(self, 'monitor_auc_rows', None) is None else self.monitor_auc_rows[train_size:]

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

            # a targets scaler created here (scale_targets switched on later) has never been fitted
            fit_Y_scaler = scaler_fit
            if(self.Y_scaler is None):
                self.Y_scaler = self.new_scaler()
                fit_Y_scaler = True

            if(fit_Y_scaler == True):
                self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)
            else:
                self.Y_train_s = self.Y_scaler.transform(self.Y_train)

            self.Y_test_s = self.Y_scaler.transform(self.Y_test)

        else:
            print('Target not scaled')
            # numpy arrays (not DataFrames): the generators index the targets by row position
            self.Y_train_s = self.Y_train.values
            self.Y_test_s = self.Y_test.values

        # generators built on the previous split are no longer valid
        self.generator = None
        self.validation_generator = None

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
        target row ``current_datetime_idx + 1`` (and, with
        ``steps_ahead = k``, to rows ``current_datetime_idx + 1 ...
        current_datetime_idx + k``).

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
            Predictions of shape ``(n_samples, n_targets * steps_ahead)``;
            column names and order are given by :meth:`output_column_names`.

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
            # the scaler works on n_targets columns: with steps_ahead > 1 each step block is descaled separately
            n_samples = predictions.shape[0]
            n_targets = predictions.shape[1] // self.steps_ahead
            predictions = self.Y_scaler.inverse_transform(predictions.reshape(-1, n_targets)).reshape(n_samples, -1)

        return predictions


    def compute_gradients(self, inputs, targets):
        """
        Gradient of the mean squared error with respect to the network inputs.

        Works with both backends (TensorFlow ``GradientTape`` or PyTorch
        autograd).

        Parameters
        ----------
        inputs : numpy.ndarray
            Input sequences, shape ``(n_samples, timesteps, n_features)``.
        targets : numpy.ndarray
            Targets, shape ``(n_samples, n_outputs)``.

        Returns
        -------
        numpy.ndarray
            Gradients with the same shape as ``inputs``.
        """
        keras = self.keras

        # MSE is used for every network type (as in fastANN): only the gradient magnitude matters here
        mse = keras.losses.MeanSquaredError()
        targets = keras.ops.convert_to_tensor(np.asarray(targets, dtype = np.float32))

        if(self.backend == 'jax'):
            # JAX: gradient of a pure function of the inputs (the model weights are constants here)
            import jax
            inputs = jax.numpy.asarray(np.asarray(inputs, dtype = np.float32))
            return np.asarray(jax.grad(lambda x: mse(targets, self.model(x)))(inputs))

        if(self.backend == 'torch'):
            # PyTorch autograd: the inputs become a leaf tensor that records its gradient
            inputs = keras.ops.convert_to_tensor(np.asarray(inputs, dtype = np.float32))
            inputs.requires_grad_(True)
            loss = mse(targets, self.model(inputs))
            loss.backward()
            return inputs.grad.detach().cpu().numpy()

        import tensorflow
        inputs = tensorflow.convert_to_tensor(np.asarray(inputs, dtype = np.float32))
        with tensorflow.GradientTape() as tape:
            tape.watch(inputs)
            loss = mse(targets, self.model(inputs))

        return tape.gradient(loss, inputs).numpy()


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

        # rebuilt every time (it is cheap) so that it always reflects the current split, timesteps and batch_size
        self.create_generators()

        # all the test sequences and their targets, exactly as seen during validation
        batches = [self.validation_generator[i] for i in range(len(self.validation_generator))]
        X_test_sequences = np.concatenate([batch[0] for batch in batches])
        Y_test_sequences = np.concatenate([batch[1] for batch in batches])

        gradients = self.compute_gradients(X_test_sequences, Y_test_sequences)

        # average over samples and timesteps: one value per feature
        feature_importance = np.mean(np.abs(gradients), axis=(0, 1))

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
