
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
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.integrate import simps
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean


# Reporting
import plotly
from plotly.figure_factory import create_candlestick
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
from IPython.display import HTML

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

class fastLSTM:

    def __init__(self,
                 X_data,
                 Y_data,
                 scaler = 'StandardScaler', # MinMaxScaler
                 model_relative_width = [1],
                 model_dropout = [0],
                 LSTM_type = 'classificator',
                 learning_rate = 0.0003,
                 activation = 'tanh', #'relu',
                 last_layer_activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 metric = 'accuracy',
                 check_point_metric = 'accuracy',
                 metric_mode = 'max',
                 early_stop_condittion = 'val_accuracy',
                 batch_size = 128,
                 timesteps = 1,
                 scale_target = False,
                 steps_ahead = 1,
                 class_weight = None,
                 save_best_only = True,
                 early_stop_patience = 200,
                 train_size_rate = 0.7,
                 data_storage_path="\\cyPredict\\",
                 model_name = 'LSTM'):
        # ... (inizializzazioni esistenti)
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
        self.loss = loss
        self.metric = metric
        self.check_point_metric = check_point_metric
        self.metric_mode = metric_mode
        self.early_stop_condittion = early_stop_condittion
        self.early_stop_patience = early_stop_patience
        self.train_size_rate = train_size_rate
        self.timesteps = timesteps
        self.scale_target = scale_target
        self.steps_ahead = steps_ahead
        self.batch_size = batch_size
        self.class_weight = class_weight
        
        self.loss_df = pd.DataFrame()
        self.model_summary = {}
        
        self.X_data = X_data
        self.Y_data = Y_data
        
        self.scaler = scaler
        
        self.X_train_s = pd.DataFrame()
        self.Y_train_s = pd.DataFrame()
        self.X_test_s = pd.DataFrame()
        self.Y_test_s = pd.DataFrame()

        if(self.scaler == 'MinMaxScaler'):
            print('MinMaxScaler selected')
            self.X_scaler = MinMaxScaler() 
            self.Y_scaler = MinMaxScaler()         
        else:
            print('StandardScaler selected')
            self.X_scaler = StandardScaler() 
            self.Y_scaler = StandardScaler() 
            
        self.set_loss_function(loss)
        
#         self.checkpoint_callback(self.save_best_only)
        self.early_stop_patience_set(self.early_stop_condittion, self.metric_mode, self.early_stop_patience)
        self.split_and_scale()
        
    def init_hyperparameters(self,
                             model_training_datetime = None,
                             model_file_name = None,
                             scaler_file_name = None,
                             training_history_file_name = None,
                             X_data_df_file_name = None,
                             Y_data_df_file_name = None):
        self.hyperparameters = {
            'model_training_datetime': model_training_datetime,
            'model_name': self.model_name,
            'model_relative_width': self.model_relative_width,
            'model_dropout': self.model_dropout,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'last_layer_activation': self.last_layer_activation,
            'loss': self.loss if isinstance(self.loss, str) else str(self.loss),
            'metric': self.metric,
            'check_point_metric': self.check_point_metric,
            'metric_mode': self.metric_mode,
            'early_stop_condittion': self.early_stop_condittion,
            'early_stop_patience': self.early_stop_patience,
            'train_size_rate': self.train_size_rate,
            'timesteps': self.timesteps,
            'scale_target': self.scale_target,
            'steps_ahead': self.steps_ahead,
            'batch_size': self.batch_size,
            'scaler_type': 'StandardScaler' if self.scaler == 'StandardScaler' else 'MinMaxScaler',
            'data_storage_path': self.data_storage_path,
            'model_file_name': model_file_name,
            'scaler_file_name': scaler_file_name,
            'training_history_file_name': training_history_file_name,
            'X_data_df_file_name': X_data_df_file_name,
            'Y_data_df_file_name': Y_data_df_file_name
        }
        

    def save_hyperparameters(self, file_name):
        with open(self.data_storage_path + file_name, "w") as file:
            json.dump(self.hyperparameters, file)
        print("Hyperparameters saved in " + self.data_storage_path + file_name)
        

    def load_all(self, hyperparameters_file_name = None, file_path_name = None):
        if file_path_name is not None:
            self.data_storage_path = file_path_name
        if hyperparameters_file_name is not None:
            self.hyperparameters_file_name = hyperparameters_file_name
        print("Loading hyperparameters from " + self.data_storage_path + self.hyperparameters_file_name)
        with open(self.data_storage_path + self.hyperparameters_file_name, "r") as file:
            self.hyperparameters = json.load(file)
        print("Hyperparameters loaded.")
        
        # Aggiorno gli attributi in base agli iperparametri caricati
        self.model_name = self.hyperparameters['model_name']
        self.model_relative_width = self.hyperparameters['model_relative_width']
        self.model_dropout = self.hyperparameters['model_dropout']
        self.learning_rate = self.hyperparameters['learning_rate']
        self.activation = self.hyperparameters['activation']
        self.last_layer_activation = self.hyperparameters['last_layer_activation']
        self.loss = self.hyperparameters['loss']
        self.metric = self.hyperparameters['metric']
        self.check_point_metric = self.hyperparameters['check_point_metric']
        self.metric_mode = self.hyperparameters['metric_mode']
        self.early_stop_condittion = self.hyperparameters['early_stop_condittion']
        self.early_stop_patience = self.hyperparameters['early_stop_patience']
        self.train_size_rate = self.hyperparameters['train_size_rate']
        self.timesteps = self.hyperparameters['timesteps']
        self.scale_target = self.hyperparameters['scale_target']
        self.steps_ahead = self.hyperparameters['steps_ahead']
        self.batch_size = self.hyperparameters['batch_size']
        
        # Caricamento del modello
        print("Loading model from " + self.data_storage_path + self.hyperparameters['model_file_name'])
        self.model = load_model(self.data_storage_path + self.hyperparameters['model_file_name'])

        # Caricamento dello scaler (sia X_scaler che Y_scaler)
        scaler_dict = joblib.load(self.data_storage_path + self.hyperparameters['scaler_file_name'])
        self.X_scaler = scaler_dict["X_scaler"]
        self.Y_scaler = scaler_dict["Y_scaler"]

        # Caricamento della cronologia di training
        self.loss_df = pd.read_csv(self.data_storage_path + self.hyperparameters['training_history_file_name'])

        # Caricamento dei dati X e Y (se salvati)
        if self.hyperparameters['X_data_df_file_name'] is not None and self.hyperparameters['Y_data_df_file_name'] is not None:
            self.X_data = pd.read_csv(self.data_storage_path + self.hyperparameters['X_data_df_file_name'])
            self.Y_data = pd.read_csv(self.data_storage_path + self.hyperparameters['Y_data_df_file_name'])
            self.split_and_scale()
        print("Load all completed.")
            
            
    def set_loss_function(self, loss):
        if loss == 'CategoricalCrossentropy':
            self.loss = CategoricalCrossentropy()
        elif loss == 'SparseCategoricalCrossentropy':
            self.loss = SparseCategoricalCrossentropy()
        elif loss == 'mse':
            self.loss = 'mse'  # oppure tensorflow.keras.losses.MeanSquaredError()
        else:
            self.loss = BinaryCrossentropy()

            
        
    def early_stop_patience_set(self, metric, metric_mode, patience):
        
        self.early_stop = EarlyStopping(monitor = metric, mode = metric_mode, verbose = 1, patience = patience)
            
         
    def checkpoint_callback(self, save_best_only):
        
        # Se gli iperparametri sono stati inizializzati e contengono il nome del modello, usalo
        if hasattr(self, 'hyperparameters') and 'model_file_name' in self.hyperparameters and self.hyperparameters['model_file_name']:
            filename = self.hyperparameters['model_file_name']
            
        else:
            # Altrimenti, costruisci il filename includendo il timestamp
            filename = self.model_training_datetime + ' - LSTM MODEL - ' + self.model_name + '.keras'
        
        self.checkpoint_callback = ModelCheckpoint(
                                                    self.data_storage_path + filename, 
                                                    monitor=self.check_point_metric, 
                                                    mode=self.metric_mode, 
                                                    verbose=1, 
                                                    save_best_only=save_best_only
                                                  )
        
        
    def network_structure_set_compile(self, timesteps = None):
        
        if(timesteps is not None):
            self.timesteps = timesteps
        
        # reset
        self.model = Sequential()
        
        
        # Aggiungi il primo livello LSTM
#         self.model.add(LSTM(units=int(self.X_train_s.shape[1] * self.model_relative_width[0]), 
#                        return_sequences=True, 
#                        input_shape=(self.timesteps, self.X_train_s.shape[1]),
#                        activation = self.activation))

        self.model.add(Input(shape=(self.timesteps, self.X_train_s.shape[1])))
        self.model.add(LSTM(
            units=int(self.X_train_s.shape[1] * self.model_relative_width[0]),
            return_sequences=True,
            activation=self.activation))
        
        self.model.add(Dropout(self.model_dropout[0]))

        
        # inner layers
        for i in range(1, len(self.model_relative_width)):
            
            print(f"added layer width: {self.model_relative_width[i]}")
            model_relative_width = self.model_relative_width[i]
            model_dropout = self.model_dropout[i]
            
            print(f'i = {i}')
            print(f'len(self.model_relative_width) = {len(self.model_relative_width)}')
            
            if(i == len(self.model_relative_width) - 1):
                return_sequences = False
            else:
                return_sequences = True
                
            self.model.add(LSTM(units=int(self.X_train_s.shape[1] * model_relative_width), 
                                return_sequences = return_sequences, 
                                activation = self.activation))
    
            self.model.add(Dropout(model_dropout))
            
            
            
        if(self.LSTM_type == 'classificator'):
            print(f'Last layer for classification {self.Y_train.shape[1],} neurons and activation = {self.last_layer_activation}')
            print(self.Y_train.shape[1])
            self.model.add(Dense(self.Y_train.shape[1], 
                                 activation = self.last_layer_activation)) 
            
        elif(self.LSTM_type == 'regressor'):
            self.model.add(Dense(self.Y_train.shape[1] * self.steps_ahead, activation='linear')) 

        # compile
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate),                      
                         loss = self.loss, 
                         metrics = [self.metric])        
            
        # report
        self.model_summary = self.model.summary()
        
        display(self.model_summary)


        
        
    def network_training(self, epochs, batch_size = None, timesteps = None):
        # (Modifica: aggiornamento dei nomi file e salvataggio dei dati, iperparametri e scaler, replicando fastANN)
        
        if(timesteps is not None):
            self.timesteps = timesteps
        
        if(batch_size is not None):            
            print(f'Batch size is not none, equal to {batch_size}')
            self.batch_size = batch_size
        else:
            print(f'Batch size is none, keep default or previous value {self.batch_size}')
        
        # Impostazione di un timestamp per identificare la sessione di training
        
        self.model_training_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        
        # Creazione dei nomi dei file (modello, scaler, cronologia training, iperparametri)
        model_file_name = self.model_training_datetime + ' - LSTM MODEL - ' + self.model_name + '.keras'
        scaler_file_name = self.model_training_datetime + ' - SCALER FOR LSTM MODEL - ' + self.model_name + '.pkl'
        training_history_file_name = self.model_training_datetime + ' - TRAINING HISTORY OF LSTM MODEL - ' + self.model_name + '.csv'
        hyperparameters_file_name = self.model_training_datetime + ' - HYPERPARAMETERS OF LSTM MODEL - ' + self.model_name + '.json'
        self.hyperparameters_file_name = hyperparameters_file_name
        
        # Salvataggio dei dati X e Y (se disponibili) – replicando fastANN
        if(self.X_data is not None and self.Y_data is not None):
            X_data_df_file_name = self.model_training_datetime + ' - X_data FOR LSTM MODEL - ' + self.model_name + '.csv'
            Y_data_df_file_name = self.model_training_datetime + ' - Y_data FOR LSTM MODEL - ' + self.model_name + '.csv'
            self.X_data.to_csv(self.data_storage_path + X_data_df_file_name, index=False)
            self.Y_data.to_csv(self.data_storage_path + Y_data_df_file_name, index=False)
            print('\nX and Y data saved.')
        else:
            X_data_df_file_name = None
            Y_data_df_file_name = None
            print('\nX and Y data not saved.')
            
        # Inizializzazione e salvataggio degli iperparametri
        print('\nInit hyperparameters.')
        self.init_hyperparameters(
            model_training_datetime = self.model_training_datetime,
            model_file_name = model_file_name,
            scaler_file_name = scaler_file_name,
            training_history_file_name = training_history_file_name,
            X_data_df_file_name = X_data_df_file_name,
            Y_data_df_file_name = Y_data_df_file_name
        )
        print('\nSave hyperparameters.')
        self.save_hyperparameters(hyperparameters_file_name)
        
        # Salvataggio dello scaler (salvo sia X_scaler che Y_scaler in un unico oggetto)
        joblib.dump({"X_scaler": self.X_scaler, "Y_scaler": self.Y_scaler}, self.data_storage_path + scaler_file_name)
        
        self.checkpoint_callback(self.save_best_only)
        
        # Creazione dei generatori per il training
        self.generator = TimeseriesGenerator(self.X_train_s, self.Y_train_s, length=self.timesteps, batch_size=self.batch_size)
        self.validation_generator = TimeseriesGenerator(self.X_test_s, self.Y_test_s, length=self.timesteps, batch_size=self.batch_size)
        
        # Training del modello (gestione opzionale di class_weight)
        if(self.class_weight is not None):
            print(f'Used class_weight: {self.class_weight}')
            self.model.fit(self.generator,
                           epochs = epochs,
                           validation_data = self.validation_generator,
                           class_weight = self.class_weight,
                           callbacks=[self.early_stop, self.checkpoint_callback])
        else:
            self.model.fit(self.generator,
                           epochs = epochs,
                           validation_data = self.validation_generator,
                           callbacks=[self.early_stop, self.checkpoint_callback])
        
        # Salvataggio della cronologia di training
        self.loss_df = pd.DataFrame(self.model.history.history)
        self.loss_df.to_csv(self.data_storage_path + training_history_file_name)
                
        # Caricamento del modello migliore salvato tramite checkpoint
        self.model = load_model(self.data_storage_path + model_file_name)
        
        # Plot della cronologia di training
        self.plot_training_history()


            
    def load_training_history(self, file_path_name = None):
        
        if(file_path_name is not None):
            self.loss_df = pd.read_csv(file_path_name)
            
        else:
            self.loss_df = pd.read_csv(self.data_storage_path + 'LSTM MODEL - ' + self.model_name + ' - TRAINING HISTORY.csv')
        
        
    def binary_network_predictions_evaluation(self, min_probability, output_dict = False):        

        # Cut off predictions with low probability
        predictions = self.model.predict(self.validation_generator)
        reshaped_predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
        predictions_df = pd.DataFrame(reshaped_predictions)
        
        # Cut off predictions with low probability
#         predictions = self.model.predict(self.X_test_s)
#         predictions_df = pd.DataFrame(predictions)
        filtered_predictions_results_df = pd.DataFrame()
    
        print(predictions_df.columns)
        print(predictions_df.shape)
        print(predictions_df.tail(5))
        print(self.Y_test.shape)        
        
        
        count = 0
        for col_name, col_data in self.Y_data.items():
            
            filtered_predictions_results_df[col_name] = predictions_df[count].apply(lambda x: 1 if x > min_probability else 0 ).values
            report  = classification_report(self.Y_test[col_name][self.timesteps:], filtered_predictions_results_df[col_name], output_dict = output_dict)
            
            print(classification_report)
            
            count += 1

        if(output_dict == False):
            return classification_report
        
        elif(output_dict == True):
            return filtered_predictions_results_df, report
        
        
    
    def plot_training_history(self):
        
        if(self.LSTM_type == 'classificator'):            
            self.loss_df[['accuracy', 'val_accuracy']].plot()
            
        elif(self.LSTM_type == 'regressor'): 
            self.loss_df[['loss', 'val_loss']].plot()
            
            
    def create_sequences(self, data, window_size):
        sequences = []
        
        print(data.shape)
        
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
            
        print(data.shape)

        # Iterate over indices based on the length of the data minus the window size
        for i in range(len(data) - window_size + 1):
            # Extract the sequence from the 'value' column of the DataFrame
            sequence = data[i:i + window_size].tolist()  # Convert the pandas series to a list
            # Append the sequence to the list of sequences
            sequences.append(sequence)

        return sequences

        
        
    def split_and_scale(self, Y_column_name = None):        
        

        train_size = int(len(self.X_data) * self.train_size_rate)
        test_size = len(self.X_data) - train_size

        self.X_train = self.X_data.head(train_size)
        self.Y_train = self.Y_data.head(train_size)
        self.X_test = self.X_data.tail(test_size)
        self.Y_test = self.Y_data.tail(test_size)
        
        
#         print(f"Shape of X_data: {self.X_data.shape}")
#         print(f"Shape of X_train: {self.X_train.shape}")
#         print(f"Shape of Y_train: {self.Y_train.shape}")
#         print(f"Shape of Y_test: {self.Y_test.shape}")
        
#         display(self.X_data.head(10))
#         display(self.X_train.head(10))

        self.X_train_s = self.X_scaler.fit_transform(self.X_train)
        self.X_test_s = self.X_scaler.transform(self.X_test)
        
        if(self.scale_target == True):
            print('Target scaled too')
            self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)
            self.Y_test_s = self.Y_scaler.transform(self.Y_test)
        
        else:
            print('Target not scaled')
            self.Y_train_s = self.Y_train.values
            self.Y_test_s = self.Y_test.values
            
        

    def binary_precision_recall_vs_scoring(self, n_points = 15, plot = True):
        
        # Inizializza le liste per memorizzare i valori di precision e recall
        precision_list = []
        recall_list = []
        cutoff_values = []

        # Ciclo for per calcolare i risultati per ogni valore di cutoff
        for cutoff in range(n_points, 100, 1):
            cutoff_value = cutoff / 100  # Calcola il valore di cutoff da 0 a 1
            print(f'Evaluting cutoof value = {cutoff_value}')
            
            filtered_predictions_results_df, dictionary = self.network_predictions_evaluation(cutoff_value, output_dict=True)

            # Aggiungi i valori di precision e recall alla lista
            precision_list.append(dictionary['1']['precision'])
            recall_list.append(dictionary['1']['recall'])
            cutoff_values.append(cutoff_value)

        # Crea un DataFrame pandas con i valori di precision, recall e cutoff
        df = pd.DataFrame({'Cutoff': cutoff_values, 'Precision': precision_list, 'Recall': recall_list})

        if(plot == True):

            # Plot di precision e recall in funzione di cutoff utilizzando Plotly
            fig = go.Figure()

            # Aggiungi linea per precision
            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Precision'], mode='lines', name='Precision'))

            # Aggiungi linea per recall
            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Recall'], mode='lines', name='Recall'))

            # Imposta i titoli degli assi e il titolo del grafico
            fig.update_layout(
                xaxis_title='Cutoff',
                yaxis_title='Value',
                title='Precision and Recall vs Cutoff'
            )

            # Mostra il grafico interattivo
            fig.show()
        
        return df       



    def prepare_input_sample(self, X, current_datetime_idx, apply_scaler=True):
        """
        Data una matrice X (array NumPy o DataFrame) e l'indice relativo al datetime corrente (current_datetime_idx),
        estrae la sequenza di lunghezza self.timesteps che termina alla riga current_datetime_idx.
        La sequenza includerà le righe precedenti fino a raggiungere self.timesteps righe totali.
    
        Restituisce un array con forma (1, self.timesteps, n_features).
        """
        # Se X è un DataFrame, ne estraiamo i valori
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        # Verifica che current_datetime_idx sia sufficiente per ottenere self.timesteps righe
        if current_datetime_idx < self.timesteps - 1:
            raise ValueError(f"current_datetime_idx deve essere almeno {self.timesteps - 1}")
    
        # Estrai la sequenza che termina alla riga current_datetime_idx
        sample = X_values[current_datetime_idx - self.timesteps + 1: current_datetime_idx + 1]
    
        # Applica lo scaling se richiesto
        if apply_scaler:
            sample = self.X_scaler.transform(sample)
    
        # Aggiungi la dimensione del batch, ottenendo forma (1, timesteps, n_features)
        sample = np.expand_dims(sample, axis=0)
    
        return sample

        

    def model_predict(self, data, apply_scaler=True, descale_result=True):
        """
        Effettua la predizione sul dataset 'data'. Se apply_scaler è True trasforma l'input
        usando X_scaler; se descale_result è True e scale_target è abilitato, applica l'inverse_transform.
        """
        if apply_scaler:
            data = self.X_scaler.transform(data)
        predictions = self.model.predict(data)
        if descale_result and self.scale_target:
            predictions = self.Y_scaler.inverse_transform(predictions)
        return predictions

        
