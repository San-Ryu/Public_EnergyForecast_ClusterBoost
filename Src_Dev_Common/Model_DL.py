## Dataset 처리에 있어서 공통으로 사용하는 기능들을 함수화
import os, sys, warnings, numpy as np, pandas as pd, math, random
from pandas import DataFrame, Series
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.path.dirname(os.path.abspath('./__file__'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./__file__'))))
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.10f}'.format

## Datetime
import time, datetime as dt
from datetime import datetime, date, timedelta

## glob
import glob, requests, json
from glob import glob

## 시각화
import matplotlib.pyplot as plt, seaborn as sns
# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]
## Font_Times New Roman
import matplotlib.font_manager as fm 
Font_TimesNewRoman = fm.FontProperties(fname='../Src_Dev_Common/Times New Roman.ttf')

## 통계
from scipy import stats

## Split, 정규화
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# K-Means 알고리즘
from sklearn.cluster import KMeans, MiniBatchKMeans

# Evaluation on Clustering 
from sklearn import metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score, rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix

## Web
import urllib
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

import tqdm
from tqdm.notebook import tqdm

## Models - ML
## Catboost
import catboost as cb
from catboost import Pool, CatBoostRegressor
## LightGBM
import lightgbm as lgbm
from lightgbm import LGBMRegressor
## XGBoost
import xgboost as xgb
from xgboost import plot_importance, plot_tree, XGBClassifier
## Decision Tree
from sklearn.tree import DecisionTreeRegressor
## RandomForest
from sklearn.ensemble import RandomForestRegressor

## sklearn.metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score

## Models - DL
## Tensorflow
import tensorflow as tf, tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras_flops import get_flops
## LSTM
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

## Model - DL (buildModel_프로젝트_모델타입)
## 순번 : 01
## 프로젝트 및 용도 : KIER Predict Energy Usage M02
## Model 유형 : 1D CNN LSTM Model Build
## 입력값 설명
##  - trainXXcolumns = trainXX.columns
##  - int_len_col_input = len(trainXXcolumns)
def buildModel_KIERM02_1DCNNLSTM(int_len_col_input):
    str_model = '1D-CNN_LSTM'
    seqLength = 3
    str_act_func = 'swish'

    ## Input
    model_input = tf.keras.layers.Input(shape=(seqLength, int_len_col_input))

    ## 1DCNN
    conv1 = tf.keras.layers.Conv1D(512, 1, activation = str_act_func)(model_input)
    pool1 = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1)(conv1)
    bat01 = tf.keras.layers.BatchNormalization()(pool1)
    conv2 = tf.keras.layers.Conv1D(1024, 1, activation = str_act_func)(bat01)
    pool2 = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1)(conv2)
    bat02 = tf.keras.layers.BatchNormalization()(pool2)

    ## LSTM
    lstm0 = tf.keras.layers.LSTM(1024, activation = str_act_func, dropout = 0.15, return_sequences = True)(bat02)
    lstm1 = tf.keras.layers.LSTM(512, activation = str_act_func, dropout = 0.15, return_sequences = True)(lstm0)
    lstm2 = tf.keras.layers.LSTM(256, activation = str_act_func, dropout = 0.15, return_sequences = True)(lstm1)

    ## Dense
    bat03 = tf.keras.layers.BatchNormalization()(lstm2)
    dense1 = tf.keras.layers.Dense(256, activation = str_act_func)(bat03)
    bat04 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(128, activation = str_act_func)(bat04)
    bat05 = tf.keras.layers.BatchNormalization()(dense2)
    dense3 = tf.keras.layers.Dense(64, activation = str_act_func)(bat05)
    bat06 = tf.keras.layers.BatchNormalization()(dense3)

    ## Output
    model_output = tf.keras.layers.Dense(1)(bat06)
    model = tf.keras.models.Model(model_input, model_output)

    # model.summary()

    return str_model, model

## 순번 : 02
## 프로젝트 및 용도 : KIER Predict Energy Usage M02
## Model 유형 : 1DCNN Seq2Seq Model Build
## 입력값 설명
##  - trainXXcolumns = trainXX.columns
##  - int_len_col_input = len(trainXXcolumns)
def buildModel_KIERM02_1DCNNSeq2Seq(input_shape):
    str_model = '1D-CNN_Seq2Seq'
    seqLength = 64
    str_act_func = 'swish'

    model_input = tf.keras.layers.Input(shape=input_shape)

    # for feature extracting
    conv1 = tf.keras.layers.Conv1D(1024, 1, activation='swish')(model_input)
    pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')(conv1)
    bat01 = tf.keras.layers.BatchNormalization()(pool1)
    conv2 = tf.keras.layers.Conv1D(512, 1, activation='swish')(bat01)
    pool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')(conv2)
    bat02 = tf.keras.layers.BatchNormalization()(pool2)

    # 인코더 - 디코더 선언
    encoder_lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, activation='swish')
    encoder_lstm2 = tf.keras.layers.LSTM(512, return_sequences=True, activation='swish')
    encoder_lstm3 = tf.keras.layers.LSTM(1024, return_state=True, return_sequences=True, activation='swish')

    decoder_lstm1 = tf.keras.layers.LSTM(1024, return_sequences=True, activation='swish')
    decoder_lstm2 = tf.keras.layers.LSTM(512, return_sequences=True, activation='swish')
    decoder_lstm3 = tf.keras.layers.LSTM(256, return_sequences=True, activation='swish')

    # 인코더
    encoder_output_lstm1 = encoder_lstm1(bat02)
    encoder_output_lstm2 = encoder_lstm2(bat01)
    encoder_output_lstm4, state_h, state_c = encoder_lstm3(encoder_output_lstm2)

    #디코더
    decoder_lstm1_output = decoder_lstm1(encoder_output_lstm4, initial_state=[state_h, state_c])
    decoder_lstm2_output = decoder_lstm2(decoder_lstm1_output)
    decoder_lstm3_output = decoder_lstm3(decoder_lstm2_output)

    flatten = tf.keras.layers.Flatten()(decoder_lstm3_output)
    model_output = tf.keras.layers.Dense(1)(flatten)
    
    model = tf.keras.models.Model(model_input, model_output)

    # model.summary()

    return str_model, model