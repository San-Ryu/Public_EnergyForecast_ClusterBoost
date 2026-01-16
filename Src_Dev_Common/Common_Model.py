import os, sys, warnings
import numpy as np, pandas as pd, math, random
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
import glob
import time
from datetime import datetime, date, timedelta
from scipy import stats

# Configure display
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.path.dirname(os.path.abspath('./__file__'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./__file__'))))
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.10f}'.format

# 시각화 설정
plt.rcParams['figure.figsize'] = [10, 8]
Font_TimesNewRoman = fm.FontProperties(fname='../Src_Dev_Common/Times New Roman.ttf')

# Split, 정규화
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# K-Means 알고리즘
from sklearn.cluster import KMeans, MiniBatchKMeans

# Evaluation on Clustering 
from sklearn import metrics
from sklearn.metrics import (
    homogeneity_score, completeness_score, v_measure_score, 
    adjusted_rand_score, silhouette_score, rand_score, 
    calinski_harabasz_score, davies_bouldin_score,
    mean_absolute_error, mean_absolute_percentage_error, 
    mean_squared_error, mean_squared_log_error, r2_score
)
from sklearn.metrics.cluster import contingency_matrix

# Models - ML
import catboost as cb
from catboost import Pool, CatBoostRegressor
import lightgbm as lgbm
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import plot_importance, plot_tree, XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Models - DL
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# Import_Local
from Src_Dev_Common import (
    COM_Analysis as com_Analysis,
    COM_Clustering as com_clustering,
    COM_Datetime as com_date,
    KMA_Weather as com_KMA,
    KECO_AirKor as com_KECO,
    KASI_Holiday as com_Holi,
    KIER_Usage_M02 as com_KIER_M02
)
#endregion Module_Import

#region Model
## Data Split
## Input 
##  1) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
##  2) float_rate : Test 비율
##  3) str_col_tar : Target Column명
def split_build_DataSet(df_tar, float_rate, str_col_tar, seqLength):
    ## Data Split
    trainSet_Origin, testSet_Origin = train_test_split(df_tar, test_size = float_rate, shuffle = False)

    trainSet, testSet = trainSet_Origin, testSet_Origin

    ## Input / Target Split
    trainXX, trainYY = trainSet.drop([str_col_tar],axis=1), trainSet[[str_col_tar]]
    testXX, testYY = testSet.drop([str_col_tar],axis=1), testSet[[str_col_tar]]

    trainXXindex, trainYYindex = trainXX.index, trainYY.index
    trainXXcolumns, trainYYcolumns = trainXX.columns, trainYY.columns

    testXXindex, testYYindex = testXX.index, testYY.index
    testXXcolumns, testYYcolumns = testXX.columns, testYY.columns

    d_trainXX, d_trainYY = pd.DataFrame(trainXX, index=trainXXindex, columns=trainXXcolumns), trainYY

    d_testXX, d_testYY = pd.DataFrame(testXX, index=testXXindex, columns=testXXcolumns), testYY

    ## Build Dataset
    trainX, trainY = buildDataSet(trainXX, trainYY, seqLength)
    testX, testY = buildDataSet(testXX, testYY, seqLength)

    return trainX, testX, trainY, testY

## Build Dataset
def buildDataSet(traindata, testdata, seqLength):
    xdata = []
    ydata = []

    for i in range(len(traindata)-seqLength+1):
        tx = traindata.iloc[i:i+seqLength]
        ty = testdata.iloc[i+seqLength-1]
        xdata.append(tx)
        ydata.append(ty)

    return np.array(xdata), np.array(ydata)

## Model Predict
## Input 
##  1) int_model
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_dl_predict_KIERM02(trainX, trainY, testX, testY, model):
    ## Train Start
    tm_start = time.time()

    ## Model Analysis
    earlystopping = EarlyStopping(monitor = 'loss', patience = 15)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adamax(learning_rate = 3e-4, clipnorm = 1.0), metrics = ['mae'])

    hist = model.fit(trainX, trainY, epochs = 500, batch_size = 32, callbacks = [earlystopping])

    ## Train Over
    tm_code = time.time() - tm_start

    ## Visualization
    # if int_model == 2: ## LGBM
    #     lgbm.plot_importance(model)
    #     plt.show()
    # elif int_model == 3: ## RF
    #     ftr_importances_values = model.feature_importances_
    #     ftr_importances = pd.Series(ftr_importances_values, index=df_trainXX.columns)
    #     ftr_top = ftr_importances.sort_values(ascending=False)[:20]
    #     plt.figure(figsize=(8, 6))
    #     sns.barplot(x=ftr_top, y=ftr_top.index)
    #     plt.show()
    # elif int_model == 4: ## XGBoost
    #     ## 주요 변수 판단
    #     plot_importance(model)

    model_pred = model.predict(testX)
    model_preds = model_pred.reshape(-1,1)

    d_actual = testY.reshape(-1,1)

    return d_actual, model_preds, tm_code

## 단일 Model Analysis
## Input 
##  1) int_model : 모델 번호
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_dl_analysis_single_KIERM02(df_tar, model, float_rate, str_col_tar, seqLength):
    ## Data Split
    trainX, testX, trainY, testY = split_build_DataSet(df_tar, float_rate, str_col_tar, seqLength)

    d_actual, model_preds, tm_code = model_dl_predict_KIERM02(trainX, trainY, testX, testY, model)

    return d_actual, model_preds, tm_code

## KFold Model Analysis
## Input 
##  1) int_model : 모델 번호 {0 : CB, 1 : DT, 2 : LGBM, 3 : RF, 4 : XGB}
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_dl_analysis_with_KFold(df_tar, float_rate, model, str_col_tar, int_fold, str_shuffle, seqLength):
    ## 초기 변수 생성
    list_kf_mae, list_kf_mape, list_kf_mse, list_kf_rmse, list_kf_msle, list_kf_mbe, list_kf_r2, list_kf_tm_code = [], [], [], [], [], [], [], []
    list_kf_scores, list_kf_hists = [], [] ## 최종 출력될 Score List / Crossvalidation 진행에 따른 History List

    ## K-Fold 객체 생성
    k_fold = KFold(n_splits = int_fold, shuffle = str_shuffle)

    ## K-Fold 연산 수행
    # int_seq = 0
    for df_train, df_test in k_fold.split(df_tar):
        ## K-Fold 연산 수행 확인
        # int_seq = int_seq + 1
        # str_txt = "../kf_hist/kf_log_" + str(int_model) + '_' + str(int_seq) + '.txt'
        # file_txt = open(str_txt, 'w')
        # print(df_train, file = file_txt)
        # print(df_test, file = file_txt)

        ## Model Analysis
        d_actual, model_preds, tm_code = model_dl_analysis_single_KIERM02(df_tar, model, float_rate, str_col_tar, seqLength)

        ## Score 산출
        list_scores = model_sk_metrics(d_actual, model_preds)

        ## 각 list_score별 수집
        list_kf_mae.append(list_scores[0])
        list_kf_mape.append(list_scores[1])
        list_kf_mse.append(list_scores[2])
        list_kf_rmse.append(list_scores[3])
        list_kf_msle.append(list_scores[4])
        list_kf_mbe.append(list_scores[5])
        list_kf_r2.append(list_scores[6])
        # list_kf_tm_code.append(list_scores[7])
        list_kf_tm_code.append(tm_code)
    
    list_kf_hists.append(list_kf_mae)
    list_kf_hists.append(list_kf_mape)
    list_kf_hists.append(list_kf_mse)
    list_kf_hists.append(list_kf_rmse)
    list_kf_hists.append(list_kf_msle)
    list_kf_hists.append(list_kf_mbe)
    list_kf_hists.append(list_kf_r2)
    list_kf_hists.append(list_kf_tm_code)

    score_kf_mae = sum(list_kf_mae) / len(list_kf_mae)
    score_kf_mape = sum(list_kf_mape) / len(list_kf_mape)
    score_kf_mse = sum(list_kf_mse) / len(list_kf_mse)
    score_kf_rmse = sum(list_kf_rmse) / len(list_kf_rmse)
    score_kf_msle = sum(list_kf_msle) / len(list_kf_msle)
    score_kf_mbe = sum(list_kf_mbe) / len(list_kf_mbe)
    score_kf_r2 = sum(list_kf_r2) / len(list_kf_r2)
    score_kf_tm_code = sum(list_kf_tm_code) / len(list_kf_tm_code)

    list_kf_scores.append(round(score_kf_mae, 4))
    list_kf_scores.append(round(score_kf_mape, 4))
    list_kf_scores.append(round(score_kf_mse, 4))
    list_kf_scores.append(round(score_kf_rmse, 4))
    list_kf_scores.append(round(score_kf_msle, 4))
    list_kf_scores.append(round(score_kf_mbe, 4))
    list_kf_scores.append(round(score_kf_r2, 4))
    list_kf_scores.append(round(score_kf_tm_code, 4))

    ## [추가예정] list_d_actual / list_model_pred는 시각화 표현하고 싶을 때 변수로서 사용
    return list_kf_scores, list_kf_hists

## KFold Model Analysis
## Input 
##  1) int_model : 모델 번호 {0 : CB, 1 : DT, 2 : LGBM, 3 : RF, 4 : XGB}
##  2) df_tar : 타겟 데이터 (Train/Test 분리 X 원본)
def model_ml_analysis_with_KFold(df_tar, int_model, float_rate, str_col_tar, int_fold):
    ## 초기 변수 생성
    list_kf_mae, list_kf_mape, list_kf_mse, list_kf_rmse, list_kf_msle, list_kf_mbe, list_kf_r2, list_kf_tm_code = [], [], [], [], [], [], [], []
    list_kf_scores, list_kf_hists = [], [] ## 최종 출력될 Score List / Crossvalidation 진행에 따른 History List

    ## K-Fold 객체 생성
    k_fold = KFold(n_splits = int_fold)

    ## K-Fold 연산 수행
    # int_seq = 0
    for df_train, df_test in k_fold.split(df_tar):
        ## K-Fold 연산 수행 확인
        # int_seq = int_seq + 1
        # str_txt = "../kf_hist/kf_log_" + str(int_model) + '_' + str(int_seq) + '.txt'
        # file_txt = open(str_txt, 'w')
        # print(df_train, file = file_txt)
        # print(df_test, file = file_txt)

        ## Data Split
        trainXX, trainYY = df_tar.iloc[df_train].drop([str_col_tar],axis=1), df_tar.iloc[df_train][[str_col_tar]]
        testXX, testYY = df_tar.iloc[df_test].drop([str_col_tar],axis=1), df_tar.iloc[df_test][[str_col_tar]]

        trainXXindex, trainYYindex = trainXX.index, trainYY.index
        trainXXcolumns, trainYYcolumns = trainXX.columns, trainYY.columns

        testXXindex, testYYindex = testXX.index, testYY.index
        testXXcolumns, testYYcolumns = testXX.columns, testYY.columns

        d_trainXX, d_trainYY = pd.DataFrame(trainXX, index = trainXXindex, columns = trainXXcolumns), trainYY
        d_testXX, d_testYY = pd.DataFrame(testXX, index = testXXindex, columns = testXXcolumns), testYY

        ## Model Analysis
        d_actual, model_preds, tm_code = model_ml_predict(d_trainXX, d_trainYY, d_testXX, d_testYY, int_model)

        ## Score 산출
        list_scores = model_sk_metrics(d_actual, model_preds)

        ## 각 list_score별 수집
        list_kf_mae.append(list_scores[0])
        list_kf_mape.append(list_scores[1])
        list_kf_mse.append(list_scores[2])
        list_kf_rmse.append(list_scores[3])
        list_kf_msle.append(list_scores[4])
        list_kf_mbe.append(list_scores[5])
        list_kf_r2.append(list_scores[6])
        # list_kf_tm_code.append(list_scores[7])
        list_kf_tm_code.append(tm_code)
    
    list_kf_hists.append(list_kf_mae)
    list_kf_hists.append(list_kf_mape)
    list_kf_hists.append(list_kf_mse)
    list_kf_hists.append(list_kf_rmse)
    list_kf_hists.append(list_kf_msle)
    list_kf_hists.append(list_kf_mbe)
    list_kf_hists.append(list_kf_r2)
    list_kf_hists.append(list_kf_tm_code)

    score_kf_mae = sum(list_kf_mae) / len(list_kf_mae)
    score_kf_mape = sum(list_kf_mape) / len(list_kf_mape)
    score_kf_mse = sum(list_kf_mse) / len(list_kf_mse)
    score_kf_rmse = sum(list_kf_rmse) / len(list_kf_rmse)
    score_kf_msle = sum(list_kf_msle) / len(list_kf_msle)
    score_kf_mbe = sum(list_kf_mbe) / len(list_kf_mbe)
    score_kf_r2 = sum(list_kf_r2) / len(list_kf_r2)
    score_kf_tm_code = sum(list_kf_tm_code) / len(list_kf_tm_code)

    list_kf_scores.append(round(score_kf_mae, 4))
    list_kf_scores.append(round(score_kf_mape, 4))
    list_kf_scores.append(round(score_kf_mse, 4))
    list_kf_scores.append(round(score_kf_rmse, 4))
    list_kf_scores.append(round(score_kf_msle, 4))
    list_kf_scores.append(round(score_kf_mbe, 4))
    list_kf_scores.append(round(score_kf_r2, 4))
    list_kf_scores.append(round(score_kf_tm_code, 4))

    ## [추가예정] list_d_actual / list_model_pred는 시각화 표현하고 싶을 때 변수로서 사용
    return list_kf_scores, list_kf_hists

## Define Metric : MBE
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
## Output
##  1) Metric
def mean_bias_error(d_actual, model_preds):
    mbe_loss = np.sum(d_actual - model_preds)/d_actual.size

    return mbe_loss

## Metrics
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
## Output
##  1) Metrics
def model_sk_metrics(d_actual, model_preds):
    score_mae = round(mean_absolute_error(d_actual, model_preds), 4)
    score_mape = round(mean_absolute_percentage_error(d_actual, model_preds), 4)
    score_mse = round(mean_squared_error(d_actual, model_preds), 4)
    score_rmse = round(mean_squared_error(d_actual, model_preds, squared = False), 4)

    ## ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
    try : score_msle = round(mean_squared_log_error(d_actual, model_preds), 4)
    except ValueError : score_msle = np.nan

    score_mbe = round(mean_bias_error(d_actual, model_preds), 4)
    score_r2 = round(r2_score(d_actual, model_preds), 4)

    list_scores = [score_mae, score_mape
                   , score_mse, score_rmse
                   , score_msle
                   , score_mbe
                   , score_r2]

    # ## Mean_Absolute_Error
    # print('MAE  : ', score_mae)
    # ## Mean_Absolute_Percentage_Error
    # print('MAPE : ', score_mape)
    # ## Mean_Squared_Error
    # print('MSE  : ', score_mse)
    # ## Root_Mean_Squared_Error
    # print('RMSE : ', score_rmse)
    # ## Mean_Squared_Log_Error
    # print('MSLE : ', score_msle)
    # ## Mean_Bias_Error
    # print('MBE  : ', score_mbe)
    # ## R2_Score
    # print('R2   : ', score_r2)

    ## 필요시 변수화하여 사용
    return list_scores

## 시각화
## Input
##  1) d_actual : 대조군 (실제 데이터)
##  2) model_preds : 실험군 (예측 데이터)
##  3) str_title : Title
## Output
##  1) 시각화 : 대조군과 실험군에 대한 시각화 비교
def model_visualization(d_actual, model_preds, str_title, str_path = None):
    plt.figure(figsize=(50, 15), dpi = 500)
    plt.plot(d_actual,color='red',label='Actual')
    plt.plot(model_preds,color='blue',label='Predicted')
    plt.title(str_title, fontproperties = Font_TimesNewRoman, fontsize = 80)
    plt.xlabel('Period', fontproperties = Font_TimesNewRoman, fontsize = 50)
    plt.xticks(fontsize=30)
    plt.ylabel('Energy Usage', fontproperties = Font_TimesNewRoman, fontsize = 50)
    plt.yticks(fontsize=30)
    plt.legend(fontsize = 50)

    if str_path == None : plt.savefig(str_title, dpi = 500)
    else : plt.savefig(str_path, dpi = 500)
    plt.show()
#endregion Model