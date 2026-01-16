import os, sys, warnings
import numpy as np, pandas as pd, math, random
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
import glob, json, requests
import time
from datetime import datetime as dt, date, timedelta
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

## Excel/CSV
import openpyxl, xlrd

import urllib
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus

########## 관측소 메타데이터 관련 부분은 사용되지 않음 (관측소 번호 직접 사용) ##########
## KECO AirKorea 대기정보 Raw Data의 Column명을 변경
## Desc
##  : Public Data_Column명을 변경하고, 사용할 컬럼만 지정하여 df_tar로 출력
## Input
##  1) df_tar
##     ['지역'
##      , '측정소명', '측정소코드'
##      , '측정일시'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']
## Output
##  1) df_tar
##     ['METER_DATE'
##      , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
def Rename_KECO_AirKor(df_tar):
    df_tar = df_tar.rename(columns = {'지역' : 'REGION'
                                      , '측정소명' : 'NM_OBSERVATORY', '측정소코드' : 'CD_OBSERVATORY'
                                      , '측정일시' : 'METER_DATE'
                                      , 'SO2' : 'SO2', 'CO' : 'CO', 'O3' : 'O3', 'NO2' : 'NO2', 'PM10' : 'PM10'})
    df_tar = df_tar[['METER_DATE'
                     , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
                     , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
    return df_tar

## 종관기상관측데이터에 대한 기본 Interpolation
## Desc
##  : KIER Energy Project_Weather Data에 대한 Interpolation 수행
## Input
##  1) df_tar
##     ['METER_DATE'
##      , 'REGION', 'CD_OBSERVATORY', 'NM_OBSERVATORY'
##      , 'SO2', 'CO', 'O3', 'NO2', 'PM10']]
## Output
##  1) df_tar : Input과 동일
def Interpolate_KMA_ASOS(df_tar):
    ## Date 형식 지정
    df_tar['METER_DATE'] = pd.to_datetime(df_tar['METER_DATE'])
    ## Data 보간
    df_tar['SO2'] = df_tar['SO2'].interpolate()
    df_tar['CO'] = df_tar['CO'].interpolate()
    df_tar['O3'] = df_tar['O3'].interpolate()
    df_tar['NO2'] = df_tar['NO2'].interpolate()
    df_tar['PM10'] = df_tar['PM10'].interpolate()
    return df_tar