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

## 단일 년도에 대한 국경일 정보 조회
## Input
##  1) year_tar : 대상년도
##  2) str_key : 발급받은 개인 키
## Output
##  1) 해당 휴일 정보가 포함된 데이터셋
def KASI_holiDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## 단일 년도에 대한 공휴일 정보 조회
## Input
##  1) year_tar : 대상년도
##  2) str_key : 발급받은 개인 키
## Output
##  1) 해당 휴일 정보가 포함된 데이터셋
def KASI_restDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## 단일 년도에 대한 기념일 정보 조회
def KASI_anniDay(year_tar, str_key):
    ## Parameter for Request
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getAnniversaryInfo"
    key = str_key

    params = "?" + urlencode({
        quote_plus("ServiceKey") : key
        , quote_plus("_type") : "json"
        , quote_plus("solYear") : str(year_tar)
        , quote_plus("numOfRows") : 100
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["response"]["body"]["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')