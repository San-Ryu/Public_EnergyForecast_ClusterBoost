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

url = "https://www.bigdata-transportation.kr/api"

## [API] 톨게이트 목록 현황
## Input
##  1) str_key : 발급받은 개인 키
## Output
##  1) 톨게이트 목록
def KorEx_Tollgates(str_key):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020307" ## 고유코드
        , quote_plus("numOfRows") : 999
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["unitLists"]#["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## [API] 영업소별 입구, 출구 현황
def KorEx_Tollgates_inOut(str_key, str_Tollgate, str_inOut = "None"):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    if str_inOut == "None":
        str_inOut = ""

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020305" ## 고유코드
        , quote_plus("unitCode") : str_Tollgate
        , quote_plus("inoutType") : str_inOut
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["laneStatusVO"]#["items"]["item"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')

## [Unused][API] 톨게이트 입/출구 교통량
## Input
##  1) str_key : 발급받은 개인 키
##  2) tmType : 자료 구분 ("1" : 1시간, "2" : 15분)
##  3) unitCode : 영업소 코드 (청주 : 111 / 남청주 : 112)
## Output
##  1) 톨게이트별 교통량 현황 (현재 시간 기준)
def KorEx_Tollgates_Traffic(str_key, tmType, unitCode):
    ## Parameter for Request
    # url = "https://www.bigdata-transportation.kr/api"
    key = str_key

    params = "?" + urlencode({
        quote_plus("apiKey") : key
        , quote_plus("productId") : "PRDTNUM_000000020308" ## 고유코드
        , quote_plus("tmType") : str(tmType) ## "1" : 1시간, "2" : 15분
        , quote_plus("unitCode") : str(unitCode) ## 청주 : 111 / 남청주 : 112
        , quote_plus("numOfRows") : 999
    })

    req = urllib.request.Request(url + unquote(params))
    response_body = urlopen(req, timeout=600).read()
    data_json = json.loads(response_body)  # convert bytes data to json data
    data_items = data_json["result"]["trafficIc"]

    return pd.DataFrame.from_dict(data=data_items, orient='columns')