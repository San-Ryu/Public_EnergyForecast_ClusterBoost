## dev-shryu

#region Info
## Hist
## [2024-04-03] Created

## Desc
## Dataset 처리에 있어서 공통으로 사용하는 기능들을 함수화
#endregion Info

#region Module_Import
## Basic Import
## Basic
import os, sys, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.path.dirname(os.path.abspath('./__file__'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./__file__'))))
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from pandas import DataFrame, Series
pd.options.display.float_format = '{:.10f}'.format

import math, random

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
#endregion Module_Import

#region List
## 시각화
# def resample_by_1H_last(df_tar, col_tar):
#endregion List

#region Energy Data Treatment
## 

## Domain 및 관련 ACCU/INST Column명 출력 (하드코딩 기반)
## Input
##  1) int_domain : 대상 domain Num
## Output
##  1) str_domain : 대상 domain명
##  2) str_col_accu : 대상 ACCU Column명
##  1) str_col_inst : 대상 INST Column명
def create_domain_str (int_domain):
    ## Dictionary 생성
    dict_domain = {0:"ELEC", 1:"HEAT", 2:"WATER", 3:"HOT_HEAT", 4:"HOT_FLOW", 5:"GAS"}

    dict_col_accu = {0 : "ACTUAL_ACCU_EFF" ## ELEC
                     , 1 : "ACCU_HEAT" ## HEAT
                     , 2 : "ACCU_FLOW" ## WATER
                     , 3 : "ACCU" ## HOT 열량
                     , 4 : "ACCU" ## HOT 유량
                     , 5 : "ACCU_FLOW"} ## GAS
    dict_col_inst = {0 : "INST_EFF" ## ELEC_ACCU/INST_EFF
                     , 1 : "INST_HEAT" ## HEAT_ACCU/INST_HEAT
                     , 2 : "INST_FLOW" ## WATER_ACCU/INST_FLOW
                     , 3 : "INST" ## HOT_ACCU/INST_HEAT
                     , 4 : "INST" ## HOT_ACCU/INST_FLOW
                     , 5 : "INST_FLOW"} ## GAS_ACCU/INST_FLOW
     
    ## 입력받은 Int와 Dict로부터 Str 구성
    str_domain = str(dict_domain[int_domain])

    str_col_accu, str_col_inst = str(str_domain + "_" + str(dict_col_accu[int_domain])), str(str_domain + "_" + str(dict_col_inst[int_domain]))
    print(str(int_domain) + ' : ' + str_domain)

    return str_domain, str_col_accu, str_col_inst

## Domain 및 관련 Directory명 출력 (하드코딩 기반)
## Input
##  1) str_domain : 대상 domain명
## Output
##  1) str_dirData : 수집된 Raw Data 디렉터리
##  2) str_dir_raw : 1차 Cleansing된 Data 디렉터리
##  3) str_dirName_bld : 분리된 데이터_Building 디렉터리
##  4) str_dirName_f : 분리된 데이터_Floor 디렉터리
##  5) str_dirName_h : 분리된 데이터_Household 디렉터리
def create_dir_str (str_domain):
    str_dirData = "../data_Energy_KIER/"
    str_dir_raw, str_dir_cleansed = '../data_Energy_KIER/KIER_0_Raw/', '../data_Energy_KIER/KIER_1_Cleansed/'
    str_dirName_bld, str_dirName_h = '../data_Energy_KIER/KIER_2_BLD/', '../data_Energy_KIER/KIER_3_H_' + str_domain + '/'

    return str_dirData, str_dir_raw, str_dir_cleansed, str_dirName_bld, str_dirName_h

## Time Interval 및 Target File Name 출력 (하드코딩 기반)
## Input
##  1) str_domain : 대상 domain명
##  2) int_interval : 대상 domain명
## Output
##  1) str_fileRaw : 대상 domain명
##  2) str_fileRaw_hList : 대상 ACCU Column명
##  3) str_file : 대상 INST Column명
def create_file_str (str_domain, int_interval = None):
    ## Dictionary 생성
    ## Interval 범위 축소 : 연구 효용성 없음
    # dict_interval = {0 : '10MIN', 1 : '30MIN', 2 : '1H', 3 : '12H', 4 : '1D', 5 : '1W', 6 : '2W', 7 : '1M'}
    dict_interval = {0 : '10MIN', 1 : '1H', 2 : '1D', 3 : '1W', 4 : '1M'}

    if int_interval == None : str_interval = 0
    else : str_interval = dict_interval[int_interval]

    str_fileRaw, str_fileRaw_hList = str('KIER_RAW_' + str_domain + '_2024-06-07.csv'), str('KIER_hList_common.csv')

    ## 각 Interval별 CSV 파일
    str_file = 'KIER_' + str_domain + '_INST_' + str(str_interval) + '_Resampled.csv'
    ## [미구현] 분기 단위
    # str_file = 'KIER_' + str_domain + '_ACCU_MAXACCU_InstBaseUpdated.csv' 
    ## [미사용] 기간 총합
    # str_file = 'KIER_' + str_domain + '_ACCU_MAXACCU_InstBaseUpdated.csv' 

    print('str_fileRaw : ' + str_fileRaw)
    print('str_fileRaw_hList : ' + str_fileRaw)
    print('str_file : ' + str_file)

    return str_interval, str_fileRaw, str_fileRaw_hList, str_file
#endregion Energy Data Treatment