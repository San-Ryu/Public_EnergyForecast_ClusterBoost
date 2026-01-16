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

## 시각화
## Input
##  1) df_tar : 대상 Dataset
##  2) col_tar : 대상 Datatime Column
## Output
##  -
def visualization_df(df_tar, str_tarCol, str_color):
    ## "METER_DATE" Column 생성
    df_tar['METER_DATE'] = 0
    for i in range(0, len(df_tar)):
        df_tar['METER_DATE'].iloc[i] = dt.datetime(int(df_tar['YEAR'].iloc[i])
                                                       , int(df_tar['MONTH'].iloc[i])
                                                       , int(df_tar['DAY'].iloc[i])
                                                       , int(df_tar['HOUR'].iloc[i])
                                                       , 0, 0)

    df_tar = df_tar[['METER_DATE', 'YEAR', 'MONTH', 'DAY'
                     , 'code_day_of_the_week'
                     , 'HOUR', str_tarCol]]

    ## 날짜 범위 지정
    date = pd.to_datetime(df_tar['METER_DATE'])

    ## 시각화
    fig, ax1 = plt.subplots(figsize=(30,5))
    title_font = {'fontsize': 20, 'fontweight': 'bold'}

    plt.title(str_tarCol, fontdict=title_font, loc='center', pad = 20)
    ax1.plot(date, df_tar[str_tarCol], color = str_color)