## Public_EnergyForecast_ClusterBoost

Clustering 알고리즘을 기반으로 주거 시설 내 348개 세대의 전력 사용량 예측 향상을 위한 Ensemble Framework

### - 관련 링크  
A Machine Learning Ensemble Framework Based on a Clustering Algorithm for Improving Electric Power Consumption Performance  
전력 소비 성능 향상을 위한 군집화 알고리즘 기반 머신러닝 앙상블 프레임워크  
https://doi.org/10.1038/s41598-025-23978-w

### - 개요  
아래 목적을 위해 전통적인 머신러닝 기반 시계열 예측 방식을 개선, 군집화 기반의 Feature 최적화를 수행하여 모델의 성능을 향상  
  1) 건물의 전력 에너지에 대한 예측 최적화
  2) 군집화를 통해 다수의 세대에 걸친 에너지 사용 패턴을 분류, 시계열 모델에 적용하여 모델 성능 및 분석 업무에 대한 효율성 강화

### - 프로젝트 개요
![Uploading {image file name}](https://github.com/user-attachments/assets/b0523d46-c7b8-4887-8b46-5b6bde858539)


모듈의 구성은 아래와 같음  
1) 데이터 수집 모듈  
     : 에너지 사용량/날씨/날짜 데이터를 Cleansing 및 Merge  
2) 데이터 전처리 모듈  
     : 분리된 세대별 평균 대치법/Spline 보간/IQR 기반 이상치 제거  
3) 군집화 모듈  
     : 데이터에 대한 최적의 군집의 수 K 도출 및 조건별 군집화 수행  
4) 데이터 분석 및 예측 모듈  
     : 비군집화 또는 군집화 데이터 세트에 대한 모델 분석/예측 수행  



아래와 같이 비교 대상 모델을 생성 및 비교  
  (1) 대조군(General Method)  
&emsp;&ensp;&nbsp;: 기존 전통적인 방식의 DL/ML 분석 방법론  
&emsp;&emsp;&ensp;① 데이터 수집 및 전처리  
&emsp;&emsp;&ensp;② 모델 구축 및 학습  
&emsp;&emsp;&ensp;③ 시계열 분석 및 예측  
&emsp;&emsp;&ensp;④ 모델 성능 평가  
  (2) 실험군 01(Model for each clusters)  
&emsp;&ensp;&nbsp;: 군집화 기법을 도입, 데이터를 분할하여 처리  
&emsp;&emsp;&ensp;① 데이터 수집 및 전처리  
&emsp;&emsp;&ensp;② 군집화를 위한 최적의 군집화 계수 K 도출  
&emsp;&emsp;&ensp;③ 도출된 K를 기반으로 군집화 및 라벨링  
&emsp;&emsp;&ensp;④ 분류된 세대별 데이터 분할   
&emsp;&emsp;&ensp;⑤ 모델 구축 및 분할된 데이터 학습  
&emsp;&emsp;&ensp;⑥ 군집별 생성된 모델 성능 평가  
  (3) 실험군 02(Ensemble Method)을 비교  
&emsp;&ensp;&nbsp;: 군집화 기법을 도입, 데이터를 분할하여 처리  
&emsp;&emsp;&ensp;① 데이터 수집 및 전처리  
&emsp;&emsp;&ensp;② 최적의 K 도출, 군집화 수행 및 라벨링  
&emsp;&emsp;&ensp;③ 세대별 데이터 분할 및 모델 학습  
&emsp;&emsp;&ensp;④ 군집별 우수한 모델을 기반으로 Ensemble Model 구현 및 평가  
  
각 모델의 성능을 MAE, MSE, RMSE, R2 Score 기준으로 비교

### 연구 결과 요약
  - 프로젝트의 최종 단계에서 대조군, 실험군 01, 실험군 02에 대한 성능 비교
  - Ensemble Model은 대조군에 비해 모든 측도(MAE, MSE, RMSE, R2 Score)에 따라 향상된 성능을 보임.  
  **대조군 대비 MAE 55.6%, MSE 80.1%, RMSE 59.3%의 절감률, R2 Score 0.94~0.95의 수준을 달성**
