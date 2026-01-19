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
<img width="2003" height="1182" alt="Image" src="https://github.com/user-attachments/assets/ea7d65ee-aa0b-4577-8b3d-35b73d4c02fa" />

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

### 연구 결과 상세

**(1) 데이터 수집 모듈**
- Target  
 &nbsp;: 가정용 스마트미터를 통해 수집된 전력 사용량  
- Input Variable  
&nbsp;: 기상청 기상자료개방포털, 한국천문연구원 Open API 기반의 실증지역 날씨 및 특일 정보  
- 데이터 취득 기간 (에너지 사용량 기준)  
&nbsp;: 2022년 07월 17일 23시 20분 ~ 2024년 06월 05일 15시 30분  
- 실증지역 및 규모  
&nbsp;: 대한민국 경기도 내 아파트 단지 (총 3개 동, 348 세대)  
- 총 데이터 수 (에너지 사용량 기준)  
&nbsp;: 33,837,156 건 (세대당 약 99,170 건)  

**(2) 데이터 전처리 모듈**
- 각 세대별 수집된 에너지 사용량(10분 간격)는 평균 99,170건이었으며,  
이 중 평균 1,820건의 결측치(전체 대비 1.835%), 1,979건의 이상치(1.996%)가 존재
- 결과적으로 보간이 진행된 데이터는 3,799건(전체 대비 3.831%)  
&nbsp;: 이론 상 중대한 오류를 발생시킬 요인이 될 수 없었지만,  
&ensp;&nbsp;데이터 상세 분석 과정에서 8시간 이상의 장기 결측 사례 등이 관측되어 모델 학습에 영향을 미칠 것으로 판단됨
- 따라서, 단순 Linear 보간을 유일한 보간법으로 선택하지 않는 것을 선택
- 1차 보간으로는, 시간별 전 세대의 평균 사용량으로 대치(Imputation)  
2차 보간으로는 Linear 보간을 적용

**(3) 데이터 군집화 모듈**
- 군집화의 타당성을 결정 및 평가하는 데에는 개인의 주관이 다소 포함될 위험이 존재  
&nbsp;--> 정량적인 방법을 통해 군집화 조건 및 결과 분석  
&nbsp;&emsp;&emsp;① Inertion(Elbow-Method)  
&nbsp;&emsp;&emsp;② Silhouette Score  
&nbsp;&emsp;&emsp;③ Calinski-Harabasz Index  
&nbsp;&emsp;&emsp;④ Dunn Index  
- 데이터의 시간 간격(Interval)이 증가할수록 아래와 같은 변화가 나타남  
&nbsp;① 4개의 군집화 평가 지수(Inertia/Silhouette Score/CHI/Dunn Index)의 편차 감소  
&nbsp;② 군집의 크기가 1~2 정도 되는 극소 군집이 발생하는 현상이 감소    
&emsp;&emsp;(이는 Iteration 및 Coefficient of Variation을 정량적으로 측정하여 증명)
- 시간 간격과 군집화 계수 K에 따른 10회 Iteration 결과,  
&nbsp;1 Week 또는 1 Month의 Temporal Resolution 및 K=2의 조건을 최적의 조건으로 선정

**(4) ML 모델 기반의 데이터 분석 및 예측 모듈**
- **실험군(Clustering 적용)은 대조군(기존 방식)에 비해 더욱 개선된 성능(MAE, MSE, RMSE)을 보임**  
- Best Model : CatBoost, LightGBM 파생 모델  

<대조군과 최종 Model(Ensemble)의 예측 결과 시각화>
<img width="2003" height="735" alt="Image" src="https://github.com/user-attachments/assets/38c0fa23-3f13-4fd1-ba81-766f18bd99b7" />

<대조군과 최종 Model(Ensemble)의 예측 성능 시각화>
<img width="2003" height="735" alt="Image" src="https://github.com/user-attachments/assets/80172f92-3e6a-4976-bb63-724c8bdc890f" />

### Conclusion(결론)

**(1) Contribution**
- **제안된 Ensemble 모델은 Non-Clustered 모델 대비 MAE 55.6%, MSE 80.1%, RMSE 59.3%의 개선 효과를 달성**  
- 복합 수준에서 R2 Score 0.94~0.95의 성능을 달성  
- 예측 성능을 강화하는 과정에서, 새로운 모델 설계나 데이터 추가 수집 등의 소요가 필요하지 않음  

**(2) Limitation**
- 데이터 수집 오차 (결측치 및 이상치)  
   → 잘못된 시간값 / 음수 사용량 등 통상적인 측정치 범위 외 데이터 / 단순 결측 및 장기간 연속 결측  
   &emsp;&nbsp;(특정 보간법을 단독으로 사용하기에 제한이 있음)
- 기상 데이터 이외 입력변수에 대한 수집 난이도가 높음  
   → 사회적 요인(구성원 수, 재실 등) / 환경적 요인(건물 구조 등) / 경제적 요인(사용 요금, 가계 소득 등)
- 과도한 데이터 전처리 소요 (시간 및 용량)  
- 에너지 사용자의 이주 등에 따른 사용패턴 변화 반영 어려움  
   ① 세대 구성원 수나 재실 시간 등 정보가 없는 경우,  
   &emsp;&nbsp;사용자 변경 직후 해당 세대에 대한 예측이 어려움  
   ② 군집화를 위한 최적의 사전 데이터 수집 기간에 대한 정의가 어려움  

**(3) Future Works**
- Transformer를 비롯한 최신 모델을 활용한 Framework 구성
- TCN을 활용, CNN계열의 시계열 모델을 활용한 Framework 구성
