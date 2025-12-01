---
layout: post
title: "Map Matching Algorithm"
date: 2025-11-24
author: knon0501
tags: [algorithm]
---

## 개요

Map Matching이란 Vehicle의 GPS 좌표 목록이 주어졌을 때 실제 도로상에서 어떠한 경로로 이동했는지를 복원하는 문제입니다. 다시 말해 입력으로 위경도 timeseries 데이터와 방향그래프로 구성된 도로 데이터가 주어졌을 때 도로상의 경로를 출력하는 문제입니다. 이 포스팅에서는 2009년 마이크로소프트 연구진에서 저술한 Hidden Markov Map Matching
Through Noise and Sparseness 논문에서 제시하는 Map Matching 문제를 해결하는 알고리즘에 대해 설명하겠습니다.

## 맵매칭 문제의 어려움 

맵매칭을 하는 가장 단순한 방법은 gps 좌표를 가장 가까운 도로로 매칭하는 것입니다. 그러나 gps 측정은 완벽하지 않고 오차가 존재하기 때문에 이러한 방법은 많은 문제가 있습니다. 때문에 단순히 가장 가까운 도로로 매칭해서는 안 되며 다른 좌표들과의 context를 고려해야합니다. 이를 위해 과거에는 gps 좌표의 집합을 한 번에 곡선으로 매칭하는 기하학적 방법을 사용했습니다. 이는 어느 정도의 성과가 있었지만 노이즈에 민감하고 높은 time rate가 필요하다는 제약이 있었습니다. 

## Dynamic Programming으로의 변환
이 논문에서는 Hidden Markov Model과 Viterbi Algorithm을 사용하여 gps 노이즈와 도로 네트워크를 동시에 고려하고 가장 확률이 높은 경로를 예측합니다. 생소한 용어이지만 그냥 dynamic programming과 똑같습니다.
이제 dp를 정의해봅시다. $a_i$를 $i$번째 위경도 데이터로 정의합시다. $b_{i,j}$를 $a_i$가 매칭될 수 있는 후보로 정의합시다. 후보들은 raw gps 좌표에서 주변 도로에 내린 수선의 발들입니다. $dp[i][j] = a_i\text{가} b_{i,j}\text{에 매칭될 확률}$ 으로 정의합시다. dp만 모두 계산하고 역추적을 하면 가장 확률 높은 경로를 얻을 수 있습니다. 이제 점화식을 유도하기 위해 어떤 요소가 확률에 영향을 주는지 생각해봅시다. 

## Measuremnt Probabilities
첫번째로는 $a_i$와 $b_{i,j}$의 great circle distance(구면상의 최단거리)입니다. 거리가 멀수록 확률이 매칭될 작아질 것입니다. 이것을 measuremnt probability라고 하며 gps 오차를 정규분포로 생각하면 다음과 같은 식을 얻을 수 있습니다.

$$p_m(b_{i,j}|a_i) = \frac{1}{\sqrt{2\pi}\sigma}e^{-0.5(\frac{||b_{i,j}-a_i||_{\text{great circle}}}{\sigma})^2}$$

실제로 gps 오차는 정규분포를 정확히 따르지는 않지만 이 논문에서는 실험적으로 효율적이라고 합니다. 
당연하게도 지구상의 모든 도로에 대해 이를 계산할 필요는 없습니다. 연산량을 줄이기 위하여 일정 거리 이내의 도로에 내린 수선의 발들에 대해서만 계산해야합니다. 이 논문에서는 200m로 설정하였습니다.

## Transition Probabilities
도로상에서 $b_{i-1,k}$에서 $b_{i,j}$로 이동한 경로를 그려보았을 때 어떤 경로는 매우 이상하고 어떤 경로는 그럴듯합니다. 예를 들면 10초동안 유턴을 5번 하는 건 직관적으로 이상합니다. 이것을 정량적인 형태로 표현할 수 있을까요? 이 논문에서는 도로상의 최단거리(route distance)가 측정점 사이의 great circle distance와 비슷할수록 좋다고 주장합니다. 즉 $$|\, {|| a_i-a_{i-1}||}_{\text{great circle}} - ||b_{i,j} - b_{i-1,k}||_{\text{route}}\,|$$가 작을수록 확률이 높습니다. 이 확률을 transition probablity라고 합니다. transition probability는 실험적으로 지수분포를 따름이 밝혀졌습니다. 즉 다음 식이 성립합니다. 

$$p_t(d_{i,j}) = \frac{1}{\beta}e^{d_{i,j}/\beta}$$

$$ d_i = |\, || a_i-a_{i-1}||_{\text{great circle}} - ||b_{i,j} - b_{i-1,k}||_{\text{route}}\,|$$

이제 다음과 같은 점화식을 얻을 수 있습니다. 
$dp[i][j] = \max dp[i-1][k]\times p(b_{i,j}|a_i)\times p_t(d_{i,j})$


## 최적화
알고리즘을 실제로 구현할 때에는 최적화가 필요합니다. 논문에서는 이전 포인트에서 2표준편차 이내의 데이터포인트를 제거하는 등의 전처리와 180km이상의 속도를 내는 비현실적 경로를 제외하는 등의 최적화를 소개합니다. 


## 실험 결과
테스트 결과 1초 간격 샘플링에서는 0%의 오류, 30초 간격에서도 0.11%의 오류로 뛰어난 성능을 보였습니다. 이는 저장 공간과 대역폭 절약에 중요한 시사점을 제공하였습니다. 

원본 데이터에 인공적인 정규분포 노이즈를 심어서 테스트한 결과, 50m 표준편차까지 준수한 성능을 보였습니다.이는 WiFi나 기지국 기반 위치 측정에도 적용 가능함을 시사합니다. 흥미롭게도 샘플링 간격이 1초일 때 샘플링 간격이 길 때보다 노이즈에 민감하였습니다.


## 결론 
이 논문은 Vahalla 등 오픈소스 라우팅 엔진이나 카카오와 같은 대기업에서 사용하는 맵매칭 알고리즘의 토대가 되었다는 점, 실험 데이터를 공개했다는 점에서 의의가 있습니다. 
특이한 점은 gps정보로부터 heading(어느 방향을 향하는지) 정보를 얻을 수 있음에도 활용하지 않는다는 것입니다. 이는 heading 데이터는 측정 사이의 시간 간격이 긴 경우 별 쓸모가 없고 오히려 정확도를 떨어뜨릴 수 있기 때문입니다. 
저는 처음 이 논문을 보고 간결하고 우아한 접근만으로 실용적인 문제를 해결할 수 있다는 점이 감명깊었습니다. 이 포스팅을 보고 많은 분들이 geoengineering에 관심이 생기면 좋겠습니다.

## 참고문헌
P. Newson and J. Krumm, "Hidden Markov map matching through noise and sparseness," in Proc. 17th ACM SIGSPATIAL Int. Conf. Advances in Geographic Information Systems (ACM GIS), Seattle, WA, USA, Nov. 2009, pp. 336-343.