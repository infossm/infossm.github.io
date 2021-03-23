---
layout: post
title: Segment의 개수를 이용하는 순열 경우의 수 문제
date: 2021-03-23 04:00
author: rdd6584
tags: [algorithm, mathematics]
---

## 소개
특정 조건을 만족하는 순열의 개수를 세는 문제 중에서, segment(이하 구간)의 개수를 이용하는 특이한 꼴의 다이나믹 프로그래밍 문제를 다뤄보려고 합니다. 예시 문제를 통해 살펴봅시다.

## 문제1. 길이가 $N$, Increasing segment의 개수가 $K$개인 순열의 개수
Increasing segment란, $l+1 \leq i \leq r$를 만족하는 $i$에 대해 $A_{i-1} < A_i$를 모두 만족하는 $[l, r]$이라고 합니다. 단, 다른 increasing segment에 포함되는 구간은 무시하는 걸로 합시다.

예를 들어, $1\space4\space5\space2\space3\space6\space8\space7$의 increasing segment는 $[1, 3], [4, 7], [8, 8]$로 총 3개입니다.

우리는 길이가 $N$이고 increasing segment의 개수가 $K$개인 순열의 개수를 구하려고 합니다. 이 문제는 어떻게 해결할 수 있을까요?
이 문제는 다이나믹 프로그래밍을 이용하여, $O(NK)$에 해결할 수 있습니다.
$D_{n,k}$를 길이가 $n$이고 수 $1 \sim n$만을 사용하며, increasing segment의 개수가 $k$개인 순열의 개수라고 합시다.
위 조건을 만족하는 임의의 순열에서  $n+1$이라는 수를 임의의 자리에 추가한다고 생각해봅시다. 경우의 수는 크게 아래와 같이 4가지로 분류할 수 있습니다.


<img src="/assets/images/rdd6584_1/segper01.png" width="100%" height="100%">



1. $n+1$을 넣으려는 위치의 왼쪽 값이 오른쪽보다 큰 경우입니다. 이 경우 increasing segment의 개수가 증가하지 않습니다.
2. $n+1$을 넣으려는 위치의 왼쪽 값이 오른쪽보다 작은 경우입니다. 이 경우 increasing segment의 개수가 증가합니다.
3. $n+1$을 순열의 맨 왼쪽에 넣는 경우입니다. 이 경우 increasing segment의 개수가 증가합니다.
4. $n+1$을 순열의 맨 오른쪽에 넣는 경우입니다. 이 경우 increasing segment의 개수가 증가하지 않습니다.

increasing segment의 개수를 증가시키지 않는 1번 경우는 $k-1$개, 4번 경우는 1개로 총 $k$개의 경우가 존재합니다. increasing segment의 개수를 증가시키는 경우는 $n+1-k$개일 것입니다. 따라서, 아래와 같은 상태 변화를 얻을 수 있습니다.

$D_{1,1} = 1$

$D_{n,k} \times k \Rightarrow D_{n+1, k}$

$D_{n,k} \times (n+1-k) \Rightarrow D_{n+1, k+1}$

이 점화식을 이용하면 우리는 최종적으로 $D_{N,K}$를 $O(NK)$만에 구할 수 있게 됩니다.


## 문제2. Kangaroo([링크](https://www.acmicpc.net/problem/13188))

위치 $1 \sim N$이 있습니다.
캥거루가 $cs$에서 출발하여 모든 위치를 정확히 한번씩 방문하여 $cf$에서 마무리하는 경우의 수를 구하는 문제입니다. 단, 캥거루의 직전 위치를 prev, 현재 위치를 current, 다음 위치를 next라고 할때, 
$prev < current$라면, $current > next$

$prev > current$라면, $current < next$를 만족해야 합니다.

시작점과 끝점이 정해져 있어서 무척 어렵게 느껴집니다. 이 문제도 마찬가지로 segment의 개수가 포함된 점화식으로 문제를 해결할 수 있습니다. 캥거루가 이동한 경로는 하나의 순열일 것입니다. 아래 그림을 봅시다.

<img src="/assets/images/rdd6584_1/segper02.png" width="100%" height="100%">

한 직사각형 내부의 모습처럼 캥거루가 이동한 위치는 커졌다가 작아졌다가를 반복하는 형태일 것입니다. 위 그림처럼 양쪽 끝이 아래로 내려가있는 형태의 경로를 하나의 "구간"으로 정의해봅시다. 위 그림에서 구간의 개수는 3개입니다.

$D_{n,k}$를 길이가 $n$이고 수 $1 \sim n$만을 사용하며, 위와 같은 구간의 개수가 k개인 순열의 수라고 합시다.

위 조건을 만족하는 임의의 순열에서 $n+1$이라는 수를 임의의 자리에 추가한다고 해봅시다. 위 그림과 같이 $n+1$을 두 구간의 사이에 넣는다면, 두 구간은 하나로 합쳐지게 됩니다. 그 외의 위치는 구간을 하나 증가시키게 됩니다.

이와 같은 상태 변화를 표현하면,
$D_{n,k} * (k-1) \Rightarrow D_{n+1,k-1}$

$D_{n,k} * ((n+1)-(k-1) \Rightarrow D_{n+1,k+1}$입니다.

그런데, 시작과 끝 위치는 정해져 있으므로 n+1이 cs 혹은 cf인 경우는 구간의 개수를 1개 증가시키면서 맨앞 혹은 맨뒤로 배치될 것입니다.
