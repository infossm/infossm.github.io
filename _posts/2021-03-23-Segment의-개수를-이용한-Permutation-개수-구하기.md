---
layout: post
title: Segment의 개수를 이용한 Permutation 개수 구하기
date: 2021-03-23 04:00
author: rdd6584
tags: [algorithm, mathematics]
---

## 소개
특정 조건을 만족하는 순열의 개수를 세는 문제 중에서, segment(이하 구간)의 개수를 이용하는 특이한 꼴의 다이나믹 프로그래밍 문제를 다뤄보려고 합니다. 예시 문제를 통해 살펴봅시다.


### 길이가 N, Increasing segment의 개수가 K개인 순열의 개수
Increasing segment란, $l+1 \leq i \leq r$를 만족하는 $i$에 대해 $A_{i-1} < A_i$를 모두 만족하는 $[l, r]$이라고 합니다. 단, 다른 increasing segment에 포함되는 구간은 무시하는 걸로 합시다.

예를 들어, $1\space4\space5\space2\space3\space6\space8\space7$의 increasing segment는 [1, 3], [4, 7], [8, 8]로 총 3개입니다.

우리는 길이가 $N$이고 increasing segment의 개수가 $K$개인 순열의 개수를 구하려고 합니다. 이 문제는 어떻게 해결할 수 있을까요?

이 문제는 다이나믹 프로그래밍을 이용하여, $O(NK)$에 해결할 수 있습니다.
$D_{n,k}$를 길이가 $n$이고 수 $1~n$만을 사용하며, increasing segment의 개수가 $k$개인 순열의 개수라고 합시다.
위 조건을 만족하는 임의의 순열에서  $n+1$이라는 수를 임의의 자리에 추가한다고 생각해봅시다. 경우의 수는 크게 아래와 같이 4가지로 분류할 수 있습니다.


<img src="/assets/images/rdd6584_1/segper01.png" width="100%" height="100%">



1. n+1을 넣으려는 위치의 왼쪽 값이 오른쪽보다 큰 경우입니다. 이 경우 increasing segment의 개수가 증가하지 않습니다.
2. n+1을 넣으려는 위치의 왼쪽 값이 오른쪽보다 작은 경우입니다. 이 경우 increasing segment의 개수가 증가합니다.
3. n+1을 순열의 맨 왼쪽에 넣는 경우입니다. 이 경우 increasing segment의 개수가 증가합니다.
4. n+1을 순열의 맨 오른쪽에 넣는 경우입니다. 이 경우 increasing segment의 개수가 증가하지 않습니다.

increasing segment의 개수를 증가시키지 않는 1번 경우는 k-1개, 4번 경우는 1개로 총 k개의 경우가 존재합니다. increasing segment의 개수를 증가시키는 경우는 n+1-k개일 것입니다. 따라서, 아래와 같은 상태를 얻을 수 있습니다.

$D_{1,1} = 1$
$D_{n,k} x k --> D_{n+1, k}$
$D_{n,k} x (n+1-k) --> D_{n+1, k+1}$
이 점화식을 이용하면 우리는 최종적으로 $D_{N,K}$를 $O(NK)$만에 구할 수 있게 됩니다.

