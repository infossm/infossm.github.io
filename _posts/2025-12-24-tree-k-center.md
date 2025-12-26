---
layout: post
title: "K-Center Problem in Tree"
date: 2025-12-24
author: jinhan814
tags: [algorithm, graph-theory]
---

## 1. Introduction

k-center problem은 그래프에서 $k$개의 센터를 선택하여 모든 정점으로부터 가장 가까운 센터까지의 거리의 최댓값을 최소화하는 최적화 문제입니다. 이는 도시 내에 소방서나 응급 의료 센터와 같은 필수 시설을 배치할 때, 가장 가까운 시설까지의 거리가 일정 수준을 넘지 않도록 해야 하는 상황에서 주로 사용됩니다.

일반 그래프에서의 k-center problem은 NP-hard로 알려져 있으며, $P \neq NP$ 가정 하에 2-approximation보다 나은 근사 알고리즘 또한 NP-hard임이 증명되어 있습니다. 때문에 일반 그래프에서는 근사 알고리즘을 이용한 접근이나, 그래프의 특수한 구조를 이용하는 접근이 주로 시도됩니다.

그래프가 트리라면 두 정점 사이의 경로가 유일하므로, k-center problem을 전역적인 최적화 문제를 국소적인 최적화 문제로 환원할 수 있습니다. 이때 국소적인 최적화 문제는 greedy하게 해결할 수 있어 트리에서의 k-center problem은 다항 시간 내에 정확한 해를 구할 수 있습니다.

트리의 k-center problem은 문제 상황에 따라 여러 변형을 가집니다. 대표적으로 간선 가중치의 존재 여부나 센터를 정점에만 둘 수 있는지 혹은 간선의 중간에도 둘 수 있는지에 따른 변형이 존재합니다. 이 글에서는 기본적인 트리의 k-center problem을 시작으로, 이러한 변형 문제를 해결하는 방법을 단계적으로 알아보겠습니다.

## 2. K-Center Problem In General Graph

그래프의 k-center problem은 일반 그래프 $G = (V, E)$에서 $k$개의 센터를 선택하여, 모든 정점으로부터 가장 가까운 센터까지의 거리의 최댓값을 최소화하는 문제입니다.

$$\min_{S \subseteq V, |S| = k} \max_{v \in V} \min_{s \in S} \text{dist}(v, s)$$

그래프의 k-center problem이 어려운 이유는 그래프에서 한 정점의 선택이 다른 모든 정점에 미치는 영향이 서로 복잡하게 얽혀있기 때문입니다. 한 정점에서 다른 정점으로 이동하는 경로는 여러 개가 존재하기에, 일반 그래프에서는 거리의 전파가 국소적인 구조로 분해되지 않습니다.

![Fig.1](/assets/images/2025-12-24-tree-k-center/fig1.png)

예를 들면, 위와 같은 그래프에서 $k = 3$인 k-center problem의 최적해는 $2$입니다. 가능한 $|S| = 3$이면서 $\max_{v\in v}\min_{s\in S}\text{dist}(v, s)$가 최소인 집합 $S$로는 $\{ 1, 3, 7 \}, \{ 1, 4, 7 \}, \{ 2, 3, 7 \}, \{ 2, 4, 7 \}$이 있습니다.

일반 그래프에서 k-center problem은 다항 시간 풀이가 알려져 있기 않기에 그 자체로는 자주 출제되지는 않으며, 추가 조건이 있는 경우가 많습니다. 예시로는 $k = 1$이고 $\text{dist}$ 함수가 metric이라는 조건을 추가한 [BOJ 19352](https://www.acmicpc.net/problem/19352), [4360](https://www.acmicpc.net/problem/4360), [20631](https://www.acmicpc.net/problem/20631)이 있습니다.

## References

[1] [~](~)