---
layout: post
title: "ES-tree로 decremental SCC 관리하기"
date: 2026-04-25
author: leejseo
tags: [algorithm, graph-theory, graph-algorithm]
---

## 서론

Decremental SCC 문제는 방향 그래프에서 다음의 질의들을 빠르게 처리하는 문제이다:

* $\newcommand{\delete}{\mathrm{Delete}} \delete(u, v)$: 방향 간선 $(u, v)$를 삭제한다.
* $\newcommand{\check}{\mathrm{Check}}  \check(u, v)$: 두 정점 $u$와 $v$가 같은 SCC에 속하는지 확인한다.

Decremental single-source reachability(SSR) 문제는 방향 그래프와 고정된 정점 $s$에서 다음의 질의들을 빠르게 처리하는 문제이다:

* $\delete(u, v)$: 방향 간선 $(u, v)$를 삭제한다.
* $\check(u)$: $s$에서 출발해 $u$에 도달하는 경로가 존재하는지 확인한다.

방향 그래프에서 같은 SCC에 속하는 정점들 사이에는 서로 도달 가능하고, SCC들을 압축하면 DAG가 되므로 decremental SCC를 잘 해결할 수 있다면 decremental SSR 역시 잘 해결할 수 있을 것이다. 그동안 이 두 문제에 대한 여러 연구가 있었고, 다음과 같은 결과들이 알려져 있다. 여기서 $n$은 정점의 수, $m$은 간선의 수를 의미한다.

|                                                       | SCC                    | SSR                     | 비고                                         |
| ----------------------------------------------------- | ---------------------- | ----------------------- | -------------------------------------------- |
| Tarjan, 1972                                          | $O(m^2)$               | $O(m^2)$                | 그 유명한 SCC를 구하는 Tarjan algorithm이다. |
| Even & Shiloach, 1981                                 |                        | $O(nm)$                 | 이 글에서 다룰 ES-tree를 제시한 논문이다.    |
| Roditty & Zwick, 2008                                 | $O(nm)$                | $O(nm)$                 | ES-tree에 기반한다. randomized.              |
| Łacki, 2013                                           | $O(nm)$                | $O(nm)$                 |                                              |
| Henzinger & Krinninger & Nanongkai, 2015              | $O(mn^{0.9 + o(1)})$   | $O(mn^{0.9 + o(1)})$    | randomized.                                  |
| Chechik & Hansen & Italiano & Łacki, Parotsidis, 2016 | $\tilde{O}(m \sqrt n)$ | $\tilde{O} (m \sqrt n)$ | randomized.                                  |
| Bernstein & Probst & Wulff-Nilsen, 2019               | $O(m \log^4 n)$        | $O(m \log^4 n)$         | randomized.                                  |

$O(m^2)$ 미만의 시간에 decremental SCC를 관리하는 것은 대단히 비자명한 일이다. 이 글에서는 ES-tree에 간단한 아이디어를 얹어 $O(nm \log n)$ 시간에 decremental SCC(혹은 SSR)를 관리하는 방법을 설명한다. 이 간단한 아이디어는 놀랍게도 다음 글에서 소개할 $O(m \log^4 n)$ 시간 알고리즘과도 깊은 관련이 있다.

## ES-tree

ES-tree는 자료구조를 고안한 Even과 Shiloach의 이름을 딴 트리 자료구조로, 시작 정점을 하나 정하고 시작 정점으로부터 다른 정점들까지의 거리를 관리한다. 엄밀하게 설명하자면, 방향 그래프 $G = (V, E)$에 대해 다음의 연산을 지원한다:

* $\mathrm{InitES}(r, G, \delta)$: "루트" 정점 $r$과 $r$로부터 "거리"를 정확하게 관리할 최대 반경 $\delta$를 받아 자료구조를 초기화한다.
* $\mathrm{Distance}(r, v)$: 만약 $\mathrm{dist}_G(r, v) \le \delta$라면, $\mathrm{dist}_G(r, v)$를 반환한다. 아니라면, $\infty$를 반환한다.
* $\mathrm{Delete}(u, v)$: $E$에서 방향 간선 $(u, v)$를 삭제한다.

이때, 최종적으로 $m$개의 간선 전체를 삭제하는 동안 $\mathrm{InitES}$와 $\mathrm{Delete}$ 연산을 처리하는 데 총 $O(m \delta)$ 이내의 시간이 사용되며, 그 사이의 어느 시점에서도 $\mathrm{Distance}$ 연산은 worst-case $O(1)$에 동작한다. 이 자료구조 자체에 대한 자세한 설명은 [koosaga.com/332](https://koosaga.com/332)를 참고하면 좋다.

우리가 SCC를 관리함에도 불구하고 single-source reachability 혹은 single-source shortest path와 관련된 ES-tree에 관심을 가지는 이유는, 관리해야 하는 "반경" $\delta$가 크지 않다면 decremental query를 굉장히 효율적으로 처리할 수 있기 때문이다. 반대로 말하면, 이 $\delta$를 어떻게 작게 유지할지가 다음 문제이다.

## SCC를 2개의 ES-tree로 관리하기

방향 그래프 $G = (V, E)$에 대해 다음의 접근을 생각해볼 수 있다:

* $\mathrm{Init}$

  모든 SCC를 $O(m)$ 시간에 구한 뒤, 각 SCC $X$에 대해 루트 $r \in X$를 정한다.

  $G'$을 $G$의 역방향 그래프라고 할 때, $X$에 대한 induced subgraph $G[X]$와 $G'[X]$에서 ES-tree를 $\mathrm{InitES}(r, G[X], \delta), \mathrm{InitES}(r, G'[X], \delta)$로 초기화한다. 각각 out-tree / in-tree라고 부르자. 여기서 $X$의 지름은 $\delta$ 이하이다. 두 ES-tree는 모두 $O(m \delta)$ total update time을 가짐을 쉽게 확인할 수 있다.

* $\mathrm{Delete(u, v)}$

  만약 $u$와 $v$가 이미 다른 SCC에 속한다면, 아무것도 변하지 않는다.

  같은 SCC $X$에 속한다면, $(u, v)$를 $X$의 out-tree와 in-tree에서 제거한다. 만약 두 ES-tree가 여전히 $X$를 span한다면, $X$는 여전히 하나의 SCC를 이룬다.

  **그렇지 않고, $X$가 $X_1, X_2, \cdots, X_k$로 쪼개졌다고 하자. $r \in X_1$이라면, $X_1$에서는 $r$을 루트로 만든 ES-tree를 재활용하자. $X_2, \cdots, X_k$에 대해서만 새로 ES-tree를 만들자.**

위 접근에서 굵게 표시한 부분에 대해 다음을 관찰할 수 있다. 만약 루트 $r$을 $X$에서 *uniformly random*하게 골랐다면,

* 모든 $X_1, X_2, \cdots, X_k$의 크기가 $\frac{1}{2} \vert X \vert$ 이하이거나,
* $\frac12$ 이상의 확률로 재활용된 SCC 조각 $X_1$의 크기가 $\frac{1}{2} \vert X \vert$ 이상이다.

이 관찰 덕분에, 각 정점은 평균적으로 $O(\log n)$개의 ES-tree에만 속하고, 관리할 $\delta$의 최댓값을 $\delta_{\max}$라 할 때 total update time은 $O(m \log n \cdot \delta_{\max}) \le O(mn \log n)$이 된다.

## 결론

우리는 decremental SCC를 ES-tree를 이용해 $O(m\delta_{\max} \log n)$ 시간에 관리하는 방법을 살펴보았다. 아이디어가 간단하여 코드로 구현하는 데에도 어려움이 없을 것이다.

물론 $\delta_{\max}$는 최악의 경우 $O(n)$이지만, 매번 간선을 삭제할 때마다 Tarjan 알고리즘을 돌려서 SCC를 새로 구하는 것에 비해서는 획기적인 개선이다. 다음의 후속 질문을 자연스레 던질 수 있을 것이다:

>  $\delta_{\max}$가 커지지 않도록 하면서 ES-tree를 이용해 SCC를 관리할 수 있을까?

Bernstein & Probst & Wulff-Nilsen (2019) 논문에서는, 대략적으로 다음과 같은 아이디어를 사용한다.

1. 그래프의 hierarchy $\hat G_0, \hat G_1, \cdots$와 separator set $S_0 \supseteq S_1 \supseteq \cdots$를 구축한다.
2. $S$-distance라는 "distance"를 일반화한 개념을 도입하고, distance 대신 $S$-distance를 관리하는 generalized ES-tree를 사용하여 $S$-distance가 크지 않은 SCC 조각을 관리한다.
3. 간선의 삭제가 발생할 때, hierarchy를 적절히 업데이트하여 모든 조각의 diameter가 너무 커지지 않게 한다.

다음 글에서는 이를 시각화 자료와 함께 자세히 살펴볼 것이다.

## Reference

A. Bernstein, M. Probst and C. Wulff-Nilsen, Decremental strongly-connected components and single-source reachability in near-linear time (STOC 2019)
