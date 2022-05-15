---
layout: post
title: Probabilistic Method
author: queuedq
date: 2022-05-15
tags:
- graph-theory
- probability-theory
---

그래프 이론과 확률. 왠지 어울리지 않을 것 같은 두 개념을 처음으로 엮은 것은 헝가리의 수학자 폴 에르되시 (Paul Erdős)였습니다. 그는 확률을 이용해서 그래프 이론과 조합론 분야의 정리를 증명하는 "확률론적 방법론" (Probabilistic Method)을 창안했습니다. 이번 글에서는 확률론적 방법론이 어떤 방식의 증명 방법인지 몇 가지 정리를 통해 알아봅시다.
$
\newcommand{\Pr}{\mathbf{Pr}}
\newcommand{\Ex}{\mathbf{E}}
$

## Warm-Up: 2-Colorable Hypergraphs

다음을 증명해 봅시다.

---

**Problem.** 집합 $S$가 있고, $S$의 부분집합 $m$개가 있다. 이 부분집합들은 모두 크기가 $k$ 이상이다. $m<2^{k-1}$일 때, $S$의 원소들을 빨강이나 파랑으로 칠해서, 단색으로 칠해진 부분집합이 없도록 만드는 것이 항상 가능함을 보이시오.

---

그래프 이론의 언어로 바꿔서 말하면 다음과 같습니다.

---

**Theorem.** 하이퍼그래프 (hypergraph) $\mathcal{H}$가 $m$개의 하이퍼엣지 (hyperedge)를 가지고, 각각의 크기는 $k$ 이상이라고 하자. $m < 2^{k-1}$이면 $\mathcal{H}$는 2-colorable하다.

---

우선 정의부터 짚고 넘어갑시다. 하이퍼그래프는 그래프를 확장한 개념입니다. 기존의 그래프에서는 간선이 두 개의 정점만을 이었다면, 하이퍼그래프에서는 간선이 몇 개의 정점이든 포함할 수 있습니다. 즉, 하이퍼엣지는 정점의 부분집합으로 정의됩니다.

그래프에서 색칠 문제는 정점을 적절히 색칠해서 각 간선에 포함된 두 정점의 색이 다르도록 만드는 문제였습니다. 하이퍼그래프 색칠 문제도 마찬가지로, 각 하이퍼엣지가 단색으로 (monochromatic) 이루어지지 않도록 색칠하는 것이 목표입니다. 하이퍼그래프가 2-colorable하다는 것은 각 정점을 빨간색 또는 파란색으로 색칠해서, 빨간색 정점만으로 혹은 파란색 정점만으로 구성된 하이퍼엣지가 존재하지 않도록 만드는 것이 가능하다는 이야기입니다. 이렇게 색칠한 것을 proper (vertex) 2-coloring이라고 부릅니다.

일반적인 그래프가 2-colorable한지 판별하는 것은 간단한 그래프 탐색을 통해 가능합니다. 한 정점이 빨간색이라면 이웃한 정점은 반드시 파란색이어야 하고, 그 반대도 마찬가지이기 때문입니다. 하지만, 하이퍼그래프가 2-colorable한지 판별하는 것은 NP-complete 문제로 알려져 있습니다. 위 정리는 하이퍼그래프가 2-colorable할 충분조건을 하나 제시하고 있습니다.

왠지 적절한 알고리즘을 찾아서 규칙에 따라 색칠하고 나면 각 간선이 monochromatic하지 않음을 증명할 수 있을 것 같습니다. 하지만 직접 시도해 보면 그런 알고리즘이 쉽사리 찾아지지 않을 겁니다. 여기서 확률론적 방법론의 강력함이 드러납니다. 실제 색칠 방법을 구성하지 않고도 존재성을 증명할 수 있기 때문입니다.

### Proof

각 정점을 $\frac{1}{2}$ 확률로 빨간색 또는 파란색으로 색칠해 봅시다. 크기가 $k' \ge k$인 하이퍼엣지가 monochromatic할 확률은 얼마일까요? 각 정점을 독립적으로 색칠했기 때문에, 확률은 $p = 2 \times \left(\frac{1}{2}\right)^{k'} = \left(\frac{1}{2}\right)^{k'-1} \le \left(\frac{1}{2}\right)^{k-1}$ 이 됩니다,

이를 이용해서 그래프에 존재하는 단색 하이퍼엣지 개수의 기댓값을 구해보겠습니다. 각 하이퍼엣지 $e$에 대해서, 랜덤 변수 $X_e$를 $e$가 monochromatic할 때 $1$, 아닐 때 $0$으로 정의합시다. 단색 하이퍼엣지의 개수는 랜덤 변수 $X = \sum_{e\in E(\mathcal{H})} X_e$ 로 나타낼 수 있습니다. 기댓값의 선형성을 이용하면,

$$
\Ex[X] = \sum_{e\in E(\mathcal{H})} \Ex[X_e] = \sum_{e\in E(\mathcal{H})} \Pr(X_e = 1) \le \sum_{e\in E(\mathcal{H})} \left(\frac{1}{2}\right)^{k-1} = m \left(\frac{1}{2}\right)^{k-1} < 1
$$

임을 알 수 있습니다.

기댓값이 1보다 작다는 것은 $X=0$일 확률이 양수임을 의미합니다. 즉, 단색 하이퍼엣지가 존재하지 않는 2-coloring이 반드시 존재함을 알 수 있습니다. $\blacksquare$

---

확실한 사실을 증명하는 데에 확률을 사용하는 것이 꺼림칙하게 느껴질 수도 있습니다. 하지만 이 증명에서 확률이 나타내는 것은 무작위 시행을 하는 것이 아니라, 단순한 경우의 수의 비율에 불과합니다. 위의 증명을 다시 적으면, 모든 2-coloring에 대해서 단색 하이퍼엣지의 수의 평균을 내었더니 1보다 작더라는 이야기가 됩니다. 평균이 1보다 작으니 단색 하이퍼엣지가 0개인 2-coloring이 존재하는 것이죠. 만약 각 정점을 $\frac{1}{2}$ 이 아닌 다른 확률로 색칠했다면 평균 대신 가중치 평균이라고 생각할 수 있겠습니다.

눈여겨볼 점은, 이 증명이 실제 구성 방법에 대해서 말해주지는 않는다는 점입니다. 사실상 모든 가능성을 따져 보는 방식이기 때문에 존재성에 대해서만 알려주고 어떤 방식으로 색칠해야 하는지는 알 수 없습니다.

2-colorable하지 않은 $k$-uniform (모든 하이퍼엣지의 크기가 $k$인) 하이퍼그래프의 하이퍼엣지 수의 최솟값을 $m(k)$로 표기합시다. 위의 정리는 $m(k) \ge 2^{k-1}$임을 말하고 있습니다. 에르되시는 마찬가지로 확률론적 방법론을 이용해서 $m(k) < k^2 2^{k+1}$임을 보였습니다.

## Graphs of High Girth and High Chromatic Number

이번엔 그래프의 채색수 (chromatic number)에 대한 정리를 살펴보겠습니다. 그래프 $G$의 chromatic number란 proper coloring에 필요한 최소 개수의 색깔을 말하며, $\chi(G)$로 표기합니다.

그래프 $G$가 주어졌을 때 $k$개의 색으로 proper coloring이 가능한지 판별하는 문제는 NP-complete입니다. 하지만 그래프의 형태가 복잡하고 조밀할수록 $\chi(G)$가 크다는 사실은 유추할 수 있습니다. 예를 들어, 트리를 비롯한 이분 그래프에서는 $\chi(G) = 2$이고, 만약 그래프가 $k$-clique (크기가 $k$인 부분 완전 그래프)를 포함한다면 $\chi(G) \ge k$입니다. 또한 그래프의 최대 degree를 $\Delta(G)$라고 하면, 간단한 관찰을 통해 $\chi(G) \le \Delta(G) +1$임을 알 수 있습니다. 그렇다면 반대로, 높은 chromatic number를 달성하기 위해서는 $G$가 복잡하고 조밀한 일부분을 포함해야 하지 않을까요?

그래프가 얼마나 복잡하고 조밀한 부분을 포함하는지 말해주는 척도 중 하나가 바로 girth입니다. Girth란 그래프에서 가장 짧은 사이클의 길이를 말하며, $g(G)$로 표기합니다. 사이클이 없는 forest의 경우 $g(G) = \infty$로 정의됩니다. Girth가 큰 그래프는 국소적으로 트리 형태를 가지므로, 국소적으로는 적은 수의 색깔만으로 색칠하는 것이 가능합니다. 그렇다면 그래프 전체를 색칠하는 데 필요한 색깔의 수도 적을 것이라고 추측할 수 있습니다.

하지만 놀랍게도, $G$의 girth가 얼마나 크든 간에 $G$는 얼마든지 큰 chromatic number를 갖는 것이 가능합니다. 위에서 설명한 직관과 상반되는 사실입니다.

---

**Theorem.** (Erdős, 1959) 임의의 $k, l$에 대해서, $\chi(G) > k$이고 $g(G) > l$인 그래프 $G$가 존재한다.

---

증명에 앞서서 간단한 확률론적 사실 하나를 짚고 넘어가겠습니다. 마르코프 부등식 (Markov's inequality)이라고 불리는 정리입니다.

---

**Theorem.** (Markov's inequality) 음이 아닌 랜덤 변수 $X$와 양수 $a>0$에 대해서 다음이 성립한다.

$$
\Pr(X \ge a) \le \frac{\Ex[X]}{a}
$$

---

직관적으로 설명하면 이렇습니다. 기댓값을 계산할 때 $X \ge a$인 항들만 생각하면, $a$ 이상의 값들이 도합 $\Pr(X \ge a)$의 확률로 곱해집니다. 따라서 $\Ex[X] \ge a \cdot \Pr(X \ge a)$를 만족하고, 이를 정리하면 위의 부등식을 얻을 수 있습니다.

### Proof

랜덤 그래프 $G_{n,p}$를 $n$개의 정점이 있고, 각 정점 사이에 간선이 있을 확률이 $p$인 그래프로 정의하겠습니다.

증명의 개요는 다음과 같습니다.

- 충분히 큰 $n$과 적당한 확률 $p$에 대해서, $G_{n,p}$에 길이가 짧은 사이클의 수가 적을 확률이 절반보다 큼을 보입니다.
- 마찬가지로, $G_{n,p}$의 chromatic number가 높을 확률이 절반보다 큼을 보입니다.
- 각각의 확률이 절반보다 크므로, 짧은 사이클의 수가 적으면서 chromatic number가 큰 그래프가 존재합니다. 이렇게 만들어진 그래프에서 몇 개의 정점을 제거해서, chromatic number는 높게 유지한 채 길이가 $l$ 이하인 사이클을 모두 제거합니다.

첫 번째 과정을 위해 먼저 길이가 $i$인 사이클을 세어 봅시다. 정점 $i$개의 순열을 생각했을 때 사이클을 이룰 확률은 $p^i$이고, 이러한 순열의 개수는 $n^i$ 이하입니다. (서로 다른 순열이 같은 사이클을 나타낼 수 있음을 고려하면, 사이클을 이룰 수 있는 경우의 수는 더 작아집니다.) 따라서, 길이가 $l$ 이하인 사이클의 수를 $X$라고 하면

$$
\Ex[X] \le \sum_{i=3}^{l} n^i p^i
$$

가 성립합니다. 계산의 편의를 위해 $p = n^{\lambda-1}$으로 두겠습니다. 그러면

$$
\Ex[X] \le \sum_{i=3}^{l} n^{\lambda i} \le \frac{n^{\lambda l}}{1 - n^{-\lambda}}
$$

입니다. $\lambda l < 1$이 되도록 $\lambda$를 잡으면 우변이 $o(n)$이므로, 충분히 큰 $n$에 대해 $\Ex[X] < \frac{n}{4}$입니다. Markov's inequality에 따르면 $\Pr(X \ge \frac{n}{2}) < \frac{1}{2}$임을 알 수 있습니다. 따라서 절반보다 큰 확률로 $X < \frac{n}{2}$입니다.

이번에는 절반보다 큰 확률로 chromatic number가 크다는 사실을 보일 차례입니다. 그래프의 coloring은 다루기 까다로우므로, 비교적 간단한 문제로 변형해 봅시다. $k$가지 색으로 그래프를 색칠하는 것은 정점 집합을 $k$개의 독립 집합 (independent set, 서로 인접하지 않은 정점의 집합)으로 분할하는 것으로 생각할 수 있습니다. 따라서, 최대 독립 집합의 크기를 $\alpha(G)$라고 했을 때, $\chi(G)\alpha(G) \ge n$이 성립합니다. 즉, $\alpha(G) < \lceil\frac{n}{k}\rceil$를 보이면 $\chi(G) > k$임을 알 수 있습니다.

최대 독립 집합의 크기가 $a$ 이상일 확률 $\Pr(\alpha(G_{n,p}) \ge a)$를 계산해 봅시다. 최대 독립 집합의 크기가 $a$ 이상이라는 것은 크기가 $a$인 독립 집합이 존재한다는 것과 동치입니다. 크기가 $a$인 집합의 수는 $\binom{n}{a} \le n^a$이고 각 집합이 독립 집합일 확률은 $(1-p)^{a(a-1)/2} < e^{-pa(a-1)/2}$이므로,

$$
\Pr(\alpha(G_{n,p}) \ge a) \le \binom{n}{a} (1-p)^{a(a-1)/2} < n^a e^{-pa(a-1)/2}
$$

입니다. $n \to \infty$일 때 위 식이 0에 수렴하도록 만들기 위해 $a = \lceil \frac{3}{p} \ln n \rceil$으로 두면,

$$
\Pr(\alpha(G_{n,p}) \ge a) < n^a n^{-3(a-1)/2} = n^{-a/2+3/2} \to 0
$$

임을 알 수 있습니다. 따라서 충분히 큰 $n$에 대해 $\Pr(\alpha(G_{n,p}) \ge a) > \frac{1}{2}$입니다.

각각의 확률이 절반보다 크므로, $X < \frac{n}{2}$이면서 $\alpha(G) < a$인 그래프 $G$가 존재합니다. $X$개의 짧은 사이클 각각에서 최대 한 개의 정점을 제거하면 길이가 $l$ 이하인 사이클이 없는 그래프 $G'$을 얻을 수 있습니다. 즉, $\lvert V(G')\rvert > \frac{n}{2}$이고 $g(G') > l$입니다. 한편, 정점을 제거하는 것은 최대 독립 집합의 크기를 늘리지 않으므로, $\alpha(G') \le \alpha(G) < a$입니다. 마지막으로 이 그래프의 chromatic number를 계산해 봅시다.

$$
\chi(G') = \frac{\lvert V(G')\rvert}{\alpha(G')} > \frac{n/2}{a} \ge \frac{n/2}{(3\ln n) / p} = \frac{n/2}{3n^{1-\lambda}\ln n} = \frac{n^\lambda}{6\ln n}.
$$

이는 $\lambda$가 얼마나 작든, $n \to \infty$일 때 무한대로 발산합니다. 따라서 충분히 큰 $n$을 잡으면 $\chi(G') > k$인 그래프 $G'$를 찾을 수 있습니다. $\blacksquare$

---

이 증명 역시 girth와 chromatic number가 큰 그래프의 구성 방법을 말해주지는 않습니다. 실제로 이 정리가 증명된 이후 10년 동안이나 확률적인 방법을 사용하지 않는 construction이 나오지 않았다고 합니다. 확률론적 방법론의 장점이자 단점입니다.

## 마무리

이번 글에서는 확률론적 방법론을 통해 그래프 이론의 정리들을 증명하는 방법을 알아보았습니다. 여기서는 확률과 기댓값을 사용하는 증명만을 소개했지만, 분산이나 Lovász local lemma 등 더 깊이 있는 확률론적 도구를 사용하는 증명들도 있습니다. 관심 있는 독자들은 참고 자료 [1, 2, 3]에 있는 책과 강의 노트를 참고하시기 바랍니다.

## 연습 문제

1. $\frac{n!}{2^{n-1}}$개 이상의 해밀턴 경로를 가지는 크기 $n$의 tournament (완전 방향 그래프)가 존재함을 증명하시오.
2. $m$개의 간선을 가지는 그래프 $G$가 $\frac{m}{2}$개 이상의 간선을 가지는 이분 그래프를 부분 그래프로 가짐을 증명하시오.
3. $k\ge 3$과 $n=\lfloor\sqrt{2}^k\rfloor$에 대해서, $n$개의 정점을 가진 완전 그래프의 간선을 빨간색 또는 파란색으로 색칠하자. 크기가 $k$인 단색 clique가 존재하지 않도록 색칠할 수 있음을 증명하시오. (Erdős, 1947)
4. $\alpha(G) \ge \sum_{v\in V(G)} \frac{1}{\deg(v)+1}$임을 증명하시오. (Caro (1979), Wei (1981))
5. 평면 위에 점 10개가 있다. 점들이 어떻게 배치되어 있든, 반지름이 1인 동전 10개를 겹치지 않게 놓아서 이 점들을 전부 덮을 수 있음을 증명하시오.

## 연습 문제 힌트

- 4번: 정점들의 무작위 순열을 생각해 봅시다. 어떤 기준으로 정점들을 고르면 독립 집합을 만들 수 있을까요?
- 5번: 무한한 동전으로 평면을 채워 봅시다.

## 참고 자료

1. Bruce Reed, Michael Molloy. [Graph Colouring and the Probabilistic Method](https://link.springer.com/book/10.1007/978-3-642-04016-0)
2. Noga Alon, Joel H. Spencer. [The Probabilistic Method](https://www.wiley.com/en-us/The+Probabilistic+Method%2C+4th+Edition-p-9781119061953)
3. Jiří Matoušek, Jan Vondrák. [The Probabilistic Method Lecture Notes](https://www.cs.cmu.edu/~15850/handouts/matousek-vondrak-prob-ln.pdf)
4. Paddy. [Probabilistic Methods in Graph Theory / Lecture 2: High Girth and High Chromatic Number](https://web.math.ucsb.edu/~padraic/mathcamp_2010/class_graph_theory_probabilistic/lecture2_girth_chromatic.pdf)
5. Ehssan Khanmohammadi. [The Probabilistic Method In Graph Theory](http://sites.psu.edu/ehssan/wp-content/uploads/sites/7257/2013/10/The-Probabilistic-Method.pdf)
6. [Probabilistic method - Wikipedia](https://en.wikipedia.org/wiki/Probabilistic_method)
7. [Graph coloring - Wikipedia / Graphs with high chromatic number](https://en.wikipedia.org/wiki/Graph_coloring#Graphs_with_high_chromatic_number)
