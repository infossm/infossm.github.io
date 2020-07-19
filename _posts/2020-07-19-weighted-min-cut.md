
---
layout: post
title: "Weighted Min-Cut: Sequential, Cut-Query and Streaming Algorithms"
author: koosaga
date: 2020-07-19
tags: [graph-theory, trees, random-algorithm]
---

# Weighted Min-Cut: Sequential, Cut-Query and Streaming Algorithms

그래프의 최소 컷 (Minimum cut) 은 그래프를 연결되지 않게 하기 위해서 지워야 하는 간선의 최소 개수, 혹은 간선 가중치의 최소 합이다. 만약 간선의 최소 개수로 컷을 정의한다면, 최소 컷은 그래프의 [connectivity](https://en.wikipedia.org/wiki/Connectivity_(graph_theory)) 를 정의하는 수량이 된다. 고로 최소 컷은 그래프가 주어졌을 때 계산하고 싶은 가장 기초적인 수량에 해당되며, 응용 예시 또한 무수히 많다.

그래프의 최소 컷을 계산하는 방법은 크게 3가지가 있다. 아래에 해당 방법의 발견 시간 순으로 나열한다. (아래 요약은 [이 논문에서](https://arxiv.org/pdf/1911.01145.pdf) 따왔다.)

* Min-cut Max-flow 접근. Global min-cut은 Global $s - t$ 컷 중 최소이니 이를 반복적으로 찾는 방법이다. 단순하게는 $\frac{n(n-1)}{2}$ 번 컷을 찾아야 한다. 하지만 Gomory, Hu가 $O(n)$ 번의 컷 계산으로 문제를 해결할 수 있음을 보였다 (*Gomory-Hu Tree*). 이보다 적은 횟수의 컷 계산으로 문제를 해결하는 방법은 알려지지 않았다. $s - t$ 최소 컷을 찾는 것은 비자명하지만, Min-cut max-flow theorem을 사용하여 플로우 문제로 변환하면 다항 시간에 계산할 수 있다. 현대 기술로는 플로우를 굉장히 빠르게 계산할 수 있기는 하지만, 그럼에도 $O(n)$ 번의 플로우 계산이 필요하여 시간을 줄이는 데 한계가 있다.

* Edge contraction 접근. 최소 컷을 가로지르지 않는 간선을 찾아낼 수 있다면, 이 간선을 contraction 시켜도 된다. 이런 식으로 정점 개수를 줄이면서 문제를 해결할 수 있다. 블로그에서도 소개한, [Karger-Stein algorithm](https://koosaga.com/71), [Stoer-Wagner algorithm](https://koosaga.com/192) 이 이런 류의 알고리즘에 속한다. 이 알고리즘들은 $O(nm)$ 근처의 시간에 작동하니, Max-flow가 선형 시간에 풀리지 않는 한 위 접근보다 우수하다. 여전히 선형 시간에 근접하는 방법은 알려지지 않았다.

* Tree packing 접근. Nash-Williams Theorem에 의해서, 최소 컷이 $2k$ 이상인 그래프에서는 $k$ 개의 Edge-disjoint spanning tree를 찾을 수 있다. 이는 블로그의 [Matroid intersection 연습 문제](https://koosaga.com/252)에 간략히 소개되어 있다. 최소 컷이 $c$ 인 그래프에서, 최대 개수의 Edge-disjoint spanning tree로 그래프를 packing하자. 스패닝 트리의 개수는 $\frac{c}{2}$개 이상이니, 이 중 한 스패닝 트리는 최소 컷과의 교집합이 2개 이하일 것이다. 그렇다면 그 스패닝 트리에서 2개 이하의 간선을 끊어 나올 수 있는 모든 컷을 시도해 보면 된다. 이제 다음과 같은 4가지 문제들이 생긴다.

  * Weighted graph에서는 Packing 접근이 가능할까? Nash-Williams theorem은 가중치가 붙었을 때 성립하지 않는다.
  * Packing을 하는 알고리즘은 무엇일까? 가장 단순한 다항 시간 접근인 Matroid Union은 매우 느리다. Gabow는 STOC'91 에서 매트로이드적 접근을 Dynamic tree와 결합한 $O(mc \log m)$ packing algorithm을 제안하였으나 이는 $c$ 에 비례하는 시간이 든다.
  * Packing을 한 후 $c$ 개의 스패닝 트리를 전부 Guess해야 하지 않을까? 이 역시 $c$ 에 비례하는 시간을 요구한다.
  * 스패닝 트리에서 2개 이하의 간선을 끊어 나올 수 있는 모든 컷을 어떻게 빠르게 시도할까? 단순한 접근은 모든 간선을 끊은 후 분리된 서브트리를 컷으로 시도해 보는 전략으로, $O(n^2m)$ 시간이 든다. 

  STOC'96 에서 Karger는 이 4가지 문제를 전부 해결하고 $O(m \log^3 n)$ 시간에 작동하는 [랜덤 알고리즘을 제안했다](https://arxiv.org/pdf/cs/9812007.pdf). 고로 이 접근은 위 두 접근과 다르게 선형 시간에 근접한 알고리즘이 존재한다. Weighted graph일 경우, 간선의 가중치에 비례하는 개수의 중복 간선이 있다고 생각한 후, Gabow의 알고리즘을 랜덤 샘플링과 결합해서 $O(m + n\log^3 n)$ 시간에 tree packing을 구성한다. 이 tree packing은 $c$개가 아니라, 중요한 $O(\log n)$개의 스패닝 트리만 샘플해서 반환하기 때문에 전부 시도해 볼 수 있다. 마지막 루틴은 $O(m \log^2 n)$ 시간에 해결 가능하다. 이 루틴은 이 글의 핵심 주제이니 나중에 자세히 설명한다. 

위에서 소개한 부분문제인, 그래프에 스패닝 트리 $T$ 가 주어졌을 때, $T$ 와의 간선 교집합이 2개 이하인 컷 중 최소 크기의 컷을 찾는 문제를 **2-respecting min-cut** 이라고 부른다. 이 2-respecting min-cut을 찾는 시간이 $T_{rsp}(n, m)$ 이라고 하자. 위에서 간략히 설명한 Karger의 논문의 결과를 정확히 설명하면 다음과 같다.

**Theorem.** 가중치 있는 그래프가 주어졌을 때, 2-respecting minimum cut을 찾는 알고리즘의 복잡도가 $T_{rsp}(n, m)$ 이라면, $O(T_{rsp}(n, m)  \log n + m + n \log^3 n)$ 시간, 혹은 $O(T_{rsp}(n, m) \frac{\log n}{\log \log n} + n \log^6 n)$ 시간에 랜덤 알고리즘을 통해서 최소 컷을 찾을 수 있다 (Karger 2000).

이후 Karger는 Link-cut tree와 *Bough decomposition* 을 사용하여 다음과 같은 결과를 유도했다. Bough decomposition에 관해서는 [다음 문제를 참고하는 것이 도움이 될 것이다.](https://atcoder.jp/contests/apc001/tasks/apc001_h)

**Theorem.** 2-respecting min-cut은 $O(m\log^2 n)$ 시간에 찾을 수 있다. (Karger 2000)

이후 나온 최소 컷에 관한 state-of-the-art 내용을 요약하면 다음과 같다.

* Randomized, simple: $O(m \log n)$, $O(m + n \log^2 n)$ ([GNT20](https://arxiv.org/pdf/1909.00844.pdf), [GMW19](https://arxiv.org/pdf/1911.01145.pdf))
* Randomized, weighted: $O(m \log^2 n)$, $O(m\frac{\log^2 n}{\log \log n} + n\log^6 n)$ ([GMW19](https://arxiv.org/pdf/1911.01145.pdf))
* Deterministic, simple: $O(m\log^2 n \log \log^2 n)$ ([HRW17](https://arxiv.org/abs/1704.01254)) 
* Deterministic, weighted: $O(mn)$ (Nagamochi, Ibaraki 1992)

## Goal of this article

STOC 2020에서는, 2-respecting min cut을 $O(m \log n + n \log^4 n)$ 시간에 찾는 [알고리즘에 대한 논문](https://arxiv.org/pdf/1911.01651.pdf)이 소개되었다. Karger의 첫 알고리즘은 $O(m \log^2 n)$ 에 작동하니, dense graph에 대해서는 이보다 더 효율적으로 작동하는 것이다. 이 글에서는 이 알고리즘에 대해 소개한다. 

이 글에서 소개한 내용은 20년 만에 Karger의 알고리즘보다 빠르게 작동하는 첫 알고리즘이지만, 이 논문이 arxiv에 제출된 그 다음날에 2-respecting min cut을 $O(m \log n)$ 에 찾는 논문 ([GMW19](https://arxiv.org/pdf/1911.01145.pdf)) 이 arxiv에 제출되었다. 고로 소개할 논문은 더 이상 2-respecing min cut을 찾는 가장 효율적인 알고리즘이 아니다. 

이 논문의 의의는 그 접근 방법이 Karger와 GMW19과 다르다는 것이다. 스패닝 트리가 $T$ 상의 두 간선 $e, f$ 를 잘랐을 때의 컷의 값을 $cut(e, f)$ 라고 정의하자. 2-respecting min cut을 계산하는 가장 자연스러운 알고리즘은 이 $cut(e, f)$ 를 모든 $e, f \in E(T)$ 쌍에 대해서 계산하는 것이고, 이 논문 외의 모든 알고리즘은 암시적으로라도 $cut(e, f)$ 의 값을 모두 계산한 후 최솟값을 반환한다. 하지만 이 논문에서는 $\tilde{O}(n)$ 개의 중요한 2-respecting min cut 후보를 나열한 후 이들에 대해서만 값을 계산한다. 

이 상황을 수학적으로 엄밀하게 정의하기 위해서 *cut-query* 라고 불리는, 다음과 같은 계산 모델을 생각해 보자. 

* 가중치 있는 그래프 $G$가 있고 $G$ 의 스패닝 트리 $T$ 가 주어진다. 당신은 $T$ 는 알지만 $G$는 전혀 모른다. 한편, 당신은 $S \subseteq V(T)$ 가 주어졌을 때 $S$ 와 $V(T) \setminus S$ 간을 잇는 간선의 가중치 합을 알 수 있다. 이를 $\Delta(S)$ 라고 하자. (고로 자연스럽게 $cut(e, f)$ 도 알 수 있다). $G$ 의 2-respecting min-cut을 찾을 수 있을까?

이 논문은 이 계산 모델에 대한 첫 $\tilde{O}(n)$ 알고리즘을 제시한다. 고로, 만약에 해결하려는 인스턴스에서 그러한 컷의 값을 매우 빠르게 계산할 수 있다면 (대표적으로, 그래프의 구조가 특수하거나, parallel한 계산이 가능할 때) 이 알고리즘의 가치가 커진다. 

일반적인 sequential model에서도, 2차원 자료구조를 사용하여 주어진 2-respecting cut에 대한 값을 $O(\log n)$ 에 구할 수 있음이 잘 알려진 사실이다. 논문에 나온 알고리즘은, 이렇게 제한적인 계산 모델에 기반하여 제시되었음에도 불구하고 dense graph에서 2-respecting cut을 계산하는 기록을 갱신할 수 있었다. 

## An schematic algorithm for 2-respecting min-cut

루트 있는 트리 $T$ 에 대해서 $v^{\downarrow}$ 는 $v$ 를 루트로 하는 서브트리 정점 집합을 뜻한다. $e = (par(v), v)$ 라면, $e^{\downarrow}$ 는 $v^{\downarrow}$ 이다. 또한, 주어진 스패닝 트리는 모두 임의의 정점을 루트로 한 rooted tree로 간주한다. 그래프가 주어졌다고 생각하지 않고, cut-query 모델이라고 생각한다. 즉, $cut(e, f)$ 를 계산하는 알고리즘이 무엇인지 신경쓰지 않는다 (Chapter 3에서 다룰 것이다). 그냥 이러한 값을 계산하는 알고리즘 (oracle) 이 존재하고, 이 알고리즘을 호출하는 횟수를 토대로 알고리즘의 시간 복잡도를 분석한다.

또한, 스패닝 트리 $T$ 가 주어졌을 때 $T$ 의 간선과의 교집합이 1개 이하인 최소 컷인 *1-respecting min-cut* 을 모두 $O(n)$ 번의 호출로 찾았다고 생각하고 넘어가자. 만약 컷과 $T$ 와의 교집합이 아주 비어있다면, 컷을 지운 이후에도 그래프에 스패닝 트리가 여전히 존재하니, 컷이라는 가정 자체에 모순이 되고, 고로 교집합이 정확히 1개인 경우만 고려하면 된다.

### 1.1 When $T$ is a path

먼저 $T$ 가 경로인 특수 경우에 대해서 문제를 해결해 보자. 이 경우 우리는 $T$ 에 대해서 분할 정복을 시도한다. 배열에 분할 정복을 하듯이, $T$ 의 구간 $[l, r]$ 에서 두 개의 간선을 끊어서 얻을 수 있는 컷의 최솟값을 계산하는 재귀 함수를 생각하자. 이러한 재귀 함수를 만들면 $T$ 의 전체를 호출함으로써 분할 정복을 할 수 있다.

분할 정복을 할 때는, 구간의 중점인 정점 $r$ 을 잡고, 끊는 두 간선이 

* 모두 $r$ 의 왼쪽에 있는 경우를 재귀적으로 해결하고
* 모두 $r$ 의 오른쪽에 있는 경우를 재귀적으로 해결하고
* $r$ 을 기준으로 왼쪽에 하나, 오른쪽에 하나 있는 경우를 **효율적** 으로 해결

한다. 우리는 세 번째 경우를 $O(n \log n)$ 번의 $cut(e, f)$ 쿼리만으로 해결할 것이다. 나머지 경우를 귀납적으로 해결하면, 마스터 정리에 의해 이 케이스에서 오라클을 최대 $O(n \log^2 n)$ 번 호출한다. 

![i1](http://www.secmem.org/assets/images/mincut/i1.png)

$r$ 을 기준으로 끊게 될 에지를 $e_1, e_2, \ldots, e_n, e^{\prime}_1, e^{\prime}_2, \ldots, e^{\prime}_m$ 이라고 하자. $r$ 을 기준으로 $e$ 는 번호가 왼쪽에서 오른쪽으로 매겨져 있고 $e^{\prime}$ 은 번호가 왼쪽에서 오른쪽으로 매겨져 있다. 물론, 번호는 우리가 관심있는 $T$ 의 구간 $[l, r]$ 안에서만 매긴다. $F(i, j) = cut(e_i, e^{\prime}_j)$ 라고 하자. 이제 다음과 같은 사실을 증명한다.

**Theorem 1.1.** 모든 $1 \le i \le n-1, 1 \le j \le m-1$ 에 대해, $A(i, j) + A(i+1, j+1) \le A(i, j+1) + A(i+1, j)$ 를 만족하면, $A$ 를 Monge array라고 하자. $F$ 는 Monge array이다. 

이를 증명하기 위해 다음과 같은 Lemma를 사용한다. 이후에도 자주 사용할 것이다. 증명은 생략한다.

**Lemma 1.2**. 어떠한 간선 $g = (u, v)$가 $cut(e, f)$ 에 속한다는 것은, $T$ 상에서 $(u, v)$ 를 잇는 unique path 상에 $e, f$ 중 정확히 하나가 있다는 것과 동치이다.

**Proof of Theorem 1.1.** $F_e(i, j)$ 를, 어떠한 $T$ 상에 속하지 않은 간선 $e$ 가 $F(i, j)$ 에 기여하는 값으로 정의하자. 즉, $F(i, j) = \sum_{e \in E(G) - E(T)} F_e(i, j)$ 이다. Monge array의 합은 Monge array이기 때문에 (자명), $F_e(i, j)$ 각각이 Monge array임을 증명하면 된다. 세 가지 케이스가 있다.

* Case 1. 간선이 이루는 구간이 $r$ 을 포함하지 않으며 그의 왼쪽에 있음
* Case 2. 간선이 이루는 구간이 $r$ 을 포함
* Case 3. 간선이 이루는 구간이 $r$ 을 포함하지 않으며 그의 오른쪽에 있음

각 케이스에 따른 Monge array의 기여도를 그림으로 표현하면 이렇다 (왼쪽에서 오른쪽으로 순서대로 Case 1, 2, 3). 왜 이렇게 나오는지는 Lemma 3을 사용하여 확인할 수 있다. Monge array의 정의를 사용하면, 이 세 가지 케이스에서 만들어지는 행렬이 모두 Monge array임을 확인할 수 있다. 

![i2](http://www.secmem.org/assets/images/mincut/i2.png)

**Theorem 1.3.** $n \times m$ Monge array의 최솟값은 $O((n + m) \log n)$ 번의 $F$ 호출을 통해 찾을 수 있다.

**Proof of Theorem 1.3.** 프로그래밍 대회에서 [Divide and Conquer optimization이라는 이름으로 잘 알려진 내용이다](https://koosaga.com/242). 간략히 다시 소개하자면, $GlobalMin([i_1, i_2], [j_1, j_2]) = min_{i \in [i_1, i_2], j \in [j_1, j_2]} F(i, j)$ 라고 하자. 이 함수를 호출하면 전체 최솟값을 알 수 있다. $m = (i_1 + i_2)$ 라고 할 때, $min_{j \in [j_1, j_2]} F(m, j)$ 를 직접 계산하고, 최솟값이 나온 argument $j_{opt}$도 계산하자. 이제 Monge array의 정의에 따라서 $[i_1, m - 1]$ 구간에서 최솟값은 $j_{opt}$ 혹은 그 이후 구간에서 나오며, $[m + 1, i_2]$ 구간에서 최솟값은 $j_{opt}$ 혹은 그 이전 구간에서 나온다. 이를 토대로 분할정복에서 $j$ 로 가능한 구간을 줄이면, 각 분할 정복 레벨에서 최대 $O(n + m)$ 개의 값만 호출하게 된다.

**Remark.** SMAWK 알고리즘을 사용하면 $O(n+m)$ 번의 호출도 가능하지만 논문 저자는 non-sequential model에 적합한 알고리즘을 위해 SMAWK를 사용하지 않는다고 밝혔다.

### 1.2. When $T$ is a star graph

이제 $T$ 가 성게 (star graph) 일 때를 해결하자. star graph는 모든 정점이 루트거나 루트에 인접한 형태인 트리를 뜻한다. 즉, 루트를 제외한 모든 노드가 리프이다. $deg(i)$ 를, $i$ 에 인접한 모든 간선의 가중치 합으로 정의하고, $C(i, j)$ 를, $i$ 와 $j$를 잇는 간선의 가중치 (없으면 0) 이라고 정의하자. $e_i$를 $i$ 와 루트를 잇는 간선이라고 정의하자 ($e_{root}$ 는 정의되지 않는다).

**Observation 1.4**. 모든 minimum 2-respecting cut가 정확히 두 개의 트리 에지를 포함한다고 가정하자. 2-respecting min cut을 주는 두 간선 쌍이 $(e_i, e_j)$ 라면, $deg(i) < 2C(i, j), deg(j) < 2C(i, j)$ 가 성립한다.

**Proof.** $cut(e_i, e_j) = deg(i) + deg(j) - 2C(i, j)$ 이다. $deg(i)$ 는 $e_i$ 를 교집합으로 하는 1-respecting cut의 값이므로, $cut(e_i, e_j) < deg(i)$ 가 성립한다. 같은 이유로 $cut(e_i, e_j) < deg(j)$ 이다. 이에 위 등식을 대입하면 된다.

모든 minimum 2-respecting cut이 두 개의 트리 에지를 포함하지 않으면 그냥 1-respecting cut을 찾는 단계에서 모든 것이 끝났으므로 더 신경쓸 것이 없다. 고로 가정이 참이라고 하자. 어떠한 간선 $e_i$를 포함하는 minimum 2-respecting cut을 찾고 싶다면, $deg(i) < 2C(i, j)$ 를 만족하는 $j$ 에 대해서만 $cut(e_i, e_j)$ 를 확인해 보면 된다. 그런데, $\frac{deg(i)}{2} < C(i, j)$ 인 $j$는 많아야 하나밖에 존재할 수 없다. 두 개 이상일 경우 $deg(i)$ 의 값을 넘어가기 때문이다. 그러한 $j$ 는 $i$에 인접한 가장 가중치가 큰 간선이다. 만약 이러한 $j$를 찾았다고 치면, $cut(e_i, e_j)$ 로 호출해야 할 것이 하나밖에 없으니 $O(n)$ 개의 호출로 문제를 해결할 수 있다. 

이제 이러한 $j$ 를 찾는 방법을 서술한다. cut-query 모델에서는, 서로소인 두 정점 부분집합 $A, B \subseteq V(G), A \cap B = \emptyset$ 에 대해서, 한 끝점이 $A$ 에 있고, 다른 한 끝점이 $B$ 에 있는 간선의 가중치 합 $between(A, B)$ 을 3번의 cut-query로 구할 수 있다. $\frac{\Delta(A) + \Delta(B) - \Delta(A + B)}{2}$ 가 그 값이기 때문이다. 이제 $i$ 에 대해서, 그러한 $j$ 를 찾는 것은 이분 탐색으로 가능하다. 가능한 $j$ 의 후보 $S$ 를 $S_1, S_2$ 로 나눈 후, $between(\{i\}, S_1), between(\{i \}, S_2)$ 를 계산하고 큰 쪽으로 움직이면 되기 때문이다. 고로 $O(n \log n)$ 개의 호출로 문제가 해결된다.

### 1.3 General tree $T$: Orthogonal pair of edges

여기서도 Observation 1.4와 같이 모든 minimum 2-respecting cut가 정확히 두 개의 트리 에지를 포함한다고 가정하자. $cut(e, f)$ 를 계산하는 데 있어서 다음과 같은 두 가지로 경우를 나눈다.

* $e, f$ 가 *orthogonal*함: $e^{\downarrow} \cap f^{\downarrow} = \emptyset$ 이다.
* 그렇지 않음: $e^{\downarrow} \subseteq f^{\downarrow}$ 거나 그 반대이다. (*orthogonal* 하지 않으면 항상 이러함을 쉽게 알 수 있다.)

이 단락에서는 일단 $e, f$가 *orthogonal* 한 경우만 다룬다. 

$e, f$ 가 *orthogonal* 한 경우 Observation 1.4의 명제를 거의 그대로 가져올 수 있다.

**Observation 1.5**. 모든 minimum 2-respecting cut가 정확히 두 개의 트리 에지를 포함한다고 가정하자. 2-respecting min cut을 주는 두 간선 쌍이 $(e_i, e_j)$ 이고 둘이 orthogonal하다면, $\Delta(e_i^{\downarrow}) < 2 \times between(e_i^{\downarrow}, e_j^{\downarrow}), \Delta(e_j^{\downarrow}) < 2\times between(e_i^{\downarrow}, e_j^{\downarrow})$ 가 성립한다. 

**Proof of Observation 1.5.** $cut(e_i, e_j) = \Delta(e_i^{\downarrow}) + \Delta(e_j^{\downarrow}) - 2\times between(e_i^{\downarrow}, e_j^{\downarrow})$ 이다. $\Delta(e_i^{\downarrow})$ 는 $e_i$ 를 교집합으로 하는 1-respecting cut의 값이므로, $cut(e_i, e_j) < \Delta(e_i^{\downarrow})$ 가 성립한다. 같은 이유로 $cut(e_i, e_j) < \Delta(e_j^{\downarrow})$ 이다. 이에 위 등식을 대입하면 된다.

고로 자연스럽게, 모든 $e_i$ 에 대해서 이러한 조건을 만족하는 $e_j$ 를 열거하면 좋을 것이다. 다음과 같은 정의를 도입한다.

**Definition (cross-interesting).** 어떠한 $e_i$ 에 대해 $\Delta(e_i^{\downarrow}) < 2\times between(e_i^{\downarrow}, e_j^{\downarrow})$ 를 만족하며, $e_j$ 가 $e_i$ 와 orthogonal 하다면, $e_j$ 를 $e_i$ 에 대해 *cross-interesting* 하다고 정의한다.

**Lemma 1.6**. 모든 $e_i$ 에 대해서, cross-interesting한 $e_j$ 는 어떠한 정점 $v$에서 루트 방향으로 가는 경로를 이룬다.

**Proof of Lemma 1.6.** 만약 어떠한 $e_j$ 가 cross-interesting 하다면, $e_{par(j)}$ 는 $e_i$ 와 orthogonal하다는 조건 하에 여전히 cross-interesting하다. $between$ 함수의 우변의 집합이 커지고, 고로 값도 단조증가하기 때문이다. 어떠한 두 orthogonal한 $e_j, e_k$ 가 $e_i$ 에게 모두 cross-interesting하다면, $\Delta(e_i^{\downarrow}) < between(e_i^{\downarrow}, e_j^{\downarrow}) + between(e_i^{\downarrow}, e_k^{\downarrow}) = between(e_i^{\downarrow}, e_j^{\downarrow} \cup e_k^{\downarrow})$ 가 성립한다. 하지만 between 함수는 우변이 커질수록 값이 커지고, $\Delta(e_i^{\downarrow}) = between(e_i^{\downarrow}, V(T) - e_i^{\downarrow})$ 이니 모순이다.

$e_i$ 에 대해서 이러한 $e_j$ 의 경로를 찾는 것은 쉽지 않다. 일반적인 *cut-query* 환경에서는, 이러한 경로를 *cut sparsifier* 에서 찾을 수 있다. 이 글에서는 *cut sparsifier* 가 무엇인지는 설명하지 않는다 (작성자가 이해를 못 했다). Sequential algorithm에서는 cross-interesting한 간선의 집합을 random sampling을 사용해서 찾을 수 있다. 일단 이 내용은 다음 단락으로 넘어가고, cross-interesting한 $e_j$ 의 경로를 구했다고 하자.

$T$ 에 대해서 Heavy-light decomposition (HLD) 를 수행하자. HLD를 할 경우, 어떠한 노드에서 루트 방향으로 가는 경로는 최대 $O(\log n)$ 개의 경로로 분해할 수 있다. 고로, 모든 $e_i$ 에 대해서 cross-interesting한 간선은 $O(\log n)$개의 경로의 합집합으로 표현된다. 이는 문제를 다음과 같은 쿼리 문제로 변환한다:

* 쿼리 $(e_i, P)$: 간선 $e_i$, 그리고 HLD 상의 트리 경로 $P$ 가 주어질 때, $e_j \in P$ 이며, $e_i$ 와 $e_j$ 가 orthogonal한 $P$에 대해 $cut(e_i, e_j)$ 의 최솟값을 계산하라.

편의상 $P$ 는 HLD로 분해된 경로 상의 구간이 아니라 구간 **전체** 라고 간주한다. 이들은 cross-interesting하지 않으나, 더 많은 후보를 체크한다고 손해볼 것이 없다. 우리는 이러한 쿼리가 $O(n \log n)$ 개 있을 때 이를 효율적으로 처리해야 한다. 여기서, 실질적인 계산 후보를 줄이기 위해, $cut(e_i, e_j)$ 를 계산하기 위해서는 $e_j$ 가 $e_i$ 에 대해 cross-interesting할 뿐만 아니라 **그 반대도**, 즉 $e_i$가 $e_j$ 에 대해서 interesting해야 한다고 하자. 이 조건은 정확히 Observation 1.5에서 처음에 정의한 명제에 대응되고, 단지 우리가 Lemma 1.6을 소개하는 데 편의를 위해 간과했을 뿐이다.

$e_i$ 가 속하는 HLD 경로를 $Q$ 라고 하였을 때, $\{Q, P\}$ 가 같은 쿼리를 한번에 처리하자. 이 들이 "집합 기준" 으로 같은 쿼리를 한번에 처리할 것이다, 즉 $(Q, P), (P, Q)$ 모두 한번에 처리된다. 서로 다른 $(Q, P)$ 쌍은 쿼리 개수와 동일하게 $O(n \log n)$ 개 존재한다.

이제 첫 번째로, $Q$ 나 $P$ 에 속하지 않는 간선들은 전부 contract하자. 이렇게 되면 contract한 그래프는 일직선 모양이거나, $Q$ 가 $P$ 에 합쳐지는 (혹은 그 반대) 꼴의 삼거리 모형일 것이다. 삼거리 모형일 경우, 루트로 가는 방향의 간선은 또 contract해 주면 된다. *orthogonal* 할 수 없기 때문이다. 고로 contract한 그래프는 일직선이라고 가정할 수 있다. 

두 번째로, 쿼리 $(e_i, P)$ 혹은 $(e_i, Q)$ 에서 $e_i$ 의 형태로 등장하지 않은 간선 역시 전부 contract하자. 이는 위에서 언급했듯이 cross-interesting이 **양방향** 으로 적용되지 않는 쌍을 전부 무시해줘도 되기 때문에 진행할 수 있는 작업이다. 이럴 경우, 한 쿼리 묶음에서 등장한 서로 다른 $e_i$ 의 집합이 우리가 고려하게 될 간선의 집합이 된다. 이러한 $e_i$ 의 집합에 대해 1.1에서 고려한 분할 정복을 사용하면, 서로 다른 $e_i$ 의 집합을 $S$ 라고 할 때 $S \log^2 S$ 번의 $cut$ 쿼리로 문제를 해결할 수 있다. 총 고려하는 쌍의 개수가 $O(n \log n)$ 이니 이 경우를 해결하기 위해 $O(n \log^3 n)$ 번의 cut 쿼리가 필요했다. 여기서 말한 모든 contract는 명시적으로 해 줄 필요가 전혀 없고, 그냥 우리가 끊는 데 있어서 고려하지 않는다 정도로만 이해해도 된다. 

### 1.3.1 Random sampling of interesting edge set in sequential setting

먼저 첫 번째로, Lemma 1.6에 의해 cross-interesting한 간선의 집합 전체를 찾기 위해서는 집합 안의 간선들 중 가장 깊이가 깊은 간선만 알아도 충분하다는 것을 기억하자. 여기서도 $e_i$ 에 대해 그러한 가장 깊이가 깊은 간선 $e_j$ 를 찾을 것이다. 

일단 모든 간선의 가중치가 1이라고 가정하자. 이 경우 $\Delta(e_i^{\downarrow})$ 에서 **아무 간선** 을 랜덤한 확률로 샘플링하면, $\frac{1}{2}$ 의 확률로 $between(e_i^{\downarrow}, e_j^{\downarrow})$ 에 해당 간선이 속하게 된다. 충분히 높은 확률 (with high probability) 를 만족하기 위해서는 $\log n$ 개의 간선을 샘플링하면 된다. 간선의 가중치가 1이 아니지만, 각 간선을 가중치에 선형 비례하는 확률로 샘플링하면 상관이 없다. 이렇게 샘플링한 간선에서 $e_i^{\downarrow}$ 에 속하지 않는 끝점을 $v$ 라고 하면, $e_j$ 는 $v$ 와 루트를 잇는 경로 상에 존재한다. $between$ 함수는 cut-query로 계산이 가능하니, 이러한 $e_j$ 를 이분 탐색을 통해서 찾을 수 있고, 찾은 모든 $e_j$ 중 깊이가 가장 낮은 것을 반환하면 된다. 이렇게 할 경우 $O(\log^2 n)$ 개의 cut-query로 $e_j$ 를 찾을 수 잇다.

마지막으로 $\Delta(e_i^{\downarrow})$ 에서 아무 간선을 샘플링하는 것은 2차원 점을 처리하는 자료 구조를 통해서 할 수 있다. 가장 설명이 간단한 것은 Persistent segment tree이다. $T$ 에 대한 오일러 투어를 하면, 우리는 한 끝점이 $e_i^{\downarrow}$ 에 속하고 다른 한 끝점은 그렇지 않은 간선들을 원한다. $e_i^{\downarrow}$ 에 속하는 정점은 오일러 투어 상의 구간에 대응하니, *속하지 않는 정점* 들도 두 개의 오일러 투어 상 구간의 합집합에 대응된다. 고로 한 끝점은 구간 $A$, 다른 끝점은 구간 $B$ 에 있는 간선들을 가중치에 선형인 확률로 랜덤하게 고르면 된다. 전체 작업은 $O(m \log n)$ 시간 전처리 후, 각 간선당 $O(\log n)$ 시간에 할 수 있다. 

### 1.4 General tree $T$: Non-orthogonal pair of edges

Orthogonal한 경우랑 비슷하게 처리할 수 있다. Observation과 Lemma를 전부 베끼고 시작한다.

**Observation 1.7**. 모든 minimum 2-respecting cut가 정확히 두 개의 트리 에지를 포함한다고 가정하자. 2-respecting min cut을 주는 두 간선 쌍이 $(e_i, e_j)$ 이고,  $e_j^{\downarrow} \subseteq e_i^{\downarrow}$ 라면, $\Delta(e_i^{\downarrow}) < 2\times between(V(T) - e_i^{\downarrow}, e_j^{\downarrow})$, $\Delta(e_j^{\downarrow}) < 2\times between(V(T) - e_i^{\downarrow}, e_j^{\downarrow})$ 가 성립한다. 

**Proof of Observation 1.7.** $cut(e_i, e_j) = \Delta(e_i^{\downarrow}) + \Delta(e_j^{\downarrow}) - 2\times between(V(T) - e_i^{\downarrow}, e_j^{\downarrow})$ 이다. $\Delta(e_i^{\downarrow})$ 는 $e_i$ 를 교집합으로 하는 1-respecting cut의 값이므로, $cut(e_i, e_j) < \Delta(e_i^{\downarrow})$ 가 성립한다. 같은 이유로 $cut(e_i, e_j) < \Delta(e_j^{\downarrow})$ 이다. 이에 위 등식을 대입하면 된다.

**Definition (down-interesting).** 어떠한 $e_i$ 에 대해 $\Delta(e_i^{\downarrow}) < 2\times between(V(T) - e_i^{\downarrow}, e_j^{\downarrow})$ ) 를 만족한다면 $e_j$ 를 $e_i$ 에 대해 *down-interesting* 하다고 정의한다.

**Lemma 1.8**. 모든 $e_i = (par(i), i)$ 에 대해서, down-interesting한 $e_j$ 는 어떠한 정점 $v$에서 $i$ 방향으로 가는 경로를 이룬다.

**Proof of Lemma 1.8.** 만약 어떠한 $e_j$ 가 down-interesting 하다면, $e_{par(j)}$ 는 $e_i^{\downarrow}$ 안에 속한다는 조건 하에 여전히 down-interesting하다. $between$ 함수의 우변의 집합이 커지고, 고로 값도 단조증가하기 때문이다. 어떠한 두 orthogonal한 $e_j, e_k$ 가 $e_i$ 에게 모두 down-interesting하다면, $\Delta(e_i^{\downarrow}) < between(V(T) - e_i^{\downarrow}, e_j^{\downarrow}) + between(V(T) - e_i^{\downarrow}, e_k^{\downarrow}) = between(V(T) - e_i^{\downarrow}, e_j^{\downarrow} \cup e_k^{\downarrow})$ 가 성립한다. 하지만 between 함수는 우변이 커질수록 값이 커지고, $\Delta(e_i^{\downarrow}) = between(V(T) - e_i^{\downarrow}, e_i^{\downarrow})$ 이니 모순이다.

1.3.1과 완전히 동일한 샘플링 방법을 통해서, down-interesting한 경로를 찾을 수 있다. 이제 다음과 같은 쿼리 문제를 해결하자.

* 쿼리 $(e_i, P)$: 간선 $e_i$, 그리고 HLD 상의 트리 경로 $P$ 가 주어질 때, $e_j \in P$ 이며 $e_j^{\downarrow} \subseteq e_i^{\downarrow}$ 인 $j$ 들에 대해 $cut(e_i, e_j)$ 의 최솟값을 계산하라.

여기서 일반성을 잃지 않고, $e_i \notin P$ 라고 가정하자. 그 경우는, 각각의 $P$ 를 자신을 제외한 후 전부 contract시킨 뒤, 1.1의 path algorithm을 사용하면 된다.

$e_i$ 가 속하는 HLD 경로를 $Q$ 라고 하였을 때, $(Q, P)$ 가 같은 쿼리를 한번에 처리하자. 조상 관계라서 $Q$ 가 $P$ 의 아래에 있기 때문에 순서는 중요하지 않다. 서로 다른 $(Q, P)$ 쌍은 쿼리 개수와 동일하게 $O(n \log n)$ 개 존재한다. 위와 같이 양방향으로 처리할 수 없기 때문에, $Q$ 에서만 $e_i$ 로 등장하지 않은 간선을 contract시키고, $P$ 는 그대로 남긴다. 물론 나머지 간선들은 전부 contract하니 결과는 위와 비슷하게 삼거리 / 일직선 형태고 삼거리 케이스는 $Q$ 쪽 아랫 가지를 정리하면 된다.

그 후 $O((P + Q) \log^2 (P + Q))$ 시간에 분할 정복을 수행한다. $Q$ 에 대해서는 등장 횟수 합이 $O(n \log n)$ 이어서 결과를 바로 보일 수 있지만, $P$ 에 대해서는 이것이 불가능하다. 한편, $P$ 에 대응할 수 있는 $Q$ 는 $P$ 에서 루트로 올라가면서 만날 수 있는 경로 집합 뿐이고, $O(\log N)$ 개밖에 되지 않는다. 고로, $P$ 에 대해서는 따로 압축을 안 해도, 어차피 한 경로가 $O(\log N)$ 번 등장하기 때문에, path algorithm에 추가한 점은 어찌되었던 $O(n \log n)$ 개가 된다. 

### 1.5 Final algorithm

모든 조각을 조합해서 최종적인 알고리즘을 정리하자. 먼저

* 1-respecting cut을 $O(n)$ 번의 cut-query로 구한다.
* $T$ 의 HLD를 계산한 후, HLD로 분해된 경로들 각각에 대해서, path case의 알고리즘을 사용해서 2-respecting cut을 찾는다. 이 단계에서, 지우는 두 간선이 하나의 HLD 경로에 속하는 케이스가 처리된다.
* $T$ 의 간선 $e$ 에 대해서, cross-interesting path, down-interesting path 를 구한다. Lemma 1.6/1.8에 의해서, $e$ 에 대응해서 컷을 찾아야 하는 간선은 이 두 경로에만 있다.
* Cross-interesting path에서 1.3에서 소개한 환원을 통해서 path algorithm을 호출한다.
* Down-interesting path에서 1.4에서 소개한 환원을 통해서 path algorithm을 호출한다.

최종적으로, $O(n \log^3 n)$ 시간에 최소 컷을 구했다.

