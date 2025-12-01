---
layout: post
title: "Graph Measure와 Tree-decomposition (1): Tree-independence Number"
date: 2025-11-07
author: leejseo
tags: [algorithm, graph theory]
---

## 0. Preliminaries

Treewidth를 이용한 알고리즘에 대해 다루는 국문 블로그 글이 상당수 있어 많은 독자들이 익숙할 것으로 보이나, treewidth에 익숙하지 않은 독자들이 있을 수 있으니, treewidth에 대해 먼저 소개하고자 한다.

Graph 상의 다양한 NP-hard 문제들 가운데 상당수는 트리 혹은 트리와 유사한, 특수한 종류의 그래프에서 쉽게 다항시간에 해결할 수 있는 경우가 많다. 이에, 그래프가 *얼마나 tree에 가까운지*에 대한 척도를 생각해볼 수 있다.

그래프 $G$에 대해 $G$의 tree-decomposition은 다음 두 조건을 만족하는 트리 $T$와 $\beta : V(G) \to 2^{V(G)}$ 의 pair $(T, \beta)$를 말한다:

* $\bigcup_{t \in V(T)} G[\beta(t)] = G$
* $T[ \{t \in V(T) \mid v \in \beta(t) \}]$ is a tree for any $v \in V(G)$

다시 말해, 어떤 트리를 만들고, 트리의 각 노드에 $G$의 정점을 넣는데, $G$의 모든 정점이 최소 하나 이상의 노드에 포함되고, 트리 상에서 하나의 정점을 포함하는 노드들이 연결되어 있도록 하는 것을 의미한다.

이 때, $t \in V(T)$ 에 대해 $\beta(t)$ 를 $t$에 대응되는 bag 이라 부른다.

그래프의 모든 정점을 하나의 bag에 넣으면 $\lvert V(G)\rvert $ 크기의 bag을 가지는 노드 하나 짜리 tree-decomposition을 만들 수 있을 것이고, 그래프가 트리라면, 모든 bag의 크기가 1인 tree-decomposition을 만들 수 있을 것이다. 그런데, tree-decomposition을 다양한 그래프에 대해 손으로 만들어보고자 하면, 각 bag의 크기가 작은 tree-decomposition을 만드는 것이 어렵다는 것을 알 수 있다.

Tree-decomposition $(T, \beta)$의 width를

$$ \newcommand{\w}{\mathsf{width}} \w(T, \beta) := \max_t \vert \beta(t)\vert - 1 $$

로 정의하며, graph $G$의 treewidth를

$$ \newcommand{\tw}{\mathsf{tw}} \tw(G) := \min_{(T, \beta)} \w (T, \beta) $$

로 정의한다. Tree의 treewidth는 0이며, treewidth가 작다는 것은 (어떤 의미에서) tree에 가까운 graph라고 생각할 수 있다. 이와 관련한 예시나 응용은 koosaga 님 등이 작성한 다른 튜토리얼을 읽어보면 좋다.

Treewidth가 유용한 이유는, 다양한 NP-hard 문제에 대해 treewidth에 기반한 FPT algorithm 혹은 XP algorithm이 있기 때문이다. 쉽게 말하자면, 다양한 graph 상의 NP-hard 문제에 대해 treewidth가 $k$인 그래프에 대해 NP-hard 문제를 $2^k O(n^2)$ 과 같은 시간복잡도 내지는 $n^{k}$ 와 같은 시간복잡도로 해결하는 알고리즘이 존재한다는 것이다.$\newcommand{\atw}{\alpha\mathsf{-tw}}$ 

## 1. Introduction

Treewidth를 정의할 때, tree-decomposition 상에서 bag의 최대 크기를 해당 decomposition의 척도(width)로 정의하였다. 그런데, 이는 보다 더 일반화될 수 있다. 필자는 이에 대해 최근 수강중인 강의에서 다양한 흥미로운 내용을 배웠고, 이를 두 편 정도의 글에 걸쳐서 정리해보고자 한다.

이 글에서는 bag의 independence number를 척도로 사용하는, tree-independence number (혹은 $\alpha$-treewidth) 로 불리는 개념을 소개하고, 이를 기반으로 독립집합 문제를 동적계획법을 통해 해결하는 방법을 알아본다. 마지막으로, tree-independence number가 작은 tree decomposition을 계산하는 것과 관련해 알려진 결과들을 소개한다.

## 2. Tree independence number

그래프 $G$와 $G$의 tree-decomposition $(T, \beta)$를 고려해보자. 최대 독립집합 문제에 대한 tree-decomposition 기반의 dynamic programming을 생각해본다면, 각 bag 상의 정점으로 만들 수 있는 최대 독립집합의 크기가 작은 것이 좋을 것이다. 이것을 아예 graph parameter로 쓰는 것이 tree-independence number이다. Tree-decomposition의 independence number(혹은 $\alpha$-width)를

$$
\alpha(T, \beta) := \max_t \alpha(G[\beta(t)])
$$

라고 할 때, 그래프 $G$의 tree-independence number $\atw(G)$ 는 다음과 같이 정의된다:

$$
\atw(G) := \min_{(T, \beta)} \alpha(T, \beta).
$$

Tree-independence number에 대해 익숙해질 수 있는 예시 몇 가지를 살펴보자:

* $G$가 트리인 경우: $G$ 그 자체를 tree-decomposition으로 사용하면, 각 bag의 independence number가 1이라 tree-independence number는 1이 될 것이다.
* $G$가 완전그래프인 경우: $\atw(G) = 1$이 된다. (왜냐하면, 어떤 bag을 잡아도 bag이 clique을 이뤄서, bag의 independence number가 1이다.) 완전그래프 $K_n$이 $n-1$의 treewidth를 가진다는 사실에 기반해보면, treewidth가 크지만 tree-independence number가 1인 그래프가 존재한다는 사실도 알 수 있다.
* 그래프 자체가 큰 독립집합을 갖는데 tree-indpendence number가 작은 경우가 있을까?
  * $G = P_n$ (정점 $n$개인 경로)을 생각해보자. $P_n$은 트리이므로, $\atw(G) = 1$이다. 그런데, $\alpha(G) = \lceil n/2 \rceil$ 이다.
  * 조금 덜 단순한 예시로는 chordal graph가 있다. Chordal graph는 각 bag이 clique인 tree-decomposition이 존재한다. 따라서, chordal graph의 tree-independence number는 1이다.

## 3. 최대 독립집합 문제에 대한 알고리즘

그래프 $G$와 width가 작은 tree-decomposition이 주어졌을 때 이를 기반으로 동적 계획법을 활용하여 다양한 NP-hard 문제를 효율적으로 해결하는 알고리즘들과 마찬가지로, independence number가 작은 tree-decomposition 이 함께 주어졌을 때 $G$ 상의 최대 독립집합을 구하는 문제를 효율적으로 해결하는 알고리즘을 살펴보자. 이는 $\atw$ 개념을 활용하는 좋은 예시가 될 것이다.

### 3.0. Tree-decomposition 기반의 동적 계획법을 설계하는 recipe

$G$의 tree-decomposition $(T, \beta)$ 가 있다고 할 때, $T$의 임의의 정점을 루트로 잡아 루트 있는 tree-decomposition $(T, r, \beta)$ 을 고려해보자. 기본적으로 Tree-decomposition 기반의 동적 계획법 알고리즘은 여기에서 tree DP를 돌린다고 생각하면 좋다. 이에 대한 큰 그림을 제공하자면, 아래와 같다. 그러나, 구체적인 예시 없이 보면 이해하기 어렵다고 생각되기에, 3.1절 이후를 전부 살펴본 후 다시 보는게 이해에 도움이 될 것이다.

어떤 노드 $t$가 부모 $d$와 자식 $c_1, c_2, \cdots, c_k$ 를 가진다고 하자. $t$를 루트로 하는 subtree의 bag 들에 의한 $G$의 induced subgraph $G[\beta(T_t)]$를 생각해보자. 우리는 $G[\beta(T_{c_i})]$ 들에 대해 어떤 정보들을 가지고 있을 것이고, 이를 기반으로 $G[\beta(T_t)]$에 대한 어떠한 정보를 계산하는 방식으로 DP를 돌린다. 그리고 이는 당연히 $G[\beta(T_d)]$의 DP 값을 계산하는 데에 활용될 수 있어야 할 것이다.

* 여기에서, $T$의 어느 subtree $T'$ 에 대해 $\beta(T') := \bigcup_{t \in V(T')} \beta(t)$ 로 정의된다. 

이것을 하기 위해, 대개는 subtree와 subtree root의 bag의 '좋은' subset간 pair를 만들어서, 해당 pair에 대한 문제를 해결한다. 이와 관련해서 상태전이가 일어나는 과정은, 보통 다음과 같이 요약된다.

1. $(G[\beta(T_{c_i})], \beta(c_i) \cap \beta(t))$ 에 대해 DP 값을 구해둔다.
2. 1의 각 DP 값들에 $\beta(t)$를 *잘 합쳐서* $(G[\beta(T_{c_i}) \cup \beta(t)], \beta(t))$ 에 대한 DP 값으로 확장해둔다.
3. 2에서 구한 DP값을 살펴보면, $i \neq j$ 에 대해 $V(G[\beta(T_{c_i}) \cup \beta(t)]) \cap V(G[\beta(T_{c_j}) \cup \beta(t)]) = \beta(t)$ 임을 알 수 있다. (왜 이런지 확인해보는 것은 tree-decomposition의 정의를 응용하며 익숙해질 수 있는 좋은 연습문제이다.) 이를 기반으로 2의 값들을 합쳐나가면서 $(G[\beta(T_t)], \beta(t))$에 대한 DP 값을 구해둔다.
4. 마지막으로, 3에서 구한 것을 루트 방향으로 확장해서 $(G[\beta(T_t)], \beta(t) \cap \beta(d))$ (단, $t = r$ 이면 $\beta(d) = \emptyset$ 으로 둔다.)에 대한 결과로 확장한다.

### 3.1. 동적계획법 테이블 정의

그래프 $G$와 정점 집합의 부분집합 $X \subseteq V(G)$, 그리고 $Y \subseteq X$에 대해 동적계획법 테이블 $D((G, X), Y)$를 $I \cap X = Y$ 인 $G$ 상의 독립집합 $I$의 최대 크기로 정의하자. 이를 어떤 $((G, X), Y)$에 대해 계산할지는 3.0을 참고해보면 좋다.

### 3.2. 상태전이 (1): tree-decomposition 상에서 '부모노드와 합쳐진' 자식 노드들 끼리 합치기

이 절에서 다루는 내용은 3.0의 3번에서 활용된다.

두 그래프 $G_1, G_2$가

1. $V(G_1) \cap V(G_2) = X$, $G_1[X] = G_2[X]$
2. $\alpha(G_1[x]), \alpha(G_2[X]) \le k$

를 만족한다고 하자. 이 때, $(G_1, X)$와 $(G_2, X)$ 로 부터 $(G_1 \cup G_2, X)$를 채워보자.

일단, 2번 가정에 의해 $(G_1 \cup G_2, X)$에 대해 테이블을 채워야 하는 $Y$의 개수는 $\vert X \vert^k \le \vert V(G)\vert ^k$  이하이다.

이  $Y$ 에 대해서는 $D((G_1 \cup G_2, X), Y) := D((G_1, X), Y) + D((G_2, X), Y) - \vert Y\vert$ 의 transition을 주면 된다. 왜냐하면,

* $I$가 $X \cap Y = I$ 인 independent set of $G_1 \cup G_2$ 라면,  $I_1 := I \cap V(G_1)$ 과  $I_2 := I \cap V(G_2)$ 는 $G_1, G_2$ 각각에서 independent 하다. 그리고, $I_1 \cap I_2 = Y$ 이다. 따라서, $\vert I\vert \le \vert I_1 \vert + \vert I_2 \vert - \vert I_1 \cap I_2\vert  \le D((G_1, X), Y) + D((G_2, X), Y) - \vert Y\vert$ 가 된다.  고로, $D((G_1 \cup G_2, X), Y) \le D((G_1, X), Y) + D((G_2, X), Y) - \vert Y\vert$.
* $G_1, G_2$에 대한 가정에 의해 $V(G_1) - X$ 와 $V(G_2) - X$ 사이에는 간선이 없다. 따라서, DP 값을 $D((G_1, X), Y), D((G_2, X), Y)$로 실제로 주는 두 독립집합 $I_1, I_2$  에 대해 $I_1 \cup I_2$ 를 고려해보면, 이 역시도 $G_1 \cup G_2$ 에서 독립집합이다. 고로, $D((G_1 \cup G_2, X), Y) \ge D((G_1, X), Y) + D((G_2, X), Y) - \vert Y\vert$

따라서, $X$를 공유하는 두 그래프에 대한 DP 테이블을 $\vert V(G)\vert^k$ 시간에 합쳐줄 수 있다.

### 3.3. 상태전이 (2): tree-decomposition 상에서 부모노드와 자식노드를 합치기

이 절에서 다루는 내용은 3.0의 2에서 활용된다.

두 그래프 $H, G$와 $Y \subseteq V(H)$, $X \subseteq V(G)$에 대해 $(H, Y)$와 $(G, X)$를 고려할 것이다. 이 때, $H = G - (X - Y)$, $Y \subseteq X$, $\alpha(G[X]) \le k$이며, $G-X$의 정점과 $X - Y$의 정점 사이를 잇는 간선은 없다고 가정한다. $D((H, Y), *)$에서 $D((G, X), *)$을 구하는 상태전이를 만들어보자.

* 다소 작위적이어 보일 수 있으나, 문제 입력의 그래프를 $G_0$라 할 때, $G := G_0[\beta(T_c) \cup \beta(t)]$, $Y := \beta(c)$, $X := \beta(t)$로 두면, $H = G_0[\beta(T_c) \cup \beta(t)] - (\beta(t) - \beta(c)) = G_0 [\beta(T_c)]$가 된다. 즉, 3.0의 2에서 하고자 하는 자식 서브트리에 부모 노드를 합치는 작업과 정확하게 일치한다.

이를 위해, ${\vert X \vert \choose k} \le \vert V(G) \vert ^k$ 개의 크기가 $k$ 이내인 $Z \subseteq X$를 고려하자. $\alpha(G[X]) \le k$ 이므로, 이로서 모든 independent 한 $Z \subseteq X$를 고려하기에 충분하다. 각 independent한 $Z$에 대해, $D((G, X), Z) := D((H, Y), (Z \cap Y)) + \vert Z - Y \vert$ 로 테이블을 채워주자. (등호가 성립하는 이유도 앞에서 처럼 간단한 논증을 통해 확인할 수 있다.)

이 역시도 총 $\vert V(G) \vert^k$ 시간에 가능하다.

### 3.4. 상태전이 (3): 노드의 DP값을 부모노드 방향으로 확장하기

$\alpha(G[X]) \le k$ 인 $(G, X)$에 대한 DP값으로 부터 $X' \subseteq X$인 $X'$에 대해 $(G, X')$의 DP 값을 구해보자.

독립인 $X'$의 subset은 $X$의 subset이기도 하다. 따라서, 앞에서와 같이 $\vert V(G) \vert^k$ 개이내의 독립인 $Y \subseteq X'$을 살펴보자.

$I \cap X' = Y$ 인 독립 집합 $I$를 생각해보면, $Y \subseteq I \cap X \subseteq X$ 이며, $I \cap X$ 역시도 독립집합이다. 따라서, $I' := I \cap X$을 guess 해보면서,

$$
D((G, X'),Y) := \max_{I' \subseteq X, I' \text{은 독립집합}, I' \cap X' = Y} D((G, X), I')
$$

으로 갱신해줄 수 있다. 이 과정에서 하나의 $Y$에 대해 최대 $\vert V(G) \vert ^k$ 개의 후보를 살펴봐야 한다.

위와 같이 하면 $(G, X')$을 $\vert V(G) \vert^{2k}$ 시간에 구할 수 있다.

### 3.5. 리프 노드에 대한 DP값 계산

리프 노드 $l$ 의 경우, $G[\beta(T_l)] = G[\beta(l)]$ 이므로, $\alpha(G[\beta(T_l)]) \le k$ 이다. 따라서, 리프 노드 $l$과 $X \subseteq \beta(T_l)$ 에 대해 $(G[\beta(T_l)], X)$ 에 대한 DP값은 brute-force로 $\vert V(G) \vert ^k$ 시간에 계산할 수 있다.

### 3.6. 동적 계획법 알고리즘

이제, 다시 3.0으로 돌아가서, $G$와 $\alpha(T, \beta) \le k$ 인 rooted tree-decomposition $(T, r, \beta)$에 대해 다음 recipe를 살펴보자:

1. $(G[\beta(T_{c_i})], \beta(c_i) \cap \beta(t))$ 에 대해 DP 값을 구해둔다.
2. 1의 각 DP 값들에 $\beta(t)$를 *잘 합쳐서* $(G[\beta(T_{c_i}) \cup \beta(t)], \beta(t))$ 에 대한 DP 값으로 확장해둔다. (3.3)
3. 2에서 구한 DP값을 살펴보면, $i \neq j$ 에 대해 $V(G[\beta(T_{c_i}) \cup \beta(t)]) \cap V(G[\beta(T_{c_j}) \cup \beta(t)]) = \beta(t)$ 임을 알 수 있다. (왜 이런지 확인해보는 것은 tree-decomposition의 정의를 응용하며 익숙해질 수 있는 좋은 연습문제이다.) 이를 기반으로 2의 값들을 합쳐나가면서 $(G[\beta(T_t)], \beta(t))$에 대한 DP 값을 구해둔다. (3.2)
4. 마지막으로, 3에서 구한 것을 루트 방향으로 확장해서 $(G[\beta(T_t)], \beta(t) \cap \beta(d))$ (단, $t = r$ 이면 $\beta(d) = \emptyset$ 으로 둔다.)에 대한 결과로 확장한다. (3.4)

이를 기반으로, DFS를 하며 tree DP를 수행하게 되면, 자식 노드의 값들로 부터 한 노드의 DP값을 계산하는 데에 $\vert V(G) \vert ^ {O(k)} \cdot (  \text{자식의 수})$ 시간이 걸리므로, 총 $\vert V(G) \vert^{O(k)}$ 시간에 $(G[\beta(T_r)], \emptyset) = (G, \emptyset)$ 에 대한 DP값을 구할 수 있다. 그리고 이것이 문제에 대한 답이 된다.

## 4. $\alpha(T, \beta)$ 가 작은 Tree-decomposition에 대해

우리는 3절에서 $\alpha(T, \beta) \le k$인 tree-decomposition이 주어졌을 때 $\vert V(G) \vert ^{O(k)}$ 시간에 최대 독립집합을 구하는 알고리즘을 살펴보았다. 그래프 $G$에 대해 $\alpha(T, \beta)$ 가 작은 tree-decomposition을 trivial하게 구성할 수 있는 graph class에 적용하기에는 충분한 알고리즘이나, (예를 들어 chordal graph에 쉽게 적용할 수 있을 것이다.) 일반적인 그래프에 대해 $k$가 작은 tree-decomposition을 구하는 것은 자명하지 않은 일이라는 한계가 있다.

실제로, $k$ 이하 크기의 $\alpha(T, \beta)$를 가지는 tree-decomposition이 존재하는지 판별하고, 존재한다면 구하는 parametrized problem은 W[1]-hard 계산 복잡도 class에 속하는 문제이며, 이는 비자명함을 의미한다고 할 수 있다.

다만, [1]의 알고리즘을 활용하면, $G$와 정수 $k$를 입력으로 받아 $\atw (G) > k$ 임을 판별하거나, $\alpha(T, \beta) \le 8k$ 인 tree-decomposition을 구하는 것이 $2^{O(k^2)} \vert V(G) \vert^{O(k)}$ 시간에 가능하다. 이 결과는, 우리가 3절에서 함께 살펴본 알고리즘이 $2^{O(k^2)} \vert V(G) \vert ^{O(k)}$ 시간에 동작함 또한 알려준다.

## 여담

다음 글에서는 $\alpha$-treewidth를 더 일반화 하여 임의의 graph의 local한 property를 보존하는 graph measure에 대해 확장한 버전의 $\mu$-treewidth를 소개해볼 것이다.

## Reference

이 자료는 KAIST의 전산학 특강(알고리즘 그래프 구조 이론) 수업에서 배운 내용을 상당부분 참고하였다.

[1] Clément Dallard, Fedor V. Fomin, Petr A. Golovach, Tuukka Korhonen, Martin Milanič. Computing Tree Decompositions with Small Independence Number https://arxiv.org/abs/2207.09993