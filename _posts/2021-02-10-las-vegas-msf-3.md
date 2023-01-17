---
layout: post
title: "Dynamic MSF with Subpolynomial Worst-case Update Time (Part 3)"
author: koosaga
date: 2021-02-10
tags: [graph-theory, theoretical-computer-science, data-structure]
---

# Dynamic MSF with Subpolynomial Worst-case Update Time (Part 3)

## Chapter 4: Continued

### 4.2. Contraction

#### 4.2.1 Properties of Contraction

Lemma 4.2를 사용하여 우리는 Non-tree edge의 개수가 작은 Decremental MSF 알고리즘으로 문제를 환원하였다. 이제 이를 Edge의 개수가 작은 Decremental MSF 알고리즘으로 다시 환원한다. 다행이도, 이 부분은 Competitive Programming에서 익숙한 내용이라서 배경 지식이 있다면 빠르게 이해할 수 있다.

**Definition 4.11.** (Connecting Paths). 트리 $T = (V, E)$ 와 터미널의 집합 $S \subseteq V$ 가 있을 때, $P_{S}(T)$ 는 다음과 같이 정의된다.

* $P_{S}(T)$ 는 edge-disjoint paths의 집합이다.
* 그래프 $\cup_{P \in P_S(T)}P $ 는 $T$ 의 연결된 서브트리를 이룬다.
* 모든 터미널 $u \in S$ 에 대해서, $u$ 는 어떠한 경로 $P \in P_S(T)$ 의 끝점이다.
* $P_S(T)$ 의 어떠한 경로의 끝점 $u$ 는, $u \in S$ 를 만족하거나, 다른 두 경로 $P^\prime, P^{\prime\prime}$ 의 끝점이다.

**Remark.** $P_S(T)$ 의 개념은 흔히 *트리 압축* 이라고 부른다. [JOI 2014 Factories](https://www.acmicpc.net/problem/11933) 의 풀이로 이미 Competitive Programming에 잘 알려져 있다. 고로 이 글에서 자세하게 설명하지 않는다.

**Lemma 4.12**. 임의의 트리 $T = (V, E)$ 와 터미널의 집합 $S \subseteq V$ 에 대해 $P_S(T)$ 는 유일하게 정의된다. (증명은 생략한다.)

**Proposition 4.13**. 간선 집합 $E$ 에 대해 $end(E)$ 를 $E$ 에 속하는 간선의 양 끝점의 합집합이라고 하자. 임의의 그래프 $G = (V, E)$ , 포레스트 $F \subseteq E$, 어떠한 집합 $end(E - F) \subseteq S$ 에 대해서, $P_S(F)$ 를 이루는 어떠한 경로는 $G$ 에서 induced path를 이룬다.

**Proof.** $p \in P_S(F)$ 가 induced path가 아니라고 하자. 이 경우 $p$ 의 끝점을 제외한 위치에 $deg_G(v) \geq 3$ 인 정점이 존재하며, path 상의 다른 정점 $w \in p$ 를 잇는 간선 $(v, w)$이 존재한다. 만약 $v \in S$ 이면 바로 모순이다. 그렇지 않다면 이 정점에 인접한 모든 간선은 $F$ 에 속한다. 그 중 끝점이 $w$ 인 간선을 찾게 된다면 $F$ 에 사이클이 존재한다. 모순이다.

**Definition 4.14.** (Contracted Graphs/Forests and Super Edges) 어떠한 가중치 있는 그래프 $G = (V, E, w)$, 포레스트 $F \subseteq E$, 집합 $end(E - F) \subseteq S$ 에 대해서, contracted (multi)-graph $G^\prime$ 과 contracted forest $F^\prime$ with respect to $S$ 는 $G, F$ 로부터 다음과 같이 만들 수 있다.

* $F \setminus \cup_{P \in P_S(F)} P$ 에 속하는 모든 간선을 제거 (즉, 차수 1 간선을 포레스트에서 반복적으로 제거)
* 모든 압축된 경로 $P = (u, \ldots, v)$ 를 하나의 super edge로 대체, 이 때 간선의 가중치는 경로 최댓값.

($F^\prime$ 는 $F$ 에서 $S$를 일반적인 형태로 트리 압축하여 만들고, 여기에 non-tree edge들을 추가하여 $G^\prime$ 를 만든다고 보면 된다.)

이제 $(G^\prime, F^\prime) = Contract_S(G, F)$ 로 둔다. 임의의 $e \in P_{uv} \subseteq F$ 에 대해서 $(u, v) \in F^\prime$ 을 $e$를 *커버* 하는 super edge라고 한다.

**Proposition 4.15.** $E(G^\prime) = N \cup E(F^\prime)$, $E(G) \le N + 2S$ (증명 생략).

**Proposition 4.16.** $F = MSF(G)$ 일 때 $F^\prime = MSF(G^\prime)$ 이다 (증명 생략).

#### 4.2.2 Main Reduction

이제 아래 Proposition을 통해서, 우리는 Non-tree edge가 작은 Decremental MSF 알고리즘의 크기를 어떻게 줄이는지 알 수 있다.

**Proposition 4.17.** $F = MSF(G)$ 라고 하자. $G_1 = G - e, F_1 = F - e$ 라고 하고 ($e \notin F$ 일 수 있음에 유의), $(G_1^\prime, F_1^\prime) = Contract_{end(E - F)}(G_1, F_1)$ 이라고 하자. 다음이 성립한다.

* $e \in N$ 일 경우 $(G_1^\prime, F_1^\prime) = (G^\prime - e, F)$ 이다. $MSF(G_1) = F_1$ 이며, 고로 $MSF(G_1^\prime) = F_1^\prime$ 이다. (단순히 $N$ 에서 $e$ 가 빠진다.)
* 그렇지 않고, $e \in F \setminus \cup_{P \in P_S(F)} P$ 라고 하면, $(G_1^\prime, F_1^\prime) = (G^\prime, F^\prime)$ 이다. $MSF(G_1) = F_1$ 이며, 고로 $MSF(G_1^\prime) = F_1^\prime$ 이다. ($MSF(G_1) = F_1$ 이 성립하는 이유는 $e$ 가 절선이기 때문이다.)
* 그렇지 않으면, 즉 $e \in \cup_{P \in P_S(F)} P$ 이며 $e^\prime = (u^\prime, v^\prime) \in F$ 가 $e$ 를 커버하면, $(G_1^\prime, F_1^\prime) = (G^\prime - e, F^\prime - e)$ 이며 두 가지 경우 중 하나가 성립한다.
  * $MSF(G_1) = F - e = F_1$ 이며 $MSF(G^\prime_1) = F_1^\prime$ 이다. ($e$ 가 절선인 경우)
  * $f \in N$ 이 존재하여, $MSF(G_1) = F \cup f - e = F \cup f$ 이고, $MSF(G_1^\prime) = F^\prime \cup f - e^\prime = F_1^\prime \cup f$ 이다. ($e$ 가 절선이 아니고 대체 간선이 있는 경우)

아래 Lemma는 Proposition 4.17을 구현하는 기술적인 바탕을 제시한다.

**Lemma 4.18** (Lemma 15 of **[Holm et. al 2001](https://dl.acm.org/doi/10.1145/502090.502095)**). 다음 두 단계로 작동하는 알고리즘 $C$ 가 존재한다.

* 첫 단계에서, $C$ 는 간선이 최대 $m$ 개인 포레스트를 관리할 수 있다. 간선 삽입 / 삭제 쿼리를 $O(\log m)$ 에 지원한다.
* 다음 단계에서, 임의의 간선 집합 $N$ 이 주어질 때, $C$ 는 $(G^\prime, F^\prime) = Contract_{end(E - F)}(G, F)$ 를 $O(N \log m)$ 에 반환할 수 있다. 이 때 $G = (V, F \cup N)$ 이다. 또한 임의의 간선 $e \in F$ 에 대해서, $C$ 는 $e$ 를 커버하는 슈퍼 간선 $e^\prime = (u^\prime, v^\prime) \in F^\prime$  을 반환할 수 있다. (존재하지 않을 경우 반환하지 않는다.) 이 시간 복잡도는 $O(\log m)$ 이다.

**Proof.** HDLT는 Top tree를 사용하여 증명하지만, Euler tour tree를 사용하여도 될 것으로 보인다.

이제 이 단락의 Main Reduction을 보인다.

**Lemma 4.19.** 다음과 같은 성질을 가진 Decremental MSF 알고리즘 $A^\prime$이 있다고 하자:

* 그래프에는 $m^\prime$ 개의 간선이 있다.
* 전처리 시간이 $t_{pre}(m^\prime, p)$ 이며 업데이트 시간이 $t_u(m^\prime, p)$ 이다.

그렇다면, 다음과 같은 성질을 가진 Decremental MSF 알고리즘 $B^\prime$ 이 존재한다.

* $G = (V, E)$ 를 $m$ 개의 간선과 $k$ 개의 non-tree edge를 가지는 그래프라고 하자. 이 때 $5k \le m^\prime$ 을 만족한다.
* 알고리즘은 입력으로 $F = MSF(G), N = E - F$를 받는다. 또한, 알고리즘 $C$ 가 $F$에 대한 첫 단계를 이미 실행하고 있는 상태로 주어진다. (초기화 시간을 논외로 둘 수 있다.)
* $B^\prime$ 은 $G$ 에 대한 Decremental MSF 알고리즘이다.
* 전처리 시간은 $t_{pre}^\prime(m, k, p) = t_{pre}(5k, p) + O(k \log m)$ 이다.
* 업데이트 시간은 $t_u(m, k, p) = t_u(5k, p) + O(\log m)$ 이다. 
* $1 - p$ 의 확률로 위 조건을 모두 만족한다.

**Proof sketch.** 알고리즘의 전처리 과정은 다음과 같다. $S = end(E - F)$ 라고 두자. 

* $C$ 를 두 번째 단계로 전환하고,  $(G^\prime, F^\prime) = Contract_S(G, F)$ 를 $O(N \log m)$ 에 계산한다. 이 때 $E(G^\prime) \le N + 2S \le 5k$ 이다.
* $A^\prime$ 을 $G^\prime$ 으로 초기화한다. 이 때 시간은 $t_{pre}(E(G^\prime), p) = t_{pre}(5k, p)$ 이다. 고로 전체 시간은 $t_{pre}(5k, p) + O(k \log m)$ 이다.

쿼리를 처리하는 것은, $F = MSF(G), (G^\prime, F^\prime) = Contract_{S}(G, F)$ 라는 불변량을 유지시키면 가능하다. 위에서 언급한 케이스들 중 우리가 속하는 케이스가 무엇인지를 $C$ 와 주어진 불변량으로 파악할 수 있으며, 파악이 된 경우 $A^\prime$ 에 $O(1)$ 개의 연산을 하거나 $F$ 에 간선이 들어가는지의 마킹을 $O(1)$ 개 바꿔주면 불변량을 보존할 수 있다. $\blacksquare$

### 4.3 Combining two reduction

이제 Lemma 4.2와 Lemma 4.19를 결합한다.

**Lemma 4.20.** 다음과 같은 성질을 가진, Las-Vegas Decremental MSF Algorithm $A^\prime$ 이 있다고 가정하자.

* 그래프에는 $m^\prime$ 개의 간선이 있다.
* 전처리 시간이 $t_{pre}(m^\prime, p)$ 이며 업데이트 시간이 $t_u(m^\prime, p)$ 이다.

이 때 $5k \le m^\prime$ 을 만족하는 임의의 정수 $B$ 에 대해, 다음과 같은 성질을 가진, Las-Vegas Fully dynamic MSF algorithm $Ffn$ 이 존재한다.

* 전처리 시간이 $t^\prime_{pre}(m, k, B, p) = T_{pre}(5k, p^\prime) + O(m \log^2 m)$ 이다.
* $B$ 개의 간선 추가, 혹은 1개의 간선 제거를 하는 데 드는 시간이 $t_u^\prime(m, k, B, p) = O(\frac{B \log k}{k} \times t_{pre}(5k, p^\prime) + B\log^2 m + \log k \times t_u(5k, p^\prime))$ 이다.

이 때 $p^\prime = O(p / \log k)$ 이며, 각  시간 복잡도는 $1 - p$ 확률로 만족된다.

**Proof sketch.** 두 Lemma를 결합하는 점에서 자명하지 않은 점은 $C$ 를 초기화하는 것이다. 각 레벨마다 6개의 $C_i$ 자료구조를 둘 것이다. 자료구조는 각각 $6 \times 2^i$ 시간의 사이클을 돈다. 하나의 $D_i^\prime$ 이 레벨 $i$ 에서 초기화 후 "살아 있는" 시간은 $5 \times 2^i $ 이다. 고로 이 시간 동안은 $D_i^\prime$ 에서 필요한 자료구조로써의 역할을 하면 된다. 이후 $2^i$ 시간 동안 $F$ 에 들어온 업데이트의 수는 최대 $O(B 2^i)$ 이다. 이를 6배의 속도로 돌리면서 반영해 주면 $O(B \log m)$ 의 시간이 사용된다. 모든 레벨에 대해 이것이 일어나니 $O(B \log^2 m)$ 의 추가 시간을 사용하게 된다. 이제 이 점을 반영하여 식 전개를 하면 원하는 결과를 얻을 수 있다. $\blacksquare$

이제 Main Theorem으로 가기 직전까지 왔다. 한 가지만 관찰하자.

**Lemma 4.21.** Theorem 4.1의 알고리즘 $D$ 가 존재한다고 가정한다면, 다음과 같은 성질을 가진, Las-Vegas Decremental MSF Algorithm $D^\prime$ 이 존재한다.

* 그래프에는 $m^\prime$ 개의 간선이 있다.
* $T(m^\prime)$ 길이의 간선 삭제를 수행한다.
* 전처리 시간이 $t_{pre}(m^\prime, p) + O(m^\prime)$ 이며 Worst-case 업데이트 시간이 $1- p$ 확률로 $O(m^\prime / T(m^\prime)) + 3t_u(3m^\prime, p)$ 이다.

**Proof.** 각 정점을 degree 크기 만큼의 이진 트리로 변환하면 된다. 간선의 가중치는 $\infty$ 로 두자. $\blacksquare$

**Proof sketch of the Theorem 4.1.** 

* Section 4.1에 나온 Decremental Reduction with Few Non-tree Edges을 적용하고
* Section 4.2에 나온 Compression으로 Few Edges를 가진 Decremental Problem으로 변환하고
* Lemma 4.21을 통하여 최대 차수가 3인 Decremental Problem으로 변환하면

된다. 자명하지 않은 건 확률 분석과 수식 도출인데, 알아서 하도록 하자. $\blacksquare$

## Chapter 5. MSF Decomposition

아래 Hierarchial Decomposition을 정의한다. Centroid Decomposition에 대해서 알고 있는 독자는 이를 상상하면서 읽으면 이해에 도움이 될 것이다. (물론 둘은 많이 다르다. 그냥 그래프를 Hierarchial decompose한다는게 무슨 뜻인지만 상상하자.)

**Definition 5.1.** (Hierarchcial Decomposition). 임의의 그래프 $G = (V, E)$ 에 대해서, $G$의 hierarchial dcomposition $H$ 는 루트 있는 트리이다. $H$ 의 각 노드 $C$ 는 *클러스터* 라고 불리는 $G$ 의 서브그래프를 이룬다. $H$ 가 만족해야 하는 조건은 다음과 같은 두 가지이다:

* $H$ 의 루트 클러스터는 $G$ 전체여야 한다.
* 임의의 리프가 아닌 클러스터 $C \in H$ 에 대해서 $C_1, C_2, \ldots$ 를 $C$ 의 자식이라고 하자. $C_i$ 는 $C$ 의 정점의 분할(partition) 이다.

Hierarchial Decomposition의 각 노드의 *레벨* 을 정의할 수 있다. 루트 클러스터의 레벨은 1이며, 다른 클러스터는 부모 클러스터의 레벨에 1을 더한 레벨을 가진다. 레벨의 최댓값을 트리의 *깊이* 라고 한다. 

$E^C = E(C) - \cup_i E(C_i)$ 라고 하자. $e \in E^C$ 를 $C$-own edge라고 부르고, $e \in \cup_i E(C_i)$ 를 $C$-child edge라고 부른다.

여담으로, 각 클러스터는 연결될 필요도 없고, induced subgraph일 필요도 없다. $u, v$ 가 공통으로 속하는 클러스터 $C$ 에 대해서, $(u, v) \notin E(C)$ 일 수 있다.

이제 다음과 같은 Main Theorem을 소개한다.

**Theorem 5.2.** $MSFdecomp$ 라는 랜덤 알고리즘은 다음을 입력으로 받는다:

* 연결 그래프 $G = (V, E, w)$. 이 때 $V = n, E= m, w : E \rightarrow \{1, \ldots, m\}$ 이며 각 정점의 최대 차수는 3이다.
* 실패 확률 파라미터 $p \in (0, 1]$, 전도율 파라미터 $\alpha \in [0, 1]$. 파라미터 $d, s_{low}, s_{high}$. 이 때 $d \geq 3, s_{high} \geq s_{low}$

그리고 $\gamma = n^{O(\sqrt{ \log \log n / \log n})}$ 일 때 시간 $\tilde{O}(nd\gamma \log \frac{1}{p})$ 에 알고리즘은 

* 그래프 $G^\prime = (V, E, w^\prime)$. 이 때 $w^\prime : E \rightarrow \mathbb{R}$ .
* $G^\prime$ 의 Hierarchial decomposition $H$

를 반환하며, 이들은 다음과 같은 성질을 가진다.

1. (*가중치 단조증가*) 모든 $e \in E$ 에 대해 $w^\prime(e) \geq w(e)$
2. (*가중치 유사성*) $\{e \in E  w(e) \neq w^\prime(e)\} \le \alpha d \gamma n$ 
3. (*귀납적 MSF 구성*) 임의의 클러스터 $C \in H$ 와 임의의 간선 집합 $D$ 에 대해, $MSF(C - D) = \cup_{C^\prime : \text{child of } C} MSF(C^\prime - D) \cup (MSF(C - D) \cap (E^C - D))$
4. (*균형 잡힌 트리*) $H$ 의 깊이는 $d$ 이하이다.
5. (*균일한 리프 1*) $E(C) \le s_{high} \iff (C$ 가 리프 클러스터) 
6. (*균일한 리프 2*) 각 리프 클러스터는 최소 $s_{low}/3$ 개의 정점을 포함한다.
7. (*균일한 레벨*) 레벨 $i$ 에서 $\cup_{C : \text{non-leaf, level-}i} E^C \le n/(d-2) + \alpha \gamma n$ 이다.
8. (*높은 전도율 보장*) $1 - p$ 의 확률로 모든 루트가 아닌 클러스터 $C \in H$ 의 전도율은 $\phi(C) = \Omega(\alpha / s_{low})$ 이다. 이 값을 *conductance guarantee* 라고 한다.

이 Decomposition에서 가장 중요한 것은 3번째 항목이다. $D$ 라는 간선 집합을 제거한 이후에도 어떠한 노드의 MSF를 귀납적으로 생성할 수 있다. 각 컴포넌트에 대해서 $MSF(C) \cap E^C$ 만 효율적으로 관리해 주면 이를 전역적으로 합쳐줌으로써 전체 그래프의 MSF를 관리할 수 있다. 

마음 아프지만, 우리는 이 글에서 이 Theorem을 증명하지 않는다. 이를 사실로 간주하고, 다음 Chapter로 넘어가자.

## Chapter 6. Dynamic MSF Algorithm

드디어 우리는 전체 논문의 Main Theorem에 대해 논의한다.

**Theorem 6.1.** $n$ 개의 노드와 $m$ 개의 간선을 가진 그래프에서, 전처리 $O(m^{1 + O(\sqrt{\log \log m / \log m})} \log \frac{1}{p}) = O(m^{1 + o(1)} \log \frac{1}{p})$ 시간과 worst-case 업데이트 $O(n^{O(\log \log \log n / \log \log n)} \log \frac{1}{p}) = O(n^{o(1)} \log \frac{1}{p})$ 시간을 $1 - p$ 의 확률로 보장하는 fully dynamic MSF 알고리즘이 존재한다.

Chapter 4의 Theorem 4.1을 사용하면, 다음 Lemma를 증명함으로 충분함을 보일 수 있다.

**Lemma 6.2.** $n$ 개의 노드와 $m$ 개의 간선을 가진 최대 차수 3 이하의 그래프와, 길이 $T = \Omega(n^{1 - O(\log \log \log n / \log \log n)})$ 의 간선 삭제 수열이 주어진다고 하자. 전처리 $O(m^{1 + O(\sqrt{\log \log n / \log n})} \log \frac{1}{p}) $ 시간과 worst-case 업데이트 $O(n^{O(\log \log \log n / \log \log n)} \log \frac{1}{p})$ 시간을 $1 - p$ 의 확률로 보장하는 decremental MSF 알고리즘 $A$가 존재한다.

알고리즘의 High-level idea는, $MSF(G)$ 를 관리하기 위해서 *sketch graph* $H$ 를 관리하는 것이다. 매 순간 $MSF(G) = MSF(H)$ 를 만족하며, $H$ 에는 non-tree edge의 개수가 많지 않다. 이제 $H$ 를 non-tree edges가 작을 때 작동하는 또 다른 MSF 알고리즘으로 관리해 주면 된다. 

마음 아프지만, 우리는 이 Theorem 역시 엄밀히 증명하지 않는다. 단지 증명에 사용된 최종 알고리즘만 서술한다.

### 6.1 Preprocess

알고리즘 $A$ 는 재귀적이다. 다른 말로, 우리의 증명은 간선 개수 $m$ 에 대해서 수학적 귀납법을 사용한다. $m - 1$ 개 이하의 간선이 주어질 때 Lemma 6.2가 참이라고 가정하고, 이 알고리즘의 인스턴스를 $A(G, p)$ 라고 부르자. ($E(G) < m$). 

$G$ 가 연결되어 있다고 가정하자 (아닐 경우 각 컴포넌트에 대해서 따로 해결한다). 고로 $n = \Theta(m)$ 이며 $T(m) = o(m)$ 개의 삭제만 처리할 것이니 이는 항상 성립한다.

$\gamma = n^{O(\sqrt{ \log \log n / \log n})}$ 이라고 두고 Theorem 5.2의 $MSFdecomp$ 알고리즘을 실행한다. 이 때 $\alpha = 1/\gamma^3, d = \gamma, s_{low} = \gamma, s_{high} = n/\gamma$ 이다. 우리는 이를 통해서 conductance guarantee가 $O(1/\gamma^4)$ 인 hierarchial decomposition $H$ 를 얻을 수 있다. 이제 필요한 것들을 정의하자. 

* $E^{\neq} = \{e \in E  w(e) \neq w^\prime(e)\}$ 라고 정의하고 $E^{\neq}(w), E^{\neq}(w^\prime)$ 을 $E^\neq$ 에 속하는 에지들을 각 weight 함수에 따라 가중치 매긴 집합이라고 하자.
* 리프 클러스터를 *작은 클러스터*, 리프 아닌 클러스터를 *큰 클러스터* 라고 부르자. 
  * **Proposition 6.3.** 임의의 클러스터 $C \in H$ 의 큰 자식 클러스터의 개수는 $O(n / s_{high})$ 이다.
  * Theorem 5.2의 5번 성질에 의하여 $C \in H$ 가 리프 클러스터라는 것은 $E(C) \le s_{high}$ 임과 필요충분이다 (간선 삭제 전 기준). 
* $G_{small} = \cup_{C:\text{small}} C$ 라고 하자. 즉 작은 클러스터의 합집합이다. 
* $M_{small} = MSF(G_{small})$ 이다. 
  * 우리는 $G_{small}$의 MSF $M_{small}$ 을, 각 small cluster에 대해서 $A(C, p)$ 를 초기화함으로써 관리할 것이다. 이는 $s_{high} \le m - 1$ 이라 가능하다.
* 큰 클러스터 $C$에 대해서는 $M_{small}(C) = \cup_{C^\prime:\text{$C$ 의 작은 자식 클러스터}} MSF(C^\prime)$ 이라고 정의하자. 여기서 우리는 다음 사실을 관찰할 수 있다.
  * **Proposition 6.4.** $M_{small} = \cup_{c:\text{large}} M_{small}(C)$ 
  * 루트 클러스터가 아닌 모든 큰 클러스터들에 대해서 Theorem 3.1에서 다룬 Dynamic Expander Pruning 알고리즘을 사용한다. 이 때 conductance guarantee는 $\alpha_0$ 을 사용한다. 어떠한 간선 $e \in C$ 가 지워질 때 Pruning은 $P_0^C \in V(C)$ 집합을 업데이트 할 것이다 (초기에 $P_0^C = \emptyset$ 이다.) 
* 루트 클러스터를 포함한 모든 클러스터들에 대해서 $P^C$ 를 정의할 수 있다.  $P^C = \cup_{C^\prime:\text{$C$ 의 큰 자식 클러스터}} P^{C^\prime}$ 이다. 이를 $C$ 의 *total pruning set* 이라고 하자. 
* 어떤 집합 $U \subseteq V(C)$ 에 대해서 $\overline{E}^C(U) = \{(u, v) \in E^C  u \in U \text{ or } v \in U\}$ 를 *$U$ 에 인접한 $C$-own edges* 라 정의한다. 
  * Chapter 5에서 $C$-own edges $E^C$ 를 정의한 것을 상기하자.

* **Definition 6.5 (Compressed cluster).** 임의의 큰 클러스터 $C$ 와 정점 집합 $U \subseteq \cup_{c^\prime: \text{large child of }C} P_0^{C^\prime}$ 에 대해서, *compressed cluster of $C$ with respect to $U$*, 혹은 $\overline{C}(U)$ 는 $C$ 에서 아래 세 연산을 해서 얻어진 그래프를 뜻한다.

  * 작은 자식 클러스터 $C^\prime$ 를 $MSF(C^\prime)$으로 대체하고
  * 큰 자식 클러스터 $C^\prime$ 를 하나의 노드로 contract하고 (super node 라 부름)
  * $U$ 에 인접하던 간선들을 모두 지우는 (즉 $\overline{E}^C(U)$ 를 지우는)

  * 이 때. 편의상, contract 과정에서 $C$-own edges들은 루프의 형태로 남겨두고, $C$-child edges들만 지우는 것으로 한다.

* $E^{\overline{C}(U)} = E^C - \overline{E}^C(U)$ 로 정의한다. 헷갈리지 않는 것이 불가능할 정도로 잘못된 표현이지만, 끔찍하게도 이 간선 집합은 이후 논의에서 핵심적이다. 알아서 조심해서 읽자.

아래 Lemma 6.6은 compressed cluster의 성질을 보여준다.

**Lemma 6.6**. 임의의 큰 클러스터 $C \in H$, $U \subseteq V(C)$ 에 대해서 우리는

* $E(\overline{C}(U)) = M_{small}(C) \cup E^{\overline{C}(U)}$
* $M_{small}(C) \subseteq MSF(\overline{C}(U))$

**Proof.** 

* 꼭지 1: 정의에 의해 $E(C) = \bigcup_{C^\prime: C \text{의 작은 자식 클러스터}} E(C^\prime) \cup \bigcup_{C^\prime: C \text{의 큰 자식 클러스터}} E(C^\prime) \cup E^C$ 가 성립한다. $\overline{C}(U)$ 를 만들면서, 작은 자식 클러스터에 대해서 $E(C^\prime) \rightarrow MSF(C^\prime)$이 되고, 큰 자식 클러스터에 대해서 $E(C^\prime)$ 는 공집합이 된다. 마지막으로, $U$ 에 인접한 에지들을 제거하면서 $\overline{E}^C(U)$ 가 사라진다. Proposition 6.4와 결합하면 첫 식이 성립한다.
* 꼭지 2의 증명은, $MSF(C - D) = \cup_{C^\prime : \text{child of } C} MSF(C^\prime - D) \cup (MSF(C - D) \cap (E^C - D))$ 라는 Theorem 5.2의 3번 성질로 증명할 수 있다. 일단. $M_{small}(C) \cup \bigcup_{C^\prime: C\text{의 큰 자식 클러스터}} MSF(C^\prime) \subseteq MSF(C)$ 이다. 이 성질에 의하여 contraction까지 단계를 거쳐도 $M_{small}(C) \subseteq MSF(C)$ 가 성립한다. 여기서 $U$ 에 인접한 간선들을 지운다 하더라도, 이들은 $M_{small}(C)$ 에 속하지 않는다. $M_{small}(C)$ 에 속하려면 양 끝점 모두가 작은 자식 클러스터에 속해야 하기 때문이다.

**Definition 6.7.** $S_{\overline{C}(U)} \subseteq V(\overline{C}(U))$ 를 $\overline{C}(U)$ 의 슈퍼 노드의 집합이라고 하자. Proposition 6.3에 의하여 $S_{\overline{C}(U)} = O(\gamma)$ 이다. 

이제 우리는 

* $E^{\overline{C}(U)} = E^C - \overline{E}^C(U)$ 를 세 가지 집합 $E_1^{\overline{C}(U)},E_2^{\overline{C}(U)},E_3^{\overline{C}(U)}$ 으로 나눌 것이다. 이 때 $E_i^{\overline{C}(U)}$ 는 $e$ 의 $i-1$ 개의 끝점이 $S_{\overline{C}(U)}$ 와 인접한 간선의 집합을 뜻한다. 
* $\overline{C}_i(U) = (V(\overline{C}), M_{small}(C) \cup E_i^{\overline{C}(U)})$ 이다 ($i = 1, 2, 3$). 

이렇게 정의한 이유를 다음 명제에서 알 수 있다.

**Proposition 6.8.** $i \in \{2, 3\}$ 에 대해서 $\overline{C}_i(U)$ 에 속한 모든 non-tree edge들은 $S_{\overline{C}(U)}$ 에 한 끝점이 있다.

**Proof.** $M_{small}(C) \subseteq MSF(\overline{C_i}(U))$ 이다. 고로 모든 non-tree edges들은 $E_i^{\overline{C}(U)}$ 에서 올 수밖에 없다. 이들은 정의상 $S_{\overline{C}(U)}$ 에 한 끝점이 있다.

우리는 각 큰 클러스터 $C$ 에 대해서 항상 $U = P^C$ 로 두고 compressed cluster를 관리할 것이다. $U = P^C$ 일 때 $C$ 의 Compressed cluster를 $\overline{C}$ 라고 부르자. 

**Proposition 6.9.** Compressed cluster들은 모두 Edge-disjoint하다. (증명 생략)

고로 이제 우리는 $\overline{C_1}, \overline{C_2}, \overline{C_3}$ 라는 세 가지 클러스터로 이 그래프에서 일어나는 일을 정리하였다. 각 클러스터는 다음과 같이 관리한다.

* $\overline{C_1}$ 은 non-tree edge가 작다.  사실 우리가 Theorem 4.1에서 정의한 알고리즘은 non-tree edge가 적을 때 효율적이다. 고로 그 알고리즘을 사용한다.
* $\overline{C_2}$ 는 Part 1의 맨 처음에 state한 Fact 5의 알고리즘을 사용한다.
* $\overline{C_3}$ 는 Part 1의 맨 처음에 state한 Fact 4의 알고리즘을 사용한다.

최종적으로 우리는 Sketch graph $H$를 정의할 수 있다: $E(H) = E^\neq(w) \cup M_{small} \cup \bigcup_{C:\text{large}} (\overline{E}^C(P^C) \cup \bigcup_{i = 1, 2, 3} MSF(\overline{C}_i) \cup J^C)$. 여기서 $J$ 는 *Junk edge* 라고 불리는 간선 집합인데 각 $C$ 마다 정의된다. 이후 후술한다. 초기에 $J^C = \emptyset$ 이다. 

이로써 우리는 전처리 알고리즘을 정리할 수 있다.

1. $\gamma = n^{O(\sqrt{ \log \log n / \log n})}$ 이라고 두고 Theorem 5.2의 $MSFdecomp$ 알고리즘을 실행한다. 이 때 $\alpha = 1/\gamma^3, d = \gamma, s_{low} = \gamma, s_{high} = n/\gamma$ 이다. 
2. 작은 클러스터 $C$ 에 대해서 $A(C, p)$ 를 초기화하고 $M_{small} = MSF(G_{small})$ 을 초기화한다.
3. 루트가 아닌 큰 클러스터 $C$ 에 대해서 Pruning을 conductance parameter $\alpha_0 = \Omega(1/\gamma^4)$ 로 두고 초기화한다.
4. 큰 클러스터 $C$ 에 대해서 $\overline{C_1}, \overline{C_2}, \overline{C_3}$ 을 구성한 후, $C_1$ 은 few non-tree edge decremental MSF 알고리즘으로 초기화하고, $C_2, C_3$ 은 각각 Fact 5 / 4의 알고리즘을 사용한다.
5. Sketch graph $H$를 구성하고, few non-tree edge dynamic MSF 알고리즘으로 초기화한다. (Bulk update를 지원하는 버전이다.)

큰 틀에서, few non-tree edge 알고리즘을 사용할 수 있는 이유가 우리가 사용한 expander pruning과, conductance guarantee를 보장하는 MSF decomposition 덕분이라고 생각하면 된다.

### 6.2 Update

우리는 위 조건을 만족시키면서 간선 제거를 어떻게 처리하는지 간단히 살펴본다. 보존해야 할 값은 $MSF(G) = MSF(H)$ 이며 $H$ 가 굉장히 sparse하다는 조건들이다.

관리해야 할 값은 5가지가 있다.

* $E^\neq(w)$
* $M_{small}$
* 모든 큰 클러스터 $C$ 에 대해서 $\overline{E}^C(P^C)$
* 모든 큰 클러스터 $C$ 에 대해서 $MSF(\overline{C}_i)$
* 모든 큰 클러스터 $C$ 에 대해서 $J^C$

지워야 할 간선을 $e$ 라고 하자. $G \leftarrow G - e, G^\prime \leftarrow G^\prime - e$ 라고 둔다. 항상 $e$ 가 $C$-own edge가 되는 유일한 클러스터 $C$ 가 정의상 존재하니, $E^C \leftarrow E^C - e$ 로 제거해 줄 수 있다. (이는 모든 조상에도 진행해야 한다.) 이에 따라서 Pruning set $P^C$ 도 바뀌며 $\overline{E}^C$ 역시 따라서 잘 바뀐다. ($P^C$ 가 항상 커지기만 한다는 점을 관찰하자). $M_{small}, E^\neq(w)$ 는 각각의 자료구조를 직접 건드리면서 수정할 수 있다. $M_{small}$ 과 $\overline{E}^C$ 가 바뀌면 이에 따라서 자연스럽게 $C_i$ 도 바뀐다. 최종적으로, 어떠한 간선이 $MSF(\overline{C}_i)$ 를 빠져나지만 실제로 $G$ 에서 제거된 것이 아니면 이 간선을 *Junk edge* 집합에 넣는다.

