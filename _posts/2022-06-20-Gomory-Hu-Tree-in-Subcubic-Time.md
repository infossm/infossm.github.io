---
layout: post
title:  "Gomory-Hu Tree in Subcubic Time"
date:   2022-06-20
author: ainta
tags: [algorithm, graph theory]
---


## Introduction

### mincut terminology

무방향 그래프 $G = (V,E)$ 의 각 edge에 대해 양의 정수 weights $w: E \rightarrow Z_{+}$ 정의되어 있다고 하자. 정점들의 집합 $S$가 $s \in S, t \notin S$를 만족할 때 $S$를 $(s,t)$-cut 이라 한다. 이 때, $S$의 weight $\delta(S) = \Sigma_{e\in E(S, V \setminus S)} w(e)$ 가 최소인 경우를 ($s$,$t$)-mincut 이라 하고, 그 weight를 $\lambda(s,t)$라 한다.

### Gomory-Hu Tree

Gomory-Hu Tree는 모든 $(s,t)$에 대한 mincut을 encoding하는 Tree이다. 다음과 같은 두 definition으로 정의 가능하다.

**Definition 1.** $G = (V,E)$의 Gomory-Hu Tree $T$에서 두 점 $(s,t)$ path 상의 minimum weighted edge의 weight는 $\lambda(s,t)$와 동일하다. 

**Definition 2.** $G = (V,E)$의 Gomory-Hu Tree $T$의 각 edge $e = (u, v)$에 대해 $T-e$ 의 두 component(subtree) 중 $u$를 포함한 것을 $T(u)$라 하면 $T(u)$는 $G$에서 $(u,v)$-mincut이다.

**Theorem 1.** $G = (V,E)$에 대해 Definition 2를 만족하는 트리 $T$는 Definition 1을 만족한다.

**Proof.** Definition 2의 성질을 만족하는 트리 $T$가 있다고 하자. 세 정점 $u, v, w$에 대해 $\lambda(u,v) \ge min(\lambda(u,w), \lambda(v,w))$ 가 성립함은 자명하다. 따라서, 두 정점 $u, v$ 에 대해 $(u, v)$ path 상의 edge들을 $e_1, e_2, .., e_k$라 하면 $\lambda(u,v) \ge min(w(e_1), w(e_2), ..., w(e_k))$가 성립한다. 한편, $e_1, .., e_k$ 중 minimum weighted edge를 $e = (s, t)$라 하면 $T-e$에서 $s$를 포함하는 component $T(s)$는 정의에 의해 $(s,t)$-mincut이 되고, $T(s)$는 $u$를 포함하며 $v$를 포함하지 않으므로 $\lambda(u,v) \le w(e)$가 된다. 따라서, $\lambda(u,v) = min(w(e_1), w(e_2), ..., w(e_k))$ $\blacksquare$

![Figure 1. 그래프(왼쪽)과 Gomory-Hu Tree(오른쪽)](/assets/images/gomory-hu-tree/img1.jpeg)

Gomory-Hu Tree는 (n-1)번의 Minimum Cut으로 해결할 수 있으며, 최근까지도 그것이 알려진 가장 효율적인 알고리즘이었다. 이 포스트에서는 Gomory-Hu Tree를 해당 방법보다 빠르게 구축할 수 있는 방법에 대해 소개한다.

### Basic Gomory-Hu Tree Algorithm

$G = (V,E)$ 에 대해, Gomory-Hu Tree를 (n-1)번의 Minimum cut oracle call로 구하는 방법을 소개한다. 먼저, 임의의 서로 다른 두 정점 $s, t$를 골라 $(s,t)$-mincut $A$를 구한다. 최종 트리에서 $A$와 $V \setminus A$는 각각 subtree가 될 것이고, 둘을 연결하는 edge의 weight은  $\delta(A)$가 된다. 이제 마찬가지로 $A$에서도 두 정점 $p, q$를 골라 $(p,q)$-mincut $B$를 구하자. 여기서 주목할 점은 $A \cap B$ 역시 $(p,q)$-mincut이 된다는 것이다 (Lemma 2). 그러면 이제 $A$ 역시 $A \cap B$ 와 그 외 부분의 두 subtree로 나눌 수 있다. 이와 같은 성질을 이용하면 $V$ 에서 시작해서 재귀적으로 $A$와 $A \setminus V$에서 문제를 풀고, 두 개의 트리가 나오면 $s, t$ 를 $\delta(A)$ weight의 edge로 이어주면 Definition 2를 만족하는 트리를 구축할 수 있다.

**Lemma 2.** $G = (V,E)$에서 두 정점 $s, t$에 대해 $A$가 $(s,t)$-mincut, $p, q \in A$에 대해 $B$가 $(p,q)$-mincut인 경우 $A \cap B$ 역시 $(p,q)$-mincut이다.

**Proof.** $B$와 $V \setminus B$ 중 하나는 $t$를 포함한다. 일반성을 잃지 않고, $t \notin B$라고 가정하자. cut function의 **submodularity**에 의해, $\delta(A \cap B) + \delta(A \cup B) \le \delta(A) + \delta(B)$가 성립한다. 만약 $A \cap B$가 $(p, q)$-mincut이 아니라면, $\delta(A \cap B) > \delta(B)$ 이므로 $\delta(A \cup B) < \delta(A)$가 성립한다. 그런데 $s \in A \cup B, t \notin A \cup B$ 이므로 이는 $A$가 $(s,t)$-mincut 이라는 가정에 모순이다. 따라서, $A \cap B$는 $(p,q)$-mincut $\blacksquare$

# Gomory-Hu Tree in Subcubic Time

## 개요

Gomory-Hu Tree는 1961년 발표된 오래된 알고리즘이지만 오랜 시간동안 (n-1)번의 mincut(maxflow) 계산보다 효율적인 알고리즘이 나오지 않았다. [Gomory-Hu Tree in Subcubic Time](https://arxiv.org/pdf/2111.04958.pdf) 은 여러가지 최신 테크닉을 이용해 이를 $O(n^{2+o(1)})$ 시간만에 해결하는 방법을 소개하는데, 이 포스트에서 모두 다루기에는 너무 방대한 내용이 있어 중요한 몇몇 부분과 elementary한 내용을 주로 다룬다.

여기서 Gomory-Hu Tree 문제를 효율적으로 접근하는 방법을 세 개의 스텝으로 나누면 아래와 같다.

1. Gomory-Hu Tree problem을 Single-Source Mincut problem으로 reduction
2. Single-Source Mincut problem을 Guide Tree가 주어졌을 때 효율적으로 해결
3. 주어진 그래프에 대해 Guide Tree를 construct

Single-Source Mincut이나 Guide Tree가 무엇인지에 대해서는 앞으로 차차 소개할 것이다. 먼저 세 개 중 1번 스텝은 이 포스트에서 크게 다루지 않을 것이다. 자세한 reduction은 [A Nearly Optimal All-Pairs Min-Cuts Algorithm in Simple Graphs](https://arxiv.org/pdf/2106.02233.pdf)의 부록에 설명되어 있다. 이번 포스트에서는 2번 스텝에 집중하여 어떤식으로 문제를 해결하였는지 살펴볼 것이다. elemnatary한 문제부터 시작해서 조금씩 더 어려운 문제를 해결해나가는 과정이 제법 흥미롭게 전개된다. 그 다음에 3번 스텝에서 Guide Tree를 만드는데 어떤 방법을 썼는지까지 간단하게 알아볼 것이다.

## Single Source mincut

### terminology

$G = (V,E)$에서 서로소인 두 정점 집합 $A, B \subset V$에 대해 $A \subset S, B \cap S = \phi$ 일 때 $S \subset V$를 $(A,B)$-cut이라 한다(정점 $(s,t)$ cut의 확장으로 볼 수 있다). 이 때 $\delta(S)$가 최소인 경우를 $(A,B)$-mincut이라 한다. 

$(A,B)$-mincut $S$가 모든 $(A,B)$-mincut에 포함될 때, $S$를 $A$-closest $(A,B)$-mincut이라 한다. 서로소인 $A, B$에 대해 $A$-closest $(A,B)$ mincut의 존재성은 아래 Lemma로부터 간단히 보일 수 있다 

**Lemma 3.** 두 $(A,B)$ mincut의 교집합은 $(A,B)$ mincut이다.

**Proof.** $(A,B)$-mincut $S_1, S_2$ 에 대해, $S_1 \cap S_2, S_1 \cup S_2$는 $(A,B)$-cut이다. 앞서도 살펴본 cut function의 submodularity에 의해 $\delta(S_1 \cap S_2) + \delta(S_1 \cup S_2) \le \delta(S_1) + \delta(S_2)$ 이므로, $S_1 \cap S_2$는 $(A,B)$-mincut일 수밖에 없다. $\blacksquare$

$(A,B)$-mincut은 maxflow로 쉽게 구할 수 있고, $A$-closest $(A,B)$-mincut은 $(A,B)$-mincut이 주어지면 linear time에 구할 수 있음이 알려져 있다.

### Isolating Cut

**Theorem 4 (Isolating Cuts).** $G=(V,E)$와 terminal set $T \subset V$가 주어졌을 때, 모든 $v \in T$에 대해 $v$-closest $(v, T \setminus v)$-mincut $S_v$를 계산할 수 있다 ($S_v$를 min $v$-isolating cut이라 한다).running time은 $G$에서 maxflow를 계산하는 시간을 $T_{maxflow}$라 할 때, $T_{maxflow} \log \lvert T \rvert$ 이다.

**Proof.** 먼저, $\lvert T \rvert$개의 terminal을 $h = \log \lvert T \rvert$ 개의 비트로 나타내자. 그리고 $i$번째 비트가 0인 terminal들의 집합을 $A_i$, 1인 terminal들의 집합을 $B_i$라 하고, $C_i$를 $(A_i, B_i)$-mincut이라 하자. terminal $v$에 대해서, 각 $i$마다 $C_i$와 $V \setminus C_i$ 중의 정확히 하나가 $v$를 포함할 것이다. 이를 $D_i(v)$라 하고, $D_0(v), D_1(v), .., D_{h-1}(v)$의 교집합을 $U_v$라 하자. $v$가 아닌 다른 terminal은 $v$와 적어도 하나의 비트에서 다르므로 $U_v \cap T =v$가 성립한다. 

한편, $U_v$는 min $v$-isolating cut을 포함하는데, min $v$-isolating cut $S_v$에 대해 $S_v \cap U_v$ 역시 cut function의 submodularity에 의해 min $v$-isolating cut이기 때문이다.  따라서, 각 $v \in T$에 대해 $V \setminus U_v$를 contract하여 한 정점 $t_v$로 압축한 그래프를 $G_v$라 하면 $G_v$의 $v-closest (v, t_v)$-mincut은 $G$에서의 min $v$-isolating cut $S_v$가 된다. 

$\log \lvert T \rvert$번의 maxflow를 통해 $(A_i, B_i)$-mincut을 구하면 이를 통해 선형 시간에 각 $v$에 대한 $U_v$를 구할 수 있다 (각 mincut의 edge를 disconnect한 후 component 구하기). $V$의 각 정점은 $C_i$와 $V \setminus C_i$ 둘 중 하나에만 포함되므로 하나의 $U_v$에만 포함된다. 한 edge는 두 정점을 잇기 때문에 각 edge도 최대 2개의 $G_v$에서만 계산된다. 따라서, 각 $G_v$에 대해 $v-closest (v, t_v)$-mincut을 구하는 시간은 $2T_{maxflow}$라고 볼 수 있다. 총 running time은 처음 $\log \lvert T \rvert$번의 maxflow에 의해 결정되므로 $T_{maxflow} \log \lvert T \rvert$. $\blacksquare$

### Steiner Mincut

**Definition 3 (Steiner Mincut).** $G=(V,E)$와 terminal set $T \subset V$가 주어졌을 때, 모든 $s, t \in T$ 에 대한 $(s,t)$-mincut 중 minimum weighted cut $S$ 을 Steiner mincut이라 하고, 그 weight를 $\lambda(T)$라 한다.

**Theorem 5.** Steiner Mincut을 $O(\log^3 n) \cdot T_{maxflow}$ 의 running time으로 구하는 알고리즘이 존재한다.

**Proof.** $\lvert S' \cap T \rvert \le \lvert T \setminus S' \rvert$ 를 만족하는 Steiner mincut $S'$ 를 생각하자 ($S'$가 부등식을 만족하지 않으면 $V \setminus S'$를 잡으면 된다). 

만약 $\lvert S' \cap T \rvert = 1$이라면, 둘 모두에 포함되는 유일한 정점 $v$에 대해 min $v$-isolating cut $S_v$는 $\delta(S_v) \le \delta(S')$ 이므로 $S_v$도 steiner mincut이 된다. 따라서, $\lvert S' \cap T \rvert = 1$를 만족하는 steiner mincut $S'$가 존재하는 경우 $IsolatingCut(T)$를 호출하여 모든 terminal에 대한 mincut중 minimum weight인 것이 steiner mincut이다.

이 아이디어를 교집합에 원소가 하나 초과인 경우로 확장해 보자. $\lvert S' \cap T \rvert \in [2^i, 2^{i+1})$ 인 $i$가 존재할 것이다. $T' = Sample(T, \frac{1}{2^i})$라 하자. 즉, $T$의 각 정점을 독립적으로  $\frac{1}{2^i}$의 확률로 포함시켜 만든 집합을 $T'$라 하면 $Pr(\lvert S' \cap T' \rvert = 1) \in \Omega(1)$ 이 성립한다. 이에 따라, 샘플링하여 $T'$를 만드는 작업을 $O(\log n)$번 반복하면 $\lvert S' \cap T' \rvert = 1$이 성립하는 경우가 나올 것이고, 이는 앞서 살펴본 $\lvert S' \cap T \rvert = 1$의 케이스와 같다. 

정리하면, $i = 0, 1, .., \log \lvert T \rvert$에 대해 각각 $T' = Sample(T, \frac{1}{2^i})$ 을 만들어 $IsolatingCut(T')$을 호출하는 작업을 $O(\log n)$번 함으로써 그 중 min-weighted cut을 리턴하는 것으로 Steiner Mincut을 구할 수 있다. running time은 $\log \lvert T \rvert \log n \cdot T_{IsolatingCut}$ = $O(\log^3 n) \cdot T_{maxflow}$ $\blacksquare$

### Single Source Mincut Threshold

**Definition 4 (Single Source Mincut Threshold).** $G=(V,E)$와 terminal set $T \subset V$와 source $s \in V$, threshold $\tau$가 주어졌을 때 각 $v \in T$에 대해 $\lambda(s, v) > \tau$가 성립하닌지 아닌지를 판정하는 문제를 Single Source Mincut Threshold (SSMT)라 한다.

**Theorem 6.** Single Source Mincut Threshold를 $O(poly\log n) \cdot T_{maxflow}$ 시간에 해결하는 알고리즘이 존재한다.

**Proof.** $T_{small}$을 $\lambda(s, v) \le \tau$가 성립하는 terminal $v$들의 집합이라고 정의하자. 당연하게도 $T_{small}$을 구하면 SSMT는 해결이 된다. $T_{small}$의 부분집합을 리턴하는 subroutine $Thresholdstep(T)$이 존재하여 $\mathbb{E}[ \lvert Thresholdstep(T) \rvert] \in \Omega( \frac{\lvert T_{small} \rvert}{ \log \lvert T \rvert } )$을 만족한다고 가정하자. 

$T' = T$에서 시작하여  $T' \leftarrow T' \setminus Thresholdstep(T')$를 반복해 나간다면 어느 순간 $T' = T \setminus T_{small}$이 될 것이라고 짐작할 수 있다. $(1 - \frac{1}{\log \lvert T \rvert})^k$는 $k = O(\log^2 n)$에 대해 충분히 작은 값이 되므로 $O(\log^2 n)$번의 Thresholdstep call을 통해 SSMT를 해결할 수 있다. 

그러면 이제 조건을 만족하는 $Thresholdstep(T)$이 존재함을 보이자.

**Lemma 7.** $T_{small}$의 부분집합을 리턴하는 subroutine $Thresholdstep(T)$이 존재하여 $\mathbb{E}[ \lvert Thresholdstep(T) \rvert] \in \Omega( \frac{\lvert T_{small} \rvert}{ \log \lvert T \rvert } )$을 만족한다.

**Proof.** 각 $i = 0, 1, .., \log \lvert T \rvert$에 대해 독립적으로 다음과 같은 과정들을 수행하자.

1. $R \leftarrow \{ s \} \cup Sample(T, \frac{1}{2^i})$
2. $IsolatingCut(R)$ 로 각 $v \in R$에 대해 $S_v$ 계산
3. $F \leftarrow \{ S_v \lvert \delta(S_v) \le \tau, v \neq s \}$
4. $D \leftarrow \cup_{S_v \in F} (S_v \cap T)$

$Thresholdstep(T)$를 모든 iteration에서 계산된 $D$들의 합집합을 리턴하는 subroutine으로 정의하자.

각 $v \in D$에 대해, $v$는 어떤 $u \in R$에 대해 $S_u$에 포함되고 $\delta(S_u) \le \tau, s \notin S_u$가 성립하므로 $\lambda(s, v) \le \tau$이다. 따라서, $D \subset T_{small}$임을 알 수 있다. 또한 하나의 iteration에서 isolating cut들은 disjoint하므로 $\lvert D \rvert = \Sigma_{S_v \in F} \lvert S_v \cap T \rvert$이다.

한편, $v \in T_{small}$인 $v$에 대해 $v$-closest $(s,v)-mincut$을 $S_v$라 하면 iteration $i = \log \lvert S_v \cap T \rvert$ 에서 $S_v \cap R = \{ v \}$ 를 만족할 확률이 $\Omega(\frac{1}{\lvert S_v \cap T  \rvert})$ 가 되고, 그 경우 $D$의 원소의 개수가 $\lvert  S_v \cap T \rvert$ 만큼 증가한다. 즉, 각 정점 $v \in T_{small}$ 가 iteration $i = \log \lvert S_v \cap T \rvert$ 의 $D$에서 증가시키는 원소 개수의 기댓값은 $\Omega(1)$이다.

따라서, 모든 iteration에서 계산되는 $D$의 크기의 합은 $\Omega(\lvert T_{small} \rvert)$ 이며, $\mathbb{E}[ \lvert Thresholdstep(T) \rvert] \in \Omega( \frac{\lvert T_{small} \rvert}{ \log \lvert T \rvert } )$가 성립한다. $\blacksquare$

이제 Single Source Mincut Threshold로 돌아가 running time을 계산해보자. 하나의 Thresholdstep이 $\log \lvert T \rvert$번의 Isolating cut으로 이루어지고, $O(\log^2 n)$ 번의 Thresholdstep으로 Single Source Mincut Threshold를 해결할 수 있으므로 SSMT를 $O(poly\log n) \cdot T_{maxflow}$ 시간에 해결하는 알고리즘이 존재한다. $\blacksquare$

**Corollary 8.** $G=(V,E)$와 terminal set $T \subset V$와 source $s \in V$가 주어졌을 때 $t \in T인 (s, t)$-mincut중 maximum weighted mincut을 $O(poly\log n) \cdot T_{maxflow}$ 시간에 구할 수 있다.

**Proof.** Theorem 6의 결과에서 $\tau$에 대한 binary search를 통해 간단히 해결할 수 있다.

### Single Source Mincut

### terminology

$G = (V,E)$와 terminal $U \subset V$에 대해, $U$를 포함하는 tree를 $U$-steiner tree라 한다 ($G$의 subgraph일 필요는 없음). $U$-steiner tree $T$를 생각하자. $S \subset V$가 $T$의 edge와 최대 $k$번 cross할 때, 즉 $x \in S, y \notin S$ 인 $(x, y) \in E_T$가 $k$개 이하일 때 $S$를 $T$에 대한 $k$-respecting set이라 한다.  모든 $v \in U \setminus s$ 에 대해, $(s,t)$-mincut $C_v$가 존재하여 $C_v$가 $T$에 대한 $k$-respecting set인 경우 $T$를 $k$-respecting **guide tree**라 한다. 또한, 트리 $T_1, T_2,.., T_h$이 존재하여 모든 $v \in U \setminus s$에 대해 $(s,v)$-mincut이 존재하여 적어도 하나의 $T_i$의 $k$-respecting set일 때 $T_1, T_2,.., T_h$를 $k$-respecting guide trees라 한다.

**Definition 5 (Single Source Mincut).** $G = (V,E)$와 terminal $U \subset V$, source $s \in U$가 주어졌을 때, 모든 $v \in U$에 대해 $\lambda(s, v)$를 계산하는 문제를 **Single Source Mincut** 이라 한다.

**Theorem 9 (Single Source Mincut Given Guide Tree).** $G = (V,E)$와 terminal $U \subset V$, source $s \in U$, $U$-steiner tree $T$가 주어졌다고 하자. 양의 정수인 상수 $k$에 대해 특정 알고리즘이 존재하여 모든 $v \in U$에 대해 $\tilde{\lambda}(s, v)$를 계산한다. 이 때, 해당 알고리즘은 $\lambda(s, v) \le \tilde{\lambda}(s, v)$을 만족하고 만약 $(s,v)$-mincut이 존재하여 $T$에 대한 $k$-respecting set이라면  $\lambda(s, v) = \tilde{\lambda}(s, v)$이 성립한다. running time은 $O(poly\log n) \cdot T_{maxflow}$ 이다.

Theorem 8의 알고리즘을 살펴보기 전에, 여기 이용될 Isolating Cut에 대해 알아보자. 여기서 이용되는 Isolating Cut은 약간 다르게 terminal set 대신 disjoint set들의 collection들이 주어지는 버전이다.

**Lemma 10 (Isolating Cuts - Set version).** $G=(V,E)$와 set들의 collection $T_1, T_2, .., T_h$가 주어졌을 때, 모든 $T_i$에 대해 $T_i$-closest $(T_i, \cup_{j \neq i} T_j)$-mincut $S_i$를 계산할 수 있다. running time은 $G$에서 maxflow를 계산하는 시간을 $T_{maxflow}$라 할 때, $T_{maxflow} \log \lvert T \rvert$ 이다.

Set version도 vertex 버전과 동일하게 작동하고, 동일한 방법으로 쉽게 증명할 수 있다. 이제 진짜 알고리즘을 소개할 차례다.

**Algorithm 1.**

$SingleSourceMincutEstimation(G = (V,E), T, k)$은 그래프 $G$와 $s$를 포함하는 트리 $T$, 양의 정수 $k$를 인자로 가지는 알고리즘이다 (이후에는 $SSME(G,T,k)$로 표기). $V(T) = T \cap V$를 terminal 정점들이라고 한다. 이 알고리즘은 $v \in V(T)$에 대해 $\tilde{\lambda}(s, v)$를 계산하며, 처음에는 모두 $\tilde{\lambda}(s, v) = \infty$로 초기화된 상태이고 알고리즘 내에서 이 값을 $x$로 update하는 작업을 몇 번 하여 mincut을 estimate하는데, update한다는 뜻은 이때까지 계산한 값보다 작은 경우에 갱신해준다는 의미이다. 이제 실제 알고리즘이 어떻게 동작하는지 살펴보자.

0. 상수 $C$에 대해 $\lvert V(T) \rvert \le C$ 이라면 각 $t \in V(T)$에 대해 $(s, t)$를 $O(1)$ 번의 maxflow call로 계산하여 update한다. 다음 스텝부터는 $V(T)$의 크기가 $C$보다 크다고 가정한다.

1. $T$에서 정점 $c$가 존재하여 $c$를 root로 놓았을 때, 각 child들의 subtree가 $V(T)$의 vertex들 중 절반 이하만 포함하도록 할 수 있다. 이 때의 $c$를 centroid라 한다. 만약 $c \in V(T)$이면 $(s,c)$-mincut을 계산하여 update한다.

2. $c$의 자식 정점 $u_1, ... u_l$에 대해 $u_i$를 root로하는 subtree를 $T_i$라 하자. $V(T_1), V(T_2), .., V(T_l), \{ c \}$에 대해 Isolating Cuts - Set version(Lemma 9)을 호출하여 $S_i = (V(T_i), V(T) \setminus V(T_i))$-mincut을 계산한다.

3. 각 $T_i$에 대해, $S_i$ 이외의 정점들을 하나의 정점으로 contract해서 만든 새로운 그래프 $G_i$를 생각하자. 
 $s \notin S_i$라면 contract해서 나온 정점을 새로운 $s$로 두고 $T_i' = T_i \cup (s,u_i)$로 두고 $SSME(G_i, T_i', k)$를 호출한다.
 $s \in S_i$라면 contract해서 나온 정점을 $c_i$로 라벨링하고 $T_i' = T_i \cup (c_i,u_i)$로 두고 $SSME(G_i, T_i', k)$를 호출한다.
$v \in V(T_i)$에 대해 $SSME(G_i, T_i', k)$에서 나온 $\tilde{\lambda}(s, v)$로 update한다. 또한, $s \in S_i$인 경우 $v \in V(T) \setminus V(T_i)$에 대해 $\tilde{\lambda}(s, c_i)$로 update한다.

4. $T_1, .., T_l$ 중 $s$를 포함하는 것을 $T_s$라 하자. $T_s$를 제외한 $T_i$들을 $\frac{1}{2}$확률로 sampling하고, $T_s$는 $1$의 확률로 sampling하여 $T$에서 sampling되지 않은 모든 $T_i$들을 제거하여 만든 트리를 $T'$이라 하자. $SSME(G, T', k-1)$를 호출하여 $t \in V(T')$에 대해 호출에서 나온 $\tilde{\lambda}(s, t)$로 update한다. 이 과정을 $O(\log n)$ 번 반복한다.

5. $s \neq c$인 경우에만 다음 과정을 진행한다. 먼저, Corollary 8을 이용해  $\lambda_{max} = \max \{ \lambda(s,t): t \in V(T) \setminus V(T_s) \}$ 와 $argmax$에 해당하는 $t$들을 구한다. 해당 $t$ 들에 대해 $\tilde{\lambda}(s, t)$를 $\lambda_{max}$로 update하고, $t$들 중 하나를 골라 새로운 $s$로 둔다. $SSME(G, T-T_s, k-1)$을 호출하여 (여기서 $T_s$는 새로운 $s$가 아닌 이전 $s$가 있던 $T_s$) 나온 값으로 $t \in T-T_s$에 대해 $\tilde{\lambda}(s, t)$를 update한다.

**Lemma 11.** Algorithm 1은 Theorem 9의 조건을 만족하는 알고리즘이다.

**Proof.** $(s,t)$-mincut $C$가 $T$에 대한 $k$-respecting set이라고 하자. 만약 centroid $c$가 $t$와 같다면 step 1에서 $\lambda(s,t)$가 올바르게 계산된다. 이제 $c \neq t$인 경우를 생각하자. $t$를 포함하는 $T_i$를 $T_t$라 놓자. $E_T$ 중 $C$가 cross하는 edge들을 cut edge라고 할 떄, cut edge는 $k$개 이하이다. 또한, $s$와 $t$를 잇는 path위의 edge중 적어도 하나는 cut edge이고, 이 edge들은 $T_t$ 또는 $T_s$와 인접하다. 

cut edge가 어떤 subtree와 인접한지에 따라 3가지 경우로 나누어 살펴보자.

Case 1. 모든 cut edge가 오직 하나의 $T_i$와 인접한 경우
 
$C$와 $V \setminus C$ 중 하나는 $V(T)$에 포함되는 모든 정점이 $V(T_i)$에 포함된다. 이를 $A$라 하면 $A \cap V(T) = A \cap V(T_i)$가 성립한다. 이에 따라, k-respecting set $A$는 step 3의 $SSME(G_i, T_i', k)$ 호출에서 계산되어 $\lambda(s,t)$에 update될 것이다.

Case 2. cut edge $e$가 존재하여 $T_s$나 $T_t$가 아닌 다른 $T_i$와 인접한 경우

Step 4에서 $T_i$가 sampling되지 않는다면 $C$는 $T'$의 $k-1$-respecting set이 되어 $SSME(G, T-T_s, k-1)$에서 올바른 값을 계산하여 update한다.

Case 3. $T_s$와 인접한 cut edge, $T_t$와 인접한 cut edge가 있는 경우

Case 1, 2에 포함되지 않으므로 $T_s \neq T_t$ 이다. 만약 $\lambda(s, t) = \lambda_{max}$이면 Step 5에서 $\lambda_{max}$로 올바르게 계산된다. 그렇지 않다면, $(s,t)$-mincut $C$에 대해 $s'$는 $s$와 같은 편에 있게 된다 ($t$와 같은 편이라면 $\lambda(s, s') = \lambda_{max}$에 모순). 따라서, $T$에서 $s$와 $s'$의 위치를 바꾸어도 $C$는 $T$에 대한 $k$-respecting set이 되고, cut edge는 트리의 동일한 위치에 존재하게 된다. 한편, $T_s$는 cut edge가 적어도 하나 존재하므로 $C$는 $T-T_s$에 대한 $k-1$-respecting set이다. 따라서, $SSME(G, T-T_s, k-1)$에서 올바르게 계산되어 $\lambda(s,t)$가 update된다.

따라서, Algorithm 1에서 계산한 $\tilde{\lambda}$는 $\lambda(s, v) \le \tilde{\lambda}(s, v)$을 만족하고 $v$에 대해 $(s,v)$-mincut이 존재하여 $T$에 대한 $k$-respecting set이라면  $\lambda(s, v) = \tilde{\lambda}(s, v)$이 성립한다.

시간복잡도를 분석해보면, 각각의 재귀호출에서 $V(T)$가 절반으로 감소하거나 $k$가 감소함을 알 수 있다. 또한 각 step들에서 계산은 $O(poly\log n) \cdot T_{maxflow}$ 시간이 소모되므로 $k$가 constant일 떄 running time은 $O(poly\log n) \cdot T_{maxflow}$ 가 된다. 따라서, Algorithm 1은 Single Source Mincut Given Guide Tree를 해결하는 조건에 맞는 알고리즘이다. $\blacksquare$

$k$-respecting guide trees $T_1, T_2, .., T_h$가 주어지면 각각의 guide tree에 대해 위 Algorithm 1로 계산 후 가장 최소값을 선택하면 각 $t$에 대해 적어도 하나의 mincut이 하나의 guide tree에 대한 $k$-respecting set이므로 $\lambda(s,t)$를 올바르게 계산한다.

이제 guide tree(s)가 주어졌을 때 Single-Source Mincut이 효율적으로 해결 가능함을 보였다. 그렇다면 guide tree는 어떤 식으로 구성할 수 있을까?

## Constructing Guide Trees

**Definition 5 (U-steiner subgraph packing).** $G = (V,E)$와 terminal $U \subset V$가 주어졌을 때, $G$의 subgraph $H$에서 $U$의 모든 vertex가 connected일 때 $H$를 $U$-steiner subgraph라 한다. $U$-steiner subgraphs $H_1, H_2, .., H_k$에 대해 양의 value를 부여하여 각 $i \le k$에 대해 $val(H_i) > 0$, 각 $e \in E$에 대해 $\Sigma_{e \in H} val(H) \le w(e)$가 성립할 때 $U$-steiner subgraph의 collection $P = H_1, .., H_k$를 feasible $U$-steiner subgraph packing이라 하고, $val(P) = \Sigma_{H \in P} val(H)$ 를 packing $P$의 value라 한다. 

$pack(U)$를 feasible $U$-Steiner subgraph packing value의 최댓값이라 하면, 이 값은 Steiner mincut $\lambda(U)$의 근삿값이 된다.

**Lemma 12.** $\frac{\lambda(U)}{2} \le pack(U) \le \lambda(U)$

Multiplicative Weight Update (MWU)를 기반으로 한 Steiner subgraph packing에 대한 $(2 + \epsilon)$-approximation algorithm이 존재한다. 이를 통해, $val(P) \ge \frac{\lambda(U)}{4+\epsilon}$인 packing $P$를 구할 수 있다.

$S$를 $U$에 대한 steiner mincut이라 하자.

$$val(P) \le \Sigma_{H \in P} val(H) \le \Sigma_{H \in P} \lvert E_H(S, V \setminus S) \rvert val(H) = \lambda(U)$$

가 성립하고, $P$의 steiner subgraph들 중 $S$와 5번 이상 cross하는 것들의 집합을 $P_5$, 그렇지 않은 것을 $P_4$라 하면 위 부등식과 동일한 방법으로 $val(P_5) \le \frac{\lambda(U)}{5}$임을 보일 수 있다. $\epsilon$을 작게 잡으면 $\frac{\lambda(U)}{4+\epsilon} > \frac{\lambda(U)}{5}$가 되도록 하게 할 수 있으므로, $P_4 \ge c_{\epsilon}\lambda(U)$를 보장하는  $c_{\epsilon}$이 존재한다.

만약 모든 $\lambda(s,t)$가 $\lambda(U)$ 근처의 값을 가진다면, 위와 같은 방법으로 해당 $(s,t)$-mincut과 4번 이하만 cross하는 $U$-steiner subgraph의 존재성을 보장해줄 수 있다. 실제로, 모든 $s, t \in U$에 대해 $\lambda(U) \le \lambda(s,t) \le 1.1 \lambda(U)$를 가정한 상태에서 위에서 소개한 것과 유사한 아이디어로 guide trees를 construct할 수 있다.

## Return to Gomory-Hu Tree

$\lambda(U) \le \lambda(s,t) \le 1.1 \lambda(U)$라는 강한 조건이 있을 때만 guide trees를 만들 수 있다면 그 때만 Single-Source Min cut을 해결할 수 있을텐데, 그러면 Gomory-Hu Tree를 만들기에는 역부족이 아닐까 의문이 들 수 있다. 그러나 [A Nearly Optimal All-Pairs Min-Cuts Algorithm in Simple Graphs](https://arxiv.org/pdf/2106.02233.pdf) 에서 소개한 Gomory-Hu Tree의 reduction은 해당 조건 하에서 Single-Source Min Cut까지도 필요하지 않고 Single-Source Min Cut의 값이 주어지면 맞는지 판정하는 verification만 주어진다면 Gomory-Hu Tree를 구할 수 있음을 보이기 때문에 충분하다.

## 참고 자료

* https://www.youtube.com/watch?v=0tDBxHzy-pg (recommended)
* https://en.wikipedia.org/wiki/Gomory%E2%80%93Hu_tree
* https://arxiv.org/pdf/2111.04958.pdf
* https://arxiv.org/pdf/2106.02233.pdf
