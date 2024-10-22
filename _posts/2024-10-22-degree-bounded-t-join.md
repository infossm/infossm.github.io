---
layout: post
title: "Iterated Rounding을 이용해 Degree Bounded T-join 문제 해결하기"
date: 2024-10-22
author: leejseo
tags: [algorithm, graph]
---

# Introduction

$T$-join 문제는 가중치 있는 무향 그래프 $G = (V, E, c)$ 와 짝수 크기의 정점의 부분집합 $T \subseteq V$ 가 주어질 때, $T$에 속하는 정점의 차수는 홀수, 그 외 정점의 차수는 짝수가 되게 하는 최소 비용의 $J \subseteq E$ 를 찾는 문제이다.
이 문제는 다양한 응용이 있으며, 대표적인 예로는 [중국인 우채부 문제(Chinese Postman Problem)](https://en.wikipedia.org/wiki/Chinese_postman_problem)와 외판원 문제(TSP)에 대한 [Christofides Algorithm](https://en.wikipedia.org/wiki/Christofides_algorithm)이 있다.
과거 필자가 소개한 $s$-$t$ 경로 외판원 문제의 근사 알고리즘 또한 $T$-join 문제를 이용해 해결하였다.

$T$-join 문제의 경우, 다음과 같이 정의되는 LP relaxation의 optimal solution이 integral 하다는 좋은 성질이 알려져 있어 이를 기반으로 다항 시간에 해결할 수 있다.

$$ \begin{align*} \text{minimize:} & & \sum_{e \in E} c_e \cdot x_e && \\ \text{subject to:} & & x(\delta(S)) \ge 1 & &\forall S \subseteq V, \lvert S \cap T \rvert \equiv 1\pmod2 \\ & & x_e \ge 0 & & \forall e \in E \end{align*}$$

참고로, 여기에서 $\delta(S)$는 $S$에 대응하는 cut set을 의미하며, $A \subseteq E$에 대해 $x(A) = \sum_{e \in A} x_e$로 정의한다.

그러나, 정점에 차수 제약 조건이 있는 경우의 $T$-join을 생각해본다면, 이 문제를 해결하는 것은 그리 자명하지 않음을 알 수 있다. 이 글에서는 degree bounded $T$-join 문제를 해결하는 LP 기반의 iterated rounding 알고리즘을 소개한다.

# Problem Definition

가중치 있는 무향 그래프 $G = (V, E, c)$와 짝수 크기의 정점 부분집합 $T \subseteq V$, 그리고 일부 정점에 대한 차수 제한 $b_v (v \in W)$ 가 주어지는 상황을 고려하자. 여기에서, $W$는 차수 제한이 있는 정점의 집합이며, $b_v \equiv 1 \pmod 2 \iff v \in T$ 가 성립하도록 $b_v$ 값들이 주어진다.

이 때, $T$에 속하는 정점들의 차수가 홀수가 되도록 하며, 차수 제약 조건을 만족하는 최소 비용의 $J \subseteq E$ 를 찾는 문제를 degree-bounded $T$-join 문제로 정의한다.

이 문제에 대해 다음과 같은 LP-relaxation을 고려할 수 있다.

$$ \begin{align*} \text{minimize:} & & \sum_{e \in E} c_e \cdot x_e && \\ \text{subject to:} & & x(\delta(S)) \ge 1 & &\forall S \subseteq V, \lvert S \cap T \rvert \equiv 1\pmod2 \\ & & x(\delta(v)) \le b_v & & v \in W \\ & & x_e \ge 0 & & \forall e \in E \end{align*}$$

여기에서, $\delta(v)$는 $\delta(\{v\})$ notation을 abuse 한 것이고, 편의상 singleton set에 대한 cut set은 이 글 전체에서 $\delta(v)$로 표기한다.

참고로, LP-relaxation이 infeasible 한 경우, 당연히 원래의 문제도 solution이 없기 때문에, LP가 feasible 한 경우만 고려할 것이다.

# Algorithm

이 문제에 대해 필자가 찾은 iterated rounding algorithm은 다음과 같다.

* $J := \emptyset$
* $E \neq \emptyset$ 인 동안:
  * $(V, E, b, T)$에 대해 LP-relaxation을 해결해 basic optimal solution $x \in \mathbb{R}^E$ 를 얻는다.
  * $E := E(x)$ // $x$의 support 상의 간선들만 남긴다. ($x$값이 0이 아닌 간선)
  * 만약 $E \neq \emptyset$ 이라면:
    * $x_{uv} = 1$인 간선 $uv$를 찾는다. $\cdots (\star)$ 
    * $J := J \cup \{uv\}$
    * $T := T \Delta\{u, v\}$
    * $u \in W$ 라면 $b_u := b_u - 1$
    * $v \in W$ 라면 $b_v := b_v - 1$
    * $E := E \setminus uv$
* return $J$

만약 $(\star)$이 항상 가능하다고 가정한다면, 이 알고리즘은 반복적으로 feasible 한 instance에 대해 LP를 해결하게 되며, 결론적으로 optimal 한 degree-bounded $T$-join을 반환함을 쉽게 확인할 수 있다. 하지만, $(\star)$이 가능하다는 것이 가장 비자명한 부분이고, 이 글에서는 $(\star)$ 이 항상 가능함을 확인해볼 것이다.

# Correctness Overview

Degree-bounded $T$-join의 basic feasible solution $x$를 생각하자. 편의상 $x_e = 0$ 인 간선은 없다고 가정하자. (i.e. $E := E(x)$)
그리고 귀류법으로 모든 간선에 대해 $0 < x_e < 1$을 가정하자. 이 상황에서 모순을 보이는 것이 목표이다.

**Definition.** 정점 $v \in W$가 $x(\delta(v)) = b_v$를 만족하면 tight 하다고 하자. 비슷하게 $T$-odd cut $S$가 $\delta(S) = 1$ 을 만족하면 tight 하다고 정의하자.

**Lemma.** $S, S'$ 가 intersecting 하는 두 tight odd cut이라면, $S \cup S'$ 의 subset 중 non-intersect 하는 서로 다른 두 tight odd cut $W_1, W_2$가 존재해 $\chi_{\delta(S)} + \chi_{\delta(S')} = \chi_{\delta(W_1)} + \chi_{\delta(W_2)}$ 를 만족한다.

참고로, 여기에서 $\chi_S$ 는 $S$의 incidence vector를 의미한다.

*Proof Sketch.* $\lvert S\cap S'\rvert$ 의 기우성을 기준으로 나누어 생각해보면,
* 홀수인 경우: $W_1 := S \cup S'$, $W_2 := S \cap S'$ 이라 하면, 이 둘은 모두 $T$-odd 이고, $2 = x(\delta(S)) + x(\delta(S')) \ge x(\delta(S \cup S')) + x(\delta(S \cap S')) = x(\delta(W_1)) + x(\delta(W_2)) $ 가 되므로, $W_1, W_2$ 모두 tight 하다.
* 짝수인 경우: $W_1 := S \setminus S'$, $W_2 := S' \setminus S$ 로 잡으면, 비슷하게 $W_1, W_2$ 모두 tight 하다.

두 경우 모두 $\delta(W_1) \uplus \delta(W_2) \subseteq \delta(S) \uplus \delta(S')$ 인데, 모든 간선에 대해 $x$ 값이 0 초과이므로, equality가 성립하게 된다. $\square$

이제, 이를 기반으로 tight odd cut 들의 laminar collection을 생각해볼 것이다.

**Definition.** $\newcommand{\T}{\mathcal{T}} \T$ 를 모든 tight set for LP solution $x$의 collection이라고 정의하자. 즉, $\T := \{S \subseteq V : x(\delta(S)) = 1, S \text{ is an odd cut w.r.t. } T\}$. 그리고 $\newcommand{\span}{\mathrm{span}} \span(\T) := \span\{\chi_{\delta(S)} : S \in \T\}$ .

**Lemma.** 어떤 laminar collection of sets $\newcommand{\L}{\mathcal{L}} \L$ 이 존재해 $\span(\T) \subseteq \span(\L)$, each $S \in \L$ is a tight odd cut,  $\chi_{\delta(S)}$ are linearly independent.

*Proof Sketch.* 
- Maximal 한 $\L$을 잡는다.
- 귀류법으로 $\chi_{\delta(S)} \in \span(\T) - \span(\L)$ 인 $T$-odd cut $S$가 존재한다고 가정하고, 이 중 자신과 intersect 하는 $\L$ 의 원소의 수가 최소인 것을 잡는다. (0개보다는 많다. 왜냐하면 $\L$이 maximal.)
- $S$와 intersect 하는 $\L$의 원소 $X$를 잡는다.
- $W_1, W_2$ 를 위의 Lemma를 사용해서 잡는다. (s.t. $\chi_{\delta(S)} + \chi_{\delta(X)} = \chi_{\delta(W_1)} + \chi_{\delta(W_2)}$.)
- $\chi_S \not \in \span(\L)$ 이므로 $\chi_{W_1}, \chi_{W_2}$중 $\span(\L)$ 에 속하지 않는 것이 존재한다.
- $W_1, W_2$ 중 어느게 속하지 않는지 및 $\lvert S \cap X\rvert $ 의 기우성을 기준으로 4가지 경우가 있다. 각각의 경우에 대해  모순을 유도하여 증명을 완료할 수 있다. $\square$ 

이를 기반으로, tight vertex 까지 동시에 고려하면, 다음의 Lemma를 증명할 수 있다. 공간의 제약으로 증명은 생략한다.

**Lemma.** $Z \subseteq W$ 와 collection $\L$ of subsets of vertices가 존재해 다음의 조건을 모두 만족한다.
1. $\forall S \in \L$, $S$ is odd and is tight.
2. $\forall v \in Z$, $v$ is tight.
3. $\chi_{\delta(S)}$ and $\chi_{\delta(v)}$ are linearly independent.
4. $\lvert \L \rvert + \lvert Z \rvert = \lvert E \rvert$.
5. $\L$ is a laminar collection.

Laminar collection $\L$의 원소들이 이루는 트리 구조를 생각해볼 수 있을 것이나, 이 구조로 부터 바로 모순을 이끌어 내기에는, 정점 집합들 간의 원래 그래프 상에서의 특별한 관계가 없다. 따라서, 이들의 induced subgraph를 자연스레 생각해볼 수 있고, 이로 부터 다음을 증명할 수 있다.

**Corollary.** $Z \subseteq W$ 와 collection $\L$ of subsets of vertices가 존재해 다음을 만족한다.

1. $\forall S \in \L$, $S$ is odd and is tight.
2. $\forall v \in Z$, $v$ is tight.
3. $\chi_{\delta(S)}$ and $\chi_{\delta(v)}$ are linearly independent.
4. $\lvert \L \rvert + \lvert Z \rvert = \lvert E \rvert$.
5. $\L$ is a laminar collection.
6. For each $S \in L$, $G[S]$ is connected.

*Proof Sketch.* 위 Lemma로 부터 얻은 $Z, \L$ 을 가지고 시작할 것이다. 만약, 모든 $S \in \L$ 에 대해 $G[S]$ 가 connected 라면, 아무것도 할 필요가 없다. 그렇지 않은 경우를 가정하자.

$S \in \L$ 을 하나 고정하고 생각해봤을 때, induced subgraph $G[S]$ 가 여러개의 component를 갖고 있다면, 그 중 하나는 $T$-odd 할 것이다. 그러한 컴포넌트 $C$ 를 택하자. $\delta(C) \subseteq \delta(S)$라면 이러한 $C$ 또한 tight 함을 알 수 있다.

그리고 이를 확인하기 위해 귀류법으로 $\delta(C) \not\subseteq \delta(S)$ 를 가정해보면, $e \in \delta(C) - \delta(S)$ 를 잡을 수 있을 것인데, $C$의 정의에 의해 $C$에 속하지 않는 $e$ 의 끝점은 $S$의 원소일 수 없다. 이는 $e \in \delta(S)$를 의미해 모순이다. 따라서, $C$ 또한 tight 하다.

이제 우리는 $\L \cup \{C\} \setminus \{S\}$ 가 여전히 위 Lemma의 조건 다섯가지를 만족함을 확인할 것이다. 먼저, $C$는 $T$-odd 하며, tight하다. 또한, $G := G(x)$ 를 가정했으므로, $\delta(C) = \delta(S)$ 이므로, 3을 만족함에 있어서도 문제가 없다. 마지막으로, $\L \cup \{C\} \setminus \{S\} $가 laminar 임도 간단히 확인할 수 있다. $\square$ 

그런데, $\L' := \L \cup \{\{v\} : v \in Z\}$ 또한 여전히 laminar collection 임이 명백하고, 1개의 정점으로 이루어진 induced subgraph는 connected이다. 따라서, 다음이 성립한다.

**Corollary.** Collection $\L'$ of subsets of vertices가 존재해 다음을 만족한다.

1. $\forall S \in \L$, $S$ represents a tight constraint
2. $\chi_{\delta(S)}$ are linearly independent.
3. $\lvert \L \rvert  = \lvert E \rvert$.
4. $\L$ is a laminar collection.
5. For each $S \in L$, $G[S]$ is connected.

이제 우리는 induced subgraph가 connected인 vertex subset 들의 tree (정확히는 forest) 구조를 생각할 수 있다. $S \in \L$과 그의 children $C_1, C_2, \cdots, C_k$ 을 생각하자. 각 child를 contract 하는 접근을 생각해볼 수 있을 것이다.


**Lemma.** $S \in \L$과 그의 children $C_1, C_2, \cdots, C_k$ 을 생각하자. $G[S]$에서 children 들을 각각 정점 하나로 contract 시킨 그래프 $G_S'$ 을 생각하자. 이 때, $\lvert E_S'\rvert  \ge k $ 이다.

*Proof.*

1. 만약, $S \setminus C_1 \setminus C_2 \setminus \cdots \setminus C_k$ 가 emptyset이 아니라면, $G_S'$이 connected 이므로, $\lvert E_S'\rvert \ge \lvert V_S'\rvert -1 \ge (k+1)-1 = k$ 가 되어 성립한다.
2. 이제, $S = C_1 \cup \cdots \cup C_k$ 라 가정하자. 만약, $G_S'$ 에 cycle이 있다면, $\lvert E_S'\rvert \ge \lvert V_S'\rvert = k$ 가 되어 성립한다.
3. 이제, $G_S'$ 이 트리라고 가정하자. $S \in L$, $C_1, \cdots, C_k \in L$ 이므로, $\lvert S \cap T \rvert$ 와 $\lvert C_i \cap T\rvert (1 \le i \le k)$ 모두 홀수가 된다. 따라서, 기우성에 의해 $k$ 또한 홀수가 된다.

   $k = \sum_i x(\delta(C_i)) = x(\delta(S)) + 2x(E_S') = 1 + 2x(E_S')$ 이 된다. 따라서, $x(E_S') = {k-1 \over 2} $ 가 된다.

   그런데, 트리는 이분 그래프이므로, $G_S'$ 을 두개의 정점집합 $A, B$ 로 나눌 수 있다. 모든 $E_S'$ 의 간선은 $A$와 $B$ 사이에 놓인다. 고로, $x(\delta(A, B)) = x(E_S') = {k-1 \over 2}$ 가 된다.

   일반성을 잃지 않고, $\sum_{i \in A} x(\delta(C_i)) = \lvert A\rvert \le\lvert B\rvert =  \sum_{i \in B} x(\delta(C_i))$ 라 하자. 그러면, $k$ 가 홀수이므로, $\sum_{i \in A} x(\delta(C_i)) = \lvert A \rvert \le \frac{k-1}{2}$ 가 된다.

   그런데, $x(\delta(A, B)) = \sum_{i \in A} x(\delta(C_i))$ 이므로, 이는 모든 $i \in A$ 에 대해 $\delta(C_i, V-S) =\emptyset$  임을 의미한다. (아래와 같은 상황). 그런데, 이는 $\chi_{\delta(S)} = \sum_{i \in B} \chi_{\delta(C_i)} - \sum_{i \in A} \chi_{\delta(C_i)}$ 임으로, $\L$ 에 속하는 원소들의 linear independence에 모순이다. 즉, 3과 같은 케이스는 애초에 불가능하다. $\square$

참고로, $\L'$에 대해서도 위 lemma가 비슷하게 성립함은 쉽게 확인할 수 있다.

이제, 위 Lemma를 기반으로 모순을 유도하기 위해 각 edge에 토큰을 하나씩 주는 상황을 생각할 것이다. 그리고 이를 다음의 두 단계로 나누어 $\L'$의 원소에 re-distribute 할 것이다.

1. Root가 아닌 vertex subset에 다음의 방법을 통해 edge에 있는 토큰을 옮겨, 이들 각 vertex subset이 1개씩 토큰을 갖고 있도록 하기.
   - $S \in \L'$와 children $C_1, C_2, \cdots, C_l \in \L'$가 있다고 하자.
   - 위의 Lemma에 의해 $S$가 induced subgraph에 자신을 포함하는 vertex subset in $\L'$ 중 가장 작은 것이 되는 edge가 $l$개 이상 존재.
   - 이 edge 들 중 아무거나 $k$개 택하고, 이들 위의 토큰을 $C_1, C_2, \cdots, C_l$에 하나씩 부여.
   - 이렇게 하면, root vertex subset 들을 $R_1, R_2, \cdots, R_k$라 할 때, $G[R_1], G[R_2], \cdots, G[R_k]$ 에 속하는 edge들만 사용하여 non-root vertex subset들에 전부 토큰을 하나씩 옮길 수 있다.
2. Root인 vertex subset에 다음의 방법을 통해 edge에 있는 토큰을 옮겨, 이들 각 vertex subset이 1개씩 토큰을 갖고 있도록 하기.
   - Root vertex subset 들을 $R_1, R_2, \cdots, R_k$라 하자.
   - 각 $R_i$ 들을 정점 하나로 contract 하여 얻어지는 그래프 $G'$을 생각해보면, $x(\delta(R_i)) = 1$ 이므로, $\lvert \delta(R_i) \rvert \ge 2$ 가 된다.
   - 따라서, $G'$에 간선이 $k$개 이상 있으므로, 이 간선들의 토큰을 옮겨 각 $R_i$ 마다 토큰이 하나씩 놓여있게 분배할 수 있다.

이렇게 토큰을 분배하고 나면, $\lvert L' \rvert (= \lvert E \rvert)$개의 정점 집합들에 하나씩 토큰을 줬으므로, 남는 토큰이 있어서는 안된다.

하지만,
- $\lvert \delta(R_i) \rvert \ge 3$ 인 $R_i$ 가 존재한다면, 2에서 남는 토큰이 발생해 모순이 된다.
- 아니라면, $G'$의 각 컴포넌트는 $R_i$ 들 중 짝수개를 포함하는 cycle이 될 것이다. 하나의 컴포넌트를 잡고, WLOG, $R_1, R_2, \cdots, R_l$이 하나의 cycle을 이룬다고 하자. 그러면, $\sum_{i = 1}^l (-1)^i \chi_{R_i} = 0$이 되며, 이는 $\L'$ 의 원소들 간의 linear independence에 모순이 된다.


따라서, $x_e = 1$인 간선 $e$가 반드시 존재한다. $\square$

# Conclusion

이로써, degree bounded $T$-join 문제를 다항 시간에 해결하는 방법을 알아보았고, basic solution에 대해 constraint 들이 가지는 laminar structure를 이용하여 정확성을 증명하였다. 이 알고리즘을 응용하면 필자가 과거에 소개한 다음 글의 문제 또한 근사하는데에 도움을 받을 수 있다. [링크](https://infossm.github.io/blog/2024/01/24/dbtsp/)