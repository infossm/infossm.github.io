---
layout: post
title: "Perfect Graph"
author: ainta
date: 2021-06-20
tags: [graph theory]
---

## Perfect Graph

Graph Theory에서 유명한 그래프 종류 중에 하나로 완전그래프(Complete Graph)를 꼽을 수 있을 것이다. 완전그래프는 모든 vertex 쌍 사이에 edge가 하나 존재하는 그래프이다.
그렇다면 Complete와 비슷하게 또 완전하다, 완벽하다라는 뜻을 가진 perfect 라는 형용사가 붙는 Perfect Graph는 무슨 뜻일까?

안타깝게도 Perfect Graph는 Complete Graph처럼 직관적인 종류의 그래프는 아니다. Perfect Graph에 대해 정의하기 전에 다음을 정의하자.

- vertex set이 $V$, edge set이 $E$인 그래프를 $G = (V,E)$ 라고 표기한다.

- $V' \subset V$ 에 대해, $E' = \\{ (u,v) \in E \mid u,v \in V' \\}$ 라 하면 $G' = (V',E')$는 $G$의 $V'$에 대한 **induced subgraph**라고 한다.

- $G = (V,E)$ 에 대해 $G' = (V,E')$가 모든 서로 다른 $u,v \in V$ 에 대해 $(u,v) \in E \Leftrightarrow (u,v) \notin E'$ 를 만족할 때 $G'$를 $G$의 complement graph 또는 **complement**라고 하고, $\bar{G}$로 나타낸다.

- $G = (V,E)$에 대해, $f : V \rightarrow C$ 가 $(u, v) \in E \Rightarrow f(u) \neq f(v)$ 를 만족할 때 $f$를 **coloring**이라 하고, $f$의 치역의 크기를 coloring에 사용된 색상의 수 라고 한다.

- $G$의 **chromatic number $\chi(G)$**는 $G$의 coloring 중 가장 적은 색상이 사용된 coloring의 색상의 수로 정의한다. 즉, $G$를 coloring하는데 필요한 최소 색의 수이다.

- **$\omega(G)$** 는 $G$가 포함하는 가장 큰 완전그래프의 크기이다. 즉, $G$의 maximum complete induced subgraph의 크기이다.

$G$의 maximum complete induced subgraph $H$에 대해, $G$의 어떤 coloring에서도 $H$의 원소들은 서로 다른 색으로 칠해지므로 $\chi(G) \ge \omega(G)$가 성립한다.

- 그래프 $G$의 모든 induced subgraph $H$ 에 대해 $\chi(G) = \omega(G)$ 가 성립할 때, $G$를 **Perfect Graph**라 한다.

### $\chi(G) = \omega(G)$ 가 성립하는 그래프들

$G$가 이분그래프(bipartite graph)일 때, $\chi(G) = \omega(G)$ 임은 자명하다. edge가 0개일 때는 두 값이 모두 1, 그렇지 않을 때는 2이다.

$G$가 이분그래프의 complement일 때, $H = \bar{G}$라 하면 $\omega(G)$는 $H$의 maximum independent set의 크기이다. 한편, $\chi(G)$는 $H$를 동일한 색의 vertex 사이에는 edge가 존재하도록 칠할 때 필요한 최소 색의 수 이므로, $\left\vert V \right\vert$ 에서 최대 매칭을 뺀 값이다. Kőnig's theorem 에 의해 $\omega(G) = \chi(G)$이 성립한다.

그래프 $G$에 대해, **Line graph** $L(G)$는 다음과 같이 정의된다.

$L(G)$의 각 vertex는 $G$의 각 edge를 나타낸다.

$L(G)$의 두 vertex 사이에 edge가 있을 조건은 $G$에서 대응되는 두 edge가 끝점을 공유하는 것이다.

이분 그래프 $G$ 의 Line graph $L(G)$에 대해, $\omega(L(G))$ 는 $G$의 vertex들의 최대 degree이다. $\chi(L(G))$ 는 $G$의 edge coloring (vertex를 공유하는 edge끼리는 색이 서로 다르게 edge를 칠하는 방법)의 최소 색상 수인데, 이는 $G$의 max degree와 동일함이 알려져 있다. 따라서, $\chi(L(G)) = \omega(L(G))$

이분 그래프 $G$ 의 Line graph $L(G)$의 complement $H$에 대해, $\omega(H)$ 는 $G$에서 정점을 공유하지 않는 edge들의 집합의 최대 크기이므로, $G$의 최대 매칭이다. $\chi(H)$ 는 $G$를 vertex를 공유하지 않는 edge끼리는 색이 서로 다르게 edge를 칠할 때 최소 색상 수인데, 이는 $G$의 minimum vertex cover의 크기와 같음을 알 수 있다. 따라서 Kőnig's theorem에 의해 $\chi(H) = \omega(H)$

**Chordal Graph** $G = (V,E)$ 와 이에 대한 perfect elimination ordering $v_1, v_2, ..., v_n$ 을 생각하자. (Chordal Graph 또는 Perfect Elimination Ordering의 정의를 모르는 경우, [필자의 글](http://www.secmem.org/blog/2019/03/10/Finding-perfect-elimination-ordering-in-choral-graphs/)에 설명이 되어 있다)

$G$ 를 coloring할 때, $v_n$ 부터 $v_1$까지 순서대로 색칠한다고 하자.
$v_i$를 색칠할 때, $i < j$를 만족하는 $v_i$와 adjacent한 $v_j$ 들의 집합 $S$ 에 대해 perfect elimination ordering의 정의에 의해 $S \cup \\{ v_i \\}$ 는 완전그래프를 이루므로, $\left\vert S \right\vert \le \omega(G) - 1$ 이다. 따라서, 각 $v_i$를 칠할 때 칠할 수 없는 색상이 최대 $\omega(G) - 1$개 이므로 $\omega(G)$ 가지의 색만으로 coloring이 가능하다. 즉, $\chi(G) \le \omega(G)$. 그런데 모든 그래프에 대해 $\chi(G) \ge \omega(G)$ 이므로 $\chi(G) = \omega(G)$.

위에서 언급한 그래프들은 각각에 induced subgraph를 취했을 때 그 특성을 잃지 않는다. 즉, bipartite graph의 induced subgraph는 bipartite subgraph이고, bipartite graph의 line graph의 induced subgraph도 마찬가지로 bipartite graph의 line graph의 induced subgraph이다. 따라서, 위에 언급한 그래프 종류들인 bipartite graph와 그 complement, bipartite graph의 line graph와 그 complement, chordal graph는 모두 perfect graph이다. chordal graph에는 tree, forest, interval graph 등이 포함되므로 이들도 모두 perfect graph에 속한다.

### Perfect Graph에서 풀리는 문제들

그래프 $G$가 주어졌을 때 $\chi(G)$를 구하는 문제는 graph coloring problem이라는 NP-Hard 문제이다. $\omega(G)$를 구하는 문제 역시 maximum clique problem이라는 NP-Hard 문제이다. maximum independent set을 구하는 문제는 complement를 취하면 maximum clique problem과 동일해지므로 이 역시 NP-Hard 문제이다. 그러나 perfect graph에서는 위 3가지 문제가 모두 다항시간 내에 해결 가능함이 알려져 있다. 실제 알고리즘은 너무 어려우니 여기서는 다루지 않는다.

### Weak Perfect Graph Theorem

위에서 perfect graph의 예시 몇가지를 살펴보았을 떄, bipartite graph와 그 complement, 그리고 bipartite graph의 line graph와 그 complement가 모두 perfect graph였다. 그러면 여기에서 다음과 같은 가설 한 가지를 생각해 볼 수 있다.

**Hypothesis 1 (Weak Perfect Graph Theorem). $G$가 perfect graph이면 $\bar{G}$ 역시 perfect graph이다.**

위 가설은 Berge가 perfect graph에 대해 세운 두 가설 중 하나로, 실제로 증명되어 weak perfect graph theorem이라 부른다. 이에 대한 증명은 어렵지 않기 때문에 여기에서 다루고자 한다.

$G$ 의 maximum independent set의 크기를 **$\alpha(G)$** 라 하자. 임의의 그래프 $G = (V,E)$에서 $\chi(G)\alpha(G) \ge \left\vert V \right\vert$ 가 성립한다. coloring에서 하나의 색으로 칠해진 vertex는 independent set을 이루므로 이는 자명하다.

**Theorem 1. $G$가 perfect graph인 것과 $G$의 모든 induced subgraph $H$가 $\left\vert V(H) \right\vert \le \alpha(H)\omega(H)$를 만족하는 것은 동치이다.**

$\alpha(\bar{G})=\omega(G)$,$\omega(\bar{G})=\alpha(G)$ 이므로 Theorem 1이 성립하면 weak perfect graph theorem이 증명된다. 이제 Theorem 1을 증명하자.

_Proof_

먼저, $\Rightarrow$ 방향은 간단하다. $G$가 perfect graph이면 $G$의 induced subgraph $H$에 대해 $\alpha(H)\omega(H) = \alpha(H)\chi(H) \ge \left\vert V(H) \right\vert$이다.

$\Leftarrow$ 방향을 보이자. $G$의 모든 induced subgraph $H$가 $\left\vert V(H) \right\vert \le \alpha(H)\omega(H)$를 만족하며 $G \neq H$ 인 경우 $H$가 perfect graph라고 가정했을 때, $G$가 perfect graph임을 보이면 충분하다. (vertex 개수에 대한 수학적 귀납법)

$G$가 perfect graph가 아니라고 가정하자. 즉, $\chi(G) > \omega(G)$ 이다.
$\chi(G)$ 를 간단히 $\chi$, $\omega(G)$를 간단히 $\omega$라고 쓰자.

$G$의 임의의 independent set $A$에 대해 $H = G \setminus A$ 는 perfect graph이므로 $\omega(G \setminus A)$ 색으로 coloring이 가능하다. 따라서, $G$는 $\omega(G \setminus A) + 1$ 개 색으로 coloring이 가능하다. $\omega < \chi \le \omega(G \setminus A) + 1$ 이므로 모든 independent set $A$에 대해 $\omega = \omega(G \setminus A)$.
이에 따라, $G$의 임의의 independent set $A$에 대해 $G \setminus A$에 $\omega$ 크기의 clique $K_A$ 가 존재한다.

$\alpha$ 크기의 independent set $A_0$을 하나 잡고, $A_0$ = $\\{ a_1, a_1, ..., a_{\alpha} \\}$ 라 하자.
임의의 $1 \le i \le \alpha$에 대해, $G - a_i$ 는 $\omega$ 색으로 coloring 가능하다. 이 때 각 색상이 칠해진 vertex들의 집합 $\omega$ 개를 $A_{1}^i, ... A_{\omega}^i$ 라 하자.
$\mathcal{A} = \\{ A_0,A_{1}^1, ...., A_{\omega}^{\alpha} \\}$ 는 $\alpha\omega + 1$ 개의 independent set이다.

**Claim 1. 크기 $\omega$인 임의의 clique $K$에 대해, $K$와 disjoint한 $A \in \mathcal{A}$는 정확히 하나 존재한다.**

_Proof_

$A_0$의 원소 $a_i$를 하나 잡자.
$a_i \notin K$ 인 경우, $K$는 $G - a_i$ 의 크기 $\omega$인 clique이다. 따라서, $A_{1}^i, ... A_{\omega}^i$ 와 정확히 1개의 vertex씩을 공유한다.
한편, $a_i \in K$이면 $K - a_i$는 $G - a_i$ 의 크기 $\omega - 1$인 clique이므로 $A_{1}^i, ... A_{\omega}^i$ 중 하나와 disjoint하고 나머지와는 1개의 vertex를 공유한다.
$K$와 $A_0$이 1개의 원소를 공유하는 경우 $K$와 disjoint한 $A_{i}^{j}$ 가 하나 존재하고, $K$와 $A_0$이 disjoint하면 $K$와 disjoint한 $A_{i}^{j}$가 존재하지 않으므로
$K$와 disjoint한 $A \in \mathcal{A}$는 정확히 하나 존재한다. $\blacksquare$

$\mathcal{A}$ 의 원소를 $\\{A_0, A_1, .., A_{\alpha\omega} \\}$ 로 리넘버링하자. 그리고 각 $A_i$에 대해 $A_i$와 disjoint한 크기 $\omega$ 인 clique을 잡아 $K_i$라 하자.

$A_i$와 $K_j$는 $i=j$일 때 disjoint하고 $i \neq j$ 일때 하나의 vertex를 공유한다.

$\left\vert V \right\vert = n$ 라 하고 vertex에 $v_1, v_2, .., v_n$의 번호를 붙였을 때,
행렬 $X, Y$를 다음과 같이 정의하자.

- $X, Y$ 는 크기 $(\alpha\omega + 1) \times n$의 행렬
- $X_{ij}$는 $v_j \in A_i$이면 1, $v_j \notin A_i$ 이면 0
- $Y_{ij}$는 $v_j \in K_i$이면 1, $v_j \notin K_i$ 이면 0

$(\alpha\omega + 1) \times (\alpha\omega + 1)$ 행렬 $XY^{T}$은

$$
XY^{T} =
\begin{bmatrix}
0 & 1 & \cdots & 1 & 1 \\
1 & 0 & \cdots & 1 & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
1 & 1 & \cdots & 0 & 1 \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}
$$

를 만족하고, 이 행렬의 rank는 $\alpha\omega + 1$이다.

한편, $X$ 와 $Y$ 의 rank는 $n$을 넘지 않으므로, $XY^{T}$ 의 rank 역시 $n$ 이하이다.
그런데 가정에 의해 $n = \left\vert V(G) \right\vert \le \alpha(G)\omega(G) = \alpha\omega$ 이므로 이는 모순이다.
따라서, $G$가 perfect graph가 아니라는 가정이 잘못되었고, 수학적 귀납법에 의해 Theorem 1은 참이다.$\blacksquare$

Theorem 1이 참이므로 $G$가 perfect graph이면 $G$의 모든 induced subgraph $H$에 대해 $\left\vert V(H) \right\vert \le \alpha(H)\omega(H) \Rightarrow
\left\vert V(\bar{H}) \right\vert \le \omega(\bar{H})\alpha(\bar{H})$ 이므로 $\bar{G}$ 역시 perfect graph가 된다. 따라서 Weak Perfect Graph Theorem이 증명되었다.

### Strong Perfect Graph Theorem

앞서 Berge가 perfect graph에 대해 제시한 가설 weak perfect graph theorem이 실제로 성립함을 확인했다.
Berge는 1961년에 또 하나의 더 강력한 가설을 제시했는데, 이는 2002년 Maria Chudnovsky 등 4명이 증명함으로써 Strong Perfect Graph Theorem이 되었다.

Strong Perfect Graph Theorem에 대해 다루기 전에, 몇 가지 정의를 살펴보자.

- vertex의 개수가 3 이상이고 모든 vertex의 degree가 2인 connected graph를 cycle이라 하고, 정점이 $n$개인 경우 $C_n$이라 한다.

- $k \ge 2$에 대해, $C_{2k+1}$을 odd hole, $C_{2k}$를 even hole 이라 한다. ($C_3$은 hole 이 아님에 주의)

odd hole은 $\chi(C_{2k+1}) = 3$, $\omega(C_{2k+1}) = 2$ 을 만족한다.
odd hole의 complement는 $\chi(\bar{C_{2k+1}}) = k+1$, $\omega(\bar{C_{2k+1}}) = k$를 만족한다.

따라서, 그래프 $G$가 induced subgraph로 odd hole 또는 그 complement를 가지는 경우 perfect graph가 될 수 없다.

**Hypothesis 2 (Strong Perfect Graph Theorem). $G$가 perfect graph인 것과 $G$가 induced subgraph로 odd hole 또는 그 complement를 가지지 않는 것은 동치이다.**

strong perfect graph theorem 은 2002년 증명되었는데, 과정이 길고 많은 casework을 포함한다. 이를 수박 겉핥기식으로 살펴보자.

induced subgraph로 odd hole 또는 그 complement를 가지지 않는 그래프를 **Berge Graph** 라 하자.
그래프가 perfect이면 Berge인 것은 우리가 이미 앞서 보였다. 이제 Berge이면 perfect임을 보이면 충분하다.

그래프 $G$가 perfect graph가 아니며 자기 자신이 아닌 모든 induced subgraph가 perfect일 때 $G$를 **minimal imperfect graph**라 한다.

Berge graph이면 perfect하다는 결론은 아래와 같은 큰 두 개의 과정으로 이루어진다.

- Claim 1. 모든 Berge graph는 basic case에 포함되거나, 어떤 특정한 decomposition을 만족한다. basic case의 그래프들은 perfect graph이다.

- Claim 2. $G$가 minimal imperfect graph이면 $G$는 Claim 1의 decomposition을 만족하지 않는다.

여기서 basic case란 우리가 앞서 말한 bipartite graph와 그 complement, bipartite graph의 line graph와 그 complement, 그리고 double split graph이다.
double split graph가 무엇인지는 넘어가도록 하자. 정의를 살펴보면 double split graph가 Berge graph이며 perfect graph라는 사실은 쉽게 알 수 있다.
Berge Graph이고 basic case에 포함되지 않으면 decomposition을 만족하고, minimal imperfect graph는 decomposition을 만족할 수 없다는 증명은 매우 복잡하다. 이에 대한 내용은 [Strong Perfect Graph Theorem 논문](https://arxiv.org/pdf/math/0212070.pdf) 에 상세히 나와있다.

### Perfect Graph 판별

주어진 그래프가 [Chordal Graph](http://www.secmem.org/blog/2019/03/10/Finding-perfect-elimination-ordering-in-choral-graphs/)인지 아닌지는 놀랍게도 Linear Time에 확인할 수 있었다. 평면그래프 여부도 linear time에 판별 가능하다(Planarity testing). 그렇다면 perfect graph인지도 빠른 시간 내에 확인할 수 있을까?

아쉽게도, 주어진 그래프가 perfect graph인지 현재까지 매우 빠르게 판별하는 알고리즘은 존재하지 않는다. $G$가 perfect graph인 것은 $G$와 $\bar{G}$가 odd hole을 포함하지 않는 것과 동치인데, odd hole이 있는지 detect하는 알고리즘은 다항시간에 해결됨이 최근에 밝혀졌고, [시간복잡도는 $O(N^9)$ 로 굉장히 느리다](https://arxiv.org/pdf/1903.00208.pdf).

## 맺음말

이상으로 Perfect Graph의 정의와 Weak/Strong Perfect Graph Theorem, Perfect Graph에서 다항시간 내에 해결 가능한 문제에 대해 간략히 알아보았다.

이름 있는 그래프의 카테고리 중 상당수가 Perfect Graph에 포함되며, 또한 각각이 Perfect Graph라는 것 자체가 유명한 정리와 동치인 경우도 꽤 존재한다.

앞서서 이분그래프의 complement가 perfect graph인 것은 Kőnig's theorem와 동치였고, partially ordered set에서 Dilworth's theorem은 Comparability graph가 perfect라는 것과 동치이다. Permutation graph가 perfect인 것 역시 어떤 수열의 Longest Increasing Subequence의 길이는 decreasing subsequence 들로 분할하는 최소 개수와 같다는 성질과 같은 의미이다. 이에서 알 수 있듯 perfect graph에 포함되는 그래프 종류들을 알면 여러 그래프 관련 이론에 대해 연관지어 생각할 수 있다.

비록 perfect graph에 적용될 수 있는 알고리즘이나 perfect graph 판별법 등은 너무 복잡하여 사용하기 쉽지 않지만, 이 글을 통해 perfect graph라는 그래프 군에 대해 알아봄으로써 여러 종류의 그래프에 대해 좀더 알아볼 수 있는 기회가 되었으면 좋겠다.
