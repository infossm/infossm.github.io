---
layout: post
title: "Nearly optimal communication complexity of Bipartite Maximum Matching"
author: TAMREF
date: 2022-10-20
tags: [linear-algebra, graph-theory]
---

## Introduction

Bipartite Maximum Matching (BMM)은 Bipartite graph $(V = X \sqcup Y, E \subseteq X \times Y)$에 대해 크기 $F$ 이상인 Matching이 존재하는지를 묻는 decision problem, 혹은 Maximum Matching의 크기를 계산하는 문제를 이야기합니다. 굉장히 오래 전부터 연구되어 온 문제이고, 정점이 $n$개, 간선이 $m$개인 BMM은 $m^{1 + o(1)}$ 시간 안에 해결할 수 있다는 것이 알려졌습니다.

원본 문제의 시간 복잡도가 subpolynomial 수준에서 사실상 optimum을 달성했기 때문에, 비전형적인 세팅에서 BMM을 해결하는 기법들이 등장하기 시작했습니다. 그 예시가 multi-party communication으로, 두 사람 Alice와 Bob이 간선 집합의 일부만 나누어 가지고 있는 경우입니다. 이 때 통신에 사용하는 비트의 크기를 어떻게 최소화할 수 있을까요?

구체적으로 전체 그래프 $(V = X \sqcup Y, E)$에 대해 Alice는 $(V, E_{A} \subseteq E)$을, Bob은 $(V, E_{B})$를 가지고 있고, $E_{A} \cup E_{B} = E$라고 할 때, 오늘 리뷰할 논문인 "Nearly Optimal Communication and Query Complexity of Bipartite Matching" (Joakim Blikstad et al.) 은 $O(n \log^{2} n)$ 비트의 정보 교환이면 충분하다는 것을 증명하였습니다.

## Folklore

낯선 세팅의 문제를 마주했으니, 이 문제의 자명한 lower bound와 upper bound를 생각해보겠습니다. 우선 BMM의 요구 조건을 좀 더 명확히 따져볼 필요가 있습니다.

만약 Alice나 Bob 모두가 Maximum matching을 구성하는 간선들을 알아내야 한다고 가정하면, matching의 크기가 최대 $n$까지 커질 수 있고, $E_{A} = E, E_{B} = \emptyset$인 경우를 생각하면 Bob에게 maximum matching을 전달하기 위해 자명한 $\Omega(n \log n)$ lower bound가 발생하게 됩니다.

놀라운 것은 Maximum Matching의 크기만 구하기 위해서도 $\Omega(n \log n)$ 비트는 주고받아야 한다는 점입니다.

**Theorem (Hanjal et al, 1988)** 주어진 그래프 $G$와 $s, t \in V(G)$에 대해, $s-t$ 사이에 경로가 존재하는지 판별하는 모든 deterministic communication protocol은 $\Omega(n \log n)$비트를 필요로 한다.

이 때, $s-t$ connectivity의 lower bound로부터 bipartite perfect matching의 lower bound 또한 이끌어낼 수 있음이 알려져 있습니다.


이번에 리뷰할 알고리즘은 실제로 $O(n \log^{2} n)$ upper bound로 매칭을 construct할 수 있기 때문에 두 lower-bound에 대해서 모두 이론적 optimuim의 $O(\log n)$배를 달성합니다. 이것이 얼마나 어려운지를 생각해보기 위해 upper bound를 생각해 봅시다.

그래프를 전부 전달하는 방법은 간선을 전부 전달하는 $O(m \log n)$ 짜리 방법, boolean 인접행렬을 전달하는 $O(n^2)$ 방법이 있습니다. 두 방법이 우선 가장 자명한 upper bound가 됩니다. 조금 덜 자명한 upper bound를 생각해보면 $O(n^{1.5} \log n)$ 짜리 알고리즘을 생각할 수 있습니다.

Hopcroft-Karp bipartite matching algorithm을 일반적인 그래프 (sequential setting)에서 푸는 방법은 우선 residual graph에서 BFS를 통해 level graph를 만들고, 이후 DFS를 통해 blocking flow를 찾는 것입니다. 그런데 각 BFS와 DFS는 모두 $O(n \log n)$ 비트만 이용하여 communication setting에서도 풀 수 있고, 그 과정에서 발생하는 level graph, blocking flow 등도 $O(n \log n)$ 메모리 안에 담아 공유할 수 있습니다. Hopcroft-Karp에서는 이러한 iteration이 $O(\sqrt{n})$ 번만에 끝난다는 것이 알려져 있으므로 communication complexity 또한 $O(n^{1.5} \log n)$이 됩니다.

하지만 이것에 비해 더 나은 upper bound를 찾기는 어려운데요, 특히 Hopcroft-Karp는 $O(m \cdot n^{0.5})$ sequential 알고리즘을 $O(n \cdot n^{0.5} \log n)$ 알고리즘으로 쉽게 변환한 예시이지만, 일반적으로 $O(m^{1 + o(1)})$ sequential 알고리즘을 $O(n^{1 + o(1)})$ communication 알고리즘으로 바꾸는 것은 어렵다고 생각하고 있습니다.

## Dual LP approach

Konig's theorem에 의해, Bipartite graph에서 maximum matching의 크기와 minimum vertex cover의 크기는 같고, minimum vertex cover가 주어지면 maximum matching 또한 construct할 수 있습니다. 따라서, 크기 $F$인 vertex cover를 찾을 수 있다면 maximum matching의 크기는 $F+ 1$ 미만이고, 그렇지 않다면 크기 $F$ 이상의 매칭이 존재한다고 주장할 수 있습니다.

Minimum vertex cover를 LP로 formulate합시다. 크기 $F$ 이하인 vertex cover는 아래 세 조건식을 만족하는 feasible solution입니다. LP는 totally unimodal하기 때문에 정수 solution이 존재하고, non-integer solution이 존재하더라도 weight를 재분배하여 integral solution으로 바꿀 수 있습니다.

- $\sum_{v \in V(G)} x_{v} \le F$
- $e = uv \in E(G)$에 대해, $x_{u} + x_{v} \ge 1$.
- $0 \le x_{v} \le 1$.

위 LP에 의해 정의되는 $2n$차원상의 polytope을 $P^{F}$로 정의합시다. 편의상 bipartite graph의 한 쪽 정점 크기를 $n$으로 정의했기 때문입니다.

 위에서 한 integral solution의 이야기를 polytope의 언어로 번역하면, $P^{F} \neq \emptyset$이면 $P^{F}$는 격자점을 포함하고, $P^{F}$의 임의의 한 점에서 격자점을 찾을 수 있습니다.

이 때 약간 조건을 relax한 $P^{F + 1 / 3}$을 생각해도, $P^{F + 1/3} \neq \emptyset$이면 $P^{F}$의 격자점이 존재하고, $P^{F + 1 / 3}$의 임의의 한 점에서 $P^{F}$의 격자점을 찾을 수 있습니다. 따라서 앞으로는 $P^{F + 1/3}$을 해결하는 데 집중합니다. 굳이 relax를 해주는 이유는, 아래 Lemma에서처럼 polytope의 부피를 어느 정도 확보하기 위해서입니다.

**Lemma.** $P^{F+ 1/3} \neq \emptyset$이라고 하면, $\mathrm{vol}(P^{F+ 1/3}) \ge (20n)^{-2n}$.

*Proof.* 길이가 $\frac{1}{20n}$인 구간 $I_{1}, \cdots, I_{2n}$에 대해 $I_{1} \times \cdots \times I_{2n} \subseteq P^{F+1/3}$을 보이면 충분합니다. $P^{F + 1/3} \neq \emptyset$이므로, 어떤 격자점 $x \in P^{F}$가 존재할 것입니다. 이 때 $x_{i}$를 어느 정도 perturbation해줘도 $P^{F + 1/3}$안에 있는 것을 보이면 됩니다.

- $x_{i} = 0$이라면, $y_{i}$를 $[\frac{1}{20n}, \frac{1}{10n}]$ 사이의 임의의 점으로 둡니다.
- $x_{i} = 1$이라면, $y_{i}$를 $[1 - \frac{1}{20n}, 1]$ 사이의 임의의 점으로 둡니다.

이렇게 두면 새로 만든 solution $y$는 3개의 조건을 모두 만족하므로, 우리가 원하는 hypercube를 만들 수 있습니다. $\square$

위와 같이 부피가 $\exp(-o(n \log n))$인 polytope의 한 점을 찾는 문제로 바꾸었는데요, 이 lemma의 의미는 차후 설명하도록 하겠습니다.

### Cutting-plane method

LP $P$의 feasible solution을 찾는 문제는 sequential setting에서 다항 시간에 풀 수 있습니다. 이는 ellipsoid method라는 유명한 알고리즘이 존재하는데요, 이를 아우르는 LP 해결의 방법론들을 cutting-plane method라고 합니다. 일반적으로 다음과 같은 framework에서 동작합니다.

- $P \subseteq Q$를 만족하는 커다란 $Q$를 찾는다.
- 어떤 $x \in Q$를 잡고, $x \in P$인지를 판정한다.
- $x \in P$라면 종료하고, 아니면 $x$에 의해 violate되는 조건들로 구성된 hyperplane을 생각한다. 가령 $c^{T} x \le d$라는 조건이 만족되지 않는다면, hyperplane $H_{c, d} := \lbrace z : c^{T} z \le c^{T}x \rbrace$ 를 생각한다. 이들을 "cutting plane"이라고 부른다.
- $Q$와 cutting plane들의 교집합을 포함하는 polytope $Q'$을 찾는다. 이 때, $\mathrm{vol}(Q')$가 $\mathrm{vol}(Q)$에 비해 유의미하게 작다.

$Q$를 무엇으로 잡는지, 또 $Q$의 한 점 $x$를 어떻게 잡는지에 따라 복잡도가 달라집니다. 잘 알려진 polynomial time algorithm인 ellipsoid의 경우, $Q$를 ellipsoid, $x$를 $Q$의 중심이 되도록 유지합니다. 이제 이 framework를 우리가 생각한 vertex cover LP로 옮겨봅시다.

- 맨 처음 $Q$는 거대한 simplex $\lbrace x \in [0, 1]^{2n} : \sum_{v} x_{v} \le F + 1 / 3\rbrace$에서 출발한다.
- Alice가 아무 (하지만 $Q$에 의해 유일하게 결정되는) $z \in Q$를 잡고, 모든 $uv \in E_{A}$에 대해 $z_{u} + z_{v} \ge 1$를 만족하는지 판정한다.
  - 만족한다면 OK를 Bob에게 보낸다.
  - 그렇지 않다면 edge $uv$와 그에 의해 정의되는 cutting-plane $H_{uv} : \{y : y_{u} + y_{v} \le z_{u} + z_{v}\}$를 Bob에게 넘겨준다. 이 과정에서 $O(\log n)$ 비트를 사용한다.
- 이를 받은 Bob은 $Q$를 $Q \cap H_{uv}$로 업데이트한 뒤 같은 과정을 반복한다.
- 만약 Alice, Bob이 모두 OK를 output한다면 $z$가 원하는 답이 된다.
- 충분히 큰 $K$에 대해 $K$번의 iteration 이후에도 답을 찾을 수 없다면, $P^{F + 1/3} = \emptyset$.

**Lemma.** $N$차원 Ellipsoid method에서, ellipsoid의 크기는 $O(e^{-\frac{1}{2(N+1)}})$배 감소한다. $\square$

따라서 ellipsoid method를 사용하고 $K = \Theta(n^{2} \log n)$ 으로 두면, $K$번 iteration을 한 뒤 ellipsoid의 부피가 $\exp(-\Theta(n \log n))$이 되므로 앞선 lemma에 의해 $P^{F+ 1/3} = \emptyset$일 수밖에 없습니다. 하지만 이 과정에서 $K \log n = \Theta(n^{2} \log^{2} n)$ 크기의 communication cost가 발생하므로, $K$를 훨씬 더 줄일 필요가 있습니다.

### Center-of-gravity method

사실 convex polytope $Q$에 대해, $x \in Q$의 한 점을 잡는다고 생각했을 때 center-of-gravity $g(Q) = \frac{\int_{Q} zdz}{\int_{Q} dz}$는 가장 대표적인 점 중 하나입니다. 따라서 위의 framework에서 아무 $x \in Q$를 잡는 과정을 $x = g(Q)$로 대체할 수 있습니다. 이 방법은 최강의 장점과 최악의 단점을 각각 보유하고 있습니다.

**Theorem (Rademacher, 2007).** Computing center-of-gravity is $\#P$-hard. Furthermore, centroid is even hard to approximate.

**Theorem (Grunbaum, 1960).** Polytope $P$와 $z = g(P)$, 그리고 $z$를 지나는 아무 hyperplane $H$에 대해, $\mathrm{vol}(P \cap H) / \mathrm{vol}(P) \in [e^{-1}, 1 - e^{-1}]$.

우선 center-of-gravity를 계산하기 힘들다는 단점은 배제할 수 있는 것이, 우리는 Alice와 Bob이 무슨 문제를 푸는지에 관심이 없고 다만 그들의 통신 비용을 최적화하고 있기 때문입니다. 장점에 대해 부연하자면, 우리가 "cutting plane" $H_{uv}$을 선정하고 이를 따라 $Q'= Q \cap H_{uv}$로 두면, $Q'$은 $Q$에 비해 $(1 - e^{-1})$배 부피가 작습니다.

따라서 $K = \Theta(n \log n)$으로 둘 수 있고, communication cost는 $O(n \log^{2} n)$이 됩니다. 따라서 우리는 $O(n \log^{2} n)$에 BMM의 decision 버전을 해결할 수 있습니다. $\square$

### Removing Parametric search

여기서 남는 의문은, decision problem에 대한 이진 탐색으로 maximum matching의 크기 계산 / 값 찾기를 하면 communication cost가 $O(n \log^{3} n)$이 될 것 같다는 것인데, 사실 이는 LP를 좀 더 개선하면 이진 탐색 없이도 해결할 수 있습니다.

초기값으로 $F = 2n$을 설정하고 decision problem을 해결하면, 결과로 $z \in [0, 1]^{2n}$을 얻게 됩니다. 이 $z$는 크기 $F' = \lfloor z_{1} + \cdots + z_{2n} \rfloor$의 vertex cover로 변환할 수 있으므로, $F'$을 minimum vertex cover의 후보로 생각하고 여기에 constraint $\lbrace x : x_{1} + \cdots + x_{n} \le F' - 1 + \frac{1}{3}\rbrace$를 추가합니다.

이와 같이 계속 LP를 진행하다가 infeasible하다는 결론이 난다면 마지막으로 minimum vertex cover의 후보였던 값으로부터 vertex cover를 복구할 수 있고, 언급하였듯 이로부터 maximum matching도 복구할 수 있습니다. LP를 멈추지 않고 계속 진행하면 할수록 cutting plane에 의해 부피가 constant factor로 감소하고 있으므로, communication cost는 역시 $O(n \log^{2} n)$입니다. $\square$

## Query model

이 framework는 상당히 범용성이 높아, communication model이외의 query model 등에도 이용할 수 있습니다. Query model이란 내가 그래프의 정점만 알았지 간선에 대해 아는 것이 없고, 특정한 쿼리를 통해서만 간선에 access할 수 있는 문제 세팅을 말합니다. 이 논문에서 개선이 있었던 몇 가지 query model은 다음과 같습니다.

- OR query: 집합 $S \subseteq X \times Y$에 대해, $\lvert S \cap E \rvert \ge 1$인가?
- XOR query: $\lvert S \cap E \rvert \mod 2$는?
- Independent Set query: $U \subseteq X, W \subseteq Y$에 대해 $G[U \cup W]$에 간선이 있는가?

**Theorem.** Bipartite Maximum Matching은 $O(n \log^{2} n)$번의 OR-query로 해결할 수 있다.

*Proof.* 결국 $O(\log n)$번의 쿼리로 $z_{u} + z_{v} < 1$을 만족하는 violating edge $(u, v)$를 찾을 수 있으면, 모든 세팅이 communication 세팅과 동일하여 $O(n \log^{2} n)$번에 문제를 해결할 수 있게 됩니다.

$S = \lbrace (u, v) : z_{u} + z_{v} < 1\rbrace$로 두고 한 번 쿼리를 날리고, 답이 yes가 돌아오면 이분탐색으로 $(u, v)$를 하나 찾아주면 됩니다.

**Theorem. (Beniamini, 2021)**  Bipartite Perfect Matching의 존재 여부를 deterministic하게 판별하기 위해서는 $\Omega(n^2)$번의 XOR query가 필요하다. $\square$

하지만 XOR-query를 $O(\log n)$번 사용하여, 높은 확률로 맞는 violating edge oracle을 만들 수 있습니다.

**Theorem.** $O(n \log^{2} n)$번의 XOR-query를 이용하여, 높은 확률로 BMM을 계산할 수 있다.

*Proof.* $S = \lbrace (u, v) : z_{u} + z_{v} < 1\rbrace$의 랜덤한 subset $T \subseteq S$에 대해 XOR-query를 날립시다. $S$에 violating edge가 있지만 $T$에 대한 답이 $0$으로 나올 확률은 $1 / 2$ 이하이므로, $\Theta(\log n)$번 이를 수행하면 답이 $1$일 확률이 $1 - n^{-\Omega(1)}$ 이상이 됩니다. 이후에는 $O(\log n)$번의 이분탐색으로 동일하게 violating edge를 찾을 수 있으므로 OR query model과 동일하게 해결할 수 있습니다. 답을 틀리게 되는 케이스는 violating edge가 있지만 존재하지 않는다고 판정하고 terminate하므로, 답을 틀릴 확률은 역시 그대로 $1 - n^{-\Omega(1)}$이 됩니다. $\square$

**Theorem.** $O(n \log^{2} n)$번의 IS-query를 이용하여 BMM을 계산할 수 있다.

*Proof.* LP를 풀면서 얻은 $z \in Q$가 integral point인 경우, OR-query에서 정의한 $S = \lbrace (u, v) \in E(G): z_{u} + z_{v} < 1\rbrace$에 대해 $S = (z^{-1}(0) \cap X) \times (z^{-1}(0) \cap Y)$가 됩니다. 따라서 IS query와 OR query가 완전히 동등하게 됩니다.

그런데 크기 $f$인 fractional vertex cover는 크기 $\lfloor f \rfloor$의 integral vertex cover로 변환할 수 있으므로, 조건을 만족하는 integral vertex cover를 계산한 뒤 OR query와 동일하게 처리해주면 됩니다.

## Conclusion

BMM처럼 sequential model에서 이미 closed problem인 경우에도 계산 모델을 query model, communication model 등으로 바꿨을 때 상당히 흥미로운 lower bound, upper bound를 얻을 수 있고, query model 또한 여러 가지로 두고 문제를 해결할 수 있다는 것을 알 수 있습니다.

이 경우, 우리는 communication model에 대해서 특수한 vertex-cover LP를 해결하였습니다. 이 경우 violating constraint에 대해 각 식을 이루는 variable의 개수가 아주 적어 communication cost를 획기적으로 줄일 수 있었는데 일반적인 경우에 대해 통하는 접근은 아닙니다. 하지만 최근, communication setting에서 polynomially bounded communication cost로 일반적인 LP를 해결할 수 있는 것이 밝혀졌습니다.

**Theorem (Vempala et al, 2020).** LP $P$의 차원이 $d$이고 모든 coefficient가 최대 $L$비트를 차지할 때, communication cost $\Theta(d^{2} L)$에 $P$의 feasible point를 찾을 수 있다.

이후에도 다양한 communication / query setting에서 해결할 수 있는 유명한 문제들에 대해 리뷰해보도록 하겠습니다.

## Reference

- Blikstad, Joakim, et al. "Nearly Optimal Communication and Query Complexity of Bipartite Matching." arXiv preprint arXiv:2208.02526 (2022).
  - 저자의 [TCS+ Talk 영상](https://www.youtube.com/watch?v=t-BkFOigYKY) 을 여기에서 찾아볼 수 있습니다.
- Hajnal, András, Wolfgang Maass, and György Turán. "On the communication complexity of graph properties." Proceedings of the twentieth annual ACM symposium on Theory of computing. 1988.
- Rademacher, Luis A. "Approximating the centroid is hard." Proceedings of the twenty-third annual symposium on Computational geometry. 2007.
-  Branko Gr¨unbaum. Partitions of mass-distributions and of convex bodies by hyperplanes. Pacific Journal of Mathematics, 10:1257–1261, 1960.
-  Vempala, Santosh S., Ruosong Wang, and David P. Woodruff. "The communication complexity of optimization." Proceedings of the Fourteenth Annual ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2020.