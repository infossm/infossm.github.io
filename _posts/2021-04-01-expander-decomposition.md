
---
layout: post
title:  "Expander Decomposition and Pruning: Faster, Stronger, and Simpler"
date:   2021-04-01 01:00:00
author: koosaga
tags: [algorithm, graph-theory, data-structure, flow]
---

# Expander Decomposition and Pruning: Faster, Stronger, and Simpler

알고리즘에서 **분할 정복** 은 큰 문제를 부분 문제로 나누는 과정을 뜻한다. 이 때 부분 문제들이 가져야 하는 특징은, 원래 문제보다 쉬워야 하고, 부분 문제를 합칠 수 있어야 한다. 예를 들어서, Heavy Light Decomposition은 트리에서 큰 문제를 부분 문제로 나누는 과정에서 자주 등장한다. 각 문제가 쉽고 (직선), 합치는 것이 가능 (Light edge를 통해서 묶음) 하기 때문이다.

트리의 경우에는 HLD 외에도 다양한 분할 정복 기법이 존재하지만, 그래프를 분할 정복하는 것은 쉽지 않다. 일반적으로, Treewidth와 같이 그래프의 특수한 성질을 요구하는 경우가 많다. 이번에 다룰 **Expander Decomposition** 은, 그래프의 특수한 성질과 무관하게 그래프를 Expander로 분할한다. Expander는 좋은 성질들을 가지고 있어서 문제가 *쉬워지고*, Expander 간의 연결이 적기 때문에 합치는 것이 가능하다.

Expander Decomposition을 설명하기 전에 몇 가지 정의를 짚고 넘어간다.

* 우리는 이 글에서 가중치 없는 그래프를 다룬다.
* $G = (V, E)$ 의 *컷* 은 비지 않고, 전체 집합이 아닌 정점 부분집합을 뜻한다. $S \neq \emptyset, S \subsetneq V$
* 정점 부분집합 $S \subseteq V$ 에 대해 $G[S]$ 는 $S$ 의 induced subgraph이다.
* 두 서로소 집합 $A, B \subseteq V$ 에 대해서, $E(A, B)$ 는 $A$ 와 $B$ 에 각 끝점이 있는 간선의 집합을 뜻한다.
* 컷의 **값** 은 $\delta(S) = |E(S, V - S)|$ 이다.
* 컷의 **볼륨 (volume)** 은 $vol_G(S) = \sum_{v \in S} deg(v)$ 이다.
* 컷의 **전도율 (conductance)** 는 $\Phi_G(S) = \frac{\delta(S)}{min(vol_G(S), vol_G(V - S))}$ 이다. 편의를 위해 $vol_G(S) \le vol_G(V - S)$ 라고 가정하라.
* 그래프의 **전도율 (conductance)** 는 $\Phi_G = min_{S\text{ 는 } G \text{ 의 컷 }} \Phi_G(S)$ 이다. 컷이 존재하지 않는다면 ($|V| = 1$) $\Phi_G = 1$ 이다.
* $\Phi_G \geq \phi$ 이면 $G$ 는 $\phi$-expander 이다.

Expander에서는 왜 문제가 쉬울까? Expander는 잘 연결된 그래프라는 성질을 활용한다. 

* **Decremental Spanning Forest** 문제를 생각해 보자. 트리 간선을 지웠을 때 Spanning Forest를 관리하려면, 모든 back-edge들 중 두 컴포넌트를 연결하는 간선을 찾아야 한다. 만약, 그래프가 Expander라면, 임의의 Back edge가 두 컴포넌트를 연결할 확률이 $\phi$ 이기 때문에, 랜덤 샘플링 만으로 찾을 수 있다. 
* **Global Minimum Cut** 문제를 생각해 보자. Dense한 그래프에서 컷은 높은 확률로 상당히 치우쳐 있다. 즉, 작은 쪽의 정점이 큰 쪽의 정점보다 월등히 적을 가능성이 높다. 이러한 직관에서 출발하면, Global Min-Cut을 Expander에서 구할 때 작은 쪽의 크기가 상당히 작다고 가정할 수 있다. 작은 쪽의 크기가 $polylog(n)$ 정도임이 보장될 때 컷을 구할 수 있다면 전체 문제가 해결된다. Deterministic Mincut in Almost-Linear Time (Jason Li, STOC'21) 의 큰 방향성이 이러한 식이다.

두 예시 모두 $\phi = 1 / \log^{O(1)}m $ 정도의 크기를 가정함을 참고하면 좋다.

## 정의와 사전 연구

### Expander Decomposition

그래프 $G$ 와 두 실수 인자 $\epsilon, \phi$ 가 주어졌을 때, $(\epsilon, \phi)$-expander decomposition은 정점 집합 $V$ 의 분할 $V_1 \cup V_2 \cup \ldots \cup V_k$ 로,

* $G[V_i]$ 가 $\phi$-expander이며 (쉬움)
* 서로 다른 집합을 잇는 간선의 개수가 최대 $\epsilon |E|$ 개임 (합칠 수 있음)

아주 좋은 퀄리티의 Expander Decomposition을 구하는 알고리즘이 있다. 심지어, 매우 쉬운 알고리즘이다.

**Lemma.** 임의의 $\phi \in (0, 1)$ 에 대해서 $(\phi \log m, \phi)$ expander가 존재한다.

**Proof by recursive algorithm.** $G$ 가 Expander라면, 바로 종료한다. 그렇지 않다면, conductance가 최소인 컷을 찾은 후 $G[S], G[V - S]$ 를 재귀적으로 분할한다. $T(m) = T(a) + T(m - a) + \phi min(a, m - a)$ 이기 때문에, $T(m) \le \phi m \log m$ 이고, \epsilon \le \phi \log m$이다.

**Remark.** 고정된 $\phi$ 에 대해 더 작은 $\epsilon$ bound를 얻을 수 없음을 증명할 수 있다. 

이 알고리즘의 문제는 시간 복잡도다. conductance가 최소인 컷을 찾는 것이 NP-hard라서, 다항 시간 알고리즘이 될 수 없다. 최소 전도율 컷을 이제부터 sparsest cut이라고 부르자.

Expander Decomposition을 다항 시간에 해결한 첫 알고리즘 (Kannan, Vempala, Veta, FOCS'00) 은 sparsest cut의 approximation을 찾은 후 이에 따라서 decomposition하는 방법을 사용한다. 이 근사 알고리즘은 $(\phi \log^{1.5} m, \phi)$-expander decomposition을 구하며, 다항 시간에 작동하지만, 여전히 느리다.

이 알고리즘을 near-linear time으로 최적화하려면 두 가지 전제조건이 필요하다. 먼저 Sparse cut을 Near-linear time에 근사할 수 있어야 하며, 또한 Sparse cut으로 분할한 두 컷이 균형잡혀 있어야 한다. Sparse cut을 선형 시간에 찾는다 하더라도, 두 컷이 균형잡히지 않았다면 $O(n^2)$ 시간을 피할 수가 없다. 이 점을 개선한 것이 Spielman-Tang의 유명한 Spectral Sparsification 알고리즘 (STOC'04) 로, 두 가지 성질을 활용한다.

* Spectral method를 사용하여 Sparse cut을 근사한다.
* 위 알고리즘은, 찾은 컷이 균형잡히지 않았다면, 컷의 큰 부분이 *어떠한* Expander subgraph에 포함된다는 것을 보장한다. 고로, 큰 부분은 *일종의* expander가 되어서, 더 재귀적으로 분할하지 않는다. 이 때 큰 부분을 포함하는 그 Expander가 무엇인지는 찾을 수 없다.

큰 부분에 속하는 컷은 좋은 성질을 만족하지만, expander는 아니다. 고로 엄밀히 말하면 ST04는 expander decomposition을 구하지 못한다. 하지만 대부분의 경우. ST04로 만든 Expander decomposition은 충분히 강력하다. 이 알고리즘은 $(\sqrt \phi \log^{O(1)}m, \phi)$-expander 들을 찾는다.

마지막으로, 이 논문은, $(\phi \log^3 m, \phi)$-expander decomposition을 $O(m \log^4 m / \phi)$ 시간에 구하는 Randomized algorithm을 소개한다. 

### Expander Pruning

위에서 언급한 Decremental Spanning Forest 예시를 다시 살펴보자.

* **쉬운 부분문제: Expander**. Expander에서는 DSF를 위에서 말한 방법으로 관리할 수 있다.
* **합병가능한 부분문제: Cross-partition edges**. 서로 다른 파티션을 잇는 간선의 개수가 polylogarithmic하게만 많기 때문에, Naive하게 생각해 줄 수 있다. (자세한 설명은 생략한다.)

이렇게 볼 경우, DSF 문제는 Expander Decomposition을 사용하여 간단하게 해결할 수 있어 보인다. 하지만, 간선을 지울 경우 conductance가 변하기 때문에, Expander decomposition이 깨지게 된다.

이 문제를 해결하기 위해 Expander pruning을 도입한다. 간략하게 말해서, Expander에서 간선을 지웠을 때, 해당 간선 근처에 있는 정점 몇개를 같이 지워나감으로써, 원래 Expander를 (남은 Expander) + (지운 간선) 으로 분리하는 것이다. 다른 말로, 각 Expander에 대해서, **pruned set** $P \subseteq V$ 를 관리하여, $G[V - P]$ 가 Expander고, $P$ 가 적당히 작게 유지되는 것이 목표이다. 만약 이러한 pruned set을 관리할 수 있다면, $G[V - P]$ 는 Expander에서 DSF를 관리하는 요령으로 관리하고, $P$ 는 Incremental하게 관리한 후, 둘의 합집합을 취하는 식으로 문제를 해결할 수 있다.

이 논문의 알고리즘은 Expander Pruning을 구하는 데도 사용될 수 있다. 정확히 어떠한 성능의 Pruning을 구하는 지 아래 엄밀히 정의한다.

**Theorem.** $G$ 를 간선이 $m$ 개인 $\phi$-expander 라고 하자. 최대 $q \le \phi m / 10$ 개의 간선 삭제 쿼리를 처리할 수 있는 결정론적 알고리즘이 존재한다. $i$ 번째 삭제 이후, 알고리즘은 다음 조건을 만족하는 집합 $P_i$ 를 관리한다.

* $P_0 = \emptyset, P_i \subseteq P_{i+1}$
* $vol(P_i) \le \frac{8i}{\phi}$ and $|\delta(P_i)| \le 4i$
* $G_i[V - P_i]$ 가 $\phi / 6$ expander이다. 이 때 $G_i$ 는 $i$ 개의 간선을 지운 그래프를 뜻한다.

업데이트에 사용되는 총 시간은 $O(q \log m / \phi^2)$ 이다.

## 알고리즘

이 논문은 Sparse cut을 근사할 때, 플로우 알고리즘에 기반한 방법을 사용한다. 정확히는, Goldberg와 Tarjan이 제안한 *Push-relabel flow algorithm* 을 사용한다. 이산적인 알고리즘에 기반해 있어서 상대적으로 간단한 편이고, 실용적 가치도 높다. 

SW19 역시 Sparse cut을 근사하는 것은 ST04와 동일하지만, 만약 cut이 균형잡히지 않았을 경우 SW19는 큰 쪽의 컷이 *nearly $\phi$-expander* 임을 보장해 준다. 이후, *nearly $\phi$-expander* 에서 정점을 몇 개 제거하여 이를 진짜 expander로 만들어준다. 큰 쪽의 컷이 expander임이 보장되니까, 재귀적으로 내려갈 필요가 없고, 고로 시간 복잡도가 보장된다. ST04와 다르게, 실제 Expander Decomposition을 찾는다는 것이 주목할 점이다.

알고리즘은 크게 두 부분 문제를 사용하는데, 바로 **Cut-Matching Game** 과 **Trimming** 이다.

**Cut-Matching Game** 은 $(G, \phi)$ 가 주어졌을 때 $\Phi_G \geq \phi$ 임을 확인하거나 전도율이 $O(\phi \log^2 m)$ 임이 보장되는 sparse cut을 반환한다. 이 때 컷은 $min(vol(A), vol(V - A)) \geq \Omega(m / \log^2 m)$ 을 만족하거나, 큰 쪽의 컷이 *nearly $\phi$-expander* 임을 보장한다. 시간 복잡도는 $O((m \log m) / \phi)$ 이다.

**Trimming** 은 nearly-$\phi$ expander $A$ 를 받아서, $A^\prime \subseteq A$ 인 subgraph를 반환한다. 이 때 $G[A^\prime]$ 은 $\phi/6$ expander이다. $A^\prime$ 이 갑자기 너무 작아져서, 원래 매우 작은 쪽이었던 컷이 역전되어 매우 큰 컷이 되는 일은 일어나지 않고, 고로 near-linear time이 보장된다. 시간 복잡도는 $O(|E(A, \overline{A})| \log m / \phi^2) \le O((m \log m) / \phi)$ 이다.

이 두 부분문제가 풀린다고 가정하면 **Expander Decomposition 알고리즘** 을 설명할 준비가 끝났다. 그래프 $(G, \phi)$ 에 대해 Expander Decomposition은 다음과 같이 작동하는 재귀적 알고리즘이다.

* 먼저 $\Phi_G \geq 6 \phi$ 인지를 **Cut-Matching Game** 으로 확인한다. 만약 이것이 성립한다면 끝이다.
* 그렇지 않다면, **Cut-Matching Game** 은 전도율이 $O(\phi \log^2 m)$ 인 sparse cut을 반환한다. 이 컷이 Balanced되어 있다면, 재귀적으로 두 조각에 대해서 해결하면 된다.
* 그렇지 않다면, 큰 쪽의 subgraph $A$ 는 nearly-$6\phi$ expander이다. **Trimming** 으로 $\phi$-expander $A^\prime$ 으로 변환한다.
* $V - A^\prime$ 의 크기가 충분히 작으니, 해당 조각에 대해서만 재귀적으로 해결한다.

이제 알고리즘이 올바름을 증명한다.

* **Correctness.** 알고리즘은 항상 $\phi$-expander, $6\phi$-expander 만을 만든다.
* **Time.** 만약 **Cut-Matching**의 sparse cut이 balanced되어 있다면, 재귀의 깊이가 최대 $O(\log^3 m)$ 이다. 그렇지 않은 경우, Trimming은 그래프의 크기를 $3/4$ 비율로 줄임이 보장된다. 고로 재귀의 깊이가 여전히 보장된다. 한편, 각 깊이에 대해서 고려되는 간선의 집합은 disjoint하니, 각 깊이에 대한 계산량 합이 $O(m \log m / \phi)$ 이고, 고로 시간 복잡도는 $O(m \log^4 m / \phi)$ 이다.
* **Edge Count.** Sparse cut에 대한 $O(\log^2 m)$ approximation을 사용하며, Trimming 역시 이 bound를 깨지 않는다. 고로 $O(\phi \log m \log^2 m) = O(\phi \log^3 m)$ 이다. 

이제 각 부분문제를 해결하는 방법에 대해 다룬다.

### Cut-Matching Game

Cut-Matching Game은 sparsest cut에 대한 근사 알고리즘 전략 중 하나이다. 이 전략을 사용하면, $O(\log^2 n)$-factor에 sparsest cut을 근사하는 near-linear time 알고리즘을 유도할 수 있다. 이 논문에서 새롭게 소개된 테크닉은 아니다. 이 글에서는 high-level idea만 소개할 것이며, 실제 Cut-Matching Game의 정의를 변형하여 설명한다.

그래프 $G$ 에 대해 **Cut** 플레이어와 **Matching** 플레이어가 있다. **Cut** 플레이어 $C$ 는 sparsest cut이 존재한다고 믿고, **Matching** 플레이어 $M$은 $G$ 가 expander라고 믿는다. $C$는 정점 집합의 이분할(bisection) 을 주고, $M$ 은 이분할 사이를 잇는 큰 매칭을 찾는다.

만약 $M$ 이 $O(\log^2 n)$ 번의 턴 동안 항상 큰 매칭을 찾을 수 있다면, 우리는 이들의 합집합을 취해서 $G$ 가 Expander임을 증명할 수 있다. 만약 이 중 한 턴에서 $M$ 이 실패한다면, $C$ 는 작은 매칭을 사용하여 sparse cut을 계산한다. 매칭도 Flow이기 때문에, 컷을 찾을 수 있다. 이분할을 찾는 전략은, $M$ 이 제공한 매칭들을 통하여 샘플링하는 spectral method이다. 여기서 큰 집합이 near-$6\phi$ expander를 주기 위해서는 추가적인 변형이 필요한데, 이에 대해서는 생략한다.

### Trimming

Trimming은 Expander decomposition에서의 불균형을 해소하는 단계로, 이 논문의 주요한 기여이다. 여기서부터, $vol(S)$ 는 $S$ 가 무슨 induced subgraph에 속해있는지와 상관 없이, 항상 원래 그래프 $G$ 의 차수 합으로 정의하자.

**Definition (Nearly Expander).** $A \subset V$ 가 $G$ 의 nearly $\phi$-expander 라는 것은, $\forall S \subseteq A, vol(S) \le vol(A) / 2 \rightarrow |E(S, V - S)| \geq \phi vol(S)$ 임을 뜻한다.

여기서, $A$ 가 $\phi$-expander 라는 것은 $|E(S, A - S)| \geq \phi vol(S)$ 임을 기억하자.

Nearly expander는 Expander의 정의를 relax한 것으로, induced subgraph 밖으로 나가는 간선을 추가로 세어주는 차이가 있다.

**Definition (Trimming).** Trimming은 다음 조건을 만족하는 $A^\prime \subseteq A$ 를 찾는다: $\forall S \subseteq A^\prime, vol(S) \le vol(A^\prime) / 2 \rightarrow|E(S, A^\prime - S)| \geq \phi vol(S) / 6$.

가장 깔끔한 상황은, Trimming을 전혀 하지 않아도 $A$ 가 그냥 $\phi/6$-expander 가 되는 상황이다. 이 경우에는 바로 문제가 해결된다. 이렇지 않은 경우를 상상해 보자. $A$ 는 $\phi$-near expander 지만 $\phi/6$ expander가 아니니, nearly expander의 relax가 굉장히 큰 효과를 발휘했음을 뜻한다. 수식으로 적으면, 다음과 같은 $S$ 가 존재한다: $|E(S, V - A)| \geq 5|E(S, A - S)|$.

![flow](http://www.secmem.org/assets/images/toptree/flow.png)

이제 다음과 같은 Flow 인스턴스를 상상해보자. 

* $E(A, V - A)$ 에 있는 각 간선은 source로부터 $2/\phi$ 용량의 간선으로 연결된다.
* 각 정점은 $deg_G(v)$ 용량의 간선으로 sink로 연결된다.
* 각 내부 간선은 $2/\phi$ 용량을 가진다.

이 경우, 공급량의 합은 $\frac{2}{\phi}|E(A, V - A)|$ 이다. 이제 다음을 보인다.

**Claim.** $A$ 가 $\phi/6$ expander가 아니면, 최대 유량이 공급량 합 이하이다.

**Proof.**  $|E(S, A - S)| \le \frac{|E(S, V - A)|}{5}$ 를 만족하는 집합 $S$ 가 존재한다. 이 때

* $S$ 로 들어오는 유량 $\geq  \frac{2}{\phi} |E(S, V - A)| \geq \frac{1}{\phi} |E(S, V - A)| + \frac{5}{\phi} |E(S, A - S)|$
* $S$ 에서 나가는 유량 $\le vol(S) + \frac{2}{\phi} E(S, A- S) \le \frac{1}{\phi} (E(S, V - A) + E(S, A - S)) + \frac{2}{\phi} E(S, A - S)$


고로 $S$ 에서 유량 손실이 존재한다.

이는 위 그래프에서 최대 유량이 전체 공급량 합과 동일하면, $A$ 가 $\phi/6$ expander라는 것을 증명할 수 있다는 것이다. 

그렇지 않다면, 우리는 공급량을 만족하는 데 실패했다. 이 그래프에서 최소 컷을 찾자. 최대 유량은, 최소 컷을 가로지르는 간선의 모든 유량을 충족시킨다. 최소 컷에서, source 방향에 있는 컷을 $A$ 에서 제거하고 다시 최대 유량을 계산해 보자. 최소 컷이 Bottleneck이었으니, 이번에도 최소 컷이 동일한 곳에서 형성되는데, 이 때 컷이 형성된 위치는 $E(A, V - A)$ 간선 집합이 있는 위치이다. 달리 말해, 최대 유량이 전체 공급량 합과 동일해 졌다는 것이다. 우리는 이로써 다음과 같은 결론을 내릴 수 있다:

**Subquadratic Trimming.** Dinic's algorithm과 같은 subquadratic flow 알고리즘을 사용하여 최대 유량을 계산한 후, 소스 방향에 있는 컷을 prune한다.

이제 source-side cut의 크기가 bound되는 것과, 이 과정에서 conductance가 크게 증가하지 않음을 검증해야 한다. 이 둘은 적절한 수식 계산으로 확인할 수 있으며, 둘 모두 위 알고리즘을 깨지 않는 정도의 bound를 줌을 보장한다. 

**Near-linear time Trimming.** 플로우는 Near-linear time에 계산할 수 없으니, 시간 복잡도를 개선하려면 approximate max-flow 기술을 사용해야 한다. 우리가 위에서 구성한 플로우 그래프가 tight하지 않기 때문에 이것이 가능하다. Approximate max-flow 기술을 사용하면 한 번의 Flow를 빠르게 계산할 수 있지만, 최소 컷에서 source 방향에 있는 컷을 한번 제거한다고 바로 $\phi/6$ expander가 나오지 않는다. 고로 이를 여러 번 반복해야 하는데, 다행이도 어느 정도의 반복 이후 수렴함을 보일 수 있다. 여전히 반복을 여러 번 하는 것이 부담스러운데, 이를 위해서는 컷을 제거한 후에 Maximum flow를 처음부터 다시 계산해야 한다. Push-relabel flow를 개선하면, 이것이 가능하다.

**Trimming implies Expander Pruning.** High-level의 아이디어는, 단순히 간선 제거 후 near-linear time trimming을 시행하고, pruning set을 키우는 것이다. 역시 push-relabel flow를 사용하여 어느 정도 효율적으로 구현해야, 원하는 time bound를 얻을 수 있다.

