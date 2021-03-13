---
​---
layout: post
title:  "매트로이드에서의 Submodular Maximization에 대한 Deterministic algorithms"
date:   2021-03-10 09:00:00
author: leejseo
tags: [optimization, algorithm, matroid]
​---

---

이 글에서는 매트로이드 상에서의 submodular maximization과 관련한 여러 연구의 결과들을 소개합니다. 보다 구체적으로는, 매트로이드와 submodular function 등 여러 개념의 정의를 소개하고, 매트로이드 상에서 submodular function을 최적화 하는 것이 실생활의 어떤 문제에 관련이 있는지 먼저 살펴봅니다. 그 후, 이 문제와 관련한 여러 연구 결과들을 살펴봅니다. 특히, 그 중에서도 (이 글에서는) 결정론적인 알고리즘들을 다룹니다.

## 1. 여러 개념 및 정의

먼저, submodular function 및 그에 대한 marginal gain을 정의합니다.

**정의 1.** 유한집합 $\mathcal{N}$에 대해 함수 $f : 2^\mathcal{N} \to \mathbb{R}$ 을

- $\forall X, Y \in \mathcal{N}, f(X) + f(Y) \ge f(X \cup Y) + f(X \cap Y)$ 라면 submodular function이라 부릅니다.
- $\forall X \subseteq N, f(X) \ge 0$  이면 non-negative라 부릅니다.
- $\exists X \subset Y, f(X) > f(Y)$ 이면 non-monotone 이라 부릅니다. 반대로, $X \subseteq Y \implies f(X) \le f(Y)$ 이라면, monotone이라 부릅니다.

**정의 2.** Submodular function에 대해 $X$의 $Y$ 에 대한 marginal gain은 처음과 같이 정의됩니다. 원소가 1개인 집합에 대해서는 표기의 편의를 위해 두 번째와 같은 표기도 사용합니다.

- $f( X \mid Y) = f(X \cup Y) - f(Y)$
- $f(x \mid Y ) = f(\{x\} \mid Y)$

이제, 매트로이드를 정의합시다.

**정의 3.** 유한집합 $\mathcal{N}$과 $\mathcal{I} \subseteq 2^\mathcal{N}$ 에 대해 $(\mathcal{N}, \mathcal{I})$ 가 1, 2를 만족하면 independence system으로, 1, 2, 3 을 만족하면 매트로이드라 부릅니다. 특히, $\mathcal{I}$의 원소를 independent라 부릅니다.

1. $\varnothing \in \mathcal{I}$
2. $A \subseteq B, B \in \mathcal{I} \implies A \in \mathcal{I}$ (hereditary property)
3. $A, B \in \mathcal{I}, |A| < |B|  \implies \exists x \in B-A, A \cup \{x\} \in \mathcal{I}$ (exchange property)

**정의 4.** 매트로이드(혹은 independence system) $(\mathcal{N}, \mathcal{I})$ 와 $X \subseteq \mathcal{N}$ 에 대해 $X$ 의 base는 maximally independent subset으로 정의됩니다.

매트로이드의 자세한 성질들은 이 글에서 다루지 않습니다. 매트로이드의 자세한 성질이 궁금하신 분 께서는 기존 소프트웨어 멤버십 블로그의 다른 글들을 참고하시기 바랍니다. 특히, 매트로이드의 정의와 성질 외에 그에서의 알고리즘이 궁금하신 분들은, 2/3번째 글을 읽어봐도 좋을 것입니다.

- http://www.secmem.org/blog/2019/05/15/introduction-to-matroid/
- http://www.secmem.org/blog/2019/06/17/Matroid-Intersection/
- http://www.secmem.org/blog/2019/11/14/intro-matroid-union/

이제, 여러 개념들을 정의했으니, 이 글에서 다루고자 하는 문제가 무엇인지를 살펴봅시다.

**문제.** 매트로이드 $(\mathcal{N}, \mathcal{I})$ 와 submodular function $f$가 주어졌을 때, $\max \{ f(S) : S \in \mathcal{I}\}$ 를 계산하여라.

## 2. 문제의 응용

이 글에서 다루는 문제를 해결하는 알고리즘을 살펴보기에 앞서, 실제 현실의 어디에서 응용될 수 있는지를 살펴봅시다. 이 부분은 [1] 및 그의 선행 연구에서 제시된 내용입니다.

먼저, 이 문제는 SNS를 모니터링 하는데 있어 응용될 수 있습니다. 각 사용자를 정점으로 하고, 간선 $(u, v)$ 의 가중치 $w(u, v)$ 를 $(u, v)$ 를 통해 전달될 수 있는 최대 정보량으로 하는 그래프로 SNS를 모델링 합시다. 사용자의 성향에 따른 정점집합 $V$ 의 분할 $V_1, V_2, \cdots, V_h$ 를 생각할 수 있습니다. 이 때, SNS를 모니터링 하는데 사용할 정점 집합 $S \subset V$ 를

- 최대한 많은 양의 정보를 모니터링 할 수 있게 하되, (maximize $f(S) = \sum_{(u, v) \in E, u \in S, v \not\in S} w(u, v)$)
- 특정 집단에 편향되지 않은/최대한 다양한 정보를 얻고자 하는 상황 (constraints: $|S \cap V_i| \le k$ for each $i$ and some constant $k$)

을 생각해볼 수 있고, 이 상황이 matroid 상에서의 submodular maximization의 한 예시입니다.

또 다른 예시로는, 여러 상품을 바이럴(입소문) 마케팅 하는 상황이 있습니다. 즉, SNS에서 몇몇 사람들을 seed로 하여 여러 상품을 홍보하려 하는 상황입니다. 앞선 예시와 마찬가지로 SNS 그래프를 생각해볼 수 있는데, 간선에 가중치가 있는 것이 아닌, 각 정점의 가중치 $c(u)$ 가 있다고 합시다. 이 가중치는 사람 $u$ 를 seed로 했을 때 소요되는 비용입니다. 그리고 각 상품 $i$ 에 대해 정점집합 $S$ 를 해당 상품의 seed들의 집합으로 하였을 때의 매출을 나타내는 함수 $S \mapsto f_i(S)$ 가 있고, 하나의 정점은 최대 하나의 상품에 대한 seed만 될 수 있다고 합시다. 이 때,

- 최대 가용 예산 $B$ 와 최대 선택 가능한 seed의 수 $k$ 가 있을 때,
- 매출을 최대화 하는 동시에 예산을 절약하려는 상황 (maximize $\sum_{i } f_i(S_i) + (B - \sum_{i} \sum_{v \in S_i} c(v))$)

을 생각해볼 수 있습니다. 여기에서 $S_i$ 는 상품 $i$ 의 seed 정점들의 집합입니다. $f_i$ 가 submodular 와 같은 적당한 조건이 있으면, 이 상황 또한 본 글에서 다루는 문제의 instance가 됩니다.

## 3. $f$가 non-monotone 할 수 있는 경우

먼저, 보다 일반적인 상황인 (maximize 해야 할 함수 $f$ 가) non-monotone 할 수 있는 상황에 대해 알아봅시다. 현재까지 알려진 가장 높은 approximation ratio는 0.385이나, 이는 랜덤에 의존합니다. 결정론적인 알고리즘 중 가장 높은 approximation ratio는 $1/4$ 입니다. [1]에서는 $O(nr)$ 시간에 $1/4$ approximation ratio를 가지는 알고리즘과 $O((n / \epsilon) \log (r / \epsilon))$ 시간에 작동하면서 $1/4 - \epsilon$ approximation ratio를 가지는 알고리즘들을 소개했습니다. 이 알고리즘들은 간단하면서도 대단히 흥미롭기 때문에, 이 글에서 소개하고자 합니다.

### TwinGreedy

처음으로 소개할 알고리즘은 ([1]에서) TwinGreedy라고 불리는 알고리즘입니다.

**TwinGreedy** $(\mathcal{N}, \mathcal{I}, f)$

- $S_1, S_2 := \emptyset$ 
- repeat:
  - Find $(e, i)$ such that $e \in \mathcal{N} - S_1 - S_2$, $i \in \{1, 2\}$, and $f(e \mid S_i)$ is maximized
  - break if no such $(e, i)$ exists or $f(e \mid S_i) \le 0$
  - $S_i := S_i \cup \{e\}$
- return $\max(f(S_1), f(S_2))$



이 알고리즘은 위에 적힌 바와 같이 두 개의 solution set $S_1, S_2$ 를 유지하면서 동작하는데, 반복적으로 그리디하게 두 집합 중 하나를 업데이트 해나갑니다. 시간 복잡도는 $O(nr)$ 이며, $1/4$ 의 approximation ratio를 가집니다.

### TwinGreedyFast

두 번째로 소개할 알고리즘은 ([1]의) TwinGreedyFast 알고리즘입니다. 이 알고리즘 또한 TwinGreedy와 비슷하게 두 개의 solution set을 유지하면서 작동하는데, 반복적으로 조금씩 감소하는 어떤 *threshold* 를 이용하여 원소를 solution set에 추가할지 여부를 결정합니다.

**TwinGreedyFast** ($\mathcal{N}, \mathcal{I}, f, \epsilon$)

- $S_1, S_2 := \emptyset$, $\tau_{max} = \max\{f(e) : \{e\} \in \mathcal{I}\}$
- for ($\tau := \tau_{max}$; $\tau > \epsilon \tau_{max} / (r(1+\epsilon))$; $\tau := \tau / (1 + \epsilon)$)
  - for $e \in \mathcal{N} - (S_1 \cup S_2)$ 
    - $\Delta_1, \Delta_2 := -\infty$
    - if $S_1 \cup \{e\}$ is independent: $\Delta_1 := f(e \mid S_1)$
    - if $S_2 \cup \{e\}$ is independent: $\Delta_2 := f(e \mid S_2)$
    - $i := \arg\max_j \Delta_j$  (tie-breaking is done arbitrarily)
    - if $\Delta_i \ge \tau$: $S_i := S_i \cup \{e\}$
- return $\max(f(S_1), f(S_2))$



이 알고리즘은 $O((n / \epsilon) \log (r / \epsilon))$ 시간 복잡도를 지니며, approximation ratio는 $1/4 - \epsilon$ 입니다.

## 4. $f$가 monotone한 경우

이 경우에는, [2]에서 0.5008의 approximation ratio를 가지는 알고리즘을 제시했습니다. 1978년에 0.5의 approximation ratio를 가지는 알고리즘이 제시된 이래, 최초로 이를 넘은 (deterministic) 알고리즘 입니다. 이 알고리즘은 Split과 Residual Random Greedy라는 두 개의 작은 알고리즘과, 이들을 사용하는 메인 알고리즘으로 구성됩니다. ($\mathcal{M}$: 매트로이드, $k$: 매트로이드의 rank)

**Split**($f, \mathcal{M}, p$)

* $A_0, B_0 := \emptyset$
* for $i=1$ to $k$
  * $u_i^A := \arg\max_{u \in \mathcal{M} - A_{i-1}-B_{i-1}} f(u \mid A_{i-1})$
  * $u_i^B := \arg\max_{u \in \mathcal{M} - A_{i-1}-B_{i-1}} f(u \mid B_{i-1})$
  * if $p f(u_i^A \mid A_{i-1}) \ge (1-p) f(u_i^B \mid B_{i-1})$
    * $A_i := A_{i-1} + \{u_i^A\}$
  * else
    * $B_i := B_{i-1} + \{u_i^B\}$
* return $(A_k, B_k)$

**RRGreedy**$(f, \mathcal{M})$

- $A_0 := \emptyset$
- for $i = 1$ to $k$
  - $M_i :=$ base of $\mathcal{M} / A_{i-1}$ maximizes $\sum_{u \in M_i} f(u \mid A_{i-1})$
  - $A_i := A_{i-1} + u_i$, $u_i$ is chosen uniformly randomly at $M_i$
- return $A_k$

**Main** $(f, \mathcal{M})$

- $A_1, B_1 := Split(f, \mathcal{M}, p)$ // $p$ is some value, described in the paper
- $A_2 := RRGreedy(f(\cdot \mid A_1), \mathcal{M} / A_1)$
- $B_2 := RRGreedy(f(\cdot \mid B_1), \mathcal{M}/B_1)$
- return $\max(f(A_1 \dot\cup A_2), f(B_1 \dot\cup B_2))$



이 Main 알고리즘은 위에 있는 바와 같이 split algorithm을 통해 union이 base이며 좋은 성질을 가지는 두 disjoint한 set을 얻습니다. 이후, ([2]의 선행 연구에서 제시된) 1/2-approximation ratio를 가지는 RRGreedy 알고리즘을 사용합니다. 이를 통해 RRGreedy 알고리즘보다 더 좋은 $1/2+\epsilon$-approximation을 얻게 되며, $O(nk^2)$ 정도의 연산을 사용하게 됩니다. 이는 다항시간이므로, 적당히 효율적이라고 생각할 수 있습니다.

하지만, RRGreedy에 random이 관여하게 되어, 이 글에서 관심을 가지는 determinstic 알고리즘이 아닙니다. 이를 위해 [2]의 저자들은 RRGreedy를 대체할 수 있도록 derandomized 된 Residual Parallel Greedy 알고리즘을 제시했습니다.

Residual Parallel Greedy 알고리즘은 maximum weight perfect matching 알고리즘을 이용하는데, 이를 위한 방법으로는 MCMF나 Hungarian 알고리즘을 포함한 여러 효율적인 알고리즘이 알려져 있습니다. 작동 과정은 다음과 같습니다.

**RPGreedy**($f, \mathcal{M}, B$)

- $A_0^j := \emptyset, B_0^j := B$ for every $j = 1, \cdots, k$
- for $i = 1$ to $k$
  - for $j = 1$ to $k$
    - $M_i^j$ := base of $M / A_{i-1}^j$ maximizes $\sum_{u \in M_i^j} f(u \mid A_{i-1}^j)$
  - Define a weighted bipartite graph $G_i (V_L, V_R, E, w)$:
    - $V_L := B$
    - $V_R := \{1, 2, \cdots, k\}$
    - for each $u \in M_{i^j}$ and $v \in B$:
      - if $v \in B_{i-1}^j$ and $(A_{i-1}^j + u) \dot\cup (B_{i-1}^j - v)$ is base of $\mathcal{M}$ and $f(u \mid A_{i-1}^j) \ge f(v \mid A_{i-1}^j)$
        - add an edge $e = (v, j)$ 
        - assign $w_e = f(u \mid A_{i-1}^j)$
  - $R_i :=$ maximum weight perfect matching of $G$
  - for $j = 1$ to $k$
    - $e (= (v_i^j, j)) :=$ signle edge of $R_i$ hits $j$
    - $u_i^j (\in M_i^j) := $ the element corresponds to $e$
    - $A_i^j := A_{i-1}^j + u_i^j$
    - $B_i^j := B_{i-1}^j + v_i^j$
- return best one among $A_k^{*}$

**Main** $(f, \mathcal{M})$

- $A_1, B_1 := Split(f, \mathcal{M}, p)$ // $p$ is some value, described in the paper
- $A_2 := RPGreedy(f(\cdot \mid A_1), \mathcal{M} / A_1, B_1)$
- $B_2 := RPGreedy(f(\cdot \mid B_1), \mathcal{M}/B_1, A_1)$
- return $\max(f(A_1 \dot\cup A_2), f(B_1 \dot\cup B_2))$



이 알고리즘의 경우 $\tilde{O} (nk^2 + k T(maximum \, weighted \,matching))$ 정도 시간에 동작합니다.

## 5. 끝맺음

이 글에서는 matroid 상에서의 submodular maximization에 대한 여러 결정론적 알고리즘과 application에 대해 다루어 보았습니다. 개인적으로는 application 쪽이 특히 흥미로웠으며, 알고리즘의 경우에는 증명 등을 싣자면, 글의 지면이 너무 길어지는 것에 비해 흥미로운 내용은 아니라고 생각해서 싣지 못했고, 논문의 알고리즘들을 소개하는 선에서 그치게 되었습니다. 논문의 알고리즘들을 소개하는 정도였던 만큼, 알고리즘의 작동 과정에 대한 pseudo code 등도 논문의 것을 거의 가져오게 되었던 것 같습니다.

이 주제에 조금 더 관심있는 사람은 참고문헌의 논문들 혹은 random이 관여하는 알고리즘들에 대해 찾아 읽어봐도 좋을 것 같습니다.

## 참고문헌

- [1] K. Han 외 3인. "Deterministic Approximation for Submodular Maximization over a Matroid in Nearly Linear Time", Neurips 2020
- [2] N. Buchbbinder 외 2인, "Deterministic ($1/2 + \epsilon$)-Approximation for Submodular Maximization over a Matroid", SIAM 2019 
