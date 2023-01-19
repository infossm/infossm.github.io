---
layout: post
title: "Deterministic Almost-Linear Time Edge Connectivity, with and without expander decomposition"
author: TAMREF
date: 2023-01-18
tags: [graph-theory]
---

## Introduction

이곳에 글을 연재하기 시작할 즈음 두 개의 포스트에서 Minimum Cut 문제의 state-of-the-art라고 볼 수 있는 두 알고리즘을 소개했습니다. 하나는 적절한 random sampling을 통해 모든 min-cut을 sparse한 cut으로 근사한 [Karger 2000](https://infossm.github.io/blog/2021/03/21/minimum-cuts-in-near-linear-time/)에 대해 다루었고, 다른 하나는 almost-linear time complexity를 달성한 최초의 알고리즘인 [Li 2021](https://infossm.github.io/blog/2021/05/16/Deterministic-Mincut-in-Almost-Linear-Time/) 에 대해 다루었습니다. 둘 다 "seminal work"이라는 표현이 아깝지 않을 만큼 좋은 논문이지만, 이 중 Li에 대해서는 Expander Decomposition에 대한 제 이해가 부족하여 논문의 outline만 짚고 넘어갔습니다.

Li의 논문을 리뷰하며 스스로 느꼈던 것은 그가 (또한 최근의 flow, cut 관련 일련의 연구들이) 아주 당연하게 사용하고 있는 Expander Decomposition 기반의 black box를 저는 잘 받아들이지 못하고 있었고, Li가 왜 이 framework를 채택했는지, 기존의 방법에 비해 왜 효과적인지를 잘 알고 있지 못하다는 것이었습니다. 2000-2010년대에 들어서야 폭발적으로 발전하기 시작한 분야이다보니, 교과서 등의 정제된 자료를 접하기 어려운 것 또한 하나의 장벽이었습니다.

앞으로 몇 번에 걸쳐, Expander Decomposition의 supremacy나 그 이론적인 기원에 대해 다루는 글을 쓰고자 합니다. 관련된 연구를 active하게 하는 Jason Li, Thatchaphol Saranurak 등의 저자들이 최근 괄목할 성과들을 내고 있기 때문에, 이를 알아두면 이론 전산학의 최신 동향에 대해 지식이 깊어질 것이라 생각합니다. 첫 소재는 Li (2021) 보다는 좀 더 간단한 세팅의, **simple unweighted** graph에서 min-cut, 즉 edge connectivity를 구하는 문제에 대해 다루어 보려고 합니다. Saranurak (2020)의 Expander decomposition을 이용한 간단한 증명과, 첫 near-linear time algorithm인 Kawarabayashi & Thorup (2015)를 비교해봅니다. 둘은 모두 간단한 중심 아이디어에 기반하고 있지만, Saranurak의 논문은 5페이지로 끝나고 K&T의 논문은 50페이지를 넘기는데, *almost linear time deterministic expander decomposition (CGL+, 2019)* 의 맹활약을 볼 수 있습니다.

## Background

이 문제의 randomized version은 Karger에 의해 이미 정복되었지만, Deterministic version은 20년동안 Gabow의 tree packing algorithm이 SOTA였습니다. 이 알고리즘은 output-sensitive algorithm으로 min-cut의 값을 $\lambda$라고 할 때, $O(m + \lambda^{2} n \log(n / \lambda))$ 시간에 동작하는 알고리즘입니다. Minimum degree $\delta \le \frac{2m}{n}$이 $\delta \ge \lambda$를 만족하니, 대략 $\tilde{O}(mn)$이나 $\tilde{O}(m^{2} / n)$ 정도의 quadratic scale로 생각할 수 있습니다. 보통은 $\tilde{O}(\lambda m)$ 시간에 동작하는 알고리즘으로 생각하고 black box로 씁니다.

K&T의 기본적인 관찰은 simple graph에서 $\delta$가 커지면 non-trivial min cut의 conductance가 그에 반비례하여 감소한다는 것에서 출발합니다.

**Definition.** 
- 그래프 $G = (V, E)$의 두 정점 집합 $A, B$에 대해 $E(A, B)$를 $A, B$에 endpoint를 두고 있는 간선의 집합이라고 정의하자. 편의상 $E(A, B)$와 집합의 크기 $\left\lvert E(A, B) \right\rvert$를 혼용한다.

- 비어 있지 않은 정점 집합 $S$에 대해 $E(S, V-S)$ (혹은 $S$ 그 자체)를 **cut**이라고 한다. 또한 이 중 크기가 최소인 것을 **minimum cut**이라고 한다. $E(v, V - v)$ 또한 cut이기 때문에 minimum cut은 minimum degree 이하임에 주목하자.

- 역시 정점 집합 $S$에 대해 $\sum_{v \in S} \deg(v)$를 $\mathrm{vol}(S)$로 정의한다. 이제 $S$에 연결된 간선들 중 $S$ 바깥으로 나가는 간선의 비율을 $\Phi(S) := \frac{E(S, V-S)}{\mathrm{vol}(S)}$로 정의한다. 일반성을 잃지 않고 $\mathrm{vol}(S) \le \mathrm{vol}(V - S)$를 가정하자. $\Phi(V - S) = \Phi(S)$로 정의하면 된다.

- 두 집합 모두의 크기가 $2$ 이상인 cut을 **non-trivial cut**이라고 하자. 그렇지 않은 cut 중 최솟값은 minimum degree $\delta$로, 이들은 trivial cut이라고 부른다. non-trivial cut의 최솟값을 $\lambda_{nt}$라고 두면 $\lambda = \min(\lambda_{nt}, \delta)$가 된다.

**Observation.** Min-cut 중 non-trivial minimum cut $S$를 생각하자. 이 때 $\Phi(S) \le 1 / \delta$를 만족한다.

우리가 $\delta$ 가 매우 큰 상황만 보면 된다는 사실에 주목합시다. K&T를 기준으로 $\delta \ge \log^{6} n$ 인 상황만 볼 건데, 그 이하의 $\delta$는 그냥 Gabow95의 알고리즘을 돌리면 $\tilde{O}(\delta m) = \tilde{O}(m)$ 시간에 알고리즘이 동작하기 때문입니다. 이를 감안하면, min-cut에 관여하는 간선은 전체 간선에 비해 매우 적다는 것을 관찰할 수 있고, **대부분의 min-cut에 관여하지 않는 component**들을 contract시키고 싶다는 생각을 하게 됩니다. 이 관찰을 우선 증명하고 넘어가도록 합시다.

*Proof.* $G$가 simple graph라는 사실에 강하게 의존합니다. $E(S, V-S) \le \delta$와 $\deg(v) \ge \delta$를 생각하면, $\delta \lvert S \rvert \le \mathrm{vol}(S) \le \lvert S \rvert (\lvert S \rvert - 1) + \delta$가 성립하게 됩니다. 오른쪽 부등식은 $G$가 simple이기 때문에 성립하는 성질임에 주목합시다. 식을 정리하면 $(\lvert S \rvert - 1)(\lvert S \rvert - \delta) \ge 0$이 성립해야 하는데, $\lvert S \rvert > 1$이므로 $\lvert S \rvert \ge \delta$를 만족해야 합니다. 그러면 자연스럽게 $\mathrm{vol}(S) \ge \delta^{2}$이 얻어지니 $\Phi(S) \le \frac{1}{\delta}$가 성립합니다. $\square$

모든 유의미한 min-cut은 low conductance를 가지니, min-cut에 관여하지 않는 component란 결국 high conductance를 갖는 component일 것입니다. 이는 널리 알려진 expander의 정의와 일치합니다.

## Saranurak 2020: Using Expander Decomposition

**Definition.** 임의의 $S \subseteq V$에 대해 $\Phi(S) \ge \phi$이면, $G = (V, E)$를 $\phi$-expander라고 한다. 또한, 다음 조건을 만족하는 vertex partition $\mathcal{X} = \lbrace X_{1}, \cdots, X_{k}\rbrace$ 를 $(\varepsilon, \phi)$-expander decomposition이라 한다.

- 당연하게도, $X_{1} \sqcup \cdots \sqcup X_{k} = V$.
- $X_{i}$는 각각 $\phi$-expander이다.
- 간선 중 서로 다른 $X_{i}, X_{j}$를 연결하는 것은 많아야 $\varepsilon \lvert E \rvert$개이다.

우리는 $\phi \ge \frac{1}{\delta} = \log^{-c}n$가 필요하고, $\varepsilon = \phi \cdot m^{o(1)}$으로 최대한 작기를 바랍니다. 알려진 사실은
- $\varepsilon = O(\phi \log m)$ 이 가능한 하한으로 알려져 있지만, NP-hard입니다.
- $\varepsilon = O(\phi \log^{3} m)$인 poly-log time randomized algorithm이 존재합니다. (Saranurak, 2018)
- 비슷한 테크닉으로, $\gamma(m, r) = m^{o(1)}$에 대해 ($r$은 조절 가능한 parameter) $\varepsilon = O(\phi\gamma)$인 $O(m\gamma)$ 시간 **deterministic** 알고리즘이 존재합니다. (CGL+, 2019)

CGL+의 결과를 black box로 사용할 겁니다. 정확히는, 두 번째 조건을 좀 더 강화하여 사용합니다.

**Lemma. (Saranurak 2020, Based on CGL+ 2019)** $(O(\phi\gamma), \phi)$-expander decomposition 중 아래 조건을 추가로 만족하는 것을 찾을 수 있다.
- $S \subseteq X_{i}$는 $\frac{E(S, X_{i} - S)}{\mathrm{vol}_{G}(S)} \ge \phi$를 만족한다.

각각이 $\phi$-expander라는 말은 위 statement에 $\mathrm{vol}_{G}$ 대신 induced subgraph $G[X_{i}]$에 대해 $\mathrm{vol}_{G[X_{i}]}$를 넣어야 정확한 statement가 됩니다. 이를 $\mathrm{vol}_{G}(S)$로 바꾸는 일은 어렵지 않은데, 각 정점에 $\deg_{G}(v)$개의 self loop를 추가한 그래프 $\overline{G}$에서 expander decomposition을 찾으면 됩니다. CGL+의 결과는 $G$의 simple-ness에 의존하는 결과가 아닙니다. $\square$

이제 위 Lemma를 사용하여 얻어진 $G$의 expander decomposition을 $E(G, \phi)$라고 둡시다. $\phi \ge \frac{1}{\delta}$를 원하니 $\mathcal{X} := E(G, \frac{40}{\delta})$를 구한다고 하면, 우리는 각 $X \in \mathcal{X}$를 contract해서 간선의 개수를 $O(m / \delta)$ 개 정도로 줄이는 것이 목표입니다. 다만 어떤 $v \in X_{i}$가 너무 많은 간선을 잃어버려서 degree가 $\delta / 100 \ll \frac{1}{\phi}$ 미만으로 떨어진 경우 $G[X_{i}]$를 잘라버리는 non-trivial min cut이 존재할 수도 있습니다. 이 경우 $X_{i}$를 contract하면 안되고, 이러한 $v$들을 반복적으로 제거해줍시다. 정확히는 $\deg_{G[X_{i}]}(v) < \frac{2}{5}\deg_{G}(v)$인 $v$를 반복적으로 제거하고, 제거가 끝난 정점들을 $X_{i} := \mathsf{Trim}(X_{i})$이라고 합시다.

$\mathsf{Trim}(X_{i})$에 min-cut의 정점이 별로 없다는 걸 보이기 위해, 우선 $X_{i}$도 상당히 적은 개수의 min-cut정점만을 포함하고 있음을 보입시다.

**Lemma 1.** $G$의 Non-trivial min cut $C$와 $X \in \mathcal{X}$에 대해, $\min(\lvert C \cap X \rvert, \lvert (V - C) \cap X \rvert) \le \frac{\lambda}{40}.$

*Proof.* 일반성을 잃지 않고 $\lvert C \cap X \rvert \le \lvert (V - C) \cap X \rvert$라고 둡시다. $E(C \cap X, (V - C) \cap X) \le \lambda$일테니, expander의 성질에 의해

$$\begin{aligned}\lambda &\ge E(C \cap X, (V - C) \cap X)\\ &\ge \frac{40}{\delta}\min(\mathrm{vol}(C \cap X), \mathrm{vol}((V - C) \cap X))\\ &\ge 40\min(\lvert C \cap X \rvert, \lvert (V - C) \cap X \rvert).\end{aligned}$$

**Lemma 2.** $Y := \mathsf{Trim}(X)$에 대해, $\min(\lvert C \cap Y \rvert, \lvert (V - C) \cap Y \rvert) \le 2.$

*Proof.* Saranurak (2020)의 전개를 따라 가도록 하겠습니다. 똑같이 $\lvert C \cap Y \rvert \le \lvert (V- C) \cap Y \rvert$를 가정합니다. $E(C \cap Y, (V - C) \cap Y) = \mathrm{vol}_{G[Y]}(Y \cap C) - 2\lvert E(G[C \cap Y])\rvert \ge \frac{2}{5}\delta\lvert C \cap Y \rvert - \lvert C \cap Y \rvert^{2}$가 성립합니다. $\mathsf{Trim}$의 정의에 따라 minimum degree가 $\frac{2}{5}\delta$임에 주목합시다. $y := \lvert C \cap Y \rvert$라고 두면

$$
\begin{aligned}
\delta \ge \lambda \ge E(C \cap Y, (V - C) \cap Y) \ge \frac{2}{5}\delta y - y^{2} = y(\frac{2}{5}\delta - y)
\end{aligned}
$$

이 때 $y \le \min(\lvert C \cap X \rvert, \lvert X - C \rvert) \le \frac{\delta}{40} < \frac{2}{5}\delta$이므로, $y \ge 3$일 경우 위 부등식에서 $y \ge \frac{\delta}{20}$를 얻어 모순입니다. 이제 $\mathsf{Trim}(X)$에 있는 대부분의 정점은 min-cut과 교차하지 않지만, $X$를 contract하려면 아예 non-trivial min-cut과 교차하는 점이 없어야 합니다. 조금 더 정제해서, $\mathsf{Shave}(X)$를 $X$의 정점 중 neighbor의 과반이 $X$에 있는 정점들이라고 정의합시다. 정확히는 $\deg_{G[X]}(v) > \deg_{G}(v) / 2 + 1$인 점들로 정의합니다.

**Lemma 3.** $Z := \mathsf{Shave}(Y) = \mathsf{Shave}(\mathsf{Trim}(X))$에 임의의 non-trivial min cut $C$는 $Z \subseteq C$ 혹은 $Z \cap C = \empty$를 만족한다. 즉, $\min(\lvert Z \cap C \rvert, \lvert Z - C \rvert) = 0$.

*Proof.* 편의상 $\lvert Y \cap C \rvert \le \lvert Y - C \rvert$를 가정하고, 귀류법으로 $\min(\lvert Z \cap C \rvert, \lvert Z - C \rvert) \ge 1$이라고 가정합시다. 즉, $v \in Z \cap C \subseteq Y \cap C$를 찾을 수 있습니다. 이 때 $\lvert Y \cap C \rvert \le 2$이니, $v$는 $Y \cap C$로 많아야 한 개의 간선을 뻗고 있습니다. $V - C$는 $Z - C$를 포함하고 있으니, 최소한 $\frac{\deg_{G}(v)}{2}$개보다 많은 $v$의 간선이 cut edge로 관여하고 있는 셈입니다. 따라서 $v$를 $V - C$로 넘기면 더 작은 (non-trivial은 아닐 수 있지만) cut을 만들 수 있고, 이는 $C$가 min-cut이라는 조건에 모순이 됩니다. $\square$

따라서 Expander Decomposition + $\mathsf{Trim}$ + $\mathsf{Shave}$ 를 통해 contract하더라도 non-trivial min-cut structure에 지장이 없는 컴포넌트들을 구할 수 있습니다. contraction에서 $X_{i}$의 internal edge들은 삭제될텐데, 몇 개의 간선들이 남아 있을까요?

- 우선 맨 처음에 inter-component edge는 $O(\phi \gamma m) = O(m^{1 + o(1)} / \delta)$개가 존재할 것입니다. 이들을 Rank 0 간선이라고 합시다.
- 다음으로 $\mathsf{Trim}$ 과정에서 방출된 정점에 연결된 간선들이 있는데, Rank 0 간선 하나는 많아야 2개의 방출된 정점에 기여할 수 있습니다. 즉, Rank 0 간선이 $x$개라면 $2x$점의 점수가 쌓여 있다고 생각할 수 있습니다. 또 정점 하나를 방출하기 위해서는 최소 $\frac{3}{5}\deg (v)$점이 필요하고, $v$에 의해 새로 공급되는 (다른 정점을 방출시킬 수 있는) 점수는 $\frac{2}{5} \deg(v)$ 점입니다. 따라서 최소한 $\frac{1}{5}\deg(v)$ 점의 점수를 소모하여 많아야 $\frac{2}{5}\deg(v)$의 간선을 빼내는 셈이니, 모든 점수를 효율적으로 쓴다고 해도 많아야 $4x$개의 간선이 Trim 과정에서 새로 방출됩니다.
- 마지막으로 $\mathsf{Shave}$ 과정에서 새로 방출되는 간선들을 생각해봅시다. 방출되는 정점 $v$마다 최소 $\frac{\deg(v)}{2} - 1$ 개 초과의 간선이 $\mathsf{Trim}(X)$ 바깥으로 향해야 하기 때문에 $\delta \ge 4$를 가정하면 $v$가 $\mathsf{Trim}(X)$에 뻗는 간선의 개수는 이미 $\mathsf{Trim}(X)$ 바깥으로 뻗어나가던 간선의 개수의 $\frac{\deg(v) / 2 + 1}{\deg(v) / 2 - 1} \le 3$배 미만입니다. 그런데 $\mathsf{Trim}(X_{1}), \cdots, \mathsf{Trim}(X_{k})$ 의 inter-component edge는 많아야 $5x$개이니 $\mathsf{Shave}$에서 새로 꺼내어지는 간선의 개수는 많아야 $30x$개가 됩니다. (inter-component 간선 하나 당 최대 2번의 기여를 할 수 있기 때문)

즉, $\mathsf{Shave}(\mathsf{Trim}(X_{i}))$들을 모두 contract시킬 경우 $O(\frac{m\gamma}{\delta})$개의 간선만 남아 있는, non-trivial min cut structure를 보존하는 그래프를 만들 수 있고, 여기서 Gabow's algorithm을 적용하면 $O(m^{1 + o(1)})$ 시간에 edge connectivity를 구할 수 있습니다.

여담으로 이 압축 알고리즘은 non-trivial min cut을 모두 보존하기 때문에, Cactus Representation of Minimum Cut 알고리즘도 그대로 적용할 수 있습니다.

## K&T 2015: Using PageRank diffusion Method

이렇게 "conceptually simple"한 Saranurak 2020 소개를 마쳤습니다. 듣기만 해서는 Expander decomposition이 중요한 건지, Trim, Shave 등의 연산이 중요한 건지 감이 오지 않을 수 있는데, 사실 Trim-Shave로 대표되는 framework는 K&T가 이미 주장했습니다.

K&T, Henzinger et al, Saranurak은 모두 Certify-or-cut이라는 큰 틀의 framework에서 동작합니다.

- 적당한 컴포넌트 $C$에 대해, 다음을 판별한다:

  1. 어떤 non-trivial minimum cut도 $C$와 만나지 않는다. 이는 대개 $C$가 expander임을 증명한 후, $\mathsf{Trim}, \mathsf{Shave}$등의 연산을 적용하여 얻는다.
  2. $C$에서 conductance가 $o(1 / \log m)$인 cut을 하나 찾는다.

- 2의 경우, cut edge들을 모두 꺼낸다. 이 때 꺼내는 간선의 수는 $m$에 비해 "많지 않다".
- 1의 경우 $C$를 contract한다. 이 때 생기는 **parallel edge등을 주의해서 처리한다.**
  
- 그래프의 간선이 $O(m / \delta)$개로 떨어져 Gabow's algorithm을 적용할 수 있다면, 꺼내두었던 low-conductance cut edge들을 모두 집어넣은 뒤 Gabow's algorithm을 돌린 후 종료한다.

Saranurak에서는 기본적으로 Expander Decomposition이 1의 경우를 모두 커버해주고 있기 때문에 문제가 안 되지만, K&T와 Henzinger에서는 이를 처리하는 게 쉽지 않습니다. 때문에 비교적 1 or 2를 판별하는 알고리즘은 간단한 데 비해 논문들이 가공할 길이를 자랑합니다.

2의 경우에 꺼내는 간선이 많지 않다는 말은 $o(m)$개의 간선만을 꺼내고 있다는 말과 동일합니다. 결국 cut $(S, V - S)$이 반복적으로 제거되면서 여러 연결 컴포넌트가 생겨나는 와중에, $\mathrm{vol}(S) \le \mathrm{vol}(V - S)$라고 가정하면 cut edge는 $S$의 간선들 중에서 $o(1 / \log m)$의 비율만 차지합니다. 그런데 한 간선이 volume이 작은 small-side에만 계속해서 있을 수는 없습니다. $E(S, V - S)$를 제거하고 나면 $\mathrm{vol}(S)$는 절반 이하로 줄어들고, 또 $E(T, S - T)$를 제거하고 나면 작은 쪽의 volume은 절반으로 줄어듭니다. 즉 언젠가 cut에 걸려서 나가거나 expander에 들어가기 전까지, 각 간선이 small side에 붙어있을 수 있는 기회는 $\log m$번입니다. 따라서 2의 경우에 걸려서 빠져나오는 cut edge의 개수는 $o(1 / \log m) \cdot m \lg m = o(m)$개가 됩니다.

Expander Decomposition의 가공할 위력을 실감하고 글을 끝내는 것도 좋겠지만, 간단히 K&T의 아이디어를 소개하려고 합니다. 저자들 본인도 PageRank를 low-conductance cut을 찾는데 활용하는 신선한 아이디어를 찾아낸 것에 자부심을 갖고 있는 듯합니다.

PageRank 알고리즘은 구글의 검색 엔진으로도 잘 알려져 있지만, Anderssen et al (2007) 에 의해 low conductance cut을 찾는 알고리즘으로도 잘 알려졌습니다. Initial mass distribution $p_{0}$가 주어졌을 때, 임의의 정점을 골라 아래의 `PUSH` 과정을 반복합니다.

- $r$: residual mass distribution. $v$의 아직 이동할 수 있는 질량. 초기값은 $p_{0}$.
- $p$: settled mass distribution. $v$에 붙어 움직이지 않는 질량. 초기값은 $0$.
- $\alpha$: teleporation constant. $v$에서 다른 정점들로 이동하는 질량의 "비율"

`push v`는 $r(v)$를 $(1 - \alpha)r(v) / 2$으로 만들고, $\alpha r(v)$는 $p(v)$에, $v$의 각 이웃 $w$의 $r(w)$에 $(1 - \alpha) r(v) / 2\deg(v)$씩을 더해주는 방식의 update를 말합니다. 이 때 각 $p_{0}$에 대해, 유일한 limit distribution $p^{*}$가 존재합니다.

**Fact 1.** 임의의 간선 $(u, v)$에 대해, 이 간선의 net flow는 결국 $\frac{1-\alpha}{2\alpha}\left(\frac{p(u)}{\deg(u)} - \frac{p(v)}{\deg(v)}\right)$이다. 

**Corollary 2.** 어떤 시점에서 모든 점의 residual mass $r(u)$가 $r(u) \le \mu$로 bound된다면, 어떤 min-cut을 넘어갈 수 있는 mass의 총량은 최대 $\frac{\mu}{2\alpha}$ 이하이다.

즉, 어떤 정점 $v$가 non-trivial min cut $(S, V-S)$에서 $S$의 "center"에 위치한다고 합시다. 이는 곧 neighbor의 많은 부분이 역시 $S$에 속한다는 의미입니다. initial mass를 $r(v) = 1$의 point mass로 두고 PageRank를 실행하면, 우선 $v$의 대부분의 mass가 $S$에 균등하게 퍼질 것이고, 언젠가는 모든 점의 mass가 특정 값 아래로 내려가게 됩니다. 이후부터는 cut을 통해 넘어갈 수 있는 mass가 굉장히 제한적이기 때문에, $S$에 비정상적으로 많은 mass가 모여 있게 됩니다. 이러한 set이 존재하는 경우, Anderssen의 결과를 이용하여 near-linear time에 low conductance cut을 찾을 수 있게 됩니다.

원래는 이러한 $v \in S$를 찾는 것이 가능하지 않기 때문에, random guess를 통해 여러 번 시도하는 것이 일반적이었습니다. (low conductance cut을 찾지 못한 경우 적절한 error를 report할 수 있기 때문에 이는 가능합니다) De-randomization을 위해 수많은 기술적인 부분이 들어가 있지만, 모두 생략하고 핵심만 전달하면 다음과 같습니다.

- 충분히 많은 $v$에 대해 위의 알고리즘을 시도하여, $v$를 포함하는 low conductance cut이 있는지 판별한다.
- 그렇지 않은 경우, 이제까지의 $v$들은 전부 $V - S$에 있거나 non-trivial min cut $(S, V-S)$는 존재하지 않는다.
- 이제까지의 $v$들에 uniform mass를 주고 PageRank를 실행하면, $S$에는 비정상적으로 낮은 mass가 모이게 된다.
- 매우 낮은 mass가 모인 경우에 low conductance cut을 구하거나 error를 뱉는 알고리즘 또한 존재한다. (Anderssen, K&T) 이를 이용해서 low conductance cut을 찾거나 그러한 cut이 존재하지 않는다고 결론을 내린다.

결국 이는 "conductance"라는 표현에 어느 정도 직관을 제공합니다. $v$에서 어떤 사람 (혹은 입자) $\alpha$의 확률로 영영 $v$에 영영 드러눕거나, 혹은 이웃한 정점으로 uniform하게 이동하는 random walk에 대응시켜 생각해 볼 때, 충분한 시간이 지난 후 이 사람의 probability distribution은 low-conductance cut을 경계로 출발점을 포함하는 쪽과 그렇지 않은 쪽에 상당히 불균등하게 분포하게 됩니다. 반대로 그래프가 expander라면 시간이 지났을 때 probability distribution이 모든 정점에 고르게 mix-up되어, 이 사람이 어떤 정점에서 출발했는지 분간하는 일이 매우 어렵게 됩니다.

## Conclusion

Simple graph에서 edge connectivity를 찾는 두 가지의 알고리즘을 다루었는데, 두 알고리즘 모두 현재 SOTA를 점유하고 있지는 않습니다. 하지만 둘 모두 expander가 갖는 여러 유용한 성질을 exploit하여 문제에 접근했다는 점은 동일하며, 최근의 복잡한 논문들에 비해 expander graph의 beneficial한 면을 직관적으로 보여주는 방법들이기도 합니다.

또한 graph의 expander decomposition을 randomized(Saranurak, 2018)이건 deterministic이건(CGL+, 2019) 빠른 시간 복잡도 안에 구할 수 있다는 사실이 가진 강력한 함의도 엿볼 수 있었습니다. 다만 Expander decomposition의 간단하고 efficient한 구현은 알려진 바가 없고,(간단하지 않은 구현은 아래 [Github](https://github.com/Skantz/expander-decomposition) 등이 있습니다) 기존의 방법들이 현실적으로 구현되기까지는 사실상 멀고 험난한 시간이 예상됩니다. 수많은 문제의 복잡도를 almost-linear time으로 닫아버린 expander-decomposition 기반의 방법론이 fast matrix multiplication처럼 이론 속의 신기루로 머물게 될지, 경시에서 사용이 가능할 만한 혁신적인 알고리즘이 될지 지켜보는 것도 재미있을 듯합니다.

## References

- Saranurak, Thatchaphol. "A simple deterministic algorithm for edge connectivity." Symposium on Simplicity in Algorithms (SOSA). Society for Industrial and Applied Mathematics, 2021.

- Kawarabayashi, Ken-ichi, and Mikkel Thorup. "Deterministic edge connectivity in near-linear time." Journal of the ACM (JACM) 66.1 (2018): 1-50.


- Henzinger, Monika, Satish Rao, and Di Wang. "Local flow partitioning for faster edge connectivity." SIAM Journal on Computing 49.1 (2020): 1-36.

위 세 논문은 모두 deterministic edge connectivity를 다룬 논문입니다. SOSA 2021에 등재된 Saranurak의 논문이 메인이고, Kawarabayashi의 논문이 후반에 다룬 논문입니다. Henzinger의 논문은 edge connectivity를 $O(m \log^{2} n \log\log^{2} n)$ 시간에 구하는, state-of-the art 시간복잡도를 보유한 논문입니다.

- Chuzhoy, Julia, et al. "A deterministic algorithm for balanced cut with applications to dynamic connectivity, flows, and beyond." 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 2020.
- Saranurak, Thatchaphol, and Di Wang. "Expander decomposition and pruning: Faster, stronger, and simpler." Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2019.

Expander decomposition을 빠르게 구하는, 각각 deterministic / randomized algorithm입니다. 후자는 이 글에서 다루지 않았지만, [koosaga님이 자세하게 다룬 적이 있습니다.](https://infossm.github.io/blog/2021/04/01/expander-decomposition/) Deterministic version에서도 후자의 cut-matching game을 중요하게 다루기 때문에, 관심이 있다면 일독을 권합니다.

- https://youtu.be/G8P766sGgFo : Mikkel Thorup이 본인의 아이디어를 설명한 유튜브 강의입니다. 논문의 introduction 부분을 간단하게 설명하고 있으니, 50페이지짜리 논문을 모두 읽는 것이 두렵다면 이걸 보는 것도 좋습니다.

- https://youtu.be/otAimttq59c : 마찬가지로 Thatchaphol Saranurak이 본인의 논문을 설명한 영상입니다. 논문이 어렵지 않으니 그냥 논문을 읽으셔도 좋습니다.

