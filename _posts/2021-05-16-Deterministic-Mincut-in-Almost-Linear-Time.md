---
layout: post
title: "Deterministic Mincut in Almost-Linear Time"
author: TAMREF
date: 2021-05-16
tags: [graph-theory, maximum-flow]
---
# Introduction

지난 [3월 포스팅 “Minimum Cuts in Near Linear Time”](http://www.secmem.org/blog/2021/03/21/minimum-cuts-in-near-linear-time/)에서 Weighted min-cut을 near-linear time에 구해주는 Karger의 Monte-Carlo algorithm을 소개한 적이 있었습니다. 이 글에서는 Karger의 알고리즘을 almost-linear complexity로 유지하며 derandomize하는 데 성공한 Jason Li의 2021년 논문 [Determinstic Mincut in Almost Linear Time](http://www.cs.cmu.edu/~jmli/papers/deterministic-mincut-in-almost-linear.pdf)을 리뷰합니다. 논문 자체가 기술적으로 복잡한 부분이 많은 만큼, 깊숙한 디테일을 모두 다루기보다는 특별한 경우에 대해 이 논문에서 소개한 방법론이 어떻게 적용되는지를 다루어보도록 합니다.

## Terminologies

일부는 3월 포스팅과 동일하지만, Li (2021)의 문법을 대체로 따르면서 변화한 부분이 있습니다.

- 모든 weight는 $[\frac{1}{n^{O(1)}}, n^{O(1)}]$인 양의 실수입니다. 따라서 입력의 길이는 $O(m \log n)$입니다.

- Graph $G$의 정점 부분집합 $S$에 대해, **cut**또는 **cutset**을 $\partial S$로 표기하고, 이는 $E(S, V - S)$; 즉, 한 끝점은 $S$에, 다른 끝점은 $V - S$에 맞닿은 간선들의 집합으로 정의합니다. $S$와 $V - S$는 공집합이 아닙니다.

- 간선 집합 $A$의 크기(unweighted graph의 경우) 또는 weight의 합(weighted graph의 경우)를 일괄적으로 $w(A)$로 씁니다. 한 예로, 정점 $v$의 차수 $\deg (v)$를 $w(\partial\lbrace v\rbrace)$로 쓸 수 있습니다.
- Min-Cut이란 $w(\partial S)$가 가능한 $S$들중 최소인 정점 집합 $S^{\ast}$, 혹은 cut $\partial S^{\ast}$ 그 자체, 아니면 $c = w(\partial S^{\ast})$를 의미합니다. 그래프를 disconnect하기 위해 끊어야 하는 에지의 양이기 때문에 edge-connectivity라고도 곧잘 부르며, $\lambda$로도 씁니다. 당연하지만 min-cut은 여러 개 있을 수 있습니다.
- $V - S$의 모든 정점과 그에 연결된 간선을 지워서 만든 부분그래프를 induced subgraph $G[S]$로 씁니다.
- 정점 집합 $S$의 volume $\mathrm{vol}(S)$를 $\sum _ {v \in S} \deg (v)$로 정의합니다.
  $w(\partial S) = \mathrm{vol}(S) - \sum _ {u \neq v \in S} w(u, v) = \mathrm{vol}(S) - 2\sum _ {e \in G[S]} w(e)$ 임에 주목하세요. minimum degree를 $\delta$로 쓰곤 합니다. $\delta \ge \lambda$임에 주목하세요.
- $S$의 conductance $\Phi(S)$를 $w(E(S, V-S)) / \min\lbrace  \mathrm{vol}(S), \mathrm{vol}(V - S) \rbrace$로 씁니다. 일반성을 잃지 않고 $\mathrm{vol}(S) \le \mathrm{vol}(V - S)$가 성립한다고 가정할 수 있습니다. $S$의 간선들 중에 바깥으로 새어나가는 간선들의 비율을 나타냅니다. 항상 $\Phi \le 1$임에 주목하세요.
- 실수 $\phi$에 대해 $\min _ {S} \Phi(S) \ge \phi$이면 그래프 $G$를 $\phi$-expander라고 부릅니다. 즉, 어떤 $S$를 가져와도 cutset이 “상대적으로 크다”는 것을 의미하기 때문에, 잘 연결된 그래프라고 생각하면 받아들이기 좋습니다.

## Updated History

Weighted MinCut에 대한 고전적인 접근은 3월 포스팅의 첫 문단에 상세하게 설명되어 있습니다. 여기서는 Karger 이후, deterministic한 weighted min-cut을 구하기 위한 학계의 역사를 간단히 짚어봅니다. 이 요약 역시 Li(2021)에 기반하고 있습니다. 편의상 이 단원에서는 Low-conductance cut / Expander Decomposition 등이 무엇인지 설명하지 않습니다.

첫 번째 Breakthrough는 Kawarabayashi & Thorup (2015)로, **Unweighted graph**의 minimum cut을 $O(m \log^{12} n)$에 구하는 알고리즘을 제안했습니다. 여기서 Cut의 당연한 upper bound 중 하나는 minimum degree $\delta \le 2m / n$이 되는데, 이것보다 작은 non-trivial cut들을 구하기 위해 Low-conductance cut을 활용합니다. 즉, Low-conductance cut이 Global min-cut에 활용될 여지가 있다는 것을 최초로 입증한 논문인 셈이 됩니다. 이후 몇 가지 최적화를 통해 복잡도는 $O(m \log^{2} n \log\log^{2} n)$까지 내려가 있는 상태입니다. (Henzinger et al, 2017) 이후 Saranurak (2021)이 Expander Decomposition을 이용하여 간단한 $m^{1 + o(1)}$ 시간 알고리즘을 제안하기도 했습니다.

한편 Li (2021)은 Gomory-Hu 이후로 멈춰 있었던 Min cut with flow oracle에 대한 유의미한 결과를 지속적으로 내놓고 있었습니다. 단순한 방법으로  min-cut을 구하기 위해서는 $\binom{n}{2}$번의 maximum flow ($s$-$t$ min cut) 연산이 필요했던 것을 Gomory-Hu가 $n-1$번으로 줄인 것이 1961년인데, 이후 거의 50년만에 $O(\log^{O(1)}n)$번의 maximum flow 연산으로 global min-cut을 구할 수 있다는 사실이 입증되었습니다. 다만 flow algorithm의 발전 속도가 예상만큼 빠르지 않은데, unweighted graph에 대해서는 $\tilde{O}(m^{4/3 + o(1)})$ (Liu & Sidford), weighted graph에 대해서는 $\tilde{O}(m^{3/2})$ (Goldberg-Rao) 정도로 almost linear time을 기대하기는 어려운 상황입니다. 결국 Li 자신이 flow-independent한 almost linear time min-cut을 제시함으로써 그 속도를 앞지르고야 말았습니다.

# Approach from Li

## How to de-randomize Karger’s sampling?

[3월 포스팅 “Minimum Cuts in Near Linear Time”](http://www.secmem.org/blog/2021/03/21/minimum-cuts-in-near-linear-time/)을 읽고 오시면 이해하시기 더욱 좋습니다.

Karger (2000)의 알고리즘에서 가장 핵심적인 부분이자, 유일하게 랜덤이 들어가는 부분을 다시 봅시다. Random sampling을 통해 이루고자 하는 바는, 적당한 상수 $\varepsilon$에 대해 다음을 만족하는 $G$의 **unweighted** subgraph $H$를 $O(m + n\log^{3} n)$ 시간 안에 만드는 것입니다.

- $H$의 간선은 $m' = O(n\varepsilon^{-2} \log n)$개이다.
- $H$의 minimum cut은 $c’ = O(\varepsilon^{-2} \log n)$이다.

- $G$의 minimum cut은 높은 확률로 $H$의 $(1 + \varepsilon)$-minimum cut이다.

편의를 위해 $\varepsilon = \frac{1}{6}$이라는 적당한 값을 잡아 봅시다. $H$의 maximum tree packing은 **Gabow92**를 사용하면 $O(\varepsilon^{-4} n \log^{3} n)$ 시간에 구할 수 있고, 그 크기는 $O(\varepsilon^{-2}\log n)$일 것입니다 (by weight). 이 때, **Theorem 1**에 의해 개중 $\frac{1}{2}(3 - \frac{1 + \varepsilon}{1 / 2}) = \frac{1}{3}$ 이상의 비율이 $G$의 min-cut을 $2$-constrain할 것입니다. 따라서 $O(\log n)$개의 트리를 모두 시도하면 min-cut을 $2$-constrain하는 트리를 잡아낼 수 있습니다. 따라서 전체 시간복잡도가 $O(T _ {2}(m, n) \log n + m + \varepsilon^{-4}n\log^{3} n)$이 됩니다. 다만, Karger(2000)의 알고리즘은 $H$를 항상 결정론적으로 생성할 수 없기 때문에, 이 모든 사실이 “높은 확률로 성립한다”고 state하는 것에 만족해야 했습니다.

Karger의 알고리즘에서 원하는 것은 결국 $m’, c’$가 $m’c’ = O(m^{1 + o(1)})$으로 충분히 작고, $G$의 min-cut은 항상 $H$의 approximate min-cut이어서 maximum packing의 많은 트리들을 $2$-respect하는 “어떤 그래프” $H$입니다. $H$가 $G$의 subgraph일 필요가 없음에 주목하세요. 이러한 바람을 바탕으로 $H$에 걸린 조건들을 relax해봅시다.

- $H$는 unweighted graph이고, 간선이 $m’$개인 동시에 min-cut이 $c’$이다. 당연하지만 $V(H) = V(G)$.
- 상수 $k$가 고정되어 있을 때,
  - $G$의 임의의 min-cut $S^{\ast}$에 대해 $w _ {H}(\partial S^{\ast}) \le (1 + \varepsilon)k$이 성립한다. 이를 편의상 “$H$가 min-cut들을 $\varepsilon$-cap한다”고 표현하겠습니다.
  - $G$의 임의의 *cut* $S$에 대해 $w _ {H}(\partial S) \ge (1 - \varepsilon)k$이 성립한다. 편의상 “$H$가 모든 cut들을 $\varepsilon$-support한다”고 표현합니다.

이 조건을 모두 만족하는 $H$를 만들 수 있다면 $G$의 min-cut은 $H$의 $(1 + \varepsilon) / (1 - \varepsilon)$ min-cut이 됩니다. $\varepsilon = 0.01$ 정도로만 두어도 $7/6$-approximation min cut은 충분히 보장할 수 있습니다. 이 논문에서 가장 중요한 theorem이 등장합니다:

**Central Theorem (Li, 2021)**: $\varepsilon$-cap, $\varepsilon$-support 조건을 모두 만족하고, 
$m’ = m^{1 + o(1)}$, $c’ = n^{o(1)}$, $k = \lambda / W = n^{o(1)}$인 unweighted graph $H$와 
상수 $W = \varepsilon^{4}\lambda /2^{O(\log n)^{5/6} \log\log^{O(1)} n}$을 $\varepsilon^{-4}2^{O(\log n)^{5/6} \log\log^{O(1)} n} m$ 시간에 deterministic하게(!) 찾을 수 있다.

복잡도가 수상하게 생겼지만, 어쨌든 $m \cdot n^{o(1)} = m^{1 + o(1)} = O(m^{1.0000\dots01})$이니 “almost linear time”입니다. Karger처럼 $m \cdot \mathrm{polylog}(n)$ 정도는 아니지만, 거의 선형 시간에 문제를 풀었으니 사실상 문제를 close했다고 보아도 좋을 정도입니다.

## Sketch in Rough

Karger에서는 모든 $\alpha$-min cut이 $H$에서도 $[1 - \varepsilon, 1 + \varepsilon]\alpha$-min cut이 되어야 한다는 강력한 조건이 필요했습니다. 즉, 모든 cut의 approximation ratio가 $1\pm \varepsilon$배 바깥으로 나가서는 안됐던 것이죠. 일단은 문제를 풀기에 충분한 더 약한 조건으로 분해하는 데는 성공했습니다. 이 조건을 만족하는 $H$를 만들기도 만만치 않지만, 이를 극복해낸 Li의 핵심 아이디어 둘을 소개합니다.

### Merge-in-the-middle

첫번째 착상은 “작은 cut”들을 위한 $H _ {\mathrm{small}}$과 “안 작은 cut”들을 위한 $H _ {\mathrm{big}}$을 만들어서 둘을 잘 접합시키는 전략입니다. “작은 cut”이 무엇인지는 전개될 문맥에 따라 계속 일반화되겠지만, **모든 min-cut들은 작은 cut이어야 한다는 사실은 동일합니다**. 여기서는 이해를 돕기 위해 모든 $\alpha$-min cut, 즉 $w(\partial S) \le \alpha \lambda$인 cut들을 “작은 cut”이라고 부르겠습니다. 실제와는 다르고 설명을 위한 예시일 뿐임에 주의하세요.

- $H _ {\mathrm{small}}$은 min-cut을 비롯한 “작은 cut”들을 근사하는 데 사용합니다. 때문에 $\varepsilon$-cap과 $\varepsilon$-support를 모두 만족하도록 **정성껏** 그래프를 만듭니다. 아예 Karger의 요구조건을 만족하도록 – 즉 모든 cut의 approximation ratio가 $1 \pm \varepsilon$ 배 안쪽에서 보존되도록 $H _ {\mathrm{small}}$을 만듭니다. 여기서 나온 $W$를 $W _ {\mathrm{small}}$이라고 둡시다.
- $H _ {\mathrm{big}}$은 신경을 비교적 덜 씁니다. $1 + \varepsilon$ 같은 섬세한 근사를 모든 cut에 적용하기란 여간 까다로운 일이 아니지만, 다행히 적당한 $\gamma = n^{o(1)}$에 대해 모든 cut의 approximation ratio가 $\gamma$배 이상 빗겨나가지 않도록 하는 deterministic approximator가 존재합니다 (Chuzhoy et al, 2019). 사실 $\gamma = 1 + \varepsilon$이 가능했다면 당장 Karger의 방법을 가져다 쓸 수 있으니, $\gamma$가 나쁜 값인 건 어쩔 수 없습니다. 일반적으로 이를 “Lossy unweighted sparsifier” 정도로 부릅니다. 여기서 얻은 $W$를 $W _ {\mathrm{big}}$으로 둡니다.
- 이제 $H _ {\mathrm{small}}$과 $H _ {\mathrm{big}}$을 말 그대로 “겹쳐” 놓는데… $H _ {\mathrm{big}}$이 얹어지면 $H _ {\mathrm{small}}$에서 기껏 정성들여 만들어 놓은 $\varepsilon$-cap property가 깨질 위험이 있습니다. 그러니 weight를 잘 섞어서, weighted graph 
  $H _ {\mathrm{mixed}} = W _ {\mathrm{small}} H _ {\mathrm{small}} + \frac{\varepsilon}{\gamma} \cdot W _ {\mathrm{big}}H _ {\mathrm{big}}$을 만듭시다.
  - $H _ {\mathrm{mixed}}$에서 (min-cut을 포함한) 작은 cut들은 $2\varepsilon$-cap이 보장됩니다. weight가 $w$인  $G$ 의 cut은 $(1 + \varepsilon)w + \frac{\varepsilon}{\gamma} (\gamma w) = (1 + 2\varepsilon)w$ 이하가 되기 때문입니다.
  - $H _ {\mathrm{mixed}}$에서 weight가 $w$인 “큰 cut”들은 최소한 $\frac{\varepsilon}{\gamma^{2}}w$ 이상의 weight를 갖게 됩니다. $w \ge \alpha \lambda$이고, hyperparameter인 $\alpha$를 $\gamma^{2} / \varepsilon$보다 충분히 크게 잡으면 큰 cut들은 $0$-support가 보장됩니다.
- 따라서 문제에 필요한 모든 조건이 만족되지만, 딱 하나 $H _ {\mathrm{mixed}}$가 “weighted” graph인 점이 걸립니다. parameter를 기술적으로 조정하면 $W _ {\mathrm{small}} / W _ {\mathrm{big}} = n^{o(1)}$인 정수가 되도록 할 수 있고, $W = W _ {\mathrm{big}}$으로 잡은 뒤 multi edge를 $W _ {\mathrm{small}} / W _ {\mathrm{big}}$개 정도 만들어주면 됩니다. 첨자 small/big과 $W _ {\mathrm{small}}, W _ {\mathrm{big}}$은 무관하다는 것에 주목해주세요.

### Mock Sampling: Pessimistic Estimator

이제 **정성껏** 작은 cut들의 approximation ratio를 $1 \pm \varepsilon$배 안쪽으로 유지하는 deterministic한 방법에 대해 대략적으로 알아봅시다. *Pessimistic Estimator*란 random sampling을 deterministic하게 흉내내면서, 사실은 남은 선택에 따라 approximation이 실패할 확률의 상한을 계산해주는 oracle을 말합니다. Li는 조건부 확률 개념의 일반화로 소개합니다.

- 아무 조건도 결정되지 않은 상태에서, pessimistic estimator $\Psi(\emptyset)$의 값은 $1$보다 작아야 합니다.
- 각각의 선택 상황에서 $\Psi$의 값을 증가시키지 않는 “선택”이 존재해야 합니다.
- 결과적으로 모든 선택이 끝나서 더 이상 결정할 것이 없을 때도 $\Psi(\text{All labels}) < 1$이 성립하게 됩니다. 이는 결정할 게 더 없는데 실패 확률이 $1$보다 작다는 소리고, 결국 실패 확률이 $0$이라는 뜻이니 approximation이 성공적으로 끝난다는 것을 의미합니다.

말은 좋지만, 수많은 “작은 cut”들이 $(1 \pm \varepsilon)$-sparsify될 수 있을지 따져보는 일은 쉬운 것이 아닙니다. 단순히 $\alpha$-min cut들이 “작은 cut”인 상황만 생각해도 최악의 경우 $n^{\Theta(\alpha)}$개의 cut이 존재할 수 있기 때문에, $\alpha = n^{o(1)}$이 필요한 상황에서 모든 cut을 무식하게 다 보는 방법으로는 Pessimistic estimator를 성공적으로 구현할 수 없습니다. 결국 모든 “작은 cut”들의 *structural representation*이 요구되고, Li는 그에 대한 해답으로 “작은 cut”들을 weight의 크기가 아니라 Expander Hierarchy의 시각에서 다시 정의할 것을 제안합니다.

# Inspecting unweighted $\phi$-expanders

Unweighted $\phi$-expander case는 다루고자 하는 경우와 완전히 같지는 않지만, 윗 문단에서 이야기한 핵심적인 아이디어들을 간소화해서 볼 수 있다는 장점이 있습니다. **Merge-in-the-middle**을 어떻게 하는지, **Pessimistic Estimator**가 어떤 값인지, “작은 cut” (=**unbalanced cut**)이 어떻게 정의되고 어떤 **structural representation**으로부터 얻어지는지를 생각하며 읽어주시면 좋을 것 같습니다.

**Central Theorem (Expander Case)**. Unweighted $\phi$-expander $G$에 대해, $m^{1 + o(1)}$ 시간에 모든 min-cut을 $\varepsilon$-cap하고, 모든 cut을 $\varepsilon$-support하고, $k = \lambda / W$인 unweighted graph $H$와 $W = \varepsilon^{3} \lambda / n^{o(1)}$을 계산할 수 있다.

## Graph Laplacian and unbalanced cuts

“작은 cut”들의 structural representation을 돕기 위한 도구는 다름아닌 Graph Laplacian $L _ {G}$입니다. 일반적으로 Expander graph와 Graph laplacian은 밀접한 관계가 있습니다. Cut과 관련된 다음의 유용한 항등식이 있습니다.

$w(\partial S) = 1 _ {S}^{T} L _ {G}1 _ {S} = \sum _ {u, v \in S} (L _ {G}) _ {uv}$

그래서 $L _ {G}$의 각 항을 근사할 수 있다면, 즉 $\left\lvert{(L _ {G}) _ {uv} - W \cdot (L _ {H}) _ {uv}}\right\rvert < \varepsilon' \lambda$ 가 되도록 근사할 수 있다면 $\left\lvert{w(\partial _ {G} S) - W \cdot w(\partial _ {H} S)}\right\rvert \le \left\lvert S \right\rvert^{2}\varepsilon'\lambda$가 되기 때문에 $W \cdot \partial _ {H} S$가 $\partial _ {G} S$를 $(1 + \left\lvert S\right\rvert^{2}\varepsilon’)$-approximate할 수 있게 됩니다. 따라서 $S$가 작거나, 반대로 $V - S$가 작다면 충분히 “작은 cut”이라고 부름직합니다. $\partial _ {G} S = \partial _ {G} (V - S)$이니까요.

**Definition.** Parameter $\alpha = n^{o(1)}$에 대해, $\min\lbrace \mathrm{vol}(S), \mathrm{vol}(V - S)\rbrace \le \alpha\lambda / \phi$이면 $S$를 **unbalanced cut**이라고 한다.

**Proposition 1.** $\mathrm{vol}(S) \le \alpha\lambda / \phi$이면 $\left\lvert S \right\rvert \le \frac{\alpha}{\phi}$이다.
$\frac{\alpha\lambda}{\phi} \ge \mathrm{vol}(S) \ge \delta \cdot \left\lvert S \right\rvert \ge \lambda \cdot \left\lvert S \right\rvert {\square}$

**Proposition 2 (all min-cuts are small).** $\alpha \ge 1$에 대해, **모든 min-cut은 unbalanced.**

$\phi \le \frac{w(\partial _ {G}S^{*})}{\mathrm{vol}(S^{*})} = \frac{\lambda}{\mathrm{vol}(S^{*})} \implies \mathrm{vol}(S^{*}) \le \frac{\lambda}{\phi} \le \frac{\alpha\lambda}{\phi}. \square$

굳이 $\left\lvert S \right\rvert$가 아니라 $\mathrm{vol}(S)$가 작다는 강한 조건을 걸어준 이유는 approximation bound를 잘 잡아주기 위함도 있고, 이후에 일반적인 케이스에서 unbalanced cut을 정의하는 방법과 호환되기 때문도 있습니다. Unbalanced cut $S$와 $u, v \in S$에 대해 $u = v$라면 $(L _ {G}) _ {uu} = \deg(u) \le \mathrm{vol}(S)$, $u \neq v$라면 $\left\lvert(L _ {G}) _ {uv}\right\rvert = w(u, v) \le \mathrm{vol}(S)$이기 때문에 결국 $\left\lvert (L _ {G}) _ {uv} \right\rvert \le \frac{\alpha\lambda}{\phi}$라는 추가적인 함의를 갖고 있습니다.

결과적으로 $(L _ {G}) _ {uv}$를 $W \cdot (L _ {H}) _ {uv}$가 additive error 기준 $(\phi / \alpha)^{2} \varepsilon \lambda$ 안쪽으로 근사할 수 있으면 될 것 같습니다. 그럼 최소한 작은 cut들을 근사하는 $H _ {\mathrm{small}}$을 만들 수 있겠죠. 일단은 편의상 $H$로 지칭하겠습니다. 이제 Sampling 확률 $p = \Theta(\frac{\alpha \log n}{\varepsilon^{2} \phi \lambda})$를 고정하고, weight $W = 1 / p$로 둡시다.

$u \neq v$를 고정하고, 이제 각 간선에 대해, $k = w(u, v)$개의 random variable $x _ {1}, \cdots, x _ {k}$를 잡아봅시다. $x _ {i}$는 $k$개의 중복 간선 중 $i$번째 간선이 $H$에 있으면 $1$, 없으면 $0$이 되는 변수입니다. 당연히 $\Pr[x _ {i} = 1] = p$이겠죠. $X = x _ {1} + \cdots + x _ {k}$로 두면 $X$가 $w _ {H}(u, v)$를 나타내는 random variable이 됩니다. 여기서 수학을 좀 사용해서 Pessimistic estimator를 이끌어봅시다.

---

**Theorem. (Chernoff’s bound)**

상수 $a$, random variable $X$, $t > 0$에 대해

- $\Pr[X > a] = \Pr[e^{tX} > e^{ta}] \le e^{-ta}\mathbb{E}[e^{tX}]$.
- $\Pr[X < a] = \Pr[e^{-tX} > e^{-ta}] \le e^{ta} \mathbb{E}[e^{-tX}]$.

위 부등식들은 Markov’s inequality의 당연한 따름정리로, 이 결과를 종합하면 다음의 loose bound를 얻을 수 있다. $\mu = \mathbb{E}[X]$와 $0 < \delta \le 1$에 대해,

- $\Pr[\left\lvert{X - \mu}\right\rvert > \delta \mu] < 2e^{-\delta^{2} \mu / 3}$.

---

$\delta = \varepsilon\phi / 2\alpha$, $\mu = p\lvert L _ {G}\rvert _ {uv}$에 대해

$\Pr[\left\lvert (L _ {H}) _ {uv} - p (L _ {G}) _ {uv} \right\rvert > \delta \mu] \le e^{-\ln(1 + \delta)(1 + \delta)\mu} \mathbb{E}[e^{\ln(1 + \delta)X}] + e^{\ln(1-\delta)(1-\delta)\mu}\mathbb{E}[e^{-\ln(1-\delta)X}] \le 2e^{-\delta^{2} \mu / 3}.$

이 때 $\delta^{2} \mu = \Theta(\lvert (L _ {G}) _ {uv} \rvert \cdot \frac{\phi\log n}{\alpha\lambda})$이고, 따라서 위 식의 값은 $2\exp(-\Theta(\log n))$으로 bound됩니다. $p$ 값을 상수 배 범위 안에서 잘 조정하면 충분히 모든 $u, v$에 대해 위 식의 우변, 즉

$\Psi _ {u,v}(X) = e^{-\ln (1 + \delta)(1 + \delta) p \lvert L _ {G} \rvert _ {uv}} \mathbb{E}[e^{\ln(1 + \delta)X}] + e^{\ln (1 - \delta)(1 - \delta) p \lvert L _ {G} \rvert _ {uv}} \mathbb{E}[e^{-\ln(1 - \delta)X}]$

을 모두 합한 값인 $\Psi = \sum _ {u, v} \Psi _ {u, v}$가 $1$보다 작도록 할 수 있습니다. 또 모든 간선에 대해 $x _ {i}$들은 독립적이므로, $\mathbb{E}[e^{tX}] = \prod _ {i = 1}^{k} \mathbb{E}[e^{tx _ {i}}]$가 성립합니다. 따라서 $\Psi$는 총 $m$개의 random variable이 들어간 식의 곱 형태로 표현됩니다.

이 때 기댓값의 선형성에 의해, 어떤 간선 $x _ {i}$의 값으로 가능한 둘 중 하나는 $\Psi$ 값을 증가시키지 않습니다. 어떤 간선을 사용할지 사용하지 않을지 결정한다는 것은, 사실상 그 변수에 대해서만 확률을 $1$ 또는 $0$으로 바꾸는 행위와 동치입니다. 따라서 $\Psi$값이 어떻게 변화하는지 상수 시간에 계산할 수 있고, $\Psi$값이 증가하지 않는 방향으로 간선을 sampling할 수 있습니다.

마지막에 가서는 결국 $\Psi < 1$이 될 테고, 모든 $u, v$에 대해 $\left\lvert (L _ {H}) _ {uv} - p (L _ {G}) _ {uv} \right\rvert \le \delta \mu$가 성립하게 됩니다. 그말인즉슨 $\lvert (L _ {G}) _ {uv} - W(L _ {H}) _ {uv}\rvert \le \lvert L _ {G} \rvert _ {uv} \cdot \delta \le \varepsilon\lambda$가 성립하게 됩니다. Cut에 대한 오차는 $\left\lvert w(\partial _ {G}S) - Ww(\partial _ {H}S) \right\rvert \le \delta \cdot \sum _ {u, v \in S} \lvert L _ {G} \rvert _ {uv} \le \delta \cdot 2\mathrm{vol}(S) \le \varepsilon\lambda$가 성립함을 알 수 있습니다. unbalanced cut을 모두 근사하는 데 성공했네요!

## $\gamma$-approximation for balanced cuts

$\min\lbrace \mathrm{vol}(S), \mathrm{vol}(V-S)\rbrace \ge \alpha\lambda / \phi$이면 $S$를 balanced cut이라고 부릅니다. Balanced cut을 적당히 근사하기 위해, $\deg _ {K}(v) = \Theta(\deg _ {G}(v) / \lambda)$인 $\Theta(1)$-expander $K$를 잡읍시다. $K$를 어떻게 잡는지는 이 글에서 관심 대상이 아니고, 일반적인 경우에 너무 복잡하기 때문에 다루지 않으려 합니다. 일단 그런 그래프 $K$를 잡을 수 있다고 믿고 잡아봅시다. 그렇다면 모든 balanced cut에 대해

$w(\partial _ {K}S) \ge \Theta(1) \min\lbrace \mathrm{vol} _ {K}(S), \mathrm{vol} _ {K}(V-S)\rbrace = \Theta(1 / \lambda)\min\lbrace \mathrm{vol} _ {G}(S), \mathrm{vol} _ {G}(V-S)\rbrace = \Theta(\alpha / \phi)$가 됩니다.

## Merge-in-the-middle

이제 $H$와 $K$를 겹쳐서 새로운 weighted graph $H’$를 만들어야 합니다. $H’$는 $WH + W’K$로 만들어야 하는데, 일단 $W’ = \Theta(\varepsilon \phi \lambda)$로 둬 봅시다. 이렇게 그래프를 합칠 경우에, 두 가지 사실이 보장됩니다.

- 모든 **min cut.** 들은 $K$에 대해 크게 영향을 받지 않는다.

  $w(\partial _ {H’} S^{\ast}) = Ww(\partial _ {H} S^{\ast}) + W'w(\partial _ {K} S^{\ast}) \le (1 + \varepsilon)\lambda + W' \cdot \mathrm{vol} _ {K}(S^{\ast}) \le (1 + \varepsilon)\lambda + \Theta(\varepsilon\phi)  \cdot \mathrm{vol} _ {G}(S^{\ast}) \le (1 + \varepsilon + \Theta(\varepsilon))\lambda.$

  이로부터 $O(\varepsilon)$-cap이 여전히 유지되는 것을 확인할 수 있습니다.

- Balanced cut은 $H’$에서 최소한 $\lambda$ 이상의 weight를 갖는다.

  $w(\partial _ {K} S) \ge \Theta(\alpha / \phi)$이므로, $\alpha = \Theta(1 / \varepsilon)$으로 두고 $W’ \cdot K$를 고려하면 최소한 $\lambda$ 이상의 weight를 갖게 됩니다.

따라서 $W’ = \Theta(\varepsilon \phi \lambda)$로 둔 것이 적절했음을 확인할 수 있고, $O(\varepsilon)$-cap 및 $\varepsilon$-support가 깨지지 않도록 두 그래프를 잘 섞을 수 있습니다. 하지만 여전히 $H$가 weighted graph인 것이 불만인데요, $W = \Theta(\varepsilon^{2} \phi \lambda / \alpha \log n) = \Theta(\varepsilon^{3} \phi \lambda / \log n)$이고, $W’ = \Theta(\varepsilon \phi \lambda)$이므로 $W / W’ = \Theta(\varepsilon^{2} / \log n)$으로 상수 개가 됩니다. 따라서 적당히 $W$가 $W’$의 정수배가 되도록 하고, 그만큼 multi edge를 만들어주면 $H’ / W’$을 unweighted graph로 만들 수 있습니다.

# General Case

Expander case보다도 기술적인 색채가 강한 일반적인 경우를 이 글에서 모두 소개하기에는 지면에 무리가 있습니다. 따라서 일반적인 경우는 정성적인 기술 위주로 매듭짓도록 하겠습니다.

당장 일반적인 그래프에서는 Expander Case에서 사용한 많은 가정들을 쓸 수 없기 때문에, **분할 정복**을 통해서 그래프에서 많은 Expander들을 찾고, 그 Expander들을 한 정점으로 뭉쳐서 간선이 꽤 적은 새로운 그래프로 recursion을 하는 것이 일반적입니다. 이 분할 정복 과정을 Expander Decomposition, 또는 Expander Hierarchy로 부릅니다. Spielman-Teng (2004) 이후로 Expander Decomposition과 관련된 framework는 비약적으로 발전해왔고, 이 논문에서 사용하는 Expander decomposition의 변주도 최신 논문 (Goranci, 2020)에서 개발된 것입니다.

**Theorem. (Boundary Linked Expander Decomposition, BLED)**

Graph $G$와 parameter $r \ge 1$, $\beta \le (\log n)^{-O(r^4)}$, $\phi \le \beta$에 대해 $O(m^{1 + O(1/r)}) + \tilde{O}(m / \phi^2)$ 시간 안에 $G$의 정점 집합 $V$를 아래 조건을 만족하는 $V _ {1} \sqcup \cdots \sqcup V _ {k}$로 분할할 수 있다.

- $G[V _ {i}]$는 $\beta$-boundary-linked $\phi$-expander이다. 이 때 boundary-linked expander란 $V _ {i}$에서 $V _ {i}$ 바깥으로 연결되어 있는 모든 간선을 weight가 $\beta / \phi$인 **self loop**로 대체한 그래프를 말한다. 즉,


  
  $  \min _ {S} \frac{w(\partial _ {G}S)}{\min(\mathrm{vol} _ {G[V _ {i}]}(S) + \frac{\beta}{\phi}w(E(S, V - V _ {i}),\mathrm{vol} _ {G[V _ {i}]}(V _ {i} -S) + \frac{\beta}{\phi}w(E(V _ {i} - S, V - V _ {i}))} \ge \phi.$

  편의상 $\mathrm{vol} _ {G[V _ {i}]}(S) + \frac{\beta}{\phi}w(E(S, V - V _ {i})$가 더 작은 것으로 가정한다. 이는 두 가지 식이 동시에 성립하는 것을 의미한다.

  $\frac{w(\partial _ {G} S)}{\mathrm{vol} _ {G[V _ {i}]}(S)} \ge \phi \quad\wedge\quad \frac{w(\partial _ {G} S)}{w(E(S, V - V _ {i}))} \ge \beta $

  즉, $G[V _ {i}]$가 $\phi$-expander일 뿐만 아니라, cutset의 크기가 cluster를 넘어가는 간선에 비해서도 결코 작지 않다.

- $V _ {i}$들 사이에는 간선이 많지 않다. 즉, $w(\partial V _ {1} \cup \cdots \cup \partial V _ {k}) = (\log n)^{O(r^4)} \cdot \phi \cdot \mathrm{vol}(V)$으로, 분할 정복을 하기에 나쁘지 않은 조건 정도로 이해할 수 있다.

**Definition. (Canonical decomposition sequence)**

$G = G^{0}, \cdots, G^{L}$을 $G$의 BLED 과정이라고 하자. 이 때, $S = S^{0}, \cdots, S^{L}$을 아래와 같은 방법으로 정의한다.

$V(G^{i}) =: U^{i} = U _ {1}^{i} \sqcup \cdots \sqcup U _ {k}^{i}$에 대해 $U _ {j}^{i}$를 한 정점 $u _ {j}$로 contract해서 $U^{i+1}$이 되는 것은 정의상 당연하다. 이 때, $u _ {j} \in S^{i+1}$인지에 결부되어 있는 $D _ {j}^{i} \subseteq U _ {j}^{i}$를 $u _ {j} \in S^{i+1} \iff D _ {j}^{i} = U _ {j}^{i} - S^{i}$, $u _ {j} \notin S^{i+1} \iff D _ {j}^{i} = U _ {j}^{i} \cap S^{i}$가 되도록 정의하자. 즉 $S$가 $G^{i}$에 “두고 가야만 하는 정점들”의 집합이 $D _ {j}^{i}$라고 생각할 수 있다. 이 때, $D _ {j}^{i}$를 다음을 보장하는 쪽으로 선택해서 얻은 $\lbrace  S^{i}\rbrace$를 $S$의 canonical decomposition sequence라고 한다.
$\mathrm{vol} _ {G^{i}[U^{i} _ {j}]}(D _ {j}^{i}) + \frac{\beta}{\phi} w(E(D _ {j}^{i}, U^{i} - U _ {j}^{i})) \le \mathrm{vol} _ {G^{i}[U _ {j}^{i}]}(U^{i} - D _ {j}^{i}) + \frac{\beta}{\phi}w(E(U^{i} - D _ {j}^{i}, U^{i} - U _ {j}^{i}))$

즉 $\beta$-boundary linked $\phi$-expander의 이로운 성질을 $D _ {j}^{i}$ 쪽에 남겨두도록 $D _ {j}^{i}$를 고른다는 의미가 됩니다. 복잡한 정의를 구태여 소개한 것은, $D _ {j}^{i}$를 이용해서 unbalanced/balanced cut을 나누고 unbalanced cut을 근사하게 될 것이기 때문입니다. 중요한 사실들을 몇 가지 건조하게 나열해보겠습니다.

**Proposition.** 모든 $S$에 대해,

- $\partial _ {G} S \subseteq \bigcup _ {i =0}^{L} \bigcup _ {j} \partial _ {G^{i}} D _ {j}^{i}$.
- $\sum _ {i = 0}^{L}\sum _ {j} w(\partial _ {G^i} D _ {j}^{i}) \le \beta^{-O(L)} w(\partial _ {G} S)$.

**Definition. (General unbalanced cut)** 

Parameter $\tau$에 대해, $\sum _ {i}\sum _ {j} \mathrm{vol} _ {G^i} (D _ {j}^{i}) \le \frac{\tau \lambda}{\phi}$이면 $S$를 $\tau$-unbalanced cut이라고 한다.

**Proposition (All mincuts are small)** $\tau = \beta^{-\Omega(L)}$로 충분히 크면, 모든 min-cut은 $\tau$-unbalanced.

*Proof.* $\sum _ {i, j}\mathrm{vol} _ {G^i}(D _ {j}^{i}) \le \frac{1}{\phi}\sum _ {i, j} w(\partial _ {G^i} D _ {j}^{i}) \le \frac{\beta^{-O(L)}}{\phi}w(\partial _ {G}S^{\ast}) \le \frac{\tau \lambda}{\phi}$.

**Proposition (Laplacian to cuts)**

$u \in U^{i}$에 대해, $V$의 정점들 중 $u$로 압축되어 들어가는 정점들의 집합을 $\overline{u}$로 표기하자. 이 때, $D _ {j}^{i}$의 정의에 따라 모든 $S$에 대해

$w(\partial _ {G} S) = \sum _ {i, j, k, l} \sum _ {u \in D _ {j}^{i}, v \in D _ {l}^{k}} \pm 1 _ {\overline{u}} L _ {G}1 _ {\overline{v}}$

단, $1 _ {\overline{x}}$는 $x \in \overline{x}$인 원소에 대해서만 $1$, 나머지는 $0$인 벡터.

따라서 Expander hierarchy의 형태로부터 cut의 Laplacian representation을 얻을 수 있습니다. 또한 각각의 $D _ {j}^{i}$가 좋은 성질을 가졌다는 것으로부터 위 식의 각 항을 $H$가 근사하는 방법에 대한 pessimistic estimator를 자연스럽게 고안할 수 있고, derandomized된 방법으로 unbalanced cut을 근사할 수 있습니다. 물론 매 스텝마다 $D _ {j}^{i}$에 대한 성질을 강하게 사용해야 하고, Matula (1993) 등을 이용해 min cut의 $3$-approximation을 필요로 하는 등 복잡다단한 과정을 거쳐야 합니다.

Balanced cut의 경우에도 훨씬 복잡한 방법을 써서 $H _ {\mathrm{small}}$과 $H _ {\mathrm{big}}$을 모두 구한 뒤, Merge-in-the-middle의 과정에서 $r, \beta, \phi, \tau$등의 parameter들을 모두 resolve하면 맨 처음의 수상한 복잡도를 얻을 수 있습니다. 이 복잡도가 $m^{1 + o(1)}$이라는 사실에 감탄하는 시간을 가져보도록 합시다.

$m \cdot 2^{O(\log n)^{5/6}\log\log^{O(1)}n}$

이로써 min-cut은 deterministic almost-linear time에 해결되었지만, 여전히 practical하게 쓰이기에는 어려움이 있다는 사실까지 함께 알 수 있습니다.

# Further Reading

- Jason Li (2021, not printed), Deterministic Mincut in Almost-Linear Time
- Julia Chuzhoy, Yu Gao, Jason Li, Danupon Nanongkai, Richard Peng, and Thatchaphol Saranurak (2019). A deterministic algorithm for balanced cut with applications to dynamic connectivity, flows, and beyond – 약칭 **CGL+19**입니다. 
- Gramoz Goranci, Harald Racke, Thatchaphol Saranurak, and Zihan Tan (2020). The expander hierarchy and its applications to dynamic graph algorithms. – 약칭 **GRST20**. BLED 등의 framework를 고안한 논문으로 알려져 있습니다.
- Ken-ichi Kawarabayashi and Mikkel Thorup (2015, STOC 2018). Deterministic edge connectivity in near-linear time.
- Monika Henzinger, Satish Rao, and Di Wang. Local flow partitioning for faster edge connectivity. (SODA 2017) – 위 논문을 최적화한 논문입니다. deterministic min-cut에 대한 SOTA를 지키고 있었습니다.
- Thatchaphol Saranurak. A simple deterministic algorithm for edge connectivity (SOSA 2021) – Expander decomposition에 대해 더 공부한 뒤에 읽어봄직합니다.
