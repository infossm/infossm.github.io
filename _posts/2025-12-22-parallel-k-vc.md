---
layout: post
title: "Parallel Small Vertex Connectivity In Near-Linear Work and Polylogarithmic Depth"
date: 2025-12-22
author: TAMREF
tags: [algorithm, graph theory, parallel]
---

## Introduction

이 글에서는 제가 Yonggang Jiang과의 공동 연구로 작성한 [논문](https://arxiv.org/abs/2504.06033v1)에서 다룬 문제인 k-vertex connectivity의 효율적인 병렬 알고리즘에 대해 설명합니다. 최근 유명한 그래프 문제들을 병렬/분산 환경에서 해결하는 연구에 많은 진전이 있었는데, 이 글을 통해 다양한 계산 모델에서 효율적인 알고리즘을 설계할 때 어떤 관점이 유용한지 제 나름의 직관을 풀어보고자 합니다.

## Preliminary

무방향, 무가중치 그래프 $G$가 주어졌을 때, $V(G)$의 분할 $(L, S, R)$이 $L, R \neq \emptyset$이고, $L$과 $R$ 사이에 간선이 없으면 이를 Vertex Cut이라고 합니다. 이때 가능한 $S$의 크기의 최솟값을 **Vertex Connectivity**라고 합니다.

특정한 정점 $s, t \in V(G)$에 대해 $s \in L, t \in R$ 인 경우 $(L, S, R)$을 $s, t$-vertex cut이라고 하고, 비슷하게 가능한 $\lvert S \rvert$의 최솟값을 **$(s, t)$-vertex connectivity**라고 합니다.

즉 $G$를 disconnect할 수 있는 최소 크기의 정점 집합을 찾는 문제이고, 대상을 **간선 집합**으로 바꾸면 edge connectivity, 혹은 **min cut**이라고 부릅니다.

이런 문제들을 결정 문제로 바꾼 것을 $k$-(vertex) connectivity라고 부릅니다. 즉, $\lvert S \rvert < k$인 vertex cut $S$가 존재하는지를 묻는 문제고, 그렇지 않다면 그래프가 $k$-connected라고 합니다. 이름에서 알 수 있듯이 $k = 1$인 경우에는 그래프가 연결되어 있는지 묻는 문제가 되고, $k = 2$인 경우에는 단절점을 찾거나 그래프가 biconnected인지 묻는 문제가 됩니다.

여기서 병렬 알고리즘의 “Parallel”은 일반적으로 PRAM 계산 모델을 이야기합니다. 이 글에서는 shared memory에 많은 수의 프로세서가 동시에 읽고, 동시에 쓸 수 있는 CRCW 모델을 염두에 두겠습니다. 다만 어떤 세부 모델이냐보다는 이러한 모델에서 효율성을 어떻게 측정하는지가 핵심입니다.

알고리즘 전체가 사용하는 연산량의 총합을 Work이라고 하고, 알고리즘이 terminate하는 데까지 걸리는 시간을 Depth(Span)라고 합니다. 초기 논문들에서는 “$P$개의 processor로 $T$만큼의 시간이 걸린다”라고 쓰기도 하는데, 이 경우 Work는 $O(PT)$, Depth는 $T$가 됩니다. 계산 과정과 그에 따른 의존성을 하나의 DAG로 나타낼 수 있을 텐데, Depth는 이 DAG의 longest path 길이, Work는 모든 노드가 사용하는 계산량의 합으로 생각할 수 있습니다.

따라서 우리가 풀고자 하는 문제는 “Work과 Depth가 모두 작은 병렬 $k$-vertex connectivity 알고리즘이 존재하는가?”가 되겠습니다.

## Background

### Sequential Approaches

이러한 문제 자체는 너무나 유명하고, 예전부터 엄청나게 연구되었습니다. 그래프의 정점이 $n$개, 간선이 $m$개라고 하고, 표기부터 정리하겠습니다. 구체적인 복잡도가 큰 의미가 없는 경우에는

- $O(m \log^{O(1)} n)$ 을 $\widetilde{O}(m)$으로 표기하고, near-linear라고 부르겠습니다.
- $m^{1 + o(1)}$을 $\widehat{O}(m)$으로 표기하고, almost-linear라고 부르겠습니다.

Edge Min Cut은 Karger (2000)에 의해 이미 $O(m \log^{3} n)$ 랜덤 알고리즘이 알려졌습니다. 예전에 제가 [다른 포스트](https://infossm.github.io/blog/2021/03/21/minimum-cuts-in-near-linear-time/)를 통해 소개한 적이 있습니다. Gawrychowski (2019)에 의해 $O(m \log^{2} n)$ 으로 개선되었고, 결정적인 알고리즘은 위 글에서 이야기했듯이 가중치가 없는 경우 Kawarabayashi-Thorup (2014)에 의해 near-linear, 가중치가 있는 경우에도 Li (2021)에 의해 almost-linear 시간에 해결되었습니다.

Vertex Cut의 경우에는 조금 더 어려웠지만, Li et al. (2021)에 의해 maximum flow를 $\widetilde{O}(1)$ 번 사용하면 랜덤 알고리즘을 얻을 수 있다는 것이 알려졌습니다. Chen et al. (2022)에 의해 maximum flow가 almost-linear time에 해결되면서, 자연스럽게 almost-linear time 알고리즘을 갖게 되었습니다. 다만 결정론적인 알고리즘에 대해서는 아직 알려진 바가 없습니다.

### Parallelism

두 문제 모두 maximum flow를 이용하면 해결할 수 있지만, vertex cut의 경우 flow-independent한 near-linear algorithm이 아직 없습니다. 또한 maximum flow 알고리즘을 병렬화하는 것은 생각보다 훨씬 어려워서, 당장 그보다 훨씬 쉬운 “방향 그래프에서 $s$에서 $t$로 가는 경로가 있느냐?”라는 $(s, t)$-reachability조차 highly parallel(예: almost-linear work, subpolynomial depth) 알고리즘이 없는 실정입니다.

Edge Connectivity의 경우에는 다양한 방법이 있어서 highly-parallel 병렬 알고리즘이나 cut query model (Mukhopadhyay & Nanongkai, 2020), 그리고 분산 알고리즘 (Ghaffari & Zuzic, 2022)에서 효율적인 알고리즘들이 고안되었습니다.

### k-Connectivity

그냥 Vertex Connectivity는 너무 어려우니까, $k$-connectivity를 생각해 봅시다. $k = 1, 2, 3$인 경우에는 connected, biconnected, triconnected component들에 대해 많은 연구가 되어 있어서 linear work / $O(\log n)$ depth 알고리즘들이 80년대에 알려져 있었습니다. 다만 $k = 4$ 이상부터는 linear time 알고리즘이 존재하지 않습니다.

일반적인 $k$에 대해서는 주로 Menger's theorem을 이용한 병렬 알고리즘들이 있었는데, 즉 $k$-vertex cut을 찾거나 끝점만 공유하는 $k$개의 경로를 찾는 접근입니다. 구체적으로, 그래프에서 어떤 $\{v _ 1, \cdots, v _ k\}$를 효율적으로 찾을 수 있는데, 모든 $u \in V$에 대해서 $v_i$와 $u$를 연결하는 끝점만 공유하는 $k$개의 서로 다른 경로를 찾을 수 있다면 전체 그래프가 $k$-connected임을 보일 수 있습니다.

그래서 $(s, t)$-$k$-connectivity를 총 $kn$번 해결하면 $k$-connectivity를 해결할 수 있고, $(s, t)$-$k$-connectivity 자체는 $O(k^{2}\log n)$ depth, $O(k^{2}n \log n)$ work으로 해결할 수 있어서 전체 문제를 약 $\widetilde{O}(mk^{2} + n^{2}k^{4})$ work, $O(k^{2}\log n)$ depth에 해결할 수 있습니다. 그래서 dense graph, $k = \widetilde{O}(1)$에 대해서는 highly parallel algorithm이 있는 셈입니다.

다만 sparse graph에 대해서는 여전히 병렬 알고리즘에서 결과가 없었고, 그러니 만약 이 문제를 해결할 수 있다면 유의미한 성과를 낼 수 있는 셈입니다.

“일반적인 $k$에 대해 $k$-vertex connectivity를 $\widetilde{O}(mk^{O(1)})$ work, $\widetilde{O}(k^{O(1)})$ depth에 해결하는 알고리즘이 존재하는가?”

Sequential algorithm에서는 비슷한 류의 결과가 있었는데, Saranurak & Yingchareonthawornchai (2022)의 경우 $mk^{O(k^2)}$의 (fixed $k$에 대해) “near-linear” algorithm이 있었고, Li et al. (2021)에서 $O(mk^{2})$ 알고리즘을 고안했습니다. Korhonen (2025)는 tree decomposition의 관점에서 $O(m \cdot \exp(k^{O(1)}))$ 알고리즘을 고안하기도 했습니다.

일단 $k$에 대해 exponential한 알고리즘들은 너무 어려우니 배제하고, Li et al. (2021)의 알고리즘은 max flow 전체를 subroutine으로 갖지 않는다는 점에서 더 좋습니다. 이 알고리즘에선 “Local Vertex Cut”이라는 접근을 소개하는데, 저희 연구도 이 접근에 기초해서 다음의 결과를 얻어냈습니다.

**Theorem.** (Jiang & Yun, 2025) $k$-vertex connectivity를 $\widetilde{O}(mk^{12})$ work, $\widetilde{O}(k^3)$ depth에 해결하는 랜덤 알고리즘이 존재한다.

## Algorithm

### Local Vertex Cut

Local Vertex Cut은 small vertex cut을 처음으로 subquadratic하게 해결한 방법인데, global vertex cut을 local하게 구한다는 게 직관적으로 잘 와닿지 않을 수 있습니다. 이를 이해하기 위해서는 global vertex cut은 단순하게 생각하면 $(s, t)$ vertex cut을 $O(n^2)$번 구하면 정확하게 구할 수 있다는 사실에서 출발하면 됩니다.

균일한 확률로 간선을 하나 뽑고, 그 두 끝점 중 하나를 역시 랜덤으로 고르면 한 정점이 이 시행에서 선택될 확률은 $\deg(v) / 2m$입니다. (이 과정을 독립적으로 두 번 반복해 $(s, t)$를 뽑는다고 생각하면 됩니다.) 따라서 어떤 vertex cut $(L, S, R)$에 대해 랜덤으로 고른 $(s, t)$가 각각 $L, R$에 들어가서 $(s, t)$ vertex cut이 $(L, S, R)$을 capture하게 될 확률은 $\deg(L)\deg(R) / 4m^{2}$가 됩니다. 여기서 $\deg(X) := \sum _ {v \in X} \deg(v)$를 의미합니다.

일반성을 잃지 않고 $\deg(L) \le \deg(R) \le k^{O(1)} \cdot \deg(L)$이라고 하면, $\lvert S \rvert = n^{o(1)}$라고 할 때 위의 확률은 대략 $n^{-o(1)}/k^{O(1)}$ 꼴로 볼 수 있습니다. 따라서 $\widetilde{O}(k^{O(1)})$번 해당 시행을 반복하면 높은 확률로 $(L, S, R)$을 capture할 수 있습니다. 이런 경우를 대개 balanced cut이라고 합니다.

Local Vertex Cut은 이 시행으로 capture하기 어려운 경우를 다룹니다. 일반성을 잃지 않고 어떤 $\mu \ll m$에 대해 $\deg(L) \in [\mu, 2\mu]$라고 합시다. 언급한 것처럼 정점을 뽑는 시행을 하면 $\mu / 2m$의 확률로 $L$의 정점을 하나 고르게 되니, 약 $\widetilde{O}(m/\mu)$개의 정점을 뽑으면 그 중에 하나는 $L$에 있을 확률이 높습니다.

이제 뽑은 정점 $x$에 대해, 둘 중 하나를 출력하는 알고리즘을 $\widetilde{O}(\mu \cdot k^{O(1)})$ work, $\widetilde{O}(k^{O(1)})$ depth에 구현할 수 있다고 해 봅시다.

- Vertex cut $(L, S, R)$: $x \in L$, $\deg(L) \le 2\mu$.
- `Fail`: 그런 vertex cut이 존재하지 않는다.

그렇다면 한 스케일 $\mu$에 대해 $\widetilde{O}(m/\mu)$번의 샘플링을 통해 (높은 확률로) 필요한 경우를 커버할 수 있고, 각 샘플을 처리하는 비용이 $\widetilde{O}(\mu \cdot k^{O(1)})$이므로 전체 work는
$\widetilde{O}(m/\mu \cdot \mu \cdot k^{O(1)}) = \widetilde{O}(mk^{O(1)})$가 됩니다. 그리고 $\mu = 2^{i}$에 대해 알고리즘을 반복하면 모든 케이스를 다 처리할 수 있습니다. 이처럼 “나를 포함하는 작은 vertex cut”을 빠르게 찾는다는 점에서 local vertex cut이라고 부릅니다.

### Fractional Cut Approach

Li et al.에서는 간단한 DFS 기반 루틴을 통해 $O(\mu k^{2})$ 시간에 해당 문제를 해결하는 알고리즘을 찾아냈지만, 이는 필연적으로 reachability를 사용하기 때문에 효율적인 병렬화가 어렵습니다. 대신 저희 접근은 큰 틀은 공유하지만 병렬화 가능한 루틴들을 많이 사용합니다.

$w : V(G) \to \mathbb{R}^{+}$와 $s, t \in V(G)$, $c \ge 0, K > 0$에 대해, $w$가 다음 조건을 만족하면 $(s, t)$ fractional cut of value $c$라고 합니다.

1. $w(s) = w(t) = 0$.
2. $\sum _ {v \in V(G)} w(v) = K$.
3. 모든 $(s, t)$ path $P$에 대해, $w(P) := \sum_{v \in V(P)} w(v) \ge cK$.

즉, $(s, t)$ 사이의 최단경로 길이가 $cK$ 이상이 되도록 하는 vertex weight assignment를 fractional cut으로 정의합니다. vertex cut $(L, S, R)$이 주어졌을 때, 모든 $v \in S$에 대해 $w(v)=1/\lvert S \rvert$를 주고 나머지는 0으로 두면 $w$는 value $1/\lvert S \rvert$의 fractional cut이 됩니다. 반대로, value $c$인 fractional cut $w$가 주어졌을 때 크기 $\lfloor 1/c \rfloor$의 vertex cut을 (병렬적으로) 복구할 수 있다는 사실을 사용할 수 있습니다.

보통은 2번에서 $K=1$로 두지만, 여기서는 매번 normalization을 명시하기 번거로워서 $K$를 남겨 둔 것입니다. 앞으로 표기하는 모든 “weight”는 다 $K$로 나누어져 있다고 상정합니다.

그래서 local vertex cut에서 vertex cut 대신 fractional cut을 찾기로 해도 무방합니다. 그러면 문제는 “fractional cut을 어떻게 찾을 것인가?”가 됩니다.

### Relaxed Local Vertex Cut

Local Vertex Cut의 언어에 맞춰 목표를 조정하면, 둘 중 하나를 얻어야 합니다.

- 내가 뽑은 $x$와 어떤 $y$에 대해 value가 $1/(k - 0.5)$ 이상이고, $O(\mathrm{poly}(k)\cdot \mu)$ “크기”를 갖는 fractional $(x, y)$-cut
- `Fail`: 그런 cut이 없다는 확신

여기서 “없다는 확신”은 보통 다음 형태로 구현합니다.

“우리가 기를 쓰고 $w$를 바꿔봐도, 충분히 많은(예: $\mu \cdot \mathrm{poly}(k)$ 정도의 volume을 가진) 정점들에 $1/(k-0.5)$보다 낮은 cost로 도달할 수 있더라.”

여기서 “기를 쓰고 $w$를 바꿔봐도”라는 부분은 MWU(multiplicative weight update)가 보장하고, “충분히 많은…” 부분은 $x$에서 시작해서 충분히 많은 점으로 가는 shortest path tree 또는 (근사) 거리 정보를 제공하면 됩니다. 만약 $\deg(L) \le 2\mu$인 작은 vertex cut이 있다면, 이전에 언급한 것처럼 $S$에 균일 가중치를 주는 방식으로는 절대 작은 cost로 $\deg(L)$보다 큰 volume을 넘어서는 쪽으로 도달할 수 없습니다.

Shortest path는 reachability보다 어려워 보일 수 있지만, 무방향 그래프에서는 (근사) single-source shortest path tree를 work-efficient하게 병렬로 구하는 것이 알려져 있습니다. 예를 들어 다음 형태의 결과를 사용할 수 있습니다.

**Theorem.** (예: Li (2020) 및 후속 연구; Rozhoň et al. (2022)에서 관련 결과들을 정리해 언급) 간선 가중치가 nonnegative인 무방향 그래프에서 정점 $x$에 대해, $x$를 루트로 하는 $(1+\epsilon)$-approximate shortest path tree를 $\widetilde{O}(m/\epsilon^2)$ work, $\widetilde{O}(1)$ depth에 구할 수 있다.

이때 $T$가 $(1+\epsilon)$-approximate shortest path tree라는 건 모든 $y$에 대해 $x,y$를 잇는 $T$의 유일한 경로 길이가 실제 최단경로보다 최대 $(1+\epsilon)$배 길다는 뜻입니다. 우리 문제의 경우에는 $\epsilon = 1/(10k)$ 정도를 주면 exact shortest path tree가 없어도 진행이 가능합니다. 또한 우리는 가중치가 간선이 아니라 정점에 있지만, 간선 $e=(u,v)$의 가중치를 $(w(u)+w(v))/2$로 두면 경로 가중치가 정점 가중치 합과 거의 동일하게 대응되어, 이 결과를 가져다 쓸 수 있습니다.

이제 MWU를 사용하는 루틴을 간단히 적어 보겠습니다. 저는 MWU를 이산적인 gradient descent 정도로 이해하고 있는데, 여기서는 작동 방식 설명 대신 알고리즘 형태에 집중하겠습니다.

**Initialization:** $w^{(0)}(v) = 1$ for all $v \in V - \{x\}$, $w^{(0)}(x) = 0$.

이후 $i = 1, \cdots, \tau = O(k^{3}\log n)$에 대해 다음을 반복합니다.

**Shortest Path:** $w^{(i-1)}$을 가중치로 갖는 그래프에서 $x$를 소스로 하는 $(1+1/10k)$-approximate single-source shortest path tree(또는 거리)를 구하고, $1/(k-0.6)$ 이하의 가중치로 도달할 수 있는 정점들의 degree sum을 센다.

**Output Case:** 만약 이 값이 $\mu \cdot \mathrm{poly}(k)$ 미만이라면 그 즉시 $w^{(i-1)}$를 리턴한다.

**Update Case:** 아니라면, 해당 정점들 중 하나를 degree에 비례하여 랜덤으로 고른다. 이를 $y$라고 하면 $w^{(i)}(y)=0$으로 두고, $x$와 $y$를 잇는 (근사) 최단경로 위의 모든 정점 가중치에 $1+\epsilon$을 곱한다. 이렇게 수정된 가중치를 $w^{(i)}$로 둔다.

**Termination:** Output Case에 한 번도 도달하지 않았다면 `Fail`을 출력하고 종료한다.

알고리즘을 직관적으로 이해하려면 $x$를 포함하는 local cut $(L, S, R)$이 있다고 가정하고 생각하면 쉽습니다. Output Case는 volume이 $\mu \cdot \mathrm{poly}(k)$인 어떤 $L'$을 나머지와 분리하는 cut을 얻은 것이니 간단합니다. cut이 있지만 Update Case에 빠졌다고 생각해 봅시다. degree sum에 비례해서 정점을 뽑으면 $\deg(L) \le 2\mu$를 가정했을 때 $1 - 1/\mathrm{poly}(k)$ 정도의 확률로 $y \in R$을 뽑게 되고, $x$와 $y$를 잇는 경로를 업데이트하면 그 경로는 $S$를 지나야 하므로 $S$ 쪽 가중치가 누적됩니다. 이런 식으로 전체 weight 중에서 $S$가 차지하는 비중이 늘어나고, 이 Update를 $O(k^{3}\log n)$번 반복하면 cut이 있기만 하다면 결국 Output Case로 빠질 수밖에 없다는 결론을 (확률적으로) 보일 수 있습니다.

다 좋은데, Shortest Path는 조상님이 찾아주지 않습니다. 위 알고리즘을 그대로 쓰면 매 라운드마다 전역적으로 SSSP를 풀어야 해서 비용이 커집니다. 특히 우리가 원하는 건 $\mu \cdot \mathrm{poly}(k)$ 수준으로 localize된 work인데, $\mu$가 꽤 작을 때도 매번 $O(m)$ 규모의 전역 계산을 하면 목표를 달성할 수 없습니다. 그래서 shortest path를 찾다가 말고, 도달 가능한 영역의 “크기”(volume)가 $\mu \cdot \mathrm{poly}(k)$를 넘어가면 끊어주는 형태로 바꿔야 하는데 이건 쉽지 않습니다. 다음 섹션에서 이를 해결하는 아이디어를 아주 간단히 다루겠습니다.

### Single-Source Distance Sparsification

이 섹션은 이 글을 통틀어 가장 지엽적이고 복잡하기 때문에, MWU를 잘 모르시면 스킵하셔도 괜찮습니다.

일반적으로 $w^{(0)}(v)=1$로 초기화하는 건 MWU의 수렴 속도를 보장하기 위한 전형적인 선택인데, 이렇게 되면 정점 가중치가 조금만 바뀌어도 shortest path tree가 크게 바뀌기 때문에 전체 shortest path tree를 매번 $O(m)$에 가깝게 재계산해야 하는 문제가 생깁니다.

한 가지 트릭은 $x$의 neighbor에만 $n^{3}$ 정도의 큰 가중치를 주고, 나머지에는 1의 가중치를 줍니다. 그리고 shortest path를 계산할 때는 initialization 이후로 가중치가 1에서 안 바뀐 점은 “안 중요하다”라고 보고 0으로 취급합니다. 이런 새로운 가중치를 $\tilde{w}^{(i)}$라고 둡시다.

왜 이런 일을 할까요?

- $x$에서 출발한 경로 $P$에 대해 “안 중요한” 점의 가중치 합은 커봐야 $n$이고, $x$의 neighbor를 반드시 지나기 때문에 전체 가중치는 최소 $n^3$입니다. 그래서 $w^{(i)}(P)$와 $\tilde{w}^{(i)}(P)$는 최대 $1 + 1/n^2$ 정도의 비율로만 달라지고, 우리가 쓰는 근사 오차 및 MWU의 임계값 설정에서는 무시할 수 있는 수준으로 만들 수 있습니다.
- 그런데 $\tilde{w}^{(i)}(P)$는 Update Case가 실제로 “중요하게 만든” 정점들에서만 nonzero이기 때문에, shortest path $P$를 신중하게 골라주기만 하면 $\tilde{w}^{(i)}(P) > 0$인 “중요한” 정점의 수를 $\mu \cdot \mathrm{poly}(k)$ 이하로 강제할 수 있습니다.
- “안 중요한” 정점들은 $\tilde{w}=0$이기 때문에, 연결된 중요한 정점들만 보면 됩니다. 결국 중요한 정점들 사이의 연결성만 관리하면 되는데, 이는 shortest path를 직접 관리하는 것보다 훨씬 쉽습니다. 실제로는 $\tilde{w}$가 다시 0으로 돌아가는 경우가 없으니 decremental connected component(또는 decremental spanning forest)를 관리하면 됩니다.
- 추가적으로, 이렇게 왜곡된 weight로 인해 MWU의 수렴 속도가 바뀌긴 하는데, 많아야 $O(\log n)$ 정도의 factor가 붙습니다.

그래서 중요하지 않은 점들을 contract하면, 매번 $\mu \cdot \mathrm{poly}(k)$개의 정점과 간선을 갖는 그래프에서 approximate shortest path를 구해주면 되기 때문에 알고리즘을 localize하는 데 성공합니다.

추가적으로 안 중요한 점들을 관리하는 decremental spanning forest가 필요한데, 디테일은 논문의 Appendix를 참고해주시면 되겠습니다.

## Conclusion & Lower Bounds

저희 알고리즘은 $k = n^{o(1)}$에 대해서는 always almost-linear algorithm을 줍니다. 결국 “Reachability 없이 어디까지 갈 수 있냐?”에 대해 긍정적으로 답한 결과인 셈입니다. 일관되게 그래프를 localize해서 “나를 포함하는 작은 cut”을 찾는 방법을 고안하였습니다.

추가적으로 general $k$, 정확히는 polynomially large $k$에 대해서는 “reachability가 반드시 필요할 것 같다”는 방향의 barrier를 논문에서 논의합니다. 요지는 $k$에 대한 polynomial dependence를 유지하면서 reachability-free로 더 강한 병렬화(almost-linear work, subpolynomial depth)를 얻는 것은 현재 이해로는 어렵다는 것입니다.

논문 Appendix C에서는 Dense Maximum Bipartite Matching(D-MBM)이 dense reachability를 함의한다는 folklore reduction을 언급하고, 이를 direct-sum 형태로 확장한 문제를 사용합니다. 구체적으로 “여러 개의 dense bipartite 그래프 중에서 perfect matching이 없는 그래프가 하나라도 있는가?” 같은 판별 문제(t-DPBM)를 생각하고, 이를 $k$-vertex connectivity로 환원합니다.

만약 $k = \Omega(n^{\delta})$에 대해 $k$-vertex cut을 almost-linear work, subpolynomial depth에 해결할 수 있다면, 각각의 그래프 크기가 대략 $n^{\delta}$인 서로 다른 $t = n^{1-\delta}$개의 dense bipartite 그래프들에 대해 위와 같은 direct-sum 판별을 almost-linear work 및 subpolynomial depth에 해결할 수 있습니다. 입력 크기 관점에서 각 인스턴스의 간선 수가 $\Theta((n^{\delta})^{2})$이므로 전체 입력 크기는 $\Theta(t \cdot (n^{\delta})^{2}) = \Theta(n^{1+\delta})$이고, almost-linear work는 $\widehat{O}(n^{1+\delta})$ 꼴이 됩니다.

즉, direct-sum 문제를 각 인스턴스를 독립적으로 처리하는 것보다 polynomially 더 쉽게 풀 수 있지 않다고 믿는다면, polynomial $k$에 대해 reachability-free로 더 강한 병렬화를 기대하기는 어렵다는 직관을 줍니다.

## References

글의 엄밀한 버전은 논문의 Introduction부터 천천히 읽어보시면 확인할 수 있습니다. 아래는 같이 참고할만한 글들입니다.

- [Isolating Cut](https://hackmd.io/@U0nm1XUhREKPYLt1eUmE6g/rku_qeUn9) - vertex cut을 polylogarithmic max flow로 해결할 수 있는 방법에 대한 글입니다.
- [Deterministic Near-Linear Small Vertex Cut (YouTube)](https://www.youtube.com/watch?v=dBNWSfgoCg8) - 처음으로 near-linear small vertex connectivity를 해결한 논문의 영상 설명입니다.
