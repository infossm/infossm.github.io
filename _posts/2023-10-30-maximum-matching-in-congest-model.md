---
layout: post

title: "Maximum matching in CONGEST model"

date: 2023-10-30

author: ainta

tags: [graph-algorithm, distributed-algorithm]
---

# Introduction

이전에 작성한 글 [A Near-Optimal Deterministic Distributed Synchronizer](https://infossm.github.io/blog/2023/05/21/A-Near-Optimal-Deterministic-Distributed-Synchronizer/)에서 잠깐 언급하였듯이, Distributed Graph Algorithm에서는 어떠한 모델을 가정하고 있는지에 따라 문제를 접근하는 방법이 달라지게 된다. 해당 포스트에서는 Asynchronized model을 가정하고 synchronized model처럼 문제를 해결할 수 있도록 도움을 주는 synchronizer에 대한 내용을 작성하였다. 이번 포스트에서는 synchronized model 중 하나인 CONGEST model을 가정하고, graph theory의 중요한 문제 중 하나인 maximum matching problem이 어떤 식으로 해결 가능한지에 대해 알아보도록 할 것이다.

## Synchronized message-passing model

분산 컴퓨팅 네트워크를 undirected graph $G=(V,E)$ 로 표현하자. 각 Node는 하나의 컴퓨터 또는 프로세서의 역할을 한다. Synchronized setting에서 계산 및 커뮤니케이션은 synchronous round에 이루어진다. 각 라운드마다 각 노드는 가지고 있는 데이터로 계산을 한 후 neighbor에 하나의 메시지를 보낼 수 있고, 이 때 보낼 수 있는 메시지의 크기에 따라 두 가지 모델로 나뉜다.

-  **LOCAL** 모델: 메시지의 크기가 입력의 크기의 다항함수로 표현되는 정도 (poly-size). 사실상 무제한이라고 볼 수 있다.

-  **CONGEST** 모델: 보낼 수 있는 메시지의 크기를 $O(\log N) = B$ bit로 가정

또한 편의상 다음 몇 가지를 추가로 가정한다.

- 보낸 메시지가 다 도착한 후 다음 round가 시작한다고 가정한다.

- 시작할 때 각 노드는 전체적인 $G$를 모르고, 자신의 identifier(노드 번호) 및 neighbor만 아는 상태이다. 

- 각 round 사이에 각 노드에서는 poly-time 연산을 할 수 있다.

## CONGEST Model vs LOCAL Model

간단히 그래프의 spanning tree를 구하는 문제를 생각해보자.  spanning tree를 올바르게 구하는 distributed algorithm이라는 것은 어떻게 정의될 수 있을까? 각 노드의 최종 상태로 이를 정의할 수 있을 것이다. 예를 들어, 알고리즘이 종료된 후 1번 노드가 아닌 모든 노드 $v$에 대해 정점 $p_v$가 저장되어 있고 각 $(v, p_v) \in E$이며 $(v, p_v)$들이 $G$의 spanning tree를 이룸이 보장이 될 때 이를 spanning tree를 올바르게 구하는 distributed algorithm이라 부를 수 있을 것이다. 이와 다르게 정의하는 것도 물론 가능하다.

Spanning tree의 경우, $G$의 각 vertex에서 가까운 local한 부분의 성질만 가지고 해결하는 것이 쉽지 않을 것임을 추측할 수 있다. 실제로, 그래프의 지름($G$에서 최단거리가 가장 점 두 node사이의 거리)을 $D$라 할 때, spanning tree를 구하는 distributed algorithm은 LOCAL, CONGEST 모델 모두 최소 $\Omega(D)$ 번의 round가 필요하다. 

한편, LOCAL 모델의 경우 그래프 전체의 정보를 모든 노드에 $D$번만에 전파 가능하기 때문에, LOCAL 모델에서 $O(D)$번 이상의 round를 사용하는 것은 큰 의미가 없다. 반면 CONGEST 모델의 경우에는 한 round에 한 간선을 통해 $n$ 이하의 정수 상수 개 정도만 전달 가능하므로 $O(D)$번의 round에 문제를 해결하는 naive한 방법은 존재하지 않고, 좀더 고민의 여지가 있음을 알 수 있다.

# Results

오늘 알아볼 결과를 먼저 소개한다.

- Unweighted maximum matching : bipartite graph $O(n \log n)$ rounds, gerneral graph $O(n^2)$ rounds.

- $(1-\epsilon)$-approximation for fractional weighted maximum matching in general graphs:  $O(\log (\Delta W)/\epsilon^2)$

이제 각각의 결과가 어떤 식으로 얻어지는지 알아보자.

# Maximum bipartite matching

maximum bipartite matching은 우리가 일반적으로 생각하듯 augmenting path를 찾아 업데이트하는 식으로 해결할 수 있다.

가장 짧은 augmenting path를 찾아 그 길이를 $l$이라 하고, 이를 업데이트하는 것을 반복하면 $l$들의 총합은 $O(n \log n)$이 된다. (이는 길이 $l$ 이상의 augmenting path의 개수는 $O(n/l)$을 넘지 않기 때문이고, Hopcroft-Karp 알고리즘이 $O(\sqrt{n})$번의 BlockFlow phase에 종료되는 것과 같은 원리이다.) 따라서, 해당 과정을 $O(l)$번의 round에 수행할 수 있다면 우리가 찾는 $O(n \log n)$ round algorithm이 될 것이다.

하나의 augmenting path를 찾는 과정은 다음과 같이 이뤄진다.

0. 먼저 이번에 찾을 augmenting path 길이의 upper bound $k$를 알고 있다고 하자.
1. matching되지 않은 각 node에 대해 해당 node의 번호가 담긴 odd 토큰을 생성한다.
2. 각 round에서, odd 토큰이 존재하는 노드는 matching edge가 아닌 모든 edge에 대해 상대 끝점으로 동일한 번호의 even 토큰을 전달한다.
3. 각 round에서, even 토큰이 존재하는 노드는 matching edge에 대해 동일한 번호의 odd 토큰을 전달한다.
4. 아직 토큰이 없는데 현재 round에서 토큰을 receive하는 node의 경우, 해당 토큰을 저장한다. 만약 여러 토큰이 동시에 온 경우 가장 작은 번호의 토큰만 저장한다. 또한 토큰을 전달해준 노드를 해당 노드의 parent로 기록한다.
5. 만약 round를 진행하다가 어떤 인접한 노드 $u,v$가 존재하여 동시에 $u$에서 $v$로, $v$에서 $u$로 서로 다른 번호의 토큰이 전달되었다면, 두 토큰의 번호가 $s, t$일 때 $s$와 $t$ 사이의 augmenting path를 찾은것이다.
6. $u, v$가 round $r$에 augmenting path를 detect했고 token의 번호가 $src_u, src_v$라면 augmenting path의 identifier는 $(r, \{ src_u, src_v \}, \{ u, v \})$이다. 
7. 하나의 node가 여러 augmenting path를 detect하는 경우, 가장 사전순으로 작은 것을 가지고 있는다.
8. augmenting path를 detect한 후에는 parent에게 augmenting path의 identifier를 전달하는 것을 반복하여 최종적으로 free node까지 전달되도록 한다.
9. augmenting path를 전달받은 free node는 path 의 confirmation을 path를 따라 전달한다. (이는 그동안 parent를 따라가며 전달하는것의 역방향이므로 전달할 때 표지를 해 놓음으로써 가능하다) confimation이 완료되면 path를 따라 update를 진행한다.


upper bound $k$는 현재 매칭 $r$에서 $n/(r+1)$로 줄 수 있으므로, $O(n \log n)$ round에 bipartite maximum matching 문제를 해결할 수 있다.

위 알고리즘에서 bipartite라는 조건은 augmenting path의 길이를 bound하는 데에만 사용되었다. 따라서, 동일한 방법으로 $O(n^2)$ round에 maximum matching in general graph를 해결할 수 있다.


# Fractional Matching Approximation

## LP representation

**Notation**. 그래프 $G = (V,E)$에서 각 간선 $e \in E$에 weight $w_e$가 주어져있을 때, Maximum weighted fractional matching problem은 다음과 같은 packing linear program(LP)로 쓸 수 있다.

$\text{Maximize} \sum_{e \in E} y_e$

where

$\forall v\in V : \sum_{e \in E, v \in e} \frac{y_e}{w_e} \le 1, \forall e \in E : y_e \ge 0$

식을 찬찬히 보면 $y_e$가 0 또는 $w_e$의 값만 가질 수 있을 때는 우리가 일반적으로 weighted general matching이라고 하는 문제임을 알 수 있고, 이것이 $[0, w_e]$의 값으로 relaxation된 문제가 **weighted fractional maximum matching** problem이다.

위 LP에 대한 dual을 생각하여 이를 식으로 정리하면 다음과 같다.


$\text{Minimize} \sum_{x \in V} x_v$

where

$\forall e = (u,v) \in E : \frac{x_u}{w_e} + \frac{x_v}{w_e} \ge 1 , \forall v\in V : x_v \ge 0$

이는 bipartite matching의 dual LP가 각 정점에 값을 주어 각 edge의 양 끝 점의 값의 합이 1 이상이 되도록 하는 문제, 즉 vertex cover가 됨을 생각하면 거의 동일하다.

## The Distributed Fractional Matching Algorithm

앞서서 살펴본 LP를 해결하는 distributed algorithm을 제시할 것이다. 

먼저, 알고리즘은 real-valued 파라미터 $\alpha$와 integer-valued 파라미터 $f$를 가진다. 이 때, $\alpha > 1, f \ge 1$이어야 한다.

알고리즘에서는 각 vertex와 edge에 대한 변수 $x_v, y_e, r_e$를 관리할 것이다. $x_v, y_e \ge 0, r_e \in [0,1]$을 만족한다. 여기서 $x_v, y_e$는 위 LP에서 등장하는 변수이다. 

처음에는 $x_v, y_e$는 모두 0으로, $r_1$은 모두 1로 초기화하며 알고리즘이 진행됨에 따라 $x_v, y_e$들은 단조 증가, $r_e$는 단조 감소하는 성질을 가진다.

먼저 알고리즘의 한 phase를 의사코드로 나타내면 다음과 같다.

<p align="center">
    <img src="/assets/images/maximum-matching-in-congest-model/fig1.png" width="800"/>
    <br>
</p>

위 phase에서, 각 노드에 대해 일반화된 degree $\gamma(v) = \sum_{e\in E, v\in e} \frac{r_e}{w_e}$라는 값을 도입한 것을 알 수 있다. $\hat{\gamma}(v) = max_{u \in (v \cup N(v))} \gamma(u)$는 자신과 인접한 노드에 대한 $\gamma$의 최댓값이다.

알고리즘의 실행에서 각 노드 $v$는 $\gamma(v) > 0$인 동안 activate된 상태로 알고리즘에 참여하게 되며 $\gamma(v) = 0$이 되면 더이상 알고리즘에 영향을 끼치지 않는다.

phase의 line 6-9을 보면, $x_v$가 1 증가하면 $y_e$들의 총합 역시 1 증가하며, $r_e$를 $\alpha ^ {1/w_e}$로 나눈다. 그런데 $r_e$는 $\alpha^{-f}$가 되면 $0$으로 설정하므로 모든 $e = (u,v)$에 대해 $x_u + x_v \ge w_e \cdot f$가 성립하고, 이를 다시 쓰면 $\frac{x_u}{w_e} + \frac{x_v}{w_e} \ge f$이다. $f \ge 1$이므로, 이는 $x_v$에 대한 LP의 condition을 만족함을 알 수 있다.

$x_v$는 $y_e$의 dual LP이고 합이 증가하는 정도가 같으므로 LP의 값($x_v$의 합 = $y_e$의 합)은 같다. 이 때 알고리즘의 Approximation ratio는 $f/\max_{v\in V} Y_v \le 1$의 lower bound를 가진다. 이를 통해 다음을 증명할 수 있다.

**Theorem.** $\alpha = 1+{\epsilon}/{c},  f = 2c \ln(\Delta W)/\epsilon^2$ for sufficiently large constant $c$로 설정했을 때, 위 fractional matching algorithm은 $(1-\epsilon)$-approximate fractional weighted matching을 계산한다. 또한, 이에 사용되는 round의 수는 최대 $O(\log(\Delta W)/\epsilon^2$이다.

위 Theorem에 의해, 우리는 두 번째 결과인 $(1-\epsilon)$-approximation for fractional weighted maximum matching in general graphs:  $O(\log (\Delta W)/\epsilon^2)$를 얻었다.