---
layout: post
title: "Unique Games Conjecture"
author: leejseo
date: 2022-11-12 09:00
tags: [algorithm, complexity theory, NP complete]
---

본인은 최근 들어 Theoretical Computer Science 분야 중에서도 Computational Hardness에 관련한 결과들을 다루는 스터디에 참여하고 있고, Unique Games Conjecture와 관련하여 발표할 일이 있었다. 나름 의미가 있는 주제임에도 불구하고, 이와 관련한 국문 자료는 (본인이 검색해본 한에서는) 전혀 찾아볼 수 없었어서 이 글을 작성하게 되었다.

이 글은 Unique Games Conjecture가 무엇이고, 이를 통해 어떤 결과들을 얻을 수 있는지에 대해 전달하는 것을 목표로 한다. Unique Games Conjecture에서 어떤 형태로 reduction을 구성할 수 있는지 그 느낌 또한 전달하는 것 또한 목표로 하되, reduction에 대한 formal한 증명 등을 다루는 것은 목표로 하지 않는다.

## 1. Introduction

NP-hard 문제들을 다항 시간 내에 해결할 수 있는 방법이 현재로는 알려져 있지 않다. 대신, NP-hard의 계산 복잡도를 가지는 최적화 문제들을 다항 시간 내에 근사하는 방법과 관련하여 많은 연구가 이루어졌다. NP-hard 문제의 근사와 관련해서는 "얼마나 잘 근사할 수 있는가?" 와 "얼마나 근사하기 어려운가?" 의 두 가지의 근원적인 질문이 있다.

한 가지 예로, MAX-3SAT 문제를 살펴보도록 하자. MAX-3SAT 문제는 최적화 문제 형태로 정의된 3SAT 문제로, 3CNF가 주어졌을 때, 최대한 많은 clause를 만족시키는 문제다. 이 문제의 경우,

- 다항 시간 내에 동작하는 $7/8$-approximation 알고리즘이 알려져 있으며,
- $P \neq NP$ 라는 가정 하에, 다항 시간 내에 동작하는 $(7/8 + \varepsilon)$-approximation 알고리즘이 존재하지 않음이 증명되어 있다.

즉, MAX-3SAT 문제의 경우 approximation ratio에 있어 upper bound와 lower bound가 실제로 일치한다.

또 다른 예로, Vertex Cover 문제를 살펴보자. 이 문제의 경우,

- 다항 시간 내에 동작하는 2-approximation 알고리즘이 알려져 있으며,
- $P \neq NP$ 라는 가정 하에, 다항 시간 내에 동작하는 $(1.3606 + \varepsilon)$-approximation 알고리즘이 존재하지 않음이 증명되어 있다. (참고: 이 사실은 꽤나 복잡한 수학적인 이론을 통해 증명되었다.)

Vertex Cover 문제의 경우, MAX-3SAT 문제와 다르게 approximation ratio에 있어 upper bound와 lower bound가 실제로 일치하지 않고, 꽤나 큰 간극이 있다.

$P \neq NP$ 가정 하에 Vertex Cover 문제에 대해 일치하는 lower/upper bound를 얻기 위해서는 무엇을 해야할까?

- Vertex Cover 문제에 대한 더 좋은 근사 알고리즘을 찾는다
  - 2-approximation 알고리즘의 작동 과정은 아주 단순하며, 발견된지 아주 오래 되었으며, 수십년간 더 개선된 알고리즘을 찾지 못했다.
  - 더 좋은 근사 알고리즘을 찾을 가능성과 관련하여 학계에서는 부정적으로 바라보고 있다.
- 더 복잡한 수학적인 이론 등을 도입하여 더 좋은 lower bound를 찾는다.
  - 이것 또한 성공할 가능성이 낮다는 것을 시사하는 연구가 있다.

이 글에서 다룰 ($P \neq NP$ 보다는 강한 가정인) Unique Games Conjecture를 가정하면

- Vertex Cover와 다른 여러 문제에 대한 일치하는 lower/upper bound를 얻을 수 있고,
- 여러 문제에 대한 개선된 lower bound를 얻을 수 있고,
- Unique Games Conjecture의 변형된 버전으로 부터 또 여러 문제에 대한 개선된 lower bounds를 얻을 수 있다.

## 2. UGC

Unique Games Conjecture를 가정하기에 앞서, 먼저 Label Cover 문제를 정의할 것이다. Label Cover 문제는 다음과 같이 정의된다.

- Instance
  - 이분 그래프 $G = (V \cup W, E)$ 와 두 양의 정수 $a \ge b$
  - 각 간선 $e \in E$ 마다 전사 함수 $\pi_e : [a] \to [b]$
- Question
  - 레이블 $\ell: V \cup W \to [a] \cup [b]$, $\ell(V) \subseteq [a], $ $\ell(W) \subseteq [b]$ 이 존재하여 각 간선 $e = (v, w) \in E$ 에 대해 $\pi_e(\ell(v)) = \ell(w)$ 를 만족하는가?

Label Cover 문제의 approximation 버전의 경우 간선들에 걸린 contraints 중 $\delta$ 이상의 비율을 만족시키는 레이블이 존재하는지 묻는 문제로 정의할 수 있다. Label Cover 문제는 NP-complete임이 알려져 있다.

Label Cover 문제의 신기한 성질은 자체를 해결하는 것 뿐만 아니라, Label Cover 문제의 만족될 수 있는 instance와 나쁜 instance를 구분하는 것 조차 어렵다는 것이다. Label cover 문제의 instance $U$ 에 대해 constraint가 만족될 수 있는 최대 비율이 $\delta$ 라 할 때, $OPT(U) = \delta$ 로 정의하자.

**Theorem.** 각 $0 < \delta < 1$ 에 대해 어떤 상수 $C$ 가 존재하여 다음을 만족시킨다.

- $U$ 가 $a = \Theta((1/\delta)^C)$ 를 만족하는 label cover 문제의 instance라 하자.
- 다음 두 경우를 구분하는 것은 NP-hard 이다.
  - $OPT(U) = 1$ (label covering이 존재)
  - $OPT(U) \le \delta$ ($\delta$ 보다 좋은 label covering이 존재하지 않음)

Label Cover 문제와 비슷한 문제인 Unique Label Cover 문제를 살펴보자. 이 문제의 경우, $a = b$ 인 Label Cover 문제라고 생각할 수 있다.

- Instance
  - 이분 그래프 $G = (V \cup W, E)$ 와 양의 정수 $[a]$
  - 각 간선 $e \in E$ 마다 전단사 함수 $\pi_e : [a] \to [a]$
- Question
  - 레이블 $\ell: V \cup W \to [a]$ 이 존재하여 각 간선 $e = (v, w) \in E$ 에 대해 $\pi_e(\ell(v)) = \ell(w)$ 를 만족하는가?

신기하게도, Unique Label Cover 문제 자체의 경우 만족될 수 있는 instance와 나쁜 instance를 구분하는 것 자체는 쉽다.

**Theorem.** 각 $0 < \delta < 1$ 에 대해 다음은 다항 시간에 구분할 수 있다.

- $OPT(U) = 1$
- $OPT(U) \le \delta$

그래서 Unique Label Cover 문제에서 어려운 instance를 뽑아내려면, "나쁜" 쪽 뿐만 아니라 "좋은" 쪽에도 에러(2-side error)를 허용해야 한다. 이것이 어려운지는 밝혀져 있지 않으며, 어렵다는 사실이 바로 Unique Games Conjecture 이다.

**Conjecture.** (The Unique Games Conjecture, UGC) 각 $0 < \varepsilon, \delta < 1$ 에 대해 상수 $C$ 가 존재하여 다음을 만족시킨다.

- $U$ 가 $a = C$ 를 만족하는 unique label cover 문제의 instance라 하자.
- 다음 두 경우를 구분하는 것은 NP-hard 이다.
  - $OPT(U) \ge 1 - \varepsilon$ (좋은 instance)
  - $OPT(U) \le \delta$ (나쁜 instance)

여담으로, Unique Games Conjecture 라는 이름에서 "Unique"가 어디에서 왔는지는 Label Cover를 먼저 살펴보고, 그 이후에 Unique Label Cover를 살펴보면서 궁금증이 해소되었으리라 생각한다. 그렇다면 "Game"은 어디에서 왔을까?

Label Cover/Unique Label Cover 문제는 두 명의 Prover와 한 명의 Verifier로 구성된 게임으로 생각할 수 있다. 이 게임에서 각 Prover는 Verifier와만 통신할 수 있고, Verifier에게 자신들이 레이블을 들고 있음을 납득시키려 한다. 게임의 각 라운드는 Verifier가 임의의 간선을 뽑고, 각 Prover에게 양 끝점 중 하나씩을 알려주고, 각 Prover는 자신이 받은 정점의 레이블을 Verifier에게 반환하는 식으로 구성된다. 양 끝점의 레이블이 해당 간선에 대한 contstraint를 만족하면 Prover가 이기고, 아니면 Verifier가 이긴다.

이러한 게임 세팅으로 부터 UGC의 이름에 있는 "Game" 이라는 단어가 나왔고, $OPT(U)$ 는 $U$ instance 하에서 게임을 할 때 Prover 들이 이길 수 있는 최대 비율과도 같게 된다.

## 3. Using UGC: Max Cut

이 섹션에서는 UGC를 이용해 Max Cut에 대한 일치하는 upper/lower bound 결과를 얻어보는 것을 목표로 한다. 앞에서 예시로 든 Vertex Cover 문제가 아닌 Max Cut 문제를 살펴보는 이유는 Max Cut 문제에 대한 reduction이 그나마 더 이해하기 쉽기 때문이다. (사실 이 조차도 꽤나 난해하다고 생각한다. 이 섹션의 목표는 어떤 식으로 reduction을 구상하는지, 그 느낌을 얻어가는 정도이며 전부 이해/증명하는 것을 목표로 하지는 않는다.)

### 3.1. Approximating Max Cut

Max Cut 문제의 경우, 그래프가 주어졌을 때 정점 집합을 두 개로 분할하여 두 집합 사이를 가로지르는 간선의 수 (혹은 간선의 가중치의 합) 를 최대화 하는 것을 목표로 한다. 이 문제의 경우, Semidefinite Programming(SDP) 이라는 framework 을 이용하여 다항 시간에 근사할 수 있다.

SDP의 경우, 다음과 같이 정의된다.

- Instance
  - 각 $1 \le i, j \le n$ 에 대해 수 $c_{i, j}$
  - 각 $1 \le i, j \le n$, $1 \le k \le m$ 에 대해 수 $a_{i, j, k}$
  - 각 $1 \le k \le m$ 에 대해 수 $b_k$
- Question
  - Minimize $\sum_{1 \le i, j \le n} c_{i, j} (x^i \cdot x^j)$
  - Subject to $\sum_{1 \le i, j \le n} a_{i, j, k} (x^i \cdot x^j) \le b_k$ for every $1 \le k \le m$ where $x^1, \cdots, x^n \in \mathbb{R}^n$

참고로, SDP의 이름에 Semidefinite 라는 단어가 들어가는 이유는 이 문제를 다른 형태(행렬 이용)로 정의 했을 때 Positive Semidefinite Matrix 등의 개념이 등장하기 때문이라고 생각하면 된다.

Max Cut 문제를 다음과 같이 생각할 수 있다.

- Constraints: $x_1, x_2, \cdots, x_n \in \{-1, 1\}$
- Objective Function: maximize ${1 \over 2} \sum_{(i, j) \in E} w_{i, j} (1 - x_ix_j)$

이 문제에 대한 SDP Relaxation을 생각할 수 있다. 구체적으로는, $x_i$ 가 $\pm 1$ 중 하나라는 조건을 $x_i$ 가 unit vector 라는 조건으로 완화시킨다.

- Constraints: unit vector $x_i \in \mathbb{R}^n$ for $i = 1, 2, \cdots, n$
- Objective Function: maximize ${1 \over 2} \sum_{i, j \in E} w_{i, j} (1 - x_i x_j)$

SDP Relaxation 버전의 문제를 해결하여 얻은 벡터들은 $n$ 차원 단위 구 상에 존재할 것이다. 구의 중심을 지나는 hyperplane을 하나 랜덤하게 택해서 각 정점에 대응되는 벡터가 어느 쪽 반구에 속하는지를 기준으로 정점 집합을 분할하는 randomized approximation 알고리즘을 생각할 수 있을 것이다. (사실 따져봐야 하는 반구의 개수 자체는 다항식 개수 만큼만 있어서 deterministic한 다항 시간 근사 알고리즘으로 바꿀 수 있을 것 같으나, 깊게 생각 해보지는 않았다.) 이 알고리즘의 approximation ratio와 관련하여 다음의 사실이 증명되어 있다. (어렵지 않다! 심심하면 직접 해보길...)

**Theorem.** (Integrality Gap) $\mathbb{E}[\text{cut found}] \ge 0.87586 SDPOPT \ge 0.87586OPT$.

여기에서 $SDPOPT$ 는 SDP relaxation 하에서의 optimal value 이며, $OPT$는 원본 문제의 optimal value 이다.

### 3.2. Inapproximability of Max Cut

이제 Unique Games Conjecture로 부터 Max Cut의 Inapproximability와 관련한 결과를 유도해보자. Unique Games Conjecture로 부터 Max Cut으로의 Reduction 구상은 기본적으로 다음과 같이 이루어진다.

- Unique Label Cover의 좋은 instance를 Max Cut의 크기가 큰 instance로 (completeness)
- Unique Label Cover의 나쁜 instance를 Max Cut의 크기가 작은 instance로 (soundness)

Reduction의 구상 자체는 PCP와 비슷한 느낌으로 이루어진다. 여기에는 Long Code라는 개념이 등장한다.

**Definition.** (Long Code) 메시지 $m \in [q]$ 를 $f_m(x) = x_m$ 인 $f : \{-1, 1\} ^ q \to \{-1, 1\}$ 로 encode 한다.

비록 $\log q$ 비트의 정보를 담은 메시지를 Long Code 로 나타내면 $2^q$ 비트가 필요하지만, Long Code로 표현했을 때 여러 이점들이 있어 Long Code를 사용한다. 그리고 "PCP와 비슷한 느낌" 이라는 말을 조금 더 구체적으로 설명하자면, reduction 자체가 작을 필요는 없다. Reduction 자체는 아주 거대하게 구상하되, 이를 explicit 하게 들고 있는게 아니라, "테스트" 할 때 reduction으로 만든 무언가의 "일부분"을 효율적으로 "inference" 해서 쓰면 되기 때문이다.

Long Code의 성질 중 가장 중요한 것은 바로 어떤 함수가 Long Code와 비슷한지 테스트 하기 위해서는 단 2개의 비트만 읽어보면 된다는 것이다.

**Definition.** ($k$-junta: Generalized Version of the Long Code) 만약 어떤 함수 $f : \{-1, 1\}^q : \{-1, 1\}$ 의 값이 $k$ 개의 변수에 의해서만 결정된다면, 이를 $k$-junta 라고 부른다.

**Theorem.** (Testing the Long Code) $x \in \{-1, 1\}^q$가 주어졌을 때, $z$를 $x$의 각 비트를 $p$ 의 확률로 뒤집어서 만들자. $f$ 가 long code라면, $\Pr[f(x) = -f(-z)] = 1-p$ 이다.

**Definition.** (Majority Function) Majority function $Maj : \{-1, 1\}^q \to \{-1, 1\}$ 를 $Maj(x) := \mathrm{sign}(\sum x_i)$ 로 정의하자. 즉, $1$ 의 등장 횟수가 $-1$ 의 등장 횟수보다 많으면 1, 아니면 -1의 값을 지닌다.

**Theorem.** (Testing the Majority Function) $\Pr[Maj(x) = -Maj(-z)] = \arccos(1 - 2p)/\pi$.

**Theorem.** 임의의 $\varepsilon, p$ 에 대해 어떤 $k, \delta$ 가 존재하여 모든 $f: \{-1, 1\}^q \to \{-1, 1\}$ 에 대해 다음 중 하나를 만족시킨다:

- $\Pr[f(x) = -f(-z)] \le \Pr[Maj(x) = -Maj(-z)] + \varepsilon$,
- $f$ is $\varepsilon$-close to a $k$-junta.

즉, 어떤 함수는 Long Code와 비슷한 무언가($k$-junta) 이거나, 혹은 테스트를 통과하지 못할 확률이 크다.

이제, Max Cut에 대한 PCP verifier를 다음과 같이 만들 수 있다.

- 정점 $v \in V$ 와 이의 두 이웃 $w, w' \in W$ 를 무작위로 고른다.
- $x \in \{-1, 1\}^a$ 를 무작위로 고른다.
- $z \in \{-1, 1\}^a$ 를 $x$ 의 각 비트를 $p$ 의 확률로 반전시켜 무작위로 샘플한다.
- Accept if and only if $f_{\ell(w)} (x \circ \pi_{v, w}) \neq f_{\ell(w')} ((-z) \circ \pi_{v, w'})$
  - 여기에서, index에 permutation을 적용하는 것으로 생각하면 된다.

이 Reduction에 대한 Completeness, Soundness 증명의 proof sketch를 살펴보자. Soundness 증명의 경우 Long Code에 대한 성질과 어려운 수학적인 내용이 들어가 "큰 그림" 정도만 살펴본다.

- Completeness
  - 어떤 labeling이 $(1 - \varepsilon)$ 비율 이상의 간선 조건을 만족시킨다면, PCP에서 사용되는 두 간선이 모두 만족될 확률은 $(1 - 2\varepsilon)$ 이상이 된다.
  - PCP에서 사용되는 두 간선이 모두 labeling에 의해 만족되는 경우, accept 될 확률은 $(1 - p)$ 이다.
  - 따라서, PCP verifier를 통과할 확률은 $(1 - 2\varepsilon)(1- p)$ 이상이 된다.
- Soundness
  - 대우 명제를 보인다.
  - 어떤 Long Code가 accept 될 확률이 $\arccos(1 - 2p)/\pi + \varepsilon$ 이상이라면, $\varepsilon, p$ 에 의해 결정되는 어떤 상수 $\delta'$ 에 대해 간선 중 $\delta'$ 비율 이상을 만족시키는 labeling을 얻을 수 있다. (이 과정이 몹시 어렵다.)
  - 이 Labeling은 label set size $a$ 에 의존하지 않으므로, $a$ 를 충분히 크게 잡으면 증명이 종료된다.

식 $\displaystyle \frac{\arccos(1 - 2p)/\pi}{p}$ 의 최솟값을 계산해보면, 0.87586 이라는 상수를 얻게 된다.

## 4. Interesting Results from UGC

이 섹션은 UGC로 부터 얻어진 연구 결과 가운데 일부를 정리해놓았다. 더 상세한 정보는 Reference에 있는 책이나 본인이 발표하면서 만들었던 슬라이드를 참고하면 좋다.

### 4.1. UGC Implies Optimal Lower Bounds

$P \neq NP$ 로 부터는 일치하는 lower/upper bound를 얻지 못했지만, UGC로 부터 일치하는 lower/upper bound를 얻은 문제로는 다음이 있다.

- Vertex Cover
- Vertex Cover on $k$-uniform hypergraphs
  - 참고: $k$-uniform hypergraph라 함은 $E \subseteq {V \choose k}$ 임을 의미한다.
- Max Cut
- Max 2-SAT

### 4.2. UGC Implies Good but Not-Optimal Lower Bounds

일치하는 lower/upper bound는 아니어도, UGC로 부터 개선된 lower bound를 얻어낸 문제에는 다음이 있다.

- Feedback Arc Set
  - Instance: 유향 그래프 $G = (V, E)$
  - Question: 각 유향 사이클로 부터 최소 하나의 간선을 포함하는 간선 집합의 최소 크기는 얼마인가?
  - 참고: Vertex Cover로 부터의 reduction을 통해 $1.36$ 의 lower bound가 증명되어 있었으나, UGC로 부터는 $\omega(1)$ 의 결과를 얻어낼 수 있었다.
- A variant of Min-2SAT-Deletion
  - Instance: $\bar x \lor \bar y$ 형태의 clause를 포함하지 않는 2-CNF formula
  - Question: 만족될 수 있는 clause의 최대 개수는 얼마인가?
  - 참고: APX-hard 임이 알려져 있었으나, UGC로 부터 $\omega(1)$ 의 결과를 얻어낼 수 있었다.

### 4.3. UGC+ Implies Some Lower Bounds

UGC의 variant로 부터 개선된 lower bound를 얻어낸 문제의 예시로는 다음이 있다.

- Scheduling with Precedence Constraints
  - Instance: Job 들의 선후관계를 나타내는 DAG $G$. 각 job은 추가로 수행시간 $p_i$ 와 중요도 $w_i$ 를 지니고 있다.
  - Question: Job 들의 순서를 출력한다. 출력한 순서대로 job들을 수행했을 때 $i$ 번째 job이 $t_i$ 시각에 완료된다고 하자. $\sum c_i w_i$ 를 최소화해야 한다.
  - 참고: 이 문제에 대한 inapproximability 결과로는 아무 것도 알려져있지 않았으나 UGC+를 가정하여 $2\varepsilon$ 의 lower bound를 얻어낼 수 있다고 한다.

## References

- Computational Intractability: A Guide to Algorithmic Lower Bounds https://hardness.mit.edu/
- Stanford CS354 - Topics in Intractability: Unfulfilled Algorithmic Fantasies http://web.stanford.edu/class/cs354/
- Optimal Inapproximability Results for MAX-CUT and Other 2-Variable CSPs? https://www.cs.cmu.edu/~odonnell/papers/maxcut.pdf
- Washington CSE533 - The PCP Theorem and Hardness of Approximation https://courses.cs.washington.edu/courses/cse533/05au/
- On the Unique Games Conjecture (Survey) https://cs.nyu.edu/~khot/papers/UGCSurvey.pdf
