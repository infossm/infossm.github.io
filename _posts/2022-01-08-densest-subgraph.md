---
layout: post
title:  "Efficient Primal-Dual Algorithms for MapReduce"
date:   2022-01-08
author: koosaga
tags: [algorithm, graph theory]
---

# Efficient Primal-Dual Algorithms for MapReduce

## 1. Introduction
MapReduce 라는 프로그래밍 프레임워크는 대용량의 데이터를 처리하는 데 있어서 높은 성능을 보여주고, Apache Hadoop과 같은 오픈 소스 구현체들을 통해서 실용적으로도 그 가치를 증명하였다. 이 글에서는 그래프 최적화 문제를 푸는 데 효과적인 방법 중 하나인 Primal-Dual Method를 MapReduce 프레임워크에서 적용하는 새로운 프레임워크를 소개하고, 이를 통해서 Densest Subgraph Problem의 Near-linear time algorithm을 얻고자 한다. 정확히는, 이 글에서는 $O(\frac{\log n}{\epsilon^2})$ 번의 MapReduce iteration을 통하여 Densest Subgraph problem의 $(1 + \epsilon)$ approximation을 찾는다. 이는 일반적인 복잡도 모델에서 $O(m \frac{\log n}{\epsilon^2})$ 에 대응되고, near linear time이다.

Densest subgraph problem은 그래프가 주어졌을 때, $f(X) = \frac{E(G[X])}{V(G[X])}$ 를 최대화하는 $X \subset V(G), X \neq \emptyset$ 을 찾는 문제이다. 이제부터 $D^* = \max_{X \subset V(G), X \neq \emptyset} f(X)$ 라고 하자. 이 문제는 이론적 및 실용적인 면에서 모두 중요하다. 실용적 중요성은 몇 가지를 쉽게 짐작할 수 있겠지만, 조금 비자명한 응용으로는 스팸 링크 판별, 거리 쿼리 인덱싱, 생물정보학 등이 있다.

Densest subgraph problem의 이론적 중요성은 문제 자체의 단순함에서 나온다고도 생각할 수 있고 그 역시 합당하지만, 더 강력한 이유를 한 마디로 설명하자면 다음과 같다: Densest subgraph problem은 Bipartite $b$-matching보다 어렵고, Directed flow보다는 쉽다. Goldberg는 1984년 Densest subgraph problem을 parametric search를 사용해 Maximum flow로 환원하는 방법을 찾았으며, Directed flow는 Madry의 subquadratic algorithm보다 빠른 알고리즘이 알려져 있지 않다. 한편 Bipartite $b$-matching은 Densest subgraph problem으로 환원할 수 있으며, Near linear time algorithm이 알려져 있다. 이러한 것을 토대로 Static/Dynamic한 경우에 있어 Densest subgraph problem을 효율적으로 해결하는 것은 Directed flow가 near-linear domain에 들어가는 문제인지, 그보다 어려운 문제인지를 판별하는 데 가까이 다가갈 수 있는 중간점이 될 수 있다.

Competitive Programming에서, Densest subgraph problem은 [Hard Life](https://www.acmicpc.net/problem/3611) 라는 문제로 NEERC 2006에 출제된 바가 있다. 당시 대회에서는 아무도 풀지 못했으며, Goldberg의 알고리즘을 사용하면 해결할 수 있는 형태의 제한이다. Goldberg의 알고리즘은 현재 기준으로는 전형적인 Minimum cut/Maximum flow의 프레임워크 안에서 유도할 수 있기 때문에, 당시 참가자들이 느끼는 난이도와 현재 참가자들이 느끼는 난이도에는 차이가 있을 수 있다.

## 2. Linear Programming
먼저 이 문제를 해결하는 선형 계획법 모델링에 대해서 알아보자. $v \in V$ 에 대해서 $x_v \in \{0, 1\}$ 은 $v$ 가 $X$ 에 포함되는지 나타내고, $e \in E$ 에 대해서 $y_e \in \{0, 1\}$ 은 $e$ 의 양 끝점이 모두 $X$ 에 포함되는지 나타낸다. 모든 변수는 정수여야 할 때, 이를 다음과 같은 최적화 문제로 표현할 수 있다.

*(P1)* Maximize $\sum_{e} y_e / \sum_{v} x_v$ s.t

* $y_e \le x_v$ for all $e \in E$ incident on $v$
* $\sum_{v} x_v \ge 1$
* $x_v, y_e \ge 0$

이 식은 LP도 아니고 변수도 정수이니, 쉽게 변형해야 한다. 먼저 $x_v, y_e$ 가 정수여야 한다는 조건을 제거하고, $x_v, y_e$ 라는 항 전체를 $\sum_{v} x_v$ 로 나눠보자. 이 경우 문제는 다음과 같이 변형된다:

*(P2)* Maximize $\sum_{e} y_e$ s.t

* $y_e \le x_v$ for all $e \in E$ incident on $v$
* $\sum_{v} x_v \le 1$
* $x_v, y_e \ge 0$

두 번째 조건은 정확히는 $\sum_{v} x_v = 1$ 로 표현하는 것이 맞겠지만, 최적해는 항상 $\sum_{v} x_v = 1$ 가 맞다고 일반성을 잃지 않고 가정해 줄 수 있다. 이제 이 식은 LP이니, 다항 시간에 해결할 수 있다.

두 번째 최적화 문제로 변형하기까지 Leap of faith가 필요하였다. 고로 두 최적화 문제가 동치임을 따로 증명해야 한다.

**Lemma 1 ([Charikar 2000](https://link.springer.com/chapter/10.1007/3-540-44436-X_10)).** 임의의 정점 집합 $X \subseteq V, X \neq \emptyset$ 에 대해 두번째 LP의 값은 $f(X)$ 이상이다.

**Proof.** $f(X)$ 의 값을 가지는 assignment를 구성하자. $x_v$ 는 $v \in X$ 일 때 $\frac{1}{X}$ 이며 아니면 $0$ 이다. $y_e$ 는 모든 끝점에 대해 $x_v = \frac{1}{X}$ 이면 $\frac{1}{X}$ 이고 아니면 $0$이다. 이 assignment는 두 번째 프로그램의 모든 조건을 만족하며 이 때 $\sum_{e} y_e = f(X)$ 이다. $\blacksquare$

**Lemma 2 ([Charikar 2000](https://link.springer.com/chapter/10.1007/3-540-44436-X_10)).** 값 $v$ 를 가지는 위 LP의 해가 주어질 때, $f(X) \geq v$ 를 만족하는 집합 $X$ 가 존재하고, 다항 시간에 구성할 수 있다.

**Proof.** 값 $v$ 를 가지는 해를 $x, y$ 라고 하자. 실수 $r \geq 0$ 에 대해 집합 $V(r) = \{i  x_i \geq r\}, E(r) = \{i  y_i \geq r\}$ 라고 정의하자. 일반성을 잃지 않고 $y_e = min_{v \in incident(e)} x_v$ 라고 하면, $E(r)$ 은 $V(r)$ 에 의해서 Induce된 간선 집합에 대응된다.

먼저 $X$ 의 존재성을 증명하자. 우리는 $E(r)/V(r) \geq v$ 인 실수 $r \in [0, max(x_i)]$ 이 존재함을 보인다. 아래 두 사실을 관찰하자:
* $\int_{0}^{\max(x_i)} V(r) dr = \sum_{v}x_i \le 1$
* $\int_{0}^{\max(x_i)} E(r) dr = \sum_{e}y_e  = v$

고로, 만약에 그러한 실수가 없었다고 하면

$\int_{0}^{\max(x_i)} E(r) dr < v \int_{0}^{\max(x_i)} V(r) dr \le v$

로 가정에 모순이다. 서로 다른 집합의 개수는 서로 다른 $x_i$ 의 개수와 동일하니, 실제 집합을 찾는 것은 $r$ 에 모든 가능한 $x_i$ 를 대입해 봄으로써 $O(nm)$ 시간에 가능하고, 스위핑을 사용하면 $O(n \log n + m)$ 까지 줄일 수 있다. $\blacksquare$

Lemma 1, 2에 의해 위 LP는 $D^*$ 를 계산하며, Lemma 2에 의해 LP를 해결하면 다항 시간에 $f(X) = D^*$ 인 정점 집합 $X$ 를 계산할 수 있다. 고로 이 문제를 해결하는 새로운 다항 시간 풀이를 유도할 수 있다.

이제 P2의 Dual Program을 구성하자. $\alpha_{ev}$ 를 첫 번째 조건에 대응되는 Dual variable, $D$ 를 두 번째 조건에 대응되는 Dual variable이라고 하면:

*(P3)* Minimize $D$ s.t

* $\sum_{v \in incident(e)} \alpha_{ev} \ge 1$ for all $e \in E$
* $\sum_{e \in incident(v)} \alpha_{ev} \le D$ for all $v \in v$
* $D, \alpha_{ev} \ge 0$

마지막으로 Strong duality에 의해 P3의 최적해는 $D^*$ 이다.

## 3. Solving the Dual efficiently

최종적으로 우리는 Densest subgraph problem을 P3에 나온 것과 같은 간단한 LP로 변환할 수 있었다. $D$ 를 고정하고 $D^* \le D$ 인지를 판별하는 결정 문제를 생각해 보자. 이 결정 문제는 최적화가 필요없으며, LP의 조건을 모두 만족하는 해가 존재하는가 만을 판별하면 된다. 결정 문제를 해결할 수 있다면 $D^*$ 는 이분 탐색으로 해결할 수 있으니 (이후 이에 대해 자세히 다룰 것이다) 결정 문제로 문제를 변형시켜 생각해 보자.

다음 조건을 만족하는 벡터들의 Polytope을 $P(D)$ 라고 하자:
* $\sum_{e \in incident(v)} \alpha_{ev} \le D$ for all $v \in v$
* $D, \alpha_{ev} \ge 0$

이 때, 우리가 풀고자 하는 결정 문제는, 다음 조건을 만족하는 $x \in P(D)$ 가 존재하는가를 찾는 문제이다.

* $\sum_{v \in incident(e)} \alpha_{ev} \ge 1$ for all $e \in E$

우리는 이 문제를 해결하기 위해 *Multiplicative Weight Update Framework* (MWU Framework) 의 힘을 빌린다. 다음과 같은 종류의 결정 문제를 생각해 보자:
* **$COVERING$**: $r \times s$ 행렬 $A$ 와, $\mathbb{R}^s$ 의 convex set인 $P$ 가 주어진다. $x \in P$ 에 대해 $A x \geq 0$ 이 항상 만족함이 보장될 때, $Ax \geq 1$ 을 만족하는 $x \in P$ 가 존재하는가?

우리가 풀고자 하는 결정 문제는 정확히 $COVERING$ 에서 다루는 인스턴스 중 하나임을 볼 수 있다. 이 결정 문제를 해결하기 위해서는 다음과 같은 효율적인 오라클이 필요하다.

* **$ORACLE(y)$**: $r$ 차원 벡터 $y \geq 0$ 이 주어질 때, $C(y) = \max_{z \in P}y^T A z$ 를 해결하라.

$a_i$ 를 행렬 $A$ 의 row vector들이라고 할 때 *width* $\rho$ 는 $\rho = \max_{i} \max_{x \in P} a_i x$ 로 정의된다. 이 때 다음 사실이 알려져 있다.

**Theorem 3 (abridged).** $\frac{3 \rho^2 \log r}{\epsilon^2}$ 번의 Oracle 호출으로 $Ax \geq (1 - \epsilon)1$ 을 만족하는 해를 찾거나, $Ax \geq 1$ 이 infeasible함을 판별하는 알고리즘이 존재한다.
**Proof.** 길어서 Appendix A에서 계속한다. $\blacksquare$

결론적으로, 우리는 효율적인 Oracle을 구성할 수 있다면 MWU Framework를 통해서 전체 문제를 해결할 수 있다.

여기서 문제가 하나 있는데, $\alpha_{ev} \le D$ 이기 때문에 width가 $2D$까지 갈 수 있다는 것이다. 사실, 위 결정 문제의 해가 존재한다면, $\alpha_{ev} \le 1$인 해 역시 존재한다. 고로 Polytope $P(D)$에 $\alpha_{ev} \le 1$ 이라는 조건만 추가해주면 width를 2로 bound 시킬 수 있기 때문에 큰 문제가 되지 않는다. 하지만 이후 서술할 기술적 문제로 조건을 1로 걸 수는 없다. 대신 적당한 작은 정수 $q \geq 1$ 에 대해서 Polytope $P(D, q)$ 를 다음과 같이 정의하자.

* $\sum_{e \in incident(v)} \alpha_{ev} \le D$ for all $v \in v$
* $\alpha_{ev} \le q$
* $D, \alpha_{ev} \ge 0$

이 때 $A$ 의 width는 $2q$ 이하임을 쉽게 볼 수 있다. 굳이 $q = 1$ 이 아닐 이유는 이후 소개한다. 이제 Polytope $P(D, q)$ 에 대한 선형 시간 Oracle이 존재함을 보인다.

**Lemma 4.** $ORACLE(y)$ 는 선형 시간에 계산할 수 있다.
**Proof.** 식을 풀어서 쓰면, $\alpha \in P(D, Q)$ 에 대해 $\sum_{v} \sum_{e \in incident(v)} y_e \alpha_{ve}$ 를 최대화하는 문제임을 알 수 있다. $\alpha_{ve}$ 는 모두 $q$ 보다 작아야 하며, $\sum_{e \in incident(v)} \alpha_{ve} \le D$ 여야 한다. 이 조건들이 모두 독립이기 때문에 이것은 아주 쉽게 해결할 수 있다.

$r = \lfloor D / q \rfloor$ 라고 하면, 각 정점에 인접한 간선들 중 $y_e$ 가 가장 큰 $r$ 개에 대해서는 $\alpha_{ve} = q$, $r+1$ 번째에 대해서는 $\alpha_{ve} = D - qr$ 로 두면 된다. $r$ 번째 큰 수는 선형 시간에 찾을 수 있으니 (Selection algorithm) 전체 문제 역시 선형 시간에 계산할 수 있다.  $\blacksquare$

결론적으로 우리는 결정 문제를 해결할 수 있다.

**Theorem 5.** 임의의 수 $D \geq 0$, 정수 $q \geq 1$, $\epsilon \in [0, 1]$ 에 대해서, $O(m \frac{\log m}{\epsilon^2})$ 시간에 $Dual(D, q)$ 가 infeasible하거나, 모든 간선에 대해 $\sum_{v \in incident(e)} \alpha_{ve} \geq 1 - \epsilon$ 을 만족하는 해 $\alpha$ 를 찾을 수 있는 알고리즘이 존재한다.

## 4. Binary search for $D^*$
이분 탐색을 Naive하게 할 경우, 문제 해결에 Extra log factor가 더 붙게 된다. 이를 해결하는 방법이 논문에 정확하게 나와 있지는 않다. 일단 내 생각에는 대략 이러한 뜻인 것 같고, 논문의 내용이 설령 다르더라도 내 방법대로 하면 될 것 같다. 개략적으로 말하면, $\epsilon$ 의 크기가 현재 이분 탐색을 진행하고 있는 구간의 크기에 비례하게끔 할 수 있다. Theorem 3을 조금 더 정확히 파고 들어가면, Thm 3에서 $Ax \geq 1$ 이 infeasible함을 판별하였다면 이 정보는 항상 믿어도 되지만, $Ax \le (1 - \epsilon) 1$ 임을 판별하였더라도 문제가 infeasible할 수 있다. 하지만, 만약 틀리다 하더라도 이 때의 해를 $\frac{1}{1 - \epsilon}$ 만큼 Scaling하면, 결국 결정 문제가 $\frac{D}{1 - \epsilon}$ 에서는 확실히 feasible함을 알 수 있다. 이분 탐색의 하한과 상한을 $L$ 이라고 했을 때 어떠한 상수에 대해서 $\epsilon = cL/m$ 이라고 하자. 결정 문제를 $D = $ (하한) + $(1 - c)/2 L$ 정도의 위치에서 해결했다면 매번 구간의 길이가 $(1 + c) / 2$ 배 된다. 고로 $\epsilon$ 도 해당 속도만큼 증가하고, 이에 따라 매 순간 $\frac{1}{\epsilon^2}$ 의 합은 $\frac{m^2}{L^2} \times \frac{1}{c^2} (1 + ((1+c)/2)^{-2} + ((1+c)/2)^{-4} + \ldots + ((1 + c)/2)^{2\frac{\log (m/ \epsilon)}{\log{(1+c)/2}}})$ 와 같은 형태의 식이 나오게 된다. $c$ 가 적당한 상수라면, 이는 $O(\frac{1}{\epsilon^2})$ 가 되고, 결론적으로 이분 탐색을 추가해도 전체 문제를 푸는 asymptotic complexity는 동일함을 알 수 있다.

## 5. Rounding Step: Recovering the Densest subgraph
들어가기 앞서 Theorem 3의 statement를 더 보강한다. 간략히 말해서, Theorem 3은 $Ax \geq 1$ 인지를 판별할 뿐만 아니라, $Ax \geq \lambda 1$ 가 성립하는 최대 $\lambda$ 를 근사적으로 찾아줄 수도 있다는 내용이다. 즉, 일종의 최대화 문제를 풀 수 있다.

**Theorem 3 (full).** $\frac{3 \rho^2 \log r}{\epsilon^2}$ 번의 Oracle 호출으로 $Ax \geq (1 - \epsilon)1$ 을 만족하는 해를 찾거나, $Ax \geq 1$ 이 infeasible함을 판별하는 알고리즘이 존재한다. 만약 전자의 경우, 이 알고리즘은 다음 조건을 만족하는 벡터 $y$를 반환한다:
 * $\lambda^* \times \sum y_i \geq (1 - \epsilon) Oracle(y)$

이 때, $\lambda^* = \max\{\lambda  Ax \geq \lambda 1 \land x \in P\}$ 이다.

이를 감안했을 때, 우리가 위 단계에서 찾은 해는 다음 조건을 만족한다.

**Proposition 6.** 임의의 $\epsilon \in [0, 1/3]$ 과 $q \geq 1$ 에 대해, 값 $\tilde{D}$ 와 해 $(\alpha, y)$ 는 다음 조건을 만족한다.
 * $\tilde{D} \in [D^* (1 - \epsilon), D^* (1 + \epsilon)]$
 * $\sum_{e}y_e \ge (1 - 3 \epsilon) Oracle(y)$

**Proof.**  자명한 편이지만, 서술하기 위해서는 위에서 적은 이분 탐색에 대한 정확한 명세가 필요하다. 이 부분이 설명하기 까다로운 편이니 생략한다. 다만, $(1 - 3 \epsilon)$ 인 이유는 $\lambda$ 에 $1 - \epsilon$이 붙고 해에 $1 - \epsilon$ 이 붙으며 $(1 - 3 \epsilon) \le (1 - \epsilon)^2$ 이기 때문이다.

이제 이 Dual solution을 통해서 Primal solution을 찾아야 한다. 하지만 이를 가로막는 몇가지 기술적 문제가 존재한다. 문제의 근원은, MWU framework를 사용하기 위해 $A$ 의 width를 bound해야 했고, 고로 $\alpha_{ev} \le q$ 라는 부등식을 추가해야 했기 때문이다. 해당 부등식을 추가한 상태 그대로 다시 Dual을 취했을 때 얻는 문제는 다음과 같다:

(P4) Find $y, z, x$ such that $\frac{\sum_{e} y_e}{\sum_{e, v} (D x_v + q z_{ve})} \geq 1$ where $y_e \le min(x_u + z_{ev}, x_v + z_{ev})$

여기서 원래 Primal에 존재하지 않던 행은 $z$ 라는 벡터이다. 사실, 최적해의 Dual을 구할 경우 $z = 0$ 이 항상 성립하여서 이것은 크게 문제가 되지 않으나, 우리가 지금 구하는 것은 $\epsilon-$ approximate 해이고 여기서는 $z \neq 0$ 일 수 있다. 이 때는, 우리가 Lemma 2에서 사용했던 Rounding 방법을 적용하기가 까다로워진다.

이럼에도 불구하고, 우리는 $q = 2$ 일 때 $\epsilon$-approximate solution을 토대로 Densest subgraph problem의 $\epsilon$-approximate solution을 만들 수 있음을 보인다. 여기서 주목해야 할 점은 우리가 Dual solution을 토대로 Primal solution을 찾음을 보이는 것이 아니라, 그냥 문제의 답이 되는 집합 $X$ 를 바로 찾는다는 것이다. 정확히는, Primal solution을 실제로 찾기는 하지만, 이것을 Lemma 2의 방법을 사용해서 Rounding하는 것이 불가능하기 때문에, 이 Rounding 역시 새로운 방법으로 진행하고 이를 하나의 방법으로 압축하여 설명하는 것이다.

먼저, $P(D, q)$ 에 대한 Oracle $Oracle(y)$ 를 어떻게 계산하는지 다시 짚어보자.$r = \lfloor D/q \rfloor, s = D - qr$ 이라고 하고, $v$ 에 인접한 간선들의 $y_e$ 값을 감소순으로 정렬한 것을 $y_1(v) \ge y_2(v) \ge \ldots \ge y_n(v)$ 라고 하면:

$Oracle(y) = \sum_{v} (\sum_{k = 1}^{r} q y_k(v) + s y_{r + 1}(v))$

가 성립한다. 여기서 $Oracle(y)$ 가 $y$ 에 대한 linear function임에 주목하자.

이제 Theorem 3의 Output $y$ 을 토대로 집합을 복원한다. $Ax \geq \lambda 1$ 의 해가 아니라 $y$ 만을 사용함에 유의하라. 먼저 몇 가지 처리를 진행한다. Proposition 6이 성립하는 한 $y$ 라는 해를 적당히 변형해 줄 수 있다. 고로 계산량을 줄이고 서술을 간단하게 하기 위한 몇 가지 기술적 과정을 거친다.

 * $Y = \max_e y_e$ 라고 하면, 모든 $y$ 값에 $\frac{1}{Y}$ 를 곱해서 $Y = 1$ 을 맞춰주자.
 * $y_e \le \epsilon / m^2$ 일 경우 이 두 값이 $\sum_e y_e$ 와 $C(y, D, q)$ 에 기여하는 양은 모두 $\epsilon / m$ 이하이다. 이들의 값을 0으로 맞춰줄 경우, Proposition 6의 상수만 $\sum_{e}y_e \ge (1 - 4 \epsilon) Oracle(y)$ 으로 바꿔주면 여전히 성립한다.
 * $y_e$ 를 가장 가까운 $(1 + \epsilon)$ 의 지수승으로 round down한다. 이는 답을 최대 $(1 + \epsilon)$ 배 바꾼다.  Proposition 6의 상수만 $\sum_{e}y_e \ge (1 - 6 \epsilon) Oracle(y)$ 으로 바꿔주면 여전히 성립한다.

이 과정을 모두 거치면, 서로 다른 $y_e$ 값의 수는 최대 $O(\frac{\log m}{\epsilon})$ 개 존재함을 확인할 수 있다.

이제 Lemma 2의 증명과 유사하게 진행한다. 임의의 $\gamma \geq 0$ 에 대해서, $I(z)$ 를 $z \geq \gamma$ 이면 $1$, 아니면 $0$ 인 indicator라고 두자.  $G(\gamma) = (V, E(\gamma))$ 를 $y_e \geq \gamma$ 인 간선 $e$ 로 구성한 서브그래프라고 하자. $d_v(\gamma)$ 는 $G(\gamma)$ 에서 $v$ 의 차수 ($v$ 에 인접한 간선의 수) 로 정의된다. 즉, $d_v(\gamma) = \sum_{e \in incident(v)} I(y_e)$ 이고, $E(\gamma) = \sum_e I(y_e)$이다. 추가적으로, $H_v(\gamma) = \sum_{k = 1}^{r} I(y_k(v)) + \frac{s}{q} I(y_{r+1}(v))$ 라고 두자.

**Lemma 7.** $G(\gamma)$ 가 비지 않고, $E(\gamma) \geq q(1 - 6\epsilon)\sum_{v} H_v(\gamma)$ 인 $\gamma$ 가 존재하며, 이를 $O(\frac{m\log m}{\epsilon})$ 시간에 찾을 수 있다.

**Proof.** 다음과 같은 식이 성립한다.
* $\sum_{e} y_e = \int_{\gamma = 0}^{1} E(\gamma) d \gamma$
* $Oracle(y) = q \int_{\gamma = 0}^{1} \sum_{v} H_v(\gamma) d \gamma$
* $\sum_{e}y_e \ge (1 - 6 \epsilon) Oracle(y)$

Lemma 7의 모순을 가정할 경우 위 세 식을 모두 만족시킬 수 없다. 서로 다른 $y_e$ 값의 수가 $O(\frac{\log m}{\epsilon})$ 이고, 고로 서로 다른 $G(\gamma)$의 개수도 그 만큼이기 때문에 모든 $\gamma$ 를 전부 시도해 볼 수 있다. 고정된 $\gamma$ 에 대해서 위 식을 판정하는 것은 $O(m)$ 시간에 가능하다 (Lemma 4). $\blacksquare$

**Theorem 8.** $\epsilon \in (0, 1/12)$ 에 대해서, $f(X) \geq D^* (1 - \epsilon)$ 인 비지 않은 $X$ 를 $O(\frac{m \log m}{\epsilon^2})$ 에 찾을 수 있다.

**Proof.** Theorem 3과 Lemma 7에 의해, 조건을 만족하는 $\gamma$ 를 $O(\frac{m \log m}{\epsilon^2})$ 에 찾을 수 있다. $V_1$ 이라는 집합을 $y_{r + 1}(v) \geq \gamma$ 를 만족하는 정점들의 집합으로 정의하고, $V_2 = V - V_1$ 로 정의한다. $H_v(\gamma)$ 는 $v \in V_1$ 에 대해서 $D / q$, $v \in V_2$ 에 대해서 $d_v(\gamma)$ 이다. 고로 $\sum_v H_v(\gamma) = D/q V_1 + \sum_{v \in V_2} d_v(\gamma)$ 이 성립한다.

$G(V_1, E_1)$ 을 $V_1$ 의 Induced subgraph라고 하자. 먼저 $E_1 \geq E(\gamma) - \sum_{v \in V_2} d_v(\gamma)$ 이다. $E(\gamma) - E_1 \cap E(\gamma)$ 에 속한 간선들은 $v \in V_2$ 에 최소 하나의 정점이 인접해 있기 때문이다. 고로 $E(\gamma) - E_1 \cap E(\gamma) \leq \sum_{v \in V_2} d_v(\gamma)$ 이고 전개 후 $E_1 \cap E(\gamma) \le E_1$ 을 사용하며 위 식이 유도된다. 전개를 계속하면

$E_1 \geq E(\gamma) - \sum_{v \in V_2} d_v(\gamma) \geq q(1 - 6 \epsilon) (\frac{D}{q} V_1 + \sum_{v \in V_2} d_v(\gamma)) - \sum_{v \in V_2} d_v(\gamma)$

이에 따라, 만약 $q (1 - 6\epsilon) \geq 1$ 이면 $E_1 \geq (1 - 6\epsilon) D V_1$ 이다.

이제 $E_1 > 0$ 임을 보인다. 만약 $\sum_{v \in V_2} d_v(\gamma) = 0$ 이면, $E_1 \geq E(\gamma)$ 인데 $G(\gamma)$ 가 Lemma 7에 의해 비지 않으니 $E_1 > 0$ 이다. 그렇지 않다면, $q(1 - 6 \epsilon) (\frac{D}{q} V_1 + \sum_{v \in V_2} d_v(\gamma))$ 가 0 초과이다. 고로 $E_1 > 0$ 이다. $\blacksquare$

$q = 2$ 를 설정해야 하는 이유가 최종적으로 여기서 밝혀진다: $q (1 - 6\epsilon) \geq 1$ 이 성립해야 위 등식을 보일 수 있기 때문이다.

## Appendix A. MWU framework
이제 MWU Framework를 소개하면서 Theorem 3의 증명을 완성한다. 이를 설명하기 위해 먼저 *Expert Problem* 이라는 toy problem을 소개하고, 이에 대한 해결책을 논의한다.

**Expert problem.** $n$ 명의 주식 전문가가 있다. 매일 아침 전문가들은 *오늘 xx전자가 오르는가?* 라는 문제에 대해서 YES/NO 로 예상을 준다. 당신은 이 예상에 따라서 *오늘 xx전자가 오르는가?* 에 YES 혹은 NO 라는 답을 한다. 이후 하루가 끝나게 되면, xx전자의 주가가 올랐는지 내렸는지를 확인할 수 있다. 당신의 목표는, 10년이 지났을 때 이들 전문가 중 가장 예상을 적게 틀린 전문가와 비슷한 횟수로 틀리는 것이다.

**Solution: Randomized Weighted Majority Algorithm.** 각각의 전문가 $1 \le i \le n$에 대해서 이 전문가가 지금까지 틀린 횟수를 $m_i$ 라고 하자. $p_i = e^{-\frac{\epsilon m_i}{2}}$ 라고 할 때, 매일 아침 전문가 $i$ 를 $\frac{p_i}{\sum p_i}$ 의 확률로 뽑은 후 해당 전문가의 의견을 따른다. 하루가 끝나고, 틀린 전문가들에 대해서 $m_i$ 를 증가시킨다.

증명을 위해 각 값을 매 시간에 대해서 기록하자. $p_i(t)$ 를 $t$ 일 오전 결정을 내릴 때의 $p_i$ 라고 하자. $m_i(t)$ 를 $t$ 일이 지난 후 $m_i$ 의 값이라고 하자. 즉, $i$ 번 전문가가 $t$ 일에 걸쳐서 틀린 횟수이다. 비슷하게, $m(t)$ 를 알고리즘이 $t$ 일에 걸쳐서 틀린 횟수라고 하자.

**Theorem A1.**  $t \geq \frac{3 \ln m}{\epsilon^2}$ 에 대해서, $E[\frac{m(t)}{t}] \le \frac{m_i(t)}{t} + \epsilon$ 이 모든 $i \in [n]$ 에 대해 성립한다.

**Proof.** 다음을 정의한다.
* Potential function $\Phi(t) = \sum_{i \in [n]} p_i(t + 1)$ 를 정의한다.
* $I(t)$ 를 $t$ 번째에 알고리즘이 틀렸을 경우 1, 맞았을 경우 0인 indicator로 정의한다.
* $S(t) \subseteq [n]$ 을 $t$ 번째에 틀린 답을 낸 전문가의 집합이라고 하자.

이 때, 정의에 의해 $E[I(t)] = \sum_{i \in S(t)} \frac{p_i(t)}{\sum_{i \in [n]} p_i(t)}$ 이고, 기댓값의 선형성에 의해 $E[m(t)] = \sum_{t^\prime \in [t]} E[I(t)]$ 이다.

이제 다음 식을 도출할 수 있다.

$\Phi(t) = \sum_{i \in [n]} p_i(t + 1) = \sum_{i \in [n]} e^{\frac{-\epsilon m_i(t)}{2}} =  \sum_{i \in S(t)} e^{\frac{-\epsilon m_i(t-1)+1}{2}} +  \sum_{i \notin S(t)} e^{\frac{-\epsilon m_i(t-1)}{2}}
\\=\sum_{i \in S(t)} e^{\frac{-\epsilon m_i(t-1)+1}{2}} +  \sum_{i \notin S(t)} e^{\frac{-\epsilon m_i(t)}{2}}
\\=\sum_{i \in S(t)} p_i(t) e^{\frac{-\epsilon}{2}} +  \sum_{i \notin S(t)} p_i(t)
\\=\sum_{i \in S(t)} p_i(t) e^{\frac{-\epsilon}{2}} +  \sum_{i \notin S(t)} p_i(t)
$

$\epsilon \in (0, 1)$ 이라면 $e^{-\frac{\epsilon}{2}} \le 1 - \frac{\epsilon}{2} + \frac{\epsilon^2}{6}$ 이 성립한다 (테일러 급수 생각). 고로

$\Phi(t) \le \sum_{i \in S(t)} p_i(t) (1 - \frac{\epsilon}{2} + \frac{\epsilon^2}{6})+  \sum_{i \notin S(t)} p_i(t)
\\\le \sum_{i \in [n]} p_i(t) (1 + \frac{\epsilon^2}{6}) - \frac{\epsilon}{2} \sum_{i \in S(t)} p_i(t)
\\= \sum_{i \in [n]} p_i(t) (1 + \frac{\epsilon^2}{6}) - (\sum_{i \in [n]} p_i(t)) \times \frac{\epsilon}{2} E[I(t)]
\\= \sum_{i \in [n]} p_i(t) (1 + \frac{\epsilon^2}{6} - \frac{\epsilon}{2} E[I(t)])
\\= \sum_{i \in [n]} p_i(t) e^{\frac{\epsilon^2}{6} - \frac{\epsilon}{2} E[I(t)])}$ ($1 + x \le e^x$)

고로, $\Phi(t) \le \Phi(t - 1) e^{\frac{\epsilon^2}{6} - \frac{\epsilon}{2} E[I(t)]}$ 이다. 이를 연립하면 $\Phi(t) \le \Phi(0)e^{\frac{\epsilon^2}{6}t - \frac{\epsilon}{2} E[m(t)]}$ 를 얻는다.

여기서 $\Phi(0) = n$ 이고, 모든 $i \in [n]$ 에 대해 $\Phi(t) \geq p_i(t + 1)$ 이니

$e^{-\frac{\epsilon}{2} m_i(t)} \le n e^{\frac{\epsilon^2}{6}t - \frac{\epsilon}{2} E[m(t)]}$

양변에 로그를 취하고 $\frac{\epsilon t}{2}$ 로 나누면

$E[\frac{m(t)}{t}] \le \frac{2 \ln n}{\epsilon t} + \frac{\epsilon}{3} + \frac{m_i(t)}{t}$

$t \geq \frac{3 \ln n}{\epsilon^2}$ 로 두면 위 식을 얻을 수 있다. $\blacksquare$

**Generalized Expert Problem.** 변형된 Expert Problem에서는 $n$ 명의 주식 전문가가 있다. 매일 xx전자의 주식에 대해서 $P = m$ 개의 이벤트가 일어날 수 있다. 만약 내가 $i$ 번째 전문가의 의견을 따랐고, 그 후 $j$ 번 이벤트가 일어났다면, $M(i, j)$ 라는 비용을 지불하게 된다. 이 때, $M$ 이라는 함수는 계산 가능하고, $P$ 는 무한집합일 수 있다. 원래 Expert Problem과 비교하면, $P = \{$오른다, 내린다$\}$ 이고, $M(i, j)$ 는 $i$ 가 내린 의견이 $j$ 와 일치하지 않을 경우 1이다. 이 문제의 풀이는 앞에서 제시한 Randomized Weight Update Algorithm과 동일하다. 다만 $m_i$ 가 틀린 횟수를 나타내지 않고, 지금까지 전문가 $i$ 의 말을 따랐을 때 지불할 비용의 합으로 정의된다. 즉, $x(t)$ 가 $t$ 일의 결과라고 하면, $m_i = \sum_{t^\prime \in [t]} M(i, x(t^\prime))$ 이다.

**Theorem A2.** $\rho = \max(\max_{i \in [n], x \in P} M(i, x), 1)$ 라고 할 때, $t \geq \frac{3 \rho^2 \ln m}{\epsilon^2}$ 에 대해서, $E[\frac{m(t)}{t}] \le \frac{m_i(t)}{t} + \epsilon$ 이 모든 $i \in [n]$ 에 대해 성립한다.

**Proof.** 사실상 A1과 동일한 증명이다. 아래에 그 증명을 적지만, 글을 읽는 독자들은 연습 삼아 직접 해 보는 것을 추천한다.

* Potential function $\Phi(t) = \sum_{i \in [n]} p_i(t + 1)$ 를 정의한다.
* $I(t)$ 를 $t$ 일에 지불할 비용을 나타내는 확률변수로 정의한다.

정의에 의해 $E[I(t)] = \sum_{i \in [n]} \frac{p_i(t)M(i, x(t))}{\sum_{i \in [n]} p_i(t)}$ 이고, 기댓값의 선형성에 의해 $E[m(t)] = \sum_{t^\prime \in [t]} E[I(t)]$ 이다.

이제 다음 식을 도출할 수 있다.

$\Phi(t) = \sum_{i \in [n]} p_i(t + 1) = \sum_{i \in [n]} e^{\frac{-\epsilon m_i(t)}{2}}
\\=  \sum_{i \in [n]} e^{\frac{-\epsilon (m_i(t-1)+M(i, t))}{2}}
\\\le \sum_{i \in [n]} e^{\frac{-\epsilon (m_i(t-1)+\rho )}{2}}
\\= \sum_{i \in [n]} p_i(t) e^{\frac{-\epsilon \rho}{2}}
\\\le \sum_{i \in [n]} p_i(t) (1 - \frac{\epsilon \rho}{2} + \frac{\epsilon^2 \rho^2}{6})
\\\le \sum_{i \in [n]} p_i(t) (1 + \frac{\epsilon^2 \rho^2}{6} - \frac{\epsilon}{2}E[I(t)])
\\\le \sum_{i \in [n]} p_i(t) e^{\frac{\epsilon^2 \rho^2}{6} - \frac{\epsilon}{2}E[I(t)]}$


고로, $\Phi(t) \le \Phi(t - 1) e^{\frac{\epsilon^2 \rho^2}{6} - \frac{\epsilon}{2} E[I(t)]}$ 이다. 이를 연립하면 $\Phi(t) \le \Phi(0)e^{\frac{\epsilon^2 \rho^2}{6}t - \frac{\epsilon}{2} E[m(t)]}$ 를 얻는다.

여기서 $\Phi(0) = n$ 이고, 모든 $i \in [n]$ 에 대해 $\Phi(t) \geq p_i(t + 1)$ 이니

$e^{-\frac{\epsilon}{2} m_i(t)} \le n e^{\frac{\epsilon^2 \rho^2}{6}t - \frac{\epsilon}{2} E[m(t)]}$

양변에 로그를 취하고 $\frac{\epsilon t}{2}$ 로 나누면

$E[\frac{m(t)}{t}] \le \frac{2 \ln n}{\epsilon t} + \frac{\epsilon \rho^2 }{3} + \frac{m_i(t)}{t}$

알고리즘이 $\epsilon$ 을 입력으로 받았다면, 위 알고리즘에 대해서는 $\epsilon^\prime = \frac{\epsilon}{\rho^2}$ 를 설정한 후 실행시키자. 모든 단위를 $\epsilon^\prime$ 으로 환원하면:

$E[\frac{m(t)}{t}] \le \frac{2 \ln n}{\epsilon^\prime t} + \frac{\epsilon^\prime \rho^2 }{3} + \frac{m_i(t)}{t}$

$t \geq \frac{3 \ln n}{\epsilon^{\prime 2} \rho^2}$ 로 두면 위 식을 얻을 수 있다. 이는 $t \geq \frac{3 \rho^2  \ln n}{\epsilon^2}$ 이다. $\blacksquare$

이제 Theorem 3을 증명할 준비를 마쳤다.

**$Oracle(y)$**: $r$ 차원 벡터 $y \geq 0$ 이 주어질 때, $C(y)=\max_{x \in P}y^T Ax$ 를 해결하라.

**Algorithm A3.** LP의 각 Row ($A_i x \geq 1$)에 전문가를 대응시키고, 비용 함수를 $A_i x - 1$ 로 정의한다. 또한, 매 iteration의 outcome $x(t) = Oracle(p(t - 1))$ 으로 정의한다. Outcome은 그날 일어난 일에 대응된다. 이를 토대로 Randomized Weight Update Algorithm을 수행하고, 나온 벡터들을 $x(1), x(2) \ldots, x(t)$ 라고 하자 ($t \geq \frac{3 \rho \log r}{\epsilon^2}$). 만약 이 중
* $C(x(i)) < \sum_j{x(i)_j}$ 인 $x(i)$ 가 존재한다면, 모든 $x \in P$ 에 대해 $y^T Ax < y^T 1$ 인 $y \geq 0$ 가 존재한다는 뜻이다. $Ax \geq 1$ 의 해가 존재한다면 이는 일어날 수 없다. 고로 infeasible함을 반환한다.
* 그렇지 않다면, $\frac{\sum x(i)}{t}$ 가 조건을 만족하는 $x$ 이고, $\frac{\sum p(t)_i}{Oracle(p(t))}$ 를 최대화하는 $p(t)$ 를 $y$ 로 둔다.

**Theorem 3.** $\frac{3 \rho^2 \log r}{\epsilon^2}$ 번의 Oracle 호출으로 $Ax \geq (1 - \epsilon)1$ 을 만족하는 해를 찾거나, $Ax \geq 1$ 이 infeasible함을 판별하는 알고리즘이 존재한다. 만약 전자의 경우, 이 알고리즘은 다음 조건을 만족하는 벡터 $y$를 반환한다:
* $\lambda^* \times \sum y_i \geq (1 - \epsilon) Oracle(y)$

이 때, $\lambda^* = \max\{\lambda  Ax \geq \lambda 1 \land x \in P\}$ 이다.

**Proof.** Algorithm A3이 Theorem의 조건을 만족하는 알고리즘임을 증명한다. 먼저, $x$ 의 정당성을 증명한다. Theorem A2에 의해, $E[\frac{m(t)}{t}] \le \frac{m_i(t)}{t} + \epsilon$ 이 모든 $i \in [n]$ 에 대해 성립한다. 이 때, $E[\frac{m(t)}{t}] = \frac{1}{t} \sum_{t^\prime \in [t]} (\sum_{i = 1}^{m}{\frac{p_i(t^\prime)}{\sum_i p_i(t^\prime)} M(i, x(t^\prime))})$ 이다. 여기서, $\sum_{i \in [m]}{p_i(t^\prime) M(i, x(t^\prime))} = \sum_{i \in [m]}p_i(t^\prime) (A_i x(t^\prime) - 1) = p(t^\prime)^T A x(t) - p(t)^T$ 이기 때문에 Oracle의 가정에 따라 항상 0 이상이다. 결과적으로 $E[\frac{m(t)}{t}] \geq 0$ 이고, $-\epsilon \le \frac{m_i(t)}{t} \le \frac{\sum_{t^\prime \in [t]} M(i, x(t^\prime))}{t} =
\frac{\sum_{t^\prime \in [t]} A_i x(t^\prime) - b_i}{t}  = A_i \overline{x}(t) - 1$ 이 성립한다.

이제 $y$ 의 정당성을 증명한다. $\lambda$ 가 고정된 상황에서 $Ax - \lambda 1$ 이라는 결정 문제를 해결한다고 생각하자. $p_i = e^{-\frac{\epsilon m_i(t)}{2}}$ 으로 정의된다. 이는, $\lambda$ 에 무슨 값이 들어와도 모든 $p_t$ 에 일정한 값이 곱해지기 때문에 알고리즘이 답을 결정하는 데 영향을 주지 못한다. 고로 알고리즘이 반환하는 $x$ 값은 동일하다. 고로 $\lambda$ 값과 무관하게 우리가 받는 $p(t)$ 값은 상수 배 안에서 일정해야 함을 뜻한다. 즉, $\frac{\sum p(t)_i}{Oracle(p(t))}$ 는 $\lambda$ 에 독립적으로 고정되어 있다. 만약 $\lambda$ 값을 설정한 것에 따라서 전체 문제가 infeasible하다면, 앞선 내용에 따라서 A3은 무조건 그 반례를 제시해야 한다. 이 반례는 우리가 가지고 있는 고정된 $y$의 집합에서 존재하고, 그 중 가장 범용적인 반례는 $\frac{\sum p(t)_i}{Oracle(p(t))}$ 를 최대화하는 $p(t)$ 이다. 이 반례는 답이 성립하지 않는 최소의 $\lambda$, 즉 $\lambda^*$ 에서 Equality를 만족한다.$\blacksquare$
