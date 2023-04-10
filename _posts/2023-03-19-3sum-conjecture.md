---
layout: post
title: "Introduction to Hardness Proofs and 3SUM Conjecture"
date: 2023-03-19 17:00:00
author: karuna
tags: []
---

# Introduction to Hardness Proofs and 3SUM Conjecture

## Introduction

어떤 문제를 푸는 가능한 한 빠른 알고리즘을 찾아내는 것은 이론전산학 분야의 주된 관심사 중 하나입니다. 어떠한 문제든 우리가 원하는 만큼 빠른 알고리즘이 존재한다면 좋겠지만, 아쉽게도 이는 사실이 아닙니다. 예를 들어, $N$개의 정수를 크기 순으로 정렬하는 데에는 적어도 $\mathcal{O}(N\log N)$ 시간이 필요하다는 사실이 알려져 있습니다.

그렇다면 질문을 바꿔서, 어떤 문제를 해결하는 알고리즘의 시간 복잡도의 하한을 알 수는 있을까요? 이를 다른 말로 해당 문제의 **hardness**을 알아낸다고 말하는데, 이 역시도 대답하기 어려운 질문입니다. 7대 밀레니엄 문제 중 하나인 $P$-$NP$ 가설이 바로 어떤 문제들을 해결하는 다항 시간 알고리즘이 존재하는 지를 묻는 문제이며, 이 문제가 오랜 시간동안 난제로 남아있다는 사실 자체가 hardness에 대한 이야기가 일반적으로 굉장히 어려움을 설명합니다.

Hardness를 다루기가 쉽지 않기 때문에, 사람들은 기준이 될 수 있는 문제들을 선택해 이 문제들을 **빠르게 해결할 수 없다고 가정**하고, 이를 기준으로 다른 문제들의 hardness를 얻어내는 접근을 선택합니다. 즉 'A라는 문제가 다항 시간 내에 풀리지 않는다는 것을 가정할 때, B라는 문제 역시 다항 시간에 풀리지 않는다' 와 같은 이야기를 하는 것이 목표입니다. $P$-$NP$ 가설은 $NP$-complete라는 집합에 속한 문제들이 다항 시간에 풀리지 않는다는 것을 가정하며, 다항 시간과 비-다항 시간을 구분하는 경계가 됩니다. 이와 비슷하게 complexity class의 다른 경계들에 대해서도 이를 구분하는 가설들이 존재합니다. 이 글에서는 $\mathcal{O}(N^2)$ 경계를 다루는 **3SUM conjecture**에 대해 알아볼 예정입니다. 


## Definition

3SUM은 아래와 같이 정의되는 문제입니다.

**Definition (3SUM)** $N$개의 정수가 주어졌을 때, 합이 $0$이 되는 세 개의 정수 $a, b, c$가 존재하는 지를 판별하시오.

위 문제는 모든 $(a, b, c)$ 쌍을 확인해보는 방식으로 $\mathcal{O}(N^3)$ 시간에 해결할 수 있으며, 나아가서 $\mathcal{O}(N^2)$ 시간에 3SUM 문제를 해결하는 어렵지 않은 방법도 잘 알려져 있습니다. 하지만 3SUM 문제를 $\mathcal{O}(N^2)$ 시간보다 빠르게 해결하는 방법에 대해서는 딱히 알려진 바가 없습니다. 3SUM을 $\mathcal{O}(N^2)$ 시간보다 빠르게 해결하기 위한 다양한 시도가 있었으나, 좋은 방법이 잘 나오지 않았고 사람들은 다음과 같은 추측을 하게 됩니다.

***Conjecture*** 3SUM을 해결하는 $\mathcal{O}(N^2)$ 보다 빠르게 작동하는 알고리즘은 없다.

하지만 위 추측은 실제로 $\mathcal{O}(N^2)$보다 아주 약간 빠르게 작동하는 알고리즘이 발견되면서 깨지게 됩니다.

**Theorem (Baran et al., 2007)** 3SUM은 $\mathcal{O}(\frac{N^2 (\log \log N)^2}{(\log N)^2})$ expected time에 해결할 수 있다.

추측은 깨졌으나, 시간 복잡도에 $N^2$ 항이 여전히 남아있음에 주목합시다. 이제 사람들은 3SUM을 해결하는 알고리즘의 시간 복잡도에서 $N^2$ 항을 떼어낼 수 없다고 추측하였고, 이는 아직까지도 깨지지 않은 추측입니다. 

**Definition (Subquadratic algorithm)** 어떤 알고리즘이 적당한 $\epsilon > 0$에 대해서 $\mathcal{O}(n^{2 - \epsilon})$ expected time에 작동할 때, 이 알고리즘을 **subquadratic**하다고 한다.

**3SUM Conjecture** 3SUM을 해결하는 subquadratic한 알고리즘은 없다. 

위의 내용에서 시간 복잡도에 expected time이라는 조건이 추가로 붙었음에 유의합시다. 즉, deterministic하지 않은 알고리즘까지 고려하더라도 3SUM을 subquadratic하게 해결할 수 없다는 것이 3SUM conjecture의 내용입니다.


## Reduction

이런 추측이 존재하는 데에는 결국 다른 문제들의 hardness에 대해서 논의하기 위한 목적이 있음을 기억합시다. 3SUM 문제를 제곱 시간보다 빠르게 풀기 어렵다는 건 알겠는데, 이걸 이용해서 다른 문제에 대한 논의로 어떻게 이어가면 좋을까요? 여기서 **reduction** 이라는 개념을 사용하게 됩니다.

**Definition (Subquadratic reduction)** $A$랑 $B$라는 문제에 대해, $B$를 해결하는 subquadratic한 알고리즘을 이용해서 $A$를 subquadratic하게 해결할 수 있다고 하자. 이때 $A$에서 $B$로 가는 **subquadratic reduction**이 존재한다고 하며, $A \leq_{sq} B$라고 표기한다.

**Definition (3SUM-hard)** 문제 $A$에 대해 3SUM $\leq_{sq} A$ 가 성립하면, $A$는 3SUM-hard하다고 말한다.

즉, $B$를 해결하는 subquadratic한 알고리즘이 존재한다면 역시 $A$를 해결하는 subquadratic한 알고리즘도 존재한다는 의미가 됩니다. 이제, $A$가 3SUM일 때, $B$를 해결하는 subquadratic 알고리즘이 존재한다면 3SUM을 해결하는 subquadratic한 알고리즘이 존재하므로 3SUM conjecture에 모순이 발생합니다. 따라서 가설에 따라 $B$ 역시도 subquadratic하게 해결할 수 없다는 결론에 도달합니다. 즉 **3SUM에서 다른 문제로 가는 subquadratic reduction을 찾음**으로서 다른 문제들의 hardness를 증명할 수 있는 것입니다.

일반적으로, **oracle**의 개념으로 reduction을 설명하기도 합니다. $B$를 특정 시간에 해결할 수 있음을 가정하고, 이로부터 $A$를 특정 시간에 해결할 수 있는 알고리즘을 설계해 모순을 이끌어내는 것이 reduction의 목적이기 때문에, $B$를 해결할 수 있는 장치(oracle)가 주어지고 $A$를 해결하기 위해서 이 장치를 마음대로 호출하는 상황을 생각합니다. 예를 들어, 3SUM을 해결하기 위해 $B$를 해결하는 oracle을 크기 $\mathcal{O}(N^{1/3})$의 인스턴스로 $N$번 호출하는 알고리즘이 있다면, $B$를 해결하는 $\mathcal{O}(N^3)$보다 빠른 알고리즘이 없다는 사실을 이끌어낼 수 있는 것입니다. 이런 설명을 도입하면 $\mathcal{O}(N^2)$ 경계가 아닌 다른 lower bound에 대한 이야기도 가능하게 됩니다.

Subquadratic reduction을 이용해서 다음 문제가 3SUM-hard함을 보일 수 있습니다.

**Definition (3SUM')** 정수 집합 $A$, $B$, $C$에 대해서 $a + b = c$인 $a\in A$, $b\in B$, $c\in C$가 존재하는 지 판별하시오.

**Theorem** 3SUM $\leq_{sq}$ 3SUM'이고 3SUM' $\leq_{sq}$ 3SUM.

**Proof)** 
* 3SUM'를 푸는 subquadratic한 알고리즘이 존재한다고 가정하고, 3SUM을 푸는 subquadratic한 알고리즘이 존재함을 보이면 됩니다. 3SUM의 인스턴스인 정수 집합 $S$가 주어졌을 때, $A = S$, $B = S$, $C = -S$ 로 두고 3SUM'을 풀면 3SUM을 해결할 수 있습니다.
* 3SUM을 푸는 subquadratic한 알고리즘이 존재한다고 가정하고, 3SUM'을 푸는 subquadratic한 알고리즘이 존재함을 보이면 됩니다. 3SUM'의 인스턴스인 정수 집합 $A, B, C$가 주어졌을 때, 적당히 큰 자연수 $M$을 잡아서 $\{a + M : a\in A\}$, $\{b + 2M : b\in B\}$, $\{c - 3M : c\in C\}$를 모두 모은 뒤 3SUM을 풀면 3SUM'을 해결할 수 있습니다. $\square$

3SUM을 살짝 변형시킨 문제인 3SUM'은, 위 정리에 의하면 시간 복잡도의 관점에서 3SUM과 동등하게 어렵습니다. 즉, 어떤 문제가 3SUM-hard 함을 보이기 위해서 3SUM 대신 3SUM'에서 그 문제로 가는 subquadratic reduction이 존재함을 보여도 된다는 것을 의미합니다.


## Lower Bounds on Geometric Problems

3SUM은 다양한 기하 문제들로 subquadratic reduction을 찾을 수 있고, 이는 3SUM의 정의에서 등장하는 $a + b + c = 0$ 조건과 세 점의 collinearity를 연결짓는 것에서 출발합니다.

**Definition ($\text{COLLINEAR}$)** 평면 위에 있는 $N$개의 격자점에 대해서, 일직선 상에 놓인 세 점이 있는 지 판별하시오.

**Theorem** 3SUM $\leq_{sq}$ $\text{COLLINEAR}$. 따라서 $\text{COLLINEAR}$는 3SUM-hard하다.

**Proof)**

$\text{COLLINEAR}$를 푸는 subquadratic한 알고리즘이 존재한다고 가정하고, 3SUM을 푸는 subquadratic한 알고리즘이 존재함을 보이면 됩니다. 3SUM의 인스턴스인 정수 집합 $S$가 주어졌을 때, $x\in S$를 좌표평면 위의 점 $(x, x^3)$ 으로 보내는 변환을 생각합시다. $S$의 모든 원소들을 변환하면, 좌표평면 위에 $N$개의 점이 존재하게 됩니다. 이 점들 중 어느 세 점 $(a, a^3)$, $(b, b^3)$, $(c, c^3)$이 일직선 상에 놓여있을 조건을 계산해보면 $a + b + c = 0$과 같음을 알 수 있습니다. 따라서 이 점들에 대해서 $\text{COLLINEAR}$를 해결하면 3SUM을 해결할 수 있습니다. $\square$

**Definition** ($\text{GEOMBASE}$) $x$좌표가 $0$, $1$, $2$중 하나인 $N$개의 격자점에 대해서, $x$좌표가 모두 다르면서 일직선 상에 놓인 세 점이 있는 지 판별하시오.

**Theorem** 3SUM' $\leq_{sq}$ $\text{GEOMBASE}$. 따라서 $\text{GEOMBASE}$는 3SUM-hard하다.

**Proof)**

3SUM'의 인스턴스인 정수 집합 $A$, $B$, $C$가 주어졌을 때, $\{(0, a) : a\in A\}$, $\{(2, b) : b\in B\}$, $\{(1, c/2) : c\in C\}$를 모은 집합을 생각합시다. 이 점들에 대해서 $\text{GEOMBASE}$를 해결하면, $a + b = 2 \times c/2$ 인 세 정수 $a\in A$, $b\in B$, $c\in C$가 존재하는 지 알 수 있으므로, 3SUM'를 해결할 수 있습니다. $\square$

3SUM을 통해서 곧바로 얻을 수 있었던 위의 정리들을 이용하면 많은 기하 문제들의 3SUM-hardness를 얻어낼 수 있습니다. 아래의 두 문제는 언뜻 보기에 직관적이지 않지만, $\text{GEOMBASE}$를 이용하면 간단하게 hardness를 보일 수 있습니다.

**Definition** ($\text{SEPARATOR}$) 평면 위에 있는 $N$개의 선분에 대해서, 선분들과 모두 교차하지 않는 직선을 하나 그어 선분들을 두 개의 비어 있지 않은 집합으로 분리할 수 있는 지 판별하시오.

**Theorem** $\text{GEOMBASE}$ $\leq_{sq}$ $\text{SEPARATOR}$. 따라서 $\text{SEPARATOR}$는 3SUM-hard하다.

**Proof)**

GEOMBASE의 인스턴스 $S$에 대해서, 다음과 같은 변환을 진행합니다.
* 충분히 큰 실수 $M$에 대해서, $x = 0, 1, 2$에 대해 $(x, -M)$와 $(x, M)$를 잇는 세 선분을 그린다. $y = -M, M$에 대해 $(0, y)$과 $(2, y)$를 잇는 두 선분을 그린다.
* $(x, y)\in S$에 대해서, 좌우로 충분히 작은 $\epsilon > 0$ 길이의 구멍을 뚫는다.

이 변환의 결과는 $\mathcal{O}(N)$개의 선분을 갖는 $\text{SEPARATOR}$의 인스턴스가 되며, 여기에서 $\text{SEPARATOR}$를 풀면 $\text{GEOMBASE}$의 답을 얻을 수 있습니다. $\square$

**Definition** ($\text{STRIPS}$) 평면 위에 $N$개의 *strip*과, 목적 직사각형 하나가 주어진다. 여기서 *strip*은, 평행한 두 직선 사이의 무한한 공간과 같은 형태를 가지는 도형을 의미한다. 이 strip들이 주어진 직사각형을 완전히 덮는 지 판별하시오.

**Theorem** $\text{GEOMBASE}$ $\leq_{sq}$ $\text{STRIPS}$. 따라서 $\text{STRIPS}$는 3SUM-hard하다.

**Proof** 

$\text{GEOMBASE}$를 Point-Line duality를 이용해 변환한 문제를 생각하면, 평면 위에 있는 기울기가 $0$, $1$, $2$ 중 하나인 $N$개의 직선들에 대해서 한 점에서 만나는 세 직선이 존재하는 지 판별하는 문제가 됩니다. 각 기울기 $m$을 갖는 직선들 $c_m$개에 대해, $\mathbb{R}^2$에서 해당 직선들의 여집합으로서 얻어지는 $c_m+1$개의 strip을 생각합시다. 그러면 총 $c_0 + c_1 + c_2 + 3$개의 strip이 얻어지고, 이 strip들에 의해서 덮이지 않는 평면 상의 영역이 있는 지 판별하는 문제가 됩니다. 

즉, 충분히 큰 직사각형과 $c_0 + c_1 + c_2 + 3$개의 strip을 입력으로 하여 $\text{STRIPS}$를 해결하면, $\text{GEOMBASE}$를 해결할 수 있습니다. $\square$

$\text{STRIPS}$는 여러 covering problem들의 3SUM-hardness를 얻어내는 데에 base problem으로 사용되기도 합니다.

**Theorem** 다음 문제들은 모두 $\text{STRIPS}$에서 가는 subquadratic reduction이 존재한다.
* $\text{TRIANGLE COVER}$ - 평면 위에 $N$개의 삼각형과, 목적 삼각형 하나가 주어진다. $N$개의 삼각형들이 목적 삼각형을 완전히 덮는 지 판별하시오.
* $\text{TRIANGLE MEASURE}$ - 평면 위에 $N$개의 삼각형이 주어진다. 이 삼각형들의 합집합의 넓이를 구하시오.
* $k\text{-POINT COVER}$ - 평면 위에 $N$개의 반평면이 주어지고, 정수 $K$가 주어진다. 적어도 $K$개의 반평면에 포함되는 점 $P$가 존재하는 지 판별하시오.


## On Lower Bounds of Dynamic Data Structures

3SUM-conjecture를 이용해서 할 수 있는 이야기는 기하 문제에만 국한되어 있는 것은 아닙니다. Mihai Pătrașcu의 2010년 논문인 *Towards polynomial lower bounds for dynamic problems* 에서는 3SUM-conjecture를 이용해서 자료구조 문제들의 Hardness를 보이고 있습니다. 하나의 예시로, 다음과 같은 정리가 증명되어 있습니다.

**Theorem (Pătrașcu, 2010)** 3SUM conjecture가 참이라고 가정할 때, 방향 그래프 $G$에 대해서 간선의 추가 / 삭제 및 두 정점 $u, v$에 대해 $u$에서 $v$로 가는 경로가 존재하는 지 묻는 쿼리를 쿼리 당 $\mathcal{O}(N^{0.5 - \epsilon})$보다 빠르게 처리할 수 있는 자료구조는 없다.

위와 같은 논의를 하기 위해서는 3SUM의 문제 상황을 어떻게든 그래프 문제에 적용시킬 수 있는 장치(gadget)가 필요합니다. 앞선 챕터에서는 기하학의 공선점 조건을 3SUM의 합 조건에 대응시켜서 reduction을 찾을 수 있었다면, 그래프에 대해서는 마땅한 방법이 떠오르지 않습니다. 3SUM에서 정수들 사이의 합이 대수적인 관계를 나타낸다면 그래프는 조합론적인 구조이므로, 3SUM을 조합론적인 구조와 연결시켜줄 수 있는 중간 단계가 필요합니다.

**Definition (Convolution 3SUM)** 길이 $N$인 정수 수열 $A$가 주어졌을 때, $A[i] + A[j] = A[i + j]$인 $i, j$가 존재하는 지 판별하시오.

**Theorem (Pătrașcu, 2010)** 3SUM $\leq_{sq}$ Convolution 3SUM. 따라서 Convolution 3SUM은 3SUM-hard하다.

이 정리를 증명하는 핵심 아이디어는 핵심 아이디어는 3SUM의 인스턴스로 들어오는 정수들을 덧셈 구조를 *대부분* 보존하는 적당한 해시 함수 $h:\mathbb{Z} \rightarrow \mathbb{Z}$을 통해서 변환하는 것입니다. 예를 들어 $A[h(x)]$에 $x$를, $A[h(y)]$에 $y$를 넣고, $A[h(x + y)]$에 $x + y$를 넣었을 때 $h(x) + h(y) = h(x + y)$가 된다는 것을 보장할 수 있으면 Convolution 3SUM을 해결함으로서 3SUM을 해결할 수 있습니다.

위와 같은 linear hashing은 3SUM과 관련된 다른 문제들의 hardness를 증명하기 위해서 자주 사용되는 테크닉이므로 이를 이용한 증명을 소개합니다.

**Proof** Convolution 3SUM을 subquadratic한 시간에 풀 수 있는 oracle이 존재한다고 하고, 3SUM을 subquadratic expected time에 해결하는 randomized 알고리즘을 설계합니다. 

여기서는 $h(x)$를, 적당한 정수 $a, s, w$에 대해서 $a\cdot x\ (\bmod\ 2^w)$의 첫 $s$개의 bit로 설정합니다. 즉 어떤 정수 $x$에 대해서라도 $h(x)$의 결과값은 $0$부터 $2^s - 1$ 사이의 정수입니다. 이렇게 해시 함수를 잡은 중요한 이유로 임의의 두 정수 $x, y$애 대해서 $h(x) + h(y) = h(x + y)$이거나, $h(x) + h(y) + 1 = h(x + y)$가 성립한다는 사실이 있습니다.

이제 3SUM의 인스턴스인 크기 $N$인 정수 집합 $S$가 주어졌을 때, $S$의 각 원소 $x$를 $h(x)$로 해싱한 뒤, $2^s$개의 버킷 중 $h(x)$번째에 해당하는 버킷에 $x$를 넣습니다. 우리가 바라는 이상적인 상황은 $N$개의 정수들이 모두 다른 버킷에 들어가서 충돌이 발생하지 않는 경우인데, 이런 경우에는 Convolution 3SUM oracle을 적절히 사용해서 3SUM을 subquadratic한 시간에 해결할 수 있기 때문입니다.

* 길이 $8\cdot 2^s$의 정수 배열 $A$을 생각합니다. t번째 버킷에 유일하게 들어있는 원소 $x$에 대해서, $A[8t + 1] = A[8t + 3] = A[8t + 4] = x$로 설정하고, 배열의 나머지 원소들은 적당히 큰 정수 $M$으로 설정합니다.
* $A$에 대해서 Convolution 3SUM의 해가 존재한다면, 세 해는 각각 $8t_1 + 1$, $8t_2 + 3$, $8(t_1 + t_2) + 4$ 꼴이여야 하므로, 원래의 3SUM 인스턴스에서 $x + y = z$이고 $h(x) + h(y) = h(z)$인 해와 정확하게 대응됩니다.
* $x + y = z$이고 $h(x) + h(y) = h(z) - 1$인 해를 찾기 위해서는 $A[8t + 4]$ 대신 $A[8(t - 1) + 4]$에 $x$를 넣고 한 번 더 Convolution 3SUM을 해결하면 됩니다.

하지만 이런 일이 항상 일어나지는 않으므로 충돌이 일어나는 경우를 고려해야 합니다. $h(x)$를 정의하는 인자인 $a$를 랜덤하게 설정함으로써, 각 원소가 버킷으로 해싱되는 과정을 $N$개의 공을 $2^s$개의 상자에 랜덤하게 던지는 모델로 생각할 수 있습니다. 이때 $3N/2^s$개 이상의 공을 포함하는 상자들을 $X$라고 하면, $X$에 포함되어 있는 공들의 개수의 기댓값이 $\mathcal{O}(2^s)$ 임을 증명할 수 있습니다. 

이 $\mathcal{O}(2^s)$개의 원소들에 대해서는 투 포인터 기법을 이용해 각각 $\mathcal{O}(N)$ 시간에 이 원소를 포함하는 3SUM의 해가 존재하는 지를 확인해줄 수 있습니다. 따라서, $\mathcal{O}(N\cdot 2^s)$ 시간에 이 원소들을 제거해주고 나면, 나머지 버킷들은 최대 $3N/2^s$ 개의 원소를 포함하게 됩니다.

이제 모든 $(i, j, k) \in \{0, \cdots, 3N/2^s\}^3$ 에 대해서 버킷의 $i$번째, $j$번째, $k$번째 원소들로 이루어진 3SUM의 해가 존재하는 지를 위의 방법으로 Convolution 3SUM oracle을 이용해서 $\mathcal{O}(N^{2-\epsilon})$ 시간에 해결합니다. 각 oracle 호출의 크기가 최대 $3\cdot 2^s$이므로 이 과정에서 $\mathcal{O}((3N/2^s)^3 \cdot (3\cdot 2^s)^{2-\epsilon})$ 만큼의 시간을 사용하게 됩니다.

최종 시간 복잡도는 $\mathcal{O}(N\cdot 2^s + (3N/2^s)^3 \cdot (3\cdot 2^s)^{2-\epsilon})$이 됩니다. 여기서 충분히 작은 $\delta$에 대해 $2^s = \mathcal{O}(N^{1-\delta})$가 되도록 $s$를 설정해주면 3SUM을 해결하는 subquadratic한 알고리즘을 얻게 됩니다. $\square$

Convolution 3SUM의 3SUM-hardness를 이용해서 다음 그래프 문제의 hardness를 증명할 수 있습니다.

**Definition (Triangle Listing)** 무향 그래프 $G$와 정수 $k$가 주어졌을 때, $G$에서 $k$개의 삼각형을 찾으시오.

**Theorem (Pătrașcu, 2010)** Convolution 3SUM $\leq_{sq}$ Triangle Listing. 

마지막으로 Triangle Listing이 3SUM-hard하다는 사실을 이용해서 다음을 증명할 수 있습니다.

**Definition (Multiphase Problem)** $k$개의 집합 $S_1, \cdots, S_k \subseteq \{1, \cdots, n\}$가 주어지고, 집합 $T\subseteq \{0\cdots, n\}$가 주어질 때, $i\in \{1, \cdots, k\}$ 에 대해 $S_i\cap T = \varphi$ 인지 묻는 쿼리를 처리하는 다음 자료구조를 설계하시오.
1. $S_1, \cdots, S_k$가 주어졌을 때, $\mathcal{O}(nk\cdot \tau)$ 시간에 자료구조를 전처리한다.
2. $T$가 주어졌을 때 $\mathcal{O}(n\cdot \tau)$ 시간에 자료구조를 업데이트한다.
3. 각 쿼리를 $\mathcal{O}(\tau)$ 시간에 처리한다.

**Theorem (Pătrașcu, 2010)** 3SUM conjecture가 참이라고 가정할 때, $k = \Theta(n^{2.5})$로 설정하면 Multiphase Problem에서 $\tau$의 값은 최소 $\Omega(n^{0.5})$이다.

Multiphase problem은 set disjointness problem의 일반화된 버전으로, 굉장히 일반적인 문제로 reduce할 수 있는 강력한 base problem입니다. 예를 들어, 위에서 들었던 dynamic reachability 문제도 multiphase problem에서의 reduction을 구하는 방식으로 쉽게 증명할 수 있습니다.

**Theorem (Pătrașcu, 2010)** 3SUM conjecture가 참이라고 가정할 때, 방향 그래프 $G$에 대해서 간선의 추가 / 삭제 및 두 정점 $u, v$에 대해 $u$에서 $v$로 가는 경로가 존재하는 지 묻는 쿼리를 쿼리 당 $\mathcal{O}(N^{0.5 - \epsilon})$보다 빠르게 처리할 수 있는 자료구조는 없다.

**Proof)**

그러한 자료구조가 있다고 가정하고, multiphase problem에 대한 위 theorem에 모순이 있음을 보입니다. Multiphase problem의 인스턴스에 대하여, $S_1, \cdots, S_k$에 해당하는 $k$개의 정점과 $1, 2, \cdots, n$에 해당하는 $n$개의 정점, 그리고 하나의 sink로 이루어진 다음 그래프를 만듭니다.

1. $S_1, \cdots, S_k$가 주어졌을 때, $y\in S_x$인 $(x, y)$에 대해서 $x\rightarrow y$ 간선을 추가합니다. 이 과정에서 $\mathcal{O}(nk)$ 시간이 걸립니다.
2. $T$가 주어졌을 떄, $y\in T$인 $y$에 대해서 $y\rightarrow$sink 간선을 추가합니다. 이 과정에서 $\mathcal{O}(n)$ 시간이 걸립니다
3. 각 쿼리에 대해서 $i$가 주어졌을 때, $i$에서 sink으로 가는 경로가 존재하는 지 판별합니다. 이 과정에서 $\mathcal{O}(n^{0.5 - \epsilon})$ 시간이 걸립니다.

이제 Multiphase problem에서 $\tau = \mathcal{O}(n^{0.5 - \epsilon})$이 되므로 모순이 생깁니다. $\square$

## References

* Baran, I., Demaine, E. D., & Pǎtraşcu, M. (2007). Subquadratic Algorithms for 3SUM. Algorithmica, 50(4), 584–596.
* Patrascu, M. (2010). Towards polynomial lower bounds for dynamic problems. Proceedings of the 42nd ACM Symposium on Theory of Computing - STOC ’10.
