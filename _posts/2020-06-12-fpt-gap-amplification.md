---
layout: post
title: "Parametrized inapproximability for Steiner Orientation by Gap Amplification"
author: koosaga
date: 2020-06-11
tags: [graph-theory, complexity-theory]
---

# Parametrized inapproximability for Steiner Orientation by Gap Amplification

이 글에서는 `k​-STEINER ORIENTATION` 문제와 `MAX (k, p)-DIRECTED MULTICUT` 문제에 대한 [FPT hardness result를 소개하는 논문을 정리한다](https://arxiv.org/pdf/1907.06529.pdf). 

먼저, `k-STEINER ORIENTATION` 문제는 다음과 같이 정의된다:

* 입력: *mixed graph* $G$ 와 $k$ 개의 terminal pair $T_G = \{(s_1, t_1), (s_2, t_2), \ldots, (s_k, t_k)\}$ (*mixed graph* 는 무방향 간선과 방향성 간선이 둘 다 존재할 수 있는 그래프를 뜻한다.)
* 출력: $G$ 의 모든 무방향 간선에 방향성을 주어서, $s_i \rightarrow t_i$ 로 가는 경로가 존재하는 쌍 $(s_i, t_i)$ 의 개수를 maximize하여야 한다.

$k$ 에 대해서 parametrize했을 때, 현재까지 $O(k)$-factor 보다 더 나은 approximation algorithm은 알려지지 않았다. 이 글에서는 이 문제에 대한 $O((\log k)^{o(1)})$-approximation이 존재하지 않음을 증명한다. 고로 이 문제는 $O(1)$-factor, $O(\log \log k)$-factor 등에 approximation하는 것이 불가능하다.

다음으로, `MAX (k, p)-DIRECTED MULTICUT` 문제는 다음과 같이 정의된다:

* 입력: 방향 그래프 $G$ 와 $k$ 개의 terminal pair $T = \{(s_1, t_1), (s_2, t_2), \ldots, (s_k, t_k)\}$ , 그리고 예산 제한 $p$
* 출력: $G$ 에서 최대 $p$ 개의 간선을 제거해서 $s_i \rightarrow t_i$ 로 가는 경로가 존재하지 않는 쌍 $(s_i, t_i)$ 의 개수를 maximize하여야 한다.

$k, p$ 모두에 대해서 parametrize했을 때, 이 글에서는 이 문제에 대한 $O(k^{\frac{1}{2} - \epsilon})$ 알고리즘이 존재하지 않음을 증명한다.

## Overview of the results

위 논문에는 5개의 Main theorem이 있다:

**Theorem 1.** $\beta(k) \rightarrow 0$ 이 계산 가능한 비증가 함수일 때, $\alpha(k) = (\log k)^{\beta(k)}$ 라고 하자. 주어진 $k$-STEINER ORIENTATION 이 다음 두 경우 중 하나임을 구분하는 것은 $k$ 에 대해 parametrize했을 때 $W[1]$-hard이다:

* 모든 $k$ 개의 terminal pair에 대한 경로가 존재하는 orientation이 가능
* 어떤 orientation에 대해서도 최대 $\frac{k}{\alpha(k)}$ 개의 terminal pair에 대한 경로만을 만족시킬 수 있음 

**Theorem 2.** $\beta(n) \rightarrow 0$ 이 계산 가능한 비증가 함수일 때, $\alpha(n) = (\log n)^{\beta(n)}$ 라고 하자. $W[1]$ Class가 false-biased FPT 알고리즘을 포함하지 않는 이상. 주어진 $k$-STEINER ORIENTATION 이 다음 두 경우 중 하나임을 구분하는 것을 다항 시간 안에 하는 것은 불가능하다:

* 모든 $k$ 개의 terminal pair에 대한 경로가 존재하는 orientation이 가능
* 어떤 orientation에 대해서도 최대 $\frac{k}{\alpha(n)}$ 개의 terminal pair에 대한 경로만을 만족시킬 수 있음 

**Theorem 3.** $k$-STEINER ORIENTATION 은 $W[1]$-complete이다.

**Theorem 4.** 임의의 $\epsilon > 0$ 과 함수 $\alpha(k) = O(k^{\frac{1}{2} - \epsilon})$ 이 주어질 때, 주어진 $(k, p)$-DIRECTED-MULTICUT이 다음 두 경우 중 하나임을 구분하는 것은 $k+p$ 에 대해 parametrize했을 때  $W[1]$-hard이다:

* 모든 $k$ 개의 terminal pair에 대한 경로를 차단하는 크기 $p$ 의 cut이 존재
* 어떠한 크기 $p$ 의 컷도 최대 $\frac{k}{\alpha(k)}$ 개의 terminal pair에 대한 경로만을 차단할 수 있음

**Theorem 5.** $NP \not\subseteq co-RP$ 임을 가정하면, $\epsilon > 0$ 이고 $\alpha(m) = O(m^{\frac{1}{2} - \epsilon})$ 일 때, 주어진 $(k, p)$-DIRECTED-MULTICUT이 다음 두 경우 중 하나임을 구분하는 것은 다항 시간 안에 불가능하다:

* 모든 $k$ 개의 terminal pair에 대한 경로를 차단하는 크기 $p$ 의 cut이 존재
* 어떠한 크기 $p$ 의 컷도 최대 $\frac{k}{\alpha(m)}$ 개의 terminal pair에 대한 경로만을 차단할 수 있음 ($m = E(G)$)

이 글에서는 Theorem 1과 Theorem 4의 증명을 살펴본다.

## Proof of Theorem 1

### Introduction: Gap Amplification

어떠한 $k$-STEINER ORIENTATION instance $I = (G, T_G)$가 주어질 때,  $R = (r_1, r_2, \ldots, r_k)$ 이며, $1 \le r_i \le k$ 인 정수 $k$-tuple을 생각해보자. 이러한 튜플로 가능한 경우는 $k^k$ 개 있다. $R$ 이 정해졌을 때, 다음과 같은 두 개의 레이어로 구성된 새로운 그래프를 생각해보자:

* 상단 레이어에는 $G$ 의 copy가 $k$ 개 있다. 각각을 $G_1, G_2, \ldots, G_k$ 라고 하자. $G_i(S, i), G_i(T, i)$ 는 각각 $G_i$ 에서 $(s_i, t_i)$ 번째 source-sink 쌍을 뜻한다. 
* 하단 레이어에는 $G$ 의 copy가 1개 있다. 이를 $H$ 라고 하자. $H(S, i), H(T, i)$ 는 각각 $H$ 에서 $(s_i, t_i)$ 번째 source-sink 쌍을 뜻한다. 
* 모든 $1 \le i \le k$ 에 대해 $G_i(T, r_i)$ 에서 $H(S, i)$ 로 가는 방향성 간선을 추가한다.

이제, 이 그래프에 $(G_i(S, r_i), H(T, i))$ 과 같은 $k$개의 source-sink pair를 지정한 후 $k$-STEINER ORIENTATION instance를 구성한다. 이를 $I^2$ 라고 하자. 

$R$ 로 가능한 모든 $k^k$ 개의 경우 중, 하나가 유일하게 정해졌다고 하자. 이 경우 $I^2$ 의 최적해의 기댓값 $X$의 상한은 다음과 같이 계산할 수 있다.

* 만약 $I$ 의 최댓값이 $k$ 라면, 모든 copy에 대해서 그냥 동일한 orientation을 주면 되기 때문에, 최적해의 기댓값 역시 $k$ 이다.
* $I$ 의 최댓값이 $k$ 미만이라면, 모든 $G_i$ 에 대해서 $G_i(S, j_i) \rightarrow G_i(T, j_i)$ 로 가는 경로가 없는 $1 \le j_i \le k$ 가 존재한다. 이를 $j_1, j_2, \ldots, j_k$ 라고 하고, $Y = \{i  j_i \neq r_i\}$ 라고 하자.
  * 기댓값의 선형성에 의해 $E[Y] = k - 1$ 임을 쉽게 보일 수 있다. 
  * 위쪽에서 끊기면 절대 연결이 불가능하니 $X \le Y$ 이다. 
  * $Y$ 와 상관없이 $H$ 에서 모든 쌍을 연결할 수 없으니 $X \le k - 1$ 이다.
* 종합하면 $E[X] \le p(Y \le k - 1) \times E[Y] +p(Y = k) \times (k - 1)$ 이라는 부등식을 얻을 수 있다. 이 때, $p(Y = k) = (\frac{k-1}{k})^k$ 이다. 식을 정리하면, $E[X] = (k - 1) - (\frac{k-1}{k})^k$ 이다.

랜덤성을 가정하고 *기댓값* 이라는 언어로 해석했지만, 이렇게 생각하지 말고 그냥 최적해가 저러한 조건을 만족했다고 생각해 보자. 초기 두 최적해의 Gap은 $\frac{k}{k - 1}$ 정도였지만, 한 번의 반복을 통해서 $\frac{k}{k - 1 - \frac{k-1}{k}^k}$ 로 줄여낼 수 있었다. 이를 같은 방법을 사용해서 계속 반복하면 Gap은 자랄 수 있을 때까지 커지지 않을까?

한편, 랜덤성에 대한 가정은 어렵지 않게 없앨 수 있다. 앞서 언급했지만 알고리즘이 $k$ 에 대한 FPT이기 때문에 하단 레이어에 $G$ 의 copy를 매우 많이 배치해도 되기 때문이다. 하단 레이어에 $G$ 의 copy를 $k^k$ 개 배치하고, 모든 가능한 튜플에 대해서 위와 같은 식으로 $H$ 를 구성하자. 이렇게 될 경우 $I$ 의 최적해가 $k$ 라면 $I^2$ 의 최적해는 $k^{k+1}$, $k-1$ 이하라면 $(k-1)k^k - (k-1)^k$ 가 된다. 

여기까지가 이 논문의 핵심을 이루는 Gap Amplication 테크닉의 개략적인 설명이다. 실제로 이를 Formal proof로 바꾸기 위해서는 그래프 빌딩 과정이 더 효율적이어야 하고, 그 외에도 보여야 할 부분들이 많다. 하지만 여기서는 직관적인 수준의 이해만 하도록 하고, 명확한 증명은 다음 단락에서 다루도록 하자.

### Step 1. Statement of Lemma and Proof of Main Theorem. 

**Lemma 1.** 파라미터 $q$와 $k$-STEINER ORIENTATION의 인스턴스 $I_1 = (G, T_G), k = T_G$ 가 주어졌을 때, 아래 조건을 만족하는 새로운 인스턴스  $I_2 = (H, T_H), k_0 = T_H$ 를 만드는 알고리즘이 존재한다. $S(I_1) = S(G, T_G)$ 는 최적해의 크기를 뜻한다.

* $k_0 = 2^{q^{O(k)}}$
* $\lim_{q \rightarrow \infty} k_0(q) = \infty$.
* $V(H) \le V(G) \times k_0^2$
* $S(G, T_G) = k$ 이면, $S(H, T_H) = k_0$을 만족.
* $S(G, T_G) < k$ 이면, 최소 $\frac{1}{k_0}$ 의 확률로 $S(H, T_H) \le \frac{k_0}{q}$ 을 만족 

이 알고리즘의 랜덤 버전은 $H$ 에 비례하는 시간 복잡도로 작동하고, 결정론적 버전은 $f(k, q) \times G$ 시간에 작동한다.

Lemma 1의 증명은 다음 단락으로 넘긴다. 이제 Lemma 1이 참이라고 가정하고, Theorem 1을 증명한다.

**Theorem 1.** $\beta(k) \rightarrow 0$ 이 계산 가능한 비증가 함수일 때, $\alpha(k) = (\log k)^{\beta(k)}$ 라고 하자. 주어진 $k$-STEINER ORIENTATION 이 다음 두 경우 중 하나임을 구분하는 것은 $k$ 에 대해 parametrize했을 때 $W[1]$-hard이다:

* 모든 $k$ 개의 terminal pair에 대한 경로가 존재하는 orientation이 가능
* 어떤 orientation에 대해서도 최대 $\frac{k}{\alpha(k)}$ 개의 terminal pair에 대한 경로만을 만족시킬 수 있음 

**Proof of Theorem 1.** 정의상, $k_0(q) \leq 2^{q^{ck}}$ 를 항상 만족시키는 $c$ 가 존재한다. 또한, $\beta(k) \rightarrow 0$ 이 비증가 함수이고, $k_0(q) \rightarrow \infty$ 이기 때문에, $\beta(k_0(q)) ck \leq 1$ 을 만족하는 충분히 큰 $q$ 도 존재한다. $k$ 가 주어질 때 이러한 $q$ 를 계산하자. 이제

$\alpha(k_0) = (\log k_0)^{\beta(k_0)} \le (q^{ck_0})^{\beta(k_0)} \le q$

를 만족한다. 이제 Lemma 1을 사용하여 $S(G, T_G) = k$ 이면 $S(H, T_H) = k_0$, $S(G, T_G) < k$ 이면 $S(H, T_H) \le \frac{k_0}{\alpha(k_0)}$ 을 만족시키는 $H$ 를 찾을 수 있다. 또한, $H$ 의 크기는 고정된 $k$ 에 대해 선형이다. 고로, 만약 저 두 경우를 구분할 수 있다면, $k$-STEINER-ORIENTATION 은 $k$ 에 대한 FPT이고, 가정에 모순이다. $\blacksquare$

### Step 2. Proof of Lemma 1

이제 Lemma 1에 대한 Formal proof를 소개한다. 

#### 레이어 구성

우리는 인스턴스 $(G, T_G)$ 가 주어졌을 때, 더 큰 인스턴스 $(H, T_H)$를 만들어서 $(G, T_G)$ 의 모든 쌍을 만족시킬 수 있으면 $(H, T_H)$ 도 모두 만족시킬 수 있지만, 하나라도 만족시킬 수 없으면 $(H, T_H)$ 에서 최대 $\frac{1}{q}$ 개의 쌍만 만족시킬 수 있게끔 하고 싶다. 

이를 위해, 우리는 인스턴스들의 무한 수열 $(H^1, T^1), (H^2, T^2), \ldots$ 을 구성할 것이다. $(H^1, T^1) = (G, T_G)$이다. 그 외 $(H^i, T^i)$ 는 $(H^{i-1}, T^{i-1})$ 의 구성에 따라서 귀납적으로 구성된다. 그 구성 과정은 다음과 같다.

$H^{i+1}$ 은 $(H^i, T^i)$ 의 $k$ 개의 복사본, 그리고 $(G, T_G)$ 의 몇 개의 복사본으로 구성된다. 이 때 취한 $G$ 의 복사본의 개수를 $p_{i+1}$라고 정의하며, 이 값이 정확히 얼마인지는 나중에 알아 볼 것이다. 아래에 깔린 것이 $(H^i, T^i)$ 의 복사본, 위에 깔린 것이 $(G, T_G)$ 의 복사본이다. 

![pic1](http://www.secmem.org/assets/images/gapamp/pic1.png)

각 복사본을 $G_R$ 이라고 부르자. 모든 $p_{i+1}$개의 복사본 $(G_R, T_{G_R})$ 에 대해서 다음과 같은 방식으로 $k$ 개의 간선늘 만든다. 각 $k$개의 source $G_R(S, j)$ 에 대응할 $H_i$ 의 sink $H_{j}^{i}(T, Choice_j = random(1, T_i))$를 골라 준 후, 골라 준 쌍들에 대해서 모두 그들을 잇는 간선을 아래에서 ($H_j^{i}$) 위로 ($G_R$) 올라오는 방향으로 만든다. $p_{i+1}$ 개의 copy에 대해서 $k$ 개의 간선을 추가했으니 이 과정 후 총 $p_{i+1}k$ 개의 간선이 만들어진다. 대응할 sink는 모든 source에 대해서 랜덤하게 고른다. 즉, 각 source에 대해서 가능한 간선의 경우의 수는 $T_i$ 개가 되고, 총 가능한 경우의 수는 $T_i^{k \times p_{i+1}}$ 가 되며, 이 모든 배정 중 하나를 랜덤하게 고르는 것이다. 

이렇게 $H^{i+1}$ 이라는 그래프를 형성한 후, 각각의 $G_R$ 에 대해서 $H_{j}^{i}(S, Choice_j)$ 와 $G_R(T, j)$ 를 source-sink pair로 추가한다. 이렇게 되면 각 $G_R$ 마다 $k$ 개의 pair가 생기고, 최종적으로 $p_{i + 1} \times k$ 개의 pair를 만들 수 있다.

마지막으로, $p_i$ 라는 값을 어떻게 설정하는 지 알아보자. 정의에 의해 $p_1 = 1$ 이다. $i \geq 1$ 에 대해, $p_{i+1} = O(k^4 q^{2k} p_i)$ 이다. 이렇게, 인스턴스들의 수열의 정의가 끝난다.

우리의 목표는, 적당한 $(H^M, T^M)$ 을 잡았을 때 우리가 처음에 원하던 성질이 만족되는 것이 가능하고 고로 Lemma 1이 증명되는 것이다. 이 목표를 이뤄 보자.

**Lemma 2.** $y_i = \frac{S(H^i, T^i)}{T^i}$ 라고 하자. $S(G, T_G) < k, y_i \geq \frac{1}{q}$ 일 경우, 최소 $\frac{1}{2}$ 의 확률로 $y_{i + 1} \le y_i - \frac{1}{2k}q^{-k}$ 가 만족된다.

**Proof of Lemma 2.** 먼저, $(s_j, t_j) \in T^i$ 쌍들에 대해서, 이들을 잇는 경로들은 서로 다른 $(G, T_G)$ 의 복사본 $i$ 개를 거치고, 또한 이 때 거치는 복사본들의 수열은 유일함을 관찰하자. 예를 들어, 맨 위 레이어에 깔려 있는 한 복사본의 $j$ 번 source-sink pair는 $H_{i}^{j}$ 방향으로 이어지게 되어 있다 (그 이후 어느 레이어를 방문하게 될지는 $Choice_j$ 의 선택에 따라 비슷하게 결정된다). 고로, $(s_j, t_j)$ 경로가 만족되기 위해서는, 이 $i$ 개의 복사본들에 대해서 모두 원하는 경로가 존재해야 한다. 

아래 층에 깔려있는 $H_{j}^{i}$ 를 모두 어떤 식으로 orient했다고 생각하자. 이 때 $H_{j}^{i}(S, v), H_{j}^{i}(T, v)$ 사이에는 경로가 있을 수도 있고 없을 수도 있다. 이 중 경로가 있는 모든 $v \in T_{j}^{i}$ 를 $C_j$ 라는 집합으로 부르자. 즉, $C_j = \{v  v \in T_{j}^{i}, \text{there exists path from } H_{j}^{i}(S, v), H_{j}^{i}(T, v) \text{ for some orientation yet to defined}\}$ 이다. $C_j$ 로 가능한 조합의 개수는 최대 $2^{T_i}$ 개이다. 튜플 $C = (C_1, \ldots, C_k)$ 를 *configuration* 이라고 정의하자. 가능한 *configuration* $C$ 의 경우의 수는 최대 $(2^{T_i})^k$ 이다. 어떠한 *configuration* $C$ 에 대해, $f_C = Choice \rightarrow [0, 1]$ 을 source-sink pair를 $Choice$ 에 따라서 선택했을 때 만족 가능한 terminal pair의 **최대** 비율이라고 정의하자. 즉, $f_C(Choice)$ 는, $H_{j}^{i}(S, Choice_j)$ 와 $G_R(T, j)$ 를 source-sink pair로 추가했을 때 만족되는 terminal pair의 비율의 최댓값이라고 볼 수 있다. 

$f_C$ 의 뜻에 대해서 고찰해 보자. $f_C$ 가 영향을 받는 인자는 정확히 다음 세 가지이다.

* $Choice$ 로 무엇을 고를 것인지 (인자로 주어짐)
* $C$ 가 어떻게 설정 되어 있는지 (인자로 주어짐)
* $G_R$ 을 어떻게 orient했는지 (이것을 잘 결정해서 만족시키는 비율을 **최대화** 해야 한다.)

한편, $f_C$ 는 $H_{j}^{i}$ 가 어떤 식으로 orient되어 있는지와는 상관이 없다. Configuration이 고정되면 내부 그래프의 orientation은 알 필요가 없기 때문이다. 귀납 단계에서 $H_{j}^{i}$ 에 대해 우리가 알고 싶은 것은 개개의 간선들이 어떻게 orient되어 있는지가 아니라 $C$ 에 저장된 정보를 알고 싶다. 고로 $C$는 $H_{j}^{i}$ 에서 필요한 정보를 간소화한 결과라고 생각할 수 있다. 이 간소화된 정보와, $G_R$ 의 orientation, 그리고 랜덤하게 배정된 $Choice$ 가 $f_C$ 를 결정한다. 이렇게 결정된 $f_C$ 는, $y_{i + 1}$ 에서 표현하려고 하는, $\frac{S(H^{i+1}, T^{i+1})}{T^{i+1}}$ 값을 표현한다. $G_R$ 을 orient하는 것은 우리의 몫이고, $Choice$ 는 랜덤하게 고르고, $C$ 는 $y_i$ 값과 연관이 있다고 이해할 수 있다.

이제 $H_1^i, H_2^i, \ldots, H_k^i $ 의 orientation이 정해졌다고 하자. Orientation이 정해졌으니 $C$ 역시 정해진다. 이제 $Choice$ 를 랜덤 변수로 했을 때, $f_C$ 의 기댓값의 상한을 계산한다. 이 단락부터는 위 Gap Amplication에서 기댓값 $X$ 의 상한을 계산하는 단락을 참고하면 좋을 것이다.

랜덤 변수 $Choice = (Choice_1 = random(1, T_i), Choice_2 = random(1, T_i), \ldots, Choice_k)$ 에 대해 때, $Y_j$ 는 $Choice_j \in C_j$ 임을 (즉, $H_{j}^{i}(S, Choice_j) \rightarrow H_{j}^{i}(T, Choice_j)$ 경로가 존재함을) 나타내는 확률 변수라고 하자. $Y = \sum_{j = 1}^{k} \frac{Y_j}{k}$ 라고 하면, $f_C(Choice) \le Y$ 가 항상 만족된다. $Choice_j \notin C_j$ 임은 아래 레이어에서 경로가 없어졌음을 뜻하기 때문이다. $E[Y_j] = E[Choice_j \in C_j] = \frac{C_j}{T_i}$ 이다. 이 값을 $c_j = E[Y_j]$ 라고 하면 $E[Y] = \sum_{j = 1}^{k}\frac{c_j}{k}$ 이다. 

한편, $\prod_{j = 1}^{k} c_j$ 의 확률로, 모든 $j$에 대해 $Choice_j \in C_j$ 가 만족된다. 이 때 $Y = 1$ 이다. 하지만, 가정에 의해 $S(G, T_G) < k$ 이다. 고로 $f_C(Choice) = 1$ 이 되는 것은 불가능하고,  실제로는 $f_C(Choice) \le 1 - \frac{1}{k}$ 를 만족한다. 이 간극을 통해서 Gap Amplication을 하자. 

$E[f_C(Choice) - Y] \leq -\frac{1}{k} \times \prod_{j = 1}^{k} c_j$ 

$E[f_C(Choice)]  \le \frac{1}{k} (\sum_{j = 1}^{k} c_j - \prod_{j = 1}^{k} c_j)$

이 때, $y_i$ 의 정의에 의해 $c_j \le y_i$ 이고, 또한 가정에 의해 $y_i \geq \frac{1}{q}$ 이다. $(\sum_{j = 1}^{k} c_j - \prod_{j = 1}^{k} c_j)$ 를 $c_i$ 에 대해 편미분하면, $(1 - \prod_{j \in [k], j \neq i} c_j)$ 이다. 이 값은 물론 0 이상이다. 고로, $c_j < y_i$ 일 경우 $c_j = y_i$ 라고 둬도 여전히 위 부등식이 성립하게 된다. 우변이 증가만 할 뿐이기 때문이다. 고로,

$E[f_C(Choice)] \le \frac{1}{k} (\sum_{j = 1}^k y_i - \prod_{j = 1}^{k} y_i) \le y_i - \frac{1}{k} q^{-k}$

가 모든 $C$ 에 대해서 성립한다. 기댓값은 모든 가능한 $T_i^k$ 개의 $Choice$ 에 대한 $f_C(Choice)$ 의 평균이다. 고로 모든 가능한 $T_i^k$ 개의 Choice에 대해서 복사본을 만들면, 다른 말로 $p_{i + 1} = T_i^k$ 로 두면, $y_{i + 1} \le y_i - \frac{1}{k}q^{-k}$ 가 성립함을 알 수 있다. 한편, 우리는 $p_{i + 1}$ 를 이것보다 작게 만들고 싶다. 다른 말로, 가능한 모든 $Choice$ 중에서 $p_{i + 1}$ 개만을 랜덤하게 샘플링해도, 높은 확률로 기댓값 (평균) 이 보존된다는 것을 보여야 한다. 이러한 성질을 다음과 같이 정의한다.

**Definition.** Multiset $X_H$ 가 $X$ 의 부분집합이라는 것은 ($X_H \subseteq X$) $X_H$ 의 모든 원소가 $X$에 속함을 뜻한다.

**Definition.** 유한한 multiset $X_H \in X$ 와 함수 $f : X \rightarrow \mathbb{R}$ 에 대해서 $E_{x \sim U(X_H)} f(x) = \frac{1}{X_H} \sum_{x \in X_H} f(x)$ 이다. 즉, $E_{x \sim U(X_H)} f(x)$ 는, $X_H$ 에서 임의의 원소 $x$ 를 uniform distribution으로 샘플링했을 때 $f(x)$ 의 기댓값을 뜻한다. 

**Definition 3.** $X \rightarrow [0, 1]$ 로 가는 함수들의 집합 $F$ 에 대해서, $\delta$-*biased sampler family* 는 모든 $f \in F$ 에 대해 다음 성질을 만족하는 multiset $X_H \subseteq X$ 를 뜻한다: $E_{x \sim U(X_H)} f(x) - E_{x \sim U(X)} f(x) \leq \delta$

**Lemma 4.** 주어진 $X, F, \delta > 0$ 에 대해, $X$에서 $O(\delta^{-2} \log (F))$ 개의 원소를 샘플링하면 (repetition을 허용하고, 각 원소마다 같은 확률로 독립적으로) 이 집합은 최소 $\frac{1}{2}$ 의 확률로 $\delta$-*biased sampler family* 를 이룬다.

**Proof of Lemma 4.** $M = 10 \delta^{-2} \log(F)$ 라고 두자. 각각의 $f \in F$ 에 대해서, size-$M$ sample $X_H \subseteq X$ 가 $E_{x \sim U(X_H)} f(x) - E_{x \sim U(X)} f(x) > \delta$ 를 만족하는 사건 $A_f$ 와 그 확률 $P(A_f)$ 의 상한을 계산한다. [Hoffeding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality) 에 의하여 $P(f) \le 2 exp(-2 \delta^{2} M)$ 이다. 여기에 우리가 고른 $M$ 을 대입하면, $P(f) \le 2 exp(-20 \log F) \le \frac{2}{F^{20}} \le \frac{1}{2F}$ 이다. $P(\cup_{f \in F} A_f) \le \sum_{f \in F} P(A_f) \le \frac{1}{2}$ 이다. Lemma 4가 증명된다.  $\blacksquare$

$F = \{f_C  C \text{ is a valid configuration }\}, \delta = \frac{1}{2k} q^{-k}$ 로 두자. $\log F \le T_i k \le p_i k^2$. $\delta^{-2} = {4k^2}q^{2k}$ 이니, $p_{i + 1}  = O(\delta^{-2} \log (F)) = O(k^4 q^{2k} p_i)$ 로 두면 Lemma 4에 의해 $y_{i + 1} \le y_i - \frac{1}{k} q^{-k} + \frac{1}{2k} q^{-k}$ 가 성립한다. 고로  Lemma 2가 증명된다. $\blacksquare$

이제, Lemma 1의 증명으로 다시 돌아온다. $B = 2kq^k$ 에 대해 $(H, T_H) = (H^B, T^B)$ 라고 두자.

* $k_0 = 2^{q^{O(k)}}$
  * $k_0 = p_B$ 로 한다. 정의는 아래에 있다. 
* $\lim_{q \rightarrow \infty} k_0(q) = \infty$.
  * 아래에서 보게 될 정의에 의해 자명하다. 
* $V(H) \le V(G) \times k_0^2$
  * $q \ge 2$ 를 가정하자. 그러면 $k \le q^k$ 이다.  $p_{i + 1} = p_i \times O(k^4 q^{2k})$ 였으니, $p_B = O((k^4 q^{2k}) ^ {2kq^k}) = q^{O(k^2q^k)} = 2^{\log q \times q^{O(k)}} = 2^{q^{O(k)}}$ 이다. $V(H) = \sum_{i = 0}^{B} p_i \times k \times V(G) \le (B + 1) \times V(G) \times 2^{q^{O(k)}}$ 이다. $B + 1 \le 2^B \le p_B$ 이니 $V(H) \le p_B^2 V(G)$ 이다.
* $S(G, T_G) = k$ 이면, $S(H, T_H) = k_0$을 만족.
  * 모든 $(G, T_G)$ 의 복사본을 최적으로 orient 시키면 되니 자명하다. 
* $S(G, T_G) < k$ 이면, 최소 $\frac{1}{k_0}$ 의 확률로 $S(H, T_H) \le \frac{k_0}{q}$ 을 만족 
  * $y_{i + 1} \le y_i - \frac{1}{2k}q^{-k}$ 를 만족하고, $y_0 \le 1$ 이니, Lemma 2에 의해 $y_B < \frac{1}{q}$ 이거나 $y_{B} \le 1 - \frac{2kq^k}{2k} q^{-k} \le 0$ 이다. 이 확률은 $2^{-B} \ge \frac{1}{p_B}$ 이다. 

고로 Lemma 1이 증명된다. $\blacksquare$

마지막으로, 이 증명에서 랜덤성을 사용하는 곳은 Lemma 4인데, $O(\delta^{-2} \log (F))$ 개의 원소를 샘플링하는 대신, 모든 가능한 $O(\delta^{-2} \log (F))$-tuple을 나열한 후 이 중 $\delta$-*biased sampler family* 를 찾아보는 방식으로 결정론적인 증명을 얻을 수 있다. 이렇게 하더라도 시간 복잡도는 $f(k, q) \times V(G)$ 가 됨을 확인해 볼 수 있다.

## Proof of Theorem 4

Theorem 4의 증명은 Theorem 1과 비슷한 구성이지만, 더 간단하다. Gap Amplication을 여러 번 거칠 필요 없이, 한 번만으로도 환원이 가능하다는 것이 큰 차이점이다. 

`MAX (k, p)-DIRECTED MULTICUT` 의 인스턴스는 $(G, T)$로 표시하고, $p$ 개의 간선을 지워서 분리할 수 있는 terminal pair의 개수는 $S(G, T, p)$ 로 표시한다.

### Step 1. Statement of Lemma and Proof of Main Theorem. 

**Lemma 5**. 파라미터 $q$ 와 `DIRECTED MULTICUT` 의 인스턴스 $(G, T_G, p), T_G = 4$ 가 주어졌을 때, 아래 조건을 만족하는 새로운 인스턴스 $(H, T_H, p_0), k_0 = T_H$ 를 만드는 알고리즘이 존재한다.

* $k_0 =\Theta(p\times q^2 \log q)$
* $p_0 = \Theta(p^2 \log q)$
* $E(H) = E(G) \times p_0 + O(k_0 \times p_0)$
* $S(G, T_G, p) = 4$ 이면, $S(H, T_H, p_0) = k_0$ 이 항상 만족됨
* $S(G, T_G, p) < 4$ 이면 $S(H, T_H, p_0) \le \frac{k_0}{q}$ 가 $\frac{1}{2}$ 이상의 확률로 항상 만족됨

이 알고리즘의 랜덤 버전은 $H$ 에 비례하는 시간 복잡도로 작동하고, 결정론적 버전은 $f(p, q) \times G$ 시간에 작동한다.

Lemma 5의 증명은 다음 단락으로 넘긴다. 이제 Lemma 5이 참이라고 가정하고, Theorem 4을 증명한다.

**Theorem 4.** 임의의 $\epsilon > 0$ 과 함수 $\alpha(k) = O(k^{\frac{1}{2} - \epsilon})$ 이 주어질 때, 주어진 $(k, p)$-DIRECTED-MULTICUT이 다음 두 경우 중 하나임을 구분하는 것은 $k+p$ 에 대해 parametrize했을 때  $W[1]$-hard이다:

* 모든 $k$ 개의 terminal pair에 대한 경로를 차단하는 크기 $p$ 의 cut이 존재
* 어떠한 크기 $p$ 의 컷도 최대 $\frac{k}{\alpha(k)}$ 개의 terminal pair에 대한 경로만을 차단할 수 있음

**Proof of Theorem 4.** $\epsilon > 0$ 을 고정하고, $L = \lceil \frac{2}{\epsilon} \rceil$ 이라 하자. 고로 $L \geq \frac{2}{\epsilon}$ 이다. `DIRECTED MULTICUT` 의 인스턴스 $(G, T_G, p), T_G = 4$ 가 주어졌을 때, Lemma 5에서 파라미터 $q = p^L$ 으로 설정하고 새로운 인스턴스를 만든다. 이 인스턴스의 $k_0, p_0$ 은 모두 $p, q$ 에 대한 함수이다. 만약에 $k = 4$ 일때 모든 경로를 차단할 수 있었다면 새 인스턴스에서도 그렇다. 그렇지 않다면, 새 인스턴스는 $\frac{k_0}{q} = \Theta(pq \log q) = O(p q^{1 + \frac{1}{L}}) = O(p^{L + 2})$ 개 이하의 terminal pair를 차단시킬 수 있다. $\frac{k_0}{\alpha(k_0)} = \Omega(k_0^{\frac{1}{2} + \epsilon})  = \Omega(p^{(2L+1) \times (\frac{1}{2} + \epsilon)})$ 이다. 고로 $(2L+1)(\frac{1}{2} + \epsilon) \geq L + 2$  임을 보이면 된다. 이항하면 $\epsilon \geq \frac{1.5}{2L + 1}$ 이다. $\epsilon \geq \frac{2}{L} \geq \frac{2}{L + 0.5}$ 이니 이는 참이다. 고로 Theorem 4가 거짓이면, Directed multicut with 4 pairs가 FPT이기 때문에 가정에 모순이다.

### Step 2. Proof of Lemma 5

![pic2](http://www.secmem.org/assets/images/gapamp/pic2.png)

$(G, T_G)$ 의 $M = 3(p + 1) \log q$ 개의 복사본 $(G_1, T_1), \ldots, (G_M, T_M)$ 을 만들자. 각각의 $M$ 개의 source에 대해서, 새로운 terminal pair를 하나 만들고, 이 terminal pair에 랜덤한 source-sink pair를 이어주자. 즉, 1 이상 4 이상의 정수 값을 가지는 random $M$-tuple $Choice = (choice_1 = random(1, 4), choice_2 = random(1, 4), \ldots, choice_M)$ 을 고른 후, 새로운 source-sink $s_{Choice}, t_{Choice}$에 대해서 $s_{Choice} \rightarrow S(G_i, r_i), T(G_i, r_i) \rightarrow t_{Choice}$ 로 가는 간선 두개를 만들어 주는 것이다. 이를 $k_0 = O(pq^2 \log q)$ 번 반복한다. 즉, source-sink 쌍을 $k_0$ 개 만든 후 각각에 대해서 랜덤하게 $2M$ 개의 간선을 이어주는 것이다. 마지막으로, 예산 $p_0 = 3p(p+1)\log q = pM$ 이다.

만약 $S(G, T_G, p) = 4$ 라면, $(G, T_G)$ 에서 해당 source-sink pair를 끊어준 방식을 그대로 $M$ 개의 복사본에 대해서 따라하면 $k_0$ 개의 쌍이 분리된다. 그 외 경우, 우리는 각 $(G, T_G)$ 에서 4개의 쌍을 모두 분리하기 위해 최소 $p+1$ 개의 간선을 끊어야 하고, 현재 예산 상에서 이를 최대 $3p \log q$ 개의 복사본에서 할 수 있다. 고로, 어떠한 해에 대해서도, 적어도 하나의 terminal pair가 분리되지 않은 복사본이 $3 \log q$ 개 존재한다.

Lemma 1의 증명과 똑같이, $4M$ 개의 복사된 source-sink pair에 대해서 해당 pair를 잇는 경로가 있는지 없는지를 나타내는 정보 $(C_1, C_2, \ldots, C_M)$을 ($C_i \subseteq [4]$) *configuration* 이라고 한다. *configuration* 은 모든 $4M$ 개의 source-sink pair의 부분집합이니, 모든 가능한 *configuration* 의 경우는 $2^{4M} = 16^M$ 이다. 이제, $f_C : Choice \rightarrow \{0, 1\}$ 은 $s_{Choice} \rightarrow t_{Choice}$ 로 가는 경로가 없으면 1, 아니면 0 인 함수로 정의하자. $S(G, T_G, p) < 4$ 일 경우, $C_i \neq \emptyset$ 인 쌍이 최소 $3 \log q$ 이다. 고로 $E_{Choice \sim U(Choice)} f_C(Choice) \le \frac{3}{4}^{3 \log q} \le 2^{-\log (2q)} \le \frac{1}{2q}$ 이다. 

이제 Lemma 4를 적용하자. $F = \{f_C  C \text{ is a valid configuration }\}, \delta = \frac{1}{2q}$ 로 두자. 가능한 configuration의 경우는 $16^M = 2^{O(p \log q)}$ 이다. 고로 $\log F \le O(p \log q)$ 이다. 고로 $\frac{1}{2}$의 확률로 위 알고리즘에서 분리된 terminal pair의 비율은 $E_{Choice \sim U(Choice)} f_C(Choice) + \frac{1}{2q} \le \frac{1}{q}$이다. $\blacksquare$

Derandomization은 위에서 사용한 것과 동일하게 sampler를 전체 탐색하면 된다. 하지만 이번 Lemma에서는 poly-time reduction이 가능해서, 그러한 방법을 원할 때에는 다른 방법이 필요하다. 이 부분은 이 글에서는 생략한다.