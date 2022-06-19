---
layout: post
title:  "Combinatorial Nullstellensatz"
author: leejseo
date: 2022-06-17 09:00
tags: [combinatorics]
---

## 1. 서론

그래프와 같은 여러 조합적인 대상의 수학적 성질은 이론 전산학 등의 분야에서 여러 방향으로 응용될 수 있다. Combinatorial Nullstellensatz는 그래프를 포함한 여러 조합적인 대상의 성질을 증명하는데에 응용될 수 있다. 이 글에서는 Combinatorial Nullstellensatz을 증명하고, 이후 여러 조합적인 대상에 이를 적용해본다.

## 2. Combinatorial Nullstellensatz

**Theorem.** (Combinatorial Nullstellensatz) 체 $\mathbb{F}$와 다항식 $f \in \mathbb{F}[x_1, x_2, \cdots, x_n]$ 이 있다고 하자. $\deg(f) = d = \sum_{i=1}^n d_i$ 이고, $\prod_{i=1}^n x_i^{d_i} \neq 0$ 라면, $\lvert L_i \rvert > d_i$인 $\mathbb{F}$의 부분집합 $L_1, L_2, \cdots, L_n$에 대해 $a_1 \in L_1, a_2 \in L_2, \cdots, a_n \in L_n$이 존재해 $f(a_1, a_2, \cdots, a_n) \neq 0$ 를 만족시킨다.

*Proof.* $n$에 대한 귀납법을 사용하자. $n = 1$인 경우에는 명백하므로, $n > 1$이라 가정하자.

일반성을 잃지 않고 $\lvert L_n\rvert = d_n + 1$을 가정할 수 있다. 이제, $d_n + 1$차 다항식 $f_n (x_n) = \displaystyle \prod_{t \in L_n} (x_n - t)$ 를 생각해볼 수 있다. 다항식 $h_n(x_n) = x_n^{d_n + 1}-f_n(x_n)$ 을 정의하면, $h_n(x_n)$의 차수는 $d_n$ 이하가 된다. 이제, $f$에서 $x_n^{d_n + 1}$을 $h_n(x_n)$으로 반복적으로 "교체"하여 얻어진 다항식 $\tilde f$ 를 생각하자. $x_n \in L_n$에 대해 $x_n^{d_n + 1} = h_n(x)$가 된다. 따라서, $x_n \in L_n$에 대하여 $f(x_1, x_2, \cdots, x_n) = \tilde f(x_1, x_2, \cdots, x_n)$이 된다. $x_1^{d_1} x_2^{d_2} \cdots x_n^{d_n}$의 계수는 $f$와 $\tilde f$에서 같음을 관찰할 수 있다. $\tilde f = \sum_{i = 0}^{d_n} g_i (x_1, x_2, \cdots, x_{n-1})x_n^i$ 로 다시 쓰고, $g_{d_n}$에 귀납 가설을 적용하면, $g_{d_n} (a_1, a_2, \cdots, a_{n-1}) \neq 0$ 인 $a_1 \in L_1, a_2 \in L_2, \cdots, a_{n-1} \in L_{n-1}$이 존재한다. $a_i$ 들을 고정하게 되면, $\tilde f$는 차수 $d_n$의 다항식이 되고, 고로 $\tilde f(a_1, a_2, \cdots, a_n) \neq 0$인 $a_n \in L_n$이 존재한다. $\square$

## 3. Examples

### 3.1. $p$-regular subgraph

**Example.** 루프 없는 그래프 $G$의 평균 차수가 $2p-2$ 보다 크고 최대 차수가 $2p - 1$보다 작다면, $G$는 $p$-regular subgraph를 가진다.

*Proof.* 모든 연산은 modulo $p$ 상에서 이루어진다고 가정하자. 각 간선 $e$에 대해 변수 $x_e$를 만들자. $x_e = 1$이면 간선을 택하는 것을, $x_e = 0$이면 그렇지 않는 것을 의미한다. 이제, 다음과 같은 다항식을 생각해보자:

$\displaystyle f(x_1, x_2, \cdots, x_{\lvert E \rvert }) = \prod_{v \in V} \left ( 1 - \left( \sum_{e \text{ is incident to }v}x_e \right)^{p-1} \right) - \prod_{e \in E} (1 - x_e).$

두 개의 항의 차수를 비교해보면, 첫 번째 항의 차수는 $\sum_{v \in V} (p-1) = \lvert V\rvert(p-1)$이 되며, 두 번째 항의 차수는 $\lvert E\rvert = 2 \sum_{v \in V} \deg(v) \ge 2\lvert V\rvert (p-1) > \lvert V\rvert (p-1)$이 된다. 따라서, $f$의 차수는 두 번째 항의 차수와 같고, 첫 번째 항의 차수 보다는 strict하게 크다.

$f$의 최고차항의 계수를 생각해보면, $x_1x_2 \cdots x_{\lvert E\rvert}$ 의 계수와 같고, 이는 $(-1)^{\lvert E\rvert -1}$ 로 non-zero가 된다. 따라서, Combinatorial Nullstellensatz에 의해 $f(\bar x_1, \bar x_2, \cdots, \bar x_{\lvert E\rvert }) \neq 0$인 $\bar x \in \{0, 1\}^{\lvert E\rvert }$ 을 찾을 수 있다.

$f(0) = 0$ 이므로, $\bar x \neq 0$이어서 두 번째 항은 0이 된다. 고로, 첫 번째 항이 0이 아님을 알 수 있다.

$\bar x_i = 1$인 간선만 택해서 만든 그래프 $G'$을 생각하자. 각 정점 $v$에 대해 $1 -( \sum_{e \text{ is incident to }v \text{ in G}} \bar x_e )^{p-1} \neq 0$ 이고, 이는 $\deg'(v)^{p-1} \neq 1$ 임을 의미한다. 페르마의 소정리에 의해 $\deg'(v) \equiv 0 \pmod p$ 가 되며, 최대 차수에 대한 조건에 의해 $\deg'(v) \in \{0, p\}$가 된다. 그리고 $\bar x \neq 0$ 이므로, $G'$에 속하는 간선은 하나 이상 존재한다. $\square$

### 3.2. Cauchy-Davenport Inequality

**Theorem.** 소수 $p$와 공집합이 아닌 $A, B \subseteq \mathbb{Z}_p$가 있다고 하자. 그러면, $\lvert A + B\rvert \ge \min(p, \lvert A \rvert + \lvert B\rvert  - 1)$이다.

*Proof.* $\lvert A\rvert + \lvert B \rvert > p$ 이면, $x \in \mathbb{Z}_p$ 에 대해 $A \cap (x - B) \neq \varnothing $ 이므로 $a + b = x$가 되는 $a \in A, b \in B$가 반드시 존재하여 명백하다. 고로, $\lvert A\rvert + \lvert B\rvert  \le p$를 가정하자.

귀류법으로 $\lvert A + B\rvert  \le \lvert A\rvert  + \lvert B\rvert - 2$를 가정하자. $\lvert C\rvert  = \lvert A\rvert  + \lvert B\rvert  - 2$ 이고, $A+B \subset C$ 인 $C$를 잡을 수 있다. 이제, 다항식 $f(x, y) = \prod_{c \in C} (x + y - c)$를 생각할 수 있다. $d_a = \lvert A \rvert  - 1, d_b = \lvert B \rvert - 1$ 에 대해 $x^{d_a} y^{d_b}$의 $f$ 에서의 차수는 ${d_a + d_b \choose d_a}$ 인데, $d_a + d_b < p$ 이므로 non-zero가 된다.

따라서, Combinatorial Nullstellensatz에 의해 $a \in A$, $b \in B$가 존재하여 $f(a+b) \neq 0$을 만족한다. 이는 $a + b \not \in C$를 의미하므로, $C$의 정의에 모순이다. $\square$

이 부등식을 이용하면, Erdos-Ginzburg-Ziv Theorem을 증명할 수 있다. 이는 프로그래밍 문제들로도 나올 정도로 유명한 정리이다.

- https://www.acmicpc.net/problem/18790
- https://www.acmicpc.net/problem/18791
- https://www.acmicpc.net/problem/18792

### 3.3. Erdos-Ginzburg-Ziv Theorem

**Theorem.** $n \ge 2$에 대해 $2n - 1$개의 정수가 있을 때, 이들 중 $n$개를 골라 합한 결과가 $n$의 배수가 되도록 하는 것이 항상 가능하다.

*Proof.* $n$에 대한 귀납법을 적용하자. Base case로 $n$이 소수일 때 성립한다고 가정하자. (이는 추후에 보일 것이다.) $n = ab$라 하면, 귀납 가설로 부터 $2a - 1$개의 수를 뽑으면, 그 중 $a$개의 합이 $a$의 배수가 되도록 할 수 있다. 이렇게, 합이 $a$의 배수인 $a$개의 수로 이루어진 쌍 $2b-1$개들을 묶어낼 수 있다. 이들의 합을 $S_1, S_2, \cdots, S_{2b-1}$이라 하자. 그러면, $S_1/a, S_2/a, \cdots, S_{2b-1}/a$ 가운데 합이 $b$의 배수인 $b$개의 수를 (귀납 가설에 의해) 묶어낼 수 있다. 이렇게 선택된 $S_i/a$ 형태의 수 각각을 원래와 같은 형태로 풀어쓰면, 결국 $ab$개의 수의 합이 $ab$의 배수이게 된다.

이제, $n $이 소수인 경우에 대해 보이자. 일단 모든 수들을 modulo $n$으로 생각하자. $a_1 \le a_2 \le \cdots \le a_{2n-1}$라 할 때, $a_i < a_{i+n-1}$을 가정할 수 있다. (아니라면, $a_i + a_{i+1} + \cdots + a_{i + n - 1} \equiv 0 \pmod n$)이 되기 때문이다. 이제, $A_i = \{a_i, a_{i+n-1}\}$로 정의하고, Cauchy-Davenport Inequality를 반복적으로 적용하면 $\lvert A_1 + A_2 + \cdots + A_{n-1}\rvert \ge \min(n, \sum_{i=1}^{n-1} \lvert A_i\rvert - (n-2))= \min(n, 2(n-1) - n - 2) = n$ 이 된다. 따라서, $-a_{2n - 1} \in A_1 + A_2 + \cdots + A_{n-1}$ 이 된다. $\square$

