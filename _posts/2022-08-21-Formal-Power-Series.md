---
layout: post
title:  "Formal Power Series와 Operation의 빠른 계산 방법"
date:   2022-08-21
author: ainta
tags: [algorithm, FFT]
---


# Introduction

## Polynomial Ring

Field $\mathbb{F}$에 대해 **Polynomial Ring** $\mathbb{F}[x]$는 다항식(polynomial)들의 집합으로 정의되며, 다항식들은 각 계수 $p_i$가 $\mathbb{F}$의 원소인 $p = p_0 + p_1x + ... + p_mx^m$ 꼴로 표현됩니다. 여기에서 Ring이란 간단히 말해서 덧셈과 곱셈이 정의되어 있고 해당 연산들에 대해 결합법칙이 성립하고 항등원이 정의되어 있으며 덧셈에 대해서는 교환법칙이 성립하는 집합을 말합니다. Field는 여기에 0을 제외한 모든 원소에 대해 곱셈에 대한 역원이 존재하는 경우입니다. Polynomial Ring 같은 경우는 다항식의 곱셈과 덧셈으로 자연스럽게 정의됨을 알 수 있습니다.

## Formal Power Series

**Formal Power Series** 는 다항식과 거의 동일한 개념이라고 볼 수 있는데, 하나의 차이점은 formal power series는 항의 개수가 무한할 수 있다는 점입니다. 미적분학을 공부하면서 접하게 되는 테일러 급수(Taylor series)와 같은 power series가 수렴을 하는 $x$의 범위 등 수렴성에 대한 논의를 필요로 한다면 formal power series에서는 $x$를 어떠한 값을 가지는 수라고 생각하지 않고 $x, x^2, .., x^m$ 등을 단순히 coefficient의 위치를 나타내는 수단으로 취급합니다. 예를 들어, $A = 1 - 3x + 5x^2 - 7x^3 + 9x^4 - 11x^5 + ...$ 에서 $A$는 계수들의 수열 $(1, -3, 5, -7, 9, -11, ...)$과 완전히 동일한 의미라고 볼 수 있습니다.


## The ring of formal power series

Field $\mathbb{F}$에 대해 formal power series의 원소 하나는 $\mathbb{F}$의 원소로 이루어진 무한 수열 하나에 대응되므로, $\mathbb{F}^\mathbb{N}$과 일대일 대응됨을 알 수 있습니다. Ring of formal power series $\mathbb{F}[[x]]$에서의 덧셈은

$(\sum_{i \in \mathbb{N}} a_ix^i) + (\sum_{i \in \mathbb{N}} b_ix^i) = \sum_{i \in \mathbb{N}} (a_i + b_i)x^i$

$(\sum_{i \in \mathbb{N}} a_ix^i) \times (\sum_{i \in \mathbb{N}} b_ix^i) = \sum_{i \in \mathbb{N}} (\sum_{k=0}^i a_{k}b_{i-k})x^i$

로 정의됩니다. polynomial을 앞 유한개의 항을 제외한 모든 항이 0인 formal power series로 생각하면 polynomial의 덧셈과 곱셈의 결과는 이 Ring에서도 polynomial ring의 결과와 동일함을 확인할 수 있습니다.

# Formal power series의 Operation과 계산 방법

Formal power series $P(x),Q(x) \in \mathbb{F}[[x]]$에 대해, 이들의 첫 $N$개 항을 알고 있다고 가정해봅시다.
$P, Q$에 대한 연산이 주어졌을 때, 그 결과의 첫 $N$개 항을 구할 수 있는지를 알아봅시다.
먼저, 쉽게 생각할 수 있는 연산들에 대해 알아봅시다.

 - $P(x)+Q(x)$의 첫 $N$개 항은 단순히 계수를 더하여 구할 수 있습니다. $P(x)-Q(x)$도 마찬가지입니다. **(합, 차)**
 - $\frac{dP(x)}{dx}$의 첫 $N-1$개 항이나 $\int  P(x) dx$의 첫 $N+1$개 항도 $O(N)$ 시간에 계산 가능합니다. **(미분, 적분)**
 - $P(x)Q(x)$는 FFT를 통해 $O(N log N)$ 시간에 계산 가능합니다. $\mathbb{F}$가 $\mathbb{Z}/p\mathbb{Z}$인 경우 $p$가 Number-theoretic transform(NTT)가 적용 가능한 소수일 때는 NTT를 통해 $O(N log N)$ 시간에 계산 가능합니다. **(곱셈)**

$\mathbb{F}$가 NTT가 적용 가능한 $998244353(=119 \times 2^{23}+1)$과 같은 소수 $p$에 대해 $\mathbb{F} = \mathbb{Z} / p\mathbb{Z}$라고 가정해봅시다. 그리고 추가적으로 $N < p$라고 가정합시다.
이와 같은 조건에서, 다음과 같은 문제들도 빠르게 해결 가능할까요?

 - **(Inverse).** $\frac{1}{P(x)}$. 즉, $f(x)P(x) \equiv 1 (\mod x^N)$ 를 만족하는 $f(x) = \sum_{i=0}^{N-1} a_ix^i$ 구하기
 - **(Exp).** $exp(P(x))$. 즉, $f(x) \equiv \sum_{k=0}^{N-1} \frac{P(x)^k}{k!} (\mod x^N)$ 를 만족하는 $f(x) = \sum_{i=0}^{N-1} a_ix^i$ 구하기
 - **(Log).** $\log(P(x))$. 즉, $P(x) \equiv \sum_{k=0}^{N-1} \frac{f(x)^k}{k!} (\mod x^N)$ 를 만족하는 $f(x) = \sum_{i=0}^{N-1} a_ix^i$ 구하기
 - **(Square Root).** $\sqrt{P(x)}$. 즉, $f(x)^2 \equiv P(x) (\mod x^N)$ 를 만족하는 $f(x) = \sum_{i=0}^{N-1} a_ix^i$ 구하기

위에서 왜 Exp와 Log가 $\sum_{k=0}^{\infty}$이 아닌 $\sum_{k=0}^{N-1}$ 꼴로 쓰여졌는지는 추후에 Exp을 계산하는 파트에서 설명하도록 하겠습니다.

놀랍게도, 위 문제들 모두 해가 존재하는지 판정 및 존재하는 경우 $O(N log N)$ 시간에 구하는 것이 가능합니다. 이제 각각의 문제에 대해 찬찬히 알아보겠습니다.

## Inverse
먼저, $I_k$를 $I_kP \equiv 1 (\mod x^k)$를 만족하는 $\sum_{i=0}^{k-1} a_ix^i$ 꼴의 formal power series로 정의합니다. $I_1$은 $P$의 상수항의 역원이고, 만약 이 값이 존재하지 않는다면 $P$의 inverse가 존재하지 않음이 자명합니다.

$g := (2 - I_kP)I_k$라 하면

$gP = (2 - I_kP)I_kP = (1 + (1 - I_kP))(1 - (1 - I_kP)) = 1 - (1 - I_kP)^2$인데

$I_kP \equiv 1 (\mod x^k)$ 이므로 $gP \equiv 1 (\mod x^{2k})$가 성립합니다.

따라서, $I_{2k} = g = (2 - I_kP)I_k$ 로  $I_k$로부터 $I_2k$를 계산할 수 있습니다. $(2-I_kP)I_k$를 계산할 때 $P$의 첫 $2k$개 항만 가지고 계산을 하면 되기 때문에, 이 과정의 시간복잡도는 $O(k log k)$입니다. $I_1$부터 시작해서 $I_2$, $I_4$, ..., $I_{2^m}$을 $2^m \ge N$을 만족하는 $m$까지 만들면 문제를 해결할 수 있고, 총 시간복잡도는 $O(N log N)$입니다.

## Log

$\frac{d\log P}{dx} = \frac{dP}{dx}\frac{1}{P}$ 에서

$\log P = \int (\frac{dP}{dx}\frac{1}{P}) dx$ 가 성립합니다.

$\frac{dP}{dx}$, $\frac{1}{P}$를 계산할 수 있고 둘을 곱한 후 적분하면 $log P$가 나오게 됩니다.

$\frac{1}{P}$이나 두 formal power series의 곱을 계산하는 데 $O(N log N)$ 시간이 소요되므로 총 시간복잡도도 이와 같습니다.

## Exp
$exp(P) = \sum_{k=0}^{\infty} \frac{P^k}{k!}$에서 $p$번쨰 항부터는 분모가 0이되어 정의가 되지 않게 됩니다 ($\mathbb{F}$가 $\mathbb{Z}/p\mathbb{Z}$ 꼴이므로). 그러나 여기에 $P$의 상수항이 0이라는 조건을 달 수 있다면 $exp(P) \mod x^N$는 $N$번째 항부터는 영향을 받지 않기 때문에 $\sum_{k=0}^{N-1} \frac{P^k}{k!} \mod x^N$으로 잘 정의됩니다. 

먼저, $E_k$를 $E_k = exp(P) \mod x^k$로 정의합시다.

$P$의 상수항이 0이라는 조건을 달았으므로 $E_1 = 1$이 성립합니다.

$g := (P + 1 - \log E_k)E_k$로 놓아봅시다.

$\log E_k = P + t_kx^N$인 $t_k$를 잡을 수 있습니다.

그러면 $g = (P + 1 - \log E_k)E_k = (P + 1 - (P + t_kx^N))exp(P + t_kx^N) = (1 - t_kx^N)exp(t_kx^N)exp(P)$

$exp(t_kx^N) \equiv 1+t_kx^N (\mod x^{2N})$ 이 성립하므로

$g \equiv (1 - t_kx^N)( 1+t_kx^N) \equiv 1 (\mod x^{2N})$

따라서, $E_{2k} = g = (P + 1 - \log E_k)E_k$ 로  $E_k$로부터 $E_2k$를 계산할 수 있습니다. 이를 통해 Inverse와 마찬가지 방법으로 총 시간복잡도 $O(N log N)$에 $exp(P)$를 계산할 수 있음을 보일 수 있습니다.

## Square Root

Square Root같은 경우는 해가 존재하는지 판별이 위 함수들보다 복잡합니다. 먼저, $P$의 최저차항의 차수가 홀수이면 square root가 존재하지 않음이 자명합니다. 또한, 최저차항의 계수가 $p$의 이차잉여가 아닐 때도 square root가 존재하지 않습니다. 이차잉여 판별은 $\frac{p-1}{2}$거듭제곱을 했을 때 $p$로 나눈 나머지가 $1$이면 이차잉여이고 $-1$이면 이차잉여가 아님을 알 수 있습니다. 그 외의 경우들에는 $P$의 square root가 존재합니다.

최저차항이 $x^{2t}$이고 그 계수가 $a_{2t}$인 경우 $P$를 $a_{2t}x^{2t}$로 나누면 상수항이 1이 되게 할 수 있고 그에 대한 답을 구한 후 원래 답을 쉽게 복원할 수 있으므로 상수항이 1인 케이스에 대해서만 문제를 해결하면 충분합니다.

$S_k$를 $(S_k)^2 \equiv P (\mod x^k)$가 성립하는 $\sum_{i=0}^{k-1} a_ix^i$ 꼴의 formal power series로 정의합니다. 상수항이 1인 케이스만 보고 있으므로 $S_1 = 1$입니다.

$g = (S_k + PJ_{2k})/2$ 로 놓아봅시다. 여기서 $J_{2k}$는 $S_k$와의 곱이 $\mod x^{2k}$로 1이 되는 $\sum_{i=0}^{2k-1} a_ix^i$ 꼴의 formal power series 입니다.

$4g^2  = (S_k + PJ_{2k})^2 = S_k^2 + 2PS_KJ_{2K} + P^2J_{2K}^2$

$S_k^2 = P + tx^k$인 $t$를 잡아봅시다.

$4g^2 \equiv P + tx^k + 2P + P^2J_{2K}^2 (\mod x^{2k})$

한편, $P^2S_k^2J_{2K}^2 \equiv P^2 (\mod x^{2k})$ 이고 $S_k^2 = P + tx^k$ 이므로 $P^2J_{2K}^2 \equiv P - tx^k (\mod x^{2k})$

그러므로 $4g^2 \equiv P + tx^k + 2P  + P^2J_{2K}^2 \equiv P + tx^k + 2P + P - tx^k \equiv 4P (\mod x^{2k})$

따라서, $S_{2k} = g =(S_k + PJ_{2k})/2$ 로  $S_k$로부터 $S_2k$를 계산할 수 있습니다. 이를 통해 Inverse와 마찬가지 방법으로 총 시간복잡도 $O(N log N)$에 $\sqrt{P}$를 계산할 수 있음을 보일 수 있습니다.


# Application

위에서 Formal Power Series의 첫 $N$개 항에 대한 연산을 빠르게 하는 방법에 대해 알아보았습니다. 그러면 이를 통해 실제로 어떠한 문제들을 해결할 수 있을까요?

$N, K$이 주어졌을 때, $F(N, k) = \sum_{n=0}^N n^k$를  $k=0, 1, 2, .., K$에 대해 구해야 하는 상황을 생각해봅시다.

이는 가장 간단하게는 $O(NK)$ 시간에 해결할 수 있고, $N$이 크지 않은 경우에는 kitamasa 법을 이용해 $F(N,k)$의 첫 몇개 항을 이용하여 해결할 수 있을 것입니다. 하지만 $N$이 크고 $K$도 작지 않은 경우 이 문제를 빠르게 해결하는 것은 쉽지 않습니다. Formal power series의 operation을 이용하면 이와 같은 문제를 쉽게 해결할 수 있습니다.

$\sum_{t=0}^N exp(tx)$의 첫 $K+1$개 항은 $\sum_{k=0}^K \frac{F(N,k)}{k!}x^k$이므로 

$\sum_{t=0}^N exp(tx)$의 첫 $K+1$개 항을 빠르게 구하면 충분함을 알 수 있습니다.

그런데 $\sum_{t=0}^N exp(tx) = \frac{1 - exp((N+1)x)}{1 - exp(x)}$ 이므로, 이는 위에서 설명한 operation 중 inverse와 exp operation을 이용하면 $O(K log K)$ 시간에 문제를 해결할 수 있습니다.

이외에도 여러가지 생성함수를 이용한 경우의 수 문제들을 위 operation을 이용하면 보다 간단하게 해결할 수 있습니다.

## 참고 자료

* https://en.wikipedia.org/wiki/Formal_power_series
* https://en.wikipedia.org/wiki/Ring_(mathematics)
* https://judge.yosupo.jp/
* https://codeforces.com/blog/entry/56422
