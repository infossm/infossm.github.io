---
layout: post
title: "Montgomery, Barrett Reduction을 이용한 모듈러 연산의 고속화"
date: 2025-06-03
author: jinhan814
tags: [algorithm, mathematics, problem-solving]
---

## 1. Introduction

현대 CPU에서 나눗셈과 모듈러 연산은 연산 비용이 상당히 큰 편입니다. Agner Fog의 [instruction tables](https://www.agner.org/optimize/instruction_tables.pdf)에 따르면, Intel Skylake 아키텍처 기준으로 `ADD`, `SUB`의 latency는 1 cycle, `MUL`은 3~4 cycle인데 반해, `DIV`는 32비트는 26 cycle, 64비트는 최소 35 cycle, 최대 88 cycle까지 소요됩니다.

그런데 실제 코드에서 나눗셈 연산이 그렇게까지 느리게 동작하지 않는 경우가 많습니다. 이는 대부분의 현대 컴파일러가 <code>x / c</code>, <code>x % c</code>처럼 나누는 수 <code>c</code>가 상수일 때, 해당 연산을 곱셈(<code>*</code>)과 비트 시프트(<code>>></code>)로 변환해 최적화해주기 때문입니다. 이때 <code>c</code>가 <code>const</code>로 명시되어 있어야 컴파일 타임에 최적화가 적용됩니다. 만약 <code>c</code>가 일반 변수라면 `DIV` 명령어가 사용되기 때문에 실행 시간이 급격히 늘어날 수 있습니다.

또한, 대부분의 아키텍처에서 SIMD(Single Instruction Multiple Data) 명령어 집합은 정수 나눗셈(<code>/</code>)이나 모듈러 연산(<code>%</code>)을 지원하지 않습니다. 하지만 위와 동일하게 컴파일러가 사용하는 최적화 방법을 이용하면 비트 시프트(<code>>></code>), 덧셈(<code>+</code>), 곱셈(<code>*</code>) 등의 SIMD에서 지원하는 연산을 이용해 나눗셈을 구현할 수 있습니다.

이 글에서는 이러한 최적화 기법 중 대표적인 방법인 Barrett Reduction과 Montgomery Reduction 기법을 소개하고, 이를 통해 모듈러가 컴파일 타임에 주어지지 않는 상황이나 SIMD 기반의 병렬 연산을 구현할 때 나눗셈, 모듈러 연산의 성능을 개선하는 방법을 설명합니다.

## 2. Barrett Reduction

Barrett Reduction은 $0 \leq n < m^2$일 때,
$$
\left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor
$$
이 성립하는 충분히 큰 $k$를 찾아 나눗셈을 곱셈과 시프트 연산으로 대체하는 정수 나눗셈 최적화 기법입니다.

**Lemma 1.** Floor Equality

$$
\begin{aligned}
&0 \leq a &&(a \in \mathbb{Z}) \\[1.2ex]
&2 \leq b &&(b \in \mathbb{Z}) \\[1.2ex]
&0 \leq e < \dfrac{1}{b} &&(e \in \mathbb{R}) \\[1.5ex]
&\Rightarrow \quad \left\lfloor \dfrac{a}{b} + e \right\rfloor = \left\lfloor \dfrac{a}{b} \right\rfloor
\end{aligned}
$$

**Proof.**

$$
\begin{aligned}
& \quad a = qb + r \quad (q = \lfloor \frac{a}{b} \rfloor, \, 0 \leq r \leq b - 1) & \\[1.2ex]

&\Rightarrow \quad 0 \leq \frac{r}{b} \leq 1 - \frac{1}{b} \\[1.2ex]

&\Rightarrow \quad 0 \leq \frac{r}{b} + e < 1 \\[1.2ex]

&\therefore \quad \left\lfloor \frac{a}{b} + e \right\rfloor 
= \left\lfloor q + \left( \frac{r}{b} + e \right) \right\rfloor 
= q \quad \square
\end{aligned}
$$

**Lemma 2.** Barrett Approximation

$$
\begin{aligned}
&0 \leq n < m^2 &&(n \in \mathbb{Z}) \\[1.2ex]
&2 \leq m       &&(m \in \mathbb{Z}) \\[1.2ex]
&2^{\lfloor \log_2 (m - 1) \rfloor} \cdot \max(2n,\, m) < 2^k &&(k \in \mathbb{Z}) \\[1.5ex]
&\Rightarrow \quad \left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor
\end{aligned}
$$

**Proof.**

$$
\begin{aligned}
&s = \left\lfloor \log_2 (m - 1) \right\rfloor = \left\lceil \log_2 m \right\rceil - 1 
&&\Rightarrow \quad 2^s < m \leq 2^{s+1} \\[1.2ex]

&x = \left\lceil \frac{2^k}{m} \right\rceil,\quad r = x \cdot m - 2^k 
&&\Rightarrow \quad 0 \leq r < m \\[1.2ex]

&e = \frac{n x}{2^k} - \frac{n}{m}
&&\Rightarrow \quad 0 \leq e < \frac{1}{m} \\[1.5ex]

&\therefore \quad \left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor \frac{n x}{2^k} \right\rfloor 
&&\quad \square
\end{aligned}
$$

## 3. Montgomery Reduction

~

## References

[1] [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)

[2] [https://modoocode.com/313](https://modoocode.com/313)

[9] [https://cp-algorithms.com/algebra/montgomery_multiplication.html](https://cp-algorithms.com/algebra/montgomery_multiplication.html)