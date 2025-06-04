---
layout: post
title: "Montgomery, Barrett Reduction을 이용한 모듈러 연산의 고속화"
date: 2025-06-03
author: jinhan814
tags: [algorithm, mathematics, problem-solving]
---

## 1. Introduction

현대 CPU에서 나눗셈과 모듈러 연산은 연산 비용이 상당히 큰 편입니다. Agner Fog의 [instruction tables](https://www.agner.org/optimize/instruction_tables.pdf)에 따르면, Intel Skylake 아키텍처 기준으로 `ADD`와 `SUB`의 latency는 1 cycle, `IMUL`은 3~4 cycle인데 반해, `IDIV`는 32비트는 26 cycle, 64비트는 최소 42 cycle, 최대 95 cycle까지 소요됩니다.

그런데 실제 코드에서 나눗셈 연산이 그렇게까지 느리게 동작하지 않는 경우가 많습니다. 이는 대부분의 현대 컴파일러가 <code>x / c</code>, <code>x % c</code>처럼 나누는 수 <code>c</code>가 상수일 때, 해당 연산을 곱셈(<code>*</code>)과 비트 시프트(<code>>></code>)로 변환해 최적화해주기 때문입니다. 이때 <code>c</code>가 <code>const</code>로 명시되어 있어야 컴파일 타임에 최적화가 적용됩니다. 만약 <code>c</code>가 일반 변수라면 `IDIV` 명령어가 사용되기 때문에 실행 시간이 급격히 늘어날 수 있습니다.

또한, 대부분의 아키텍처에서 SIMD(Single Instruction Multiple Data) 명령어 집합은 정수 나눗셈(<code>/</code>)이나 모듈러 연산(<code>%</code>)을 지원하지 않습니다. 하지만 위와 동일하게 컴파일러가 사용하는 최적화 방법을 이용하면 비트 시프트(<code>>></code>), 덧셈(<code>+</code>), 곱셈(<code>*</code>) 등의 SIMD에서 지원하는 연산을 이용해 나눗셈을 구현할 수 있습니다.

이 글에서는 이러한 최적화 기법 중 대표적인 방법인 Barrett Reduction과 Montgomery Reduction 기법을 소개하고, 이를 통해 모듈러가 컴파일 타임에 주어지지 않는 상황이나 SIMD 기반의 병렬 연산을 구현할 때 나눗셈, 모듈러 연산의 성능을 개선하는 방법을 설명합니다.

## 2. Barrett Reduction

- $0 \leq N < 2^{b-1}$
- $2 \leq M < 2^b$
- $s = \lfloor\log_2(M-1)\rfloor = \lceil\log_2M\rceil-1$
- $X = \displaystyle\lceil\frac{2^{b+s}}{M}\rceil$

다음이 성립합니다.

- $\displaystyle\lfloor\frac{NX}{2^{b+s}}\rfloor = \lfloor\frac{N}{M}\rfloor$

## 3. Montgomery Multiplication

~

## References

[1] [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)

[2] [https://modoocode.com/313](https://modoocode.com/313)