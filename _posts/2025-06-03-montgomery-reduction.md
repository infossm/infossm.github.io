---
layout: post
title: "Montgomery, Barrett Reduction을 이용한 모듈러 연산의 고속화"
date: 2025-06-03
author: jinhan814
tags: [algorithm, mathematics, problem-solving]
---

## 1. Introduction

현대 CPU에서 나눗셈과 모듈러 연산은 덧셈이나 곱셈 등의 기본적인 산술 연산에 비해 상당히 느립니다.

이번 글에서는 모듈러 연산을 고속화하는 여러 테크닉에 대해 소개합니다.

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

[1] []()