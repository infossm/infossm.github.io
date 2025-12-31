---
layout: post
title: "Quantum Complexity"
date: 2025-12-28
author: red1108
tags: [quantum, quantum-computing]
---

아마도 복잡도 이론에서 가장 자주 들어봤을 만한 용어는 P, NP, BPP, BQP, PSPACE 등일 것이다. 이런 개념들의 관점은 특정 문제를 푸는 데에 필요한 자원(시간, 공간)을 기준으로 문제를 분류하는 것이다. 이번 글에서는 좀더 low-level 하게 특정 계산을 수행하는 '회로' 관점에서 복잡도를 살펴보고자 한다. 특히, 양자 복잡도 이론에서 가장 흥미롭게 연구되는 클래스인 $\text{QAC}^0$ 에 소개하는 것이 이번 글의 목표이다.

AC, NC, TC 간단한 소개.
complexity zoo

복잡도 상에서 QAC는 단일큐빗 U와 임의 크기 CZ를 허용한다.
임의크기 CZ를 허용하는것과 임의크기 CNOT을 허용하는것은 동치이다.

QNC는 단일큐빗 U와 CZ(또는 CNOT) 게이트를 허용한다.

풀고 못푸는 예시 문제
parity, majority.

lightcone argument.

exact degree, approximate degree 수학적 개념 소개.

pauli degree의 저차 집중 문제.

QAC0 <= QNC0 <= QAC1 은 well known.

open problems:
QAC0 가 parity를 풀 수 있을까? 아직까지 증명은 안 되었지만, 아마 불가능할 것으로 추정중.

[1] learning shallow quantum circuits

[2] On the Pauli Spectrum of QAC0

[3] On the Computational Power of QAC0 with Barely Superlinear Ancillae

[4] learning shallow quantum circuits multi qubits

[5] Random unitaries in extremely low depth

[6] Random Unitaries in Constant (Quantum) Time