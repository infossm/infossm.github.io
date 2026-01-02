---
layout: post
title: "Quantum Complexity"
date: 2025-12-28
author: red1108
tags: [quantum, quantum-computing]
---

# Introduction

계산 복잡도 이론(Computational Complexity Theory)을 처음 접할 때, 우리는 대개 **튜링 머신(Turing Machine)** 을 기준으로 정의된 클래스들을 마주하게 됩니다. 아마도 가장 친숙한 이름들은 다음과 같을 것입니다.

- **P**: 결정론적 튜링 머신이 '다항 시간' 내에 '풀' 수 있는 문제들의 집합입니다. 보통 우리가 효율적으로 해결할 수 있는 문제라고 한다면 이 클래스에 속하는 문제들을 의미합니다.

- **NP**: 비결정론적 튜링 머신이 '다항 시간' 내에 '검증'할 수 있는 문제들의 집합입니다. 즉, 어떤 해답이 주어졌을 때, 그 해답이 올바른지 빠르게 확인할 수 있는 문제들을 포함합니다.

- **BPP**: 확률적 튜링 머신이 '다항 시간' 내에 '높은 확률'로 올바른 답을 내놓을 수 있는 문제들의 집합입니다. 이는 무작위성을 활용하여 효율적으로 문제를 해결하는 알고리즘들을 포함합니다.

- **PSPACE**: 결정론적 튜링 머신이 '다항 공간' 내에 문제를 해결할 수 있는 문제들의 집합입니다. 이는 시간보다는 공간(메모리) 사용량에 초점을 맞춘 클래스입니다.

이러한 **튜링 머신** 기반의 분류법은 문제를 해결하는데 필요한 시간적, 공간적 자원을 대략 파악하는 데 유용합니다. 그러나 우리가 사용하는 컴퓨터는 low-level 관점에서는 '회로(circuit)'로 구현되고 있습니다. 0, 1 bit를 나르는 선과 논리 게이트(AND, OR, NOT)로 구성된 회로를 상상해 봅시다. n개의 입력 비트를 사용한다면, 가능한 입력은 010100100... 이런 형태의 입력일 것입니다. 여기서 어떤 문제를 정의할 수 있을까요? 그리고 여기서 복잡도는 어떻게 정의될까요?

쉬운 문제를 살펴봅시다. 예를 들어, 입력 비트의 개수가 n일 때, 모든 비트가 0인지 확인하는 문제를 생각해 봅시다. 이 문제는 AND 게이트를 사용하여 쉽게 해결할 수 있습니다. n개의 입력 비트를 모두 AND 게이트에 연결하면, 출력이 1이 되는 경우는 오직 모든 입력이 1일 때뿐입니다. 따라서 이 문제는 매우 간단한 회로로 해결할 수 있습니다.

# Classical Circuit Complexity

## 회로의 크기와 깊이

n개의 입력을 가지는 circuit에서 **크기(size)** 와 **깊이(depth)** 라는 두 가지 중요한 개념이 있습니다.

- **크기(size)**: 회로를 구성하는 게이트의 총 개수를 의미합니다. 만약 모든 게이트를 하나씩 순서대로 처리해야 한다면 **순차적 실행 시간(Sequential Time)** 과 유사한 개념이 됩니다. 즉, 알고리즘이 수행해야 하는 총 작업량을 의미합니다.

- **깊이(depth)**: 입력에서 출력까지 도달하는 데 필요한 최대 게이트 층의 수를 의미합니다. 병렬 처리가 가능하다면 실질적인 실행 시간에 더 가까운 개념입니다.

그럼 n개의 입력 비트가 주어졌을때 모든 비트가 1인지 확인하는 문제의 복잡도를 어떻게 될까요? 사용 가능한 게이트를 어디까지 허용하냐에 따라 달라집니다.

<p align="center"><img src="/assets/images/red1108/qac_1.png" width="60%"></p>
<center><b>그림 1.</b> 두 종류의 AND gate</center><br/>

위 그림은 두 종류의 AND 게이트를 보여줍니다. 위에 있는 건 5-input AND gate입니다. 이걸 2-input AND 게이트로 구성한 모습이 아래에 있습니다.

만약 우리가 **2-input AND 게이트** 만 허용한다면, n개의 입력 비트가 모두 1인지 확인하는 문제를 해결하기 위해서는 log(n) 깊이의 회로가 필요합니다. 왜냐하면, 각 층에서 2개의 입력을 AND 연산하여 하나의 출력을 만들고, 이 출력을 다시 다음 층에서 AND 연산하는 과정을 반복해야 하기 때문입니다. 따라서 깊이는 log(n)이 됩니다. 회로의 크기는 n-1 입니다.

만약 우리가 임의 크기의 **n-input AND 게이트** 를 허용한다면, 이 문제는 깊이 1의 회로로 해결할 수 있습니다. 모든 입력 비트를 한 번에 AND 연산하여 출력할 수 있기 때문입니다. 회로의 크기는 1입니다.

이처럼 논리 게이트가 받을 수 있는 입력의 수가 제한이 없는 회로를 unbounded fan-in gate라고 부릅니다.

| 허용된 게이트 종류       | 회로 깊이      | 회로 크기     |
|---------------------|-------------|------------|
| bounded fan-in (2-input)      | log(n)      | n - 1
| unbounded fan-in      | 1           | 1          |

이처럼, 회로의 복잡도는 사용 가능한 게이트의 종류에 따라 크게 달라질 수 있습니다.

## PARITY 문제

이번에는 조금 더 복잡한 문제인 **PARITY** 문제를 살펴봅시다. PARITY 문제는 n개의 입력 비트 중에서 1의 개수가 홀수인지 짝수인지를 판단하는 문제입니다. 즉, 입력 비트의 합을 2로 나누었을 때 나머지가 1이면 홀수, 0이면 짝수입니다.

PARITY 문제를 AND, OR, NOT 게이트만으로 어떻게 해결할 수 있을까요? 사실, PARITY 문제는 이러한 게이트들만으로는 효율적으로 해결하기 어렵습니다. n-input XOR 게이트를 허용해 주면 한번에 되겠지만, 기본적으로 AND, OR, NOT 게이트만을 사용하는 게 원칙입니다 (나중에 AC, NC 클래스를 정의할 때 다시 다룹니다). 

| 허용된 게이트 종류       | 회로 깊이      | 회로 크기     |
|---------------------|-------------|------------|
| bounded fan-in (2-input) | O(logn)        | O(n)       |
| unbounded fan-in      | O(1) **불가능**        | Poly(n)       |
| unbounded fan-in      | 2         | O($n2^n$)       |


PARITY 문제는 회로 크기를 Poly(n) 으로 제한한다면 깊이 O(1) 로는 해결할 수 없음이 증명되어 있습니다[1]. 즉, PARITY 문제는 상수 깊이의 회로로는 해결할 수 없는 문제 중 하나입니다. 만약 깊이를 O($\log n$) 까지 허용한다면 아주 쉽게 풀립니다. (XOR 게이트를 트리 구조로 쌓으면 되니까요.)

만약 회로의 크기에 제한을 두지 않고 $O(n \cdot 2^n)$ 수준까지 허용한다면 깊이 2 만으로도 해결이 가능합니다 [2]. 이는 모든 함수를 '논리합의 논리곱(DNF)' 형태로 표현하여, 가능한 모든 입력 조합에 대한 결과를 하드코딩하는 방식입니다. 그런데 회로 크기가 너무 커서 써먹을 수는 없습니다.

이처럼 회로 크기와 depth 사이에는 trade-off 관계가 존재하며, 특정 문제를 해결하는 데 필요한 최소 회로 깊이와 크기는 어떤 gate set을 허용하느냐에 따라 달라집니다.

# AC, NC 클래스

여기에 소개 입력

# Quantum Circuit Complexity

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


# References

[1] Furst, M., Saxe, J. B., & Sipser, M. (1984). Parity, circuits, and the polynomial-time hierarchy. Mathematical Systems Theory, 17(1), 13-27.

[2] Arora, S., & Barak, B. (2009). Computational Complexity: A Modern Approach. Cambridge University Press. (Chapter 6: Circuit Complexity)

[1] learning shallow quantum circuits

[2] On the Pauli Spectrum of QAC0

[3] On the Computational Power of QAC0 with Barely Superlinear Ancillae

[4] learning shallow quantum circuits multi qubits

[5] Random unitaries in extremely low depth

[6] Random Unitaries in Constant (Quantum) Time

# 참고한 사진들

- [사진1] 