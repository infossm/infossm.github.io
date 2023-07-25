---
layout: post
title: "symmetry in variational quantum algorithm"
date: 2023-07-25
author: red1108
tags: [quantum, variational-quantum-circuit, variational-quantum-eigensolver, vqa, vqc, vqe]
---

## 서론

> Keywords: `VQC(Variational Quantum Circuit)`, `VQA(Variational Quantum Algorithm)`, `VQE(Variational Quantum Eigensolver)`, `Observable`, `Hamiltonian`

> 기본적인 양자상태의 표현과 gate가 어떻게 작동하는지는 이해하고 있다는 가정하에 글을 작성하였다.

> Variational quantum cirquit, variational quantum algorithm등에 대한 개념이 생소하다면 [이전 글](https://infossm.github.io/blog/2023/06/25/variational-algorithm/)을 참고하길 바란다. 

이번 글에서는 `Variational Quantum Algorithm(VQA)` 을 굉장히 효율적으로 사용할 수 있게 해주는 방법 중 하나를 다뤄보고자 한다. 바로 symmetry를 사용하는 것이다.

저번 글에서는 Parameterized Quantum Cirquit(PQC)와 Variational Quantum Algorithm(VQA)를 다루었다. 하지만 기존의 단순한 PQC기반 VQA는 한계가 존재한다. 이번 글에서는 이 한계에 대해 다뤄보고, 이 문제점을 해결하는 방법 중 하나인 symmetry를 소개하고자 한다. (참고로, symmetry를 사용하지 못하는 케이스도 존재한다)

먼저 저번 글의 내용인 PQC와 VQA를 간단하게 요약하고 넘어가자.

## Parameterized Quantum Cirquit(PQC)

 PQC는 양자 회로를 사용하여 최적 해를 찾으려고 할 때 일반적으로 가장 많이 활용되는 기법이다. 한 줄로 요약해보자면, PQC는 "고전적인 변수들에 의해 조정되는 양자 회로" 라 요약할 수 있다.

 예를들어 1번째 큐빗에 Rx(2.1)을 가하고,  2번째 큐빗에 Rz(-1.3)의 회전 연산을 가한다고 하자. 여기서 (2.1, -1.3)의 파라미터는 고전적으로 들고 있는 것이다. 만약 이 파라미터를 바꾸면 당연히 회로의 결과도 달라질 것이다.

 양자상태를 지속적으로 측정하면 몇가지 정보를 얻어낼 수 있다. 특히, 각 파라미터를 어떻게 조정해야 좋을지를 알아낼 수 있다.  이때 쓰이는 것이 parameter shift rule이라는 아주 고마운 정리인데, 이것에 대해서는 나중에 다룰 생각이다. 결론적으로, PQC를 활용한 알고리즘은 아래와 같은 프로세스를 따른다.
 
 1. 파라미터를 넣어서 양자회로의 결과를 얻어낸다. 원하는 결과를 얻으면 종료한다.
 2. 결과를 보고 파라미터를 어떻게 조정해야 하는지 판단한다
 3. 파라미터를 조정하고 1로 간다.

## Variational Quantum Algorithm(VQA)

VQA는 PQC를 활용하여 주어진 Observable의 최솟값을 찾는 알고리즘이다. 여기서 최소값은 Observable을 행렬로 나타냈을 때의 최소의 고유값이다. 그리고 마지막 상태는 해당 고윳값에 대응하는 고유벡터의 상태이다 (구체적으로, 고유벡터가 $|v\rangle$ 이라면 최종 양자상태는 $|v\rangle\langle v|$ 이다.)

여기까지는 앞선 글에서 다루어본 내용이다. 하지만 VQA에는 명확한 한계가 존재한다.

## 지수적으로 증가하는 파라미터 수

n큐빗 system에서 주어진 어떤 Observable의 값도 PQC를 통해 최소화시키려면 파라미터 수가 몇 개나 되어야 할까?

Observable의 값을 최소화시키는 경우는 고유값이 최소인 고유벡터일 것이다. 하지만 해당 고유벡터는 2^n개의 원소를 가지고 있다.
각 원소는 a+bi꼴이므로 원소당 2개의 자유도가 있다고 할 수 있다. 하지만 고유벡터의 norm이 1이어야 하는 조건으로 인해 실제 자유도는 좀 더 작지만, 대략 따져보면 자유도는 4^n정도일 것이다. (최소한 2^n은 넘는다)

파라미터 m개를 사용한 PQC로 자유도 4^n인 벡터를 표현할 수 있다면 m은 O(4^n)에 비례해야 함이 자명하다.
어떠한 Observable이 주어져도 PQC로 해를 찾을 수 있음을 보장하려면 큐빗 수에 지수적으로 증가하는 개수의 파라미터 수가 필요하다.

파라미터는 결국엔 **고전적으로** 저장되고, 업데이트되기 때문에 지수적인 파라미터 개수의 증가는 아무리 양자컴퓨터 시대가 와도 실현이 불가능하다. 큐빗 수가 15개 정도만 되어도 10억개 가량의 파라미터가 필요하다. 최적화를 한다고 해도 수십 큐빗 단위에서나 가능할 것이다.

이 때문에 더 큰 시스템의 최적화 문제를 해결하기 위해서는 PQC를 더 효율적으로 디자인해야 한다. 여기서 symmetry가 나온다.

## 결론

Symmetry를 사용하면 그렇지 않을 때보다 훨씬 적은 파라미터를 사용하여 훨씬 빠른 시간에 정답을 찾을 수 있다. 또한, Symmetry를 활용하지 않은 경우 정답을 찾음이 보장되려면 큐빗 수에 대해 exponential한 파라미터 수가 필요하여 실제로 적용하기에 문제가 많다. Symmetry도 symmetry나름이겠지만, 위의 경우들에는 다항 개수의 파라미터만 사용하여도 정답을 찾을 수 있으므로 굉장히 효율적이다.

따라서 VQA를 사용하여 주어진 문제를 해결하고 싶다면, 해당 문제에 대칭성이 있는지를 파악한 뒤에 만약 존재한다면 최대한 활용해야 한다.

### 참고문헌

[1] https://docs.pennylane.ai/en/stable/introduction/chemistry.html

[2] https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html