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

## 결론

Symmetry를 사용하면 그렇지 않을 때보다 훨씬 적은 파라미터를 사용하여 훨씬 빠른 시간에 정답을 찾을 수 있다. 또한, Symmetry를 활용하지 않은 경우 정답을 찾음이 보장되려면 큐빗 수에 대해 exponential한 파라미터 수가 필요하여 실제로 적용하기에 문제가 많다. Symmetry도 symmetry나름이겠지만, 위의 경우들에는 다항 개수의 파라미터만 사용하여도 정답을 찾을 수 있으므로 굉장히 효율적이다.

따라서 VQA를 사용하여 주어진 문제를 해결하고 싶다면, 해당 문제에 대칭성이 있는지를 파악한 뒤에 만약 존재한다면 최대한 활용해야 한다.

### 참고문헌

[1] https://docs.pennylane.ai/en/stable/introduction/chemistry.html

[2] https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html