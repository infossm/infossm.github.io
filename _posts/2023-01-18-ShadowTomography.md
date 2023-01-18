---
layout: post
title:  "Introduction to Shadow Tomography"
date:   2023-01-18
author: red1108
tags: [quantum, quantum information, quantum computing, quantum meqsurements]
---
# 서론

 양자컴퓨팅에는 다양한 양자 알고리즘이 존재한다. 하지만 양자컴퓨터의 결과를 해석하는 법은 computational basis를 기준으로 한 "측정" 뿐이다. Grover algorithm이나 Quantum Phase Estimation 알고리즘 등의 알고리즘은 결과 큐빗이 0인지 1인지 관측하는 것 만으로 충분하다.
 
하지만 임의의 n-qubit 양자상태는 $2^n \times 2^n$ 복소 행렬로 표현된다. 이 때문에 측정을 기반으로 하여 원래의 상태 행렬 자체를 알아내는 것 또한 활발하게 연구되었다. 이를 **Quantum State Tomography**라고 한다.

> 이 글은 양자역학에서 state를 density matrix로 어떻게 표현하는지, 관측을 어떻게 수학적으로 기술하는지에 대한 기본적인 개념에 대한 설명을 생략하였다. 만약 해당 내용을 모른다면 먼저 공부하고 읽는 것을 추천한다.

# Linear Inversion