---
layout: post
title:  "Introduction to Classical Shadow"
date:   2023-01-18
author: red1108
tags: [quantum, quantum information, quantum computing, quantum meqsurements]
---
#  Quantum state tomography

## 서론

 양자컴퓨팅에는 다양한 양자 알고리즘이 존재한다. 하지만 양자컴퓨터의 결과를 해석하는 법은 computational basis를 기준으로 한 "측정" 뿐이다. Grover algorithm이나 Quantum Phase Estimation 알고리즘 등의 알고리즘은 결과 큐빗이 0인지 1인지 관측하는 것 만으로 충분하다.
 
임의의 n-qubit 양자상태는 $2^n \times 2^n$ 복소 행렬로 표현된다. 이 때문에 측정을 기반으로 하여 원래의 상태 행렬 자체를 알아내는 것 또한 활발하게 연구되었다. 이를 **Quantum state tomography**라고 한다.

하지만 n-qubit의 양자상태 $\rho$를 알아내는 것은 최소한 O($rank(\rho)2^n$)번의 측정이 필요한 것이 증명되어 있다. 10-qubit state만 되어도 최소한 백만번의 측정을 해야 한다. 이로인해 양자상태 자체를 알아내는것이 아닌 *Observables, Fidelity, Entanglement entropy*등을 빠르게 알아내는 연구가 진행되었으며 그 중에서 가장 성공적인 예시인 *classical shadow*[2]를 소개하고자 한다. 그 이전에 먼저 Quantum state tomography에 대한 간단한 소개를 하고 넘어가고자 한다.

> 이 글은 양자역학에서 state를 density matrix로 어떻게 표현하는지, 관측을 어떻게 수학적으로 기술하는지에 대한 기본적인 개념에 대한 설명을 생략하였다. 만약 해당 내용을 모른다면 먼저 공부하고 읽는 것을 추천한다.

## 양자 상태를 빠르게 알아내는 것의 중요성

 임의의 양자상태의 density matrix를 알아내려면 측정을 통해 알아내는 수 밖에 없다. 하지만 수직하지 않은 임의의 두 양자상태는 측정 한번에 100%의 정확도로 알아내는 법이 존재하지 않는다[1]. 가능한 density matrix들의 후보들이 주어질때, 측정을 단 한번 사용해서 최대한의 정확도로 어느 상태인지 구별하는 것은 **Quantum state discrimination**이란 별개의 분야로 존재하니 관심 있으면 찾아보길 바란다.

 양자상태를 알아내는것이 어려운 이유는 아래 두가지 속성 때문이다.
 
 1. 측정 이후엔 상태가 손상된다
 2. 미지의 양자상태를 복사하는것은 불가능하다 

 이 때문에 양자상태를 알아내기 위해서는 여러개의 copies를 준비하여 측정을 여러번 하는 수 밖에 없다.

 결론적으로는 구하고자 하는 양자상태 $\rho$가 A 프로세스를 거쳐서 나왔다고 하면, $\rho$의 복사본 n개를 구하려면 해당 프로세스를 n번 반복해야 한다. 따라서 최소한의 복사본(또는 측정횟수)으로 양자상태 알아내는 것이 중요하다. 따라서 앞으로는 copy complexity만을 따질 것이다. O($n^2$)라는 표현은 n-qubit state를 tomography하기 위해 O($n^2$)개의 복사본이 필요하다는 의미이다.

## Linear inversion

Quantum state tomography중에서 제일 단순한 Linear inversion알고리즘을 먼저 소개하고자 한다. 

0. 미지 양자상태 $\rho$와 잘 알고 있는 POVM elements $\{E_1,E_2,...,E_m\}$가 주어진다.
1. POVM을 적용하였을때 $E_i$가 선택될 확률은 $P(E_i|\rho)=tr(E_i\rho)$ 이다. 이 값은 $E_i$와 $\rho$의 원소들의 일대일 곱이다. 따라서 대응되는 colum vector를 만들면 $tr(E_i\rho)=\vec{E_i}^\dagger\vec{\rho}$로 다시 쓸 수 있다.
2. 이제 $A=\begin{pmatrix}
\vec{E_1}^\dagger \\
\vec{E_2}^\dagger \\
\vec{E_3}^\dagger \\
\vdots  \\
\end{pmatrix}$ 로 구성하면 $A\vec\rho = \begin{pmatrix}
\vec{E_1}^\dagger\vec\rho \\
\vec{E_2}^\dagger\vec\rho \\
\vec{E_3}^\dagger\vec\rho \\
\vdots  \\
\end{pmatrix}=\begin{pmatrix}
P(E_1|\rho) \\
P(E_2|\rho) \\
P(E_3|\rho) \\
\vdots  \\
\end{pmatrix}=\vec p$ 임을 알 수 있다.
3. A가 가역행렬이란 보장은 없지만, POVM elements $\{E_1,E_2,...,E_m\}$가 *tomographically complete*하다면 $A^T A$는 가역이다. 따라서 $\vec \rho = (A^T A)^{-1} A^T \vec p$ 로 미지의 양자상태 $\rho$를 구할 수 있다.

위의 설명은 이론적인 증명이다. 실제 적용은 단순한데, 단순히 측정을 n번 적용하여 $\vec p$에 대한 추정량 $\hat{\vec p}$를 구한 다음,  $\hat{\vec \rho} = (A^T A)^{-1} A^T \hat{\vec p}$ 을 계산하여 $\rho$를 추정할 수 있다.

하지만 Linear inversion기법은 이렇게 구한 $\hat \rho$가 density matrix라는점이 보장되지 않는다. positive semi-definite하지 않을 수 있는데, 이는 $\hat{\vec p}$가 $\vec p$에 대한 추정량이지 참값이 아니기 때문이다. 이러한 문제점으로 인해 실제로는 거의 사용되지 않고, 훨씬 좋은 방식들도 많다.

## Quantum shadow tomography의 다른 방식들

n-qubit density matrix $\rho$를 알아내는 최소한의 측정횟수는 O($rank(\rho)2^n$)이라는 점을 밝혔다. 이는, 모든 copies들에 동시에 작용하는 measurements를 구현한다면 O($rank(\rho)2^n$)번으로 가능함이 밝혀져 있다[3]. 같은 연구에서는 좀 더 실현 가능한 O($rank(\rho)^2 2^n$)방식도 연구하였으니 궁금하다면 원리를 찾아보는것도 좋을 것 같다.

# Classical shadow

## 서론

이제부터는 $D \times D$ density matrix $\rho$를 알아내는 것이 아닌 다른 것에 집중해볼 것이다. Aaronson은 POVM elements $\{E_1,E_2,...,E_m\}$가 주어졌을때 $tr(E_i \rho)$값을 알아내는 것은 굉장히 적은 수의 측정으로 가능하다는 점을 알아냈다[4]. $O(D^2)$또는 $O(M)$방식은 trivial 하지만 poly(logM, logD)만에 가능하다는 것을 보인 것이다. 


## Reference

> Nielsen, M.A. & Chuang, I.L., 2011. Quantum Computation and Quantum Information: 10th Anniversary Edition, Cambridge University Press 책은 QCQI라고 줄여서 부르자

 [1] QCQI 2.2.4장
 
 [2] Huang, H.-Y., Kueng, R., & Preskill, J. (2020). Predicting many properties of a quantum system from very few measurements. Nature Physics, 16(10), 1050–1057.

 [3] Haah, J., Harrow, A. W., Ji, Z., Wu, X., & Yu, N. (2017). Sample-optimal tomography of quantum states. IEEE Transactions on Information Theory, 1–1.

 [4] Aaronson, S. Shadow tomography of quantum states. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC 2018) 325–338 (ACM, 2018)
