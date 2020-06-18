---
layout:     post
title:      "양자 컴퓨팅 입문 (2) - Grover's Algorithm"
date:       2020-06-18 15:00
author:     evenharder
image:      /assets/images/evenharder-post/quantum/pexels-light-1210276.jpg
tags:
  - quantum

---

Grover's Algorithm은 정렬되지 않은 데이터베이스에 있는 $N$개의 항목 중 특정한 조건을 만족하는 항목을 $O(\sqrt{N})$에 찾는 알고리즘입니다.

고전 컴퓨팅에서 이 문제를 해결하려면, 간단하지만 느리게 갈 수밖에 없습니다. 전수 조사(brute force)가 유일한 해결책입니다. 당연히 시간 복잡도는 $O(N)$이며, 랜덤하게 셔플해서 순서를 바꾼다 해도 마찬가지입니다. 그리고 이보다 더 나은 시간복잡도로 찾을 수는 없습니다.

Grover's Algorithm은 이 시간복잡도를 $O(\sqrt{N})$으로 낮추는데 성공하였고, 이 시간복잡도가 최적이라는 것이 알려져 있습니다. 또다른 유명한 양자 알고리즘인 Shor's Algorithm과는 달리 알고리즘 자체가 간단하면서도 심도 있기 때문에 이론적으로 탐구해보고자 합니다.

# 알고리즘

$n$개의 qubit $\left\vert \psi \right>$가 $\left\vert 0 \right>^{\otimes n}$의 상태로 있고, 함수 $f$가 다음과 같이 정의되어 있다고 합시다.

$$f(\psi) = \begin{cases} 1 & \text{if } \left\vert \psi \right> \in A \subset \{\left\vert 0 \right>,\left\vert 1 \right>\}^{\otimes n} \\ 0 & \text{otherwise}\end{cases}$$

즉 $\psi$가 특정 기저 상태면 (달리 표현해 $A$에 속하면) 1이고, 아니면 0인 함수입니다. 이런 $A$의 예시로는 데이터베이스에서 특정 쿼리의 조건을 만족하는 항목들이나, 암호화 과정에서 후보가 될 수 있는 key의 집합이 있습니다.

$A$의 크기가 $M$이고 $N = 2^n$이라 할 때, Grover's Algorithm은 $f(\psi) = 1$를 만족하는 $\psi$를 오라클 호출 $\frac{\pi}{4} \sqrt{\frac{N}{M}}$번 후 확률 $1 - \frac{M}{N}$로 찾을 수 있습니다.

알고리즘은 다음과 같습니다.

1. $\left\vert 0 \right>^{\otimes n}$ (모든 큐빗을 $\left\vert 0 \right>$ 상태로 만든다.)
2. $H^{\otimes n} \left\vert 0\right>^{\otimes n} = \dfrac{1}{\sqrt{2^n}} \sum\limits_{x=0}^{2^n-1}\left\vert x\right> = \left\vert \psi\right>$ (모든 큐빗에 Hadamard operator를 건다.)
3. $[(2 \left\vert \psi \right> \left<\psi\right\vert - I) (-1)^f]^R \left\vert \psi \right> \approx \left\vert \psi' \right>$ (Grover iteration을 $R \approx \frac{\pi}{4}\sqrt{2^n/M}$ 번 시행한다.)
4. $\psi'$ (관찰한다.)

풀어쓰면 큐빗에 $H$를 전부 건 다음, Grover iteration이라는 과정을 통해 $A$에 속하는 기저들의 진폭을 증가, 나머지는 감소시킵니다. Grover Iteration을 $R$번 반복하면 위의 확률로 $A$에 속하는 기저 상태를 관찰할 수 있습니다.

이 밑에 나오는 설명은 편의상 $M = 1$을 가정합니다. $M > 1$일 때의 증명 및 설명은 참고문헌으로 대체합니다.

## Quantum Oracle

$f$는 다음과 같은 quantum oracle $\mathcal{O}$로 해석할 수 있습니다.

$$\mathcal{O}\left\vert x \right> \left\vert y \right> = \left\vert x \right> \left\vert y \oplus f(x) \right>$$

$\oplus$로 표기할 수 있는 이유는, 양자 컴퓨팅의 CNOT이 고전 컴퓨팅의 XOR에 대응되기 때문입니다. 그럼 Grover's Algorithm에 필요한 quantum oracle인

$$\mathcal{O}\left\vert x \right>\left\vert y \right> \to  (-1)^{f(x)} \left\vert x \right>\left\vert y\right>$$

은 어떻게 구상해야 할까요? 다양한 방법이 있겠지만 가장 간편하고 널리 알려진 방법 중 하나인 phase kickback trick을 이용할 수 있습니다. $\left\vert y \right> = \left\vert - \right> = \dfrac{\left\vert 0 \right> - \left\vert 1 \right>}{\sqrt{2}}$로 하고 $\mathcal{O}$를 적용하면 놀라운 일이 벌어집니다.

$$\begin{aligned} \mathcal{O}\left\vert x \right>  \left\vert - \right> &= \dfrac{1}{\sqrt{2}} (\mathcal{O}\left\vert x \right>\left\vert 0 \right> - \mathcal{O}\left\vert x \right>\left\vert 1 \right>) \\ &= \dfrac{1}{\sqrt{2}} (\left\vert x \right>\left\vert f(x) \right> - \left\vert x \right>\left\vert 1 \oplus f(x)\right>) \\ &= \begin{cases} \frac{1}{\sqrt{2}}(\left\vert x \right>\left\vert 0 \right> - \left\vert x \right>\left\vert 1 \right>) = \left\vert x \right>\left\vert - \right> & f(x) = 0 \\ \frac{1}{\sqrt{2}}(\left\vert x \right>\left\vert 1 \right> - \left\vert x \right>\left\vert 0 \right>) = -\left\vert x \right>\left\vert - \right> & f(x) = 1 \end{cases} \\ &=(-1)^{f(x)}\left\vert x \right>\left\vert - \right>\end{aligned}$$

분명히 $\mathcal{O}\left\vert x \right> \left\vert y \right> = \left\vert x \right> \left\vert y \oplus f(x) \right>$인데도 불구하고 $y$쪽은 그대로인채, $x$만 변한 것을 알 수 있습니다.

$\omega \in A$일 때, 이 과정은 수학적으로 $(I - 2 \left\vert \omega \right> \left<\omega \right\vert)$으로 표현할 수 있습니다. 지금은 $A = \{\omega\}$를 전제하고 있지만 $M > 1$이 되어도 비슷한 논리를 적용할 수 있습니다.

이런 식으로 특정 기저의 진폭 부호를 뒤집는 방식을 conditional sign flip이라고 합니다.

## Diffusion Transform

$(2 \left\vert \psi \right> \left<\psi\right\vert - I)$는 diffusion transform이라 불리는 과정입니다. 수학적으로 $I - 2\mathbf{v}\mathbf{v}^\dagger$는 Householder Transformation인데, Hermitian이고 unitary하면서 기하학적 대칭변환으로 해석할 수 있습니다.

이 과정은 각 기저의 진폭을 평균에 대해 대칭시킵니다. 위에서 $A$에 속한 기저의 진폭이 음수가 되었고, 일반적으로 $A$의 크기가 $N$에 비해 작기 때문에, 평균 진폭값에 비해 큰 차이가 납니다. 이 상태에서 diffusion transform을 진행하면 음수 진폭이 양수가 됨과 동시에 다른 진폭보다 더 큰 값을 가지게 됩니다. 평균에서 더 멀리 떨어져있었기 때문입니다.

## 관측 확률

비록 계산결과를 직접 쓰진 않았지만,  Grover Iteration은 $k^2 + l^2(N-1) = 1$을 만족하는 실수쌍 $(k, l)$을 $(\frac{N-2}{N}k + \frac{2(N-1)}{N}l, \frac{N-2}{N}l - \frac{2}{N}k)$로 변환합니다. 이를 점화식 꼴로 쓰면

$$\begin{aligned} k_0 &= l_0 = \frac{1}{\sqrt{N}} \\ (k_{j+1}, l_{j+1}) &= \left(\frac{N-2}{N}k_j + \frac{2(N-1)}{N}l_j, \frac{N-2}{N}l_j - \frac{2}{N}k_j\right) \end{aligned}$$

가 됩니다. 놀랍게도 일반항이 기하학적인 꼴로 나옵니다. $\sin^2 \theta = \frac{1}{N}$인 $\theta$를 잡으면

$$\begin{aligned} k_j &= \sin((2j+1)\theta) \\ l_j &= \frac{1}{\sqrt{N-1}} \cos((2j+1)\theta)  \end{aligned}$$

와 같이 나오게 됩니다. $k$가 우리가 원하는 특정 성질이 있는 기저의 확률이므로, $(2m+1)\theta = \pi/2 \implies m = \dfrac{\pi - 2\theta}{4\theta}$가 될 때 관찰 확률이 1이 됩니다. 때문에 $M = \lfloor \pi/{4\theta}\rfloor \approx \lfloor \frac{\pi}{4} \sqrt{N}\rfloor$ 정도 돌리면 충분해보임을 알 수 있고, 논문에 의하면 그렇습니다.

## 구현체

Grover's Algorithm의 구현체를 찾아보면 예상외로 간단한 편입니다. 오라클 적용과 diffusion transform의 구현이 간단한 탓입니다.

$\left\vert - \right> = HX\left\vert 0 \right>$이기 때문에, 위에서 살펴본 phase kickback trick을 이용하면 오라클 적용 파트는 쉽게 넘어갈 수 있습니다. 다만 diffusion transform은 조금 더 풀어서 설명을 해보겠습니다.

$(2 \left\vert \psi \right> \left<\psi\right\vert - I) = H^{\otimes n} (2 \left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n} - I) H^{\otimes n}$인지라, $(2 \left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n} - I)$를 해석해보아야 합니다. 살펴보면 이는 $\left\vert 0 \right>^{\otimes n}$을 제외하고 전부 진폭의 부호를 뒤집는 연산입니다.

$$(2 \left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n} - I)\left\vert x \right> = \begin{cases} 2 \left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n}\left\vert x \right> - \left\vert x \right> = - \left\vert x \right> & \text{if }\left\vert x \right> \neq \left\vert 0 \right>^{\otimes n} \\ 2 \left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n}\left\vert x \right> - \left\vert x \right> = \left\vert x \right> & \text{if }\left\vert x \right> = \left\vert 0 \right>^{\otimes n}\end{cases} $$

$\left<0\right\vert^{\otimes n}\left\vert x \right>$ 가 $\left\vert x \right> \neq \left\vert 0\right>^{\otimes n}$이면 서로 다른 두 기저벡터의 내적이기 때문에 $0$이 됨을 이용합니다. 때문에 부호를 거꾸로 한 $(I - 2\left\vert 0 \right>^{\otimes n} \left<0\right\vert^{\otimes n})$는 $\left\vert 0 \right>^{\otimes n}$의 진폭만 부호를 뒤집는 연산입니다.

큐빗 전체에 $X$연산을 적용하면  $\left\vert 1 \right>^{\otimes n}$만 뒤집는 연산으로 볼 수 있고, 이건 특이한 트릭으로 구현 가능합니다. 첫 $n-1$개의 큐빗을 controlled qubit으로 놓고, 마지막 큐빗을 target qubit으로 두어 $Z$연산을 하면 됩니다. 여기서 $Z$ 연산은 Pauli-Z gate로

$$
Z = \begin{bmatrix} 1 & 0 \\ 0 &-1\end{bmatrix}
$$

입니다. controlled가 적용되는 첫 $n-1$개의 큐빗은 기저가 $\left\vert 1 \right>^{\otimes n-1}$일 때이며, $Z$ 연산에 의해 target이 $\left\vert 1 \right>$일 때만 부호가 반전됩니다. 그러므로 $\left\vert 1 \right>^{\otimes n}$의 진폭만 부호를 반전시킵니다.

종합하면 Grover's Algorithm을 구현할 수 있습니다. 구현체는 인터넷에 많이 공개되어 있기 때문에 알고리즘을 잘 이해하지 못했더라도 black-box function처럼 사용할 수 있습니다.

# 관련 연구

Grover's Algorithm의 시간복잡도와 시행횟수가 최적이라는 점이 알려져 있기 때문에, 적은 횟수만큼 돌리고 관찰을 한 다음에 그 결과에 따라 추가적으로 몇 번 더 돌리는 전략에 대한 연구도 진행되고 있습니다. $M$의 값을 알고 돌리면 편하겠지만, 모르더라도 $O(\sqrt{N/M})$번 정도 돌려서 원하는 상태를 알아낼 수 있는 접근법이 존재합니다.

이전 글에서도 소개했지만, Grover's Algorithm을 고전 알고리즘과 결합하여 시간복잡도 향상을 꾀하는 경우가 많습니다.

# 결론

비록 엄밀한 증명을 다루지는 않았지만 대략적으로 Grover's Algorithm이 어떻게 동작하고, 구현은 어떻게 되는지를 살펴보았습니다. 이해가 어려울 수 있지만 양자컴퓨팅의 기초와도 같은 알고리즘이고, 유용한 기법도 많이 사용되기에 숙지해두면 좋겠습니다. 다음에는 Microsoft Q#의 syntax를 다루면서 Grover's Algorithm의 구현체를 살펴보도록 하겠습니다.

# 참고 문헌

- Grover, L. K. (1996, July). A fast quantum mechanical algorithm for database search. In Proceedings of the twenty-eighth annual ACM symposium on Theory of computing (pp. 212-219).
  + 원본 논문입니다.
- [Grover's Algorithm - Qiskit](https://qiskit.org/textbook/ch-algorithms/grover.html).
  + 전반적으로 알고리즘, 구현체, 이미지, 수작업 모두 잘 설명되어있습니다.
- Boyer, M., Brassard, G., Høyer, P., & Tapp, A. (1998). Tight bounds on quantum searching. Fortschritte der Physik: Progress of Physics, 46(4‐5), 493-505.
  + Grover's Algorithm에 대한 응용이 상세하게 수학적으로 기술되어 있습니다.
- Strubell, E. (2011). An introduction to quantum algorithms. COS498 Chawathe Spring, 13, 19.
  + Grover's Algorithm의 수작업 예제가 있습니다.
- [Run Grover's search algorithm in Q#](https://docs.microsoft.com/en-us/quantum/quickstarts/search).
  + Grover's Algorithm을 Q#으로 구현한 Microsoft의 코드가 있습니다.

