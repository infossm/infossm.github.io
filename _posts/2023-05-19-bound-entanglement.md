---
layout: post
title: "bound entanglement"
date: 2023-05-19
author: red1108
tags: [quantum, quantum-information, entanglement]
---

## 서론

양자 얽힘 (quantum entanglement)는 크게 두 가지로 분류할 수 있다. 대부분의 entanglement는 `free enntanglement`이며, 일부가 `bound entanglement`이다. 특히, `bound entanglement`는 아직 한국에 잘 소개되지 않았으며, 한국어 명칭도 정착되지 않은 개념이다. 그리고 bound entanglement와 깊게 관련된 LOCC(Local Operations and Classical Communication)도 일반 사람들을 대상으로 많이 소개되지 않아 이에 대해 이번 글에서 간단히 소개해 보고자 한다.

양자 상태부터 설명을 시작해서 bound entanglement까지 설명하기에는 너무 긴 분량이 될 것이므로 초반부는 간략하게 설명하였으니 이 글을 읽고 흥미가 생긴 사람은 더 찾아보길 권한다.

## 큐비트

고전적인 비트는 0 또는 1의 값만 가질 수 있다. 큐비트도 마찬가지로 0의 상태, 1의 상태를 가질 수 있다. 이 상태를  $\vert 0 \rangle$, $\vert 1 \rangle$ 이라 한다. 여기서 더 나아가 큐비트는 0 또는 1을 가질 확률이 정의된다. 예를 들어, 0.5 확률로 $\vert 0 \rangle$, 0.5 확률로 $\vert 1 \rangle$ 을 가진다면 이 상태는 $\vert \psi\rangle = \frac{1}{\sqrt2}\vert 0 \rangle + \frac{1}{\sqrt2}\vert 1 \rangle$ 이렇게 정의된다. 즉, 상태 앞에 붙어있는 수의 제곱을 해야 실제 확률이 된다. 곱해지는 수는 여기서는 실수이지만, 실제로는 복소수 범위이다.
이제 위에서 만든 $\vert \psi\rangle$을 관측한다면 50%확률로 $\vert 1\rangle$, 50%확률로 $\vert 1\rangle$ 의 상태가 관측되는 것이다.

잘 생각해 보면 $\psi$ 는 어차피 $\vert 0 \rangle$과 $\vert 1 \rangle$에 적당히 수를 곱한 상태이다. 따라서 매번 $\vert \psi\rangle = \frac{1}{\sqrt2}\vert 0 \rangle + \frac{1}{\sqrt2}\vert 1 \rangle$ 이렇게 쓰기보다 계수만 가져와서 벡터로 표현하는게 더 편하다. $\vert \psi\rangle = \begin{bmatrix} \frac{1}{\sqrt2}\\ \frac{1}{\sqrt2}\end{bmatrix}$ 이렇게 표현한다. 사실, $\vert x\rangle$ 이라는 표현 자체가 열 벡터를 의미한다. $\langle x\vert $ 는 그것의 켤레전치이다(conjugate transpose). 즉, $\langle x\vert  = \vert x\rangle ^ \dagger = \begin{bmatrix}\frac{1}{\sqrt2}&\frac{1}{\sqrt2}\end{bmatrix}$ 이다.


### 여러 큐비트

2개의 큐비트를 관측하면 얻을 수 있는 값은 00, 01, 10, 11일 것이다. 이를 $\vert 00\rangle$, $\vert 01\rangle$, $\vert 10\rangle$, $\vert 11\rangle$ 로 표현한다. 수학적으로는 $\vert xy\rangle = \vert x\rangle \otimes \vert y\rangle$ 이다.

## 양자 얽힘

요즈음은 유튜브에도 양자 얽힘에 대해 정말 많은 내용이 나오고 있다. 특히 양자 얽힘 상태가 존재한다면 광속을 넘어 정보를 전달할 수 있는것이 아닌가? 라는 물음에서 출발한 EPR역설은 과학 유튜버의 단골 소재이다. Bound entanglement에 대해 알기 위해서는 그것의 상위 분류인 entanglement부터 알아야 한다.

> 양자 얽힘이란 여러 양자 상태들의 그룹이 각각의 양자 상태들에 대해 독립적으로 기술되지 않는 상태를 말한다.

예를들어, $\frac{1}{2}(\vert 00\rangle+\vert 01\rangle+\vert 10\rangle+\vert 11\rangle)$에서 첫 번째 큐비트를 먼저 관측한다고 해보자. 0이 나올 확률도 50%이고, 1이 나올 확률도 50%이다. 반대로, 저 상태에서 두 번째 큐비트를 관측해도 0이 나올 확률과 1이 나올 확률은 같다.

첫 번째 큐비트를 관측하면 두 번째 큐비트에 대한 정보가 달라지는가? 여전히 두 번째 큐비트는 0일수도 1일수도 있다. 아무런 영향을 주지 못한다.

그렇다면 $\frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)$ 을 살펴 보자. 첫 번째 큐비트와 두 번째 큐비트를 따로따로 본다면 0과 1의 확률은 반반이다. 그럼 첫 번째 큐비트를 관측해 보자. 만약 0이 나왔다면, 두 번째 큐비트의 값은 반드시 0이어야 한다. 왜냐하면 가능한 상태 조합 중에서 01은 없었기 때문이다.

이와 같이 여러 큐비트를 포함한 계가 각 큐비트의 독립적인 상태들의 텐서 곱으로 표현되지 않는다면 해당 계는 얽혀있다고 한다.

특별히, 대표적인 4가지 얽힘 상태를 **Bell state**라고 부른다.

$\Phi^+ = \frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)$

$\Phi^- = \frac{1}{\sqrt2}(\vert 00\rangle-\vert 11\rangle)$

$\Psi^+ = \frac{1}{\sqrt2}(\vert 01\rangle+\vert 10\rangle)$

$\Psi^- = \frac{1}{\sqrt2}(\vert 01\rangle-\vert 10\rangle)$

### 번외: EPR역설

$\frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)$ 을 구성하는 두 큐비트를 Alice와 Bob이 나눠 가진다고 해보자. 이론적으로 두 입자를 잘 보존하면서 멀리 떨어진다면 얽힘상태를 유지하면서 무한정 멀어질 수 있다.

이때 "정보"는 절대 광속을 넘어서 도달할 수 없는데, Alice가 자신의 큐비트를 측정해서 결과가 나와버린 순간 B의 큐비트가 뭐였는지를 즉시 알게 된다. 어떻게 B의 큐비트에 대한 정보가 광속보다 빨리 전달될 수 있었을까?

> 해답: 저것은 정보의 전달이 아니다. 예컨데, 정보의 전달이란 Alice가 자신의 번호인 10010을 Bob에게 그대로 10010의 시퀀스를 전달하는 것이다. 하지만 저런 얽힘 상태를 이용해서 정보를 전달하려면 **양자상태가 0으로 관측될지, 1로 관측될지를 정할 수 있어야 한다.** 본질적으로 양자 상태는 확률에 기반한 완벽한 무작위이므로 0을 보내고 싶어도 0이 관측될 거란 보장이 없고, 1을 보내고 싶어도 1이 관측될 거란 보장이 없다. 따라서 Alice와 Bob사이에는 얽힘을 사용하더라도 광속을 뛰어넘어 아무런 정보도 전송할 수 없다.

### density matrix(밀도 행렬)

위에서는 양자 상태를 확률 벡터로 표현하는 법을 다루었다. 하나의 확률 벡터로 표현되는 양자 상태는 pure state라고 한다. 하지만 어떤 양자 상태는 본질적으로 하나의 확률 벡터로 표현되지 않을 수 있다.

예를 들어 70%확률로 $\frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)$, 30%확률로 $\frac{1}{\sqrt2}(\vert 01\rangle-\vert 10\rangle)$인 상태 또한 존재한다. 이러한 상태는 어떻게 표현해야 할까? 첫번째 방법은 이러한 앙상블을 집합으로써 관리하는 것이다.

$$x = \{(\Phi^+, 0.7), (\Psi^-, 0.3) \}$$

하지만 이런 표현법은 다양한 양자 연산들을 계산하기가 어렵고 비직관적이다. 따라서 이러한 상태를 표현하는 또 다른 방법이 필요하다. 이때 사용되는 것이 density matrix이다.

density matrix란 아래 식으로 표현되는 행렬이다.

$$\rho = \sum_i p_i \vert \psi_i\rangle\langle\psi_i\vert $$

$\vert x\rangle$은 $\langle x\vert $의 켤레전치 행렬로서 열 벡터이다. 따라서 둘을 곱하면 n*n 행렬이 나오게 된다. 이처럼 density matrix는 큐비트가 n 개라면 2^n by 2^n 행렬이 된다.

위의 예시를 density matrix로 표현하면 아래와 같다.

$$\rho = 0.7\frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)\frac{1}{\sqrt2}(\langle00\vert +\langle11\vert ) + 0.3\frac{1}{\sqrt2}(\vert 01\rangle-\vert 10\rangle)\frac{1}{\sqrt2}(\langle01\vert -\langle10\vert )$$

$$= \frac{0.7}{2}(\vert 00\rangle\langle00\vert +\vert 00\rangle\langle11\vert +\vert 11\rangle\langle00\vert +\vert 11\rangle\langle11\vert ) + \frac{0.3}{2}(\vert 01\rangle\langle01\vert -\vert 01\rangle\langle10\vert -\vert 10\rangle\langle01\vert +\vert 10\rangle\langle10\vert )$$

$$= \frac{0.7}{2}\begin{pmatrix}1&0&0&1\\0&0&0&0\\0&0&0&0\\1&0&0&1\end{pmatrix} + \frac{0.3}{2}\begin{pmatrix}0&0&0&0\\0&1&-1&0\\0&-1&1&0\\0&0&0&0\end{pmatrix}$$

$$= \begin{pmatrix}0.35&0&0&0.35\\0&0.15&-0.5&0\\0&-0.15&0.15&0\\0.35&0&0&0.35\end{pmatrix}$$

density matrix의 정의로부터, states의 확률은 모두 0 이상이므로 density matrix는 positive semi-definite이다. 이는 density matrix의 고유값이 모두 0 이상임을 의미한다.

이처럼 density matrix는 pure state가 아닌 양자 상태를 표현하는데 유용하다.

Density matrix에서 고윳값들을 뽑아냈다고 생각해 보자. 예를 들어 3큐빗짜리 8*8 density matrix의 고윳값이 0.5, 0.4, 0.1이라면 이 density matrix는 3가지 상태의 앙상블이며, 각각의 상태의 확률이 50%, 40%, 10%라는 것을 바로 알 수 있다. 또한 고유벡터까지 구한다면 각 상태가 무엇인지도 알 수 있다.

이로부터 쉽게 유추할 수 있는 사실은 density matrix 의 고윳값을 전부 더하면 1이 될 것이라는 사실이다. 밀도 행렬의 고윳값의 성질을 정리해 보자.

1. 고윳값은 모두 0 이상 1 이하이다. (positive semi-definite)
2. 고윳값들의 합은 1이다. (확률의 합이 1이기 때문에 자명하다)

그렇다면 density matrix가 $\rho$가 주어졌을 때, 이 상태가 pure state인지 mixed state인지는 어떻게 구별할까? 아래와 같은 관찰을 해보자.

> 만약 pure state라면 밀도 행렬이 저장 중인 앙상블에는 상태 하나만 있을 것이고, 당연히 확률은 1일 것이다. 따라서 확률을 제곱해서 더해도 1이다.

> 만약 mixed state라면 밀도 행렬이 저장 중인 앙상블에는 여러 상태가 있을 것이고, 확률의 제곱을 더하면 1보다 작을 것이다. 예를 들어, 확률이 0.5, 0.4라면 제곱해서 더하면 0.41 < 1 이다.

즉, $\text{tr}(\rho^2) = 1$ 이라면 $\rho$는 pure state이다. 역으로, $\text{tr}(\rho^2) \lt 1$ 이라면 $\rho$는 mixed state이다.

이번 섹션에서 다루었듯이, density matrix의 고윳값과 관련된 다양한 성질들은 실제 양자 상태를 분석하는데 큰 도움이 된다.

## LOCC (Local Operations and Classical Communication)

LOCC는 양자 상태를 다룰 수 있는 메서드 중에서 하나이다. 굉장히 다양한 분야에 활용된다. 사실, 개념 자체는 어렵지 않다. LOCC란 각각 자신이 가진 양자 상태에 임의의 Operator을 가할 수 있고, 그 결과를 (고전적으로) 서로 주고받을 수 있다는 것이다.

주어진 2 qubit 양자 상태가 $\Phi^+ = \frac{1}{\sqrt2}(\vert 00\rangle+\vert 11\rangle)$또는 $\Psi^+ = \frac{1}{\sqrt2}(\vert 01\rangle+\vert 10\rangle)$ 라고 하자. Alice는 첫번째 큐비트를 가져가고 Bob은 두 번째 큐비트를 가져간다. 주어진 양자 상태는 충분히 많아서 각자 자신이 배정받은 큐비트에 어떤 연산도 할 수 있다고 하자.

만약 Alice와 Bob이 정보를 주고받을 수 없다면 받은 큐비트가 아무리 많더라도 둘은 주어진 상태가 $\Phi^+$인지 $\Psi^+$인지 절대로 구분할 수 없을 것이다. 50% 확률로 0, 50% 확률로 1이 나올텐데, 이 정보로는 $\Phi^+$와  $\Psi^+$를 구분할 수 없다.

하지만 Alice와 Bob이 정보를 주고받을 수 있으면 결과가 달라진다. 만약 Alice가 측정한 결과가 0이고 Bob이 측정한 결과가 1이라고 하자. 그리고 서로의 측정 결과를 공유한다. 그렇다면 둘은 원래 상태가 $\Psi^+$이란 것을 쉽게 알 수 있다. 원래 상태가 $\Phi^+$였다면 절대로 두 큐비트가 다를 수 없기 때문이다.

이를 표로 정리하면 다음과 같다.

|Alice 측정결과|Bob 측정결과|주어진 상태|
|:---:|:---:|:---:|
|0|0|$\Phi^+$|
|0|1|$\Psi^+$|
|1|0|$\Psi^+$|
|1|1|$\Phi^+$|

따라서 Alice와 Bob은 2qubit system중에서 1qubit밖에 조작할 수 없지만, LOOC를 통해 주어진 상태를 구별할 수 있다.

## 얽힘 증류 (entanglement distillation)

Quantum Teleportation, Quantum Communication등등 양자 기술을 사용한 어떠한 정보학적 행위들의 핵심은 Bell state(pure entanglement state)를 활용하는 것이다. entanglement가 pure하지 않을수록 확률적인 노이즈가 생길 수 있다. 따라서 임의의 양자 상태로부터 pure singlet form($\Psi^-$)을 추출해 내는 것이 굉장히 중요하다[3]. 이와 관련된 것이 **얽힘 증류**이다.

얽힘 정도를 정의하는 척도에 따르면 2 qubit에서 가장 순수한 얽힘 상태는 Bell states들이다(왠지 그럴 꺼 같지 않은가?). 나머지 얽힌 상태들도 얽혀 있긴 하지만, 순수하게 얽혀 있지는 않다.

entanglement distillation은 얽힘을 포함하고 있는 양자 상태에서 좀 더 순수한 얽힘을 추출해내는 개념이다. 정확히는, 얽힘을 포함하고 있는 양자 상태에서 연산을 가해서 임의 두 큐빗이 Bell state에 가까워지도록 만드는 방법이다.

양자 얽힘을 매듭에 비유해보자. Distillation은 아래처럼 제일 기본적인 매듭을 분리해내는 것이다.

![img](https://images.pexels.com/photos/4021575/pexels-photo-4021575.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)


하지만 아주 복잡한 매듭이라면 저런 기본적인 매듭을 분리해내지 못할 수도 있다. 이와 마찬가지로 제일 기본적인 bell state entanglement를 분리해낼 수 없는 얽힘 상태가 존재한다. 이를 bound entanglement라고 한다.

## bound entanglement

bound entanglement란 얽혀 있지만, distillation이 불가능한 얽힘 상태이다. distillation이 가능한 얽힘 상태를 fure entanglement라고 특별히 지칭한다[3]. 현실 세계에서는 free entanglement보다 bound entanglement가 훨씬 더 희귀한 것으로 추정된다[4].

그렇다면 임의의 양자 상태가 bound entanglement인지는 어떻게 알아낼 수있을까? 정의에 따라 (1) entanglement가 있어야 하고 (2) distillation이 불가능해야 한다. 이 두 조건을 만족하는 것과 bound entanglement임은 필요충분 조건이다.

따라서 bound entanglement를 알아내기 위해서는 (1) entanglement가 있는지, (2) distillation이 가능한지를 알아내야 한다. 이 두 방법을 각각 살펴보자.

#### (1) check entanglement

주어진 양자 상태가 entanglement를 포함하고 있는지 어떻게 알아낼 까? 우선 양자 상태는 n*n 행렬로 표현될 것이다. 이제 이 행렬에 얽힘이 있는지 없는지를 알아내면 된다.

놀랍게도 이 문제는 NP-hard이다[4]. 대신 얽힘이기 위한 필요조건 그나마 쉽게 판단 가능하다. 즉, 어떤 상태가 얽힘이 아닌 조건은 찾을 수 있다. 그 조건을 **Peres–Horodecki criterion**이라 한다.

> **Peres-Horodecki criterion**
density matrix가 seperable(얽힘이 아닌) 상태라면, partial transpose를 취한 density matrix의 eigenvalue는 모두 0 이상이다.

먼저 상태의 분리 가능성을 본다면 어떤 상태가 **잘 정의되는** 두 subsystem으로 분리되어야 한다. 잘 정의된다는 것은 상태의 고윳값이 모두 0 이상이란 점이다. 따라서 이러한 조건이 나온 것이다. 하지만, 두 subsystem의 고윳값이 모두 0 이상이라고 해서 상태가 분리 가능하다고 할 수는 없다. 이는 상태가 얽혀 있지만, partial transpose를 취한 상태의 고윳값이 모두 0 이상일 수도 있기 때문이다.

이러한 조건 때문에 PPT criterion(Positive Partial Transpose)이라고도 한다. 이 조건은 2x3 상태까지는 얽힘을 판단하는데 필요충분 조건이라는 것이 증명되어 있지만 그 위 차원부터는 아니다. 실제로 다양한 반례들이 있는데, 이것이 우리가 살펴볼 bound entanglement 상태들이다.

#### (2) check distillable

distillable을 판단하는 것은 얽힘을 판단하는 것보다 쉽다[3]. 그리고 그 방법은 동일하게 PPT criterion을 사용하는 것이다 즉, partial transpose를 취한 상태의 고윳값이 모두 0 이상이면 distillable하다고 판단한다.

#### detect bound entanglement

bound entanglement는 위의 명제들을 조금만 정리해 보면 등장한다.

1. PPT criterion을 만족하면 undistillable하다
2. PPT criterion을 만족하지 않으면 얽혀있지 않지만, 만족할 때는 얽힘 여부를 모른다.
3. 따라서 PPT criterion을 만족하는 얽힘 상태가 존재할 수 있다.
4. 이러한 상태를 bound entanglement라고 한다.

이제 얽혀 있지만 얽힘 증류는 불가능한 상태의 존재 가능성을 발견했다. 실제로 그 예시가 존재한다.

BES 2x4 state matrix:

\[
\frac{1}{7a+1}
\begin{bmatrix}
a & 0 & 0 & 0 & 0 & a & 0 & 0 \\
0 & a & 0 & 0 & 0 & 0 & a & 0 \\
0 & 0 & a & 0 & 0 & 0 & 0 & a \\
0 & 0 & 0 & a & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{1+a}{2} & 0 & 0 & \frac{\sqrt{1-a^2}}{2} \\
a & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & a & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & a & 0 & \frac{\sqrt{1-a^2}}{2} & 0 & 0 & \frac{1+a}{2} \\
\end{bmatrix}
\]

조금만 계산을 해 보면, 위의 density matrix는 PPT criterion을 만족한단 것을 확인할 수 있다. 따라서 undistillable하다. 하지만, 이 상태는 얽혀 있다. 따라서 위 예시가 bound entanglement state의 예시이다.

PPT criterion은 2x3 이하 크기의 상태에는 얽힘 여부와 필요충분 조건이다. 따라서 bound entanglement는 2x4 이상의 상태부터 존재할 수 있으며, 실제로 위 예시는 2x4 state인 bound entanglement의 예시이다.

### 참고문헌

[1] https://wiki.quist.or.kr/index.php/%EC%96%91%EC%9E%90%EC%A0%95%EB%B3%B4_%EA%B0%9C%EC%9A%94

[2]Bennett, Charles H., et al. "Purification of noisy entanglement and faithful teleportation via noisy channels." Physical review letters 76.5 (1996): 722.

[3] M. Horodecki, P. Horodecki, and R. Horodecki, Mixed-State Entanglement and Distillation: Is there a “Bound” Entanglement in Nature?, Physical Review Letters 80, 5239 (1998)

[4] Hiesmayr, Beatrix C. "Free versus bound entanglement, a NP-hard problem tackled by machine learning." Scientific Reports 11.1 (2021): 19739.