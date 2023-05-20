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

고전적인 비트는 0 또는 1의 값만 가질 수 있다. 큐비트도 마찬가지로 0의 상태, 1의 상태를 가질 수 있다. 이 상태를  $|0 \rangle$, $|1 \rangle$ 이라 한다. 여기서 더 나아가 큐비트는 0 또는 1을 가질 확률이 정의된다. 예를 들어, 0.5 확률로 $|0 \rangle$, 0.5 확률로 $|1 \rangle$ 을 가진다면 이 상태는 $|\psi\rangle = \frac{1}{\sqrt2}|0 \rangle + \frac{1}{\sqrt2}|1 \rangle$ 이렇게 정의된다. 즉, 상태 앞에 붙어있는 수의 제곱을 해야 실제 확률이 된다. 곱해지는 수는 여기서는 실수이지만, 실제로는 복소수 범위이다.
이제 위에서 만든 $|\psi\rangle$을 관측한다면 50%확률로 $|1\rangle$, 50%확률로 $|1\rangle$ 의 상태가 관측되는 것이다.

잘 생각해 보면 $\psi$ 는 어차피 $|0 \rangle$과 $|1 \rangle$에 적당히 수를 곱한 상태이다. 따라서 매번 $|\psi\rangle = \frac{1}{\sqrt2}|0 \rangle + \frac{1}{\sqrt2}|1 \rangle$ 이렇게 쓰기보다 계수만 가져와서 벡터로 표현하는게 더 편하다. $|\psi\rangle = \begin{bmatrix} \frac{1}{\sqrt2}\\ \frac{1}{\sqrt2}\end{bmatrix}$ 이렇게 표현한다. 사실, $|x\rangle$ 이라는 표현 자체가 열 벡터를 의미한다. $\langle x|$ 는 그것의 켤레전치이다(conjugate transpose). 즉, $\langle x| = |x\rangle ^ \dagger = \begin{bmatrix}\frac{1}{\sqrt2}&\frac{1}{\sqrt2}\end{bmatrix}$ 이다.


### 여러 큐비트

2개의 큐비트를 관측하면 얻을 수 있는 값은 00, 01, 10, 11일 것이다. 이를 $|00\rangle$, $|01\rangle$, $|10\rangle$, $|11\rangle$ 로 표현한다. 수학적으로는 $|xy\rangle = |x\rangle \otimes |y\rangle$ 이다.

### density matrix(밀도 행렬)

하나의 확률 벡터로 표현되는 양자 상태는 

## 양자 얽힘

요즈음은 유튜브에도 양자 얽힘에 대해 정말 많은 내용이 나오고 있다. 특히 양자 얽힘 상태가 존재한다면 광속을 넘어 정보를 전달할 수 있는것이 아닌가? 라는 물음에서 출발한 EPR역설은 과학 유튜버의 단골 소재이다. Bound entanglement에 대해 알기 위해서는 그것의 상위 분류인 entanglement부터 알아야 한다.

> 양자 얽힘이란 여러 양자 상태들의 그룹이 각각의 양자 상태들에 대해 독립적으로 기술되지 않는 상태를 말한다.

예를들어, $\frac{1}{2}(|00\rangle+|01\rangle+|10\rangle+|11\rangle)$에서 첫 번째 큐비트를 먼저 관측한다고 해보자. 0이 나올 확률도 50%이고, 1이 나올 확률도 50%이다. 반대로, 저 상태에서 두 번째 큐비트를 관측해도 0이 나올 확률과 1이 나올 확률은 같다.

첫 번째 큐비트를 관측하면 두 번째 큐비트에 대한 정보가 달라지는가? 여전히 두 번째 큐비트는 0일수도 1일수도 있다. 아무런 영향을 주지 못한다.

그렇다면 $\frac{1}{\sqrt2}(|00\rangle+|11\rangle)$ 을 살펴 보자. 첫 번째 큐비트와 두 번째 큐비트를 따로따로 본다면 0과 1의 확률은 반반이다. 그럼 첫 번째 큐비트를 관측해 보자. 만약 0이 나왔다면, 두 번째 큐비트의 값은 반드시 0이어야 한다. 왜냐하면 가능한 상태 조합 중에서 01은 없었기 때문이다.

이와 같이 여러 큐비트를 포함한 계가 각 큐비트의 독립적인 상태들의 텐서 곱으로 표현되지 않는다면 해당 계는 얽혀있다고 한다.

특별히, 대표적인 4가지 얽힘 상태를 **Bell state**라고 부른다.

$\Phi^+ = \frac{1}{\sqrt2}(|00\rangle+|11\rangle)$

$\Phi^- = \frac{1}{\sqrt2}(|00\rangle-|11\rangle)$

$\Psi^+ = \frac{1}{\sqrt2}(|01\rangle+|10\rangle)$

$\Psi^- = \frac{1}{\sqrt2}(|01\rangle-|10\rangle)$

### 번외: EPR역설

$\frac{1}{\sqrt2}(|00\rangle+|11\rangle)$ 을 구성하는 두 큐비트를 Alice와 Bob이 나눠 가진다고 해보자. 이론적으로 두 입자를 잘 보존하면서 멀리 떨어진다면 얽힘상태를 유지하면서 무한정 멀어질 수 있다.

이때 "정보"는 절대 광속을 넘어서 도달할 수 없는데, Alice가 자신의 큐비트를 측정해서 결과가 나와버린 순간 B의 큐비트가 뭐였는지를 즉시 알게 된다. 어떻게 B의 큐비트에 대한 정보가 광속보다 빨리 전달될 수 있었을까?

> 해답: 저것은 정보의 전달이 아니다. 예컨데, 정보의 전달이란 Alice가 자신의 번호인 10010을 Bob에게 그대로 10010의 시퀀스를 전달하는 것이다. 하지만 저런 얽힘 상태를 이용해서 정보를 전달하려면 **양자상태가 0으로 관측될지, 1로 관측될지를 정할 수 있어야 한다.** 본질적으로 양자 상태는 확률에 기반한 완벽한 무작위이므로 0을 보내고 싶어도 0이 관측될 거란 보장이 없고, 1을 보내고 싶어도 1이 관측될 거란 보장이 없다. 따라서 Alice와 Bob사이에는 얽힘을 사용하더라도 광속을 뛰어넘어 아무런 정보도 전송할 수 없다.

## LOCC (Local Operations and Classical Communication)

LOCC는 양자 상태를 다룰 수 있는 메서드 중에서 하나이다. 굉장히 다양한 분야에 활용된다. 사실, 개념 자체는 어렵지 않다. LOCC란 각각 자신이 가진 양자 상태에 임의의 Operator을 가할 수 있고, 그 결과를 (고전적으로) 서로 주고받을 수 있다는 것이다.

주어진 2 qubit 양자 상태가 $\Phi^+ = \frac{1}{\sqrt2}(|00\rangle+|11\rangle)$또는 $\Psi^+ = \frac{1}{\sqrt2}(|01\rangle+|10\rangle)$ 라고 하자. Alice는 첫번째 큐비트를 가져가고 Bob은 두 번째 큐비트를 가져간다. 주어진 양자 상태는 충분히 많아서 각자 자신이 배정받은 큐비트에 어떤 연산도 할 수 있다고 하자.

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

### check entanglement

주어진


### 참고문헌

[1] https://wiki.quist.or.kr/index.php/%EC%96%91%EC%9E%90%EC%A0%95%EB%B3%B4_%EA%B0%9C%EC%9A%94

[2]Bennett, Charles H., et al. "Purification of noisy entanglement and faithful teleportation via noisy channels." Physical review letters 76.5 (1996): 722.

[3] M. Horodecki, P. Horodecki, and R. Horodecki, Mixed-State Entanglement and Distillation: Is there a “Bound” Entanglement in Nature?, Physical Review Letters 80, 5239 (1998)

[4] Hiesmayr, Beatrix C. "Free versus bound entanglement, a NP-hard problem tackled by machine learning." Scientific Reports 11.1 (2021): 19739.