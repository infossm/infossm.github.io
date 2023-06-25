---
layout: post
title: "variational quantum algorithm"
date: 2023-06-25
author: red1108
tags: [quantum, variational-quantum-circuit, variational-quantum-eigensolver, vqa, vqc, vqe]
---

## 서론

> Keywords: `VQC(Variational Quantum Circuit)`, `VQA(Variational Quantum Algorithm)`, `VQE(Variational Quantum Eigensolver)`, `Observable`, `Hamiltonian`

> 기본적인 양자상태의 표현과 gate가 어떻게 작동하는지는 이해하고 있다는 가정하에 글을 작성하였다.

`VQC`, `VQA`, `VQE` 모두 양자 컴퓨팅에서 중요한 개념들이지만 한국어로 제대로 소개된 글이 없기에 이 글을 통해 소개하고자 한다

## Quantum Circuit

<p align="center"><img src="https://qiskit.org/documentation/locale/ko_KR/_images/tutorials_circuits_advanced_03_advanced_circuit_visualization_7_0.png" ></p>
<center><b>그림 1. 양자회로의 모습</b></center>

위 그림 1은 일반적인 양자회로의 모습을 나타내고 있다. 고전 회로에서 1-input gate인 Inverter과 2-input gate인 AND, OR, XOR게이트, 3-input gate인 3-input OR 등등이 있듯이 양자 회로에도 다양한 개수의 입력을 받는 회로가 있다.

양자 회로를 나타낸 그림 1에서는 1-input gate인 H, X gate와 2-input gate인 CNOT(Controlled-Not)게이트를 확인할 수 있다. 이 게이트들은 각각 고정된 역할을 맡으며 파라미터에 의해 조정되지 않는다.

## Variational Quantum Circuit(VQC)


<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/5ab9e3a8-0315-4786-af93-d734c4b52afe"></p>
<center><b>그림 2. 파라미터에 의해 조정되는 양자 회로의 모습</b></center>

하지만 위 그림 2와 같은 회로는 `RX(rotate x)`, `RY(rotate y)`, `RZ(rotate z)` 게이트가 사용되었고, 이 게이트들은 회전 게이트로써 얼마나 **회전** 시킬지에 대한 정보를 파라미터로 받는다. 잘 보면 그림 2의 아래쪽에 써져 있는 값들을 볼 수 있는데 저 값이 파라미터이다. 예를 들어, RX(0.3) 은 0.3만큼 X축으로 회전시키는 게이트이다.

우리가 파라미터들을 가지고 있고, 이 값을 조정한다면 회로의 출력 또한 자연히 바뀔 것이다. 이러한 회로를 **Variational Quantum Circuit**이라고 한다.

## Variational Quantum Algorithm(VQA)

양자 컴퓨팅 세계에서 관측 가능한 모든 값들은 `Observable`로 표현된다. 예를 들어, 운동량을 나타내는 `Observable`, 또는 에너지를 나타내는 `Observable`, 단순히 별 의미 없는 특성을 나타내는 `Observable` 등등이 있다.

그럼 파라미터들로 조정되는 회로를 구성한 뒤, Observable의 값을 측정하는 회로를 생각해 보자. 파라미터의 값이 바뀔수록 당연히 Observable의 기댓값도 바뀔 것이다.

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/58caaeff-e84c-436c-9ec6-f454107e15c4"></p>
<center><b>그림 3. VQA의 대략적인 모습</b></center>

파라미터의 값이 바뀔 때 Observable의 기댓값이 바뀐다면 이를 최소화하는 파라미터를 찾는 것이 가능하지 않을까 하는 생각이 든다. 이 아이디어에서 출발한 것이 `Variational Quantum Algorithm(VQA)`이다. 

더 구체적으로는, 고전적으로 회로를 시뮬레이션 할 때는 계산과정을 전부 알고 있으므로 머신러닝에서 gradient를 계산하는것과 같은 방법으로 최적화가 가능하다. 머신러닝과 거의 유사하다. Optimizer도 Adap optimizer 등등을 사용한다.

만약 실제 양자컴퓨터를 사용한다면 이 방법이 불가능할까? 다행히도 parameter shift rule이라는 좋은 방법이 존재한다. 양자컴퓨터에서 내부적으로 얽힘을 사용한 복잡한 계산을 하여도, 그 계산과정을 따라갈 수 없는데도 parameter의 gradient를 구할 수 있게 해주는 방법이다. 이것도 한국어로 제대로 소개된 글이 없으니 다음에 소개하도록 하겠다.

여튼, 파라미터를 바꿔가며 Observable을 cost function으로 생각했을때 최소값을 찾는 것이 가능하다. 이것이 **Variational Quantum Algorithm**이다.

## Variatonal Quantum Eigensolver(VQE)

VQE는 VQA의 특수한 집합이다. VQA는 파라미터 조정을 통한 최적화가 가능한 모든 문제에 적용 가능한데, VQE는 이름에서 알 수 있듯이 주어진 Observable의 eigenvector을 찾는 알고리즘이다.

갑자기 왜 observable의 고유벡터가 튀어나올까? 이건 Observable의 성질과 관련 있다. Observable의 관측값은 Observable의 고유값의 조합이다. 임의 상태 벡터를 Observable의 고유벡터들의 선형 결합으로 표현되었을때, 그 확률에 비례하여 고윳값 중에 하나가 관측된다. 따라서 Observable의 평균값은 고유값의 조합이 된다.

이제 Observable의 관측값을 최소화하는 VQA를 생각해 보자. 모든 관측값은 Observable의 고윳값들에 확률벡터를 내적한 값이므로 **관측값이 최소가 된다는 것은 Observable의 최소 고유값**이 된다는 듯이다. 이때의 상태는 당연히 Observable의 최소 고유값에 대응되는 고유벡터가 된다. 이 알고리즘이 Variational Quantum Eigensolver라고 불리는 이유가 여기에 있다.

## Hamiltonian

이제 Hamiltonian을 생각하자. Hamiltonian는 Observable의 일종으로, 대응되는 값이 **에너지**일때 그 Observable을 hamiltonian이라고 부른다.

다시 앞의 VQE를 떠올려 보자. VQE에서 hamiltonian의 최소 관측값을 찾았다는 것은 hamiltonian의 최소 고유값을 찾았다는 것이다. 그런데 Hamiltonian의 최소 고윳값은 정의에 의해 계가 취할 수 있는 여러 에너지 준위들 중 최소값이라는 의미이다.

즉, VQE를 사용하여 Hamiltonian의 최소 고유값을 찾는 것은 계의 바닥상태 에너지를 찾는 것과 같다.

## VQE의 응용 - Quantum chemistry

### 배경 설명

이제 VQE의 가장 간단한 응용인 오비탈의 바닥상태에서 전자배치를 찾는 문제를 풀어 보자. 이 문제가 좀 더 복잡한 이유는, 양자역학적으로 **중첩**된 상태가 바닥상태가 될 수 있다는 점이다. 즉, s와 p오비탈에 전자를 두개 배치하는 경우를 떠올려보자. 일반적으로 바닥상태는 당연히 s오비탈에 스핀 업, 다운으로 전자 두개를 배치하는 경우이다. 그러나 중첩을 고려한다면

<br>

<center>90%확률로 s오비탈에 전자 두개, 10%확률로 s오비탈, p오비탈에 전자 하나씩</center>

<br>

이 배치가 좀 더 에너지가 낮을 수도 있다. 굉장히 신기한 일이다. 일반적으로 s오비탈에 전자가 전부 배치되는것이 제일 에너지가 낮을 것이라 생각하지만 중첩을 고려한다면 여러 배치가 중첩된 상태가 에너지가 더 낮을 수도 있다.

이를 고려하려면 당연히 고전적인 계산보다는 양자컴퓨팅을 사용한 계산이 자연스러울 것이다. 따라서 양자 컴퓨팅을 사용한 해결방법을 다뤄 보자. 간편하게 VQE를 코딩할 수 있는 `pennylane` 을 사용하여 설명할 것이다.

### VQE를 사용한 해결 - 개요

VQE를 사용하여 대부분의 문제를 해결하는 방법은 아래와 같이 요약할 수 있다.

1. 문제를 파악하고, 어떻게 상태를 양자 상태에 인코딩하여 표현할지 결정한다.
2. 인코딩을 바탕으로 그 계의 에너지를 설명할 Hamiltonian을 구성한다.
3. VQE를 사용하여 그 Hamiltonian의 최소 고유값을 찾는다.
4. 구한 최소 고윳값이 계의 최소 에너지이고, 그때의 상태가 계의 바닥상태이다.

### VQE를 사용한 해결 - 양자 상태 인코딩

hartree-fock state. second quantization.

### VQE를 사용한 해결 - Hamiltonian 구성

pennylane에서 계산해 줌. 배경에는 Jordan-Wigner Transformation이 있다.

### VQE를 사용한 해결 - 최소 고유값 찾기

#### ansatz구성

#### cost function 구성

#### 최적화

### 참고문헌