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

VQA는 PQC를 활용하여 주어진 Observable의 최솟값을 찾는 알고리즘이다. 여기서 최소값은 Observable을 행렬로 나타냈을 때의 최소의 고유값이다. 그리고 마지막 상태는 해당 고윳값에 대응하는 고유벡터의 상태이다 (구체적으로, 고유벡터가 $\vert v\rangle$ 이라면 최종 양자상태는 $\vert v\rangle\langle v\vert $ 이다.)

여기까지는 앞선 글에서 다루어본 내용이다. 하지만 VQA에는 명확한 한계가 존재한다.

## 지수적으로 증가하는 파라미터 수

n큐빗 system에서 주어진 어떤 Observable의 값도 PQC를 통해 최소화시키려면 파라미터 수가 몇 개나 되어야 할까?

Observable의 값을 최소화시키는 경우는 고유값이 최소인 고유벡터일 것이다. 하지만 해당 고유벡터는 2^n개의 원소를 가지고 있다.
각 원소는 a+bi꼴이므로 원소당 2개의 자유도가 있다고 할 수 있다. 하지만 고유벡터의 norm이 1이어야 하는 조건으로 인해 실제 자유도는 좀 더 작지만, 대략 따져보면 자유도는 4^n정도일 것이다. (최소한 2^n은 넘는다)

파라미터 m개를 사용한 PQC로 자유도 4^n인 벡터를 표현할 수 있다면 m은 O(4^n)에 비례해야 함이 자명하다.
어떠한 Observable이 주어져도 PQC로 해를 찾을 수 있음을 보장하려면 큐빗 수에 지수적으로 증가하는 개수의 파라미터 수가 필요하다.

파라미터는 결국엔 **고전적으로** 저장되고, 업데이트되기 때문에 지수적인 파라미터 개수의 증가는 아무리 양자컴퓨터 시대가 와도 실현이 불가능하다. 큐빗 수가 15개 정도만 되어도 10억개 가량의 파라미터가 필요하다. 최적화를 한다고 해도 수십 큐빗 단위에서나 가능할 것이다.

이 때문에 더 큰 시스템의 최적화 문제를 해결하기 위해서는 PQC를 더 효율적으로 디자인해야 한다. 여기서 symmetry가 나온다.

# symmetry란?

symmetry는 대칭을 의미한다. 수학적으로는 어떠한 행렬 작용에 대해 값이 불변이라는 의미로 symmetry를 사용하며 U(1) symmetry, SU(2) symmetry, SU(N) symmetry등등이 있다. 하지만 여기서는 수학적인 개념을 제쳐 두고, symmetry를 활용하면 어떻게 **더 적은 파라미터 수로도 답을 찾을 수 있는지**를 다뤄보고자 한다. 결국 이게 symmetry를 사용하는 목적이기 때문이다.

바로 위 장에서 우린 모든 가능한 변환을 PQC로 구현하기 위해서는 큐빗 수에 대해 지수적으로 증가하는 파라미터 수가 필요하다는 것을 러프하게 알아보았다. 하지만 굳이 모든 경우를 볼 필요가 없다면? 여기서부터 symmetry를 사용한 아이디어가 시작된다. 예시로 양자화학에서 많이 인용되는 전자의 오비탈 배치 문제를 살펴보자.

## 전자의 최적 오비탈 배치

<p align="center"><img src="http://butane.chem.uiuc.edu/pshapley/GenChem1/L10/h3.gif"></p>
<center><b>그림 1. 수소분자의 전자 오비탈</b>(출처:  uiuc.edu)</center>

수소분자 H2를 고려해 보자. 분자에 존재하는 전자는 총 2개이고, 전자가 배치되어야 하는 오비탈도 2개이다. 일반적으로 생각해 보면 위 그림에서 최적의 배치는 $\sigma$ 오비탈에 전자 두개가 업스핀, 다운스핀으로 배치되는 경우이다. 바닥상태부터 전자가 배치되어야 하기 때문이다.

하지만 양자역학적인 효과를 고려해 보면, 전자가 여러 오비탈에 배치되는 경우가 확률적으로 **중첩**되어 있을 수 있다. 예컨데, 99%확률로 $\sigma$오비탈에 2개가 배치되는 경우 + 1%확률로 $\sigma$ 오비탈에 하나, $\sigma^*$오비탈에 하나가 배치되는 경우가 중첩될 수도 있다. 하지만 양자화학적으로는 알고 있는게 거의 없어서 여기까지만 배경 설명을 하고 넘어가도록 하겠다.

H2분자의 오비탈에 전자를 배치하는 경우를 살펴보자. 오비탈은 2개이고, 각 오비탈에 업 스핀, 다운 스핀이 들어갈 수 있으므로 실제로 전자를 배치할 수 있는 "자리"는 2*2=4자리이다.

<p align="center"><img src="https://github.com/younginch/subcloud-chrome/assets/17401630/64d6bf37-bdcf-4944-95b0-5c7be64256ac"></p>
<center><b>그림 2. 오비탈 4개에 전자가 배치될 수 있는 경우</b></center><br/>


이제 전자의 배치를 0,1 인코딩으로 표현할 수 있다. 두 비트씩 바닥상태 오비탈에 대응되며 첫 큐빗은 업스핀, 두번째 큐빗은 다운스핀에 대응하는 것이다. 위의 **그림 2**를 생각하면 쉽게 이해가 될 것이다. 만약 두번째 오비탈에 다운스핀, 네번째 오비탈에 업스핀으로 전자를 두개 배치했다면 00010010 으로 인코딩 할 수 있다.

이렇게 인코딩된 스트링이 11000000, 10100000 등등에 따라 **전자 배치의 에너지**도 달라질 것이다. 아마도 즉, 상태 -> 에너지로 매핑해주는 함수를 자연스럽게 떠올려 볼 수 있다. 양자상태에서 에너지를 표현하는 관측량을 Hamiltonian이라고 하는데, 이제 이 문제를 그대로 양자컴퓨팅에 맞게 옮겨오기만 하면 된다.

다시 수소 원자로 돌아와보면 가장 안정적인 단일 상태는 **1100**일 것이다. 바닥상태 오비탈에 전자를 꽉 채웠기 때문이다. 이제 자연스럽게 이 인코딩을 큐비트로 옮겨올 수 있다. $\vert 1100\rangle$ 이런식으로 옮기면 쉽게 대응된다. $\vert 0110\rangle$은 첫 오비탈의 다운스핀에 전자가 배치되고, 두번째 오비탈에 업스핀으로 전자가 배치된 경우이다. 이제 전자의 배치를 양자상태로 인코딩 하였으니, 이에 대응되는 에너지를 나타낼 Hamiltonian을 설계할 차례이다. 이는 우리가 직접 만들 필요는 없고, pennylane의 qchem라이브러리를 사용하면 자동으로 Hamiltonian을 알려준다.

``` python
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
basis_set = "sto-3g"
electrons = 2

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    basis=basis_set,
)
print("hamiltonian is\n\n", H)
print("qubits is", qubits)
```

이 코드의 실행결과는 아래와 같다.

```text
hamiltonian is

   (-0.2427450126094144) [Z2]
+ (-0.2427450126094144) [Z3]
+ (-0.042072551947439224) [I0]
+ (0.1777135822909176) [Z0]
+ (0.1777135822909176) [Z1]
+ (0.12293330449299361) [Z0 Z2]
+ (0.12293330449299361) [Z1 Z3]
+ (0.16768338855601356) [Z0 Z3]
+ (0.16768338855601356) [Z1 Z2]
+ (0.17059759276836803) [Z0 Z1]
+ (0.1762766139418181) [Z2 Z3]
+ (-0.044750084063019925) [Y0 Y1 X2 X3]
+ (-0.044750084063019925) [X0 X1 Y2 Y3]
+ (0.044750084063019925) [Y0 X1 X2 Y3]
+ (0.044750084063019925) [X0 Y1 Y2 X3]
qubits is 4
```

내부적으로 다양한 변환 (Jorden-Wigner Transformation)등을 거쳐서 뽑아낸 결과이긴 하지만, hamiltonian이 구해진 걸 확인할 수 있다. 원래 Hamiltonian은 2^n * 2^n 크기의 행렬이지만 여기서는 파울리 기저들로 행렬을 분해해서 표현했다.

>여기까지가 배경 설명이었다. VQA를 사용하면 주어진 Observable(여기서는 Hamiltonian)을 최소화 하는것이 가능하다. 이제 Observable이 주어졌으니 최적화 하기만 하면 된다... 그런데 복잡한 분자로 넘어가면 어떨까? 당장 질소 분자만 보아도 배치해야할 고려해야 할 오비탈이 8개 정도인데, 그럼 16큐비트나 필요하다. 러프하게 생각해 보면 최소한 2^16 ~ 4^16개의 파라미터가 필요하므로 돌려보는게 거의 불가능할 것이다.

## Symmetry 등장

파라미터가 기하급수적으로 늘어나는 이유는 모든 경우를 다 탐색해야 하기 때문이다. 하지만 이 경우엔 그럴 필요가 없으며, 오히려 그러면 잘못된 결과를 구하게 된다.
왜 그럴까? 여기서 발생하는 제한 조건 때문이다.

수소분자 혼성 오비탈에 전자를 두개 배치하는 가능한 경우는 4C2 = 6개 뿐이다. $\vert 1100\rangle$, $\vert 1010\rangle$, $\vert 1001\rangle$, $\vert 0110\rangle$, $\vert 0101\rangle$, $\vert 0011\rangle$이다. 굳이 모든 경우를 탐색해볼 필요 없이, 저 6개의 경우만 탐색해도 충분하다. 그럼 **입력을 $\vert 1100\rangle$으로 주고 1의 개수를 보존하는 게이트만 사용하여 회로를 구성하면 되지 않을까?** 이것이 symmetry를 활용하는 핵심 아이디어이다.

1의 개수를 보존하는 게이트는 어떤 게 있을까? 쉽게 떠오르는건 SWAP게이트가 있다. 두 큐빗을 swap하므로 당연히 1의 개수도 보존한다. 다른 게이트들도 많지만, 대표적으로 `SingleExcitation` gate가 있다. 굉장히 생소한 게이트이긴 하지만, 이 게이트의 작용은 다음과 같다.

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/5add0436-a407-41f9-8353-c9f2727c5327"></p>
<center><b>그림 3. SingleExcitation gate의 행렬</b></center><br/>

잘 보면 $\vert 00\rangle$, $\vert 11\rangle$은 그대로이고 $\vert 01\rangle$ $\vert 10\rangle$사이의 변환만이 존재한다. 따라서 1 개수를 유지한다. 그럼 왜 SingleExcitation gate를 사용할까? 바로 '전자 1개'가 들뜬 것을 고려하기 위해서이다. 하지만 지금은 전자를 배치할 수 있는 방이 4개이고 전자를 2개 배치해야 하므로 전자 2개가 모두 들뜨는 경우도 가능하다. 따라서 전자 두개가 들뜬 경우를 고려하기 위해 차용한 gate가 `DoubleExcitation gate`이다.

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/ccc7f388-f376-4bfd-8051-597bf649e570"></p>
<center><b>그림 4. DoubleExcitation의 작용</b></center><br/>

이 역시 1의 개수를 보존함을 확인할 수 있다.

위 두개의 게이트에 모두 파라미터가 들어간단 점을 눈치챘는가? 이제 전자 1개가 들뜨는 경우, 전자 2개가 들뜨는 경우를 파라미터화해서 그 값을 수정한다면 효율적으로 **가능한 전자 배치**만을 탐색하고 최적값을 찾을 수 있다. 파라미터는 몇개나 필요할까? 전자는 스핀을 유지하면서 들떠야 하기 때문에 전자 1개가 들뜨는 경우 2개, 전자가 모두 들뜨는 경우가 1개로 총 3개이다. 따라서 위의 게이트를 사용하면 **파라미터 3개만으로** PQC를 구성하여 답을 찾을 수 있다.

## 결과

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/81869393-0675-4ed0-a672-7b62762ffe08"></p>
<center><b>그림 5. 최적화 결과</b></center><br/>

$E_{hf}$는 초기 상태였던 $\vert 1100\rangle$의 에너지를 말한다. 즉 우리가 고전적으로 생각하는 바닥상태의 에너지이다. 학습 결과 그보다 에너지가 더 낮은 상태가 발견되었다. 그리고 이 값은 처음에 계산한 Hamiltonian으로부터 바로 고전적으로 계산한 최적해와 동일했다. 따라서 단 3개의 파라미터로도 고전적인 최솟값과 동일한 결과를 얻어내었다.

그럼 대체 어떤 상태이길래 $\vert 1100\rangle$보다 에너지가 더 낮은 것일까? 코드로 확인해 보니 $\vert 1100\rangle$이 98.9%, $\vert 0011\rangle$이 1.1%로 중첩된 경우였다. 즉, 양자역학적으로 바닥상태가 아닌 다른 상태가 중첩되어 있을 때, 전체 에너지가 낮아진 것이다. 이는 굉장히 신기한 결론이다. 물론 이 결론은 파라미터를 많이 쓴 일반 PQC로도 얻을 수 있음에 유의하자.

## 결론

Symmetry를 사용하면 그렇지 않을 때보다 훨씬 적은 파라미터를 사용하여 훨씬 빠른 시간에 정답을 찾을 수 있다. 또한, Symmetry를 활용하지 않은 경우 정답을 찾음이 보장되려면 큐빗 수에 대해 exponential한 파라미터 수가 필요하여 실제로 적용하기에 문제가 많다. Symmetry도 symmetry나름이겠지만, 위의 경우들에는 다항 개수의 파라미터만 사용하여도 정답을 찾을 수 있으므로 굉장히 효율적이다.

따라서 VQA를 사용하여 주어진 문제를 해결하고 싶다면, 해당 문제에 대칭성이 있는지를 파악한 뒤에 만약 존재한다면 최대한 활용해야 한다.

### 참고문헌

[1] https://docs.pennylane.ai/en/stable/introduction/chemistry.html

[2] https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html

[3] https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.molecular_hamiltonian.html