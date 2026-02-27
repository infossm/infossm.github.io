---
layout: post
title: "QAC0 와 PARITY 문제"
date: 2025-02-26
author: red1108
tags: [quantum, quantum-computing, quantum-complexity]
---

### 소개

이 글은 [Quantum Complexity](https://infossm.github.io/blog/2025/12/28/quantum-complexity/)의 후속편입니다. 지난 글에서 QAC$^0$, QAC$^0_f$, PARITY 문제의 배경을 소하고, fourier analysis 를 소개했다면, 이번 글에서는 QAC$^0$ 회로가 PARITY를 풀수 있는지 없는지에 대한 현재까지의 연구를 소개하는게 목표입니다.

### 이번 글에서 다룰 것들

먼저 문제의 세팅을 짚고 넘어갈 예정입니다. QAC$^0$, QAC$^0_f$ 와 같은 복잡도 클래스를 간단히 다루고 PARITY 문제를 다시 소개합니다.

QAC$^0$가 PARITY를 계산할 수 있는지 없는지는 아직까지 Open problem입니다. 이와 관련되어 lower bound, upper bound를 증명하는 연구들을 소개하겠습니다. 이 내용에는 pauli analysis, nekomata state와의 연관성 등을 다룹니다.

마지막으로는 앞으로의 연구 방향에 대한 아이디어를 간략히 소개하겠습니다.

# PARITY란?

입력 $x=(x_1,\dots,x_n)\in\{0,1\}^n$에 대해

$$
\mathrm{PARITY}(x)=x_1\oplus x_2\oplus\cdots\oplus x_n
$$

를 계산하는 문제가 PARITY 문제입니다.

# QAC$^0$란?

이전 글에서 소개했으니, 간략하게만 언급하겠습니다. QAC$^0$란 아래 조건을 만족하는 회로 클래스입니다.

1. 깊이(depth)가 상수
2. 크기(size)가 다항식
3. unbounded fan-in 게이트를 허용 (보통 임의 크기 CZ를 허용)

대응되는 고전 회로 클래스와 함께 봅시다. n은 입력 크기입니다.

| Class | Depth | Gate set | Size |
|---|---|---|---|
| NC$^0$ | O(1) | AND/OR/NOT (bounded fan-in) | Poly(n) |
| AC$^0$ | O(1) | AND/OR/NOT (unbounded fan-in) | Poly(n) |
| QNC$^0$ | O(1) | 2-local unitaries | Poly(n) |
| QAC$^0$ | O(1) | 1-qubit + CZ (unbounded fan-in) | Poly(n) |
| QAC$^0_f$ | O(1) | 1-qubit + CZ + FANOUT | Poly(n) |

<p align="center"><img src="/assets/images/red1108/qac0_qac0.jpeg" width="65%"></p>
<center><b>그림 1.</b> QAC$^0$ 회로의 예시</center><br/>

아래 내용들은 쉽게 유추 가능합니다.

$$
\text{NC}^0 \subsetneq \text{AC}^0,
\qquad
\text{QNC}^0 \subsetneq \text{QAC}^0 \subseteq \text{QAC}^0_f
$$

여기서 가장 먼저 떠오르는 두 가지 질문이 있습니다.

1. QAC$^0$ = QAC$^0_f$ 일 수 있나?
2. AC$^0$ $\subsetneq$ QAC$^0$ 인가?

사실 두번째 질문도 굉장히 흥미롭습니다. 놀랍게도, 결정 문제에 대해서 QAC$^0$가 AC$^0$보다 더 강력하다는 결과는 아직 없습니다. 그래서 QAC$^0$가 AC$^0$보다 더 강력한지 여부는 여전히 열린 문제입니다.

마찬가지로, QAC$^0$와 QAC$^0_f$의 관계도 아직 완전히 밝혀지지 않았습니다. FANOUT을 허용한 QAC$^0_f$ 는 굉장히 강력하단 것이 잘 알려져 있지만[3], QAC$^0$에서 이런 강력함을 보이는지는 아직도 open problem입니다.

QAC$^0_f$가 얼마나 강력한지 일부 예시를 보여드리겠습니다. QAC$^0_f$는 아래 문제들을 풀 수 있습니다.

1. *PARITY:* $U|x\rangle|y\rangle = |x\rangle|y\oplus \mathrm{PARITY}(x)\rangle$
2. *FANOUT:* $U|x\rangle|y_1,\dots,y_k\rangle = |x\rangle|y_1\oplus x,\dots,y_k\oplus x\rangle$
3. *MOD$_p$:* $U|x\rangle|y\rangle = |x\rangle|(y\oplus x) \bmod p\rangle$
4. *global phase kickback:* $U|x\rangle = e^{i\phi(x)}|x\rangle$
5. *CAT(GHZ) state preparation:* $U|0^n\rangle = \frac{|0^n\rangle + |1^n\rangle}{\sqrt2}$
6. *Quantum Fourier Transform (Approx):* $U|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y\in\{0,1\}^n} e^{2\pi i x\cdot y/2^n}|y\rangle$
7. *Addition*: $U|x\rangle|y\rangle = |x\rangle|y+x \bmod 2^n\rangle$
8. *Comparator:* $U|x\rangle|y\rangle = |x\rangle|y\oplus \mathbf 1_{x\ge y}\rangle$
9. *Sorting:* $U|x_1,\dots,x_n\rangle = |x_{\pi(1)},\dots,x_{\pi(n)}\rangle$ (where $\pi$ is the permutation that sorts the input)


놀라운 점은, QAC$^0_f$는 단지 상수 깊이 라는 것입니다. 이런 연산들을 상수 깊이로 해결한다는게 고전 회로에서는 상상도 못할 일입니다.

> 특히, Sorting이 상수깊이로 구현 가능하다는 점이 흥미로워 보이는데, 양자 중첩을 활용해서 모든 $n(n-1)/2$ 쌍에 대해 comparator 결과를 상수 깊이로 모두 계산하고, 본인보다 작은 수의 개수를 합하면 본인의 위치를 알 수 있기 때문입니다.

하지만 FANOUT gate가 그렇게 강력하다면, 쉽게 구현될 리가 없습니다... 우리가 실제 양자컴에서 다루는 게이트들은 single qubit unitary들과 2 qubit entanglement gate (CNOT 또는 CZ) 게이트이기 때문입니다. 이건 QNC$^0$에 해당하는 게이트셋입니다.

하지만 QNC$^0$는 light cone argument를 사용하면 PARITY 조차도 계산할 수 없다는게 너무 명확하게 보입니다.

### 왜 QNC$^0$ 는 PARITY를 계산할 수 없을까?

<p align="center"><img src="/assets/images/red1108/qac0_lightcone.jpg" width="92%"></p>
<center><b>그림 2.</b> bounded fan-in 상수깊이 회로에서의 lightcone 직관</center><br/>

이런 식의 문제에 흔히 쓰이는 방식이 light cone argument입니다. 각 게이트는 상수 width를 가지기 때문에, 출력 큐빗에 영향을 줄 수 있는 이전 결과값들을 역추적해 가다 보면, 최대 $w^d$ 개의 입력 비트에만 의존할 수 있다는 것을 알 수 있습니다. (여기서 $w$는 게이트의 fan-in, $d$는 회로의 깊이입니다.) 하지만 PARITY는 모든 입력 비트에 의존하기 때문에, 이런 상수 깊이 회로로는 PARITY를 계산할 수 없습니다. 최소한 $d \ge \log n$ 정도는 필요합니다. 그리고 이게 하한이라는건 XOR gate를 단순히 토너먼트 대진표 짜듯이 쌓기만 하면 쉽게 보일 수 있습니다. 즉 QNC$^1$는 PARITY를 계산할 수 있지만, QNC$^0$는 불가능합니다.

### QAC$^0$가 PARITY를 계산할 수 있는가?

QNC$^0$는 실제 양자컴에서 좀 더 다루기 쉬운 게이트셋이지만, 너무 약해서 PARITY조차도 계산할 수 없습니다. 반면에 QAC$^0_f$는 FANOUT이 있기 때문에 PARITY를 계산할 수 있습니다. 그렇다면 그 중간으로 가 봅시다. QAC$^0$는 PARITY를 계산할 수 있을까요?

QNC$^0$과의 차이점은 CZ 게이트 너비에 제한이 없다는 점입니다. 하지만 FANOUT이 없어서 QAC$^0$ 만큼 강력하진 않습니다 (아직은 증명된 바는 없지만). 그래서 QAC$^0$가 PARITY를 계산할 수 있는지 여부가 그 다음 질문이 됩니다.

이 질문이 중요한 이유는, 이게 단순히 한 문제의 정답 여부가 아니라 **회로가 전역 정보를 처리할 수 있는지**를 물어보는 질문이기 때문입니다. PARITY 는 모든 입력 비트의 정보를 고르게 고려해야만 풀 수 있는 문제이고, QAC$^0$가 PARITY를 계산할 수 있다면, QAC$^0$가 전역 정보를 처리할 수 있다는 것을 의미합니다. 그리고 상수 깊이 양자 회로가 상수 깊이 고전 회로보다 강력하다는 것을 보여주는 중요한 증거가 될 것입니다.

### Parity와 동치인 문제들

흥미롭게도, PARITY 문제는 다른 여러 문제들과 동치입니다. QAC$^0$에서 PARITY를 계산할 수 있는지 여부는 다음과 같은 문제들 중 하나라도 QAC$^0$에서 구현 가능하냐에 달려 있습니다.

- FANOUT
- MOD$_p$
- global phase kickback
- CAT(GHZ) state preparation

즉, PARITY를 QAC$^0$에서 계산할 수 있다면, FANOUT도 QAC$^0$에서 구현할 수 있게 되고, QAC$^0$ = QAC$^0_f$가 됩니다. 다시 말해, PARITY만 풀 수 있다는걸 보여도, 위에서 언급한 QFT, 덧셈뺄셈, 정렬, modular 등등등 다양한 문제들을 QAC$^0$에서 풀 수 있게 되는 것입니다.

이제 다른 문제들을 볼 필요 없이 PARITY 만 집중해서 다룰 이유가 충분해 졌습니다. 한번,고전 회로에서 AC$^0$가 PARITY를 못 푼다는걸 어떻게 보였는지 살펴보고, 양자 회로에서는 지금까지 어떤 하한과 상한이 나왔는지 살펴보도록 하죠.

### PARITY $\notin$ AC$^0$ [2]

왜 그럴까요? 핵심 증명 스텝을 요약해서 보면 다음과 같습니다.

1. AC$^0$ 회로를 AND/OR가 번갈아 나오는 형태로 정리합니다. AND-AND 층이 있다면 AND 층 하나로 압축되고, OR-OR 층이 있다면 OR 층 하나로 압축됩니다. 이렇게 하면 AND/OR가 번갈아 나오는 형태로 회로를 정리할 수 있습니다.

2. random restriction + switching lemma를 써서 바닥층 구조를 바꿉니다. random restriction을 걸어서 입력 비트의 일부를 고정시키면, 회로의 구조가 단순해지는 효과가 있습니다. 이렇게 입력 비트 일부를 고정하면 AND-OR 층을 OR-AND 층으로 바꿀 수 있고, 이제 뒤에 있던 AND층과 만나서 층 하나를 줄일 수 있습니다. 풀어 쓰면 AND-OR-AND-> OR-AND-AND -> OR-AND로 바뀌는 식입니다.

3. 2의 과정을 반복해서 회로의 깊이를 계속 깎아 나갑니다. 회로의 입력 일부를 계속 고정해 나가면서, 깊이 2까지 줄일 수 있습니다.

4. 결국 회로가 얕은 decision tree로 붕괴합니다. 깊이 2까지 줄어든 회로는 결국 입력 비트 몇 개만 보고 결정을 내리는 형태가 됩니다. 하지만 PARITY는 모든 입력 비트를 봐야 하는 문제이기 때문에, 이런 형태로는 PARITY를 계산할 수 없습니다.

이렇게 요약해서 적으니 증명이 간단해 보이지만, 매 단계에서 입력 일부를 고정하면서 깊이를 줄이는데, 이미 입력을 다 고정해 버려서 더 이상 고정할 입력이 없으면 어떻게 할까요? 논문에서는 고정 비율을 잘 설정해서 이런 일이 발생하지 않도록 합니다.

AC$^0$가 PARITY를 계산할 수 없다는 것은 86년도에 증명되었지만, QAC$^0$가 PARITY를 계산할 수 없다는 것은 아직도 증명되지 않았습니다.

# worst-case vs average-case

논문을 보면 이 둘이 계속 섞여 나오는데, 처음 보면 헷갈리기 쉽습니다. 그래서 먼저 분리해 놓고 갑시다.

목표 함수를 $f:\{0,1\}^n\to\{0,1\}$, 회로 출력을 $q(x)$라고 하겠습니다.

### Worst-case hardness

어떤 입력 하나라도 취약하면 실패로 보는 기준입니다.

$$
\min_{x\in\{0,1\}^n}\Pr[q(x)=f(x)]\le\frac12+\epsilon
$$

### Average-case hardness

랜덤 입력에서 평균 성능을 보는 기준입니다.

$$
\mathbb E_{x\sim\{0,1\}^n}[\Pr[q(x)=f(x)]]\le\frac12+\epsilon'
$$

비유하자면 Yes/No 를 찍는 이지선다 문제가 100개 있다고 합시다.

> Worst case hardness에서는 "이 사람이 만점은 받지 못할거야, 왜냐면 이 문제는 찍어야만 하거든" 이렇게 접근하는 방식입니다.

> Average case hardness에서는 "이 사람은 50점밖에 못 받을거야, 왜냐면 이 사람이 사실 문제를 잘 푸는것처럼 보여도 사실 찍고있는거랑 다를 게 없거든" 이렇게 접근하는 방식입니다.

직관적으로, average-case hardness를 보이는게 더 강력한 결과라는게 와닿으시나요? 이제 현재까지 알려진 하한/상한을 살펴봅시다.

# 현재까지 알려진 하한 결과

한 번에 흐름을 보려고 표로 정리하면 아래와 같습니다.

| Ancilla/조건 | 설정 | 연구 |
|---|---|---|
| $a \ge n2^{-d}-1$ | exact | [1] |
| depth 2 불가능 (poly ancilla) | exact/average | [4] |
| depth 2 불가능 (ancilla 무관), depth 3 불가능 | exact/average, exact | [5] |
| $d=7$에서 $a\le \exp(O(n\log n/\epsilon))$ 범위 하한 | worst case | [6] |
| $a\ge n^{\Omega(1/d)}$ | average case | [7] |
| $a\ge n^{1+3^{-d}}$ | average/worst case | [8] |
| $a\ge n^{1+2^{-d}}$ | average/worst case | [10] |

이 결과들을 보면 아직 QAC$^0$가 PARITY를 계산할 수 없는지 여부는 확정되지 않았지만, ancilla의 하한이 점점 강화되고 있다는 것을 알 수 있습니다.

# Pauli Analysis

앞선 글에서 Fourier spectrum 분석이 고전 회로에서 중요한 도구라고 했는데, 양자 회로에서는 Fourier spectrum 분석이 직접적으로 적용되기 어렵습니다. 왜냐하면 양자 회로에서는 단순히 Boolean 함수가 아니라, 양자 채널의 행동을 분석해야 하기 때문입니다. 그래서 양자 회로에서는 Fourier spectrum 대신에 **Pauli spectrum**을 분석하는 방식이 주로 사용됩니다.

### Hilbert-Schmidt 공간과 Pauli 분해

$N=2^n$일 때, $M_N$ 위 inner product를

$$
\langle A,B\rangle := \mathrm{Tr}(A^\dagger B)
$$

로 둡니다. 그러면 Frobenius norm은

$$
\|A\|_F^2 = \langle A,A\rangle = \sum_{i,j}|A_{ij}|^2
$$

가 됩니다.

Pauli basis $\mathcal P_n$에 대해 임의의 operator $A$는

$$
A = \sum_{P\in\mathcal P_n}\widehat A(P)P,
\qquad
\widehat A(P)=\frac{1}{N}\langle P,A\rangle
$$

로 분해됩니다.

여기서

$$
\deg(A)=\max\{|P|:\widehat A(P)\neq0\}
$$

를 Pauli degree,

$$
\widetilde{\deg}_\epsilon(A)=\min\{\deg(B):\|A-B\|\le\epsilon\}
$$

를 근사 차수로 둡니다.

왜 근사 차수가 필요할까요? 어차피 양자 회로에서 출력은 연속적인 값이기 때문에 정확히 0/1이 될 필요가 없습니다. 보통 1/3을 기준으로 사용하는데, 1/3보다 작다면 0으로 해석하고, 2/3보다 크다면 1로 해석하면 되기 때문입니다. 그래서 근사를 허용해 줄 테니, 이때 문제가 얼마나 어려운지 보는게 핵심입니다.

자주 쓰는 성질은 아래 세 가지입니다.

### Lemma (Pauli degree algebra)

$$\deg(A+B)\le\max(\deg(A),\deg(B))$$
$$\deg(AB)\le\deg(A)+\deg(B)$$
$$\deg(A\otimes B)=\deg(A)+\deg(B)$$

이 식은 근사 차수에서도 동일하게 성립합니다.

또한 Parseval 관계는 아래와 같습니다.

$$
\frac{1}{N}\|A\|_F^2 = \sum_{P\in\mathcal P_n}|\widehat A(P)|^2
$$

$$
\frac{1}{N}\langle A,B\rangle = \sum_{P\in\mathcal P_n}\widehat A(P)^*\widehat B(P)
$$

### 왜 unitary 전체 degree만 보면 안 될까?

모든 큐빗에 X게이트를 적용하는 $U=X^{\otimes n}$을 생각해 봅시다. 굉장히 간단한 회로이고, depth 1인데 pauli degree는 n입니다. 이러면 pauli analisys 가 아무 도움이 안 되는 것처럼 보입니다.

그럼 depth와 degree가 무관한 걸까요?

사실 지금까지 고전 회로와 양자 회로를 불공정하게 비교하고 있었습니다. 어차피 관심있는거는 1 bit/qubit 출력인데, 이렇게 전체 unitary의 degree를 보면 출력 n 큐빗을 다 보는 것이기 때문에 의미가 없습니다. 우리가 진짜 궁금한 건 **최종 출력 1큐빗이 얼마나 복잡한 정보를 담을 수 있는가**입니다.

그래서 n큐빗 출력 중에 큐빗 하나에만 집중합니다. 나머지는 trace-out 해버립니다.

<p align="center"><img src="/assets/images/red1108/qac0_common_setting.png" width="92%"></p>
<center><b>그림 3.</b> 입력 n큐빗, ancilla a큐빗, 출력 1큐빗 표준 세팅</center><br/>

입력 $n$큐빗 상태 $\rho$에 대해

$$
\mathcal E_{U,\psi}(\rho) = \mathrm{Tr}_{\mathrm{out}^c}\big(U(\rho\otimes\psi)U^\dagger\big)
$$

로 채널을 정의합니다.

이 채널의 Choi representation

$$ \Phi_\mathcal E = (I^{\otimes n}\otimes \mathcal E)(|\mathrm{EPR}_n\rangle\langle\mathrm{EPR}_n|)
$$

을 Pauli basis로 분해합니다.

$$
\Phi_\mathcal E = \sum_{P\in\mathcal P_m}\widehat\Phi_\mathcal E(P)P
$$

그리고 level-$k$ Pauli weight을

$$\mathbf W^{=k}[\Phi_\mathcal E]=\sum_{|P|=k}|\widehat\Phi_\mathcal E(P)|^2$$

로 정의합니다.

### Proposition (Fourier weight과 Pauli weight의 유사성)

Boolean 함수 $f$가 유도하는 classical channel $\mathcal E_f$에 대해 아래 식이 성립합니다.

$$\Phi_{\mathcal E_f}=\frac12 I^{\otimes(n+1)}+\frac12\sum_{S\subseteq[n]}\widehat f(S)\,Z^S\otimes Z$$

즉, 고전 복잡도에서 쓰던 Fourier 계수가 양자 채널의 Pauli 계수로 거의 그대로 넘어옵니다.

### Proposition (Pauli spectrum controls success probability)

$$
\Pr_x[\mathcal E(x)=f(x)]
\le
\frac12
+\frac12\sqrt{\mathbf W^{\le k}[f]}
+\frac12\sqrt{\mathbf W^{>k+1}[\Phi_\mathcal E]}.
$$

이 식은 $f$의 low-degree Fourier weight과 $\mathcal E$의 high-degree Pauli weight이 성공 확률을 어떻게 제어하는지를 보여줍니다. 직관적으로 생각하면 고차 pauli weight를 충분히 만들지 못한다면 고차 함수를 계산하는데 어려움이 있을 것입니다. 그 직관을 그대로 식으로 옮긴 것이 위의 부등식입니다.

PARITY는 어떤 함수였죠? 고차수에 질량이 거의 전부 몰린 함수입니다.

$$
\mathbf W^{\le n-1}[\mathrm{PARITY}]=0, \mathbf W^{=n}[\mathrm{PARITY}]=1.
$$

그래서 $k=n-1$을 대입하면

$$
\Pr_x[\mathcal E(x)=\mathrm{PARITY}(x)]
\le
\frac12+\frac12\sqrt{\mathbf W^{>n}[\Phi_\mathcal E]}.
$$

이 말은 곧, PARITY를 잘 맞히려면 채널의 고차수 weight가 충분히 커야 한다는 뜻입니다. 그럼, QAC$^0$ 회로가 고차수 weight를 충분히 만들 수 있을까요?

### Theorem (Low-degree pauli spectrum concentration)

depth-$d$ QAC$^0$ + ancilla $a$가 구현하는 $n\to1$ 채널 $\mathcal E$에 대해 아래가 성립합니다.

$$
\mathbf W^{>k}[\Phi_\mathcal E]\le 2^{-\Omega(k^{1/d}-a)}.
$$

즉, QAC$^0$로 만드는 출력 함수는 저차수에 성분이 몰려 있고, 고차수로 갈수록 성분이 지수적으로 감소한다는 것을 보여줍니다. 이걸 앞선 결과와 결합하면, 필요한 ancillae 수의 하한을 유도할 수 있습니다.

$$
\Pr_x[\mathcal E(x)=\mathrm{PARITY}(x)]
\le
\frac12+2^{-\Omega(n^{1/d}-a)}
$$

가 나오고, 결국

$$
a\ge \Omega\big(n^{1/d}-\log(1/\varepsilon)\big)
$$

가 필요합니다. 이렇게 lower bound를 구해냈습니다. 하지만 아직 sublinear~near-linear 레벨입니다. QAC$^0$가 PARITY를 계산할 수 없다고 확정하기 위해서는, 임의의 polynomial size ancillae를 허용해도 계산할 수 없음을 보여야 합니다. **하한을 더 끌어올려야 합니다.**

# Barely-superlinear ancilla 하한으로의 개선

[8]의 연구에서는 PARITY를 필요하기 위해 $$a\ge n^{1+3^{-d}}$$ 의 ancillae가 필요하다는 더 개선된 lower bound 증명했습니다.

핵심 관찰은 큰 CZ 게이트는 사실 그 영향이 미미하다는 것입니다. 그동안 이 CZ게이트가 임의 크기가 가능해서 여러 큐빗의 영향을 다 주어서 light cone argument를 못 쓰는게 골칫거리였는데, 실제로는 큰 CZ 게이트가 고차수 Pauli weight를 만들어내는데 그렇게 도움이 되지 않는다는 것을 보인 것입니다.



<p align="center"><img src="/assets/images/red1108/qac0_cz.png" width="30%"></p>
<center><b>그림 4.</b> large-CZ 근사 차수 아이디어</center><br/>

잘 보면 맨 오른쪽 아래 원소 한개만 -1이고, 나머지는 모두 항등행렬과 동일합니다. 이 성질을 잘 활용하면 아래 결과를 얻을 수 있습니다.

$$
\widetilde{\deg}_\varepsilon(|1\rangle\langle1|^{\otimes n})=\widetilde O(\sqrt n)
$$

이고,

$$
\mathrm{CZ}_n=I-2|1\rangle\langle1|^{\otimes n}
$$

이므로

$$
\widetilde{\deg}_\varepsilon(\mathrm{CZ}_n)=\widetilde O(\sqrt n)
$$

를 얻습니다.

이제 어떤 thershold $t$를 정해서, CZ 게이트를 $t$보다 작은 것과 큰 것으로 나눠 봅시다. 그러면 다음과 같은 trade-off가 생깁니다.

1. 크기가 $t$보다 작은 게이트는 light cone argument로 제어 가능
2. 크기가 $t$보다 큰 게이트는 항등행렬과 비슷하니 근사 차수로 제어 가능

최적 $t$를 잘 고르면 좋은 결과가 나올 것만 같아 보입니다...

### Lightcone vs Approximation balance

한 레이어 내부에 게이트들을 $\le t$ 게이트와 $>t$ 게이트로 분리하면

$$
\widetilde{\deg}_\varepsilon(U^\dagger A U)
\le
\underbrace{\ell t}_{\text{lightcone}}
+
\underbrace{2\frac{n+a}{\sqrt t}}_{\text{approximation}}
$$

형태가 나오고, 이 형태는 흔히 보이는 산술기하 평균 부등식 형태입니다. 직관적으로 $t=(n+a)^{2/3}\ell^{-2/3}$로 설정하는것이 두 항의 균형점을 만들어서 전체 bound를 가장 잘 줄이는 방법입니다. 따라서

$$
\widetilde{\deg}_\varepsilon(U^\dagger A U)
\le
3\cdot\widetilde{\deg}_\varepsilon(A)^{1/3}(n+a)^{2/3}
$$

를 얻습니다. 이제 이걸 depth $d$ 레이어에 걸쳐서 반복하면

$$ \widetilde{\deg}_\varepsilon(U^\dagger A U)=O\big(\ell^{3^{-d}}(n+a)^{1-3^{-d}}\big)
$$

그래서 출력확률 함수 $p(x)$에 대해

$$
\widetilde{\deg}_\varepsilon(p)
\le
\widetilde O\big((n+a)^{1-3^{-d}}\big)
$$

를 얻습니다. PARITY의 근사 차수는 n이고, 이를 위해서는 

$$a\ge\Omega(n^{1+3^{-d}})$$

를 얻습니다. 나중에 이 bound는 동일한 저자의 이후 연구 [10]에서 $a\ge n^{1+2^{-d}}$ 로 개선됩니다.

## 하한을 어디까지 높여야 할까?

이 부분이 굉장히 흥미롭습니다. 단순히 생각해 보면 PARITY $\notin$ QAC$^0$를 보여주려면, polynomial size ancillae를 허용해도 PARITY를 계산할 수 없다는 것을 보여야 합니다. 하지만 지금까지 나온 결과들은 $a \ge n^{1+3^{-d}}$ 수준입니다. 예를 들면, $O(n^{100})$ 의 ancillae를 허용해도 계산을 못한다는걸 보여야 하는데, 아직 lower bound는 너무나도 낮아 
보입니다.

**실제로는, $O(n^{1 + exp(-o(d))})$ 의 lower bound만 보여도 PARITY $\notin$ QAC$^0$를 얻을 수 있습니다.**

왜 그런지 한번 살펴 봅시다. n-bit parity푸는 회로가 먼저 깊이 d, ancillae는 $n^c$로 를 사용한다고 합시다. 이를 
$$
[d\mid n\mid n^c]
$$

로 표기합시다.

PARITY는 재귀적으로 작은 문제를 풀고, 그 결과를 조립해서 큰 문제를 해결할 수 있습니다.


<p align="center"><img src="/assets/images/red1108/qac0_build1.png" width="60%"></p>
<center><b>그림 5.</b> 블록 재사용으로 ancilla 지수를 줄이는 구성</center><br/>

위 식은

$$
[d\mid n\mid n^c] + [d\mid n\mid n^c] \Longrightarrow [d+1\mid 2n\mid O(2n^c)]
$$

을 나타내고 있습니다. 이제 좀더 효율적으로 블럭들을 조립해 봅시다. $[d\mid n\mid n^c]$ 회로를 각 층마다 n개 쌓고, 그 결과를 다시 조합하는걸 k 번 반복하면

$$
O(n^{k-1})\times[d\mid n\mid n^c]
\Longrightarrow
[kd\mid n^k\mid O(n^{c+k-1})]
$$

를 얻습니다. 이로부터 다음 정리가 나옵니다.

### Step 1: 차수 낮춘 단위 블록 만들기.

$[d\mid n\mid n^c]$가 PARITY$_n$을 계산하면,

$$
[O(cdk)\mid m\mid m^{1+2^{-k}}]
$$

꼴 회로로 PARITY$_m$을 계산할 수 있습니다. 점점 ancilla 지수가 줄어드는 것을 볼 수 있습니다.

즉, 다항 ancilla 가정에서 barely-superlinear ancilla 영역으로 환원 경로가 생깁니다.


### Step 2: 단위 블록 쌓아서 차수 더 낮추기.

이제 구체적으로 어떻게 ancilla의 지수를 줄여나가는지 살펴봅시다. 핵심 아이디어는 각 층(layer)마다 서로 다른 크기의 입력값을 처리하는 블록들을 계층적으로 배치하는 것입니다.

먼저, 기본 블록 $[D \mid n \mid n^2]$이 존재한다고 가정해 봅시다. 이 블록들을 다음과 같이 확장하여 쌓아 올립니다.

- Layer 1: $[D \mid n \mid n^2]$
- Layer 2: $[D \mid n^2 \mid (n^2)^2]$
- Layer 3: $[D \mid n^4 \mid (n^4)^2]$
- ...
- Layer k: $[D \mid n^{2^{k-1}} \mid (n^{2^{k-1}})^2]$1.


이러한 구조에서 최종적인 전체 입력 크기 $m$은 각 레이어 입력 크기의 곱으로 나타납니다\dots

$$m = \prod_{i=1}^{k} n^{2^{i-1}} = n^{\sum_{i=0}^{k-1} 2^i} = n^{2^k - 1}$$

이때 각 레이어에서 사용되는 ancilla의 총합은 지배적인 항을 기준으로 $O(n^{2^k})$ 수준으로 억제됩니다.

우리가 얻은 결과인 $[kD \mid m \mid m^{1+\epsilon}]$에서 $m$과 ancilla의 관계를 살펴봅시다.

$$n^{2^k} = (n^{2^k-1})^{\frac{2^k}{2^k-1}} = m^{1 + \frac{1}{2^k-1}} \approx m^{1+2^{-k}}$$

결과적으로, 임의의 $k$에 대하여 PARITY$_m$을 $[kD \mid m \mid m^{1+2^{-k}}]$ 수준의 회로로 계산할 수 있게 됩니다.

### Conclusion

즉, $[d\mid n\mid n^c]$가 PARITY$_n$을 계산할 수 있다면, 우리는 이 블럭을 조합해서 ancillae 의 차수를 낮출 수 있습니다. 최종적으로, PARITY$_m$을 $[O(cdk)\mid m\mid m^{1+2^{-k}}]$ 수준의 회로로 계산할 수 있게 됩니다.

**따라서, PARITY를 계산하기 위해 $a \ge n^{1+exp(-o(d))}$ 의 ancillae가 필요하다는 것을 보여주는 것만으로도, PARITY $\notin$ QAC$^0$를 얻을 수 있습니다.**

즉, 현재의 lower bound $a \ge n^{1+3^{-d}}$ 도 충분히 고지에 접근한 셈입니다. 만약 누군가가 $a \ge n^{1+exp(-o(d))}$ 수준의 lower bound를 증명한다면, 그 즉시 PARITY $\notin$ QAC$^0$라는 중요한 결과를 얻을 수 있습니다.

# Summary

- QAC$^0$가 AC$^0$보다 더 강력한지 여부는 아직도 열린 문제입니다.
- PARITY 문제는 QAC$^0$가 전역 정보를 처리할 수 있는지 여부를 가르는 핵심 문제입니다.
- Pauli spectrum 분석이 QAC$^0$ 회로의 출력 채널을 분석하는 중요한 도구입니다.
- 현재까지 나온 하한 결과들은 ancillae의 필요성을 점점 더 강화하고 있지만, 아직도 polynomial size ancillae를 허용해도 계산할 수 없다는 것을 보여주지는 못하고 있습니다.
- 하지만, $a \ge n^{1+exp(-o(d))}$ 수준의 lower bound를 증명하는 것만으로도 PARITY $\notin$ QAC$^0$를 얻을 수 있기 때문에, 현재의 lower bound도 꽤 목표에 접근한 셈입니다.

# 앞으로의 방향

PARITY $\notin$ QAC$^0$ 문제는, 처음 제안된 이후로 26년동안 Open Problem으로 남아 있습니다. 하지만 최근 몇 년간 이 문제에 대한 관심이 높아지면서, 여러 연구자들이 다양한 접근법을 시도하고 있습니다. 제 개인적인 생각으로는 조만간 증명될 것 같습니다. 앞으로의 연구 방향으로는 다음과 같은 것들이 있습니다.

1. ancilla 하한 지수를 조금 더 올려서 PARITY notin QAC$^0$를 직접 겨냥
2. unitary $k$-design 관점으로 우회 접근 [13]
3. path-recording 관점(전역 기록을 상수 깊이에 지울 수 있는가)
4. AC$^0$ restriction 논법의 양자 버전(clean-up lemma) 확장 [5]

# References

[1] Fang, Maosen, et al. "Quantum lower bounds for fanout." *arXiv:quant-ph/0312208* (2003).

[2] Hastad, Johan. "Almost optimal lower bounds for small depth circuits." *STOC* (1986).

[3] Høyer, Peter, and Robert Špalek. "Quantum circuits with unbounded fan-out." *STACS* (2003).

[4] Paduavs, Daniel, et al. "Depth-2 QAC circuits cannot simulate quantum parity." *arXiv:2005.12169* (2020).

[5] Joshi, Malvika Raj, et al. "Improved Lower Bounds for QAC0." *arXiv:2512.14643* (2025).

[6] Rosenthal, Gregory. "Bounds on the QAC$^0$ Complexity of Approximating Parity." *arXiv:2008.07470* (2020).

[7] Nadimpalli, Shivam, et al. "On the Pauli spectrum of QAC0." *STOC* (2024).

[8] Anshu, Anurag, et al. "On the computational power of QAC0 with barely superlinear ancillae." *STOC* (2025).

[9] Anshu, Anurag, and Tony Metger. "Concentration bounds for quantum states and limitations on the QAOA from polynomial approximations." *Quantum* 7:999 (2023).

[10] Dong, Yangjing, et al. "Linear-Size QAC0 Channels: Learning, Testing and Hardness." *arXiv:2510.00593* (2025).

[11] Mele, Antonio Anna. "Introduction to Haar measure tools in quantum information: A beginner's tutorial." *Quantum* 8:1340 (2024).

[12] Schuster, Thomas, et al. "Random unitaries in extremely low depth." *Science* 389, 6755 (2025): 92-96.

[13] Foxman, Ben, et al. "Random unitaries in constant (quantum) time." *arXiv:2508.11487* (2025).
