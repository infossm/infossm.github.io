---
layout: post
title: "Using linear-optical quantum computing model to prove #P-hardness of Permanent"
author: TAMREF
date: 2022-12-18
tags: [quantum,complexity-theory]
---

# Introduction

지난 달 작성한 글과, 최근 진행하는 project-hardness 스터디를 진행하면서 $\sharp P$-hardness에 대한 내용을 자주 다루었습니다. 하지만 $\sharp P$라는 complexity가 고안된 근본적인 이유와도 같은 permanent에 대해서는 언급하지 않았습니다. Shortest Even Cycle Problem을 다룬 문제에서 지나가듯 이야기했지만 길게 이야기하지 않았죠. 실제로 이 문제를 해결한 Valiant(1979)의 논문은 3000회가 넘는 인용수를 기록했으며, Permanent를 비롯한 수많은 복잡도 이론에 대한 공헌으로 Valiant은 튜링상까지 수상하게 됩니다.

이런 위대한 업적에도 불구하고 Valiant이 어떤 아이디어로 Permanent의 hardness를 증명했는지는 비교적 알려져 있지 않은데, 문제 해결을 위한 construction이 _notoriously opaque_ 하다고 다른 논문에 언급되어 있을 정도로 복잡하기 때문입니다. 11월에 작성한 글에서 느낄 수 있듯, 일반적으로 $\sharp P$-hardness에 대한 증명은 잘 알려진 문제에서 굉장히 정교한 reduction을 통해서 이루어집니다.

**Theorem (Valiant, 1979)** 주어진 3-SAT formula $F$에 대해, $F$를 satisfy하는 assignment의 개수를 $s(F)$라고 하자. 다음을 만족하는 $\{-1, 0, 1, 2, 3\}$-matrix $M$을 다항 시간 안에 계산할 수 있다. 여기서 $t(F)$는 $F$로부터 자연스럽게 계산할 수 있는 자연수 값이다.

$$
\mathrm{per}(M) = 4^{t(F)} \cdot s(F)
$$

여기서 Valiant은 다음과 같은 과정을 거쳐 Theorem을 유도합니다.
- $M$을 그래프의 인접행렬로 보고, $\mathrm{per}(M)$을 그래프의 cycle cover들의 weight sum으로 표현한다.
- $F$의 variable, clause들을 적절한 **그래프 $G_{F}$ 위의 gadget으로 표현한다.**
- $G_{F}$의 인접행렬에 대한 permanent가 정확히 $4^{t(F)}s(F)$가 되도록 **간선들의 weight를 미세 조정한다.**

큰 그림 자체는 이해할 수 있지만, 볼드체로 나타낸 두 부분이 _extremely opaque_ 하다는 평가를 받는 부분입니다. 똑같은 아이디어를 사용하되, 미세 조정 부분을 다소 간소화한 논문이 나오기도 했습니다.

**Theorem (Ben-dor & Halevi, 1995)** Clause가 $m$개인 3-SAT formula $F$에 대해, $\mathrm{per}(M) = 12^{m} \cdot s(F)$를 만족하는 행렬 $M$을 다항 시간 안에 계산할 수 있다.

이 논문은 비록 "simple proof"라는 제목을 가지고 있지만, 그럼에도 불구하고 gadget을 표현할 때 직접 손으로 깎은 $7 \times 7$ 행렬을 첨부하는 등 아직 이해할 수 있는 영역까지 내려오지는 않았습니다. Permanent가 가진 fundamental한 성질에 비하면 여전히 증명이 너무 복잡하지만, 적당한 그래프의 cycle cover weight sum과의 대응 외에는 hardness를 증명할 마땅한 방법이 고안되지도 않고 있었습니다.

그런데 오늘 리뷰할 논문인 Aaronson (2011)은, 생각지도 못했던 **양자 컴퓨팅**을 통해 permanent의 hardness를 아주 멋진 방법으로 해결했습니다.

# Linear-Optical Quantum Computing

기존의 양자 컴퓨팅을 앞으로 (standard) qubit model 이라고 부를 텐데, 오늘 소개할 양자 컴퓨터 모델은 약간 다릅니다. qubit model들은 번호를 붙일 수 있는 $n$개의 qubit들이 $2^{n}$개의 state들을 unitary transformation을 통해 탐사하는 기본적인 모델을 가지고 있지만, Linear optical quantum computing에서는 광자(photon)를 사용합니다. 광자는 전자 등의 다른 입자들과 다르게 "구분이 가지 않는" 특징을 가지고 있어서, $m$개의 mode (편광, 진동수 등) 들과 각 mode에 배정된 입자의 개수만으로 qubit state가 표시됩니다. 즉, 가능한 상태의 개수가 $2^{n}$이 아니라 $\binom{m + n - 1}{n}$개가 됩니다.

기존 $3$-qubit이 $\left\lvert 001 \right>$ 등으로 상태를 표시했다면, $4$개의 mode를 갖는 $3$-linear-optical qubit은 $\left\lvert 0, 2, 1, 0 \right>$와 같이 표현되는 식입니다. 이는 $2$번째 mode에 2개의 광자가, $3$번째 mode에 $1$개의 광자가 있음을 나타냅니다.

$m, n$의 값은 다양하게 고를 수 있지만, 이 글에서는 LOQC의 모든 이론이 필요하지는 않기 때문에 $m = 2n$으로 두고, 가장 기본이 되는 idle state를 $\left\lvert I \right> := \left\lvert 0, 1, 0, 1, \cdots, 0, 1 \right>$으로 정의하겠습니다. 짝수 번째 mode들을 "바닥 mode", 홀수 번째 mode들을 "들뜬 mode"들로 생각하면 언뜻 qubit model과 닮은 점이 있어서인지도 모르겠습니다.

이렇게 복잡한 computing model을 사용하는 이유가 바로 아래 줄에서 드러납니다.

**Theorem. (Ref: Aaronson 2011)**. $U$가 $m \times m$ unitary matrix라고 하자. 두 linear-optical qubit $P, Q$에 대해 $\varphi(U)$를 아래와 같이 정의한다.

$$
\left< P \right\rvert \varphi(U) \left\lvert Q \right> := \frac{\mathrm{Per}(U_{P, Q})}{\sqrt{p_{1}! \cdots p_{m}! q_{1}! \cdots q_{m}!}}
$$

이 때, $P = \left\lvert p_{1}, \cdots, p_{m} \right>$, $Q = \left\lvert q_{1}, \cdots, q_{m} \right>$이며,

$U_{P, Q}$의 $(i, j)$ element는 $P, Q$를 각각 multiset $P = 1^{p_1}2^{p_2} \cdots m^{p_m}, Q = 1^{q_1} \cdots m^{q_m}$으로 표현할 때 $P$의 $i$번째, $Q$의 $j$번째 element를 $p^{i}, q^{j}$라고 하면 $U_{p^i, q^j}$와 같다.

(이상 정의)

이 때, $\varphi(U)$는 unitary transform일 뿐만 아니라 $\varphi(UV) = \varphi(U)\varphi(V)$를 만족한다.

사실 $\varphi$는 "물리적으로 보면" 당연하게 정의된 함수라고 볼 수 있는데, $m \times m$ 행렬인 $U$의 $(i, j)$ entry $u_{ij}$를 "mode $j$에서 mode $i$로 전이하기 위한 확률 진폭 (확률은 $\left\lvert u_{ij} \right\rvert^{2}$)으로 생각해주면, $Q$의 상태에 있는 $n$개의 qubit들이 $P$로 전이할 확률이 위 식과 같이 나오기 때문입니다. 따라서 $\varphi$가 homomorphism이며, $\varphi(U)$가 unitary가 된다는 사실 또한 물리적인 justification이 가능합니다.

**위 사실은 물리적으로 보면 중요하지만, 우리에게는 Permanent가 식에 나왔다는 사실이 가장 중요합니다.**

이렇게 만든 permanent를 두고, 잠시 문제의 반대쪽으로 가서 어떤 hardness result를 끌어올지 논의해봅니다.

# SAT in quantum model

SAT보다 더 일반적으로, boolean function $C : \lbrace 0, 1 \rbrace^{n} \to \lbrace -1, 1 \rbrace$를 생각해 봅시다. 다항 시간 안에 evaluation은 되어야 하니 SAT와 크게 다르지는 않습니다. 여기서 $\Delta_{C} := \sum_{x} C(x)$를 계산하는 것은 $\sharp 3 SAT$보다 어려우니 $\sharp P$-hard일 것이 분명합니다. 그런데 여기서 $\Delta_{C}$ 에 대한 정보를 진폭에 담는 quantum circuit $Q$ (qubit model)를 디자인할 수 있습니다.

**Lemma.** 모든 $1$-qubit들과 아래의 유일한 2-qubit gate **CSIGN**으로 이루어진 gate들의 집합을 $G$라고 하면, $G$는 Toffoli gate를 만들어낼 수 있다. 따라서 $G$는 모든 classical computation을 흉내낼 수 있다.

*Proof.* [Wikipedia](https://en.wikipedia.org/wiki/Toffoli_gate)에 Toffoli gate를 CNOT gate와 1-qubit gate로 구현하는 회로도가 나와 있는데, 비슷한 느낌으로 CZ gate와 Hadamard gate, 다른 하나의 gate를 사용하면 됩니다. 

**Theorem.** 주어진 boolean function $C$에 대해서, $G$의 게이트만 사용하는 (특히  $\Gamma := \mathrm{poly}(n, \lvert C \rvert)$개의 **CSIGN** 게이트를 사용하는) quantum circuit $Q_{C}$가 존재하여, 아래 식을 만족한다.

$$
\left<0 \right\rvert^{\otimes n} Q_{C} \left\lvert 0 \right>^{\otimes n} = \frac{\Delta_{C}}{2^n}
$$

*Proof.* $G$의 게이트들로 Toffoli gate를 계산할 수 있으니 모든 classical computation을 할 수 있습니다. 즉, binary string $x$가 주어졌을 때 $\left\lvert x \right> \mapsto C(x)\left\lvert x \right>$를 계산하는 $2^{n} \times 2^{n}$ diagonal matrix $D_{C}$를 polynomial 개수의 CSIGN gate를 사용하여 만들어줄 수 있습니다. $\Delta_{C}$를 계산하는 것은

$$
\left< 0^{n} \right\rvert H^{\otimes n} D_{C} H^{\otimes n} \left\lvert 0^{n} \right> = \frac{1}{2^n} \sum_{x, y} C(x) \left< y\mid x\right> = \frac{\Delta_{C}}{2^n}
$$
으로 간단해집니다.

두 가지 접근을 통해 LOQC에서는 진폭에 Permanent를, standard qubit QC에서는 진폭에 SAT 문제를 담았습니다. 이제 두 진폭을 연결지어주는 **KLM theorem**을 통해 permanent의 $\sharp P$-hardness를 유도합니다.

## KLM theorem: implementing standard qubit model with Postselected LOQC

그냥 LOQC만으로 standard qubit operation을 모두 emulate할 수 있다면 좋겠지만, 아쉽게도 둘 사이에는 post-selection이라는 추가적인 연산이 필요합니다. post-selection이란, 특정 qubit이 참인 상태들만 남기고 나머지를 drop하는 다소 강력한 연산을 말합니다. 기존 양자 컴퓨팅에 post-selection이 주어진 경우 매우 어려운 문제들을 해결할 수 있다는 사실이 알려져 있습니다. (Aaronson, PostBQP=PP theorem)

KLM theorem의 결과를 소개하면 이렇습니다.

**Theorem. (KLM)** Standard $n$-qubit model $Q$에 대해, $2n$개의 mode를 갖는 post-selected LOQC circuit $L \in \mathbf{SU}(2n)$이 존재하여 $\left< I \mid \varphi(L) \mid I \right> = \frac{\left< 0^{n} \mid Q \mid 0^{n} \right>}{4^{\Gamma}}$ 를 만족한다. 이 때 $\Gamma$는 $Q$에서 사용하는 CSIGN 게이트의 개수.

KLM theorem을 포함하여 post-selected quantum circuit이 갖는 다양한 성질은 추후 포스팅으로 미루고, 지금까지 theorem이 의미하는 결과를 정리해보겠습니다.

- Classic boolean formula $C$가 주어졌을 때, polynomial size quantum gate $Q_{C}$를 **다항 시간**에 찾을 수 있다.
- KLM theorem에 의해, $Q_{C}$로부터 $2n$-mode post selected LOQC $L(Q_{C})$를 역시 **다항 시간**에 찾을 수 있다.
- 이 때 $\left< I \mid \varphi(L) \mid I \right> = \mathrm{Per}(L_{I, I})$이고, $L$은 **다항 시간 안에 계산할 수 있는 크기**의 행렬이다.
- 그런데 이 값은 $\frac{\Delta_{C}}{2^{n}4^{\Gamma}}$와 같다. 따라서 Permanent를 충분한 precision으로 계산하면, $\Delta_{C}$는 정수이므로 정확하게 계산할 수 있다.
- $\Delta_{C}$는 $\sharp P$-hard이므로, $L_{I, I}$의 permanent를 계산하는 것 또한 $\sharp P$-hard.

Classic에서 standard qubit model로 가는 과정이나, 생략된 KLM theorem들의 증명에서도 "A(classic computer, resp. standard quantum computer)의 연산을 B(std. quantum computer, resp. post-selected LOQC)의 연산으로 구현할 수 있다"만 주장할 뿐, 실제로 quantum operation을 사용하지는 않습니다. 즉, 우리가 가져온 두 quantum computing model은 reduction tool로 기능할 뿐 실제로 연산에 사용되지는 않는 것입니다. 그럼에도 불구하고 Permanent와 boolean function 사이의 멋드러진 reduction을 주어, permanent computation의 hardness를 증명하는 데 성공했습니다.

여담이지만 $\Delta_{C}$를 이용하면, Permanent sign을 계산하는 것 또한 $\sharp P$-hard하다는 사실을 증명할 수 있습니다. $C$에 몇 개의 input을 추가하여 $C(x) = -1$인 input이 $k$개 더 추가되도록 하면 새로운 circuit $C^{\prime}$에 대해 $\Delta_{C^{\prime}} = \Delta_{C} - k$가 됩니다. 이러한 $C^{\prime}$에 대해 permanent sign oracle을 반복적으로 적용하면 결국 이분 탐색을 통해 $\Delta_{C}$의 정확한 값을 계산할 수 있게 되기 때문입니다. 기존 Valiant의 reduction에서는 쉽게 떠오르지 않는 결과인 만큼 독자적인 결과를 창출했다고 볼 수 있겠습니다.

Quantum complexity theory는 "양자컴이 무엇을 계산할 수 있는가?", 즉 "Turing machine에서 주어진 complexity hierarchy의 어디까지 커버하는가?" 에 대한 질문 뿐만 아니라, 기존 complexity theory를 바라보는 새로운 시각을 제공한다는 면에서 여러 흥미로운 면을 가지고 있습니다. 다음 글에서는 이에 대한 보다 generic한 조사를 담아보도록 하겠습니다.

## References

- Ben-Dor, Amir, and Shai Halevi. "Zero-one permanent is not= P-complete, a simpler proof." [1993] The 2nd Israel Symposium on Theory and Computing Systems. IEEE, 1993.

- Valiant, Leslie G. "The complexity of computing the permanent." Theoretical computer science 8.2 (1979): 189-201.

- Aaronson, Scott. "A linear-optical proof that the permanent is# P-hard." Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 467.2136 (2011): 3393-3405.

- Aaronson, Scott, and Alex Arkhipov. "The computational complexity of linear optics." Proceedings of the forty-third annual ACM symposium on Theory of computing. 2011.