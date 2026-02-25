---
layout: post
title: "Counting Trees with Fixed External Legs and Quantum Field Theory"
date: 2026-02-25
author: defwin
tags: [combinatorics, quantum field theory, generating function]
---

## 0. Intro
이번 글에서는 다음 문제를 풉니다.

- 허용된 정점 차수 집합(예: 3차, 4차)이 주어졌을 때, 외부 다리가 $n$개인 트리($n$-point tree)의 개수를 계산합니다.

처음 보면 이 문제는 꽤 난감해 보입니다. 이유는 단순합니다.

1. 어떤 차수 정점을 몇 개 쓸지부터 경우가 갈립니다.
2. 같은 정점 개수를 써도 외부 다리 배치와 내부 연결 방식이 많습니다.
3. 같은 모양을 중복으로 세지 않도록 정리해야 합니다.

하지만 생성함수 방법을 쓰면 이 문제가 한 줄짜리 함수방정식으로 정리되어 쉽게 해결할 수 있고, 이 과정에서 역함수를 테일러전개하는 과정이 필요해, 라그랑주 역변환(Lagrange Inversion)을 해야 합니다. 

그리고 사실 이 문제는 양자장론에서 상호작용의 종류가 정해져 있을 때 n-point function을 계산하기 위한 다이어그램의 개수를 세는 것과 같습니다. 신기하게도 양자장론 관점에서도 같은 생성함수를 얻을 수 있는데요, 이에 대해서도 설명하겠습니다.

## 1. Generating Function Method and Lagrange Inversion
### 1.1 Generating Function for $n$-point Trees
핵심은 "루트가 있는 트리"를 세는 것입니다. 한 개의 특별한 외부 다리(소스 다리)를 루트로 고정하면, 나머지 구조는 재귀적으로 붙습니다.

루트 기준으로 가능한 첫 단계는 두 가지뿐입니다.

1. 루트가 아무 정점도 만나지 않고 바로 끝나는 경우: 기여가 $J$.
2. 루트가 어떤 $d$차 정점을 만나는 경우: 그 정점에는 루트로 이미 한 다리가 들어왔으므로, 남은 $d-1$개 다리에 다시 같은 종류의 서브트리가 붙습니다.

루트 트리 생성함수를 $\phi(J)$라고 두면, 두 번째 경우는 $\phi^{d-1}$ 형태가 됩니다. 그리고 $d-1$개 가지는 서로 구분되지 않는 자리이므로 중복을 나누는 $(d-1)!$가 분모에 생깁니다.

그래서 허용 차수 집합이 $D$일 때

$$
\phi(J)=J+\sum_{d\in D}\frac{\phi(J)^{d-1}}{(d-1)!}
$$

를 얻습니다.

예를 들어 $D=\{3,4\}$이면

$$
\phi(J)=J+\frac{\phi^2}{2!}+\frac{\phi^3}{3!}
$$

입니다. 이 식은 "모든 트리를 다 그린다"는 정의를 재귀적으로 번역한 결과입니다.

### 1.2 계수와 다이어그램 개수의 대응
위 식을 풀어 $\phi(J)$를 테일러 전개로 구했다고 해봅시다.

$$
\phi(J)=J+\sum_{n\ge 2} c_n\frac{J^{n-1}}{(n-1)!}
$$

위와 같이 놓으면 $c_n$이 $n$-point tree 개수가 됩니다.

$n$-point tree의 개수를 $J^{n-1}$의 계수로 뽑아내는 이유는, 루트로 고정한 특별한 한 다리를 제외하면 실제 외부 다리가 $n-1$개 남기 때문입니다. 그리고 $(n-1)!$는 이 외부 다리들의 라벨링 순서를 정규화하는 입니다.

정리하면, 위 $\phi(J)$의 테일러 전개를 구해 그 계수를 보면 $n$-point tree의 개수를 알아낼 수 있습니다.

### 1.3 Applying Lagrange Inversion
위 문제를 조금 더 들여다 봅시다. 먼저 다음과 같이 $F$를 정의합시다.

$$
\phi=x+F(\phi),\qquad F(u)=\sum_{d\in D}\frac{u^{d-1}}{(d-1)!}
$$

이제 위 식을 $\Psi(x)=\phi(x)^{-1}$를 이용해 다시 표현하면 다음과 같이 됩니다.

$$
x=\Psi(\phi),\qquad \Psi(u)=u-F(u)
$$

따라서 저희가 해야 할 일은 "역함수의 계수 추출"이고, 이는 라그랑주 역변환을 통해 할 수 있음이 알려져 있습니다.

일반적인 라그랑주 역변환을 유도해봅시다. $y(x)$가

$$
x=\Psi(y),\qquad \Psi(0)=0,\ \Psi'(0)\neq 0
$$

를 만족하고, $G$가 해석적 함수이면

$$
[x^n]\,G(y(x))
=\frac1n[u^{n-1}]\,G'(u)\left(\frac{u}{\Psi(u)}\right)^n
$$

가 성립한다는 것이 라그랑주 역변환의 결과입니다. 여기서 $[x^n]f(x)$는 $f(x)$의 $x^n$ 항의 계수를 의미합니다.

짧게 유도하면 다음과 같습니다.

$$
[x^n]G(y(x))
=\frac{1}{2\pi i}\oint \frac{G(y(x))}{x^{n+1}}\,dx
=\frac{1}{2\pi i}\oint G(u)\frac{\Psi'(u)}{\Psi(u)^{n+1}}\,du.
$$

그리고

$$
\frac{d}{du}\!\left(\frac{G(u)}{\Psi(u)^n}\right)
=\frac{G'(u)}{\Psi(u)^n}
-n\,G(u)\frac{\Psi'(u)}{\Psi(u)^{n+1}}
$$

를 정리하고 폐곡선 적분에서 전체미분 항이 0임을 쓰면

$$
[x^n]G(y(x))
=\frac1n\frac{1}{2\pi i}\oint \frac{G'(u)}{\Psi(u)^n}\,du
=\frac1n[u^{n-1}]\,G'(u)\left(\frac{u}{\Psi(u)}\right)^n
$$

를 얻습니다.

이제 우리 문제로 돌아오면 $y(x)=\phi(x)$, $\Psi(u)=u-F(u)$이므로

$$
[x^n]\,G(\phi(x))
=\frac1n[u^{n-1}]\,G'(u)\left(\frac{u}{u-F(u)}\right)^n.
$$

여기서 $\phi$의 $x^n$ 계수를 얻고 싶은 것이기 때문에, $G(u)=u$로 두면 됩니다.따라서 최종적으로

$$
[x^n]\phi(x)
=\frac1n[u^{n-1}]\left(\frac{u}{u-F(u)}\right)^n
$$

를 얻습니다.

### 1.4 Example: $D=\{3,4\}$, $n=3,4$
위 방법을 이용하면 트리의 개수를 얻어낼 수 있음을 설명하였는데요, 실제로도 이 방법론이 정확한 답을 주고 있다는 것을 간략하게 예시를 통해 확인해봅시다. $D=\{3,4\}$로 고정하면, $\phi$는 다음과 같이 주어집니다.

$$
\phi = J+\frac{\phi^2}{2}+\frac{\phi^3}{6}
$$

$\phi$의 꼴을 다음과 같이 쓰고 계수를 직접 구해봅시다.

$$
\phi(J)=J+a_2J^2+a_3J^3+\mathcal O(J^4).
$$

그러면 다음과 같이 전개됩니다.

$$
\phi^2=J^2+2a_2J^3+\mathcal O(J^4),
\qquad
\phi^3=J^3+\mathcal O(J^4).
$$

원식에 대입하면 다음과 같이 됩니다.

$$
J+a_2J^2+a_3J^3
=J+\frac12\left(J^2+2a_2J^3\right)+\frac16J^3+\mathcal O(J^4).
$$

차수별로 비교하면 각 계수를 다음과 같이 구할 수 있습니다.

$$
a_2=\frac12,
\qquad
a_3=a_2+\frac16=\frac12+\frac16=\frac23.
$$

따라서

$$
\phi(J)=J+\frac{1}{2}J^2+\frac{2}{3}J^3+\cdots
=J+(1)\frac{J^2}{2!}+(4)\frac{J^3}{3!}+\cdots
$$

이고

$$
c_3=1,
\qquad
c_4=4.
$$

가 됩니다. 실제 트리의 개수를 세어보면 다음과 같습니다.

1. $n=3$: 3-point vertex 하나만 가능하므로 1개
2. $n=4$: 총 4개
<p align="center"><img src="/assets/images/defwin/4ptTrees.png" width="80%"></p>
<center><b>그림 1.</b> 4-point Trees</center><br/>


로 정확하게 구해내는 것을 확인할 수 있습니다.

## 2. Implementing the Strategy
실제 구현에서는 1.3의 결과를 이용하지 않고 함수 방정식을 바로 풀 수도 있겠습니다만, closed form을 구하기 위해 라그랑주 역변환을 해본 김에 이를 이용하여 문제를 풀어봅시다. 1.3에서의 결과를 다시 쓰면 다음과 같이 쓸 수 있습니다.

$$
c_n=(n-2)!\,[u^{n-2}]\,H(u)^{n-1},
$$

$$
A(u):=1-\sum_{d\in D}\frac{u^{d-2}}{(d-1)!},\qquad
H(u):=\frac{1}{A(u)}.
$$

즉 $A(u)$의 역원 $H(u)$를 차수 $n-2$까지 구한 뒤 $H(u)^{n-1}$에서 $u^{n-2}$ 계수를 추출하면 문제를 해결할 수 있게 됩니다. 여기서 역원을 구하는 방법이 생소할 수 있는데요, Newtonian Iteration을 이용하면 됨이 알려져 있습니다.

### 2.1 Newton Iteration for Series Inverse
형식적 멱급수에서 $A(0)\neq 0$이면 역원이 존재합니다.  
$B$가

$$
A(u)B(u)\equiv 1 \pmod{u^m}
$$

을 만족한다고 가정합니다. 오차를

$$
E(u):=1-A(u)B(u)
$$

라고 두면 $E(u)=O(u^m)$입니다. 이제

$$
B_{\text{new}}(u):=B(u)\bigl(2-A(u)B(u)\bigr)
$$

로 갱신하면,

$$
A B_{\text{new}}
=AB(2-AB)
=(1-E)(1+E)
=1-E^2.
$$

$E=O(u^m)$이므로 $E^2=O(u^{2m})$이고, 따라서

$$
A(u)B_{\text{new}}(u)\equiv 1 \pmod{u^{2m}}
$$

입니다. 즉 뉴턴 갱신 1회마다 정확한 차수가 $m\to 2m$으로 두 배가 됩니다. 초기값은 $B_0=1/A(0)$로 시작하면 됩니다.

### 2.2 Improvement to $O(n\log n)$
지금까지 살펴본 방법을 이용할 때 주의할 점은 $H$의 차수가 최대 $O(n)$이고, $H^{n-1}$은 차수가 최대 $O(n^2)$이지만 실제로는 $O(n)$까지의 계수만 알고 있으면 되므로 거듭제곱 중간에 $n$차로 계속해서 잘라줘야 한다는 점입니다. 

이렇게 할 경우 FFT를 한 번 하고 그 결과값을 계속 곱한 뒤에 역변환을 한 번만 하는 최적화를 할 수 없는데요, 중간에 차수를 날려버릴 수 없고 전체 차수를 모두 가지고 있어야 하므로 $O(n^2)$만큼의 계수를 들고 있어야 하기 때문입니다. 그래서 $H$를 거듭제곱할 때 한 번 곱할 때마다 계속해서 변환과 역변환을 해주어야 하는데요, 이 경우 총 시간복잡도가 $O(n\log^2 n)$이 됩니다.

하지만 여기서 거듭제곱을 직접 반복하지 않고 최적화를 하는 방법이 있습니다. 다음 관찰을 합시다.

$$
H^{n-1}=\exp\!\bigl((n-1)\log H\bigr)
$$

1. $\log H$는
   $$
   \log H=\int \frac{H'}{H}\,du
   $$
   로 계산합니다. 핵심 비용은 역원/곱셈이고 FFT와 위에서 살펴본 뉴턴 방법을 쓰면 $O(n\log n)$입니다.
2. $G:=(n-1)\log H$를 만든 뒤 $\exp(G)$를 계산합니다.  
   $\exp$ 역시 뉴턴 doubling으로 $O(n\log n)$에 계산할 수 있습니다. 구체적으로 $Y=\exp(G)$를

   $$
   \Phi(Y):=\log Y-G=0,\qquad Y(0)=1
   $$

   의 해로 보고 반복합니다.
   
   먼저 $Y$가 $x^m$까지 맞는 근사해라고 가정하면
   
   $$
   \log Y-G=O(x^m)
   $$
   
   입니다. 이때
   
   $$
   \Delta:=G-\log Y,\qquad
   Y_{\text{new}}:=Y(1+\Delta)\pmod{x^{2m}}
   $$
   
   로 갱신합니다.
   
   왜 정확도가 두 배가 되는지는 다음처럼 확인할 수 있습니다.
   
   $$
   \log Y_{\text{new}}
   =\log Y+\log(1+\Delta)
   =\log Y+\Delta+O(\Delta^2)
   =G+O(x^{2m}).
   $$
   
   여기서 $\Delta=O(x^m)$이므로 $\Delta^2=O(x^{2m})$입니다. 따라서 한 번 갱신하면 정확한 차수가 $m\to 2m$으로 증가합니다.
   
   각 단계에서 필요한 연산은 $\log Y$ 계산(미분, 역원, 곱셈, 적분)과 마지막 곱셈 한 번이고, 전부 FFT 기반 다항식 곱셈으로 처리할 수 있습니다. 단계 크기가 $1,2,4,\dots$로 증가하므로 전체 합은

   $$
   O\!\bigl(n\log n\bigr)
   $$

   이 됩니다.
4. 결과적으로 $H^{n-1}$ 전체 계산이 $O(n\log n)$으로 떨어집니다.

정리하면, 이진 거듭제곱에서 생기는 추가 $\log n$ 인자를 log/exp 변환으로 제거할 수 있습니다.


## 3. Relation to Quantum Field Theory
### 3.1 QFT in a brief
양자장론(Quantum Field Theory)의 핵심 내용 중 하나는 입자의 산란진폭(scattering amplitude)을 구하는 방법에 관한 내용입니다. 이 때 장을 다루므로, 고전역학에서처럼 입자를 점으로 생각하는 것이 아니라 장으로 생각하기 때문에 훨씬 복잡해집니다. 

또한 "양자"장론이라는 이름에 걸맞게 양자적인 효과를 고려하는 것이 핵심이 되는데요, 이를 모두 설명할 수는 없으므로 어떤 일을 하는지 간략하게 소개드리겠습니다. 양자이론이란 물체의 상태를 추상적인 벡터공간(보통 힐베르트 공간)의 원소로 생각하는 이론을 말하며, 물리량들은 이 상태에 가하는 선형 연산자로 생각합니다. 여기서 "양자장"이라는 것은 시공간에 펼쳐진 연산자라고 보시면 됩니다. 아무튼 이 양자장론을 이용해 산란진폭을 계산할 때는 다음 과정을 따릅니다.

1. 진공에서의 $n$개의 시간으로 정렬된 장의 기댓값($n$-point function)을 구하면 LSZ reduction이라는 것을 통해 산란 진폭을 계산할 수 있습니다.
2. 생성함수를 만들고, 미분을 통해 $n$-point function을 추출할 수 있습니다. 이 생성함수를 partition function이라고 합니다.

즉 "생성함수 + 미분 + 계수 추출"이라는 구조가 핵심이고, 우리가 앞에서 사용한 조합론적 틀과 매우 유사합니다.

### 3.2 Partition Function, $n$-point function, and Feynman Diagrams
QFT에서 기본 객체는 source $J$가 붙은 생성함수(=partition function)

$$
Z[J]=\int \mathcal D\phi\,\exp\Big(iS[\phi]+i\int J\phi\Big)
$$

입니다. 여기서 $J$는 새로운 물리량이라기보다, "미분으로 계수를 뽑기 위한 보조 입력"입니다. 여기서 $S$와 $\mathcal{D}\phi$가 무엇인지 궁금할 수 있는데요, $S$는 고전역학에서 작용이라고 불리는 물리량으로 계를 표현하는 스칼라 범함수라고 생각하면 되고, $\mathcal{D}\phi$는 모든 가능한 장의 경우를 다 더하라는 기호로 생각하면 됩니다. 양자 이론은 상태를 힐베르트 공간의 원소로 보는 이론이라고 말씀드렸는데요, 다른 관점에서는 고전적으로 가능한 모든 가능한 경우를 고려하는 것으로 생각할 수 있습니다. 이 것이 파인만 경로적분이고, 위 식은 파인만 경로적분의 양자장론 형태입니다.

$J$를 기준으로 함수미분하면 $n$-point 함수를 얻습니다.

$$
\frac{1}{Z[0]}
\left.\frac{\delta^n Z[J]}{i^n\delta J(x_1)\cdots\delta J(x_n)}\right|_{J=0} = \frac{1}{Z[0]}\int \mathcal{D}\phi \phi(x_1)\cdots \phi(x_n) e^{iS[\phi]} = \langle \phi(x_1) \cdots \phi(x_n) \rangle
$$

작용을 자유 부분과 상호작용 부분으로 분해해봅시다.

$$
S[\phi]=S_0[\phi]+S_{\text{int}}[\phi].
$$

생성함수는

$$
Z[J]=\exp\left(i\int \mathcal L_{\text{int}}\!\left(\frac{1}{i}\frac{\delta}{\delta J}\right)\right)Z_0[J]
$$

꼴로 쓸 수 있게 됩니다. 보통의 양자장론에서 $Z_0[J]$는 다음과 같은 함수적분형 가우시안으로 쓸 수 있음이 알려져 있습니다.

$$
Z_0[J]\propto \exp\left(-\frac{i}{2}\int d^dx\,d^dy\,J(x)\Delta_F(x-y)J(y)\right).
$$

여기서 $\Delta_F$는 Feynman propagator로 불리는 것으로, 고전적인 운동방정식의 그린 함수에 해다됩니다. 이제 계산을 조합문제로 바꿀 수 있습니다.

1. $\mathcal L_{\text{int}}$ 전개 차수는 정점 차수(3점, 4점 등)를 정합니다.
2. 각 정점은 $(\delta/\delta J)^d$ 같은 미분 묶음으로 나타납니다.
3. 가우시안에 미분을 작용시키면 미분쌍(pairing)마다 전파자 하나가 생깁니다(Wick 정리).
4. 바깥에서 취한 $n$번 미분은 외부선 $n$개를 의미합니다.

결국 "미분 pairing을 어떻게 만들 수 있는가"가 본질이고, 이 pairing 패턴을 그래프로 그려 놓은 표기법이 Feynman diagram입니다. Propagator는 선으로 표현되며, 상호작용은 점으로 표현되어 그래프를 이루게 됩니다.

### 3.4 Tree-Level Diagrams from Classical Equation of Motion
Tree-level에서는 같은 구조를 고전 방정식 반복해법으로도 볼 수 있습니다.

$$
\Box\phi = J + \sum_{d\ge 3}\frac{\lambda_d}{(d-1)!}\phi^{d-1}.
$$

여기에 그린함수 $G=\Box^{-1}$를 적용하면

$$
\phi = G\cdot J + \sum_{d\ge 3}\frac{\lambda_d}{(d-1)!}\,G\cdot\phi^{d-1}
$$

이고($\cdot$는 convolution), 이 식을 반복 대입하면 "하위 구조를 계속 붙이는 재귀"가 됩니다. 
이 재귀는 트리 생성 규칙과 같은 형태입니다.

이제 개수 세기만 남기기 위해 단순화를 하면

1. 전파자 효과($G$)를 1로 둡니다.
2. 결합계수 $\lambda_d$를 1로 둡니다.
3. 시공간 의존성을 제거합니다.

그러면

$$
\phi = J + \sum_{d\ge 3}\frac{\phi^{d-1}}{(d-1)!}
$$

를 얻고, 이는 앞에서 살펴본 생성함수 식과 완벽히 동일합니다!  
따라서 계수는 곧 다이어그램 개수이며, $c_3=1$, $c_4=4$ 결과도 같은 방식으로 재현됩니다.

## 4. Outro
이번 글에서는 외부 다리가 $n$개로 고정된 상황에서 정점 차수로 가능한 집합 $D$가 주어졌을 때 총 트리의 개수를 세는 방법에 대해 살펴보았습니다. 생성함수를 활용하여 방정식을 얻을 수 있었고, 라그랑주 역변환 및 뉴턴 방법, FFT 등을 활용해 $O(n \log n)$에 해결하는 방법도 알아보았습니다. 

추가적으로 이 문제가 양자장론에서 등장하고, 양자장론의 언어로도 같은 생성함수를 얻을 수 있음을 확인할 수 있었습니다. 이처럼 전혀 관련 없어보이는 이산수학과 양자장론이 Feynman diagram으로 연결된 부분이 있다는 점이 재미있는 것 같습니다.

## References
- Clifford Cheung, *TASI Lectures on Scattering Amplitudes*, Appendix A (Counting Feynman Diagrams), arXiv:1708.03872.  
  https://arxiv.org/abs/1708.03872
- [Lagrange inversion theorem (Wikipedia)](https://en.wikipedia.org/wiki/Lagrange_inversion_theorem)
- Mark Srednicki, *Quantum Field Theory*, Cambridge University Press.
