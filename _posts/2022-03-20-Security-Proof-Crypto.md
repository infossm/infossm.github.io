---
layout: post
title:  "암호학의 안전성 증명 테크닉"
date:   2022-03-20 11:00:00
author: blisstoner
tags: [cryptography]
---

# 1. Introduction

안녕하세요, 이번 글에서는 암호학에서 특정 구조에 대한 안전성을 증명하는 테크닉을 알아보겠습니다. 이 테크닉을 통해 실제 Feistel cipher의 안전성을 증명해볼 것입니다.

# 2. 암호학의 안전성

암호학의 안전성에 대해서는 지금으로부터 대략 1년 전에 이미 [포스팅](https://www.secmem.org/blog/2021/01/24/Indistinguishability-in-Cryptography/)을 한 적이 있습니다. Indistinguishability(구분 불가능성)이라는 용어를 처음 들어보신다면 해당 포스팅을 먼저 확인해보시는걸 추천드립니다. 이전 글에서도 언급했듯 Indistinguishability는 달성이 굉장히 어려운 성질입니다. 또한 만약 어떤 암호 체계가 랜덤한 메시지와 구분 불가능하다는건 공격자의 입장에서 그 어떤 방법으로 공격을 시도하더라도 아무런 의미있는 정보를 얻을 수 없다는 의미이기 때문에 굉장히 강력한 성질이기도 합니다.

작은 예시를 들어 아주 간단하게 Indistinguishability에 대해 다시 짚어보겠습니다. 우리에게 주사위가 주어졌다고 생각하겠습니다. 주사위는 X, Y 두 종류가 있습니다. 이 중 주사위 X는 1에서 6의 눈이 나올 확률이 1/6으로 전부 동일하고 주사위 Y는 1, 2, 3, 4의 눈이 나올 확률은 1/6인 반면 5가 나올 확률은 1/12이고 6이 나올 확률은 1/4입니다. 우리는 X인지 Y인지 알 수 없는 어떤 주사위를 받았고 이 주사위가 종류 X인지 종류 Y인지를 알아맞춰야 합니다. 주사위를 딱 한 번만 굴려볼 수 있다고 할 때 어떻게 하면 알 수 있을까요? 조금 더 수학적으로 깔끔하게 표현하기 위해 여기서 `Advantage`를 정의해봅시다. `Advantage`는 말 그대로 어드밴티지, 즉 내가 종류 X와 종류 Y를 구분할 확률을 나타냅니다. 이 말은 곧 주사위가 종류 X일 때 이를 X로 판단할 확률에서 종류 Y일 때 이를 X로 판단할 확률을 빼면 됩니다. 공격자가 주어진 주사위를 종류 X로 예상한다면 1을, Y로 예상한다면 0을 반환한다고 해봅시다. 공격자가 어떤 적절한 알고리즘 A를 사용한다고 할 때 Advantage를 식으로 표현하면 $Adv(A) = \|Pr[A(X) = 1] - PR[A(Y) = 1] \|$입니다. $A(X), A(Y)$ 각각은 말 그대로 알고리즘 A에 종류가 X(resp. Y)인 주사위를 넣을 때의 출력을 의미합니다.

공격자가 만약 조금 멍청해서 주사위를 굴린다음 그 결과는 보지 않고 무조건 주사위는 종류 X라고 판단을 한다고 해봅시다. 그러면 $Pr[A(X) = 1] = Pr[A(Y) = 1] = 1$이기 때문에 공격자의 Advantage는 0입니다. 혹은 주사위를 굴린 후 마찬가지로 그 결과를 보지 않고 주사위의 종류를 랜덤으로 찍어서 X 혹은 Y를 각각 절반의 확률로 반환한다고 해봅시다. 그러면 $Pr[A(X) = 1] = Pr[A(Y) = 1] = 1/2$여서 Advantage는 마찬가지로 0입니다. Advantage는 0에서 1 사이의 값을 가지고 0에 가까울수록 공격자가 X와 Y를 구분할 능력이 없음을, 1에 가까울수록 공격자가 X와 Y를 구분할 능력이 있음을 의미합니다.

다시 문제의 상황으로 돌아와서 공격자는 Advantage를 높이기 위해 어떤 전략을 사용해야 할까요? 만약 결과가 1, 2, 3, 4라면 달리 방법이 없고 찍을 수 밖에 없습니다. 결과가 1, 2, 3, 4일 때 이를 X로 판단하든 Y로 판단하는 상관이 없습니다(뒤에서 서술의 편의를 위해 1, 2, 3, 4가 나오면 그냥 X로 판단을 하겠습니다). 그런데 만약 결과가 5라면, 주사위 X에서 5가 나올 확률이 Y에서 5가 나올 확률보다 높기 때문에 주어진 주사위를 X라고 예상하는 것이 바람직해보입니다. 만약 6이 나왔다면 마찬가지 논리로 따졌을 때 주어진 주사위를 Y로 예상할 수 있습니다. 이런 전략을 사용한다면 앞에서 멍청한 공격자가 떠올렸던 무지성 알고리즘보다는 더 좋아보이고, 사실 이 방법 말고는 딱히 공격자가 할 수 있는 더 나은 방법이 있어보이지도 않습니다. 이 전략의 Advantage를 계산해봅시다. $px_i$를 종류 X에서 주사위 눈이 $i$일 확률이라고 할 때 $Pr[A(X) = 1] = px_1 \cdot 1 + px_2 \cdot 1 + px_3 \cdot 1 + px_4 \cdot 1 + px_5 \cdot 1 = 5/6$입니다. 그리고 $py_i$를 종류 Y에서 주사위 눈이 $i$일 확률이라고 할 때 $Pr[A(Y) = 1] = py_1 \cdot 1 + py_2 \cdot 1 + py_3 \cdot 1 + py_4 \cdot 1 + py_6 \cdot 1 = 5/6 + 1/12$입니다. 결론적으로 공격자의 Advantage는 $1/12$입니다.

이 작은 예시를 통해 우리는 안전성의 의미를 이해할 수 있습니다. 또한 가능한 각 sample(주사위의 예시에서는 1, 2, 3, 4, 5, 6)이 나올 확률이 두 종류에서 다를 경우 해당 차이가 공격자에게 Advantage로 작용함을 알 수 있습니다. 

암호학에서도 상황은 동일합니다. 두 함수 A, B를 구분하고 싶을 때, 만약 함수의 출력으로 가능한 모든 값에 대해 A에서 나올 확률과 B에서 나올 확률이 동일하면 A와 B를 구분할 수 있는 방법이 없습니다. 만약 둘의 확률이 차이나는 sample이 있다면(편의상 A일 확률이 더 높다고 할 경우), 블랙 박스에서 해당 sample을 받았을 경우 해당 블랙박스를 A라고 판단할 수 있습니다. 그리고 공격자의 최대 Advantage는

$$\Sigma_{s \in sample} \frac{1}{2} \cdot \| Pr[output(A) = s] - Pr[output(B) = s] \|$$

임을 알 수 있습니다. 말로 풀어서 쓰면 각 sample에서 확률 차이의 합의 절반입니다.

# 3. round function을 이용한 암호 구성

우리는 이상적인 round function $f$을 가지고 있다고 해봅시다. 이 이상적인 round function은 $k$비트의 키와 $n$비트의 입력을 받아서 $n$비트의 출력을 내고, random function과 구분이 불가능합니다(=$2^n$개의 모든 가능한 출력이 나올 확률이 전부 $2^{-n}$으로 동일합니다). 이 round function $f$을 가지고 $2n$비트의 입력을 받아 $2n$비트의 출력을 내는 암호를 만들고 싶다고 해봅시다. 이렇게 구조를 만들어서 암호로 사용하면 어떨까요?

- 임의로 $k_1$과 $k_2$를 정하고 메시지 $2n$비트의 $m$에 대해 왼쪽 $n$비트를 $m_l$, 오른쪽 $n$비트를 $m_r$이라고 하자. 알고리즘 $E$는 $E(m) = f(k_1, m_1) \| f(k_2, m_2)$로 정의된다.

일단 이 알고리즘 $E$는 $2n$비트를 받아 $2n$비트를 출력하긴 합니다. 그렇지만 $E$는 먼저 permutation이 보장이 안됩니다. 그리고 그건 차치하고서라도 공격자가 임의의 메시지를 이 암호에 질의할 수 있는 환경이라면 공격자는 $2n$비트의 random function과 $E$를 단 두 번의 질의로 구분해낼 수 있습니다. 공격자는 $0^n \| 0^n$, $0^n \| 1^n$을 질의해서 나온 결과의 왼쪽 $n$비트가 동일한지 확인하면 됩니다. 동일하다면 $E$로, 동일하지 않으면 random function으로 판단할 수 있습니다.

이와 같이 이상적인 round function $f$를 가지고 있다고 해도 이를 이용해 암호를 만드는게 생각보다 쉽지는 않은데, 우리는 여기서 페이스텔 구조를 이용할 수 있습니다.

# 4. 페이스텔 구조

페이스텔 구조는 암호 알고리즘의 구조 중 하나입니다. 페이스텔 구조를 가지는 암호 중 가장 유명한 것이 DES입니다. 페이스텔 구조에 대한 설명은 [이 글](https://www.secmem.org/blog/2019/02/06/block-cipher/)에서 간단하게 다룬 적이 있습니다. 포스팅을 꽤 오래전부터 했었어서 특히 암호학 관련 내용들은 예전 글에서 가져올 수 있는게 많네요 :p

$r$라운드 페이스텔 구조는 round function $f$와 round key $k_1, \dots, k_r$에 대해 아래와 같이 정의됩니다.

1. Input : $L_0 \| R_0$
2. $(L_i, R_i) = (R_{i-1}, L_{i-1} \oplus f(R_{i-1}, k_i)) \space \text {for} \space i = 1, \dots, r$ 
3. Output : $R_r \| L_r$

이 페이스텔 구조는 round function $f$의 역함수가 존재하지 않아도 되고, 임의의 round function $f$와 round key $k_1, \dots, k_r$에 대해 페이스텔 구조는 permutation입니다(다른 입력에 대해 동일한 출력이 없다는 의미이고, permutation이 아니면 애초에 암호로 사용할 수가 없습니다).

앞에서 한 가정과 같이 우리가 이상적인 round function $f$을 가지고 있다고 해봅시다. 그리고 라운드 키를 랜덤으로 생성해 페이스텔 구조를 만든 후 이 페이스텔 구조를 암호로 사용하고자 합니다. 과연 이 때 라운드는 얼마만큼으로 두어야 안전할까요? 참고로 DES에서는 16라운드였습니다.

이상적인 round function $f$가 있을 때 페이스텔 구조는 3라운드 이상일 때 CPA 환경에서 안전하고, 4라운드 이상일 때 CCA 환경에서 안전합니다. 만약 라운드가 2라운드 이하일 경우 CPA 환경에서 공격을 할 수 있는 방법이 있고, 3라운드 이하일 경우 CCA 환경에서 공격을 할 수 있는 방법이 있습니다. 공격이 그렇게 복잡하지 않고 논리적으로 생각해보면 쉽게 답을 얻을 수 있기 때문에 각각의 공격은 여러분에게 맡기겠습니다.

한편으로 `안전하다`는 표현은 다소 모호합니다. 누군가는 Advantage가 0이어야 안전하다고 생각할 수 있고, 누군가는 공격자가 $q$번의 입력을 넣어볼 수 있을 때 Advantage가 $q/2^{n}$ 이하여서 공격자가 대략 $2^n$번의 질의를 해야 random permutation과 주어진 구조를 구분할 수 있다면 안전하다고 생각할 수 있습니다. 3라운드 페이스텔 구조에서는 CPA 환경에서 공격자의 Advantage가 $q^2/2^{n}$이고 4라운드 페이스텔 구조에서는 CCA 환경에서 공격자의 Advantage가 $q^2/2^{n-1}$입니다. 두 경우 모두 공격자가 대략 $2^{n/2}$번의 질의를 해야 random permutation과 페이스텔 구조를 구분할 수 있습니다. 이 정도의 bound를 birthday bound라고 합니다. 

# 5. H-coefficient Technique

이번 글에서는 CPA 환경에서 3라운드 페이스텔 구조의 안전성을 같이 증명해보려고 합니다. 공격자가 $q$번의 질의를 할 수 있다고 한다면 공격자는 질문과 답변을 한데 모아 $\tau = (X_i, Y_i) (i \in 1, \dots, q)$을 받게 됩니다. 이러한 값 $\tau$를 transcript라고 부릅니다. 그러면 Advantage는

$$\Sigma_{s \in sample} \frac{1}{2} \cdot \| Pr[output(A) = s] - Pr[output(B) = s] \|$$

이니까 모든 가능한 transcript들에 대해 random permutation에서의 등장 확률과 3라운드 페이스텔 구조의 등장 확률을 전부 구하고 그 차이를 전부 더하면 되긴 하는데 사실 좀 많이 막막합니다. 이를테면 3라운드 페이스텔 구조에서 $X_i = 0^{2n}, Y_i = 0^{2n} (i \in 1, \dots, q)$일 확률을 우리가 쉽게 구할 수 있을지 생각해보면 딱히 그렇지는 않아보입니다.

그래서 안전성 증명을 수월하게 하기 위한 도구 중 하나인 H-coefficient Technique를 이용해 Advantage를 간접적으로 알아낼 생각입니다.

H-coefficient Technique은 개별적인 transcript sample에 대해 각각의 확률을 모두 구하는 대신 transcript를 Bad 혹은 Good 두 종류로 나누어 Advantage를 추산하는 방법입니다.

우리는 만약 공격자가 Bad에 속하는 transcript를 만나게 된다면 그냥 공격자가 둘을 완벽하게 구분할 수 있다고 가정할 것입니다. 이 가정은 확실히 공격자에게 유리한 가정입니다. 그리고 Good에 속하는 transcript 중에서 random permutation에서의 등장 확률과 3라운드 페이스텔 구조의 등장 확률의 차가 가장 큰 것이 얼마인지를 계산해서 공격자가 Good에 속하는 transcript를 만나게 된다면 공격자에게 가장 유리한(Good에 속하는 transcript 중에서 등장 확률의 차가 가장 큰 값만큼의) Advantage를 얻을 수 있다고 가정할 것입니다.

$$ \frac{Pr[\text{output}(\text{feistel}) \in \text{Good}]}{Pr[\text{output}(\text{random}) \in \text{Good}]} \leq 1 - \epsilon_1$$

이고 $Pr[\text{output}(\text{random}) \in \text{Bad}] \leq \epsilon_2$이라고 할 때(갑자기 식이 튀어나와서 당황할 수 있지만 첫 번째 식은 Good에 속하는 transcript 중에서 등장 확률의 차를, 두 번째 식은 공격자가 Bad transcript를 얻을 확률을 의미합니다), 공격자의 Advantage는 $\epsilon_1 + \epsilon_2$ 이하입니다.

이 H-coefficient Technique를 이용하면 개별적인 transcript의 등장 확률을 계산하는 대신 Bad를 적절하게 정의한 후

$$\frac{Pr[\text{output}(\text{feistel}) \in \text{Good}]}{Pr[\text{output}(\text{random}) \in \text{Good}]}$$

와 $Pr[\text{output}(\text{random}) \in \text{Bad}]$

만을 계산하면 됩니다.

# 6. CPA 환경에서 3라운드 페이스텔 구조의 안전성 증명

드디어 증명을 해볼 시간입니다. 우리는 증명에 도움이 되는 Bad를 잘 정의해야 합니다. transcript $\tau = (X_i, Y_i) (i \in 1, \dots, q)$에서 $X_i = (L_i, R_i), Y_i = (S_i, T_i)$로 두겠습니다. 우리는 어떤 $(S_i = S_j), i \neq j$, 즉 $S_i$에서 충돌이 발생한다면 이를 bad transcript로 두겠습니다. $S_i$에서의 충돌을 bad transcript로 두는 이유는 만약 충돌이 났을 경우 공격자가 random과 feistel을 높은 확률로 구분할 수 있기 때문인데, 구체적으로 어떻게 가능한지는 설명을 생략하겠습니다.

H-coefficient technique을 쓰기 위해 $Pr[\text{output}(\text{random}) \in \text{Bad}]$를 계산해야 하는데, random에서는 $S_i$가 랜덤으로 선택되니 임의의 $S_i, S_j$가 겹칠 확률은 $2^{-n}$이고 최종적으로 bad transcript가 나올 확률의 상한은 $q^2/2^{n+1}$으로 계산됩니다.

다음으로

$$\frac{Pr[\text{output}(\text{feistel}) \in \text{Good}]}{Pr[\text{output}(\text{random}) \in \text{Good}]}$$

을 계산해야 합니다. 어떤 임의의 good transcript $\tau$에 대해 random에서 $\tau$를 얻을 확률은 $2^{-2nq}$입니다. 문제는 ideal에서 $\tau$를 얻을 확률을 계산하는게 까다로운데, 일단 3개의 round function이 아주 잘 선택되어서 각 $i = 1, \dots, q$에 대해 $X_i$를 $Y_i$로 보내주어야 $\tau$를 얻을 수 있기 때문에 아래와 같은 식을 세울 수 있긴 합니다.

$$ \frac{ \vert \{ (f_1, f_2, f_3) \in \mathcal{F}^3_n : \text{feistel}[f_1, f_2, f_3](X_i) = Y_i \text{ for } i \in 1, \dots, q  \}     \vert }{ \vert \mathcal{F}^3_n \vert }$$

갑자기 너무 긴 식이 튀어나와서 난해할 수 있는데, 차근차근 뜯어서 보면 당연한 얘기입니다. $\mathcal{F}^3_n$는 round function 3개의 tuple을 의미합니다. 먼저 한 개의 round function의 개수를 보면 $2^n$개의 입력 각각이 $2^n$개의 값 중 하나를 가지기 때문에 $\vert \mathcal{F}_n \vert = 2^{n2^n}$입니다. 이 값을 우리는 $F_0$으로 표현하겠습니다. 그리고 $\vert \mathcal{F}^3_n \vert = (F_0)^3$입니다.

결국 우리는 각 $i = 1, \dots, q$에 대해 $X_i$를 $Y_i$로 보내는 라운드 function $(f_1, f_2, f_3)$ 쌍의 수만 세면 끝입니다. 이를 계산하기 위해 $f_1, f_2, f_3$ 각각을 잘 선택해봅시다.

## $f_1$ 선택
 
우리는 각 $i, j = 1, \dots, q$에 대해 $L_i \oplus f_1(R_i) \neq L_j \oplus f_1(R_j)$인 $f_1$을 선택할 계획입니다. 물론 $L_i \oplus f_1(R_i)$에서 충돌이 발생하더라도 우리가 원하는 transcript $\tau$를 얻을 가능성이 있지만 계산의 편의를 위해 충돌이 발생하면 이를 배제할 계획입니다.

이러한 $f_1$의 개수를 생각해보면, 만약 $R_i = R_j$라면 $L_i \neq L_j$이므로($L_i = L_j$라면 공격자는 동일한 입력을 2번 질의했다는 의미이므로 공격자의 입장에서 이런 짓을 하는건 손해) $L_i \oplus f_1(R_i) \neq L_j \oplus f_1(R_j)$이 늘 성립합니다.

만약 $R_i \neq R_j$라면 $L_i \oplus f_1(R_i) = L_j \oplus f_1(R_j)$가 되게 하는 $f_1$은 고정된 $i, j$에 대해 $F_0 / 2^n$개입니다. 그렇기 때문에 $i, j = 1, \dots, q$에 대해 $L_i \oplus f_1(R_i) \neq L_j \oplus f_1(R_j)$를 만족하는 $f_1$은 최소

$$F_0 - \frac{q(q-1)}{2} \cdot \frac{F_0}{2^n}$$

개가 있습니다.

## $f_2$ 선택

$f_2$에서는 오른쪽 $n$비트의 값이 출력에서의 왼쪽 $n$비트($S_i$) 값과 일치해야 합니다. 즉 각 $i = 1, \dots, q$에 대해 $f_2(L_i \cdot f_1(R_i)) = R_i \oplus S_i$가 만족되어야 하므로 $f_2$는 정확히 

$$\frac{F_0}{2^{qn}}$$

개가 있습니다.

## $f_3$ 선택

$f_3$에서는 왼쪽 $n$비트의 값이 출력에서의 오른쪽 $n$비트($T_i$) 값과 일치해야 합니다. 즉 각 $i = 1, \dots, q$에 대해 $f_3(S_i) = L_i \oplus f_1(R_i) \oplus T_i$가 만족되어야 하므로 $f_3$ 또한 마찬가지로 정확히

$$\frac{F_0}{2^{qn}}$$

개가 있습니다.

최종적으로 $(f_1, f_2, f_3)$ 쌍의 수는

$$ \left(F_0 - \frac{q(q-1)}{2} \cdot \frac{F_0}{2^n}\right) \cdot \left(\frac{F_0}{2^{qn}}\right) \cdot \left(\frac{F_0}{2^{qn}}\right) \leq \frac{F^3_0}{2^{2qn}} \left( 1 - \frac{q^2}{2^{n+1}}  \right)$$

개 이상입니다. 그리고

$$\frac{Pr[\text{output}(\text{feistel}) \in \text{Good}]}{Pr[\text{output}(\text{random}) \in \text{Good}]} \leq 1 - \frac{q^2}{2^{n+1}}$$

이 되어 공격자의 Advantage는 최대

$$\frac{q^2}{2^{n+1}} + \frac{q^2}{2^{n+1}} = \frac{q^2}{2^{n}}$$

입니다.

마지막으로 Advantage가

$$\frac{q^2}{2^{n+1}} + \frac{q^2}{2^{n+1}} = \frac{q^2}{2^{n}}$$

이하임은 H-coefficient technique을 통해 알았고, 실제 이 bound만큼의 advantage를 주는 공격을 찾아내면 Advantage의 tightness를 증명할 수 있습니다. 어떤 공격이 가능한지는 여러분에게 맡기겠습니다.

# 7. Conclusion

이번 글에서는 안전성 증명을 알아보고 기초적인 증명 방법인 H-coefficient technique를 익혀 실제 3라운드 페이스텔 구조의 안전성을 증명해보았습니다. 수식으로 빼곡해 단순히 눈으로만 보고 따라가기에는 굉장히 난이도가 있지만 안전성 증명에 대해 제대로 이해를 하고싶다면 직접 종이로 계산 과정을 따라가며 이해해보는걸 추천드립니다.
