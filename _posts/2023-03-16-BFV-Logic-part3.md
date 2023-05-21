---
layout: post
title:  "BFV scheme에 대한 소개 - 3"
date:   2023-03-16 09:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

지금까지 BFV Scheme에 관한 [설명글](https://infossm.github.io/blog/2023/01/17/BFV-Logic-part1/)과, BFV scheme의 practical한 구현을 위해서 RNS module이라는 걸 사용한다는 것 그리고, RNS module에서 operations들이 기본적으로 어떻게 돌아가는지에 대해 설명하는 [글](https://infossm.github.io/blog/2023/01/17/BFV-Logic-part2/)을 썼습니다. 이번 글에서는 RNS module에서 구현하기 까다로운 연산들을 어떻게 처리하는지에 대해 설명하겠습니다.

# Backgrounds

저번 글에 좀 더 자세히 설명하고 있지만, Background를 한 번 더 짚고 넘어가겠습니다. RNS module은 널리 알려진 CRT(Chinese Remainder Theorem)에 기반하여 큰 범위의 수를 작은 수 여러 개로 표현하는 방식입니다. Homomorphic Encryption은 보안상의 이유 때문에 대부분 아주 큰 수를 사용하고, 이를 practical하게 구현할 때 RNS module을 써서 컴퓨터에서 쉽게 표현할 수 있는 범위의 수 여러개로 표현하는 방식을 채택하는 것입니다.

RNS module에서는 덧셈, 뺄셈, 곱셈은 쉽게 할 수 있지만, 수 끼리의 나눗셈 같은 연산은 적용하기 힘듭니다. BFV scheme에서는 나눗셈을 사용하는 decryption이 존재하기 때문에, RNS modulue을 사용해서 구현하는 것이 쉽지 않습니다. 이를 위해서는 나누는 수를 big integer등을 통해 직접 복원한다던가 하는 것이 필요합니다. 이 과정은 복잡하고 시간도 굉장히 오래 걸리기 때문에, 이 문제를 해결하기 위해 많은 연구가 진행됐습니다.

그리고 BFV scheme이 제안된 후에 RNS module을 사용해서 decyprtion 등도 수행 가능한 BFV scheme에 대한 연구들이 여럿 나왔습니다. 오늘은 그것들 중에서 가장 유명한 연구인 [BEHZ](https://eprint.iacr.org/2016/510)에 기반해서 BFV scheme 상에서 decryption을 어떻게 RNS module 상태에서 구할 수 있는지 설명하도록 하겠습니다.

# BEHZ Decryption

논문 저자들의 이름을 순서대로 따로 BEHZ라 불려지는 scheme은 BFV scheme의 operation들을 RNS로 표현해서 수행할 수 있게 합니다.

이제 BEHZ에서 어떻게 decryption을 수행하는지 살펴보겠습니다. 그 전에, 이전 글에서 나왔던 BFV scheme에서 decryption 식을 한번 더 봅시다.

$$ M = [\lfloor \frac{t [C_{0}+ C_{1} \cdot sk]_q}{q} \rceil]_t $$

식을 보면, $q$로 일단 나누고, 그 나눈 결과를 rounding 까지 한다는 걸 알 수 있습니다. 지금까지 나눗셈이 RNS 상에서 잘 안되서 힘들다고 설명했는데, 정확히는 나누고, rounding까지 해야합니다.

단순하게 접근해서는 이 문제를 해결하기 힘들어 보입니다. 어떻게 해야할까요?

BEHZ에서는 일단 직접적으로 문제를 해결하기 어려우니, 우선 보조함수를 도입해 approximate한 결과를 구하고, 적절히 correcttion을 해주어 문제를 해결합니다.

## Fast RNS base conversion

위에서 언급한 보조함수 중 하나가 Fast RNS base conversion입니다. 우선 ciphertext의 공간과, plaintext의 공간이 서로 다르다는 것은 알고 계실 겁니다. 엄밀히 ciphertext는 구성하는 다항식이 $R_q$에 속하고, plaintext는 $R_t$의 다항식입니다. 즉, RNS를 이루는 base가 다르다는 것입니다.

어떤 동일한 두 수를 표현하는 서로 다른 RNS base가 있으면 RNS base에 맞게 변환해주는 함수가 여기에서 설명할 Fast RNS base conversion 입니다.

정석적으로 하자면, 먼저 기존 RNS base에서 residue 정보를 토대로 복원하고, 그 것을 새로운 RNS base에 따라 residue들을 구해주는 방법이 있겠지만, 아무래도 좀 느릴 것입니다. big integer 같은 자료가 필요할 것이기도 하고요.

빠른 변환을 위해서, 다음과 같은 식을 가지는 conversion을 생각할 수 있습니다.

현재의 base가 q이고, 새로운 base $B$로 $0 \le x < q$인 $x$를 옮긴다고 하면, 단, $B$는 $q$와 서로소입니다.

$$ \textbf{FastBConv}(x, q, B) = (\sum^{k}_{i=1}{\lvert x_{i} \frac{q_i}{q} \rvert _{q_i} \times \frac{q}{q_i} \ mod \ m})_{m \in B}$$

이 함수를 통해서, 서로 다른 base들 간에 conversion을 빠르게 할 수 있습니다. 다만, 위 함수가 아주 정확하게 conversion을 하는 것은 아닙니다. 정확히는 위 함수는 정수 $x$를 $x+ \alpha _{k}q$로 옮깁니다. 여기에서 $\alpha _{k}$는 $0$이상 $k$미만입니다. 그렇기 때문에 이 함수는 어디까지나 'approximate한' conversion을 수행합니다.

## Approximate RNS rounding

이제 다음으로, decryption의 정의에 쓰였던 복잡한 식을, 좀 더 RNS에서 풀기 편한 형태로 변형시켜봅시다.

이를 위해 몇 가지를 관찰하면 다음과 같습니다. 먼저, $t/q$가 다항식에 곱해진 형태이기 때문에, $[C_{0}+ C_{1} \cdot sk]$에 계수가 $q$의 배수인 다항식을 얼마든지 더해도 상관없습니다. $t/q$가 곱해지고 마지막에 $t$에 대한 나머지를 구할 것이기 때문입니다.

첫 번째 이유에서 파생된 이유로, $[ct(\textbf{s})]$ 대신 $\lvert ct(\textbf{s}) \rvert$를 써도 상관없습니다. 전자는 $0$이상 $q$미만, 후자는 $-q/2$이상, $q/2$이하의 범위의 정수에 대응됩니다.

위의 사실을 종합하면 다음과 같이 식을 변형시키는 것이 가능합니다.

$$ \lfloor \frac{t [ct(\textbf{s})]_q}{q} \rfloor = \frac{t \lvert ct(\textbf{s}) \rvert _{q} \ - \ \lvert t \cdot ct(\textbf{s}) \rvert _{q}}{q} $$

이제 rounding 식이 없어졌고, exact하게 나눗셈을 적용시키니 RNS에서 보다 쉽게 적용이 가능합니다. 그리고, 이 결과에 결국 $mod \ t$를 취할 것이기 때문에, $t \lvert ct(\textbf{s}) \rvert _{q}$ 부분은 어차피 없어질 것이고, $\ \lvert t \cdot ct(\textbf{s}) \rvert _{t}$를 계산하면 되는데, 이 것은 $q$에 대한 결과를 위의 Fast RNS base conversion을 사용하면 변환할 수 있습니다.

그런데 Fast RNS base conversion 에서, 위 함수가 approximate한 변환이라고 언급했습니다. approximate한 결과의 correcting을 하면, 정확한 결과를 얻을 수 있습니다. 

## BEHZ Decryption Alogrithm

본격적으로 Algorithm을 설명하기 위해, 다음 Lemma들을 사용할 것입니다.

Lemma 1

$ct(\textbf{s}) _q = \Delta [ M ] _t + v + qr$일 때, $ \textbf{v}_{\textbf{c}} := t \textbf{v} - [ M ] _t \lvert q \rvert _t$ 라고 하자. $ \gamma $가 $q$와 서로소인 정수라고 할 때, $m \ = \ \{t, \gamma \}$ 인 base $m$을 생각하자. 다음이 성립한다.

$$ \textbf{FastBConv}( \lvert \gamma t \cdot ct(\textbf{s})\rvert _q , q, m ) \times \lvert - q^{-1} \rvert _m = \lfoor \gamma \frac{t}{q} [ct(\textbf{s})] _q \rceil - \textbf{e} = \gamma (M + tr) + \lfoor \gamma \frac{\textbf{v}_{\textbf{c}}}{q} \rceil - \textbf{e} $$

여기에서 error $\textbf{e}$는 $[0, \  k]$를 계수 범위로 하는 error polynomial을 의미합니다.  
  

Lemma 2

$\lVert \textbf{v}_{\textbf{c}} \rVert _ {\infty} \le q(\frac{1}{2} - \varepsilon)$ 이고, $\textbf{e}$가 $[0, \  k]$를 계수 범위로 하는 error polynomial이고, $\gamma$가 자연수일 때, 다음이 성립한다.

$$ \gamma \varepsilon \ge k \Rightarrow [ \lfloor \gamma \frac{\textbf{v}_{\textbf{c}}}{q} \rceil - \textbf{e} ] _{\gamma} = \lfloor \gamma \frac{\textbf{v}_{\textbf{c}}}{q} \rceil - \textbf{e} $$

위의 두 Lemma를 사용해서, 다음과 같은 Theorem이 성립하게 됩니다.  
  

Theorem 1

$ct(\textbf{s}) = \Delta [ M ] _t + \textbf{v} (mod \ q)$이고, $\gamma$가 $t,q$와 서로소인 정수일 때, $\gamma > 2k / (1 - \frac{t \lvert q \rvert _t}{q})$를 만족하고, $\textbf{v}$가 다음 bound를 만족하면, Alg 1. 은 올바른 $M$을 반환한다.

$$ \lVert \textbf{v} \rVert _{\infty}  \le \frac{q}{t}(\frac{1}{2}-\frac{k}{\gamma}) - \frac{\lvert q \rvert _t}{2}$$

Alg 1.은 다음과 같습니다.

$$ \textbf{s} ^{(t)} \ \leftarrow \  \textbf{FastBConv}(\lvert \gamma t \cdot ct(\textbf{s})\rvert _q, q, \{ t \}) \times \lvert - q^{-1} \rvert _t$$
$$ \textbf{s} ^{(\gamma)} \ \leftarrow \  \textbf{FastBConv}(\lvert \gamma t \cdot ct(\textbf{s})\rvert _q, q, \{ \gamma \}) \times \lvert - q^{-1} \rvert _{\gamma}$$
$$ \tilde{\textbf{s}} ^{(\gamma)} \leftarrow  [ \textbf{s} ^{(\gamma)} ] _{\gamma}$$
$$ M \leftarrow [ (\textbf{s} ^{(t)} - \tilde{\textbf{s}} ^{(\gamma)}) \times \lvert \gamma ^ {-1}\rvert _t] _t$$

Lemma 들 및 Theorem에 대한 증명은 어렵지 않으나, 여기에 다 적기는 애매한 감이 있어 넘어가도록 하겠습니다. Lemma와 Theorem에 대한 자세한 증명은 BEHZ를 제안한 [paper](https://eprint.iacr.org/2016/510)에 자세히 나와 있으니 참고하시면 좋겠습니다.

brief하게만 설명을 드리면, Lemma 1에서 $ \textbf{s} ^{(t)}, \textbf{s} ^{(\gamma)}$ 가 어떤 형태가 되는지 알아냈고, Lemma 2에 의해 $\tilde{\textbf{s}} ^{(\gamma)}$가 error항이 됨을 알 수 있습니다. 이 error항은 $\textbf{s} ^{(t)}$에도 공통적으로 있으니 이걸 빼주면 Lemma 1에서 본 $\gamma (M+tr)$항만 남고, $M$을 구하려면 $\gamma$의 $t$에 대한 역수를 곱하고, $tr$은 $t$가 곱해져 있으니 $t$에 대한 $mod$를 취하면 자연스럽게 사라집니다.

# Conclusion

BFV scheme에서 decryption 등의 연산은 RNS를 적용하기 어렵기 때문에, RNS를 사용하기 위해 여러 가지 방법들이 제안됐고, 그 중 하나인 BEHZ에서의 decrryption을 알아보았습니다. 사실 BEHZ 말고도 제안된 것들은 더 있고, HPS역시 유명하지만, 아쉽게도 여기에 다 담기엔 좀 힘들었습니다.

다음에는 HPS에서의 decyprtion 혹은 RNS에서 BFV scheme의 Multiplication에 대해 설명하는 글을 쓰도록 하겠습니다.

# Reerence

- https://eprint.iacr.org/2016/510

