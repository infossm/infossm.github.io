---
layout: post
title:  "BFV scheme에 대한 소개 - 1"
date:   2023-01-17 23:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

저번 [글](https://infossm.github.io/blog/2022/11/22/Homomorphic-Encryption-Introduce/)에서 Homomorphic Encryption, 동형암호에 대해서 소개하는 글을 썼습니다.

이번 글부터는 저번 글에서 소개한 여러가지 FHE scheme 중에서, BFV scheme에 대해 설명하는 글을 쓰려고 합니다. 저번 글에 Homomorphic Encryption에 대한 간단한 설명과 특징을 설명했습니다. 이 글에서는 Homomorphic Encryption에 기본 개념에 대해서 알고 있다고 생각하고 설명할 것이기 때문에, 이번 글을 읽기 전에 저번 글을 읽고 오시는 것을 추천드립니다.

이번 글에서는 BFV scheme의 다양한 연산들에 대해서 설명하도록 하겠습니다.

# Key Generation

BFV scheme에서 key 생성을 어떻게 하는지 설명하기에 앞서, BFV scheme에서 다루는 공간에 대해서 설명하도록 하겠습니다. 다른, FHE scheme들과 마찬가지로, BFV scheme에서는 주어진 정보를 encode해서 plaintext로 옮기고, plaintext를 ciphertext로 encrypt하고, ciphertext들에 대해서 addition, multiplication 등을 진행합니다. 그 후 ciphertext를 다시 plaintext로 decrypt한 후, 그 plaintext를 decode해서 정보를 얻습니다.

저번 글에서 설명한 것처럼 BFV scheme은 RLWE problem에 기반합니다. 그래서 plaintext의 경우 다항식(polynomial) 하나, ciphertext의 경우 다항식의 쌍으로 표현됩니다. 이때, plaintext와 ciphertext의 경우 속하는 다항식의 공간이 다릅니다. 엄밀하게 표현하면, plaintext의 경우 어떤 자연수 $t$에 대해 $R_{t} = \mathbb{Z}_{t}[x]/(x^{n}+1)$, ciphertext의 경우 어떤 자연수 $q$에 대해 $R_{q} = \mathbb{Z}_{q}[x]/(x^{n}+1)$가 다항식이 속하는 공간이 됩니다. 이때, $t\lll{q}$여야 합니다. $n$의 경우, 적당한 자연수 $d$에 대해서 $n = 2^d$입니다.

이제 위의 $t,q,n$이 정해지면, 이 값들에 따라 필요한 key들을 생성합니다. 생성할 key들은 다음과 같이 3개 입니다. $x \leftarrow R$은 ring $R$에서 random하게 하나를 뽑은 것을 의미합니다.

먼저 secret key입니다. (여기서부터 sk라고 쓰겠습니다) sk는 다른 key들을 생성하고, decryption등 중요한 연산에 사용됩니다. sk의 경우 $sk \leftarrow R$와 같이 생성됩니다. 여기서 $R = R[x]/(x^{n}+1)$이고, $R[x]$의 경우, 계수가 $-1,0,1$ 중 하나인 다항식 ring입니다. 

다음은 public key입니다. (여기서부터 pk라고 쓰겠습니다) pk는 encryption 등에 사용됩니다. pk는 다음과 같이 생성됩니다. 이미 생성된 sk와 $a \leftarrow R_q$, $e \leftarrow \chi$과 같이 생성된 $a, e$에 대해서, 다음과 같이 생성됩니다. 여기에서 $\chi$는 정해진 $\mu, \sigma$에 대한 discrete normal distribution에 따라 각 계수를 뽑은 다항식 ring을  $\chi [x]$라고 하면, $\chi = \chi [x]/(x^{n}+1)$입니다.

\begin{align*}
    & pk = ([-(a \cdot sk + e)]_{q}, a)
\end{align*}

마지막으로 evaluation key입니다. (여기서부터 ek라고 쓰겠습니다) ek는 relinearization 등에 사용됩니다. ek의 경우 relinearization logic 등에 따라서 생성방식이 달라질 수 있지만, 본문에서 설명할 방식에 필요한 key는 다음과 같이 생성됩니다.

\begin{align*}
    & ek = ([-(a \cdot sk + e)+sk^{2}]_{q}, a)
\end{align*}

이때, ek를 생성할 때의 $a$는 pk와 독립적입니다.

위에서 설명한 것처럼, $t,q,n$이 정해지면 BFV scheme에서 필요한 key들을 생성할 수 있습니다. 따라서, $t,q,n$을 잘 설정하는 것이 중요합니다. $t,q,n$을 어떻게 설정하느냐에 따라 security level이 달라집니다. (정확히는 $q,n$이 관여합니다. $t$의 경우 security level과 관련이 업습니다.) 일반적으로 $t\lll{q}$를 만족하면 BFV로 기능할 수는 있습니다만, 여러 가지 이유 때문에 $t$는 특정 조건을 만족하는 소수, $q$의 경우 특정 조건을 만족하는 소수들의 곱으로 설정됩니다. 이 조건들의 경우 나중에 다시 자세하게 설명하도록 하겠습니다.

# Encode and Decode

이제 BFV에서 encode와 decode를 어떻게 하는지 알아보도록 합시다. encode는 원하는 정보를 대응되는 plaintext로 옮기는 것이고, decode는 그 반대입니다. 사실 encode의 경우에는 BFV scheme에서 어떻게 해야 한다는 원칙같은 것이 없기 때문에, 적절히 encoding 방식을 선택하면 됩니다. 정보의 형태에 따라서도 encoding 방식이 달라질 것입니다.

예를 들어, 정수 하나를 encode한다고 합시다. encode하기를 원하는 정수를 $k$, 옮기고자 하는 plaintext의 공간을 $R_t$라고 합시다. 먼저 간단하게, 다음과 같이 표현할 수 있을 것입니다.

\begin{align*}
    & k + 0 \cdot x + 0 \cdot x^{2} + \dots
\end{align*}

$k$를 상수항으로 하고, 나머지 계수들은 $0$인 $R_t$의 다항식에 대응시킨 것입니다. 매우 간단하고, 직관적인 방식입니다. 하지만 크게 두 가지 문제점이 있습니다.

- 첫 번째로, 낭비가 너무 심합니다. 상수항을 제외한 나머지 값들이 모두 $0$으로 고정돼있습니다.
- 두 번째로, 저장할 수 있는 $k$의 값의 상한이 너무 작습니다. 즉, 변환할 수 있는 정보의 범위가 한정돼있습니다. plaintext의 경우 $R_t$에 속하는데, 이 말은 다항식의 계수 값이 $\matthbb{Z}_t$에 속한다는 것입니다. 따라서 이 방식으로 저장 가능한 서로 다른 정보의 개수가 $t$개에 불과합니다.

따라서 위와 같은 방식은 좋은 encoding 방식이라고 하기 어렵습니다. 또 다른 방식 중 하나는 다음과 같습니다. $k$가 $t$진법으로 표현했을 때, $k = \overline{k_{m-1}k_{m-2} \cdots k_{0}}$와 같이 표현된다고 할 때, 다음과 같이 표현하는 것입니다.

\begin{align*}
    & k_{0} + k_{1}x + k_{2}x^{2} + \dots + k_{m-1}x^{m-1} + 0 \cdot x^{m} + \dots
\end{align*}

즉, $m$차 미만은 진법표현에서 각 자릿수를 계수로, $m$차 이상의 계수는 전부 0으로 하는 것입니다. 이 방식을 사용하면 첫번째 방식에 나온 단점을 어느 정도 해소할 수 있습니다. 우선 상수항만 사용하는 것이 아니니 낭비도 줄일 수 있고, 저장할 수 있는 $k$값의 상한도 $n$을 늘리면 충분히 원하는 만큼 늘릴 수가 있습니다. (base의 경우 다른 것을 사용해도 됩니다.)

한 개의 정수가 아니라 여러 개의 정수를 저장해야 한다면 어떻게 해야 할까요? 위의 방식을 응용할 수도 있을 것이고, 그 외에 여러 가지 방법이 있을 수 있습니다. 만약 저장해야하는 정수들이 $Z_t$의 범위안에 들어간다면, batch encode라는 방식을 사용할 수 있습니다.

batch encode는 Number Theoretic Transform, NTT에 기반하여 encoding 방식입니다. NTT에 대한 자세한 설명은 다른 좋은 글들이 많으니 구글링을 통해 공부하시면 좋을 것입니다.

encode를 할 때는 inverse NTT를 사용합니다. 저장하고자 하는 정수들을 $\matthbb{Z}^{n}$ vector처럼 생각한 후, vector에 대응되는 $Z_{t}$에 속하는 다항식으로 변환하는 것이 batch encode 입니다. 이때, $t \ mod \ 2n \equiv 1$을 만족해야 합니다. 그렇지 않으면 batch encoding을 쓸 수 없습니다. batch encoding을 사용할 수 있는 것이 여러모로 편리하기 때문에, $t$의 경우 위 조건을 만족하는 작은 소수로 정하는 경우가 많습니다.

batch encode를 실제로 구현하는 구현체의 경우, MS SEAL을 예로 들면 [이 곳](https://github.com/microsoft/SEAL/blob/main/native/src/seal/batchencoder.cpp)에서 확인 가능합니다. BatchEncoder 객체를 만들 때 실제로 NTT가 가능한 parameter setting인지 확인하는 코드가 있음을 확인할 수 있습니다.

그 외에 정수가 아니라 실수일 때, 음수가 섞여 있을 때 등 다양한 상황이 있을 것이고, 상황에 따라 적절한 encoding 방식을 선택하면 됩니다. 되도록이면 넓은 범위의 정보를 저장할 수 있는 방식이 좋을 것입니다.

decode의 경우, encode의 반대가 되게 하면 됩니다. 위에서 설명한 방식 중 batch encode의 경우, decode를 할 때에는 그냥 NTT를 하면 될 것입니다.

# Encryption

다음으로 encryption을 설명하도록 하겠습니다. encryption의 경우 encode, decode와는 달리 decryption과 logic이 좀 다른 편이라, section을 나누어 설명합니다.

BFV scheme에서 ciphertext의 경우, $R_q$에 속한 두 다항식의 쌍으로 표현됩니다. encrypt 하고자 하는 plaintext를 $M$이라고 합시다. $M$을 encrypt해서 생성한 ciphertext를 $C \ = \ (C_{0}, C_{1})$라고 하면, 다음과 같이 생성됩니다.

$u \leftarrow R$, $e_{0}, e_{1} \leftarrow \chi$로 생성된 $u, e_{0}, e_{1}$과, $\Delta = \lfloor q/t \rfloor$, $pk = (pk_{0}, pk_{1})$에 대하여 $C_{0}, C_{1}$은 다음과 같습니다.

\begin{align*}
    & C_{0} = [pk_{0} \cdot u + e_{0} + \Delta M]_{q} \\
    & C_{1} = [pk_{1} \cdot u + e_{1}]_{q}
\end{align*}

# Decryption

encryption을 설명했으니, 이제 decryption을 설명할 차례입니다. 어떤 ciphertext $C = (C_{0},C_{1})$를 decrypt하면 생성되는 평문 $M$은 다음과 같습니다.

\begin{align*}
    & M = [\lfloor \frac{t [C_{0}+ C_{1} \cdot sk]_{q}}{q} \rceil]_{t}
\end{align*}

수식이 약간 복잡합니다. 괄호 안에서부터 해석해보면, $R_q$ 아래에서 $C_{0}+ C_{1} \cdot sk$를 먼저 계산한 후, $t/q$를 곱한 것을 반올림하여 $R_t$의 다항식의 결과를 도출하면 됩니다.

좀 더 자세히 살펴보면, 우선 $C_{0}+ C_{1} \cdot sk$의 경우 다음과 같이 전개할 수 있습니다.

\begin{align*}
    C_{0}+ C_{1} \cdot sk & = pk_{0} \cdot u + e_{0} + \Delta M + (pk_{1} \cdot u + e_{1}) \cdot sk \\
    & = -(a \cdot sk + e) \cdot u + e_{0} + \Delta M + (a \cdot u + e_{1}) \cdot sk \\
    & = - a \cdot u \cdot sk  - e \cdot u + e_{0} + \Delta M + a \cdot u \cdot sk + e_{1} \cdot sk \\
    & = \Delta M - e \cdot u + e_{0} + e_{1} \cdot sk \\
    & = \Delta M + v
\end{align*}

따라서 위의 결과를 이용하면 $[C_{0}+ C_{1} \cdot sk]_{q} = \Delta M + v + q \cdot f$라고 둘 수 있습니다. ($f$는 정수 계수 다항식) 이제 여기에 $t/q$를 곱하면 다음과 같이 될 것입니다.

\begin{align*}
    & \lfloor \frac{t [C_{0}+ C_{1} \cdot sk]_{q}}{q} \rceil = M + \lfloor (t/q)(v - \epsilon \cdot M) \rceil + t \cdot f
\end{align*}

여기서 $\epsilon = q/t - \Delta < 1$입니다. 결과를 $R_t$안에서 계산하므로, 마지막 항은 결과에 영향을 주지 않고, 두 번째 항, 즉 $(t/q)(v - \epsilon \cdot M)$의 값이 rounding 되어 decrypt가 올바르게 되려면, $(t/q) \left\Vert (v - \epsilon \cdot M) \right\Vert < 1/2$를 만족하면 됩니다. 그렇지 않으면 overflow가 일어나 올바르게 decrypt가 되지 않습니다.

# Addition

위에서 encode, decode, encrypt, decrypt 에 대한 설명을 마쳤습니다. 남은 것은 evaluations, 즉 ciphertext 끼리의 연산에 대한 설명입니다. BFV에서 사용하는 연산은 addition, multiplication 입니다. 먼저 addition에 대해서 설명하도록 하겠습니다.

addition은 그렇게 어렵지 않습니다. 두 ciphertext를 이루는 다항식을 순서대로 더해주면 됩니다. 더하고자 하는 두 ciphertext $C_{0} = (A_{0}, B_{0}), C_{1} = (A_{1}, B_{1})$에 대하여, 두 ciphertext를 더한 ciphertext $C = (A,B)$는 다음과 같습니다.

\begin{align*}
    & A = A_{0} + A_{1} \\
    & B = B_{0} + B_{1}
\end{align*}

# Multiplication and Relinearization

addition과 달리, multiplication은 약간 더 복잡합니다. 크게 두 가지 단계가 있는데, 하나는 ciphertext끼리 계산의 결과에 의해 다항식 3개를 얻는 단계이고, 나머지 하나는 이렇게 얻는 다항식 3개를 다시 2개로 줄이는 단계입니다.

수식과 관련된 부분은 나중에 설명하고, 먼저 BFV에서 ciphertext의 곱셈이 개략적으로 어떻게 이루어지는지 설명하도록 하겠습니다. 앞서 decrypt를 할 때, 어떤 ciphertext $C = (A,B)$에 대해서, $A + B \cdot sk = \Delta M + v$를 계산했습니다. 오차항에 해당하는 $v$를 일단 제외하고 생각해보면, 저 수식의 값이 대략적으로 plaintext $M$에 $\Delta$를 곱한 형태라는 것을 알 수 있습니다.

그렇다면 $C_{0} = (A_{0}, B_{0}), C_{1} = (A_{1}, B_{1})$를 곱할 때, $(A_{0} + B_{0} \cdot sk)(A_{1} + B_{1} \cdot sk) \simeq \Delta ^{2} M_{0} M_{1}$입니다. 따라서 $t/q(A_{0} + B_{0} \cdot sk)(A_{1} + B_{1} \cdot sk) \simeq \Delta M_{0} M_{1}$이 될 것입니다.

$(A_{0} + B_{0} \cdot sk)(A_{1} + B_{1} \cdot sk) = A_{0}A_{1} + (A_{0}B_{1} + A_{1}B_{0}) \cdot sk + B_{0}B_{1} \cdot sk ^{2}$이므로, 우선 $A_{0}A_{1}, (A_{0}B_{1} + A_{1}B_{0}), B_{0}B_{1}$를 계산하고 $t/q$를 곱한 것을 순서대로 $ct_{0}, ct_{1}, ct_{2}$라고 합시다. 이 $ct_{0}, ct_{1}, ct_{2}$를 찾는 것이 첫 번째 단계입니다.

$ct_{0} + ct_{1} \cdot sk + ct_{2} \cdot sk^{2} \simeq \Delta M_{0} M_{1}$이고, 이때, $ct_{0} + ct_{1} \cdot sk + ct_{2} \cdot sk^{2} \simeq A + B \cdot sk$인 $C = (A,B)$를 찾으면 3개인 다항식을 2개로 줄일 수 있습니다. 이 과정이 두 번째 단계입니다. 두 번째 단계의 결과에서 보듯이 원래 다항식 3개에 대해서는 sk의 차수가 2차이지만, 두 번째 단계를 거쳐 다시 2개로 줄면 sk의 차수가 다시 1차입니다. 즉, 다시 선형으로 복귀시켰기 때문에, 두 번째 단계를 'relinearization'이라고 합니다.

위에서 개략적으로 설명을 마쳤으니, 결과를 수식으로 정리합시다. 두 ciphertext $C_{0}, C_{1}$을 곱한 결과로 다항식 3개인 $ct_{0}, ct_{1}, ct_{2}$를 얻어야 합니다. 이에 대한 자세한 계산은 분량이 너무 길어지기 때문에, 우선 결과만 설명하도록 하겠습니다. 자세한 계산은 참고문헌을 확인하시기 바랍니다.

\begin{align*}
    & ct_{0} =[\lfloor \frac{t [A_{0} \cdot A_{1}]_{q}}{q} \rceil]_{q} \\ 
    & ct_{1} =[\lfloor \frac{t [A_{0} \cdot B_{1} + A_{1} \cdot B_{0}]_{q}}{q} \rceil]_{q} \\
    & ct_{2} =[\lfloor \frac{t [B_{0} \cdot B_{1}]_{q}}{q} \rceil]_{q}
\end{align*}

두 번째 단계(relinearization)의 결과는 다음과 같습니다. 실제로 BFV를 제안한 paper에서는 noise 관련 처리 때문에 좀 더 복잡한 처리를 합니다만, 이 글에서는 그런 것을 일단은 신경쓰지 않았습니다.

\begin{align*}
    & A = [ct_{0} + ek_{0} \cdot ct_{2}]_{q} \\
    & B = [ct_{1} + ek_{1} \cdot ct_{2}]_{q}
\end{align*}

# Conclusion

지금까지 BFV scheme에 대해 간략하게 알아보았습니다. Multiplication에 대해서 설명할 때 언급했습니다만, BFV scheme과 같은 FHE scheme에서는 noise growth에 대한 관리가 중요하며, 실제로 BFV를 제안한 paper에서는 error의 상한 등을 수식으로 명확하게 정리하고 있습니다. 또, error growth를 줄이기 위해 좀 더 어렵고 귀찮은 처리를 하기도 합니다. 하지만, 여기에서는 그런 것들을 전부 서술하면 너무 복잡해질 것으로 예상되어 넘어갔습니다. 이 부분에 대해서 좀 더 자세히 알고 싶은 분들은 참고문헌으로 공부하시면 좋을 것 같습니다.

BGV, BFV, CKKS 등의 FHE scheme들은 모두 매우 큰 수를 기반으로 합니다. 실제로 BFV도 보통 $q$의 값이 매우 큰 정수입니다. 하지만 computer가 64-bit 이상의 정수를 표현하고 다루는 것이 힘들기 때문에, 보통 BGV, CKKS 같은 경우 scheme이 제안된 후에, RNS를 통해서 큰 공간을 작은 공간 여러 개의 곱으로 쪼갠 후 encrypt, decrypt 등이 적용될 수 있도록 합니다.

이때, BGV, CKKS와는 다르게, BFV의 경우 decrypt등에서 RNS를 적용하기 힘들어 많은 연구를 거쳐서 RNS에 대해 BFV를 적용한 논문이 나왔습니다. 그 중 BEHZ scheme, HPS scheme이 특히 유명합니다. 다음 글에서는 RNS를 적용한 BFV sheme에 대해서 설명하는 글을 작성하도록 하겠습니다. 부족한 글을 읽어주셔서 감사합니다!

# Reference

- https://www.inferati.com/blog/fhe-schemes-bfv
- https://eprint.iacr.org/2012/144

