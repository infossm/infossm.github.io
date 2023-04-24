---
layout: post
title:  "BFV scheme에 대한 소개 - 4"
date:   2023-04-23 09:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

지금까지 BFV Scheme에 관한 설명글들을 썼고, 저번 [글](https://infossm.github.io/blog/2023/03/16/BFV-Logic-part3/)에서는 RNS form에서 처리하기 까다로운 연산 중에서 BFV decryption을 어떻게 처리하는지에 대해서 설명했습니다. 이번 글에서는 multiplication을 어떻게 처리하는지에 대해서 설명하도록 하겠습니다.

# Backgrounds

이번에 설명할 연산은 RNS module의 BFV scheme에서 multiplication 입니다. 얼핏 생각했을 때, decryption에서 rounding을 correction하는 법도 이미 구했고, Multiplication 역시 비슷하게 하면 되지 않을까? 라는 생각을 할 수 있습니다. NTT등을 사용하면 속도도 빠르게 나올 것입니다. 하지만, 그냥 곱하는 방법을 쓰면 문제가 있는데, 바로 Noise Growth가 너무 크다는 사실입니다.

BFV scheme에서는 encryption등의 과정에서 ciphertext의 security를 위해서, error polynomial을 더합니다. (error polynomial을 더하지 않으면 풀기 쉬워집니다.) 그리고 decryption을 할 때 에러에 해당하는 부분은 1보다 작게 해서 rounding 등을 통해 error를 날려버리고, 원래 결과를 얻게 됩니다.

그런데 연산들을 거듭하면 거듭할 수록, 이 error polynomial에 해당하는 값이 점점 커지게 되기 때문에, Noise Growth가 계속되 특정 범위를 초과하게 되면, 원래 poloynomial이 어떤 polynomial이었는지 알 수 없게 될 수도 있습니다.

그렇기 때문에, BFV scheme을 비롯한 scheme들에서는 이런 Noise 관리가 굉장히 중요한 이슈입니다.

다시 원래로 돌아가서, 그냥 곱하는 방법을 쓰면 Noise가 너무 커집니다. 그래서 radix base $\omega$에 대한 $\omega$진법을 통해서 쪼개는 등의 노력을 하게 됩니다.

위의 방법을 쓸 경우 몇 가지 단점이 있습니다.  또, 나중에 설명하겠지만, decryption 시 사용했던 gamma-correction을 적용하기 힘들 기 때문에, 다른 rounding correction을 써야할 필요도 있습니다. BEHZ scheme에서는 앞으로 소개할 step을 따라서 multiplication을 수행합니다.

# Step 1.

먼저, multiplication을 정상적으로 수행하기 위해서, Base의 크기를 늘릴 필요가 있습니다. 왜 Base의 크기를 늘려야 하는지는, 다음 예시를 보면 알 수 있습니다.

$a_0+a_{1}x+a_{2}x^2+\dots$와, $b_0+b_{1}x+b_{2}x^2+\dots$를 곱한 결과인 $c_0+c_{1}x+c_{2}x^2+\dots$를 계산한다고 합시다.
그럼, $c_k$는 다음과 같이 표현됩니다. $c_k = a_{0}b_{k}+a_{1}b_{k-1}+\dots+a_{k-1}b_{1}+a_{k}b_{0}$ 이때, 계수들을 곱하고, 그것들을 $k$번 더한 것의 결과를 정확히 저장해야 합니다. 그렇기 때문에, multiplication을 할 경우에는 Base의 확장이 필요합니다.

## Auxilary Base

위에서 설명한 것처럼, 확장을 위한 base가 필요한데, 이를 auxilary base라고 합니다. 기존의 base인 $q$를 확장시킨, base $B$가 있을 때, 또 다른 modulus $m_{sk}$를 합쳐서 $B_{sk} = B \cup \{ m_{sk} \}$로 정의합시다. 그리고 또 다른 modulus $\tilde{m}$이 있다고 하면, $q$에서 $B_{sk} \cup \{ \tilde{m} \}$로 확장시키게 됩니다.

이때, 확장 과정은 지난 글에서 설명했던 Fast RNS Base conversion 함수를 그대로 사용합니다.

## Small Montgomery Reduction

하지만 저번 글에서 설명했다시피, Fast RNS Base Conversion을 실행할 경우, 현재 Base의 확장이 일어나고 있기 때문에 redundant한 $q$의 배수가 더해질 수 있습니다. (즉, $q-\text{overflow}$가 일어날 가능성이 있습니다.)

그렇기 때문에, 여기에서는 Small Montgomery Reduction을 통해 redundant한 $q$의 배수를 지워주게됩니다.

Small Montgomery Reduction은 다음과 같습니다.

주어진 $B_{sk} \cup \{ \tilde{m} \}$의 ciphertext $c$에 대해, 

$$ r_{\tilde{m}} \leftarrow [-c_{\tilde{m}}/q]_{\tilde{m}} $$
$$ c'_{m} \leftarrow \lvert (c_{m} + qr_{\tilde{m}})\tilde{m}^{-1} \rvert _{m} \ \text{for} \ m \ \in \ B_{sk} $$

위의 $c'_m$는 각 modulus $m$에 대한 polynomial이고, 각 $c'_m$을 모두 모으면 전체 modular base에 대한 polynomial을 구성할 수 있습니다. 위의 logic이 정당하다는 것은, (정확히 말해서, $q-\text{overflow}$를 잘 제거한다는 것은) BEHZ paper의 Lemma 4에 의해서 보장됩니다.

Lemma 4의 경우 내용이 복잡하기 때문에, 여기에서는 생략하도록 하겠습니다.

# Step 2.

이제 적절하게 base를 확장했으니, 이제는 실제로 곱할 차례 입니다. 그런데, BFV에서 encryption을 할 때, $\Delta = \lfloor q/t \rfloor$를 원래 메시지 $m$에 곱했다는 것을 생각해봅시다. 따라서, 곱한 후에 이 $\Delta$를 적절한 방식으로 나누어줘야 할 필요가 있습니다. 그렇게 하지 않으면 대략적으로 $m$에 $\Delta$의 제곱인 $\Delta^2$이 곱해진 형태가 될 것입니다.

현재 modular base의 집합은 $q \cup B_sk \cup \{ m_{sk} \}$이고, 이 base의 modulus 각각에 대해서, 곱한 결과를 구해봅시다.

우선, Multiplication의 결과 $ct$를 구한다고 했을 때, $ct[0], ct[1], ct[2]$는 다음과 같이 정해집니다. (아직 Relinearize를 하직 않았으므로)

ciphertext $ct1, ct2$를 곱한다고 했을 때, (둘 다 $q \cup B_sk \cup \{ m_{sk} \}$로 확장을 마친 상태입니다.)

$ct = ct1 \cdot ct2$라고 하면, 

$ct[0] = ct1[0] \cdot ct2[0]$이고,
$ct[1] = ct1[1] \cdot ct2[0]+ct1[0] \cdot ct2[1]$이 되고,
$ct[2] = ct1[1] \cdot ct2[1]$이 됩니다.

위의 식이 왜 성립하는지 모르겠다고 하시는 분은 BFV scheme의 연산을 다루는 이전 [글](https://infossm.github.io/blog/2023/01/17/BFV-Logic-part1/)을 참고하시기 바랍니다.

각 modulus $m$에 대해, 위 식을 그대로 계산하기만 하면 됩니다. 예를 들어, $ct[0]$을 $m$으로 나눈 나머지인 $ct[0]_m$을 구하려면, $ct[0]_m = ct1[0]_m \cdot ct2[0]_m$을 계산하면 될 것입니다.

# Step 3.

이제 decryption에서처럼, $t/q$를 곱한 결과를 적절히 계산해줄 필요가 있습니다.

하지만 안타깝게도 decryption에서 썼던 gamma-correction을 여기에서 그대로 쓸 수는 없습니다. 이유는 gamma-correction을 쓸 수 있는 조건이 만족된다는 보장을 할 수 없기 때문입니다.

대신에, multiplication에서는 다음과 같은 단계를 통해 결과를 구하게 됩니다.

## Fast RNS floor

먼저 floor 연산을 약간 부정확하더라도 빠르게 구할 방법이 필요합니다. 우리는 이미 Fast Base Conversion을 알고 있으니, 이 함수를 활용하는 방향으로 고민을 해봅시다.

어떤 modulus $m$에 대한 floor를 구한다고 하면, 다음과 같은 식을 구성할 수 있습니다.

$$ FastRNSFloor_q(\textbf{a}, m) := (\textbf{a} - FastBConv(\lvert \textbf{a} \rvert _q, q, m)) \times \lvert q^{-1} \rvert _m $$

직관적으로, $\textbf{a}$의 $m$으로 대한 표현으로 옮긴다음, 나머지를 뺀 후, $q$로 나누어줬다고 볼 수 있습니다.

이제 Step 2의 결과를 $ct$라고 했을 때, 

$$ FastRNSFloor_q(t \cdot ct[j], B_{sk})  = \lfloor \frac{t}{q}ct[j] \rceil + \textbf{b}_{j} \ \text{for} \ j \ = 0, \ 1, \ 2 $$

이렇게 하면 '대략적인' 결과를 알 수 있습니다.

여기에 correction 작업을 더 하면 원하는 결과를 얻을 수 있게 됩니다.

# Conclusion

이번 글에서는 제 개인 사정상 multiplication의 일부밖에 소개하지 못했습니다. 다음 글에서는 relinearize 과정까지 포함한, multiplication 전체 과정을 마무리하도록 하겠습니다.

부족한 글 읽어주셔서 감사합니다.
