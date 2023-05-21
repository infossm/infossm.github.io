---
layout: post
title:  "BFV scheme에 대한 소개 - 5"
date:   2023-05-21 09:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

지금까지 BFV Scheme에 관한 설명글들을 썼고, 저번 [글](https://infossm.github.io/blog/2023/04/23/BFV-Logic-part4/)에서는 RNS form에서 처리하기 까다로운 연산 중에서 BFV Multiplication을 어떻게 처리하는지에 대해서 설명을 했습니다. 이번 글에서는 저번 글에서 다 하지 못한 설명을 마무리하도록 하겠습니다.

# Backgrounds

이번 글은 기본적으로 저번 글에서 설명한 내용을 이어서 하는 것이기 때문에, 따로 Background는 필요하지 않으나, 저번 [글](https://infossm.github.io/blog/2023/04/23/BFV-Logic-part4/)의 내용을 이해하는 것이 중요합니다.

특히 FastBConv 함수의 정의에 대해서 잘 기억해 주세요.

# Step 4.

앞의 step 3의 결과를 다시 불러와 봅시다.

$$ FastRNSFloor_q(t \cdot ct[j], B_{sk})  = \lfloor \frac{t}{q}ct[j] \rceil + \textbf{b}_{j} \ \text{for} \ j \ = 0, \ 1, \ 2 $$

위와 같이 결과가 나온다고 했을 때, 앞으로 편의상 $ct_{mult}[j] = \lfloor \frac{t}{q}ct[j] \rceil + \textbf{b}_{j}$로 표시합시다.

이때, $ct_{mult}$의 경우 $q$가 아니라 $B_{sk}$에 있습니다. 따라서, $q$로 옮기는 과정이 필요합니다.

# Step 5.

그런데 저번 [글](https://infossm.github.io/blog/2023/04/23/BFV-Logic-part4/)에서 언급했던 것처럼, Step 1에서 썼던 gamma-correction을 다시 쓸 수는 없습니다.

대신에, $B_{sk}$를 만들 때 미리 추가해두었던 $m_{sk}$의 성질을 이용합니다.
Shenoy and Kumaresan like conversion 를 쓸 수 있습니다.

논문의 Lemma 6.에 의해서, 다음과 같은 함수 $FastBconvSK$를 정의합시다.

$M = \prod _{m \in B} m$

$\alpha_{sk, x} = [(FastBConv(x, B, \{ m_{sk} \})-x_{sk})M^{-1}]_{m_sk}$

$FastBconvSK(x, B_{sk}, q) = (FastBConv(x, B, q)- \alpha_{sk, x}M)$

위의 Correction이 어떻게 성립하는지는 [논문](https://ieeexplore.ieee.org/document/16508)을 참고하시기 바랍니다.

step 5까지의 과정을 마치고 나면, 

$ct = ct1 \cdot ct2$라고 했을 때, $ct[0]+ct[1] \dot s+ ct[2] \dot s^2 \approx (ct1[0]+ct1[1] \dot s)(ct2[0]+ct2[1] \dot s)$를 만족하는 $ct$를 구할 수 있습니다.

저번 글들을 읽으신 분들이라면 알겠지만, 이제 Relinearize라고 불리는 마지막 과정이 남았습니다.

# Step 6.

이 Relinearize는 기존 Relinearize와 다르지 않습니다. 먼저 RNS가 아닐 경우에는 radix base $w$를 도입해서 Noise를 줄여줬다면, RNS form에서는 RNS form을 그대로 활용하면 됩니다. 

좀 더 엄밀하게는, $\overline c_2 = ct[2]$라고 합시다. 다음과 같이

$\xi _q (\overline c_2)$와 $P_{RNS, q} (s^2)$를 정의합시다. 

$\xi _q (\overline c_2) = (\lvert \overline c_2 \frac{q_1}{q} \rvert _{q_1}, \lvert \overline c_2 \frac{q_2}{q} \rvert _{q_2}, \dots , \lvert \overline c_2 \frac{q_k}{q} \rvert _{q_k})$

$P_{RNS, q} (s^2) = (\lvert \overline s^2 \frac{q}{q_1} \rvert _{q_1}, \lvert \overline s^2 \frac{q}{q_2} \rvert _{q_2}, \dots , \lvert \overline s^2 \frac{q}{q_k} \rvert _{q_k})$

이때, 임의의 $c$에 대해서, 

$\langle \xi_q (c), P_{RNS, q} (s^2)\rangle \equiv cs^2 $가 성립함을 이용합니다.

위의 $\overline c_2$와 비슷하게 $\overline c_0, \overline c_1$을 정의합시다.

$ct_{ans}[0] = \overline c_0 + \langle \xi_q (c), P_{RNS, q} (s^2) - (\overrightarrow e + \overrightarrow a s)\rangle$

$ct_{ans}[1] = \overline c_1 + \langle \xi_q (c), \overrightarrow a \rangle$

라고 하면, $ct_{ans}$가 우리가 원하는 답이 됩니다.

# Conclusion

## Summary

저번글과 합해서 총 6~7개 정도의 과정을 거쳐 RNS form에서 Multiplication을 계산할 수 있게 됩니다. 제가 지금까지 소개한 것은 BEHZ algorithm으로 더 잘 알려져 있습니다.

과정을 다시 한번 간략하게 요약하자면 다음과 같습니다.

- Auxillary base로 확장한다.
- gamma-correction을 통해 q-overflow를 제거한다.
- RNS form에서 각 element-wise 하게 곱한다.
- FastRNSFloor 함수를 사용해 $t/q$를 곱한 곳을 구한다.
- Shenoy and Kumaresan like conversion 를 이용해 $q$로 옮긴다.
- Relinearize를 한다.

## gadget decomposition

RNS form이 없을 때에서 radix base $w$를 정해서 분해하여 Noise를 낮췄는데, RNS form도 기본적으로 각 자리마다 다른 base를 사용하는 진법 처럼 이해할 수 있기 때문에, RNS form에 대해 관리하면 본질적으로 radix base $w$를 설정하는 것과 같은 효과를 얻을 수 있습니다. 그리고 $w$를 따로 신경 써도 되지 않기 때문에 더 효율적입니다.

이렇게 분해(?)하는 것과 관련해서 더 일반적인 개념으로 gadget decomposition이라는 것이 있는데, 이는 나중에 자세히 설명할 도록 하겠습니다.

## Relinearize

Relinearize는 결국 선형모양으로 돌려주기 위해 $s^2$를 $s$로 encrypt한 것을 사용하는 연산이라고 할 수 있습니다. 그런데, 이게 정말 안전할까요? FV scheme은 기본적으로 RLWE problem 의 hardness를 기반으로 안전성을 보장합니다. 그런데 $s^2$를 $s$로 encrypt하면, 식의 형태가 달라져서, 안전하다고 확실하게 보장할 수 없게 됩니다. "안전할 것이다"라고 예상할 뿐입니다.

그렇기 때문에, 대표적인 HE Library 중 하나인 SEAL에서는 Reliearize를 하지 않고, Ciphertext가 다항식 2개가 아니라 $k \ge 2$개를 들고 있는 식으로 저장할 수 있게 해서, Relinearize 적용하지 않아도 되게 합니다.

다만, 일반적으로 Relinearize를 적용시키는 것이 속도 등 여러 측면에서 훨씬 효율적이기 때문에, 대개 Relinearize를 항상 적용합니다. 그리고 SEAL의 Relinearize는 $k = 3$인 경우에만 Relinearize가 잘 동작하도록 설계 됐습니다.

BFV scheme에 대한 글은 아마 이 글이 마지막이 될 것 같습니다.

BEHZ 말고도 HPS라는 다른 scheme이 존재하긴 합니다만, 따로 글을 쓰지는 않을 것 같습니다.

부족한 글 읽어주셔서 감사합니다.


# Reference

- https://eprint.iacr.org/2016/510
