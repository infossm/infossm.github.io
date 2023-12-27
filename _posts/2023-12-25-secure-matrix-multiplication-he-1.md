---
layout: post
title:  "Secure Matrix Multiplication with Homomorphic Encryption - 1"
date:   2023-12-25 19:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

동형암호 (Homomorphic Encryption)은 암호문간의 연산을 평문의 정보유출 없이 가능하게 하는 암호 scheme입니다. 간단하게 말해서, 어떤 두 plaintext $m_0, m_1$을 encrypt한 것이 $ct_0, ct_1$이라고 할 때, $dec(ct_0+ct_1) = m_0 + m_1$이 성립합니다. 동형암호 scheme 중에서도, 두 가지 연산, 즉 addition과 multiplication을 제한 없이 사용할 수 있다면 그런 동형암호 scheme을 Fully Homomorphic Encryption, 줄여서 FHE라고 부릅니다. 현재 나와 있는 대부분의 동형암호 scheme의 경우, RLWE problem의 Hardness에 의존하고 있습니다.

대표적인 FHE scheme으로는 BGV, BFV, CKKS, TFHE 등이 있습니다.

동형암호의 경우 그 특성상 Clould computing등의 환경에 적용하는 것이 용이하기 때문에, 이와 관련해서 많은 연구가 이루어지고 있습니다. 그 중에서도 가장 관심을 많이 받는 연구 중 하나는 Matrix Multiplication과 HE를 결합하는 것일 것입니다.

HE가 적용된 상태로 Matrix Multiplication을 할 수 있다는 것은, 안전성과 HE의 장점을 가진 상태로 Matrix Multiplication이 활용되는 다양한 Application을 사용할 수 있다는 것과 같기 때문입니다. 가장 대표적인 Application으로는 역시 Machine Learning이 있습니다. Machine Learning이 주요한 연산들이 Matrix Multiplication으로 표현할 수 있다는 것은 널리 알려진 사실입니다.

이 글에서는 HE를 Matrix Multiplication에 적용하여, Secure하게 Matrix Multiplication을 수행할 수 있는 방법에 대해 설명합니다.

# Backgrounds

## Operations of CKKS

CKKS scheme의 경우, BFV와 Polynomial Ring $R_q$에 속한 다항식의 쌍으로 Ciphertext가 표현됩니다.

두 Ciphertext $ct_a, ct_b$에 대해, $ct_a = (ct_{a,0}, ct_{a,1}), ct_b = (ct_{b,0}, ct_{b,1})$이면, $ct_a + ct_b = (ct_{a,0}+ct_{b,0}, ct_{a,1}+ct_{b,1})$이 됩니다.

Multiplication의 경우 조금 더 복잡한데, BFV 처럼 3개의 항으로 늘어나고 Relinearization을 통해서 다시 줄어드는 것은 같으나, 세부적인 과정에서 차이가 있고, Rescale이란 과정을 거치게 됩니다.

Encryptoin, Decryption의 경우 BFV와 별로 차이가 없습니다. 예를 들어 Decryption의 경우 secret key를 $s$라 할 때, $ct = (ct_0, ct_1)$에 대해 $m = ct_0+ct_1 \cdot s$입니다.

그 외에 더 자세한 설명은 제안된 [논문](https://eprint.iacr.org/2016/421.pdf)이나 구글링을 하시면 좋은 자료가 많습니다. BFV의 경우 블로그에 제가 쓴 글들이 있습니다.

## Level HE

일부 FHE scheme들의 경우, Ciphertext에 level이 존재합니다. 대표적인 scheme으로는 CKKS의 경우가 있습니다.

CKKS의 경우를 바탕으로 예를 들면, Multiplication을 할 때 마다 level이 하나씩 줄어들게 됩니다.

CKKS에서 Ciphertext는 다항식의 쌍으로 표현됩니다. 그리고 그 다항식의 계수 크기는 parameter에 따라 정해지는데, 일반적으로 매우 큰 수를 선택합니다. BigInteger 같은 것을 쓰면 performance가 떨어질 수 밖에 없기 때문에, 일반적으로는 상한은 적당히 큰 소수들의 곱으로, 계수는 그 소수들에 대한 나머지 쌍으로 equivalent하게 표현할 수 있도록 설정합니다. 이를 Residue Number System, RNS라고 합니다.

이 RNS system을 구성하는 소수들의 개수를 level이라고 합니다. CKKS scheme 에서 Multiplication을 수행할 때, rescale이란 과정을 거치게 됩니다. rescale은 context에 따라 결정된 수 scaling factor를 나눠주는 것과 대응됩니다. scaling factor를 나눠줄 때, RNS system을 구성하는 소수 하나가 빠지게 됩니다. 따라서, Multiplication연산을 1번 수행할 때마다 rescale연산도 같이 수행됩니다. 따라서, Multiplication이 1회 실행될 때마다, level이 1씩 줄어들게 됩니다. 일반적으로 쓰는 parameter의 경우 level의 크기가 아주 크지 않기 때문에, fresh ciphertext의 경우 다른 조작을 하지 않는다고 할 때 수행 가능한 Circuit의 최대 depth가 굉장히 한정됩니다.

이렇다면 FHE란 이름이 붙을 수 없을 것입니다. 이를 해결하기 위한 연산이 바로 Bootrstrapping 입니다.

## Bootstrapping

일반적으로 HE scheme들이 가지는 공통점이 있다면, 바로 연산을 반복하면 반복할 수록 error가 늘어난다는 점입니다. addition을 예로 들어봅시다.

secret key가 $s$이고, 두 Ciphertext $ct_a, ct_b$에 대해, $ct_a = (ct_{a,0}, ct_{a,1}), ct_b = (ct_{b,0}, ct_{b,1})$라고 합시다.

이때, $ct_{a,0} + s \cdot ct_{a,1} = m_{a} + e_{a}, ct_{b,0} + s \cdot ct_{b,1} = m_{b} + e_{b}$라고 합시다.

그럼, $ct_a + ct_b = (ct_{a,0}+ct_{b,0}, ct_{a,1}+ct_{b,1})$이고, 이때,

$$dec(ct_a + ct_b) = m_{a} + m_{b} + e_{a} + e_{b}$$

가 되므로, error 항이 늘어나는 것을 확인할 수 있습니다.

사실 Addition의 경우 error가 늘어나는 것이 크게 문제되지는 않습니다. Multiplication의 경우, Addition에 비해 훨씬 더 큰 폭으로 증가합니다.

이렇게 연산들이 반복될 때마다 error가 늘어나고, 결국에는 제대로 Decryption을 할 수 없을 정도로 커지게 되는데, 이렇게 되면 연산, 특히 Multiplication의 횟수에 제한이 생깁니다. 이래서야 FHE라고 할 수는 없습니다. 따라서, 이를 해결하기 위해 error의 크기를 다시 낮추는 연산이 필요한데, 이를 Bootstrapping이라고 합니다.

Bootstrapping이 등장하면서 FHE가 등장했고, 그 후 FHE 연구가 가속화됩니다.

# Naive Approach







# Column Order method





