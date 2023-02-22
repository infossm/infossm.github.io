---
layout: post
title:  "BFV scheme에 대한 소개 - 2"
date:   2023-02-14 09:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

저번 [글](https://infossm.github.io/blog/2023/01/17/BFV-Logic-part1/)에서 BFV scheme에 대해서 소개하는 글을 썼습니다. 이 글에선 저번 글에서 사용한 notation들을 따와서 설명을 할 것이기 때문에, 이전 글을 다시 한번 읽고 오시면 이해가 더 쉬울 것입니다.

이번 글에서는 BFV scheme의 실제적인 구현을 위한 알고리즘들을 좀 더 설명하는 글을 쓰도록 하겠습니다. 이 글에서 사용한 Microsoft SEAL의 코드 구현은 이 글이 쓰여진 시점에서 최신 version의 SEAL(206648d)의 구현을 따릅니다.

# RNS representation

## Definition

본격적으로 설명하기에 앞서, RNS representation에 대해 먼저 설명을 하도록 하겠습니다.

RNS(Residue Number System)이란, 간단히 말해서 어떤 큰 modular system에 속한 정수를, 여러 개의 작은 서로 서로소인 moduli에 대한 나머지들로 표현하는 것을 의미합니다. CRT(Chinese Remainder Theorm)에 의해 가능합니다. PS 문제에서도 가끔씩 사용하는 알고리즘이기 때문에, 이미 익숙하신 분들도 많을 것 같습니다.

예를 들어서 어떤 수를 $30$에 대한 나머지 값을 저장한다고 그 수를 $2$로 나눈 나머지, $3$으로 나눈 나머지, $5$로 나눈 나머지들만 저장하고 있어도 원래 정수를 복원할 수 있습니다.

좀 더 formal 하게 표현하면, $\mathbb{Z}_n = \mathbb{Z} / n \mathbb{Z}$라고 두고,
$q \ = \ q_{1}q_{2} \cdots q_{m}$이라고 할 때, 다음과 같다고 할 수 있습니다.

$$\mathbb{Z}_{q} \simeq \mathbb{Z}_{q_{1}} \times \mathbb{Z}_{q_{2}} \cdots \mathbb{Z}_{q_{m}}$$

위 식이 성립하기 때문에, 즉, isomorphic하기 때문에 표현이 가능하다고 할 수 있습니다. CRT와 RNS에 대한 더 자세한 설명은 [이글](https://m.blog.naver.com/kks227/221635322468)이나 [이글](https://rkm0959.tistory.com/180) 등의 좋은 자료가 많기 때문에, 구글링을 하시면 될 것 같습니다.

요약하자면, 아주 큰 수에 대한 modular를 표현할 때 moduli들로 쪼개서 저장하기 쉽게 변환할 수 있다는 점이 요지가 될 것입니다.

## Operation with RNS representation

어떤 큰 수를 RNS representation으로 변환했다고 할 때, 그 변환한 상황에서 덧셈, 뺄셈, 곱셈은 어떻게 하면 될까요? 요소들끼리 계산해주면 됩니다. 예를 들어서 덧셈에 대한 증명을 생각해봅시다.

$q \ = \ q_{1}q_{2} \cdots q_{m}$에 대해, $c \equiv a + b \ (mod \ q)$이고, $a,b,c$의 RNS representation이 각각 $(a_{1}, a_{2}, \dots , a_{m})$, $(b_{1}, b_{2}, \dots , b_{m})$, $(c_{1}, c_{2}, \dots , c_{m})$라고 합시다.

$c \equiv a + b (mod \ q)$이므로, $1 \le i \le m$에 대해 $c \equiv a + b (mod \ q_{i})$이고 $c \equiv c_{i}(mod \ q_{i})$이고, $a+b \equiv a_{i} + b_{i} (mod \ q_{i})$ 이므로, $c_{i} \equiv a_{i} + b_{i} (mod \ q_{i})$입니다.

뺄셈, 곱셈도 비슷한 방식으로 증명에 성공할 수 있습니다.

그리고 이로부터 다항식들을 더하고, 빼고, 곱하기만 한다면 여전히 RNS를 적용해 각각 계산한 뒤 결과를 합쳐도 원래 수식과 결과가 동일할 것임도 쉽게 증명할 수 있습니다.

자세한 증명은 충분히 위의 덧셈에 대한 증명과 비슷하게 설명할 수 있기 때문에, 여기에선 생략하도록 하겠습니다.

## Why use RNS?

RNS representation이 대충 어떤 원리로 변환되는지 알아 보았습니다. 이제 굳이 RNS represenation을 설명한 이유에 대해 설명할 것입니다.

BFV scheme에서 ciphertext의 경우 $R_q$에 속한 다항식 2개를 저장해야 합니다. 그런데 security level 이슈 상 $q$의 크기는 충분히 커야 합니다. 여기에서 충분히 크다는 말은, 수백 bit 이상의 정수를 의미합니다.

그런데 보통 C++ 기준으로 long long이 64-bit 정수라는 것을 생각하면, 수백 bit 이상의 정수들을 계수로 가지는 다항식을 다루려면 무리가 있어 보입니다. 물론 Biginteger를 다루는 언어들도 존재하고, 구현하려면 구현할 수 있겠지만, 속도가 매우 느려질 것임은 분명합니다.

이 문제점은 비단 BFV scheme에만 해당하는 것은 아닙니다. CKKS등의 scheme들도 비슷한 문제를 안고 있습니다. 그럼 이를 어떻게 해결할 수 있을까요?

예상하셨겠지만, 위에서 설명한 RNS representation을 사용하는 방법이 있습니다. BFV scheme의 경우, plaintext modulus와 ciphertext modulus에 해당하는 $t, q$에 대한 제한이 없습니다. ($t \lll q$라는 조건만 빼면) 그래서 modulus를 설정할 때, $60$bit 정도 크기의 prime들 몇 개의 곱으로 modulus를 설정하면, 각 prime들의 나머지는 C++ 기준으로 long long 범위의 변수에도 충분히 저장할 수 있으니까, 좀 더 다루기 편리해집니다. 수백 bit 짜리 정수를 저장하는 type을 따로 만들지 않아도 되는 것이죠.

위에서 보듯이 RNS form으로 표현할 경우 좀 더 실제 구현이 쉬워지는 경향이 있기 때문에, 어떤 scheme이 개발되고, 그 scheme의 RNS version이 후속 연구로 나오는 경우가 많습니다.

# BFV scheme with RNS

이제 RNS representation을 사용한 BFV scheme에 대해서 설명하겠습니다. 먼저 설명하기 쉬운 연산들부터 설명하도록 하겠습니다.

## Encoding, Decoding

encode, decode의 경우 달라지는 것은 없습니다. encode, decode의 경우 저장하는 정보와 $R_t$의 다항식간의 변화기 때문입니다. 그리고 $t$의 경우 적당히 작은 크기의 소수를 선택하는 것이 대부분이라서 (즉, C++에서 long long 안에 들어올 수준) RNS를 적용할 필요가 없는 경우가 대부분입니다.

## Encryption

encryption의 경우, 먼저 encryption을 수행하는 수식을 살펴 봅시다.

$$C_{0} = [pk_{0} \cdot u + e_{0} + \Delta M]_{q} \\ C_{1} = [pk_{1} \cdot u + e_{1}]_{q}$$

위 수식을 보면, $M$은 결국 $R_t$의 원소인 어떤 메시지에 해당하고, $u, e_{0}, e_{1}$야 추출한 것입니다. $pk_{0}, pk_{1}$ 역시 다항식입니다.

현재 수식 상에서 다항식을 곱하고, 더하는 연산들밖에 없기 때문에, 위의 절에서 설명한 것처럼, RNS를 적용할 수 있습니다. 여기서 직관적으로 이해가 가지 않을 수 있는 부분은 $\Delta$가 $q/t$이므로, 아주 큰 수이기 때문에 RNS를 어떻게 적용할지 잘 와닿지 않을 수 있습니다.

여기서 생각할 수 있는 것은, 결국 한번 $t, q$ 같은 parameter가 정해지면 적어도 encryption, addition, multiplication, decryption 등이 다 끝날 때 까지는 같은 parameter를 계속 사용할 것입니다. 따라서, parameter가 정해지면 그때부터 $\Delta$는 상수니까, 미리 전처리를 해서 계산을 해놓는 방식을 생각할 수 있습니다.

대표적인 FHE library인 SEAL의 해당부분 구현을 보면, Biginteger를 다루는 구조에서 $q/t$에 해당하는 값을 계산한 뒤, 그 뒤에 각 modlui의 나머지에 해당하는 값을 저장하는 식으로 구현이 돼있습니다.

[링크](https://github.com/microsoft/SEAL/blob/main/native/src/seal/context.cpp#L319)에서 실제로 그렇게 구현되고 있음을 확인할 수 있습니다.

[링크](https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/scalingvariant.cpp#L77)에서 위의 값을 호출하는 예를 볼 수 있습니다.

위의 $\Delta$부분을 제외하면 나머지 부분은 그저 다항식을 곱하고, 더하는 것 뿐이니 비교적 자연스럽게 RNS form을 적용할 수 있습니다.

덧셈, 곱셈을 어떻게 효율적으로 처리하면 될지 고민하면 되지요. 덧셈이야 차수가 같은 것끼리 맞춰서 각각 더한 뒤 결과를 내면 된다고 됩니다. 곱셈은 어떻게 할까요? 보통 FHE에서 다루는 다항식은 차수가 $100000$을 넘는 경우도 많기 때문에, 단순히 나이브하게 $O(n^2)$로 계산하면 아마 굉장히 느려질 것입니다.

여기서 쓸 수 있는 방법은, 많은 분들이 짐작하셨을 테지만 batchencode에서도 활용한 바 있는 NTT를 활용하는 것입니다. NTT를 활용하면 $O(nlogn)$에 계산을 마칠 수 있겠지요. 카라츠바 알고리즘같은 것을 활용할 수도 있겠으나, 아무래도 속도가 중요하다 보니, moduli들을 설정할 때 보통 각각의 moduli에 NTT를 활용할 수 있게 설정하는 것이 보통입니다.

NTT의 경우 encryption 뿐만 아니라 FHE scheme 상에서 다항식의 곱셈을 계산할 때 일반적으로 사용됩니다.

때문에 NTT를 빠르게 계산할 수록 FHE scheme 연산들의 전체 속도가 빨라지는 경향이 있어, NTT의 최적화는 굉장히 많은 사람들이 관심을 가지는 주제입니다. 이에 대한 내용은 나중에 쓸 기회가 있으면 쓰도록 하겠습니다.

## Addition

addition의 경우 encryption 보다 더 간단합니다. 수식을 봅시다.

$$A = A_{0} + A_{1} \\ B = B_{0} + B_{1}$$

그냥 각각 더하는 것이 전부입니다. 따라서 각 moduli에 대해서 그냥 덧셈을 진행해주면 끝입니다.

## Complex Operations

위에서 언급한 연산들은 사실 몇 가지 처리만 해주면 RNS를 적용하는게 어렵지 않은 연산들입니다. 실제로 encryption, addition의 경우 수식이 덧셈, 곱셈으로만 구성돼있습니다.

하지만 BFV scheme에는 RNS presentation을 바로 적용하기 어려운 연산들이 존재합니다. decryption, multiplication이 그렇습니다.

decryption의 경우 수식에 나눗셈이 등장하기 때문에, 바로 RNS representation을 적용하기 어렵습니다. multiplication의 경우 RNS presentation을 그대로 적용할 경우 중간과정에서 문제가 발생할 수 있습니다.

그렇기 때문에, 효율적인 RNS version의 BFV scheme을 작성하는 것은 사실 저 두 가지 연산을 어떻게 다룰 것인가 하는 문제와 직결됩니다. 저 문제를 해결하는 것이 굉장히 어려운 편이기 때문에, 실제로 BFV는 다른 scheme들보다 RNS version의 scheme이 나오는 것이 다른 scheme (BGV, CKKS)보다 상대적으로 느렸습니다.

BFV를 최초로 제안한 [paper](https://eprint.iacr.org/2012/144)이후, 여러 가지 scheme들이 나왔습니다만, 가장 널리 알려진 것은 BEHZ로 널리 알려진 이 [paper](https://eprint.iacr.org/2016/510)에서 제안된 것과, HPS로 알려진 이 [paper](https://eprint.iacr.org/2018/117)에서 제안된 것이 가장 유명합니다.

공통적으로 둘 다 처리가 복잡하다는 공통점이 있기 때문에, 이 글에서 서술하지 못한 것은 다음 글에 이어서 계속 설명하도록 하겠습니다.

# Reference

- https://github.com/microsoft/SEAL
- https://en.wikipedia.org/wiki/Residue_number_system
