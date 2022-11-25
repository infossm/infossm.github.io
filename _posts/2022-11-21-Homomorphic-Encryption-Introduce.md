---
layout: post
title:  "Homomorphic Encryption에 대한 소개"
date:   2022-11-22 23:00:00
author: cs71107
tags: [cryptography]
---

# Introduction

올 한해 [크립토랩](https://www.cryptolab.co.kr/)이라는 스타트업에서 근무할 기회가 있었습니다. 크립토랩은 Homomorphic Encryption, 즉 동형암호 scheme 중 CKKS를 바탕으로 창업한 스타트업입니다. (CKKS scheme을 제안하신 천정희 교수님이 대표입니다.) 회사에서 근무하며 그동안 잘 몰랐던 동형암호에 대해서 배울 기회가 있었습니다. 현재 국내에는 동형암호에 대해 서술하는 자료가 그리 많지 않은 것으로 보여, 동형암호에 대한 소개글을 작성하기로 했습니다.

# What is Homomophic Encryption?

먼저 동형암호가 무엇인지 간단하게 설명하도록 하겠습니다.
우선 보통의 암호 scheme의 경우 평문(plaintext)와 암호문(ciphertext)의 관계를 최대한 random하게 만드려고 합니다. 평문과 암호문 사이 관계에 어떤 규칙이 있다면, 그 규칙을 통해 공격하기 쉽기 때문입니다.
그렇기 때문에, 만약 보통의 scheme에서 암호문 끼리 더하거나(add), 곱하거나(mult)등의 연산을 한다면, 위의 특성에 의해서 해당 결과를 복호화한다고 해도, 원래 plaintext와는 동떨어진 결과를 얻게 될 것입니다. 그래서 만약에 암호화해서 저장해둔 어떤 data들에 대해서 더하거나, 곱하거나 하는 등 연산을 적용하고자 한다면, 암호화된 data를 다시 복호화한 다음에, 연산을 적용하고, 다시 암호화해서 저장해야 할 것입니다.
이런 단점을 해결할 수 있는 암호 scheme이 Homomorphic Encryption, 즉 동형암호입니다.

동형암호란, 간단하게 설명하면 복호화 없이 암호화된 data들 끼리 연산을 해도 무방한 scheme들을 말합니다. 예를 들어서, 어떤 scheme에서 평문과 암호문에서 더하기가 정의된다고 합시다. 이 scheme에서 임의의 평문 $m_{1}, m_{2}$에 대응되는 암호문이 $ct_{1}, ct_{2}$라고 합시다. 즉 $enc(m_{1}) = ct_{1}, enc(m_{2}) = ct_{2}$이고, $dec(ct_{1}) = m_{1}, dec(ct_{2}) = m_{2}$입니다.
이 scheme이 Homomorphic Encryption이라면, $dec(enc(m_{1})+enc(m_{2})) = m_{1}+m_{2}$를 만족합니다. 더 나아가서, 암호문 상태에서 아무리 더해도 나중에 복호화 한번만 해주면 더해진 암호문들의 평문들의 합과 같은 값이 나오게 될 것 입니다.

동형암호의 종류는 다음과 같이 크게 4가지가 있습니다. 첫 번째로, 위와 같이 homomorphic한 연산이 하나 있는 scheme의 경우 Partial homomorphic encryption 이라고 합니다. 두 번째로, homomorphic한 연산이 두 가지 존재하나, 모든 circuit이 아니라 일부에만 homomorphic하면 Somewhat homomophic encryption 이라고 합니다. 세 번째로, homomorphic한 연산이 두 가지 존재하고, 일정 depth이하의 모든 연산 circuit에 대해 homomorphic할 때 Leveled fully homomorphic encryption 이라고 합니다. 마지막 네 번째로, 두 가지 homomorphic 연산이 존재하고, 모든 circuit에 대해서 성립할 때, Fully homomorphic encryption 이라고 합니다. 나중에 소개할 BFV, CKKS 등 대부분의 HE Scheme들이 Fully homomorphic encryption입니다. 자주 쓰이는 대부분의 scheme이 Fully homomorphic encryption입니다. 단, 첫번째에서 마지막으로 갈 수록 construction 난이도가 상승하기 때문에, 각 종류의 scheme들은 block으로서 중요합니다.

앞서 설명했듯이, 동형암호의 경우 보통의 암호 scheme과 달리 homomorphic한 성질이 존재해야 하기 때문에, 기존의 RSA, ECC 등의 문제와 다른 문제에 기반한 scheme이 필요합니다. 동형암호의 경우 lattice-base, 즉 격자와 관련된 문제에 기반을 두고 있습니다. 나중에 소개할 scheme 중에서 BGV, BFV, CKKS 등의 경우 RLWE problem (Ring Learning with Erros problem)에 기반하고 있습니다. RLWE 문제가 무엇인지 더 궁금하신 분들은 [여기](https://en.wikipedia.org/wiki/Ring_learning_with_errors), 그리고 구글링을 통해 알아보시면 됩니다. 그 외에 격자 기반 암호학에 관한 글들은 이미 이 블로그에도 다수 있으니 아래 목록의 글들을 읽으면서 공부하시면 좋을 것 같습니다.

- https://www.secmem.org/blog/2020/11/22/Post-Quantum-Cryptography/
- https://www.secmem.org/blog/2020/12/20/SABER/
- https://www.secmem.org/blog/2020/10/23/SVP-and-CVP/

더해서, 위의 문제들은 어렵다고 증명이 돼있으며, 특히 양자 컴퓨터로도 뚫기 힘들다는 것이 알려져 있기 때문에, PQC 환경에서도 동형암호를 쓸 수 있게 됩니다.

# History of Homomorphic Encryption

동형암호에서는 유명한(major한) 몇 가지 scheme들이 있습니다. 이 scheme들을 소개하기 앞서, 동형암호의 역사에 대해 간략히 서술하고자 합니다. 현재 쓰이는 scheme들의 경우 각 generation의 대표주자들이 자주 쓰이기 때문입니다.

### 첫 번째 Generation

[Gentry](https://en.wikipedia.org/wiki/Craig_Gentry_(computer_scientist))가 처음으로, Somewhat homomorphic encryption에 해당하는 scheme을 제안합니다. 이후 이 [논문](https://eprint.iacr.org/2009/616)을 통해 Fully homomorphic encryption scheme을 제안합니다.

### 두 번째 Generation

이 Generation 부터는 현재에도 자주 쓰이는 scheme들이 등장하기 시작합니다. 이 Generation의 scheme들의 경우 대부분 Bootstrapping 과정이 따로 필요하지 않다는 특징이 있습니다. Bootstrapping이란, 암호문에 존재하는 noise를 줄이는 연산을 말합니다.
이때 나온 scheme들로는 BGV, BFV, LTV, BLLN scheme이 있습니다. 이 중 BGV, BFV의 경우 아직까지도 많이 사용되고 있습니다. 다만, LTV, BLLN의 경우 BGV, BFV처럼 격자 기반이 아니라, 다른 문제에 기반하는데, subfield lattice attack에 취약해 현재는 사용되지 않습니다.

### 세 번째 Generation

GSW cryptosystem에 기반한 scheme들이 출현합니다. GSW cryptosystem이란, 이 [논문](https://eprint.iacr.org/2013/340)에서 제시된 내용에 기반한 cryptosystem을 의미합니다.
이때 나온 scheme들로는 FHEW, TFHE scheme이 있습니다.

### 네 번째 Generation

CKKS scheme이 출현합니다.

# Schemes of Fully Homomorphic Encryption

위에서 소개한 동형암호의 역사에서 보이듯, 다양한 동형암호 scheme이 제안 됐고, 사용되고 있습니다. 이 절에서는 Fully homomorphic encryption 중 일부 scheme에 대한 간단한 소개를 하고자 합니다.

BGV scheme
- 두 번째 Generation scheme에 속합니다.
- https://eprint.iacr.org/2011/277 에서 제안됐습니다.
- 정수로 이루어진 data에 대해서 동작하며, 결과가 정확하게 나옵니다.

BGV scheme
- 두 번째 Generation scheme에 속합니다.
- https://eprint.iacr.org/2012/144 에서 제안됐습니다.
- 정수로 이루어진 data에 대해서 동작하며, 결과가 정확하게 나옵니다.
- BGV와 유사하나, 두 scheme에는 몇 가지 차이점이 존재하는데, 이 내용들은 다음에 쓸 기회가 있으면 서술하도록 하겠습니다.

CKKS scheme
- 네 번째 Generation sheme에 속합니다.
- https://eprint.iacr.org/2016/421 에서 제안됐습니다.
- 실수(복소수) data에 대해서 동작합니다.
- 결과가 정확하게 나오지 않고, approxmiate 하게 나옵니다.

위의 세 가지 scheme 외에도 다양한 scheme들이(FHEW, TFHE) 존재합니다. 다른 scheme들과, 각 scheme에 대한 설명은 다음에 글을 쓸 기회가 있으면, 쓰도록 하겠습니다.

# Operations of Fully Homomorphic Encryption

이 절에서는 Fully homomorphic Encryption 중에서도, 더하기, 곱하기가 정의된 scheme들에서 볼 수 있는 연산들에 대해서 설명하도록 하겠습니다. 즉, FHEW, TFHE 같이 Gate 연산이 존재하는 scheme에 해당되는 내용은 아닙니다. 하지만, BGV, BGV, CKKS등의 scheme에서는 유효하기 때문에 간략하게나마 설명하도록 하겠습니다.

Encode
- 다루고자 하는 data를 평문(plaintext)에 대응시킵니다.
- BFV, CKKS 등에서는 encode에서 NTT를 사용하기도 합니다.

Decode
- 평문을 원래 encode 하기 전의 공간에 해당되는 data로 다시 대응시킵니다.
- encode와 비슷하게 NTT를 사용하기도 합니다. encode에서 forwardNTT를 통해서 encode했다면, decode에서는 backwardNTT를 통해서 decode한다던가 하는 식으로 사용합니다.

Encryption
- 평문을 암호화합니다.
- 보통 주어진 Ring에 대한 다항식(Polynomial)의 쌍(pair)의 형태로 암호화합니다.

Decryption
- 암호문을 다시 평문으로 되돌립니다.
- scaling이 필요하다면 적용하기도 합니다.

Addition
- 두 암호문을 더합니다.
- 다항식의 쌍 형태라면, 첫번째 다항식끼리, 두번째 다항식끼리 짝지어서 각각 다항식의 덧셈을 적용해 결과를 구합니다.

Multiplication
- 두 암호문을 곱합니다.
- 곱하고자 하는 암호문을 $(a_{1}, b_{1}), (a_{2}, b_{2})$라고 하면, 곱한 결과는 $(a_{1}a_{2}, a_{1}b_{2}+a_{2}b_{1}, b_{1}b_{2})$이 됩니다.

Relinearization
- 위의 Multiplication의 결과를 다시 다항식 두 개의 쌍 형태로 되돌리는 것을 연산입니다.
- Multiplication과 합쳐서 전체를 Multiplication이라고 하는 경우도 있습니다.
- 보통 Multiplication을 하면 항상 Relinearization도 같이 적용해줍니다.

Bootstrapping
- 암호문의 noise를 줄입니다.
- 보통 Addition, Multiplication을 반복하며 noise가 쌓이기 때문에, Bootstrapping을 거쳐 noise를 낮춰서 계속 연산을 진행합니다.

# Libraries of Homomorphic Encryption

이 절에서는 동형암호 scheme들을 직접 구현한 library들을 소개합니다. 동형암호 scheme들의 경우 logic이 어려운 경우가 많고, parameter setting 등이 어려운 경우가 많기 때문에, library의 구현체들을 가져오는 경우가 많습니다. 많은 library가 오픈소스이기도 합니다. 여기에서는 동형암호에서 유명한 몇 개의 library들을 소개합니다.

HELib
- IBM의 동형암호 library입니다.
- https://github.com/homenc/HElib 에서 확인할 수 있습니다.
- BGV, CKKS scheme을 지원합니다.

SEAL
- Microsoft의 동형암호 library입니다.
- https://github.com/microsoft/SEAL 에서 확인할 수 있습니다.
- BGV, BFV, CKKS scheme을 지원합니다.

OpenFHE
- https://github.com/openfheorg/openfhe-development 에서 확인할 수 있습니다.
- BGv, BFV, FHEW, TFHE, CKKS scheme등을 지원합니다.
- https://www.openfhe.org/ 에서 더 많은 정보를 얻을 수 있습니다.

# Etc

이 절에서는 위에서 서술하지 못한 몇 가지 주제에 대해 설명하려 합니다.

### 동형암호를 적용할 수 있는 분야

처음에 동형암호를 소개하며 언급했듯이, 동형암호는 secret key에 대한 접근을 최소화할 수 있습니다. 따라서, private data analysis와 같이, privacy에 민감한 분야에 적용하기 제안되고 있습니다. 그 외에 cloud 분야 등에도 활용할 수 있을 것입니다.

### 동형암호가 아직 널리 사용되지 않는 이유는?

간단하게 말하면 비효율성 때문입니다. 아직 동형암호 연산들은 널리 사용될 정도로 빠르게 구현되지 못하고 있습니다. 아직 너무 비효율적이라서, 동형암호가 가지고 있는 여러 장점에도 불구하고 널리 쓰이지 못하고 있는 것입니다. 현재 이에 대한 여러 연구가 진행되고 있습니다.

### 동형암호에 대한 정보를 더 얻고 싶다면?

동형암호의 유명한 scheme들의 경우 제안된 paper는 구글링으로 쉽게 얻을 수 있고, 쉽게 풀어쓴 자료들도 많습니다. 
만약에 code들을 직접 작성해보고 싶다면, library들에 example을 제공하는 경우가 많기 때문에, 이 코드들을 활용하는 것을 추천드립니다. 예를 들어 SEAL의 경우 [링크](https://github.com/microsoft/SEAL/tree/main/native/examples)에서 보이듯이 각 scheme과 주제에 대한 예시 코드들을 제공합니다. 주석이 상당히 자세하게 작성되어 있으니, 주석을 읽으시면서 공부하는 것을 추천드립니다.
https://fhe.org/ 에서는 여러 가지 Resource와 함께 Community를 소개하고 있습니다. 관심 있는 분들은 디스코드 등에 들어가셔도 좋을 것 같습니다.

# Conclusion

지금까지 동형암호에 대해서 간략하게 설명을 했습니다. 동형암호가 차세대 기술로 주목되고 있는 만큼, 관심있는 분들은 더 공부해보시는 것도 좋을 것입니다.
저도 아직 공부가 부족해서 실수가 있을 수 있으니, 질문과 지적 모두 감사하게 받겠습니다. 다음에 동형암호 관련 글을 쓴다면 BFV, CKKS 등 scheme들에 대해서 좀 더 자세히 설명하는 글을 쓸 예정입니다. 부족한 글 읽어주셔서 감사합니다!

# Reference

- https://en.wikipedia.org/wiki/Homomorphic_encryption
- https://www.keyfactor.com/blog/what-is-homomorphic-encryption/
