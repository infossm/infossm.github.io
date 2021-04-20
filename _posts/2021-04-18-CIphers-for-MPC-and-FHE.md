---
layout: post
title:  "Ciphers for MPC and FHE"
date:   2021-04-18 17:00:00
author: blisstoner
tags: [Cryptography]
---

# 1. Introduction

최근의 두 포스트에서 Multi-party computation에 대해 다루었습니다. Garbled circuit의 발전사와 최종적인 형태를 보면 결론적으로 AND 연산의 수가 적으면 적을수록 MPC의 성능이 올라감을 알 수 있습니다. 또한 ZK(Zeroknowlege Proof, prover가 verfier에게 어떤 문장이 참임을 증명할 때 그 문장이 참이라는 것을 제외하면 그 어떤 정보도 prover에게 노출하지 않는 프로토콜),FHE(Fully Homomorphic Encryption, 데이터를 암호화한 상태로 연산할 수 있는 암호화 방법) 등에서도 AND 연산이 XOR 연산에 비해 훨씬 큰 비중을 차지합니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/mpcfhe.png)

기존에 널리 쓰이는 암호화 알고리즘으로는 AES가 있습니다. 물론 AES를 설계할 당시 속도에 대한 부분도 비교 대상에 들어갔지만 고전적인 관점에서 속도를 비교할 때 AND와 XOR은 비슷한 비중을 차지하는 반면 MPC, ZK, FHE 등에서는 XOR은 거의 영향을 주지 않는 반면 AND는 성능에 큰 영향을 주기 때문에 자연스럽게 AND를 최대한 적게 활용한 암호화 알고리즘을 만들 필요가 있습니다.

이번 논문에서 새롭게 제안된 암호인 `LowMC`(Low Multiplicative Complexity)는 AES와 같은 기존의 암호에 비해 AND를 적게 사용합니다.

AND 연산을 적게 사용한다고 했을 때, 이 표현은 아래와 같은 2가지의 의미를 가집니다.

1. 전체적인 AND 연산의 개수가 작다.
2. AND 연산의 깊이가 작다.

($(a \cdot b) \cdot c) \oplus (a \cdot d)$의 경우 AND 연산의 개수는 3개, 깊이(=AND 연산이 최대로 중첩되는 횟수)는 2)

MPC에서는 1번과 같이 AND 연산의 개수가 적을수록 성능이 좋아지고 FHE에서는 2번과 같이 AND 연산의 깊이가 작을수록 성능이 좋아집니다. LowMC에서는 설계에서 사용하는 인자들을 조절해 필요에 따라 AND 연산 개수의 관점에서 효율적이거나 깊이의 관점에서 효율적이게 설계할 수 있습다다.

저자가 소개하는 LowMC의 장점은 아래와 같습니다.

1. AND 연산의 개수 / 깊이가 적다.
2. S-box를 부분으로 적용한다.
3. 다양한 암호학적 공격에 대해 안전함을 분석했다.
4. 설계가 유동적이어서 라운드 수 / 블록 크기 등을 자유롭게 조절할 수 있다.

# 2. Structure of LowMC

아래는 LowMC의 구조를 도식화한 그림입니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/1.png)

$$ \text{LowMCRound}(i) = \\ \text{KeyAddition}(i) \circ \text{ConstantAddition}(i) \circ \text{LinearLayer}(i) \circ \text{SboxLayer}$$

LowMC는 AES와 같이 SPN(Substitution-Permutation Network) 구조입니다. 블록의 크기는 $n$, 키의 크기는 $k$, S-box의 개수는 $m$, data complexity는 $d$이고 이 값들은 독립적으로 선택됩니다.

data complexity는 입력이 가질 수 있는 복잡도를 의미합니다. 예를 들어 입력이 128비트이고 입력에 아무런 제한이 없다면 $d = 128$이고, 입력이 200비트인데 앞 8비트는 고정되어 있다면 $d = 192$입니다.

그림에서 볼 수 있듯 $\text{SboxLayer}, \text{LinearLayer}, \text{ConstantAddition}, \text{KeyAddition}$을 차례로 수행하는 것이 LowMC의 한 라운드입니다.

먼저 $\textbf{LinearLayer}(i)$는 GF(2)에서의 행렬 곱셈입니다. 곱해지는 행렬은 binary $n \times n$ matrix $\text{Lmatrix}[i]$입니다.

$\textbf{ConstantAddition}(i)$은 in GF(2)에서의 행렬 덧셈입니다. $\text{roundconstant}[i]$을 더합니다.

$\textbf{KeyAddition}(i)$ in GF(2)에서의 행렬 덧셈입니다. $\text{roundkey}[i]$을 더하고 이 값은 마스터 키 $\text{key}$를 $n \times k$ matrix $\text{Kmatrix}[i]$와 곱해서 얻어집니다.

위에서 언급된 $\text{Lmatrix}, \text{roundconstant}, \text{Kmatrix}$ 들은 설계 당시 정해진 상수입니다.

$n, k, m, d$에 따른 권장 라운드 수는 아래와 같습니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/2.png)

처음 5개는 PRESENT라는 이름의 경량 암호 수준의 안전성을 보장하기 위한 라운드 수입니다. 중간은 AES 수준의 안전성을 보장하기 위한 라운드 수이고 마지막은 128-bit PQC 수준의 안전성을 보장하기 위한 라운드 수입니다.

## Nothing-up-my-sleeve

LowMC 구조에서 $\textbf{LinearLayer}(i)$은 diffusion을 수행해주는 단계입니다. 이외에도 $\textbf{ConstantAddition}(i)$, $\textbf{KeyAddition}(i)$ 에서 Substitution이 추가로 들어가고, 이들은 설계 당시 정해진 $\text{Lmatrix}, \text{roundconstant}, \text{Kmatrix}$ 값들에 따라 동작이 결정됩니다.

여기서 문제는, 설계자가 아닌 다른 사람들에게 해당 상수가 정말 랜덤하게 선택되었고 백도어가 있지 않다는 것을 납득시킬 수 있습니다. 이 Nothing-up-my-sleeve를 위해 설계자들은 Grain LFSR을 이용해 난수를 생성했습니다.

## S-box

LowMC에서 S-box는 3비트 입력을 받아 3비트 출력을 합니다. $S(a, b, c) = (a \oplus bc,  a \oplus b \oplus ac , a \oplus b \oplus c \oplus ab)$으로 나타낼 수 있고 표로 나타내면 아래와 같습니다.

| x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| - | - | - | - | - | - | - | - | - |
| S(x) | 0 | 1 | 3 | 6 | 7 | 4 | 5 | 2 |

이 S-box는 Linear cryptanalysis와 Differential cryptanalysis가 둘 다 최대한 어렵게끔 설계되었습니다. 먼저 Differential characteristic을 보면 아래와 같습니다.

* Differential table $(x \oplus y = \alpha, S(x) \oplus S(y) = \beta)$

| $\alpha / \beta$  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :-: | - | - | - | - | - | - | - | - |
| **0** | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 2 | 0 | 2 | 0 | 2 | 0 | 2 |
| **2** | 0 | 0 | 2 | 2 | 0 | 0 | 2 | 2 |
| **3** | 0 | 2 | 2 | 0 | 0 | 2 | 2 | 0 |
| **4** | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 2 |
| **5** | 0 | 2 | 0 | 2 | 2 | 0 | 2 | 0 |
| **6** | 0 | 0 | 2 | 2 | 2 | 2 | 0 | 0 |
| **7** | 0 | 2 | 2 | 0 | 2 | 0 | 0 | 2 |

Linear correlation 또한 균등하게 잘 퍼져있습니다.

* Correlation table $(\alpha^T \cdot x = \beta^T \cdot S(x))$

| $\alpha / \beta$  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :-: | - | - | - | - | - | - | - | - |
| **0** | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| **1** | 4 | 2 | 4 | 6 | 4 | 6 | 4 | 6 |
| **2** | 4 | 4 | 6 | 6 | 4 | 4 | 6 | 2 |
| **3** | 4 | 6 | 2 | 4 | 4 | 6 | 6 | 4 |
| **4** | 4 | 4 | 4 | 4 | 6 | 2 | 6 | 6 |
| **5** | 4 | 6 | 4 | 6 | 6 | 4 | 2 | 4 |
| **6** | 4 | 4 | 6 | 2 | 6 | 6 | 4 | 4 |
| **7** | 4 | 6 | 6 | 4 | 2 | 4 | 4 | 6 |

# 3. Comparison with other ciphers

![](/assets/images/Ciphers-for-MPC-and-FHE/3.png)

위의 사진을 보면 동일한 안전성을 제공하는 암호들 중에서 LowMC가 독보적으로 ANDdepth 혹은 bit당 사용하는 AND 개수가 적음을 알 수 있습니다.

# 4. Cryptanalysis

사실 암호 시스템이야 누구나 만들 수 있습니다. 극단적으로 말해 키 $K$가 주어지면 $C = P \oplus K$로 암호화하는 시스템을 생각해보면 이 암호 시스템은 AND 연산이 전혀 필요하지 않아 빠르게 계산 가능한 암호입니다. 하지만 결국 중요한 것은 암호 시스템이 얼마나 안전한지이고, 암호 시스템이 안전하다는 것을 보일 수 있는 현존하는 방법은 현재까지 제안된 여러 공격들에 대해 안전함을 보이는 일입니다.

논문에서는 Differential Cryptanalysis, Linear Cryptanalysis, Boomerang attacks, Higher order attacks, Interpolation attack에 대해 LowMC가 안전함을 보입니다.

## Differential Cryptanalysis

안타깝게도 비단 LowMC 뿐만 아니라 다른 암호에서도 주어진 암호에서 가장 Differential chracteristics가 큰 path를 찾는 것은 사실상 불가능합니다. 그렇기에 논문에서는 실제 제안된 상수 값들에 대해 Differential chracteristics를 탐색하는 것이 아닌, Linear layer가 충분히 랜덤하다고 보고 확률이 $2^{-d}$이상인 differential path가 있을 가능성이 굉장히 낮다는 방식으로 확률을 계산합니다.

자세한 계산 식은 생략하지만 아래와 같이 $m = 42, d = 128$일 때 8라운드를 넘기면 확률이 $2^{-d}$이상인 differential path가 있을 가능성이 $2^{-128}$ 미만임을 알 수 있습니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/4.png)

## Linear Cryptanalysis

Linear characteristics에서 최대 correlation이 Differential characteristics와 같이 $2^{-2}$였기 때문에 Linear Cryptanalysis를 통해 계산되는 bound는 Differential Cryptanalysis와 완전히 동일힙니다.

## Boomerang attacks

Boomerang attacks는 Differential Cryptanalysis의 발전된 형태로, 여러 partial differential characteristics을 연결하는 공격 기법입니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/boomerang.png)

편의상 각각의 확률이 $p_1, p_2$인 두 differential characteristics를 엮는다고 할 때 공격에 필요한 복잡도는 $1/(p_1^2p_2^2)$입니다.

그러면 우리는 각 path의 확률의 곱이 $2^{-d/2}$ 이상이 되는 라운드 수를 정하면 됩니다. 계산 식은 생략합니다.

## Higher order attacks

이 공격 또한 Differential Cryptanalysis의 발전된 형태입니다.

2nd-order attack을 예로 들어보면 아래와 같은 식을 얻을 수 있습니다.

- $\Delta_1 = E(M) \oplus E(M \oplus \alpha_1)$

- $\Delta_2 = E(M_1 \oplus \alpha_2) \oplus E((M_1 \oplus \alpha_2) \oplus \alpha_1)$

- $\Delta = \Delta_1 \oplus \Delta_2$

즉 order가 1 증가할 때 마다 필요한 데이터의 양은 2배로 증가하고, $t$-nd-order attack을 수행하기 위해서는 적어도 $O(2^t)$의 시간이 필요합니다.

또한 대수학적 관점에서 보면 derivative를 취하는 것은 곧 algebraic degree를 1 감소시키는 것입니다. 그렇기 때문에 암호문의 각 bit을 대수학적으로 표현했을 때 degree가 $d$ 이상이라면 Higher order attack으로부터 안전합니다.

현재까지의 공격들을 모아보면 아래와 같이 필요한 라운드 수를 계산할 수 있습니다.

![](/assets/images/Ciphers-for-MPC-and-FHE/6.png)

## Interpolation attack

이 공격은 초기 LowMC에서 고려하지 못했던 공격입니다. 초기 LowMC는 이 공격으로 인해 취약했습니다.

만약 암호문의 특정 bit을 다항식으로 나타낼 수 있다면 그 다항식의 계수는 선형 연립 방정식을 해결함으로서 얻어낼 수 있습니다.

예를 들어 1-round interpolation attack을 생각해보면 각 S-box의 output은 최대 2차이기 때문에 $3m$개의 quadratic terms를 가지고 있고, 항의 개수가 제한되어 있기 때문에 충분한 평문을 모으면 선형 연립 방정식을 풀어 키를 복원해낼 수 있습니다.

`2. Structure of LowMC`에서 언급된 $n, k, m, d$에 따른 권장 라운드 수는 이러한 점을 고려해 공격의 복잡도가 $2^{-d}$를 넘도록 하는 라운드 수입다다.

# 5. Benchmarks

## MPC - Single block

![](/assets/images/Ciphers-for-MPC-and-FHE/7.png)

## MPC - Multi block

![](/assets/images/Ciphers-for-MPC-and-FHE/8.png)

## FHE

![](/assets/images/Ciphers-for-MPC-and-FHE/9.png)

실제 Benchmarks를 통해 확인할 수 있듯 LowMC는 MPC, FHE에서 효율적으로 쓰일 수 있습니다.

# 6. Conclusion

논문에서는 AND를 적게 사용하기 때문에 MPC, FHE, ZK에서 효율적으로 사용할 수 있는  LowMC 암호를 제안하고 해당 암호가 충분히 안전함을 보였습니다.

이번 글에서는 다루지 않았지만 이 암호를 이용해 Post-Quantum Zero-Knowledge와 Post-Quantum Digital Signature Algorithm을 만들 수 있고, 이렇게 만든 `Picnic`이라는 이름의 전자서명 시스템은 NIST Post-Quantum Cryptography 공모에서 라운드 3 Alternate candidates로 선택되었습니다.

Picnic에 대해 자세한 설명은 아마 다음 글에서 드릴 것 같습니다만, Picnic에서 가장 중요한 점은 동일한 키에 대해 평문-암호문 쌍이 단 1개만 제공된다는 점입니다. 그렇기 때문에 Picnic에 LowMC가 쓰였을 때 이것이 안전하지 않음을 보이기 위해서는 평문-암호문 쌍이 단 1개만 있을 때에 의미있는 공격을 수행할 수 있어야 하고 관련한 대회가 2020년부터 진행되었습니다.[LowMC Cryptanalysis Challenge](https://lowmcchallenge.github.io/)

한편 LowMC 이후에도 Ciminion과 같은 다양한 MPC-friendly cipher가 제안되었습니다(Ciminion은 Eurocrypt 2021에 선정되었습니다). 아직 논문을 자세히 읽어보지는 않았지만 해당 논문에서는 LowMC에서 Diffusion을 위해 행렬을 곱하는 과정이 실제로는 굉장히 고비용의 연산임을 지적하면서 이와 다르게 효율적인 방식으로 설계가 되어있습니다.

이처럼 AND 연산을 줄이고 안전한 암호를 설계하는 연구는 앞으로도 여러 방향으로 이어질 것으로 예측됩니다.

# 6. Open problems

저자는 논문에서 아래와 같은 open problems를 제안했습니다.

1. 만약 AND 연산이 XOR 연산보다 더 비용이 들지 않는 환경에서는 암호를 어떻게 설계해야 할 것인가?

2. LowMC의 과정에서 $\text{ConstantAddition}(i)$가 없다고 해도 안전하지 않을까?

3. Linear layer에 과정을 더 추가해 더 안전하게 만들수는 없을까?

4. 만약 3비트보다 더 큰 입력을 받는 S-box를 쓰고 싶다고 하면 어떻게 그러한 S-box를 찾을 수 있을까?