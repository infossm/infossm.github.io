---
layout: post
title:  "Triple DES의 안전성"
date:   2022-06-19 10:00:00
author: blisstoner
tags: [cryptography]
---

# 1. Introduction

Triple DES는 DES를 3번 진행하는 암호이고, DES가 낮은 키의 엔트로피(56비트)로 인해 컴퓨터 성능이 발전함에 따라 점차 충분한 안전성을 제공할 수 없게 되자 다음 암호인 AES가 등장하기 전까지 임시로 사용되었습니다. 

Triple DES는 DES를 서로 다른 키 $K_1, K_2, K_3$에 대해 3번 적용합니다. 즉 키의 길이는 168비트입니다. 이 때 Single DES와의 호환성을 위해 처음 두 번의 DES는 암호화, 세 번째의 DES는 복호화로 둡니다. 또한 경우에 따라 $K_1 = K_3$으로 두어 112비트의 키를 사용하기도 합니다.

Triple DES는 DES를 3번 적용하기 때문에 일단 AES보다 느립니다. 또 이 글에서 살펴보겠지만 키의 길이가 168비트임에도 불구하고 최소 $O(2^{128})$ 비트의 안전성을 제공하는 AES보다 더 안전하지도 않습니다. 물론 그렇다고 해서 Single DES처럼 아예 하루만에 깰 수 있다거나 하는 수준은 아니긴 합니다.

이번 글에서는 Triple DES의 특성과 제시된 여러 공격들을 살펴보겠습니다.

# 2. Cascade Encryption

작은 키를 가진 블록 암호 구조를 가지고 지금처럼 안전성을 높여야한다면 Triple DES에서 그렇게 한 것 처럼 동일한 블록 암호를 여러 번 중첩시킨다는 발상을 쉽게 떠올릴 수 있습니다. 그리고 이 경우 구조의 안전성이 어떻게 될지는 이미 연구가 이루어진 바가 있습니다`[LR86, Mye99, MT09, Tes11, Vau03, CPS14]`. 이렇게 같은 블록 암호를 중첩시킨 구조를 Cascade encryption이라 부르고 수식을 이용해 아래와 같이 정의할 수 있습니다.

$$E'_{k_1, \dots, k_r}(x) := (E_{k_r} \circ \dots \circ E_{k_1})(x).$$

블록암호 $E$가 $\epsilon$의 안전성을 제공한다고 할 때 키 $k_1, \dots, k_r$이 독립일 경우에는 $E'$이 동일한 쿼리와 시간의 제한 조건 안에서 $r \cdot \epsilon^r$의 안전성을 제공해 $E$보다 $E'$이 더 좋은 안전성을 가집니다.

그리고 키 $k_1, \dots, k_r$이 모두 같을 경우에는 마찬가지 조건에서 $E'$은 $\epsilon + (2r+1)q / N$($q$는 쿼리의 수)의 안전성을 제공해 $E$보다 $E'$이 더 안좋은 안전성을 가집니다. 전부 같은 키를 사용할거면 안전성이 더 안좋아지는데 굳이 $r$번 중첩시킬 필요가 있나 하는 의문이 들 수 있지만, 이 구조는 공격자의 전수 조사 공격을 느리게 하는 목적으로 사용될 수 있습니다. 또한 안전성에 붙게 되는 $(2r+1)q / N$은 라운드 수 $r$과 쿼리의 수 $q$에 대해 선형으로 증가하기 때문에 같은 블록 암호를 중첩시킨다고 해서 안전성이 그렇게 약화되지도 않음을 보였습니다.

# 3. Double DES

Triple DES에 대해 알아보기 전, 키 2개를 사용하는 Double DES를 생각해봅시다. Double DES는 $y = E_{K_2}(E_{K_1}(x))$으로 정의할 수 있습니다. DES의 키가 56비트이니 Double DES는 112비트의 안전성을 제공해줄 것 같지만 사실은 Double DES도 고작 56비트의 안전성만을 제공해줍니다. 그 이유는 구조의 특성상 MITM(Meet-In-The-Middle)이 가능하기 때문입니다. 어떤 평문-암호문 쌍 $(x, y)$이 주어졌다고 할 때

$$E_{K_1}(x) = E^{-1}_{K_2}(y)$$

이 성립하기 때문에 $2^{56}$개의 $K_1$ 후보와 $K_2$ 후보에 대해 $E_{K_1}(x)$, $E^{-1}_{K_2}(y)$를 계산하고 둘 사이에 일치하는게 있는지를 해시 자료구조나 정렬 후 이분탐색을 이용해 구하면 간단히 키를 복원해낼 수 있습니다. 그렇기 때문에 Double DES는 암호학적 관점에서 큰 의미가 없습니다.

# 4. Triple DES

Double DES는 확인한 것과 같이 충분한 안전성을 제공해주지 못하기 때문에 최소한 Triple DES를 사용해야 합니다. Triple DES를 $y = E_{K_3}(E_{K_2}(E_{K_1}(x)))$으로 정의하는게 다소 자연스럽지만, $y = E^{-1}_{K_3}(E_{K_2}(E_{K_1}(x)))$과 같이 마지막에는 복호화를 해서 $K_3 = K_2$로 두면 Single DES로 활용할 수 있게끔 했습니다.

## A. 공격 1 - MITM

Triple DES 또한 MITM이 가능합니다. 원래 키의 길이는 168비트이지만 $E_{K_2}(E_{K_1}(x)) = E_{K_3}(y)$이므로 $O(2^{112})$에 공격이 가능합니다. 또 테이블에 $E_{K_3}(y)$의 값들을 전부 저장해둔 후 $E_{K_2}(E_{K_1}(x))$은 하나 하나 계산한대로 테이블에서 매칭이 되는 값을 찾으면 되므로 공간복잡도는 $O(2^{112})$가 아닌 $O(2^{56})$입니다.

## B. 공격 2 - Lucks' attack

Triple DES에서 쉽게 착안할 수 있는 MITM은 위에서 확인한 것과 같이 키를 2개/1개로 나누어 MITM을 수행하는 방법입니다. 이 방법은 아쉽게도 전수조사의 비트를 정확히 절반으로 떨구는 대신 2/3으로 떨굴 수 있었습니다.

Lucks는 조금 더 발전된 형태의 공격을 제안했습니다`[Lucks98]`. 공격은 아래와 같이 진행됩니다. $2^{32}$개의 평문-암호문 쌍 $(x_i, y_i)$가 주어져있다고 가정하겠습니다.

1. 중간값 후보 $u_1, \dots, u_{2^{32}}$를 임의로 선택합니다. 우리는 $u_i$가 어떤 $j$에 대해 $u_i = E_{K_2}(E_{K_1}(x_j))$, 즉 2번 암호화한 후의 중간값이길 기대합니다. 평문-암호문 쌍이 $2^{32}$개이고 중간값 또한 $2^{32}$개를 택했으니 $u_i = E_{K_2}(E_{K_1}(x_j))$인 $u_i, x_j$가 한 쌍 정도는 있을 것으로 기대할 수 있습니다.

2. 각 $2^{56}$개의 키의 후보와 $2^{32}$개의 평문에 대해 $E_k(x_i)$를 전부 계산한 후 테이블 1에 저장합니다.

3. 각 $2^{56}$개의 키의 후보와 $2^{32}$개의 중간값 후보에 대해 $E^{-1}_k(u_i)$를 전부 계산한 후 테이블 2에 저장합니다.

4. 테이블 1과 테이블 2에서 겹치는 값을 찾아 $E_{K_1}(x_i) = E^{-1}_{K_2}(u_j)$인 $K_1, K_2$를 구합니다. 이후 $2^{56}$개의 키의 후보에 대해 $E_{K_3}{y_i} = u_j$을 만족하는 $K_3$을 찾습니다. 이러한 $K_3$이 있다면 $(K_1, K_2, K_3)$은 올바른 키의 후보이고, 이들이 실제 올바른 키가 맞는지를 여러 평문-암호문 쌍으로 검증합니다.

Lucks' attack은 $2^{32}$개의 중간값을 예측하는 방식으로 MITM을 수행합니다. 이 방법을 통해 $2^{88}$의 DES 쿼리로 Triple DES를 공격할 수 있습니다.

# 5. 2-key Triple DES

Triple DES는 위에서 확인한 것과 같이 간단한 MITM으로는 $O(2^{112})$, Lucks' attack으로는 $O(2^{88})$에 공격을 할 수 있습니다. 공격의 시간복잡도를 살펴보면 전성에 문제가 없다는 가정 하에 굳이 키를 168비트를 쓸 필요가 없고 112비트만 쓰더라도 상관이 없음을 알 수 있습니다. 실제로 Triple DES에서 $K_1 = K_3$으로 두는 2-key Triple DES가 존재하고 이는 2016년 까지도 ISO/IEC와 같은 표준에 등록이 되어있습니다. 어떻게보면 2-key Triple DES는 Triple DES의 하위호환인 만큼 Lucks' attack을 사용하면 일단 $O(2^{88})$에 공격할 수 있음은 자명합니다. 그리고 2-key Triple DES에 특화된 또 다른 공격이 존재합니다.

## A. van Oorschot-Wiener attack [OW91]

먼저 van Oorschot-Wiener attack를 살펴보면 아래와 같은 방법으로 진행이 됩니다. 공격자는 동일한 키 $(K_1, K_2)$의 2-key triple DES로 만들어낸 $n$개의 평문/암호문 쌍을 가지고 있습니다.

1. 수집한 $n$개의 평문-암호문 쌍을 평문 $P$를 키로 하는 테이블 1에 넣습니다.

2. 임의의 중간값 $A$를 선택합니다. 만약 $A = e_{K_1}(P)$가 만족되는 $P$가 있다면 키를 찾아낼 수 있습니다.

3. 모든 $2^{56}$개의 $K_1$ 키 후보 $i$에 대해 $P_i = d_i(A)$를 계산하고 $P_i$를 테이블 1에서 찾습니다. 만약 $P_i$가 테이블 1에 있다면 대응되는 암호문 $C$를 가지고 $B = d_i(C)$를 계산합니다. 그리고 $(B, i)$를 테이블 2에 넣습니다. 키는 $B$이고 중복되는 $B$가 여러번 등장할 수 있습니다.

4. $2^{56}$개의 $K_2$ 후보 $j$에 대해 $B_j = d_j(A)$를 계산하고 $B_j$가 해시 테이블 2에 포함되어 있는지 확인합니다. $B_j$가 해시 테이블 2에 있다면 해시 테이블 안의 값 $i$와 $j$가 각각 $K_1, K_2$의 후보 키이므로 아무 2개의 평문/암호문 쌍에 대해 실제로 2-key triple DES를 수행해 키가 올바른지 확인합니다.

5. 만약 키를 찾아내는데 실패했다면 새로운 $A$를 선택해 3-4번 과정을 반복합니다.

이 때 $A = E_{K_1}(P)$가 만족되는 $P$가 있는 $A$를 택할 확률은 $n / 2^{56}$이고, 올바른 $A$를 골랐을 때 키를 복원할 확률은 3번 과정에서 가능한 $2^{64}$개의 값 중에서 $2^{56}$개의 값만 확인하기 때문에 $1/2^{8}$입니다다. 종합하면 각 $A$에 대해 2-4번 과정에서 키를 복원할 확률은 $n / 2^{64}$이고, 매번 $2^{57}$번의 DES 연산을 필요로 하므로 공격의 시간복잡도는 $O(2^{121}/n)$입니다. 과정을 보면 Lucks' attack과 어느 정도 비슷한 것을 확인할 수 있고 실제로 $n = 2^{32}$일 땐 Lucks' attack과 동일한 시간복잡도를 가집니다.

## B. Generalized van Oorschot-Wiener attack [Mit16]

van Oorschot-Wiener attack이 1991년 처음 제시된지 대략 25년이 지난 후에 van Oorschot-Wiener attack을 약간 더 개선한 공격이 제안되었습니다. 이 일반화된 공격은 서로 다른 키를 이용해 암호화된 평문/암호문 쌍이 주어졌을 때에도 활용이 가능합니다. 일반화된 공격은 아래와 같습니다.

수집한 $n$개의 평문-암호문 쌍과 키에 대한 라벨을 사용한 키의 종류에 따라 평문 $P$를 키로 하는 별개의 테이블에 넣습니다.

임의의 중간값 $A$를 선택한다. 만약 $A = E_{K_1}(P)$가 만족되는 $P$가 있다면 키를 찾아낼 수 있습니다. 

모든 $2^{56}$개의 $K_1$ 키 후보 $i$에 대해 $P_i = d_i(A)$를 계산하고 $P_i$를 테이블 1에서 찾습니다. 만약 $P_i$가 해시 테이블 1에 있다면 대응되는 암호문 $C$를 가지고 $B = E^{-1}_i(C)$를 계산합니다. 그리고 $(B, i, s)$를 테이블 2에 넣습니다($s$는 키에 대한 라벨). 키는 $B$이고 중복되는 $B$가 여러번 등장할 수 있습니다.

각 $2^{56}$개의 $K_2$ 후보 $j$에 대해 $B_j = E^{-1}_j(A)$를 계산하고 $B_j$가 테이블 2에 포함되어 있는지 확인합니다. $B_j$가 테이블 2에 있다면 테이블 안의 값 $i$와 $j$가 각각 라벨 $s$에 대한 $K_1, K_2$의 후보 키이므로 아무 2개의 평문/암호문 쌍에 대해 실제로 2-key triple DES를 수행해 키가 올바른지 확인합니다.

만약 키를 찾아내는데 실패했다면 새로운 $A$를 선택해 3-4번 과정을 반복합니다.

해시 테이블 1에서 키에 대한 라벨이 추가된다는 것 이외에는 원래의 공격과 크게 다른 것이 없고 시간복잡도 또한 동일합니다.

# 6. 결론

이번 글에서는 Triple DES와 관련된 여러 가지 성질과 공격을 살펴보았습니다. 보통 Triple DES와 같은 Cascade Encryption은 신뢰할 수 있는 암호학적 구조 1개를 가지고 키의 길이를 늘리고 싶을 때 사용할 수 있습니다. 이 때 MITM을 이용해 전수조사보다 빠르게 동작하는 공격을 찾을 수 있음을 항상 인지해야 합니다.

# 7. 참고 문헌

[LR86] Michael Luby and Charles Rackoff. Pseudo-random Permutation Generators and Cryptographic Composition. In Symposium on Theory of Computing - STOC ’86, pages 356–363. ACM, 1986

[Mye09] Steven Myers. On the Development of Block-Ciphers and Pseudo-Random Function Generators Using the Composition and XOR Operators. PhD thesis, University of Toronto, 1999.

[Vau03] Serge Vaudenay. Decorrelation: A Theory for Block Cipher Security. Journal of Cryptology, 16(4):249–286, 2003.

[CPS14] Benoit Cogliati, Jacques Patarin, and Yannick Seurin. Security Amplification for the Composition of Block Ciphers: Simpler Proofs and New Results. In Antoine Joux and Amr M. Youssef, editors, Selected Areas in Cryptography - SAC 2014, volume 8781 of LNCS, pages 129–146. Springer, 2014.

[MS15] Minaud, B., & Seurin, Y. (2015, August). The iterated random permutation problem with applications to cascade encryption. In Annual Cryptology Conference (pp. 351-367). Springer, Berlin, Heidelberg.

[Lucks98] Lucks, Stefan. "Attacking triple encryption." International Workshop on Fast Software Encryption. Springer, Berlin, Heidelberg, 1998.

[Mit16] Mitchell, Chris J. "On the security of 2-key triple DES." IEEE transactions on information theory 62.11 (2016): 6260-6267.

[OW91] P. C. van Oorschot and M. J. Wiener, “A known plaintext attack on two-key triple encryption,” in Advances in Cryptology—EUROCRYPT (Lecture Notes in Computer Science), vol. 473, I. B. Damgard, Ed. Berlin, Germany: Springer-Verlag, 1991, pp. 318–325.
