---
layout: post
title:  "Forgery Attack on ElGamal Signatures"
date:   2019-12-15 23:20
author: RBTree
tags: [cryptography, digital-signature, number-theory]
---

# 서론

최근에 한 문제에 대한 질문을 받았습니다. FuzzyLand라는 사이트의 [WebShop 2.0](https://fuzzy.land/challenges#WebShop%202.0) 이라는 문제로, [ElGamal signature](https://en.wikipedia.org/wiki/ElGamal_signature_scheme) 에 대한 공격을 요구하는 문제였습니다.

문제를 풀다보니 아이디어가 흥미로워서 공유하게 되었습니다.

# 본론

## ElGamal Signatures

ElGamal signature는 RSA signature와 비슷하게 discrete logarithm의 어려움에 바탕을 둔 signature scheme입니다. ElGamal signature는 다음과 같은 방식으로 돌아갑니다.

---

**파라미터 및 생성**

1. 어떤 $N$-bit 소수 $p$를 고릅니다. 그리고 Hash function $H$를 고릅니다.
2. 2부터 $p-1$ 사이의 어떤 수 $g$를 고릅니다. 이 수를 generator라고 부릅니다.

이 때 generator라고 불리는 이유를 대략적으로 설명하자면, 어떤 $x$에 대한 $g^x (\mathbb{mod}\ p)$를 통해서 1부터 $p-1$까지 모두 만들 수 있기 때문입니다.

**키 생성**

1. 1부터 $p - 2$ 사이에서 어떤 수 $x$를 고릅니다. 이 수는 **비밀키** 입니다.
2. $y := g^x (\mathbb{mod}\ p)$를 계산합니다. 이 $y$는 **공개키** 입니다.

**서명**

1. 2부터 $p-2$ 사이에서 $p-1$과 서로소인 $k$를 고릅니다.
2. $r := g^k (\mathbb{mod}\ p)$를 계산합니다.
3. $s := (H(m)-xr)k^{-1} (\mathbb{mod}\ p-1)$를 계산합니다. $s = 0$ 일 시 1번으로 돌아가 다른 $k$를 사용합니다.

4. 서명은 $(r, s)$입니다.

**검증**

1. $0 < r < p, 0 < s < p-1$인지 확인합니다.
2. $g^{H(m)} \equiv y^r r^s (\mathbb{mod}\ p)$인지 확인합니다.

---

보다시피, 무작위로 생성된 $k$가 서명에 들어갑니다. 이런 random nonce가 들어간 경우 random nonce가 유출되거나 예측할 수 없도록 주의해야 합니다. 많은 서명이 이런 nonce-based signature scheme이고, [EdDSA](https://en.wikipedia.org/wiki/EdDSA)와 같이 random nonce가 없어 한 message에 대해서 항상 같은 signature를 생성하는 signature scheme도 있습니다.

## WebShop 2.0

WebShop 2.0 문제는 다음과 같은 문제입니다.

- 파라미터 $p, g$가 소스 코드에 주어져 있고, 서버 측에서 서명에 사용하는 키 쌍 중 공개키 값이 $y$가 소스 코드에 주어져 있습니다.
- 유저는 코인을 처음에 1개 들고 있습니다. 그에 따라서 서버는 수 $1$에 대해서 signature를 생성하고, 이를 유저에게 공유해줍니다.
- 유저가 어떤 수 $m$과 함께 그에 대한 signature를 보내면 해당 signature를 검증한 뒤, 올바른 signature일 시 유저의 코인 개수를 $m$개로 바꿔줍니다.
- 유저가 코인을 99999개 가지고 있을 시, 문제의 정답에 해당하는 flag를 구입할 수 있습니다.

문제만 보면 99999 이상의 수에 대한 signature를 어떻게든 생성해서 보내는 문제로 보입니다. 그래서 ElGamal signature에 대한 코드를 살펴보면 뭔가 이상한 점을 알 수 있습니다.

```python
    def sign(self, m):
        """
        Signs the message m using the private key.
        Makes sure to generate a good randomness k, we don't want to be Sony...
        """
        assert(1 <= m <= self.p - 1)

        k = self.rand.randint(2, self.p - 2)
        while gcd(k, self.p - 1) != 1:
            k = self.rand.randint(2, self.p - 2)

        r = powmod(self.g, k, self.p)
        k_inv = invert(k, self.p-1)
        s = ((m - self.x * r) * k_inv ) % (self.p - 1)

        return (r,s)

    def verify(self, sig, m):
        """
        Verify a signature in the form (r,s) for a message m.
        Returns True if the signature sig verifies for the message m,
                False otherwise,
        """
        try:
            (r,s) = sig
            if not 1 < r < self.p:
                return False

            left = powmod(self.g, m, self.p)
            right = (powmod(self.y, r, self.p) * powmod(r, s, self.p)) % self.p
            return left == right
        except Exception as e:
            # print(e)
            return False
```

앞서 살펴봤던 ElGamal signature에서는 hash function $H$를 정의해 서명/검증 과정에서 메시지 $m$에 대해 사용하고 있습니다. 하지만 해당 코드에서는 $H(m)$이 아닌, $m$을 그대로 서명/검증 과정에서 사용하고 있습니다.

그렇다면 이 부분이 취약하지 않을까요?

## Signature Forgery Attack

Signature forgery attack이라고 불리는 공격 방식이 있습니다. 공격 자체는 아니고 공격에 대한 분류인데, 어떤 $m$에 대해서 signer가 아닌 다른 그룹이 과거에 생성된 적이 없는 signature $\sigma$를 생성하는 형태의 공격을 일컫습니다.

Signature forgery attack을 세 가지로 더 분류해볼 수 있습니다.

- **Existential forgery**: 기존에 생성된 적 없는 메시지/서명 쌍 $(m, \sigma)$를 만들 수 있는 forgery attack입니다. 이 때 $m$은 우리가 임의로 지정할 수 있는 어떤 값이 아닙니다. 즉, $m$을 골라서 서명 $\sigma$를 만들 수는 없더라도, 수학적으로 검증 과정을 통과할 수 있는 어떤 $(m, \sigma)$ 쌍을 임의로 생성할 수 있는지가 이 공격의 핵심입니다.
- **Selective forgery**: 공격자가 공격을 시행하기 전에 선택한 메시지 $m$에 대해서, 서명 $\sigma$를 생성할 수 있는 경우입니다.
- **Universal forgery**: 임의의 $m$에 대해서, 서명 $\sigma$를 만들 수 있는 경우입니다.

보다시피, Universal forgery > Selective forgery, > Existential forgery 순으로 공격이 어렵습니다.

### Existential Forgery in ElGamal Signatures w/o Hashing

Hash function이 없는 ElGamal signature에서는 두 가지 방법의 existential forgery가 가능하다고 알려져 있습니다.

#### 첫 번째 방법

첫 번째 방법은 다음과 같습니다.

1. 2부터 $p-2$ 사이의 어떤 수 $e$를 고릅니다.
2. $r := g^e y(\mathbb{mod}\ p)$, $s := -r (\mathbb{mod}\ p-1)$을 계산합니다.
3. $m := es (\mathbb{mod}\ p-1)$을 계산합니다.
4. $(r, s)$는 계산한 $m$에 대해 valid한 signature입니다.

앞서 검증 과정을 그대로 따라가보면, 우항 $y^rr^s$는

$y^r r^s \equiv y^r (g^e y)^s \equiv g^{es} y^{r+s} \equiv g^{es} \equiv g^m (\mathbb{mod}\ p)$

로 좌항 $g^m$과 합동임을 알 수 있습니다.

#### 두 번째 방법

두 번째 방법은 다음과 같습니다.

1. 2부터 $p-2$ 사이의 어떤 수 $e, v$를 고릅니다. 이 때 $\gcd(v, p-1) = 1$이여야 합니다.
2. $r := g^e y^v (\mathbb{mod}\ p)$, $s := -rv^{-1} (\mathbb{mod}\ p-1)$을 계산합니다.
3. $m := es (\mathbb{mod}\ p-1)$을 계산합니다.
4. $(r, s)$는 계산한 $m$에 대해서 valid한 signature입니다.

이 방법 역시 검증 과정을 그대로 따라가보면, 우항 $y^r r^s$는

$y^r r^s \equiv y^r (g^e y^v)^s \equiv g^{es} y^{r + sv} \equiv g^{es} y^{r - r} \equiv g^m (\mathbb{mod}\ p)$

로 좌항 $g^m$과 합동임을 알 수 있습니다.

## Solving WebShop 2.0

두 방법을 사용하면서, $m$이 99999 이상인 경우만 출력하게 하면 됩니다. 이 때 $p$는 충분히 크기 때문에, 계산한 $m$이 99999 미만일 확률은 매우 낮습니다.
Python3에서 PyCryptodome 라이브러리를 사용해 작성했습니다.

```python
from Crypto.Util.number import GCD as gcd, inverse
import random

p = 2673971600395909170632... # It's too long! omitted
g = 2331230152027837170250...
y = 2376273319883441079494...

# First method
while True:
    e = random.randint(2, p - 2)

    r = pow(g, e, p) * y % p
    s = -r % (p - 1)
    m = e * s % (p - 1)

    if m >= 99999:
        print("{}:{}:{}".format(m, r, s))
        break

# Second method method
while True:
    v = random.randint(2, p - 2)
    if gcd(v, p - 1) != 1:
        continue
    e = random.randint(2, p - 2)

    r = pow(g, e, p) * pow(y, v, p) % p
    s = -r * inverse(v, p - 1) % (p - 1)
    m = e * s % (p - 1)

    if m >= 99999:
        print("{}:{}:{}".format(m, r, s))
        break
```

이를 실행해서 얻은 값을 문제에 넣고 시도해보면 다음과 같이 답을 얻을 수 있습니다.

<img src="/assets/images/rbtree/webshop2.png">

(답을 올려놓으면 정말 그대로 가져다 쓰는 사람이 있을 것 같아 최소한 solver는 돌려보라고 가려두었습니다.)

# 결론

독학으로 공부하다보니 이런 형태의 공격은 처음 만나봐서 신선했습니다. 일반적인 경우에는 서명에서는 당연히 hash 과정을 거치게 되기 때문에 이런 공격을 만날 일은 매우 드뭅니다.

대회 문제나 실제 프로그램에서 사용하는 signature scheme에 hash function이 들어있지 않다면, forgery attack을 의심해보는 것이 좋은 습관일 것이라고 생각합니다.
