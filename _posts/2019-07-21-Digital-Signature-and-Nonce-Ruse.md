---
layout: post
title:  "Digital Signature and Nonce Reuse"
date:   2019-07-21 23:20
author: RBTree
tags: [digital-signature, DSA, security, number-theory]
---

# 서론

Digital signature, 혹은 디지털 서명은 컴퓨터 네트워크에서 빠져서는 안 될 중요한 요소 중 하나이다. 수신자는 발신자로부터 메시지를 수신한 뒤, 메시지에 같이 붙어 온 digital signature를 검증해 해당 메시지가 정말로 발신자로부터 온 것인지 확인할 수 있다. 이를 보장할 수 있는 근간에는 다양한 수학 이론이 깔려있고, 이 중 중심이 되는 것은 정수론과 타원 곡선이라고 할 수 있다.

오늘 이 포스트에서는 이 중에서도 가장 쉽게 이해할 수 있고 널리 쓰이고 있는 DSA에 대해서 알아보고, nonce reuse가 왜 위험한지 알아보고자 한다.

# 서명 알고리즘

## 서명 알고리즘의 동작 원리

서명 알고리즘에는 주로 세 개의 함수가 정의되어 있고, 세 함수는 각각 Key generation(키 생성), Sign(서명), Verify(검증) 역할을 담당한다.

Key generation은 키를 생성하는 함수이다. 이 함수를 사용하면 공개키와 비밀키의 키 쌍이 생성된다.

$(x, y) \leftarrow key\_generation()$

위 식에서 $x$는 비밀키, $y$는 공개키에 해당한다. 비밀키는 서명을 생성할 때 사용되고, 공개키는 서명을 검증할 때 사용된다. 비밀키는 메시지에 서명할 사람이 은밀하게 감추고 있어야 하고, 공개키는 서명을 검증하는 누구에게나 공유될 수 있다.

Sign은 서명을 작성하는 함수이다.

$sig \leftarrow sign(msg, x)$

Verify는 메시지와 서명, 공개키를 받으면 서명이 올바른지를 검사한다.

$\text{True}\ or\ \text{False} \leftarrow verify(msg, sig, y)$

## DSA

DSA는 Digital Signature Algorithm의 약자로, NIST에서 2013년에 발표한 FIPS 186-4에 따라서 표준으로 정해져 있는 서명 알고리즘 중 하나이다.

자세한 설명에 대해서는 [위키피디아](https://en.wikipedia.org/wiki/Digital_Signature_Algorithm) 내지는 [FIPS 186-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf)를 참고하도록 하고, 이 글에서는 원리 위주로 설명하려고 한다.

- 주의할 점으로, 아래에서 나오는 모든 $\mod p $들은 $p$로 나눈 나머지를 구하는 연산을 의미한다.

### 환경

우선 key length $L, N$을 정의한다. 이는 일반적으로 (1024, 160), (2048, 224), (2048, 256), (3072, 256) 중 하나이다.

메시지를 해싱할 함수 $H$를 정의한다. $H$의 output의 길이 $\mid H\mid$ 는 $N$ 이상이어야 한다.

$L, N$을 정하면 $N$-bit 소수 $q$를 구하고, $p - 1$이 $q$의 배수가 되는 $L$-bit 정수 $p$를 구한다.

2에서 $p$-2 사이에서 정수 $h$를 고르고, $g = h^{(p-1)/q} \mod p$를 계산한다. $g = 1$이면 다시 계산한다. 일반적으로 $h = 2$를 고른다.

이렇게 전부 고르면, 서명의 parameter로 $(p, q, g)$가 쓰이게 된다.

### Key Generation

키 생성은 매우 간단하다. 1에서 $q$-1 사이의 정수를 하나 임의로 골라 해당 값을 비밀키 $x$로 사용하고, 공개키 $y$는 $y = g^x \mod p$로 계산한다.

중요한 점은 $y$ 값으로부터 $x$ 값을 쉽게 구하지 못한다는 것이다. 이는 이산 로그 문제(Discrete Lograithm Problem)으로도 널리 알려져 있다.

### Sign

1. 키 생성과 비슷하게, 1에서 $q$-1 사이의 정수를 하나 임의로 골라 이를 $k$로 한다. 이 $k$값은 random nonce로 불리기도 하는데, 뒤에서 서술하는 nonce reuse의 nonce이기도 하다.
   $k$를 고르면, $r$과 $s$ 두 값을 만들면 끝난다.
2. $r$은 $r = (g^k \mod p) \mod q$를 통해 구한다. $r = 0$일 경우 $k$를 다시 고른다.
3. $s$는 $s = (k^{-1} (H(m) + xr)) \mod q$를 통해 구한다. 역시 $s = 0$일 경우 $k$를 다시 고른다.

이렇게 나온 $(r, s)$ 를 서명으로 사용한다.

### Verify

이제 서명을 검증을 해보자.

1. $w = s^{-1} \mod q$를 계산한다.
2. $u_1 = H(m) \cdot w \mod q$, $u_2 = r \cdot w \mod q$를 계산한다.
3. $v = (g^{u_1} y^{u_2} \mod p) \mod q$를 계산한다.
4. $v = r$일 경우 서명이 성립한다.

### Correctness of the algorithm

앞서 $g$를 $h^{(p-1)/q} \mod p$로 정의했으므로, $g^q = h^{(p-1)} = 1 \mod p$이다. 그러므로 $a = b (\text{mod}\ q)$ 이라면 분명 $g^a = g^b (\text{mod}\ p)$ 여야 한다.

만약 올바른 값이 들어왔다면, $s = (k^{-1} (H(m) + xr)) \mod q$가 성립해야 하고, 이는 곧 $k = s^{-1} (H(m) + xr) (\text{mod}\ q)$를 의미한다.

곧,

$g^k = g^{H(m)s^{-1}} \cdot g^{xrs^{-1}} = g^{H(m)s^{-1}} \cdot y^{rs^{-1}} = g^{u_1} y^{u_2} (\text{mod}\ p)$

가 성립한다.

그러므로, $r = (g^k \mod p) \mod q = (g^{u_1} y^{u_2} \mod p) \mod q = v$가 성립하며, 곧 알고리즘이 성립한다.

## Nonce reuse

Nonce reuse는 말 그대로, sign 과정에서 쓰인 $k$ 값을 다시 사용하는 것을 의미한다. DSA의 경우 nonce reuse를 통해서 비밀키를 얻을 수 있다.

$k$ 값이 정확히 뭔지 모르나 $k$ 값이 같을 때 두 서명 $(r, s_1), (r, s_2)$ 가 있다고 하자.

$s_1 - s_2 = k^{-1} (H(m_1) + xr - H(m_2) - xr) = k^{-1} (H(m_1) - H(m_2)) (\text{mod}\  q)$

이므로,

$k = (H(m_1) - H(m_2)) / (s_1 - s_2) (\text{mod}\ q)$

를 통해 $k$를 복구할 수 있게 된다. $k$ 값을 알게 되었으므로, $s_1, r, H(m_1), k$로부터 $x$를 복구할 수 있게 된다.

### Nonce reuse가 왜 문제일까?

보통의 경우라면 당연히 random nonce가 같도록 구현하는 사람은 없을 것이라고 생각할 것이다. 하지만 놀랍게도 random nonce에 salt, message를 같이 hash function에 넣는 구현을 **블록체인**에서 종종 살펴볼 수 있다.

일반적으로 이는 다음과 같은 이유 때문이다:

1. Random nonce를 만들 때 어떤 Random을 사용해야 할지 명확하지가 않다. 세상에 Uniform하면서 예측 불가능한 random 함수라는 것은 존재하지 않기 때문이다.
   대신 여러 유사 난수 생성기를 사용해볼 수 있겠지만, 해당 부분에서 취약점이 발견되면 어떻게 할 것인가?
2. 그래서 Hash 함수를 사용하면 Hash collision이 발생해야지만 된다. 블록체인 자체가 Hash function에 의지하고 있는 구조이므로, 차피 Hash collision이 발생하면 블록체인이 깨지게 된다.

하지만 좋은 선택지라고 할 수 있는 지는 의문이다. 경우에 따라서 유사 난수 생성기의 seed value를 [urandom](https://en.wikipedia.org/wiki//dev/random)과 같은 함수를 통해 생성한 값으로 설정해서 사용하는 것도 쉽게 찾아볼 수 있다.

### 더 나아가서

DSA가 아닌 전자 서명도 많이 존재한다. [Schnorr 서명](https://en.wikipedia.org/wiki/Schnorr_signature)이나 ECDSA가 대표적인 예인데, 그 중 Schnorr 서명의 경우 간편하고 가벼워 블록체인에서 많이 쓰이는 서명 알고리즘 중 하나이다.

Schnorr 서명은 random nonce가 완전히 같지 않아도 random nonce의 LSB가 일부 동일할 경우 비밀 키를 일부 알아낼 수 있는 공격 기법이 존재한다. ([링크](https://ecc2017.cs.ru.nl/slides/ecc2017-tibouchi.pdf))

# 구현

Nonce reuse를 체크해보기 위한 목적이기 때문에 parameter 값을 다소 줄여서 구현했다.

Python을 통해서 구현하며, PyCryptodome과 gmpy2 라이브러리를 사용한다.

## 환경 설정 및 키 생성

우선 환경 설정을 간단하게 해보자.

```python
from Crypto.Util import number
import gmpy2
import random

while True:
    q = number.getPrime(128)
    t = random.randrange(2**128, 2**129)
    p = t * q + 1
    if gmpy2.is_prime(p):
        break

g = pow(2, t, p)

print("p=", p)
print("q=", q)
print("g=", g)

x = random.randrange(2, q - 1)
y = pow(g, x, p)

print("x=", x)
print("y=", y)
```

$p - 1$이 $q$의 배수여야 하므로 처음부터 random한 value $t$를 통해 $tq  + 1$이 소수인지 검증하는 것이 편하다. 소수를 검증하는 방법에 대해서는 [이 블로그 글](http://www.secmem.org/blog/2019/06/17/PrimeNumber/)를 참고하도록 하자.

## 서명 및 검증

구현의 편의성을 위해 MD5를 사용했다.

```python
# 생략
import hashlib

param = (p, q, g)

def hash(m):
    hsh = hashlib.md5(m.encode()).hexdigest()
    return int(hsh, 16)

def sign(param, x, m):
    p, q, g = param
    while True:
        k = random.randint(1, q - 1)
        r = pow(g, k, p) % q
        if r == 0:
            continue
        s = gmpy2.invert(k, q) * (hash(m) + x * r) % q
        if s == 0:
            continue
        return (r, s)

def verify(param, y, m, sig):
    p, q, g = param
    r, s = sig
    w = gmpy2.invert(s, q)
    u1 = hash(m) * w % q
    u2 = r * w % q
    v = ( pow(g, u1, p) * pow(y, u2, p) % p ) % q
    return v == r

msg = "testtesttest"

sig = sign(param, x, msg)
print(sig)
print(verify(param, y, msg, sig))
```

실행해보면 정상적으로 나오는 것을 확인할 수 있다.

```
('p=', 90890473100806502209633626649256124920658829561999228421969081266437996326563L)
('q=', 227294674667840785251370091774332876291L)
('g=', 83611091912362920352531663660114702915390136278820307425860945160511486602316L)
('x=', 166410020120349346168182592340744837859L)
('y=', 81412179961039272174520583470749818861446746395524658435938860149677827700674L)
(223686346388008772767437265575324053029L, 141448019585108684871512092277589442512L)
True
```

## Nonce reuse

이제 다음과 같이 `sign_nonce`를 정의하자.

```python
def sign_nonce(param, x, m, k):
    p, q, g = param
    r = pow(g, k, p) % q
    if r == 0:
        return None
    s = long ( gmpy2.invert(k, q) * (hash(m) + x * r) % q )
    if s == 0:
        return None
    return (r, s)
```

두 가지 메시지에 대해서 같은 nonce로 sign한 뒤 위에 언급된 식으로 복구해보자.

```python
msg1 = "testtesttest"
msg2 = "wowwowwowwow"

nonce = random.randint(1, q - 1)
r1, s1 = sign_nonce(param, x, msg1, nonce)
r2, s2 = sign_nonce(param, x, msg2, nonce)

k = long((hash(msg1) - hash(msg2)) * gmpy2.invert(s1 - s2, q) % q)
print("nonce= ", nonce)
print("recov= ", k)
```

결과는 다음과 같이 같게 나온다.

```
('p=', 117459278754153661182508877877140946297134031725089062803855247167036101719467L)
('q=', 173259219392685643180537146421514781919L)
('g=', 4617892938984277053799598676502901321285192167628858021927259171947948684900L)
('x=', 155176436562786051719216230144459592234L)
('y=', 66073472909714257388454484532460477110940276815095536320070409777007543078772L)
('nonce= ', 7938953579620221034755555489784667677L)
('recov= ', 7938953579620221034755555489784667677L)
```

여기서 더 나아가, 비밀키를 복구해보자.

```python
x_recov = long(( s1 * k - hash(msg1) ) * gmpy2.invert(r1, q) % q)
print("x_recov= ", x_recov)
```

역시 비밀키도 잘 구해진다.

```
('x=', 152666895057751117878897057402166919985L)
('y=', 18163801915042524226206931051364567119681139372902088439991166513525360287247L)
('nonce= ', 97354161698547862727916496044386824318L)
('recov= ', 97354161698547862727916496044386824318L)
('x_recov= ', 152666895057751117878897057402166919985L)
```

# 마무리
가장 대표적인 전자 서명 방식인 DSA의 작동 원리와 nonce reuse의 위험성에 대해서 알아봤다. 구현할 때 해당 실수를 쉽게 범하지 않도록 주의하는 자세가 필요하겠다.

또한 관련 문제를 풀어보고 싶다면, Samsung CTF 2018 Final에 나온 [Salty DSA](https://github.com/kaishack/sctf2018/tree/master/crypto/SaltyDSA) 문제를 풀어보는 것을 추천한다.
