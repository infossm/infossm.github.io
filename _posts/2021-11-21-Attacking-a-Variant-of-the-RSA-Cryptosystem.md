---
layout: post
title:  "Attacking a Variant of the RSA Cryptosystem"
date:   2021-11-21 23:59
author: RBTree
tags: [cryptography]
---

# 서론

이번 글은 저번 달의 글에 이어서 [pbctf 2021](https://ctftime.org/event/1371)를 출제하면서 사용한 논문 하나를 리뷰하려고 합니다. 이 논문은 [Yet Another RSA](https://github.com/perfectblue/pbCTF-2021-challs/tree/master/crypto/yet-another-rsa)라는 문제로 작성하게 되었습니다.

해당 논문의 제목은 Classical Attacks on a Variant of the RSA Cryptosystem인데요, RSA의 변종에 대해 $d$가 작을 경우의 공격법을 다루는 논문입니다.

![Yet Another RSA](/assets/images/rbtree/rsa_meme.jpg)

# 본론

## RSA Variant

이 논문에서 소개하는 RSA의 변종은 정수가 아닌 group 위에서 정의됩니다. 해당 그룹에 대해서 정의하자면 다음과 같습니다.

Field $(\mathbb{F}, +, \cdot)$에 대해서 non-cubic integer $r \in \mathbb{F}$를 하나 뽑읍시다. 그러면 $t^3 - r$은 $\mathbb{F}[t]$ 위에서 irreducible 하므로, quotient field $\mathbb{A} = \mathbb{F}[t]/(t^3-r)$를 정의할 수 있습니다.

이제 이를 바탕으로 quotient group $B = \mathbb{A}^*/\mathbb{F}^*$를 생각하면, 이 위의 원소는 $m + nt + t^2$, $m + t$, 혹은 1의 꼴을 가짐을 알 수 있습니다. 그러므로 $B$를 다음과 같이 나타낼 수도 있습니다.

$B = (\mathbb{F} \cross \mathbb{F}) \cup (\mathbb{F} \cross \{\alpha\}) \cup (\{\alpha, \alpha\})$

이 때 이 위에서 operation $\odot$을 다음과 같이 정의합시다.

- $(m, \alpha) \odot (p, \alpha) = (mp, m + p)$
- If $n+p \neq 0$, then $(m, n) \odot (p, \alpha) = (\frac{mp+r}{n+p}, \frac{m+np}{n+p})$
- If $n+p=0$, and $m - n^2 \neq 0$, then $(m, n) \odot (p, \alpha) = (\frac{mp+r}{m-n^2}, \alpha$
- If $n + p = 0$, and $m - n^2 = 0$, then $(m, n) \odot (p, \alpha) = (\alpha, \alpha)$
- If $m+p+nq \neq 0$, then $(m, n) \odot (p, q) = (\frac{mp+(n+q)r}{m+p+nq}, \frac{np+mq+r}{m+p+nq})$
- If $m + p + nq = 0$ and $np + mq + r \neq 0$, then $(m, n) \odot (p, q) = (\frac{mp+(n+q)r}{m+p+nq}, \alpha)$
- If $m+p+nq=0$ and $np+mq+r=0$, then $(m, n) \odot (p, q) = (\alpha, \alpha)$

$(B, \odot)$은 commutative group이면서, 소수 $p$에 대해서는 $B_p$는 $p^2+p+1$ order를 가지는 cyclic group이 됩니다. 이에 대해서 자세히 알고 싶은 분은 [이 논문](https://www.semanticscholar.org/paper/A-Novel-RSA-Like-Cryptosystem-Based-on-a-of-the-Murru-Saettone/d2d0008d5e911993c04233f88545ca2256395017)을 참고해 주세요. 고로, $B_p$ 위에서 $(m, n)^{\odot (p^2+p+1)} = (\alpha, \alpha) \pmod p$가 성립합니다.

이 때 이를 $N = pq$에 대해 $B_N$을 생각하면, order가 $(p^2 + p + 1)(q^2 + q + 1)$인 group이 됩니다. 고로 RSA와 비슷하게 $e$를 정의한 뒤 $d \equiv e^{-1} \pmod{(p^2+p+1)(q^2+q+1)}$ 를 정의하면, RSA와 같이 사용하는 것이 가능합니다.

이를 이제 코드로 구현해봅시다. $r = 2$로 정의했습니다.

```python
def genPrime():
    while True:
        a = random.getrandbits(256)
        b = random.getrandbits(256)

        if b % 3 == 0:
            continue

        p = a ** 2 + 3 * b ** 2
        if p.bit_length() == 512 and p % 3 == 1 and isPrime(p):
            return p

def add(P, Q, mod):
    m, n = P
    p, q = Q

    if p is None:
        return P
    if m is None:
        return Q

    if n is None and q is None:
        x = m * p % mod
        y = (m + p) % mod
        return (x, y)

    if n is None and q is not None:
        m, n, p, q = p, q, m, n

    if q is None:
        if (n + p) % mod != 0:
            x = (m * p + 2) * inverse(n + p, mod) % mod
            y = (m + n * p) * inverse(n + p, mod) % mod
            return (x, y)
        elif (m - n ** 2) % mod != 0:
            x = (m * p + 2) * inverse(m - n ** 2, mod) % mod
            return (x, None)
        else:
            return (None, None)
    else:
        if (m + p + n * q) % mod != 0:
            x = (m * p + (n + q) * 2) * inverse(m + p + n * q, mod) % mod
            y = (n * p + m * q + 2) * inverse(m + p + n * q, mod) % mod
            return (x, y)
        elif (n * p + m * q + 2) % mod != 0:
            x = (m * p + (n + q) * 2) * inverse(n * p + m * q + r, mod) % mod
            return (x, None)
        else:
            return (None, None)


def power(P, a, mod):
    res = (None, None)
    t = P
    while a > 0:
        if a % 2:
            res = add(res, t, mod)
        t = add(t, t, mod)
        a >>= 1
    return res
```

`genPrime`의 경우 $r$이 non-cubic임을 보장하기 위해서 해당 꼴과 같이 만들었습니다.

## Attacking the RSA Variant

자, 이제 $d$가 작을 때 어떻게 공격할지 생각해봅시다. 이 글에서 초점을 맞추고 싶은 것은 **이런 RSA의 변종에서 기존의 방법을 응용해서 어떻게 풀 수 있을지** 입니다.

이를 위해서는 원래 RSA에서의 유명한 공격인 Boneh-Durfee를 살펴볼 필요가 있습니다. Boneh-Durfee에서는 다음과 같은 식 유도 과정을 거칩니다.

$ed \equiv 1 \pmod{\phi(N)}$

$\implies ed \equiv k\phi(N)+1$ for some $k$

$\implies k\phi(N) + 1 = k(N+1-p-q) + 1 \equiv 0 \pmod{e}$

이를 바탕으로 $f(x, y) = 1 + x(A+y)$ 라는 이변수다항식을 푸는 문제로 생각할 수가 있습니다.

이제 위에서 정의한 RSA의 변종으로 넘어와봅시다. 식 $ed - k(p^2+p+1)(q^2+q+1) = 1$에 대해서 위와 동일한 과정을 거칠 수 있을까요?

우선 $(p^2+p+1)(q^2+q+1)$의 꼴이 위와 같이 변형하기 힘들어보입니다. 위에서는 $p+q = y$와 같이 변형합니다. 해당 식을 전개해봅시다.

$(p^2+p+1)(q^2+q+1) = p^2q^2 + p^2q + p^2 + pq^2 + pq + p + q^2 + q + 1$

우리는 $N = pq$임을 알고 있으므로, 다음과 같이 다시 전개할 수 있습니다.

$ = N^2 + N(p+q) + p^2 + q^2 + N + p + q + 1$

$ = (p+q)^2 + (N+1)(p+q) + N^2 - N + 1$

그러므로, $f(x, y) = x(y^2+ay+b) + 1$라는 변형 이변수 다항식을 생각해볼 수 있습니다. 이 때 Coppersmith를 Boneh-Durfee와 같이 적용해볼 수 있습니다. 보통 $m, t$라는 변수 두 개를 잡고 lattice의 크기를 구성하는데, 위의 변형에서는 $m, t$를 다음과 같이 사용합니다.

$g_{k,i,j}(x,y) = x^{i-k} y^{j-2k} f(x,y)^k e^{m-k}$ for $k = 0, \dots, m, i = k, \dots, m, j = 2k, 2k+1$ or $k = 0, \dots, m, i = k, j = 2k + 2, \dots, 2i + t$

$x = N^{\gamma}$일 경우, $X = N^\gamma, Y = 2\sqrt{2} N^{\frac{1}{2}}$에 대해서 $g_{k,i,j}(xX,yY)$를 바탕으로 다음과 같이 lattice를 구성할 수 있습니다.

![](/assets/images/rbtree/rsa_lattice.png)

이 때 lattice의 size에 따른 제한 조건은 논문을 참고하시면 되겠습니다. 다만, 주의할 부분이 있습니다. 논문의 제한 조건의 경우 증명 과정이 매우 생략되어있고, 간략화 과정이 많이 들어가 있습니다. **실제로 본인의 조건에서 사용 가능한지는 lattice의 determinant를 직접 계산해보시기를 권장합니다.**

이제 이를 바탕으로 Yet Another RSA 문제에 대한 공격 코드를 작성해봅시다.

##Yet Another RSA

pbctf에 나온 Yet Another RSA의 코드를 살펴봅시다.

```python
#!/usr/bin/env python3

from Crypto.Util.number import *
import random


def genPrime():
    while True:
        a = random.getrandbits(256)
        b = random.getrandbits(256)

        if b % 3 == 0:
            continue

        p = a ** 2 + 3 * b ** 2
        if p.bit_length() == 512 and p % 3 == 1 and isPrime(p):
            return p


def add(P, Q, mod):
    m, n = P
    p, q = Q

    if p is None:
        return P
    if m is None:
        return Q

    if n is None and q is None:
        x = m * p % mod
        y = (m + p) % mod
        return (x, y)

    if n is None and q is not None:
        m, n, p, q = p, q, m, n

    if q is None:
        if (n + p) % mod != 0:
            x = (m * p + 2) * inverse(n + p, mod) % mod
            y = (m + n * p) * inverse(n + p, mod) % mod
            return (x, y)
        elif (m - n ** 2) % mod != 0:
            x = (m * p + 2) * inverse(m - n ** 2, mod) % mod
            return (x, None)
        else:
            return (None, None)
    else:
        if (m + p + n * q) % mod != 0:
            x = (m * p + (n + q) * 2) * inverse(m + p + n * q, mod) % mod
            y = (n * p + m * q + 2) * inverse(m + p + n * q, mod) % mod
            return (x, y)
        elif (n * p + m * q + 2) % mod != 0:
            x = (m * p + (n + q) * 2) * inverse(n * p + m * q + r, mod) % mod
            return (x, None)
        else:
            return (None, None)


def power(P, a, mod):
    res = (None, None)
    t = P
    while a > 0:
        if a % 2:
            res = add(res, t, mod)
        t = add(t, t, mod)
        a >>= 1
    return res


def random_pad(msg, ln):
    pad = bytes([random.getrandbits(8) for _ in range(ln - len(msg))])
    return msg + pad


p, q = genPrime(), genPrime()
N = p * q
phi = (p ** 2 + p + 1) * (q ** 2 + q + 1)

print(f"N: {N}")

d = getPrime(400)
e = inverse(d, phi)
k = (e * d - 1) // phi

print(f"e: {e}")

to_enc = input("> ").encode()
ln = len(to_enc)

print(f"Length: {ln}")

pt1, pt2 = random_pad(to_enc[: ln // 2], 127), random_pad(to_enc[ln // 2 :], 127)

M = (bytes_to_long(pt1), bytes_to_long(pt2))
E = power(M, e, N)

print(f"E: {E}")
```

Flag를 절반으로 나눈 뒤, `random_pad`를 통해서 늘리는 과정을 거쳐 그대로 사용하는 것을 볼 수 있습니다. Output은 다음과 같습니다.

```
N: 144256630216944187431924086433849812983170198570608223980477643981288411926131676443308287340096924135462056948517281752227869929565308903867074862500573343002983355175153511114217974621808611898769986483079574834711126000758573854535492719555861644441486111787481991437034260519794550956436261351981910433997
e: 3707368479220744733571726540750753259445405727899482801808488969163282955043784626015661045208791445735104324971078077159704483273669299425140997283764223932182226369662807288034870448194924788578324330400316512624353654098480234449948104235411615925382583281250119023549314211844514770152528313431629816760072652712779256593182979385499602121142246388146708842518881888087812525877628088241817693653010042696818501996803568328076434256134092327939489753162277188254738521227525878768762350427661065365503303990620895441197813594863830379759714354078526269966835756517333300191015795169546996325254857519128137848289
> 
Length: 71
E: (123436198430194873732325455542939262925442894550254585187959633871500308906724541691939878155254576256828668497797665133666948295292931357138084736429120687210965244607624309318401630252879390876690703745923686523066858970889657405936739693579856446294147129278925763917931193355009144768735837045099705643710, 47541273409968525787215157367345217799670962322365266620205138560673682435124261201490399745911107194221278578548027762350505803895402642361588218984675152068555850664489960683700557733290322575811666851008831807845676036420822212108895321189197516787046785751929952668898176501871898974249100844515501819117)
```

이제 공격 코드를 작성해봅시다. 우선, 위에서 정의한 $g_{k,i,j}$를 구하고 monomial들을 정렬합니다.

```python
#!/usr/bin/env sage

from Crypto.Util.number import *

N = 144256630216944187431924086433849812983170198570608223980477643981288411926131676443308287340096924135462056948517281752227869929565308903867074862500573343002983355175153511114217974621808611898769986483079574834711126000758573854535492719555861644441486111787481991437034260519794550956436261351981910433997
e = 3707368479220744733571726540750753259445405727899482801808488969163282955043784626015661045208791445735104324971078077159704483273669299425140997283764223932182226369662807288034870448194924788578324330400316512624353654098480234449948104235411615925382583281250119023549314211844514770152528313431629816760072652712779256593182979385499602121142246388146708842518881888087812525877628088241817693653010042696818501996803568328076434256134092327939489753162277188254738521227525878768762350427661065365503303990620895441197813594863830379759714354078526269966835756517333300191015795169546996325254857519128137848289

P.<x, y> = PolynomialRing(ZZ)

m = 4
t = 2
X = int(N ^ 0.4)
Y = 3 * int(N ^ 0.5)

a = N + 1
b = N^2 - N + 1

f = x * (y^2 + a * y + b) + 1

gs = []

for k in range(m + 1):
    for i in range(k, m + 1):
        for j in range(2 * k, 2 * k + 2):
            g = x^(i - k) * y^(j - 2 * k) * f^k * e^(m - k)
            gs.append((i, j, k, g))
    i = k
    for j in range(2 * k + 2, 2 * k + t + 1):
        g = x^(i - k) * y^(j - 2 * k) * f^k * e^(m - k)    
        gs.append((i, j, k, g))

gs.sort()

monomials = []
for tup in gs:
    for v in tup[-1].monomials():
        if v not in monomials:
            monomials.append(v)
```

그리고 이를 바탕으로 matrix를 정의합시다. Monomial들을 정렬했으므로 쉽게 $g_{k,i,j}(xX,yY)$를 통해 matrix를 구성할 수 있습니다.

```python
mat = [[0 for j in range(len(monomials))] for i in range(len(gs))]

for i, tup in enumerate(gs):
    for j, mono in enumerate(monomials):
        mat[i][j] = tup[-1].monomial_coefficient(mono) * mono(X, Y)

mat = Matrix(ZZ, mat)
mat = mat.LLL()

pols = []

for i in range(len(gs)):
    f = sum(mat[i, k] * monomials[k] / monomials[k](X, Y) for k in range(len(monomials)))
    pols.append(f)
```

그리고 matrix를 통해 구한 polynomial로부터 solution을 구하는 코드를 작성합니다. 보통 논문에서는 이 부분을 생략하는데, 꼭 코드를 보고 이해하시는 과정을 거치면 좋겠습니다. (인터넷에 널리 퍼져있는 [Boneh-Durfee 공격 코드](https://github.com/mimoo/RSA-and-LLL-attacks/blob/master/boneh_durfee.sage)도 비슷하게 작성되어 있습니다.)

```python

found = False

for i in range(len(gs)):
    for j in range(i + 1, len(gs)):
        f1, f2 = pols[i], pols[j]

        rr = f1.resultant(f2)
        if rr.is_zero() or rr.monomials() == [1]:
            continue
        else:
            try:
                PR.<q> = PolynomialRing(ZZ)
                rr = rr(q, q)
                soly = int(rr.roots()[0][0])
                ss = f1(q, soly)
                solx = int(ss.roots()[0][0])

                print(i, j)
                print(solx, soly)
                assert f1(solx, soly) == 0
                assert f2(solx, soly) == 0

                found = True
            except:
                pass
        if found:
            break
    if found:
        break

b, c = soly, N
Dsqrt = int(sqrt(b^2 - 4*c))
p, q = (b + Dsqrt) // 2, (b - Dsqrt) // 2
assert p * q == N
```

이제 복구한 $p, q$를 바탕으로 flag를 복구하면 됩니다.

# 결론

이번 글을 읽으면서 꼭 이해했으면 하는 부분은 다음과 같습니다.

- Boneh-Durfee와 같은 공격을 알고 있으면 변종에 대해서도 공격을 스스로 정의해 볼 수 있다. (증명 과정은 약할지라도)
- 다른 공격 코드를 읽고 이해해서 스스로 공격을 구현해볼 수 있다.

그리고 혹시나 논문을 읽게 된다면, 다음을 유의하면서 보면 좋겠습니다.

- 논문에서는 해당 변종에 대해서 공격할 수 있는 small $d$의 범위가 늘어나기 때문에 RSA보다 더 취약하다고 주장합니다. 하지만 기존 RSA에 비해 group의 크기가 제곱으로 늘어났기 때문에, 이는 당연한 결과입니다.
- 논문의 증명 과정에서 생략된 부분이 많고, 잘못 유도된 식도 간간히 존재합니다. 이를 스스로 찾아보는 과정이 있으면 좋겠습니다.

개인적으로 좋은 논문이라고 생각하지는 않습니다. 하지만 기존 공격을 어떻게 응용하는지 배우기에는 적합한 논문이기 때문에, 암호에 관심이 있으시다면 꼭 한 번 읽어보시면 좋겠습니다.

# 참고 문헌

1. Classical Attacks on a Variant of the RSA Cryptosystem https://eprint.iacr.org/2021/1160.pdf
2. mimoo/RSA-and-LLL-attacks https://github.com/mimoo/RSA-and-LLL-attacks

