---
layout: post
title:  "Rabin-Karp 해싱의 충돌쌍 찾기"
date:   2019-12-15 23:50:00
author: blisstoner
tags: [Algorithm, Cryptography]]
---

안녕하세요, 이번 글에서는 Rabin-Karp 해싱의 충돌쌍을 찾는 다양한 방법에 대해 알아보겠습니다.

# Rabin-Karp 해싱이란?

Rabin-Karp 해싱은 문자열의 해쉬함수입니다. 이 글의 내용에서 알 수 있듯이 암호학적으로 안전한 함수는 아니지만 $$O(N)$$의 전처리를 거치고 나면 문자열 내의 임의의 substring의 해쉬 값을 $$O(1)$$에 알 수 있다는 특성 덕분에 해당 특성이 필요할 때 활용하면 효과적입니다.

또한 구현이 쉬운 편이기 때문에 굳이 암호학적으로 안전한 함수가 필요없는 상황일 경우에는 Java와 기본 String hashcode를 포함한 다양한 곳에서 사용되고 있습니다.

해싱에 대한 자세한 설명은 [제 블로그 글](https://blog.encrypted.gg/857)을 참고해주세요.

코드는 아래와 같이 아주 간단하게 만들 수 있습니다.

```python
def RabinKarp(S, p, m):
  h = 0
  for ch in S:
    h = (h * p + ord(ch)) % m
  return h
```

서술의 편의를 위해 글에서도 곱해지는 값과 modulo를 언급할 때 각각 $$p$$, $$m$$으로 기술하겠습니다.

# 충돌쌍 찾기 1 - Birthday Attack

해쉬함수의 값의 종류가 $$n$$개일 때 $$n+1$$개의 입력에 대해 해쉬값을 확인하면 비둘기집의 원리에 따라 반드시 충돌쌍이 존재함은 널리 알려진 사실입니다.

그런데 [Birthday Paradox](https://en.wikipedia.org/wiki/Birthday_problem)에 의해 실제로는 대략 $$n^{0.5}$$개의 입력에 대해서만 해쉬값을 확인해도 대략 50% 정도의 확률로 충돌쌍을 찾아낼 수 있습니다.

그렇기에 Rabin-Karp 해싱에서 $$m$$이 $$2^{50}$$ 이하 정도로 그다지 크지 않을 경우 합리적인 시간 안에 충돌쌍을 찾아낼 수 있습니다.

```python
import random
def RabinKarp(S, p, m):
  h = 0
  for ch in S:
    h = (h * p + ord(ch)) % m
  return h

charSet = 'abcdefghijklmnopqrstuvwxyz'

def Collision1(p, m):
  D = dict()
  while True:
    slen = random.randint(20,30)
    s = ''.join([random.choice(charSet) for i in range(slen)])
    h = RabinKarp(s, p, m)
    if h in D:
      if D[h] != s:
        return D[h], s
    D[h] = s
```

# 충돌쌍 찾기 2 - Weak p

라빈 카프 함수에서 $$p, m$$이 서로소일 때 오일러-파이 정리에 의해 $$p^{\phi(m)} \equiv 1 (mod \; m)$$ 입니다. 이 말은 곧 라빈 카프 함수가 $$\phi(m)$$을 주기로 같은 값이 곱해짐을 의미합니다.

그렇기에 예를 들어 $$m = 7$$일 때 `axxxxxxb`와 `bxxxxxxa`는 동일한 해쉬 값을 가지게 됩니다.

하지만 $$m$$이 크면 클수록 $$\phi(m)$$ 또한 굉장히 커져 적어도 $$\phi(m)+1$$ 개의 문자열이 있어야 충돌쌍을 만들어낼 수 있는 상황에서는 크게 의미가 없습니다. $$m$$이 그다지 크지 않다면 그냥 birthday attack을 사용하면 되기 때문입니다.

그런데 이 방법은 $$Z_p$$에서 $$p$$의 order가 그다지 크지 않을 때 유용합니다.

예를 들어 $$p = 3054, m = 14116281909712281624732079749048115$$일 때 $$p^{10} \equiv 1 (mod \; m)$$ 입니다. 그렇기에 `axxxxxxxxxb`와 `bxxxxxxxxxa`는 동일한 해쉬 값을 가지게 됩니다.

이 공격은 $$p$$의 order가 그다지 크지 않을 때에만 유효한 방법으로, 일반적으로는 이러한 공격이 잘 먹히지 않을 것입니다.

# 충돌쌍 찾기 3 - Characteristic Polynomial

이 방식은 [ho94949님의 블로그](https://blog.kyouko.moe/29)에서 이미 설명이 나와있는 방식입니다.

C, Java와 같이 기본 자료형의 크기가 제한되어있을 경우에는 $$m$$을 $$2^{32}$$ 혹은 $$2^{64}$$로 둔다면 아예 나머지를 구하는 연산이 필요가 없어지기 때문에 구현의 편의 혹은 속도의 개선을 위해 $$m$$을 $$2^{32}$$나 $$2^{64}$$로 두는 경우가 종종 있습니다.

그런데 이 경우 충돌쌍을 쉽게 만들어낼 수 있습니다. 바로 임의의 홀수 $$x$$에 대해 $$(1-x^{2^N})$$이 $$2^N$$의 배수라는 성질을 이용하는 것입니다.(이 성질은 귀납적으로 증명할 수 있습니다.)

우선 $$(1-x)(1-x^2)(1-x^4)(1-x^8)(1-x^{16})(1-x^{32})(1-x^{64})(1-x^{128})(1-x^{256})$$을 생각해봅시다. 이 다항식은 512차 다항식이고 임의의 홀수에 대해 $$2^{36}$$의 배수입니다.

또한 전개를 했을 때 $$x^k$$의 항은 $$k$$의 parity, 즉 $$k$$를 이진수로 표현했을 때 1이 짝수개일 경우 1, 1이 홀수개일 경우 -1이 됩니다.

이 때 $$m$$이 $$2^{32}$$이면 모든 홀수 $$p$$에 대해 길이 512인 문자열을 두고 각 자리의 parity에 따라 문자 a 혹은 c를 넣어 마치 다항식의 계수를 -1 혹은 1로 두는 효과를 줄 수 있습니다. 이를 이용해 `bbbbb...bbb`와 ``caaca...cca`의 해쉬 충돌을 발생시킬 수 있습니다.

만약 $$m$$이 $$2^{64}$$이면$$(1-x)(1-x^2)(1-x^4)(1-x^8)(1-x^{16})(1-x^{32})(1-x^{64})(1-x^{128})(1-x^{256})(1-x^{512})(1-x^{1024})$$를 이용하면 됩니다.

```python
def parity(x):
  ret = 0
  while x:
    ret ^= (x&1)
    x >>= 1
  return ret

# m = 2**32
def Collision3_32():
  s1 = ''
  s2 = 'b'*512
  p = 243423223425 # any odd
  m = 2**32
  val = 0
  for i in range(512):
    if parity(i):
      s1 += 'a'
    else:
      s1 += 'c'
  print(s1)
  print(RabinKarp(s1,p,m) == RabinKarp(s2,p,m))

# m = 2**64
def Collision3_64():
  s1 = ''
  s2 = 'b'*2048
  p = 98743212351231 # any odd
  m = 2**64
  val = 0
  for i in range(2048):
    if parity(i):
      s1 += 'a'
    else:
      s1 += 'c'
  print(RabinKarp(s1,p,m) == RabinKarp(s2,p,m))
```

이 방식은 $$m$$이 $$2^{32}$$ 혹은 $$2^{64}$$일 때에는 강력하지만 임의의 다른 자연수일 때에는 적절한 characteristic polynomial을 찾기가 힘들어 활용하기가 까다로운 방식입니다.

# 충돌쌍 찾기 4 - LLL algorithm

마지막으로 소개할 방법은 [LLL algorithm](https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm)을 이용하는 방법입니다. 이 알고리즘의 full name은 `Lenstra–Lenstra–Lovász (LLL) lattice basis reduction algorithm`입니다.

이 알고리즘을 아는 분들은 이미 잘 알고계실 것이고, 아예 모르시는 분들을 대상으로 이 글에서 짧게 설명하기에는 다소 난해한 내용입니다. 그나마 쉽게 설명이 되어있는 포스팅 2개의 링크를 첨부할테니 만약 이 알고리즘을 처음 접해보았다면 일단 한번 포스팅을 참고해보시고, 그래도 도저히 이해가 가지 않는다면 이해는 다음으로 미뤄두고 이 글에서 제시할 공격 방법만 가져가시는 방식을 추천드립니다. [포스팅 1 - ebmoon님](https://eyebrowmoon.github.io/hacking/crypto/rsa/2019/05/23/RSA_Attack_Using_LLL.html), [포스팅 2 - Rbtree님](http://www.secmem.org/blog/2019/11/15/On-Factoring-Given-Any-Bits/)

LLL algorithm을 이용하면 $$m$$이 $$2^{128}$$ 정도로 크더라도 대략 길이 50 이내의 충돌쌍을 아주 쉽게, 그리고 많이 만들어낼 수 있습니다.

길이가 같은 두 문자열이 있다고 할 때 각 자리에서 두 문자열의 차를 생각해봅시다. 예를 들어 "ccc"와 "adb"일 경우 차이는 $$2, -1, 1$$입니다. 그리고 문자열의 길이를 $$n$$이라고 하겠습니다.

이 때 $$d_1 \cdot p^{n-1} + d_2 \cdot p^{n-2} + \dots + d_n \cdot p^{0} = 0$$ 이면서 모든 $$d_i$$가 25 이하인 적절한 수열 $$d$$를 찾는다면 $$d$$를 두 문자열의 차로 두어 해쉬값이 일치하는 두 문자열을 얻을 수 있습니다.

이제 그러면 저 수열 $$d$$를 어떻게 찾냐가 문제인데, 놀랍게도 LLL 알고리즘을 이용해 이를 해결할 수 있습니다.

LLL 알고리즘을 통해 우리는 행렬에서 norm이 bound된 vector를 얻어낼 수 있습니다. 이 때 아래와 같은 $$N\;by\;N+1$$ 행렬을 만든다면 어떨까요?

$$\begin{matrix} 1 & 0 & 0 & \dots & 0 & B \cdot p^{n-1} \; mod \; m \\ 0 & 1 & 0 & \dots & 0 & B \cdot p^{n-2} \; mod \; m \\ 0 & 0 & 1 & \dots & 0 & B \cdot p^{n-3} \; mod \; m \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & \dots & 1 & B \cdot p^0 \; mod \; m  \end{matrix}$$

이 때 $$B$$는 100000 정도로 적당히 큰 상수입니다.

이 행렬에서 LLL 알고리즘을 적용해 row끼리 연산을 하다 보면 자연스럽게 $$v[1] \cdot p^{n-1} + v[2] \cdot p^{n-2} + \dots + v[N] \cdot p^{0} = v[N+1]$$ 을 만족하게 되고, $$N+1$$번째 열에 상수를 전부 곱해놓았기 때문에 자연스럽게 $$v[N+1] = 0$$을 만족하면서 $$abs(v[1 \dots N])$$ 이 작아지려는 방향으로 구해질 것입니다.

마침 sage에 LLL algorithm이 구현되어 있기에 이를 이용해 코드를 작성할 수 있습니다.

([이 곳](https://galhacktictrendsetters.wordpress.com/2017/09/05/tokyo-westerns-ctf-2017-palindromes-pairs-challenge-phase/)의 코드를 수정했습니다.)

```python
p = 19738112312161
m = 2**128 + 2**64 + 2**32 + 2**16 + 1

N = 30

def sparse_poly():
  L_rows = []
  B = 100000

  for i in xrange(N):
    row = [1 if j==i else 0 for j in xrange(N)] + [B*Integer(pow(p,N+1-i,m))]
    L_rows.append(row)

  L_matrix = Matrix(L_rows)
  L_redux = L_matrix.LLL()
  sparse = L_redux[0]
  print sparse[:-1]
  poly, vals = sparse[:N], sparse[N:]

  assert all(abs(x) <=25 for x in poly)
  assert all(x == 0 for x in vals)

  return poly

def build_strings(poly):
  a = [0 for _ in xrange(N)]
  b = [0 for _ in xrange(N)]

  # a[i] - b[(N-1)-i] = poly[i]

  for i in xrange(N):
    if poly[i] >= 0:
      a[i] = poly[i]
      b[i] = 0
    else:
      a[i] = 0
      b[i] = -poly[i]

  a_str = ''.join(chr(ord('a')+v) for v in a)
  b_str = ''.join(chr(ord('a')+v) for v in b)
  return a_str, b_str

def solve():
  poly = sparse_poly()
  s1, s2 = build_strings(poly)
  print s1, s2

solve()
```

코드에서 $$p, m$$ 값을 임의로 바꿀 수 있고, $$p = 19738112312161
m = 2^{128} + 2^{64} + 2^{32} + 2^{16} + 1$$일 때 `aaaaaaaaaeaaaoafagfagcgaaanaei`, `dcefdbfdcagaoaeaaaadaaabfbalaa` 라는 길이 30짜리 충돌쌍을 얻어낼 수 있었습니다.

참고로 $$N$$이 크면 클수록 vector의 값들이 작아집니다.

# 마무리

이번 글에서는 Rabin-Karp 해싱의 충돌쌍을 찾아보았습니다. 글에서 알 수 있듯 Rabin-Karp는 암호학적으로 안전하지 않은 해쉬함수이기 때문에 충돌쌍이 발견될 경우 악의적인 공격자가 성능을 저하시키거나 DDOS를 유발할 수 있는 상황에서는 Rabin-Karp 해싱을 사용하는 대신 `sha256`과 같은 안전한 해쉬함수를 사용하는 것이 좋습니다.
