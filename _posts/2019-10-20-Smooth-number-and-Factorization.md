---
layout: post
title:  "Smooth number and Factorization"
date:   2019-10-20 22:00
author: RBTree
tags: [cryptography, number-theory, CTF]
---

# 서론

이번 HITCON CTF 2019 Quals에서 안타깝게 14등으로 본선을 진출하지 못하게 되었습니다. [결과 링크](https://ctf2019.hitcon.org/dashboard/scoreboard)

아쉬운 부분은 푼 팀이 적은 암호학 문제들을 풀지 못했다는 건데, 그 중 한 문제가 소인수분해와 관련이 있어 이번 포스팅을 작성하게 되었습니다.

# RSA

위에서 말한 문제는 Lost Key Again이라는, RSA와 관련된 문제입니다.

RSA에 대해서 간편하게 살펴보자면, $a \equiv b (\mathbb{mod}\ c)$는 $a$를 $c$로 나눈 나머지와 $b$를 $c$로 나눈 나머지가 같다는 뜻으로 이해할 수 있습니다. 편의상 $a$를 $b$로 나눈 나머지를 $a (\mathbb{mod}\ b)$로 표기하겠습니다. 이는 논문에서도 쓰이는 표현 방식입니다.

RSA에서 공개키는 $(e, N)$이고, 비밀키는 $(d, N)$입니다. RSA에서 암호화는 plaintext $m$에 대해서 $m^e (\mathbb{mod}\ N)$ 을 계산하는 것입니다. 복호화는 ciphertext $c$에 대해서 $c^d (\mathbb{mod}\ N)$을 계산하는 것입니다. 이를 정리해보면, $(m^e)^d \equiv m^{ed} \equiv m (\mathbb{mod}\ N)$이라는 뜻입니다. 이게 어떻게 성립할까요?

이런 과정이 성립하는 이유는 $e$와 $d$의 성질에 있습니다.

우선 오일러의 함수와 오일러의 정리부터 알아봅시다. 오일러 함수 $\phi(N)$ 는 1부터 $N$까지의 서로소인 수의 개수를 의미합니다. 이 때, $N$이 서로소인 $a, b$에 대해서 $N=ab$라면, $\phi(N) = \phi(a)\phi(b)$가 성립합니다. 또한, 소수 $p$에 대해서 $\phi(p) = p-1$이 성립합니다. 이 때, 오일러의 정리는 $N$과 서로소인 $a$에 대해서 $a^{\phi(N)} \equiv 1 (\mathbb{mod}\ N)$ 이라는 것입니다.

RSA에서는 매우 큰 소수 $p$와 $q$를 곱한 수를 $N$으로 정의해서 사용합니다. 이 때, $p$와 $q$를 알고 있다면 $\phi(N) = (p-1)(q-1)$을 쉽게 구할 수 있고, 이를 통해 어떤 수 $e$에 대해서 $ed = 1 (\mathbb{mod}\ \phi(N))$ 인 $d$를 구할 수 있습니다. ([Modular inverse](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse)) 그러면 $ed$는 어떤 정수 $k$가 있어서 $ed = k\phi(N) + 1$이라는 뜻이고, 곧 $m^{ed} \equiv m^{k\phi(N)+1} \equiv m (\mathbb{mod}\ N)$이라는 사실을 알 수 있습니다.

이 때 RSA의 안전성은 두 가지 요소에 기인합니다.

- $m^e (\mathbb{mod}\ N)$ 로부터 $m$을 구하기 어렵다.
- $d$를 구하면 위 상황에서 $m$을 쉽게 구할 수 있지만, $N$만으로는 $p$와 $q$ 값을 모르므로 $\phi(N)$ 값을 계산하지 못하니 $e$ 값을 알아도 $d$ 값을 구할 수 없다.

곧, $N$이 소인수분해하기 어렵다는 것이 기본 가정으로 깔려있어야 합니다.

# Challenge

Lost Key Again에서 제가 할 수 있는 일은 메시지를 보내면 그 메시지 앞에 `X: `를 붙인 뒤 암호화한 결과를 받는 것 뿐입니다. 이 문제에서는 심지어 공개키도 무엇인지 주어지지 않습니다.

문제의 코드는 다음과 같습니다.

```python
#!/usr/bin/env python
from Crypto.Util.number import *
import os,sys

sys.stdin  = os.fdopen(sys.stdin.fileno(), 'r', 0)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def read_key():
    key_file = open("key")
    n,e,d = map(int,key_file.readlines()[:3])
    return n,e,d

def calc(n,p,input):
  data = "X: "+input
  num = bytes_to_long(data)
  res = pow(num,p,n)
  return long_to_bytes(res).encode('hex')

def read_flag():
  flag = open('flag').read()
  assert len(flag) >= 50
  assert len(flag) <= 60
  prefix = os.urandom(68)
  return prefix+flag

if __name__ == '__main__':
  n,e,d = read_key()
  flag =  calc(n,e,read_flag())
  print 'Here is the flag!', flag
  for i in xrange(100):
    m = raw_input('give me your X value: ')
    try:
      m = m.decode('hex')[:15]
      print calc(n,e,m)
    except:
      print 'no'
      exit(0)
```

우리는 공개키 값 $(e, N)$조차 모르는데 무엇을 할 수 있을까요?

## Getting N

다행인 점은, 암호화된 결과로부터 $N$을 구할 수 있는 트릭이 있다는 점입니다.

우선, 세 메시지 `a1 = "X: \x00", a2 = "X: \x00\x00", a3 = "X: \x00\x00\x00"`에 대해서, `a1 * a3 = a2 * a2`가 성립합니다. (bytes_to_long은 big-endian으로 string을 integer로 변환합니다.)

우리가 `a1`, `a2`, `a3`을 각각 암호화한 결과를 받아왔다고 합시다. 이는 각각 `a1^e (mod N), a2^e (mod N), a3^e (mod N)`일 것입니다. 우리는 `a1^e * a3^e = a2^e * a2^e`라는 사실을 알고 있으며, 곧 `(a1^e (mod N)) * (a3^e (mod N))` 라는 값과 `(a2^e (mod N))^2` 가 $N$으로 나눈 나머지가 같다는 것을 알 수 있습니다.

그런데 이 두 값은 서로 완전히 같지는 않기 때문에, 두 값을 서로 뺀 값인 `(a1^e (mod N)) * (a3^e (mod N)) - (a2^e (mod N))^2`를 계산하게 되면 0이 아닙니다. $N$으로 나눈 나머지가 같은 두 값을 뺐으니, 이 값은 분명 $N$의 배수여야 합니다.

이제 똑같은 방법으로 `a1' = "X: \x01", a2' = "X: \x01\x00", a3' = "X: \x01\x00\x00"` 와 같은 값에 대해서 똑같은 방식을 통해 $N$의 배수에 해당하는 값을 얻어봅시다. 그러면 이 두 값의 최대공약수를 구하면 이는 매우 높은 확률로 $N$입니다.

만약 이 때 나온 값이 여전히 $N$이 아닌 $N$의 배수가 아닐까 걱정된다면 여러 번 더 시도해서 완벽하게 만들면 됩니다. 이렇게 나온 $N$은 1012-bit 정수였습니다.

## What should I do now?

하지만 여기서 끝입니다. $e$ 값을 구해보기 위해서 1부터 하나씩 $e$인지 탐색해보지만 1억이 넘도록 $e$값은 나오지 않습니다. $d$에 대해서도 탐색해보지만 마찬가지입니다. 여기서 과연 무슨 트릭을 써야하는 것일까요?

놀랍게도, 이 $N$은 소인수분해가 쉬운 수였습니다. 대회가 끝나고 이걸 알게 된 저는 어이가 없었습니다.

> 아니, 코드에서는 그냥 key라는 파일로부터 N 값을 읽어오기만 하는데 N이 소인수분해가 쉬운지 아닌지 어떻게 알아?

뭔가 특수한 트릭이나 논문이 있는지 찾아보고 있던 저로써는 너무나 실망스러운 풀이였지만, 가장 기초적인 방법인 소인수분해를 시도해보지 않은 제 잘못도 있다고 생각했습니다.

그런데, '소인수분해가 쉬운 수' 라니, $N$이 1012-bit나 되는데 소인수분해를 쉽게 할 방법이 있는 걸까요?

# Smooth Integer and Factorization

## Smooth Integer

Smooth Integer라고 하는 개념이 있습니다. $B$-smooth integer는 그 수를 소인수분해 했을 때, 소인수 중에 $B$보다 큰 소인수가 없다는 의미입니다. 예를 들어, 1620은 $2^2 \times 3^4 \times 5$로 가장 큰 소인수가 5이므로, 5-smooth integer입니다.

여기서 더 나아가서, Powersmooth Integer라고 하는 개념도 있습니다. $B$-powersmooth integer는 소인수 $p$ 뿐만 아니라 소인수 $p$ power $p^k$가 약수일 때 모든 $p^k$가 $B$ 이하라는 의미입니다. 앞서 언급했던 1620을 보면, 1620는 $3^2, 3^3, 3^4$로 나뉘므로 5-powersmooth integer는 아닙니다. 하지만 81(=$3^4$)-powersmooth integer라고 할 수 있습니다.

이 Smooth integer과 Powersmooth integer 개념은 소인수분해 알고리즘에서 유용하게 사용됩니다. 예시를 들자면, Elliptic-curve Factorization Method(ECM)과 Pollard's p-1 알고리즘이 있는데, ECM의 경우 이 글에서 설명하기에는 난해하므로 설명하지 않겠습니다.

## Pollard's p-1 algorithm

앞서 나왔던 오일러의 정리를 생각해보면, 소수 $p$에 대해서 $\phi(p) = p-1$이므로 $p$와 서로소인 $a$에 대해서 $a^{p-1} \equiv 1 (\mathbb{mod}\ p)$ 일 것입니다. 이 소수 $p$에 한정한 버전을 페르마의 소정리라고도 합니다.

이를 확장하자면, 어떤 정수 $k$에 대해서 $a^{k(p-1)} \equiv 1 (\mathbb{mod}\ p)$ 이기도 할 것입니다. 즉, $a^{k(p-1) - 1} - 1$은 $p$의 배수라는 의미이기도 합니다.

그런데, 만약 $p-1$이 $B$-powersmooth라면 어떻게 될까요? 다음과 같은 수 $M$을 생각해봅시다.

$M = \prod_{\mathbb{primes}\ p\leq B} p^{\lfloor \log_q B \rfloor}$

이 $M$은 $B$-powersmooth인 수 중에서 가장 큰 수이면서, $B$-powersmooth integer들의 최소공배수입니다. 곧, 이 값이 $p-1$의 배수라는 것은 너무나 명확합니다. 그러므로, 앞서 얘기했던 대로 $a^M - 1$은 $p$의 배수일 것입니다. 그러므로 $a^M - 1$과 $N$의 최대공약수를 구해보면, $p$가 나올지도 모릅니다.

놀랍게도, $p$는 큰 소수일지 몰라도, $p-1$이 작은 $B$에 대해서 $B$-powersmooth인 경우가 생각보다 많습니다. 그러므로 다음과 같이 알고리즘을 생각해볼 수 있습니다.

1. $B$를 정한다.
2. $B$에 대해서 위에서 정리한 식대로 $M$을 구한다.
3. $N$과 서로소인 어떤 $a$를 정한다. 보통은 2를 사용한다.
4. $a^M-1$을 계산하고, $\gcd(a^M-1, N)$을 계산한다. 이 때, $a^M - 1$ 대신 $a^M - 1 (\mathbb{mod}\ N)$을 사용해도 괜찮다.
5. 계산한 값이 1이나 $N$이 아니라면, 해당 값은 $N$의 소인수일 것이므로 출력한다. 그렇지 않다면, 실패했다고 출력한다.

이것이 바로 Pollard's p-1 알고리즘입니다. 작은 $B$에 대해서 하나씩 늘려가면서 알고리즘을 시행해본다면, $B$가 작다는 가정 하에 $N$으로부터 $p$를 구할 수 있을 것입니다.

## Coding

앞서 구한 $N$ 값은 

```
28152737628466294873353447700677616804377761540447615032304834412268931104665382061141878570495440888771518997616518312198719994551237036466480942443879131169765243306374805214525362072592889691405243268672638788064054189918713974963485194898322382615752287071631796323864338560758158133372985410715951157
```

라는 값이었습니다.

이를 바탕으로 Pollard's p-1 algorithm을 시행하는 코드를 Python으로 작성해봅시다.

```python
from Crypto.Util.number import GCD
from Crypto.Math.Primality import miller_rabin_test
import math


N = 28152737628466294873353447700677616804377761540447615032304834412268931104665382061141878570495440888771518997616518312198719994551237036466480942443879131169765243306374805214525362072592889691405243268672638788064054189918713974963485194898322382615752287071631796323864338560758158133372985410715951157
B = 10000

# Prime
primes = []
for i in range(2, B + 1):
    flag = True
    for j in primes:
        if j * j > i:
            break
        if i % j == 0:
            flag = False
            break
    if flag:
        primes.append(i)

# Pollard's p-1 algorithm
a = 2
for p in primes:
    wow = pow(p, math.floor(math.log(B, p)))
    a = pow(a, wow, N)

print(GCD(a - 1, N))
```

이 코드를 바탕으로 B 값을 바꾸면서 시행해보면 (1씩 늘리는 것은 시간이 꽤 오래 걸립니다), 50000 정도에서는 1을 출력하지만 100000을 넘어가면 $N$ 값이 그대로 나오는 것을 볼 수 있습니다. 즉, 100000 위로는 $p$의 배수가 나오기는 했지만 $q$의 배수이기도 해서 $N$ 이 나온다고 생각해볼 수 있습니다. 그래서 90000을 대입하면 다음과 같은 결과를 얻을 수 있습니다.

```python
531268630871904928125236420930762796930566248599562838123179520115291463168597060453850582450268863522872788705521479922595212649079603574353380342938159
```

이 값은 소수입니다. 드디어 $p$를 구하는 데 성공했습니다.

이렇게 소인수한 $p$와 $q$ 값을 알게 되면, [Pohlig-Hellman Algorithm](https://en.wikipedia.org/wiki/Pohlig%E2%80%93Hellman_algorithm)을 통해서 위의 문제에서의 $e$ 값을 구할 수 있습니다. 이에 대해서 이 글에서는 다루지 않겠습니다만, 해당 알고리즘은 $a$ 값과 $a^b (\mathbb{mod}\ N)$ 값을 알고 있을 때 $b$를 구할 수 있는 강력한 알고리즘입니다.

# 마무리

최근 소인수분해가 가능한 수에 대한 문제를 경험해본 적이 거의 없어서 너무나 뼈아픈 경험이었습니다. 하지만 이제는 당하지 않을 자신이 있습니다. 또한 소수를 구할 때는 $p-1$이 작은 $B$에 대해서 powersmooth한지 체크할 필요가 있겠습니다.

또한, 여러 암호학 라이브러리에서 소수를 생성하는 함수에는 Strong prime을 구하는 flag를 True/False로 줄 수 있게 되어 있습니다. Strong prime $p$의 [암호학적 정의](https://eprint.iacr.org/2001/007)는 다음과 같습니다.

- $p$가 충분히 클 것.
- $p-1$이 큰 소인수 $q_1$을 가질 것.
- $q_1 - 1$ 또한 큰 소인수 $q_2$를 가질 것.
- $p + 1$이 큰 소인수 $q_3$를 가질 것.

위의 코드는 Python 구현체이므로 생각한 것 이상으로 느릴 수가 있습니다. 이에 대해서 추천하는 프로그램/웹 사이트는 다음과 같습니다.

- Msieve: https://github.com/radii/msieve
- Integer factorization calculator: https://www.alpertron.com.ar/ECM.HTM

마지막으로, 비단 이런 분야가 아니더라도, 문제를 풀 때는 초심으로 돌아가 생각하는 습관을 꼭 기르도록 합시다. 이 대회의 교훈이었습니다.