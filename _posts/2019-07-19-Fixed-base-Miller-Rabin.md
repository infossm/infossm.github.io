---
layout: post
title:  "Fixed-base Miller-Rabin"
date:   2019-07-19 09:00:00
author: blisstoner
tags: [cryptography]
---
shjgkwo님이 작성하신 [포스팅](http://www.secmem.org/blog/2019/06/17/PrimeNumber/)에서 Miller-Rabin Algorithm을 소개하고 있습니다. 마침 저희 팀이 [WCTF](https://blog.encrypted.gg/864)에서 출제했던 문제 중에 Miller-Rabin에서 base가 고정되어있을 때 실제로는 합성수이나 소수로 판정되는 수를 쉽게 찾을 수 있다는 취약점을 이용하는 문제가 있었기에 이 부분을 설명드리려고 합니다.

# Miller-Rabin Algorithm

shjgkwo님의 포스팅에도 정보가 있지만 다시 한 번 설명하고 가겠습니다. 독자가 기초적인 정수론에 대한 지식은 가지고 있다고 가정하고 진행하겠습니다.

보통 어떤 수 $$N$$이 소수인지 판별하기 위해서는 1부터 $$N^{0.5}$$까지의 모든 수로 나눠보는 알고리즘을 많이 사용 합니다. 이 알고리즘은 $$N$$이 그다지 크지 않을 때 흔히 사용하는 알고리즘인데, 시간복잡도는 $$O(N^{0.5})$$이지만 $$N$$이 소수인지 합성수인지를 정확하게 알 수 있습니다.

이와 달리 Miller-Rabin Algorithm은 특정한 수가 소수인지 판단하는 `확률적 알고리즘`입니다. 정확히 말해 특정 수가 합성수임에도 불구하고 소수로 판단될 수 있는 여지가 있습니다. 그러나 알고리즘의 시간복잡도가 $$O(log^2nloglognlogloglogn)$$ 으로 굉장히 빠르고(단, 만약 FFT를 사용하지 않을 경우 $$O(log^3n)$$) 합성수임에도 불구하고 소수로 판단할 확률을 0에 가깝게 낮출 수 있기 때문에 RSA, Diffie Hellman 키 교환과 같이 큰 소수가 필요할 때, 임의로 만들어낸 수가 소수가 맞는지 확인하기 위해 쓰입니다.

Miller-Rabin Algorithm은 페르마의 소정리를 이용한 소수 판정법입니다. 소수 $$p$$와 서로소인 임의의 자연수 $$a$$에 대해 $$a^{p-1} \equiv 1 (mod \; p)$$임은 널리 알려진 사실입니다. 또한 $$x^2 \equiv 1 (mod \; p)$$일 경우 $$(x-1) \equiv 0 $$ or $$ (x+1) \equiv 0 (mod \; p)$$ 이므로 $$x \equiv \pm1 (mod \; p)$$ 입니다.

이 두 사실로부터 $$p-1 = 2^sd$$라고 나타냈을 때, $$a^d \equiv 1 (mod \; p)$$ 이거나 $$a^{2^0d} \equiv -1 (mod \; p)$$, $$a^{2^1d} \equiv -1 (mod \; p)$$, $$a^{2^2d} \equiv -1 (mod \; p)$$, $$\dots a^{2^{s-1}d} \equiv -1 (mod \; p)$$ 중에서 하나는 반드시 성립함을 알 수 있습니다.

조금 더 알아보기 쉽게 작성하면, 소수 $$p$$와 $$p$$의 배수가 아닌 임의의 자연수 $$a$$에 대해 다음 식 중 한가지가 성립함을 의미합니다.

+ Eq1. $$a^d \equiv 1 (mod \; p)$$ 
+ Eq2. $$a^{2^sd} \equiv -1 (mod \; p)$$ for some $$ 0 \leq r \leq s-1$$

이를 이용해 자연수 $$n = 2^sd+1$$이 임의의 자연수 $$a$$에 대해 $$a^d \equiv 1 (mod \; n)$$이거나 for some $$ 0 \leq r \leq s-1$$, $$a^{2^rd} \equiv -1 (mod \; n)$$ 일 경우 $$n$$은 소수일 것이라고 판단하는 것이 Miller-Rabin Algorithm입니다.

# Miller-Rabin Algorithm의 정확도

페르마의 소정리는 역이 성립하지 않습니다. 예를 들어 341은 합성수이지만 $$2^{340} \equiv 1 (mod \; 341)$$ 입니다. 이렇게 자연수 $$a$$에 대해 $$a^{n-1} \equiv 1 (mod \; n)$$인 합성수 $$n$$을 `pseudoprime for the base a` 라고 부릅니다.

비록 합성수 $$n$$이 pseudoprime for the base a라고 하더라도 Miller-Rabin Algorithm에서 base a에 대해 소수라고 판정이 되는 것은 아닙니다. 당장 341만 해도 $$2^{85} \equiv 32, 2^{170} \equiv 1 (mod \; 341)$$이기 때문에 341은 base 2에 대해 pseudoprime임에도 불구하고 합성수라고 판정이 됩니다.

그러나 만약 합성수 $$n$$이 base $$a$$에서 Miller-Rabin Test를 통해 소수라고 판정이 된다면 $$n$$은 pseudoprime for the base a임을 알 수 있습니다. 즉 합성수 $$n$$이 base $$a$$에서 Miller-Rabin Test를 통해 소수라고 판정이 될 필요 조건이 바로 $$n$$이 pseudoprime for the base a입니다.

만약 합성수 $$n$$에 대해 $$n$$을 소수로 판정하는 base가 많다면 이 판정법으로 소수를 제대로 걸러낼 수 없기 떄문에 문제가 될 것입니다. 그러나 다행히도 합성수 $$n$$에 대해 $$n$$을 소수로 판정하는 base는 최대 $$\phi(n) / 4$$ 임이 증명이 되어있습니다.(Schoof, René (2004), "Four primality testing algorithms", [link](http://www.mat.uniroma2.it/~schoof/millerrabinpom.pdf))

위의 식에서 $$\phi$$는 오일러 파이 함수를 의미하고 $$\phi(n) < n$$이기 때문에 합성수 $$n$$에 대해 $$n$$을 소수로 판정하는 base는 $$n/4$$개 미만임을 알 수 있습니다. 이는 곧 하나의 base에 대해 Miller-Rabin Algorithm이 합성수를 소수로 잘못 판단할 확률이 $$4^{-1}$$ 미만이라는 의미입니다. 더 나아가 base를 한 개가 아닌 여러 개를 사용한다면 합성수를 소수로 잘못 판단할 확률은 $$4^{-1}, 4^{-2}, 4^{-3}, \dots $$과 같이 계속 낮아지게 됩니다.

일반화 리만 가설(Generalized Riemann Hypothesis)가 참이라면 $$2log^2n$$개의 base에 대해 모두 소수라고 판단된 수는 실제로 소수입니다. 실제로 몇 개의 base에 대해 진행할지는 정하기 나름이지만 보통의 경우 랜덤으로 택한 50개 이상의 수에 대해 모두 소수라고 판단되는지를 확인하는 방식으로 실제 암호 관련 모듈에서 사용하고 있습니다.

# Fixed-base Miller-Rabin

그러나 Miller-Rabin Algorithm에서 base가 고정되어있으면 아주 심각한 문제를 불러일으킬 수 있습니다. 고정된 base에 대해 Miller-Rabin Algorithm 결과가 소수가 되는 합성수 $$n$$을 만들어낼 수 있기 때문입니다. Gerhard Jaeschke, Math. Comp. 61 (1993), 915-926, "On strong pseudoprimes to several bases"[(link)](http://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192971-8/home.html)에 의하면
8038374574536394912570796143419421081388376882875581458374889175222974273765333652186502336163960045457915042023603208766569966760987284043965408232928738791850869166857328267761771029389697739470167082304286871099974399765441448453411558724506334092790222752962294149842306881685404326457534018329786111298960644845216191652872597534901는 base 2, 3, 5, 7, 11, 13, 17, 19, 23, 29에 대해 소수로 판정되나 실제로는 합성수입니다.

실제 암호 관련 모듈에서 고정된 base를 사용하거나, 취약한 random을 이용한 base를 사용하는 경우가 있습니다. 이 경우 악의적인 사용자가 실제로는 합성수인 수를 가지고 무언가 장난을 칠 수 있는 상황이 나오게 됩니다.

그런데 base가 고정되어있다고 할 때 이러한 수를 어떤 식으로 찾을 수 있을까요? 임의의 합성수를 골라 그 합성수가 주어진 base들로 판정한 결과가 전부 소수인지를 확인하는 작업을 계속 시행하면 될까요? 안타깝게도 그렇지 않습니다. 편의상 그냥 base가 5개라고 합시다. 비록 위에서 언급한 것과 같이 합성수 $$n$$이 5단계의 Miller-Rabin Algorithm을 거쳐 소수라고 판정될 확률은 $$4^{-5}$$ 미만이지만 실제 하한은 이보다 훨씬 낮습니다. $$n$$이 512-bit의 수라고 할 때 실제로 $$n$$이 소수라고 판정될 확률은 $$2^{-85}$$ 미만입니다. `(Ronald Burthe: Further investigations with the strong probable prime
test, Math. Comp. 65, pp.373-381, 1996.)` 그렇기에 단순하게 합성수를 골라 해당 수가 Miller-Rabin Algorithm을 통과하는지 확인하는 방식으로는 이러한 수를 찾는 것이 불가능에 가깝습니다.

대신, 이를 어떤식으로 하면 되는지는 `(Bleichenbacher D. (2005) Breaking a Cryptographic Protocol with Pseudoprimes. In: Vaudenay S. (eds) Public Key Cryptography - PKC 2005. PKC 2005. Lecture Notes in Computer Science, vol 3386. Springer, Berlin, Heidelberg)` 이라는 논문에 상세히 나와있습니다. 수를 만들어내는 과정에는 Korselt, Erdos의 선행 연구가 쓰이고 다소 깊은 정수론적 지식이 쓰이기 때문에 왜 이 과정이 Miller-Rabin Algorithm을 bypass하는 합성수를 만들어내는지에 대한 증명은 생략하겠습니다. 대신, 논문의 과정에 따라 실제로 고정된 base에 대해 strong pseudoprime을 만들어보겠습니다.

논문에서는 `2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41`에 대해 진행했지만 저는 `777777773, 1000000007, 1000000009, 998244353, 65537` 이 5개 소수를 base로 하는 Miller-Rabin test를 통과하는 합성수를 만들어보겠습니다. 편의상 5개의 base를 $$b_0, b_1, b_2, b_3, b_4$$ 라고 하겠습니다.

+ 약수가 굉장히 많은 짝수 $$M$$을 임의로 정합니다. 단 $$M$$은 4의 배수가 아니어야 합니다. 저는 $$M = 2^1 \cdot 3^3 \cdot 5^2 \cdot 7^1 \cdot 11^1 \cdot 13^1 \cdot 17^1 \cdot 19^1 \cdot 23^1 \cdot 29^1 \cdot 31^1 \cdot 37^1$$ 으로 정했습니다.

+ 위에서 정한 $$M$$에 대해, $$M$$의 짝수인 약수 중에서 1을 더하면 소수인 수들을 찾습니다. 이렇게 만들어진 소수를 $$r$$이라고 했을 때, $$(\frac{r}{b_0}), (\frac{r}{b_1}), (\frac{r}{b_2}), (\frac{r}{b_3}), (\frac{r}{b_4})$$ 이 동일한 $$r$$들의 집합을 만들어냅니다. 여기서 $$(\frac{a}{b})$$는 르장드르 기호를 의미하고, 가능성은 32가지이기 때문에 총 32개의 집합이 만들어집니다. 이 집합을 편의상 $$S_0, S_1, S_2, \dots S_{31}$$ 이라고 부르겠습니다.

+ 각 $$S_i$$에 대해, $$S_i$$의 부분집합 $$T$$에서 $$T$$에 속한 모든 원소의 곱이 $$mod \; M$$에 대해 1과 합동이라면 $$T$$에 속한 모든 원소의 곱이 곧 $$b_0, b_1, b_2, b_3, b_4$$에 대한 strong pseudoprime이 됩니다.

3번 과정에서 Meet In the Middle을 이용해 시간복잡도를 줄일 수 있고, 실제로 이를 구현한 python 코드는 아래와 같습니다.

```python
import sys
from Crypto.Util import number
from Crypto.Random import random
sys.setrecursionlimit(500000)

B = [777777773, 1000000007, 1000000009, 998244353, 65537]

def is_prime(n):
  if n == 2: return True
  if n <= 1 or (not n&1): return False
  return all(miller_rabin_round(n, base) for base in [2, 325, 9375, 28178, 450775, 9780504, 1795265022])

def miller_rabin_round(n, base):
  base %= n
  if base == 0: return True
  d = n-1
  r = 0
  while not d & 1:
    d >>= 1
    r += 1
  assert(n-1 == 2**r * d)
  
  x = pow(base, d, n)
  if x == 1 or x == n-1: return True
  for _ in range(r-1):
    x = x * x % n    
    if x == n-1: return True
  return False

def miller_rabin(n):  
  return all(miller_rabin_round(n,base) for base in B)

def factorize(n):
  factors = []
  p = 2
  while True:
    while(n % p == 0 and n > 0): #while we can divide by smaller number, do so
      factors.append(p)
      n = n // p
    p += 1  #p is not necessary prime, but n%p == 0 only for prime numbers
    if p > n // p:
      break
  if n > 1:
    factors.append(n)
  return factors

# https://martin-thoma.com/how-to-calculate-the-legendre-symbol/
def legendre(a, p):    
  if a >= p or a < 0:
    return legendre(a % p, p)
  elif a == 0 or a == 1:
    return a
  elif a == 2:
    if p%8 == 1 or p%8 == 7:
      return 1
    else:
      return -1
  elif a == p-1:
    if p%4 == 1:
      return 1
    else:
      return -1
  elif not is_prime(a):
    factors = factorize(a)
    product = 1
    for pi in factors:
      product *= legendre(pi, p)
    return product
  else:
    if ((p-1)//2)%2==0 or ((a-1)//2)%2==0:
      return legendre(p, a)
    else:
      return (-1)*legendre(p, a)


def get_even_divisor(factor):
  divisor = [2]
  for f in factor:
    p, e = f
    if p == 2: continue
    divisor += [d * p**a for d in divisor for a in range(1,e+1)]
  return divisor

def recover(S, st, en, idx):
  ret = 1
  for i in range(en-1, st-1, -1):
    if idx & 1: ret *= S[i]
    idx >>= 1
  return ret

# want to find subset of S which production is 1 mod m(upper n-bit)
def MITM(S, m, n):
  ret = []
#  print(S)
  k = len(S)//2
  table1 = [(1,0)]
  for i in range(k):
    table1 = [(elem[0], elem[1]<<1 | 0) for elem in table1] + [(elem[0]*S[i]%m, elem[1]<<1 | 1) for elem in table1]
  table1.pop(0)

  table2 = [(1,0)]
  for i in range(k+1,len(S)):
    table2 = [(elem[0], elem[1]<<1 | 0) for elem in table2] + [(elem[0]*S[i]%m, elem[1]<<1 | 1) for elem in table2]
  table2.pop(0)

  table2 = [(number.inverse(elem[0],m), elem[1]) for elem in table2]
  idx1,idx2 = 0,0
  table1.sort()
  table2.sort()
  idx1 = 0
  idx2 = 0
  while idx1 < len(table1) and idx2 < len(table2):
    if table1[idx1][0] < table2[idx2][0]:
      idx1 += 1
    elif table1[idx1][0] > table2[idx2][0]:
      idx2 += 1
    else:
      idx = (table1[idx1][1] << (len(S)-k)) | table2[idx2][1]
      val = recover(S,0,len(S),idx)
      if val.bit_length() >= n:
        factor = []
        for i in range(len(S)-1, -1, -1):
          if idx & 1: factor.append(S[i])
          idx >>= 1
        return val,factor
      idx1 += 1
      idx2 += 1
  return None, None

def generator():
  Mfactor = ((2,1),(3,3),(5,2),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1),(31,1), (37,1))  

  M = 1
  for f in Mfactor:
    M *= f[0]**f[1]

  divisor = get_even_divisor(Mfactor)
  divisor.sort()
  hash_table = [[] for _ in range(2**len(B))]      
  for d in divisor:
    if d.bit_length() > 40: break
    r = d+1
    if not is_prime(r): continue
    h = 0
    for base in B:
      l = legendre(base, r)
      if l == 0:
        h = -1
        break
      elif l == -1: h = (h<<1)
      else: h = (h<<1) | 1
    if h != -1:
      hash_table[h].append(r)

  for i in range(2**len(B)):
    print("hash table sz :", len(hash_table[i]))
    q, factors = MITM(hash_table[i][:50],M,512)
    if not q: continue
    if not miller_rabin(q):
      print("SOMETHING WRONG...")
      continue
    print("strong pseudoprime : {}".format(q))
    print(factors)        

generator()
```

이를 통해 $$127 \cdot 131 \cdot 1483 \cdot 2851 \cdot 6007 \cdot 16831 \cdot 89839 \cdot 114479 \cdot 278807 \cdot 505051 \cdot 535991 \cdot 644491 \cdot 1261639 \cdot 1647031 \cdot 2195359 \cdot 7017347 \cdot 8746651 \cdot 10336951 \cdot 14827411 \cdot 19153051 \cdot 59998951 \cdot 95014151 \cdot 164390851 \cdot 691537771 \cdot 2772725671 \cdot 10204223563 \cdot 12471828799$$ 이라는 strong pseudoprime을 얻어낼 수 있습니다. 개인 pc로 길어야 10분 내로 찾아집니다.

후보군을 많이 두기 위해 $$M$$이 크게 하면 할수록 $$S_0, S_1, \dots$$의 크기는 커지지만 이들에서 $$mod M$$에 대해 1과 합동인 부분집합을 찾는게 힘들어집니다.

# 응용

그렇다면 이 사실을 어떻게 응용할 수가 있을까요? 암호에서 소수가 쓰이는 경우를 생각해봅시다. 첫 번째로는 Diffie Hellman 키교환입니다. Diffie Hellman 키 교환에서 Alice와 Bob은 서로 대면하지 않은 상황에서 $$g^a \; mod \; p$$와 $$g^b \; mod \; p$$를 주고 받아 $$g^{ab} mod \; p$$를 키로 사용하게 됩니다. 이 과정에서 공격자가 $$g^a, g^b$$를 계산하더라도 공격자는 이산 대수 문제의 어려움으로 인해  $$g^{ab}$$를 알아낼 수 없습니다.

그러나 이전에 제가 작성한 이산 로그 관련 포스트([link](http://www.secmem.org/blog/2019/05/17/%EC%9D%B4%EC%82%B0-%EB%A1%9C%EA%B7%B8/))에 기술해두었듯 $$p-1$$이 작은 소수의 곱이라면 이산 로그 문제를 굉장히 쉽게 해결할 수 있게 됩니다. 이를 방지하기 위해 $$p = nq+1$$($$n$$은 작은 수, $$q$$는 소수)로 두어 $$p-1$$이 큰 소인수 $$q$$를 가지게끔 하는 것이 일반적입니다.

그런데 $$p, q$$가 소수인지 판단하기 위해 Miller-Rabin Algorithm을 사용하는데 base를 고정해두었다면 어떻게 될까요? 공격자가 실제로는 소수가 아닌데 소수로 판정되는 strong pseudoprime을 $$q$$로 두고 $$p$$는 $$nq+1$$이 소수가 되는 적당한 $$n$$을 찾아 이를 $$p, q$$로 설정한다면 Diffie Hellman 키 교환은 안전하지 않을 것입니다. 저희 팀이 WCTF에서 출제했던 문제는 Diffie Hellman 키 교환에서 취약한 $$p, q$$를 선정해 취약점을 Trigger하는 문제였습니다.

두 번째로, RSA 과정에서도 소수 $$p, q$$를 정하는데, 이 때 $$p$$가 strong pseudoprime일 경우 $$pq$$가 아주 쉽게 소인수분해가 되어 $$p, q$$가 복원될 수 있을 것입니다.

# 결론 및 제언
이번 글에서 Miller-Rabin에 대해 공부하고 base가 고정되었을 때 어떤 문제가 생길 수 있는지를 알아보았습니다. 다소 작위적인 상황이 아닌가 싶을 수도 있지만, Miller-Rabin Algorithm이 구현되어있는 암호 모듈에서 base를 고정해둔 경우는 굉장히 자주 있고 CVE-2014-9742, CVE-2018-4398과 같이 이로 인한 취약점이 실제로 현실 세계에서 찾아지기도 합니다. Miller-Rabin을 사용할 때에는 반드시 임의의 base를 선택할 수 있도록 합시다.
