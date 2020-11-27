---
layout: post
title:  "SVP and CVP"
date:   2020-10-23 14:30
author: RBTree
tags: [ctf, cryptography]
---

# 서론

이번에 N1CTF에 Super Guesser 팀으로 참전해서 4등을 차지했습니다. ([ctftime](https://ctftime.org/event/1099)) 한국의 강력한 crypto hacker rkm0959님과 함께 할 수 있는 좋은 자리였는데요, N1CTF에 나온 한 문제가 SVP(Shortest Vector Problem)과 CVP(Closest Vector Problem)에 대해서 다루기 좋아서 이번에 이렇게 글을 작성하게 되었습니다.

# 본론

## Lattice

우선 SVP와 CVP에 대해서 살펴보기 전에 Lattice에 대해서 알아봐야 합니다. Lattice는 이름에서 알 수 있듯이 격자 형태를 나타내는 수학적 개념입니다.

Lattice는 서로 independent한 vector $\{v_1, v_2, \ldots v_n\}$이 있을 때 다음과 같은 집합을 의미합니다.

$L = \{\sum_{i=1}^n a_iv_i \vert a_i \in \mathbb{Z}\}$

즉, vector $v_i$의 정수배의 합으로 나타내어지는 vector의 집합인데요, vector $v_i$ 간격의 격자들로 이루어진 $n$차원 공간으로 이해할 수도 있겠습니다. 이 때 Lattice $L$ 을 이루는 vector $v_i$ 들을 $L$의 basis라고 합니다.

## SVP and CVP

SVP(Shortest Vector Problem)과 CVP(Closest Vector Problem)은 Lattice에서 정의되는 가장 유명한 두 문제입니다. 이 두 문제를 간략하게 설명하자면 다음과 같습니다.

---

### SVP

Lattice $L$ 위에서 영점 $0$과 가장 가까운 non-zero vector $v$를 구하여라.

### CVP

Lattice $L$ 위에서 주어진 vector $u$ ($L$ 위에 있지 않을 수도 있음)에 가장 가까운 vector $v$를 구하여라.

---

SVP는 NP-hard인 것이 증명된 문제이며, CVP는 SVP만큼 어렵다는 것이 증명된 문제입니다. 해당 문제들을 직접 푸는 것은 불가능하지만, 특정 범위 내에서 근사해서 푸는 알고리즘이 나와있습니다. 바로 LLL algorithm과 Babai's algorithm입니다.

## LLL algorithm and Babai's Algorithm

LLL algorithm은 원래 Lattice $L$과 basis $B$가 주어져 있을 때, 더 짧은 vector들로 구성된 새로운 basis $B'$를 구하는 알고리즘입니다. 하지만 SVP를 풀 때는 이렇게 나온 basis가 SVP의 근사해가 됩니다. LLL algorithm의 수학적 원리에 대해서 자세히 설명하는 것은 어려운 일이기 때문에, 생략하고 sage를 통해서 어떻게 구할 수 있는지는 다음 간단한 코드를 참조해주시기 바랍니다.

```python
sage: A = Matrix(ZZ, [[10, 9, 8], [7, 6, 5], [12, 42, 39]])
sage: A.LLL()
[ 1  0 -1]
[-3 -3 -3]
[-4 12 -5]
```

일반적으로 LLL의 output의 각 row는 위에서 아래로 갈수록 사이즈가 커집니다. 이는 위의 예시의 output을 보셔도 알 수 있습니다. 즉, 첫 번째 row에서 그 다음 row로 갈수록 SVP의 답일 확률은 더 낮아진다고 생각해도 무방합니다.
Babai's algorithm은 CVP를 푸는 데 있어서 강력한 알고리즘입니다. Babai's algorithm은 LLL을 통해서 구한 basis를 바탕으로 Gram-Schmidt process를 실행한 뒤, 이렇게 나온 결과를 바탕으로 target과 가까운 점을 구하는 과정을 거칩니다.

```python
from sage.modules.free_module_integer import IntegerLattice

def Babai_CVP(Lattice, target):
    M = IntegerLattice(Lattice, lll_reduce=True).reduced_basis
    print("LLL Done")
    G = M.gram_schmidt()[0]
    print("Gram-Schmidt Done")
    diff = target
    for i in reversed(range(G.nrows())):
        diff -=  M[i] * ((diff * G[i]) / (G[i] * G[i])).round()
    return target - diff
```

이제 이를 응용해서 N1CTF에서 나온 문제를 하나 풀어봅시다.

## N1CTF - Easy RSA?

Easy RSA? 문제는 다음과 같은 sage 코드가 주어집니다.

```python
from Crypto.Util.number import *
import numpy as np

mark = 3**66

def get_random_prime():
    total = 0
    for i in range(5):
        total += mark**i * getRandomNBitInteger(32)
    fac = str(factor(total)).split(" * ")
    return int(fac[-1])

def get_B(size):
    x = np.random.normal(0, 16, size)
    return np.rint(x)

p = get_random_prime()
q = get_random_prime()
N = p * q
e = 127

flag = b"N1CTF{************************************}"
secret = np.array(list(flag))

upper = 152989197224467
A = np.random.randint(281474976710655, size=(e, 43))
B = get_B(size=e).astype(np.int64)
linear = (A.dot(secret) + B) % upper

result = []
for l in linear:
    result.append(pow(l, e, N))

print(result)
print(N)
np.save("A.npy", A)

```

코드를 읽어보면, `A * secret + B`를 통해서 나온 linear output은 secret과 직접적 관련이 있고, 이 linear output은 RSA에 의해서 한 번 더 encryption됩니다. 그러므로, 1. N을 소인수분해 한 뒤, 2. 복구한 linear 값으로부터 secret을 복구하는 과정이 필요하겠습니다.

### 1. N의 소인수 분해

N을 생성할 때 사용되는 소수 생성 함수 `get_random_prime`을 보면, 소수 $p, q$는 작은 $a_i$ 에 대해서 $\sum 3^{66i} a_i$의 가장 큰 소인수임을 알 수 있습니다. 더 나아가면, $N = pq$는 $\sum 3^{66i} a_i * \sum 3^{66i} b_i$라는 수의 큰 인수임을 알 수 있습니다. $a_i b_j$ 자체도 $2^{64}$ 보다 작으므로 $N$보다 상대적으로 작기 때문에, 다음과 같은 Lattice를 만드는 것이 가능합니다.

```python
mark = 3**66
big_num = 2**100
mat = matrix(
    [
        [big_num, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [mark * big_num, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [mark^2 * big_num, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [mark^3 * big_num, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [mark^4 * big_num, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [mark^5 * big_num, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [mark^6 * big_num, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [mark^7 * big_num, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [mark^8 * big_num, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [N * big_num, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
)

l = mat.LLL()
```

다음 lattice에 대해서 LLL을 실행하게 되면, $N = \sum_{i=0}^8 3^{66i} c_i$를 구성하는 $c_i$를 얻을 수 있고, 미지수 $x$에 대해서 $\sum_{i=0}^8 c_ix^i$를 인수분해 함으로써 $a_i$와 $b_i$를 복구하는 것이 가능합니다. 이를 하는 코드는 다음과 같이 작성할 수 있습니다.

```python
N = 32846178930381020200488205307866106934814063650420574397058108582359767867168248452804404660617617281772163916944703994111784849810233870504925762086155249810089376194662501332106637997915467797720063431587510189901

mark = 3**66
big_num = 2**100
mat = matrix(
    [
        [big_num, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [mark * big_num, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [mark^2 * big_num, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [mark^3 * big_num, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [mark^4 * big_num, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [mark^5 * big_num, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [mark^6 * big_num, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [mark^7 * big_num, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [mark^8 * big_num, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [N * big_num, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
)

L = mat.LLL()

R.<x> = ZZ[]
for i in range(10):
    if L[i][0] != 0:
        continue
    
    f = 0
    for j in range(9):
        f -= L[i][j + 1] * x^j
    
    print(f.factor())
```

이렇게 실행하게 되면 결과 중 `(2187594805*x^4 + 2330453070*x^3 + 2454571743*x^2 + 2172951063*x + 3997404950) * (3053645990*x^4 + 3025986779*x^3 + 2956649421*x^2 + 3181401791*x + 4085160459)` 가 있습니다. 이를 바탕으로 복구한 수를 통해서 소인수분해가 가능함을 확인할 수 있습니다.

```python
sage: (2187594805*x^4 + 2330453070*x^3 + 2454571743*x^2 + 2172951063*x + 3997404950)(3^66)
1995161838028207402648932593559579403854561289639473871304480171837396129516627469957637764917297720390425917325316755744300520010585575
sage: (3053645990*x^4 + 3025986779*x^3 + 2956649421*x^2 + 3181401791*x + 4085160459)(3^66)
2785030359448062888404614030934990637983801536220504550904528896981372246003302501286442127881138927902357190949892601236464509136512680
sage: 199516183802820740264893259355957940385456128963947387130448017183739612951662746995763776491729772039042591732531
....: 6755744300520010585575.factor()
5^2 * 461 * 2126903 * 4779209 * 139268956397 * 122286683590821384708927559261006610931573935494533014267913695701452160518376584698853935842772049170451497
sage: 278503035944806288840461403093499063798380153622050455090452889698137224600330250128644212788113892790235719094989
....: 2601236464509136512680.factor()
2^3 * 3 * 5 * 11 * 19^2 * 15113509 * 1439719226465297 * 268599801432887942388349567231788231269064717981088022136662922349190872076740737541006100017108181256486533
sage: 122286683590821384708927559261006610931573935494533014267913695701452160518376584698853935842772049170451497 *  26
....: 8599801432887942388349567231788231269064717981088022136662922349190872076740737541006100017108181256486533
32846178930381020200488205307866106934814063650420574397058108582359767867168248452804404660617617281772163916944703994111784849810233870504925762086155249810089376194662501332106637997915467797720063431587510189901
```

### 2. Secret 복구

이제 다음 파트를 파훼해야합니다.

```python
def get_B(size):
    x = np.random.normal(0, 16, size)
    return np.rint(x)

upper = 152989197224467
A = np.random.randint(281474976710655, size=(e, 43))
B = get_B(size=e).astype(np.int64)
linear = (A.dot(secret) + B) % upper
```

그런데 `get_B` 함수를 살펴보면 역시 값이 무척 작다는 것을 대략적으로 알 수 있습니다. 즉, $B$는 일종의 작은 error이고, $A$라는 matrix로 이루어진 lattice 상에서 linear와 가장 가까운 점을 찾으면 secret을 복구할 수 있을 것입니다. 즉, 이 문제에는 Babai's algorithm을 사용해볼 수 있을 것입니다. 다음 소스 코드와 같이 Lattice를 구성해서 Babai's algorithm을 적용하면, 플래그를 복구하는 것이 가능합니다.

이 때 꼭 A의 전체를 사용하지 않아도 충분히 답이 나올 수 있기 때문에, `sel`을 정의해서 A의 일부만 사용할 수 있도록 만들었습니다. 정확하게 Babai's algorithm을 돌리고 싶다면 127개의 column 전체를 사용해야겠지만, 그렇게 하게 되면 LLL과 Gram-Schmidt process에서 상당히 오랜 시간이 소요되기 때문에 이와 같이 정의해서 실행했고 실제로도 충분히 플래그를 구할 수 있습니다.

```python
from sage.modules.free_module_integer import IntegerLattice
import numpy as np

N = 32846178930381020200488205307866106934814063650420574397058108582359767867168248452804404660617617281772163916944703994111784849810233870504925762086155249810089376194662501332106637997915467797720063431587510189901
p = 122286683590821384708927559261006610931573935494533014267913695701452160518376584698853935842772049170451497
q = 268599801432887942388349567231788231269064717981088022136662922349190872076740737541006100017108181256486533

e = 127
n = p * q
phi = (p - 1) * (q - 1)
d = inverse_mod(e, phi)

# Sorry I'm so lazy
with open('res.txt', 'r') as f:
    res = eval(f.readline())

linear = []
for x in res:
    linear.append(pow(x, d, n))

np.set_printoptions(threshold=sys.maxsize)
A = np.load("A.npy")
A = np.ndarray.tolist(A)

## Using Babai's algorithm

upper = 152989197224467
sel = 15 # sel can be large as 127, but that's too slow
M = Matrix(ZZ, sel + 43, sel + 43)
for i in range(0, 43):
    for j in range(0, sel):
        M[i, j] = A[j][i]
    M[i, sel + i] = 1
for i in range(43, 43+sel):
    M[i, i-43] = upper

target = vector([0 for _ in range(sel + 43)])
for i in range(0, sel):
    target[i] = linear[i] - 8
for i in range(sel, sel + 43):
    target[i] = 128 // 2 # Because it's printable

def Babai_CVP(Lattice, target):
    M = IntegerLattice(Lattice, lll_reduce=True).reduced_basis
    print("LLL Done")
    G = M.gram_schmidt()[0]
    print("Gram-Schmidt Done")
    diff = target
    for i in reversed(range(G.nrows())):
        diff -=  M[i] * ((diff * G[i]) / (G[i] * G[i])).round()
    return target - diff

TT = Babai_CVP(M, target)

print(TT)

res = ""
for i in range(sel, sel + 43):
    res += chr(TT[i])

print(res)
```

이를 통해서 얻은 flag는 `N1CTF{f55bfc7e-7955-412a-81a9-ed2650b50564}` 입니다.

# 결론

이번 대회에서 LLL과 Babai's algorithm의 강력함에 대해서 많이 느꼈습니다. 공부는 해본 적이 있지만 잘 사용하지 못하곤 했었는데, rkm님의 도움으로 이번에 어떻게 하면 좀 더 쉽게 응용이 가능한지 터득할 수 있었습니다.

또한 Easy RSA?의 1단계의 lattice의 경우 제가 아이디어를 떠올렸었는데, 이와 같이 이쁜 Lattice를 정의하는 것도 중요한 능력이라는 것을 다시금 한 번 느낄 수 있었습니다.

# 참고 문헌

1. rkm0959님의 write-up [Link](https://rkm0959.tistory.com/167)
2. N1CTF 공식 write-up [Link](https://github.com/Nu1LCTF/n1ctf-2020/blob/main/N1CTF2020%20Writeup%20By%20Nu1L.pdf)
3. 34C3 CTF - LOL write-up [Link](https://oddcoder.com/LOL-34c3/)
4. Aero CTF - Magic II write-up [Link](https://hackmd.io/@hakatashi/B1OM7HFVI)