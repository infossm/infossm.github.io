---
layout: post
title: Finite Nimber 계산
author: Sait2000
date: 2022-04-14
---




Aeren 님의 [Nimber](http://www.secmem.org/blog/2022/01/18/Nimber/) 포스트에서도 알 수 있듯이 $2^{2^k}$ 미만의 음이 아닌 정수는 nimber 덧셈(xor)과 nimber 곱에 대해서 체를 이룹니다. 이 글에서는 finite nimber에 대해서 nimber 곱, 제곱근, 곱셈 역원 등을 실제로 계산하기 위한 알고리즘을 소개합니다.

<https://www.ics.uci.edu/~eppstein/numth/>를 참고했습니다.




## 기본 규칙


이 파트에서 소개하는 성질을 포함해서 앞으로 nimber의 몇가지 성질을 증명하지 않고 주어진 것으로 받아들이도록 하겠습니다. Nimber는 체를 이루므로 덧셈, 곱셈에 대해서 교환 법칙, 결합 법칙, 분배 법칙 등이 성립합니다. 따라서 앞으로 표기에서 불필요한 괄호를 생략하고, 곱셈은 덧셈보다 먼저 계산하기로 합니다. 곱셈 기호가 생략되어있을 경우 nimber 곱을 의미합니다.

> 음이 아닌 정수 $a$, $b$, $c$에 대해서 다음이 성립한다.
>
> $ \begin{align}
  & a \oplus b = b \oplus a \newline
  & a \otimes b = b \otimes a \newline
  & a \oplus (b \oplus c) = (a \oplus b) \oplus c \newline
  & a \otimes (b \otimes c) = (a \otimes b) \otimes c \newline
  & a \otimes 1 = a \newline
  & a \otimes (b \oplus c) = a \otimes b \oplus a \otimes c \end{align} $


이때 $a \oplus b$는 $a$와 $b$의 xor이고, $a \otimes b$는 체의 성질과 다음 규칙으로 계산할 수 있습니다.

> $2^{2^k}$(Fermat 2-power)와 $a < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $ \begin{align}
  & 2^{2^k} \otimes a = 2^{2^k} \times a = a \ll 2^k \newline
  & 2^{2^k} \otimes 2^{2^k} = 2^{2^k} \oplus 2^{2^k - 1} \end{align} $


위 성질에서 $2^{2^k}$ 미만의 nimber를 $2^{2^{k-1}}$ 미만의 nimber로 표현할 수 있습니다.

> $k \geq 1$일 때 $a < 2^{2^k}$에 대해서 $a_{hi}, a_{lo} < 2^{2^{k-1}}$가 존재해 다음이 성립한다.
>
> $a = a_{hi} \otimes 2^{2^{k-1}} \oplus a_{lo} = a_{hi} \ll 2^{k-1} \| a_{lo}$


앞으로 $a = \left(a_{hi}, a_{lo}\right)$와 같이 적겠습니다. 앞으로 소개할 알고리즘은 대부분 $2^{2^k}$ 미만의 nimber에 대한 연산을 $2^{2^{k-1}}$ 미만의 nimber에 대한 연산을 이용해서 재귀적으로 계산합니다.




## 곱셈


곱셈의 내부 과정으로 $2^{2^k}$ 미만의 nimber $a$에 대해서 $\mathrm{half} _ {k}(a) = a \otimes 2^{2^k - 1}$을 계산하게 됩니다. 이름은 $2^{2^k - 1}$이 $2^{2^k}$의 반이라서 half인 것 같습니다.

> $k \geq 1$일 때 $a < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $\mathrm{half} _ {k}(a) = \left(\mathrm{half} _ {k-1}(a_{hi} \oplus a_{lo}), \mathrm{half} _ {k-1}(\mathrm{half} _ {k-1}(a_{hi}))\right)$

**증명**

$\begin{align} \mathrm{half} _ {k}(a) &= a \otimes 2^{2^k - 1} \newline
 &= \left(a_{hi}, a_{lo}\right) \otimes \left(2^{2^{k-1} - 1}, 0\right) \newline
 &= \left(a_{hi} \otimes 2^{2^{k-1} - 1}\right) \otimes \left(2^{2^{k-1}} \otimes 2^{2^{k-1}}\right) \oplus \left(a_{lo}\otimes 2^{2^{k-1} - 1}\right) \otimes 2^{2^{k-1}} \newline
 &= \mathrm{half} _ {k-1}(a_{hi}) \otimes \left(2^{2^{k-1}} \oplus 2^{2^{k-1}-1}\right) \oplus \mathrm{half} _ {k-1}(a_{lo}) \otimes 2^{2^{k-1}} \newline
 &= \left(\mathrm{half} _ {k-1}(a_{hi}) \oplus \mathrm{half} _ {k-1}(a_{lo})\right) \otimes 2^{2^{k-1}} \oplus \mathrm{half} _ {k-1}(a_{hi}) \otimes 2^{2^{k-1}-1} \newline
 &= \left(\mathrm{half} _ {k-1}(a_{hi} \oplus a_{lo}), \mathrm{half} _ {k-1}(\mathrm{half} _ {k-1}(a_{hi}))\right) \end{align}$


$2^{2^k}$ 미만의 nimber $a$, $b$에 대해 $a \otimes b$는 다음과 같이 3번의 $2^{2^{k-1}}$ 미만 곱셈과 1번의 $\mathrm{half} _ {k-1}$로 계산할 수 있습니다.

> $k \geq 1$일 때 $a, b < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $a \otimes b = \left( \left( a_{hi} \oplus a_{lo} \right) \left( b_{hi} \oplus b_{lo} \right) \oplus a_{lo} b_{lo}, \mathrm{half} _ {k-1}(a_{hi} b_{hi}) \oplus a_{lo} b_{lo}\right)$

**증명**

$\begin{align} a \otimes b &= \left(a_{hi}, a_{lo}\right) \otimes \left(b_{hi}, b_{lo}\right) \newline
 &= \left(a_{hi}, a_{lo}\right) \otimes \left(b_{hi}, b_{lo}\right) \newline
 &= \left(a_{hi}, 0\right) \left(b_{hi}, 0\right) \oplus \left(a_{hi}, 0\right) \left(0, b_{lo}\right) \oplus \left(0, a_{lo}\right) \left(b_{hi}, 0\right) \oplus \left(0, a_{lo}\right) \left(0, b_{lo}\right) \newline
 &= \left(a_{hi} b_{hi}, \mathrm{half} _ {k-1}(a_{hi} b_{hi})\right) \oplus \left(a_{hi} b_{lo}, 0\right) \oplus \left(a_{lo} b_{hi}, 0\right) \oplus \left(0, a_{lo} b_{lo}\right) \newline
 &= \left(a_{hi} b_{hi} \oplus a_{hi} b_{lo} \oplus a_{lo} b_{hi}, \mathrm{half} _ {k-1}(a_{hi} b_{hi}) \oplus a_{lo} b_{lo}\right) \newline
 &= \left( \left( a_{hi} \oplus a_{lo} \right) \left( b_{hi} \oplus b_{lo} \right) \oplus a_{lo} b_{lo}, \mathrm{half} _ {k-1}(a_{hi} b_{hi}) \oplus a_{lo} b_{lo}\right) \end{align}$


```python
def calc_level(a):
    k = 0
    while a >= 2 ** 2**k:
        k += 1
    return k

def half(a, k):
    if k == 0:
        return a
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    return (
        half(ahi ^ alo, k - 1)        << 2**(k-1) |
        half(half(ahi, k - 1), k - 1)
    )

def mul(a, b):
    k = max(calc_level(a), calc_level(b))
    if k == 0:
        return a & b
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    bhi = b >> 2**(k-1)
    blo = b & ((1 << 2**(k-1)) - 1)
    alo_blo = mul(alo, blo)
    return (
        (mul(ahi ^ alo, bhi ^ blo) ^ alo_blo) << 2**(k-1) |
        half(mul(ahi, bhi), k - 1) ^ alo_blo
    )
```




## 제곱


곱셈 공식에서 $a = b$이면, $\left( a_{hi} \oplus a_{lo} \right) \left( a_{hi} \oplus a_{lo} \right) = a_{hi} a_{hi} \oplus a_{hi} a_{lo} \oplus a_{lo} a_{hi} \oplus a_{lo} a_{lo} = a_{hi} a_{hi} \oplus a_{lo} a_{lo}$이므로 다음과 같이 식이 간단해집니다.

> $k \geq 1$일 때 $a < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $a \otimes a = \left( a_{hi} a_{hi}, \mathrm{half} _ {k-1}(a_{hi} a_{hi}) \oplus a_{lo} a_{lo}\right)$


```python
def square(a):
    k = calc_level(a)
    if k == 0:
        return a
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    ahi_ahi = square(ahi)
    return (
        ahi_ahi                            << 2**(k-1) |
        half(ahi_ahi, k - 1) ^ square(alo)
    )
```




## 제곱근


제곱 공식으로부터 다음과 같이 finite nimber에 대해 제곱근이 존재함을 알 수 있습니다.

> $k \geq 1$일 때 $a < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $\mathrm{sqrt}(a) = \left( \mathrm{sqrt}(a_{hi}), \mathrm{sqrt}(\mathrm{half} _ {k-1}(a_{hi}) \oplus a_{lo}) \right)$


```python
def sqrt(a):
    k = calc_level(a)
    if k == 0:
        return a
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    return (
        sqrt(ahi)                    << 2**(k-1) |
        sqrt(half(ahi, k - 1) ^ alo)
    )
```




## 곱셈 역원


체의 성질에서 $a > 0$에 대해 $a \otimes \mathrm{inv}(a) = 1$을 만족하는 $a$의 곱셈 역원 $\mathrm{inv}(a)$가 존재합니다. 편의상 $\mathrm{inv}(0) = 0$으로 정의하면 다음과 같이 계산할 수 있습니다.

> $k \geq 1$일 때 $a < 2^{2^k}$에 대해서 다음이 성립한다.
>
> $\mathrm{inv}(a) = \left( a_{hi} \otimes \mathrm{inv}\left( \mathrm{half} _ {k-1}(a_{hi} a_{hi}) \oplus a_{lo} (a_{hi} \oplus a_{lo}) \right),  (a_{hi} \oplus a_{lo}) \otimes \mathrm{inv}\left( \mathrm{half} _ {k-1}(a_{hi} a_{hi}) \oplus a_{lo} (a_{hi} \oplus a_{lo}) \right) \right)$

**증명**

$a = 0$일 때는 $a$를 대입하면 양변 모두 $0$이 나옵니다.

$a > 0$이라 합시다. $b = \mathrm{inv}(a)$라 하면 $a \otimes b = \left(a_{hi} b_{hi} \oplus a_{hi} b_{lo} \oplus a_{lo} b_{hi}, \mathrm{half} _ {k-1}(a_{hi} b_{hi}) \oplus a_{lo} b_{lo}\right) = (0, 1)$입니다. 

$hi$ 부분에서 $(a_{hi} \oplus a_{lo})b_{hi} = a_{hi} b_{lo}$의 식이 나옵니다. 따라서 어떤 $x$가 존재하여 $b_{hi} = a_{hi} x$이고 $b_{lo} = (a_{hi} \oplus a_{lo}) x$입니다. ($a > 0$에서 $a_{hi}$와 $(a_{hi} \oplus a_{lo})$ 중 하나 이상이 $0$이 아니므로 $x$가 존재합니다)

$b_{hi}$와 $b_{lo}$를 $x$에 대해 정리한 것을 $lo$ 부분 식에 대입하면 다음과 같습니다.

$\mathrm{half} _ {k-1}(a_{hi} b_{hi}) \oplus a_{lo} b_{lo} = \left( \mathrm{half} _ {k-1}(a_{hi} a_{hi}) \oplus a_{lo} (a_{hi} \oplus a_{lo}) \right) x = 1$

따라서 $x = \mathrm{inv}\left( \mathrm{half} _ {k-1}(a_{hi} a_{hi}) \oplus a_{lo} (a_{hi} \oplus a_{lo}) \right)$입니다.


```python
def inv(a):
    k = calc_level(a)
    if k == 0:
        return a
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    x = inv(half(square(ahi), k - 1) ^ mul(alo, ahi ^ alo))
    return (
        mul(ahi, x)       << 2**(k-1) |
        mul(ahi ^ alo, x)
    )
```




## 생성자를 이용한 전처리

위처럼 재귀적으로 nimber 연산을 계산할 때, $k$가 작은 경우를 전처리하면 더 빠르게 계산할 수 있습니다.

$2^{2^k}$ 미만의 nimber는 유한체를 이루므로, $1$ 이상 $2^{2^k}$ 미만의 nimber들은 nimber 곱에 대해 순환군을 이룹니다. 즉, 어떤 $1 \leq g < 2^{2^k}$가 존재하여 $g^{0} = 1$, $g^{n} = g \otimes g^{n - 1}$이라 할 때, $g^{0}$, $g^{1}$, &hellip;, $g^{2^{2^k} - 2}$는 모두 서로 다르고, $g^{0} = g^{2^{2^k} - 1} = 1$입니다.

따라서 $0$, $1$, &hellip;, $2^{2^k} - 2$과 $g^{0}$, $g^{1}$, &hellip;, $g^{2^{2^k} - 2}$ 사이의 일대일 대응을 미리 계산해두면 $g^{n} \otimes g^{m} = g^{n+m}$을 통해서 곱셈을 빠르게 계산할 수 있습니다.

전처리 기준으로 $k = 4$, 즉 $2^{2^k} = 65536$를 사용한다고 하면 $g = 258$을 사용하면 됩니다.

(아래 코드에서 전처리 부분의 `mul` 함수는 위에서 나온 `mul` 함수입니다)

```python
g = 258

index_to_pow_g = [1]
while len(index_to_pow_g) < 65535:
    index_to_pow_g.append(mul(index_to_pow_g[-1], g))

assert len(index_to_pow_g) == len(set(index_to_pow_g))
assert mul(index_to_pow_g[-1], g) == 1

pow_g_to_index = [-1] * 65536
for i, v in enumerate(index_to_pow_g):
    pow_g_to_index[v] = i

def mul(a, b):
    if max(a, b) < 65536:
        if a == 0 or b == 0: return 0
        idx_a = pow_g_to_index[a]
        idx_b = pow_g_to_index[b]
        return index_to_pow_g[(idx_a + idx_b) % 65535]
    k = max(calc_level(a), calc_level(b))
    if k == 0:
        return a & b
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    bhi = b >> 2**(k-1)
    blo = b & ((1 << 2**(k-1)) - 1)
    alo_blo = mul(alo, blo)
    return (
        (mul(ahi ^ alo, bhi ^ blo) ^ alo_blo) << 2**(k-1) |
        half(mul(ahi, bhi), k - 1) ^ alo_blo
    )
```




## 연습문제

다음은 [BOJ 18630번](https://www.acmicpc.net/problem/18630)을 푸는 코드입니다. 이 외에도 [BOJ 19103번](https://www.acmicpc.net/problem/19103), [BOJ 23132번](https://www.acmicpc.net/problem/23132) 등이 있습니다.

<details>
<summary>정답코드</summary>
<div markdown="1">

```python
def bootstrap():
    def calc_level(a):
        k = 0
        while a >= 2 ** 2**k:
            k += 1
        return k

    def half(a, k):
        if k == 0:
            return a
        ahi = a >> 2**(k-1)
        alo = a & ((1 << 2**(k-1)) - 1)
        return (
            half(ahi ^ alo, k - 1)        << 2**(k-1) |
            half(half(ahi, k - 1), k - 1)
        )

    def mul(a, b):
        k = max(calc_level(a), calc_level(b))
        if k == 0:
            return a & b
        ahi = a >> 2**(k-1)
        alo = a & ((1 << 2**(k-1)) - 1)
        bhi = b >> 2**(k-1)
        blo = b & ((1 << 2**(k-1)) - 1)
        alo_blo = mul(alo, blo)
        return (
            (mul(ahi ^ alo, bhi ^ blo) ^ alo_blo) << 2**(k-1) |
            half(mul(ahi, bhi), k - 1) ^ alo_blo
        )

    g = 258

    mul_g = [0] * 65536
    for i in range(1, 65536):
        if i & (i - 1):
            j1 = i & -i
            j2 = i ^ j1
            mul_g[i] = mul_g[j1] ^ mul_g[j2]
        else:
            mul_g[i] = mul(i, g)

    index_to_pow_g = [1]
    while len(index_to_pow_g) < 65535:
        index_to_pow_g.append(mul_g[index_to_pow_g[-1]])

    pow_g_to_index = [-1] * 65536
    for i, v in enumerate(index_to_pow_g):
        pow_g_to_index[v] = i

    return index_to_pow_g, pow_g_to_index

index_to_pow_g, pow_g_to_index = bootstrap()
index_to_pow_g *= 2

def half16(a):
    if a == 0: return 0
    idx_a = pow_g_to_index[a]
    return index_to_pow_g[idx_a + pow_g_to_index[1<<15]]

def half32(a):
    k = 5
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    return (
        half16(ahi ^ alo)   << 2**(k-1) |
        half16(half16(ahi))
    )

def mul16(a, b):
    if a == 0 or b == 0: return 0
    idx_a = pow_g_to_index[a]
    idx_b = pow_g_to_index[b]
    return index_to_pow_g[idx_a + idx_b]

def mul32(a, b):
    k = 5
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    bhi = b >> 2**(k-1)
    blo = b & ((1 << 2**(k-1)) - 1)
    alo_blo = mul16(alo, blo)
    return (
        (mul16(ahi ^ alo, bhi ^ blo) ^ alo_blo) << 2**(k-1) |
        half16(mul16(ahi, bhi)) ^ alo_blo
    )

def mul64(a, b):
    k = 6
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    bhi = b >> 2**(k-1)
    blo = b & ((1 << 2**(k-1)) - 1)
    alo_blo = mul32(alo, blo)
    return (
        (mul32(ahi ^ alo, bhi ^ blo) ^ alo_blo) << 2**(k-1) |
        half32(mul32(ahi, bhi)) ^ alo_blo
    )

def square16(a):
    if a == 0: return 0
    idx_a = pow_g_to_index[a]
    return index_to_pow_g[idx_a * 2]

def square32(a):
    k = 5
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    ahi_ahi = square16(ahi)
    return (
        ahi_ahi                         << 2**(k-1) |
        half16(ahi_ahi) ^ square16(alo)
    )

def inv16(a):
    if a == 0: return 0
    idx_a = pow_g_to_index[a]
    return index_to_pow_g[65535 - idx_a]

def inv32(a):
    k = 5
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    x = inv16(half16(square16(ahi)) ^ mul16(alo, ahi ^ alo))
    return (
        mul16(ahi, x)       << 2**(k-1) |
        mul16(ahi ^ alo, x)
    )

def inv64(a):
    k = 6
    ahi = a >> 2**(k-1)
    alo = a & ((1 << 2**(k-1)) - 1)
    x = inv32(half32(square32(ahi)) ^ mul32(alo, ahi ^ alo))
    return (
        mul32(ahi, x)       << 2**(k-1) |
        mul32(ahi ^ alo, x)
    )


import sys; input = sys.stdin.readline
n = int(input())
mat = [list(map(int, input().split())) for i in range(n)]

det = 1

for i in range(n):
    for j in range(i, n):
        if mat[j][i]:
            break
    if mat[j][i] == 0:
        det = 0
        break
    if j != i:
        mat[i], mat[j] = mat[j], mat[i]
    head = mat[i][i]
    det = mul64(det, head)
    invhead = inv64(head)
    for j in range(i + 1, n):
        if mat[j][i] == 0:
            continue
        weight = mul64(mat[j][i], invhead)
        mat[j][i] = 0
        for k in range(i + 1, n):
            mat[j][k] ^= mul64(weight, mat[i][k])

print('First' if det else 'Second')
```

</div>
</details>
