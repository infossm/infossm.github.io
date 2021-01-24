---
layout: post
title:  "Understanding CRC"
date:   2020-08-19 23:59
author: RBTree
tags: [mathematics, CTF]
---

# 서론

작년 중순, 한 CTF에서 CRC와 관련된 문제가 하나 나왔습니다. 문제의 요지인 즉슨, `flag{x}`의 CRC 값이 다시 `x`가 나오게 하는 `x`를 구하는 것이었습니다. 이 과정에서 CRC의 특성에 대해서 이해하고 있는 사람이 별로 없다는 것을 알게 되었습니다. 또한 네트워크 수업에서도 Error code를 소개하는 자리에서 CRC가 나왔지만, CRC를 구하는 방법에 대해서만 알려줄 뿐 CRC가 어떠한 특성을 가지고 있는지는 알려주지 않았습니다.

최근에도 CRC에 대한 CTF 문제가 나오고 있고, 저도 CRC에 대해서 헷갈리던 부분이 있어서 이번에 한 번 정리하게 되었습니다.

# 본론

## Finite Field

CRC를 이해하기에 앞서, CRC에서 사용되는 수학 개념인 Finite Field에 대해서 짚고 넘어가야 합니다.

Finite field는 field의 조건을 만족하는 유한집합을 의미합니다. 그러면 field는 무엇일까요? 간단하게 설명하자면, 덧셈과 곱셈, 뺄셈과 나눗셈이 모든 원소에 대해서 잘 정의되는 집합을 의미합니다. 물론 나눗셈은 나누는 값이 0일 때를 제외합니다. 가장 대표적인 예시는 소수 $p$에 대해 $\text{mod}\ p$로 정의되는 $\{0, 1, \ldots, p-1\}$ 집합입니다. 이 집합에서는 사칙연산이 잘 정의가 되는 것을 쉽게 확인할 수 있습니다.

Finite field에 대해서는 재밌는 사실이 많이 있습니다.

- Finite field의 크기는 항상 어떤 소수 $p$에 대해서 $p^k$ 꼴입니다. 또한 $p^k$ 크기의 finite field에서는, 똑같은 원소를 $p$번 더하면 항상 0이 됩니다. 그렇기에 $p^k$ 크기의 finite field에서의 $p$를 field의 charateristic이라고 합니다.
- 같은 크기의 finite field는 isomorphic합니다. 두 field가 isomorphic하다는 것을 간단히 설명하자면, 두 field 사이의 일대일 대응이 존재하며 이 일대일 대응에 따라 한 field에서 다른 field로 값들을 변환했을 때 똑같이 덧셈, 곱셈, 뺄셈과 나눗셈 연산 관계가 동일하게 성립한다는 것을 의미합니다.

크기 $q$인 finite field는 일반적으로 $\bold{F}_q$, $\mathbb{F}_q$, 혹은 $\text{GF}(q)$로 표기합니다. 이 글에서는 $\text{GF}(q)$ 표기를 사용하도록 하겠습니다.

그런데 소수 $p$에 대해서 $\text{GF}(p)$는 $\text{mod}\ p$로 쉽게 이해할 수 있는데, $\text{GF}(p^k)$는 어떻게 정의를 해야할까요? $\text{mod}\ p^k$로는$(p+1)/p$ 와 같이 $p$로 나눌 수가 없기 때문에 정의할 수가 없습니다.

이를 해결하기 위해서 우선 $\text{GF}(p)$를 확장해봅시다. $\text{GF}(p)[X]$는 $\text{GF}(p)$를 계수로 가지는 변수 $X$에 대한 다항식의 집합입니다. 예를 들어, $\text{GF}(2)[X]$ 위에서는 $0$, $1$, $X^2+1$, $X^3+X^2+1$와 같은 값들이 존재할 수 있습니다. 여기에서 유의할 점은, $2X$와 같은 값은 $\text{GF}(2)$상에서 2와 0은 동일하기 때문에 $0$과 같다는 점입니다.

이제 여기에서 $\text{GF}(p^k)$를 정의하기 위해서는 $k$차 기약 다항식 $P$가 필요합니다. 기약 다항식은 인수분해가 불가능한 다항식을 의미합니다. 예를 들어서, $\text{GF}(2)[X]$ 상에서는 $X^3+X+1$과 같은 식을 예로 들 수 있습니다. 반대로, $X^2+1$의 경우 $(X+1)^2$이기 때문에 기약 다항식이 아닙니다. 기약 다항식 $P$를 정하게 되면, $k$차 이상의 항 $aX^l$ 에 대해서 $-aX^{l-k}P$를 더해 $k$차 이상의 항을 전부 없앰으로써 $k-1$차 이하의 항만 남길 수 있습니다. 계수는 0부터 $p-1$까지 가능하니, $k-1$차 이하의 항들의 합으로 표현 가능한 값이 총 $p^k$일 것입니다. 이렇게 정의되는 집합을 $\text{GF}(p)[X]/(P)$라고 적으며, 이는 $GF(p^k)$입니다.

예를 들어서, 집합 $\text{GF}(2^3) = \text{GF}(2)[X]/(X^3+X+1)$을 생각해봅시다. 이 집합 위에서 $X^2 + X + 1$와 $X^2 + 1$을 곱하는 과정은 다음과 같습니다.

$(X^2 + X+1)(X^2+1) = X^4+X^3+X+1=X^4+X^3+X+1-X(X^3+X+1)=X^3+X^2+X+1=X^3+X^2+X+1-(X^3+X+1)=X^2+1$

## CRC

Cyclic redundancy check, 줄여서 CRC는 사실 주어진 string을 $\text{GF}(2^k)$ 위의 값으로 표현하는 과정입니다. 그래서 [위키피디아](https://en.wikipedia.org/wiki/Cyclic_redundancy_check#CRC_catalogues)를 살펴보면 CRC마다 사용되는 irreducible polynomial을 살펴볼 수 있습니다. 그런데 다음 CRC32 코드를 보면 $\text{GF}(2^{32})$와 쉽게 매칭이 되지 않습니다. 어떻게 된 것일까요?

```c
unsigned int crc32b(unsigned char *message) {
   int i, j;
   unsigned int byte, crc, mask;

   i = 0;
   crc = 0xFFFFFFFF;
   while (message[i] != 0) {
      byte = message[i];            // Get next byte.
      crc = crc ^ byte;
      for (j = 7; j >= 0; j--) {    // Do eight times.
         mask = -(crc & 1);
         crc = (crc >> 1) ^ (0xEDB88320 & mask);
      }
      i = i + 1;
   }
   return ~crc;
}
```

우선 $\text{GF}(2)$에서의 덧셈을 생각해보면, 사실 덧셈과 XOR은 차이가 없습니다. $0+0=0, 1+0=0, 0+1=0, 1+1=0$이기 때문입니다. 이를 연장해서 $\text{GF}(2)[X]$ 위에서의 덧셈을 생각해보면, 각 $k$차항의 계수끼리 XOR한 것이 덧셈한 결과의 $k$차항의 계수가 될 것입니다. 여기서 각 항의 계수를 이진수로 표현한다면, 두 수의 XOR이 결국 $\text{GF}(2)[X]$ 위에서의 덧셈과 동일하다는 것을 알 수 있습니다.

해당 코드에서는 `unsigned int`를 통해서 $\text{GF}(2^{32})$를 표현하고 있습니다. 그리고 while loop에서는 `0xFFFFFFFF`로 시작하는 `crc` 값에 계속 `message[i]` 값을 더하는 것을 알 수 있습니다.

그 다음 for loop를 보면, `crc & 1`이 0이라면 `mask`가 0이 되고, `crc & 1`이 1이라면 `mask`는 `0xFFFFFFFF`가 될 것입니다. 이를 `0xEDB88320`와 and 연산을 하므로, `crc & 1`이 1일 때만 `0xEDB88320`를 더하고 싶음을 알 수 있습니다. 이 과정의 의미가 뭘까요?

사실 이 과정에서 `crc`의 LSB는 $X^{31}$의 계수이고, MSB는 상수항입니다. 즉, 낮은 비트일수록 높은 차수의 항의 계수를 나타내고 있습니다. 해당 for loop는 총 8비트에 해당하는 `message[i]`를 `crc`에 더한 뒤, `crc` 값에 $X$를 8번 곱하는 과정을 나타냅니다. 이 때 사용되는 기약 다항식은 `0xEDB88320`로 표현되는 다항식에 $X^{32}$를 더한 것입니다. 

이 과정을 합쳐서 풀어 써봅시다. `message`의 길이가 $l$ 바이트이며, `message`를 단순히 이어붙여 이진수로 표현한 값을 $msg$라고 하고, $i$번째 바이트 값을 $msg_i$라고 합시다. `message[i]`를 `crc`에 더하는 과정은 하위 8비트에 더하는 것이기 때문에, 기본적으로 $X^{24}$를 $msg_i$에 곱한 뒤 $crc$에 더하고 $X^8$을 곱하는 것입니다. 그러므로, $msg \times X^{32} + \text{0xFFFFFFFF} \times X^{8l} + \text{0xFFFFFFFF}$ 가 바로 계산하고자 하는 값임을 이해할 수 있습니다. (마지막에 `~crc`를 반환하므로 `0xFFFFFFFF`를 한 번 더 더하는 것이 맞습니다.)

다른 CRC 방법도 살펴보면, 항상 이런 형태로 계산하지는 않습니다. `0xFFFFFFFF`가 아니라 `0`에서 시작하는 계산 방법도 있으며, 직관적으로 MSB가 최고차항의 계수이고 LSB가 상수항을 나타내는 계산 방법(Normal form)도 있습니다. 하지만 일반적으로 사용되는 방법들은 `0xFFFFFFFF`에서 시작해 계산한 뒤 `0xFFFFFFFFF`를 다시 더하고, LSB를 최고차항으로 삼는 Reversed form을 주로 사용합니다.

## Playing with CRC

이제 CRC를 좀 더 깊게 이해하기 위해서, Python을 통해 위의 CRC와 $\text{GF}(2^{32})$를 구현해봅시다. 여기에서는 구현의 편의상 Reversed form이 아닌 Normal form을 사용합니다.

```python
poly = 0x104C11DB7

def normalize(x):
    while x.bit_length() > 32:
        x ^= poly << (x.bit_length() - 33)
    return x

def mult(x, y):
    res = 0
    for i in range(32):
        if y & (1 << i) != 0:
            res ^= (x << i)
    return normalize(res)

def bytes_to_gf32(s):
    val = 0
    for c in s:
        rev = int(format(c, '08b')[::-1], 2)
        val = (val << 8) | rev
    return normalize(val)

def crc32(s):
    l = len(s)
    m = bytes_to_gf32(s)
    return normalize((m << 32) ^ (0xFFFFFFFF << (8 * l)) ^ 0xFFFFFFFF)

def crc32b(message):
    crc = 0xFFFFFFFF
    for i in range(len(message)):
        byte = message[i]
        crc = crc ^ byte
        for j in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

message = b"test"

crc1 = crc32(message)
crc2 = crc32b(message)

print(format(crc1, '032b'))
print(format(crc2, '032b')[::-1])
```

`crc32`는 식 $msg \times X^{32} + \text{0xFFFFFFFF} \times X^{8l} + \text{0xFFFFFFFF}$을 토대로 작성한 것이고, `crc32b`는 위에서 언급된 코드를 Python으로 옮긴 것입니다. 실행해보면 다음과 같이 동일하게 값이 나오는 것을 확인할 수 있습니다.

```
> python .\test.py
00110000011111101111111000011011
00110000011111101111111000011011
```

이제 마지막으로, 서론에서 언급했던 `flag{x}`의 CRC 값이 다시 `x`가 나오게 하는 문제를 떠올려봅시다. `x`는 32-bit, 즉 4-byte string입니다. $msg \times X^{32} + \text{0xFFFFFFFF} \times X^{8l} + \text{0xFFFFFFFF}$를 바탕으로 생각해보면 $"flag\{x\}"\times X^{32} + \text{0xFFFFFFFF} \times X^{8l} + \text{0xFFFFFFFF} = x$ 가 되어야 하므로, 일종의 일차방정식을 푸는 문제라고 생각할 수 있습니다. 그런데 이를 위해서는 나눗셈을 정의해야합니다.

다행히도 $\text{GF}(q)$ 상의 0이 아닌 원소 $a$에 대해, $a^{q-1} = 1$이 성립합니다. 이를 통해 Modular inverse를 $a^{q-2}$로 구할 수 있습니다. 이를 바탕으로 위의 $x$를 구하는 코드를 작성해봅시다.

```python
def pow(x, y):
    if y == 0:
        return 1
    elif y == 1:
        return x
    else:
        res = pow(x, y // 2)
        res = mult(res, res)
        if y & 1:
            res = mult(res, x)
        return res

def inverse(x):
    return pow(x, 2 ** 32 - 2)

const = crc32(b"flag{\0\0\0\0}")
coef = normalize((1 << 40) ^ 1)
x = mult(const, inverse(coef))

print(format(x, '032b')[::-1])
# 01110011 10011011 01000101 00000111
#     0x73     0x9b     0x45     0x07

print(hex(x))
print(hex(bytes_to_gf32(b"\x07\x45\x9b\x73")))
print(hex(crc32(b"flag{\x07\x45\x9b\x73}")))
print(hex(crc32b(b"flag{\x07\x45\x9b\x73}")))
```

```
> python .\test.py
01110011100110110100010100000111
0xe0a2d9ce
0xe0a2d9ce
0xe0a2d9ce
0x739b4507
```

원하는 `x`는 `\x07\x45\x9b\x73` 임을 알 수 있었습니다.

# 결론

CRC가 어떤 수학적 바탕을 두고 있는지 알아보고, CRC에 대해서 깊이 이해하는 시간을 가졌습니다. CRC 자체에 대해서도 이해함으로써 CRC가 어떤 string으로부터 나올 수 있는지, 한 CRC 값에 대해서 어떤 string들이 가능한지, 그리고 위와 같이 CRC와 관련된 특수한 식을 구하는 것 또한 가능함을 알 수 있었습니다.

비단 CRC가 아니더라도 finite field 자체가 다른 곳에서도 자주 쓰이는 개념이기 때문에, 나중에 헷갈리는 부분이 있을 때 참고할 수 있었으면 좋겠습니다.

# 참고 문헌

1. https://en.wikipedia.org/wiki/Finite_field
2. https://en.wikipedia.org/wiki/Cyclic_redundancy_check
3. https://stackoverflow.com/questions/21001659/crc32-algorithm-implementation-in-c-without-a-look-up-table-and-with-a-public-li