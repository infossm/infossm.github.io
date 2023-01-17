---
layout: post
title:  "Breaking Python random module"
date:   2021-05-18 23:00
author: RBTree
tags: [cryptography]
---

# 서론

PlaidCTF 2021에서 [Fake Medallion](https://plaidctf.com/challenge/5) 이라는 문제가 나왔습니다. 문제를 푸는 과정은 다음과 같았습니다.

1. Qubit 30개가 `|0>, |1>, |+>, |->` 4개 중 하나의 state로 존재합니다. 각 qubit마다 여분의 빈 2개의 qubit이 주어지는데, quantum teleportation과 probabilistic quantum cloning을 통해서 이 stage를 통과할 수 있었습니다. (자세한 설명은 이 글의 주제가 아니라서 생략하겠습니다.)
2. Qubit 30개를 초기화할 때 Python random 모듈의 `random.getrandbits(30)`을 사용합니다. 한 번 1번 stage를 성공하면 이 값들을 수많이 받아올 수 있습니다. 이 값들을 모은 뒤 다음 `random.getrandbits(30)`을 예측해 점수를 얻고, 이를 통해 문제를 풀 수 있습니다.

(혹시 풀이를 보고 싶으신 분은 [write-up](https://github.com/perfectblue/ctf-writeups/tree/master/2021/plaidctf-2021/fake-medallion)을 참조해주시기 바랍니다.)

흥미로운 부분은 바로 `random.getrandbits(30)` 을 통해서 복구하는 부분입니다. 제가 알기로는 Python random은 [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)를 사용하는데, 32-bit output 624개가 있을 때 state를 온전히 복구할 수 있다는 것 이외에는 처음 들어보았기 때문입니다. 문제를 풀 때 다른 팀원이 [not_random](https://github.com/fx5/not_random) 이라는 프로젝트의 코드를 사용했는데, 이번 기회에 어떻게 복구를 하는지 알아보기로 했습니다.

# 본론

## Mersenne Twister

Mersenne Twister는 624개의 32-bit integer를 state로 가지고 있으며, 매번 32-bit integer를 output으로 출력한 뒤 state를 변화시킵니다. 이를 알아보기 위해서 직접 Python의 random 구현을 살펴보기로 해봅시다. ([Link](https://github.com/python/cpython/blob/23362f8c301f72bbf261b56e1af93e8c52f5b6cf/Modules/_randommodule.c))

```c
/* Period parameters -- These are all magic.  Don't change. */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfU    /* constant vector a */
#define UPPER_MASK 0x80000000U  /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffU  /* least significant r bits */

static uint32_t
genrand_uint32(RandomObject *self)
{
    uint32_t y;
    static const uint32_t mag01[2] = {0x0U, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */
    uint32_t *mt;

    mt = self->state;
    if (self->index >= N) { /* generate N words at one time */
        int kk;

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1U];

        self->index = 0;
    }

    y = mt[self->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}
```

코드를 살펴보면

1. `self->index`는 호출될 때마다 1씩 늘어납니다. 만약 호출할 때 `self->index`가 `N` 이상이라면, 현재 state를 다음 state로 변화시킵니다.
2. `mt[self->index]`를 변형하여 output으로 내놓습니다.

State를 변경하는 과정은 각 `kk`마다 `y = (mt[kk] & UPPER_MASK) | (mt[(kk + 1) % N] & LOWER_MASK)` 를 취한 뒤, `mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ mag01[y & 0x1U]` 라는 과정을 거치는 것을 알 수 있습니다.

만약,

```c
    y = mt[self->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
```

이 과정을 역연산 할 수 있다면, state를 그대로 복구할 수 있기 때문에 624개의 32-bit output을 가지고 있으면 다음 값을 예측하는 것이 가능할 것입니다.

## Recovering from 32-bit full outputs

해당 과정을 temper라고 하는데, 다행히도 untemper하는 방법이 존재합니다. ([Link](http://mslc.ctf.su/wp-content/uploads/files/svalka/mt.py))

```python
TemperingMaskB = 0x9d2c5680
TemperingMaskC = 0xefc60000

def untemper(y):
    y = undoTemperShiftL(y)
    y = undoTemperShiftT(y)
    y = undoTemperShiftS(y)
    y = undoTemperShiftU(y)
    return y

def undoTemperShiftL(y):
    last14 = y >> 18
    final = y ^ last14
    return final

def undoTemperShiftT(y):
    first17 = y << 15
    final = y ^ (first17 & TemperingMaskC)
    return final

def undoTemperShiftS(y):
    a = y << 7
    b = y ^ (a & TemperingMaskB)
    c = b << 7
    d = y ^ (c & TemperingMaskB)
    e = d << 7
    f = y ^ (e & TemperingMaskB)
    g = f << 7
    h = y ^ (g & TemperingMaskB)
    i = h << 7
    final = y ^ (i & TemperingMaskB)
    return final

def undoTemperShiftU(y):
    a = y >> 11
    b = y ^ a
    c = b >> 11
    final = y ^ c
    return final

```

과정을 하나씩 살펴보면 쉽게 이해할 수 있을 것입니다. 결국 `y`와 `y`를 shift한 값을 **XOR** 하기 때문에 역연산을 어렵지 않게 해낼 수 있습니다. 이제 이를 바탕으로 테스트해봅시다.

```python
import random

state = random.getstate()

outputs = [ random.getrandbits(32) for _ in range(1000) ]
recovered_state = (3, tuple([ untemper(v) for v in outputs[:624] ] + [0]), None)
random.setstate(recovered_state)
for i in range(1000):
    assert outputs[i] == random.getrandbits(32)
```

`recovered_state`의 첫 번째 인자는 random version을 의미하는 것으로 알고 있습니다. 이는 무조건 3으로 설정합니다. 그 다음은 state + index 값을 넣어줍니다. 이 경우에는 복구한 state에 index를 0으로 설정해줍니다. 세 번째 인자는 `None`으로 설정합니다.

이렇게 복구한 뒤 비교해보면 `assert` 문에 걸리지 않고 잘 실행되는 것을 볼 수 있습니다.

## Recovering from partial outputs

그런데, 만약 32-bit output이 아니라 앞서 서론에서 언급한 문제처럼 `python.getrandbits(30)` 같이 partial output을 내놓는 경우는 어떻게 처리해야 할까요?

우선 `getrandbits` 구현을 살펴봅시다.

```c
static PyObject *
_random_Random_getrandbits_impl(RandomObject *self, int k)
/*[clinic end generated code: output=b402f82a2158887f input=8c0e6396dd176fc0]*/
{
    int i, words;
    uint32_t r;
    uint32_t *wordarray;
    PyObject *result;

    if (k < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "number of bits must be non-negative");
        return NULL;
    }

    if (k == 0)
        return PyLong_FromLong(0);

    if (k <= 32)  /* Fast path */
        return PyLong_FromUnsignedLong(genrand_uint32(self) >> (32 - k));

    words = (k - 1) / 32 + 1;
    wordarray = (uint32_t *)PyMem_Malloc(words * 4);
    if (wordarray == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    /* Fill-out bits of long integer, by 32-bit words, from least significant
       to most significant. */
#if PY_LITTLE_ENDIAN
    for (i = 0; i < words; i++, k -= 32)
#else
    for (i = words - 1; i >= 0; i--, k -= 32)
#endif
    {
        r = genrand_uint32(self);
        if (k < 32)
            r >>= (32 - k);  /* Drop least significant bits */
        wordarray[i] = r;
    }

    result = _PyLong_FromByteArray((unsigned char *)wordarray, words * 4,
                                   PY_LITTLE_ENDIAN, 0 /* unsigned */);
    PyMem_Free(wordarray);
    return result;
}
```

32비트 이하에 대해서는 `genrand_uint32(self) >> (32 - k)` 를 통해 상위 `k` 비트만 취하는 것을 살펴볼 수 있습니다. 그것보다 긴 경우에는 `genrand_uint32`를 여러번 호출해 long을 생성합니다.

즉, `getrandbits(30)` 과 같은 값을 받게 된다면 32-bit output의 상위 30비트를 알고 있다는 뜻입니다. untemper도 불가능한데, 과연 푸는 것이 가능할까요?

이를 알기 위해서는 다시 state를 변경하는 과정, 그리고 temper 과정을 살펴볼 필요가 있습니다.

> State를 변경하는 과정은 각 `kk`마다 `y = (mt[kk] & UPPER_MASK) | (mt[(kk + 1) % N] & LOWER_MASK)` 를 취한 뒤, `mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ mag01[y & 0x1U]` 라는 과정을 거치는 것을 알 수 있습니다.

```c
    y = mt[self->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
```

자세히 관찰해보면, 대부분의 연산이 XOR로 이루어져 있음을 확인할 수 있습니다. 유일하게 걸리는 과정이 AND와 OR 연산, 그리고 `mag01[y & 01U]` 과정입니다. 그러나 `UPPER_MASK`랑 `LOWER_MASK`는 우선 서로 겹치지 않습니다! 즉 `y`는 `mt[kk]`의 상위 한 비트를 떼오고, `mt[(kk + 1) % N]`의 하위 31비트를 떼어내 붙인 것이라는 것을 알 수 있습니다.

```c
#define MATRIX_A 0x9908b0dfU    /* constant vector a */
#define UPPER_MASK 0x80000000U  /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffU  /* least significant r bits */
```

또한 `mag01[y & 0x1U]`는 `y & 1`이 1일 경우 `MATRIX_A`, 아닐 경우 `0`을 취하는 연산임을 알 수 있습니다. 예를 들어서 `y & 1`이 a였다고 합시다. 그러면 다음과 같이 생각해볼 수 있습니다.

```
MATRIX_A =        0b 10011001000010001011000011011111
mag01[y & 0x1U] = 0b a00aa00a0000a000a0aa0000aa0aaaaa
```

그러므로, 다음과 같이 생각해봅시다: 32-bit integer를 32개의 비트가 이어진 array라고 생각해봅시다. 맨 처음 state는 624개의 32-bit 정수로 이루어져있으므로, 624 * 32 = 19968개의 1-bit 변수로 이루어져 있다고 생각할 수 있습니다. 그러면 다음 state를 계산하는 과정은 비트끼리의 XOR로만 구성되어있을 뿐이므로, 다음 state의 각 32-bit 정수의 각 bit 또한 이 19968개의 1-bit 변수들의 XOR로 표현이 가능합니다. AND, OR은 필요가 없습니다!

즉 이는 $GF(2)$ 위에서의 19968개의 변수의 선형방정식으로 볼 수 있습니다. 충분한 수의 선형방정식이 모인다면, matrix inversion을 통해서 19968개 변수의 값을 역연산하는 것이 가능할 것입니다.

## Implementation

이제 구현을 해보도록 하겠습니다. 19968개의 변수로 이루어진 선형방정식을 어떻게 나타낼지 고민을 많이 해보았는데, integer를 사용한 bitmask 방식이 충분히 효율적이지 않을까 해 이를 사용하기로 했습니다.

(Python의 `bitarray` 라이브러리를 사용해보는 것도 고려해보고 사용해보았지만, 하단의 Solver쪽 코드에서 속도가 지나치게 느려서 사용하지 못했습니다. 혹시 $GF(2)$ Matrix를 다룰 수 있는 더 나은 라이브러리가 있다면 추천 받습니다.)

우선 다음과 같이 19968개의 변수에 대한 bitmask를 반환하는 Twister 를 작성했습니다.

```python
class Twister:
    N = 624
    M = 397
    A = 0x9908b0df

    def __init__(self):
        self.state = [ [ (1 << (32 * i + (31 - j))) for j in range(32) ] for i in range(self.N)]
        self.index = 0
    
    @staticmethod
    def _xor(a, b):
        return [x ^ y for x, y in zip(a, b)]
    
    @staticmethod
    def _and(a, x):
        return [ v if (x >> (31 - i)) & 1 else 0 for i, v in enumerate(a) ]
    
    @staticmethod
    def _shiftr(a, x):
        return [0] * x + a[:-x]
    
    @staticmethod
    def _shiftl(a, x):
        return a[x:] + [0] * x

    def get32bits(self):
        if self.index >= self.N:
            for kk in range(self.N):
                y = self.state[kk][:1] + self.state[(kk + 1) % self.N][1:]
                z = [ y[-1] if (self.A >> (31 - i)) & 1 else 0 for i in range(32) ]
                self.state[kk] = self._xor(self.state[(kk + self.M) % self.N], self._shiftr(y, 1))
                self.state[kk] = self._xor(self.state[kk], z)
            self.index = 0

        y = self.state[self.index]
        y = self._xor(y, self._shiftr(y, 11))
        y = self._xor(y, self._and(self._shiftl(y, 7), 0x9d2c5680))
        y = self._xor(y, self._and(self._shiftl(y, 15), 0xefc60000))
        y = self._xor(y, self._shiftr(y, 18))
        self.index += 1

        return y
    
    def getrandbits(self, bit):
        return self.get32bits()[:bit]
```

`state`는 32개의 bitmask로 이루어진 list를 총 624개 담고 있습니다. 32개의 bitmask는 big endian 순으로 정렬되어 있음에 유의합시다.

그리고 이를 바탕으로 다음과 같이 Solver를 작성했습니다.

```python
class Solver:
    def __init__(self):
        self.equations = []
        self.outputs = []
    
    def insert(self, equation, output):
        for eq, o in zip(self.equations, self.outputs):
            lsb = eq & -eq
            if equation & lsb:
                equation ^= eq
                output ^= o
        
        if equation == 0:
            return

        lsb = equation & -equation
        for i in range(len(self.equations)):
            if self.equations[i] & lsb:
                self.equations[i] ^= equation
                self.outputs[i] ^= output
    
        self.equations.append(equation)
        self.outputs.append(output)
    
    def is_solvable(self):
        print(len(self.equations))
        return len(self.equations) == 624 * 32
    
    def solve(self):
        if not self.is_solvable():
            assert False, "Not solvable"
        
        num = 0
        for i, eq in enumerate(self.equations):
            assert eq == (eq & -eq), "Should be reduced now"
            if self.outputs[i]:
                num |= eq
        
        state = [ (num >> (32 * i)) & 0xFFFFFFFF for i in range(624) ][::-1]
        return state
```

`insert()`를 통해 bitmask와 그에 해당하는 output bit를 설정해주면, 현재 존재하는 bitmask들로 표현이 가능한지 확인하고 아니라면 `self.equations`에 추가합니다. (즉, matrix의 rank가 늘어난다면 추가합니다.)

그러나 이렇게 작성하고 다음과 같이 풀이를 작성해보았지만 동작하지 않았습니다. 귀신이 곡할 노릇이 아닐 수 없습니다.

```python
import random

num = 1500
bit = 30
twister = Twister()
outputs = [ random.getrandbits(bit) for _ in range(num) ]
equations = [ twister.getrandbits(bit) for _ in range(num) ]

solver = Solver()
for i in range(num):
    for j in range(bit):
        print(i, j)
        solver.insert(equations[i][j], outputs[i] >> (bit - 1 - j))
        if solver.is_solvable():
            break
    if solver.is_solvable():
        break

state = solver.solve()
recovered_state = (3, tuple(state + [0]), None)
random.setstate(recovered_state)

for i in range(num):
    assert outputs[i] == random.getrandbits(bit)
```

아무리 시도해보아도 Solver의 equation 수가 19968개에 도달하지 못했습니다. 이상해서 모든 bitmask를 OR해서 확인해보았습니다.

```python
num = 1500
bit = 30
twister = Twister()
outputs = [ random.getrandbits(bit) for _ in range(num) ]
equations = [ twister.getrandbits(bit) for _ in range(num) ]

v = 0
for i in range(num):
    for j in range(bit):
        v |= equations[i][j]

print(format(v, '19968b'))
```

그 결과는 충격적이게도 다음과 같았습니다.

```
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111...11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111101
```

즉, 어떻게 해도 state의 첫 번째 값의 아래에서 두 번째 bit를 구할 수 없다는 뜻이었습니다.

## Missing bits

`getrandbits`의 인자에 따라서 사라진 비트가 몇 개가 되는지 궁금해서 다음과 같이 코드를 작성해보았습니다.

```python
for bit in range(1, 33):
    twister = Twister()
    prev, prev_cnt, v, cnt = 0, 0, 0, 0
    while True:
        cnt += 1
        eqs = twister.getrandbits(bit)
        for eq in eqs:
            v |= eq
        
        if prev == v:
            prev_cnt += 1
        else:
            prev, prev_cnt = v, 1
        
        if prev_cnt == 100:
            break
    
    print(bit, cnt - prev_cnt + 1, format(prev, '19968b').count('1'))
```

그 결과는 다음과 같았습니다.

```
1 10592 19940
2 9969 19943
3 4985 19948
4 4985 19950
5 3739 19954
6 3739 19957
7 2493 19959
8 1870 19963
9 1870 19964
10 1247 19965
11 1247 19965
12 1247 19965
13 1247 19965
14 1247 19965
15 1247 19965
16 1247 19965
17 1247 19965
18 1247 19965
19 1247 19965
20 1247 19965
21 1247 19965
22 1247 19965
23 1247 19965
24 1247 19965
25 1247 19965
26 1247 19966
27 1247 19966
28 1247 19967
29 1247 19967
30 1247 19967
31 624 19968
32 624 19968
```

즉, `getrandbits(30)`은 1247개 정수를 받으면 19967 비트의 정보를 담고 있지만, 나머지 1비트는 10000번을 더 시행해도 찾을 수 없다는 의미입니다. Mersenne Twister의 구조상 모든 비트가 골고루 들어갈 것이라고 생각했는데, 그렇지 않아서 무척 당황스러웠습니다.

이 부분에서 검증하는데 시간을 상당히 많이 썼지만, 결론적으로 실제로 해당 비트는 알아낼 수 없다는 결론을 얻을 수 있었습니다. 즉, 초기 state의 특정 bit는 32-bit output의 특정 위치에만 영향을 준다는 것입니다.

그러므로 free variable을 무시하고 푸는 코드로 다시 작성해야 했습니다.

## Final code

최종 코드는 다음과 같습니다.

```python
class Twister:
    N = 624
    M = 397
    A = 0x9908b0df

    def __init__(self):
        self.state = [ [ (1 << (32 * i + (31 - j))) for j in range(32) ] for i in range(624)]
        self.index = 0
    
    @staticmethod
    def _xor(a, b):
        return [x ^ y for x, y in zip(a, b)]
    
    @staticmethod
    def _and(a, x):
        return [ v if (x >> (31 - i)) & 1 else 0 for i, v in enumerate(a) ]
    
    @staticmethod
    def _shiftr(a, x):
        return [0] * x + a[:-x]
    
    @staticmethod
    def _shiftl(a, x):
        return a[x:] + [0] * x

    def get32bits(self):
        if self.index >= self.N:
            for kk in range(self.N):
                y = self.state[kk][:1] + self.state[(kk + 1) % self.N][1:]
                z = [ y[-1] if (self.A >> (31 - i)) & 1 else 0 for i in range(32) ]
                self.state[kk] = self._xor(self.state[(kk + self.M) % self.N], self._shiftr(y, 1))
                self.state[kk] = self._xor(self.state[kk], z)
            self.index = 0

        y = self.state[self.index]
        y = self._xor(y, self._shiftr(y, 11))
        y = self._xor(y, self._and(self._shiftl(y, 7), 0x9d2c5680))
        y = self._xor(y, self._and(self._shiftl(y, 15), 0xefc60000))
        y = self._xor(y, self._shiftr(y, 18))
        self.index += 1

        return y
    
    def getrandbits(self, bit):
        return self.get32bits()[:bit]

class Solver:
    def __init__(self):
        self.equations = []
        self.outputs = []
    
    def insert(self, equation, output):
        for eq, o in zip(self.equations, self.outputs):
            lsb = eq & -eq
            if equation & lsb:
                equation ^= eq
                output ^= o
        
        if equation == 0:
            return

        lsb = equation & -equation
        for i in range(len(self.equations)):
            if self.equations[i] & lsb:
                self.equations[i] ^= equation
                self.outputs[i] ^= output
    
        self.equations.append(equation)
        self.outputs.append(output)
    
    def solve(self):
        num = 0
        for i, eq in enumerate(self.equations):
            if self.outputs[i]:
                # Assume every free variable is 0
                num |= eq & -eq
        
        state = [ (num >> (32 * i)) & 0xFFFFFFFF for i in range(624) ]
        return state

import random

num = 1247
bit = 30
twister = Twister()
outputs = [ random.getrandbits(bit) for _ in range(num) ]
equations = [ twister.getrandbits(bit) for _ in range(num) ]

solver = Solver()
for i in range(num):
    for j in range(bit):
        print(i, j)
        solver.insert(equations[i][j], (outputs[i] >> (bit - 1 - j)) & 1)

state = solver.solve()
recovered_state = (3, tuple(state + [0]), None)
random.setstate(recovered_state)

for i in range(num):
    assert outputs[i] == random.getrandbits(bit)
```

실제로 30비트 이외에도 앞서 언급한 not_random 프로젝트와 같은 8비트에 대해서도 잘 풀어내는 것을 확인할 수 있었습니다. 


# 결론

이번 기회에 Mersenne Twister의 일부 비트만으로 복구하는 방법에 대해서 알아봤습니다. Mersenne Twister에 대해서 깊이 이해할 수 있는 시간을 가질 수 있었으며, 특히 32-bit output 중 일부 bit만 사용하게 되면 모든 state를 복구하지 못한다는 점은 흥미로운 부분이었습니다. Mersenne Twister의 구조를 살펴보면 계속 state 값을 shift (`y >> 1`) 하기 때문에 언젠가 모든 비트에 모든 state의 값들이 영향을 줄 것이라고 생각했는데, 그렇지 않았다는 것이 놀라운 부분이 아닐 수 없습니다.

또한 이해할 수 없는 점은, 앞서 언급한 not_random 프로젝트는 8-bit output을 3115개를 얻어내서 풀어내며, 3115개를 통해서 **완전히 복구**할 수 있다고 주장하고 있습니다. 하지만 결과 상으로는 8비트 output은 1870개가 있을 때 최대 19963비트만을 복구할 수 있다는 것을 알 수 있었습니다.  (실제로 복구하기 위해서는 equation 개수가 더 늘어나야 할 것입니다. `1870 * 8 = 14960`이기 때문에...) 즉, not_random이 완벽하게 state를 복구해내지는 못하고 있다는 것입니다. 아마도 후일 다른 사람이 이를 검증해주고 새로 코드를 작성해줄 것이라고 믿습니다.

그리고 현재 구현에서는 `solver.insert`를 통해서 equation과 output을 넣을 때 따로 output이 `[0, 1]` 내인지 확인하지 않고 있습니다. 사실 이 부분을 활용해서 output에 0 또는 1 대신 bitmask를 넣으면 훌륭하게 matrix inverse를 구하는 것도 가능할지 모릅니다.

마지막으로, 현재 구현은 Python long int를 활용한 bitmask를 통해서 $GF(2)$ 상의 linear system을 풀어내고 있습니다. 하지만 분명 C 언어 등으로 작성된 더 효율적인 라이브러리가 있을 것으로 생각됩니다. 해당 라이브러리를 활용해서 속도를 더 늘릴 수 있다면 좋을 것으로 생각됩니다. (현재 30-bit output 1257개로부터 복구하는데는 11분 36초, 8-bit output 3000개로부터 8분 53초가 걸렸습니다.)

# 참고 문헌

1. PlaidCTF 2021 Fake Medallion https://plaidctf.com/challenge/5
2. Mersenne Twister https://en.wikipedia.org/wiki/Mersenne_Twister
3. Python random implementation https://github.com/python/cpython/blob/23362f8c301f72bbf261b56e1af93e8c52f5b6cf/Modules/_randommodule.c
4. not_random https://github.com/fx5/not_random

