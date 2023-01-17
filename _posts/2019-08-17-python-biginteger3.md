---
layout: post
title:  "Python의 큰 정수 표현법 3 - 기타 연산"
date:   2019-08-17-17:34
author: djm03178
tags: [big-integer, Python]
---

## 이전 글 ##
* [Python의 큰 정수 표현법 1](http://www.secmem.org/blog/2019/06/16/python-biginteger)
* [Python의 큰 정수 표현법 2](http://www.secmem.org/blog/2019/07/21/python-biginteger2)

## 개요 ##
지난 글들에서는 Python의 `int`가 큰 정수를 표현하기 위해 어떤 구조를 사용하고, 그 값을 C의 기본 자료형이나 문자열로  변환하는 방법이 무엇인지, 오브젝트끼리의 비교를 어떻게 하는지, 사칙연산은 어떻게 적용시키는지 등에 대해 알아보았습니다. 이번 글에서는 한 걸음 더 나아가, 지금까지 다루지 않은 몇 가지 연산들과 기타 기능들에 대해 알아보겠습니다.

## 단항 연산자 ##
본격적으로 어렵고 복잡한 연산자들에 들어가기 전에, 간단한 단항 연산자들을 몇 가지 살펴봅시다..

### 부호 변경 ###
`-x` 에 해당하는 연산자입니다. Python이 `int`를 저장하는 방식 덕분에, 매우 간단한 코드로 구현이 됩니다.

```c
static PyObject *
long_neg(PyLongObject *v)
{
    PyLongObject *z;
    if (Py_ABS(Py_SIZE(v)) <= 1)
        return PyLong_FromLong(-MEDIUM_VALUE(v));
    z = (PyLongObject *)_PyLong_Copy(v);
    if (z != NULL)
        Py_SIZE(z) = -(Py_SIZE(v));
    return (PyObject *)z;
}
```

크기가 1인 경우에 대한 간단한 예외 처리 후, 단순히 오브젝트를 복제해서 크기에만 마이너스를 붙여주는 것을 볼 수 있습니다.

### 절댓값 ###
내장 함수 `abs`의 기능입니다. 양수이면 그대로, 음수이면 위의 부호 변경 연산을 수행합니다.
```c
static PyObject *
long_abs(PyLongObject *v)
{
    if (Py_SIZE(v) < 0)
        return long_neg(v);
    else
        return long_long((PyObject *)v);
}
```

### 참 / 거짓 ###
Python에서는 정수 $0$은 거짓, 그 외의 정수값은 모두 참으로 간주합니다. 단순히 그 값이 $0$인지 아닌지 `ob_size`를 비교하기만 하면 됩니다.

```c
static int
long_bool(PyLongObject *v)
{
    return Py_SIZE(v) != 0;
}
```

## 해시 ##
Python에서는 `int` 오브젝트에 대한 `hash()` 함수를 기본으로 제공합니다. CPython에서의 해시의 결과는 오브젝트의 값을 $2^{31}-1$로 나눈 나머지이고, 부호가 유지됩니다(ex: `hash(-2**31-5)` = $-6$). 한 가지 예외가 있는데, 이 값이 $-1$인 경우에는 특별히 $-2$로 변경됩니다. 즉, `hash(-1)` = `hash(-2)` = $-2$입니다. 이렇게 하는 이유는 CPython에서는 $-1$이라는 반환값을 에러를 나타내는 값으로 예약해두었기 때문에 이와 충돌을 피하기 위함입니다.

코드는 다음과 같습니다.

```c
static Py_hash_t
long_hash(PyLongObject *v)
{
    Py_uhash_t x;
    Py_ssize_t i;
    int sign;

    i = Py_SIZE(v);
    switch(i) {
    case -1: return v->ob_digit[0]==1 ? -2 : -(sdigit)v->ob_digit[0];
    case 0: return 0;
    case 1: return v->ob_digit[0];
    }
    sign = 1;
    x = 0;
    if (i < 0) {
        sign = -1;
        i = -(i);
    }
    while (--i >= 0) {
        /* Here x is a quantity in the range [0, _PyHASH_MODULUS); we
           want to compute x * 2**PyLong_SHIFT + v->ob_digit[i] modulo
           _PyHASH_MODULUS.
           The computation of x * 2**PyLong_SHIFT % _PyHASH_MODULUS
           amounts to a rotation of the bits of x.  To see this, write
             x * 2**PyLong_SHIFT = y * 2**_PyHASH_BITS + z
           where y = x >> (_PyHASH_BITS - PyLong_SHIFT) gives the top
           PyLong_SHIFT bits of x (those that are shifted out of the
           original _PyHASH_BITS bits, and z = (x << PyLong_SHIFT) &
           _PyHASH_MODULUS gives the bottom _PyHASH_BITS - PyLong_SHIFT
           bits of x, shifted up.  Then since 2**_PyHASH_BITS is
           congruent to 1 modulo _PyHASH_MODULUS, y*2**_PyHASH_BITS is
           congruent to y modulo _PyHASH_MODULUS.  So
             x * 2**PyLong_SHIFT = y + z (mod _PyHASH_MODULUS).
           The right-hand side is just the result of rotating the
           _PyHASH_BITS bits of x left by PyLong_SHIFT places; since
           not all _PyHASH_BITS bits of x are 1s, the same is true
           after rotation, so 0 <= y+z < _PyHASH_MODULUS and y + z is
           the reduction of x*2**PyLong_SHIFT modulo
           _PyHASH_MODULUS. */
        x = ((x << PyLong_SHIFT) & _PyHASH_MODULUS) |
            (x >> (_PyHASH_BITS - PyLong_SHIFT));
        x += v->ob_digit[i];
        if (x >= _PyHASH_MODULUS)
            x -= _PyHASH_MODULUS;
    }
    x = x * sign;
    if (x == (Py_uhash_t)-1)
        x = (Py_uhash_t)-2;
    return (Py_hash_t)x;
}
```

이전부터 여러 번 보았듯이 여기에도 수가 작은 경우에 대한 빠른 처리를 위해 자릿수가 1인 경우를 예외로 처리하고 있습니다. 수의 절댓값이 $2^{30}$ 미만이므로, 자릿수가 한 자리일 때의 해시값은 그 값 자체임을 알 수 있습니다. 위에서 설명한 대로, 음수일 경우 값이 $-1$인 경우에만 특별히 $-2$로 바꾸어주고 있습니다.

그 외의 경우에는 나이브한 방법을 통해 $2^{31}-1$로 나눈 나머지를 구해야 합니다. `int` 오브젝트가 저장된 형태와 모듈로 값의 관계 덕분에, CPU에게 있어 비교적 빠른 연산인 비트 연산자들 위주로 해시값을 구할 수 있습니다. 가장 높은 자리부터 하나씩 `_PyHASH_MODULUS`로 나눈 나머지를 구하고 더해가는 식으로 최종 해시값을 구하게 됩니다. 여기서도 마찬가지로 결과가 $-1$인 경우에는 $-2$로 바꾸어주는 예외 처리가 되어 있습니다.

## 비트 연산자 ##
Python의 `int`는 각종 비트 연산도 지원합니다. 물론 CPU에게 `int` 오브젝트 하나에 대해 통째로 비트 연산을 수행하게 할 수는 없지만, `int` 오브젝트의 구성도 각 자리가 일반 레지스터와 비슷하게 2의 거듭제곱 단위이므로 유사한 방법으로 비트 연산들도 수행할 수 있습니다.

### 시프트 ###
대부분의 상용 언어들, 그리고 CPU 자체에서 지원하는 시프트 연산을 Python에서도 지원합니다. 비트를 전부 왼쪽으로 옮기는 왼쪽 시프트와, 오른쪽으로 옮기는 오른쪽 시프트 연산이 모두 제공됩니다.

시프트를 수행하는 최상위 함수는 `long_lshift`와 `long_rshift`지만, 그 전에 한 가지 보고 넘어갈 것이 있습니다. 시프트를 한 자리의 크기 내에서 움직일 정도로만 한다면, 한 자릿수에서 잘려나간 나머지를 다음 자리로 넘겨 붙이는 것을 반복하면 되지만, 그보다 더 큰 폭을 시프트하려면 어떻게 해야 할까요? 그런 경우 통째로 옮길 자릿수와 그 후 각 자리에서 시프트할 폭을 나누어주게 됩니다. 이를 구해주는 것이 `divmod_shift` 함수입니다.

```c
/* wordshift, remshift = divmod(shiftby, PyLong_SHIFT) */
static int
divmod_shift(PyObject *shiftby, Py_ssize_t *wordshift, digit *remshift)
{
    assert(PyLong_Check(shiftby));
    assert(Py_SIZE(shiftby) >= 0);
    Py_ssize_t lshiftby = PyLong_AsSsize_t((PyObject *)shiftby);
    if (lshiftby >= 0) {
        *wordshift = lshiftby / PyLong_SHIFT;
        *remshift = lshiftby % PyLong_SHIFT;
        return 0;
    }
    ...
}
```

가장 윗줄의 주석이 이 함수의 동작을 간단하게 요약해주고 있습니다. `PyLong_SHIFT`, 즉 $30$으로 나눈 몫과 나머지를 구해서 `wordshift`와 `remshift`에 각각 담아주는 역할입니다. 정상적인 경우라면 여기까지 실행되고 바로 반환할 것이고, 아래의 생략된 부분에서는 오버플로가 발생한 경우에 대한 예외 처리를 해줍니다.

#### 왼쪽 시프트 ####
우선 (제 경험상) 더 자주 사용하게 되는 왼쪽 시프트 연산부터 알아봅시다. 기본적으로 먼저 호출되는 함수는 `long_lshift`입니다.

```c
static PyObject *
long_lshift(PyObject *a, PyObject *b)
{
    Py_ssize_t wordshift;
    digit remshift;

    CHECK_BINOP(a, b);

    if (Py_SIZE(b) < 0) {
        PyErr_SetString(PyExc_ValueError, "negative shift count");
        return NULL;
    }
    if (Py_SIZE(a) == 0) {
        return PyLong_FromLong(0);
    }
    if (divmod_shift(b, &wordshift, &remshift) < 0)
        return NULL;
    return long_lshift1((PyLongObject *)a, wordshift, remshift);
}
```
우선 음수만큼 시프트하는 것은 정의되어있지 않으므로 이 경우에 대한 예외 처리가 먼저 들어갑니다. 그 후 여기서도 또한 신속한 처리를 위해 $0$을 시프트하려는 경우, 즉, `b`의 값에 관계 없이 무조건 결과가 $0$이 되는 경우를 재빠르게 잡아 바로 $0$을 반환해버립니다.

그 다음에는 위에서 살펴본 `divmod_shift` 함수를 호출해서 시프트해야 할 자릿수와, 각 자리에서 시프트할 양을 분리하여 `long_lshift1` 함수에 넘겨줍니다.

```c
static PyObject *
long_lshift1(PyLongObject *a, Py_ssize_t wordshift, digit remshift)
{
    /* This version due to Tim Peters */
    PyLongObject *z = NULL;
    Py_ssize_t oldsize, newsize, i, j;
    twodigits accum;

    oldsize = Py_ABS(Py_SIZE(a));
    newsize = oldsize + wordshift;
    if (remshift)
        ++newsize;
    z = _PyLong_New(newsize);
    if (z == NULL)
        return NULL;
    if (Py_SIZE(a) < 0) {
        assert(Py_REFCNT(z) == 1);
        Py_SIZE(z) = -Py_SIZE(z);
    }
    for (i = 0; i < wordshift; i++)
        z->ob_digit[i] = 0;
    accum = 0;
    for (i = wordshift, j = 0; j < oldsize; i++, j++) {
        accum |= (twodigits)a->ob_digit[j] << remshift;
        z->ob_digit[i] = (digit)(accum & PyLong_MASK);
        accum >>= PyLong_SHIFT;
    }
    if (remshift)
        z->ob_digit[newsize-1] = (digit)accum;
    else
        assert(!accum);
    z = long_normalize(z);
    return (PyObject *) maybe_small_long(z);
}
```

우선 시프트의 결과로 만들어질 오브젝트의 크기를 미리 계산합니다. 단, 이 크기는 정확한 크기는 아니고 상한선을 대충 잡아놓은 것입니다. 왼쪽으로 시프트를 하니 기존의 크기에 시프트할 자릿수 `wordshift`만큼이 추가로 들어가고, 여기에 `remshift`가 조금이라도 있다면 한 자리가 더 늘어날 가능성이 있기 때문에 그런 경우 하나를 더 증가시켜줍니다.

새로운 오브젝트를 만들고 부호를 지정해준 뒤 가장 먼저 하는 일은 하위 `wordshift`개의 자릿수를 $0$으로 채우는 것입니다. 왼쪽으로 시프트할 때 새로 만들어지는 하위 비트들은 $0$이 된다는 사실을 반영한 것입니다.

그 후 나머지에 대해서는 그 자리 내에서만 시프트를 하고, 넘쳐흐르는 부분을 다음 자리로 넘겨주는 것을 반복합니다. 이전 자리에서 넘어온 값 `accum`에 현재 자리를 `remshift`만큼 왼쪽으로 시프트한 값을 더하고 하위 30비트를 새로운 현재 자리의 값으로 하게 됩니다. `accum`은 하위 30비트를 제외한 나머지 부분만을 다시 취하고 다음 자리로 넘어가게 됩니다.

만일 `remshift`가 $0$이 아니라면 마지막 자리까지 수행한 후에도 `accum`이 남아있을 수 있습니다. 이를 오브젝트의 최상위 자리에 넣어줍니다. 하지만 기존의 최상위 자리를 시프트한 후에도 새로운 자리까지 영향이 미치지 않아 이 값이 $0$으로 남아있을 수도 있습니다. 이러면 오브젝트의 최상위 자리에 불필요한 $0$이 남게 됩니다. 이를 제거하기 위해 실행되는 함수가 `long_normalize`입니다. 이 함수는 시프트 외에도 코드 곳곳에서 종종 볼 수 있습니다.

#### 오른쪽 시프트 ####
왼쪽 시프트를 살펴보았으니, 오른쪽 시프트에 대해서도 알아봅시다. 기본적인 구조나 동작 원리는 비슷합니다. 하지만 큰 차이점도 하나 존재합니다. 우선 기본적으로 먼저 호출되는 함수는 다음과 같습니다.

```c
static PyObject *
long_rshift(PyObject *a, PyObject *b)
{
    Py_ssize_t wordshift;
    digit remshift;

    CHECK_BINOP(a, b);

    if (Py_SIZE(b) < 0) {
        PyErr_SetString(PyExc_ValueError, "negative shift count");
        return NULL;
    }
    if (Py_SIZE(a) == 0) {
        return PyLong_FromLong(0);
    }
    if (divmod_shift(b, &wordshift, &remshift) < 0)
        return NULL;
    return long_rshift1((PyLongObject *)a, wordshift, remshift);
}
```
왼쪽 시프트와 거의 똑같습니다. 마찬가지로 구체적인 오른쪽 시프트 동작을 위해 `long_rshift1`이 호출됩니다.

```c
static PyObject *
long_rshift1(PyLongObject *a, Py_ssize_t wordshift, digit remshift)
{
    PyLongObject *z = NULL;
    Py_ssize_t newsize, hishift, i, j;
    digit lomask, himask;

    if (Py_SIZE(a) < 0) {
        /* Right shifting negative numbers is harder */
        PyLongObject *a1, *a2;
        a1 = (PyLongObject *) long_invert(a);
        if (a1 == NULL)
            return NULL;
        a2 = (PyLongObject *) long_rshift1(a1, wordshift, remshift);
        Py_DECREF(a1);
        if (a2 == NULL)
            return NULL;
        z = (PyLongObject *) long_invert(a2);
        Py_DECREF(a2);
    }
    else {
        newsize = Py_SIZE(a) - wordshift;
        if (newsize <= 0)
            return PyLong_FromLong(0);
        hishift = PyLong_SHIFT - remshift;
        lomask = ((digit)1 << hishift) - 1;
        himask = PyLong_MASK ^ lomask;
        z = _PyLong_New(newsize);
        if (z == NULL)
            return NULL;
        for (i = 0, j = wordshift; i < newsize; i++, j++) {
            z->ob_digit[i] = (a->ob_digit[j] >> remshift) & lomask;
            if (i+1 < newsize)
                z->ob_digit[i] |= (a->ob_digit[j+1] << hishift) & himask;
        }
        z = maybe_small_long(long_normalize(z));
    }
    return (PyObject *)z;
}
```

왼쪽 시프트와는 다르게 분기가 나누어져 있습니다.

##### 양수 오른쪽 시프트 #####
먼저 양수를 오른쪽 시프트하는 더 쉬운 경우, 즉 else 부분을 먼저 살펴봅시다.

왼쪽 시프트와 마찬가지로, 결과 오브젝트의 크기의 상한선을 대충 계산해 놓습니다. 기존 오브젝트에서 최소 `wordshift` 개의 자릿수는 없어질 것이므로 그만큼을 뺀 자릿수가 상한선이 됩니다. `remshift`의 값은 상한선 계산에서는 중요하지 않습니다.

`lomask`와 `himask`는 각각 시프트 한 이후에 기존 자리와 바로 위의 자리에서 넘어온 값이 차지하게 될 비트들을 표시해놓은 것입니다. `lomask`는 현재 자리가 시프트된 후 현재 자리에 그대로 남아있을 부분이고, `himask`는 위의 자리에서 시프트된 후 현재 자리로 넘어오는 부분입니다.

낮은 자리부터 순차적으로 돌면서 현재 자리와 위의 자리를 각각 시프트한 값을 `lomask`와 `himask`와 비트 AND 연산을 하여 현재 자리의 값을 구하는 방식으로 진행하게 됩니다. 모든 과정이 끝난 이후 역시 최상위 자리가 $0$이 될 수 있으므로 normalize해서 이 $0$을 없애주게 됩니다.

##### 음수 오른쪽 시프트 #####
음수 오른쪽 시프트는 주석에 쓰여있듯이 조금 더 복잡합니다. 그 이유는 실제로 Python이 내부적으로 정수를 저장하는 방법과는 달리, 개발자에게는 마치 이 값이 2의 보수를 사용한 형태로 저장되는 것처럼 보이게 하기 때문입니다.

이게 어떤 의미인지 몇 가지 예시를 들어보겠습니다. $1$을 이진법으로 나타내면 $...0001$이므로, 다음과 같은 결과를 얻습니다.

```
>>> (1 >> 1)
0
```

하지만 $-1$을 2의 보수법으로 나타내면 $...1111$이므로, 오른쪽으로 $1$ 시프트해도 $...1111$입니다.

```
>>> (-1 >> 1)
-1
```

이는 음수를 왼쪽으로 시프트하는 것이, 절댓값을 취하고 시프트한 뒤 부호만 뒤집으면 되는 것과는 상반됩니다. 또한 이러한 동작은 Python에서의 정수 나눗셈이 보여주는 성질과 같습니다. 즉, 오른쪽으로 $n$번 시프트하는 것은 $2**n$으로 나눈 몫을 구하는 것과 같습니다. 하지만 나눗셈을 통해 값을 구하는 것은 느리기만 하므로, 오른쪽 시프트 연산을 할 때는 직접 절댓값을 invert한 뒤 시프트를 수행하고, 이를 다시 invert하는 과정을 수행하게 됩니다.

Invert를 수행하는 함수는  `long_invert`입니다. 이 함수의 역할은 단순히 $-(x+1)$을 수행하는 것입니다. 이제 `long_rshift1`은 재귀호출을 해서 invert한 값에 대한 시프트 연산을 수행합니다. 이때에는 위에서 살펴본 else 부분의 코드가 수행되고, 이렇게 시프트된 결과를 다시 `long_invert`를 해서 결과를 얻게 됩니다.

### 비트 AND / XOR / OR ###
비트 연산자 중 AND, XOR, OR의 세 가지 연산은 유사성이 높고 공통적으로 수행할 연산들이 많기 때문에 크게 하나의 함수에서 처리하고 있습니다. 코드가 긴 편이니, 조금씩 쪼개서 살펴보겠습니다.

```c
/* Bitwise and/xor/or operations */

static PyObject *
long_bitwise(PyLongObject *a,
             char op,  /* '&', '|', '^' */
             PyLongObject *b)
{
    int nega, negb, negz;
    Py_ssize_t size_a, size_b, size_z, i;
    PyLongObject *z;
```

이 세 가지 연산을 수행하는 함수입니다. 두 개의 연산자 `a`, `b`와 수행할 연산의 종류 `op`를 '&', '|', '^' 중 하나로 받습니다.

그 다음 할 일은 `a`와 `b`의 부호에 따른 분기입니다.

```c
/* Bitwise operations for negative numbers operate as though
   on a two's complement representation.  So convert arguments
   from sign-magnitude to two's complement, and convert the
   result back to sign-magnitude at the end. */

/* If a is negative, replace it by its two's complement. */
size_a = Py_ABS(Py_SIZE(a));
nega = Py_SIZE(a) < 0;
if (nega) {
    z = _PyLong_New(size_a);
    if (z == NULL)
        return NULL;
    v_complement(z->ob_digit, a->ob_digit, size_a);
    a = z;
}
else
    /* Keep reference count consistent. */
    Py_INCREF(a);

/* Same for b. */
size_b = Py_ABS(Py_SIZE(b));
negb = Py_SIZE(b) < 0;
if (negb) {
    z = _PyLong_New(size_b);
    if (z == NULL) {
        Py_DECREF(a);
        return NULL;
    }
    v_complement(z->ob_digit, b->ob_digit, size_b);
    b = z;
}
else
    Py_INCREF(b);
```

피연산자가 음수인 경우의 비트 연산자의 동작은 그 수를 2의 보수로 표현한 상태에서 해야 하기 때문에 `v_complement` 함수를 통해 변환을 해줍니다. `a`, `b`에 각각 적용해줍니다.

그 다음은 보다 일반적이고 편리한 연산을 위해 `a`의 크기가 `b`의 크기 이상이 되도록 조정해줍니다.

```c
/* Swap a and b if necessary to ensure size_a >= size_b. */
if (size_a < size_b) {
    z = a; a = b; b = z;
    size_z = size_a; size_a = size_b; size_b = size_z;
    negz = nega; nega = negb; negb = negz;
}
```

그 다음은 시프트 연산 때 했던 것처럼 결과 오브젝트의 길이의 상한선을 대충 계산합니다.

```c
/* JRH: The original logic here was to allocate the result value (z)
   as the longer of the two operands.  However, there are some cases
   where the result is guaranteed to be shorter than that: AND of two
   positives, OR of two negatives: use the shorter number.  AND with
   mixed signs: use the positive number.  OR with mixed signs: use the
   negative number.
*/
switch (op) {
case '^':
    negz = nega ^ negb;
    size_z = size_a;
    break;
case '&':
    negz = nega & negb;
    size_z = negb ? size_a : size_b;
    break;
case '|':
    negz = nega | negb;
    size_z = negb ? size_b : size_a
    break;
default:
    Py_UNREACHABLE();
}
```

여기서도 최적화를 위한 노력을 볼 수 있는데, 원래는 두 피연산자 중 길이가 더 긴 것을 상한으로 했었으나 몇 가지 경우를 고려하여 상한선을 낮추고 있습니다. 두 양수끼리의 AND 연산이나 두 음수끼리의 OR 연산은 길이가 더 짧은 쪽이 상한이 되고, 서로 부호가 다를 때의 AND 연산은 양수인 쪽의 길이가, OR 연산은 음수인 쪽의 길이가 상한이 됩니다.

그 다음은 계산 후 혹시 2의 보수에 의해 오버플로가 발생하는 경우를 대비해 음수인 경우 한 자리를 더 확보해 둡니다.

```c
/* We allow an extra digit if z is negative, to make sure that
   the final two's complement of z doesn't overflow. */
z = _PyLong_New(size_z + negz);
if (z == NULL) {
    Py_DECREF(a);
    Py_DECREF(b);
    return NULL;
}
```

이제 사전 준비가 모두 끝났으니, 실제로 각 연산자별 비트 연산을 두 피연산자끼리 겹치는 자리들에 대해 C의 비트 연산자들을 이용하여 해줍니다.

```c
/* Compute digits for overlap of a and b. */
switch(op) {
case '&':
    for (i = 0; i < size_b; ++i)
        z->ob_digit[i] = a->ob_digit[i] & b->ob_digit[i];
    break;
case '|':
    for (i = 0; i < size_b; ++i)
        z->ob_digit[i] = a->ob_digit[i] | b->ob_digit[i];
    break;
case '^':
    for (i = 0; i < size_b; ++i)
        z->ob_digit[i] = a->ob_digit[i] ^ b->ob_digit[i];
    break;
default:
    Py_UNREACHABLE();
}
```

한쪽 피연산자의 길이가 더 긴 경우, 그 부분에 대한 나머지 처리를 합니다.

```c
/* Copy any remaining digits of a, inverting if necessary. */
if (op == '^' && negb)
    for (; i < size_z; ++i)
        z->ob_digit[i] = a->ob_digit[i] ^ PyLong_MASK;
else if (i < size_z)
    memcpy(&z->ob_digit[i], &a->ob_digit[i],
           (size_z-i)*sizeof(digit));
```

한 가지 경우만 예외로 처리하면 되는데,, XOR 연산이면서 `b`가 음수인 경우 `a`의 비트를 전부 뒤집어야 합니다. 그 외에는 `a`에서 남은 부분, 정확히는 위에서 계산했던 상한선까지의 남은 부분을 그대로 붙이면 됩니다.

```c
/* Complement result if negative. */
if (negz) {
    Py_SIZE(z) = -(Py_SIZE(z));
    z->ob_digit[size_z] = PyLong_MASK;
    v_complement(z->ob_digit, z->ob_digit, size_z+1);
}

Py_DECREF(a);
Py_DECREF(b);
return (PyObject *)maybe_small_long(long_normalize(z));
}
```

마지막으로 결과가 음수이면 다시 2의 보수 형태를 풀어주어 원래 형태로 만들고 반환하게 됩니다.

## 시리즈를 마치며 ##
지금까지 세 번의 글에 걸쳐 Python의 `int`가 어떻게 큰 정수의 표현과 연산을 지원하는지 알아보았습니다. 세 번이나 장문의 글을 썼는데도, 아직 설명하지 못한 부분이 설명한 부분보다 훨씬 많은 것 같습니다. 우리가 너무나 당연하게 생각하고 편리하게 사용해왔던 큰 정수 `int`의 여러 연산들이 실제로 이루어지는 과정을 한 단계씩 따라가보니, 그 뒤에는 이렇게 거대하고 정교한 노력이 들어간 코드가 있음을 볼 수 있었습니다.

이 복잡한 작업을 우리 대신 해서 무료로 배포해준 Python 개발자들에게 감사한 마음을 가져야 할 것 같습니다.

## 참고 자료 ##
* [CPython git repository](https://github.com/python/cpython)
