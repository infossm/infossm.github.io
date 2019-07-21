---
layout: post
title:  "Python의 큰 정수 표현법 2 - 사칙연산"
date:   2019-07-21-16:00
author: djm03178
tags: biginteger, python, 파이썬, 큰수
---
## 개요 ##
[지난 글](http://www.secmem.org/blog/2019/06/16/python-biginteger/)에 이어, 이번 글에서는 Python의 `int`가 실제로 여러 가지 연산을 수행하는 방법에 대해 파헤쳐 보겠습니다. 이전에 살펴본 것처럼 Python은 큰 정수를 표현하기 위해 크기를 변화시킬 수 있는 배열을 사용하고, 배열의 각 원소는 4바이트의 크기를 가지며 $0$ 이상 $2^{30}-1$ 이하의 값을 가집니다. $i$번째 원소를 $i$제곱하여 더한 값이 그 `int`가 나타내는 실제 값이라고 했었습니다.

당연하지만 CPU가 이런 구조에 대한 직접적인 연산을 지원하지 않기 때문에, 간단한 연산, 비교 연산이나 사칙연산조차도 CPU가 계산할 수 있는 단위까지 쪼개어 연산하는 기능을 직접 구현해줘야 합니다. CPU에 내장된 연산에 비하면 터무니없이 느린 것은 어쩔 수 없지만, 수의 범위에 걱정 없이 사용할 수 있는 편리한 `int`를 구현하려면 감수해야 할 부분입니다.

이번 글에서는 모든 정수 연산의 기본이라고 할 수 있는 사칙연산을 Python의 `int`가 수행하는 과정을 한 번 파헤쳐 보겠습니다.

## 덧셈과 뺄셈 ##
먼저 사칙연산 중에서도 가장 기본이라고 할 수 있는 덧셈과 뺄셈부터 살펴보겠습니다. 먼저, 마이너스 부호를 고려한 연산에 들어가기 전에 자연수만을 이용한 덧셈과 뺄셈부터 보겠습니다.

### 절댓값 덧셈 ###
절댓값 덧셈 코드는 다음과 같습니다.

```c
/* Add the absolute values of two integers. */

static PyLongObject *
x_add(PyLongObject *a, PyLongObject *b)
{
    Py_ssize_t size_a = Py_ABS(Py_SIZE(a)), size_b = Py_ABS(Py_SIZE(b));
    PyLongObject *z;
    Py_ssize_t i;
    digit carry = 0;

    /* Ensure a is the larger of the two: */
    if (size_a < size_b) {
        { PyLongObject *temp = a; a = b; b = temp; }
        { Py_ssize_t size_temp = size_a;
            size_a = size_b;
            size_b = size_temp; }
    }
    z = _PyLong_New(size_a+1);
    if (z == NULL)
        return NULL;
    for (i = 0; i < size_b; ++i) {
        carry += a->ob_digit[i] + b->ob_digit[i];
        z->ob_digit[i] = carry & PyLong_MASK;
        carry >>= PyLong_SHIFT;
    }
    for (; i < size_a; ++i) {
        carry += a->ob_digit[i];
        z->ob_digit[i] = carry & PyLong_MASK;
        carry >>= PyLong_SHIFT;
    }
    z->ob_digit[i] = carry;
    return long_normalize(z);
}
```

큰 정수 연산을 직접 구현해봤다면 사용해보았을 평범한 방식입니다. 초등학교 때 덧셈을 배우는 방법과도 유사합니다. 서로 같은 자리의 두 수와 carry를 더하고, 자릿수가 늘어나면 늘어난 자리의 수를 다음 carry로 넘겨서 더해가는 방식입니다. 물론, 일반적으로 사용하는 수 체계인 10진법과는 달리 Python의 `int`는 $2^{30}$진법을 사용하므로, 각 자리의 결과 역시 $2^{30}$ 미만이 되고 초과한 부분을 $2^{30}$으로 나눈 몫이 carry로 넘어가게 됩니다.

나누기 연산은 CPU 상에서 매우 느리기 때문에 보다 효율적인 방법을 사용하고 있는 것을 볼 수 있는데, $2^{30}$ 미만의 수를 저장한다는 것은 0번 비트부터 29번 비트까지의 내용을 그대로 사용할 수 있다는 점입니다. 이 0번 비트부터 29번 비트까지가 모두 켜져있는 값이 `PyLong_MASK`이며, 이와의 bitwise AND 연산을 통해 그 자리의 수를 쉽게 구할 수 있으며, `PyLong_SHIFT`만큼을 오른쪽으로 시프트해서 그 비트들을 없애는 것이 carry로 남게 됩니다.

크기는 반드시 `a`가 `b`보다 크거나 같도록 정하고 시작하기 때문에, `a`와 `b`의 같은 자리들의 덧셈이 끝난 후 `a`에 남은 자리들이 생길 수 있습니다. 그 부분에 대해서도 역시 우리가 익히 배워왔던 것과 같이 계속해서 carry를 이용하여 더해가야 합니다.

드물지만 여기까지 하고도 끝내 carry가 하나 남는 경우가 생길 수 있는데, 이를 위해 마지막에 해당 carry를 넣어놓고, long_normalize 함수를 통해 다시 남는 부분(leading zero)이 생기지 않도록 만들어주는 것을 볼 수 있습니다.

### 절댓값 뺄셈 ###
절댓값 뺄셈의 코드는 다음과 같습니다.

```c
/* Subtract the absolute values of two integers. */

static PyLongObject *
x_sub(PyLongObject *a, PyLongObject *b)
{
    Py_ssize_t size_a = Py_ABS(Py_SIZE(a)), size_b = Py_ABS(Py_SIZE(b));
    PyLongObject *z;
    Py_ssize_t i;
    int sign = 1;
    digit borrow = 0;

    /* Ensure a is the larger of the two: */
    if (size_a < size_b) {
        sign = -1;
        { PyLongObject *temp = a; a = b; b = temp; }
        { Py_ssize_t size_temp = size_a;
            size_a = size_b;
            size_b = size_temp; }
    }
    else if (size_a == size_b) {
        /* Find highest digit where a and b differ: */
        i = size_a;
        while (--i >= 0 && a->ob_digit[i] == b->ob_digit[i])
            ;
        if (i < 0)
            return (PyLongObject *)PyLong_FromLong(0);
        if (a->ob_digit[i] < b->ob_digit[i]) {
            sign = -1;
            { PyLongObject *temp = a; a = b; b = temp; }
        }
        size_a = size_b = i+1;
    }
    z = _PyLong_New(size_a);
    if (z == NULL)
        return NULL;
    for (i = 0; i < size_b; ++i) {
        /* The following assumes unsigned arithmetic
           works module 2**N for some N>PyLong_SHIFT. */
        borrow = a->ob_digit[i] - b->ob_digit[i] - borrow;
        z->ob_digit[i] = borrow & PyLong_MASK;
        borrow >>= PyLong_SHIFT;
        borrow &= 1; /* Keep only one sign bit */
    }
    for (; i < size_a; ++i) {
        borrow = a->ob_digit[i] - borrow;
        z->ob_digit[i] = borrow & PyLong_MASK;
        borrow >>= PyLong_SHIFT;
        borrow &= 1; /* Keep only one sign bit */
    }
    assert(borrow == 0);
    if (sign < 0) {
        Py_SIZE(z) = -Py_SIZE(z);
    }
    return long_normalize(z);
}
```

전반적으로는 절댓값 덧셈과 비슷합니다. 덧셈에서 자릿수가 올라가는 것을 carry를 이용해 다음 자리에 적용했다면, 뺄셈에서는 반대로 다음 자릿수에서 값을 빌려와야 할 수 있기 때문에 이를 borrow에 담아두고 다음 자릿수에서 추가로 빼게 됩니다. 덧셈과 마찬가지로 시프트 연산자를 사용하여 각 자리의 수를 빠르게 구하고 borrow도 효율적으로 바꿔나갈 수 있습니다.

여기에 뺄셈에서는 고려해야 할 점이 한 가지 더 있습니다. 비록 피연산자로는 양의 정수를 두 개 받지만, 연산 결과는 음수가 될 수 있다는 점입니다. 이를 위해 뺄셈에서는 반드시 더 큰 쪽에서 작은 쪽을 빼도록 순서를 정하고, 순서가 바뀐 경우 마지막에 부호를 마이너스로 바꾸어주는 방식을 사용합니다. 덧셈에서는 `a`의 크기, 즉 자릿수가 `b`보다 크거나 같게 만들었다면, 뺄셈에서는 `a`의 값이 `b`의 값보다 더 크게끔 자릿수가 같아도 실제로 더 큰 수가 판명날 때까지 최상위 자리부터 내려오면서 처음으로 달라지는 지점을 찾습니다. 두 수가 완전히 같다면 곧바로 0을 나타내는 오브젝트를 만들어 반환하고, 그렇지 않다면 뺄셈을 진행한 뒤 `a`, `b`의 swap 여부에 따라 음수 부호를 붙여서 반환하게 됩니다.

### 일반 덧셈과 뺄셈 ###
이제 피연산자가 양수인지, 음수인지, 0인지 모르는 상태에서의 동작에 대해 알아봅시다. 이러한 연산들은 절댓값을 이용한 연산에 부호만 적절히 맞춰주면 되기 때문에 비교적 간단합니다.

#### 덧셈 ####
```c
static PyObject *
long_add(PyLongObject *a, PyLongObject *b)
{
    PyLongObject *z;

    CHECK_BINOP(a, b);

    if (Py_ABS(Py_SIZE(a)) <= 1 && Py_ABS(Py_SIZE(b)) <= 1) {
        return PyLong_FromLong(MEDIUM_VALUE(a) + MEDIUM_VALUE(b));
    }
    if (Py_SIZE(a) < 0) {
        if (Py_SIZE(b) < 0) {
            z = x_add(a, b);
            if (z != NULL) {
                /* x_add received at least one multiple-digit int,
                   and thus z must be a multiple-digit int.
                   That also means z is not an element of
                   small_ints, so negating it in-place is safe. */
                assert(Py_REFCNT(z) == 1);
                Py_SIZE(z) = -(Py_SIZE(z));
            }
        }
        else
            z = x_sub(b, a);
    }
    else {
        if (Py_SIZE(b) < 0)
            z = x_sub(a, b);
        else
            z = x_add(a, b);
    }
    return (PyObject *)z;
}
```

일반 덧셈은 몇 가지 케이스를 나누어 처리합니다. 우선 두 수가 모두 한 자릿수인 경우, 두 수를 더한 수도 32비트 정수형 내에 들어오므로 빠르게 두 수를 덧셈으로 처리해서 새로운 오브젝트를 만들어 반환합니다.

그 외에는,
* `a`도 음수이고 `b`도 음수: 두 수의 절댓값을 더하고 음수 부호를 붙임
* `a`가 음수이고 `b`는 양수 (0 포함): `b`에서 `a`의 절댓값을 뺌
* `a`가 양수이고 `b`는 음수: `a`에서 `b`의 절댓값을 뺌
* `a`도 양수이고 `b`도 양수: 두 수를 더함

와 같이 분류하여 계산합니다.

#### 뺄셈 ####
덧셈과 거의 같은 방식으로 처리가 가능합니다.
```c
static PyObject *
long_sub(PyLongObject *a, PyLongObject *b)
{
    PyLongObject *z;

    CHECK_BINOP(a, b);

    if (Py_ABS(Py_SIZE(a)) <= 1 && Py_ABS(Py_SIZE(b)) <= 1) {
        return PyLong_FromLong(MEDIUM_VALUE(a) - MEDIUM_VALUE(b));
    }
    if (Py_SIZE(a) < 0) {
        if (Py_SIZE(b) < 0)
            z = x_sub(a, b);
        else
            z = x_add(a, b);
        if (z != NULL) {
            assert(Py_SIZE(z) == 0 || Py_REFCNT(z) == 1);
            Py_SIZE(z) = -(Py_SIZE(z));
        }
    }
    else {
        if (Py_SIZE(b) < 0)
            z = x_add(a, b);
        else
            z = x_sub(a, b);
    }
    return (PyObject *)z;
}
```

덧셈과 마찬가지로 두 수의 자릿수가 모두 1인 경우 원시 뺄셈 연산자를 사용하여 반환합니다. 그 외에는,

* `a`도 음수이고 `b`도 음수: 두 수의 절댓값끼리 뺀 값에 음수 부호를 붙임
* `a`가 음수이고 `b`는 양수 (0 포함): 두 수의 절댓값끼리 더한 값에 음수 부호를 붙임
* `a`가 양수이고 `b`는 음수: `a`에 `b`의 절댓값을 더함
* `a`도 양수이고 `b`도 양수: `a`에서 `b`의 절댓값을 뺌

과 같이 분류하여 연산하게 됩니다.

## 곱셈 ##
같은 사칙연산이라도 곱셈부터는 덧셈 / 뺄셈과는 차원이 달라집니다. 이는 연산 자체가 덧셈이나 뺄셈에 비해 심화된 개념이기 때문이기도 하지만, 평범하게 계산하는 것이 느리기 때문에 더 효율적이지만 훨씬 복잡한 알고리즘을 사용하기 때문이기도 합니다. 이 알고리즘을 여기서 깊이 있게 다루기에는 무리가 있으니, 전체적인 실행 흐름을 따라가보는 것을 주로 하겠습니다.

우선, 곱셈을 수행할 때 호출되는 함수는 다음과 같습니다.

```c
static PyObject *
long_mul(PyLongObject *a, PyLongObject *b)
{
    PyLongObject *z;

    CHECK_BINOP(a, b);

    /* fast path for single-digit multiplication */
    if (Py_ABS(Py_SIZE(a)) <= 1 && Py_ABS(Py_SIZE(b)) <= 1) {
        stwodigits v = (stwodigits)(MEDIUM_VALUE(a)) * MEDIUM_VALUE(b);
        return PyLong_FromLongLong((long long)v);
    }

    z = k_mul(a, b);
    /* Negate if exactly one of the inputs is negative. */
    if (((Py_SIZE(a) ^ Py_SIZE(b)) < 0) && z) {
        _PyLong_Negate(&z);
        if (z == NULL)
            return NULL;
    }
    return (PyObject *)z;
}
```


덧셈, 뺄셈과 마찬가지로 두 피연산자의 길이가 1이면 단순 곱셈을 통해 바로 리턴합니다. 단, 곱셈의 경우에는 덧셈 / 뺄셈과 달리 결과의 자릿수가 두 배로 늘어날 수 있기 때문에, 64비트 자료형으로 변환하여 곱셈을 수행한 뒤 다시 Python `int`형으로 변환하여 반환합니다.

그 외에는 모두 `k_mul`이라는 함수를 통해 곱셈을 계산하게 되는데, 이 함수의 형태를 간단히 살펴봅시다.

```c
/* Karatsuba multiplication.  Ignores the input signs, and returns the
 * absolute value of the product (or NULL if error).
 * See Knuth Vol. 2 Chapter 4.3.3 (Pp. 294-295).
 */
static PyLongObject *
k_mul(PyLongObject *a, PyLongObject *b)
```

주석에 설명된 것과 같이 이 함수는 기본적으로 [카라츠바 알고리즘](https://ko.wikipedia.org/wiki/%EC%B9%B4%EB%9D%BC%EC%B6%94%EB%B0%94_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)을 사용합니다. 이후에 이 함수와 관련된 코드만 약 300줄에 달하는데, 굳이 이렇게 복잡한 방법을 통해 연산을 해야 하는 이유는 우리가 학교에서 배우는 나이브한 곱셈은 두 수의 자릿수를 $N$이라고 할 때 $O(N^2)$ 시간이 소요되어, 곱하려는 두 수가 큰 경우에는 매우 긴 시간을 요구할 수 있기 때문입니다. 반면에 카라츠바 알고리즘은 $O(N^{1.585})$ 정도만을 요구하기 때문에, 자릿수가 많더라도 비교적 무난한 시간 내에 곱셈을 수행할 수 있습니다. 다만 시간 복잡도라는 개념이 원래 그렇듯이 모든 경우에 카라츠바 알고리즘이 더 빠른 것은 아니며, 자릿수가 상당히 많아져야 비로소 의미가 생기게 됩니다. 그래서 자릿수가 적은 수일 때(`KARATSUBA_CUTOFF`, `KARATSUBA_SQUARE_CUTOFF`)는 아래와 같이 `x_mul`이라는 함수를 사용하여 평범한 $O(N^2)$의 곱셈을 수행합니다.

```c
/* For int multiplication, use the O(N**2) school algorithm unless
 * both operands contain more than KARATSUBA_CUTOFF digits (this
 * being an internal Python int digit, in base BASE).
 */
#define KARATSUBA_CUTOFF 70
#define KARATSUBA_SQUARE_CUTOFF (2 * KARATSUBA_CUTOFF)

...

/* Use gradeschool math when either number is too small. */
    i = a == b ? KARATSUBA_SQUARE_CUTOFF : KARATSUBA_CUTOFF;
    if (asize <= i) {
        if (asize == 0)
            return (PyLongObject *)PyLong_FromLong(0);
        else
            return x_mul(a, b);
    }

...

/* Grade school multiplication, ignoring the signs.
 * Returns the absolute value of the product, or NULL if error.
 */
static PyLongObject *
x_mul(PyLongObject *a, PyLongObject *b)
```

## 나눗셈 ##
한 정수를 다른 정수로 나누는 것은 일반적으로 정수가 나오지 않을 수도 있지만, Python에서는 `//` 연산자와 `%` 연산자를 통해 나누기의 결과를 정수형의 몫과 나머지로 구할 수 있게 해줍니다. 또한 이 둘을 동시에 수행하여 `tuple`로 반환하는 `divmod`라는 함수도 존재합니다. 여기서는 이 몫과 나머지를 구하는 나눗셈에 대해서만 다룹니다.

나눗셈도 무척이나 복잡하고 효율적인 연산을 요구하기 때문에 코드 전체를 이 글에서 전부 설명하는 것은 무리이므로, 큰 실행 흐름을 따라가 보겠습니다.

우선 다른 연산들과 마찬가지로, 매우 작은 범위에서 단순하게 처리하는 코드부터 살펴보겠습니다.

```c
/* Fast modulo division for single-digit longs. */
static PyObject *
fast_mod(PyLongObject *a, PyLongObject *b)
{
    sdigit left = a->ob_digit[0];
    sdigit right = b->ob_digit[0];
    sdigit mod;

    assert(Py_ABS(Py_SIZE(a)) == 1);
    assert(Py_ABS(Py_SIZE(b)) == 1);

    if (Py_SIZE(a) == Py_SIZE(b)) {
        /* 'a' and 'b' have the same sign. */
        mod = left % right;
    }
    else {
        /* Either 'a' or 'b' is negative. */
        mod = right - 1 - (left - 1) % right;
    }

    return PyLong_FromLong(mod * (sdigit)Py_SIZE(b));
}

/* Fast floor division for single-digit longs. */
static PyObject *
fast_floor_div(PyLongObject *a, PyLongObject *b)
{
    sdigit left = a->ob_digit[0];
    sdigit right = b->ob_digit[0];
    sdigit div;

    assert(Py_ABS(Py_SIZE(a)) == 1);
    assert(Py_ABS(Py_SIZE(b)) == 1);

    if (Py_SIZE(a) == Py_SIZE(b)) {
        /* 'a' and 'b' have the same sign. */
        div = left / right;
    }
    else {
        /* Either 'a' or 'b' is negative. */
        div = -1 - (left - 1) / right;
    }

    return PyLong_FromLong(div);
}
```

두 수가 모두 한 자릿수일 때의 함수들로, 최대한 원시 연산자의 도움을 받습니다. 그런데 여기서 의문이 들 수 있는 것은, 단순하게 `mod = left % right`, `div = left / right`로 쓰지 않고 두 피연산자의 부호가 같지 않은 경우에는 별도의 식을 통해 처리를 하고 있다는 점입니다. 그 이유는 Python의 나눗셈 / 나머지 연산의 요구 사항이 C의 나눗셈 / 나머지 요구 사항과 100% 일치하지 않기 때문입니다. 예를 들어, C99에서의 나눗셈 표준은 몫의 결과를 0에 가까운 쪽으로 올림 / 내림하도록 하고 있어, `-3 / 2`를 -1로 연산하도록 합니다. 반면에 Python의 나눗셈은 나눈 결과를 내림한 것을 결과로 하도록 규정하고 있어, `-3 // 2`의 결과는 -2가 됩니다. 이러한 차이점이 존재하기 때문에 Python의 나눗셈 구현체는 한 자릿수의 연산 결과라도 C의 원시 나눗셈을 그대로 사용하지 못하고 이와 같이 분기를 나누어 처리를 해주어야 합니다.

보다 일반적으로 나누기와 나머지 연산을 수행하는 함수는 `long_div`와 `long_mod`, 그리고 `long_divmod` 함수로, 이 셋은 피연산자들의 크기에 따라 위의 빠른 함수들이 쓰이거나, `long_divrem` 함수를 통해 더 큰 수의 나눗셈을 위한 준비를 하게 됩니다.

```c
/* Int division with remainder, top-level routine */

static int
long_divrem(PyLongObject *a, PyLongObject *b,
            PyLongObject **pdiv, PyLongObject **prem)
{
    Py_ssize_t size_a = Py_ABS(Py_SIZE(a)), size_b = Py_ABS(Py_SIZE(b));
    PyLongObject *z;

    if (size_b == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                        "integer division or modulo by zero");
        return -1;
    }
    if (size_a < size_b ||
        (size_a == size_b &&
         a->ob_digit[size_a-1] < b->ob_digit[size_b-1])) {
        /* |a| < |b|. */
        *prem = (PyLongObject *)long_long((PyObject *)a);
        if (*prem == NULL) {
            return -1;
        }
        Py_INCREF(_PyLong_Zero);
        *pdiv = (PyLongObject*)_PyLong_Zero;
        return 0;
    }
    if (size_b == 1) {
        digit rem = 0;
        z = divrem1(a, b->ob_digit[0], &rem);
        if (z == NULL)
            return -1;
        *prem = (PyLongObject *) PyLong_FromLong((long)rem);
        if (*prem == NULL) {
            Py_DECREF(z);
            return -1;
        }
    }
    else {
        z = x_divrem(a, b, prem);
        if (z == NULL)
            return -1;
    }
    /* Set the signs.
       The quotient z has the sign of a*b;
       the remainder r has the sign of a,
       so a = b*z + r. */
    if ((Py_SIZE(a) < 0) != (Py_SIZE(b) < 0)) {
        _PyLong_Negate(&z);
        if (z == NULL) {
            Py_CLEAR(*prem);
            return -1;
        }
    }
    if (Py_SIZE(a) < 0 && Py_SIZE(*prem) != 0) {
        _PyLong_Negate(prem);
        if (*prem == NULL) {
            Py_DECREF(z);
            Py_CLEAR(*prem);
            return -1;
        }
    }
    *pdiv = maybe_small_long(z);
    return 0;
}
```

자세히 보면, 이 함수 역시 제대로 된 나눗셈을 수행하는 함수가 아니라, 피연산자의 특성에 따라 분기를 나누어 실제 나눗셈을 어떤 방법으로 수행할지를 고르는 것을 볼 수 있습니다. 이 글에서는 각 분기가 어떤 함수를 호출하고, 그 함수의 역할이 어떤 것인지를 간단히 설명하겠습니다.

먼저 나누는 수가 0인 경우는 당연히 에러가 나야 하므로, C 코드에서 런타임 에러를 띄우지 않고 Python의 선에서 끊어 `ZeroDivisionError`를 발생시키도록 미리 예외 처리를 해둡니다.

그 다음 있는 분기는 나누는 수가 나누어지는 수보다 큰 경우를 미리 거르는 것으로, 이 경우 답은 당연히 0이기 때문에 곧바로 0 오브젝트를 반환해줍니다.

마지막으로 `b`의 크기가 1인 경우와 그렇지 않은 경우를 나누는데, 먼저 1인 경우에 실행되는 `divrem1` 함수는 내부적으로 다시 `inplace_divrem1` 함수를 호출해서 나누는 수가 한 자리인 경우를 빠르게 계산합니다. `inplace_divrem1` 함수는 다음과 같습니다.

```c
/* Divide long pin, w/ size digits, by non-zero digit n, storing quotient
   in pout, and returning the remainder.  pin and pout point at the LSD.
   It's OK for pin == pout on entry, which saves oodles of mallocs/frees in
   _PyLong_Format, but that should be done with great care since ints are
   immutable. */

static digit
inplace_divrem1(digit *pout, digit *pin, Py_ssize_t size, digit n)
{
    twodigits rem = 0;

    assert(n > 0 && n <= PyLong_MASK);
    pin += size;
    pout += size;
    while (--size >= 0) {
        digit hi;
        rem = (rem << PyLong_SHIFT) | *--pin;
        *--pout = hi = (digit)(rem / n);
        rem -= (twodigits)hi * n;
    }
    return (digit)rem;
}
```

이 함수의 동작을 간단히 설명하면, 높은 자리부터 수를 나누어 그 자리의 몫을 구하고, 남는 나머지는 $2^{30}$을 곱해 낮은 자리에 더한 뒤 다시 나누어 그 자리의 몫을 구하는 것을 반복하는 식입니다. 나누는 수가 한 자리의 수이기 때문에, 각 자리에 대해 원시 나눗셈 연산을 사용하여 빠르게 구할 수 있기 때문에 이와 같은 방법을 사용합니다.

나누는 수가 한 자리가 아닌 경우에는, 어쩔 수 없지만 복잡한 방법을 통해 느리게 나누기 위해 `x_divrem` 함수를 호출합니다. 이 함수에서의 나누기는 역시 Knuth의 The Art of Computer Programming을 참고했다고 합니다.

```c
/* Unsigned int division with remainder -- the algorithm.  The arguments v1
   and w1 should satisfy 2 <= Py_ABS(Py_SIZE(w1)) <= Py_ABS(Py_SIZE(v1)). */

static PyLongObject *
x_divrem(PyLongObject *v1, PyLongObject *w1, PyLongObject **prem)

...

/* We follow Knuth [The Art of Computer Programming, Vol. 2 (3rd
       edn.), section 4.3.1, Algorithm D], except that we don't explicitly
       handle the special case when the initial estimate q for a quotient
       digit is >= PyLong_BASE: the max value for q is PyLong_BASE+1, and
       that won't overflow a digit. */
```

## 2편을 마치며 ##
이번 글에서는 정수 연산의 가장 핵심이라고 할 수 있는 사칙연산에 대해서 살펴보았습니다. Python을 사용하면서 편하게 아무 생각 없이 `+`, `-`, `*`, `//`, `%` 등으로 써왔던 연산자가 실제로 하는 일이 이렇게나 복잡하고 많다는 것을 보면, 이런 어려운 일을 대신 미리 해주어 프로그래머들이 힘들게 큰 수의 연산자를 구현하지 않아도 되게 해준 Python 개발자들에게 감사한 마음을 가져야 할 것 같습니다.

심지어, 우리는 아직 Python이 제공하는 모든 연산자를 다 본 것도 아닙니다. 다음 3편에서는 아직까지 다루지 않은 기타 연산자들에 대해서 다루어보도록 하겠습니다.

## 참고 자료 ##
* [CPython git repository](https://github.com/python/cpython/tree/master/Python)
