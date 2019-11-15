---
layout: post
title:  "Python의 큰 정수 표현법 1"
date:   2019-06-16 20:44:00
author: djm03178
tags: biginteger, python, 파이썬, 큰수
---
## 개요 ##
대부분의 프로그래밍 언어, 특히 거의 모든 저레벨 언어에는 정수형 크기에 제한이 있습니다. 대체로 바이트 단위로 끊어서 1바이트, 2바이트, 4바이트, 8바이트 정도의 정수형들을 사용할 수 있고, 언어와 컴파일러에 따라서는 16바이트 정수형이 제공되기도 합니다.

이와 같은 제한은 현대 프로세서들의 연산 능력을 고려하여 디자인된 것이라고 할 수 있습니다. 최근까지도 32비트/64비트 프로세서들이 주류를 이루고 있고, 이는 곧 프로세서가 연산을 수행하는 레지스터의 크기가 4/8바이트 정도이며 한 번의 연산 단위가 될 수 있다는 뜻이 됩니다. 따라서 이 크기에 맞추어 자료형을 설정하는 것은 프로그램의 성능을 극대화하는 데에 중요한 요소입니다.

그러나 때때로 실무에서는 이보다 훨씬 더 넓은 범위의 천문학적인 수를 표현해야 하기도 합니다. 큰 정수의 표현과 연산은 대부분의 언어의 컴파일러와 프로세서가 직접적으로 제공하지 않기 때문에, 여러 세부적인 연산으로 나누어 계산하고 여러 컨테이너에 나누어 담아 표현하기 위해 자료구조를 만들고 연산자들을 정의해줘야 합니다. 대표적인 것으로 Java의 `BigInteger` 클래스가 있는데, 수십 가지의 연산을 위한 메서드를 제공하며 구현 코드의 길이도 4천줄을 훌쩍 뛰어넘습니다. 자료구조를 직접 만들고, 모든 연산의 구체적인 방법을 전부 직접 명시해주어야 하기 때문입니다.

이 글에서 살펴볼 것은 그 라이브러리 중에서도 가장 쉽게 사용할 수 있고 대중적으로 굉장히 많은 사람들이 이미 사용 중인, Python에서의 큰 정수형 구현 방법입니다. 사실 Python에는 큰 정수라는 개념이 없습니다. 기본적인 정수형인 `int`가 바로 큰 정수형이기 때문입니다. 성능보다는 편리함과 명료성에 초점을 둔 Python이기 때문에, 기본적인 정수형 연산에서부터 오버플로우와 같은 문제에 대한 걱정 없이 원하는 정수를 마음껏 표현할 수 있도록 이 기능을 아예 기본 정수형으로서 내장시켜놓은 것입니다. 1편에서는 Python의 가장 표준이 되는 CPython 구현체를 바탕으로 `int`형의 기본적인 구조와 기초적인 연산 몇 가지에 대해 알아보겠습니다.

## \_longobject 구조체 ##
CPython의 git repository는 https://github.com/python/cpython 이며, 이 중 int의 구현을 담당하는 소스 코드는 크게 [세](https://github.com/python/cpython/blob/master/Include/longintrepr.h) [개](https://github.com/python/cpython/blob/master/Include/longobject.h)[로](https://github.com/python/cpython/blob/master/Objects/longobject.c) 이루어져 있습니다. 이 중 하나의 `int`형 오브젝트를 나타내는 자료형은 longintrepr.h에 있는  `struct _longobject`입니다.

```c
/* Long integer representation.
   The absolute value of a number is equal to
        SUM(for i=0 through abs(ob_size)-1) ob_digit[i] * 2**(SHIFT*i)
   Negative numbers are represented with ob_size < 0;
   zero is represented by ob_size == 0.
   In a normalized number, ob_digit[abs(ob_size)-1] (the most significant
   digit) is never zero.  Also, in all cases, for all valid i,
        0 <= ob_digit[i] <= MASK.
   The allocation function takes care of allocating extra memory
   so that ob_digit[0] ... ob_digit[abs(ob_size)-1] are actually available.
   CAUTION:  Generic code manipulating subtypes of PyVarObject has to
   aware that ints abuse  ob_size's sign bit.
*/

struct _longobject {
    PyObject_VAR_HEAD
    digit ob_digit[1];
};
```
기본적인 설명은 친절하게 주석에 적혀있습니다. 어떤 의미인지 분석해 봅시다.

### 구조 ###
`PyObject_VAR_HEAD`는 오브젝트들의 첫머리에 공통적으로 들어갈 것들을 모아놓은 매크로로, garbage collection을 위한 reference count 변수와 `int` 타입 자체에 대한 정의를 가진 구조체에 대한 포인터 등을 담고 있습니다. 또한 `ob_size`라는 변수도 하나 생성되는데, 이는 이 객체가 가진 원소의 수, 여기서는 `ob_digit`의 실제 크기를 나타냅니다. `ob_digit`이 실제로 이 객체가 나타내는 정수의 값을 담게 되며, `digit`은 설정에 따라 최종적으로 `unsigned short` 또는 `uint32_t`로 변환됩니다. 여기서는 `uint32_t`, 즉 32비트 부호 없는 정수형으로 생각하겠습니다.

#### ob_digit ####
구조체의 생김새로만 봐서는 `ob_digit`의 크기가 1로 고정되어 유동적으로 크기를 변환할 수 없는 것처럼 보입니다. 하지만 Python 구현체에서는 내부적으로 메모리 관리를 담당하는 부분이 있어, 하나의 `_longobject`가 실제로 사용할 수 있는 메모리가 `sizeof(_longobject)`보다 커질 수 있게, `ob_digit[abs(ob_size)-1]`까지를 사용할 수 있도록 만들어 줍니다.

#### ob_size ####
`ob_size`는 일반적으로 객체의 크기를 나타내지만, `int`에 있어서는 한 가지 활용 용도가 더 있습니다. 바로 이 객체가 나타내는 정수형의 부호를 결정하는 역할을 겸하게 됩니다. 예를 들어 `ob_digit`의 다섯 자리를 사용하는 크기이고 양수라면 `ob_size`는 5가 되고, 음수라면 -5가 되며, 0인 경우에는 특별히 이 값을 0으로 두어 객체의 값을 나타냅니다.

### 값의 표현 ###
객체의 실제 값은 `ob_digit` 배열에 저장되며, $0$ 이상 `abs(ob_size)-1` 이하의 모든 $i$에 대해 $i$번째 원소의 값에 $2^{30i}$를 곱한 값들을 전부 더하고 `ob_size`의 부호를 곱한 것이 실제 값이 됩니다. 각각의 원소는 $0$ 이상 $2^{30i}-1$ 이하의 값을 갖습니다.

예를 들어 정수값 9982443531000000007은 다음과 같이 저장됩니다.

![9982443531000000007](/assets/images/python-biginteger/1.png)

## 형변환 ##
정수를 나타내는 구조체를 만들었으니, 가장 먼저 기본적으로 할 것은 우선 기본 자료형으로 표현된 값을 `_longobject` 타입으로 변환하는 것과 그 반대를 수행하는 인터페이스를 만드는 것입니다.

### \_longobject 타입으로의 형변환 ###
longobject.c에서는 다양한 C 기본 자료형의 값을 `_longobject` 타입으로 바꾸는 인터페이스들이 구현되어 있습니다. longobject.h에 그 원형이 선언되어 있고, 다음은 그 중 일부입니다.
```c
PyAPI_FUNC(PyObject *) PyLong_FromLong(long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLong(unsigned long);
PyAPI_FUNC(PyObject *) PyLong_FromSize_t(size_t);
PyAPI_FUNC(PyObject *) PyLong_FromSsize_t(Py_ssize_t);
PyAPI_FUNC(PyObject *) PyLong_FromDouble(double);
PyAPI_FUNC(long) PyLong_AsLong(PyObject *);
PyAPI_FUNC(long) PyLong_AsLongAndOverflow(PyObject *, int *);
```
이 중 대표적으로 `long`형이 `_longobject`형으로 변환되는 과정을 간단히 살펴보겠습니다. 먼저 절댓값이 $2^{30}$ 미만인 수들에 대해서는 Fast path를 사용합니다. 즉, 어차피 `ob_digit[0]` 내에 다 들어갈 수 있는 값이므로 일반적인 변환 과정을 거치지 않고 곧바로 한 자리의 오브젝트를 생성해서 바로 반환해줍니다.
```c
/* Fast path for single-digit ints */
if (!(abs_ival >> PyLong_SHIFT)) {
    v = _PyLong_New(1);
    if (v) {
        Py_SIZE(v) = sign;
        v->ob_digit[0] = Py_SAFE_DOWNCAST(
            abs_ival, unsigned long, digit);
    }
    return (PyObject*)v;
}
```
이보다 큰 값들에 대해서는 `ob_size`가 얼마가 되어야 하는지를 $2^{30}$ 씩 나누어가며 구하고, 그 크기의 오브젝트를 생성해서 값을 차례대로 담게 됩니다.

```c
/* Larger numbers: loop to determine number of digits */
t = abs_ival;
while (t) {
    ++ndigits;
    t >>= PyLong_SHIFT;
}
v = _PyLong_New(ndigits);
if (v != NULL) {
    digit *p = v->ob_digit;
    Py_SIZE(v) = ndigits*sign;
    t = abs_ival;
    while (t) {
        *p++ = Py_SAFE_DOWNCAST(
            t & PyLong_MASK, unsigned long, digit);
        t >>= PyLong_SHIFT;
    }
}
return (PyObject *)v;
```

### 기본 타입으로의 형변환 ###
기본 타입으로 `_longobject`로 변환하는 인터페이스가 있으니, 그 반대도 있어야 할 것입니다. 대표적으로 다음과 같은 것들이 있습니다.

```c
PyAPI_FUNC(PyObject *) PyLong_FromLongLong(long long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLongLong(unsigned long long);
PyAPI_FUNC(long long) PyLong_AsLongLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLongMask(PyObject *);
PyAPI_FUNC(long long) PyLong_AsLongLongAndOverflow(PyObject *, int *);
```
이와 같은 함수들은 `ob_size`의 절댓값이 1 이하이면 곧바로 값을 얻어와서 반환, 더 크면 `__PyLong_AsByteArray`라는 함수를 통해, 바이트 단위로 값을 쪼개어 넣습니다. `ob_digit` 자체가 30비트 단위로 값을 끊어서 저장하므로, 8비트씩 끊어내도록 구현이 되어있습니다.
```c
int
_PyLong_AsByteArray(PyLongObject* v,
                    unsigned char* bytes, size_t n,
                    int little_endian, int is_signed)

...

/* Copy over all the Python digits.
   It's crucial that every Python digit except for the MSD contribute
   exactly PyLong_SHIFT bits to the total, so first assert that the int is
   normalized. */
assert(ndigits == 0 || v->ob_digit[ndigits - 1] != 0);
j = 0;
accum = 0;
accumbits = 0;
carry = do_twos_comp ? 1 : 0;
for (i = 0; i < ndigits; ++i) {
    digit thisdigit = v->ob_digit[i];
    if (do_twos_comp) {
        thisdigit = (thisdigit ^ PyLong_MASK) + carry;
        carry = thisdigit >> PyLong_SHIFT;
        thisdigit &= PyLong_MASK;
    }
    /* Because we're going LSB to MSB, thisdigit is more
       significant than what's already in accum, so needs to be
       prepended to accum. */
    accum |= (twodigits)thisdigit << accumbits;

    /* The most-significant digit may be (probably is) at least
       partly empty. */
    if (i == ndigits - 1) {
        /* Count # of sign bits -- they needn't be stored,
         * although for signed conversion we need later to
         * make sure at least one sign bit gets stored. */
        digit s = do_twos_comp ? thisdigit ^ PyLong_MASK : thisdigit;
        while (s != 0) {
            s >>= 1;
            accumbits++;
        }
    }
    else
        accumbits += PyLong_SHIFT;

    /* Store as many bytes as possible. */
    while (accumbits >= 8) {
        if (j >= n)
            goto Overflow;
        ++j;
        *p = (unsigned char)(accum & 0xff);
        p += pincr;
        accumbits -= 8;
        accum >>= 8;
    }
}
```
### 문자열로의 변환 ###
저장된 값을 실제로 우리가 원하는 형태, 즉, 10진수 형태로 출력하기 위해서는 복잡한 변환 과정을 거쳐야 합니다. 구체적인 과정이나 코드는 너무 길고 어렵기 때문에 간략하게만 설명하자면, 우선 출력할 길이의 상한선을 대략적으로 측정하여 배열을 할당받고, 그 배열에 Knuth의 The Art of Computer Programming에 명시된 방법을 통해 2진수의 값을 10진수의 값으로 변환하여 채워넣습니다. 그 후 배열의 각 원소를 개별적으로 나누기 / 나머지 연산을 통해 문자로 차례대로 변환한 뒤, 이를 역순으로 출력하는 과정을 거치게 됩니다.

이를 수행하는 함수가 `long_to_decimal_string_internal`입니다.

```c
/* Convert an integer to a base 10 string.  Returns a new non-shared
   string.  (Return value is non-shared so that callers can modify the
   returned value if necessary.) */

static int
long_to_decimal_string_internal(PyObject *aa,
                                PyObject **p_output,
                                _PyUnicodeWriter *writer,
                                _PyBytesWriter *bytes_writer,
                                char **bytes_str)
```

## 비교 ##
어떤 정수가 더 큰 수인지를 판별하는 것은 어렵지 않습니다. 코드도 비교적 간단합니다.
```c
static int
long_compare(PyLongObject *a, PyLongObject *b)
{
    Py_ssize_t sign;

    if (Py_SIZE(a) != Py_SIZE(b)) {
        sign = Py_SIZE(a) - Py_SIZE(b);
    }
    else {
        Py_ssize_t i = Py_ABS(Py_SIZE(a));
        while (--i >= 0 && a->ob_digit[i] == b->ob_digit[i])
            ;
        if (i < 0)
            sign = 0;
        else {
            sign = (sdigit)a->ob_digit[i] - (sdigit)b->ob_digit[i];
            if (Py_SIZE(a) < 0)
                sign = -sign;
        }
    }
    return sign < 0 ? -1 : sign > 0 ? 1 : 0;
}
```
우선 두 객체의 부호가 다르면 한 쪽이 더 큰 것은 명확합니다. 이를 `ob_size`의 값을 비교하여 곧바로 알아낼 수 있습니다. 부호가 같더라도 `ob_size`의 값이 서로 다르다면 그 값이 더 큰 쪽이 더 큰 수일 것도 명확합니다.

만일 부호가 같다면, 가장 높은 자리부터 비교하면서 처음으로 값이 달라지는 digit까지 내려와야 합니다. 최하위 자리까지 비교했는데도 같다면 둘은 완전히 같은 값인 것이고, 그렇지 않다면 처음으로 달라진 지점에서 부호를 고려하여 더 큰 값을 가지는 쪽이 더 큰 값이 될 것입니다.

## 자주 사용하는 값 ##
Java의 `Integer` 클래스와 비슷하게, Python의 `int`도 빈번하게 사용되는 일부 정수들에 대한 객체를 미리 생성해둡니다. 그 범위는 longobject.c에 $-5$부터 $256$까지로 정의되어 있습니다.
```c
#ifndef NSMALLPOSINTS
#define NSMALLPOSINTS           257
#endif
#ifndef NSMALLNEGINTS
#define NSMALLNEGINTS           5
#endif

...

/* Small integers are preallocated in this array so that they
   can be shared.
   The integers that are preallocated are those in the range
   -NSMALLNEGINTS (inclusive) to NSMALLPOSINTS (not inclusive).
*/
static PyLongObject small_ints[NSMALLNEGINTS + NSMALLPOSINTS];
```
Python의 객체를 하나 생성한다는 것은 단순히 프로세서가 메모리에 정수값 하나를 위한 공간을 할당받고 쓰는 것과는 비교할 수 없이 느린 연산이기 때문에, 이와 같이 자주 사용하는 값들에 대한 객체를 미리 생성해두는 것은 추후 이 값들이 반복적으로 사용될 때 상당한 성능 향상을 기대할 수 있습니다.

## 1편을 마치며 ##
지금까지 Python에서 큰 정수형을 표현하기 위한 기본적인 구조를 살펴보았습니다. 그런데, 생각해 보면 아직 설명하지 않은 것들이 많습니다. 당장 사칙연산조차 아직 언급하지 않았습니다. 단지 Python에서 이 값들이 어떤 방식으로 저장되고, 어떻게 기초 자료형과 양방향으로 변환하고, 그 모습을 눈으로 볼 수 있게 만드는지만을 살펴보았을 뿐입니다. 그럼에도 불구하고, 이런 기초적인 내용만으로도 큰 정수를 표현한다는 것이 얼마나 복잡하고 어려운 일인지 충분히 느껴졌을 것입니다.

다음 2편에서는 이 객체들이 실질적인 연산의 대상으로 사용되기 위한 연산들에 대해 알아볼 것입니다. 그 내용을 들어가기 전에, 1편에서 설명한 `_longobject`의 기초적인 구조가 확실하게 이해되어야 한다고 볼 수 있을 것입니다.

## 참고 자료 ##
* https://github.com/python/cpython/tree/master/Python
* https://rushter.com/blog/python-integer-implementation/
