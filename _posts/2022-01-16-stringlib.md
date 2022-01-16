---
layout: post
title:  "Python 3의 String 라이브러리 둘러보기"
date:   2022-01-16 17:26:34
author: djm03178
tags: python, string
---

## 개요 ##
[이전에 Python 3에서 문자열 내의 부분 문자열을 빠르게 찾기 위한 fastsearch 기법에 대한 글을 쓴 적이 있습니다.](http://www.secmem.org/blog/2021/08/18/fastsearch/) 그 글에서는 문자열 중에서도 특정 기능에 대한 심도 있는 분석을 해보았었는데, Python의 철학이 담긴 대단히 세밀한 디자인과 구현체를 볼 수 있었습니다. 하지만 그것만으로는 Python의 `str`이 가진 수많은 특징 중 극히 일부 단면만을 보인 듯한 느낌이 들었습니다. 그래서, 이번에는 좀더 `str`의 전체적인 모습을 두루 살펴볼 수 있도록 보다 큰 그림을 통해 Python 3의 `str`의 구조는 어떻게 되며, string 라이브러리에는 어떤 기능이 있는지 알아보는 글을 써보려고 합니다.

문자열 라이브러리는 https://github.com/python/cpython/tree/main/Objects/stringlib 에서 찾아볼 수 있습니다.

## `str`의 구조 ##
Python에서 `str`는 `int`, `float`, `bool`, `dict`, `set`, `list`, `tuple` 등과 더불어 가장 대표적인 내장 클래스 중 하나입니다. 별개의 라이브러리가 아닌, C를 사용하여 순수 내장으로 구현된 만큼 Python 인터프리터 위에서 최대한 효율적으로 동작할 수 있게끔 강력히 최적화된 구조를 가지고 있습니다. Python에서 문자열은 크게 1바이트 단위로 문자를 저장하는 bytesobject 형식과 유니코드 문자열을 저장하는 unicodeobject 형식으로 나뉘어 있는데, 문자열을 저장하는 방식에서만 차이가 있고 문자열 라이브러리는 같은 것을 공유하여 사용합니다. Python에서 기본으로 사용하는 문자열 형식은 유니코드 형식입니다.

Bytesobject 형식은 https://github.com/python/cpython/blob/main/Include/cpython/bytesobject.h 에 정의되어 있으며, 다음과 같습니다.

```c
typedef struct {
    PyObject_VAR_HEAD
    Py_hash_t ob_shash;
    char ob_sval[1];

    /* Invariants:
     *     ob_sval contains space for 'ob_size+1' elements.
     *     ob_sval[ob_size] == 0.
     *     ob_shash is the hash of the byte string or -1 if not computed yet.
     */
} PyBytesObject;
```

구조는 단순합니다. Python의 모든 오브젝트가 공통적으로 가지는 헤드 부분과, 오브젝트의 해시값, 그리고 `char` 형식의 1칸짜리 배열이 있습니다. 잠깐, 분명 여러 바이트를 저장하기 위한 건데 왜 1칸일까요? 이는 이전에 [Python의 큰 정수 표현법 1](https://www.secmem.org/blog/2019/06/16/python-biginteger/)에서 설명한 바가 있는데, 코드상으로는 1바이트만을 할당하는 것처럼 보이지만 실제로는 Python 인터프리터 내부의 별도의 메모리 관리 기법을 통해 항상 동적으로 메모리를 할당하고 이 구조체는 해당 위치를 가리키는 포인터로서만 동작하기 때문에 필요에 의해 얼마든지 이 배열이 실제로 사용할 수 있는 메모리를 늘릴 수 있습니다.

Unicodeobject 형식은 https://github.com/python/cpython/blob/main/Include/cpython/unicodeobject.h 에 있는데, 아까보다는 조금 복잡하며 더 세분화되어 있습니다.

```c
/* --- Unicode Type ------------------------------------------------------- */

/* ASCII-only strings created through PyUnicode_New use the PyASCIIObject
   structure. state.ascii and state.compact are set, and the data
   immediately follow the structure. utf8_length and wstr_length can be found
   in the length field; the utf8 pointer is equal to the data pointer. */
typedef struct {
    PyObject_HEAD
    Py_ssize_t length;          /* Number of code points in the string */
    Py_hash_t hash;             /* Hash value; -1 if not set */
    struct {
        unsigned int interned:2;
        unsigned int kind:3;
        unsigned int compact:1;
        unsigned int ascii:1;
        unsigned int ready:1;
        unsigned int :24;
    } state;
    wchar_t *wstr;              /* wchar_t representation (null-terminated) */
} PyASCIIObject;

/* Non-ASCII strings allocated through PyUnicode_New use the
   PyCompactUnicodeObject structure. state.compact is set, and the data
   immediately follow the structure. */
typedef struct {
    PyASCIIObject _base;
    Py_ssize_t utf8_length;     /* Number of bytes in utf8, excluding the
                                 * terminating \0. */
    char *utf8;                 /* UTF-8 representation (null-terminated) */
    Py_ssize_t wstr_length;     /* Number of code points in wstr, possible
                                 * surrogates count as two code points. */
} PyCompactUnicodeObject;

/* Strings allocated through PyUnicode_FromUnicode(NULL, len) use the
   PyUnicodeObject structure. The actual string data is initially in the wstr
   block, and copied into the data block using _PyUnicode_Ready. */
typedef struct {
    PyCompactUnicodeObject _base;
    union {
        void *any;
        Py_UCS1 *latin1;
        Py_UCS2 *ucs2;
        Py_UCS4 *ucs4;
    } data;                     /* Canonical, smallest-form Unicode buffer */
} PyUnicodeObject;
```

이 구조를 이해하기 위해서는 Python의 유니코드 문자열의 종류를 먼저 알아야 합니다. Python의 유니코드 문자열은 크게 4가지로 구분되며 다음과 같습니다.

* Compact ascii (`PyASCIIObject`)
* Compact (`PyCompactUnicodeObject`)
* Legacy string, not ready (`PyUnicodeObject`)
* Legacy string, ready (`PyUnicodeObject structure`)

Compact라는 것은 문자열의 내용이 구조체와 같은 블록 내에 있는 것을 의미합니다. 반대로 compact가 아닌 것은 문자열의 내용이 문자열 객체와는 다른 블록에 저장되어 있습니다. Legacy string은 `PyUnicode_FromUnicode()` 또는 `PyUnicode_FromStringAndSize(NULL, size)` 함수의 호출에 의해 만들어진 문자열을 의미하며, not ready로 있다가 `PyUnicode_READY()`가 호출되면 비로소 ready 상태가 됩니다.

## String 라이브러리가 호출되는 메커니즘 ##
위에서 string 라이브러리들은 문자열 종류에 관계 없이 공유된다고 했습니다. 그런데 이는 완전히 똑같은 함수가 모든 종류의 문자열에 대해 사용된다는 것은 아닙니다. 정확히는 각 문자열의 종류에 따라 그 종류의 문자열을 위한 라이브러리가 별개로 만들어집니다. 이에 대해 간략하게 살펴보겠습니다.

먼저, 바이트 문자열을 위한 파일로는 https://github.com/python/cpython/blob/main/Objects/stringlib/stringdefs.h 가 있습니다.

```c
#ifndef STRINGLIB_STRINGDEFS_H
#define STRINGLIB_STRINGDEFS_H

/* this is sort of a hack.  there's at least one place (formatting
   floats) where some stringlib code takes a different path if it's
   compiled as unicode. */
#define STRINGLIB_IS_UNICODE     0

#define FASTSEARCH fastsearch
#define STRINGLIB(F) stringlib_##F
#define STRINGLIB_OBJECT         PyBytesObject
#define STRINGLIB_SIZEOF_CHAR    1
#define STRINGLIB_CHAR           char
#define STRINGLIB_TYPE_NAME      "string"
#define STRINGLIB_PARSE_CODE     "S"
#define STRINGLIB_ISSPACE        Py_ISSPACE
#define STRINGLIB_ISLINEBREAK(x) ((x == '\n') || (x == '\r'))
#define STRINGLIB_ISDECIMAL(x)   ((x >= '0') && (x <= '9'))
#define STRINGLIB_TODECIMAL(x)   (STRINGLIB_ISDECIMAL(x) ? (x - '0') : -1)
#define STRINGLIB_STR            PyBytes_AS_STRING
#define STRINGLIB_LEN            PyBytes_GET_SIZE
#define STRINGLIB_NEW            PyBytes_FromStringAndSize
#define STRINGLIB_CHECK          PyBytes_Check
#define STRINGLIB_CHECK_EXACT    PyBytes_CheckExact
#define STRINGLIB_TOSTR          PyObject_Str
#define STRINGLIB_TOASCII        PyObject_Repr
#endif /* !STRINGLIB_STRINGDEFS_H */
```
열 몇가지의 #define만이 들어있는 간단한 파일입니다. 여기에서 정의하고 있는 것들은 전부 `STRINGLIB`로 시작하는 상수들인데, 각종 매크로들을 바이트 문자열을 위한 값으로 대체하는 역할을 하고 있음을 볼 수 있습니다. 예를 들어 바이트 문자열은 내부적으로 `char`로 표현되므로 `STRINGLIB_CHAR`은 `char`로 변환되며, 개행 문자는 '\n' 또는 '\r', 숫자에 해당하는 문자인지를 검사하는 함수는 즉석에서 매크로 함수로 '0' 이상 '9' 이하인지를 체크하고 있습니다.

이와는 대조적으로, 유니코드 문자열을 위한 헤더 https://github.com/python/cpython/blob/main/Objects/stringlib/unicodedefs.h 는 다음과 같이 되어있습니다.

```c
#ifndef STRINGLIB_UNICODEDEFS_H
#define STRINGLIB_UNICODEDEFS_H

/* this is sort of a hack.  there's at least one place (formatting
   floats) where some stringlib code takes a different path if it's
   compiled as unicode. */
#define STRINGLIB_IS_UNICODE     1

#define FASTSEARCH               fastsearch
#define STRINGLIB(F)             stringlib_##F
#define STRINGLIB_OBJECT         PyUnicodeObject
#define STRINGLIB_SIZEOF_CHAR    Py_UNICODE_SIZE
#define STRINGLIB_CHAR           Py_UNICODE
#define STRINGLIB_TYPE_NAME      "unicode"
#define STRINGLIB_PARSE_CODE     "U"
#define STRINGLIB_ISSPACE        Py_UNICODE_ISSPACE
#define STRINGLIB_ISLINEBREAK    BLOOM_LINEBREAK
#define STRINGLIB_ISDECIMAL      Py_UNICODE_ISDECIMAL
#define STRINGLIB_TODECIMAL      Py_UNICODE_TODECIMAL
#define STRINGLIB_STR            PyUnicode_AS_UNICODE
#define STRINGLIB_LEN            PyUnicode_GET_SIZE
#define STRINGLIB_NEW            PyUnicode_FromUnicode
#define STRINGLIB_CHECK          PyUnicode_Check
#define STRINGLIB_CHECK_EXACT    PyUnicode_CheckExact

#define STRINGLIB_TOSTR          PyObject_Str
#define STRINGLIB_TOASCII        PyObject_ASCII

#define STRINGLIB_WANT_CONTAINS_OBJ 1

#endif /* !STRINGLIB_UNICODEDEFS_H */
```

아까와는 달리 `STRINGLIB_CHAR`는 `Py_UNICODE`로, 개행 체크는 `BLOOM_LINEBREAK` 함수로, 숫자 체크는 `Py_UNICODE_ISDECIMAL`로 달라져 있습니다. 이와 같이 사용하는 문자열의 종류에 종속되어 라이브러리 함수들도 만들어지게 됩니다.

String 라이브러리에서는 또한 ASCII, UCS1, UCS2, UCS4를 지원하기 때문에 이를 위한 다음과 같은 헤더 파일들 또한 존재합니다.
* https://github.com/python/cpython/blob/main/Objects/stringlib/asciilib.h
* https://github.com/python/cpython/blob/main/Objects/stringlib/ucs1lib.h
* https://github.com/python/cpython/blob/main/Objects/stringlib/ucs2lib.h
* https://github.com/python/cpython/blob/main/Objects/stringlib/ucs4lib.h

그러면 이제 이러한 라이브러리가 어떤 식으로 호출되는지 예시를 들어 설명하겠습니다. 이전에 fastsearch의 시작 지점으로 언급했었던 STRINGLIB(find) 함수를 유니코드 형식의 문자열에서 호출하면 다음과 같은 과정을 거치게 됩니다.

```c
Py_ssize_t
PyUnicode_Find(PyObject *str,
               PyObject *substr,
               Py_ssize_t start,
               Py_ssize_t end,
               int direction)
{
    if (ensure_unicode(str) < 0 || ensure_unicode(substr) < 0)
        return -2;

    return any_find_slice(str, substr, start, end, direction);
}
```

`PyUnicode_Find`는 우선 `any_find_slice` 함수를 호출합니다. 이 함수는 다음과 같이 구성되어 있습니다.

```c
static Py_ssize_t
any_find_slice(PyObject* s1, PyObject* s2,
               Py_ssize_t start,
               Py_ssize_t end,
               int direction)
{
    int kind1, kind2;
    const void *buf1, *buf2;
    Py_ssize_t len1, len2, result;

    kind1 = PyUnicode_KIND(s1);
    kind2 = PyUnicode_KIND(s2);
    if (kind1 < kind2)
        return -1;

    len1 = PyUnicode_GET_LENGTH(s1);
    len2 = PyUnicode_GET_LENGTH(s2);
    ADJUST_INDICES(start, end, len1);
    if (end - start < len2)
        return -1;

    buf1 = PyUnicode_DATA(s1);
    buf2 = PyUnicode_DATA(s2);
    if (len2 == 1) {
        Py_UCS4 ch = PyUnicode_READ(kind2, buf2, 0);
        result = findchar((const char *)buf1 + kind1*start,
                          kind1, end - start, ch, direction);
        if (result == -1)
            return -1;
        else
            return start + result;
    }

    if (kind2 != kind1) {
        buf2 = unicode_askind(kind2, buf2, len2, kind1);
        if (!buf2)
            return -2;
    }

    if (direction > 0) {
        switch (kind1) {
        case PyUnicode_1BYTE_KIND:
            if (PyUnicode_IS_ASCII(s1) && PyUnicode_IS_ASCII(s2))
                result = asciilib_find_slice(buf1, len1, buf2, len2, start, end);
            else
                result = ucs1lib_find_slice(buf1, len1, buf2, len2, start, end);
            break;
        case PyUnicode_2BYTE_KIND:
            result = ucs2lib_find_slice(buf1, len1, buf2, len2, start, end);
            break;
        case PyUnicode_4BYTE_KIND:
            result = ucs4lib_find_slice(buf1, len1, buf2, len2, start, end);
            break;
        default:
            Py_UNREACHABLE();
        }
    }
    else {
        switch (kind1) {
        case PyUnicode_1BYTE_KIND:
            if (PyUnicode_IS_ASCII(s1) && PyUnicode_IS_ASCII(s2))
                result = asciilib_rfind_slice(buf1, len1, buf2, len2, start, end);
            else
                result = ucs1lib_rfind_slice(buf1, len1, buf2, len2, start, end);
            break;
        case PyUnicode_2BYTE_KIND:
            result = ucs2lib_rfind_slice(buf1, len1, buf2, len2, start, end);
            break;
        case PyUnicode_4BYTE_KIND:
            result = ucs4lib_rfind_slice(buf1, len1, buf2, len2, start, end);
            break;
        default:
            Py_UNREACHABLE();
        }
    }

    assert((kind2 != kind1) == (buf2 != PyUnicode_DATA(s2)));
    if (kind2 != kind1)
        PyMem_Free((void *)buf2);

    return result;
}
```

복잡해 보이지만 눈여겨 볼 부분은 `len2 == 1`인 경우와 그렇지 않은 경우를 나누는 부분입니다. `len2 == 1`인 경우 전체 문자열에서 한 글자를 찾는 것이므로 `findchar`가 대신 호출됩니다. 그 외의 경우에는 두 가지로 나누어, 원래 순서대로 문자열을 찾는 경우와 역순으로 찾는 경우 두 가지로 분기합니다. 여기서 각 유니코드의 세부 종류에 따라 호출하는 함수가 달라지는 것을 볼 수 있습니다. 예를 들어 아스키 문자열인 경우 `asciilib_find_slice`가 호출되고, UCS1, UCS2, UCS4에 대해서는 각각 `ucs1lib_find_slice`, `ucs2lib_find_slice`, `ucs4lib_find_slice`가 호출됩니다.

이 함수들은 각각(?) https://github.com/python/cpython/blob/main/Objects/stringlib/find.h 에 있습니다.

```c
Py_LOCAL_INLINE(Py_ssize_t)
STRINGLIB(find_slice)(const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                     const STRINGLIB_CHAR* sub, Py_ssize_t sub_len,
                     Py_ssize_t start, Py_ssize_t end)
{
    return STRINGLIB(find)(str + start, end - start, sub, sub_len, start);
}
```

`STRINGLIB`는 위의 헤더 파일들에서 정의한 바에 따라 앞에 접두어로 `asciilib_`, `ucs1lib_`, `ucs2lib_`, `ucs4lib_` 등을 붙이게 됩니다. 그래서 `asciilib_find_slice`와 같은 함수들이 개별적으로 같은 코드로부터 만들어지게 되고, 실제로 호출되는 함수는 이렇게 `any_find_slice` 함수에서의 분기에 따라 달라지게 됩니다.

이 함수가 마지막으로 `STRINGLIB(find)` 함수를 호출하며, 이전 글에서 보았던 본격적인 `find` 루트가 시작됩니다.

## 문자열 라이브러리의 헤더 파일들 ##
여기까지 문자열 라이브러리의 구성과 동작 방식을 전체적으로 알아보았습니다. 지금부터는 실제로 문자열 라이브러리가 가진 다양한 기능들이 헤더 파일들에 어떻게 구현되어 있는지 살펴보도록 하겠습니다.

### [codecs.h](https://github.com/python/cpython/blob/main/Objects/stringlib/codecs.h) ###
이 헤더 파일에는 문자열 인코딩을 위한 함수들이 들어있습니다. UTF8, UTF16, UTF32에 대한 인코딩을 지원하며, 정의된 인코딩 형식에 따라 한 글자씩 보며 인코딩을 수행하는 자신의 역할에 충실한 함수들이 있습니다.

### [count.h](https://github.com/python/cpython/blob/main/Objects/stringlib/count.h) ###
문자열 내에서 문자열을 세기 위한 `count` 기능을 지원하는 함수입니다...만 실제로 이 헤더에서 하는 일은 그다지 없습니다. 이 헤더는 다음이 전부입니다.

```c
/* stringlib: count implementation */

#ifndef STRINGLIB_FASTSEARCH_H
#error must include "stringlib/fastsearch.h" before including this module
#endif

Py_LOCAL_INLINE(Py_ssize_t)
STRINGLIB(count)(const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                const STRINGLIB_CHAR* sub, Py_ssize_t sub_len,
                Py_ssize_t maxcount)
{
    Py_ssize_t count;

    if (str_len < 0)
        return 0; /* start > len(str) */
    if (sub_len == 0)
        return (str_len < maxcount) ? str_len + 1 : maxcount;

    count = FASTSEARCH(str, str_len, sub, sub_len, maxcount, FAST_COUNT);

    if (count < 0)
        return 0; /* no match */

    return count;
}
```
이와 같이 단 하나의 함수 `STRINGLIB(count)`를 가지고 있으며, 문자열의 길이에 따른 예외 처리를 수행하는 것 외에는 모두 `FASTSEARCH`에게 일을 넘깁니다.

### [ctype.h](https://github.com/python/cpython/blob/main/Objects/stringlib/ctype.h) ###
C 표준 헤더인 ctype.h와 유사한 역할을 하는 헤더입니다. 다만 이들을 Python이 지원하는 바이트 문자 형식에 맞게 동작할 수 있게끔 구현한 함수들이 들어 있습니다. 오로지 바이트 문자에 대해서만 쓸 수 있으며 유니코드 형식에 대해서는 동작하지 않습니다.

### [eq.h](https://github.com/python/cpython/blob/main/Objects/stringlib/eq.h) ###
유니코드 형식의 두 문자열이 같은지를 비교하는 함수 `unicode_eq` 하나만이 들어있는 헤더 파일입니다. Dictobject와 setobject용으로 특별히 빠르게 검사하기 위해 만들어졌으며, 속도를 높이기 위해 `memcmp`를 사용합니다.

```c
/* Fast unicode equal function optimized for dictobject.c and setobject.c */

/* Return 1 if two unicode objects are equal, 0 if not.
 * unicode_eq() is called when the hash of two unicode objects is equal.
 */
Py_LOCAL_INLINE(int)
unicode_eq(PyObject *aa, PyObject *bb)
{
    assert(PyUnicode_Check(aa));
    assert(PyUnicode_Check(bb));
    assert(PyUnicode_IS_READY(aa));
    assert(PyUnicode_IS_READY(bb));

    PyUnicodeObject *a = (PyUnicodeObject *)aa;
    PyUnicodeObject *b = (PyUnicodeObject *)bb;

    if (PyUnicode_GET_LENGTH(a) != PyUnicode_GET_LENGTH(b))
        return 0;
    if (PyUnicode_GET_LENGTH(a) == 0)
        return 1;
    if (PyUnicode_KIND(a) != PyUnicode_KIND(b))
        return 0;
    return memcmp(PyUnicode_1BYTE_DATA(a), PyUnicode_1BYTE_DATA(b),
                  PyUnicode_GET_LENGTH(a) * PyUnicode_KIND(a)) == 0;
}
```

### [fastsearch.h](https://github.com/python/cpython/blob/main/Objects/stringlib/fastsearch.h) ###
문자열 내에서 다른 문자열을 빠르게 찾기 위한 알고리즘이 구현된 헤더 파일로, 자세한 설명은 [이전 글](http://www.secmem.org/blog/2021/08/18/fastsearch/)을 참고하면 됩니다.

### [find.h](https://github.com/python/cpython/blob/main/Objects/stringlib/find.h) ###
Fastsearch와 같이 윗글에 설명되어 있습니다.

### [find_max_char.h](https://github.com/python/cpython/blob/main/Objects/stringlib/find_max_char.h) ###
이 헤더 파일에는 버퍼 내의 유니코드 문자들의 최적의 크기를 찾아내기 위한 함수인 `STRINGLIB(find_max_char)`가 들어있습니다. 이 함수는 정의된 매크로에 따라 두 가지의 버전으로 나뉘는데, 문자 하나의 크기가 1인 경우와 그렇지 않은 경우에 따라 갈리게 됩니다. 유니코드 문자열에 대해서만 사용 가능합니다.

문자의 크기가 1인 경우에 대해서만 간단히 살펴보면 다음과 같습니다. 127을 넘는 값을 담을 여지가 있는 문자열이면 255가, 그렇지 않은 경우 127을 최댓값으로 반환하게 됩니다.

```c
Py_LOCAL_INLINE(Py_UCS4)
STRINGLIB(find_max_char)(const STRINGLIB_CHAR *begin, const STRINGLIB_CHAR *end)
{
    const unsigned char *p = (const unsigned char *) begin;

    while (p < end) {
        if (_Py_IS_ALIGNED(p, ALIGNOF_SIZE_T)) {
            /* Help register allocation */
            const unsigned char *_p = p;
            while (_p + SIZEOF_SIZE_T <= end) {
                size_t value = *(const size_t *) _p;
                if (value & UCS1_ASCII_CHAR_MASK)
                    return 255;
                _p += SIZEOF_SIZE_T;
            }
            p = _p;
            if (p == end)
                break;
        }
        if (*p++ & 0x80)
            return 255;
    }
    return 127;
}
```

### [join.h](https://github.com/python/cpython/blob/main/Objects/stringlib/join.h) ###
이 헤더 파일에는 바이트 문자열의 `join` 기능에 대한 구현체가 들어있습니다. 이를 수행하는 함수의 원형은 다음과 같습니다.

```c
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(bytes_join)(PyObject *sep, PyObject *iterable);
```

Python 문법에서 사용하는 방식 그대로 구분자 `sep`와 주로 리스트로 쓰게 되는 `iterable`로 이루어져 있습니다. `iterable`의 각 원소들을 `sep`를 사이사이에 끼워넣으며 한 문자열로 합치는 역할을 합니다.

### [localeutil.h](https://github.com/python/cpython/blob/main/Objects/stringlib/localeutil.h) ##
천 단위로 끊어읽기를 지원하는 기능인 `_PyUnicode_InsertThousandsGrouping()`의 헬퍼 함수들이 있는 헤더 파일입니다.

### [partition.h](https://github.com/python/cpython/blob/main/Objects/stringlib/partition.h) ###
특정 문자열을 기준으로 분할을 수행하는 `partition` 기능이 구현된 헤더 파일입니다. 이 함수 역시 정방향과 역방향으로 분할을 수행하기 위한 함수가 각각 다음과 같이 제공됩니다.

```c
Py_LOCAL_INLINE(PyObject*)
STRINGLIB(partition)(PyObject* str_obj,
                    const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                    PyObject* sep_obj,
                    const STRINGLIB_CHAR* sep, Py_ssize_t sep_len);
Py_LOCAL_INLINE(PyObject*)
STRINGLIB(rpartition)(PyObject* str_obj,
                     const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                     PyObject* sep_obj,
                     const STRINGLIB_CHAR* sep, Py_ssize_t sep_len);
```

### [replace.h](https://github.com/python/cpython/blob/main/Objects/stringlib/replace.h) ###
문자열의 특정 부분 문자열을 다른 문자열로 대체하는 기능인 `replace`의 구현체가 있어야 할 것 같지만... 실제로는 그 중에서도 하나의 문자를 다른 문자로 빠르게 바꾸기 위한 특별한 함수 `STRINGLIB(replace_1char_inplace)`만이 있는 헤더 파일입니다. 이 함수는 순서대로 한 글자씩 보면서 해당 문자를 찾아 바꾸는데, 바꿔야 할 문자가 자주 발견되면 그대로 한 글자씩 루프를 돌리지만 그 간격이 넓은 경우 fastsearch를 이용하여 다음 나타나는 위치를 더 빠르게 찾기 위해 노력(?)합니다.

```c
Py_LOCAL_INLINE(void)
STRINGLIB(replace_1char_inplace)(STRINGLIB_CHAR* s, STRINGLIB_CHAR* end,
                                 Py_UCS4 u1, Py_UCS4 u2, Py_ssize_t maxcount);
```

### [split.h](https://github.com/python/cpython/blob/main/Objects/stringlib/split.h) ###
특정 문자열을 기준으로 문자열을 분할하여 리스트로 만드는 `split` 기능이 구현된 파일입니다. 여기서도 최적화를 위해 기준이 되는 문자열이 하나의 문자인 경우와 둘 이상의 문자로 이루어진 문자열인 경우로 나누어져 있으며, 공백 문자인 경우도 분리되어 있고, 역순으로 찾는 함수들도 쌍으로 존재합니다. 또한 개행을 기준으로 나누는 함수도 부록으로 있습니다.

```c
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(split_whitespace)(PyObject* str_obj,
                           const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                           Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(split_char)(PyObject* str_obj,
                    const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                    const STRINGLIB_CHAR ch,
                    Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(split)(PyObject* str_obj,
                const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                const STRINGLIB_CHAR* sep, Py_ssize_t sep_len,
                Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(rsplit_whitespace)(PyObject* str_obj,
                            const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                            Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(rsplit_char)(PyObject* str_obj,
                      const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                      const STRINGLIB_CHAR ch,
                      Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(rsplit)(PyObject* str_obj,
               const STRINGLIB_CHAR* str, Py_ssize_t str_len,
               const STRINGLIB_CHAR* sep, Py_ssize_t sep_len,
               Py_ssize_t maxcount);
Py_LOCAL_INLINE(PyObject *)
STRINGLIB(splitlines)(PyObject* str_obj,
                    const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                    int keepends);
```

### [transmogrify.h](https://github.com/python/cpython/blob/main/Objects/stringlib/transmogrify.h) ###
앞서 살펴본 `replace.h`에는 없었던, 둘 이상의 문자로 이루어진 문자열을 `replace`하는 함수들이 들어있는 헤더 파일입니다. 바이트 문자열에 대해서만 사용 가능하며, 상당히 복잡한 알고리즘을 통해 효율성을 극대화시키고자 하고 있습니다. Fastsearch와 비슷하게, 문자열 내에서 문자열들을 찾아야 하는 작업이기에 자칫하면 큰 시간 복잡도가 될 수 있을 뿐 아니라 시간 복잡도를 보장하는 구현이 자칫하면 일상에서 쓰이는 대부분의 간단한 패턴에도 무겁게 동작할 수 있기에 각별히 신경을 써서 만든 듯 합니다.

이 최적화를 위한 기법들에도 흥미로운 요소가 많으니, 언젠가 기회가 된다면 이에 대한 분석글을 따로 작성해볼 만도 한 것 같습니다.

## 마치며 ##
문자열의 처음을 C로 접한 제게 문자열은 참으로 다루기 어려운 분야라는 인식이 강했습니다. 그러다가 C++의 `std::string`을 쓰기 시작하면서 이들을 다루기 편하게 만든 라이브러리들에 점차 익숙해져, 그 기능들을 실제로 구현하는 것이 얼마나 어려운지를 오랫동안 잊고 있었던 것 같습니다. 이번에 두 차례에 걸쳐 Python의 `str` 구현 코드를 분석해 보면서, 우리가 당연하게 사용하고 있었던 문자열 관련 기능들 뒤에는 한없이 복잡하고 어려운, 그렇지만 너무나 깔끔하게 맞물려 돌아가는 거대한 구현체가 있다는 것을 느낄 수 있었습니다.
