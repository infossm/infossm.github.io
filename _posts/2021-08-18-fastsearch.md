---
layout: post
title:  "Python 3의 문자열 fastsearch 기법들"
date:   2021-08-18 19:47:26
author: djm03178
tags: python,str,string,find
---

## 개요 ##
Python 3는 내장된 수많은 기능을 통해 사용자에게 편의성을 제공합니다. 여기에는 문자열 자료형인 `str`과 관련 라이브러리들이 포함되는데, 대부분의 기능이 언어 자체에 내장되어 일반 리스트처럼 대괄호를 사용한 인덱스 접근도 가능하고, '문자'와 '문자열'의 구분도 없으며, 널 문자같은 귀찮은 요소도 신경쓰지 않아도 되고 마음대로 인덱스를 지정해 잘라내거나(slicing) `+` 연산자로 이어붙이고 심지어는 `*` 연산자로 반복까지 할 수 있는 등 만능의 성격을 갖추고 있습니다.

이처럼 높은 접근성을 고려한 구현은 CPython의 문자열 구현체에서도 드러나는데, 문자열 내에서 문자열을 검색하기 위한 `find` 기능이 대표적입니다. 현실에서 사용될 법한 다양한 케이스들을 고려하여, 단순한 알고리즘 하나로 탐색을 수행하는 데에 그치지 않고 여러 휴리스틱한 방법들을 통해 일반적인 상황에서 더욱 빠른 탐색 시간을 달성하는 데에 초점을 두고 있습니다.[^1] 이 글에서는 CPython의 fastsearch 구현체를 분석하면서 여기에 사용된 다양한 기법들이 어떤 이점이 있는지 알아보도록 하겠습니다.

## 전체적인 실행 흐름 ##
구체적으로 들어가기 전에, 우선 문자열에서 탐색을 수행할 때의 전체적인 실행 흐름을 따라가 보겠습니다.

탐색의 시작은 [find.h](https://github.com/python/cpython/blob/main/Objects/stringlib/find.h)에 있는 `STRINGLIB(find)` 함수부터입니다.

```c
Py_LOCAL_INLINE(Py_ssize_t)
STRINGLIB(find)(const STRINGLIB_CHAR* str, Py_ssize_t str_len,
               const STRINGLIB_CHAR* sub, Py_ssize_t sub_len,
               Py_ssize_t offset)
{
    Py_ssize_t pos;

    assert(str_len >= 0);
    if (sub_len == 0)
        return offset;

    pos = FASTSEARCH(str, str_len, sub, sub_len, -1, FAST_SEARCH);

    if (pos >= 0)
        pos += offset;

    return pos;
}
```

본격적으로 탐색을 시작하기 전 간단한 검사와 offset에 대한 처리를 위한 함수로, 실질적인 탐색은 `FASTSEARCH` 함수에서 이루어집니다. 이 함수는 [fastsearch.h](https://github.com/python/cpython/blob/main/Objects/stringlib/fastsearch.h)에 있습니다. `str` 객체 내부적으로 저장하고 있는 원초적인 문자 배열(`s`, `p`)과 각각의 길이를 저장하는 값(`n`, `m`)을 이용하는 것을 볼 수 있습니다.

```c
Py_LOCAL_INLINE(Py_ssize_t)
FASTSEARCH(const STRINGLIB_CHAR* s, Py_ssize_t n,
           const STRINGLIB_CHAR* p, Py_ssize_t m,
           Py_ssize_t maxcount, int mode)
{
    ...
}

```

이 함수는 탐색 모드(`mode`)와 문자열의 길이 등에 따라 탐색 방법을 정하는 분기점의 역할을 합니다. 우선, 처음으로 하는 것은 찾으려는 문자열(`p`)이 원래 문자열(`s`)보다 더 길거나 매칭되는 위치를 최대 0개(?) 찾는 경우에 대한 예외 처리입니다.

```c
if (n < m || (mode == FAST_COUNT && maxcount == 0)) {
    return -1;
}
```

그 다음은 찾는 문자열의 길이가 1 이하인 특수인 경우에 대한 빠른 처리를 위한 분기입니다.

```c
/* look for special cases */
if (m <= 1) {
    if (m <= 0) {
        return -1;
    }
    /* use special case for 1-character strings */
    if (mode == FAST_SEARCH)
        return STRINGLIB(find_char)(s, n, p[0]);
    else if (mode == FAST_RSEARCH)
        return STRINGLIB(rfind_char)(s, n, p[0]);
    else {
        return STRINGLIB(count_char)(s, n, p[0], maxcount);
    }
}
```

우선 찾는 문자열이 빈 문자열이면 찾을 수 없는 것으로 처리하고 -1을 반환하는데, 이는 `find` 메서드가 이런 경우 0을 반환하는 것과는 차이가 있습니다. 이는 위에서 살펴본 `STRINGLIB(find)` 함수에서 이런 경우에 대한 예외 처리를 미리 하기 때문입니다. 다른 경로를 통해 `FASTSEARCH`가 호출되었을 때에만 의미가 있는 문장입니다.

그 외에는 `mode`에 따라서 문자열에서 한 글자만을 찾는 특별한 함수들을 호출합니다. `FAST_SEARCH`와 `FAST_RSEARCH`는 각각 문자를 앞에서부터 찾을지, 뒤에서부터 찾을지를 나타내는 값으로 각각 `STRINGLIB(find_char)`와 `STRINGLIB(rfind_char)` 함수를 통해 탐색합니다. 두 모드 모두 아닌 경우는 최대 `maxcount` 개수만큼을 찾는 `STRINGLIB(count_char)`를 호출합니다.

특수 케이스가 아닌 경우 일반적인 탐색 루틴으로 들어가는데, 여기서도 여러 분기가 있습니다.

```c
if (mode != FAST_RSEARCH) {
    if (n < 2500 || (m < 100 && n < 30000) || m < 6) {
        return STRINGLIB(default_find)(s, n, p, m, maxcount, mode);
    }
    else if ((m >> 2) * 3 < (n >> 2)) {
        /* 33% threshold, but don't overflow. */
        /* For larger problems where the needle isn't a huge
           percentage of the size of the haystack, the relatively
           expensive O(m) startup cost of the two-way algorithm
           will surely pay off. */
        if (mode == FAST_SEARCH) {
            return STRINGLIB(_two_way_find)(s, n, p, m);
        }
        else {
            return STRINGLIB(_two_way_count)(s, n, p, m, maxcount);
        }
    }
    else {
        /* To ensure that we have good worst-case behavior,
           here's an adaptive version of the algorithm, where if
           we match O(m) characters without any matches of the
           entire needle, then we predict that the startup cost of
           the two-way algorithm will probably be worth it. */
        return STRINGLIB(adaptive_find)(s, n, p, m, maxcount, mode);
    }
}
else {
    /* FAST_RSEARCH */
    return STRINGLIB(default_rfind)(s, n, p, m, maxcount, mode);
}
```

먼저 역순으로 찾는 모드가 아닌 경우입니다. 여기서 휴리스틱하게 커트를 정해서, 문자열의 길이가 너무 크지 않은 경우는 `STRINGLIB(default_find)` 함수를 통해 문자열을 찾습니다. 그렇지 않고 만일 찾고자 하는 문자열이 원래 문자열의 길이의 ${1}\over{3}$ 미만인 경우 `STRINGLIB(_two_way_find)` 혹은 `STRINGLIB(_two_way_count)`를 통해 찾으며, ${1}\over{3}$ 이상인 경우에는 `STRINGLIB(adaptive_find)`를 호출합니다.

역순으로 찾는 경우에는 `STRINGLIB(default_rfind)`를 호출합니다.

## 한 문자 찾기: `find_char` ##
찾고자 하는 문자열이 문자 하나로 이루어진 경우는 문자열로 취급하는 것보다 CPU의 도움을 받아 문자 단위로 탐색하는 것이 훨씬 빠릅니다. 이를 적극 활용하는 것이 `STRINGLIB(find_char)` 함수입니다.

```c
Py_LOCAL_INLINE(Py_ssize_t)
STRINGLIB(find_char)(const STRINGLIB_CHAR* s, Py_ssize_t n, STRINGLIB_CHAR ch)
{
    const STRINGLIB_CHAR *p, *e;

    p = s;
    e = s + n;
    if (n > MEMCHR_CUT_OFF) {
#if STRINGLIB_SIZEOF_CHAR == 1
        p = memchr(s, ch, n);
        if (p != NULL)
            return (p - s);
        return -1;
#else
        /* use memchr if we can choose a needle without too many likely
           false positives */
        const STRINGLIB_CHAR *s1, *e1;
        unsigned char needle = ch & 0xff;
        /* If looking for a multiple of 256, we'd have too
           many false positives looking for the '\0' byte in UCS2
           and UCS4 representations. */
        if (needle != 0) {
            do {
                void *candidate = memchr(p, needle,
                                         (e - p) * sizeof(STRINGLIB_CHAR));
                if (candidate == NULL)
                    return -1;
                s1 = p;
                p = (const STRINGLIB_CHAR *)
                        _Py_ALIGN_DOWN(candidate, sizeof(STRINGLIB_CHAR));
                if (*p == ch)
                    return (p - s);
                /* False positive */
                p++;
                if (p - s1 > MEMCHR_CUT_OFF)
                    continue;
                if (e - p <= MEMCHR_CUT_OFF)
                    break;
                e1 = p + MEMCHR_CUT_OFF;
                while (p != e1) {
                    if (*p == ch)
                        return (p - s);
                    p++;
                }
            }
            while (e - p > MEMCHR_CUT_OFF);
        }
#endif
    }
    while (p < e) {
        if (*p == ch)
            return (p - s);
        p++;
    }
    return -1;
}
```

이 함수는 빌드 옵션에 따라 두 가지 형태로 갈라집니다. 문자의 크기가 1바이트인 경우는 단순하게 바이트 단위로 일치하는 값을 찾아주는 `memchr` 함수를 통해 나온 값을 그대로 반환하는 것이 가장 신속한 방법이므로 간단하게 처리할 수 있습니다. 하지만 유니코드 등 1바이트를 넘는 크기인 경우에는 조금 복잡해집니다. `memchr`를 활용하면 일반적으로 CPU의 도움을 크게 받을 수 있기에 여전히 사용은 하지만, 문자를 구성하는 모든 바이트가 정확하게 일치하는 것을 찾아주지는 못하기 때문입니다.

그래도 이를 최대한 활용하기 위해, 하위 1바이트만을 추출한 `needle`이라는 변수를 통해 `memchr`으로 '후보'를 찾아내고, 그 위치에 글자 전체가 일치하는 값이 있는지를 반복적으로 검사하는 방식으로 실제 위치를 찾는 방법을 사용합니다. `memchr`가 효율적일 정도로 범위가 큰 동안 이를 반복하고, 좁은 범위가 되면 평범한 루프로 바로 비교하여 찾아냅니다.

역순으로 찾는 함수는 `STRINGLIB(rfind_char)`인데, 이는 GCC 확장 함수인 `memrchr` 함수를 사용하고 루프도 뒤쪽부터 앞쪽의 방향으로 실행하여 가장 뒤쪽에서 일치하는 위치를 찾는다는 점을 제외하고는 같은 역할을 합니다.

일치하는 개수를 찾는 `STRINGLIB(count_char)` 함수는 매우 단순한 나이브한 루프로 구성되어 있습니다.

```c
static inline Py_ssize_t
STRINGLIB(count_char)(const STRINGLIB_CHAR *s, Py_ssize_t n,
                      const STRINGLIB_CHAR p0, Py_ssize_t maxcount)
{
    Py_ssize_t i, count = 0;
    for (i = 0; i < n; i++) {
        if (s[i] == p0) {
            count++;
            if (count == maxcount) {
                return maxcount;
            }
        }
    }
    return count;
}
```

## 일반적인 문자열 검색: `default_find` ##
적당히 긴 문자열 속에서 짧은 문자열을 찾는, 가장 일반적인 경우에 실행하게 되는 함수입니다. 실제 상황에서 많이 나올 법한 케이스를 빠르게 처리하고자 하는 Python의 철학이 담긴 부분이기도 합니다.

이 함수는 크게 봤을 때 가장 널리 사용되는 문자열 탐색 기법 중 하나인 Boyer-Moore-Horspool 알고리즘에 몇 가지 변형을 가한 형태입니다. [이 문서](https://web.archive.org/web/20201107074620/http://effbot.org/zone/stringlib.htm)에 이를 통해 이루고자 하는 목표들이 서술되어 있는데, 이를 옮겨적으면 다음과 같습니다.

* 기존의 브루트포스 방식보다 빠르다.
* 준비 오버헤드가 적다.
* 좋은 케이스들에서 $\mathcal{O}(n/m)$의 sublinear 시간 복잡도를 낸다.
* 최악의 경우에도 기존 알고리즘의 $\mathcal{O}(nm)$을 넘지 않는다.
* 8비트, 16비트, 32비트 문자형 모두에서 동작한다.
* 대부분의 현실적인 케이스에서 좋고, 나쁜 케이스는 거의 없다.
* 구현체가 간단하다.

### Boyer-Moore-Horspool 알고리즘 ###
코드를 보기 전에 우선 이 함수가 기반으로 하는 Boyer-Moore-Horspool 알고리즘이 무엇인지 간단하게 알아보겠습니다. 기존의 Boyer-Moore 알고리즘을 단순화시킨 이 알고리즘은 길이가 $\mathcal{O}(n)$인 문자열에서 길이가 $\mathcal{O}(m)$인 패턴 문자열을 찾을 때 평균 $\mathcal{O}(n)$ 시간, 최악의 경우 $\mathcal{O}(nm)$이 걸립니다.

이 알고리즘의 기본 원리는 bad character rule입니다. 모든 종류의 문자에 대해 각 문자가 발견되지 않을 경우 건너뛰어도 되는 최대의 문자 수를 저장해두는 것으로, 패턴 문자열의 길이에서 각 문자가 마지막으로 등장한 위치를 뺀 값이 그 문자에 대한 skip값이 됩니다. 등장하지 않는 문자의 경우 패턴 문자열의 길이로 적용합니다.

여기까지에 대한 [위키백과의 의사 코드](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm)는 다음과 같습니다.

```python
function preprocess(pattern)
    T ← new table of 256 integers
    for i from 0 to 256 exclusive
        T[i] ← length(pattern)
    for i from 0 to length(pattern) - 1 exclusive
        T[pattern[i]] ← length(pattern) - 1 - i
    return T
```

문자열을 검색할 때에는 각 문자가 일치하지 않을 경우 원래 문자열의 해당 문자에 대해 건너뛰어도 되는 만큼을 한 번에 건너뛰게 됩니다. 평균적 / 현실적인 경우 길게 일치하는 케이스가 많지 않기 때문에 긴 거리를 빠르게 이동할 수 있습니다.

```python
function search(needle, haystack)
    T ← preprocess(needle)
    skip ← 0
    while length(haystack) - skip ≥ length(needle)
        haystack[skip:] -- substring starting with "skip". &haystack[skip] in C.
        if same(haystack[skip:], needle, length(needle))   
            return skip  
        skip ← skip + T[haystack[skip + length(needle) - 1]]
    return not-found
```

### 변형 ###
이제 `STRINGLIB(default_find)` 함수에서 이를 어떻게 변형했는지 알아보겠습니다. 가장 주된 변경점은 여기에 Sunday 알고리즘의 아이디어를 적용했다는 것입니다. 이 알고리즘은 매칭에 실패한 문자가 아닌 현재 보고 있는 부분문자열의 바로 오른쪽에 있는 문자를 기준으로 스킵을 적용하는 것입니다. 자세한 설명은 [여기](https://www.inf.fh-flensburg.de/lang/algorithmen/pattern/sundayen.htm)에서 볼 수 있습니다.

이들을 적용한 전체 코드는 다음과 같습니다.

```c
static inline Py_ssize_t
STRINGLIB(default_find)(const STRINGLIB_CHAR* s, Py_ssize_t n,
                        const STRINGLIB_CHAR* p, Py_ssize_t m,
                        Py_ssize_t maxcount, int mode)
{
    const Py_ssize_t w = n - m;
    Py_ssize_t mlast = m - 1, count = 0;
    Py_ssize_t gap = mlast;
    const STRINGLIB_CHAR last = p[mlast];
    const STRINGLIB_CHAR *const ss = &s[mlast];

    unsigned long mask = 0;
    for (Py_ssize_t i = 0; i < mlast; i++) {
        STRINGLIB_BLOOM_ADD(mask, p[i]);
        if (p[i] == last) {
            gap = mlast - i - 1;
        }
    }
    STRINGLIB_BLOOM_ADD(mask, last);

    for (Py_ssize_t i = 0; i <= w; i++) {
        if (ss[i] == last) {
            /* candidate match */
            Py_ssize_t j;
            for (j = 0; j < mlast; j++) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            if (j == mlast) {
                /* got a match! */
                if (mode != FAST_COUNT) {
                    return i;
                }
                count++;
                if (count == maxcount) {
                    return maxcount;
                }
                i = i + mlast;
                continue;
            }
            /* miss: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
            else {
                i = i + gap;
            }
        }
        else {
            /* skip: check if next character is part of pattern */
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
        }
    }
    return mode == FAST_COUNT ? count : -1;
}
```

`STRINGLIB_BLOOM`은 비트마스킹을 이용하여 bloom filter를 구현한 것으로 각 문자가 패턴 문자열에 포함되는지 여부를 빠르게 확인하는 용도로 사용하고 있습니다. Boyer-Moore-Horspool 알고리즘과 Sunday 알고리즘의 장점을 섞어서, 매칭에 실패한 문자가 패턴에 포함된 문자가 아니면 Sunday 알고리즘처럼 패턴의 길이(`m`)만큼을 건너뛰고, 포함된 경우 패턴의 마지막 문자의 skip(`gap`)만큼을 건너뛰는 방식으로 보다 휴리스틱하게 적용하고 있습니다.

## 패턴 문자열이 상대적으로 짧을 때: `_two_way` ##
`STRINGLIB(default_find)` 함수는 원래 문자열과 패턴 문자열의 길이가 짧다면 최악의 경우가 좋지 않더라도 평균 시간을 믿고 충분히 해볼만하지만, 아무리 그렇다 해도 문자열의 길이가 길면서도 최악의 패턴을 갖추고 있을 때에도 이 방법을 고집할 수는 없습니다. 대신 패턴 문자열의 길이가 상대적으로 짧을 때에 최악의 시간 복잡도도 보장하면서도 평균 시간 복잡도가 좋은 다른 방법을 사용할 수 있는데, 바로 two-way 알고리즘입니다.

이 경우 `mode`가 `FAST_SEARCH`일 때와 아닐 때로 나누어 `STRINGLIB(_two_way_find)`와 `STRINGLKIB(_two_way_count)` 함수가 각각 호출되는데, 이들은 다음과 같으며 마지막에 `STRINGLIB(_two_way)` 함수로 모아집니다.

```c
static Py_ssize_t
STRINGLIB(_two_way_find)(const STRINGLIB_CHAR *haystack,
                         Py_ssize_t len_haystack,
                         const STRINGLIB_CHAR *needle,
                         Py_ssize_t len_needle)
{
    LOG("###### Finding \"%s\" in \"%s\".\n", needle, haystack);
    STRINGLIB(prework) p;
    STRINGLIB(_preprocess)(needle, len_needle, &p);
    return STRINGLIB(_two_way)(haystack, len_haystack, &p);
}


static Py_ssize_t
STRINGLIB(_two_way_count)(const STRINGLIB_CHAR *haystack,
                          Py_ssize_t len_haystack,
                          const STRINGLIB_CHAR *needle,
                          Py_ssize_t len_needle,
                          Py_ssize_t maxcount)
{
    LOG("###### Counting \"%s\" in \"%s\".\n", needle, haystack);
    STRINGLIB(prework) p;
    STRINGLIB(_preprocess)(needle, len_needle, &p);
    Py_ssize_t index = 0, count = 0;
    while (1) {
        Py_ssize_t result;
        result = STRINGLIB(_two_way)(haystack + index,
                                     len_haystack - index, &p);
        if (result == -1) {
            return count;
        }
        count++;
        if (count == maxcount) {
            return maxcount;
        }
        index += result + len_needle;
    }
    return count;
}
```

코드에서 보이듯이 two-way 알고리즘은 전처리 과정을 동반합니다. 다음과 같은 `STRINGLIB(prework)` 구조체와 `STRINGLIB(_preprocess)` 함수가 사용됩니다.

```c
typedef struct STRINGLIB(_pre) {
    const STRINGLIB_CHAR *needle;
    Py_ssize_t len_needle;
    Py_ssize_t cut;
    Py_ssize_t period;
    Py_ssize_t gap;
    int is_periodic;
    SHIFT_TYPE table[TABLE_SIZE];
} STRINGLIB(prework);

static void
STRINGLIB(_preprocess)(const STRINGLIB_CHAR *needle, Py_ssize_t len_needle,
                       STRINGLIB(prework) *p)
{
    ...
}
```

전처리가 수행된 후에는 다음의 `STRINGLIB(_two_way)` 함수가 호출됩니다.

```c
static Py_ssize_t
STRINGLIB(_two_way)(const STRINGLIB_CHAR *haystack, Py_ssize_t len_haystack,
                    STRINGLIB(prework) *p)
{
    ...
}
```

코드도 길고 알고리즘이 상대적으로 많이 복잡하므로 핵심적인 아이디어만 소개하자면, 보장된 시간 복잡도의 문자열 탐색으로 널리 알려진 Knuth–Morris–Pratt(KMP) 알고리즘을 *정방향으로* 탐색하는 것과, 위에서 살펴본 Boyer-Moore 알고리즘을 *역방향으로* 탐색하는 것을 합쳐놓은 알고리즘이라고 할 수 있습니다. 이렇게 두 방향의 탐색을 수행하기 때문에 이름도 two-way라고 지어진 것입니다. 최악의 경우에도 시간 복잡도 $\mathcal{O}(n)$을 보장하며 현실의 데이터에서도 제법 빠르기 때문에 glibc 등에서도 사용되는 등 자주 쓰이는 알고리즘입니다. 자세한 설명 및 시각화는 https://www-igm.univ-mlv.fr/~lecroq/string/node26.html 를 참고하면 좋을 것 같습니다.

## 그 외의 경우: `adaptive_find` ##
지금까지와 같이 빠른 탐색 기법을 선택하기 어려운 큰 데이터인 경우에는 `STRINGLIB(adaptive_find)` 함수로 들어갑니다. 이름에서도 드러나듯이 이 함수는 문자열을 탐색하는 중 특성을 분석하여 어떤 알고리즘을 적용하는 것이 더 유리할지 결정하는 방법을 사용합니다. 구체적으로는 `STRINGLIB(default_find)`의 코드를 사용하면서, 다음과 같은 부분이 추가된 형태입니다.

```c
static Py_ssize_t
STRINGLIB(adaptive_find)(const STRINGLIB_CHAR* s, Py_ssize_t n,
                         const STRINGLIB_CHAR* p, Py_ssize_t m,
                         Py_ssize_t maxcount, int mode)
{
    ...

    Py_ssize_t hits = 0, res;

    ...

    for (Py_ssize_t i = 0; i <= w; i++) {
        if (ss[i] == last) {
            /* candidate match */
            Py_ssize_t j;
            for (j = 0; j < mlast; j++) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }

            ...

            hits += j + 1;
            if (hits > m / 4 && w - i > 2000) {
                if (mode == FAST_SEARCH) {
                    res = STRINGLIB(_two_way_find)(s + i, n - i, p, m);
                    return res == -1 ? -1 : res + i;
                }
                else {
                    res = STRINGLIB(_two_way_count)(s + i, n - i, p, m,
                                                    maxcount - count);
                    return res + count;
                }
            }

            ...
        }
    }
    ...
}
```

`j`가 `m / 4`번 이상 증가되고 아직 탐색하지 않은 문자가 상수 개 이상이면 반드시 if문에 걸리므로 여기까지의 총 탐색 시간은 최악의 경우에도 $\mathcal{O}(m)$이 보장됩니다. 이만큼 `STRINGLIB(default_find)`의 방법으로 탐색을 했는데도 아직 패턴의 매칭을 찾지 못했다면 그냥 포기하고 나머지 부분은 two-way 알고리즘을 적용하는 것이 더 이득일 것으로 판단한다는 의미입니다.

이처럼 문자열 탐색이라는 단순한 작업을 하나 하는 데에도 일방적인 코드 몇 줄에 그치지 않고 다양한 휴리스틱과 최적화 기법을 적용하여 평균 및 현실적인 데이터에서 매우 빠르게 동작하도록, 그러면서도 최악의 경우에도 크게 나빠지지는 않도록 신경을 써서 노력한 흔적을 많이 찾을 수 있습니다.

## 한계 ##
많은 휴리스틱들이 그렇지만 이렇게 평균이나 현실의 데이터에서 이득을 보기 위해 최악의 경우의 성능이 좋지 않은 알고리즘을 사용하는 경우 저격을 당하기 쉽다는 문제점이 있습니다. 비록 cutoff를 두어 일정 이상으로 늘어나지는 않도록 했지만, 그 경계선에 아슬아슬하게 걸리는 최악의 데이터를 반복적으로 처리하게 하는 경우 처음부터 안전한 방법을 사용하는 것에 비해서는 대단히 큰 손해가 발생할 수밖에 없습니다.

특히 알고리즘 문제를 푸는 데에 익숙해져 있는 저같은 사람들에게는 이와 같은 단점이 크게 느껴질 수밖에 없습니다.[^2] 예를 들어 문자열과 패턴이 $T$개의 테스트 케이스에 걸쳐 주어질 수 있는데 총 문자열 / 패턴의 길이에만 제한이 있는 문제라면 얼마든지 이 코드가 $\mathcal{O}(Tnm)$에 동작하게 만드는 것이 가능합니다. $n=2499, m=1500$인 경우에도 이 코드는 $\mathcal{O}(nm)$을 수행하며 이는 빠르다고만은 할 수 없는 크기입니다. 최악의 경우를 중점적으로 고려해야 하는 PS 분야 등에서는 한계점으로 느껴질 수 있는 부분입니다.

## 마치며 ##
실용적인 라이브러리를 만든다는 것은 곧 현실에서 빈번하게 발생하는 시나리오들을 폭넓게 고려해야 한다는 것을 의미합니다. 비록 태생이 매우 느릴 수밖에 없는 Python이지만, 그럼에도 할 수 있는 한 최선을 다해 성능을 개선하고 대다수의 사용자들에게 만족스러운 경험을 주기 위해 이렇게 문자열 검색 기능 하나에도 많은 철학을 담아낸 것을 볼 수 있었습니다. 또한 이들이 사용한 기존의 실용적인 문자열 검색 알고리즘들에 대해서도 알아볼 좋은 기회가 되었습니다.

## 참고 자료 ##
* https://github.com/python/cpython/blob/main/Objects/stringlib/find.h
* https://github.com/python/cpython/blob/main/Objects/stringlib/fastsearch.h
* https://web.archive.org/web/20201107074620/http://effbot.org/zone/stringlib.htm
* https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm

[^1]: 널리 알려진 문자열 탐색 기법인 KMP 알고리즘 등을 액면 그대로 구현하는 것은 최악의 경우에도 빠르게 동작하는 것을 보장할 수 있지만, 현실의 데이터는 실제로는 중복이나 단순한 패턴들이 존재하는 경우가 많습니다. 이런 특성들을 적극 활용하여 일상적인 상황에서 더 빠르게 답을 찾아내는 것이 fastsearch의 목표입니다.
[^2]: 동시에 이에 의존한 많은 코드들을 저격하는 데이터를 추가할 수 있지 않을까 하는 기대감도 있습니다.
