---
layout: post
title:  "Intel Intrinsics(SIMD) 가이드"
date:   2022-02-15 03:00:00
author: blisstoner
tags: []
---

# 1. Introduction

우리가 주로 사용하는 Intel(AMD) 아키텍쳐에서는 SIMD(Single Instruction Multiple Data)를 이용할 수 있습니다. SIMD는 말 뜻 그대로 하나의 명령을 통해 여러 값의 연산을 동시에 처리하는 명령어 셋으로, 단순 덧셈/비교/뺄셈과 같은 작업을 병렬화해서 실행 시간을 줄일 수 있습니다. 영상 처리, 머신러닝 분야에서 SIMD는 빼놓을 수 없는 존재이고, 암호 모듈 구현에서도 SIMD를 통해 성능을 극한으로 끌어올려 벤치마크를 확인합니다.

저는 처음 SIMD를 건드려본게 암호 모듈 구현체를 수정해야 할 일이 있어서였습니다. 아무래도 처음 쓰는거라 낯설다보니 이런저런 자료를 찾아가며 맨 땅에 헤딩을 해서 사용법을 익힐 수 있었는데, 당시 익혔던 지식을 정리하는 차원에서 글을 작성합니다.

많은 독자분들이 PS에 관심이 많을 것을 고려해 PS와의 관련성을 간단하게 기술해보면, PS에 사실 특이점이 오지 않는 이상 PS 분야에서 SIMD가 정해인 일은 발생하지 않겠지만 특히 쿼리 문제에서 시간복잡도가 난해한 경우 오히려 SIMD가 더 빠르게 동작하는 경우를 찾아볼 수도 있고([예시](https://blog.naver.com/jinhan814/222322878603)), 출제자/검수자의 입장에서도 정말 깐깐하게 검수를 하고자 한다면 SIMD를 이용한 풀이로 문제가 통과되지는 않는지 확인해볼 수 있습니다.

# 2. SIMD 튜토리얼

## A. SIMD 지원 여부 확인

우리가 보통 SIMD라고 통칭해서 부르는 명령어 집합(Instruction Set)에는 `MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2`등 종류가 굉장히 많습니다. cpu 모델에 따라 명령어 집합의 지원 여부가 다를 수 있습니다. [ssecheck.cpp](https://gist.github.com/hi2p-perim/7855506) 코드를 실행해서 자신의 cpu에서 지원되는 명령어 집합을 확인하거나 [HWINFO](https://www.hwinfo.com/download/) 프로그램을 통해서도 확인할 수 있습니다. 확인 결과 지금 저의 컴퓨터 환경인 `Intel Core i7-10700F CPU @ 2.90GHz` 기준으로는 `SSE, SSE2, SSE3, SSE4.1, SSE4.2, AVX, AVX2` 등이 지원되는 한편 `AVX-512`는 지원되지 않음을 확인할 수 있었습니다.

## B. SIMD 기초 예제

Intel Intrinsics 함수 목록은 [여기](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)에서 확인할 수 있습니다. 끝도 없이 보이는 함수들과 다소 불친절한 설명에 당황할 수 있지만 마치 처음 프로그래밍을 배운 후 제일 먼저 Hello world를 출력해보듯 아주 간단한 예시부터 실행시켜 보겠습니다. 이번에 실행해보고자 하는 함수는 `_mm_add_epi32`로, 32비트 정수의 덧셈 4개를 한 번에 처리하는 함수입니다.

```cpp
#pragma GCC target("sse2")

#include <immintrin.h>
#include <iostream>
using namespace std;

alignas(16) int arr1[4] = {1,2,3,4};
alignas(16) int arr2[4] = {5,6,7,8};
alignas(16) int result1[4];
alignas(16) int result2[4];


void add_naive(){
  result1[0] = arr1[0] + arr2[0];
  result1[1] = arr1[1] + arr2[1];
  result1[2] = arr1[2] + arr2[2];
  result1[3] = arr1[3] + arr2[3];
}

void add_simd(){
  *(__m128i*)(result2) = _mm_add_epi32(*(__m128i*)(arr1), *(__m128i*)(arr2));
}

int main() {
  add_naive();
  for(int i = 0; i < 4; i++) cout << result1[i] << ' ';
  cout << '\n';
  add_simd();
  for(int i = 0; i < 4; i++) cout << result2[i] << ' ';
}
```

`add_naive` 함수는 덧셈을 4번 하는 방식으로 처리한 함수이고 `add_simd` 함수는 `_mm_add_epi32`를 사용한 함수입니다. `#pragma GCC target("sse2")`는 명령의 target을 지정하는 전처리 구문입니다. `-msse2` 컴파일 옵션을 줘도 됩니다 `_mm_add_epi32`가 `SSE2` 명령어 집합에 속한 함수이기 때문에 해당 구문이 필요합니다.

`immintrin.h`는 `AVX, SSE4_2 + SSE4_1 + SSSE3 + SSE3 + SSE2 + SSE + MMX`를 포함하고 있는 헤더파일입니다. 다른 헤더 파일 목록은 [여기](http://www.g-truc.net/post-0359.html)를 참고해보세요. 컴파일은 `g++ -o t t.cpp`로 진행했습니다.

`alignas(16)`는 선언하는 변수의 주소를 16바이트 단위로 align하는 명령입니다. 왜 align이 필요한지는 아래에서 `값 불러오기`에서 설명하도록 하겠습니다.

```cpp
#ifndef __SSE2__
#pragma GCC push_options
#pragma GCC target("sse2")
#define __DISABLE_SSE2__
#endif /* __SSE2__ */
```

먼저 `add_naive` 함수를 디스어셈블한 결과를 확인하면 아래와 같습니다.

```
0x0011a9 <+0>:     endbr64
0x0011ad <+4>:     push   rbp
0x0011ae <+5>:     mov    rbp,rsp
0x0011b1 <+8>:     mov    edx,DWORD PTR [rip+0x2e59]        # 0x4010 <arr1>
0x0011b7 <+14>:    mov    eax,DWORD PTR [rip+0x2e63]        # 0x4020 <arr2>
0x0011bd <+20>:    add    eax,edx
0x0011bf <+22>:    mov    DWORD PTR [rip+0x2f9b],eax        # 0x4160 <result1>
0x0011c5 <+28>:    mov    edx,DWORD PTR [rip+0x2e49]        # 0x4014 <arr1+4>
0x0011cb <+34>:    mov    eax,DWORD PTR [rip+0x2e53]        # 0x4024 <arr2+4>
0x0011d1 <+40>:    add    eax,edx
0x0011d3 <+42>:    mov    DWORD PTR [rip+0x2f8b],eax        # 0x4164 <result1+4>
0x0011d9 <+48>:    mov    edx,DWORD PTR [rip+0x2e39]        # 0x4018 <arr1+8>
0x0011df <+54>:    mov    eax,DWORD PTR [rip+0x2e43]        # 0x4028 <arr2+8>
0x0011e5 <+60>:    add    eax,edx
0x0011e7 <+62>:    mov    DWORD PTR [rip+0x2f7b],eax        # 0x4168 <result1+8>
0x0011ed <+68>:    mov    edx,DWORD PTR [rip+0x2e29]        # 0x401c <arr1+12>
0x0011f3 <+74>:    mov    eax,DWORD PTR [rip+0x2e33]        # 0x402c <arr2+12>
0x0011f9 <+80>:    add    eax,edx
0x0011fb <+82>:    mov    DWORD PTR [rip+0x2f6b],eax        # 0x416c <result1+12>
0x001201 <+88>:    nop
0x001202 <+89>:    pop    rbp
0x001203 <+90>:    ret
```

어셈블리에 친숙하지 않다면 굉장히 난해하게 보일 수 있겠지만 아무튼 (IceLake 기준) Latency 1, Throughput 0.25인 `add` 명령을 4번 수행하는걸 확인할 수 있습니다.

다음으로 `add_simd` 함수를 디스어셈블한 결과를 보겠습니다.

```
0x001204 <+0>:     endbr64
0x001208 <+4>:     push   rbp
0x001209 <+5>:     mov    rbp,rsp
0x00120c <+8>:     lea    rax,[rip+0x2e0d]        # 0x4020 <arr2>
0x001213 <+15>:    movdqa xmm0,XMMWORD PTR [rax]
0x001217 <+19>:    lea    rax,[rip+0x2df2]        # 0x4010 <arr1>
0x00121e <+26>:    movdqa xmm1,XMMWORD PTR [rax]
0x001222 <+30>:    lea    rax,[rip+0x2f47]        # 0x4170 <result2>
0x001229 <+37>:    movaps XMMWORD PTR [rbp-0x20],xmm1
0x00122d <+41>:    movaps XMMWORD PTR [rbp-0x10],xmm0
0x001231 <+45>:    movdqa xmm1,XMMWORD PTR [rbp-0x20]
0x001236 <+50>:    movdqa xmm0,XMMWORD PTR [rbp-0x10]
0x00123b <+55>:    paddd  xmm0,xmm1
0x00123f <+59>:    movaps XMMWORD PTR [rax],xmm0
0x001242 <+62>:    nop
0x001243 <+63>:    pop    rbp
0x001244 <+64>:    ret
```

여기서는 128비트 레지스터인 `xmm0`과 `xmm1`을 이용하고 (IceLake 기준) Latency 1, Throughput 0.5인 `paddd` 1번에 덧셈 계산을 완료합니다.

이와 같이 SIMD를 사용하면 더 적은 사이클을 소모해서 계산을 할 수 있습니다.

## C. SIMD 함수

위의 Intel Intrinsics 함수 목록을 확인해보시면 단순히 덧셈/뺄셈과 같은 연산을 최적화하는 것 뿐만 아니라 `AES, SHA, CRC32`와 같은 암호 모듈을 빠르게 계산해주는 명령도 있고 확률 분포와 관련한 계산을 해주는 명령도 있는 등 종류가 굉장히 많습니다. 성능이 매우 중요해서 로우 레벨 단위의 최적화가 필요하다면 상황에 맞는 적절한 Intrinsics 함수를 택해 사용을 해야겠지만 아쉽게도 이 모든 목록을 살펴보는 대신 범용적으로 자주 쓰일 함수들을 소개해드리겠습니다. (이번 `C. SIMD 함수`의 내용은 maomao90님의 [코드포스 블로그 글](https://codeforces.com/blog/entry/98594)을 참고했습니다.)

### 자료형

SIMD와 관련된 자료형은 `__m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i`가 있습니다.

`__m128, __m256, __m512`는 각각 32비트 실수를 4개, 8개, 16개 담고 있는 자료형입니다.

`__m128d, __m256d, __m512d`는 각각 64비트 실수를 2개, 4개, 8개 담고 있는 자료형입니다.

`__m128i, __m256i, __m512i`는 각각 정수를 담고 있는 자료형이고 정수는 8비트일 수도, 16비트일 수도, 32비트일 수도, 64비트일 수도 있습니다. 저장된 값을 몇 비트 단위로 생각할지는 순전히 프로그래머의 마음이고 생각하는 단위에 따라 적절한 함수를 사용해야 합니다.

한편으로 `__m512, __m512d, __m512i`는 CPU가 `AVX512`를 지원해야 사용할 수 있는데, 현재(2022년 2월) 기준으로 일반적으로 쓰이는 CPU에서는 사실상 거의 다 지원이 된다고 봐도 되는 `SSE2, AVX2` 등과 다르게 아직 `AVX512`는 지원하지 않는 CPU가 꽤 많습니다. 저의 컴퓨터 환경인 `Intel Core i7-10700F CPU @ 2.90GHz`에서도 지원이 되지 않았고 백준 온라인 저지의 채점 환경 `Intel Xeon E5-2666v3`에서도 지원이 되지 않았습니다.

### 함수 이름

Instinsic 함수의 함수 이름은 `_mm_{instruction}_{datatype}`(128비트 단위 연산)/`_mm256_{instruction}_{datatype}`(256비트 단위 연산)/`_mm512_{instruction}_{datatype}`(512비트 단위 연산) 이라는 형식을 가지고 있습니다. `instruction`에는 `add`, `max`, `sub`, `mul` 등의 명령 이름이 위치하고 `datatype`에는 `si128`(부호 있는 128비트 정수), `epi8, epi16, epi32, epi64`(부호 있는 8/16/32/64비트 정수), `epu8, epu16, epu32, epu64`(부호 없는 8/16/32/64비트 정수), `ps`(32비트 실수), `pd`(64비트 실수) 등이 있습니다.

앞에서 살펴본 `_mm_add_epi32`를 보면 함수 이름을 통해 해당 함수가 부호 있는 32비트 정수 4개를 더하는 연산임을 알 수 있습니다.

### 값 불러오기

`int, char` 등의 정수형 배열로부터 `_m256i`를 불러오기 위해서는 아래와 같이 `_mm256_load_si256` 함수를 사용하면 됩니다. `_mm256_load_epi32`, `_mm256_load_epi64` 라는 이름의 함수도 있지만 이들은 `AVX512` 명령어 집합에 포함된 함수입니다.

```cpp
alignas(32) int arr1[8] = {1,2,3,4,5,6,7,8};
__m256i a1 = _mm256_load_si256((__m256i *)arr1);
```

비록 `arr1`은 `int` 배열이지만 이를 `(__m256i *)`으로 형변환하는건 오류를 일으키지 않습니다. 또한 `__m256i`를 불러오기 위해서는 32-bytes로 align이 되어있어야 합니다. 만약 그렇지 않을 경우 오류가 발생할 수 있습니다.

만약 align이 되어 있지 않을 수 있는 곳에서 값을 가져와야 할 경우 `_mm256_loadu_si256` 함수를 활용하면 됩니다. 

`__mm256i` 대신 `__m256`이나 `__m256d`와 같은 값을 불러와야 할 경우에는 각각 `__mm256_load_ps`, `__mm256_load_pd`와 같은 함수를 이용할 수 있습니다.

(256비트 단위 연산을 사용할 수 있다면 굳이 128비트 단위 연산을 사용할 일이 딱히 없겠지만) `__mm128, __mm128i, __m128d` 등에 값을 넣고 싶다면 `_mm_load_si128`, `mm_load_ps`, `mm_load_pd`와 같은 함수를 이용하면 됩니다.

int 값들로부터 `__mm256i`를 만들고 싶다면 아래와 같이 `_mm256_setr_epi32` 함수를 이용해 만들면 됩니다.

```cpp
__m256i a1 = _mm256_setr_epi32(1,2,3,4,5,6,7,8);
```

`_mm256_set_epi32` 함수도 있는데, 해당 함수를 사용하면 제일 앞의 인자로 준 값이 나중에 배열로 변환했을 때 제일 뒤에 위치하게 됩니다.

### 값 저장하기

`__m256, __m256i, __m256d` 등의 값은 그대로 출력이 불가능합니다. 출력을 하고 싶다면 해당 값을 다시 배열로 옮겨담아야 합니다.

옮겨담을 때에는 위에서 본 함수 명에서 `load`만 `store`로 바꾼 `__mm256_store_si256` 등을 사용하면 됩니다. 마찬가지로 alignment의 상황에 따라 `store` 혹은 `storeu`를 사용하면 됩니다. 아래의 예시를 확인해보세요.

```cpp
#pragma GCC target("avx2")

#include <immintrin.h>
#include <iostream>
using namespace std;

alignas(32) int arr1[8] = {1,2,3,4,5,6,7,8};
alignas(32) int result[8] = {};

int main() {
	__m256i a1 = _mm256_load_si256((__m256i *)arr1);
	__m256i a2 = _mm256_add_epi32(a1, a1);

	_mm256_store_si256((__m256i *)result, a2);
	for(int i = 0; i < 8; i++)
		cout << result[i] << ' ';
}
```

이 때 주의할 점이 있습니다. 아래와 같이 `int` 배열을 `(__m256i *)`으로 형변환을 한 것 처럼 그냥 `__m256i` 자료형인 `a2`을 강제로 `int *`으로 읽어들여 값을 보면 안되냐는 생각을 할 수 있습니다.

```cpp
// !!!!!!!!!CAUTION!!!!!!!!!!
for(int i = 0; i < 8; i++)
  cout << *(((int *)&a2) + i) << ' ';
}
```

그러나 실제로 `g++ -o t t.cpp`으로 컴파일한 후 실행을 해보면 쓰레기 값이 출력됨을 확인할 수 있습니다. 이건 strict aliasing과 관련이 있는데, `-fstrict-aliasing` 옵션이 활성화되어 있으면 strict aliasing을 지켜야 하기 때문에 위와 같이 작성을 할 수 없습니다. 해당 옵션은 `O1` 이상의 최적화 레벨에서 활성화되어 있기 때문에([컴파일 옵션 정보](https://www.keil.com/support/man/docs/armclang_ref/armclang_ref_sam1465825195202.htm)) `O1` 이상의 최적화 레벨에서는 `-fno-strict-aliasing` 옵션을 통해 `strict aliasing`을 끄지 않았다면 `__m256i`를 int 배열로 간주해서는 안되고 `_mm256_store_@@` 혹은 `_mm256_storeu_@@` 함수를 통해 값을 읽어들여야 합니다. 반대로 말해 `-fno-strict-aliasing` 옵션을 줬거나 `O0` 최적화 레벨로 프로그램을 컴파일했다면 위의 코드가 정상적인 값을 출력하는 것을 확인할 수 있습니다. [관련 글](https://stackoverflow.com/questions/13257166/print-a-m128i-variable/46752535#46752535)

# 3. SIMD 실습

백문이 불여일견, 직접 실습을 해봅시다. 사용 예시를 몇 번 보다보면 금방 익숙해질 수 있습니다.

## 문제 1 - BOJ 10868

[BOJ 10868 - 최솟값](https://www.acmicpc.net/problem/10868) 해당 문제는 Segment Tree 혹은 Binary Lifting을 사용해 $O(MlgN)$에 해결 가능하지만 $O(MN)$ 나이브 풀이를 SIMD로 구현해보았습니다. BOJ가 사용하고 있는 `Intel Xeon E5-2666v3`는 2.60 GHz(1초에 $2.6 \times 10**9$개의 사이클을 계산)이고, 8개의 값을 병렬화해서 계산하니 시간복잡도는 $O(MN/8) = 1.25 \times 10^9$여서 사실 SIMD를 통해 통과가 가능할줄 알았지만 BOJ에서는 시간 초과가 발생했습니다 [코드 링크](http://boj.kr/1644b44aeadd42ec8e5fccfbffc64f41). 일단 코드포스 폴리곤에서 `N, M = 100,000, a = rnd.next(1,1000), b = rnd.next(99000,100000)`으로 선택한 20개의 데이터에 대해서 최대 919ms 내에 결과가 출력되고 또 여러 랜덤 데이터에 대해 정답이 나옴을 확인할 수 있었습니다. SIMD 최적화 없이 구현한 코드는 최대 6005ms가 소모되어 대략 6배 정도의 성능 차이가 남을 확인할 수 있었습니다.

다소 찝찝하지만 올바르게 구현했다고 믿고 코드를 살펴보면, 26-27번째 줄과 같이 우선 align을 맞춰주기 위해 8의 배수에 맞지 않는 값들은 별도로 처리해주고 `_mm256_min_epi32` 함수를 통해 최솟값을 계속 들고가는 방식으로 구현했습니다.

## 문제 2 - BOJ 9484

[BOJ 9484 - Triangles](https://www.acmicpc.net/problem/9484) 해당 문제는 스위핑을 이용해 $O(N^2lgN)$으로 풀이가 가능한 문제로 알고 있으나 언젠가 스위핑을 공부하다가 포기하고 그냥 $O(N^3)$으로 풀고 치웠던 문제입니다. 혹시 시간 제한이 줄거나 데이터 추가가 들어올까 마음 한켠에 찝찝함을 안고 살았는데 이번 기회에 SIMD로 최적화를 해보았습니다. [코드](http://boj.kr/64b032d635a347369ab2a928eb838910)를 확인해보세요.

바깥 for문의 `i, j`에 대해 `val = x[i]*y[j]-x[j]*y[i]`, `factorx = (y[i]-y[j])`, `factory = (x[j]-x[i])`로 두고난 후 `val + factorx * x[k] + factory * y[k]`를 계산하면 되기 때문에 안쪽 `k`에 대해 병렬화를 진행했습니다. `_mm256_mul_epi32`는 32비트 두 값을 곱해 64비트로 저장하는 함수이기 때문에 이를 쓰는 대신 `_mm256_mullo_epi32`를 사용해야 합니다. SIMD 최적화가 없을 때에는 4674ms가 걸린 반면([코드](http://boj.kr/658272375c3047e081e3b95c45775300)), 최적화를 적용한 결과 실행 시간을 896ms로 줄일 수 있습니다.

또 한편으로, `wisqa`님의 코드를 보다가 발견한 사실인데 굉장히 어처구니없게도 직접 SIMD를 짜느라 고생하지 않아도 나이브하게 짠 후 그냥 `#pragma GCC target("avx,avx2")` 옵션을 주면 알아서 최적화를 해줍니다 -_-([예시](http://boj.kr/d741aaac65f84da68617ab661e0370fe)).

# 4. 맺음말

이번 글에서는 SIMD에 대해 소개하고 SIMD를 사용해 나이브 풀이의 실행 시간을 개선해보았습니다. 사실 `SIMD 백준` 이라는 키워드로 검색해보면 이 글 이전에도 여러 소개글([링크 1](https://blog.naver.com/PostView.nhn?blogId=jinhan814&logNo=222322477829), [링크 2](https://justicehui.github.io/hard-algorithm/2021/11/15/simd-in-ps/)) 이 있고 글을 작성할 때 많은 도움을 받았습니다.

한편으로 alignment의 필요성, 컴파일 플래그, strict aligning 등과 같이 관성적으로 써왔지만 이론적으로 확신이 없었던 내용들을 최대한 꼼꼼하게 소개해드리고 싶었는데 목적을 제대로 달성했는지 모르겠습니다.

마지막으로 이번 글에서는 CPU cycle의 Latency와 Throughput의 차이를 제대로 설명드리지 못했습니다. 일례로 `_mm256_mullo_epi32` 함수는 Skylake에서 Latency가 10, Throughput이 0.66이기 때문에 병렬화에 신경을 아주 잘 쓴다면 20개의 `_mm256_mullo_epi32`이 정교하게 맞물려서 동시에 실행되도록 만들 수 있습니다. 이 부분은 아쉽지만 미완의 과제로 남겨두겠습니다.
