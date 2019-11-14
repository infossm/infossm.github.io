---
layout: post
title:  "C++ 컨테이너의 메모리 사용량"
date:   2019-11-11 19:00
author: evenharder
image:  /assets/images/evenharder-post/c++-container/pexel-pok-rie-1655166.jpg
tags:   c++ data-structure
---

C++로 Problem Solving을 하는 사람도, 일반적인 프로그래밍을 하는 사람도
C++의 컨테이너는 매우 유용하게 사용합니다. 대표적인 예시가 동적 배열인 `vector`,
이진 탐색 트리인 `set` 등입니다. 이러한 자료구조를 적재적소에 사용하지 않으면
시간/공간 복잡도가 예상을 뛰어넘을 수 있습니다. 대표적인 예시가
"`priority_queue`가 `set`보다 빠르다"는 말입니다. 이는 일반적으로 사실이지만,
왜 그럴까요? 오늘은 이와 관련된 질문에 답하기 위해 C++ 컨테이너가
메모리를 원소별로 얼마나 할당하고 그 이유는 무엇일지 `gmem`이라는 라이브러리를 통해
알아보도록 하겠습니다.

## `gmem` 라이브러리
`gmem`은 동적으로 메모리를 할당할 때마다 로그를 남기고,
프로그램이 종료될 때 반환하지 않은 메모리가 있으면 경고를 해주는 라이브러리입니다.
[여기](https://github.com/snoopspy/gmem)에서 다운로드받을 수 있으며, Qt 프로젝트
기반으로 구현되었으나 Qt 없이도 사용할 수 있습니다.
이 포스트에서는 Qt 없이 `g++ 7.4.0`에서 바로 시험해보자 합니다.

### `gmem` 사용법
`gmem`을 사용하기 위해 `gmem_test.cpp`를 다음과 같이 작성해봅시다.
```c++
#include <cstdint>  // uint64_t, 8-byte type
#include <iostream> // IO
#include <list>     // std::list
#include <map>      // std::map
#include <queue>    // std::queue
#include <stack>    // std::stack
#include <set>      // std::set
#include <vector>   // std::vector
#include "gmem.h"
int main()
{
    gmem_set_verbose(true);
    return 0;
}
```
컴파일을 하기 위해 `/src/mem`에 있는 모든 `.h` 파일과 `.cpp` 파일을
동일 디렉토리에 놓고, 다음과 같이 컴파일합니다.
```bash
g++ -std=c++14 -o gmem_test gmem_test.cpp \
    gmemfunc.cpp gmemhook.cpp gmemmgr.cpp -ldl
```

## `list` 컨테이너
`list`는 doubly-linked list 형태로 구현되어 있습니다. 즉, 각 원소마다 (내부적으로)
원소의 값, 이전 원소를 가리키는 포인터, 다음 원소를 가리키는 포인터를
가지고 있습니다. 때문에 `std::list<uint64_t>` 형 `list`에 원소를 삽입하면
내부적으로 이 원소를 저장하기 위한 **24바이트**의 메모리를 동적으로 할당합니다.

이를 확인하기 위한 `list_test` 함수를 작성해보았습니다.

```c++
void list_test()
{
    std::cout << "\n[+] ---- list test ----\n";
    std::list<uint64_t> l;
    
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(5);
    l.push_back(4);
    
    std::cout << "\n[-] ---- list test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- list test ----
[gmem] new(24) > 0x562554a7f280
[gmem] new(24) > 0x562554a7f2f0
[gmem] new(24) > 0x562554a7f340
[gmem] new(24) > 0x562554a7f2d0
[gmem] new(24) > 0x562554a7f400

[-] ---- list test ----
[gmem] delete(0x562554a7f280)
[gmem] delete(0x562554a7f2f0)
[gmem] delete(0x562554a7f340)
[gmem] delete(0x562554a7f2d0)
[gmem] delete(0x562554a7f400)
```

`gmem`이 할당된 메모리를 어떻게 로깅하는지 알 수 있습니다. `new`를 통해
24바이트의 메모리가 5번 할당되었고, 함수 호출이 끝난 직후 `list`의 원소를
저장하기 위해 할당된 메모리가 순차적으로 반환되는 것을 알 수 있습니다.

`iterator`가 앞뒤로만 이동할 수 있는 `list` 특성상,
할당되는 메모리는 연속되어 있지 않습니다.
때문에 바로 뒤에 나올 `vector`와 성질이 상이할 수밖에 없습니다.

## `vector` 컨테이너
`vector`는 흔히 동적 배열로 알려져 있습니다. `list`와 가장 큰 차이라면
임의의 원소를 접근할 수 있다는 점입니다. 이는 메모리 관점에서
'연속된 공간을 차지해야 한다'는 뜻인데, 항상 한 칸씩 늘려나갈 수 있는 것이 아니므로
정해진 크기 이상으로 메모리를 차지해야 하면 재할당을 진행합니다.

재할당 전략에는 다양한 방법이 있겠지만 공통적으로 삽입이
amortized constant time complexity를 가져야 합니다.
즉, 평균적으로 삽입이 $O(1)$이여야 합니다.
대표적인 방법이 최대 크기를 2배씩 키워나가는 방법입니다.
원소 0개가 할당된 상태에서 삽입이 필요하면
1개의 원소를 담을 수 있는 메모리를 할당하고,
그 이후로 삽입하면서 추가적인 메모리가 필요하면 기존 크기의 2배의 메모리를
재할당하는 방식입니다. 현재 할당한 메모리에 최대로 들어갈 수 있는
원소의 개수를 `capacity`, 현재 삽입된 원소의 개수를 `size`라고 합니다.

`g++`도 이 방식을 택했습니다. `vector_test` 함수를 살펴보겠습니다.
```c++
void vector_test()
{
    std::cout << "\n[+] ---- vector test ----\n";
    std::vector<uint64_t> v;

    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    v.push_back(5);
    v.push_back(9);
    v.push_back(8);
    v.push_back(7);
    v.push_back(6);
    
    std::cout << "\n[-] ---- vector test ----\n";
}
```
`main` 함수에서 `vector_test`를 호출한 결과는 다음과 같습니다.
```
[+] ---- vector test ----
[gmem] new(8) > 0x562554a7f400
[gmem] new(16) > 0x562554a7f2d0
[gmem] delete(0x562554a7f400)
[gmem] new(32) > 0x562554a7f420
[gmem] delete(0x562554a7f2d0)
[gmem] new(64) > 0x562554a7f450
[gmem] delete(0x562554a7f420)
[gmem] new(128) > 0x562554a7f4a0
[gmem] delete(0x562554a7f450)

[-] ---- vector test ----
[gmem] delete(0x562554a7f4a0)
```

일련의 과정을 요약하면 다음과 같습니다.
+ 첫 번째 삽입에서 현재 `size`는 0, `capacity`도 0입니다.
때문에 크기 **1(8바이트)**의 메모리를 할당하고 삽입을 진행합니다.
+ 두 번째 삽입에서 현재 `size`는 1, `capacity`는 1입니다.
때문에 크기 **2(16바이트)**의 메모리를 재할당하고, 값을 복사한 다음, 삽입을 합니다.
기존의 크기 1(8바이트) 메모리는 반환됩니다.
+ 세 번째 삽입에서 현재 `size`는 2, `capacity`는 2입니다.
때문에 크기 **4(32바이트)**의 메모리를 재할당하고, 값을 복사한 다음, 삽입을 합니다.
기존의 크기 2(16바이트)의 메모리는 반환됩니다.
+ 네 번째 삽입에서 현재 `size`는 3, `capacity`는 4입니다.
여유 공간이 있으므로 삽입을 진행합니다.
+ 다섯 번째 삽입에서 현재 `size`는 4, `capacity`는 4입니다.
때문에 크기 **8(64바이트)**의 메모리를 재할당하고, 값을 복사한 다음, 삽입을 합니다.
기존의 크기 4(32바이트)의 메모리는 반환됩니다.
+ 여섯, 일곱, 여덟번째 삽입까지는 재할당이 이루어지지 않고, 아홉 번째 삽입에서는
크기 **16(128바이트)**의 메모리를 재할당하고 동일한 절차가 진행됩니다.
+ 마지막으로 변수가 반환될 때는 128바이트 메모리를 한 번에 반환합니다.

`capacity`는 `reserve` 함수를 통해 설정할 수 있습니다. 삽입 횟수에 상한이 있다면
미리 `reserve`를 통해 메모리를 점유하여 필요없는 재할당을 막을 수 있습니다.
`reserve`를 통해 설정된 capacity도 재할당 시에는 그 두 배로 증가합니다.

## `priority_queue` 컨테이너

## `set` 컨테이너

## `map` 컨테이너

## `unordered_map` 컨테이너

