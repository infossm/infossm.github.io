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
    std::cout << "[+] ---- list test ----" << '\n';
    std::list<uint64_t> l;
    
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(5);
    l.push_back(4);
    
    std::cout << "[-] ---- list test ----" << '\n';
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- list test ----
[gmem] new(24) > 0x55718b336280
[gmem] new(24) > 0x55718b3362f0
[gmem] new(24) > 0x55718b336340
[gmem] new(24) > 0x55718b3362d0
[gmem] new(24) > 0x55718b336400
[-] ---- list test ----
[gmem] delete(0x55718b336280)
[gmem] delete(0x55718b3362f0)
[gmem] delete(0x55718b336340)
[gmem] delete(0x55718b3362d0)
[gmem] delete(0x55718b336400)
```

`gmem`이 할당된 메모리를 어떻게 로깅하는지 알 수 있습니다. `new`를 통해
24바이트의 메모리가 5번 할당되었고, 함수 호출이 끝난 직후 `list`의 원소를
저장하기 위해 할당된 메모리가 순차적으로 반환되는 것을 알 수 있습니다.

`iterator`가 앞뒤로만 이동할 수 있는 `list` 특성상,
할당되는 메모리는 연속되어 있지 않습니다.
때문에 바로 뒤에 나올 `vector`와 성질이 상이할 수밖에 없습니다.

## `vector` 컨테이너

## `priority_queue` 컨테이너

## `set` 컨테이너

## `map` 컨테이너

## `unordered_map` 컨테이너

