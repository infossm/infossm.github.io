---
layout: post
title:  "C++ STL 컨테이너의 메모리 사용량 (1)"
date:   2019-11-14 19:00
author: evenharder
image:  /assets/images/evenharder-post/c++-container/pexel-pok-rie-1655166.jpg
tags:   C++ data-structure
---

C++로 Problem Solving을 하는 사람도, 일반적인 프로그래밍을 하는 사람도
C++의 컨테이너는 매우 유용하게 사용합니다. 대표적인 예시가 동적 배열인 `vector`,
이진 탐색 트리인 `set` 등입니다. 이러한 자료구조를 적재적소에 사용하지 않으면
시간/공간 복잡도가 예상을 뛰어넘을 수 있습니다. 대표적인 예시가
"`priority_queue`가 `set`보다 빠르다"는 말입니다. 이는 일반적으로 사실이지만,
왜 그럴까요? 이와 관련된 질문에 답하기 위해 C++ 컨테이너가
메모리를 원소별로 얼마나 할당하고 그 이유는 무엇일지 `gmem`이라는 라이브러리를 통해
파트별로 알아보도록 하겠습니다.

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
`list`는 doubly-linked list 형태로 구현되어 있습니다. 즉, 각 원소마다
(내부적으로) 원소의 값, 이전 원소를 가리키는 포인터,
다음 원소를 가리키는 포인터를 가지고 있습니다.
때문에 `std::list<uint64_t>` 형 `list`에 원소를 삽입하면
내부적으로 이 원소를 저장하기 위한 **24바이트**의 메모리를 동적으로 할당합니다.

이를 확인하기 위한 `list_test` 함수를 작성해보았습니다. `list` 특성상,
임의의 위치에 대한 삽입이 자유롭습니다.

```c++
void list_test()
{
    std::cout << "\n[+] ---- list test ----\n";
    std::list<uint64_t> l;
    std::list<uint64_t>::iterator it = l.begin();
    
    for(int i=1;i<=2;i++) {
        std::cout << "[*] list push_back " << i << '\n';
        l.push_back(i);
    }
    
    for(int i=3;i<=4;i++) {
        std::cout << "[*] list push_front " << i << '\n';
        l.push_front(i);
    }
    
    for(int i=5;i<=6;i++) {
        std::cout << "[*] list insert " << i << '\n';
        l.insert(it, i);
    }
    
    for(int i=1;i<=3;i++) {
        std::cout << "[*] list pop_front " << i << '\n';
        l.pop_front();
    }
    
    std::cout << "\n[-] ---- list test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- list test ----
[*] list push_back 1
[gmem] new(24) > 0x55d3a02d3280
[*] list push_back 2
[gmem] new(24) > 0x55d3a02d32f0
[*] list push_front 3
[gmem] new(24) > 0x55d3a02d3340
[*] list push_front 4
[gmem] new(24) > 0x55d3a02d32d0
[*] list insert 5
[gmem] new(24) > 0x55d3a02d3400
[*] list insert 6
[gmem] new(24) > 0x55d3a02d3450
[*] list pop_front 1
[gmem] delete(0x55d3a02d32d0)
[*] list pop_front 2
[gmem] delete(0x55d3a02d3340)
[*] list pop_front 3
[gmem] delete(0x55d3a02d3280)

[-] ---- list test ----
[gmem] delete(0x55d3a02d32f0)
[gmem] delete(0x55d3a02d3400)
[gmem] delete(0x55d3a02d3450)
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

    for(int i=1;i<=9;i++) {
        std::cout << "[*] vector push_back " << i << '\n';
        v.push_back(i);
    }
    
    std::cout << "\n[-] ---- vector test ----\n";
}
```
`main` 함수에서 `vector_test`를 호출한 결과는 다음과 같습니다.
```
[+] ---- vector test ----
[*] vector push_back 1
[gmem] new(8) > 0x55fc59b54280
[*] vector push_back 2
[gmem] new(16) > 0x55fc59b542f0
[gmem] delete(0x55fc59b54280)
[*] vector push_back 3
[gmem] new(32) > 0x55fc59b542a0
[gmem] delete(0x55fc59b542f0)
[*] vector push_back 4
[*] vector push_back 5
[gmem] new(64) > 0x55fc59b54370
[gmem] delete(0x55fc59b542a0)
[*] vector push_back 6
[*] vector push_back 7
[*] vector push_back 8
[*] vector push_back 9
[gmem] new(128) > 0x55fc59b543c0
[gmem] delete(0x55fc59b54370)

[-] ---- vector test ----
[gmem] delete(0x55fc59b543c0)
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

## `deque` 컨테이너
`deque`는 맨 앞과 맨 뒤에서만 삽입/삭제가 가능한 자료구조이자, C++에서 
`stack`과 `queue`의 기본 구현체이기도 합니다. 맨 앞의 원소를 삭제하는 시간 복잡도가
원소의 개수에 비례하는 `vector`와는 달리 `deque`은 앞도 자유자재로 뺄 수 있으며,
random access 또한 가능합니다. 대체 무슨 원리로 이게 가능한 걸까요?

`deque`은 `vector`와는 달리 모든 데이터가 하나의 연속된 메모리에
올라가 있지 않습니다. 몇 바이트 단위의 chunk로 쪼개져있으며,
이 chunk 또한 내부적으로 관리가 됩니다. Problem solving에 익숙하다면
bucket을 생각하시면 될 것 같습니다. 구현체에 따라 다르겠으나 이 chunk는
512바이트에 가깝게 설정되는 것으로 보이고, 내부적으로 배열을 이용해
random access를 구현하는 것 같습니다. 즉, `T`형 `deque`는 `T**` 형 변수로
조직된다고 생각할 수 있습니다. 어떻게 동작하는지를 보기 위해
`stack`과 `queue`를 이용해보도록 하겠습니다.

### 512바이트 자료형 `deque`에 `push_back`
스택이 First-in-last-out의 자료구조은 잘 알고 계시리라고 생각합니다. `deque`에서
뒷쪽 삽입/삭제만 있는 버전인 `stack`에 512 byte 자료를 36개 넣어보겠습니다. 다음은
이를 시행하는 함수인 `stack_512_test`입니다.

```c++
struct data{
    uint64_t a[64]; // 512 bytes
};
void stack_512_test()
{
    std::cout << "\n[+] ---- stack test ----\n";
    std::stack<data> st;
    
    for(int i=1;i<=36;i++) {
        std::cout << "[*] stack insert " << i << '\n';
        st.push(data());
    }
    
    for(int i=1;i<=36;i++) {
        std::cout << "[*] stack delete " << i << '\n';
        st.pop();
    }
    std::cout << "\n[-] ---- stack test ----\n";
}
```
이를 `main`에서 호출하면 로그가 길어져서 [별도 링크](https://gist.github.com/evenharder/5885e28e2abb8e30ee29b3f6ad97cb05#file-stack_512_test-txt)로 첨부하도록 하겠습니다.
알 수 있는 점을 보자면 다음과 같습니다.

+ 스택이 생성되면서 **64바이트 메모리**와 **512바이트 메모리**가 동적할당됩니다.
후자는 다음에 들어올 512바이트 원소를 저장할 공간이라도 쳐도, 전자는 무엇일까요?
+ 매 512 바이트짜리 원소를 삽입한 후, 다음을 위한 512바이트를 새롭게 할당합니다.
지금은 자료형이 512바이트이기 때문에, 쓰면 바로 또 재할당을 해야 해서 그렇습니다.
+ 5번째 삽입 과정에서 64바이트 메모리가 해제되고 **144바이트 메모리**가 새롭게
할당됩니다.
+ 12번째 삽입 과정에서 144바이트 메모리가 해제되고 **304바이트 메모리**가
새롭게 할당됩니다.
+ 26번째 삽입 과정에서 304바이트 메모리가 해제되고 **624바이트 메모리**가 새롭게
할당됩니다.
+ 삭제 과정은 나중에 생성된 512바이트 메모리들이 먼저 삭제됩니다.
+ 함수가 종료될 때, 가장 처음으로 생성되었던 512바이트 메모리와 26번째 삽입 과정에서
생긴 624바이트 메모리가 해제됩니다.

약간의 수식을 동원해봅시다. 64/8 = 8이고, 144/8 = 18, 304/8 = 38,
624/8 = 78입니다. 재할당이 일어나기 전 삽입 횟수와 비교하면 (4, 8), (11, 18), 
(25, 38)로 2/3와 비슷함을 알 수 있습니다. 여기서 든 생각이,
'재할당을 한 다음에 기존 chunk를 새로운 배열의 가운데에 복사하고,
양쪽으로 여유 공간을 어느 정도 할당하지 않을까'였습니다.

### 512바이트 자료형 `deque`에 `push_back` + `push_front`
`deque`으로 건너와서, 
`push_front`까지 진행하는 테스트 코드를 작성해보도록 하겠습니다.

```c++
void deque_512_push_test()
{
    std::cout << "\n[+] ---- deque push test ----\n";
    std::deque<data> dq;
    
    for(int i=1;i<=40;i++) {
        if(i % 2 == 1) {
            std::cout << "[*] deque push_back " << i << '\n';
            dq.push_back(data());
        }
        else {
            std::cout << "[*] deque push_front " << i << '\n';
            dq.push_front(data());
        }
    }

    std::cout << "\n[-] ---- deque push test ----\n";
}
```
`deque_512_push_test`를 호출해서 실행해본 결과, 위의 가설이 어느 정도 맞다는
것을 확인할 수 있었습니다. 로그는 [여기에](https://gist.github.com/evenharder/5885e28e2abb8e30ee29b3f6ad97cb05#file-deque_512_push_test-txt) 있습니다.

+ 데크가 생성되면서 **64바이트 메모리**와 **512바이트 메모리**가 동적할당됩니다.
+ 매 512 바이트짜리 원소를 삽입한 후, 다음을 위한 512바이트를 새롭게 할당합니다.
+ `i == 8`일 때 `push_front`에서 64바이트 메모리가 해제되고
**144바이트 메모리**가 새롭게 할당됩니다.
+ `i == 18`일 때 `push_front`에서 144바이트 메모리가 해제되고
**304바이트 메모리**가 새롭게 할당됩니다.
+ `i == 38`일 때 `push_front`에서 304바이트 메모리가 해제되고
**624바이트 메모리**가 새롭게 할당됩니다.

중요한 점은 재할당이 일어나는 index가 이전에 할당되었던 원소의 개수와
일치한다는 사실입니다.

`push_front`를 먼저 진행하는 `deque_512_push_test2` 함수도 만들어보았습니다.

```c++
void deque_512_push_test2()
{
    std::cout << "\n[+] ---- deque push test ----\n";
    std::deque<data> dq;
    
    for(int i=1;i<=40;i++) {
        if(i % 2 == 0) {
            std::cout << "[*] deque push_back " << i << '\n';
            dq.push_back(data());
        }
        else {
            std::cout << "[*] deque push_front " << i << '\n';
            dq.push_front(data());
        }
    }

    std::cout << "\n[-] ---- deque push test ----\n";
}
```

실행 결과는 `deque_512_push_test`와 매우 흡사하였으나,
재할당이 이루어지는 index가 (8, 18, 38)에서 (7, 18, 37)로 변경되었습니다. 초기
시작 위치 차이로 인해 +1/-1 정도의 차이가 나는 것 같습니다. [여기서](https://gist.github.com/evenharder/5885e28e2abb8e30ee29b3f6ad97cb05#file-deque_512_push_test2-txt)
확인할 수 있습니다.

한 쪽으로만 삽입을 진행하던 (5, 12, 26)와 chunk를 저장하는 배열의 길이
(8, 18, 38), 그리고 교대로 삽입하면 이 배열에 딱 맞게 저장할 수 있다는 점을
종합하면 다음과 `deque`의 동작방식을 다음과 같이 추론해볼 수 있습니다.

+ 처음에 chunk 8개를 보관할 배열과, 첫 번째 chunk를 동적할당합니다.
+ 삽입을 해나갈 때는 가운데에서부터 양옆으로 퍼져나갑니다.
+ 퍼져나가다가 배열의 끝에 도달하게 되면, 현재 capacity에 1을 더하고 2배를
한 크기만큼 **재할당**을 하고, 기존 배열을 새 배열의
**가운데에 복사**하고 반환합니다.
+ 이 과정을 반복합니다.

첫 번째 테스트 결과를 여기에 적용해보면 다음과 같습니다. (분할에서
~1칸 정도의 오차는 있을 수 있습니다.)
+ 처음에 길이가 8이고, 가운데부터 `push_back`을 하면,
다섯 번째에서 재할당이 이루어집니다. 그 이후 12번째에서 재할당이 이루어지므로
오른쪽으로 7칸의 여유 공간이 있음을 짐작할 수 있습니다. 즉 재할당 과정에서
7 : 5 : 6로 좌측의 여유 공간, 점유된 공간, 우측의 여유 공간이 분할된다고
생각할 수 있습니다.
+ 이후 12번째 삽입에서 재할당이 일어날 때는 총 길이 12, 다음 최대 길이 38으로
변경되므로 13 : 12 : 13 으로 분할된다고 볼 수 있습니다.
+ 26번째 삽입에서는 총 길이 26, 최대 길이 78이므로 26 : 26 : 26으로
분할될 것입니다.

비율이 2/3로 나온 것도, 수학적으로 재할당시 좌측 공간 : 현재 공간 : 우측 공간이
균등분할되어서 그런 것 같습니다. 물론 이 경우엔 `deque`의 효율이 상당히 낮을
것입니다.

반면, 두 번째 테스트를 적용해보면 다음과 같습니다.
+ 8번째 삽입 전까지 8번 동적할당이 되므로, 배열이 꽉 차 있는 상태입니다.
+ 8번째 삽입에서 최대 길이 18로 할당되며 5 : 8 : 5로 분할됩니다.
+ 18번째 삽입 전까지는 위의 공간을 알뜰하게 채워나가고, 18번째에서는
최대 길이 38로 할당되며 10 : 18 : 10으로 분할됩니다.

이론이 얼추 맞을 것 같다는 생각은 들지만, `deque`을 생성하면서 동적 할당되는
첫 번째 chunk가 어디에 들어가 있고, `push_back`과 `push_front`에서 아직 꽉
차있지 않은 (양 끝의) chunk가 어떻게 관리되는지는 잘 모르겠습니다.

### 72바이트 자료형 `deque`에 `push_back` + `push_front`
지금까지의 예제는 chunk 내부의 크기가 1이였지만, 그렇지 않은 상황을 적용해보도록
하겠습니다. 72바이트 자료형의 `deque`에 200개를 넣되, 4의 배수번째일 때만
`push_front`를 하는 코드입니다.

```c++
struct data_9 {
    uint64_t a[9];
};
void deque_72_push_test()
{
    std::cout << "\n[+] ---- deque push test ----\n";
    std::deque<data_9> dq;
    
    for(int i=1;i<=200;i++) {
        if(i % 4) {
            std::cout << "[*] deque push_back " << i << '\n';
            dq.push_back(data_9());
        }
        else {
            std::cout << "[*] deque push_front " << i << '\n';
            dq.push_front(data_9());
        }
    }

    std::cout << "\n[-] ---- deque push test ----\n";
}
```
실행해본 결과 이전 실험에서는 알 수 없던 내용을 알 수 있었습니다. 로그는
[여기서](https://gist.github.com/evenharder/5885e28e2abb8e30ee29b3f6ad97cb05#file-deque_72_push_test-txt) 확인할 수 있습니다.

+ 한 chunk의 크기는 504바이트로, `data_9`형 변수가 7개 들어갈 수 있습니다.
512바이트에 맞추어 잡히는 것을 알 수 있습니다.
+ 처음에 `push_back`을 할 때는 chunk의 처음부터 들어갑니다.
+ 이후 4번째 삽입에서 `push_front`를 할 때는 chunk를 새로 하나 할당받습니다.
여기서는 chunk의 뒤부터 들어갑니다.
+ 35번째 `push_back`에서 크기 18로 재할당을 합니다. 이를 통해
    - 초기 상태처럼, 한 칸의 여유 공간이 필요하여 36번째가 아닌 35번째에
    재할당이 일어남을 알 수 있습니다.
    - 다만 왜 28번째가 아닌 35번째인지는 알 수 없었습니다. 보통 `push_back`을
    더 많이 할 것이라는 믿음에 근거해, 첫 chunk가 처음 크기 8의 배열에서
    4번째 위치에 있지 않나 추측할 뿐입니다.

이후 77번째 `push_back`에서 크기 38로 재할당을 하는 것 외에는 특기할 만한 사항은
없었습니다.

## 고찰
이번 포스트에서는 `gmem` 라이브러리를 통해 `list`와, `vector`, `deque`가
어떻게 크기를 키워나가고, 메모리를 관리하는지 알 수 있었습니다.
`deque`의 구현 방법은 'vector의 vector' 말고도 chunking array를
circular하게 저장하는 방법도 있어 보입니다.

`vector`나 `deque`, 그리고 `deque`를 통해 구현되는 `stack`과 `queue`는
재할당 과정이 종종 일어나게 됩니다. 이런 내막을 알고 나면 `vector`의 `reserve`의
의미나 "`stack`은 느려서 잘 안 쓰게 돼요"나 "재귀 함수에서 `vector`를 매번
생성하는 건 피해야 합니다"라는 말을 이해할 수 있게 됩니다.
비단 Problem solving뿐만 아니라, 이론적인 추상 자료형(abstract data type)의
구현이 어떻게 되고 어떤 기법들이 사용되는지 파악하는 것은 의미가 있으리라
생각합니다.

다음 포스트에서는 `priority_queue`, `set`, `map`, `unordered_map`을 다루도록
하겠습니다. 스포일러를 하자면 `priority_queue`는 `set`에 비해 많이 가볍습니다.


[//]: # ## `priority_queue` 컨테이너

[//]: # %## `set` 컨테이너

[//]: # %## `map` 컨테이너

[//]: # %## `unordered_map` 컨테이너

