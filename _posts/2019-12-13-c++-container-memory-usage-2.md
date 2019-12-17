---
layout: post
title:  "C++ STL 컨테이너의 메모리 사용량 (2)"
date:   2019-12-13 23:00
author: evenharder
image:  /assets/images/evenharder-post/c++-container/pexel-nick-collins-1293120.jpg
tags:   c++ data-structure

---

지난 포스트 [C++ STL 컨테이너의 메모리 사용량 (1)](http://www.secmem.org/blog/2019/11/14/c++-container-memory-usage/)에서는 `list`, `vector`, `deque`의 내부 메모리 사용량을 분석하고, 어떤 식으로 구현되어있는지 추측해보았습니다. 이번 시간에는  `priority_queue`, `set`, `map`, `unordered_map`을 다루도록 하겠습니다.



## `priority_queue` 컨테이너

`priority_queue` 컨테이너는 다음과 같은 형태로 정의됩니다.

```c++
template<
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type>
> class priority_queue;
```

`T`는 타입, `Container`는 내부적으로 사용할 컨테이너, `Compare`는 heap을 구축하는데 사용할 함수를 의미합니다. 기본적으로 `priority_queue`는 `std::vector<T>`를 이용하고, `std::less`를 통해 max heap을 만듭니다.

`priority_queue`는 heapify 등을 통해서 heap을 만들기 때문에, in-place로 heap을 구성합니다. 때문에 메모리 할당도 `Container` 방식을 따라갑니다.

확인을 위해 기본 생성자를 사용하는 `pq_test` 함수를 작성해보았습니다.

```c++
void pq_test()
{
    std::cout << "\n[+] ---- pq test ----\n";
    std::priority_queue<uint64_t> pq;
    for(int i=0;i<10;i++)
    {
        uint64_t x = (10 * i + 3) % 17;
        std::cout << "[*] pq push " << x << '\n';
        pq.push(x);
    }
            
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] pq pop\n";
        pq.pop();
    }
    std::cout << "\n[-] ---- pq test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- pq test ----
[*] pq push 3
[gmem] new(8) > 0x55adcc4cbe80
[*] pq push 13
[gmem] new(16) > 0x55adcc4cbef0
[gmem] delete(0x55adcc4cbe80)
[*] pq push 6
[gmem] new(32) > 0x55adcc4cbea0
[gmem] delete(0x55adcc4cbef0)
[*] pq push 16
[*] pq push 9
[gmem] new(64) > 0x55adcc4cbf70
[gmem] delete(0x55adcc4cbea0)
[*] pq push 2
[*] pq push 12
[*] pq push 5
[*] pq push 15
[gmem] new(128) > 0x55adcc4cbfc0
[gmem] delete(0x55adcc4cbf70)
[*] pq push 8
[*] pq pop
[*] pq pop
[*] pq pop
[*] pq pop
[*] pq pop

[-] ---- pq test ----
[gmem] delete(0x55adcc4cbfc0)
```

`vector`의 메모리 관리 방식을 그대로 따라가는 것을 알 수 있습니다.

`std::deque`도 `priority_queue`에 사용할 수 있는 컨테이너의 조건을 만족하므로, 다음과 같은 함수를 만들어 시험해볼 수 있습니다.

```c++
void pq2_test()
{
    std::cout << "\n[+] ---- pq2 test ----\n";
    std::priority_queue<uint64_t, std::deque<uint64_t>> pq2;
    for(int i=0;i<10;i++)
    {
        uint64_t x = (10 * i + 3) % 17;
        std::cout << "[*] pq2 push " << x << '\n';
        pq2.push(x);
    }
    
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] pq2 pop\n";
        pq2.pop();
    }  
    std::cout << "\n[-] ---- pq2 test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- pq2 test ----
[gmem] new(64) > 0x55adcc4cbf70
[gmem] new(512) > 0x55adcc4cc050
[*] pq2 push 3
[*] pq2 push 13
[*] pq2 push 6
[*] pq2 push 16
[*] pq2 push 9
[*] pq2 push 2
[*] pq2 push 12
[*] pq2 push 5
[*] pq2 push 15
[*] pq2 push 8
[*] pq2 pop
[*] pq2 pop
[*] pq2 pop
[*] pq2 pop
[*] pq2 pop

[-] ---- pq2 test ----
[gmem] delete(0x55adcc4cc050)
[gmem] delete(0x55adcc4cbf70)
```

`std::deque`의 형식을 따라가는 것을 알 수 있습니다. 여기에 삽입 $O(\lg n)$, `top`을 보는 시간복잡도 $O(1)$이므로 최솟값만 보고자 하면 매우 효율적인 자료구조임을 알 수 있습니다.



## `set` 컨테이너

`set` 컨테이너의 꼴은 다음과 같습니다.

```c++
template < class T,                        // set::key_type/value_type
           class Compare = less<T>,        // set::key_compare/value_compare
           class Alloc = allocator<T>      // set::allocator_type
           > class set;
```

클래스, 비교 연산자, 그리고 할당자로 구성되어 있습니다.

`set` 컨테이너는 기본적으로 balanced binary search tree(BBST)의 조건에 부합합니다. C++ 표준에서 `set`에 특정 BBST 자료구조를 쓰라고는 명시가 되어 있지 않지만, 일반적으로 Red-black tree를 사용합니다.

`set` 컨테이너가 할 수 있는 역할을 생각하면 다음과 같습니다.

+ 순회. `set`의 `iterator`는 양방향으로 움직일 수 있습니다.
+ 정렬. $n$개의 원소가 들어있는 `set`에서 임의의 원소 검색은 $O(\lg n)$만에 할 수 있습니다 (BBST의 특징이기도 합니다).
+ 삽입 및 삭제. 역시 $O(\lg n)$ 만에 할 수 있습니다.

때문에 한 노드가 들고있는 자료가 좀 많습니다. 위의 설명을 통해 다음과 같은 원소들이 있다고 추측해볼 수 있습니다.

+ 현재 Node의 값(key)
+ 왼쪽 자식 노드를 가리키는 포인터
+ 오른쪽 자식 노드를 가리키는 포인터
+ 정렬 순서상 이전 노드를 가리키는 포인터
+ 정렬 순서상 다음 노드를 가리키는 포인터

트리의 형태이기 때문에, 원소가 연속적인 메모리 공간에 있지 않아도 됨을 추론해볼 수 있습니다.`set_test` 함수를 통해 알아보도록 하겠습니다.

```c++
void set_test()
{
    std::cout << "\n[+] ---- set test ----\n";
    std::set<uint64_t> s;
    for(int i=0;i<10;i++)
    {
        uint64_t x = (10 * i + 3) % 17;
        std::cout << "[*] set insert " << x << '\n';
        s.insert(x);
    }
    
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] set erase\n";
        s.erase(s.begin());
    }     
    std::cout << "\n[-] ---- set test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- set test ----
[*] set insert 3
[gmem] new(40) > 0x55adcc4cbea0
[*] set insert 13
[gmem] new(40) > 0x55adcc4cbf40
[*] set insert 6
[gmem] new(40) > 0x55adcc4cc290
[*] set insert 16
[gmem] new(40) > 0x55adcc4cc330
[*] set insert 9
[gmem] new(40) > 0x55adcc4cc390
[*] set insert 2
[gmem] new(40) > 0x55adcc4cc3f0
[*] set insert 12
[gmem] new(40) > 0x55adcc4cc450
[*] set insert 5
[gmem] new(40) > 0x55adcc4cc4b0
[*] set insert 15
[gmem] new(40) > 0x55adcc4cc510
[*] set insert 8
[gmem] new(40) > 0x55adcc4cc570
[*] set erase
[gmem] delete(0x55adcc4cc3f0)
[*] set erase
[gmem] delete(0x55adcc4cbea0)
[*] set erase
[gmem] delete(0x55adcc4cc4b0)
[*] set erase
[gmem] delete(0x55adcc4cc290)
[*] set erase
[gmem] delete(0x55adcc4cc570)

[-] ---- set test ----
[gmem] delete(0x55adcc4cc330)
[gmem] delete(0x55adcc4cc510)
[gmem] delete(0x55adcc4cbf40)
[gmem] delete(0x55adcc4cc450)
[gmem] delete(0x55adcc4cc390)
```

value가 8바이트이고, 포인터가 8바이트이니 8 + 8 * 4 = 40으로 40바이트가 나온다고 볼 수 있습니다. `insert`를 할 때마다 할당되고, `erase`를 할 때마다 `delete`가 해제가 일어나는 것을 알 수 있습니다.

추가적인 확인을 위해 3개의 `uint64_t`로 구성된 `std::tuple` 타입의 `std::set`을 보도록 하겠습니다. 한 원소가 24바이트이기 때문에, `set_24_test`를 작성해보았습니다.

```c++
using p64_3 = std::tuple<uint64_t,uint64_t,uint64_t>;
void set_24_test()
{
    std::cout << "\n[+] ---- set_24 test ----\n";
    std::set<p64_3> s;
    for(int i=0;i<10;i++)
    {
        uint64_t x = (10 * i + 3) % 17;
        std::cout << "[*] set_24 insert " << x << '\n';
        s.insert({x, x, x});
    }
         
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] set_24 erase\n";
        s.erase(s.begin());
    }
    std::cout << "\n[-] ---- set_24 test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- set_24 test ----
[*] set_24 insert 3
[gmem] new(56) > 0x55adcc4cc2f0
[*] set_24 insert 13
[gmem] new(56) > 0x55adcc4cc5d0
[*] set_24 insert 6
[gmem] new(56) > 0x55adcc4cc610
[*] set_24 insert 16
[gmem] new(56) > 0x55adcc4cc650
[*] set_24 insert 9
[gmem] new(56) > 0x55adcc4cc690
[*] set_24 insert 2
[gmem] new(56) > 0x55adcc4cc6d0
[*] set_24 insert 12
[gmem] new(56) > 0x55adcc4cc710
[*] set_24 insert 5
[gmem] new(56) > 0x55adcc4cc750
[*] set_24 insert 15
[gmem] new(56) > 0x55adcc4cc790
[*] set_24 insert 8
[gmem] new(56) > 0x55adcc4cc7d0
[*] set_24 erase
[gmem] delete(0x55adcc4cc6d0)
[*] set_24 erase
[gmem] delete(0x55adcc4cc2f0)
[*] set_24 erase
[gmem] delete(0x55adcc4cc750)
[*] set_24 erase
[gmem] delete(0x55adcc4cc610)
[*] set_24 erase
[gmem] delete(0x55adcc4cc7d0)

[-] ---- set_24 test ----
[gmem] delete(0x55adcc4cc650)
[gmem] delete(0x55adcc4cc790)
[gmem] delete(0x55adcc4cc5d0)
[gmem] delete(0x55adcc4cc710)
[gmem] delete(0x55adcc4cc690)
```

`40`이 `56`으로 바뀐 것을 제외하고는 크게 변한 것이 없습니다. 위의 추론이 맞다고 볼 수 있겠습니다.



이를 통해 최솟값만 저장하는 과정이라면, 메모리 할당 측면만 보아도 `std::set`이 많이 느릴 수밖에 없음을 알 수 있습니다. 한 원소에 대해 `priority_queue`는 1-2개의 qword를 요구로 하지만, `set`은 5 qword를 요구하기 때문입니다. 게다가 `priority_queue`를 `vector`를 이용하서 만들 때 발생하는 재할당 문제도 [다음과 같이 미리 `reserve`를 하여 우회할 수 있습니다](https://stackoverflow.com/a/29236236).

## `map` 컨테이너

`map`은 다음과 같은 형태로 정의됩니다.

```c++
template < class Key,                                     // map::key_type
           class T,                                       // map::mapped_type
           class Compare = less<Key>,                     // map::key_compare
           class Alloc = allocator<pair<const Key,T> >    // map::allocator_type
           > class map;
```

`set`과 비슷하나, key가 되는 타입과 값이 되는 타입이 따로 있습니다.

`map` 컨테이너는 `set`과 흡사하지만, key에 value를 대응시킬 수 있다는 점에서 차이가 납니다. 내부적인 구현이 차이가 날 수는 있겠지만,  직관적으로 위 `set`의 구조에 각 `key`에 따른 `value`만 추가하면 될 것 같습니다. 찾아보니 [실제로도 그렇게 구현되어 있는 것 같습니다](https://stackoverflow.com/q/5288320). 다만 STL을 만든 Alexander Stepanov에 의하면 지금 다시 STL울 만든다면 cache의 발전에 맞추어 [Red-black tree 대신 B*-tree로 구성할 것이다](https://interviews.slashdot.org/story/15/01/19/159242/interviews-alexander-stepanov-and-daniel-e-rose-answer-your-questions)라는 의견도 있습니다.

비슷하게 `map_test` 함수를 만들어보았습니다.

```c++
void map_test()
{
    std::cout << "\n[+] ---- map test ----\n";
    std::map<uint64_t, uint64_t> mp;
    for(int i=0;i<10;i++)
    {
        uint64_t x = (10 * i + 3) % 17;

        std::cout << "[*] map insert " << x << '\n';
        mp[x]++;
    }
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] map erase\n";
        mp.erase(mp.begin());
    }
    std::cout << "\n[-] ---- map test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- map test ----
[*] map insert 3
[gmem] new(48) > 0x55adcc4cc790
[*] map insert 13
[gmem] new(48) > 0x55adcc4cc650
[*] map insert 6
[gmem] new(48) > 0x55adcc4cc7d0
[*] map insert 16
[gmem] new(48) > 0x55adcc4cc610
[*] map insert 9
[gmem] new(48) > 0x55adcc4cc750
[*] map insert 2
[gmem] new(48) > 0x55adcc4cc2f0
[*] map insert 12
[gmem] new(48) > 0x55adcc4cc6d0
[*] map insert 5
[gmem] new(48) > 0x55adcc4cc690
[*] map insert 15
[gmem] new(48) > 0x55adcc4cc5d0
[*] map insert 8
[gmem] new(48) > 0x55adcc4cc710
[*] map erase
[gmem] delete(0x55adcc4cc2f0)
[*] map erase
[gmem] delete(0x55adcc4cc790)
[*] map erase
[gmem] delete(0x55adcc4cc690)
[*] map erase
[gmem] delete(0x55adcc4cc7d0)
[*] map erase
[gmem] delete(0x55adcc4cc710)

[-] ---- map test ----
[gmem] delete(0x55adcc4cc610)
[gmem] delete(0x55adcc4cc5d0)
[gmem] delete(0x55adcc4cc650)
[gmem] delete(0x55adcc4cc6d0)
[gmem] delete(0x55adcc4cc750)
```

예상대로 한 원소당 48바이트가 생성되는 것을 알 수 있습니다. 이를 통해 `std::set`과 `std::map`은 상수 측면에서도 그리 큰 차이가 나지 않음을 추론할 수 있습니다. 달리 생각하면 그만큼 느린 쪽에 속한다고도 볼 수 있겠습니다.

## `unordered_map` 컨테이너

`unordered_map`은 다음과 같은 형태로 정의됩니다.

```c++
template < class Key,                                    // unordered_map::key_type
           class T,                                      // unordered_map::mapped_type
           class Hash = hash<Key>,                       // unordered_map::hasher
           class Pred = equal_to<Key>,                   // unordered_map::key_equal
           class Alloc = allocator< pair<const Key,T> >  // unordered_map::allocator_type
           > class unordered_map;
```

`map`과 비교해서 `Compare`가 사라지고, hash와 관련된 `Hash`와 `Pred`가 추가되었습니다.

C++11에서 `unordered_set`과 함께 추가된 `unordred_map`은 지금까지 논한 세 개의 자료구조와는 방향성부터 다릅니다. unordered라는 이름이 시사하듯이, 검색을 정렬을 통해서가 아니라 hash를 통해서 하기 때문입니다. 즉, C++의 hash map입니다. 각 원소들은 여러 개의 bucket에 hash 값을 통해서 여기저기 들어가게 됩니다. 때문에 삽입 및 검색의 시간 복잡도가 평균적으로 $O(1)$입니다.

그럼 각 원소마다 무슨 값들이 필요할까요? 다음과 같이 생각해볼 수 있습니다.

+ key
+ value
+ key의 hash값

hash를 하면 알아서 indexing을 통해 (C++의 경우, 소수로 나눈 나머지를 통해) bucket에 삽입됩니다. 각 bucket은 linked list로 구성되어 있습니다. 이 추측이 맞다면, `uint64_t` 원소가 삽입될 때마다 24바이트가 할당될 것으로 예상해볼 수 있습니다.

비슷하게 `umap_test`를 작성해보았습니다. 이번에는 20개의 서로 다른 원소를 넣어보았습니다.

```c++
void umap_test()
{
    std::cout << "\n[+] ---- unordered_map test ----\n";
    std::unordered_map<uint64_t, uint64_t> ump;
    for(int i=0;i<20;i++)
    {
        uint64_t x = (256 * i + 3) % 401;
        std::cout << "[*] unordered_map insert " << x << '\n';
        ump[x]++;
    }
	
    for(int i=0;i<5;i++)
    {
        std::cout << "[*] unordered_map erase\n";
        ump.erase(ump.begin());
    }
    std::cout << "\n[-] ---- unordered_map test ----\n";
}
```

이를 `main` 함수에서 호출한 출력 결과는 다음과 같습니다.

```
[+] ---- unordered_map test ----
[*] unordered_map insert 3
[gmem] new(24) > 0x562fe16a2ed0
[gmem] new(24) > 0x562fe16a2ef0
[*] unordered_map insert 259
[gmem] new(24) > 0x562fe16a2e80
[*] unordered_map insert 114
[gmem] new(24) > 0x562fe16a3810
[gmem] new(56) > 0x562fe16a35d0
[gmem] delete(0x562fe16a2ef0)
[*] unordered_map insert 370
[gmem] new(24) > 0x562fe16a2ef0
[*] unordered_map insert 225
[gmem] new(24) > 0x562fe16a3830
[*] unordered_map insert 80
[gmem] new(24) > 0x562fe16a3850
[*] unordered_map insert 336
[gmem] new(24) > 0x562fe16a3870
[gmem] new(136) > 0x562fe16a3890
[gmem] delete(0x562fe16a35d0)
[*] unordered_map insert 191
[gmem] new(24) > 0x562fe16a3920
[*] unordered_map insert 46
[gmem] new(24) > 0x562fe16a3940
[*] unordered_map insert 302
[gmem] new(24) > 0x562fe16a3960
[*] unordered_map insert 157
[gmem] new(24) > 0x562fe16a3980
[*] unordered_map insert 12
[gmem] new(24) > 0x562fe16a39a0
[*] unordered_map insert 268
[gmem] new(24) > 0x562fe16a39c0
[*] unordered_map insert 123
[gmem] new(24) > 0x562fe16a39e0
[*] unordered_map insert 379
[gmem] new(24) > 0x562fe16a3a00
[*] unordered_map insert 234
[gmem] new(24) > 0x562fe16a3a20
[*] unordered_map insert 89
[gmem] new(24) > 0x562fe16a3b70
[gmem] new(296) > 0x562fe16a3b90
[gmem] delete(0x562fe16a3890)
[*] unordered_map insert 345
[gmem] new(24) > 0x562fe16a3cc0
[*] unordered_map insert 200
[gmem] new(24) > 0x562fe16a3ce0
[*] unordered_map insert 55
[gmem] new(24) > 0x562fe16a3d00
[*] unordered_map erase
[gmem] delete(0x562fe16a3d00)
[*] unordered_map erase
[gmem] delete(0x562fe16a3ce0)
[*] unordered_map erase
[gmem] delete(0x562fe16a3b70)
[*] unordered_map erase
[gmem] delete(0x562fe16a2e80)
[*] unordered_map erase
[gmem] delete(0x562fe16a2ef0)

[-] ---- unordered_map test ----
[gmem] delete(0x562fe16a3920)
[gmem] delete(0x562fe16a3960)
[gmem] delete(0x562fe16a3850)
[gmem] delete(0x562fe16a3cc0)
[gmem] delete(0x562fe16a39e0)
[gmem] delete(0x562fe16a3a20)
[gmem] delete(0x562fe16a39a0)
[gmem] delete(0x562fe16a3830)
[gmem] delete(0x562fe16a3870)
[gmem] delete(0x562fe16a3810)
[gmem] delete(0x562fe16a2ed0)
[gmem] delete(0x562fe16a3980)
[gmem] delete(0x562fe16a39c0)
[gmem] delete(0x562fe16a3940)
[gmem] delete(0x562fe16a3a00)
[gmem] delete(0x562fe16a3b90)
```

원소 삽입마다 24바이트가 할당되는 것을 보아 이론이 맞는 것으로 보입니다. 버킷을 보관하는 배열의 크기가 점차 커지는 것 또한 관찰할 수 있습니다. 8로 나눈 값을 나열하면 3 - 7 - 17 - 37 로 변화하였습니다. 위 로그에는 나와있지 않지만 그 다음 값은 79, 167입니다.

얼핏 보면 '적당히 2p+1 이상의 소수로 커지나보네'라고 생각할 수 있지만, [Codeforces에서 `unordered_map`을 hack하는 방법으로 유명해진 글](https://codeforces.com/blog/entry/62393)에 의하면 bucket의 개수로 정해지는 소수의 목록이 내부적으로 따로 있는 것으로 보입니다. 해당 글에서는 대부분의 g++에서 `unordered_map`이 hash문제로 인해 유난히 오래 걸리는 종류의 수열을 소개하기도 합니다. 다음 코드를 한 번 실행해보시는 걸 권장합니다.

```c++
// https://codeforces.com/blog/entry/62393
#include <ctime>
#include <iostream>
#include <unordered_map>
using namespace std;

const int N = 2e5;

void insert_numbers(long long x) {
    clock_t begin = clock();
    unordered_map<long long, int> numbers;

    for (int i = 1; i <= N; i++)
        numbers[i * x] = i;

    long long sum = 0;

    for (auto &entry : numbers)
        sum += (entry.first / x) * entry.second;

    printf("x = %lld: %.3lf seconds, sum = %lld\n", x, (double) (clock() - begin) / CLOCKS_PER_SEC, sum);
}

int main() {
    insert_numbers(107897);
    insert_numbers(126271);
}
```

일반적으로 `unordered_map`은 `map`보다는 조금 빠른 것으로 알려져 있습니다. 그러나 최악의 경우에는  `unordered_map`의 삽입 및 검색에 $O(n)$이 소모될 수 있습니다.

```
x = 107897: 41.178 seconds, sum = 2666686666700000
x = 126271: 0.028 seconds, sum = 2666686666700000
```

20만개를 삽입하고 순회하는데 41초나 걸리는 게, 0.02초와는 너무 대비가 되는 것을 알 수 있습니다.

## 결론

이번에 둘러본 네 개의 컨테이너는 자료구조 수업 시간에도 배울만한 유명한 자료구조로 구성되어 `vector`나 `deque`보단 이해하기 수월했습니다. 실험을 통해 `priority_queue`가 `set`이나 `map`보다 일반적으로 빠를 수밖에 없는 이유를 추론하였습니다. 또한 앞에서 논한 컨테이너와는 본질적으로 다른 `unordered_map`도 살펴보았습니다. 각 자료구조를 적재적소에 활용할 줄 아는 프로그래머가 되어야겠습니다.

