---
layout: post
title: "cracking mysql_hash_password"
date: 2025-03-26
author: kwoncycle
tags: [cryptanalysis, meet in the middle]
---

## Intro

안녕하세요, kwoncycle입니다.

이번 글에서는 `mysql_hash_password` 라는 해시 함수를 crack하는 방법에 대해 다룹니다. `mysql_hash_password` 함수는 2000년 이전에 MySQL에서 사용된 비밀번호 해시 함수로, 해시 로직은 아래의 코드와 같습니다. 

```python
def mysql_hash_password(password):
    nr = 1345345333
    add = 7
    nr2 = 0x12345671

    for c in password:
        if c == 32 or c == 9:
            continue
        nr ^= (((nr & 63) + add) * c) + (nr << 8) & 0xFFFFFFFF
        nr2 = (nr2 + ((nr2 << 8) ^ nr)) & 0xFFFFFFFF
        add = (add + c) & 0xFFFFFFFF

    return "%08x%08x" % (nr & 0x7FFFFFFF,nr2 & 0x7FFFFFFF)
```

안타깝게도 이 함수는 암호학적으로 안전하지 못한 해시 함수이며, 2000년대 이후부터 MySQL에서는 더 안전한 sha 기반의 해시 함수로 이를 교체했습니다. 그렇다면 `mysql_hash_password` 함수는 구체적으로 왜 암호학적으로 위험할까요? 

이 해시 함수의 취약점은 [link](https://link.springer.com/chapter/10.1007/11927587_23)의 논문에서 분석이 되어있으며, Codegate CTF에서 해당 논문을 참조해서 문제가 출제된 적이 있습니다. 해당 논문에 의하면, $N=2^{64}$에 대해 $O(N^{1/2}log N)$ 정도의 시간복잡도로 해시 함수를 쉽게 crack할 수 있습니다. 이는 현대의 컴퓨터로 비교적 빠르게 처리할 수 있으며, 안전한 해시함수는 절대 crack이 빠르게 되어서는 안되기에 `mysql_hash_password`는 암호학적으로 위험한 해시함수라고 할 수 있습니다.

그런데 놀랍게도 논문의 $O(N^{1/2}log N)$의 시간복잡도보다 **훨씬 빠르게** crack을 해낼 수 있는 알고리즘이 존재합니다. Codegate 대회 중에 발견한 방법으로, 알고리즘의 난이도가 크게 어렵지 않음에도 `mysql_hash_password` 해시가 이미 사장되어 연구가 진행이 되지 않은 탓인지, 아직 이에 대해 다룬 글이나 논문을 보지 못해 삼성 블로그를 통해 소개를 할까 합니다. 이 글을 통해 `mysql_hash_password`를 완전히 해체분석한 후, 역사 속으로 매장시켜줍시다!



## 간단한 용어 정리

- $N$ := 해시 함수의 codomain size입니다. `mysql_hash_function`은 해시 output으로 64bit의(엄밀히는 62bit) 수를 뱉으므로, 이 글에서 $N = 2^{64}$가 되겠습니다.
- `s[:-1]` := 문자열 `s`가 주어질 때, `s`에서 마지막 글자를 뗀 문자열을 `s[:-1]`이라고 정의합시다.


## Method 0: bruteforce
일반적으로 가장 하기 쉬운 해시 crack으로는 무작정 bruteforce를 돌리는 겁니다. 해시 함수가 적당히 잘 설계되었다는 가정 하에, 임의의 input 값에서 특정한 해시 값이 나올 확률은 $1 \over N$이며, 따라서 랜덤한 input으로 $O(N)$번의 해시 연산을 진행하면 원하는 해시값을 얻어 crack을 성공하게 됩니다. 하지만 지금은 $N=2^{64}$로, 현대의 컴퓨터라면 이만큼 연산을 하는게 불가능한 수준은 아니지만, 신속한 crack을 하기엔 다소 무리가 있다고 할 수 있습니다.

시간복잡도: $O(N)$

## Method 1: Meet in the middle
논문에 소개된 방법입니다. 먼저 다음 property 1을 봅시다.

**property 1)** 어떤 비밀 문자열 `s`에 대해, `H = mysql_hash_password(s)`의 값, `s`의 마지막 글자 `c`, 그리고 `s`의 문자 ascii 값의 총 합 `sum(s)`를 알고 있다고 가정해봅시다. 이때, `s[:-1]`의 해시값으로 가능한 값이 얼마 안되며, 이를 빠르게 구할 수 있습니다.

**proof)** 해시 함수의 로직에서 for문 내부 로직을 관찰해보면, `s`의 문자가 1개씩 차례대로 for문을 돌며 `nr`, `nr2`, `add` 값을 바꾸는 것을 알 수 있습니다. for문이 `s[:-1]`까지 돈 직후의 `nr`, `nr2`, `add`의 값을 각각 `nr_before`, `nr2_before`, `add_before`라고 하고, `s`를 전부 다 돈 후의 값들을 `nr_after`, `nr2_after`, `add_after`라고 합시다. Property의 가정으로부터 `nr_after`, `nr2_after`, `add_after`의 값은 알 수 있으며, property 증명을 위해서는 `nr_before`, `nr2_before`, `add_before`의 값을 다 빠르게 구할 수 있으면 됩니다. 이들은 아래와 같이 구할 수 있습니다.

- `add_before`는 `add_after = (add_before + c) & 0xFFFFFFFF`이라는 관계식으로부터 쉽게 구할 수 있습니다.
- `nr_before`은 `nr_after = nr_before ^ ((((nr_before & 63) + add_before) * c) + (nr_before << 8) & 0xFFFFFFFF)`이라는 관계식을 세울 수 있으며, `nr_before`의 뒤 8비트(`nr_before & 0xFF`)의 값을 알면 나머지 비트가 고정이 된다는 사실을 관찰할 수 있습니다. 그리고 위 관계식의 뒤 8비트만 떼서 실제로 cryptanalysis를 해보면, `nr_before & 0xFF`으로 가능한 값이 8개 내외라는 결론에 도달할 수 있으며, 256개를 전부 브루트포스하여 이를 구할 수 있습니다. 그리고 이 과정을 미리 전처리 table을 만들어 처리하면 매번 O(1)에 처리할 수 있습니다
- `nr2_before`은 `nr2_after = (nr2_before + ((nr2_before << 8) ^ nr_after)) & 0xFFFFFFFF`라는 관계식을 세울 수 있으며, 8비트씩 따로 생각해보면 값이 유일하게 결정됨을 관찰할 수 있고, 이를 O(1)에 구할 수 있습니다.

즉, 적당히 `add`의 값을 지정하면, 최종 해시값에 해당하는 node `H = (nr, nr2, add)`에 도달할 수 있는 이전의 state `(nr_before, nr2_before, add_before)`들을 각 `c`마다 알 수 있으며, 이걸 반복하면 `(nr, nr2, add)`에 도달할 수 있는 충분히 많은 state들을 만들 수 있습니다. 여기서 `c`의 값을 잘 조정하면, 특정한 `add_mid`의 값을 가지면서 `H = (nr, nr2, add)` node에 도달할 수 있는 `(nr, nr2, add_mid)` node를 충분히 많이 생성할 수 있습니다. \\
그런데 해시의 첫 state인 `(nr = 1345345333, add = 7, nr2 = 0x12345671)`로부터 `sum(s')=add_mid`의 값을 가지는 임의의 `s'`을 뽑는 걸 반복하면, 첫 state에서 도달할 수 있는 `(nr, nr2, add_mid)` node들도 충분히 많이 생성할 수 있게 됩니다. 이를 활용해 Meet in the middle 기법을 쓰면 $O(\sqrt{N}log N)$의 시간복잡도로 crack을 할 수 있습니다. 코드게이트에 출제된 문제의 출제자 정해는 이 method를 사용하며, 해시 crack을 위해서 1시간~3시간 정도의 시간이 소요되었습니다.

시간복잡도: $O(\sqrt{N}log N)$ \\
공간복잡도: $O(\sqrt{N})$

## Method 2: Pinning `nr`
이 method에서는 다음 property 2가 사용됩니다.

**property 2)** 어떤 문자열 `s`에 대해, `s`의 임의의 위치에 `"\x00" * 4`를 끼워넣어도 최종 해시의 `nr` 값이 불변합니다.

**proof)** 먼저 `nr2` 값은 `nr`에 아무런 영향을 주지 않음을 관찰할 수 있습니다. 이제 특정 state `(nr_0, nr2_0, add_0)`에서 `\x00` 4개가 연달아 왔을 때 변수들의 값이 어떻게 변하는지 보면, 

- `nr ^= (((nr & 63) + add) * c) + (nr << 8) & 0xFFFFFFFF`의 관계식에 `c = 0`을 대입 후 정리하면, `nr_new = nr ^ (nr << 8)`이 됩니다. 
- 따라서 `nr_0` -> `nr_0 ^ (nr_0 << 8)` -> `nr_0 ^ (nr << 16)` -> `nr_0 ^ (nr << 8) ^ (nr << 16) << (nr << 24)` -> `nr_0` 순서로 값이 변하고, 결론적으로 `nr` 값이 이전과 동일합니다.
- `add_0`의 경우 자명하게 값이 불변합니다.
- `nr2`의 값은 변화하지만, 다행히 `nr2`의 값은 `add`나 `nr` 값에 영향을 주지 않습니다.

따라서 property 2가 성립합니다.

위의 property 2로부터 해시 crack을 위한 새로운 전략을 세울 수 있습니다. \\
1) bruteforcing을 통해 최종 해시값 `H`와 같은 해시 `nr` 값을 가지는 문자열 `s0`를 찾습니다. 이때 평균적으로 $O(\sqrt{N})$의 시간복잡도가 소요되는데, `nr`의 codomain size가 32bit 정수이기 때문입니다. \\
2) 1)에서 구한 `s0`의 몇몇 문자 사이사이에 `"\x00" * 4`를 끼워넣습니다. 이때 `s0`의 어디어디에 `"\x00" * 4`를 끼워넣냐에 따라서 새로운 문자열을 기하급수적으로 많이 만들 수 있습니다. 이렇게 만들어진 새 문자열 `s`는 property에 의해 해시의 `nr` 값이 같으며, `nr2` 값은 랜덤하게 결정될 것입니다. \\
3) 2)를 원하는 `nr2`가 나올 때까지 bruteforce합니다. 이때도 평균적으로 $O(\sqrt{N})$의 시간복잡도가 소요되며, 이는 `nr2`의 codomain size입니다.

따라서 평균 $O(\sqrt{N})$의 시간복잡도에 해시 crack을 성공할 수 있게 됩니다. 실제 대회에서 저는 이 method를 사용했으며, 별도의 최적화 없이 약 5분의 시간 안에 플래그를 딸 수 있었습니다.

시간복잡도: $O(\sqrt{N})$ \\
공간복잡도: $O(1)$

## Method 3: Double - Meet in the middle!
method 2에서는 `nr`과 `nr2`를 독립적으로 만들어 구하는 방법을 고안해 빠르게 crack을 진행할 수 있었으나, 각 `nr`, `nr2`를 구하기 위해 원초적인 bruteforce를 했다는 단점이 있습니다. method 3에서는 이 bruteforce 과정을 method 1처럼 meet in the middle로 구하여 더 빠르게 진행하는 것을 목표로 합니다.

해시 crack을 위한 대략적인 전략은 다음과 같습니다. \\
1) method 1의 meet in the middle 방법을 진행하되, state를 `(nr, add)`만 보도록 합니다. 이때 $O({N}^{1 \over 4} log N)$의 시간복잡도 및 $O({N}^{1 \over 4})$의 공간복잡도가 소요됩니다. 이 과정을 통해 최종 해시값 `H`와 같은 해시 `nr` 값을 가지는 문자열 `s0`를 찾을 수 있습니다. \\
2) 이제 `nr2`를 찾아야 합니다. 이번에도 Meet in the middle이 사용됩니다. 구한 `s0`를 절반으로 나눕니다. 나눈 문자열을 각각 `s0_front, s0_back`라고 합시다. 먼저 `s0_front`의 몇몇 문자 사이에 `"\x00" * 4`를 끼워넣은 새로운 `s0_front_pluged` 문자열들을 $O({N}^{1 \over 4})개 정도 만든 후, 이들의 해시를 계산해 처음 state에서 도달할 수 있는 `(nr', nr2_front)` state들을 계산합니다. 이때 property 2에 의해 `nr'` 값이 고정됩니다. \\
3) 이후 `s0_back`의 문자 사이에 `"\x00" * 4`를 끼워넣어 새로운 `s0_back_pluged` 문자열들을 $O({N}^{1 \over 4})개 정도 만든 후, method 1에서 사용한 crack 과정을 각각의 `s0_back_pluged` 문자열에 돌립니다. 이 과정에서 원하는 구하는 해시의 state 값에 최종적으로 도달할 수 있는 `(nr', nr2_back)` state들을 구할 수 있습니다. \\
4) 2)와 3)에서 구한 state들은 `nr'` 값을 공유하기에, 각 state의 크기가 $O({N}^{1 \over 4})$ 정도만 되면 값 충돌이 발생합니다. 이를 통해 원하는 해시값을 가지는 문자열 `s0_pluged`를 구할 수 있습니다!

시간복잡도: $O({N}^{1 \over 4} log N)$ \\
공간복잡도: $O({N}^{1 \over 4})$

## Conclusion
Method 3을 통해 기존 논문의 $O(\sqrt{N} log N)$ 의 시간복잡도에 작동하는 알고리즘을 $O({N}^{1 \over 4} log N)$로 대폭 발전시킬 수 있었습니다. 보통 Problem solving과 realworld가 꽤 괴리가 있다는 인식이 있는데, 이번 글을 작성하면서 Meet in the middle을 포함한 몇몇 PS-테크닉을 통해 실제로 쓰였던 해시 함수의 cryptanalysis을 할 수 있었다는 점이 개인적으로 상당히 인상적으로 느껴졌습니다. 또, 현재는 `mysql_hash_password`가 사용되지 않고 있기에 이 글의 method를 직접적으로 사용하는 것은 힘들겠지만, 테크닉 자체는 다른 해시 함수 및 암호의 cryptanalysis를 하는데 충분히 적용을 해볼 수 있을 것이라고 생각합니다.