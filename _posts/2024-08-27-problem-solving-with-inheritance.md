---
layout: post
title: "PS에서 C++ 상속 활용하기"
date: 2024-08-27 20:00:00
author: psb0623
tags: []
---

C++에는 클래스가 있고, 클래스에는 상속이 있다는 사실은 대부분 알고 계실 것입니다. 상속의 개념을 `Animal`, `Dog`, `Cat`의 관계로 설명하는 것을 한 번쯤은 보셨을 것입니다.

하지만 실제 PS를 할 때 C++의 상속 개념을 활용하는 경우는 그닥 많지 않습니다. 그러나 상속을 잘 활용하면, PS에서도 더 편하고 깔끔하게 코드를 짤 수 있게 됩니다. 이 글에서는 PS에서 실용적으로 상속을 활용할 수 있는 몇 가지 사례를 소개합니다.

## 모든 원소에 특정 값 더하기

아래의 쿼리를 해결하는 문제를 생각해봅시다.

- 배열에 $x$ 추가하기
- 배열에서 가장 큰 값을 출력하기
- 배열에서 가장 큰 값을 하나 제거하기
- 배열의 모든 원소에 $c$ 더하기

위에서부터 3개의 쿼리는 C++ STL의 `priority_queue` 컨테이너가 기본으로 제공하는 기능입니다. 따라서 일단 최대 힙을 사용한다고 생각하고, 마지막 쿼리를 어떻게 처리할지 생각해봅시다. 힙의 모든 원소에 $c$를 더하는 연산을 구현하는 방법은 여러 가지가 있을 수 있지만, 가장 효율적인 방법은 다음과 같습니다.

- $offset$이라는 변수를 만들고, 힙의 원소를 읽을 때마다 항상 $offset$을 더해서 사용하도록 합니다.

이렇게 하면 단순히 $offset$에 $c$를 더하는 $O(1)$의 작업만으로 힙의 모든 원소에 $c$를 더한 것과 동일한 효과를 얻을 수 있게 됩니다. 하지만 이러한 방식을 그대로 구현하는 경우 프로그래머가 신경써야 할 것이 많아집니다. 예를 들어, 아래의 규칙을 항상 지켜야 합니다.

- 힙에서 원소를 읽을 때는 $offset$을 더한 후 읽어야 한다.
- 힙에 원소를 삽입할 때는 $offset$을 뺀 후 삽입해야 한다.

이처럼, 프로그래머는 힙과 상호작용하는 모든 부분에서 $offset$을 일일이 더하거나 빼주어야 합니다. 이는 당연히 실수가 발생하기 쉬운 부분이며, 단 하나라도 실수하면 틀린 프로그램이 되기 때문에 주의가 필요합니다.

따라서, 여러분 중 대부분은 이러한 문제를 **함수**를 통해 해결할 것입니다.

```c++
priority_queue<int> pq;
int offset;

void add_all(int c) {
    offset += c;
}

void push(int x) {
    pq.push(x - offset);
}

int top(int x) {
    return pq.top() + offset;
}
```
위처럼 함수를 통해 작업을 실행하도록 하면, $offset$을 일일이 관리할 필요 없이 편하게 처리할 수 있습니다. 하지만 이러한 방법에도 여러 가지 한계가 존재합니다.

### 일관적이지 않은 인터페이스

위 방식에서 원소를 삽입하려면 `push(x)`를 호출해야 하고, 가장 큰 원소를 제거하려면 `pq.pop()`을 호출해야 합니다. 즉, 어떤 연산은 전역으로 선언된 함수를 활용해야 하고, 어떤 연산은 `priority_queue`의 멤버 함수를 활용해야 합니다. 따라서, 힙과 상호작용하는 인터페이스가 통일되어있지 않으며, 코드의 가독성에 악영향을 미칩니다. 

또한, $offset$을 고려하여 삽입하는 전역 함수 `push(x)`와 $offset$을 고려하지 않고 삽입하는 멤버 함수 `pq.push(x)`가 공존하게 되는 것도 문제입니다. 비슷한 역할을 하는 두 인터페이스 중 어떤 것을 활용할지는 전적으로 프로그래머의 책임이며, 실수로 헷갈리는 경우 오류를 찾기 특히 어려울 수 있습니다.

### 힙이 여러 개라면?

만약 동일한 기능을 제공하는 힙을 2개 관리해야 한다면 어떻게 해야 할까요?

```c++
priority_queue<int> pq1, pq2;
int offset1, offset2;

void push1(int x) {
    pq1.push(x - offset1);
}

void push2(int x) {
    pq2.push(x - offset2);
}
/* 하략 */
```
가장 쉬운 방법은, 위처럼 똑같은 함수를 각각의 힙에 대해 2개 구현하는 것입니다. 그러나 일일이 복사 붙여넣기를 해야 하고, 코드가 더러워진다는 단점이 있습니다. 또, 만약 임의의 자연수 $n$에 대해 $n$개의 힙을 관리하고 싶은 경우에는 활용할 수 없습니다.

따라서, 여러 개의 힙에 대해서도 동일한 코드를 활용할 수 있도록 '힙에 대한 참조'와 $offset$을 매개변수로 받아서 처리하는 방법을 생각해볼 수 있습니다.

```c++
priority_queue<int> pq1, pq2, ...;
int offset1, offset2, ...;

void insert(priority_queue<int>& s, int offset, int x) {
    s.insert(x - offset);
}
/* 하략 */
```
하지만 이러한 경우에는 원소를 삽입하기 위해 `insert(pq1, offset1, x)`처럼 복잡하게 호출해야 한다는 단점이 있습니다. 특히, `pq1`과 `offset1`이 명시적으로 연결되어 있지 않아 프로그래머가 직접 지정해줘야 한다는 점에서 실수가 발생하기 쉽습니다. 예를 들어, `insert(pq1, offset2, x)`처럼 호출하게 되면 잘못된 $offset$을 이용해 삽입하게 될 것입니다.

결국, 어느 방법을 택하더라도 코드를 깔끔하게 짤 수 없습니다.

### 원소 타입을 바꾼다면?

예를 들어, 문제를 풀다가 `int`로는 값을 저장하기에 충분하지 않다는 것을 깨닫고 원소의 타입을 `long long`으로 바꾸고 싶을 수 있습니다.

```c++
priority_queue<int> pq;
int offset;

void add_all(int c) {
    offset += c;
}

void push(int x) {
    pq.push(x - offset);
}

int top(int x) {
    return pq.top() + offset;
}
```
그러한 경우, 위 코드에 존재하는 모든 `int`를 `long long`으로 바꿔주어야 합니다. 위 코드에만 해도 6개의 `int`가 있으며, 만약 함수가 더 많아진다면 바꿔주어야 하는 타입의 개수도 훨씬 많아질 것입니다. 이는 프로그래밍에 있어 매우 소모적인 작업을 유발합니다.

### 모듈화

위에서 제시한 한계들은 모두 **모듈화**를 통해 해결할 수 있습니다. 즉,

- 배열에 $x$ 추가하기
- 배열에서 가장 큰 값을 출력하기
- 배열에서 가장 큰 값을 하나 제거하기
- 배열의 모든 원소에 $c$ 더하기

위 4가지 기능을 모두 지원하는 **클래스**를 아예 새로 만들면 된다는 뜻입니다.

해당 클래스의 이름을 `offset_pq`라고 합시다. 그러면, `offset_pq`에는 `push()`, `pop()` 등의 기존 메서드와, 모든 원소에 $c$를 더하는 `add_all()` 메서드가 존재할 것입니다.

`offset_pq`를 어떻게 잘 구현했다고 합시다. `offset_pq`를 활용하면, 위에서 제시한 모든 문제가 해결됩니다. 먼저, 힙과의 상호작용이 모두 `offset_pq`의 멤버 함수를 통해 일어나기 때문에 일관적인 인터페이스를 가지게 됩니다. 따라서 위에 언급한 `push(x)`와 `pq.push(x)`의 공존 문제도 해결됩니다.

또한, `offset_pq`는 여러 개를 만들거나 원소 타입을 바꾸는 작업도 매우 간단하게 해결할 수 있습니다.

```c++
offset_pq<int> pq1, pq2;
offset_pq<long long> pq3;
offset_pq<double> pq4;
```

필요한 모든 기능이 이미 클래스에 구현되어 있기 때문에, 프로그래머는 아무것도 신경쓰지 않고 활용할 수 있기 때문입니다. 마치 `priority_queue`를 사용할 때와 같이 말이죠.

새로운 클래스를 만드는 것의 장점을 확인했으니, 이제는 실제로 어떻게 `offset_pq`를 구현할지 고민할 차례입니다.

### 구현 1: priority_queue를 멤버로 사용하기

가장 첫 번째로 떠올릴 수 있는 방법은 `priority_queue`와 `offset`을 멤버 변수로 가지는 클래스를 구현하는 것입니다. 

```c++
template<typename T>
class offset_pq {
private:
    priority_queue<T> pq;
    T offset;
public:
    void add_all(T c) {
        offset += c;
    }
    void push(T x) {
        pq.push(x - offset);
    }
    T top() {
        return pq.top() + offset;
    }
    void pop() {
        pq.pop();
    }
    size_t size() {
        return pq.size();
    }
    /* 하략 */
};
```
`push()`, `pop()`, `size()` 등 모든 메서드를 `pq`와 `offset`을 이용해 적절히 구현해줄 수 있습니다. 이를 통해 모듈화라는 목표를 달성할 수 있고, 실제로도 꽤 유용하게 활용할 수 있습니다.

그러나 이렇게 구현하면 기존 `priority_queue`에 존재하는 모든 함수를 다시 작성해야 한다는 문제점이 있습니다. `offset_pq`에서 `push()`, `pop()` 등의 기본 기능뿐만 아니라 `size()`, `empty()` 등의 다양한 기능을 활용하려면, `offset_pq` 클래스에 같은 이름의 함수를 만들고 구현해주어야 합니다. 따라서 어쩔 수 없이 코드 길이가 길어지게 됩니다.

특히, 실제로 `offset`과 관련한 추가적인 처리가 필요한 함수는 `add_all()`, `push()`, `top()` 3개 뿐이며 나머지 함수들은 단순히 `priority_queue`의 메서드를 그대로 호출해서 반환할 뿐입니다. 불필요한 반복이 계속된다는 생각이 들 수 밖에 없죠.

### 구현 2: priority_queue를 상속받기

좀 더 깔끔하게 코드를 짜려면 어떻게 해야 할까요? `offset_pq`가 `priority_queue`의 메서드를 대부분 수정 없이 그대로 활용한다는 점에서, **상속**을 고려할 수 있습니다.

`priority_queue`를 상속받으면 기본적으로 `priority_queue`의 멤버 변수와 메서드가 그대로 포함되며, 이 상태에서 원하는 멤버 변수나 메서드를 추가할 수 있게 됩니다. 또한, `push()`, `top()`처럼 기존 `priority_queue`에서 수정해야 하는 메서드는 **오버라이딩**을 통해 새로운 구현으로 대체해줄 수 있습니다. 이와 같은 방식으로 `offset_pq`를 구현한 예시는 아래와 같습니다.

```c++
template<typename T>
class offset_pq: public priority_queue<T> {
private:
    typedef priority_queue<T> super;
    T offset = 0;
public:
    void add_all(T c) {
        offset += c;
    } 
    void push(const T& x) {
        super::push(x - offset);
    }
    T top() {
        return super::top() + offset;
    }
};
```

보시다시피 확실히 깔끔한 코드로 `offset_pq`를 구현할 수 있습니다. 구현에 대한 설명은 아래와 같습니다.

- private 멤버 변수로 `offset`을 추가했습니다.
- `typedef`를 이용해 부모 클래스(혹은 슈퍼 클래스)인 `priority_queue<T>`에 편하게 접근할 수 있도록 `super`라는 약칭을 붙였습니다.
- `offset`에 값을 더해주는 `add_all()` 메서드를 추가하였습니다.
- `push()`는 `offset`을 이용해 입력 값을 적절히 처리한 뒤, 부모 클래스인 `priority_queue<T>`의 메서드를 호출하여, `priority_queue`에서 상속받은 실제 컨테이너가 적절한 원소를 가지도록 합니다. `top()`도 비슷하게 구현되었습니다.

이처럼, 상속을 활용하면 매우 짧고 간결한 코드로 모듈화를 달성할 수 있습니다. 그러나, 모든 경우에 상속을 활용할 수 있는 것은 아닙니다.

예를 들어, `priority_queue`의 경우 원소에 접근할 수 있는 인터페이스가 `top()` 하나뿐이므로 `top()`에 대해서만 `offset`을 더하고 읽도록 오버라이딩 해주면 올바른 구현이 됩니다.

그러나 모든 원소에 $c$를 더하는 기능을 `vector`에 대해서 구현하려고 하면, 상속을 활용하여 해결하는 것이 사실상 불가능하다는 것을 알 수 있습니다. `vector`의 원소에 접근할 수 있는 인터페이스가 인덱스 연산자 `operator[]`뿐만이 아니기 때문입니다. 예를 들어, `v.begin()` 등의 iterator를 통한 접근, 또는 원소 자체의 참조자나 포인터를 통한 접근이 가능합니다. 이처럼, 가능한 모든 접근 방법에 대해 일관적으로 `offset`을 더한 후 반환하도록 구현하는 것은 불가능한 일입니다.

따라서 먼저 컨테이너의 특성을 파악한 다음, 상속을 활용해 기능을 추가할지, 아니면 다른 구현 방법을 택할지 적절히 선택하는 것이 중요하겠습니다.

### 실전 예시: Slope Trick

위에서 소개한 방법을 효과적으로 활용할 수 있는 대표적인 알고리즘은 **Slope Trick**입니다. Slope Trick이 무엇인지에 대해서는 Slope Trick에 대해 설명한 다른 글들을 참고하시길 부탁드립니다.

결국 Slope Trick에서 요구되는 것은 piecewise linear convex function을 효율적으로 관리하는 것이며, 특히 function의 평행이동, function과 function의 합 등의 연산을 효율적으로 수행할 수 있어야 합니다.

자세한 설명은 생략하겠지만, 보통 Slope Trick에서는 piecewise linear convex function을 표현할 때 기울기가 바뀌는 지점의 $x$좌표를 `priority_queue` 2개를 활용하여 관리합니다. (각각 최대 힙, 최소 힙) 따라서, function을 $x$축 방향으로 $c$만큼 평행이동하려면 2개의 `priority_queue`의 모든 원소에 $c$를 더해야 합니다. 이 때, `priority_queue` 대신 상속으로 구현한 `offset_pq`를 활용하여 훨씬 간결하게 해당 기능을 구현할 수 있습니다.

아래는 `offset_pq`를 활용하여 백준 온라인 저지의 [BOJ 수열 1](https://www.acmicpc.net/problem/13323)을 푸는 코드입니다. 최대 힙과 최소 힙을 모두 활용하기 위해 기존 `offset_pq`에서 조금 수정된 부분이 있습니다.

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

template<typename T, bool rev>
class offset_pq : public priority_queue<T> {
private:
	typedef priority_queue<T> super;
	T offset = 0;
public:
	void add_all(T c) {
		offset += c;
	}
	void push(const T& x) {
		super::push(rev ? (offset - x) : (x - offset));
	}
	T top() {
		return (rev ? -super::top() : super::top()) + offset;
	}
};

struct Function { 
	ll offset = 0;
	offset_pq<ll, 0> l; offset_pq<ll, 1> r;
	void add_inc(ll x) {
	    // [x, inf) 구간에 (x, 0)을 지나고 기울기 1로 증가하는 일차함수를 더한다.
		if(!l.empty() && x < l.top()) {
			offset += l.top() - x;
			r.push(l.top()); l.pop(); l.push(x);
		} else r.push(x);
	}
	void add_dec(ll x) {
	    // (inf, x] 구간에 (x, 0)을 지나고 기울기 -1로 감소하는 일차함수를 더한다.
		if(!r.empty() && r.top() < x) {
			offset += x - r.top();
			l.push(r.top()); r.pop(); r.push(x);
		} else l.push(x);
	}
	void add(Function& o) { // consumes o
	    // 다른 Function을 현재 Function에 더한다.
		offset += o.offset;
		while(!o.r.empty()) {
			ll x = o.r.top();
			add_inc(x);
			o.r.pop();
		}
		while(!o.l.empty()) {
			ll x = o.l.top();
			add_dec(x);
			o.l.pop();
		}
	}
} f;

ll n, a[1000010];

int main() {
	ios_base::sync_with_stdio(0); cin.tie(0);
	cin >> n;
	for(int i=0;i<n;i++) cin >> a[i];
	
	for(int i=0;i<n;i++) {
		f.r = offset_pq<ll, 1>();
		f.l.add_all(1); f.r.add_all(1);
		f.add_inc(a[i]); f.add_dec(a[i]);
	}
	cout << f.offset;
}
```

## 음수 인덱스를 허용하는 배열 만들기

가끔, 편의상의 이유로 음수 인덱스에 무언가 의미를 부여하고 싶은 경우가 있습니다. 예를 들어, 배열 `a[0]`, `a[1]`, …, `a[N-1]`에서 각 인덱스까지의 누적합 `sum[i] = a[0] + a[1] + … + a[i]`를 계산하는 경우를 생각해봅시다.

```c++
for(int i=0;i<N;i++) {
    sum[i] = (i-1 >= 0 ? sum[i-1] : 0) + a[i];
}
```
이 때, `i`가 0인 경우 `sum[-1]`에 접근하여 오류가 발생할 수 있기 때문에 `i`가 0인 경우는 따로 예외 처리를 해주어야 합니다. 그런데 만약 `sum[-1] = 0`이라고 정의하고 읽을 수 있다면, 예외 처리가 필요 없고 코드도 더 깔끔해지지 않을까요?

다른 예시로, 다이나믹 프로그래밍을 할 때도 점화식에 포함된 `d[i-j]` 등에서 `i-j < 0`이면, 즉 존재하지 않는 값이면 0을 대신 사용하도록 하는 경우가 많습니다. 그런데 이 때 음수 인덱스에 접근하면 오류가 발생하기 때문에, 항상 `if`문이나 `(i-j >= 0 ? d[i-j] : 0)`처럼 예외 처리를 해주어야 했습니다.

마찬가지로 `d[i-j]`에서 음수 인덱스를 오류 없이 읽을 수 있고, 또 음수 인덱스를 읽은 결과가 항상 0이라면, 귀찮은 예외 처리 없이도 다이나믹 프로그래밍을 올바르게 구현할 수 있지 않을까요?

사실 `map<int,int>`를 활용하면 인덱스로 음수를 사용하는 것이 가능하긴 하지만, 원소 접근 시간복잡도가 $O(\log n)$이므로 원소 접근이 $O(1)$인 배열에 비해 많이 느립니다. 어떻게 하면 배열의 성능을 그대로 활용하면서 음수 인덱스를 허용할 수 있을까요?

우선 생각할 수 있는 방법은 적당한 $offset$을 두어, 인덱스에 항상 $offset$을 더해 사용하는 방식입니다. `a[i+offset]`처럼 말이죠. 그러면 `i`가 `-offset`과 `0` 사이의 음수더라도 `a[i+offset]`에서는 음수 인덱스 접근이 일어나지 않습니다. 그러나 문제는 마찬가지로 프로그래머가 배열에 접근하기 전 일일이 인덱스에 `offset`을 더해주어야 한다는 점이며, 실수가 발생하기도 쉬운 구조입니다. 

이러한 문제 역시 상속과 오버라이딩으로 해결할 수 있습니다.

### 구현: array 상속받기

결국 array의 모든 기능은 동일하게 사용하면서 음수 인덱스만 허용하고 싶은 것이므로, array를 상속받은 후 인덱스 연산자 `operator[]`를 오버라이딩하여 구현할 수 있습니다.(편의상 인덱스 연산자를 제외한, iterator나 다른 방식으로 원소에 접근하는 경우는 고려하지 않겠습니다.) 구현 예시는 아래와 같습니다.

```
template<typename T, int N>
struct offset_array : array<T, 2*N> {
	int offset = N;
	T& operator[](int idx) { return array<T,2*N>::operator[](idx + offset); }
};
```
우선 사용자가 `offset_array<T, N>`을 선언하면, `-N`과 `N-1` 사이의 인덱스를 허용하는 배열이 생성됩니다. 이 때, 실제로 관리되는 배열은 부모 클래스인 `array<T, 2*N>`에 의해 관리되는 길이 `2*N`의 배열입니다.

오버라이딩된 `operator[]` 메서드에 따라, `offset_array`의 어떤 인덱스 `idx`에 대한 접근은 부모 클래스의 인덱스 `idx + offset`에 대한 접근으로 변환됩니다. 즉, 자동으로 `offset`을 더해 해석하는 기능이 구현되어 있는 것입니다.

따라서, 상속을 활용하면 아주 짧은 코드만으로 아무 문제 없이 음수 인덱스에 접근할 수 있는 배열을 구현할 수 있습니다.

### 실전 예시: 2-SAT

2-SAT 문제를 풀 때, 어떤 불리언 변수 $x_i$에 대해서 $x_i$와 $\neg x_i$를 표현하는 노드를 각각 만들어주어야 합니다. 이 때, 각 노드에 대해 어떻게 노드 번호를 붙일지는 프로그래머의 몫입니다. 예를 들어, $x_i$의 노드 번호를 `2*i`, $\neg x_i$의 노드 번호를 `2*i + 1`로 붙이는 방법이 있을 수 있습니다. 그러나 이러한 방식으로 노드 번호를 붙이게 되면 어떤 것이 $x_i$에 해당하는 노드이고 어떤 것이 $\neg x_i$에 해당하는 노드인지 *직관적으로* 알기는 힘들었습니다.

이 때, 음수 인덱스를 활용하여 $x_i$의 노드 번호를 `i`, $\neg x_i$의 노드 번호를 `-i`로 붙여버리면 각 노드 번호가 나타내는 불리언 변수를 훨씬 직관적으로 파악할 수 있게 됩니다. 이러한 방식으로 구현한 2-SAT 코드는 아래와 같습니다. SCC와 2-SAT 알고리즘 모두 음수 인덱스 위에서 동작하도록 구현되었으며, 백준 온라인 저지의 [2-SAT - 4](https://www.acmicpc.net/problem/11281)를 통과하는 코드입니다.

```c++
#include<bits/stdc++.h>

using namespace std;

constexpr int N = 10010;

template<typename T, int N>
struct offset_array : array<T, 2*N> {
	int offset = N;
	T& operator[](int idx) { return array<T,2*N>::operator[](idx + offset); }
};

struct SCC {
	offset_array<vector<int>, N> v, rv;
	offset_array<int, N> vis, g;
	vector<int> s, st; // s: 그래프에 존재하는 모든 노드 번호의 리스트
	vector<vector<int>> scc;
	
	void insert(int a, int b) {
	    v[a].push_back(b);
	    rv[b].push_back(a);
	}
	void dfs1(int cur) {
	    if(vis[cur]) return;
	    vis[cur] = 1;
	    for(auto nxt:rv[cur]) dfs1(nxt);
	    st.push_back(cur);
	}
	void dfs2(int cur) {
	    if(vis[cur]) return;
	    vis[cur] = 1;
	    for(auto nxt:v[cur]) dfs2(nxt);
	    scc.back().push_back(cur);
	    g[cur] = scc.size();
	}
	void init() {
		for(auto x:s) if(!vis[x]) dfs1(x);
		for(auto x:s) vis[x] = 0;
		while(!st.empty()) {
		    int x = st.back();
		    st.pop_back();
		    if(!vis[x]) scc.emplace_back(),
		    dfs2(x);
		}
	}
};

struct SAT {
	SCC scc;
	offset_array<int, N> vis, val;
	void insert(int a, int b) { scc.insert(-a, b); scc.insert(-b, a); }
	bool init(int n) {
		for(int i=1;i<=n;i++) scc.s.push_back(i), scc.s.push_back(-i); // 그래프에 존재하는 모든 노드 번호를 등록
		scc.init();
		for(int i=1;i<=n;i++) if(scc.g[i] == scc.g[-i]) return 0;
		for(auto& t:scc.scc) for(int x:t) if(!vis[x]) val[x] = !val[-x], vis[x] = 1; 
		return 1;
	}
} sat;

int n, m;

int main() {
	ios_base::sync_with_stdio(0); cin.tie(0);
	cin >> n >> m;
	for(int i=0;i<m;i++) {
		int a, b;
		cin >> a >> b;
		sat.insert(a, b);
	}
	bool ret = sat.init(n);
	cout << ret << "\n";
	if(ret) for(int i=1;i<=n;i++) cout << sat.val[i] << " ";
}
```

## 마치며

지금까지 PS에서 C++ 상속을 활용하는 법에 대해 알아보았습니다. 또, 상속 뿐만 아니라 C++이 제공하는 다른 여러 기능을 PS에서 적극적으로 활용해보는 것도 하나의 재미가 될 수 있을 거라 생각합니다. C++로 PS를 하시는 분들께 이 글이 도움이 되었기를 바라며, 글을 마치도록 하겠습니다. 읽어주셔서 감사합니다!
