---
layout: post
title:  "GCC 확장 기능"
date:   2020-01-17-01:05
author: djm03178
tags: [GCC, extension, C, C++]
---

## 서론 ##
프로그래밍을 하면서 C나 C++의 레퍼런스를 찾다 보면 표준의 디테일에 놀라게 되는 경우가 많습니다. 평소에는 무심하게 사용해왔던 문법에 들어있는 다양한 제약 조건이나, 키워드들의 알지 못했던 쓰임새들을 종종 보게 됩니다.

그런데 그렇게 방대한 스펙을 가지고도 사람들은 더 많은 것을 바라기 마련입니다. 표준상으로는 지원되지 않는 문법을 통해 더욱 편리한 코드를 만들거나, 어셈블리 명령을 직접 적어넣지 않아도 프로세서의 좋은 기능을 컴파일러가 최대한 살릴 수 있게 보장해 주었으면 하는 등 끝없이 더 많은 기능을 원하고 있습니다.

이를 충족시키기 위해 대부분의 이름 있는 컴파일러들은 독립적인 라이브러리로서 존재하는 것이 아닌 컴파일러 자체가 알고 있는 확장 기능들을 다수 제공하고 있습니다. 이 글에서는 자주 사용하게 되는 컴파일러 중 하나인 GCC에서 어떤 확장 기능들을 제공하고, 어떤 곳에 유용하게 사용될 수 있는지를 알아보겠습니다.

## GCC 확장 기능 ##
GCC는 매우 다양한 확장 기능을 제공합니다. C/C++ 표준에만 존재하는 문법을 반대로 C++/C에도 적용시킨다거나, 컴파일러 내부적으로 사용하기 위한 함수, 그리고 특정 프로세서들에서만 지원되는 기능 등 다방면에 걸쳐 지원을 하고 있습니다. 또한 컴파일러의 특정 기능을 사용할 수 있게 해주는 #pragma도 확장 기능에 해당되지만, 이 글에서는 자세히 다루지는 않겠습니다.

크게 '문법 확장'과 '내장 함수', 그리고 '프로세서 의존 기능'으로 분류를 나누어 설명해 보겠습니다.

### 문법 확장 ###
문법 확장은 표준상 문법적으로 지원되지 않는 기능을 언어 자체에 추가한 것입니다. 새로운 키워드를 추가하거나, 특정 키워드가 들어갈 수 없는 자리에도 쓸 수 있게 만들어주는 것, 또는 키워드의 사용법을 추가하는 것 등이 해당됩니다.

이들 중 어떤 확장들은 완전히 새로운 키워드이거나 새로운 사용법을 추가한 것이지만, 일부는 C/C++의 특정 버전의 기능을 다른 버전/언어에서 사용할 수 있도록 한 것들도 있습니다. 어떠한 것들은 특정 표준 버전에서 '보편적인 확장'으로 언급되기도 합니다.[^1]

#### 특정 버전의 기능 이식 ####
GCC는 C/C++ 표준을 가능한 따르고 있고, C/C++의 여러 표준 버전에 따른 컴파일을 지원합니다. 이는 때때로 불편한 상황을 만들어 내는데, C90 표준에 맞추어 작성된 코드를 개선하기 위해 C99에서 추가된 유용한 기능(예를 들어, 가변 길이 배열)을 사용할 수 없다면 안타까울 것입니다. GCC에서는 해당 표준의 지향점을 해치지 않는 선에서 이러한 기능들을 버전을 넘어 사용할 수 있도록 해주는 확장 기능을 제공하고 있습니다. 이러한 문법 확장은 단순히 C 버전끼리, C++ 버전끼리만이 아닌 C와 C++ 사이에서도 이루어집니다.

#### 추가된 문법 ####
GCC 확장 문법 중에는 단순히 호환성을 위한 것이 아닌, 특별한 목적을 위해 자체적으로 추가한 문법들도 있습니다. 이러한 문법들은 다른 컴파일러들에서는 잘 보기 어려우며, 편의성을 극대화 시켜줍니다. 주로 새로운 키워드를 통해 기존에 없던 문법을 사용하게 됩니다.

### 내장 함수 ###
GCC의 내장 함수들은 대체로 `__builtin_`으로 시작합니다. 이 함수들은 C/C++의 일반 함수와 똑같은 형식의 프로토타입을 가지고 있기 때문에 문법적인 측면에서는 달라지는 것이 없지만, 이후 컴파일러가 이 함수들이 들어간 자리에 마법을 부리는 것을 가능하게 합니다.

이들 중에는 매우 흥미롭고 사용법이 복잡한 것들도 많기 때문에, 추후 내장 함수들만을 모아 설명하는 글을 따로 작성해보려고 합니다.

### 프로세서 의존 기능 ###
프로세서에는 많은 종류가 있고, 각 프로세서가 지원하는 명령어들은 서로 다릅니다. 그런데 어떤 프로그램들은 타겟 프로세서가 특정되지 않을 수 있습니다. 특정 프로세서들이 지원하는 기능을 사용하기 위해서는 타겟에 따라 그에 맞는 어셈블리 코드를 직접 작성해야 하는데, 어셈블리를 프로세서마다 손으로 짜넣는다는 것은 매우 번거로운 일이며 컴파일러의 최적화와 충돌을 일으키지 않는다는 보장도 어렵습니다. 이러한 경우를 위해 GCC는 C 문법 내에서 해당 프로세서들의 기능을 사용할 수 있게 해주는 다양한 확장 기능을 제공합니다.

## 확장 기능 목록 ##
이제 구체적으로 GCC에서 제공하는 확장 기능들에 어떤 것들이 있는지 살펴보겠습니다. 이러한 기능들은 너무나 많기 때문에 이 글에서 전부 설명하는 것은 불가능하고, 일부 유용하거나 흥미로운 기능들을 위주로 설명하려고 합니다. 이후 기회가 된다면 이 글에서 언급하지 못한 것들에 대해서도 추가로 글을 써보도록 하겠습니다.

### C++ 스타일의 주석 ###
대부분의 C 컴파일러들이 이를 지원하고 있기 때문에 잘 눈치채지 못하는 기능이지만, 본래 `//`로 시작하는 한 줄 주석은 C90까지는 C에 존재하지 않던 주석이었습니다. GCC에서도 C99 이전의 표준을 사용한다는 것을 명시적으로 표현하지 않는 한 이들을 사용할 수 있도록 지원해주고 있습니다.

### `inline` 키워드 ###
함수 호출을 함수 몸체 자체로 대체할 수 있도록 지시하는 `inline` 키워드 역시 C90까지는 존재하지 않았으나 사용할 수 있도록 확장이 지원됩니다. 때때로 매우 가볍고 자주 호출되는 함수들의 경우에는 '호출' 과정을 생략하는 것이 성능에 큰 영향을 줄 수 있다는 점에서 매우 유용하다고 할 수 있습니다.

### 가변 길이 배열 ###
C99를 제외한 C/C++ 표준에는 원래 가변 길이 배열이 없지만, GCC에서는 그 편리성을 인정하여 C90과 C++에서도 사용할 수 있게 하고 있습니다. 현재 문제 풀이 분야에서는 매우 애용되는 확장 기능 중 하나입니다.

가변 길이 배열은 컴파일 시간에 그 크기가 결정되지 않고, 그 배열이 선언되는 시점에 지정된 크기만큼의 배열이 만들어지는 것을 의미합니다. 예를 들면 아래와 같은 코드가 해당됩니다.

```cpp
#include <iostream>
using namespace std;

int main()
{
	int n;
	cin >> n;
	int a[n];
	cout << sizeof(a) << endl;
}
```

`sizeof(int)`가 4인 환경에서 이를 실행하면 결과로는 4*(입력된 크기)가 나오는 것을 볼 수 있습니다. 즉, `a` 배열은 인덱스를 입력된 크기만큼만 사용할 수 있도록 만들어진 것입니다.

### 이름 없는 구조체와 공용체 ###
이 기능은 C11에 추가된 기능이지만 다른 버전에서도 사용할 수 있도록 GCC가 지원해 줍니다. 예를 들면 아래와 같은 코드가 있습니다.

```c
struct {
  int a;
  union {
    int b;
    float c;
  };
  int d;
} foo;
```

`foo`의 타입은 이름 없는 구조체이며, 이에 속한 공용체 역시 이름이 없습니다. `b`, `c`는 각각 `foo.b`, `foo.c`로 접근이 가능합니다. 이와 같은 이름이 서로 겹쳐서 모호하게 만들면 안 된다는 것은 제약 조건입니다.

### 길이 0의 배열 ###
GCC에서는 길이가 0인 배열이 허용됩니다. 대체 길이가 0인 배열을 어디에 쓰려고 그런 기능을 추가했을까요? 대표적인 사용 예시로는 '가변 길이 객체'가 있습니다.

```c
struct line {
  int length;
  char contents[0];
};

struct line *thisline = (struct line *)
  malloc (sizeof (struct line) + this_length);
thisline->length = this_length;
```

`struct line`은 2개의 멤버 `length`와 `contents`를 가지는데, `contents`의 내용물은 비어있을 수도 있는 경우를 가정한 것입니다. `contents`에 아무것도 넣지 않는 경우 굳이 메모리를 할당해 줄 필요가 없는데, 크기가 1 이상이어야 한다면 조금이라도 낭비가 될 수 있습니다.

그래서 이 코드에서는 길이 0의 배열을 사용하여, 실제로 이 구조체에 동적 할당이 이루어질 때 원하는 길이만큼의 여분을 할당받아놓고, 단순히 `구조체의 마지막 원소`로서의 역할로 `contents` 배열을 사용하여 이 영역에 접근하는 방법을 선택하고 있습니다.

### 가변 개수 매크로 인자 ###
이 기능은 C99에도 있는 기능이지만, GCC에서는 이들을 좀 더 편리하게 사용하기 위한 문법들을 제공합니다. 표준대로라면 기본적인 사용법은 다음과 같습니다.

```c
#define debug(format, ...) fprintf (stderr, format, __VA_ARGS__)
```

GCC에서는 `__VA_ARGS__`와 같은 이름을 쓰기 원치 않는 사람들을 위해 다음과 같은 용법을 허용합니다.

```c
#define debug(format, args...) fprintf (stderr, format, args)
```

`...`에 `args`라는 이름을 붙여 보다 읽기 쉽게 작성할 수 있습니다.

그런데 이 용법에는 사소한 문제가 있는데, 반드시 `args`에 전달되는 인자가 필요하다는 점입니다. 이 자리에 빈 인자를 전달하는 것이 가능하므로, 다음과 같이 쓸 수는 있지만,

```c
debug("abc",);
```
이 경우 전처리 후 대체된 코드에서 콤마 뒤에 실제 인자가 있어야 하는데 존재하지 않으므로 역시 컴파일 에러가 발생하게 됩니다. 이런 경우를 대처하기 위해 다음과 같은 용법을 사용할 수 있습니다.

```c
#define debug(format, args...) fprintf (stderr, format, ##args)
```

이제 아래와 같이 사용하면 `##`에 의해 콤마가 아예 없어지므로 정상적인 컴파일이 가능해집니다.

```c
debug("abc");
```

### 범위 `case`문 ###
GCC에서는 다음과 같은 코드를 사용할 수 있습니다.

```c
#include <stdio.h>

int main(void)
{
	int n;
	scanf("%d", &n);
	switch (n)
	{
	case 1 ... 10:
		printf("between 1 and 10\n");
		break;
	case 11 ... 100:
		printf("between 11 and 100\n");
		break;
	default:
		printf("not in between 1 and 100\n");
		break;
	}
	return 0;
}
```

직관적이게도 입력받은 값이 [1,10] 범위일 때, [11,100] 범위일 때, 그 외의 경우를 나누어 처리하는 것을 `if` ~ `else`가 아닌 `switch` ~ `case`문으로 처리할 수 있게 해줍니다. 주의할 사항으로는 수들과 점 사이에 공백이 있어야 올바르게 인식한다는 점이 있습니다.

또한 아스키 문자들을 사용해서도 범위를 지정할 수 있습니다.

```c
case 'A' ... 'Z':
```

### 식별자 이름에 `$` 사용 ###
GCC에서는 식별자 이름에 `$`를 사용할 수 있습니다. 예를 들어 [이 문제](https://www.acmicpc.net/problem/1000)를 다음과 같이 풀 수 있습니다.

```c
#include <stdio.h>

int main()
{
    int $, $;
    scanf("%d%d", &$, &$);
    printf("%d\n", $+$);
    return 0;
}
```

### statement를 expression으로 사용 ###
GCC에서는 소괄호로 둘러싸인 compound statement를 하나의 expression으로 사용하는 것을 허용합니다. 누구나 한 번쯤 짜보는 (그러나 허점이 많은) `max` 매크로는 보통 다음과 같이 생겼습니다.

```c
#define max(a,b) ((a) > (b) ? (a) : (b))
```

이 코드의 문제점 중 하나는 `a`나 `b`의 평가가 두 번 이루어질 수 있다는 점인데, 매크로는 전처리 과정에서 그대로 코드를 대체하는 것에 불과하기 때문입니다. 이를 이 기능을 사용하여 해결해보면 다음과 같이 할 수 있습니다.

```c
#define maxint(a,b) \
  ({int _a = (a), _b = (b); _a > _b ? _a : _b; })
```

중괄호 내에 다수의 statement가 들어가 있으며, 그 중 마지막 statement의 값이 compound statement의 값이 되며 이 전체를  소괄호로 묶어 하나의 expression으로 사용이 가능합니다. 따라서 아래와 같은 코드를 사용할 수 있게 됩니다.

```c
int x = 1, y = 2;
int z = maxint(x, y);
```

### `typeof` ###
그러나 바로 전 문단의 코드는 한 가지 아쉬운 점이 있는데, 해당 매크로를 오로지 `int`형에 대해서만 사용할 수 있다는 점입니다. 이를 해결해줄 수 있는 다른 키워드를 GCC에서 같이 제공해줍니다.

`sizeof` 키워드와 유사한 용법을 가지고 있는 `typeof`는 이름 그대로 해당 값의 자료형을 얻어내는 데에 쓰입니다. ISO 표준을 따르는 옵션을 붙인 경우에는 `__typeof__`를 대신 사용할 수 있습니다. 예를 들어 전 문단의 `maxint` 매크로는 다음과 같이 더 일반화된 `max` 매크로로 대체할 수 있습니다.

```c
#define max(a,b) \
  ({ typeof (a) _a = (a); \
      typeof (b) _b = (b); \
    _a > _b ? _a : _b; })
```

이 함수를 이용하여 다음과 같은 코드를 작성할 수 있습니다.

```c
#include <stdio.h>

#define max(a,b) \
  ({ typeof (a) _a = (a); \
      typeof (b) _b = (b); \
    _a > _b ? _a : _b; })

int f(int x)
{
	printf("function called\n");
	return x * x;
}

int main(void)
{
	printf("%d\n", max(f(1), f(2)));
	return 0;
}
```

컴파일 시간에 `max` 매크로에 전달되는 값의 자료형이 `int`라는 것을 감지하여, 해당 매크로는 `int`형의 변수들을 선언하여 평가된 값들을 담은 뒤, 둘을 비교하도록 동작한 것입니다.

### 지역적이지 않은 goto문 ##
이 기능은 C 표준에 정의된 `setjmp`, `longjmp` 기능과 매우 유사하지만, 섞어쓸 수는 없는 기능입니다. 다음의 두 내장 함수로 구성되어 있습니다.

```c
int __builtin_setjmp (intptr_t *buf);
void __builtin_longjmp (intptr_t *buf, int val);
```

이 함수들을 사용하면 마치 서로 다른 함수들 사이에서 `goto`를 사용한 것과 같이 실행 흐름을 자의적으로 바꿀 수 있습니다. 단, 이 함수들은 GCC가 함수 호출 및 종료 시에 스택을 사용하여 레지스터를 저장하고 복원하는 기법으로서 내부적으로만 사용되는 함수이므로, 개발자의 직접적인 사용은 권장되지 않습니다.

### atomic 연산을 위한 함수들 ###
멀티스레드 환경에서는 어떤 연산들이 atomic[^2]하게 수행되어야 하는 경우들이 있습니다. 다음의 함수가 멀티스레드로 실행되었을 때 `cnt`의 증가 횟수를 예측할 수 없다는 것은 널리 알려진 사실입니다.

```c
int cnt = 0;

void *thread_main(void *arg)
{
	for (int i = 0; i < 1000000; i++)
		cnt++;
	return NULL;
}
```

그 이유는 `++`이라는 연산은 실제로는 '메모리에서 값을 레지스터에 읽어들이고', '그 값을 증가시키고', '다시 메모리에 쓰는' 여러 단계를 거치기 때문입니다. 이를 일반적으로 해결하는 방법은 lock을 걸어 이 critical section을 보호하는 것이지만, GCC에서는 이런 단순한 연산에 대해서는 보다 효율적으로 race condition을 방지할 수 있는 여러 내장 함수들을 지원합니다.

이전에는 'Intel Itanium Processor-specific Application Binary Interface'과 호환되는 `__sync_`로 시작하는 함수들을 주로 사용했었지만, 현재는 C++11에서 제시하는 메모리 모델에 대략적으로 맞게 동작하는 `__atomic_`으로 시작하는 함수들의 사용이 권장됩니다. 정수형 또는 포인터 형식에 대해 사용할 수 있습니다.

이 함수들의 특징으로는 memory order[^3]를 직접 설정할 수 있다는 점이 있으며, `__ATOMIC_RELAXED`, `__ATOMIC_CONSUME`, `__ATOMIC_ACQUIRE`, `__ATOMIC_RELEASE`, `__ATOMIC_ACQ_REL`, `__ATOMIC_SEQ_CST`의 6단계를 지원합니다. 각 함수마다 사용할 수 있는 order의 종류에는 제한이 있습니다.

몇 가지 atomic 함수들을 나열해보면 다음과 같습니다.

* `type __atomic_load_n (type *ptr, int memorder)`: 지정된 주소로부터 값을 읽어옵니다.
* `void __atomic_store_n (type *ptr, type val, int memorder)`: 지정한 주소에 값을 씁니다.
* `type __atomic_exchange_n (type *ptr, type val, int memorder)`: 지정한 주소에 값을 쓰고, 쓰기 전의 값을 반환합니다.
* `bool __atomic_compare_exchange_n (type *ptr, type *expected, type desired, bool weak, int success_memorder, int failure_memorder)`: 흔히 comapre-and-swap이라고 부르는 연산으로, `ptr`의 내용물이 `expected`의 내용물과 같으면 `ptr`이 가리키는 주소에 `desired` 값을 쓰고 `true`를 반환합니다. 같지 않다면 쓰기를 하지 않고 `false`를 반환합니다.
* `type __atomic_add_fetch (type *ptr, type val, int memorder)`: 원하는 값만큼을 더한 뒤, 그 값을 반환합니다. 자매품으로 sub, and, xor, or, nand 버전도 있습니다.
* `type __atomic_fetch_add (type *ptr, type val, int memorder)`: 원하는 값만큼을 더하고 더하기 전의 값을 반환합니다. 마찬가지로 자매품들이 있습니다.
* `bool __atomic_test_and_set (void *ptr, int memorder)`: `bool` 또는 `char`형에 대해 값이 `false`이면 `true`로 바꾸고 `true`를 반환합니다. 그렇지 않으면 `false`를 반환합니다.

### `__int128` ###
쓰인 그대로 128비트의 정수형이 지원되는 프로세서에서 그 기능을 사용하기 위한 자료형입니다. 안타깝게도 이들을 입출력하는 라이브러리까지 제공하지는 않기 때문에, 그를 위한 함수는 직접 만들어 써야 합니다.

### 10진법 부동소수점 자료형 ###
GCC는 10진법 부동소수점이 지원되는 프로세서에 대해 해당 자료형을 지원합니다. 고정소수점이란 C의 일반 부동소수점 자료형 `float`, `double`, `long double`과는 달리 표현 방법이 10진법으로 고정되어 있어[^4] 10진법에서의 소수점 연산에 대한 오차를 줄일 수 있다는 장점이 있습니다.

이 자료형의 종류로는 `_Decimal32`, `_Decimal64`, `_Decimal128`이 있는데, 역시 `__int128`과 마찬가지로 직접적인 입출력이 라이브러리로 지원되지 않기 때문에 함수를 직접 만들어 써야 합니다.

## 결론 ##
이번 글에서는 C와 C++에 수많은 기능을 추가하여 편의성을 높이기 위한 GCC의 노력을 살펴보았습니다. 제법 많은 것들을 설명한 것 같은데도, 여기에 나열한 확장 기능들은 사실 전체의 1%도 되지 않는 것 같을 정도로 GCC에는 방대한 기능들이 숨겨져 있습니다.

호환성을 생각한다면 이러한 기능에 너무 의존하지는 말아야 하겠지만, GCC만을 사용하는 프로젝트라면 다양한 확장 기능들에 대해 알고 있으면 유용하게 활용하여 더 편하게 작업을 하거나, 성능을 극대화시키는 등 이점을 볼 수 있는 경우가 많지 않을까 생각합니다.

## 참고 자료 ##
* https://gcc.gnu.org/onlinedocs/gcc/C-Extensions.html

[^1]: 반드시 구현할 필요는 없지만, 이러한 것이 자주 쓰인다는 것을 적어놓은 것입니다. 구체적으로 어떻게 구현되어야 하는지도 잘 명시하지 않습니다.
[^2]: 그 동작이 온전하게 수행되거나, 전혀 수행되지 않는 두 가지 경우만이 존재하는 것을 말합니다. 번역하여 '원자성'이라고도 하고, all-or-nothing으로 묘사하기도 합니다.
[^3]: 컴파일러는 실행 성능을 향상시키기 위해 메모리를 읽고 쓰는 연산의 순서를 바꿀 수가 있습니다. 그러나 이것이 멀티스레드 환경에서는 예기치 않은 동작을 초래할 수 있기 때문에 그 행동에 거는 제약 조건을 memory order라고 합니다.
[^4]: `float`, `double`, `long double`은 특정 진법을 사용해야 한다는 규정이 없습니다.
