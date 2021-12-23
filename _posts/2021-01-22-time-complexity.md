---
layout: post
title:  "시간 복잡도를 고려한 코드 설계 전략"
date:   2021-01-22 11:00:23
author: djm03178
tags: time complexity
---

## 개요 ##
알고리즘 문제에는 여러 가지 제한이 주어집니다. 여기에는 입력으로 주어지는 값의 범위, 그 값들이 가지는 성질 등의 조건도 있지만, 대부분의 경우 메모리 제한과 **시간 제한**이 같이 주어집니다. 많은 문제들에서는 이 제한을 크게 고려하지 않고도 답을 찾아내는 로직만을 생각하여 코드로 구현하면 통과할 수 있지만, 일부 문제에서는 시간 제한 때문에 직관적인 해결책으로는 통과할 수 없는 경우도 존재합니다. 따라서 알고리즘 문제를 풀 때에는 코드를 작성하기 전후로 로직과 코드의 **시간 복잡도**를 계산하는 것이 중요합니다.

시간 복잡도는 알고리즘을 공부할 때 반드시 같이 배우게 되는 개념이지만, 실제로 문제를 풀 때 어느 정도의 시간 복잡도를 목표로 해야 하는지에 대한 감을 잡기가 어렵고, 특정 알고리즘의 전형적인 형태에서 벗어난 로직이 어떤 시간 복잡도를 가지는지 분석하는 것 또한 쉬운 일이 아닙니다. 이 글에서는 문제에 주어진 시간 제한과 입력값의 범위를 보고 시간 복잡도상 효율적인 풀이를 생각해내기까지의 과정을 진행해보며 어떤 전략을 사용하여 코드를 설계할 수 있는지 알아보겠습니다.

## 시간 복잡도와 시간 제한 ##
로직을 생각하기 전에 우선 어느 정도의 시간 복잡도의 코드가 통과 가능성이 있는지 생각해보아야 합니다. 오래된 자료들에서는 주로 이를 "1초에 1억 회 연산" 정도로 설명하지만, 최근에는 컴퓨터의 성능과 컴파일러의 최적화 기법이 아주 좋아져서 간단한 사칙연산, 대입문 등을 기준으로 **1초에 약 10억 회** 정도로 보는 것이 더 현실성이 있습니다. 그러나 이는 "더 빠른 문제 풀이 코드를 위한 상수 줄이기"([1](http://www.secmem.org/blog/2019/10/17/Constant/), [2](http://www.secmem.org/blog/2019/11/13/Constant-2/)) 글에서 설명한 것과 같이 연산의 무게에 따라서 같은 시간 동안 수행 가능한 연산의 수에 큰 차이가 발생할 수 있고, 시간 복잡도는 곱해지는 상수 등을 보통 무시하고 표기하기 때문에 주의가 필요합니다.

문제 출제자의 입장에서는 제한을 정할 때 다음의 두 가지를 고려하여 출제하게 됩니다.

1. 정해, 또는 충분히 통과시켜줘도 된다고 생각하는 시간 복잡도를 가진 코드는 어느 정도 비효율적이어도 통과
2. 통과해서는 안 된다고 생각하는 시간 복잡도를 가진 코드는 아무리 최적화를 해도 통과 불가

그래서 비록 컴퓨터가 1초에 10억 회 연산이 가능하다고 하더라도, $\mathcal{O}(n)$의 시간 복잡도가 정해인 문제를 $n \le 10^9$로 1초만에 수행하게끔 하는 경우는 거의 없습니다. 일반적으로는 그보다 훨씬 여유를 두고 제한을 설정하더라도 시간 복잡도가 더 높은 코드는 아무리 최적화를 해도 통과하기 어렵고, 더 느린 다른 언어(Java, Python 등)도 고려해야 할 경우도 있기 때문입니다. 가령 $n \le 10^6$으로 제한을 설정하더라도 $\mathcal{O}(n^2)$의 코드는 상수가 아무리 작아도 보통은 시간 내에 통과할 수 없을 것입니다. $\mathcal{O}(n\log{n})$과 같이 로그가 붙을만한 풀이가 존재하는 경우에는 상황이 조금 복잡해지지만, 대개의 경우 로그가 하나 붙는 것과 붙지 않는 풀이는 구분하기 힘들기 때문에 보통은 둘 다 충분히 허용할 수 있는 수준으로 $n$의 제한을 설정합니다.

다음은 1초의 시간 제한에서, 문제들에서 자주 볼 수 있는 제한과 의도된 풀이의 시간 복잡도의 예시들입니다.

* $n \le [10^9, 10^{18}]$: $\mathcal{O}(1)$, $\mathcal{O}(\log{n})$, $\mathcal{O}(\log^2{n})$ 등이 정해인 문제들입니다. 오래된 문제들에서는 $10^9$이 많이 사용되었지만 계산이 매우 빠르게 가능한 경우 $\mathcal{O}(n)$이 통과될 수 있는 경우가 많아 최근에는 대부분 8바이트 `long long` 범위를 적극 활용하여 $n \le 10^{18}$로 사용됩니다. 간혹 $\mathcal{O}(\sqrt{n})$와 같이 특이한 시간 복잡도를 원하는 경우 $n \le 10^{12}$와 같은 제한을 사용하기도 합니다.
* $n \le [10^5, 10^6]$: $\mathcal{O}(n\log{n})$의 시간 복잡도를 요구하는 문제에 사용됩니다. 오래된 문제들의 경우 주로 $n \le 10^5$를 많이 사용했으나, 최근에는 상수가 작은 $\mathcal{O}(n^2)$ 풀이가 존재하는 경우 1초 내에 통과되는 경우가 많아 $n \le 2 \cdot 10^5$으로 많이 사용하는 추세이고, 전체적으로 정해의 연산이 가볍고 타이트한 제한을 사용하려는 경우 $n \le 5 \cdot 10^5$이나 $n \le 10^6$까지도 사용됩니다. 로그가 붙는 것을 허용하지 않게 완전히 $\mathcal{O}(n)$ 풀이만을 원하는 경우 $n \le 10^8$ 정도의 제한이 사용되는 경우도 있습니다.
* $n \le [1000, 10000]$: 주로 $\mathcal{O}(n^2)$을 요구하는 문제에 사용됩니다. 이 제한 역시 이전에는 $n \le 1000$으로 사용한 경우가 많았지만 $n^3$이 10억밖에 되지 않기 때문에 $\mathcal{O}(n^3)$ 풀이가 통과되는 경우가 아주 많아[^1] 최근에는 대체로 최소 $n \le 5000$ 이상이 많이 쓰이고 있습니다. $n \le 1000$이나 $n \le 2000$은 $\mathcal{O}(n^2\log{n})$를 요구하는 문제에도 종종 사용됩니다.
* $n, m \le [1000, 10000]$: 위와 비슷하게 $\mathcal{O}(nm)$ 정도의 시간 복잡도를 요구하는 경우에 많이 사용됩니다. 다만, 격자판의 상태를 입력으로 주는 것과 같이 입력량도 $\mathcal{O}(nm)$인 문제에서는 입력 속도를 고려해서 최대 $n, m \le 3000$ 정도가 제한으로 많이 사용됩니다.
* $n \le [200, 500]$: $\mathcal{O}(n^3)$을 요구하는 문제에 많이 사용됩니다.
* $n \le [10, 20]$: $\mathcal{O}(2^n)$, $\mathcal{O}(n \cdot 2^n)$,  $\mathcal{O}(n!)$ 등의 지수 이상 시간 복잡도를 요구하는 문제에 주로 사용됩니다. 특히 $\mathcal{O}(n!)$의 경우에는 매우 빠르게 증가하기 때문에 $n$은 거의 $10$ 이하로 제한을 걸게 됩니다.

## 최악의 경우 시간 복잡도 ##
문제를 풀기 위한 로직을 생각할 때, *대체로 빠르게 실행될 것 같은* 코드는 사실 빠르지 않을 가능성이 높습니다. 가능하면 정확하게, 어려운 경우 대략적으로라도[^2] 시간 복잡도를 계산해서 윗 문단에서와 같이 문제의 시간 제한에 충분히 들어올 수 있는 정도의 연산량이 만들어질지 생각해보아야 합니다. 이는 시간 복잡도에 대해서도 마찬가지인데, *대부분의 경우에 낮은 시간 복잡도를 가지는* 풀이는 위험할 수 있습니다. 기본적으로는 **최악의 경우 시간 복잡도**를 항상 고려해서 풀이를 생각하도록 해야 합니다.

평균 시간 복잡도와 최악의 경우 시간 복잡도에 큰 차이가 있는 대표적인 알고리즘으로 퀵소트 (Quicksort)가 있습니다. 퀵소트는 평균 시간 복잡도가 $\mathcal{O}(n\log{n})$이지만, 최악의 경우에는 $\mathcal{O}(n^2)$입니다. 따라서 $n$이 10만 이상이 될 수 있는 문제에서 퀵소트를 사용하는 것은 적절하지 못한 선택입니다.[^3] 단, 모든 알고리즘에 대해 최악의 경우를 고려해야 하는 것은 아닌데, 다음과 같이 케이스를 나누어 볼 수 있습니다.

1. 평범한 데이터에 대해 쉽게 저격되는 알고리즘: 퀵소트의 경우 단순히 원소가 오름차순이나 내림차순으로 정렬되어 있기만 하더라도 그것이 곧 최악의 경우가 됩니다. 이와 같은 데이터는 출제자가 복잡한 생각을 하지 않고 몇 가지 극단적인 패턴만 고려하더라도 쉽게 넣어놓을 수 있는 데이터입니다. 이렇게 특정 코드를 저격하려고 하지 않더라도 최악의 경우 시간 복잡도가 만들어질 수 있는 알고리즘은 절대적으로 피해야 합니다.
2. 출제자가 쉽게 예측할 수 있는 저격 가능한 알고리즘: 이 경우는 문제의 데이터가 좀 더 철저하게 준비되어 있을 때 위험할 수 있습니다. 어떤 솔루션들이 가능할지 출제자가 미리 예측하고 인위적으로 특정 알고리즘에 대한 저격 데이터를 넣어놓는 경우로, 다익스트라를 정해로 하는 문제에 SPFA (Shortest Path Faster Algorithm)를 저격하는 데이터를 넣거나, C++의 `unordered_map`의 기본 해시 함수를 저격하는 것 등이 포함됩니다. 이러한 알고리즘들은 일반적인 문제에서 당장은 통과될 가능성이 높지만, BOJ와 같이 추후 데이터가 추가될 수 있는 경우 언제든 시간 초과로 변하게 될 위험성이 있다는 것을 인지해야 합니다.
3. 코드를 보고 저격이 가능한 경우: 이는 아주 특수한 경우로 Codeforces와 같이 핵 (hack)이 가능한 시스템에서 가능한 피해야 하는 코드입니다. 코드를 보기 전까지는 저격이 불가능하지만, 코드의 설계에 맞추어 그 코드를 최악으로 만드는 것이 가능한 경우 핵을 당할 수 있습니다. 물론 BOJ에서도 저격을 당할 수도 있지만, 누구에게도 그다지 득이 없기 때문에 그런 데이터가 추가될 염려는 거의 없고, 코드를 비공개로 설정하기만 해도 저격당할 수 없습니다.

일반적으로는 최소한 1과 2까지는 고려하여 풀이를 작성하는 것이 바람직합니다.

## 시간 복잡도의 분석 ##
시간 복잡도의 분석에는 많은 연습이 필요합니다. 간단한 코드에서는 명확하게 드러나기도 하지만, 많은 변수가 얽히면 분석이 어려워지고, 수학적인 이유로 분석하는 것 자체의 난이도가 높은 경우도 있습니다. 여기서는 몇 가지 기본적인 코드 형태에 대해 예시를 보며 어떤 방향에서 접근하여 분석을 시도할 수 있는지를 알아보겠습니다.

### 상수 시간의 코드 묶음 찾기 ###
상수 시간 복잡도 ($\mathcal{O}(1)$)를 가지는 코드의 묶음을 찾는 것은 시간 복잡도 분석의 첫 단계입니다. 예시로 다음의 C++ 코드를 보겠습니다.

```cpp
#include <iostream>
using namespace std;

int main()
{
	int a, b;
	cin >> a >> b;
	cout << a + b << '\n';
	cout << a - b << '\n';
	cout << a * b << '\n';
	cout << a / b << '\n';
}
```

위 코드는 두 수를 입력받고 사칙연산의 결과를 차례대로 출력합니다. 출력하는 줄 수가 여럿이지만, 어느 문장도 두 번 이상 실행되지 않고 오로지 한 번만 실행됩니다. 이와 같이 출력하는 문장이 코드상에 아무리 여러 줄 길게 존재하더라도, 그것들이 한 번만 실행된다면 시간 복잡도에 전혀 고려할 필요 없이 모두 상수 시간으로 계산할 수 있습니다.

그러나 이것이 꼭 코드에 루프가 없어야만 성립하는 것은 아닙니다. 입력 변수와 무관하게 특정 횟수 이하로만 실행된다면 여전히 상수 시간의 코드 묶음으로 볼 수 있습니다. 다음의 코드를 봅시다.

```cpp
#include <iostream>
using namespace std;

int main()
{
	int a, b;
	cin >> a >> b;

	for (int i = 0; i < 5; i++)
	{
		cout << a + b << '\n';
		cout << a - b << '\n';
		cout << a * b << '\n';
		cout << a / b << '\n';
	}
}
```

이 코드는 루프 다섯 번을 돌면서 같은 문장을 여러 번 실행하지만, 입력값을 어떻게 주더라도 그 횟수가 변하지 않습니다. 따라서 루프 안의 문장들은 여전히 한정된 횟수로만 실행된다는 것을 알 수 있고, 이 코드는 여전히 상수 시간 복잡도를 가집니다.

분기를 통해 특정 문장이 실행될 수도 있고 아닐 수도 있는 경우에도, 최대 횟수만 상수 횟수 이하로 제한이 된다면 똑같이 상수 시간 복잡도를 가지는 코드 묶음으로 볼 수 있습니다.

### 내장 / 라이브러리 함수의 시간 복잡도 ###
겉으로 보이는 루프가 없다고 해서 항상 상수 시간이 되는 것은 아닙니다. 사용하는 언어의 키워드나 내장 함수, 또는 라이브러리 함수 등은 제각각 시간 복잡도를 가지고 있습니다. 이를 고려하지 않고 시간 복잡도를 분석하는 경우 비효율적인 알고리즘을 설계하게 될 가능성이 높습니다. 다음의 예시를 보겠습니다.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main()
{
	int n;
	cin >> n;
	vector<int> a(n);
	for (int i = 0; i < n; i++)
		cin >> a[i];
	for (int i = 1; i <= n; i++)
	{
		if (find(a.begin(), a.end(), i) != a.end())
			cout << "1\n";
		else
			cout << "0\n";
	}
}
```
이 코드는 $n$을 입력받고 다시 `n`개의 수를 벡터 `a`에 입력받은 뒤, 1부터 `n`까지의 수 각각이 `a`에 존재하면 1을, 존재하지 않으면 0을 출력합니다. 언뜻 보기에는 루프를 도는 부분이 입력을 받는 부분과 1부터 `n`까지의 수를 찾기 위해 각각 루프를 도는 부분 둘뿐인 것처럼 보입니다. 하지만 이 코드에서 한 문장처럼 보이지만 사실은 $\mathcal{O}(1)$이 아닌 곳이 두 군데 존재합니다.

첫째는 `vector<int> a(n);`입니다. 이 코드는 단순히 `a`라는 벡터를 정의만 하는 것 같지만, 인자로 벡터의 크기인 `n`을 같이 주고 있습니다. 벡터의 크기를 같이 넘겨주는 경우 벡터는 그 크기만큼의 메모리를 미리 할당받아놓고 내용을 초기화하는데, 이 메모리를 할당받는 것도 $\mathcal{O}(n)$이며, 모든 원소를 초기화하는 것도 $\mathcal{O}(n)$입니다. 그래도 이 문장 자체는 코드 전체에서 한 번만 실행되므로 전체 시간 복잡도도 $\mathcal{O}(n)$이기 때문에 문제가 되지는 않습니다.

중요한 것은 둘째 부분인 `find(a.begin(), a.end(), i)`입니다. 이 코드도 언뜻 보기에는 `find`라는 라이브러리 함수를 한 번만 호출하므로 상수 시간인 것으로 착각할 수 있습니다. 하지만 이 함수의 역할이 무엇인지를 생각해보면 상수 시간에 수행되는 것이 불가능한 기능임을 알 수 있습니다. 벡터에서 어떤 원소가 내부에 존재하는지 찾으려면 벡터 전체를 순회하면서 하나씩 비교하는 수밖에 없는데, 최악의 경우에는 벡터의 모든 원소를 순회할 때까지 찾지 못하는 경우이므로 $\mathcal{O}(n)$ 시간이 소요됩니다. 이 코드를 둘러싸고 있는 `for`문 역시 $\mathcal{O}(n)$번 반복되므로, 이 코드의 시간 복잡도는 $\mathcal{O}(n^2)$이 됩니다.

코드의 시간 복잡도를 계산할 때에는 컴퓨터가 수행할 수 있는 기본 연산을 단위로 해야 합니다. 그래서 라이브러리에서 보다 상위의 기능을 간편하게 제공한다고 해서 그것을 하나의 연산으로 보면 안 되고, 그 라이브러리가 내부적으로 가지는 구조가 무엇인지 알고 그 구조상에서의 연산들은 각각 어떤 시간 복잡도를 가지는지를 함께 공부하는 것이 좋습니다. 이를 고려하지 못한 자주 보이는 케이스의 예시로는 다음과 같은 것들이 있습니다.

* (C) 길이가 $n$인 문자열에 대한 `strlen` 함수의 시간 복잡도는 $\mathcal{O}(n)$입니다. C에서 문자열의 길이를 판별하는 방법은 그 문자열의 시작부터 해서 한 글자씩 보면서 처음으로 널 문자 (`'\0'`)가 나타나는 위치를 찾는 것인데, 이는 위에서 본 벡터에서 특정 원소를 찾는 것과 같습니다. 이것이 특히 문제가 될 수 있는 경우는 반복문의 조건문에 `strlen`을 넣는 경우인데, 문자열의 길이만큼의 루프를 돌기 위해 `for (int i = 0; i < strlen(s); i++)`과 같은 문장을 사용할 경우 루프를 한 바퀴 돌 때마다 `strlen(s)`가 실행되므로 $\mathcal{O}(n^2)$의 시간이 걸리게 됩니다.
* (C++) `vector::erase(vector::begin())`, (Java) `ArrayList.remove(0)`, (Python) `list.pop(0)`은 모두 $\mathcal{O}(n)$입니다. 이는 이 자료구조들이 일반적인 배열로 이루어져있기 때문입니다. 배열이라는 것은 모든 원소가 연속적으로 배치되어 있고, 그래서 항상 상수 시간에 원하는 인덱스에 접근할 수 있도록 해주는 성질을 가지고 있습니다. 그런데 여기서 임의의 위치의 원소를 지우면 연속성이 깨지므로 그 뒤쪽의 원소들을 전부 한 칸씩 당겨오도록 설계되어 있는데, 첫 번째 원소를 지울 경우 당겨와야 하는 원소의 수는 기존에 들어있던 원소의 수 - 1입니다. 일반적인 배열로 구현된 자료구조 라이브러리에서 임의의 위치의 원소를 지우는 것은 반드시 피해야 합니다.
* (Python) `in list`도 $\mathcal{O}(n)$입니다. `in`이 단 두 글자밖에 되지 않아 아주 가벼울 것처럼 생겼지만 전혀 그렇지 않습니다. `list`가 일반적인 배열 구조이기 때문에 특정 원소를 찾으려면 배열 전체를 순회하는 방법밖에 없기 때문입니다. 반면에 `set`은 해시를 사용하기 때문에 일반적으로 $\mathcal{O}(1)$으로 볼 수 있지만, 경우에 따라 저격될 수 있고 이 경우 `in list`와 마찬가지로 $\mathcal{O(n)}$이 될 수 있다는 점을 명심해야 합니다.

위의 예시들은 모두 '배열'이라는 기초 자료구조의 성질과 연산의 동작 방식을 고려하지 않고 오로지 기능만을 생각했기 때문에 발생하는 문제들입니다. 알고리즘을 공부할 때 라이브러리의 사용은 매우 간편하지만, 내부적으로 사용하는 자료구조들에 대해서도 함께 공부하고, 확실하지 않은 것들은 각 언어의 레퍼런스 사이트를 참조해서 정확한 시간 복잡도를 아는 것이 매우 중요하겠습니다. 또한 자신이 그 자료구조를 직접 구현한다면 요청한 연산을 어떻게 코드로 표현할지 생각해보는 것도 많은 도움이 됩니다.

### 루프의 시간 복잡도 ###
다음은 루프가 있는 경우에 시간 복잡도를 계산하는 방법을 살펴보겠습니다. 루프에 대한 시간 복잡도 계산의 기본은 다음과 같습니다.

* 중첩된 루프끼리의 시간 복잡도는 곱한다.
* 중첩되지 않은 문장끼리의 시간 복잡도는 더한다.

간단한 예시 코드를 살펴보겠습니다.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main()
{
	int n, m;
	cin >> n >> m;
	vector<int> a(n);
	for (int i = 0; i < n; i++)
		cin >> a[i];

	int ans = 0;
	for (int i = 0; i < m; i++)
	{
		int x;
		cin >> x;

		for (int j = 1; j <= m; j++)
			ans += x;

		for (int j = 0; j < n; j++)
			ans += a[j];
	}
	cout << ans << '\n';
}
```
이 코드는 `n`개의 수를 벡터 `a`에 입력받고, 그 후 `m`개의 수를 각각 `x`에 입력받고 `x`를 `m`번만큼 `ans`에 더하고, 그 때마다 `a`의 모든 원소들을 `ans`에 더해주는 프로그램입니다. 그다지 의미는 없는 프로그램이지만, 컴파일러가 최적화를 수행하지 않고 루프를 정직하게 돌린다는 가정 하에 이 프로그램의 시간 복잡도는 다음과 같이 분석할 수 있습니다.

`n`개의 수를 입력받는 부분까지는 이전 문단의 코드와 같이 $\mathcal{O}(n)$입니다. 이후 나타나는 `for`문은 중첩되지 않고 나란히 실행되므로 시간 복잡도를 서로 더해야 합니다. `for`문 내부는 `x`를 입력받는 문장 $\mathcal{O}(1)$, `x`를 `m`번 `ans`에 더하는 루프 $\mathcal{O}(m)$, 그리고 `a`의 모든 원소를 `ans`에 더하는 루프 $\mathcal{O}(n)$으로 이루어져 있는데, 이들은 서로 중첩되지 않으므로 시간 복잡도를 더해야 합니다. 그리고 이들은 모두 바깥의 `for` 루프에 중첩된 상태이므로 각각 곱해져야 합니다.

따라서 이 코드의 시간 복잡도는,

$$\mathcal{O}(n) + \mathcal{O}(m) \times (\mathcal{O}(1) + \mathcal{O}(m) + \mathcal{O}(n)) \\
= \mathcal{O}(n) + \mathcal{O}(m) + \mathcal{O}(m^2) + \mathcal{O}(nm)$

으로 계산할 수 있으며, 점근 표기법에서는 최고 차항만을 고려해도 동일하므로, 보다 단축하여

$\mathcal{O}(m^2 + nm)$

으로 쓸 수 있습니다.

어떤 경우에는 이 코드처럼 온전하게 $m^2$, $nm$번의 루프를 돌지 않는 경우도 있습니다. 예를 들어 다음과 같은 코드에서는,

```cpp
for (int i = 0; i < n; i++)
  for (int j = i + 1; j < n; j++)
    cout << a[i] * a[j] << '\n';
```

`i`는 `n`번의 루프를 도는 것이 확실하지만, `j`는 `i`가 증가함에 따라 루프를 도는 횟수가 점점 줄어듭니다. 그렇다면 시간 복잡도도 `j`가 항상 0부터 `n` - 1까지 도는 것과 달라질까요? 그렇지는 않습니다. 이런 경우에 대한 시간 복잡도를 계산하려면 기본적인 수학 지식을 사용해야 합니다. 안쪽 루프는 처음에는 $n - 1$번을 돌고, 그 다음에는 $n - 2$번, 그 다음에는 $n - 3$번, ..., 그리고 마지막에는 $0$번의 루프를 돌게 됩니다. 이 모두를 수행해야 하는 코드이므로, 각 횟수를 모두 더해보면, 등차수열의 합 공식에 의해 $\cfrac{n(n - 1)}{2}$를 얻을 수 있고, 이 식의 최고 차항은 $n^2$이므로, 이 코드 덩어리 전체의 시간 복잡도도 $\mathcal{O}(n^2)$이 됩니다. 이와 같이 안쪽 루프의 반복 횟수가 바깥 루프의 변수의 값에 영향을 받게 되는 경우, 각각의 시간 복잡도를 곱하는 것이 아닌 바깥 루프가 반복될 때마다의 안쪽 루프의 반복 횟수 식을 구하고 모든 경우에 대한 횟수를 더하는 방식으로 전체 시간 복잡도를 구할 수 있습니다.

조금 더 특별한 상황으로는 루프가 1씩 증가하는 것이 아니라 몇 배씩 증가하는 경우가 있습니다. 다음의 코드를 보겠습니다.

```cpp
for (int i = 1; i <= n; i *= 2)
  cout << i << '\n';
```

이 코드에서는 `i`가 `n` 이하인 동안 계속 두 배씩 증가하고 있는데, 이는 이 루프의 반복 횟수는 $\mathcal{O}(\log{n})$임을 의미합니다. $i$가 지수꼴로 빠르게 증가하기 때문에, 특정 수까지 도달하는 데에 걸리는 시간은 그 수의 로그에 비례합니다. 2배가 아니더라도 임의의 상수배로 증가한다면 모두 같은 로그 시간 복잡도로 계산할 수 있는데, 이는 로그의 밑만이 달라지는 것이기 때문에 로그의 성질상 상수배의 차이이기 때문입니다.

### 각 원소에 대한 연산 횟수로 시간 복잡도 계산하기 ###
이제 조금 더 복잡한 상황을 보겠습니다. 탐색 알고리즘의 대표적인 형태인 DFS와 BFS는 정점의 수가 $V$개, 간선의 수가 $E$개인 그래프에 대해서 중복 방문을 하지 않을 때 $\mathcal{O}(V+E)$임이 잘 알려져 있습니다. 그런데 이들을 구현한 코드만을 봐서는 왜 이 알고리즘들이 그런 시간 복잡도를 가지게 되는지 이해하기 어렵습니다. 여기서 사용할 수 있는 방법이 각 원소에 대한 연산 횟수를 세는 것입니다.

DFS를 예시로 들었을 때, 일반적인 DFS 함수의 형태는 다음과 같습니다.

```cpp
const int N = 100005;
vector<int> adj[N];
bool visited[N];

void dfs(int idx)
{
	visited[idx] = true;
	for (int x : adj[idx])
		if (!visited[x])
			dfs(x);
}
```

`adj` 배열의 각 원소는 각 정점과 연결된 다른 정점의 목록을 가진 벡터이고, 각 정점의 방문 여부를 `visited` 배열에 체크하여 DFS를 수행하는 코드입니다. 이제 한 정점에서 `dfs` 함수를 호출하여 탐색을 시작하면, 연결된 모든 정점을 연쇄적으로 재귀호출을 통해 방문하게 될 것입니다.

이 코드의 시간 복잡도를 분석하기 위해 이 상황에서 '원소'로 볼 것들로는 '정점'과 '간선'이 있습니다. `visited` 배열을 통해 중복 방문을 막는 코드이므로, 각 '정점'에 대한 방문은 최대 한 번만 이루어질 것입니다. 즉, `dfs` 함수의 호출 횟수는 $\mathcal{O}(V)$입니다. 그러나 각 `dfs` 함수 실행 시 내부의 `for`문이 도는 횟수는 일정하지 않습니다. 각 정점에 연결된 간선의 수가 서로 다를 수 있기 때문입니다. 따라서 이 경우 '간선'에 대한 연산 횟수는 별도로 계산하여 각 간선을 확인하는 횟수의 합을 따로 구하는 것이 합리적입니다. 간선은 하나의 정점에서 다른 정점으로 이어지는 것이고, 어떤 간선을 확인하는 연산은 그 간선의 시작 정점을 방문했을 때에 이루어집니다. 그런데 각 정점이 최대 한 번만 방문된다고 했으므로, 각 간선도 최대 한 번만 체크하게 됩니다. 따라서 최악의 경우에도 모든 간선을 한 번씩 체크하여 $\mathcal{O}(E)$의 시간만이 소요되고, 이는 '정점'과는 별도로 시간 복잡도를 계산한 것이므로 두 시간 복잡도를 더한 $\mathcal{O}(V+E)$가 전체 시간 복잡도가 됩니다.

이 계산 과정은 BFS에도 동일하게 적용할 수 있습니다. 중복 방문을 허용하지 않는다면 큐에 들어가는 원소 (정점)는 최대 $\mathcal{O}(V)$개이고, 각 정점이 큐에서 나올 때마다 그 정점에 연결된 간선들만을 확인하므로 각 간선 역시 최대 한 번만 확인됩니다. 따라서 모든 정점과 모든 간선이 한 번씩 확인되는 경우가 최악의 경우이고 이때의 시간 복잡도는 $\mathcal{O}(V+E)$가 됩니다.

이와 같은 계산법은 동적 계획법 코드에도 비슷하게 적용할 수 있습니다. [동적 계획법 모델화하기](http://www.secmem.org/blog/2020/10/24/dp/) 글에서 다루었던 것처럼 동적 계획법의 상태는 일반적으로 DAG 형태로 표현할 수 있으며, 각 상태를 정점으로 두고 상태간의 연결을 간선으로 보면 각 정점을 확인하는 횟수와 각 간선을 확인하는 횟수를 서로 개별적으로 구한 뒤 더하여 전체 시간 복잡도를 구할 수 있게 됩니다.[^4]

### 내 코드를 공격해보기 ###
지금까지 시간 복잡도를 분석하는 기본적인 테크닉들에 대해 살펴보았는데, 실제로 문제를 풀 때 작성하는 코드들은 이보다 훨씬 복잡하게 얽혀 있어 여전히 분석하기가 어려운 경우가 많습니다. 이럴 때 사용할 수 있는 방법 중 하나는 **자신의 코드가 가장 오래 걸릴 것 같은 케이스**를 생각해보는 것입니다. 다음의 코드를 보겠습니다.

```cpp
#include <iostream>
#include <set>
using namespace std;

int main()
{
	int n;
	cin >> n;
	set<int> s;
	for (int i = 0; i < n; i++)
	{
		int x;
		cin >> x;
		while (s.find(x) != s.end())
			x++;
		s.insert(x);
	}
	for (int x : s)
		cout << x << '\n';
}
```

이 코드는 `n`개의 수를 입력받는데, `x`에 입력받을 때마다 `x` 이상이면서 지금까지 `s`에 포함되지 않은 가장 작은 수를 찾아 `s`에 추가합니다. `s`는 `set`이므로 `insert`, `find` 등의 연산이 모두 $\mathcal{O}(\log{n})$입니다. 여기까지 계산해보았을 때, 중요한 것은 안쪽 루프인 `while`문이 총 몇 번을 돌게 될까입니다.

입력되는 수의 범위가 넓고, 랜덤으로 수가 주어진다면 이 `while`문은 대체로 매우 적은 반복 횟수 내에 루프를 탈출하게 될 것임을 쉽게 짐작할 수 있습니다. 하지만 이러한 코드가 효율적으로 동작하는 것은 평균적인 경우일 뿐이고, 항상 최악의 경우를 생각해야 합니다. 이 `while`문이 아주 많이 실행되게 만들려면 입력되는 `x`마다 지금까지 `s`에 포함된 모든 수를 전부 `find`로 찾게 만들면 될 것입니다. 이를 달성하는 쉬운 방법은 같은 원소 (예를 들면 1)를 `n`번 입력하는 것입니다. 그러면 처음 입력된 1은 바로 `s`에 들어가겠지만, 그 다음 입력되는 1은 이미 `s`에 있으므로 2까지 가야 루프를 탈출할 것이고, 그 다음 입력되는 1은 3까지 가야 하고, 그 다음 입력되는 1은 4까지 루프를 돌며, `n`번째 입력되는 1은 `n`번째 루프에서야 탈출하게 될 것입니다. 그러면 이 `while`문의 총 시간 복잡도는 $\mathcal{O}(n^2)$이 되고, 루프를 돌 때마다 $\mathcal{O}(\log{n})$의 연산이 수행되므로 총 $\mathcal{O}(n^2\log{n})$이 됩니다.[^5] 이렇게 자신의 코드를 공격하는 케이스를 만들다가 최악의 경우의 시간 복잡도를 찾아낼 수 있는 경우도 있습니다.

## 마치며 ##
알고리즘 문제의 해결은 단순히 답을 찾는 것만이 아니라 그 답을 빠른 시간 내에 효율적으로 찾아내는 것을 포함하며, 시간 복잡도를 낮추는 것은 그의 첫 단계입니다. 시간 복잡도가 비효율적인 로직으로 코드를 설계하면 아무리 수정에 수정을 거듭해도 시간 내에 통과되지 못할 수 있기 때문에, 처음부터 철저한 시간 복잡도 분석을 통해 생각한 로직이 문제의 시간 제한 내에 수행될 수 있는 수준인지 파악한 뒤 코드 작성에 돌입하는 것이 중요하다고 할 수 있곘습니다.

[^1]: $n \le 2000$도 1초 내에 실행되는 코드가 많으며, 심지어는 $n \le 3000$도 상수나 컴파일러의 최적화 가능성에 따라 1초 내에 실행되는 경우가 간혹 있습니다.
[^2]: "적어도 ~는 넘지 않는다" 정도의 계산을 할 수 있으면 됩니다.
[^3]: 대부분의 언어의 표준 라이브러리에서 제공하는 정렬 함수는 최악의 경우 $\mathcal{O}(n\log{n})$을 보장하므로 그대로 사용하면 됩니다. 예외로 Java의 `Arrays.sort`에 primitive type 배열을 전달하는 경우 퀵소트를 사용한다는 것과, C의 `qsort` 함수가 표준상으로는 시간 복잡도에 대한 규정을 하지 않고 있다는 것 (그래도 대개의 경우 사용해도 문제가 없는) 정도가 있습니다. 라이브러리를 사용하지 못하게 하는 코딩 테스트 등에서는 상대적으로 구현이 쉬운 병합 정렬을 추천하며, 자신이 있다면 힙정렬을 사용해도 됩니다.
[^4]: 사실, 대부분의 그래프에서는 간선의 개수가 정점의 개수와 비슷하거나 더 많기 때문에 간선의 개수만 세더라도 시간 복잡도 분석에 무리가 없습니다.
[^5]: 여담으로, 이와 같은 동작을 입력되는 수의 범위가 크더라도 union-find를 사용하여 $\mathcal{O}(n\log^2{n})$이나 조금 더 낮은 시간 복잡도로 효율적으로 수행하는 방법이 존재합니다.