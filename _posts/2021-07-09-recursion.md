---
layout: post
title:  "재귀 함수에 대한 이해"
date:   2021-07-09 22:29:19
author: djm03178
tags: recursion
---

## 개요 ##
재귀 함수는 알고리즘 문제 풀이뿐만 아니라 프로그래밍 전반에 있어 매우 중요한 기법 중 하나입니다. 여러 복잡해 보이는 문제들을 재귀 함수 하나면 손쉽게 구현할 수 있는 경우가 많기 때문에 매우 '강력하다'는 표현을 자주 쓰게 됩니다. 문제 풀이에서는 DFS를 구현하는 가장 기본적인 방법으로 널리 사용되기 때문에 최근에 핫한 코딩 테스트들에서도 사용 빈도가 매우 높습니다.

그러나 재귀 함수는 알고리즘 초보가 쉽게 벽을 느끼게 될 수 있는 기법이기도 합니다. 순차적으로 그냥 실행 흐름을 따라가면 되는 다른 코드들과는 달리, 재귀 함수는 코드의 어느 부분에서 어느 부분으로 언제 오고 가게 되는지, 현재 어떤 상태에 있는 건지 파악하는 것이 매우 어렵기 때문입니다. 이 글에서는 재귀 함수를 보다 쉽게 이해하는 방법을 설명하고, 어떤 문제를 해결하는 데에 재귀 함수를 사용할 수 있는지 예시를 통해 알아보도록 하겠습니다.

## 비유를 통한 이해 ##
흔히 재귀를 설명할 때 컴퓨터의 메모리 구조에서 호출 시마다 스택에 호출 정보가 push되고 종료될 때 pop되고... 하는 근본적인 원리부터 시작하고는 합니다. 그러나 이것은 개인적으로 재귀를 쓰는 이유를 이해하는 데에 적합한 방법이 아니라고 생각합니다. 컴퓨터가 재귀를 어떻게 실행하는지는 일단 제쳐두고, 조금 더 상위 수준에서, 재귀를 현실의 작업에 비유해서 재귀가 어떤 용도로 쓰이는지를 이해해봅시다.

프로그램이 실행되는 전체 과정을 공장의 생산 라인에서 여러 직원들이 제품을 생산하는 과정으로 비유해 보겠습니다. 그러면 프로그램의 입력은 곧 제품의 기초 재료가 되고, 소스 코드는 직원들의 매뉴얼에 비유할 수 있습니다. 각 직원은 하나의 함수를 담당한다고 가정하겠습니다. 그러면 프로그램이 실행되는 흐름은 다음과 같아집니다.

* 프로그램에 들어온 입력을 (생산 라인에 들어온 기초 재료를) 여러 함수들이 (여러 직원들이) 순차적으로 처리하여 (순차적으로 가공하여) 정답을 구해내는 (제품을 완성하는) 과정

그러면 이제 프로그램의 실행 과정에 재귀 함수가 포함되는 경우를 추가해 봅시다. 이를 생산 라인에 비유하기 위해, 직원을 다음과 같이 조금 더 명확히 재정의 해보겠습니다.

* 매뉴얼에 적힌 내용대로 **실제로** 일을 하는 주체

중요한 것은 소스 코드는 그저 매뉴얼에 불과하다는 사실입니다. 즉, 매뉴얼은 하나만 있지만 **같은 매뉴얼대로 작업하는 직원은 여럿이 될 수도 있습니다**. 이것이 재귀 함수를 이해하는 첫 단계입니다.

하나의 매뉴얼이라고 하더라도 세부적인 작업 내용은 정확히 그 직원이 이전 단계로부터 넘겨받은 재료의 상태에 따라 달라질 수 있습니다. 이렇게 이전 단계로부터 넘겨받은 재료는 곧 함수의 인자 (또는 전역 변수)가 됩니다.

즉, 재귀 호출은 다음과 같이 비유할 수 있습니다.

* 한 직원이 자신과 같은 매뉴얼에 따라 작업하는 다른 직원에게 특정 상태의 재료들을 건네며 작업을 요청하고 그 결과물을 돌려받는 과정

한 가지 더 눈여겨볼 점은 여기서는 오로지 순차적인 작업만을 가정하고 있기 때문에 직원 A가 직원 B에게 일을 넘겼다면 B가 일을 모두 마치고 결과물을 A에게 돌려주기 전까지 A는 아무것도 하지 않는다는 점입니다. 즉, A는 B가 최종적으로 만들어 낸 결과물만이 필요할 뿐 그 사이에 무슨 일이 일어나는지에 대해서는 전혀 신경쓰지 않아도 됩니다.

## 재귀의 특성 ##
지금까지 비유를 통해 알아본 재귀의 특성을 정리해 봅시다.

1. 재귀는 같은 일을 하는 함수끼리 상태만 달리해서 호출하는 것이다.
2. 재귀 호출된 함수가 무슨 일들을 했는지 (예를 들면 그 안에서 또 어떤 재귀 호출들이 있었는지)는 중요하지 않다. 그것이 어떤 결과를 돌려주는지만이 중요하다.
3. 재귀를 사용하는 함수는 반드시 재귀 호출을 하지 않는 경우 (기저 케이스)를 하나 이상 포함해야 한다. 그러지 않으면 재귀 호출이 무한히 발생하게 된다.

여기서 2번과 3번은 어디선가 많이 본 개념입니다. 바로 점화식입니다. 재귀 호출도 점화식과 아주 유사한 성질을 가지고 있고, 실제로 점화식을 직접적으로 코드로 표현하는 데에도 재귀만한 것이 없습니다. $A_{1}$이 특정한 값으로 정해져 있고 $A_{i}$가 $A_{i-1}$ 또는 더 이전의 항들에 대한 식으로 표현될 수 있는지만을 알면 굳이 일반항을 구하지 않아도 식의 정당성을 입증할 수 있습니다. 재귀도 마찬가지입니다. 호출하는 / 호출되는 재귀 함수 사이의 관계만 명확하게 정의하면 되고, 호출된 함수가 그 안에서 또 재귀 호출을 어떤 식으로 했는지를 보지 않아도 프로그램의 정당성을 보일 수 있습니다.

이를 저는 개인적으로 이렇게 표현합니다. **"믿음을 가지면 된다."** 어떤 재귀 함수가 왜 잘 동작하는지 일일이 호출 스택을 따라가면서 직접 눈으로 다 확인해봐야만 재귀 함수의 동작을 입증할 수 있는 건 아닙니다. 이 점화식을 올바르게 세웠다면, 나머지는 재귀 함수가 *알아서* 잘 해줄 것이라고 *믿으면* 됩니다.

## 재귀 함수의 주의사항 ##
단, 재귀 함수가 아무리 강력하더라도 아무 때나 이를 사용할 수 있는 것은 아닙니다. 제약 조건도 있으며, 반복문을 사용하는 것이 훨씬 깔끔한 경우도 많습니다.

### 사이클이 없어야 한다 ###
여기서 말하는 사이클은 재귀 호출이 연쇄적으로 이루어지는 과정에서 같은 '상태'를 가진 채로 호출되는 경우를 말합니다. 쉽게 말하면 `f(x)`가 다시 `f(x)`를 호출하거나, `f(x)`가 `f(y)`를 호출한 뒤 `f(y)`가 다시 `f(x)`를 호출하는 것입니다.

사이클이 없는지를 쉽게 판단하는 방법 중 하나는 기저 케이스 판별에 사용되는 인자나 전역 변수가 그 기저 케이스에 가까워지는 방향으로만 나아가는지를 확인하는 것입니다.[^1] 예를 들어 `f(i)`가 `f(i+1)`만을 재귀 호출하고 `i`가 `n`에 도달하면 종료하게끔 설정되어 있는 경우에는 인자가 더 큰 쪽에서 더 작은 쪽으로는 호출될 수 없기 때문에 사이클이 발생하지 않습니다. 또한 재귀 호출이 기저 상태에 도달하지 않을 수 있다면 무한히 계속해서 다른 상태로 나아가기 때문에 문제가 발생합니다.

다른 흔한 예시로 DFS를 재귀로 구현한 경우, 호출을 수행할 때마다 반드시 방문 체크된 정점의 개수가 증가하는 방향으로만 재귀 호출을 하게 되므로, 그리고 모든 정점이 방문된 경우 더 이상 재귀 호출을 하지 않을 것이므로 사이클이 존재할 수 없음을 알 수 있습니다. `f(x, y, z)`가 `f(x+1, y, z)`, `f(x, y+1, z)`, `f(x, y, z+1)`을 각각 재귀 호출하는 경우에도 역시 이전의 상태가 다시 호출되는 것이 불가능합니다.

### 단순 순차 작업은 반복문으로 ###
재귀가 유용해지는 경우는 하나의 함수 호출이 둘 이상의 재귀 호출을 할 가능성이 있거나, 재귀 호출의 결과물에 추가적인 작업을 해야 하는 경우입니다. 단순히 지금까지의 작업을 재귀 호출되는 함수에 넘겨주고 더 이상 아무런 작업을 하지 않는다면 굳이 재귀 함수를 작성할 필요가 없이 반복문을 사용하는 것이 깔끔하고 성능상으로도 이득입니다. 간혹 BFS도 재귀로 구현할 수 있다면서 아래와 같은 코드를 작성하는 사람들이 종종 있습니다.

```cpp
const int N = 100;

vector<int> adj[N];
bool visited[N];

void BFS(queue<int> q)
{
	if (q.empty())
		return;
	queue<int> q2;
	while (!q.empty())
	{
		int x = q.front();
		q.pop();
		for (int y : adj[x])
		{
			if (!visited[y])
			{
				visited[y] = true;
				q2.push(y);
			}
		}
	}
	BFS(q2);
}
```

이는 틀린 방법은 아니지만 재귀 함수의 이점을 살린 것으로는 볼 수 없습니다. 반복문의 기능을 그대로 재귀로 옮긴 것에 불과합니다. 재귀 호출을 제외한 함수 전체를 `while`문으로 감싼 뒤 루프의 마지막 부분에서 `q2`를 `q`에 넣어주는 것으로 똑같은 기능을 구현할 수 있습니다. 이렇게 일직선으로 재귀 호출이 이어지는 경우 여러 성능상의 저하도 발생하는데, 우선 재귀 호출 자체가 상당히 무거운 연산이며, [꼬리 재귀](https://ko.wikipedia.org/wiki/%EA%BC%AC%EB%A6%AC_%EC%9E%AC%EA%B7%80)가 이루어지지 않는 경우 재귀 호출의 깊이만큼 메모리 사용량도 계속해서 늘어나게 된다는 문제점도 있습니다.

## 완전 탐색 ##
재귀 함수의 동작 원리는 기본적으로 DFS에 기반한다고 볼 수 있습니다. 함수의 인자들과 전역 변수의 조합으로 만들어지는 각각의 '상태'가 그래프의 정점이 되며, 이 그래프를 깊이 우선으로 탐색해 나가는 것이 곧 재귀 호출입니다. 입력으로 명시적으로 주어진 그래프를 DFS로 탐색하는 기법은 별도로 공부하는 것으로 하고, 여기서는 암시적인 그래프를 완전 탐색하기 위해 재귀 함수를 활용하는 방법을 논의해 보겠습니다. 이를 연습하기 위한 가장 좋은 문제는 역시 [N과 M](https://www.acmicpc.net/workbook/view/2052) 시리즈입니다.

우선 [N과 M (1)](https://www.acmicpc.net/problem/15649) 문제를 보겠습니다. 이 문제의 입력은 그냥 `N`과 `M` 두 개의 정수뿐으로 어떻게 생각해도 그래프의 형태로는 보이지 않습니다. 그렇지만 문제의 내용에서 이미 우리가 탐색해야 할 그래프의 형태는 정해져 있습니다.

이 그래프의 각 정점을 결정할 '상태'를 구성하는 것으로 '지금까지 선택한 수들의 목록'을 지정할 수 있습니다. 편의를 위해서 각 수에 대해 방문 표시한 배열을 $V$라고 하고, '지금까지 선택한 수의 개수'를 저장하는 변수를 추가로 $c$라고 할 수 있습니다. 이 변수들은 전체 목록에 종속적이기 때문에 추가 '상태'를 만들어내지는 않습니다.

문제에서 요구하는 것은 `M`개를 뽑는 모든 경우를 출력하는 거니까, 전체 그래프에서 $c$가 `M`이 되게 하는 모든 상태에 방문을 해야 할 것입니다. 또한 이러한 상태들은 재귀 함수의 기저 상태이기도 합니다. 즉, 완전 탐색을 통해 이러한 기저 상태들에 모두 방문하도록 하는 것이 문제를 푸는 방법입니다.

우선 코드를 먼저 보고, 어떤 식으로 이를 구현하는지 확인해 보겠습니다.

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> a;
bool v[9];
int n, m;

void f(int c)
{
	if (c == m)
	{
		for (int x : a)
			cout << x << ' ';
		cout << '\n';
		return;
	}

	for (int i = 1; i <= n; i++)
	{
		if (!v[i])
		{
			v[i] = true;
			a.push_back(i);
			f(c + 1);
			a.pop_back();
			v[i] = false;
		}
	}
}

int main()
{
	cin >> n >> m;
	f(0);
}
```

선택한 수를 순차적으로 저장하는 배열은 `a`, 각 수에 대한 방문 표시는 `v` 배열에 하고 있고, 재귀 함수의 인자로는 현재까지 선택한 수의 개수를 세는 `c`만을 사용하고 있습니다. 기저 상태로는 `c == m`를 체크하여, 이 상태에 도달한 경우 지금까지 선택한 수를 순차적으로 모두 출력하게끔 합니다. 현재 상태에서 다음 상태로 넘어갈 때 `c`가 항상 증가하고 `m`이 되는 것이 기저 상태이기 때문에 사이클도 발생하지 않음을 알 수 있습니다.

사전순으로 앞서는 것부터 출력해야 하기 때문에, 순차적으로 뽑을 때 항상 작은 수부터 뽑아야 합니다. 그래서 반복문을 돌면서 `i`는 1부터 `n`까지 차례대로 증가하면서, 아직 뽑지 않은 수가 있다면 그것을 뽑는 상태로 탐색해보는 것을 반복해보면 됩니다. 그 이후에서 어떤 식으로 탐색할지는 신경쓰지 않고, 오로지 현재 상태에서 다음 상태로 넘어가는 과정만 정확하게 구현해주면, 이후 상태는 재귀 호출된 함수가 알아서 잘 처리할 것입니다. `f(c, v)`가 `c`와 `v`에 의해 만들어진 상태에 대한 답을 구해주는 함수라는 것을 정의했다면 그게 더 나아가 어떤 재귀 호출들을 하게 되는지 등은 몰라도 됩니다.

이와 같이 전체에서 특정 개수를 뽑는 모든 경우를 검사하는 형태의 완전 탐색은 코딩 테스트에서도 아주 빈번하게 출제되기 때문에 코딩 테스트를 준비한다면 비슷한 유형을 많이 연습해보는 것이 좋습니다.

## 탑-다운 DP (메모이제이션) ##
재귀를 많이 사용하는 또 다른 예시로는 탑-다운 동적 계획법이 있습니다. [동적 계획법 모델화하기](https://www.secmem.org/blog/2020/10/24/dp/) 글에서 서술한 바와 같이 DP의 상태들은 대체로 사이클 없는 방향성 그래프 (DAG)를 이루며, 이는 재귀 함수를 사용하기 위한 조건에 정확하게 부합합니다.

탑-다운 DP와 완전 탐색의 코드 구조상의 차이는 사실상 하나뿐입니다. 바로 메모이제이션의 유무입니다. 즉, 각 상태에 대한 답을 이미 구했다면 (이전에 같은 상태를 방문한 적이 있다면) 이전에 구한 답을 그대로 반환하는 메모이제이션을 수행하는 것이 탑-다운 DP입니다. 만일 재귀로 탑-다운 DP를 작성했는데 메모이제이션이 올바르게 수행되지 않았다면 최악의 경우 그대로 완전 탐색으로 이어지게 됩니다.

탑-다운 DP를 재귀로 구현하는 예시 문제로는 [파이프 옮기기 2](https://www.acmicpc.net/problem/17069)를 보겠습니다.

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int n;
ll dp[32][32][3];
int a[33][33];

inline bool g(int y, int x)
{
	return a[y][x] == 0;
}

inline bool h(int y, int x, int d)
{
	switch (d)
	{
	case 0:
		return g(y, x + 1);
	case 1:
		return g(y, x + 1) && g(y + 1, x) && g(y + 1, x + 1);
	case 2:
		return g(y + 1, x);
	}
	return true;
}

ll f(int y, int x, int d)
{
	if (y == n - 1 && x == n - 1)
		return 1;
	else if (y >= n || x >= n)
		return 0;
	ll &ret = dp[y][x][d];
	if (ret != -1)
		return ret;
	ret = 0;
	switch (d)
	{
	case 0:
		if (h(y, x, 0))
			ret += f(y, x + 1, 0);
		if (h(y, x, 1))
			ret += f(y + 1, x + 1, 1);
		break;
	case 1:
		if (h(y, x, 0))
			ret += f(y, x + 1, 0);
		if (h(y, x, 1))
			ret += f(y + 1, x + 1, 1);
		if (h(y, x, 2))
			ret += f(y + 1, x, 2);
		break;
	case 2:
		if (h(y, x, 1))
			ret += f(y + 1, x + 1, 1);
		if (h(y, x, 2))
			ret += f(y + 1, x, 2);
		break;
	}
	return ret;
}


int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);

	cin >> n;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			cin >> a[i][j];
	memset(dp, -1, sizeof dp);
	cout << f(0, 1, 0);
}
```

재귀로 문제를 풀 때에는 '상태'를 결정하는 것이 무엇인지 판단하는 것이 가장 중요합니다. 각 '상태'에서 끝점까지 도달하는 경우의 수를 구할 수 있도록 메모이제이션을 수행하고 싶은데, 이 '상태'를 단순히 눈에 보이는 공간적인 위치인 좌표만으로 결정하려고 하면 안 됩니다. 현재 좌표에서 파이프의 방향이 오른쪽인지, 아래쪽인지, 아니면 대각선인지에 따라서 앞으로 갈 수 있는 경우가 전혀 달라지기 때문입니다. 따라서 여기서는 '상태'에 '파이프의 현재 방향'까지 같이 고려를 해줘야 합니다.

기저 상태로는 파이프가 끝 지점에 도달한 경우와 격자 밖으로 나간 경우를 지정해줄 수 있습니다. 끝 지점에 도달한 경우 하나의 경우를 센 것이므로 1을 반환하고, 격자 밖으로 나간 경우 잘못된 경우이므로 0을 반환합니다. 그 외의 경우 이미 메가 된 상태라면 이전에 계산한 값을 그대로 반환해주고, 처음 방문한 상태라면 현재 파이프의 방향에 따라 갈 수 있는 다음 좌표 및 방향을 검사하여 가능한 경우를 모두 방문하고 더해주면 됩니다.

`f(y, x, d)`가 `(y, x)` 좌표에서 파이프가 `d` 방향일 때의 답을 구하는 함수라고 정의했다면 `f(y, x, d)`를 호출하면서 그 함수가 `f(y, x+1, d)`를 호출하게 되는지, 아니면 `f(y+1, x, 2)`과 `f(y+1, x+1, 1)`을 부르는지와 같은 것은 신경쓰지 않아도 됩니다. 그 부분은 재귀 호출된 함수가 알아서 잘 처리할 것이기 때문입니다. 이들을 그냥 믿으면 되고, 우리가 해야 할 일은 기저 상태에 대한 처리와 인접한 상태간의 연결 고리를 올바르게 만들어주는 것뿐입니다.

여기서 사이클이 없음을 확인하는 쉬운 방법은 `y+x`의 값이 항상 증가하고 있음을 보는 것입니다. 하나 이상이 충분히 커지면 기저 상태에 도달하게 되므로 이 재귀 호출은 항상 끝을 만나게 됩니다.

## 결론 ##
이상에서 재귀 함수를 비유를 통해 쉽게 이해하는 법을 제시하고 간단한 연습문제를 해설해 보았습니다. 재귀적으로 호출된 함수는 각각 독립적으로 자신이 맡은 '상태'에 대한 처리만을 한다는 것을 인지하고 자신의 역할에만 충실할 수 있도록 코드를 잘 작성해주면, 그 다음은 그냥 믿어주면 됩니다. 재귀 함수는 처음 보면 동작 방식을 이해하기 어렵지만, 재귀 함수를 어떤 곳에 사용할 수 있고 고려해야 할 점들이 무엇인지를 알고 나면 매우 강력한 도구가 될 수 있을 것입니다.

[^1]: 사이클이 없는 모든 경우가 이 방법으로 쉽게 판단될 수 있는 것은 아니지만 문제 풀이 과정에서는 대부분 이것으로 충분합니다.
