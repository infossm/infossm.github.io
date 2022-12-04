---
layout: post
title:  "Segment Tree Beats와 Kinetic Segment Tree"
date:   2022-12-04
author: koosaga
tags: [algorithm, icpc, competitive-programming]
---
# Segment Tree Beats와 Kinetic Segment Tree
이 글에서는 Kinetic Segment Tree라는 새로운 세그먼트 트리를 소개한다. 어떠한 원소가 Kinetic하다는 것은 시간에 따라서 움직인다는 것으로, 쉽게 말해 그 원소가 일차함수거나 다항함수라는 것이다. 세그먼트 트리도 대회에 자주 나오고, Kinetic한 원소도 대회에 자주 나오니 (대표적으로 컨벡스 헐 트릭), 그것의 조합 역시 익혀보면 도움이 될 것이다. 또한, Kinetic한 원소와 전혀 상관이 없는 문제들에서도 Kinetic한 성질이 파악된다는 점에서 착안하여 (대표적으로 CHT를 사용한 DP 최적화), 이 Kinetic Segment Tree가 어떻게 응용될 수 있는지 역시 생각해 보면 좋을 것이다.

이 자료구조에 대해서는 예전 코드포스 글을 통해서 한번 들어보고 메모만 해 놓았으나, 최근에 시간을 내서 제대로 공부해보았다. 작성 시점 기준으로 Kinetic Segment Tree의 경우 많은 사람들에게 생소한 자료구조가 아닐까 생각이 든다. 새로운 자료 구조에 관심이 많다면 읽어보는 것을 추천한다.

원론적인 자료구조 문제들의 경우 풀이가 아주 어려운 경우가 많아서, 보통 직접 생각하기 보다는 다른 자료들을 보면서 배우고 익히는 경우가 많다. 하지만 이러한 문제들은 풀이가 어려운 만큼 제대로 이해했다면 여러가지 개념을 배울 수 있다는 장점이 있다. 이 글의 목표는 Segment Tree Beats에 대한 심도 있는 이해를 바탕으로 Kinetic Segment Tree를 배우며, 자료구조의 장단점을 익히고 자료구조를 사용하여 풀 수 있는 문제가 무엇인지를 익힌다.

## 1. Segment Tree Beats
Segment Tree Beats는 일반적인 Lazy Propagation으로는 풀리지 않는 구간 쿼리들을 해결하게 해주는 기법으로, Lazy propagation의 응용이라고 볼 수 있다. [HDU 5306 Gorgeous Sequence](http://acm.hdu.edu.cn/showproblem.php?pid=5306) 라는 문제를 옛날에 외국인한테서 들었는데, 분명히 단순한 구간 쿼리인데도 일반적인 Lazy propagation으로는 잘 풀리지 않아서 신기하다고 생각했다. IOI 2016 국가대표 교육에서 이 문제를 조교 분들과 같이 오랫동안 논의를 해도 도저히 풀 수가 없었고, 풀이도 중국어인데다가 이해가 어려워서 잠시 접어둔 기억이 있다.

6년이 지난 지금은 백준 온라인 저지에 있는 [같은 문제](https://www.acmicpc.net/problem/17474) 가 200명 이상에게 풀렸고, Segment Tree Beats를 잘 설명한 좋은 자료들도 많다. 고급 알고리즘 중에서는 잘 알려진 축에 속한다고 볼 수 있을 것이다. 사실 나도 풀이를 배운 건 2020년 정도라서 그때는 이미 볼만한 글이 여럿 있었다. 난 이 3개를 보고 공부한 것 같다.
* [Segment Tree Beats (rdd6584)](https://github.com/infossm/infossm.github.io/blob/7571d3d0ac66f7dd60f35cc4e826d20995ca1875/_posts/2019-10-19-Segment-Tree-Beats.md)
* [A simple introduction to "Segment tree beats" (jiry_2)](https://codeforces.com/blog/entry/57319)
* [Segment tree beats (Errichto)](https://www.youtube.com/watch?v=wFqKgrW1IMQ)

그래서 굳이 다시 설명할 필요가 있나 싶기는 하지만, 증명을 제대로 다룬 글이 많지 않아서 간단하게 복습하는 느낌으로 다시 짚고 넘어간다.

위에서 언급한 [수열과 쿼리 26](https://www.acmicpc.net/problem/17474) 문제를 예시로 들자. 만약에 합을 구할 필요가 없이, 구간 갱신만 할 경우의 코드는 다음과 같을 것이다. `maxv` 는 구간 최댓값을 뜻한다.

```cpp
void update(int s, int e, int ps = 1, int pe = n, int p = 1, int v) {
	if (e < ps or pe < s)
		return;
	if (s <= ps && pe <= e) {
		tree[p].maxv = min(tree[p].maxv, v);
		return;
	}
	lazydown(p); // lazydown() 함수는 p번 노드의 Lazy propagation 값을 자식으로 전파
	int pm = (ps + pe) / 2;
	update(s, e, ps, pm, 2 * p, v);
	update(s, e, pm + 1, pe, 2 * p + 1, v);
	pull(p); // pull() 함수는 p번 노드의 자식 값을 토대로 p번 노드의 값을 재구성
}
```
보다시피 $A[i] = min(A[i], v)$ 라는 갱신이 주어졌을 때 구간의 최댓값은 위와 같은 방식으로 관리를 할 수 있다. 하지만 합을 구하는 쿼리가 주어졌을 때는 이야기가 다르다. 위 업데이트를 통해서 구간에 있는 합이 얼마나 바뀌었는지는 관리하기 어렵다.

한 가지 해결 방법은, 구간 업데이트를 포기하는 것이다. 업데이트되는 구간의 길이가 모두 1이면, 합의 변홧값을 추적할 수 있다. 이 경우의 코드는 다음과 같다. `sum` 은 구간 합을 뜻한다.

```cpp
void update(int s, int e, int ps = 1, int pe = n, int p = 1, int v) {
	if (e < ps or pe < s)
		return;
	if (ps == pe) {
		// 길이 1 구간의 최댓값은 그냥 원소의 값이니까 쉽게 계산 가능
		if (tree[p].maxv > v) {
			tree[p].sum -= (tree[p].maxv - v);
			tree[p].maxv = v;
		}
		return;
	}
	/* ... */
}
```
이 코드의 장점은, 3번 쿼리를 지원한다는 것이다. 단점은, 느리다는 것이다. 구간 쿼리를 점 쿼리로 모두 바꿨으니, update 함수의 시간 복잡도는 $O(n)$ 이다. 여기서 관찰할 수 있는 것은, 갱신 조건이 `s <= ps && pe <= e` 와 같이 강할 경우 알고리즘은 빠르게 종료하지만 할 수 있는 연산의 자유도가 낮고, 갱신 조건이 `ps == pe` 와 같이 약하다면 알고리즘이 느리지만 할 수 있는 연산의 자유도가 높다. Segment Tree Beats는 여기의 **적절한 중간선** 을 찾아서 알고리즘의 복잡도를 해치지 않으면서 자유도를 늘리는 기법이다. 그리고 그 중간선이 되게 기발한데, 다음과 같다.

```cpp
void update(int s, int e, int ps = 1, int pe = n, int p = 1, int v) {
	if (e < ps or pe < s)
		return;
	if (s <= ps && pe <= e) {
		if (tree[p].maxv <= v) {
			// 갱신 안 함
			return;
		}
		// tree[p].maxv > v
		if (tree[p].smax < v) {
			tree[p].sum -= (tree[p].maxv - v) * tree[p].maxCnt;
			tree[p].maxv = v;
			return;
		}
		// tree[p].smax >= v, tree[p].maxv >= v 일 경우 종료 안 함
	}
	/* ... */
}
```
`smax` (second max) 는 구간의 최댓값보다 **작은** 원소들의 최댓값이고, `maxCnt` 는 구간의 최댓값의 등장 횟수이다. 즉, 구간에서 `maxCnt + 1` 번째로 큰 원소가 `smax` 가 된다. 이 알고리즘이 항상 올바른 답을 찾는다는 사실은 어렵지 않게 이해할 수 있다. 중요한 것은 이 알고리즘이 왜 빠른지에 대한 증명이다. 이 부분에 있어서는 구간의 서로 다른 수를 통한 쉬운 증명, 그리고 *태그* 라는 것을 정의하는 어려운 증명이 있다. 두 증명 모두 숙지하는 것이 좋다.

**쉬운 증명.** $f([s, e])$ 를 구간 $[s, e]$ 에 있는 서로 다른 수의 개수라고 하자. 세그먼트 트리 상의 모든 구간 $[s, e]$ 의 $f([s, e])$ 합을 토대로 관찰할 것이다. 초기에 이 값은 $T(n) = T(n/2) + O(n)$ 이니 $O(n \log n)$ 이다. 이제 갱신 과정을 살펴보자.
* `s <= ps && pe <= e` 를 만족하는 노드들 중:
  * `tree[p].maxv <= v or tree[p].smax < v` 를 만족하는 노드들은 $f([s, e])$ 가 변하지 않으며, 그 서브트리에 있는 노드들에도 마찬가지다.
  * `tree[p].maxv >= v && tree[p].smax >= v` 를 만족하는 노드들은 $f([s, e])$ 가 최소 하나 감소한다. `maxv != smax` 인데 둘 다 `v` 로 합쳐질 것이기 때문이다.
* `e < ps or pe < s` 를 만족하는 노드들은 고려할 필요 없다.
* 둘 다 아닌 노드들은, 서브트리에 $v$ 라는 새로운 수가 하나 등장할 수 있으니 $f([s, e])$ 가 최대 하나 증가한다.

매 쿼리에 *둘 다 아닌 노드* 가 $O(\log n)$ 개 있으니 초기화 및 갱신 과정에서 $f([s, e])$ 의 합은 $O((n + q) \log n)$ 만큼 증가했다.

갱신 쿼리는 $O(\log n)$ 개의 노드, 추가로 `tree[p].smax >= v && tree[p].maxv >= v` 인 노드들 전부를 방문한다. `tree[p].smax >= v && tree[p].maxv >= v` 인 노드를 한번 방문할 때마다, $f([s, e])$ 의 합은 최소 1 감소한다. 즉, 모든 쿼리를 통틀어서 저러한 노드의 방문 횟수는 $O((n + q) \log n)$ 으로 한정된다. $\blacksquare$

**어려운 증명.** 최댓값을 *태그* 라는 방식으로 표현하자. 다음과 같은 조건을 만족하는 노드들에는 *태그* 라는 정수를 붙일 것이다.
* 이 노드가 루트이다.
* 이 노드의 서브트리 최댓값과, 이 노드의 부모 노드 서브트리 최댓값이 다르다.

이 때 붙는 *태그* 는 해당 노드의 서브트리 최댓값이다. 세그먼트 트리의 노드들에 최댓값을 저장할 때, 같은 최댓값을 가진 노드를 하나의 연결 컴포넌트로 묶어주고, 컴포넌트의 루트에만 값을 저장하는 식으로 압축한다고 보면 된다. 어떠한 노드의 두 번째 최댓값은, strict subtree (본인 제외 서브트리) 의 태그 최댓값이다. $f([s, e])$ 를, 노드 $[s, e]$ 에 있는 서브트리의 태그 수 합이라고 하자. 초기 이 값은 $O(n \log n)$ 이다.

이제 업데이트를 관찰하자. 구간 업데이트를 할 때, 기본적으로 업데이트한 노드들에 각각 태그를 하나씩 달게 될 것이다. 노드의 개수가 $O(\log n)$ 이고 깊이가 각각 $O(\log n)$ 이니 ($f([s, e])$ 가 늘어나는 노드는 태그가 새로 생긴 노드의 조상 수와 동일하다.) 여기서 $O(\log^2 n)$ 만큼 퍼텐셜이 증가한다. 이제 두 번째 최댓값이 현재 업데이트되는 값 이상인 (그래서 재귀적으로 내려가야 하는) 경우를 생각해 보자. 이 경우 해당 두 번째 최댓값을 주는 태그는 명백히 사라지게 될 것이다. 즉, 재귀적으로 내려갈 때마다 퍼텐셜이 하나 감소한다. 고로 $f([s, e])$ 의 합은 $O((n + q \log n) \log n)$ 만큼 증가하고, 매 쿼리가 나이브하게 내려갈 때 이 값이 1 이상 감소한다. 고로 시간 복잡도가 $O((n + q \log n) \log n)$ 이 된다. $\blacksquare$

몇 가지 짚고 넘어갈 부분이 있다:
* 이 증명을 조금만 수정하면 $O((n + q) \log n)$ 을 증명할 수 있다 - 자세한 것은 맨 위의 *A simple introduction to "Segment tree beats"* 글을 참고하라. 편의상 여기서는 $O(\log n)$ 이 붙는 방식으로 설명했지만 이 증명은 실제로는 쉬운 증명의 상위 호환이다.
* 이해하기 조금 더 간단한 방법은 $f([s, e])$ 에 대해서 생각하지 않고 그냥 **전체 태그의 수** 에 대해서 생각하는 것이다. 태그 하나를 떼기 위해서는 $O(\log n)$ 의 시간이 사용된다. 부모에서 쭉 내려와야 하기 때문이다. 태그는 처음에 $n$ 개 있고, 매 쿼리마다 $O(\log n)$ 개 추가된다. 고로 총 $O(n + q \log n)$ 번 추가됨이 명백하다. 태그를 제거하기 위해서 Naive하게 내려가는 과정은, $k$ 개의 태그를 제거했다면 최대 $k \log n$ 의 시간을 소모한다. 고로 $O((n + q \log n) \log n)$ 이 된다.

이 증명을 사용하면, [수열과 쿼리 29](https://www.acmicpc.net/problem/17477) 와 같이 요구하는 쿼리의 종류가 많아져도 문제가 없다. 예를 들어
* $A[i] = min(A[i], Y)$
* $A[i] = A[i] + X$
와 같은 쿼리를 지원할 수 있다. $A[i] = A[i] + X$ 연산 역시 새로 추가한 태그가 명백히 $O(\log n)$ 개이기 때문이다. 서로 다른 수가 아니라 *태그* 라는 개념을 사용할 경우 더 다양한 연산에 맞게 증명할 수 있다.

여담으로, $f([s, e])$ 라는 값을 보통 **퍼텐셜 함수** (potential function) 이라고 자주 부르며, 어려운 알고리즘 분석에서는 자주 나온다. 퍼텐셜 함수를 쓰는 가장 쉬운 예시 중 하나가 세그먼트 트리 비츠인 것 같다.

## 2. Kinetic Segment Tree (Kinetic Tournament)
이 단락은 [다음 글](https://codeforces.com/blog/entry/82094)을 참고하여 작성되었다.

Kinetic Segment Tree (KST) 라는 새로운 자료구조를 소개한다. 좁게 말해서, Kinetic Segment Tree는 초기 시간 $t = 0$ 을 가지며, 다음과 같은 쿼리를 지원하는 자료구조이다.
* `update(i, a, b):` $(A[i], B[i]) = (a, b)$ 를 배정한다.
* `query(s, e)` $min_{i \in [s, e]} (A[i] \times t + B[i])$ 를 계산한다.
* `heaten(T):` 시간이 $t \le T$ 일 때, $t = T$ 로 둔다.

넓게 볼 때 Kinetic Segment Tree는 꽤나 유동적이다. 일단 관리하는 함수가 일차함수일 필요는 없고, 그냥 *두 서로 다른 함수의 대소 관계가 $t$ 에 따라 $O(1)$ 번 정도만 바뀐다* 정도만 가정해도 크게 무리는 없다. Heaten이라는 개념도 전역적이지 않고, 구간만 heaten한다는 느낌으로 생각해도 괜찮다. 일단 글 내에서는 위에 적은 정의로만 생각을 하고, 위와 같은 일반화는 연습 문제에서 따로 다루는 것으로 하자.

위의 좁게 본 KST만 생각하였을 때, 알고리즘은 일부 경우에서 몇가지 잘 알려진 자료구조를 대체할 수 있다. 예를 들어, 만약에 모든 쿼리가 단조 증가한다면, KST는 구간 쿼리를 지원하는 [Li-Chao Tree](https://blog.myungwoo.kr/137) 라고 생각할 수 있다. 표로 그 기능을 비교하면, 다음과 같다:

|  | Fully dynamic? | 구간 쿼리 | 쿼리 단조성 | 함수 종류 |
| -- | -- | -- | -- | -- |
| Li-Chao Tree | 삽입만 | X | 필요 없음 | 대소관계가 $t$ 에 따라 $O(1)$번 정도만 바뀌는 모든 함수 |
| Kinetic Segment Tree | 삽입 + 삭제 | O | 필요함 | 대소관계가 $t$ 에 따라 $O(1)$번 정도만 바뀌는 모든 함수 |

기술적으로, Kinetic Segment Tree는 Segment Tree Beats랑 유사한 점이 많다. 다른 세그먼트 트리와 유사하게, 리프 노드에는 각 함수 $A[i] x + B[i]$ 를 저장하고, 리프가 아닌 노드들에 대해서, 서브트리에 있는 노드들 중 $A[i] t + B[i]$를 최소로 하는 함수를 저장한다. $t$ 가 고정되어 있는 상태에서, 이는 쉽게 비교 가능하고, 구간 쿼리 및 업데이트 역시 문제가 없다. 고로, Kinetic Segment Tree의 `update` 와 `query` 함수는, 다른 세그먼트 트리와 동일하게 $O(\log n)$ 시간에 가능하다.

이제 $t$ 가 증가할 때 자료구조의 내부 정보를 맞춰주는 `heaten` 함수에 대해서 살펴보자. 어떠한 리프가 아닌 노드는 서브트리의 두 함수를 비교하여, 그 중 현재 시간 $t$ 를 기준으로 값이 작은 것을 취한다. 이 대소 관계가 역전되는 $t$ 초과의 최초 시간을 $insec(v, t)$ 라고 할 때, $melt[v]$를 $v$ 의 서브트리에 있는 모든 $insec(v, t)$ 의 최솟값이라고 정의하자. 이 경우, $melt[v]$ 가 현재 시간 이하인 모든 노드들을 루트에서 돌면서, bottom-up으로 우선 순위를 다시 계산해 준다.

알고리즘의 설명은 이것이 전부이다. 이 알고리즘 역시 항상 올바른 답을 찾는다는 사실은 어렵지 않게 이해할 수 있는데, 알고리즘의 시간 복잡도를 분석해 보자.

**Theorem 1.** `update` 쿼리가 없다고 가정할 때, Kinetic Segment Tree의 `heaten` 함수는 총 $O(n \log^2 n)$ 번의 연산을 필요로 한다.
**Proof.** 어떠한 구간 $[s, e]$ 에 대해서, 이 구간에 있는 직선들만을 가지고 Lower envelope를 만들어 보자. Lower envelope를 이루는 직선은 최대 $e - s + 1$ 이고, 최솟값을 주는 직선이 바뀌는 횟수는 $e - s$ 번 이하일 것이다. 노드 $v = [l, r]$ 에 대해 $f(v)$ 를, 구간 $[l, r]$ 에서 최솟값을 주는 직선이 바뀌는 *남은 횟수* 라고 정의하고, $\Phi(v)$ 를, $v$ 의 서브트리의 $f(v)$ 합이라고 정의하자. 구간의 길이가 $n$ 이라면, $f(v)$ 는 $O(n)$ 의 크기고, $\Phi(v)$ 는 $O(n \log n)$의 크기이다. 고로, KST를 처음 만든 시점의 $\Phi(v)$ 의 크기 합은 $O(n \log^2 n)$ 이 된다. `heaten` 함수가 어떠한 노드 $v$ 를 방문할 때, $v$의 서브트리에 있는 어떤 노드의 교점 하나가 소모될 것이니, $\Phi(v)$ 가 1 감소한다. 고로 `heaten` 함수의 방문 횟수는 $\Phi(v)$ 의 크기 합 이하이며 이는 $O(n \log^2 n)$ 이다.
**Remark.** 위 분석은 퍼텐셜을 정석적으로 사용해서 읽기 좀 딱딱하다. 이렇게 이해하면 쉬울 것 같다. 각 노드의 Lower envelope의 크기가 $O(n)$이니, 우리가 처리하게 될 교점의 총 개수는 $O(n \log n)$ 개이고, 교점 하나를 갱신하기 위해서는 루트로 가는 경로에 있는 모든 노드들을 방문해야 하니, 교점 하나를 처리하는 데 $O(\log n)$ 이 된다. 고로 이를 곱하면 $O(n \log^2 n)$ 이 된다.

**Theorem 2.** `update` 쿼리가 있을 경우, Kinetic Segment Tree의 `heaten` 함수는 총 $O(n \log^2 n \alpha(n))$ 번의 연산을 필요로 한다. $\alpha(n)$ 은 Inverse Ackermann Function이다.
**Proof.** 시간 축으로 보았을 때, Update 쿼리는 KST가 다루는 객체를 *직선* 이 아닌 *선분* 으로 바꾼다고 생각할 수 있다. 각 업데이트 쿼리가 현재 있는 직선을 파괴하여 오른쪽 끝점을 만들고, 새로운 직선의 왼쪽 끝점을 만든다고 생각할 수 있기 때문이다. 고로, $n$ 개의 선분들을 가지고 Lower envelope를 만들었을 때, Lower envelope에서 최솟값을 주는 선분이 최대 몇번 바뀌는지에 따라서 복잡도가 달라진다. $n$ 개의 선분으로 Lower envelope를 만들었을 때, 최솟값을 주는 선분이 바뀌는 횟수는 $O(n \alpha(n))$ 번 이하이고, 실제로 $\Omega(n \alpha(n))$ 번 바뀌는 Construction이 존재한다. 자세한 내용은 [Davenport–Schinzel Sequences and Their Geometric Application](http://www.cs.tau.ac.il/~michas/dssurvey.pdf) 라는 글을 찾아보면 좋다. 고로, Theorem 1의 증명을 그대로 사용하면 위와 같은 결과를 얻는다.

**Theorem 3.** Kinetic Segment Tree가 다루는 객체가 일차함수가 아니라, 두 서로 다른 객체가 최대 $s$ 번 교차할 수 있는 함수라면, `heaten` 함수는 총 $O(\lambda_{s + 2}(n) \log^2 n)$ 번의 연산을 필요로 한다. $\lambda_{s}(n)$ 은 Davenport–Schinzel sequence of order s의 $n$ 번째 항이다.
**Theorem 3.1.** Kinetic Segment Tree의 `update` 함수가 insertion 형태의 업데이트만을, 혹은 deletion 형태의 업데이트만을 한다면, `heaten` 함수는 총 $O(\lambda_{s + 1}(n) \log^2 n)$ 번의 연산을 필요로 한다. 여기서 "빈 함수" 는 $f(x) = \infty$ 를 생각하면 된다.
**Theorem 3.2.** Kinetic Segment Tree의 `update` 함수를 호출하지 않는다면, `heaten` 함수는 총 $O(\lambda_{s}(n) \log^2 n)$ 번의 연산을 필요로 한다.
**Remark.** 작은 $s$에 대한 Davenport-Schinzel 수열의 값은 다음과 같다 ([Wikipedia](https://en.wikipedia.org/wiki/Davenport%E2%80%93Schinzel_sequence)):
* $\lambda_0(n) = 1$
* $\lambda_1(n) = n$
* $\lambda_2(n) = 2n-1$
* $\lambda_3(n)  = 2n\alpha(n) + O(n)$
* $\lambda_4(n) = O(n2^{\alpha(n)})$
* $\lambda_5(n) = O(n2^{\alpha(n)}\alpha(n))$

즉, $s = O(1)$ 일 경우 Davenport-Schinzel Sequence는 실질적으로 $O(n)$ 이라고 생각해도 된다. 정확한 계산은 아니지만, $\alpha(n) \le 4$ 라고 생각하고 써보면 $\lambda_s(n) = 2^s n$ 근처로 나오는 것 같다.

Theorem 1, 2, 3의 Worst case가 나오게 되려면, 세그먼트 트리의 *모든 노드* 에 많은 교점이 존재해야 하며 이 교점들을 heaten 함수가 전부 열거해야 한다. 사실 $O(n \log^2 n \alpha(n))$ 의 반례를 실질적으로 KST에 구성할 수 있는지도 의문이다. $\alpha(n)$ 이라는 함수 자체가 크지 않으니, 실질적으로는 `update` 쿼리 유무와 무관하게 $O(\log^2 n)$ 정도라고만 생각해도 충분할 것 같다.

## 3. 구현 및 KST를 사용한 문제 해결
### 3.1. Convex Hull Trick 풀기
[BOJ 12795. 반평면 땅따먹기](https://www.acmicpc.net/problem/12795) 문제를 예시로 하여 설명한다. 반평면 땅따먹기 문제는 $ax + b$ 형태의 선분이 삽입될 때, 주어진 쿼리 $x$ 에 대해 $max(ax + b)$ 를 빠르게 계산하는 문제이다. 일반적인 Convex Hull Trick만으로는 이 문제를 해결할 수 없고, 흔히 이 문제를 해결하는 방법은 4가지가 알려져 있다.

* Li-Chao Tree 사용: $O(Q \log Q)$.
* Set을 사용하여 선분의 upper envelope 관리 (LineContainer): $O(Q \log Q)$.
* Bentley-Saxe 기법을 사용하여 Convex Hull Trick을 incremental하게 수정: $O(Q \log^2 Q)$.
* 오프라인으로 쿼리를 처리한 후, 일반적인 Convex Hull Trick을 세그먼트 트리로 관리: $O(Q \log^2 Q)$.

이 문제를 Kinetic Segment Tree를 사용하여 오프라인으로 $O(Q \log^2 Q)$ 에 처리할 수 있다. 모든 선분을 초기에 입력받은 후 선분에 대한 Kinetic Segment Tree를 만들자. 각 쿼리는, 선분의 prefix에 대해서 $ax + b$ 의 최댓값을 묻는 쿼리가 된다. 쿼리들을 $x$ 가 증가하는 순서대로 정렬한 후 처리하면 Kinetic Segment Tree를 사용할 수 있다.

Kinetic Segment Tree를 사용하여 BOJ 12795를 해결한 [본인의 코드](https://gist.github.com/koosaga/5a15cc4e51c3ba0de93e911bb6882ec4)이다. (update 코드를 짜고 안 썼다는 것을 포함해서) 구현이 그렇게 어렵지는 않다.

그 외 Kinetic Segment Tree를 사용해서 풀 수 있는 문제들은 다음과 같다:
* [Facebook Hacker Cup 2020 Round 2 Problem D: Log Drivin' Hirin'](https://www.facebook.com/codingcompetitions/hacker-cup/2020/round-2/problems/D)
* [Yosupo Judge: Line Add Get Min](https://judge.yosupo.jp/problem/line_add_get_min)

사실 위 예시들 모두 다른 자료구조를 사용해서 비슷하거나 더 빠른 시간에 풀 수 있고, 다른 자료구조가 KST에 비해서 코딩량이 엄청나게 많은 것도 아니라서 아주 인상적인 예시는 아니다. KST로만 풀 수 있는 직선 관리 문제를 만들 수 있기는 할 텐데, 어쨌든 KST가 강한 형태의 문제는 다른 종류라고 생각하며 이는 아래에 설명한다.

### 3.2. 구간 `heaten` 연산
이제 설명하게 될 연산은 KST의 *heaten* 연산을 구간 쿼리에 대해서 지원하는 것이다. 구체적으로, 길이가 같은 두 배열 $[a_1, \ldots, a_n], [b_1, \ldots, b_n]$ 다음과 같은 쿼리를 지원하는 자료구조를 구성할 것이다:
* 구간 최댓값: $max_{i \in [s, e]} b_i$
* 구간에 *heaten*: $b_i := b_i + t \times a_i$ for $i \in [s, e], t > 0$.

KST의 원리를 사용해서 이 연산을 지원하는 것은 어렵지 않다. 결국 *순서가 꼬이는 부분이 있으면 Naive하게 들어가서 다 풀어준다* 가 기본 원리이기 때문이다. 구간 최댓값의 경우 기존과 완전히 동일하게 하면 된다. 구간 Heaten 연산의 경우, 대응되는 구간들 중 $melt[v] \le t$ 인 구간들을 모두 탐색하면서 순서를 다 바꿔주면 되며, 만약 $melt[v] > t$ 라면 해당 구간에 대해서는 Lazy propagation으로 처리할 수 있다. 만약에 Naive하게 처리하는 부분이 heaten 연산에 의해서 너무 많이 늘어나지만 않으면, 결국 기존의 KST와 시간 복잡도도 크게 차이가 나지 않을 것이다. 실제로도 크게 차이가 나지 않는데, 그러한 이유를 아래에 설명한다.

**Theorem 4.** 위 쿼리를 Kinetic Segment Tree를 사용하여 처리하면 $O((n + q \log n) \log^2 n)$ 시간에 해결할 수 있다.
**Proof.** Heaten 연산이 전체 구간에 대해서만 적용된다고 가정하자. 각각의 리프 노드에 대해서 이 리프의 **영역** 을, 현재 이 리프의 $b_i$ 값이 최대인 노드들의 집합이라고 하자. 리프 $x$ 의 **영역** 은 $x$ 를 포함하고, $x$의 부모 쪽으로 가는 경로를 이룸을 알 수 있다. Heaten 연산으로 풀어주는 일이 있을 때 두 서브트리 중 큰 서브트리가 *역전* 되는 일이 있다. 즉, 두 리프 $x, y$ 에 대해서 $y$ 번 리프가 $x$ 번 리프보다 좋은 값을 얻게 되는 것이다. 이 경우 $x$ 번 리프는 본인의 영역을 잃는다. 이러한 일이 일어났다는 것은 $a_x < a_y, b_x < b_y$ 라는 것이기 때문에, $x$ 번 리프가 다시 해당 영역을 되찾는 일은 없다는 것을 관찰하자: 무조건 $y$ 번 리프의 값이 더 크다. 고로

* 초기 영역의 크기 합은 $O(n)$ 이다.
* 각 리프는 최대 $O(\log n)$ 만큼의 영역을 얻을 수 있다.
* Heaten 연산은 한 리프의 영역을 무조건 줄이며, 이 과정은 비가역적이고, 이 줄이는 연산마다 $O(\log n)$ 의 시간을 소모한다.
* 모든 리프는 최대 $O(n \log n)$ 만큼의 영역을 잃을 수 있다. 영역을 잃는 연산은 $O(\log n)$ 의 시간을 소모하니 $O(n \log^2 n)$ 시간에 문제가 해결된다.

이제 Heaten 연산을 구간에만 적용한다고 하자. 구간 안에 완전히 포함되는 노드의 경우, 연산의 특성상 여전히 *역전* 의 가능성은 없다. 구간에 전혀 포함되지 않는 서브트리의 경우, 아무 일도 일어나지 않는다. 즉 구간과 부분적으로 겹치는 $O(\log n)$ 개의 노드에 대해서만 변화가 있을 수 있다. 이 노드들의 영역이 감소하고 있었다 하더라도, 이번의 업데이트를 통해서 루트 위까지 올라갈 기회가 허용된다. 고로 $O(\log n)$ 개의 리프가 $O(\log n)$ 개의 영역을 얻게 되니, 리프는 최대 $O((n + q \log n) \log n)$ 만큼의 영역을 얻는다. 위와 같은 논리로 $O((n + q \log n) \log^2 n)$ 시간에 문제가 해결된다. $\blacksquare$

여담으로 Theorem 4는 Theorem 1을 함의한다. Segment Tree Beats의 "쉬운 증명"과 "어려운 증명" 의 관계처럼, 같은 증명이지만 이 쪽이 조금 더 어렵고 조금 더 많은 것을 증명할 수 있다. 또한 Theorem 4의 연산은 Heaten 연산에 대해서 많은 것을 가정하지 않기 때문에, 이 외에도 몇 가지 구간 연산들을 더 지원하는 것이 가능하다. ($t < 0$ 인 경우의 Heaten 연산은 증명이 불가능하다. 서브트리 *안밖* 에서만 역전이 있어야 한다. $t < 0$ 의 경우 서브트리 안에서 순서가 바뀐다.)

위와 같은 방법으로 다음 문제들을 해결할 수 있다.
* [Codeforces 1178G. The Awesomest Vertex](https://codeforces.com/contest/1178/problem/G)
* [Codeforces 573E. Bear and Bowling](https://codeforces.com/contest/573/problem/E)
* [BOJ. 꺾이지 않는 마음 3](https://www.acmicpc.net/problem/26144)
