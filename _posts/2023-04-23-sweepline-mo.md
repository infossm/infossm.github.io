---
layout: post
title:  "sweepline MO"
date:   2023-04-23 23:00
author: edenooo
tags: [algorithm, data-structure]
---

## 개요
다음 문제를 생각해 봅시다.

### Static Range Inversions Query

길이가 $N$인 수열 $A = (A_1, A_2, \dots, A_{N})$가 주어지면, 다음 $Q$개의 쿼리를 수행해야 합니다. $(1 \leq N \leq 10^5, 1 \leq Q \leq 10^5, 0 \leq A_i \leq 10^9)$

> $1 \leq l \leq r \leq N$을 만족하는 $l,r$이 주어지면, $l \leq i < j \leq r, A_i > A_j$ 를 만족하는 $(i,j)$쌍(=inversion)의 개수를 출력한다.

이 글에서는 다음의 내용들을 다룰 것입니다.

1. Static Range Inversions Query 문제를 기존의 MO's Algorithm으로 $O(N \sqrt Q \log N)$에 해결하는 방법

2. sweepline MO 알고리즘 소개

3. sweepline MO를 이용해 위 풀이의 시간복잡도를 $O(N (\sqrt N + \sqrt Q) + Q)$ 로 개선하는 새로운 방법

4. sweepline MO로 풀 수 있는 다른 문제들



## MO's Algorithm

MO's Algorithm을 소개하는 좋은 글[**(링크)**](https://infossm.github.io/blog/2019/02/09/mo's-algorithm/)이 S/W멤버십 블로그에 이미 작성되어 있으며 제가 이전에 작성한 글[**(링크)**](https://infossm.github.io/blog/2021/07/19/distinct-value-query/)에서도 소개되어 있으므로, 이 글에서 추가로 설명하지는 않겠습니다.



## MO's Algorithm을 이용한 Static Range Inversions Query

MO's Algorithm을 Static Range Inversions Query 문제에 적용하겠습니다.

- $C[x] = [l,r]$ 구간에 값 $x$가 등장하는 횟수

현재 구간 $[l,r]$에 대해 위와 같은 정보를 관리합시다.

$[l,r] \rightarrow [l,r+1]$로의 상태 전이가 일어나면 $r+1$번째 원소가 추가되면서 $l \leq i \leq r$와 $A_i > A_{r+1}$를 만족하는 $i$의 개수만큼 inversion이 증가합니다. 다시 말해 $C[A_{r+1}]$이 $1$만큼 증가하고 $\sum_{A_{r+1} < x \leq 10^9} C[x]$만큼 inversion이 증가하므로, 수열 $A$의 원소들을 $[1,N]$ 범위로 좌표 압축하고 $C$ 배열을 펜윅 트리로 관리하면 상태 전이의 시간복잡도는 $O(\log{N})$이 됩니다.

- $[l,r] \rightarrow [l,r+1]$
- $[l,r] \rightarrow [l-1,r]$
- $[l,r] \rightarrow [l,r-1]$
- $[l,r] \rightarrow [l+1,r]$

위의 4가지 상태 전이를 모두 비슷한 방식으로 처리할 수 있고 MO's Algorithm에서의 왼쪽 포인터와 오른쪽 포인터의 이동 횟수의 총합이 $O(N \sqrt Q)$번이므로, 최종 시간복잡도는 $O(N \sqrt Q \log N)$이 됩니다.



이 방법으로 Library Checker의 [Static Range Inversions Query](https://judge.yosupo.jp/problem/static_range_inversions_query) 문제를 해결하는 코드는 다음과 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

#define INF 1234567890
#define ll long long

const int SZ = 316; // MO에서 사용할 block의 크기
int N, Q;
int A[100001];
ll res[100001];

// 펜윅 트리
int fen[100001];
int Sum(int idx)
{
	int ret = 0;
	for(; idx; idx &= idx-1)
		ret += fen[idx];
	return ret;
}
void Add(int idx, int val)
{
	for(; idx<=N; idx+=idx&-idx)
		fen[idx] += val;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> Q;
	vector<int> com;
	for(int i=1; i<=N; i++)
	{
		cin >> A[i];
		com.push_back(A[i]);
	}

	// 좌표 압축
	sort(com.begin(), com.end());
	com.erase(unique(com.begin(), com.end()), com.end());
	for(int i=1; i<=N; i++)
		A[i] = lower_bound(com.begin(), com.end(), A[i]) - com.begin() + 1;

	// MO's Algorithm으로 쿼리를 처리할 순서를 재배열
	vector<array<int, 3> > q;
	for(int i=1; i<=Q; i++)
	{
		int l, r;
		cin >> l >> r;
		l++; // 1-based, closed interval
		q.push_back({l, r, i});
	}
	sort(q.begin(), q.end(), [&](const array<int, 3> &a, const array<int, 3> &b){
		if (a[1]/SZ != b[1]/SZ) return a[1]/SZ < b[1]/SZ;
		return a[0] < b[0];
	});

	// 오프라인 쿼리
	ll sum = 0;
	int l = 1, r = 0;
	for(auto [L,R,idx] : q) // [l,r] -> [L,R]로의 전이
	{
		while(r < R)
		{
			r++;
			Add(A[r], 1);
			sum += (r-l+1) - Sum(A[r]);
		}
		while(L < l)
		{
			l--;
			Add(A[l], 1);
			sum += Sum(A[l]-1);
		}
		while(R < r)
		{
			sum -= (r-l+1) - Sum(A[r]);
			Add(A[r], -1);
			r--;
		}
		while(l < L)
		{
			sum -= Sum(A[l]-1);
			Add(A[l], -1);
			l++;
		}
		res[idx] = sum;
	}

	for(int i=1; i<=Q; i++)
		cout << res[i] << "\n";
	return 0;
}
```

![](/assets/images/edenooo/sweepline-mo/fenwick.png)

800ms 정도의 시간에 해결할 수 있습니다.



## sweepline MO

### 아이디어

이전 풀이의 시간복잡도에 $O(\log N)$이 붙는 이유는 왼쪽 포인터나 오른쪽 포인터를 한 칸 이동할 때마다 펜윅 트리에서 업데이트와 쿼리를 모두 수행하기 때문입니다. 만약 업데이트의 횟수를 줄일 수 있다면, 펜윅 트리보다 업데이트가 더 느리지만 쿼리가 더 빠른 자료구조를 대신 사용해서 전체 시간복잡도를 개선할 수 있습니다. sweepline MO는 쿼리에 성립하는 특수한 성질을 관찰해서 업데이트의 횟수를 $O(N)$회로 줄이는 알고리즘입니다.

sweepline MO를 설명하기 위한 예시로, Static Range Inversions Query 문제의 기존 MO's Algorithm 풀이에 sweepline MO를 어떻게 적용할 수 있는지를 알아보겠습니다.

- $f(l,r,p) = [l,r]$ 구간의 $A_p$ 초과 값의 개수
- $g(l,r,p) = [l,r]$ 구간의 $A_p$ 미만 값의 개수

위와 같은 함수 $f,g$를 정의하면 $[l,r] \rightarrow [l,r+1]$로의 상태 전이에서 inversion의 증가량은 $f(l,r+1,r+1)$처럼 작성할 수 있고, 다른 상태 전이들도 $f$나 $g$에 대한 수식으로 비슷하게 표현할 수 있습니다. 따라서 $f,g$ 함수만 계산할 수 있다면 모든 상태 전이에서의 inversion의 변화량을 알아낼 수 있습니다.

- $F(r,p) = [1,r]$ 구간의 $A_p$ 초과 값의 개수
- $G(r,p) = [1,r]$ 구간의 $A_p$ 미만 값의 개수

그런데 함수 $f,g$는 $f(l,r,p) = f(1,r,p) - f(1,l-1,p), \ g(l,r,p) = g(1,r,p) - g(1,l-1,p)$라는 성질을 만족해서, 위처럼 새로운 함수 $F,G$를 정의하면 $f(l,r,p) = F(r,p) - F(l-1,p), \ g(l,r,p) = G(r,p) - G(l-1,p)$로 줄여 쓸 수 있습니다. 구간에 대한 쿼리를 prefix에 대한 쿼리로 바꾸어서 생각할 수 있다는 의미입니다.

### 오프라인 쿼리

이제 $O(N \sqrt Q)$개의 $F,G$ 함수들의 값만 빠르게 알아낼 수 있다면 전체 문제도 빠르게 해결할 수 있습니다. 이를 위해 두 번의 MO's Algorithm을 수행할 것입니다.

#### 첫 번째 MO

첫 번째 MO's Algorithm에서는 inversion의 변화량 계산은 무시하고, 상태 전이 과정에서 계산해야 하는 $F,G$ 함수가 어떤 형태인지만을 모두 알아냅니다. 만약 상태 전이 과정 도중에 $F(i,p)$의 계산을 요청한다면, 실제 $F(i,p)$값의 계산은 무시하고 $(i,p)$ 쌍을 기억해 두기만 한다는 뜻입니다.

#### 오프라인 스위핑

이제 $i=1,2,\dots,N$의 순서대로 펜윅 트리에 업데이트를 수행하며, 매 $i$번째 업데이트 직후마다 이전에 기억해 두었던 모든 $(i,\ast)$ 쌍에 대해 $F(i,\ast)$이나 $G(i,\ast)$값을 펜윅 트리를 통해 알아내고, 알아낸 모든 결과값들을 저장합니다.

#### 두 번째 MO

두 번째 MO's Algorithm에서는 계산하고 싶은 $F(i,p),G(i,p)$ 값을 모두 얻은 이후이므로 실제로 inversion의 변화량을 계산할 수 있습니다. 쿼리가 정렬된 순서대로 inversion의 변화량을 누적하면 모든 쿼리에 대한 답을 얻을 수 있습니다.

### 메모리 커팅

첫 번째 MO에서 구하고 싶은 모든 $F(i,p)$에 대해 $(i,p)$값을 일일이 저장한다면 메모리 사용량이 $O(N \sqrt Q)$로 너무 커집니다.

이를 개선하기 위해 각 상태 전이에서 계산하려는 $F$ 함수의 형태를 자세히 살펴봅시다. $[l,r] \rightarrow [l,r+1]$로의 상태 전이는 $F(r+1,r+1) - F(l-1,r+1)$의 계산을 요청합니다.

- $F(r+1,r+1)$의 계산: 모든 $1 \leq i \leq N$에 대해 $F(i,i)$값을 전처리해 두면 됩니다.
- $F(l-1,r+1)$의 계산: $[l,r] \rightarrow [l,r+1] \rightarrow [l,r+2] \rightarrow \dots \rightarrow [l,R]$처럼 같은 상태 전이가 연속적으로 일어난다면 그룹으로 묶어서 생각합시다. MO's Algorithm의 성질에 의해 이러한 그룹은 최대 $Q$개가 존재합니다. 하나의 그룹은 모든 $r+1 \leq p \leq R$에 대해 $F(l-1,p)$의 계산을 요청하므로, 모든 요청을 $(l-1,r+1,R)$ 이라는 하나의 정보로 묶어서 생각할 수 있습니다.

상태 전이의 종류는 $[l,r] \rightarrow [l,r+1], \ [l,r] \rightarrow [l-1,r], \ [l,r] \rightarrow [l,r-1], \ [l,r] \rightarrow [l+1,r]$로 4종류가 존재하므로 묶인 정보는 최대 $4Q$개이고 메모리 사용량을 $O(N + Q)$로 개선할 수 있습니다.

### 시간복잡도

#### 첫 번째 MO

첫 번째 MO에서는 $4Q$개의 묶인 정보를 기억해 두기만 하므로 $O(Q)$에 수행됩니다.

#### 오프라인 스위핑

오프라인 스위핑에서는 펜윅 트리에 업데이트를 $O(N)$번 수행하고, $F(i,i),G(i,i)$들을 전처리하는 데에 쿼리를 $O(N)$번 수행하고, 묶인 정보들을 순회하면서 쿼리를 $O(N \sqrt Q)$번 수행하기 때문에 $O(N \sqrt Q \log N)$의 시간이 걸립니다.

이대로는 느리지만, 쿼리의 횟수가 $O(N \sqrt Q)$회로 많은 반면에 업데이트의 횟수가 $O(N)$회로 적다는 점을 이용할 수 있습니다. 업데이트와 쿼리의 시간복잡도가 동일한 펜윅 트리 대신에, sqrt decomposition을 사용해서 업데이트 $O(\sqrt N)$, 쿼리 $O(1)$에 동작하는 자료 구조를 만들 수 있고, 이를 사용하면 $O(N \sqrt N + N \sqrt Q)$로 시간복잡도를 최적화할 수 있습니다.

#### 두 번째 MO

두 번째 MO에서는 포인터를 한 칸 이동할 때마다 전처리해 두었던 $F(i,i),G(i,i)$값을 일일이 참조하면 $O(N \sqrt Q)$의 시간이 걸리지만, 앞의 과정에서 $F(i,i),G(i,i)$값들의 prefix sum을 전처리해 둔다면 이를 활용해 포인터를 여러 칸씩 스킵할 수 있어서 $O(Q)$로 개선이 가능합니다.

따라서 모든 과정을 종합하면 최종 시간복잡도는 $O(N (\sqrt N + \sqrt Q) + Q)$가 됩니다.


### 구현

코드는 다음과 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

#define INF 1234567890
#define ll long long

struct sqrt_decomposition {
	static const int N = 100489, SZ = 317, CNT = N/SZ; // N % SZ == 0, [0, N-1], [0, SZ-1], [0, CNT-1]
	int lo[N], hi[N], LO[CNT], HI[CNT];
	void insert(int i)
	{
		for(int j=i+1; j<i/SZ*SZ+SZ; j++) lo[j]++;
		for(int j=i/SZ+1; j<CNT; j++) LO[j]++;

		for(int j=i-1; j>=i/SZ*SZ; j--) hi[j]++;
		for(int j=i/SZ-1; j>=0; j--) HI[j]++;
	}
	int lo_cnt(int i) { return lo[i] + LO[i/SZ]; }
	int hi_cnt(int i) { return hi[i] + HI[i/SZ]; }
} sq;

const int SZ = 317; // MO에서 사용할 block의 크기
int N, Q;
int A[100001];
ll res[100001];

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> Q;
	vector<int> com;
	for(int i=1; i<=N; i++)
	{
		cin >> A[i];
		com.push_back(A[i]);
	}

	// 좌표 압축
	sort(com.begin(), com.end());
	com.erase(unique(com.begin(), com.end()), com.end());
	for(int i=1; i<=N; i++)
		A[i] = lower_bound(com.begin(), com.end(), A[i]) - com.begin() + 1;

	// MO's Algorithm으로 쿼리를 처리할 순서를 재배열
	vector<array<int, 3> > q;
	for(int i=1; i<=Q; i++)
	{
		int l, r;
		cin >> l >> r;
		l++; // 1-based, closed interval
		q.push_back({l, r, i});
	}
	sort(q.begin(), q.end(), [&](const array<int, 3> &a, const array<int, 3> &b){
		if (a[1]/SZ != b[1]/SZ) return a[1]/SZ < b[1]/SZ;
		return (a[1]/SZ % 2) ? (a[0] < b[0]) : (a[0] > b[0]);
	});

	// 첫 번째 MO
	vector<vector<array<int, 4> > > f(N+1), g(N+1); // 묶인 정보들
	int l = 1, r = 0;
	for(auto [L,R,i] : q) // [l,r] -> [L,R]로의 전이
	{
		// f(l, r+1, r+1) = F(r+1, r+1) - F(l-1, r+1)
		if (r < R) f[l-1].push_back({r+1, R, i, -1}), r = R;
		// g(l, r, l-1) = G(r, l-1) - G(l-1, l-1)
		if (L < l) g[r].push_back({L, l-1, i, +1}), l = L;
		// -f(l, r, r) = -F(r, r) + F(l-1, r)
		if (R < r) f[l-1].push_back({R+1, r, i, +1}), r = R;
		// -g(l+1, r, l) = -G(r, l) + G(l, l)
		if (l < L) g[r].push_back({l, L-1, i, -1}), l = L;
	}

	// 오프라인 스위핑
	vector<ll> ans(Q+1);
	vector<ll> F(N+1), G(N+1);
	for(int i=1; i<=N; i++)
	{
		// sqrt decomposition에 업데이트
		sq.insert(A[i]);
		// F(i,i), G(i,i)의 prefix sum 계산
		F[i] = F[i-1] + sq.hi_cnt(A[i]);
		G[i] = G[i-1] + sq.lo_cnt(A[i]);

		// 묶인 정보들을 순회하며 답 계산
		for(auto [L,R,idx,sgn] : f[i])
			for(int j=L; j<=R; j++)
				ans[idx] += sq.hi_cnt(A[j]) * sgn;
		for(auto [L,R,idx,sgn] : g[i])
			for(int j=L; j<=R; j++)
				ans[idx] += sq.lo_cnt(A[j]) * sgn;
	}

	// 두 번째 MO
	l = 1, r = 0;
	ll sum = 0;
	for(auto [L,R,i] : q)
	{
		if (r < R) ans[i] += F[R] - F[r], r = R;
		if (L < l) ans[i] -= G[l-1] - G[L-1], l = L;
		if (R < r) ans[i] -= F[r] - F[R], r = R;
		if (l < L) ans[i] += G[L-1] - G[l-1], l = L;
		sum += ans[i]; // 이번 쿼리의 inversion의 변화량 누적하기
		res[i] = sum;
	}

	for(int i=1; i<=Q; i++)
		cout << res[i] << "\n";
	return 0;
}
```

![](/assets/images/edenooo/sweepline-mo/sqrt.png)

160ms 정도에 해결할 수 있습니다. 기존 풀이에 비해서 5배 정도 빠르게 작동합니다.



## 연습 문제

### [Static Range Inversions Query](https://judge.yosupo.jp/problem/static_range_inversions_query)

위에서 설명한 문제입니다.

### [수열과 쿼리 23](https://www.acmicpc.net/problem/16979)

위와 동일한 문제입니다.

![](/assets/images/edenooo/sweepline-mo/sq23.png)

MO's Algorithm으로도 해결할 수 있지만, sweepline MO로 개선하면 실행 시간 1위를 달성할 수 있습니다.

### [Inverzije](https://www.acmicpc.net/problem/25462)

위와 동일한 문제입니다.

![](/assets/images/edenooo/sweepline-mo/inverzije.png)

역시 sweepline MO를 사용한 풀이는 다른 풀이에 비해 빠르게 동작합니다.

### [간단한 쿼리 문제](https://www.acmicpc.net/problem/27937)

매 $(l,r)$ 쿼리마다 $\sum_{l \leq i \leq j \leq r} \lvert A_i - A_j \rvert$를 구하는 문제입니다. 절댓값 기호를 제거하기 위해 $A_i < A_j$인 경우와 $A_i > A_j$인 경우로 분리해서 생각합시다.

MO's Algorithm에서 $[l,r] \rightarrow [l,r+1]$로의 전이를 할 때 $l \leq i \leq r, A_i < A_{r+1}$를 만족하는 $i$의 개수를 $c$라 하고 그러한 $A_i$들의 합을 $s$라 하면 정답이 $c \cdot A_{r+1} - s$만큼 증가합니다. $l \leq i \leq r, A_i > A_{r+1}$를 만족하는 $i$에 대해서도 비슷하게 할 수 있습니다.

Static Range Inversions Query와 매우 유사하므로, $i$의 개수를 관리하는 sqrt decomposition과 $A_i$들의 합을 관리하는 sqrt decomposition을 각각 만들어서 sweepline MO로 해결할 수 있습니다.

### [수열과 쿼리 9](https://www.acmicpc.net/problem/13554)

$A_p \cdot B_q \leq k$인 경우만 세기 때문에, $A_p$와 $B_q$ 둘 중 작은 값이 항상 $\lfloor \sqrt{10^5} \rfloor = 316$ 이하인 경우만 세어도 됩니다.

- $A_p \leq B_q$인 개수를 센다고 하면 $1 \leq a \leq 316$에 대해 ($A_p = a$인 $p$의 개수) $\times$ ($a \leq B_q \leq \frac{k}{a}$인 $q$의 개수)를 세면 됩니다.

- $A_p > B_q$인 개수를 센다고 하면 $1 \leq b \leq 316$에 대해 ($B_q = b$인 $q$의 개수) $\times$ ($b < A_p \leq \frac{k}{b}$인 $p$의 개수)를 세면 됩니다.

따라서 아래에 정의된 함수 $f,g$를 계산할 수 있으면 전체 문제가 해결됩니다.

- $f(l,r,x) = l \leq p \leq r, A_p \leq x$인 $p$의 개수
- $g(l,r,x) = l \leq q \leq r, B_q \leq x$인 $q$의 개수

$f,g$는 $f(l,r,x) = f(1,r,x)-f(1,l-1,x), \ g(l,r,x) = g(1,r,x)-g(1,l-1,x)$가 성립하므로 sweepline MO를 적용할 수 있습니다.

MO's Algorithm의 상태 전이를 펜윅 트리로 관리해서 $O(N \sqrt Q \log N + Q \sqrt{10^5} \log N)$에 해결하는 풀이가 잘 알려져 있지만, sweepline MO를 사용해 펜윅 트리를 sqrt decomposition으로 대체하면 $O((N+Q) \sqrt{10^5})$에 더 빠르게 해결할 수 있습니다.

### 코드

코드는 아래와 같습니다.

문제의 특수성 때문에 MO's Algorithm의 상태 전이에서 포인터를 실제로 이동할 필요가 없고, 심지어 쿼리를 정렬된 순서대로 방문할 필요마저 없어서 코드가 간단해집니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

#define INF 1234567890
#define ll long long

struct sqrt_decomposition {
	static const int N = 100489, SZ = 317, CNT = N/SZ; // N % SZ == 0, [0, N-1], [0, SZ-1], [0, CNT-1]
	int a[N], A[CNT], b[N], B[CNT];
	void addA(int i)
	{
		for(int j=i; j<i/SZ*SZ+SZ; j++) a[j]++;
		for(int j=i/SZ+1; j<CNT; j++) A[j]++;
	}
	void addB(int i)
	{
		for(int j=i; j<i/SZ*SZ+SZ; j++) b[j]++;
		for(int j=i/SZ+1; j<CNT; j++) B[j]++;
	}
	int sumA(int i) { return a[i] + A[i/SZ]; }
	int sumB(int i) { return b[i] + B[i/SZ]; }
} sq;

const int SZ = 317; // MO에서 사용할 block의 크기
int N, Q;
int A[100001], B[100001];
pair<int, int> C1[100001][317], C2[100001][317];

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N;
	for(int i=1; i<=N; i++)
		cin >> A[i];
	for(int i=1; i<=N; i++)
		cin >> B[i];

	// 첫 번째 MO
	cin >> Q;
	vector<vector<array<int, 3> > > v(N+1); // 묶인 정보들
	for(int i=1; i<=Q; i++)
	{
		int l, r, k;
		cin >> l >> r >> k;
		v[r].push_back({k, i, +1});
		v[l-1].push_back({k, i, -1});
	}

	// 오프라인 스위핑
	for(int i=1; i<=N; i++)
	{
		sq.addA(A[i]);
		sq.addB(B[i]);
		for(auto [k,idx,sgn] : v[i])
		{
			for(int a=1; a<=316; a++) // Case 1. A[p] <= B[q]인 경우
			{
				C1[idx][a].first += (sq.sumA(a) - sq.sumA(a-1)) * sgn; // A[p] == a인 A[p]의 개수 구하기
				if (a-1 < k/a) C1[idx][a].second += (sq.sumB(k/a) - sq.sumB(a-1)) * sgn; // B[q] >= a && a * B[q] <= k인 B[q]의 개수 구하기
			}
			for(int b=1; b<=316; b++) // Case 2. A[p] > B[q]인 경우
			{
				C2[idx][b].second += (sq.sumB(b) - sq.sumB(b-1)) * sgn; // B[q] == b인 B[q]의 개수 구하기
				if (b < k/b) C2[idx][b].first += (sq.sumA(k/b) - sq.sumA(b)) * sgn; // A[p] > b && b * A[p] <= k인 A[p]의 개수 구하기
			}
		}
	}

	for(int i=1; i<=Q; i++)
	{
		ll res = 0;
		for(int a=1; a<=316; a++) // Case 1. A[p] <= B[q]인 경우
			res += (ll)C1[i][a].first * C1[i][a].second; // (a의 개수) * (B[q] >= a && a * B[q] <= k인 B[q]의 개수)
		for(int b=1; b<=316; b++) // Case 2. A[p] > B[q]인 경우
			res += (ll)C2[i][b].first * C2[i][b].second; // (b의 개수) * (A[p] > b && b * A[p] <= k인 A[p]의 개수)
		cout << res << "\n";
	}
	return 0;
}
```

![](/assets/images/edenooo/sweepline-mo/sq9.png)

이처럼 sweepline MO를 적절한 상황에 적용하면 여러 문제를 효율적으로 해결할 수 있습니다.



## 참고 자료
- [Codeforces box’s blog: [Tutorial] Square root decomposition and applications](https://codeforces.com/blog/entry/83248)