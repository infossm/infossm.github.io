---
layout: post
title:  "CDQ Divide and Conquer, Relaxed Convolution"
date:   2023-09-24 23:00
author: edenooo
tags: [algorithm, divide-and-conquer, FFT]
---

# 개요

**CDQ Divide and Conquer**는 중국 프로그래머 CDQ의 이름을 딴 분할정복 기법의 일종입니다. 딱히 이름이 붙을 정도로 거창한 기법은 아니지만, 이후에 다룰 Relaxed Convolution의 이해를 돕기 위해 간단하게 소개하겠습니다.

**Relaxed Convolution**은 CDQ Divide and Conquer를 사용해서 Convolution을 온라인으로 처리하는 알고리즘으로, Online Convolution, Online FFT, Relaxed Multiplication, Divide and Conquer FFT 등의 다양한 이름으로 불리기도 합니다. 이 글에서는 Relaxed Convolution이라는 용어를 일관적으로 사용하겠습니다.

이 글은 FFT의 작동 원리를 이해하지 않고 빠른 다항식 곱셈 라이브러리를 blackbox로 사용하더라도 읽는 데에 지장이 없도록 작성했습니다.



# CDQ Divide and Conquer

CDQ Divide and Conquer를 설명하기 위한 예시로 LIS 문제에 CDQ를 적용해 보겠습니다.

## 문제

여기에서 해결할 LIS(Longest Increasing Subsequence, 최장 증가 부분 수열) 문제는 아래와 같이 정의됩니다.

길이가 $N$인 수열 $A = (A_1, A_2, \cdots, A_N)$이 입력으로 주어지면, $A$의 strictly increasing subsequence들 중에 최대 길이를 출력해야 합니다. $(1 \leq N \leq 1,000,000, -10^9 \leq A_i \leq 10^9)$

예를 들어 $A = (1, 5, 3, 3, 4, 2)$라면 $A$의 LIS는 $(1, 3, 4)$이고 정답은 LIS의 길이인 $3$이 됩니다.

## 풀이

- $dp[i]$ : $A_i$를 마지막 원소로 갖는 $A$의 strictly increasing subsequence들 중에 최대 길이

위처럼 정의되는 DP 배열을 계산할 수 있다면 정답은 $\max_{1 \leq i \leq N} (dp[i])$로 구할 수 있습니다.

- $dp[i] = \max_{j < i, A_j < A_i} (dp[j]) + 1$

DP 점화식은 위와 같지만 이를 단순하게 2중 반복문으로 계산하면 $O(N^2)$의 시간이 들기 때문에 최적화가 필요합니다.

여기서 상태 전이의 조건이 $j < i$와 $A_j < A_i$로 두 개나 있다는 점이 까다로운 부분인데, CDQ Divide and Conquer를 사용하면 조건 하나를 지우고 문제를 더 낮은 차원에서 풀 수 있습니다.

$[l,r]$ 구간에 대한 답(여기에서는 DP 배열의 값)을 구하는 문제에서 CDQ Divide and Conquer는 아래와 같이 진행됩니다.

1. $m = \left \lfloor \frac{l+r}{2} \right \rfloor$라 할 때, $[l,r]$ 구간에 대한 문제를 $[l,m]$과 $[m+1,r]$ 두 구간에 대한 부분문제로 쪼개서 생각합니다.

2. $[l,m]$ 구간의 부분 문제를 해결합니다.

3. $[l,m]$ 구간의 답이 $[m+1,r]$ 구간의 답에 미치는 영향을 모두 계산합니다.

4. $[m+1,r]$ 구간의 부분 문제를 해결합니다.

5. $[l,m]$ 구간의 답과 $[m+1,r]$ 구간의 답을 합쳐서 $[l,r]$ 구간의 답을 얻습니다.

CDQ Divide and Conquer가 일반적인 분할 정복과 다른 점은 3번 파트의 존재입니다. LIS 문제의 DP 점화식 계산을 예로 들면, $[l,m]$ 구간의 $dp[j]$값이 $[m+1,r]$ 구간의 $dp[i]$값에 미치는 영향을 계산할 때에는 $j < i$ 라는 조건 하나를 무시해도 되기 때문에, 전처리로 $(A_l, A_{l+1}, \cdots, A_r)$을 merge sort해 두었다면 DP 배열의 상태 전이는 투 포인터로 $O(r-l)$에 계산할 수 있습니다. 따라서 분할 정복 과정 전체에서 드는 시간의 총합은 $O(N \log N)$이 됩니다.

## 코드

백준 온라인 저지의 [**가장 긴 증가하는 부분 수열 3**](https://www.acmicpc.net/problem/12738) 문제를 CDQ로 해결하는 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

int N;
int A[1000001], dp[1000001];
vector<pair<int, int> > v[2100000];

void chmax(int &x, int y) { x = max(x, y); }

void merge_sort(int n, int l, int r)
{
	if (l == r) { v[n].push_back({A[r], r}); return; }
	int mid = (l+r)/2;
	merge_sort(n*2, l, mid);
	merge_sort(n*2+1, mid+1, r);
	v[n].resize(r-l+1);
	merge(v[n*2].begin(), v[n*2].end(), v[n*2+1].begin(), v[n*2+1].end(), v[n].begin());
}

void dnc(int n, int l, int r)
{
	if (l == r) return;
	int mid = (l+r)/2;
	dnc(n*2, l, mid); // 1. 왼쪽 절반의 답 구하기

	// 2. 왼쪽의 답이 오른쪽에 주는 영향을 계산하기
	const auto &L = v[n*2], &R = v[n*2+1];
	for(int i=0,j=0,mx=0; i<L.size() || j<R.size(); ) // 투 포인터
	{
		if (j == R.size() || i < L.size() && L[i].first < R[j].first) chmax(mx, dp[L[i++].second]);
		else chmax(dp[R[j++].second], mx + 1);
	}

	dnc(n*2+1, mid+1, r); // 3. 오른쪽 절반의 답 구하기
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N;
	for(int i=1; i<=N; i++)
		cin >> A[i];
	merge_sort(1, 1, N);
	dp[1] = 1;
	dnc(1, 1, N);
	cout << *max_element(dp+1, dp+N+1) << "\n";
	return 0;
}
```



# Relaxed Convolution

길이가 $N$인 수열 $A = (A_0, A_1, \cdots, A_{N-1})$와 길이가 $M$인 수열 $B = (B_0, B_1, \cdots, B_{M-1})$의 **Convolution** $C = A \ast B$는 $C_i = \sum_{j+k = i} A_j B_k$로 정의되는 길이가 $N+M-1$인 수열 $C = (C_0, C_1, \cdots, C_{N+M-2})$를 반환합니다.

여기부터는 Convolution을 $O((N+M) \log (N+M))$의 시간복잡도로 계산할 수 있음을 전제로 하고 진행하겠습니다. [**AtCoder Library에 내장된 Convolution 코드**](https://github.com/atcoder/ac-library/blob/master/document_en/convolution.md)를 그대로 사용할 것이므로, Convolution의 구현 방법을 전혀 모르더라도 괜찮습니다.

**Relaxed Convolution**은 기존의 Convolution과 동일한 작업을 **온라인**으로 수행하는 알고리즘으로, 수열 $A$와 $B$가 한꺼번에 주어지지 않고 항들이 차례대로 주어지는 경우에도 수열 $C$를 구할 수 있습니다.

다시 말해, 아래와 같은 쿼리를 $O(N \log^2 N)$의 시간복잡도로 수행할 수 있습니다.

- $i = 0,1,2,\cdots,N-1$에 대해, $A_i$와 $B_i$가 주어질 때마다 $C_i = \sum_{j+k = i} A_j B_k$를 구한다.

Relaxed Convolution은 수열 $C$의 어떤 원소를 구하기 위해서 $C$의 다른 원소가 필요한 경우에 유용합니다. 예를 들어 카탈란 수의 점화식 $C_i = \sum_{0 \leq j \leq i-1} C_j C_{i-1-j}$은 Relaxed Convolution을 통해 바로 계산할 수 있습니다.



## Semi-Relaxed Convolution

Relaxed Convolution에서 수열 $A$는 미리 한꺼번에 주어져 있고 수열 $B$만 숨겨져 있는 경우를 **Semi-Relaxed Convolution**이라 합니다. 이러한 경우 일반적인 Relaxed Convolution보다 간단하게 계산할 수 있으므로 먼저 다루고 넘어가겠습니다.

### 풀이

CDQ Divide and Conquer를 그대로 적용합니다. $C[l,r]$을 구하려면 다음과 같이 분할 정복을 하면 됩니다.

1. $m = \left \lfloor \frac{l+r}{2} \right \rfloor$라 할 때, $C[l,r]$을 $C[l,m]$과 $C[m+1,r]$로 쪼개서 생각합니다.

2. $C[l,m]$을 구합니다.

3. $B[l,m]$이 $C[m+1,r]$에 미치는 영향을 모두 계산합니다. 구체적으로 $m+1 \leq i \leq r$에 대해 $\sum_{l \leq j \leq m, j+k = i} A_k B_j$를 $C_i$에 더하면 됩니다. 이 때 $k > r-l$인 $A_k$ 값들은 사용되지 않으므로, $\textrm{convolution}(A[0,r-l], B[l,m])$을 구해 두면 $O((r-l) \log (r-l))$시간에 계산할 수 있습니다.

4. $C[m+1,r]$을 구합니다.

5. $C[l,m]$과 $C[m+1,r]$을 합쳐서 $C[l,r]$을 얻습니다.

3번 파트에서는 $B_r$이 $C_r$에 미치는 영향만 계산되지 않았으므로, 분할 정복 과정에서 $l = r$인 $[l,r]$ 구간(=리프 노드)로 들어갈 때마다 $C_r \leftarrow C_r + A_0 B_r$을 추가로 진행해 주어야 합니다.

3번 파트의 시간복잡도인 $O((r-l) \log (r-l))$을 모두 합해서 $O(N \log^2 N)$의 시간복잡도로 수열 $C$를 구할 수 있습니다.

### 코드

[**Library-Checker의 Convolution 문제**](https://judge.yosupo.jp/problem/convolution_mod)를 AtCoder Library로 구현된 Semi-Relaxed Convolution으로 해결하는 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
#include<atcoder/modint>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

int N, M;
vector<modular> A, B, C;

void dnc(int l, int r)
{
	if (l == r) // B[r]이 주어지면 C[r]을 계산한다.
	{
		int x;
		if (r < M) cin >> x, B[r] = x;
		C[r] += A[0] * B[r];
		cout << C[r].val() << " ";
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid); // 1. 왼쪽 절반의 답 구하기

	// 2. 왼쪽의 B가 오른쪽의 C에 주는 영향을 계산하기
	auto AA = vector<modular>(A.begin() + 0, A.begin() + r-l + 1);
	auto BB = vector<modular>(B.begin() + l, B.begin() + mid + 1);
	auto CC = convolution(AA, BB);
	for(int i=mid+1; i<=r; i++)
		C[i] += CC[i-l];

	dnc(mid+1, r); // 3. 오른쪽 절반의 답 구하기
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M;
	A = B = C = vector<modular>(N+M-1);
	for(int i=0,x; i<N; i++)
		cin >> x, A[i] = x; // A는 미리 주어져 있다.
	dnc(0, N+M-2);
	return 0;
}
```



## Relaxed Convolution

이번에는 수열 $A,B$가 초기 상태에서 모두 숨겨져 있는 일반적인 케이스도 해결할 수 있도록 기존의 Semi-Relaxed Convolution 풀이를 확장하겠습니다. 기존 풀이와 거의 동일하지만, 약간의 Case Work가 필요합니다.

### 풀이

Semi-Relaxed Convolution에서 사용했던 CDQ Divide and Conquer의 3번 파트를 다시 가져와 봅시다.

> $B[l,m]$이 $C[m+1,r]$에 미치는 영향을 모두 계산합니다. 구체적으로 $m+1 \leq i \leq r$에 대해 $\sum_{l \leq j \leq m, j+k = i} A_k B_j$를 $C_i$에 더하면 됩니다. 이 때 $k > r-l$인 $A_k$ 값들은 사용되지 않으므로, $\textrm{convolution}(A[0,r-l], B[l,m])$을 구해 두면 $O((r-l) \log (r-l))$시간에 계산할 수 있습니다.

Semi-Relaxed Convolution의 풀이를 그대로 Relaxed Convolution에서도 사용할 수 없는 이유는, $r-l > m$일 경우 $A_{m+1}, A_{m+2}, \cdots, A_{r-l}$이 아직 주어져 있지 않아서 $\textrm{convolution}(A[0,r-l], B[l,m])$을 계산할 수 없기 때문입니다. 이러한 경우는 $l = 0$인 $[l,r]$ 구간에서만 발생함을 확인할 수 있습니다.

따라서, 3번 파트를 다음과 같이 두 개의 케이스로 나눠서 처리하면 됩니다.

- $l = 0$인 $[l,r]$ 구간에서는 $\textrm{convolution}(A[l,m], B[l,m])$이 $C[m+1,r]$에 미치는 영향을 모두 계산합니다.

- $l \neq 0$인 $[l,r]$ 구간에서는 $\textrm{convolution}(A[0,r-l], B[l,m])$이 $C[m+1,r]$에 미치는 영향과 $\textrm{convolution}(A[l,m], B[0,r-l])$이 $C[m+1,r]$에 미치는 영향을 모두 계산합니다.

마지막 디테일로, 분할 정복 과정에서 $l = r$인 $[l,r]$ 구간(=리프 노드)로 들어갈 때마다 $C_r \leftarrow C_r + A_r B_0 + A_0 B_r$을 추가로 진행해 주어야 합니다. ($r = 0$인 경우 중복으로 더해지지 않게 주의해야 합니다.)

전체 시간복잡도는 Semi-Relaxed Convolution과 동일하게 $O(N \log^2 N)$이지만, convolution의 횟수가 늘어났기 때문에 Semi-Relaxed Convolution에 비해 2배 정도 느리게 동작합니다.

### 코드

[**Library-Checker의 Convolution 문제**](https://judge.yosupo.jp/problem/convolution_mod)를 AtCoder Library로 구현된 Relaxed Convolution으로 해결하는 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
#include<atcoder/modint>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

int N, M;
vector<modular> A, B, C;

void dnc(int l, int r)
{
	if (l == r) // A[r],B[r]이 주어지면 C[r]을 계산한다.
	{
		C[r] += A[r] * B[0];
		if (r) C[r] += A[0] * B[r]; // r = 0인 경우 중복 계산 주의
		cout << C[r].val() << " ";
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid); // 1. 왼쪽 절반의 답 구하기

	// 2. 왼쪽의 A,B가 오른쪽의 C에 주는 영향을 계산하기
	if (l == 0)
	{
		auto CC = convolution(vector<modular>(A.begin() + l, A.begin() + mid + 1), vector<modular>(B.begin() + l, B.begin() + mid + 1));
		for(int i=mid+1; i<=r; i++)
			C[i] += CC[i-l];
	}
	else
	{
		auto AB = convolution(vector<modular>(A.begin() + 0, A.begin() + r-l + 1), vector<modular>(B.begin() + l, B.begin() + mid + 1));
		auto BA = convolution(vector<modular>(B.begin() + 0, B.begin() + r-l + 1), vector<modular>(A.begin() + l, A.begin() + mid + 1));
		for(int i=mid+1; i<=r; i++)
			C[i] += AB[i-l] + BA[i-l];
	}

	dnc(mid+1, r); // 3. 오른쪽 절반의 답 구하기
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M;
	A = B = C = vector<modular>(N+M-1);
	for(int i=0,x; i<N; i++)
		cin >> x, A[i] = x;
	for(int i=0,x; i<M; i++)
		cin >> x, B[i] = x;
	dnc(0, N+M-2);
	return 0;
}
```



# 연습 문제

글이 길어져서 Relaxed Convolution 문제들은 다른 글에서 이어서 작성하겠습니다.

## [KOI 2018 고등부 3번. 조화로운 행렬](https://www.acmicpc.net/problem/15977)

$M = 2$는 $M = 3$의 하위 호환이므로, $M = 3$ 케이스에 대해서만 풀이를 작성하겠습니다.

### 풀이

세 개의 행을 각각 $A,B,C$라 하면, $(i_1, i_2, \cdots, i_k)$ 열들을 선택하여 만든 열-부분행렬이 조화로운 행렬일 조건은 $(A_{i_1}, A_{i_2}, \cdots, A_{i_k})$와 $(B_{i_1}, B_{i_2}, \cdots, B_{i_k})$와 $(C_{i_1}, C_{i_2}, \cdots, C_{i_k})$가 모두 strictly increasing인 것과 같습니다.

- $dp[i] = \max_{A_j < A_i, B_j < B_i, C_j < C_i} (dp[j]) + 1$

LIS 문제와 매우 유사하므로, 위처럼 정의되는 DP 배열을 계산할 수 있다면 정답은 $\max_{1 \leq i \leq N} (dp[i])$로 구할 수 있습니다.

각 열마다 좌표압축한 $A_i$값을 인덱스로 해서 정렬한 뒤 CDQ Divide and Conquer를 사용하겠습니다.

1. $m = \left \lfloor \frac{l+r}{2} \right \rfloor$라 할 때, $[l,r]$ 구간에 대한 문제를 $[l,m]$과 $[m+1,r]$ 두 구간에 대한 부분문제로 쪼개서 생각합니다.

2. $[l,m]$ 구간의 DP값을 계산합니다.

3. $[l,m]$ 구간의 DP값이 $[m+1,r]$ 구간의 DP값에 미치는 영향을 모두 계산합니다.

4. $[m+1,r]$ 구간의 DP값을 계산합니다.

5. $[l,m]$ 구간의 답과 $[m+1,r]$ 구간의 답을 합쳐서 $[l,r]$ 구간의 답을 얻습니다.

3번 파트의 $j \in [l,m]$에서 $i \in [m+1,r]$로의 상태 전이는 $A_j < A_i$ 조건을 무시하고 풀 수 있습니다. 따라서 점화식은 아래와 같이 3차원에서 2차원으로 축소됩니다.

- $dp[i] = \max_{B_j < B_i, C_j < C_i} (dp[j]) + 1$

$B$값의 오름차순으로 스위핑하면서 좌표압축한 $C$값을 인덱스로 하는 펜윅 트리에 $dp$값을 업데이트해 주면 $[l,r]$ 구간에 대한 3번 파트를 $O((r-l) \log (r-l))$에 계산할 수 있습니다. 따라서 최종 시간복잡도는 $O(N \log^2 N)$이 됩니다.



### 코드

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

int M, N;
int A[200000], B[200000], C[200000], dp[200000], fen[200001];

void chmax(int &x, int y) { x = max(x, y); }
int Max(int i)
{
	int ret = 0;
	for(; i; i&=i-1) chmax(ret, fen[i]);
	return ret;
}
void Add(int i, int x)
{
	for(; i<=N; i+=i&-i) chmax(fen[i], x);
}

void dnc(int l, int r)
{
	if (l == r) return;
	int mid = l+r>>1;
	dnc(l, mid); // 1. 왼쪽 절반의 답 구하기

	// C값을 좌표 압축
	vector<int> com;
	for(int i=l; i<=r; i++)
		com.push_back(C[i]);
	sort(com.begin(), com.end());
	memset(fen, 0, sizeof(int)*(com.size()+1)); // 펜윅 트리 초기화

	// 2. 왼쪽의 답이 오른쪽의 답에 주는 영향을 계산하기
	vector<pair<int, int> > v;
	for(int i=l; i<=r; i++)
		v.push_back({B[i], i});
	sort(v.begin(), v.end()); // B값의 오름차순으로 스위핑
	for(auto [x,i] : v)
	{
		int j = lower_bound(com.begin(), com.end(), C[i]) - com.begin() + 1; // 좌표압축된 C값
		if (i <= mid) Add(j, dp[i]); // 왼쪽의 DP값을 펜윅 트리에 업데이트하기
		else chmax(dp[i], Max(j-1) + 1); // 펜윅 트리의 값을 오른쪽의 DP값에 갱신하기
	}

	dnc(mid+1, r); // 3. 오른쪽 절반의 답 구하기
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> M >> N;
	for(int i=0; i<N; i++)
		cin >> A[i];
	for(int i=0; i<N; i++)
		cin >> B[i];
	for(int i=0; i<N; i++)
	{
		if (M == 2) C[i] = B[i];
		else cin >> C[i];
	}

	// A값을 인덱스로 해서 정렬
	vector<array<int, 3> > v;
	for(int i=0; i<N; i++)
		v.push_back({A[i], B[i], C[i]});
	sort(v.begin(), v.end());
	for(int i=0; i<N; i++)
		B[i] = v[i][1], C[i] = v[i][2];

	// CDQ Divide and Conquer
	dp[0] = 1;
	dnc(0, N-1);

	cout << *max_element(dp, dp+N) << "\n";
	return 0;
}
```



# 참고 자료

- <https://robert1003.github.io/2020/01/31/cdq-divide-and-conquer.html>

- <https://qiita.com/Kiri8128/items/1738d5403764a0e26b4c>