---
layout: post
title:  "Relaxed Convolution (2)"
date:   2023-10-24 23:00
author: edenooo
tags: [algorithm, divide-and-conquer, FFT]
---

# 개요
[**이전에 작성한 글**](https://infossm.github.io/blog/2023/09/24/relaxed-convolution/)에서는 **Relaxed Convolution**의 개념과 성질, 구현 방법에 대해서 다루었습니다. 이 글에서는 Relaxed Convolution을 Problem Solving에 활용하는 방법을 다룹니다.



# 연습 문제

## [AtCoder Beginner Contest 213 H. Stroll](https://atcoder.jp/contests/abc213/tasks/abc213_h)

### 문제

정점이 $N$개이고 간선이 매우 많은 무방향 그래프가 주어집니다.

간선 집합은 다음과 같은 방식으로 정의됩니다.

- $0 \leq i < M, 1 \leq d \leq T$를 만족하는 모든 $(i,d)$ 쌍에 대해, 두 정점 쌍 $(a_i, b_i)$를 잇는 길이가 $d$인 간선이 $p_{i,d}$개 존재합니다.

이 그래프의 $1$번 정점에서 출발해서 $1$번 정점으로 되돌아오는, 길이가 정확히 $T$인 walk의 개수를 $998244353$으로 나눈 나머지를 출력해야 합니다.

$(2 \leq N \leq 10; 1 \leq M \leq \min \left(10, \frac{N(N-1)}{2} \right); 1 \leq T \leq 40000; 0 \leq p_{i,j} < 998244353)$



### 풀이

$dp[u][k]$를, $1$번 정점에서 출발해서 $u$번 정점에 도착하는 길이가 정확히 $k$인 walk의 개수라고 정의합시다.

모든 간선의 길이가 1 이상이므로 상태 전이에는 사이클이 없고, 점화식을 아래와 같이 작성할 수 있습니다.

$dp[u][k] = \sum_{0 \leq i < M, \{a_i, b_i\} = \{u, v\}} \sum_{1 \leq d \leq T} dp[v][k-d] \cdot p_{i,d}$

위 점화식을 단순히 계산하면 $O(MT^2)$로 시간 초과가 발생하므로, $dp[u]$의 상태 전이가 $dp[v]$와 $p_{i}$의 Convolution임을 이용해야 합니다.

$dp[u][k]$를 구할 때 $dp[v][0 \dots k-1]$이 필요하고, $dp[v][k]$를 구할 때에도 $dp[u][0 \dots k-1]$이 필요하므로 Convolution을 바로 적용할 수는 없지만, $k=1,2,\cdots,T$ 순서대로 $dp[1 \dots N][k]$을 계산한다면 Semi-Relaxed Convolution을 적용할 수 있습니다.

최종 시간복잡도는 $O(MT \log^2 T)$가 됩니다.



### 코드

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
#include<atcoder/modint>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

#define INF 1234567890
#define ll long long

int N, M, T;
int A[10], B[10];
vector<modular> p[10];
vector<modular> dp[11];

void dnc(int l, int r)
{
	if (l == r) return;
	int mid = (l+r)/2;
	dnc(l, mid);
	for(int i=0; i<M; i++) // A[i] -> B[i]로 이동하는 경우
	{
		auto AA = vector<modular>(p[i].begin() + 0, p[i].begin() + r-l + 1);
		auto BB = vector<modular>(dp[A[i]].begin() + l, dp[A[i]].begin() + mid + 1);
		auto CC = convolution(AA, BB);
		for(int j=mid+1; j<=r; j++)
			dp[B[i]][j] += CC[j-l];
	}
	for(int i=0; i<M; i++) // B[i] -> A[i]로 이동하는 경우
	{
		auto AA = vector<modular>(p[i].begin() + 0, p[i].begin() + r-l + 1);
		auto BB = vector<modular>(dp[B[i]].begin() + l, dp[B[i]].begin() + mid + 1);
		auto CC = convolution(AA, BB);
		for(int j=mid+1; j<=r; j++)
			dp[A[i]][j] += CC[j-l];
	}
	dnc(mid+1, r);
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M >> T;
	for(int i=0; i<M; i++)
	{
		cin >> A[i] >> B[i];
		p[i].resize(T+1);
		for(int d=1,x; d<=T; d++)
			cin >> x, p[i][d] = x;
	}
	for(int i=1; i<=N; i++)
		dp[i].resize(T+1);
	dp[1][0] = 1; // base case
	dnc(0, T);
	cout << dp[1][T].val() << "\n";
	return 0;
}
```



## [AtCoder Beginner Contest 315 Ex. Typical Convolution Problem](https://atcoder.jp/contests/abc315/tasks/abc315_h)

### 문제

수열 $(A_1, A_2, \cdots, A_N)$이 주어지면, 다음과 같이 정의되는 수열 $(F_0, F_1, \cdots, F_N)$의 각 원소를 $998244353$으로 나눈 나머지를 구해야 합니다.

- $F_0 = 1$

- $1 \leq n$에서 $F_n = A_n \sum_{i+j < n} F_i F_j$

$(1 \leq N \leq 200,000; 0 \leq A_i < 998244353)$



### 풀이

$F$의 누적합 수열인 $S_n = \sum_{i=0}^{n} F_i$를 정의합시다.

수식을 다시 풀어 적으면, $F_n = A_n \sum_{0 \leq i \leq n-1} F_i S_{n-1-i}$이 됩니다.

$F$와 $S$의 Relaxed Convolution을 구하면 $O(N \log^2 N)$에 해결할 수 있습니다.



### 코드

$F_i$와 $S_j$를 곱한 결과물을 $F_{i+j}$에 더하는 것이 아니라 $F_{i+j+1}$에 더해야 한다는 디테일에 주의해서 구현해야 합니다.

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
#include<atcoder/modint>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

#define INF 1234567890
#define ll long long

int N;
int A[200001];
vector<modular> F, S;

void dnc(int l, int r)
{
	if (l == r) // F[r+1], S[r+1]을 구한다.
	{
		F[r+1] += F[r] * S[0];
		if (r) F[r+1] += F[0] * S[r];
		F[r+1] *= A[r+1];
		S[r+1] = S[r] + F[r+1];
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid);
	if (l == 0)
	{
		auto CC = convolution(vector<modular>(F.begin() + l, F.begin() + mid + 1), vector<modular>(S.begin() + l, S.begin() + mid + 1));
		for(int i=mid+1; i<=r; i++)
			F[i+1] += CC[i-l];
	}
	else
	{
		auto AB = convolution(vector<modular>(F.begin() + 0, F.begin() + r-l + 1), vector<modular>(S.begin() + l, S.begin() + mid + 1));
		auto BA = convolution(vector<modular>(S.begin() + 0, S.begin() + r-l + 1), vector<modular>(F.begin() + l, F.begin() + mid + 1));
		for(int i=mid+1; i<=r; i++)
			F[i+1] += AB[i-l] + BA[i-l];
	}
	dnc(mid+1, r);
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N;
	for(int i=1; i<=N; i++)
		cin >> A[i];
	F.resize(N+1), S.resize(N+1);
	F[0] = 1, S[0] = 1; // base case
	dnc(0, N-1);
	for(int i=1; i<=N; i++)
		cout << F[i].val() << " ";
	cout << "\n";
	return 0;
}
```



## [삼각분할](https://www.acmicpc.net/problem/7737)

일반적으로 알려진 풀이와 다르게 Relaxed Convolution으로도 해결할 수 있습니다.

### 문제

$N$과 $M$이 주어집니다.

$i$각형을 삼각분할하는 경우의 수를 $T_i$라 했을 때, $T_3 + T_4 + \cdots + T_N$을 $M$으로 나눈 나머지를 구해야 합니다.

$(3 \leq N \leq 100,000; 2 \leq M \leq 10^9)$



### 풀이

$C_i$를 $i$번째 카탈란 수라고 하면 $T_i = C_{i-2}$임이 잘 알려져 있습니다. 따라서 $C_1 + C_2 + \cdots + C_{N-2} \pmod M$을 구하면 됩니다.

카탈란 수의 점화식은 아래와 같습니다.

- $C_0 = 1$

- $C_i = \sum_{j=0}^{i-1} C_i C_{i-1-j}$

Relaxed Convolution을 적용할 수 있는 형태이므로 $O(N \log^2 N)$에 해결할 수 있습니다.



### 코드

$M$이 임의로 주어지기 때문에 Convolution 구현체를 잘 선택해야 하는데, 저는 [**Aeren님의 코드**](https://github.com/Aeren1564/Algorithms/blob/c84e0d71cdb0f485fcda7de47339438ac8ebac7a/Algorithm_Implementations_Cpp/Power_Series/Transformations/fast_fourier_transform.sublime-snippet)와 [**justiceHui님의 코드**](https://github.com/justiceHui/icpc-teamnote/blob/master/code/Math/Convolution.cpp)를 적당히 섞어서 사용했습니다.

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

#define INF 1234567890
#define ll long long

using real_t = double; using cpx = complex<real_t>;
void FFT(vector<complex<double>> &a, bool invert = false){
	int n = (int)a.size();
	static vector<complex<double>> root(2, 1);
	static vector<complex<long double>> root_ld(2, 1);
	for(static int k = 2; k < n; k <<= 1){
		root.resize(n), root_ld.resize(n);
		auto theta = polar(1.0L, acos(-1.0L) / k);
		for(auto i = k; i < k << 1; ++ i) root[i] = root_ld[i] = i & 1 ? root_ld[i >> 1] * theta : root_ld[i >> 1];
	}
	for(auto i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(a[i], a[j]);
	}
	for(auto k = 1; k < n; k <<= 1) for(auto i = 0; i < n; i += k << 1) for(auto j = 0; j < k; ++ j){
		auto x = (double *)&root[j + k], y = (double *)&a[i + j + k];
		complex<double> z(x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]);
		a[i + j + k] = a[i + j] - z, a[i + j] += z;
	}
	if(invert){
		reverse(a.begin() + 1, a.end());
		auto inv_n = 1.0 / n;
		for(auto &x: a) x *= inv_n;
	}
}
vector<ll> multiply_mod(const vector<ll> &a, const vector<ll> &b, const ll mod){
  int N = 2; while(N < a.size() + b.size()) N <<= 1;
  vector<cpx> v1(N), v2(N), r1(N), r2(N);
  for(int i=0; i<a.size(); i++) v1[i] = cpx(a[i] >> 15, a[i] & 32767);
  for(int i=0; i<b.size(); i++) v2[i] = cpx(b[i] >> 15, b[i] & 32767);
  FFT(v1); FFT(v2);
  for(int i=0; i<N; i++){
    int j = i ? N-i : i;
    cpx ans1 = (v1[i] + conj(v1[j])) * cpx(0.5, 0);
    cpx ans2 = (v1[i] - conj(v1[j])) * cpx(0, -0.5);
    cpx ans3 = (v2[i] + conj(v2[j])) * cpx(0.5, 0);
    cpx ans4 = (v2[i] - conj(v2[j])) * cpx(0, -0.5);
    r1[i] = (ans1 * ans3) + (ans1 * ans4) * cpx(0, 1);
    r2[i] = (ans2 * ans3) + (ans2 * ans4) * cpx(0, 1);
  }
  vector<ll> ret(N); FFT(r1, true); FFT(r2, true);
  for(int i=0; i<N; i++){
    ll av = llround(r1[i].real()) % mod;
    ll bv = ( llround(r1[i].imag()) + llround(r2[i].real()) ) % mod;
    ll cv = llround(r2[i].imag()) % mod;
    ret[i] = (av << 30) + (bv << 15) + cv;
    ret[i] %= mod; ret[i] += mod; ret[i] %= mod;
  }
  while(ret.size() > 1 && ret.back() == 0) ret.pop_back();
  return ret;
}

int N, M;
vector<ll> C;

void dnc(int l, int r)
{
	if (l == r) // C[r+1]을 구한다.
	{
		(C[r+1] += C[r] * C[0]) %= M;
		if (r) (C[r+1] += C[0] * C[r]) %= M;
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid);
	if (l == 0)
	{
		auto CC = multiply_mod(vector<ll>(C.begin() + l, C.begin() + mid + 1), vector<ll>(C.begin() + l, C.begin() + mid + 1), M);
		CC.resize(r-l+1);
		for(int i=mid+1; i<=r; i++)
			(C[i+1] += CC[i-l]) %= M;
	}
	else
	{
		auto CC = multiply_mod(vector<ll>(C.begin() + 0, C.begin() + r-l + 1), vector<ll>(C.begin() + l, C.begin() + mid + 1), M);
		CC.resize(r-l+1);
		for(int i=mid+1; i<=r; i++)
			(C[i+1] += CC[i-l] * 2) %= M;
	}
	dnc(mid+1, r);
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M;
	N -= 2;
	C.resize(N+1);
	C[0] = 1; // base case
	dnc(0, N-1);
	cout << accumulate(C.begin() + 1, C.end(), 0LL) % M << "\n";
	return 0;
}
```



## [Codeforces Round 309 (Div. 1) E. Kyoya and Train](https://codeforces.com/contest/553/problem/E)

### 문제

정점이 $N$개, 간선이 $M$개인 단순 방향 그래프가 주어지고 두 정수 $T,X$가 주어집니다.

$0 \leq i < M$에 대해 간선 $i$는 정점 $a_i$에서 정점 $b_i$로 비용 $c_i$를 지불하고 이동할 수 있습니다. 이 때 이동에 걸리는 시간은 확률 분포로 주어지고 매 이동마다 랜덤하게 결정되는데, 정수 $1 \leq d \leq T$에 대해 $d$의 시간이 걸릴 확률은 $p_{i,d}$입니다. $(0 \leq p_{i,d} \leq 1; \sum_{d=1}^{T} p_{i,d} = 1)$

이 그래프에서 매번 어떤 간선을 따라 이동할지를 최적으로 선택해서, $1$번 정점에서 출발해서 $N$번 정점에 도착하는 비용의 기댓값을 최소화해야 합니다. 단, 걸린 시간이 $T$를 초과하면 지각이므로 비용을 $X$만큼 추가로 지불해야 합니다.

$(2 \leq N \leq 50; 1 \leq M \leq 100; 1 \leq T \leq 20000; 0 \leq X \leq 10^6)$



### 풀이

위 연습 문제인 Stroll과 전반적으로 비슷한 문제입니다.

$dp[u][k]$를, 앞으로 $k$ 이상의 시간을 소모하면 지각일 때 $u$번 정점에서 출발해서 $N$번 정점에 도착하는 비용의 기댓값의 최솟값이라고 정의합시다.

$dp[u][k] = \min_{0 \leq i < M, (a_i,b_i) = (u,v)} \left( \left(\sum_{1 \leq d \leq T} dp[v][\max(0, k-d)] \cdot p_{i,d} \right) + c_i \right)$

위 점화식을 단순히 계산하면 $O(MT^2)$로 시간 초과가 발생하므로, $dp[u]$의 상태 전이가 $dp[v]$와 $p_i$의 Convolution임을 이용해야 합니다.

$ep[i][k] = \sum_{1 \leq d \leq k-1} dp[b_i][k-d] \cdot p_{i,d}$를 새로 정의하고 수식 정리를 합시다.

$dp[u][k] = \min_{0 \leq i < M, (a_i,b_i) = (u,v)} \left(ep[i][k] + dp[v][0] \cdot \left( \sum_{d=k}^{T} p_{i,d} \right) + c_i \right)$

이제 $ep$가 Semi-Relaxed Convolution이 가능하므로 $O(MT \log^2 T)$에 해결할 수 있습니다.

$k = 0$일 때에만 상태 전이에 사이클이 존재하므로, 이 경우는 다익스트라 등의 알고리즘으로 따로 전처리해야 합니다.



### 코드

$N$ 제한이 매우 작아서 구현의 편의를 위해 플로이드-워셜 알고리즘을 사용했습니다.

이 문제에서는 기댓값을 double로 구해야 하기 때문에, Convolution 구현체로 AtCoder Library 대신에 [**Aeren님의 코드**](https://github.com/Aeren1564/Algorithms/blob/c84e0d71cdb0f485fcda7de47339438ac8ebac7a/Algorithm_Implementations_Cpp/Power_Series/Transformations/fast_fourier_transform.sublime-snippet)와 [**justiceHui님의 코드**](https://github.com/justiceHui/icpc-teamnote/blob/master/code/Math/Convolution.cpp)를 적당히 섞어서 사용했습니다.

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;

#define INF 1e18
#define ll long long

using real_t = double; using cpx = complex<real_t>;
void FFT(vector<complex<double>> &a, bool invert = false){
	int n = (int)a.size();
	static vector<complex<double>> root(2, 1);
	static vector<complex<long double>> root_ld(2, 1);
	for(static int k = 2; k < n; k <<= 1){
		root.resize(n), root_ld.resize(n);
		auto theta = polar(1.0L, acos(-1.0L) / k);
		for(auto i = k; i < k << 1; ++ i) root[i] = root_ld[i] = i & 1 ? root_ld[i >> 1] * theta : root_ld[i >> 1];
	}
	for(auto i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(a[i], a[j]);
	}
	for(auto k = 1; k < n; k <<= 1) for(auto i = 0; i < n; i += k << 1) for(auto j = 0; j < k; ++ j){
		auto x = (double *)&root[j + k], y = (double *)&a[i + j + k];
		complex<double> z(x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]);
		a[i + j + k] = a[i + j] - z, a[i + j] += z;
	}
	if(invert){
		reverse(a.begin() + 1, a.end());
		auto inv_n = 1.0 / n;
		for(auto &x: a) x *= inv_n;
	}
}
vector<double> multiply(const vector<double> &_a, const vector<double> &_b){
  vector<cpx> a(_a.begin(), _a.end()), b(_b.begin(), _b.end());
  int N = 2; while(N < a.size() + b.size()) N <<= 1;
  a.resize(N); b.resize(N); FFT(a); FFT(b);
  for(int i=0; i<N; i++) a[i] *= b[i];
  vector<double> ret(N); FFT(a, 1);
  for(int i=0; i<N; i++) ret[i] = a[i].real(); //
  while(ret.size() > 1 && ret.back() == 0) ret.pop_back();
  return ret;
}

int N, M, T, X;
int A[100], B[100], C[100];
vector<double> p[100], P[100], dp[51], ep[100];

void dnc(int l, int r)
{
	if (l == r) // dp[u][r] 구하기
	{
		for(int i=0; i<M; i++)
			dp[A[i]][r] = min(dp[A[i]][r], ep[i][r] + dp[B[i]][0] * (P[i][T] - P[i][r-1]) + C[i]);
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid);
	for(int i=0; i<M; i++)
	{
		auto AA = vector<double>(p[i].begin() + 0, p[i].begin() + r-l + 1);
		auto BB = vector<double>(dp[B[i]].begin() + l, dp[B[i]].begin() + mid + 1);
		auto CC = multiply(AA, BB);
		CC.resize(r-l+1);
		for(int k=mid+1; k<=r; k++)
			ep[i][k] += CC[k-l];
	}
	dnc(mid+1, r);
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cout << setprecision(12) << fixed;
	cin >> N >> M >> T >> X;
	for(int i=0; i<M; i++)
	{
		cin >> A[i] >> B[i] >> C[i];
		p[i].resize(T+2);
		P[i].resize(T+1);
		for(int d=1,x; d<=T; d++)
		{
			cin >> x, p[i][d] = x / 100000.0;
			P[i][d] = P[i][d-1] + p[i][d]; // 누적 합
		}
		ep[i].resize(T+2);
	}
	for(int i=1; i<=N; i++)
		dp[i].resize(T+2, INF);
	for(int i=1; i<=T+1; i++)
		dp[N][i] = 0;
	
	// dp[u][0] 전처리하기 (플로이드)
	vector<vector<double> > dist(N+1, vector<double>(N+1, INF));
	for(int i=1; i<=N; i++)
		dist[i][i] = 0;
	for(int i=0; i<M; i++)
		dist[A[i]][B[i]] = C[i];
	for(int k=1; k<=N; k++)
		for(int i=1; i<=N; i++)
			for(int j=1; j<=N; j++)
				dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
	for(int i=1; i<=N; i++)
		dp[i][0] = dist[i][N] + X;

	dnc(1, T+1);
	cout << dp[1][T+1] << "\n";
	return 0;
}
```



## [Gennady Korotkevich Contest 5. Bin](https://www.acmicpc.net/problem/18743)

### 문제

$N,K$가 주어지면, 다음의 두 조건을 만족하는 리프가 $N$개인 이진 트리의 개수를 $998244353$으로 나눈 나머지를 구해야 합니다.

- 모든 정점은 0개 또는 2개의 자식을 갖는다.

- 2개의 자식을 갖는 모든 정점에 대해, (왼쪽 서브트리의 리프 개수) $\leq$ (오른쪽 서브트리의 리프 개수) + $K$를 만족한다.

$(2 \leq N \leq 10^6, 0 \leq K \leq 100)$



### 풀이

$D_n$을 위의 두 조건을 만족하는 리프가 $n$개인 이진 트리의 개수라고 정의하면, 간단하게 아래 점화식을 찾을 수 있습니다.

- $D_0 = 0, D_1 = 1$

- $i \geq 2$에서 $D_i = \sum_{1 \leq l+r \leq i-1; \ l \leq r+K} D_l D_r$

수식 정리를 해서 문제를 풀기 좋게 변형해 봅시다.

점화식을 다시 작성하면 $D_i = \sum_{1 \leq j \leq \min(i-1, \frac{i+K}{2})} D_j D_{i-j}$이 됩니다.

$i \leq K+1$에 대해 나이브하게 계산하면 $i > K+1$부터는 $D_i = \sum_{1 \leq j \leq \frac{i+K}{2} } D_j D_{i-j}$이 됩니다.

$j > \frac{i}{2}$를 나이브하게 계산한다고 치면, 결국 $D_i = \sum_{1 \leq j \leq \frac{i}{2} } D_j D_{i-j}$를 구하기만 해도 됩니다.

마지막으로 parity에 따라 케이스를 나누면,

- $i$가 홀수일 때 $D_i = \left( \sum_{1 \leq j \leq i-1} D_j D_{i-j} \right) / 2$

- $i$가 짝수일 때 $D_i = \left( (D_{i/2})^2 + \sum_{1 \leq j \leq i-1} D_j D_{i-j} \right) / 2$

이제 Relaxed Convolution으로 문제를 해결할 수 있습니다.

시간복잡도는 $O(K^2 + NK + N \log^2 N)$이 됩니다.



### 코드

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
#include<atcoder/modint>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

#define INF 1234567890
#define ll long long

int N, K;
vector<modular> D;

void dnc(int l, int r)
{
	if (l == r)
	{
		int i = r;
		if (i > K+1)
		{
			// parity에 따라 케이스 나누기
			if (i % 2 == 0) D[i] += D[i/2] * D[i/2];
			D[i] /= 2;
			// j > i/2 나이브
			for(int j=i/2+1; j<=(i+K)/2; j++)
				D[i] += D[j] * D[i-j];
		}
		return;
	}
	int mid = (l+r)/2;
	dnc(l, mid);
	if (l == 0)
	{
		auto DD = convolution(vector<modular>(D.begin() + l, D.begin() + mid + 1), vector<modular>(D.begin() + l, D.begin() + mid + 1));
		DD.resize(DD.size() + 1);
		for(int i=mid+1; i<=r; i++)
			if (i > K+1)
				D[i] += DD[i-l];
	}
	else
	{
		auto DD = convolution(vector<modular>(D.begin() + 0, D.begin() + r-l + 1), vector<modular>(D.begin() + l, D.begin() + mid + 1));
		for(int i=mid+1; i<=r; i++)
			if (i > K+1)
				D[i] += DD[i-l] * 2;
	}
	dnc(mid+1, r);
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> K;
	D.resize(max(N+1, K+2));
	D[0] = 0;
	D[1] = 1;
	// 작은 케이스 전처리
	for(int i=2; i<=K+1; i++)
		for(int j=1; j<=min(i-1, (i+K)/2); j++)
			D[i] += D[j] * D[i-j];
	dnc(0, N);
	cout << D[N].val() << "\n";
	return 0;
}
```