---
layout: post
title:  "Taylor Shift, Sampling Points Shift"
date:   2023-11-26 23:00
author: edenooo
tags: [algorithm, mathematics, FFT]
---

# 개요

최근에 Polynomial Shift를 사용하는 문제를 여러 대회에서 보았기 때문에 이 글을 작성하게 되었습니다.

다항식 $f(x)$와 정수 $c$가 주어졌을 때 새로운 다항식 $f(x+c)$를 구하는 것을 **Taylor Shift**라고 부릅니다.

$N$차 미만의 다항식 $f(x)$가 숨겨져 있고 값 $f(0), f(1), \cdots, f(N-1)$과 정수 $c,M$이 주어졌을 때 $f(c), f(c+1), \cdots, f(c+M-1)$을 구하는 것을 **Sampling Points Shift**라고 부릅니다.

위의 두 문제는 조합론 문제를 해결할 때 종종 유용한 도구가 되어 줍니다. 두 문제 모두 단순하게 풀면 너무 느리지만, FFT를 사용한다면 효율적으로 해결할 수 있습니다.

이 글에서는 아래의 주제들을 다룹니다.

- $N$차 다항식에 대한 Taylor Shift를 $O(N \log N)$에 계산하기

- Sampling Points Shift를 $O((N+M) \log (N+M))$에 계산하기

- Sampling Points Shift를 활용해서 $N! \pmod {998244353}$을 $O(\sqrt{N} \log N)$에 계산하기

- Taylor Shift를 활용한 문제 풀이

이 글은 FFT의 작동 원리를 이해하지 않고 빠른 다항식 곱셈 라이브러리를 blackbox로 사용하더라도 읽는 데에 지장이 없도록 작성했습니다.



## [Polynomial Taylor Shift](https://judge.yosupo.jp/problem/polynomial_taylor_shift)

### 문제

길이가 $N$인 수열 $a = (a_0, a_1, \cdots, a_{N-1})$와 정수 $c$가 주어집니다.

차수가 $N-1$인 다항식 $f(x) = \sum_{i=0}^{N-1} a_i x^i$를 정의합시다.

이 때,  $f(x+c) = \sum_{i=0}^{N-1} b_i x^i$를 만족하는 수열 $b = (b_0, b_1, \cdots, b_{N-1})$를 구해야 합니다. 수가 커질 수 있으므로, $998244353$으로 나눈 나머지를 출력합니다.

$(1 \leq N \leq 2^{19} = 524288; 0 \leq c,a_i < 998244353)$

### 풀이

나이브하게 계산하면 $O(N^2)$로 너무 느립니다. $f(x+c)$를 전개한 뒤 수식을 잘 바꿔서 최적화해 봅시다.

이항정리에 의해 $f(x+c) = \sum_{i=0}^{N-1} a_i (x+c)^i = \sum_{i=0}^{N-1} \sum_{j=0}^{i} a_i \binom{i}{j} x^j c^{i-j}$가 됩니다.

이를 $b_j$에 대한 식으로 정리하면,

$b_j = \sum_{j \leq i} a_i \binom{i}{j} c^{i-j}$

$b_j = \sum_{j \leq i} a_i \frac{i!}{j!(i-j)!} c^{i-j}$

$b_j \cdot j! = \sum_{j \leq i} a_i \cdot i! \cdot \frac{ c^{i-j} } {(i-j)!}$

$A_i = a_i \cdot i!$와 $B_i = b_i \cdot i!$와 $C_i = \frac{c^i}{i!}$를 정의하면, $B_j = \sum_{j \leq i} A_i C_{i-j}$

$i' = N-1-i$를 정의하면, $B_{N-1-j'} = \sum_{i' \leq j'} A_{N-1-i'} C_{j'-i'}$

수열 뒤집기 연산 $\textrm{Rev}(D)_i = D_{N-1-i}$을 정의하면, $\textrm{Rev}(B)_{j'} = \sum_{i' \leq j'} \textrm{Rev}(A)_{i'} C_{j'-i'}$

이제 이 식은 convolution이므로 $\textrm{Rev}(B) = \textrm{Rev}(A) \ast C$로 적을 수 있고, FFT로 $O(N \log N)$에 계산 가능합니다.

### 코드

[**AtCoder Library에 내장된 Convolution 코드**](https://github.com/atcoder/ac-library/blob/master/document_en/convolution.md)를 사용한 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

vector<modular> polynomial_taylor_shift(const vector<modular> &a, modular c)
{
	int N = a.size();
	vector<modular> pw(N), fa(N), ifa(N), A(N), C(N);
	pw[0] = 1;
	fa[0] = 1;
	for(int i=1; i<N; i++)
	{
		pw[i] = pw[i-1] * c;
		fa[i] = fa[i-1] * i;
	}
	ifa[N-1] = 1 / fa[N-1];
	for(int i=N-2; i>=0; i--)
		ifa[i] = ifa[i+1] * (i+1);
	
	for(int i=0; i<N; i++)
	{
		A[i] = a[i] * fa[i];
		C[i] = pw[i] * ifa[i];
	}

	reverse(A.begin(), A.end());
	auto B = convolution(A, C);
	B.resize(N);
	reverse(B.begin(), B.end());

	for(int i=0; i<N; i++)
		B[i] *= ifa[i];
	return B;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int N, c;
	cin >> N >> c;
	vector<modular> a(N);
	for(int i=0,x; i<N; i++)
		cin >> x, a[i] = x;
	for(auto x : polynomial_taylor_shift(a, c))
		cout << x.val() << " ";
	cout << "\n";
	return 0;
}
```



## [Shift of Sampling Points of Polynomial](https://judge.yosupo.jp/problem/shift_of_sampling_points_of_polynomial)

### 문제

$N$차 미만의 다항식 $f(x)$가 숨겨져 있고 $N$개의 값 $f(0), f(1), \cdots, f(N-1)$과 정수 $c,M$이 주어집니다. (따라서 다항식 $f(x)$는 unique하게 결정됩니다.)

이 때 $f(c), f(c+1), \cdots, f(c+M-1)$을 각각 $998244353$으로 나눈 나머지를 출력합니다.

$(1 \leq N,M \leq 2^{19} = 524288; 0 \leq c,f(i) < 998244353)$

### 풀이

[**다음 글**](https://infossm.github.io/blog/2019/06/16/Multipoint-evaluation/)과 [**다음 글**](https://infossm.github.io/blog/2019/09/14/factor/)에서 소개된 Lagrange Interpolation과 Multipoint Evaluation을 사용하면 $O((N+M) \log^2 (N+M))$에 해결할 수 있음은 곧바로 알 수 있습니다. 하지만 이 문제의 특수한 성질을 이용해서 더 빠르게 해결할 것입니다.

#### Interpolation

$f(0), f(1), \cdots, f(N-1)$ 값을 이용해서 원래 다항식 $f(x)$를 복원할 수 있습니다.

$0 \leq i,x \leq N-1$에서 $\prod_{j=0, j \neq i}^{N-1} \frac{x-j}{i-j}$는 $x = i$일 때에만 $1$을 기여하고 $x \neq i$일 때에는 $0$을 기여합니다.

따라서 $f(x) = \sum_{i=0}^{N-1} f(i) \prod_{j=0, j \neq i}^{N-1} \frac{x-j}{i-j}$로 복원하면 됩니다. (명시적으로 복원하지는 않고, 이러한 성질만을 이용할 것입니다.)

#### Evaluation

이제 $0 \leq k < M$에 대해 $f(c+k) = \sum_{i=0}^{N-1} f(i) \prod_{j=0, j \neq i}^{N-1} \frac{c+k-j}{i-j}$를 구하면 됩니다.

$A_i = f(i) \frac{1}{i!} \frac{1}{(N-1-i)!} (-1)^{N-1-i}$를 정의하면 $f(c+k) = \sum_{i=0}^{N-1} A_i \prod_{j=0, j \neq i}^{N-1} (c+k-j)$로 다시 작성할 수 있습니다.

$(c+k-j)$들의 곱은 prefix sum을 이용해 $(c+k)(c+k-1) \cdots (c+k-N+1)$을 모두 곱해 두고 나중에 $(c+k-i)$로 나눠 주기로 하면 됩니다. 주의할 점으로, $(c+k-i) = 0$이 되어서 $0$으로 나눌 수 없는 경우에 대한 예외 처리를 해 주어야 합니다.

이제 $f(c+k) = \sum_{i=0}^{N-1} \frac{A_i}{c+k-i}$는 FFT로 계산 가능한 형태이므로 $O((N+M) \log (N+M))$에 전체 문제가 해결됩니다.

### 코드

코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/convolution>
using namespace std;
using namespace atcoder;
using modular = modint998244353;

vector<modular> sampling_points_shift(const vector<modular> &f, modular c, int M)
{
	int N = f.size();
	vector<modular> fa(N), ifa(N), A(N), B(N+M-1), S(N+M), res(M);
	fa[0] = 1;
	for(int i=1; i<N; i++)
		fa[i] = fa[i-1] * i;
	ifa[N-1] = 1 / fa[N-1];
	for(int i=N-2; i>=0; i--)
		ifa[i] = ifa[i+1] * (i+1);

	for(int i=0; i<N; i++)
		A[i] = f[i] * ifa[i] * ifa[N-1-i] * (N-1-i & 1 ? -1 : 1);
	for(int i=0; i<N+M-1; i++)
		B[i] = (c-N+1+i != 0 ? 1 / (c-N+1+i) : 0);
	auto C = convolution(A, B);

	S[0] = 1;
	for(int i=1; i<N+M; i++) // 1-based, [c-N+1, c+M-1]
		S[i] = S[i-1] * (c-N+i != 0 ? c-N+i : 1);

	for(int k=0; k<M; k++)
	{
		int i = -1;
		if ((c+k).val() <= N-1)
		{
			i = (c+k).val(); // (i != -1) => (c+k-i == 0)
			if (i+modular::mod() <= N-1) continue; // small mod, two zero case
		}
		if (i == -1) res[k] += C[k+N-1] * S[k+N] / S[k]; // non-zero case
		else res[k] += f[i]; // zero case
	}
	return res;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int N, M, c;
	cin >> N >> M >> c;
	vector<modular> f(N);
	for(int i=0,x; i<N; i++)
		cin >> x, f[i] = x;
	auto g = sampling_points_shift(f, c, M);
	for(int i=0; i<M; i++)
		cout << g[i].val() << " ";
	cout << "\n";
	return 0;
}
```

### 활용

Sampling Points Shift를 활용하면 $N! \pmod {998244353}$을 $O(\sqrt N \log N)$의 시간복잡도로 구할 수 있습니다.

정수 $v$를 $\sqrt{N}$ 이상에서 최초로 등장하는 $2^k$ 꼴의 수라고 정의합시다.

아이디어: 만약 다항식 $F(x) = (vx+1)(vx+2) \cdots (vx+v)$를 구할 수 있다면, $F(0) F(1) F(2) \cdots$를 차례로 곱해 나가면서 $N!$을 구할 수 있습니다.

새로운 다항식 $f_d(x) = (vx+1)(vx+2) \cdots (vx+d)$를 정의합시다. 앞으로는 다항식을 들고 있지 않고, $x$좌표와 그에 대응되는 $y$좌표 값들인 $\lbrace (0, f_d(0)), (1, f_d(1)), \cdots, (d, f_d(d)) \rbrace$를 대신 들고 있을 것입니다.

또한 수열 $A_d = (f_d(0), f_d(1), \cdots, f_d(d))$를 정의합시다.

$A_1$에서 시작해서 $A_d$를 $A_{2d}$로 변환하는 과정을 매번 $O(d \log d)$에 수행할 수 있다면, $F(x) = f_v(x), A_v = (F(0), F(1), \cdots, F(v))$이므로 전체 문제가 $O(v \log v) = O(\sqrt N \log N)$에 해결될 것입니다.

- $B_d = A_d$를 오른쪽으로 $d$칸 Sampling Points Shift한 값들

- $C_d = A_d$를 오른쪽으로 $(d+1)v$칸 Sampling Points Shift한 값들

- $D_d = A_d$를 오른쪽으로 $(d+1)v + d$칸 Sampling Points Shift한 값들

이렇게 새로운 세 수열 $B,C,D$를 정의하고 나면, $A_d \cdot B_d + C_d \cdot D_d$에서 맨 뒤 원소를 제거한 수열이 $A_{2d}$가 됩니다.

#### 코드

$N! \pmod {998244353}$을 테스트하는, `sampling_points_shift`를 제외한 코드는 아래와 같습니다. 이 문제의 악명과는 달리, 좋은 라이브러리와 함께라면 15줄 정도로 매우 간단하게 구현할 수 있습니다.

```cpp
using modular = modint998244353;

modular fact(int N)
{
	int v = 1;
	while(v * v < N) v *= 2;
	vector<modular> A = {1, v+1};
	for(int d=1; d<v; d*=2)
	{
		auto B = sampling_points_shift(A, modular(d)/v, A.size());
		auto C = sampling_points_shift(A, d+1, A.size());
		auto D = sampling_points_shift(A, d+1 + modular(d)/v, A.size());
		for(int i=0; i<=d; i++) A[i] *= B[i], C[i] *= D[i];
		A.resize(d+d+1);
		for(int i=d+1; i<=d+d; i++) A[i] = C[i-d-1];
	}
	modular ret = 1;
	int i = 0;
	while(i+v <= N) ret *= A[i/v], i += v;
	while(i+1 <= N) ret *= ++i;
	return ret;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int N;
	cin >> N;
	cout << fact(N).val() << "\n";
	return 0;
}
```



# 연습 문제

## [2023-2024 ICPC Brazil Subregional Programming Contest K. K for More, K for Less](https://www.acmicpc.net/problem/29996)

### 문제

두 다항식 $t(x)$와 $p(x)$가 주어지면, $q(x) = t(x+K) + p(x-K)$인 다항식 $q(x)$를 구하는 문제입니다.

### 풀이

$t$와 $p$에 대해 각각 Taylor Shift를 적용하면 곧바로 해결할 수 있습니다.

### 코드

`polynomial_taylor_shift`를 제외한 코드는 아래와 같습니다.

```cpp
using modular = modint998244353;

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int N, K;
	cin >> N >> K;
	vector<modular> t(N+1), p(N+1);
	for(int i=0,x; i<=N; i++)
		cin >> x, t[i] = x;
	for(int i=0,x; i<=N; i++)
		cin >> x, p[i] = x;

	t = polynomial_taylor_shift(t, K);
	p = polynomial_taylor_shift(p, -K);
	for(int i=0; i<=N; i++)
		cout << (t[i] + p[i]).val() << " ";
	cout << "\n";
	return 0;
}
```



## [solved.ac Grand Arena #3 Div1C. 교차 구간 크기 합](https://www.acmicpc.net/problem/30808)

### 풀이

좌표압축을 하고 각 단위 interval의 기여도를 세어 봅시다.

단위 interval $i$가 길이가 $L_i$이고 $C_i$개의 집합에서 등장한다면 $X_{C_i}$에 $L_i$를 더합시다.

이 때 $k$에 대한 정답은 $\sum_{k \leq i} \binom{i}{k} \cdot X_i$가 됩니다.

$f(x) = \sum_{i=0}^{N} X_i x^i$라 하면,

$f(x+1) = \sum_{i=0}^{N} X_i (x+1)^i = \sum_{i=0}^{N} \sum_{k=0}^{i} X_i \binom{i}{k} x^k$이므로,

$f(x+1)$의 $k$차항의 계수가 $k$에 대한 정답이 됩니다.

이제 Taylor Shift를 사용하면 $O(N \log N)$에 해결할 수 있습니다.

### 코드

`polynomial_taylor_shift`를 제외한 코드는 아래와 같습니다.

```cpp
using modular = modint998244353;

int N;
int l[300001], r[300001], C[600000];

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N;
	vector<int> com;
	for(int i=1; i<=N; i++)
	{
		cin >> l[i] >> r[i];
		r[i]++; // [l, r)
		com.push_back(l[i]);
		com.push_back(r[i]);
	}
	sort(com.begin(), com.end());
	com.erase(unique(com.begin(), com.end()), com.end());
	for(int i=1; i<=N; i++)
	{
		l[i] = lower_bound(com.begin(), com.end(), l[i]) - com.begin();
		r[i] = lower_bound(com.begin(), com.end(), r[i]) - com.begin();
		C[l[i]]++;
		C[r[i]]--;
	}
	for(int i=1; i<com.size(); i++)
		C[i] += C[i-1];

	vector<modular> X(N+1);
	for(int i=0; i+1<com.size(); i++)
		X[C[i]] += com[i+1] - com[i];

	auto res = polynomial_taylor_shift(X, 1);
	for(int i=1; i<=N; i++)
		cout << res[i].val() << "\n";
	return 0;
}
```


## [AtCoder Grand Contest 005 F. Many Easy Problems](https://atcoder.jp/contests/agc005/tasks/agc005_f)

### 문제

정점이 $N$개인 트리가 주어집니다.

각 $K = 1, 2, \cdots, N$마다, 트리에서 서로 다른 $K$개의 정점을 고르는 모든 $\binom{N}{K}$가지 경우에 대해, $K$개의 정점을 모두 포함하는 최소 크기의 connected subgraph의 크기의 합을 $924844033$(소수)으로 나눈 나머지를 구해야 합니다. (크기는 정점 개수로 정의됩니다.)

$(2 \leq N \leq 200,000)$

### 풀이

서브트리의 정점 개수가 $s$일 때 간선의 개수는 $s-1$개입니다. 각 간선의 기여도를 센 다음에 $\binom{N}{K}$는 따로 더해 줍시다.

간선을 지웠을 때 절단된 두 컴포넌트의 크기를 각각 $L, R$이라고 하면, $\binom{N}{K} - \binom{L}{K} - \binom{R}{K}$만큼이 답에 기여됩니다.

따라서 각 간선마다 $X_N$에 $1$을 더하고 $X_L, X_R$에 $1$을 뺀다면, $K$에 대한 정답은 $\sum_{K \leq i} \binom{i}{K} \cdot X_i$가 됩니다.

$f(x) = \sum_{i=0}^{N} X_i x^i$라 하면,

$f(x+1) = \sum_{i=0}^{N} X_i (x+1)^i = \sum_{i=0}^{N} \sum_{K=0}^{i} X_i \binom{i}{K} x^K$이므로,

$f(x+1)$의 $K$차항의 계수가 $K$에 대한 정답이 됩니다.

이제 Taylor Shift를 사용하면 $O(N \log N)$에 해결할 수 있습니다.

### 코드

$924844033 = 441 \cdot 2^{21} + 1$이기 때문에 AtCoder Library의 NTT를 사용할 수 있습니다.

`polynomial_taylor_shift`를 제외한 코드는 아래와 같습니다.

```cpp
using modular = static_modint<924844033>;

int N;
vector<int> g[200001];
int sz[200001];
vector<modular> X;

void DFS(int n, int prev)
{
	sz[n] = 1;
	for(int next : g[n])
	{
		if (next == prev) continue;
		DFS(next, n);
		sz[n] += sz[next];
		X[N]++;
		X[sz[next]]--;
		X[N-sz[next]]--;
	}
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N;
	for(int i=0; i<N-1; i++)
	{
		int a, b;
		cin >> a >> b;
		g[a].push_back(b);
		g[b].push_back(a);
	}
	X.resize(N+1);
	X[N]++;
	DFS(1, 0);
	auto res = polynomial_taylor_shift(X, 1);
	for(int i=1; i<=N; i++)
		cout << res[i].val() << "\n";
	return 0;
}
```



# 참고 자료

- <https://codeforces.com/blog/entry/77551>

- <https://codeforces.com/blog/entry/115696>

- <https://hyperbolic.tistory.com/5>

- <https://maspypy.com/%E5%A4%9A%E9%A0%85%E5%BC%8F%E3%83%BB%E5%BD%A2%E5%BC%8F%E7%9A%84%E3%81%B9%E3%81%8D%E7%B4%9A%E6%95%B0-%E9%AB%98%E9%80%9F%E3%81%AB%E8%A8%88%E7%AE%97%E3%81%A7%E3%81%8D%E3%82%8B%E3%82%82%E3%81%AE>

- <https://nyaannyaan.github.io/library/fps/sample-point-shift.hpp>