---
layout: post
title:  "플로우 그래프 커팅"
date:   2023-12-25 21:00
author: edenooo
tags: [algorithm, graph-theory, maximum-flow]
---

# 개요

Problem Solving을 하다 보면, 주어진 문제에서 플로우 그래프를 모델링했을 때 그래프의 크기가 너무 커서 단순히 플로우 알고리즘을 사용할 경우에 시간 초과를 받게 되는 상황이 종종 발생합니다.

이 글에서는 위와 같은 상황에서 추가적인 관찰을 통해 플로우 그래프의 크기를 줄이는 몇 가지 방법을 소개합니다.

문제마다 필요한 관찰과 해결 방법이 제각기 다를 수 있으므로 다양한 연습 문제를 예시로 들어서 설명하겠습니다.

디닉 알고리즘의 구현체를 통일하기 위해 AtCoder Library의 [**MaxFlow**](https://github.com/atcoder/ac-library/blob/master/document_en/maxflow.md)를 사용하겠습니다.



# 연습 문제

## [AtCoder Beginner Contest 320 G. Slot Strategy 2 (Hard)](https://atcoder.jp/contests/abc320/tasks/abc320_g)

### 문제

바퀴가 $N$개인 슬롯 머신이 주어집니다.

$i$번 바퀴는 $M$개의 숫자로 이루어진 문자열 $S_i$를 갖고 있습니다. $(1 \leq i \leq N)$

정수 시각 $t \geq 0$마다 바퀴를 하나 골라서 멈추거나, 또는 아무 행동도 하지 않을 수 있습니다.

시각 $t$에 $i$번 바퀴를 멈추면, $i$번 바퀴는 영구적으로 숫자 $S_{i,(t \bmod M) + 1}$를 표시하게 됩니다.

모든 바퀴를 정확히 한 번씩 멈추어서 표시되는 $N$개의 숫자가 서로 같도록 하고 싶습니다. 목표를 달성할 수 있는 최초의 시각을 출력해야 합니다. (불가능하면 대신 -1을 출력)

$(1 \leq N \leq 100; 1 \leq M \leq 10^5; 0 \leq S_{i,j} \leq 9)$

### 느린 풀이

이분 탐색을 해서 "답이 $X$ 이하인가?" 라는 결정 문제로 바꾸어 해결하겠습니다.

각 숫자 $x$에 대해 독립적으로 답을 구할 것입니다.

바퀴를 왼쪽 정점으로, 시각을 오른쪽 정점으로 하는 이분 그래프를 생각합시다.

바퀴 $i$가 시각 $t$에 표시하는 숫자가 $x$일 때 바퀴 $i$와 시각 $t$를 잇는 간선을 만들면 완전 매칭의 존재성 판별 문제가 됩니다.

만약 완전 매칭이 가능하다면 시간이 $M$만큼 지날 때마다 적어도 한 번씩은 매칭이 가능하므로, $NM$ 이상의 시각은 검사할 필요가 없습니다.

따라서 정점이 $O(NM)$개, 간선이 $O(NM)$개인 그래프에서 max flow를 구하면 최종 시간복잡도는 $O(\log NM \cdot 10 \cdot N^2M)$으로, 시간 초과를 받게 됩니다.

### 빠른 풀이

관찰: 바퀴 $i$의 degree가 $N$ 초과라면, 연결된 가장 작은 $N$개의 시각을 제외한 다른 모든 시각으로 가는 간선을 끊어 버려도 됩니다.

위 관찰은 귀류법으로 간단하게 증명할 수 있습니다. 최대 매칭이 $N$ 이하이므로, 만약 큰 시각을 사용해서 완전 매칭을 만들었다면 다른 작은 시각으로 교체해 줄 수 있기 때문입니다.

이제 정점이 $O(N^2)$개, 간선이 $O(N^2)$개인 그래프에서 max flow를 구하면 최종 시간복잡도는 $O(\log NM \cdot 10 \cdot N^3)$으로 정답을 받을 수 있습니다.

### 코드

AtCoder Library의 MaxFlow를 제외한 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/maxflow>
using namespace std;
using namespace atcoder;

int N, M;
string s[100];
vector<int> v[100][10];
int ord[10][10000000], P[10];

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> N >> M;
	for(int i=0; i<N; i++)
	{
		cin >> s[i];
		for(int j=0; j<M; j++)
			v[i][s[i][j]-'0'].push_back(j);
	}

	int lo = 0, hi = N*M+1;
	for(int t=0; t<25; t++)
	{
		int mid = lo+hi>>1;
		// [0, mid)초만을 사용해서 완전 매칭이 가능한가?
		int ok = 0;
		for(int x=0; x<10; x++) // 숫자 x만을 사용해서 가능한가?
		{
			mf_graph<int> graph(N+N*N+2);
			int src = N+N*N, snk = src+1;
			for(int i=0; i<N; i++) graph.add_edge(src, i, 1);
			for(int i=0; i<N*N; i++) graph.add_edge(N+i, snk, 1);
			for(int i=0; i<N; i++)
			{
				int len = v[i][x].size();
				if (len == 0) continue;
				for(int j=0; j<N; j++)
				{
					int k = v[i][x][j%len] + j/len * M;
					if (mid <= k) break;
					if (!ord[x][k]) ord[x][k] = ++P[x];
					graph.add_edge(i, N+ord[x][k]-1, 1);
				}
			}
			if (graph.flow(src, snk) == N) { ok = 1; break; }
		}
		if (ok) hi = mid;
		else lo = mid;
	}
	if (hi == N*M+1) cout << "-1\n";
	else cout << hi-1 << "\n";
	return 0;
}
```



## [BAPC 2018 I. In Case of an Invasion, Please...](https://www.acmicpc.net/problem/16312)

### 문제

정점이 $N$개, 간선이 $M$개인 간선 가중치 무방향 그래프가 주어집니다.

간선의 가중치가 $w$라면 이 간선을 지나갈 때 $w$의 시간이 걸립니다.

$i$번 정점에는 사람이 $p_i$명 살고 있습니다.

이 그래프에는 $S$개의 쉘터가 있고, $i$번 쉘터는 $s_i$번 정점에 위치하며 사람 $c_i$명을 수용 가능합니다.

이 때, 모든 사람이 쉘터로 대피하기 위해 걸리는 시간을 최소화해야 합니다. (항상 대피 방법이 존재함이 보장됩니다.)

$(1 \leq N \leq 10^5; 0 \leq M \leq 2 \cdot 10^5; 1 \leq S \leq 10; p_i, c_i, w \leq 10^9)$

### 느린 풀이

이분 탐색을 해서 "답이 $X$ 이하인가?" 라는 결정 문제로 바꾸어 해결하겠습니다.

각 쉘터를 source로 하는 다익스트라를 돌려서 최단경로 배열을 미리 전처리해 두면, 각 정점마다 이 쉘터로 시간 $X$ 이내에 이동할 수 있는지를 즉시 판별할 수 있습니다.

이제 사람을 왼쪽에 두고 쉘터를 오른쪽에 둔 이분 그래프를 만들면 max flow가 $\sum p_i$와 같은지 판별하는 문제가 됩니다.

여기에서 매번 단순히 max flow를 구한다면 플로우 그래프의 정점이 $O(N+S)$개, 간선이 $O(NS)$개로 충분히 많기 때문에 시간 초과가 발생합니다.

### 빠른 풀이

관찰: $N$은 큰 반면에 $S$는 매우 작다는 점을 활용할 수 있습니다.

두 왼쪽 정점이 갖는 "인접한 쉘터들의 집합"이 서로 같다면, 두 정점의 capacity를 합쳐서 하나의 정점으로 바꾸어 줄 수 있습니다.

서로 다른 "인접한 쉘터들의 집합"은 $2^S$ 가지 이하이므로, $N$이 $2^S$ 이하로 줄어들게 되고 시간 제한 안에 문제를 해결할 수 있습니다.

### 코드

AtCoder Library의 MaxFlow를 제외한 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/maxflow>
using namespace std;
using namespace atcoder;

#define ll long long

int N, M, S;
int p[100001], C[10];
vector<pair<int, int> > g[100001];
ll D[10][100001];

void Dijkstra(ll dist[], int src)
{
	priority_queue<pair<ll, int>, vector<pair<ll, int> >, greater<pair<ll, int> > > pq;
	pq.push({0, src}), dist[src] = 0;
	while(!pq.empty())
	{
		auto [d,n] = pq.top(); pq.pop();
		if (dist[n] < d) continue;
		for(auto [next,cost] : g[n])
			if (dist[next] > dist[n] + cost)
				pq.push({dist[n] + cost, next}), dist[next] = dist[n] + cost;
	}
}

int main()
{
	memset(D, 0x3f, sizeof(D));
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> N >> M >> S;
	for(int i=1; i<=N; i++)
		cin >> p[i];
	for(int i=0; i<M; i++)
	{
		int a, b, c;
		cin >> a >> b >> c;
		g[a].push_back({b, c});
		g[b].push_back({a, c});
	}
	for(int i=0; i<S; i++)
	{
		int s;
		cin >> s >> C[i];
		Dijkstra(D[i], s);
	}

	ll lo = 0, hi = (ll)1e14;
	for(int t=0; t<50; t++)
	{
		ll mid = lo+hi>>1;
		// mid 시간 이내에 모두 shelter로 이동 가능한가?
		ll A[1024] = {0, };
		for(int i=1; i<=N; i++)
		{
			int b = 0;
			for(int k=0; k<S; k++)
				if (D[k][i] <= mid)
					b |= 1<<k;
			A[b] += p[i];
		}

		mf_graph<ll> graph(S + (1<<S) + 2);
		int src = S + (1<<S), snk = src + 1;
		for(int i=0; i<(1<<S); i++)
			if (A[i])
			{
				graph.add_edge(src, S+i, A[i]);
				for(int k=0; k<S; k++)
					if (i & 1<<k)
						graph.add_edge(S+i, k, (ll)1e18);
			}
		for(int k=0; k<S; k++)
			graph.add_edge(k, snk, C[k]);

		if (graph.flow(src, snk) == accumulate(A, A+(1<<S), 0LL)) hi = mid;
		else lo = mid;
	}
	cout << hi << "\n";
	return 0;
}
```



## [NAIPC 2017 G. Apple Market](https://www.acmicpc.net/problem/14512)

### 문제

$N \times M$ 격자가 주어집니다. $i$행 $j$열에 있는 상점에서는 $a_{i,j}$개 이하의 사과를 판매할 수 있습니다.

사람이 $K$명 있습니다. 각 사람마다 $t,b,l,r,x$로 5개의 값이 주어지는데, 해당 사람은 $(t,l)$과 $(b,r)$을 대각선의 양 끝점으로 하는 직사각형 내의 상점에서만 총합 $x$개 이하의 사과를 구매할 수 있음을 의미합니다.

각 상점이 각 사람에게 사과를 몇 개 판매할지를 모두 잘 결정해서 판매되는 사과의 개수의 총합을 최대화해야 합니다.

$(1 \leq N,M \leq 50; 1 \leq K \leq 10^5; 0 \leq a_{i,j},x \leq 10^9)$

### 풀이

문제에서 주어진 그대로 최대 매칭을 구현하면 정점이 $O(NM + K)$개, 간선이 $O(NMK)$로 간선이 너무 많아 시간 초과가 발생합니다. (사람,상점) 쌍을 간선으로 일일이 잇지 말고 직사각형 형태임을 이용해서 최적화해야 합니다.

2D sparse table을 만드는 느낌으로 간선의 개수를 최적화할 것입니다.

전처리 과정으로 $1 \leq i \leq N, 1 \leq j \leq M, 0 \leq p < 6, 0 \leq q < 6$를 만족하는 각 $(i,j,p,q)$ tuple마다, $[i,i+2^p) \times [j,j+2^q)$ 모양의 직사각형을 대표하는, 다시 말해 이 직사각형 내의 모든 상점으로 플로우를 흘릴 수 있는 정점을 하나씩 생성합시다.

각 정점 $(i,j,p,q)$마다,

- $(i,j,p-1,q-1)$ (왼쪽 위 subrectangle)

- $(i+2^{p-1},j,p-1,q-1)$ (왼쪽 아래 subrectangle)

- $(i,j+2^{q-1},p-1,q-1)$ (오른쪽 위 subrectangle)

- $(i+2^{p-1},j+2^{q-1},p-1,q-1)$ (오른쪽 아래 subrectangle)

으로 가는 4개 이하의 간선만 만들어 주어도 재귀적으로 흘러 내려가면서 직사각형 범위 내의 모든 상점으로 플로우를 흘릴 수 있게 됩니다.

이제 각 사람마다 대응되는 $[t,b] \times [l,r]$ 직사각형은, 기존에 생성했던 서로 overlap되어도 상관없는 4개 이하의 subrectangle로 완전히 표현 가능하므로, 사람 하나당 $1 + 4 = 5$개 이하의 간선만을 추가해도 됩니다.

이렇게 간선 개수가 크게 줄어든 새로운 그래프에서 max flow를 구하면 전체 문제를 해결할 수 있습니다.

### 코드

AtCoder Library의 MaxFlow를 제외한 코드는 아래와 같습니다.

```cpp
#include<bits/stdc++.h>
#include<atcoder/maxflow>
using namespace std;
using namespace atcoder;

#define INF ((ll)1e14)
#define ll long long

int N, M, K, src = 0, snk = 1, V = 2;
int A[50][50];
int dp[50][50][50][50];
mf_graph<ll> graph(50*50*6*6 + 100000 + 2);

int make_rec(int u, int d, int l, int r)
{
	int &ret = dp[u][d][l][r];
	if (ret) return ret;
	ret = V++;
	if (u == d && l == r) graph.add_edge(ret, snk, A[d][r]);
	else
	{
		int dy = 1, dx = 1;
		if (u != d) dy = 1<<__lg(d-u);
		if (l != r) dx = 1<<__lg(r-l);
		graph.add_edge(ret, make_rec(u, u+dy-1, l, l+dx-1), INF);
		if (u != d) graph.add_edge(ret, make_rec(d-dy+1, d, l, l+dx-1), INF);
		if (l != r) graph.add_edge(ret, make_rec(u, u+dy-1, r-dx+1, r), INF);
		if (u != d && l != r) graph.add_edge(ret, make_rec(d-dy+1, d, r-dx+1, r), INF);
	}
	return ret;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> N >> M >> K;
	for(int i=0; i<N; i++)
		for(int j=0; j<M; j++)
			cin >> A[i][j];
	for(int i=0; i<K; i++)
	{
		int u, d, l, r, x;
		cin >> u >> d >> l >> r >> x;
		u--; d--; l--; r--; // 0-based
		graph.add_edge(src, make_rec(u, d, l, r), x);
	}
	cout << graph.flow(src, snk) << "\n";
	return 0;
}
```



# 추가 문제

글이 길어져서 풀이는 생략하지만, 아래 문제들도 플로우 그래프의 크기를 줄여서 해결할 수 있으니 풀어 보시기 바랍니다.

- [**초콜릿과 친구들의 습격**](https://www.acmicpc.net/problem/25798)

- [**병사 분배**](https://www.acmicpc.net/problem/29772)

- [**Educational Codeforces Round 24 G. Four Melodies**](https://codeforces.com/contest/818/problem/G)

- [**Ada and Replant**](https://www.spoj.com/problems/ADAGROW/)