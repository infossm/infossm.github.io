---
layout: post
title:  "Dulmage-Mendelsohn Decomposition (Part 2)"
date:   2023-07-25 23:00
author: edenooo
tags: [algorithm, graph-theory]
---

## 개요

[**이전에 작성한 글**](https://infossm.github.io/blog/2023/06/25/DM-decomposition/)에서는 Dulmage-Mendelsohn Decomposition(DM 분해)의 개념과 성질을 소개하고 이를 구하는 방법에 대해서 다루었습니다. 이 글에서는 Dulmage-Mendelsohn Decomposition의 코드와 PS에서의 활용에 대해 다룹니다.



## 코드

이전 글에서 언급한 구현 방법을 그대로 사용할 것입니다.

`BipartiteMatching`은 이분 그래프를 받아서 최대 매칭을 구한 뒤에 각 정점이 어느 정점과 매칭되었는지를 반환하는 함수입니다. Kuhn's Algorithm으로 $O(VE)$에 구현했지만, 더 빠른 시간복잡도를 원한다면 Hopcroft-Karp Algorithm 등의 다른 구현체로 대체해서 사용해도 됩니다.

`StronglyConnectedComponents`는 방향 그래프를 받아서 SCC들의 번호를 위상정렬 순으로 매기고, SCC의 개수와 함께 각 정점이 몇 번째 SCC에 속해 있는지를 반환하는 함수입니다. Kosaraju's Algorithm으로 $O(V+E)$에 구현했지만, 취향에 따라 Tarjan's Algorithm 등으로 대체해서 사용해도 됩니다.

`DulmageMendelsohn`은 이분 그래프가 주어지면 이전 글의 DM 분해 파트에서 언급한 $\lbrace (L_0, R_0), (L_1, R_1), \cdots, (L_{K+1}, R_{K+1}) \rbrace$을 반환하는 함수입니다. 여기에 덤으로 $0 \leq i \leq K+1, 0 \leq j < \min(\lvert L_i \rvert, \lvert R_i \rvert)$에서 정점 $L_{ij}$와 $R_{ij}$가 최대 매칭 $M$에서 서로 매칭되어 있도록 구현했기 때문에, 이 함수의 반환값만을 보고도 최대 매칭 $M$을 복원할 수 있습니다.

코드는 아래와 같습니다.

```cpp
pair<vector<int>, vector<int> > BipartiteMatching(int L, int R, const vector<pair<int, int> > &e) {
	vector<vector<int> > g(L);
	for(auto [a,b] : e)
		g[a].push_back(b);

	vector<int> matL(L, -1), matR(R, -1);
	vector<bool> vis(L);
	auto DFS = [&](auto &self, int n)->bool {
		if (vis[n]) return false;
		vis[n] = true;
		for(int next : g[n])
			if (matR[next] == -1)
			{
				matL[n] = next, matR[next] = n;
				return true;
			}
		for(int next : g[n])
			if (self(self, matR[next]))
			{
				matL[n] = next, matR[next] = n;
				return true;
			}
		return false;
	};

	for(int i=0; i<L; i++)
	{
		fill(vis.begin(), vis.end(), false);
		DFS(DFS, i);
	}
	return {matL, matR};
}

pair<int, vector<int> > StronglyConnectedComponents(int N, const vector<pair<int, int> > &e) {
	vector<vector<int> > g(N), rg(N);
	for(auto [a,b] : e)
		g[a].push_back(b), rg[b].push_back(a);

	vector<bool> vis(N);
	vector<int> stk;
	auto DFS = [&](auto &self, int n)->void {
		vis[n] = true;
		for(int next : g[n])
			if (!vis[next])
				self(self, next);
		stk.push_back(n);
	};
	for(int n=0; n<N; n++)
		if (!vis[n])
			DFS(DFS, n);

	int SCC = 0;
	vector<int> myscc(N);
	auto DFS2 = [&](auto &self, int n)->void {
		vis[n] = true;
		for(int next : rg[n])
			if (!vis[next])
				self(self, next);
		myscc[n] = SCC;
	};
	fill(vis.begin(), vis.end(), false);
	while(!stk.empty())
	{
		int n = stk.back(); stk.pop_back();
		if (vis[n]) continue;
		DFS2(DFS2, n);
		SCC++;
	}
	return {SCC, myscc};
}

vector<pair<vector<int>, vector<int> > > DulmageMendelsohn(int L, int R, const vector<pair<int, int> > &e) {
	auto [matL, matR] = BipartiteMatching(L, R, e);
	vector<int> mat(L+R);
	for(int i=0; i<L; i++)
		mat[i] = (matL[i] == -1 ? -1 : L+matL[i]);
	for(int i=L; i<L+R; i++)
		mat[i] = matR[i-L];

	vector<vector<int> > g(L+R);
	for(auto [a,b] : e)
	{
		g[a].push_back(L+b);
		g[L+b].push_back(a);
	}
	vector<int> col(L+R, -1); // (-1, 0, 1) = (U, E, O)
	auto DFS = [&](auto &self, int n)->void {
		col[n] = 0;
		for(int next : g[n])
		{
			col[next] = 1;
			if (col[mat[next]] == -1)
				self(self, mat[next]);
		}
	};
	for(int i=0; i<L+R; i++)
		if (mat[i] == -1)
			DFS(DFS, i);

	vector<pair<int, int> > E; // directed edges
	for(auto [a,b] : e)
		if (col[a] == -1 && col[L+b] == -1)
			E.push_back({a, L+b});
	for(int i=0; i<L; i++)
		if (col[i] == -1 && mat[i] != -1)
			E.push_back({mat[i], i});
	auto [K, myscc] = StronglyConnectedComponents(L+R, E);

	vector<pair<vector<int>, vector<int> > > ret(K+2);
	for(int i=0; i<L; i++)
		if (matL[i] != -1)
		{
			if (col[i] == 1) ret[0].first.push_back(i), ret[0].second.push_back(matL[i]);
			else if (col[i] == 0) ret[K+1].first.push_back(i), ret[K+1].second.push_back(matL[i]);
			else ret[myscc[i]+1].first.push_back(i), ret[myscc[i]+1].second.push_back(matL[i]);
		}
	for(int i=0; i<L; i++)
		if (matL[i] == -1)
			ret[K+1].first.push_back(i);
	for(int i=0; i<R; i++)
		if (matR[i] == -1)
			ret[0].second.push_back(i);
	return ret;
};
```



## 연습 문제

### [ABC223G. Vertex Deletion](https://atcoder.jp/contests/abc223/tasks/abc223_g)

트리가 주어지면, $u$를 사용하지 않는 최대 매칭이 존재하는 정점 $u$의 개수를 출력하는 문제입니다.

트리도 이분 그래프이므로, Dulmage-Mendelsohn Decomposition을 하고 나면 $\lvert \mathcal{E} \rvert$가 정답이 됩니다.

### 코드

코드는 아래와 같습니다.

```cpp
int N;
vector<int> g[200201];
vector<pair<int, int> > e;

void DFS(int n, int prev, int c)
{
	for(int next : g[n])
	{
		if (c == 0) e.push_back({n, next});
		if (next == prev) continue;
		DFS(next, n, c^1);
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
		a--; b--; // 0-based
		g[a].push_back(b);
		g[b].push_back(a);
	}
	DFS(0, 0, 0);
	auto v = DulmageMendelsohn(N, N, e);
	cout << v[0].second.size() + v.back().first.size() - N << "\n";
	return 0;
}
```



### [Selfish Spies 1](https://yukicoder.me/problems/no/1744)

이분 그래프가 주어지면, 각 간선 $e$에 대해 $e$를 사용하지 않는 최대 매칭이 존재한다면 `Yes`, 존재하지 않는다면 `No`를 출력하는 문제입니다.

Dulmage-Mendelsohn Decomposition을 하고 나서 $e$의 양 끝점이 크기가 2인 서로 같은 그룹에 속한다면 `No`, 그렇지 않다면 `Yes`를 출력하면 됩니다.

### 코드

코드는 아래와 같습니다.

```cpp
int N, M, L;

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M >> L;
	vector<pair<int, int> > e;
	for(int i=0; i<L; i++)
	{
		int a, b;
		cin >> a >> b;
		a--; b--; // 0-based
		e.push_back({a, b});
	}
	auto v = DulmageMendelsohn(N, M, e);
	vector<int> pos(N+M);
	for(int i=0; i<v.size(); i++)
	{
		for(int j : v[i].first) pos[j] = i;
		for(int j : v[i].second) pos[N+j] = i;
	}

	for(auto [a,b] : e)
	{
		if (pos[a] == pos[N+b] && v[pos[a]].first.size() + v[pos[a]].second.size() == 2) cout << "No\n";
		else cout << "Yes\n";
	}
	return 0;
}
```



### [Selfish Spies 2](https://yukicoder.me/problems/no/1745)

이분 그래프가 주어지면, 각 간선 $e$에 대해 $e$를 사용하는 최대 매칭이 존재한다면 `Yes`, 존재하지 않는다면 `No`를 출력하는 문제입니다.

Dulmage-Mendelsohn Decomposition을 하고 나서 $e$의 양 끝점이 서로 같은 그룹에 속한다면 `Yes`, 그렇지 않다면 `No`를 출력하면 됩니다.

### 코드

코드는 아래와 같습니다.

```cpp
int N, M, L;

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> N >> M >> L;
	vector<pair<int, int> > e;
	for(int i=0; i<L; i++)
	{
		int a, b;
		cin >> a >> b;
		a--; b--; // 0-based
		e.push_back({a, b});
	}
	auto v = DulmageMendelsohn(N, M, e);
	vector<int> pos(N+M);
	for(int i=0; i<v.size(); i++)
	{
		for(int j : v[i].first) pos[j] = i;
		for(int j : v[i].second) pos[N+j] = i;
	}

	for(auto [a,b] : e)
	{
		if (pos[a] == pos[N+b]) cout << "Yes\n";
		else cout << "No\n";
	}
	return 0;
}
```



### [2021 Japan Domestic Contest F. Princess' Perfectionism](https://www.acmicpc.net/problem/23727)

완전 매칭이 존재하는 이분 그래프가 주어지면, 어떤 간선 $e$를 선택하더라도 $e$를 사용하는 완전 매칭이 존재하는 이분 그래프가 되도록 최소 개수의 간선을 추가하는 문제입니다.

주어진 이분 그래프는 완전 매칭이 존재하므로 모든 정점이 $\mathcal{U}$에 속하게 됩니다. Dulmage-Mendelsohn Decomposition을 하고 나서 SCC들을 압축하여 만들어진 condensation graph(DAG) $G'$ 위에서 생각하면, 모든 간선의 양 끝점이 서로 같은 SCC에 속하도록 최소 개수의 간선을 추가하는 문제가 됩니다.

$G'$에서 indegree가 0인 정점을 source, outdegree가 0인 정점을 sink라고 부르겠습니다. 고립 정점(source이면서 동시에 sink인 정점)을 모두 무시하고 생각해도 동일한 문제가 됩니다. 문제의 조건을 만족시키려면 일단 source와 sink가 없도록 만들어야 하는데 새로 추가하는 간선 하나당 source와 sink의 개수를 각각 최대 하나씩만 감소시킬 수 있으므로, 답의 하한이 max(source의 개수, sink의 개수)임을 확인할 수 있습니다.

이제 이 하한이 실제로 달성 가능함을 구성적으로 보이겠습니다. 더 강한 버전의 문제인, $G'$가 strongly connected가 되도록 최소 개수의 간선을 추가하는 문제를 해결할 것입니다.

먼저, source에서 sink로 가는 극대 개수의 vertex-disjoint path들을 아무렇게나 구합니다. 최대가 아니라 극대만으로도 충분하기 때문에, DFS로 한 번 방문한 정점은 재방문하지 않는 그리디로 $O(V+E)$에 구할 수 있습니다. 이렇게 구한 path들을 각각 $S_0 \rightarrow T_0, S_1 \rightarrow T_1, \cdots, S_{k-1} \rightarrow T_{k-1}$이라 하면, 각 $0 \leq i < k$에 대해 $(T_i, S_{i+1 \bmod k})$ 형태의 간선 $k$개를 추가해서 하나의 거대한 사이클을 이루도록 합니다.

다음으로, 위 과정에서 어떤 path에도 포함되지 않은 min(source의 개수, sink의 개수)를 $l$이라 하고, 어떤 path에도 포함되지 않은 source들 중에 아무거나 $l$개를 뽑아서 각각 $\lbrace S'_0, S'_1, \cdots, S'_{l-1} \rbrace$이라 하고, 어떤 path에도 포함되지 않은 sink들 중에 아무거나 $l$개를 뽑아서 각각 $\lbrace T'_0, T'_1, \cdots, T'_{l-1} \rbrace$이라 합시다. 위에서 극대 개수의 vertex-disjoint path들을 골랐기 때문에 $S'_{i}$에서 거대 사이클로 가는 경로가 존재하고 거대 사이클에서 $T'_{i}$로 가는 경로가 존재합니다. 따라서 각 $0 \leq i < l$에 대해 $(T'_i, S'_i)$ 형태의 간선을 추가하면 모든 $S'_{i}$와 $T'_{i}$가 거대 사이클과 같은 SCC에 속하게 됩니다.

마지막으로, 아직 거대 사이클과 같은 SCC에 속하지 않은 source $s$가 존재한다면 $(T_0, s)$ 간선을 추가하고, 거대 사이클과 같은 SCC에 속하지 않은 sink $t$가 존재한다면 $(t, S_0)$ 간선을 추가하는 과정을 반복합니다. 이제 모든 source와 sink가 하나의 SCC에 속하기 때문에 모든 정점이 하나의 SCC에 속하게 되고, 전체 그래프가 strongly connected가 되었습니다.

condensation graph에서 새로 추가한 간선들을 주어진 이분 그래프에서 대응되는 간선들로 변환해 주면 전체 문제를 해결할 수 있습니다.

### 코드

코드는 아래와 같습니다.

```cpp
int N, M;
vector<int> g[2001]; // condensation graph
int posL[2001], posR[2001], out[2001], in[2001];
bool vis[2001];
vector<pair<int, int> > C; // 거대 사이클

bool DFS(int n)
{
	vis[n] = true;
	if (out[n] == 0)
	{
		C.push_back({-1, n});
		return true;
	}
	for(int next : g[n])
		if (!vis[next])
			if (DFS(next))
				return true;
	return false;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	while(1)
	{
		cin >> N >> M;
		if (N == 0 && M == 0) break;
		// clear
		for(int i=0; i<N+1; i++)
			g[i].clear();
		memset(out, 0, sizeof(int)*(N+1));
		memset(in, 0, sizeof(int)*(N+1));
		memset(vis, 0, sizeof(bool)*(N+1));
		C.clear();

		vector<pair<int, int> > e;
		for(int i=0; i<M; i++)
		{
			int a, b;
			cin >> a >> b;
			a--; b--; // 0-based
			e.push_back({a, b});
		}
		auto v = DulmageMendelsohn(N, N, e);
		assert(v.size() >= 3);
		assert(v[0].first.empty() && v[0].second.empty());
		assert(v.back().first.empty() && v.back().second.empty());
		int K = (int)v.size()-2;

		// make condensation graph
		for(int i=1; i<=K; i++)
		{
			for(int j : v[i].first) posL[j] = i;
			for(int j : v[i].second) posR[j] = i;
		}
		for(auto [a,b] : e)
		{
			a = posL[a], b = posR[b];
			if (a == b) continue;
			g[a].push_back(b);
			out[a]++, in[b]++;
		}

		// find maximal vertex-disjoint source-sink paths
		vector<int> S, T;
		for(int i=1; i<=K; i++)
			if (in[i] == 0 && out[i] != 0 && !vis[i])
			{
				if (DFS(i)) C.back().first = i;
				else S.push_back(i);
			}
		if (C.empty()) { cout << "0\n"; continue; } // 예외 처리

		for(int i=1; i<=K; i++)
			if (out[i] == 0 && in[i] != 0 && !vis[i])
				T.push_back(i);

		// 거대 사이클 만들기
		vector<pair<int, int> > res;
		for(int i=0; i<C.size(); i++)
			res.push_back({C[i].second, C[(i+1)%C.size()].first});

		// 남은 source와 sink 처리하기
		for(int i=0; i<max(S.size(), T.size()); i++)
			res.push_back({i < T.size() ? T[i] : C[0].second, i < S.size() ? S[i] : C[0].first});

		// condensation graph의 간선을 이분 그래프의 간선으로 변환해서 출력
		cout << res.size() << "\n";
		for(auto [a,b] : res)
			cout << v[a].first[0]+1 << " " << v[b].second[0]+1 << "\n"; // 1-based
	}
	return 0;
}
```



### [ARC161F. Everywhere is Sparser than Whole (Judge)](https://atcoder.jp/contests/arc161/tasks/arc161_f)

그래프의 density를 (간선의 개수) / (정점의 개수)로 정의하겠습니다.

density가 $D$인 단순 무향 그래프 $G = (V, E)$가 주어지면, $V$의 nonempty proper subset으로부터 만들어지는 어떤 vertex-induced subgraph에 대해서도 density가 $D$ 미만인 경우 Yes를, 그렇지 않다면 No를 출력하는 문제입니다.

먼저, "$D$ 미만" 대신에 "$D$ 이하"에 대해서 해결해 봅시다.

어떤 nonempty vertex subset으로부터 만들어지는 vertex-induced subgraph에 대해서도 density가 $D$ 이하라는 조건은, 어떤 nonempty edge subset으로부터 만들어지는 edge-induced subgraph에 대해서도 density가 $D$ 이하라는 조건과 동치입니다.

$\lvert E' \rvert / \lvert V' \rvert \leq D$는 $\lvert E' \rvert \leq \lvert V' \rvert \cdot D$로 바꿔 적을 수 있습니다.

이분 그래프의 정점 부분집합 $V'$에 대해 $V'$ 내부의 정점 중 하나 이상과 인접한 정점들의 집합을 $N(V')$라 정의하겠습니다.

$E$의 각 간선 $e = (u,v)$마다 $F$에 $(e,u), (e,v)$ 간선을 추가한 새로운 이분 그래프 $H = (E \sqcup V, F)$를 만들면, 모든 $\emptyset \subsetneq E' \subseteq E$에 대해 항상 $\lvert E' \rvert \leq \lvert N(E') \rvert \cdot D$인지를 판별하는 문제가 됩니다. 이는 $H$의 오른쪽 정점 집합을 $D$배 복제한 새로운 이분 그래프 $H'$에서는 $\lvert E' \rvert \leq \lvert N(E') \rvert$가 되고, [**홀의 결혼 정리**](https://infossm.github.io/blog/2022/03/20/latin-rectangle-hall/)에 의해 완전 매칭의 존재성을 판별하는 문제가 되어 최대 이분 매칭 알고리즘으로 해결할 수 있습니다. 완전 매칭이 존재하지 않았다면 density가 $D$ 초과인 경우가 존재하므로 답이 No임을 알 수 있습니다.

위 방법으로 원래의 문제인 "$D$ 미만"을 해결하지 못하는 이유는 전체 그래프의 density가 정확히 $D$라서 proper vertex subset만을 고려하기 어렵기 때문입니다. 그래도 위 방법으로 density가 $D$ 초과인 경우를 걸러냈기 때문에 이제는 density가 정확히 $D$가 되게 하는 nonempty proper vertex subset의 존재성만 판별해도 됩니다.

이제 $H'$는 반드시 완전 매칭이 존재하므로 Dulmage-Mendelsohn Decomposition을 하고 나면 모든 정점이 $\mathcal{U}$에만 속하게 됩니다. $\mathcal{U}$의 정점들을 SCC들로 압축해서 condensation graph $G'$를 만들면, $G'$의 SCC가 1개(=strongly connected)임이 답이 Yes일 필요충분조건입니다.

SCC가 2개 이상 $\Rightarrow$ 답이 No 증명: $G'$에서 outdegree가 0인 SCC에 대응되는 $H'$의 왼쪽 정점 부분집합 $E'$를 선택하면 $\lvert E' \rvert = \lvert N(E') \rvert$이므로 원래 그래프 $G$에서는 density가 정확히 $D$이고, proper이므로 답이 No입니다.

SCC가 정확히 1개 $\Rightarrow$ 답이 Yes 증명: $H'$에서 어떤 $\emptyset \subsetneq E' \subsetneq E$를 고르더라도 $G'$에서 하나의 SCC 안에 속하므로 $E'$에서 $E \setminus E'$로 가는 간선이 존재합니다. 따라서 $\lvert E' \rvert < \lvert N(E') \rvert$를 항상 만족하고 답이 Yes입니다.

### 코드

코드는 아래와 같습니다.

```cpp
int T;
int N, D;

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cin >> T;
	while(T--)
	{
		cin >> N >> D;
		vector<pair<int, int> > e;
		for(int i=0; i<D*N; i++)
		{
			int a, b;
			cin >> a >> b;
			a--; b--; // 0-based
			for(int j=0; j<D; j++)
			{
				e.push_back({i, j*N+a});
				e.push_back({i, j*N+b});
			}
		}
		auto v = DulmageMendelsohn(D*N, D*N, e);
		if (v[0].first.empty() && v[0].second.empty() && v.back().first.empty() && v.back().second.empty() && v.size() == 3) cout << "Yes\n";
		else cout << "No\n";
	}
	return 0;
}
```



## 참고 자료

- <https://hitonanode.github.io/cplib-cpp/graph/dulmage_mendelsohn_decomposition.hpp.html>