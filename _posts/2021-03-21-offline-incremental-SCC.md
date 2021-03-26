---
layout: post 
title: "Offline Incremental SCC" 
author: cheetose
date: 2021-03-21
tags: [algorithm, graph-theory]
---


본 글에서는 간선이 하나씩 추가됨에 따라 SCC를 관리하는 Incremental SCC를 오프라인으로 처리하는 방법에 대해서 설명하겠습니다.

[Link Cut Digraph](https://www.acmicpc.net/problem/19028) 문제를 보겠습니다. 문제를 간단하게 요약하자면 $N$개의 정점이 있고 $M$개의 간선을 추가하는 쿼리가 있을 때, 간선을 추가할 때마다 u에서 v로 가는 경로가 있고, v에서 u로 가는 경로가 존재하는 (u, v)(u < v, u != v)쌍의 개수를 구하는 문제입니다.

방향 그래프에서 u에서 v로, v에서 u로 갈 수 있다는 것은 u와 v가 서로 같은 SCC에 있다는 것을 의미합니다. 따라서 위 문제는 아래 문제로 바꿀 수 있습니다.

> 매 쿼리마다 SCC의 개수를 $K$개라고 하고, 각 SCC에 속한 정점의 개수를 $a_1$, $a_2$, ... , $a_K$개라고 했을 때, $\sum_i \binom{a_i}{2}$를 구하시오.

#### Naive 풀이

Naive 풀이를 생각해봅시다. 가장 쉽게 생각할 수 있는 풀이는 간선이 추가될 때마다 SCC를 새롭게 구하고, 각 SCC의 원소의 개수를 구해 위의 식을 계산하는 방법이 있습니다. 그렇게 어려운 내용은 아니기 때문에 자세한 내용은 생략하겠습니다. 혹시 SCC를 모르시는 분은 [이 글](http://www.secmem.org/blog/2019/04/10/Graph-SCC-BCC/)에서 공부하실 수 있습니다. 해당 알고리즘을 구현해보면 다음과 같습니다.

```cpp
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;

vector<int> v[100001], st;
int num[100001], low[100001], chk[100001], cnt;
ll ans;
inline ll nC2(int x){
	return 1LL * x * (x - 1) / 2;
}
void dfs(int n){
	chk[n] = 1;
	st.push_back(n);
	num[n] = ++cnt;
	low[n] = cnt;
	for (int next : v[n]){
		if (num[next] == 0){
			dfs(next);
			low[n] = min(low[n], low[next]);
		}
		else if (chk[next])
			low[n] = min(low[n], num[next]);
	}
	if (num[n] == low[n]){
		int t = 0; // 해당 SCC에 속하는 정점의 개수
		while (!st.empty()){
			t++;
			int x = st.back();
			st.pop_back();
			chk[x] = 0;
			if (n == x)
				break;
		}
		ans += nC2(t);
	}
}
int main(){
	int n, m;
	scanf("%d%d", &n, &m);
	for(int i = 0; i < m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		v[x].push_back(y);

		ans = cnt = 0; // 초기화
		for(int j = 1; j <= n; j++){
			num[j] = low[j] = 0; // 초기화
		}
		for(int j = 1; j <= n; j++){
			if(!num[j]) dfs(j);
		}
		printf("%lld\n", ans);
	}
}
```

이 때의 시간복잡도를 계산해보면 $M$번 간선이 추가될 때마다 $O(N+M)$의 시간복잡도를 갖는 SCC 알고리즘이 수행되므로 총 시간복잡도는 $O(M(N+M))$가 됩니다. 이 문제의 $N$ 제한과 $M$ 제한이 각각 $1 \leq N \leq 10^5$, $1 \leq M \leq 2.5 \times 10^5$이므로 시간초과를 받게 됩니다.

#### 알고리즘 개선

여기서 중요한 관찰을 해야합니다. 어떤 두 정점 u, v가 같은 SCC 내에 속해 있는 상태라고 할 때, 임의의 간선을 추가했을 때 역시 u, v는 같은 SCC 내에 속해있을 것입니다. 따라서 같은 SCC에 속해있는 정점들을 유니온파인드로 관리할 수 있습니다. 이를 활용하기 위해 다음과 같은 질문을 생각해볼 수 있습니다.

> u에서 v로 가는 간선 (u, v)가 있을 때, u와 v가 같은 SCC에 **처음** 속하게 되는 것은 몇 번째 간선이 추가되었을 때인가?

모든 간선에 대해서 위의 질문에 대한 답을 안다면 각 쿼리 별로 합쳐지는 간선들의 양 끝점을 유니온 파인드로 합쳐가며 각 SCC에 있는 정점의 개수를 업데이트하며 정답을 구할 수 있습니다.

이를 병렬 이분 탐색을 이용하면 구할 수 있을 것 같지만 계속해서 SCC를 구해야하기 때문에 이 역시 시간이 오래 걸려 좋은 방법은 아닙니다. 대신에 병렬 이분 탐색의 아이디어만 살짝 채용하는(?) 분할정복을 통해 구해보고자 합니다.

$solve(L, R, edge)$를 edge 벡터에 있는 간선들이 구간 $[L, R]$내에서 **처음** 합쳐진다는 것을 의미하는 함수라고 해봅시다. $M=(L+R)/2$라고 했을 때, $[1, M]$ 내의 간선을 전부 합쳐 SCC를 만들었을 때 edge 벡터에 있는 간선 중에서 $[L, R]$번째 간선임과 동시에 양 끝점이 같은 SCC에 속한 간선들은 $[L, M]$에서 처음 합쳐지는 간선이고 그렇지 않은 간선들은 $[M+1, R]$에서 처음 합쳐지는 간선입니다. 따라서 edge 벡터 내의 간선을 위의 조건에 따라 두 개의 집합으로 나눈뒤 다시 분할정복을 진행해주면 됩니다.

이제 마지막으로 남은 문제는 $[1, M]$의 간선을 합친 결과를 빠르게 구하는 것입니다. 이는 분할정복을 preorder 느낌으로 구현하면서 리프 노드에 도달했을 때 같은 SCC에 속하는 정점들을 합쳐준다면 $solve(L, R, edge)$가 끝날 때 $[1, R]$에 속하는 모든 간선을 이용해 만드는 SCC가 합쳐지게됩니다. 즉, $solve(L, R, edge)$를 보고있다면 $[1, L-1]$에서 처음 합쳐지는 간선들의 양 끝점은 이미 유니온-파인드를 통해 합쳐진 상태이고 이를 이용하여 더 적은 간선을 이용해서 SCC를 구현할 수 있습니다.

위의 내용들을 종합해서 아래와 같이 구현할 수 있습니다. 주석에 더 자세한 설명을 하겠습니다.

```cpp
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;

struct edge {
	int x, y;
}edge[250000];
int parent[100005];
int find(int x) {
	if (parent[x] == x)return x;
	return parent[x] = find(parent[x]);
}
void merge(int x, int y) {
	x = find(x), y = find(y);
	if (x != y) {
		parent[x] = y;
	}
}

vector<int> v[100005], tmp; // tmp에는 SCC를 구하고자하는 정점들을 모아둡니다.
vector<int> st;
int num[100005], low[100005], sn[100005], cnt, SN;
bool chk[100005];
inline void add_edge(int i, int j) {
	v[i].push_back(j);
	tmp.push_back(i);
	tmp.push_back(j);
}
void dfs(int n) {
	chk[n] = 1;
	st.push_back(n);
	num[n] = ++cnt;
	low[n] = cnt;
	for (int next : v[n]) {
		if (num[next] == 0) {
			dfs(next);
			low[n] = min(low[n], low[next]);
		}
		else if (chk[next])
			low[n] = min(low[n], num[next]);
	}
	if (num[n] == low[n]) {
		while (!st.empty()) {
			int x = st.back();
			st.pop_back();
			sn[x] = SN;
			chk[x] = 0;
			if (n == x)
				break;
		}
		SN++;
	}
}

void scc_clear() {	// solve(L,R,v)에서 처음에 한 번 실행시키는데 이는 [1, L-1] 간선이 이미 합쳐진 상태이므로 SCC를 새로 구하는데 더 이상 필요가 없어서 tmp에 담겨있는 정점들을 없애줍니다.
					// 이 과정을 통해 모든 정점과 간선이 O(log M)번만 SCC를 구하는데에 사용됩니다.
	for (int i : tmp)v[i].clear();
	tmp.clear();
}

void get_scc() {
	for (int i : tmp)num[i] = low[i] = sn[i] = chk[i] = 0;
	cnt = 0, SN = 0;
	for (int i : tmp)if (!num[i])dfs(i);
}


int n, m;
void solve(int S, int E, vector<int>& v) {
	if (S == E) { // 리프 노드에서는 v에 속해있는 모든 간선들이 S번째 간선이 추가될 때 합쳐진다는 뜻입니다.
		for (int i : v) {
			merge(edge[i].x, edge[i].y);
		}
		return;
	}
	int M = (S + E) / 2;
	scc_clear();
	vector<int> lv, rv;
	for (int i : v) { // [1, M]에 속하는 간선을 전부 합칩니다. 이 떄 [1, L-1]에서 합쳐진 간선의 양 끝점은 유니온 파인드에서 같은 집합에 있으므로 이를 활용합니다. 
		if (i > M)continue;
		auto [x, y] = edge[i];
		x = find(x), y = find(y);
		add_edge(x, y);
	}
	get_scc();
	for (int i : v) {
		if (i > M) {
			rv.push_back(i);
			continue;
		}
		auto [x, y] = edge[i];
		x = find(x), y = find(y);
		if (sn[x] == sn[y])lv.push_back(i); // 둘의 scc number가 같다면 [L, M]에서 합쳐진다는 의미이므로 lv에 추가하고
		else rv.push_back(i); // 아니라면 rv에 추가합니다.
	}
	solve(S, M, lv);
	solve(M + 1, E, rv);
}
int main() {
	scanf("%d%d", &n, &m);
	for (int i = 1;i <= n;i++)parent[i] = i;
	for (int i = 0;i < m;i++)scanf("%d%d", &edge[i].x, &edge[i].y);
	vector<int> v(m);
	iota(v.begin(), v.end(), 0);
	solve(0, m, v);
}
```

주석에 잠깐 언급하고 간 내용이지만 모든 정점과 간선이 $O(log M)$번 SCC를 구하는 데에 사용되고, SCC의 시간복잡도가 $O(N+M)$이므로 총 시간복잡도는 $O((N+M)log M)$이 됩니다.

실제 각 SCC 내에 원소들이 몇 개씩 있는지는 관리하지 않았지만 매우 쉽게 할 수 있으니 간단한 숙제로 남겨두겠습니다.

이 알고리즘을 이용한 다른 문제를 하나 소개하고 글을 마치겠습니다.

[Godzilla][https://www.acmicpc.net/problem/8496], ONTAK 2009

번역이 안되어있기 때문에 문제를 간단하게 설명을 하자면, 방향그래프가 있을 때 어떤 간선을 $Q$번 제거함에 따라 indegree가 0인 SCC의 개수를 구하는 문제입니다.

$Q<M$일 수도 있기 때문에 지워지지 않는 간선들도 지워진다고 가정하여 $Q=M$으로 만들고 쿼리를 역순으로 처리하면 간선이 증가하는 문제로 바꿀 수 있습니다. indegree를 관리하는 방법 역시 숙제로 남겨두겠습니다.