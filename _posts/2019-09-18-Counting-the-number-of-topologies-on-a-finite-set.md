---
layout: post
title:  "Counting the number of topologies on a finite set"
date:   2019-09-17 23:30:00
author: ainta
tags: [algorithm, mathematics, topology, problem-solving]



---

# Topology

수학에서, topological space란 다음 조건을 만족하는 집합 $$X$$와 $$X$$의 subset들의 collection $$\tau$$에 대해 ordered pair $$ (X, \tau) $$ 를 말한다.

1. $$ \phi, X \in \tau $$
2. Any arbitrary union of members of $$ \tau$$ still belongs to $$ \tau$$. ($$ \tau$$의 원소들의 합집합은 $$ \tau $$의 원소이다)
3. The intersection of any finite number of members of $$ \tau$$ still belongs to $$ \tau$$. ($$ \tau$$의 원소 유한개의 교집합은 $$ \tau $$의 원소이다)

이때 $$\tau$$의 원소들을 open set이라 하고, $$\tau$$를 topology on $$X$$라 한다.


## Counting the number of topology

만약 $$X$$가 유한집합이라면, topology on $$X$$의 개수도 당연히 유한할 것이다. 이 개수를 어떻게 counting할 수 있을까?

더 나아가, $$X$$의 어떤 subset들의 집합 $$A = \left\{a_1, a_2, .., a_n \right\} $$에 대해, $$A$$의 원소들을 모두 open set으로 가지는 topology의 개수를 효율적으로 구할 수 있을까?

이를 실제로 모든 topology를 구하지 않고 개수만 세는 것은 쉽지 않다. 그렇다면, 모든 topology를 나열하는 것을 빠르게 할 수 있을까?



다음은 필자가 Open cup, Grand Prix of Daejeon에 낸 문제이다.

[문제 링크](https://www.acmicpc.net/problem/17458)



topological space의 정의에서, 만약 $$X$$가 유한 개의 원소로 이루어져 있다면, 2번 조건 및 3번 조건을 각각 임의의 원소들의 합집합, 유한개 원소들의 교집합이 아닌 단 두 개의 원소의 합집합/교집합으로 바꿔도 같은 정의이다. 따라서, 문제에서 구하는 good set의 정의는 $$X = \left\{ 0, 1, ... , k-1 \right\}$$ 일 때, topology on $$X$$의 정의와 거의 일치한다. 공집합과 전체집합을 포함한다는 조건이 빠져 있는데, 이는 쉽게 처리할 수 있다.

그렇다면, 다음 문제를 해결하는 것으로 충분하다 : $$X = \left\{ 0, 1, ... , k-1 \right\}$$의 어떤 subset들의 집합 $$A = \left\{a_1, a_2, .., a_n \right\} $$에 대해, $$A$$의 원소들을 모두 open set으로 가지는 topology의 개수를 효율적으로 구할 수 있을까?

그래프 $$G = (V, E)$$ 가 $$(u, v), (v,w) \in E$$이면 $$(u, w) \in E$$를 만족할 때, $$G$$를 transitive digraph라고 한다. 놀랍게도, $$n$$개의 원소로 이루어진 집합 $$X$$에서 topology on $$X$$와 $$n$$개의 vertex로 이루어진 transitive digraph들은 일대일 대응 관계에 있다.

$$X = \left\{ 0, 1, ... , k-1 \right\}$$ 에서의 Topology $$T$$에 대해, 그래프 $$f(T) = (V, E)$$ 를 다음과 같이 정의하자 : 서로 다른 ​$$u, v$$에 대해, ​$$T$$의 ​$$u$$를 원소로 가지는 모든 open set들이 ​$$v$$ 역시 원소로 가질 때만 ​$$(u, v) \in E$$.

이 때, $$f(T)$$가 transitive digraph인 것은 정의에 의해 자명하다. 

$$V = \left\{ 0, 1, ... , k-1 \right\}$$인 transitive digraph $$G = (V, E)$$에 대해, $$V$$의 부분집합들의 집합 $$g(G)$$를 다음과 같이 정의하자. 

$$A \subset V$$인 $$A$$에 대해, $$A$$에서 $$A^C$$로 가는 간선이 없는 경우에만 $$A \in g(G)$$.

$$g(G)$$가 $$V$$에서의 topology임은 topology가 될 조건 1, 2, 3을 이용해 간단히 보일 수 있고, $$f(g(G)) = G$$ 및 $$g(f(F))=F$$가 성립하므로 $$f, g$$는 topology와 transitive digraph의 bijection이다. 따라서, 둘은 일대일 대응 관계에 있다.

그러면 이제 주어진 subset들을 포함하는 good set들을 구하는 대신에, $$g(G)$$가 주어진 모든 subset을 포함하는 transitive digraph의 개수를 세는 것으로 문제를 해결할 수 있다. backtracking으로 빠른 시간내에 모든 답을 열거할 수 있는데, 그래프의 edge를 $$(0,1), (1,0), (0,2), (2,0), (1,2), (2,1), ..., (k-3, k-1), (k-1, k-3), (k-2, k-1), (k-1, k-2)$$와 같이 $$max(u,v)$$가 작은 순서대로 추가할지 말지 결정하면서 backtracking을 하면 bitmask를 이용하는 경우 edge 하나를 추가할 지 말지 결정하고 다음 가지로 뻗어나가는 것까지 $$O(1)$$ 시간에 처리 가능하다.

총 시간복잡도를 생각해 보면, 크기가 $$m$$인 집합의 topology 개수를 $$T_m$$이라 하면  backtracking에서 하는 연산은 $$\sum_{i=1}^k T_i \times i$$ 개의 $$O(1)$$ 연산임을 알 수 있다. 

$$\left\{T_1, T_2, T_3, T_4, T_5, T_6, T_7\right\}$$ = $$ \left\{1, 4, 29, 355, 6942, 209527, 9535241\right\}$$ 이고  $$\sum_{i=1}^k T_i \times i \le 7 \times 10^7$$이므로, 빠른 시간 (1초) 이내에 조건을 만족하는 모든 good set을 탐색할 수 있다.



다음은 이 문제를 해결하는 코드이다.



```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
int w[8], AY[8], AN[8], BY[8], BN[8], res, cnt, E[8][8];
struct Edge {
	int a, b;
}Ed[60];
void DFS(int pv) {
	if (pv == cnt) {
		res++;
		return;
	}
	int x = Ed[pv].a, y = Ed[pv].b;
	if (!(AY[x] & BY[y])) { // (x,y) 간선을 추가하지 않는 경우
		AN[x] ^= (1 << y);
		BN[y] ^= (1 << x);
		DFS(pv + 1);
		AN[x] ^= (1 << y);
		BN[y] ^= (1 << x);
	}
	if (!(AN[x] & AY[y]) && !(BY[x] & BN[y]) && E[x][y]) { // (x,y) 간선을 추가하는 경우
		AY[x] ^= (1 << y);
		BY[y] ^= (1 << x);
		DFS(pv + 1);
		AY[x] ^= (1 << y);
		BY[y] ^= (1 << x);
	}
}
void Do(int n) {
	int i, j;
	for (i = 0; i < n; i++)AN[i] = BN[i] = 0, AY[i] = BY[i] = (1 << i);
	cnt = 0;
	for (i = 0; i < n; i++)for (j = 0; j < i; j++) {
		Ed[cnt++] = { j,i }; //ED에 있는 directed edge들의 순서대로 backtracking에서 간선을 추가할지 말지 선택한다.
		Ed[cnt++] = { i,j };
	}
	DFS(0);
}
int v[256], K, m;
void Calc(int LB, int MB) {
	int i, j, k, sz;
	vector<int>T;
	for (j = 0; j < K; j++) {
		if (((LB^MB) >> j) & 1) T.push_back(j); // LB에는 포함되지 않고 MB에는 포함되는 원소가 실제로 고려할 대상들이다.
	}
	sz = T.size();
	for (i = 0; i < sz; i++)for (j = 0; j < sz; j++)E[i][j] = 1;
	for (i = 0; i < (1 << K); i++) {
		if (!v[i])continue;
		if ((i&LB) != LB)return;
		if ((i&MB) != i)return;
		for (j = 0; j < sz; j++)for (k = 0; k < sz; k++)if (((i >> T[j]) & 1) && !((i >> T[k]) & 1)) E[k][j] = 0;
	}
	Do(sz);
}
int main() {
	int i, a, j, k;
	scanf("%d%d", &K, &m);
	for (i = 0; i < m; i++) {
		scanf("%d", &a);
		v[a] = 1;
	}
	for (i = 0; i < (1 << K); i++) {
		for (j = 0; j < (1 << K); j++) {
			if ((i&j) != i)continue;
			Calc(i, j); //topology에서 공집합 및 전체집합 포함 조건이 빠져있는데, i가 공집합에 해당하고 j가 전체집합에 해당하는 경우를 탐색하겠다는 뜻이다. 즉, good set 중 최소원소가 i고 최대원소가 j인 것의 개수를 구한다.
		}
	}
	printf("%d\n", res);
}
```







