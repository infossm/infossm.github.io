---
layout: post
title: "Small To Large Merging"
date: 2021-05-18 23:00:00
author: JooDdae
tags: [algorithm]

---



# Small To Large Merging

"작은거를 큰거에 합친다", 이름만 들으면 무엇인지 유추가 잘 되지 않습니다. 하지만 알고리즘을 공부하는 사람이라면 한 번쯤은 간접적으로라도 접한적이 있을 것입니다. 바로 Union Find의 Union by size에서 시간복잡도를 보장하기 위해 쓰이기 때문입니다. 알고리즘에서 소개한대로 작은 집합을 큰 집합에 합친다면 한번의 Find 연산이 $O(\log N)$만큼 걸린다는 것이 증명되어 있습니다. 이 글에서는 Small to Large Merging을 소개하면서 왜 이런 시간복잡도가 보장되는지 알아보고, 응용되는 문제의 풀이를 설명하겠습니다.



간단한 문제를 하나 풀면서 시작하겠습니다. 무작위의 수가 1개씩 들어있는 $N$개의 집합이 주어지고, 두가지 쿼리를 해결해야 합니다. 첫번째 쿼리는 서로 다른 두 집합을 하나로 합치는 것이고, 두번째 쿼리는 집합에 포함된 수의 종류를 출력하는 것입니다. 그리고 설명을 위해서 첫번째 쿼리는 $N-1$번 호출하여 모든 쿼리가 완료된 이후 모든 집합이 하나의 집합으로 합쳐진다고 합시다.

Union Find가 바로 떠오를 수도 있겠지만 단순하게 집합의 크기를 더하는 것만으로는 중복되는 수를 처리하지 못합니다. 그러므로 문제를 풀기 위해 집합마다 set을 하나씩 관리할 것입니다. 우리가 Union Find를 쓸때 주로 사용하는 Path Compression을 이용해 코드를 짜보자면 :

```cpp
int find(int a){
	if(a == p[a]) return a;
	return p[a] = find(p[a]);
}

void merge(int a, int b){
	a = find(a), b = find(b);
	for(auto x : s[b]) s[a].insert(x);
	p[b] = a;
}
```

이 되는데 과연 이게 효율적인 풀이일까요?  최악의 경우를 산정해 봅시다. 아래 그림과 같은 순서대로 집합을 합친다고 한다면

![](https://user-images.githubusercontent.com/51346964/118853479-c7574c80-b90e-11eb-9b13-7bd4c2190198.PNG)

코드에서 insert를 $O(N^2)$번 호출하게 되므로 1번 쿼리의 시간복잡도는 총 $O(N^2 \log N)$이 됩니다. 당연히 이는 마음에 들지 않는 시간복잡도이고 개선할 방법이 필요해보입니다. 그림을 본다면 알겠지만 큰 집합의 원소가 이동하게 되고, 이보단 작은 집합의 원소를 큰 집합에 추가하는 것이 더 효율적으로 보입니다. 

```cpp
void merge(int a, int b){
	a = find(a), b = find(b);
	if(s[a].size() < s[b].size()) swap(a, b);
	for(auto x : s[b]) s[a].insert(x);
	p[b] = a;
}
```

if문으로 추가한 이 규칙이 이번 글의 주제로, 이것을 지키는 것만으로 어떠한 순서대로 집합을 합치더라도 $O(N \log^2 N)$이 보장됩니다.

증명은 어렵지 않습니다. 각 원소가 이동하는 최대 횟수를 살펴보면 이 시간복잡도가 보장되는 이유를 알 수 있습니다. 원소가 속한 집합의 크기가 $X$라고 했을 때 이동하는 경우는 $X$보다 큰 집합과 합칠 때이므로 원소가 이동하는 Union을 할 때마다 $X$는 $2$배 이상 커지게 됩니다. 그러므로 원소가 $Y$번을 이동했을 때 집합의 크기는 최소 $2^Y$가 될 것이므로 집합의 개수의 상한이 $N$일 때 $O(\log N)$번을 초과해서 이동할 수 없습니다. 최악의 경우에 $N$개의 모든 원소가 $O( \log N)$번 이동한다고 해도 이동하는 횟수는 모두 합해서 $O(N \log N)$번이고 set의 insert 함수의 복잡도를 곱한다면 최종 시간복잡도 $O(N \log^2 N)$을 가지게 됩니다.

결국 Small To Large의 로직 자체는 $O(N \log N)$의 시간복잡도를 가지고 $O(1)$에 작동하는 다른 연산을 한다고 하면 그대로의 시간복잡도를 가지게 될 것입니다.




단순한 방법인 만큼 응용의 폭도 넓어 다양한 문제에 사용됩니다. 이번 글을 통해 그러한 예를 몇 가지 앎으로써 Small To Large Merging에 대해 좀 더 폭넓은 이해를 얻어가신다면 좋을 것 같습니다.



# Union Find

Union Find에서 집합에 포함된 원소의 종류, 개수, 번호 등을 같이 관리해야 할 경우 쓰입니다.



## [BOJ 17469. 트리의 색깔과 쿼리](https://www.acmicpc.net/problem/17469)

우리는 매우 많은 상황에서, 제거보다는 추가가 쉽다는 것을 알고 있습니다. 이 문제에서도 마찬가지인데, 쿼리의 순서를 반대로 하면 $1$번 쿼리를 간선 제거 대신 간선 추가로 바꾸는 게 가능합니다. 정점의 색을 원소로 둔 set을 이용해 중복 없이 집합을 관리할 수 있으며, $2$번 쿼리는 해당 정점이 속한 집합의 크기를 출력하기만 하면 됩니다.



쿼리를 반대로 뒤집어야 한다는 것만 뺀다면 본문에서 설명했던 문제와 똑같습니다. 마찬가지로 시간복잡도 $O(N \log^2 N)$으로 집합을 관리할 수 있습니다.



```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q, p[100100], c[100100], parent[100100];
set<int> s[100100];
stack<pair<int, int>> query;
stack<int> answer;

int find(int a){
    if(a == parent[a]) return a;
    return parent[a] = find(parent[a]);
}

void merge(int a, int b){
    a = find(a), b = find(b);
    if(s[a].size() < s[b].size()) swap(a, b); // small to large
    for(int x : s[b]) s[a].insert(x);
    parent[b] = a;
}

int main(){
    scanf("%d %d",&n,&q);
    for(int i=2;i<=n;i++) scanf("%d",p+i);
    for(int i=1;i<=n;i++) scanf("%d",c+i);
    for(int i=1;i<q+n;i++){
        int q, a; scanf("%d %d",&q,&a);
        query.push({q, a});
    }

    for(int i=1;i<=n;i++) parent[i] = i, s[i].insert(c[i]);
    while(!query.empty()){
        auto [q, a] = query.top(); query.pop();
        if(q == 1) merge(a, p[a]);
        else answer.push(s[find(a)].size());
    }

    while(!answer.empty()) printf("%d\n",answer.top()), answer.pop();
}
```





## [BOJ 21132. Best Solution Unknown](https://www.acmicpc.net/problem/21132)

S[L,R]을 구간 [L,R]에서 우승할 수 있는 사람들의 집합이라고 합시다.

처음에는 아무 구간도 없고, $a_i$의 오름차순으로 순회하면서 각 $i$마다 다음 과정을 진행합니다.

1. 새로운 구간 $[i,i]$를 삽입한다. 처음에는 $S[i,i]=\{i\}$이다.
2. $i-1$을 포함하는 구간 $[le,i-1]$이 존재한다면 $[le,i-1]$과 $i$가 포함된 구간($[i, i]$)을 합친다.
3. $i+1$을 포함하는 구간 $[i+1,ri]$가 존재한다면 $i$가 포함된 구간($[i,i]$ 혹은 $[le,i]$)과  $[i+1,ri]$를 합친다.

두 구간 $[L,M]$과 $[M+1,R]$을 합쳐서 $[L,R]$로 만들 때 $S[]$의 변화는 다음과 같습니다.

1. 각 $i \in S[L,M]$에 대해 $a_i+(M-L) \geq a_{M+1}$을 만족하는 경우에만 $S[L,R]$에 $i$를 넣는다.
2. 각 $i \in S[M+1,R]$에 대해 $a_i+(R-M) \geq a_{M}$을 만족하는 경우에만 $S[L,R]$에 $i$를 넣는다.



모든 구간이 합쳐질 때까지 반복하면 우승할 수 있는 사람만 집합에 남게 되고 이는 문제의 정답입니다.

우선순위 큐를 이용해 집합의 원소를 관리한다면 중간에 집합에서 지워야 할 원소를 쉽게 제거할 수 있습니다. 우선순위 큐의 push와 pop은 set의 insert와 같은 $O(\log N)$의 시간복잡도를 가지므로 최종 시간복잡도는 $O(N \log^2 N)$입니다.



비슷한 풀이로 이 [문제](https://www.acmicpc.net/problem/10760)를 풀 수 있습니다.



# 트리

현재 노드의 서브트리를 관리하기 위해서 현재 노드와 자식 노드의 서브트리의 정보를 합쳐야 할때 사용합니다.



## [2020 Sogang Programming Contest (Champion) G. Confuzzle](https://www.acmicpc.net/problem/20297)

각 정점에서 "서브트리에 저장된 수"와 "수가 저장된 정점 중 가장 낮은 깊이"를 짝지어 원소로 관리한다고 합시다. 서브트리의 값을 채우기 위해서는 자식 노드의 정보가 필요하므로 DFS를 이용해 리프 노드부터의 값을 채울 수 있습니다.

부모 정점과 자식 정점의 두 집합을 합치는데 "subtree에 저장된 수"가 두 집합에 중복되어 있다면 같은 수를 가진 두 정점이 subtree에 존재한다는 것이므로 두 정점의 거리를 계산하여 답을 갱신해주면 됩니다.

이렇게 원소를 관리한다면 해당 정점을 lca로 가지는 경로 중에 정답의 후보는 모두 찾으므로 문제의 정답 또한 구할 수 있습니다.



결국 트리의 간선의 개수만큼의 Union이 일어나므로 풀이의 시간복잡도는 $O(N \log^2 N)$입니다.



```cpp
#include<bits/stdc++.h>
using namespace std;
using ll = long long;

int n, c[100100], id[100100], lev[100100], ans = 1e9;
vector<int> v[100100];
map<int, int> mp[100100];

void dfs(int u, int p){
	id[u] = u; // map의 index 설정
    lev[u] = lev[p] + 1; // 루트부터 떨어진 거리
    mp[id[u]][c[u]] = lev[u]; // 현재 정점 또한 subtree에 속한 정점이다.

    for(auto x : v[u]) if(x != p){
        dfs(x, u);
        if(mp[id[x]].size() > mp[id[u]].size()) swap(id[x], id[u]); // small to large

        for(auto [c, h] : mp[id[x]]){
        	if(mp[id[u]].count(c)){ // 수가 중복되어 있다면
                ans = min(ans, mp[id[u]][c] + h - lev[u] * 2); // 두 점 사이의 거리를 계산하여 답을 갱신한다.
                mp[id[u]][c] = min(mp[id[u]][c], h); // 더 낮은 깊이로 값을 갱신해준다.
            }else{
                mp[id[u]][c] = h;
            }
        }
    }
}

int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++) scanf("%d",c+i);
    for(int i=1;i<n;i++){
        int a, b; scanf("%d %d",&a,&b);
        v[a].push_back(b), v[b].push_back(a);
    }
    dfs(1, 0);
    printf("%d",ans);
}
```

집합을 합치면서 필요한 정보가 들어있는 map의 index가 달라질 수 있기 때문에 원하는 집합을 바로 접근할 수 있도록 id 배열을 잘 관리해야 합니다.



비슷한 풀이로 이 [문제](https://www.acmicpc.net/problem/5820)를 풀 수 있습니다.





# 분할 정복

큰 집합에 작은 집합의 원소를 추가하면서 집합을 합치는 것이 시간복잡도가 보장되는 것처럼 역순으로 하나의 집합에서 작은 집합의 원소를 떼어내면서 두 집합으로 분리하는 것 또한 시간복잡도가 보장됩니다. set의 insert 대신 delete를 한다고 하면 마찬가지로 $O(N\log N)$번을 호출하게 될 것입니다.



## [KOI 2006 초등부 3번. 마법색종이](https://www.acmicpc.net/problem/2574)

점에 의해서 잘라진 두 색종이는 서로 영향을 주지 않습니다. 즉, 두 색종이에 속한 점을 분리해서 문제를 풀 수 있습니다.



색종이를 하나의 집합으로 보고 점을 원소로 관리합니다.

문제에서 주어진 규칙을 따라 색종이를 자르기 위해 필요한 정보는 3가지입니다.

1. 다음에 자를 기준이 되는 점의 번호
2. 1번 규칙에 의해 분리될 두 색종이에 각각 속하게 되는 점의 개수를 알기 위한 y좌표
3. 2번 규칙에 의해 분리될 두 색종이에 각각 속하게 되는 점의 개수를 알기 위한 x좌표

그러므로 우리는 문제를 풀기 위해 "번호", "x 좌표", "y 좌표"를 각각 자료구조로 관리할 것입니다. "번호"는 set으로 가장 작은 값을 빠르게 찾을 수 있고 나머지 두 좌표는 pbds로 관리해서 두 색종이에 포함될 점의 개수를 찾을 수 있습니다.

합칠 때 작은 집합의 원소를 하나씩 이동시켰듯이 분리할 때에도 점이 더 적게 포함되는 구간의 점을 집합에서 하나씩 빼주면서 새로운 집합에 넣어주면 됩니다. 점이 많이 포함되는 구간은 필요한 정보가 남아 있는 집합을 계속 사용하고 아닌 부분은 새롭게 만든 집합을 이용해 반복하면 문제를 해결할 수 있습니다.



풀이의 시간복잡도는 합칠 때와 같은 $O(N \log^2 N)$입니다.



```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

#define y1 JOODDAE

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<class key, class value, class cmp = std::less<key>>
using ordered_map = tree<key, value, cmp, rb_tree_tag, tree_order_statistics_node_update>;

int n, k, X[30030], Y[30030], x1[30030], x2[30030], y1[30030], y2[30030], col[300300];

set<int> s[30030];
ordered_map<int, int> x[30030], y[30030];

void Insert(int id, int k){
	s[id].insert(k);
	x[id][X[k]] = k;
	y[id][Y[k]] = k;
}

void Erase(int id, int k){
	s[id].erase(k);
	x[id].erase(X[k]);
	y[id].erase(Y[k]);
}

void solve(int id){ // id = 집합의 번호
	if(s[id].empty()) return; // 조각에 점이 없으면 더 이상 나누지 않는다.

	int u = *s[id].begin(); // 이번에 사용할 점을 찾는다.
	s[id].erase(u), x[id].erase(X[u]), y[id].erase(Y[u]); // 사용할 점은 집합에서 지워준다.

	++k; // 새로운 집합을 만든다.
	x1[k] = x1[id], x2[k] = x2[id], y1[k] = y1[id], y2[k] = y2[id];

	queue<int> q; // 새로운 집합으로 옮겨질 점을 저장할 자료구조

	if(col[id]){ // 검은색 조각이면 x좌표를 기준으로 나눈다.
		int cnt = x[id].order_of_key(X[u]);

		if(cnt <= s[id].size() / 2){ // 왼쪽 조각에 속한 점의 개수가 더 적으면
			x2[k] = x1[id] = X[u];
            // 왼쪽 조각에 속한 점을 나이브하게 찾는다.
			for(auto it=x[id].begin();it!=x[id].end() && it->first < X[u];it++) q.push(it->second);
		}else{ // 오른쪽 조각에 속한 점의 개수가 더 적으면
			x1[k] = x2[id] = X[u];
            // 오른쪽 조각에 속한 점을 나이브하게 찾는다.
			for(auto it=x[id].rbegin();it!=x[id].rend() && it->first > X[u];it++) q.push(it->second);
		}
	}else{ // 흰색 조각이면 y좌표를 기준으로 나눈다.
		int cnt = y[id].order_of_key(Y[u]);

		if(cnt <= s[id].size() / 2){ // 아래쪽 조각에 속한 점의 개수가 더 적으면
			y2[k] = y1[id] = Y[u];
            // 아래쪽 조각에 속한 점을 나이브하게 찾는다.
			for(auto it=y[id].begin();it!=y[id].end() && it->first < Y[u];it++) q.push(it->second);
		}else{ // 위쪽 조각에 속한 점의 개수가 더 적으면
			y1[k] = y2[id] = Y[u];
            // 위쪽 조각에 속한 점을 나이브하게 찾는다.
			for(auto it=y[id].rbegin();it!=y[id].rend() && it->first > Y[u];it++) q.push(it->second);
		}
	}

	while(!q.empty()){
		Insert(k, q.front()), Erase(id, q.front()); // 나이브하게 구한 옮겨질 점들을 집합에서 뺴주고, 삽입한다.
		q.pop();
	}

	col[k] = col[id] = !col[id]; // 색은 반전시킨다.
	solve(k), solve(id);
}

int main(){
	scanf("%d %d",x2,y2);
	scanf("%d",&n);
	for(int i=1;i<=n;i++) scanf("%d %d",X+i,Y+i);

	for(int i=1;i<=n;i++) s[0].insert(i), x[0][X[i]] = y[0][Y[i]] = i;
	solve(0);

	int mx = 0, mn = 1e9;
	for(int i=0;i<=n;i++){
		mx = max(mx, (x2[i] - x1[i]) * (y2[i] - y1[i]));
		mn = min(mn, (x2[i] - x1[i]) * (y2[i] - y1[i]));
	}
	printf("%d %d",mx,mn);
}
```





## [CERC 2012 D. Non-boring sequences](https://www.acmicpc.net/problem/3408)

구간에서 unique한 수가 있다면 그 수를 포함하는 모든 부분 수열은 non-boring하기 때문에 unique한 수를 기준으로 두 구간으로 나눌 수 있습니다. 그렇게 두 구간을 나누는 것을 구간의 길이가 1일때까지 재귀적으로 반복할 수 있으면 non-boring한 수열입니다.



구간 내에서 나이브하게 unique한 수를 찾는다면 최악의 경우 $O(N^2)$으로 시간초과가 나게 됩니다. 그러므로 글의 주제인 small to large를 활용할 수 있는 방법으로 탐색할 것입니다.

$[l, r]$ 구간에서 $l$, $r$, $l+1$, $r-1$, $l+2$, ... 순서대로 양방향으로 탐색해서 처음으로 찾는 unique한 수의 위치를 $m$이라 하면 $[l,m-1]$과 $[m+1,r]$ 두 구간으로 분리됩니다. 이때, 둘 중 짧은 구간의 길이를 $small$이라 하면 아까 양방향으로 탐색하는 데에 드는 비용이 $small \times 2$ 이하가 됩니다. 따라서 큰 집합에서 작은 집합을 분리해내는 거랑 마찬가지로 $O(N \log N)$번의 탐색만으로 정답의 여부를 확인할 수 있습니다.



구간 내에서 수가 unique한지는 $O(N \log N)$의 전처리로 연산마다 $O(1)$에 구할 수 있으므로 총 시간복잡도는 $O(N \log N)$입니다.



```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int tc, n, a[200200], L[200200], R[200200];

bool unique(int l, int r, int x){ // [l, r] 구간내에 x번째 수가 unique하다면 true를 반환합니다.
	return L[x] < l && r < R[x];
}

bool solve(int l, int r){
	if(l >= r) return false; // 구간의 길이가 1이하면 non-boring한 구간입니다.

	int L = l, R = r;
	while(L <= R){ // 양방향 탐색
		if(unique(l, r, L)) return solve(l, L-1) || solve(L+1, r); // unique한 수를 찾았다면 두 구간으로 나눕니다.
		if(unique(l, r, R)) return solve(l, R-1) || solve(R+1, r);
		L++, R--;
	}

	return true; // 구간 내에 unique한 수가 없으므로 boring한 수열입니다.
}

int main(){
	cin.tie(0)->sync_with_stdio(0);
	cin >> tc;
	while(tc--){
		cin >> n;
		for(int i=1;i<=n;i++) cin >> a[i];

        // a[i]와 같은 값을 가지면서 왼쪽에서 가장 가까운 수의 위치를 L[i]에 저장하고, 오른쪽에서 가장 가까운 수의 위치를 R[i]에 저장합니다.
		map<int, int> mp;
		for(int i=1;i<=n;i++){
			int x = mp[a[i]];
			R[x] = i, L[i] = x;
			mp[a[i]] = i;
		}
		for(auto [x, y] : mp) R[y] = n+1;

		cout << (solve(1, n) ? "boring" : "non-boring") << "\n";
	}
}
```
