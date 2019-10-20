---
layout: post
title:  "2013-2014 Petrozavodsk Winter Training Camp K 풀이"
date:   2019-10-20 23:53
author: kajebiii
tags: [problemsolving, contest, codeforces]
---

## 문제 요약
[문제 링크](http://codeforces.com/gym/101237/problem/K)
[문제 셋 링크](http://codeforces.com/gym/101237)

$1$부터 $N$까지 번호가 매겨져 있는 트리가 있다.

노드 $A$와 $B$의 *단순 경로*는 두 노드 사이의 유일한 경로를 의미한다.

단순 경로 $P$의 *이웃*은 정확히 한 정점만 경로위에 존재하는 간선들의 집합을 의미한다.

모든 간선은 blocked 또는 unblocked 상태이다. 초기 상태는 모두 blocked 상태이다.

다음 두 쿼리를 대응하는 프로그램을 작성하라.

1. $a$와 $b$의 단순 경로의 blocked 상태의 간선의 개수를 구하라.

2. $a$와 $b$의 단순 경로의 간선들을 unblocked 상태로 바꾸고, $a$와 $b$의 단순 경로의 이웃을 blocked 상태로 바꾼다.

$ 1 \le N \le 2 \times 10^5, 1 \le Q \le 3 \times 10^5$

## 풀이
### 1. $O(QN)$
Naive 한 방법.

각각의 쿼리에 대해서 $O(N)$ 으로 처리한다.

### 2. $O(Q \log^{2} N)$
1번 쿼리에 비해 2번 쿼리를 처리하는 방법이 난감해보인다. 처리하기 까다로운 2번 쿼리를 적절히 변형시켜보자.

현재는 간선에 상태 (blocked 또는 unblocked)가 존재한다. 문제에서는 정의하지 않았지만, 정점에도 상태 ($0$부터 $Q$까지의 수)로 정의하자.
간선의 상태는 간선의 두 정점의 상태로 결정하겠다. 만약 두 정점의 숫자가 같다면 unblocked 상태로, 다르다면 blocked 상태로 정의하자.

즉, 기존의 문제에서 제시한 간선의 상태를 아이디어를 이용하여 정점의 상태로 변환하였다.
초기 정점상태를 모두 $0$으로 생각하면 된다. 그러면 모든 정점의 상태가 같으므로, 모든 간선은 blocked 상태가 될 것이다.

이제 1번 쿼리와 2번 쿼리를 다시 생각해보자.

1. $a$와 $b$의 단순 경로의 양 정점의 상태가 같은 것 간선의 개수를 구하라.

2. $a$와 $b$의 단순 경로의 위의 정점의 상태를 기존에 없던 상태 $q$로 변환한다. ($q$는 현재 처리하고 있는 쿼리의 번호)

다시 생각해본 결과, 2번 쿼리가 단순해졌다. 2번 쿼리가 위와 같이 처리해도 되는 이유를 예시로 설명해보자.
만약 5번째로 입력된 쿼리를 처리하고 있을 경우 정점의 상태는 $0$부터 $4$까지만 존재할 것이다.
이제 $a$와 $b$의 단순 경로 위의 정점들의 상태를 $5$로 바꾸게 되면, 새롭게 등장한 상태로 인해서 이웃(간선)들의 상태는 blocked 가 될 것이다.
또한, 단순 경로 위의 정점들의 상태가 모두 같아졌으므로 단순 경로 위의 간선들의 상태는 모두 unblocked 가 된다.

----------------------------------------------------------------

문제 변환은 잘되었고, 바로 트리에서 문제를 풀기에는 힘들어보이니 $1-2-3-\cdots-N$ 꼴의 일직선인 경우에 대해서 문제를 풀어보자.

Lazy segment tree 를 이용하여 1번 쿼리와 2번 쿼리를 해결할 것이다.

각 segment tree 에서 각각의 노드가 들고 있는 정보는 양 끝 정점의 상태(아래 코드에서 lv, rv)와 노드에 속한 간선들 중 blocked 상태의 개수 (아래 코드에서 sum)이다.

그러면 연속한 노드를 merge 하는 과정은 아래 코드의 merge() 함수와 같다. lv을 왼쪽의 lv로 정해주고, rv를 오른쪽의 rv로 정해주고, 
sum은 왼쪽의 sum와 오른쪽의 sum을 더한 뒤, 왼쪽의 rv와 오른쪽의 lv가 다른 경우 $1$을 추가적으로 더해주면 된다. 이를 이용하여 1번 쿼리를 $O(\log N)$으로 해결할 수 있다.

2번 쿼리는 lazy propagation 을 이용하여 해결할 수 있다. 아래 코드에서 lazy 변수가 그 역할을 한다.

따라서 일직선인 경우 문제를 해결 할 수 있다.

----------------------------------------------------------------

일직선인 경우 문제에서 해결 가능한 경우, heavy light decomposition (HLD) 을 이용하여 트리에서 문제를 해결 할 수 있다.

간단히 HLD 을 설명하면, 트리의 간선들을 적절한 일자 묶음으로 나누어서, 임의의 두 정점 사이의 경로는 $O(\log N)$개의 묶음으로 표현하는 기법이다.

우리는 일직선인 경우 문제를 해결하는 방법을 알고있으므로, HLD 을 이용하면 $O(\log N)$ 을 추가하여 트리에서도 해결할 수 있다.

따라서 시간복잡도는 $O(Q \log ^2 N)$가 된다.


## 코드
```cpp
#include <bits/stdc++.h>
 
using namespace std;
 
#define SZ(v) ((int)(v).size())
#define ALL(v) (v).begin(),(v).end()
#define one first
#define two second
typedef long long ll;
typedef pair<int, int> pi;
const int INF = 0x3f2f1f0f;
const ll LINF = 1ll * INF * INF;
 
const int MAX_N = 2e5 + 100, LOG_N = 21;
const int MAX_Q = 3e5 + 100;
 
struct NODE{
	int l, r;
	int lv, rv;
	NODE *lp, *rp;
	int lazy, sum;
	void merge() {
		sum = 0;
		if(lp != NULL) {
			sum += lp -> sum;
			sum += rp -> sum;
			if(lp -> rv != rp -> lv) sum++;
			lv = lp -> lv;
			rv = rp -> rv;
		}
	}
	NODE(int l_, int r_) {
		l = l_;
		r = r_;
		lp = rp = NULL;
		lazy = -1;
		sum = 0;
 
		if(l == r) {
			lv = rv = l;
			return;
		}else{
			int m = (l+r) >> 1;
			lp = new NODE(l, m);
			rp = new NODE(m+1, r);
			merge();
		}
	}
	void pro() {
		if(lazy != -1) {
			lp -> update(lazy);
			rp -> update(lazy);
			lazy = -1;
		}
	}
	void update(int val) {
		lazy = val;
		sum = 0;
		lv = rv = val;
	}
	void paint(int x, int y, int val) {
		if(r < x || y < l) return;
		if(x <= l && r <= y) {
			update(val);
			return;
		}
		pro();
		lp -> paint(x, y, val);
		rp -> paint(x, y, val);
		merge();
	}
	using ti = tuple<int, int, int>;
	ti getSum(int x, int y) {
		if(r < x || y < l) return ti(0, -1, -1);
		if(x <= l && r <= y) {
			return ti(sum, lv, rv);
		}
		pro();
		int lsum, llv, lrv; tie(lsum, llv, lrv) = lp -> getSum(x, y);
		int rsum, rlv, rrv; tie(rsum, rlv, rrv) = rp -> getSum(x, y);
		return ti(lsum+rsum+(lrv != -1 && rlv != -1 && lrv != rlv), llv != -1 ? llv : rlv, rrv != -1 ? rrv : lrv);
	}
	int getVal(int x) {
		if(l == r) return lv;
		pro();
		int m = (l+r) >> 1;
		if(x <= m) return lp -> getVal(x);
		else return rp -> getVal(x);
	}
};
int N, Q;
vector<int> Ed[MAX_N];
int S[MAX_N], Dep[MAX_N], Par[LOG_N][MAX_N];
int preDFS(int v, int p) {
	S[v] = 1, Dep[v] = Dep[p] + 1; Par[0][v] = p;
	for(int w : Ed[v]) if(w != p) S[v] += preDFS(w, v);
	return S[v];
}
int En[MAX_N], EN, Head[MAX_N];
void setHLD(int v, int p, int head) {
	if(p != 0) En[v] = ++EN, Head[v] = head;
	int ix = -1; for(int w : Ed[v]) if(w != p && (ix == -1 || S[ix] < S[w])) ix = w;
	if(ix == -1) return;
	setHLD(ix, v, head);
	for(int w : Ed[v]) if(w != p && w != ix) setHLD(w, v, w);
}
int getLCA(int a, int b) {
	if(Dep[a] > Dep[b]) swap(a, b);
	for(int p=0; p<LOG_N; p++) if((Dep[b]-Dep[a])&(1<<p)) b = Par[p][b];
	if(a == b) return a;
	for(int p=LOG_N-1; p>=0; p--) if(Par[p][a] != Par[p][b]) 
		a = Par[p][a], b = Par[p][b];
	return Par[0][a];
}
NODE *root;
int get0(int v, int last) {
	int head = Head[v];
	if(Dep[last] < Dep[head]) {
		int sum; tie(sum, ignore, ignore) = root -> getSum(En[head], En[v]);
		int lv = root -> getVal(En[head]);
		int rv = root -> getVal(En[Par[0][head]]);
		return sum + get0(Par[0][head], last) + (lv != rv);
	}else{
		int sum; tie(sum, ignore, ignore) = root -> getSum(En[last], En[v]);
		return sum;
	}
}
void do1(int v, int last, int val) {
	int head = Head[v];
	if(Dep[last] < Dep[head]) {
		root -> paint(En[head], En[v], val);
		return do1(Par[0][head], last, val);
	}else{
		root -> paint(En[last], En[v], val);
		return;
	}
}
int main() {
	scanf("%d", &N);
	for(int i=1; i<N; i++) {
		int x, y; scanf("%d%d", &x, &y);
		Ed[x].push_back(y);
		Ed[y].push_back(x);
	}
	preDFS(1, 0);
	for(int p=1; p<LOG_N; p++) for(int i=1; i<=N; i++) Par[p][i] = Par[p-1][Par[p-1][i]];
	setHLD(1, 0, 1);
	root = new NODE(0, N);
	int cnt = N;
 
	scanf("%d", &Q);
 
	for(int q=0; q<Q; q++) {
		int t, x, y; scanf("%d%d%d", &t, &x, &y);
		int lca = getLCA(x, y);
		if(t == 0) {
			printf("%d\n", get0(x, lca) + get0(y, lca));
		}else if(t == 1) {
			cnt++;
			do1(x, lca, cnt);
			do1(y, lca, cnt);
		}else assert(false);
	}
	return 0;
}
```