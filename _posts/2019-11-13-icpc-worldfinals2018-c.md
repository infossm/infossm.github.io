---
layout: post
title:  "2018 ICPC world Finals C. Conquer the world와 Tree DP optimization"
date:   2019-11-17 23:30:00
author: ainta
tags: [algorithm, dynamic-programming, ICPC, data-structure, optimization]



---


2018년 World Finals에서 어느 팀도 풀지 못했던 문제인 Conquer the world(https://www.acmicpc.net/problem/15691) 문제에 대한 풀이와 사용된 아이디어에 대해 간단히 소개한다.

# 문제

문제 자체는 굉장히 간단하다. edge마다 이동할 때 드는 cost가 있는 트리가 있고, vertex $$i$$에 현재 $$X_i$$명이 있으며 최종 상태에는 적어도 $$Y_i$$명이 있어야 할 때, 사용해야 하는 cost를 minimize하는 문제이다.

# Heavy-Light Decomposition

이미 상당히 유명해진 트릭인 heavy-light decomposition에 대해 먼저 간략히 설명하고 넘어갈 것이다. rooted tree에서 heavy edge란, vertex $$v$$의 자식들 중 가장 subtree의 크기가 큰(vertex 개수가 많은) 자식을 $$u$$라 할 때, $$(u, v)$$를 heavy edge라고 하고, $$u$$를 $$v$$의 heavy child라고 한다. $$v$$와 다른 child를 잇는 edge는 light edge가 된다. 이렇게 각 edge에 heavy, light 라벨을 붙이면 어떤 vertex $$v$$에 대해서도 $$v$$와 root를 잇는 path에 light edge는 $$log N +1$$개 이하만 존재할 수 있다. 이유는 간단한데, $$v$$가 $$u$$의 parent이고 $$(u, v)$$가 light edge이면 $$v$$를 root로 하는 subtree가 $$u$$를 root로 하는 subtree보다 2배 이상 크기 때문이다.

Heavy light decomposition으로 할 수 있는 것 중 대표적인 것은 tree의 모든 path를 heavy chain $$O(log N)$$개로 나눌 수 있으므로 segment tree를 이용하여 여러 값을 계산하거나 업데이트할 수 있다는 것이다. 그러나, 이 외에도 heavy light decomposition을 응용할 수 있는 아이디어는 많다. 특히, 직접 heavy-light decomposition을 구현하지 않고 아이디어만 사용하는 경우가 매우 빈번하다.

예를 들어, rooted tree에서 모든 vertex $$v$$에 대해 $$v$$의 subtree들의 집합 $$subtree(v) $$의 각 노드에 쓰인 수들의 집합을 계산해야 한다고 하자. 이를 계산할 때, $$v$$의 모든 자식들이 들고 있는 집합을 다 꺼내서  정렬한 후 합치면 시간이 오래 걸릴 수 있다. 그러나, $$v$$의 heavy child가 $$u$$라고 할 때, $$u$$가 아닌 다른 자식에 있는 집합의 원소들을 다 꺼내어 $$u$$의 집합에 넣는다면, 어느 vertex도 root까지 light edge가 $$O(log N)$$개이기 때문에 $$O(log N)$$번만 삽입되게 되고, 집합을 구현한 자료구조가 STL::set인 경우 한 번 삽입하는 데에 $$O(log N)$$시간이 소모되므로 $$O(N log^2 N)$$ 시간에 모든 vertex $$v$$에 대해 집합을 계산할 수 있게 된다.

이러한 아이디어는 때때로 $$O(N^2)$$시간이 소모되는 tree DP 문제를 $$O(Nlog^2N)$$이나 $$O(NlogN)$$ 시간에 해결할 수 있도록 도움을 준다. 보통 각 vertex마다 계산해야 하는 DP table의 크기가 (subtree의 vertex 개수)나 (subtree의 높이)인 경우에, DP 값이 convex한 경우 이를 사용하여 해결하는 풀이가 존재할 수 있다. (모든 경우 풀리는 것은 아니다)

# Conquer the world

이제 다시 conquer the world 문제로 돌아가 보자. 다음과 같은 naive한 DP를 생각할 수 있다.

$$D[u][i]$$: $$u$$의 subtree에서 초기 상태보다 최종 상태에 사람 수가 $$i$$명 줄어들었을 때, 조건을 만족하도록 하면서($$Y_i$$ 이상) $$u$$의 subtree 내에서 이동하는 cost의 최솟값. 즉, $$i$$가 0 이상인 경우 subtree 내에서 사람들을 이동시킨 후 $$u$$에서 $$parent[u]$$로 $$i$$명이 이동해야 하고, $$i$$가 0 이하인 경우 처음에 $$parent[u]$$에서 $$u$$로 $$-i$$명이 온 후에 subtree 내에서 사람들을 이동시키는 cost이다.

그러면 $$D[root][0]$$이 구하고자 하는 최종 답이 될 것이다.

$$F(u, i) = D[u][i] + cost(u, parent[u]) * abs(i)$$ 로 놓으면 $$F(u,i)$$는 $$u$$와 $$parent[u]$$를 잇는 edge까지 고려한 cost가 된다. 그러면 이제 $$D2[u][i]$$를 $$u$$의 subtree에서 정점 $$u$$는 제외했을 때 계산한 dp값이라고 하면 $$F$$값을 이용해 $$D2$$의 값은 쉽게 계산할 수 있다. $$u$$의 자식 $$c_1,.. ,c_k$$에 대해, $$D2[u][i] = Min_{j_1 + j_2 + .. + j_k = i}(F(c_1, j_1) +F(c_2,j_2)+...+F(c_k,j_k))$$이다. 그 후 정점 $$u$$를 고려해주면 $$D[u][i + X_u- Y_v] = D2[u][i]$$ 이므로 $$u$$의 dp값을 $$u$$의 자식들의 dp값으로부터 계산할 수 있다.

DP를 이제 다음과 같이 조금 변형해서 정의해 보자.

$$D[u][i]$$: $$u$$의 subtree에서 초기 상태보다 최종 상태에 사람 수가 $$i$$명 **이상** 줄어들었을 때, 조건을 만족하도록 하면서($$Y_i$$ 이상) $$u$$의 subtree 내에서 이동하는 cost의 최솟값

만약 $$u$$가 leaf라면, $$D[u][i]$$는 $$i > X_u - Y_u$$이면 무한일 것이고, $$i \le X_u - Y_u$$이면 0일 것이다. 정점 $$u$$에 대해, $$D[u][i]$$값이 무한이 아닌 최대 $$i$$를 $$M(u)$$라 하자.

$$F(u,i) = Min_{i \le j}(D[u][j] +cost(u, parent[u]) * abs(j))$$ 로 놓으면, $$u$$에서 $$parent[u]$$로 i명 이상 이동할 때 최소 cost가 계산이 된다. 이는 우리가 $$D[u][i]$$를 이산적으로 보지 않고 $$i$$에 대한 함수 $$d(i)$$로 생각을 한다면 $$f(i) = d(i) + cost * \lvert i \rvert$$로 놓은 후 에, $$F(i) = Min_{i \le j}(f(j))$$로 정의한 것과 같다. 만약 $$d(i)$$가 $$ i \le M(u)$$에서 convex라면, 즉 $$d(i) - d(i-1) \le d(i+1) - d(i)$$가 성립한다면 $$f(i)$$ 역시 convex하고, $$F(i)$$도 convex한 형태가 됨을 알 수 있다. $$u$$의 자식 $$c_1, ..., c_k$$에 대해 $$F(c_1), ..., F(c_2)$$로부터 $$D_2[u]$$를 계산하는 것은 convex function의 민코프스키 합을 계산하는 것과 같은데, 이는 단순히 모든 기울기를 정렬한 후 그대로 나열하는 것으로 충분하다. 

따라서, 모든 vertex $$u$$에 대해 함수 $$d(i)$$를 계산하기 위해서는 다음과 같은 연산을 수행할 수 있어야 한다.

1. $$d_1(i), d_2(i), .., d_k(i)$$에 대한 민코프스키 합 계산
2. $$f(i) = d(i) + c * \lvert i \rvert$$ 계산 
3. $$F(i) = Min_{i \le j}(f(j))$$ 계산
4. $$X_u - Y_u$$만큼 평행이동

1번은 $$d(i)$$의 정의역에 대해 $$d(x+1)-d(x)$$의 값을 모두 알고 있다면, priority queue나 multiset을 이용해 기울기를 정렬된 상태로 합칠 수 있다.

2번은 $$d(x+1)-d(x)$$의 값이 $$x < 0$$에서는 $$c$$만큼 작아지고, $$x \ge 0$$에서는 $$c$$만큼 커지므로 구간에 어떤 수를 더하는 쿼리를 수행할 수 있다면 해결할 수 있다.

3번의 경우, $$f(i)$$가 convex이면 $$F(i)$$는 $$f(x)-f(x-1) < 0$$이 성립하는 가장 큰 $$x$$를 $$x_0$$이라고 하면 $$x \le x_0$$인 모든 $$x$$에 대해 $$F(x) = f(x_0)$$이고, $$x \ge x_0$$인 모든 $$x$$에 대해 $$F(x) = f(x)$$가 된다. 따라서, $$f(i) - f(i-1)$$를 들고 있는 multiset이나 priority queue가 있다면 가장 작은 원소를 보면서 0보다 작다면 제거해주는 식으로 구현 할 수 있다.

4번의 경우, 얼마만큼 평행이동이 되었다는 offset을 그래프에 표시해주면 된다.

다음은 위 연산들을 naive하게 수행하는 느린 코드이다.

```c++
#include<cstdio>
#include<algorithm>
#include<vector>
#include<map>
#define N_ 251000
using namespace std;
int n, Y[N_], X[N_], pL[N_]; // pL[a] : cost(a, parent[a])
vector<int>E[N_], L[N_], Ch[N_];


void DFS(int a, int pp) {
    for (int i = 0; i < E[a].size(); i++) {
        int x = E[a][i];
        if (x == pp)continue;
        DFS(x, a);
        pL[x] = L[a][i];
        Ch[a].push_back(x);
    }
}

int Right[N_], Left[N_]; //Left[a]~Right[a] : dp[a]의 정의역 (Left[a] 이하에서는 dp값이 0, Right[a] 초과에서는 infinity)

long long D[N_]; 
map<int, long long>Map[N_]; // Map[a][i] : dp(a,i+1)-dp(a,i), 기울기

void UDT(int a) { // vertex a의 dp값을 가공
    int d = X[a] - Y[a]; //d만큼 평행이동
    int i;
    map<int, long long>T;
    int c = pL[a];
    Right[a] += d, Left[a] += d;
    for (auto &t : Map[a]) {
        int x = t.first + d;
        int f = t.second;
        if (x >= 0)T[x] = f + c; // 그래프에 y = c|x|를 더함
        else T[x] = f - c;
    }
    if (Left[a] > 0) {
        for (i = 0; i < Left[a]; i++) {
            T[i] = c;
        }
        Left[a] = 0;
    }
    if (Left[a] < 0) {
        long long z = D[a] - 1ll*Left[a] * c;
        while (!T.empty()) { //기울기가 0보다 작은 부분을 지움
            auto it = T.begin();
            if (it->second > 0)break;
            z += it->second;
            Left[a]++;
            T.erase(it);
        }
        D[a] = z;
    }
    Map[a] = T;
}

void Do(int a) {
    if (Ch[a].empty()) {
        Right[a] = Left[a] = 0;
        D[a] = 0;
        return;
    }
    for (auto &x : Ch[a]) {
        Do(x);
    }
    vector<long long>Z;
    int ls = 0, rs = 0;
    D[a] = 0;
    for (auto &x : Ch[a]) {
        UDT(x);
        ls += Left[x];
        rs += Right[x];
        D[a] += D[x];
        for (auto &t : Map[x]) {
            Z.push_back(t.second);
        }
    }
    sort(Z.begin(), Z.end()); //기울기를 정렬하여 민코프스키 합 계산
    for (int i = 0; i < rs - ls; i++) {
        Map[a][ls + i] = Z[i];
    }
    Left[a] = ls, Right[a] = rs;
}

int main() {
    int i, a, b, c;
    scanf("%d", &n);
    for (i = 1; i < n; i++) {
        scanf("%d%d%d", &a, &b, &c);
        E[a].push_back(b);
        E[b].push_back(a);
        L[a].push_back(c);
        L[b].push_back(c);
    }
    for (i = 1; i <= n; i++)scanf("%d%d", &X[i], &Y[i]);
    DFS(1, 0);
    Do(1);
    UDT(1);
    long long s = D[1];
    for (i = Left[1]; i < 0; i++)s += Map[1][i];
    printf("%lld\n", s);
}

```

### Splay Tree



2번 연산이 없다면 priority queue나 multiset만으로 문제를 쉽게 해결할 수 있지만, 구간에 합을 더하는 쿼리가 존재하기 때문에  이 문제를 쉽게 해결할 수 없다. 

이를 위해서는 splay tree 등의 BBST가 필요하다. splay tree에서는 lazy propagation을 쉽게 할 수 있고, 이에 따라 1, 2, 3, 4번의 쿼리를 모두 처리할 수 있다.

다음은 이 문제를 $$O(N log^2 N)$$ 시간에 해결하는 코드이다. ($$X_i$$의 합과 $$Y_i$$의 합이 1000000 이하인데, 시간복잡도의 $$N$$은 정점 개수 뿐만이 아닌 이 수치도 포함된 값이다)



```c++
#include<cstdio>
#include<algorithm>
#include<vector>
#include<queue>
#define N_ 251000
#define szz(x) (int)(x.size())
using namespace std;
int n, Y[N_], X[N_], pL[N_];
vector<int>E[N_], L[N_], Ch[N_];



int Right[N_], Left[N_];
long long D[N_], Num[N_];



struct SplayTree {
    struct node {
        node *l, *r, *p;
        int cnt;
        long long key, lazy;
    } *tree = NULL;

    void Update(node *x) {
        x->cnt = 1;
        if (x->l) x->cnt += x->l->cnt;
        if (x->r) x->cnt += x->r->cnt;
    }

    void Lazy(node *x) {
        x->key += x->lazy;
        if (x->l) {
            x->l->lazy += x->lazy;
        }
        if (x->r) {
            x->r->lazy += x->lazy;
        }
        x->lazy = 0;
    }

    void Rotate(node *x) {
        node *p = x->p;
        node *b;
        if (x == p->l) {
            p->l = b = x->r;
            x->r = p;
        }
        else {
            p->r = b = x->l;
            x->l = p;
        }
        x->p = p->p;
        p->p = x;
        if (b) b->p = p;
        (x->p ? p == x->p->l ? x->p->l : x->p->r : tree) = x;

        Update(p);
        Update(x);
    }


    void Splay(node *x) {
        while (x->p) {
            node *p = x->p;
            node *g = p->p;
            if (g) Rotate((x == p->l) == (p == g->l) ? p : x);
            Rotate(x);
        }
    }

    void Find_Kth(int k) {
        node *x = tree;

        Lazy(x);
        while (1) {
            while (x->l && x->l->cnt > k) {
                x = x->l;
                Lazy(x);
            }
            if (x->l) k -= x->l->cnt;
            if (!k--) break;
            x = x->r;
            Lazy(x);
        }
        Splay(x);
    }

    void Interval(int l, int r) {
        Find_Kth(l - 1);
        node *x = tree;
        tree = x->r;
        tree->p = NULL;
        Find_Kth(r - l + 1);
        x->r = tree;
        tree->p = x;
        tree = x;
    }

    void Add(int l, int r, long long z) { //l부터 r까지 z만큼 더한다
        if (l == 0 && r == tree->cnt - 1) {
            tree->lazy += z;
            return;
        }
        if (l == 0) {
            Find_Kth(r + 1);
            node *x = tree->l;
            x->lazy += z;
            return;
        }
        if (r == tree->cnt - 1) {
            Find_Kth(l - 1);
            node *x = tree->r;
            x->lazy += z;
            return;
        }
        Interval(l, r);
        node *x = tree->r->l;
        x->lazy += z;
    }


    void Insert(long long key) {
        node *p = tree, **pp;
        if (!p) {
            node *x = new node;
            tree = x;
            x->l = x->r = x->p = NULL;
            x->cnt = 1;
            x->lazy = 0;
            x->key = key;
            return;
        }
        while (1) {
            Lazy(p);
            if (key < p->key) {
                if (!p->l) {
                    pp = &p->l;
                    break;
                }
                p = p->l;
            }
            else {
                if (!p->r) {
                    pp = &p->r;
                    break;
                }
                p = p->r;
            }
        }
        node *x = new node;
        *pp = x;
        x->l = x->r = NULL;
        x->p = p;
        x->key = key;
        x->lazy = 0;
        x->cnt = 1;
        Splay(x);
    }

    long long DeleteLeft() { // 가장 작은 기울기가 0보다 작으면 제거한다.
        Find_Kth(0);
        Lazy(tree);
        long long res = tree->key;

        
        node *p = tree;
        tree = tree->r;
        if(tree)tree->p = NULL;
        delete p;
        return res;
    }

    int SZ() {
        if (tree == NULL)return 0;
        return tree->cnt;
    }

}IT[N_];



void DFS(int a, int pp) {

    for (int i = 0; i < E[a].size(); i++) {
        int x = E[a][i];
        if (x == pp)continue;
        DFS(x, a);
        pL[x] = L[a][i];
        Ch[a].push_back(x);
    }
}

void UDT(int a) {
    int d = X[a] - Y[a];
    int c = pL[a];
    Right[a] += d, Left[a] += d; //X[a] - Y[a]만큼 평행이동
    int u = Num[a]; //dp[a]는 IT[u]에 저장되어 있음
    if (Left[a] > 0) {
        for (int i = 0; i < Left[a]; i++) {
            IT[u].Insert(0);
        }
        Left[a] = 0;
    }
    //y = c|x| 더하기
    if (Left[a] < 0 && Left[a] < Right[a]) {
        IT[u].Add(0, min(Right[a],0) - Left[a]-1, -c);
    }
    if (Right[a] > 0) {
        IT[u].Add(-Left[a], Right[a] - Left[a] - 1, c);
    }
    long long z = D[a] - 1ll * Left[a] * c;
    int i, cc = 0;
    //기울기 0이하인 왼쪽부분 제거
    for (i = Left[a]; i < Right[a]; i++) {
        IT[u].Find_Kth(0);
        if (IT[u].tree->key <= 0) {
            Left[a]++;
            z += IT[u].tree->key;
            IT[u].DeleteLeft();
        }
        else break;
    }
    D[a] = z;
}

void Do(int a) {
    if (Ch[a].empty()) {
        Right[a] = Left[a] = 0;
        D[a] = 0;
        Num[a] = a;
        return;
    }
    for (auto &x : Ch[a]) {
        Do(x);
    }
    vector<long long>Z;
    int ls = 0, rs = 0;
    D[a] = 0;
    int Mx = -1, pv = -1; //vertex a에 대한 dp값의 기울기는 splay tree인 IT[Num[a]]에 저장된다
    for (auto &x : Ch[a]) {
        UDT(x);
        ls += Left[x];
        rs += Right[x];
        D[a] += D[x];
        if (Mx < IT[Num[x]].SZ()) {
            Mx = IT[Num[x]].SZ();
            pv = x;
        }
    }
    Num[a] = Num[pv];
    for (auto &x : Ch[a]) {
        if (x == pv)continue; // heavy child인 경우 넘어감
        int t = Num[x];
        int sz = IT[t].SZ();
        for (int i = 0; i < sz; i++) {
            IT[Num[a]].Insert(IT[t].DeleteLeft()); //light child의 모든 노드를 heavy child에 추가
        }
    }
    Left[a] = ls, Right[a] = rs;
}
int main() {
    int i, a, b, c;
    scanf("%d", &n);
    for (i = 1; i < n; i++) {
        scanf("%d%d%d", &a, &b, &c);
        E[a].push_back(b);
        E[b].push_back(a);
        L[a].push_back(c);
        L[b].push_back(c);
    }
    for (i = 1; i <= n; i++)scanf("%d%d", &X[i], &Y[i]);
    DFS(1, 0);
    Do(1);
    UDT(1);
    long long s = D[1];
    for (i = Left[1]; i < 0; i++) {
        s += IT[Num[1]].DeleteLeft();
    }
    printf("%lld\n", s);
}

```



# 마치며

Conquer the world는 이런 방식으로 풀리는 문제 중 굉장히 까다로운 문제이고, 보통의 경우 splay tree 등이 필요하지 않고 priority queue나 set을 사용하여 풀 수 있다. 이러한 방식으로 풀리는 대표적인 문제로는

APIO 2016 Fireworks (https://www.acmicpc.net/problem/12736)

이 있다.

또한, 꼭 트리가 아니더라도 2차원 dp를 그래프로 생각하여 priority queue나 set으로 해결하는 아이디어도 많이 쓰이는 아이디어이다. 이에 관련된 문제로는

JAG Practice Contest 2017 Farm Village (https://www.acmicpc.net/problem/15527) 

등의 문제들이 있다.

Conquer the world를 도전하기 전에 이 문제를 먼저 풀어보는 것을 추천한다.
