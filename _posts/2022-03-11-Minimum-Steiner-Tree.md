---
layout: post
title:  "Minimum Steiner Tree의 계산"
author: ho94949
date: 2022-03-10 03:00
tags: [DP, Bitmask, Stiner Tree]
---

# 서론
그래프 $G$와, 정점의 부분집합 $T$가 주어졌을 때, $T$를 모두 포함하는 Spanning Tree ($G$의 부분그래프 중, 정점과 간선 일부를 선택하여 만든 트리)를 $T$의 Steiner Tree라고 한다. 가중치가 주어진 방향 없는 그래프 $G$에 대해 간선 가중치 합이 가장 작은 Steiner Tree를 계산하는 문제는 [일반적인 상황에서 NP-Hard](http://www.secmem.org/blog/2020/08/18/karp-21-np-complete/)임이 알려져있다. 이 게시글에서는, 해당 Steiner Tree를 계산하는 동적계획법을 다룬다.

## Naïve

편의상, 그래프의 $G = (V, E)$의 정점 개수를 $N$, 간선 개수를 $M$, 정점 부분집합 $T$의 크기를 $K$라고 하자.

가장 쉽게 생각할 수 있는 방법은, 가능한 모든 부분그래프 $2^M$ 개를 모두 구해서, $O(M)$의 시간을 사용해서 steiner tree인지를 확인하는 방법이 있다. 이 경우 시간 복잡도가 $O(M 2^M)$정도이다.

생각할 수 있는 다른 방법은 시간복잡도 $O(M \log M + 2^{N-K} M)$정도의 시간을 사용해서 해결하는 풀이이다. $K$ 개의 정점을 모두 포함하는 정점의 부분집합은 총 $2^{N-K}$ 개가 있다. 모든 가능한 부분집합에 대해서 Minimum Spanning Tree문제를 해결한 후, 해당 Minimum Spanning Tree중의 최솟값을 구하면, 그것이 답이 된다.

두 가지 시간복잡도 모두 매우 크며, 우리는 $K$가 작은 경우에는 해결하지 못하고 있다. 다음에 소개할 알고리즘은, $K$가 작은 경우에, 해당 이점을 이용할 수 있다.

## Dreyfus-Wagner

시간복잡도 $O(NM + 3^K N + 2^K N^2)$에 문제를 해결하는 방법이 존재하고, 이 방법에는 Dreyfus-Wagner법이라는 이름이 붙어있다.

어떤 임의의 Spanning Tree가 존재한다고 생각하자. 이 Tree의 어떤 간선 $p-q$를 잡아서 해당 간선의 양쪽이, 왼쪽은 정점의 부분집합 $A$와 $p$가 포함된 Steiner Tree, 오른쪽은 정점의 부분집합 $B$와 $q$가 포함된 Spanning Tree라고 하자. $p-q$를 이어서 만들 수 있는 새로운 Spanning Tree는 정점의 부분집합 $A \cup B$와 정점 $p, q$를 포함한 Spanning Tree이다.

모든 Spanning Tree는 이와 같은 방법으로 만들 수 있다. (임의의 트리 간선을 계속 끊어나가면 된다.) 그리고, 우리가 관심을 가질 것은, 정점 부분 집합의 모든 간선이 아닌, 주어진 $T$의 부분집합 뿐이다. 그렇기 때문에, 다음과 같은 방법으로 동적계획법을 사용할 수 있다: $D_{S,\ p}$를 $S \cup \{p\}$의 모든 정점을 포함하는 최소 크기의 Spanning Tree라고 하자. 또한, $d(p, q)$를 두 정점 $p, q$ 사이의 거리라고 하자. 우리는 $S$를 $T$의 부분집합인 경우만 생각 할 것이다.

- $S = \{q\}$인 경우, $D_{S,\ p} = d(p, q)$가 된다. 이는 $p$와 $q$를 모두 포함하는 최소 크기의 트리이다.
  - Bellman-Ford등을 사용하여, $O(NM)$ 시간에 모든 정점 쌍의 최단거리를 구할 수 있다.
- $\lvert S \rvert \ge 2$ 인 경우, 재귀적으로 구해나간다. $D_{S,\ p} = \min_{A \cup B = S,\ q \in V}(D_{A,\ p} + D_{B,\ q} + d(p, q))$가 된다.
  - 시간복잡도는 $O(3^K N^2)$이고, 이는 $T$의 모든 부분집합에 대해서, 해당 부분집합의 부분집합을 순회하는 시간복잡도가 $O(3^K)$이기 때문이다.

여기서 한 가지의 아이디어를 사용하여 시간복잡도를 줄일 수 있는데, 우리가 트리를 구성할 때, $p-q$를 끊어가면서 트리를 구성했다. 우리는 가능한 $p-q$ 경로를 끊는 방법으로 문제를 해결했고, 이것이 시간복잡도 $O(N^2)$에 기여한다. 이를, 정점 하나를 기준으로 여러개의 서브트리를 붙여나가고, 정점 하나에서 경로를 연장한다는 아이디어를 사용하자. 이렇게 되면, $\lvert S \rvert \ge 2$인 경우를 다음 두 가지로 나눌 수 있다.

- $D_{S,\ p} = \min_{A \cup B = S} (D_{A,\ p} + D_{B,\ p})$
  - 시간복잡도는 $O(3^KN)$이다. 이 방법으로, 하나의 정점을 기준으로 여러개의 Steiner Tree를 모을 수 있다.
- $D_{S,\ p} = \min_{q \in V} (D_{S,\ q} + d(p, q))$
  - 시간복잡도는 $O(2^K N^2)$이다. 이 방법으로, $T$의 부분집합을 모으는 기준점을 바꿔줄 수 있다.

이를 통해, 총 시간복잡도 $O(NM + 3^K N + 2^K N^2)$에 문제를 해결할 수 있다.



$K$가 큰 경우의 알고리즘인 $O(M \log M + 2^{N-K} M)$을 같이 사용해서, $K \sim (\log_62) N \approx 0.387N$ 을 기준으로 $K$가 작을 경우에는 Dreyfus-Wagner를, 클 경우 Spanning Tree를 계산하는 방식을 사용하면 $\tilde O(1.53^N)$ 정도 시간에 정점이 $N$ 개 있는 그래프의 Steiner Tree를 구할 수 있다.

## 구현

아래는 C++17로 구현한 Minimum Steiner Tree를 구하는 구현체이다. MST를 이용하는 방법과 Dreyfus Wagner를 이용하는 방법 모두가 구현이 되어있다. `N`은 정점 개수, `E`는 (가중치, 간선의 한 정점, 간선의 다른 정점)을 차례대로 표현한 `tuple`의 `vector`, `T`는 집합 $T$에 포함된 정점을 차례로 나열한 것이다. 모든 정점은 0-based로 인덱싱 되어있으며, 모든 가중치의 합이 `INT_MAX/2`를 넘지 않음을 가정한다.

해당 구현체는 $N = 35, \lvert K \rvert = 15$ 일 때 Dreyfus-Wagner로 500ms, $N=36, \lvert K \rvert = 16$ 일 때 MST로 1000ms 정도를 사용한다. 

```cpp
#include <algorithm>
#include <climits>
#include <numeric>
#include <tuple>
#include <vector>
using namespace std;

namespace SteinerTree {
    struct DSU {
        vector<int> p;
        DSU(int N) { init(N); }
        void init(int N) { p.resize(N); iota(p.begin(), p.end(), 0); }
        int Find(int a) { return a == p[a] ? a: p[a] = Find(p[a]); }
        bool Union(int a, int b) { a = Find(a), b = Find(b), p[a] = b; return a != b; }
    };

    int steiner_tree_mst(int N, vector<tuple<int, int, int>> E, vector<int> T) {
        int K = T.size();
        vector<bool> C(N); for(int t: T) C[t] = 1;
        vector<int> S; for(int i=0; i<N; ++i) if(!C[i]) S.push_back(i);
        sort(E.begin(), E.end(), [&](auto a, auto b){ return get<2>(a) < get<2>(b); });

        int ans = INT_MAX; DSU dsu(N);
        for(int i=0; i<1<<(N-K); ++i) {
            dsu.init(N); int cnt = K;
            for(int j=0; j<N-K; ++j)
                if(i&(1<<j)) C[S[j]] = 1, ++cnt;
                else C[S[j]] = 0;
            int r = 0;
            for(auto [u, v, w]: E) if(C[u] && C[v] && dsu.Union(u, v)) {
                r += w;
                if(--cnt == 1) break;
            }
            if(cnt == 1) ans = min(ans, r);
        }
        return ans;
    }

    int steiner_tree_dreyfus_wager(int N, vector<tuple<int, int, int>> E, vector<int> T) {
        int K = T.size();
        vector<vector<int>> d(N, vector<int>(N, INT_MAX/2));
        for(int i=0; i<N; ++i) d[i][i] = 0;
        for(auto [u, v, w]: E) d[u][v] = d[v][u] = min(d[u][v], w);
        for(int k=0; k<N; ++k) for(int i=0; i<N; ++i) for(int j=0; j<N; ++j)
            d[i][j] = min(d[i][j], d[i][k] + d[k][j]);

        vector<vector<int>> D(1<<K, vector<int>(N, INT_MAX/2));
        for(int i=0; i<K; ++i) for(int j=0; j<N; ++j) D[1<<i][j] = d[T[i]][j];
        for(int s=1; s<1<<K; ++s) {
            for(int a=0; (a=(a-s)&s);) {
                int b = s-a;
                if(b<a) break;
                for(int i=0; i<N; ++i) D[s][i] = min(D[s][i], D[a][i] + D[b][i]);
            }
            for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) D[s][i] = min(D[s][i], D[s][j] + d[j][i]);
        }
        return *min_element(D.back().begin(), D.back().end());
    }
}
```

