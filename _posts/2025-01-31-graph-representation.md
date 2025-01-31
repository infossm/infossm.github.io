---
layout: post

title: "다양한 그래프 표현 방법"

date: 2025-01-31

author: jinhan814

tags: [algorithm, graph-theory]
---

## 개요

그래프는 다양한 방법으로 표현될 수 있습니다. 각 방법은 특정 상황에서 장단점을 가지며, 문제의 요구 사항에 따라 적절한 표현 방법을 선택하는 것이 중요합니다.

이 글에서는 그래프를 표현하는 4가지 방법인 인접 행렬(Adjacency Matrix), 인접 리스트(Adjacency List), CSR(Compressed Sparse Row), 그리고 XOR Linked Tree에 대해 설명합니다.

(XOR Linked Tree 개념은 코드포스 글 [https://codeforces.com/blog/entry/135239](https://codeforces.com/blog/entry/135239)을 참고해 작성했습니다)

## 1. 인접 행렬 (Adjacency Matrix)

### 개념
인접 행렬은 그래프를 2차원 배열로 표현하는 방법입니다. 그래프에 `i`번 정점에서 `j`번 정점으로 가는 간선이 있다면 `i`행 `j`열의 원소 `g[i][j]`를 $1$로, 그렇지 않으면 $0$으로 표시합니다. 가중치 그래프의 경우는 `g[i][j]`에 가중치를 저장할 수도 있습니다.

인접 행렬은 `i`번 정점에서 `j`번 정점로 가는 간선이 존재하는지 여부를 $\mathcal{O}(1)$에 구할 수 있다는 장점이 있으며, 공간 복잡도가 $\mathcal{O}(V^2)$이고 인접한 정점을 순회하는 시간 복잡도가 $\mathcal{O}(V)$라는 단점이 있습니다.

이러한 성질에 의해 인접 행렬은 주로 $E \simeq V^2$인 밀집 그래프(dense graph)가 주어지는 상황에 자주 사용됩니다.

## 2. 인접 리스트 (Adjacency List)

### 개념
인접 리스트는 그래프를 각 정점마다 연결된 정점들의 리스트를 저장하며 표현하는 방법입니다. 인접 리스트는 `i`번 정점에 대해 `i`번 정점에서 `j`번 정점으로 가는 간선이 있는 정점 `j`들을 `adj[i]`에 저장합니다. 가중치 그래프의 경우는 간선의 가중치 `c`를 `j`와 함께 pair로 `adj[i]`에 저장할 수 있습니다.

인접 리스트를 이용하면 $\mathcal{O}(V + E)$의 공간 복잡도에 그래프를 저장할 수 있으며 인접한 정점을 순회하는 시간 복잡도가 실제로 인접한 정점의 개수인 $\mathcal{O}(deg(i))$와 같다는 장점이 있습니다. 단점은 `i`번 정점에서 `j`번 정점으로 가는 간선이 존재하는지 여부를 확인하는 시간복잡도가 $\mathcal{O}(deg(i))$로 인접 행렬에 비해 비효율적입니다.

인접 리스트는 $E \ll V^2$인 희소 그래프(sparse graph)에서 효율적이며, 정점과 간선의 개수가 모두 $10^5$ scale인 그래프 문제에서 자주 사용됩니다.

## 3. CSR(Compressed Sparse Row) 표현

### 개념
인접 리스트를 이용한 그래프 표현법은 $V$가 $10^5$ scale로 커도 그래프를 $\mathcal{O}(V + E)$의 공간 복잡도로 저장할 수 있습니다. 하지만 `std::vector`와 같은 동적 배열을 이용하면 array doubling에 의한 성능 저하가 있고, cache hit rate가 떨어진다는 한계점이 있습니다.

인접 행렬을 CSR(Compressed Sparse Row) 표현을 이용해 저장하면 인접 리스트의 단점을 보완할 수 있습니다. $E \ll V^2$인 경우 인접 행렬의 대부분의 원소는 $0$입니다. 이 사실을 이용하면 $V^2$의 메모리를 이용해 명시적으로 인접 행렬을 저장하는 대신 `g[u][v]`가 $0$이 아닌 `(u, v)`에 대한 정보만 저장하며 메모리 사용량을 줄일 수 있습니다.

CSR 표현은 길이가 $E$인 배열 `csr[0], ..., csr[E - 1]`과 길이가 $V + 1$인 배열 `cnt[0], ..., cnt[V]`로 이루어집니다.

- `csr` 배열은 `g[u][v]`가 $0$이 아닌 `(u, v)` pair를 오름차순 정렬한 뒤 `v`를 나열한 배열입니다.
- `cnt` 배열은 $u$에서 나가는 간선의 개수를 `out[u]`라 할 때 `out` 배열에 대한 prefix sum 배열입니다.

예를 들어 그래프가 아래의 인접 행렬 표현을 가진다고 합시다.

$$
\begin{bmatrix}
0 & 1 & 0 & 0 & 1 \\
1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
1 & 0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 & 0 \\
\end{bmatrix}
$$

이 그래프의 CSR 표현을 구하면 `csr`은 $\{ 2, 5, 1, 2, 4, 1, 5, 3 \}$, `cnt`는 $\{ 0, 2, 4, 5, 7, 8 \}$이 됩니다. CSR 표현은 간선이 고정된 길이의 메모리 상에서 인접한 하나의 배열에 저장되기 때문에 array doubling이 일어나지 않고, cache hit rate가 높다는 장점이 있습니다.

### 구현 코드
```cpp
vector e(m, tuple(0, 0, 0));
for (auto& [a, b, c] : e) cin >> a >> b >> c;

vector cnt(n + 2, 0);
vector csr(m, pair(0, 0));
for (auto [a, b, c] : e) cnt[a + 1]++;
for (int i = 1; i < cnt.size(); i++) cnt[i] += cnt[i - 1];
for (auto [a, b, c] : e) csr[cnt[a]++] = pair(b, c);

int cur = /* ... */;
for (int i = cnt[cur - 1]; i < cnt[cur]; i++) {
	auto [nxt, cost] = csr[i];
	/* ... */
}
```

위의 코드는 $n$개의 정점과 $m$개의 간선으로 이루어진 가중치가 있는 방향 그래프를 csr 표현을 이용해 저장하는 예시 코드입니다.

`e`에 가중치가 `c`인 `a → b` 간선이 `(a, b, c)` 형태로 저장되어 있다고 합시다. 세 개의 for문을 이용하는 counting sort 과정에서 `cnt[i]`는 `i - 1`번 정점에서 나가는 간선의 개수, `1, ..., i - 1`번 정점에서 나가는 간선의 개수, `1, ..., i`번 정점에서 나가는 edge의 개수로 차례로 바뀝니다.

counting sort가 끝난 뒤 `csr` 배열의 $[cnt[i - 1], cnt[i])$ 범위에는 `i`번 정점에서 나가는 edge가 저장되고, 따라서 `csr` 배열에서 해당 범위를 순회하며 인접 리스트에서와 같이 인접한 정점을 순회할 수 있습니다.

### 성능 비교
1. [G4] BOJ 1753 - 최단경로
- link : [https://www.acmicpc.net/problem/1753](https://www.acmicpc.net/problem/1753)
- adjacency list (112ms, 9112kb) : [http://boj.kr/3ea45a03506446bd9fcc608ace0fd99c](http://boj.kr/3ea45a03506446bd9fcc608ace0fd99c)
- csr representation (88ms, 8552kb) : [http://boj.kr/a071cb4eb4da47a2b2f19365dcc88c16](http://boj.kr/a071cb4eb4da47a2b2f19365dcc88c16)

2. [S2] BOJ 11725 - 트리의 부모 찾기
- link : [https://www.acmicpc.net/problem/11725](https://www.acmicpc.net/problem/11725)
- adjacency list (48ms, 10404kb) : [http://boj.kr/737a1eff8a88480ea4512770b4344067](http://boj.kr/737a1eff8a88480ea4512770b4344067)
- csr representation (32ms, 7028kb) : [http://boj.kr/ba47a71aec3449879c60b86d4aad65fd](http://boj.kr/ba47a71aec3449879c60b86d4aad65fd)

인접 리스트를 이용한 구현과의 성능 비교를 보면 CSR 표현은 일반적인 그래프 문제에서 메모리 사용량과 실행 시간을 유의미하게 줄여준다는 걸 알 수 있습니다. 이러한 이유로 atcoder library를 포함한 많은 라이브러리에서 그래프의 저장에 CSR 표현을 이용합니다.

때문에 간단한 구현에는 구현량이 적은 인접 리스트를 이용하는 것도 좋지만, 성능 최적화가 필요한 경우에는 CSR 표현을 같이 고려해보면 좋습니다.