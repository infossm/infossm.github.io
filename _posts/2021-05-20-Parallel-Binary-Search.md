---
layout: post
title: Parallel Binary Search
date: 2021-05-20 04:00
author: rdd6584
tags: [algorithm]
---

이 게시글은 이분 탐색에 대한 지식을 필요로 합니다.
그리고 union-find, segment tree 자료구조에 대해 알고 있으면 좋습니다.

아래 문제를 봅시다.
## 제한된 메모리([링크](https://www.acmicpc.net/problem/12921))

길이 $N$의 배열 $A$에서, $q$번째로 작은 원소를 구하는 쿼리를 여러 번 해결하는 문제입니다.
하지만, 문제 지문에 나와 있듯이, 이 문제의 메모리 제한은 4MB로 $A$에 대한 메모리를 할당할 수 없습니다.

하나의 쿼리에 대해 해결해 봅시다.

$f(a) =$ 배열 $A$에서 $a$ 이하인 수의 개수라고 합시다.
$f(a) \geq q+1$인 경우, $q$번째로 작은 원소는 $a$ 이하입니다.

함수 $f$는 단조증가하므로 이분 탐색을 적용할 수 있고, 점화식을 통해 만들어지는 $N$개의 수를 순서대로 확인하면서 $f(mid)$를 계산할 수 있습니다.

이 경우,
$X$를 배열의 최댓값이라고 할 때, 하나의 쿼리를 해결하는 데에 $O(NlogX)$의 시간이 소요되어, 전체 문제를 $O(QNlogX)$의 시간에 해결할 수 있습니다.

여전히 많이 느립니다. 이를 개선해 봅시다.
이 방법에서 주목할 만한 성질은,

1. 각 쿼리를 이분 탐색으로 해결할 수 있고
2. 모든 쿼리는 동일한 배열을 대상으로 독립적으로 수행되는 쿼리라는 것입니다.

어차피, 동일한 배열에 대한 것이니까 모든 쿼리를 한꺼번에 처리하는 방법이 존재할 것 같습니다.

생각해 보면, 1회의 배열 생성 과정을 통해 $Q$개의 $f(a_i)$에 대한 값을 전부 계산할 수 있습니다.
배열의 원소를 순서대로 확인하면서, $A_i$ 이상인 $a_i$에 대해 카운팅을 해주면 되겠죠.



이 과정은 정렬, 이분 탐색, 부분합을 이용하여 $O(N+QlogQ)$에 해결할 수 있습니다.

```c++
// 배열 A는 명시적이며, 직접 선언할 수 없음에 유의하세요.
sort(a, a + Q);
for (int i = 0; i < n; i++) {
    int idx = lower_bound(a, a + Q, A[i]) - a;
    f[idx]++;
}
for (int i = 1; i < n; i++) f[i] += f[i - 1];
```



### 병렬 이분 탐색

$Q$개의 쿼리에 대해, 이분 탐색을 병렬적으로 진행한다고 해봅시다.

우리는 $O(N+QlogQ)$에 각 쿼리의 $mid_i$ 값인 $a_i$를 계산할 수 있으므로, 이를 바탕으로 각 쿼리의 $left_i$ 혹은 $right_i$를 갱신해나갈 수 있습니다. 이 과정이 $O(logX)$번 수행되므로, 프로그램의 전체 수행 시간은 $O(logX(N+QloqQ))$가 됩니다.



문제의 전체 코드입니다.

```c++
#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;
typedef long long ll;
typedef pair<int, int> pii;

pii lr[1000];   // 각, 쿼리에 대한 이분 탐색의 left값과 right값
int mid[1000], query[1000], psum[1001];

int main() {
    int n;
    ll x, a, b;
    scanf("%d %lld %lld %lld", &n, &x, &a, &b);

    int q, t;
    scanf("%d", &q);

    for (int i = 0; i < q; i++) {
        scanf("%d", &t);
        query[i] = t + 1;
        lr[i] = {0, MOD - 1};
    }
    sort(query, query + q);

    for (int cnt = 0; cnt < 31; cnt++) {
        memset(psum, 0, sizeof(psum));
        for (int i = 0; i < q; i++) mid[i] = (lr[i].first + lr[i].second) / 2;
        
        /* 앞서 쿼리를 크기순으로 정렬해두었으므로, mid값도 병렬이분탐색 과정에서 오름차순으로 유지됩니다.*/

        ll now = x;
        for (int i = 0; i < n; i++) {
            int idx = lower_bound(mid, mid + q, now) - mid;
            psum[idx]++;
            now = (now * a + b) % MOD;
        }
        for (int i = 1; i < q; i++) psum[i] += psum[i - 1];
        for (int i = 0; i < q; i++) {
            if (psum[i] >= query[i]) lr[i].second = mid[i] - 1;
            else lr[i].first = mid[i] + 1;
        }
    }

    ll sum = 0;
    for (int i = 0; i < q; i++) sum += lr[i].first;

    printf("%lld", sum);
}
```



## 크루스칼의 공([링크](https://www.acmicpc.net/problem/1396))

그래프 $G$가 주어지고, 가중치가 $c$이하인 간선들만 남겨서 정점 $x$와 $y$를 연결할 수 있다고 할 때, 가장 작은 $c$와 그때의 $x$가 속한 컴포넌트의 크기를 구하는 쿼리가 $Q$개 주어집니다.



이 문제에서, 두 정점의 연결 여부에 대한 함수 $f(x, y, c)$는 $c$에 대해 단조성을 가집니다.

따라서, 이분 탐색으로 $c$이하의 간선들만을 남겨서 두 정점이 연결되어 있는지 확인할 수 있습니다.

또한, 각 쿼리는 동일한 그래프을 대상으로 독립적으로 수행됩니다.



위 문제와 마찬가지로 병렬 이분 탐색을 이용해서 문제를 해결할 수 있을 것 같고, 실제로도 그렇습니다.

각, 이분 탐색 과정에서

0. 간선이 존재하지 않는 그래프 $G'$ 존재.
1. $G$에서 가중치가 작은 간선부터 순서대로 보면서, 이 간선과 가중치가 같은 간선들을 전부 $G'$에 추가
2. $f(x_i, y_i, mid_i)$에서 이 간선의 가중치 이하인 $mid_i$를 가지는 쿼리에 대해 확인.
3. 확인된 정보들을 토대로 $left_i$ 혹은 $right_i$ 갱신

union-find를 이용하면 비교적 쉽게 구현할 수 있습니다.



이를 구현한 코드는 아래와 같습니다.

```c++
#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;

struct node {
    int u, v, c;
} edg[100001];
pii query[100000], lr[100000];
int ans[100000];    // 쿼리에서 컴포넌트 크기

int p[100000], rotn[100000];    // union-find 관리 변수

int find(int a) {
    if (p[a] == -1) return a;
    return p[a] = find(p[a]);
}

void merge(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return;
    rotn[a] += rotn[b];
    p[b] = a;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int a, b, c;
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &a, &b, &c);
        a--; b--;
        edg[i] = { a, b, c };
    }
    edg[m++] = { 0, 0, 1000005 };
    sort(edg, edg + m, [](node v1, node v2) { return v1.c < v2.c; });

    int q;
    scanf("%d", &q);
    for (int i = 0; i < q; i++) {
        scanf("%d %d", &query[i].first, &query[i].second);
        query[i].first--; query[i].second--;
        lr[i] = { 0, m - 1 };
    }

    for (int cnt = 0; cnt < 18; cnt++) {
        memset(p, -1, sizeof(p));
        fill(rotn, rotn + n, 1);
        vector<int> v[100001];

        for (int i = 0; i < q; i++)
            if (lr[i].first <= lr[i].second)
                v[(lr[i].first + lr[i].second) / 2].push_back(i);

        int l = 0;
        for (int i = 0; i < m; i++) {
            while (l < m && edg[i].c >= edg[l].c) {
                merge(edg[l].u, edg[l].v);
                l++;
            }
            for (int j : v[i]) {
                a = find(query[j].first), b = find(query[j].second);
                if (a == b) {
                    ans[j] = rotn[a];
                    lr[j].second = i - 1;
                }
                else lr[j].first = i + 1;
            }
        }
    }

    for (int i = 0; i < q; i++) {
        if (ans[i]) printf("%d %d\n", edg[lr[i].first].c, ans[i]);
        else printf("-1\n");
    }
}
```


## 기타 연습 문제
### 1. 유성([링크](https://www.acmicpc.net/problem/8217))
각, 회원국에 대한 정답을 쿼리라고 합시다.
$f(j, x)$에서 $x$번째 유성우까지 내렸을 때, $j$에 내린 유성의 수라고 합시다. 이 함수는 $x$에 대해 단조성을 가지므로, $p_j$와 비교함으로 각 쿼리를 이분 탐색으로 해결할 수 있습니다.
모든 쿼리는 동일한 유성우 과정을 대상으로, 독립적으로 처리되므로 병렬 이분 탐색을 적용할 수 있다고 추측할 수 있고 실제로 그렇습니다.

각, 이분 탐색 과정에서
유성우를 순서대로 내리면서, 이 유성우를 $mid$값으로 가지는 회원국에 대해서 $left, right를$ 결정해 주면 되고. 각 유성우를 내리는 과정은 세그먼트 트리, 펜윅 트리와 같은 자료구조로 해결할 수 있습니다.


### 2. Hangar Hurdles([링크](https://www.acmicpc.net/problem/13952))
문제에서 상자의 크기는 홀수이므로, 상자의 크기가 2만큼 커지는 것은 모든 벽들이 상하좌우 대각선 8방향으로 넓어지는 것과 같습니다.

각 쿼리에서, 상자 크기에 대해 이분 탐색을 진행합시다. 상자 크기만큼 벽을 넓혀서 두 정점 간의 연결을 확인해 줄 수 있습니다.
모든 쿼리는 동일한 행렬을 대상으로, 독립적으로 처리되므로 병렬 이분 탐색을 적용할 수 있다고 추측할 수 있고 실제로 그렇습니다.

각, 이분 탐색 과정에서
아주 큰 상자부터 크기 1까지 순서대로 고려하면서 벽을 1칸씩 좁혀 나갑시다. 이 과정을 위해서, 각 상자 크기마다 어떤 벽이 삭제되어야 하는 지 미리 계산해두면 좋습니다.
크루스칼의 공 문제와 마찬가지로, 연결 가능한 땅들에 대해 union-find로 관리하고 각 쿼리에 대한 정보를 갱신해 주면 됩니다.


### 3. 히스토그램에서 가장 큰 직사각형과 쿼리([링크](https://www.acmicpc.net/problem/16977))
마찬가지로, 모든 쿼리는 동일한 히스토그램을 대상으로 독립적으로 처리되므로 병렬 이분 탐색을 적용할 수 있다고 추측할 수 있습니다.

각 쿼리에서, 높이가 결정되면 넓이도 유일하게 결정되므로 높이를 최대화하는 문제와 동일합니다.

$f(l, r, x)$ : 높이 $x$이상인 막대들만 남겼을 때, $l$ 과 $r$ 사이에 $m$개의 연속한 직사각형이 존재하는지.

이 함수는 $x$에 대해 단조성을 가지므로 이분 탐색 적용이 가능합니다.

하지만, 이 함수를 적용하는 것이 쉽지 않아 보입니다.

이 함수의 계산은 maximum subarray 문제와 비슷하게 분할 정복으로 간단히 할 수 있지만,
우리는 모든 쿼리에 대해 병렬적으로 처리하는 것에 더 관심이 있으므로 비슷하게,
아래와 같은 세그먼트 트리를 생각해 봅시다.

각 노드에서는 변수 $l, r, w$를 관리합니다.

$l$ : 이 구간의 왼쪽에서부터 연속한 직사각형의 개수.

$r$ : 오른쪽에서부터 연속한 직사각형의 개수.

$w$ : 이 구간에 속한 연속한 직사각형의 개수 중 최댓값.

이런 구성을 가지는 세그먼트 트리를 만들고 구간 $[l, r]$에 대한 $w$값을 확인하여 $m$과 비교해 주면 됩니다.

이제, 앞선 문제의 방식들과 비슷하게 높이가 가장 낮은 막대부터 하나씩 추가하면서, 각 쿼리에 대해 결정할 수 있습니다.


읽어주셔서 감사합니다.
