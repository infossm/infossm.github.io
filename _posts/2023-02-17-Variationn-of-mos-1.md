---
layout: post
title: "Variation of Mo's Algorithm 1"
date: 2023-02-17 23:50
author: jthis
tags: [algorithm]
---

안녕하세요 jthis 입니다.

이 글에서는 Mo's Algorithm의 variation에 대해 소개하겠습니다.

# Mo's Algorithm
Mo's Algorithm은 구간 쿼리를 offline으로 해결하는 알고리즘입니다. Mo's Algorithm에 대한 자세한 설명은 [Mo's Algoithm](https://infossm.github.io/blog/2019/02/09/mo's-algorithm/)에서, Mo's Algorithm을 사용한 트리의 서브 트리에 대한 쿼리나 경로에 대한 쿼리는 [Mo's on Tree](https://infossm.github.io/blog/2019/12/17/Mos-Algorithm-on-Trees/) 에서 찾아보실 수 있으니 읽고 오시면 도움이 될 것입니다.

# Mo's Algorithm with Rollback
Mo's Algorithm을 사용할 때 원소를 제거만 하거나 삽입만 하고 싶다는 생각을 해보신 적 있으실 겁니다.

간략하게 설명하자면 원소를 삽입하는 연산만 하고 싶을 때는 원소를 제거하는 연산을 원소를 삽입하는 연산을 취소하는 연산으로 대체 하여 원소를 삽입하는 연산과 취소하는 연산만으로 쿼리를 처리할 수 있습니다. 문제를 풀면서 설명하겠습니다.

[Matryoshka Dolls](https://www.acmicpc.net/problem/23162) 이 문제를 보고 옵시다.

구간 $[l,r]$에 있는 마트료시카에 대해 가장 작은 마트료시카를 다음 크기의 마트료시카에 집어넣습니다.

그때 걸리는 시간이 두  마트료시카의 위치 차이일 때, 위 과정을 반복하여 마트료시카가 하나 남을 때까지 걸린 총시간을 구하는 문제입니다.

가장 쉬운 풀이는 Mo's를 사용하여 $i$번째 마트료시카가 추가 및 삭제될 때마다 $p_i$ 보다 작은 가장 큰 값의 위치와 $p_i$보다 큰 가장 작은 값의 위치를 찾아 더하거나 빼면 됩니다.

이 연산은 Segment Tree 또는 Ordered Set을 이용하면 $O(log N)$의 시간복잡도로 해
결할 수 있어 총 $O(N \sqrt N logN)$의 시간 복잡도에 해결할 수 있습니다.
```c++
const int inf = 100'101;
int arr[inf];
int solve[inf * 5];

//https://codeforces.com/blog/entry/61203
inline int64_t hilbertOrder(int x, int y, int pow, int rotate) {
    if (pow == 0) return 0;
    int hpow = 1 << (pow - 1);
    int seg = (x < hpow) ? ((y < hpow) ? 0 : 3) : ((y < hpow) ? 1 : 2);
    seg = (seg + rotate) & 3;
    const int rotateDelta[4] = {3, 0, 0, 1};
    int nx = x & (x ^ hpow), ny = y & (y ^ hpow);
    int nrot = (rotate + rotateDelta[seg]) & 3;
    int64_t subSquareSize = int64_t(1) << (2 * pow - 2);
    int64_t ans = seg * subSquareSize;
    int64_t add = hilbertOrder(nx, ny, pow - 1, nrot);
    ans += (seg == 1 || seg == 2) ? add : (subSquareSize - add - 1);
    return ans;
}

struct query {
    int a, b, c;
    long long order;

    void init() {
        order = hilbertOrder(a, b, 21, 0);
    }

    bool operator<(const query &t) const {
        return order < t.order;
    }
} q[inf * 5];

set<pair<int, int>> st;
int res;

void pluses(int v) {
    int w = arr[v];
    if (st.empty()) {
        st.insert({w, v});
        return;
    }
    auto k = st.lower_bound({w, 0});
    if (st.end() != k && k != st.begin()) {
        res -= abs(k->second - prev(k)->second);
    }
    if (k != st.end()) {
        res += abs(k->second - v);
    }
    if (k != st.begin()) {
        res += abs(prev(k)->second - v);
    }
    st.insert({w, v});
}

void minuses(int v) {
    int w = arr[v];
    st.erase({w, v});
    if (st.empty())return;
    auto k = st.lower_bound({w, 0});
    if (st.end() != k && k != st.begin()) {
        res += abs(k->second - prev(k)->second);
    }
    if (k != st.end()) {
        res -= abs(k->second - v);
    }
    if (k != st.begin()) {
        res -= abs(prev(k)->second - v);
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
    }
    for (int i = 0; i < m; i++) {
        cin >> q[i].a >> q[i].b;
        q[i].c = i;
        q[i].init();
    }
    sort(q, q + m);
    int s = 1, e = 0;
    for (int i = 0; i < m; i++) {
        while (s > q[i].a)pluses(--s);
        while (e < q[i].b)pluses(++e);
        while (s < q[i].a)minuses(s++);
        while (e > q[i].b)minuses(e--);
        solve[q[i].c] = res;
    }
    for (int i = 0; i < m; i++)
        cout << solve[i] << '\n';
}
```

하지만 위 코드 및 Segment Tree를 사용하는 풀이는 느려서 시간초과를 받게 됩니다.
조금 더 생각을 해 봅시다. 만약 제거하는 연산만 존재 한다고 하면 더 빠르게 풀 수 있을까요?

지금 보고 있는 구간 $[s,e]$에 있는 마트료시카의 정렬된 $p_i$에 대해 왼쪽과 오른쪽의 $index$를 전처리합니다. 그 후 $i$번째 마트료시카를 지우면 왼쪽의 오른쪽 $index$를 $i$의 오른쪽 $index$로 바꿔주고, 오른쪽의 왼쪽 $index$를 $i$의 왼쪽 $index$로 바꿔 주면 됩니다.

이 연산은 $O(1)$의 시간 복잡도로 해결됩니다. 이 관찰을 어떻게 사용할 수 있는지 봅시다

 쿼리를 $(L,R)$ 라고 하면, $\lfloor {L \over \sqrt N} \rfloor$가 같은 원소끼리 묶어 봅시다. 편의상 쿼리 $(L,R)$가 들어있는 묶음을 $\lfloor {L \over \sqrt N} \rfloor$번 묶음이라고 합시다.

이제 $i$번 묶음을 처리해 봅시다.

묶음의 수는 많아야 $\lceil {N \over \sqrt N} \rceil$개 이므로 $[i * \sqrt N, N]$까지의 마트료시카를 미리 전처리해 줍니다.

그 후 $i$번 묶음에 들어있는 원소들을 $R$에 대해 내림차 순으로 정렬해 줍니다. 
 $i$번 묶음에 들어있는 쿼리를 순서대로 $(L_1,R_1), (L_2,R_2), ... , (L_{k-1},R_{k-1}), (L_k,R_k)$라고 합시다.

이제 쿼리를 순서대로 처리합니다. $j$번 쿼리를 처리할 때를 보면 포인터가 $[i * \sqrt N, R_j]$가 될 때까지 오른쪽 포인터를 감소시키면서 마트료시카를 하나씩 뺍니다.
그 후 왼쪽 포인터를 증가시키면서 마트료시카를 하나씩 빼 포인터를 $[L_j,R_j]$로 만들어 답을 취합니다.

그 뒤 왼쪽 포인터의 증가를 롤백시켜 다시 포인터를 $[i * \sqrt N, R_j]$로 만듭니다.
 묶음에서의 오른쪽 포인터의 감소는 $R_1 - R_k = O(N)$이고 쿼리당 왼쪽 포인터의 증가는 최대 $max_{1<=j<k}|L_j - L_{j+1}|$ = $O(\sqrt N)$이므로, 묶음에서의 왼쪽 포인터의 증가는 $O(k \sqrt N)$입니다. 바뀌는 값이 $O(1)$개 이므로 한 묶음을 처리할 때 $O(N+k\sqrt N)$의 시간복잡도가 걸리고 $k$의 합은 $Q$이기 때문에 전체 $O((N+Q)\sqrt N)$에 문제를 해결할 수 있습니다.

롤백은 바뀌는 값을 스택에 저장한 뒤 스택에서 값을 빼면서 값을 이전 값으로 바꿔주면 됩니다.
```c++
const int inf = 100'101;
int arr[inf];
int arr2[inf];
ll solve[inf * 5];
const int sd = 300;

struct query {
    int a, b, c;
    bool operator<(const query &t) const {
        return b > t.b;
    }
};

vector<query> q[inf * 5 / sd + 10];
ll res;
int pv[inf], nx[inf];
vector<int> ud;

int abss(int a, int b) {
    if (!a || !b)return 0;
    return abs(a - b);
}

void undo() {
    int w = ud.back();
    ud.pop_back();
    res -= -abss(pv[w], w) - abss(nx[w], w) + abss(pv[w], nx[w]);
    nx[pv[w]] = w;
    pv[nx[w]] = w;
}

void minuses(int w) {
    ud.push_back(w);
    res += -abss(pv[w], w) - abss(nx[w], w) + abss(pv[w], nx[w]);
    nx[pv[w]] = nx[w];
    pv[nx[w]] = pv[w];
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n, m, a, b, c;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
        arr2[arr[i]] = i;
    }
    for (int i = 0; i < m; i++) {
        cin >> a >> b;
        q[a / sd].push_back({a, b, i});
    }
    for (int i = 1; i < n; i++) {
        res += abs(arr2[i] - arr2[i + 1]);
    }
    for (int i = 1; i <= n; i++) {
        nx[i] = arr2[arr[i] + 1];
        pv[i] = arr2[arr[i] - 1];
    }
    for (int i = 1; i < n; i += sd) {
        if (i == 1 + sd)i--;
        while (!ud.empty())undo();
        for (int j = max(1, i - sd); j < i; j++)minuses(j);
        ud.clear();
        sort(q[i / sd].begin(), q[i / sd].end());
        int s = i, e = n;
        for (auto j: q[i / sd]) {
            while (e > j.b)minuses(e--);
            while (s < j.a)minuses(s++);
            solve[j.c] = res;
            while (s > i)undo(), s--;
        }
    }
    for (int i = 0; i < m; i++)
        cout << solve[i] << '\n';
}
```
### 연습문제
[Petrozavodsk Programming Camp Summer 2021 Day 7: Moscow IPT Contest D번](https://www.acmicpc.net/problem/23162) (본문에서 설명한 문제입니다.)

[Seoul Nationalwide Internet Competition 2021 A번](https://www.acmicpc.net/problem/23238)

[Chef and Graph Queries](https://www.codechef.com/problems/GERALD07)


# 마무리
이번 글에서는 Mo's Algorithm에서 롤백을 이용하여 삽입 혹은 제거 연산 중 하나를 배제하는 테크닉에 대해 서술하였습니다. 다음 글에서는 단일 업데이트 연산이 있을 때 구간 쿼리를 Mo's Algorithm을 사용하여 Naive보다 빠르게 해결하는 방법에 대해 소개하겠습니다.