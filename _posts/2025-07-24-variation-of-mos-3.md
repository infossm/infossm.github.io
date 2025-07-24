---
layout: post
title: Variation of Mo's Algorithm 3
date: 2025-07-24 13:50
author: jthis
tags:
  - algorithm
---

안녕하세요 **jthis** 입니다.

이 글에서는 [**Variation of Mo's Algorithm 1**](https://infossm.github.io/blog/2023/02/17/Variationn-of-mos-1/)과  [**Variation of Mo's Algorithm 2**](https://infossm.github.io/blog/2023/06/23/Variationn-of-mos-2/)에 이어, 또 하나의 **Mo’s Algorithm**의 변형에 대해 소개하겠습니다.

# Online Mo's Algorithm

**Mo’s Algorithm**은 오프라인 쿼리에서 $(L, R)$ 포인터만 최소한으로 이동시켜 구간 쿼리를 빠르게 처리하는 기법이지만, 쿼리 순서를 바꿀 수 없는 **온라인** 문제에 그대로 적용하기 어렵습니다.

예를 들어, [BOJ 14897](https://www.acmicpc.net/problem/14897)를 생각해 봅시다. 이 문제에서는 다음 연산을 처리해야 합니다.

- $l$ $r$: $l$번째 수부터 $r$번째 수 중에서 서로 다른 수의 개수를 세고 출력한다.

이 문제는 **Mo's Algorithm**을 사용하여 쉽게 문제를 해결할 수 있습니다.

```c++
void pluses(int v){  
    if(!cnt[v]) res++;  
    cnt[v]++;  
}  
void minuses(int v){  
    cnt[v]--;  
    if(!cnt[v]) res--;  
}  
int main(){  
    int s=1,e=0;  
    for(int i=0;i<q;i++){  
        while(s>q[i].l)pluses(arr[--s]);  
        while(e<q[i].r)pluses(arr[++e]);  
        while(s<q[i].l)minuses(arr[s++]);  
        while(e>q[i].r)minuses(arr[e--]);  
        solve[q[i].c]=res;  
    }  
    for(int i=0;i<q;i++)  
        printf("%d\n",solve[i]);  
}
```

---
해당 문제와 달리, [BOJ 14898](https://www.acmicpc.net/problem/14898)번은 쿼리가 **온라인**으로 주어져 **순서를 바꿀 수 없습니다**. 이 상황에서 **Mo’s Algorithm**을 어떻게 적용할 수 있을지 생각해봅시다.

**Mo’s Algorithm**은 쿼리의 시작 위치를 기준으로 수열을 $O\left(\frac{N}{B}\right)$개의 구간(**bucket**)으로 나누어 처리합니다. 각 쿼리는 $[L/B, R/B]$를 기준으로 정렬되며, 우리는 이 정렬 구조를 **미리 고정**합니다. 그런 다음, 가능한 모든 $[L/B, R/B]$ 조합에 대한 상태를 **전처리**해두면, 쿼리 $(L, R)$에 대해 미리 계산된 $[L/B + 1, R/B - 1]$ 구간의 결과를 사용하고, 남은 양쪽 포인터를 $O(2B)$번 이동시켜 답을 구할 수 있습니다.

하지만 이 방식에는 다음과 같은 문제가 있습니다. 각 **bucket** $[i, j]$마다 구간 상태를 저장해야 하므로, 하나의 **bucket** 마다 $O(M)$ 크기의 정보를 유지해야 합니다. 따라서 총 메모리 사용량은 $O\left(\frac{N^2}{B^2} \times M\right)$이 됩니다.

이를 줄이기 위해 $B = O(N^{2/3})$으로 설정하면, 메모리는 $O(N^{2/3} \times M)$, 시간 복잡도는 $O((Q + N) \times N^{2/3})$이 됩니다. 하지만 현재 문제에서 $N = 10^6$이므로, 이 방식은 **시간 초과**가 발생합니다.

```c++
const int inf = 100'010;
int arr[inf];
vector<int> pk;
const int sq = 4505;

struct bucket {
    int cnt[inf];
    int res;
    void add(int x) {
        x = arr[x];
        if (cnt[x] == 0)res++;
        cnt[x]++;
    }
    void rem(int x) {
        x = arr[x];
        cnt[x]--;
        if (cnt[x] == 0)res--;
    }
} ans[inf / sq + 2][inf / sq + 2];

bucket globalCnt;

void init(int n) {
    for (int i = 1; i < n / sq; i++) {
        for (int j = i; j < n / sq; j++) {
            for (int k = i * sq; k < (j + 1) * sq; k++)ans[i][j].add(k);
        }
    }
}

int solveSmall(int l, int r) {
    for (int j = l; j <= r; j++) {
        globalCnt.add(j);
    }
    int res = globalCnt.res;
    for (int j = l; j <= r; j++) {
        globalCnt.rem(j);
    }
    return res;
}

int solveBig(int l, int r) {
    bucket &res = ans[l / sq + 1][r / sq - 1];
    int s = (l / sq + 1) * sq;
    int e = r / sq * sq - 1;
    int S = s, E = e;
    while (s > l)res.add(--s);
    while (e < r) res.add(++e);
    int ret = res.res;
    while (s < S) res.rem(s++);
    while (e > E) res.rem(e--);
    return ret;
}

int main() {
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
    }
    init(n);
    int m;
    cin >> m;
    for (int i = 0; i < m; i++) {
        int l, r;
        cin >> l >> r;
        if (l / sq == r / sq) {
            cout << (solveSmall(l, r)) << '\n';
        } else {
            cout << (solveBig(l, r)) << '\n';
        }
    }
}
```

---
위 알고리즘의 **Bottleneck**은 **버킷이 저장해야 하는 메모리 크기**에 있습니다. 이를 개선하기 위해, **서로 다른 수의 개수를 구하는 문제**를 다른 방식으로 접근해봅시다.

각 원소 $A_i$에 대해 다음과 같이 정의합니다:

- **suffix<sub>i</sub>**: $A_i = A_{\text{suffix}_i}$이며 $i < \text{suffix}_i$인 가장 작은 인덱스
- **prefix<sub>i</sub>**: $A_i = A_{\text{prefix}_i}$이며 $\text{prefix}_i < i$인 가장 큰 인덱스

이 정보를 미리 구해두면, 어떤 원소를 구간에 추가할 때 **해당 원소의 prefix나 suffix가 이미 구간에 포함되어 있는지** 확인하고, **포함되어 있지 않을 경우에만** 서로 다른 수의 개수를 증가시키면 됩니다.

이러한 방식으로 구현하면, 각 버킷은 $O(1)$ 크기의 상태만 유지하면 되므로, 버킷 크기를 $O(\sqrt{N})$으로 하여 전체 시간복잡도 $O((N + Q)\sqrt{N})$에 문제를 해결할 수 있습니다.
```c++
const int inf = 1'000'101;
int arr[inf];
int pre[inf];
int preIdx[inf], nxtIdx[inf];
vector<int> pk;
const int sq = 1400;

int ans[inf / sq + 10][inf / sq + 10];

void init(int n) {
    for (int i = sq; i <= n; i += sq) {
        int s = i, e = i - 1;
        int res = 0;
        for (int j = i; j <= n; j++) {
            ++e;
            if (preIdx[e] < s)res++;
            ans[i / sq][j / sq] = res;
        }
    }
}

int solveSmall(int l, int r) {
    int s = l, e = l - 1;
    int res = 0;
    for (int j = l; j <= r; j++) {
        if (preIdx[++e] < s)res++;
    }
    return res;
}

int solveBig(int l, int r) {
    int res = ans[l / sq + 1][r / sq - 1];
    int s = (l / sq + 1) * sq;
    int e = r / sq * sq - 1;
    while (s > l) 
        if (e < nxtIdx[--s])res++;
    while (e < r) 
        if (preIdx[++e] < s)res++;
    return res;
}

int main() {
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
        pk.push_back(arr[i]);
    }
    sort(pk.begin(), pk.end());
    pk.erase(unique(pk.begin(), pk.end()), pk.end());
    for (int i = 1; i <= n; i++) {
        arr[i] = lower_bound(pk.begin(), pk.end(), arr[i]) - pk.begin() + 1;
        int &x = pre[arr[i]];
        preIdx[i] = x;
        nxtIdx[i] = n + 1;
        nxtIdx[x] = i;
        x = i;
    }

    init(n);
    int m;
    cin >> m;
    int Q = 0;
    for (int i = 0; i < m; i++) {
        int x, r;

        cin >> x >> r;
        int l = x + Q;
        if (l / sq == r / sq) {
            cout << (Q = solveSmall(l, r)) << '\n';
        } else {
            cout << (Q = solveBig(l, r)) << '\n';
        }
    }
}
```

---
이제 **최빈값 쿼리** 문제를 살펴봅시다. 이 문제는 상태 크기가 $O(M)$이기 때문에, 위 방식으로는 $O(QN^{2/3})$ 시간에 해결 가능합니다. 더 깊게 생각해보면, 각 쿼리에서 포인터는 많아야 $O(2B)$만큼만 이동하며, 실질적으로 **메모리에 접근하는 횟수**는 $O(2B)$에 불과합니다.

따라서 접근하는 **위치 정보만 전처리**하여 저장해두면, 각 쿼리마다 $O(2B)$ 크기의 정보만 필요하게 됩니다. 이때 $B = O(\sqrt{N})$으로 설정하면 전체 시간복잡도는 $O(N\sqrt{N})$이 됩니다.

이를 일반화하면, **Mo’s Algorithm**이 적용 가능한 대부분의 문제는 포인터를 $O(2B)$만큼만 이동합니다. 이때 사용하는 자료구조에 따라 실제로 접근하는 메모리의 수는 다음과 같습니다.

* **Static Array**를 사용할 경우: $O(2B)$
* **Segment Tree**를 사용할 경우: $O(2B \log N)$
* **K-Dimensional Segment Tree**를 사용할 경우: $O(2B\log^kN)$
* **Wavelet Tree**를 사용할 경우: $O(2B \log A)$
* **Van Emde Boas Tree**를 사용할 경우: $O(2B \log \log A)$

모든 경우에 대해 접근할 위치들을 **전처리**해두면 효율적으로 문제를 해결할 수 있습니다.

```c++
const int inf = 100'010;  
const int sq = 700;  
  
int arr[inf];  
bool use[inf];  
int useIdx[inf];  
vector<array<int, 2>> sv;  
  
struct bucket {  
    int res;  
    int cnt[2 * sq + 3];  
    int nxt[sq + 3], prv[sq + 3];  
    int s, e;  
  
    void addR(int idx) {  
        int id = nxt[idx - e];  
        if (!use[id]) {  
            sv.push_back({id, cnt[id]});  
            use[id] = true;  
        }  
        res = max(res, ++cnt[id]);  
    }  
  
    void addL(int idx) {  
        int id = prv[s - idx];  
        if (!use[id]) {  
            sv.push_back({id, cnt[id]});  
            use[id] = true;  
        }  
        res = max(res, ++cnt[id]);  
    }  
  
    void clear() {  
        memset(cnt, 0, sizeof(cnt));  
        memset(nxt, 0, sizeof(nxt));  
        memset(prv, 0, sizeof(prv));  
        res = 0;  
    }  
  
    void rollback() {  
        while (!sv.empty()) {  
            cnt[sv.back()[0]] = sv.back()[1];  
            use[sv.back()[0]] = false;  
            sv.pop_back();  
        }  
    }  
} ans[inf / sq + 2][inf / sq + 2];  
  
struct globalBucketStruct {  
    int cnt[inf];  
    int ans = 0;  
  
    void clear() {  
        memset(cnt, 0, sizeof(cnt));  
        ans = 0;  
    }  
  
    void add(int x) {  
        ans = max(ans, ++cnt[x]);  
    }  
  
    void rem(int x) {  
        ans = 0;  
        cnt[x] = 0;  
    }  
} globalBucket;  
  
void makeBucket(int s, int n) {  
    int i = s * sq;  
    int ns = s;  
    for (int j = i; j <= n; j++) {  
        globalBucket.add(arr[j]);  
        int ne = (j + 1) / sq - 1;  
        if (i < j && (j + 1) % sq == 0) {  
            ans[ns][ne].res = globalBucket.ans;  
  
            int pv = 0;  
  
            for (int k = 1; k <= sq; k++) {  
                int now = arr[i - k];  
                if (!useIdx[now]) {  
                    useIdx[now] = ++pv;  
                    ans[ns][ne].cnt[pv] = globalBucket.cnt[now];  
                }  
                ans[ns][ne].prv[k] = useIdx[now];  
            }  
            for (int k = 1; j + k <= n && k <= sq; k++) {  
                int now = arr[j + k];  
                if (!useIdx[now]) {  
                    useIdx[now] = ++pv;  
                    ans[ns][ne].cnt[pv] = globalBucket.cnt[now];  
                }  
                ans[ns][ne].nxt[k] = useIdx[now];  
            }  
            for (int k = 1; k <= sq; k++) useIdx[arr[i - k]] = 0;  
            for (int k = 1; j + k <= n; k++) useIdx[arr[j + k]] = 0;  
        }  
    }  
    globalBucket.clear();  
}  
  
void init(int n) {  
    for (int i = 1; i < n / sq; i++) {  
        for (int j = i; j < n / sq; j++) {  
            ans[i][j].s = i * sq;  
            ans[i][j].e = j * sq + sq - 1;  
            ans[i][j].clear();  
        }  
    }  
    for (int i = sq; i + sq <= n; i += sq) makeBucket(i / sq, n);  
}  
  
int solveSmall(int l, int r) {  
    for (int j = l; j <= r; j++) globalBucket.add(arr[j]);  
    int res = globalBucket.ans;  
    for (int j = l; j <= r; j++) globalBucket.rem(arr[j]);  
    return res;  
}  
  
int solveBig(int l, int r) {  
    bucket &res = ans[l / sq + 1][r / sq - 1];  
    int s = (l / sq + 1) * sq;  
    int e = r / sq * sq - 1;  
    int rollbackRes = res.res;  
    while (s > l) res.addL(--s);  
    while (e < r) res.addR(++e);  
    int ret = res.res;  
    res.rollback();  
    res.res = rollbackRes;  
    return ret;  
}  
  
int main() {  
    int n;  
    cin >> n;  
    for (int i = 1; i <= n; i++) cin >> arr[i];  
    globalBucket.clear();  
    init(n);  
    int m;  
    cin >> m;  
    for (int i = 0; i < m; i++) {  
        int l, r;  
        cin >> l >> r;  
        if (l / sq + 1 >= r / sq) {  
            cout << (solveSmall(l, r)) << '\n';  
        } else {  
            cout << (solveBig(l, r)) << '\n';  
        }  
    }  
}
```

---
다음과 같은 규칙에 따라 **이진 트리**를 구성하는 문제를 생각해봅시다.

* 길이 $N$인 수열 $A$가 주어집니다.
* 이 수열로 이진 트리를 만드는 규칙은 다음과 같습니다:

  1. 현재 서브트리의 깊이가 **짝수**일 경우:
     구간 내 **최빈값 중 가장 왼쪽에 위치한 값**을 선택합니다.
  2. 현재 서브트리의 깊이가 **홀수**일 경우:
     구간 내 **최빈값 중 가장 오른쪽에 위치한 값**을 선택합니다.
  3. 선택한 값을 **루트**로 하여, 해당 위치를 기준으로 수열을 좌우로 나눈 뒤,
     각각에 대해 위와 같은 과정을 재귀적으로 수행합니다.
  4. 재귀적으로 수행한 후, 좌우 서브트리의 루트를 현재 서브트리의 루트의 자식으로 연결합니다.

이 과정을 통해 만들어지는 **이진 트리의 구조**를 출력하는 것이 목표입니다.

해당 문제는 **두 가지** 풀이 방법이 가능합니다.

1. **분할 정복**
	구간에서 최빈값을 구한 뒤, 더 큰 쪽을 먼저 처리합니다. 부모가 사용한 table을 재사용하여 계산에 활용하고, 리프까지 내려간 뒤 아직 처리되지 않은 작은 쪽 구간도 같은 방식으로 재귀적으로 처리합니다. 이 방식은 **Heavy-Light Decomposition**의 증명 방식과 유사하게 시간복잡도를 증명할 수 있으며, 시간 복잡도는 $O(N \log N)$입니다.
	
2. **Online Mo’s Algorithm 활용**
	이 문제의 핵심은 구간 최빈값을 빠르게 구하는 것입니다. 이를 위해 위에서 설명한 **Online Mo's Algorithm**을 활용할 수 있습니다. 시간복잡도는 $O(N\sqrt{N})$ 혹은 $O(N\sqrt{N}\log N)$ 혹은 $O(N \sqrt{N} \log^2 N)$에 문제를 해결 할 수 있습니다.
	
---
이 알고리즘은 **Incremental + Rollback** 혹은 **Decremental + Rollback**으로 치환할 수 있으며, **update** 연산도 응용할 수 있습니다.

## 연습문제

[**서로 다른 수와 쿼리 2**](https://www.acmicpc.net/problem/14898)

[**수열과 쿼리 6**](https://www.acmicpc.net/problem/13548)

[**Matryoshka Dolls**](https://www.acmicpc.net/problem/23162)

[**Machine Learning**](https://codeforces.com/contest/940/problem/F)

## 참고자료

[**Online Mo's with Update**](https://github.com/ShahjalalShohag/code-library/blob/main/Data%20Structures/MOs%20Online.cpp)
