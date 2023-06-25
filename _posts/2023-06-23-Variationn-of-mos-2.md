---
layout: post
title: "Variation of Mo's Algorithm 2"
date: 2023-06-23 03:50
author: jthis
tags: [algorithm]
---

안녕하세요 jthis 입니다.

이 글에서는 [Variation of Mo's Algorithm 1](https://infossm.github.io/blog/2023/02/17/Variationn-of-mos-1/)에 이어 또다른 Mo’s Algorithm의 variation에 대해 소개하겠습니다.

# Mo's Algorithm with Update
Mo's Algorithm은 쿼리 문제를 효율적으로 해결하는 알고리즘입니다. 하지만 일반적인 Mo's Algorithm은 업데이트 연산을 처리할 수 없어, 해당 제약으로 인해 사용하기 어려운 경우가 있습니다. 이런 상황에서 사용할 수 있는 3D Mo's Algorithm이라고도 불리는 Update Mo's Algorithm에 대해 소개하고자 합니다.

예를 들어, [쿼리문제](https://www.acmicpc.net/problem/18436)를 생각해 봅시다. 이 문제에서는 다음과 같은 연산이 주어집니다:

- 1 i x: $A_i$를 x로 바꾼다.
- 2 l r: $l \le i \le r$에 속하는 모든 $A_i$중에서 짝수의 개수를 출력한다.
- 3 l r: $l \le i \le r$에 속하는 모든 $A_i$중에서 홀수의 개수를 출력한다.

이 문제는 일반적인 세그먼트 트리를 사용하여 $O(N + Q\log N)$에 해결할 수 있습니다. 하지만 Mo's Algorithm을 이용하여 {$\frac{l}{K}, \frac{r}{K}, i$}를 기준으로 정렬하고, 포인터를 움직이면서 쿼리를 처리할 수도 있습니다. 이런 접근 방식은 왜 시간 내에 잘 작동하는지 살펴보겠습니다.

먼저 시간 복잡도를 계산해 봅시다. 
같은 $\frac{L}{N}$이 모여있는 쿼리 그룹을 $L$그룹 같은 $\frac{R}{N}$이 모여있는 쿼리 그룹을 $R$그룹이라고 합시다.

$L$ 포인터는 같은 $L$그룹 내에서 $O(K)$만큼 이동합니다. 인접 그룹으로의 전체 이동은 $O(N)$입니다. 따라서 $L$ 포인터의 시간 복잡도는 $O(QK + N)$입니다.

$R$ 포인터는 같은 $R$그룹 내에서 최대 $K$만큼 이동하며, 같은 $L$그룹 내에서의 전체 이동은 $O(N)$입니다. $L$그룹은 $\frac{N}{K}$개 있으므로 $O(\frac{N^2}{K})$입니다. 따라서 $R$ 포인터의 시간 복잡도는 $O(QK + \frac{N^2}{K})$입니다.

마지막으로 $I$ 포인터를 살펴봅시다. $I$ 포인터의 이동은 같은 $L$그룹 같은 $R$그룹에서 $O(N)$입니다 따라서 $L$그룹수 * $R$그룹 수 * $N$인 $O(\frac{N^3}{K^2})$가 됩니다.

이 알고리즘의 전체 시간 복잡도는 $L,\ R,\ I$ 포인터의 시간 복잡도의 합입니다. 따라서 $O(QK + N + \frac{N^2}{K} + \frac{N^3}{K^2})$의 시간 복잡도가 됩니다. $K$는 $N$보다 작으므로 결국 $O(QK + \frac{N^3}{K^2})$이 됩니다.

$K$를 최적인 $N^{\frac{2}{3}}$으로 설정하면 전체 시간 복잡도는 $O(QN^\frac{2}{3} + N^\frac{5}{3})$가 됩니다. 이렇게 하면 문제를 효율적으로 해결할 수 있습니다.

구현 코드는 다음과 같습니다.
```c++
#define all(x)x.begin(),x.end()
int arr[100'010];
const int bs = 2714; 
struct query {
    int l, r, t, idx, cmd;
    bool operator < (query b) {
        if (l / bs == b.l / bs) {
            if (r / bs == b.r / bs) return t < b.t;
            return r < b.r;
        }
        return l < b.l;
    }
}qq[100'010];
struct update {
    int idx, val;
}; 
vector<update> chg;
int odd, even;

void upd(int x, int y) {
    if (x & 1) odd += y;
    else even += y;
}

int ans[100'010];
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> arr[i];
    int q, a, b, c;
    cin >> q;
    int qcnt = 0;
    for (int i = 0; i < q; i++) {
        cin >> a >> b >> c;
        if (a == 1) {
            chg.push_back({b,arr[b]});
            arr[b] = c;
        } else {
            qq[qcnt] = {b,c,(int) chg.size() - 1,qcnt,a};
            qcnt++;
        }
    }
    sort(qq, qq + qcnt);
    int s = 1, e = 0, now = (int) chg.size() - 1;
    for (int i = 0; i < qcnt; i++) {
        while (now < qq[i].t) {
            now++;
            update & A = chg[now];
            if (s <= A.idx && A.idx <= e) {
                upd(arr[A.idx], -1);
                upd(A.val, 1);
            }
            swap(arr[A.idx], A.val);
        }
        while (qq[i].t < now) {
            update & A = chg[now];
            if (s <= A.idx && A.idx <= e) {
                upd(arr[A.idx], -1);
                upd(A.val, 1);
            }
            swap(arr[A.idx], A.val);
            now--;
        }
        while (e < qq[i].r) upd(arr[++e], 1);
        while (qq[i].l < s) upd(arr[--s], 1);
        while (qq[i].r < e) upd(arr[e--], -1);
        while (s < qq[i].l) upd(arr[s++], -1);
        if (qq[i].cmd == 2) ans[qq[i].idx] = even;
        else ans[qq[i].idx] = odd;
    }
    for (int i = 0; i < qcnt; i++) cout << ans[i] << '\n';
}
```



### 연습문제
[수열과 쿼리 37](https://www.acmicpc.net/problem/18436) 본문에서 설명한 문제입니다.

[F. Machine Learning](https://codeforces.com/contest/940/problem/F)

좌표압축을 하고 update Mo's를 돌리면 됩니다. 개수와 개수의 개수를 관리하면 답은 최대 $O(sqrt(N))$이기 때문에 $O(QN^\frac{2}{3} + N^\frac{5}{3} + QsqrtN)$에 풀 수 있습니다.

코드는 다음과 같습니다.
```c++
#define all(x)x.begin(),x.end()
#define pack(x)sort(all(x));x.erase(unique(all(x)),x.end())
int arr[100'010];
const int bs = 2714;
struct query {
    int l, r, t, idx;
    bool operator < (query b) {
        if (l / bs == b.l / bs) {
            if (r / bs == b.r / bs) return t < b.t;
            return r < b.r;
        }
        return l < b.l;
    }
}qq[100'010];
struct update {
    int idx, val;
};
vector<update> chg;
int cnt[200'010];
int pcnt[200'010];
void upd(int x, int y) {
    pcnt[cnt[x]]--;
    cnt[x] += y;
    pcnt[cnt[x]]++;
}

int ans[100'010];
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);cout.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> sv;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
        sv.push_back(arr[i]);
    }
    int a, b, c;
    int qcnt = 0;
    for (int i = 0; i < q; i++) {
        cin >> a >> b >> c;
        if (a == 2) {
            chg.push_back({b,arr[b]});
            arr[b] = c;
            sv.push_back(c);
        } else {
            qq[qcnt] = {b,c,(int) chg.size() - 1,qcnt};
            qcnt++;
        }
    }
    pack(sv);
    for (int i = 1; i <= n; i++) {
        arr[i] = lower_bound(all(sv),arr[i]) - sv.begin() + 1;
    }
    for(auto &j:chg){
        j.val = lower_bound(all(sv),j.val) - sv.begin() + 1;
    }
    pcnt[0] = 1e9;
    sort(qq, qq + qcnt);
    int s = 1, e = 0, now = (int) chg.size() - 1;
    for (int i = 0; i < qcnt; i++) {
        while (now < qq[i].t) {
            now++;
            update & A = chg[now];
            if (s <= A.idx && A.idx <= e) {
                upd(arr[A.idx], -1);
                upd(A.val, 1);
            }
            swap(arr[A.idx], A.val);
        }
        while (qq[i].t < now) {
            update & A = chg[now];
            if (s <= A.idx && A.idx <= e) {
                upd(arr[A.idx], -1);
                upd(A.val, 1);
            }
            swap(arr[A.idx], A.val);
            now--;
        }
        while (e < qq[i].r) upd(arr[++e], 1);
        while (qq[i].l < s) upd(arr[--s], 1);
        while (qq[i].r < e) upd(arr[e--], -1);
        while (s < qq[i].l) upd(arr[s++], -1);
        for(int j=1;;j++){
            if(!pcnt[j]){
                ans[qq[i].idx]=j;
                break;
            }
        }
    }
    for (int i = 0; i < qcnt; i++) cout << ans[i] << '\n';
}
```

[Primitive Queries](https://www.codechef.com/FEB17/problems/DISTNUM3)
트리에서 update Mo's를 쓰는 문제입니다.

[F. Two Subtrees](https://codeforces.com/contest/1767/problem/F)
업데이트는 없지만 동일한 아이디어로 풀 수 있습니다.

[수열과 쿼리 12](https://www.acmicpc.net/problem/13887)



# 참고자료
[Mo's algorithm and 3D Mo](https://codeforces.com/blog/entry/83630)