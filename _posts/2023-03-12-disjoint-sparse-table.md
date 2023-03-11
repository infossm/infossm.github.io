---
layout: post
title: "Disjoint Sparse Table"
date: 2023-03-12 00:00:00
author: JooDdae
tags: [data-structure]
---

Disjoint Sparse Table은 효율적인 쿼리 처리를 위한 자료구조 중 하나로, 1차원 배열의 range query를 해결하는 데 사용됩니다. 이 자료구조는 $O(N log N)$으로 전처리되며 $O(1)$의 시간복잡도로 range query를 구할 수 있습니다. 

일반적인 Sparse Table로 range query를 처리할 때와 같은 시간복잡도를 가지지만, Sparse Table의 경우 쿼리로 찾은 두 구간의 값을 합칠 때 겹치는 부분이 존재하기 때문에 $x \circ x = x$를 만족하는 연산(ex: $max$, $min$, $gcd$)만 가능하지, Disjoint Sparse Table은 그렇지 않은 연산(ex: $+$, $\times$)도 지원합니다.

Sparse Table을 이용한 방법은 다음 [소프트웨어 멤버십 블로그 글](https://infossm.github.io/blog/2019/03/27/fast-LCA-with-sparsetable/)에 소개되어 있으니, 모르는 내용이라면 참고하면 좋을 듯 합니다.

## $O(N \log N)$ precomputation, $O(\log N)$ query

$2$의 거듭제곱인 길이 $N$의 배열이 있다고 합시다. 만약 배열의 길이가 $2$의 거듭제곱이 아니라면 $2$의 거듭제곱이 될 때까지 배열의 뒤에 $0$을 추가하면 됩니다.

길이 $N$의 배열을 두 개의 같은 크기의 배열로 쪼개고, 쪼개진 부분을 기준으로 prefix와 suffix의 값을 구합니다. 이때, 쪼개진 부분을 기준으로 구하고자 하는 쿼리가 배열과 같이 쪼개졌다면(쿼리 $[L, R]$의 $L$이 쪼개진 왼쪽 배열에 있으며 $R$이 오른쪽 배열에 있다면), 구한 prefix와 suffix의 값을 이용해 답을 $O(1)$에 구할 수 있게 됩니다.
![range sum query 1](https://user-images.githubusercontent.com/51346964/224488090-0c627689-cf0b-4501-8dee-bdba57535177.png)

그러나, 쪼개진 부분을 기준으로 구하고자 하는 쿼리가 배열과 같이 쪼개지지 않으면 답을 구할 수 없습니다(예시로 sum query를 들었기 때문에 가능하다고 생각할 수도 있겠지만, min이나 max query를 생각해봅시다). 

그러므로 우리는 쪼개진 배열을 다시 같은 크기의 두 배열로 쪼개주고 prefix와 suffix의 값을 각각 구해줍니다. 이때, 배열과 함께 쿼리도 같이 쪼개진다면 답을 $O(1)$에 바로 구할 수 있습니다.
![range sum query 2](https://user-images.githubusercontent.com/51346964/224488838-a579a66b-02c6-4063-93b3-b223d99d8dea.png)
이번 단계에서 쿼리가 쪼개지지 않더라도, 위와 같은 과정을 반복하여 배열이 더 이상 쪼개지지 않을 때까지 작업을 수행한다면 언젠가는 쿼리가 쪼개질 것이고, 저장된 값으로 답을 구할 수 있을 것입니다. 
![range sum query 3](https://user-images.githubusercontent.com/51346964/224489314-a48d4d16-4efb-464f-b667-b204cd68b343.png)

아래는 위에서 설명한 내용을 그대로 구현한 코드입니다. 0-base 로 구현되어 있음을 주의해야 합니다.
```cpp
#include <bits/stdc++.h>
using namespace std;

int n, a[1 << 20];
int sp[20][1 << 20];

void build(int lev, int l, int r) {
    if(lev == 0) {
        sp[lev][l] = a[l];
        return;
    }

    int m = (l + r) >> 1;
    sp[lev][m] = a[m], sp[lev][m+1] = a[m+1];
    for(int i=m-1;i>=l;i--) sp[lev][i] = sp[lev][i+1] + a[i];
    for(int i=m+2;i<=r;i++) sp[lev][i] = sp[lev][i-1] + a[i];

    build(lev-1, l, m), build(lev-1, m+1, r);
}

int main(){
    cin.tie(0)->sync_with_stdio(0);
    cin >> n;
    for(int i=0;i<n;i++) cin >> a[i];

    int log = __lg(max(n - 1, 1)) + 1;
    build(log, 0, (1 << log)-1);
}
```
이 코드의 시간복잡도는 $T(N) = 2T(N/2) + O(N) = O(N \log N)$로, 우리가 목표했던 시간복잡도와 일치합니다.

이제 테이블을 만들었으니 이를 이용해 쿼리가 주어졌을 때 답을 효율적으로 구할 방법을 찾아야 합니다. 여기서 우리가 생각할 수 있는 가장 간단한 방법은, 쿼리가 쪼개지는 위치를 찾을 때까지 쪼개진 배열을 타고 내려가는 것입니다. 이때 쪼개지는 횟수는 최대 $\log N$번이므로 $O(\log N)$에 쪼개지는 위치를 찾아낼 수 있고, 앞에서 만들어놓은 테이블을 이용해 답을 $O(1)$에 찾을 수 있습니다. 아래는 위에서 테이블을 만드는 코드와 함께 사용해 구간 쿼리를 $O(\log N)$에 해결할 수 있는 코드입니다.
```cpp
int query(int nl, int nr, int lev, int l, int r) {
    if(lev == 0) return a[l];

    int m = (l + r) >> 1;
    if(nl <= m && m < nr) return sp[lev][nl] + sp[lev][nr];

    if(nr <= m) return query(nl, nr, lev-1, l, m);
    return query(nl, nr, lev-1, m+1, r);
}
```
위의 두 코드를 결합해 range sum query가 필요한 문제를 해결한 코드가 있는 [링크](http://boj.kr/131206e740084bc5b3a504fad8fb4590)입니다.

이제 우리는 쿼리를 $O(\log N)$에 찾을 수 있지만, 이는 목표로 하던 시간복잡도가 아닙니다. 어떻게 해야 이 테이블을 이용해 더 효율적으로 쿼리를 해결할 수 있을까요?

## $O(1)$ Query

쪼개지는 위치를 구하는 시간이 $O(\log N)$이 걸리기 때문에 range query를 해결하는 데 $O(\log N)$시간이 들었습니다. 그렇다면 이를 부분을 $O(1)$로 구하는 방법을 찾는다면 우리는 목표하던 시간복잡도를 얻어낼 수 있을 것입니다.

위에서 $O(\log N)$에 답을 구하는 query의 5번째 줄의 if문은 다음 if문으로 교체해도 같은 [결과](https://www.acmicpc.net/source/share/6d2ed893557647fdb39d90f98ee81005)를 얻을 수 있습니다.
```cpp
if((nl & (1 << lev-1)) != (nr & (1 << lev-1)))
```
우리가 가장 위에서 $N$을 $2$의 거듭제곱이 되도록 뒤에 $0$을 추가한 이유가 여기에 있습니다. $N = 2^K$라고 할 때, L과 R의 상위 $K-lev$개의 비트는 같습니다. 그리고 다음 비트인 $K-(lev-1)$번째 비트가 달라졌을 때 재귀 함수를 호출하지 않고 바로 테이블에서 값을 찾아 반환해주는 것입니다. 즉, 이 코드는 L과 R을 이진수로 나타냈을 때 값이 달라지는 가장 상위 비트를 찾는 코드라고 볼 수 있습니다.

그 위치를 찾는 데 $O(\log N)$의 시간이 걸린 것이고, 사실 이것은 L과 R을 XOR한 값에서 켜져 있는 가장 상위 비트를 찾는 것으로 $O(1)$에 해결할 수 있습니다(자세한 건 코드를 참고해주세요).

이제 우리는 쪼개지는 위치를 $O(1)$에 찾아낼 수 있으므로 range query를 $O(1)$에 해결할 수 있게 되었습니다. 아래는 위 과정을 구현한 코드입니다. 추가로 재귀 함수로 테이블을 만들던 부분을 반복문으로 수정했습니다.
```cpp
int n, a[1 << 20];
int sp[20][1 << 20];

void build() {
    int log = __lg(max(n - 1, 1)) + 1;
    memcpy(sp[0], a, sizeof a);

    int N = 1 << log;

    for(int lev=1;lev<=log;lev++) {
        for(int l=0;l<N;l+= 1 << lev) {
            int r = l + (1 << lev) - 1;

            int m = (l + r) >> 1;
            sp[lev][m] = a[m], sp[lev][m+1] = a[m+1];
            for(int i=m-1;i>=l;i--) sp[lev][i] = min(sp[lev][i+1], a[i]);
            for(int i=m+2;i<=r;i++) sp[lev][i] = min(sp[lev][i-1], a[i]);
        }
    }
}

int query(int l, int r) {
    assert(l <= r);
    if(l == r) return a[l];
    int lev = __lg(l ^ r) + 1;
    return min(sp[lev][l], sp[lev][r]);
}
```
위의 코드를 이용해 range min query가 필요한 문제를 해결한 코드가 있는 [링크](http://boj.kr/d0af6a530705487db0031ebbbdf031fa)입니다.

## Benchmark

다음은 $N = 2^{16}, 2^{20}, 2^{24}$인 1차원 배열에서 $Q = 2^{16}, 2^{20}, 2^{24}$회 랜덤한 range min query를 구했을 때 Sparse Table과 Disjoint Sparse Table의 실행시간을 비교한 표입니다.

Sparse Table / Disjoint Sparse Table Algorithm runtimes

| $Q\,\backslash \, N $ | $2^{16}$                | $2^{20}$               | $2^{24}$                |
| ------------------ |  ---------------- | ----------------- | ----------------- |
| $2^{16}$                 | 0.00080 / 0.01180 | 0.01400 / 0.03800 | 0.50520 / 0.77320 |
| $2^{20}$                 | 0.00020 / 0.00540 | 0.00960 / 0.03680 | 0.20400 / 0.65980 |
| $2^{24}$                 | 0.00060 / 0.00540 | 0.01060 / 0.03700 | 0.20440 / 0.66400 |


Disjoint Sparse Table에서는 가능하지만 Sparse Table에서는 불가능한 연산이 문제로 나오는 경우가 거의 없으며, 두 방법의 시간복잡도가 같더라도 Sparse Table의 연산이 더 간단하여 실행 속도가 더 빠릅니다. 그러므로 일반적으로는 Sparse Table을 사용하고, 특수한 상황에서만 Disjoint Sparse Table을 사용하는 것이 좋습니다.


## 참고 자료
<ul>
	<li>https://codeforces.com/blog/entry/79108</li>
	<li>https://discuss.codechef.com/t/tutorial-disjoint-sparse-table/17404</li>
</ul>
