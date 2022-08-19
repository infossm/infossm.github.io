---
layout: post
title: "O(N) Precomputation, O(1) RMQ (Farach-Colton and Bender Algorithm)"
date: 2022-08-19 00:00:00
author: JooDdae
tags: [algorithm, data-structure]

---

이 글은 Sparse Table을 이용한 $O(N \log N)$ 전처리, $O(1)$ LCA 쿼리([소멤 글 링크](http://www.secmem.org/blog/2019/03/27/fast-LCA-with-sparsetable/))와 sqrt decomposition를 이용한 $O(N)$ 전처리, $O(\sqrt N)$ RMQ([cp-algorithms 링크](https://cp-algorithms.com/data_structures/sqrt_decomposition.html#description))를 선행 지식으로 가지고 있다면 더 쉽게 이해할 수 있습니다.

## $O(N)$ 전처리, $O(1)$ LCA 쿼리
우리는 LCA 쿼리를 Euler Tour 테크닉을 통해 RMQ로 변환시킬 수 있습니다. 이는 위 Sparse Table을 이용한 $O(1)$ LCA 쿼리에서도 사용하는 방법이기 때문에 위 링크된 소멤 글에 자세히 설명되어 있으므로 이 글에서는 설명을 생략하겠습니다. 이 테크닉을 이용해 주어진 트리에서 만든 배열을 $A$라고 하고, 이 배열의 길이를 $N$이라고 합시다.

$A$를 sqrt decomposition을 하듯 $K = 0.5 \log N$ 의 크기의 블럭으로 쪼개줍니다. sqrt decomposition 에서는 블럭 사이의 RMQ를 쿼리마다 $O(\sqrt N)$으로 처리했었는데, 이 알고리즘에서는 블럭의 RMQ를 $O(1)$에 처리하기 위해 Sparse Table을 만듭니다 (Sparse Table을 이용한 $O(1)$ RMQ 또한 위 소멤 글 링크에 설명되어 있습니다). 이때의 시간복잡도는 $O(\frac{N}{K} \log (\frac{N}{K}))$인데,

$\frac{N}{K} \log (\frac{N}{K}) = \frac{2N}{\log(N)} \log (\frac{2N}{\log(N)}) \leq \frac{2N}{\log(N)} \log(2N) = \frac{2N}{\log(N)} (1 + \log(N)) = \frac{2N}{\log(N)} + 2N = O(N)$

 이므로 $O(N)$의 시간복잡도를 가진 전처리로 블럭의 RMQ를 $O(1)$에 해결할 수 있습니다. 하지만 모든 쿼리가 블럭의 크기에 맞게 주어지지 않기 때문에 각 블럭 내부의 RMQ를 따로 전처리해 줘야 $O(1)$로 쿼리를 처리할 수 있습니다.

Euler Tour 테크닉을 통해 만들어진 배열 $A$의 인접한 두 원소의 level의 차이는 1입니다. 우리는 이를 이용해 각 블럭을 인접한 두 원소의 차로 가능한 $-1$과 $+1$을 뜻하는 $0$과 $1$로만 이루어진 길이 $K-1$의 배열로 표현할 수 있습니다.
![http://www.secmem.org/blog/2019/03/27/fast-LCA-with-sparsetable/](http://www.secmem.org/assets/images/fast-LCA/LCA1.png)
위 그림을 예시로 들어보겠습니다. 편의상 블럭의 크기를 5로 설정한다고 했을 때, $A$가 길이 15의 배열이므로 $[1, 5]$, $[6, 10]$, $[11, 15]$ 구간을 뜻하는 3개의 블럭이 나옵니다. 첫 번째 블럭 내부의 인접한 두 원소의 level 차이를 살펴보면 $+1$, $+1$, $-1$, $+1$ 이므로 $[1, 1, 0, 1]$로 표현 할 수 있고, 마찬가지로 두 번째, 세 번째 블럭을 $[1, 0, 0, 1]$과 $[1, 0, 0, 0]$으로 표현할 수 있습니다.

이렇게 표현될 수 있는 서로 다른 배열의 개수는 $2^{K-1}$개 이고,

$2^{K-1} = 2^{0.5 \log(N)-1} = 0.5(2^{\log(N)})^{0.5} = 0.5 \sqrt N = O(\sqrt N)$

이므로 만들어질 수 있는 $O(\sqrt N)$개의 배열을 모두 비트마스크를 이용해 하나의 수로 나타내어 $\text{block}[\text{mask}][l][r]$ 배열을 채워준다면 $O(\sqrt{N} K^2) = O(\sqrt{N} \log^2(N)) = O(N)$에 만들어질 수 있는 모든 블럭 내부의 RMQ를 미리 계산해줄 수 있습니다.

이제 $O(N)$으로 블럭의 RMQ 전처리와 블럭 내부의 RMQ 전처리를 끝냈으므로 이제 LCA를 $O(1)$에 구할 수 있습니다. 아래는 위 과정을 모두 담은 코드입니다. 0-base 로 구현되어 있음을 주의해야 합니다.
```cpp
int n;
vector<vector<int>> v;

int k, block_cnt;
vector<int> lev, in, euler;

vector<int> log_2, block_mask;
vector<vector<int>> sp;
vector<vector<vector<int>>> in_block;

void dfs(int u, int p) {
    in[u] = euler.size(), euler.push_back(u);
    for(auto x : v[u]) if(x != p) {
        lev[x] = lev[u]+1, dfs(x, u);
        euler.push_back(u);
    }
}

int get_min(int X, int Y) {
    return lev[euler[X]] < lev[euler[Y]] ? X : Y;
}

void build(int root) {
    lev.resize(n, 0), in.resize(n, 0);
    dfs(root, -1);

    int N = euler.size();

    log_2.resize(N+1, 0);
    for(int i=2;i<=N;i++) log_2[i] = log_2[i/2]+1;

    k = max(1, log_2[N]/2);
    block_cnt = N/k + !!(N%k);

    sp.resize(block_cnt, vector<int>(log_2[block_cnt]+1));
    for(int i=0;i<N;i+=k) {
        auto &mn = sp[i/k][0];
        mn = i;
        for(int j=1;j<k && i+j<N;j++) mn = get_min(i+j, mn);
    }

    for(int j=1;j<=log_2[block_cnt];j++) {
        for(int i=0;i<block_cnt;i++) if(i+(1<<j)-1 < block_cnt) {
            sp[i][j] = get_min(sp[i][j-1], sp[i+(1<<j-1)][j-1]);
        }
    }


    block_mask.resize(block_cnt);
    in_block.resize(1 << k-1);

    for(int i=0;i<N;i+=k) {
        auto &mask = block_mask[i/k];
        for(int j=1;j<k;j++) mask = (mask << 1) | (i+j < N && lev[euler[i+j-1]] < lev[euler[i+j]]);

        if(!in_block[mask].empty()) continue;
        in_block[mask].resize(k, vector<int>(k, 0));

        for(int l=0;l<k;l++) {
            int s = 0, mn = 0, mnid = l;
            in_block[mask][l][l] = l;

            for(int r=l+1;r<k;r++) {
                s += ((mask >> k-1-r) & 1) ? 1 : -1;
                if(s < mn) mn = s, mnid = r;
                in_block[mask][l][r] = mnid;
            }
        }
    }
}

int lca_in_block(int block, int l, int r) {
    return in_block[block_mask[block]][l][r] + block * k;
}

int lca(int a, int b) {
    int l = in[a], r = in[b];
    if(l > r) swap(l, r);

    int bl = l/k, br = r/k;
    if(bl == br) return euler[lca_in_block(bl, l%k, r%k)];

    int ans = get_min(lca_in_block(bl, l%k, k-1), lca_in_block(br, 0, r%k));
    if(bl+1 < br) {
        int b = log_2[br-bl-1];
        ans = get_min(ans, get_min(sp[bl+1][b], sp[br-(1<<b)][b]));
    }
    return euler[ans];
}
```
위 코드를 사용해서 [LCA 2](https://www.acmicpc.net/problem/11438) 문제를 푼 코드가 있는 [링크](https://www.acmicpc.net/source/share/855e69cc5fe042b18a78c5af43501252)입니다.

## $O(N)$ 전처리, $O(1)$ RMQ

RMQ 문제를 LCA로 환원하려면 Carteian Tree를 알아야 합니다. ([위키피디아](https://en.wikipedia.org/wiki/Cartesian_tree) 링크)
배열로 만든 Carteian Tree에서의 LCA는 배열의 RMQ와 동일하기에 주어진 배열로 Cartesian Tree를 만든 다음 LCA를 구할 수 있다면 RMQ를 처리할 수 있고, Carteian Tree는 stack을 이용해 $O(N)$으로 만들 수 있기에 위 알고리즘과 함께 사용한다면 총 $O(N)$ 전처리, $O(1)$ RMQ를 할 수 있습니다.

아래는 배열을 Carteian Tree로 $O(N)$에 만들어주는 코드입니다. 위와 마찬가지로 0-base로 구현되어 있습니다.
```cpp
vector<int> build_cartesian_tree(const vector<int> A) {
    int n = A.size();
    vector<int> parent(n, -1);
    stack<int> st;

    for(int i=0;i<n;i++) {
        int last = -1;
        while(!st.empty() && A[st.top()] >= A[i]){
            last = st.top(), st.pop();
        }
        if(!st.empty()) parent[i] = st.top();
        if(last != -1) parent[last] = i;
        st.push(i);
    }
    return parent;
}
```
두 코드를 합해서 RMQ를 구하는 문제([최솟값](https://www.acmicpc.net/problem/10868))을 해결한 코드가 있는 [링크](http://boj.kr/c6140163cd774e75926222604d75e026) 또한 올리겠습니다.

## 참고 자료
<ul>
	<li>https://cp-algorithms.com/graph/lca_farachcoltonbender.html</li>
	<li>https://cp-algorithms.com/graph/rmq_linear.html</li>
</ul>
