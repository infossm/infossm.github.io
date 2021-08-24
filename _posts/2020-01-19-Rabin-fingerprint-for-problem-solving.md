---
layout: post
title: Rabin fingerprint for problem solving
date: 2020-01-19 10:00
author: rdd6584
tags: [algorithm, mathematics]
---

# Rabin fingerprint란?

Rabin fingerprint는 길이 $n$의 배열 $m$, 임의의 값 $\space x$에 대해 아래와 같은 수식을 가지는 일종의 해시 함수입니다.

$f(x)=m_{0}+m_{1}x+\ldots +m_{n-1}x^{n-1}$

주로, 라빈카프 알고리즘에서 사용되기 때문에 해당 알고리즘을 공부하신 분께는 친숙할텐데요. 위 해시함수를 응용하여 문제를 해결하는 방법에 대해서 소개하고자 합니다. 모든 설명에서 $x$가 $max(m_i)$보다 큰 상황을 가정합니다.



### 가장 긴 문자열([링크](https://www.acmicpc.net/problem/3033))

$m_0$ ~ $m_i$를 $g(x,\space 0,\space i) = m_0 + m_1x + ... + m_{i}x^{i}$와 같이

$m_j$ ~ $m_{j+i}$를 $g(x,\space j,\space j+i) = (m_jx^{j} + m_{j+1}x^{j+1} + ... + m_{j+i}x^{j+i})\space/\space x^j$와 같이 표현할 수 있습니다.



여기서, $g(x,\space 0,\space i) == g(x,\space j,\space j+i)$라면, $m_0$ ~ $m_i$와 $m_j$ ~ $m_{j+i}$는 같습니다.

$p_i$를 $g(x,\space 0,\space i)$라고 정의하고 배열 $p$를 미리 만들어둔다면, 두 부분 배열이 일치하는 지 $O(1)$만에 확인할 수 있게 됩니다. 따라서, 같은 해시값을 가지는 부분 문자열을 찾는 것으로 문제를 간단하게 바꿀 수 있습니다.

두 번 이상 등장하는 부분 문자열이 존재함을 1로, 존재하지 않음을 0으로 표현한다면, 부분 문자열의 길이 y에 대해 각 상태에 대한 표현값은 단조감소하게 됩니다. 따라서 문제에 이분탐색을 적용할 수 있습니다. 길이 y에 대해 겹치는 해시값이 존재함은 set과 같은 자료구조를 이용해 해결할 수 있습니다.

이를 구현한 코드입니다.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int BASE = 256; // x에 해당하는 값
const ll MOD[2] = {1000000007, 1000000009}; // 너무 큰 해시값을 관리하기 어려우므로 사용.
ll power[200001][2]; // x의 y제곱을 각 모듈로에 맞게 미리 계산
ll psum[200001][2]; // 해시값의 부분합 배열
int n;
char s[200002];

ll getHash(int i, int j) {
    ll ret[2];
    for (int k = 0; k < 2; k++) {
        ret[k] = (psum[j][k] - psum[i - 1][k] + MOD[k]) % MOD[k];
        ret[k] = ret[k] * power[n - i][k] % MOD[k]; // 적절한 power를 곱해 문자열 등장 위치를 보정한다.
    }

    return ret[0] * MOD[1] + ret[1]; // 두 모듈로에서 얻은 해시값을 조합.
}

int main() {
    scanf("%d", &n);
    scanf("%s", s + 1);

    power[0][0] = power[0][1] = 1;

    for (int i = 0; i < 2; i++)
        for (int j = 1; j <= n; j++) {
            power[j][i] = power[j - 1][i] * BASE % MOD[i];
            psum[j][i] = (psum[j - 1][i] + s[j] * power[j][i]) % MOD[i];
        }

    int l = 1, r = n, mid;
    unordered_set<ll> us;

    while (l <= r) {
        mid = (l + r) / 2;
        us.clear();

        int flag = 0;

        for (int i = 1; i <= n - mid + 1; i++) {
            ll temp = getHash(i, i + mid - 1);
            if (us.find(temp) != us.end()) {
                flag = 1;
                break;
            }
            us.insert(temp);
        }

        if (flag) l = mid + 1;
        else r = mid - 1;
    }
    printf("%d", r);
}
```





### 문자열과 쿼리([링크](https://www.acmicpc.net/problem/13713))

접미사는 생각하기가 까다로우므로, 문자열을 뒤집어서 가장 긴 공통 접두사를 찾는 것으로 합시다. 이는 Z알고리즘으로 문자열의 모든 위치 $i$에 대해 구할 수 있는 것이 알려져 있는데요. Rabin fingerprint 함수를 이용해도 쉽게 구할 수 있습니다. '가장 긴 문자열' 문제와 동일하게 문자열에 대한 해시값 prefix sum 배열 $p$를 미리 만들어봅시다. 그러면 $S_0S_1S_2 ... S_{n-1}$과 $S_iS_{i+1}S_{i+2}...S_{n-1}$에 대해 가장 긴 공통 접두사의 길이는 이분탐색과 배열 $p$를 이용해 쉽게 해결할 수 있습니다. 이를, 각 쿼리마다 구해주는 것으로 문제를 해결할 수 있습니다.



마찬가지로, [찾기](https://www.acmicpc.net/problem/1786)에서 사용되는 단순한 패텅매칭이나 [LCS](https://www.acmicpc.net/problem/9249) 같은 알고리즘까지 비슷한 방법으로 구할 수 있습니다. Problem solving에서 강력한 도구인 해시를 조금 더 응용해볼까요?



임의의 배열에 대한 해시값이 $f$일때, 어떤 원소의 삽입, 삭제, 변경 등의 연산이 적용된 해시값 $f'$을 $O(logN)$에 구할 수 있습니다.

모든 원소 $i$에 대해 $m_ix^i$를 세그먼트 트리에 저장하고 관리해봅시다. 인덱스 $j$에 값 $y$를 삽입하는 연산은 기존의 $j$ ~ $n-1$에 위치한 값들에 $x$를 곱하고 $yx^j$를 더하는 것으로 표현할 수 있습니다. 비슷한 방식으로 삭제, 변경 연산이 가능합니다.

또한, 삽입, 삭제, 변경과 같은 연산이 배열의 끝에서만 일어난다면 prefix sum을 관리함으로써 $O( 1)$만에 계산할 수 있게 됩니다.



위 사실을 이용하여 풀 수 있는 문제를 소개하겠습니다.

### Stack Exterminable Arrays([링크](https://codeforces.com/contest/1240/problem/D))

스택에 원소를 넣는데, 인접한 두 원소의 값이 같으면 두 원소는 사라지게 됩니다. 어떤 배열을 순서대로 스택에 넣었을 때 스택이 비어있는 상태로 끝이 나면 Stack Exterminable하다고 합니다. 이때, 주어진 배열에서 Stack Exterminable한 부분 배열의 개수를 찾는 문제입니다. 부분 배열 $a_i, a_{i+1}, a_{i+2}$, ..., $a_{j}$가 Stack Exterminable 하다는 것은 $a_0, a_1, ..., a_{i-1}$의 결과와 $a_0, a_1, ..., a_j$ 의 결과스택이 일치한다는것과 동치입니다.

결과스택이 일치하는 지 여부는 어떻게 빠르게 판단할까요? 이때 사용되는 것이 해싱입니다. 결과스택은 하나의 수열로 표현될 것이므로 이에 대한 해시값을 각각 구해놓고 일치하는 지 판단하는 식입니다.

이를 구현한 코드입니다.

```cpp
#include <bits/stdc++.h>
#define sz(x) (int)(x).size()
using namespace std;

typedef long long ll;
const int BASE = 300007;
const ll MOD[2] = { 1000000009, 1000000021 };
ll power[300001][2];

unordered_map<ll, int> um;
vector<int> v;

int main() {
    int tc;
    scanf("%d", &tc);

    power[0][0] = power[0][1] = 1;
    for (int i = 0; i < 2; i++)
        for (int j = 1; j <= 300000; j++)
            power[j][i] = power[j - 1][i] * BASE % MOD[i];

    while (tc--) {
        int n;
        scanf("%d", &n);

        um.clear();
        v.clear();

        ll ret[2] = {0};
        ll ans = 0;
        um[0]++;

        for (int i = 0; i < n; i++) {
            int t;
            int szz = sz(v);
            scanf("%d", &t);

            if (!v.empty() && v.back() == t) {
                for (int j = 0; j < 2; j++)
                    ret[j] = (ret[j] - t * power[szz - 1][j] % MOD[j] + MOD[j]) % MOD[j];
                v.pop_back();
            }
            else {
                for (int j = 0; j < 2; j++)
                    ret[j] = (ret[j] + t * power[szz][j]) % MOD[j];
                v.push_back(t);
            }

            ans += um[ret[0] * MOD[1] + ret[1]]++;
        }

        printf("%lld\n", ans);
    }
}
```



### Laminar Family([링크](https://www.acmicpc.net/problem/15294))

$N$개의 정점을 가진 트리에서 $f$개의 경로가 주어집니다. 이 경로들 중 서로 엇갈리는 것이 있는 지 판단하는 문제입니다. 경로들을 각 경로의 길이가 긴 순서대로 정렬하고 이 순서대로 각 경로를 살펴봅시다. 경로들을 순서대로 보면서 각 정점마다 이 정점을 어떤 경로들이 지나갔는 지 기록해봅시다.

지금 보고 있는 경로에 존재하는 모든 정점에서 지났던 경로들의 집합이 전부 일치한다면, 이 경로와 엇갈리는 길이가 더 긴 경로는 존재하지 않습니다. 즉, 각 정점들의 경로 집합이 일치하는 지 판별해야 하며 이 집합을 해시값으로 나타낼 수 있습니다.

Heavy-light decomposition을 이용하면, 경로를 $O(logN)$개의 구간으로 나타낼 수 있으므로 세그먼트 트리에 각 집합의 정보를 담아서 관리하면 됩니다. 여기서 어느 구간에 속한 정점들의 집합에 원소를 하나 추가하는 연산은 lazy propagation으로 해당하는 경로에 대응되는 해시값을 업데이트 해주는 것으로 해결할 수 있습니다



### Cow Patterns([링크](https://www.acmicpc.net/problem/7038))

$N$개의 수를 가지는 배열이 주어지고, 길이 $K$의 등수 패턴이 주어질 때 배열에서 등수 패턴으로 표현되는 부분 배열을 전부 구하는 문제입니다. 패턴에서 같은 수를 가지는 것들끼리 묶어서 그 수가 등장하는 위치의 집합을 해시로 표현해봅시다. $f(x)$를 배열 $x$의 Rabin fingerprint 해시값이라고 해봅시다. 만약 패턴이 $3\space 2\space 1\space 2\space 2\space 3\space 3\space 1$이라면, $p_1  = f([00100001])$, $p_2 = f([01011000)]$, $p_3 = f([10000110])$로 표현할 수 있을겁니다. 여기서 전체 패턴에 해당하는 해시값을 얻어내는 것은 쉽습니다. $1p_1 + 2p_2 + 3p_3$가 되겠지요. 우리는 $p $를 관리함으로써 특정 값을 가지는 모든 수를 다른 값으로 치환하는 것을 쉽게 할 수 있게 되었습니다. 

이제 모든 길이 $K$의 부분 배열을 검사하면서, 각 부분 배열에 해당하는 해시값과 치환된 등수패턴의 해시값을 비교하는 것으로 문제를 해결하시면 됩니다.



이상으로 글을 마치겠습니다.

읽어주셔서 감사합니다.
