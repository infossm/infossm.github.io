---
layout: post
title:  "XOR 관련 문제를 푸는 다양한 접근법"
date:   2021-07-17-18:00
author: cs71107
tags: [algorithm,linear-algebra]
---

## 들어가기 전에 ##

XOR (bitwise exclusive or)은 대표적인 bit 연산 중 하나입니다. 다른 bit 연산인 AND,OR,NOT이 갖지 못하는 특성 때문에, PS 및 CP에서도 관련 문제가 심심치 않게 출제되곤 합니다. 하지만 문제 제목이나 지문에 XOR이 있으면 일단 긴장하거나, 기피하시는 분들도 꽤 자주 본 것 같습니다. 그래서, 이번 글에서는 이런 XOR 관련 문제들에 접근하는 방법 몇 가지를 설명해 보려고 합니다.

## Basic Part ##

본격적으로 들어가기에 앞서, XOR 연산이 가지고 있는 성질 몇 가지에 대해서 간단하게 짚고 넘어가려고 합니다. 여러 가지 성질 중 다음 4가지를 짚고 넘어가겠습니다.

- $x \oplus 0 = x$
- $x \oplus x = 0$
- $x \oplus y = y \oplus x$
- $(x \oplus y) \oplus z = x \oplus (y \oplus z)$

위의 성질들은 생각해보면 너무나 자명하게 보입니다. 하지만 자칫하다가 놓치기 쉬우므로, 관련 문제를 풀 때 위의 성질들을 항상 기억하는 것이 좋습니다.저 성질들만 제대로 이해해도 풀 수 있는 문제들이 상당히 많습니다. 
후에 설명할 주제들 같이 특정 알고리즘/내용을 알면 풀 수 있는 문제도 XOR 관련 문제들 중에선 많습니다만, 사실 XOR 관련 문제의 대다수는 위의 성질들을 적절히 이용해 풀어야 하는 Ad-hoc인 경우가 많습니다. (Ad-hoc에 대한 설명은 [링크](https://codeforces.com/blog/entry/85172?#comment-728307)를 참고하세요.) 어쩌면 그래서 사람들이 어려워하는 것인지도 모르겠습니다.

위의 성질들을 이해하면 접근가능한 연습문제들과 힌트를 남깁니다.

[BOJ 12844](https://www.acmicpc.net/problem/12844)

두 번째 성질을 이해하고, lazy propgartion에 대한 지식이 있다면 쉽게 해결 가능합니다.

[BOJ 18254](https://www.acmicpc.net/problem/18254)

XOR 연산의 성질들을 잘 이해하고 있다면 접근 가능합니다. 다만, 제출 통계에서 볼 수 있듯이, 시간 제한이 빡빡한 편이니 주의하시기 바랍니다.

## Substring에 대한 XOR합 ##

이 유형도 여러 가지 상황이 있습니다만, 다음과 같은 형태의 문제를 공통적으로 해결하고자 하는 경우가 많습니다.

- $n$개의 음이 아닌 정수로 이루어진 수열 $A_{1}, A_{2}, \dots, A_{n}$이 있을 때, 수열의 substring $A_{L \cdots R}$에 대해서, Substring의 XOR합 ($A_{L} \oplus A_{L+1} \oplus \dots \oplus A_{R}$)에 대한 문제

문제를 좀 더 편하게 다루기 위해서, 다음과 같이 $V_{i}$를 정의합시다.

- $V_{0} = 0$, $V_{i} = A[{1}] \oplus A[{2}] \oplus \cdots \oplus A[{i}]$

그렇다면, 다음과 같은 사실을 얻을 수 있습니다.

- 어떤 substring $A[{L}\dots{R}]$ 에 대해서, 이 substring의 XOR 합은 $V_{L-1} \oplus V_{R}$이다.

증명은 다음과 같습니다.

- proof : $A[{L}\dots{R}]$의 XOR 합은 $A[{L}] \oplus A[{L + 1}] \oplus \cdots \oplus A[{R}]$ 이 된다. 그러므로, 이때, $k = (A[{1}] \oplus A[{1}] \oplus \cdots \oplus A[{L - 1}])$로 두자. 그럼 $k \oplus k = 0$이다. 교환 및 결합 법칙이 성립하므로 $A[{L}] \oplus A[{L + 1}] \oplus \cdots \oplus A[{R}]= k \oplus k \oplus (A[{L}] \oplus \cdots \oplus A[{R}]) = k \oplus (k \oplus A[{L}] \oplus \cdots \oplus A[{R}]) = V_{L-1} \oplus V_{R}$ 가 된다. 증명 끝.

$V_{0}, V_{1}, \dots ,V_{n}$는 누적합을 계산하듯 $O(n)$만에 쉽게 계산 가능합니다. 이제 문제는 $V_{0}, V_{1}, \dots ,V_{n}$에서 두 개의 원소를 뽑은 후 XOR한 값들에 대한 문제로 변환됩니다.
그리고 일반적인 누적합은 indices의 순서를 고려해야 하지만, XOR에서는 교환법칙이 성립하므로, 그럴 필요가 없습니다. 이 차이 때문에, 일반적인 누적합과 차이점이 생기니 문제를 풀 때 이 점을 꼭 기억하시기 바랍니다.

문제 하나를 풀면서 실제 문제를 푸는 데 어떻게 적용할지 감을 잡아봅시다.

[BOJ 10464](https://www.acmicpc.net/problem/10464)

두 양의 정수 $S,F$에 대해, $S$에서 $F$까지의 모든 정수를 XOR한 결과를 묻는 문제입니다. 단순히 $S$부터 $F$ 까지의 정수를 모두 XOR하는 방식으론 시간 제한 안에 돌아가지 않을 것이 분명합니다. 수열 $A$가 $A_{i} = i$인 무한한 수열이라고 할 때, 위의 문제는 수열 $A$의 substring $A[S \dots F]$의 XOR합을 묻는 문제가 되므로, 위의 방법으로 접근해서 해결이 가능합니다. 단, $F$의 최댓값으로 $1000000000$까지 가능하므로, 여전히 너무 큽니다. 여기서 $A_{i} = i$라는 특수한 상황이기 때문에, $i$를 $4$로 나눈 나머지가 $3$일 때, $V_{i} = 0$이라는 사실을 이용하면 쉽게 해결할 수 있습니다. 증명은 간단합니다.

코드는 다음과 같습니다.

```cpp
#include <bits/stdc++.h>
#define f first
#define s second
#define MOD 1000000007
#define PMOD 998244353
#define pb(x) push_back(x)
using namespace std;

typedef long long int ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> plii;
typedef pair<int, pii> piii;
const int INF = 1e9+10;
const ll LINF = 1LL*INF*INF;
const int MAXN = 2e5+10;
const int MAXM = 5e3+10;

int solve(int x)
{
    int res = 0;

    for(int i=(x>>2)<<2; i<=x; i++){
        res^=i;
    }

    return res;
}

int main()
{
    int x,y,res;
    int tc;

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin>>tc;

    while(tc--){
        cin>>x>>y;

        res = solve(y)^solve(x-1);

        cout<<res<<"\n";
    }

    return 0;
}

```

연습 문제와 힌트를 남깁니다.

[BOJ 14878](https://www.acmicpc.net/problem/14878)

위처럼 접근해야 합니다, 다만 이 문제는 XOR convolution을 해결할 수 있어야 합니다. XOR convolution에 대한 내용은 다음에 쓸 기회가 있다면 쓰도록 하겠습니다.

## XOR한 값들의 합과 관련된 문제를 해결해야 하는 경우 ##

XOR에서는 각 bit가 독립적으로 결정되기 때문에, 이 사실에 집중해서 각 bit가 1인 경우가 몇 가지인지를 구하면 쉽게 구할 수 있는 경우가 많습니다. 문제를 하나를 풀어보면서 감을 잡아봅시다.

[BOJ 2830](https://www.acmicpc.net/problem/2830)

주민들의 이름을 $A_{0}, A_{1}, \dots, A_{n-1}$라고 하면, 문제의 정답은 $\sum_{i=0}^{n} \sum_{j=i+1}^{n} A_{i} \oplus A_{j}$를 계산하는 것입니다. naive하게 한다면 $O(n^2)$의 계산량이 필요하겠지만, 복잡도를 줄일 수 있는 방법이 있습니다.

위에서 언급했다시피, XOR에선 각 bit가 독립적으로 결정됩니다. 이 사실에 주목해서, $A_{i} \oplus A_{j}$ 값들 중 $a$번째 bit가 1인 수가 몇 개 있을 지 생각해봅시다.
$A_{i} \oplus A_{j}$의 $a$번째 bit가 1이려면 $A_{i}$의 $a$번째 bit와 $A_{j}$의 $a$번째 bit는 달라야 합니다. 그러므로 $A_{i} \oplus A_{j}$값들 중 $a$번째 bit가 1인 수들의 개수는 $A_{i}$값들 중 $a$번째 bit가 0인 것의 개수와 $a$번째 bit가 1인 것의 개수의 곱이 됩니다. 비트 수가 최대 $k$라고 할 때, 시간 복잡도 $O(kn)$으로 구현할 수 있습니다.

($a$번째 bit란 수를 2진법으로 나타냈을 때 오른쪽에서 $a$번째에 있는 bit를 의미합니다. 예를 들어 $10({1010}_{2})$의 $0$번째 bit는 $0$, $1$번째 bit는 $1$입니다.)

위의 문제를 푸는 코드입니다.

```cpp
#include <bits/stdc++.h>
#define f first
#define s second
#define MOD 1000000007
#define PMOD 998244353
#define pb(x) push_back(x)
using namespace std;

typedef long long int ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> plii;
typedef pair<int, pii> piii;
const int INF = 1e9+10;
const ll LINF = 1LL*INF*INF;
const int MAXN = 1e6+10;
const int MAXM = 5e3+10;

int cal[110];

int main()
{
    int n,x;

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin>>n;

    for(int i=0;i<n;i++){
        cin>>x;
        for(int j=19;j>=0;j--){
            if((1<<j)&x)cal[j]++;
        }
    }

    ll curv = 0,res = 0;

    for(int i=0;i<20;i++){
        curv = 1LL*cal[i]*(n-cal[i]);
        res += (curv<<i);
    }

    cout<<res<<"\n";

    return 0;
}

```

위의 내용을 응용해 풀 수 있는 연습문제와 힌트입니다.

[BOJ 13710](https://www.acmicpc.net/problem/13710)

substring에 대한 문제이므로, substring의 XOR합에 대한 내용을 다시 읽어보세요. 어렵지 않게 해결할 수 있으실 것입니다.

## 수들의 합들을 XOR한 값을 구해야 하는 경우 ##

여러 가지 variation이 있을 수 있습니다만, 기본적으로 다음과 같은 형태입니다.

$n$개의 수 $A_{0}, A_{1}, \dots , A_{n-1}$에 대해, $A_{i}+A_{j}(0 \leq i \leq j < n)$값들을 XOR한 결과를 구하라.

합인 만큼 독립적으로 뭔가를 하긴 힘듭니다. XOR한 값들의 합에서 했던 것처럼 각 bit에 집중합시다. $a$번째 bit가 $1$이 되기 위해선, $A_{i}+A_{j}$값들 중 $a$번째 bit가 $1$인 것의 개수가 홀수 개이면 됩니다.

XOR한 값들의 합을 구했을 때 했던 것과 비슷하게, $a$번째 bit가 $1$인 것의 개수를 세면 됩니다. 정답이 $A_{i}$값들의 순서에 대해선 영향을 받지 않으므로, $A_{i}$가 정렬돼있다고 생각합시다. 각 $A_{i}$에 대해서 $A_{i}+A_{j}$ 값의 $a$번째 bit가 $1$인 ${j}$가 몇 개인지 구하면 되는데, 현재 $A_{i}$ 값들이 정렬돼 있기 때문에, 조건을 만족하는 $j$값의 집합은 연속한 정수구간 여러개의 형태가 됩니다.
이때, 각 $A_{i}$를 $2^{a+1}$로 나눈 나머지에 대해서만 생각해도 답이 달라지지 않습니다. $A_{i}$를 $2^{a+1}$로 나눈 나머지에 대해서 생각하면 구간의 개수를 최대 두 개 정도로 줄일 수 있습니다. 이제 각 $A_{i}$에 대해 원하는 구간을 two pointer 등의 방법으로 찾고, 그 개수를 세어 주기만 하면 정답의 $a$번째 bit가 1인지 0인지 결정할 수 있습니다. 
대부분의 문제들의 경우 bit수가 $30 \sim 60$정도 되기 때문에 적절히 최적화 하여 빠르게 동작하게 하는 것이 중요합니다.

위의 문제 그대로인 [BOJ 21860](https://www.acmicpc.net/problem/21860)를 푸는 코드입니다.
코드가 상당히 길어서, [링크](http://boj.kr/421590a2c3824df3a043f7061e4f410f)로 대체하겠습니다.

관련 연습 문제와 힌트를 남깁니다.

[ARC092 B](https://atcoder.jp/contests/arc092/tasks/arc092_b)

이번에는 서로 다른 수열 $a_{i},b_{j}$에 대해서 해결해야 합니다. 풀이는 비슷합니다.

[BOJ 15896](https://www.acmicpc.net/problem/15896)

XOR이 아니고 AND이긴 합니다만, 원리가 비슷해 추가했습니다.

## Xor Maximization ##

이제 약간 어려운 주제를 알아봅시다. XOR 관련된 문제에서는 특정 집합 (또는 배열)에 대해서, XOR과 관련된 최댓값 / 최솟값을 알아내라는 유형의 문제가 굉장히 자주 출제됩니다. 문제의 조건에 따라 접근 방식이 달라지곤 하는데, 이 글에서는 가장 대표적인 두 가지 유형에 대해 알아 보도록 하겠습니다.

## Xor Maximization Part1 (2 element) ##

첫 번째로 설명할 유형은 어떤 두 원소의 XOR 결과 중 최댓값(또는 최솟값)을 구하게 하는 문제입니다. 문제를 조금 formal 하게 정리하겠습니다.

- $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$ 들에 대해 $X_{i} \oplus X_{j}(0 \leq i, j < n)$의 최댓값을 구하여라.

이제 각 $X_{i}$에 대해, $X_{i} \oplus X_{j}$의 최댓값을 찾는방법을 알아 봅시다.

편의상 $X_{0}, X_{1}, \dots ,X_{n-1}$들 모두 2진법으로 나타냈을 때 $k$자리로 표현 가능하다고 합시다. 이때, 현재 $X_{i}$를 2진법으로 표현했을때의 $i$ 번째 자리수가 $a_{i}$라고 합시다. 예를 들어서 $X_{i} = 5$라면, $a_{0} = 1, a_{1} = 0, a_{2} = 1$이고, $i \geq 3$인 $i$에 대해 $a_{i} = 0$입니다.
$X_{i} \oplus X_{j}$의 값이 최대가 되는 $X_{j}$의 각 자리수를 결정합니다. 그 결정해야 하는 자리수들을 $b_{0}, b_{1}, \dots , b_{k-1}$라고 합시다. 그리디하게 접근해봅시다. $k-1$번째 자리부터 $0$번째 자리까지 본다고 합시다. 현재 $i$ 번째 자리를 보고 있고, $b_{k-1},b_{k-2}, \dots, b_{i+1}$가 결정된 상황이라고 합시다. 그럼 현재 상황은 $b_{i}$를 결정해야합니다. $a_{i} = 1$인 경우, $b_{i} = 0$인 $X_{j}$가 존재한다면, 즉 2진수로 나타냈을 때  $b_{k-1}b_{k-2}\dots{b_{i+1}0}$가 접두사인 수가 존재한다면 $b_{i} = 0$으로 결정하면 됩니다. 반대로 없다면, $b_{i} = 1$이 됩니다. $a_{i} = 0$이면 $b_{i} = 1$이 될 수 있으면 $b_{i} = 1$, 아니면 $b_{i} = 0$으로 결정하면 됩니다. 이 로직이 성립한다는 것을 증명하는 것은 별로 어렵지 않습니다.

위의 로직을 구현할 때 가장 까다로운 부분은 수를 2진수로 나타냈을 때 특정 수를 접두사로 가지는 수가 존재 여부를 판단하는 부분입니다. 이부분은 이분탐색으로 판별하거나, 2진수 표현을 문자열처럼 생각해서 Trie를 구성한 다음 구현할 수도 있습니다. 이부분은 적당히 빠르게만 동작하도록 자유롭게 구현하시면 됩니다. 단, 문제에 따라 구현하는데 더 편한 방식이 있을 순 있습니다.

다음 문제를 풀어보면서 감을 잡아봅시다.

[BOJ 13505](https://www.acmicpc.net/problem/13505)

구하고자 하는 것이 매우 명확합니다. 위의 알고리즘을 그대로 적용해서 답을 구하면 됩니다. 위에선 $i \neq j$를 가정하지 않았으나, 만약 $i = j$라면 어차피 0이므로, 음이 아닌 정수끼리 XOR한 값 중 최댓값을 찾는 데에는 전혀 문제가 없습니다. 만약 최솟값을 구해야 했다면 신경써줘야 하겠지만요.

코드는 다음과 같습니다. 아래 코드는 간단한 trie를 사용해 구현했습니다.

```cpp
#include <bits/stdc++.h>
#define f first
#define s second
#define MOD 1000000007
#define PMOD 998244353
#define pb(x) push_back(x)
using namespace std;

typedef long long int ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> plii;
typedef pair<int, pii> piii;
const int INF = 1e9+10;
const ll LINF = 1LL*INF*INF;
const int MAXN = 1e5+10;
const int MAXM = 5e3+10;

priority_queue<int> pq;
vector<vector<int> > graph;
queue<int> que;

int A[MAXN];

int trie[MAXN*30][2];
int seq;

void update(int x){

    int idx = 0;

    for(int i=29;i>=0;i--){

        if((1<<i)&x){
            if(trie[idx][1]!=-1){
                idx = trie[idx][1];
            }
            else {
                seq++;
                trie[idx][1] = seq;
                idx = seq;
            }
        }
        else {
            if(trie[idx][0]!=-1){
                idx = trie[idx][0];
            }
            else {
                seq++;
                trie[idx][0] = seq;
                idx = seq;
            }
        }
    }

    return;
}

int getans(int x){

    int idx = 0;
    int res = 0;

    for(int i=29;i>=0;i--){

        if((1<<i)&x){
            if(trie[idx][0]!=-1){
                idx = trie[idx][0];
                res|=(1<<i);
            }
            else {
                idx = trie[idx][1];
            }
        }
        else {
            if(trie[idx][1]!=-1){
                idx = trie[idx][1];
                res|=(1<<i);
            }
            else {
                idx = trie[idx][0];
            }
        }

        assert(idx!=-1);
    }

    return res;
}

int main()
{
    int n;
    int res = 0;

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin>>n;

    for(int i=0;i<n;i++)
        cin>>A[i];

    memset(trie,-1,sizeof(trie));

    update(A[0]);

    for(int i=1;i<n;i++){
        res = max(res,getans(A[i]));
        update(A[i]);
    }

    cout<<res<<"\n";

    return 0;
}

```

위 문제 외에 위의 알고리즘을 이해하면 접근 및 해결 가능한 연습문제들과 힌트를 남깁니다.

[BOJ 13504](https://www.acmicpc.net/problem/13504)

substring에 대한 문제네요. 위에서 알아본대로 $V_{i}$를 계산한 다음 동일한 문제를 풀면 됩니다.

[BOJ 13445](https://www.acmicpc.net/problem/13445)

XOR한 결과가 k이하인 쌍이 몇 개인지 구하는 문제입니다. 최댓값을 구하는 것은 아니지만, 비슷한 알고리즘으로 해결할 수 있습니다.

[BOJ 13538](https://www.acmicpc.net/problem/13538)

위의 로직을 이해하면 위 문제의 2번 쿼리를 어떻게 처리해야 할지 알 수 있습니다. 위의 문제는 그 외에도 자료구조에 대한 배경지식이 조금 필요합니다.

[BOJ 16901](https://www.acmicpc.net/problem/16901)

문제를 풀기 위한 구현에 위의 알고리즘이 사용됩니다. 굉장히 재밌으니 풀어보시는 걸 추천드립니다.

## Xor Maximization Part2 ##

두 번째로 설명할 유형은 어떤 집합의 부분집합의 XOR 합 중 최댓값(또는 최솟값)을 찾는 유형입니다. 흔히 XOR Maximiaztion이라고 하면 이 유형을 가리킵니다. 문제를 좀 더 formal 하게 표현해 보겠습니다.
문제에 따라 몇 가지 변형이 있을 수 있겠습니다만, 기본적으론 아래와 같은 문제를 해결하고자 합니다.

- $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$ 들에 대해서, 집합 $S = \lbrace 0, 1, \dots , n-1 \rbrace$의 부분집합 $T({T}\subset{S})$에 대해 $T = \lbrace t_{0}, t_{1}, \dots , t_{k-1} \rbrace$라고 할 때, $X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{k-1}}$의 최댓값을 구하여라. 단, $T = \emptyset$인 경우, 그 값은 $0$으로 한다.

편의를 위해서, 위의 문제에서 $S$의 부분집합 $T({T}\subset{S})$에 대해 $T = \lbrace t_{0}, t_{1}, \dots , t_{k-1} \rbrace$라고 할 때, $X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{k-1}}$값을 $X_{0}, X_{1}, \dots ,X_{n-1}$에 대한 $T$의 XOR 합이라고 정의합시다. $\emptyset$의 경우 XOR합은 $0$입니다.

첫 번째 유형과는 다른 방식의 접근이 필요합니다. 어떻게 해야 할까요? 약간의 선형대수 지식이 필요하다는 것을 미리 밝히겠습니다.

우선 XOR이라는 연산에 대해 다시 한번 생각해 봅시다. 어떤 두 수 $a, b$를 XOR 한 결과를 생각합시다. $a \oplus b = X$라고 두고, $X$를 이진법으로 표현했을 때, $0, 1, 2, \dots $번째 자리수가 각각 $x_{0}, x_{1}, x_{2}, \dots $라고 했을 때, $x_{0}, x_{1}, x_{2}, \dots $는 각각 독립적으로 결정된다는 것을 알 수 있습니다. 

그러므로, 어떤 수 $a$를 $GF(2)$위의 vector로 생각할 수 있습니다. 그리고 XOR은 $GF(2)$에서 vector의 덧셈이 됩니다. $GF(2)$에 대한 자세한 내용은 참고자료[^1]와 추가적인 검색을 통해서 알아보시기 바랍니다.

다시 원래 문제로 돌아가서, $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$ 들에 대해서, 집합 $S = \lbrace 0, 1, \dots , n-1 \rbrace$의 부분집합 $T({T}\subset{S})$의 XOR합들의 집합은 위의 관점을 적용하면 $GF(2)$위의 vector space가 됩니다. 편의상 그 집합을 $U$라고 합시다.

$U$의 기저를 구한다면 그걸 이용해서 원하는 정답을 얻을 수 있을 것 같습니다. $U$의 기저를 구해봅시다. 이때, 약간 특수한 기저를 구해서, $U$의 최대원소를 구할 때 기저에서 쉽게 최댓값을 구할 수 있으면 좋을 것 같습니다.

이런 기저는 생각보다 쉽게 얻을 수 있는데, 각 vector들을 행렬의 row vector로 생각한 뒤, 그 행렬의 RREF를 구한 뒤 얻어지는 zero vector가 아닌 vector들이 됩니다.

행렬의 RREF(Reduced Row Echelon Form)나 REF(Row Echelon Form)에 대한 더 자세한 내용은 참고자료[^2]를 참고하시기 바랍니다.

여기서는 RREF의 다음과 같은 성질을 기억하면 좋습니다.

- 선행성분이 있는 열에 대해서, 그 선행성분을 가지는 행을 제외한 행에서의 값은 전부 0이다.

사실 REF만 구해도 위의 문제에서 최댓값을 구하는 데에는 사실 문제가 없습니다. 하지만 RREF를 사용하면 다른 문제에 대해서 확장하기 더 쉬우므로, RREF를 사용합시다.

실제 RREF를 구하는 방법으론 Gauss-Jordan Elimination이 있습니다. 이 역시 참고자료에 자세히 설명돼 있으므로, 여기선 넘어가도록 하겠습니다. 참고자료[^3]를 확인하시기 바랍니다.

이제 RREF를 얻는 방법을 알았으니, 기저를 구할 수 있습니다. 이제 이렇게 RREF에서 얻은 기저를 $v_{0}, v_{1}, \dots, v_{m-1}$라고 하면 가능한 XOR합들은 모두 $x = \sum_{i=0}^{m-1} e_{i}v_{i}$로 표현되고, 각 수들을 표현하기 위해서 $e_{i}$들을 정할 때, 각 $e_{i}$들은 독립적으로 정해집니다. 
이때, 현재 기저가 RREF에서 나왔으므로, RREF의 성질을 생각해보면 기저의 각 원소에 대응 되는 행에서 선행성분의 열 번호를 차례로 $a_{0}, a_{1}, \dots a_{m-1}$라고 하면 $e_{i}$가 $x$의 $a_{i}$번째 bit가 되고, $a_{0}, a_{1}, \dots a_{m-1}$를 제외한 나머지 bit들은 종속적으로 정해지게 됩니다. 그런데 위에서 언급한 성질 때문에, 최댓값이 되려면 결국 모든 $i$에 대해 $e_{i} = 1$이 돼야 합니다. 그러므로 모든 기저들을 구한 뒤, 그 기저들을 모두 XOR 하면 원하는 답을 얻을 수 있습니다.

현재 푼 문제와 똑같은 [문제(BOJ 11191)](https://www.acmicpc.net/problem/11191)를 푸는 코드입니다.

```cpp
#include <bits/stdc++.h>
#define f first
#define s second
#define MOD 1000000007
#define PMOD 998244353
#define pb(x) push_back(x)
using namespace std;

typedef long long int ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> plii;
typedef pair<int, pii> piii;
const int INF = 1e9+10;
const ll LINF = 1LL*INF*INF;
const int MAXN = 2e5+10;
const int MAXM = 5e3+10;

priority_queue<int> pq;
vector<vector<int> > graph;
queue<int> que;

ll A[110];

int main()
{
    int n;
    ll x;

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin>>n;

    for(int i=0;i<n;i++)//get REF
    {
        cin>>x;
        for(int i=60;i>=0;i--){
            if((1LL<<i)&x){
                if(!A[i]){
                    A[i] = x;
                    break;
                }
                x^=A[i];
            }
        }
    }

    for(int i=0;i<60;i++){//get RREF
        if(A[i]){
            for(int j=i+1;j<60;j++){
                if((1LL<<i)&A[j]){
                    A[j]^=A[i];
                }
            }
        }
    }

    ll res = 0;

    for(int i=60;i>=0;i--){
        res ^= A[i];
    }

    cout<<res<<"\n";

    return 0;
}

```

이제 위의 경우에서 살짝 특수한 경우를 한번 살펴보겠습니다. 다음 문제를 봅시다.

[BOJ 16685](https://www.acmicpc.net/problem/16685)

알아본 문제와 거의 비슷하나, 이번에는 부분집합의 크기가 **짝수**라는 제한 조건이 있습니다. 위의 방법을 바로 적용하기는 힘들어 보입니다. 어떻게 해야 할까요?

의외로 간단하게 해결할 수 있습니다. 원래 수열이 $X_{0}, X_{1}, \dots ,X_{n-1}$라고 합시다. 그렇다면 $X_{0} \oplus X_{0}, X_{1} \oplus X_{0}, \dots ,X_{n-1} \oplus X_{0}$에 대해서 위의 알고리즘을 적용시키면 됩니다. 왜 이 사실이 성립하는지 다음과 같이 증명할 수 있습니다.

- proof : $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$ 들에 대해서, 집합 $S = \lbrace 0, 1, \dots , n-1 \rbrace$의 부분집합 $T_{0}(T \subset S \ ,\  \left\lvert T_{0} \right\rvert = 2k(k \in (\lbrace 0 \rbrace \cup \mathbb{Z^{+}})))$에 대해, $X_{0}, X_{1}, \dots , X_{n-1}$에 대한 $T_{0}$의 XOR 합들의 집합을 $A$라고 하자. 그리고 $Y_{i}=X_{i} \oplus X_{0}\ (0 \leq i < n)$이라고 뒀을 때, $S = \lbrace 0, 1, \dots , n-1 \rbrace$의 부분집합 $T(T \subset S)$에 대해, $Y_{0}, Y_{1}, \dots , Y_{n-1}$에 대한 $T$의 XOR 합들의 집합을 $B$라고 하자.

이제 $A \subset B$임을 증명하자. $A$의 각 원소는 $X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{2k-1}}$와 같이 표현된다. ($\lbrace t_{0},t_{1},\dots,t_{2k-1} \rbrace \subset T_{0}$) $X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{2k-1}} = (X_{t_{0}} \oplus X_{0}) \oplus (X_{t_{1}} \oplus X_{0}) \oplus \dots \oplus (X_{t_{2k-1}} \oplus X_{0}) = Y_{t_{0}} \oplus Y_{t_{1}} \oplus \dots \oplus Y_{t_{2k-1}}$가 되므로, $A \subset B$임이 증명된다.

이제 $B \subset A$임을 증명하자. $B$의 각 원소는 $Y_{t_{0}} \oplus Y_{t_{1}} \oplus \dots \oplus Y_{t_{k-1}}$와 같이 표현된다. ($\lbrace t_{0},t_{1},\dots,t_{k-1} \rbrace \subset T$) 이때, $k$가 홀수일 때, $0 \in \lbrace t_{0},t_{1},\dots,t_{k-1} \rbrace$인 경우, 일반성을 잃지 않고  $t_{0} = 0$라고 했을 때, $Y_{0} = X_{0} \oplus X_{0} = 0$이므로, $Y_{t_{0}} \oplus Y_{t_{1}} \oplus \dots \oplus Y_{t_{k-1}} = Y_{t_{1}} \oplus \dots \oplus Y_{t_{k-1}}$가 성립한다.

$0 \notin \lbrace t_{0},t_{1},\dots,t_{k-1} \rbrace$인 경우에도 $Y_{t_{0}} \oplus Y_{t_{1}} \oplus \dots \oplus Y_{t_{k-1}} = Y_{0} \oplus Y_{t_{0}} \oplus \dots \oplus Y_{t_{k-1}}$가 되므로, $k$가 짝수인 경우와 표현할 수 있는 수의 집합이 같다는 것을 알 수 있다. 이제 $k$가 짝수인 경우만 본다면, $Y_{t_{0}} \oplus Y_{t_{1}} \oplus \dots \oplus Y_{t_{k-1}} = (X_{t_{0}} \oplus X_{0}) \oplus (X_{t_{1}} \oplus X_{0}) \oplus \dots \oplus (X_{t_{k-1}} \oplus X_{0}) = X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{k-1}}$가 된다. $k$가 짝수이므로, $B \subset A$임이 증명된다.

$A \subset B$이고, $B \subset A$이므로, $A = B$이다. 따라서 $A$의 최대원소는 $B$의 최대원소와 같고, $X_{0}, X_{1}, \dots ,X_{n-1}$대신 $X_{0} \oplus X_{0}, X_{1} \oplus X_{0}, \dots ,X_{n-1} \oplus X_{0}$에 대해서 풀면 원하는 결과를 얻는다. 증명 끝.

위의 문제에서는 $\emptyset$을 제외하고 XOR 합 중 최댓값을 찾는다고 돼있긴 합니다만, 어차피 $\emptyset$에 대해선 XOR합이 0이므로, 최댓값을 구할 경우 무시해도 상관없습니다.

부분집합의 크기가 짝수인 경우에는 위와 같이 해결할 수 있지만, 홀수일 경우는 어떨까요? 홀수일 경우를 해결하기 위해서, 원래 문제의 확장인 다음 문제를 생각해 봅시다.

- 어떤 상수 $K$와 $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$ 들에 대해서, 집합 $S = \lbrace 0, 1, \dots , n-1 \rbrace$의 부분집합 $T({T}\subset{S})$에 대해 $T = \lbrace t_{0}, t_{1}, \dots , t_{k-1} \rbrace$라고 할 때, $K \oplus X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{k-1}}$의 최댓값을 구하여라. 단, $T = \emptyset$인 경우, 그 값은 $K$으로 한다.

원래 문제에서는 $K = 0$입니다. 원래 문제에서처럼 RREF를 구하고, 기저의 각 원소에 대응 되는 행에서 선행성분의 열 번호를 차례로 $a_{0}, a_{1}, \dots a_{m-1}$라고 했을 때, $K$와 XOR한 결과에서 $a_{0}, a_{1}, \dots a_{m-1}$번째 bit의 값은 1이 되도록 정합시다. 그럼 그 결과가 최댓값이라는 것을 어렵지 않게 알 수 있습니다. 증명은 숙제로 남깁니다.

이제 다음과 같은 알고리즘을 생각해봅시다. 편의상 위에서 확장된 문제를 $K$와 $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$들에 대한 XOR Maximization 문제라고 합시다.

- $0 \leq i < n$인 각 $i$에 대해서, $X_{i}$와 $n$개의 음이 아닌 정수 $X_{0} \oplus X_{0}, X_{1} \oplus X_{0}, \dots , X_{n-1} \oplus X_{0}$들에 대한 XOR Maximization 문제를 풀고, 그 답을 $W_{i}$라고 하면, $res = max(W_{0},W_{1},\dots,W_{n-1})$를 구한다. $res$가 답이다.

이제 위의 알고리즘이 어떻게 홀수인 경우를 해결할 수 있는지를 봅시다. 부분집합의 크기가 홀수일 경우인 경우, $n$개의 음이 아닌 정수 $X_{0}, X_{1}, \dots ,X_{n-1}$에 대해서 해결한다고 하면, 구하는 최댓값(편의상 $ans$라고 둡시다.)은 어떤 집합 $\lbrace t_{0},t_{1},\dots,t_{2k} \rbrace$에 대해서(단, $k$는 음이 아닌 정수, 각 $i$에 대해 $\lbrace t_{0},t_{1},\dots,t_{2k} \rbrace \subset \lbrace 0, 1, \dots, n-1 \rbrace$), $X_{t_{0}} \oplus X_{t_{1}} \oplus \dots \oplus X_{t_{2k}}$꼴로 표현됩니다. 따라서, $i = t_{0}$일 때 $W_{i} = ans$가 될 것입니다. 그러므로, $res \geq ans$ 입니다. 반대로 $ans \geq res$임도 쉽게 증명할 수 있습니다. 자세한 증명은 짝수 개를 다룰 때 했던 증명을 조금만 변형하면 되니, 숙제로 남겨두겠습니다.

시간 복잡도는 어떻게 될까요? 일반적으로 $k$자리수에 대해서 $n$개 수에 대해 위의 문제를 푼다고 할 때, 시간 복잡도는 $O(nk^2)$가 됩니다. 이렇게 보면 복잡도가 매우 큰 것 같지만, 대부분의 문제에선 $k \leq 60$ 정도로, 정수 자료형의 xor로도 충분히 연산할 수 있습니다. 따라서, 실제로 작동하는데는 그렇게 많은 시간이 걸리지 않습니다. $k \geq 1000$ 정도로 큰 문제들에 대해선 따로 최적화를 적용해야 풀리는 경우가 많습니다.

[이 글](https://www.secmem.org/blog/2019/12/14/%EC%A3%BC%EC%96%B4%EC%A7%84-%EC%88%98%EB%93%A4%EC%9D%98-XOR-%EC%97%B0%EC%82%B0%EC%9C%BC%EB%A1%9C-%EB%A7%8C%EB%93%A4-%EC%88%98-%EC%9E%88%EB%8A%94-%EC%88%98/)에서도 XOR maximization에 대해서 다루긴 하나, 조금 다른 방식으로 접근해봤습니다. 이 방법도 알아두면 좋으니 참고하시면 좋겠습니다.

위 문제 외에 위의 내용을 이해하면 접근 및 해결 가능한 연습문제와 힌트를 남깁니다.

[BOJ 19245](https://www.acmicpc.net/problem/19245)

위의 내용을 이해하면 접근 가능합니다. 더 많이 언급하면 큰 spolier가 될 수 있으니 여기까지만 언급합니다.

[BOJ 16904](https://www.acmicpc.net/problem/16904)

각 시점에서의 RREF를 관리하면 풀 수 있습니다. offline으로 어떻게 해야할지 생각해보세요.

[BOJ 20349](https://www.acmicpc.net/problem/20349)

자세한 내용은 너무 큰 spolier이므로, 위의 내용을 이해해야 풀 수 있다는 정도만 언급하겠습니다.

[BOJ 20557](https://www.acmicpc.net/problem/20557)

위의 문제의 업그레이드 버전입니다.

## 마치며 ##

지금까지 다양한 XOR 문제에 대해 접근하는 방법에 대해서 알아보았습니다. XOR문제 중에는 재밌는 문제가 정말 많습니다. 하지만 위에서 언급했던 Ad-hoc 적인 부분이 많아 대부분의 글이 특정 파트에 대한 부분만 다뤘던 것 같습니다. 이 글도 모든 부분을 다뤘다고 할 수는 없습니다만(xor fft 등...), 여기 있는 내용만 잘 이해하셔도 대부분의 XOR 관련 문제를 접근 및 해결하는데에는 큰 어려움이 없으리라 생각합니다. 위의 연습 문제들의 풀이는 다른 곳에서 쓰거나, 다음에 다시 쓸 때 다루도록 하겠습니다.

위 연습 문제를 풀고 더 많은 문제를 풀어보고 싶으시다면, 연습 문제들을 정할 때 한국의 강력한 osu mania 국가대표 cheetose님의 [문제집](https://www.acmicpc.net/workbook/view/1613)을 많이 참고 했습니다. cheetose님께 감사를 전합니다. 그 외에 Codeforces나 Atcoder에서도 심심하면 관련 문제가 출제되니 찾아보시면 되겠습니다.

전 다음 글에서 돌아오겠습니다. 감사합니다.

## 참고 자료 ##

[^1]: <https://en.wikipedia.org/wiki/GF(2)>
[^2]: <https://en.wikipedia.org/wiki/Row_echelon_form>
[^3]: <https://math.jhu.edu/~bernstein/math201/RREF.pdf>
