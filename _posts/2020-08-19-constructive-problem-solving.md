---
layout: post
title:  "Constructive 문제를 어떻게 풀 것인가"
date:   2020-08-18 15:00:00
author: 16silver
tags: [constructive]
---

# Constructive 문제를 어떻게 풀 것인가

## 들어가며

알고리즘 대회에서 **Constructive Algorithm(구성적 알고리즘)**은, 말뜻 그대로 답을 증명할 수 있는 실제 예시를 구성하는 것이다. 이 용어는 수학의 Constructive Proof(구성적 증명, 존재성의 증명 등에서 귀류법이나 귀납법을 쓰지 않고 직접 조건을 만족하는 예시를 만들어 증명하는 방법)에서 왔다고 할 수 있다. Constructive 문제에서 구성해야 하는 대상은 대개 **Sequence**나 **Graph**이다. 무에서 유를 창조해내야 하는 문제의 특성상, 많은 배경 지식을 필요로 하기도 하며, 특정 문제에서의 테크닉을 일반화하기 어렵다는 Ad-hoc의 특징도 가지고 있다.

**그렇다고 Constructive 문제를 아예 대비를 할 수 없다고는 생각하지 않는다.**

Constructive 문제를 푸는 데 있어 가장 중요한 것 중에 하나는 **풀이의 논리적 흐름을 잡는 것**이다. 대부분의 Constructive 문제는 문제에서 주어진 조건을 계속해서 변형해나가야 한다. 이 때 **필요조건**을 찾을 수도 있고, **충분조건**을 찾을 수도 있다. **필요조건**은 여전히 답을 그 조건 내에서 구할 수 있다는 장점이 있지만, 문제의 복잡도를 줄이지 못하는 경우가 많아 주의가 필요하다. 반면 **충분조건**은 문제를 간단하게 만들지만, 충분조건을 사용한 시점부터는 문제에서 원하는 답의 존재성을 보장하지 못하기 때문에 많은 시행착오를 하게 되는 원인이 되기도 한다.

간단한 예시 하나를 보자. 길이가 $N$인 아무 팰린드롬 문자열을 만드는 문제를 생각해보자. **필요조건**으로 이 문제를 접근한다는 것은, "첫 번째 글자와 $N$번째 글자는 같아야 한다", "두 번째 글자와 $N-1$번째 글자는 같아야 한다"와 같은 조건들을 모아서 문제의 조건을 만족할 때까지 답의 범위를 좁혀나가는 것이다. 예를 들면 $N=5$인 경우에는 첫 번째와 다섯 번째로 들어갈 같은 문자, 두 번째와 네 번째로 들어갈 같은 문자, 그리고 세 번째로 들어갈 문자를 결정하여 문자열을 구성할 수 있다. 반면 **충분조건**으로 이 문제를 접근한다면, 여러 방향이 있겠지만, $N$개의 문자를 모두 같은 문자로 만들면 조건을 만족한다는 사실을 알 수 있다. 후자의 경우 구현이 매우 간단하다는 장점이 있지만, 일반적인 문제에서 이렇게 아이디어를 쉽게 찾아낼 수 있는 것은 아니며 이는 직관이 필요한 영역이다.

Constructive 문제는 많이 풀어보는 것이 답일까? 꼭 그렇지는 않다고 본다. 오히려 평소 문제 푸는 습관을 잘 들이는 것이 Constructive 문제를 잘 푸는 방법이 아닐까 생각한다. Competitive Programming의 여러 분야들 중에서는 빨리 푸는 것이 중요한 분야도 있지만, Constructive는 푸느냐 마느냐가 포인트인 분야라고 할 수 있다. 시간적인 여유를 가지고 문제를 풀 수 있을 때는, Constructive 문제가 아니더라도 본인의 풀이의 정당성을 확인하는 연습을 해 보는 게 좋은 것 같다. 엄밀하게 증명을 쓸 필요까지는 없지만, 어떻게 그 풀이가 나오게 되었는지 그 논리를 정리해보고, 또 실패한 접근과 성공한 접근이 각각 어떠한 접근이었는지를 되짚어보도록 하자. 분야에 대한 설명은 이 정도로 하고, 본격적으로 문제들을 보자. **문제별로 조건들을 어떻게 변형해 나가는지를 중점적으로 설명할 것이며, 문제마다 그 흐름을 짧게 정리해놓을 것이다.**

이 글에서 소개하는 건 가능한 한 가지 방법일 뿐이다. 대부분의 Constructive 문제들은 해법이 여러 가지이므로, 자신만의 멋진 풀이를 찾아보는 것도 좋은 연습이 될 것이다! 아래부터는 문제의 풀이에 대한 직접적인 설명이 담겨 있으므로, 스포일러를 당하고 싶지 않은 사람들이라면 스크롤을 맨 아래로 내리거나, 아래의 문제 목록만 확인한 뒤 창을 닫으면 된다.

[BOJ 1201. NMK](https://www.acmicpc.net/problem/1201)

[BOJ 19565. 수열 만들기](https://www.acmicpc.net/problem/19565)

[BOJ 15769. PuyoPuyo](https://www.acmicpc.net/problem/15769)

[BOJ 18162. Adler32](https://www.acmicpc.net/problem/18162)

[BOJ 15948. 간단한 문제](https://www.acmicpc.net/problem/15948)

[BOJ 17525. Enumeration](https://www.acmicpc.net/problem/17525)

## [BOJ 1201. NMK](https://www.acmicpc.net/problem/1201)

### 풀이

문제의 내용이 매우 간단하다. 최대 부분 증가 수열의 길이가 $M$이고, 최대 부분 감소 수열의 길이가 $K$인 길이 $N$짜리 수열 하나를 만들면 된다. 먼저 가능한지를 판단해야 하므로, $M$과 $K$를 고정했을 때 가능한 수열의 길이 $N$의 범위를 정해보자.

최대 부분 증가 수열과 최대 부분 감소 수열에서 공통인 원소는 하나일 수밖에 없다. 그렇기 때문에 수열의 전체 원소의 개수는, 최대 부분 증가 수열과 최대 부분 감소 수열의 합집합의 크기인 $M+K-1$ 이상이어야 한다. 또한, [Mirsky's Theorem](https://en.wikipedia.org/wiki/Mirsky%27s_theorem)이나 [Dilworth's Theorem](https://en.wikipedia.org/wiki/Dilworth%27s_theorem)에 의해 수열의 길이 $N$은 $MK$ 이하이다. 종합하면 $M+K-1 \le N \le MK$이다. 지금까지의 논리로는 이것이 **필요조건**이 된다. 그리고 이것이 **충분조건**도 된다는 사실을 알 수 있다.

수열 구성의 아이디어는, 증가 수열 $K$개를 모아놓는 것이다. 가장 큰 $M$개의 수를 오름차순으로 수열의 맨 처음에 놓고, 그 다음으로 큰 $M$개의 수를 오름차순으로 그 다음에 놓는 식으로 반복하는 것이다. 각 증가 수열의 길이는 $1$ 이상 $M$ 이하여야 하므로, 수열의 길이들의 합이 $N$이 되도록 잘 분배하면 된다. 물론 길이가 $M$인 것이 최소 하나는 있어야 한다.

``` c++
#include <bits/stdc++.h>
using namespace std;
void process(int mx, int len){
    for(int i = mx - len + 1; i <= mx; i++){
        printf("%d ", i);
    }
}
int main(){
    int N, M, K;
    scanf("%d%d%d", &N, &M, &K);
    if(M+K-1 <= N && N <= M*K){
        process(N, M);
        N -= M;
        if(K == 1) return 0;
        int q = N / (K-1), r = N % (K-1);
        for(int i = 0; i < r; i++){
            process(N, q+1);
            N -= q+1;
        }
        for(int i = r; i < K-1; i++){
            process(N, q);
            N -= q;
        }
    }
    else{
        puts("-1");
    }
}
```

### 흐름 정리

- M,K를 고정하고 수열의 길이 N에 대한 조건 찾기 **(필요조건)**
- 증가 수열 K개를 원소가 큰 순서대로 나열하기 **(충분조건)**

## [BOJ 19565. 수열 만들기](https://www.acmicpc.net/problem/19565)

### 풀이

문제의 요구사항을 한 마디로 정리하자면, "정점이 $n$개인 완전 방향 그래프의 모든 간선을 지나는 경로"이다. 수열의 각 수를 정점으로 생각하고 수열을 정점의 방문 순서로 생각하면 이웃한 두 수의 순서쌍은 간선이 되고, 문제의 조건은 $(a_i, a_{i+1})$들 중 같은 순서쌍이 없다는 것이므로 같은 간선을 여러 번 지나지 않는다는 뜻이 된다. 이 때 가능한 모든 $n^2$개의 순서쌍이 한 번씩만 나오게 할 수 있는지는, 정점이 $n$개인 완전 방향 그래프에서 오일러 회로가 존재하는지와 같은 이야기가 된다. 그리고 이것이 존재한다는 것은 이 그래프의 [모든 정점들의 indegree와 outdegree가 $n$으로 같다](https://en.wikipedia.org/wiki/Eulerian_path#Properties)는 데서 나온다.

그렇다면 어떻게 이 그래프를 구성하면 될까? 사실 오일러 회로의 존재 조건인 "indegree와 outdegree가 같음"은 "그래프를 사이클들로 분할할 수 있음"과 같다. 그리고 정점이 $1$부터 $n$까지 있는 완전 방향 그래프에서 생각해내기 가장 쉬운 사이클들은 `(1)-(2)-(3)-...-(n)-(1)`, `(1)-(3)-(5)-...-(1) / (2)-(4)-(6)-...-(2)`, ..., `(1)-(n-1)-(n-2)-...-(2)-(1)`처럼 일정한 간격으로 이동하는 사이클이다. 사이클의 개수가 얼마일지 세는 것은 쉽지 않지만, 사이클의 구성 자체는 쉽고, 어쨌든 그래프의 모든 간선이 이러한 사이클들 중 정확히 하나에 속한다는 사실은 알 수 있다. 이렇게 분할한 사이클들을 모든 정점이 나오는 것이 보장되는 `(1)-(2)-(3)-...-(n)-(1)`부터 시작해서, 각 정점에 끼워넣을 수 있는 사이클을 끼워넣는 식으로 구현하면 된다. 끼워넣는 것은 stack을 이용하여 구현한다. `(1)-(1)`과 같은 self-loop도 들어가야 하므로, 정점이 처음 나올 때 한 번 더 출력해준다.

``` c++
#include <bits/stdc++.h>
using namespace std;
int chkd[1001][1001];
vector<int> st;
int main(){
    int N;
    scanf("%d",&N);
    printf("%d\n",N*N+1);
    st.push_back(0);
    while(!st.empty()){
        int tmp = st.back();
        st.pop_back();
        printf("%d ",tmp+1);
        if(!chkd[tmp][0]){
            st.push_back(tmp);
            chkd[tmp][0]=1;
        }
        else{
            for(int i=1;i<N;i++){
                if(!chkd[tmp][i]){
                    int x=tmp;
                    st.push_back(x);
                    chkd[x][i]=1;
                    for(x=(x+(N-i))%N;x!=tmp;x=(x+(N-i))%N){
                        st.push_back(x);
                        chkd[x][i]=1;
                    }
                    break;
                }
            }
        }
    }
}

```

### 흐름 정리

- 문제를 그래프의 오일러 경로 문제로 변환하기 **(필요충분조건)**
- 그래프를 사이클들로 분해하여, 각 점에 덧붙이기 **(충분조건)**

## [BOJ 15769. PuyoPuyo](https://www.acmicpc.net/problem/15769)

### 풀이

이 문제에서는 PuyoPuyo 게임이 등장하고 있다. PuyoPuyo를 소재로 한 문제들이 여럿 있는데, Chain이 작동하는 방식 같은 것들이 문제로 만들 만한 소재가 되는 것 같다. 이 문제는 PuyoPuyo와 관련된 Constructive 문제인데, 사실 이 문제에서는 굳이 Chain을 쓸 필요가 없고, 쓰면 더 불편해진다. 터뜨리는 건 맨 위에서 터뜨리는 것만 해도 충분하다. 그리고 심지어는 모든 뿌요뿌요 쌍을 수직으로 떨어뜨려도 된다!

먼저 한 줄을 원하는 형태로 채워넣는 방법을 생각해보자. 간단하게, 아래부터 두 개씩 채워나가면 된다. 이렇게 하면 채워야 하는 뿌요의 개수가 짝수 개일 때는 해결된다. 홀수 개일 때는 어떻게 할까? 뿌요는 언제나 쌍으로 내려오기 때문에, 남는 뿌요의 수를 홀수로 만들기 위해서는 홀수 개의 뿌요가 모여서 터져야 한다. 5개가 모여서 터지면 되므로, 마지막 한 개의 뿌요를 넣어야 하는 상황에서는, 그 뿌요와 다른 색깔의 뿌요를 같이 넣은 뒤, 그 색깔 뿌요 2개로 된 뿌요 쌍을 두 개 넣으면 된다. 예를 들어, 열에 (1,2,3)을 채워야 하는 경우는, (1,2)-(3,1)-(1,1)-(1,1) 순서대로 넣으면 된다. 이렇게 하면 20개의 뿌요를 채워야 하는 열 하나에 대해, 최대 (19+5)/2=12개의 뿌요 쌍만을 넣어서 원하는 형태를 만들 수 있다. 12*20=240이니까 이 작업을 모든 줄에 대해 진행하면 된다.

여러 줄에 대해 작업을 진행할 때 유의해야 할 점은 무엇일까? 홀수 개의 뿌요를 터트리는 상황에서, 양 옆의 열에 기존에 만들어놓은 뿌요가 같이 터지지 않도록 하는 것이다. 이를 위해서 열을 채워넣는 순서가 중요한데, 양옆에 아무 것도 없는 것이 생각하기 편하므로, 채워 넣어야 하는 뿌요의 개수가 적은 열부터 채워 넣으면 별다른 추가 조치가 필요 없다.

```c++
#include <bits/stdc++.h>
using namespace std;
int b[21][21];
int h[21];
int R,C,K;
vector<pair<int,int>> v;
int f(int x){
    return (x%2==0?x/2:x/2+3);
}
void process(int x){
    if((R-h[x])%2==1){
        for(int i=R-1;i>h[x];i-=2){
            printf("%d %d %d %d\n",1,x+1,b[i-1][x],b[i][x]);
        }
        int tmp=(b[h[x]][x])%K+1;
        printf("%d %d %d %d\n",1,x+1,tmp,b[h[x]][x]);
        printf("%d %d %d %d\n",1,x+1,tmp,tmp);
        printf("%d %d %d %d\n",1,x+1,tmp,tmp);
    }
    else{
        for(int i=R-1;i>=h[x];i-=2){
            printf("%d %d %d %d\n",1,x+1,b[i-1][x],b[i][x]);
        }
    }
}
int main(){
    int cnt=0;
    scanf("%d%d%d",&R,&C,&K);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            scanf("%d",&b[i][j]);
        }
    }
    for(int j=0;j<C;j++){
        for(int i=R-1;i>=0;i--){
            if(b[i][j]==0){
                h[j]=i+1;
                cnt += f(R-h[j]);
                v.emplace_back(-h[j],j);
                break;
            }
        }
        if(h[j]==0){
            cnt += f(R-h[j]);
            v.emplace_back(-h[j],j);
        }
    }
    sort(v.begin(),v.end());
    printf("%d\n",cnt);
    for(int j=0;j<C;j++){
        int x=v[j].second;
        process(x);
    }
}

```

### 흐름 정리

- 한 줄일 때의 문제 풀기 **(필요조건)**
- 여러 줄인 경우로 일반화하기 **(충분조건)**

## [BOJ 18162. Adler32](https://www.acmicpc.net/problem/18162)

### 풀이

이 문제에서는 `(string, uint)`에서 `uint`로 가는 함수인 adler32를 소개하며, $adler32(ans, 0) = x,$ $adler32(ans, 1) = y$인 문자열 $ans$를 찾는 것이 목적이다.

일단 adler32 함수를 좀 분석해볼 필요가 있다. 문자열 $s=s_1s_2...s_n$에 대해 $adler32(s, a)$에서 $A$는 초기값 a에 각 문자 $s_i$들을 모두 합한 값이다. 즉, $A=a+\sum_{i=1}^n s_i (\text{mod} P)$이 된다. $B$는 중간 과정에서 $A$들의 합이 되는데, 문자별로 따져보자면 $s_i$는 $A$에 $s_i$가 포함되는 시점인 $i$번째부터 $n$번째까지에서 더해지므로 총 $n-i+1$번 더해진다. 그리고 초기값 $a$는 $s_1$과 함께 $n$번 더해진다. 그러므로 $B=an + \sum_{i=1}^n (n-i+1)s_i (\text{mod} P)$이다. (그림 참고)

그렇다면 $a=0$일 때의 $A, B$ 값 $A_0, B_0$와 $a=1$일 때의 $A, B$ 값 $A_1, B_1$을 비교하면 어떻게 될까? $A_1 = A_0 + 1 (\text{mod} P), B_1 = B_0 + n (\text{mod} P)$가 된다. 첫 번째 식은 출력이 존재하기 위해 필요한 제약 조건에 불과하지만, 두 번째 식은 (만약 문자열이 존재한다면) 출력으로 가능한 문자열에 대한 좋은 정보를 준다. 바로 **문자열의 길이 $n$**이다.

문제 풀이의 첫 스텝은 답이 존재할 필요충분조건을 찾는 것이다. 지금까지 나온 필요조건은 $A_0, B_0, A_1, B_1$이 모두 $0$ 이상 $P-1$ 이하여야 한다는 것과, $A_1 = A_0 + 1 (\text{mod} P)$뿐이다. 과연 이 조건이 충분조건이 될까? 결론부터 말하자면 **YES**이다. 그리고 Constructive 문제답게, 예시를 직접 만들어서 증명하면 된다.

$A_1 = A_0 + 1 (\text{mod} P), B_1 = B_0 + n (\text{mod} P)$의 두 식을 적용하고 나면 정보는 $A_0, B_0, n$만 남는다고 봐도 무방하다. 적당히 offset을 잘 조정하면, 문제는 $\sum_{i=1}^n f_i = X (\text{mod} P), \sum_{i=1}^n (n-i+1)f_i = Y(\text{mod} P)$를 만족하며 수열의 각 원소가 $0$ 이상 $25$ 이하의 정수인 수열 $f$를 찾는 것으로 변한다. 첫 번째 식이 그나마 좀 더 다루기 좋은 식이기에, 첫 번째 식을 만족시키면서 두 번째 식의 좌변을 얼마나 바꿔나갈 수 있는지를 확인해보자.

수열 $f$의 이웃한 두 항 $f_i, f_{i+1}$을 생각한다. 만약 $f_i$를 $1$ 감소시키고, $f_{i+1}$을 $1$ 증가시킨다면 첫 번째 식의 좌변 값은 그대로지만, 두 번째 식의 좌변 값은 $1$ 줄어든다. 이 과정을 얼마만큼 반복할 수 있을까? 수열의 맨 앞 항부터 채워나간 수열에서 시작해서, 수열의 맨 뒤 항부터 채워나간 수열까지 $1$씩 줄이는 것이 가능하다. 그 안에 $Y$가 있다면 문제 해결이다!

물론 여러 예외 사항이 있다. $n$이 작은 경우 범위 안에 $Y$가 들어오지 않거나, $X$의 값에 따라서는 수열의 맨 뒤 항부터 채워나간 수열 자체가 아예 존재하지 않을 수도 있다. 이 때는 $X$에 $P$를 계속 더해가 보거나, 수열의 길이를 $n+P$로 잡으면 된다는 사실을 확인할 수 있다.

실제 수열의 Construction 역시도 **1씩 감소시킨다**는 방식으로 하면 된다. 필자의 경우에는 약간의 효율성을 위해, 앞부터 채워나간 수열 $z..z?a..a$을 $?z..za..a, ?a..az..z, a..a?z..z$ 순서대로 만들어가면서 문자 하나씩을 옮기는 방식으로 구현하였다.

코드는 다음과 같다. 처리해야 하는 예외 케이스가 좀 많아서, 이 문제를 검수할 당시에 여러 (의도적으로) 틀린 코드들을 만들었다.

``` c++
#include <cstdio>
#include <string>
using namespace std;
const long long MOD = 65521;
char ans[150000];

int solve(long long p, long long q, long long d){
    for(int i=0;i<d;i++) ans[i]='a';
    ans[d]='\0';
    long long mx = 0LL;
    int idx;
    for(idx=0;(idx+1)*25<=q;idx++){
        ans[idx]+=25;
        mx += (d-idx)*25;
    }
    ans[idx]+=q%25;
    mx += (d-idx)*(q%25);
    if(q%25==0) idx--;
    if(mx < p) return 0;
    long long goal = (mx-p) % MOD;
    // Step 1: z-z?a-a -> ?z-za-a
    for(int i = idx; i > 0; i--){
        if(goal == 0){
            puts(ans);
            return 1;
        }
        if(goal >= 'z'-ans[i]){
            goal -= 'z'-ans[i];
            ans[i-1] = ans[i];
            ans[i] = 'z';
        }
        else{
            ans[i] += goal;
            ans[i-1] -= goal;
            goal = 0;
        }
    }
    // Step 2: ?z-za-a -> ?a-az-z
    for(int i = idx; i > 0; i--){
        if(goal == 0){
            puts(ans);
            return 1;
        }
        if(goal >= (d-1-idx)*25){
            goal -= (d-1-idx)*25;
            ans[d-1-idx+i] = 'z';
            ans[i] = 'a';
        }
        else{
            for(int j=0;j<25;j++){
                if(goal >= (d-1-idx)){
                    goal -= d-1-idx;
                    ans[d-1-idx+i]++;
                    ans[i]--;
                }
                else{
                    ans[i+goal]++;
                    ans[i]--;
                    goal = 0;
                    break;
                }
            }
        }
    }
    // Step 3: ?a-az-z -> a-a?z-z
    if(goal == 0){
        puts(ans);
        return 1;
    }
    else if(goal <= (d-1-idx)*(ans[0]-'a')){
        for(int j=0;ans[0]>'a';j++){
            if(goal >= (d-1-idx)){
                goal -= d-1-idx;
                ans[d-1-idx]++;
                ans[0]--;
            }
            else{
                ans[goal]++;
                ans[0]--;
                goal = 0;
                break;
            }
        }
    }
    if(goal == 0){
        puts(ans);
        return 1;
    }
    return 0;
}

int main(){
    long long x,y;
    scanf("%lld%lld",&x,&y);
    long long p=x/65536LL,q=x%65536LL,r=y/65536LL,s=y%65536LL;
    if(p >= MOD || q >= MOD || r >= MOD || s >= MOD){
        puts("-1");
        return 0;
    }
    if((MOD + s - q) % MOD != 1LL){
        puts("-1");
        return 0;
    }
    long long d = (MOD + r - p) % MOD;
    if(d==0) d += MOD;
    q = (q + 'a'*(MOD - d)) % MOD;
    p = (p + 'a'*(MOD*MOD - d*(d+1)/2)) % MOD;
    for(int t = q; t <= 25*d; t += MOD){
        if(solve(p, t, d)){
            return 0;
        }
    }
    d += MOD;
    for(int t = q; t <= 25*d; t += MOD){
        if(solve(p, t, d)){
            return 0;
        }
    }
}
```

### 흐름 정리

- $A, B$의 식을 문자열의 각 문자에 대한 식으로 나타내어 $A_0, A_1$, $B_0, B_1$ 간의 관계 찾기 **(필요조건)**
- 문자열의 길이가 정해졌을 때, $A$를 고정하고 $B$의 값을 $1$씩 변화시키는 방법 찾기 **(충분조건)**

## [BOJ 15948. 간단한 문제](https://www.acmicpc.net/problem/15948)

### 풀이

2018년 UCPC 본선에 출제된 문제이다. [2013 IMO 1번(Shortlist N2)](https://www.imo-official.org/problems/IMO2013SL.pdf)와 사실상 동일한 문제라서 대회 이후에도 여러 이야기가 나왔던 문제이다. 어째서 동일한 문제인가부터 살펴보자.

$$1+\frac{2^m-1}{n} = \frac{(A_1+B_1)(A_2+B_2)\cdots(A_m+B_m)}{B_1 B_2 \cdots B_m}$$

에서 $B_i = A_i x_i$로 놓으면

$$1+\frac{2^m-1}{n} = \frac{(1+x_1)(1+x_2)\cdots(1+x_m)}{x_1 x_2 \cdots x_m}$$

가 되어 2013 IMO 1번과 정확히 똑같은 문제가 된다. 이는 **충분조건**이지만, 문제 전체로 보자면 $A_i$들이 모두 $1$일 때도 답을 찾을 수 있어야 하므로 어떻게 보면 필요조건이라고 할 수도 있다. 이 문제를 푸는 접근은 두 가지로 나뉘는데, 두 접근 모두 소개하는 의미가 있다고 생각한다.

#### 접근 1. 부분 문제로 분리하기

**수학적 귀납법**은 명제를 증명하는 데 널리 쓰이는 방법 중 하나다. 이 문제를 수학적 귀납법으로 증명한다고 하면 가장 중요한 부분은, $1+(2^m-1)/n$을 $m$개의 $(1+x_i)/x_i$들의 곱으로 나타낸 결과를 이용하여 $1+(2^{m+1}-1)/n$을 $m+1$개의 $(1+x_i)/x_i$들의 곱으로 나타내는 것이다. 이렇게만 보면 사실 생각이 잘 안 날 수 있다. 그래서 다음과 같이 식을 조금 바꾸어서 살펴보자.

$$1+\frac{2^m-1}{t} = \frac{t+2^m-1}{t}$$

이렇게 놓고 보면, **분자가 분모보다 $2^m-1$만큼 크다**고 해석할 수 있다. 이 해석에 어떤 의미를 부여할 수 있을까? 분모와 분자에 $2$를 곱하면 $(2t+2^{m+1}-2)/2t$가 되어, **분자가 분모보다 $2^{m+1}-2$만큼 크다**가 된다. 그러면 여기서 분자를 $1$만큼만 늘리면 분자가 분모보다 $2^{m+1}-1$만큼 큰 수를 만들 수 있으므로 증명이 끝난다. 어떻게 늘릴지만 생각해보면 되는데, $2t/(2t-1)$을 곱하는 방법과 $(2t+2^{m+1}-1)/(2t+2^{m+1}-2)$를 곱하는 방법이 있다. 각각의 경우에서 분모를 $x_{m+1}$의 값으로 잡으면 된다. 경우를 나누는 기준은, 곱의 결과를 생각해보면 $n=2t-1$인 경우와 $n=2t$인 경우가 된다. 이렇게 증명은 끝난다.

```c++
#include <bits/stdc++.h>
using namespace std;
void solve(long long N, long long M){
    long long A;
    if(M == 0) return;
    if(N % 2LL == 0LL){
        scanf("%lld", &A);
        printf("%lld ", A * (N + (1LL << M) - 2LL));
        solve(N / 2LL, M - 1LL);
    }
    else{
        scanf("%lld", &A);
        printf("%lld ", A * N);
        solve((N+1LL) / 2LL, M - 1LL);
    }
}
int main(){
    long long N, M;
    scanf("%lld%lld",&N,&M);
    solve(N, M);
}

```

#### 접근 2. 분모와 분자의 차를 각 분수로 분배하기

위에서 했던, **분자가 분모보다 $2^m-1$만큼 크다**는 접근을 그대로 가져간다. $(a+1)/a \cdot (a+3)/(a+1) = (a+3)/a$를 보면, **"분자와 분모의 차를 더해 나간다"**는 생각이 가능하다. 그렇다면 $2^m-1=1+2+4+\cdots+2^{m-1}$이므로 $m$개의 분수가 각각 $1, 2, 4, \cdots, 2^{m-1}$을 담당하면 된다. 하지만 차를 만들기 위해서는 제약 조건이 따른다. $(a+k)/a$가 $(x+1)/x$ 꼴이기 위해서는 $a$가 $k$의 배수여야 한다. 그렇다면 $k$가 클수록 조건을 성립시키기 어려우니까, **가능한 한 가장 큰 수부터 더해 나가면 되지 않을까?** 그리고 이게 가능하다는 사실을 알 수 있다.

더할 수 있는 가장 큰 수는 그 수의 rightmost bit에 해당하는 자릿수 값인데, 이 값을 더하면 받아올림이 일어나면서 rightmost bit가 더 왼쪽으로 이동하여 더할 수 있는 값이 더 커진다. 이렇게 더해가다가 $2^{m-1}$에 도달하게 되면, 그 다음부터는 더하지 않은 수들을 큰 순서대로 더하게 된다. 이 이야기를 엄밀한 식으로 써놓은 것은 IMO Shortlist 공식 풀이의 Solution 2(53-54페이지)에서 확인할 수 있다.


```c++
#include <bits/stdc++.h>
using namespace std;
bool chkd[50];
void solve(long long N, long long M){
    long long A;
    for(int i=0;i<M;i++){
        for(long long j=M-1;j>=0;j--){
            if(!chkd[j] && N % (1LL<<j) == 0LL){
                scanf("%lld",&A);
                printf("%lld ", A * (N / (1LL<<j)));
                N += (1LL<<j);
                chkd[j] = true;
                break;
            }
        }
    }
}
int main(){
    long long N, M;
    scanf("%lld%lld",&N,&M);
    solve(N, M);
}

```

흥미롭게도, 두 풀이는 본질적으로 같다. 정확히는, $x_i$의 집합이 동일하다.

### 흐름 정리

- $A_i$가 모두 $1$일 때의 문제로 환원 **(충분조건)**
1. $m-1$일 때의 결과를 $m$에 적용하여, 주어진 문제를 부분 문제로 분해 **(충분조건)**
2. 분모에  $1, 2, 4, \cdots, 2^{m-1}$를 더해가는 것으로 생각하기 **(충분조건)**



## [BOJ 17525. Enumeration](https://www.acmicpc.net/problem/17525)

### 풀이

2019년 ICPC Seoul Regional 인터넷 예선의 G번 문제로 출제된 문제이다. 당시 전체 팀 중 단 두 팀만이 풀었던 문제이고, 그래서 문제가 온라인 저지에 공개된 이후에도 거의 풀리지 않았는데, 이 문제는 약간 과대평가되어 있다고 생각한다.

알파벳 n개 중 k개의 서로 다른 문자를 골라 오름차순으로 정렬한 k-word들을, 이웃한 k-word들끼리 문자 하나만 다르도록(나머지를 모두 공통으로 가지고 있도록) 나열하는 문제다. 이 문제도 사실 그래프 문제로 환원할 수 있다. 각 k-word들을 정점으로 보고 조건에 따라 이웃한 것들을 연결하여 나오는 그래프에서 해밀턴 경로를 찾는 문제가 된다. 하지만 이 변환은 그리 좋지는 않은데, [해밀턴 경로를 찾는 알고리즘](https://en.wikipedia.org/wiki/Hamiltonian_path_problem)은 NP-Complete 문제로, 다항 시간 안에 동작하는 알고리즘이 없다. 이 문제에서는 그래프에 대한 생각은 조금 접어두고, **문제를 부분 문제로 나누려는 시도**를 해야 한다.

n,k가 작은 케이스를 먼저 살펴보자. n=3인 경우, k가 1이든 2든, 모든 k-word들의 쌍이 서로 이웃해도 되기 때문에, 어떻게 나열해도 된다. 이제 n=4, k=2인 경우를 보자. 알파벳이 abcd일 때, k-word들을 사전순으로 나열하면 ab, ac, ad, bc, bd, cd이다. 사전순으로 나열해놓고 보면, 앞의 3개의 word들은 a가 포함되어 있고, 뒤의 3개는 그렇지 않다. 그렇다면 **뒤의 3개는 알파벳이 bcd인 (n=3, k=2)의 케이스와 같아진다!** 앞의 3개는 어떨까? a를 빼놓고 보면 b,c,d가 되는데, 어차피 a가 공통이므로 a를 제외한 나머지 부분의 문자열이 조건을 만족하고 있다면 a를 포함시킨 문자열들도 조건을 만족한다. 따라서 **앞의 3개는 알파벳이 bcd인 (n=3, k=1)의 케이스에 a를 덧붙인 케이스와 같아진다!** 일반적으로, **(n,k)일 때의 답은 (n-1, k-1)의 답에 문자 하나를 덧붙인 것과 (n-1,k)의 답을 이어 붙이면 된다!** 예를 들면 위의 경우 (b,c,d)에 a를 붙인 (ab, ac, ad)와, (bd, bc, cd)를 이어서 (ab, ac, ad, bd, bc, cd)의 답을 만들 수 있다.

문제에서는 시작 단어와 끝 단어가 정해져 있는데, 그렇게 큰 문제는 되지 않는다. 일단 부분 문제를 나누는 기준이 되는 문자 하나를 잘 정하면 되는데, $S$에는 있고 $T$에 없는 문자면 충분하다. 전체 문제를 푸는 함수 f를 다음과 같이 정의한다.

- $f(\Sigma, k, S, T, prefix):$ 알파벳 $\Sigma$의 모든 k-word를 이웃한 k-word끼리 문자 하나만 다르도록, 그리고 S에서 시작하여 T에서 끝나도록 나열한 것에, prefix를 덧붙인(합집합한) 것

이렇게 정의했을 때, 먼저 $k=1$이거나 $k=n-1$인 경우는 아무렇게나 나열해도 되므로 $S$로 시작해서 $T$로 끝나는 아무 sequence면 된다. 그렇지 않은 경우, $S$에 있고 $T$에 없는 문자 $x$에 대해, $f(\Sigma, k, S, T, prefix) = f(\Sigma-x, k-1, S-x, Y, prefix+x) + f(\Sigma-x, k, Z, T, prefix)$가 된다. 여기서 $Y, Z$를 어떻게 정할지가 살짝 문제가 될 수 있는데, 다음 세 가지 조건을 만족해야 한다.

- $S-x$는 $Y$와 다르다.
- $Z$는 $Y$의 모든 문자를 포함한다.
- $Z$는 $T$와 다르다.

$Y$는 정말로 $S-x$가 아닌 아무 문자열을 정하면 된다. $Z$는 $Y$의 모든 문자들을 포함시킨 다음, $T$와 같지 않도록 남은 문자 중 아무 문자나 포함시키면 된다. 문자열을 고를 때, 선택지가 두 개 이상만 있으면 고를 수 있음이 보장된다. 선택지가 두 개 이상 있음을 확인하는 것은 어렵지 않다.

#### 구현 시 사소한 유의사항

이 글을 쓰는 시점(2020. 8. 18)의 BOJ 스페셜 저지는 줄의 맨 뒤 공백을 허용하지 않고 있다. 혹시 맞게 구현했는데 계속 틀리다는 채점 결과이 나온다면 이것을 확인해보자.

``` c++
#include <bits/stdc++.h>
using namespace std;
vector<string> v;
string sorted(string s){
    sort(s.begin(),s.end());
    return s;
}
void solve(int n, int k, string pfx, string alp, string st, string ed){
    if(k == 1){
        v.push_back(pfx + st);
        for(int i=0;i<n;i++){
            string tmp = alp.substr(i,1);
            if(tmp != st && tmp != ed) v.push_back(pfx + tmp);
        }
        v.push_back(pfx + ed);
    }
    else if(k == n-1){
        v.push_back(pfx + st);
        for(int i=0;i<n;i++){
            string tmp = alp.substr(0,i) + alp.substr(i+1);
            if(tmp != st && tmp != ed) v.push_back(pfx + tmp);
        }
        v.push_back(pfx + ed);
    }
    else{
        for(int i=0;i<n;i++){
            if(st.find(alp[i]) != string::npos && ed.find(alp[i]) == string::npos){
                string new_alp = alp.substr(0,i) + alp.substr(i+1);
                int x = st.find(alp[i]);
                string new_st = st.substr(0,x) + st.substr(x+1);
                string new_ed = new_alp.substr(0,k-1);
                string newnew_st = new_ed + new_alp[k-1];
                if(new_st == new_ed){
                    new_ed = new_alp.substr(n-k,k-1);
                    newnew_st = new_alp[n-k-1] + new_ed;
                    if(newnew_st == ed){
                        newnew_st = new_alp[n-k-2] + new_ed;
                    }
                }
                else{
                    if(newnew_st == ed){
                        newnew_st = new_ed + new_alp[k];
                    }
                }
                solve(n-1,k-1,pfx+alp[i],new_alp,sorted(new_st),sorted(new_ed));
                solve(n-1,k,pfx,new_alp,sorted(newnew_st),ed);
                break;
            }
        }
    }
}

int main(){
    int n,k;
    string alp, st, ed;
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cin >> n >> k;
    cin >> alp >> st >> ed;
    solve(n,k,"",alp,st,ed);
    cout << v.size() << "\n";
    for(int i=0;i<v.size();i++){
        cout << (i?" ":"") << sorted(v[i]);
    }
}

```

### 흐름 정리

- 특정 문자가 들어간 k-word와 그렇지 않은 k-word로 나누어서 부분문제 풀기 **(충분조건)**
    - 구분 기준으로 사용할 문자 정하기 **(필요충분조건)**

## 마무리하며

Constructive 문제는 아이디어를 생각해내기가 많이 어려울 뿐 아이디어의 이해나 구현은 쉬운 경우도 많아서, 막상 풀이를 듣고 나면 허무한 경우가 많다. 아직 풀이를 안 봤으나 문제가 너무 안 풀린다면, 그나마 "흐름 정리" 부분이 대략적인 설명을 담고 있으니 이 부분을 읽어보면 좋을 것 같다.

Constructive 문제는 구현 능력보다는 **사고력**이 중요한 문제이다. 사실 Constructive 문제는 "알고리즘 대회 쪽에서 어느 정도까지의 수학을 낼 수 있는가?" 라는 질문에서 경계선을 넘나들기도 하는 분야이다. 여기서 소개한 문제들이 대부분 학생이나 졸업생 분들이 만든 문제들이기는 하지만, 작년 ICPC에서의 고난도 문제로도(Enumeration) 출제되었다. 분야 자체가 문제라기보다는, 문제들이 너무 다양하고 그 풀이가 Case-By-Case로 많이 차이가 나다 보니 다른 여러 분야들과도 관련되어서 이야기가 나오는 것 같다.

자주 나오는 분야는 아니기에 과하게 투자할 필요까지는 없지만, 그래도 Constructive는 적당한 수준의 대비를 할 필요는 있는 분야라고 생각한다. 다른 걸 다 떠나서, **오랜 생각 끝에 아이디어를 떠올려냈을 때의 쾌감**을 느껴보고 싶지 않은가? Constructive 문제들은 여러분에게 새로운 어려움의 맛(?)과 희열을 선사해줄 수 있을 것이다.
