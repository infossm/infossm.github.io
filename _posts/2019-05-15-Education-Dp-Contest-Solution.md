---
layout: post
title: "Atcoder Educational DP contest 풀이"
date: 2019-05-15 13:00
author: kjp4155
tags: [dynamic-programming, editorial]
---
<h1> </h1>

Atcoder에서 열렸던 [Educational DP contest](https://atcoder.jp/contests/dp/tasks) 문제들에 대한 풀이입니다. 자주 나오는 다이나믹 프로그래밍 유형들을 연습할 수 있는 좋은 셋이라고 생각합니다.

문제마다 간략한 풀이를 작성했습니다. 코드는 Atcoder에서 자유롭게 열람이 가능하므로 문제마다 링크를 달아두었습니다. 어느정도는 난이도 순서대로 정렬되어 있지만, 사람마다 느끼는 난이도가 다를 수 있습니다.

## A. Frog 1

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_a)

$$d_i :$$ $$i$$ 번째 Stone까지 도달하는 최소 비용
{: style="text-align: center"}

$$d_i = min(d_{i-1} + |h_i - h_{i-1}|,\ d_{i-2} + |h_i - h_{i-2}|)$$

위와 같은 점화식으로 간단히 해결할 수 있는 쉬운 문제입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241581)

## B. Frog 2

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_b)

위 Frog 1 문제와 거의 비슷하게, 

$$d_i :$$ $$i$$ 번째 Stone까지 도달하는 최소 비용
{: style="text-align: center"}

$$d_i = min( d_{i-k} + |h_i - h_{i-k}| ) \ (1 \leq k \leq K)$$ 

와 같이 쉽게 해결할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241589)

## C. Vacation

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_c)

$$ A_i :$$ $$i$$번째 날에 A행동을 했을 때, $$i$$번째 날까지 얻을 수 있는 최대 happiness
{: style="text-align: center"}

$$ B_i :$$ $$i$$번째 날에 B행동을 했을 때, $$i$$번째 날까지 얻을 수 있는 최대 happiness
{: style="text-align: center"}

$$ C_i :$$ $$i$$번째 날에 C행동을 했을 때, $$i$$번째 날까지 얻을 수 있는 최대 happiness
{: style="text-align: center"}

위와 같이 DP 식을 정의하면, 점화식은 다음과 같이 간단히 유도됩니다.

$$ A_i = max( B_{i-1}, C_{i-1} ) + a_i $$
{: style="text-align: center"}

$$ B_i = max( C_{i-1}, A_{i-1} ) + b_i $$
{: style="text-align: center"}

$$ C_i = max( A_{i-1}, B_{i-1} ) + c_i $$
{: style="text-align: center"}

위 점화식을 $$i$$가 커지는 순서에 따라서 채워주면 답은 $$max(A_N, B_N, C_N)$$ 으로 구할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241638)

## D. Knapsack 1

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_d)

유명한 냅색 문제입니다.

$$d_{i,j} :$$ $$i$$ 번째 item까지 고려했을 때, $$j$$의 무게만큼 사용했을 때 얻을 수 있는 최대 value.
{: style="text-align: center"}

라고 정의합시다. 그러면 $$i$$번째 item을 넣느냐 마느냐에 따라서 다음과 같은 점화식을 얻을 수 있습니다.

$$d_{i,j} = max( d_{i-1,j-w_i} + v_i , d_{i-1,j})$$

따라서 2차원 배열을 위 점화식에 따라 채우기만 하면 되므로 $$O(NW)$$ 시간복잡도에 해결할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241684)

## E. Knapsack 2

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_e)

위 Knapsack 1 문제와 비슷하지만, 이번에는 무게 $$W$$의 범위가 크고, value $$v$$의 범위가 작다는 것에 착안해야 합니다. DP 정의를 조금 바꿔서

$$d_{i,j} :$$ $$i$$ 번째 item까지 고려했을 때, $$j$$의 value만큼 얻었을 때 가능한 최소 무게.
{: style="text-align: center"}

라고 정의합시다. 그러면 점화식은 위와 비슷하게 $$i$$번째 item을 넣느냐 마느냐에 따라서 다음과 같이 작성할 수 있습니다.

$$d_{i,j} = min( d_{i-1,j-v_i} + w_i , d_{i-1,j})$$

이제 위 점화식에 따라서 2차원 DP 배열을 모두 계산한 뒤, DP값이 $$W$$보다 작은 원소들 중 최대의 $$j$$를 고르는 것이 답이 됩니다.

Knapsack 1/2 문제처럼 인자의 범위에 따라서 DP 정의를 유동적으로 바꿔야 하는 경우가 많습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241716)

## F. LCS

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_f)

유명한 LCS 문제입니다. 

$$d_{i,j} :$$ $$s$$의 $$i$$번째 문자, $$t$$의 $$j$$번째 문자까지 고려했을 때 최대 LCS 길이.
{: style="text-align: center"}

로 정의하면, $$s_i$$와 $$t_j$$가 같은지 다른지에 따라서 다음과 같은 점화식을 생각할 수 있습니다.

$$ d_{i,j} = \begin{cases} max(d_{i-1,j},\ d_{i,j-1}) \\  max(d_{i-1,j},\ d_{i,j-1},\ d_{i-1,j-1}+1)\ \ (if s_i == t_j) \end{cases} $$ 

이러한 2차원 DP배열을 채우고 나면, 최종적인 LCS길이는 $$d_{\lvert s \rvert, \lvert t \rvert}$$ 가 될 것입니다. 문제를 해결하려면 답을 역추적해야 되는데, 이는 해당 DP값이 $$(i-1,j)$$, $$(i,j-1)$$, $$(i-1,j-1)$$ 의 세 경우 중 어느 곳에서 왔는지 역으로 따라가면 됩니다. $$(i-1,j-1)$$의 transition이 발생할 때마다 해당 문자가 선택되었다는 뜻이므로 답에 추가해주면 됩니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241758)

## G. Longest

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_g)

$$d_i :$$ $$i$$번 노드에서 시작하는 최장 경로 길이.
{: style="text-align: center"}

와 같이 정의한 뒤, Memoization을 이용해 Top Down 방식으로 각 DP값을 계산해 주면 됩니다. Cycle이 존재하지 않음이 보장되어 있으므로 $$O(N+M)$$ 시간복잡도에 해결이 가능합니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241772)

## H. Grid 1

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_h)

$$d_{i,j} :$$ $$(i,j)$$위치에 도달할 수 있는 경우의 수.
{: style="text-align: center"}

와 같이 정의하면 DP 점화식은 다음과 같이 자연스럽게 유도됩니다.

$$ d_{i,j} = d_{i-1,j} + d_{i,j-1} $$ (두 위치 중 벽이 아닌 곳들만 더해주어야 합니다)
{: style="text-align: center"}

이제 이 점화식을 통해 DP table을 $$O(NM)$$ 시간에 계산하면 문제를 해결할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241781)

## I. Coins

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_i)

$$d_{i,j} :$$ $$i$$번째 동전까지 고려했을 때, head가 $$j$$ 개 나올 확률
{: style="text-align: center"}

와 같이 정의합시다. 그러면 각 동전마다 head가 나올 확률이 $$p_i$$ 이므로 다음과 같은 점화식이 성립합니다.

$$d_{i,j} = p_i d_{i-1,j-1} + (1-p_i) d_{i-1,j}$$ 

이 점화식을 이용해 DP table의 값을 모두 계산한 뒤, $$d_{N,j}$$ 들 중 $$j>N-j$$ 인 값들을 모두 더해주면 곧 답으로 원하는 확률이 나옵니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241794)

## J. Sushi

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_j)

$$d_{i,j,k} :$$ 초밥이 $$1,2,3$$ 개 남은 접시가 각각 $$i,j,k$$개 남아있을 때, 다 먹을때까지 필요한 operation 횟수의 기댓값
{: style="text-align: center"}

와 같이 정의합시다. DP transition은 주사위를 굴렸을 때 나올 수 있는 경우에 따라서 경우를 나눌 수 있습니다. 먼저, 초밥이 하나도 없는 접시의 번호가 나오는 경우를 생각해 봅시다. 이 경우에는 $$d_{i,j,k}$$ 의 계산에서 $$d_{i,j,k}$$를 참조하기 때문에 cycle이 생기게 됩니다. 따라서 초밥이 적어도 하나 있는 접시의 번호가 나올 때까지 주사위를 굴리는 기댓값을 따로 구해주어야 합니다. 이는 [Geometric Distribution](https://en.wikipedia.org/wiki/Geometric_distribution)이라는 것을 알 수 있는데, 원하는 기댓값은 곧 $$ \frac {N-(i+j+k)} {i+j+k} $$ 이라는 것이 잘 알려져 있습니다. 따라서 이 값을 먼저 더해준 뒤, 초밥이 $$1,2,3$$ 개 남은 접시가 나오는 경우를 처리해주면 됩니다. 결론적으로 다음과 같은 DP 점화식을 얻을 수 있습니다. 

$$ d_{i,j,k} = \frac {N-(i+j+k)} {i+j+k} + \frac {i} {i+j+k} d_{i-1,j,k} + \frac {j} {i+j+k} d_{i+1,j-1,k} + \frac {k} {i+j+k} d_{i,j+1,k-1} $$

이 점화식에 따라 DP table을 계산해준 뒤 답을 구할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4246720)

## K. Stones

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_k)

기초적인 게임이론 문제입니다.

$$d_i :$$ $$i$$개의 돌로 게임을 시작하면 선공이 이기는가?
{: style="text-align: center"}

와 같이 정의합시다. 만약 $$d_{i-k}$$ 가 false인 $$k \in A$$ 가 존재한다면 $$d_i$$ 는 true가 되고, 이러한 $$k$$가 존재하지 않는다면 $$d_i$$는 false가 됩니다.

따라서 $$O(NK)$$ 시간복잡도로 모든 $$d_i$$값을 계산해 주면 $$d_N$$의 값에 따라 답을 결정할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241917)

## L. Deque

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_l)

Taro의 전략은 $$X-Y$$를 최대화하는 것이고, Jiro의 전략은 $$Y-X$$를 최대화하는 것입니다. 따라서 둘 모두 (내 점수)-(상대방 점수) 를 최대화하고 싶어하는 것을 알 수 있습니다. 이를 통해 다음과 같은 DP 정의를 생각해 봅시다.

$$d_{l,r} :$$ $$[l...r]$$ 구간의 수가 남아있을 때, (내 점수)-(상대 점수)의 최댓값.
{: style="text-align: center"}

위와 같이 정의한 뒤 DP transition을 어떻게 할 수 있을지 생각해 봅시다. 내가 $$a_l$$을 골랐다면 최종적으로 (내 점수)-(상대 점수)는 $$a_l - d_{l+1,r}$$ 이 되고, $$a_r$$을 골랐다면 $$a_r - d_{l,r-1}$$이 됩니다. 따라서 이 두 값 중 큰 값을 고르는 방식으로 점화식을 정의할 수 있습니다.

위 transition으로 DP table을 모두 채우면 $$d_{1,N}$$이 최종적인 답이 됩니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4241940)

## M. Candies

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_m)

$$d_{i,j} :$$ $$1,...,i$$ 번째 아이들에게 총 $$j$$개의 사탕을 나누어주는 방법의 수.
{: style="text-align: center"}

로 정의합시다. 그러면 $$i$$번째 아이에게 사탕을 몇 개 나누어주었느냐에 따라 다음과 같은 점화식이 성립합니다.

$$d_{i,j} = \sum_{k=0}^{a_i} d_{i-1,j-k}$$ 

그런데 이를 naive 하게 계산하면 시간복잡도 $$O(NK^2)$$으로 충분히 빠르지 못합니다. 여기서 유명한 구간합 테크닉을 사용하면 되는데,

$$s_{i,j} = \sum_{k=0}^{K} d_{i,k}$$

와 같이 구간합 배열을 정의한 뒤, 이를 사용해서 $$d_{i,j} = s_{i-1,j-a_i}$$ 와 같이 빠르게 계산할 수 있습니다. 따라서 시간복잡도는 $$O(NK)$$가 됩니다. 최종적인 답은 $$d_{N,K}$$가 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4246203)

## N. Slimes

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_n)

[파일 합치기](https://www.acmicpc.net/problem/11066) 와 완전히 동일한 문제입니다.

$$d_{l,r} :$$ $$l,...,r$$ 번째 파일을 하나로 합치는 최소 비용.
{: style="text-align: center"}

으로 정의합시다. $$l,...,r$$ 번째 파일이 하나로 합쳐지는 과정의 마지막 순간을 생각해 봅시다. 어떤 $$k$$가 존재해서, $$l,...,k$$번째 파일들과, $$k+1,...,r$$번째 파일들이 각각 하나의 파일이 되어있는 상태에서 두 파일이 최종적으로 합쳐질 것입니다. 따라서 모든 가능한 $$k$$에 대해 시도해보는 것으로 최소 비용을 계산할 수 있습니다. 

$$d_{l,r} = \min_{k=l}^{r-1} d_{l,k} + d_{k+1,r}$$

위와 같은 점화식을 순서에 맞게 잘 채워주면 최종 답은 $$d_{1,N}$$ 이 됩니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4246275)

## O. Matching

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_o)

$$N$$ 명의 여성들 중, 매칭된 여성들을 비트마스크로 나타낸 집합을 $$m$$으로 표현합니다. 다음과 같은 DP 정의를 생각합니다.

$$d_{i,m} :$$ $$0...i$$번 남성까지 매칭시켰을 때, 매칭된 여성들의 집합이 $$m$$인 경우의 수. 
{: style="text-align: center"}

그러면 $$i$$번 남성이 어떤 여성과 매칭되느냐에 따라 다음과 같은 DP 점화식을 세울 수 있습니다.

$$d_{i,m} =  \sum_{j=0}^{N-1}\begin{cases} d_{i-1, m-(1<<j)} \ \ \ \ (j \notin m \& a_{i,j} = 1 ) \\ 0 \ (otherwise) \end{cases} $$ 
{: style="text-align: center"}

위 식을 $$i, m$$이 커지는 순서로 잘 계산해 주면 문제를 해결할 수 있습니다. $$O(N^2 2^N)$$ 시간복잡도로는 약간의 컷팅이 더 필요할 수 있습니다. 코드를 보시면 한가지 방법이 나와 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4246962)

## P. Independent Set

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_p)

가장 기본적인 Tree DP 문제입니다. 

$$d_{x} :$$ x를 반드시 색칠할 때, x를 루트로 하는 서브트리를 색칠하는 방법의 수.
{: style="text-align: center"}

$$e_{x} :$$ x를 반드시 색칠하지 않을 때, x를 루트로 하는 서브트리를 색칠하는 방법의 수.
{: style="text-align: center"}

와 같이 두가지 경우로 나눠서 생각합시다.

$$x$$가 색칠되면, 바로 밑의 자식들은 반드시 색칠되지 않아야 할 것입니다. $$x$$가 색칠되지 않으면, 바로 밑의 자식들은 색칠되거나, 안되거나 두 경우 모두 가능할 것입니다. 따라서 다음과 같은 점화식이 쉽게 유도됩니다. $$E[x]$$는 노드 $$x$$의 자식들의 집합입니다.

$$d_x = \displaystyle \prod_{y \in E[x]}^{} e_y $$

$$e_x = \displaystyle \prod_{y \in E[x]}^{} (d_y + e_y)$$

이제 위 값들을 DFS finish time 기준으로 채워 주면 됩니다. 편의상 1번 노드를 루트라고 하면 최종 답은 $$d_1 + e_1$$이 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4247019)

## Q. Flowers

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_q)

$$h_1, h_2, ... , h_N$$이 주어졌을 때, 증가 부분 수열 중 그 합이 최대가 되도록 고르는 문제입니다.

다음과 같은 DP 정의를 생각해 봅시다.

$$d_i :$$ $$h_i$$에서 끝나는 증가 부분 수열 중 최대 합. 
{: style="text-align: center"}

그러면 자연스럽게 아래처럼 점화식을 유도할 수 있습니다.

$$d_i = \max_{j=1}^{i-1} d_j + h_i $$ $$(if h_j \leq h_i)$$
{: style="text-align: center"}

그런데 이를 $$i$$가 증가하는 순서대로 naive하게 계산하면 $$O(N^2)$$ 시간복잡도로 충분히 빠르지 못합니다.

$$h_i$$들을 오름차순으로 정렬한 뒤, 증가하는 순서대로 본다고 생각해 봅시다. $$h_i$$가 증가하는 순서대로 보고 있기 때문에, $$d_i$$를 계산하려면 이미 등장한 수들 중 $$i$$보다 왼쪽에 있는 수들만 고려하면 될 것입니다. 또한, DP 점화식을 살펴보면 간단한 range max 쿼리 형태라는 것을 알 수 있습니다. 따라서 세그먼트 트리를 이용해 효율적으로 DP 값을 계산할 수 있습니다.

$$h_i$$가 커지는 순서대로 $$d_i$$를 계산할 때 세그먼트 트리의 range max 쿼리를 이용하고, $$d_i$$가 계산될 때마다 세그먼트 트리에 값을 update 해주는 방식으로 $$O(NlgN)$$ 시간복잡도에 모든 DP 값을 계산할 수 있습니다. 최종적인 답은 $$d_N$$이 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4248397)

## R. Walk

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_r)

$$d_{i,k}:$$ $$i$$번 노드에서 끝나는 길이 $$k$$짜리 경로의 가짓수.
{: style="text-align: center"}

처럼 정의합시다. 다음과 같이 DP transition을 생각할 수 있습니다.

$$d_{i,k} =  \sum_{j=1}^{N}\begin{cases} d_{j, k-1} \ \ \ \ (a_{j,i} = 1 ) \\ 0 \ (a_{j,i} = 0) \end{cases} $$ 
{: style="text-align: center"}

이 식의 형태를 생각해 봅시다. 길이가 $$k$$인 $$i$$에서 끝나는 경로의 가짓수는 결국 길이가 $$k-1$$인 경로들 중, 마지막 노드에서 $$i$$로 향하는 edge가 존재하는 것들의 합이 된다는 것입니다. 이를 이용해 다음과 같이 행렬을 이용한 형태로 식을 표현할 수 있습니다.

$$     \begin{bmatrix}
    d_{1,k} \\
    d_{2,k} \\
    ... \\
    d_{N,k} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    a_{1,1} & a_{1,2} & ... & a_{1,N} \\
    a_{2,1} & a_{2,2} & ... & a_{2,N} \\
    ... & ... & ... & ... \\
    a_{N,1} & a_{N,2} & ... & a_{N,N} \\
    \end{bmatrix}
    
    \begin{bmatrix}
    d_{1,k-1} \\
    d_{2,k-1} \\
    ... \\
    d_{N,k-1} \\
    \end{bmatrix}
    
$$

따라서 각 $$d_{i,K}$$는 다음과 같이 계산할 수 있습니다.

$$ \begin{bmatrix}
    d_{1,K} \\
    d_{2,K} \\
    ... \\
    d_{N,K} \\
    \end{bmatrix}
    = 
    { 
        \begin{bmatrix}
        a_{1,1} & a_{1,2} & ... & a_{1,N} \\
        a_{2,1} & a_{2,2} & ... & a_{2,N} \\
        ... & ... & ... & ... \\
        a_{N,1} & a_{N,2} & ... & a_{N,N} \\
        \end{bmatrix}
    } ^ {K}

    \begin{bmatrix}
    d_{1,0} \\
    d_{2,0} \\
    ... \\
    d_{N,0} \\
    \end{bmatrix}

    =
    { 
        \begin{bmatrix}
        a_{1,1} & a_{1,2} & ... & a_{1,N} \\
        a_{2,1} & a_{2,2} & ... & a_{2,N} \\
        ... & ... & ... & ... \\
        a_{N,1} & a_{N,2} & ... & a_{N,N} \\
        \end{bmatrix}
    } ^ {K}

    \begin{bmatrix}
    1 \\
    1 \\
    ... \\
    1 \\
    \end{bmatrix}
$$

이제 거듭제곱을 $$O(lg)$$시간복잡도에 수행하는 알고리즘을 이용해 adjacency matrix의 $$K$$승을 계산해주기만 하면 답을 구할 수 있습니다. 최종적인 시간복잡도는 $$O(N^3lgK)$$ 가 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4250060)

## S. Digit Sum

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_s)

큰 자릿수부터 순서대로 채워 내려간다고 생각해 봅시다.

$$ d_{i,j} :$$ $$i$$번째 자릿수까지 $$K$$와 일치했고, 자릿수 합을 $$D$$로 나눈 나머지는 $$j$$
{: style="text-align: center"}

$$ e_{i,j} :$$ $$i$$번째 자릿수 또는 그 이전에서 $$K$$보다 작아졌고, 자릿수 합을 $$D$$로 나눈 나머지는 $$j$$
{: style="text-align: center"}

이제 $$i$$번째 자릿수에 어떤 수를 채워넣느냐에 따라 DP transition을 생각할 수 있습니다. 

$$ d_{i,j} $$ 의 경우 반드시 $$i$$번째 자리에 $$K_i$$를 넣어야 할 것이고, $$e_{i,j}$$의 경우에는 $$0..9$$를 자유롭게 넣을 수 있습니다. 다만 여기서, $$i$$번째 자리에서 처음으로 $$K$$보다 작아지는 경우만 잘 고려하면 됩니다. 따라서 다음과 같은 DP transition으로 계산할 수 있습니다.

$$ d_{i,j} = d_{i-1, j-K_i}$$
{: style="text-align: center"}

$$ e_{i,j} = \sum_{k=0}^{9} e_{i-1, j-k} + \sum_{k=0}^{K_i-1} d_{i-1, j-k} $$
{: style="text-align: center"}

$$e_{i,j}$$를 계산할 때, 두번째 항이 바로 $$i$$번째 위치에서 처음으로 $$K$$보다 작아지는 경우를 처리하는 것입니다. 뺄셈 연산을 할 때 음수가 되는 경우를 주의합시다.

위 식에 따라서 DP값을 모두 계산해 주면 $$ d_{\|K\| , 0 } + e_{\|K\|, 0} - 1 $$이 최종적인 답이 됩니다. 1을 빼주는 이유는 0을 답에서 제외해야 하기 때문입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4250832)

## T. Permutation

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_t)

Permutation의 앞 수부터 하나씩 정해간다고 생각해 봅시다. 다만, $$ p_1, p_2, ..., p_n $$을 $$ 1, 2, ... , n$$의 수를 사용해서 만든다고 생각합시다.

$$ d_{n,k} :$$ $$n$$번째 수까지 $$1, 2,... n$$를 이용해 정했고, 마지막 수 $$p_n$$이 $$k$$인 방법의 수. 
{: style="text-align: center"}

이제 DP transition을 생각하기 위해 $$ d_{n,k} $$ 를 어떻게 채울 것인가 생각해 봅시다. $$ p_1, p_2, ... p_{n}$$을 $$1,2,...,n$$을 이용해 만드는 상황입니다. 마지막 수가 $$k$$라는 것은, $$k$$를 제외한 나머지 $$n-1$$개의 수 $$1,...,k-1,k+1,...n-1$$이 $$p_1, p_2, ..., p_{n-1}$$을 구성하고 있다는 의미입니다. 이제 이러한 $$n-1$$개의 수들을 상대적인 크기에 따라 다시 $$1, ..., n-1$$으로 압축해서 생각할 수 있습니다. 다시 말해, 첫 $$n-1$$개의 수를 $$1,...,n-1$$으로 고정시키고, 추가되는 $$n+1$$번째 수를 $$0.5, 1.5, ... n-0.5$$처럼 생각한다는 것입니다. 그러면 다음과 같은 DP transition이 가능합니다.


$$d_{n,k} =  \begin{cases}
\displaystyle \sum_{i=1}^{k-1} d_{n-1,i} \ \ \ (s_{n-1} \ =  \ <) \\ 
\displaystyle \sum_{i=k}^{n-1} d_{n-1,i} \ \ \ (s_{n-1} \ = \ >)
\end{cases} $$ 
{: style="text-align: center"}

위 식을 그대로 계산하면 $$O(N^3)$$이지만, 구간합 배열을 간단히 응용해 $$O(N^2)$$에 모든 DP값을 계산할 수 있습니다. 최종적인 답은 $$\sum_{i=1}^{N} d_{N,i}$$가 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4251327)


## U. Grouping

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_u)

토끼들의 집합 $$M$$을 비트마스크로 나타낸 것을 소문자 $$m$$으로 표현합시다.

먼저 다음과 같은 배열을 $$O(2^{N}N^2)$$ 시간에 계산해 둡시다.

$$ s_m :$$ 집합 $$M$$에 해당하는 토끼들이 한 그룹으로 묶였을 때, 그 그룹으로 얻는 점수 
{: style="text-align: center"}

이제 다음과 같은 DP 식을 생각합시다.

$$ d_m :$$ 집합 $$M$$에 해당하는 토끼들을 잘 나누어서, 최대로 얻을 수 있는 점수
{: style="text-align: center"}

집합 $$M$$의 임의의 부분집합 $$K$$를 모두 시도해보는 것으로 다음과 같은 점화식을 생각할 수 있습니다.

$$ d_m = \max_{K \subset M} (d_{m \backslash k} + d_k) $$

이를 순서를 잘 정해서 채워주기만 하면 최종적인 답은 $$d_{2^N-1}$$가 됩니다.

집합 $$M$$으로 $$2^N$$개가 가능하고, 각 $$d_m$$을 채울 때 최대 $$2^N$$개의 부분집합을 시도해야 해서 총 시간복잡도가 $$O(2^{2N})$$이 된다고 생각할 수 있으나, 실제로는 부분집합의 개수가 그렇게 많지 않기 때문에 훨씬 빠르게 작동하는 것을 알 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4250239)


## V. Subtree

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_v)

문제의 조건은 결국 색칠된 정점들이 하나의 연결된 component를 이루어야 한다는 것입니다. 다음과 같은 두 DP 정의를 생각합시다.

$$ down_x :$$ 정점 $$x$$를 반드시 색칠하고, $$x$$를 루트로 하는 서브트리를 색칠하는 방법의 수.
{: style="text-align: center"}

$$ up_x :$$ 정점 $$x$$를 반드시 색칠하고, $$x$$의 위쪽 서브트리를 색칠하는 방법의 수.
{: style="text-align: center"}

여기서 $$x$$의 위쪽 서브트리란, 전체 트리에서 $$x$$를 루트로 하는 서브트리를 제외한 트리를 말합니다.

위 두 값을 모두 계산해두면, $$x$$를 반드시 색칠하는 경우의 답은 $$ down_x up_x $$ 가 됩니다.

두 식은 각각 다음과 같이 계산할 수 있습니다. 여기서 $$E[x]$$는 $$x$$의 자식 정점들의 집합이고, $$p$$는 $$x$$의 부모 정점입니다.

$$ down_x = \displaystyle \prod_{e \in E[x]} (down_e + 1) $$

$$ up_x = 1 + up_p \times \displaystyle \prod_{e \in E[p] \backslash x} (down_e + 1) $$

위 식을 일반적인 tree dp 를 하듯이 계산해 주면 됩니다. $$down_x$$는 간단한 DFS로 쉽게 계산할 수 있고, $$up_x$$는 구간 곱 배열을 응용하면 총 $$O(N)$$시간에 모든 값을 계산할 수 있습니다. 자세한 구현방법은 코드를 보는 것이 도움이 될 것입니다. 저는 DFS를 돌면서 현재 정점의 자식들의 $$up$$값을 계산해주는 방식으로 구현했습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/5404968)


## W. Intervals

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_w)

$$ d_i : $$ string의 $$i$$번째 글자까지 결정했고, 마지막 $$i$$번째 글자가 1인 경우의 최대 score.
{: style="text-align: center"}

위와 같은 DP 정의를 생각합시다. $$i$$번째 글자 직전에 나오는 1의 위치를 $$j$$라고 생각하면 다음과 같은 점화식을 생각할 수 있습니다. $$i$$번째 위치의 1에 의해 score에 반영해야 하는 구간들의 점수를 더해주는 것입니다.

$$ d_i = \displaystyle \max_{j < i} \Biggl( d_j + \sum_{m=1}^M
\begin{cases} 
a_m \ \ (l_m \leq i \leq r_m \& \& !(l_m \leq j \leq r_m) )
\\
0 \ \ (otherwise)
\end{cases} \Biggr)
$$

위 식을 그대로 계산하면 $$O(NM)$$ 시간복잡도가 되어 충분히 빠르지 못합니다.

$$d_j$$값이 뒤의 $$d_i$$ 값에 어떻게 영향을 주는 지 생각해 봅시다. $$j$$보다 이후에 열리고, $$i$$보다 이후에 닫히는 구간들의 값이 $$d_j$$에 더해져서 $$d_i$$에 반영이 되는 형태입니다. 따라서 모든 구간들을 open, close 위치 기준으로 정리해 놓읍시다. index $$i$$를 하나하나 증가시켜 가면서, $$i$$위치에서 열리는 구간들의 값을$$[1,i-1]$$ 구간의 $$d_j$$값들에 더해 줍시다. 또한 $$i$$위치에서 닫히는 구간들의 값도 $$[1,i-1]$$ 구간의 $$d_j$$값들에 빼 줍니다. 이후 $$d_i$$를 결정하려면 위 값들이 반영된 $$d_1, d_2, ... d_{i-1}$$ 중 최댓값이 $$d_i$$가 될 것입니다. Range add / Range max query 를 처리하기 위해 segment tree + lazy propagation을 사용하면 $$O((N+M)lgN)$$ 시간복잡도에 문제를 해결할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4255050)

## X. Tower

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_x)

먼저, 블록들을 어떤 순서로 쌓아야 최적일지 생각해 봅시다. $$(w_1, s_1)$$속성을 가지는 1번 블록과, $$(w_2, s_2)$$ 속성을 가지는 2번 블록의 순서를 결정한다고 생각합시다. 1번 블록이 2번 아래에 오는 경우에는 위로 더 쌓을 수 있는 무게가 $$s1-w2$$입니다. 반대로 2번 블록이 1번 아래에 오는 경우에는 위로 더 쌓을 수 있는 무게가 $$s2-w1$$입니다. 1번 블록이 아래에 오는 것이 유리하려면 $$s1-w2 < s2-w1$$부등식을 만족해야 된다는 것을 알 수 있습니다. 이를 다시 적어보면, $$s1+w1 < s2+w2$$가 되는데, 즉 $$s_i+w_i$$가 작은 블록일수록 아래에 오는 것이 유리하다는 뜻이 됩니다. (이러한 테크닉을 Exchange Arguments Technique 이라고도 합니다.)

위 결과에 따라 모든 블럭을 $$s_i+w_i$$의 오름차순으로 정렬하고 시작합시다. 정렬된 순서대로 선택한다는 것을 알고 있으니, 어떤 블록들을 선택할지만 문제가 됩니다. 이제 다음과 같은 DP 정의를 생각합시다.


$$ d_{n,k} :$$ $$n$$번째 블록까지 고려했고, 위로 $$k$$만큼의 무게를 더 견딜 수 있을 때 최대 value.
{: style="text-align: center"}

이제 $$n+1$$번째 블록을 선택하는지 여부에 따라 다음과 같은 transition 이 정의됩니다. 이 때 $$d_{n,k}$$ 기준으로 다음 state에 뿌려주는 방식으로 구현하는 것이 편합니다. 다음과 같은 형태로 구현이 될 것입니다.

$$ d_{n+1,k} \leftarrow d_{n,k} $$

$$ d_{n+1,k} \leftarrow d_{n,min(s_{n+1},k-w_{n+1})} + v_{n+1} $$

$$k$$의 최대 범위는 $$10^4$$이므로 무난히 $$O(N \times 10^4)$$ 시간복잡도로 통과 가능합니다. 최종적인 답은 가능한 $$k$$들에 대해 $$d_{N,k}$$들 중 최댓값이 될 것입니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4254560)

## Y. Grid 2

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_y)

H. Grid 1 문제의 고난이도 버전입니다. $$H, W$$가 커져서 $$O(HW)$$시간복잡도로는 문제를 해결할 수 없습니다. 

모든 벽을 $$(r,c)$$의 오름차순으로 정렬하고 시작합시다. 그리고 다음과 같이 DP 정의를 생각합시다. 

$$ d_n :$$ 빈칸만 거쳐서 $$n$$번째 벽에 도달하는 방법의 수. 다시 말해, $$n$$번째 벽에 도달하는데 그것이 처음으로 만나는 벽인 경로의 수.
{: style="text-align: center"}

벽들을 좌표 순서대로 정렬한 이유는 $$n$$번째 벽에 도달하기 전에 거쳐올 수 있는 벽들은 반드시 $$n$$보다 index가 작아지도록 하기 위함이었습니다.

처음으로 만나는 벽이라는 조건 없이 $$(r_n, c_n)$$에 도달하는 경우의 수는 $$ \frac {(r_n+c_n)!} {r_n! c_n!}$$ 입니다. 여기서 이전에 이미 벽을 만난 경우를 빼 주면 $$d_n$$을 계산할 수 있습니다. 

$$n$$번째 벽 직전에 만난 벽은 어떤 것들일지 생각해 봅시다. 위에서 벽들을 좌표 순서대로 정렬했기 때문에 index $$i$$는 반드시 $$n$$보다 작아야 할 것입니다. 또한, $$r_i \leq r_n$$ 과 $$c_i \leq c_n$$을 모두 만족해야 할 것입니다. 이 조건들을 만족하는 모든 $$i$$들에 대해,  
$$ d_i \frac {(r_n+c_n-r_i-c_i)!} {(r_n-r_i)! (c_n-c_i)!}$$ 를 $$d_n$$에서 빼 주면 됩니다.

DP transition은 이미 충분히 설명했으므로 식을 따로 적진 않겠습니다. 위와 같은 방법으로 $$d_n$$을 계산하면, 정의대로 모든 경로를 잘 고려한다는 것을 이해하는 것이 중요합니다.

$$(H,W)$$에 $$N+1$$번째 점이 있다고 가정하고 문제를 풀면 $$d_{N+1}$$이 답이 됩니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4253016)

## Z. Frog 3

[문제 링크](https://atcoder.jp/contests/dp/tasks/dp_z)

아주 유명한 Convex hull trick 문제입니다. Convex hull trick문제와 해결방법에 대해서는 [다른 글(링크)](https://jinpyo.kim/lichao-tree) 을 참고하시기 바랍니다.

먼저 다음과 같은 간단한 DP 정의를 생각합니다.

$$ d_i :$$ $$i$$번 stone에 도달하는 최소 비용.
{: style="text-align: center"}

DP transition은 다음과 같이 정의됩니다.

$$ d_i = \min_{j < i} \Bigl( d_j + (h_i-h_j)^2 + C \Bigr)$$

이 식을 조금 변형해 봅시다.

$$ \begin{align}

 d_i    &= \min_{j < i} \Bigl(d_j + (h_i-h_j)^2 + C \Bigr) \\
        &= \min_{j < i} \Bigl(d_j + h_i^2 + h_j^2 - 2 h_i h_j + C \Bigr) \\
        &= \min_{j < i} \Bigl((-2h_j)h_i + (d_j + h_j^2) \Bigr) + h_i^2 + C

\end{align}$$

기울기가 $$-2h_j$$이고, $$y$$절편이 $$d_j + h_j^2$$인 직선이 추가되는 Convex hull trick 문제의 형태로 변형되었습니다.

이제 Convex hull trick을 푸는 다양한 방법들 중 아무 것이나 적용해서 문제를 해결하면 됩니다. 이 문제의 경우는 $$i$$가 커질수록 추가되는 직선의 기울기가 작아진다는 성질을 이용하면 [Lichao Tree](https://jinpyo.kim/lichao-tree) 등의 자료구조를 굳이 사용하지 않아도 해결할 수 있습니다.

[코드(링크)](https://atcoder.jp/contests/dp/submissions/4251525)
