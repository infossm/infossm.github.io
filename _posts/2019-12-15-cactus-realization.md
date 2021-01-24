---
layout: post
title:  "Cactus graph realization of degree sequence"
date:   2019-12-15 05:30:00
author: ainta
tags: [algorithm, graph-theory, cactus, degree-sequence, ICPC]
---



# Degree sequence

그래프에서, **Degree sequence**란 undirected graph의 각 정점의 차수(degree)를 늘어놓은 수열을 말한다. 

**Graph realization problem**이란, 수열이 주어졌을 때 그 수열을 degree sequence로 갖는 그래프를 실제로 construct하는 문제를 말한다. 여기서 다루는 그래프는 self-loop나 multiedge가 존재하지 않는 simple graph이다.

어떤 Degree sequence가 주어졌을 때, 이를 만족하는 simple graph가 존재할 조건은 **Erdos - Gallai theorem** 으로 널리 알려져 있다.



 **정리 1 (Erdos - Gallai theorem). $$d_1 \ge d_2 \ge ... \ge d_n \ge 0$$  가 finite simple graph의 degree sequence일 필요충분조건은 모든 $$1 \le k \le n$$에 대해   $$\sum_{i=1}^{k} d_i \le k(k-1) + \sum_{i=k+1}^n min(d_i, k)$$ 가 성립하는 것이다.**

이것이 필요조건인 것은 차수가 가장 큰 $$k$$개 정점과 작은 $$n-k$$개의 정점으로 나누어 생각해보면 쉽게 알 수 있다.

충분조건임을 증명하는 것은 간단하지 않고,  앞으로 다룰 내용과 큰 관련이 있지 않기 때문에 생략한다.



# Degree sequence of Tree

어떤 수열이 주어졌을 때, 이 수열을 degree sequence로 갖는 트리가 존재하는지는 어떻게 판정할 수 있을까?

$$d_1, d_2 , ... , d_n$$이 어떤 트리의 degree sequence일 때, 간선의 개수가 $$n-1$$개이므로 $$d_1 + d_2 +... + d_n = 2(n-1)$$임을 쉽게 알 수 있고, connected graph이므로 $$n \ge 2$$일 때 $$d_1, d_2, ... , d_n \ge 1$$임을 알 수 있다.



**정리 2. $$n \ge 2$$일 때, $$d_1, d_2, ..., d_n \ge 1$$이 tree의 degree sequence일 필요충분조건은 $$d_1 +... + d_n = 2(n-1)$$ 이다.**

필요조건임은 앞서 보였고, 충분조건인 것만 보이면 충분하다. 이는 수학적 귀납법으로 증명할 수 있다.

먼저, $$n = 2$$인 경우는 $$d_1 = d_2 = 1$$인 한 가지 경우밖에 없으므로 자명하다. 

$$n = k (k \ge 2)$$일 때 성립한다고 하자. $$n = k+1$$인 경우, $$d_1, d_2, .., d_n \ge 2$$이면 $$d_1+d_2+..+d_n \ge 2n$$이 성립해야 하므로 $$d_u = 1$$인 $$u$$가 존재함을 알 수 있다. 또한 $$n \ge 3$$이면 $$2(n-1) > n$$이므로 $$d_v \ge 2$$인 $$v$$가 존재한다. 

$$d_u$$와 $$d_v$$를 1 감소시키면 $$d_u$$는 0이 되고 $$d_v$$는 1 이상이다. 그러면 이를 degree sequence로 가지는 정점이 $$n-1$$개인 트리를 구성할 수 있다.

그 트리에서 $$u$$번 정점과 $$v$$번 정점을 잇는 간선을 추가하면 $$d_1, d_2, .., d_n$$을 degree sequence로 가지는 트리가 만들어진다. 따라서 $$n = k+1$$일 때 역시 claim이 성립한다.

위에서 증명에 사용한 방식을 이용하면, $$d_1, d_2, ..., d_n$$이 주어졌을 때 실제로 이를 degree sequence로 가지는 Tree를 $$O(N log N)$$시간에 construct할 수 있음을 알 수 있다. ( (남은 차수, 정점번호)를 set으로 관리 후 최소원소와 최대원소를 찾는것을 반복 )



# Degree sequence of Cactus



**Spoiler warning: NERC(Northern Eurasia Finals 2019)의 문제에 대한 Spoiler가 포함되어 있습니다**



**Cactus**란, 어떤 edge도 두 개 이상의 simple cycle에 포함되지 않는 undirected connected simple graph를 말한다. Cactus는 트리의 일반화된 형태로, 모든 트리는 cactus이다. 일반적으로 그래프에서 빠른 시간 내에 해결하기 어려운 문제들을 트리에서 빠르게 해결할 수 있는 것과 마찬가지로, cactus 역시 매우 특이한 형태의 그래프이기 때문에 좋은 성질들을 가진다.

수열 $$d_1, d_2 , ... , d_n \ge 1$$이 주어졌을 때, 이것이 어떤 cactus의 degree sequence가 될 조건은 무엇일까?



**Observation 1. 정점의 개수와 간선의 개수의 상관관계**

먼저,  cactus는 planar graph이기 때문에 edge 개수를 $$m$$, simple cycle의 개수를 $$c$$라 하면 ​$$n - m + c = 1$$이 성립한다(Euler's characteristic). 여기서 ​$$m = \frac{1}{2}\sum_{i=1}^n d_i$$이다.

모든 simple cycle은 적어도 3개 이상의 edge로 이루어지고, 어떠한 edge도 둘 이상의 simple cycle에 포함되지 않으므로 $$3c \le m$$이 성립한다. 따라서, $$n = m-c+1 \ge \frac{2}{3}m+1$$이 성립한다. 한편, connected graph이므로 $$n-1 \le m$$인 것은 자명하다.

따라서, $$c = m - (n-1)$$에서 $$0 \le c \le \frac{n-1}{2}$$가  성립한다. 이는 정수이므로 $$ 0 \le c \le \lfloor \frac{n-1}{2} \rfloor$$라고 할 수 있다.



**Observation 2. 모든 차수가 짝수인 경우**

**보조정리 1. 모든 차수가 짝수인 경우, $$ 0 \le c \le \lfloor \frac{n-1}{2} \rfloor$$ 는 $$d_1, d_2, ...,d_n$$이 cactus의 degree sequence일 필요충분조건이다.**

**Proof**

$$d_1 = d_2 = ... = d_n = 2$$인 경우, 모든 정점을 지나는 cycle이 된다. 

그 외의 경우, $$d_u \ge 4$$인 $$u$$가 존재하고, $$d_v = d_w = 2$$인 서로 다른 정점 $$v$$, $$w$$가 존재한다. (그렇지 않으면 $$d_1 + ... + d_n \ge 4(n-1) + 2$$이므로 $$m \ge 2(n-1)+1$$이고, 따라서 $$ c \ge n-1 > \lfloor \frac{n-1}{2} \rfloor$$)

이제 $$d_u$$를 2 감소시키고 $$d_v$$와 $$d_w$$를 없앤 수열을 생각하면 이 $$n-2$$개의 수가 위 조건을 만족함을 알 수 있다.

($$n-2$$개의 수에 대해 $$c$$ 값을 계산하면, 모든 차수가 2 이상이므로 $$0 \le c$$ 이다. 또한, n은 2, m은 3만큼 감소하였으므로 c값은 1 감소하였고, 따라서 $$c \le \lfloor \frac{n-3}{2} \rfloor $$)

이를 만족하는 cactus를 구성한 뒤 $$u, v, w$$ 를 지나는 크기 3인 cycle을 추가하면 $$d_1, d_2,..., d_n$$을 degree sequence로 가지는 cactus가 나온다. (수학적 귀납법을 이용한 증명)



**Observation 3. 홀수차수 정점이 있는 경우**

그래프에서, 어떤 edge를 제거하면 component의 개수가 늘어나는 경우 그 edge를 **bridge**라고 한다. 트리의 경우 모든 edge가 bridge인 특수한 형태이다. 

degree가 홀수인 정점을 **odd vertex**, 짝수인 정점을 **even vertex**라 하자.

또한, 트리의 leaf를 일반화해서 connected graph에서도 차수가 1인 정점을 **leaf**라 부르기로 하자.

bridge의 개수를 $$b$$, leaf의 개수를 $$l$$, odd vertex의 개수를 $$o$$라 놓자.

$$n \ge 3$$인 Cactus에서, leaf끼리는 edge로 연결되서는 안되고, odd vertex는 적어도 하나의 cycle에 포함되지 않은 edge와 인접하므로  $$b \ge max(\frac{o}{2}, l)$$이 성립한다. 

한편, bridge는 어느 cycle에도 포함되지 않는 간선들이므로 $$ 3c \le m-b$$이고, 여기에서 $$3(m-n+1) \le m - b$$ , 즉 $$2(m-n+1) \le n-b-1$$에서  $$c \le \lfloor \frac{n-1-b}{2} \rfloor$$가 성립한다.

따라서, $$c \le \lfloor \frac{n-1-max(\frac{o}{2}, l)}{2} \rfloor$$이다.



**정리 3. $$n \ge 3$$일 때,  $$0 \le c \le \lfloor \frac{n-1-max(\frac{o}{2}, l)}{2} \rfloor$$은 주어진 sequence가 cactus의 degree sequence일 필요충분 조건이다.**

**Proof**

이는 보조정리 1의 증명과 비슷하게 수학적 귀납법을 이용하여 construction할 수 있다.

odd vertex가 존재하지 않는다면, 보조정리 2에 의해 성립한다. odd vertex가 존재한다고 가정하자.

먼저, leaf $$u$$가 존재하는 경우를 생각해보자. 

만약 leaf가 아닌 odd vertex $$v$$가 존재한다면, 둘을 연결하고 $$u$$ 를 지워보자. 그러면 $$max(\frac{o}{2}, l)$$이 1 감소하므로, 남은 $$n-1$$개 정점의 degree가 조건을 만족하게 된다.

모든 odd vertex가 leaf라면, leaf가 아닌 정점 $$w$$를 선택해 $$w$$와 $$u$$를 연결하고 $$w$$를 지워보자. $$o = l$$이므로 $$max(\frac{o}{2}, l)$$가 역시 1 감소하고, 따라서 남은 $$n-1$$개 정점의 degree가 조건을 만족하게 된다.

마지막으로 leaf가 존재하지 않지만 odd vertex는 존재하는 경우를 생각해보자.

$$d_u \ge 3$$인 $$u$$가 존재하고, $$d_v = d_w = 2$$인 서로 다른 정점 $$v$$, $$w$$가 존재한다.

$$u, v, w$$를 잇는 cycle을 만들고 $$v, w$$를 지워보자. $$c$$는 1 감소하지만 $$d_u$$를 제외하고는 모두 2 이상으로 유지되기 때문에 $$c$$는 0보다 작아지지 않는다.

한편, $$\lfloor \frac{n-1-max(\frac{o}{2}, l)}{2} \rfloor$$ 은 최대 1만큼 감소하므로 $$0 \le c \le \lfloor \frac{n-1-max(\frac{o}{2}, l)}{2} \rfloor$$는 여전히 성립하게 된다.

cactus의 한 정점에 새로운 leaf를 붙이거나, 새로운 정점 두 개를 포함하는 크기 3인 cycle을 붙이면 여전히 cactus이므로 수학적 귀납법에 의해 본 정리가 증명된다.



다음은 수열이 주어질 때 그 수열을 degree sequence로 가지는 cactus가 있으면 답을 찾아 출력하고, 존재하지 않으면 -1을 출력하는 프로그램이다. (NERC 2019 C.Cactus Revenge)

시간복잡도는 $$O(N^2)$$ 이지만 $$O(N log N)$$으로 어렵지 않게 개선할 수 있다.



```c++
    #include<bits/stdc++.h>
     
    using namespace std;
        
    typedef long long ll;
    typedef pair<int, int> pii;
    typedef pair<ll, ll> pll;
    #define pb push_back
    #define pli pair<long long,int>
        
    #define Fi first
    #define Se second
        
    struct point {
        int a, c; // vertex number a, degree c
        bool operator <(const point &p)const {
            return c < p.c;
        }
    };
     
    vector<pii>Ans;
    void Add(int a, int b) {
        Ans.push_back({ a,b });
    }
     
    bool OK(vector<point>w){ // Is it valid degree sequence of Cactus?
        if(w.empty())return true;
        sort(w.begin(),w.end());
        reverse(w.begin(),w.end());
        int i, n = w.size(), m = 0, o = 0, l = 0;
        for(i=0;i<w.size();i++){
            m+=w[i].c;
            if(w[i].c%2)o++;
            if(w[i].c==1)l++;
            if(w[i].c<=0)return false;
        }
        if(m%2==1)return false;
        m/=2;
        int c = m-n+1;
        if(c<0)return false;
        if(c>(n-1)/2)return false;
        if(n==2){
            Add(w[0].a,w[1].a);
            return true;
        }
        if(w[0].c==2 && w[n-1].c == 2){
            for(i=0;i<n;i++) Add(w[i].a,w[(i+1)%n].a);
            return true;
        }
        int b = max(o/2,l);
        if(c>(n-1-b)/2)return false;
        if(l){
            int u = w[n-1].a;
            for(i=0;i<n;i++){
                if(w[i].c%2==1 && w[i].c!=1){
                    w[i].c--;
                    Add(w[i].a,u);
                    w.pop_back();
                    OK(w);
                    return true;
                }
            }
            Add(w[0].a, u);
            w[0].c--;
            w.pop_back();
            OK(w);
            return true;
        }
        int x = w[0].a, y = w[n-2].a, z = w[n-1].a;
        Add(x,y);Add(y,z);Add(z,x);
        w.pop_back();
        w.pop_back();
        w[0].c-=2;
        OK(w);
        return true;
    }
     
    int main() {
        int i, n;
        scanf("%d", &n);
        vector<point>w(n);
        for (i = 0; i < n; i++) {
            scanf("%d", &w[i].c);
            w[i].a = i+1;
        }
        if (!OK(w)) {
            puts("-1");
            return 0;
        }
     
        printf("%d\n", Ans.size());
        for (auto &t : Ans) {
            printf("2 %d %d\n", t.first, t.second);
        }
    }
```



# More

트리의 degree sequence realization을 구현했다면 

KTU Programming Camp Day 2 D(https://codeforces.com/gym/100738/problem/D)에서 채점해볼 수 있다.

지금은 무방향 그래프에 대해서 다뤘지만, 이와 관련된 주제로

directed complete graph인 tournament graph에서의 score sequence가 있다.

이는 $$n$$개의 팀이 있을 때 모든 팀 쌍이 한번씩 경기를 한 후, 각 팀의 승리 횟수를 나열한 수열로 생각할 수 있다.

이와 관련한 문제로는 2016 ICPC 대전 리저널 C. 축구 게임(https://www.acmicpc.net/problem/13560)이 있다.
