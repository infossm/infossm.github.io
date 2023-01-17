---
layout: post
title:  "Codeforces Virtual을 통한 스터디와 문제풀이"
date:   2019-09-18 20:30:00
author: shjgkwo
tags: [algorithm, Codeforces, study]
---

# 목차

- [1. 개요](#개요)
- [2. Codeforces](#Codeforces)
- [3. 문제풀이](#문제풀이)
- [4. 마무리](#마무리)

# 개요
 문제풀이를 하는 사람들 중에 코드포스는 본인의 실력을 가늠하기에 매우 훌륭한 지표입니다. 특히나 코드포스 블루(expert) 등급에 해당하는 사람들은 웬만한 회사의 코딩테스트는 쉽게 통과한다라는 말이 있을 정도로 본인의 실력의 지표로서는 매우 유용한 가치가 있습니다. 하지만 그런 지표뿐만이 아니라 본인의 실력을 늘리는데에도 코드포스 만큼 좋은 사이트는 몇 없습니다. 백준을 제외하고서라도 콘테스트 형식, 즉 제한된 시간에 빠른 알고리즘을 구상해내야하는 코딩테스트와 비슷한(물론, ACM-ICPC의 형태와 조금 더 가깝습니다.) 환경으로서 본인의 제한된 시간안에 효율적으로 문제를 해결하는, 본인의 문제해결능력을 길러줄 수 있는 좋은 사이트가 코드포스입니다. 저는 8월초 부터 같은 동아리 사람들과 코드포스 버츄얼을 돌리고 돌린 후에 풀이를 공유하고 같이 푸는 시간을 가지는 스터디를 하였습니다. 이번 포스트는 코드포스 스터디의 과정과, 가장 최근에 풀었던 문제들 중 괜찮았던 것 6개를 골라서 공유해보고자 합니다.


 # Codefroces
  [Codeforces](https://codeforces.com)는 ACM-ICPC 형식을 조금 차용하고, 핵과 점수 감소제, pretest와 maintest를 나누는 등의 자체룰을 적용한 프로그래밍 콘테스트 사이트입니다. Codeforces 에서 원하는 라운드에 참여하거나, 과거에 진행되었던 라운드에서 가상으로 콘테스트를 진행하는 등의 다양한 활동이 가능합니다. 특히 이러한 기능을 활용하여 ACM-ICPC에 준비하는 스터디를 하였는데요. 그 과정을 지금부터 설명하고자 합니다.

## 간단 소개

 위 링크에 접속하면 아래와 같은 화면이 나올것입니다.

![사진1](/assets/images/codeforces-study-shjgkwo/main-page.png)

 여기서 콘테스트를 누르면 과거 콘테스트 항목과 현재 진행중인 콘테스트, 진행 예정인 콘테스트 목록이 전부 나타납니다.

![사진2](/assets/images/codeforces-study-shjgkwo/main-contest.png)

 여기서 이미 지나간 콘테스트에 접속하면(Enter) 다음과 같은 항목이 나올것입니다.

![사진3](/assets/images/codeforces-study-shjgkwo/contest.png)

 이 사진은 제가 제일 잘했던 콘테스트의 사진입니다. 이 당시 6등이었습니다.
 내 등수를 확인하려면 standing 을 누르면 확인할 수 있습니다.
 이 중에 friend standing 을 고르면 나와 내가 친구등록(일방적으로)한 친구들을 볼 수 있습니다.

![사진4](/assets/images/codeforces-study-shjgkwo/friend-standing.png)

 그러면 친구 등록은 어떻게 할까요?
 기본적으로 친구의 개인정보 페이지에 들어가야합니다.

![사진5](/assets/images/codeforces-study-shjgkwo/shjgkwo.png)

 이것은 저의 페이지입니다.
 그러면 유명한 tourist 의 페이지를 살펴볼까요?

![사진6](/assets/images/codeforces-study-shjgkwo/tourist.png)

 이름 옆에 별이 보이시나요? 저것을 눌러서 노랗게 점등이 되면 친구 등록이 완료된 것 입니다.
 이렇게되면 친구가 나와 같은 대회를 치뤘다면 friend standing 에서 등수와 푼 문제를 볼 수 있습니다.
 나중에 친구의 코드를 구경할 때 유용합니다.

## Virtual Contest

 이제 이 포스트의 주된 내용인 Virtual Contest 하는 법 입니다.
 먼저 이미 지나간 콘테스트 중에 좋아보이는 콘테스트를 하나 고릅니다.

![사진7](/assets/images/codeforces-study-shjgkwo/select-contest.png)

 이 사진에서 Virtual Participation 이라는 링크가 보이시나요?
 이 링크를 누르면 다음과 같은 화면이 나옵니다.

![사진8](/assets/images/codeforces-study-shjgkwo/virtual-contest-registration.png)

 여기서 시간만 조정하고 내용의 동의하고 시작하시면 됩니다!
 그러면 실제로 콘테스트를 치루는 기분으로 현재 내 등수등도 확인 할 수 있습니다.
 마지막으로, 이전에 내가 치룬 콘테스트는 맨 처음 콘테스트들의 목록 화면에서 볼 수 있습니다.

![사진9](/assets/images/codeforces-study-shjgkwo/recent-virt-contest.png)

 이제 몇가지 제가 재미있게 풀었던 문제를 공유해볼까요?

# 문제풀이

## Hello 2018 C Problem

 이 [링크](https://codeforces.com/contest/913/problem/C)를 통하여 문제를 볼 수 있습니다.
 이 문제는 파티를 열었는데 레몬에이드가 부족하여 필요한 리터수 만큼 레몬에이드를 사러 가는 문제입니다.
 우선 가게에서는 1, 2, 4, 8, ... 이렇게 $2^{k}$ 꼴의 2의 멱수승 꼴의 리터 단위로 레몬에이드를 팝니다.
 그리고 각 리터단위에는 가격이 붙어있습니다. 이럴 때 적어도 $m$이상의 레몬에이드가 필요할 때, 그 레몬에이드를 마련하는 데
 필요한 비용은 얼마인가 물어보는 문제인데요. 이 문제의 솔루션은 간단합니다.

 생각해보면 각 비용에서 2배를 한다면 바로 내 옆의 리터, 즉 1리터를 2개를 사면 2리터, 2리터를 2개를 사면 4리터 가 됨을 알 수 있습니다.
 이때 그 비용의 2배가 훨씬 작다면 당연히 덮어 쓰는게 좋겠죠? 그럼 1리터 부터 시작합니다. 예제의 경우 20 30 70 90 꼴로 되어있는데요.
 우리는 이것을 20 30 60 90 으로 고쳐줄 수 있습니다! 여기서 하나더, 그럼 이제 16리터를 사는 데 필요한 비용은 8리터에서 곱하기 2하고
 32리터는 16리터에서 곱하기 2하면 됩니다. 그렇다면 이런식으로 $2^{30}$ 리터까지 모두 구합니다.

 왜 굳이 불필요한 30번째 까지 구했냐구요? 이 문제는 당연히 다이나믹 프로그래밍으로 풀 수 없으니 그리디한 접근법을 보여주기 위함입니다.

 ```cpp
    long long ans = 1ll << 60, sum = 0;
    for(int i = 30; i >= 0; i--) {
        if(m - (1ll << i) > 0) {
            m -= 1ll << i;
            sum += p[i];
        }
        if(sum + p[i] < ans) ans = sum + p[i];
    }
```

이 코드에서 m은 필요한 리터수고 여기서 30번째 레몬에이드, 즉, $2^{30}$ 리터의 레몬에이드 부터 천천히 빼봅니다.
이때, 0보다 클때만 빼주는 게 포인트이며, 빼주고 나서, 다음 레몬에이드를 한번 더 뺐을 때, 그것이 이득인지 검사합니다.
이득이면 후보값을 갱신하고 아니면 무시합니다.

이렇게 하면 항상 정답을 구할 수 있는데요. 이것에 대한 증명은 여러분에게 숙제로 남기겠습니다.

[전체 코드](https://github.com/Byeong-Chan/codeforces/blob/master/Hello-2018/C.cpp)는 여기서 보실 수 있습니다.

## Hello 2018 D Problem

 이 [링크](https://codeforces.com/contest/913/problem/D)를 통하여 문제를 확인할 수 있습니다.
 이 문제는 문제가 너무 쉬워서 특정 개수보다 적게 풀어야만 1점을 획득하는 기묘한 상황에서 어떤 문제의 푸는 시간과 특정 개수가 주어졌을 때
 어떤 문제를 골라야 최대한 점수를 얻는지 찾는 문제입니다.

 우리는 이런 문제를 보면 본능적으로 결정문제로 바꾸면 쉽게 풀 수 있다는 걸 알 수 있어야 합니다.

 생각해보면 내가 2점을 획득할 수 있다고 가정한다면, 2문제만 풀면 되기 때문에 특정개수가 2 보다 크거나 같은 애들중에 푸는 시간이 제일
 짧은 애들만 찾아서 2개를 찾아내어 더했을때, 제한시간을 초과하지 않으면 더 늘려보고 아니면 더 줄이는 방식으로 바이너리 서치를 하면 됩니다!
 
 참으로 간단한 문제입니다. 주의할 점은 정렬하는 과정에서 순서가 깨지기 때문에 순서를 미리 기억해 두세요!

 [전체 코드](https://github.com/Byeong-Chan/codeforces/blob/master/Hello-2018/D.cpp)는 여기서 보실 수 있습니다.

## Hello 2018 E Problem

 이 [링크](https://codeforces.com/contest/913/problem/E)를 통하여 문제를 확인할 수 있습니다.
 이 문제는 진리표가 주어지면 해당 진리표에 걸맞는 진리식을 길이가 제일 짧으면서, 동시에 길이가 제일 짧은게 여러개 있으면 사전순으로
 제일 앞서는 것을 고르는 문제입니다.

 이 문제를 보자마자 벙쪘는데요. 끝날때까지, 카노맵만 생각하다가 결국 끝날때 까지 풀지 못했습니다.
 하지만 다시 도전하여 결국은 풀어냈는데요. 우리는 사람기준에서 생각하면 안되고 컴퓨터 기준에서 생각해보아야 합니다.

 기본적으로 컴퓨터는 x, !x, y, !y, z, !z 를 &로 붙이거나 |로 붙이거나 괄호로 묶거나 묶은뒤 !을 붙이거나 할 것 입니다.
 또한 &로 묶일때, 만약 현재 진리식에서 |가 있었다면 괄호로 묶어야 할 것입니다.
 그럼 여기서 뭔가 감이 잡히십니까?

 네, 255개의 상태에서 |로 묶인것이 있는지 없는지 판단하는 상태를 2개 추가해서 2x255 배열을 만들어서 다이나믹 프로그래밍을 하거나
 문자열의 길이와 사전순정렬을 값으로 하는 최단경로 문제를 푸는것과 완전히 같은 문제가 됩니다!

 ```cpp
string dist[2][256]; // 0 : no |,   1 : yes |
 
struct node {
    string s;
    int types;
    int state;
    node() { s = ""; types = 0; state = 0; }
    node(string s, int types, int state) { (this->s).assign(s); this->types = types; this->state = state; }
    const bool operator< (const node &a) const {
        if(s.length() == a.s.length()) return s > a.s;
        return s.length() > a.s.length();
    }
};
 ```

 우선 거리배열과 구조체를 적당히 정의 해줍니다. 저 구조체를 활용하여 우선순위 큐를 만들어서 다익스트라를 실행할 것 입니다.
 그렇다면 앞에다 & 붙이기 뒤에다 & 붙이기 앞에다 | 붙이기 뒤에다 | 붙이기 네가지로 나눌 수 있으며
 
 이동할때마다 변하는 상태는 비트마스크로 관리해주면 됩니다.
 주의할점은 |로 묶인것을 &로 다시 묶으려고 할때 반드시 괄호로 다시 묶어줘야 한다는 사실을 잊지 마세요!

 여기서 끝나면 문제가 참 쉽겠죠?

 절대 여기서 끝날리 없습니다. 이전에 완성된 상태를 다시 사용하는 경우도 있으니 그것을 고려해야합니다!

 이것은 그것에 대한 코드입니다.

```cpp
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 255; j++) {
                    if(dist[i][j].length() == 0) continue;
                    
                    string ss;
                    if(i == 1) ss = "(" + dist[i][j] + ")";
                    else ss = dist[i][j];
                    
                    sub.s = "(" + tmp.s + ")&" + ss;
                    sub.types = 0;
                    sub.state = tmp.state & j;
                    
                    if(dist[sub.types][sub.state].length() == 0) pq.push(sub);
                    
                    sub.s = ss + "&(" + tmp.s + ")";
                    sub.types = 0;
                    sub.state = tmp.state & j;
                    
                    if(dist[sub.types][sub.state].length() == 0) pq.push(sub);
                    
                    sub.s = tmp.s + "|" + ss;
                    sub.types = 1;
                    sub.state = tmp.state | j;
                    
                    if(dist[sub.types][sub.state].length() == 0) pq.push(sub);
                    
                    sub.s = ss + "|" + tmp.s;
                    sub.types = 1;
                    sub.state = tmp.state | j;
                    
                    if(dist[sub.types][sub.state].length() == 0) pq.push(sub);
                }
            }
```

시간복잡도는 완전 그래프라고 가정하면 $V^{2}$ 개의 간선이 있을테니 $O(V^{2} log V)$ 의 시간복잡도를 같습니다!
다이나믹 프로그래밍으로 푸는 방법은 $O(V^{3})$ 이라고 하네요!

참고로 V는 256 이라는 점을 잊지 마세요!

[전체 코드](https://github.com/Byeong-Chan/codeforces/blob/master/Hello-2018/E.cpp)는 여기서 확인할 수 있습니다!


## Div.2 #423 C Problem

 이 [링크](https://codeforces.com/contest/828/problem/C)를 통하여 문제를 확인할 수 있습니다.
 이 문제는 어떤 부분 문자열이 어느 위치에서 출현했는지 주어졌을 때, 이 문자열중 사전순으로 가장 앞서는 것을 찾는 문제입니다.
 당연하게도 길이가 가장 짧은게 좋을것이기 때문에 제일 마지막까지 사용하는 문자열을 찾아내어 최대길이를 지정하고 나서
 군데군데 빠져있는 문자열은 'a'로 채워넣으면 되는 문제입니다.
 그렇다면 문제는 채워넣는 작업이 시간복잡도상으로 너무 오래 걸리는데 이것을 어떻게 해결하느냐가 문제인데요
 이 문제는 해결하는 것은 매우 간단합니다. Union-Find를 활용하여 이미 채워진 위치는 건너뛰는 방식을 활용하면 됩니다!
 기존에 Path Compression 을 이용하며, 거기에 추가적으로 다음으로 이동하는 배열을 추가하여 건너 뛰는 것을 만들면 매우 쉽게
 해결 할 수 있습니다.

```cpp
#include <algorithm>
 
using namespace std;
 
int parents[10000010];
int nxt[10000010];
 
int fd(int u) {
    if(u == parents[u]) return u;
    else return parents[u] = fd(parents[u]);
}
void uni(int x, int y) {
    x = fd(x);
    y = fd(y);
    
    int tmp = max(nxt[x], nxt[y]);
    nxt[x] = tmp;
    nxt[y] = tmp;
    
    parents[y] = x;
}
 
char o[1000010];
char ans[10000010];
 
int main() {
    for(int i = 0; i < 10000010; i++) parents[i] = i;
    for(int i = 0; i < 10000010; i++) nxt[i] = i + 1;
    return 0;
}
```

 이와 같은 코드로 건너뛰는 작업을 쉽게 해줄 수 있습니다.

```cpp
            int pre = -1;
            for(int j = k; ; j = nxt[fd(j)]) {
                if(j >= k + len) break;
                if(pre != -1) uni(pre, j);
                ans[j] = o[j - k];
                pre = j;
            }
```

이동은 이런 방식으로 해주면 되겠네요!

[전체 코드](https://github.com/Byeong-Chan/codeforces/blob/master/DIV2-423/C.cpp)는 여기서 확인할 수 있습니다.

## Div.2 #423 D Problem

 이 [링크](https://codeforces.com/contest/828/problem/D)를 통하여 문제를 확인할 수 있습니다.
 이 문제는 리프노드의 개수가 k개인 노드가 n개인 트리에서 트리의 지름이 제일 짧은 것을 구하는 문제입니다.

 아주 심플하고 좋은 문제인데요. 아이디어 접근법은 다음과 같습니다. 우선 1을 루트로 하고, 내가 필요한 만큼 k개의 가지를 만들어냅니다.
 약간 기름때에 비누거품이 붙는 모습을 그린 모습이 될것입니다. 가운데에 동그라미 하나에 털이 k개 만큼 송송송 나고 그 털 끝에 동그라미가 하나
 더 달린 그림이 될 것입니다. 1을 기준으로 2 ~ 2+k-1 의 노드가 그런식으로 달라 붙었을 것입니다. 이러면 지름이 2인 트리가 완성이 되었죠?
 하지만 애석하게도 우리는 n개의 노드를 전부 써야합니다. 즉 2+k ~ 2+k+k-1 의 노드를 아까 그 털끝에 노드에 한번씩 더 붙여 보면서 도중에
 n개가 되면 멈추면 됩니다. 정말 당연하게도 이것이 최선이며, 앞서 말했지만 n개 다 채워지면 멈추면 됩니다.

 한번 직접 구현해보면 좋을것 같습니다. 지름을 구하는 것도 숙제로 남겨 두겠습니다.

 [전체 코드](https://github.com/Byeong-Chan/codeforces/blob/master/DIV2-423/D.cpp)는 여기서 보실 수 있습니다.


## Div.2 #423 E Problem

 이 [링크](https://codeforces.com/contest/828/problem/E)를 통하여 문제를 확인할 수 있습니다.
 이 문제는 염기서열이 주어졌을때, 주기적으로 반복되는 염기서열 문자열이 주어졌을 떄, 그 반복되는 과정에서 서로 같은 부분이 몇개 있는지
 체크하고, 도중에 변하는 문자가 있을때 그것을 쿼리로 처리하는 문제입니다.

 우선 이 문제는 BIT로 푸는 법과 BIT를 쓰지않고 푸는 법 두가지가 있는데요.

 우선 BIT를 쓰지 않고 푸는 법을 생각해봅시다.
 생각해보면 길이 제한이 10 이기 때문에 1~10을 약수로 갖는 숫자중 제일 작은수를 생각해보면 2520 이라는것을 알 수 있습니다.
 그렇다면 2520 을 기준으로 배열을 쪼갠뒤, 2520씩 건너뛰게 만들면, 즉, 0, 2520, 5040, 7560... 이런식으로 이 라인에 대한
 A, C, G, T 의 개수를 누적합으로 구할 수 있습니다. 이걸 업데이트하는데 걸리는 시간은 대략 100000/2520 이며, 이는 40정도에 근접합니다.

 그리고 이것에 대한 구간 쿼리를 수행하는 것은 숙제로 남겨두겠습니다. 이렇게 해결하면 대략 2억번 정도의 for문을 돌며 아슬아슬하게 1초 정도로
 맞을 수 있습니다.

 [코드](https://github.com/Byeong-Chan/codeforces/blob/master/DIV2-423/E-1.cpp)

 이제 좀 더 고급적인 접근을 해볼까요?

 어차피 범위가 1~10 이므로, 공간을 [N][1, 10][0, 9] 로 나누어서 BIT로 관리한다면 각 구간에 대해서 내가 몇번째에 어떤 규칙으로 출현하는지 알아낼 수 있습니다!
 이것을 이동하는 범위만 잘 체크하면 BIT로 간단하게 풀 수 있습니다!

 [코드](https://github.com/Byeong-Chan/codeforces/blob/master/DIV2-423/E-2.cpp)

 풀이가 참 좋은 문제였습니다.

# 마무리

 이번 포스트를 통하여 많은 분들이 코드포스에 관심을 가지고 Virtual Contest 를 많이 돌려보고 좋은 문제를 많이 접해보는 계기가 되었으면 좋겠습니다. 이상으로 포스트를 마칩니다. 감사합니다.
