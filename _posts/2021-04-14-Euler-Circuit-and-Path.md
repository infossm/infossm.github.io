---
layout: post
title:  "오일러 회로와 경로"
date:   2021-04-14 08:00:00
author: JooDdae
tags: [algorithm, graph-theory]
---



**이 글에서는 연결 무방향 단순 그래프를 다룹니다.**



# 오일러 회로

그래프의 모든 간선을 단 한 번씩 지나서 시작점으로 돌아오는 경로를 오일러 회로라고 합니다.

연결 그래프면서 차수가 홀수인 정점이 없다면 오일러 회로가 존재합니다. 그리고 오일러 회로가 존재한다면 차수가 홀수인 정점이 없습니다. 오일러 회로와 관련된 문제를 풀기 위해서는 이 필요충분조건만 알고 있어도 되지만 아래에 서술할 증명이 간단하기에 한 번쯤 읽고 넘어가는 것을 추천합니다.



그래프에서 사이클이 존재하지 않기 위해선 트리가 되어야 하지만 트리에는 차수가 홀수(1개)인 리프 노드가 존재하기 때문에 트리가 될 수 없습니다. 그러므로 연결 그래프에서 차수가 홀수인 정점이 없다면 사이클이 존재합니다.

연결 그래프에서 찾은 하나의 사이클에 속한 간선을 모두 지우면 여러 개의 그래프로 분리됩니다. 정점마다 짝수개의 인접한 간선이 제거되므로 각 정점의 차수는 여전히 짝수입니다. 그러므로 각 그래프에서 사이클을 찾을 수 있고 다시 분리할 수 있습니다. 이를 간선이 존재하지 않을 때까지 반복할 수 있습니다.

그러므로 차수가 홀수인 정점이 없다면 사이클이 주렁주렁 달린 그래프라고 볼 수 있습니다.

![](https://user-images.githubusercontent.com/81361400/114190799-ace09980-9986-11eb-8ff4-ed8c8b6013b7.gif)

그리고 사이클은 오일러 회로이고, 두 오일러 회로를 연결해서 새로운 오일러 회로를 찾을 수 있기에 차수가 홀수인 정점이 없는 연결 그래프에서 오일러 회로를 찾을 수 있습니다. 그러므로 연결 그래프에서 차수가 짝수인 정점만 있다면 오일러 회로가 존재합니다.



반대로, 오일러 회로가 존재한다면 연결 그래프에서 차수가 홀수인 정점이 없다는 것을 증명하는 것은 위보다 더 쉽습니다. 그래프에서 오일러 회로가 존재한다면 정점마다 들어오는 간선과 나가는 간선의 개수가 같을 것이기에 홀수인 정점이 있다면 이를 만족하게 할 수 없습니다.



# Hierholzer's Algorithm

Hierholzer's Algorithm은 $O(E)$의 효율적인 시간복잡도를 가지고 있고 내용과 구현이 어렵지 않아 오일러 회로를 구할 때 주로 쓰이는 알고리즘입니다.

알고리즘의 작동 방식은 아래와 같습니다.

1. 시작 정점 v를 선택해서 v로 다시 돌아오는 경로를 찾습니다.
2. 경로에 포함되어 있는 정점 중에 사용되지 않은 간선이 연결된 정점 u를 찾고, 아직 쓰이지 않은 간선들을 사용해 u에서 시작해서 u로 돌아오는 경로를 찾은 뒤 원래의 경로에 삽입합니다.
3. u를 찾을 수 없을 때까지 2를 반복합니다.



작동 방식이 쉽더라도 글로만 읽는 것보다 한번 작동 과정을 보는 것이 알고리즘의 이해를 도와주기에 아래 예시 그래프에서 알고리즘이 작동하는 과정을 설명하겠습니다.



![](https://user-images.githubusercontent.com/81361400/114203427-1e264980-9993-11eb-9106-a50b156c0942.png)

처음에는 어떤 경로도 찾지 못한 상태입니다. 보통 그래프 탐색을 할 때 1번 정점부터 시작하므로 시작 정점을 1번 정점으로 선택하겠습니다.

![](https://user-images.githubusercontent.com/81361400/114203456-27afb180-9993-11eb-8930-7301c7fb0095.png)

1번 정점에서 시작해서 1번 정점으로 돌아오는 경로인 <span style="color:red">**1 - 2 - 3 - 4 - 5 - 6 - 1**</span>을 찾습니다.

![](https://user-images.githubusercontent.com/81361400/114203534-39915480-9993-11eb-9509-416b96957725.png)

<span style="color:brown">**1 - 2 - 3 - 4 - 5 - 6 - 1**</span>에 포함된 정점 중, 6번 정점에 아직 사용되지 않은 간선이 있어 6번 정점에서 시작해서 6번 정점으로 돌아오는 경로인 <span style="color:red">**6 - 2 - 4 - 6**</span>을 찾고 이전에 찾았던 경로에 삽입합니다. 경로는 <span style="color:brown">**1 - 2 - 3 - 4 - 5 - 6**</span> <span style="color:red">**- 2 - 4 - 6**</span> <span style="color:brown">**- 1**</span>이 됩니다.

![](https://user-images.githubusercontent.com/81361400/114203540-3ac28180-9993-11eb-92a6-4e16a1456608.png)

마찬가지로 5번 정점에서 시작해서 5번 정점으로 돌아오는 경로인 <span style="color:red">**5 - 10 - 11 - 5**</span>를 찾고 이전의 경로에 삽입합니다. 경로는 <span style="color:brown">**1 - 2 - 3 - 4 - 5**</span> <span style="color:red">**- 10 - 11 - 5**</span> <span style="color:brown">**- 6 - 2 - 4 - 6 - 1**</span>이 됩니다.

![](https://user-images.githubusercontent.com/81361400/114203548-3bf3ae80-9993-11eb-8ab4-152ae2b0f66d.png)

마지막으로 3번 정점에서도 <span style="color:red">**3 - 7 - 8 - 9 - 3**</span>를 찾고 경로에 삽입하여 최종적으로 오일러 회로인 경로 <span style="color:brown">**1 - 2 - 3**</span> <span style="color:red">**- 7 - 8 - 9 - 3**</span> <span style="color:brown">**- 4 - 5 - 10 - 11 - 5 - 6 - 2 - 4 - 6 - 1**</span>를 찾았습니다.



DFS를 이용해 이 알고리즘을 구현한다면 모든 정점을 순회하면서 간선을 사용하지 않은 정점을 재귀적으로 쉽게 찾을 수 있습니다. 

아래는 오일러 회로의 기본 문제([링크](https://www.acmicpc.net/problem/1199))를 DFS로 구현한 코드입니다.

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int n, v[1010][1010], nxt[1010];

void dfs(int cur){
    for(int &x=nxt[cur];x<=n;x++){ // 봤었던 곳까지는 기록해 놓아야 시간복잡도가 보장됩니다.
        while(x<=n && v[cur][x]){
            v[x][cur]--, v[cur][x]--;
            dfs(x);
        }
    }
    printf("%d ",cur);
}

int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) scanf("%d",&v[i][j]);

    for(int i=1;i<=n;i++){
        int deg = 0;
        for(int j=1;j<=n;j++) deg += v[i][j];
        if(deg % 2) return !printf("-1");
    }

    for(int i=1;i<=n;i++) nxt[i] = 1;
    dfs(n);
}
```

인접 행렬로 구현했기 때문에 코드는 $O(V^2 + E)$의 시간복잡도를 가집니다. $E$의 상한이 약 $V^2$개이므로 기본 문제를 푸는 데에는 지장이 없지만 $V$가 커지면 문제가 될 수 있으므로 시간복잡도를 줄여야 합니다. 큐, 스택, 링크드 리스트 등 insert와 pop을 하는 데에 $O(1)$의 시간이 소요되는 자료구조를 이용해 인접 리스트로 구현한다면 시간복잡도를 줄일 수 있습니다. 스택을 이용해 알고리즘 부분이 $O(E)$로 작동하는 코드를 추가로 첨부하겠습니다. (입력 형식 때문에 전체 시간복잡도는 여전히 $O(V^2 + E)$입니다).

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int n, a[1010][1010], id;
stack<pair<int, int>> v[1010];
vector<int> chk;

void dfs(int u){
    while(1){
        while(!v[u].empty() && chk[v[u].top().second]) v[u].pop(); // 이미 쓰여진 간선이면 pop
        if(v[u].empty()) break;

        auto [x, y] = v[u].top(); v[u].pop();
        chk[y] = 1, dfs(x); // y번 간선을 사용했음을 표시하고 알고리즘을 계속한다.
    }

    printf("%d ",u);
}

int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) scanf("%d",&a[i][j]);

    for(int i=1;i<=n;i++){
    	for(int j=i+1;j<=n;j++){
    		while(a[i][j]){
    			a[i][j]--, id++;
    			v[i].push({j, id}), v[j].push({i, id}); // 간선에 번호를 부여한다.
    		}
    	}
    }
    chk.resize(id + 1);

    for(int i=1;i<=n;i++) if(v[i].size() % 2) return !printf("-1");

    dfs(1);
}
```



# 오일러 경로

그래프의 모든 간선을 단 한 번씩만 지나면서 시작점과 끝점이 다른 경로를 오일러 경로라고 합니다. 오일러 경로 또한 필요충분조건이 잘 알려졌는데, 연결 그래프에서 차수가 홀수인 정점이 두 개 있다면 그 두 정점을 시작점과 끝점으로 하는 오일러 경로가 존재한다는 것입니다.

오일러 회로와 크게 다르지 않은 개념인 만큼, 오일러 경로를 구하는 것은 오일러 회로를 조금만 응용하면 됩니다. 차수가 홀수인 정점이 두 개이므로 그 두 정점을 임의의 간선으로 이어준다면 차수가 홀수인 정점이 없어져서 오일러 회로를 찾을 수 있고, 찾은 오일러 회로에서 추가한 간선을 지우더라도 하나의 경로로 연결되어 있어서 오일러 경로의 조건을 만족합니다. 오일러 회로와 마찬가지로 Hierholzer's Algorithm을 이용한다면 오일러 경로를 어렵지 않게 구할 수 있습니다. 단, 시작 정점은 차수가 홀수인 정점으로 선택해야 합니다.



# 응용 문제

마지막으로 오일러 회로와 경로를 응용하는 문제 몇 개를 설명하면서 마치도록 하겠습니다.



## 퍼레이드 ([링크](https://www.acmicpc.net/problem/16168))

주어진 그래프에 오일러 경로가 존재하는지 확인하는 문제입니다.

그래프가 연결 그래프이며, 차수가 홀수인 정점이 2개 이하인지 확인해주면 됩니다.



## Sebin loves Euler Circuit ([링크](https://www.acmicpc.net/problem/17414))

모든 정점을 방문하는 오일러 회로가 존재하도록 그래프에 간선을 추가하는 문제입니다.

주어진 그래프가 연결 그래프일 때에는(서브태스크 1)에서는 차수가 홀수인 정점끼리 이어주기만 하면 됩니다. 연결 그래프에서 차수가 홀수인 정점은 늘 짝수 개이기 때문에 불가능한 경우는 없습니다.

여러 개의 분리된 그래프가 주어진다면 분리된 그래프끼리 연결하는 간선을 먼저 추가해서 하나의 연결 그래프로 만든 뒤에 차수가 홀수인 정점끼리 이어주면 됩니다. 하나의 연결 그래프로 만들 때 분리된 그래프의 모든 정점의 차수가 짝수인 경우도 유의하면서 구현하면 됩니다.



## 도로 청소 ([링크](https://www.acmicpc.net/problem/1743))

간단하게 말하면 두붓그리기를 하는 문제입니다.

오일러 경로를 임의의 간선을 추가해서 오일러 회로를 찾은 뒤 분리하는 방식으로 구했듯이 차수가 홀수인 정점끼리 잇는 임의의 간선을 $K$개 추가한 뒤에 오일러 회로를 찾고 임의의 간선으로 분리한다면 $K$개의 경로를 찾을 수 있습니다. 그러므로 차수가 홀수인 정점의 개수가 $K\times2$개라면 $K$개의 경로로 분리할 수 있습니다. 이 문제에서는 두붓그리기만을 요구하기에 차수가 홀수인 정점이 4개 이하인지 확인한 뒤에 정답이 되는 경로를 구하면 됩니다. (경로를 찾을 필요 없이 오일러 경로가 되기 위해 추가해야 하는 간선의 개수만 세는 [문제](https://www.acmicpc.net/problem/18250)도 있습니다.)

풀이 자체는 간단하지만, 구현은 까다로울 수 있습니다. 구현하는 데에 도움이 되기를 바라면서 제 코드를 첨부하겠습니다.

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll = long long;

int tc, n, m, c[1010], s[50050], e[50050], chk[50050];
queue<pair<int, int>> q[1010];
queue<int> ans[3], out;

void dfs(int u, int k){
    while(1){
        while(!q[u].empty() && chk[abs(q[u].front().second)]) q[u].pop();
        if(q[u].empty()) break;
        auto [x, y] = q[u].front(); q[u].pop();
        chk[abs(y)] = 1;
        dfs(x, y);
    }
    if(k) out.push(-k);
}

int main(){
    ios_base::sync_with_stdio(0);cin.tie(0);
    cin >> tc;
    while(tc--){
        cin >> n >> m;

        while(!out.empty()) out.pop();
        for(int i=0;i<3;i++) while(!ans[i].empty()) ans[i].pop();
        memset(c, 0, sizeof(c)), memset(chk, 0, sizeof(chk));
        for(int i=1;i<=n;i++) while(!q[i].empty()) q[i].pop();

        for(int i=1;i<=m;i++) cin >> s[i];
        for(int i=1;i<=m;i++) cin >> e[i];

        for(int i=1;i<=m;i++){
            c[s[i]]++, c[e[i]]++;
            q[s[i]].push({e[i], i}), q[e[i]].push({s[i], -i});
        }

        queue<int> in;
        for(int i=1;i<=n;i++) if(c[i] % 2) in.push(i);

        if(m == 1 || in.size() > 4){
            cout << "0\n0\n";
            continue;
        }

        for(int i=1;!in.empty();i++){
            int s = in.front(); in.pop();
            int e = in.front(); in.pop();
            q[s].push({e, m+i}), q[e].push({s, -m-i});
        }

        dfs(1, 0);

        int k = 0;
        while(!out.empty()){
            int x = out.front(); out.pop();
            if(abs(x) > m) k++;
            else ans[k].push(x);
        }

        while(!ans[0].empty()) ans[2].push(ans[0].front()), ans[0].pop();
        for(int i=0;i<2;i++) if(ans[2-i].empty()) ans[2-i].push(ans[2-!i].front()), ans[2-!i].pop();

        for(int i=1;i<3;i++){
            cout << ans[i].size() << " ";
            while(!ans[i].empty()) cout << ans[i].front() << " ", ans[i].pop();
            cout << "\n";
        }
    }
}
```



## 옥상 정원 ([링크](https://www.acmicpc.net/problem/17446))

격자에서 이동 방향을 매번 바꾸는 오일러 경로를 찾는 문제입니다.

세로 방향, 가로 방향으로 번갈아 가면서 이동해야 합니다. 그리고 격자 그래프는 이분 그래프입니다. 이분 그래프의 한쪽 부분은 가로 방향으로만 나가게 하고 나머지 부분은 세로 방향으로 나가도록 방향성을 준다면 그래프를 탐색할 때 이동 방향을 매번 바꾸는 경로만 존재하게 됩니다. 그렇게 만들어진 방향 그래프에서 오일러 경로를 구하면 정답을 찾을 수 있습니다.

방향 그래프에서 오일러 경로를 찾는 것은 무향 그래프와 별반 다를 게 없습니다. 무향 그래프에서 차수가 짝수인 정점은 들어오는 간선과 나가는 간선의 개수가 같게 경로를 정해줄 수 있기에 조건에 사용되었던 것이므로 이 조건은 방향 그래프에서 indegree = outdegree 로 볼 수 있습니다. 또한 차수가 홀수인 정점은 각 정점을 시작점과 끝점으로 정해주기 위해서 찾았던 것으로, 이 두 점은 각각 outdegree - indegree = 1, indegree - outdegree = 1인 조건으로 바꿀 수 있습니다. 방향 그래프가 위에서 찾은 세 조건에 만족하는지만 확인한다면 오일러 경로가 존재하는지 확인할 수 있고, 존재한다면 무향 그래프와 마찬가지로 Hierholzer's Algorithm으로 경로를 찾을 수 있습니다.

가로 방향으로 나가는 부분과 세로 방향으로 나가는 부분을 바꾼다면 indegree와 outdegree를 바꿀 뿐으로, 바꾸는 것만으로 정답의 존재 여부가 달라지지 않으므로 한 번만 확인해주면 됩니다.

