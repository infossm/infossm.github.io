---
layout: post
title: "PS Training 2"
date: 2019-09-17 23:00:00
author: tataky
tags: [PS]
---



안녕하세요, 이번 달에도 Problem Solving 관련 글을 쓰게 되었습니다. 최근에는 참가한 대회가 없지만, 연습 과정에서 만난 문제들 중 괜찮은 문제들을 추려 모아 보았습니다. 이번에도 문제의 풀이와 소스 코드, 링크 등을 정리합니다.



# Circling Round Treasures

Codeforces Round #221 (Div. 1)의 C번 문제입니다.

$$N * M$$ 격자 그리드 안에 다음과 같은 것들이 놓여 있습니다.

- 폭탄
- 빈 칸
- 보물
- 장애물

주인공은 주어진 어떤 시작점에서 출발해, 상하좌우 중 빈 칸으로 이동할 수 있습니다. 이동하는 데는 1원이 듭니다. 이동을 마치고 시작점으로 돌아왔을 때, 주인공의 경로로 감싸진 영역의 내부에 있는 보물의 가치 합에서 이동 비용을 뺀 값을 최대화 하세요. 단, 주인공의 경로로 만들어진 영역 내부에 폭탄이 하나라도 있으면 안 됩니다.

주인공은 단순 다각형을 만들 필요가 없고, 시작점을 지나쳐 여행을 계속하는 것도 가능합니다.

시작하자마자 여행을 끝내는 것도 가능하며, 이 경우엔 아무 보물도, 폭탄도 얻지 않은 것으로 칩니다.

제약조건은 아래와 같습니다.

$$ 1 <= N, M <= 20 $$, 폭탄과 보물의 개수 합은 8개를 넘지 않음, 보물의 가치는 -200 이상 200 이하인 정수

입력은 아스키 문자 그리드 형태로 주어지며, 시작점은 'S', 폭탄은 'B', 빈 칸은 '.', 보물은 한 자리 숫자로 주어집니다. 그리드가 모두 입력된 뒤, 주어진 보물의 개수만큼, 보물의 가치가 공백으로 구분되어 주어지게 됩니다.

예를 들어,

```
7 7
.......
.1###2.
.#...#.
.#.B.#.
.3...4.
..##...
......S
100
100
100
100
```

위와 같은 그리드에서는 아래처럼 이동할 수 있습니다.

![treasure1](\assets\images\tataky_0918\treasure1.png)

위의 경로에서는 가치가 100인 보물 1, 2, 3, 4를 각각 얻게 되며, 이동 경로의 길이는 36이므로 총 364의 이득을 얻게 됩니다. 또한, 이것이 최적해 중 하나입니다.



## 풀이

우선, 폭탄을 가치가 -10000000인 보물인 것으로 둡니다. 이렇게 할 경우, 폭탄을 포함하는 해는 자동으로 최적해가 아니게 되어 예외처리가 필요없게 됩니다.

풀이에 앞서, 어떤 점이 다각형 내부에 있는지 판정하는 방법에는 여러 가지가 있지만, 그 중 이 문제에서 굉장히 유용하게 쓰이는 방법 하나가 있습니다. 가장 간단한 케이스부터 하나씩 살펴보면 다음과 같습니다.

![treasure2](\assets\images\tataky_0918\treasure2.png)

사각형 내부에 있는 점 A와 외부에 있는 점 B입니다. 일반적으로, 위와 같은 볼록 다각형에서 어떤 점이 다각형의 내부에 있는지를 판정하려면, 볼록다각형의 각 꼭지점을 시계 반대 방향으로 순회하며, 인접한 두 꼭지점이 만드는 방향성 있는 선분에 대해 점이 항상 반시계 방향에 있는지 체크하면 됩니다. 이것이 가능한 이유는, 볼록 다각형은 모든 내각이 180도 이하이기 때문에, 어떤 선분을 잡더라도 항상 그 반시계 방향에만 볼록 다각형의 내부가 존재하기 때문입니다.

하지만 위와 같은 방법은 이 문제에서는 사용할 수가 없습니다. 우선, 만들어질 다각형이 볼록 다각형이 아니며, 다각형이 완성된 뒤에야 판정이 가능한 방법이기 때문입니다. 따라서, 위와 같은 방법보다 더 일반적인 방법이 필요합니다.

그러한 방법 중 가장 유명하고, 또 이 문제에 사용하기에 가장 적절한 방법은 Ray Casting입니다. 주어진 점에서 출발하는 어떤 광선 하나를 발사합니다. 이 방향성 있는 반직선이 다각형과 교차한 횟수가 홀수이면 이 점은 다각형 내부에 있으며, 아니면 외부에 있습니다. 그림으로는 아래와 같습니다.

![treasure3](\assets\images\tataky_0918\treasure3.png)

도형의 외부에 있는 점 B에서 출발한 반직선(빨강)은 도형과 0회 또는 2회 교차하므로 짝수번 교차하고, 도형의 내부에 있는 점 A에서 출발한 반직선(파랑)은 도형과 1회 교차합니다. 이 때, 점 A에서 출발한 보라색 반직선처럼 도형의 꼭지점을 정확히 지나는 반직선을 판정에 사용하게 될 경우, 판정이 제대로 되지 않을 가능성이 있습니다. A에서 출발한 반직선이 도형과 두 번 교차한다고, 즉 해당 꼭지점에 인접한 두 선분을 지난다고 판정하게 되거나, B에서 출발한 반직선이 도형과 한 번 교차한다고 판정하게 되는 등의 이슈가 존재합니다. 따라서, 이 방법을 사용할 때에는, 가능하다면 이론상 꼭지점을 절대로 지나지 않는 기울기의 반직선을 이용하면 좋고, 여의치 않다면 랜덤한 반직선 여러 개에 대해 테스트해보는 방식을 사용합니다. 이 문제의 경우, 그리드의 사이즈가 충분히 작기 때문에, 반직선의 기울기를 1/59, 1/71 등으로 잡으면 출발 지점이나 도형의 형태와 상관없이 꼭지점과 절대로 교차하지 않습니다.

이러한 판정의 결과가 항상 옳다는 것은 직관적으로 알 수 있습니다. 다각형의 선분 하나를 지나친다는 것은, 다각형의 내부에 있었다면 외부로 나가고, 외부에 있었다면 내부로 들어가는 것과 동치입니다. 짧게 말해, 내/외부 상태가 flip되는 것입니다. 또한, 무한히 먼 위치의 점은 다각형의 외부에 있음이 자명하므로, 반직선이 충분히 멀리 가는 동안 홀수 번 교차했다면 원래 점은 다각형의 외부와 다른 상태, 즉 내부에 있는 것이고, 짝수 번 교차했다면 다각형의 외부와 같은 상태, 즉 다각형의 외부에 있는 것이라고 할 수 있습니다. 그리고 이 서술은 다각형이 볼록하거나 단순하다는 가정을 하지 않아도 성립하기 때문에, 임의의 도형에 대해서도 똑같이 적용이 가능합니다.

그리고 이러한 방식은, **다각형이 완성되기 전에도** 어떤 점이 결론적으로 다각형 내부에 포함될지, 혹은 그렇지 않을 지를 계속해서 판정해나갈 수 있습니다. 이 방식을 이용하여, 아래와 같이 정의합시다.

```
d[mask][row][col] = 현재 (row,col)에 있고, 각 보물들에서 출발한 어떤 반직선이
현재까지 다각형과 교차한 횟수가 홀수이면 1, 아니면 0으로 만든 8자리 비트마스크가
mask인 상태까지 도달하는 데에 필요한 최소 이동 횟수
```

위와 같이 정의하게 될 경우, 가능한 상태는 총 $${2}^{8}NM$$이며, 4개의 가능한 방향에 대해 한 칸 움직이고, 8개의 보물에서 출발한 반직선과 현재 이동하며 만들어진 길이 1인 선분의 교차 여부를 판정하여 마스크를 변경해주면 되므로, 최종적으로 $$4×8×{2}^{8}NM$$ 정도의 연산량을 가지는 BFS로 위의 배열을 모두 채울 수가 있습니다. 이 연산량은 최대 3276800이고, 통과하기에 매우 넉넉합니다.

배열을 모두 채운 뒤엔, (row,col)이 시작점인 모든 mask에 대해 비용을 계산해 최댓값을 찾아 주기만 하면 됩니다. 미리 각 mask의 비용을 전처리해 두면 이 과정에 걸리는 연산량은 배열을 채우는 연산량에 비해 매우 작기 때문에 신경쓰지 않아도 됩니다.



## 구현

구현은 설명보다는 조금 더 복잡한데, 선분 교차 알고리즘이 필요하기 때문입니다. 여러 가지 방식이 있으나, 저는 실수 오차를 피하기 위해 정수 연산만을 사용했습니다. 선분 A와 선분 B가 교차하려면, 선분 A를 기준으로 B의 양 끝점이 하나는 왼쪽, 하나는 오른쪽에 있어야 하며, 마찬가지로 선분 B를 기준으로 선분 A의 양 끝점이 하나는 왼쪽, 하나는 오른쪽에 있어야 합니다. 사실 왼쪽/오른쪽이라는 표현보다는, 시계/반시계 방향이라고 표현하는 것이 맞으나, 더 직관적인 설명을 위해 방향으로 표현하였습니다.

소스 코드의 전문은 아래와 같습니다. 설명한 대로 BFS를 시도하고 최대 이득을 계산합니다.

```cpp
#include <stdio.h>
#include <string.h>
#include <queue>
#include <tuple>
#include <vector>
#include <algorithm>
 
using namespace std;
 
typedef tuple<int,int,int> ti;
 
int d[20][20][1<<16], n, m;
int dr[4]={-1,0,1,0}, dc[4]={0,1,0,-1};
char b[20][21];
 
bool pos(int r, int c) {
	return 0<=r&&r<n&&0<=c&&c<m&&b[r][c]=='.';
}
 
vector<ti> v;
int r[16][4], val[16];
 
int ccw(int x1, int y1, int x2, int y2, int x3, int y3) {
	int d=x1*y2+x2*y3+x3*y1-(x1*y3+x2*y1+x3*y2);
	if(d<0) return -1;
	else if(d==0) return 0;
	else return 1;
}
 
bool on(int x1, int y1, int x2, int y2, int x3, int y3) {
	return ccw(x1,y1,x2,y2,x3,y3)==0 &&
		min(x1,x2)<=x3 && x3<=max(x1,x2) &&
		min(y1,y2)<=y3 && y3<=max(y1,y2);
}
 
bool cross(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4) {
	if(on(x1,y1,x2,y2,x3,y3) || on(x1,y1,x2,y2,x4,y4) ||
		on(x3,y3,x4,y4,x1,y1) || on(x3,y3,x4,y4,x2,y2)) return true;
	return ccw(x1,y1,x2,y2,x3,y3)*ccw(x1,y1,x2,y2,x4,y4)<0 &&
		ccw(x3,y3,x4,y4,x1,y1)*ccw(x3,y3,x4,y4,x2,y2)<0;
}
 
int main() {
	scanf("%d %d",&n,&m);
	int sr, sc;
	int cc=0;
	for(int i=0;i<n;i++) {
		scanf("%s",b[i]);
		for(int j=0;j<m;j++) {
			if('1'<=b[i][j] && b[i][j]<='9') {
				int w=b[i][j]-'0';
				if(w>cc) cc=w;
			}
		}
	}
	for(int i=0;i<cc;i++)
		scanf("%d",&val[i]);
	for(int i=0;i<n;i++) {
		for(int j=0;j<m;j++) {
			if(b[i][j]=='S') {
				sr=i; sc=j;
				b[i][j]='.';
			} else if(b[i][j]=='B') {
				r[v.size()][0]=i; r[v.size()][1]=j;
				r[v.size()][2]=37; r[v.size()][3]=1999;
				v.push_back(ti(i,j,-1234567));
			} else if('1'<=b[i][j] && b[i][j]<='9') {
				r[v.size()][0]=i; r[v.size()][1]=j;
				r[v.size()][2]=37; r[v.size()][3]=1999;
				v.push_back(ti(i,j,val[b[i][j]-'1']));
			}
		}
	}
	memset(d,-1,sizeof(d));
	queue<ti> q;
	d[sr][sc][0]=0;
	q.push(ti(sr,sc,0));
	while(!q.empty()) {
		int cr=get<0>(q.front()), cc=get<1>(q.front()), h=get<2>(q.front());
		q.pop();
		for(int i=0;i<4;i++) {
			int nr=cr+dr[i], nc=cc+dc[i], nh=h;
			if(!pos(nr,nc)) continue;
			for(int j=0;j<v.size();j++) {
				if(cross(cr,cc,nr,nc,r[j][0],r[j][1],r[j][2],r[j][3]))
					nh^=(1<<j);
			}
			if(d[nr][nc][nh]<0) {
				d[nr][nc][nh]=d[cr][cc][h]+1;
				q.push(ti(nr,nc,nh));
			}
		}
	}
	int res=0;
	for(int i=0;i<(1<<v.size());i++) {
		if(d[sr][sc][i]>=0) {
			int sum=0;
			for(int j=0;j<v.size();j++) if(i&(1<<j))
				sum+=get<2>(v[j]);
			int w=sum-d[sr][sc][i];
			if(w>res) res=w;
		}
	}
	printf("%d\n",res);
	return 0;
}
```





# Looking for Owls

Codeforces Round #203 (Div. 2) 의 D번입니다. 앞 문제에 이어, 또 기하 분야입니다.

좌표평면 위에 선분 $$N$$개와 원 $$M$$개가 놓여 있습니다. 우리는 **올빼미**를 다음과 같이 정의합니다.

* 올빼미는 원 두 개(i, j)와 선분 하나(k)로 이루어진다.
* 두 원은 단 한 개의 점도 공통으로 가져서는 안 되며, 선분 k에 대해 선대칭이어야 한다.
* 선분 k는 두 원의 중심을 잇는 선분과 교점이 있어야 한다.
* i와 j는 반지름이 동일한 원이어야 한다.

몇 개의 예시를 살펴보면 다음과 같습니다.

![owl1](\assets\images\tataky_0918\owl1.png)

위는 올바른 올빼미의 예시입니다. 선분을 기준으로 반지름이 동일하며 교점이 없는 두 원이 선대칭을 이루고 있으며, 선분은 두 원을 중심을 잇는 선분과 만납니다.

![owl2](\assets\images\tataky_0918\owl2.png)

위는 올빼미가 아닌 예시입니다. 선분을 기준으로 반지름이 동일하며 교점이 없는 두 원이 선대칭을 이루고 있지만, 선분이 두 원의 중심을 잇는 선분과 만나지 않습니다.

![owl3](\assets\images\tataky_0918\owl3.png)

위 또한 올빼미가 아닌 예시입니다. 두 원이 선분을 기준으로 선대칭이 아닙니다.

제한은 아래와 같습니다.

$$1 <= N <= 300000, 2 <= M <= 1500$$, 각 좌표와 원의 반지름의 범위는 절댓값이 10000 이하이며, 길이가 0인 선분 또는 반지름이 0인 원은 존재하지 않는다. 또한, 주어진 모든 선분들과 원들 중 완전히 동일한 것은 없다.



# 풀이

원의 개수가 선분의 개수에 비해 훨씬 적은 것에 집중합시다. 올빼미는 원 두 개와 선분 하나로 이루어지므로, 두 원을 선택하고, 이 두 원과 함께 묶여 올빼미가 될 수 있는 선분의 개수를 빠르게 세는 쪽으로 접근하면 풀이가 보이게 됩니다.

우선, 원 두 개를 고릅니다. 두 원의 반지름이 다르거나, 두 원의 교점이 존재하면 (두 원 중심 사이의 거리가 반지름의 합보다 작거나 같으면) 자명히 올빼미가 아니므로 건너뜁니다. 그렇지 않다면, 두 원 사이에 놓여 올빼미를 만들어 줄 선분의 개수를 세면 됩니다.

원 두 개가 고정되면, k의 후보가 될 선분의 기울기는 바로 결정됩니다. 두 원의 중심을 잇는 선분과 수직이어야만 두 원을 선대칭으로 만들 수 있기 때문에, 기울기를 곧바로 얻어낼 수 있습니다. 또한, 선대칭을 만들기 위해서는 반드시 두 원의 중심을 잇는 선분의 **중점**을 지나야 하므로, 평행이동에 관한 상수값도 바로 결정됩니다. 즉, 어떤 유일한 직선 위에 존재하며, 그 위에서 두 원의 중심을 잇는 선분과 교차한다는 조건을 만족하는 선분의 개수를 셀 수 있으면 됩니다. 그림으로 표현하면 아래와 같습니다.

![owl4](\assets\images\tataky_0918\owl4.png)

원 두 개가 결정되었다면 위의 빨간 점선 위에 놓여 있는 선분들 중, 두 원의 중심을 가로지르는 선분과 교점이 존재하는 것의 개수를 세면 됩니다. 선분은 유일하지 않으나, 그러한 선분이 놓일 수 있는 영역을 의미하는 직선은 빨간 점선에 해당하는 하나로 유일합니다.

이 문제에서는 $$y=ax+b$$ 꼴의 직선 대신, $$ax+by+c=0$$ 꼴의 암시적 직선을 사용하는 것이 훨씬 편리합니다. 기울기가 무한대인 직선을 따로 예외처리하지 않을 수 있기 때문입니다. 우리는 주어진 모든 선분에 대해, 그 선분을 포함하는 유일한 직선 $$ax+by+c=0$$을 미리 계산합니다. 이 때, 이러한 직선의 표현은 $$ax+by+c=0$$과 $$-ax-by-c=0$$ 꼴로 두 가지가 존재하므로, $$a, b, c$$중 0이 아닌 가장 앞 값이 양수인 표현 형태 하나만 올바른 것으로 둡니다. 또한 어떤 상수 $$k$$에 대해 $$kax+kby+kc=0$$ 꼴의 직선도 동일한 직선이기 때문에, $$a, b, c$$의 최대공약수로 $$a, b, c$$를 나누어 주고 계산해야 합니다.

이렇게, 각 선분을 해당 선분이 놓여 있는 직선 $$ax+by+c=0$$에 대해 $$(a,b,c)$$ 튜플을 키로 하는 자료구조에 담아 관리하면, 후보 직선이 얻어졌을 때, 해당 직선의 $$(a,b,c)$$값을 계산하여 후보가 될 선분만 모여 있는 장소에 바로 접근이 가능합니다. cpp의 경우, map 등을 사용하면 편리합니다.

이제 대칭, 기울기 등의 모든 조건은 고려하지 않아도 됩니다. $$(a,b,c)$$를 계산하여 다른 모든 조건을 만족하는 선분들의 목록을 얻었기 때문에, 두 원의 중심을 잇는 선분과 교차할 수 있는 선분의 개수만 세면 됩니다.

이는 일종의 구간 쿼리로 볼 수 있습니다. 입력되는 모든 선분에 전처리로 일정하게 방향성을 부여하고 (위->아래 혹은 왼쪽->오른쪽 등), 각 $$(a,b,c)$$ 더미마다, 해당 더미에 포함된 선분들의 시작점을 따로 모아 정렬, 끝점을 따로 모아 정렬해둡시다. 그렇게 하고 나면, 두 원의 중점을 기준으로, 그 중점 뒤에 존재하는 시작점과 그 중점 앞 존재하는 끝점의 개수를 따로 세어 합해 주면, 답이 되지 "않을" 선분의 개수를 셀 수 있습니다. 두 집합은 disjoint하기 때문에 따로 세어 합해도 무방하며, 이 과정은 바이너리 서치를 통해 쉽게 할 수 있습니다. 답이 될 선분의 개수는 더미에 존재하는 전체 선분의 수에서 방금 구한 답이 되지 않을 선분의 수를 빼 주면 됩니다.

최종 시간복잡도는 $$O({M}^{2}logN)$$이 됩니다. 상수가 조금 큰 편이지만, 시간 제한이 넉넉하여 충분히 통과합니다.



## 구현

구현은 충실하게 위의 과정을 따르면 됩니다. convert 함수는앞서 설명한 방식대로 직선을 정제하여 유일한 표현을 만드는 함수이며, cnt 함수는 어떤 $$(a,b,c)$$ 더미 내에서 바이너리 서치를 진행하는 함수입니다.

```cpp
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
 
using namespace std;
 
typedef long long lli;
typedef pair<lli, lli> ip;
typedef tuple<int, int, int> ti;
 
lli gcd(lli a, lli b) {
	return b ? gcd(b, a%b) : a;
}
 
class p {
public:
	lli x, y;
};
 
class seg {
public:
	p s, e;
};
 
class cc {
public:
	p c;
	lli r;
};
 
seg a[300000];
cc b[1500];
int n, m;
 
ti convert(lli dx, lli dy, lli x1, lli y1) {
	lli a = dy, b = -dx, c = dx*y1 - dy*x1;
	lli g = gcd(abs(a), gcd(abs(b), abs(c)));
	a /= g; b /= g; c /= g;
	if (a == 0) {
		if (b < 0) {
			b = -b; c = -c;
		}
	}
	else if (b == 0) {
		if (a < 0) {
			a = -a; c = -c;
		}
	}
	else {
		if (a < 0) {
			a = -a; b = -b; c = -c;
		}
	}
	return ti(a, b, c);
}
 
map<ti, vector<lli> > st, ed;
 
lli cnt(vector<lli>& v, lli pv, bool big) {
	if (v.empty()) return 0;
	if (big) {
		if (v.back() <= pv) return 0;
		if (v[0] > pv) return v.size();
		int lo = 0, hi = v.size() - 1;
		while (lo + 1 < hi) {
			int mid = (lo + hi) / 2;
			if (v[mid] > pv) hi = mid;
			else lo = mid;
		}
		return v.size() - hi;
	}
	else {
		if (v[0] >= pv) return 0;
		if (v.back() < pv) return v.size();
		int lo = 0, hi = v.size() - 1;
		while (lo + 1 < hi) {
			int mid = (lo + hi) / 2;
			if (v[mid] < pv) lo = mid;
			else hi = mid;
		}
		return lo + 1;
	}
}
 
lli process(cc c1, cc c2) {
	if (c1.r != c2.r) return 0;
	if ((c1.c.x - c2.c.x)*(c1.c.x - c2.c.x) + (c1.c.y - c2.c.y)*(c1.c.y - c2.c.y) <= 4 * c1.r*c1.r) return 0;
	ti t = convert(c1.c.y - c2.c.y, c2.c.x - c1.c.x, (c1.c.x + c2.c.x) / 2, (c1.c.y + c2.c.y) / 2);
	map<ti, vector<lli> >::iterator it = st.find(t);
	if (it == st.end()) return 0;
	lli res = it->second.size();
	if (get<1>(t) == 0) {
		lli pv = (c1.c.y + c2.c.y) / 2;
		res -= cnt(it->second, pv, true);
		res -= cnt(ed[t], pv, false);
	}
	else {
		lli pv = (c1.c.x + c2.c.x) / 2;
		res -= cnt(it->second, pv, true);
		res -= cnt(ed[t], pv, false);
	}
	return res;
}
 
int main() {
	scanf("%d %d", &n, &m);
	for (int i = 0; i < n; i++) {
		scanf("%lld %lld %lld %lld", &a[i].s.x, &a[i].s.y, &a[i].e.x, &a[i].e.y);
		a[i].s.x *= 2; a[i].s.y *= 2; a[i].e.x *= 2; a[i].e.y *= 2;
	}
	for (int i = 0; i < m; i++) {
		scanf("%lld %lld %lld", &b[i].c.x, &b[i].c.y, &b[i].r);
		b[i].c.x *= 2; b[i].c.y *= 2; b[i].r *= 2;
	}
	for (int i = 0; i < n; i++) {
		ti m = convert(a[i].e.x-a[i].s.x,a[i].e.y-a[i].s.y,a[i].s.x,a[i].s.y);
		if (a[i].s.x == a[i].e.x) {
			st[m].push_back(min(a[i].s.y, a[i].e.y));
			ed[m].push_back(max(a[i].s.y, a[i].e.y));
		}
		else {
			st[m].push_back(min(a[i].s.x, a[i].e.x));
			ed[m].push_back(max(a[i].s.x, a[i].e.x));
		}
	}
	for (map<ti, vector<lli> >::iterator it = st.begin(); it != st.end(); it++) {
		sort(it->second.begin(), it->second.end());
	}
	for (map<ti, vector<lli> >::iterator it = ed.begin(); it != ed.end(); it++) {
		sort(it->second.begin(), it->second.end());
	}
	lli res = 0;
	for (int i = 0; i < m; i++) {
		for (int j = i + 1; j < m; j++) {
			res += process(b[i], b[j]);
		}
	}
	printf("%lld\n", res);
	return 0;
}
```





# The Shortest Statement

Educational Codeforces Round 51의 F번입니다.

문제 제목처럼 매우 간결한 문제입니다.

$$N$$개의 정점과 $$M$$개의 간선으로 이루어진, 가중치 있는 무향 연결 그래프가 주어집니다. 이 때, 두 정점 $$U$$와 $$V$$가 주어지면, $$U$$에서 $$V$$로 가는 최단 경로의 길이를 출력하는 질의를 $$Q$$번 처리하세요.

제약 조건은 아래와 같습니다.

$$ 1 <= N, M <= 100000, M - N <= 20, 1 <= Q <= 100000 $$



## 풀이

일반적인 것과 다른 제약 조건 하나가 있습니다. 간선의 수가 정점의 수 + 20을 넘지 않는다는 조건입니다. 이 조건이 없다면 (잘 알려진 방법으로는) 풀 수 없는 문제이기 때문에, 분명히 이 조건을 사용해야 할 것 같습니다.

일단, 연결 그래프이므로 이 그래프에는 스패닝 트리가 존재합니다. 임의의 스패닝 트리 하나를 만듭시다. 이 트리는 $$N-1$$개의 간선으로 이루어져 있기 때문에, 스패닝 트리에 쓰이지 않은 간선의 최대 개수는 많아야 21개입니다.

우리가 원하는 최단 경로는 다음과 같이 분류할 수 있습니다.

* 스패닝 트리 위에서만 움직인다
* 스패닝 트리 외에도, 트리에 포함되지 않은 21개의 간선을 하나 이상 사용한다

첫 번째 케이스의 경우, 트리에서의 최단 경로는 LCA를 활용하면 $$O(logN)$$에 찾을 수 있습니다. 따라서, 이 값이 답의 후보가 됩니다.

이제 두 번째 경우인, 스패닝 트리에 포함되지 않은 어떤 간선 중 적어도 하나를 반드시 이용하는 경우만 생각하면 됩니다.

그렇게 사용되는 간선 중 어떤 임의의 하나를 $$(s,t)$$ 라고 합시다. 우리의 최단 경로는 $$U$$->$$s$$->$$t$$->$$V$$의 형태로 생겼으며, $$s$$->$$t$$의 경우엔 간선 하나가 됩니다. 이 때, 그래프가 무향 그래프이기 때문에, $$U$$->$$s$$ 최단 경로의 길이는 $$s$$->$$U$$ 최단 경로의 길이와 같습니다.

따라서, $$(s,t)$$가 될 수 있는 모든 정점, 많아야 42개의 정점에 대해, 해당 정점을 시작점으로 다익스트라 알고리즘을 초기에 실행해 둡니다. 스패닝 트리와는 관련이 없으며, 이제 이 42개의 정점과 다른 임의의 정점을 잇는 최단 경로는 O(1)에 알 수 있게 됩니다. 이제, 쿼리마다 다음의 작업을 진행하기만 하면 됩니다.

* 스패닝 트리 위에서의 최단 경로 계산
* 트리에 없는 21개의 간선 $$(s,t)$$에 대해, ($$s$$->$$U$$ 최단 경로) + ($$t$$->$$V$$ 최단 경로) + ($$s$$->$$t$$ 간선 길이)를 계산.

이렇게 만들어진 후보 경로들 중 가장 가중치 합이 작은 것 하나를 택하면 됩니다. 이 때, $$(s,t)$$는 $$(t,s)$$ 로도 읽을 수 있으므로, 최단거리의 후보는 총합 43개가 됩니다. 이것들을 모두 계산하면 되며, 시간복잡도는 전처리 과정에서 다익스트라 알고리즘에 $$O(43MlogN)$$, LCA 테이블 제작에 $$O(NlogN)$$, 그 이후 쿼리당 $$O(42+logN)$$이 되어 최종 $$O(43MlogN + Q(42+logN))$$이 됩니다.



## 구현

스패닝 트리를 만들고, LCA 테이블과 다익스트라 전처리를 마친 뒤, 쿼리마다 모든 후보를 살펴봅니다. 여러 개의 알고리즘을 구현해야 하기 때문에 코드는 긴 편이지만, 모두 기본적인 알고리즘이기 때문에 어렵지 않게 구현할 수 있습니다.

```cpp
#include <stdio.h>
#include <string.h>
#include <vector>
#include <queue>
#include <utility>
#include <functional>
#include <algorithm>
#include <assert.h>
 
using namespace std;
 
typedef long long lli;
typedef pair<lli,int> ip;
 
class e {
public:
	int v, w;
	e(int v, int w)
	:v(v),w(w)
	{}
};
 
class edge {
public:
	int u, v, w;
	edge(int u, int v, int w)
	:u(u),v(v),w(w)
	{}
};
 
bool cmp(const edge& i, const edge& j) {
	return i.w<j.w;
}
 
vector<edge> elist;
vector<e> con[100001], conp[100001];
vector<int> unv;
 
lli dd[44][100001];
 
void dij(int s) {
	int st=unv[s];
	priority_queue<ip,vector<ip>,greater<ip> > q;
	q.push(ip(0,st));
	dd[s][st]=0;
	while(!q.empty()) {
		lli dis=q.top().first;
		int u=q.top().second;
		q.pop();
		if(dd[s][u]<dis) continue;
		for(int i=0;i<conp[u].size();i++) {
			int v=conp[u][i].v, w=conp[u][i].w;
			if(dd[s][v]<0 || dd[s][v]>dis+w) {
				dd[s][v]=dis+w;
				q.push(ip(dd[s][v],v));
			}
		}
	}
}
 
int up[100001];
 
int fd(int u) {
	if(up[u]==0) return u;
	return up[u]=fd(up[u]);
}
 
void un(int u, int v) {
	up[fd(v)]=fd(u);
}
 
int dep[100001];
int par[17][100001];
lli d[100001];
 
int lca(int u, int v) {
	if(u==v) return u;
	if(dep[u]>dep[v]) swap(u,v);
	for(int i=16;i>=0;i--) {
		if(dep[par[i][v]]>=dep[u]) {
			v=par[i][v];
		}
	}
	if(u==v) return u;
	for(int i=16;i>=0;i--) {
		if(par[i][u]!=par[i][v]) {
			u=par[i][u]; v=par[i][v];
		}
	}
	return par[0][u];
}
 
lli qry(int u, int v) {
	return d[u]+d[v]-2LL*d[lca(u,v)];
}
 
int n, m;
 
void setlca() {
	for(int h=1;h<17;h++) {
		for(int i=1;i<=n;i++) {
			par[h][i]=par[h-1][par[h-1][i]];
		}
	}
}
 
void dfs(int pre, int u, lli l, int dp) {
	d[u]=l;
	dep[u]=dp++;
	for(int i=0;i<con[u].size();i++) {
		int v=con[u][i].v, w=con[u][i].w;
		if(pre==v) continue;
		par[0][v]=u;
		dfs(u,v,l+w,dp);
	}
}
 
bool used[100001];
 
int main() {
	scanf("%d %d",&n,&m);
	for(int i=0;i<m;i++) {
		int u, v, w;
		scanf("%d %d %d",&u,&v,&w);
		elist.push_back(edge(u,v,w));
		conp[u].push_back(e(v,w));
		conp[v].push_back(e(u,w));
	}
	sort(elist.begin(),elist.end(),cmp);
	for(int i=0;i<m;i++) {
		int u=elist[i].u, v=elist[i].v, w=elist[i].w;
		if(fd(u)!=fd(v)) {
			un(u,v);
			con[u].push_back(e(v,w));
			con[v].push_back(e(u,w));
			used[i]=true;
		}
	}
	dfs(-1,1,0,1);
	setlca();
	int cnt=0;
	for(int i=0;i<m;i++) if(!used[i]) {
		unv.push_back(elist[i].u);
		unv.push_back(elist[i].v);
		cnt++;
	}
	assert(cnt+n-1==m);
	sort(unv.begin(),unv.end());
	unv.erase(unique(unv.begin(),unv.end()),unv.end());
	memset(dd,-1,sizeof(dd));
	for(int i=0;i<unv.size();i++)
		dij(i);
	int q;
	scanf("%d",&q);
	while(q--) {
		int u, v;
		scanf("%d %d",&u,&v);
		if(u==v) {
			puts("0");
			continue;
		}
		lli res=qry(u,v);
		for(int i=0;i<unv.size();i++) {
			lli w=dd[i][u]+dd[i][v];
			if(w<res) res=w;
		}
		printf("%lld\n",res);
	}
	return 0;
}
```



# 마무리

이번에도, 각 문제를 풀어볼 수 있는 링크를 첨부하며 마치겠습니다.

* Circling Round Treasures : https://codeforces.com/contest/375/problem/C
* Looking for Owls : https://codeforces.com/contest/350/problem/D

* The Shortest Statement : https://codeforces.com/contest/1051/problem/F



감사합니다.