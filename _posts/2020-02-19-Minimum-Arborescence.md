---
layout: post
title:  "Minimum Arborescence"
date: 2020-02-19 23:00
author: junodeveloper
tags: [graph theory]
---

안녕하세요. 이번 글에서는 weighted directed graph에서 minimum arborescence를 찾는 알고리즘을 소개해드리려고 합니다. minimum arborescence는 minimum spanning tree의 directed 버전이라고 할 수 있습니다.

## 문제

가중치 있는 방향 그래프 $G=(V,E)$ 와 루트 정점 $r\in V$이 주어집니다. 가중치는 $e\in E$에 대해 $w(e)$로 정의됩니다. 이때 모든 정점 $u$ 에 대하여 $r\rightarrow u$의 경로가 유일하게 존재하며, 가중치의 합이 최소가 되도록 $\|V\|-1$개의 간선들을 적절히 선택하는 것이 목표입니다. 편의상 loop와 multi edge는 존재하지 않고, 모든 정점이 $r$ 에서 도달 가능하다고 가정하겠습니다.

해당 문제는 다음 링크에서 연습해볼 수 있습니다. http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_2_B

## Chu-Liu-Edmonds' algorithm

이 문제를 해결하는 대표적인 알고리즘으로는 $O(\|V\|\|E\|)$ 시간에 동작하는 Chu-Liu-Edmonds' algorithm이 있습니다. 이 알고리즘은 크게 두 가지 스텝으로 나뉩니다.

### Step 1. 모든 정점 $u\neq r$  에 대하여, $u$로 들어오는 간선들 중 가중치가 최소인 것을 선택한다.

### Step 2. 선택된 간선들 중 일부가 싸이클을 형성한다면, 해당 싸이클을 하나의 정점으로 묶고 Step 1으로 돌아간다.

먼저 Step 1에서 선택한 간선들이 싸이클을 형성하지 않는다고 가정해봅시다. 이 경우 각 정점에서 간선을 따라 올라가면 항상 루트에 도달하고, 그 경로는 유일합니다. 또한 항상 최소인 간선만을 선택했으므로 가중치의 합이 최소임을 알 수 있습니다. 따라서 단순히 선택된 가중치들의 합을 리턴하면 됩니다.

만약 싸이클이 하나라도 존재한다면, 해당 싸이클은 루트로 도달할 수 없기 때문에 반드시 내부에서 최소 한 개의 간선을 끊고 외부와 연결해 주어야 합니다. 이를 위해 싸이클을 하나의 정점으로 묶고, 남은 간선들에 대해 재귀적으로 문제를 해결할 것입니다.

싸이클 정점 집합을 $C=\{a_1,a_2,...,a_k\}$, 이들을 묶을 새로운 정점을 $a_C$ 라고 합시다. 또한 싸이클 내의 정점 $u$에 대하여 $p(u)=(u$  바로 이전의 정점$)$ 으로 정의하겠습니다 (즉, $p(a_i)=a_{i-1}$ ). 기존의 모든 간선 $(u,v)\in E$ 를 순회하며 다음의 규칙에 따라 새로운 간선 집합 $E'$과 가중치 $w'$을 구합니다.

#### i) $u\in C, v\notin C$ : $E'\leftarrow E' \cup (a_C, v), w'(a_C,v)=w(u,v)$

#### ii) $u\notin C,v\in C$ : $E'\leftarrow E' \cup (u,a_C), w'(u,a_C)=w(u,v)-w(p(v),v)$

#### iii) $u\notin C, v\notin C$ : $E'\leftarrow E' \cup (u,v), w'(u,v)=w(u,v)$

$u\in C, v\in C$ (싸이클 내부의 간선)인 경우 $E'$에 추가하지 않습니다.

ii)의 가중치 식에서 알 수 있듯이, 이후 재귀적으로 문제를 해결하는 과정에서 $C$로 들어오는 간선을 선택할 경우 기존 싸이클 간선 하나를 제거하게 됩니다. 따라서 모든 정점의 indegree가 1이 되도록 유지할 수 있습니다.

이제 기존의 $V$ 대신 $V'=V\setminus C \cup \{a_C\}$ , $E$ 대신 $E'$, 그리고 $w$ 대신 $w'$을 대입하여 Step 1로 돌아가 재귀적으로 문제를 해결합니다. 다음 단계 재귀에서 해결한 답이 $F'$이었다면, 원래 문제의 답은 $F=F'+$($C$에 포함된 간선들의 가중치 합) 이 됩니다.

## 구현

Step 1의 경우 간선들을 순회하며 각 정점의 최솟값 배열을 갱신하는 방식으로 $O(\|V\|+\|E\|)$ 시간에 간단하게 처리할 수 있습니다. 이때 정점 $u$로 들어오는 간선의 가중치 최솟값을 $mn(u)$라고 하겠습니다.

Step 2에서 싸이클을 추출하는 부분은 DFS 등의 방법으로 $O(\|V\|+\|E\|)$ 시간에 처리할 수 있습니다. 이때 추출한 모든 싸이클에 번호를 매기고, $grp(u)$는 $u$가 속한 싸이클의 번호라고 하겠습니다. 싸이클에 속해있지 않은 정점들의 경우 다른 유니크한 번호를 부여합니다. 아래의 구현에서는 싸이클의 개수를 $N_c$라고 했을 때, $grp(u)<N_c$이면 싸이클의 번호를 나타내고 $grp(u)\geq N_c$이면 일반 정점의 번호를 나타냅니다.

새로운 간선 집합을 구하는 부분은 추가적인 메모리를 할당하지 않고 기존의 간선 정보를 수정하는 것만으로도 가능합니다. 모든 간선 $(u,v)\in E$ 를 순회하면서, $u$를 $grp(u)$, $v$를 $grp(v)$로 치환합니다. 즉, 간선 $(u,v)$를 제거하고 간선 $(grp(u),grp(v))$를 추가하는 것과 같습니다. 만약 $v$가 어떤 싸이클에 속한다면 (즉, $grp(v)<N_c$ 이면) 현재 간선의 가중치에서 $mn(v)$를 빼줍니다. 또한 $grp(u)=grp(v)$ 인 경우(즉, 싸이클 내부의 간선인 경우)에는 원래 간선 집합에서 제거해야 하지만, 나중에 간선을 순회할 때 $u=v$이면 continue하는 식으로 구현해주면 erase 연산을 사용하지 않아도 됩니다.

마지막으로, 루트 정점의 번호를 $r$에서 $grp(r)$로 치환합니다.

Step 1과 Step 2 모두 $O(\|V\|+\|E\|)$에 동작하고, 매 단계마다 정점의 개수가 최소 한 개 이상 줄어들기 때문에 총 시간복잡도는 $O(\|V\|(\|V\|+\|E\|))=O(\|V\|\|E\|)$가 됩니다.

아래는 Chu-Liu-Edmonds' algorithm을 C++로 구현한 것입니다.

```c++
namespace ChuLiuEdmonds {
	const ll INF=1e18;
	struct edge {
		int u,v; ll w;
		int prv;
		edge(int u,int v,ll w):u(u),v(v),w(w),prv(-1){}
	};
	ll solve(vector<edge>& _edges,int n,int r) {
		vector<edge> edges=_edges;
		ll ans=0;
		while(1) {
			vector<ll> mn(n,INF);
			vector<int> grp(n,-1),par(n,0);
			vector<bool> vis(n),fin(n);
			int nn=0;
			for(auto& it:edges) {
				if(it.u==it.v) continue;
				if(it.w<mn[it.v]) {
					mn[it.v]=it.w;
					par[it.v]=it.u;
				}
			}
			for(int i=0;i<n;i++) {
				if(i==r) continue;
				if(mn[i]==INF) return INF;
			}
			fin[r]=vis[r]=true;
			bool cycle=false;
			for(int i=0;i<n;i++) {
				if(fin[i]) continue;
				int j=i;
				while(!vis[j]) {
					vis[j]=true;
					j=par[j];
				}
				if(!fin[j]) {
					do {
						fin[j]=true;
						grp[j]=nn;
						ans+=mn[j];
						j=par[j];
					} while(!fin[j]);
					cycle=true;
					nn++;
				}
				j=i;
				while(!fin[j]) {
					fin[j]=true;
					j=par[j];
				}
			}
			if(!cycle) {
				for(int i=0;i<n;i++)
					if(i!=r) ans+=mn[i];
				break;
			}
			int cn=nn;
			for(int i=0;i<n;i++)
				if(grp[i]==-1) grp[i]=nn++;
			for(auto& it:edges) {
				if(grp[it.u]!=grp[it.v]&&grp[it.v]<cn)
					it.w-=mn[it.v];
				it.u=grp[it.u];
				it.v=grp[it.v];
			}
			r=grp[r];
			n=nn;
		}
		return ans;
	}
}
```

## 마치며

minimum arborescence는 대회에 자주 나오는 편은 아니지만, 원리가 어렵지 않기 때문에 한 번쯤 가볍게 이해해보는 것도 괜찮을 것이라 생각합니다. 기출문제로는 서울대학교 교내대회에 출제된 "미생물 키우기"(boj.kr/16127)가 있습니다. minimum arborescence에 대한 아이디어가 있다면 보다 쉽게 문제에 접근할 수 있을 것입니다.

이번 글에서 소개드린 알고리즘은 $O(\|E\|\|V\|)$ 이지만, Tarjan이 제안한 $O(\|E\|log\|V\|)$알고리즘이나 피보나치 힙으로 개선한 $O(\|E\|+\|V\|log\|V\|)$ 알고리즘도 존재합니다. 기회가 된다면 다음에 이러한 알고리즘들에 대한 리뷰도 작성해보도록 하겠습니다.