---
layout: post
title: "Karp의 21대 NP-완전 문제"
date: 2020-08-18
author: jh05013
---

[Richard Karp](https://en.wikipedia.org/wiki/Richard_M._Karp)는 알고리즘의 선두를 이끈 인물 중 한 명입니다. Problem solving을 하는 사람들에게는 Edmonds-Karp 최대 유량, Hopcroft-Karp 이분 매칭, Rabin-Karp 부분문자열 탐색 등으로 잘 알려져 있는데, 이 분의 업적으로 NP-완전성을 빼놓을 수 없습니다.

# 배경 지식

P, NP, 다항 시간 환원, 그리고 NP-완전 문제에 대해서는 koosaga님의 글 [계산 복잡도 위계와 불리언 식](https://koosaga.com/233)의 "QBF 문제" 직전까지를 참조해 주시기 바랍니다.

# NP-완전 문제

1971년, Stephen Cook은 [The complexity of theorem proving procedures](https://dl.acm.org/citation.cfm?coll=GUIDE&dl=GUIDE&id=805047) 논문에서 "비결정적 튜링 기계로 다항 시간에 결정할 수 있는 문제는 SAT 문제로 다항 시간에 환원할 수 있다"는 것을 증명합니다. 그 다음 해인 1972년에 나온 [Reducibility Among Combinatorial Problems](https://people.eecs.berkeley.edu/~luca/cs172/karp.pdf) 논문에서 Karp는 NP-완전성을 정의하고, 21개의 예시 문제가 모두 NP-완전임을 증명함으로써 이 문제들이 계산 복잡도 상으로 서로 엮여 있음을 보입니다. 이것이 학부 알고리즘 수업에서 흔히 배우는 NP-완전성의 시작이 되었습니다.

SAT은 프로그램 논리와 밀접한 연관이 있으니 NP-완전이라는 것이 납득이 가지만, 그래프 같은 구조에서 정의되는 문제는 어떻게 NP-완전성을 증명했을까요? 이미 NP-완전임이 알려져 있는 문제를 하나만 골라서 우리가 원하는 문제로 환원하면 됩니다. 그러면 임의의 NP 문제가 주어졌을 때, 우리가 고른 NP-완전 문제로 다항 시간에 환원할 수 있다는 사실은 이미 알고 있고, 그렇게 환원한 문제를 또 다항 시간에 우리가 원하는 문제로 환원할 수 있습니다. 이 모든 과정에 다항 시간이 걸리므로, 임의의 NP 문제를 우리가 원하는 문제로 다항 시간에 환원할 수 있다는 결론을 얻습니다.

다음은 Karp의 21대 NP-완전 문제입니다. 현대의 용어를 사용하기 위해 논문과 다른 이름을 사용하였으며, 이해가 쉽도록 설명을 살짝 바꾸었습니다.

1. **Satisfiability** - 주어진 CNF 식 $$\phi$$의 해가 존재하는가?
2. **0-1 Integer Programming** - 정수 행렬 $$C$$와 정수 벡터 $$d$$가 주어졌을 때, 모든 $$i$$에 대해 $$(Cx)_i \geq d_i$$가 성립하는 0-1 벡터 $$x$$가 존재하는가?[^1]
3. **Clique** - 그래프 $$G$$와 양의 정수 $$k$$가 주어졌을 때, $$G$$의 부분그래프 중 크기가 $$k$$ 이상인 클릭이 존재하는가? 여기서 *클릭*이란, 모든 정점 쌍에 대해 그 둘을 잇는 간선이 존재하는 그래프를 말합니다.
4. **Set Packing** - 집합 $$X$$의 부분집합들로 이루어진 집합 $$S$$, 그리고 양의 정수 $$\ell$$이 주어졌을 때, $$S$$ 안에서 $$\ell$$개 이상의 원소를 골라 어느 두 원소의 교집합도 공집합이도록 할 수 있는가?
5. **Vertex Cover** - 그래프 $$G$$와 양의 정수 $$\ell$$이 주어졌을 때, $$\ell$$개 이하의 정점을 골라 모든 간선이 그 중 한 정점과 연결되도록 할 수 있는가?
6. **Set Cover** - 집합 $$X$$의 부분집합들로 이루어진 집합 $$S$$, 그리고 양의 정수 $$k$$가 주어졌을 때, $$S$$ 안에서 $$k$$개 이하의 원소를 골라 합집합이 $$X$$이도록 할 수 있는가?
7. **Feedback Vertex Set** - 방향 그래프 $$H$$와 양의 정수 $$k$$가 주어졌을 때, $$k$$개 이하의 정점을 골라 모든 사이클이 그 정점들 중 하나를 지나게 할 수 있는가?
8. **Feedback Edge Set** - 방향 그래프 $$H$$와 양의 정수 $$k$$가 주어졌을 때, $$k$$개 이하의 간선을 골라 모든 사이클이 그 간선들 중 하나를 지나게 할 수 있는가?
9. **Directed Hamiltonian Circuit** - 방향 그래프 $$H$$가 주어졌을 때, 모든 정점을 지나는 사이클이 존재하는가?
10. **Undirected Hamiltonian Circuit** - 그래프 $$G$$가 주어졌을 때, 모든 정점을 지나는 사이클이 존재하는가?
11. **3-SAT** - 주어진 3-CNF 식 $$\phi$$의 해가 존재하는가? 여기서 *3-CNF*란, 각각의 절이 정확히 세 개의 변수를 담고 있는 CNF를 말합니다.
12. **Chromatic Number** - 그래프 $$G$$와 양의 정수 $$k$$가 주어졌을 때, 각 정점에 $$1$$ 이상 $$k$$ 이하의 양의 정수 중 하나를 배정하여 어느 이웃한 두 정점에도 다른 양의 정수가 배정되도록 할 수 있는가?
13. **Clique Cover** - 그래프 $$G$$와 양의 정수 $$\ell$$이 주어졌을 때, 정점 집합을 $$\ell$$개 이하의 부분집합으로 분할하여 각 부분집합이 클릭을 이루도록 할 수 있는가?
14. **Exact Cover** - 집합 $$X$$의 부분집합들로 이루어진 집합 $$S$$가 주어졌을 때, $$S$$의 원소 몇 개를 골라 합집합이 $$X$$와 같으면서 어느 두 원소의 교집합도 공집합이도록 할 수 있는가?
15. **Hitting Set** - 집합 $$X$$의 부분집합들로 이루어진 집합 $$S$$가 주어졌을 때, $$X$$의 부분집합 $$W$$를 잡아 $$S$$의 각 원소와의 교집합의 크기가 정확히 1이도록 할 수 있는가?
16. **Steiner Tree** - 간선에 가중치가 있는 그래프 $$G$$와 정점 부분집합 $$R$$, 그리고 양의 정수 $$k$$가 주어졌을 때, $$R$$의 모든 정점을 포함하는 가중치 합 $$k$$ 이하의 서브트리가 존재하는가?
17. **3-Dimensional Matching** - 크기가 같은 집합 $$A, B, C$$, 그리고 각 집합의 원소로 이루어진 순서쌍 $$(a \in A, b \in B, c \in C)$$의 집합 $$U$$가 주어졌을 때, $$U$$의 부분집합을 잡아 $$A, B, C$$의 모든 원소가 정확히 한 번씩 포함되도록 할 수 있는가?
18. **Subset Sum**[^2] - 정수 $$a_1, \cdots, a_r, b$$가 주어졌을 때, $$\sum_{i=1}^{r} a_ix_i = b$$가 성립하도록 $$x_1, \cdots, x_r \in \{0,1\}$$을 잡을 수 있는가?
19. **Job Shob Scheduling** - $$p$$개의 작업이 있다. $$i$$번째 작업은 시간 $$D_i$$ 이내에 완료되어야 하며, 수행하는 데 $$T_i$$의 시간이 걸리고, 완수하지 않으면 $$P_i$$의 페널티가 주어진다. 한 번에 한 작업만 수행할 수 있고, 수행 중간에 다른 작업으로 교체할 수 없다. 음이 아닌 정수 $$k$$가 주어진다. 이때 총 페널티가 $$k$$ 이하이도록 작업을 수행할 수 있는가?
20. **Partition** - 정수 리스트 $$S$$가 주어졌을 때, 이를 $$X$$와 $$S \backslash X$$로 분할하여 $$X$$ 안의 원소의 합이 $$S \backslash X$$ 안의 원소의 합과 같도록 할 수 있는가?
21. **Maximum Cut** - 간선에 가중치가 있는 그래프 $$G$$, 그리고 양의 정수 $$W$$가 주어졌을 때, 정점 집합 $$S$$를 잡아 $$S$$ 안의 정점과 $$S$$ 밖의 정점을 잇는 간선의 가중치의 합이 $$W$$ 이상이도록 할 수 있는가?

각 문제가 NP라는 것은 쉽게 알 수 있으므로 생략하고, SAT에서 시작해서 각 문제로 다항 시간 환원을 하면 모든 문제가 NP-난해임을 증명할 수 있습니다. 각각의 환원이 올바른 다항 시간 환원이라는 것은 대부분 어렵지 않게 볼 수 있고, 일부는 개략적으로 증명하겠습니다.

# 예? Subset Sum이요?

그 전에 한 가지를 짚고 넘어갑시다. 여기서 "Subset sum은 다이나믹 프로그래밍으로 다항 시간에 풀 수 있지 않나?"라는 의문이 들 수 있습니다. 정수들의 최댓값을 $$w$$라고 하면 $$O(nw)$$에 풀 수 있습니다.

하지만 이건 사실 다항 시간이 아닙니다. 왜냐하면 $$w$$는 **입력의 크기에 대한** 다항식이 아니기 때문입니다. $$w$$를 나타내는 데에는 $$O(\log w)$$ 비트만 필요하므로, $$O(nw)$$는 사실 비트의 개수에 대해 지수 시간입니다.

그렇다면 정수의 표현법을 바꿔서, $$w$$를 나타내는 데에 $$O(w)$$ 비트가 필요하게 하면 어떻게 될까요? 그러면 $$O(nw)$$는 다항 시간이 되겠지만, 아래에 제시할 "Exact Cover에서 Subset Sum으로"의 환원이 더 이상 다항 시간이 아니게 되어서, Subset Sum이 NP-완전이라는 것을 증명할 수 없게 됩니다.

# 환원

본론으로 돌아가서, SAT가 NP-완전임을 알고 있다고 가정하고 나머지 모두가 NP-완전이라는 것을 증명해 봅시다.

## SAT에서 0-1 Integer Programming으로

$$\phi$$의 어떤 절이 특정 변수 $$x_j$$와 $$\neg x_j$$를 모두 포함하면, 그 절은 무조건 참이므로 제거합니다. 그 결과 모든 절이 제거되면 $$\phi$$는 무조건 참이므로, 답이 참인 integer programming 문제를 아무렇게나 만듭니다.

절이 남아 있으면, $$C$$와 $$b$$를 다음과 같이 만듭니다.

* $$C$$의 크기는 $$n \times m$$입니다.
* $$C_{i,j}$$는 $$x_j$$가 절 $$i$$에 있으면 $$1$$, $$\neg x_j$$가 절 $$i$$에 있으면 $$-1$$, 아니면 $$0$$입니다.
* $$b_i$$는, 절 $$i$$에 있는 $$\neg x_j$$의 개수를 $$a$$라고 할 때, $$1-a$$입니다.

$$(Cx)_i$$를 최소화하려면 모든 $$x_j$$를 0과 곱하고, 모든 $$\neg x_j$$를 1과 곱해야 합니다. 이는 절 $$i$$가 만족하지 않음과 동치입니다. 또한 이때 $$(Cx)_i = -a$$이므로, $$(Cx)_i \geq
 1-a$$는 절 $$i$$가 만족함과 동치입니다.

## SAT에서 Clique로

* 각 절 $$i$$의 literal $$\sigma$$에 대해 정점 $$v_{i,\sigma}$$를 만듭니다.
* 두 정점 $$v_{i,\sigma}$$와 $$v_{j,\delta}$$에 대해, $$i \neq j$$이고 $$\sigma \neq \neg \delta$$이면 두 정점을 잇습니다. 그 외의 경우에는 잇지 않습니다. 즉 두 정점이 서로 다른 절에 속하면서 서로 모순을 일으키지 않아야 합니다.
* $$k = n$$.

이 그래프에서 크기 $$n$$의 클릭을 만들려면 각 절마다 변수를 하나씩 고르되 모순이 없어야 합니다.

## Clique에서 Set Packing으로

* 그래프의 각 정점에 번호 $$1, 2, \cdots, n$$을 붙입니다.
* $$X = \{\{i,j\} \mid i,j \in V(G), i \neq j\}$$.
* 각 정점 $$i$$에 대해, $$S_i = \{\{i,j\} \mid j \in V(G), i \neq j\} - E(G)$$로 둡니다. 즉 $$S_i$$는 $$i$$와 인접하지 **않은** 모든 $$j$$에 대해 $$\{i,j\}$$의 집합입니다.
* $$S = \{S_1, \cdots, S_n\}$$.
* $$\ell = k$$.

Clique 대신 Independent Set 문제를 생각하면 이해하기 편할 것입니다. Independent Set이란, 그래프 $$G$$가 주어졌을 때, 어느 두 정점도 인접하지 않도록 $$k$$개 이상의 정점을 잡을 수 있는지를 묻는 문제입니다. $$G$$의 클릭은 $$G$$의 complement의 independent set과 같기 때문에, 이 두 문제는 서로 쉽게 환원할 수 있습니다.

## Clique에서 Vertex Cover로

* $$G$$의 complement를 취합니다.
* $$\ell = \mid V(G) \mid - k$$.

이 문제도 Clique 대신 Independent Set 문제를 생각하면 이해하기 편할 것입니다.

## Vertex Cover에서 Set Cover로

* $$X = E(G)$$.
* 각 정점 $$i$$에 대해, $$S_i$$를 $$i$$와 연결된 간선의 집합으로 둡니다.
* $$S = \{S_1, \cdots, S_n\}$$.
* $$k = \ell$$.

## Vertex Cover에서 Feedback Vertex Set으로

* $$V(H) = V(G)$$.
* $$G$$의 각 간선 $$\{i, j\}$$에 대해, $$H$$에 간선 $$i \rightarrow j$$와 $$j \rightarrow i$$를 넣습니다.
* $$k = \ell$$.

## Vertex Cover에서 Feedback Edge Set으로

* $$G$$의 각 정점 $$v$$에 대해, $$H$$에 정점 $$v_{in}$$과 $$v_{out}$$을 넣습니다.
* $$G$$의 각 정점 $$v$$에 대해, $$H$$에 간선 $$v_{in} \rightarrow v_{out}$$을 넣습니다. 또한, $$G$$의 각 간선 $$\{i, j\}$$에 대해, $$H$$에 간선 $$i_{out} \rightarrow j_{in}$$과 $$i_{out} \rightarrow j_{in}$$을 넣습니다.
* $$k = \ell$$.

$$H$$의 사이클은 항상 $$x_{in} x_{out} y_{in} y_{out} z_{in} z_{out} \cdots x_{in}$$의 형태이고, feedback edge set을 만들려면 $$v_{in} \rightarrow v_{out}$$ 형태의 간선만 선택하는 것이 최적입니다.

## Vertex Cover에서 Directed Hamiltonian Circuit으로

$$\ell > \mid V(G) \mid$$이라면 Vertex Cover의 답이 항상 참이므로, 답이 참인 Directed Hamiltonian Circuit 문제를 아무거나 만듭니다.

$$G$$의 간선에도 번호를 붙입시다.

* $$V(H) = \{a_1, \cdots, a_\ell\} \cup \{(u,v)_{in}, (u,v)_{out} \mid \{u,v\} \in E(G) \}$$. 간선 $$\{u,v\}$$가 $$G$$에 있으면 정점 $$(u,v)_{in}, (u,v)_{out}, (v,u)_{in}, (v,u)_{out}$$을 모두 만들어야 합니다.
* 각 $$(u,v)_{in}$$에서 $$(u,v)_{out}$$으로 간선을 긋습니다.
* 각 $$(u,v)_{in}$$에서 $$(v,u)_{in}$$으로, 각 $$(u,v)_{out}$$에서 $$(v,u)_{out}$$으로 간선을 긋습니다.
* 각 $$u$$에 대해, $$G$$에서 $$u$$와 연결된 간선을 간선의 번호 순으로 정렬합니다. 이를 $$\{u,v_1\}, \cdots, \{u,v_k\}$$라고 할 때, 각 $$(u,v_i)_{out}$$에서 $$(u,v_{i+1})_{in}$$으로 간선을 긋고, $$(u,v_k)_{out}$$에서 $$a_j$$로, $$a_j$$에서 $$(u,v_1)_{in}$$으로 간선을 긋습니다.

환원 및 증명이 꽤 복잡한 편이니, 자세한 것은 [이 렉쳐 노트](https://cs.nyu.edu/courses/fall11/CSCI-UA.0453-001/chapter5.pdf)의 24-27페이지를 참조해 주시기 바랍니다.

## Directed Hamiltonian Circuit에서 Undirected Hamiltonian Circuit으로

* $$H$$의 각 정점 $$v$$에 대해, $$G$$에 정점 $$v_{in}, v_{mid}, v_{out}$$을 넣습니다.
* $$H$$의 각 정점 $$v$$에 대해, $$G$$에 간선 $$\{v_{in}, v_{mid}\}$$와 $$\{v_{mid}, v_{out}\}$$을 넣습니다.
* $$H$$의 각 간선 $$x \rightarrow y$$에 대해, $$G$$에 간선 $$\{x_{out}, y_{in}\}$$을 넣습니다.

## SAT에서 3-SAT으로

각각의 절을 다음과 같이 변환합니다.

* $$(x_1)$$을 $$(x_1 \vee x_1 \vee x_1)$$로
* $$(x_1 \vee x_2)$$를 $$(x_1 \vee x_2 \vee x_2)$$로
* $$(x_1 \vee x_2 \vee x_3)$$은 그대로
* $$(x_1 \vee x_2 \vee x_3 \vee x_4)$$를 $$(x_1 \vee x_2 \vee y_1) \wedge (\neg y_1 \vee x_3 \vee x_4)$$로
* ...
* $$(x_1 \vee \cdots \vee x_n)$$을 $$(x_1 \vee x_2 \vee y_1) \wedge (\neg y_1 \vee x_3 \vee y_2) \wedge \cdots \wedge (\neg y_{n-4} \vee x_{n-2} \vee y_{n-3}) \wedge (\neg y_{n-3} \vee x_{n-1} \vee x_n)$$으로

## 3-SAT에서 Chromatic Number로

* $$V(G) = \{x_1, \cdots, x_m, \neg x_1, \cdots, \neg x_m, v_1, \cdots, v_{m+1}, C_1, \cdots, C_n\}$$.
* $$\{v_1, \cdots, v_m\}$$은 클릭을 이룹니다.
* $$i \neq j$$인 $$m$$ 이하의 각 $$i$$와 $$j$$에 대해, $$v_i$$와 $$x_j$$, 그리고 $$v_i$$와 $$\neg x_j$$를 잇습니다.
* 각 $$x_i$$와 $$\neg x_i$$를 잇습니다.
* 각 $$C_i$$와 $$v_{m+1}$$을 잇습니다.
* $$i$$번째 절이 $$x_j$$를 포함하지 않는다면, $$C_i$$와 $$x_j$$를 잇습니다. $$i$$번째 절이 $$\neg x_j$$를 포함하지 않는다면, $$C_i$$와 $$\neg x_j$$를 잇습니다.
* $$k = m+1$$.

모든 $$v_i$$는 서로 다른 색으로 칠해져야 합니다. $$v_i$$의 색을 $$i$$번째 색이라고 하고, $$m+1$$번째 색을 "엑스트라"라고 부릅시다. 또한 $$x_i$$와 $$\neg x_i$$ 중 하나는 $$i$$번째 색, 다른 하나는 엑스트라로 칠해져야 합니다. 각 절은 엑스트라 색으로 칠할 수 없고, 그 절에 포함되지 않은 변수의 색으로도 칠할 수 없습니다. 따라서 그 절에 포함된 변수 중 하나가 엑스트라 색이 아닐 때에만 그 절을 색칠할 수 있습니다.

## Chromatic Number에서 Clique Cover로

* $$G$$의 complement를 취합니다.
* $$\ell = k$$.

같은 색으로 칠한 정점들은 $$G$$의 complement에서 클릭을 이룹니다.

## Chromatic Number에서 Exact Cover로

* $$X = V(G) \cup E(G) \cup \{(u,v,i) \mid \{u,v\} \in E(G) \wedge i \in \{1, \cdots, k\}\}$$.
* 각 $$u$$와 $$i$$에 대해, $$\{u\} \cup \{(u,v,i) \mid \{u,v\} \in E(G) \}$$을 $$S$$에 넣습니다.
* 각 간선 $$e = \{u,v\}$$와 $$i_1 \neq i_2$$인 각 $$i_1, i_2$$에 대해, $$\{e\} \cup \{(u,v,i) \mid i \neq i_1\} \cup \{(v,u,i) \mid i \neq i_2\}$$을 $$S$$에 넣습니다.

## Exact Cover에서 Hitting Set으로

Exact Cover 문제에서 $$X = \{x_1, \cdots, x_n\}, S = \{S_1, \cdots, S_m\}$$이라고 합시다.

* $$X' = \{S_1, \cdots, S_m\}$$.
* $$S' = \{\{S_j \mid x_i \in S_j\} \mid x_i \in X\}$$.

사실 직관적으로 Exact Cover와 Hitting Set은 매우 비슷한 문제입니다. Exact Cover는 집합을 먼저 고르고 각 원소가 고른 집합에 정확히 한 번씩 포함되게 하는 것이라면, Hitting Set은 원소를 먼저 고르고 각 집합이 고른 원소를 정확히 하나 포함하게 하는 것입니다. 이 사실에 따라 원소와 집합 관계를 그대로 뒤집은 것이 바로 위에서 제시한 환원입니다.

## Exact Cover에서 Steiner Tree로

$$S = \{S_1, \cdots, S_m\}$$이라고 합시다.

* $$V(G) = \{v\} \cup X \cup S$$.
* $$R = \{v\} \cup X$$.
* $$v$$에서 각 $$S_i$$로 가중치 $$\mid S_i \mid$$의 간선을 긋습니다.
* 각 $$S_i$$에서 $$x_j \in S_i$$인 각 $$x_j$$로 가중치 0의 간선을 긋습니다.
* $$k = \mid X \mid$$.

이 그래프에서 모든 가중치가 0 이상이므로, 가중치 합이 $$k$$ 이하인 서브트리가 존재한다는 것은 가중치 합이 $$k$$ 이하인 연결된 서브그래프가 존재함과 동치입니다. 따라서 각 $$x_j$$가 적어도 한 번씩은 포함되도록 $$S_i$$를 고르는 문제가 되는데, 가중치의 합이 $$\mid X \mid$$ 이하이려면 각 $$x_j$$는 꼭 한 번씩만 포함되어야 합니다.

## Exact Cover에서 3-Dimensional Matching으로

$$X = \{x_1, \cdots, x_n\}, S = \{S_1, \cdots, S_m\}$$이라고 합시다.

각 $$\mid S_i \mid \geq 2$$라고 가정합시다. 만약 $$\mid S_i \mid = 1$$이라면, $$X$$에 두 개의 원소 $$y, y'$$을 새로 추가하고, $$S_i$$에 $$y, y'$$을 넣고, 새로운 집합 $$\{y, y'\}$$을 $$S$$에 넣으면 됩니다.

* $$B = C = \{t_{ij} \mid x_i \in S_j\}$$. 위의 가정에 의해 $$\mid B \mid \geq 2n$$입니다.
* $$A = \{x_1, \cdots, x_n\} \cup \{\epsilon_1, \cdots, \epsilon_{\mid\ B \mid\ - n}\}$$.
* 순열 $$\pi : T \rightarrow T$$를 잡되, 각 $$j$$에 대해 $$\{t_{i,j} \mid x_i \in S_j\}$$가 $$\pi$$에서 하나의 사이클을 이루도록 합니다.
* 각 $$t_{ij}$$에 대해 $$(x_i, t_{ij}, t_{ij})$$를 $$U$$에 넣습니다.
* 각 $$t_{ij}$$와 $$a \in A$$에 대해, $$a \neq x_i$$라면, $$(a, t_{ij}, \pi (t_{ij}))$$를 $$U$$에 넣습니다.

각 $$j$$에 대해, $$B$$와 $$C$$에 있는 $$\{t_{i,j} \mid x_i \in S_j\}$$를 덮으려면, 모든 $$(x_i, t_{ij}, t_{ij})$$를 사용하거나 모든 $$(a, t_{ij}, \pi (t_{ij}))$$를 사용해야 합니다.

## Exact Cover에서 Subset Sum으로

$$X = \{x_1, \cdots, x_n\}, S = \{S_1, \cdots, S_m\}$$이라고 합시다.

* $$d = m + 1$$이라고 합시다.
* $$a_i = \sum_{x_j \in S_i} (m+1)^{j-1}$$.
* $$b = \sum_{j = 1}^{n} (m+1)^{j-1}$$.

각 $$S_i$$를 $$(m+1)$$진법 수로 인코딩했다고 생각하면 됩니다. 진법이 충분히 크기 때문에, 집합을 어떻게 선택하더라도 받아올림이 일어나지 않습니다.

## Subset Sum에서 Job Shop Scheduling으로

$$(\sum_{i=1}^{r} a_i) < b$$이라면 subset sum의 답은 항상 거짓이므로, 답이 거짓인 Job shop scheduling 문제를 아무렇게나 만듭니다.

$$(\sum_{i=1}^{r} a_i) \geq b$$이라면,

* $$p = r$$.
* $$D_i = b$$.
* $$T_i = a_i$$.
* $$P_i = a_i$$.
* $$k = (\sum_{i=1}^{r} a_i) - b$$.

## Subset Sum에서 Partition으로

* $$A = \sum_{i=1}^{r} a_i$$라고 둡시다.
* $$S = (a_1, \cdots, a_r, b+1, A+1-b)$$.

모든 원소의 합은 $$2A+2$$이므로, $$X$$의 원소의 합은 $$A+1$$이어야 합니다. 따라서 $$S$$의 마지막 두 원소는 같은 분할에 속할 수 없습니다. 일반성을 잃지 않고 $$A+1-b$$이 들어있는 분할을 $$X$$라고 하면, $$a_1, \cdots, a_r$$ 중 일부를 적당히 넣어서 합을 $$b$$로 만들어야 합니다.

## Partition에서 Maximum Cut으로

$$S = (s_1, \cdots, s_n)$$이라고 합시다.

* $$G$$는 크기가 $$n$$인 완전그래프입니다.
* $$\{i, j\}$$의 가중치는 $$4s_i s_j$$입니다.
* $$W = (\sum_{i=1}^{n} s_i)^2$$.

$$X$$의 원소의 합을 $$a$$라고 하면 $$S \backslash X$$의 원소의 합은 $$(\sum_{i=1}^{n} s_i) - a$$입니다. 이렇게 나눴을 때 가중치의 합은 분배법칙에 의해 $$4a ((\sum_{i=1}^{n} s_i)-a)$$이고, 이는 $$a = \frac{(\sum_{i=1}^{n} s_i)}{2}$$일 때 최댓값 $$W$$를 갖습니다.

# 마치며

지금까지 Karp가 제시한 21대 NP-완전 문제 및 그 증명을 살펴보았습니다. Karp가 제시한 방향 외에도 여러 흥미로운 증명을 몇 개 소개합니다.

* [3-SAT에서 Chromatic Number로](http://cs.bme.hu/thalg/3sat-to-3col.pdf): 여기서는 $$k$$가 3으로 고정되더라도 NP-완전임을 증명합니다.
* [3-SAT에서 Directed Hamiltonian Circuit으로](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-045j-automata-computability-and-complexity-spring-2011/lecture-notes/MIT6_045JS11_lec16.pdf)
* [3-SAT에서 3-Dimensional Matching으로](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/3dm.pdf)

NP-완전성에 대해 더 자세히 알고 싶으시다면 Michael Garey와 David Johnson의 책 [Computers and Intractability](https://en.wikipedia.org/wiki/Computers_and_Intractability)를 추천드립니다.

# 주석

[^1]: 논문에는 $$Cx = d$$라고 되어있는데, 오타로 보입니다. (이외에도 논문에 오타가 몇 군데 있더군요...)
[^2]: 논문에서는 Knapsack이라는 이름을 사용하였습니다.
