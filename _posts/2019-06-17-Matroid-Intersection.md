---
layout: post
title:  "Matroid Intersection"
date:   2019-05-15 23:30:00
author: ainta
tags: [algorithm, matroid, greedy algorithm, structure, graph theory]



---

# Matroid Intersection



### Recall



**matroid $$\mathcal{M} = (S,  \mathcal{I})$$ 에서 $$S$$는 유한집합, $$ \mathcal{I} \subset 2^S$$ 는 독립집합(independent set)들의 collection이다. 이 때, $$I$$는 다음 세 가지 조건을 만족하여야 한다.**

1. $$\phi \in  \mathcal{I}$$
2. $$Y \subset X, X \in  \mathcal{I} \Rightarrow Y \in  \mathcal{I}$$ 
3. $$X, Y \in  \mathcal{I}, \lvert X \rvert < \lvert Y \rvert$$ 이면 $$X + y \in  \mathcal{I}$$ 를 만족하는 $$y \in Y \setminus X$$가 존재

매트로이드는 다양한 집합에서 정의될 수 있다. 그 중 대표적인 예 몇 가지로는 **Vector matroid, Graphic matroid, Uniform matroid, Transversal matroid** 등이 있다.

$$\mathcal{M} = (S, \mathcal{I})$$의 independent set $$I$$에 대해 $$I$$를 진부분집합으로 갖는 independent set이 없다면 $$I$$를 $$\mathcal{M}$$의 **base**라고 한다. 모든 base의 크기는 같다.

**$$S$$의 각 부분집합에 대해서 base를 정의한다**. $$T \subset S$$에 대해, $$I \subset T$$가 independent이고 $$I \subset I_2 \subset T$$를 만족하는 independent set $$I_2$$가 존재하지 않는다면 $$I$$를 $$T$$의 base라고 한다. $$T$$의 모든 base의 크기는 같다. (base라는 용어를 쓸 때 특정한 집합에 대한 base라는 언급이 있을 때만 집합의 base이고, 그 외의 경우는 matroid의 base이다.)

matroid $$\mathcal{M} = (S, \mathcal{I})$$에서 **rank function $$r_\mathcal{M} : 2^S \rightarrow Z_+$$** 은 $$S$$의 부분집합 에서 정의되며, $$r_\mathcal{M}(X)$$는 $$X$$의 부분집합 중 maximum independent set의 크기이다.

matroid에서 **maximum weight independent set**은 weight가 큰 원소부터 **Greedy**하게 삽입하는 알고리즘으로 구할 수 있다.



### Matroid Intersection problems



같은 집합 $$S$$에서 정의된 두 매트로이드 $$\mathcal{M}_1 = (S, \mathcal{I}_1)$$, $$\mathcal{M}_2 = (S, \mathcal{I}_2)$$가 있을 때, 다음과 같은 문제들을 matroid intersection problem이라 부른다.

**문제 1. $$\mathcal{M}_1$$과 $$\mathcal{M}_2$$의 공통된 base가 존재하는가?**

**문제 2. $$I \in \mathcal{I_1} \cap \mathcal{I_2}$$를 만족하는 $$I$$ 중 가장 크기가 큰 것을 구하시오. (maximum cardinality common independent set)**

**문제 3. $$S$$의 각 원소에 가중치가 정의되어 있을 때,  $$I \in \mathcal{I_1} \cap \mathcal{I_2}$$를 만족하는 $$I$$ 중 원소들의 weight 합이 최대인 것을 구하시오. (maximum weight common independent set)**

문제 1은 문제 2의 답을 구하면 바로 해결할 수 있고, 문제 2는 문제 3의 가중치 1인 버전이므로 아래쪽 문제를 해결하면 위쪽 문제는 자동으로 해결됨을 알 수 있다. 이 글에서는 문제 2의 해결방법에 대해 증명과 함께 자세히 다룰 것이고, 문제 3은 문제 2와 비슷한 방법으로 해결되므로 별도의 증명 없이 알고리즘만 제시할 것이다. 



### Examples



**예시 1. 최대 이분 매칭(maximum bipartite matching)**

$$G = (V, E)$$가 bipartition $$V_1$$과 $$V_2 $$를 가지는 이분그래프(bipartite graph)일 때, 

$$\mathcal{I_1} = \left\{ I : I \subset E, V_1의 \: 각 \: 정점은 \: I에 \: 포함되는 \: 간선 \: 최대 \: 하나의 \: 끝점  \right\}$$으로 두면  $$\mathcal{M_1} = (V_1, \mathcal{I})$$ 은 matroid이다.

$$\mathcal{I_2} = \left\{ I : I \subset E, V_2의 \: 각 \: 정점은 \: I에 \: 포함되는 \: 간선 \: 최대 \: 하나의 \: 끝점  \right\}$$으로 두면  $$\mathcal{M_2} = (V_2, \mathcal{I})$$ 은 matroid이다.

$$I \in \mathcal{I_1} \cap \mathcal{I_2}$$가 의미하는 바는 $$I$$가 matching이라는 것이므로, 두 매트로이드의 maximum cardinality common independent set을 구하면(문제 2) 최대 이분 매칭을 구할 수 있다. 



**예시 2. 최대 가중치 이분 매칭(maximum weight bipartite matching)**

예시 2와 똑같이 매트로이드를 구성한 후, 두 매트로이드의 maximum weight common independent set을 구하면(문제 3) 최대 가중치 이분 매칭을 구할 수 있다.



**예시 3. vector matroid와 graphic matroid의 intersection**

다음과 같은 문제를 생각하자: $$G=(V,E)$$의 각 edge에 weight가 정의되어 있다. $$E$$의 부분집합 $$I$$가 cycle을 포함하지 않고, 또한 $$I$$의 edge의 가중치들의 집합 $$w(I)$$는 임의의 공집합이 아닌 부분집합을 골랐을 때 원소들을 xor한 값이 0이 되지 않는다고 한다. 이러한 조건을 만족하는 $$I$$ 중 크기가 최대인 것을 구하시오.

이 문제는 체 $$GF(2)$$ 에서 정의된 vector matroid와 graphic matroid의 maximum cardinality common independent set을 구하는 문제이므로, 문제 2로 환원된다.



**예시 4. Colorful spanning tree**

다음과 같은 문제를 생각하자: $$G=(V,E)$$의 각 edge는 1부터 K 범위의 색깔을 가진다. edge 몇 개를 골라 spanning tree를 이루도록 하되, 각 색깔 $$i$$에 대해 색깔이 $$i$$인 edge를 최대 $$H_i$$개까지만 사용할 수 있다고 하자. 조건을 만족하는 spanning tree가 존재하는지 판별하라.

이 문제는 partition matroid와 graphic matroid의 maximum cardinality common independent set을 구하는 문제이므로, 문제 2로 환원된다. 문제 3을 해결하는 알고리즘을 이용하면 weight가 추가된 그래프에서 minimum weight colorful spanning tree를 구하는 것도 가능하다.

matroid intersection으로 해결 가능한 간단한 4가지 문제를 살펴보았다. 앞의 2문제는 matroid intersection으로 해결하는 것이 확실하게 overkill이지만, 이미 알고 있는 문제도 matroid intersection으로 접근 가능하다는 것을 보여준다. 예시 3, 4는 매우 기본적인 예이며, 좀 더 생각이 필요한 문제들은 문제 2,3을 해결하는 알고리즘을 살펴보고 나서 알아보자.



### Finding a maximum cardinality common independent set(문제 2)



앞으로 살펴볼 정리와 증명들에서는 저번 게시물(Introduction to Matroid intersection)에서 살펴본 성질들을 사용하므로, 참고하면 좋다.



**정리 1(Strong Base Exchange Theorem).** $$\mathcal{M} = (S, \mathcal{I})$$의 서로 다른 두 base $$B, B'$$가 있다. 이 때, 임의의 $$x \in B \setminus B'$$에 대해 어떤 $$y \in B' \setminus B$$가 존재하여 $$B - x + y$$, $$B + y - x$$가 둘 모두 base이다.

**Proof:** $$B \setminus B'$$에서 아무 원소나 골라 $$x$$라고 하자. $$B'$$는 base이므로, $$B' + x$$는 unique circuit $$C$$를 포함한다. 그러면 $$(B \cup C) - x$$는 base를 포함한다. 한편, $$B-x$$는 independent set이다. 따라서, $$(B \cup C) - x$$에 포함되면서 $$B - x$$를 포함하는 base가 존재한다. 이를 $$B''$$라 하면 어떤 $$y \in C-x$$가 존재하여 $$B'' = B-x+y$$가 성립한다. (base의 크기는 모두 같으므로)

$$B'-y+x$$가 base가 아니라고 가정하자. 그러면 $$B' - y + x$$는 어떤 circuit $$C'$$를 가진다. $$y \in C \setminus C'$$이므로 $$B' + x$$는 두 서로 다른 circuit $$C$$, $$C'$$을 가진다. 그런데 base에 원소 하나를 추가하면 unique circuit만을 가지므로 이는 모순이다. 따라서, $$B-x+y, B+y-x$$는 둘 모두 $$\mathcal{M}$$의 base이다.

**따름정리 2.** $$\mathcal{M} = (S, \mathcal{I})$$의 두 independent set $$I_1, I_2$$이 $$\lvert I_1 \rvert = \lvert I_2 \rvert$$를 만족할 때, 임의의 $$x \in I_1 \setminus I_2$$에 대해 $$y \in I_2 \setminus I_1$$이 존재하여 $$I_1 - x + y$$, $$I_2 - y +x$$가 둘 모두 independent set이다.

**Proof:**  $$\lvert I_1 \rvert = \lvert I_2 \rvert = K$$라고 하자. $$\mathcal{I'}$$를 $$\mathcal{I}$$의 원소 중 크기가 $$K$$이하인 것들의 집합이라고 했을 때 $$\mathcal{M'} = (S, \mathcal{I'})$$가 matroid인 것은 자명하다. 또한 $$\mathcal{M'}$$의 모든 independent set의 크기가 $$K$$ 이하이므로 $$I_1$$, $$I_2$$는 $$\mathcal{M'}$$의 base이다. 따라서, Strong Base Exchange Theorem에 의해 따름정리 2가 성립한다.



**정의 1.**  matroid $$\mathcal{M} = (S, \mathcal{I})$$에서 independent set $$I$$에 대해 directed bipartite graph $$D_M(I)$$를 다음과 같이 정의하자: $$D_M(I) = (S, E)$$, $$ E = \left\{ (y,z) \in E \: : \:  y \in I, z \in S \setminus I , I-y+z \in \mathcal{I}  \right\}$$. $$D_M(I)$$의 bipartition은 $$(I, S \setminus I)$$이다.

**보조정리 3.** $$\mathcal{M} = (S, \mathcal{I})$$의 두 independent set $$I_1, I_2$$이 $$\lvert I_1 \rvert = \lvert I_2 \rvert$$를 만족할 때, $$D_M(I)$$ 는 $$I_1 \Delta I_2$$에서 perfect matching을 갖는다. (단, $$A \Delta B = (A \cup B) \setminus (A \cap B)$$)

**Proof:**  $$\lvert I_1 \Delta I_2 \rvert$$에 대한 수학적 귀납법으로 증명할 것이다. $$\lvert I_1 \Delta I_2 \rvert = 0$$ 이면 자명하다. $$\lvert I_1 \Delta I_2 \rvert \ge 1$$인 경우, 따름정리 2에 의해 $$y \in I_1 \setminus I_2$$, $$z \in I_2 \setminus I_1$$가 존재하여 $$I_1' = I_1 - y + z$$가 independent set이다. 이 떄, $$\lvert I_1' \Delta I_2 \rvert < \lvert I_1 \Delta I_2 \rvert$$이고 $$\lvert I_1' \rvert = \lvert I_2 \rvert$$이므로, 수학적 귀납법에 의해 $$D_M(I)$$는 $$I_1' \Delta I_2$$에서 perfect matching $$N$$을 갖는다. 그러면 $$N \cup \left\{(y,z)\right\}$$는 $$I_1 \Delta I_2$$에서의 perfect matching이다.



**정리 4.** $$\mathcal{M} = (S, \mathcal{I})$$의 independent set $$I$$와 $$S$$의 부분집합 $$J$$가 $$\lvert I \rvert = \lvert J \rvert$$를 만족한다. 이 때, $$D_M(I)$$가 $$I \Delta J$$에서 unique perfect matching을 가지면 $$J$$는 independent set이다.

다음과 같은 성질을 이용하면 정리 4를 증명할 수 있다.

**성질 5.** $$G = (X, Y, E)$$는 bipartition $$(X,Y)$$를 가지는 bipartite graph이다. $$G$$가 unique perfect matching $$N$$을 가질 때, 다음 조건을 만족하도록 $$X$$의 원소들을 $$x_1, ..., x_t$$, $$Y$$의 원소들을 $$y_1, ..., y_t$$로 라벨링하는 것이 가능하다:

조건: $$N = \left\{(x_1, y_1), ..., (x_t, y_t) \right\}$$이고, 모든 $$i<j$$에 대해  $$(x_i ,y_i) \notin E$$ 를 만족한다.

성질 5의 경우 증명이 간단하지 않아 생략한다. 그러나 이분 그래프의 perfect matching에서 중요한 성질 중 하나이므로 알아두면 좋을 것이다. 그러면 이제 성질 5를 이용해 정리 4를 증명해보자.

**Proof:**  그래프 $$D_M(I)$$에서 $$I \Delta J$$에 포함되는 vertex들과 그 vertex 사이의 간선만 남긴 그래프를 $$G$$라 하자 ( $$G$$ : subgraph of $$D_M(I)$$ induced by $$I \Delta J$$ ). $$G$$는 bipartite graph 이므로 [성질 5]에 의해 $$I \setminus J$$의 vertex들을 $$y_1,.., y_t$$, $$J \setminus I$$의 vertex들을 $$z_1, .., z_t$$로 라벨링하여 $$N = \left\{ (y_1, z_1), .., (y_t, z_t) \right\}$$이고 $$ \forall  1 \le i < j \le t $$,  $$(y_i, z_j) \notin E(G)$$  가 성립하도록 할 수 있다.

$$D_M(I)$$가 $$I \Delta J$$에서 unique perfect matching을 가지는데 $$J$$가 independent set이 아니라고 하자.  $$C$$는 $$J$$의 circuit이다. $$i$$를 $$z_i \in C$$가 성립하는 가장 작은 수라고 하자. $$z_j \in C - z_i$$이면 $$j > i$$이므로 $$(y_i, z_j) \notin D_M(I)$$ 가 성립한다. $$C-z_i$$의 임의의 원소 $$z$$는 $$I \cap J$$의 원소이거나 어떤 $$j$$에 대해 $$z = z_j$$가 성립하므로 $$z \in span(I-y_i)$$가 성립한다 ($$span(X)$$는 $$X$$의 원소 및 $$X$$에 추가했을 때 dependent한 원소들의 집합이다). $$C$$는 circuit이므로, $$C \subset C-z_i \subset span(I-y_i)$$가 성립한다. 따라서 $$z_i \in span(I-y_i)$$인데, 이는 $$(y_i, z_i) \in N$$, 즉 $$I-y_i+z_i$$가 independent하다는 사실에 모순이다. 따라서, $$J$$는 independent set이다.



그러면 이제 두 매트로이드에서 가장 큰 common independent set을 구할 준비를 마쳤다.

**정리 5.** $$\mathcal{M_1} = (S, \mathcal{I_1})$$, $$\mathcal{M_2} = (S, \mathcal{I_2})$$ 가 각각 rank function $$r_1, r_2$$를 가진다고 하자. 이 때 두 매트로이드에서 가장 큰 common independent set의 크기는 (size of maximum cardinality set in $$\mathcal{I_1} \cap \mathcal{I_2}$$) 다음과 같다: $$min_{U \subset S}r_1(U) + r_2(S \setminus U)$$ 

**Proof:**  $$I \in \mathcal{I_1} \cap \mathcal{I_2}$$ , $$U \subset S$$ 에 대해 $$I \cap U \in \mathcal{I_1}$$, $$I \setminus U \in \mathcal{I_2}$$이므로  $$\lvert I \rvert = \lvert I \cap U \rvert + \lvert I \setminus U \rvert \le r_1(U) + r_2(S \setminus U)$$이므로 한쪽 부등식이 증명되었다. 반대방향의 부등식은 실제로 어떤 $$U$$에 대해 $$r_1(U) + r_2(S \setminus U)$$ 크기의 matroid intersection을 찾는 알고리즘을 제시하여 증명할 것이다. 이 알고리즘은 이분 매칭에서 augmenting path를 찾는 방법과 비슷하게 크기를 늘려가는 방법을 사용한다. 즉, $$I \in \mathcal{I_1} \cap \mathcal{I_2}$$가 주어지면 $$J \in \mathcal{I_1} \cap \mathcal{I_2}, \lvert J \rvert = \lvert I \rvert + 1$$을 만족하는 $$J$$를 찾거나 아니면 그러한 $$J$$가 없다는 것을 $$I$$의 크기가 어떤 $$U$$에 대해 $$r_1(U) + r_2(S \setminus U)$$와 같다는 것을 이용하여  보인다. 이 증명을 위해서는 몇 가지 준비가 더 필요하다:

**정의 6.** $$D_{M_1, M_2}(I) = (S, A(I))$$ where $$A(i) = \left\{ (y, z) : y \in I, z \in S \setminus I, I-y+z \in \mathcal{I_1} \right\} \cup \left\{ (z', y') : y' \in I, z' \in S \setminus I, I-y'+z' \in \mathcal{I_2} \right\}$$. 다르게 말하면, $$D_{M_1, M_2}(I)$$는 $$D_{M_1}(I)$$와 $$D_{M_2}(I)$$의 reverse의 union이다. 

$$X_1 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_1}\right\}$$, $$X_2 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_2}\right\}$$ 라 하자. $$D_{M_1,M_2}(I)$$에서 $$X_1$$에서 출발해 $$X_2$$에서 끝나는 경로 중 최소 길이인 것을 $$P$$라 하자. ($$P$$는 존재하지 않을 수도 있다)

**보조정리 7**. 만약 $$P$$가 존재하지 않는다면 ( $$D_{M_1,M_2}(I) $$에서 $$X_1$$에서 $$X_2$$로 가는 path가 없다면 ) $$I$$는 $$\mathcal{I_1} \cap \mathcal{I_2}$$의 maximum cardinality set이다.

**Proof:** $$X_1$$이나 $$X_2$$가 빈 집합이라면 $$I$$는 두 매트로이드 중 하나의 base이므로 maximum cardinality set임이 자명하다. $$X_1 \neq \phi, X_2 \neq \phi$$라 하자. $$U$$를 $$X_2$$에 도달할 수 있는 vertex들의 집합이라 하면 $$X_1$$에서 $$X_2$$로 가는 path가 없으므로 $$X_1 \cap U = \phi$$이다. 이제 $$r_1(U) \le \lvert I \cap U \rvert$$, $$r_2(S \setminus U) \le \lvert I \setminus U \rvert$$ 임을 보이면 앞서 모든 $$U$$에 대해  $$\lvert I \rvert = \lvert I \cap U \rvert + \lvert I \setminus U \rvert \le r_1(U) + r_2(S \setminus U)$$가 성립함을 보였으므로 $$I$$가 $$\mathcal{I_1} \cap \mathcal{I_2}$$의 maximum cardinality set임을 보일 수 있다. 

**claim 1.** $$r_1(U) \le \lvert I \cap U \rvert$$

**Proof:** $$r_1(U) > \lvert I \cap U \rvert$$ 이면 $$z \in U \setminus (I \cap U)$$인 $$z$$가 존재하여 $$(I \cap U) + z \in \mathcal{I_1}$$을 만족한다. 만약 $$I + z \in \mathcal{I_1}$$이면 $$z \in X_1$$, $$z \in U$$, $$X_1 \cap U \neq \phi$$ 이므로  모순이다. 그렇지 않다면 $$I+z \notin \mathcal{I_1}$$, $$(I \cap U) + z \in \mathcal{I_1}$$이므로 $$y \in I \setminus U$$가 존재하여 $$I-y+z \in \mathcal{I_1}$$이 성립한다. 그러나 이 경우 $$(y, z) \in A(I)$$이고, $$z \in U$$이므로 $$y$$는 $$z$$를 거쳐 $$X_2$$에 도달할 수 있어서 $$y \in U$$이여 하는데 이는 $$y \in I \setminus U$$에 모순이다. 따라서,  $$r_1(U) \le \lvert I \cap U \rvert$$.

**claim 2.** $$r_2(S \setminus U) \le \lvert I \setminus U \rvert$$ : claim 1과 유사하게 증명 가능하다.

claim 1, 2가 증명되었으므로,  $$D_{M_1,M_2}(I) $$에서 $$X_1$$에서 $$X_2$$로 가는 path가 없다면  $$I$$는 $$\mathcal{I_1} \cap \mathcal{I_2}$$의 maximum cardinality set이다.



**정리 8.** $$X_1$$에서 $$X_2$$로 가는 최소 길이의 path $$P$$ 에 대해, $$I' = I \Delta V(P) \in \mathcal{I_1} \cap \mathcal{I_2}$$이다. 또한, $$\lvert I' \rvert = \lvert I \rvert +1$$이다.

**Proof:** $$P$$는 bipartite graph  $$D_{M_1,M_2}(I) $$에서  $$S \setminus I$$ 로부터 시작해  $$S \setminus I$$로 끝나는 경로이므로 $$V(P)$$에서 $$I$$에 포함되는 원소보다 $$S \setminus I$$에 포함되는 원소가 항상 하나 더 많다. 따라서 $$\lvert I' \rvert = \lvert I \Delta V(P) \rvert = \lvert I \rvert+1$$이 성립한다. $$P = z_0, y_1, z_1, ..., z_{t-1}, y_t, z_t$$라 두자($$z_i \in S \setminus I, y_i \in I$$). 집합  $$J = \left\{z_1, ..., z_t \right\} \cup (I \setminus \left\{ y_1, .., y_t\right\})$$ 에 대해 $$J \subset S$$, $$\lvert J \rvert = \lvert I \rvert$$이고 $$\left\{ y_1, ..., y_t \right\} 에서 \left\{ z_1, ..., z_t \right\}$$로 가는 간선들은 $$I \setminus J$$에서 $$J \setminus I$$로 가는 unique perfect matching을 이룬다 (matching이 존재함은 자명하고, unique하지 않다면 [성질 5]에 의해 $$i < j$$인 간선 $$(y_i, z_j)$$가 존재하는데 이는 $$P$$가 shortest path임에 모순이다). 따라서, 정리 4에 의해 $$J \in \mathcal{I_1}$$.

모든 $$i \ge 1$$에 대해 $$z_i \notin X_1$$ 가 성립하므로 $$z_i +I \notin \mathcal{I_1}$$이고, 따라서 $$r_1(I \cup J) = r_1(I) = r_1(J) = \lvert I \rvert = \lvert J \rvert$$이다. $$I+z_0 \in \mathcal{I_1}$$이므로 $$x \in I+z_0 \setminus J$$가 존재하여 $$J + x \in \mathcal{I_1}$$인데, $$x \in I$$ 인 경우 $$x + J \in I_1$$이면 $$r_1(I \cup J) = \lvert J \rvert$$인 것에 모순이므로 $$x = z_0$$일 수밖에 없다. 즉, $$J + z_0 = I' \in \mathcal{I_1}$$이다. 마찬가지 방법으로 $$I' \in \mathcal{I_2}$$임도 쉽게 보일 수 있다 ($$z_0, .., z_{t-1}$$과 $$y_1, ..., y_t$$에 대해 같은 방법을 쓰면 된다). 따라서, $$I' \in \mathcal{I_1} \cap \mathcal{I_2}$$.



앞서 증명한 정리들로 얻을 수 있는 최종 결과는 다음과 같다.

**정리 9.** 다음 알고리즘은 두 matroid의 intersection 내 maximum cardinality set을 다항 시간 내에 올바르게 구한다.

Step 1. $$I = \phi$$로 초기화한다.

Step 2. $$D_{M_1,M_2}$$를 만들고, 집합 $$X_1 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_1}\right\}$$, $$X_2 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_2}\right\}$$ 를 구하자.

Step 3. $$X_1$$에서 $$X_2$$로 가는 shortest path $$P$$를 구한다.

Step 4. 만약 $$P$$가 존재하지 않으면 $$I$$가 maximum cardinality set이므로 $$I$$를 리턴하고 종료한다. 그렇지 않으면 $$I$$에 $$I \Delta V(P)$$를 대입한 뒤 Step 2로 돌아간다.





### Finding a maximum weight common independent set(문제 3)

정리 9를 다시 한번 살펴보자.

**정리 9.** 다음 알고리즘은 두 matroid의 intersection 내 maximum cardinality set을 다항 시간 내에 올바르게 구한다.

Step 1. $$I = \phi$$로 초기화한다.

Step 2. $$D_{M_1,M_2}$$를 만들고, 집합 $$X_1 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_1}\right\}$$, $$X_2 = \left\{ z \in S \setminus I : I + z \in \mathcal{I_2}\right\}$$ 를 구하자.

Step 3. $$X_1$$에서 $$X_2$$로 가는 shortest path $$P$$를 구한다.

Step 4. 만약 $$P$$가 존재하지 않으면 $$I$$가 maximum cardinality set이므로 $$I$$를 리턴하고 종료한다. 그렇지 않으면 $$I$$에 $$I \Delta V(P)$$를 대입한 뒤 Step 2로 돌아간다.



위 알고리즘에서 shortest path P를 구할 때, maximum cardinality set에서는 그냥 모든 간선의 길이를 1로 두고 shortest path를 구했지만, weighted case에서는 $$x \in I$$인 $$x$$에 대해 $$x$$의 weight를 $$w(x)$$, $$x \notin I$$인 $$x$$에 대해 weight를 $$-w(x)$$로 assign한 후 $$P$$를  $$X_1$$에서 $$X_2$$로 가는 path 중 minimum length path ($$V(P)$$ 내 정점의 weight 합이 최소가 되는 path)로 잡고, 그런 것이 여러 개 있다면 그 중에서 가장 적은 edge를 지나는 path를 $$P$$로 두어야 한다. 즉, 새로 만들어진 $$I' = I \Delta V(P)$$의 weight이 최대가 되는 path 중 가장 적은 개수의 edge를 지나는 path를 $$P$$로 두는 것이다. 그 부분만 수정을 해 주면 알고리즘이 리턴하는 최종 $$I$$는 $$\mathcal{I_1} \cap \mathcal{I_2}$$에서 가장 weight가 큰 set이 된다. 또한, matroid에서 maximum weighted independent set을 구할 때와 마찬가지로, 각 step 2~4를 반복하면서 얻는 $$I$$들은 크기가 $$I$$와 같은 intersection 중에서는 weight가 maximum인 set이다. (자세한 증명은 생략한다)



### 문제 풀이

##### SWERC 2011. Coin Collecting (https://www.acmicpc.net/problem/3836)

문제를 간단하게 설명하자면 다음과 같다: N쌍의 봉투가 있고 각각의 봉투에는 서로 다른 종류의 두 동전이 있다. 이때 각각의 쌍에서 최대 하나의 봉투만을 골라 총 N개 이하의 봉투를 고르려고 한다. 이 때 지켜야 할 조건이 있는데, 고른 봉투 중 1개 이상을 선택하여 선택한 봉투들에 들어 있는 동전만 놓고 보았을 때 모든 종류의 동전이 짝수개가 되도록 선택하는 방법이 존재하지 않도록 봉투를 고르려고 한다. 최대한 많은 개수의 동전을 모으는 것이 목표일 때, 최대 몇 개까지 가능한가?

이 문제는 결국 각각의 동전 종류를 vertex로, 각각의 봉투를 edge로 치환하면 edge쌍 N개가 주어질 때 각 쌍에서 최대 하나의 edge만을 선택하여 forest가 되도록 할 때, 선택한 edge의 개수를 최대화하는 문제가 된다. 이렇게 그래프 문제로 치환을 해 놓으면 이것은 partition matroid와 graphic matroid의 intersection에서 maximum independent set을 구하는 문제임을 쉽게 관찰할 수 있다.

다음은 이 문제의 AC 코드이다 (주의 : 이 문제는 시간 제한이 넉넉하지 않아 코드에 matroid intersection의 핵심이 아닌 부분이 다수 존재하므로 다음 문제의 코드를 읽는 것이 더 나을 수 있음)

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
#include<vector>
#include<cstring>
#define N_ 610
using namespace std;
map<int, int>Map;
int m, n, chk[N_ * 2], vis[N_], UF[N_ * 2], CK1[N_], CK2[N_], Q[N_], Path[N_];
bool vv[N_ * 2], inTP[N_ * 2];
bool E[N_][N_];
vector<int> G[N_ * 2], TP;
struct Edge {
	int a, b;
}w[N_];
int Num(int a) {
	if (!Map.count(a))Map[a] = ++n;
	return Map[a];
}
void PushTP(int a) {
	if (!inTP[a]) {
		TP.push_back(a);
		inTP[a] = 1;
	}
}
void Del_Edge(int a, int b) {
	PushTP(a);
	PushTP(b);
	G[a].erase(find(G[a].begin(), G[a].end(), b));
	G[b].erase(find(G[b].begin(), G[b].end(), a));
}
void Add_Edge(int a, int b) {
	PushTP(a);
	PushTP(b);
	G[a].push_back(b);
	G[b].push_back(a);
}
void Make(int x) {
	while (1) {
		Add_Edge(w[x].a, w[x].b);
		chk[x] = 1;
		if (Path[x] == -1)break;
		Del_Edge(w[Path[x]].a, w[Path[x]].b);
		chk[Path[x]] = 0;
		x = Path[Path[x]];
	}
}
void DFS(int a, int r) {
	vv[a] = 1;
	UF[a] = r;
	for (auto &x : G[a]) {
		if (!vv[x])DFS(x, r);
	}
}
void Build() { // 정리 9의 알고리즘에서 step 2. 그래프를 만드는 부분.
	int i;
	memset(vv, 0, (n + 1));
	memset(inTP, 0, (n + 1));
	for(auto &t : TP){
		if (!vv[t])DFS(t, t);
	}
	TP.clear();
}
bool Go() { // 정리 9의 알고리즘
	int i, j;
	memset(E, 0, sizeof(E));
	for (i = 0; i < m; i++) {
		Path[i] = -1;
		CK1[i] = CK2[i] = 0;
		vis[i] = 0;
	}
	Build();
	for (j = 0; j < m; j++) {
		if (chk[j])continue;
		if (UF[w[j].a] != UF[w[j].b])CK2[j] = 1;
	}
	for (j = 0; j < m; j++) {
		if (!chk[j]) {
			if (!chk[j^1])CK1[j] = 1;
			continue;
		}
		Del_Edge(w[j].a, w[j].b);
		Build();
		chk[j] = 0;
		for (i = 0; i < m; i++) {
			if (chk[i])continue;
			if (UF[w[i].a] != UF[w[i].b])E[i][j] = 1;
			if (!chk[i^1]) {
				E[j][i] = 1;
			}
		}
		chk[j] = 1;
		Add_Edge(w[j].a, w[j].b);
	}
	int head = 0, tail = 0;
	for (i = 0; i < m; i++) {
		if (CK1[i]) {
			Q[++tail] = i;
			vis[i] = 1;
			if (CK2[i]) {
				Make(i);
				return true;
			}
		}
	}
	while (head < tail) { // BFS를 통해 X1-X2 shortest path를 찾는다.
		int x = Q[++head];
		for (int y = 0; y < m; y++) {
			if (E[x][y] && !vis[y]) {
				Q[++tail] = y;
				vis[y] = 1;
				Path[y] = x;
				if (CK2[y]) {
					Make(y);
					return true;
				}
			}
		}
	}
	return false;
}
void Solve() {
	int i, a, b, res = 0;
	Map.clear();
	TP.clear();
	n = 0;
	m *= 2;
	for (i = 0; i < m; i++) {
		scanf("%d%d", &a, &b);
		a = Num(a), b = Num(b);
		w[i] = { a,b };
	}
	for (i = 1; i <= n; i++) {
		UF[i] = i;
		G[i].clear();
	}
	for (i = 0; i < m; i++) chk[i] = 0;


	for (int i = 0; i < m; ++i) {
		Build();
		if (!chk[i ^ 1] && UF[w[i].a]!=UF[w[i].b]) {
			Path[i] = -1;
			Make(i);
			res += 2;
		}
	}

	while (Go())res+=2;
	printf("%d\n", res);
}
int main() {
	srand(1879);
	while (1) {
		scanf("%d", &m);
		if (!m)break;
		Solve();
	}
}
```

##### NAIPC 2018 G. Rainbow Graph 

이 문제는 앞선 문제와 달리 matroid intersection 알고리즘을 완벽하게 이해했더라도 풀기 쉽지 않고, 생각할 거리가 있는 문제이므로 만약 알고리즘을 이해했다면 혼자 풀어보려고 시도해보는 편이 좋다. 

문제의 간단한 설명은 다음과 같다: 그래프 $$G = (V,E)$$의 각 edge에는 R, G, B의 3가지 색깔 및 가중치가 있다. 이 때, 정확히 $$K$$개의 edge를 골라 edge들 중 색깔이 R 또는 G 인것만 남겨도 $$G$$가 connected이고, edge들 중 색깔이 G 또는 B인 것만 남겨도 $$G$$가 connected가 되도록 $$K$$개의 edge를 고르려고 한다. 고르는 경우가 여러 방법이 있다면 그 중 고른 edge들의 weight 합이 최소가 되도록 고르고자 한다. 이 때, 모든 $$1$$부터 $$\lvert E \rvert$$까지의 모든 $$K$$에 대해 이 문제를 해결하시오 (불가능하면 -1, 가능하면 최소 weight 합을 출력한다).

이전 글 Introduction to matroid에서 graphic matroid를 소개할 때, edge들이 forest를 이루도록 하는 일반적인 graphic matroid만 설명한 것이 아니라 변형된 몇 가지 graphic matroic를 소개하였다. 그 중 하나가 바로 이 문제에 사용되는 매트로이드인데,  $$G = (V, E)$$가 무향 연결 그래프일 때, $$\mathcal{I} = \left\{I : I \subset E, E - I \: connects \: all \: vertex \: in \: G \right\}$$로 두면 $$\mathcal{M} = (E, \mathcal{I})$$는 matroid이다. 이 문제를 풀기 위해 이를 살짝 변형해보자. $$G = (V, E)$$가 무향 연결 그래프일 때, $$\mathcal{I_1} = \left\{I : I \subset E, E - I \: 중 \: 색깔이 \: R\: 또는 \: G \:인 \:것만 \: 고려해도 \:G는 \:connected \right\}$$로 두면 $$\mathcal{M_1} = (E, \mathcal{I_1})$$는 matroid이다.  $$\mathcal{I_2} = \left\{I : I \subset E, E - I \: 중 \: 색깔이 \: G\: 또는 \: B \:인 \:것만 \: 고려해도 \:G는 \:connected \right\}$$로 두면 $$\mathcal{M_2} = (E, \mathcal{I_1})$$는 matroid이다. 

주어진 $$K$$에 대해, 두 matroid의 intersection 에서 크기가 $$\lvert E \rvert-K$$인 것들 중  maximum weighted set을 구하면 그 여집합이 답이 됨을 쉽게 알 수 있다. 그럼 이제 [문제 3]을 해결하는 알고리즘을 이용하면 이 문제를 해결할 수 있다.

다음은 이 문제의 AC 코드이다. 

```c++
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
int n, m, Res[110], used[110];
struct Edge{
    int a, b, c;
    char ch;
}w[110];

struct DSU{ //union find
    int UF[110];
    void init(){
        for(int i=1;i<=n;i++)UF[i]=i;
    }
    int Find(int a){
        if(a==UF[a])return a;
        return UF[a] = Find(UF[a]);
    }
    bool Merge(int a, int b){
        a=Find(a),b=Find(b);
        if(a==b)return false;
        UF[a]=b;
        return true;
    }
    bool ok(const char *ch){
        init();
        int i, com = n;
        for(i=0;i<m;i++){
            if(used[i]){
                if(w[i].ch == ch[0] || w[i].ch == ch[1]){
                    if(Merge(w[i].a,w[i].b))com--;
                }
            }
        }
        return com == 1;
    }
}RG, GB;

struct Graph{
    vector<int>E[110], L[110];
    int D[110], inQ[110], Q[101000], Path[110];
    void init(){
        for(int i=0;i<=m+2;i++)E[i].clear(),L[i].clear();
    }
    void Add_Edge(int a, int b, int c){
        E[a].push_back(b);
        L[a].push_back(c);
    }
    bool SPFA(){ //find minimum length path from X1 to X2
        int i, head = 0, tail = 0;
        for(i=0;i<=m+2;i++)D[i]=1e9, inQ[i] = 0;
        D[m] = 0; Q[++tail] = m, inQ[m] = 1;
        while(head < tail){
            int x = Q[++head];
            inQ[x] = 0;
            for(i=0;i<E[x].size();i++){
                if(D[E[x][i]] > D[x] + L[x][i]){
                    D[E[x][i]] = D[x] + L[x][i];
                    Path[E[x][i]] = x;
                    if(!inQ[E[x][i]]){
                        inQ[E[x][i]] = 1;
                        Q[++tail] = E[x][i];
                    }
                }
            }
        }
        if(D[m+1] > 8e8)return false;
        int x = m+1;
        while(x!=m){
            if(x<m)used[x] = !used[x];
            x = Path[x];
        }
        return true;
    }
}GG;

bool Do(){
    int i, j;
    GG.init();
    for(i=0;i<m;i++){ //make graph GG
        if(used[i]){
            used[i] = 0;
            if(RG.ok("RG"))GG.Add_Edge(m,i,-w[i].c*1000+1);
            if(GB.ok("GB"))GG.Add_Edge(i,m+1,1);
            used[i] = 1;
        }
        for(j=0;j<m;j++){
            if(!used[i] && used[j]){
                used[i] = 1, used[j] = 0;
                if(RG.ok("RG"))GG.Add_Edge(i,j,-w[j].c * 1000 +1);
                if(GB.ok("GB"))GG.Add_Edge(j,i,w[i].c*1000 + 1);
                used[i] = 0, used[j] = 1;
            }
        }
    }
    if(!GG.SPFA())return false;
    return true;
}

int main(){
    int i, j;
    char pp[3];
    scanf("%d%d",&n,&m);
    for(i=0;i<m;i++){
        scanf("%d%d%d%s",&w[i].a,&w[i].b,&w[i].c, pp);
        w[i].ch = pp[0];
    }
    for(i=1;i<=m;i++){
        Res[i] = -1;
        used[i-1] = 1;
    }

    if(RG.ok("RG") && GB.ok("GB")){
        Res[m] = 0;
        for(i=0;i<m;i++){
            Res[m] += w[i].c;
        }
        for(i=m-1;i>=1;i--){
            if(!Do())break;
            Res[i] = 0;
            for(j=0;j<m;j++)if(used[j])Res[i] += w[j].c;
        }
    }
    for(i=1;i<=m;i++)printf("%d\n",Res[i]);
}
```



### 요약

두 글에 걸쳐서 matroid 및 matroid intersection에 대해 알아보았다. 저번 글에서는 matroid의 개념, 예시 및 관련 용어(base, rank function) 등에 대해 간략히 소개하고 maximum weight independent set을 구하는 알고리즘 및 그 정당성을 자세하게 설명하였다. 본 글에서는 두 matroid의 common independent set 중 크기가 가장 큰 것, 가중치가 주어졌을 때는 최대 가중치를 가지는 것을 구하는 알고리즘을 증명과 함께 제시하였다. 그리고 두 글 모두 제시한 알고리즘을 이용해 풀어 볼 수 있는 문제를 여러 가지 제시하였다. Problem solving에서는 matroid 관련 문제가 그렇게 자주 나오지는 않지만, matroid에 관한 재미있는 사실은 많이 있으니 관심이 있는 사람들은 찾아보는 것을 추천한다.
