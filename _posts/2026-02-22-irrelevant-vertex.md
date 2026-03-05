---
layout: post
title: "Irrelevant Vertex Technique"
date: 2026-02-22
author: leejseo
tags: [algorithm, graph theory]
---

## 서론

지난 두 편의 글에서 $\mathcal{F}$-deletion 문제에 대해 살펴보았다. $\mathcal{F}$-deletion 문제의 optimal FPT 알고리즘에서는 큰 flat wall 중심부에 irrelevant vertex가 존재한다는 사실이 매우 중요한 요소로 사용되었다.

Irrelevant vertex technique은 Graph Minor Theory에서 등장하는 일종의 '축소' 기법으로, Robertson과 Seymour의 Graph Minors 논문 시리즈에서 처음 등장했다. 이는 그래프에서 특정한 패턴 혹은 구조를 찾는 문제들에서 활용되며, 다음의 아이디어에 기반한다:

> 어떤 문제에 대해 $G$ 안에 찾고자 하는 구조의 존재 여부에 전혀 영향을 주지 않는 정점 $v$가 존재한다면, $v$를 삭제해도 문제의 정답은 변하지 않는다.

Decision problem의 맥락에서 생각하자면, $G$가 YES instance $\iff$ $G-v$가 YES instance 가 되도록 하는 정점 $v$를 찾는 기법이라고 생각하면 된다.

이 기법을 활용하여 해결할 수 있는 가장 대표적인 문제는 $k$-Disjoint Paths 문제이다.

**Problem.** ($k$-Disjoint Paths) 그래프 $G$와 정점의 쌍 $(s_1, t_1), (s_2, t_2), \cdots, (s_k, t_k)$ 가 주어졌을 때, $s_i$ 와 $t_i$ 를 양 끝점으로 하며 서로 vertex disjoint 한 경로 $P_1, P_2, \cdots, P_k$ 가 존재하는가?

이 문제는 일반적으로는 NP-난해 문제이나, $k$를 고정한 경우, FPT 알고리즘이 존재한다. 여기에서의 핵심 아이디어가 바로 irrelevant vertex technique이다. 이 글에서는

- irrelevant vertex technique,
- irrelevant vertex technique을 활용한 $k$-disjoint paths 문제의 FPT 알고리즘,
- irrelevant vertex에 기반한 최근의 후속 연구

를 살펴볼 것이다. 참고로, 이 글을 읽기 위해서는 tree decomposition과 branch decomposition에 대해 알고 있으면 좋다.

## Irrelevant Vertex

이 절에서는 [1]에서 제안한 irrelevant vertex theorem을 소개한다. 세부적인 증명을 소개하기 보단, FPT 알고리즘을 설계할 때 black-box로 사용할 수 있는 도구로서 그 결과와, 이를 이해하기 위한 여러 개념들을 소개한다.

### Irrelevant Vertex의 정의

**Minor Model.** 그래프 $H$가 $G$의 minor 임은 $G$에서 간선과 정점을 삭제하는 연산과, 간선을 contract 하는 연산 만으로 $H$를 만들 수 있음을 의미한다. 반대로 말해서, mapping $\phi : V(H) \to 2^{V(G)}$가 존재해,

- 각 정점 $v$에 대해 $\phi(v)$ 는 $G$의 연결 부분그래프 (branch set) 이며,
- 서로 다른 두 정점 $u, v \in V(H)$에 대해 $\phi(u) \cap \phi(v) = \emptyset$ 이며 (mutually vertex disjoint),
- 각 간선 $e = uv$에 대응되는 $\phi(u) $ 와 $\phi(v)$ 를 잇는 $G$의 간선이 존재

한다는 것이다. 이 때, $\phi$ 를 $G$에서 $H$의 minor model이라 부른다.

**Rooted Graph.** 그래프 $G$가 있을 때, $G$의 일부 정점 $v_1, v_2, \cdots, v_k$ 를 뽑아서 root로 label한 rooted graph $(G, v_1, v_2, \cdots, v_k)$ 를 생각할 수 있다. 대충 말해서, '건드리지 않을' 정점들에 이런 label을 단다고 생각하면 좋다.

Rooted graph에 대해서도 비슷하게 minor 개념을 정의할 수 있다. $(H, u_1, u_2, \cdots, u_k)$가 $(G, v_1, v_2, \cdots, v_k)$ 의 minor임은, mapping $\phi$가 존재해

- $\phi$가 $G$에서 $H$의 minor model이고,
- $\phi(u_i) = \{v_i\}$

임으로 정의하고, 이러한 $\phi$ 를 rooted minor model이라 부른다. 즉, root 들은 삭제되지 않고, minor를 취할 때도 여전히 root로 남아있게 하고 싶은 것이다.

그리고 어떤 rooted digraph가 *작다*는 것을 말하기 위해 *detail*을 정의한다. Rooted graph $(H, u_1, u_2, \cdots, u_k)$ 이 detail $\le \delta$ 임은

* $\vert E(H) \vert \le \delta$
* $\vert V(H) \setminus \{u_1, \cdots, u_k \} \vert \le \delta$

으로 정의한다.

**Folio.** Rooted graph $(G, v_1, \cdots, v_k)$ 의 $\newcommand{\folio}{\mathsf{folio}} \folio$를 이 rooted graph가 가지는 모든 rooted minor (up to isomorphism)의 집합으로 정의하며, $\newcommand{\dfolio}{\text{-}\folio} \delta \dfolio$를 folio 에서 detail $\le \delta$ 인 원소만 모은 집합으로 정의한다.

여기에서 중요한 사실은, $k, \delta$가 고정되었을 때, $\delta \dfolio$ 의 가짓수는 유한하다.

그리고 그래프 $G$와 $Z \subseteq V(G), \vert Z \vert = k$ 가 주어졌을 때, $G$ relative to $Z$의 $\delta \dfolio$ 는 $Z = \{v_1, \cdots, v_k \}$ 라 할 때 $(G, v_1, \cdots, v_k)$ 의 $\delta \dfolio$ 로 정의한다.

**Irrelevant Vertex.** 이 글에서 다룰 disjoint paths 나 $\mathcal{F}$-deletion 문제에서의 minor containment를 비롯한 다양한 문제를 $\delta\dfolio$ 를 계산하는 문제로 바꿀 수 있다. 그래서 irrelevant vertex는 folio의 관점에서 정의되고, 활용된다.

$v \not \in Z$ 인 정점 $v$ 가 $G$ relative to $Z$ 의 $\delta \dfolio$ 에 대해 irrelevant 함은

$$ \delta \dfolio (G \text{ rel }Z ) = \delta \dfolio (G\setminus v \text { rel } Z) $$

으로 정의된다. 즉, 직관적으로 말하자면, root 집합 $Z$를 기준으로 봤을 때, $v$ 를 지워도 detail $\delta$ 이내의 작은 rooted minor에 대한 정보가 변하지 않음을 뜻하며, 그래서 $v$가 "문제"의 정답에 아무런 영향이 없는 정점이 되어 지울 수 있게 된다.

### Wall 과 Wall의 형태

우리가 살펴볼 irrelevant vertex theorem은 *homogeneously* labeled *wall*을 필요로 한다. 이를 위해 wall과 그의 homogeneity를 정의해야 한다. 먼저, wall의 정의에 대해 살펴보자.

![img](/assets/images/leejseo/wall.png)

**Wall.** Elementary $r$-wall은 $r \times 2r$ 격자에서 alternating pattern으로 vertical edge를 제거하고, 재귀적으로 차수 1인 정점을 제거해서 만들어진 그래프이고, 위 그림은 elementary 5-wall을 시각화 한 것이다.

Elementary $r$-wall을 subdivide한 형태의 그래프들을 $r$-wall이라 한다. 편의상 $r$-wall의 높이 $h$를 $r-1$로 정의하자. (즉, 높이 $h$는 세로 방향 레이어 수 같은 것을 의미한다고 보면 된다.) Wall 상에서 다음과 같은 개념들을 정의할 것이다.

- perimeter $C$: wall의 바깥 테두리 cycle
- corner: 좌상단, 우상단, 좌하단, 우하단 4개의 정점.
- peg: $C$ 상의 차수 2인 정점 가운데 subdivision 이전의 elementary wall 에서 온 정점. corner를 포함.
  - 참고로 $C$ 상에 subdivision vertex가 있을 수 있기 때문에, peg와 corner의 choice로 여러 조합이 가능할 수 있다.
- subwall: 큰 wall 안에 포함되는 더 작은 wall (이 역시도 $r' \times 2r'$ 격자로 부터 얻어져야 한다.)
- middle vertex: subdivision 이전 elementary wall 에서 정중앙에 있는 두 정점에 대응되는 두 정점. 위 그림에서는 3행의 왼쪽에서 5,6번째 정점이 이에 해당한다.

그래프가 wall을 부분 그래프로 가지고 있을 때, 우리는 wall과 주변부를 disk 위에 그리는 상황을 다룰 것이다. 이를 위해 wall의 주변부(compass)를 논하기 위한 개념을 살펴보자.

**Compass.** $H$가 $G$ 안의 wall 이고, perimeter를 $C$ 라고 하자. Compass $K$는

* $C$ 와
* $G \setminus V(C)$ 에서 $H \setminus V(C)$ 가 포함되는 컴포넌트

를 합친 부분 그래프로 정의된다. 그리고 이와 4개 corner의 $C$ 상의 순서를 나타내는 cyclic permutation $\Omega$ 를 합쳐서 society $(K, \Omega)$ 라고 부른다.

**Drawing.** Society $(G, \Omega)$가 있을 때, wall이 planar 함에도 불구하고, compass $G$ 자체는 planar 하지 않을 수 있다. 그래서 $G$를 여러 조각으로 쪼갠 후, 각 조각 안에서는 non-planar 함을 허용하더라도, 조각들 간 연결은 planar 하도록 할 것이다. 이를 위해 $G$를 쪼개기 위한 *division* 이라는 도구와 조각들 간 연결이 planar 함을 의미하는 *rural* 이라는 개념을 도입한다.

*Division.* $G$의 부분 그래프 $A$에 대해 $\Omega \cap A$ 와 $E(G) \setminus E(A)$ 의 간선에 incident 한 $A$의 정점들을 모아 $\partial A$로 표기할 것이다. 즉, $\partial A$는 $A$의 경계를 의미한다. 이를 기반으로 다음을 만족하는 $G$의 부분 그래프들의 집합 $\mathcal {A}$가 *division* 이라고 부를 것이다:

* $\mathcal{A}$ 의 원소를 전부 합집합 하면 $G$가 됨
* 서로 다른 두 $A, A' \in \mathcal{A}$ 에 대해 $E(A) \cap E(A') = \emptyset$, $V(A) \cap V(A') = \partial A \cap \partial A'$
* 각 $A \in \mathcal{A}$에 대해 $\vert \partial A \vert \le 3$ 이며, $\partial A$ 와 $\Omega$ 사이에 $\vert \partial A \vert $ 개의 mutually vertex-disjoint path가 존재
* 각 $A \in \mathcal{A}$와 $u, v \in \partial A$ 에 대해 $\partial A$에 속하는 정점을 internal vertex로 가지지 않는 $A$ 상의 경로 존재
* 서로 다른 $A, A' \in \mathcal{A}$ 에 대해 $\partial A \neq \partial A'$
* 각 $A \in \mathcal{A}$ 에 대해 다른 $\mathcal{A}$ 의 원소에는 속하지 않는 $A$ 상의 정점 혹은 간선이 존재

*Rural Division.* $(G, \Omega)$ 가 rural 임은 $\Omega$ 의 정점을 disk의 경계에 차례로 나열하는 $G$의 drawing이 존재함으로 정의된다. 그리고 society $(G, \Omega)$ 의 division $\mathcal{A}$ 가 *rural division* 임은

* 이분 그래프 $G'$ 을
  * $V' = V(A) \cup (\bigcup \{\partial A \mid A \in \mathcal{A} \})$,
  * $v \in \partial A$ 이면 $A$ 와 $v$ 를 간선으로 이어서 만들었을 때
* $(G', \Omega)$ 가 rural 임으로 정의한다.

즉, division에 추가적으로 planar-like 한 성질을 강제하기 위한 조건들이 붙었다고 보면 좋다.

*요약.* Division과 Rural division의 정의가 다소 복잡하나, 실질적으로 중요한 것은,

* rural division에서 각 '조각' $A$는 $\vert \partial A \vert \le 3$ 이며,
* 각 '조각' $A$ 들이 society의 boundary $\Omega$ 와 '잘' 연결된 형태

라는 사실 정도라고 할 수 있겠다.

### Flat Wall

Graph Minor Theory의 결과로, flat wall theorem에서 다음이 알려져 있다:

> $G$의 treewidth가 충분히 크고, 큰 clique을 minor로 가지지 않는다면, 적당한 크기의 집합 $X \subseteq V(G)$ 가 존재해 $G \setminus X$ 에 큰 wall $H$가 존재하고, $G \setminus X$ 안에서 $H$의 society $(K, \Omega)$ 에 대한 rural division이 존재한다.

Wall과 함께 irrelevant vertex technique을 응용하는 FPT 알고리즘의 설계는 대개 flat wall theorem에 의존하기 때문에, 이 절 및 다음 절에서는 아래와 같은 상황을 가정한다.

- $G$: 그래프
- $X \subseteq V(G)$, $\vert X \vert = q$: root 집합 (이후 $\delta \dfolio$를 relative to $X$로 볼 것이다.)
- $H$: $G \setminus X$ 안의 wall, $C$를 perimeter로 가짐
- $\mathcal{A}$: $G \setminus X$ 에서 $H$의 society $(K, \Omega)$에 대한 rural subdivision

**조각 다루기.** $A \in \mathcal{A}$ 가 $A \cap C = \emptyset$ 이라면, 이를 interior라 부르자. 이 interior '조각'$A$를 생각해보면, 자신의 밖과 3개 이내의 정점 $\partial A$ 를 통해서만 이어질 수 있고, 추가적으로 root 집합 $X$ 와도 이어질 수 있다.

그래서 interior $A$ 마다 $\partial A$ 를 $s_1, s_2, \cdots, s_k$ ($k = \vert \partial A \vert \le 3$) 로 번호를 붙이고, root $X = \{x_1, \cdots, x_q \}$ 와 붙여서 attachment sequence $\pi(A) = (s_1, \cdots s_k, x_1, \cdots, x_q)$ 를 정의한다. attachment sequence는 $A$가 '밖'과 상호작용 하는데 사용할 수 있는 정점들을 root로 묶어서 folio를 계산하기 쉽게 만들어준다.

$G \setminus X$ 상의 rural subdivision $\mathcal {A}$ 를 $G$와 함께 고려하기 위해, 각 $A \in \mathcal{A}$ 마다 $G$의 부분 그래프 $\tilde A$ 를 다음을 만족하게 잡자:

* $V(\tilde A) = V(A) \cup X$
* $\tilde A \setminus X = A$
* $G$의 간선 중 양 끝점이 모두 $X \cup V(K)$ 에 속하는 간선은 하나의 $A$에 대해서만 $E(\tilde A)$ 에 속함

즉, $X$ 내부 간선들이 '조각' $A$ 들에 중복 없이 배정되도록 잡았다고 생각하면 된다. 이 상에서, $T_A := \delta \dfolio((\tilde A, \pi(A)))$ 로 $A$ 마다 label을 붙여줄 것이다.

이렇게 잡은 $T_A$는 $G$ 에서 $A$가 '줄 수 있는' 작은 rooted minor가 무엇이 있는지, 그 정보를 나타낸다고 생각하면 좋다.

**조각의 위치.** 각 interior '조각' $A$ 마다 하나의 정점 $v(A) \in V(H)$ 를 골라서

* $K$ 안에 $\partial A$ 에서 $v(A)$ 로 가는 경로가 있고,
* 그 경로가 wall $H$ 의 정점을 $v(A)$ 말고는 만나지 않게

할 것이다. $v(A)$는 각 조각 $A$가 wall의 어느 부분에 있는지를 wall 정점을 통해 알려주는 역할을 한다고 보면 된다. 이 때의 모든 정보 (각 조각 $A$, 그마다의 $\tilde A$, $\delta \dfolio(\tilde A, \pi(A))$, $v(A)$) 를 통틀어서 *vision* 이라고 부를 것이다.

### Homogeneous Wall과 Irrelevant Vertex Theorem

Vision까지 고정한 상태에서, $H$의 subwall $H'$ 이 $h$-homogeneous 함은 다음과 같이 정의된다:

* $v(A) \in H'$ 인 모든 interior 조각 $A$와
* $H'$ 의 모든 높이 $h$ subwall $H''$에 대해
* $H''$ 내의 다른 interior 조각 $A'$이 존재해 $T_A = T_{A'}$, $v(A') \in V(H'')$ 을 만족한다.

즉 다시 말해, 어떤 'label'($\delta \dfolio$)이 $H'$에 한 번이라도 나타나면, 그 label은 $H'$ 내의 모든 다른 높이 $h$ subwall에 나타난다는 것이다. 즉, $H'$ 은 균일한 $\delta\dfolio$ 를 가지고 있다고 말 할 수 있다.

이를 기반으로, [1]의 Irrelevant vertex theorem이 주는 결과는 다음과 같다.

**Irrelevant Vertex Theorem.** ([1]의 10.2) 지금 까지의 setup을 가정하자. $q, \delta$ 를 고정했을 때, 어떤 함수 $f_{q, \delta} : 2\mathbb{N} \to 2\mathbb{N}$ 이 존재해, $H' \subseteq H$ 가 높이 $f(h)$ 인 $h$-homogeneous subwall이면, $H'$의 두 middle vertex는 모두 $G$ relative to $X$ 의 $\delta \dfolio$ 에 대해 irrelevant 하다.
다시 말해, 각 middle vertex $m$에 대해 $\delta \dfolio(G \text{ rel } X) = \delta \dfolio(G \setminus m \text{ rel } X) $.

이의 증명은 매우 난해하나, 다음과 같은 직관으로 정당화할 수 있다.

* 앞에서 label $T_A$ 로 각 '조각' 들이 wall에 붙는 방식과, 이로 인해 생길 수 있는 작은 minor 들을 나타냈다.
* 그리고 $h$-homogeneous 함은 이 패턴이 wall 전체에서 반복됨을 의미한다.
* 그러면, homogeneous한 영역이 충분히 크다면, 어떤 작은 rooted minor (detail $\le \delta$) 가  middle vertex를 사용하는 경우가 있을 때, homogeneity를 기반으로 이를 middle vertex을 피하도록 살짝 옮겨줘도, 여전히 같은 $T_A$ 들의 조합으로 나타낼 수 있다.
* 그래서 middle vertex를 지워도 $\delta \dfolio$ 가 변하지 않는다.

우리는 wall 버전의 irrelevant vertex theorem을 살펴봤지만, [1]에서는 clique 버전의 irrelevant vertex theorem도 제공한다.

이는, 어느 그래프 $G$가 $g(\delta , \vert X \vert)$ 이상 크기의 clique을 minor로 가진다면, 그 안에 $G$ relative to $X$의 $\delta \dfolio$ 에 irrelevant한 정점이 존재한다는 것이다.

## Disjoint Paths 문제에 적용하기

이전 섹션에서 그래프가 포함하는 작은 구조들을 의미하는 $\delta\dfolio$ 와, 이에 기반한 irrelevant vertex theorem을 살펴보았다. Folio와 disjoint paths 문제가 어떻게 이어지는지 살펴보자.

**Lemma.** $G$ relative to $Z$의 $\delta \dfolio$ 를 계산하는 $\vert Z \vert$ 에 대한 FPT 알고리즘이 존재한다면, disjoint paths 문제의 FPT 알고리즘 또한 존재한다.

*Proof.* $Z = \{s_1, t_1, \cdots, s_k, t_k \}$ 에 대해 $G$ relative to $Z$ 의 0-folio를 살펴보는 방식으로 disjoint paths 문제를 $\delta \dfolio$ 를 계산하는 문제로 reduce 할 수 있다. $ \square$

그래프 $G$와 $Z$, $\delta$가 주어졌을 때, 대강 기술하자면 아래와 같은 방법으로 $\delta\dfolio$ of $G$ relative to $Z$ 에 대한 FPT 알고리즘을 [1]에서 제시하였다.

1. $G$의 treewidth가 큰 경우 (in terms of $\delta, \vert Z \vert$)
   * 큰 clique이 있다면, 이를 찾고 clique 버전의 irrelevant vertex theorem을 적용해, $\delta\dfolio$ 에 대한 irrelevant vertex를 찾고 삭제한다.
   * 큰 clique이 없다면, 큰 wall 구조를 찾을 수 있을 것이다. 여기에서 wall 버전의 irrelevant vertex theorem을 적용해 $\delta \dfolio$ 에 대한 irrelevant vertex를 찾고 삭제한다.
2. $G$의 treewidth가 작은 경우
   * 이 경우, $G$의 branchwidth 또한 작다.
   * 따라서, branch decomposition을 기반으로 dynamic programming 하는 접근을 생각할 수 있다.

이 방법론에서 1번 경우의 재귀 반복 횟수나, 2번 경우에 관리하는 상태의 수가 매우 큰 것은 사실이다. 하지만, 이는 $G$의 크기가 아닌, $\delta$, $\vert Z \vert$ 에 depend 하므로, FPT 알고리즘을 얻게 된다.

## 최근의 후속 연구

실제 수행 시간의 측면에서 본다면, $G$의 크기에 대한 다항식 부분과 $\delta, \vert Z \vert$ 에 depend 하는 term이 있다. 후자는 알고리즘에 사용되는 그래프 구조 관련 정리들의 bound 개선과 직접적인 연관이 있다.

전자와 관련해서는 [2]와 같은 결과가 2011년에 출판되었다.

후자와 관련해서는 Graph Minors 논문 시리즈의 이후 부분에서 나오는 Unique Linkage 등의 개념과 관련하여, planar graph와 같은 특수한 그래프에서 개선하는 [3]과 같은 결과가 알려져있다.

## Reference

[1] N. Robertson and P.D.Seymour, Graph Minors XIII. The Disjoint Paths Problem, Journal of Combinatorial Theory, Series B, 1995

[2] K-i. Kawarabayashi, Y. Kobayashi and B. Reed, The disjoint paths problem in quadratic time, Journal of Combinatorial Theory, Series B, 2011

[3] I. Adler, S. G. Kolliopoulous, P. K. Krause, D. Lokshtanov, S. Saurabh and D. M. Thilikos, Irrelevant vertices for the planar Disjoint Paths Problem, Journal of Combinatorial Theory, Series B, 2016
