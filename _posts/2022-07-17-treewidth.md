---
layout: post
title:  "Treewidth를 사용한 PS 문제 해결"
date:   2022-07-17
author: koosaga
tags: [algorithm, graph theory, problem solving]
---

알고리즘에서 다루는 많은 문제들은 그래프 문제로 환원할 수 있는데, 일반적인 그래프에서 어떤 문제들은 효율적으로 해결이 불가능한 경우가 있다. 이러한 비효율성의 대표적인 예시는 NP-hard로, 어떠한 문제가 NP-hard일 경우 다항 시간으로 푸는 것이 아예 불가능할 가능성이 높다. 그 외에도, 최단 경로 문제와 같이 다양한 쿼리에 대해서 빠른 시간 안에 해결하는 것이 어려운 경우 등, 비효율성의 예시는 NP-hard에 한정되지 않는다.

이러한 비효율성에 당착했을 때 자주 취하는 전략은 환원한 그래프의 특수성에 의존하는 것이다. 예를 들어, NP-hard 문제들이라 하더라도 그래프가 직선, 트리, 선인장, 이분 그래프일 때는 새로운 알고리즘적 기법들을 활용할 수 있다. 대표적인 패턴이자 이 글의 메인 주제가 되는 **트리** 의 경우는, 동적 계획법 (트리 DP) 나 경로 분할 (HLD), 분할 정복 (Centroid decomposition) 등 효율화할 수 있는 다양한 알고리즘 테크닉이 알려져 있기 때문에, 여러 상황에서 매력적인 구조가 된다.

그래프가 트리가 아니더라도, 트리와 유사한 구조들은 다양한 상황에서 만날 수 있다. 대표적인 예시로는 겹치지 않는 사이클들의 트리로 표현되는 *Cactus*, 볼록 다각형의 삼각 분할 형태로 표현되는 *Outerplanar graph*, 직렬/병렬 회로의 이진 트리 형태로 표현되는 *Series-parallel graph*, 삼각형들의 삼진 트리 형태로 표현되는 *Apollonian network*, 클리크들의 직선으로 표현되는 *Interval graph*, 클리크들의 트리로 표현되는 *Chordal graph* 등이 존재한다. 이러한 그래프는 문제에서 명시적으로 주어지는 경우도 있고, 문제의 성질을 분석하였을 때 자연스럽게 발생하는 경우도 존재한다. PS에서도 이러한 경우는 정말 많이 등장하며, 문제에 *Cactus* 나 *Outerplanar graph* 라는 표현이 직접 존재하지 않아도 내부를 살펴보면 이러한 그래프의 성질을 띄고 있는 경우가 많다.

이러한 *유사 트리* 들에 대해서도 트리에서 사용한 알고리즘 테크닉을 적용할 수 있는 경우가 많다. 하지만, 매번 다른 패턴이 나왔을 때 이에 맞춰 ad-hoc하게 방법을 수정해야 한다는 것이 큰 불편함으로 작용한다. Treewidth라는 개념은, 다양한 형태의 *유사 트리* 들을 일반화하여, 트리의 테크닉을 적용하기 편한 형태로 표현한다. 예를 들어, Cactus, Outerplanar graph, Series-parallel graph, Apollonian network는 모두 Treewidth가 3 이하인 그래프로, Treewidth가 아주 작을 경우 선형 시간에 이를 트리 형태의 표현으로 변환해 줄 수 있다.

이 글에서는 이러한 Treewidth의 정의와, Treewidth가 작은 그래프의 효율적인 Recognition, 그리고 Treewidth가 작은 그래프에서의 문제 해결 전략을 짚는다.

## 1. Treewidth의 정의
### 1.1 Pathwidth와 Interval Graph
Treewidth라는 개념은 직관적으로 이해하기가 쉽지 않은 개념이라, 설명에 앞서 조금 더 간단한 개념인 Pathwidth를 조명한다. Pathwidth라는 것은 [Treewidth와 함께 자주 언급되는 유사한 개념](https://en.wikipedia.org/wiki/Treewidth#Pathwidth) 에 속하지만, 사실 이 글에서 Pathwidth라는 개념은 중요하지 않다. 순수하게 Treewidth의 이해를 돕는 목적으로 보면 좋을 것 같다.

Interval Graph는 다음과 같이 정의되는 그래프이다:
* 각 정점에 대응되는 구간 $[S_i, E_i]$ 이 존재한다.
* 두 정점간에 간선이 있다는 것은, 두 구간에 교집합이 있다는 것과 동일하다. (즉, $max(S_i, S_j) \le min(E_i, E_j)$)

즉, 수직선 상에 $N$ 개의 주어졌을 때, 이 $N$ 개의 구간 간에 겹치는 부분이 있으면 간선을 이어준 것이라고 볼 수 있다. Interval Graph의 **Pathwidth** 를, (이 그래프에서 가장 많은 구간이 겹쳤을 때 겹친 구간의 개수) - 1이라고 하자. 즉, $stab(x) = \{i | S_i \le x \le E_i\}$ 라고 하면, Pathwidth는 $max_{x}|stab(x)| - 1$ 이 된다. (1을 안 뺄 경우 간선이 있는 그래프의 Pathwidth는 무조건 2 이상에서 시작한다. 아마 그래서 1을 빼고 정의하는 게 아닌가 싶다. 사실 나도 굳이 뺄 이유가 뭔지 모르겠다.)

만약 Interval Graph의 Pathwidth가 작을 경우, 어떤 문제들을 풀 수 있을까? 다음과 같은 예시를 살펴보자. (사실 아래 문제는 Pathwidth와 무관하게 $O(n \log n)$ 에 풀 수 있으나 그냥 설명을 위한 풀이로 이해하면 좋을 것 같다.)

**Minimum Dominating Set.** 어떠한 정점 부분집합 $S$가 *Dominating Set* 이라는 것은, 모든 정점 $v$ 에 대해 $v$ 혹은 $v$ 에 인접한 어떠한 정점이 $S$ 에 속한다는 것을 뜻한다. Pathwidth가 작은 Interval Graph에서 Dominating Set의 최소 크기를 출력하라.
**Solution: Bitmask DP.** $S_i, E_i$ 를 모두 좌표압축하여, $1 \le S_i \le E_i \le 2N$ 을 가정하자. $DP[i][S_1][S_2]$ 를, 현재 좌표상 위치 $i$ 를 보고 있으며, $S_2 \subseteq stab(i)$ 에 있는 정점들은 Dominating Set에 있거나 이에 인접하고, $S_1 \subseteq S_2$ 는 Dominating Set에 있다고 하자. 상태 전이를 다음과 같이 할 수 있다.
* 새롭게 Dominating Set에 넣을 집합 $X \subseteq stab(i) - S_1, X \neq \emptyset$ 을 고른 후, $DP[i][S_1 + X][S_2 + adj(X)] + |X|$ 에서 값을 가져온다.
* 넣을 것이 없다면 $DP[i + 1][S_1 \cap stab(i + 1)][S_2 \cap stab(i + 1)]$ 를 받아오는데, 이 때 $S_2$ 에 속하지 않으면서 커버가 안 된 정점이 사라진다면 (즉, $E_x = i$ 라서 $x \in stab(i) - stab(i + 1)$ 이라면) 잘못된 상태 전이이니 하지 않는다.

이와 같이, Pathwidth가 작은 Interval Graph의 경우, Pathwidth에 대해서**만** 비효율적인, 심지어 Exponential한 알고리즘을 얻더라도 전체적으로 보았을 때는 충분히 효율적인 알고리즘이 될 수 있다. 이러한 알고리즘들을 흔히 Fixed-parameter tractable algorithm (FPT algorithm) 이라고 부른다.

이제 Pathwidth라는 개념을 Interval Graph를 넘어서 일반 그래프에 대해서도 정의한다.

**Pathwidth of Interval Graph.** 그래프 $G$ 가 Interval graph라는 것은 위와 같은 정의가 만족하는 구간 수열 $[S_1, E_1], \ldots, [S_n, E_n]$ 이 존재함을 뜻한다. $G$ 의 Pathwidth는 모든 올바른 구간 수열 중 $max_{x}|stab(x)|-1$ 의 최솟값이다.
**Pathwidth (Definition 1).** 그래프 $G = (V, E)$ 에 대해, 어떠한 간선 집합 $E^\prime$ 이 *interval completion* 이라는 것은 $E \cap E^\prime = \emptyset$ 이며 $(V, E \cup E^\prime)$ 이 *interval graph* 임을 뜻한다. $G$ 의 모든 *interval completion* 중 최소 Pathwidth를 $G$ 의 Pathwidth라고 정의한다.

말이 길었지만, 쉽게 말해 그래프에 적당히 간선을 추가하여 Interval graph를 만들어서 최소 Pathwidth를 만들었을 때, 가능한 Pathwidth의 최솟값이 그래프의 Pathwidth라는 것이다. 아니면 이렇게 Pathwidth를 정의할 수도 있다.

**Pathwidth (Definition 1.5).** 그래프 $G$ 의 구간 수열 $[S_1, E_1], \ldots, [S_n, E_n]$ 이 *올바르다* 는 것은, 모든 간선 $(u, v) \in E$ 에 대해 $max(S_u, S_v) \le min(E_u, E_v)$ 가 성립함을 뜻한다. $G$ 의 *Pathwidth* 는 모든 가능한 올바른 구간 수열 중  $max_{x}|stab(x)|-1$ 의 최솟값이다.

겹친다와 간선의 존재성이 필요충분이 아니기 때문에, *interval completion* 의 개념이 자연스럽게 유도된다.

사실 위에서 정의한 Pathwidth 개념은 "겹치는 구간 개수" 등을 정의하면서 다소간에 난잡한 면이 있다. 아래와 같은 정의를 사용하면, 겹치는 구간 같은 개념 없이 집합만을 사용하여 정의할 수 있다.

**Pathwidth (Definition 2).** 그래프 $G$ 의 *Path decomposition* 은 $V$ 의 부분 집합 $X_1, X_2, \ldots, X_p \subseteq V$ 로 구성되며 다음과 같은 성질을 만족한다.
* $\bigcup_{i} X_i = V$
* $i < j$ 에 대해서, $v \in X_i \cap X_j$ 일 경우, 모든 $i < k < j$ 에 대해서 $v \in X_k$ 가 만족된다.
* 모든 간선 $(u, v) \in G$ 에 대해서, $u, v$ 를 모두 포함하는 집합 $X_i$ 가 존재한다.

Path decomposition의 *width* 는 $max |X_i| - 1$ 이다. $G$ 의 *Pathwidth* 는 Path decomposition이 가질 수 있는 최소 *width*이다.

$v \in X_i$ 라는 것이 $S_i \le v \le E_i$ 에 대응됨을 관찰하면 정의를 이해할 수 있을 것이다. Definition 2이 가장 표준적으로 사용되는 정의이며, Definition 1이 가장 직관적인 이해에 가까운 정의라고 생각한다. *Path decomposition* 의 quality를 좌우하는 요인 중 하나는 width와 $p$ 가 될 텐데, 좌표 압축의 예시에서 보았듯이, 모든 Path decomposition에 대해 $p = O(|V|)$ 이 성립하도록 줄여줄 수 있다.

### 1.2 Treewidth와 Chordal Graph
앞서 Pathwidth를 설명하기 위하여, Interval Graph라는 구조를 설명하고 이를 통해서 정의를 유도했다. Treewidth에서 이에 대응되는 그래프에는 Chordal Graph가 있으나, Interval Graph만큼 직관적이거나 친숙한 구조가 아니기 때문에 반대로 Treewidth에서 Chordal Graph의 정의를 유도하도록 한다.

Path decomposition은 특정한 정점을 **수직선 상의 구간**에 대응시키는 형태의 모델이었으나, Tree decomposition에서는 조금 더 오브젝트가 복잡하다. Tree decomposition에서, 특정한 정점은 **어떠한 트리 상의 연결된 서브트리**에 대응된다. 이 *어떠한 트리* 는 Tree decomposition 알고리즘이 원본 그래프 $G$ 를 받고 반환할 것이며, 대략 $G$가 *트리 비슷한 무언가* 라면 알고리즘은 여기서 $G$ 의 *뼈대가 되는 트리* 를 반환한다고 이해하면 좋다. Path decomposition과 다르게, 사영되는 대상이 수직선과 같은 Universal한 객체가 아니라, 매 그래프마다 다를 수 있는 어떠한 트리라는 점에 주의하라.

여기까지 했으니, 이제 정의를 제시한다.

**Treewidth (Definition 2).** 그래프 $G$ 의 *Tree decomposition* 은 $V$ 의 부분 집합 $X_1, X_2, \ldots, X_p \subseteq V$ 와, $p$ 개의 정점을 가진 트리 $T$ 로 구성되며 다음과 같은 성질을 만족한다.
* $\bigcup_{i} X_i = V$
* $i, j$ 에 대해서, $v \in X_i \cap X_j$ 일 경우, $T$ 상에서 $i, j$ 를 잇는 경로 상에 있는 $k$ 에 대해서 $v \in X_k$ 가 만족된다.
* 모든 간선 $(u, v) \in G$ 에 대해서, $u, v$ 를 모두 포함하는 집합 $X_i$ 가 존재한다.

Tree decomposition의 *width* 는 $max |X_i| - 1$ 이다. $G$ 의 *Treewidth* 는 Tree decomposition이 가질 수 있는 최소 *width*이다. 각 집합 $X_i$ 를 흔히 *bag* 라는 이름으로 부른다.

**Corollary.** 그래프의 Treewidth는 Pathwidth보다 항상 작거나 같다.

Interval Graph의 Pathwidth 정의를 일반화할 때 중요한 것은, *간선의 존재 여부* 와 *집합의 교차 여부* 의 필요충분 관계를 없앤 것이다. 반대로 말해, Treewidth에서 Chordal Graph의 정의를 얻기 위해서는 필요충분 관계를 형성하는 것으로 충분하다.

**Chordal Graph.** 무방향 그래프 $G = (V, E)$가 Chordal Graph라는 것은, $p$ 개의 정점을 가진 트리 $T = (V_T, E_T)$와, 트리의 정점 부분집합 $X_1, X_2, \ldots, X_{|V|}$ 가 존재하여
* 각 정점 부분 집합이 비지 않았으며, 연결된 서브트리를 이루고
* $(i, j) \in E \iff (X_i \cap X_j) \neq \emptyset$

임을 뜻한다.

**Remark.** 짧게 말해, 트리의 connected subtree의 intersection graph이다.
**Corollary.** 모든 Interval Graph는 Chordal Graph이다.

### 1.3 Chordal Graph의 성질
위에서 우리는 Tree decomposition을 사용하여 Chordal Graph를 정의할 수 있음을 보였다. 사실 Chordal Graph는 일반적으로 Tree decomposition을 사용하여 정의하지 않고, 동치의 다른 방법으로 정의하는 것이 일반적이다. 아래에 동치인 정의들을 몇개 나열하는데, 이것들이 동치임을 이해하는 것이 이후 문단을 따라가는데 큰 도움이 될 것이다.

**Chordal Graph (Definition 1).** 무방향 그래프 $G = (V, E)$가 Chordal Graph라는 것은, $p$ 개의 정점을 가진 트리 $T = (V_T, E_T)$와, 트리의 정점 부분집합 $X_1, X_2, \ldots, X_{|V|}$ 가 존재하여
* 각 정점 부분 집합이 비지 않았으며, 연결된 서브트리를 이루고
* $(i, j) \in E \iff (X_i \cap X_j) \neq \emptyset$

임을 뜻한다.

**Chordal Graph (Definition 2).** 무방향 그래프 $G = (V, E)$ 가 Chordal Graph라는 것은, 임의의 길이 4 이상의 단순 사이클에 대해서, 사이클에 속하지는 않으나 사이클 상 두 정점을 잇는 간선 (*chord*) 가 존재함을 뜻한다.

**Chordal Graph (Definition 3).** 무방향 그래프 $G = (V, E)$ 의 Perfect Elimination Ordering은, 다음과 같은 성질을 만족하는 정점 순열 $p = \{v_1, v_2, \ldots, v_{|V|}\}$ 을 뜻한다:
* 모든 $i$ 에 대해서, $S(i) = \{v_j | j > i, (v_j, v_i) \in E\}$ 가 클리크를 이룬다.

$G$ 가 Chordal Graph라는 것은, $G$ 의 Perfect Elimination Ordering이 존재한다는 것이다.

이제 세 정의의 동치 관계를 증명한다.

**Def1 -> Def2:** Chord가 없는 길이 4 이상의 단순 사이클 $v_1, v_2, \ldots, v_k$ 가 존재한다고 가정하자. $a_i$ 를, $X_{v_i} \cap X_{v_{i + 1}}$ 에 속하는 $T$ 위의 정점이라고 하자 (notation abuse로 $k + 1 = 1$ 임에 양해 부탁). 또한, $p_i$ 를 $a_i$ 와 $a_{i + 1}$ 을 잇는 $T$ 위의 경로라고 하자. $p_i \subseteq X_{v_{i + 1}}$ 이기 때문에, $p_i \cap p_j \neq \emptyset$ 이기 위해서는 $|i - j| \equiv 1 \mod k$ 여야 한다. 만약 $p_i \cap p_{i + 1}$ 이 단일 정점이 아니라면, 둘이 갈라지는 위치까지 $a_{i+1}$ 를 당겨오면서 단일 정점이 되도록 만들어주자 (경로의 길이가 줄며, 위에서 가정한 성질이 모두 보존되기 때문에 일반성을 잃지 않고 할 수 있음). 최종적으로 $p_i \cap p_{i + 1} = a_{i+1}$ 이 만족하게 할 수 있고, 이 경우 $\cup p_i$ 는 단순 사이클을 이룬다. $T$ 는 트리이기 때문에 가정에 모순이다.

**Def2 -> Def3:** 어떠한 정점 $v$가 *simplicial* 하다는 것은 $v$ 에 인접한 정점들이 클리크를 이룬다는 것이다. 먼저 다음 Lemma를 증명한다.

**Lemma.** 모든 Chordal Graph(Def2) 가 simplicial한 정점을 가지고 있으며, $G$ 가 클리크가 아니라면, 두 인접하지 않은 simplicial한 정점이 존재한다.
**Proof.** $G$ 를 위 명제가 성립하지 않으며 정점 개수를 최소화하는 반례라고 하자. $G$가 클리크면 반례가 아니니, $G$ 에는 인접하지 않은 두 정점 $a, b$ 가 존재한다. $S \subseteq V - \{a, b\}$ 를, $V - S$ 상에서 $a, b$ 가 같은 컴포넌트에 속하지 않는 최소 크기의 부분집합이라고 하고, $G_a, G_b$ 를 $V - S$ 에서 $a, b$ 가 속하는 연결 컴포넌트라고 하자. 먼저 $S$ 가 클리크임을 관찰할 수 있다. $G$ 는 최소 반례이기 때문에 연결 그래프이다. 만약 $S$ 에 인접하지 않은 두 정점 $x, y$ 가 존재한다고 하자. 먼저, $x$와 $y$는 모두 $G_a, G_b$ 각각에 인접한 간선이 하나 존재한다: 만약 그렇지 않다면, $S$ 에서 $x$ 나 $y$ 를 빼도 무방하여 가정에 모순이다. 그렇다면, $x$ 에서 $G_a$ 내부를 최단경로로 거쳐 $y$ 로 가는 경로와, $G_b$ 내부를 최단경로로 거쳐 $y$ 로 가는 경로의 합집합을 취해줄 수 있다. 일단 각 경로 자체는 최단 경로기에 chord가 없으며, 두 경로 사이에 chord가 존재한다면 둘이 연결이 아니라는 가정에 모순이기 때문이다. 최종적으로, 이 사이클의 길이는 4 이상임이 자명하다.

Lemma에 의해 Def2를 가정할 경우 simplicial한 정점 $v$ 가 존재한다. 이제 귀납법으로 $G - \{v\}$ 에 대해 PEO를 구해준 후 $v$ 를 맨 앞에 배치하면 된다.

**Def3 -> Def1:** PEO의 suffix에 대해 귀납법을 사용한다. Suffix의 길이가 1일 경우, $T$ 를 정점 하나의 그래프라 하고, $X_{v_{|V|}}$ 를 그 하나의 정점으로 두자. 2 이상일 경우, $v_i$ 에 인접한 정점 $adj(v_i)$ 들은 클리크를 이룬다. 정점 $c \in V_T$ 에 대해 $\min_{w \in X_j} dist_T(w, c) = 0$ 일 경우 $c \in X_j$ 이다. $T$ 위의 정점 $c \in V_T$ 를 $\sum_{j \in adj(v_i)} \min_{w \in X_j} dist_T(w, c)$ 를 최소화 하는 정점이라 하자. $c$ 를 제거할 경우 $T$ 는 여러 서브트리로 나뉜다. 최소화된 값이 0이 아닐 경우, $c \notin X_j$ 인 $X_j$ 는 하나의 서브트리에만 존재한다. $c \in X_j$ 인 $X_j$ 도, $c \notin X_j$ 인 집합과 교집합이 있어야 하기 때문에, 해당 서브트리 방향에 인접한 정점을 포함한다. $c$ 를 이 인접한 정점으로 옮기면 strictly small한 값을 얻을 수 있기 때문에 가정에 모순이고, 고로 $c$ 는 모든 $X_j$ 에 속한다. $T$ 에 새로운 정점 $d$ 와, 간선 $(c, d)$ 를 만드는데, $v_i$ 및 그에 인접한 정점의 $X$ 에 $d$ 를 추가해 준다. Reduction의 사이즈가 선형임을 ($p = |V|$) 관찰하면 좋다.

**Remark.** Perfect Elimination Ordering을 쉽게 설명하자면, Chordal Graph의 Tree Decomposition을 이루는 트리에서 리프를 하나씩 떼는 것이라고 상상할 수 있다. Def2 -> Def3 의 증명을 보면 알겠지만, Perfect Elimination Ordering은 Simplicial vertex를 반복적으로 제거하는 것으로 다항 시간에 계산할 수 있다. 이를 최적화하면, 선형 시간에도 계산할 수 있다. 이에 대해서는 [이 글](https://www.secmem.org/blog/2019/03/10/Finding-perfect-elimination-ordering-in-choral-graphs/) 을 참고하라.

## 2. Tree Decomposition 구하기
Tree Decomposition을 정의했으니 이제 이를 계산하는 알고리즘에 대해 논의한다. 사실 이 글에서는 Treewidth가 2 이하인 그래프에 대해서 Tree Decomposition을 구하는 방법만 서술할 것이고, 일반적인 케이스에 대해서는 언급하지 않을 것이다. 지금까지 내가 PS에서 봤던, Tree decomposition을 요구하는 *거의* 모든 인스턴스의 경우 Treewidth가 2 이하였기 때문이다. (3인 경우도 있었고, 그래프의 성질을 ad-hoc하게 분석해서 tree decomposition을 해야 했었다.) 아래에 일반적인 Tree Decomposition에 대한 설명을 조금 할 텐데, 관심이 있으면 읽어보면 되겠다.

그래프가 주어졌을 때 최적의 Tree Decomposition을 구하는 것은 NP-hard 문제로, 다항 시간에 불가능하다. 심지어는, Treewidth를 Constant-factor로 Approximate하는 것 역시 [NP-hard일 것으로 추측되고 있다](https://arxiv.org/abs/1109.4910). 고로 일반적으로, Approximation도 하면서 $k$ 에 fixed-parameter tractable하기까지 한 알고리즘을 사용해야 한다.

[위키백과](https://en.wikipedia.org/wiki/Treewidth#Computing_the_treewidth)에 이에 대한 Survey가 나와 있는데, 이 중에서 의미있는 결과를 나열하자면 다음과 같다. Treewidth가 $k$ 인 그래프에 대해서

* $O(k \times \sqrt{\log k})$ 크기의 Tree Decomposition을 $n^{O(1)}$ 시간에 찾을 수 있다.
* $2k + 1$ 크기의 Tree Decomposition을 $2^{O(k)} O(n)$ 시간에 찾을 수 있다.
* $k$ 크기의 Tree Decomposition을 $2^{O(k^3)} O(n)$ 시간에 찾을 수 있다.

사실 나도 Tree decomposition 알고리즘을 몇 개 읽어봤는데. 내가 읽어본 것들은 Big-O notation 안에 붙어 있는 상수가 모두 경시대회에서 사용이 불가능한 수준으로 이해하고 있다. 이론적인 부분을 넘은, 범용적으로 사용 가능한 Tree decomposition 구현체에 대해 관심이 있다면 [PACE 2017 Challenge](https://pacechallenge.org/past/) 같은 시도를 살펴보면 좋을 것 같다.

## 2.1. Tree Decomposition on width 2
Width가 2 이하인 그래프는 Tree Decomposition에 대해서 상상하기 쉬운 편이라 알고리즘을 고안하기 용이하다. 편의상, 그래프가 Treewidth가 2인 Chordal Graph라고 생각하자. 이 그래프의 PEO (Perfect Elimination Ordering) 을 구하는 과정은 위에서 말했듯이 simplicial vertex를 제거하는 것이다. 즉, 다음 둘 중 한 연산이 될 것이다:

* 그래프에서 차수가 1인 정점을 제거한다
* 그래프에서 차수가 2이고, 연결된 두 정점 간에 간선이 있는 (즉, 클리크인) 정점을 제거한다

과정을 끝낸 이후 정확히 하나의 정점이 남았다면, Tree Decomposition이 존재하는 것이다.

이 과정에서 Tree Decomposition을 같이 만드는 방법을 생각해 보자. 위 PEO 과정을 관찰하면 그래프가 일종의 **삼각형들로 이루어진 트리** 라고 생각할 수 있고, 트리의 각 정점이 *삼각형* 이라는 가정 하에 리프부터 차근차근 떼면 된다.

먼저, Tree Decomposition의 Bag ($X_i$) 를 이루는 정점들은 각 단계에서 지운 간선들에서 유추한다. 즉, 차수 1 정점을 제거했을 때는 제거한 정점에 인접한 간선이 Bag이 되고, 차수 2 정점을 제거했을 때는 제거한 정점을 포함하는 *삼각형* 이 Bag이 된다. Bag $X_i$ 를, 위 과정에서 $i$ 번 정점을 지웠을 때 만들게 되는 Bag라고 정의하자 (최후에 남는 하나의 정점은, 자신만을 포함하는 크기 1의 정점 집합이 Bag가 된다) 최후에 정점이 1개 남았을 때를 제외하고, 모든 연산은 *부모 삼각형* 을 포함하고 있다: 예를 들어, 차수가 1인 정점 $u$와 간선 $(u, v)$ 를 제거할 때는, $X_u$ 의 부모가 $X_v$ 라고 하면 되고, 차수가 2인 정점 $u$ 를 제거할 때는, $u$ 를 포함하는 삼각형 상에서, $u$ 를 포함하는 간선이 지워지는 시점의 $X_j$ 를 부모로 두면 된다. (이 간선이 지워지는 시점은, $u$ 의 인접한 정점이 $v_1, v_2$ 라고 했을 때, 둘 중 처음 지워지는 시점이 될 것이다.) 이렇게 했을 때
 * Bag의 합집합은 전체 집합을 이루며 (자명)
 * 각 간선에 대해서 간선의 양끝을 모두 포함하는 Bag이 존재함을 알 수 있고 (자명)
 * 특정 정점이 속하는 Bag는 연결된 서브트리를 이룬다. Tree Decomposition은 최후에 남은 정점을 루트로 하는 크기 $|V|$ 의 Rooted tree일 것이다. 여기서 $u$ 를 포함하는 Bag는, $X_u$ 를 루트로 한 연결된 서브트리를 이루는데, 이는 Tree Decomposition에 대한 귀납법을 사용하여 증명할 수 있다. 정확히는, $X_i$ 를 만들 때, 모든 $v \in X_i, i \neq v$ 에 대해서, 부모 Bag 역시 $v$ 를 포함함을 증명하면 된다.

실제 문제에서는 Chordal Graph가 아니기 때문에, 연결된 두 정점 간에 간선이 없을 수 있다. 이러한 일이 있을 경우, 가상 간선을 만들어서 매 단계에 항상 클리크가 되도록 해 주어야 한다 (chordal completion이 꼭 필요한 시점에서 항상 이를 진행해 준다고 볼 수 있다). 이 점만 조심하여 구현하면, 실제로 그래프가 Chordal 인지 아닌지는 크게 상관이 없다.

이상의 내용을 구현한 나의 코드는 [Github](https://github.com/koosaga/olympiad/blob/master/Library/codes/graph_etc/tree_decomposition_width_2.cpp) 에 있다.
### 2.1.1. 연습 및 응용 문제
* [Yosupo Judge: Tree Decomposition (width 2)](https://judge.yosupo.jp/problem/tree_decomposition_width_2)
* [KAIST 2018 가을대회: Electronic Circuit](https://www.acmicpc.net/problem/16183)
* SCPC 2018 예선: 우주정거장
* SCPC 2018 본선: 우주정거장2

## 3. Tree Decomposition 을 이용한 최단 경로 쿼리
Tree Decomposition이 주어졌으니, 이를 사용해서 어떠한 문제를 해결할 수 있는지 살펴보자. 이 글에서는 가장 대표적인 예시로 *최단 경로 쿼리* 문제를 다룬다. 일반적인 트리에서 두 정점을 잇는 경로는 유일하고, LCA를 빠른 시간에 계산하면 쿼리당 $O(\log n)$ 에 해결할 수 있음이 잘 알려져 있다. Tree decomposition이 있는 그래프에서도, 아래의 방법을 사용하면 $O(\log n)$ 시간에 최단 경로 쿼리를 해결할 수 있다.

Bounded treewidth에서 최단 경로를 구하는 문제는 여러 가지 변형된 상황에서 이미 많은 대회에 출제되었다. 대부분 그래프가 *트리와 비슷한* 형태를 띄는 응용 문제로, 출제 의도는 그래프의 특수한 성질을 관찰해야 해결할 수 있게끔 되어 있다. Tree decomposition을 사용하면 이러한 특수한 성질을 자동적으로 추상화할 수 있기 때문에, 문제에 대한 성질 관찰이 필요없이 Black-box solver를 사용하는 것만으로 문제를 해결하기 충분하다.

$G$ 의 Tree decomposition을 $T, X_1, X_2, \ldots, X_p$ 라고 하자. 다음과 같은, 총 정점이 $\sum |X_i|$ 개인 그래프 $G_T$ 를 구성한다:
* 각 Bag의 원소 $X_{i, j}$ 에 대해 정점을 만든다.
* 같은 Bag의 서로 다른 두 원소 $X_{i, a}, X_{i, b}$ 에 대해, $X_{i, a} \rightarrow X_{i, b}$ 로 가는 방향 간선이 원래 그래프에 있었다면, 새로운 그래프에도 같은 가중치로 추가한다.
* 간선으로 인접한 두 Bag의 같은 원소 $X_{i, a} = X_{j, b}$ 에 대해 ($(i, j) \in T$), 가중치 $0$ 의 양방향 간선으로 이어준다.

$G_T$ 에서 $v$ 에 대응되는 정점들은 모두 $0$ 의 가중치 간선으로 연결되어 있다. Tree decomposition에서 $v$ 를 포함하는 Bag들은 연결된 서브트리를 이루기 때문이다. 고로, $G_T$ 에서 임의의 $s$ 에 대응되는 정점과, 임의의 $e$ 에 대응되는 정점 간에 최단 경로를 찾으면, 이는 원래 그래프에서의 $s, e$ 최단 경로와 동일하다는 것을 알 수 있다. 그래프를 *트리와 비슷하게* 바꿔준다는 것이 대략 이러한 느낌이다.

이제 트리와 비슷한 그래프를 얻었으니, 트리와 비슷한 방식으로 최단 경로를 구해보자. 트리에서 일반적으로 최단 경로를 찾는 것은, LCA를 구한 후 각 정점에서 LCA로 가는 경로의 길이를 구하는 것이다. 하지만 Tree decomposition에서는 사정이 조금 더 복잡하다. 같은 Bag 상에 있는 두 정점을 오갈 때, Bag에 있는 간선을 사용하는 대신 서브트리나 부모 쪽으로 돌아서 가는 것이 더 이득일 수도 있다. 즉, 두 정점이 심지어 같은 Bag에 있어도 문제를 해결하기 위해서는 트리 전체의 정보가 필요하다는 것이다.

여기서는 이 문제를 해결하기 위해 조금 더 강력한 도구인 Centroid decomposition을 사용한다. Tree decomposition의 Base가 되는 트리를 기준으로 Centroid decomposition을 하면, 두 Bag $a, b$ 에 대해서 $a$ 와 $b$ 를 포함하는 가장 낮은 Centroid tree $c$ 를 구하고, $a \rightarrow c, c \rightarrow b$ 로 가는 경로를 합쳐주면 된다. 확실한 것은, Bag $a$ 에서 Bag $b$ 로 움직일 때 $c$ 에 있는 정점 중 하나는 거친다. 만약 최단 경로가 특정 정점 $v$ 를 거친다는 것이 확실하면, $v$ 에서 나가는 최단 경로 및 $v$로 오는 최단 경로를 모두 Dijkstra로 선형 시간에 계산하면 모든 경로 쿼리를 해결할 수 있다. 고로 다음과 같은 식으로, treewidth가 $k$ 인 그래프에서 $O(n k^3 \log^2 n)$ 시간에 Centroid decomposition을 구성할 수 있다:

* $T$ 의 Centroid를 $c$ 라고 한다.
* 모든 $X_{c, i}$ 에 대해서, $X_{c, i}$ 에서 나가는, 그리고 $X_{c, i}$ 로 들어오는 최단 경로를 $2k$ 번의 Dijkstra's algorithm으로 구해준다. (간선이 총 $nk^2$ 개이다.)
* 이후 Bag $c$ 를 제거하고, subproblem을 해결해 준다.

Bag $c$ 를 제거할 때 조심해야 하는 것은, 사실 subproblem들에 대해서도 Bag $c$ 를 거치는 경로가 충분히 있을 수 있다는 것이다. 만약에 subproblem에 있는 경로가 Bag $c$ 를 거치는 것이 $X_{d, u_1} \rightarrow X_{c, t_1} \rightarrow \ldots \rightarrow X_{c, t_2} \rightarrow X_{d, u_2}$ 와 같다고 하자. $t_1 \neq t_2$ 일 수 밖에 없으며 (아니면 최단 경로가 단순하지 않음), $X_{d, u_1} = X_{c, t_1}, X_{d, u_2} = X_{c, t_2}$ 이다. 고로, $c$ 에서 사용한 Dijkstra를 통해서, Bag $d$ 에서 $u_1 \rightarrow u_2$ 로 가는 간선의 길이를 업데이트해주면 이 역시 해결이 된다. 각 쿼리는 이후 단순히 $O(k + \log n)$ 시간에 해결할 수 있다.

Dijkstra's algorithm을 사용하지 않고, Tree decomposition의 성질에 더 집중해서 알고리즘을 더 효율화하자. 만약 우리가, 하나의 Bag에 있는 서로 다른 두 노드 간의 *최단 경로* 를 알 수 있다면, 문제는 다음과 같은 Top-down DP를 사용하여 해결할 수 있다.
* $DP[v][t_1][t_2]$: $X_{c, t_1}$ 에서 $X_{v, t_2}$ 로 가는 최단 경로의 길이
* $RDP[v][t_1][t_2]$: $X_{v, t_2}$ 에서 $X_{c, t_1}$ 로 가는 최단 경로의 길이

상태 전이는, $X_{par(v), t_3} = X_{v, t_2}$ 인 $DP[par(v)][t_1][t_3]$ 에서 가져오거나, 아니면 같은 Bag에 있는 다른 노드에서 가져오는 식으로 진행하면 된다. 이러한 전이가 되기 위해서는 "동일한 Bag의 서로 다른 노드를 잇는 최단 경로가 잘 계산되었다" 라는 전제가 붙는데, 이 전제 자체는 달성하기 꽤 까다롭지만 한 가지 관찰을 통해서 비슷한 상황을 만들 수는 있다. Tree decomposition 상에서 $v_0 = c$ 가 Centroid이고, $v_1, v_2, v_3 \ldots, v_k$ 가 Centroid에서 자식 방향으로 내려가는 경로라고 하자. $X_{c, t}$ 에서 $X_{v_k, u}$ 로 내려가는 최단 경로를 관찰해 보면, 결국 최단 경로는
* $c$ 에서, 서브트리로 내려가서 방황하다가, 다시 $c$ 로 복귀 후 $v_1$ 로 내려감
* $v_1$ 에서, 부모 방향이 아닌 서브트리로 내려가서 방황하다가, 다시 $v_1$ 로 복귀 후 $v_2$ 로 내려감
* ...
* $v_k$ 에서, 부모 방향이 아닌 서브트리로 내려가서 방황하다가, 다시 $v_k$ 로 복귀 후 끝

여기서 *부모 방향이 아닌* 이라는 전제를 할 수 있는 이유는, 만약에 부모 방향으로 올라갔을 경우, 이는 위쪽에서 이미 계산한 방황이라고 묶어서 생각해 줄 수 있기 때문이다. 고로, 동일한 Bag의 서로 다른 노드를 잇는 최단 경로를 계산하는 대신 다음과 같은 값을 계산해 줄 수 있다.
* $inter[v][t_1][t_2]$: $X_{v, t_1}$ 에서, $v$ 의 서브트리에 있는 Bag들만을 거쳐, $X_{v, t_2}$ 로 가는 최단 경로의 길이

서브트리만 본다는 조건이 붙어있기 때문에, 이것은 Bottom-up Tree DP로 계산해 줄 수 있다. 이렇게, 트리 DP 두 번에 전체 문제를 해결하게 되면, 전처리 시간 복잡도가 $O(n k^4 \log n)$ 정도로 정리된다. $O(n k^3 \log n)$ 정도에도 해결을 할 수가 있는데, 본인이 짠 가장 빠른 코드의 시간 복잡도는 저렇게 나왔다. $k$ 가 아주 작기 때문에 $k$ 에 붙은 Exponent는 큰 의미 없고 cache miss같은 게 더 중요할 거 같다.

이상의 내용을 구현한 나의 코드는 [GitHub](https://github.com/koosaga/olympiad/blob/master/JOI/camp17_railway_trip_alternative.cpp) 에서 찾을 수 있다.

### 3.1 연습 문제
아래 문제들은 위 방법대로 그대로만 구현하면, 문제의 특수 조건이나 관찰을 *사용하지 않고* 지문에 주어진 그래프를 그대로 Solver에 먹여서 해결할 수 있다. 예를 들어, Station 문제의 경우 $l, r$ 배열의 단조성이 정해에 필요한데, 위 풀이를 사용하면 그런 조건을 전부 무시할 수 있다.

* [Izborne pripreme 2014: GRAD](https://www.acmicpc.net/problem/10064)
* [NEERC 2015: Distance on Triangulation](https://www.acmicpc.net/problem/11738)
* [JOI Spring Camp 2017: Railway Trip](https://www.acmicpc.net/problem/17697)
* [Ptz Winter 2022 Day6: Station](https://www.acmicpc.net/problem/24710)
* [UCPC 2019 예선: %](https://www.acmicpc.net/problem/17366)
* [Google Code Jam Round 2: Emacs++](https://codingcompetitions.withgoogle.com/codejam/round/000000000019ffb9/000000000033893b#problem)
