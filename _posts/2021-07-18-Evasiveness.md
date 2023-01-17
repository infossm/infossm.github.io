---
layout: post
title: "A Topological Approach to Evasiveness"
author: TAMREF
date: 2021-07-18 00:00
tags: [graph-theory, topology]
---



## 스무고개

간단한 게임을 하나 생각해 봅시다. A와 B가 게임을 하는데, 아직 결정되지 않은 $n$자리 이진수가 있습니다. A는 B에게 $i$번째 비트가 $1$인지 물어볼 수 있고, B는 이를 답해줍니다. 질문을 $n$번보다 적게 써서 이 수가 $3$의 배수인지 맞힐 수 있다면 A의 승리, 그렇지 않다면 B의 승리입니다. 물론 B는 특정 수를 정해놓고 시작해야 하는 게 아니기 때문에, 질문에 따라 얼마든지 마음속으로 답을 바꿀 수 있습니다. 누가 승리할까요? leading zero 등의 제한 조건은 없습니다.

당연하게도, 이 게임은 B의 손쉬운 승리입니다. 어떤 질문을 하든 계속 $0$을 부르다가 $n-1$자리가 채워질때까지도 A는 이 수가 $3$의 배수일지 아닐지 알 수 없습니다. 마지막 자리가 $0$인지 $1$인지에 따라 답이 갈리기 때문이죠. 이런 식의 스무고개 게임을 보다 일반적으로 나타낼 수 있습니다.

유한집합 $E$ ($n$개의 비트들)에 대해, 우리가 원하는 “성질” $P$는 $P \subset 2^{E}$로 나타낼 수 있습니다. 여기서는 3의 배수가 되는 이진수들이 되겠네요. 가령 $1001_{2} = 9$가 $3$의 배수이므로, $\lbrace 1, 4\rbrace \in P$입니다. 게임에서 사용하는 “결정되지 않은” 이진수는 $X \subseteq E$로 나타낼 수 있습니다. 적어도 A는 $X$에 대해서 아무것도 알지 못하는 것이죠. 어떤 비트가 켜져 있느냐는 질문은 A가 아무 $e \in E$를 골라 “$e \in X$이냐?”를 묻는 질문으로 생각할 수 있습니다. 결국 A의 목적은 $X \in P$인지 혹은 $X \notin P$인지 $\left\lvert E \right\rvert$번 미만의 질문을 써서 알아내는 것이고, B의 목적은 그를 막는 것이 될 것입니다.

그래프 이론에서도 이와 같은 질문에 관심이 많았습니다. $n$개의 정점이 주어져 있지만 어떤 간선이 이어져 있는지 모르는 그래프에 대해서, “이 간선이 존재하느냐?”라는 질문만으로 특정 graph property에 대한 단서를 잡아낼 수 있는지는 많은 사람들의 관심사였습니다.

**Exercise 1.** $\left\lvert P\right\rvert$가 홀수인 경우, 항상 B에게 필승전략이 있음을 보이세요. (Hint: 홀수는 둘로 쪼개도 홀수가 하나 남습니다.)

## Adjacency Query Complexity

$n$개의 정점이 주어진 단순그래프 $G$에 대해서, graph property $P$란 graph isomorphism (정점 renumbering)에 의해 바뀌지 않는 성질을 말합니다. “연결그래프이다.” “최소 차수가 3 이상이다” 등이 해당합니다. 정점 개수가 고정되어 있기 때문에 앞으로 $G$는 그냥 “간선 집합”으로 간주하고, $G$가 성질 $P$를 가지면 $G \in P$로 쓰도록 하겠습니다.

$G \in P$인지를 확실히 알아내기 위해 필요한 질문의 최소 횟수를 $c(P)$라고 쓰고, decision tree complexity, 또는 query complexity 등으로 부릅니다. 만약 무방향 그래프에서 $c(P) = n(n-1)/2$이라면, $P$는 “evasive”한 성질이라고 부릅니다. 마찬가지로 방향그래프에서 $c(P) = n(n-1)$이라면 역시 $P$는 evasive합니다. 그리고 쉽게 예상할 수 있듯, 정말 많은 property가 evasive합니다.

### Connectedness is evasive

$P$가 “connectedness”, 즉 "$G$가 연결그래프이다"를 나타낸다고 합시다. $P$가 evasive하다는 것은 [IOI 2014 Game](https://www.acmicpc.net/problem/10071) 문제로 출제된 바 있습니다. 가능한 B의 전략은 $i = 2, \cdots, n$에 대해 “부모 정점” $1 \le p_{i} < i$번 정점을 최대한 게으르게 결정해주는 것입니다. 즉, $i < j$에 대해 $(i, j)$ 간선이 있는지 묻는 질문에 계속 없다고 답하다가, 마지막 $j-1$번째로 들어오는 질문에 대해서만 간선이 있다고 답하면 됩니다. 이 경우 모든 질문에 대한 답이 처리되기 전까지 $G$는 connected가 아니고, 모든 질문이 들어온 뒤에야 connected가 되기 때문에 (그리고 connected가 아닐 가능성이 있기 때문에) evasive하게 됩니다.

```c++
#include "game.h"
#include <bits/stdc++.h>
using namespace std;

vector<int> cnt;
void initialize(int n) {
    cnt = vector<int>(n);
}

int hasEdge(int u, int v) {
    if(u > v) swap(u, v);
    return ++cnt[v] == v;
}
```



### Nonevasive Nontrivial property

가장 간단한 nonevasive property는 “간선이 0개 이상이다” 또는 “최대 차수가 $n$ 이상이다”와 같은 trivial한 property가 있습니다. $P = \emptyset$이거나 $P = 2^{E}$인 것인데, 이런 성질에는 아무도 관심이 없으므로 $P \neq \emptyset, 2^{E}$인 non-trivial property 중에서 nonevasive한 property를 찾아나서 봅시다. 하지만 역시 쉽지 않은 모양입니다..

**Conjecture (Rosenberg, 1973).** 모든 nontrivial graph property $P$에 대해 $c(P) = \Omega(n^2)$일 것이다.

다행히, 이 질문은 Aanderaa와 van Emde Boas에 의해 거짓인 것으로 밝혀졌습니다. 방향 그래프와 무방향 그래프에서 각각 반례가 발견된 것입니다.

- 방향그래프에서, $indeg(v) = n -1$이고 $outdeg(v) = 0$인 “sink”가 존재하는지를 $3n$번 이하의 쿼리로 알아낼 수 있다.
- 무방향그래프에서, 어떤 그래프가 **전갈스러운지** $O(n)$번의 쿼리로 알아낼 수 있다.

“sink”는 그렇다치고, 그래프가 “전갈스럽다”라는 건 대체 뭘까요? 전갈 그래프에는 가시, 꼬리, 몸통, 발의 4가지 정점이 있습니다:

- 가시 - 꼬리 - 몸통은 길이 $3$의 path를 이루고, 가시의 차수는 1, 꼬리의 차수는 2, 몸통의 차수는 $n - 2$.
- 즉, 몸통은 꼬리와 발들에 모두 연결되어 있으며, 나머지 “발”들의 연결 상태에는 제약이 없다.

전갈 그래프를 $5n$번 정도의 쿼리로 판별하는 방법은 [koosaga님의 블로그](https://koosaga.com/130) 등지에서 찾아볼 수 있습니다.

이런 반례들에도 불구하고 여전히 non-evasiveness를 보유한 “익숙한 성질”을 찾기란 어려웠습니다. 결국 Aanderaa, Rosenberg는 1975년에 강화한 추측을 내놓습니다.

**Conjecture (A-R, 1975)** 모든 nontrivial *monotone* graph property $P$에 대해 $c(P) = \Omega(n^2)$이다.

Karp는 아예 한술 더 떠서, 모든 monotone graph property가 evasive할 것이라는 추측을 내놓습니다. 이를 합쳐서 AKR conjecture라고 부릅니다.

여기서 어떤 property $P$가 monotone하다는 것은 $G \in P$이면 아무 $e$를 뺀 $G - e$도 $P$를 만족한다거나 (monotone decreasing), $G \in P$이면 아무 $e$를 추가한 $G + e$도 $P$를 만족하거나 (monotone increasing) 둘 중 하나가 성립하는 경우를 말합니다. 당연하지만, monotone decreasing인 동시에 increasing이면 trivial합니다. $P$가 monotone decreasing이면 $E - P = \lbrace E - G \mid G \in P\rbrace$가 monotone increasing이기 때문에, 앞으로는 monotone decreasing (e.g. “disconnected”, “최대 차수가 $2$이하”) 인 경우만 고려하기로 합니다.

A-R conjecture는 Rivest(1978)에 의해 처음으로 해결되는데, nontrivial monotone property $P$에 대해 $c(P) > \frac{1}{16}n^{2} + o(n^2)$가 성립하는 것을 증명했습니다. 이후 Kleitman(1980)에 의해 상수가 $1/9$로 바뀌었지만, evasiveness에 다가가기에는 요원한 상황이었습니다.

이 글에서는 대수위상적 방법론을 도입하여 $n = p^{k}$ 꼴에 대한 evasiveness를 증명하고, 상수를 $1/4$로 끌어올린 Kahn, Saks and Sturtevant (1983)의 “A Topological Approach to Evasiveness”를 살펴봅니다. 비록 상수는 Scheidweiler(2012)에 의해 $1/3$으로 추후 개선이 있었으나, 이 논문은 처음으로 위상적 방법론을 도입한데다, 특정 $n$에 대한 evasiveness를 증명했다는 점에서 그 공헌이 지대하다고 말할 수 있습니다.

### Preliminaries

- Simplicial complex와 그 homology
- Group theory
- Homotopy, Homeomorphism 등 위상수학 “단어들”

대수위상은 학부 고학년이나 대학원 과목에서 처음 만나게 될 정도로 난도가 있는 분야이고, 저도 그 분야를 제대로 알지 못합니다. 다만 우리가 주로 다루고자 하는 simplicial complex의 경우, 선형대수학/학부 대수학 정도의 지식으로 필요한 내용을 엿볼 수 있습니다.

## Non-evasive complex

Monotone decreasing graph property는 자연히 simplicial complex $\Delta$를 이룹니다. Simplicial complex $\Delta$의 geometric realization을 $\lVert \Delta \rVert$로 쓰기로 합니다.

이 때, 다음과 같은 $\Delta$의 sub-complex들을 정의합니다.

- $\mathrm{lk}_{\Delta}(x) = \lbrace A - x \mid A \in \Delta\rbrace$
- $\mathrm{del}_{\Delta}(x) = \lbrace A \in \Delta \mid x \notin A \rbrace$

$\Delta$가 1차원 simplicial complex인 그래프라면, 각각은 $x$의 contraction과 deletion으로 생각할 수 있습니다. deletion-contraction이 귀납법에 유용하게 사용되는 만큼, non-evasiveness argument를 위해 두 집합을 귀납법에 적극적으로 활용해봅시다.

한편, $F \in \Delta$에 대해 $F \subsetneq G \in \Delta$를 만족하는 maximal한 $G \in \Delta$가 유일하게 존재한다면, $F$를 $\Delta$의 free face라고 부릅니다. 그래프의 “리프”에 대응되는 개념으로 생각할 수 있습니다. 모든 간선이 maximal face인 그래프에서, 실제로 free face는 리프 정점들 뿐입니다.

$\Delta$에서 free face $x$와 그것을 포함하는 유일한 maximal face를 함께 제거하는 과정을 $\Delta$의 “Collapse”라고 부릅니다. 그래프에서 리프가 제거되는 과정을 생각해도 좋은데, 리프를 반대편 정점쪽으로 “서서히 밀어서 붙인다”고 생각하면 좋습니다. 2차원 simplex의 한 변을 collapse시키면 $\Lambda$모양의 1차원 simplicial complex가 됩니다. 즉, collapse는 geometric realization 아래에서 “연속적인 변형”과 같다는 데 주목하세요.

만약 collapse를 지속해서 $\Delta$를 한 점으로 만들 수 있다면 $\Delta$를 collapsible하다고 합니다. 실제로 사이클이 있는 그래프의 경우 collapsible하지 않지만, 트리의 경우는 collapsible하다는 것을 확인할 수 있습니다. 연속적인 변형을 가해서 점으로 만들 수 있는만큼, collapsible complex는 “재미없는(trivial)” 위상적 성질을 가집니다.

**Fact.** Collapsible $\implies$ Contractible $\implies$ $\mathbb{Z}$-acyclic $\implies$ $\mathbb{Z}_{p}$-acyclic and $\chi(\Delta) = 1$.

$\Delta$의 Euler characterestic $\chi(\Delta)$는 $\sum_{F \in \Delta, F \ne \emptyset} (-1)^{\dim F}$로 정의됩니다. 어떤 점에 대해서 $\chi(\mathrm{pt}) = 1$이고, 위상동형인 두 complex에 대해서 보존되는 값이기 때문에 모든 contractible한 complex는 $\chi = 1$을 갖게 됩니다. $\dim F \le 2$인 다면체(polytope)의 경우, 그 유명한 $v - e + f$가 euler characteristic이 됩니다.

Non-evasiveness는 스무고개 게임에서 나왔고, Collapsibility는 위상적인 성질입니다. 그런데 이 둘이 관련있다는 다음의 사실은 언뜻 놀랍습니다.

**Proposition 1.** 모든 non-evasive complex $\Delta$는 collapsible하다.

*Proof.* $\Delta$가 trivial한 경우는 자명합니다. 만약 $\Delta$가 nontrivial하다면, “good first question” $x \in E$가 존재합니다. 즉, $n = \lvert E \rvert$회 미만의 질문으로 답을 얻는 전략의 첫 질문이 존재해야만 합니다. 만약 이 질문에 대한 답이 “yes”, 즉 $x \in X$이라면 우리는 $X - x \in \mathrm{lk}_{\Delta}(x)$임을, “no”라면 $X \in \mathrm{del}_{\Delta}(x)$임을 알 수 있습니다. 두 complex는 모두 non-evasive하므로, 적당한 귀납가정을 통해 $\mathrm{lk}_{\Delta}(x), \mathrm{del}_{\Delta}(x)$가 모두 collapsible하다고 가정할 수 있습니다.

$\mathrm{lk}$의 collapse에 사용되는 free face를 $L_{1}, \cdots, L_{k}$, $\mathrm{del}$의 collapse에 사용되는 free face들을 역시 차례대로 $D_{1}, \cdots, D_{l}$이라고 둡시다. 그러면 $L_{i} + x$가 $\Delta$의 free face가 됨을 알 수 있고, $L_{1} + x, \cdots, L_{k} + x$를 collapse시키면 $\mathrm{del}$이 된다는 사실을 알 수 있습니다. 이로부터 $\Delta$가 collapsible임을 증명했습니다. $\square$

이 사실은 단순히 graph property가 아닌, 모든 monotone evasiveness problem에 적용됩니다. 다시 말해, 우리는 아직 “graph property”라는 단어가 가진 힘을 사용하지 않았습니다.

## Graph isomorphism – Transitive group action

Monotone *graph* property를 나타내는 complex $\Delta$는 한 가지 특징이 있는데, graph isomorphism $\varphi$에 대해서 $\varphi\Delta = \Delta$라는 점입니다. 이는 $\varphi$를 $E$의 순열로 봤을 때, 자연스럽게 $\Delta$에 act시킬 수 있기 때문입니다. 즉 모든 graph isomorphism $\varphi$가 $\mathrm{Aut}(\Delta)$의 원소이고, 이 graph isomorphism들은 transitive합니다; 다시 말해, 모든 $e \neq f \in E$에 대해 $\varphi(e) = f$인 적당한 $\varphi$가 있습니다. 이런 성질들로부터 여러 가지 추측들이 세워졌습니다.

**Conjecture (KSS83).** $\Delta$가 “재미없는 위상을 가진” simplicial complex이고, *$\mathrm{Aut}(\Delta)$가 transitively act한다고 하자*. (즉, orbit이 하나뿐이다) 그렇다면 $\Delta$는 trivial하다.

이탤릭체로 표기된부분이 나름 “graph property이다”라는 말의 encoding인 셈입니다. 하지만 이 추측을 증명하는 것은 녹록치가 않았고, 결국은 **특이한 조건**을 만족하는 $\Gamma \le \mathrm{Aut}(\Delta)$에 대해서만 conjecture를 증명하는 데 성공합니다.

**Definition (Oliver type)** $\Gamma$가 다음을 만족하면 oliver type이라고 하고, $\Gamma \in \mathcal{G}_{p, q}$로 쓴다.

- $\Gamma_{1} \unlhd \Gamma_{2} \unlhd \Gamma$인 $\Gamma_{1}, \Gamma_{2}$가 존재하고,
- $\Gamma_{2} / \Gamma_{1}$이 cyclic이며, $\Gamma_{1}$이 $p$-group이고, $\Gamma_{2}$가 $q$-group이다. $p$는 prime, $q$는 1 또는 prime, $p = q$일 수 있다.

**Example.**

$GF(p^{\alpha})$에 act하는 group $\Gamma$를 $\Gamma = \lbrace x \mapsto ax + b \mid a, b \in GF(p^{\alpha}); a \neq 0 \rbrace$라고 두면, $\Gamma$는 좋은 “정점 renumbering”이 됩니다; graph isomorphism으로 자연스럽게 확장할 수 있습니다. 한편 $\Gamma$의 normal subgroup $\Gamma_{1} = \lbrace x \mapsto x + b \mid b \in GF(p^{\alpha})\rbrace$를 찾을 수 있고, Quotient group $\Gamma / \Gamma_{1}$은 $a$에 따라 좌우되므로 $GF(p^{\alpha})^{\times}$와 isomorphic합니다. 따라서 $\Gamma \in \mathcal{G}_{p, 1}$이 됩니다.

**Theorem 2(KSS83)** $\Delta$가 nonempty $\mathbb{Z}$-acyclic complex라고 하고, Oliver-type group $\Gamma \le \mathrm{Aut}(\Delta)$가 vertex-transitive하게 act한다고 하자. 이 때 $\Delta$는 trivial하다. (i.e. simplex이다)

**Corollary.** $\Gamma$를 Example의 group으로 설정하면, **Proposition 1**과 결합하여 $n = p^{\alpha}$개의 정점을 가진 nontrivial monotone graph property는 모두 evasive함을 보일 수 있다.

- $\Delta$가 nonevasive $\implies$ $\Delta$가 $\mathbb{Z}$-acyclic 이고, $n$때문에 $\mathrm{Aut}(\Delta)$에 $\Gamma$가 부분군으로 있음 $\implies$ Theorem으로 인해 $\Delta$는 trivial.

## Proof sketch of Theorem 2 – Fixed point complex

**Theorem 2**의 증명에 사용되는 argument는 다소 난이도가 있으나, 특기할 만한 아이디어를 사용합니다. 바로 $\Delta$에서 “$\Gamma$-fixed point”, 즉, $\Delta^{\Gamma}$를 찾는 것입니다.

만약 어떤 face $A \in \Delta$가 모든 $\Gamma$-fixed point라고 합시다. $\Gamma$는 transitive하므로, 사실상 모든 간선이 $A$에 포함되어 있어야 하고, 즉 $A = E$여야 합니다. 이는 곧 $\Delta$가 simplex임을 의미하게 됩니다. 이제 $\Gamma$가 transitively act한다는 사실을 잠시 잊어버리고, fixed-point complex $\Delta^{\Gamma}$가 공집합이 아니라는 사실을 “$\Gamma$가 oliver type이라는 사실만으로” 이끌어내어보도록 합시다.

$\Delta$의 geometric realization에서, $\Gamma$에 대한 fixed point들의 집합 $\lVert \Delta \rVert^{\Gamma}$ 또한 하나의 complex를 이룹니다. 이는 다음과 같이 abstract하게 만들어진 fixed-point complex의 realization과 **homotopically equivalent**합니다.

- $\Delta^{\Gamma}$의 vertex는 $E$의 $\Gamma$-orbits. 즉, 각 vertex는 $\Gamma$-invariant face들이 된다.
- $\lbrace A_{1}, \cdots, A_{k} \rbrace \in \Delta^{\Gamma} \iff A_{1} \cup \cdots \cup A_{k} \in \Delta$.

이 때, $\Delta^{\Gamma}$의 각 꼭짓점을 $A_{i}$의 무게중심으로 두면 $\lVert \Delta \rVert^{\Gamma}$와 $\lVert \Delta^{\Gamma} \rVert$의 동일성을 확인할 수 있습니다.

위상공간에 작용하는 group action에 대한 fixed point는 그 자체로 대수위상의 주요한 주제였고, 때문에 다음과 같은 훌륭한 결과들이 많이 나와 있습니다. 이들은 더 이상 simplicial complex 위주의 언어로 기술되어 있지 않은데, 설명 가능한 증명을 알게 되면 추후 추가하도록 하겠습니다.

**Theorem (Smith, 1941)**: $\Delta$가 $\mathbb{Z}_{p}$-acyclic하기만 하고, $\Gamma$가 그저 $p$-group이라고 하자. $\Delta^{\Gamma} \neq \emptyset$이고, 그 또한 $\mathbb{Z}_{p}$-acyclic이 된다.

**Theorem (Oliver, 1975)**: $\Delta$가 $\mathbb{Z}_{p}$-acyclic이고, $\Gamma \in \mathcal{G}_{p, 1}$이라고 하자. $\chi(\Delta^{\Gamma}) = 1$이다; 즉, $\Delta^{\Gamma} \neq \emptyset$.

이 두 결과만으로 사실 Corollary를 유도할 수 있습니다. 하지만 보다 일반적인 결론을 얻고자 한다면;

**Theorem (Oliver, 1975)**:  $\Delta$가 $\mathbb{Z}_{p}$-acyclic이고, $\Gamma \in \mathcal{G}_{p, q}$이라고 하자.  $ \Delta^{\Gamma} \neq \emptyset$.

위 결과로부터 Theorem 2를 증명할 수 있습니다.

### $n \neq p^{\alpha}$인 경우

가능한 모든 nontrivial monotone property에 대해 $c(P)$의 최댓값을 $c_{n}$이라고 둡시다. 다음이 알려져 있습니다:

**Theorem (Kleitmann and Kwiatkowski, 1980)**: $c_{n} \ge \min(c_{n-1}, y(n-y))$. 단, $y$는 $n/2$에 가장 가까운 prime power.

위 사실의 증명도 짧지 않기 때문에 넘어가도록 하겠습니다. $n/2 - y = o(n)$임을 감안하면, $c_{n} \ge n^{2} / 4 + o(n^2)$를 일반적인 경우에 대해 얻습니다.

## 마무리

AKR conjecture는 계속해서 연구되고 있는 주제이고, 다양한 파생 주제들도 등장하고 있습니다. 특정 알고리즘 기법을 심도 있게 다루는 주제는 아니지만, 이 논문의 시도와 같이 algorithmic lower bound의 증명에 위상수학이 활용되는 경우가 점점 늘어나고 있습니다. 특히 non-evasive monotone complex가 trivial한 위상을 갖는다는 점은 Scheidweiler(2012)에서도 “breakthrough”라고 불리며 그 가치를 인정받고 있습니다. 부족한 글이지만 KSS83의 유려한 정신이 독자 여러분께 조금이나마 전달되었으면 하는 마음입니다.



## Reference

KSS83: *Kahn, Jeff, Saks, Michael, and Sturtevant, Dean. "A Topological Approach to Evasiveness." Combinatorica (Budapest. 1981) 4.4 (1984): 297-306. Web.*

Scheidweiler12: *Scheidweiler, Robert, and Triesch, Eberhard. "A Lower Bound for the Complexity of Monotone Graph Properties." SIAM Journal on Discrete Mathematics 27.1 (2013): 257-65. Web.*

KK80: *Kleitman, D.J, and Kwiatkowski, D.J. "Further Results on the Aanderaa-Rosenberg Conjecture." Journal of Combinatorial Theory. Series B 28.1 (1980): 85-95. Web.*

Oliver75: *Oliver, Robert. "Fixed-Point Sets of Group Actions on Finite Acyclic Complexes." Commentarii Mathematici Helvetici 50 (1975): 155. Web.*

