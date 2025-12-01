---
layout: post
title: "F-deletion 문제의 FPT 알고리즘"
date: 2025-12-01
author: leejseo
tags: [algorithm, graph theory]
---

## Introduction

NP-hard로 분류되는 여러 그래프 관련 문제들을 아래와 같이 *다른 관점*에서 살펴보자.

1. Vertex Cover

   Vertex Cover란 모든 간선의 최소 하나의 끝점을 포함하도록 하는 최소 크기의 vertex subset(minimum size vertex cover)을 찾는 문제이다. 이는, 다르게 생각하면 그래프에서 최소 개수의 정점을 제거하여, 그래프의 간선이 없도록 만드는 문제와 같다.

   즉, $G$가 $\newcommand{\F}{\mathcal{F}}\mathcal{F} = \{K_2\}$ 를 minor로 포함하지 않도록 하는 것과 동치이다.

2. Feedback Vertex Set

   Feedback Vertex Set 문제는 그래프에서 최소 개수의 정점을 제거하여 cycle을 포함하지 않도록 하는 문제이다.

   모든 cycle은 $C_3$를 minor로 가지고, $C_3$를 minor로 가지는 모든 그래프는 cycle이 있으므로, 이는 $G$가 $\mathcal{F}= \{C_3\}$ 를 minor로 포함하지 않도록 하는 것과 동치이다.

3. Planarization

   Planarization 문제는 그래프를 평면 그래프로 만들기 위해 최소 개수의 정점을 제거하는 문제이다.

   (Kuratowski) Wagner의 정리를 생각해보면, non-planar if and only if "$K_5$ 혹은 $K_{3,3}$를 (topological) minor로 가짐" 이므로, 이는 $G$가 $\mathcal{F} = \{K_5, K_{3, 3}\}$ 을 (topological) minor로 가지지 않도록 하는 것과 동치이다.

이렇듯, 많은 계산적으로 난해한 문제들이 $G$에서 최소 개수의 정점을 제거하여 어떤 graph collection의 원소를 minor 혹은 topological minor로 가지지 않게 하는 것으로 formulation 되는 것을 볼 수 있었다. 그래서 (많은 수의 문제를 일반화하는) 다음 두 문제를 생각해볼 수 있다.

**$\mathcal{F}$-M-DELETION**: 그래프 $G$와 유한개의 그래프를 모아 놓은 $\mathcal{F}$가 주어졌을 때 $G-S$가 $\F$의 원소를 minor로 포함하지 않도록 하는 $\vert S \vert \le k$가 존재하는가?

**$\F$-TM-DELETION**: 그래프 $G$와 유한개의 그래프를 모아 놓은 $\F$가 주어졌을 때 $G-S$가 $\F$의 원소를 topological minor로 포함하지 않도록 하는 $\vert S \vert \le k$가 존재하는가?

앞서 살펴본 그래프 관련 NP-hard 문제들이 treewidth 등에 기반한 FPT 알고리즘이 알려져있는 만큼, 새로이 제시한 두 문제에 대해서도 FPT 알고리즘을 생각해보는 것이 자연스러울 것이다.

> Courcelle's Theorem. MSO logic으로 표현 가능한 모든 그래프 성질은 어떤 함수 $f$가 존재하여 treewidth $\le k$인 그래프에서 $f(k) \cdot n$ 시간에 decidable 하다.

$\F$-M-DELETION과 $\F$-TM-DELETION 모두 Courcelle's theorem에 의해 treewidth가 bounded인 그래프에서 FPT 알고리즘이 존재한다는 사실은 알 수 있다. 하지만, Courcelle's theorem이 우리에게 주는 알고리즘의 "$f$" 함수는 아주 빠르게 증가하는 함수로 (사실상 $2^{2^{2^{\cdots^k}}}$ 정도의 scale 이다) 아주 promising한 결과는 아니라고 할 수 있겠다.

즉, treewidth $\le k$인 그래프에서 $f(k) \cdot n^{O(1)}$ 시간에 해결 가능한 "최소의" $f$를 찾는 것은 상당히 흥미로운 문제이며, 이 글에서는 J. Baste 및 2인에 의해 2020년 발표된 $2^{2^{O(k \log k)}} \cdot n^{O(1)}$ 시간에 동작하는 알고리즘을 소개한다. 개인적으로 알고리즘을 공부하며 detail을 확인할 필요가 있어 다소 글이 장황해진 면이 있으나, 마지막의 결론 부분을 읽으면 intuition 위주로 확인하기에 편할 것이라 생각한다.

## Minor와 Topological Minor

어떤 그래프 $H$가 그래프 $G$의 minor임은, $G$의 정점/간선의 삭제, 간선의 contraction 만으로 $H$를 만들 수 있음으로 정의된다. $H$가 $G$의 minor임을 $H \preceq_m G$ 로 표기하자.

$H$가 $G$의 topological minor임은, $H$의 subdivision(간선 중간에 점 찍는게 허용됨)이 $G$의 subgraph임으로 정의된다.  $H$가 $G$의 topological minor임을 $H \preceq_{tm} G$로 표기하자. 편의상, $\F$의 원소 중 하나를 $G$가 topological minor로 가짐을 $\F \preceq_{tm} G$ 로 표기하자. 그리고 $\F$를 topological minor로 가지지 않는 graph들의 집합을 $ex_{tm}(\F)$로 표기하자.

Topological minor가 minor임은 알려져 있으나, 그 역이 일반적으로 성립하지 않음 또한 알려져있다.

하지만, 어떤 그래프 $H$를 minor로 가지는 graph 중 topological minor relation에 대해 minimal 한 graph 들의 집합 $tpm(H)$를 생각해보자. $G$가 $H$를 minor로 가짐은 $G$가 $tpm(H)$의 어느 원소를 topological minor로 가짐과 동치이고, $tpm(H)$의 각 원소의 정점 집합의 크기는 $\vert V(H)\vert$에 대한 함수로  bound 됨이 알려져있다.

따라서, 우리는 $\F$-TM-DELETION 문제만 생각해도 충분하다. 왜냐하면, 앞서 한 관찰 때문에 $\F$-TM-DELETION 문제에 대한 FPT 알고리즘은 $\F$-M-DELETION 문제에 대한 FPT 알고리즘이기도 할 것이기 때문이다.

## Boundaried Graph와 Folio

음이 아닌 정수 $t$에 대해 $t$-boundaried graph $\newcommand{\G}{\mathbf{G}}\G = (G, R, \lambda)$를 다음과 같이 정의하자:

* $G$ 는 graph이다. (여기에서, $G$를 $\G$의 underlying graph라 부른다.)
* $R \subseteq V(G)$는 $\vert R \vert = t$ 를 만족한다. ($R$의 정점을 boundary vertex라 부른다.)
* $\lambda : R \to \mathbb{N}^+$ 는 단사 함수이다.

Boundaried graph를 정의하는 이유는, 나중에 boundaried graph 상에 relation을 잘 정의해 어떠한 equivalence class를 생각할 것이고, 이를 통해 시간 복잡도를 bound 할 것이기 때문이다. $t$-boundaried graph들의 collection을 $\newcommand{\B}{\mathcal{B}} \B^{(t)}$ 로 표기하자.

Boundaried graph의 index map을 $\psi_{\G}(v) = \vert \{u \in R \mid \lambda(u) \le \lambda(v) \} \vert$ 로 정의하자. 이는 $t$-boundaried graph의 $\lambda$의 공역을 사실상 $[1, t]$로 제한시켜 생각할 수 있도록 하는 효과를 준다. 그리고 이를 기반으로 두 $t$-boundaried graph 사이의 isomorphism을 정의할 수 있다.



**Definition.** 두 $t$-boundaried graph $\G_1 = (G_1, R_1, \lambda_1)$, $\G_2 = (G_2, R_2, \lambda_2)$가 isomorphic 함은 전단사함수 $\sigma: V(G_1) \to V(G_2)$가 존재해서

* $\sigma$가 $G_1, G_2$ 사이의 graph isomorphism이고,
* $\psi_{\G_1}^{-1} \circ \psi_{\G_2} \subseteq \sigma$ 임 (다시 말해, $\G_1$의 boundary vertex가 $\G_2$의 같은 인덱스의 boundary vertex로 map 됨)

으로 정의된다.



**Definition.** 두  $t$-boundaried graph $\G_1 = (G_1, R_1, \lambda_1)$, $\G_2 = (G_2, R_2, \lambda_2)$에서 index가 같은 boundary vertex 간의 mapping이 구조를 보존하면(즉, $\psi_{\G_1}^{-1} \circ \psi_{\G_2}$이 $G_1[R_1]$과 $G_2[R_2]$ 사이의 graph isomorphism이면) $\G_1, \G_2$를 boundary-isomorphic 하다고 부르고, $\G_1 \sim \G_2$ 로 표기한다.



그리고, 다음과 같이 boundaried graph 간의 연산을 두개 정의하자.



**Definition.** $\G_1 = (G_1, R_1, \lambda_1)$ 과 $\G_2 = (G_2, R_2, \lambda_2)$가 두 $t$-boundaried graph라 하자. *gluing* operation $\G_1 \oplus \G_2$를 $G_1$과 $G_2$를 합치되(formally, disjoint union of vertex set), "각 $x \in [t]$에 대해 $\psi_{\G_1}^{-1}(x)$와 $\psi_{\G_2}^{-1}(x)$를 하나의 정점으로 붙여서" 만든 graph라 하자.



**Definition.** $\G_1 = (G_1, R_1, \lambda_1)$과 $\G_2 = (G_2, R_2, \lambda_2)$가 두 boundaried graph라 하자. $I = \lambda_1(R_1) \cap \lambda_2(R_2)$에 대해 $(\vert R_1 \vert + \vert R_2 \vert - \vert I \vert)$-boundaried graph $\G_1 \odot \G_2$를 $G_1$과 $G_2$를 합치되, "각 $x \in I$ 에 대해 $\lambda_1^{-1}(x) (\in R_1)$과 $\lambda_2^{-1}(x) (\in R_2)$를 하나의 정점으로 붙여서" 만든 boundaried graph이다.

즉, $\G_1 \odot \G_2 = (G, R, \lambda)$라 할 때,

* $G$: 위에 설명한 대로 만들어짐
* $R := R_1 \cup R_2$
* $\lambda := \lambda_1 \cup \lambda_2$ (이건 notation을 약간 abuse 한것이긴 하다.).

그리고 이 연산을 merging 연산이라 부른다.



즉, boundary set size가 같은 두 graph들에 대해 gluing 연산은 잘 정의되며, 그 결과로 새로운 (boundaried graph가 아닌 일반) 그래프가 나온다. merging 연산은 두 boundaried graph에 추가적으로 요구하는 조건이 없으며, 결과로 boundaried graph가 나온다. 

이제, boundaried graph 간의 equivalence relation을 살펴보자. 유한 개의 그래프들의 집합 $\F$와 음이 아닌 정수 $t$에 대해 $\B^{(t)}$  상의 equivalence relation $\equiv^{(\F, t)}$를 다음과 같이 정의한다:

* $\G_1 \equiv ^{(\F, t)} \G_2$ if and only if:
  * 임의의 $G \in \B^{(t)}$에 대해 $\F \preceq \G \oplus \G_1$임은 $\F \preceq \G \oplus \G_2$임과 동치

이 equivalence relation이 주는 각 class에서 대표 원소를 하나씩 뽑아 모은 집합을 $\newcommand{\R}{\mathcal{R}} \R^{(\F, t)}$ 로 표기하자. 이 때, 대표원소를 가장 간선 개수가 적은 것(여러 개 있다면, 그 중 가장 정점의 개수가 적은 것)으로 고르자. 그리고 $\G$가 속하는 class의 대표 원소를 $rep_{\F}(G)$와 같이 표기하자. 다음과 같은 사실을 관찰할 수 있다.

* $\G_1 \equiv^{(\F, t)} \G_2$ 임은 "모든 $R \in \R^{(\F, t)}$에 대해 $\F \preceq_{tm} R \oplus \G_1 \iff \F \preceq_{tm} R \oplus \G_2$ "임과 동치임을 쉽게 확인할 수 있다. 다시 말 해, 같은 equivalence class에 속하는지 확인할 때 각 class의 대표 원소와의 gluing 만 고려해보면 충분하다는 것이다.
* $\G = (G, R, \lambda)$가 $t$-boundaried이고, $\F \preceq_{tm} G$ 라면, $rep_{\F}(G)$는 가장 간선의 개수가 작은 $\F$의 원소(와 정점이 $t$개 미만인 경우 isolated vertices들이 추가될 것)로 부터 구성된 $t$-boundaried graph임을 알 수 있다. 이를 $F^{(\F, t)}$라 부르자. 그러면, $\R^{(\F, t)} - \{ F^{(\F, t)}\} \subseteq ex_{tm}(\F)$임을 알 수 있다.
* $\G = (G, R, \lambda) \in \R^{(\F, t)}$ 라면, $V(G) \setminus R$ 에는 최대 $size(\F)$ 개 이내의 isolated vertices만 있음을 관찰할 수 있다. (여기에서,  $size(\F)$는 $\F$에 속하는 그래프들 중 가장 정점이 많은 것의 정점 개수이다.) 대표 원소의 정의로 부터 쉽게 확인할 수 있다.

이제 이를 기반으로 *folio*를 정의할 것이다. Folio는 기본적으로 boundaried graph를 붙여서 어떤 $\F$의 원소가 나타날 것인지에 대한 signature라고 생각하면 좋다.



**Definition.** $\newcommand{\A}{\mathcal{A}} \A_{\F, r}^{(t)}$ 를 최대 $r$개의 non-boundary 정점을 갖고, label set이 $[1, t]$의 subset인 pairwise nonisomorphic ($t$-boundaried가 아닐수도 있음) boundaried graph의 집합이라고 하자. 이를 기반으로, $t$-boundaried graph $\newcommand{\BB}{\mathbf{B}} \BB$ 와 $r \ge 0$에 대해 $folio(\BB, \F, r)$을 $\A_{\F, r}^{(t)}$의 원소 가운데 $\BB$의 topological minor를 모아놓은 것으로 정의하자. 추가로, $\F \preceq_{tm} \BB$ 이면 $folio(\BB, \F, r)$에 $F^{(\F, t)}$ 또한 추가하자.



결국, folio란 boundary label set의 size가 $t$이고, 내부 정점이 $r$개 이하이면서 'topological minor로 금지'하고 싶은 $\F$의 그래프를 형성하는데 기여할 수 있는 모든 '가능성'을 모아둔 것이라 생각하면 된다. 이 folio 들을 전부 포함하는 집합 $\newcommand{\FF}{\mathfrak{F}} \FF_{\F, r}^{(t)} := 2^{A^{(t)}_{\F, r} \cup \{F^{\F, t} \}}$ 을 정의하자.

나중에 우리는 이 folio를 살펴보는 branch decomposition 기반의 동적 계획법 알고리즘을 살펴볼 것인데, 결국 알고리즘의 시간 복잡도는 가능한 모든 folio의 개수에 비례할 것이다. 이와 관련하여, 다음의 사실이 성립한다.



**Lemma.** $t$-boundaried graph $\BB$ 와 $r \ge 0$에 대해 $\vert \A_{\F, r}^{(t)}\vert = 2^{O_{r+d} (t \log t)}$가 성립한다. 참고로, 이로 부터 $\vert folio(\BB, \F, r) \vert = 2^{O_{r+d} (t \log t)}$ 이며, $\vert \FF_{\F, r}^{(t)} \vert = 2^{\vert A_{\F, r}^{(t)} \vert } = 2^{2^{O_{r+d}(t \log t)}}$ 가 된다. 여기에서, $d = size(\F)$ 이다. 그리고 후술할 Lemma 때문에 최종적으로 $\vert \R^{(\F, t)} \vert  \leq 2^{2^{O_d(t \log t)}}$.

*Proof Sketch.* $ex_{tm}(\F)$에 속하는 그래프에서는 $\vert E \vert / \vert V \vert$가 bound 됨이 알려져 있다. 편의상 $\vert E \vert / \vert V \vert \le c$ 라 하자. 이를 이용하여 $(G, R, \lambda) \in \A_{\F, r}^{(t)}$  를 만드는 경우의 수를 계산해보면 쉽게 구할 수 있다. $\square$

**Lemma.** $d = size(\F)$ 라 할 때, 두 $t$-boundaried graph $\BB_1, \BB_2$가 $folio(\BB_1, \F, d) = folio(\BB_2, \F, d)$ 를 만족한다면, $\BB_1 \equiv^{(\F, t)} \BB_2$이다. (증명은 skip)



이제,  $\F$-TM-DELETION 문제에 대해 동적 계획법 알고리즘을 알아보기 위한 준비는 끝났다.

## Dynamic Programming Algorithm

먼저, boundaried graph $\G = (G, R, \lambda)$의 branch decomposition은 다음을 만족하는 순서쌍 $(T, \sigma)$로 정의된다:

* $T$: 정점의 차수가 최대 3인 트리(ternary tree)
* 전단사함수 $\sigma : E(G) \cup \{R \} \to L(T)$; 즉, $G$의 간선과 boundary vertex set를 $T$의 리프 노드와 일대일 대응 시킴

여기에서, $R$이 mapping 된 리프 정점 $r$을 $T$의 root로 두고, $T$의 각 간선 $e$에 대해 $e$를 끊어서 생기는 양쪽의 컴포넌트(subtree) 중 루트를 포함하지 않는 쪽을 $T_e$로 부르자. 이 때, 간선 $e$에 대응되는 'sub-' boundaried graph $\G_e = (G_e, R_e, \lambda_e)$를 다음과 같이 정의하자:

* $G_e$의 간선들은 $T_e$의 리프들에 매칭된 $G$의 간선들이고, 이 간선들에 한 번 이상 등장한 정점들을 모아 $G_e$에 넣어주자.
* $\G_e$의 boundary $R_e$는 $G_e$에 속한 간선의 끝점인 동시에 $G$의 나머지 부분(루트쪽 컴포넌트의 리프에 등장하는 간선) 혹은 $\G$의 boundary set $R$에 등장하는 것들을 모아 만들자.
* $\lambda_e$는 $\lambda \mid_{R_e}$ 를 확장하여 만든 labeling이라 생각하자.

여기에서 branch decomposition $(T, \sigma)$의 width는 $\vert R_e \vert$ 가운데 최댓값으로 정의하고, $\G$의 branchwidth는 $\G$의 모든 branch decomposition을 고려했을 때 얻을 수 있는 최소 width로 정의한다.

참고로, $bw(\G) \le bw(G) + \vert R \vert$ 임을 쉽게 증명할 수 있으며, $bw(G)/tw(G) = \Theta(1)$ 임이 알려져있다.

이제, branch decomposition에 기반한 알고리즘을 살펴보자.

### 1단계: Branch Decomposition 구성

* Treewidth가 $w$인 graph가 주어졌다고 가정하자. 이 때, $2^{O(w)} \cdot n$ 시간에 width가 $O(w)$인 $G$ (=$(G, \emptyset, \emptyset)$)에 대한 branch decomposition을 계산하자.
* 여기에서, 루트 $r$에 연결된 간선 $e_r$에 대해 $\G_{e_r} = (G, \emptyset, \emptyset)$ 임을 일러둔다.

### 2단계: 동적 계획법 테이블 정의

각 간선 $e$에 대해 다음과 같은 $(L, \newcommand{\C}{\mathcal{C}}\C)$ 상태를 정의하고, 이를 $e$-pair 로 부르자.

* $L \subseteq R_e$: 현재의 boundary set $R_e$ 중 '삭제된' 정점들의 집합
* $\C \in \FF_{\F, d}^{t'_e}$: 삭제되지 않고 살아 남은 그래프가 갖는 folio (여기에서, $t'_e$는 남은 정점의 수를 나타내며, $t'_e = t_e - \vert L \vert$) 

각 $e-pair$ 에 대해 DP 테이블($tm_{\F}^{e}$)에는 해당 상태를 만들기 위해 $\G_e$ 내부에서 제거해야 했던 최소의 정점 수를 저장하며 트래킹한다.
$tm_{\F}^{(e)}(L, \C) := \min \{ \vert S \vert \mid  S \subseteq V(G_e), L = R_e \cap S, \C = folio(\G_e \setminus S, d)\}.$

### 3단계: 트리 DP로 테이블 채우기

트리의 리프부터 루트까지 차례로 테이블을 채워 줄 것이다. 참고로, 각 노드의 $e$-pair의 수(DP 테이블 엔트리의 수)는 $2^{2^{O_d(w \log w)}}$이다.

1. 리프 노드인 경우:

   * 리프 노드 $l(\neq r)$에 인접한 유일한 트리 $T$의 간선 $e_l$을 잡자.
   * 여기에서, 가능한 상태($e_l$-pair)들은 $\Theta(1)$개 뿐이므로, $O_d(1)$ 시간에 테이블을 채울 수 있다.

2. 리프 노드가 아닌 경우:

   * 간선 $e$에 의해 생기는 subtree $T_e$의 루트 노드에서 생각할 것이다. 두 자식 간선 $e_1, e_2$를 합쳐주는 것은 다음과 같이 할 수 있다:

     * **for** $e_1$-pair $(L_1, \C_1)$: **for** $e_2$-pair $(L_2, \C_2)$:

       * $(L_1, \C_1)$, $(L_2, \C_2)$ 를 잘 합쳐서 $e$-pair $(L, \C)$가 된다면: $tm_{\F}^{(e)} (L, \C) = \min(tm_{\F}^{(e)}(L, \C), tm_{\mathcal{F}}^{(e_1)}(L_1, \mathcal{C_1}) + tm_{\mathcal{F}}^{(e_2)}(L_2, \mathcal{C_2}))$
     
   * 합치는데 걸리는 시간은 $O_d(\sharp (L_1, \C_1) \times \sharp (L_2, \C_2))  = 2^{2^{O_d(w \log w)}}$ 만큼 걸린다.

### 4단계: 최종 정답

결국 루트($e_r$) 까지 다 채우고 나면, 남은 그래프가 $\F$의 그래프를 포함하지 않는 모든 folio를 고려하여
$tm_{\F}(G) = \min \{ tm_{\F}^{(e_r)}(\emptyset, \C) \mid \C \in 2^{\A_{\F, d}^{(t)}} \}$

로 최종 정답을 구할 수 있다. 두 subtree를 합치는데 $2^{2^{O_d(w \log w)}}$ 만큼의 시간이 걸리고, 그런 이벤트는 $O(n)$번 일어나므로, 시간 복잡도는 총 $2^{2^{O(w \log w)}} \cdot n$ 이 된다.

## 결론 및 여담

이번 글에서는 $\F$-M-DELETION 및 $\F$-TM-DELETION 문제에 대한 알고리즘을 살펴보았다. 이 알고리즘은 branch decomposition 위에서 동적 계획법을 수행한다. 그래프의 구조에 기반한 동적 계획법이라는 점에서 흥미로웠으리라 생각된다.

동적 계획법을 수행하는 동안 각 단계에서 어떤 상태를 관리할지 고려하는게 핵심인데, 우리가 살펴본 알고리즘은 branch decomposition의 boundary를 기준으로 이미 처리한 그래프가 미래에 $\F$에 있는 *forbidden* graph를 형성하는데 기여할 수 있는 모든 가능성을 모아둔, **folio**라는 개념을 정의하여 관리하였다.

Folio의 크기 자체가 $2^{O(tw \log tw)}$로 bound 되었는데, 우리는 동적 계획법을 수행하는 과정에서 서로 다른 folio들의 집합을 관리하고, 합쳐나갔어야 하므로, 총 $2^{2^{O(tw \log tw)}} \cdot n$ 정도의 시간복잡도를 가지는 알고리즘을 얻을 수 있었다.

여담으로, 특수한 그래프 클래스에서는 조금 더 tight 한 folio의 가짓수 upper bound를 얻을 수 있을 것이며, 최근 $2^{tw \log tw} \cdot n$ 시간에 동작하는 알고리즘 또한 같은 연구진에 의해 발표되었다. 다만, 이 알고리즘의 경우 설명이 상당히 난해하여, 블로그 지면 상에 소개할 기회가 있을 지는 의문이다.

## Reference

[1] J. Baste, I. Sau, and D. M. Thilikos (2020). Hitting Minors on Bounded Treewidth Graphs I: General Upper Bounds. SIAM J. Discrete Math Vol 34, No. 3, pp. 1623-1648
