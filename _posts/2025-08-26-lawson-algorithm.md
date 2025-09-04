---
layout: post
title: "Lawson Algorithm을 통한 Delaunay triangulation 구하기"
date: 2025-08-27
author: azberjibiou
tags: [algorithm, problem-solving, computational-geometry]
---

# Lawson Algorithm을 통한 Delaunay triangulation 구하기
본 글은 평면 위 점들의 집합을 가장 안정적이고 균일한 삼각형들로 분할하는 **Delaunay Triangulation**을 종합적으로 탐구한다. 
먼저 보로노이 다이어그램과의 관계 및 '빈 외접원 속성'을 통해 들로네 삼각분할의 수학적 정의를 명확히 하고, 이어서 고전적인 구축 방법인 **Lawson's algorithm**을 분석한다. 
마지막으로 이 알고리즘이 왜 항상 정확한 결과를 보장하는지를 증명하기 위해 2차원에서의 문제를 3차원으로 lifting하는 우아한 기법을 사용한다.


## **1. Delaunay triangulation의 정의와 최적성**

평면에 분포된 유한한 점의 집합을 겹치지 않는 삼각형들로 분할하는 triangulation은 계산기하학의 근본적인 문제 중 하나로, 유한요소해석, 컴퓨터 그래픽스, 지형 모델링 등 다양한 분야에서 핵심적인 역할을 한다. 수많은 가능한 삼각분할 중에서도 **Delaunay Triangulation**은 특정 기하학적 최적성 기준을 만족하여, 수치적으로 안정적이고 시각적으로 균일한 결과를 제공하는 표준적인 방법으로 널리 인정받는다.

---

### **1.1. Voronoi Diagram과의 duality**

평면 위의 서로 다른 점들의 유한 집합 $P = \{p_1, p_2, \dots, p_n\}$가 주어졌을 때, 각 점 $p_i \in P$의 **Voronoi Cell** $V(p_i)$은 평면 위의 다른 어떤 점보다 $p_i$에 더 가까운 점들의 집합으로 정의된다.

이러한 셀들의 집합으로 구성된 **Voronoi Diagram** Vor(P)는 평면을 각 점의 최근접 영역으로 분할한 구조이다.

Delaunay Triangulation DT(P)는 Voronoi Diagram의 **Geometric Dual**로 가장 자연스럽게 정의된다. 정확히는 두 점 $p_i$와 $p_j$의 voronoi cell $V(p_i)$와 $V(p_j)$가 서로 인접하여 교집합이 공집합이 아닐 때 두 점 사이에 간선이 존재한다.

**Proposition.** 점 집합 P의 어떠한 네 점도 한 원 위에 있지 않을 때, $DT(P)$는 triangulation이다.

**Proof.**
어떤 graph가 triangulation이 되려면 다음 두 조건을 만족해야 한다: (A) Planar graph여야 하고, (B) 모든 bounded face가 삼각형이어야 한다.

Planarity 증명:
$Vor(P)$는 평면을 분할하는 구조이므로 그 자체로 planar graph이다. Planar graph의 dual graph는 항상 planar graph임이 잘 알려져 있다. 따라서 $DT(P)$는 planar graph이다.

Bounded Face가 삼각형임의 증명:
Dual graph의 한 면의 경계를 이루는 간선의 수는, 그 면에 대응하는 원래 그래프의 정점의 차수와 같다.
$P$의 어떤 네 점도 한 원 위에 있을 수 없다. 만약 한 Voronoi vertex에 4개 이상의 cell이 만난다면, 이는 4개 이상의 점이 동일 원 위에 있음을 의미하므로 가정에 모순이다. 따라서 $Vor(P)$의 모든 정점의 차수는 정확히 3이다.
그러므로 $DT(P)$의 모든 bounded face는 반드시 3개의 간선으로 둘러싸인 삼각형이다.

위의 1과 2에 의해, $DT(P)$는 planar graph이고 모든 bounded face가 삼각형이므로 triangulation이다. Q.E.D.

---

### **1.2. Delaunay Triangulation의 핵심 조건: 빈 외접원 속성**

Voronoi diagram과의 dual 관계로부터 Delaunay triangulation을 판별하는 강력하고 독립적인 조건인 **빈 외접원 속성**(Empty Circumcircle Property)이 유도된다.

**Theorem.** 한 triangulation $T$가 Delaunay triangulation일 필요충분조건은, $T$에 속한 모든 삼각형의 외접원이 그 삼각형을 구성하는 세 정점 외에 집합 $P$에 속한 다른 어떤 점도 내부에 포함하지 않는 것이다.

**Proof.**

**(1) ($\Rightarrow$): Delaunay triangle 내부에는 정점이 없다.**

이 증명은 duality에서 명확히 드러난다. $DT(P)$에 속한 임의의 삼각형 $\triangle p_i p_j p_k$를 생각해보자. 이 삼각형은 $DT(P)$의 face이므로, dual 관계에 의해 $Vor(P)$의 vertex $v$에 대응된다. 이 vertex $v$는 정의상 세 cell $V(p_i), V(p_j), V(p_k)$가 만나는 점이며, 세 점 $p_i, p_j, p_k$에서 동일한 거리에 있다. 따라서 $v$는 삼각형 $\triangle p_i p_j p_k$의 외심(circumcenter)이다.

이제, 만약 이 외접원 내부에 다른 점 $p_l$이 존재한다고 가정해보자. 그렇다면 점 $p_l$은 외심인 $v$에 $p_i, p_j, p_k$보다 더 가깝다. 이는 $v$가 $p_l$의 Voronoi cell인 $V(p_l)$에 속해야 함을 의미하며, 이는 $v$가 $V(p_i), V(p_j), V(p_k)$의 경계에 있다는 사실에 모순이다. 따라서 외접원의 내부는 반드시 비어 있어야 한다.

**(2) ($\Leftarrow$): 빈 외접원 속성을 만족하는 triangulation은 Delaunay triangulation이다.**

이제 역을 증명하기 위해, 모든 삼각형이 빈 외접원 속성을 만족하는 triangulation $T$가 있다고 가정하자.

1.  $T$에 속한 임의의 삼각형 $\triangle pqr$을 선택하자. 가정에 의해, 이 삼각형의 외접원은 내부에 다른 점을 포함하지 않는다. 이 외접원의 중심을 $c$라고 하자.
2.  빈 외접원 속성은 외접원의 중심 $c$가 $P$에 속한 다른 어떤 점 $x$보다 점 $p, q, r$에 더 가깝거나 거리가 같다는 것을 의미한다. 즉, $d(c, p) = d(c, q) = d(c, r) \le d(c, x)$ for all $x \in P$이다.
3.  이 조건은 $c$가 Voronoi cell $V(p), V(q), V(r)$의 교집합에 속한다는 정의와 정확히 일치한다. 따라서 외접원의 중심 $c$는 이 세 cell이 만나는 Voronoi vertex이다.
4.  $V(p), V(q), V(r)$이 한 정점에서 만난다는 것은, 이 세 cell이 서로 인접(adjacent)하여 공통된 간선을 공유함을 의미한다.
5.  Dual의 정의에 따라, 이는 간선 $(p,q), (q,r), (r,p)$가 모두 Delaunay edge임을 뜻한다.
6.  이 논리는 $T$에 속한 모든 삼각형에 적용되므로, $T$를 구성하는 모든 간선은 Delaunay edge이다. 두 triangulation의 간선 집합이 동일하므로, $T$는 Delaunay triangulation $DT(P)$와 반드시 같아야 한다. Q.E.D.

---

## **2. Lawson's Algorithm: Implementation via Iterative Flipping**

1장에서 Delaunay triangulation의 정의와 그것이 왜 최적의 triangulation으로 간주되는지를 정립했다. 이제부터는 주어진 점 집합 $P$에 대해 이러한 Delaunay triangulation을 실제로 구축하는 알고리즘에 대해 논의한다. 그중 가장 기본적이고 직관적인 방법 중 하나가 바로 **Lawson's algorithm**이다.

Lawson's algorithm의 핵심 아이디어는 점진적으로 완벽한 구조를 쌓아 올리는 것이 아니라, 일단 어떤 형태든 상관없이 완전한 triangulation을 만든 후, 국소적인 결함들을 반복적으로 수정하여 점차 전체를 최적의 상태로 개선해 나가는 **iterative improvement** 방식에 있다. 알고리즘은 더 이상 개선할 부분이 없을 때, 즉 Delaunay triangulation이 완성되었을 때 종료된다.

---

### **2.1. 알고리즘의 개념과 The Illegal Edge**

Lawson's algorithm의 반복적인 수정 과정은 '결함'을 찾아 수정하는 단일 연산에 기반한다. 여기서 '결함'이란 Delaunay 조건을 만족하지 못하는 국소적인 기하학적 형태를 의미하며, 이는 **illegal edge**라는 개념으로 정의된다.

> **Definition: The Illegal Edge**
> Triangulation의 내부 간선 $(p_i, p_j)$가 두 삼각형 $\triangle p_i p_j p_k$와 $\triangle p_i p_j p_l$에 의해 공유될 때, 만약 점 $p_l$이 삼각형 $\triangle p_i p_j p_k$의 외접원 내부에 위치한다면, 이 간선 $(p_i, p_j)$는 **illegal**하다고 정의된다.

이 정의는 1.2절에서 다룬 빈 외접원 속성을 직접적으로 위반하는 상황을 가리킨다. 사각형 $p_i p_k p_j p_l$을 고려했을 때, 대각선인 $(p_i, p_j)$가 Delaunay 조건을 만족하지 못하는 것이다. 대칭적으로, 점 $p_k$가 $\triangle p_i p_j p_l$의 외접원 내부에 있을 때도 이 간선은 illegal하다.

Lawson's algorithm은 triangulation 내에 존재하는 모든 illegal edge를 찾아, 다음 절에서 설명할 edge-flipping 연산을 통해 legal edge로 바꾸는 과정을 더 이상 수정할 illegal edge가 없을 때까지 반복한다.

---

### **2.2. The Edge-Flipping Operation**

2.1절에서 정의한 illegal edge를 수정하여 Delaunay 조건을 만족시키는 국소적 변환 방법이 바로 **edge-flipping** 연산이다. 이 연산은 Lawson's algorithm의 심장과도 같은 핵심 메커니즘이다.

Edge-flipping은 두 개의 인접한 삼각형이 공유하는 내부 edge에 대해서만 적용될 수 있다. 내부 edge $(p_i, p_j)$가 두 삼각형 $\triangle p_i p_j p_k$와 $\triangle p_i p_j p_l$에 의해 공유되고, 이들이 함께 볼록 사각형(convex quadrilateral) $p_i p_k p_j p_l$을 형성한다고 가정하자.

이때 edge-flipping 연산은 다음과 같이 수행된다.

1.  공유되던 기존 간선 $(p_i, p_j)$를 제거한다.
2.  사각형의 다른 두 정점인 $p_k$와 $p_l$을 잇는 새로운 간선 $(p_k, p_l)$를 추가한다.

이 연산을 통해 기존의 두 삼각형 $\triangle p_i p_j p_k$와 $\triangle p_i p_j p_l$은 사라지고, 새로운 두 삼각형 $\triangle p_k p_l p_i$와 $\triangle p_k p_l p_j$로 재구성된다. 사각형의 외부 경계는 그대로 유지된 채 내부의 triangulation 방식만 바뀌게 된다. Lawson's algorithm에서는 이 edge-flipping 연산을 illegal edge에 대해서만 선택적으로 적용하여 triangulation의 품질을 점진적으로 개선한다.

---

### **2.3. Edge-Flipping의 수렴성과 정확성 (Convergence and Correctness)**

지금까지 illegal edge를 찾아 edge-flipping을 통해 수정하는 Lawson's algorithm의 핵심 연산을 정의했다. 이제 이 반복적인 과정이 유한한 횟수 안에 반드시 종료되며, 그 결과가 정확히 Delaunay triangulation임을 증명해야 한다. 이 증명은 2차원 평면의 triangulation 문제를 3차원 공간의 볼록한 곡면 문제로 변환하는 강력한 기법인 **Lifting Map**을 통해 이루어진다.

---

#### **The Lifting Map과 핵심 Lemma**

2차원 평면의 각 점 $p = (p_x, p_y)$를 3차원 공간의 포물면(paraboloid) 위의 점 $\hat{p}$로 대응시키는 Lifting Map을 정의하자.
$$\hat{p} = (p_x, p_y, p_x^2 + p_y^2)$$
이 변환의 가장 중요한 속성은 2차원에서의 in-circle 테스트를 3차원에서의 orientation 테스트로 바꿔준다는 것이다.

> **Lemma.** 점 $s$가 삼각형 $\triangle pqr$의 외접원 내부에 있을 필요충분조건은, 3차원으로 lift된 점 $\hat{s}$가 $\hat{p}, \hat{q}, \hat{r}$을 지나는 평면보다 아래(below)에 놓이는 것이다.

**Proof Sketch.** 표준적인 in-circle 테스트는 다음과 같은 4x4 행렬식의 부호로 판별할 수 있다.

$$
\det \begin{pmatrix}
p_x & p_y & p_x^2+p_y^2 & 1 \\
q_x & q_y & q_x^2+q_y^2 & 1 \\
r_x & r_y & r_x^2+r_y^2 & 1 \\
s_x & s_y & s_x^2+s_y^2 & 1
\end{pmatrix}
$$

이 행렬식의 값은 다음과 같은 두 가지 조건 하에서 0이 된다:
1.  2D에서 네 점 $p, q, r, s$가 한 원 위에 있을 때 (cocircular).
2.  3D에서 lift된 네 점 $\hat{p}, \hat{q}, \hat{r}, \hat{s}$가 한 평면 위에 있을 때 (coplanar).

이 두 조건은 서로 동치이며, 원의 일반 방정식 $A(x^2 + y^2) - Bx - Cy + D = 0$으로부터 유도된다. 행렬식의 부호는 점 $s$가 원의 내부에 있는지(평면 아래에 있는지) 혹은 외부에 있는지를 결정한다.

---

#### **수렴성 증명**

1.  각 edge-flip은 2차원에서 illegal edge를 제거하는 연산이다. Lemma에 따라, 이는 3차원에서 움푹 들어간 non-convex한 부분을 '펴서' 더 볼록한 형태로 만드는 것과 같다.
2.  이 과정은 lift된 점들의 아래쪽 볼록 껍질(lower convex hull)을 찾아가는 과정으로 해석할 수 있다. 각 flip은 볼록 껍질에 더 가까워지는 방향으로 단조롭게 진행된다.
3.  주어진 $n$개의 점으로 만들 수 있는 edge의 총개수는 $\binom{n}{2}$로 유한하다. 각 flip은 단조로운 개선을 이루므로 이전에 flip한 간선은 다시 flip될 수 없다. 따라서 이 과정은 유한한 횟수($O(n^2)$ 이하) 안에 반드시 종료된다.

---

#### **정확성 증명**

알고리즘은 더 이상 illegal edge가 없을 때 종료된다. 이 최종 상태의 triangulation을 $T_f$라고 하자.

1.  귀류법을 사용하기 위해, $T_f$가 Delaunay triangulation이 아니라고 가정하자.
2.  가정에 따라, $T_f$에는 empty circumcircle property를 위반하는 삼각형 $\triangle pqr$이 존재한다. 즉, 이 삼각형의 외접원 내부에 다른 점 $s$가 있다.
3.  Lifting Map Lemma에 의해, 3차원에서 점 $\hat{s}$는 평면 $H$($\hat{p}, \hat{q}, \hat{r}$을 지나는) 아래에 놓인다.
4.  lift된 triangulation의 vertex인 $\hat{s}$가 그 triangulation의 face인 $\triangle pqr$이 정의하는 평면 $H$ 아래에 놓여 있으므로, $T_f$를 lift한 3D 표면은 전체적으로 convex하지 않다.
5.  convex하지 않은 다면체 표면에는 반드시 non-convex edge가 존재한다. 즉, 두 삼각형 면이 만나 '계곡'을 이루는 edge $(\hat{a}, \hat{b})$가 반드시 존재한다.
6.  이러한 non-convex edge $(\hat{a}, \hat{b})$는 2차원에서 정확히 illegal edge $(a,b)$에 해당한다.
7.  이는 더 이상 illegal edge가 없다는 알고리즘의 종료 조건에 모순이다.

따라서 $T_f$가 Delaunay triangulation이 아니라는 가정은 거짓이며, 알고리즘의 결과는 반드시 Delaunay triangulation이다. Q.E.D.

---

## **3. 결론**

본고에서는 평면 위 점 집합에 대한 최적의 삼각분할인 Delaunay triangulation의 수학적 정의와 핵심 속성에 대해 탐구하고, 이를 구축하기 위한 고전적이고 직관적인 방법인 Lawson's algorithm을 심도 있게 분석했다.

Lawson's algorithm은 임의의 triangulation에서 시작하여, Delaunay 조건을 국소적으로 위반하는 illegal edge를 찾아 edge-flipping 연산으로 수정하는 단순한 규칙을 반복한다. 이 반복적 개선 과정은 3차원 포물면에 점들을 투영하는 Lifting Map을 통해, 3D 표면의 오목한 부분을 점차 펴내어 아래쪽 볼록 껍질(lower convex hull)을 찾아가는 과정과 같음을 보였다. 각 flip이 단조로운 개선을 보장하므로 이 과정은 $O(n^2)$ 시간 복잡도 내에 반드시 종료되며, 그 결과는 Delaunay triangulation의 정의를 완벽하게 만족함을 증명했다.

Lawson's algorithm은 현대의 더 빠른 $O(n \log n)$ 알고리즘에 비해 성능 면에서는 한계가 명확하다. 하지만 그 학술적 의의는 매우 크다. 이 알고리즘은 어떤 triangulation에서든 Delaunay triangulation에 도달할 수 있음을 보여주는 강력하고 건설적인 증거가 되며, edge-flipping이라는 연산은 다른 복잡한 기하 알고리즘에서도 핵심적인 부품으로 사용된다. 결국 Lawson's algorithm은 Delaunay triangulation이라는 목표의 우아한 기하학적 구조를 가장 명확하고 직관적인 방식으로 보여주는 근본적인 알고리즘으로 평가할 수 있다.

## **4. 참고자료**

https://n.ethz.ch/~vkuperber/flipgraph.pdf
