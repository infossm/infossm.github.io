---
layout: post
title:  "Segment Tree를 활용한 2D Range Update/Query"
date:   2018-12-23 15:00:00
author: 박수찬
tags: [segment-tree]
---

이 포스트는 Nabil Ibtehaz, Mohammad Kaykobad, Mohammad Sohel Rahman의 [Multidimensional segment trees can do range queries and updates in logarithmic time](https://www.researchgate.net/publication/328758294_Multidimensional_segment_trees_can_do_range_queries_and_updates_in_logarithmic_time) 논문에서 핵심 아이디어를 가져와 작성한 것입니다.

독자가 1차원 세그먼트 트리와 Lazy propagation에 대한 지식을 알고 있다고 가정하고 글을 작성합니다.
아래에 제시된 코드는 모두 Kotlin으로 작성하였습니다.

# 목표

이 글의 목표는 2차원 세그먼트 트리를 이용해 
2차원 배열 $$A[0..(n-1)][0..(m-1)]$$에 다음과 같은 연산을 $$O(\log^2 n)$$ 시간에 수행하는 것입니다.

- Update: 모든 $$(x, y) \in [x_1, x_2] \times [y_1, y_2]$$와 주어진 값 $$c$$에 대해 $$A[x][y] \leftarrow A[x][y] \star c$$
- Query: $$\star_{(x, y) \in [x_1, x_2] \times [y_1, y_2]} A[x][y]$$ (즉, $$[x_1, x_2] \times [y_1, y_2]$$ 범위의 모든 $$A[x][y]$$를 $$\star$$ 연산으로 reduce한 결과) 구하기

이 때 $$\star$$는 결합법칙과 교환법칙이 성립한다고 가정하고 진행합니다.
1차원에서는 원소가 순서대로 놓여 있기 때문에 결합법칙만 성립하면 되는데,
2차원에서는 Query에서 연산을 적용할 순서를 정하기가 어렵기 때문에 교환법칙이 필요하다는 조건을 추가했습니다.
$$\star$$로 가능한 연산에는 $$+$$, 스칼라 곱, $$\min$$, $$\max$$ 등이 있으며,
불가능한 연산에는 행렬 곱 등이 있습니다.

또한 $$a \star^n b$$를 $$a \underbrace{ \star b \star b \star \cdots \star b}_{n\text{ times}}$$로 정의합니다.
예를 들어 $$\star$$가 덧셈이라면, $$a \star^n b$$의 값은 $$a + b \times n$$과 같습니다.

# 2차원 세그먼트 트리의 구조

일반적인 방법과 같이, 1차원 세그먼트 트리의 각 노드에 1차원 세그먼트 트리를 저장하는 방식을 사용합니다.
첫 번째 차원을 $$x$$축, 두 번째 차원을 $$y$$축이라고 하겠습니다.

먼저, $$x$$축을 기준으로 하는 1차원 세그먼트 트리의 구조를 잡습니다.
이 트리의 각 노드에는 $$y$$축에 대한 1차원 세그먼트 트리의 루트 `yRoot`와
자신의 자식 노드 `left`, `right`만 저장합니다.
이 노드가 담당하고 있는 $$x$$축의 구간은 Update 및 Query 과정에서 구할 수 있으므로 
메모리 절약을 위해 따로 저장하지 않습니다.

``` kotlin
data class xNode(
  val yRoot : yNode? = null, 
  val left : xNode? = null, 
  val right: xNode? = null
)
```

다음으로, $$y$$축을 기준으로 하는 1차원 세그먼트 트리의 구조를 잡습니다.
이 트리의 각 노드에는 질의를 처리하는 데에 필요한 정보를 담은 
`globalLazy`, `globalValue`, `localLazy`, `localValue`와,
자신의 자식 노드 `left`, `right`를 저장하며,
이 노드가 담당하는 구간은 `xNode`와 같은 이유로 저장하지 않습니다.
각 변수의 의미는 다음 섹션에서 다룹니다.

``` kotlin
data class yNode(
  val globalLazy: Long = 0,
  val globalValue: Long = 0,
  val localLazy: Long = 0,
  val localValue: Long = 0,
  val left: yNode? = null,
  val right: yNode? = null
)
```

