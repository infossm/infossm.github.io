---
layout: post
title:  "Segment Tree를 활용한 2D Range Update/Query"
date:   2018-12-23 15:00:00
author: 박수찬
tags: [segment-tree]
---

이 포스트는 Nabil Ibtehaz, Mohammad Kaykobad, Mohammad Sohel Rahman의 [Multidimensional segment trees can do range queries and updates in logarithmic time](https://www.researchgate.net/publication/328758294_Multidimensional_segment_trees_can_do_range_queries_and_updates_in_logarithmic_time) 논문에서 핵심 아이디어를 가져와 작성한 것입니다.

독자가 1/2차원 세그먼트 트리와 Lazy propagation에 대한 지식을 알고 있다고 가정하고 글을 작성합니다.
아래에 제시된 코드는 모두 Kotlin으로 작성하였습니다.

# 목표

이 글의 목표는 2차원 세그먼트 트리를 이용해
$$H \times W$$ 크기의 2차원 배열 $$A$$에 다음과 같은 연산을 $$O(\log n \log m)$$ 시간에 수행하는 것입니다.

- Update: 모든 $$(x, y) \in [x_1, x_2) \times [y_1, y_2)$$와 주어진 값 $$c$$에 대해 $$A[x][y] \leftarrow A[x][y] + c$$
- Query: $$\sum_{(x, y) \in [x_1, x_2) \times [y_1, y_2)} A[x][y]$$ 구하기

실제로는 교환법칙과 결합법칙이 성립하는 모든 연산자(곱셈, xor, and, or, min, max 등)에 대해서도
Update와 Query를 수행할 수 있습니다.

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
  var left : xNode? = null, 
  var right: xNode? = null
)
```

다음으로, $$y$$축을 기준으로 하는 1차원 세그먼트 트리의 구조를 잡습니다.
이 트리의 각 노드에는 질의를 처리하는 데에 필요한 정보를 담은 
`globalRowSum`, `globalRowLazy`, `localTotalSum`, `localTotalLazy`와,
자신의 자식 노드 `left`, `right`를 저장하며,
이 노드가 담당하는 구간은 `xNode`와 같은 이유로 저장하지 않습니다.
각 변수의 의미는 다음 섹션에서 다룹니다.

``` kotlin
data class yNode(
  var globalRowSum: Value = Value(),
  var globalRowLazy: Value = Value(),
  var localTotalSum: Value = Value(),
  var localTotalLazy: Value = Value(),
  var left: yNode? = null,
  var right: yNode? = null
)
```

# 업데이트

아래와 같은 함수를 구현하고자 합니다.

``` kotlin
fun update(x1: Int, x2: Int, y1: Int, y2: Int, c: Value) {
  fun updateX(xnd: xNode, nx1: Int, nx2: Int, ux1: Int, ux2: Int) {
    if(nx1 != ux1 || nx2 != ux2) {
      val nxm = (nx1 + nx2) / 2
      if(ux1 < nxm) {
        if(xnd.left == null) xnd.left = xNode()
        updateX(xnd.left!!, nx1, nxm, ux1, minOf(ux2, nxm))
      }
      if(nxm < ux2) {
        if(xnd.right == null) xnd.right = xNode()
        updateX(xnd.right!!, nxm, nx2, maxOf(nxm, ux1), ux2)
      }
    }

    fun updateY(ynd: yNode, ny1: Int, ny2: Int, uy1: Int, uy2: Int) {
      if(ny1 == uy1 && ny2 == uy2) {
        // TODO
      }else {
        val nym = (ny1 + ny2) / 2
        
        if(uy1 < nym) {
          if(ynd.left == null) ynd.left = yNode()
          updateY(ynd.left!!, ny1, nym, uy1, minOf(uy2, nym))
        }
        
        if(nym < uy2) {
          if(ynd.right == null) ynd.right = yNode()
          updateY(ynd.right!!, nym, ny2, maxOf(uy1, nym), uy2)
        }
        
        // TODO
      }
    } 
    updateY(xnd.yRoot, 0, cols, y1, y2)
  }
  updateX(root, 0, rows, x1, x2)
}
```