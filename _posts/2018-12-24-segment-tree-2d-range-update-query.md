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
  var totalSum: Value = Value(),
  var totalLazy: Value = Value(),
  var left: yNode? = null,
  var right: yNode? = null
)
```

# Update

예시를 들며 설명해 보겠습니다.
$$H = 7$$일 때, $$x$$축 세그먼트 트리는 아래와 같이 구축할 수 있습니다.

![크기가 7인 구간을 덮는 세그먼트 트리](/assets/images/segment-tree-2d-range-update-query/x-segtree.png)

이 상태에서, 어떤 $$x$$축 구간 $$[x_1, x_2)$$를 업데이트하려고 한다고 합시다.
루트에서 아래로 내려가면서, 
노드가 덮고 있는 구간이 $$[x_1, x_2)$$에 완전히 포함되면 노란색,
노란색 노드에 닿기 위해 방문해야 할 조상 노드들을 푸른색으로 색칠합니다.
아래 그림은 $$[2, 6)$$ 구간에 대해 각 노드를 색칠한 결과입니다.
이제 색에 따라 다른 방식으로 각 노드에 저장된 $$y$$축 세그먼트 트리를 업데이트해야 합니다.

![](/assets/images/segment-tree-2d-range-update-query/x-segtree-updated-26.png)

## 노란색 노드 업데이트하기

노란색 노드가 덮고 있는 $$x$$축 반개구간을 $$[nx_1, nx_2)$$라고 합시다. 
$$[nx_1, nx_2)$$가 $$[x_1, x_2)$$에 완전히 포함되어 있기 때문에,
배열 $$A$$의 모든 행(줄)의 똑같은 $$y$$축 위치에 똑같은 값이 더해지게 된다고 생각할 수 있습니다.

![](/assets/images/segment-tree-2d-range-update-query/yellow-node-in-array.png)

따라서, 이 경우에는 노란색 노드가 갖고 있는 $$y$$축 세그먼트 트리가
$$nx_1$$번 행(아무 행이나 잡아도 됨)의 정보를 관리하도록 하면 충분합니다.
그러면 업데이트는 1차원 세그먼트 트리에서 $$[y_1, y_2)$$ 구간에 $$c$$를 더하는 것과 같으므로,
전형적인 Lazy propagation 방식을 그대로 이용하면 됩니다.
여기에 사용되는 변수가 `globalRowSum`과 `globalRowLazy`입니다.

이 과정은 lazy propagation을 활용한 1차원 구간 업데이트 알고리즘에서 lazy tag에 $$c$$를 더하는 것에 비유할 수 있습니다.

## 푸른색 노드 (및 노란색 노드) 업데이트하기

다시 1차원 구간 업데이트 알고리즘을 생각해 보면, 
노란색 노드를 처리하고 난 뒤 루트로 돌아오면서 각 노드에 저장된 부분합을 관리해 주었죠.
노란색 노드의 경우 합에 $$c \times $$ (구간 길이)를 더했고,
푸른색 노드의 경우 두 자식 노드에 저장된 두 부분합을 합쳤습니다.

2차원에서도 비슷한 느낌으로 각 $$x$$축 세그먼트 트리의 노드가 어떤 부분합을 가지고 있어야 할 것입니다.
따라서, 푸른색 노드가 담당하고 있는 모든 행벡터의 합을 $$y$$축 세그먼트 트리에서 관리합니다.

다른 방식으로 표현하자면, $$y$$축 세그먼트 트리의 한 노드가 담당하고 있는 구간이 $$[nx_1, nx_2) \times [ny_1, ny_2)$$일 때, 이 노드는 $$\sum_{(x, y) \in [nx_1, nx_2) \times [ny_1, ny_2)} A[x][y]$$의 값을 관리하는 것입니다.

![](/assets/images/segment-tree-2d-range-update-query/blue-node-row-vector-sum.png)

그러면 업데이트는 1차원 세그먼트 트리에서 $$[y_1, y_2)$$ 구간에 
$$c \times |[x_1, x_2) \cap [nx_1, nx_2)|$$를 더하는 것과 같으므로,
역시 전형적인 Lazy propagation 방식을 그대로 이용하면 됩니다.
여기에 이용되는 변수가 `totalSum`과 `totalLazy`입니다.

## 코드

``` kotlin
fun update(x1: Int, x2: Int, y1: Int, y2: Int, c: Value) {
  // [x1, x2) x [y1, y2) 범위에 c를 더하는 함수

  fun updateX(xnd: xNode, nx1: Int, nx2: Int, ux1: Int, ux2: Int) {
    // x축 세그먼트 트리를 순회하며, 각 노드의 y축 세그먼트 트리를 업데이트하도록 하는 함수
    //  현재 노드가 xnd이며, 이 노드는 [nx1, nx2) 구간을 담당함.
    //  업데이트할 범위는 [ux1, ux2)이며, 이는 노드가 담당하는 범위에 완전히 포함됨.

    if(nx1 != ux1 || nx2 != ux2) { 
      // xnd가 푸른색 노드이므로, 더 아래로 내려가야 함

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
      // y축 세그먼트 트리에서 lazy propagation을 하는 함수
      //  현재 노드가 ynd이며, 이 노드는 [nx1, nx2) x [ny1, ny2) 구간을 담당함.
      //  업데이트할 범위는 [ux1, ux2) x [uy1, uy2)이며, 이는 노드가 담당하는 범위에 완전히 포함됨.

      if(ny1 == uy1 && ny2 == uy2) {
        if(nx1 == ux1 && nx2 == ux2) { // xnd가 노란색 노드이므로, global 값들을 업데이트함
          ynd.globalRowSum += c * (uy2 - uy1)
          ynd.globalRowLazy += c
        }

        ynd.totalSum += c * (ux2 - ux1) * (ny2 - ny1)
        ynd.totalLazy += c * (ux2 - ux1)
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
        
        if(nx1 == ux1 && nx2 == ux2) {
          ynd.globalRowSum = (ynd.left ?.globalRowSum ?: Value()) +
                              (ynd.right ?.globalRowSum ?: Value()) +
                              ynd.globalRowLazy * (ny2 - ny1)
        }

        ynd.totalSum = (ynd.left ?.totalSum ?: Value()) + 
                        (ynd.right ?.totalSum ?: Value()) +
                        ynd.totalLazy * (ny2 - ny1)
      }
    }
    
    // y축 세그먼트 트리의 [y1, y2) 범위를 업데이트함
    updateY(xnd.yRoot, 0, cols, y1, y2)
  }
  
  updateX(root, 0, rows, x1, x2)
}
```

# Query