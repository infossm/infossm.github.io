---
layout: post
title: "Treewidth Parametrized Dynamic Programming for Local Graph Problems"
author: TAMREF
date: 2023-02-19
tags: [graph-theory, dynamic-programming]
---

## Introduction

Vertex Cover, Independent Set, Dominating Set 등은 복잡도 이론에 관심이 있다면 익히 들어보았을 법한 문제들입니다. 각 문제의 정의는 다음과 같습니다.

무방향 단순그래프 $G = (V, E)$에 대해,

- Vertex Cover: 모든 $e = (u, v) \in E$에 대해 $u \in S 또는 $v \in S$를 만족하는 $S \subseteq V$를 찾아라.
- Independent Set: 모든 $u, v \in S$에 대해 $(u, v) \notin E$인 $S \subseteq V$를 찾아라.
- Dominating Set: 모든 $v \in V$에 대해 $v \in S$ 또는 $\mathcal{N}(v) \cap S \neq \emptyset$인 $S$ 를 찾아라.
  - 여기서 $\mathcal{N}(v)$란 $v$와 이웃한 정점들의 집합.

보통은 Minimum Vertex Cover, Maximum Independent Set, Minimum Dominating Set 형태의 문제를 해결해야 합니다.

세 문제는 모두 NP-complete으로, 일반적인 그래프에서 해결하는 방법이 알려지지 않았습니다. 동시에 세 문제는 트리에 대해서는 linear time solution이 알려져 있습니다. 재미있게도, 트리에서 해결하는 기법이 모두 비슷합니다.

트리의 루트를 고정하고, $D _ {v, b}$를 $v$의 서브트리에서, $b = (v \in S)$를 나타내는 1비트짜리 정보라고 할 때 최적해로 정의합시다. 이 때 $D _ {v, b}$는 $v$의 child $c _ {1}, \cdots, c _ {k}$에 대해 $D _ {c _ {i}, \ast}$ 만 참조하여 구할 수 있고, 세 문제를 모두 트리에서는 해결할 수 있습니다.

일반 그래프에서 이 전략이 통하지 않는 이유는 뭘까요? 가령 DFS tree 를 구한 뒤 비슷하게 문제를 해결하고자 한다면, $D _ {v, b}$는 $v$의 자식 $c _ {i}$들 뿐만 아니라 front edge로 $v$와 인접한 정점인 $f _ {i}$에 대해 $D _ {f _ {i}, b}$, back edge로 인접한 정점인 $b _ {i}$에 대해 $D _ {b _ {i}, b}$ 등을 모두 고려하여 구해야 할 것입니다. 하지만 이 정보들은 트리에서와 달리 서로 독립적이지 않기 때문에, 정보들이 서로 모순을 일으키지 않는 assignment들을 최대 $\mathcal{O}(2^{n})$개의 assignment 중에서 찾아야 합니다. 트리에서는 자식들의 assignment가 서로 모순을 유도하지 않았다는 점과 대조적입니다.

이렇듯 $v$에 인접한 정점들의 정보만 이용해서 partial solution을 계속 업데이트해나가는 형태의 문제를 Local Problem이라고 부르겠습니다. Local Problem의 경우 "트리같은" 그래프에 대해서 다항 시간 해결법이 많이 알려져 있고, 이는 경시대회 문제의 단골 소재가 됩니다. 어떤 그래프가 "트리같다"라는 것은 일반적으로 작은 treewidth를 갖는다는 것인데, treewidth의 정의와 그 직관적인 이해는 [koosaga님의 다음 글](https://koosaga.com/295) 을 참고하시면 좋습니다. 여기서는 treewidth와 tree decomposition의 정의를 풀어 설명하진 않기로 합니다.

**Definition.** (Tree decomposition) 그래프 $G$의 tree decomposition 이란 트리 $T$와 각 정점 $v \in V(T)$마다 $X _ {v} \subseteq V(G)$가 존재하여 다음 조건을 만족하는 것을 말합니다.

- $\bigcup _ {v} X _ {v} = V(G)$.
- $(x, y) \in E(G)$일 때, $x, y \in X _ {v}$인 $v \in V(T)$가 존재한다.
- $u, v \in V(T)$에 대해, $w$가 $u, v$를 잇는 경로 위에 있다면 $X _ {u} \cap X _ {v} \subseteq X _ {w}$.

이 때 $\max _ {v} \lvert X _ {v} \rvert - 1$을 decomposition의 width라고 하고, $G$의 tree decomposition 중 width의 최솟값을 $G$의 treewidth $\mathbf{tw}(G)$라고 정의한다. $X _ {1} = \lbrace 1, \cdots, n\rbrace$으로 두면 $\mathbf{tw}(G) \le n-1$은 항상 성립한다.

**Definition.** ($k$-tree) $k$-tree는 다음과 같은 그래프들을 말한다.
- 정점이 $k$개인 $k$-tree는 $K _ {k}$뿐이다.
- $k$-tree $T$의 크기 $k$ clique $C$에 모두 인접한 정점 $v$를 추가하여 새로운 $k$-tree $T + v$를 만들 수 있다.

**Proposition.** $\mathbf{tw}(G)$는 $G$가 $k$-tree의 subgraph가 되는 가장 작은 $k$와 같다.

$k$-tree에는 자연스럽게 정점을 추가한 순서대로 정렬할 수 있습니다. $\mathcal{P}(v)$를 $v$ + ($v$에 인접하면서 먼저 추가된 정점들)로 정의되는 크기 $k+1$짜리 집합이라고 할 때, $DP(v, B _ {P})$를 정점 $v$까지, $\mathcal{P}(v)$의 선택 여부를 나타내는 비트마스크가 $B _ {P}$인 최적해로 정의하면 $2^{\mathcal{O}(k)} n$ 시간 복잡도의 DP를 자연스럽게 생각할 수 있고, subgraph에 대해서도 마찬가지입니다. 따라서 우리가 생각한 Vertex Cover, Independent Set, Dominating Set은 모두 bounded treewidth (given tree decomposition) 그래프에 대해 선형 시간에 해결할 수 있게 됩니다.

사실 집합론의 영역으로 넘어가면 "Monadic Second-Order Logic"으로 표현 가능한 property는 모두 $f(\mathbf{tw}) \cdot \mathcal{O}(n)$ 시간에 해결할 수 있다는 Meta-theorem 격에 해당하는 Courcelle's theorem이 있고, 우리가 아는 대부분의 Local Problem은 저 monadic second order logic으로 나타낼 수 있다는 것 또한 알려져 있습니다. Practical한 영역과는 아득히 먼 알고리즘이니 알아둘 필요는 없지만, 어떤 type의 문제가 bounded treewidth에서 쉽게 풀 수 있는지를 알게 해주는 black-box인만큼 기회가 되면 추후 다루도록 하겠습니다.

어찌되었든 Local Problem은 treewidth에 대해 single-exponential한 시간 복잡도로 해결할 수 있다는 것을 알게 되었습니다. 경시대회에서는 대개 exponent인 treewidth가 $2$ 또는 $3$인 경우만 다루기에 exponent를 줄이는 게 특별한 의미가 있어 보이진 않으나, Practical하게는 생물정보학 등에서 treewidth 10 등의 그래프도 자주 등장하기 때문에 exponent를 줄이는 것도 유의미한 contribution이 될 수 있습니다. 오늘은 elementary한 방법으로 Dominating Set 문제에 대해 $\mathcal{O}(4^{k} n)$ 복잡도를 달성한 Alber(2002)의 방법과, Fast subset convolution을 이용하여 $\mathcal{O}(k^{2} 3^{k} n)$ 복잡도를 달성한 van Rooij (2009)의 방법을 리뷰합니다.

## Nice Tree Decomposition

Tree decomposition에서 DP를 간편하게 하는 방법으로, Nice tree decomposition으로 바꾸는 방법이 있습니다. width $t$의 tree decomposition이 주어져 있으면, 선형 시간 안에 똑같은 width의 tree decomposition을 만들 수 있음이 알려져 있습니다.

**Definition.** $G$의 Nice tree decomposition은 rooted binary $T$ 위에 주어진 모든 bag $X _ {i}$가 다음 4가지 중 하나로 분류되는 tree decomposition을 말한다.

1. Leaf Bag
2. Introduce Bag: $T$에서 $i$의 자식은 $c$하나뿐이고, $X _ {i} = X _ {c} + v$ for some $v \notin X _ {c}$.
3. Forget Bag: $T$에서 $i$의 자식은 $c$하나뿐이고, $X _ {i} = X _ {c} - v$ for some $v \in X _ {c}$.
4. Join Bag: $T$에 $i$의 두 자식 $l, r$이 존재하고, $X _ {i} = X _ {l} = X _ {r}$.

일반적으로 그래프 문제를 풀 때 dfs tree를 만들고 나면 간선이 tree edge / back edge / forward edge / cross edge로 분류되면서 단순해지듯, nice tree decomposition을 구해놓고 나면 4가지 경우에 대해서만 dp transition을 정의하면 되니 한층 편한 감이 있습니다. 일반적으로 Introduce / Forget보다는 Join이 가장 어렵습니다.

## Alber (2002)

Dominating set problem에 있어, 각 정점의 상태 $c _ {v}$는 아래와 같이 분류할 수 있겠습니다.
- $c _ {v} = 1$: 정점 $v$가 dominating set에 들어 있음.
- $c _ {v} = 0 _ {1}$: $v$가 dominating set에 들어 있지는 않으나, dominating set의 원소와 adjacent함.
- $c _ {v} = 0 _ {?}$: $v$가 dominating set에 들어 있지 않고, dominating set의 원소와 adjacent한지 아닌지 모름.

state를 잡는 방법은 여러 가지가 있겠지만 이 방법이 가장 편리합니다. tree dp에서 많이 보았던 형태의 dp와 비슷합니다.

집합 $X$의 원소 $x _ {1}, \cdots, x _ {k}$에 대해, $c _ {x _ 1}, \cdots, c _ {x _ {k}}$를 벡터로 묶어서 그냥 $\mathbf{c} _ {X}$로 표기하겠습니다.

$D _ {i, \mathbf{c}}$를 $i$의 서브트리까지만 봤을 때, $\mathbf{c} _ {X _ i} = \mathbf{c}$를 만족하는 minimum dominating set이라고 정의하겠습니다. 각 node에서 DP transition은 다음과 같습니다.

1. **Leaf Node:** $\mathbf{c}$에 모순이 없으면 $\sharp _ {1}(\mathbf{c})$ ($\mathbf{c}$의 1 개수) 로 $D _ {i, \mathbf{c}}$를 초기화하고, 어떤 노드가 $0 _ {1}$이지만 인접한 노드 중 $1$인 노드가 없는 등 모순이 있으면 $D _ {i, \mathbf{c}} = \infty$로 둡니다.

2. **Forget Node:** 빠지는 노드 $v$가 편의상 가장 마지막 비트라고 할 때, $D _ {i, \mathbf{c}} \leftarrow D _ {u, \mathbf{c} + \lbrace 0 _ {1}, 1 \rbrace}$ 으로 업데이트합니다. $u$는 $i$의 유일한 child.


3. **Introduce Node:** 마찬가지로 들어가는 노드 $v$가 편의상 가장 마지막 비트라고 할 때,
   1. $D _ {i, \mathbf{c} + 0 _ {?}} \leftarrow D _ {u, \mathbf{c}}$
   2. $D _ {i, \mathbf{c} + 0 _ {1}}$은 $v$와 인접한 $X _ {i}$의 원소들 중 $1$이 있는 경우에만 $D _ {u, \mathbf{c}}$로 업데이트해줍니다.
   3. $D _ {i, \mathbf{c} + 1}$은 $\mathbf{c}$에서 $v$와 인접하고 $0 _ {1}$인 비트들을 $0 _ {?}$로 바꿔준 새로운 문자열을 $\phi(\mathbf{c})$라고 할 때, $D _ {u, \phi(\mathbf{c})} + 1$로 업데이트해줍니다. $0 _ {1}$을 $0 _ {?}$로 바꿔줄수록 $D$값이 감소하기 때문에 이 경우만 봐줘도 됩니다.

4. **Join node:** 나머지 모든 DP transition은 $3^{\lvert X \rvert} \le 3^{\mathbf{tw}(G)}$ 시간 만에 가능하지만, 여기서 유일하게 $4^{\lvert X \rvert}$시간이 소요됩니다.
   - $i$의 두 자식을 $l, r$이라고 하면, $D _ {l, \mathbf{c^{\prime}}}$과 $D _ {r, \mathbf{c^{\prime\prime}}}$을 가져와서 $D _ {i, \mathbf{c}}$를 업데이트할 것인데, 다음과 같은 성질이 성립해야 합니다.
   - $c _ {v} = 1$이면, ${c} _ {v}^{\prime} = {c} _ {v}^{\prime\prime} = 1$이어야 한다.
   - $c _ {v} = 0 _ {?}$이면, 마찬가지로 ${c} _ {v}^{\prime} = {c} _ {v}^{\prime\prime} = 0 _ {?}$인 경우만 봐도 된다. DP가 최소가 되는 점만 봐도 되기 때문
   - $c _ {v} = 0 _ {1}$이면, $(c _ {v}^{\prime}, c _ {v}^{\prime\prime}) = (0 _ {1}, 0 _ {?}), (0 _ {?}, 0 _ {1}), (0 _ {1}, 0 _ {1})$ 인 경우만 봐도 되는데, 사실 $(0 _ {1}, 0 _ {1})$인 경우는 무조건 DP값이 더 크므로 앞 2개 상태만 보면 된다.

- 조건을 만족하는 $l, r$의 상태 $\mathbf{c}^{\prime}, \mathbf{c}^{\prime\prime}$에 대해 $D _ {l, \mathbf{c}^{\prime}} + D _ {r, \mathbf{c}^{\prime\prime}} - \sharp _ {1}(\mathbf{c})$로 $D _ {i, \mathbf{c}}$를 업데이트해주면 됩니다.

Join node에서 한 노드를 업데이트하기 위해 들여다보는 노드 수를 계산해보면, $0 _ {1}$의 개수가 절대적입니다. $0 _ {1}$이 $r$개인 노드가 $2^{k-r}\binom{k}{r}$개이고 참고하는 상태의 개수가 $2^{r}$개이니, 전체 transition에 걸리는 시간은 $\sum _ {r} 2^{k-r}2^{r}\binom{k}{r} = 4^{k}$입니다. 여기서 $k = \lvert X _ {i} \rvert$를 의미합니다.

마지막에는 루트 노드에 대해 $\lbrace 0 _ {1}, 1\rbrace$로만 구성된 상태의 DP최솟값을 구해주면 됩니다.

## van Rooij (2009)

앞선 Join node에 대해서 $0 _ {1}, 0 _ {?}$만 주목해서 보면, $0 _ {1}$을 $\mathbf{1}$, $0 _ {?}$을 $\mathbf{0}$이라는 새로운 비트에 볼 때 결국 $\mathbf{c}^{\prime} \vert \mathbf{c}^{\prime\prime} = \mathbf{c}$가 되는 두 disjoint한 비트열 $\mathbf{c}^{\prime}, \mathbf{c}^{\prime\prime}$에 대해 DP를 고려해주는 꼴입니다. 비트열을 집합꼴로 보면 $h(S) := \sum _ {X \subset S} f(X) g(S - X)$를 모든 $S \subseteq \lbrace 1, \cdots, n \rbrace$에 대해 구해주는 것과 비슷합니다. 물론 덧셈, 곱셈 대신 min, 덧셈이 사용되기 때문에 다르지만, 위 식은 Bjorklund (2006)에 의해 널리 알려진 subset convolution이라는 type의 문제입니다. 해당 문제에 대해서는 Naive한 $\mathcal{O}(3^{n})$에 비해 빠른 $\mathcal{O}(n^{2} 2^{n})$ 알고리즘이 존재합니다.

van Rooij는 "minimum dominating set의 개수 세기"라는 관점으로 subset convolution을 활용하여 더 빠르게 동일한 문제를 해결합니다. 다음의 확장된 DP를 정의합시다.

$C(v, \mathbf{d}, r)$: $v$의 서브트리까지 커버하는, $\mathbf{d} _ {X _ {v}} = \mathbf{d}$인 크기 $r$인 dominating set의 개수.

이 때 상태를 나타내는 데 더 이상 $c _ {v} \in \lbrace 1, 0 _ {?}, 0 _ {1} \rbrace$를 사용하지 않고, $d _ {v} \in \lbrace 1, 0 _ {?}, 0 _ {0} \rbrace$ 를 사용합니다. $0 _ {0}$은 dominating set에 있지도 않고, dominating set에 인접하지도 않은 정점의 state를 말합니다. 실제로 답에 필요한 것은 $\lbrace 1, 0 _ {1} \rbrace$ 이지만, $\mathbf{c}$와 $\mathbf{d}$ 간의 변환은 Mobius transform 등으로 잘 알려진, 비트 별로 포함 배제를 해주는 방법으로 $O(n 3^{k})$ 만에 간단하게 해줄 수 있기 때문에 별 문제가 되지 않습니다. 물론 기존에 구하던 minimum dominating set의 크기는 포함 배제와 compatible하지 않기 때문에, alber의 알고리즘에 $\lbrace 1, 0 _ {?}, 0 _ {0} \rbrace$의 상태를 도입해봐야 별 이득이 없습니다.

즉 각 bag마다 $\mathcal{O}(n 3^{k})$ 정도씩 총 $\mathcal{O}(n^{2} 3^{k})$ 크기의 DP table을 유지하게 됩니다. $k$는 줄곧 treewidth입니다.

**Theorem.** 모든 $v \in V(T)$, $1 \le r \le n$에 대해 $C(v, \mathbf{d}, r)$ 를 $\mathcal{O}(n^2 \log n \cdot 3^{k})$번의 곱셈으로 계산할 수 있다. 물론 $T$는 $G$의 tree decomposition이고, $k$는 treewidth.

*Proof.* Join 이외의 node에서는 Alber와 크게 차이가 없기 때문에, Join에 대해서만 문제를 해결하도록 합시다. $v$의 두 자식 $a, b$에 대해 $C(a, \mathbf{d} _ {a}, r _ {a})$와 $C(b, \mathbf{d} _ {b}, r _ {b})$을 활용하여 $C(v, \mathbf{d}, r)$을 채워주면 됩니다. 이 상태가 유용한 점은 $\mathbf{d} = \mathbf{d} _ {a} = \mathbf{d} _ {b}$인 경우만 봐주면 된다는 것입니다. 그래서 $r _ {a} + r _ {b} = r + \sharp _ {1}(\mathbf{d})$인 $(r _ {a}, r _ {b})$에 대해 $\sum _ {(r _ {a}, r _ {b})} C(a, \mathbf{d}, r _ {a})C(b, \mathbf{d}, r _ {b})$ 를 FFT로 계산해주면 됩니다. 시간 복잡도는 $\mathcal{O}(n \log n \cdot 3^{k})$입니다. 이걸 모든 Join node에 대해서 해주면 주어진 시간 복잡도를 얻습니다. $\square$

물론 $\lbrace 1, 0 _ {?}, 0 _ {1} \rbrace$를 이용한 상태로 정의된 비슷한 $\tilde{C}(v, \mathbf{c}, r)$ 에 대해서 subset convolution을 돌려도 동일한 결과를 얻는데, van Rooij에서 소개하고 있는 증명은 아닙니다. 본질적으로 subset convolution이 Mobius transform을 이용한 기저 변환 + FFT와 거의 비슷하기 때문에 사실상 동등한 증명입니다.

**Theorem.** 위 Theorem을 응용하여, minimum dominating set의 개수를 $\mathcal{O}(nk^{2} 3^{k})$ 번의 곱셈으로 셀 수 있다.

*Proof.* $\lbrace 1, 0 _ {1}, 0 _ {0} \rbrace$ 으로 나타나는 상태 $\mathbf{e}$를 사용하면, $C(v, \mathbf{c}, r)$과 비슷하게 $E(v, \mathbf{e}, r)$을 정의할 수 있습니다. 각 $\mathbf{e}$에 대해 $E(v, \mathbf{e}, r) \neq 0$인 $r$의 최솟값을 (존재한다면) $r _ {\mathbf{e}}$라고 둡시다. $r _ {\mathbf{e}}$는 그 최솟값과 최댓값이 $k = \lvert X _ {v} \rvert$ 이하로 차이가 난다는 사실을 알 수 있습니다. 위 Theorem에서 나왔던 convolution-like formula는 $E$에 대해서 아래와 같이 쓸 수 있습니다.

$E(v, \mathbf{e}, r) = \sum _ {\mathbf{f} \mid \mathbf{g} = \mathbf{e}, r _ {a} + r _ {b} = r + \sharp _ {1}(\mathbf{e})} E(a, \mathbf{f}, r _ {a}) E(b, \mathbf{g}, r _ {b})$

이 때 유의미한 $(r _ {a}, \mathbf{f})$ pair가 많아야 $\mathcal{O}(k3^{k})$개에 불과하므로 convolution을 계산하는 데에 $n$ factor를 제거할 수 있습니다. 전체 시간복잡도를 추렴하면 $\mathcal{O}(nk^{2} 3^{k})$가 됩니다.

## Conclusion

이번 글에서는 bounded treewidth에서 dominating set 문제를 해결하는 exponential time algorithm (with small exponent) 들을 알아보았습니다. 사용하는 최적화 기법이 복잡하지 않기 때문에 경시 대회 수준에서도 어떤 treewidth의 그래프가 주어지느냐에 따라 사용될 여지가 얼마든지 있다고 생각합니다. minimum dominating set뿐만 아니라 perfect matching의 개수도 비슷한 방법으로 세어볼 수 있는데, 이에 대해서는 추후 다루겠습니다.

## References

- Van Rooij, Johan MM, Hans L. Bodlaender, and Peter Rossmanith. "Dynamic Programming on Tree Decompositions Using Generalised Fast Subset Convolution." ESA. Vol. 5757. 2009.
- Alber, Jochen, and Rolf Niedermeier. "Improved tree decomposition based algorithms for domination-like problems." LATIN 2002: Theoretical Informatics: 5th Latin American Symposium Cancun, Mexico, April 3–6, 2002 Proceedings 5. Springer Berlin Heidelberg, 2002.
  
두 main 논문입니다. Van Rooij는 이 뿐만 아니라 bounded treewidth graph의 DP에 대해 다양한 연구를 했으니 일독을 권합니다.

- Koosaga, ["TreeWidth를 사용한 PS 문제 해결"](https://koosaga.com/295) 

Treewidth를 PS하는 사람의 관점에서 볼 때 가장 이해하기 쉽게 써놓은 글입니다. 뿐만아니라 bounded tw graph에서 shortest path query를 해결하는 방법에 대해서 논합니다.

- Björklund, Andreas, et al. "Fourier meets Möbius: fast subset convolution." Proceedings of the thirty-ninth annual ACM symposium on Theory of computing. 2007.

Subset convolution에 대한 논문입니다.

- Adamant, [Subset convolution interpretation](https://codeforces.com/blog/entry/92153) 

subset convolution을 비롯해, 여러 형태의 convolution과 연습문제, 일반적인 이해법에 대해 다룹니다.