---
layout: post
title: "Near-Linear time Laplacian Equation Solver"
author: TAMREF
date: 2023-03-19
tags: [graph-theory, linear-algebra]
---

# Introduction

Graph $G$의 Laplacian은 많은 graph problem에 관여하는 중요한 object입니다. 과거 [Matrix-Tree theorem에 대해 다룬 글](https://infossm.github.io/blog/2021/04/18/resistors/)에서 볼 수 있듯 Laplacian의 determinant는 $G$의 스패닝 트리 개수와도 관계가 있고, 오늘 다룰 Laplacian system을 푸는 것도 굉장히 중요한 문제입니다. 또한 Laplacian의 고유값(eigenvalue)들 또한 graph의 well-connectedness와 관련이 있는 유의미한 지표로 사용할 수 있습니다.

정점이 $n$개, 간선이 $m$개인 무방향 그래프 $G$의 Laplacian $L _ {G}$는 $D - A$로 정의합니다. 여기서 $D$는 $D _ {i, i} = \deg(i)$를 만족하는 대각행렬이고, $A$는 $G$의 인접행렬입니다. 가령 완전그래프 $K _ {3}$의 Laplacian은 $\begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2\end{pmatrix}$로 쓸 수 있습니다. 여기서 Laplacian이 만족하는 일반적인 성질을 증명할 수 있습니다.

- $L$은 symmetric, positive semi-definite matrix입니다. 즉, 모든 벡터 $x \in \mathbb{R}^{n}$에 대해 $x^{T}Lx \ge0$이 성립합니다.
- row-sum, column-sum이 항상 $0$입니다. 즉 one-vector $\mathbf{1} \in \mathbb{R}^{n}$에 대해 $L\mathbf{1} = 0$입니다.
- 그래프가 connected라면, $\mathrm{ker}(L) = \mathrm{span}(\mathbf{1})$입니다.
  - 앞으로 항상 $G$는 connected인 것으로 가정합니다.
  - 따라서, $b \perp \mathbf{1}$이라면 $Lx = b$의 solution $x \perp \mathbf{1}$를 항상 유일하게 찾을 수 있습니다.
- 그래프의 각 간선에 양의 실수로 가중치가 주어지면, degree는 그 가중치의 합으로 정의합니다. 당연히 가중치가 주어진 그래프에서도 위의 성질은 그대로 성립합니다.

$Lx = b$라는 system of equation을 푸는 것이 왜 중요할까요? 예제를 통해서 알아보겠습니다.

## Hitting time

그래프에서의 Random walk를 생각해봅시다. 내가 정점 $a$에 있을 때 다른 정점 $b$로 이동할 확률이 간선 $ab$의 가중치 $A _ {ab}$에 비례한다고 합시다. 이 확률은 정확히 $A _ {ab} / D _ {aa}$로 나타낼 수 있기에, 정점 $a$에서 다른 정점 $b$로 넘어갈 확률을 나타내는 transition matrix를 $W := AD^{-1}$로 쓸 수 있습니다.

이 때 시작 정점 $s$가 주어져 있고, 다른 정점 $t \neq s$에 대해 처음으로 $t$에 도달하기 까지 걸리는 시각을 hitting time $H _ {s, t}$로 정의합시다. Hitting time의 기댓값은 어떻게 구할 수 있을까요?

편의상 $h := (\mathbb{E}H _ {s,1}, \cdots, \mathbb{E}H _ {s, n}) \in \mathbb{R}^{n}$으로 정의합니다. $h _ {s} = 0$이 자명하고, $t \neq s$에 대해 다음의 recurrence relation이 성립합니다.

$$
h _ {t} = 1 + \sum _ {u = 1}^{n} W _ {ut}h _ {u}
$$

이를 모든 $t \neq s$에 대해 나열해주면, $h _ {s} = 0$인 것과 함께 나열하여 다음의 식을 얻습니다.

$$
h - (1 + \left<\mathbf{e} _ {s},  W^{T}h\right>)\mathbf{e} _ {s} = \mathbf{1} + W^{T}h
$$

이 때 $\alpha := (1 + \left<\mathbf{e} _ {s},  W^{T}h\right>)$라 두면 $(I - W^{T})h = \mathbf{1} - \alpha \mathbf{e} _ {s}$를 해결해야 하고, 이는 곧 $Lh = D(\mathbf{1} - \alpha \mathbf{e} _ {s})$를 푸는 것과 같습니다.

이 때 아직 $\alpha$ 또한 $h$에 의존하는 변수이기 때문에 resolve해줘야 하는데, $D(1 - \alpha \mathbf{e} _ {s}) \in \mathrm{im}(L) = \mathrm{span}(\mathbf{1})^{\perp}$을 만족하려면 $\sum _ {i} \deg(i) = \alpha \deg(s)$를 만족해야 합니다. 여기서 $\alpha = \frac{\sum _ {i} \deg(i)}{\deg(s)}$를 얻습니다. 

따라서 $b :=D(1 - \alpha \mathbf{e} _ {s})$라고 두면, 우리에게 남은 것은 $Lh = b$ 라는 Laplacian equation을 푸는 것입니다. 이와 관련한 문제로 [BOJ 18451. Expected Value](https://www.acmicpc.net/problem/18451)를 알 수 있습니다.

## Solving Sparse Laplacian Equation

Laplacian Equation $Lx = d$를 푸는 방법은 어떤 것이 있을까요? 일반적으로 $G$는 connected, $d \perp \mathbf{1}$을 가정합시다.

정점 수인 $n$이 작은 경우에는 $\mathcal{O}(n^3)$ 정도의 시간을 들여 directly solve하는 방법이 있겠습니다. 열심히 행렬곱을 최적화해서 $\widetilde{\mathcal{O}}(n^{\omega})$ 정도에 푸는 방법이 있겠지만 논외로 합시다.

$n$이 더 큰 경우에는 sparse graph인 경우, 즉 $m = \widetilde{O}(n)$ 정도인 경우에 관심이 있습니다. 여기서 $\widetilde{O}$는 $n$의 poly-log factor를 감추는 big-O notation의 변형입니다.

$G$가 sparse한 경우 $L$도 sparse matrix가 됩니다. 어떤 행렬 $M$의 non-zero entry의 개수를 $\mathrm{nnz}(M)$으로 표기할 때, $\mathrm{nnz}(L) = O(m + n)$이 됩니다. 이러한 경우 Sparse-linear system을 푸는 방법을 적용하면 $O(mn)$ 시간에 문제를 해결할 수 있습니다. 이는 앞서 언급한 Expected Value 문제의 풀이이기도 합니다.

여기에 더해 Laplacian의 성질을 활용하면 $\widetilde{O}(m)$ 시간에 문제를 해결할 수 있고, 이 사실은 현대 그래프 알고리즘 분야에 지대한 영향을 낳았습니다. 처음으로 Near-Linear Time algorithm을 제시한 Spielman-Teng의 경우 1000회에 육박하는 인용수를 기록하고 있으며, 비슷한 approach를 채용한 알고리즘들이 수많은 complexity barrier를 갱신하였습니다.

오늘은 그 중에서도 높은 확률로 $O(m \log^{3} n)$ 시간 안에 Laplacian Equation의 approximation solution을 구해주는 Kyng (2015)의 논문을 리뷰합니다.

## Exit

앞으로 이어질 본문의 내용은 상당히 길고 복잡하기 때문에, 여기까지만 읽고 싶으신 분들을 위한 해당 논문의 결과 statement를 이 문단에서 해설하려고 합니다. "높은 확률로", "approximation solution"이라는 말은 그 자체로는 모호하기 때문입니다.

- 알고리즘은 "높은 확률로" $O(m\log^{3} n)$ 시간 안에 terminate합니다. $m \log^{3} n$은 어떤 sparse matrix의 $X$의 nonzero term의 개수와 관련되어 있고, $\mathrm{nnz}(X) > Cm\log^{3} n$일 확률이 $O(n^{-5})$ 정도로 bound되어 있습니다. 그와 별개로 approximation quality는 보장됩니다.
- $Lx = b$의 유일한 (up to translation along $\mathbf{1}$) solution을 $x^{\ast}$라고 할 때, 알고리즘은 $O(m\log^{3} n \log (1 / \varepsilon))$ 시간 안에 $\lVert x - x^{\ast} \rVert _ {L} < \varepsilon \lVert x^{\ast} \rVert _ {L}$을 만족하는 $x$를 찾을 수 있습니다. 여기서 norm $\lVert \cdot \rVert _ {L}$은 $\lVert x \rVert _ {L} := \left(x^{T} L x\right)^{1/2}$으로 정의하고, norm으로의 조건을 모두 만족합니다. 시간 복잡도의 leading term $m \log^{3} n$은 위에서 언급한 $\mathrm{nnz}(X)$에 의존합니다.

# Optimziation Part

## Notations

다음의 notation들을 가정합니다.

- PSD: positive semidefinite, PD: positive definite
- For PSD matrices $A, B$, $A \le B$ if $B - A$ is PSD. $A < B$ if $B - A$ is PD
- For a symmetric matrix $A$, $\lVert A \rVert := \sup _ {x^{T}x  =1} \left(x^{T}Ax\right) = \max _ {i} \lvert \lambda _ {i} \rvert$ where $\lambda _ {1} \le \cdots \le \lambda _ {n}$

## Preconditioned System Solver

PSD matrix $A, B$에 대해 $\frac{1}{1 + K}A < B < (1 + K)A$이면 $A \approx _ {K} B$로 표기하고,
어떤 matrix $M$에 대해 위에서 말한것처럼  $\lVert x - x^{\ast} \rVert _ {M} < \varepsilon \lVert x^{\ast} \rVert _ {M}$을 만족하는 $x$를 $\varepsilon$-approximator of $x^{\ast}$라고 부릅니다. 이 때

**Theorem.** PD matrix $M$에 대해 $M \approx _ {K} \mathcal{L}\mathcal{L}^{T}$인 $\mathcal{L}$이 주어져 있다고 하자. 이 때 $Mx = d$의 $\varepsilon$-approximating solution $\widetilde{x}$를 $(1 + K)\log(K / \varepsilon)(T _ {1}(n) + T _ {2}(n) + n)$ 시간에 찾을 수 있다.

이 때,
- $T _ {1}(n) := \mathrm{Time}\left\lbrace z \mapsto Mz \right\rbrace$ for arbitrary $z \in \mathbb{R}^{n}$.
  - 일반적으로 $O(n^2)$이나 $M$이 sparse한 경우 줄일 수 있는 항입니다.
- $T _ {2}(n) := \mathrm{Time}\left\lbrace z \mapsto \mathcal{L}^{-1} z\right\rbrace$ for arbitrary $z \in \mathbb{R}^{n}$.

*Proof.*

정의상 모든 $x$에 대해 $\frac{x^{T} M x}{x^{T} \mathcal{L}\mathcal{L}^{T} x} \in \left[ \frac{1}{1+K}, 1 + K \right]$가 성립합니다. $x = (\mathcal{L}^{T})^{-1}y$로 두면, $M^{\prime} := \mathcal{L}^{-1}M(\mathcal{L}^{T})^{-1}$의 eigenvalue가 $\left[ \frac{1}{1+K}, 1 + K \right]$에 존재하게 됩니다. 이 때 $M^{\prime}$의 condition number가 $(1 + K)^{2}$이므로, 이 때 $M^{\prime}y = \mathcal{L}^{-1} d$를 $O((1 + K)\log(K / \varepsilon))(n + T _ {1}(n) + T _ {2}(n))$ 시간에 Chebyshev method, Conjugate Gradient 등을 사용하여 $\varepsilon$-approximate할 수 있습니다. $\square$

따라서 PD matrix $L$에 대해서 $L \approx _ {K}  \mathcal{L}\mathcal{L}^{T}$인 sparse **lower triangular** matrix $\mathcal{L}$을 찾기만 한다면 $Lx = d$를 빠르게 해결할 수 있습니다. 일반적으로 Lower(Upper) triangular matrix $L$에 대해서는 $T _ {2}(n) = \mathcal{O}(\mathrm{nnz}(\mathcal{L}))$ 이 성립하기 때문입니다. 하지만 $L$이 invertible이 아니기 때문에 사소한 문제가 발생하는데, 이는 큰 무리 없이 메울 수 있습니다.

**Refined Theorem.** PSD matrix $M$에 대해 $M \approx _ {K} \mathcal{L}\mathcal{D}\mathcal{L}^{T}$인 **invertible** $\mathcal{L}$이 주어져 있다고 하자. 이 때 $Mx = d$의 $\varepsilon$-approximating solution $\widetilde{x}$를 $(1 + K)\log(K / \varepsilon)(T _ {1}(n) + T _ {2}(n) + T _ {3}(n) + n)$ 시간에 찾을 수 있다.

이 때,
- $T _ {1}(n) := \mathrm{Time}\left\lbrace z \mapsto Mz \right\rbrace$ for arbitrary $z \in \mathbb{R}^{n}$.
  - 일반적으로 $O(n^2)$이나 $M$이 sparse한 경우 줄일 수 있는 항입니다.
- $T _ {2}(n) := \max\left(\mathrm{Time}\left\lbrace z \mapsto \mathcal{L}^{-1} z\right\rbrace, \mathrm{Time}\left\lbrace z \mapsto \mathcal{D}^{+} z\right\rbrace\right)$ for arbitrary $z \in \mathbb{R}^{n}$.
  - $\mathcal{D}^{+}$는 $\mathcal{D}$의 Moore-Penrose Pseudo inverse.
- $T _ {3}(n) := \mathrm{Time}\left\lbrace z \mapsto \Pi _ {M} z\right\rbrace$.
  - $\Pi _ {M}$은 $\mathrm{ker}(M)^{\perp}$으로의 projection matrix.


Laplacian의 경우 $\Pi _ {L}(x) = x - \frac{1}{n}\mathbf{1}(\mathbf{1}^{T}x)$로 $T _ {3}(n)= O(n)$이 성립하고, $\mathcal{L}$과 $\mathcal{D}$를 잘 조정하여 PD case와 거의 동일하게 문제를 해결할 수 있습니다.

# Preconditioning Part

우리의 목표는 다음 Theorem을 증명하는 것입니다.

**Theorem (Kyng 2015).** Weighted graph Laplacian $L$에 대해, $L \approx _ {0.5} \mathcal{L}\mathcal{L}^{T}$를 만족하는 lower triangular matrix $\mathcal{L}$을 찾을 수 있다. (Especailly, $\mathcal{L} _ {n, :} = 0)$ 이 때 $\mathrm{nnz}(\mathcal{L}) = O(m \log^{3} n)$이 $1 - O(n^{-5})$의 확률로 성립하며, 동일한 확률로 이 알고리즘은 $O(m \log^{3} n)$ 안에 terminate한다.

이 Theorem을 증명할 수 있으면 조건을 만족하는 $\mathcal{L}$을 찾은 뒤, **Refined Theorem**에 대입하기 위해 $(n, n)$에만 $1$을 대입한 matrix $\widetilde{L}$과, $I _ {n}$에서 $(n, n)$만 $0$인 matrix $\mathcal{D}$를 이용하여 $\mathcal{L}\mathcal{L}^{T} = \widetilde{L}\mathcal{D}\widetilde{L}^{T}$가 성립하도록 할 수 있기 때문에 전체 문제인 Laplacian Solver 찾기를 해결할 수 있습니다.

## Exact Approach

$L = \mathcal{L}\mathcal{L}^{T}$를 정확히 만족하는 $\mathcal{L}$을 찾는 방법은 Cholesky Decomposition이라고 하여 이미 널리 알려져 있습니다. 구체적으로 다음과 같은 방법을 사용합니다:

- $S _ {0} := L$.
- For $1 \le i < n$.
  - $l _ {i} := S _ {i-1}(i, i)^{-1/2} \cdot S _ {i-1}(:, i)$. (Rescaled column)
  - $S _ {i} = S _ {i-1} - l _ {i}l _ {i}^{T}$.
- $l _ {n} = 0$.
- $\mathcal{L} := (l _ {1}, \cdots, l _ {n})$

이 때 $\mathcal{L}$이 lower triangular가 됨을 Laplacian의 성질로부터 알 수 있습니다. $l _ {i}l _ {i}^{T}$의 의미를 breakdown하기 위해 다음의 Lemma를 먼저 state합니다.

**Lemma. (Characterization of Laplacian)** PSD matrix $M$이 다음 조건을 만족하면 $M$을 Laplacian으로 갖는 그래프가 존재한다.
- $M \mathbf{1} = 0$.
- $M$의 diagonal entry는 nonnegative.
- $M$의 off-diagonal entry는 nonpositive.

이제 $l _ {i}l _ {i}^{T}$를 두 Laplacian의 선형 결합으로 나타낼 수 있습니다. Laplacian $L$로 characterize되는 그래프에 대해 $\bigstar(v, L)$을 정점 $v$와 그 인접한 간선들만으로 induce되는 subgraph의 Laplacian으로 정의하고, $\Delta(v, L) := \bigstar(v, L) - \frac{1}{L(v, v)}L(v, :)L(v, :)^{T}$로 정의합시다. 이 때 $\Delta(v, L)$은 새로운 그래프의 Laplacian (Y-Delta Transformation)이 됩니다.

이렇게 쓰고 나면 앞선 알고리즘에서 $l _ {i}l _ {i}^{T} = \bigstar(i, S _ {i-1}) - \Delta(i, S _ {i-1})$으로 나타낼 수 있습니다. 여담으로 $\Delta(i, S _ {i-1})$은 in general non-sparse하기에 $S _ {i}$의 sparseness를 망치는 주범이 됩니다. 따라서 $\Delta(i, S _ {i-1})$을 적절한 Sampling으로 대체하자는 생각이 떠오르게 됩니다.

## Sampling approach

따라서 다음의 alternative approach를 생각합시다.

- $S _ {0} := L$.
- Take a random permutation $\pi$ of $[n]$.
- For $1 \le i < n$.
  - $l _ {i} := S _ {i-1}(\pi(i), \pi(i))^{-1/2} \cdot S _ {i-1}(:, \pi(i))$. (Rescaled column)
  - $S _ {i} = S _ {i-1} - \bigstar(\pi(i), S _ {i-1}) + \nabla(\pi(i), S _ {i-1})$.
    - $\nabla(\pi(i), S _ {i-1})$ is a sparse sampled matrix replacing $\Delta(\pi(i), S _ {i-1})$.
- $l _ {n} = 0$.
- $\mathcal{L} := (l _ {1}, \cdots, l _ {n})$

생각한 것까진 좋은데, 이제 $\nabla$가 $\Delta$를 얼마나 가깝게 묘사하는지, 이에 따라 $\mathcal{L}\mathcal{L}^{T}$와 Laplacian $L$이 얼마나 멀어질지를 생각해야 합니다. 우선 $\nabla$의 구체적인 형태부터 하나 잡아봅시다.

$\nabla(v, L)$은 다음과 같은 random process를 통해 계산합니다.

- $Y _ {v} := 0$
- 모든 Multi-edge $e = (v, i)$ (if clarified) of $L$에 대해,
  - Neighbor $j$를 $-L _ {vj} / L _ {vv}$의 확률로 뽑고
  - $i \neq j$인 경우 $Y _ {v} \leftarrow Y _ {v} - \frac{L _ {vj}L _ {vi}}{L _ {vj} + L _ {vi}} b _ {ij}b _ {ij}^{T}$
    - 단, $b _ {ij} := \mathbf{e} _ {i} - \mathbf{e} _ {j}$.
- `return` $Y _ {v}$

결국 $\Delta(v, L)$을 재현하는 데 충실하게, $v$의 두 neighbor $i \neq j$를 sparse sampling하는 과정입니다. 가중치로 붙은 $-\frac{L _ {vj}L _ {vi}}{L _ {vj} + L _ {vi}}$는 기댓값을 맞추기 위한 amplifying term으로 볼 수 있겠습니다.

Multi-edge는 weight와는 별개의 개념으로, 이를 어떻게 잡는지는 추후에 언급하겠습니다. 지금은 그냥 "모든 neighbor $i$에 대해 일정 횟수만큼 sampling을 반복한다" 정도로 생각해도 충분합니다. neighbor $j$의 sampling은 Alias method 등을 이용하여 $O(\deg _ {L}(v))$ init, $O(1)$ query로 수행할 수 있습니다.

**Lemma.** $\nabla(v, L)$는 Laplacian이고, $\mathbb{E}\nabla(v, L) = \Delta(v, L)$.

*Proof.* Laplacian임은 sampling process에 의해 자명하고, $v$의 두 neighbor $i \neq j$에 대해 entry $(i, j)$의 기댓값은 $-\left(-\dfrac{L _ {vj}L _ {vi}}{L _ {vj} + L _ {vi}} \cdot -\left(\frac{L _ {iv}}{L _ {vv}} + \frac{L _ {jv}}{L _ {vv}}\right)\right) = -\dfrac{L _ {vj}L _ {vi}}{L _ {vv}} = \Delta(v, L) _ {ij}$가 됩니다. $\square$

## Bounding The Spectral Norm

이후의 부분은 문제 자체의 디테일을 resolve하는 과정이고, 수학적 성격이 너무 짙기 때문에 아주 개략적으로만 다루도록 하겠습니다.

결국 우리는 $0.5L \le \mathcal{L}\mathcal{L}^{T} \le 1.5L$이 높은 확률로 성립하는 것을 보여야 합니다. $\Phi(A) := (L^{+})^{1/2}A(L^{+})^{1/2}$로 쓰고 식을 정리해주면, $\lVert \Phi(\mathcal{L}\mathcal{L}^{T} - L) \rVert \le 0.5$ 가 성립해야 합니다. 일반적으로 matrix spectral norm이 작다는 statement (matrix concentration) 로 알려진 것은 Bernstein's matrix concentration inequality가 있습니다.

**Theorem. (Bernstein)** $X _ {1}, \cdots, X _ {n}$이 independent random matrix이고, $\lVert X _ {i} \rVert \le R$, $\mathbb{E}X _ {i} = 0$을 가정하자. $X := \sum _ {i} X _ {i}$라고 두고 $\sigma^{2} := \lVert \mathbb{E}X^{2} \rVert$이라고 두면, 다음 식이 성립한다.

$$
\mathrm{Pr}[\lVert X \rVert \ge t] \le 2n\exp\left({-\dfrac{t^2}{2Rt + 4\sigma^{2}}}\right)
$$

결국 $X _ {i}$의 spectral norm, $X$의 variance를 잘 control하면 $X$의 spectral norm 또한 높은 확률로 guarantee할 수 있다는 것입니다. 일반적으로 우리가 만든 $\nabla(v, L)$의 경우 independent하지 않지만 한편으로는 Martingale 조건을 만족합니다. 따라서
- Martingale 조건에 대해 비슷한 theorem을 state하고
- $\nabla(v, L)$을 잘 조작한 뒤
- Hyperparameter tuning을 통해 원래의 spectral norm, variance간의 tradeoff resolve

의 순서로 진행하게 됩니다.

우리의 alternative approach에서 얻은 $S _ {i}, l _ {i}$에 대해, $L _ {i} := S _ {i} + l _ {1}l _ {1}^{T} + \cdots + l _ {i}l _ {i}^{T}$를 생각합시다. $L _ {0} := S _ {0} = L$로 정의합니다. $L _ {n} = \mathcal{L}\mathcal{L}^{T}$가 될 것이니 이론상 $L _ {n}$이 $L _ {0}$에서 너무 많이 벗어나면 안됩니다.

이 때 $\nabla(\pi(i), S _ {i-1})$을 결정하기 전까지의 random variable ($\pi(i)$, 그리고 $S _ {i-1}$을 결정하기 위한 모든 random variable들)이 결정되었을 때의 conditional expectation of $L _ {i}$를 $\mathbb{E}[L _ {i} \mid \nabla _ {i}]$라고 쓰면, (설명한 condition을 $\nabla _ {i}$라고 통용) $\mathbb{E}[L _ {i} \mid \nabla _ {i}] = L _ {i-1}$이 성립하게 됩니다. $L _ {i-1}$은 이미 결정되어 있는 값임에 주목하세요. 즉, $L _ {i}$는 martingale이 됩니다. 증명은 크게 비자명한 부분이 없으므로 생략하겠습니다.

Martingale difference sequence $L _ {i} - L _ {i-1}$을 생각해보면 $l _ {i}l _ {i}^{T} + S _ {i} - S _ {i-1} = l _ {i}l _ {i}^{T} +\nabla(\pi(i), S _ {i-1}) - \bigstar(\pi(i), S _ {i-1}) = \nabla(\pi(i), S _ {i-1}) - \Delta(\pi(i), S _ {i-1}) = \nabla(\pi(i), S _ {i-1}) - \mathbb{E}[\nabla(\pi(i), S _ {i-1}) \mid \nabla _ {i}]$가 됩니다. $\nabla _ {i}$ 조건이 주어져 있을 때 각 multi edge $e$에 대해서 sampling한 edge를 $\nabla _ {\pi(i), e}$라고 두면, $X _ {\pi(i), e} = \Phi(\nabla _ {\pi(i), e} - \mathbf{E}[\nabla _ {\pi(i), e} \mid \nabla _ {i}])$가 zero mean variable이 되고, $\Phi(L _ {i} - L _ {i-1}) = \sum _ {e \in \bigstar(\pi(i), S _ {i-1})} X _ {\pi(i), e}$가 됩니다. 최종적으로 $\Phi(L _ {n}- L _ {0}) = \sum _ {i = 1}^{n} \sum _ {e \in \bigstar(\pi(i), S _ {i-1})} X _ {\pi(i), e}$로 나타낼 수 있습니다. 이 때 고정된 $i$에 대해 $X _ {\pi(i), e}$ 끼리는 independent하나, 서로 다른 $i$에 대해서는 dependent한 RV의 sum을 보게 됩니다.

Bernstein-like theorem을 증명하기 위해, 각 단위 RV $X _ {\pi(i), e}$가 bounded spectral norm을 갖는다는 것까지만 증명하겠습니다. $L _ {n} - L _ {0}$의 variance를 보이기 위해서는 이것만으로 끝나지 않고 복잡한 operator norm과 관련된 부등식을 여럿 사용해야 하기 때문입니다. 요는 $S _ {i-1}$에서 $S _ {i}$로 바뀔 때 $\bigstar(\pi(i), S _ {i-1})$의 간선들이 사라지고 $\nabla(\pi(i), S _ {i-1})$의 간선이 새로 생겨나는데, 기존 간선들의 norm이 bounded면 새로 생긴 간선들의 norm 또한 bounded라는 것입니다.

**Theorem.** $e=(u,v) \in \bigstar(\pi(i), S _ {i-1})$이 $\Phi(-L _ {uv}b _ {u,v}b _ {u,v}^{T}) \le rI$를 만족한다고 하자. $\nabla(\pi(i), S _ {i-1})$을 만드는 과정에서 새로 만들어진 간선 $e^{\prime}=(u^{\prime}, v^{\prime})$의 가중치를 $w$라고 하면 $\Phi(wb _ {u^{\prime}v^{\prime}}b _ {u^{\prime}v^{\prime}}^{T}) \le rI$ 또한 성립한다.

*Proof.* 그래프의 effective resistance $R _ {eff}(u, v) = \lVert\sqrt{L^{+}}b _ {u,v}\rVert _ {2}^{2}$이 distance임을 이용합니다. 또한 벡터 $v$에 대해 $\lVert vv^{T} \rVert = \lVert v \rVert _ {2}^{2}$라는 자명한 사실을 사용합니다.

$$
\lVert \Phi(wb _ {u^{\prime}v^{\prime}}b _ {u^{\prime}v^{\prime}}^{T}) \rVert = w \lVert \Phi(b _ {u^{\prime}v^{\prime}}b _ {u^{\prime}v^{\prime}}^{T}) \rVert = w R _ {eff}(u^{\prime}, v^{\prime}) = w(R _ {eff}(u^{\prime}, v) + R _ {eff}(v^{\prime}, v))
$$

이 때 $w^{-1} = w _ {u^{\prime}, v}^{-1} + w _ {v^{\prime}, v}^{-1}$임과 $R _ {eff}(x, v) \le \frac{1}{w _ {xv}} r$임을 생각하면 $\lVert \Phi(wb _ {u^{\prime}v^{\prime}}b _ {u^{\prime}v^{\prime}}^{T}) \rVert \le r$을 얻을 수 있습니다. $\square$

이제 $L _ {0}$의 spectral norm을 bound하는 hyperparameter $K$를 설정하고, 드디어 **multi-edge**를 잡도록 하겠습니다. 모든 간선을 $K$번 copy하여 multi edge로 만들고, 각 간선에 weight $1 / K$를 부여합니다. 이 경우 graph laplacian은 동일하나 총 $Km$개의 multi edge가 존재하게 되고, 각 multi edge $e$에 대해 $\Phi(w(e)b _ {e}b _ {e}^{T}) \le \frac{1}{K}I$가 성립하게 됩니다. 이러한 $K$는 언급하지 않은 sample variance등을 고려하여 약 $200\log^{2} n$으로 설정하게 됩니다.

alternative approach를 따라가는 과정에서, 그래프에 고를 수 있는 multi edge는 평균 $Km$개가 남아 있게 되고, (평균 하나가 사라지고 하나가 생겨나기 때문) $\pi(i)$를 랜덤하게 고르기 때문에 $Km / (n - i + 1)$개의 multi-degree에 대해 sampling을 하여 $\nabla(\pi(i), S _ {i-1})$을 만들어주어야 합니다. 따라서 새로이 생겨나는 간선은 총 $\sum _ {i} \frac{Km}{n - i + 1} = O(m \log^{3} n)$개가 되어, 이것이 곧 $\mathrm{nnz}(\mathcal{L})$에 대한 upper bound가 됩니다. $\square$

# Conclusion

이번 글에서는 Laplacian equation의 의미와 그 용례, 그리고 near-linear time solution에 대해 알아보았습니다. 시간 복잡도를 증명하는 부분이 매우 복잡한 것에 비해 실제 구현 상에 드는 비용은 random sampling과 적절한 gradient descent method 구현으로 크지 않은 편이라 실제 구현 및 병렬화가 가능할 것으로 예상됩니다. 구현 작업에 관심 있으신 분들은 연락 주시면 감사하겠습니다.

# References

- Kyng, Rasmus, and Sushant Sachdeva. "Approximate gaussian elimination for laplacians-fast, sparse, and simple." 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 2016.
  - 메인으로 다룬 논문입니다. 저자는 Advanced Graph Algorithm and Optimization이라는 이름의 강의를 소속 대학에서 열고 있는데, 본인의 업적인 Laplacian Equation 또한 해당 강의에서 다루고 있습니다. [강의 노트](https://kyng.inf.ethz.ch/courses/AGAO22/)를 무료로 제공하니 일독을 권합니다.
  
- Spielman, Daniel A., and Shang-Hua Teng. "Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear systems." Proceedings of the thirty-sixth annual ACM symposium on Theory of computing. 2004.
  - Spectral sparsification 등의 아이디어를 최초로 제공하고, 첫 Laplacian Equation Solver를 제시한 논문입니다. 다만 시간 복잡도가 $O(m \log^{70} n)$에 육박하는 등 simplicity와는 다소 거리가 있습니다.

- d’Aspremont, Alexandre, Damien Scieur, and Adrien Taylor. "Acceleration methods." Foundations and Trends® in Optimization 5.1-2 (2021): 1-245.
  - Accelerated Gradient Descent 등을 다룬 survey입니다. 우리가 다루는 Laplacian Equation의 경우 Quadratic function case이므로 극초반만 읽어도 좋습니다.

