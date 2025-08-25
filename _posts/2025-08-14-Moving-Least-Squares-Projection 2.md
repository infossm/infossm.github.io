---
layout: post
title: "Moving Least Squares Projection 2 - Optimization"
date: 2025-08-11
author: mhy908
tags: [Point-Set-Surfaces, algorithm]
---

## 지난 글
[Moving-Least-Squares-Projection](https://infossm.github.io/blog/2024/08/11/Moving-Least-Squares-Projection/)


## 기존 MLS Projection의 한계

지난 글의 결론 부분에서 간략하게 짚어보았지만, 전에 설명한 방법은 성능적으로도, 그리고 컴퓨팅적으로도 한계가 명확했습니다.

첫째, 측정하고자 하는 물체의 뾰족한 부분이 제대로 표현되지 않고 뭉툭한 형태로 구해집니다. 이는 근본적으로 기존 MLS Projection은 smoothing 기반 기법이기 때문입니다. 아래 정리해본 기존 MLS Projection의 로직을 생각해보면 당연한 결과입니다.

- 뾰족한 부분에서는 하나의 뚜렷한 Local Approximating Hyperplane이 존재하지 않고, 여러 개의 근사 평면이 서로 교차하게 됩니다. 따라서 알고리즘이 도출하는 결과는 이들 사이의 ‘평균’ 평면이 되며, 그 과정에서 원래 물체가 가진 뾰족한 특성이 사라지고 둔화되는 효과가 나타납니다.
- regression 단계에서 다항함수 등의 매끈한 함수를 사용해, 기존 물체가 가지고 있던 불연속적인 성질이 근사됩니다.

둘째, Projection 1회당 드는 연산량이 $O(N)$입니다. 매 사영마다 모든 점에 대한 가중치를 매기기 때문입니다. 혹자는 이를 간단하게, 사영하고자 하는 점의 K-Nearest-Neighbours만 고려해서 사영하는 아이디어를 떠올릴 수 있습니다. 하지만 이 방법을 통해 만든 MLS Hypersurface은 불연속하다는 아주 큰 단점이 있습니다. 기존에 사용한 연산들은 모두 미분 가능한 연산인 반면, K-Nearest-Neighbours를 구하는 과정에서 불연속성이 주입되기 때문입니다.

![위 사진은 20-Nearest-Neighbours를 적용한 결과입니다. 결과로 구한 면이 불연속적임을 확인할 수 있습니다.](/assets/images/Moving-Least-Squares-Projection/pic7.png)


## 개선

첫번째 문제점은, Local Approximating Hyperplane 대신 Local Approximating Sphere를 사용하는 것으로 어느정도 해결할 수 있음이 알려져 있습니다. 사영하는 면 자체를 곡선으로 두어, 뾰족한 부위를 처리함에 있어 기존보다 더 나은 성능을 보입니다.

또한, 물체를 측정할때 기존에는 점의 위치정보만 사용했는데, 같이 측정할 수 있는 메타데이터 중 하나인 면의 법선 정보와, 측정한 영역의 넓이를 활용하여, 이를 가중치로 반영해 뾰족한 부위에서의 모서리 방향을 알려주는 힌트로 사용할 수 있습니다. 

![참고로, 점 정보와 함께 이런 정보까지 주어진 형태를 Surfel이라고 합니다.](/assets/images/Moving-Least-Squares-Projection/pic9.png)

두번째 문제점은 KNN을 사용하는 대신, 다른 근사 기법을 사용하여 실제와 매우 유사한 결과를 얻는것으로 해결할 수 있습니다. 정밀도와 연산 속도 사이의 tradeoff라고 생각할 수 있습니다. 그러나 이 경우 우리가 구하고자 하는 값은 결국 점 정보들에서 추출한, 어떤 closed form 결과에 불과하므로, 여기서 말하는 '정밀도'의 의미는 퇴색되긴 합니다. 

## Moving Level-Of-Detail Projection

Moving Level‑Of‑Detail Projection (이하 MLOD Projection 또는 MLOD 사영)은 앞서 언급한 첫 번째 문제점을 개선하는 동시에, 근사 기법을 활용해 연산 속도를 $O(log N)$ 수준으로 끌어올린 알고리즘입니다.
이번 포스트에서는 MLOD Projection의 구체적인 동작 방식과 함께, 고차원 기하학에도 적용할 수 있는 다양한 아이디어들을 하나씩 짚어가며 살펴보겠습니다.

### Definitions

입력은 다음과 같이 정의할 수 있습니다. 3차원 공간의 점 입력을 가정하겠습니다.

- $p_i$ = $i$번째 점의 위치벡터
- $n_i$ = $i$번째 점의 정규화된 법선벡터
- $\sigma _i$ = $i$번째 점이 근사하는 영역의 반지름
- $q$ = 사영하고자 하는 점의 위치벡터
- $w_i$ = 사영하고자 하는 점에 대한 $i$번째 점의 가중치 값

일반적으로 $w_i=\theta(q, p_i)=\theta(\Vert q-p_i \Vert)$를 사용합니다. (지난 글 참고)

또한, 구하고자 하는 Local Approximating Sphere은 다음과 같이 정의할 수 있습니다.

$S(x, y, z) = (u_0, u_1, u_2, u_3, u_4)\cdot(1, x, y, z, (x^2+y^2+z^2))=0$

이렇게 정의한다면, 점 집합에 대해 잘 fit 된 Sphere는 다음 성질을 만족하게 할 수 있습니다.

$S(p_i) \approx 0, \nabla S(p_i) \approx n_i $

### Formula

자세한 유도과정은 이 글에서는 생략하겠습니다. 결과는 다음과 같습니다.

$$
u_4 = 
\frac{
    \sum_i w_i \sigma_i \mathbf{p}_i \cdot \mathbf{n}_i 
    - \sum_i w_i \sigma_i \mathbf{p}_i \cdot \sum_i w_i \sigma_i \mathbf{n}_i / \sum_i w_i \sigma_i
}{
    \sum_i w_i \sigma_i \|\mathbf{p}_i\|^2 
    - \|\sum_i w_i \sigma_i \mathbf{p}_i\|^2 / \sum_i w_i \sigma_i
}, \\[2mm]
\mathbf{u}_{123} = 
\frac{
    \sum_i w_i \sigma_i \mathbf{n}_i - 2 \mathbf{u}_4 \sum_i w_i \sigma_i \mathbf{p}_i
}{
    \sum_i w_i \sigma_i
}, \\[1mm]
u_0 = 
- \frac{
    \sum_i w_i \sigma_i \mathbf{p}_i \cdot \mathbf{u}_{123} + \mathbf{u}_4 \sum_i w_i \sigma_i \|\mathbf{p}_i\|^2
}{
    \sum_i w_i \sigma_i
}
$$

### Idea

위 식을 그대로 계산하려 하면, $O(N)$으로 너무 느립니다. 그래서 MLOD는 다음 사실에 기반해 근사를 시도합니다.

$w_i$가 비슷한 점 집합 $R$에 대해, $w$가 그 $w_i$들의 대표값이라면 $\sum_{i \in R} w_iF(i) \approx w \sum_{i \in R} F(i)$ 를 만족합니다.

위 공식을 유심히 관찰하면, 계산이 필요한 항들은 모두 위의 형식을 갖추고 있습니다. 즉, 점들을 적절히 분할하여 각 집합에 대해 전처리로 필요한 $\sum_{i \in R} F(i)$를 계산해두면, 1회의 사영에 드는 시간복잡도를 $O(분할한 집합의 수)$로 최적화를 할 수 있습니다.

MLOD는 이 아이디어에 octree를 결합합니다.

### Algorithm

먼저, octree 각 노드 $\eta$에 대해 $\delta _{node}(\eta)$를 계산하고자 합니다.

$$
\delta_{\text{node}}(\eta) = 
\begin{cases}
\sigma = \sum_{i\in \eta} \sigma_i \in \mathbb{R} \\[6pt]
\mathbf{n}_{\alpha} = \sum_{i\in \eta} \sigma_i \mathbf{n}_i \in \mathbb{R}^3 \\[6pt]
pn_{\beta} = \sum_{i\in \eta} \sigma_i \mathbf{p}_i \cdot \mathbf{n}_i \in \mathbb{R}  \\[6pt]
\mathbf{p}_{\alpha} = \sum_{i\in \eta} \sigma_i \mathbf{p}_i \in \mathbb{R}^3 \\[6pt]
p_{\beta} = \sum_{i\in \eta} \sigma_i \| \mathbf{p}_i \|^2 \in \mathbb{R}
\end{cases}
$$

$\delta _{node}(\eta)$ 의 값은 $q$에 무관하며, 단순히 octree 내부 점들에 해당하는 정보들의 합에 불과합니다. 따라서 이는 Tree DP 와 비슷한 방식으로 $O(N)$ 에 전처리할 수 있습니다.

위에서 언급한 근사 아이디어를 채용하기 위해서, 노드 $\eta$에 대한 가중치의 대푯값 $w_{\eta}$를 다음과 같이 정의합니다.

$w_{\eta} = \theta(q, \mathbf{p}_{\alpha}(\eta)/\sigma (\eta))$

여기서 $\mathbf{p}_{\alpha}(\eta)/ \sigma(\eta)$ 는 정의상 $\eta$에 있는 점들의 크기를 고려한 평균위치에 해당합니다.

입력 $q$에 대한 Local Approximating Sphere를 구하는데 필요한 값 $\delta(\eta, q)$는 다음과 같습니다.

$$
\delta(\eta, q) = 
\begin{cases}
\sigma = \sum_{i\in \eta} w_i\sigma_i \in \mathbb{R} \\[6pt]
\mathbf{n}_{\alpha} = \sum_{i\in \eta} w_i\sigma_i \mathbf{n}_i \in \mathbb{R}^3 \\[6pt]
pn_{\beta} = \sum_{i\in \eta} w_i\sigma_i \mathbf{p}_i \cdot \mathbf{n}_i \in \mathbb{R}  \\[6pt]
\mathbf{p}_{\alpha} = \sum_{i\in \eta} w_i\sigma_i \mathbf{p}_i \in \mathbb{R}^3 \\[6pt]
p_{\beta} = \sum_{i\in \eta} w_i\sigma_i \| \mathbf{p}_i \|^2 \in \mathbb{R}
\end{cases}
$$

$w_i$들의 값이 비슷하다면, $\delta(\eta, q) \approx w_{\eta}\delta_{node}(\eta) = \delta_{approx}(\eta, q)$ 를 만족한다는 사실을 사용하고자 합니다.

octree의 루트부터 시작해 다음의 탐색 알고리즘을 사용합니다. 현재 노드의 모든 자식 노드들에 대해 다음의 값들을 전부 더한 뒤, 그 값을 리턴할 것입니다.

먼저 각 영역에 대해, 영역에 외접하면서 영역을 전부 포함하는 작은 구를, $2> \lambda > 1$만큼의 비율로 확대한 영역을 가정합니다.

만약 $q$가 이 영역 내부에 있다면, 그대로 재귀적으로 탐색을 진행해, 그 결과를 답에 더합니다.
만약 $q$가 영역 외부에 있다면, Hierarchical Transition Function(HTF)이라 불리는, 일종의 smoothing function을 $\delta_{approx}(\eta, q)$에 곱해 답에 더합니다. HTF는 구에 가까울수록 1, 멀어질수록 0이 되는 미분 가능한 임의의 함수이며, 이로 인해 전체 결과가 $q$에 대해 연속적이게 됩니다.

![HTF를 적용하지 않는다면, octree의 경계부에서 불연속이 나타납니다.](/assets/images/Moving-Least-Squares-Projection/pic11.png)

![예시 HTF의 그래프입니다.](/assets/images/Moving-Least-Squares-Projection/pic10.png)



이 알고리즘은 결국 전처리 $O(N)$ 시간을 통해, 쿼리로 들어오는 $q$에 대해 그 MLOS를 $O(logN)$ 시간에 구할 수 있습니다. $\lambda$가 충분히 작기 때문에, 이로 인한 오버헤드의 증가는 미미합니다.

## Reference

[Corentin Mercier, Thibault Lescoat, Pierre Roussillon, Tamy Boubekeur, and Jean-Marc Thiery. 2022. Moving level-of-detail surfaces.]

https://cgl.ethz.ch/publications/tutorials/eg2002

[Levin, David. "Mesh-independent surface interpolation." Geometric modeling for scientific visualization. Springer Berlin Heidelberg, 2004.]



