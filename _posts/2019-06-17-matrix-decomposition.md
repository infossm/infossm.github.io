---
layout: post
title:  "행렬 분해(Matrix Decomposition)"
author: jihoon
date: 2019-06-15 16:22
tags: [matrix, linear algebra, decomposition]
---


# 행렬 분해(Matrix Decomposition)

행렬 분해(matrix decomposition)는 여러 특정 구조를 가진 행렬들의 곱으로 기존의 행렬을 나타내는 것을 의미합니다.

예를 들어, $$ A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}$$ 와 같은 행렬을 생각해 봅시다.

이 경우 $$ A $$를 아래와 같이 $$ 3 \times 1 $$ 크기의 두 행렬과 $$ 1 \times 1 $$ 크기의 [대각 행렬](https://ko.wikipedia.org/wiki/%EB%8C%80%EA%B0%81%ED%96%89%EB%A0%AC) 을 통해 나타낼 수 있습니다.

$$ A = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} ^ {T} $$

행렬 분해의 방법에는 LU 분해, QR 분해, 고유값 분해, 특이값 분해 등 여러 방법이 있습니다. 이 포스트에서는 여러 방법들 중 고유값 분해와 특이값 분해에 대해서 다룹니다.

## 활용

행렬 분해는 먼저 행렬의 차원을 축소할 수 있다는 장점이 있습니다. 위의 예시에서 $$ 3 \times 3 $$ 크기의 행렬을 $$ 3 \times 1 $$ 크기의 행렬과 $$ 1 \times 1 $$ 크기의 대각 행렬로 줄인 것이 대표적인 예시입니다. 차원 축소를 하게 되면, 데이터를 저장하는 데 필요한 공간의 크기를 줄일 수 있다는 장점이 있습니다.

# 고유값 분해 (Eigenvalue Decomposition)

고유값 분해에 대해 알아보기 전에 먼저 고유값(Eigenvalue)과 고유벡터(Eigenvector)에 대해서 알아봅시다. 어떠한 행렬 $$A$$가 있을 때, 벡터 $$x$$와 상수 $$\lambda$$가 있어서, $$Ax = \lambda x$$라는 식이 성립할 때, 우리는 $$\lambda$$를 A의 고유값이라고 하고, $$x$$를 $$A$$의 고유벡터라고 합니다. 고유값과 고유벡터는 유일하지 않을 수 있습니다.

예를 들어, $$ A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} $$ 인 경우에 대해 생각해봅시다.
$$ \lambda = 0, x = \begin{bmatrix} 1 \\ -1 \end{bmatrix} $$ 인 경우, $$Ax = \lambda x$$가 성립하므로, $$ \lambda $$는 고유값이 되고, $$ x $$는 고유벡터가 됩니다.

마찬가지로, $$ \lambda = 2, x = \begin{bmatrix} 1 \\ 1 \end{bmatrix} $$ 인 경우에도 $$Ax = \lambda x$$가 성립하므로, $$ \lambda $$는 고유값이 되고, $$ x $$는 고유벡터가 됩니다.

## 정의

고유값 분해는 이러한 고유값과 고유벡터들을 이용하여 행렬을 분해하는 방법입니다. 일반적으로, 정사각행렬 $$A$$의 고유벡터들을 열로 하는 행렬 $$X$$와 $$A$$의 고유값들을 대각 행렬의 값으로 하는 행렬 $$\Lambda$$을 이용하여, $$ A  = X \Lambda X^{-1} $$  과 같이 나타냅니다. 

위에서 예시로 들었던 $$ A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} $$ 에 대해서 생각해봅시다. 고유값과 고유벡터를 이용하여 $$ X = \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$$, $$\Lambda = \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix}$$ 로 놓을 수 있고, $$A = X \Lambda X^{-1}$$ 이 성립하는 것을 확인할 수 있습니다.

$$A$$가 대칭 행렬인 경우 ($$A = A^{T}$$), 고유값이 서로 다른 두 고유벡터는 항상 서로 직교해서, 두 벡터를 내적하면 영벡터가 됩니다. 그러므로 만약 norm이 1인 고유벡터들을 이용하여 행렬 $$X$$를 만들면 $$ XX^{T} = X^{T}X = I $$가 성립하고, $$A = X \Lambda X^{T}$$와 같이 표현할 수 있습니다.

## 고유값 분해의 증명

$$ n \times n$$ 크기의 정사각행렬 $$A$$가 있다고 하고, $$i = 1, 2, 3, \ldots , n$$에 대해서 $$Ax_{i} = \lambda_{i} x_{i}$$가 성립한다고 가정합시다.

$$X = \begin{bmatrix} x_{1} & x_{2} & \cdots & x_{n} \end{bmatrix} $$, $$\Lambda = \begin{bmatrix} \lambda_{1} & & & \\ & \lambda_{2} & & \\ & & \ddots & \\ & & & \lambda_{n} \end{bmatrix}$$ 로 놓는다면,

정의에 의해서 $$AX = \begin{bmatrix} \lambda_{1} x_{1} & \lambda_{2} x_{2} & \cdots & \lambda_{n} x_{n} \end{bmatrix} = X\Lambda$$이 성립합니다.

$$X$$의 역행렬이 존재한다면 (즉, invertable)한 경우에 양변의 왼쪽에 $$X^{-1}$$를 곱해주면, 원하는 식인 $$A = X \Lambda X^{-1} $$ 를 얻을 수 있습니다.

## 고유값 분해의 문제점

모든 행렬에 대해서, 고유값 분해가 가능한 것은 아닙니다. 고유값 분해는 $$X$$의 역행렬이 존재하지 않는 경우, 고유값 분해를 할 수 없다는 단점을 가지고 있습니다. 대표적으로 정사각행렬이 아닌 모든 행렬의 경우 고유값 분해가 존재하지 않고, 정사각행렬이라고 하더라도 $$X$$의 역행렬이 존재하지 않으면 고유값 분해가 불가능합니다.

고유값 분해가 불가능한 경우에는 [조르당 표준형(Jordan normal form)](https://ko.wikipedia.org/wiki/%EC%A1%B0%EB%A5%B4%EB%8B%B9_%ED%91%9C%EC%A4%80%ED%98%95)을 이용하여 유사하게 행렬 분해를 수행할 수 있지만, 많은 수학적 지식을 요구하므로, 여기서는 언급하지 않고 넘어가도록 하겠습니다.

## 고유값 분해의 활용

고유값 분해가 가능한 행렬에서, 모든 정수 $$k$$에 대해서 $$A = X \lambda^{k} X^{-1}$$이 성립한다는 것을 매우 간단하게 증명할 수 있습니다. 이러한 성질을 이용해서 행렬의 제곱을 빠른 속도로 계산할 수 있습니다. 예를 들어, $$n \times n$$ 크기의 정사각행렬 $$A$$가 있고, $$A^{k}$$을 계산하는 경우를 생각해봅시다.

먼저 고유값 분해를 사용하지 않고 계산한다면, 행렬 곱셈의 정의를 이용하여 Naive하게 접근하는 경우 행렬 곱셈을 $$k-1$$번 수행하고, 한 번의 행렬 곱셈의 시간복잡도가 $$O(n^{3})$$이므로, 전체 시간복잡도는 $$O(n^{3} k)$$이 됩니다. 좀 더 최적화를 한다면, 분할 정복을 사용하여 행렬 곱셈 횟수를 $$O(log k)$$로 줄이고, [슈트라센 알고리즘](https://ko.wikipedia.org/wiki/%EC%8A%88%ED%8A%B8%EB%9D%BC%EC%84%BC_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98) 을 사용하여 행렬 곱셈의 시간복잡도를 $$O(n^{log_{2} 7})$$ 로 줄여 전체 시간복잡도를 $$O(n^{log_{2} 7} log k)$$ 로 줄일 수 있습니다.

고유값 분해를 사용한다면 어떻게 될까요? 먼저 $$\Lambda^{k}$$를 구하는 과정의 시간복잡도는 $$O(nk)$$이 됩니다. 분할 정복을 사용하는 경우 $$O(n log k)$$까지 줄일 수 있습니다. 그 후 행렬 곱셈을 두 번만 진행하면 되므로, 위의 최적화된 경우처럼 슈트라센 알고리즘을 활용한다면 전체 시간복잡도는 $$O(n^{log_{2} 7} + n log k)$$가 됩니다. 즉, 위의 두 경우보다 훨씬 빠르게 행렬의 제곱을 구할 수 있습니다.

# 특이값 분해 (SVD, Singular Value Decomposition)

벡터 $$v, u$$ 그리고 상수 $$\sigma$$가 있을 때, $$Av = \sigma u $$가 성립하는 경우 $$v$$를 오른쪽 특이 벡터, $$u$$를 왼쪽 특이 벡터, 그리고 $$\sigma$$를 특이값이라고 부릅니다.

고유값 분해가 고유값과 고유벡터를 이용했던 것처럼, 특이값 분해는 특이값(singular value)과 왼쪽 특이 벡터(left singular vector), 그리고 오른쪽 특이 벡터(right singular vector)를 사용하여 행렬 분해를 하는 방법을 의미합니다. 행렬 $$A$$의 왼쪽 특이 벡터들을 열로 하는 행렬 $$U$$와 오른쪽 특이 벡터들을 열로 하는 행렬 $$V$$, 그리고 $$A$$의 특이값들을 대각 행렬의 값으로 하는 행렬 $$\Sigma$$을 이용하여, $$ A  = U \Sigma V^{T} $$  과 같이 나타냅니다. 이 때, $$U^{T} U = U U^{T} = V^{T} V = V V^{T} = I$$가 성립하여야 합니다.

$$A$$가 $$ n \times n$$ 크기의 정사각행렬인 경우, $$U, V, \Sigma$$의 크기는 모두 $$ n \times n$$이 됩니다. 고유값 분해와 다르게 특잇값 분해의 경우에는 정사각 행렬이 아닌 행렬에 대해서도 수행할 수 있습니다. $$A$$가 $$ m \times n (m > n)$$ 크기의 행렬인 경우,  $$U$$의 크기는 $$m \times n$$, $$V$$와 $$\Sigma$$의 크기는 $$n \times n$$가 됩니다. 0이 아닌 특이값이 $$r$$개 일 때, 특이값이 0인 경우를 생략하여 사용하는 행렬의 크기를 줄일 수 있습니다. 이 때, $$U$$의 크기는 $$m \times r$$, $$V$$의 크기는 $$n \times r$$, $$\Sigma$$의 크기는 $$r \times r$$가 됩니다. 이를 reduced SVD라고 부릅니다.


## 특이값 분해의 증명
$$m \times n$$ 크기의 행렬 $$A$$가 있다고 하고, $$i = 1, 2, 3, \ldots , n$$에 대해서 $$Av_{i} = \sigma_{i} u_{i}$$가 성립한다고 가정합시다.

$$U = \begin{bmatrix} u_{1} & u_{2} & \cdots & u_{n} \end{bmatrix} $$, $$V = \begin{bmatrix} v_{1} & v_{2} & \cdots & v_{n} \end{bmatrix} $$, 그리고 $$\Sigma = \begin{bmatrix} \sigma_{1} & & & \\ & \sigma_{2} & & \\ & & \ddots & \\ & & & \sigma_{n} \end{bmatrix}$$ ($$ \sigma_{1} \geq \sigma_{2} \geq \cdots \geq \sigma_{n}$$) 로 놓는다면,

정의에 의해서 $$AV = \begin{bmatrix} Av_{1} & Av_{2} & \cdots & Av_{n} \end{bmatrix} = \begin{bmatrix} \sigma_{1} u_{1} & \sigma_{2} u_{2} & \cdots & \sigma_{n} u_{n} \end{bmatrix} = U\Sigma$$가 성립합니다.

양변의 오른쪽에 $$V^{T}$$를 곱해주면, $$AV V^{T} = U \Sigma V^{T}$$ 를 얻을 수 있습니다. $$V V^{T} = I$$를 만족하므로 $$A = U \Sigma V^{T}$$를 만족하게 됩니다.

## 특이값과 관련된 정리들

특이값과 특이 벡터에 대해서 다음과 같은 정리들이 성립합니다:

**정리 1** : 모든 행렬 $$A$$는 특이값 분해가 존재한다. 그리고 $$u$$와 $$v$$가 단위벡터일 때, 한 특이값 $$\sigma$$에 대해서, $$u$$와 $$v$$는 유일하게 결정된다.

고유값 분해와는 다르게, 정리 1로부터 모든 행렬에 대해서 특이값 분해가 가능하다는 사실을 알 수 있습니다.

**정리 2** : $$r =rank(A)$$일 때, $$r$$은 0이 아닌 특이값의 개수이다.

**정리 3** : $$\Vert A \Vert _{2}$$ ($$L_{2}$$ norm) = $$\sigma_{1}$$ and $$\Vert A \Vert _{F} ^{2}$$ ([Frobenius norm](<https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>)) = $$\sqrt{\sigma_{1}^{2} + \sigma_{2}^{2} + \cdots + \sigma_{n}^{2}}$$

**정리 4** : $$A$$의 0이 아닌 특이값은 $$A^{T}A$$ 또는 $$AA^{T}$$의 고유값의 양의 제곱근이다.

**정리 4의 간단한 증명**:  $$A^{T}A$$ = $$ V \Sigma^{T} U^{T} U \Sigma V^{T} $$ = $$ V \Sigma^{2} V^{T}$$가 되고, $$A^{T}A$$는 대칭 행렬이므로, $$ V \Sigma^{2} V^{T} $$는 $$A$$의 고유값 분해의 형태와 같아집니다. 그러므로, $$A$$의 특이값은 $$A^{T}A$$의 고유값의 제곱근으로 볼 수 있고, $$V$$는 $$A^{T}A$$의 고유벡터들을 열로 하는 행렬로 볼 수 있습니다. 그리고 이 때, $$V^{T} V = V V^{T} = I$$가 성립합니다. 또한, $$A^{T}A$$가 [양의 준정부호 배열](https://ko.wikipedia.org/wiki/정부호_행렬) (positive semidefinite matrix) 이 되므로, 이 행렬의 고유값은 모두 음수가 아니어서, 특이값은 모두 실수가 됩니다. 마찬가지로, $$A A^{T}$$ = $$ U \Sigma V^{T} V \Sigma^{T} U^{T} $$ = $$ U \Sigma^{2} U^{T}$$가 되고, 이 때에도 고유값 분해의 형태와 같습니다. 그러므로, $$U$$를 $$A A^{T}$$의 고유벡터들을 열로 하는 행렬로 볼 수 있고, $$U^{T} U = U U^{T} = I$$가 성립합니다. 그러므로, $$A^{T}A$$와 $$A A^{T}$$의 고유값과 고유벡터를 구하면, 특이값 분해를 구할 수 있습니다.

**정리 5** : $$\vert det(A) \vert = \sigma_{1} \sigma_{2} \cdots \sigma_{n}$$

**정리 5의 간단한 증명**: $$\vert det(A) \vert = \vert det(U) \vert \vert det(\Sigma) \vert \vert det(V) \vert = 1 \cdot \sigma_{1} \sigma_{2} \cdots \sigma_{n}$$

**정리 6** : $$A = A^{T}$$일 때, $$A$$의 특이값은 고유값의 절댓값과 같다.

**정리 6의 간단한 증명**: $$A$$가 대칭 행렬이므로, 고유값 분해 $$A = X \Lambda X^{T}$$는 특이값 분해와 형태가 완전히 동일합니다. ($$U = V = X$$로 두었을 때, $$U^{T}U = U U^{T} = V^{T} V = V V^{T} = I $$가 성립합니다!) 그러므로 특이값의 절댓값과 고유값의 절댓값이 같습니다. 그리고 특이값은 어떠한 행렬의 고유값의 양의 제곱근이므로, 특이값은 고유값의 절댓값과 같습니다.

## 특이값 분해를 사용한 Low-Rank Approximation

Low-Rank Approximation은 행렬의 rank에 제약을 두면서 원래의 행렬과 가장 비슷한 행렬을 찾는 문제입니다. 근사한 행렬이 원래의 행렬과 얼마나 비슷한지는 보통 행렬의 norm을 사용하여 나타냅니다.

특이값 분해에 사용하는 행렬은 아래와 같이 나타낼 수 있습니다.

$$U = \begin{bmatrix} u_{1} & u_{2} & \cdots & u_{n} \end{bmatrix} $$

$$V = \begin{bmatrix} v_{1} & v_{2} & \cdots & v_{n} \end{bmatrix} $$

$$\Sigma = \begin{bmatrix} \sigma_{1} & & & \\ & \sigma_{2} & & \\ & & \ddots & \\ & & & \sigma_{n} \end{bmatrix}$$ ($$ \sigma_{1} \geq \sigma_{2} \geq \cdots \geq \sigma_{n}$$) 

$$\Sigma_{i} = \begin{bmatrix} \sigma_{1} & & & & \\ & \ddots & & & \\ & & \sigma_{i} & \\ & & & 0 & \\ & & & & \ddots  \end{bmatrix}$$ 로 정의를 하고, $$A_{i} = U \Sigma_{i} V^{T}$$로 정의합시다.

그러면 $$rank(A_{i}) = i$$와

$$A - A_{i} = \begin{bmatrix} u_{i+1} & u_{i+2} & \cdots & u_{n} \end{bmatrix} \begin{bmatrix} \sigma_{i+1} & & & \\ & \sigma_{i+2} & & \\ & & \ddots & \\ & & & \sigma_{n} \end{bmatrix} \begin{bmatrix} v_{i+1} & v_{i+2} & \cdots & v_{n} \end{bmatrix}$$가 성립합니다.

그리고 다음과 같은 중요한 정리가 성립한다고 알려져 있습니다.

**정리**: $$\underset{rank(B) \leq i}{\operatorname{argmin}} \Vert A - B \Vert _{2} = \underset{rank(B) \leq i}{\operatorname{argmin}} \Vert A - B \Vert  _{F} = A_{i}$$

즉, SVD를 이용하여 best low-rank approximation을 구할 수 있습니다.


## 특이값 분해의 활용

특이값 분해를 설명할 때 가장 대표적인 예시로 드는 것은 바로 이미지 압축 (Image Compression)입니다. 흑백 이미지의 경우, 픽셀의 정보들을 행렬로 나타낼 수 있습니다. 그래서 위에서 언급했던 Low-rank Approximation을 활용해서, 흑백 이미지를 압축할 수 있습니다. 컬러 이미지의 경우, R, G, B에 해당하는 값들을 모아 각각 행렬을 만든 뒤, 각각을 approximation하면 이미지를 압축할 수 있습니다.

![](/assets/images/matrix-decomposition-jihoon/cat.png)

왼쪽 위의 사진이 고양이를 촬영한 원본 사진이고, 나머지 세 사진은 각각 rank를 200, 50, 10으로 제약을 두었을 때의 best low-rank approximation입니다. rank를 10으로 두었을 때는 형태가 제대로 나타나지 않지만, rank가 50일 때에는 고양이의 형태가 흐릿하게 보이고, rank가 200일 때는 원본과 큰 차이가 나지 않는 모습을 볼 수 있습니다.

또한, 사용자들이 여러 아이템들에 대해서 평점을 매기고, 이러한 평점을 행렬 A로 저장하는 경우, 특이값 분해를 사용하여, 사용자들을 나타내는 벡터와 아이템들을 나타내는 벡터를 얻을 수 있습니다. 

![](/assets/images/matrix-decomposition-jihoon/user-item-matrix.png)

위의 사진은, 유저가 영화에 평점을 매긴 행렬에서 특이값 분해를 한 결과를 나타냅니다. 이 때, 첫 번째 유저를 나타내는 벡터는 $$\begin{pmatrix} 0.13 & 0.02 & -0.01 \end{pmatrix}$$가 되고, 첫 번째 아이템(영화)를 나타내는 벡터는 $$\begin{pmatrix} 0.56 & 0.12 & 0.40 \end{pmatrix}$$가 됩니다.

그러나 평점을 원소로 하는 행렬에서 모든 항목이 다 채워져있기를 기대하기는 힘듭니다. (모든 유저가 모든 아이템에 대해서 평점을 매겨야 하기 때문입니다.) 이러한 경우, 아직 유저가 아직 평가하지 않은 아이템의 평점을 예측하려면 어떻게 해야 할까요? 먼저 평가가 매겨지지 않은 위치에 값을 임의로 채우고 특이값 분해를 합니다.  그 후 rank를 제한을 두어서 low-rank approximation을 수행합니다. 그렇다면, low-rank approximation을 한 결과가 아직 평가하지 않은 항목에 대한 평점의 예측이 됩니다.

![](/assets/images/matrix-decomposition-jihoon/user-item-prediction.png)

위의 사진에서는 평가가 매겨지지 않은 위치에 값을 모두 0으로 채웠지만, 채우는 값이 다른 경우 예측값이나 예측의 품질이 달라질 수 있습니다.


# Reference

[J. Leskovec, A. Rajaraman, J. Ullman: Mining of Massive Datasets (Chapter 11. Dimensionality Reduction: SVD & CUR)](http://www.mmds.org/)

Lloyd N. Trefethen, David Bau: Numerical Linear Algebra (Chapter 4-5), SIAM

