---
layout: post

title: "Geometric Deep Learning"

date: 2023-01-18

author: ainta

tags: [deep-learning]
---

## Introduction

먼저, geometric deep learning에 대해 좀 더 알아보고 싶으신 분은 이 분야의 founder들이 작성한 [놀라울 정도로 잘 쓰인 글](https://arxiv.org/pdf/2104.13478.pdf)이 있기 때문에 일독을 권합니다. 앞으로 이 글의 거의 대부분의 내용들은 위 자료를 따릅니다.

지난 10여년간 딥러닝은 엄청나게 빠르게 발전하며 컴퓨터 비전을 비롯하여 chatgpt에 이르기까지 다양한 산업에 수많은 변화를 가져왔습니다. 그리고 아직도 매일같이 수많은 논문이 쏟아져 나오고 있습니다. 각  논문들에서는 학습하는 데이터에 최적화된 각기 다른 neural network 구조를 사용하고 있고, 결국 우리는 성공적으로 동작하는 CNN, RNN, Transformer 등 여러가지 architecture들을 가지게 되었지만 이들을 통합적으로 설명할 수 있는 원칙은 거의 없습니다. 이 base principle의 부재는 흔히 바퀴의 재발명으로 일컫어지는 동일한 개념의 재발명으로 이어지게 됩니다. 또한 후에 발명될 새로운 architecture의 예측을 위해서라도, 현재까지 나온 major한 structure들을 설명하고 앞으로 나올 structure를 설계하는 데 기준이 될 무엇인가는 분명 필요하다고 할 수 있습니다.

Geometric Deep Learning은 다음과 같은 두 가지 방법으로 이를 해결하고자 하는 시도입니다.
 - 공통적인 하나의 수학적 framework로 현재 가장 널리 쓰이는 neural network architecture들을 도출해냅니다.
 - base principle에 입각하여 새로운 architecture를 구축하기 위한 절차인 "Blueprint of Geometric Deep Learning"을 제시합니다.


## The Curse of Dimensionality

Supervised machine learning 문제를 하나 생각해봅시다. 가장 간단하게 생각할 수 있는 문제 중 하나는 강아지 또는 고양이의 grayscale 256x256 image가 주어질 때 강아지인지 고양이인지를 판단하는 문제입니다. 2D image의 각 row를 이어 1D vector로 보면 결국 이 문제는 unknown function $f:\mathbb{R}^{65536} \rightarrow 0 \cup 1$ 을 estimate하는 문제입니다. 즉, Supervised learning은 기본적으로 function apporoximation의 형태를 가집니다.

Universal approximation theorem에 의하면, 가장 기본적인 2-layer perceptron 구조로도 어떤 continuous function이든 원하는 오차 이하로 근사가 가능합니다. 그러나, domain의 차원이 올라가게 된다면 이 근사에는 너무 많은 sample이 필요하게 됩니다. 구하고자 하는 함수 $f$를 input에 따른 output의 변화가 크지 않은 Lipschitz continuous function으로 한정하더라도, apporoximation에 필요한 sample의 개수는 dimension $d$에 exponential하게 증가합니다.

다시 이전의 강아지와 고양이를 분류하는 문제로 돌아가봅시다. 256x256 image가 입력으로 주어진다고 해서 이를 65536차원의 general한 input으로 보는 것은 주어진 정보가 강아지 또는 고양이의 이미지라는 사실을 전혀 이용하지 못하고 있습니다. 대부분의 ML 문제들이 이처럼 추가적인 implicit한 구조를 가지고 있습니다. Computer vision에서 각 object는 기하학적 구조를 가지고, NLP(자연어 처리)에서도 이전 단어들에 따라 다음에 나올 단어의 확률이 변화합니다.

주어지는 input을 도메인 $\Omega$에서 정의된 하나의 signal $x: \Omega \rightarrow \mathcal{C}$로 생각할 수 있습니다. 2D grayscale image같은 경우 $\Omega$ 는 256x256 grid, $\mathcal{C}$는 실수들의 집합이 됩니다. 그리고 이러한 signal들의 집합 $\mathcal{X}(\Omega)$ 은 fuction $x: \Omega \rightarrow \mathcal{C}$들의 function space입니다.

## Symmetry Group, group action

우리가 학습하고자 하는 함수 $f$는 어떤 조건을 만족해야 할까요? 먼저, 강아지나 고양이의 이미지가 몇 픽셀 shift한다고 해서 그 의미가 바뀌지 않기 때문에 $f$는 평행이동(translation)에 대해 동일한 결과를 돌려주어야 합니다. 또한, 강아지나 고양이가 있는 부분이 확대/축소되어도 $f$는 동일한 결과를 돌려주어야 할 것입니다. 객체나 시스템에서, **Symmetry**란 해당 객체나 시스템의 특성 속성이 변하지 않는 변환을 뜻합니다. 이에 앞에서 말한 평행이동, 확대/축소는 모두 Symmetry에 해당된다고 할 수 있습니다.

256x256 image에서는 $256^2$가지의 평행이동들의 집합이 있습니다. 이를 $\mathfrak{G}$라 하면 평행이동 $g, h \in \mathfrak{G}$에 대해 $h \circ g$ 역시 평행이동이고, binary operation $\circ$에 대해 $\mathfrak{G}$는 하나의 **group**을 이룹니다. 앞서 평행이동은 symmetry이므로, 이를 **symmetry group**이라 합니다 (formal한 정의는 아닙니다).

신호 $x$에 대해 $x(u)$를 위치 $u$에서의 이미지 픽셀값이라 하면, 평행이동 $g$에 대해 $gx(u) = x(g^{-1}u)$가 성립합니다. 이는 그림 위에 좌표평면이 그려진 모눈종이를 올렸을 때, 평행이동 하기 전 원점에 있던 부분이 $(-x,-y)$만큼 평행이동 후 $(x,y)$에 오게 되는 것을 생각하면 이해하기 쉽습니다.

방금 말씀드린  $gx(u) = x(g^{-1}u)$ 라는 식은 사실 평행이동에 국한된 것은 아닙니다. 즉, group $\mathfrak{G}$에 대해 group action $\mathfrak{G} \times \Omega \rightarrow \Omega$이 정의될 때, $\mathfrak{G} \times \mathcal{X}(\Omega) \rightarrow \mathcal{X}(\Omega)$ 에 대한 group action $gx(u) = x(g^{-1}u)$이 자연스럽게 나오게 됩니다.

방금 설명드린 group, group action에 대해 이해하기 어렵다면 $\Omega$는 2차원 grid, $\mathfrak{G}$는 평행이동들의 집합, $\mathcal{X}(\Omega)$는 2차원 grid 위의 이미지(신호) 들의 집합이며 평행이동 $g \in \mathfrak{G}$ 에 대해 2차원 grid가 $g$에 해당하는 평행이동을 한 것과 이미지가 $g^{-1}$에 해당하는 평행이동을 한 것이 동일하다 라고 생각하면 편할 것입니다.

수학에서 가장 많이 연구된 group 중 하나는 $n \times n$ invertible 행렬들의 집합 general linear group $GL(n)$입니다. 이에, group $\mathfrak{G}$가 있을때 만약 map $\rho: \mathfrak{G} \rightarrow GL(n)$이 존재하여 $g, h \in \mathfrak{G}$에 대해 $\rho(gh) = \rho(g)\rho(h)$가 존재한다면 $\rho$를 $\mathfrak{G}$의 **representation**이라 합니다. 이 때, $\rho(g)$는 $g$와 일대일 대응이 되므로 앞으로 $\rho(g)x$라고 표기하면 이때까지 설명드린 $gx$와 동일한 뜻이라고 이해하시면 됩니다.

domain의 underlying structure는 Symmetry group에 의해 나타내어지고, 우리가 학습하는 함수들은 invariant 또는 equivariant라는 특성을 가지게 됩니다:

**Definition ($\mathfrak{G}$-invariant)**. A function $f: \mathcal{X}(\Omega) \rightarrow \mathcal{Y}$ is $\mathfrak{G}$-invariant if $f(\rho(g)x) = f(x)$ for all $g \in \mathfrak{G}$ and $x \in \mathcal{X}(\Omega)$.

$f$의 결과값이 input에 대한 group action에 불변일 때 $f$를 $\mathfrak{G}$-invariant라고 부릅니다.

**Definition ($\mathfrak{G}$-equivariant)**. A function $f: \mathcal{X}(\Omega) \rightarrow \mathcal{X}(\Omega)$ is $\mathfrak{G}$-equivariant if $f(\rho(g)x) = \rho(g)f(x)$ for all $g \in \mathfrak{G}$ and $x \in \mathcal{X}(\Omega)$.

함수 $f$의 출력이 group action에 의해 입력과 동일한 형태로 변환되는 경우 $f$를 $\mathfrak{G}$-equivariant라 합니다. equivariant function의 composition 역시 equivariant합니다.

강아지와 고양이 분류와 같은 image classification task는 $\mathfrak{G}$-invariant function을 학습하는 전형적인 예시입니다. 반면, object가 나타나는 box를 찾는 object detection은 $\mathfrak{G}$-equivariant function의 대표적인 예시입니다.

## Scale Separation

고화질 이미지를 용량 절감을 위해 크기가 작은 이미지로 만들 때, 픽셀 사이즈는 줄어들어도 이미지의 퀄리티만 약간 떨어질 뿐 의미가 바뀌지는 않습니다. 이와 같은 변환을 coarse-graining이라 합니다. 

**coarse-graining operator $P: \mathcal{X}(\Omega) \rightarrow \mathcal{X}(\Omega')$**

$j$가 높아질수록 coarse해지는 domain $\Omega_j$들과 이에 대한 신호들의 집합 $\mathcal{X}_j(\Omega_j)$가 있을 때, 함수 $f$가 $f \approx f_j \circ P_j$를 만족시킬 때($P_j$: non-linear coarse graining), $f$가 **locally stable**하다고 합니다. 즉, metric이 정의되어있는 $\Omega$에서 $f$는 최종적으로는 멀리 떨어져있는 두 점의 interaction에 의존할 수는 있더라도 이는 coarse graining 이후에는 국지적인 interaction으로 볼 수 있다는 뜻입니다. 즉, locally stable한 함수 $f$는 처음에는 localized interaction을 보고, coarse-graining(local pooling) 이후 localized interaction을 보고, .. 하는 식으로 결정됩니다. 따라서, 처음에는 localized interaction에 초점을 맞춘 후 coarse graining을 통해 전파함으로써 서로 다른 scale의 interaction을 분리할 수 있습니다. 이를 **scale separation**이라 합니다.

## Blueprint of Geometric Deep Learning

앞서 이야기한 Symmetry group과 이에 대한 invariant / equivariant fucntion, 그리고 Scale Separation의 개념을 통해 Geometric Deep Learning에서 설계하는 neural network architecture의 모양을 생각해볼 수 있습니다.

가장 간단한 구조는 equivariant layer가 여러개 있고 마지막에 invariant layer가 있는 구조일 것입니다. equivariant function $f_1, f_2, .., f_n$과 invariant function $g$에 대해 $f_n \circ ... \circ f_1$은 equivariant, $g \circ f_n \circ ... \circ f_1$는 invariant인 것을 쉽게 확인할 수 있으므로 전체 network는 invariant함을 쉽게 알 수 있습니다. 그리고 마지막 invariant layer에서는 결과값을 도출해야 하므로 global한 aggregation layer가 되어야 할 것입니다.

equivariant layer로는 가장 대표적으로 CNN의 convolution layer가 있습니다. CNN의 convolution layer가 어떤식으로 나오게 되었는지를 Geometric Deep Learning의 방식으로 설명할 수 있는데, 이는 일단 다음 챕터로 제쳐두겠습니다.

그리고 Scale separation을 위해 중간에 coarse graining을 수행하는 Local pooling layer가 들어가주면 보다 일반적인 형태의 neural network architecture가 완성됩니다.

앞서 representation으로 group을 표현했던 만큼, 앞서 말한 equivalent layer는 linear layer입니다. 따라서, universal approximation을 위한 nonlinearity $\sigma$가 추가적으로 필요합니다. (ReLU / sigmoid)

**Geometric Deep Learning blueprint** : Equivariant layer - nonlinearity - Local pooling layer - Equivariant layer - nonlinearity - Local pooling layer - .... - Equivariant layer - nonlinearity - Invariant global aggregate layer

앞서서는 Domain이 2D grid인 예시만 들었지만, graph와 같은 다른 domain에서도 위의 blueprint를 적용할 수 있습니다. 대표적인 neural network 몇 가지에 대해 어떤 경우에 해당되는지 나열해보면 아래와 같습니다.
- CNN은 Domain $\Omega$이 Grid, Symmetry group $\mathfrak{G}$가 Translation인 경우에 해당됩니다.
- Mesh CNN은 Euclidean Grid에서 벗어나 확장된 형태로, $\Omega$는 Manifold, $\mathfrak{G}$는 Isometry $Iso(\Omega)$ 또는 Gauge Symmetry $SO(2)$인 경우입니다. 
- GNN(Graph Neural Network)는 $\Omega$가 graph, $\mathfrak{G}$는 symmetric group(permutation들의 group, Symmetry group과 다릅니다)인 경우입니다.
- LSTM은 $\Omega$가 1D grid, $\mathfrak{G}$는 Time warping 인 경우입니다.
- Transformer는 $\Omega$가 complete graph, $\mathfrak{G}$는 symmetric group 인 경우입니다. (단어가 vertex에 대응)

## Convolution Layer

Computer Vision에서 가장 획기적인 성과를 도출한 neural network는 CNN입니다. 그리고 CNN의 핵심은 Convolution layer라고 볼 수 있습니다. Geometric Deep Learning의 관점에서, computer vision task는 image에 대한 function apporoximation이고, 이는 shift invariant의 특성을 가집니다. 이 관점에서 우리는 convolution layer가 왜 나오게 되었는지 알아볼 것입니다.

$n$-dimensional vector $x, w$에 대해, discrete convolution은 다음과 같이 정의됩니다:

$$(x * w)_i = \sum_{k=0}^{n-1} w_{i-k} x_k$$

편의를 위해 여기에서 $x, w$는 circular vector입니다. 즉, $w_{-3} = w_{n-3}$ 입니다. 

$n \times n$ 행렬 $C(w)$가 존재하여 $(x*w) = C(w)x$가 성립합니다. 이 때 $C(w)_{i,j} = w_{i-j}$ 입니다.

$C(w)$처럼 각 row가 바로 윗 row에서 한칸씩 cyclic shift된 형태의 행렬을 **circulant matrix**라 합니다. 반대로, 모든 Circulant matrix는 맨 윗 row에 해당하는 vector를 $w$라 할 때 $C(w)$로 표현됩니다. Circulant matrix의 중요한 property로는 다음이 있습니다.

- Circulant Matrices $A, B$에 대해, $AB=BA$가 성립한다. 즉, Circulant Matrix들은 Commute.

한 칸 평행이동을 의미하는 operator $S = C([0,1,0,0,...,0]^T)$를 생각해봅시다. $S$는 그 자체로 circulant matrix이므로 $S$와 circulant matrix들과 commute합니다. 역으로, $S$와 commute하는 matrix들을 생각하면 $i$행 $j$열의 값과 $i+1$행 $j+1$열의 값이 일치해야 합니다. 즉, circulant matrix여야 합니다.

따라서, shift operator와 commute하는 것과 circulant matrix는 필요충분조건이 됩니다. commute한다는 뜻은 shift operator를 먼저 취하든 나중에 취하든 동일한 값이 나온다는 것이고, 이는 shift operator에 **equivariant**하다는 것입니다. 그리고 Convolution $(x*w) = C(w)x$ 에서, 우리는 다음과 같이 Convolution을 재정의할 수 있습니다:

- Convolution은 shift-equivariant한 linear operator들의 집합이다.

즉, Convolution은 $n$-dimensional vector $w$들에 대한 $C(w)$들의 집합이고, 이는 shift-equivariant한 matrix들의 집합과 동치입니다.

따라서, shift 연산에 대한 equivariant layer는 필연적으로 convolution layer일 수 밖에 없었던 것이라는 결론을 도출할 수 있습니다.

## Fourier Transform

아예 관련이 없는 것은 아니지만, 잠시 Geometric Deep Learning 과는 약간 떨어진 이야기를 해보겠습니다. 앞서 말한 shift operator와 circulant matrix로부터 convolution theorem을 설명할 수 있습니다.

**Theorem 1(Convolution Theorem).** $\mathcal{F}(x * w) = (\mathcal{F}(x) \cdot \mathcal{F}(w))$, where $\cdot$ is pointwise multiplication.

먼저, Commute하는 두 matrix는 jointly diagonalizable합니다. 즉, 같은 eigenvector들의 집합으로 대각화됩니다. 모든 Circulant matrix는 commute하므로, 하나를 골라 그 eigenvector를 잡으면 모든 circulant matrix의 eigenvector가 됩니다. shift operator $S$를 대표로 잡으면 fourier basis $\varphi_k = \frac{1}{\sqrt n}(1, e^{2\pi i k/n}, ..., e^{2\pi i (n-1) k/n})^T$ 는 $S$의 eigenbasis입니다. 

Fourier matrix $\Phi = (\varphi_1, .., \varphi_n)$ 에 대해, $n$-dimensional vector $x$의 DFT(discrete fourier transform)은 $\Phi^* x$로 정의됩니다. 반대로, inverse discrete fourier transform(IDFT)는 $\Phi x$입니다. 정의에 의해 Ciruclant Matrix는 Fourier Transform에 의해 Diagonalize됩니다.


$\mathcal{F}(x*w) = \Phi^*(x*w) = \Phi^*C(w)x = \Phi^*(\Phi\Lambda\Phi^*)x = \Lambda \Phi^* x$

$\mathcal{F}(w) = \Phi^* w = (\lambda_1,.., \lambda_n)^T$

이므로, Convolution Theorem이 성립합니다.

이상의 과정으로 shift invariant matrix에서 circulant matrix, convolution을 거쳐 circulant matrix의 eigenvector를 구함으로써 fourier transform과 fourier theorem까지를 자연스럽게 유도할 수 있습니다.

## GNN and Transformer

Graph Neural Networks(GNNs) 는 Geometric Deep Learning blueprint 의 graph에서의 realization으로 볼 수 있습니다. 이 때 Symmetry group은 permutation들의 group인 symmetric group이 됩니다. 그래프에서 각 vertex의 번호가 바뀌더라도 그래프의 컴포넌트 개수, 가장 긴 최단경로의 길이 등의 global property는 바뀌지 않기 때문에, permutation은 property를 변화시키지 않는 symmetry가 됩니다.

graph $G = (V,E)$는 가장 간단하게 adjacency matrix $A$로 표현할 수 있습니다. $A$는 당연히 permutation invariant 하지 않기 때문에 함수 $F(A)$는 permutation invariant할 수 없습니다. 하지만 함수 $F(A)$는 permutation equivariant할 수는 있습니다. GNN에서 equivariance layer를 만들 때는 보통 neighborhood의 순서에 대해 invariant한, 즉 permutation에 대해 locally invariant한 function을 이용합니다.

현재까지 알려진 GNN 구조에서, permutation invariant function over local neighborhoods는 인접한 vertex (neighborhood)들에 대한 aggregration으로 계산됩니다. 이에는 크게 다음과 같은 3가지 방법이 있습니다:

- Convolutional:  $h_u = \phi(x_u, \sum_{v \in N_u} c_{uv} \psi(x_v))$

- Attentional:  $h_u = \phi(x_u, \sum_{v \in N_u} a(x_u,x_v) \psi(x_v))$

- Message-Passing:  $h_u = \phi(x_u, \sum_{v \in N_u} \psi(x_u, x_v))$

즉, 기본적으로 edge에 가중치가 있고 neighborhood로부터 계산된 함수에 가중치를 붙여 그 합을 계산하는 형태입니다. 그리고 Attentional의 경우 가중치를 학습할 수 있는 형태($a$), message passing의 경우 vertex마다 representation vector가 있어 인접한 정점들의 정보를 통해 이를 업데이트해 나가는 형태입니다.

각 방법들은 convolution $\subset$ attention $\subset$ message-passing 의 순서로 포괄적입니다.

여기서 Attention을 주목해 보면, Transformer에서 핵심 layer인 self-attention이 이와 동일한 방식으로 구성되어 있음을 알 수 있습니다. Sentence를 complete graph of words 로 보았을 떄, attention weight를 곱하고 softmax를 취해 더하는 과정인 해당 layer는 위에서 attentional의 수식과 일치합니다. 즉, Transformer의 self-attention layer를 Geometric Deep Learning의 graph domain에서의 realization인 GNN의 layer로서 해석할 수 있습니다.

## Conclusion

Geometric Deep Learning의 Blueprint와 간단한 예시에 대해서까지 먼저 살펴보았습니다. Geometric Deep Learning의 perspective에서 진행되는 연구는 현재 굉장히 활발하며, graph와 grid를 넘어 매우 다양한 domain으로 확장되고 있습니다. Grassmann manifold가 domain인 [Grassmannian Learning](https://arxiv.org/pdf/1808.02229.pdf)은 2018년에 진행된 시간이 조금 지난 연구이고, 최근에는 Riemmanian manifold에서의 diffusion 등의 새로운 연구가 진행되고 있습니다 ([Riemmanian Diffusion Models](https://arxiv.org/pdf/2208.07949.pdf)). 결국 여기서도 기존의 연구들처럼 domain을 무작위로 확장하여 여러 구조가 난립하게 되는 것이 아니냐고 물을 수 있겠지만, 이를 invariant를 유지하는 symmetry라는 것으로 설명할 수 있도록 한다는 점에서 차별화가 된다고 볼 수 있습니다. 

Geometric Deep Learning은 딥러닝의 neural network가 더이상 black box로서의 기능이 아니라 symmetry에 불변하는 function을 학습한다는 의미를 주기 때문에 앞으로 보다 발전된 neural network 구조를 만들고 또한 설명하는 데에 큰 도움을 줄 것입니다.

## References

- [M. Bronstein, Geometric Deep Learning
Grids, Groups, Graphs,
Geodesics, and Gauges](https://arxiv.org/pdf/2104.13478.pdf)
- [Jiayao Zhang, Grassmannian Learning](https://arxiv.org/pdf/1808.02229.pdf)
- [Chin-Wei Huang, Riemmanian Diffusion Models](https://arxiv.org/pdf/2208.07949.pdf)
- [(Medium) Deriving convolution from first principles](https://towardsdatascience.com/deriving-convolution-from-first-principles-4ff124888028)