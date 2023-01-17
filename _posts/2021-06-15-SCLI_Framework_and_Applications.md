---

layout: post

title:  "SCLI Framework and its Applications on Minimax Problems"

date: 2021-06-15

author: rkm0959

tags: [optimization, machine-learning]

---

# Introduction

Machine Learning, Artificial Intelligence의 가장 기본적인 구조는 주어진 데이터에 대한 loss function을 만들고, 이를 최소화하는 것입니다. 
loss function $f$를 design 했다면, 이 $f$를 최소화하는 것은 최적화 알고리즘의 영역에 들어오게 됩니다. 
특히, ML/AI 분야에서는 $f$를 최소화하기 위하여 그 gradient $\nabla f$를 사용하는 gradient-based optimization을 주로 사용합니다. 이러한 환경에서, 최적화 알고리즘을 연구하는 사람들이 자연스럽게 최적화 알고리즘에 대하여 주로 관심을 가지게 되는 정보는 크게 다음과 같습니다. 

- $f$에 대한 특정 조건이 주어졌을 때, 주어진 알고리즘이 얼마나 빠르게 최적해로 수렴함을 보장할 수 있는가?
- $f$에 대한 특정 조건이 주어졌을 때, 가장 빠른 알고리즘은 무엇이고 그 성능은 어느 정도인가?

이 글에서는 두 번째 질문에 대한 논의를 합니다. 이 질문을 완벽하게 해결하려면, 세 가지 과정을 거쳐야 합니다. 

- **문제의 정의** : $f$에 대한 조건과, 다루고자 하는 최적화 알고리즘의 범위를 엄밀하게 설정
- **Upper Bound** : 범위에 들어가는 좋은 최적화 알고리즘을 설계하여, 이 알고리즘이 빠르게 수렴함을 증명
- **Lower Bound** : 범위에 있는 알고리즘을 모두 고려해도, 성능이 특정 경계선보다 더 좋아질 수는 없음을 증명 

만약 Upper Bound의 결과가 Lower Bound의 결과와 일치한다면, 문제가 풀린 것입니다. 물론, 이렇게 완벽하게 두 결과가 일치함을 (exact match) 보이는 것은 상당히 어려운 일입니다. 대신, 두 bound 사이의 차이가 상수배라면 (match up to a constant) 이 역시 큰 의미가 있는 결과입니다. 하나의 Remark를 하자면,
- 상수배 차이란, 원하는 정확도의 해를 얻기 위한 계산량에 대한 두 bound 사이에 상수배 차이가 난다는 것

Upper Bound의 경우, 보통 연구가 이루어지는 방법은 대략적으로 다음과 같습니다. 

- 문제에 대한 좋은 알고리즘을 설계
- 알고리즘의 성능을 증명하기 위한 Lyapunov Function 설계
- Lyapunov Function에 대한 부등식을 증명

Lower Bound의 경우에는 상황이 조금 더 복잡합니다. 범위에 있는 최적화 알고리즘을 모두 고려해야 한다는 것이 문제의 난이도를 끌어올리는 점입니다. 이 경우, 연구가 이루어지는 방법은 대강 다음과 같습니다. 

- 범위에 속하는 알고리즘 $\mathcal{A}$를 하나 임의로 잡는다.
- 이 $\mathcal{A}$의 성능이 기준선을 넘을 수 없음을 증명하는 함수 $f$ 및 초기조건을 찾는다.

이 글에서는 **Lower Bound에 대한 이야기**를 주로 합니다. 글의 흐름은, 

- 이미 해결된 smooth convex minimization에 대한 문제의 해결과정을 살펴봅니다. 
- 이 과정에서 잠시 등장하는 Quadratic Function에 대하여 여러 논의를 합니다. 
- 이 논의를 기반으로, 2015년에 등장한 SCLI Framework에 대하여 소개합니다. 
- SCLI Framework를 Minimax Problem에 적용한 최근 결과에 대하여 소개합니다. 

# The Main Idea

## Review of Acceleration

여기서는 [1] 의 내용을 빠르게 복습합니다. 제가 만든 슬라이드인 [2]도 같이 참고하면 좋습니다. 

**Definition** : $0 \le \mu < L < \infty$에 대하여, continuously differentiable function $f : \mathbb{R}^d \rightarrow \mathbb{R}$이
 
($L$-smoothness) : $L$-smooth 하다는 것은, 모든 $x, y \in \mathbb{R}^d$에 대하여 

$f(x) \le f(y) + \langle \nabla f(y), x - y \rangle + \frac{L}{2} \lVert x-y \rVert^2$

($\mu$-strong convexity) : $\mu$-strongly convex 하다는 것은, 모든 $x, y \in \mathbb{R}^d$에 대하여 

$f(x) \ge f(y) + \langle \nabla f(y), x-y \rangle + \frac{\mu}{2} \lVert x-y \rVert^2$

특히, $\mu$-strong convexity의 조건에서 $\mu = 0$인 경우가 convexity의 조건이다. 

다룰 문제는 $L$-smooth convex function $f$의 값을 최소화하는 문제입니다. 즉, 초깃값이 $x_0$이고 $f$가 최솟값 

$f_{*} = f(x_{*}) = \inf_x f(x)$

을 가질 때, $f(x) - f_* < \epsilon$인 $x$를 효율적으로 찾는 것이 목표입니다. 

이를 위해서 다음 형태를 갖는 first-order method를 모두 고려합시다. 

$x_k \in x_{k-1} + \text{span} (\nabla f(x_0), \nabla f(x_1), \cdots , \nabla f(x_{k-1}))$

이때, $f(x_k) - f_{*}$의 값을 어떻게 작게 할 수 있는지, 그리고 최적의 알고리즘을 적용했을 때는 얼마나 작게 할 수 있는지에 대하여 수십년간 연구가 진행되었습니다. 그 결과를 간략하게 정리하면 아래와 같습니다. 

**Upper Bound** : 가장 기본적이고 유명한 결과는 

$x_k = x_{k-1} - \frac{1}{L} \nabla f(x_{k-1})$

형태를 갖는 Gradient Descent입니다. 이 경우, 

$f(x_N) - f_{*} \le \frac{L \lVert x_0 - x_{*} \rVert^2}{2N}$

이라는 결과를 얻습니다. 즉, $\mathcal{O}(1/N)$ 수렴 속도를 얻습니다. 

그 후, Nesterov가 Accelerated Gradient Method를 제시하면서 이 수렴속도가 $\mathcal{O}(1/N^2)$으로 빨라졌습니다. 

**Lower Bound** : 임의의 $k \le (d-1)/2$와 초깃값 $x_0$에 대하여, 어떠한 first-order method를 사용하더라도 적당한 $L$-smooth convex function $f : \mathbb{R}^d \rightarrow \mathbb{R}$가 있어 $k$번의 Gradient 계산으로 $x_k$를 얻었을 때

$f(x_k) - f_{*} \ge \frac{3L \lVert x_0 - x_{*} \rVert^2}{32(k+1)^2}$

이 성립함이 증명됩니다. 즉, 첫 $(d-1)/2$번의 iteration에 대해서는 $\mathcal{O}(1/N^2)$의 수렴속도가 최적입니다. 

정확하게 말하자면, $x_k \in x_{k-1} + \text{span}(\nabla f(x_0), \cdots , \nabla f(x_{k-1}))$이기만 하면 위 정리가 성립합니다. 
그 증명은 예상보다 매우 간단한데, 일반성을 잃지 않고 $x_0 = 0$이라 한 다음 

$f(x) = \frac{L}{4} \left( \frac{1}{2} x^TAx - e_1^T x \right)$

로 둡니다. 여기서 

$e_1 = (1, 0, \cdots , 0)^T, \quad A = \text{tridiag}(-1, 2, -1)$

로 설정합니다. $f$를 잡기만 하면, 실제 증명은 간단한 계산으로 할 수 있습니다. 

**Finish** : 최종적으로 문제를 해결하기 위해서, 연구자들은 

$x_k = x_{k-1} - \sum_{i=0}^{k-1} h_{k, i} \nabla f(x_i)$

형태의 first-order 알고리즘을 적용했을 때, 모든 $f$에 대하여 가능한

$\frac{f(x_N) - f_*}{\lVert x_0 - x_* \rVert^2}$

의 최댓값을 최소화하는 parameter $h_{k, i}$를 찾는 연구를 진행했습니다. 

- Drori, Teboulle은 [3]에서 이를 SDP를 이용하여 수치적으로 해결하는 방법을 제시했습니다.
- Kim, Fessler는 [4]에서 이 SDP를 해석적으로 풀어, Optimized Gradient Method를 얻습니다. 

이는 first-order 알고리즘 중에서는 Optimized Gradient Method가 최적이라고 볼 수 있음을 의미합니다. 

이후, Drori는 이 Optimized Gradient Method가 정말 최적임을 [5]에서 다음과 같이 보입니다.

우리가 $f$에 대한 최소화 문제를 풀기 위하여 사용할 수 있는 도구가 다음 black box 하나 뿐이라고 가정합시다. 
- First-Order Oracle : $x$를 넣으면, $f(x)$와 $\nabla f(x)$를 알려준다. 

Drori는 각 $N \le d-1$에 대하여, $N$번의 First-Order Oracle을 사용한 임의의 first-order method에 대하여, $f(x_N) - f_*$에 대하여 보장할 수 있는 
최선의 bound가 Optimized Gradient Method를 통해 얻어지는 bound와 일치함을 증명합니다. 이는 결국 Optimized Gradient Method가 최적의 알고리즘임을 보여줍니다. 

$L$-smooth, $\mu$-strongly convex function $f$에 대해서도 비슷한 이야기를 할 수 있습니다. 

**Upper Bound** : $\kappa = L / \mu$라고 하고, 다시 Gradient Descent 

$x_k = x_{k-1} - \frac{1}{L} \nabla f(x_{k-1})$

을 적용하면, $\mathcal{O}((1-1/\kappa)^N)$의 수렴속도를 얻습니다. 

Nesterov's Acceleration을 적용하면, 이 속도가 $\mathcal{O}((1-1/\sqrt{\kappa})^N)$으로 빨라집니다. 

**Lower Bound** : 앞선 Lower Bound 결과와 같은 context에서, 이번에는 

$f(x_k) - f_{*} \ge \frac{\mu}{2} \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^{2k} \lVert x_0 - x_* \rVert^2$

임을 증명할 수 있습니다. 증명하는 방식은 비슷한데, $f$를 

$f(x) = \frac{\mu (\kappa - 1)}{4} \left( \frac{1}{2} x^TAx - e_1^T x \right) + \frac{\mu}{2} \lVert x \rVert^2$

으로 두고 열심히 계산을 하면 얻을 수 있습니다. $A, e_1$의 정의는 이전의 정의와 같습니다. 

**Finish** : 앞선 $L$-smooth convex의 경우와 비슷하지만 약간 다르게, 이번에는 

$\frac{\lVert x_N - x_* \rVert^2}{\lVert x_0 - x_* \rVert^2}$

으로 가능한 최댓값을 최소화하는 first-order method를 찾습니다. 그 결과 Drori, Taylor는 [6]에서 Information Theoretic Exact Method라는 새로운 알고리즘을 얻고, 
[7]에서 이 알고리즘이 First-Order Oracle을 사용했을 때 Optimized Gradient Method처럼 최적의 알고리즘이라는 것을 증명합니다. 

이 최적의 알고리즘들의 수렴속도는 대략 $\mathcal{O}((1-1/\sqrt{\kappa})^{2N})$입니다. 

## Focusing on Quadratics

위 흐름을 보면, 확인할 수 있는 점이 여러가지 있습니다. 
- 실제로 한 알고리즘이 완벽하게 (exact) 최적이라는 것은 증명하기 매우 어렵습니다.
- 그런데 up to constant 최적인 Lower Bound는 $f$가 Quadratic일 때만 고려해도 얻을 수 있었습니다. 
  
$f$가 Quadratic인 경우만을 고려할 때 얻게 되는 점, 잃게 되는 점이 무엇인지를 생각해봅시다. 이제부터

$f = \frac{1}{2} x^TAx + b^T x$

라 하고, $A$가 symmetric matrix라고 가정합시다.

**얻는 점** : 우선 $\nabla f(x) = Ax + b$가 Linear 하다는 점이 매우 편리합니다. 
이 점을 이용하면 일반적인 $f$에 대해서는 할 수 없었던 식 조작이나 계산이 가능해질 것이고, Linearity의 여러 성질을 사용할 수 있습니다.

또한, $f$의 $L$-smoothness나 $\mu$-strong convexity를 $A$의 eigenvalue로 따질 수 있습니다. 
함수에 대한 조건이 행렬에 대한 조건이 된다는 건데, $f$가 $\mu$-strong convex, $L$-smooth 할 필요충분조건은 

$\mu I \preceq A \preceq L I$

입니다. 단, $A \preceq B$는 $B - A$가 positive semidefinite이란 뜻입니다. 

마지막으로, first-order method가 모두 linear transformation이 됩니다. 예를 들어, Gradient Descent는 

$x_k = x_{k-1} - \frac{1}{L} \nabla f(x_{k-1}) = x_{k-1} - \frac{1}{L} (Ax_{k-1} + b) = \left( I - \frac{1}{L} A \right) x_{k-1} - \frac{1}{L}b$

가 됩니다. 이러한 Linear Transformation이 한 점으로 얼마나 빠르게 수렴하는지는 행렬의 Spectral Radius로 얻을 수 있습니다. 
그러므로, 주어진 알고리즘의 수렴속도에 대한 문제를 행렬의 eigenvalue에 대한 문제로, 또는 다항식의 (characteristic polynomial) root에 대한 문제로 바꿀 수 있습니다. 

특히, $f$가 convex 할 경우 $f$의 optimal point를 찾는 문제는 linear system $Ax + b = 0$을 푸는 것과 같습니다. 
이 문제를 approximate 하게 푸는 문제는 이미 긴 역사가 있으니, 여기에서 아이디어를 얻을 수도 있을 것입니다. 

**잃는 점** : 가장 당연한 잃는 점은 가능한 $f$의 후보를 스스로 쳐냈다는 점입니다. 
그러므로, Quadratic Function 만을 고려하여 얻은 Lower Bound가 최적이 아닐 수도 있다는 점이 남아있습니다. 
목표가 up to constant 최적인 Lower Bound를 찾는 것이라면, Quadratic Function으로 얻은 Lower Bound와 up to constant 일치하는 Upper Bound를 찾아주면 됩니다. 
하지만 목표가 exact 최적이라면 이 방식으로 문제를 풀기는 어려울 것입니다.

별개로, 만약 우리가 최적의 알고리즘을 찾는 것이 목적이 아니라 이미 주어진 하나의 알고리즘의 수렴속도를 분석하는 것이 목적이라면, Quadratic Function 만을 고려하는 것은 매우 위험합니다. 사례를 하나 봅시다. 

Heavy Ball Method는 $L$-smooth, $\mu$-strongly convex Quadratic Function에 대해서는 optimal 수준의 강력한 성능을 보이는데, 일반적인 $L$-smooth, $\mu$-strongly convex function에 대해서는 optimal point로 수렴하는 것조차 보장할 수 없다는  (cycle에 빠지는 경우가 있음) [8]에서 Lessard에 의하여 증명되었습니다. 

**$Ax + b = 0$의 해결, 그리고 앞선 증명들** : 이제 다시 앞서 Quadratic Function을 이용해서 증명한 결과를 봅시다. 

이 section에서 하는 논의는 이번 글의 주제를 소개한 논문인 [9]의 Introduction에서 가져왔습니다.

$k \le (d-1)/2$이고 $L$-smooth, $\mu$-strongly convex function을 볼 때, 어떤 first-order 알고리즘을 사용해도

$f(x_k) - f_{*} \ge \frac{\mu}{2} \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^{2k} \lVert x_0 - x_* \rVert^2$

이도록 하는 Quadratic Function $f$가 존재함을 증명했습니다. 이는 $f(x) - f_{*} < \epsilon$인 $x$를 얻으려면 
- 적어도 $\Omega(d)$번 이상의 Gradient를 계산하거나 ($k$에 대한 조건)
- 아니면 $\Omega( \sqrt{\kappa} \log \epsilon^{-1})$번의 Gradient를 계산해야 (부등식)

합니다. 즉, 필요한 Gradient의 계산 횟수는 (First-Order Oracle 사용 횟수) 최소 

$\tilde{\Omega}( \min(d, \sqrt{\kappa} \log \epsilon^{-1}))$

임을 알 수 있습니다. 그런데 저 차원 $d$가 굉장히 걸립니다. 지금의 bound로는 $d$번 이상의 Gradient를 계산한 경우, 추가적으로 얻을 수 있는 정보가 없습니다. 
저 $d$를 어떻게 제거할 수는 없을까요? 안타깝지만 Quadratic Function 만을 보고 있는 지금 setting에서는 없습니다. 
그 이유는 생각보다 간단한데, $\Omega(d)$번의 Gradient 계산이면 무조건 $Ax + b = 0$을 만족하는 $x$를 찾을 수 있기 때문입니다. 
- 애초에 직관적으로 $d \times d$ matrix $A$에 대하여 $(x, Ax + b)$ 쌍을 $\Omega(d)$개 알면 $A, b$를 복원할 수 있습니다. 
- $A, b$를 복원하면, $Ax + b = 0$의 해를 찾는 것은 당연히 가능합니다. 
- 굳이 이러한 논의를 하지 않더라도, Conjugate Gradient Descent Method를 사용하면 됩니다. 

여기서 한 번 더 짚고 넘어가야 하는 점은, 지금 당장 논의하고 있는 Lower Bound는 **계산 자체의 효율을 따지고 있지 않습니다**.
즉, 우리가 앞서 증명한 $\tilde{\Omega}( \min(d, \sqrt{\kappa} \log \epsilon^{-1}))$의 bound는 
- Function Value 및 Gradient를 계산하는 횟수, 즉 First-Order Oracle 사용 횟수에 제약을 두지만
- First Order Method에서 사용할 coefficient의 계산에 드는 연산량에는 제약이 없습니다.
- 실제 위 증명들에서 필요한 사실은 $x_k \in x_{k-1} + \text{span}(\nabla f(x_0), \cdots \nabla f(x_{k-1}))$이 전부입니다. 

즉, 지금 다루고 있는 bound는 First-Order Oracle을 사용했을 때 이 정보가 갖는 가치만을 다루고, 실제 계산의 효율은 따지지 않습니다. 
특히, 앞서 다룬 Conjugate Gradient Descent나 $Ax + b = 0$의 해를 직접 구하는 것은 많은 계산을 필요로하는 알고리즘이고, 우리가 보통 최적화에서 다루는 알고리즘과 거리가 있습니다. 
정리하면,
- 필요한 iteration의 횟수에 대한 bound를 구하고자 하는데, 전부 $d$에 대해 제약이 있다. 
- $d$에 대한 제약을 걸게 하는 실제 알고리즘들을 보면, iteration 당 실제 계산량은 지나치게 많다. 
- $d$에 대한 제약을 걸게 하는 실제 알고리즘들을 보면, 우리가 실제로 다루고 하는 알고리즘과 다른 면이 많다.

이 문제를 해결하는 방법은, 알고리즘의 범위 자체에 제약을 거는 것입니다. 
즉, Conjugate Gradient Descent 등 알고리즘은 제외하고, Gradient Descent나 Accelerated Gradient Descent 등은 포함하는 알고리즘의 class를 하나 설계하고,
이 알고리즘들만을 보았을 때 Lower Bound가 어떻게 계산되는지를 파악하면 될 것입니다. 

우리가 다룰 새로운 class의 알고리즘들의 기본적인 특징은
- 특히 중점적으로 다룰 알고리즘들은 iteration이 간단하고, 쉽게 계산이 가능합니다.
- 최적화에서 주로 다루는 여러 중요한 알고리즘들을 대부분 포함합니다.
- Lower Bound를 계산했을 때, $d$에 대한 제약이 추가가 되지 않습니다.
- 여전히 다루는 $f$는 Quadratic Function이고, 이에 대한 얻는 점/잃는 점을 모두 가져갑니다. 
- 현재 증명된 Lower Bound는 tight 하지 않은 (않을 것으로 예상되는) 경우도 많습니다.  

# Stationary Canonical Linear Iterative (SCLI) Optimization

## Definitions

이제 이번 글의 주제인 SCLI에 대해 소개하겠습니다. 대부분의 내용은 [9]에서 나온 Arjevani의 결과입니다. 

우리가 주로 다루게 되는 대상은 

$f_{A, b}(x) = \frac{1}{2} x^TAx + b^Tx$

를 최적화하는 것으로, $A$는 symmetric positive definite입니다. 즉, strong convexity를 가정합니다. 

**Definition** : 최적화 알고리즘 $\mathcal{A}$가 $p$-SCLI optimization algorithm이란 것은, 적당한 $p+1$개의 random function 
$C_0(X), C_1(X), \cdots , C_{p-1}(X), N(X) : \mathbb{R}^{d \times d} \rightarrow \mathbb{R}^{d \times d}$가 존재하여, 
initialization, iteration이 각각

$x_0, x_1, \cdots , x_{p-1} \in \mathbb{R}^d$

$x_k = \sum_{j=0}^{p-1} C_j(A) x_{k-p+j} + N(A) b$

특히, 각 iteration에 대하여 $C_j(A)$와 $N(A)$는 이전 시행에 독립적으로 선택됨을 가정한다. 
또한, $\mathbb{E}(C_j(A))$ 들은 모두 유한하고 simultaneously triangularizable 하다고 가정한다. 
$C_j(A)$들을 coefficient matrix, $N(A)$를 inversion matrix라고 부른다. 
이때, $\mathcal{A}$의 characteristic polynomial을 

$\mathcal{L}_{\mathcal{A}}(\lambda, X) = I_d \lambda^p - \sum_{j=0}^{p-1} \mathbb{E}(C_j(A)) \lambda^j$

라고 정의하고, $X \in \mathbb{R}^{d \times d}$가 주어졌을 때 $\mathcal{L}_{\mathcal{A}}(\lambda, X)$의 root radius를 

$\rho_{\lambda}(\mathcal{L}_{\mathcal{A}}(\lambda, X)) = \max \{|\lambda'| : \det \mathcal{L}_{\mathcal{A}}(\lambda', X) = 0\}$

으로 정의한다. 또한, $p$-SCLI algorithm이 consistent 하다는 것은 $f_{A, b}$에 알고리즘을 적용하면 

$x_k \rightarrow A^{-1}b$

즉, 현재 다루고자 하는 $f$에 대하여 알고리즘이 항상 optimal point로 수렴하면 consistent한 것이다.

Remark : 두 행렬이 commute 하면 simultaneously triangularizable 함은 잘 알려져 있다. 
실제로 우리는 $C_j$들을 대부분의 경우 $A$에 대한 다항식으로 설계할 것이므로, 이 조건에 대해서는 크게 걱정하지 않아도 된다. 

이제 약간의 계산을 하면, 다음 결과를 얻을 수 있습니다. 
- Gradient Descent는 1-SCLI 알고리즘입니다.
- Heavy Ball Method, Accelerated Gradient Method는 2-SCLI 알고리즘입니다. 
- Conjugate Gradient Descent는 SCLI 알고리즘이 아닙니다. 

조금 더 자세하게 식을 써보면, Gradient Descent는 

$\displaystyle x_{k+1} = \left(I - \frac{1}{L} A \right) x_k - \frac{1}{L} b$

Heavy Ball Method는 $\displaystyle \alpha = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}$, $\displaystyle \beta = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2$에 대하여 

$x_{k+1} = ((1 + \beta)I - \alpha A) x_k - \beta x_{k-1} - \alpha b$
 
Accelerated Gradient Descent는 $\displaystyle \alpha = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}$에 대하여 

$x_{k+1} = (1 + \alpha) \left(I - \frac{1}{L} A \right) x_k - \alpha \left( I - \frac{1}{L} A \right) x_{k-1} - \frac{1}{L} b$

가 됩니다. 이 세 알고리즘이 순서대로 1-SCLI, 2-SCLI, 2-SCLI임을 확인할 수 있습니다. 

이제 다룰 알고리즘의 범위를 정했으니, 각각의 성능을 측정하는 방식을 정해야 합니다. 기본적으로 우리가 필요로 하는 연산의 양은 
(iteration 횟수)와 ($C_j(A), N(A)$를 생성하고 iteration을 돌리기 위해 필요한 연산량)의 곱이 됩니다.
물론, deterministic 알고리즘의 경우 $C_j(A), N(A)$는 전처리할 수 있습니다. 
$C_j(A), N(A)$를 계산하기 위해 걸리는 시간도 매우 중요한 요소지만, 일단은 iteration의 횟수에 집중합시다. 

$p$-SCLI 알고리즘 $\mathcal{A}$와 양수 $\epsilon$, 함수 $f_{A, b}$와 initialization $\mathcal{X}_0$에 대하여 

$\mathcal{IC}_{\mathcal{A}}(\epsilon, f_{A, b}, \mathcal{X}_0)$

를
 
$\lVert \mathbb{E}(x_k - x_{*}) \rVert < \epsilon$

이 모든 $k \ge K$에 대해서 성립하게 하는 $K$의 최솟값이라 합시다. 
물론, $x_* = A^{-1} b$입니다. 

이제부터 $p$-SCLI 알고리즘의 $\mathcal{IC}$, 즉 iteration complexity에 집중해봅시다. 

## From Optimization to Spectral Radius and Polynomials

$p$-SCLI 알고리즘을 적용하면, $\mathbb{E}(x_k)$는 이전 $p$개의 iterate $\mathbb{E}(x_{k-p}), \cdots \mathbb{E}(x_{k-1})$에 대한 linear function이 됩니다. 
그러므로, $\mathbb{E}(x_k - x_{*})$의 수렴속도는 linear map의 spectral radius를 통해서 알 수 있을 것이라고 예상할 수 있습니다. 
여기서 사용되는 것이 바로 앞서 정의된 root radius입니다. 다음 두 결과를 얻을 수 있습니다. 

**Theorem** : $p$-SCLI 알고리즘 $\mathcal{A}$가 $f_{A, b}$에 대하여 consistent 할 필요충분조건은 
- $\mathcal{L}_{\mathcal{A}}(1, A) = -\mathbb{E}(N(A))A$
- $\rho_{\lambda}(\mathcal{L}(\lambda, A)) < 1$

과 같다. 즉, $x_* = A^{-1}b$를 fixed point로 갖고 root radius가 1보다 작아야 한다. 

**Theorem** : $\mathcal{A}$가 $p$-SCLI 알고리즘이고 $\rho = \rho_{\lambda}(\mathcal{L}(\lambda, A)) < 1$라 하자. 

(1) : 적당한 initial point $\mathcal{X}_0$가 존재하여, 다음이 성립한다.

$\mathcal{IC}_{\mathcal{A}}(\epsilon, f_{A, b}, \mathcal{X}_0) = \tilde{\Omega}\left( \frac{\rho}{1-\rho} \log \epsilon^{-1} \right)$

(2) : 임의의 initial point $\mathcal{X}_0$에 대하여, 다음이 성립한다.

$\mathcal{IC}_{\mathcal{A}}(\epsilon, f_{A, b}, \mathcal{X}_0) = \tilde{\mathcal{O}}\left( \frac{1}{1-\rho} \log \epsilon^{-1} \right)$

일반적으로 $\rho \ll 1$인 경우는 드물기 때문에, 이는 결국 iteration complexity가 

$ \frac{1}{1-\rho} \log \epsilon^{-1} $

와 up to constant 일치한다는 것을 의미합니다. 즉, $\rho$가 성능을 결정합니다. 

이제 root radius에 대해서 다시 생각해봅시다. 편의상 deterministic 알고리즘만 다루겠습니다. 

$p$-SCLI 알고리즘 $\mathcal{A}$를 고정합시다. $C_j(A)$들은 simultaneously diagonalizable이므로, 

$T_j = Q^{-1} C_j Q$

가 upper triangular가 되는 $Q$가 존재합니다. 그러면 

$\det \mathcal{L}(\lambda, X) = \det (Q^{-1} \mathcal{L}( \lambda, X) Q) = \det \left( I_d \lambda^p - \sum_{j=0}^{p-1} T_j \lambda^j \right)$

가 됩니다. $\sigma_1^j, \cdots, \sigma_d^j$를 $T_j$의 대각성분이라 하면, 이는 

$l_j(\lambda) = \lambda^p - \sum_{k=0}^{p-1} \sigma_j^k \lambda^k$

라 할 때 

$\det \mathcal{L}(\lambda, X) = \prod_{j=1}^d l_j(\lambda)$

임을 의미합니다. 즉, 특히, $l_j(\lambda)$들이 $\mathcal{L}(\lambda, X)$의 eigenvalue가 됩니다. 
여기서 알 수 있는 사실이 두 가지가 있는데, 첫 번째 사실은 root radius가 결국 $l_j$들의 root의 절댓값 중 최댓값이란 사실입니다. 
두 번째 사실은, $\lambda = 1$을 대입하였을 때 $\mathcal{L}(1, A) = -N(A)A$임을 알고 있으므로, 그 eigenvalue가 정해져 있다는 것입니다. 

즉, $p$-SCLI 알고리즘에서 $N(A)$를 정했을 때 root radius를 최소화하고 싶다면 풀어야 하는 문제는 
- $l_j(1)$의 값이 정해졌을 때, $l_j$의 root의 절댓값의 최댓값을 최소화하는 $l_j$는 어떤 다항식인가?

가 되고, 결국 우리의 문제가 순도 100% 다항식 문제로 바뀌게 됩니다.

**Definition** : 다항식 $q$에 대하여 $q$의 root radius를 

$\rho(q) = \max \{|\lambda| : q(\lambda) = 0\}$

이라 정의하자. 즉, root의 절댓값 중 최댓값이다. 

**Lemma** : $q$가 monic, real polynomial이고 $\deg q = p$라 하자. 이때, 

$q(1) < 0 \implies \rho(q) > 1$

$q(1) \ge 0 \implies \rho(q) \ge \left| \sqrt[p]{q(1)} - 1 \right|$

이 성립한다. 특히, 두 번째 부등식의 경우, 등호가 성립할 필요충분조건은 

$q(z) = \left( z - \left(1 - \sqrt[p]{q(1)} \right) \right)^p$

이 Lemma를 사용하면 기본적으로 우리가 얻은 다항식 문제가 풀립니다.

결국 우리는 Lemma에 의하여 

$\rho_{\lambda}(\mathcal{L}(\lambda, X)) \ge \max \left| \sqrt[p]{\sigma_i(-N(A)A)} - 1 \right|$

을 얻습니다. 여기서 $\sigma_i$는 $i$번째 eigenvalue를 의미합니다. 

특히, 위 식의 우변을 $\rho_*$라 하면 해당 SCLI 알고리즘의 iteration complexity가 

$\tilde{\Omega} \left( \frac{\rho_*}{1 - \rho_*} \log \epsilon^{-1} \right)$

로 lower bound가 된다는 점을 확인할 수 있습니다. 

## Main Result 1 : Lower Bounds

이제 본격적으로 Lower Bound를 계산해봅시다. 사용할 setting은 
- $\mu I \preceq A \preceq L I$, 즉 $f$는 smooth/strongly convex
- $p$-SCLI algorithm 하나를 고정하고, 특히 $N(A)$가 scalar matrix임을 가정.

$N(A) = \nu I$라 하고, $A = \text{diag}(L, \mu, \cdots , \mu)$라 합시다. 
$-N(A)A$의 eigenvalue는 $-\nu L$과 $-\nu \mu$가 되고, 

$\rho_* = \max \{ \left| \sqrt[p]{-\nu L} - 1 \right|, \left| \sqrt[p]{-\nu \mu} - 1 \right| \}$

이 됩니다. 위 식의 우변을 $\nu$에 대하여 최소화하는 간단한 계산을 하면, 결국 

$\rho_* \ge \frac{\sqrt[p]{\kappa} - 1}{\sqrt[p]{\kappa} + 1}$

을 얻습니다. 물론, 여기서 $\kappa = L / \mu$입니다. 결론적으로,

**Theorem** : $N(A)$가 scalar matrix인 임의의 $p$-SCLI 알고리즘 $\mathcal{A}$에 대해, 적당한 $f_{A, b}$가 있어 

$\mathcal{IC}_{\mathcal{A}}(\epsilon, f_{A, b}) = \tilde{\Omega} \left( \frac{\sqrt[p]{\kappa} - 1}{2} \log \epsilon^{-1} \right)$

이 성립하게 할 수 있다. 물론 여기서 $\mu I \preceq A \preceq L I$. 

특히, $N(A)$가 diagonal인 경우에도 같은 결과가 성립함을 증명할 수 있습니다.

이 Lower Bound가 tight 할까요? 적어도 up to a constant 최적의 Lower Bound임은 보일 수 있습니다. 
- $p=1$인 경우, Gradient Descent의 Upper Bound와 일치합니다.
- $p=2$인 경우, Accelerated Gradient Descent의 Upper Bound와 일치합니다.

여기서 일치한다는 것은, up to a constant 일치한다는 것입니다. 

$p \ge 3$인 경우는 약간 특별합니다. 결론부터 말하면, up to a constant 최적인 Lower Bound는 맞습니다. 
그런데 이상합니다. 3-SCLI 알고리즘으로 $\kappa^{1/3} \log \epsilon^{-1}$급의 Gradient 계산으로 최적화가 가능하다면, 이미 유명했을 것 같죠. 
$p \ge 3$인 경우의 최적의 알고리즘은 분명 존재하고, 앞선 Lemma의 등호조건을 통해서 유도할 수 있지만, 실제로 계산을 해보면 
$C_j(A)$를 계산하기 위해서 $A$의 eigenvalue를 모두 계산해야 함을 확인할 수 있습니다. 이는 애초에 $Ax + b = 0$을 푸는 것만큼이나 어려운 문제고, 비효율적입니다.
그러니 이러한 알고리즘은 실제로 쓰기 불가능한 수준이고, 이론적인 결과일 뿐입니다. 이 점이 바로 이 알고리즘이 이미 유명하지 않은 이유겠죠.

이러한 점을 보았을 때, 다룰 $p$-SCLI 알고리즘의 폭을 더욱 줄일 필요가 있음을 느낄 수 있습니다. 
$C_j(A)$들을 "계산하기 쉬운 행렬"로 국한시켜야, 실제로 사용할만한 알고리즘이 유도가 되겠죠. 
이를 위해서 저자들은 $C_j(A)$를 Linear Coefficient Matrix, 즉 $C_j(A) = a_j A + b_j I_d$ 형태로 잡을 것을 제안합니다. 

저자들은 이렇게 $C_j$의 범위를 축소시키면 $p \ge 3$인 $p$-SCLI 알고리즘에 대해 강한 bound를 증명할 수 있을 것이라고 추측합니다. 
즉, $\mathcal{A}$가 Linear Coefficient Matrix를 갖고 Inversion Matrix가 Diagonal이면, 적당한 $A$를 잡아

$\rho_{\lambda}(\mathcal{L}_{\mathcal{A}} ( \lambda, A)) \ge \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$

이도록 할 수 있을 것이라고 추측합니다. 물론 $\mu I \preceq A \preceq LI$가 성립하도록 잡습니다. 

이 추측 역시 다항식과 그 root의 크기에 대한 추측으로 변환할 수 있습니다. 

## Main Result 2 : Upper Bounds

왜 갑자기 Upper Bound로 갈까요? 이를 위해서는 Acceleration에 대한 논의를 약간 더 해야합니다. 
Nesterov의 Accelerated Gradient Descent는 매우 강력하고 중요한 알고리즘이지만, 그 원리가 아직도 제대로 이해되고 있지 않은 알고리즘입니다.
알고리즘의 수렴속도는 물론 증명되었지만, 알고리즘을 자연스럽게 유도하는 방법이나 설명하는 방법, 그리고 알고리즘이 왜 빠른지에 대한 직관은 지금까지도 연구가 되고 있는 부분입니다. 

앞서 우리는 Lower Bound에 대한 이야기를 하다가, $C_j$에 대한 제약이 없으면 지나치게 사용하기 복잡한 알고리즘도 다루게 됨을 알게 되었습니다. 
이 문제를 해결하기 위해서 저자들은 Linear Coefficient Matrix와 Scalar/Diagonal Inversion Matrix를 도입하여, 다룰 알고리즘의 범위를 크게 좁혔습니다. 
좋은 점은 이 축소된 범위도 Gradient Descent, Accelerated Gradient Descent, Heavy Ball Method를 전부 포함한다는 점입니다. 

더욱 놀라운 점은, Linear Coefficient Matrix + Scalar Inversion Matrix setting을 사용하면 Accelerated Gradient Descent와 Heavy Ball Method가 자연스럽게 유도된다는 것입니다. 
즉, SCLI framework를 사용하면 두 알고리즘을 자연스럽게 유도할 수 있다는 것입니다. 이는 이 자체로도 큰 의미가 있습니다. 

$p$-SCLI 알고리즘 $\mathcal{A}$에 대해, $C_j(X) = a_j X + b_j I_d$, $N(X) = \nu I$라고 합시다. 그러면 

$\mathcal{L}(\lambda, X) = \lambda^p I_d - \sum_{j=0}^{p-1} (a_j X + b_j I_d) \lambda^j$

를 얻고, $X$의 eigenvalue를 $\sigma_1, \sigma_2, \cdots , \sigma_d$라 하면 

$l_i(\lambda) = \lambda^p - \sigma_i \sum_{j=0}^{p-1} a_j \lambda^j - \sum_{j=0}^{p-1} b_j \lambda^j$

를 얻습니다. 이제 

$a(\lambda) = \sum_{j=0}^{p-1} a_j \lambda^j, \quad b(\lambda) = \sum_{j=0}^{p-1} b_j \lambda^j$

$l(\lambda, \eta) = \lambda^p - \eta a(\lambda) - b(\lambda)$

라 하면, 우리가 원하는 bound는 $\mu I \preceq A \preceq LI$인 모든 $A$에 대해 성립해야 하니
- $l(1, \eta) = -\nu \eta$가 $\eta \in [\mu, L]$에 대해 성립 (consistency)
- **목표** : $\eta \in [\mu, L]$에 대한 $l(\lambda, \eta)$의 root radius의 최댓값을 최소화해야 함

을 알 수 있습니다. 우리가 원하는 알고리즘을 유도하려면 2-SCLI를 봐야합니다. 

즉, $p=2$인 경우를 봅시다. 이때 문제를 풀기 위해서, Lemma의 직관을 이용,

$l(\lambda, \mu) = (\lambda - (1 - \sqrt{-\nu \mu}))^2$

$l(\lambda, L) = (\lambda - (1 - \sqrt{-\nu L}))^2$

이 성립하도록 $a_j, b_j$를 설계합니다. 이렇게 하면 $a_0, a_1, b_0, b_1$를 $\nu$에 대한 식으로 표현할 수 있습니다. 
- $\nu = - 1/L$인 경우, Accelerated Gradient Descent를 얻습니다. 
- $\displaystyle \nu = - \left( \frac{2}{\sqrt{L} + \sqrt{\mu}} \right)^2$인 경우, Heavy Ball Method를 얻습니다. 
  
마지막 단추를 채웁시다. 2-SCLI 알고리즘은 엄연히 Quadratic Function의 최소화에만 사용할 수 있는 알고리즘입니다. 
이에 비해, Accelerated Gradient Descent는 임의의 smooth, strongly convex function에 대하여 잘 작동합니다. 
이 사이의 간격을 채울 수 있을까요? 결론만 설명하자면, 다음과 같은 결과를 증명할 수 있습니다. 
  
Linear Coefficient Matrix를 사용하는 경우, SCLI 알고리즘을 실제 Gradient를 이용하는 알고리즘으로 쉽게 변환이 가능합니다. 
단순히 $Ax + b$를 $\nabla f(x)$에 대응시키면 됩니다. 
또한, initialization이 optimal point에 충분히 가깝게 이루어지는 경우, 대응되는 Gradient을 이용하는 알고리즘은 기존 SCLI 알고리즘과 같은 속도로 수렴합니다. 

initialization이 충분히 잘 되어야 한다는 조건은 Heavy Ball Method의 경우에서 알 수 있듯이 꼭 필요합니다.

# Applications on Minimax Problems

## An Introduction to Minimax Problems

여기서 다루는 문제는 minimax 문제로, 다음 형태를 가집니다. 

$\min_{x \in \mathbb{R}^m} \max_{y \in \mathbb{R}^n} f(x, y)$

ML/AI를 공부한 분이라면 GAN 등 여러 대상에서 이러한 형태의 최적화 문제를 만난 적이 있을 것입니다. 
여기서는 특히 $f$가 smooth 하고, $x$에 대하여 convex, $y$에 대하여 concave 한 경우를 봅니다. 
이 경우, 이 문제를 smooth convex-concave saddle point problem 이라고도 부릅니다. 빠르게 이 문제에 대한 결과를 설명하겠습니다. 
- 이 문제에 대한 최적의 수렴속도가 $\mathcal{O}(1/N)$임은 증명되었습니다. 
- 특히, $\mathcal{O}(1/N)$의 수렴속도를 갖는 알고리즘이 이미 있습니다.
- 하지만 이들은 Ergodic Average에 대한 결과로, Last Iterate에 대한 결과가 아닙니다. 

이제부터 

$Z_{*} = (x_{*}, y_{*}), \quad Z_N = (x_N, y_N)$

라는 notation을 사용하겠습니다. 첫 번째 notation은 optimal point, 두 번째 notation은 iterate입니다. 

앞서 언급한 결과의 의미는, 계산된 iterate들이 순서대로 $Z_1, Z_2, \cdots, Z_k$라면, 

$\overline{Z_k} = \frac{1}{k}(Z_1 + Z_2 + \cdots + Z_k)$

이 얼마나 optimal 한지에 대한 bound가 증명되어 있지만, 정작 $Z_k$에 대한 결과는 아니라는 것입니다. 

[10]에서 증명된 관련된 결과의 일부를, 증명없이 몇 가지 소개하겠습니다. 

$V(x, y) = (\nabla_x f, -\nabla_y f)$라 하면, 우리의 목표는 

$\langle V(z_*), z - z_* \rangle \ge 0$

이 모든 $z$에 대하여 성립하게 되는 $z_*$를 찾는 것과 같습니다. 여기에 추가하는 가정은
- 위 조건을 만족하는 $z_*$가 존재한다고 가정합니다.
- $V$가 $\beta$-Lipschitz 하다고 가정합니다. 즉, $\lVert V(z') - V(z) \rVert \le \beta \lVert z'-z \rVert$.
- $V$가 monotone 하다고 가정합니다. 즉, $\langle V(z') - V(z), z' - z \rangle \ge 0$
  
물론, smooth convex-concave saddle point problem은 기본적으로 위 사실을 만족합니다. 

최적화 알고리즘에서는 한 점이 optimal point와 얼마나 다른지 수치적으로 표현할 방법이 필요합니다. 
단순히 목표가 $f$의 값을 최소화하는 경우에는 $f(x) - f_*$를 그 값으로 사용할 수 있었습니다. 
이제는 푸는 문제가 완전히 다르니, 다른 performance measure가 필요합니다. 이 논문에서는 

$\text{Err}_R(\hat{z}) = \max_{\lVert z \rVert \le R} \langle V(z), \hat{z} - z \rangle$

을 그 기준으로 삼습니다. 기존의 $\mathcal{O}(1/N)$ 수렴을 보장하는 알고리즘 중 하나인 Extra-Gradient 알고리즘은 

$Z_{t+1/2} = Z_t - \gamma_t V(Z_{t}), \quad Z_{t+1} = Z_t - \gamma_t V(Z_{t+1/2})$

형태를 갖습니다. 즉, 한 번의 iteration을 위해서 두 번의 $V$ 계산이 필요합니다. 

[10]에서는 한 번의 iteration에 한 번의 $V$ 계산이 필요한 알고리즘을 연구합니다. 그 중 하나는 

$Z_{t+1/2} = Z_t - \gamma_t V(Z_{t-1/2}), \quad Z_{t+1} = Z_t - \gamma_t V(Z_{t+1/2})$

입니다. 이를 Past Extra-Gradient (PEG) 알고리즘이라 합니다. 

**Theorem** : PEG 알고리즘에서 $\displaystyle \gamma < \frac{1}{2\beta}$를 step-size로 사용하면, 각 $R>0$에 대하여 

$\text{Err}_R(\overline{Z_N}) \le \frac{R^2 + \lVert Z_1 - Z_{1/2} \rVert^2}{2\gamma N}$

이다. 이때, $\overline{Z_N} = N^{-1} \sum_{s=1}^N Z_{s + 1/2}$는 iterate의 Ergodic Average. 

$\text{Err}_R$은 원래 $\lVert z \rVert \le R$에 대한 최댓값이지만, 여기서는 $\lVert z - Z_1 \rVert \le R$을 이용한다. 

또 다른 적당한 performance measure는 Primal-Dual Gap 또는 Nikaido-Isoda function

$\text{NI}(\hat{x}, \hat{y}) = \sup_{y} f(\hat{x}, y) - \inf_{x} f(x, \hat{y})$

입니다. 특히, 이를 $\lVert (x, y) - Z_1 \rVert \le R$에 대해서 $\sup$, $\inf$를 계산하는 "restricted variant" $\text{NI}_R$을 생각할 수 있습니다. 
이는 $\text{Err}$라는 measure 보다는 더 minimax 문제에 직관적으로 맞는 measure라고 느껴집니다.

**Theorem** : 앞에서와 같은 형태의 PEG 알고리즘에서, 각 $R>0$에 대하여 

$\text{NI}_R(\overline{Z_N}) \le \frac{R^2 + \lVert Z_1 - Z_{1/2} \rVert^2}{2 \gamma N}$

이다. 이때, $\overline{Z_N} = N^{-1} \sum_{s=1}^N Z_{s + 1/2}$는 iterate의 Ergodic Average. 

## Main Result : Last Iterate vs Ergodic

지금까지 Ergodic Average에 대한 수렴성은 Lower Bound와 Upper Bound가 up to a constant 일치하는 상황임을 알아보았습니다. 
이제부터 Last Iterate $Z_N$은 수렴하는지, 얼마나 빠르게 수렴하는지에 대해서 알아보겠습니다. 이 내용은 Golowich의 논문 [11]에서 가져왔습니다. 결론부터 말하면, 증명할 사실은
- Last Iterate에 대한 수렴속도는 $\mathcal{O}(1/\sqrt{N})$이 최선이며, 이는 실제로 가능하다.
- 즉 "Last Iterate is Slower than Averaged Iterate" ([11]의 논문 제목)

먼저 smooth convex-concave 문제의 Upper Bound를 봅시다. 필요한 가정은 
- $f$는 smooth convex-concave 하고, 두 번 연속적으로 미분이 가능
- $F(x, y) = (\nabla_x f, -\nabla_y f)$라 할 때 ($F$는 monotone) 다음이 성립.
- $F$는 $L$-Lipschitz : $\lVert F(z') - F(z) \rVert \le L \lVert z'-z \rVert$
- $F$는 $\Lambda$-Lipschitz한 derivative를 가짐 : $\lVert \partial F(z') - \partial F(z) \rVert \le \Lambda \lVert z' - z \rVert$
- $F(x, y) = 0$인 $(x, y)$가 존재함 : 즉, 원하는 해가 존재함

이때, 앞서 소개한 Extra-Gradient 알고리즘을 사용했을 때 다음 결과를 얻을 수 있습니다. 

**Theorem** : 초기점 $Z_0$를 잡고, $\lVert Z_0 - Z_* \rVert \le D$인 optimal point $Z_*$가 존재한다고 가정하자. 
step-size의 크기가 

$\displaystyle \gamma \le \min \left( \frac{5}{\Lambda D}, \frac{1}{30L} \right)$

을 만족한다면, $F$의 크기에 대한 bound

$\lVert F(Z_N) \rVert \le \frac{2D}{\eta \sqrt{N}}$

이 성립하며, Primal-Dual Gap에 대한 아래 bound 역시 성립한다.

$\sup_{y' \in \mathcal{B}(y_*, D)} f(x_N, y') - \inf_{x' \in \mathcal{B}(x_*, D)} f(x', y_N) \le \frac{2\sqrt{2} D^2}{\eta \sqrt{N}}$

이는 우선 $\mathcal{O}(1/\sqrt{N})$에 해당하는 Upper Bound를 증명하기 충분한 결과입니다. 
- 아래 내용과 큰 관련은 없지만, Proximal Point 알고리즘도 이를 만족함이 [11]에서 증명됩니다. 

문제는 Lower Bound입니다. [11]은 1-SCLI 알고리즘을 다루고, 이는 Extra-Gradient 알고리즘를 포함합니다.
앞서 SCLI 알고리즘을 다룰 때 풀려고 한 문제와 지금 풀려고 하는 문제는 다르니, 약간의 재정의가 필요합니다. 

이제부터 특히 집중해서 볼 $f$는 bilinear function, 즉 

$f(x, y) = x^TMy + b_1^Tx + b_2^Ty$

입니다. 이들은 bilinear하므로 convex-concave하고,

$z = \left( \begin{matrix} x \\ y \end{matrix} \right), \quad A = \left( \begin{matrix} 0 & M \\ -M^T & 0 \end{matrix} \right), \quad b = \left( \begin{matrix} b_1 \\ -b_2 \end{matrix} \right) $

라 하면 $f$에 대응되는 $F$는 

$F(z) = Az + b$

가 됩니다. 결국 다시 $Az+b=0$을 푸는 문제가 되었습니다. 우리가 이제 가정하는 조건은 
- $M$은 $n/2 \times n/2$ matrix, $b_1, b_2, x, y$는 $\mathbb{R}^{n/2}$의 원소
- $F$는 $L$-Lipschitz : $\lVert F(z') - F(z) \rVert \le L \lVert z'-z \rVert$
- $M, A$은 full rank고, $\lVert A^{-1}b \rVert = D$ (뒤에서 시작점 $Z_0 = 0$을 가정할 것입니다)

이 조건을 만족시키는 $F$의 집합을 

$\mathcal{F}_{n, L, D}$

라 부르겠습니다. 
목표는 smooth convex-concave의 일부인 bilinear function만을 보더라도, 
$\mathcal{O}(1/\sqrt{N})$의 Lower Bound가 1-SCLI 알고리즘에 대하여 성립한다는 것을 보이는 것입니다. 
특히, 여기서는 deterministic한 1-SCLI 알고리즘에 대해서만 증명하도록 하겠습니다. 이 경우, 1-SCLI 알고리즘은 정의에서 

$Z_t = C_0(A) Z_{t-1} + N(A) b$

형태를 가집니다. 다시 Extra-Gradient 알고리즘을 돌아보면 이는 

$Z_t = (I - \gamma A + (\gamma A)^2) Z_{t-1} - (I - \gamma A) \gamma b$

라는 형태로 변형되고, 결국 1-SCLI 알고리즘이 됨을 알 수 있습니다.
 
이제 iteration complexity를 재정의합시다. 우리가 iterate이 얼마나 optimal한지를 performance measure $\mathcal{L}$을 이용하여 
계산한다고 합시다. 예를 들어, $\mathcal{L}$을 Primal-Dual Gap으로 둘 수 있습니다. 

이제, SCLI 알고리즘 $\mathcal{A}$의 iteration 횟수 $N$과 performance measure $\mathcal{L}$에 대한 iteration complexity를 

$\mathcal{IC}_{n, L, D}(\mathcal{A}, \mathcal{L}, N) = \sup_{F \in F_{n, L, D}}  \mathcal{L}( Z_N(F) ) $

라고 둡니다. 즉, iteration complexity는 
- 가능한 범위의 $F$를 모두 고려했을 때, $N$번째 iterate가 가질 수 있는 최악의 performance measure

라고 해석할 수 있습니다. 그러니, 이 값이 $\Omega(1/\sqrt{N})$임을 보이면 Lower Bound가 증명됩니다. 

이 글에서 다룰 performance measure는 Upper Bound와 마찬가지로 두 가지입니다. 
- **Hamiltonian** : $\text{Ham}(Z) = \lVert F(Z) \rVert$
- **Primal-Dual Gap** : 앞서 $\sup$, $\inf$의 범위에 제약을 걸었던 것처럼, 여기서도 

$\text{Gap}(x, y) = \sup_{y' \in \mathcal{B}(y_*, D)} f(x, y') - \inf_{x' \in \mathcal{B}(x_*, D)} f(x', y)$

로 둡니다. 

논문에서는 function value에 대한 논의도 하지만, 여기서는 생략합니다.

마지막으로 consistency에 대한 정의도 해야합니다. 이는 기존의 정의처럼, 
- SCLI 알고리즘의 각 iterate이 무조건 $A^{-1}b$로 수렴하면, 그 알고리즘을 consistent하다고 한다.
  
를 그대로 따라갑니다. 이 경우, 앞서 SCLI 알고리즘을 처음 소개할 때 언급한 것처럼 
- $\mathcal{L}(1, A) = -N(A) A$. 즉, $C_0(A) = I + N(A)A$

가 성립해야 함을 확인할 수 있습니다. 이제 본격적으로 결과를 소개합니다. 

**Theorem** : $\mathcal{A}$는 consistent한 1-SCLI로, inversion matrix $N(A)$는 $A$에 대한 차수가 $k-1$ 이하인 실계수 다항식이다.
이때, 다음과 같은 iteration complexity에 대한 Lower Bound가 성립한다. 

(1) : **Hamiltonian**에 대해서, $\Omega(1/\sqrt{N})$의 Lower Bound 

$\mathcal{IC}_{n, L, D}(\mathcal{A}, \text{Ham}, N) \ge \frac{LD}{k\sqrt{20N}}$

(2) : **Primal-Dual Gap**에 대해서, $\Omega(1/\sqrt{N})$의 Lower Bound

$\mathcal{IC}_{n, L, D}(\mathcal{A}, \text{Gap}, N) \ge \frac{LD^2}{k\sqrt{20N}}$

Remark : 앞서 Upper Bound는 $Z_{s+1/2}$의 Ergodic Average에 대해서 얻었고, 
지금의 Lower Bound는 $Z_s$들에 대한 것입니다. 이 차이가 찝찝할 수 있는데, 
실제로는 위와 비슷한 Lower Bound가 $Z_{s+1/2}$에 대해서도 성립함을 보일 수 있습니다. 
즉, $Z_s$에 대한 Lower Bound를 가지고 $Z_{s+1/2}$에 대한 Lower Bound를 유도할 수 있습니다. 

위 Theorem의 증명의 흐름을 간단하게 설명하도록 하겠습니다. 

**Step 1** : 단순 계산. 일반성을 잃지 않고 $Z_0 = 0$이라 합시다. 귀납법 및 단순 계산으로

$\text{Ham}(Z_t) = \lVert C_0(A)^t b \rVert, \quad \text{Gap}(Z_t) = D \lVert C_0(A)^t b \rVert = D \cdot \text{Ham}(Z_t)$

임을 보일 수 있습니다. 그러니 사실 (1)만 보이면 (2)가 자동으로 증명됩니다. 

**Step 2** : $M, A, b_1, b_2$의 정의, 그리고 문제를 다항식 문제로 바꾸기. 

적당한 $\nu \in (0, L]$을 변수로 잡아줍시다. 이제 $M, b_1, b_2$를 각각 

$M = \nu I, \quad b_1 = b_2 = \frac{\nu D}{\sqrt{n}} (1, 1, \cdots ,1)^T, \quad A = \left( \begin{matrix} 0 & M \\ -M^T & 0 \end{matrix} \right), \quad b = \left( \begin{matrix} b_1 \\ -b_2 \end{matrix} \right) $

로 잡아줍시다. $b_1, b_2$는 $\mathbb{R}^{n/2}$의 원소고, $M$ 역시 $n/2 \times n/2$ 행렬입니다. 

여기서 $\nu \in (0, L]$인 이유는 $F$의 $L$-Lipschitz 여부 때문입니다. 이제 

$D = \lVert A^{-1}b \rVert = \nu^{-1} \lVert b \rVert$

이므로, 이를 이용하면 

$\text{Ham}(Z_t) = D \cdot \frac{\nu \lVert C_0(A)^t b \rVert}{\lVert b \rVert}$

임을 얻습니다. 한편, $C_0(A) = I + N(A)A$이므로 $C_0(A)$는 $A$에 대한 $k$차 다항식입니다. 
이를 식으로 쓰면 

$C_0(A) = q_0 I + q_1 A + \cdots + q_k A^k$

이고, 특히 $q_0 = 1$입니다. 이제 

$q(x) = q_0 + q_1 x + \cdots + q_k x^k$

라고 정의합시다. $\text{Ham}(Z_t)$를 보면, $C_0(A)$의 eigenvalue가 매우 중요하게 생겼습니다. 
$C_0(A)$ 자체는 $A$에 대한 다항식이니, $A$의 eigenvalue만 보면 충분합니다. 그런데 
$A$의 eigenvalue는 $\nu i$가 $n/2$개, $-\nu i$가 $n/2$개임을 쉽게 확인할 수 있습니다. 
그러므로 $C_0(A)$의 eigenvalue도 $q(\nu i)$가 $n/2$개, $q(-\nu i) = \overline{q(\nu i)}$가 $n/2$개가 됩니다. 

게다가 $A$의 특수한 형태 때문에 실제로 임의의 $b \in \mathbb{R}^n$에 대하여 

$\lVert C_0(A) b \rVert = |q(\nu i)| \cdot \lVert b \rVert$

가 성립함을 확인할 수 있습니다. 정리하면 

$\sup_{\nu \in (0, L]} \frac{\nu \lVert C_0(A)^t b \rVert}{\lVert b \rVert} = \sup_{\nu \in (0, L]} \nu |q(\nu i)|^t$

입니다. 이제 위 식의 우변에 대한 결과를 내면 됩니다. 즉, 
- **조건** : $q$는 $q(0) = 1$이고 $k$차 실계수 다항식
- **목표** : $\sup_{\nu \in (0, L]} \nu \lvert q(\nu i) \rvert^t$에 대한 좋은 Lower Bound

**Step 3** : 다항식에 대한 문제 해결. 우선 문제를 별 차이가 없는 

$\sup_{\nu \in (0, L]} \nu^2 \lvert q(\nu i) \rvert^{2t}$

에 대한 문제로 바꾸겠습니다. 생각해보면 

$\lvert q(\nu i) \rvert \ge \lvert \mathfrak{R} (q(\nu i)) \rvert = \lvert \sum_{i=0}^{\lfloor k/2 \rfloor} q_{2i} (-1)^i \nu^{2i} \rvert $

이므로, 결국 다음 부등식이 성립합니다.

$\sup_{\nu \in (0, L]} \nu^2 \lvert q(\nu i) \rvert^{2t} \ge \sup_{\nu \in (0, L]} \nu^2 \lvert \sum_{i=0}^{\lfloor k/2 \rfloor} q_{2i} (-1)^i \nu^{2i} \rvert^{2t}$

이제 $\nu^2$을 $y$로 치환하면, 위 식의 우변을 

$\sup_{y \in (0, L^2]} y \lvert \sum_{i=0}^{\lfloor k/2 \rfloor} q_{2i} (-1)^i y^i \rvert^{2t}$

라고 둘 수 있습니다. 이는 결국 $p(0)=1$인 차수 $\lfloor k/2 \rfloor$ 이하인 다항식 $p$에 대하여 

$\sup_{y \in (0, L^2]} y |p(y)|^{2t}$

에 대한 bound를 요구하는 것과 같습니다. Chebyshev Polynomial 느낌의 문제가 되었습니다. 

**Lemma** ([12]) : 자연수 $k$와 실수 $L > \mu > 0$와 차수 $k$ 이하이고 $p(0) = 1$인 실계수 다항식 $p(y)$가 있다. 
또한, $T_k$를 $k$번째 Chebyshev Polynomial of Second Kind이라 하고

$q(y) = T_k \left( \frac{2y - (\mu + L)}{L - \mu} \right) \bigg/ T_k \left( \frac{L + \mu}{L- \mu} \right)$

라 정의하자. 그러면 $q$ 역시 $q(0) = 1$를 만족하는 차수 $k$ 이하인 실계수 다항식이며, 

$\sup_{y \in [\mu, L]} |p(y)| \ge \sup_{y \in [\mu, L]} |q(y)|$

가 성립한다. 즉, 범위에서 다항식의 절댓값의 최댓값을 최소화하는 문제는 Chebyshev Polynomial이 해결한다. 

**Lemma** : 자연수 $k$와 실수 $L > \mu > 0$와 차수 $k$ 이하이고 $p(0) = 1$인 실계수 다항식 $p(y)$가 있다. 이때, 

$ \sup_{y \in [\mu, L]} |p(y)| > 1 - \frac{6k^2}{(\sqrt{L/\mu} - 1)^2}$

**Lemma** : 자연수 $k$와 실수 $L>0$, 그리고 차수 $k$ 이하이고 $p(0) = 1$인 실계수 다항식 $p(y)$가 있다. 이때, 

$ \sup_{y \in (0, L]} y |p(y)|^t > \frac{L}{40tk^2}$

이는 결국 원하는 형태의 Lower Bound인 

$\sup_{y \in (0, L^2]} y |p(y)|^{2t} \ge \frac{L^2}{20tk^2}$

을 유도해주고, 원래 문제로 돌아가면 결국 (1)의 결과를 증명해줍니다. 

## Further Reading on Minimax Problems and SCLI

아주 간략하게 두 개의 논문을 소개합니다. 

[13]은 $p$-SCLI framework를 $n$명이 참가하는 게임에 대해 확장합니다. 
즉, 게임에 대한 Nash 균형을 빠르게 찾는 문제를 해결하는 것에 대한 Lower Bound를 증명합니다. 
처음 $p$-SCLI를 소개한 [9]와 굉장히 유사한 결과들을 게임에 대하여 증명하는 논문입니다. 
특히, $p, n$이 매우 작은 경우에 대해서는 여러 explicit bound를 제시합니다. 

[14]은 strong convexity가 있는 경우에서 (정확하게는 strongly monotone) 1-SCLI를 분석합니다.
상당히 난이도가 있어보이는 논문으로, 저도 아직은 자세히 읽지 못했습니다. 공부가 더 필요할 것 같습니다.

# Conclusion

지금까지의 흐름을 정리해보는 것으로 글을 마치겠습니다.  

Introduction
- optimization algorithm이 필요로 하는 iteration 횟수에 대한 Lower Bound는 흥미로운 문제
- 완벽한 Lower Bound의 계산은 매우 어렵지만, up to constant 일치하는 Lower Bound는 조금 더 쉬움
- 특히, smooth convex minimization 문제에서 up to constant 일치는 Quadratic Function만으로 가능
- 그런데 여기서 문제의 dimension에 관련된 제약이 있었고, 이 제약은 우리의 context와 거리가 있음
- 이를 해소하기 위해서 다룰 알고리즘의 폭을 줄였고, 이것이 $p$-SCLI 알고리즘 
  
SCLI에 대한 소개와 [9]
- $p$-SCLI 알고리즘에서는 다루는 문제가 Quadratic Minimization임
- 이 경우 $p$-SCLI 알고리즘은 일종의 Linear Transformation이 됨
- 그러니 Spectral Radius가 튀어나오고, 나아가서 다항식의 root에 대한 논의가 나오게 됨
- $p$-SCLI 알고리즘에 대한 기본적인 bound를 이를 통해서 얻을 수 있었고
- 이 bound는 up to constant 일치하지만 대응되는 알고리즘이 실제로 적용하기 어려웠음
- 이에 대응하기 위해 SCLI 알고리즘 중 실제로 계산이 쉽게 가능한 것으로 알고리즘의 폭을 더욱 줄였음
- 이것이 Linear Coefficient Matrix고, 여기서 특히 $p=2$인 경우를 이용해 여러 알고리즘을 유도할 수 있었음

Minimax 문제와 SCLI
- Minimax 문제는 매우 중요한 문제인데, Ergodic Average에 대한 결과만 알려져 있었음
- SCLI Framework를 이용해서, Last Iterate이 Ergodic Average보다 느리게 수렴함을 증명할 수 있었음 
- 결론적으로 Ergodic Average는 $\mathcal{O}(1/N)$, Last Iterate은 $\mathcal{O}(1/\sqrt{N})$으로 수렴하고 이게 최적
  
SCLI는 최근에 등장한 Framework이니, 이후에도 더 많은 곳에서 쓰일 수 있을 것이라고 기대됩니다. 

# Related Links & References
- [1] : arxiv.org/pdf/2101.09545.pdf (arXiv preprint 2021) 
- [2] : github.com/rkm0959/rkm0959_presents/blob/main/acceleration_convex_optimization.pdf
- [3] : arxiv.org/pdf/1206.3209.pdf (Mathematical Programming 2014)
- [4] : arxiv.org/pdf/1406.5468.pdf (Mathematical Programming 2016)
- [5] : arxiv.org/pdf/1606.01424.pdf (Journal of Complexity 2017)
- [6] : arxiv.org/pdf/2101.09741.pdf (arXiv preprint 2021)
- [7] : arxiv.org/pdf/2101.09740.pdf (arXiv preprint 2021)
- [8] : arxiv.org/pdf/1408.3595.pdf (SIAM Journal on Optimization 2016)
- [9] : arxiv.org/pdf/1503.06833.pdf (JMLR 2016)
- [10] : arxiv.org/pdf/1908.08465.pdf (NeurIPS 2019)
- [11] : arxiv.org/pdf/2002.00057.pdf (COLT 2020)
- [12] : arxiv.org/pdf/1605.03529.pdf (ICML 2016)
- [13] : arxiv.org/pdf/1906.07300.pdf (ICML 2020)
- [14] : arxiv.org/pdf/1906.05945.pdf (AISTATS 2020)
