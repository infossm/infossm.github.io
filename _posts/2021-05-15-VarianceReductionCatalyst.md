---

layout: post

title:  "Variance Reduction Algorithms and Catalyst Acceleration"

date: 2021-05-04

author: rkm0959

tags: [optimization]

---

# 서론 
본 글에서는 단순히 convex function $f$를 최소화 하는 것이 아니라, 이들의 합인 

$$F = \frac{1}{n} \sum_{i=1}^n f_i(x)$$

를 최소화하는 알고리즘에 대해서 알아보고, 이에 Catalyst를 적용해보겠습니다. 

이와 같은 형식의 함수 $F$는 Logistic Regression 등 Machine Learning에서 등장합니다.

원래는 $F$에 추가적인 "proximal function"이 있어, 

$$F = \frac{1}{n} \sum_{i=1}^n f_i(x) + \psi(x)$$

형태를 가지나 (예를 들어 $l_1$ regularization) 여기서는 편의상 $\psi$에 대한 논의를 생략하겠습니다.



글의 순서는 대략적으로 다음과 같습니다. 

- $F$를 최소화하는 알고리즘이 발전한 과정의 큰 흐름에 대해서 알아보겠습니다.
- $F$를 최소화하는 알고리즘 중 SAGA에 대해서 알아보고, 구현 방식도 알아보겠습니다.
- SAGA에 Catalyst Acceleration을 적용하는 방법을 알아보고, 논문의 실험을 재현해보겠습니다. 
  
본 글은 4월 글에 이어지는 글이니, 해당 글을 읽고 이 글을 읽는 것을 추천합니다.  

# 본론 

## 최적화 알고리즘의 발전 과정

$F$ 역시 convex function이므로, $F$를 하나의 함수로 보고 이에 대하여 first-order method를 적용하여 최적화를 진행할 수 있습니다. 그러나, $F$의 gradient를 계산하려면 $\nabla f_i(x)$를 모든 $i$에 대하여 전부 계산해야 한다는 점에서, 이는 효율적인 알고리즘이 되기 어려워 보입니다. 그래서 사람들은 $\nabla F(x)$를 전부 계산하지 않고 최적화를 하는 알고리즘에 대하여 고민하기 시작합니다. 

Allen-Zhu는 2016년 Katyusha Algorithm을 소개하는 논문의 Introduction에서, $F$를 최소화하는 알고리즘의 발전 과정을 크게 세 단계로 분류하였습니다. 이 글에서는 이 세 단계를 모두 볼 수 있습니다.

아래의 설명은 대부분 Allen-Zhu의 논문의 번역이고, 제가 직접 쓴 글과 거리가 멉니다. 

### 단계 1 : Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent의 아이디어는 각 단계마다 $\nabla F(x_k)$를 직접 계산하지 않고, 조건  

$$ \mathbb{E}[ \tilde{\nabla}_k ] = \nabla F(x_k)$$ 

을 만족하는 random vector $\tilde{\nabla}_k$를 잡은 다음 Gradient Descent 

$$ x_{k+1} = x_k - \gamma \tilde{\nabla}_k$$

를 적용하는 방법을 말합니다. 특히, $\tilde{\nabla}_k$의 값은 $1 \le i \le n$을 uniform random 하게 선택한 후, $\nabla f_i(x_k)$를 계산하면 얻을 수 있습니다. 이 방식은 한 iteration을 계산하는 과정이 기존의 $\nabla F(x)$를 전부 계산하는 방법보다 매우 빠르나, 수렴 속도가 빠르지 않다는 점에서 아쉬운 면을 가지고 있습니다. 

### 단계 2 : Variance Reduction (SVRG, SAG, SAGA, etc)

SGD를 발전시킨 다음 아이디어는 Variance Reduction입니다. 간단하게 설명하자면, 

$$ \text{Var}[\tilde{\nabla}_k]$$

의 값이 작다면 수렴의 속도도 빨라질 것이라는 아이디어입니다. 이 목표를 이루는 방법은 다양하지만, 예시를 하나만 (SVRG) 들어보겠습니다. SGD를 적용하면서 $2n$번마다 한 번씩 현재 vector를 snapshot vector $\tilde{x}$로 설정하고, 이때 $\nabla F(\tilde{x})$를 전부 계산해줍시다. $2n$번마다 계산하는 것이므로, 전체 계산량에는 큰 타격이 없습니다. 이제, $\nabla F(x_k)$에 대한 estimator를 $\nabla f_i(x_k)$로 두지 않고, 대신 

$$ \tilde{\nabla}_k = \nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla F(\tilde{x})$$

로 설정합시다. 이 방법을 적용하면 variance가 감소하고, 이에 따라 수렴 속도도 빨라집니다. $F$가 $\mu$-strongly convex, $L$-smooth이고 $\kappa = L / \mu$라 할 때, 위 알고리즘들은 $O((n + \kappa) \log 1/\epsilon)$ 정도의 계산을 필요로 합니다. 여기에 속하는 알고리즘 중 하나가 SAGA이고, 이 글에서 알아보겠습니다. 

### 단계 3 : Acceleration (Catalyst, etc)

이 단계에서는 Acceleration을 적용하여 $\kappa$의 값이 큰 경우 시간복잡도가 $\kappa$ 대신 $\sqrt{\kappa}$에 비례하도록 합니다. 즉, 이번 단계에서 목표로 하는 시간복잡도는 대강 

$$O((n + \sqrt{n\kappa}) \log 1/\epsilon)$$

정도가 됩니다. 이전에 살펴보았던 Catalyst Acceleration은 이와 비슷한 수준의 성능을 가지고 있습니다. 이 section의 내용은 앞에서도 언급했다시피 Allen-Zhu의 Katyusha 논문에서 가져온 것인데, 이 알고리즘은 Catalyst Acceleration보다 강한 성능을 보이고 정확히

$$O((n + \sqrt{n\kappa}) \log 1/\epsilon)$$

의 계산을 필요로 합니다. 이 방법에 대한 논의는 미래의 글로 미루도록 하겠습니다. 

참고로 이러한 최적화 문제를 해결하는 알고리즘이 필요로 하는 최선의 시간복잡도의 lower bound에 대한 연구 결과가 있습니다. http://proceedings.mlr.press/v37/agarwal15.pdf를 참고하세요.

## SAGA와 그 구현 준비

우선 SAGA 알고리즘의 원본부터 알아봅시다. 

먼저 initial vector $x^0$를 정하고 그 gradient $f_i'(x^0)$의 값들을 모두 계산합시다. 여기서 $\phi_i^0 = x^0$이라고 정의합니다. 이전 $k$번의 iteration에서 $x^k$의 값과 $f_i'(\phi_i^k)$의 값이 모두 계산되어 있다고 가정하고, 이제 $k+1$번째 iteration을 계산한다고 합시다. 이때, 다음을 순서대로 진행합니다. 

- 먼저 $j$를 $1$과 $n$ 사이에서 uniform random 하게 선택합니다.
- 이제 $\phi_{j}^{k+1} = x^{k}$이라 하고, 다른 $\phi_i^{k+1}$은 그대로 $\phi_i^k$로 둡니다.
- $f'_j(\phi_j^{k+1})$을 계산하고, 이에 맞게 $x^{k+1}$을 update 하기 위해서 

$$x^{k+1} = x^k - \gamma \left( f'_j(\phi_j^{k+1}) - f'_j(\phi_j^k) + \frac{1}{n} \sum_{i=1}^n f'_i(\phi_i^k) \right)$$

라는 식을 이용합니다. 이 알고리즘은 step-size

$$ \gamma = \frac{1}{2(\mu n + L)}$$

을 사용했을 때 $\lVert x^k - x_{*} \rVert^2$의 크기가 

$$ 1 - \frac{\mu}{2(\mu n + L)}$$

을 convergence rate로 가지며 linear 하게 수렴합니다. 

문제는 이 알고리즘의 효율적인 구현입니다. 이를 위해서는,

- $f_i(x)$가 $g(\langle x, a_i \rangle)$ 형태를 가지고 있다 가정합시다. 여기서 $a_i$는 $i$번째 데이터입니다. 이 경우, $f_i'(x)$는 사실 $a_i$의 스칼라배이므로, $f_i'(x)$를 전부 저장하지 않고 $f_i'(x) = ca_i$인 경우 $c$의 값만을 저장해도 문제없습니다. 
- 처음에 $f_i'(x^0)$을 전부 계산하는 대신, 계산된 gradient 만을 가지고 $x^{k+1}$의 값을 계산합니다. 즉,

$$ \frac{1}{n} \sum_{i=1}^n f_i'(\phi_i^k)$$

를 사용하는 대신 실제로 계산되어 저장된 gradient 들의 평균을 사용하는 것입니다. 

- $f_i(x)$가 $g(\langle x, a_i \rangle)$ 형태를 가지고 있다고 하면, 사실 $f_i$의 계산에서 중요한 $x$의 entry는 $a_i$가 nonzero entry를 갖는 index의 값들 뿐입니다 그러니, $x^{k+1}$을 전부 계산하지 말고, 다음 iteration에서 $j$를 고른 다음 $a_j$의 nonzero entry의 index만 계산하는 전략을 사용합시다. 이는 데이터가 sparse 한 경우 큰 도움이 됩니다. 
- 이번에는 regularization $\frac{\lambda}{2} \lVert x \rVert^2$이 objective에 추가되었다고 가정합시다. 이 경우, $x^{k+1}$를

$$x^{k+1} = (1 - \gamma \mu) x^k - \gamma \left( f'_j(\phi_j^{k+1}) - f'_j(\phi_j^k) + \frac{1}{n} \sum_{i=1}^n f'_i(\phi_i^k) \right)$$

로 계산해도 괜찮습니다. 이 때, $x^{k}$를 계산하는 과정에서 계속해서 scale을 곱하게 되는데, 이는 $x$의 각 entry를 계속 바꾸는 방식으로 구현하는 대신 scale 값을 하나 따로 저장하는 방식으로 효율적으로 구현할 수 있습니다. 

이러한 방식은 SAG에서도 사용되었으며, 많은 method들에서 공통적으로 사용되는 테크닉들입니다. 

## Catalyst Acceleration의 구현 준비

Catalyst의 후속 논문에서는 warm start 방법과 stopping criterion을 3개 제시합니다. 이렇게 알고리즘의 세부 디테일이 복잡하고 선택의 여지가 많다는 점은 어떻게 보면 이 알고리즘의 단점이고, 이는 앞서 언급했던 Katyusha Acceleration에서 보안되는 부분 중 하나입니다. 이 글에서는 warm start와 stopping criterion 방법을 제시만 하고, 자세한 이론적 결과는 생략하도록 하겠습니다. 이전 글처럼, strong convexity를 가정합니다.

자세한 결과가 궁금하시다면, https://arxiv.org/pdf/1712.05654.pdf를 참고하시면 됩니다. 

Catalyst Acceleration을 복습해봅시다. $y_0 = x_0$, $q = \mu / (\mu + \kappa)$를 잡고, $\alpha_0 = \sqrt{q}$를 잡습니다. 

각 iteration에서, first-order algorithm $\mathcal{M}$을 사용하여 

$$x_k \approx \text{argmin} \left( h(x) \equiv f(x) + \frac{\kappa}{2} \lVert x - y_{k-1} \rVert^2 \right) $$ 

을 찾고 $\alpha_k$를 계산하기 위하여 

$$\alpha_k^2 = (1- \alpha_k) \alpha^2_{k-1} + q \alpha_k$$

를 풉니다. 마지막으로 Nesterov's extrapolation

$$y_k = x_k + \frac{\alpha_{k-1}(1-\alpha_{k-1})}{\alpha_{k-1}^2 + \alpha_k} (x_k - x_{k-1})$$

을 적용하는 것을 반복합니다. 문제는 

- $x_k$의 계산에서 초깃값을 어떻게 설정할 것인가. (warm start)
- $x_k$를 얼마나 정확히 계산할 것인가. (stopping criterion)

### Method 1

여기서는 $x_k$의 정확도를 absolute accuracy로 측정합니다. 즉, 적당한 $\epsilon_k$에 대해

$$ h(x_k) - h_{*} \le \epsilon_k$$

인 $x_k$를 찾을 때까지 $\mathcal{M}$을 반복하면 됩니다.

$\epsilon_k$의 경우, 저번 글에서 설명하였듯이 

$$\epsilon_k = \frac{1}{2} (1-\rho)^k (f(x_0) - f_{*})$$

를 잡으면 되며, 여기서 $\rho < \sqrt{q}$로 실전에서는 $\rho = 0.9 \sqrt{q}$를 잡으면 됩니다. 

저번 글에서도 언급했던 문제는 

$$f(x_0) - f_{*}, \quad h_{*}$$

의 값을 모른다는 것인데, duality gap을 사용하여 근사하거나 $h$의 $L+\kappa$-smoothness를 이용하여 $h$의 gradient로 $h$의 gap을 근사하는 방법으로 처리할 수 있습니다. 

이때, warm start는 $\mathcal{M}$의 initial point를 

$$z_0 = x_{k-1} + \frac{\kappa}{\kappa + \mu}(y_{k-1} - y_{k-2})$$

로 잡는 것으로 할 수 있습니다. 

### Method 2

여기서는 $x_k$의 정확도를 relative accuracy로 측정합니다. 즉, 적당한 $\delta_k$에 대해 

$$ h(x_k) - h_{*} \le \frac{\delta_k}{2} \lVert x - y_{k-1} \rVert^2$$

인 $x_k$를 찾을 때까지 $\mathcal{M}$을 반복하면 됩니다. 

여기서 $h(x_k) - h_{*}$의 값을 근사하는 것은 Method 1의 방법과 비슷하게 하면 됩니다. 문제는 $\delta_k$를 잡는 것이며, 

$$ \delta_k = \frac{\sqrt{q}}{2 - \sqrt{q}}$$

로 잡는 것으로 충분합니다. 

warm start의 경우, 여기서는 단순히 $z_0 = y_{k-1}$로 잡는 것으로 충분합니다. 

### Method 3

이번에는 criterion이 아니라 heuristic에 가까운 접근을 취합니다. 

stopping criterion은 특별히 없고, 단순히 $\mathcal{M}$의 iteration 횟수를 데이터의 갯수 $n$으로 고정해버립니다. 예를 들어, Catalyst-SAGA에서는 $\mathcal{M}$을 SAGA로 잡는데, 여기서 "stochastic gradient descent"를 $n$번 하고 그 결과가 얼마나 정확한지 확인하지 않고 바로 $\mathcal{M}$을 중단한다는 것입니다. 

이 경우, warm start는 Method 1의 방식으로 얻은 initial point를 사용하거나 $x_{k-1}$을 initial point로 사용합니다. 둘 중 어느 것을 사용할 것인지는 $h_k$ 값이 어느 점에서 더 작냐로 결정합니다. 

아이러니컬하게도, 가장 이론적 근거가 떨어져보이는 Method 3이 실험적으로는 가장 성능이 좋습니다. 

## SAGA와 Catalyst-SAGA의 비교

이제 본격적으로 구현을 할 준비가 완료되었습니다. 실험에서는 
- Catalyst Acceleration의 Method 3을 사용합니다.
- LibSVM에 있는 real-sim dataset을 사용합니다. 
- $l_2$-regularized logistic regression 문제를 해결합니다.
- 논문과 같은 $\mu = 1/(2^8n)$을 regularization으로 사용합니다. 

결과는 https://github.com/rkm0959/rkm0959_implements/tree/main/Catalyst_Acceleration 에서 확인할 수 있습니다. 참고로, 원 논문과 SAGA algorithm의 step size를 다르게 정하여 논문과 결과가 다릅니다.

# 결론

본 글에서는 저번 글에서 알아보았던 다양한 Variance Reduction을 활용한 알고리즘들의 구현 방법과, 이에 Catalyst Acceleration을 실제로 구현하는 방법을 알아보았습니다. 이를 실제로 논문에서 사용된 데이터셋에 적용하여, Catalyst Acceleration의 힘을 확인하였습니다. 다음에 최적화를 다루는 글에서는 앞선 글에서는 다루지 못했던 엄밀한 증명에 대해서 더욱 집중해볼 예정입니다. 감사합니다.

