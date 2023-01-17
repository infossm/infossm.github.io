---

layout: post

title:  "Catalyst Acceleration"

date: 2021-04-15

author: rkm0959

tags: [optimization]

---

# 서론 

Convex Optimization의 주요 목적 중 하나는, convex function $f$가 있을 때 최적화 문제 

$ f_{*} = \min_{x \in \mathbb{R}^d} f(x) $

를 효율적으로 해결하는 것입니다. 특히, $f$가 수학적으로 좋은 조건을 만족하는, 즉 convex, closed, proper function인 경우를 (CCP function) 주로 다룹니다. Gradient Descent와 같은 first-order algorithm들은 이 목표를 달성하기 위해서 Gradient 또는 Subgradient를 이용합니다. 

이제부터 다룰 대상은 Gradient를 사용하여 최적화를 하는 대신, Proximal Operator 

$ prox_{\lambda f}(x) = \text{argmin}_y \left( f(y) + \frac{1}{2\lambda} || x-y||^2 \right) $

를 활용하여 최적화를 진행하는 방법입니다. 글의 흐름은 다음과 같습니다. 


- 먼저 first-order algorithm이 얼마나 효율적인지 복습합니다.
- Proximal Operator의 의미와, 특징들을 간단하게 살펴봅니다.
- Proximal Operator를 활용한 알고리즘들의 예시를 살펴봅니다.
- Proximal Operator를 활용하여 다양한 first-order algorithm의 성능을 강화하는 Framework인 Catalyst Acceleration에 대하여 살펴보고, 몇몇 사례에 대한 이론적인 성능 분석을 합니다.

이 글에서는 계산이 복잡한 수학적 증명을 생략하도록 하겠습니다. 대부분의 경우, 이러한 증명들은 first-order methods에서도 많이 등장하는 Lyapunov Analysis를 이용하여 증명이 됩니다. 

이 글은 Gradient Descent등 기초적인 최적화 알고리즘에 대한 사전지식을 가정합니다.

이 글은 Acceleration에 대한 survey paper인 https://arxiv.org/pdf/2101.09545.pdf의 5장 내용에 일부 내용을 더하고 빼서 정리한 글이기도 합니다. Lyapunov Analysis와 first-order algorithm에 대한 내용이 4장에 있으니, 읽어보시면 좋을 것 같습니다. 최근에 4장 내용과 관련된 세미나를 했는데, 그 자료는 제 개인 블로그 https://rkm0959.tistory.com/221 에서 찾아보실 수 있습니다. 해당 내용을 읽고 오시는 것을 추천합니다.

# 본론 

# First-Order Algorithm의 복습 

이제부터 $\mu$-strongly convex, $L$-smooth CCP function의 집합을 $\mathcal{F}_{\mu, L}$이라 합시다. 
추가적으로 $f$가 미분 가능한 함수라고 가정합시다. 이제부터 first-order algorithm에 대한 중요 결과들을 소개합니다.

잠깐 익숙한 대상인 Gradient Descent로 넘어가봅시다. 이 알고리즘은, $f \in \mathcal{F}_{0, L}$의 최솟값을 구하기 위해서 initial point $x_0$를 잡고 Gradient를 활용하는 update

$x_{k+1} = x_k - \frac{1}{L} \nabla f(x_k)$

를 반복해서 적용합니다. 이때, **Lyapunov Analysis**라는 증명 기법을 이용하면 

$ f(x_N) - f_{*} \le \frac{L ||x_0 - x_{*}||^2}{2N}$

이라는 결과를 얻을 수 있습니다. 즉, "실제 답과의 차이"가 $O(1/N)$ 속도로 감소함을 증명할 수 있습니다.

이제 나타나는 자연스러운 질문은, **"이것보다 더 빠르게 갈 수 있을까"** 입니다. 정확하게는, initial point $x_0$에서 시작하고 $f$나 $\nabla f$의 값을 계산하는 것을 $N$번 시행하여 얻은 결과가 $x_N$일 때, "실제 답과의 차이" $f(x_N) - f_{*}$가 얼마나 작을 수 있는가를 물어볼 수 있습니다. 수학자들이 이에 대하여 얻은 결론을 간단하게 정리하면, 

- $O(1/N^2)$ 속도로 감소하는 "accelerated" 알고리즘이 존재함
- 그보다 빠른 속도로 감소하는 알고리즘은 존재하지 않음

입니다. 비슷하게, $f \in \mathcal{F}_{\mu, L}$에 대해서도 같은 질문을 할 수 있습니다. 이 경우에는, Gradient Descent의 경우 $O((1-\mu/L)^N)$의 속도로 수렴하고, 최적의 알고리즘은 $O((1-\sqrt{\mu / L})^{2N})$의 속도로 수렴합니다.

자세한 내용은 Survey Paper의 4장이나, 제 세미나 자료를 참고하시기 바랍니다. 중요한 점은 Gradient를 사용하는 알고리즘들은 수렴 속도에 한계가 있다는 것입니다. 그럼 Gradient보다 더 강력한 도구를 쓰면 어떨까요?

# Proximal Operator와 Inexactness

Proximal Operator는, 위에서도 언급했듯이

$ prox_{\lambda f}(x) = \text{argmin}_y \left( f(y) + \frac{1}{2\lambda} || x-y||^2 \right) $

로 정의됩니다. 이는 $f$가 CCP function이라면 항상 well-defined 함수입니다. 정의 자체가 하나의 optimization problem이니, 그 계산이 확실히 Gradient/Subgradient 보다는 어렵겠죠? 하지만 proximal operator는 많은 함수 $f$에 대해서 계산이 쉽고, 기본적으로 가지고 있는 좋은 성질들이 많아서 자주 사용되는 도구 중 하나입니다. 

그런데 Proximal Operator를 통해서 $f$를 최소화한다는 말은 뭔가 이상합니다. 당장 $\lambda$를 매우 크게 잡으면, 

$ f(y) + \frac{1}{2\lambda} || x - y||^2 \approx f(y)$

이니, 이때 proximal operator의 값이 $f$를 최소화하는 점이 되는 게 아닐까요? 

이는 좋은 지적입니다. Proximal Operator의 정확한 계산이 쉬운 경우에는 직관적으로 최소화 문제도 쉬울 겁니다. 
하지만, 신기한 점은 **Proximal Operator를 근사적으로 계산하기만 해도** 강력한 결과를 얻을 수 있다는 겁니다. 

$x_{k+1} = \text{prox}_{\lambda_k f}(y_k)$

라면, 적당한 $g_f(x_{k+1}) \in \partial f(x_{k+1})$이 있어 

$\lambda_k g_f(x_{k+1}) + x_{k+1} - y_k = 0$

이 성립합니다. 그러니 Inexact한 계산에서는 좌변의 값이 $0$ 대신 작은 값이도록 하면 되겠습니다. 즉,

$e_k = \lambda_k g_f(x_{k+1}) + x_{k+1} -y_k$

라 할 때, $\lVert e_k \rVert \le \delta \lVert x_{k+1} - y_k \rVert$가 성립한다면 

$x_{k+1} \approx_{\delta} \text{prox}_{\lambda_k f} (y_k)$

라고 할 수 있습니다. 이제부터 Inexact Proximal Operation으로 어떤 알고리즘을 만들 수 있는지 봅시다. 

# Accelerated Proximal Point Algorithms

먼저 Inexact Accelerated Proximal Point Method를 (Monteiro, Svaiter 2013) 봅시다.

initial point $z_0 = x_0$를 잡고 $A_0 = 0$이라 합시다. $k$에 대하여 다음을 반복합니다. 


$a_k = \frac{1}{2} \left( \lambda_k + \sqrt{\lambda_k^2 + 4A_k \lambda_k} \right)$

$A_{k+1} = A_k + a_k$

$y_k = \frac{A_k}{A_k + a_k} x_k + \frac{a_k}{A_k + a_k} z_k$

$x_{k+1} \approx_{\delta} \text{prox}_{\lambda_k f}(y_k), \quad \delta \in [0, 1]$

$z_{k+1} = z_k - a_k g_f(x_{k+1})$

이 경우, Lyapunov Analysis를 적용하여 다음과 같은 결론을 얻습니다. 

$f(x_k) - f_{*} \le \frac{2 ||x_0 - x_{*}||^2}{\left( \sum_{i=0}^{k-1} \sqrt{\lambda_i} \right)^2} $

즉, Inexact Proximal Operator를 잘 계산할 자신이 있다면 $\lambda_i$를 임의로 크게 하여 수렴 속도를 빠르게 할 수 있습니다. 심지어 필요한 가정도 $f \in \mathcal{F}_{0, \infty}$ 뿐입니다. Proximal Operator의 힘을 느낄 수 있습니다.

Strong Convexity 역시 활용할 수 있습니다. $f \in \mathcal{F}_{\mu, \infty}$라고 합시다.

Accelerated Hybrid Proximal Extragradient Method를 (A-HPE)
(Barre et al. 2020) 소개합니다.

initial point $z_0 = x_0$를 잡고 $A_0 = 0$이라 합시다. $k$에 대하여 다음을 반복합니다.

$A_{k+1} = A_k + \frac{1}{2} \left(\lambda_k + 2 A_k \lambda_k \mu + \sqrt{4A_k^2 \lambda_k \mu (\lambda_k \mu + 1) + 4 A_k \lambda_k (\lambda_k \mu + 1) + \lambda_k^2} \right)$

$y_k = x_k + \frac{(A_{k+1} - A_k)(A_k \mu + 1)}{A_{k+1} + 2\mu A_k A_{k+1} - \mu A_k^2} (z_k - x_k) $

$x_{k+1} \approx_{\delta} \text{prox}_{\lambda_k f}(y_k), \quad \delta \in [0, \sqrt{1 + \lambda_k \mu}]$

$z_{k+1} = z_k + \mu \frac{A_{k+1} - A_k}{1 + \mu A_{k+1}} (x_{k+1} - z_k) - \frac{A_{k+1} - A_k}{1 + \mu A_{k+1}} g_f(x_{k+1})$

역시 열심히 Lyapunov Analysis를 하여 다음 결론을 얻습니다.

$f(x_k) - f_{*} \le \prod_{i=1}^{k-1} \left( 1 - \sqrt{\frac{\lambda_i \mu}{1 + \lambda_i \mu}} \right) \cdot \frac{||x_0 - x_{*}||^2}{2 \lambda_0} $

# Catalyst Acceleration - The Idea

이제부터 Catalyst Acceleration에 (Lin et al. 2015) 대해 알아보겠습니다.

위 논의에서 빠진 점이 무엇인지 생각해보면, 다음과 같은 문제점을 찾을 수 있습니다.

식만 보면 **여전히** 큰 $\lambda_i$를 잡으면 무조건 효율적인 알고리즘을 얻을 수 있는 것처럼 보입니다. 그런데 **여전히** 큰 $\lambda_i$에 대해서 Proximal Operation을 계산하는 것은, 계산의 결과가 Inexact 해도 된다는 것을 감안하더라도 어려운 일입니다. 애초에 $\lambda \approx \infty$에서는 Inexact Proximal Operation을 구하는 것을 $f$의 최솟값을 근사하는 것, 즉 애초에 우리의 목표와 같은 것으로 생각하는 것도 가능하니까요. 뭔가 이상합니다.

이는 사실 우리가 Inexact Proximal Operation을 계산하는 것이 얼마나 어려운 문제인지에 대한 논의를 하지 않았기 때문에 발생하는 문제라고 생각할 수 있습니다. Catalyst Acceleration은 이 논의까지 포함하여, 전체적으로 보았을 때 가장 효율적인 알고리즘을 찾고자하는 Framework입니다. Catalyst Acceleration은 Strong Convexity가 없어도 적용할 수 있지만, 여기서는 Strong Convexity가 있다고 가정하겠습니다.

미분 가능한 convex, $L$-smooth function $f$와 CCP function $\psi$가 있다고 합시다. 풀 문제는

$F_{*} = \min_x \left\{ F(x) \equiv f(x) + \psi(x) \right\}$

와 그 값을 얻어내는 $x_{*}$를 구하는 것입니다. $F$가 $\mu$-strongly convex라고 합시다.

여기서는 Proximal Operator의 $\lambda$의 값이 상수인 경우만 생각합니다. 

$\kappa, \alpha_0$를 나중에 정할 음이 아닌 실수라고 하고, $y_0 = x_0$로 초기화를 합시다. $q = \mu / (\mu + \kappa)$라고 합시다. 

Catalyst Acceleration에는 핵심 절차가 두 개 있습니다. 

**Inexact Proximal Operation** : 말 그대로입니다. 

$x_k \approx \text{argmin}_x \left\{ G_k(x) \equiv F(x) + \frac{\kappa}{2} ||x-y_{k-1}||^2 \right\}, \quad G_k(x_k) - G_k^{*} \le \epsilon_k $

를 계산합니다. Inexact Proximal Operation의 정의가 약간 달라졌는데, 여전히 "Inexact한 Proximal Operation"이라는 의미는 가지고 있습니다. $\epsilon_k$의 값은 후술합니다.

이 값을 계산하기 위해서는 이미 알고 있는 optimization method $\mathcal{M}$을 사용합니다. 

**Extrapolation Update** : Nesterov's Estimate Sequence에서 등장하는 형태의 Extrapolation을 여기에서 적용합니다. 이에 대해서는 https://rkm0959.tistory.com/217를 참고하시기 바랍니다. 식을 써보자면, 

$\alpha_k^2 = (1-\alpha_k) \alpha^2_{k-1} + q \alpha_k, \quad \alpha_k \in (0, 1) $

$y_k = x_k + \frac{\alpha_{k-1} (1 - \alpha_{k-1})}{\alpha^2_{k-1} + \alpha_k} (x_k - x_{k-1})$

을 적용합니다. 이 두 방법을 잘 조합해서 최적의 알고리즘을 만들어봅시다. 

최적의 알고리즘을 만들기 위해서는, 다음과 같은 "기본 재료"들이 필요합니다.

- Catalyst Acceleration 자체의 수렴 속도를 알아야 합니다. (Outer Iteration)
- $x_k$를 계산하는 시간을 알아야 합니다. 즉, $\mathcal{M}$의 수렴 속도를 알아야 합니다. (Inner Iteration)
- 위 분석을 종합하여, 수렴 속도를 $\kappa$에 대한 식으로 나타낸 후, $\kappa$에 대한 최적화를 합니다.

# Catalyst Acceleration - The Analysis

**Outer Iteration** : $\alpha_0 = \sqrt{q}$라고 하고, $\rho < \sqrt{q}$인 $\rho$를 하나 잡아줍시다.

$\epsilon_k = \frac{2}{9} (F(x_0) - F_{*}) (1-\rho)^k$

라 둔다면, Catalyst Acceleration의 $\{x_k\}$들은 

$F(x_k) - F_{*} \le C(1-\rho)^{k+1} (F(x_0) - F_{*}), \quad C = \frac{8}{(\sqrt{q} - \rho)^2}$

를 만족합니다. 저자들은 다음과 같은 comment를 달았습니다.

- $\rho$는 우리가 고를 수 있는 값으로, $\rho = 0.9 \sqrt{q}$로 두면 실전에서 안전하게 사용할 수 있습니다.
- $\epsilon_k$는 우리가 아는 값이 아닙니다. 이는 $\epsilon_k$의 실전 투입을 어렵게 만드는데, 위에서 주어진 $\epsilon_k$보다 큰 값을 사용해도 괜찮습니다. 대신, 결과에 대한 bound도 그에 맞게 (상수배) 커집니다.
- 증명은 Nesterov's Estimate Sequence와 비슷한데, error term $\epsilon_k$에 대한 추가적 처리가 필요합니다. 


**Inner Iteration** : Optimization Method $\mathcal{M}$이 initial point $z_0$가 주어졌을 때, 

$G_k(z_t) - G_k^{*} \le A(1 - \tau_{\mathcal{M}})^t (G_k(z_0) - G_k^{*})$

를 만족하는 $\{z_t\}$를 제공한다고 가정합시다. $z_0 = x_{k-1}$로 **warm start**를 한다면, 목표 error인 $\epsilon_k$ 이하에 도달하기 위하여 $T_{\mathcal{M}} = \tilde{O}(1/\tau_{\mathcal{M}})$번의 iteration이 필요합니다. 여기서 짚어갈 점은 

- 대부분의 optimization method $\mathcal{M}$이 이러한 형태의 부등식을 만족합니다.
- 물론 아닌 경우도 있고, 이 경우에는 따로 분석을 해주어야 합니다. 
- **warm start**의 아이디어가 제 생각에는 핵심적인데, $\epsilon_k$는 계속 감소하지만 그 error에 도달하기 위한 iteration의 횟수는 그대로 유지시켜주기 때문입니다. 실제로 후속 논문에서는 (Lin et al. 2018) **warm start**와 **stopping criterion**에 대한 연구가 많이 이루어졌습니다. 참고하시면 좋을 것 같습니다.

**분석** : 총 $s$번의 계산을 하면 대략 $s/T_{\mathcal{M}}$번의 Outer Iteration을 돌릴 수 있으니, 이때 얻는 해 $x_{s / T_{\mathcal{M}}}$은 대략

$F(x_{s / T_{\mathcal{M}}}) - F_{*} \le C(1-\rho)^{s/T_{\mathcal{M}}} (F(x_0) - F_{*}) \le C \left(1 - \frac{\rho}{T_{\mathcal{M}}} \right)^s (F(x_0) - F_{*})$

을 만족하고, 그러니 실제 수렴 속도를 결정하는 값은

$ \rho / T_{\mathcal{M}} = \tilde{O}(\tau_{\mathcal{M}} \sqrt{\mu} / \sqrt{\mu + \kappa})$

임을 알 수 있습니다. $\tau_{\mathcal{M}}$ 역시 $\kappa$에 따라 달라지므로, 목표는 

$\tau_{\mathcal{M}} / \sqrt{\mu + \kappa}$

를 최대화하는 $\kappa$를 잡는 것이 되겠습니다. $G_k$가 $\mu + \kappa$-strongly convex, $L + \kappa$-smooth 함에 집중합시다. 

이제 $\mathcal{M}$에 맞게 $\kappa$를 각각 설정해주면 됩니다. 이는 단순한 계산 과정이고, 이것으로 Catalyst의 설명이 끝납니다.

마지막으로 몇 가지 comment를 추가하겠습니다.

- 이미 언급했지만 Catalyst Acceleration은 Strong Convexity 없이도 적용이 됩니다.
- Nesterov's Estimate Sequence와 같은 Extrapolation을 사용하지 않고, 앞에서 다룬 A-HPE 같은 알고리즘으로 대체해도 비슷한 분석을 할 수 있을 것입니다. 사실 별 차이가 없을 가능성이 더 큽니다.
- Stochastic 알고리즘에서도 비슷한 분석이 가능합니다. Markov's Inequality만을 활용하는 것 같습니다.
- 이러한 분석들은 Survey Paper의 5장에 간결하게 잘 정리되어 있으니 참고하세요.

# Applications of Catalyst Acceleration

여러 알고리즘에 대해서 $\mathcal{M}$의 성능과 그에 대응되는 $\kappa$, 그리고 최종 결과를 분석해봅시다. 내용은 Catalyst 논문의 (Lin et al. 2015) Table 1을 가져왔습니다. 이제부터 함수 $f$가 머신러닝에서 많이 보이는 형태인 

$ f(x) \equiv \frac{1}{n} \sum_{i=1}^n f_i(x)$

형태라고 가정하겠습니다. 이 context에서는 $n$이 **데이터의 개수**가 됩니다. 

이제부터 Gradient의 계산은 $\nabla f(x)$ 전체가 아닌 $\nabla f_i(x)$ 하나의 계산을 말합니다. 

**Stochastic Average Gradient** (Schmidt et al. 2015) : 이 알고리즘은 

$ \tau_{\mathcal{M}} = \min \left( \frac{\mu}{16L}, \frac{1}{8n} \right)$

을 만족함이 알려져 있습니다. 

특히, $n > 2L/\mu$인 "well-conditioned" 경우에서는 $n$번의 Gradient 계산으로 목표값이 **상수배만큼 감소**하는 강력한 성능을 갖고 있습니다. 그렇다면 $n < 2L/ \mu$인 "ill-conditioned" 경우는 어떨까요?

Catalyst Acceleration을 적용하기 위해 $\kappa$를 잡아봅시다. 결국 우리는 

$\min \left( \frac{\sqrt{\mu + \kappa}}{16(L + \kappa)}, \frac{1}{8n\sqrt{\mu + \kappa}}\right) $

를 최대화하면 되고, 이는 

$ \kappa = \frac{2(L- \mu) }{n-2}- \mu > 0$

에서 이루어짐은 쉽게 보일 수 있습니다. 대입하면 

$ \tau_{\mathcal{M}} \sqrt{\mu} / \sqrt{\mu + \kappa} = O(\sqrt{\mu / nL})$

을 얻습니다. 즉, 우리는 $\epsilon$의 오차로 답을 구하기 위해서 기존의 "ill-conditioned" 경우에서는 

$O(L/\mu \cdot \log(1/\epsilon))$

만큼의 Gradient 연산이 필요했지만, Catalyst의 적용 이후에는 

$\tilde{O}(\sqrt{nL/\mu} \cdot  \log(1/\epsilon) )$

만큼의 Gradient 연산이면 충분합니다. 

이는 $n$이 작은 경우에 Catalyst가 유의미한 성능 향상을 가져왔다는 것을 보여줍니다. 

**SAGA** (Defazio, Bach) : 이 알고리즘은 

$ \tau_{\mathcal{M}} = \frac{\mu}{2(\mu n + L)} $

을 만족함이 알려져 있습니다. 마찬가지로 Catalyst Acceleration을 적용하기 위해 $\kappa$를 잡아줍시다. 우리는 

$ \frac{\sqrt{\mu + \kappa}}{2((\mu + \kappa)n + L + \kappa)}$

를 최대화하면 됩니다. 이는 

$ \kappa = \frac{L - \mu}{n + 1} - \mu $

에서 얻어집니다. 이 경우에서도 $n < L / \mu$일 때 SAG와 같은 결과를 보여줍니다.

# 결론

Proximal Operator는 계산하기 어렵지만, Inexact 하게 계산하더라도 그 위력을 발휘할 수 있었습니다. 하지만 Proximal Operator는 Inexact 하게 계산하는 것조차 어렵습니다. 그러니, Proximal Operator의 강력함과 그 계산 난이도를 잘 섞어서 최적의 알고리즘을 얻어보자는 생각을 할 수 있습니다.  

Catalyst Acceleration은 이러한 생각을 경우에 맞게 쉽게 구체화할 수 있도록 일반화를 해주는 강력한 Framework였습니다. 저희는 이 글에서 Framework를 구성하는 주요 수식과 아이디어를 알아보았고, 실제로 어떤 예시에서 사용될 수 있는지, 그리고 그 결과 여러 알고리즘이 얼마나 향상되었는지 알아보았습니다.

Catalyst의 원 논문은 NIPS에 수록된 논문으로, 인용 횟수가 굉장히 많은 논문입니다. 이렇게 강력하고 유용한 "큰 그림"은 어느 분야에서나 인정받을 것 같습니다 :) 이런 연구를 할 수 있는 사람이 되면 기쁘겠네요.

다음 글에서는 실제로 Catalyst를 구현하는 방법에 대해서 알아보겠습니다. 구체적인 **warm start**과 **stopping criterion**을 알아보고, 코드로 옮긴 후 Catalyst 적용 전후의 성능을 비교해봅시다. 또한, 앞서 언급한 SAGA, SAG 알고리즘의 원리와 구현에 대해서도 간략하게 알아보는 시간을 갖겠습니다.

질문은 항상 rkm0959.tistory.com에서 받습니다. 감사합니다.
