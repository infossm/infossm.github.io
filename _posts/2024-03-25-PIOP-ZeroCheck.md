---
layout: post

title: "PIOP Improvements for ZeroCheck"

date: 2024-03-25

author: rkm0959

tags: [cryptography, blockchain]
---

Circle STARK Part 2를 하기 이전에, 최근 읽은 논문 중 간단하면서도 강력한 게 하나 있어서 이를 공유합니다. 이번 글에서는 https://eprint.iacr.org/2024/108 의 내용을 정리합니다. 

# Zero Check 

Multivariate Polynomial을 기반으로 한 proof system들은, 일반적으로 boolean hypercube $H_n$ 위에서 어떤 함수 $C(x)$가 전부 0으로 계산됨을 증명하는 것으로 문제를 환원시킵니다. 결국 최종적으로 

$$C(x) = 0 \quad \forall x \in H_n$$

를 보여야 하는데, 이를 위해서 verifier는 랜덤한 $\alpha$를 정한다음 

$$\sum_{x \in H_n} \tilde{eq}(x, \alpha) \cdot C(x) = 0$$

을 증명합니다. 이때, 

$$\tilde{eq}(x, \alpha) = \prod_{i=1}^{n} (x_i \alpha_i + (1-x_i)(1-\alpha_i))$$

는 $\tilde{eq}(a, b) = (a == b)$에 대한 multilinear extension입니다. 

이 변환이 가능한 이유는, 위 식을 $\alpha$에 대한 다항식으로 보면 $x \in H_n$에 대해서 $C(x)$로 evaluate 되는 다항식이므로, 랜덤한 $\alpha$에 대해서 0으로 계산되려면 Schwartz-Zippel Lemma에 의해서 높은 확률로 $C(x) = 0$이 모든 $x \in H_n$에 대해서 성립해야 하기 때문입니다. 

이제 문제가 $H_n$ 위에서 어떤 함수의 합을 더한 결과가 0이 되는지 여부를 판별하는 것으로 환원되었으니, Sumcheck를 사용할 수 있습니다. 

이번 논문은 이 Sumcheck 과정을 최적화하는 방법에 대한 것이므로, Sumcheck를 복습합시다. 

# Classic Sumcheck

목표가 $v_0 = \sum_{x \in H_n} f(x)$를 증명하는 거라고 합시다. 또한, 각 $x_i$에 대해서 $f$의 degree가 최대 $d$라고 합시다. 

$i$th round에서 prover의 목표는 

$$v_i = \sum_{x \in H_{n-i}} f(r_0, \cdots, r_{i-1}, x_i, \cdots, x_{n-1})$$

를 보이는 것입니다. 이를 위해서, 

$$v_{i+1}(X) = \sum_{x \in H_{n-1-i}} f(r_0, \cdots, r_{i-1}, X, x_{i+1}, \cdots, x_{n-1})$$

를 각 $X = 0, 1, \cdots, d$에 대해 계산해 verifier에게 보냅니다. 

이러면 일단 

$$v_i = v_{i+1}(0) + v_{i+1}(1)$$

이 성립하는지 verifier가 확인할 수 있습니다. 그 후, verifier는 $r_i$를 정하고 각자 

$$v_{i+1} = v_{i+1}(r_i)$$

를 생각할 수 있습니다. 최종 단계에서는 verifier가 $[f]$를 query 해, 

$$v_n = f(r_0, \cdots, r_{n-1})$$

이 성립함을 확인합니다. 

특히 $p$를 작은 값으로 사용하는 경우, $\mathbb{F}_p$ 위에서 sumcheck를 돌리면 security가 강하지 않기 때문에 일반적으로 extended field를 사용합니다. 

$$\mathbb{F} = \mathbb{F}_p, \quad \mathbb{G} = \mathbb{F}_{p^k}$$

라고 합시다. 이러면 witness들은 전부 $\mathbb{F}$ 위에 있지만, sumcheck의 모든 과정 ($r_i$ 등)은 $\mathbb{G}$ 위에서 얻어집니다. 실제로 $p$가 64bit라면 $k \ge 4$이므로, $\mathbb{G}$의 연산과 $\mathbb{F}$의 연산 사이의 cost 차이가 상당합니다. 이 점은 이 논문에서 최적화를 위해서 크게 고려되는 부분입니다. 

witness들이 $2^n \times l$ 형태 table로 있고, 각 $2^n$개의 row들을 multilinear polynomial로 interpolate 했다고 생각합시다. 즉, 

$$C(T_{i,0}, \cdots, T_{i, l-1}) = 0$$

라는 constraint를 나타내기 위해 $T_{i, j}$들을 interpolate 해서

$$C(\omega_0(X), \cdots, \omega_{l-1}(X)) = 0, \quad \forall X \in H_n$$

으로 생각할 수 있습니다. 결국, 우리의 sumcheck 과정은 

$$\sum_{x \in H_n} \tilde{eq}(x, \alpha) \cdot C(\omega_0(x), \cdots, \omega_{l-1}(x)) = 0$$

을 증명하는 과정이라고 볼 수 있습니다. 

Prover Cost를 생각해보면, 결국 

$$\sum_{x \in H_{n-1-i}} \tilde{eq}([\vec{r}, X, \vec{x}], \alpha) \cdot C(\omega_0(\vec{r}, X, \vec{x}), \cdots, \omega_{l-1}(\vec{r}, X, \vec{x}))$$

를 각 $X = 0, \cdots, d$에 대해서 계산하는 게 핵심입니다. 

기본적으로 $\omega_0, \cdots, \omega_{l-1}$들은 multilinear polynomial이므로, 이 함수들을 $(\vec{r}, X, \vec{x})$에 대해서 전부 evaluate 하는 것은 어렵지 않습니다. 또한, $v_{i+1}$의 interpolation 등은 매우 작은 다항식을 interpolate 하는 것이므로, 어렵지 않은 과정입니다. 결국, 시간이 가장 많이 걸리는 어려운 과정은 $\omega_i(\vec{r}, X, \vec{x})$ 등이 전부 계산되어 있을 때, 이를 기반으로 $C$ 값을 계산하는 과정입니다. 

$C_{\mathbb{F}}, C_{\mathbb{G}}$를 $C$를 계산하기 위해 걸리는 시간이라고 둡시다. 단, $\mathbb{F}, \mathbb{G}$는 underlying field입니다. 

$\tilde{eq} \cdot C$는 degree $d+1$이므로, $d+2$개의 점에서 계산을 해야 interpolation이 가능합니다. 그러므로, evaluation의 총 수를 따져보면 대략

$$(d+2)(2^{n-1}C_{\mathbb{F}} + \sum_{i=1}^{n-1} 2^{n-1-i} C_{\mathbb{G}}) = (d+2) 2^{n-1}(C_{\mathbb{F}} + C_{\mathbb{G}})$$

가 됩니다. 이제 이 값을 최적화하는 것이 이 논문의 목표가 되겠습니다. 

# Optimization 1: Data Optimization

우선 흥미로운 점은 verifier가 $v_{i+1}(0), v_{i+1}(1)$을 전부 받은 다음 

$$v_i = v_{i+1}(0) + v_{i+1}(1)$$

임을 확인한다는 점입니다. 사실 애초에 이럴 거였으면, $v_{i+1}(0)$만 받은 다음 

$$v_{i+1}(1) = v_i - v_{i+1}(0)$$

을 사용하면 됩니다. 비슷하게, 정상적인 경우에서는 $v_1(0) = v_1(1) = 0$가 성립하기 때문에 verifier는 이 값들을 prover가 줬다고 가정하고 sumcheck를 진행하면 됩니다. 

이러면 prover cost가 $d \cdot 2^{n-1} C_{\mathbb{F}} + (d+1) 2^{n-1} C_{\mathbb{G}}$로 감소합니다. 

# Optimization 2: Degree Reduction via $\tilde{eq}$

$\tilde{eq}$가 어마어마하게 단순한 함수인 점을 사용해서 최적화를 할 수 있습니다. 

$$v'_{i+1}(X) =  \sum_{x \in H_{n-1-i}} \tilde{eq}(x, \alpha_{[i+1, n)}) \cdot C(\vec{r}, X, \vec{x}) $$

$$v_{i+1}(X) = \tilde{eq}(\vec{r}, \vec{\alpha}_{[0, i)}) \cdot (X \alpha_i + (1-X)(1-\alpha_i)) \cdot v'_{i+1}(X)$$

가 성립함을 생각합시다. 이제 $v'_{i+1}$은 degree가 하나 작습니다. 또한, 

$$v_i = v_{i+1}(0) + v_{i+1}(1)$$

를 생각해보면 

$$v_i' = (1 - \alpha_i) v'_{i+1}(0) + \alpha_i v'_{i+1}(1)$$

로 변환됨을 알 수 있습니다. 이제 Optimization 1까지 덮을 수 있습니다. 

이제 모든 연산을 $v_i'$에 대해서 할 수 있어, 연산량이 $(d-1)2^{n-1} C_{\mathbb{F}} + d \cdot 2^{n-1} C_{\mathbb{G}}$가 됩니다. 

# Optimization 3: Algebraic Optimization

세번째 최적화를 위해서는 생각을 다른 방식으로 해야합니다. $C$가 $H_n$에서 vanish하면, 

$$R_n = C, \quad R_{i+1} = x_i(x_i-1)Q_i + R_i, \quad \deg_{x_i}(R_i) \le 1$$

로 두면 $C$가 $H_n$에서 vanish 한다는 사실에서 

$$C = \sum_{i=0}^{n-1} x_i(x_i-1)Q_i(x_0, \cdots, x_{n-1})$$

이 성립함을 알 수 있으며, 

$$\deg_{x_i}(Q_j) \le 1, \quad \forall j < i$$

임을 알 수 있습니다. 이제 

$$\overline{Q}_i(\vec{r}, \vec{X}) = \sum_{j=0}^{i-1} r_j(r_j-1)Q_j(\vec{r}, \vec{X})$$

라고 하면 이 식은 $x_i, x_{i+1}, \cdots, x_{n-1}$ 각각에 대해 linear 하고 $\vec{x} \in H^{n-i}$에 대해서 $C(\vec{r}, \vec{x})$와 같기 때문에, 결국 $C(\vec{r}, \vec{x})$의 multilinear extension임을 알 수 있습니다. 

한편, 

$$C(\vec{r}, X, \vec{x}) - \overline{Q}_i(\vec{r}, X, \vec{x}) = X(X-1)Q_{i}(\vec{r}, X, \vec{x})$$

가 모든 $\vec{x} \in H^{n-1-i}$에 대해서 성립함을 알 수 있습니다. 

이제 이를 기반으로 최적화를 해봅시다. 

Round 0에서, $C(X, x)$를 계산하면서 

$$C(X, x) = X(X-1)Q_0(X, x), \quad \forall x \in H_{n-1}$$

인 $Q_0$를 계산할 수 있으며, 이를 $X = 2, \cdots d$에 대해서 계산합니다. 

$$\overline{Q}_1(r_0, x) = r_0(r_0 - 1) Q_0(r_0, x)$$

를 만족시킵니다. $C(r_0, x) = \overline{Q}_1(r_0, x)$가 $x \in H_{n-1}$에 대해 성립하므로, $C$를 굳이 더 계산할 필요가 없습니다. 또한, 

$$C(r_0, X, \vec{x}) = \overline{Q}_1(r_0, X, \vec{x}) + X(X-1) Q_1(r_0, X, \vec{x})$$

이고, $\overline{Q}_1$은 $x_1$에 대해서 linear하므로 $\overline{Q}_1(r_0, 0, \vec{x})$와 $\overline{Q}_1(r_0, 1, \vec{x})$를 기반으로 $\overline{Q}_1(r_0, X, \vec{x})$를 계산할 수 있으며, 그러니 $C(r_0, X, \vec{x})$를 계산할 때, $Q_1(r_0, X, \vec{x})$까지 같이 계산됩니다. 이제 이를 기반으로 

$$\overline{Q}_2(r_0, r_1, \vec{x}) = r_0(r_0 - 1) Q_0(r_0, r_1, \vec{x}) + r_1(r_1 - 1) Q_1(r_0, r_1, \vec{x})$$

를 얻습니다. 이를 반복하면, 결국 degree $d-2$를 기반으로 작업을 할 수 있어 연산량이 $(d-1)2^{n-1}(C_{\mathbb{F}} + C_{\mathbb{G}})$로 줄어듭니다. 

# Optimization 4: Using $\mathbb{F}$-computation

생각해보면 모든 것을 $C_{\mathbb{F}}$ 연산으로 할 수 없는 이유는 한 번 random input을 받는 순간 모든 연산이 $\mathbb{G}$ 위에서 이루어지기 때문입니다. random input을 받기 전의 연산은 결국 첫 번째 round이므로, 이후에 $\mathbb{G}$ 위의 연산을 적게 하기 위해서 첫 round에서 $\mathbb{F}$ 위의 연산을 더 하는 방식의 tradeoff가 가능하다면 꽤 유용할 거라는 생각을 할 수 있습니다.

이 tradeoff를 만드는 방법은 첫 번째 변수의 범위를 $\{0, 1\}$보다 더 크게 하는 것입니다. 즉, boolean hypercube $H_n$을 $\{0, 1, \cdots 2^k - 1\} \cdot H_{n-k} = D \times H_{n-k}$로 바꿔치는 것입니다. 

evaluation이 필요한 점의 개수를 생각해보면, degree를 생각하면 

$$2^{n-k} (d(2^k - 1) + 1 - 2^k) = 2^{n-k}(2^k-1)(d-1)$$

입니다. 그러므로, final prover cost는 

$$(2^n - 2^{n-k})(d-1) C_{\mathbb{F}} + \left(\sum_{i=1}^{n-k} 2^{n-k-i}(d-1) C_{\mathbb{G}} \right)$$

이므로, 식을 정리하면

$$(d-1)((2^n - 2^{n-k})C_{\mathbb{F}} + 2^{n-k}C_{\mathbb{G}})$$

를 얻습니다. 대신, 첫 라운드에서 verifier가 $2^k \cdot d$짜리 interpolation을 해야합니다. 

# Estimating Final Prover Cost

논문의 예시는 $d = 3$, $k = 4$, $C_{\mathbb{G}} = 16 C_{\mathbb{F}}$입니다. 이 경우, 기존 알고리즘은 $(d+2)2^{n-1}(C_{\mathbb{F}} + C_{\mathbb{G}}) = 5 \cdot 17 \cdot 2^{n-1} \cdot C_{\mathbb{F}}$만큼의 연산이 필요합니다. 하지만, 강화된 알고리즘은 

$$(3-1)((2^n - 2^{n-4})C_{\mathbb{F}} + 2^{n-4} \cdot 16 \cdot C_{\mathbb{F}}) = 31 \cdot 2^{n-3} \cdot  C_{\mathbb{F}}$$

만큼의 연산이 필요합니다. 이를 비교하면 대략 

$$\frac{31}{85 \cdot 4} \approx \frac{1}{11}$$

만큼이므로, prover의 cost가 1/11 수준으로 줄어들었음을 알 수 있습니다. Verifier cost도 증가하기는 하지만, 크기 48짜리 interpolation 한 번입니다. (추가적으로, 이 verifier의 interpolation은 한 점에서 값 계산만 할 수 있으면 되므로, linear time에 작동합니다.)

그러므로, 전체적인 ZeroCheck의 성능이 크게 개선되었음을 알 수 있습니다. 

간단한 기교, 대수학적 접근, 그리고 관찰을 기반으로 한 minor modification을 통해서 10x를 얻을 수 있다는 점이 인상적이었던 논문이었습니다. 