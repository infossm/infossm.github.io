---
layout: post

title: "Proximity Testing with Logarithmic Randomness"

date: 2024-01-28

author: rkm0959

tags: [cryptography, blockchain]
---

## Code-Based PCS의 복습

Linear code $V$에 대하여, 벡터 $x$와 $V$의 거리를 

$$d(x, V) = \lVert x - V \rVert_0 = \min_{y \in V} (\lVert x - y \rVert_0)$$

또, 행렬 $X$와 $V$의 거리를 각 column이 전부 $V$의 원소인 행렬 $Y$에 대하여, $X - Y$가 갖는 non-zero row의 최소 개수라고 정의한다. 이 거리의 정의를 기반으로, $q$-close와 $q$-far를 정의할 수 있다. 이때, code-based PCS의 안정성을 증명하는 가장 핵심적인 결과는

$$\left\lVert \sum_{i=1}^n r_i x_i - V \right\rVert_0 \le q \implies X \text{  is  } q\text{-close to } V$$

이다. 이때, $r_i$는 $\mathbb{F}_q$의 uniform random 원소들이다. 이 사실은 $V$의 minimum distance가 $d$일 때, $q < d/3$에 대하여 성립함이 증명되어있다. 즉, $x_i$들이 전부 $V$와 가깝고 그 가까운 위치가 비슷하다는 사실을 (이를 correlated agreement라고 부른다) 확인하고 싶다면, $n$개의 random element를 가져와서 선형결합을 시키고, 그 결과가 $V$와 가깝다는 것을 확인하면 된다. 

이 사실에 대한 증명은 https://hackmd.io/k8_1AfQNTfy25N23QTmZ6g 를 참고하자. 

## Proximity Testing with Logarithmic Randomness

위 과정의 문제점 중 하나는 randomness가 $n$개 필요하다는 것이다. 이 랜덤성은 실제로는 Fiat-Shamir를 통해서 생성되어야 하므로, 결국 verifier가 이 값들이 제대로 생성되었음을 언젠가는 검증해야 한다. 즉, verifier가 직접 계산해서 $\mathcal{O}(n)$번의 연산을 해야하거나, 결과값을 prover가 일단 넘긴 후, 그 결과가 실제로 Fiat-Shamir를 정확하게 따랐다는 사실을 증명하는 ZKP를 추가로 제공하는 식의 접근이 필요하다. 이러한 수고를 피하기 위해서는, randomness가 더 적게 사용되는 방법이 필요하다는 것을 알 수 있다.

또 다른 재밌는 점은, multilinear PCS를 위해서 code-based PCS를 사용하는 경우를 생각했을 때, 결국 evaluation phase에서 $r_i$에 대응되는 벡터가 사실은 

$$(1 - r_0, r_0) \otimes (1 - r_1, r_1) \otimes \cdots \otimes (1 - r_{\log n - 1}, r_{\log n - 1}) $$

형태임을 알 수 있다. 여기에 한 번 더 나아가서, multilinear PCS가 사용되는 context를 생각해보자. 보통 multilinear PCS의 evaluation이 사용되는 곳은 sumcheck의 최종 단계인데, 핵심적인 포인트는 evaluation point가 random 하다는 것이다. 즉, multilinear PCS가 지원해야 하는 것은 random point에 대한 evaluation이지, 임의의 점에 대한 evaluation이 아니다. 이제 다시 돌아와서 생각을 해보면, 결국 correlated agreement를 한 번에 확인하기 위한 선형결합의 계수를 

$$(1 - r_0, r_0) \otimes (1 - r_1, r_1) \otimes \cdots \otimes (1 - r_{\log n - 1}, r_{\log n - 1}) $$

형태로 잡을 수 있다면, 두 마리 토끼를 한 번에 잡을 수 있다. 

- 일단 randomness가 $\log n$개로 줄어들어, verifier가 편하다
- evaluation 과정과 proximity testing 과정이 겹쳐, 성능이 좋아진다 

## Testing via Tensor Product

> For any $[n, k, d]$ code $V \in \mathbb{F}\_q^n$ and $e < d/3$, given $u\_0, \cdots, u\_{m-1} \in \mathbb{F}\_q^n$ such that 
> 
> $$\text{Pr}_{(r_0, \cdots, r_{\log m - 1}) \in \mathbb{F}_q} \left( d( [\otimes_{i=0}^{\log m - 1} (1 - r_i, r_i)] \cdot [u_0, \cdots, u_{m-1}]^T, V) \le e \right) > 2 \cdot \log m \cdot \frac{e}{q}$$
> 
> then $U = [u_0, \cdots, u_{m-1}]$ is $e$-close to $V$. 

**증명**: 일단 $l = \log m$에 대한 귀납법. $l=1$인 경우는 이미 잘 알려진 결과로, code-based PCS를 가능하게 하는 중요한 명제다. 증명을 읽고 싶다면 https://hackmd.io/k8_1AfQNTfy25N23QTmZ6g 를 참고하면 좋다. 특히, $l=1$인 경우에는 우변이 $(e+1)/q$여도 성립함이 잘 알려져 있다. 

이제 $l-1$에서 $l$로 가는 법을 생각하자. $(r_0, \cdots, r_{l-2})$에 대하여, 

$$M_0 = [\otimes_{i=0}^{l-2} (1 - r_i, r_i)] \cdot [u_0, \cdots , u_{2^{l-1}-1}]^T$$

$$M_1 = [\otimes_{i=0}^{l-2} (1 - r_i, r_i)] \cdot [u_{2^{l-1}}, \cdots , u_{2^{l}-1}]^T$$

$$R_0 = \{(r_0, \cdots, r_{l-2}) : d(M_0, V) \le e\}, \quad R_1 = \{(r_0, \cdots, r_{l-2}) : d(M_1, V) \le e\}$$

$$p(r_0, \cdots, r_{l-2}) = \text{Pr}_{r_{l-1} \in \mathbb{F}_q} \left[ d((1-r_{l-1}) M_0 + r_{l-1} M_1, V ) \le e \right]$$

$$R^\star = \left\{(r_0, \cdots , r_{l-2}) : p(r_0, \cdots, r_{l-2}) > \frac{e+1}{q} \right\}$$

를 생각하자. 먼저 $l=1$의 결과를 쓰면 $R^\star \in R_0 \cap R_1$를 얻는다. 

이제 $\mu$를 density라고 하고, $\mu(R^\star) > 2e(l-1)/q$를 보이자. 가정에 따라 

$$\text{Average}_{(r_0, \cdots , r_{l-2})} (p(r_0, \cdots , r_{l-2})) > 2el / q$$

이다. 여기서 좌변은 최대

$$(1 - \mu(R^\star)) \cdot (e+1) / q + \mu(R^\star)$$

이므로

$$\mu(R^\star) + (e+1) / q \ge  \text{Average}_{(r_0, \cdots , r_{l-2})} (p(r_0, \cdots , r_{l-2})) > 2el / q$$

를 얻어 $\mu(R^\star) > 2e(l-1)/q$가 증명된다. 

이제 귀납 가정에 의해서 $[u_0, \cdots , u_{2^{l-1}-1}]$ and $[u_{2^{l-1}}, \cdots ,u_{2^l-1}]$에 대해서 명제가 성립함을 사용할 수 있다. 

이에 따라 

$$e_0 = d([u_0, \cdots, u_{2^{l-1} - 1}], V), \quad e_1 = d([u_{2^{l-1}}, \cdots , u_{2^l - 1}] , V)$$

를 정의할 수 있고, 가장 가까운 codeword $v_0, v_1, \cdots, v_{2^{l-1}-1}, v_{2^{l-1}}, \cdots, v_{2^l-1}$를 정의할 수 있다. 여기서 나오는 correlated disagreement set을 $D_0, D_1$이라고 하고, 

$$N_0 = [\otimes_{i=0}^{l-2} (1 - r_i, r_i)] \cdot [v_0, \cdots , v_{2^{l-1}-1}]^T$$

$$N_1 = [\otimes_{i=0}^{l-2} (1 - r_i, r_i)] \cdot [v_{2^{l-1}}, \cdots , v_{2^{l}-1}]^T$$

$$B_0 = \{(r_0, \cdots , r_{l-2}) : d(M_0, N_0) < e_0 \}, \quad B_1 = \{(r_0, \cdots, r_{l-2}) : d(M_1, N_1) < e_1\}$$

를 정의할 수 있다. 

이제 $\mu(B_0), \mu(B_1) \le e(l-1)/q$를 보이자. $B_0$에 대해서만 보이면 된다. 각 $j \in D_0$에 대해, 

$$C_{j} = \{(r_0, \cdots , r_{l-2}) : M_{0, j} = N_{0, j} \}$$

를 생각하면, 결국 nonzero $l-1$-variate nonzero multilinear polynomial의 해들이라고 볼 수 있다. 그러니, Schwartz-Zippel을 쓰면 $\mu(C_{b, j}) \le (l - 1) / q$를 얻고 union bound에 의해 

$$\mu(B_0) = \mu \left( \cup_{j \in D_0} C_{j} \right) \le e_0 (l-1)/q \le e(l-1)/q$$

를 얻는다. 그러므로, 적당한 원소 $(r_0^\star, \cdots, r_{l-2}^\star)$가 있어 

$$(r_0^\star, \cdots, r_{l-2}^\star) \in R^\star \setminus (B_0 \cup B_1)$$

이 성립한다. 이 값에 대응되는 $M_0^\star, M_1^\star, N_0^\star , N_1^\star$를 생각해보자. $R^\star$ 안에 있으니, 결국 $M_0^\star, M_1^\star$가 correlated disagreement를 가짐을 알 수 있다. 그 disagreement를 $D^\star$라고 정의하자. 물론, 그 크기는 $e$ 이하다. 이제 $D_0 \cup D_1 \subset D^\star$를 보여서 증명을 마무리한다. 역시 $D_0 \subset D^\star$만 보이면 충분하다. 

$(r_0^\star, \cdots, r_{l-2}^\star) \notin B_0$이므로, $M_0^\star$ and $N_0^\star$가 다른 점은 정확히 $D_0$다. 그런데 $M_0^\star$는 $D^\star$의 부분집합에서만 다른 codeword $O_0$가 존재함을 알고 있다. 지금 unique decoding radius에 있으므로, 결국 $N_0^\star = O_0$를 얻고 $D_0 \subset D^\star$를 얻어 증명이 끝난다.

## Optimized Polynomial Commitment 

이를 기반으로 polynomial commitment를 얻는다. $2l = 2 \log m$-variate polynomial $t$에 대해서 $t$의 coefficient를 $m \times m$ matrix로 만든 다음 각 row를 code로 encode 한 후, 이를 merkle commit한다. 

각 row와 그의 encoding을 $t_i, u_i$라고 하자. 

이제 $t(r_0, \cdots, r_{2l-1}) = s$를 증명하기 위해서
- Prover는 $t' = [\otimes_{i=l}^{2l-1} (1 - r_i, r_i)] \cdot [t_0, \cdots, t_{m-1}]^T$를 보내고 
- Verifier는 $\{0, \cdots, n-1\}$에서 $\kappa$개의 index를 뽑는다. 이를 $J$라 하자.
- Prover는 $\kappa$개의 column과 그의 merkle proof를 보낸다.
- Verifier는 이를 검증하고, 

$$[\otimes_{i=l}^{2l-1} (1 - r_i, r_i)] \cdot [u_{0, j}, \cdots, u_{m-1, j}]^T = \text{Enc}(t')_j$$

를 확인한다.
- 그 후, Verifier는 $t' \cdot [\otimes_{i=0}^{l-1} (1 - r_i, r_i)]^T = s$를 확인한다.

## Extractability Proof 

Extractability가 원하는 것은 대강 설명하면 다음과 같다.
- adversary가 verify가 성공하는 proof를 내는데 성공했다면 
- negligible probability를 제외하고 PPT로 다음을 할 수 있다.
    - polynomial commitment를 open 할 수 있고
    - 그 open 된 polynomial이 adversary가 제공한 명제를 만족한다. 
    - 그 과정에서, adversary를 rewind 할 수 있다. 

기존 논문의 아이디어를 먼저 소개한다. 매우 깔끔하고 공부가 되는 증명인데, 뒤에서 이야기 하겠지만 이 논문의 extractability 증명에는 허점이 하나 있었다. 이 허점은 code-based PCS에서 extractability를 논하는 모든 논문에서 (Brakedown 등 포함) 발생한 오류였는데, 다행히도 수정하는 것이 그렇게 어렵지는 않다. 이 과정에서 필자는 이번 글에서 소개한 논문의 저자 중 한 명인 Benjamin Diamond와 소통해서, 논문을 수정했다. Acknowledgement에도 들어가있다! 

기본적인 extractor의 구조는 다음과 같다. 
- 먼저 adversary $\mathcal{A}$가 증명 하나를 제출하게 하고, 이게 통과하는지 확인한다. 실패한다면, 종료.
- $\mathcal{A}$의 random oracle query를 가지고, merkle tree를 전부 역으로 extract 한다. 쉽게 생각하면, merkle tree에서 hash 된 input들 만을 보고 merkle tree를 역으로 그리라는 건데, 직관적으로 어렵지 않고 실제로도 어렵지 않음이 알려져 있다 (BSCS16)
- $\mathcal{A}$가 $m$개의 성공적인 증명을 낼 때까지 rewinding을 반복한다. 
- 그러면 input들 $(r_{i, 0}, \cdots, r_{i, 2l-1}), t_i'$이 $0 \le i \le m-1$에 대해서 모인다.
- 이제 $\left( [ \otimes_{j=l}^{2l-1} (1 - r_{i, j}, r_{i, j}) ] \right)_{i=0}^{m-1}$이 invertible 하지 않으면 terminate.
- invertible 하다면, 이를 invert 한 후 $[t_0', \cdots, t_{m-1}']$에 곱해서 $t_i$들을 얻는다. 

$\mathcal{A}$의 성공확률이 $\epsilon$이라면, expected time은 $1 + \epsilon \cdot (m - 1) / \epsilon = m$이 되어 일단 extractor 자체는 PPT이다. 이제 extractor가 다른 곳에서 실패할 확률이 negligible 함을 보인다. 

먼저, $d([u_0, \cdots, u_{m-1}], V) < d/3$을 보인다. $e = \lfloor (d - 1) / 3 \rfloor$이라 하고, 

$$u' = [\otimes_{i=l}^{2l-1} (1 - r_i, r_i)] \cdot [u_0, \cdots, u_{m-1}]^T$$

로 두자. 귀류법으로, 

$$d((u_i)_{i=0}^{m-1}, V) \ge d/3$$

이라고 하면, $d(u', V) > e$가 $2el/q$ 정도 확률을 제외하고는 성립한다. 그러므로, $\mathcal{A}$의 성공 확률은 최대 

$$ \frac{2el}{q} + \left( 1 - \frac{e}{n} \right)^\kappa \le \frac{2dl}{3q} + \left( 1 - \frac{d-3}{3n} \right)^\kappa$$ 

가 되어 이는 negligible 하다. 그러니, 이 경우는 아예 제외하고 생각해도 된다. 

이제 closest codeword $t_0, t_1, \cdots, t_{m-1}$을 생각할 수 있다. 

이제 $t_0' = [\otimes_{i=l}^{2l-1} (1 - r_{0, i}, r_{0, i})] \cdot [t_0, \cdots, t_{m-1}]^T$가 성립함을 보인다. 

마찬가지로 $e = \lfloor (d - 1) / 3 \rfloor$, $u' = [\otimes_{i=l}^{2l-1} (1 - r_i, r_i)] \cdot [u_0, \cdots, u_{m-1}]^T$라고 하고, $v' = [\otimes_{i=l}^{2l-1} (1 - r_i, r_i)] \cdot [\text{Enc}(t_0), \cdots, \text{Enc}(t_{m-1})]^T$라고 하자. 그러면 일단 $d(u', v') \le e$. 그런데 만약 $\text{Enc}(t') \neq v'$이라고 한다면, 삼각부등식에서 

$$d(u', \text{Enc}(t')) \ge d(v', \text{Enc}(t')) - d(u', v') \ge d - e > 2d/3$$

가 성립해, verify가 성립할 확률이 최대 

$$\left( 1 - \frac{2d}{3n} \right)^\kappa$$

가 되고, 이는 negligible 하다. 

이제 다음 두 사실을 보인다 
- $t_i' = [\otimes_{j=l}^{2l-1} (1 - r_{i, j}, r_{i, j})] \cdot [t_0, \cdots, t_{m-1}]^T$가 모든 $i$에 대해서 성립
- $\left( [ \otimes_{j=l}^{2l-1} (1 - r_{i, j}, r_{i, j}) ] \right)_{i=0}^{m-1}$은 높은 확률로 invertible

두 사실을 증명하는 방법은, 조건부확률을 잘 활용한 기교다. 지금의 증명이 첫 번째 $\mathcal{A}$가 성공했을 때를 다루는 경우와 다른 이유는, 첫 성공은 말 그대로 $\mathcal{A}$를 한 번 돌려서 성공하는 것을 다루고, 그 이후부터는 $\mathcal{A}$가 성공할 때까지 돌리기 때문이다. 즉, $\mathcal{A}$가 성공한 다는 것이 조건부확률로 깔려있다고 생각해야한다. 

다음과 같이 두 사건을 정의해보자.
- $V$: $\mathcal{A}$가 올바른 증명을 제출함
- $E$: $\mathcal{A}$가 $t' = [\otimes_{j=l}^{2l-1} (1 - r_{j}, r_{j})] \cdot [t_0, \cdots, t_{m-1}]^T$를 제대로 사용함

이때, 우리가 앞서 증명한 것이 무엇인지 생각해보면 

$$\text{Pr}[V|\neg E] = \delta $$

가 negligible 하다는 사실이다. 그러니, $\epsilon \le \sqrt{\delta}$면 $\mathcal{A}$의 성공 확률은 negligible이므로, 이 경우도 무시가 가능하다. 그러므로, $\epsilon > \sqrt{\delta}$라고 가정해도 좋다. 즉, $\mathcal{A}$의 성공확률이 $\delta$보다는 overwhelming 하게 크다고 가정하는 것이다. 그렇게 되면, 베이즈 정리를 사용해서 $V$가 성공했다면 $E$도 이루어짐을 확인할 수 있다. 즉, 

$$\sqrt{\delta} < \epsilon = \text{Pr}[V] = \text{Pr}[V \cap E] + \text{Pr}[V | \neg E] \cdot \text{Pr}[\neg E] \le \text{Pr}[V \cap E] + \delta$$

$$\text{Pr}[E | V] = \frac{\text{Pr}[V \cap E]}{\text{Pr}[V \cap E] + \text{Pr}[V | \neg E] \cdot \text{Pr}[\neg E]} > \frac{ \sqrt{\delta} - \delta }{\sqrt{\delta} - \delta + \delta } = 1 - \sqrt{\delta}$$

를 얻고, 모든 경우에서 $V$가 되었을 때 $E$가 따라올 확률 역시 $(1-\sqrt{\delta})^{m-1}$이 되어 $1$과의 차이가 negligible 하다. 

비슷하게, $\left( [ \otimes_{j=l}^{2l-1} (1 - r\_{i, j}, r\_{i, j}) ] \right)\_{i=0}^{m-1}$이 invertible 함도 증명할 수 있다. 궁극적으로, 새로운 $(r\_l, \cdots, r\_{2l-1})$이 linear dependence를 만든다는 것은 최소한 이 값으로 만들어진 $[\otimes_{i=l}^{2l-1} (1 - r\_{i}, r\_{i})]$이 어떤 고정된 hyperplane 안에 속한다는 것이다. 그런데, 그 조건과 확률을 생각해보면 결국 $l$-variate multilinear non-zero polynomial의 근을 생각하는 것과 같으니 Schwartz-Zippel에 의해 그 확률은 최대 $l/q$다. 

즉, 이번에는 $E$를 linear dependence가 생기지 않을 사건이라고 생각하면 이번에는 

$$\text{Pr}[\neg E] = l/q$$

가 negligible 하다는 것을 알고 있다. 이제 비슷한 베이즈 트릭을 쓰면 증명 끝. 

이 $\sqrt{\delta}$ 트릭이 상당히 강력함을 알 수 있다. negligible 한 케이스들을 쉽게 제거할 수 있다. 

## Issue with Extractability Proof 

그런데 이 증명에는 minor한 문제가 하나 있었다. 그 점은 바로 merkle tree를 extract 하는 과정에 있다. 사실 지금까지 해당 PCS와 비슷한 계열의 PCS의 open 정의는, merkle root에 대응되는 모든 $u_i$들을 전부 공개하는 것이었다. 그런데, 사실 adversary가 commitment를 만들 때 모든 column을 merkle commit 하지 않아도 된다! 예를 들어, $n$개의 column을 다 merkle 하는 대신, $n-1$개만 merkle 하고 나머지 하나에 괴상한 값을 넣었다고 하자 (예를 들면, random-oracle preimage도 모르는). 그러면 현재 정의상 open은 아예 불가능하다. 그런데, verifier가 column query를 제대로 된 $n-1$개의 column에만 하면, prover는 merkle proof를 전부 제공할 수 있고 아예 통과하는 증명을 낼 수 있다. column query가 잘못된 하나의 column을 피할 확률은 

$$(1 - 1/n)^\kappa$$

로, negligible 하지도 않다. 그러므로, 이 경우에서는 adversary가 non-negligible probability로 성공하면서 정작 extract는 불가능한 것이다. 

이를 해소하기 위한 아이디어는, open의 정의를 갈아엎는 것이다. 즉, $u_i$들을 전부 공개하는 것이 아니라 일부 column들 $M$은 공개하지 않아도 되도록 조건을 relax 시키고, $M$은 전부 $0$으로 채운 뒤에 대신 $u_i$들과 $\text{Enc}(t_i)$들의 correlated disagreement와 $M$의 합집합의 크기가 $d/3$ 이하이도록 하는 것이다. 즉, "애초에 correlated disagreement인 column들은 어떤 값이 있어도 문제 없으니 merkle proof를 요구하지 않는다"는 아이디어다. 

이렇게 되면 증명을 살짝 고쳐야 한다. 우선 merkle tree extractor를 사용하면
- extract가 되는 column들
- extract가 non-negligible probability로 가능하지 않은 column들

을 바로 구분할 수 있다. 후자에 해당하는 column을 $M$이라고 하자. 

만약 $\lvert M \rvert \ge d/3$이라면, 첫 $\mathcal{A}$의 시도에서 성공할 확률이 최대 

$$\left(1 - \frac{d}{3n} \right)^\kappa$$

가 되어 이는 negligible 하다. 그러므로 $\lvert M \rvert < d/3$을 가정해도 된다. 

이제 핵심은 모든 code를 $M$에 해당하는 index를 전부 다 날린 punctured code로 옮기는 것이다. 그러면 code의 minimum distance는 $d - \lvert M \rvert$가 되고, 우리가 원하는 것은 punctured code에서 $d/3 - \lvert M \rvert$ 거리를 확인하는 것이다. 다행히도, 

$$d/3 - \lvert M \rvert \le (d - \lvert M \rvert) / 3$$

이어서, 여전히 correlated agreement에 대한 정리들을 사용할 수 있다. 즉, 이제 
- punctured code에서의 correlated disagreement에 속하는 column들
- $M$에 속하는 column들 

을 verifier가 고르는 순간 증명 검증이 실패할 것이므로, 앞선 증명과 마찬가지 방법을 통해서 증명을 마무리 할 수 있다. 이와 관련해서 Benjamin Diamond와 소통한 내용은 아래에서 볼 수 있다. 

https://hackmd.io/qb-NrfZ7SgWMvPGNF4xPxw

다음 트위터 쓰레드 역시 참고하면 좋다. 

https://twitter.com/rkm0959/status/1746723799012442565

