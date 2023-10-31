---
layout: post

title: "Multilinear PCS from Univariate PCS"

date: 2023-10-22

author: rkm0959

tags: [cryptography, blockchain]
---

저번 포스팅에서는 Multilinear Polynomial에 대한 linear-time commitment 중 하나인 Brakedown에 대해서 알아보았습니다. Sumcheck 관련 기법들이 떠오르면서, Multilinear Polynomial의 commitment에 대한 기법들이 더욱 중요해졌습니다. 그 방법에는 Brakedown 및 이를 강화하는 Orion, Orion+ 뿐만 아니라 Dory, Hyrax 등 다른 기법도 존재합니다. 특히, Univariate Polynomial에 대한 Commitment를 기반으로 Multilinear Polynomial에 대한 Commitment 기법을 유도하는 기법들도 여러 가지 존재합니다. 여기서는 그 방법인 Gemini와 Zeromorph에 대해서 알아보도록 하겠습니다. 

이 포스팅에 대응되는 논문은 아래와 같습니다. 
- https://eprint.iacr.org/2022/420.pdf Section 5
- https://eprint.iacr.org/2023/917


# KZG + Gemini

Multilinear Polynomial의 representation을 Lagrange basis로 두지 않고, $\prod_{i=1}^\mu x_i^{e_i}$으로 생각합시다. 물론 여기서 $0 \le e_i \le 1$. 이제 multilinear polynomial $f_{mult}$를 $(z_1, z_2, \cdots, z_\mu)$에서 evaluate 한다는 것은, $f_{mult}$의 (위에서 언급한 representation 기반) coefficient vector $\vec{f}$에 대하여 

$$\langle \vec{f}, \otimes_{i=1}^\mu (1, z_i) \rangle = y$$

를 증명하는 것과 같습니다. 

이제 $\vec{f}$를 coefficient vector로 갖는 univariate polynomial $f_{uni}$를 생각하고, 이에 대한 KZG Commitment를 생각합니다. 이를 $f_{mult}$의 commitment로 생각합시다. 

이제 evaluation proof를 진행하는 방법을 생각합시다. FRI의 느낌으로, $f_0 = f_{uni}$로 두고 

$$f_{i-1}(X) = g_{i-1}(X^2) + X h_{i-1}(X^2), \quad f_i(X) = g_{i-1}(X) + z_i h_{i-1}(X)$$

이도록 $f_1, f_2, \cdots, f_\mu$를 계산하고 commit하여 verifier에게 보냅니다. 

이제 verifier가 random challenge $\beta$를 보내면, prover는 

$$a_i = f_i(\beta), \quad b_i = f_i (-\beta), \quad c_i = f_{i+1}(\beta^2)$$

여기서 $f_{\mu} = y$라는 상수함수이므로, 이를 감안합니다. verifier는 

$$c_i = \frac{a_i + b_i}{2} + z_i \cdot \frac{a_i - b_i}{2\beta}$$

가 전부 성립하는지 확인합니다. Prover의 입장에서는 $f_1, \cdots, f_\mu$에 대한 commitment를 각각 계산하고, $a_i, b_i, c_i$들을 전부 계산하고, 그 값들에 대한 batch opening proof를 계산해서 verifier에게 전달해야 합니다. Verifier의 입장에서는, $\mu$개의 식들에 대한 검증과 함께 KZG batch proof를 검증하면 됩니다. 

위 방식의 soundness / completeness는 전형적인 방법으로 증명할 수 있습니다. 

$N = 2^\mu$라고 하면, 위 방식의 performance는 다음과 같이 정리할 수 있습니다. 
- Commitment Size는 $1 \cdot \lvert G_1 \rvert$
- Commitment Time은 $\mathcal{O}(N) \cdot G_1$
- Proof Size는 $\mathcal{O}(\log N) \cdot \lvert G_1 \rvert$
- Prover Time은 $\mathcal{O}(N) \cdot G_1$
- Verifier Time은 $\mathcal{O}(\log N) \cdot G_1$ (constant number of pairings)

각각에 대한 분석 역시 꽤 전형적인 편입니다. 잘 생각해보면, batch prove/verify가 가능하면 KZG가 아니더라도 이 reduction을 적용할 수 있음을 파악할 수 있습니다. 

# Zeromorph

아이디어가 상당합니다. 핵심 아이디어는 $f \in \mathbb{F}[X_0, \cdots, X_{n-1}]$가 

$$f(u_1, \cdots, u_n) = v$$

를 만족한다면, 적당한 $q_k \in \mathbb{F}[X_0, \cdots X_{k-1}]$가 있어 

$$f - v = \sum_{k=0}^{n-1} (X_k - u_k) q_k$$

가 성립한다는 것입니다. 정확하게는, 이것이 필요충분조건입니다. 

우선 위 식을 만족하는 $q_k$들이 존재한다면 자명히 $f(u_1, \cdots, u_n) = v$입니다. 

반대방향이 문제인데, $n$에 대한 귀납법으로 가능합니다. 예를 들어, 

$$f - f(X_0, \cdots, X_{n-2}, u_{n-1}) = (X_{n-1} - u_{n-1}) q_{n-1}$$

이도록 하는 $q_{n-1}$을 잡을 수 있고, 다시 

$$f(X_0, \cdots, X_{n-2}, u_{n-1})$$

에 대해서 induction hypothesis를 적용하면 증명을 끝낼 수 있습니다. 나아가서, 간단한 대입으로

$$q_k = f(X_0, \cdots, X_{k-1}, u_k + 1, u_{k+1}, \cdots, u_{n-1}) - f(X_0, \cdots, X_{k-1}, u_k, u_{k+1}, \cdots, u_{n-1})$$

임도 증명할 수 있습니다. 

이제 multivariate polynomial에서 univariate polynomial로 가는 linear transform을 하나 잡아야합니다. 앞선 Gemini에서는 "standard"한 basis를 기반으로 변환을 했지만, 여기서는 Lagrange basis를 사용합니다. 여기서는 isomorphism

$$U_n: L_{i_0, i_1, \cdots, i_{n-1}}(X) \rightarrow X^{2^0i_0 + 2^1i_1+ \cdots 2^{n-1}i_{n-1}}$$

을 사용합니다. 쉽게 생각하면 $L_i(X) \rightarrow X^i$라고 봐도 무방합니다. 이제 목표는 

$$U_n(f) - U_n(v) = \sum_{k=0}^{n-1} \left( U_n(X_kq_k) - u_k U_n(q_k) \right)$$

를 만족하는 $q_k \in \mathbb{F}[X_0, \cdots X_{k-1}]$의 존재를 보이는 것입니다. 

이를 위해서, 열심히 계산을 해서 다음 값들에 대한 고찰을 해야합니다.
- $U_n(v)$의 값은 무엇인가 
- $q_k \in \mathbb{F}[X_0, \cdots X_{k-1}]$와 동치인 $U_n(q_k)$의 성질
- $q_k \in \mathbb{F}[X_0, \cdots X_{k-1}]$와 동치인 $U_n(X_k q_k)$의 성질

이를 위해서 $\phi_n(X) = \sum_{i=0}^{2^n - 1} X^i$를 정의합니다. 

적당한 계산으로 다음을 증명할 수 있습니다. 

## Fact 1 

$$U_n(v) = v \phi_n(X)$$

## Fact 2 

$\hat{f} \in \mathbb{F}[X]^{< 2^n}$과 $f = U_n^{-1}(\hat{f})$를 생각하면, $f \in \mathbb{F}[X_0, \cdots, X_{k-1}]$인 것은 

$$\hat{f}(X) = \phi_{n-k}(X^{2^k}) \hat{f}^{< 2^k} (X)$$

임과 동치이며, 이때 $\hat{f}^{<2^k}(X) = U_k(f)$가 성립한다. 

## Fact 3

$f \in \mathbb{F}[X_0, \cdots, X_{k-1}]$이면, 

$$(X^{2^k} + 1) U_n(X_k f) = X^{2^k} U_n(f)$$

이를 종합하여, 증명하고자 하는 식을 

$$U_n(f) - v \cdot \phi_n(X) = \sum_k (X^{2^k} \cdot \phi_{n-k-1}(X^{2^{k+1}}) - u_k \cdot \phi_{n-k}(X^{2^k})) \cdot U_n(q_k)^{< 2^k}$$

로 변환할 수 있습니다. 그러므로,

- $U_n(f)$를 univariate commitment로 commit 하고 
- prove 과정에서 $U_n(q_k)^{<2^k}$를 commit 하고 degree check를 적용한 뒤 

verifier가 random 하게 $x$를 sample 하여 전달, 이를 가지고 

$$Z_x(X) = U_n(f)(X) - v \cdot \phi_n(x) - \sum_k (x^{2^k} \cdot \phi_{n-k-1}(x^{2^{k+1}}) - u_k \cdot \phi_{n-k}(x^{2^k})) \cdot U_n(q_k)^{<2^k}(X)$$

가 $x$에서 0으로 evaluate 됨을 증명하면 됩니다. 이는 batch proof로 쉽게 가능합니다. 

Zeromorph 논문은 상당한 지면을 hiding KZG commitment에 할애합니다. 그리고 이를 기반으로
- hiding + batch + degree check + evaluation proof 

를 얻는 방법에 대해서 한 section을 할애합니다. 이 방법 역시 KZG에 대한 전형적 트릭의 연장선에 가깝습니다.

나아가서, shifted polynomial에 대한 commitment도 더 쉽게 할 수 있습니다. 




