---
layout: post

title: "Folding Part 2: HyperNova"

date: 2023-12-25

author: rkm0959

tags: [cryptography, blockchain]
---

이 글에서는 ZKP에서 사용되는 테크닉인 folding의 두 대표적인 논문인 ProtoStar와 HyperNova 중 HyperNova에 대해서 다룬다. 

- https://eprint.iacr.org/2023/573.pdf

# Preliminaries

## Incremental Verifiable Computation 

Incrementally Verifiable Computation은 
- Generator $\mathcal{G}(1^\lambda) \rightarrow pp$: security parameter $\lambda$를 받고 public parameter $pp$를 만든다
- Encoder $\mathcal{K}(pp, F) \rightarrow (pk, vk)$: public parameter $pp$와 반복할 함수 $F$를 받고, prover key 및 verifier key $pk, vk$를 만든다
- Prover $\mathcal{P}(pk, (i, z_0, z_i), \omega_i, \Pi_i) \rightarrow \Pi_{i+1}$은 결과 $z_i$와 이에 대한 IVC proof $\Pi_i$를 받고, $z_{i+1} = F(z_i, w_i)$를 계산하여 이에 대한 새 IVC proof $\Pi_{i+1}$을 만든다. 
- Verifier $\mathcal{V}(vk, (i, z_0, z_i), \Pi_i) \rightarrow \{0, 1\}$은 IVC proof를 검증해 정당한지 확인한다. 

이 경우, 원하는 조건은 
- Completeness: 제대로 계산하여 생성된 IVC proof는 verifier를 통과한다.
- Knowledge Soundness: 임의의 PPT Adversary에 대해서 PPT extractor가 있어 adversary가 IVC proof를 통과시킨다면, extractor가 $w_i$들을 추출해내 $z_{i+1} = F(z_i, w_i)$이며 최종 결과가 verifier의 input 결과와 동일하다. 

## Multi-Folding Scheme

Relation $R_1$과 $R_2$와 이들이 만족해야 하는 구조에 대한 predicate $compat$, 그리고 size parameter $\mu, \nu$가 있을 때, $(R_1, R_2, compat, \mu, \nu)$에 대한 multi-folding scheme은 
- Generator $\mathcal{G}(1^\lambda) \rightarrow pp$: security parameter $\lambda$를 받고 public parameter $pp$를 만든다
- Encoder $\mathcal{K}(pp, (s_1, s_2)) \rightarrow (pk, vk)$: public parameter $pp$와 structure $s_1, s_2$를 받고, prover key 및 verifier key $pk, vk$를 만든다
- Prover $\mathcal{P}(pk, (\vec{u_1}, \vec{w_1}), (\vec{u_2}, \vec{w_2})) \rightarrow (u, w)$: $pk$, $R_1$의 instance 및 witness vector $(\vec{u_1}, \vec{w_1})$와 $R_2$의 instance 및 witness vector $(\vec{u_2}, \vec{w_2})$를 받고, 이를 fold 해서 $R_1$의 새 instance $(u, w)$를 만든다. 
- Verifier $\mathcal{V}(vk, (\vec{u_1}, \vec{u_2})) \rightarrow u$는 $vk$와 instance vector 2개를 받아서 새 instance를 내는 과정이다.

$\mathcal{P}, \mathcal{V}$가 multi-folding scheme을 돌리면, $((pk, vk), (\vec{u_1}, \vec{w_1}), (\vec{u_2}, \vec{w_2}))$ 위에서 $\mathcal{P}$는 Prover를 돌려서 $w$를 얻고, $\mathcal{V}$는 Verifier를 돌려서 $u$를 얻은 후 $(u, w)$가 최종 output이 된다. 

이 경우, 원하는 조건은 
- Completeness: 각 instance/witness pair가 전부 맞는 relation을 만족시키는 경우, Prover/Verifier를 돌려서 나온 $(u, w)$는 $\mathcal{R}_1$을 만족시킨다.
- Knowledge Soundness: 임의의 PPT adversary $\mathcal{A}$와 $\mathcal{P}^\star$에 대하여, $(\vec{u_1}, \vec{u_2})$에 대하여 $\mathcal{P}^\star, \mathcal{V}$가 multi-folding scheme을 돌려 $\mathcal{R}_1$을 만족시키는 $(u, w)$를 만드는데 성공할 확률은, 적당한 PPT extractor $\mathcal{\xi}$가 있어 $(\vec{w_1}, \vec{w_2})$를 추출할 수 있고, 이들이 각각 $\mathcal{R}_1, \mathcal{R}_2$를 만족하는 instance/witness vector가 될 확률과 비슷하다. 

## Customizable Constraint Systems

$\mathcal{R}_{CCS}$의 structure는 
- matrix $M_1, \cdots , M_t \in \mathbb{F}^{m \times n}$: 단, non-zero entry는 $N = \Omega(\max(m, n))$개 
- $q$개의 multiset $S_1, \cdots, S_q$. 단, 각 원소는 $\{1, \cdots, t\}$의 원소이며 각 multiset의 크기는 최대 $d$
- $q$개의 constant $c_1, \cdots, c_q$ 

로 구성되었으며, instance는 public input $x \in \mathbb{F}^l$이며 witness는 $w \in \mathbb{F}^{n - l - 1}$이다. 이들이 맞는 instance/witness pair가 되려면, $z = (w, 1, x) \in \mathbb{F}^n$에 대하여 

$$\sum_{i=1}^q c_i \cdot \otimes_{j \in S_i} (M_j \cdot z) = \vec{0}$$ 

단, $\otimes$는 vector 사이의 hadamard product. 

일반적으로 $n = 2(l+1)$로 두며 이 값이 2의 거듭제곱이 되도록 한다. 이제 Committed CCS를 설명한다. $s = \log m$, $s' = \log n$으로 두고, 각 matrix를 sparse multilinear polynomial로 전환하여, $\tilde{M_1}, \cdots \tilde{M_t}$로 만든다. 이제 instance는 $s' - 1$개의 변수로 구성된 multilinear polynomial의 commitment인 $C$와 public input $x \in \mathbb{F}^l$로 구성된다. 이제 정당한 witness $\tilde{w}$는 $s' - 1$개 변수로 구성된 multilinear polynomial로, 

$$Commit(pp, \tilde{w}) = C$$

이며, 각 $u \in \{0, 1\}^s$에 대하여 

$$\sum_{i=1}^q c_i \cdot \prod_{j \in S_i} \left( \sum_{y \in \{0, 1\}^{s'}} \tilde{M_j}(u, y) \cdot \tilde{z}(y) \right) = 0$$

단 $\tilde{z}$는 $(w, 1, x)$에 대한 multilinear extension. 

마지막으로, Linearized Committed CCS를 소개한다. 

structure는 마찬가지로 sparse multilinear polynomial $t$개와 multiset, constant $q$개로 구성되어 있다. 

instance의 경우, $(C, u, x, r, v_1, \cdots, v_t)$인데, 
- $C$는 $s'-1$변수 multilinear polynomial에 대한 commitment
- $u \in \mathbb{F}$, $x \in \mathbb{F}^l$, $r \in \mathbb{F}^s$, $v_i \in \mathbb{F}$

witness의 경우, 마찬가지로 $\tilde{w}$인데
- $C$는 $\tilde{w}$의 commitment 
- $v_i = \sum_{y \in \{0, 1\}^{s'}} \tilde{M_i}(r, y) \cdot \tilde{z}(y)$이며 이때 $z$는 $(w, u, x)$의 multilinear extension 

# multi-folding CCS 

$\mathcal{R}_1$을 Linearized Committed CCS, $\mathcal{R}_2$를 Committed CCS로 한다. 

일단 예상되면서도 아쉽게도 matrix들에 대응되는 multilinear polynomial들과 multiset, constant들은 전부 같아야 folding이 가능하다. 

이제 
- LCCCS instance $(C_1, u, x_1, r_x, v_1, \cdots, v_t)$
- LCCCS witness $\tilde{w_1}$
- CCCS instance $(C_2, x_2)$
- CCCS witness $\tilde{w_2}$

가 있을 때, Prover, Verifier의 작동을 확인하자. 

Step 1: Verifier가 $\gamma \in \mathbb{F}$와 $\beta \in \mathbb{F}^s$를 random sample하고 Prover에게 전달. 

Step 2: Verifier가 $r_x' \in \mathbb{F}^s$를 random sample 

Step 3. $r_x'$를 randomness로 하는 sumcheck를 돌리는데, 

$$g(x) = \gamma^{t+1} Q(x) + \sum_{j=1}^t \gamma^j L_j(x)$$

$$L_j(x) = \tilde{eq}(r_x, x) \cdot \left( \sum_{y \in \{0, 1\}^{s'}} \tilde{M_j}(x, y) \cdot \tilde{z_1}(y) \right)$$

$$Q(x) = \tilde{eq}(\beta, x) \cdot \left( \sum_{i=1}^q c_i \cdot \prod_{j \in S_i} \left( \sum_{y \in \{0, 1\}^{s'}} \tilde{M_j}(x, y) \cdot \tilde{z_2}(y) \right) \right)$$

예상되는 합은 물론 $\sum \gamma^j v_j$고 차수는 $d+1$. 

Step 4. $\sigma_i, \theta_i$를 계산하는데, 

$$\sigma_i = \sum_{y \in \{0, 1\}^{s'}} \tilde{M_i}(r_x', y) \cdot \tilde{z_1}(y)$$

$$\theta_i = \sum_{y \in \{0, 1\}^{s'}} \tilde{M_i}(r_x', y) \cdot \tilde{z_2}(y)$$

Step 5. $e_1 = \tilde{eq}(r_x, r_x')$와 $e_2 = \tilde{eq}(\beta, r_x')$를 계산하고 

$$g(r_x') \neq \gamma^{t+1} \cdot e_2 \cdot \left( \sum_{i=1}^q c_i \cdot \prod_{j \in S_i} \theta_i \right) + \sum_{j=1}^t \gamma^j \cdot e_1 \cdot \sigma_j$$

이면 fail. 

Step 6. Verifier는 $\rho \in \mathbb{F}$를 sample 하고 Prover에게 보낸다.

Step 7. 다음을 계산하고 마무리한다. 

$$(C', u', x', v_i', \tilde{w}') = (C_1, u, x_1, \sigma_i, \tilde{w}_1) + \rho \cdot (C_2, 1, x_2, \theta_i, \tilde{w}_2)$$

# HyperNova: IVC via Multi-Folding

Nova와 동일한 방식을 적용하면 된다. 

Augmented function $F'$는 
- $vk$: verifying key 
- $U_i$: LCCCS instance 
- $u_i$: CCCS instance 
- $(i, z_0, z_i), w_i$: IVC 결과 및 input
- $\pi$: folding proof

를 input으로 받고, 

- $i = 0$인 경우 $H(vk, 1, z_0, F(z_0, w_0), \text{LCCCS}_{\perp})$ return 
- 아닌 경우,
    - $u_i$의 public input이 $H(vk, i, z_0, z_i, U_i)$임을 확인 
    - $U_{i+1}$을 $U_i, u_i$를 folding 한 결과로 둠 
    - $H(vk, i + 1, z_0, F(z_i, \omega_i), U_{i+1})$ output 

이 $F'$를 CCCS로 옮겨, 

$$(u_{i+1}, w_{i+1}) \leftarrow \text{trace}(F', (vk, U_i, u_i, (i, z_0, z_i), \omega_i, \pi))$$

즉, 
- $\Pi_i$는 $((U_i, W_i), (u_i, w_i))$ 형태를 가지며
- $(U_{i+1}, W_{i+1}, \pi)$는 $((U_i, W_i), (u_i, w_i))$를 folding 한 결과물이고
- $(u_{i+1}, w_{i+1})$은 $(vk, U_i, u_i, (i, z_0, z_i), \omega_i, \pi)$에 대해 $F'$를 돌린 결과이며
- $\Pi_{i+1} = ((U_{i+1}, W_{i+1}), (u_{i+1}, w_{i+1}))$

검증자의 경우에는 
- $((U_i, W_i), (u_i, w_i))$를 각각 검증하고 
- $u_i$의 public input이 $H(vk, i, z_0, z_i, U_i)$인지 확인 

하면 된다. 결국 Nova와 다를 게 없으며, 아예 Nova를 black box로 보고 HyperNova를 구축할 수도 있다. 안전성 증명도 비슷하다.  