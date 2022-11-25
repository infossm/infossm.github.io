---
layout: post

title: "Pairing 제대로 알아보기"

date: 2022-11-22

author: rkm0959

tags: [cryptography]
---

자료: [Pairings For Beginners](https://static1.squarespace.com/static/5fdbb09f31d71c1227082339/t/5ff394720493bd28278889c6/1609798774687/PairingsForBeginners.pdf)


# Introduction

이 글에서는 여러 암호학의 분야에서 자주 등장하는 Elliptic Curve Pairing에 대해서 자세하게 알아보겠습니다. 독자 대상은 

- 현대대수학 기본 
- Elliptic Curve 연산에 대한 기초지식
- Elliptic Curve를 기반으로 한 Cryptography에 대한 기초지식
- Pairing을 기반으로 한 Cryptography에 대한 기초지식

이 있는 분들입니다. 즉, 이 글에서는 Pairing이 왜 쓸모있는지, 어떻게 사용되는지는 다루지 않고, Pairing이란 무엇이며 어떻게 효율적으로 계산되는지 다룹니다. 최대한 증명을 포함하려고 하나, 지나치게 증명이 어려운 경우에는 따로 알아두면 좋은 사실로 빼겠습니다.


## A Look at $E(\mathbb{F}_q)$

우선 다음 사실이 알려져 있습니다. 

$$E(\mathbb{F}_q) \simeq \mathbb{Z}_n \text{   or   } \mathbb{Z}_{n_1} \times \mathbb{Z}_{n_2} $$

즉, 타원곡선 군은 cyclic 하거나 두 cyclic group의 곱입니다. 

$r$-torsion이란, $[r]P = \mathcal{O}$를 만족하는 $P$의 집합입니다. 

한편, Hasse Bound에 의해서 

$$ \lvert \#E(\mathbb{F}_q) - (q + 1) \rvert \le 2 \sqrt{q} $$

가 성립하고, 이때 

$$t = q + 1 - \#E(\mathbb{F}_q)$$

를 trace of Frobenius라고 부릅니다. 또한, 이 경우 Hasse Bound를 만족하는 모든 $n$에 대하여 원소의 개수가 $n$인 elliptic curve over $\mathbb{F}_q$가 존재합니다. 이때 $q$는 소수. 

$E$에서 $E$로 가는 endomorphism들은 매우 중요한 연구 대상 중 하나인데, 그 중 가장 대표적인 것이 Frobenius Endomorphism 

$$\pi : E \rightarrow E, \quad (x, y) \rightarrow (x^q, y^q)$$

입니다. 이는 $E(\overline{\mathbb{F}_q})$에 있는 원소를 그대로 $E(\overline{\mathbb{F}_q})$로 보내며, 이는 Frobenius map의 성질을 생각해보면 쉽게 보일 수 있습니다. 특히, $\pi$가 $E(\mathbb{F}_q)$에서만 trivial 하게 작동함은 $\mathbb{F}_q$의 정의에서 알 수 있습니다. 

여기서 

$$\#E(\mathbb{F}_q) = \# \ker ([1] - \pi)$$

임을 얻을 수 있고, $[1] - \pi$의 degree에 대한 bound를 Cauchy-Schwarz로 얻어내면 Hasse Bound를 증명할 수 있습니다. 세부적인 과정은 Silverman의 책에 나옵니다. 

$\text{End}(E)$가 $\mathbb{Z}$보다 크다면, 즉, $P \rightarrow [m]P$ 형태가 아닌 endomorphism이 있다면, $E$가 complex multiplication을 가지고 있다고 합니다. 

Trace of Frobenius $t$의 가장 중요한 성질 중 하나는 

$$\pi^2 - [t] \circ \pi + [q] = 0$$

이 항등적으로 $E$ 위에서 성립한다는 것입니다. 

이 식을 가지고 Schoof's Algorithm을 도출할 수 있습니다. 여러 소수 $l$을 잡아서 $t \pmod {l}$을 얻은 다음, CRT로 합치고 $\lvert t \rvert \le 2\sqrt{q}$를 사용해서 $t$를 얻습니다. 

이를 위해서 $l$-torsion에 있는 점들 $(x, y)$를 가져와

$$(x^{q^2}, y^{q^2}) - [t \bmod l](x^q, y^q) + [q \bmod l](x, y) = \mathcal{O}$$

를 얻습니다. 여기서 $(x, y)$가 $l$-torsion에 있다는 조건을 추가하기 위해서 division polynomial을 가져옵니다. 

단순하게 말하면, 

$$[l](x, y) = \mathcal{O}$$

라는 조건을 $x$에 대한 식으로 표현하고, ($l$이 홀수면 가능) $y^2 = x^3 + ax + b$라는 제약 조건을 추가해서 $\mathbb{F}_q [x, y]$의 다항식들을 reduce 시키는 겁니다. 이제 적당한 brute force로 $t \bmod l$을 얻을 수 있습니다. 

## Divisors

Divisor란 formal sum

$$D = \sum_{P \in E(\overline{\mathbb{F}_q})} n_P(P)$$

를 말하며, 이때 유한한 개수의 $n_P$만이 non-zero입니다. 

이들은 자명하게 군을 이루며, $\text{Deg}(D) = \sum_P n_P$를 degree라 하며 $\text{supp}(D) = \{P : n_P \neq 0\}$를 support라고 합니다. 또한, $D$의 effective part를 

$$\epsilon(D) = \sum_{n_P \ge 0} n_P(P)$$

라고 합니다. 한편, $\text{ord}_P(f)$를 $E$ 위에서 정의된 $f$가 $P$에서 갖는 zero나 pole의 multiplicity라고 정의하면, 

$$(f) = \sum_P \text{ord}_P(f) (P)$$

를 divisor of $f$라고 정의합니다. 이때, $\text{Deg}((f)) = 0$임을 증명할 수 있습니다. 특히, $\mathcal{O}$에서 pole의 multiplicity를 세는 것이 중요합니다. 

Zero Divisor를 degree가 0인 divisor의 집합이라 하고, $((f))$ 형태로 표현할 수 있는 divisor를 principal divisor라 합시다. 이러면 

$$\text{Prin}(E) \subset \text{Div}^0(E) \subset \text{Div}(E)$$

가 성립하는데, principal divisor를 쉽게 classify 하는 조건은 

$$\sum_P n_P = 0, \quad \sum_P [n_P] P = \mathcal{O}$$

입니다. 이때, divisor class group을 

$$\text{Pic}^0(E) = \text{Div}^0(E) / \text{Prin}(E)$$

라 정의합니다. 이때, $E$의 genus (elliptic curve의 경우 1) $g$가 있어, (odd characteristic field 가정) 각 divisor $D \in \text{Pic}^0(E)$를 대표하는 unique representative가 

$$(P_1) + \cdots + (P_n) - n (\mathcal{O})$$

형태로 존재합니다. 이때, $n \le g$, $P_i + P_j \neq \mathcal{O}$가 $i \neq j$에 대해 성립하며, $2P_i = \mathcal{O}$인 $i$가 최대 하나 존재합니다. 

이를 $g = 1$에 대해서 정당화하는 것은 interpolation을 반복하고 principal divisor의 classification을 사용하는 것으로 어느 정도 할 수 있습니다. 

중요한 것은, genus 1인 elliptic curve에서는 

$$P \rightarrow (P) - (\mathcal{O})$$

라는 강력한 $E$와 $\text{Pic}^0(E)$ 사이의 homomorphism이 있다는 것입니다. 

한편, 함수 $f$를 divisor $D = \sum n_P(P)$에서 계산하는 것은 

$$f(D) = \prod_P f(P)^{n_P}$$

로 정의되며, 이때 $(f)$와 $D$의 support가 disjoint 해야 합니다.

특히, $f, g$가 $(f), (g)$의 support가 disjoint한 경우,

$$f((g)) = g((f))$$

가 성립합니다. 이를 Weil Reciprocity라 합니다.


## Pairings

기본적으로 bilinear pairing을 정의하려면 일단 $G_1, G_2, G_T$를 준비해야 합니다. 

이때, $e(P, Q)$를 계산하는 경우 기본적으로 $P, Q$는 다른 subgroup에서 나와야 합니다. 또한, 기본적으로 $G_1, G_2, G_T$는 셋 다 모두 크기 $r$인 ($r$은 소수) cyclic group이 됩니다. 

이러한 cyclic group을 찾을 수 있는 곳은 가장 대표적으로 

$$E[r] = \{P : [r]P = \mathcal{O} \}$$

즉 $r$-torsion입니다. 그런데 $E$의 underlying field의 characteristic이 0이거나 $r$과 서로소인 경우, 

$$E[r] \simeq \mathbb{Z}_r \times \mathbb{Z}_r$$

임이 알려져 있습니다. 즉, $r+1$개의 cyclic group이 있다고 볼 수 있습니다. 

이제 embedding degree $k$를 정의할 수 있는데, 이는 $q^k \equiv 1 \bmod{r}$을 만족시키는 최소의 $k$입니다. 이 경우, 

$$E[r] \subset E(\mathbb{F}_{q^k})$$

까지 성립하게 됩니다. 

한편, $P = (x, y) \in E(\mathbb{F}_{q^k})$에 대해서 trace map을 

$$Tr(P) = \sum_{i=0}^{k-1} \pi^i (P)$$

라고 하면 이는 $E(\mathbb{F}_q)$의 원소가 됩니다. 특히, 이는 group homomorphism임을 쉽게 알 수 있습니다. 

이제 

$$n = \#E(\mathbb{F}_q)$$

라 하고, $v_r(n) = 1$이라 합시다. 또한, embedding degree $k > 1$이라 합시다. 이러면 일단 $E(\mathbb{F}_q)$에는 크기 $r$ subgroup이 하나 존재하며, 이를 $\mathcal{G}_1 = E[r] \cap \ker([1] - \pi)$로 둘 수 있습니다. 

또 다른 크기 $r$ subgroup은 $\mathcal{G}_2 = E[r] \cap \ker([q] - \pi)$인데, 여기도 마찬가지로 $\pi$의 eigenspace라고 볼 수 있습니다. 특히, $P \in \mathcal{G}_2$면 $Tr(P) = \mathcal{O}$임을 보일 수 있습니다. 그래서 $\mathcal{G}_2$를 trace zero subgroup이라고 부릅니다. 

$\mathcal{G}_2$로 보내는 map은 anti-trace map 

$$aTr : P \rightarrow [k]P - Tr(P)$$

입니다. 이제 Pairing의 종류 4가지를 설명할 수 있습니다.

- Type 1: 이 경우에는 $\mathcal{G}_1$에서 다른 subgroup으로 나가는 efficient homomorphism $\phi$가 있습니다. 그래서 $G_1 = G_2 = \mathcal{G}_1$으로 선택하는게 가능하고, $e(P, Q)$를 계산할 때 $\hat{e}(P, \phi(Q))$를 계산하는 것으로 대체하는 것이 가능합니다. 이 경우, $E$는 supersingular 합니다. 
- Type 2: $G_2$가 $\mathcal{G}_1, \mathcal{G}_2$가 아닌 다른 $r-1$개의 subgroup 중 하나입니다. $G_1 = \mathcal{G}_1$입니다. 이 경우, $G_2$로 hashing 하는 것이 어렵습니다.
- Type 3: $G_2 = \mathcal{G}_2$입니다. 이러면 $G_2$로 hashing 하는 것은 어렵지 않으나, $G_2 \rightarrow G_1$으로 가는 isomorphism을 계산하는 것이 어렵습니다. 
- Type 4: $G_2 = E[r]$로 두는 방법이 있습니다. 

한편, twist를 사용하여 

$$E(\mathbb{F}_{q^k})$$

에서 작업하는 대신 

$$E(\mathbb{F}_{q^{k/d}})$$

에서 작업하는 최적화가 가능합니다. 이는 나중에 BLS12-381을 소개할 때 본격적으로 다루도록 하겠습니다. 

## Weil Pairing

Principal Divisor의 classification에 의해서 

$$(f_{m, P}) = m (P) - ([m]P) - (m-1)(\mathcal{O})$$

인 $f_{m, P}$가 존재합니다. 

이때, $P, Q$가 $E(\mathbb{F}_{q^k})$에 속하고 $r$-torsion에 있을 때, $D_P, D_Q$를 disjoint support를 갖으며 Picard group에서 

$$D_P \sim (P) - (\mathcal{O}), \quad D_Q \sim (Q) - (\mathcal{O})$$

이도록 하게 할 수 있고, principal divisor의 classification에서 

$$(f) = rD_P, \quad (g) = rD_Q$$

이도록 하는 $f, g$가 있습니다. 이때, Weil Pairing은 

$$w_r(P, Q) = \frac{f(D_Q)}{g(D_P)} \in \mu_r$$

로 정의되며, 우리가 아는 Pairing의 성질들을 전부 가집니다. 

이를 증명하기 위해서는 다음을 확인해야 합니다. 
- $D_P, D_Q$를 무엇으로 잡아도 $w_r$의 값이 고정인지
- $w_r$의 값이 $\mu_r$에 속하게 되는지 여부
- bilinearity 성질


이를 빠르게 증명해봅시다. 일단 divisor $D_1, D_2$에 대해서 

$$f(D_1 + D_2) = f(D_1) f(D_2)$$

임은 자명합니다. 이제, 예를 들어, $D_P$를 실제로 선택할 때 $(h)$라는 principal divisor가 추가가 되었다고 합시다. 그러면 

$$D_P^\star = D_P + (h), \quad (f^\star) = rD_P^\star = rD_P + r(h) $$

라 할 수 있고, 이 상태에서 계산된 Weil Pairing은 

$$\frac{f^\star(D_Q)}{g(D_P^\star)} = \frac{f(D_Q) h(D_Q)^r}{g(D_P) g((h))} = \frac{f(D_Q) h(D_Q)^r}{g(D_P)h((g))} = \frac{f(D_Q) h(D_Q)^r}{g(D_P)h(D_Q)^r} = \frac{f(D_Q)}{g(D_P)}$$

가 됩니다. 여기서 Weil Reciprocity가 사용되었습니다. 

또한, 

$$w_r(P, Q)^r = \frac{f(rD_Q)}{g(rD_P)} = \frac{f((g))}{g((f))} = 1$$

이므로 $w_r(P, Q) \in \mu_r$입니다. bilinearity도 쉽게 증명됩니다. 

## Tate Pairing

마찬가지로 $P \in \mathbb{E}(\mathbb{F}_{q^k})$라 하고, $r$-torsion에 있다고 합시다. $Q$는 

$$\mathbb{E}(\mathbb{F}_{q^k})/rE(\mathbb{F}_{q^k})$$

에 속하는 원소이며, $D_Q$는 $(Q) - (\mathcal{O})$와 Picard에서 equivalent한 divisor입니다. 또한, $(f) = r(P) - r(\mathcal{O})$인 함수 $f$가 있으며, $(f)$와 $D_Q$는 disjoint support를 갖도록 합니다. 이때, Tate Pairing $t_r$은 

$$t_r(P, Q) = f(D_Q) \in \mathbb{F}^\star_{q^k} / (\mathbb{F}^\star_{q^k})^r$$

로 정의됩니다. 한편, 

$$v_r(\#E(\mathbb{F}_{q^k})) = 2$$

인 경우에는 $E[r]$ 자체가 $E/rE$를 represent 할 수 있게 되어, Tate Pairing을 정의하기가 조금 더 간편해집니다. 

물론, 여기에 $(q^k - 1) / r$승을 추가하여 $\mu_r$로 보내는 것도 충분히 가능합니다. 이러면 reduced Tate Pairing이 완성됩니다. 보통 이걸 Tate Pairing이라고 하기도 합니다. 

이제 정당성을 증명해봅시다. 다하면 매우 길어지므로, 일부만 하겠습니다. 

$D_Q$에 함수 $h$에 대한 divisor $(h)$가 추가되면, 

$$f(D_Q + (h)) = f(D_Q) \cdot f((h)) = f(D_Q) \cdot h((f)) = f(D_Q) \cdot h((P) - (\mathcal{O}))^r$$

이 되므로 기존 값과 같은 coset에 들어가게 됩니다. 

또한, 

$$D_{Q_1 + Q_2} - D_{Q_1} - D_{Q_2} \sim (Q_1 + Q_2) - (Q_1) - (Q_2) + (\mathcal{O}) \sim 0$$

이 성립하므로 $Q$쪽에 대한 linearity가 성립합니다. 

$Q$ 대신에 $Q + rT$가 들어가면 

$$D_{Q + rT} - D_Q - r((T+U) - (U)) = (Q + rT) - (Q) - r((T+U) - (U)) \sim 0$$

이 되고 $f((T+U) - (U))^r$ 항이 coset 안으로 들어가므로 값이 유지됩니다. 

$(f_1) = r(P_1) - r(\mathcal{O})$, $(f_2) = r(P_2) - r(\mathcal{O})$, $(f_3) = r(P_1 + P_2) - r(\mathcal{O})$라면 

$$(f_3) - (f_1) - (f_2) = r (h)$$

인 $h$가 존재함을 쉽게 파악할 수 있고, 이에 따라 $P$에 대한 linearity가 증명됩니다. 

## Miller Loop

이제 Miller Loop를 다뤄봅시다. 

$$(f_{m, P}) = m(P) - ([m]P) - (m-1)(\mathcal{O})$$

이므로, binary exponentiation과 같은 방법으로 생각하면 

$$(f_{m+1, P}) - (f_{m, P}) = (P) + ([m]P) - ([m+1]P) - (\mathcal{O})$$

가 됩니다. 이에 대한 함수를 생각해보면 

- $l_{[m]P, P}$: $P, [m]P, -[m+1]P$를 지나는 직선
- $v_{[m+1]P}$: $\pm[m+1]P$를 지나는 vertical line

이라 했을 때, 

$$(l_{[m]P, P}) = (P) + ([m]P) + (-[m+1]P) - 3(\mathcal{O})$$

$$(v_{[m+1]P}) = ([m+1]P) + (-[m+1]P) - 2(\mathcal{O})$$

이므로 

$$(f_{m+1, P}) - (f_{m, P}) = (l_{[m]P, P}) - (v_{[m+1]P})$$

가 되어, 

$$f_{m+1, P} = f_{m, P} \cdot \frac{l_{[m]P, P}}{v_{[m+1]P}}$$

를 얻습니다. 이제 $m$을 $2m$으로 바꾸는 방법을 생각해보면, 

$$(f_{2m, P}) - 2(f_{m, P}) = 2([m]P) -([2m]P) - (O)$$

가 되고, 

- $l_{[m]P, [m]P}$: $[m]P, -[2m]P$를 지나는 접선
- $v_{[2m]P}$: $\pm [2m]P$를 지나는 vertical line

이라 했을 때, 

$$(l_{[m]P, [m]P}) = 2([m]P) + (-[2m]P) - 3(\mathcal{O})$$

$$(v_{[2m]P}) = ([2m]P) + (-[2m]P) - 2(\mathcal{O})$$

가 되어, 

$$(f_{2m, P}) - 2(f_{m, P}) = (l_{[m]P, [m]P}) - (v_{[2m]P})$$

를 얻습니다. 즉, 

$$f_{2m, P} = f_{m, P}^2 \cdot \frac{l_{[m]P, [m]P}}{v_{[2m]P}}$$

이제 $f_{r, P}$를 빠르게 evaluate 할 수 있습니다. 이걸 Miller Loop이라고 부릅니다.

## Further Optimizations

우선 기본적으로 Tate Pairing이 Miller Loop 개수가 적어 빠릅니다.

### 1. Denominator Elimination

첫 번째 최적화는 reduced Tate pairing에서 마지막에 $(q^k - 1) / r$승을 취한다는 것에서 착안합니다. 

만약 $k > 1$이라면, $r$은 $q-1$과 서로소일 것이니 $(q^k - 1) / r$은 $q - 1$의 배수가 됩니다. 이 말은 즉 $\mathbb{F}_q$에 속하는 값들이 더 곱해지거나 덜 곱해져도 상관없다는 것입니다. $(q^k - 1) / r$승을 취하면 결국 $1$이 되버리니까요. 

이 논리를 확장하면, 결국 $k$가 짝수인 경우에는 Miller Loop를 계산할 때 나오는 분모를 전부 무시해도 된다는 결론이 나옵니다. 대충 $k = 2l$이라고 하면 점들의 $x$ 좌표가 전부 

$$\mathbb{F}_{q^l}$$

에 속하도록 할 수 있고, 그러면 분모들이 전부 

$$\mathbb{F}_{q^l}$$

에 속할 것이므로 $(q^{k} - 1) / r$승을 하면 위 논리와 마찬가지로 $1$이 되기 때문입니다. 

특히 여기서 evaluation을 $D_Q$에서 하는 게 아니라 그냥 $Q$에서 해도 됨을 알 수 있습니다.

### 2. Projective Coordinates

Elliptic Curve operation들을 projective coordinate에서 하면서 division-free하게 진행, 더 속도를 올릴 수 있었던 것처럼, Miller Loop도 projective coordinate을 사용하면서 더 속도를 올릴 수 있습니다.  

### 3. Towered Extension Fields

$$\mathbb{F}_{q^k}$$

를 정의하기 위해서 $k$차 irreducible polynomial을 정의하는 게 아니라, extension field의 정의를 쪼개서 tower 형태로 만드는 것이 더 효율적입니다. 예를 들어, $k = 12$라고 하면, 

$$\mathbb{F}_q$$

에서 2차 기약다항식을 가져와 

$$\mathbb{F}_{q^2}$$

을 만들고, 다시 거기서 3차 기약다항식을 가져와 

$$\mathbb{F}_{q^6}$$

을 만들고, 다시 거기서 2차 기약다항식을 가져와 

$$\mathbb{F}_{q^{12}}$$

를 만드는 것이 더 효율적입니다. BLS12-381, BN254에서 많이 보게 되는 구조입니다. 

### 4. Low Hamming Weight

Miller Loop의 덧셈 개수를 ($r$의 hamming weight) 최소화하는 방식으로 최적화를 할 수 있습니다. 

### 5. Final Exponentiation

마지막으로 $(q^k - 1) / r$승을 취하는 부분도 상당한 시간을 차지하므로, 여기도 지켜볼 필요가 있습니다. 

우선 $k = 2d$라고 가정하면

$$(q^k - 1) / r = (q^d - 1) \cdot \left((q^d + 1) / \phi_k(q)\right) \cdot \left(\phi_k(q) / r\right)$$

이라 쓸 수 있습니다. $q^d - 1$승을 하고, $(q^d + 1) / \phi_k(q)$ 승을 한 뒤에, $\phi_k(q) / r$승을 하면 계산이 끝나게 됩니다. 앞쪽 두 개는 Frobenius Endomorphism을 적당히 하면 쉽게 계산할 수 있는데, 마지막이 문제입니다. $\phi_k(q) / r$을 처리하는 게 문제인데, 다양한 접근이 있지만 Curve-Specific하게 최적화를 하는 것이 좋습니다. 특히, BLS 계열 타원곡선에서는 $q, r$이 모두 변수 $x$에 대한 다항식 형태로 나타나기 때문에, $\phi_k(q) / r$을 $q$에 대한 다항식으로 잘 나타내서 최대한 Frobenius Endomorphism을 사용할 수 있는 방향으로 끌고 가는 것을 생각해볼 수 있습니다. 즉, 

$$\phi_k(q(x)) / r(x) = \sum_i \lambda_i(x) q(x)^i$$

로 쓰고, $v^{\lambda_i}$들을 최대한 쉽게 계산할 수 있는 각을 제공하는 게 목표입니다. 

여기서 이미 $(q^d - 1)$승이 들어간 상황이고, towered extension field를 사용한 상황이기 때문에 여러모로 계산이 편리한 상황임을 활용하면 더욱 속도 향상을 얻어낼 수 있습니다. 

### 6. Other Optimizations

Pairing 자체 연산 뿐만 아니라 다른 여러 면에서 최적화를 할 부분이 많습니다. 

자세한 detail은 추후에 본격적으로 BLS12-381이나 BN254를 다룰 때 알아보도록 하겠습니다. 

- 단순히 점에 scalar multiplication을 어떻게 할 것인가
- subgroup check를 어떻게 할 것인가
- hash to (sub)group을 어떻게 할 것인가

# Conclusion

지금까지 Pairing에 대해서 알아보았습니다. 긴 글 읽어주셔서 감사합니다.
다음 글에서는 Optimal Ate Pairing, BLS12-381과 BN254에 대해서 자세하게 알아보겠습니다. 가능하면 다른 타원곡선들까지 알아보도록 하겠습니다. 