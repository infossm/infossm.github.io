---
layout: post

title: "Introduction to Bilateral Trade"

date: 2023-08-20

author: ainta

tags: [mechanism-design]
---

#### Bilateral Trade

Bilateral Trade란 기본적으로 구매자 한 명과 판매자 한 명이 item을 거래하는 것입니다. 즉, 다음과 같은 구성 요소로 이루어진다고 볼 수 있습니다:

- 두 agent: 구매자와 판매자
- 구매자의 item에 대한 평가(valuation) $B \sim F_B$, 판매자의 평가 $S \sim F_S$

여기에서 $F_B$와 $F_S$는 각각 $B$와 $S$가 선택되는 확률분포를 뜻합니다.
#### Bilateral Trade Mechanism

Bilateral Trade에서의 Mechanism이란 다음 두 가지 함수를 어떻게 설정하는지를 뜻합니다.

- 할당 함수(allocation function) $A:\mathbb{R} \times \mathbb{R} \rightarrow \{ 0, 1 \}$. 
	- 두 reported valuation $b, s$를 받아 거래가 발생할지를 결정합니다. $A(s,b) = 1$ (거래가 발생한 경우), 0 (거래가 발생하지 않은 경우).
- 지불 함수(payment function) $\Pi: \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ 
	- 거래가 발생하면 가격을 결정합니다.

이 때, 할당 함수와 지불 함수에서 받는 reported valuation은 실제 agent의 valuation과 일치하지 않을 수 있습니다. 즉, 자신이 생각하는 valuation이 아닌 값을 전략적으로 거짓 report하는 것이 가능합니다.


### Bilateral Trade

메커니즘의 성능(performance)을 측정하는 대표적인 두 값으로는 GFT와 wellfare가 있습니다:

1. 거래로부터의 이익(Gains from trade, GFT): 거래가 발생하는 경우 $B-S$, 그렇지 않으면 $0$.
2. welfare: 거래가 발생하는 경우 $B$, 그렇지 않으면 $S$.

여기서 (welfare) - (GFT) = $S$는 항상 성립함을 알 수 있습니다. 
welfare의 상한은 $\max(B, S)$, GFT의 상한은 $\max(B-S, 0)$ 이므로, 이를 “first-best” optimum이라 부릅니다.

#### Myerson-Satterthwaite Theorem

지금부터 는 Myerson와 Satterthwaite의 논문에서 사용된 표기법을 사용하여 그들의 결과를 설명하겠습니다. 

**Notation.**

- 사람 1은 판매자이며, 사람 2는 구매자이다.
- 각 개인의 평가는 $V_1$, $V_2$ (=$S,B$).
- $V_i$는 주어진 구간 $[a_i, b_i]$에서 분포되어 있다.
- $f_i$는 $V_i$에 대한 probability density function으로, $[a_i, b_i]$에서 연속적이고 양수이다.
- $F_i$는 $f_i$에 해당하는 cdf(cumulative density function)이다. ($F_1 = F_S$, $F_2 = F_B$)
- 각 개인은 자신의 평가를 알고 있지만, 다른 사람의 평가를 확률 변수로 간주한다. 즉, 사람 1(판매자)은 $V_1$와 $F_2$를 알지만, $V_2$는 알 수 없다.

#### Direct Mechanism

Bilateral trade에서, 다음과 같은 mechanism을 **Direct Mechanism**이라 합니다.  Myerson와 Satterthwaite의 논문에서의 notation을 따르기 때문에 앞서 Bilateral trade의 mechanism 정의에서 사용했던 allocation function, payment function의 notation과 기호가 달라짐에 유의하시길 바랍니다.

- 각 개인은 동시에 valuation을 report한다.
- Direct Mechanism은 두 개의 함수 $p, x$로 특징지어진다.
- $p(v_1, v_2)$: 보고된 valuation이 $v_1, v_2$일 때 거래가 발생하는 확률 (=$A(s,b)$)
- $x(v_1, v_2)$: 보고된 valuation이 $v_1, v_2$일 때 구매자에서 판매자로의 예상 지불액 (=$\Pi(s,b)$)

Direct Mechanism에서, 정직하게 보고할 때(각각 $V_1, V_2$로 보고) Bayesian Nash Equilibrium을 형성하는 경우를 **Bayesian incentive-compatible(BIC)** 라 합니다. 즉, BIC 메커니즘에서, 각 개인은 다른 사람이 정직하게 보고한다고 가정할 때, 본인이 정직하게 보고하는 것이  그 개인의 expected utility를 최대화합니다 (이득의 기댓값이 최대).


**Theorem (Revelation principle).** Bilateral trade의 Bayesian Equilibrium에 대해, 항상 동일한 결과를 제공하는 동등한 **incentive-compatible direct mechanism**이 존재한다.

Revelation principle은 mechanism design에서 기본이 되는 중요한 정리 중의 하나입니다. 위 결과를 통해, 우리는 일반성을 잃지 않고 incentive-compatible direct mechanisms만 살펴보아도 문제가 없음을 알 수 있습니다.

또한, 각 개인의 expected gain이 음수면 이 trade에 참가할 이유가 없으므로, 다음 조건을 만족하는 directed mechanism에만 집중해도 충분합니다.
- **Individual Rationality(IR)**:  각 개인의 expected gain은 0 이상
- **Bayesian Incentive Compatibility(BIC)**

Myerson-Satterthwaite Theorem은 위 조건을 만족하는 mechanism이 first-best optimum을 달성할 수 없다는 정리입니다. 풀어서 말하면, 판매자와 구매자의 expected gain이 모두 0 이상이고, 정직하게 report하는 것이 bayesian nash equilibrium(해당 상태에서 report를 수정하면 expected gain이 내려가며)인 direct mechanism의 경우, trade의 welfare가 항상 $\max(B,S)$가 되게 하는 것은 불가능하다는 의미입니다.  이를 증명해 봅시다.

먼저, 다음 값들을 생각해봅시다.

- $\bar{x_1}(v_1) = \int_{a_2}^{b_2} x(v_1, t_2)f_2(t_2)dt_2$, $\bar{x_2}(v_2) = \int_{a_1}^{b_1} x(t_1, v_2)f_1(t_1)dt_1$
- $\bar{p_1}(v_1) = \int_{a_2}^{b_2} p(v_1, t_2)f_2(t_2)dt_2$, $\bar{p_2}(v_2) = \int_{a_1}^{b_1} p(t_1, v_2)f_1(t_1)dt_1$
- $U_1(v_1) = \bar{x_1}(v_1) - v_1 \bar{p_1}(v_1)$, $U_2(v_2) = v_2 \bar{p_2}(v_2) - \bar{x_2}(v_2)$

$U_1(v_1)$ 은 판매자의 valuation이 $v_1$일 때 expected gain을, $U_2(v_2)$은 구매자의 valuation이 $v_2$일 때 expected gain을 나타냅니다.

Individual Rationality(IR)을 만족한다는 것은 $U_1(v_1) \ge 0, U_2(v_2) \ge 0$ for all $v_1, v_2$임을 뜻하고,

BIC를 만족한다는 것은 true valuation $v_i$와 임의의 $\hat{v_i}$ 에 대해서  $U_1(v_1) \ge \bar{x_1}(\hat{v_1}) - v_1 \bar{p_1}(\hat{v_1})$와 $U_2(v_2) \ge v_2 \bar{p_2}(\hat{v_2}) - \bar{x_2}(\hat{v_2})$ 가 성립함을 뜻합니다.

**Lemma.** For any BIC mechanism, $U_1(b_1) = \min_{v_1} U_1(v_1)$, $U_2(a_2) = \min_{v_2} U_2(v_2)$

**Proof.** 

모든 $v_1, \hat{v_1}$에 대해, 다음이 성립합니다.
- $U_1(v_1) = \bar{x_1}(v_1) - v_1 \bar{p_1}(v_1) \ge   \bar{x_1}(\hat{v_1}) - v_1 \bar{p_1}(\hat{v_1})$ 
- $U_1(\hat{v_1}) = \bar{x_1}(\hat{v_1}) - \hat{v_1} \bar{p_1}(\hat{v_1}) \ge   \bar{x_1}(v_1) - \hat{v_1} \bar{p_1}(v_1)$ 

따라서, $(\hat{v_1} - v_1)\bar{p_1}(v_1) \ge U_1(v_1) - U_1(\hat{v_1}) \ge (\hat{v_1} - v_1)\bar{p_1}(\hat{v_1})$.

$U_1'(v_1) = -\bar{p_1}(v_1)$ 이므로 
$U_1(v_1) = U_1(b_1) + \int_{v_1}^{b_1} \bar{p_1}(t_1)dt_1$ 는 감소함수입니다.

마찬가지로, $U_2(v_1) = U_2(a_2) + \int_{a_2}^{v_2} \bar{p_2}(t_2)dt_2$는 증가함수입니다.

따라서, $U_1(b_1) = \min_{v_1} U_1(v_1)$ and $U_2(a_2) = \min_{v_2} U_2(v_2)$


위 Lemma를 통해, 다음을 증명할 수 있습니다.

**Theorem.**         For any BIC mechanism, 

$$U_1(b_1) + U_2(a_2) = \min_{v_1} U_1(v_1) + \min_{v_2} U_2(v_2) = \\ \int_{a_2}^{b_2} \int_{a_1}^{b_1} \left[ f_1(v_1)(v_2f_2(v_2) - (1-F_2(v_2)) - f_2(v_2)(v_1f_1(v_1) - F_1(v_1)) \right] \cdot p(v_1,v_2)dv_1dv_2$$


위 정리는 $U_1(b_1) + U_2(a_2)$가 $F_i$와 $p_i$에만 의존하고, $x_i$와는 관계없음을 보여줍니다.

**Definition(Ex post efficiency).** A mechanism $(p,x)$ is ex post efficient if:

- $p(v_1, v_2) = 1$ if $v_1 < v_2$
- $p(v_1, v_2) = 0$ if $v_2 > v_1$

정의에 의해 자명하게도 ex post efficient mechanism은 first-best optimum을 달성합니다. 그리고 $\bar{p_1}(v_1) = 1-F_2(v_1)$, $\bar{p_2}(v_2) = F_1(v_2)$이 성립합니다.

$\max(a_1, a_2) < \min(b_1, b_2)$라 합시다. 즉, 판매자와 구매자의 valuation domain이 겹친다고 가정합시다. ex-post efficient mechanism에 대해,

$$
\begin{align*} U_1(b_1) + U_2(a_2) &= \int_{a_2}^{b_2} \int_{a_1}^{b_1} \left[ f_1(v_1)(v_2f_2(v_2) - (1-F_2(v_2)) \right. \\ &\qquad\left. - f_2(v_2)(v_1f_1(v_1) - F_1(v_1)) \right] \cdot p(v_1,v_2)dv_1dv_2 \\ &= -\int_{a_2}^{b_1} (1-F_2(t))F_1(t)dt \\ &< 0 \end{align*}
$$

이 성립하므로 이 mechanism은 IR을 만족할 수 없습니다. 이로써 다음 Myerson-Satterthwaite Theorem이 증명되었습니다.

**Myerson-Satterthwaite theorem.** if $(a_1, b_1)$ and $(a_2, b_2)$ intersects, then no Bayesian incentive-compatible individually rational mechanism can be ex post efficient.



## Approximations in bilateral trade

우리가 원하는 조건 (BIC, IR)을 만족하는 mechanism 중에서는 First-best optimum을 달성하는 mechanism이 존재하지 않는다는 것을 앞서 살펴보았습니다. 그러면 이제 자연스럽게 optimum에 얼마나 가깝도록 만들 수 있는지로 관심사가 쏠릴 것입니다.

분석을 보다 간단히 하기 위해서, 앞으로는 BIC보다 좀더 좁은 개념의 incentive compatibility를 만족하는 mechanism만 고려하기로 합니다. 둘의 차이는 아래와 같습니다.
- Bayesian incentive compatibility(BIC): true value를 report하는 것이 두 구매자에게 optimal, **in expectation**.
- Dominant-strategy incentive compatibility(DSIC): true value를 reporting하는 것이 **항상** optimal.

DSIC mechanism은 BIC mechanism이지만 역은 성립하지 않습니다. 우리의 목표는 IR, DSIC를 만족하는 direct mechanism의 welfare와 GFT를 최적에 가깝도록 만드는 것입니다.


2021년 발표된 Fixed-Price Approximations in Bilateral Trade 논문의 결과는 아래와 같습니다. 이 중 *가 붙은 결과는 tight bound입니다.
![Approximation ratio](/assets/images/bilateral-trade/ff.png)

각각의 결과가 어떻게 도출되는지 이제부터 알아보도록 합시다.


### DSIC Mechanisms
정의에 의해, DSIC mechanism은 다음을 만족한다:
- $U_1(b_1) - s \cdot A(s,b) \ge \Pi(s', b) - s \cdot A(s', b)$
- $b \cdot A(s,b) - \Pi(s,b) \ge b \cdot A(s, b') - \Pi(s, b')$

**Theorem.** Bilateral trade의 DSIC mechanism은 결국 Fixed-price mechanism 또는 그와 동등한 mechanism이다. 

여기서 Fixed-price mechanism이란 payment function $\Pi$ 가 $F_S$ and $F_B$에만 관련이 있는 single-value function이며 allocation function은 $A(b,s) = \mathbf{1}_{s\le p\le b}$라는 뜻입니다.

즉, 가격이 valuation의 분포에 따라 정해지며, trade는 가격 $p$가 구매자의 valuation 이하이고 판매자의 valuation 이상일 때 이루어지는 mechanism입니다.

위 정리를 통해서 고려할 mechanism을 Fixed-price mechanism으로 한정하고 나면 이제 분석이 상당히 간단하게 가능합니다. 어떻게 가능한지를 다음 단락에서 알아보도록 합니다.

## GFT and Welfare in Symmetric Cases

Bilateral trade에서, 판매자와 구매자의 valuation 분포가 동일한 경우, 즉 $F_B = F_S = F$인 경우를 **symmetric** case라 한다. 이 때 GFT와 Welfare는 다음과 같이 계산됩니다.

- Optimal gains from trade $\text{OPT-}GFT(F) = \mathbb{E}[\mathbf{1}_{B>S} (B-S)]$ 

- Gains from trade $GFT(p, F) = \mathbb{E}[\mathbf{1}_{B\ge p>S} (B-S)]$

- Optimal welfare $\text{OPT-}W(F) = \mathbb{E}[S] + \mathbb{E}[\mathbf{1}_{B>S}(B-S)] = \mathbb{E}[S] + \text{OPT-}GFT(F)$

- welfare $W(p,F) = \mathbb{E}[S] + GFT(p,F)$

cdf $F$에 대한 pdf(확률밀도함수)를 $f$라 하면, 다음이 성립합니다:

$$
\begin{aligned}
\text{OPT-}GFT(F) &= \mathbb{E}[\mathbf{1}_{B>S}(B-S)] \\
 &= \int_{0}^{\infty} \mathbf{1}_{S \le x < B} dx \\
 &= \int_{0}^{\infty} F(x)(1-F(x)) dx \\
\end{aligned}
$$

$$
\begin{aligned}
GFT(p,F) &= \mathbb{E}[\mathbf{1}_{B \ge p >S}(B-S)] \\
 &= \mathbb{E}[\mathbf{1}_{B \ge p >S}(B-p)] +\mathbb{E}[\mathbf{1}_{B \ge p >S}(p-S)] \\
 &= \mathbb{E}[\mathbf{1}_{B \ge p}(B-p)]Pr(S<p) +\mathbb{E}[\mathbf{1}_{p >S}(p-S)]Pr(B \ge p) \\
 &= F(p) \int_{p}^{\infty} (1-F(x)) dx + (1-F(p)) \int_{0}^{p} F(x) dx \\
\end{aligned}
$$


## Single Sample Approximation and Its Implications

fixed-price mechanism에서 payment function $\Pi$를 정할 때, $F$ 분포의 모든 정보를 알 수 있는 것이 아니라 $F$로부터 하나의 sample만을 얻을 수 있는 경우를 생각해 봅시다. 이 때, 다음이 성립합니다.


**Theorem.** $p \sim F$를 추출하여 이를 price로 하는 fixed-price mechanism은 OPT-GFT의 1/2를 보장한다.

### Proof

$$
\begin{aligned}
    
    & \mathbb{E}_{p \sim F}[GFT(p,F)] \\
    &= \int_{0}^{\infty} \left[ F(p) \int_{p}^{\infty} (1-F(x)) dx + (1-F(p)) \int_{0}^{p} F(x) dx \right] f(p)dp 
\end{aligned}
$$


- $\gamma_1 = \int_{0}^{\infty} f(p)F(p) \int_{p}^{\infty} (1-F(x)) dx dp$
- $\gamma_2 = \int_{0}^{\infty} f(p)(1-F(p)) \int_{0}^{p} F(x) dx dp$

라 하면
$\mathbb{E}_{p \sim F}[GFT(p,F)] = \gamma_1 + \gamma_2$가 성립한다.

$$
\begin{aligned}
    \gamma_1 &=  \int_{0}^{\infty} f(p)F(p) \int_{p}^{\infty} (1-F(x)) dx dp \\
    &= \frac{1}{2} \int_{0}^{\infty} F(p)^2 (1-F(p)) dp \text{   } (\text{some calculations are omitted})
\end{aligned}

\begin{aligned}
    \gamma_2 &=  \int_{0}^{\infty} f(p)(1-F(p)) \int_{0}^{p} F(x) dx dp \\
    &= \frac{1}{2} \int_{0}^{\infty} F(p) (1-F(p))^2 dp 
\end{aligned}
$$

따라서, 

$$\mathbb{E}_{p \sim F}[GFT(p,F)] = \gamma_1 + \gamma_2 = \frac{1}{2} \int_{0}^{\infty} F(x)(1-F(x)) dx$$

이며 이는 $\text{OPT-GFT}(F)/2$과 같다. $\blacksquare$

**Theorem.** $p \sim F$를 추출하여 이를 price로 하는 fixed-price mechanism은 Optimal Welfare의 3/4를 보장한다.

$$
\begin{aligned}
\frac{\mathbb{E}_{p \sim F}[GFT(p,F)]}{\text{OPT-W}(F)} &= \frac{\mu + \mathbb{E}_{p \sim F}[GFT(p,F)]}{\mu + \text{OPT-GFT}(F)}   \\  
&= \frac{\mu + \text{OPT-GFT}(F)/2}{\mu + \text{OPT-GFT}(F)}
\end{aligned}
$$

한편,
$$
\begin{aligned}
OPT-GFT(F) &= \int_{0}^{\infty} F(x)(1-F(x))dx \le \int_{0}^{\infty} 1 \cdot (1-F(x))dx \\
&= \int_{0}^{\infty} Pr[t \ge x]_{t \sim F} = \mu
\end{aligned}
$$

$$
\therefore \frac{\mathbb{E}_{p \sim F}[GFT(p,F)]}{\text{OPT-W}(F)} \ge \frac{3}{4}.
$$

## Best-Possible Approximation of Welfare in Symmetric Case

아까와 달리 $F$가 fully 알려져있는 상태라고 가정합시다. 이 때, 다음이 성립합니다.

**Theorem.** $p = \mu$ is optimal. That is, $p^* = \mathbb{E}[S] = \mathbb{E}[B]$

즉, 분포의 펑균으로 price를 놓는 것이 언제나 최적이 됩니다.

**Proof.**

$$
\begin{aligned}
W(p,F) &= \mathbb{E}[S] + \mathbb{E}[\mathbf{1}_{B\ge p>S} (B-S)] \\
&= \mathbb{E}[S] + \mathbb{E}[B \cdot \mathbf{1}_{B>p}]\cdot F(p) - \mathbb{E}[S \cdot \mathbf{1}_{S\le p}](1-F(p)) \\
&= \mathbb{E}[S] + (\mathbb{E}[S] - \mathbb{E}[S \cdot \mathbf{1}_{S\le p}])\cdot F(p) - \mathbb{E}[S \cdot \mathbf{1}_{S\le p}](1-F(p)) \\
&= \mathbb{E}[S](1 + F(p)) - \mathbb{E}[S \cdot \mathbf{1}_{S \le p}] \\
&= \mathbb{E}[S](1 + F(p)) - pF(p) + \int_{0}^{p}F(s)ds
\end{aligned}
$$

$$W(p,F) = \mathbb{E}[S](1 + F(p)) - pF(p) + \int_{0}^{p}F(s)ds$$

$$\frac{dW}{dp} = \mathbb{E}[S]f(p) - F(p) - pf(p) + F(p) = (\mathbb{E}[S]-p)f(p)$$

따라서, $W(p,F)$ 는 $\mu = \mathbb{E}[S] = p$일 때 maximize. $\blacksquare$

이 때 approximation ratio는 $\frac{2+\sqrt{2}}{4}$ 이상임이 보장되는 것을 case-work를 통해 보일 수 있습니다. 

## 6. Reference

- Zi Yang Kang, Francisco Pernice, and Jan Vondrák. Fixed-price approximations in bilateral trade. In
Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2964–2985. SIAM, 2022.
