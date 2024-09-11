---
layout: post

title: "New Updates on Proximity Testing with Logarithmic Randomness"

date: 2024-09-03

author: rkm0959

tags: [cryptography, zero-knowledge]
---

# 소개
[이전 글](https://infossm.github.io/blog/2024/01/28/LogarithmicRandomness/)에서, "Proximity Testing with Logarithmic Randomness"라는 논문을 소개했습니다. 이 논문을 읽은 당시에 제가 논문에 있었던 증명의 작은 오류를 찾아 이를 수정하는데 기여하기도 했으며, 그 내용도 이전 글에 소개되어 있습니다. 1월 이후, 해당 논문과 관련된 여러 새로운 발견이 이루어졌습니다. 이 글에서는 1월 이후 Proximity Testing에 대한 추가적인 발전에 대해서 다룹니다. 

# 복습 

이전 논문에서 다룬 핵심적인 정리는 다음과 같습니다. 

> For any $[n, k, d]$ code $V \in \mathbb{F}\_q^n$ and $e < d/3$, given $u\_0, \cdots, u\_{m-1} \in \mathbb{F}\_q^n$ such that 
> 
> $$\text{Pr} \left( d( [\otimes_{i=0}^{\log m - 1} (1 - r_i, r_i)] \cdot [u_0, \cdots, u_{m-1}]^T, V) \le e \right) > 2 \cdot \log m \cdot \frac{e}{q}$$
> 
> then $U = [u_0, \cdots, u_{m-1}]$ is $e$-close to $V$. 

즉, $\log m$개의 randomness를 통해서 $m$개의 vector가 $V$와 가깝고, 그 $m$개의 "가까움"이 서로 "correlated"한지, 즉 correlated agreement가 이루어지는지를 확인할 수 있습니다. 

당시 논문에서 open problem으로 남겨져 있던 것은 두 개가 있습니다. 

**문제 1**: 위 정리에서 우변의 상수 $2$를 없앨 수 있는가? 

실제로, $m = 2$인 경우 우변이 $(e+1)/q$여도 충분함이 알려져 있으므로, 우변을 

$$2 \cdot \log m \cdot \frac{e}{q} \rightarrow \log m \cdot \frac{e+1}{q}$$

로 최적화할 수 있음을 충분히 기대할 수 있습니다. 

**문제 2**: 위 명제에서 $e < d/3$의 조건을 확장할 수 있는가? $e < d/2$라면?

일반적인 code와 관련된 기법에서는 $d/3$이 자주 등장하지만, 사실 가장 중요한 기준점은 unique decoding radius $d/2$입니다. Reed-Solomon code와 같은 곳에서는 $d/2$가 사용 가능함이 증명되어 있으므로 (BCIKS20 논문 등) 이를 일반적인 code에 적용하려고 하는 것도 말이 됩니다. 

이번 글에서 다룰, 이 문제들에 대한 답을 결론부터 말하자면 다음과 같습니다. 

**문제 1**: 2를 없앨 수 있고, 최적화가 가능합니다. 이를 해결하는데 저도 기여를 조금이지만 했습니다. 

**문제 2**: 우선 $e< d/2$인 경우에는 $m = 2$인 경우조차 증명이 없습니다. 다만, $m = 2$일 때 우변의 값이 $(e+1)/q$여도 기존 명제가 성립한다는 추측은 반례가 있음이 확인되었습니다. 이에 따라, 기존 논문의 저자들은 $e < d/2$인 경우의 추측을 우변의 값이 $n/q$인 명제에 대한 추측으로 변경했습니다. 이 새로운 추측은 아직도 open problem으로 남아있습니다. 다만, Reed-Solomon code에 대해서는 이 추측이 참임이 알려져 있습니다 (BCIKS20). 한편, $m = 2$인 경우에 $e < d/2$에 대한 명제가 참이라면 $m \ge 2$ 전부에 대해서도 명제가 참임이 최근에 증명되었습니다. 즉, Reed-Solomon code에 대해서는 위 명제를 사용할 수 있어, FRI의 개선을 모색할 수 있습니다. 

최근 논문인 https://eprint.iacr.org/2024/1351.pdf 가 문제 1을 해결한 note를 인용하므로, 2024/1351을 기준으로 설명하겠습니다. 2024/1351을 제외한 다른 레퍼런스를 소개하자면, 

- 문제 1을 $e < d/3$에서 해결한 방법이 zkSummit11에서 소개되었습니다. [영상](https://www.youtube.com/watch?v=0KkmI2rU2j4&list=PLj80z0cJm8QFy2umHqu77a8dbZSqpSH54&index=23).
- 문제 2의 $(e+1)/q$에 대한 반례는 0xPARC 세션에서 독립적으로 찾아졌습니다. [링크](https://notes.0xparc.org/results/counterexample-proximity-gap/). 
    - 이 글에서는 0xPARC의 반례와 2024/1351의 반례 모두 소개합니다. 

# Proximity on Interleaved Codes

$[n, k, d]$ code $C$가 $(e, \epsilon)$에 대해서 proximity gap for affine line이 있다는 것은,

$$\text{Pr}_r (d((1-r)u_0 + ru_1, C) \le e) > \epsilon / q$$

가 성립하는 경우, $d^2((u_i)_{i=0}^1, C^2) \le e$ 역시 성립함이 보장된다는 것입니다. 

즉, line에 대한 batch testing을 통해서 correlated agreement를 얻을 수 있습니다. 이때, $e$를 proximity parameter, $\epsilon$을 false witness bound라 합니다. BCIKS20의 중요한 결과 중 하나는 Reed-Solomon code가 $(\lfloor (d-1)/2 \rfloor, n)$ proximity gap for affine line이 있다는 것입니다. 

한편, $(e, \epsilon)$에 대한 tensor-style proximity gap은 

$$\text{Pr} \left( d( [\otimes_{i=0}^{m - 1} (1 - r_i, r_i)] \cdot [u_0, \cdots, u_{2^m-1}]^T, C) \le e \right) >  m \cdot \frac{\epsilon}{q}$$

이면 $d^{2^m}([u_0, \cdots, u_{2^m-1}], C^{2^m}) \le e$가 보장된다는 것입니다.

즉, 기존 논문 2023/630에서 제시한 스타일의 proximity testing이 가능하다는 것입니다. 

중요한 첫 번째 결과는 $e < d/2$이고 $C$가 $(e, \epsilon)$-proximity gap for affine line을 갖는다면, 그 interleaving $C^m$ 역시 $(e, \epsilon)$-proximity gap for affine line을 갖는다는 것입니다. 단, $\epsilon \ge e+1$.

$C$가 해당 조건을 만족한다고 하고, $\mathbb{F}\_q^{m \times n}$의 원소 $U\_0, U\_1$을 가져옵시다. 이제 $r \in \mathbb{F}\_q$에 대하여 $U\_r = (1-r) \cdot U\_0 + r \cdot U\_1$이라고 하고, $R^\star$를 

$$R^\star = \{r \in \mathbb{F}_q : d^m(U\_r, C^m) \le e\}$$

라고 합시다. 목표는 $\lvert R^\star \rvert > \epsilon$이면 $U_0, U_1$이 $C^m$과 correlated agreement를 가짐을 보이는 것입니다. 즉, $V_0, V_1$이 $C^m$의 원소고, $U_i, V_i$ 사이의 차이가 발생하는 index의 집합인 $\Delta(U_0, V_0)$와 $\Delta(U_1, V_1)$이 correlated 되어, 두 집합의 합집합 $D$의 크기가 $e$ 이하가 된다는 것입니다. 우선 $V_0, V_1$부터 찾고 시작합시다. 

이 부분은 쉽습니다.. $(U_0)_i$와 $(U_1)_i$를 각각 $U_0, U_1$의 $i$번째 row라고 생각하면, $R^\star$의 모든 $r$에 대하여 

$$d((1-r) (U_0)_i + r (U_1)_i, C) = d((U_r)_i, C) \le d(U_r, C^m) \le e$$

가 성립하므로, $C$가 $(e, \epsilon)$ proximity gap for affine line이 있다는 걸 생각하면 결국 이에 대응하는 codeword $(V_0)_i, (V_1)_i$가 있음을 알 수 있습니다. 이를 하나로 모아서 $V_0, V_1 \in C^m$을 찾을 수 있고, 비슷하게 $V_r = (1-r) \cdot V_0 + r \cdot V_1$을 정의할 수 있습니다. 이제 

$$U = \left[ \begin{matrix} U_0 \\ U_1 \end{matrix} \right], V = \left[ \begin{matrix} V_0 \\ V_1 \end{matrix} \right]$$

를 정의합시다. 이제 correlated agreement를 보여야 합니다. 

그 전에, 우선 $r \in R^\star$에 대해서 $d^m(U_r, V_r) \le e$를 보일 수 있습니다. 사실 꽤 당연한데, $d^m(U_r, C^m) \le e$임을 이미 알고 있고, $U_r$과 가까울 수 있는 $C^m$의 원소는 $V_r$ 뿐이기 때문입니다. 이미 unique decoding radius 안에 있고, $U_r$의 row와 $e$ 거리 안에 있는 $C$의 원소는 $V_r$의 row 뿐이기 때문입니다. 

이제 본격적인 증명을 위해, 

$$R^{\star\star} = \{(r, j) \in R^\star \times \{0, 1, \cdots ,n-1\} : (U_r)^{j} = (V_r)^{j}\}$$

를 double counting 합니다. 단, $(U_r)^{j}, (V_r)^{j}$는 $U_r, V_r$의 $j$번째 column입니다. $D$를 $\Delta(U, V)$로 정의합시다. 

**Idea 1**: $\lvert R^{\star\star} \rvert \le \lvert R^\star \rvert (n - \lvert D \rvert) + \lvert D \rvert$

$D$에 있는 column을 생각을 해봅시다. 해당 column에은 $(U_0, U_1)$에서와 $(V_0, V_1)$에서의 값이 다르므로, 그 선형결합 $(1-r) \cdot U_0 + r \cdot U_1$과 $(1-r) \cdot V_0 + r \cdot V_1$에서 column의 값이 같을 수 있는 $r$의 개수는 최대 1개입니다. 

$D$에 없는 column들에 대해서는 $R^\star$ 전부에서 $U_r, V_r$의 column이 같습니다. 

**Idea 2**: $\lvert R^{\star\star} \rvert \ge (n-e) \cdot \lvert R^\star \rvert$

각 $r \in R^\star$에 대해서, $d^m(U_r, V_r) \le e$이므로, $n-e$개의 column은 같습니다. 

두 부등식을 합치면 결과적으로 

$$\lvert D \rvert \le e \cdot \frac{\lvert R^\star \rvert}{\lvert R^\star \vert - 1} < e+1$$

이 성립하여 $\lvert D \rvert \le e$가 성립합니다. 증명 끝. 

이제 $C^m$의 proximity gap for affine line을 통해서, $C$에 대한 tensor-style proximity gap을 증명할 수 있습니다. 이 부분이 상수 $2$를 제거하는 부분과 같습니다. 기본적으로, $m$에 대한 수학적 귀납법을 사용합니다. 

$m=1$은 가정에서 얻어지니, induction step만 고민합시다. 핵심 아이디어는 

$$\otimes_{i=0}^{m-1} (1 - r_i, r_i) \cdot [u_0, \cdots, u_{2^m - 1}]^T = \otimes_{i=0}^{m-2} (1-r_i, r_i) \cdot ((1-r_{m-1}) \cdot U_0 + r_{m-1} U_1)$$

라는 것입니다. 물론, $U_0 = [u_0, \cdots, u_{2^{m-1}-1}]$, $U_1 = [u_{2^{m-1}}, \cdots, u_{2^m-1}]$.

이제,

$$p(r_{m-1}) = \text{Pr}(d(\otimes_{i=0}^{m-2} (1-r_i, r_i) \cdot ((1-r_{m-1}) \cdot U_0 + r_{m-1} U_1), C) \le e)$$

$$R^\star = \{r_{m-1} : p(r_{m-1}) > (m-1) \cdot \epsilon / q\}$$

이라 합시다. 귀납법에 따라, $R^\star$에 있는 $r_{m-1}$에 대해서 

$$d^{2^{m-1}}((1 - r_{m-1}) \cdot U_0 + r_{m-1} \cdot U_1, C^{2^{m-1}}) \le e$$

를 얻을 수 있습니다. 이제 $\lvert R^\star \rvert > \epsilon$만 보이면 $C^{2^{m-1}}$ 위에서의 proximity gap over affine line을 사용하여, $U$의 correlated agreement를 증명할 수 있습니다. 그런데 이는 전형적인 확률 계산으로, 

$$m \cdot \frac{\epsilon}{q} < \frac{\lvert R^\star \rvert}{q} \cdot 1 + \frac{q - \lvert R^\star \rvert}{q} \cdot (m-1) \cdot \frac{\epsilon}{q} \le \frac{\lvert R^\star \rvert}{q} + (m-1)\cdot \frac{\epsilon}{q}$$

가 되어 $\lvert R^\star \rvert > \epsilon$이 증명됩니다. 

나아가서, 이는 Reed-Solomon code의 tensor-style proximity gap을 증명합니다. 

# Counterexamples on $e < d/2$

**Counterexample from 0xPARC**: $e = d/3$에서 $\epsilon = e+1$에 대한 반례를 찾습니다. 즉, affine line에서 $e+2$개의 점이 $C$와 $e$-close 한 경우에도, affine line 전체가 $C$와 $e$-close 하지 않을 수 있음을 보입니다. 

$k = e+2$라고 하고 $n = ke + 2d$라고 합시다. 이때, vector $a_0, a_1, \cdots, a_{k-1}, b, c$를 다음과 같이 정의합니다.
- $a_i$: index $[ie, (i+1)e)$에서 1, 그 외에는 0
- $b$: index $[ke, ke+d)$에서 1, 그 외에는 0
- $c$: index $[ke+d, ke+2d=n)$에서 1, 그 외에는 0

또한, $i \in \mathbb{F}_q$에 대해서 $u_i = b + ic$라고 정의합시다. 이는 하나의 affine line입니다. 

마지막으로, $0 \le i < k$에 대해서 $v_i = a_i + u_i = a_i + b + ic$라고 정의하고, 

$$C = \text{span}\{v_0, \cdots, v_{k-1}\}$$

으로 정의합시다. $q$는 충분히 크다고 합시다. 이제 이 $C$와 affine line이 반례가 됨을 보입시다. 

먼저 $C$가 distance $d$를 가짐을 보여야 합니다. $v$의 선형결합의 hamming weight가 $d$ 미만이 되려면, $b$의 계수도 0이 되어야 하고 $c$의 계수도 0이 되어야 합니다. 그런데, 이를 위해서는 $v_i$의 계수가 non-zero인 게 3개 이상은 되어야 하고, $v_i$ 한 개 당 $e$개의 서로 다른 hamming weight가 발생하고 $3e \ge d$가 성립하여 증명이 완료됩니다.  

자명하게, $0 \le i < k$에 대해서 $u_i$는 $C$와 $e$-close 합니다. $v_i$가 $C$에 속하는 close codeword가 됩니다. 

이제 $k \le i < q$에 대해서 $u_i$가 $C$와 $2e$-far 함을 보이면 증명이 완료됩니다. 역시, $b, c$의 계수를 맞춰주지 않으면 바로 $d$-far 하게 되므로, 이를 맞춰야 합니다. 그런데 이를 위해서는 $v_i$의 계수가 최소 2개는 non-zero여야 하고, 그러면 $a_i$ 쪽 부분 때문에 $u_i$와 거리가 $2e$ 이상이 됩니다. 증명 끝. 0xPARC에서 꾸준히 멋진 결과를 냅니다.

**Counterexample from 2024/1351**: $e = \lfloor (d-1)/2 \rfloor$에서 $\epsilon \ge n$이 필요함을 보입니다. 

Reed-Solomon code를 사용합니다. degree 1에 대한 Reed-Solomon code $C$를 사용하면 $d = n-1$이 되며, $n$을 짝수로 두면 $e = n/2-1$에 대해서 문제를 풀면됩니다. 이제 

$$u_0 = (0, 0, 2, 3, 4, \cdots, e+1, 0, 0, \cdots, 0)$$

$$u_1 = (0, 0, 0, 0, \cdots, -(e+1), -(e+2), \cdots, -2e)$$

라고 하면, 자명하게 $d(u_0, 0) \le e$, $d(u_1, 0) \le e$이므로 $u_0, u_1$은 각각 $e$-close 하고, $d((u_i)_{i=0}^1, C^2) > e$가 성립하는 것 역시 자명합니다. unique decoding 안이니, 결국 $0$을 가까운 codeword로 사용해야 하기 때문입니다.

이제 각 $0 \le r < n$에 대해서 $(1-r)u_0 + ru_1$이 $C$와 $e$-close 함을 보일 수 있습니다. $r = 0, 1$은 자명하고, $2 \le r \le e + 1$인 경우, $f(x) = r(1-x)$을 사용하면 $x = 1, e+2, e+3, \cdots, 2e+1$에서는 $(1-r)u_0 + ru_1$과 같음을 알 수 있으며, 추가적으로 $x = r$인 경우에도 $f(r) = r(1-r)$이고 $(1-r)u_0 + ru_1$의 값도 $r(1-r)$이 되어 $x = r$에서도 agreement가 발생함을 알 수 있습니다. 그러므로, $(1-r)u_0 + ru_1$는 $C$와 $e$-close 하게 되고, 이 논리는 $e+2 \le r < n$에서도 비슷하게 가능합니다. 이제 $\epsilon \ge n$이 필요함을 알 수 있습니다. 


