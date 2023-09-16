---
layout: post

title: "Brakedown Overview"

date: 2023-09-16

author: rkm0959

tags: [cryptography, blockchain]
---

이 내용은 https://eprint.iacr.org/2021/1043 의 요약입니다. 

이 논문의 목표는 
- Linear Code를 기반으로 한 Linear-Time PCS를 준비하고
- 이를 Spartan에 적용하여 Linear-Time Field-Agnostic SNARK를 얻는 것입니다. 

Spartan 계열 테크닉이 주목을 받는 시점에서 읽기 좋은 논문이라고 생각됩니다. 

다만, 길이의 문제로 각종 증명들은 생략하도록 하겠습니다. 

# Linear Time Polynomial Commitment 

Multilinear polynomial을 기준으로 생각하면, $g$의 Lagrange basis에서 coefficient를 $u$라고 했을 때 

$$g(r) = \sum_i u_i \cdot \left(\prod_{k} (r_k i_k + (1 - r_k)(1 - i_k)) \right)$$

라고 쓸 수 있고, 대괄호 안에 있는 부분은 tensor product로 나타내기 적합한 상황임을 알 수 있습니다. $g$의 input space를 $\mathbb{F}^{\log n}$이라고 하고 $n = m^2$이라고 하면, 결국 적당한 $q_1, q_2 \in \mathbb{F}^m$이 있어 

$$g(r) = \langle q_1 \otimes q_2, u \rangle$$

이도록 할 수 있습니다. 

이를 위해서, $u$를 $m \times m$ 행렬로 보고, $N = \rho^{-1} \cdot m$을 생각한 다음 linear code 

$$\text{Enc}: \mathbb{F}^m \rightarrow \mathbb{F}^N$$

를 생각합니다. 이제 $g$를 commit 하기 위해 $u$를 commit 합니다. 이를 위해서, $\hat{u} = \text{Enc}(u_i)$들을 준비한 다음 이의 merkle hash를 계산합니다. 

이 경우, verifier는 각 $u$의 row가 실제로 codeword에 가까운지 확인해야합니다. 이를 testing phase라 하는데, 이를 위한 전형적인 트릭은 random vector $r \in \mathbb{F}^m$을 잡은 뒤, prover가 $u' = \sum r_i u_i$를 보내도록 하는 것입니다. 그 후, $\text{Enc}(u')$과 $\sum r_i \hat{u}_i$가 동일한지 확인하면 됩니다. 다만, 이를 다 확인할 수는 없으니, $l$개의 entry만 두 값이 일치하는지 확인합니다. 

evaluation을 위한 test는 크게 다르지 않습니다. 

$$g(r) = \langle q_1 \otimes q_2, u \rangle = \langle \sum_i q_{1, i} u_i, q_2 \rangle$$

이므로, $\sum q_{1, i} u_i$만 제대로 계산하면 됩니다. 이에 대한 부분은 testing phase의 트릭과 동일합니다. 

# Linear Time Code 설계하기

Relative distance $1/20$과 rate $3/5$를 갖는 linear time encoding, linear code를 설계해봅시다. 

$\text{Enc}(x) = (x, z, v)$가 되도록 설계하는데, 먼저 sparse 한 $n \times n /5$ 행렬을 $x$에 곱하는 것으로 $y$라는 값을 구합니다. 그 후, $y$를 다시 encode 하여 $z$라고 하는 $n / 3$ 크기의 벡터를 만든 다음, 여기에 sparse 한 $n/3 \times n/3$ 행렬을 곱하여 $v$를 만듭니다. 

이제 rate는 $3/5$임이 자명하고, 해당 encoding이 linear함도 자명합니다. 사용하는 행렬들이 전부 충분히 sparse 하다면, 해당 encoding이 linear-time임도 자명합니다. 문제는 relative distance $1/20$을 유지하면서 sparse한 행렬을 사용하는 것입니다. linear encoding이므로, non-zero codeword의 크기만 bound 하면 됩니다. 이를 다음과 같은 흐름으로 진행합니다.

- $x$의 hamming weight가 $n/12$ 이상이면 끝
- 그게 아닌 경우 $y \neq 0$일 확률이 매우 높음을 증명
- 이렇게 되면, recursive 한 정의에서 $z$의 hamming weight는 $n/60$ 이상
- 문제는 $z$의 hamming weight가 $n/60$과 $n/12$ 사이인 경우고, 이때 $v$의 hamming weight가 $n/12$ 이상

이에 대한 자세한 분석은 https://eprint.iacr.org/2023/769 가 훌륭합니다. 

대강 두 단계로 이루어지는데, 

- 행렬을 bipartite graph로 생각해서, random left-sided $d$-regular graph를 생각합니다. 이때, $b(k)$-expander (즉, $k$ 크기 set의 neighbor는 무조건 $b(k)$ 이상이도록)가 아닐 확률을 구해서 bound 합니다.
- $b(k)$-expander인 경우, 위에서 제시한 조건들이 성립하지 않을 확률을 계산하여 bound 합니다. 그 후 위에서 계산한 확률과 union bound를 해서 총 결과를 계산합니다. 

계산은 상당히 복잡하지만 기본적인 아이디어 자체는 어렵지 않습니다. 위 논문은 분석에 대한 좋은 outline을 주고 parameter 설정에 대한 추가적인 고찰까지 하고 있으니, 이를 참고해봐도 좋을 것 같습니다. 어찌되었건 결론은 linear-time이며 linear한 code가 존재한다는 것입니다. 

# Sparse Polynomials

만약 $D$가 $2 \log M$-variate multilinear polynomial인데, 정작 lagrange coefficient 자체는 $N$개의 지점에서만 non-zero라고 해봅시다. 그러면 

$$D(r_x, r_y) = \sum_{i, j \in \{0, 1\}^{\log M}} D(i, j) \cdot  eq(i, r_x) \cdot eq(j, r_y)$$

가 되는데, summation 자체는 $D(i, j) \neq 0$인 곳에서만 진행해도 됩니다. 

이를 전환하면 

$$D(r_x, r_y) = \sum_{k \in \{0, 1\}^{\log N}} val(k) \cdot eq(b^{-1}(row(k)), r_x) \cdot eq(b^{-1}(col(k)), r_y)$$

로 쓸 수 있습니다. 여기서 $b$는 canonical injection이며, $row, col, val$은 결국 nonzero $D$의 row, col, val이 되겠습니다. $row, col, val$의 경우에는 이에 대응하는 multilinear polynomial을 생각할 수 있습니다. 

이제 $v = D(r_x, r_y)$에 대한 evaluation proof를 진행한다고 합시다. Prover가 

$$E_{rx} = eq(b^{-1}(row(k)), r_x), \quad E_{ry} = eq(b^{-1}(col(k)), r_y)$$

라는 $k$에 대한 multilinear polynomial (extension)을 제공할 수 있다고 하면, 

$$v = \sum_k val(k) \cdot E_{rx}(k) \cdot E_{ry}(k)$$

를 보이면 되며, 이는 sumcheck protocol을 사용하여 할 수 있습니다. 

문제는 $E_{rx}, E_{ry}$가 실제로 원하는 값이 맞는지 확인하는 게 쉽지 않다는 겁니다. 

생각을 해보면 결국 $E_{rx}, E_{ry}$는 $eq(i, r_x)$, $eq(i, r_y)$라는 값을 $b^{-1}(row(k))$, $b^{-1}(col(k))$라는 위치에서 access 한 것과 같다고 볼 수 있습니다. 이를 확인하는 방법으로 "offline memory checking"이라는 기법을 도입합니다. 

$v_i$가 $i$번째 값인 size $M$ table을 읽는다고 생각해봅시다. 현재 "timestamp"를 0이라고 가정하고, table을 $(i, v_i, 0)$의 집합으로 정의합니다. 현재 timestamp가 $ts$라고 하고, 현재 읽는 메모리가 $(a, v, t)$라고 합시다. 이때 $RS$라는 read-state 집합에 $(a, v, t)$를 추가하고, $ts$를 $\max(ts, t) + 1$로 update 한 뒤, $(v, ts)$를 $a$ 위치에 overwrite합니다. 그 후, $WS$라는 write-state 집합에 $(a, v, ts)$를 추가합니다. $WS$는 처음 table과 마찬가지로 $(i, v_i, 0)$으로 초기화되어있습니다. 

이 경우, $WS = RS \cup S$인 cardinality $M$ 집합 $S$가 존재하는 것이 모든 read operation이 올바른 것과 동치임을 증명할 수 있습니다. 그리고 이러한 read, write의 순서는 $rx, ry$와 무관하며 오직 $row, col$과만 관련있음을 알 수 있고, 이는 이 과정이 precompute 가능하다는 것을 의미합니다. 

결국 $E_{rx}$의 정당성을 증명하는 것은 

$$WS = \{(i, eq(i, r_x), 0)\} \cup \{(row(k), E_{rx}(k), \text{write}_{row}(k))\}$$

$$RS = \{(row(k), E_{rx}(k), \text{read}_{row}(k))\}$$

$$S = \{(i, eq(i, r_x), \text{final}_{row}(k))\}$$

일 때 $WS = RS \cup S$임을 확인하는 것과 같습니다. 대강 $\text{write}_{row}(k)$들은 $row(k)$를 지금 언제 봤는지를 기록한다고 생각하면 될 것 같습니다. 두 집합이 같은지 확인하는 것은 각 element를 잘 encode 한 후, 집합 전체의 곱이 동일함을 확인하면 충분합니다. 즉, 

$$\prod_{(a, b, c) \in S} (a \gamma^2 + b \gamma + c - \tau)$$

를 random 한 $\gamma, \tau$에 대해서 계산한 다음 비교해주면 됩니다. 다만 이를 계산하는 것이 빠르게 가능한 일은 아니어서, 추가적인 고민이 필요합니다. 다행히도 이는 쭉 곱을 계산하는 형태의 간단한 모습을 가지고 있어, 어렵지 않습니다. 이진트리처럼 생각한 다음, GKR과 비슷한 느낌의 과정을 거치면 됩니다. 

# Revisiting Spartan 

글을 마치기 전에 빠르게 Spartan을 복기해봅니다. R1CS instance를 생각하고 $Z = (W, 1, io)$를 생각합니다. 여기서 $io$는 public 값이며 $W$가 witness입니다. 편의상 $Z$의 길이가 $M$, $W$의 길이가 $M/2$라고 하면, $Z$의 multilinear extension은 $W$의 multilinear extension과 $(1, io)$의 multilinear extension으로 분해할 수 있습니다. 결국 목표는 $A, B, C$에 대한 multilinear extension을 생각했을 때

$$ T(x) = \left(\sum_y A(x, y) Z(y) \right) \cdot \left(\sum_y B(x, y) Z(y) \right) - \left(\sum_y C(x, y) Z(y) \right) = 0$$

임을 각 $x$에 대해 확인하는 것입니다. 이를 한 번에 확인하기 위해, random 한 $\tau$를 가지고 와서 

$$ \sum_x eq(\tau, x) \cdot T(x) = 0$$

을 확인합니다. 여기서 sumcheck를 돌리면 이는 $eq(\tau, x) \cdot T(x)$를 random point $r_x$에서 evaluation 하는 문제가 됩니다. $eq$ 쪽은 문제가 없고, 문제는 

$$\sum_y A(r_x, y) Z(y)$$

와 $B, C$에서 마찬가지로 필요한 값들인데, 이 역시 sumcheck를 통해서 할 수 있습니다. 필요한 것은 $A, B, C, Z$에 대한 evaluation인데, 이는 이제 PCS를 통해서 할 수 있게 됩니다. 여기에 Fiat-Shamir 휴리스틱까지 쓰면 SNARK가 완성됩니다. 
