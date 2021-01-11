---
layout: post
title: "Dynamic MSF with Subpolynomial Worst-case Update Time (Part 2)"
author: koosaga
date: 2021-01-11
tags: [graph-theory, theoretical-computer-science, data-structure]
---

# Dynamic MSF with Subpolynomial Worst-case Update Time (Part 2)

## Chapter 3. Continued

**Proof of Lemma 3.4: The Algorithm.** $G_0, \alpha_0, l$ 을 Lemma 3.4의 입력이라고 하자. 알고리즘은 $l + 1$ 개의 *레벨* 로 그래프를 관리한다. 각 레벨은 간선의 부분집합을 관리하며, one-shot expander pruning algorithm을 호출한다. 레벨이 깊을 수록 (숫자가 클 수록) one-shot expander pruning algorithm의 호출 횟수는 많아지며, 반대로 간선의 개수는 적어진다. 정확히 어떠한 원리인지는 후술하고, 아래 필요한 정의를 나열한다.

* $\delta = \frac{2}{l}$ 이다.
* $n, \delta, \alpha$ 에 대해서 $f_{n, \delta}(\alpha) = (c_0 \alpha)^{2/\delta}$ 라고 정의하자. $\alpha_0$ 이 주어졌을 때, $\alpha_i = f_{n, \delta}(\alpha_{i - 1})$ 로 정의한다. 이 때 $c_0$ 은 어떠한 상수이다. 이 때 $\alpha_i = \Omega((c_0 \alpha_0)^{l^i})$ 가 대략 성립함을 알 수 있다. 이는 Theorem 5.2에서 정의된 conductance guarantee로 사용될 것이다. 
* 어떠한 입력 $G^\prime = (V^\prime, E^\prime), D^\prime, \alpha^\prime, \delta$ 에 대해서, $(X^\prime, P^\prime) = Prune_{\alpha^\prime}(G^\prime, D^\prime)$ 을 One-shot expander pruning의 output으로 정의하자. $P^\prime$ 는 pruned set이고 (즉 제거할 정점들) $X^\prime = G^\prime[V^\prime - P^\prime]$ 은 pruned set을 제거한 후의 그래프이다. 

* 어떠한 $U \subseteq V$ 에 대해 $X = G_{\tau}[U]$ 이며 $\phi(X) \geq \alpha$ 면 $X$ 를 induced $\alpha$-expander from time $\tau$ 라고 부른다.
* 시간 구간 $[l, r]$ 에 대해서 $D_{[l, r]} \subseteq E$ 를 해당 시간 구간 동안 지워지는 간선의 집합이라고 하자.

위 정의에 의해 다음 사실이 성립한다. 증명은 Theorem 5.2와 위 정의를 조합하면 된다.

**Fact 3.5**. $X$ 가 induced $\alpha_i$-expander from time $\tau$ 이고, $(X^\prime, P^\prime) = Prune_{\alpha_i}(X, D_{[\tau + 1, \tau^\prime]})$ 일 경우, $X^\prime$ 은 induced $\alpha_{i +1}$ expander from time $\tau^\prime$ 이다. 

Pruning set $P$ 를 관리하기 위해서 각 level 에서 그래프 $X^i$ 와 pruning set $P^i$ 를 관리한다. $X_{\tau}^{i}, P_{\tau}^{i}$ 를 time $\tau$ 에서의 $X^i, P^i$ 라고 정의하자. 각 레벨에 대해서 초기에 $X_0^i = G_0, P_0^i = \emptyset$ 으로 설정한다. 이후 $d_i = n^{1 - i/l}$ 의 시간마다 우리는 이 집합들을 업데이트할 것이다. 이 때 $d_l = 1$ 임을 기억하자. 즉, 마지막 level은 매 쿼리마다 업데이트한다. 이 내용을 다시 정리하면 Fact 3.6이 된다:

**Fact 3.6**. 임의의 정수 $k \geq 0$ 과 time $\tau \in [kd_i, (k+1)d_i]$ 에 대해 $X_{\tau}^i = X_{kd_i}^{i}, P_{\tau}^{i} = P_{kd_i}^{i}$

이제 알고리즘을 서술할 준비가 되었다.

**Initialization:** $X^0 = G_0, P_0 = \emptyset$, 모든 $1 \le i \le l$ 에 대해 $d_i = n^{1 - i/l}, X_0^i = G_0, P_0^i = \emptyset$

**Algorithm:**

* 각 Level $1 \le i \le l$, 그리고 모든 숫자 $k_i \geq 0$ 에 대해, 구간 $[k_id_i + 1, (k_i + 1)d_i]$ 동안 다음 계산을 일정한 비율로 한다. (즉, 아래 계산의 연산량을 시간 단위로 나눠서, 매 업데이트마다 아래 계산 중 정확히 $1/L$ 만큼의 시간이 걸리는 만큼만 수행한다.) 
  * $k_{i - 1}$ 을 $\lfloor \frac{k_i d_i}{d_{i - 1}} \rfloor$ 라고 하자. 
  * $(X_{(k_i+1)d_i}^{i}, P_{(k_i+1)d_i}^{i}) = Prune_{\alpha_{i - 1}}(X_{k_{i-1}d_{i-1}}^{i-1}, D_{[max(1, (k_{i-1} - 1)d_{i-1} + 1), k_id_i]})$ 로 설정한다.
    * 이 과정에서 $\phi(G_b)< \alpha_b$ 가 되면 $\phi(G_0) < \alpha_0$ 임을 보고하고 종료한다.
  * $P^{i}_{(k_i+1)d_i}$ 를 pruning set $P$ 에 넣는다.
* 시간 $\tau$ 에, level $l + 1$에 대해:
  * $(X_{\tau}^{l + 1}, P_{\tau}^{l + 1}) = Prune_{\alpha_l}(X_{\tau}^{l}, D_{[\tau, \tau]})$ 로 설정한다.
  * $P_{\tau}^{l + 1}$ 을 pruning set $P$ 에 넣는다.

이제 이 알고리즘의 정당성과 시간 복잡도를 분석하자.

**Lemma 3.7**. $\phi(G_0) \geq \alpha_0$ 일 때, 모든 $1 \le i \le l$과 $k_i$ 에 대해 $X_{k_id_i}^i$ 는 time $max(0, (k_i-1)d_i)$ 에 대한 induced $\alpha_i$-expander 이다.

**Proof.** $k_i = 0$ 일 경우 자명하다. $k_i > 0$ 일 경우, $i$ 에 대한 귀납법을 사용한다. $i = 0$ 일 때 역시 자명하다. 그 외 경우, $X^{i-1}_{k_{i-1}d_{i-1}}$ 는 귀납 가정에 의해 time $(k_{i-1} - 1)d_{i-1}$ 에 대한 induced $\alpha_{i-1}$-expander 이다. 이제 Fact 3.5에 의해 $X^{i}_{(k_i + 1)d_i}$ 는 시간 $k_i d_i$ 에 대한 induced $\alpha_i$-expander 이고 이를 $d_i$ 만큼 예전으로 돌리면 증명이 완료된다.

**Lemma 3.8**. $\phi(G_0) \geq \alpha_0$ 일 때, 임의의 시간 $\tau$ 에 대해 $X_{\tau}^{l + 1}$ 은 시간 $\tau$ 에 대한 induced $\alpha_{l+1}$-expander 이다. (증명은 3.7과 거의 동일하여 생략)

**Lemma 3.9**. $\phi(G_0) \geq \alpha_0$ 일 경우 알고리즘은 절대 $\phi(G_0) < \alpha_0$ 을 보고하지 않는다.

**Proof.** $\phi(G_0) < \alpha_0$ 이 보고되었다는 것은 어떠한 $i, j$에 대해서 $Prune_{\alpha_i}(X_j^i, \cdot)$ 함수가 $\phi(X_j^i) < \alpha_i$ 를 보고했다는 것이다. 만약 $\phi(G_0) \geq \alpha_0$ 일 경우, 위 Lemma들에 의해서 $X_j^i$ 는 임의의 시간에 대해 induced $\alpha_i$-expander 이다. 고로 $\phi(X_j^i) \geq \alpha_i$ 이 성립한다.

**Proposition 3.10**. 모든 $1 \le i \le l + 1$ 과 $k_i$ 에 대해 $P_{k_id_i}^{i} \subseteq P_{k_id_i}$ 이다. (Pruning set의 구성에 의해 자명하다.)

**Lemma 3.11**. 임의의 $\tau$에 대해 $V - V(X_{\tau}^{l+1}) \subseteq P_{\tau}$ 이다.

**Proof.** 임의의 $\tau$ 에 대해 $V - P_{\tau} \subseteq V(X_{\tau}^{l + 1})$ 임을 보이면 된다. 시간 $\tau$ 에 $k_l = \tau$ 이다. 나머지는 위처럼 귀납적로 정의된다. 알고리즘에 따라서, 다음과 같은 식이 성립함을 알 수 있다:

$V(X_{\tau}^{l + 1}) = V(X_{k_ld_l}^{l}) - P_{\tau}^{l + 1}$

$= V(X_{(k_{l - 1} - 1)d_{l-1}}^{l-1}) - P_{k_ld_l}^{l} - P_{\tau}^{l + 1}$

$= V(X_{(k_{l - 2} - 2)d_{l-2}}^{l-2}) - P_{(k_{l - 1} - 1)d_{l - 1}}^{l - 1} - P_{k_ld_l}^{l} - P_{\tau}^{l + 1}$

$= V(X_0) - \bigcup_{0 < i \le l}(P_{(k_{i} - l + i)d_i}^i) - P_{\tau}^{l + 1}$

이 때 귀납적 정의에 따라 $(k_i - l + i) d_i \le k_l d_l$ 이다. 이를 Proposition 3.10과 조합하면 $P_{(k_i - l + i) d_i}^{i} \subseteq P_{k_ld_l}^{i} \subseteq P_{k_l d_l} \subseteq P_{\tau}$ 이다. 이 사실을 사용하여 정리하면

$V - P_{\tau} = V(X_0) - P_{\tau} \subseteq V(X_{\tau}^{l + 1})$

고로 증명이 완료된다.

이로써 알고리즘의 정당성의 증명을 완료할 수 있다.

**Corollary 3.12**. $\tau$ 번째 업데이트 이후, 알고리즘은 $\phi(G_0) < \alpha_0$ 임을 보고하고 종료하거나(1), pruning set $P_{\tau - 1}$ 을 $P_{\tau - 1} \subseteq P_{\tau}\subseteq V$ 인 $P_{\tau}$ 로 업데이트한다(2). 이 때 모든 $\tau$ 에 대해서 $W_{\tau} \subseteq P_{\tau}$ 이며 $G_{\tau}[V - W_{\tau}]$ 가 연결된 집합이 존재한다.

**Proof.** (1) 은 Lemma 3.9에 의해 증명된다. (2) 는 알고리즘의 정의에 의해 자명하다. $W_{\tau} = V - V(X_{\tau}^{l + 1})$ 이라고 하자. $W_{\tau}$ 는 $P_{\tau}$의 부분집합이다. $V - W_{\tau} = X_{\tau}^{l + 1}$ 인데, 이는 Lemma 3.8에 의해 0 초과의 conductance를 가진다. 고로 $W_{\tau}$ 는 연결되어 있다.

**Lemma 3.13**. 매 업데이트마다 위 알고리즘은 $O(l^2 \Delta n^{O(1 / l + \epsilon l^l)})$ 시간을 소모한다.

**Proof.** Theorem 3.2에서 One-shot expander pruning 알고리즘이 Pruning set을 *출력* 한다고 정의했음을 기억하자. 고로 위 알고리즘에서 집합에 $P$ 를 넣는 것은 $Prune$ 함수의 호출 시간에 지배된다. 즉, 알고리즘의 수행 시간은 온전히 $Prune$ 함수의 수행 시간에 좌우된다. Level $i$ 에서 소모하는 계산량은 $\overline{t_i}/d_i = \tilde{O}(\frac{\DeltaD^{1 + \delta}}{\delta \alpha_b^{6 + \delta}d_i})$ 이다. $d_i = n^{1 - i/l}, D = d_{i - 1}, \delta = \frac{2}{l}, \alpha_b = \alpha_{i-1}$ 이다. 이 때 $\frac{d_{i - 1}^{1 + \delta}}{d_i} \le d_{i-1}^{\delta} n^{1/l} \le n^{3/l}$ 이다. 고로 $\overline{t_i}/d_i = \tilde{O}(\frac{\Delta n^{3/l}}{\delta \alpha_{i-1}^{6 + \delta}})$ 이다. 이를 모든 $i$ 에 대해서 더하면

$\sum_{1 \le i \le l + 1}\tilde{O}(\frac{\Delta n^{3/l}}{\delta \alpha_{i-1}^{6 + \delta}}) = \tilde{O}(l^2\frac{\Delta n^{3/l}}{\alpha_{i-1}^{8}}) $ ($\delta = 2/l$, $\alpha \le 1$, $l \geq 1$)

$= \tilde{O}(l^2\frac{\Delta n^{3/l}}{((c_0\alpha_0)^{l^l})^{8}})$

$=\tilde{O}(l^2 \Delta n^{O(1/l)} \times n^{O(\epsilon l^l)})$ ($\alpha_0 = O(1/\epsilon^n)), \alpha_i = \Omega((c_0 \alpha_0)^{l^l})$ 

$=\tilde{O}(l^2 \Delta n^{O(1/l + \epsilon l^l)})$

**Finishing the proof of Lemma 3.4.** Corollary 3.12와 Lemma 3.13에 의하여 증명이 종료된다.

## Chapter 4. Reduction from Graphs with Few Non-tree Edges undergoing Batch Insertions

Chapter 4의 Main Theorem은 다음과 같다.

**Theorem 4.1.** 다음과 같은 성질을 가진, Las-Vegas Decremental MSF Algorithm $D$ 가 있다고 가정하자.

* 그래프에는 $m^\prime$ 개의 간선이 있으며 최대 차수가 3이다.
* $T(m^\prime)$ 길이의 간선 삭제를 수행한다.
* 전처리 시간이 $t_{pre}(m^\prime, p)$ 이며 Worst-case 업데이트 시간이 $1- p$ 확률로 $t_u(m^\prime, p)$ 이다.

이 때, $15k \le m^\prime$ 을 만족하는 임의의 정수 $B, k$ 에 대해, 다음과 같은 성질을 가진, Las-Vegas Fully dynamic MSF algorithm $F$ 가 존재한다.

* 그래프에는 $m$ 개의 간선이 있다.
* 최대 $k$ 개의 간선이 non-tree edge이다.
* 전처리 시간이 $t^\prime_{pre}(m,k,B,p) = t_{pre}(15k, p^\prime) + O(m \log^2 m)$ 이다.
* $B$ 개의 간선 추가, 혹은 1개의 간선 제거를 하는 데 드는 시간이 $t_u^\prime(m, k, B, p) = O(\frac{B \log k}{k} \times t_{pre}(15k, p^\prime) + B\log^2 m + \frac{k \log k}{T(k)} + \log k \times t_u(15k, p^\prime))$ 이다.

이 때 $p^\prime = \Theta(p / \log k)$ 이며, 각  시간 복잡도는 $1 - p$ 확률로 만족된다.

이러한 형태의 Main Theorem은 Fully dynamic MSF algorithm을 얻기 위한 일반적인 전략에 속한다. 이러한 아이디어가 되는 이유를 짧게 설명하면, Incremental MSF는 적절한 자료구조 (Top tree, Link cut tree) 를 사용하면 쉽기 때문이다. Fully dynamic MSF에 대한 Amortized $O(\log^4 m)$ 의 바운드를 얻어낸 Holm et.al 의 논문 역시 총 $O(m \log^2 n)$ 시간에 작동하는 Decremental MSF Algorithm을 얻은 후 Fully Dynamic MSF를 이에 Reduction하는 식으로 해당 결과를 얻었다. 이 Theorem의 증명 역시 조상을 타고 올라가면 해당 논문의 Theorem 7에 기반하여 있다.

#### 4.1. Reduction to Decremental Algorithms for Few Non-tree Edges

이제 Main Theorem의 증명을 시작한다. 일단 첫번째 단계로, 위에서 소개한 알고리즘 $F$ 를, non-tree edge 제약 조건이 있는 Decremental MSF Algorithm으로 reduction한다.

**Lemma 4.2.** 다음과 같은 성질을 가진, Las-Vegas Decremental MSF Algorithm $Dfn$ 이 있다고 가정하자.

* 그래프에는 $m^\prime$ 개의 간선이 있다.
* 최대 $k$ 개의 간선이 non-tree edge이다.
* 전처리 시간이 $t_{pre}(m, k, p)$ 이며 업데이트 시간이 $t_u(m, k, p)$ 이다.

이 때 $B \geq 5 \lceil \log k \rceil$ 을 만족하는 임의의 정수 $B$ 에 대해, 다음과 같은 성질을 가진, Las-Vegas Fully dynamic MSF algorithm $Ffn$ 이 존재한다.

* 전처리 시간이 $t^\prime_{pre}(m, k, B, p) = T_{pre}(m, k, p^\prime) + O(m \log m)$ 이다.
* $B$ 개의 간선 추가, 혹은 1개의 간선 제거를 하는 데 드는 시간이 $t_u^\prime(m, k, B, p) = O(\sum_{i = 0}^{\lceil \log k \rceil} t_{pre}(m, min(2^{i + 1}B, k), p^\prime) / 2^i + B \log m + \log k \times t_u(m, k, p^\prime))$ 이다.

이 때 $p^\prime = O(p / \log k)$ 이며, 각  시간 복잡도는 $1 - p$ 확률로 만족된다.

#### 4.1.1 Preprocessing

$F_1$ 의 입력을 $G = (V, E)$ 라고 하자. $F = MSF(G), N = E - F$ 라고 하면, $E \le m, N \le k$ 가 매 스텝마다 보장된다.

$L = \lceil \log k \rceil$ 이라 두고, 충분히 큰 상수 $c_0$ 에 대해 $p^\prime = \frac{p}{c_0 L}$ 라 하자. 이 알고리즘에서 우리는 $0 \le i \le L, 1 \le j \le 4$ 에 대해서 $G$ 의 서브그래프 $G_{i, j}$ 를 관리한다. 이에 더해 $G_{L, 0}$ 이라는 서브그래프 역시 정의한다. $N_{i, j} = E(G_{i, j}) - MSF(G_{i, j})$ 라고 하자. 알고리즘은 $N = \bigcup_{i, j} N_{i, j}$, $N_{i, j} \le min(2^iB, k)$ 라는 invariant를 유지한다.

$D_{i, j}$ 를 $MSF(G_{i, j})$ 를 관리하는 $Dfn$ 알고리즘의 인스턴스라고 하자. 초기에, 우리는 $G_{L, 1} = G, G_{i, j} = \emptyset$ 으로 둔다. 초기화 과정에서 우리는 단순히 $D_{L, 1}$ 을 전처리하고, $F$ 의 Top tree $T(F)$ 를 구성한다.

각 레벨의 개략적인 역할과, 4개의 서브그래프가 있는 이유는 다음과 같다. (여기서 서브그래프는 $D_{i, j}$ 를 관리하는 $Dfn$ 알고리즘을 의미한다.) 일단 각 레벨은 $2^i$ 시간의 주기에 걸쳐서 특정한 작업을 하는 프로세스라고 불 수 있다. 1번과 2번 서브그래프는 *큐* 의 역할으로, 여기서는 실제 처리가 일어나지 않고 처리할 간선을 쌓아둔다. $2^i$ 시간마다, 하나의 그래프가 쌓이게 된다. 3번과 4번 서브그래프는 실제 처리가 일어나는 곳으로, 처음에 1번과 2번 서브그래프를 그대로 가져온 후 실제 계산을 수행하게 된다. 이 때 계산된 결과는 $i+1$ 번 레벨의 1번, 2번 서브그래프에 쌓이게 된다. $i+1$ 번 레벨은 $2^{i + 1}$ 시간마다 $i$ 번 레벨의 1번과 2번 서브그래프를 3번과 4번 서브그래프에 옮긴다. 이는 서브그래프가 2번 쌓일 시간이니, 2개의 서브그래프가 필요한 것이다. 

또한, "옮기는" 연산은 포인터만 스왑하면 되기 때문에 $O(1)$ 시간이 소요된다. 3번과 4번 서브그래프가 같은 방식으로 처리되는데도 불구하고 따로 저장 공간을 쓰는 이유는, $i$ 번 레벨은 $O(1)$ 에 옮기기만 할 것이고 합치는 것은 $i+1$ 번 레벨로 올라가면서 알아서 할 것이기 때문이다.

#### 4.1.2 Update

업데이트는 여러 간선을 추가하는 것과 하나의 간선을 제거하는 두 종류로 나뉜다. 이 알고리즘에서는 두 업데이트를 처리한 후 *clean-up*이라고 불리는 후처리 과정을 공통적으로 진행할 것이다.

* **간선 추가**: $I$ 를 삽입할 간선의 집합이라고 하고, $R$ 을 clean-up의 대상이 되는 간선의 집합이라고 하자. 모든 간선 $e = (u, v) \in I$ 에 대해서, $u$ 와 $v$ 가 $F$ 에서 연결되어 있지 않다면 $F \leftarrow F \cup \{e\}$ 로 둔다. 만약 연결되어 있다면, Top-tree를 통해 $u, v$ 사이의 유일한 경로 중 가중치가 가장 큰 간선 $f$ 를 $O(\log N)$ 에 찾자. $w(e) > w(f)$ 면 $R \leftarrow R \cup \{e\}$ 로 둔다. $w(e) < w(f)$ 면 $F \leftarrow F + \{e\} - \{f\}, R \leftarrow R \cup \{f\}$ 로 둔다. $R \le I \le B$ 임을 관찰하자. 이후 clean-up을 진행한다.

* **간선 제거**: 모든 $i, j$ 에 대해 $G_{i, j}$에서 $e$ 를 제거한다. 이 과정에서 $e$ 가 제거되면서 새롭게 스패닝 포레스트 $MSF(G_{i, j})$ 에 들어가게 된 간선들의 집합을 $R_0$ 이라고 하자. 이 중 $F - \{e\}$ 의 나눠진 컴포넌트를 연결하는 가장 가중치가 작은 에지를 $f$ 라고 하자. $f$ 가 존재한다면, $F \leftarrow F - \{e\} + \{f\}, R \leftarrow R_0 - \{f\}$ 로 둔다. 아니면 $F \leftarrow F - \{e\}, R \leftarrow R_0$ 으로 둔다. $R \le 4L + 5$ 임을 관찰하자. 이후 clean-up을 진행한다.

* **Clean up:** $R$ 을 clean-up의 대상이 되는 간선 집합이라고 하자. 또한, $R^\prime$ 이라는 간선 집합을 정의하자 (무엇인지는 후술한다). 모든 $0 \le i \le L + 1$에 대해 *level $i$ 의 clean-up* 을 진행한다. 각 $i$ 에 대한 clean-up 과정은 다음과 같다.

  * $i = 0$ 일 때는 새로운 Decremental MSF Algorithm $Dfn$ 의 인스턴스 $D^\prime_{0}$ 을 정의한다. 입력으로 주어지는 그래프는 $G_0^\prime = (V, F \cup R \cup R^\prime)$ 이다. 우리는 $D_{0, j} = \emptyset$ 인 어떠한 $j \in \{1, 2\}$ 에 대해 $D_{0, j} \leftarrow D_0^\prime$ 를 대입한다. $D$ 를 적용하는 모든 과정은 상수 시간이 걸린다고 생각해도 좋다. 실제 구조가 아닌 포인터를 움직이기 때문이다.

  * $i > 0$ 일 때는, $[k \times 2^i, (k+1) \times 2^i]$ 의 시간 구간에 대해서 논의를 진행한다. 이 구간 동안 level $i$ 에서는 $D_i^\prime$ 을 초기화하고, 구간이 끝날 때 (즉 $\tau$ 가 $2^i$ 의 배수일 때) 초기화를 완료할 것이다. $D_i^\prime$ 의 내용은 후술한다. 구간이 끝날 때 (즉 $\tau$ 가 $2^i$ 의 배수일 때), 우리는 $0 \le i \le L$ 에 대해 $D_{i, j} = \emptyset$ 인 어떠한 $j \in \{1, 2\}$ 에 대해 $D_{i, j} \leftarrow D_i^\prime$ 를 대입한다. $i = L + 1$ 일 때는 $D_{L, 0} \leftarrow D_i^\prime$ 을 대입한다. 그리고, 모든 $1 \le i \le L + 1$ 에 대해 $(D_{i-1, 3}, D_{i-1, 4}) \leftarrow (D_{i-1, 1}, D_{i-1, 2})$ 를 설정하고, $(D_{i-1, 1}, D_{i-1, 2})\leftarrow (\emptyset, \emptyset)$ 을 설정한다. 이것이 $D_{i, 1}, D_{i, 2}$ 가 초기화되고 비워지는 과정의 전부이다.

    이제 $D_{i-1, 3}, D_{i-1, 4}$ 로부터 $D_i^\prime$ 을 만드는 과정을 서술한다. $G_i^\prime = (V, F \cup N_i^\prime), N_i^\prime = N_{i-1, 3} \cup N_{i-1, 4}$ 라고 정의하자. 또한 $N_{L+1}^\prime = N_{L, 0} \cup N_{L, 3} \cup N_{L, 4}$ 이다. $D_i^\prime$ 은 $G_i^\prime$ 으로부터 생성되며, 그 과정은 다음과 같다. 시간 구간을 $[\tau, \tau + 2^i)$ 라고 하자. 이를 $I_1 = [\tau, \tau + 2^{i-1}), I_2 = [\tau + 2^{i-1}, \tau + 2^i)$ 로 나눈다. $I_1$ 동안은 $G_i^\prime$ 을 사용하여 $D_i^\prime$ 을 초기화하는 작업을 진행하는데, $2^{i-1}$ 시간동안 동일한 속도로 진행한다. 앞에서와 마찬가지로, 매 순간마다 $X / 2^{i-1}$ 횟수 만큼의 연산을 나눠서 하는 것이다. 이렇게 하여 $D_i^\prime$ 은 $\tau$ 시간의 정보를 완전히 반영하는 상태까지가 된다. $I_2$ 동안은 $D_i^\prime$ 의 정보를 $\tau$ 시간에서 $\tau + 2^i$ 시간으로 업데이트 해준다. 이는 그동안 진행된 업데이트를 *2배속* 으로 돌리는 식으로 처리하면 된다: 예를 들어, $\tau + 2^{i-1} + k$ 시간에는 $\tau + 2k, \tau + 2k+1$ 시간의 업데이트를 반영해 주면 된다. 이 과정에서 간선 추가 업데이트는 물론 무시한다.

  최종적으로 우리는 $R^\prime$ 을 정의할 수 있다. 임의의 시간 $\tau$ 에 대해서, *2배속* 으로 돌린 업데이트 중 *간선 제거* 업데이트가 반환한 reconnecting edge들의 집합을 우리는 $R^\prime$ 이라고 정의한다. 이 집합은 level 0에 대한 clean-up이 일어나면서 $R^\prime \leftarrow \emptyset$ 으로 초기화된다.

#### 4.1.3 Correctness

**Proposition 4.3.** 모든 $i, j$ 에 대해서 $G_{i, j}$ 는 임의의 시간에 $G$ 의 subgraph이다.

**Proof.** 추가만 있을 때는 새로운 간선이 생기지 않아 자명하다. 삭제가 있을 때는 $G_{i, j}$ 에 바로 삭제가 반영되며, $D^\prime_i$ 도 시간차가 있지만 이후 $G_{i, j}$ 에 write될 때 삭제가 반영된다.

**Proposition 4.4.** 모든 $0 \le i \le L$ 에 대해서 $D_{i, j} \leftarrow D_i^\prime$ 을 설정할 때 $D_{i, j} = \emptyset$ 인 $j \in \{1, 2\}$ 가 존재한다.

**Proof.** $2^{i + 1}$ 의 시간동안 $D_{i, j} \leftarrow D_i^\prime$ 은 2번 일어나고, $(D_{i, 1}, D_{i, 2}) \leftarrow (\emptyset, \emptyset)$ 은 1번 일어난다.

**Lemma 4.5.** $N \subseteq \bigcup_{i, j} N_{i, j}$ 라고 가정하자. $e \in F$ 가 제거될 때 $f^*$ 를 $G$ 에서 $F$ 를 다시 연결하는 가장 가중치가 작은 간선이라고 하자. $f^* \in R_0$ 이다.

**Proof.** $f^* \in N$ 이니 어떠한 $i, j$  에 대해 $f^* \in N_{i, j}$ 이다. $G_{i, j}$ 가 $G$ 의 서브그래프이니, $f^*$ 는 $G_{i, j}$ 에서도 $e$ 에 의해 분리된 두 다른 연결 컴포넌트를 잇는 가장 가중치가 작은 간선이다. 고로 $f^*$ 는 $R^\prime$ 에 속하며 고로 $R_0$ 에도 속한다.

**Lemma 4.6.** $N = \bigcup_{i, j} N_{i, j}$ $F = MSF(G)$ 가 업데이트를 거치며 성립한다.

**Proof.** 첫 전처리 때는 두 명제가 자명히 성립한다. 시간에 대한 귀납법을 사용하여 증명한다. 즉 각 업데이트 전에 위 명제가 성립했다면 업데이트 후에도 성립함을 증명한다.

**Part 1 of Proof.** 먼저 $N \subseteq \bigcup_{i, j} N_{i, j}$ 임을 증명한다. $R$ 이란 집합의 성질을 관찰하자.

* 만약에 업데이트가 간선 추가라면, $R$ 은 새롭게 생기는 non-tree 간선들로 구성되어 있다.
* 만약에 업데이트가 간선 제거라면, $R$ 은 $G_{i, j}$에서 새롭게 tree edge가 된 non-tree 간선들로 구성되어 있다. (즉, $\bigcup_{i, j} N_{i, j}$ 에 속하지 않으나 $N$ 에 속할 가능성이 있는 간선들이다).

0번 레벨에 대한 clean-up이 진행된 이후, $N_{0, j} = R \cup R^\prime$ 이 된다. 고로 간선 추가일 때는 모든 non-tree 간선들이 $\bigcup N_{i, j}$ 에 들어가며, 간선 제거일 때도 non-tree 간선들이 $\bigcup N_{i, j}$ 를 빠져나오지 않는다.

이제 Clean-up 절차를 보자. 우리는 $i$ 번 레벨의 clean-up을 시행한 이후에도 $N \subseteq \bigcup N_{i, j}$ 가 성립함을 보인다. clean-up 과정에서, $N_{i-1, 3}, N_{i-1, 4}$ 는 사라지고, $N_i^\prime$ 은 추가가 된다. 시간 $\tau - 2^i$ 에 $N^\prime_i = N_{i-1, 3} \cup N_{i-1, 4}$ 로 초기화를 했으니, 사라질 수 있는 간선은 $[\tau - 2^i, \tau)$ 시간 동안 $N_{i, j}$ 에 들어간 간선들이다. $N_{i, j}$ 에 간선이 들어가는 경우는 Reconnecting edges 뿐인데, 이 경우들은 모두 $R^\prime$ 에서 처리한다. $i = L + 1$ 일 경우는 비슷하게 처리하면 된다. 

**Part 2 of Proof.** $\bigcup_{i, j} N_{i, j} \subseteq N$ 임을 증명한다. 다음 두 명제를 증명한다.

*  $f$ 를 $N$ 에서 제거하면 $f$ 는 $\bigcup_{i, j} N_{i, j}$ 에서도 제거된다. $f$ 가 $N$ 에서 제거되는 경우는 두 가지가 있다. 첫 번째는 $f$ 를 $G$ 에서 제거하는 경우로, 알고리즘에 의해 모든 $N_{i, j}$ 에 대해서도 $f$ 가 제거된다. 두 번째는 $f$ 가 가장 가중치가 작은 reconnecting edge인 경우다. 이 경우 $f$ 는 모든 서브그래프 $G_{i, j}$ 에 대해서도 Reconnecting edge이거나 $MSF(G_{i, j})$ 에 속한다. 고로 두 경우 모두 명제가 성립한다.
* $f$ 를 $\bigcup_{i, j} N_{i, j}$ 에 추가하면 $f$ 는 $N$ 에도 추가된다. 고정된 $i$ 에 대해서 $f$ 가 $N_{i, j}$ 에 추가되는 케이스는 Clean-up 과정에서 $D_{i, j} \leftarrow D^\prime_i$ 를 할 때 뿐이다. 하지만 $D_i^\prime$ 을 만드는 과정에서 추가적으로 $N_i^\prime$ 에 새로운 간선이 추가되지는 않는다. $D_i^\prime$ 은 Decremental하기 때문에 새로운 non-tree edge를 만들 수 없기 때문이다. 또한, $N$ 에서 간선을 지우면 $N_i^\prime$ 에서도 간선이 지워진다 ($I_2$ 에서 두배속으로 돌리는 과정). 고로 $N_i^\prime$ 은 처음 만들 때 $N$ 에 속하며, 이것이 끝날 때까지 성립하기 때문에, 이를 추가해도 이미 $N$ 에 있는 원소임이 유지된다. 

**Part 3 of Proof.** Part 1/2로 $N = \bigcup_{i, j} N_{i, j}$ 를 증명하였다. 이제 $F = MSF(G)$ 임을 보인다. $N \subseteq \bigcup N_{i, j}$ 이니, Lemma 4.5에 의해서 $f^* \in R_0$ 이다. 고로 간선을 지울 때 위 알고리즘은 lightest reconnecting edge를 찾을 수 있다. 이를 통해 간선 제거 케이스를 증명할 수 있고, 간선 추가 케이스는 자명하다.

#### 4.1.4 Running Time

**Lemma 4.7.** 임의의 $i, j$ 에 대해서 $N_{i, j} \le min(2^{i +1}B, k)$ 이다.

**Proof.** $N_{i, j} \subseteq N$ 이니 $N_{i, j} \le k$ 임은 자명하다. 간선 추가 업데이트에서 $R \le B$ 이며, 간선 제거 업데이트에서 $R \le 4L + 5$ 이다. 또한 Reconnecting 단계에서 $R^\prime \le 2L+2$ 이다. 고로 $R \cup R^\prime \le max(B, 4L+5) + 2L+2 \le 2B$ 가 성립한다. $B \geq 5L$ 이기 때문이다. $N_i$ 가 재귀적으로 형성되는 과정에 의해, level이 하나씩 올라갈 수록 크기가 최대 2배 늘어나서, $N_{0, j} \le 2B$ 이면 $N_{i, j} \le 2^{i+1}B$ 이다. 

**Lemma 4.8.** 전처리 알고리즘은 $t_{pre}(m, k, p^\prime) + O(m \log m)$ 시간을 사용한다.

**Proof.** Top tree를 $O(m \log m)$ 시간에 초기화할 수 있다.

**Lemma 4.9.** 각 업데이트마다 Clean-up에 사용한 시간은 $O(\sum_{i = 0}^{\lceil \log k \rceil} t_{pre}(m, min(2^{i+1}B, k), p^\prime) / 2^i + t_u (m, k, p^\prime)\log k )$ 이다. 이 bound가 성립할 확률은 $1 - p/2$ 이다.

**Proof.** 앞 항은 $I_1$ 에서 새로운 자료구조를 초기화 하는데 드는 시간이고, 뒷 항은 $I_2$ 에서 2배속으로 제거 업데이트를 하는 데 드는 시간이다. $p^\prime = O(p / \log k)$ 이기 때문에, 이것이 모든 레벨에 대해서 성립할 확률은 $1 - p^\prime O(L) = 1 - p/2$ 이다.

**Lemma 4.10.** 각 업데이트마다 사용하는 시간은 $O(\sum_{i = 0}^{\lceil \log k \rceil} t_{pre}(m, min(2^{i+1}B, k), p^\prime) / 2^i +  t_u (m, k, p^\prime)\log k  + B \log m)$ 이다. 이 bound가 성립할 확률은 $1 - p/2$ 이다. 

**Proof.** 삽입에 $O(B \log m)$, 삭제에 $O(t_u(m, k, p^\prime) \log k)$ 시간이 소모된다는 사실을 위 식과 결합하면 이를 유도할 수 있다.

고로 Lemma 4.2의 증명이 종료된다.

