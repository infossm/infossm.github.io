---
layout: post

title: "Faster Matroid Intersection - Part 2"

date: 2023-07-24

author: ainta

tags: [matroid]
---

이번 포스트에서는 전 글에서 subquadratic approximation algorithm의 결과를 얻었다고 소개했던 [Faster Matroid Intersection(2019)](https://arxiv.org/pdf/1911.10765.pdf)에 대해 다룬다.


## Preliminaries

전 글과 동일하게, matroid $\mathcal{M_1} = (V, \mathcal{I_1}), \mathcal{M_2} = (V, \mathcal{I_2})$가 주어진 세팅이다. 이제 본격적으로 Matroid Intersection 알고리즘에 들어가기 전에 필요한 성질들에 대해 먼저 알아보자.


**Submodularity of rank**. $A, B \subset V$에 대해, $rank(A \cup B) + rank(A \cap B) \le rank(A) + rank(B)$가 성립한다.

**Proof.**

$rank(A \cup B) - rank(A) \le rank(B) - rank(A \cap B)$를 보이면 충분하다. 이는 $rank(A + (B \setminus A)) - rank(A) \le rank(A\cap B + (B \setminus A)) - rank(A \cap B)$로 쓸  수 있고,
$X \subset Y \subset V, v \subset V \setminus Y$ 에 대해 $rank(Y+v) \le rank(X + v)$ 가 성립함에 따라 증명이 완료된다.$\blacksquare$




**Monotonicity Lemma**. $S \in \mathcal{I}_1 \cap \mathcal{I}_2$ 와 $G(S)$의 shortest $(s,t)$-path $p$에 대해 $p$를 augment해 얻은 $S'$에서 다음이 성립한다.
- $d(s,a) < d(s,t)$ 이면 $d(s,a) \le d'(s,a)$. 마찬가지로, $d(a,t) < d(s,t)$ 이면 $d(a,t) \le d'(a,t)$.
- $d(s,a) \ge d(s,t)$ 이면 $d'(s,a) \ge d(s,t)$. 마찬가지로, $d(a,t) \ge d(s,t)$ 이면 $d'(a,t) \ge d(s,t)$.

이에 대한 증명은 생각보다 까다로우므로 생략한다. (1986년에 Cunningham이 처음 발표한 Monotonicity property의 증명은 틀렸으며 [Cun86], 2015년에 Christopher Price에 의해 올바르게 다시 증명되었다 [Pri15].)


또한, Matroid에서 다음과 같은 사실이 성립한다.

**Fact 1.** Matroid $\mathcal{M} = (V, \mathcal{I})$ 에서 $S \in \mathcal{I}$라 하자. $X \subset S$와 $Y, Z \subset V \setminus S$ 가 다음 조건들을 만족한다고 하자.

-  $Y \cap Z = \phi$
-  $S+Z \in \mathcal{I}$
-  $S-X+Y \in \mathcal{I}$
-  $Y \in span(S)$. That is, $rank(S+Y) = rank(S) = S$.

이 때, $S+Z-X+Y \in I$이다.

**Proof.**

$rank(S-X+Y+Z) + rank(S+Y) \ge rank(S-X+Y) + rank(S+Y+Z)$ (Submodularity of rank). $rank(S+Y) = \lvert S \rvert, rank(S-X+Y) =  \lvert S \rvert - \lvert X \rvert + \lvert Y \rvert , rank(S+Y+Z) \ge rank(S+Z) =  \lvert S \rvert + \lvert Z \rvert$.
따라서, $rank(S-X+Y+Z) \ge  \lvert S \rvert - \lvert X \rvert + \lvert Y \rvert + \lvert S \rvert + \lvert Z \rvert - \lvert S \rvert  =  \lvert S \rvert - \lvert X \rvert + \lvert Y \rvert + \lvert Z \rvert$. $\therefore  S-X+Y+Z \in \mathcal{I}$. $\blacksquare$



## Augmenting Sets


$S \in \mathcal{I_1} \cap \mathcal{I_2}$의 corresponding exchange grpah $G(S)$에 대해, $G(S)$의 shortest $(s,t)$-path의 길이가 $2(l+1)$이라 하자. 


**Definition(Distance Layer)**. $G(S)$의 Distance layer $D_1, D_2, \cdots, D_{2l+1}$ 은 $D_i :=$ ($G(S)$에서 $s$로부터의 최단거리가 $i$인 원소들의 집합) 으로 정의된다.

**Definition(Augmenting Set)**. set들의 collection $\Pi_l := (B_1, A_1, \cdots , A_l, B_{l+1})$가 다음 조건을 만족할 때 $\Pi_l$을 Augmenting Set이라 한다:

- (a) For $1 \le k \le l+1$, we have $A_k \subset D_{2k}$ and $B_k \subset D_{2k-1}$
- (b) $\lvert B_1 \rvert  =  \lvert A_1 \rvert  = \cdots  \lvert B_{l+1} \rvert  = w$
- (c) $S + B_1 \in \mathcal{I_1}, S + B_{l+1} \in \mathcal{I_2}$
- (d) $S - A_k + B_{k+1} \in \mathcal{I_1}$
- (e) $S - A_k + B_{k} \in \mathcal{I_2}$


**Theorem.** $S' = S \oplus \Pi_l = S + B_1 - A_1 + B_2 -  \cdots + B_l - A_l + B_{l+1}$ 은 common independent set. 즉, $S' \in \mathcal{I_1} \cap \mathcal{I_2}$.

이는 Fact 1을 이용해 증명할 수 있다.

**Proof.**

$S+B_1 \in \mathcal{I_1}, S - A_1 + B_2 \in \mathcal{I_1}, rank_1(S + B_2) = rank_1(S)$ 이므로 ($\mathcal{M_1}$ 에서 $S$와 독립인 원소들은 $D_1$에 있으므로 $B_2, \cdots B_{l+1}$엔 들어갈 수 없다) $S + B_1 - A_1 + B_2 \in \mathcal{I_1}$이고, 이를 반복하면 $S' \in \mathcal{I_1}$.

한편, $S + B_{l+1} \in \mathcal{I_2}, S - A_l + B_l \in \mathcal{I_2}, rank_2(S+B_l) = rank_2(S)$ 이므로 (shortest path가 $2(l+1)$이므로 $\mathcal{M_2}$에서 $S$와 독립인 원소들은 $B_{l+1}$에만 들어갈 수 있다)  $S + B_{l+1} - A_l + B_l \in \mathcal{I_2}$이고, 이를 반복하면 $S' \in \mathcal{I_2}. _\blacksquare$


**Theorem**. Maximal augmenting set의 width가 0인 $S$는 $\mathcal{I_1} \cap \mathcal{I_2}$의 maximum cardinality set이다.

**Proof** 
$S+v \in \mathcal{I_1}$인 $v$들의 집합을 $X_1$, $S+v \in \mathcal{I_2}$인 $v$들의 집합을 $X_2$라 하자. Maximal augmenting set의 width가 0이므로, $G(S)$는 augmenting path가 존재하지 않는다. 즉, $X_1$에서 $X_2$로 가는 path가 존재하지 않는다. $X_1$이나 $X_2$가 빈 집합이라면 $S$는 두 매트로이드 중 하나의 base이므로 maximum cardinality set임이 자명하다. $X_1 \neq \phi, X_2 \neq \phi$라 하자. $U$를 $X_2$에 도달할 수 있는 vertex들의 집합이라 하면 $X_1$에서 $X_2$로 가는 path가 없으므로 $X_1 \cap U = \phi$이다. 먼저, $rank_1(U) \le \lvert S \cap U \rvert$, $rank_2(V \setminus U) \le \lvert S \setminus U \rvert$ 임을 보이자.

**claim 1.** $rank_1(U) \le \lvert S \cap U \rvert$

**Proof of claim 1:** $rank_1(U) > \lvert S \cap U \rvert$ 이면 $z \in U \setminus (S \cap U)$인 $z$가 존재하여 $(S \cap U) + z \in \mathcal{I_1}$을 만족한다. 만약 $S + z \in \mathcal{I_1}$이면 $z \in X_1$, $z \in U$, $X_1 \cap U \neq \phi$ 이므로 모순이다. 그렇지 않다면 $S+z \notin \mathcal{I_1}$, $(S \cap U) + z \in \mathcal{I_1}$이므로 $y \in S \setminus U$가 존재하여 $S-y+z \in \mathcal{I_1}$이 성립한다. 그러나 이 경우 $(y, z) \in G(S)$이고, $z \in U$이므로 $y$는 $z$를 거쳐 $X_2$에 도달할 수 있어서 $y \in U$이여 하는데 이는 $y \in S \setminus U$에 모순이다. 
따라서, $rank_1(U) \le \lvert S \cap U \rvert$.

**claim 2.** $rank_2(V \setminus U) \le \lvert S \setminus U \rvert$ 

**Proof of claim 2**: claim 1과 유사하게 증명 가능하다.  $\lvert S \setminus U \rvert < rank_2(V \setminus U)$ 이면 $z \in V \setminus (S \cup U)$ 가 존재하여 $(S \setminus U) + z \in \mathcal{I_2}$이고 $S + z \notin \mathcal{I_2}$ 이므로  $y \in S \cap U$ 존재하여 $S-y+z \in \mathcal{I_2}$. 이 때 $(z, y) \in G(S)$인데 $y \in U$이므로 $z \in U$ 여야 해서 모순.

claim 1, 2에 의해 $\lvert S \rvert = \lvert S \cap U \rvert + \lvert S \setminus U \rvert \ge rank_1(U) + rank_2(V \setminus U)$ 가 성립한다. 
한편, 임의의 $S' \in \mathcal{I_1} \cap \mathcal{I_2}$에 대해 $\lvert S' \rvert = \lvert S' \cap U \rvert +  \lvert S' \setminus U \rvert$인데
$S' \cap U \in \mathcal{I_1}, S' \setminus U \in \mathcal{I_2}$에서 $\lvert S' \cap U \rvert  \le rank_1(U),  \lvert S' \setminus U \rvert  \le rank_2(V \setminus U)$.
따라서, $\lvert S' \rvert  \le rank_1(U) + rank_2(V \setminus U) \le  \lvert S \rvert$ 이고 $S$는 $\mathcal{I_1} \cap \mathcal{I_2}$의 maximum cardinality set이다. $_\blacksquare$


따라서, $S \in \mathcal{I_1} \cap \mathcal{I_2}$가 주어졌을 때, nonempty augmenting set이  존재하지 않을 때까지 width가 0이 아닌 Augmenting Set $\Pi_l$을 찾아 $S$를 $S \oplus \Pi_l$로 업데이트할 수 있다면 해당 과정을 반복하여 $\mathcal{I_1} \cap \mathcal{I_2}$의 maximum cardinality set을 구할 수 있다. 즉, Matroid Intersection Problem을 해결할 수 있다.

**Remark.** Monotonicity Lemma를 생각하면 위 방식대로 Matroid Intersection Problem을 해결하는 동안 $l$은 감소하지 않음을 알 수 있다.

**Note.** Augmenting set은 augmenting path를 일반화시킨 개념이라고 볼 수 있다. 특히, width가 1인 Augmenting set은 하나의 $(s,t)$-path, 즉 augmenting path임을 쉽게 확인할 수 있다.

**관찰.** Common independnet set $S$가 주어졌을 떄, $S$보다 크기가 $w$만큼 큰 common independent set $S'$를 만드는 방법으로는 다음과 같은 두 가지가 있다.
- $G(S)$에서 augmenting path를 하나 찾아 $S$를 $S_1$로 업데이트하고, $G(S_1)$에서 augmenting path를 또 하나 찾아 $S_2$를 만들고 하는 과정을 $w$번 반복한다.
- $G(S)$에서 width $w$인 augmenting set $\Pi_l$를 찾아 $S' = S \oplus \Pi_l$를 얻는다.

augmenting path를 $G(S)$의 shortest $(s,t)$ path로 한정할 떄, 첫 번째처럼 하나씩 업데이트하는 방법과 두 번째처럼 통으로 업데이트하는 방법은 사실 동치가 된다. 이에 대해 곧바로 알아 볼 것이다.

## Equivalence between Consecutive Shortest paths and Augmenting Sets

**Definition(Consecutive Shortest Paths)**. $G(S)$에서 vertex-disjoint shortest $(s, t)$-paths $P_1, \cdots , P_k$가 다음 조건을 만족할 때 이를 **consecutive shortest paths** 라 한다.
- $P_i$는 $G(S \oplus P_1 \oplus P_2 \oplus P_{i-1})$의 shortest augmenting path

즉, consecutive shortest paths의 의미는 $S$에서 구한 disjoint augmenting path $P_1, \cdots P_k$를 $S$에 **순서대로** 적용하여 $S$의 크기를 1씩 늘릴 수 있다는 것이다.


**Lemma 1** Consecutive shortest paths $P_1, \cdots P_k$에 대해 각 $P_i = (s, a_{i,1}, a_{i,2}, \cdots a_{i,2l+1}, t)$라 하자.
- $B_j := \left\\{ a_{1,2j-1}, \cdots,  a_{k,2j-1} \right\\}$ 
- $A_j := \left\\{ a_{1,2j}, \cdots,  a_{k,2j}  \right\\}$ 

로 정의된 $\Pi_l = (B_1, A_1, \cdots , B_{l+1})$은 $G(S)$의 augmenting set이다.


그리고 이 역도 성립한다.

**Lemma 2** $\Pi_l = (B_1, A_1, \cdots , B_{l+1})$가 width가 $k$인 $G(S)$의 augmenting set일 때, $G(S)$의 Consecutive shortest paths $P_1, \cdots P_k$가 존재하여 다음 조건을 만족한다.
- $B_j = \left\\{ a_{1,2j-1}, \cdots,  a_{k,2j-1} \right\\}$ 
- $A_j = \left\\{ a_{1,2j}, \cdots,  a_{k,2j}  \right\\}$ 

즉, $G(S)$의 임의의 augmenting set $\Pi_l$의 width가 $w$일 때, 이를 $G(S)$에서의 $w$개의 disjoint $(s,t)$-path로 나눌 수 있으며 이들은 consecutive shortest paths를 이룬다.

다음은 위 두 Lemma를 증명하는데 사용되는 Lemma이다.

**Lemma 3.** 두 Augmenting set $\Pi, \Pi'$에 대해 $\Pi \subset \Pi'$가 성립할 때, $\Pi' \setminus \Pi$는 $G(S \oplus \Pi)$의 augmenting set이 된다.

위 Lemma들을 통해, 다음 사실을 쉽게 보일 수 있다.


**Lemma 4** $G(S)$의 shortest $(s,t)$-path의 길이가 $2(l+1)$이라 하자. maximal augmenting set $\Pi$에 대해 $G(S \oplus \Pi)$에는 길이 $2(l+1)$의 augmenting path가 존재하지 않는다.

**Proof.**

$G(S \oplus \Pi)$에 길이 $2(l+1)$의 augmenting path $p$가 존재한다고 가정하자. Monotonicity Lemma에 의해, $p$는 $G(S)$에서도 augmenting path여야 한다.

그러나, Lemma 2에 의해 $\Pi$ 에 대응되는 consecutive shortest paths $P_1, P_2, \cdots, P_k$가 존재하고, $G(S)$와 $G(S \oplus \Pi)$에서 모두 augmenting path인 $p$는 $\Pi$와 disjoint하므로 $P_1, P_2, \cdots P_k, p$는 consecutive shortest paths의 조건을 만족한다. 여기에 Lemma 1를 적용하면 $\Pi + p$는 $\Pi$를 strict하게 포함하는 augmenting set이므로 모순. $\blacksquare$


## Partial Augmenting Sets

앞서 Augmenting Set을 구하여 $S$를 업데이트하는 것을 반복하여 Matroid Intersecion Problem을 해결할 수 있음을 보였고, 특히 Maximal Augmenting Set을 구하여 $S$를 업데이트한다면 $G(S)$의 shortest $(s,t)$-path가 증가하므로 $O(n)$번의 업데이트로 답을 구할 수 있음을 보였다.

그러나, Maximal Augmenting Set을 적은 oracle로 구하는 것은 쉽지 않았고, 이에 Partial Augmenting Set의 개념이 도입되게 되었다.

**Definition(Partial Augmenting Set)**. set들의 collection $\Psi_l := (B_1, A_1, \cdots , A_l, B_{l+1})$가 다음 조건을 만족할 때 $\Psi_l$을 Patrial Augmenting Set이라 한다:

- (a) For $1 \le k \le l+1$, we have $A_k \subset D_{2k}$ and $B_k \subset D_{2k-1}$
- (b) $\lvert B_1 \rvert  \ge  \lvert A_1 \rvert \ge \cdots \ge  \lvert B_{l+1} \rvert$
- (c) $S + B_1 \in \mathcal{I_1}, S + B_{l+1} \in \mathcal{I_2}$
- (d) $S - A_k + B_{k+1} \in \mathcal{I_1}$
- (e) for $k \le l, rank_2(S - A_k + B_{k}) = rank_2(S)$



**Note**. 앞서 살펴봤듯 Augmenting set가 source $s$에서 출발하여 sink $t$에 도달하는 $w$개의 shortest disjoint path와 대응된다면, Partial Augmenting Set은 $G(S)$에서 **source에서 출발하는 $\lvert B_1 \rvert$개의 disjoint path** 와 대응된다. Augmenting Set은 **sink에 도달** 해야 하지만 Partial은 그럴 필요가 없음에 주의.

**Note**. Augmenting Set의 정의와 조건 (b), (e)에서만 차이가 있다. (e) 에서 $rank_2(S-A_k+B_k) \ge rank_2(S)$가 아니라 등호인 이유는 $rank_2(S-A_k+B_k) > rank_2(S)$일 수 없기 때문이다. $B_k$의 임의의 원소 $b$에 대해 $S + b \notin I_2$ 이므로 $rank_2(S + B_k) = rank_2(S)$이기 때문에 $rank_2(S-A_k+B_k) > rank_2(S)$ 케이스는 고려할 필요가 없다.


**Lemma 5.** augmenting set $\Pi$, partial augmenting set $\Psi$ 존재하여 $\Pi \subset \Psi$라 하자(이는 $\Pi$와 $\Psi$의 각 $A_k$들과 $B_k$들이 포함관계를 만족한다는 뜻이다). $O(n)$ 번의 oracle로 $\Psi$ 의 $\lvert B_{l+1} \rvert$ 을 width로 하는 augmenting set $\Pi'$ ($\Pi \subset \Pi' \subset \Psi$ ) 를 만들 수 있다.

Proof.

$\Pi'$를 $B_{l+1}, A_l, \cdots B_1$의 순서대로 구성한다. 처음에 $\Pi'$의 $B_{l+1}$을  $\Psi$의 $B_{l+1}$과 동일하게 둔다.

$B_{l+1}, \cdots B_{k+1}$ 가 완성되어 있을 때 그 앞의 $A_k$를 구성하는것은 $\Pi$와 $\Psi$에서 $S - A_k + B_{k+1} \in \mathcal{I_1}$ 이 성립했으므로 exchange property를 통해
$O(\lvert A_k \rvert)$ 번의 oracle로 가능하다.

$B_{l+1}, \cdots A_k$가 완성되어 있을 때 $B_k$를 구성하는 것도 마찬가지로 가능하고, 최종 oracle call의 개수는 $\lvert A_k \rvert,\lvert B_k \rvert$들의 총 합에 비례하므로 $O(n).\blacksquare$


**Lemma 6.** Maximal augmenting set끼리의 width는 $2l+4$배 넘게 차이나지 않는다. 즉, 두 maximal augmenting set의 width $w_1, w_2$에 대해, $w_2 \le (2l+4)w_1$이 성립한다.

Proof.

$\Pi = (B_1, A_1, \cdots B_{l+1})$, $\Psi = (Q_1, P_1, \cdots Q_{l+1})$ 가 각각 maximal augmenting set이라 하자.
$\Pi$의 width $w_1$와 $\Psi$의 width $w_2$에 대해 $w_2 > (2l+4)w_1$ 라 가정하자(귀류법).

다음 조건을 만족하는 Partial augmenting set $\Phi = (B_1', A_1', \cdots B_{l+1}')$ 를 생성할 것이다.

- $A_k \subset A_k' \subset A_k+P_k$
- $B_k \subset B_k' \subset B_k +Q_k$
- $\lvert A_k' \rvert  > (2l+3-2k) \lvert A_k \rvert$
- $\lvert B_k' \rvert  > (2l+4-2k) \lvert B_k \rvert$ for $1 \le k \le l$
- $\lvert B_{l+1}' \rvert  >  \lvert B_{l+1} \rvert$

그러면 Lemma 5에 의해 $\Pi'$ 내에 $\Pi$를 포함하는 width $\lvert B_{l+1}' \rvert$인 augmenting set이 존재하므로 $\Pi$가 maximal이라는 데에 모순이 되어 가정이 틀렸음을 증명할 수 있다.

$\Phi$를 생성하는 방법은 다음과 같다.

1. 처음에 $B_1'=B_1$로 놓고, $S + B_1' \in \mathcal{I_1}$ 조건 하에서 $B_1'$에 $Q_1$의 원소를 추가한다. $S + Q_1 \in \mathcal{I_1}$이므로 matroid exchange property에 의해 $\lvert B_1' \rvert  =  \lvert Q_1 \rvert  = w_2 > (2l+4)w_1$가 되도록 할 수 있다. 
2. $B_1', A_1', \cdots, B_k'$가 만들어진 상태에서 $A_k'$를 만드는 과정은 다음과 같다:
	1. $S - A_k + B_k \in \mathcal{I_2}$, $S - P_k+ Q_k \in \mathcal{I_2}$로부터 $S-(P_k+A_k)+B_k \in \mathcal{I_2}, S-P_k+(B_k' \setminus B_k) \in \mathcal{I_2}$ 가 성립한다.
	2. matroid exchange property에 의해 $Z \subset B_k'\setminus B_k$가 존재하여 $S-(P_k+A_k)+B_k+Z \in \mathcal{I_2}$, $\lvert B_k+Z \rvert  =  \lvert B_k' \setminus B_k \rvert  > (2l+4-2k) \lvert B_k \rvert  -  \lvert B_k \rvert  = (2l+3-2k)w$
	3. $S-(P_k+A_k)+B_k+Z \in \mathcal{I_2}$와 $S -A_k+B_k \in \mathcal{I_2}$에 matroid excahnge property를 적용하면   다음을 만족하는 $A_k'$가 존재함을 알 수 있다:
		1. $\lvert A_k' \rvert  =  \lvert B_k+Z \rvert  =  (2l+3-2k)w$
		2. $A_k \subset A_k' \subset A_k+P_k$
		3. $S-A_k'+B_k+Z \in \mathcal{I_2}$. 이 때, $rank_2(S-A_k'+B_k+Z) =  \lvert S-A_k'+B_k+Z \rvert  = rank_2(S)$
	4. $rank_2(S-A_k'+B_k') \ge rank_2(S-A_k'+B_k+Z) = rank_2(S)$, $rank_2(S-A_k'+B_k') \le rank_2(S+B_k') = rank(S)$ 이므로 $A_k'$는 partial augmenting set의 조건을 만족한다.
3. $B_1', A_1', \cdots A_k'$가 만들어진 상태에서 $B_{k+1}'$을 만드는 과정은 다음과 같다:
	1. $S-A_k'+B_{k+1}, S - P_k + Q_{k+1} \in \mathcal{I_1}$로부터, $Z \subset Q_{k+1}$가 존재하여 $S-A_k'+Z \in \mathcal{I_1}$ 이면서 $\lvert Z \rvert = \lvert A_k' \cap P_k \rvert  \ge  \lvert A_k' \rvert  -  \lvert A_k \rvert  > (2l+3-2k) \lvert A_k \rvert  -  \lvert A_k \rvert  = (2l+2-2k)w$
	2. $S-A_k'+B_{k+1}, S-A_k'+Z \in \mathcal{I_1}$로부터,  $B_{k+1}'$이  존재하여 다음을 만족한다:
		1. $\lvert B_{k+1}' \rvert  =  \lvert Z \rvert$
		2. $B_{k+1} \subset B_{k+1}' \subset B_{k+1} + Q_{k+1}$
		3. $S-A_k'+B_{k+1}' \in \mathcal{I_1}$
4. $B_1', A_1', \cdots, B_{l+1}'$까지 2,3번 스텝을 통해 생성했다고 하자. partial augmenting set이 되기 위해서는 $B_{l+1}'$에 대해 $S + B_{l+1}' \in \mathcal{I_2}$ 조건을 맞춰 주어야  한다. $B_{l+1} \subset B_{l+1}' \subset B_{l+1} + Q_{l+1}$ 이 성립하고, $S + Q_{l+1} \in \mathcal{I_2}$ 이므로 $B_{l+1}'$ 에서 $B_{l+1}$개 이하의 원소를 제거하여 $S + B_{l+1}' \in \mathcal{I_2}$ 조건을 맞출 수 있다. 이 때 최종 $\lvert B_{l+1}' \rvert  > (2l + 4 - 2(l+1)) \lvert B_{l+1} \rvert  -  \lvert B_{l+1} \rvert  = (2-1) \lvert B_{l+1} \rvert  =  \lvert B_{l+1} \rvert$.

위 step에 의해 생성된 $\Phi = (B_1', A_1', \cdots B_{l+1}')$은 조건을 만족하는 Partial augmenting set이 된다. $\blacksquare$

## Finding maximal augmenting set

이제 $S$가 있을 때 $G(S)$의 maximal augmenting set을 찾는 방법에 대해 알아보자.

먼저 대략적인 개요를 설명하자면, partial augmenting set을 늘려가면서 maximal augmenting set으로 만들 것이다. 이 때 중간 과정에서도 partial augmenting set의 property는 만족해야 한다. partial augmenting set의 상태는 ($B_1, A_1, \cdots , A_l, B_{l+1}$)로 표현된다. 처음에 $B_1, A_1, \cdots , A_l, B_{l+1}$이 모두 $\phi$인 상태로 시작하여, 최종적으로 maximal augmenting set이 될때까지 $A_i$와 $B_i$들을 변형시켜나갈 것이다.

이 과정이 진행중인 도중의 ($B_1, A_1, \cdots A_l, B_{l+1}$)인 상태를 생각해보자. 이 때, $A_i$또는 $B_i$에 포함된 원소들을 **selected** 상태, 아직 한번도 $A_i$나 $B_i$에 포함된 적 없는 원소들을 **fresh** 상태, selected였으나 현재는 $A_i, B_i$에서 제거된 원소들을 **removed** 상태라 하자.

$D_1, D_2, \cdots D_{2l+1}$의 각 원소들은 처음에는 모두 fresh 상태이며 알고리즘 수행 과정에서 fresh에서 selected, selected에서 removed 상태로만 전이될 수 있다.

앞으로 $D_i$에서 fresh한 원소들의 집합을 $F_i$, removed된 원소들의 집합을 $R_i$라 부를 것이다.

### Refine subroutine

($B_1, A_1, \cdots A_l, B_{l+1}$)가 maximal partial augmenting set이 되도록 만드는 알고리즘은 간단히 말해 $Refine$ 서브루틴을 반복하는 것이다. 그리고 $Refine$ 서브루틴은 $Refine1$과 $Refine2$를 각 layer에 대해 수행하는 것으로 이루어진다.

$Refine$ subroutine : 
```
for k = 0, 1, ... , l do:
	Refine1(k)
	Refine2(k)
Refine1(0)
```


**Invariants during Refining**

Refine subroutine 내에서 Refine1과 Refine2를 수행하면서, 다음 성질들은 항상 유지된다. 이 성질들은 Refine subroutine을 update가 더이상 일어나지 않을 때 까지 반복하면 ($B_1, A_1, \cdots A_l, B_{l+1}$)가 maximal augmenting set이 됨을 증명할 때 쓰일 것이다.

1. $(B_1, A_1, \cdots, B_{l+1})$는 partial augmenting set.
2. 임의의 $X \subset D_{2k+1}-R_{2k+1}$ 에  대해, $S - (A_k + R_{2k}) + X \in \mathcal{I_1}$이면 $S - A_k + X \in \mathcal{I_1}$.
	- $rank_1(W - R_{2k}) = rank_1(W) -  \lvert R_{2k} \rvert$, where $W = S - A_k + (D_{2k+1} - R_{2k+1})$과 동치. 
	- 이는 $R_{2k+1}$이 useless하다면 $R_{2k}$도 useless함을 함의
3. $rank_2(W + R_{2k-1}) = rank_2(W)$ where $W = S - (D_{2k} - R_{2k}) +B_k$
	- 이는 $R_{2k}$가 useless하다면 $R_{2k-1}$도 useless함을 함의


### Refine1

Refine1은 간단히 말해 $B_{k+1}$을 늘리고 $A_k$를 줄여서 $\lvert A_k \rvert  =  \lvert B_{k+1} \rvert$이 되도록 하는 과정이다. 이 때, $F_{2k+1}$에서 $B_{k+1}$로, $A_k$에서 $R_{2k}$로 원소의 이동이 일어난다.

Refine1($k$):

- step 1. $S-A_k + B_{k+1} \in \mathcal{I_1}$ 을 만족하도록 $B_{k+1}$에 $F_{2k+1}$의 원소들을 최대한 먹여서 늘린다(Formal하게: maximal한 $B_{k+1}$을 잡는다).
- step 2. $S-A_k + B_{k+1} \in \mathcal{I_1}$을 만족하는 조건 하에서 $A_k$의 원소를 최대한 빼서 줄인다(Formal하게: minimal한 $A_k$를 잡는다).


아래는 $\lvert A_k \rvert  =  \lvert B_{k+1} \rvert$이 되는 이유에 대한 설명이다.

먼저 $S \in \mathcal{I_1}$, $S-A_k^{old}+B_{k+1} \in \mathcal{I_1}$
에서 $\mathcal{M_1}$의 independent set $S-A_k^{old}+B_{k+1}$를 $S$에 대해 extension하면 자연스레 $A_{k}^{old}$의 원소를 몇개 추가하여 $\lvert S \rvert$와 크기가 같은 집합을 만들 수 있고, 그것이 $S-A_k + B_{k+1}$가 된다.

1이나 2번 과정에서 $\lvert A_k \rvert  <  \lvert B_{k+1} \rvert$ 이 될 수는 없을까 하는 생각이 들 수 있다.

그러나, 중간에 $\lvert A_k \rvert  <  \lvert B_{k+1} \rvert$가 된다면 matroid의 성질에 의해 $S + b \in \mathcal{I_1}$인 $b \in B_{k+1}$이 존재해야 하는데, 그러한 $b$는 $B_1$에만 존재할 수 있으므로 Refine1($0$)에서만 가능하다. 그리고 실제로 $\lvert A_k \rvert  =  \lvert B_{k+1} \rvert$는 $k \ge 1$ 일때만 성립한다.

Refine1($k$) 이전에 Invariant들이 성립했다고 가정하자. Refine1($k$)을 수행한 후에도 Invariant들을 그대로 만족하는지 체크하여야 한다.

1. partially augmenting set인가? 
	
	$A_k$ 줄고 $B_{k+1}$ 늘었으니 (e)의 rank는 줄었을 리 없고 $rank_2(S)$를 넘을 수 없으니 OK. (d)는 $k$만 바뀌었으니 OK. (a),(b),(c)는 자명.
	
2. 임의의 $X \subset D_{2k+1}-R_{2k+1}$ 에  대해, $S - (A_k + R_{2k}) + X \in \mathcal{I_1}$이면 $S - A_k + X \in \mathcal{I_1}$?
	 
	 먼저, $A_k + R_{2k} = A_k^{old} + R_{2k}^{old}$. Refine1 전에는 invariant 성립했고 $R_{2k+1}$은 변하지 않았으므로 임의의 $X \subset D_{2k+1}-R_{2k+1}$에 대해 $S - (A_k + R_{2k}) + X \in \mathcal{I_1}$이면 $S - A_k^{old} + X \in \mathcal{I_1}$.
	 
	 한편, Refine 후 $S - A_k + B_{k+1} \in \mathcal{I_1}$이므로 $\bar{A} \subset A_k^{old} - A_k$와 $\bar{B} \subset B_{k+1} - X$가 존재하여 $S - A_k^{old} + \bar{A} + X + \bar{B} \in \mathcal{I_1}$ , $\lvert \bar{A} \rvert  +  \lvert \bar{B} \rvert  =  \lvert A_k^{old} \rvert  -  \lvert A_k \rvert  +  \lvert B_{k+1} \rvert  -  \lvert X \rvert$.

	 만약 $A_k^{old} - A_k \neq \bar{A}$이면 $\lvert \bar{B} \rvert  >  \lvert B_{k+1} \rvert  -  \lvert X \rvert$이고, 이에 따라 $X + \bar{B} \subset D_{2k+1} - R_{2k+1}$ 의 크기는 $\lvert B_{k+1} \rvert$보다 크다.
	 Refine1의 1번 과정에 의해 $B_{k+1}$ 은 $S - A_{k}^{old}$ 에 maximally dependent한데, $S - A_{k}^{old} + X + \bar{B} \in \mathcal{I_1}$하므로 이는 모순이다. 따라서, $\bar{A} = A_k^{old} - A_k$ 이고 $S- A_k + X = S - A_k^{old} + \bar{A} + X \in \mathcal{I_1}$이 성립한다.
	 
3. $rank_2(W + R_{2k-1}) = rank_2(W)$ where $W = S - (D_{2k} - R_{2k}) +B_k$ ?

	 $B_{k+1}$ 증가, $R_{2k}$ 증가만 영향을 끼친다. 근데 둘다 $W$를 늘리는 역할이고 $R_{2k-1}$는 커진 바 없으므로 ok.

따라서, Refine1($k$)을 수행한 이후에도 Invariant들을 모두 만족함을 알 수 있다.

### Refine2


Refine2는 간단히 말해 $B_k$를 줄이고 $A_k$를 늘려서 $\lvert B_k \rvert  =  \lvert A_k \rvert$가 되도록 하는 과정이다. 이 떄, $B_k$에서 $R_{2k-1}$로, $F_{2k}$에서 $A_k$로 원소의 이동이 일어난다.


Refine2($k$):

- step 1. $S-(D_{2k}- R_{2k}) +B \in \mathcal{I_2}$ 를 만족하는 maximal한 $B \subset B_k^{old}$를 잡아 $B_{k}^{new} \leftarrow B$
- step 2. $S- A_k +B_{k}^{new} \in \mathcal{I_2}$ 를 만족하는 조건 하에서 $A_k$에 $F_{2k}$의 원소 최대한 추가

아래는 $\lvert A_k \rvert  =  \lvert B_k \rvert$가 되는 이유에 대한 설명이다.

먼저 $S \in \mathcal{I_2}$.
$\mathcal{M_2}$의 independent set $S- (D_{2k}- R_{2k}) + B_k^{new}$ 에서 $S$에 대해 extension하면 자연스레 $D_{2k} - R_{2k}$ 원소 몇 개가 추가된다. 그 집합이 결국 $S - A_k^{new} + B_k^{new}$ 이고, 따라서 $\lvert S - A_k^{new} + B_k^{new} \rvert  =  \lvert S \rvert$.

Refine2($k$) 수행 이후의 Invariant check는 위에서 Refine1에 대해 보인 것과 동일한 방법으로 가능하다.

한편, Refine1과 Refine2가 두 집합의 크기를 맞추는 step이므로 다음 두 Lemma가 성립한다.

**Lemma.** Refine1을 한 후, 적어도 $\lvert A_k \rvert  -  \lvert B_{k+1} \rvert$  element가 $F$ 에서 $B$로 가거나, $A$에서 $R$로 간다.

**Lemma.** Refine2을 한 후, 적어도 $\lvert B_k \rvert  -  \lvert A_{k} \rvert$  element가 $F$ 에서 $B$로 가거나, $A$에서 $R$로 간다.

**Lemma 7.** 두 Lemma에 의해, Refine subroutine에서 최소 $\lvert B_1 \rvert  -  \lvert B_{l+1} \rvert$개의 element가 이동함을 알 수 있다.


$D_1, \cdots D_{2l+1}$은 이미 알고 있는 상황이라고 가정하자. 이 때 다음이 성립한다.

**Theorem.** 한 번의 Refine subroutine에서는 $O(n)$번의 independent oracle call이 사용된다.

**Proof.**

Refine1($k$)에서는 $O(\lvert D_{2k} + D_{2k+1} \rvert)$번의 oracle call을 사용하고, Refine2($k$)에서는 $O(\lvert D_{2k-1} + D_{2k} \rvert)$번의 oracle call을 사용하므로 한 번의 Refine subroutine에서는 $O(n)$번의 independent oracle call을 사용한다.


**Theorem.** Refine을 아무 element도 이동하지 않을 때까지 하면 maximal augmenting set이 된다.

**Proof.**

Refine했는데 아무 element도 이동하지 않는다면 $\lvert B_1 \rvert  =  \lvert A_1 \rvert  = \cdots =  \lvert A_l \rvert  =  \lvert B_{l+1} \rvert$이므로 augmenting set. 이제 maximality만 보이면 된다.

maximal이 아니면 augmenting set으로 $S$를 update한 후의 exchange graph에서 augmenting path가 존재한다. 이를 $(b_1, a_1, \cdots, a_l, b_{l+1})$ 이라 하자. 이때 $a_k \notin A_k, b_k \notin B_k$. 

$b_1$은 fresh일 수가 없다. 그렇지 않다면 $Refine1(0)$에서 $B_1$에 추가되었기 때문.

$b_1, a_1, \cdots b_k$ 까지 removed 상태라고 가정하자. $a_k$가 removed가 아니면 $a_k \in F_{2k}$이고, augmenting path의 성질에 의해 $S-({A_k}+a_k) + (B_k+b_k) \in \mathcal{I_2} \Rightarrow S-(D_{2k} - R_{2k}) + (B_k + b_k) \in \mathcal{I_2}$ . 그런데 이는 invariant $rank_2(W + R_{2k-1}) = rank_2(W)$ where $W = S - (D_{2k} - R_{2k}) +B_k$에 반한다.

$b_1, a_1, \cdots a_k$ 까지 removed 상태라고 가정하자. $b_{k+1}$가 removed가 아니면 $b_{k+1} \in F_{2k+1}$이고, augmenting path의 성질에 의해 $S-({A_k}+a_k) + (B_{k+1}+b_{k+1}) \in \mathcal{I_1} \Rightarrow S- A_k - R_{2k} + (B_{k+1}+b_{k+1}) \in \mathcal{I_1}$. Invariant에 의해 $rank_1(W - R_{2k}) = rank_1(W) -  \lvert R_{2k} \rvert$, where $W = S - A_k + (D_{2k+1} - R_{2k+1})$ 를 적용하면 
$S-{A_k} + B_{k+1} + b_{k+1} \in \mathcal{I_1}$ 인데 이는 $\lvert S - A_k + B_{k-1} + b_{k-1} \rvert  >  \lvert S \rvert$, $D_{2k+1} \in span(S)$ 에 의해 모순.

따라서, $b_1, a_1, \cdots, b_{l+1}$ 는 모두 removed 상태. 그런데 $b_{l+1}$이 removed 되는 과정은 refine에 없고, 따라서 모순. $_\blacksquare$


**Lemma.** Maximal augmenting set을 얻을 때 까지 수행해야 하는 Refine subroutine의 횟수는 $O(n)$이다.

**Proof.** 모든 과정을 통틀어 각 element의 상태는 최대 2번 바뀐다. ($fresh \rightarrow selected \rightarrow removed$). 따라서, $O(n)$번 이하의 refine으로 Maximal augmenting set을 얻는다.$_\blacksquare$


### Naive algorithm for maximal augmenting set

$S$에 대해 maximal augmenting set $\Pi_l$을 얻어 $S$를 $S \oplus \Pi_l$로 업데이트하는 과정을 한 phase라 하자. maximal augmenting set을 얻기 위해 Refine이 최대 $O(n)$번 이루어지고 각 Refine에서 $O(n)$번 oracle call이 이루어지므로 한 phase당 oracle call은 최대 $O(n^2)$번 필요하다.

Lemma 4에 의해 한 phase를 수행한 후 $l$은 항상 1 이상 증가한다. $G(S)$에서 shortest $(s,t)$-path의 길이는 $n+1$ 이하이므로 nonempty augmenting set이 존재하지 않을 떄까지 필요한 phase의 수는 $O(n)$이다. 이로써 $O(n^3)$번의 oracle이 필요한 naive algorithm을 얻는다.

이전 포스트에서 다뤘던 다음 Lemma를 떠올려보자.

**Lemma(Cunningham).** $r$ 를 maximum size of common independent set이라 하고, $S$ 는 크기가 $r$보다 작은 common independent set이라 할 때, 길이 $2\lvert S\rvert /(r-\lvert S\rvert ) + 2$ 이하의 augmenting path가 존재한다.

위 Lemma에 의하면 $1-\epsilon$ approximation을 위해서는 $l$이 $O(1/\epsilon)$ 이하일 때만 phase를 수행하면 충분하므로 $O(n^2/\epsilon)$ oracle call로 approximation solution을 구할 수 있다.

## Going subquadratic

Maximal augmenting set이 구해질 때 까지 Refine을 하는 것은 expensive하다. 한편, Lemma 7에 의해 $\lvert B_1 \rvert  -  \lvert B_{l+1} \rvert$이 큰 경우 한 번의 Refine으로 많은 원소의 state가 바뀐다. 즉, 적은 횟수의 Refine만으로도 $\lvert B_1 \rvert$과 $\lvert B_{l+1} \rvert$의 크기를 어느정도 맞춰놓을 수 있다. 이 때 Lemma 5를 이용하면 $\lvert B_{l+1} \rvert$ width의 augmenting set을 구할 수 있으므로, 꽤 큰 augmenting set을 구할 수 있다고 볼 수 있다. 이제 이 아이디어를 기반으로 하나의 phase를 처리하는 **Hybrid** 서브루틴을 소개할 것이다.

상수 $p$에 대해, 하나의 phase를 수행하는 다음과 같은 알고리즘을 생각할 수 있다.

Hybrid($p$):

1. $s$를 시작점으로하는 BFS를 통해 Dinstance layer $D_1, D_2, \cdots D_{2l+1}$을 구한다. 

2. Refine subroutine을 최소 1번 하고, 반복하다가 $\lvert B_1 \rvert  -  \lvert B_{l+1} \rvert  \le p$ 가 되면 멈춘다. 이것은  Refine 횟수를 $n/p$로 bound하므로(by Lemma 7) oracle의 횟수는 $O(n^2/p)$로 bound된다.

3. 현재 구한 partial augmenting set에 대해 $B_{l+1}$ 이 일치하는 augmenting set을 구해서 update한다. (by Lemma 5)

4. shortest augmenting path의 길이가 바뀌기 전까지 $\tilde{O}(n)$ 번의 oracle로 하나의 augmenting path씩 찾아서 업데이트한다.

**Lemma.**  몇 번의 Refine을 통해 구해진 partial augmenting set $\Phi_{l} = (B_1, A_1, \cdots B_{l+1})$과 3번 과정에서 구해진 augmenting set $\Pi_l \subset \Phi_l$을 생각하자. 이 때, $\Pi_l$ 을 포함하면서 width가 $\lvert B_1 \rvert$ 이하인 maximal augmenting set $\Pi'_l$가 존재한다.

**Proof.**

$\Phi_{l}$에서 Refine 과정을 update가 일어나지 않을 때까지 반복하여 maximal augmenting set $\Pi'_l$를 얻는 과정동안 $\Pi_l$의 원소들이 removed되지 않도록 수행하는 방법을 제시할 것이다.

만약 그것이 가능하다면, $\Pi'_l$는 조건을 모두 만족한다. 먼저, $\Pi_l \subset \Pi'_l$임은 자명하다.

한편, Refine의 마지막 과정이 $Refine1(0)$이기 때문에, $\Phi_l$의 $B_1$은 maximal이다. 즉, $S+B_1+f \in \mathcal{I_1}$인 $f \in F_1$이 존재하지 않는다. 

이는 Refine을 update가 일어나지 않을 때 까지 반복해서 최종적으로 얻는 $\Pi'_l$의 $B_1$보다 현재 $\Pi_l$의 $B_1$이 작지 않음을 뜻한다. 즉, $\Pi'_l$의 width는 $\lvert B_1 \rvert$ 이하이다.  

이제 Refine 과정에서 $\Pi_l$의 원소들이 removed되지 않도록 하는 것이 가능함을 보이면 충분하다.

Refine1($k$)에서는 $\Pi_l$의 $A_k$에 해당되는 원소들이, Refine2($k$)에서 $B_k$에 해당되는 원소들이 removed되지 않도록 각 step을 그리디하게 고르면 removed되는 $\Pi_l$의 원소가 없도록 하는 것이 가능함을 어렵지 않게 보일 수 있다. $\blacksquare$

위 Lemma로부터 다음의 결론이 쉽게 도출된다.

**Theorem.** Hybrid의 4번 과정에서 찾아지는 augmenting path의 개수는 $(2l+4)p$ 이하이다.

**Proof.**

위 Lemma로부터 $\lvert \Pi'_l - \Pi_l \rvert \le p$인 $G(S)$의 maximal augmenting set $\Pi'_l$가 존재하여 $\Pi_l$를 포함한다.

Lemma 3에 의해, $\Pi'_l \setminus \Pi_l$는 $G(S \oplus \Pi_l)$의 maximal augmenting set이 되며 width는 $p$ 이하이다. 이에 Lemma 6를 적용하면 $G(S \oplus \Pi_l)$의 임의의 maximal augmenting set의 width는 $(2l+4)p$를 넘을 수 없다. 한편, 4번 과정에서 찾아지는 augmenting path들은 $G(S \oplus \Pi_l)$의 consecutive shortest path이므로 이 개수는 maximal augmenting set의 width 최댓값인 $(2l+4)p$를 넘을 수 없다. $\blacksquare$


Hybrid($p$)에서 사용되는 oracle 횟수를 세어보자. 2번 과정에서 $O(n^2/p)$회, 3번 과정에서 $O(n)$회, 4번 과정에서 $\tilde{O}(npl)$회가 필요하다. 1번 과정의 BFS는 $O(nl \log n)$회로 가능함을 저번 포스트에서 보였다. 따라서, 총 횟수는 $\tilde{O}(npl + n^2/p)$ 가 된다.


### Fast approximation algorithm for Matroid Intersection

$p = \sqrt{n\epsilon / \log r}$로 놓자.

앞서 보았던 $l$이 $O(1/\epsilon)$ 이하일 때까지 phase를 시행하는 알고리즘에서 하나의 phase를 Hybrid($p$)로 수행하도록 하자. 이 때

$$\sum_{l=1}^{1/\epsilon} (npl + n^2/p) = O((n/\epsilon)^{1.5})$$


가 성립하므로, $\tilde{O}((n/\epsilon)^{1.5})$ oracles에 $1-\epsilon$ approximation이 가능하다.

### Omitted Details

사실 Hybrid($k$)의 4번 과정을 수행함에 있어 하나의 augmenting path씩 찾아서 $S$를 업데이트하는 동안 Distance Layer가 바뀌기 때문에 Distance Layer를 업데이트하는 추가적인 시간이 소요된다.

하지만 이 부분 역시 Monotonicity Lemma 등을 이용하면 4번 과정 전체동안 

$$ O(\sum_{a \in V} (d_a^{end} - d_a^{start} \log n)) + O(n\lvert P \rvert \log n) $$

번의 oracle만으로 충분함을 증명할 수 있다. (단, $\lvert P \rvert$는 4번 과정에서 구한 augmenting path의 개수, $d_a^{end}$와 $d_a^{start}$는 4번 과정 시작 전의 $G(S)$에서 $dist(s,a)$와 4번 과정 종료 후의 $dist(s,a)$.)

Approximation algorithm에서 $d_a$ 값들은 모두 $O(1/\epsilon)$ 이므로 이부분의 oracle 횟수는 $\tilde{O}((n/\epsilon)^{1.5})$ 에 dominate된다.


## 6. Reference

- [Cun86] William H. Cunningham. Improved bounds for matroid partition and intersection algorithms. SIAM J. Comput., 15(4):948–957, 1986.

- [Pri15] Christopher Price. Combinatorial algorithms for submodular function minimization and related problems. Master’s thesis, University of Waterloo, 2015.

- [Ngu19] Huy L. Nguyen. A note on Cunningham's algorithm for matroid instersection(2019)

- [CLS+19] Deeparnab Chakrabarty, Yin Tat Lee, Aaron Sidford, Sahil Singla, and Sam Chiu-wai Wong. Faster matroid intersection. In FOCS, pages 1146–1168. IEEE Computer Society, 2019.

- [Bli21]  Joakim Blikstad. Breaking O(nr) for Matroid Intersection(2021)