---
layout: post

title: "Communication complexity"

date: 2023-12-26

author: ainta

tags: [complexity theory]
---

# Introduction


## two-party model of deterministic communication

(늘 그렇듯이) Alice와 Bob이라는 이름의 두 사람이 있다. 

함수 $f: X \times Y \rightarrow Z$에 대해, Alice는 $x \in X$, Bob은 $y \in Y$를 알고 있을 때 두 사람이 $f(x,y)$를 최소 bit의 커뮤니케이션으로 알아내는 것이 목표이다. 
즉, 목표는 $f(x,y)$의 값을 두 명이 알아내는 것이고, cost가 주고받은 bit의 개수일 때 cost를 minimize하는 문제이다.
먼저 여러 가지 $f$의 예를 살펴보며 각각 어떤 방식으로 해결하는 것이 확인해 보자.

**Example 1.**
$EQ_n: \\{ 0, 1 \\}^n \times \\{ 0, 1 \\}^n \rightarrow \\{ 0, 1 \\}$,  $EQ_n(x,y) = 1$ iff $x = y$

Solution:  Alice 가 Bob에게 $x$ 전체를 전송 ($n$ 비트), Bob이 $x=y$인지 체크하여 $f(x,y)$를 계산하고 결과 Alice에게 전송. Cost: $n+1$ bits

**Example 2.**
$PARITY_n: \\{ 0, 1 \\}^n \times \\{ 0, 1 \\}^n \rightarrow \\{ 0, 1 \\}$, $PARITY_n(x,y) = 1$일 조건은 $x,y$ 전체에 1 개수가 홀수일 때 

Solution: Alice 가 Bob에게 $\oplus_{i=1}^n x_i$ 전송, Bob이 계산 결과 Alice에게 전송. Cost: 2 bits



### Protocol Tree

Alice와 Bob이 주고받은 비트에 따른 상태 변화를 트리로 표현한 형태를 Protocol tree라 한다.

<p align="center">
    <img src="/assets/images/communication-complexity/1.png" width="400"/>
    <br>
</p>

위 protocol tree를 보자. 각 노드의 알파벳은 누구의 턴인지를, edge의 숫자는 어떤 bit가 전송되었는지를 나타낸다. 여기서는 root가 A이므로 처음에 Alice가 Bob에게 비트를 전송한다.
처음에 전송한 비트가 1이면 Bob의 턴이 되고(left child), Bob이 Alice에게 비트를 전송한다.
처음에 전송한 비트가 0이었다면 다시 Alice의 턴이 되고(right child), Alice가 Bob에게 비트를 전송한다.
protocol tree $T$에서 모든 $x,y$에 대해  마지막 communication의 비트가 $f(x,y)$ 와 동일할 때, $T$를 $f$의 protocol tree라 한다.

## Rectangles

**Definition.** A **rectangle** is a set $S \subseteq X \times Y$ of the form $S = A \times B, A \subseteq X, B \subseteq Y$

위 definition으로부터 rectangle $S$의 두 원소 $(a,b), (c,d)$에 대해 $(a,d), (b,c) \in S$ 가 성립함을 쉽게 알 수 있다.

**Definition.**  $S \subseteq X \times Y$ is **$f$-monochromatic** if $f$ is constant over $S$.

$f$가 rectangle $S$에서 모두 같은 값을 가질 때, $S$를 $f$-monochromatic하다고 한다. 편의를 위해 함수값이 모두 0인 rectangle은 0-rectangle, 1인 rectangle은 1-rectangle로 부르기로 하자. 한편, Rectangle과 protocol tree에 대해 다음 정리가 성립함은 간단히 보일 수 있다.

**Theorem 1.** protocol tree의 임의의 리프에 대해, 해당 리프에 도착하는 $(x,y) \in X \times Y$ 들의 집합은 rectangle이다.

**Corollary 2.** $f$의 모든 deterministic communication protocol은 $X \times Y$ 를 $f$-monochromatic rectangle들로 분할한다(partition). 나아가, rectangle들은 최대 $2^C$개이다. (단, $C$ 는 protocol tree의 높이 = communication cost).

함수의 Communication complexity를 다음과 같이 정의하자.

**Definition.** $D(f) :=$ minimum height of protocol tree for $f$ 를 $f$의 **communication complexity**라 한다.


## Charateristic matrices

함수 $f: X \times Y \rightarrow \\{0, 1\\}$에 대해, Charateristic matrice $M_f$는 다음과 같이 정의된다:

$$M_f = [f(x,y)]_{x\in X, y\in Y}$$

## Fooling Sets

**Definition**.  $S \subseteq X \times Y$에 대해 다음 조건을 만족하는 $z \in \\{0, 1\\}$가 존재할 때, $S$를 $f$의 **fooling set**이라 한다.
- For every $(x,y) \in S$, $f(x,y) = z$
- For any distinct $(x_1, y_1)$ and $(x_2, y_2)$ in $S$, either $f(x_1, y_2) \neq z$ or $f(x_2, y_1) \neq z$


**Theorem 3.** $f$의 fooling set $S$에 대해,  $D(f) \ge \log_2 \lvert S \rvert$.

**Proof**. 높이가 $D(f)$ 이하인 protocol tree for $f$가 존재하고, 해당 트리는 최대 $2^{D(f)}$ 개의 leaf를 가진다. 각 leaf는 monochromatic rectangle에 대응된다. 한편, fooling set의 두 번째 성질에 의해 fooling set의 각 원소는 protocol tree에서 서로 다른 리프에 도달한다. 따라서, $2^{D(f)} \ge \lvert S \rvert$. $\blacksquare$

Theorem 3을 이용하여 여러 function들의 communication complexity lower bound를 구할 수 있다. 몇 가지 예를 살펴보자.

**Example 1.**

$EQ_n(x,y)$의 characteristic matrix은 $I_{2^n}$이므로, matrix의 diagonal은 $f$의 크기 $2^n$인 fooling set이다. 따라서, $D(EQ_n) \ge n$.

**Example 2.**

$DISJ_n(x,y)$를 $n$자리 비트스트링 $x,y$에 대해 $x \cap y = \phi$이면 $1$, 아니면 $0$인 함수로 정의하자.

$S = \\{(A, A^c) \mid A \subseteq \\{1,2, \cdots, n \\} \\}$는 $f$의 fooling set이고, size가 $2^n$이다. 따라서, $D(DISJ_n) \ge n$.

## Covers and lower bounds

**Definition.** 
- $C^P(f)$ : $f$의 protocol tree의 가능한 최소 리프 개수
- $C^D(f)$ : $X \times Y$를 최소 개수의 $f$-monochromatic rectangle들로 분할할 때 그 개수
- $C^0(f):$  $f^{-1}(0)$을  $f$-monochromatic rectangle들로 cover할 때 필요한 최소 개수.
- $C^1(f)$ :   $f^{-1}(1)$을  $f$-monochromatic rectangle들로 cover할 때 필요한 최소 개수.
- $C(f)$ :  $X \times Y$을  $f$-monochromatic rectangle들로 cover할 때 필요한 최소 개수. 즉, $C(f) = C^0(f) + C^1(f)$.

**Theorem 4**. $2^{\Theta(D(f))} = C^P(f) \ge C^D(f) \ge C(f) \ge 2^{\Theta(\sqrt{D(f)})}$

**Proof.**
가운데 두 부등호는 간단하다.

- $C^P(f) \ge C^D(f)$ : Corollary 2에 의해 자명
- $C^D(f) \ge C(f)$ : partition은 cover이기도 하므로 자명

먼저, 첫 등식을 증명해보자.

**Lemma 5.** $D(f) = \Theta( \log C^P(f))$

**Proof.**

- height이 $D(f)$ 이하인 protocol tree가 존재하므로  $2^{D(f)}$ 개 이하의 leaf를 가진다. 따라서,  $2^{D(f)} \ge C^P(f)$
- 이제 $D(f) = O(\log C^P(f))$임을 보이면 충분하다. 먼저, $C^P(f) = l$ 개의 leaf를 갖는 $f$의 protocol tree가 존재한다. 이 protocol tree를 기반으로 새로운 $f$의 protocol tree를 생성하여 height이 $O( \log C^P(f))$ 이하가 되도록 하자. protocol tree는 binary tree이므로, 노드 $u$가 존재하여 $u$의 subtree가 $[l/3, 2l/3]$개의 리프를 가진다. $u$에 해당하는 rectangle을 $R_u$라 하자. Alice와 Bob은 비트 1개씩을 주고받아서 $(x,y) \in R_u$인지를 확인할 수 있고, 최종 도착점이 될 수 있는 leaf가 2/3 이하로 줄게 된다. 이를 반복하면 $O(\log C^P(f))$ bit의 cost를 갖는 communication protocol을 construct할 수 있다. 따라서, $D(f) = O( \log C^P(f))$.$\blacksquare$

이제 마지막 부등식을 증명하자.

**Lemma 6.** $D(f) \le O(\log C^0(f) \log C^1(f))$

**Proof.**

$R \subseteq f^{-1}(0), S \subseteq f^{-1}(1)$가 $f$-monochromatic rectangle일 때, $R = RX \times RY$, $S = SX \times SY$라 하면 $RX$와 $SX$가 disjoint하거나, $RY$와 $RS$가 disjoint하다.

$f^{-1}(0)$와 $f^{-1}(1)$의 optimal한 $f$-monochromatic rectangle cover $Cov_0, Cov_1$을 생각하자. 이는 각각 $C^0(f), C^1(f)$개의 rectangle로 이루어진 cover이다. $Cov_1$의 각 원소인 rectangle들에 서로 다른 id를 부여한다.
이를 통해 $O(\log C^0(f) \log C^1(f))$ 이하의 depth를 갖는 protocol을 construct할 것이다. 아이디어는 input $(x,y)$에 대해 $(x,y)$를 포함하는 1-rectangle을 찾는 것이다.

protocol의 각 round는 다음과 같이 이루어진다:

1. Alice는 다음을 만족하는 $Cov_1$의 원소 $Q = QX \times QY$가 존재하는지 찾는다:
   - $x \in QX$
   - $QY$는 $Cov_0$의 절반 이상의 원소와 disjoint
  만약 조건을 만족하는 $Q$가 존재한다면 Alice는 Bob에게 $Q$의 id를 보내고, $Cov_0$에서 $QY$와 disjoint한 원소들을 모두 제거한다.

2. Alice가 $Q$를 찾는데 실패한 경우, Bob은 다음을 만족하는 $Cov_1$의 원소 $P = PX \times PY$가 존재하는지 찾는다:
   - $y \in PY$
   - $PX$는 $Cov_0$의 절반 이상의 원소와 disjoint
  만약 조건을 만족하는 $P$가 존재한다면 Alice는 Bob에게 $P$의 id를 보내고, $Cov_0$에서 $PX$와 disjoint한 원소들을 모두 제거한다.

3. Alice와 Bob 모두 실패했다면, $(x,y)$는 $Cov_1$의 원소에 포함되지 않으므로 $f(x,y) = 0$으로 결론짓는다.

위 step 1~3을 $P$ 또는 $Q$가 $(x,y)$를 포함하거나 3번 step에서 결론이 날 때까지 반복한다.

프로토콜에서 communication cost를 계산해 보자. $P, Q$의 id가 $O(\log C^1(f))$ 비트, 라운드의 수가 $O(\log C^2(f))$ 이하이므로,  $D(f) \le O(\log C^0(f) \log C^1(f))$임을 알 수 있다. $\blacksquare$

## Nondeterminism

**Definition.**  
비결정론적 통신 복잡성(nondeterministic communication complexity)은 함수 $f : X \times Y \rightarrow \\{0,1\\}$에 대해서 증명 시스템(proof system) $f$의 최소 비용으로 정의된다. 력 $x$와 $y$에 대해 $f(x, y) = 1$ 인 경우, 전지전능한 증명자(all-powerful prover)는 두 입력 모두를 알고 Alice와 Bob에게 증명서(certificate)를 제출한다. 이 때, 증명 시스템의 비용 $C^P_f$는 $C_1 + C_2$로 정의된다. 여기서 $C_1$은 증명서의 크기(비트 단위)이고, $C_2$는 Alice와 Bob이 증명서 수신 후 사용하는 검증 프로토콜(verification protocol)의 최대 비용이다. 증명 시스템은 다음 두 조건을 반드시 만족해야 한다:

**Completeness.**  
만약 $f(x, y) = 1$ 이라면, prover가 보낼 수 있는 certificate가 존재하여 Alice와 Bob이 $f(x, y) = 1$을 선언한다.

**Soundness.**  
만약 $f(x, y) = 0$ 이라면, Alice와 Bob은 임의의 certificate에 대해 $f(x, y) = 0$을 선언한다.

**Example.**  
$DISJ_n$의 부정(negation)인 $\neg DISJ_n$에 대해 다음과 같은 proof system을 생각해 보자.
1. certificate: $x_i = y_i = 1$ 인 인덱스 $i$
2. verification protocol: Alice는 $x_i$, Bob은 $y_i$를 상대에게 전달하고, 최종적으로는 $f(x,y) = x_i \wedge y_i$라고 선언한다.

이 proof system은 Completeness와 Soundness를 모두 만족함을 알 수 있다. 즉, $\neg DISJ_n$은 $\Omega(n)$의 communication complexity를 갖지만 nondeterministic communication complexity는 $O(1)$이 된다.

proof system의 비용에 대한 두 lemma를 생각해 보자.

**Lemma 7.** 모든 함수 $f : X \times Y \rightarrow \{0,1\}$에 대해, $\log_2 C^1(f) + 2$의 비용을 가지는 valid proof system이 존재한다.

**Proof.**  
다음은 valid proof system이다.

1. prover는 rectangle $R \in Cov_1$의 인덱스를 certificate로 전달한다. ($\log_2  C^1(f)$ bit)
2. Alice는 $x \in R$을 검증하고, 결과를 Bob에게 보낸다. (1 bit)
3. Bob은 $y \in R$을 검증하고, 결과를 Alice에게 보낸다. (1 bit)

$\blacksquare$

**Lemma 8.** 함수 $f : X \times Y \rightarrow \{0,1\}$에 대한 proof system의 비용은 적어도 $\log_2  C^1(f)$ 이상이다.

**Proof.**  
proof system의 비용 $C_1 + C_2$에서, 서로 다른 certificate는 $2^{C_1}$가지, 그 이후의 protocol의 leaf 개수는 $2^{C_2}$개 이하이다. 즉, proof system은 최대 $2^{C_1+C_2}$개의 $f$-monochromatic rectangle을 cover한다고 볼 수 있고, 이것이 $f^{-1}(1)$을 모두 덮어야 한다. 따라서, $2^{C_1+C_2} \ge C^1(f)$. 즉, $C_1+C_2 \ge \log_2  C^1(f). \blacksquare$

위 두 lemma로부터, nondeterministic communication complexity의 정의하는 또다른 방법을 다음과 같이 생각할 수 있다.

**(Alternative) Definition.** The nondeterministic communication complexity of $f : X \times Y → \\{0, 1\\}$ is defined as $N(f) = \log_2 C
^1(f)$. The co-nondeterministic communication complexity of $f$ is defined as $N(\neg f) = \log_2 C
^0(f)$.

위 정의에 Lemma 6을 적용하면 다음과 같은 따름정리를 얻는다.

**Corollary 9.** $D(f) \le N(f)N(\neg f)$


Corollary 9를 사용하여 직관과 대치되는 결과를 얻을 수 있는 재미있는 예시가 있어 소개한다.

**Setting.** $G=(V,E)$가 있고, $V$의 partition $V_A, V_B$가 있다. $V_A$와 $V_B$ 집합, 그리고 $V_A$와 $V_B$를 잇는 간선의 집합 $S$은 이미 알려져 있다.

Alice는 $V_A$ 및 $V_A$에 연결된 모든 간선에 대한 정보만을, Bob은 $V_B$ 및 $V_B$에 연결된 모든 간선에 대한 정보만을 알고 있다.

$X$를 $V_A$ 내부 간선들의 연결 상태들의 집합, $Y$를 $V_B$ 내부 간선들의 연결 상태들의 집합으로 놓고, 함수 $f(X,Y)$를 $G$가 two edge-disjoint spanning tree를 가질 때 1, 그렇지 않을 때 0으로 두자. $f(X,Y)$는 잘 정의된 함수이다.

먼저, two edge-disjoint spanning tree $T_1, T_2$가 존재하는 경우를 생각해보자.  $T_i$에 $S$의 각 edge가 포함되는지 여부 및 $T_i$ 중 $V_A$ 내부 간선만 생각했을 때 $V_A$의 component, 마찬가지로 생각했을 때 $V_B$의 component 정보를 certificate으로 하면 $\tilde{O}(n + \lvert S \rvert)$의 cost를 가지는 proof system이 된다. 한편, $S$의 어떤 간선의 양 끝점도 되지 않는 정점의 경우 component 정보를 주지 않아도 된다. 즉, $N(f) = \tilde{O}(\lvert S \rvert)$가 성립한다.

two edge-disjoint spanning tree $T_1, T_2$가 존재하지 않는 경우를 살펴보자. 

**Nash-Williams theorem[Williams61].** $t$개의 edge-disjoint spanning tree가 존재하는 것은 임의의 $V$의 partition $V_1, \cdots, V_k$에 대해 crossing edge가 $t(k-1)$이상인 것과 동치이다.

위 theorem에 의해, $\neg f$의 certificate는 crossing edge가 $t$이하인 partition $V_1, \cdots, V_t$로 줄 수 있다. $f$의 경우와 마찬가지로, $S$의 어떤 간선의 양 끝점도 되지 않는 정점의 경우 partition 정보를 주지 않아도 된다. 따라서, $N(\neg f) = \tilde{O}(\lvert S \rvert)$가 성립한다.

이제 Corollary 9를 적용하면 $\tilde{O}(\lvert S \rvert ^2)$ 비트의 교환으로 $G$가 2-disjoint spanning tree를 가지는지 여부를 Alice와 Bob이 판정하는 protocol이 존재함을 알 수 있다. 그러나 $S$의 원소의 개수가 많지 않을 때 빠르게 2-disjoint spanning tree를 판정하는 communication protocol을 실제로 construct하는 것은 직관적으로 봤을 때 쉽지 않아보이기 때문에, 이는 counterintuitive한 결과라 볼 수 있다.


## Future talk
Communication complexity와 lower bound에 대해 재미있는 내용이 많은데, 그 중 하나는 communication complexity의 lower bound를 이용하여 distributed setting에서 필요한 rounds의 lower bound를 유도하는 것이다. 예를 들어, CONGEST model에서 그래프의 diameter를 구하는 알고리즘은 $\tilde{\Omega}(n)$의 lower bound를 가짐을 communication model로부터 유도할 수 있다 [FHW12]. 이와 관련한 내용도 가능하면 추후 포스트에서 다룰 예정이다.

# Reference

- [FHW12] Frischknecht, Silvio and Holzer, Stephan and Wattenhofer, Roger. Networks Cannot Compute Their Diameter in Sublinear Time, 2012. Proceedings of the Twenty-Third Annual ACM-SIAM Symposium on Discrete Algorithms, Pages 1150–1162.
- [Williams61] C. St.J. A. Nash-Williams, Edge-Disjoint Spanning Trees of Finite Graphs, Journal of the London Mathematical Society, Volume s1-36, Issue 1, 1961, Pages 445–450.
- UCLA CS289 Communication Complexity(https://web.cs.ucla.edu/~sherstov/teaching/2012-winter/)