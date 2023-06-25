---
layout: post

title: "Faster Matroid Intersection - Part 1"

date: 2023-06-25

author: ainta

tags: [matroid]
---

## 0. Introduction

필자는 [Matroid를 소개하는 글](http://infossm.github.io/blog/2019/05/16/introduction-to-matroid/) 및 [Matroid Intersection을 소개하는 글](http://infossm.github.io/blog/2019/06/18/Matroid-Intersection/)을 2019년에 작성한 바 있다. Matroid Intersection은 해당 글에도 나와 있듯 다양한 문제들을 해결할 수 있는 툴이 되며, 이에 따라 많은 연구가 진행되었다. 특히, 2019년을 기점으로 하여 Matroid Intersection을 어떻게 하면 빠르게 할 수 있을지에 대해 여러 논문의 결과가 발표되기 시작했다. 대표적인 예시로 다음과 같은 4가지 논문이 존재하고, 앞으로는 다음 논문들의 결과에 대해 다룰 것이다.

- [A note on Cunningham's algorithm for matroid intersection(2019)](https://arxiv.org/pdf/1904.04129.pdf)

- [Faster Matroid Intersection(2019)](https://arxiv.org/pdf/1911.10765.pdf)

-  [Breaking the Quadratic Barrier for Matroid Intersection](https://arxiv.org/pdf/2102.05548.pdf) 

- [Breaking $\tilde{O}(nr)$ for Matroid Intersection(2021)](https://arxiv.org/pdf/2105.05673.pdf)


결과에 대해 살펴보기 전에, 간단하게 Matroid 및 Matroid Intersection Problem의 정의만 짚고 넘어가자.


**Definition(Matroid). matroid $\mathcal{M} = (V,  \mathcal{I})$ 에서 $S$는 유한집합, $\mathcal{I} \subset 2^V$ 는 독립집합(independent set)들의 collection이다. 이 때, $I$는 다음 세 가지 조건을 만족하여야 한다.**

1. $\phi \in  \mathcal{I}$
2. $Y \subset X, X \in  \mathcal{I} \Rightarrow Y \in  \mathcal{I}$ 
3. $X, Y \in  \mathcal{I}, \lvert X \rvert < \lvert Y \rvert$ 이면 $X + y \in  \mathcal{I}$ 를 만족하는 $y \in Y \setminus X$가 존재



**Definition(Matroid Intersection Problem). 두 matroid $\mathcal{M_1} = (V,  \mathcal{I}_1), \mathcal{M_2} = (V,  \mathcal{I}_2)$에 대해, $S \in \mathcal{I_1} \cap \mathcal{I_2}$를 만족하는 $S \subset V$ 중 가장 크기가 큰 것을 구하시오. (maximum cardinality common independent set)**


## 1. Independent Oracle Query For Matroid Intersection

빠른 Matroid Intersection을 만들고 싶다고 하자. 이 때 Matroid Intersection 알고리즘의 효율성을 어떻게 측정할 수 있을까? 예를 들어, forest를 이루는 edge set들을 independent하다고 하는 graphic matroid의 경우 간선 하나를 추가하거나 제거했을 때 independent함을 알기 위해서 Link-Cut Tree와 같은 자료구조를 통해 시간복잡도를 줄일 수 있을 것이다. 

그러나 일반적인 Matroid에 대해 모두 이 방식을 적용할 수는 없다. 일반적인 Matroid에 대해 정보를 얻는 가장 general한 방법은 집합에 대해 그 집합의 independence에 대한 정보를 얻는 것이라 볼 수 있다. 따라서, 가장 일반적인 경우를 생각하고 다음과 같은 세팅을 생각할 수 있다.

Matroid $\mathcal{M} = (V, \mathcal{I})$ 에 대한 정보는 다음과 같은 Query로만 얻을 수 있다는 세팅을 가정한다:

**Independent Query Oracle**. 이 Oracle은 집합 $S \subset V$를 입력으로 받아 $S \in \mathcal{I}$가 true인지 false인지를 리턴한다.

앞으로는 위 세팅을 가정하고 오로지 Independent Query Oracle만을 이용해 Matroid Intersection을 해결하는 문제를 살펴볼 것이고, 알고리즘의 효율성 역시 Oracle을 얼마나 많이 사용하여 matroid intersection problem을 해결할 수 있는지로 측정할 것이다.

$\lvert V \rvert = n$, common independent set의 최대 크기를 $r$이라 할 때, 앞에서 열거한 논문들의 결과를 요약하면 아래와 같다.

- [A note on Cunningham's algorithm for matroid intersection(2019)](https://arxiv.org/pdf/1904.04129.pdf): exact algorithm with $O(nr \log r)$ oracles 

- [Faster Matroid Intersection(2019)](https://arxiv.org/pdf/1911.10765.pdf): $1-\epsilon$ approximation by using $\tilde{O}(n^{1.5}/\epsilon^{1.5})$ oracles

- [Breaking $\tilde{O}(nr)$ for Matroid Intersection(2021)](https://arxiv.org/pdf/2105.05673.pdf): $\tilde{O}(nr^{3/4})$ oracles for exact algorithm

단, 위에서 $\tilde{O}$는 poly-logarithmic factor가 추가적으로 붙을 수 있음을 뜻한다. 즉 $O(nr \log r)$ 알고리즘이라면 $\tilde{O}(nr)$ 알고리즘이다.

## 2. Exchange Graph for Matroid Intersection

우리의 goal은 $\mathcal{M}_1 = (V, \mathcal{I}_1)$, $\mathcal{M}_2 = (V, \mathcal{I}_2)$의 maximum cardinality common independent set을 구하는 것이다.

**Exchange Graph.**
Matroid $\mathcal{M}_1 = (V, \mathcal{I}_1)$, $\mathcal{M}_2 = (V, \mathcal{I}_2)$ 와 $S \in \mathcal{I}_1 \cap \mathcal{I}_2$ 에 대해, $S$에 대한 exchange graph는 다음과 같이 정의되는 directed graph $G(S) = (V \cup \left\\{ s, t \right\\}, E)$이다:

- $E = E_1 \cup E_2 \cup E_s \cup E_t$
- $E_1 = \left\\{ (u,v) \mid u \in S, v \in V \setminus S, S-u+v \in \mathcal{I}_1 \right\\}$
- $E_2 = \left\\{ (v,u) \mid u \in S, v \in V \setminus S, S-u+v \in \mathcal{I}_2 \right\\}$
- $E_s = \left\\{ (s,v) \mid v \in V \setminus S, S+v \in \mathcal{I}_1 \right\\}$
- $E_t = \left\\{ (v,t) \mid v \in V \setminus S, S+v \in \mathcal{I}_2 \right\\}$

**Fact**. Exchange graph $G(S)$의 shortest $(s,t)$-path $(s, v_1, \cdots v_{l-1}, t)$ 에 대해, $S' = S + v_1 - v_2 + \cdots - v_{l-2} + v_{l-1} \in \mathcal{I}_1 \cap \mathcal{I}_2$ . 또한, $(s,t)$-path 가 존재하지 않는 경우 $S$는 common independent set of maximum size.

따라서, 다음과 같은 알고리즘으로 Matroid Intersection 문제를 해결할 수 있다.

**Algorithm 1.**
- intially, $S = \phi$ 
- while **AugmentingPathExistsInExchangeGraph($S$)** found $(s,t)$-path:
	- update $S$ as $S + v_1 - v_2 + \cdots - v_{l-2} + v_{l-1}$

이 떄 서브루틴 **AugmentingPathExistsInExchangeGraph**은 결국 $G(S)$에서 $s$로부터 출발하는 BFS를 수행하여 $(s,t)$-path를 찾는 것이다.

BFS를 수행할 때 $E_s, E_t$ 의 edge에 해당하는 쿼리에는 $O(V)$ 번의 Independent Oracle이 소모된다.
그러나 BFS를 수행할 때 $E_1, E_2$ 의 edge에 해당하는 쿼리는 언뜻 $S$와 $V \setminus S$ 사이 edge들의 정보를 알아내야 하므로 $\lvert S\rvert  \times \lvert V \setminus S\rvert $번 정도의 횟수를 필요로 하는듯 보인다. 과연 그런가?

## 3. Tools for reducing independent oracle queries

**Tool 1. Edge Search via Binary Search.**

**Tool 1a.** 알고리즘 $FindOutEdge$ 가 존재하여, matroid $\mathcal{M} = (V, \mathcal{I})$ 와 $S \in \mathcal{I}$, $v \in V \setminus S, B \subset S$  가 주어져있을 때 $O(\log \lvert B\rvert  )$ 번의 independent oracle을 통해 다음을 수행한다.
- $S - u + v \in \mathcal{I}$를 만족하는 $u \in B$를 리턴하거나 그런 $u$가 존재하지 않음을 알림


위 조건을 만족하는 알고리즘 $FindOutEdge$ 는 다음과 같이 이루어진다.

먼저, 다음과 같은 array $A_S$ 를 생각해보자.
- $A_S$에는 $S$의 각 원소가 한 번씩 나타난다.
- $A_S$의 마지막 $\lvert B\rvert $ 개 원소는 $B$의 원소들이다.

이 때, $A_S[1 \cdots i] \cup \left\\{ v \right\\} \notin \mathcal{I}$ 을 만족하는 최소의 index $i$ 를 생각하자. ($S + v \in \mathcal{I}$ 인 경우는 그냥 $v$를 추가하면 되므로 고려할 필요가 없다)

- Case 1. 만약 $i \le \lvert S\rvert  - \lvert B\rvert $ 이면 조건을 만족하는 $u$가 존재하지 않음을 리턴한다.
- Case 2. $i > \lvert S\rvert  - \lvert B\rvert $ 이면 $A_S[i] \in B$ 이다. 이 때,  $u = A_S[i]$ 로 놓은 후 $S - u + v \in \mathcal{I}$ 이면 $u$를 리턴, 그렇지 않으면 조건을 만족하는 $u$가 없음을 리턴한다.

Case 1은 자명하고, Case 2가 올바르게 동작하는 이유는 Matroid에 대한 다음 property를 근거로 한다:

**Unique Circuit Property**. Matroid에서, minimal dependent set들을 **circuit**이라 한다. 만약 independent한 set $T$에 대해 $T + v$ 가 dependent해졌다면 $T+v$의 circuit은 유일하고, 이는 $v$를 포함한다. 

Unique Circuit Property에 대한 간단한 증명:

idea: $M$의 서로 다른 circuit $X, Y$에 대해 $e \in X \cap Y$ 라 했을 때 $X \cup Y \setminus \left\\{ e \right\\}$ 가 dependent함을 보이면 충분하다 .

$X \cup Y \setminus \left\\{ e \right\\} \in \mathcal{I}$ 이라고 가정하자 (귀류법).  $f \in X \setminus Y$ 을 잡았을 때, $X-f \in \mathcal{I}$ 이다. $X-f$를 $X \cup Y$ 의 maximal independent set까지 extend한 결과를 $Z$라 하자. $Y$가 dependent하므로 $Z$는 $Y$를 포함할 수 없고, 또한 $X$가 circuit이므로 $Z$는 $f$ 역시 포함할 수 없다. 따라서, $\lvert X\rvert  \le \lvert X \cup Y\rvert  - 2 < X \cup Y \setminus \left\\{e \right\\}$이고 이는 $Z$의 maximality에 모순이다. $_\blacksquare$

Unique Circuit Property에 의해, $S + v$의 unique한 circuit은 $A_S[i]$를 포함하고 Case 2에서 $FindOutEdge$는 올바르게 작동한다.

**Tool 1b.** 비슷한 방법으로, 다음과 같은 알고리즘 $FindInEdge$ 를 생각할 수 있다.
알고리즘 $FindInEdge$ 가 존재하여, matroid $\mathcal{M} = (V, \mathcal{I})$ 와 $S \in \mathcal{I}$, $T \subset S, A \subset V \setminus S$  가 주어져있을 때 $O(\lvert A\rvert  + \lvert U\rvert  \log \lvert B\rvert )$ 번의 independent oracle을 통해 다음을 수행한다.
- $U = \left\\{ u \mid S - v + u \in \mathcal{I} \text{ for some } v \in T \right\\}$ 및 $u$의 각 원소에 대해 $S-v+u \in \mathcal{I}$인 $v \in T$ 리턴

알고리즘 $FindInEdge$ 는 다음과 같이 수행된다:
1. $A$의 각 원소 $a$에 대해, $S+a$와 $S - T + a$가 independent한지 먼저 체크한다.
2. $S+a$가 independent하다면 $a \in U$이다. $S - T + a$가 independent하지 않은 $a$는 고려할 필요가 없다.
3. $S-T+a$만 independent한 경우, prefix가 $S-T$이고 그 뒤에 $T$의 원소들이 있는 array $B_T$에서 $B_T[1\cdots i] \cup \left\\{ a \right\\}$ 가 처음으로 dependent인 $i$를 binary search로 찾을 수 있다. 그 때, unique circuit property에 의해 $S - B_T[i] + a$는 independent하다.


## 4. Faster Algorithm for Matroid Intersection

일반적으로는 Hopcroft-Karp 알고리즘과 비슷한 방법으로 Matroid Intersection을 빠르게 할 수 있다.

즉, exchange graph $G(S)$에서 BFS를 하여 $s$로부터 각 vertex까지의 distance를 구해놓은 후, distance가 1 커지는 방향으로의 edge만 생각하여 Blocking Flow를 흘리는 방식으로 문제를 해결하면 BlockFlow phase는 최대 $O(\sqrt{n})$ 번만 이루어짐을 증명할 수 있다.

자세히 analyze해보면 이 때 필요한 oracle call의 횟수는 $\tilde{O}(nr^{1.5})$ 이다.

그러나, 실은 그냥 BFS를 하고 update를 하는 naive한 방식으로도 더 효율적으로 matroid intersection을 구할 수 있다. Matroid intersection의 최대 크기를 $r$에 대해, $O(nr \log r) = \tilde{O}(nr)$번의 oracle에 matroid intersection을 구할 수 있다. 이는 쿼리 수에 대한 상당한 감축이라 볼 수 있다. 여기서는 그 방법을 소개하고자 한다.

먼저, 다음 lemma가 성립한다.

**Lemma 1(Cunningham).** $r$ 를 maximum size of common independent set이라 하고, $S$ 는 크기가 $r$보다 작은 common independent set이라 할 때, 길이 $2\lvert S\rvert /(r-\lvert S\rvert ) + 2$ 이하의 augmenting path가 존재한다.

Lemma 1이 성립하는 이유를 간단히 알아보고 넘어가자. 이는 Hopcroft-Karp 알고리즘에서 쓰이는 논리와 거의 유사하다. 먼저, 다음이 성립한다:

**Lemma 2.** $S, S'$ 를 $\mathcal{M}_1, \mathcal{M}_2$ 의 common independent set이라  하고, $\lvert S'\rvert  > \lvert S\rvert$라 하자. 이 때, $J$의 exchange graph는 $\lvert S'\rvert  - \lvert S\rvert$개의  vertex-disjoint augmenting path를 포함한다. 여기서 vertex-disjoint란 $s$, $t$를 제외하고 모든 정점이 path들에서 최대 1번 사용된다는 의미이다.

Lemma 2에서 $r-\lvert S\rvert$ 개의 vertex-disjoint augmenting path가 존재하므로 길이 $2\lvert S\rvert /(r-\lvert S\rvert )$ + 2 이하의 augmenting path가 존재한다. 따라서 Lemma 1이 성립한다.

한편, matroid intersection에서 구하는 augmenting path들의 길이의 총 합은 $O(r \log r)$가 됨을 Lemma 1로부터 쉽게 알 수 있다.

그러면 이제 Matroid Intersection 문제를 해결하는 다음 알고리즘을 생각해볼 수 있다:

Input: matroid $\mathcal{M_1} = (V, \mathcal{I}_1), \mathcal{M_2} = (V, \mathcal{I}_2)$, 두 matroid의 common independent set size의 maximum은 $r$.

Output: Common independent set $S$ with maximum cardinality 

1. $S = \phi$ 로 initialize
2. $S$의 exchange graph에 대한 BFS
	1. $V_d$를 $s$에서 distance가 $d$인 집합이라 하고, 현재 $V_0, \cdots V_d$까지 계산된 상태이며, 현재까지 방문한 정점의 집합 $V_0 \cup \cdots \cup V_d$ 중 $S$에 속하는 것들을 $A$, $V \setminus S$에 속하는 것들을 $B$라 하자.
	2. Case I. d가 홀수인 경우
		1. $d$가 홀수인 경우, $V_d$는 $B \subset V \setminus S$ 의 부분집합이다. $v \in V_d$에 대해,
		2. $U = \left\\{ u \in S \setminus A : S - u + v \in \mathcal{I}_2 \right\\}$로 두자.
		3. $U$의 원소들은 $V_{d+1}$에 포함된다. 이에 $U$의 원소들을 $FindOutEdge$를 통해 $O(\lvert U\rvert \log \lvert S\rvert )$번의 oracle이 구할 수 있다.
		4. 각 $v$에 대해 $U$를 구하고 $U$의 원소를 $A$에 포함시키는 것을 반복하면 $V_{d+1}$을 구할 수 있다.
	3. Case II. d가 짝수인 경우
		1. $d$가 짝수인 경우, $V_d$는 $A \subset S$ 의 부분집합이다.  
		2. $U = \left\\{ u \in V \setminus S \setminus B : S - v + u \in \mathcal{I}_1 \text{ for some } v \in V_d \right\\}$로 두자.
		3. $V_{d+1} = U$가 성립한다. 따라서, $U$의 원소들을 $FindInEdge$를 통해 $O(\lvert V \setminus S \setminus B\rvert  + \lvert U\rvert \log \lvert S\rvert )$번의 oracle에 구할 수 있다. ($\log \lvert S\rvert $인 이유는 $\lvert V_d\rvert  \le \lvert S\rvert $이기 때문)
	4. 위 step들로 $V_d$에 $t$가 포함되거나 공집합이 될 때까지 $d$를 1씩 증가시킨다.
3. $t \in V_d$이면 augmenting path로 $S$를 업데이트한다.

한 번의 BFS에 몇 번의 Oracle이 필요한지 분석해보자.
먼저, 하나의 원소는 BFS동안 최대 한 번 $U$에 포함되게 된다. 이를 이용하면
- Case I에서 사용되는 oracle의 횟수는 $O(n \log \lvert S\rvert ) = O(n \log r)$
- Case II에서 사용되는 oracle의 총 횟수는 $O(nl + n\log\lvert S\rvert ) = O(nl + n\log r)$, 단 $l$은 $dist(s,t)$의 길이
- 그 외에 $s$와 $t$에 관한 edge를 탐색하는데 oracle의 총 횟수는 $O(n)$

한편, 모든 BFS에 대한 $l$의 합은 앞서 lemma 1에 의해 $O(r \log r)$임을 보였으므로,
Matroid intersection에 사용되는 Independent Oracle Query의 개수는 $r \cdot O(n \log r) + O(n \cdot r \log r) = O(nr \log r)$가 된다. 


## 5. Even Faster

지금까지 $\tilde{O}(nr)$번의 independent oracle로 matroid intersection problem을 해결하는 방법을 알아보았다. 이는 [A note on Cunningham's algorithm for matroid intersection(2019)](https://arxiv.org/pdf/1904.04129.pdf) 논문에서 소개된 결과이다.

현재 SOTA이자 우리가 얻고자 하는 최종 횟수는 [Breaking $\tilde{O}(nr)$ for Matroid Intersection(2021)](https://arxiv.org/pdf/2105.05673.pdf) 논문의 $O(nr^{3/4})$이다. 이 알고리즘이 동작하는 원리를 큰그림에서 간단히 설명하면 다음과 같다.

1. [Faster Matroid Intersection(2019)](https://arxiv.org/pdf/1911.10765.pdf)의 $\tilde{O}(n^{1.5}/\epsilon^{1.5})$ oracle을 통한 $1-\epsilon$ approximation을 $O(n\sqrt{r \log r}/\epsilon)$로 개선시킨 알고리즘을 이용하여, $\epsilon = r^{-1/4}$로 놓고 크기 $r-r^{3/4}$ 이상의 common independent set $S$를 구한다.

2. 앞서 살펴본 방법을 이용하여 $G(S)$에서 $(s,t)$의 최단경로가 $r^{3/4}$를 넘지 않을 때 까지 augmenting path를 구하여 $S$를 업데이트하는 과정을 반복한다.

3. [Breaking the Quadratic Barrier for Matroid Intersection](https://arxiv.org/pdf/2102.05548.pdf) 논문에서 소개된, Exchange graph에서 많은 edge와 연결된 Heavy node와 적은 edge와 연결된 Light node라는 개념을 도입하여 하나의 augmenting path를 $O(n\sqrt{r} \log n)$에 구하는 알고리즘이 있다. 이를 이용해 (s,t)-path가 존재하지 않을 때 까지 augmenting path를 구해 $S$를 업데이트한다.

1번 과정에서 $\tilde{O}(nr^{0.5}/\epsilon) = \tilde{O}(nr^{3/4})$, 2번 과정에서 $\tilde{O}(nr^{3/4})$번의 oracle을 사용함을 알 수 있고, 2번 과정까지 마친 후 $S$의 크기는 앞서 살펴본 Lemma 1(Cunningham)에 의해 $r- \lvert S \rvert$가 $O(r^{1/4})$임을 알 수 있다.

따라서, 3번 과정에서 augmenting path를 $O(r^{1/4})$번 구하므로 $\tilde{O}(nr^{3/4})$ 번의 oracle을 사용한다.

이에 1,2,3번에서 사용한 총 oracle의 수 역시 $\tilde{O}(nr^{3/4})$로 bound된다.

다음 글에서는 fast matroid intersection algorithm에서 가장 흥미로운 부분이라고도 할 수 있는 subquadratic $1-\epsilon$ approximation algorithm에 대해 알아볼 것이다.


## 6. Reference

- [Ngu19] Huy L. Nguyen. A note on Cunningham's algorithm for matroid instersection(2019)

- [CLS+19] Deeparnab Chakrabarty, Yin Tat Lee, Aaron Sidford, Sahil Singla, and Sam Chiu-wai Wong. Faster matroid intersection. Faster Matroid Intersection(2019)

- [BvdBMN21] Joakim Blikstad, Jan van den Brand, Sagnik Mukhopadhyay, and Danupon Nanongkai.
Breaking the quadratic barrier for matroid intersection(2021)

- [Bli21]  Joakim Blikstad. Breaking O(nr) for Matroid Intersection(2021)