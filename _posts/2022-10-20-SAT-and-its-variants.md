---
layout: post
title:  "SAT problem의 변형과 Schaefer’s Dichotomy Theorem"
date:   2022-10-20
author: ainta
tags: [algorithm, complexity-theory]
---


## Boolean Formula, SAT

$(x \lor \neg y) \land (\neg x \lor z)$와 같은 식을 **Boolean formula**라 한다. Boolean formula의 변수(variable)는 True/False (또는 1/0이라고도 한다)의 두가지 값만을 가질 수 있다. 위 boolean formula의 variable은 $x, y, z$이다. 각 연산자에 대해 알아보면 $\lor$ 와 $\land$는 각각 logical OR(disjunction)/ logical AND (conjunction)를 나타내고, $\neg$(negation)는 NOT을 나타내는 연산자로 피연산자가 True였다면 False로, False였다면 True로 바꿔주는 역할을 한다. 식에서 각각의 항 $x, \neg y, \neg x, z$는 **literal**이라 한다. $x, y$와 같이 negation이 붙지 않은 literal은 **positive literal**, $\neg x, \neg z$ 등은 **negative literal**이라 한다.

다음과 같은 문제를 생각해보자.
- 주어진 논리식 $\varphi$에 대해, 각 변수에 true/false 를 적당히 대입하여 최종 결과가 true가 되도록 할 수 있는가?

위 문제를 SAT (Satisfiability problem)이라고 하며, 주어진 논리식이나 구하는 해에 조건이 추가적으로 붙는 여러가지 SAT의 변형 문제가 존재한다. 그리고 다음과 같은 사실이 널리 알려져 있다.

> SAT is NP-Complete (Cook–Levin theorem)

즉, SAT를 다항식 시간에 풀 수 있는 알고리즘은 현재로서는 존재하지 않으며, 앞으로도 나올 가능성이 거의 없다.

Boolean formula 중에서도 우리가 중점을 두고 살펴볼 formula는 CNF이다. CNF가 무엇인지, 그리고 이와 유사한 형태인 DNF가 무엇인지도 먼저 살펴보자.

- $\varphi = (x \lor y \lor \neg z) \land (\neg z \lor \neg y) \land (x \lor z)$ 와 같이 OR들의 AND 형태를 가지는 경우, $\varphi$를 **CNF**(Conjunctive Normal Form)이라 한다.
- $\varphi = (x \land y \land \neg z) \lor (\neg z \land \neg y) \lor (x \land z)$ 와 같이 AND들의 OR 형태를 가지는 경우, $\varphi$를 **DNF**(Disjunctive Normal Form)이라 한다.
- CNF에서 $(x \lor y \lor \neg z)$ 처럼 $\land$로 연결되는 각각의 부분을 **clause**라 한다. DNF에서도 마찬가지로 clause를 정의할 수 있다.

SAT에서 입력으로 주어지는 boolean formula가 CNF여야 한다는 조건을 추가적으로 붙인 문제를 **CNF SAT**이라 한다. 마찬가지로, 입력이 DNF라면 **DNF SAT** 문제가 된다. CNF SAT과 DNF SAT에 대해 다음이 알려져 있다.

> CNF SAT is NP-Complete.
> 
> DNF SAT is in P.

SAT는 CNF SAT로 polynomial reduction이 가능함이 알려져 있으며, 이에 따라 CNF SAT은 NP-Complete이다. 한편, 주어진 DNF에 대해 각 clause들 중 하나라도 true가 되도록 만들 수 있다면 DNF는 satisfiable하므로 각 clause가 true가 될 수 있는지 판정할 수 있으면 충분하다. 만약 clause 내에 $x \land \neg x$와 같이 동일한 변수에 대한 positive/negative literal이 모두 있다면 그 clause는 true가 될 수 없음이 자명하다. 그렇지 않은 경우, positive만 있는 variable에는 true, negative만 있는 variable에는 false를 대입하면 해당 clause는 true가 되어 결국 전체 DNF의 결과가 true가 된다. 따라서, DNF SAT을 해결하는 linear time algorithm이 존재한다.

CNF formula $\varphi$가 주어질 때, 이와 동치인 DNF formula $\psi$를 만드는 것이 가능할까? 이에 대한 대답은 '그렇다' 이다. $\land$ 과 $\lor$ 에 대해 분배법칙이 성립하기 때문에, CNF formula가 주어지면 다음 예와 같이 전개가 가능하다:
$$(x_1 \lor x_2) \land (x_3 \lor x_4) = (x_1 \land (x_3 \lor x_4)) \lor (x_2 \land (x_3 \lor x_4)) = (x_1 \land x_3) \lor (x_1 \land x_4) \lor (x_2 \land x_3) \lor (x_2 \land x_4)$$

그런데 CNF를 동치인 DNF로 변환하는 것이 항상 가능하다면, 왜 DNF SAT은 linear time에 해결 가능한 반면 CNF SAT은 다항식 시간 내에 해결하는 알고리즘이 알려져 있지 않은 NP-Complete 문제일까? 이에 대해서는 $(x_1 \lor y_1) \land (x_2 \lor y_2) \land ... \land (x_n \lor y_n)$ 라는 식을 살펴보면 그 이유를 추측할 수 있다. 해당 CNF를 앞서 본것과 같이 전개하여 DNF로 바꾸면 $(x_1 \land x_2 \land ... \land x_n)$ 부터 $(y_1 \land y_2 \land ... \land y_n)$까지 $2^n$개의 clause를 가지는 DNF가 만들어진다. 즉, input인 CNF formula의 size의 다항식으로 표현할 수 있는 size의 DNF formula가 아닌 exponential한 크기의 DNF formula로 변환이 되는 것이다. 꼭 이렇게 전개하지 않더라도, 해당 CNF와 equivalent한 DNF formula는 적어도 $2^n$개의 clause를 가짐이 증명되어 있다. 즉, CNF formula를 DNF로 바꾸어 풀더라도 polynomial time algorithm이 되지는 않는다는 것이다.

## Variants of SAT

지금까지는 boolean formula와 CNF, DNF에 대해 알아보고 해당 form의 instance가 주어지는 satisfiability problem이 P인지, NP-Complete인지에 대해 알려진 결과를 알아보았다. 지금부터는 보다 다양한 SAT problem의 변형에 대해 다뤄볼 것이다. 먼저, 앞으로 다룰 모든 SAT problem의 input은 CNF이고 boolean formula에 대해 특별한 말이 없으면 CNF라고 가정할 것이다. 일반적인 boolean formula에 대해 linear한 크기의 CNF로 변형하는 것이 가능하기 때문에 (Tseytin transformation) 이는 무리한 가정이 아니다.

먼저 알아볼 것은 $a$SAT 이다. $a$SAT의 예로는 2SAT, 3SAT이 있으며, $a$SAT의 input의 조건은 모든 clause가 최대 $a$개의 literal을 가지는 것이다. $a$SAT에 대해 다음과 같은 사실이 알려져 있다.

- 2SAT is in P.
- for all $k \ge 3$, $k$SAT is NP-Complete.

2SAT의 경우 directed graph로 표현하여 linear time에 해결하는 알고리즘이 존재한다. 4SAT, 5SAT 등의 경우 가능한 input이 3SAT의 input을 포함하으로 3SAT이 NP-Complete이면 3보다 더 큰 $k$에 대한 $k$SAT은 자명하게 NP-Complete일 것이다. 그리고 CNF SAT이 3SAT으로 polynomial reduction이 가능하기 때문에 3SAT은 NP-Complete이다.

그러면 $a$SAT과 관련된 다른 변형들을 더 알아보자. 아래 변형들은 모두 SAT의 input instance인 boolean formula에 조건이 추가적으로 들어간 경우이다.

- E$a$SAT: 모든 clause 가 정확히 $a$개의 literal을 가진다.
- EU$a$SAT: E$a$SAT의 조건을 만족하고, 한 clause의 literal들은 서로 중복되지 않는다.
- $a$SAT-$b$: $a$SAT의 조건을 만족하고, 식에 각각의 variable들은 $b$번 이하로 등장한다.
- $a$SAT-E$b$: $a$SAT의 조건을 만족하고, 식에 각각의 variable들은 정확히 $b$번 등장한다.
- E$a$SAT-$b$, EU$a$SAT-$b$, E$a$SAT-E$b$, EU$a$SAT-E$b$ 등의 변형을 생각해볼 수 있다.
- Monotone: 각 clause의 literal들이 $x \lor y \lor z, \neg x \lor \neg y$처럼 모두 positive거나 모두 negative이다. 
- Positive: 각 clause의 literal들이 모두 positive이다.

앞서 3SAT은 NP-Complete였다. 3SAT-3 의 input은 3SAT의 input의 부분집합으로, 각 variable이 최대 3번 등장할 수 있다는 조건이 추가로 붙는다. 그렇다면 3SAT-3은 어떤 complexity class에 포함되는 문제일까?

- 3SAT-3 is NP-Complete.

**Proof.** 3SAT $\le_p$ 3SAT-3 (3SAT이 3SAT-3으로 polynomial reduction된다는 뜻)을 증명하면 3SAT-3이 NP-Complete임을 보일 수 있다.  
3SAT problem의 instance $\varphi$에 대해, 이와 동치인 3SAT-3 instance를 만들어보자. $\varphi$의 모든 variable이 3번 이하로 나온다면 이미 조건을 만족하므로 끝. $k > 3$번 등장하는 variable $x$에 대해 마지막 $k-1$번의 등장을 $y_1, .., y_{k-1}$로 바꾸고 $\varphi$ 마지막에 추가적으로 $x \lor \neg y_1$, $y_i \lor \neg y_{i+1}$, $y_{k-1} \lor \neg x$ clause들을 추가해주면 이는 원래 $\varphi$와 동치가 된다. 새로 만들어진 formula에서 $x$와 $y_i$는 최대 3번만 등장하므로 이 과정을 한번 할 때마다 3번 초과로 등장하는 variable의 개수가 하나 줄고, 최종적으로는 모든 variable이 3번 이하로 등장하는 equivalent한 formula를 얻을 수 있다. 이 formula의 크기가 처음 input $\varphi$의 크기에 linear함은 자명하므로, 3SAT은 3SAT-3으로 polynomial reduction이 된다. 따라서, 3SAT-3은 NP-Complete.

EU3SAT-3은 3SAT-3에서 한 clause의 literal들은 서로 다르다는 조건이 붙은 문제이다.

- EU3SAT-3 is in P.

**Proof.** 다음과 같은 알고리즘으로 EU3SAT-3 문제를 해결할 수 있다.

주어진 boolean formula에 대해, 각 clause에 해당하는 vertex를 왼쪽, variable에 해당하는 vertex를 오른쪽에 배치하여 clause에 variable이 들어갈 경우 edge를 연결하는 bipartite graph를 생각하자. clause vertex들은 degree가 정확히 3이고 variable vertex들은 degree가 최대 3이다. Hall's theorem에 의하면 해당 bipartite graph에서 모든 clause를 연결된 variable에 서로 다르게 matching하는 방법이 존재한다. EU3SAT이므로 clause에 매칭된 variable이 있는 literal은 정확히 한 개이고, 이를 true로 하도록 variable의 값을 설정하면 각 clause가 모두 true가 되어 formula 자체가 true가 된다. 즉, EU3SAT-3은 항상 satisfiable하다.


그러면 $a$SAT과 관련된 다른 변형들을 더 알아보자. 아래 변형들은 앞서 살펴본 변형들과 달리 input boolean formula가 아니라 solution에 조건이 추가적으로 붙는 경우이다. 즉, boolean formula를 true로 만드는 것 외에 다른 조건이 더 붙는다.

- NAE-: 모든 clause가 true인 literal과 false인 literal을 모두 포함해야 한다. 즉, (x \lor y) 에서 x, y를 모두 true로 주면 SAT의 해이지만 NAE-SAT의 해는 아니다.
- 1-IN-: 각 clause에서 정확히 하나의 literal만 true가 되어야 한다.

1-IN-3SAT, NAE-3SAT은 NP-Complete임이 알려져 있으며, positive 조건도 붙은 Positive 1-IN-3SAT, Positive NAE-3SAT까지도 NP-Complete이다. 이 중 다음 예시의 증명을 소개한다.

- 3SAT $\le_p$ 1-IN-3SAT. Therefore, 1-IN-3SAT is NP-Complete.

**Proof.** clause $(x \lor y \lor z)$에 대해 생각해보자. $(\neg x \lor a \lor b) \land (b \lor y \lor c) \land (c \lor d \lor \neg z)$ 가 1-IN-3SAT satisfiable 인 것은 $x, y, z$중 하나 이상이 true인 것과 동치이다. 따라서 $(x \lor y \lor z)$를 $(\neg x \lor a \lor b) \land (b \lor y \lor c) \land (c \lor d \lor \neg z)$로 변환하여 동치인 1-IN-3SAT instance를 생성할 수 있다. 이는 $x, y, z$가 positive literal이든 negative literal이든 관계없이 적용할 수 있으며, literal이 2개인 $(x \lor y)$와 같은 clause는 $(x \lor y \lor y)$로 생각할 수 있으므로 모든 3SAT instance를 equivalent한 1-IN-3SAT instance로 변환할 수 있다. 따라서 3SAT $\le_p$ 1-IN-3SAT.

## Schaefer’s Dichotomy Theorem

앞서 SAT와 그 변형들이 P인지 NP-Complete인지에 대해 알아보았다. 이를 보다 체계적으로 분석하기 위해 Relation이라는 개념을 도입하자.

**Definition.** A **relation** on $m$ variables are a formula on those variables. i.e. a function from $(0,1)^m$ to $(0,1)$ (블로그에서 중괄호를 쓰면 오류가 자주 나서 집합을 소괄호로 표기하였음에 유의).

Relation의 정의는 위와 같다. 이것이 무엇을 뜻하는지에 대해 예를 통해 알아보자.

instance of 1-IN-3SAT $\varphi = (x_1 \lor x_2 \lor x_3) \land (x_1 \lor x_3 \lor \neg x_4)$ 를 생각하자. Clause $(x_1 \lor x_2 \lor x_3)$ 가 있으므로 $x_1, x_2, x_3$ 중 정확히 하나만 true이고 나머지 둘은 false여야 한다. 이는 relation $R_1(x_1, x_2, x_3) = (x_1 \land \neg x_2 \land \neg x_3) \lor (\neg x_1 \land x_2 \land \neg x_3) \lor (\neg x_1 \land \neg x_2 \land x_3)$와 동치이다. 비슷하게, relation $R_2(x_1, x_2, x_3) = (x_1 \land \neg x_2 \land x_3) \lor (\neg x_1 \land x_2 \land x_3) \lor (\neg x_1 \land \neg x_2 \land \neg x_3)$ 을 정의하면 1-IN-3SAT instance $\varphi$는 $R_1(x_1, x_2, x_3) \land R_2(x_1,x_3,x_4)$와 동치이다.

### SAT-type problem

-  A SAT-type problem is a set of relations $R_1, R_2, ..., R_k$.

SAT-type problem은 relation들의 집합이다. 이것 역시 예를 통해 알아보자. 앞서 살펴본 예에서 두 relation $R_1$과 $R_2$이 등장했었다. 1-IN-3SAT instance $\varphi = (x_1 \lor x_2 \lor x_3) \land (x_1 \lor x_3 \lor \neg x_4)$은 $R_1(x_1, x_2, x_3) \land R_2(x_1,x_3,x_4)$과 동치이므로 이므로 SAT-type problem $(R_1, R_2)$에 들어간다. 그러나, 이는 relation $R_1$만으로는 표현할 수 없기 때문에 SAT-type problem $(R_1)$ 에는 들어가지 않는다. 즉, SAT의 변형 $Q$가 SAT-type problem ($R_1, .., R_n$)인지는 $Q$의 모든 instance가 relation $R_1, R_2, .., R_n$만을 이용해 표현할 수 있는지와 동일한 의미이다.

### Schaefer’s Dichotomy Theorem

**Theorem(Schafer's Dichotomy Theorem).** Relation $R_1, R_2, ..., R_k$가 다음 6가지 중 하나라도 해당되면 SAT-type problem $R( R_1, R_2, ..., R_k)$는 P이다. 그렇지 않다면 이는 NP-Complete이다.
        
- $\forall R_i, R_i$(TRUE, TRUE, .., TRUE) = TRUE. (**1-valid** relation)
- $\forall R_i, R_i$(FALSE, FALSE, ..., FALSE) = TRUE. (**0-valid** relation)
- $\forall R_i, R_i$는 최대 2개의 literal을 가진 clause들의 conjunction(AND)과 동치이다.  (**bijunctive** relation)
- $\forall R_i, R_i$는 최대 1개의 positive literal을 가진 clause들의 conjunction과 동치이다. (**weakly negative** relation)
- $\forall R_i, R_i$는 최대 1개의 negative literal을 가진 clause들의 conjunction과 동치이다. (**weakly positive** relation)
- $\forall R_i, R_i$는 어떤 **affine** formula와 동치이다. Affine formula란 xor clause들의 conjunction을 뜻한다.(e.g. $(x_1 \oplus \neg x_2 \oplus x_3) \land (x_2 \oplus x_4).$)

각 6가지 케이스에 대해 다항식 시간에 해결하는 알고리즘을 찾는 것은 크게 어렵지 않다. 1-valid이면 모든 variable을 1로, 0-valid면 0으로 설정하면 되고, 아래 4가지 경우도 간단한 알고리즘이 존재한다. 그러나 이 이외의 경우에 모두 NP-Complete임을 증명하는 것은 매우 어려운 작업처럼 보인다. Schaefer는 어떻게 이를 증명했을까? 이 절에서는 Schafer's Dichotomy Theorem의 흐름과 중요한 보조정리들을 소개하고 증명할 것이다.

**Lemma 1.** $R(x_1, x_2, x_3) = (x_1 \land \neg x_2 \land \neg x_3) \lor (\neg x_1 \land x_2 \land \neg x_3) \lor (\neg x_1 \land \neg x_2 \land x_3)$에 대해,  
        3SAT $\le_p$ SAT-type problem ($R$)이 성립한다.

**Proof.**
$R$ 은 $x_1, x_2, x_3$중 정확히 하나가 true일 때 true이다.

$A = R(x, u_1, u_4) \land R(y, u_2, u_4) \land R(u_1, u_2, u_5) \land R(u_3, u_4, u_6) \land R(z, u_3, 0)$,

$B = R(x,y,0)$

$C = R(x, x, u)$  
로 놓으면 $\exists_{(u_i)}$ $A$ 는 $(x \lor y \lor z)$ 와 동치, $B$ 는 $x \neq y$와 동치, $\exists_{u}$ $C$ 는 $\neg x$와 동치이다. 따라서, 3개의 literal로 이루어진 모든 종류의 clause에 대해 동치인 formula를 relation $R$과 additional variable을 이용해 만들 수 있다.


**Definition.** $Rep(R_1,.., R_k)$ 은 relation $R_1, .., R_k$만으로 표현되는 relation들의 집합을 나타낸다.

바로 위 Lemma 1에서 본 relation $R$의 경우, lemma의 증명 과정에서 $[x \lor y \lor z], [x \neq y], [\neg x] \in Rep(R)$임을 보였다. ($[x \neq y]$는 relation $R_1(x,y) = (x \lor y) \land (\neg x \lor \neg y)$를 보다 보기 쉽게 쓴 것이다. )

다음과 같은 4개의 Lemma가 성립한다.

**Lemma 2.** $S = ( R_1, R_2, ..., R_k )$에서 $R_i$들은 모두 0-valid는 아니고, 모두 1-valid도 아니다. 이 떄, $[x \neq y] \in Rep(S)$ 또는 $[x], [\neg x] \in Rep(S)$ 임이 보장된다. 나아가, $[x], [\neg x] \notin Rep(S)$인 경우 $Rep(S)$는 complementive하다. (Complementive는 $\forall R \in Rep(S)$, $R(x_1, .., x_m) = R(\neg x_1, ..., \neg x_m)$가 성립한다는 뜻이다)


**Lemma 3.** 만약 $S = ( R_1, R_2, ..., R_k )$가 weakly negative가 아닌 relation $P$과 weakly positive가 아닌 relation $N$을 포함한다면 $[x], [\neg x] \in Rep(S)$는 $[x \neq y] \in Rep(S)$를 imply한다.


**Lemma 4.** $R$ 이 not bijunctive일 때, $Rep(R, [x \neq y], [x \lor y])$ 는 "exactly one of x,y,z" relation을 포함한다. Lemma 1에 따르면 이는 $Rep(R, [x \neq y], [x \lor y])$가 모든 relation들의 집합임을 뜻한다.


**Lemma 5.** $R$이 affine이 아닌 경우 $[x \lor y], [\neg x \lor y], [x \lor \neg y], [\neg x \lor \neg y] \in Rep(R, [x \neq y], [x])$이다.

이 Lemma들이 참임을 일단 가정하고, Schafer's Dichotomy theorem을 증명해보자.

SAT-type problem $S = ( R_1, R_2, ..., R_k )$가 6가지의 P인 case에 포함되지 않는다고 하자. 그러면 개중에는 non-affine relation, non-weakly positive/negative relation, non-bijunctive relation이 존재한다. 

만약 $[x], [\neg x] \in Rep(S)$이면 Lemma 3에 의해 $[x \neq y] \in Rep(S)$이고, Lemma 4에 의해 "exactly one of x,y,z" relation도 $Rep(S)$에 포함된다. 따라서 Lemma 1에 의해 $3SAT \le_p S$.

$[x], [\neg x] \in Rep(S)$가 아니라면 Lemma 2에 의해 $[x \neq y] \in Rep(S)$이고 $S$는 complementive. 만약 relation 내에 변수 대신에 $0, 1$을 대입할 수 있다면 이는 $[x], [\neg x] \in Rep(S)$와 같은 의미이고, 앞서 살펴본 과정을 그대로 하면 "exactly one of x,y,z" relation이 $Rep(S)$에 포함됨을 알 수 있다. 임의의 3개의 literal로 이루어진 clause $C$는  "exactly one of x,y,z" relation으로 표현할 수 있으므로 $C$는 $S$의 relation들의 변수에 0과 1을 추가적으로 사용하여 표현할 수 있다. $0$과 $1$ 대신 $y_0$과 $y_1$로 각각 대치하고 마지막에 clause $y_0 \neq y_1$을 넣었다고 생각해보자. $(y_0, y_1)$은 $(0,1)$ 또는 $(1,0)$에 해당할 것인데 $Rep(S)$는 complementive이므로 이는 $C$와 동치가 된다. 따라서, $3SAT \le_p S$가 성립한다.


### Schaefer’s Dichotomy Theorem의 활용

이때까지 Schaefer’s Dichotomy Theorem이 무엇인지와 증명을 살펴보았다. 이를 통해 간단한 다항식 시간 알고리즘이 존재하는 케이스가 아닌 모든 SAT-type problem이 NP-Complete라는 결과를 얻었다. 이를 앞서 살펴본 SAT의 변형이 P인지 NP-Complete인지 판정하는데에 좋은 도구로써 사용할 수 있을까?

Schaefer’s Dichotomy Theorem을 이용하여 NAE-E3SAT이 NP-Complete임을 증명해보자.

Relation $R(x, y, z) = (x \lor y \lor z) \land (\neg x \lor \neg y \lor \neg z)$은 NAE-E3SAT instance $(x \lor y \lor z)$와 동치이므로, SAT-type problem $(R)$이 NP-Complete임을 보이면 충분하다.

- $R$이 1-valid나 0-valid가 아님은 자명하다.
- $R$이 bijunctive가 아님을 보이자. 2개의 literal로 이루어진 clause는 constant true가 아니면 $(x,y,z)$ assignment 8개 중 적어도 2개를 만족시키지 못한다. $R$은 8개 중 $(0,0,0), (1,1,1)$의 두 가지를 제외하고는 모두 만족하는데, 어떤 2개의 literal로 이루어진 non-constant clause도 $(0,0,0), (1,1,1)$이 아닌 6개의 assignment를 모두 만족하지 않는다. 따라서 R은 non-bijunctive.

- R이 CNF $\phi$와 동치라고 가정하자. 앞서 본것처럼 $\phi$에는 2-literal clause가 들어갈 수 없고 마찬가지 논리를 쓰면 $x \lor y \lor z, \neg x \lor \neg y \lor \neg z$ 만 가능한 clause임을 알 수 있다. 하지만 각각은 3개의 positive literal을 가진 clause와 3개의 negative literal을 가진 clause이다. $R(x,y,z) \neq x \lor y \lor z$, $R(x,y,z) \neq \neg x \lor \neg y \lor \neg z$ 에서 $R$ 는 weakly positive / weakly negative일 수 없다.

- Nontrivial affine formula는 xor들의 conjunction이기 때문에 절반이상의 assignment에서 false이다. 하지만 $R(x,y,z)$ 는 8개 중 6개의 assignment에서 true이므로 $R$은 affine이 아니다.

따라서, Schafer's Dichotomy Theorem에 의해 NAE-E3SAT은 NP-Complete이다.



## 참고 자료
* https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
* https://en.wikipedia.org/wiki/Schaefer%27s_dichotomy_theorem
* https://hardness.mit.edu/drafts/2022-09-21.pdf