---
layout: post
title:  "Suffix Automaton"
date:   2023-02-01
author: koosaga
tags: [algorithm, strings, competitive-programming]
---
# Suffix Automaton

문자열의 부분문자열에 대한 복잡한 문제를 풀 때 Suffix Array와 같은 *접미사 구조* 는 아주 강력한 도구가 된다. SCPC 2021 3번, 서울 리저널 2022 H 등 여러 중요한 대회에서도 이러한 접미사 구조를 응용한 문제들이 정말 많이 나온다. 한국에서 많은 사람들이 알고 있는 접미사 구조로는 [*Suffix Array*](https://koosaga.com/125) 가 있다. Suffix Array는 모든 문자열의 접미사를 정렬한 순열로, 흔히 부분문자열 탐색 쿼리를 빠르게 처리하거나 두 접미사의 LCP를 구할 때 많이 쓰인다.

문자열 접미사 구조 중 알려진 자료구조는 크게 *Suffix Array, Suffix Tree, Suffix Automaton* 이 있는데, 한국에서는 Suffix Array만이 알려져 있는 편이고, 그 외 다른 자료구조를 설명한 글은 거의 없다. 이 글에서는 아직 많이 알려지지 않은 다른 접미사 구조인 **Suffix Automaton** 을 소개한다. Suffix Automaton은 Suffix Array에 비해서 다음과 같은 장점이 있다:

* Suffix Array보다 구현하기 쉬움
* 선형 시간에 작동하기 때문에 일반 Suffix Array보다 효율적임
* Suffix Tree 구조를 쉽게 얻을 수 있음 (고로 Suffix Array 역시 쉽게 얻을 수 있음)
* Suffix Automaton에서만 풀 수 있는 문제들이 존재함

해외에서는 Suffix Array 대신 Suffix Automaton으로 접미사 구조를 처음 배우는 경우도 많아 보였다. Suffix Array도 훌륭한 구조이지만, 최근 ICPC 2022 H 등의 문제를 접하면서 Suffix Automaton이 주는 이점이 더 크다고 생각했다. 생각보다 어렵지 않은 자료구조이니, 잠시 시간을 내서 배우면 도움이 될 것 같다.

## Recap: Suffix Trie

Suffix Automaton을 설명하기 앞서 잠시 기초 개념을 복습하자. 다음과 같은 문자열 문제를 생각해 보자:

* 길이 $N$ 의 문자열 $S$ 과 $Q$ 개의 쿼리 $T_1, T_2, \ldots, T_Q$ 가 주어진다. 각 쿼리에 대해 주어진 문자열이 $S$ 의 부분 문자열인지 판별하라. $S$ 의 길이는 5000 정도이고, $Q$ 및 $|T_i|$ 의 길이 합은 크다.

이 문제를 단순히 해결하는 방법은 각각의 쿼리 문자열에 대해서 KMP를 사용해서 $S$ 의 부분문자열인지 판별하는 것이다. 이는 $O(NQ$) 시간이 걸려 느리다. 보통 문자열 집합 판별을 할 때 가장 많이 쓰는 것은 **트라이 (trie)** 자료 구조이다. 트라이는 $T_i$ 가 어떠한 문자열 집합의 Prefix인지를 쉽게 판별할 수 있다. 만약 $S$ 의 모든 Suffix를 Trie에 넣는다면 우리는 $T_i$ 가 $S$ 의 Suffix의 Prefix인지, 즉 부분 문자열인지를 $|T_i|$ 시간에 판별할 수 있다.

즉, Suffix Trie는 $S$ 의 모든 Suffix를 문자열 집합으로 가지는 트라이를 뜻한다. Suffix Trie는 $O(N^2)$ 공간 및 시간 복잡도를 가지기 때문에, 한번 만들면 유용하지만 만드는 과정이 비효율적이라는 단점이 있다. 이를 해결하기 위한 두 가지 시도가 Suffix Tree와 Suffix Array이다.

* Suffix Trie는 $N^2$ 개의 노드를 가지고 있지만, 가지고 있는 문자열은 많아야 $N$ 개이다. 고로 절대 다수의 노드는 *분기* 가 없이 정확히 하나의 자식만을 가질 것이다. 이 *분기* 들을 압축해서 트리의 크기를 $O(N)$ 으로 줄인 트리를 Suffix Tree라고 한다. Trie의 각 간선이 문자 하나에 대응되듯이, Suffix Tree의 각 간선은 $S$ 의 어떠한 부분 문자열 $S[i, j]$ 에 대응된다. Suffix Tree는 Ukkonen's Algorithm을 사용하여 $O(N)$ 시간에 만들 수 있는데, 꽤 복잡한 알고리즘이다.
* 꼭 트리 형태로 가지고 있지 않더라도, $S$ 의 Suffix들을 사전순으로 정렬만 해도 충분한 경우도 많다. $S$ 의 Suffix를 사전 순으로 정렬한 것을 *Suffix Array* 라고 한다. Suffix Array의 인접한 원소의 LCP를 알면, Suffix Tree의 기능을 상당수 흉내낼 수 있다. Suffix Array는 $O(N \log N)$ 에 구하는 법이 잘 알려져 있고, 꽤 복잡한 $O(N)$ 알고리즘 역시 존재한다.

일단 여기까지 복습을 하고, Suffix Automaton에 대한 설명으로 넘어간다.

## Suffix Automaton: Definition

다시 다음 문제로 돌아오자:
* 길이 $N$ 의 문자열 $S$ 과 $Q$ 개의 쿼리 $T_1, T_2, \ldots, T_Q$ 가 주어진다. 각 쿼리에 대해 주어진 문자열이 $S$ 의 부분 문자열인지 판별하라. $S$ 의 길이는 5000 정도이고, $Q$ 및 $|T_i|$ 의 길이 합은 크다.

$S$ 의 Suffix Automaton은 사이클 없는 방향 그래프 (Directed Acyclic Graph, DAG) 로
* 루트 노드가 있고 (*시작 상태* 라고 한다.)
* 문자 하나에 대응되는 간선이 있으며
* 어떠한 문자열이 $S$ 의 부분 문자열이라면, 부분 문자열을 순서대로 읽으며 간선을 따라갈 수 있음 (다시 말해, 루트에서 시작하는 경로 중 이었을 때 $S$ 의 부분 문자열인 경로가 존재)

이하에서 **상태 (state)** 라고 표현하는 말은 **정점 (노드, node)** 과 동의어이다. 상태라는 말을 사용하는 건 Suffix Automaton이 기본적으로 유한 상태 오토마타 (Finite State Automata) 이기 때문이다. 이 글을 이해하기 위해 오토마타에 대해 알 필요는 없으나, 단지 다른 영문 자료에서 "상태" 라는 단어를 쓰는 이유를 설명하기 위해 첨언한다.

Suffix Automaton은 Suffix Trie랑 상당히 유사하지만 몇 가지 차이가 있다.
* Suffix Trie는 트리지만, Suffix Automaton은 DAG이다.
* Suffix Trie에서는 하나의 정점에 대응되는 문자열이 유일하지만, Suffix Automaton에서는 하나의 정점에 대응되는 문자열이 하나 이상일 수 있다. 즉, 루트 상태에서 특정 상태로 도달하는 경로 문자열이 여럿일 수 있다.

Suffix Automaton을 접할 때 혼란스러울 수 있는 것은 하나의 상태에 여러 문자열이 대응될 수 있다는 것이다. 다행이도, 대응되는 문자열들에 어느 정도 규칙이 있어서, 상태만 알아도 어떤 식의 문자열이었는지 대강 식별할 수 있다. 규칙에 대해 설명하기 위해 살짝 다른 이야기를 하자.

### Definition: $endpos(Q)$

고정된 문자열 $S$ 에서 $Q$ 라는 문자열을 찾는 상황을 생각해 보자. 예를 들어, $S = \texttt{abbcdbcbcd}, Q = \texttt{bcd}$  이라고 하면, $Q$ 가 $S$ 에 등장하는 위치는 $\texttt{ab[bcd]bc[bcd]}$ 와 같이 두 가지가 있다. 구간 전체를 표현하면 보기 힘드니, 등장 위치의 끝점만 남기고 $\texttt{abbcd|bcbcd|}$ 와 같이 표현한다. $endpos(Q)$ 는, 이러한 끝점의 인덱스 집합이다. 즉, 현재 $S$ 에 대해서 $endpos(\texttt{bcd}) = \{5, 10\}$ 이 된다.

모든 $Q$ 의 Prefix에 대해서 위와 같이 가능한 등장 위치를 모두 표시해보면, 다음과 같다.

* $Q =  \rightarrow$ $\texttt{|a|b|b|c|d|b|c|b|c|d|}$
* $Q = \texttt{b} \rightarrow \texttt{ab|b|cdb|cb|cd}$
* $Q = \texttt{bc} \rightarrow \texttt{abbc|dbc|bc|d}$
* $Q = \texttt{bcd} \rightarrow \texttt{abbcd|bcbcd|}$

자연스럽게도, $Q$ 의 Prefix가 길어질수록 등장 위치의 수는 줄어든다.
같은 표현을 $endpos(Q)$ 라는 수 집합으로 나타내면 다음과 같다:

* $endpos() = \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
* $endpos(\texttt{b}) = \{2, 3, 6, 8, 10\}$
* $endpos(\texttt{bc}) = \{4, 7, 9, 10\}$
* $endpos(\texttt{bcd}) = \{5, 10\}$

Suffix Automaton에서 두 Substring이 같은 상태에 있을 조건은, 두 Substring의 $endpos$ 집합이 같다는 조건과 동치이다. 어려운 말로 하면, Substring들을 endpos의 equivalence relation으로 묶은 것이 Suffix Automaton의 상태이다. Suffix Automaton의 시작 상태에서 멀어지는 것은, $Q$ 의 뒤에 문자들을 붙여서 $endpos$ 집합을 바꾸는 과정과 동일하다.

이제 Suffix Automaton의 어떤 상태를 이루는 문자열 집합의 성질을 관찰해 보자. 결론만 말하자면, 한 상태를 이루는 문자열 집합은 항상 이러한 꼴을 이룬다:
* $endpos(Q)$ 가 같은 가장 짧은 문자열을 $u$, 가장 긴 문자열을 $v$ 라고 할 때, 각 상태에 대응되는 문자열 집합은 $v$ 의 길이 $|u|, |u| + 1, \ldots, |v|$ 길이의 Suffix이다.

왜 그럴까? 다음과 같은 사실들을 관찰하면 된다. $u, v$ 가 모두 $S$ 의 Substring이라 할 때:
* $endpos(u) = endpos(v)$ 이고 $|u| \le |v|$ 라면, $u$는 항상 $v$ 의 Suffix 이다.
* $S$의 임의의 두 부분문자열 $u, v$ 에 대해, $u$ 가 $v$ 의 Suffix 라면 $endpos(v) \subseteq endpos(u)$ 이고, 그렇지 않다면 $endpos(u) \cap endpos(v) = \emptyset$ 이다.

이 사실들의 증명은 $endpos$ 라는 집합의 정의를 따라 생각해 보면 어렵지 않다.

종합하면, 각 상태에 대응되는 문자열들의 길이는 서로 다르고 (1번 성질), 최소 길이 문자열의 $endpos$ 와 최대 길이 문자열의 $endpos$ 가 같기 때문에 그 사이 길이의 문자열의 $endpos$도 같아져서 (2번 성질) 앞과 같은 성질이 성립한다.

$endpos$ 의 정의까지 모두 열거했으니, 우리는 Suffix Automaton을 만드는 간단하고 느린 다항 시간 알고리즘을 얻을 수 있다.
* 모든 부분문자열을 $endpos$ 가 같은 그룹으로 묶은 후 하나의 상태로 둔다.
* 각 상태 $v$ 에 대해서, 뒤에 문자 $\alpha$ 를 하나 붙여보고, $endpos(v + \alpha)$ 를 찾고, 비지 않았다면 (즉 $S$ 의 부분문자열이라면) 해당 노드로 가는 간선 $\alpha$ 를 만든다.

이 알고리즘은 대단히 느리며, 우리는 Suffix Trie 대신 이런 알고리즘을 써야 할 이유조차 아직 이해하지 못한다. 직관을 더 얻기 위해 Suffix Link의 개념을 살펴 보자.

### Definition: Suffix Link
$S$ 의 어떠한 부분문자열 $v$에 대해서, $v$의 맨 앞 문자를 계속 제거해 나가자. 한동안 $endpos$가 달라지지 않을 때는, $v$ 가 속하는 상태가 동일하게 유지될 것이다. 그러다 어느 순간 $endpos$ 가 달라진다면 $v$ 가 속하는 상태가 다른 상태가 될 것이다. 이 다른 상태를 Suffix Link라고 한다.

구체적으로, 어떤 상태 $v$ 에 대해 $endpos$ 가 같은 가장 짧은 문자열을 $u$ 라고 했을 때, $Q$ 의 Suffix Link $slink(v)$ 는 $u$ 의 맨 앞 문자를 제거한 문자열에 대응되는 상태이다.

Suffix Automaton의 루트가 아닌 모든 상태 $v$ 에 대해서 $v \rightarrow slink(v)$ 로 가는 방향 그래프를 만들면, 이 그래프는 **트리**이다 (endpos의 크기가 증가하며, 모든 정점에 부모가 있기 때문이다). 아래 나오는 관찰들은 Suffix Link의 중요한 성질들을 조명한다.
* $v$ 에 대응되는 가장 짧은 문자열을 $u$ 라고 했을 때, $slink(v)$ 에 대응되는 가장 긴 문자열은 $u$ 의 길이 $|u| - 1$ prefix이다.
  * 그렇지 않다면 Suffix가 아닌데 $endpos(u) \subseteq endpos(v)$인 문자열 쌍을 얻는다.
* $S$ 의 Suffix Automaton에서 $v \rightarrow slink(v)$ 로 가는 방향 간선을 사용해 트리를 만들면, $rev(S)$ ($S$ 를 뒤집은 문자열) 의 **Suffix Tree** 를 얻는다. (정의상 자명)
* Suffix Tree의 노드가 $O(n)$ 개이기 때문에, Suffix Automaton의 노드 역시 $O(n)$ 개이다.
   * 모든 Suffix Tree의 노드가 $O(n)$ 개인 것은 아니지만, 이 Suffix Tree의 경우 internal node에서 $endpos$ 집합이 두 개 이상의 서로소 집합으로 분할되는 분기가 일어나기 때문에 노드의 개수가 최대 $2n-1$ 개이다.
* 위 정의에 의해 $endpos$ 집합은 Laminar Family를 이룬다. 이 Laminar Family에서 트리를 구성하면 그것이 우리가 구성한 Suffix Tree가 된다. (정의상 자명하다. 무슨 말인지 모르겠으면 넘어가도 무방하다.)

즉, 만약에 우리가 배우게 될 Suffix Automaton 알고리즘이 Suffix Link 역시 구해줄 수 있다면, 이 알고리즘은 Suffix Tree를 구하는 알고리즘이기도 하다는 것이다. 이 얘기는 이후 자세히 하고, 이제 Suffix Automaton을 구하는 알고리즘을 소개한다.

## The Algorithm
Suffix Automaton을 만드는 과정은, 처음에 **루트 상태** 만 있는 빈 문자열의 오토마톤에서 시작해서, $s_1, s_2, \ldots, s_n$ 순서대로 맨 뒤 문자를 하나씩 추가하는 방식이다. 이러한 알고리즘을 *incremental* 하다고 부르며, Suffix Automaton은 *incremental* 한 알고리즘이다.

각 상태는 다음과 같은 정보를 저장한다:
* Suffix Link (루트 상태의 경우 아무거나, 본인은 $0$ 으로 함)
* 뒤에 문자 $\alpha$ 를 추가했을 때 도달하게 되는 상태의 번호 (없을 시 $-1$)
* 대표하는 문자열 중 **가장 긴 것** 의 길이 (루트 상태의 경우 $0$)

또한 추가적으로 전체 문자열에 대응되는 상태의 인덱스를 저장하자.

위에서도 말했듯이, 맨 처음에는 빈 문자열의 오토마톤에서 시작하기 때문에, 루트 상태만 필요하다. Initialize 단계에서는, 루트 상태 ($0$번 상태) 를 만든 후, 전체 문자열에 대응되는 상태의 인덱스를 $0$ 으로 설정하고 종료한다.

$S$ 에 대한 Suffix Automaton이 있을 때 이를 $S + c$ 에 대한 Suffix Automaton으로 바꿔주는 `addChar(c)` 라는 함수를 생각해 보자. 현재 Suffix Automaton에서 우리가 새롭게 발견하는 부분 문자열은 $S + c$ 의 Suffix 들이다. 또한 이들은 새롭게 발견되었기 때문에 $endpos(Q) = \{|S| + 1\}$ 이고, 즉 $S + c$ 의 Suffix에 대응되는 단 하나의 새로운 상태만 만들어 주면 된다. $v_{new}$ 를 새로 만든 상태라고 하면, $S$ 에 대응되는 상태 $v$ 에서 시작해서, 다음을 반복한다:
* $v$ 에서 $c$ 로 뻗어 나가는 상태가 없다면 이를 $v_{new}$ 로 설정한다.
* 상태가 있었거나, $v$ 가 루트 노드라면 종료한다.
* 아니라면 $v \leftarrow slink(v)$ 로 Suffix Link를 타고 내려간다.

이게 **기존 상태** 에 해야 할 작업의 전부로, 정말 간단하게도 이게 끝이다! 그런데 사실 **새로 만든 상태** 에 대해 설명을 충분히 하지 않았다. 새로 만든 상태에서 다음 상태는 모두 $-1$ 이며, 대표하는 가장 긴 문자열의 길이는 전체 문자열의 길이이다. 하지만 우리는 **새로 만든 상태의 Suffix Link** 를 알지 못한다. 이걸 찾으면 끝인데, 이 과정이 약간 까다롭다.

$S + c$ 의 Suffix Link는 $S$ 의 부분문자열로 존재하는 가장 긴 $S + c$ 의 Suffix이다. 그래야만 $endpos$ 의 크기가 커질 수 있기 때문이다. 맨 처음 $v_{new}$ 를 만들 때 $v$ 에서 $c$ 로 뻗어 나가는 상태가 있던 최초의 $v$ 로 돌아가자 (만약 이러한 $v$ 가 없다면 Suffix Link는 루트이다). 이 $v$ 에 대응되는 가장 긴 문자열을 $long(v)$ 라 할 때, $long(v) + c$ 가 $S + c$ 의 Suffix 중 $S$ 의 부분문자열로 존재하는 가장 긴 것임을 확인할 수 있다. 기본적으로 Suffix Automaton 상에 경로가 있음과 부분문자열임이 동치이기 때문이다.

그렇다면 $long(v) + c$ 에 대응되는 상태에 단순히 Suffix Link를 이어주면 될까? 그렇지는 않다. $long(v) + c$ 에 대응되는 상태는 하나 이상의 문자열을 포함할 수 있고, 이들 중 $endpos$ 로 $|S| + 1$ 을 포함하게 되는 문자열은 정확히 $long(v) + c$ 뿐이기 때문이다. 만약 $long(v) + c$ 에 대응되는 상태가 **정확히** 하나의 문자열을 포함한다면, 그냥 이어주면 된다. 하지만 그렇지 않다면, 이 상태를 **분할** 해서, $upd = [long(v) + c, long(v) + c]$ 그리고 $prv = [long(v) + c + \alpha, long(v) + c + \alpha\beta\gamma \ldots]$ 와 같은 두 개의 상태로 쪼갠 다음 $upd$ 상태에만 Suffix Link를 이어줘야 한다.

하여튼, 새로 만든 상태의 Suffix Link는 $upd$ 이니, 이 새로운 상태들에 대한 정보들만 잘 넣어주면 되겠다. 어떻게 하면 될까?
* $prv$ 와 $upd$ 모두 다음 상태의 정점은 기존과 동일하다.
* 길이는 $prv$ 의 경우 기존과 동일하고, $upd$ 의 경우 $|long(v)| + 1$ 이다.
* $prv$ 의 Suffix Link는 $upd$ 이다. $upd$ 의 Suffix Link는 기존과 동일하다.
* 원래 $prv$ 를 다음 상태로 가리키던 노드들에 대해서 다음 상태를 $upd$ 로 바꿔줘야 한다. 이러한 노드들은 모두 $v$ 의 Suffix Link 상에서 루트 방향으로 가는 Path를 이루기 때문에, 나이브하게 올라가 주면서 바꿔주면 된다.

마지막으로, 전체 문자열에 대응되는 상태의 인덱스를 $S + c$ 상태로 바꾸는 것을 까먹지 않아주면, 알고리즘은 완성된다. 설명이 대단히 길었는데, 사실 구현은 상당히 짧은 편이기 때문에, [나의 구현](https://github.com/koosaga/olympiad/blob/master/Library/codes/string/suffix_automaton.cpp) 과 함께 보면 이해에 도움이 될 것 같다.

이제 알고리즘에 대한 사실을 나열한다.
* 알고리즘은 올바른 Suffix Automaton을 찾는다. (완전히 증명하진 않았지만 필요한 사실은 전부 있다.)
* 알고리즘은 최대 $2n-1$ 개의 상태를 만든다. (이 사실은 위에서도 증명했다.) 최대 상태를 만드는 데이터는 `abbbbbb....bbbb` 와 같다.
* 알고리즘은 최대 $3n-4$ 개의 간선을 만든다 (증명 생략). 최대 간선을 만드는 데이터는 `abbbbbb....bbbc` 와 같다.
* 위 알고리즘은 상태 생성에 $O(1)$, 다음 상태 인덱싱에 $O(1)$ 이 필요하다는 가정 하에 시간 및 공간 복잡도 $O(n)$ 에 작동한다 (증명 생략).
   * 명세마다 조금 다를 수 있다. 예를 들어 다음 상태 전이를 HashMap에 저장했다면 시간 공간 모두 $O(n), O(n)$ 이다. 다음 상태 전이를 Map에 저장했다면 시간은 $O(n \log \Sigma)$, 공간은 $O(n)$ 이다. 다음 상태 전이를 배열에 저장했다면 시간은 $O(n\Sigma)$, 공간은 $O(n\Sigma)$ 이다.

## Application

### Suffix Automaton, Suffix Tree, Suffix Array
위에서 본 것과 같이, Suffix Automaton을 구성하는 과정에서 Suffix Link를 얻을 수 있고, 이 Suffix Link로 트리를 구성하면 $rev(S)$ 의 Suffix Tree를 얻을 수 있다. 고로 Suffix Automaton 알고리즘은 **Suffix Tree를 얻는 가장 좋은 알고리즘** 이 되며, 여기서 가장 좋다는 것은
* 시간 복잡도가 $O(n)$ 이다.
* 코드가 짧다 (Ukkonen's Algorithm은 말할 것도 없고, Suffix Array를 $O(n \log n)$ 에 구한다고 해도 Suffix Automaton보다 길다.)
* 이해하기 쉽다 (이건 Suffix Array가 더 쉽다고 볼 수도 있겠다. 취향 차이일 듯 하다).
* Incremental하다 (정확히는, 각각의 Suffix에 대한 Suffix Tree를 모두 구할 수 있다.)

는 측면에서의 이야기이다.

사실 Suffix Automaton, Suffix Tree, Suffix Array는 모두 같은 구조고, 각 자료 구조를 가지고 다른 자료구조를 얻는 방법이 모든 $3 \times 2$ 가지 경우에 대해서 존재한다. 이론적으로는 각각을 구분하는 것이 크게 의미가 없기 때문에, 결국 배우기 얼마나 쉬운지, 구현하기 얼마나 쉬운지, 빠른지 정도가 핵심인 것 같다.

내 생각에는 Suffix Tree는 Suffix Automaton을 사용해서 구하는 것이 가장 좋고, Suffix Array는 그렇게 어렵지 않으니 따로 배워서 사용하는 게 나은 것 같다. Suffix Automaton이 워낙 짧아서, Suffix Tree를 Automaton으로 만들고 그 트리의 중위 순회를 해서 Suffix Array + LCP를 구해도 유용할 수도 있겠다는 생각은 있다. 일반적인 상황은 아니고, SCPC처럼 Pre-written code를 못 보는 환경인데 Suffix Array의 정배열 역배열이 맨날 헷갈린다면 그런 정도의 상황에서는 유용할 수도 있을 것 같다.

### Solving problems with Suffix Automaton
Suffix Automaton으로 문제를 푸는 경우의 절대다수는 Suffix Link (Suffix Tree) 구조만을 사용하기 위해서 Suffix Automaton을 사용한다. Suffix Automaton의 DAG가 필요한 문제도 분명히 존재하지만, 많은 문제는 DAG를 사용하게 되면 오히려 복잡하고 응용이 불가능해지는 경우가 많으니 조심히 사용해야 한다. 특히 [CP-algorithm](https://cp-algorithms.com/string/suffix-automaton.html) 에 나온 연습 문제들의 대다수는 Suffix Automaton을 사용해서 풀면 **안되고** Suffix Tree만 사용하는 것이 좋다. Suffix Automaton으로 무슨 문제를 풀어야 하는지는 아래에 설명한다.

**Problem 1.** 문자열 $S$ 가 주어졌을 때 서로 다른 부분 문자열의 개수를 세어라. [BOJ 11479](https://www.acmicpc.net/problem/11479)
**Solution using Suffix Automaton.** $S$ 의 부분 문자열임은 루트 상태에서 시작하는 경로 중 하나라는 것이다. 모든 경로의 개수를 DAG에서의 DP로 세면 된다.
**Solution using Suffix Tree.** 위와 같은 식으로 문제를 풀면 문제가 조금만 복잡해져도 응용이 불가능하다. 대신 다음과 같이 풀어야 한다. 결국 각 상태가 표현하는 문자열의 길이 합을 계산하면 되는데, 이는 $len(v) - len(slink(v))$ 이다. 모든 루트가 아닌 상태 $v$ 에 대해서 위 수량의 합을 관리하면 된다. 이 풀이를 사용하면, Suffix Automaton의 Incremental함을 응용할 수 있어서 [BOJ 16907](https://www.acmicpc.net/problem/16907) 을 Online으로 쉽게 풀 수 있다. (Suffix Array를 사용해서 Online으로 풀어보려고 하면 쉽지 않다.)

**Problem 2.** [ICPC 서울 리저널 2022 H](https://www.acmicpc.net/problem/26109)
**Solution using Suffix Tree.** https://koosaga.com/306 에 설명된 풀이를 Suffix Tree로 구현하면 짧고 쉬운 코드로 문제를 해결할 수 있다.

**Problem 3.** 문자열 $S$ 가 주어지고, Alice와 Bob이 빈 문자열 $T$ 를 가지고 게임을 한다. Alice가 먼저 시작하며, 번갈아서 게임을 진행한다. 각 턴에서 플레이어는 문자열의 뒤에 문자 하나를 삽입할 수 있다. 자기 턴 이후에 $T$ 가 $S$ 의 부분 문자열이 아니게 될 경우 해당 플레이어가 패배한다. 누가 이기는가?
**Solution using Suffix Automaton.** 이 문제는 다른 자료 구조로 쉽게 푸는 법을 잘 모르겠다. Suffix Automaton의 DAG가 상당히 유용하다고 생각한다. Suffix Automaton의 DAG에서 문제를 조명할 경우, DAG에서 토큰을 옮겨서 상대방을 outdegree 0인 노드로 몰아놓는 문제가 된다. 이는 DAG에 간단한 DP를 하여 해결할 수 있다. 이 문제가 채점이 되는 사이트는 못 찾았고, 이 문제에서 문자를 문자열 앞에도 넣을 수 있는 [어려운](https://www.acmicpc.net/problem/18966) 버전은 알고 있다.

**Problem 4.** 문자열 $S$ 가 주어졌을 때 사전순 $k$ 번째 부분 문자열을 찾아라. [BOJ 13541](https://www.acmicpc.net/problem/13541)
**Solution using Suffix Automaton.**  DAG 상에서 사전 순 $k$ 번째 경로를 찾는 것이니 경로의 개수를 Problem 1과 같이 DP로 계산하고 역추적하면 된다. 이 문제는 이것보다는 Suffix Array를 써서 푸는 것이 낫다.

**Problem 5.** [KAIST 2022 가을대회. Double-Colored Papers](https://www.acmicpc.net/problem/25729)
**Solution using Suffix Automaton.** Problem 4의 변형인데 이 문제는 Suffix Automaton을 써서 푸는 게 훨씬 간단하다. $S$ 와 $T$ 각각에 대해서 Suffix Automaton을 만들면, 결국 크기 $O(|S| + |T|)$ 의 DAG에서 사전 순 $k$ 번째 경로를 찾는 문제가 된다. Problem 4와 유사하게 해결할 수 있는데, 여기서는 만든 DAG가 그냥 Suffix Automaton이 아니라 같은 알파벳을 가진 간선이 여럿일 수 있고, 고로 위 접근을 그대로 활용할 수 없다. 이를 해결하기 위해서는 해당 DAG에서 한 문자열에 대응되는 상태가 $O(1)$ 개의 정보로 표현 가능하다는 점을 잘 써야 하는데, 어려운 문제이니 여기서 전부 설명하지는 않겠다.

**Problem 6.** [BOJ 14436. 서로 다른 부분 문자열 쿼리](https://www.acmicpc.net/problem/14436)
**Solution using Suffix Tree.** Ukkonen's Algorithm을 사용하면 성질상 어렵지 않게 되는 것 같다. 잘 모른다. 이 풀이는 아마 Online으로 문제를 해결할 수 있다.
**Solution using Suffix Tree, but without Ukkonen.** Ukkonen's Algorithm은 복잡하니까, 해당 알고리즘 없이 이 문제를 Offline에 해결해 보자. 전체 문자열을 받은 후 Suffix Tree를 구성한다. 핵심은 문자열 $S$ 의 임의의 부분 문자열 $S[i \ldots j]$ 에 대해 서로 다른 부분 문자열의 개수를 반환하는 자료 구조를 구성하는 것이다 - 이러한 자료 구조를 구성할 수 있으면 물론 전체 문제를 해결할 수 있다. 이건 Suffix Tree에 HLD를 구성하면 할 수 있는데, [이 동영상](https://www.youtube.com/watch?v=QdqESofsu_g) 에 자세한 내용을 설명하였다.

**Problem 7.** [BOJ 18544. Incomparable Pairs](https://www.acmicpc.net/problem/18544)
**Solution using Suffix Tree.** Problem 6에서 구성한 자료 구조를 조금 응용하면 위 문제의 풀이로 어렵지 않게 변형할 수 있다.

## Reference
두 글 다 그렇게 이해가 잘 되게 쓴 글은 아닌데, 첫 번째 글은 그래도 내용이 부족하지는 않아서 열심히 읽으면 이해할 수는 있다. 이 글은 첫 번째 글을 열심히 읽고 내 나름대로 재구성한 것이다.
* https://cp-algorithms.com/string/suffix-automaton.html
* https://codeforces.com/blog/entry/20861
