---
layout: post
title: "Faster Exponential Algorithm for Permutation Pattern Matching"
author: TAMREF
date: 2022-01-16
tags: [combinatorics]

---



# Introduction

“스택 수열”이라는 간단한 문제를 생각해 봅시다.

[https://www.acmicpc.net/problem/1874](https://www.acmicpc.net/problem/1874)

이 문제는 스택을 이용한 시뮬레이션으로 해결할 수 있습니다. 요약하면, $[n] := \{1, \cdots, n\}$의 순열 $\sigma _ {1}, \cdots, \sigma _ {n}$이 주어졌을 때, $1, \cdots, n$을 순서대로 스택에 push(+)했다가 pop(-)하여 $\sigma _ {1}, \cdots, \sigma _ {n}$을 만들 수 있는지를 판별하는 문제입니다.

반대로 $\sigma _ {1}, \cdots, \sigma _ {n}$을 push-pop하여 $1, \cdots, n$으로 정렬할 수 있는지를 묻기도 하는데요, 이 경우에 $\sigma$를 stack-sortable permutation이라고 합니다. “스택 수열” 문제를 푸는 과정을 거꾸로 시뮬레이션한다고 생각하면, stack-sortable permutation과 “스택 수열” 문제의 답이 될 수 있는 순열 사이에 일대일대응이 있다는 사실을 알 수 있습니다.

stack-sortable permutation을 판별하는 문제는 간단하게 해결할 수 있지만, 길이가 $n$인 stack-sortable permutation의 개수를 **세는** 문제도 해결할 수 있을까요? 답은 Yes입니다.

**Proposition 1.** 다음은 동치이다.

- $\sigma$는 stack-sortable permutation이다.
- $\sigma$는 “$231$-avoiding”이다. 다시 말해, $a < b < c$이고 $\sigma _ {c} < \sigma _ {a} < \sigma _ {b}$인 $a, b, c$는 존재하지 않는다.

*pf of Proposition 1.* 먼저 $231$-avoiding이 아니라면 stack-sortable하지 않음을 보입니다. $\sigma _ {b}$와 $\sigma _ {c}$는 뒤의 $\sigma _ {a}$ 때문에 무조건 스택에 들어가야 하는데, $\sigma _ {c}$가 $\sigma _ {b}$보다 먼저 꺼내질 수밖에 없으므로 sortable하지 않습니다.

만약 $\sigma$가 $231$-avoiding이라면, $\sigma$의 임의의 부분 수열(subsequence)도 $231$-avoiding입니다. $\sigma _ {i} = 1$이라고 해봅시다. (즉, $i = \sigma^{-1}(i)$) 그렇다면 $231$-avoiding이기 때문에 $\sigma _ {1}, \cdots, \sigma _ {i-1}$은 감소수열이어야 합니다. 따라서 이들 모두는 처음에 스택으로 들어가야 하고, $\sigma _ {i} = 1$이 스택에 들어가자마자 나오면 결국 $\sigma _ {1}, \cdots, \sigma _ {i-1}, \sigma _ {i+1}, \cdots, \sigma _ {n}$이라는 $\{2, \cdots, n\}$의 순열을 정렬하는 중간 과정이 남고, 이 수열도 $231$-avoiding이기 때문에 $n$에 대한 귀납 가정에 의해 마저 정렬해줄 수 있습니다. $\square$

**Corollary 2.** 다음은 동치이다.

- $\sigma$는 “스택 수열” 문제의 답이다.
- $\sigma$는 “$312$-avoiding”이다. 다시 말해, $a < b < c$이고 $\sigma _ {b} < \sigma _ {c} < \sigma _ {a}$인 $a, b, c$는 존재하지 않는다.

Proposition 1을 알고 있으면, 잘 알려진 permutation diagram을 이용하여 $231$-avoiding permutation의 개수가 Catalan number $C _ {n} = \frac{1}{n+1}\binom{2n}{n}$과 같다는 사실을 알 수 있습니다. 간단히 말해 $(0, 0)$부터 $(n, n)$까지 있는 격자에 $(i-1, \sigma _ {i})$ 위치에 점을 찍어 보면, $(0, 0$)에서 $(n, n)$까지 위 / 오른쪽 으로만 가는 northeast-only path 중에서 이 점을 정확히 감싸는 경로와, Catalan number의 정의 중 하나인 $(0, 0)$에서 $(n, n)$까지 $y = x$ 선을 침범하지 않고 넘어가는 경로가 일대일대응된다는 사실을 알 수 있습니다.

**Exercise.** 사실 “스택 수열”의 아이디어를 활용해서도 개수를 셀 수 있습니다. “스택 수열”에서 스택에 들어가는 연산을 `+` , 스택에서 뽑는 연산을 `-` 라고 할 때, 모든 prefix에서 `+`개수가 `-` 개수보다 많거나 같음을 보이세요. 이러한 binary string을 Dyck word라고 합니다. Dyck word와 $231$-avoiding permutation이 일대일대응됨을 보이고, Dyck word의 개수가 $C _ {n} = \frac{1}{n+1}\binom{2n}{n}$임을 보이세요.



# General question: Permutation Pattern

보다 일반적으로, 길이 $k$인 순열 (pattern) $\pi$가 길이 $n$인 순열(text) $\sigma$에서 나타난다는 것은 어떤 $1 \le i _ {1} < \cdots < i _ {k} \le n$이 존재하여 $\sigma(i _ 1), \cdots, \sigma(i _ k)$ 중에서

- $\sigma(i _ {1})$이 $\pi(1)$번째로 작음
- …
- $\sigma(i _ k)$가 $\pi(k)$번째로 작음

의 조건을 만족하는 것을 말합니다. 엄밀하게 쓰면, increasing function $g : [k] \to [n]$이 존재하여 $\sigma \circ g \circ \pi^{-1} : [k] \to [n]$이 증가함수가 된다는 것을 의미합니다. 이 때 $\pi$를 $\sigma$의 pattern이라고 하고, $\pi \le \sigma$로도 씁니다. 당연히 모든 순열들의 집합에 주어진 “pattern order” $\le$는 poset의 조건을 만족합니다.

순열 $\sigma$에 pattern $\pi$가 나타나지 않는다면, (즉, 조건을 만족하는 increasing function $g$가 없다면) $\sigma$를 $\pi$-avoiding permutation이라고 합니다. $\pi = 231$에 대입해서 생각해보면 이전의 정의에 잘 맞아떨어지는 것 같습니다. 우리는 앞 단원에서 본 것처럼, 보다 일반적으로 어떤 순열이 $\pi$-avoiding permutation인지 판별하는 문제, 혹은 $\pi$-avoiding permutation의 개수를 세는 문제에 관심을 가져보기로 합니다.

$\pi = 231$일 때 값이 예쁘게 나오는 건 좋은데, 굳이 그렇다고 일반적인 $\pi$-avoiding permutation의 개수를 알아서 좋은 점이 무엇일까요? 이 장의 지면을 조금 나누어 그에 대한 motivation을 설명하고자 합니다.

**Definition.** (Permutation class) 길이가 가변인 순열들의 모임 $C$가 “hereditary property”를 만족한다고 하자. 즉, $\pi \in C$이면 모든 $\pi$의 pattern $\tau$에 대해 $\tau \in C$이다. 이 때 $C$를 permutation class라고 한다.

Permutation class는 수학에 있어 상당히 기본적인 성질인 “hereditary class”로 정의되었습니다. 따라서 stack-sortability와 같이 우리가 궁금해하는 많은 성질이 이 형태로 나타날 가능성이 크고, permutation class를 이해하면 우리가 궁금해하는 수많은 성질에 대한 해답을 가져다줄 가능성이 큽니다.

동시에, 다음의 정리가 알려져 있습니다.

**Theorem.** 모든 permutation class $C$는 “avoiding set”으로 나타낼 수 있다. 즉, 어떤 pattern들의 minimal set $\Pi$가 존재하여 $C = \{\sigma : \forall \pi \in \Pi \; \sigma \text{ is }\pi -\text{avoiding}\}$으로 나타낼 수 있다.

이 때 $\lvert \Pi \rvert = 1$인 경우, $C$를 특별히 principal permutation class라고 부릅니다. 결국 avoiding set이 곧 permutation class이고, 가장 간단한 경우인 principal permutation class에 관심이 가는 게 당연하다고 볼 수 있습니다.

Principal permutation class, 즉 $\pi$-avoiding set의 “크기”에 대해서도 관심이 많았습니다. 길이가 $n$인 순열 중 $\pi$-avoiding permutation의 개수는 얼마나 될까요? 이에 대해서 유명한 추측이 있었습니다.

**Theorem.** (Stanley-Wilf Conjecture, Proved by Marcus & Tardos 2004).

Permutation class의 크기는 singly exponential하다. 즉, 길이가 $n$인 순열 중 $\pi$-avoiding permutation의 개수는 어떤 $C$에 대해 $O(C^n)$꼴이다.

이러한 결과들은 Permutation Pattern Matching과 관련된 computational task에 큰 도움이 되곤 합니다. 다음 단원에서는 수많은 computational task에 대해 알아보고, 이 중에서 우리가 오늘 다룰 Gawrychowski (2022)을 소개합니다.

# Computational tasks on permutation patterns

### Detection

“Given $\pi$ and $\sigma$, determine if $\pi \le \sigma$”

안타깝게도 이 문제는 NP-complete입니다. (Bose 1998) 하지만 $k$를 고정했을 때는 $n$에 대해 Linear-time에 풀리는 FPT (Fixed-Parameter Tractible) 문제입니다. 가장 최신의 결과는 Fox (2013)의 $2^{O(k^2)} \cdot n$ 시간으로 알려져 있습니다. 이 사실에는 Stanley-Wilf Conjecture의 결과가 필요하며, 그 내용이 꽤 복잡한 편입니다.

### Counting

“Given $\pi$ and $\sigma$, count the number of $\pi$-occurrences on $\sigma$”. 다시 말해, $\sigma \circ g \circ \pi^{-1}$이 증가함수가 되는 증가함수 $g$의 개수를 묻는 문제입니다.

- $\pi$가 작은 경우, 이러한 $g$의 개수를 조합하여 만든 고전적인 통계량이 있습니다. 일반적으로 $\lvert \pi \rvert = k$인 경우를 $\sigma$의 $k$-profile들이라고 하며, Kendall’s tau correlation, Spearman’s $\rho$ function, Fisher-Lee rotation invariant measure 등이 알려져 있습니다.
- $\pi$의 길이가 $k = 4$인 경우 잘 알려진 자료구조 문제가 됩니다. 가장 자명한 알고리즘은 $O(n^k)$ 정도에 동작하겠지만, $k = 4$인 경우 $O(n^{3/2})$ 정도에 2차원 자료 구조 등을 활용하여 해결 가능합니다. (Even-Zohar, 2020)
- $k, n$이 큰 경우는 당연히 모두 어렵습니다. $f(k) \cdot n^{o(k / \log k)}$ 정도의 FPT algorithm이 알려져 있습니다.

복잡도가 $n$에만 영향을 받는 경우도 생각해볼 수 있습니다. 우선 단순히 모든 $g$를 열거해보는 $O(\binom{n}{k}) = O(2^n)$ 풀이가 naive입니다.

- $O(1.6181^n)$ 정도가 기존의 최적이었습니다. (Berendsohn 2016)
- 이후 소개할 논문에서는 “간단한” argument로 복잡도를 $O(n \cdot 2^{n / 2})$ 스케일로 개선합니다. (Gawrychowski 2021)

### Growth rate measure?

- 상당히 중요한 부분이라고 생각하지만, 아직은 순수 수학의 영역으로 보입니다.

### Misc.

최근 경시대회에 자주 얼굴을 비춘 “substring pattern match”는 이보다 훨씬 간단한 경우입니다. pattern이 모든 subsequence가 아니라 연속한 구간으로 제한되기 때문인데요, 전혀 다른 풀이법으로 해결할 수 있습니다.

[2021 Seoul Regional K. Stock Price Prediction](https://www.acmicpc.net/problem/23576)

# Counting occurrence solutions

$\sigma \circ g \circ \pi^{-1}$이 증가함수가 되는 $g$를 “occurrence solution”이라고 부르도록 합시다.

앞서 언급하였듯, 단순히 increasing function $g : [k] \to [n]$를 나열하여 occurrence solution인지 체크하는 방법은 최악의 경우 $\binom{n}{k} = O (2^{n})$ 의 시간이 필요합니다. 극단적으로 $\sigma$와 $\pi$가 모두 identity permutation인 경우가 이에 해당합니다. 따라서 다른 접근 방법으로 “$[n]$의 segment decomposition”을 생각합니다. 정의하기에 앞서 간단히 언급하면, 이는 $g \circ \pi^{-1}(1), \cdots, g \circ \pi^{-1}(k)$의 범위에 제약 조건을 걸어주는 $[n]$의 **특수한** 구간들 $[l _ {i}, r _ {i}]$ ($i = 1, \cdots, k$)입니다.

$g$가 segment decomposition $A = ([l _ {i}, r _ {i}]) _ {i = 1}^{k}$를 respect한다는 것은 각 $i$에 대해 $g \circ \pi^{-1} (i) \in [l _ {i}, r _ {i}]$를 만족한다는 것으로 정의합니다. 이 때 segment decomposition을 잘 정의해서, 다음과 같이 문제를 해결할 요량입니다.

1. Segment decomposition $A$에 대해, $A$를 respect하는 occurrence solution의 개수를 $O(n)$ 시간에 셀 수 있다.
2. Segment decomposition들의 모임 $\mathcal{S}$를 잘 잡아서, 모든 occurrence solution $g$에 대해 $g$가 respect하는 유일한 segment decomposition $A _ {g} \in \mathcal{S}$가 존재하도록 할 수 있다. 사실 $\lvert \mathcal{S} \rvert = O(2^{n/2})$이고, 따라서 모든 $A \in \mathcal{S}$에 대해 1번 과정을 수행해주면 전체 시간 $O(n \cdot 2^{n/2})$에 문제를 해결할 수 있다.

## Step 1. Segment decomposition & $A$-respecting solutions

**Definition.** $[n]$의 구간 $k$개 순열 $A = ([l _ {i}, r _ {i}]) _ {i=1}^{k}$가 다음 조건을 만족하면, $A$를 $[n]$의 segment decomposition이라고 부릅니다.

- $l _ {i} \le r _ {i}$ for $i = 1, \cdots, k$
- $r _ {i} \le l _ {i+1}$ for $i = 1, \cdots, k-1$

즉 $A$의 구간들은 양 끝점에서만 겹칠 수 있습니다. 굳이 $[l _ {i} ,r _ {i}]$들이 $[n]$을 전부 cover할 필요는 없지만, cover하지 않는 경우가 필요하진 않습니다. 가령 $[1, 2], [2, 3], [4, 7], [7, 8]$은 $[8]$의 segment decomposition입니다.

**Theorem.** $[n]$의 순열 $\sigma$, $[k]$의 순열 $\pi$, $[n]$의 segment decomposition $A$가 주어져 있을 때, $A$를 respect하는 occurrence solution을 $O(n)$에 셀 수 있다.

$A _ {i}$를 $A$의 첫 $i$개 구간으로 정의합시다. 이 때 $D(i, j)$를 아래 조건을 만족하는 increasing function $g^{i} : \pi^{-1}([i]) \to [n]$의 개수로 정의합시다.

- $g^{i}(p) \in [l _ {p}, r _ {p}]$ for all $p = \pi^{-1}(1), \cdots, \pi^{-1}(i)$.
- $\sigma \circ g^{i} \circ \pi^{-1}(i) = j$.

$g^{i} \circ \pi^{-1}(i) \in [l _ {i}, r _ {i}]$이므로, $j \in \sigma[l _ {i}, r _ {i}]$일 때만 $D(i, j) > 0$일 것입니다. 편의상 $\Sigma _ {i} := \sigma[l _ {i}, r _ {i}]$로 쓰겠습니다. 우리가 원하는 답, $A$를 respect하는 occurrence solution의 개수는 $\sum _ {j \in \Sigma _ {k}} D(k, j)$가 될 것입니다.

$D(i, j)$는 동적 계획법으로 채워 줍니다. 관계식은 아래와 같이 나옵니다.

- $D(0, 0) = 1$, $D(0, j) = 0$
- $\displaystyle D(i, j) = \sum _ {t \in \Sigma _ {i-1} \text {and } t < j} D(i-1, t)$

$[l _ {i}, r _ {i}]$가 최대 한 점에서만 겹칠 수 있기 때문에, $\sigma \circ g^{i-1} \circ \pi^{-1} (i-1) = t$ 값을 기준으로 동적 계획법의 transition 식을 쓸 수 있습니다.

이제 위 알고리즘을 $O(n)$ 시간에 구현해봅시다. $D(i, j)$를 그냥 계산하면 $O(n^{2})$ 정도의 시간이 필요할 겁니다.

$\sum _ {i = 1}^{k} \lvert \Sigma _ {i} \rvert \le n + k - 1$이므로, $\Sigma _ {1}, \cdots, \Sigma _ {k}$의 원소들을 전부 오름차순으로 나열해봅시다. $O(n)$ 시간에 구현하기 위해선 $t \in \Sigma _ {a} \text{ and } t < b$ 꼴의 $t$에 대해 $D(a, t)$ 를 다 합한 값을 관리해야 하는데, 정렬된 배열을 훑으면서 $\sum D(a, \ast)$을 업데이트해주면 됩니다.

## Step 2. Even-to-even decomposition family $\mathcal{S}$

이제 적절한 segment decomposition들의 모임 $\mathcal{S}$를 찾아봅시다.

$[n] _ {2}$를 $[n]$의 짝수들의 모임으로 정의하고, $E$를 $[k] _ {2}$에서 $[n] _ {2}$로 가는 increasing function들의 모임으로 정의합시다. 모든 increasing function $f : [k] \to [n]$에 대해, $\tilde{f}(2i) = 2\left\lfloor \frac{f(2i)}{2} \right\rfloor$로 정의하면 $\tilde{f} \in E$가 됩니다.

어떤 $f \in E$에 대해, 대응되는 segment decomposition $S(f)$를 다음과 같이 정의합시다.

- $l _ {1} = 1$, $l _ {2i} = f(2i)$.
- $r _ {2i} = \min(n, l _ {2i} + 1)$, $l _ {2i+1} = r _ {2i}$.
- $r _ {2i+1} = l _ {2i+2}$ for $2i + 1 < k$, $r _ {k} = n$ if $k$ odd.

어떤 increasing function $h : [k] \to [n]$가 $S(f)$를 respect한다면, 최소한 $h(2i)$는 $\pm 0.5$로 고정되는 것을 알 수 있습니다.

$\mathcal{S} := \{S(f) : f \in E\}$로 정의하면 $\lvert\mathcal{S}\rvert = \lvert E \rvert$이고, $\lvert E \rvert = \binom{\lfloor n / 2 \rfloor}{\lfloor k / 2 \rfloor} = O(2^{n/2})$입니다. 따라서 아래 정리를 보이기만 하면 $\mathcal{S}$의 모든 원소에 대해 Step 1을 수행하여 문제를 해결할 수 있습니다.

**Lemma.** 모든 occurrence solution $h : [n] \to [k]$에 대해, $h$가 respect하는 $\mathcal{S}$의 원소가 정확히 하나 존재한다.

*Proof.* 아래 두 명제를 보이는 것으로 충분합니다.

1. 위에서 정의된 $\tilde{h}$에 대해, $h$는 $S(\tilde{h})$를 respect한다. 
   - $h(2i) \in [l _ {2i}, r _ {2i}]$는 자명합니다.
   - $l _ {1} \le h(1)$이고, $l _ {2i+1} \le l _ {2i} + 1 = \tilde{h}(2i) + 1 \le h(2i+1)$을 만족합니다.
   - $2i + 1=k$이면 $r _ {k} \ge h(2i+1)$이고, 아니라면 $r _ {2i+1} = \tilde{h}(2i+2) \ge h(2i+1)$이 성립하므로, $h(2i+1) \in [l _ {2i+1}, r _ {2i+1}]$을 보일 수 있습니다. $\square$
2. $g \neq \tilde{h}$에 대해 $h$는 $S(g)$를 respect하지 않는다.
   - $g(2j) \neq \tilde{h}(2j)$라고 두면, 최소한 $\lvert g(2j) - \tilde{h}(2j) \rvert \ge 2$입니다. 따라서 $\lvert{g(2j) - h(2j)}\rvert + \lvert{h(2j) - \tilde{h}(2j)}\rvert \ge 2$가 되는데, $h$가 $S(g)$를 respect하려면 $\lvert g(2j) - h(2j) \rvert = \lvert h(2j) - \tilde{h}(2j) \rvert = 1$인 경우뿐입니다. 이 때 $g(2j) = h(2j) - 1 = \tilde{h}(2j)$일 수 없으니 $g(2j) = h(2j) + 1$이고, 이 경우에 $h(2j) \notin [g(2j), g(2j) +1]$입니다. $\square$

## Epilogue. Lower-bound problem

이 접근에서 사뭇 비자명한 부분은 갑자기 even-to-even function의 모임 $E$를 잡더니, 대뜸 $\mathcal{S} = \{S _ {g} : g \in E\}$를 정의한 것입니다. 우리가 $\mathcal{S}$에게 기대하는 성질은, 고정된 $\sigma, \pi$와 한 occurrence solution $f$에 대해 $f$가 $S \in \mathcal{S}$ respect하는 $S$가 “단 하나” “존재하는” 조건입니다. 이 조건을 만족하는 $\mathcal{S}$를 더 작게 잡을 수 있다면 역시 문제를 $O(n \lvert \mathcal{S} \rvert)$에 해결할 수 있지 않을까요?

논문의 마지막 챕터는 $\sigma = \mathrm{id} _ {n}$, $\pi = \mathrm{id} _ {k}$인 경우에도 $\mathcal{S}$를 크게 잡을 수밖에 없다는 것을 보여주고 있습니다.

$P$를 $[\lfloor k / 2 \rfloor] \to [\lfloor (n-1) / 2 \rfloor]$로 가는 increasing function의 모임이라고 두고, $f \in P$에 대해 $g _ {f} : [k] \to [n]$을 $g _ {f}(2i-1) = 2f(i) - 1$, $g _ {f}(2i) = 2f(i)$, $k$가 홀수인 경우에 $g _ {f}(k) = n$으로 정의합시다. 이 때 $Q = \{g _ {f} : f \in P\}$로 두면 $\lvert Q \rvert = \lvert P \rvert = \binom{\lfloor (n-1) / 2 \rfloor}{\lfloor k / 2 \rfloor}$입니다.

이제 모든 $g _ {f} \in Q$에 대해, 각각이 respect하는 $[n]$의 segment decomposition이 모두 달라야 한다는 사실을 보이면 $\lvert \mathcal{S} \rvert \ge \lvert Q \rvert$가 되고, 사실상 worst case에서 $\lvert \mathcal{S} \rvert = \Omega(2^{n / 2})$가 된다는 것을 보일 수 있습니다. $g _ {f}$가 어떤 $A _ {f} = ([l _ {i}, r _ {i}]) _ {i = 1}^{k}$를 respect한다고 하면, $2f(i) - 1 \le r _ {2i-1} \le l _ {2i} \le 2f(i)$입니다. 따라서 $\lceil r _ {2i-1} / 2\rceil = f(i)$가 성립해야 하고, 이는 곧 모든 $A _ {f}$가 distinct하다는 것을 의미합니다. $\square$

## References

- [Gawrychowski 2022] Gawrychowski, Pawel & Rzepecki, Mateusz. (2022). *Faster Exponential Algorithm for Permutation Pattern Matching*. Symposium On Simplicity of Algorithm (SOSA 2022), 10.1137/1.9781611977066.21. 
  - Main 논문입니다. 주저자 Gawrychowski는 이전에 소개한 Global min cut 문제의 SOTA 보유자로도 알려져 있고, 굉장히 다양한 활동을 하고 있습니다.
- [Even-zohar 2021] Chaim Even-Zohar and Calvin Leng. 2021. Counting small permutation patterns. In *Proceedings of the Thirty-Second Annual ACM-SIAM Symposium on Discrete Algorithms* (*SODA '21*). Society for Industrial and Applied Mathematics, USA, 2288–2302.

- [Marcus 2004] Adam Marcus, Gábor Tardos, 2021. *Excluded permutation matrices and the Stanley–Wilf conjecture*, Journal of Combinatorial Theory, Series A, Volume 107, Issue 1.