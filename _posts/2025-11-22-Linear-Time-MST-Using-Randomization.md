---
layout: post
title: "Linear Time MST Using Randomization"
date: 2025-11-22
author: mhy908
tags: [algorithm, data-structure]
---

## 개요

Minimum Spanning Tree(MST)는 연결된 무방향 그래프에서 모든 정점을 포함하면서 총 간선 가중치의 합이 최소가 되는 스패닝 트리를 의미한다. 전통적으로 Kruskal, Prim, Boruvka와 같은 결정적 알고리즘이 널리 알려져 있다.

그러나 위 알고리즘들 모두 전체 그래프를 스캔하고, 정렬 혹은 우선순위 큐 등에 간선 정보를 담는 과정에 의해 시간복잡도는 $O(ElogV)$정도에 머무른다. 그 와중 상대적으로 간단한 무작위 알고리즘을 이용해 이 시간복잡도를 $O(E)$ 정도로 줄이는 아이디어가 있어, 이를 소개해보고자 한다.

## Boruvka's Algorithm

Boruvka's Algorithm은 MST를 유도하는 대표적인 알고리즘으로, Prim이나 Kruskal에 비해 상대적으로 덜 알려져 있기에 간단하게 짚고 넘어가고자 한다.

Boruvka Step은 다음과 같이 정의한다 :

- 각 정점에서, 자신에서 바깥으로 나가는 최소 가중치 간선을 선택한다.
- 선택한 간선들을 사용해, 컴포넌트끼리 하나의 큰 정점으로 병합한다.

이 과정에서 선택한 간선들의 집합을 MST에 추가하고, 전체 그래프가 하나의 정점으로 병합될때까지 위 Boruvka Step을 반복하면 알고리즘이 종료된다.

한번의 Step에서, 전체 정점의 개수는 기존의 절반 이하로 줄어들기 때문에 전체 시간복잡도는 $O(ElogV)$이다.

## Sampling Lemma

편의를 위해 간선의 가중치가 전부 unique하다고 가정하자. 실제로는 가중치가 같은 간선이라도, 번호를 차례로 부여하여 대소관계를 정의할 수 있다.

$w(x, y)$를 정점 $x$와 $y$를 잇는 간선의 가중치라 정의하자. 또한, $w_A(x, y)$를 임의의 포레스트 $A$ 위에서 정점 $x$와 $y$를 잇는 경로의 가중치 최대값이라 정의하자. 만약 $A$ 위에서 $x$와 $y$가 연결되어 있지 않다면, $w_A(x, y)=\infty$로 정의한다.

> **Sampling Lemma** : 임의의 연결 그래프 $G$에 대해 각 간선을 독립적으로 확률 $0<p\leq 1$ 로 선택하여 $H$라는 그래프를 구성하였을 때, $F$를 $H$의 MST라 정의하자. $V$를 $H$의 정점의 개수라 할때 $G$의 간선들 중, $w(u, v)\leq w_F(u, v)$를 만족하는 간선의 개수의 기대값은 최대 $V/p$이다.

간선을 가중치에 대한 오름차순으로 정렬하자. 그리고 다음과 같은 변형된 Kruskal Algorithm을 고려해보자.

초기에 $F'$와 $H'$는 각각 비어잇는 그래프이다.

- $p$의 확률로 간선 $(u, v)$를 $H'$에 추가한다. 선택되지 않았다면 아래 단계를 무시한다.
- $F'$에서 이미 $u$와 $v$가 하나의 컴포넌트라면 이 간선을 무시한다. 이미 $w(u, v)> w_{F'}(u, v)$이기 때문이다.
- 그렇지 않다면 $(u, v)$ 를 $F'$에 추가한다.

이 과정을 반복해 얻은 $F'$와 $H'$는, 위 Lemma의 $F$와 $H$과 확률적으로 대등하다.

이제, $w(u, v)\leq w_F(u, v)$인 간선의 개수의 기대값을 생각해보자.

확률 $p$를 고려하기 전에, 별개로 $F$에서 $u$와 $v$가 연결되어있는지는 알 수 있는 정보이다. 생각의 편의를 위해 $F$에서 이미 연결되어있다면 확률 $p$의 빨간 동전을 던지고, 그렇지 않다면 확률 $p$의 파란 동전을 던진다고 생각해보자.

우리가 구하고자 하는 기대값은 결국 파란 동전을 던진 횟수와 동일하다. 그러나, $F$는 포레스트이므로 $F$에 들어갈 수 있는 간선의 개수는 $V-1$로 제한되어 있다. $V-1$개의 간선이 $F$에 전부 들어갔다면, 파란 동전을 더이상 던지지 않게 된다.

즉, $w(u, v)\leq w_F(u, v)$인 간선의 개수의 기대값은, 독립적으로 확률 $p$인 동전을 뒤집었을 때 총 $V-1$개의 앞면이 나오기 까지의 시행 횟수의 기대값과 동일하며, 이는 파라미터 $V$와 $P$를 가지는 음이항분포 (Negative Binomial Distribution)을 따른다. 그리고 이 분포의 평균은 $V/p$로 알려져 있다.

이 증명에서 짚고 넘어갈 사실은, 그래프의 구조와 관계없이 기대값의 상한을 $V/p$로 결정했다는 사실이다. 즉, 이 Lemma를 사용한 Randomization Algorithm은 그래프의 구조와 무관하게 작동함을 알 수 있다. CP적인 관점에서는, **저격이 거의 불가능하다** 라고도 볼 수 있다.

## Linear Time MST Using Sampling

다음 알고리즘은 주어진 그래프에 대한 Minimum Spanning Forest를 리턴한다.

- 두번의 연속적인 Boruvka Step을 적용한다. 그 압축된 그래프를 $G$라 하자.
- $G$에서, 각 간선에 대해 확률 $1/2$로 Sampling한 Subgraph $H$를 추출한다.
- $H$에서 재귀적으로 Minimum Spanning Forest $F$를 계산한다. 이때 $H$의 간선 중, $F$에 포함되지 않는 간선들을 $G$에서 제거한다.
- $G$에서 재귀적으로 Minimum Spanning Forest를 계산하고, 이를 리턴한다. Boruvka Step에서 추출한 간선들과 같이 리턴한다.

임의의 Subgraph에서 MST에 포함되지 않는 간선은 전체 그래프에서도 MST에 포함되지 않는 성질을 고려하였을 때, 위 알고리즘은 올바른 결과를 도출한다.

## Analysis

편의상 $H$에서 MST를 구하는 과정에서의 재귀를 left recursion, 그 다음 간선이 일부 제거된 $G$에서의 재귀를 right recursion이라 하자.

> **Thm 1**. Worst-Case 시간복잡도는 $O(ElogV)$로, 기존 Boruvka 알고리즘의 시간복잡도와 동일하다.

처음 입력으로 주어진 그래프에서의 정점의 개수를 $v$, 간선의 개수를 $e$라 하자. Boruvka Step을 두번 거치면, 그래프의 정점의 개수는 $v/4$ 이하가 된다.

또한, $F$에 포함되는 간선의 개수는 많아봤자 $v/4$개이다. 이 간선들은 left/right recursion 모두에게 호출될 것이다. 

Boruvka Step 두번을 거치면서 간선이 최소 $3v/4$개는 완전히 제거되므로, 이를 종합했을 때 left/right recursion에 포함된 간선 개수의 합은 $e$보다 항상 작아진다.

반면 정점 개수는 매 step마다 $1/4$로 줄어드므로, recursion depth는 최대 $O(logV)$이다. 따라서 Worst-Case 시간복잡도는 $O(ElogV)$이다.

> **Thm 2**. 평균 시간복잡도는 $O(E)$이다.

시간복잡도는 전체 recursion tree에서 등장하는 간선의 개수의 합과 같다.

루트 노드, 혹은 임의의 부모 노드에 대한 right recursion에서 시작하는, left recursion만 따라 내려가는 left chain을 생각해보자.

left recursion은 $1/2$의 확률로 간선을 샘플링한 그래프가 주어지므로, 부모 노드의 간선 개수를 X, left recursion의 간선 개수를 Y라 하면 기대값의 선형성에 의해 $E[X]/2>E[Y]$ 를 만족한다. 따라서, 각 left chain의 최상단 노드의 간선 개수가 k라면, 그 chain 전체에 있는 간선 개수의 기대값은 최대 $k+k/2+k/4+...=2k$가 된다.

루트 노드에는 간선이 $E$개 있으므로, 여기서 시작하는 left chain의 간선 개수의 기대값은 $2E$개이다.

각 right recursion으로 주어지는 간선 개수는 $p=1/2$로 주어진 Sampling Lemma를 이용해 그 기대값의 상한이 주어지며, 이는 각각의 right recursion 정점 개수의 두배이다.

right recursion에 해당하는 정점들의 개수의 상한은 최대 $\sum_{d=1}^{\infty} \frac{2^{d-1} V}{4^{d}} = \frac{V}{2}$ 로 표현할 수 있다. 여기서 $d$는 각 right recursion의 depth, $2^{d-1}$은 각 depth에서 출현하는 right recursion의 개수, $\frac{V}{4^d}$는 그 depth에서의 정점 개수의 상한이다. 따라서, Sampling Lemma에 의해 right recursion의 간선 개수의 기대값은 이 두배인 $V$로 주어진다.

결론적으로 기대값의 선형성에 의해, 전체 recursion tree에서 등장하는 간선 개수의 기대값의 상한은 $2E+2V$로 주어지며, 이는 $O(E)$이다.

> **Thm 3**. 이 알고리즘은 $1-exp(-\Omega(E))$의 확률로 $O(E)$에 동작한다.

right recursion들과 left recursion을 따로 고려해보자.

> **Thm 3.1**. right recursion에서 등장하는 간선 개수의 합이 $3E$ 이상일 확률의 상계는 $exp(-\frac{E}{12})$이다.

기존 Sampling Lemma의 증명과 유사하게, 이번엔 단일 시행이 아니라 right recursion 전체들로 이루어진 전역적인 상황을 고려해보자.

기존 증명과 마찬가지로, $F$(샘플링한 그래프의 MST)에 들어갈 수 있는 상한은 Thm 2.의 증명 과정에서 구한 결과를 인용해 최대 $V/2$로 주어진다. 여기서는 계산의 편의를 위해, $V/2 \leq E$ 이므로 이 상한을 $E$로 늘려본다.

간선 개수가 $3E$보다 클 확률은, 확률이 $1/2$인 파란 동전을 뒤집어 앞면이 $E$회 나오기까지 $3E$번 이상의 시행 횟수가 필요할 확률과 같으며, 이 확률은 Chernoff Bound에 의해 다음과 같은 상계를 가진다.

$exp(-\frac{\mu \delta^2}{2}) = exp(-\frac{1}{2} \cdot \frac{3E-1}{2} \cdot (\frac{E+1}{3E-1})^2) = exp(−\frac{(E+1)^2}{12E-4}​) < exp(-\frac{E}{12})$

증명에서도 보다시피, $3E$는 여유롭게 잡은 임의의 배수이며, $4E$, $5E$, ... 로 잡으면 상계는 더욱 낮아진다.

> **Thm 3.2**. root recursion과, right recursion들에서 등장하는 간선 개수의 합을 $E'$이라 할 때, left recursion들에서 등장하는 간선 개수의 합이 $2E'$ 이상일 확률의 상계는 $exp(-\frac{E'}{12})$이다.

Thm 2. 에서 언급한 left chain들을 고려하자.

left recursion은, 단순히 parent recursion에서 일부 간선들을 지우고(Boruvka Step), 이를 $1/2$ 확률로 sampling한 집단이라 볼 수 있다. 상계를 구하고 있으므로, 여기서는 Boruvka Step의 영향을 무시하자.

따라서 left chain에서 등장하는 간선 개수의 합이 $2E'$보다 클 확률은, 확률이 $1/2$인 동전을 던져 앞면이 $E'$회 나오기까지 $3E'$번 이상의 시행 횟수가 필요할 확률과 같으며, 이는 Thm 3.1과 마찬가지로 $exp(-\frac{E'}{12})$라는 상계를 가진다. 여기서 $3E'$이 아닌 $2E'$인 이유는, 각 $E'$개의 간선에 대해 처음 동전을 굴리는 것은 root 혹은 right recursion에서 샘플링하는 과정에 해당하기 때문이다.

> **Thm 3.3** right recursion에서 등장하는 간선 개수의 합이 $3E$ 이하이며, left recursion에서 등장하는 간선 개수의 합이 $8E$ 이하일 확률의 하계는 $1-2exp(-\frac{E}{12})$이다. 또한 이 하계는 전체 알고리즘에서 등장하는 간선 개수의 합이 $12E$ 이하일 확률의 하계 또한 $1-2exp(-\frac{E}{12})$이다.

Thm 3.1과 Thm 3.2에 의해 직접 유도된다.

$(1-exp(-\frac{E}{12}))(1-exp(-\frac{4E}{12}))>1-2exp(-\frac{E}{12})$

Thm 3.3에 의해, Thm 3 또한 참임을 알 수 있다.

## Reference

Karger, D. R., Klein, P. N., & Tarjan, R. E. (1995). A randomized linear-time algorithm to find minimum spanning trees. Journal of the ACM (JACM), 42(2), 321-328.
