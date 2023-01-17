---
layout: post
title: "Dynamic MSF with Subpolynomial Worst-case Update Time (Part 1)"
author: koosaga
date: 2020-12-20
tags: [graph-theory, theoretical-computer-science]
---

# Dynamic MSF with Subpolynomial Worst-case Update Time (Part 1)

그래프에서 최소 스패닝 트리 (Minimum spanning tree)를 구하는 문제는 아주 잘 알려져 있고, 일반적으로 가장 처음 공부하게 되는 그래프 알고리즘에 속하며, 매우 다양한 개념에 응용된다. 그래프의 최소 스패닝 트리는 크루스칼 알고리즘을 통해서 $O(m \log m)$ 에 효율적으로 구할 수 있다. 하지만 그래프에서 간선이 추가되고 제거되는 등의 업데이트가 가해진다면, 이 알고리즘은 매 갱신마다 전체 간선 리스트를 전부 순회해야 하니 더 이상 효율적이지 않게 된다. 이렇게 그래프의 일부가 바뀔 때에도 여러 잘 알려진 문제를 효율적으로 해결하는 알고리즘들을 Dynamic Graph Algorithm이라고 한다.

이 글에서는, 다음과 같은 연산을 지원하는 알고리즘을 소개한다:

* `Preprocess(G)`: 알고리즘에 그래프 $G$ 를 입력한다. 알고리즘은 $G$ 의 최소 스패닝 포레스트 (Minimum spanning forest, MSF) $F$ 를 반환한다.
* `Insert(u, v, w)`: $G$ 에 가중치가 $w$ 인 간선 $(u, v)$ 를 추가한다. 알고리즘은 변경 이후 $F$ 에 가해진 변화 (추가되거나 제거된 간선) 의 리스트를 반환한다.
* `Delete(u, v)`: $G$ 에 간선 $(u, v)$ 를 제거한다. 알고리즘은 변경 이후 $F$ 에 가해진 변화 (추가되거나 제거된 간선) 의 리스트를 반환한다.

이 때 우리는 각 `Insert`, `Delete` 연산, 즉 그래프를 바꾸는(update) 연산의 시간 복잡도를 최소화하고 싶다. 이 문제를 *Dynamic MSF Problem* 이라고 부른다. 만약 Insert 연산만이 존재하면 *Incremental MSF Problem*, Delete 연산만이 존재하면 *Decremental MSF Problem* 이라고 부른다. Dynamic MSF Problem은 잘 알려진 Dynamic Connectivity Problem의 일반화라는 것을 알아두자. 

프로그래밍 대회에서 Dynamic MSF Problem은 이미 여러 번 출제된 적이 있다. 예를 들어 [BOJ 7148 Winter Roads](https://www.acmicpc.net/problem/7148), [BOJ 10724 판게아 2](https://www.acmicpc.net/problem/10724), [SGU 529 It's time to repair the roads](https://codeforces.com/problemsets/acmsguru/problem/99999/529) 등이 예시이다. 또한, Dynamic MSF를 응용한 문제들도 여럿 있고 ([JOI Open 2018 Collapse](https://www.acmicpc.net/problem/16793)), 다른 알고리즘이 의도되었지만 더 일반화된 Dynamic MSF를 사용해서 해결할 수 있는 문제들도 있다 ([서울대 2020 E](https://www.acmicpc.net/problem/19854)). 

이론적인 분야에서 Dynamic MSF Problem의 중요성은 Dynamic graph algorithm의 중요성과 그대로 맞닿아 있다. Dynamic graph algorithm은 빠른 속도로 변하는 거대한 그래프들에서 중요한 정보들을 추려내는 데이터 마이닝 등의 분야에 도움이 된다. 일반적으로 그래프에서 가장 기초적으로 배우는 알고리즘들은 최단 경로와 최소 스패닝 트리, 최소 컷 문제들이다. 이 중 사실상 진전이 없는 Dynamic shortest path problem을 논외로 두면, Dynamic MSF problem은 이 중 가장 연구가 활발하게 되는 분야이다. 특히, 최소 컷을 효율적으로 푸는 것은 Dynamic MSF 문제를 빠르게 푸는데 연관이 있고 반대 방향으로도 연관이 있는 등, 결과에서 사용된 테크닉이나 결과 그 자체가 다른 근본적인 문제들에 대한 새로운 발전을 함의한다.

Dynamic MSF Problem은 매우 중요한 만큼 오랜 시간 다양한 연구가 진행되어 왔다. 그 중 중요한 결과를 추리자면 다음과 같다.

* $O(\sqrt m)$ update time (Frederickson. 1985) 
* $O(\sqrt n)$ update time (Eppstein et al. 1992) 
* $O(\log^4 n)$ *amortized* update (Holm et al. 1998)
* $O(n^{0.5 - \epsilon})$ update time for some constant $\epsilon > 0$, *Las Vegas* (Wulff-Nilsen. 2017)
* $O(n^{0.4 + o(1)})$ update time, *Monte Carlo* (Nanongkai, Saranurak. 2017)
* $O(n^{0.49306})$ update time, *Las Vegas* (Nanongkai, Saranurak. 2017)

이 글에서는 Dynamic MSF Problem을 $O(n^{o(1)})$ update time에 해결하는 *Las Vegas* 알고리즘을 소개한다. 고로, 비교적 오래 전 분석이 끝난 *amortized update* 경우를 제외하면, 이 알고리즘은 worst-case에 Dynamic MSF Problem을 $O(n^{o(1)})$ 에 해결할 수 있느냐 하는 오랜 난제를 해결한다. 이 때, $o(1)$ 은 $O(\log \log \log n / \log \log n)$ 을 숨긴다. 

## Preliminaries

글을 이해하는 데 있어서 중요할 수 있으나 설명하지 않은 사실은 인용 레퍼런스를 bold하여서 강조한다. 주제에 대한 이해를 높이고 싶다면 해당 레퍼런스 참고하라.

어떠한 사건이 임의의 상수 $c$ 에 대해서 $1 - \frac{1}{n^c}$ 의 확률 이상으로 일어남이 보장된다면, 이 사건은 *높은 확률로* 일어난다고 정의한다. 어떠한 알고리즘이 Las Vegas 알고리즘이라는 것은, *높은 확률로* 시간 복잡도를 보장하는 알고리즘을 뜻한다. 이 점에서 항상 시간 복잡도를 보장하는 일반적인 알고리즘과는 다르다.

이 문제에서 그래프는 인접 리스트의 형태로 관리된다. 임의의 정점 $v$ 가 주어지면 $v$ 의 인접 리스트의 포인터를 $O(1)$ 에 알 수 있다 (배열을 사용하여). 이러한 조건을 사용하는 이유는, 또한 차수가 1인 노드들의 리스트를 관리할 수 있다고 가정한다. 

MSF에 대한 다음과 같은 자명한 사실을 사용한다.

* **Fact 1**. 임의의 간선 집합 $E_1, E_2$ 에 대해 $MSF(E_1 \cup E_2) \subseteq MSF(E_1) \cup MSF(E_2)$.
* **Fact 2**. 그래프 $G$ 와, $G$ 의 서로 다른 두 정점을 contract해서 만든 그래프 $G^\prime$ 에 대해, $MSF(G^\prime) \subseteq MSF(G)$.

다음과 같은 사실들이 잘 알려져 있다.

* **Fact 3.** $n$ 개의 노드를 가지고, 간선에 가중치가 있는 그래프에서, 그래프의 간선들이 항상 포레스트를 이룬다는 조건으로 다음과 같은 시간 복잡도에 연산을 지원하는 알고리즘이 존재한다.

  * 전처리: $O(n \log n)$
  * 간선 추가: $O(\log n)$ 
  * 간선 제거: $O(\log n)$
  * 두 정점 $u, v$ 가 연결되어 있는지 확인하고, 그렇다면 $u, v$ 를 잇는 단순 경로 상에 가장 가중치가 큰 간선을 반환: $O(\log n)$

  *Proof.* Top tree라고 하는 자료구조를 사용하면 된다. **[(Alstrup et. al. 2003)](https://arxiv.org/pdf/cs/0310065.pdf)** 만약 간선 추가/제거가 없다면 Sparse table로, Amortization이 허용된다면 Link-cut tree로 할 수 있음이 알려져 있다.

* **Fact 4**. $m$ 개의 초기 간선이 있는 크기 $n$ 의 그래프에서 $\tilde{O}(m)$ 시간 전처리 후 $\tilde{O}(n)$ 시간 worst-case 업데이트를 보장하는 결정론적 Dynamic MST 알고리즘이 존재한다. 

  * 중복 간선을 적절히 처리한 후 Frederickson이 제안한 $O(\sqrt m)$ worst-case Dynamic MST 알고리즘을 사용하면 된다.

* **Fact 5**. $n$ 개의 정점과 $m$ 개의 간선이 있으며 간선 삭제를 지원하는 그래프 $G =(V, E)$, 그리고 정점 부분집합 $S \subseteq V$ 를 생각하자. 이 때, 임의의 순간에 non-tree edge들 $E(G) - MSF(G)$ 은 정확히 하나의 끝점이 $S$ 에 속한다는 조건을 만족하며, 모든 정점 $u \in V \setminus S$ 는 상수 차수를 가진다는 조건을 만족해야 한다. 이 때 $G, S$ 를 $\tilde{O}(m)$ 시간에 전처리하며 각 간선 삭제를 $\tilde{O}(S)$ 시간에 처리할 수 있는 Decremental MSF 알고리즘이 존재한다. [**(Wulff-Nilsen et. al. 2017)**](https://arxiv.org/pdf/1611.02864.pdf)

## Chapter 1. The Extended Unit Flow Algorithm

* 그래프 $G = (V, E)$.
* *source function* $\Delta: V \rightarrow \mathbb{Z}_{\geq 0}$.
* *sink function* $T : V \rightarrow \mathbb{Z}_{\geq 0}$.

으로 이루어진 인스턴스 $\Pi = (G, \Delta, T)$ 에 대해서 다음과 같은 것을 정의한다.

* *preflow* 는 함수 $f : V \times V \rightarrow \mathbb{Z}$ 로 모든 $(u, v) \in V \times V$ 에 대해서 $f(u, v) = -f(v, u)$이며 모든 $(u, v) \in (V \times V) - E$ 에 대해 $f(u, v) = 0$ 을 만족한다.
* *preflow* 가 *source-feasible* 하다는 것은 $\sum_u f(v, u) \leq \Delta(v)$ 를 만족한다는 것을 뜻한다.  편의상 $f(v) = \Delta(v) + \sum_{u} f(u, v)$ 라 정의하면, $0 \le f(v)$ 이다.
* *preflow* 가 *sink-feasible* 하다는 것은 $\Delta(v) + \sum_u f(u, v) \leq T(v)$ 를 만족한다는 것을 뜻한다. 다르게 쓰면 $f(v) \le T(v)$ 이다.
* *preflow* 가 *source-feasible* 하며 *sink-feasible* 하면 이를 *flow* 라고 부른다.
* $cong(f) = max_{(u, v) \in V \times V} f(u, v)$ 를 $f$ 의 *congestion* 이라고 부른다.

이렇게 말하면 되게 복잡해 보이지만 네트워크 플로우를 이해하고 있다면 어렵지 않은 개념이다. 그래프 $G = (V, E)$ 의 모든 간선에 무한한 가중치를 주고, source에서 $v \in V$로 용량 $\Delta(v)$의 간선을, $v \in V$  에서 sink로 용량 $T(v)$ 의 간선을 이어주자. 약간 다른 점은, source에서 흘려줄 때 $\Delta(v)$ 이하의 유랑을 흘려주는 것이 아니라 **정확히** $\Delta(v)$ 의 유량을 흘려줘야 한다는 것이다.  source와 sink를 잇는 간선을 무시하고 flow를 흘려주면 preflow, source에서 $\Delta(v)$ 를 흘려줬더니 sink에서 나가는 플로우 양이 음수가 아니면 source-feasible flow, sink에서 나가는 플로우 양이 $T(v)$ 이하면 sink-feasible flow이다. 둘 다 만족할 경우 flow가 된다. 이제 이러한 직관 하에 다음과 같은 것을 또 정의하자.

* $ex_f(v) = max(f(v) - T(v), 0)$ 은 $f$ 에 대한 $v$ 의 *excess supply* (과잉 공급) 을 뜻한다. 
* $ab_f(v) = min(f(v), T(v))$ 은 $f$ 에 대한 $v$ 의 *absorbed supply* (흡수된 공급) 을 뜻한다. 

모든 $v$ 에 대해 $ex_f(v) + ab_f(v) = T(v)$ 이며, preflow가 sink-feasible함과 $\forall v.  ex_f(v) = 0$ 임은 동치이다. 편의를 위해

* $\Delta(\cdot) = \sum_v \Delta(v)$ 를 **총 공급량**
* $T(\cdot) = \sum_v T(v)$ 를 **총 용량**
* $ex_f(\cdot) = \sum_v ex_f(v)$ 를 **총 과잉량**
* $ab_f(\cdot) = \sum_v ab_f(v)$ 를 **총 흡수량** 

으로 정의한다.

*Remark 1.1 (입출력)*. 그래프 $G$ 는 인접 리스트의 배열 형태로 주어지기 때문에 알고리즘의 매 호출마다 전부 복사할 필요가 없다. source, sink function은 $\{(v, \Delta(v))\Delta(v) > 0\}, \{(v, T(v))  T(v) < \deg(v)\}$ 인 집합 형태로 주어진다. 출력은, $\{((u, v), f(u, v))  f(u, v) \neq 0\}$ 인 집합이다. 이 집합이 주어지면, $\{(v, ex_f(v))  ex_f(v) > 0\}$ 인 집합과 $\{(v, ab_f(v))  ab_f(v) > 0\}$ 인 집합 역시 자명하게 계산할 수 있다.

*Remark 1.2*. Remark 1.1에서 유추 가능하지만 이 글에서는 $T$ 가 $\forall v.T(v) \le \deg(v)$ 를 만족한다고 가정한다. preflow를 계산할 때는 $\Delta(v), T(v)$ 모두에 *가짜 공급* $\overline{T}(v) = \deg (v) - T(v)$ 를 더해준다. 이렇게 되어도 문제는 여전히 동치고, $T(v) = \deg(v)$ 를 만족한다고 가정할 수 있게 된다. $\overline{T}(\cdot) = 2m - \sum_v T(v)$ 를 **총 가짜 공급량** 이라고 정의한다. 이 항은 소개할 알고리즘의 시간 복잡도에 등장할 것이다.

마지막으로, 그래프 $G = (V, E)$ 에 대해

* $V$ 의 임의의 비지 않은 진부분집합 $S \subsetneq V$ 를 컷이라고 부른다. 
* 컷의 *volume* 은 $vol(S) = \sum_{v \in S} deg(v)$ 이다.
* 컷의 *크기* $\delta(S)$ 는 $S$ 와 $V\setminus S$ 를 오가는 간선의 개수를 뜻한다. 
* 컷의 *conductance* (전도율) 은 $\phi(S) = \frac{\delta(S)}{min(vol(S), vol(V - S))}$ 이다. 
* 그래프의 *conductance* (전도율) 은 $\phi(G) = min_{S \subset V, S \neq \emptyset} \phi(S)$ 이다.

전도율이라는 개념이 생소한데, 식을 놓고 보면 컷의 크기를 최대 가능한 컷의 크기로 나눈 개수가 된다. 즉, 컷을 잘라냈을 때 실제로 각 조각에서 얼마나 많은 간선이 컷에 기여했는가를 나타낸다. 예를 들어서 컷으로 잘라냈을 때 전도율이 낮다면, 이는 해당 컷을 잘라도 많은 간선들이 보존되는 편이라고 보면 된다. 전도율이 0이라면 $S$ 는 연결되지 않은 컴포넌트이다. 전도율이 1이라면 컷의 한 쪽은 독립 집합이다.

이제 Chapter 1의 **Main Theorem** 을 소개한다.

**Theorem 1.3 (Extended Unit Flow Algorithm)** Extended Unit Flow Algorithm이라는 알고리즘은 입력으로

* $m$ 개의 간선을 가진 그래프 $G = (V, E)$ (다중 간선이 있을 수 있으나 루프는 없다.)
* 양의 정수 $h \geq 1, F \geq 1$
* $\forall v. \Delta(v) \le F \deg(v)$ 를 만족하는 *source function* $\Delta: V \rightarrow \mathbb{Z}_{\geq 0}$
* $ \Delta(\cdot) \le T(\cdot)$ 와 $\forall v. T(v) \le \deg(v)$ 를 만족하는 *sink function* $T : V \rightarrow \mathbb{Z}_{\geq 0}$.

을 받으면, $O(hF(\Delta(\cdot) + \overline{T}(\cdot)) \log m)$ 시간에

* $cong(f) \le 2hF$ 인  *source-feasible preflow* $f$ 
* $ex_f(\cdot)$
* 만약 $ex_f(\cdot) \neq 0$ 일 경우, 전도율 $\phi(S) < \frac{1}{h}$ 이며 volume $vol(S) \geq \frac{ex_f(\cdot)}{F}$ 인 집합 $S \subseteq V$ 가 추가로 반환된다 ($S$ 의 노드가 명시적으로 반환된다.)

를 반환하는 알고리즘이다. Extended Unit Flow Algorithm은 존재한다.

**Theorem 1.3의 해설**. 정수 $h, F$ 는 지금 주어지는 입력이 얼마나 *좋은 입력인지* 를 나타내는 파라미터라고 생각할 수 있다. $F$ 가 작다는 것은 source function의 크기가 적당히 작다는 것이며, 그래프의 전도율 $\phi(G) \geq \frac{1}{h}$ 라는 것을 뜻한다. 이러한 *좋은 입력* 에서는 congestion이 $\tilde{O}(hF)$ 인 flow를 실제로 찾을 수 있다. 모든 집합 $S \subseteq V$ 에 대해서 공급량의 합은 $\sum_{v \in S} \Delta(v) \le F \times vol(S)$ 이하이고, $S$ 를 나가는 간선의 개수는 $\delta(S)$ 가 된다. $\frac{\delta(S)}{vol(S)} = \phi(S) \geq \phi(G) \geq \frac{1}{h}$ 이니 대략 각 간선마다 최대 $\frac{F \times vol(S)}{\frac{vol(S)}{h}} \le hF$ 의 공급이 주어진다. 우리의 알고리즘은 정확히 이러한 공급을 찾으려 할 것이다. 만약에 이것이 실패하면, 알고리즘은 입력이 나쁘다는 *certificate* 를 반환할 것이다. 이 *certificate* 는 low-conductance cut $S$ 와  에 해당된다. 또한 이 *certificate* 의 크기는 과잉량에 비례해서 커지게 되는데, 이 비례 관계는 이후 중요한 역할을 하게 될 것이다. 비례 관계를 만드는 방법을 간단히 설명하자면, $vol(S) \geq \frac{ex_f(\cdot)}{F}$ 일 경우 대략 $F \times vol(S) \ge ex_f(\cdot)$ 만큼의 공급을 $S$ 에 몰아줄 수 있고, $S$ 의 전도율이 낮다면, 예를 들어서 $\phi(G) \le \frac{1}{2h}$ 라면, 컷 밖으로 빠져나가는 유량의 합은 $\delta(S) \times cong(f) = F \times vol(S)$ 이하가 된다. 고로 $\phi(G)$ 가 충분히 낮다면 공급만큼 흡수가 되지 않을 수밖에 없고 과잉을 만들 수 있다.

마지막으로, 이 알고리즘의 시간 복잡도가 $G$ 의 크기보다 작으며, $(\Delta(\cdot) + \overline{T}(\cdot))$ 에 *준 선형* 임을 관찰하자.

이제 Theorem 1.3을 증명한다. 아래 Lemma는 [[Henzinger, Rao, Wang 2017]](https://arxiv.org/pdf/1704.01254.pdf) 의 Theorem 3.1과 Lemma 3.1의 parameter를 적당히 조정함으로써 얻을 수 있다. 이 논문의 [**[Appendix B.1]**](https://arxiv.org/pdf/1708.03962.pdf)을 참조하라.

**Lemma 1.4**. *Unit Flow* 라는 알고리즘은 Theorem 1.3과 동일한 ($G, h, F, \Delta, T$) 입력을 받으나, $\forall v. T(v) = \deg(v)$ 라는 조건을 추가적으로 가정한다. 이 알고리즘은 $O(Fh \Delta(\cdot) \log m)$ 시간에 

* $cong(f) \le 2hF$ 인 *source-feasible preflow* $f$ 
* $ex_f(\cdot)$
* 만약 $ex_f(\cdot) \neq 0$ 일 경우, 전도율 $\phi(S) < \frac{1}{h}$ 이며, 모든 $v \in S$에 대해 $ex_f(v) \le (F - 1) \deg(v)$ 이고, 모든 $v \notin S$ 에 대해 $ex_f(v) = 0$ 인 집합 $S$ 가 반환된다.

를 반환하는 알고리즘이다. Unit Flow Algorithm은 존재한다. 슬프게도 Unit Flow 알고리즘에 대한 설명은 생략하지만, 이 알고리즘은 push-relabel framework에 기반하여 있기 때문에 다음과 같은 사실이 성립한다.

**Fact 1.5** Lemma 1.4에서 반환하는 preflow $f$ 는 모든 $v \in V$ 에 대해, $f(v) < T(v) \implies \sum_{u \in V} f(v, u) \le 0$ 이다.

**Proof of Theorem 1.3**. 입력으로 ($G, h, F, \Delta, T$) 가 주어질 때, $\Delta^\prime$ 과 $T^\prime$ 을 새로운 source와 sink function으로 정의하자. 이 때 $\Delta^\prime(v) = \Delta(v) + \overline{T}(v)$, $T^\prime(v) = T(v) + \overline{T}(v) = \deg(v)$ 로 정의한다. 이 때 $\Delta^\prime(\cdot) = \Delta(\cdot) + \overline{T}(\cdot) \le T(\cdot) + \overline{T}(\cdot) = 2m$ 이며 $\forall v.\Delta^\prime(v) \le (F + 1) \deg(v)$ 이다. Remark 1.1에서 사용한 입출력 형식에 의해 이러한 표현은 $O(\Delta(\cdot) + \overline{T}(\cdot))$ 시간에 구성 가능하다. 이제 Lemma 1.4의 Unit Flow 알고리즘을 $(G, h, F + 1, \Delta^\prime, T^\prime)$ 이라는 인자로 실행하자. 이 때 이 알고리즘은 $O((F + 1) h \Delta^{\prime}(\cdot) \log m) = O(Fh (\Delta(\cdot) + \overline{T}(\cdot)) \log m)$ 에 작동하여 출력을 반환한다. 

이제 우리는 Unit Flow가 찾은 preflow $f$ 가 Extended Unit Flow가 찾을 source-feasible preflow라는 것을 증명한다. 

모든 $v \in V$ 에 대해서 정의상 다음 조건이 성립한다.

* $f^\prime(v) = \Delta^\prime(v) + \sum_{u} f(u, v)$
* $f(v) = \Delta(v) + \sum_u f(u, v)$
* $ex_f^\prime(v) = max(f^\prime(v) - T^\prime(v), 0)$
* $ex_f(v) = max(f(v) - T(v), 0)$
* $\Delta^\prime(v) = \Delta(v) + \overline{T}(v)$
* $T^\prime(v) = T(v) + \overline{T}(v) = \deg(v)$

$cong(f) \le 2hF$ 임은 자명하니, 우리는 $\forall v. f^\prime(v) \geq 0$ 을 만족할 때 우리는 $\forall v. f(v) \geq 0$ 가 성립함을 보인다. 다른 말로,  $\sum_u f(v, u) \leq \Delta(v)$ 임을 보인다.

*Claim 1.3.1.* $\sum_u f(v, u) \le max(0, \Delta^\prime(v) - T^\prime(v))$ for all $v \in V$

*Proof 1.3.1.* 두 케이스가 있다. $f^\prime(v) < T^\prime(v)$ 이면 Fact 1.5에 의해 자명하다. $f^\prime(v) \geq T^\prime(v)$ 이면 $\sum_u f(v, u) \le \Delta^\prime(v) - T^\prime(v)$ 이다. $\square$

*Claim 1.3.2.* $max(0, \Delta^\prime(v) - T^\prime(v)) \le \Delta(v)$ for all $v \in V$.

*Proof 1.3.2.* 정의상 $0 \le \Delta(v), T(v)$ 이다. 또한, $\Delta^\prime(v) - T^\prime(v) = \Delta(v) - T(v)$ 이다. $\square$

위 두 Claim을 연립하면, $f$ 는 $cong(f) \le 2hF$ 인 source-feasible preflow임을 보일 수 있고, Theorem의 첫 꼭지가 증명된다. 다음 Claim은 $ex_f(v)$ 가 보존됨을 증명한다.

*Claim 1.3.3.* $ex^\prime_f(v) = ex_f(v)$ for all $v \in v$.

*Proof 1.3.3* $f^\prime(v) - f(v) = \Delta^\prime(v) - \Delta(v) = \overline{T}(v)$. $T^\prime(v) - T(v) = \overline{T}(v)$.

이렇게 두 번째 꼭지가 증명된다. 마지막으로, $ex_f(*) \le (F - 1) vol(S) \le F vol(S)$ 이기 때문에, 세 번째 꼭지도 증명된다. $\blacksquare$

## Chapter 2. Locally Balanced Sparse Cut

Locally Balanced Sparse Cut은 이후 Chapter 3에서 유용하게 쓰이게 될 도구이다. Locally Balanced Sparse Cut은 대략 Degree의 합이 전체의 절반 정도 되고, 대충 sparse하며, 주어진 집합에 붙어있는 정도가 큰 컷을 뜻한다. 이것이 왜 중요한지는 지금은 쓰는 나도 잘 모르겠으니 글을 더 읽어보자. Chapter 1에서는 특수한 형태의 효율적으로 해결되는 Maximum Flow를 살펴보았는데, Minimum Cut을 일반적으로 Maximum Flow로 구하듯이, 이 장에서도 LBS Cut을 구하기 위해 Chapter 1의 알고리즘을 사용할 것이다.

**Definition 2.1 (Overlapping).** 그래프 $G = (V, E)$, 집합 $A \subset V$, 실수 $0 \le \sigma \le 1$ 에 대해서, 집합 $S \subset V$ 가 $(A, \sigma)$-overlapping한다는 것은 $vol(S \cap A) / vol(S) \geq \sigma$ 임을 뜻한다.

**Definition 2.2 ($\alpha$-sparse)**. 컷 $S$가 $\alpha$-sparse하다는 것은 전도율 $\phi(S) = \frac{\delta(S)}{min(vol(S), vol(V - S))} < \alpha$ 임을 뜻한다. 

**Definition 2.3.** 집합 $A \subseteq V$, overlapping parameter $\sigma$, conductance parameter $\alpha$ 가 주어질 때, $S^*$ 을 

* $\alpha$-sparse하고
* $(A, \sigma)$-overlapping하며
* $vol(S^*) \le vol(V - S^*)$ 를 만족하는 $S^*$ 중 $vol(S^\prime)$ 을 최대화하는

집합이라고 하자. $OPT(G, \alpha, A, \sigma) = vol(S^*)$ 로 정의한다. 만약 그러한 $S^*$ 가 없다면 $OPT(G, \alpha, A, \sigma) = 0$ 으로 정의한다. $\alpha$ 가 증가하면 취할 수 있는 $S$ 의 경우의 수가 늘어나므로, $\alpha_1 \leq \alpha_2 \implies OPT(G, \alpha_1, A, \sigma) \le OPT(G, \alpha_2,  A, \sigma) $ 임을 관찰하자. 

**Definition 2.4 (Locally Balanced Sparse Cut (LBS Cut)).** 그래프 $G = (V, E)$, 집합 $A \subset V$, 파라미터 $c_{size} \geq 1, c_{con} \geq 1, \alpha, \sigma$ 가 주어진다.

$vol(S) \le vol(V - S)$ 를 만족하는 컷 $S$가 $(c_{size}, c_{con})$-approximate locally balanced sparse cut with respect to $(G, \alpha, A, \sigma)$, 줄여서 $(c_{size}, c_{con}, G, \alpha, A, \sigma)$-LBS cut 이라는 것은,  

* $\phi(S) < \alpha$
* $c_{size} \times vol(S) \geq OPT(G, \frac{\alpha}{c_{con}}, A, \sigma)$

를 만족함을 뜻한다.

당황스러운 정의이지만, $(c_{size}, c_{con}, G, \alpha, A, \sigma)$-LBS cut 은 $OPT(G, \alpha, A, \sigma)$ 를 relax시킨 버전이라고 생각할 수 있다. 맨 위 정의한 $OPT$ 함수는 volume을 최대화하는 최적해만을 요구하지만, 여기서는 volume이 $c_{size}$-factor로 approximate되는 것을 허용하며, 그 최적해도 $\alpha$ 가 아닌 $\frac{\alpha}{c_{con}}$ 을 기준으로 계산된다. $\alpha$ 가 줄어들면 최적해 역시 줄어듬을 기억하자. 추가로 $OPT$ 와는 다르게 $(A, \sigma)$-overlapping에 대한 조건이 빠져있다는 것을 기억하자. cut에 대한 조건은 위에 나와 있는 것이 전부로 $S$ 는 $A$ 와 교집합이 아예 없어도 아무 상관이 없다.

**Definition 2.5 (LBS Cut Algorithm)**. 어떠한 parameter $c_{size} \geq 1, c_{con} \geq 1$ 에 대해서, **$(c_{size}, c_{con})$-approximate LBS cut algorithm** 은 입력으로 $(G, \alpha, A, \sigma)$ 를 받아서

* Case 1) $(c_{size}, c_{con}, G, \alpha, A, \sigma)$-LBS cut $S$ 을 반환하거나
* Case 2) $OPT(G, \alpha / c_{con}, A, \sigma) = 0$ 이라고 반환.

하는 알고리즘이다. 

각 케이스마다 위 알고리즘이 반환하는 출력을 분석하면

* 만약 $OPT(G, \alpha / c_{con}, A, \sigma) > 0$ 이면 Case 1으로 간다. 이 경우 그 집합이 원하는 LBS cut이 되기 때문에 항상 답이 존재한다.
* 만약 $OPT(G, \alpha / c_{con}, A, \sigma) = 0$ 이면 volume 조건이 사라지니, $vol(S) \le vol(V - S)$ 인 $\alpha$-sparse cut의 존재 여부만 중요하다. 이것이 존재하면 Case 1/2 중 아무거나 해도 되고, 존재하지 않으면 Case 2만 가능하다. 참고로 $\phi(S) = \phi(V - S)$ 라서 volume 조건이 큰 의미가 있지는 않다.

이제 이 장의 Main Theorem을 소개한다.

**Theorem 2.6 (Main Theorem)** 입력 인자 $(G, \alpha, A, \sigma)$ 가 다음 두 조건을 추가로 만족하는 LBS cut의 특수 경우에 대해서:

* $2vol(A) \le vol(V - A)$
* $\sigma \in [\frac{2vol(A)}{vol(V - A)}, 1]$

$(O(1/\sigma^2), O(1/\sigma^2))$-approximate LBS cut 알고리즘이 존재한다. 이 알고리즘의 시간 복잡도는 $\tilde{O}(\frac{vol(A)}{\alpha \sigma^2})$ 이다.

**Proof of the Theorem.** $(G, A, \sigma, \alpha)$ 가 입력일 때 아래와 같이 정의되는 입력 $(G, h, F, \Delta, T)$ 로 Extended Unit Flow algorithm을 돌리자 (Theorem 1.3). 

* $F = \lceil 1/\sigma \rceil, h = \lceil 1/\alpha \rceil$
* $\Delta(v) = F \deg(v)$ if $v \in A$, $0$ if $v \notin A$
* $T(v) = 0$ if $v \in A$, $\deg(v)$ if $v \notin A$ ($ \Delta(\cdot) \le T(\cdot)$)

이 때, $\Delta(\cdot) = \lceil \frac{1}{\sigma} \rceil vol(A) \le \lceil \frac{vol(V - A)}{2vol(A)} \rceil vol(A) \le vol(V - A) = T(\cdot)$ 이다. 고로 $ \Delta(\cdot) \le T(\cdot)$  을 만족한다. 다른 조건들이 만족되는 것은 쉽게 증명된다. 

Extended Unit Flow 알고리즘은 $O(hF (\Delta(\cdot) + \overline{T}(\cdot)) \log m)$ 시간 복잡도에 작동하며, $\overline{T}(\cdot) = \sum_{v \in A} deg(v) = vol(A)$ 이니, 이를 계산하면 $\tilde{O}(\frac{vol(A)}{\alpha \sigma^2})$ 가 된다. 

$cong(f) \le 2hF = O(\frac{1}{\alpha \sigma})$ 이다. $c_{size} = 2F/\sigma = O(1/\sigma^2)$, $c_{con} = (2\alpha \times cong(f)) / \sigma = O(1/\sigma^2)$ 로 정의하자. 다음과 같이 출력한다.

* $ex_f(\cdot) = 0$ 이면, $OPT(G, \alpha / c_{con}, A, \sigma) = 0$ 이라고 보고한다.
* 아닐 경우, Extended Unit Flow에서 반환한 집합 $S$ 를 반환한다.

이제 이를 증명하기 위해 다음 Lemma를 증명한다.

**Lemma 2.7.** $vol(S^\prime) \leq vol(V - S^\prime)$ 을 만족하는 임의의 $(A, \sigma)$-overlapping cut $S^\prime$ 이 $vol(S^\prime) > 2\frac{ex_f(\cdot)}{\sigma}$ 이면, $\phi(S^\prime) \geq \frac{\sigma}{2 cong(f)}$ 를 만족한다.

**Proof.** $\Delta(S^\prime) = \sum_{v \in S^\prime} \Delta(v)$, $T(S^\prime) = \sum_{v \in S^\prime} T(v)$, $ex_f(S^\prime) = \sum_{v \in S^\prime} ex_f(v)$ ... 라고 하자. 

이 때 $\Delta(S^\prime) = \lceil 1/\sigma \rceil vol(A \cap S^\prime)$, $ab_f(S^\prime) \le \sum_{v \in S^\prime} T(v) = vol(S^\prime - A)$ 임을 관찰하자. 

고로

$cong(f) \times \delta(S^\prime) \geq \Delta(S^\prime) - T(S^\prime) = \Delta(S^\prime) - ex_f(S^\prime) - ab_f(S^\prime)$

$\geq \frac{vol(A \cap S^\prime)}{\sigma} - ex_f(\cdot) - vol(S^\prime - A)$

이는

$cong(f) \times \phi(S^\prime) \geq \frac{1}{vol(S^\prime)} \times(\frac{vol(A \cap S^\prime)}{\sigma} - ex_f(\cdot) - vol(S^\prime - A))$

$=\frac{1}{vol(S^\prime)} \times(\frac{vol(A \cap S^\prime)}{\sigma} - ex_f(\cdot) - vol(S^\prime) + vol(S^\prime \cap A))$

$=\frac{1}{vol(S^\prime)} \times((1 + \frac{1}{\sigma})vol(A \cap S^\prime) - ex_f(\cdot) - vol(S^\prime) )$

$\geq \frac{1}{vol(S^\prime)} \times((1 + \frac{1}{\sigma})vol(A \cap S^\prime) - \frac{\sigma}{2} vol(S^\prime) - vol(S^\prime) )$

$\geq \frac{1}{vol(S^\prime)} \times((1 + \frac{1}{\sigma})\sigma vol(S^\prime) - \frac{\sigma}{2} vol(S^\prime) - vol(S^\prime) )$

$= (1 + \frac{1}{\sigma})\sigma - \frac{\sigma}{2} - 1 = \frac{\sigma}{2}$. $\blacksquare$

**Proof of The Theorem (cont).** 이제 $OPT(G, \frac{\sigma}{2cong(f)}, A, \sigma) \le 2\frac{ex_f(\cdot)}{\sigma}$ 임을 보일 수 있다. 이를 초과하는 집합이 정확히 Lemma 2.7의 조건에 의해서 걸러지기 때문이다. 고로 $ex_f(\cdot) = 0$ 일 경우 알고리즘의 정당성은 자명하다. 아닐 경우, Extended Unit Flow가 반환한 집합이 $S$ 라고 하면, Theorem 1.3에 의해 $\phi_G(S) < \frac{1}{h} \le \alpha$, $vol(S) \geq \frac{ex_f(\cdot)}{F}$ 를 만족한다. $c_{size} = \frac{2F}{\sigma}$ 이니, $vol(S) \times c_{size} \geq \frac{2 ex_f(\cdot)}{\sigma}$ 이다. Lemma 2.7과 조합하면 증명이 종료된다. $\blacksquare$

**Remark.** 위 증명에는 $vol(S) \le vol(V - S)$ 조건을 빠트렸다는 허점이 있다. 난 자명한 논리로 고치지 못하겠는데 별 것이 아니라면 알려주길 바란다.

## Chapter 3. Expander Pruning

Dynamic Expander Pruning Algorithm은 기존 연구 (Wulff-Nilsen. 2017) 에서 Dynamic MSF Algorithm을 얻는데 사용된 핵심 도구였다. Wulff-Nilsen이 사용한 Dynamic Expander Pruning Algorithm은 $O(n^{0.5 - \epsilon_0})$ 의 업데이트 시간을 가진 랜덤 알고리즘이나, 이 논문에서는 $O(n^{o(1)})$ 의 업데이트 시간을 가진 결정론적 (deterministic) 알고리즘이다. 하지만 이 논문의 다른 부분에서 랜덤화를 사용하기 때문에 최종 MSF 알고리즘은 랜덤 알고리즘이다.

**Theorem 3.1 (Dynamic Expander Pruning)**. $\epsilon(n) = o(1)$ 을 만족하는 임의의 함수 $\epsilon$ 에 대해, $\alpha_0(n) = 1/n^{\epsilon(n)}$ 이라 하자. $T = O(m \alpha_0^2(n))$ 개의 간선 삭제가 진행되는 그래프 $G$ 에 대해서, 다음과 같은 정점 집합 $P \subseteq V$ 를 관리하는 Dynamic algorithm $A$ 가 존재한다.

$G_{\tau}, P_{\tau}$ 를 $\tau$ 번째 삭제 이후의 그래프 $G$ 와 집합 $P$ 라고 하자.

* 맨 처음, $A$ 는 $P_0 = \emptyset$ 을 설정하고 $n$ 개의 정점과 $m$ 개의 간선을 가졌으며 *최대 차수가 3인* 그래프 $G_0 = (V, E)$ 을 입력받는다. 시간 복잡도는 $O(1)$ 이다 (그래프는 메모리에 저장되어 있음을 기억하자).
* $\tau$번째 삭제 이후, $A$ 는 $n^{O(\log \log \frac{1}{\epsilon(n)}/ \log \frac{1}{\epsilon(n)})} = n^{o(1)}$ 시간에, $P_{\tau - 1}$ 에 추가할 집합 $S$ 를 반환한다. 이후, $P_{\tau} = P_{\tau - 1} \cup S$ 로 정의된다.  

이때, 각 단계 $\tau$ 에서, $\phi(G_0) \geq \alpha_0(n)$ 이면, $G_{\tau}[V - W_{\tau}]$ 가 연결 그래프인 $W_{\tau} \subseteq P_{\tau}$ 가 존재한다.

Theorem 3.1에서 마지막 조건의 의미를 풀어 설명하면, 집합 $V \setminus P_{\tau}$ 이 $G_{\tau}$ 에서 하나의 연결 컴포넌트 안에 들어있다는 뜻이다. 우리는 이 집합 $P$ 를 *pruning set* 이라고 부른다. 이후 응용에서 이 개념은 $P$ 에 인접한 간선들은 그래프에서 관리하지 않는 식으로 사용될 것이다.

관찰하면 좋을 것이 두 가지 있는데, 첫 번째는 맨 처음에 $P_1 = V$ 로 설정해 놓으면 문제가 자명하게 해결된다는 것이다. 이것이 불가능한 이유는, 시간 복잡도 조건에 따라 $P_i - P_{i - 1} \le n^{o(1)}$ 을 만족해야 하기 때문이다. 두 번째는, 만약에 두 큰 연결 컴포넌트가 절선 하나로 묶여져 있고, 첫번째 간선 제거에서 그 절선이 제거된다면, $P_1 = V / 2$ 정도의 매우 큰 크기를 강요할 수 있다는 것이다. $\phi(G_0) \geq \alpha_0(n)$ 조건이 존재하는 것은 이를 위해서이다. 위와 같은 경우는 절선을 양옆으로 하는 컷의 conductance가 매우 낮기 때문에 그래프 전체의 conductance가 작다고 볼 수 있고, 위 조건에 위배된다.

이 문제를 해결하기 위해 우리는 Expander Pruning을 조금 변형한 문제인 One-shot expander pruning을 해결할 것이고, 이 알고리즘을 토대로 전체 문제를 해결할 것이다.

#### Chapter 3.1. One-shot Expander Pruning

One-shot expander pruning이 Dynamic expander pruning과 다른 점은 크게 두가지다. 첫 번째로, One-shot expander pruning은 간선을 하나 하나 삭제하는 것이 아니라, 삭제할 간선의 집합을 받아서 한 번에 삭제한 후 지울 집합 $P$ 하나만을 출력한다. 두 번째로, $P$ 를 고를 때 단순히 연결성을 토대로 고르는 것이 아니라, 특정한 conductance 이하인 *모든* 노드들을 고른다. 고로 $G[V - P]$ 는 단순히 연결되어 있는 것이 아니라 높은 conductance를 가진다. 아래에 해당 알고리즘의 정의를 엄밀하게 서술한다.

**Theorem 3.2 (One-shot Expander Pruning)**. 다음 작업을 하는 알고리즘이 존재한다.

* 입력으로는 $(G, D, \alpha_b, \delta)$ 가 주어진다.
  * $G = (V, E)$ 는 최대 차수가 $\Delta$ 인 그래프로 $V = n, E = m$ 이다. $G$ 는 **지워진 이후의 그래프** 를 나타낸다.
  * $\alpha_b$ 는 conductance parameter이다.
  * $\delta \in (0, 1)$ 역시 실수 parameter이다.
  * $D$ 는 원래 그래프에서 지울 간선들을 나타내는 것으로, $D \cap E = \emptyset$ 이다. $D = O(\alpha^2_bm / \Delta)$를 만족한다. 그래프 $G_b = (V, E \cup D)$ 는 지우기 전의 그래프를 나타낸다.
* 이후, 시간 $\overline{t} = \tilde{O}{\frac{\DeltaD^{1 + \delta}}{\delta \alpha_b^{6 + \delta}}}$ 에 $A$ 는 $\phi(G_b) < \alpha_b$ 임을 반환하거나, Pruning set $P \subset V$ 를 출력한다. 또한, $\phi(G_b) \geq \alpha_b$ 이면 
  * $vol_G(P) \le \frac{2D}{\alpha_b}$
  * *pruned graph* $H = G[V - P]$는 높은 전도율을 가지고 있다: $\phi(H) \geq \alpha = \Omega(\alpha_b^{2/\delta})$ 

이 때 $\overline{t}$ 를 *time limit*, $\alpha$ 를 $A$ 의 *conductance guarantee* 라고 한다. 이 Theorem의 증명을 위해서는 다음 Lemma가 필요하다.

**Lemma 5.3.** 시간 복잡도 $t_{LSB}(n, vol(A), \alpha, \sigma)$ 에 작동하는 $(c_{size}(\sigma), c_{con}(\sigma))$-approximate LBS cut 알고리즘이 존재한다면 ($(G, A, \sigma, \alpha)$ 는 입력 형식과 동일하다). 입력 $(G, D, \alpha_b, \delta)$ 에 대해 one-shot expander pruning algorithm이 존재하여, 

* time limit $\overline{t} = O((\frac{D}{\alpha_b})^\delta \times \frac{c_{size}(\alpha_b/2)}{\delta} \times t_{LSB}(n, \frac{\DeltaD}{\alpha_b}, \alpha_b, \alpha_b))$
* conductance guarantee $\alpha = \frac{\alpha_b}{5c_{con} (\alpha_b / 2)^{1/\delta - 1}}$

을 만족한다.

애석하게도. 이 Lemma의 증명은 여기 담기는 너무 길다. 이 논문의 **[[Appendix A]](https://arxiv.org/pdf/1708.03962.pdf)** 를 참고하라. 그냥 Lemma가 참이라고 하면, $t_{LSB}(n, vol(A), \alpha, \sigma) = \tilde{O}(\frac{vol(A)}{\alpha \sigma^2}), c_{size}(\sigma) = O(\frac{1}{\sigma^2}), c_{con}(\sigma) = O(\frac{1}{\sigma^2})$ 를 집어 넣으면 위의 결과가 나옴을 확인할 수 있다. $\blacksquare$

#### Chapter 3.2. Dynamic Expander Pruning

One-shot expander pruning algorithm에서는 우리 손으로 알아낸 것이 하나도 없으나, Dynamic expander pruning에서는 One-shot expander pruning algorithm을 사용하여 새로운 결과를 유도한다. 먼저 Theorem 3.1을 다음 형태로 다시 쓴다.

**Lemma 3.4 (Dynamic Expander Pruning again)**. $T = O(m \alpha_0^2(n)/\Delta)$ 개의 간선 삭제가 진행되는 그래프 $G$ 에 대해서, 다음과 같은 정점 집합 $P \subseteq V$ 를 관리하는 Dynamic algorithm $A$ 가 존재한다.

$G_{\tau}, P_{\tau}$ 를 $\tau$ 번째 삭제 이후의 그래프 $G$ 와 집합 $P$ 라고 하자.

* 맨 처음, $A$ 는 $P_0 = \emptyset$ 을 설정하고 $n$ 개의 정점과 $m$ 개의 간선을 가졌으며 *최대 차수가 $\Delta$ 인* 그래프 $G_0 = (V, E)$ 을 입력받는다. 그리고, 파라미터 $\alpha_0, l$ 을 입력받는다. 시간 복잡도는 $O(1)$ 이다 (그래프는 메모리에 저장되어 있음을 기억하자).
* $\tau$번째 삭제 이후, $A$ 는 $\tilde{O}(l^2\Delta n^{O(1/l+\epsilon l^l)})$ 시간에,
  * $\phi(G_0) \geq \alpha_0(n)$ 이면 $P_{\tau - 1}$을 갱신하여 $P_{\tau}$ 로 만든다. 이 때 $P_{\tau - 1} \subseteq P_{\tau} \subseteq V$ 이다.
  * 그렇지 않으면 종료한다.

이때, 각 단계 $\tau$ 에서, $\phi(G_0) \geq \alpha_0(n)$ 이면, $G_{\tau}[V - W_{\tau}]$ 가 연결 그래프인 $W_{\tau} \subseteq P_{\tau}$ 가 존재한다.

Lemma 3.4이 참이라고 가정하면 우리는 Theorem 3.1을 유도할 수 있다.

**Proof of Theorem 3.1.** $\Delta = 3, l = \frac{\log \frac{1}{\epsilon}}{2\log \log \frac{1}{\epsilon}}, \alpha = 1/n^{\epsilon}$으로 두자. 이 때 $O(l \log l) = 1/2 \log \frac{1}{\epsilon}$ 이니 $l^l = O(\frac{1}{\epsilon^{1/2}})$ 이다. $O(1/l + \epsilon l^l) = O(\epsilon^{1/2} + \log \log \frac{1}{\epsilon} / \log \frac{1}{\epsilon})  = o(1)$ 이다. 고로 각 쿼리는 $n^{o(1)}$ 에 동작한다. 이제 Lemma 3.4의 statement과 Theorem 3.1과 대응됨을 확인할 수 있다.

이제 이 단락의 과제는 Lemma 3.4을 증명하는 것이다. 이 과정을 다룬 이후에서야 우리는 본격적으로 Main Algorithm에 대해서 이야기할 수 있다. 이에 대해서는 Part 2에 이어서 설명한다.

