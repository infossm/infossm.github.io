---
layout: post
title: "The Short-Side Advantage in Random Matching Markets"
author: leejseo
date: 2022-10-01 09:00
tags: [combinatorics, algorithm, random]
---

이 글은 L. Cai와 C. Thomas의 논문 [The Short-Side Advantage in Random Matching Markets](https://arxiv.org/pdf/1910.04406.pdf) 의 결과를 간략하게 정리한 것이다.

## 1. Introduction

Stable Matching Problem은 남-여 간의 짝 매칭, 의사와 병원간의 매칭, 학생과 지도교수 간의 매칭 등 여러 상황에서 응용될 수 있는 문제로 다음과 같은 상황을 다룬다.

> $n$ 명의 의사 $\mathcal{D} = \{d_1, d_2, \cdots, d_n\}$ 와 $m$ 개의 병원 $\mathcal{H} = \{h_1, h_2, \cdots, h_m\}$ 이 있다. 각각의 의사는 병원에 대한 선호하는 순서($\prec_d$)가 존재하고, 각각의 병원은 의사에 대한 선호하는 순서($\prec_h$)(preference list)가 존재할 때, 한 명의 의사와 하나의 병원을 다음 조건을 만족하며 짝 짓는 방법(matching)을 구하여라.
>
> - 조건: 짝 지어지지 않은 의사와 병원의 쌍 $(d, h)$ 에 대해 $d$ 는 현재 짝 지어진 병원보다 $h$ 를 선호하고, $h$ 는 현재 짝 지어진 의사보다 $d$ 를 선호하는 상황이 존재하지 않는다.

보다 일반적으로는, 의사가 전체 병원 가운데 일부에 대한 preference list만 갖고 있고, 병원도 전체 의사 가운데 일부에 대한 preference list만 갖고 있고, list 외의 병원/의사에 대해서는 아예 매칭될 의사를 가지지 않는 상황 또한 다룰 수 있지만, 설명의 편의를 위해 이 글에서는 위 문단에서 정의한 상황을 다룬다.

Stable Matching Problem에 대해서는 [2]에서 제시된 의사와 병원 중 한 쪽이 반복적으로 proposing 하면서 진행되는 propose-reject algorithm이 널리 알려져 있다. 후술할 알고리즘을 $DPDA$(doctor-proposing deferred acceptance)라 부르도록 하겠다. (비슷하게 $HPDA$ 또한 정의할 수 있다.) $DPDA$ 로 찾아지는 매칭의 경우 doctor-optimal 함이 널리 알려져 있다.

```
U := D // set of unmatched doctors
while U.size() > 0 and some d in U has "unproposed" hospital
    pick such d
    h := d's favorite one among "unproposed" hospital
    propose(d, h)
    if h prefers d to its matched doctor(=d')
        unmatch(d', h)
        add(d', U)
        match(d, h)
        remove(d, U)
```

이 글에서는 모든 참여자가 uniformly random한 preference list를 갖고 있는 상황에서 평균적으로 "몇 번째로 좋아하는 사람과 매칭되는지"를 분석한 흥미로운 결과를 소개한다. $\newcommand{\rank}{\mathrm{rank}}\newcommand{\E}{\mathbb{E}}$

## 2. Balanced Market

**Definition 2.1.** 고정된 의사 $d$ 에 대해 매칭 $\mu$ 의 $\rank_d(\mu) = 1 + \lvert\{h \in \mathcal{H} : h \prec_d \mu(d)\}\rvert$ 로 정의한다. 단, $\mu(d)$ 에 해당하는 병원이 존재하지 않을 경우, $\rank_d(\mu) = 1 + \lvert\mathcal{H}\rvert$ 로 정의된다. $\rank_h(\mu)$ 또한 비슷하게 정의된다.

즉, $\rank_d(\mu)$의 경우 $d$ 가 $\mu$ 에서 몇 번째로 좋아하는 사람과 매칭되었는지 나타내는 지표로 볼 수 있다.

이제, uniformly random하고 independent 한 preference list를 갖고 있는 $n$ 명의 의사와 $n$ 개의 병원으로 이루어진 balanced market을 생각해보자. 여기에서 $DPDA$ 를 통해 찾아진 doctor-optimal한 매칭을 분석해보도록 하겠다.

그에 앞서, 한 가지 유용한 보조정리를 살펴보고 넘어가자.

**Lemma 2.2.** (Coupon Collector's Problem) 상자 하나 마다 $n$ 종류의 쿠폰 중 하나가 동일한 확률로 들어있을 때, 모든 쿠폰을 수집하기 위해 열어봐야 하는 상자의 개수의 기댓값은 $\Theta(n \log n)$ 이다.

_Proof Sketch._

- $i-1$ 번째로 모은 종류의 쿠폰이 수집된 시점부터 $i$ 번째로 모은 종류의 쿠폰이 수집된 시점까지 열어본 상자의 수를 random variable $t_i$ 로 나타내면, 우리가 구하고자 하는 값은 $T = t_1 + t_2 + \cdots + t_n$ 에 대해 $\E[T]$ 이다.

- 한 개의 상자를 열었을 때 $i$ 번째로 모으게 되는 종류의 쿠폰을 발견할 확률 $\displaystyle p_i = {n - i + 1 \over n}$.

- 기댓값의 선형성에 의해,
  $$
  \begin{align*}
  \E[T] &= \E(t_1 + t_2 + \cdots + t_n) \\
  &= \E[t_1] + \E[t_2] + \cdots + \E[t_n] \\
  &= n \cdot H_n \\
  &= \Theta(n \log n). \square
  \end{align*}
  $$

**Theorem 2.3.** Preference list가 uniformly random하고 서로 independent 하다고 가정하자. $DPDA$ 로 찾아진 매칭 $\mu$ 에 대해 다음이 성립한다:

$$
\E[\rank_d(\mu)] = O(\log n) \\
\E[\rank_h(\mu)] = \Omega \left( {n \over \log n} \right )
$$

_Proof Sketch._

- Observation 1: $DPDA$ 의 실행 과정에서 발생한 총 proposal의 수와 $\sum \rank_d(\mu)$ 는 동일하다.
- Observation 2: $n$ 개의 병원 모두 하나 이상의 proposal을 받은 순간 알고리즘이 종료한다.
- Observation 3: Preference list에 대한 조건 때문에 $DPDA$ 에서 발생하는 의사의 proposal은 전체 병원 상에서 uniform 하게 분포한다.
- Observation 4: Observation 3에 의해 $DPDA$ 에서 발생하는 총 proposal 횟수는 **proposal이 중복으로 일어나지 않는다는 사실을 제외하면** coupon collector random variable과 완전히 동등하다. 따라서, 총 proposal 횟수의 기댓값은 $O(n \log n)$ 이다.

위 observation 들에 의해 평균적으로 $O(n \log n)$ 회의 proposal이 일어나므로, $\E[\rank_d(\mu)] = O(\log n)$ 이 된다. 마찬가지로, 하나의 병원은 평균적으로 $O(\log n)$ 회의 proposal을 받는데, 병원의 rank는 propose한 의사들 가운데 가장 "선호하는" 의사에 의해 결정되므로, 평균적으로 $\Omega(n / \log n)$이 된다. $\square$

## 3. Unbalanced Market

이번에는, 비슷한 상황에서 $n+1$ 명의 의사와 $n$ 개의 병원이 있는 경우를 살펴볼 것이다. 앞서 살펴본 세팅에서 달라진 것은 한 명의 의사가 추가된 것 뿐이지만, 이 추가된 의사에 의해 시장의 구성 양상이 굉장히 달라지게 된다. 왜냐하면, 모든 의사는 아무 병원에도 매치되지 못하는 것 보다는 자신이 비선호하는 병원에 매치되는 것을 더 "원하기" 때문이다.

**Lemma 3.1.** 의사 $d^\star$ 를 고정하자. Preference list $P$ 에 대해 "$d^\star$는 자신이 $i$ 번째로 선호하는 병원까지만 매칭 의사가 있고, 이후 병원들과는 매칭되지 않으려 한다."는 조건을 추가한 preference list $P_i$ 를 정의하자. 이 때, $d^\star$ 가 rank $i$ 이하의 stable partner를 가짐과 $d^\star$ 가 $HPDA(P_i)$ 에서 매치 됨은 동치이다.

위 보조 정리를 활용하여, 다음과 같은 결과를 보일 수 있다.

**Theorem 3.2.** Preference list가 uniform하고 서로 independent 한 상황에서 $n + 1$ 명의 의사와 $n$ 개의 병원이 있는 matching market을 고려하자. 임의의 stable matching $\mu$ 가 주어졌다고 하자. 이 때, 임의의 $d \in \mathcal{D}$ 에 대해

$$
\E[\rank _d (\mu)] = \Omega(n / \log n ).
$$

_Proof Sketch._ Lemma 3.1.에 의해 의사 $d^\star$ 의 optimal outcome은 자신이 $P_i$ 에서 $HPDA$ 를 돌렸을 때 매치될 수 있는 최소의 $i$ 와 같다. 주어진 preference list $P$ 에 대해 $P'$ 를 $P$ 에서 $d^\star$ 의 preference list를 empty list로 바꾼 것으로 정의하자. (즉, $d^\star$ 는 모든 병원을 reject 한다.) $HPDA(P')$ 를 돌렸을 때 발생하는 총 proposal의 수는 Theorem 2.3. 에서와 같은 이유로 $O(n \log n)$ 이 되고, $d^\star$ 가 받게 되는 proposal 의 수의 기댓값은 $O(\log n)$ 이 된다. "$P_i$ 에서 $HPDA$ 를 돌렸을 때 $d^\star$ 가 매치될 수 있는 최소의 $i$" 의 기댓값 또한 $\Omega(n / \log n)$ 임을 계산을 통해 보일 수 있다. $\square$

## 4. Conclusion

의사-병원 매칭 문제에서 balanced market과 unbalanced market을 비교해봤을 때, 한 명의 의사가 추가되는 것 만으로, 의사에게 있어 최선의 상황에서 의사가 매치되는 병원의 rank의 기댓값이 $O(\log n)$ 에서 $\Omega(n/ \log n)$ 으로 급격히 나빠진 것을 확인해볼 수 있었다.

추가적으로, 의사가 받게 되는 불이익은 시장에 의사가 더 많아질수록 더 커진다고 한다. $\lambda > 0$ 에 대해 $(1 + \lambda) n$ 명의 의사와 $n$ 개의 병원이 있다 했을 때, 의사가 매치되는 병원의 average rank는 $\Omega(n / \log(1 + 1 / \lambda))$ 라고 한다. [3]

## Rererences

[1] L. Cai and C. Thomas. The Short-Side Advantage in Random Matching Markets. _5th Symposium on Simplicity in Algorithms_, 2022.

[2] D. Gale and L. S. Shapley. College admissions and the stability of marriage. _American Mathematical Monthly_, 69:9–14, 1962.

[3] Itai Ashlagi, Yash Kanoria, and Jacob D. Leshno. Unbalanced random matching markets: The stark effect of competition. _Journal of Political Economy_, 125(1):69 – 98, 2017.
