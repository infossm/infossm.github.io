---
layout: post
title: "양자컴퓨팅으로 PS하는법"
date: 2023-12-25
author: red1108
tags: [quantum, quantum-computing, ps, problem-ssolving]
---

> 제목을 조금 자극적이게 "양자컴퓨팅으로 PS하는 법"이라고 설명했지만, 아직 일반적인 PS를 하는 데 양자컴퓨터를 사용하는것은 어렵다. 그래도 그나마 PS스러운 max cut 문제를 양자컴퓨터로 해결하는 방법을 최대한 쉽게 소개하고자 한다. 나중에도 양자컴퓨터를 사용한 소개할만한 알고리즘이 있다면 소개해볼 계획이다.

# Max cut 문제란?

Max cut 문제는 주어진 그래프를 두 집합으로 잘 분리하여 두 집합 사이를 연결하는 간선의 수를 최대화하는 문제이다.

<p align="center"><img src="/assets/images/red1108/maxcut-introduce.png"></p>
<center><b>그림 1. Maxcut 예시.</b>흰색과 검은색 정점 집합으로 위와 같이 분리하면 둘 사이를 연결하는 간선(빨간색)이 5개로 최대가 된다. 따라서 이 그래프에서 maxcut 은 5이다.</center>

이 문제를 다항 시간에 해결할 수 있을 지 없을 지 잠시 고민해 보자. 아마 방법이 떠오르지 않을 것이다.

그리고 만약 누군가가 주어진 복잡한 그래프의 max cut이 10204531이라고 주장했다고 해 보자. 그래도 친절하게 어떻게 정점 집합을 분리해야 저 답이 나오는지는 제공했다고 치자. 그럼 저 그래프의 max cut이 실제로 10204531인지를 다항 시간에 검증할 수 있을까? 정점 집합을 제공받았으니 직접 카운팅 해서 해당 경우에 cut이 10204531이라는 건 확인할 수 있을 것이다. 그러나 저거보다 더 큰 값이 불가능하다는 것을 검증하기는 굉장히 힘들 것만 같다...

위 두 가지 문제를 통해 직감적으로 느꼈듯, 일반적인 그래프에 대해 max cut을 다항 시간에 구할 수도 없고, 다항 시간에 답을 검증할 수도 없다. 따라서 이 문제는 NP-hard 문제이다. 증명은 이 글의 목적에는 맞지 않으므로 넘어가자.

## Max cut의 근사 알고리즘

정확한 답은 절대로 다항 시간에 구할 수가 없기 때문에, 주된 연구 방향은 근사 알고리즘이다. 예를 들어, 주어진 그래프의 max cut이 M이고 어떤 알고리즘으로 구한 답이 A일 때, $A/M > 0.7$ 을 보장할 수 있다면 이 알고리즘은 0.7-approximation algorithm이라고 한다. 0.879-approximation 알고리즘은 다항 시간에 구할 수 있다는 것이 알려져 있다[1]. 하지만 이번 글의 목적은 양자 알고리즘을 소개하는 것이다. 하지만 **양자 알고리즘으로도 max cut문제를 다항 시간에 해결할 수 없다.** 양자 알고리즘으로도 아직까지는 approximation 알고리즘이 한계이며, 이 알고리즘을 소개해보고자 한다.

# Max cut을 양자 알고리즘으로 해결하기

양자 알고리즘의 접근법은 다음과 같다: (TODO: 내용 추가하자.)

## 1. 양자 상태로 표현하기

Max cut 문제의 본질은 그래프의 정점을 두 부분집합으로 분할하는 것이다. 편의상 두 집합의 이름을 1, 0으로 이름붙이자.

<p align="center"><img src="/assets/images/red1108/maxcut_graph.png" width="300px"></p>
<center><b>그림 2. 앞으로 예시로 들 그래프.</b></center>

위 그래프는 정점을 4개 가지고 있다. 그리고 누가 보아도 maxcut은 4이고, 이때의 분할은 정점 순서대롲 집합의 이름을 붙였을 때 1010, 0101이 가능한 예시이다.




## 참고문헌

[1] Goemans, Michel X., and David P. Williamson. ". 879-approximation algorithms for max cut and max 2sat." Proceedings of the twenty-sixth annual ACM symposium on Theory of computing. 1994.