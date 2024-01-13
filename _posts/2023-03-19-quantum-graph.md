---
layout: post
title: "Quantum Graph"
date: 2023-03-19
author: red1108
tags: [quantum, quantum-computing, graph-theory]
---

## 서론

그래프는 수학과 컴퓨터 과학에서 굉장히 중요하게 다루는 내용이다. 알고리즘 문제풀이만 하더라도 그래프와 관련된 알고리즘이 수 없이 많다. 이러한 그래프 이론은 최단 경로, 효율적인 네트워크 구조, 데이터 분석 등에 폭넓게 응용된다. 다들 기존에 자주 보던 그래프 (고전 그래프)에 대해서는 어느정도 익숙할 것이라 생각한다.

양자 컴퓨팅 분야가 발전하면서 고전 컴퓨팅에 존재하던 다양한 개념들을 양자 컴퓨팅으로 옮겨오는 연구들도 많이 진행되었다. 이는 한 학문이 발전하면서 흔히 보이는 현상이다.

본 글에서는 Graph의 개념을 옮겨온 Quantum Graph에 대해 소개해 보고자 한다. 그리고 고전 그래프와 양자 그래프의 관련성을 다루며, 그래프의 가장 대표적인 속성인 **연결성**의 관점에서 고전 그래프와 양자 그래프가 같은 속성을 가짐을 보일 것이다. 그리고 글에 마지막에는 다들 익숙하지 않을 그래프의 응용 방법 중 하나인 *Confusability graph*에 대해 소개할 것이다.

## Quantum Graph

이제부턴 표현을 명확히 하기 위해 일반적인 채널과 그래프는 고전 채널, 고전 그래프로 표현하기로 하자.

양자 그래프는 _operator system_(adjoint에 대해 닫혀 있고, 항등행렬을 포함하는 행렬의 집합)이다. 그리고 주어진 고전 그래프 G에 대응되는 양자 그래프는 아래와 같이 정의된다.

> **Definition**
> $$S_G = \mathrm{span}\{\vert e_i \rangle \langle e_j\vert  \; \vert  \; \text{i = j or i is adjacent to j}\}$$<br/>

또한 두 quantum graph의 곱 또한 아래와 같이 정의한다.

> **Definition**
> $$UV = \mathrm{span}\{ uv\;\vert \;u\in U, v \in V \}$$<br/>

두 quantum graph의 곱이 정의되었으므로 quantum graph의 거듭제곱도 재귀적으로 정의할 수 있다. 특별히, $U^0 = \mathbb{C}I_n$으로 정의한다.

## Connected Quantum Graph

고전 그래프에서의 연결성은 쉬운 개념이다. 여기서는 양자 그래프에서의 연결성을 다룬다. 양자 그래프의 연결성을 고전 그래프에서의 의미를 가져올 수 있도록 잘 정의하는 것이 필요하다. 따라서 양자 그래프의 연결성은 다음과 같의 정의한다.

> **Definition**
> $S \in M_n$이 연결되어 있다 $\Leftrightarrow$ $S^m = M_n$인 자연수 $m$이 존재한다.

반대로, 해당 $m$이 존재하지 않는다면 $S$는 연결되어 있지 않다.

위 정의는 잘 와닿지 않을 수 있지만, 고전 그래프에서 유사한 예시를 떠올릴 수 있다. 고전 그래프에서 직접적으로 연결되어 있으면 1, 아니면 0을 가진 인접 행렬을 생각해 보자. 간선을 한번 이하로 지나 연결되어 있는 원소들은 $M$에서 확인할 수 있고, 간선을 두번 지나는 경우는 $M^2$를 확인하면 된다. 만약 $M$을 적당히 거듭제곱 하여 행렬의 모든 원소가 0이 아닌 값을 가진다면 그래프는 연결되어 있다. 양자 그래프에서의 연결성의 정의도 이와 굉장히 유사하다.

## Disconnected Quantum Graph

고전 그래프에서 "연결되지 않음"은 그래프가 서로 연결된 간선이 없는 두 집합으로 분리 가능함을 의미한다. 이 개념을 양자 그래프에도 가져올 수 있을까? 다음과 같은 정리가 성립한다.

> **Theorem**
> Quantum graph $S$가 disconnected라 함은, $PS(I_n-P)={0}$를 만족하는 비자명 사상 행렬 $P \in M_n$이 존재함과 동치이다.

위 Theorem의 증명에는 von Neumann double commutant theorem이 사용되는데, 본 글의 범위를 벗어나는 거 같아 아이디어만 요약하여 정리하였다. 우선 $\text{dim}(S^n) \leq \text{dim}(S^{n+1})$ 임을 관찰해야 한다. 이 사실은 자명한데, Quantum graph는 operator system이므로 identity matrix를 포함하고 있기 때문이다. 따라서 <$\text{dim}(S^n)$>은 위로 유계이며 증가수열이므로 어느 $m_0$이상부터는 수렴하여 $S^{n} = S^{m_0}$ for $n>=m_0$을 만족한다.

그리고 quantum graph $S$에 대응되는 commutant 집합 $S'$을 정의하면 $S'=S_G$ 이다. 따라서 von Neumann double commutant thm에 의해 S가 연결되는 것과 $S'$이 항등행렬의 상수 배인 조건이 필요충분 조건이 된다. 이제 $S'$가 항등행렬의 상수배가 아닌 것과 필요충분인 조건을 찾으면 되는데, 그 조건이 $PS(I_n-P)={0}$를 만족하는 비자명 사상 행렬 $P \in M_n$이 존재하는 것이다.

## Equivalent connectedness

> **Corollary**
> 정점 집합 {1~n}으로 만들어진 고전 그래프 G에 대응되는 양자 그래프 $S_G$에서 두 그래프의 연결성은 동치이다.

위 따름정리를 통해 Quantum graph에서의 연결성의 정의가 정당하다는 점을 확인할 수 있다. 증명 또한 직관적이다. 먼저 1) 고전 그래프가 connected라면 양자 그래프도 connected이며, 2) 고전 그래프가 disconnected라면 양자 그래프도 disconnected임을 보일 것이다.

1. 만약 G가 connected라면 모든 $i, j \in \{1...n\}$에 대해 $\{p_1=i, p_1, ...,p_{m-1}, p_m=j \}$인 경로가 존재함을 의미한다. 그러한 경로가 존재함이 보장된다면 m은 당연히 $m \leq n$을 만족하도록 잡을 수 있다. 따라서 $\vert e_i\rangle \langle e_j \vert  = \prod_{k=1}^{m-1} \vert e_{p_k}\rangle \langle e_{p_{k+1}}\vert  \in S_G^{m} \in S_G^{n}$ 이므로 $S_G^{n}$만 하여도 충분히 $M_n$을 span함을 알 수 있다. 따라서 $S_G$또한 connected이다.

2. 만약 G가 disconnected라면 사이에 간선이 존재하지 않는 두 집합 $K, L$로 정점들이 분리가 된다. 그러면 $P=\sum_{j\in K} \vert e_j \rangle \langle e_j\vert $ 로 설정하면 $PS_G(I_n-P)={0}$이므로 정의에 의해 $S_G$또한 disconnected이다.

따라서 고전 그래프 $G$의 연결성과 양자 그래프 $S_G$의 연결성은 동치이다.

# Connectivity

## Connectivity in Classical Graph

이전 section에서는 고전 그래프와 양자 그래프에서의 _connectedness_(연결성)을 정의하였고 두 대응되는 고전/양자 그래프의 연결성이 동치임을 보였다.

이번에는 그래프의 연결성과 관련된 또 하나의 척도인 _connectivity_(연결도)를 다뤄 보자. 고전 그래프에서 *connectivity*는 그래프가 얼마나 탄탄하게 연결되었는지를 나타내는 척도이다. 이 개념은 *vertex connectivity*와 *edge connectivity*로 나뉜다. vertex connectivity는 현재 그래프가 분리된 그래프가 되거나 정점이 한개만 남은 그래프가 되기 위해 지워야 하는 최소의 정점 개수이다. edge connectivity는 현재 그래프가 분리되기 위해 지워야 하는 최소한의 간선 개수를 의미한다.

본 글에서 다루는 connectivity는 vertex connectivity에 해당한다는 점을 미리 밝힌다.

## Connectivity in Quantum Graph

양자 그래프에서 *정점의 제거*를 정의하기 위해서는 양자 그래프에도 subgraph에 대한 개념이 필요하다.

> **Definition** subgraph of quantum graph
> $S \subseteq M_n$이 quantum graph일때, 임의 projection $P \subseteq M_n$에 대해 $PSP$는 $S$의 subgraph이다.

이제 우리는 양자 그래프에서 subgraph를 정의할 수 있다. 양자 그래프의 connectivity를 알기 위해서는 몇몇 정점을 제거해야 한다. 바꿔 말하면, 주어진 그래프의 subgraph를 지워서 남은 그래프가 1)<u> disconnected이거나</u> 2)<u>정점이 1개만 남거나</u> 둘 중하나를 만족해야 한다. 이를 그대로 적용해 온다면 아래의 정의가 만들어진다.

> **Definition** *seperator*
> $S \subseteq M_n$이 quantum graph일때, 비자명 projection $P \subseteq M_n$에 대해 $(I_n-P)S(I_n-P)$가 disconnected이거나 1차원 matrix이면 $P$를 $S$의 <b>seperator</b>이라 부른다.

이제 seperator을 정의하였으니 이를 이용하여 양자 그래프에서의 connectivity를 정의하자.

> **Definition** *k-connected*
> $S \subseteq M_n$이 quantum graph일때, $S$의 모든 seperators의 rank가 최소 k 이상이라면 quantum graph $S$는 <b>k-connected</b>이다.

이제 굳이 이렇게 subgraph, seperator, k-connected를 정의한 중요한 이유가 나온다. 이렇게 정의한다면 고전 그래프에서의 *vertex connectivity*와 대응되는 양자 그래프에서의 connectivity가 동일하기 때문이다. 이 내용은 아래 정리에서 다룬다.

> **Proposition**
> 정점을 {1~n}으로 가지는 고전 그래프 $G$와, 그에 대응되는 양자 그래프 $S_G$가 있을때
G is k-connected $\Leftrightarrow$ $S_G$ is k-connected 가 성립한다.

이것 또한 증명의 아이디어만 제시하고 넘어가고자 한다. 이 Proposition역시 양쪽 방향으로 모두 증명할 것이다. 본 글에서는 한쪽 방향만 증명하고 반대쪽 방향은 생략한다.

양자 그래프가 k-connected일때 고전 그래프 또한 k-connected임을 보이자. 고전 그래프 G에서 certex cut이 $\{p_0, p_1, ..., p_m\}$ 이라면 $P=\sum_{i=1}^{m} \vert p_i \rangle \langle p_i\vert $로 정의하면 이 행렬이 rank는 m이상이며 $S_G$의 seperator임을 쉽게 보일 수 있다. 따라서 $k \leq m$이므로 G또한 k-connected이다.

고전 그래프가 k-connected일때 양자 그래프 또한 k-connected임을 보이는 과정은 조금 복잡하므로 논문 [1]을 찾아보길 권한다.
# 활용방안

양자 그래프의 활용방안은 굉장히 다양하다. 고전 그래프에 대응하는 방식으로 정의하였으므로 고전 그래프를 대응되는 양자 그래프를 만들어 연구할 수 있으며, 그 반대도 가능하다. 마지막으로 Channel을 고전 그래프로 모델링하는 예시를 소개하고 글을 마치고자 한다.

## What is _Channel_?

고전적인 채널은 어떠한 정보를 입력으로 주고, 채널의 출력이 전달이 되는 구조로 생각하면 편하다. 지금 우리가 쓰는 컴퓨터는 내부적으로 오류 정정이 거의 완벽하게 되기 때문에 정보를 전송할 때의 오류에 대해 별로 신경쓰지 않지만, 정보를 전달하는 과정에서 오류는 얼마든지 생길 수 있다.

정보 a를 전송하려고 했을 때, 오류 없이 그대로 a로 전송될 수도 있지만 오류가 생겨서 b로 전송될 수도 있다. 이 모든 것들이 확률적으로 결정된다고 가정하자. 좋은 채널이라면 정보가 손상 없이 그대로 전송될 확률이 높을 것이다. 이러한 오류 확률은 **채널의 용량** 개념과도 관련이 깊다. 이 글에서 핵심적으로 다루는 내용은 아니지만, 채널 용량은 굉장히 흥미로운 개념이다. 채널의 용량을 수학적으로 정의하고, 만약 채널에 어느 정도의 오류가 존재하더라도 정확한 정보를 전달할 수 있는 방법을 최초로 규명한 사람은 *클로드 섀넌*이다. 그렇다고 아무 채널이나 사용해서 정보를 전달할 수 있는 것은 아니다. 만약 0 또는 1을 전송하는 채널에서 오류 확률이 50%라면 이 채널을 사용해서 아무런 정보도 전달할 수 없으므로 채널의 용량은 0이다.

## Confusability Graph

어떠한 채널이 존재하면 그에 대응되는 *confusability graph*를 정의할 수 있다. 해당 그래프의 정점의 집합 V는 채널의 입력으로 가능한 정점들의 집합과 동일하다.

또한 confusability graph는 가중치가 있는 무방향 그래프이다. $\forall a, b \in V$ 에 대해, 두 정점 $a, b$가 가중치 $p$로 연결되어 있단 의미는 $a$와 $b$를 채널의 입력으로 넣었을 때 같은 출력이 나올 확률이 $p$라는 의미이다. 만약 같은 출력이 나올 확률이 0이라면 두 정점은 연결되어 있지 않다.

위의 정의를 잘 생각해 보면, confusability graph는 대응되는 채널에서 각 정점들이 얼마나 혼동될 수 있는지를 그래프의 연결로 표현한 것임을 알 수 있다.

## Quantum Confusability Graph

글의 마지막에 굳이 고전 그래프의 일종인 confusability graph를 소개한 것은 이유가 있다. confusability graph의 연결성을 따지는 것은 채널의 오류 가능성을 따지는 것이기 때문에 중요한데, 이 분석을 대응하는 quantum graph를 사용하면 효율적으로 진행할 수 있기 때문이다.

이처럼 양자그래프는 양자 채널, 양자 네트워크, 양자 암호학 등에 사용되는데, 이는 양자컴퓨팅이 가지는 이점 활용하여 그래프 이론과 연관된 문제를 더욱 효과적으로 해결할 수 있기 때문이다.

## Reference

[1] Chávez-Domínguez, J. A., & Swift, A. T. (2021). Connectivity for quantum graphs. Linear Algebra and its Applications, 608, 37-53.
