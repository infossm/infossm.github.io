---
layout: post
title: "장애물을 포함하지 않는 가장 큰 직사각형 찾기"
author: koosaga
date: 2019-10-20
tags: icpc, algorithm, computation-geometry
---

# 장애물을 포함하지 않는 가장 큰 직사각형 찾기

## Motivation

계산기하에서 장애물을 포함하지 않는 가장 큰 도형을 찾는 것은 핵심적인 문제 중 하나이다. 다양한 거리계, 그리고 도형의 모양에 따라서 서로 다른 알고리즘들이 존재한다. 예를 들어서, 다음과 같은 문제들을 생각할 수 있다.

* A) $n$ 개의 점들이 있을 때, 이 점을 포함하지 않으며 넓이가 가장 큰 원은 무엇인가?
* B) $n$ 개의 점들이 있을 때, 이 점을 포함하지 않으며 넓이가 가장 큰 직사각형은 무엇인가?
* C) $n$ 개의 점들이 있을 때, 이 점을 포함하지 않으면서 $x, y$ 축에 평행한 가장 큰 정사각형은 무엇인가?
* D) $n$ 개의 점들이 있을 때, 이 점을 포함하지 않으면서 $x, y$ 축에 평행하고, 한 변이 $x$ 축에 포함되는 가장 큰 직사각형은 무엇인가?
* E) $n$ 개의 점들이 있을 때, 이 점을 포함하지 않으면서 $x, y$ 축에 평행한 가장 큰 직사각형은 무엇인가?
* F) $n$ 개의 도형들이 있을 때, 이 도형들과 겹치거나 도형들을 포함하지 않는 가장 큰 원은 무엇인가?

이러한 문제들을 다룰 때 주의해야 할 점은 자명한 해가 존재한다는 것이다. 만약 원이나 직사각형을 어디에든지 설치할 수 있다면,  무한히 먼 곳에 큰 원이나 직사각형을 설치하면 된다. 이러한 해는 일반적으로 구하고자 하는 정답이 아니기 때문에, 보통 직사각형이나 원이 있을 수 있는 다각형 영역을 제한하는 조건이 붙어 있다. 이 글에서는 축에 평행한 직사각형 안에서만 원/직사각형을 놓을 수 있다고 제약 조건을 설정한다.

이 중, A와 C의 경우 각각 $L_1, L_2$ 거리계에서의 Voronoi Diagram을 사용하면 $O(n \log n)$ 에 구할 수 있다. 어떠한 점을 중심으로 하는 가장 큰 원의 경우 해당 점에서 가장 가까운 점만이 중요하기 때문에, 가장 가까운 점이 어디인지에 따라서 평면을 분할한 Voronoi Diagram이 문제를 해결하는 데 매우 유용하게 사용된다. 알고리즘 문제 해결의 영역에서, 이들은 보통 매우 어려운 문제로 나오거나, 입력 크기나 제약 조건들을 제시함으로써 조금 더 쉽게 풀릴 수 있는 식으로 변형되어 주어지는 편이다. KOI나 ICPC World Finals에서 이러한 식으로 비슷한 유형의 계산기하 문제들이 제시되었다. [문제 1] [문제 2] [문제 3]

D, E에 경우에는 직사각형이기 때문에 원이나 정사각형과 다르게 자유도가 2개로 늘어난다 (가로/세로 높이). 고로 일반적인 Voronoi Diagram등을 사용한 수법을 그대로 적용시키기 매우 곤란해진다. D의 경우에는 알고리즘 문제 해결에서 잘 알려진 Largest rectangle in a histogram 풀이 [문제 4] 를 변형하여 정렬 제외 $O(n)$ 시간에 계산할 수 있다. 스택을 사용하는 풀이법이 잘 알려져 있으며, 이러한 스택을 사용하는 풀이법은 이 문제 외에도 다양한 문제에 적용된다. 간단한 알고리즘은 아니지만, 일반적인 문제 해결 교육 내용에 들어가기 때문에 이 풀이는 잘 알려져 있다.

**이번 글에서 다룰 내용은 E번 문제 (가장 큰 축 평행 직사각형) 를 효율적으로 해결하는 방법이다.** 간단하게는, D번 문제의 해결방법을 응용하면 E번 문제를 $O(n^2)$ 시간에 해결할 수 있다. $x$ 축으로 가능한 후보가 많지 않기 때문에 이들을 전부 시도한 후 D번 문제를 해결하면 되기 때문이다. 이보다 빠른 알고리즘의 경우 1984년 [Chazelle et al.](https://link.springer.com/chapter/10.1007/3-540-12920-0_4), 1987년 [Aggarwal et al.](https://dl.acm.org/citation.cfm?id=41988) 등이 $O(n\log^3 n), O(n\log^2 n)$ 에 작동하는 더 빠른 알고리즘을 연구하였다. 꽤 오래 전 이미 해결이 된 문제이기 때문에, 알고리즘을 공부하는 입장에서는 이러한 논문을 읽기보다는 처음부터 생각해 보는 것도 좋은 공부라고 생각한다. 고로 이 문제를 혼자 $O(n \log^3 n)$ 에 해결해 보았으며 이에 관한 문제 역시 출제하였다[문제 5]. 

최근 ICPC 2019 Seoul Regional 예선에서 E번 문제를 약간 단순화시킨 버전[문제 6]이 가장 어려운 문제로 출제되었다. 고로 지금 ICPC 대회를 준비하는 일부 독자들은 이러한 문제에 대한 관심이 많다고 생각한다. 이 글에서 나는 해당 문제를 $O(n\log^3 n)$ 에 해결하는 방법을 소개함으로써 관련 문제들을 해결하는 데 도움을 주려고 한다. 

## 단계 1: 분할 정복으로의 환원

먼저 단순한 $O(n^2)$ 풀이를 생각하자. 직사각형의 아래쪽 변의 위치를 고정하고, Largest rectangle in a histogram 문제의 요령으로 해당 위치에 붙어 있는 최대 영역 사각형을 $O(n)$ 에 찾는다. 고정된 위쪽 점에 대해서, 해당 점을 윗변으로 포함할 때 가로/세로 변이 올 수 있는 가장 먼 위치를 계산하면 되는데. 각각의 위치는 스택을 사용하면 상수 시간에 계산할 수 있어서, 모든 답의 후보를 $O(n)$ 시간에 열거할 수 있다. 

이 풀이를 최적화하는 전략은 분할 정복을 사용하는 것이다. 점 집합이 $\{(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)\}$ 이라고 하며 일반성을 잃지 않고 모든 포인트가 $x$ 좌표로 정렬되었다고 하자. $ x = X_m $ 라인을 가로지르는 직사각형 중 최대 직사각형을 계산하려고 하는 것이 우리의 목표이다. 이를 찾으면 해당 라인을 기준으로 남은 점들을 분할할 수 있으며 이들에 대해 동일한 분할 정복을 사용하면 된다. 

직사각형이 $ x = X_m $ 라인을 가로 지르면, 직사각형이 $x$ 좌표 양 옆으로 확장하는 것을 차단하는 경계가 왼쪽 / 오른쪽에 생긴다. 이 경계는 점이거나, 혹은 점 전체를 둘러싸는 bounding box이다. 이 경계를 각각 고정시키자. 이후 사각형을 $x = X_m$ 라인을 통해 왼쪽/오른쪽 두 부분으로 자르고, 각 2개의 사각형마다 위쪽과 아래쪽을 독립적으로 늘린다. 이렇게 되면, 늘린 왼쪽/오른쪽 직사각형은 왼쪽/오른쪽 두 부분에 있는 maximal rectangle이 된다. 고로, 단순한 $O(n^2)$ 풀이에서 사용한 스택을 사용하여 열거할 수 있는 직사각형이다. 이제 관점을 반대로 바꾸고 두 개의 maximal rectangle 쌍으로 유도 된 가장 큰 사각형을 찾으면, 너비는 두 높이의 합이고 높이는 두 구간의 교집합이 된다. 이 단계까지 오는 데 사용된 계산량은 $O(n)$ 이다.

이제 **정복** 단계는 구간을 다루는 1차원 문제로 추상화되었다. 가중치가 있는 구간 집합이 두 개 주어지고 각 두 집합에서 하나의 구간을 선택하여 $ (w_x + w_y) \times |[s_x, e_x] \cap [s_y, e_y]|$을 최대화 해야 한다. 이 때 $[s_i, e_i]$ 는 구간이고, $w_i$ 는 이 구간에 배정된 무게이며, $x, y$ 는 서로 다른 구간 집합에서 비롯된다. 이러한 추상화가 있지만, 여전히 이 문제의 기하학적 특성이 필요하므로 이를 잊어서는 안된다. 

여기까지의 환원은 이해하기 어렵지 않고, 실제로 이 문제에서 가장 쉬운 부분이지만, 구간 집합을 실제로 찾는 구현은 약간 까다로운 편이다. ICPC 2019 Seoul Regional 예선에 나온 문제는 여기까지의 환원을 구하는 것이 상대적으로 간단한 편이다. Largest rectangle in a histogram을 변형 없이 그대로 사용하기만 해도 구간 집합을 구할 수 있기 때문이다. 이제부터 서술할 내용은 두 문제 모두에 해당되는 부분이다.

## 단계 2: Monotone Queue Optimization으로의 환원.

변형된 문제에는 두 가지 경우가 있다:

* 두 개의 구간을 뽑았을 때, 한 구간이 다른 구간을 포함함
* 그렇지 않음

이 두 가지를 따로 처리해 보자. 첫번째 구간은 생각보다 간단하다. 교집합의 길이는 작은 쪽의 길이로 결정된다. 작은 쪽의 구간을 고정하면, 이것은 경계선에서 가장 가까운 점을 찾는 것이 된다. 점의 $x$ 좌표를 인자로 가지는 Segment Tree를 사용하면 이는 간단히 $O(n\log n)$ 에 가능하다.

두 번째 경우에는, 두 구간에는 교집합은 있으나. 한 구간이 포함되지는 않는다. 일반성을 잃지 않고 $ s_x <s_y <e_x <e_y $라고 조건을 추가하면, 답은 $(w_x + w_y) \times (e_x-s_y)$ 임을 알 수 있습니다. 추가한 조건이 없는 경우에도 이는 계산하기가 쉽지 않다. 

조건 없이 임의의 두 구간 집합에 대해 최대 $ (w_x + w_y) \times (e_x-s_y) $를 찾는 문제를 논의해 보자. 2차원 평면에 점을 $ (e_x, -w_x), (s_y, w_y) $로 플로팅하면 최적 매칭에 어떠한 단조성이 있음을 알 수 있다. 정확히는, 점 $ (e_x, -w_x) $ 를 $x$ 좌표가 증가하는 방향으로 바꾸면 이에 대응하는 최적 매칭의 $x$좌표가 감소한다. 이것은 각각의 매칭에 의해 유도 된 사각형의 영역을 비교하면 증명할 수 있다.

최적의 매칭을 효율적으로 계산하기 위해 분할 정복 최적화 또는 *Monotone queue* 를 사용할 수 있다. *Monotone queue optimization* 은 한국에서 잘 알려져 있지는 않은 방법인데, Convex hull trick의 일반화라고 생각하면 편하다. 여기서 해당 방법에 대해서 간단히 논의한다. $ (e_x, -w_x) $ 집합 $S$, $ (s_y, w_y)$ 집합 $T$, 그리고 $T$ 에 속하는 서로 다른 두 점 $x, y$ 를 잡자. $S$ 에 있는 어떠한 점들은 $x$ 를 잡는 것이 넓이를 최대화 하며 어떠한 점들은 그 반대일 것이다. 이 비교 기준을 토대로 $S$ 를 두 점 집합으로 쪼개면, 이 두 집합은 각각 접미사와 접두사를 이룬다. 고로, $x, y$  중 어떠한 점이 최적인지 갈라지는 특정한 지점이 $S$ 에 존재한다. 이 지점은 이분 검색을 사용하여 찾을 수 있다.

이제 점을 저장하는 데이터 구조를 관리하자. $x$로 포인트를 고르는 것이 $ x + 1 $ 또는 $ x-1 $를 고르는 것과 비교하여 열등한 선택이 될 때마다 반복적으로 그러한 포인트를 제거한다. 이 때 열등한 선택이라고 함은, $x, x + 1$ 을 비교하고, $x, x - 1$을 비교하여 이 셋 중 $x$ 를 고르는 것이 최적인 지점의 구간을 계산했을 때, 이 구간이 비었다는 것을 뜻한다. 이는 볼록 껍질 최적화와 매우 유사한 스택을 사용하여 시뮬레이션 할 수 있습니다. 각 삽입은 이분 검색에 $O(\log N)$ 시간이 소요되므로 $ O (N \log N) $ 시간 내에  문제를 해결할 수 있다. 이러한 방식으로 해결할 수 있는 문제가 최근 USACO 및 서울대학교 교내 대회에 출제된 바 있다. [문제 7] [문제 8]

이제 단순한 솔루션은 2개의 중첩된 분할 및 정복을 수행하여 $ s_y, e_y $에 제한을 두는 것으로 가능하다. 직관적으로 이것은 문제를 해결하기 위해 2D 세그먼트 트리를 작성하는 것과 동치이다. 그런 다음 각 정복 단계마다 비교할 $ O (N \log ^ 2 N) $ 쌍이 있으므로 $ O (N \log ^ 3 N) $ 에 단계 2를 해결할 수 있다. $ T (N) = 2T (N / 2) + O (N \log ^ 3 N) = O (N \log ^ 4 N) $ 이므로 이 방법을 최적화해야 한다.

## 단계 3. 추가적인 단조성의 관찰과 변형된 Monotone Queue Optimization.

총 3개의 분할 정복을 사용하는 우리의 시도가 실패하였지만, 그래도 일종의 중첩된 분할 정복 접근을 계속 사용해 보자. $ s_y \le m < e_x $ 인 모든 포인트 매칭을 고려하고, 다른 종류의 포인트 매칭은 재귀적으로 해결하는 식의 분할 정복 알고리즘을 사용한다. $ s_y \le m < e_x $를 만족하는 모든 포인트 매칭를 고려하려면 $ m $ 포인트를 통과하는 모든 구간을 고려해야 하는데, 이 때 고정된 왼쪽 (3사분면) 점에 대해서 오른쪽 (1사분면) 점으로 가능한 점들의 집합은 구간을 이룬다. 이는 가로 / 세로 축으로 쪼개져서 총 4개의 체인으로 구성된 현재 상태에서, 3사분면에 있는 체인에서 각각 2/4사분면 체인에 "광선" 을 쏘아서 반사시켰다고 하면, 이러한 각 제약 조건이 시작점과 끝점을 가진다는 점을 알 수 있다. 또한, 이 구간들은 단조적이다: 구간을 시작점 순으로 정렬하면, 이 구간의 끝점들 역시 증가한다. 

이러한 제약 조건 상에서는, 놀랍게도 $ O (N \log N) $ 시간으로 매칭을 계산할 수 있다. 이해를 돕기 위해, 일치 가능한 구간의 길이가 $ K $라고 가정해 보자. 그렇다면, 첫 번째 포인트 세트를 길이 $ K $의 연속 구간으로 분해 할 수 있으며 각 쿼리마다 접두사와 접미사에 대해 문제를 풀면 된다. Monotone queue는 뒤쪽에 일종의 "삽입" 을 해 줄 수 있기 때문에, 이러한 구간이 접두사 / 접미사로 보장되는 경우 솔루션을 약간 수정하기만 하면 여전히 $ O (N \log N) $ 의 계산량으로 해결할 수 있다.

일치하는 구간의 길이는 실제로 $ K $가 아니지만 매우 유사한 방식으로 접두사 / 접미사로 분할해 줄 수 있다. 하나의 쿼리 구간을 선택한 후, 교집합이 있을 때까지 구간을 추가해 준다. 어느 순간 교집합이 없어지게 되는데, 이 때 그 교집합을 없앤 위치를 기준으로 구간을 자르면, 각 구간이 접두사 / 접미사로 깔끔하게 쪼개짐을 알 수 있다. 

이제 이를 통해서 분할 정복 알고리즘을 완성할 수 있다. $ y = m $ 줄을 지나는 구간의 경우, 모두 모아서 $ O (N \log N) $ 시간에 최적의 매칭을 계산해 준다. 구간이 $ y = m $를 교차하지 않으면 상단 또는 하단에 있으므로 삽입 위치를 알 수 있습니다. 교차하는 경우 역시 고려해줘야 하는데, 각 포인트가 왼쪽인지 오른쪽인지에 따라 위쪽 또는 아래쪽에 배치해야 한다 (둘 다 배치할 수 있음에 유의하자.) 전체적으로 각 구간은 재귀 트리의 $ O (\log N) $ 노드에 표시되므로, 매칭 알고리즘은 $ O (N \log N) $ 개의 구간을 입력으로 받아 $ O (N \log^ 2 N) $ 시간에 답을 내놓는다. 이것이 분할 정복의 Merge 단계이고, 마스터 정리를 추가로 사용하면 전체 시간 복잡도가 $O (N \log ^ 3 N) $ 이 된다. 구현하는 것이 복잡할 것이라고 생각할 것이고, 실제로도 그렇지만, 알고리즘의 복잡성에 비해서는 생각보다 짧은 코드가 나오는 편이다. 

## 제시된 문제들

[문제 1] [KOI 2004 도서실카펫](https://www.acmicpc.net/problem/2601)

[문제 2] [ICPC World Finals 2018 Panda Preserve](https://www.acmicpc.net/problem/15695)

[문제 3] [Canadian Computing Olympiad 2008 Landing](https://dmoj.ca/problem/cco08p6)

[문제 4] [BOJ 6549 히스토그램에서 가장 큰 직사각형](https://www.acmicpc.net/problem/6549) 

[문제 5] [XIX OpenCup Grand Prix of Daejeon. Bohemian Rhaksody](https://www.acmicpc.net/problem/17461)

[문제 6] [ICPC 2019 Seoul Regional Preliminary. Steel Slicing](https://www.acmicpc.net/problem/17527)

[문제 7] [USACO February 2019 Contest Platinum Division. Mowing Mischief](https://www.acmicpc.net/problem/17032)

[문제 8] [서울대학교 2019 교내대회. 꽃집](https://www.acmicpc.net/problem/17461)
