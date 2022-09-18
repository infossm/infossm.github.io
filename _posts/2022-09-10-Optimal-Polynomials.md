---
layout: post
title:  "Low degree optimal polynomial의 계산기하적 접근"
date:   2022-08-21
author: ainta
tags: [algorithm, FFT]
---


## Optimal Polynomial

일반적으로 함수 $f(x)$와 차수 $d$가 주어졌을 때, $\lvert P(x) - f(x) \rvert$의 최댓값 를 최소화하는 $d$차 다항함수를 $f$에 대한 degree $d$의 **optimal polynomial** 이라 합니다. 그러나 이러한 $P$를 구해야 하는 상황에서는 보통 함수 $f$를 모르는 상태에서, $f$가 $d$차 이하의 다항식일 것이라고 가정한 후 $P$를 추측해야 하는 경우가 잦습니다. 몇 개의 $x_i$에 대해 $f$의 실험값 $y_i = f(x_i) + \epsilon_i$ 가 주어져있을 때 오차 $\max (\lvert P(x_i) - y_i \rvert)$를 최소화하는 $P$를 어떻게 구할 수 있을까요?


## Linear Case

간단하게 $d = 1$인 경우, 즉 $P(x) = ax + b$인 경우를 먼저 살펴봅시다. 최소 오차의 범위를 탐색하는 Parametric search 방법을 이용하면 이 문제를 간단히 해결할 수 있습니다. 최소 오차가 $K$ 이하라면 $\max (\lvert ax_i + b - y_i \rvert) \le K$를 만족하는 $a, b$가 존재해야 합니다. 이는 모든 $i$에 대해 $y_i - K \le ax_i + b$ 및 $ax_i + b \le y_i + K$가 성립하는 것과 동치이고, $a, b$를 축으로 하는 좌표평면에서 각각의 부등식은 반평면 영역을 나타내이므로 [halfplane intersection](https://cp-algorithms.com/geometry/halfplane-intersection.html#half-plane-intersection-with-binary-search) 알고리즘을 통해 반평면들의 교집합이 존재하는지를 알아내면 $O(N \log N)$ 시간에 최소 오차가 $K$ 이하인지 판정할 수 있습니다. 

답이 $K$이하인지 질문해야하는 횟수는 $\log$(($y_i$ 범위) / (오차 허용 범위)) 스케일이고, 이를 간단히 $\log X$ 라고 하면 총 시간복잡도는 $O(N \log N \log X)$가 됩니다.
사실 각각의 반평면을 나누는 직선이 $K$값이 바뀜에 따라 평행이동만 하기 때문에 각 직선을 미리 기울기나 $y$절편으로 정렬을 해둔다면 그 뒤로는 $O(N)$시간에 halfplane intersection이 가능하므로 $O(N (\log N + \log X))$의 좀더 빠른 구현도 가능합니다.

## Quadratic Case

### Same Approach

$d = 2$인 경우에도 아까와 같은 방법을 적용해봅시다. 오차가 $K$ 이하이기 위해서는 $\max (\lvert ax_i^2 + bx_i + c - y_i \rvert) \le K$를 만족하는 $a, b, c$가 존재해야 합니다. 이는 모든 $i$에 대해 $y_i - K \le ax_i^2 + bx_i + c$ 및 $ax_i^2 + bx_i + c \le y_i + K$가 성립하는 것과 동치이며 각각의 부등식은 $a, b, c$를 축으로 하는 $\mathbb{R}^3$ 공간에서 halfspace가 됩니다. halfspace intersection은 $O(N^2 \log N)$ 시간에 가능하고 나아가 $O(N \log N)$ 시간에 하는 효율적인 방법도 존재하지만, 구현이 상당히 까다로운 문제 중 하나입니다. 조금 더 간단하게 이 문제를 해결할 수 있을까요?

### Convex space

앞선 방법에서는 $\max (\lvert ax_i^2 + bx_i + c - y_i \rvert) \le K$를 만족하는 $a, b, c$가 존재하는지를 판정해야 했습니다. 그렇다면, 해당 부등식을 만족하는 $(a, b, c, K)$들의 집합 $T \in \mathbb{R}^4$는 어떤 모양의 영역을 차지하고 있을까요? $p,q \in T$에 대해 선분 $pq$위의 점 $r = tp + (1-t)q$에 대해 생각해보면 $r \in T$를 만족함을 알 수 있습니다. 즉, $T$는 convex space입니다. 

$a$가 정해졌을 때 가능한 최소 오차를 $D_1(a)$, $b$가 정해졌을때 최소 오차를 $D_2(b)$, $c$가 정해졌을 때 최소 오차를 $D_3(c)$라고 하면, $T$의 convexness에 의해 $D_1, D_2, D_3$은 모두 아래로 볼록합니다. 전체 최소오차는 각각의 $i$에 대해 $D_i$의 최솟값과 동일한데, 아래로 볼록하므로 삼분탐색을 사용할 수 있는 셈입니다. $T$의 convexness를 이용하면 $a$가 고정된 상태에서 $b$에 따른 최소오차도 아래로 볼록함을 보일 수 있고, 따라서 $a$의 삼분탐색 내에서 $b$의 삼분탐색을 사용할 수 있습니다. $a, b$가 고정된 상태에서 최소오차는 선형 시간에 간단히 결정되므로 $O(N \log^2 X)$ 시간에 전체 최소오차를 구할 수 있습니다. 나아가, $d$가 2보다 커지더라도 $T$는 convex하기 때문에, 삼분탐색 $d$번을 통해 $O(N \log^d X)$ 시간에 문제를 해결할 수 있습니다.

이 문제는 SCPC 2022 본선에서 5번 문제로 출제되었으며, [링크](https://blog.kyouko.moe/m/75)의 풀이를 보시면 이해와 구현에 도움이 될 수 있습니다.

...하지만 이 문제는 $O(N \log^2 X)$보다 빠르게 해결할 수 있습니다!

### Linear case, revisited

$d=1$일 때 앞에서는 parameteric search를 이용해서 답의 범위를 절반으로 줄여나가는 방법을 제시했습니다. 바로 exact solution을 구하는 방법은 없을까요?

$P(x) = ax+ b$에서 $a$가 정해져있을때 최소 오차는 $y_i - ax_i$의 최댓값과 최솟값의 차의 절반입니다. 그리고 $y_i - ax_i$의 최댓값과 최솟값은 $(x_i, y_i)$들의 convex hull에 해당되는 점에서 나옵니다. $(x_i, y_i)$의 convex hull을 구한후 convex hull과 접하는 평행한 두 직선을 convex hull의 변을 따라 회전하는 형태의 [Rotating calipers](https://en.wikipedia.org/wiki/Rotating_calipers) 알고리즘을 사용하면 convex hull을 구한 이후 $O(N)$시간에 문제를 해결할 수 있습니다. Convex hull을 구하는데 $O(N \log N)$시간이 걸리므로 총 $O(N \log N)$의 시간복잡도를 가집니다.

### Faster solution

앞서 linear case 풀이를 quadratic case에 쉽게 적용할 수 있습니다. $a$에 대해 삼분탐색을 하고 나면 $a$가 정해진 상태에서 linear case가 됩니다 ($y_i$가 $y_i - ax_i^2$로 바뀐 형태) 그리고 $x_i$는 변하지 않으므로 처음에 입력을 받고 $x_i$ 오름차순으로 정렬을 해놓으면 convex hull을 $O(N)$시간에 구성할 수 있고, 따라서 정해진 $a$에 대해 $D_1(a)$를 $O(N)$시간에 구할 수 있습니다. 따라서 삼분탐색까지 $O(N \log X)$ 시간에 총 문제를 해결할 수 있습니다. higher degree에 이 방법을 적용하면 $O(N \log^{d-1} X)$ 시간복잡도가 되겠네요.

### Even Faster in even higher dimension

정해진 $K$에 대해 $-K \le ax_i^2 + bx_i + c - y_i \le K$를 만족하는 $(a, b, c)$가 존재하는지는 결국 linear programming 문제입니다. 그리고 놀랍게도 fixed dimension LP는 linear time에 가능함이 알려져 있습니다.
[Linear Programming in Linear Time When the Dimension Is Fixed](https://theory.stanford.edu/~megiddo/pdf/lplin.pdf) 논문에서 이를 다루고 있습니다. 정해진 차원에서 최소오차 문제를 $O(N \log X)$에 해결할 수 있는 것입니다. 

사실 $d=m$일 때의 문제는 모든 $i$에 대해 $0 \le a_mx_i^m + .... + a_0 - y_i \le K$ 가 성립할때 $K$의 최솟값을 구하는 문제이므로 하나의 LP로 표현이 가능합니다. 이를 통해 $d$가 정해져있으면 최소오차 문제는 선형에 해결이 가능함을 도출할 수 있습니다.

이제 선형보다 빠르게 할 수 없음은 자명하니, 다른 부분을 고민해봅시다.

## Halfplane Intersection, halfspace intersection

Optimal Polynomial을 구하는 빠르면서 가장 코딩이 쉬운 방법은 삼분검색을 $d$번 하는 것입니다. 한편 생각하기 가장 쉬운 방법은 아마도 linear case에서는 halfplane intersection, quadratic case에서는 halfspace intersection이 존재하는지 판정하는 것일 것입니다. 그러나 halfplane intersection은 구현이 간단하지도 않고 까딱하면 틀리기 굉장히 쉬운 알고리즘입니다. 여기에서는 일반적으로 알려진 방법인 직선들을 기울기나 절편으로 정렬한 후 stack을 이용해 구현하는 방법이 아니라, 보다 간단한 randomized 알고리즘을 소개합니다. 또한, 앞서 했듯이 이 결과를 3차원의 halfspace intersection과 더 높은 차원의 halfspace intersection 존재성 판정 알고리즘까지 이를 확장해볼 것입니다.

### Randomized algorithm

$n$개의 halfplane에 대해 $O(n \log n)$ 시간에 Halfplane intersection의 존재성 및 존재하는 경우 intersection에 포함되는 한 점을 구하는 문제를 randomized algorithm으로 해결할 수 있습니다. 여기서 핵심 아이디어는 halfplane을 하나씩 incremental하게 추가하되, intersection 전체 영역을 관리하는 것이 아니라 intersection에 포함되는 하나의 해만을 저장하는 것입니다.

 먼저, 주어진 halfplane들을 random한 순서로 나열합니다. 이를 $H_1, H_2, .., H_n$이라고 합시다. $H_1, H_2,.. ,H_i$에 모두 포함되는 점이 존재하는 경우, 점 $P_i$를 그러한 점들 중 $x$좌표가 가장 작은 점, $x$좌표가 최소인 점이 둘 이상이라면 그 중 $y$좌표가 최소인 점(lexicographically smallest)으로 정의합니다. 교집합이 없는 경우는 $P_i$가 정의되지 않습니다. 

$P_i$를 알고있는 상태에서, $P_{i+1}$을 구해봅시다. 만약 $P_i$가 $H_{i+1}$에 포함된다면, $P_{i+1} = P_i$임이 자명합니다. 그렇지 않은 경우, $P_{i+1}$는 $H_{i+1}$ 반평면을 나누는 직선 위의 점입니다. 해당 직선과 반평면 $H_1, .., H_i$와의 교집합 각각은 선분 형태로 나오고, 이들의 교집합 역시 선분 형태가 되거나 공집합이 됩니다. 공집합인 경우 halfplane intersection이 존재하지 않음을 리턴하면 되고, 선분 형태인 경우 그 중 가장 lexicographically smallest한 점을 $P_{i+1}$로 갱신하면 $O(i)$ 시간에 $P_{i+1}$을 구할 수 있습니다. $P_1,.., P_n$을 순서대로 계산하면 halfplane intersection의 존재성을 판정할 수 있고, 존재하는 경우 한 해를 구할 수 있습니다.

위에서 하나 설명하지 않은 부분이 있습니다. 처음에 $P_1$을 생각할 때 $x$좌표의 최솟값이 음의 무한대로 발산해서 $P_1$이 정의되지 않을 수 있다는 점입니다. 이는 간단한 트릭으로 해결할 수 있는데, 무한대 역할을 하는 값 $inf$를 설정하여 $x \le inf, x \ge -inf, y \le inf, y \ge -inf$의 4개의 halfplane이 맨 처음부터 있다고 가정하면 ($H_{-3}, H_{-2}, H_{-1}, H_0$과 같이 생각) $-inf \le x \le inf, -inf \le y \le inf$에 포함되는 영역만 고려할 수 있게 됩니다. 그러면 $i \ge 1$일 때 $P_i$가 항상 unique하게 정해짐이 보장됩니다. 뿐만 아니라, $H_1, .., H_{i}$의 추가되는 순서를 바꾸어도 intersection이 같기 때문에 최종 $P_i$가 항상 동일하게 나온다는 사실도 알 수 있습니다.

그러면 최종적으로 $P_1$부터 시작하여 $P_n$까지 계산하면 문제를 해결할 수 있는데, 알고리즘의 총 시간복잡도는 어떻게 될까요? $P_1, P_2, .., P_n$을 모두 계산하는데 걸리는 시간은 $P_i$가 $H_{i+1}$에 포함되지 않는 모든 $i$에 대한 합에 비례합니다. 그리고 $P_i$가 $H_{i+1}$에 포함되지 않으면 $P_{i+1}$은 $H_{i+1}$의 경계선 위의 점이 됩니다. $H_{i+1}$까지 추가했을때 intersection 영역인 convex hull을 생각했을 때, $P_{i+1}$은 convex hull에서 lexicographically smallest한 점입니다. 그리고 $H_1, .., H_{i+1}$ 중 경계선이 이 점을 지나는 halfplane은 일반적인 경우 2개 이하입니다. 이 케이스에서는 $H_1, .., H_{i+1}$의 순서가 무작위로 바뀌었을때 $P_{i+1}$을 계산할 때 갱신될 확률이 $\frac{2}{i+1}$ 이하인 것입니다.

만약 $P_{i+1}$을 지나는 Halfplane이 3개 이상인 경우는 어떨까요? $H_1, ..., H_{i+1}$ 중 $P_{i+1}$를 지나는 halfplane이 많더라도 convex hull에서 $P_{i+1}$과 인접한 변이 되는 경계선은 2개 이하입니다. 그리고 $H_{i+1}$이 그에 해당될 때만 $P_{i+1}$에서 갱신이 일어나는 것을 확인할 수 있습니다(그렇지 않으면 $P_i$가 이미 그 점이 됩니다).

따라서, $i$번째 halfplane에서 $P$의 갱신이 일어날 확률이 $\frac{2}{i}$ 이하입니다. $\frac{1}{1} + \frac{1}{2} + ... + \frac{1}{n} = O(\log n)$ 이므로 갱신이 일어나는 횟수는 평균적으로 $O(\log n)$회이고, 이에 따라 시간복잡도는 $O(n \log n)$이 됩니다. 최악의 경우 $O(n^2)$라고 생각할 수 있지만, 갱신이 $2 \log n$번 이상 일어날 확률도 $n$이 커짐에 따라 기하급수적으로 작아짐을 증명할 수 있기 때문에 사실상 practical하게는 $O(n \log n)$이 보장되는 알고리즘입니다.

### Halfspace intersection

한 차원을 높여 3차원 halfspace들의 intersection이 존재하는지 판정하고, 존재하는 경우 한 해를 구하는 문제를 생각해 봅시다. 이는 앞서 살펴본 randomized algorithm으로부터 다음과 같이 자연스럽게 확장 가능합니다. halfspace $H_1, .., H_n$에 대해 순서를 random하게 섞고, $H_1, .., H_i$의 intersection에 포함되는 점 중 lexicographically smallest한 점 $P_i$를 $i$를 1부터 차례로 증가시키면서 구해나갑니다. 만약 $P_i$가 $H_{i+1}$에 포함된다면 $P_{i+1} = P_i$로 두고, 그렇지 않다면 $H_{i+1}$의 경계면 위에 $P_{i+1}$이 놓이게 됩니다. $H_{i+1}$의 경계면과 $H_1, H_2, ..., H_i$의 교집합은 각각 halfplane이 되고, 앞서 살펴본 randomized halfplane intersection algorithm을 통해 $O(i \log i)$ 시간에 $P_{i+1}$을 계산할 수 있습니다. halfplane intersection을 할 때 3차원을 2차원 평면에 사영하면 3차원 기하를 사용하지 않고 구현이 가능합니다. 갱신이 일어나는 횟수는 halfplane intersection과 마찬가지로 생각해보면 3d convex hull에서 각 점은 3개의 평면이 결정하기 때문에 평균 $3 \log n$ 정도의 갱신횟수가 나오게 됩니다. 따라서 이 알고리즘의 최종 시간복잡도는 $O(n \log^2 n)$이 됩니다.

위에서의 결과를 확장해보면, $(d+1)$차원의 문제에서는 $P_i$가 $H_{i+1}$에 포함되지 않을 때 $d$차원의 문제를 해결하여 $P_{i+1}$을 갱신할 수 있고, 갱신 횟수는 $d \log n$ 정도가 됩니다. 이에 따라, $d$차원의 문제를 randomized algorithm으로 $O(d! \cdot n log^{d-1} n)$ 시간에 해결할 수 있습니다.


## 참고 자료

* [Halfplane intersection algorithm](https://cp-algorithms.com/geometry/halfplane-intersection.html)
* [Rotating calipers](https://en.wikipedia.org/wiki/Rotating_calipers)
* [Linear Programming in Linear Time When the Dimension Is Fixed](https://theory.stanford.edu/~megiddo/pdf/lplin.pdf)
* [Petr Mitrichev's blog](https://petr-mitrichev.blogspot.com/2016/07/a-half-plane-week.html)


