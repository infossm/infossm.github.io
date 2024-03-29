---
layout: post
title: '완전 매칭을 찾는 병렬 알고리즘'
author: junis3
date: 2023-05-21
tags: [random]
---

# PRAM Model

PRAM (Parallel Random Access Machine) 모델은 병렬 처리를 이론적으로 표현한 모델이다. $P$개의 프로세서와 크기 $M$의 공통 메모리로 이루어져 있다. PRAM 모델에서 한 번의 (병렬) 연산은 다음과 같은 과정을 통해 이루어진다:

1. 공통 메모리의 어떤 위치에 있는 값을 읽는다.
2. 간단한 (unit-step) 연산을 실행한다.
3. 공통 메모리의 어떤 위치에 값을 쓴다.

여러 개의 프로세서가 동시에 연산을 수행할 때 총돌할 가능성을 배제하기 위해, 다음과 같은 가정을 한다. 먼저, 모든 프로세서가 값 읽기 (1.)를 끝낸 다음에 연산 실행과 값 쓰기 (2., 3.) 과정을 실행한다고 가정한다. 또한, 한 번의 연산에서 여러 개의 프로세서가 값을 쓰는 위치는 모두 서로 다르다고 가정한다. 즉, 한 번의 연산에서 하나의 위치에 값이 여러 번 쓰여질 때 순서 관계에 대한 걱정을 하지 않기로 하자.

이 모델은 적절히 추상화되었으면서도 실제 병렬 알고리즘의 시간 복잡도 분석에 큰 도움을 준다. 실제로도, PRAM 모델은 지금은 역사 속으로 사라진 대회인 Distributed Codejam (DCJ)에서 다루었던 문제들과 결이 비슷하다는 것을 확인할 수 있을 것이다.

PRAM 모델에서 문제를 푸는 '효율적인' 방법이 존재하는지를 따지기 위해, 결정 문제의 집합 (class) $\mathcal{NC}$를 정의한다. 다항 개수의 프로세서를 이용할 때 알고리즘의 시간 복잡도를 다항 시간 미만 ($O(\log^k N)$)으로 줄일 수 있다면, 이 문제를 $\mathcal{NC}$ 문제라고 부른다. 이 때 $\mathcal{NC} \subseteq P$임은 자명하다. $\mathcal{NC}$ 알고리즘이 존재한다면 이 알고리즘의 동작을 하나의 프로세서로 쭉 펴서 $P$ 알고리즘을 구성할 수 있기 때문이다. 즉, $P$ 알고리즘을 병렬화해서 다항 시간 미만으로 줄일 수 있는 문제가 $\mathcal{NC}$라고 이해하면 되겠다.

$\mathcal{RNC}$는 $\mathcal{NC}$와 비슷한데, 단지 문제를 푸는 알고리즘에 차이가 있다. 결정 문제를 푸는 다음과 같은 다항시간 알고리즘이 존재할 때, 이 문제를 $RP$ 문제라고 부른다.

- 답이 "아니오"일 때에는 항상 "아니오"라고 답한다.
- 답이 "예"일 경우에는 상수 $c>0$ 이상의 확률로 "예"라고 답한다.

$RP$ 알고리즘은 충분히 많이 실행하면 충분히 높은 확률로 ($\epsilon$-$\delta$ 논법 비슷하게) 답을 얻을 수 있다. 문제를 푸는 $RP$ 알고리즘을 (다항 개수의 프로세서로) 병렬화해서 다항 시간 미만으로 줄일 수 있을 때, 이 문제를 $\mathcal{RNC}$ 문제라고 한다.

이 글에서는 PRAM 모델에서 잘 알려진 알고리즘 문제인 완전 매칭 문제를 풀어볼 것이다. 즉, 이 글에서 완전 매칭 문제를 푸는 병렬 알고리즘을 제시할 것이다. 그래프에 완전 매칭이 존재하는지 판별하는 $\mathcal{RNC}$ 알고리즘뿐 아니라, 완전 매칭 하나를 실제로 구성하는 $\mathcal{RNC}$ 알고리즘 또한 보일 것이다.

# 완전 매칭 판별하기

Tutte 정리를 이용하면 그래프의 완전 매칭이 존재하는지 간단히 판별할 수 있다.

**정리 1.** (Tutte 정리)

그래프 $G(V, E)$에서 다음 방법으로 행렬 A를 만든다. $n \times n$ 행렬 $A$의 각 원소의 값은 최초에 모두 0이다. 이제 $G$의 각 간선 $(u, v) \in E$에 대해, 변수 $x _ {uv}$를 부여해, $A _ {uv} := x _ {uv}$, $A _ {vu} := -x _ {uv}$를 대입한다. 이 행렬의 행렬식이 (변수들의 값과 무관하게 항등적으로) 0이 되면, $G$에는 완전 매칭이 존재하지 않는다. 그렇지 않다면, $G$에는 완전 매칭이 존재한다.

**증명.**

$V = \left\\{1, \cdots, n \right\\}$에 대한 각 순열 $\sigma \in \mathbf{S} _ n$에 대해, $\mathrm{val}(\sigma) = \prod _ {i=1}^{n} A _ {i \sigma(i)}$를 정의하자. 모든 $i \in V$에 대해 간선 $(i, \sigma(i))$이 $G$에 존재하면, $\mathrm{val}(\sigma)$이 0이 아닌 값을 가진다. 순열 $\sigma$를 순열 그래프로 시각화해서 생각하자. 정점이 $n$개이고 각 $i$에 대해 간선 $(i, \sigma(i))$가 있는 방향 그래프를 생각하면, 이 방향 그래프는 몇 개의 사이클로 이루어져 있다. 간선의 방향을 가리고 보면, 이 그래프는 $G$의 부분 그래프이다.

$A$의 행렬식은 다음과 같은 형태로 나타난다. ($\mathrm{sign}(\sigma)$의 값은 순열 $\sigma$의 부호에 따라 $1$ 또는 $-1$이다.)

$$
\det(A) = \sum_{\sigma \in \mathbf{S}_n} \mathrm{sign}(\sigma) \times \mathrm{val}(\sigma)
$$

만약 $\sigma$에 대한 그래프에 홀수 사이클이 존재한다면, 이 홀수 사이클에 대해서만 간선의 방향이 반대인 순열 $\sigma'$를 생각할 수 있다(홀수 사이클이 여러 개라면, 가장 번호가 작은 정점의 번호가 가장 작은 사이클로 정하는 등의 방법으로 아무튼 혼동이 생기지 않게 홀수 사이클 하나를 뒤집는다). Tutte matrix $A$가 skew-symmetric이기 때문에, $\mathrm{val}(\sigma') = -\mathrm{val}(\sigma)$이다. 따라서 순열 $\sigma$는 $\det(A)$의 값에 아무런 영향을 끼치지 못한다.

따라서, 짝수 사이클만으로 이루어진 순열들만이 $\det(A)$의 값에 영향을 끼칠 수 있다. 짝수 사이클만 으로이루어진 순열은 $\sigma$로 만들어지는 순열 그래프는 (간선의 방향을 가리고 보았을 때) $G$의 부분 그래프이기 때문에, 각 사이클의 홀수번째 간선만 고르는 방법으로 $G$의 perfect matching을 구성할 수 있다. 즉, 이러한 순열은 $G$의 perfect matching에 대응된다. $\det(A) \neq 0$는, (행렬식의 값에 영향을 미치는) 짝수 사이클만으로 이루어진 순열의 존재와 동치이고, 따라서 perfect matching의 존재와 동치이다. 증명 끝.

Tutte 정리는 General matching 알고리즘에 대해 다룬 [해당 글](https://infossm.github.io/blog/2020/08/19/general-matching/)에서도 소개되어 있다. 여기서는 실제로 완전 매칭이 존재하는지 판별할 수 있는 $RP$ 알고리즘까지 제시한다.

행렬 $A$를 구성하는 작업은 자명히 $\mathcal{NC}$로 구성 가능하고, $A$의 행렬식을 계산하는 $\mathcal{NC}$ 알고리즘도 알려져 있다고 한다. $O(n^{3.496})$개의 프로세서가 있으면, $O(\log^2n)$의 시간 복잡도로 행렬식, 역행렬과 수반행렬을 계산할 수 있다. 이 알고리즘에 대해서는 이 글에서 다루지 않을 것이다. 대신, 관심있는 사람은 아래 참고문헌 2.를 읽어보아라. 따라서 위 글의 $RP$ 알고리즘을 다항 시간 미만으로 병렬화할 수 있으며, 완전 매칭의 존재 여부를 판정하는 문제는 $\mathcal{RNC}$이다.

완전 매칭이 존재하는 그래프에서 실제로 가능한 완전 매칭 중 하나를 구성하는 것은 더 어려운 문제이다.

# 완전 매칭 구성하기

완전 매칭이 유일한 경우에는, 사실 완전 매칭을 구성하는 간단한 $\mathcal{NC}$ 알고리즘이 존재한다. 간선을 하나씩 지워본 다음에 "이 간선을 지워도 아직 완전 매칭이 존재하는지" 판정하면 된다. 만약 그래프에서 어떤 간선을 지워도 완전 매칭이 존재하면 이 간선은 완전 매칭에 포함되지 않는다. 그렇지 않다면, 이 간선은 완전 매칭에 포함된다.

위 알고리즘은 완전 매칭이 유일하게 존재한다는 특이한 성질에 의존하고 있다. 대부분의 그래프에는 완전 매칭이 여럿 존재할 것이다. 이 때에도 그래프 $G$의 완전 매칭을 효율적으로 구성하는 알고리즘이 존재할까? 이 때 완전 매칭이 하나만 존재하는 것이 아니기 때문에, 특정한 기준에 따라 하나의 완전 매칭을 찾아내어야 한다.

여기서 소개할 알고리즘은 다음 두 단계에 걸쳐 이루어진다. 먼저, $G$의 간선에 무작위의 가중치를 메긴다. 다음, 간선의 가중치의 합을 최소로 만드는 완전 매칭을 구한다. 지금은 뜬구름 잡는 이야기 같아 보이지만, 모든 것이 잘 맞아떨어져서 제대로 작동한다면 그래프의 완전 매칭 하나를 구성할 수 있을 것이다.

## 무작위 가중치 메기기

가중치를 메긴 다음 가중치의 합을 최소로 하는 완전 매칭이 유일하다면, 완전 매칭이 유일할 때의 알고리즘과 유사한 알고리즘을 사용해 이를 찾을 수 있을 것이다. 사실 다행히도, "대충 적당히 넓은 범위에서" 간선에 가중치를 메기면 가중치의 합이 최소인 매칭이 유일할 확률이 충분히 높다. 이를 **정리 2**에서 일반화하여 증명한다.

**정리 2.**

크기가 $m$인 집합 $X = \left\\{ x _ 1, \cdots, x _ m \right\\}$의 부분 집합을 모아둔 집합족 $\mathcal{F} = \left\\{S _ 1, \cdots, S _ k\right\\}$ (각 $1 \le i \le k$에 대해, $S _ i \subseteq X$)를 생각하자. 이 때 $X$의 각 원소에 $1$ 이상 $2m$ 이하의 무작위 정수 가중치 $w : X \rightarrow \left\\{1, \cdots, 2m\right\\}$를 부여하자. 그러면, $\mathcal{F}$의 원소들 중 가중치의 합이 최소인 집합이 유일할 확률은 $\frac{1}{2}$ 이상이다.

**증명.**

일반성을 잃지 않고, $X$의 각 원소 $x _ i$에 대해서, $\mathcal{F}$에 $x _ i$가 있는 집합이 하나 이상 존재하고, $x _ i$가 없는 집합이 하나 이상 존재한다고 가정하자. (그렇지 않은 원소가 있다면, 그 원소를 무시하고 생각해도 된다.)

$X$의 어떤 원소 $x _ i$에 대해, $x _ i$를 제외한 모든 가중치가 정해졌다고 하자. 즉, $w(x _ i)$를 제외한 모든 $w$의 함숫값이 정해졌다고 하자. 이 때, $\mathcal{F}$에서 $x _ i$를 포함하지 않는 집합들을 모은 부분집합족을 $\mathcal{F} _ i$, $x _ i$를 포함하지 않는 집합들을 모은 부분집합족을 $\overline{\mathcal{F} _ i}$로 두자. 또, $\mathcal{F} _ i$에서 가중치의 합이 제일 작은 집합의 가중치의 합(에서 $w(x _ i)$를 뺀 값)을 $W _ i$로, $\overline{\mathcal{F} _ i}$에서 가중치의 합이 가장 작은 집합의 가중치의 합을 $\overline{W _ i}$로 두자.

이 때 $x _ i$의 가중치 $w(x _ i)$가 $w(x _ i) = \overline{W _ i} - W _ i$일 때에는, $\mathcal{F}$에서 가중치의 합이 가장 작은 집합이 두 부분집합 $\mathcal{F} _ i$와 $\overline{\mathcal{F} _ i}$ 모두에 존재한다. 즉, 가중치의 합이 최소인 집합에 $x _ i$가 존재할 수도 있고, 존재하지 않을 수도 있다. 그러나, 그렇지 않을 때에는 가중치의 합이 최소인 집합에는 반드시 $x _ i$가 존재하거나, 반드시 $x _ i$가 존재하지 않는다. $w(x _ i) = \overline{W _ i} - W _ i$가 성립할 확률은 $\frac{1}{2m}$ 이하이다. 그리고 모든 $i$에 대해서 이 식이 성립하지 않으면, 가중치의 합이 최소인 집합은 유일하다. 이 확률은 $1 - m \times \frac{1}{2m} = \frac{1}{2}$ 이하이다. 증명 끝.

자명히, **정리 2**를 완전 매칭 문제에도 적용할 수 있다. 각 간선에 $1$ 이상 $2m$ 이하의 무작위 정수 가중치를 부여하면, $\frac{1}{2}$ 이상의 확률로 가중치의 합이 최소인 완전 매칭이 유일하게 결정된다.

## 가중치의 합이 최소인 완전 매칭 구성하기

하나의 벽을 넘었다. 이제 가중치의 합이 최소인 완전 매칭이 유일할 때 그 완전 매칭을 찾을 수 있어야 한다. Tutte 정리를 응용한다. 그래프 $G$에 대한 Tutte 행렬을 $B$로 두자. Tutte 정리에서 $G$의 각 간선에 임의의 변수 $x _ {ij}$를 부여했던 것을 기억할 것이다. $x _ {ij} = 2^{w _ {ij}}$를 대입하면, $B$에 대해 다음이 성립한다.

**정리 3.**

$G$에 가중치의 합이 최소인 완전 매칭 $M _ {\min}$이 유일하게 존재한다고 가정하자. 이 때, $\det(B) = 0$이다. 그리고, $M _ {\min}$의 가중치의 합이 $W$일 때, $\det(B)$의 2의 지수는 $2W$와 같다. 즉, $\det(B)$는 $2^{2W}$의 배수이고, $2^{2W+1}$의 배수가 아니다.

**증명.**

Tutte의 정리에 대한 증명의 일반화이다. $B$의 행렬식은 다음과 같은 형태로 나타난다는 점을 짚고 가자.

$$
\det(B) = \sum_{\sigma \in \mathbf{S}_n} \mathrm{sign}(\sigma) \times \mathrm{val}(\sigma)
$$

가능한 순열 $\sigma$ 가운데, 짝수 사이클만으로 이루어진 순열만이 $\det(B)$의 값에 영향을 끼친다. 이 중, 길이가 2인 사이클만으로 이루어진 순열을 생각하자. 이러한 순열은 $G$의 완전 매칭 $M$에 대응되고, 각 사이클은 $G$의 간선에 대응된다. $M$의 가중치의 합을 $W(M)$이라 하자. 이 때 $\mathrm{val}(\sigma) = \prod _ {i=1}^n x _ {i \sigma(i)}$의 값에는 $M$을 이루는 각 간선의 가중치가 두 번씩 곱해지기 때문에, $\mathrm{val}(\sigma) = \pm 2^{2 W(M)}$이 된다. 이는 $2^{2W}$의 배수이다. 이 중에서 가중치의 합이 최소인 완전 매칭 $M _ {\min}$은 유일하게 존재하기 때문에, 항 $\pm 2^{2W}$도 정확히 한 번 더해진다. (다음 문단을 읽으면 이 사실이 확실해진다!) 따라서, $\det(B)$는 $2^{2W}$의 배수이고, $2^{2W+1}$의 배수가 아니다.

짝수 사이클만으로 이루어진 순열들 중 길이가 4 이상인 사이클이 포함되어 있는 경우는 아직 고려하지 않았다! 사실 이 경우는 $\mathrm{val}(\sigma)$가 $2^{2W+1}$의 배수가 되기 때문에 고려할 필요가 없다. 길이가 4 이상인 사이클에서 홀수번째 간선들을 모은 매칭 $M_1$과 짝수번째 간선들을 모은 매칭 $M_2$를 생각할 수 있다. $\lvert \mathrm{val}(\sigma) \rvert = 2^{W(M _ 1)} \cdot 2^{W(M _ 2)}$이 성립한다. 두 매칭 중 $M _ {\min}$과 동일한 매칭은 1개 이하이기 때문에, $2^{W(M _ 1)}$과 $2^{W(M _ 2)}$ 중 적어도 하나는 $2^{2W+2}$의 배수이다. 따라서 $\mathrm{val}(\sigma)$는 $2^{2W+1}$의 배수이다. 증명 끝.

$B$와 $\det(B)$ 모두 $\mathcal{NC}$ 시간에 계산할 수 있으므로, 가중치의 합이 최소인 완전 매칭이 유일할 경우 $W$의 값도 $\mathcal{NC}$에 계산할 수 있다. **정리 3**은 가중치의 합이 최소인 완전 매칭의 가중치의 합을 구할 뿐이었다. 그러나 Tutte 정리를 응용해 완전 매칭이 유일할 때 해당 완전 매칭을 구성해냈듯, **정리 3**을 응용하면 가중치의 합이 최소인 완전 매칭이 유일할 때 해당 완전 매칭을 찾을 수 있다.

**정리 4.**

가중치의 합이 최소인 $G$의 완전 매칭이 $M$이고, $M$의 가중치의 합이 $W$라 하자. 간선 $(i, j)$이 $M$에 속할 필요충분조건은

$$
\frac{\det({B}^{ij}) \cdot 2^{w _ {ij}}}{2^{2W}}
$$

가 홀수인 것이다. $B^{ij}$는 $B$에서 $i$번째 행과 $j$번째 열을 제거한 $(n-1) \times (n-1)$ 행렬이다.

**증명.**

**정리 3**의 증명을 따라왔다면 눈에 보일 것이다. $\det(B)$와 $\det(B^{ij})$의 차이가 무엇일지 생각해 보자. 역시 아래 등식의 관점에서 생각하자.

$$
\det(B) = \sum_{\sigma \in \mathbf{S}_n} \mathrm{sign}(\sigma) \times \mathrm{val}(\sigma)
$$

$\det(B)$를 이루는 각 순열 $\sigma$에 대해서, 순열 그래프에서 $i$번 정점과 $j$번 정점을 합치고, $i$번 정점에서 $j$번 정점으로 가는 간선이 존재한다면 제거한 뒤의 값이 $\det(B^{ij})$이다. 즉, 간선 $(i, j)$를 포함하는 모든 순열 $\sigma$에 대해서 $\mathrm{val}(\sigma)$의 값을 $2^{w _ {ij}}$만큼 나눈 값이다. $(i, j)$이 $M_{\min}$를 나타내는 순열에 포함된다면, $M_{\min}$에 대응되는 항 $2^{2W}$을 $2^{w _ {ij}}$으로 나누게 되었을 것이고, 위 식의 값은 홀수가 되었을 것이다. 증명 끝.

이제 알고리즘을 구성하는 모든 요소들이 증명되었다. 다음 알고리즘을 통해 그래프 $G = (V, E)$의 완전 매칭 $M$을 구성할 수 있다.

1. 각 간선 $(i, j)$에 대해, $1$ 이상 $2m$ 이하의 무작위 가중치 $w _ {ij}$를 부여한다.

2. $G$의 Tutte 행렬을 구성한 다음, 각 변수 $x _ {ij}$에 $2^{w _ {ij}}$를 대입한다.

3. $\det(B)$를 계산한다.

4. $\det(B)$의 2의 지수를 계산한다. 지수는 짝수가 될 것이다. 지수를 $2W$라 두자.

5. $B$의 수반 행렬 $\mathrm{adj} (B) = \det(B) \times B^{-1}$을 계산한다. $\mathrm{adj}(B)$의 $j$행 $i$열의 값은 $\det(B^{ij})$와 같다.

6. 각 간선 $(i, j)$에 대해, $r _ {ij} = \frac{\det(B^{ij}) 2^{w _ {ij}}}{2^{2W}}$를 계산한다. $r _ {ij}$가 홀수이면 $M$에 간선 $(i, j)$를 추가한다.

1., 2., 4., 6.은 자명하게 $O(m)$개의 프로세서가 있으면 상수 시간에 처리할 수 있고, 행렬의 행렬값, 역행렬, 수반 행렬을 계산하는 과정인 3., 5.는 위에서 언급했듯 $O(n^{3.496}m)$ (행렬식의 값이 최대 $O(m)$비트의 크기까지 커질 수 있기 때문에 $m$을 한 번 더 곱해주었다) 개의 프로세서가 있을 때 $O(\log^2 n)$의 시간에 수행 가능하다. 따라서 알고리즘은 $O(n^{3.496}m)$개의 프로세서가 있을 때 $O(\log^2 n)$의 시간에 작동한다.

이 알고리즘은 그래프에 완전 매칭이 존재할 때 $\frac{1}{2}$ 이상의 확률로 하나의 완전 매칭을 찾는다. 이 알고리즘이 틀렸다면, 간선에 가중치를 무작위로 부여하는 과정에서 가중치의 합을 최소로 하는 완전 매칭이 유일하지 않았기 때문이었을 것이다. 알고리즘을 다시 시행했을 때 완전 매칭을 찾을 확률은 독립적이기 때문에, 이 알고리즘은 $\mathcal{RNC}$ 알고리즘이다.

# 결론

분산 컴퓨팅 모델인 PRAM 모델에서 완전 매칭을 찾는 $\mathcal{RNC}$ 알고리즘에 대해 알아보았다. 이 알고리즘을 응용하면 일반적인 그래프에 대해 maximum matching을 찾는 (완전 매칭이 아닐 때에도!) $\mathcal{RNC}$ Las Vegas 알고리즘으로 발전시킬 수도 있다. 이에 대해서도 추후 서술할 예정이다.

# 참고문헌

1. Rajeev Motwani and Prabhakar Raghavan, Randomized Algorithms, 1995
2. S. J. Berkowitz, On computing the determinant in small parallel time using a small number of processors. Inf. Process, 1984
