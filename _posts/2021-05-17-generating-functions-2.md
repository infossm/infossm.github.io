---
layout: post
title: 다양한 생성함수와 그 응용 (2)
date: 2021-05-17 15:21:11
author: youngyojun
tags:
 - Mathematics
 - Algorithm

---

# 개요

​	본문은 아래의 포스트와 내용이 이어지며, 다음 포스트의 내용을 부가적인 설명 없이 서술한다.

* [다양한 생성함수와 그 응용 (1)](https://www.secmem.org/blog/2021/04/18/generating-functions-1/)



​	이번에는 생성함수를 어떻게 실제 PS 문제에 적용할 수 있는지 살펴본다.

# 다변수 생성함수

## 이항계수의 생성함수

​	이항계수 $\displaystyle \binom{n}{k}$의 OGF를 유도하자.

먼저, 이항계수와 같은 값을 가지는 함수 $f(n, k)$를 귀납적으로 정의하자. 다음은 이항계수의 일반적인 귀납적 정의이다:

> 다음을 만족하는 함수 $f : \mathbb{Z}_{\ge 0}^2 \rightarrow \mathbb{R}$은 모든 음이 아닌 정수 $n$, $k$에 대하여, $\displaystyle f(n, k) = \binom{n}{k}$를 만족한다.
>
> -   모든 $n \ge 0$에 대하여, $f(n, 0) = 1$.
>
>
> -   모든 $n \ge 1$에 대하여, $f(0, n) = 0$.
>
> -   모든 $n \ge 1$, $k \ge 1$에 대하여, $f(n, k) = f(n-1, k) + f(n-1, k-1)$.

이변수 함수 $f(n, k)$의 OGF $F(x, y)$는 다음과 같이 정의하는 것이 자연스러울 것이다:

$$ F(x, y) = \sum _{n = 0}^{\infty} \sum _{k = 0}^{\infty} f(n, k) x^n y^k $$

위의 식에서 점화식 $f(n, k) = f(n-1, k) + f(n-1, k-1)$를 적용하면, 다음과 같은 식을 얻는다:

$$ \sum _{n = 1}^{\infty} \sum _{k = 1}^{\infty} f(n, k) x^n y^k = x \sum _{n = 1}^{\infty} \sum _{k = 1}^{\infty} f(n-1, k) x^{n-1} y^k + xy \sum _{n = 1}^{\infty} f(n-1, k-1) x^{n-1} y^{k-1} $$

이를 $F(x, y)$ 정의에 대입하자.

$$ \sum _{n = 1}^{\infty} \sum _{k = 1}^{\infty} f(n, k) x^n y^k = F(x, y) - \sum _{n = 1}^{\infty} f(n, 0) x^n - \sum _{k = 1}^{\infty} f(0, k) y^k - f(0, 0) = F(x, y) - \sum _{n = 0}^{\infty} x^n $$

$$ x \sum _{n = 1}^{\infty} \sum _{k = 1}^{\infty} f(n-1, k) x^{n-1} y^k + xy \sum _{n = 1}^{\infty} f(n-1, k-1) x^{n-1} y^{k-1}$$

$$ = x \left( F(x, y) - \sum _{n = 0}^{\infty} f(n, 0) x^n \right) + xy F(x, y) = x \left( F(x, y) - \sum _{n = 0}^{\infty} x^n \right) + xy F(x, y) $$

따라서, $F(x, y)$에 관한 중요한 식 하나를 알아낼 수 있다:

$$ F(x, y) - \sum _{n = 0}^{\infty} x^n = x \left( F(x, y) - \sum _{n = 0}^{\infty} x^n \right) + xy F(x, y) $$

$$ F(x, y) - \frac{1}{1-x} = x \left( F(x, y) - \frac{1}{1-x} \right) + xy F(x, y) $$

$$ F(x, y) = \frac{ 1 }{ 1 - x - xy } $$

​	정리하면, 이항계수 $\displaystyle \binom{n}{k}$의 OGF는 $\displaystyle \frac{1}{1-x-xy}$이다.

여기서, 식을 약간 조작하면 어렵지 않게 $F(x, y)$를 무한 급수 형태로 표현할 수 있다.

$$ F(x, y) = \frac{1}{1-x-xy} = \frac{1}{1 - \left( x (y+1) \right)} = \sum _{n = 0}^{\infty} \left( x (y+1) \right)^n = \sum _{n = 0}^{\infty} x^n (y+1)^n $$

​	이로부터 다음과 같은 정리를 증명할 수 있다. 이항 정리를 증명하는 새로운 방법을 찾았다고 말할 수 있다.

### 이항 정리의 새로운 증명

> 모든 양의 정수 $n$에 대하여, 다음은 일변수 실수 항등식이다.
>
> $$ (y + 1)^n = \sum _{k = 0}^{n} \binom{n}{k} y^k = \binom{n}{0} y^0 + \binom{n}{1} y^1 + \cdots + \binom{n}{n-1} y^{n-1} + \binom{n}{n} y^n $$

​	$F(x, y)$에서 $x^n$의 계수를 살펴보자.

$\displaystyle F(x, y) = \sum _{k = 0}^{\infty} (y+1)^k x^k$ 이므로, $x^n$의 계수는 $(y+1)^n$이다.

또한, 정의에 따르면 $\displaystyle F(x, y) = \sum _{n = 0}^{\infty} \sum _{k = 0}^{\infty} f(n, k) x^n y^k = \sum _{n = 0}^{\infty} \left( \sum _{k = 0}^{\infty} f(n, k) y^k \right) x^n$ 이므로, $x^n$의 계수는 $\displaystyle \sum _{k = 0}^{\infty} f(n, k) y^k = \sum _{k = 0}^{\infty} \binom{n}{k} y^k$ 와도 같다.

따라서, 주어진 식은 항상 성립한다.

​	이제, $F(x, y)$를 다음과 같이 변형해보자:

$$ F(x, y) = \frac{1}{1-x-xy} = \frac{1}{(1-x)-xy} = \frac{ \frac{1}{1-x} }{ 1 - \frac{x}{1-x} \cdot y } $$

$$ = \frac{1}{1-x} \left( 1 + \frac{x}{1-x} \cdot y + \left( \frac{x}{1-x} \right)^2 \cdot y^2 + \cdots \right) = \frac{1}{1-x} \sum _{n = 0}^{\infty} \left( \frac{x}{1-x} \right)^n y^n $$

​	여기서, $F(x, y)$의 $y^n$의 계수가 $\displaystyle \frac{x ^n}{(1-x) ^{n+1}}$이라는 사실로부터 다음 정리를 증명할 수 있다.

### 이항계수의 OGF의 전개식

> 임의의 $x \in \mathbb{R}$와 $k \in \mathbb{Z}_{\ge 0}$에 대하여 다음이 성립한다.
>
> $$ \sum _{n = 0}^{\infty} \binom{n}{k} x^n = \frac{ x^k }{ (1 - x)^{k+1} } $$

​	이 정리는 다음과 같이 해석할 수 있다. $\displaystyle \frac{1}{(1-x)^{k+1}}$을 다항식의 형태로 전개하면 $x^{n-k}$의 계수가 $\displaystyle \binom{n}{k}$와 일치한다는 사실을 함의한다.

​	따라서, $\displaystyle \frac{1}{(1-x)^k}$의 다항 전개의 $x^n$의 계수는 $\displaystyle \binom{n+k-1}{k-1}$이다. 이는 추후에, 고정된 $k$에 대하여 $\displaystyle \binom{n}{k}$의 합에 관한 식을 서술해야 할 때 유용하게 사용할 것이다.

## 카탈랑 수열의 닫힌 형태

​	이전 포스트에서 카탈랑 수열 $\displaystyle \left\\{ c_n \right\\} _{n \in \mathbb{Z} _{\ge 0}}$의 OGF $C(x)$가 다음과 같음을 보였다:

$$ C(x) = \frac{ 1 - \sqrt{ 1 - 4x } }{ 2x } $$

하지만, $C(x)$는 카탈랑 수 $c_n$의 닫힌 형태(Closed form expression)를 직접적으로 알려주지는 않는다. 이번에는 카탈랑 수의 닫힌 형태를 유도하고자 한다.

​	위에서 살펴본 이항계수 $\displaystyle \binom{n}{k}$는 복소수 범위까지 확장할 수 있다. 다음 정리는 그러한 확장 방법을 알려준다.

### 알려진 정의 (일반화된 이항계수)

> 복소수 $r \in \mathbb{C}$와 음이 아닌 정수 $n \in \mathbb{Z} _{\ge 0}$에 대하여, (확장)이항계수 $\displaystyle \binom{r}{n}$을 다음과 같이 정의한다:
>
> $$ \binom{r}{n} = \frac{ r (r-1) \cdots (r - (n-1)) }{ n! } $$

​	이제부터 언급하는 이항계수는 모두 일반화된 이항계수를 의미한다. 따라서, $\displaystyle \binom{n}{k}$에서 $n$가 음이 아닌 정수일 필요성이 사라졌음에 유의하자.

### 보조정리

> 임의의 복소수 $r \in \mathbb{C}$에 대하여 다음이 성립한다.
>
> $$ (1+x)^r = \sum _{n = 0}^{\infty} \binom{r}{n} x^n $$

​	엄밀하지는 않지만 가장 직관적인 증명은 이러하다. $(1+x)^r$의 다항 전개가 존재한다고 가정할 때, 양변의 $x^n$의 계수가 서로 같은지 살펴보자.

먼저, 좌변을 $n$번 미분한 식의 상수항은 다음과 같다:

$$ \left. \frac{ \mathrm{d}^n }{ \mathrm{d} x^n } (1+x)^r \right\rvert _{x = 0} = r (r-1) \cdots (r - (n-1)) = n! \binom{r}{n} $$

우변을 $n$번 미분한 식의 상수항은 다음과 같다:

$$ \left. \frac{ \mathrm{d}^n }{ \mathrm{d} x^n } \sum _{n = 0}^{\infty} \binom{r}{n} x^n \right\rvert _{x = 0} = n! \binom{r}{n} $$

따라서, 주어진 식은 항등식이다.

​	이제, 보조정리를 이용하면, $\displaystyle \sqrt{1 - 4x}$를 다음과 같이 표현할 수 있다:

$$ \sqrt{1-4x} = \left( 1 - 4x \right)^\frac{1}{2} = \sum _{n = 0}^{\infty} \binom{ \frac{1}{2} }{ n } \left( -4x \right)^n $$

$$ = \sum _{n = 0}^{\infty} \left( \frac{1}{2} \times \frac{-1}{2} \times \frac{-3}{2} \times \cdots \times \frac{-(2n - 3)}{2} \right) \cdot \frac{1}{n!} \times (-4)^n x^n $$

$$ = 1 + \sum _{n = 1}^{\infty} \frac{ (-1)^{n-1} \left( 1 \times 3 \times \cdots \times (2n-3) \right) }{ 2^n } \cdot \frac{(-4)^n}{n!} x^n $$

$$ = 1 + \sum _{n = 1}^{\infty} \frac{ -2^n \cdot (2n-2)! }{ 2^{n-1} \cdot (n-1)! } \cdot \frac{1}{n!} x^n $$

$$ = 1 + \sum _{n = 1}^{\infty} -\frac{2}{n} \cdot \frac{(2n-2)!}{(n-1)!(n-1)!} x^n = 1 + \sum _{n = 1}^{\infty} - \frac{2}{n} \binom{2n-2}{n-1} x^n $$

즉, $C(x)$는 다음과 같이 무한 급수 형태로 나타낼 수 있다:

$$ C(x) = \frac{ 1 - \sqrt{ 1 - 4x } }{ 2x } = \frac{1}{2x} \left( 1 - \left( 1 + \sum _{n = 1}^{\infty} -\frac{2}{n} \binom{2n-2}{n-1} x^n \right) \right) = \sum _{n = 1}^{\infty} \frac{ \binom{2n-2}{n-1} }{n} x^{n-1} = \sum _{n = 0}^{\infty} \frac{1}{n+1} \binom{2n}{n} x^n $$

​	정리하면, 아주 긴 여정 끝에 우리는 카탈랑 수의 닫힌 형태가 $\displaystyle c _n = \frac{1}{n+1} \binom{2n}{n}$임을 알아내었다. 생성함수로 수열의 새로운 성질을 알아낼 수 있다니, 놀랍지 않은가!

# 순열과 관련된 생성함수

​	어떤 순열 $p = \left( p _1, \cdots, p _n \right)$가 주어졌다면, 정점 $i$에서 $p _i$로 가는 간선으로 이루어진 그래프는 여러 개의 Disjoint한 사이클이 된다는 사실은 잘 알려져 있다. 또한, 역으로, Disjoint한 사이클을 하나의 순열 $p$로 표현할 수도 있다.

​	이번에는 사이클의 개수를 세는 문제를 순열로 해석한 후, 생성함수를 적용하여 해결하는 방법에 대하여 알아본다.

## $k$개의 사이클로 이루어진 길이 $n$의 순열의 개수 세기

​	그러한 순열의 개수를 [부호 없는 제1종 스털링 수](https://ko.wikipedia.org/wiki/스털링_수#제1종_스털링_수)라고 부른다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Stirling_number_of_the_first_kind_s%284%2C2%29.svg/2560px-Stirling_number_of_the_first_kind_s%284%2C2%29.svg.png)

<p style="text-align: center;"><b>그림 1: $n=4$, $k=2$일 때 가능한 모든 그래프</b></p>

<p style="text-align: center;">$n=4$, $k=2$일 때의 부호 없는 제1종 스털링 수는 $11$이다.</p>

​	먼저, 하나의 사이클로 이루어진 길이 $n$의 순열의 개수를 $c _n$이라고 하자. 어렵지 않게, $c _n = (n-1)!$임을 보일 수 있다.

이제, 수열 $\displaystyle \left\\{ c _n \right\\} _{n \in \mathbb{Z} _{\ge 0}}$의 EGF를 $C(x)$라고 하자. 정의에 의하여 다음이 성립함에 유의하자:

$$ C(x) = \sum _{n = 0}^{\infty} \frac{ c _n }{ n! } x^n $$

​	다시 원래 문제로 돌아오자. $k$를 고정하고, $n$가 주어졌을 때 그러한 순열의 개수를 $f _n$라고 하자. $n < k$에 대해서는 필요에 따라 무시해도 좋고, $f _n = 0$으로 생각해도 무방하다.

수열 $f$의 EGF를 $F(x)$라고 하자. 이제 우리는 $\displaystyle F(x) = \frac{1}{k!} C(x)^k$임을 보임으로써 문제를 해결할 것이다.

​	먼저, $n$개의 정점으로 이루어진 $k$개의 사이클을 생각하자. 각각의 사이클에 $1$부터 $k$까지 번호를 부여하고, $i$번 사이클의 크기를 $a _i$라고 하자. 자명하게도, $\displaystyle a _1 + \cdots + a _k = \sum _{i = 1}^{k} a _i = n$라야 한다.

$a _i$개의 정점으로 하나의 사이클을 만드는 방법은 총 $\displaystyle c _{a _i}$가지가 존재한다. 또한, 독립적으로 번호가 부여된 $k$개의 사이클을 하나의 순열로 합치는 방법은 총 $\displaystyle \frac{ n! }{ a _1 ! a _2 ! \cdots a _k ! }$가지가 있다.

따라서, 다음 식을 얻는다:

$$ f _n = \frac{ n! }{ k! } \sum _{ a _1 + a _2 + \cdots + a _k = n } \frac{ c _{a _1} c _{a _2} \cdots c _{a _k} }{ a _1 ! a _2 ! \cdots a _k ! } $$

원래 문제에서는 사이클의 순서가 중요하지 않으므로, $k!$로 나누어주어야 함에 유의한다.

​	이전 포스트에서 "7. 분할" 성질에 따르면, $\displaystyle \sum _{ a _1 + a _2 + \cdots + a _k = n } \frac{ c _{a _1} c _{a _2} \cdots c _{a _k} }{ a _1 ! a _2 ! \cdots a _k ! }$는 $\displaystyle C(x)^k$의 $x^n$의 계수와 일치한다.

$F(x)$는 EGF이므로, 정리하면 $\displaystyle F(x) = \frac{1}{k!} C(x)^k$이다.

​	만약에 $k$와 $N$가 주어졌을 때, $n = 1, 2, \cdots, N$에서 답을 모두 구해야 한다고 하자. $\displaystyle F(x) = \frac{1}{k!} C(x)^k$를 전개하면 문제를 쉽게 해결할 수 있는데

* $C(x)$의 처음 $O(N)$개의 계수를 $O(N)$에 계산할 수 있고
* $\displaystyle C(x)^k = e^{ k \ln C(x) }$임을 이용하면, $\displaystyle C(x)^k$의 처음 $O(N)$개의 계수를 $O \left( N \lg N \right)$에 계산할 수 있으므로

전체 문제를 $O \left( N \lg N \right)$이라는 효율적인 시간 복잡도로 해결할 수 있다.

## 사이클의 길이가 집합 $S$의 원소인 길이 $n$의 순열의 개수 세기

​	이전에는 사이클의 개수에 제약이 있었다면, 이번에는 사이클의 길이에 제약이 생겼다.

​	유사하게 하나의 사이클로 이루어진 길이 $n$의 순열의 개수를 $c _n$이라고 하자. 다만, $n \not\in S$라면, $c _n = 0$로 정의하자. 이로부터 우리는 사이클의 길이가 집합 $S$의 원소임을 보장할 수 있다.

동일한 논리를 적용하여, $\displaystyle F(x) = \frac{1}{k!} C(x)^k$를 얻자. 모든 $k = 1, 2, \cdots, n$에 대하여 각각 답을 계산한 뒤 더하면, 이것이 우리가 원하는 답이 된다.

즉, $\displaystyle \sum _{k = 1}^{n} \frac{1}{k!} C(x)^k$의 $x^n$의 계수를 계산하면 된다. 이 값은 $\displaystyle \sum _{k = 0}^{\infty} \frac{1}{k!} C(x)^k = e^{ C(x) }$의 $x^n$의 계수와 동일하다.

​	따라서, 집합 $S$와 $n$가 주어졌을 때, 문제의 답을 $O \left( \left\lvert S \right\rvert + n \lg n \right)$에 효율적으로 알아낼 수 있다.

## 길이 $n$의 순열의 사이클의 개수의 기댓값

​	길이 $n$의 모든 순열이 생성하는 사이클의 개수의 합을 $g _n$라고 하자. 이 값을 $n!$으로 나누면 우리가 원하는 답이 된다.

수열 $g$의 EGF를 $G(x)$라고 하자. 편의를 위하여, 다항 전개가 가능한 식 $f(x)$에 대하여, $f(x)$의 $x^n$의 계수를 $\displaystyle \left[ x^n \right] f(x)$라고 표기하자. 이전과 동일한 논리를 사용하면, 다음 등식을 얻는다:

$$ \left[ x^n \right] G(x) = \left[ x^n \right] \sum _{k = 1}^{n} k \cdot \frac{ C(x)^k }{ k! } = \left[ x^n \right] \sum _{k = 1}^{\infty} \frac{1}{(k-1)!} C(x)^k $$

$$ = \left[ x^n \right] C(x) \left( \sum _{k = 0}^{\infty} \frac{ C(x)^k }{ k! } \right) = \left[ x^n \right] C(x) e^{ C(x) } $$

$c _n = (n-1)!$라는 사실로부터, 다음을 알 수 있다:

$$ C(x) = \sum _{k = 1}^{\infty} \frac{ c _k }{ k! } x^k = \sum _{k = 1}^{\infty} \frac{ (k-1)! }{k!} x^k = \sum _{k = 1}^{\infty} \frac{ x^k }{k} = - \ln \left( 1 - x \right) $$

따라서, $\displaystyle C(x) e^{ C(x) } = - \frac{ \ln (1-x) }{ 1-x }$이다.

$\ln (1+x)$의 매클로린 급수는 $\displaystyle x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots$이므로, $n \ge 1$라면 $\displaystyle \left[ x^n \right] \left( - \ln (1-x) \right) = \frac{1}{n}$이다.

고로, "8. 구간 합" 성질을 적용하면 다음을 알 수 있다:

$$ \left[ x^n \right] \frac{ - \ln(1-x) }{ 1-x } = 1 + \frac{1}{2} + \cdots + \frac{1}{n} = \sum _{k=1}^{n} \frac{1}{k} $$

따라서, $\displaystyle \frac{g _n}{n!} = \left[ x^n \right] G(x) = \left[ x^n \right] \frac{ - \ln (1-x) }{ 1-x } = \sum _{k = 1}^{n} \frac{1}{k}$이다.

​	정리하면, 길이 $n$의 순열이 생성하는 사이클의 개수의 기댓값은 $n$번째 [조화수](https://ko.wikipedia.org/wiki/%EC%A1%B0%ED%99%94%EC%88%98) $\displaystyle 1 + \frac{1}{2} + \cdots + \frac{1}{n}$이다.

# 결론

​	이항계수의 이변수 생성함수를 유도하고, 이항계수 자체를 복소수 범위까지 확장시켰다. 이 과정에서 이항 정리의 새로운 증명을 알아보았고, $\displaystyle \frac{1}{(1-x)^k}$의 다항 전개가 이항계수 $\displaystyle \binom{n+k-1}{k-1}$와 밀접한 연관이 있음을 관찰하였다.

​	이항계수의 생성함수는 카탈랑 수의 닫힌 형태를 유도하는 데에도 큰 도움을 주었다. 보조정리를 이용하여, 카탈랑 수열의 생성함수를 이항계수의 전개 형태로 변형하였고, 이로부터 닫힌 형태를 직접적으로 밝힐 수 있다.

​	순열과 관련된 조합론 문제는 생성함수 EGF를 이용하면 유용하다는 점을 다양한 예시를 통하여 경험하였다. 일반적인 DP로는 $O \left( n^2 \right)$이나 혹은 다항 시간으로 해결하기 어려운 문제를, 생성함수를 이용하면 $O \left( n \lg n \right)$의 빠른 시간 복잡도로 해결할 수 있다.

​	다음 포스트에서는, 일반적인 그래프에 관한 조합론 문제에 어떻게 생성함수를 적용할 수 있는지와, 미분 방정식 테크닉에 관하여 다룰 것이다.
