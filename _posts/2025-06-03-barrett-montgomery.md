---
layout: post
title: "Barrett, Montgomery Reduction을 이용한 모듈러 연산의 고속화"
date: 2025-06-03
author: jinhan814
tags: [algorithm, mathematics, problem-solving]
---

## 1. Introduction

현대 CPU에서 나눗셈과 모듈러 연산은 연산 비용이 상당히 큰 편입니다. Agner Fog의 [instruction tables](https://www.agner.org/optimize/instruction_tables.pdf)에 따르면, Intel Skylake 아키텍처 기준으로 `ADD`, `SUB`의 latency는 1 cycle, `MUL`은 3~4 cycle인데 반해, `DIV`는 32비트는 26 cycle, 64비트는 최소 35 cycle, 최대 88 cycle까지 소요됩니다.

그런데 실제로 코드를 작성해보면 나눗셈 연산은 생각보다 빠르게 동작하는 경우가 많습니다. 이는 현대 컴파일러가 <code>x / c</code>, <code>x % c</code>에서 나누는 수 <code>c</code>가 compile-time const일 때 해당 연산을 곱셈(<code>*</code>)과 시프트(<code>>></code>)로 변환해 최적화해주기 때문입니다. 하지만 <code>c</code>가 run-time에 주어지는 경우는 <code>c</code>를 일반 변수로 선언해야 하기 때문에 실제로 `DIV` 명령어가 호출되어 성능 저하가 발생합니다.

또한 대부분의 아키텍처에서 SIMD(Single Instruction Multiple Data) 명령어 집합은 정수 나눗셈(<code>/</code>)과 모듈러 연산(<code>%</code>)을 지원하지 않기에 Vectorization을 이용한 고속화가 어렵습니다.

이번 글에서는 정수 나눗셈(<code>/</code>), 모듈러 연산(<code>%</code>)을 시프트(<code>>></code>), 덧셈(<code>+</code>), 곱셈(<code>*</code>)으로 대체하는 Barrett Reduction, Montgomery Reduction 기법을 소개하고, 이를 통해 모듈러가 compile-time에 주어지지 않는 상황이나 SIMD 기반의 병렬 연산을 구현할 때 나눗셈, 모듈러 연산의 성능을 개선하는 방법을 설명합니다.

## 2. Barrett Reduction

Barrett Reduction은 두 정수 $n, m(n \geq 0, m \geq 2)$이 주어질 때, 충분히 큰 $k$에 대해

$$
\left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor
$$

가 성립함을 이용하는 정수 나눗셈 최적화 기법입니다.

여기서 $\left\lceil \frac{2^k}{m} \right\rceil$은 미리 계산해둔 뒤 곱할 수 있고, $2^k$로 나누는 과정은 시프트 연산으로 처리할 수 있습니다.

### 2.1 Basic Idea

두 정수 $a, b$가 $a \geq 0$, $b \geq 2$를 만족한다면, $0 \leq e < \frac{1}{b}$인 실수 $e$에 대해 $\lfloor \frac{a}{b} + e \rfloor = \lfloor \frac{a}{b} \rfloor$가 성립합니다.

증명은 다음과 같습니다.

$$
\begin{align*}
a &= \left\lfloor \frac{a}{b} \right\rfloor b + r = qb + r \\
  &\Rightarrow 0 \leq r \leq b - 1 \\
  &\Rightarrow 0 \leq \frac{r}{b} \leq 1 - \frac{1}{b} \\
  &\Rightarrow 0 \leq \frac{r}{b} + e < 1 \\
  &\Rightarrow \left\lfloor \frac{a}{b} + e \right\rfloor = \left\lfloor q + \left( \frac{r}{b} + e \right) \right\rfloor = q
\end{align*}
$$

이제 $n \geq 0$, $m \geq 2$인 두 정수 $n, m$과 $2^{\lfloor \log_2 (m - 1) \rfloor} \cdot \max(2n, m) \leq 2^k$인 정수 $k$에 대해

$$
\begin{align*}
s &= \left\lfloor \log_2(m - 1) \right\rfloor = \left\lceil \log_2 m \right\rceil - 1 &&(n \leq 2^{k - s - 1}, \; 2^s < m \leq 2^{s + 1}) \\
x &= \left\lceil \frac{2^k}{m} \right\rceil \\
r &= x m - 2^k &&(0 \leq r < m) \\
e &= \frac{nx}{2^k} - \frac{n}{m}
\end{align*}
$$

를 정의합시다.

$e$의 정의를 전개하면,

$$
\begin{align*}
e &= \frac{nx}{2^k} - \frac{n}{m} = \frac{n}{m}(\frac{xm}{2^k} - \frac{2^k}{2^k}) = \frac{nr}{m2^k} && (0 \leq e) \\
  &\Rightarrow e - \frac{1}{m} = \frac{1}{m}(\frac{nr}{2^k} - \frac{2^k}{2^k}) < 0 &&(\because nr < nm \leq 2^{k-s-1} \cdot 2^{s+1} = 2^k) \\
  &\Rightarrow 0 \leq e < \frac{1}{m}
\end{align*}
$$

이 성립합니다.

따라서 $\left\lfloor \frac{n}{m} + e \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor$에서 $0 \leq e < \frac{1}{m}$이니 $\left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor = \left\lfloor \frac{n}{m} \right\rfloor$ 이 성립합니다.

### 2.2 Integer Division using Barrett Reduction

$2^{\lfloor \log_2 (m - 1) \rfloor} \cdot \max(2n, m) \leq 2^k$를 만족하는 정수 $k$를 이용하면, $x = \left\lceil \frac{2^k}{m} \right\rceil$를 미리 계산한 뒤 <code>n / m</code>을 <code>(n * x) >> k</code>로 대체하며 Barrett Reduction을 구현할 수 있습니다.

이때 $k$를 너무 크게 잡으면 <code>n * x</code>에서 overflow가 발생할 수 있으니, 가능한 작은 $k$를 이용하는 게 좋습니다. $k$의 최솟값은 $\lfloor \log_2 (m - 1) \rfloor + \lceil \log_2 (\max(2n, m)) \rceil$입니다. 따라서 $n, m$의 범위가 주어지면 해당 값을 계산해 $k$의 하한을 구할 수 있습니다.

한편 $k$를 최대한 작게 선택하더라도 $nx$의 범위가 커질 수 있음에 주의해야 합니다. 예를 들어 $n = m = 2^{32} - 1$인 경우, $k \geq 31 + 33$에서 $k = 64$를 선택하면 $x = \lceil \frac{2^{64}}{2^{32} - 1} \rceil = 2^{32} + 2$가 되므로 $nx = (2^{32} - 1)(2^{32} + 2) = 2^{64} + 2^{32} - 2$가 64비트 정수 자료형의 범위를 초과하게 됩니다. 따라서 $n, m$의 범위를 고려하여 $nx$의 최댓값을 계산하고, 이에 맞는 충분한 자료형을 선택하는 것이 중요합니다.

구현 코드는 다음과 같습니다.

```cpp
using u128 = unsigned __int128;
using u64 = unsigned long long;
using u32 = unsigned int;

struct intdiv_barrett {
	intdiv_barrett() {}
	intdiv_barrett(u32 m) : x(((u128(1) << 64) + m - 1) / m) {}
	u32 div(u32 n) const {
		return u128(n) * x >> 64;
	}
private:
	u64 x;
};
```

해당 코드는 $2 \leq m < 2^{32}$인 고정된 $m$에 대해 $0 \leq n < 2^{32}$인 정수 $n$이 주어질 때 $n / m$을 $k = 64$인 Barrett Reduction을 이용해 빠르게 계산합니다.

**Note.**

- $2 \leq m < 2^{31}$, $0 \leq n < 2^{31}$이라면 $k = 62$와 128비트 정수 곱셈을 이용하면 됩니다.
- $2 \leq m < 2^{63}$, $0 \leq n < 2^{63}$이라면 $k = 126$과 256비트 정수 곱셈을 이용하면 됩니다.
- $2 \leq m < 2^{64}$, $0 \leq n < 2^{64}$라면 $k = 128$과 256비트 정수 곱셈을 이용하면 됩니다.
- $k = 64$인 경우 $x = \lceil \frac{2^{64}}{m} \rceil$을 <code>u64(-1) / m + 1</code>로 구할 수 있습니다. 대표적으로 ac-library가 이 방식을 사용합니다. [(참고)](https://github.com/atcoder/ac-library/blob/master/atcoder/internal_math.hpp)
- GNU 계열 컴파일러(GCC/Clang)와 달리 Microsoft Visual C++(MSVC)는 <code>__int128</code> 자료형을 지원하지 않기에 128비트 정수 곱셈을 직접 구현해야 합니다.

### 2.3 Modular Multiplication in $\mathbb{Z}_m$ using Barrett Reduction

두 정수 $0 \leq a, b < m$에 대해 <code>(a * b) % m</code>는 <code>(a * b) - ((a * b) / m) * m</code>에서 <code>(a * b) / m</code>에 Barrett Reduction을 적용해 빠르게 구할 수 있습니다.

이 경우 $0 \leq ab < m^2$이 성립하니 $k$의 하한은 $\lfloor \log_2(m - 1) \rfloor + \lceil \log_2(m^2 - 1) \rceil + 1$이고, $nx$의 상한은 $nx < m^2(\frac{2^k}{m} + 1) < m \cdot 2^k + m^2$입니다.

대부분의 ps/cp 환경에서 $m$은 $2 \leq m < 2^{31}$ 범위이니 $k = 93$을 선택할 수 있고, 이때 $nx < m \cdot 2^k + m^2 \leq (2^{31} - 1) \cdot 2^{93} + (2^{31} - 1)^2 < 2^{124}$이니 128비트 정수 자료형을 이용해 <code>n * x</code>를 계산하며 Barrett Reduction을 구현할 수 있습니다.

구현 코드는 다음과 같습니다.

```cpp
using u128 = unsigned __int128;
using u64 = unsigned long long;
using u32 = unsigned int;

struct modmul_barrett {
	modmul_barrett() {}
	modmul_barrett(u32 m) : m(m), x(((u128(1) << 93) + m - 1) / m) {}
	u64 div(u64 n) const {
		return n * x >> 93;
	}
	u32 mul(u32 a, u32 b) const {
		u64 n = u64(a) * b;
		return n - div(n) * m;
	}
private:
	u32 m;
	u128 x;
};
```

해당 코드는 $2 \leq m < 2^{31}$인 정수 $m$에 대해 $0 \leq a, b < m$인 두 정수 $a, b$가 주어질 때 $ab \bmod m = ab - \lfloor \frac{ab}{m} \rfloor m$을 $k = 93$인 Barrett Reduction을 이용해 빠르게 계산합니다.

사용 예시는 다음과 같습니다. [(코드)](http://boj.kr/233deb0addff442ebdd782ac500d9298)

**Note.**

- $2 \leq m < 2^{32}$라면 $k = 96$과 128비트 정수 곱셈을 이용하면 됩니다.
- $2 \leq m < 2^{63}$이라면 $k = 189$와 256비트 정수 곱셈을 이용하면 됩니다.
- $2 \leq m < 2^{64}$라면 $k = 192$와 256비트 정수 곱셈을 이용하면 됩니다.
- $k$를 $m$에 대한 실제 하한 값로 설정하면 $2 \leq m < 2^{31}$에 대해 $2 \leq \left\lceil \frac{2^k}{m} \right\rceil < 2^{63}$이 성립해 <code>x</code>를 64비트 정수 자료형으로 저장할 수 있습니다. [(코드)](http://boj.kr/d4b035cb317b4d579c10656d4a5bf9ec)
- $k$가 상수가 아니라면 <code>>></code>에서 추가적인 연산이 생기기 때문에 성능 저하가 있을 수 있습니다. [(참고)](https://godbolt.org/z/vjnnM1eoT)

## 3. Montgomery Reduction

Montgomery Reduction은 $m > 2$인 홀수 정수 $m$이 주어질 때, $m \leq r$이면서 $\gcd(m, r) = 1$인 정수 $r$과 $0 \leq n < rm$인 정수 $n$에 대해

$$
n \cdot r^{-1} \equiv \frac{n - (n \cdot m' \bmod r) \cdot m}{r} \bmod m
$$

이 성립함을 이용하는 모듈러 연산 최적화 기법입니다.

여기서 $r^{-1}, m'$은 $r\cdot r^{-1} + m \cdot m' = 1$을 만족하는 정수이고, $n \bmod m$는 $n = qm + r$, $0 \leq r < m$을 만족하는 정수 $r$입니다.

### 3.1 Basic Idea

$\gcd(m, r) = 1$인 정수 $m, r$은 Bezout 항등식에 의해 $r \cdot r^{-1} + m \cdot m' = 1$이고 $0 < m' < r$인 두 정수 $r^{-1}, m'$이 존재합니다.

$$
\begin{align*}
\overline{n} &= n \cdot r \bmod m \\
f(n) &= n \cdot r^{-1} \bmod m
\end{align*}
$$

정수 $n$에 대해 $n$의 Montgomery Form $\overline{n}$과 Montgomery Reduction $f(n)$을 위와 같이 정의합시다.

Montgomery Reduction을 이용하면 세 정수 $a, b, a \cdot b$의 Montgomery Form 간의 관계를 $\overline{a \cdot b} = \overline{a} \cdot \overline{b} \cdot r^{-1} \bmod m = f(\overline{a} \cdot \overline{b})$로 나타낼 수 있습니다. 또한 $f(\overline{n}) = n \bmod m$과 $f(n \cdot r^2) = \overline{n}$을 이용하면 $n \bmod m \leftrightarrow \overline{n}$ 변환을 표현할 수 있습니다.

$$a, b \; \longrightarrow \; \overline{a}, \overline{b} \; \longrightarrow \; f(\overline{a} \cdot \overline{b}) = \overline{a \cdot b} \; \longrightarrow \; a \cdot b \bmod m$$

따라서 Montgomery Reduction $f$를 빠르게 수행할 수 있다면, $a, b$의 Montgomery Form을 구한 뒤 $f(\overline{a} \cdot \overline{b})$를 구하고 다시 역변환을 하며 $a \cdot b \bmod m$을 빠르게 구할 수 있습니다.

### 3.2 Montgomery Reduction

이제 Montgomery Reduction을 빠르게 수행하는 방법을 알아보겠습니다.

$r \cdot r^{-1} + m \cdot m' = 1$에서 임의의 $k \in \mathbb{Z}$에 대해

$$
\begin{align*}
n \cdot r^{-1} &= n \cdot \frac{r \cdot r^{-1}}{r} \\
               &= n \cdot \frac{1 - m \cdot m'}{r} \\
			   &= \frac{n - n \cdot m \cdot m'}{r} \\
			   &\equiv \frac{n - n \cdot m \cdot m' + k \cdot r \cdot m}{r} \; \bmod m \\
			   &\equiv \frac{n - (n \cdot m' - k \cdot r) \cdot m}{r} \; \bmod m \\
			   &\equiv \frac{n - (n \cdot m' \bmod r) \cdot m}{r} \; \bmod m
\end{align*}
$$
이 성립하니, 이를 이용하면 $x = n \cdot m' \bmod r$과 $y = x \cdot m$에 대해 $n \cdot r^{-1} \bmod m = \frac{n - y}{r} \bmod m$으로 $f(n)$을 표현할 수 있습니다.

이제 $0 \leq n < rm$을 가정하면 $\frac{n}{r} < m$이고, $x < r$에서 $\frac{y}{r} = \frac{x \cdot m}{r} < \frac{r \cdot m}{r} = m$이니 $-m < \frac{n - y}{r} < m$이 성립합니다. 따라서 $n < y$라면 $\frac{n - y}{r} + m$, $n \geq y$라면 $\frac{n - y}{r}$을 이용해 $f(n) = \frac{n - y}{r} \bmod m$을 구할 수 있습니다.

또한 $m$이 홀수라는 가정을 추가하면, $r$을 $m$보다 큰 $2$의 거듭제곱 $2^w$로 선택할 수 있습니다. 이때 $r$을 $2^w$ 꼴로 선택하면 $r$에 대한 모듈러 연산과 정수 나눗셈을 비트 연산으로 대체할 수 있습니다.

$$
\begin{align*}
f(n) &= n \cdot r^{-1} \bmod m \\
     &= \frac{n - y}{r} \bmod m \; (x = n \cdot m' \bmod r, \; y = x \cdot m) \\
	 &= \begin{cases} \frac{n - y}{r} + m & (n < y) \\ \frac{n - y}{r} & (n \geq y) \end{cases}
\end{align*}
$$

정리하면, $m$이 홀수이고 $0 \leq n < rm$이라면 $f(n) = n \cdot r^{-1} \bmod m$을 $r = 2^w$를 이용해 비트 연산과 조건 분기로 계산할 수 있습니다.

구현 코드는 다음과 같습니다.

```cpp
using u64 = unsigned long long;
using u32 = unsigned int;

u32 m, mr; // r * r^-1 + m * mr = 1

u32 reduce(u64 n) {          // 0 <= n < rm
	u32 x = u32(n) * mr;     // n * mr mod r
	u64 y = u64(x) * m;      // x * m
	u32 ret = (n - y) >> 32; // (n - y) / r
	return n < y ? ret + m : ret;
}
```

해당 코드는 $2 < m < 2^{32}$인 홀수 정수 $m$에 대해 $0 \leq n < rm$인 정수 $n$이 주어질 때, $r = 2^{32}$를 이용해 모듈러 연산 없이 $f(n) = n \cdot r^{-1} \bmod m$을 계산합니다.

이때 <code>mr</code>에는 $r \cdot r^{-1} + m \cdot m' = 1$이고 $0 < m' < r$인 정수 $m'$을 미리 구해뒀다 가정합니다.

**Note.**

- $m < 2^{32}$라면 $r = 2^{32}$를 이용해 $r$에 대한 모듈러 연산을 32비트 정수 자료형의 오버플로우를 이용해 추가 연산 없이 구현할 수 있습니다.
- $m < 2^{64}$라면 $r = 2^{64}$를 이용해 $r$에 대한 모듈러 연산을 64비트 정수 자료형의 오버플로우를 이용해 추가 연산 없이 구현할 수 있습니다.
- $m$이 홀수가 아니라면 $m = 2^a \cdot b$에서 $n \bmod 2^a$는 비트 연산으로, $n \bmod b$는 Montgomery Reduction으로 구한 뒤 Chinese Remainder Theorem을 이용하면 $n \bmod m$을 구할 수 있습니다.
- <code>mr</code>는 $r, m$에 대해 확장 유클리드 알고리즘을 사용해 $\mathcal{O}(\log r)$에 구할 수 있습니다.

### 3.3 Fast Inverse

$f(n)$을 빠르게 계산하기 위해선 $r \cdot r^{-1} + m \cdot m' = 1$이고 $0 < m' < r$인 정수 $m'$를 미리 구해둬야 합니다. 일반적인 경우에는 확장 유클리드 알고리즘을 이용해 $\mathcal{O}(\log r)$에 두 정수 $r^{-1}, m'$을 구할 수 있지만, $r$이 $2$의 거듭제곱이라면 더 간단한 방법으로 $m'$를 구할 수 있습니다.

$m$이 $m > 2$인 홀수 정수이고, $i$가 $i \geq 1$인 정수라 합시다.

$m \cdot x \equiv 1 \bmod 2^i$라면 임의의 $k \in \mathbb{Z}$에 대해

$$
\begin{align*}
m \cdot x \cdot (2 - m \cdot x) &= 2 \cdot m \cdot x - (m \cdot x)^2 \\
                                &= 2 \cdot (1 + k \cdot 2^i) - (1 + k \cdot 2^i)^2 \\
				                &= 2 + 2 \cdot k \cdot 2^i  - 1 - 2 \cdot k \cdot 2^i - k^2 \cdot 2^{2i} \\
								&= 1 - k^2 \cdot 2^{2i} \\
								&\equiv 1 \bmod 2^{2i}
\end{align*}
$$

에서 $m \cdot x \cdot (2 - m \cdot x) \equiv 1 \bmod 2^{2i}$가 성립합니다.

따라서 $x_0 = 1$에서 시작해 $x_{i+1} = x_i(2 - m x_i) \bmod 2^{2^i}$를 반복하면 $r = 2^w$에 대한 $m \cdot m' \equiv 1 \bmod 2^w$인 정수 $m'$을 $\mathcal{O}(\log w)$에 구할 수 있습니다.

구현 코드는 다음과 같습니다.

```cpp
using u64 = unsigned long long;
using u32 = unsigned int;

u32 minv(u32 m) { // m^{-1} mod 2^32
	u32 x = 1;
	for (int i = 0; i < 5; i++) x *= 2 - m * x;
	return x;
}
```

해당 코드는 $2 < m < 2^{32}$인 홀수 정수 $m$과 $r = 2^{32}$에 대해 $r \cdot r^{-1} + m \cdot m' = 1$이고 $0 < m' < r$인 정수 $m'$를 계산합니다.

**Note.**

- $m < 2^{64}$라면 $r = 2^{64}$에 대해 $m' = m^{-1} \bmod 2^{64}$를 구하면 됩니다.
- $r = 2^{32}$는 $\log_2 32 = 5$번의 반복으로 모듈러 역원을 구할 수 있습니다.
- $r = 2^{64}$는 $\log_2 64 = 6$번의 반복으로 모듈러 역원을 구할 수 있습니다.

### 3.4 Modular Multiplication in $\mathbb{Z}_m$ using Montgomery Reduction

지금까지의 논의를 종합하면 $2 < m < 2^{32}$인 홀수 정수 $m$에 대해 $\mathbb{Z}_m$에서의 곱셈을 효율적으로 구현할 수 있습니다.

구현 코드는 다음과 같습니다.

```cpp
using u64 = unsigned long long;
using u32 = unsigned int;

struct modmul_montgomery {
	modmul_montgomery() {}
	modmul_montgomery(u32 m) : m(m), mr(1), r2(-u64(m) % m) {
		for (int i = 0; i < 5; i++) mr *= 2 - m * mr;
	}
	u32 reduce(u64 n) const {
		u32 x = u32(n) * mr;
		u64 y = u64(x) * m;
		u32 ret = (n - y) >> 32;
		return n < y ? ret + m : ret;
	}
	u32 mul(u32 a, u32 b) const {
		return reduce(u64(a) * b);
	}
	u32 transform(u32 n) const {
		return mul(n, r2);
	}
private:
	u32 m, mr, r2;
};
```

해당 코드는 $2 < m < 2^{32}$인 홀수 정수 $m$에 대해 $0 \leq a, b < 2^{32}$인 두 정수 $a, b$가 주어질 때 $a \cdot b \bmod m$을 $r = 2^{32}$인 Montgomery Reduction을 이용해 빠르게 계산합니다.

코드에서 <code>mr</code>는 $m^{-1} \bmod 2^{32}$, <code>r2</code>는 $2^{64} \bmod m$을 나타냅니다. <code>r2</code>의 계산은 $(2^{64} - m) \bmod m = 2^{64} \bmod m$을 이용합니다.

<code>transform</code> 함수는 $0 \leq n < 2^{32}$인 정수 $n$에 대해 $\overline{n} = n \cdot 2^{32} \bmod m$을 구합니다. 이는 $\overline{n} = f(n \cdot r^2)$을 이용해 <code>n</code>과 <code>r2</code>를 Montgomery Form으로 취급해 곱셈을 수행하며 계산할 수 있습니다. <code>reduce</code> 함수는 $0 \leq n < rm$인 정수 $n$에 대해 $f(n) = n \cdot 2^{-32} \bmod m$을 구하며, <code>mul</code> 함수는 <code>reduce</code> 함수를 이용해 $0 \leq ab < rm$인 두 정수 $a, b$에 대해 $f(a \cdot b) = a \cdot b \cdot 2^{-32} \bmod m$을 구합니다.

사용 예시는 다음과 같습니다. [(코드)](http://boj.kr/88279849af73438c80aa6d05a4c8f707)

**Note.**

- Barrett Reduction과 달리 Montgomery Reduction을 사용하기 위해선 $m$이 홀수여야 합니다.
- 곱셈 연산 이전과 이후에는 <code>transform</code>, <code>reduce</code>를 이용한 Montgomery Form으로의 변환과 역변환을 사용해야 합니다.
- $2 < m < 2^{64}$라면 $r = 2^{64}$와 128비트 정수 자료형을 이용하면 됩니다.

## 4. Exact Division

$2 < m < 2^{32}$인 홀수 정수 $m$에 대해 $0 \leq n < 2^{32}$인 정수 $n$이 $m$의 배수인지 여부를 알기 위해서는 <code>n % m == 0</code>와 같은 모듈러 연산이 필요합니다. 이때 $m$이 홀수 정수라면 $m^{-1} \bmod 2^{32}$가 존재함을 이용해 해당 연산을 최적화할 수 있습니다.

$0 \leq n < 2^{32}$에 대해 $x = n \cdot m^{-1} \bmod 2^{32}$를 정의합시다. $n$이 $m$의 배수라면 $0 \leq x \leq \lfloor \frac{2^{32} - 1}{m} \rfloor$이 성립합니다. 반대로 $0 \leq x \leq \lfloor \frac{2^{32} - 1}{m} \rfloor$이라면 $xm < 2^{32}$에서 $n = n \bmod 2^{32} = n \cdot m^{-1} \cdot m \bmod 2^{32} = x \cdot m$이 성립하니 $n$은 $m$의 배수입니다.

따라서 $x \leq \lfloor \frac{2^{32} - 1}{m} \rfloor$를 이용하면 모듈러 연산 없이 $n$이 $m$의 배수인지 여부를 알 수 있습니다. 이 사실을 이용하는 최적화 기법을 Exact Division이라 합니다.

구현 코드는 다음과 같습니다.

```cpp
using u64 = unsigned long long;
using u32 = unsigned int;

struct exact_div {
	exact_div() {}
	exact_div(u32 m) : mr(1), lim(u32(-1) / m) {
		for (int i = 0; i < 5; i++) mr *= 2 - m * mr;
	}
	u32 is_mul(u32 n) {
		return n * mr <= lim;
	}
private:
	u32 mr, lim;
};
```

해당 코드는 $2 < m < 2^{32}$인 홀수 정수 $m$에 대해 $0 \leq n < 2^{32}$인 정수 $n$이 $m$의 배수인지 여부를 Exact Division을 이용해 빠르게 계산합니다.

**Note.**

- Montgomery Reduction과 마찬가지로 Exact Division을 사용하기 위해서는 $m$이 홀수여야 합니다.
- $m < 2^{64}$, $n < 2^{64}$라면 $2^{64}$를 이용하면 됩니다.

## References

[1] [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)

[2] [https://en.wikipedia.org/wiki/Barrett_reduction](https://en.wikipedia.org/wiki/Barrett_reduction)

[3] [https://modoocode.com/313](https://modoocode.com/313)

[4] [https://cp-algorithms.com/algebra/montgomery_multiplication.html](https://cp-algorithms.com/algebra/montgomery_multiplication.html)

[5] [https://en.algorithmica.org/hpc/number-theory/montgomery/](https://en.algorithmica.org/hpc/number-theory/montgomery/)

[6] [https://codeforces.com/blog/entry/103374](https://codeforces.com/blog/entry/103374)

[7] [https://simonlindholm.github.io/files/bincoef.pdf](https://simonlindholm.github.io/files/bincoef.pdf)

[8] [https://github.com/atcoder/ac-library/blob/master/atcoder/internal_math.hpp](https://github.com/atcoder/ac-library/blob/master/atcoder/internal_math.hpp)