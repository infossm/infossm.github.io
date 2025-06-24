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

또한 대부분의 아키텍처에서 SIMD(Single Instruction Multiple Data) 명령어 집합은 정수 나눗셈(<code>/</code>)과 모듈러 연산(<code>%</code>)을 지원하지 않기에 Vectorization를 이용한 고속화가 불가능합니다.

이번 글에서는 정수 나눗셈(<code>/</code>), 모듈러 연산(<code>%</code>)을 시프트(<code>>></code>), 덧셈(<code>+</code>), 곱셈(<code>*</code>)으로 대체하는 Barrett Reduction, Montgomery Reduction 기법을 소개하고, 이를 통해 모듈러가 comiple-time에 주어지지 않는 상황이나 SIMD 기반의 병렬 연산을 구현할 때 나눗셈, 모듈러 연산의 성능을 개선하는 방법을 설명합니다.

## 2. Barrett Reduction

Barrett Reduction은 $n \geq 0$, $m \geq 2$일 때, 충분히 큰 $k$에 대해

$$
\left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor
$$

가 성립함을 이용하는 정수 나눗셈 최적화 기법입니다. 여기서 $\left\lceil \frac{2^k}{m} \right\rceil$은 미리 계산해둔 뒤 곱할 수 있고, $2^k$로 나누는 과정은 시프트 연산으로 처리할 수 있습니다.

등식이 성립하는 $k$를 구하기 위해 먼저 다음의 보조 정리를 증명하겠습니다.

**Lemma 1.** Floor Equality

$$
\begin{align*}
&0 \leq a &&(a \in \mathbb{Z}) \\
&2 \leq b &&(b \in \mathbb{Z}) \\
&0 \leq e < \frac{1}{b} &&(e \in \mathbb{R}) \\
&\Rightarrow \quad \left\lfloor \frac{a}{b} + e \right\rfloor = \left\lfloor \frac{a}{b} \right\rfloor
\end{align*}
$$

**Proof.**

$$
\begin{align*}
a &= qb + r, \; q = \left\lfloor \frac{a}{b} \right\rfloor \\
  &\Rightarrow 0 \leq r \leq b - 1 \\
  &\Rightarrow 0 \leq \frac{r}{b} \leq 1 - \frac{1}{b} \\
  &\Rightarrow 0 \leq \frac{r}{b} + e < 1 \\
  &\therefore \left\lfloor \frac{a}{b} + e \right\rfloor = \left\lfloor q + \left( \frac{r}{b} + e \right) \right\rfloor = q &&\square
\end{align*}
$$

위의 결과를 이용하면 $\frac{n}{m} + e = n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k}$에서 $0 \leq e < \frac{1}{m}$가 성립하는 것을 보이며 $k$를 구할 수 있습니다.

**Lemma 2.** Barrett Approximation

$$
\begin{align*}
&0 \leq n &&(n \in \mathbb{Z}) \\
&2 \leq m &&(m \in \mathbb{Z}) \\
&2^{\lfloor \log_2 (m - 1) \rfloor} \cdot \max(2n, m) \leq 2^k &&(k \in \mathbb{Z}) \\
&\Rightarrow \quad \left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor
\end{align*}
$$

**Proof.**

$$
\begin{align*}
s &= \left\lfloor \log_2(m - 1) \right\rfloor = \left\lceil \log_2 m \right\rceil - 1 &&(n \leq 2^{k - s - 1}, \; 2^s < m \leq 2^{s+1}) \\
x &= \left\lceil \frac{2^k}{m} \right\rceil \\
r &= x m - 2^k &&(0 \leq r < m) \\
e &= \frac{nx}{2^k} - \frac{n}{m} = \frac{n}{m}(\frac{xm}{2^k} - \frac{2^k}{2^k}) = \frac{n r}{m 2^k} &&(0 \leq e) \\
  &\Rightarrow e - \frac{1}{m} = \frac{1}{m}(\frac{nr}{2^k} - \frac{2^k}{2^k}) < 0 &&(\because nr < nm \leq 2^{k-s-1} \cdot 2^{s+1} = 2^k) \\
  &\Rightarrow 0 \leq e < \frac{1}{m} \\
  &\therefore \left\lfloor \frac{n}{m} \right\rfloor = \left\lfloor \frac{n}{m} + e \right\rfloor = \left\lfloor \frac{nx}{2^k} \right\rfloor = \left\lfloor n \cdot \left\lceil \frac{2^k}{m} \right\rceil \cdot \frac{1}{2^k} \right\rfloor &&\square \\
\end{align*}
$$

Lemma 2에 의해, $2^{\lfloor \log_2 (m - 1) \rfloor} \cdot \max(2n, m) < 2^k$를 만족하는 $k$와 $x = \left\lceil \frac{2^k}{m} \right\rceil$을 구해두면 <code>n / m</code>을 <code>(n * x) >> k</code>로 나타낼 수 있습니다.



대부분의 ps/cp 환경에서 <code>m</code>은 $[2, 2^{31} - 1]$ 범위이니 $k = 93$을 이용할 수 있고, $nx < m^2 (\frac{2^k}{m} + 1) < m 2^k + m^2 < 2^{124}$이니 128비트 정수 자료형을 이용해 <code>n * x</code>를 계산하면 Barrett Reduction을 구현할 수 있습니다.

구현 코드는 다음과 같습니다.

```cpp
using u128 = unsigned __int128;
using u64 = unsigned long long;
using u32 = unsigned int;

struct barrett {
	barrett() {}
	barrett(u32 m) : m(m), x(((u128(1) << 93) + m - 1) / m) {}
	u64 div(u64 n) {
		return n * x >> 93;
	}
	u32 mul(u32 a, u32 b) {
		u64 n = u64(a) * b;
		return n - div(n) * m;
	}
private:
	u32 m;
	u128 x;
};
```

$0 \leq n < m^2$인 정수 $n$에 대해 $\left\lfloor \frac{n}{m} \right\rfloor$를 빠르게 계산할 수 있으면, $0 \leq a, b < m$인 두 정수에 대해 $a \bmod b = ab - \left\lfloor \frac{ab}{m} \right\rfloor m$를 빠르게 계산할 수 있고, 이를 이용해 $\mathbb{Z}_m$에서의 연산을 구현할 수 있습니다.

사용 예시는 다음과 같습니다. [(코드)](http://boj.kr/233deb0addff442ebdd782ac500d9298)

**Note.**

- GNU 계열 컴파일러(GCC/Clang)와 달리 Microsoft Visual C++(MSVC)는 <code>__int128</code> 자료형을 지원하지 않기에 128비트 정수 곱셈을 직접 구현해야 합니다. 같은 이유로 $m$이 <code>long long</code> 범위라면 256비트 정수 곱셈을 직접 구현해야 합니다.
- $k$를 $\left\lfloor \log_2(m - 1) \right\rfloor + \left\lfloor \log_2(m^2 - 1) \right\rfloor + 1$로 설정하면 $[2, 2^{31} - 1]$ 범위의 <code>m</code>에 대해 $2 \leq \left\lceil \frac{2^k}{m} \right\rceil < 2^{63} - 1$이 성립해 <code>x</code>를 <code>long long</code>으로 나타낼 수 있습니다. [(코드)](http://boj.kr/44f6d12b75624a138d2ff6968c7dd2ff)
- $k$가 상수가 아니라면 <code>>></code>에서 추가적인 연산이 생기기 때문에 성능 저하가 있을 수 있습니다. [(참고)](https://godbolt.org/z/vjnnM1eoT)

## 3. Montgomery Reduction

~

## References

[1] [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)

[2] [https://modoocode.com/313](https://modoocode.com/313)

[3] [https://cp-algorithms.com/algebra/montgomery_multiplication.html](https://cp-algorithms.com/algebra/montgomery_multiplication.html)

[4] [https://en.algorithmica.org/hpc/number-theory/montgomery/](https://en.algorithmica.org/hpc/number-theory/montgomery/)

[5] [https://codeforces.com/blog/entry/103374](https://codeforces.com/blog/entry/103374)

[6] [https://simonlindholm.github.io/files/bincoef.pdf](https://simonlindholm.github.io/files/bincoef.pdf)

[7] [https://github.com/atcoder/ac-library/blob/master/atcoder/internal_math.hpp](https://github.com/atcoder/ac-library/blob/master/atcoder/internal_math.hpp)