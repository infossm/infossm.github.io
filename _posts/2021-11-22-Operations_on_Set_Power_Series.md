---
layout: post
title: "Operations on Set Power Series"
author: Aeren
date: 2021-11-22
---

<h2 id="table of contents">Table Of Contents</h2>

* [Introduction](#introduction)
* [Set Power Series](#sps)
  - [Addition](#add)
  - [Or Convolution](#or)
  - [Subset Convolution](#subset)
* [Composition of Set Power Series](#comp)
  - [Exponential](#exp)
  - [General Case](#gen)
* [Example: ARC105 F](#example)



<h2 id="introduction">Introduction</h2>

안녕하세요, Aeren입니다!

이번 글에서는 set power series와 그와 관련된 연산들을 소개하고, combinatorial problem에서 어떻게 활용될 수 있는지 간단하게 소개하겠습니다.



<h2 id="sps">Set Power Series</h2>

어떤 set $G = \lbrace g _ 0,g _ 1,\cdots ,g _ {N-1} \rbrace$와 commutative ring $R$이 주어질 때, 함수 $p:\mathcal{P}(G) \rightarrow R$를 **set power series**라고 부릅니다. 그리고 일반적인 power series와 마찬가지로 indeterminate $X$를 이용해 $p _ S=p(S)$일 때,  $\sum _ {S \subseteq G} p _ SX^S$로 표기합니다. Set $G$와 ring $R$에 대한 set power series의 집합을 $\mathcal{S} _ G(R)$이라 표기하겠습니다.

이 $\mathcal{S} _ G(R)$에 binary operation을 부여함으로서 하나의 algebraic structure로 바라볼 수 있습니다.

<h3 id="add">Addition</h3>

두 set power series $p,q\in \mathcal{S} _ G(R)$에 대하여 addition을 $p+q=\sum_{S\subseteq G}(p _ S + q _ S)X^S$로 정의합니다. 

계산할 땐, 단순히 모든 $S\subseteq G$에 대하여, $p _ S$와 $q _ S$를 더해주면 됩니다. $\Theta (2^N)$만큼의 ring addition연산을 필요로 합니다.

```cpp
template<class T>
vector<T> addition(const vector<T> &p, const vector<T> &q){
	assert(p.size() == q.size() && __builtin_popcount(p.size()) == 1);
	vector<T> r = p;
	for(auto i = 0; i < (int)q.size(); ++ i) r[i] += q[i];
	return r;
}
```

<h3 id="or">Or Convolution</h3>

Binary operation $\bigoplus: \mathcal{S}_G(R) \times \mathcal{S}_G(R) \rightarrow \mathcal{S}_G(R)$을 $p \bigoplus q = \sum_{S,T\subseteq G}p_S q_T X^{S \cup T}$ 이 성립하도록 정의하겠습니다. 임의의 $S \subseteq G$를 각 $g _ i$에 대한 bitmask로 표현할 때, $X ^ a \bigoplus X ^ b = X ^ {a \vert b}$가 성립하기에, 위 연산을 **or convolution**이라 부릅니다. 이 때, $X^{g _ i} \mapsto X _ i :(\mathcal{S} _ G(R),+,\bigoplus) \rightarrow R[X _ 0, \cdots X _ {N - 1}]/(X _ 0^2 - X _ 0,\cdots X _ {N - 1} ^ 2 - X _ {N - 1})$이 ring isomorphism임은 쉽게 확인할 수 있습니다.

이 or convolution은 sum of subset dp라고 알려진 **zeta transform**을 통해 $p$와 $q$를 변환시켜 준 후, 각 term을 곱해주고, 다시 inverse zeta transform을 통해 변환시켜 주면 구할 수 있습니다. Zeta transform은 $\Theta(N\cdot 2^N)$만큼의 ring addition을 필요로 하며 곱하는데 $\Theta(2^N)$만큼의 ring multiplication연산이 필요합니다.

```cpp
template<class T>
void zeta_transform(vector<T> &a, bool invert = false){
	int n = (int)a.size();
	assert(__builtin_popcount(n) == 1);
	for(auto len = 1; len < n; len <<= 1){
		for(auto i = 0; i < n; i += len << 1){
			for(auto j = 0; j < len; ++ j){
				T u = a[i + j], v = a[i + j + len];
				a[i + j + len] += invert ? -u : u;
			}
		}
	}
}

template<class T>
vector<T> or_convolution(vector<T> p, vector<T> q){
	int n = (int)p.size();
	assert(p.size() == q.size() && __builtin_popcount(n) == 1);
	zeta_transform<T>(p), zeta_transform<T>(q);
	for(auto i = 0; i < n; ++ i) p[i] *= q[i];
	zeta_transform<T>(p, true);
	return p;
}
```

<h3 id="subset">Subset Convolution</h3>

Binary operation $\bigotimes: \mathcal{S}_G(R) \times \mathcal{S}_G(R) \rightarrow \mathcal{S}_G(R)$을 $p \bigotimes q = \sum_{S,T\subseteq G,S \cap T = \emptyset}p_S q_T X^{S \cup T}$이 성립하도록 정의하겠습니다. 즉, $p \bigotimes q$의 $X^S$의 coefficient는 $S$의 모든 크기 2의 partition $(L,R)$에 대한 $p _ L \cdot q _ R$의 합입니다. 이 때, $X^{g _ i} \mapsto X _ i :(\mathcal{S} _ G(R),+,\bigotimes) \rightarrow R[X _ 0, \cdots X _ {N - 1}]/(X _ 0^2,\cdots X _ {N - 1} ^ 2)$이 ring isomorphism임은 쉽게 확인할 수 있습니다.

$Rank_i(p)=\sum _ {S \subseteq G, \vert S \vert = i} p _ S X ^ S$라 정의하겠습니다. $p=\sum _ {i = 0} ^ N Rank _ i(p)$이 성립함은 쉽게 확인할 수 있습니다. 또한 임의의 $k$에 대하여, $\sum _ {i+j=k} Rank _ i(p) \bigotimes Rank _ j(q) = Rank _ k(p \bigotimes q)$도 성립합니다. $\vert S \vert + \vert T \vert = \vert S \cup T \vert$일 필요충분 조건은 $S \cap T = \emptyset$이 성립하는 것이므로, $Rank _ i(p) \bigotimes Rank _j(q) = Rank _ {i+j} \left( Rank _ i(p) \bigoplus Rank _ j(q) \right) $도 성립합니다. 위 관찰들에 의해, $\Theta (N^2 \cdot 2^N)$의 ring addition과 $\Theta(N^2 \cdot 2^N)$의 ring multiplication을 필요로 하는 subset convolution algorithm을 얻어낼 수 있습니다.

```cpp
template<class T>
vector<T> subset_convolution(const vector<T> &p, const vector<T> &q){
	int n = (int)p.size();
	assert((int)q.size() == n && __builtin_popcount(n) == 1);
	int w = __lg(n) + 1;
	vector a(w, vector<T>(n)), b(a); // Rank vectors
	for(auto i = 0; i < n; ++ i) a[__builtin_popcount(i)][i] = p[i];
	for(auto i = 0; i < n; ++ i) b[__builtin_popcount(i)][i] = q[i];
	for(auto bit = 0; bit < w; ++ bit){
		zeta_transform<T>(a[bit]);
		zeta_transform<T>(b[bit]);
	}
	vector<T> res(n);
	for(auto bit = 0; bit < w; ++ bit){
		static vector<T> c;
		c.assign(n, 0);
		for(auto lbit = 0; lbit <= bit; ++ lbit) for(auto i = 0; i < n; ++ i) c[i] += a[lbit][i] * b[bit - lbit][i];
		zeta_transform<T>(c, true);
		for(auto i = 0; i < n; ++ i) if(__builtin_popcount(i) == bit) res[i] = c[i];
	}
	return res;
}
```



<h2 id="comp">Composition of Set Power Series</h2>

다음 내용은 [Elegia](https://codeforces.com/profile/Elegia)의 [China Team Selection 2021](https://github.com/EntropyIncreaser/ioi2021-homework/blob/master/thesis/main.tex) 논문 중 일부입니다.

목표는 어떤 $f\in R[[X]]$ 와 적절한 set power series $p$에 대해, $f(p)$를 빠르게 계산하는 것입니다.

<h3 id="exp">Exponential</h3>

$p _ \emptyset = 0$인 Set power series $p$에 대하여, $p^k := \bigotimes _ {i = 0} ^ {k - 1} p$로 정의하면, $k > n$일 때, $p ^ k$는 constant 0입니다. $1,2,\cdots, N$이 $R$에서 multiplicative inverse를 갖는다고 가정하면, $\exp(p)=\sum _ {i=0} ^ N p ^ i / i!$가 잘 정의되며, 일반적인 monovariate power series에서의 exponential의 정의와 일치합니다. 이제 주어진 $p$에 대해 $\exp(p)$를 빠르게 계산하는 법에 대해 소개하겠습니다.

$(\mathcal{S} _ G(R),+,\bigotimes) \cong R[X _ 0, \cdots X _ {N - 1}]/(X _ 0^2,\cdots X _ {N - 1} ^ 2)$이므로 $p$를 $R[X _ 0, \cdots X _ {N - 1}]/(X _ 0^2,\cdots X _ {N - 1} ^ 2)$의 원소로서 다루겠습니다. 일반적인 monovariate power series의 exponential과 비슷하게 임의의 $0\le i < N$에 대하여 $\partial \exp (p) / \partial X _ i = \partial p / \partial X _ i \bigotimes \exp(p)$임은 쉽게 확인할 수 있습니다.

Set power series $p$와 정수 $0\le i < N, 0 \le e \le 1$에 대해 $[i,1]p=\sum _ {g _ i\in S\subseteq G} p _ S X ^ S$, 그리고 $[i,0]p=\sum _ {g _ i \notin S \subseteq G} p _ S X ^ S$라 정의하겠습니다. 위 identity에서 $i=N-1$로 놓은 후 $X _ i$의 exponent가 0인 term들을 비교해보면, $[N-1,1]\exp(p)=[N-1,1]p \bigotimes[N-1,0]\exp(p)$가 얻어집니다. 즉, $[N-1,0]\exp(p)$를 알고있다면, 한 번의 subset convolution을 통해 $[N-1,1]\exp(p)$를 구할 수 있으며, 우리는 indeterminate의 갯수가 1 줄어든 subproblem을 풀면 됩니다. 총 시간복잡도는 $T(N) - T(N - 1) \in \Theta(N^2 \cdot 2 ^ N)$에서 $T(N) \in \Theta(N^2 \cdot 2^N)$입니다. 이 method는 "pointwise Newton iteration"이라 이름붙여져 있습니다.

```cpp
template<class T>
vector<T> exponential(const vector<T> &p){
	int n = (int)p.size();
	assert(__builtin_popcount(n) == 1 && p[0] == 0);
	int w = __lg(n);
	vector<T> res{1};
	for(auto bit = 0; bit < w; ++ bit){
		auto shift = subset_convolution<T>(res, vector<T>(p.begin() + (1 << bit), p.begin() + (1 << bit + 1)));
		res.insert(res.end(), shift.begin(), shift.end());
	}
	return res;
}
```

<h3 id="gen">General Case</h3>

이제 $f(p)$가 잘 정의되는 $f \in R[[x]]$와 $p \in \mathcal{S} _ G(R)$가 주어졌을 때, $f(p)$를 빠르게 계산하는 문제를 생각해 보겠습니다. Exponential일 때와 비슷하게, 양 변에 partial derivative를 취하면 $\partial f(p)/\partial X _ i = \partial p / \partial X _ i \bigotimes f'(p)$가 얻어집니다. 이제 $i=N-1$일 때, $X _ i$의 exponent가 0인 term들을 비교해보면, $[N-1,1] f(p) = [N-1,1] p \bigotimes [N-1,0]f'(p)$이 얻어지며, $[N-1,0]f(p)$와 $[N-1,0]f'(p)$를 구하는 2개의 subproblem으로 나뉩니다. 언뜻 보면, complexity가 매우 커보이지만, 다음 layer에선 $[N-2,0][N-1,0]f(p), [N-2,0][N-1,0]f'(p)$, 그리고 $[N-2,0][N-1,0]f''(p)$의 3개의 subproblem으로 나뉘며, 각 layer마다 subproblem의 갯수가 1씩 증가함을 알수있습니다. 즉, $T(N)=\sum _ {i=0} ^ {N-1} (N-i) \cdot N^2 \cdot 2^N \in \Theta ( N^2 \cdot 2 ^ N )$입니다.



<h2 id="example">Example: ARC105 F</h2>

[문제](https://atcoder.jp/contests/arc105/tasks/arc105_f)는 $1 \le \vert V \vert \le 17$인 simple graph $G$가 주어졌을 때, edge들의 임의의 subset을 제거하여 만들 수 있는 $2 ^ {\vert E \vert}$개의 graph 중에서, connected bipartite graph의 갯수를 998244353로 나눈 나머지를 출력하는 것입니다. Editorial에는 $\Theta(\vert E \vert \cdot 2 ^ {\vert V \vert} + 3 ^ {\vert V \vert})$의 solution이 소개되어 있지만, set power series를 활용하면 $\Theta (\vert E \vert \cdot 2 ^ {\vert V \vert} + \vert V \vert ^ 2 \cdot 2 ^ {\vert V \vert})$로 해결 가능합니다.

임의의 $S \subseteq V$에 대하여, $G[S]$를 $G$의 $S$에 대한 induced subgraph, $E[S]$를 $G[S]$의 edge set으로 정의하겠습니다.

Set power series $A \in \mathcal{S} _ V (\mathbb{F} _ {998244353})$를 각 $S \subseteq V$에 대하여, $A _ S$가 $G[S]$의 edge를 일부 제거한 후, vertex 2-coloring하는 방법의 수를 나타내도록 정의하겠습니다. 첫 번째 색을 칠할 vertex set $T \subseteq S$를 고정했을 때, 그에 맞춰 edge를 제거하는 경우의 수는 $2 ^ {\vert E[S] - E[T] - E[S-T]}$이므로

$\begin{align} A = \sum _ {S \subseteq V} A _ S X ^ S = \sum _ {S \subseteq V} \left( \sum _ {T \subseteq S} 2 ^ {\vert E[S] \vert - \vert E[T] \vert - \vert E[S - T] \vert } \right) X ^ S = \sum _ {S \subseteq V} 2 ^ {\vert E[S] \vert} \left( \sum _ {T \subseteq S} 2 ^ {- \vert E[T] \vert - \vert E[S - T] \vert } \right) X ^ S \end{align}$

이며, $B = \sum _ {S \subseteq V} 2 ^ {- \vert E[S] \vert}$일 때 $A = \sum _ {S \subseteq V} 2 ^ {\vert E[S] \vert}(B \bigotimes B) _ S$임을 알 수 있습니다. 모든 $S \subseteq V$에 대하여 $\vert E[S] \vert$를 계산하는데 $\Theta ( \vert E \vert \cdot 2 ^ {\vert V \vert} )$이 걸리며, $B \bigotimes B$를 계산하는데 $\Theta ( \vert V \vert ^ 2 \cdot 2 ^ {\vert V \vert} )$이 걸리므로, $A$를 계산하는데  $\Theta (\vert E \vert \cdot 2 ^ {\vert V \vert} + \vert V \vert ^ 2 \cdot 2 ^ {\vert V \vert})$의 시간이 걸립니다.

Set power series $R \in \mathcal{S} _ V (\mathbb{F} _ {998244353})$을 각 $S$에 대해, $R _ S$가 $G[S]$가 connected component가 한 개인 bipartite graph가 되도록 edge의 일부를 제거하는 방법의 수라고 정의하겠습니다. 문제에서 출력해야 되는 값은 $R _ V$입니다. 임의의 connected bipartite graph를 vertex 2-coloring하는 방법은 정확히 두 가지이므로 $2R _ S$는 $G[S]$의 일부 edge를 G[S]가 connected이도록 제거 한 후, vertex 2-coloring하는 방법의 수를 나타내게 됩니다.

이제 $A$를 다른방법으로 enumerate해보겠습니다. 각 $S \subseteq V$에 대하여, $A _ S$는 $S$의 모든 unordered partition $\lbrace S _ 1, \cdots, S _ k \rbrace$에 대한 $\prod _ {i=1} ^ k 2R _ {S _ i}$의 합과 같습니다. 그리고 이는 $S$의 모든 ordered partition $(S _ 1, \cdots , S _ k)$에 대한 $1 / k! \cdot \prod _ {i = 1} ^ k 2R _ {S _ i}$의 합과 같습니다. 마지막으로, 이는 $S = \cup _ {i = 1} ^ k S _ i$를 만족하는 모든 non-empty $S _ i \subset V$에 대하여 $1 / k! \cdot [S _ i \mathrm{s \space are \space disjoint}] \cdot \prod _ {i = 1} ^ k 2R _ {S _ i} $의 합과 같습니다.  고정된 $k$에 대하여, 마지막 식은 $1/k! \cdot (2R)^k$의 $X ^ S$의 coefficient와 일치한다는 것을 어렵지 않게 확인할 수 있습니다. 따라서

$\begin{align} A = \sum _ {k = 0} ^ {\vert V \vert} \frac1{k!} (2R) ^ k = \exp(2R)  \end{align}$

입니다. 양변에 $\log$를 취한 후 $2$로 나눠주면, $R=1/2 \cdot \log (A)$임이 얻어집니다. 위 general case에 $f = \log$를 대입해주면, $R$을 $A$로 부터 $\Theta ( \vert V \vert ^ 2 \cdot 2 ^ {\vert V \vert} )$시간에 계산할 수 있습니다. 총 시간복잡도는 $\Theta (\vert E \vert \cdot 2 ^ {\vert V \vert} + \vert V \vert ^ 2 \cdot 2 ^ {\vert V \vert})$입니다.

```cpp
#include <bits/stdc++.h>
using namespace std;

template<typename T>
struct modular_base{
	using Type = typename decay<decltype(T::value)>::type;
	static vector<Type> _MOD_INV;
	constexpr modular_base(): value(){ }
	template<typename U> modular_base(const U &x){ value = normalize(x); }
	template<typename U> static Type normalize(const U &x){
		Type v;
		if(-mod() <= x && x < mod()) v = static_cast<Type>(x);
		else v = static_cast<Type>(x % mod());
		if(v < 0) v += mod();
		return v;
	}
	const Type& operator()() const{ return value; }
	template<typename U> explicit operator U() const{ return static_cast<U>(value); }
	constexpr static Type mod(){ return T::value; }
	modular_base &operator+=(const modular_base &otr){ if((value += otr.value) >= mod()) value -= mod(); return *this; }
	modular_base &operator-=(const modular_base &otr){ if((value -= otr.value) < 0) value += mod(); return *this; }
	template<typename U> modular_base &operator+=(const U &otr){ return *this += modular_base(otr); }
	template<typename U> modular_base &operator-=(const U &otr){ return *this -= modular_base(otr); }
	modular_base &operator++(){ return *this += 1; }
	modular_base &operator--(){ return *this -= 1; }
	modular_base operator++(int){ modular_base result(*this); *this += 1; return result; }
	modular_base operator--(int){ modular_base result(*this); *this -= 1; return result; }
	modular_base operator-() const{ return modular_base(-value); }
	template<typename U = T>
	typename enable_if<is_same<typename modular_base<U>::Type, int>::value, modular_base>::type &operator*=(const modular_base& rhs){
		#ifdef _WIN32
		unsigned long long x = static_cast<long long>(value) * static_cast<long long>(rhs.value);
		unsigned int xh = static_cast<unsigned int>(x >> 32), xl = static_cast<unsigned int>(x), d, m;
		asm(
			"divl %4; \n\t"
			: "=a" (d), "=d" (m)
			: "d" (xh), "a" (xl), "r" (mod())
		);
		value = m;
		#else
		value = normalize(static_cast<long long>(value) * static_cast<long long>(rhs.value));
		#endif
		return *this;
	}
	template<typename U = T>
	typename enable_if<is_same<typename modular_base<U>::Type, long long>::value, modular_base>::type &operator*=(const modular_base &rhs){
		long long q = static_cast<long long>(static_cast<long double>(value) * rhs.value / mod());
		value = normalize(value * rhs.value - q * mod());
		return *this;
	}
	template<typename U = T>
	typename enable_if<!is_integral<typename modular_base<U>::Type>::value, modular_base>::type &operator*=(const modular_base &rhs){
		value = normalize(value * rhs.value);
		return *this;
	}
	template<typename U>
	modular_base &operator^=(U e){
		if(e < 0) *this = 1 / *this, e = -e;
		modular_base res = 1;
		for(; e; *this *= *this, e >>= 1) if(e & 1) res *= *this;
		return *this = res;
	}
	template<typename U>
	modular_base operator^(U e) const{
		return modular_base(*this) ^= e;
	}
	modular_base &operator/=(const modular_base &otr){
		Type a = otr.value, m = mod(), u = 0, v = 1;
		if(a < (int)_MOD_INV.size()) return *this *= _MOD_INV[a];
		while(a){
			Type t = m / a;
			m -= t * a; swap(a, m);
			u -= t * v; swap(u, v);
		}
		assert(m == 1);
		return *this *= u;
	}
	Type value;
};
template<typename T> bool operator==(const modular_base<T> &lhs, const modular_base<T> &rhs){ return lhs.value == rhs.value; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> bool operator==(const modular_base<T>& lhs, U rhs){ return lhs == modular_base<T>(rhs); }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> bool operator==(U lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) == rhs; }
template<typename T> bool operator!=(const modular_base<T> &lhs, const modular_base<T> &rhs){ return !(lhs == rhs); }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> bool operator!=(const modular_base<T> &lhs, U rhs){ return !(lhs == rhs); }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> bool operator!=(U lhs, const modular_base<T> &rhs){ return !(lhs == rhs); }
template<typename T> bool operator<(const modular_base<T> &lhs, const modular_base<T> &rhs){ return lhs.value < rhs.value; }
template<typename T> bool operator>(const modular_base<T> &lhs, const modular_base<T> &rhs){ return lhs.value > rhs.value; }
template<typename T> bool operator<=(const modular_base<T> &lhs, const modular_base<T> &rhs){ return lhs.value <= rhs.value; }
template<typename T> bool operator>=(const modular_base<T> &lhs, const modular_base<T> &rhs){ return lhs.value >= rhs.value; }
template<typename T> modular_base<T> operator+(const modular_base<T> &lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) += rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator+(const modular_base<T> &lhs, U rhs){ return modular_base<T>(lhs) += rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator+(U lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) += rhs; }
template<typename T> modular_base<T> operator-(const modular_base<T> &lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) -= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator-(const modular_base<T>& lhs, U rhs){ return modular_base<T>(lhs) -= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator-(U lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) -= rhs; }
template<typename T> modular_base<T> operator*(const modular_base<T> &lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) *= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator*(const modular_base<T>& lhs, U rhs){ return modular_base<T>(lhs) *= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator*(U lhs, const modular_base<T> &rhs){ return modular_base<T>(lhs) *= rhs; }
template<typename T> modular_base<T> operator/(const modular_base<T> &lhs, const modular_base<T> &rhs) { return modular_base<T>(lhs) /= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator/(const modular_base<T>& lhs, U rhs) { return modular_base<T>(lhs) /= rhs; }
template<typename T, typename U, typename enable_if<is_integral<U>::value>::type* = nullptr> modular_base<T> operator/(U lhs, const modular_base<T> &rhs) { return modular_base<T>(lhs) /= rhs; }
template<typename T> istream &operator>>(istream &in, modular_base<T> &number){
	typename common_type<typename modular_base<T>::Type, long long>::type x;
	in >> x;
	number.value = modular_base<T>::normalize(x);
	return in;
}
template<typename T> ostream &operator<<(ostream &out, const modular_base<T> &number){ return out << number(); }
template<typename T> vector<typename modular_base<T>::Type> modular_base<T>::_MOD_INV;
template<typename T>
void precalc_inverse(int SZ){
	auto &inv = T::_MOD_INV;
	if(inv.empty()) inv.assign(2, 1);
	for(; inv.size() <= SZ; ) inv.push_back((T::mod() - 1LL * T::mod() / (int)inv.size() * inv[T::mod() % (int)inv.size()]) % T::mod());
}
template<typename T>
vector<T> precalc_power(T base, int SZ){
	vector<T> res(SZ + 1, 1);
	for(auto i = 1; i <= SZ; ++ i) res[i] = res[i - 1] * base;
	return res;
}

/*
using ModType = int;
struct VarMod{ static ModType value; };
ModType VarMod::value;
ModType &mod = VarMod::value;
using modular = modular_base<VarMod>;
*/

// constexpr int mod = 1e9 + 7; // 1000000007
constexpr int mod = (119 << 23) + 1; // 998244353
// constexpr int mod = 1e9 + 9; // 1000000009
using modular = modular_base<integral_constant<decay<decltype(mod)>::type, mod>>;

template<class T>
void zeta_transform(vector<T> &a, bool invert = false){
	int n = (int)a.size();
	assert(__builtin_popcount(n) == 1);
	for(auto len = 1; len < n; len <<= 1){
		for(auto i = 0; i < n; i += len << 1){
			for(auto j = 0; j < len; ++ j){
				T u = a[i + j], v = a[i + j + len];
				a[i + j + len] += invert ? -u : u;
			}
		}
	}
}

template<class T>
vector<T> subset_convolution(const vector<T> &p, const vector<T> &q){
	int n = (int)p.size();
	assert((int)q.size() == n && __builtin_popcount(n) == 1);
	int w = __lg(n) + 1;
	vector a(w, vector<T>(n)), b(a); // Rank vectors
	for(auto i = 0; i < n; ++ i) a[__builtin_popcount(i)][i] = p[i];
	for(auto i = 0; i < n; ++ i) b[__builtin_popcount(i)][i] = q[i];
	for(auto bit = 0; bit < w; ++ bit){
		zeta_transform<T>(a[bit]);
		zeta_transform<T>(b[bit]);
	}
	vector<T> res(n);
	for(auto bit = 0; bit < w; ++ bit){
		static vector<T> c;
		c.assign(n, 0);
		for(auto lbit = 0; lbit <= bit; ++ lbit) for(auto i = 0; i < n; ++ i) c[i] += a[lbit][i] * b[bit - lbit][i];
		zeta_transform<T>(c, true);
		for(auto i = 0; i < n; ++ i) if(__builtin_popcount(i) == bit) res[i] = c[i];
	}
	return res;
}

template<class T>
vector<T> logarithm(const vector<T> &p){
	int n = (int)p.size();
	assert(__builtin_popcount(n) == 1 && p[0] == 1);
	int w = __lg(n);
	vector<vector<T>> res(w + 1, {0});
	T fact = 1;
	for(auto bit = 1; bit <= w; ++ bit) res[bit][0] = fact, fact *= -bit;
	for(auto bit = 0; bit < w; ++ bit){
		for(auto i = 0; i < w - bit; ++ i){
			auto shift = subset_convolution<T>(res[i + 1], vector<T>(p.begin() + (1 << bit), p.begin() + (1 << bit + 1)));
			res[i].insert(res[i].end(), shift.begin(), shift.end());
		}
		res.pop_back();
	}
	return res[0];
}

int main(){
	cin.tie(0)->sync_with_stdio(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int n, m;
	cin >> n >> m;
	auto power = precalc_power(modular(2), m);
	auto invpower = precalc_power(1 / modular(2), m);
	vector<array<int, 2>> edge(m);
	for(auto &[u, v]: edge){
		cin >> u >> v, -- u, -- v;
	}
	vector<int> cnt(1 << n);
	for(auto mask = 0; mask < 1 << n; ++ mask){
		for(auto [u, v]: edge){
			if(mask & 1 << u && mask & 1 << v){
				++ cnt[mask];
			}
		}
	}
	vector<modular> f(1 << n);
	for(auto i = 0; i < 1 << n; ++ i){
		f[i] = invpower[cnt[i]];
	}
	f = subset_convolution<modular>(f, f);
	for(auto i = 0; i < 1 << n; ++ i){
		f[i] *= power[cnt[i]];
	}
	cout << logarithm<modular>(f).back() / 2 << "\n";
	return 0;
}
```

