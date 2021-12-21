---
layout: post
title: "Bostan-Mori Algorithm"
author: Aeren
date: 2021-12-21
---

<h2 id="table of contents">Table Of Contents</h2>

* [Introduction](#introduction)
* [Review of the Kitamasa's Method](#kitamasa)
* [Bostan-Mori Algorithm](#bostan-mori)
* [Optimization under the DFT Setting](#optimization)
  - [Review of DFT](#dft)
  - [DFT Doubling](#doubling)
  - [Bostan-Mori Algorithm under the DFT Setting](#fast-bostan-mori)
* [Benchmarking](#benchmarking)



<h2 id="introduction">Introduction</h2>

안녕하세요, Aeren입니다!

주어진 commutative ring $R$과 어떤 positive integer $d$에 대하여

$\begin{align} a _ {i + d} = \sum _ {j = 0} ^ {d - 1} c _ j \cdot a _ {i + j} \end{align}$

꼴의 recurrence relation으로 표현되는 sequence $a : \mathbb{Z} _ {\ge 0} \rightarrow R$을 **C-recursive sequence** 또는 **linearly recurrent sequence**라고 부릅니다. $M(d)$를 degree $d$인 $R$에서의 polynomial 두 개를 곱하는데 필요한 ring operation의 갯수라 합시다. 일반적으로  이 C-recursive sequence의 $N$번째 term을 구하는 문제는 **Kitamasa's method**라는 algorithm을 통해 $O(M(d) \cdot \log N)$의 ring operation을 통해 계산할 수 있음이 많이 알려져 있습니다. $R$이 fast fourier transform을 지원하는, 즉 $R$이 충분히 큰 $k$에 대하여 primitive $2 ^ k$-th root of unity를 가지는, field일 경우, $M(d) \in O(d \cdot \log d)$이며, 일반적으로 $M(d) \in O(d \cdot \log d \cdot \log \log d)$이므로, 총 $O(d \cdot \log d \cdot \log N)$ 또는 $O(d \cdot \log d \cdot \log \log d \cdot \log N)$번의 ring operation이 필요로 합니다.

이번 글에서 소개할 **Bostan-Mori algorithm**은  C-recursive sequence의 $N$번째 term을 같은 $O(M(d) \cdot \log N)$의 시간복잡도이지만 훨씬 작은 상수로 구하는 algorithm입니다.

이 글은 [다음](https://arxiv.org/pdf/2008.08822.pdf)글을 바탕으로 작성되었습니다.



<h2 id="kitamasa">Review of the Kitamasa's Method</h2>

임의의 $i \ge d$에 대하여

$\begin{align} a _ i = \sum _ {j = 0} ^ {d - 1} c _ j \cdot a _ {i - d + j} \end{align}$

에서 각 $a _ k$를 $X ^ k$로 대체한 식을 살펴보면

$\begin{align} X ^ i \rightarrow \sum _ {j = 0} ^ {d - 1} c _ j \cdot X ^ {i - d + j} \end{align}$

이며, $\Gamma(X) = X ^ d - \sum _ {j = 0} ^ {d - 1} c _ j X ^ j \in R[X]$라고 하면 

$\begin{align} X ^ i \rightarrow X ^ i - X ^ {i - d} \cdot \Gamma(X) \end{align}$

로 쓸 수 있으며 위 과정은 degree가 $d$미만이 될 때까지 반복됩니다. 즉, $R[X]$에서 $X ^ N$을 $\Gamma(X)$로 나눈 나머지 $\rho(X) \in R[X]$를 찾는다면

$\begin{align} a _ N = \sum _ {j = 0} ^ {d - 1} \rho _ j \cdot a _ j \end{align}$

가 성립하게 됩니다. 위 식에서 $O(d)$의 ring operation으로 $a _ N$을 구할 수 있고 $X ^ N \mod \Gamma(X)$는 binary exponentiation을 통해 $O(\log N)$번의 polynomial multiplication과 division으로 구할 수 있습니다. Polynomial multiplication과 division에는 $O(M(d))$의 ring operation이 필요하므로 총 $O(M(d) \cdot \log N)$의 ring operation을 통해 $a _ N$을 구할 수 있습니다.

연산량을 더 자세히 분석해보면, polynomial division은 inverse formal power series를 구하는 연산 한 번과 polynomial multiplication 두 번이 필요합니다. Inverse formal power series는 algorithm시작부분에서 한 번만 계산하면 되며, 각 $\lfloor \log N \rfloor$번의 iteration마다 하나의 polynomial multiplication이 추가로 필요하므로, 대략 

$3 \cdot M(d) \cdot \lfloor \log N \rfloor  + O(d \cdot (\log d + \log N))$

만큼의 ring operation이 필요합니다.

다음은 위 algorithm의 C++ implementation입니다. $R = \mathbb{Z} / 998244353\mathbb{Z}$는 $2 ^ {23}$까지의 fast fourier transform을 지원하므로 $M(d) \in O(\log d)$입니다. Implementation의 *class T*는 이 글의 마지막부분 implementation에 포함되 있는 *modular*를 사용하셔야 합니다.

```cpp
template<class T, int primitive_root = 3>
void number_theoric_transform(vector<T> &a, const bool invert = false){
	int n = (int)a.size();
	assert(n && __builtin_popcount(n) == 1 && (T().mod() - 1) % n == 0);
	const T root = T(primitive_root) ^ (T().mod() - 1) / n;
	const T inv_root = T(1) / root;
	for(auto i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(a[i], a[j]);
	}
	for(auto len = 1; len < n; len <<= 1){
		T wlen = (invert ? inv_root : root) ^ n / len / 2;
		for(auto i = 0; i < n; i += len << 1){
			T w = 1;
			for(auto j = 0; j < len; ++ j, w *= wlen){
				T u = a[i + j], v = a[i + j + len] * w;
				a[i + j] = u + v, a[i + j + len] = u - v;
			}
		}
	}
	if(invert){
		T inv_n = T(1) / n;
		for(auto &x: a) x *= inv_n;
	}
}

template<class T, int primitive_root = 3>
vector<T> convolute_ntt(vector<T> p, vector<T> q){
	if(min(p.size(), q.size()) < 60){
		vector<T> res((int)p.size() + (int)q.size() - 1);
		for(auto i = 0; i < (int)p.size(); ++ i) for(auto j = 0; j < (int)q.size(); ++ j) res[i + j] += p[i] * q[j];
		return res;
	}
	int m = (int)p.size() + (int)q.size() - 1, n = 1 << __lg(m) + 1;
	p.resize(n), q.resize(n);
	number_theoric_transform<T, primitive_root>(p);
	number_theoric_transform<T, primitive_root>(q);
	for(auto i = 0; i < n; ++ i) p[i] *= q[i];
	number_theoric_transform<T, primitive_root>(p, true);
	p.resize(m);
	return p;
}

template<class T>
vector<T> operator*(const vector<T> &p, const vector<T> &q){
	return convolute_ntt(p, q);
}
template<class T>
vector<T> &operator*=(vector<T> &a, const vector<T> &b){
	return a = a * b;
}
template<class T>
vector<T> &operator-=(vector<T> &a, const vector<T> &b){
	if(a.size() < b.size()) a.resize(b.size());
	for(auto i = 0; i < (int)b.size(); ++ i) a[i] -= b[i];
	return a;
}
// Returns the first length terms of the inverse of a
template<class T>
vector<T> inverse(const vector<T> &a, int length){
	assert(!a.empty() && a[0]);
	static vector<T> b;
	b = {1 / a[0]};
	while((int)b.size() < length){
		static vector<T> x;
		x.assign(min(a.size(), b.size() << 1), 0);
		copy(a.begin(), a.begin() + x.size(), x.begin());
		x *= b * b;
		b.resize(b.size() << 1);
		for(auto i = (int)b.size() >> 1; i < (int)min(x.size(), b.size()); ++ i) b[i] = -x[i];
	}
	b.resize(length);
	return b;
}

template<class T, int primitive_root = 3>
struct linear_recurrence_solver_kitamasa{
	int n;
	vector<T> init, coef, q, inv;
	linear_recurrence_solver_kitamasa(const vector<T> &init, const vector<T> &coef): n((int)coef.size()), init(init), coef(coef){
		assert((int)coef.size() == (int)init.size());
		for(auto &x: coef) q.push_back(-x);
		q.push_back(1);
		inv = inverse(vector<T>(rbegin(q), rend(q)), n + 1);
	}
	template<class U>
	T operator[](U i) const{
		assert(0 <= i);
		if(n == 0) return 0;
		auto merge = [&](const vector<T> &a, const vector<T> &b){
			auto res = convolute_ntt<T, primitive_root>(a, b);
			auto t = convolute_ntt<T, primitive_root>(vector<T>(rbegin(res), rbegin(res) + n + 1), inv);
			t.resize(n + 1);
			reverse(begin(t), end(t));
			res -= t * q;
			res.resize(n + 1);
			return res;
		};
		vector<T> power(n + 1), base(n + 1);
		for(power[0] = base[1] = 1; i; i >>= 1, base = merge(base, base)) if(i & 1) power = merge(power, base);
		T res = 0;
		for(auto i = 0; i < n; ++ i) res += power[i] * init[i];
		return res;
	}
};
```



<h2 id="bostan-mori">Bostan-Mori Algorithm</h2>

$\begin{align} a _ {i + d} = \sum _ {j = 0} ^ {d - 1} c _ j \cdot a _ {i + j} \end{align}$

로 정의되는 C-recursive sequence $a$의 generating function

$\begin{align} F(X) = \sum _ {j = 0} ^ \infty a _ j \cdot X ^ j \in R[[X]] \end{align}$

를 생각해보겠습니다.

$\begin{align} Q(X) = 1 - \sum _ {j = 1} ^ d c _ {d - j} \cdot X ^ j \in R[[X]] \end{align}$

그리고 $\begin{align} P(X) = F(X) \cdot Q(X) \end{align}$ 라 정의하면, 임의의 $i \ge d$에 대하여,

$\begin{align} [X ^ i]P(X) = a _ i - \sum _ {j = 0} ^ {d - 1} c _ j \cdot a _ {i - d + j} = 0 \end{align}$

이므로 $P(X)$는 degree가 $d$ 미만인 polynomial입니다. 또한, $P(X) = F(X) \cdot Q(X)$로 부터, $P(X)$를 $M(d)$의 ring operation에 계산할 수 있습니다. 그리고 $[X ^ 0]Q(X)=1$ 이므로 $Q(X)$의 unique한 inverse $Q(X) ^ {-1}$가 존재하며

$\begin{align} F(X) = \frac{P(X)}{Q(X)} := P(X) \cdot Q(X) ^ {-1} \end{align}$

이라 쓸 수 있습니다. 따라서 $a$의 $N$번째 term을 계산하기 위해서는 $[X ^ N]P(X) / Q(X)$를 계산해야 합니다. 이제 분모와 분자에 $Q(-X)$를 곱해주면

$\begin{align} [X^N]\frac{P(X)\cdot Q(-X)}{Q(X)\cdot Q(-X)} \end{align}$

이 됩니다.

여기서 모든 power series $S(X) \in R[[X]]$는 unique한 even-odd decomposition

$\begin{align} S(X) = S _ {even}(X ^ 2) + X \cdot S _ {odd} (X ^ 2) \end{align}$

을 갖는다는 점에 주목합니다.

$\begin{align} U(X) &= P(X) \cdot Q(-X) \newline V(X) &= \left( Q(X) \cdot Q(-X) \right) _ {even} \end{align}$

라 정의하면, $Q(X)\cdot Q(-X)$가 even power series이므로 $V(X ^ 2) = Q(X)\cdot Q(-X)$이며,

$\begin{align} [X^N]\frac{P(X)\cdot Q(-X)}{Q(X)\cdot Q(-X)} &= [X ^ N] \frac{U _ {even}(X ^ 2) + X \cdot U _ {odd}(X^2)}{V(X^2)} \newline &= [X ^ N] \frac{U _ {even}(X ^ 2)}{V(X ^ 2)} + [X ^ N]X \cdot \frac{U _ {odd}(X ^ 2)}{V(X ^ 2)} \newline &= \begin{cases} [X ^ N] \frac{U _ {even}(X ^ 2)}{V(X ^ 2)} & \text{if $N$ is even} \newline [X ^ N]X \cdot \frac{U _ {odd}(X ^ 2)}{V(X ^ 2)} & \text{if $N$ is odd}  \end{cases} \newline &= \begin{cases} [X ^ {N / 2}] \frac{U _ {even}(X)}{V(X)} & \text{if $N$ is even} \newline [X ^ {(N - 1) / 2}]\frac{U _ {odd}(X)}{V(X)} & \text{if $N$ is odd}  \end{cases} \end{align}$

이 됩니다. $V(X)$는 degree가 $d$인 polynomial이며 $U _ {even} (X)$와 $U _ {odd}(X)$는 degree가 $d$미만인 polynomial이므로 $N$의 크기만 절반으로 줄어든 subproblem이 얻어집니다. 위 iteration은 $\lfloor \log (N + 1) \rfloor$번 반복되며 각 iteration마다 $U(X)$와 $V(X)$를 계산하는데에 각각 $M(d)$만큼의 ring operation이 필요되므로 전체 algorithm은

$\begin{align} 2 M(d) \cdot \lceil \log N \rceil + M(d) \end{align}$

만큼의 ring operation은 필요로 합니다. Kitasama method보다 대략 $2/3$정도로 빨라졌음을 알 수 있습니다.



<h2 id="optimization">Optimization under the DFT Setting</h2>

이제 위 algorithm을 polynomial multiplication이 fast fourier transformation algorithm로 행해진다고 가정하고 최적화해 보겠습니다.



<h3 id="dft">Review of DFT</h3>

$\mathbb{K}$가 primitive $n$-th root of unity $\omega _ n$를 갖는 field일 때, polynomial $A(X) \in \mathbb{K}[X]$에 대하여,

$\begin{align} DFT _ n (A(X)) :&= \left( A(w _ n ^ 0), A(w _ n ^ 1), \cdots , A(\omega _ n ^ {n-1}) \right) \in \mathbb{K} ^ n \newline IDFT _ n(A(X)) :&= \frac{1}{n} \left( A(\omega _ n ^ 0), A(\omega _ n ^ {-1}), \cdots , A(\omega _ n ^ {-(n-1)}) \right) \in \mathbb{K} ^ n \end{align}$

을 각각 $A(X)$의 **discrete fourier transform**, **inverse discrete fourier transform**이라 부릅니다. Degree가 $n$ 미만인 polynomial들의 set을 $\mathbb{K} ^ n$와 identify시켜보면, 임의의 $A(X) \in \mathbb{K} ^ n$에 대하여

$A(X) = IDFT _ n (DFT _ n(A(X))) = DFT _ n(IDFT _ n(A(X)))$

이 성립함이 알려져 있습니다.

$E(n)$을 degree $n$인 polynomial의 discrete fourier transform / inverse discrete fourier transform을 계산하는데 필요한 field operation의 수라고 합시다. **Fast fourier transformation**이라는 algorithm을 통해 $E(n) \in 9n \cdot \log n + O(n)$이 가능함이 알려져 있습니다.

$A(X),B(X),C(X) \in \mathbb{K}[X]$가 degree $n$미만인 $A(X) \cdot B(X) = C(X)$이 성립하는 polynomial들이라 합시다. 

$\begin{align} DFT _ n(C(X)) &= \left( C(w _ n ^ 0), C(w _ n ^ 1), \cdots , C(\omega _ n ^ {n-1}) \right) \newline &= \left( (A\cdot B)(w _ n ^ 0), (A\cdot B)(w _ n ^ 1), \cdots , (A \cdot B)(\omega _ n ^ {n-1}) \right) \newline &= \left( A(w _ n ^ 0), A(w _ n ^ 1), \cdots , A(\omega _ n ^ {n-1}) \right) \cdot \left( B(w _ n ^ 0), B(w _ n ^ 1), \cdots , B(\omega _ n ^ {n-1}) \right) \newline &= DFT _ n(A(X)) \cdot DFT _ n(B(X)) \end{align}$

이므로 polynomial multiplication에 필요한 field operation은 $M(n) \in 3 E(2n) + O(d)$임을 알 수 있습니다.



<h3 id="doubling">DFT Doubling</h3>

Degree $n$ 미만인 polynomial $A(X) \in \mathbb{K}[X]$에 대하여 $\hat{A} = DFT _ n(A(X))$를 알고있을 때 $DFT _ {2n}(A(X))$를 찾고 십다고 합시다. 생각할 수 있는 가장 간단한 방법은

$\begin{align} DFT _ {2n} \left( IDFT _ n \left( \hat{A} \right) \right) \end{align}$

을 계산하는 것입니다. 이 때, 총

$\begin{align} E(n)+E(2n) \in \frac{9}{2}n \cdot \log n + O(n) \end{align}$

만큼의 연산이 필요합니다. 그런데 임의의 $0 \le i < n$에 대하여

$\begin{align} DFT _ {2n}(A(X)) _ {2i} &= \sum _ {j = 0} ^ {2n - 1} A _ i \cdot \omega _ {2n} ^ {2ij} \newline &= \sum _ {j = 0} ^ {n - 1} A _ i \cdot \omega _ n ^ {ij} \newline &= \hat{A} _ i \end{align}$

이며

$\begin{align} DFT _ {2n}(A(X)) _ {2i+1} &= \sum _ {j = 0} ^ {2n - 1} A _ i \cdot \omega _ {2n} ^ {i(2j+1)} \newline &= w _ {2n} ^ i \cdot \sum _ {j = 0} ^ {n - 1} A _ i \cdot \omega _ n ^ {ij} \newline &= \omega _ {2n} ^ i \cdot \hat{A} _ i \end{align}$

이므로,

$2E(n) + 2n \in 3n \cdot \log n + O(n)$

만큼의 field operation으로 DFT doubling을 할 수 있음을 알 수 있습니다.



<h3 id="fast-bostan-mori">Bostan-Mori Algorithm under the DFT Setting</h3>

$k$를 $2 ^ K \ge 2d + 1$인 최소의 integer라고 하겠습니다.

각 iteration마다 $P(X)$와 $Q(X)$를 직접적으로 저장하는 대신 $\hat{P} = DFT _ {2 ^ k}(P(X))$와 $\hat{Q} = DFT _ {2 ^ k}(Q(X))$를 저장하도록 하겠습니다. 이제 Bostan-Mori algorithm에서 $N$이 짝수일 경우, $DFT _ {2 ^ k}(U _ {even} (X))$와 $DFT _ {2 ^ k}(V(X))$를 계산해야 하고, 홀수일 경우, $DFT _ {2 ^ k}(U _ {odd}(X))$와 $DFT _ {2 ^ k}(V(X))$를 계산해야 합니다.

임의의 degree $d$ 미만의 polynomial $A(X),B(X) \in \mathbb{K}[X]$에 대하여, $\hat A = DFT _ {2 ^ k}(A(X))$와 $\hat B = DFT _ {2 ^ k}(B(X))$가 주어졌다고 합시다. 이 때, 임의의 $0 \le i < 2 ^ k$에 대하여

$\begin{align} DFT _ {2 ^ k}(A(-X)) _ i &= \sum _ {j = 0} ^ {2 ^ k - 1} A _ j \cdot  (-\omega _ {2 ^ k} ^ i) ^ j \newline &= \sum _ {j = 0} ^ {2 ^ k - 1} A _ j \cdot \omega _ {2 ^ k} ^ {(i \oplus 2 ^ {k - 1}) \cdot j} \newline &= \hat A _ {i \oplus 2 ^ {k - 1}} \end{align}$

입니다. (단, positive integer $x, y$에 대하여 $x \oplus y$는 $x$와 $y$의 bitwise xor연산을 나타냅니다.)

또한 임의의 $0 \le i < 2 ^ {k - 1}$에 대하여

$\begin{align} DFT _ {2 ^ {k - 1}}(A _ {even}(X)) _ i &= \sum _ {j = 0} ^ {2 ^ {k - 1} - 1} A _ {2j} \omega _ {2 ^ {k - 1}} ^ {ij} \newline &= \sum _ {j = 0} ^ {2 ^ {k - 1} - 1} A _ {2j} \omega _ {2 ^ k} ^ {i(2j)} \newline &= \frac12 \sum _ {j = 0} ^ {2 ^ k - 1} A _ {j} \left( \omega _ {2 ^ k} ^ {ij} + \left( -\omega _ {2 ^ k} ^ i \right) ^ {j} \right) \newline &= \frac12 \left( \hat A _ {i} + \hat A _ {i + 2 ^ {k - 1}} \right) \end{align}$

이고

$\begin{align} DFT _ {2 ^ {k - 1}}(A _ {odd}(X)) _ i &= \sum _ {j = 0} ^ {2 ^ {k - 1} - 1} A _ {2j + 1} \omega _ {2 ^ {k - 1}} ^ {ij} \newline &= \omega _ {2 ^ k} ^ {-i} \sum _ {j = 0} ^ {2 ^ {k - 1} - 1} A _ {2j + 1} \omega _ {2 ^ k} ^ {i(2j+1)} \newline &= \omega _ {2 ^ k} ^ {-i} \sum _ {j = 0} ^ {2 ^ k - 1} A _ j \left( \omega _ {2 ^ k} ^ {ij} - \left( - \omega _ {2 ^ k} ^ i \right) ^ j \right) \newline &= \omega _ {2 ^ k} ^ {-i} ( \hat A _ i - \hat A _ {i + 2 ^ {k - 1}}) \end{align}$

입니다. 따라서,

1. $\hat P$와 $\hat Q$로부터 $DFT _ {2 ^ k} (U(X))$를 얻어내는데 $2 ^ k$번
2. $DFT _ {2 ^ k} (U(X))$로부터 $DFT _ {2 ^ {k - 1}} (U _ {even}(X))$ 또는 $DFT _ {2 ^ {k - 1}} (U _ {odd}(X))$를 얻어내는데 $2 ^ {k - 1}$번
3. DFT doubling을 통해 $DFT _ {2 ^ {k - 1}} (U _ {even}(X))$ 또는 $DFT _ {2 ^ {k - 1}} (U _ {odd}(X))$로부터 $DFT _ {2 ^ k} (U _ {even}(X))$ 또는 $DFT _ {2 ^ k} (U _ {odd}(X))$를 얻어내는데 $2E(2 ^ {k - 1})+2 ^ k$번
4. $\hat Q$로부터 $DFT _ {2 ^ k}(Q(X) \cdot Q(-X))$를 얻어내는데 $2 ^ k$번
5. $DFT _ {2 ^ k}(Q(X) \cdot Q(-X))$로 부터 $DFT _ {2 ^ {k-1}}(V(X))$를 얻어내는데 $2 ^ {k - 1}$번
6. $DFT _ {2 ^ {k-1}}(V(X))$로 부터 $DFT _ {2 ^ k}(V(X))$를 얻어내는데 $2E(2 ^ {k - 1}) + 2 ^ k$번

로서 총

$(4E(2 ^ {k - 1}) + O(2 ^ k)) \cdot \lfloor \log N \rfloor$

의 field operation을 필요로 합니다. $d = 2 ^ l - 1$꼴일 경우, 위 식은

$4 E(d) \cdot \log N + O(d \cdot \log N)$

으로 간략화되며, 이는

$\begin{align} \frac23 M(d) \cdot \log N + O(d \cdot \log N) \end{align}$

와 같습니다. Kitasama method와 비교했을때 상수가 엄청 작아졌음을 확인할 수 있습니다.

다음은 위 algorithm의 C++ implementation입니다. $\mathbb{K} = \mathbb{Z} / 998244353\mathbb{Z}$를 가정합니다.

```cpp
template<class T, int primitive_root = 3>
void number_theoric_transform(vector<T> &a, const bool invert = false){
	int n = (int)a.size();
	assert(n && __builtin_popcount(n) == 1 && (T().mod() - 1) % n == 0);
	const T root = T(primitive_root) ^ (T().mod() - 1) / n;
	const T inv_root = T(1) / root;
	for(auto i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(a[i], a[j]);
	}
	for(auto len = 1; len < n; len <<= 1){
		T wlen = (invert ? inv_root : root) ^ n / len / 2;
		for(auto i = 0; i < n; i += len << 1){
			T w = 1;
			for(auto j = 0; j < len; ++ j, w *= wlen){
				T u = a[i + j], v = a[i + j + len] * w;
				a[i + j] = u + v, a[i + j + len] = u - v;
			}
		}
	}
	if(invert){
		T inv_n = T(1) / n;
		for(auto &x: a) x *= inv_n;
	}
}

template<class T, int primitive_root = 3>
vector<T> convolute_ntt(vector<T> p, vector<T> q){
	if(min(p.size(), q.size()) < 60){
		vector<T> res((int)p.size() + (int)q.size() - 1);
		for(auto i = 0; i < (int)p.size(); ++ i) for(auto j = 0; j < (int)q.size(); ++ j) res[i + j] += p[i] * q[j];
		return res;
	}
	int m = (int)p.size() + (int)q.size() - 1, n = 1 << __lg(m) + 1;
	p.resize(n), q.resize(n);
	number_theoric_transform<T, primitive_root>(p), number_theoric_transform<T, primitive_root>(q);
	for(auto i = 0; i < n; ++ i) p[i] *= q[i];
	number_theoric_transform<T, primitive_root>(p, true);
	p.resize(m);
	return p;
}

template<class T, int primitive_root = 3>
void double_up_ntt(vector<T> &p){
	int n = (int)size(p);
	assert(n && __builtin_popcount(n) == 1 && (T().mod() - 1) % (n << 1) == 0);
	vector<T> res(n << 1);
	for(auto i = 0; i < n; ++ i) res[i << 1] = p[i];
	number_theoric_transform(p, true);
	T w = T(primitive_root) ^ (T().mod() - 1) / n / 2, pw = 1;
	for(auto i = 0; i < n; ++ i, pw *= w) p[i] *= pw;
	number_theoric_transform(p);
	for(auto i = 0; i < n; ++ i) res[i << 1 | 1] = p[i];
	swap(p, res);
}

template<class T, class U, int primitive_root = 3>
T rational_polynomial_single_term_extraction(vector<T> p, vector<T> q, U i){
	assert(!q.empty() && q[0] != 0);
	int n = 1 << __lg((int)size(q) << 1 | 1) + 1;
	assert(i >= 0 && size(p) < size(q) && (T().mod() - 1) % n == 0);
	p.resize(n), q.resize(n);
	number_theoric_transform<T, primitive_root>(p);
	number_theoric_transform<T, primitive_root>(q);
	T inv2 = (T().mod() + 1) / 2;
	for(T w = 1 / T(primitive_root) ^ (T().mod() - 1) / n; i; i >>= 1){
		for(auto i = 0; i < n; ++ i) p[i] *= q[i ^ n / 2];
		if(~i & 1){
			for(auto i = 0; i < n / 2; ++ i) p[i] = inv2 * (p[i] + p[i ^ n / 2]);
		}
		else{
			T pw = inv2;
			for(auto i = 0; i < n / 2; ++ i, pw *= w) p[i] = pw * (p[i] - p[i ^ n / 2]);
		}
		p.resize(n / 2);
		double_up_ntt<T, primitive_root>(p);
		q.resize(n / 2);
		for(auto i = 0; i < n / 2; ++ i) q[i] *= q[i ^ n / 2];
		double_up_ntt<T, primitive_root>(q);
	}
	return accumulate(begin(p), end(p), T(0)) / accumulate(begin(q), end(q), T(0));
}

template<class T, int primitive_root = 3>
struct linear_recurrence_bostan_mori{
	int n;
	vector<T> p, q;
	linear_recurrence_bostan_mori(const vector<T> &init, const vector<T> &coef): n((int)coef.size()), p(n), q(n + 1, 1){
		assert((int)coef.size() == (int)init.size());
		for(auto i = 0; i < n; ++ i) q[n - i] = -coef[i];
		p = convolute_ntt<T, primitive_root>(init, q);
		p.resize(n);
	}
	template<class U>
	T operator[](U i) const{
		assert(0 <= i);
		return rational_polynomial_single_term_extraction<T, U, primitive_root>(p, q, i);
	}
};
```



<h2 id="benchmarking">Benchmarking</h2>

다음은 $d = 10^2, 10^3, 10^4, 10^5$이고 $N = 10^6, 10^{10}, 10 ^ {14}, 10 ^ {18}$일 때, 두 implementation의 속도를 초단위로 비교한 결과입니다. 좌측이 Kitasama method이고 우측이 Bostan-Mori algorithm입니다. Bostan-Mori algorithm이 항상 빠르다는 것을 관찰할 수 있습니다.

| $d \backslash N$ | $10^6$                       | $10 ^ {10}$                  | $10 ^ {14}$                  | $10 ^ {18}$                  |
| ---------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| $10^2$           | $0.00296 \backslash 0.00027$ | $0.00169 \backslash 0.00037$ | $0.00254 \backslash 0.00051$ | $0.00328 \backslash 0.00063$ |
| $10^3$           | $0.01187 \backslash 0.00216$ | $0.01850 \backslash 0.00373$ | $0.02672 \backslash 0.00520$ | $0.03591 \backslash 0.00616$ |
| $10^4$           | $0.25845 \backslash 0.04943$ | $0.42474 \backslash 0.08236$ | $0.60561 \backslash 0.10842$ | $0.79087 \backslash 0.13828$ |
| $10^5$           | $2.57779 \backslash 0.48705$ | $4.20486 \backslash 0.78568$ | $6.01462 \backslash 1.09658$ | $8.12139 \backslash 1.35484$ |

다음은 위 데이터를 얻는데 사용한 C++ 코드입니다.

```cpp
#include <bits/stdc++.h>
using namespace std;
#if __cplusplus > 201703L
#include <ranges>
using namespace numbers;
#endif

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

template<class T, int primitive_root = 3>
void number_theoric_transform(vector<T> &a, const bool invert = false){
	int n = (int)a.size();
	assert(n && __builtin_popcount(n) == 1 && (T().mod() - 1) % n == 0);
	const T root = T(primitive_root) ^ (T().mod() - 1) / n;
	const T inv_root = T(1) / root;
	for(auto i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(a[i], a[j]);
	}
	for(auto len = 1; len < n; len <<= 1){
		T wlen = (invert ? inv_root : root) ^ n / len / 2;
		for(auto i = 0; i < n; i += len << 1){
			T w = 1;
			for(auto j = 0; j < len; ++ j, w *= wlen){
				T u = a[i + j], v = a[i + j + len] * w;
				a[i + j] = u + v, a[i + j + len] = u - v;
			}
		}
	}
	if(invert){
		T inv_n = T(1) / n;
		for(auto &x: a) x *= inv_n;
	}
}

template<class T, int primitive_root = 3>
vector<T> convolute_ntt(vector<T> p, vector<T> q){
	if(min(p.size(), q.size()) < 60){
		vector<T> res((int)p.size() + (int)q.size() - 1);
		for(auto i = 0; i < (int)p.size(); ++ i) for(auto j = 0; j < (int)q.size(); ++ j) res[i + j] += p[i] * q[j];
		return res;
	}
	int m = (int)p.size() + (int)q.size() - 1, n = 1 << __lg(m) + 1;
	p.resize(n), q.resize(n);
	number_theoric_transform<T, primitive_root>(p), number_theoric_transform<T, primitive_root>(q);
	for(auto i = 0; i < n; ++ i) p[i] *= q[i];
	number_theoric_transform<T, primitive_root>(p, true);
	p.resize(m);
	return p;
}

template<class T>
vector<T> operator*(const vector<T> &p, const vector<T> &q){
	return convolute_ntt(p, q);
}
template<class T>
vector<T> &operator*=(vector<T> &a, const vector<T> &b){
	return a = a * b;
}
template<class T>
vector<T> &operator-=(vector<T> &a, const vector<T> &b){
	if(a.size() < b.size()) a.resize(b.size());
	for(auto i = 0; i < (int)b.size(); ++ i) a[i] -= b[i];
	return a;
}
// Returns the first length terms of the inverse of a
template<class T>
vector<T> inverse(const vector<T> &a, int length){
	assert(!a.empty() && a[0]);
	static vector<T> b;
	b = {1 / a[0]};
	while((int)b.size() < length){
		static vector<T> x;
		x.assign(min(a.size(), b.size() << 1), 0);
		copy(a.begin(), a.begin() + x.size(), x.begin());
		x *= b * b;
		b.resize(b.size() << 1);
		for(auto i = (int)b.size() >> 1; i < (int)min(x.size(), b.size()); ++ i) b[i] = -x[i];
	}
	b.resize(length);
	return b;
}

template<class T, int primitive_root = 3>
struct linear_recurrence_solver_kitamasa{
	int n;
	vector<T> init, coef, q, inv;
	linear_recurrence_solver_kitamasa(const vector<T> &init, const vector<T> &coef): n((int)coef.size()), init(init), coef(coef){
		assert((int)coef.size() == (int)init.size());
		for(auto &x: coef) q.push_back(-x);
		q.push_back(1);
		inv = inverse(vector<T>(rbegin(q), rend(q)), n + 1);
	}
	template<class U>
	T operator[](U i) const{
		assert(0 <= i);
		if(n == 0) return 0;
		auto merge = [&](const vector<T> &a, const vector<T> &b){
			auto res = convolute_ntt<T, primitive_root>(a, b);
			auto t = convolute_ntt<T, primitive_root>(vector<T>(rbegin(res), rbegin(res) + n + 1), inv);
			t.resize(n + 1);
			reverse(begin(t), end(t));
			res -= t * q;
			res.resize(n + 1);
			return res;
		};
		vector<T> power(n + 1), base(n + 1);
		for(power[0] = base[1] = 1; i; i >>= 1, base = merge(base, base)) if(i & 1) power = merge(power, base);
		T res = 0;
		for(auto i = 0; i < n; ++ i) res += power[i] * init[i];
		return res;
	}
};

template<class T, int primitive_root = 3>
void double_up_ntt(vector<T> &p){
	int n = (int)size(p);
	assert(n && __builtin_popcount(n) == 1 && (T().mod() - 1) % (n << 1) == 0);
	vector<T> res(n << 1);
	for(auto i = 0; i < n; ++ i) res[i << 1] = p[i];
	number_theoric_transform(p, true);
	T w = T(primitive_root) ^ (T().mod() - 1) / n / 2, pw = 1;
	for(auto i = 0; i < n; ++ i, pw *= w) p[i] *= pw;
	number_theoric_transform(p);
	for(auto i = 0; i < n; ++ i) res[i << 1 | 1] = p[i];
	swap(p, res);
}

template<class T, class U, int primitive_root = 3>
T rational_polynomial_single_term_extraction(vector<T> p, vector<T> q, U i){
	assert(!q.empty() && q[0] != 0);
	int n = 1 << __lg((int)size(q) << 1 | 1) + 1;
	assert(i >= 0 && size(p) < size(q) && (T().mod() - 1) % n == 0);
	p.resize(n), q.resize(n);
	number_theoric_transform<T, primitive_root>(p);
	number_theoric_transform<T, primitive_root>(q);
	T inv2 = (T().mod() + 1) / 2;
	for(T w = 1 / T(primitive_root) ^ (T().mod() - 1) / n; i; i >>= 1){
		for(auto i = 0; i < n; ++ i) p[i] *= q[i ^ n / 2];
		if(~i & 1){
			for(auto i = 0; i < n / 2; ++ i) p[i] = inv2 * (p[i] + p[i ^ n / 2]);
		}
		else{
			T pw = inv2;
			for(auto i = 0; i < n / 2; ++ i, pw *= w) p[i] = pw * (p[i] - p[i ^ n / 2]);
		}
		p.resize(n / 2);
		double_up_ntt<T, primitive_root>(p);
		q.resize(n / 2);
		for(auto i = 0; i < n / 2; ++ i) q[i] *= q[i ^ n / 2];
		double_up_ntt<T, primitive_root>(q);
	}
	return accumulate(begin(p), end(p), T(0)) / accumulate(begin(q), end(q), T(0));
}

template<class T, int primitive_root = 3>
struct linear_recurrence_solver_bostan_mori{
	int n;
	vector<T> p, q;
	linear_recurrence_solver_bostan_mori(const vector<T> &init, const vector<T> &coef): n((int)coef.size()), p(n), q(n + 1, 1){
		assert((int)coef.size() == (int)init.size());
		for(auto i = 0; i < n; ++ i) q[n - i] = -coef[i];
		p = convolute_ntt<T, primitive_root>(init, q);
		p.resize(n);
	}
	template<class U>
	T operator[](U i) const{
		assert(0 <= i);
		return rational_polynomial_single_term_extraction<T, U, primitive_root>(p, q, i);
	}
};

struct timer{
	chrono::time_point<chrono::high_resolution_clock> init = chrono::high_resolution_clock::now(), current = chrono::high_resolution_clock::now();
	void refresh(){
		current = chrono::high_resolution_clock::now();
	}
	void measure(){
		cerr << "Time Passed: " << chrono::duration<double>(chrono::high_resolution_clock::now() - current).count() << endl;
		current = chrono::high_resolution_clock::now();
	}
};

int main(){
	cin.tie(0)->sync_with_stdio(0);
	cin.exceptions(ios::badbit | ios::failbit);
	mt19937 rng(156485);
	cerr << fixed << setprecision(5);
	timer t;
	modular V1, V2;
	for(auto d: {100, 1000, 10000, 100000}){
		for(auto n: vector<long long>{1e6, 1e10, 1e14, 1e18}){
			vector<modular> init(d), coef(d);
			for(auto i = 0; i < d; ++ i) init[i] = rng(), coef[i] = rng();
			cerr << "d = " << d << ", n = " << n << endl;
			t.refresh();
			V1 = linear_recurrence_solver_kitamasa(init, coef)[n];
			t.measure();
			V2 = linear_recurrence_solver_bostan_mori(init, coef)[n];
			t.measure();
			assert(V1 == V2);
			cerr << endl;
		}
	}
	return 0;
}
```

