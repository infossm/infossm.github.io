---
layout: post
title:  "N! mod P 의 빠른 계산"
author: ho94949
date: 2019-09-17 15:00
tags: [Factorial, FFT, Lagrange Interpolation]
---

# 서론

  $$N!$$ 은 1이상 $$N$$ 이하의 모든 정수를 곱한 수 이다. 이 $$N!$$는 다양한 조합론적 상황에서 많이 사용된다. 이 $$N!$$을 특정한 소수 $$P$$로 나눈 나머지를 빠르게 ($$O(\sqrt{N} \log{N})$$ 시간에) 계산 하는 방법에 대해서 알아본다.

## Naive

  팩토리얼을 구하는 가장 쉬운 방법은 모든 1이상 $$N​$$ 이하의 수를 곱해서 $$P​$$ 로 나누는 것이다. $$ab​$$를 $$P​$$로 나눈 나머지와 $$(a \bmod P)(b \bmod P) \bmod P​$$ 가 같다는 것을 이용하면, 사이에 나오는 숫자를 항상 $$P^2​$$ 이하로 유지하면서 쉽게 구할 수 있다.

# Attacks 

우리는 팩토리얼을 빠른 시간에 구하기 위해서 다양한 공격들을 활용할 것이다. 일단, 우리는 일반적인 형태의 다항식 $$f_d(x) = (dx+1)(dx+2) \cdots (dx+d)$$ 를 생각할 것이다. 

이 다항식의 다양한 성질들은 다음과 같다.

- $$f_d(x)$$ 는 $$d$$ 차 다항식임이 자명하다.

- $$f_N(0) = N!​$$
  - $$x=0​$$을 대입하면, $$Nx=0​$$ 이기 때문에, $$f_N(0) = 1 \times 2 \times \cdots \times N = N!​$$ 이 된다.
- $$f_a(0) f_a(1) \cdots f_a(b-1) = f_{ab}(0)$$
  - $$f_a(i)$$ 가 $$ai+1$$ 부터 $$ai+a$$ 까지의 곱이기 때문에, 위의 식은 1부터 $$ab$$ 까지의 곱이라는것을 쉽게 알 수 있다.
- $$f_d(2x)f_d(2x+1) = f_{2d}(x)​$$
  - 좌변과 우변 모두 $$(2dx+1)(2dx+2) \cdots (2dx+2d)​$$ 로 같다.

이 다항식의 성질 1을 이용하면, 우리는 $$N!​$$을 구하기 위해서, $$f_N(0)​$$ 를 구하면 되는 것을 알 수 있고, 만약 우리가 적당한 $$v \sim \sqrt{N}​$$ 정도의 수를 잡고 $$f_v(0), f_v(1), \cdots, f_v(v)​$$ 를 계산 할 수 있다면, $$f_v(v+1)(0)​$$을 계산 할 수 있어서, 나머지는 $$v(v+1)+1​$$ 부터 $$N​$$ 까지의 수는 직접 곱해주는 방법으로 구할 수 있다는 것을 알 수 있다. 이 함수값들을 계산하는 시간 복잡도를 $$T(v)​$$ 라고 하면, 나머지 과정은 $$O(v)​$$ 만큼의 곱셈과 나머지 연산 뿐이므로, $$T(v) \ge v​$$인 일반적인 상황 하에서 전체 시간복잡도는 $$T(v)  = T(\sqrt{N})​$$이 된다.

## Multipoint evaluation

가장 처음 알 수 있는 것은, 다항식이 모두 같은 형태이기 때문에 Multipoint evaluation을 이용할 수 있다는 점이다. Multipoint evaluation에 대한 게시글은 [다항식 나눗셈과 다중계산](http://www.secmem.org/blog/2019/06/16/Multipoint-evaluation/) 문서에 매우 잘 정리가 되어있기 때문에 참고하면 된다. 이 때 시간복잡도는 $$T(v) = O(v \log^2 v)$$ 이기 때문에, $$O( \sqrt{N} \log^2{N})$$ 이 된다.

## Lagrange Polynomial

어떤 $$d​$$차 다항함수를 표현하는 가장 쉬운 방법은 각 다항식의 계수만 저장하는 것이다. 즉 $$ h(x) = a_d x^d + a_{d-1} x^{d-1} + \cdots + a_0 x^0​$$ 에서 $$a_d, a_{d-1}, \cdots , a_{0}​$$ 만 저장한 경우 다항식을 쉽게 계산할 수 있다. 하지만, 이 이외에도 다항식을 표현하는 방법이 있는데 그것은 Lagrange Interpolation이다.

Lagrange Interpolation은 $$h(0), h(1), \cdots, h(d)​$$ 의 값을 대신 들고 있는 것이다. 일단 이 값들로 $$d​$$차 다항식이 유일하게 결정될 수 있음을 증명하겠다.

pf) (존재성)

  $$h(x) = \sum_{i=0}^{d} h(i) \prod_{j=0, i \neq j}^{d} \frac{x-j}{i-j}​$$ 는 $$d​$$차 다항식이고, 0이상 $$d​$$ 이하의 $$x​$$에 대해서, $$h(x)​$$가 있는 합 항을 제외하고는 뒤의 $$x-j​$$ 항에 의해 모두 0이 되기 때문에, $$h(x)​$$ 항만 남는다.

pf) (유일성)

  만약 다항식이 유일하지 않다고 가정하고, 두 $$d$$차 다항식 $$h_1, h_2$$가 존재하여 $$h_1(0) = h_2(0), h_1(1) = h_2(1), \cdots, h_1(d) = h_2(d)$$ 라고 하자. 그럼 $$\delta h(x) = h_1(x)- h_2$$는 $$d$$차 다항식이며, $$0, 1, \cdots, d$$의 $$d+1$$ 개의 해를 가지며, $$h_1 \neq h_2$$ 이기 때문에 $$\delta h(x) \not \equiv 0$$이다. 이는 대수학의 기본정리에 모순이다.

## Lagrange Polynomial의 빠른 계산

우리는 이 Lagrange Polynomial을 빠른 시간 내에 계산하고 싶다. 즉, $$h(d+1), h(d+2), \cdots$$ 를 계산하고 싶다.

  $$h(x) = \sum_{i=0}^{d} h(i) \prod_{j=0, i \neq j}^{d} \frac{x-j}{i-j} = \left( \prod_{j=0}^{d}(x-j) \right) \left(\sum_{i=0}^d \frac{h(i)}{i!(d-i)!(-1)^{d-i}} \times \frac{1}{x-i}\right) $$ 이기 때문에, $$h(d+1), \cdots$$ 를 계산 하기 위해서는 앞의  $$\ \prod_{j=0}^{d}(x-j)$$ 부분을 Sliding window 등의 방법으로, 뒤의 항은 합성곱 형태이기 때문에, FFT다항식 곱셈으로 구해주면 된다. (이는 위의 다항식 나눗셈과 다중계산 문서에도 설명이 되어있다.)

## $O(\sqrt{N} \log{N})$에 계산하기

그래서, $$h=f_d​$$라 보고, $$h(d+1), \cdots, h(4d+1)​$$을 구하면, $$f_d(0), f_d(1), \cdots, f_d(4d+1)​$$ 을 구하게 되는 것인데, $$f_{2d}(0) = f_d(0) f_d(1), f_{2d}(1) = f_d(2)f_d(3), \cdots, f_{2d}(2d) = f_d(4d)f_d(4d+1)​$$ 이기 때문에, $$f_{2d}(0), f_{2d}(1), \cdots, f_{2d}(2d)​$$ 를 $$O(d \log d)​$$ 시간 안에 구할 수 있다.

즉 $$d$$를 2배 하는데 $$O(d \log d)$$ 시간이 걸리기 때문에, 원하는 $$d$$까지 늘리는데 드는 시간은 Master정리를 사용하면 역시 $$O(d \log d)$$ 시간이 든다는 것을 알 수 있고 이는 $$N$$ 에 대해서는, $$O(\sqrt{N} \log{N})$$ 시간이다.

# 구현

다음은 과제 작성을 위해서 [Baekjoon Online Judge 17467](https://www.acmicpc.net/problem/17467) 에 만든 문제와, 이에 대한 구현이다.

```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = int64_t;
using u64 = uint64_t;
using base = complex<double>;

void fft(vector<base> &a, bool inv){
	int n = a.size(), j = 0;
	for(int i=1; i<n; i++){
		int bit = (n >> 1);
		while(j >= bit){
			j -= bit;
			bit >>= 1;
		}
		j += bit;
		if(i < j) swap(a[i], a[j]);
	}
	double ang = 2 * acos(-1) / n * (inv ? -1 : 1);
	
	vector<base> roots(n/2);

	for(int i=0; i<n/2; i++)
		roots[i] = base(cos(ang * i), sin(ang * i));

	for(int i=2; i<=n; i<<=1){
		int step = n / i;
		for(int j=0; j<n; j+=i){
			for(int k=0; k<i/2; k++){
				base u = a[j+k], v = a[j+k+i/2] * roots[step * k];
				a[j+k] = u+v;
				a[j+k+i/2] = u-v;
			}
		}
	}
	if(inv) for(int i=0; i<n; i++) a[i] /= n; 
}

vector<i64> multiply(vector<i64> &v, vector<i64> &w, i64 mod){
	int n = 2; while(n < v.size() + w.size()) n <<= 1;
	vector<base> v1(n), v2(n), r1(n), r2(n);
	for(int i=0; i<v.size(); i++)
		v1[i] = base(v[i] >> 15, v[i] & 32767);
	for(int i=0; i<w.size(); i++)
		v2[i] = base(w[i] >> 15, w[i] & 32767);
	fft(v1, 0);
	fft(v2, 0);
	for(int i=0; i<n; i++){
		int j = (i ? (n - i) : i);
		base ans1 = (v1[i] + conj(v1[j])) * base(0.5, 0);
		base ans2 = (v1[i] - conj(v1[j])) * base(0, -0.5);
		base ans3 = (v2[i] + conj(v2[j])) * base(0.5, 0);
		base ans4 = (v2[i] - conj(v2[j])) * base(0, -0.5);
		r1[i] = (ans1 * ans3) + (ans1 * ans4) * base(0, 1);
		r2[i] = (ans2 * ans3) + (ans2 * ans4) * base(0, 1);
	}
	fft(r1, 1);
	fft(r2, 1);
	vector<i64> ret(n);
	for(int i=0; i<n; i++){
		i64 av = (i64)round(r1[i].real());
		i64 bv = (i64)round(r1[i].imag()) + (i64)round(r2[i].real());
		i64 cv = (i64)round(r2[i].imag());
		av %= mod, bv %= mod, cv %= mod;
		ret[i] = (av << 30) + (bv << 15) + cv;
		ret[i] %= mod;
		ret[i] += mod;
		ret[i] %= mod;
	}
	return ret;
}

/*
Input h: vector of length d+1: h(0), h(1), h(2), ..., h(d) where h is d-degree polynomial
Input P: prime, where h degree polynomial is represented as modulo 
Output: vector of length 4d+2: h(0), h(1), h(2), ..., h(4d+1)

Lagrange interpolation is given as follows:

h(x) = sum_(i=0 to d)[ h(i) prod(j=0 to d, i != j){(x-j)/(i-j)} ]
     = {prod(j=0 to d)(x-j)} * 
	   sum(i=0 to d){h(i)/(i!(d-i)!(-1)^(d-i)) * 1/(x-i) }

Letting f(i) = h(i)/(i!(d-i)!(-1)^(d-i)), g(x) = prod(j=0 to d)(x-j)
h(x) = g(x) * sum(i=0 to d){f(i) * 1/(x-i)} which is easily calculated by convolution.

f(0)...f(d) and 1/1, 1/2, ..., 1/(4d+1) will be fed to multiply logic

*/
vector<i64> lagrange(vector<i64> h, i64 P) {
	int d = (int)h.size()-1; assert(d >= 0);
	auto mul = [P](i64 a, i64 b){
		return a*b%P;
	};
	
	auto ipow = [&mul](i64 a, i64 b) {
		i64 res = 1;
		while(b) {
			if(b&1) res = mul(res, a);
			a = mul(a, a);
			b >>= 1;
		}
		return res;
	};
	
	auto modInv = [&ipow, P](i64 a) {
		return ipow(a, P-2);
	};
	
	vector<i64> fact(4*d+2), invfact(4*d+2);
	fact[0] = 1;
	for(int i=1; i<=4*d+1; ++i)
		fact[i] = mul(i,fact[i-1]);
	
	invfact[4*d+1] = modInv(fact[4*d+1]);
	for(int i=4*d; i>=0; --i)
		invfact[i] = mul(invfact[i+1], i+1);
	
	vector<i64> f(d+1);
	for(int i=0; i<=d; ++i) {
		f[i] = h[i];
		f[i] = mul(invfact[i], f[i]);
		f[i] = mul(invfact[d-i], f[i]);
		if((d-i)%2==1) f[i] = P-f[i];
		if(f[i] == P) f[i] = 0;
	}
	
	vector<i64> inv(4*d+2);
	for(int i=1; i<4*d+2; ++i)
		inv[i] = mul(fact[i-1], invfact[i]);
	
	vector<i64> g(4*d+2);
	g[d+1] = 1;
	for(int j=0; j<=d; ++j) g[d+1] = mul(g[d+1], d+1-j);
	for(int i=d+2; i<4*d+2; ++i)
		g[i] = mul(g[i-1], mul(i, inv[i-d-1]));

	vector<i64> conv = multiply(f, inv, P);

	vector<i64> ret(4*d+2);
	for(int i=0; i<=d; ++i) ret[i] = h[i];
	for(int i=d+1; i<4*d+2; ++i) ret[i] = mul(g[i], conv[i]);

	return ret;
}
vector<i64> squarepoly(vector<i64> poly, i64 P) {
	vector<i64> ss = lagrange(poly, P);
	vector<i64> ret(ss.size()/2);
	auto mul = [P](i64 a, i64 b){
		return a*b%P;
	};
	for(int i=0; i<(int)ss.size()/2; ++i)
		ret[i] = mul(ss[2*i], ss[2*i+1]);
	return ret;
}
int main() {
	i64 N, P; cin >> N >> P;
	if(P < 100)
  {
    i64 ans = 1;
    for(int i=1; i<=N; ++i)
    {
      ans = ans*i%P;
    }
    cout << ans << endl;
    return 0;
  }
  
  i64 d = 1;
	vector<i64> fact_part = {1%P, 2%P};
	while(N > d*(d+1)) {
		fact_part = squarepoly(fact_part, P);
		d *= 2;
	}
  
	auto mul = [P](i64 a, i64 b) {
		return a*b%P;
	};
	
	i64 ans = 1, bucket = N/d;
	for(int i=0; i<bucket; ++i) ans = mul(ans, fact_part[i]);
	for(i64 i=bucket*d+1; i<=N; ++i) ans = mul(ans, i);
	
	cout << ans << endl;
}
```

