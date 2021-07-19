---
layout: post
title: Factor Base를 이용한 "어려운 문제"의 속도 개선
date: 2021-06-20 00:01
author: kipa00
tags: [algorithm]
---

이 글에서는 factor base를 활용한 속도 개선을 언제 사용할 수 있는지, 사용할 경우 시간 복잡도의 개선이 정확히 얼마나 일어나는지, 실제로는 얼마나 빠른지를 알아봅니다.

# "어려운 문제"?

이 글에서의 **어려운 문제**는 크게 소인수분해와 이산 로그를 일컫습니다.

소인수분해 문제는 잘 알고 계실 것입니다.
> 수 $N$이 주어지면, $N$을 (서로 다를 필요는 없는) 소수 $p_{1}$, $p_{2}$, $\cdots$, $p_{n}$의 곱 $N = p_{1} p_{2} \cdots p_{n}$으로 분해하라.

예를 들어 $108 = 2 \cdot 2 \cdot 3 \cdot 3 \cdot 3 = 2^{2} \cdot 3^{3}$과 같은 식으로 분해할 수 있습니다. 조금 더 복잡한 예제로, $10001 = 73 \cdot 137$과 같이 분해됩니다. $n = 1$이어서 "분해할 수 없는" 경우도 있습니다. 소인수분해는 (순서를 무시하면) 유일하다는 것도 다들 잘 알고 계실 것입니다.

이산로그 문제는 다음과 같습니다.
> 고정된 수 $N$에 대해 $\mathbb{Z} _ {N}^{\times}$의 생성자인 $g$와 $x \in \mathbb{Z} _ {N}^{\times}$가 주어지면, $g^{n} = x$인 $\log_{g} x := n \in \mathbb{Z}/\phi(N)\mathbb{Z}$을 찾아라.

정의를 차근차근 이해해 봅시다. 먼저 $\mathbb{Z} _ {N}^{\times}$는 $N$과 서로소인 $N$ 미만의 음이 아닌 정수들을 모은 [군](https://en.wikipedia.org/wiki/Group_(mathematics))입니다. 왜 서로소만을 모았냐면, 위에 있는 작은 기호가 시사하듯, 연산으로 곱하기 $\times$를 사용할 것이기 때문입니다. 다만 모든 원소가 $N$ 미만이기 때문에, $x \times y := x \cdot_{\mathbb{Z}} y \mod N$이 되도록 정의합니다. ($\cdot_{\mathbb{Z}}$는 정수 곱하기입니다.) 그러면 이 집합이 실제로 군이 됨을 보일 수 있습니다. 즉 **모든 원소에 대해 역원이 존재합니다**.

$g$는 [생성자](https://en.wikipedia.org/wiki/Generating_set_of_a_group)로 정의되었는데, 군 $G$에 대해 생성자는 $\\{g^{i}: i \in \mathbb{Z}\\} = G$인 $g \in G$를 의미합니다. 즉, *앉아서 $g$만 곱하고 있어도 온 세상 $G$의 원소들을 다 만날 수 있다*는 겁니다. 그렇다면 $g^{\phi(N)} = 1$이고, 모든 $0 \leq i \le \phi(N)$에 대해 $g^{i}$들이 전부 다를 수밖에 없습니다. 여기서 [group isomorphism](https://en.wikipedia.org/wiki/Group_isomorphism) $f: \mathbb{Z}/\phi(N)\mathbb{Z} \rightarrow \mathbb{Z} _ {n}^{\times}$를 $f(i) = g^{i}$와 같이 정의합니다. $f$는 isomorphism이니 역함수가 존재합니다. 이 문제는 바로 이 $f$의 **역함수**의 특정 지점에서의 값, 즉 $f^{-1}(x)$을 찾으라는 것입니다.

조금 많이 수학적으로 접근한 것 같은데, 수학 하시는 분들은 이렇게 두 군을 같은 것으로 보고 (같은 것이니) 더 이상 건드리지 않습니다. 하지만 우리는 이 값을 실제로 구하려고 했고, 그러다 보니 $f$를 계산하는 경우와는 달리 어려움에 마주하게 된 것입니다. 위의 말이 하나도 이해가 안 되시는 분들을 위해 쉬운 버전을 준비했습니다.

> $N$을 고정하고, $N$ 미만의 음이 아닌 정수 $g$를 고정하자. $g$는 다음을 만족한다: $N$ 미만의 음이 아닌 $N$과 서로소인 임의의 정수 $x$에 대해 $g^{i}$를 $N$으로 나눈 나머지가 $x$인 $i \in \mathbb{Z}$가 존재한다. $x$가 주어지면, $\phi(N)$ 미만의 음이 아닌 정수 $i$를 실제로 구하라.

두 문제에는 공통점이 있는데, 바로 정보가 주어진 경우 두 수를 곱한 경우의 정보도 곧바로 알 수 있다는 것입니다. 소인수분해의 경우 어떤 두 소인수분해가 있으면 합치는 것은 아주 쉽습니다. 이산 로그의 경우, $g^{n} = x$이고 $g^{m} = y$였다면 $g^{n + m} = xy$이기 때문에 $\log_{g} xy = n + m$입니다. 이런 곱셈에 대한 성질을 잘 이용하면 속도 향상을 꾀할 수 있습니다.

# 이산 로그

이산 로그를 빠르게 계산하는 방법에 대해서는 [이 글](/blog/2019/05/17/이산-로그/)이 이미 상당히 많은 부분을 다루어 주고 있기 때문에 참고하시면 좋을 듯합니다. 표기를 $\mathbb{Z}_{p}$로 하고 있는데 개인적으로는 [$p$-adic integer](https://en.wikipedia.org/wiki/P-adic_number)의 존재 때문에 썩 마음에 들지는 않는 표기입니다. 이 글에서는 $\mathbb{Z} _ {N}^{\times}$와 $\mathbb{Z}/\phi(N)\mathbb{Z}$의 표기를 고수하겠습니다.

우리가 이산 로그를 구하려고 하는 $x$에 대해 만일 $x = yz$이고 $\log_{g} y$와 $\log_{g} z$를 알고 있다면, 단순히 두 값을 더해 $\log_{g} x$를 구할 수 있을 것입니다. factor base는 이런 이산 로그의 **곱셈적인** 성질을 활용하는 것입니다. 즉, 먼저 factor base의 로그 값을 모두 구하고, $x$를 factor base에 속한 수들로 쪼개어 $\log_{g} x$를 이들의 로그의 합으로 간접 계산하는 것이 주요 아이디어입니다. 이것이 언제나 가능하지 않을 수는 있지만, 어떤 $j$에 대해 $x \cdot g^{j}$가 factor base에 속한 수들 $x_{1}$, $x_{2}$, $\cdots$, $x_{n}$으로 쪼개어진다면, \\\[\log_{g} x = -j + \sum_{i=1}^{n} \log_{g} x_{i}\\\]로 값을 빠르게 구할 수 있습니다.

그러면 두 가지 문제가 있습니다.

## 주어진 factor base에 대해서 각 원소의 log 값은 어떻게 구할 것인가?

이 질문은 두 질문 중 쉬운 축에 속하는 질문입니다. 우리는 단순히 한 factor의 제곱뿐만 아니라, factor base로 분해될 수 있는 **모든** 수에 관심이 있습니다. 우리가 충분히 운이 좋아 $x = x_{1} x_{2} \cdots x_{n}$의 분해를 여럿 얻었다면, 이 분해로부터 factor base의 log 값을 거꾸로 얻어낼 수도 있습니다.

예를 들어 $N = 993244853$, $\omega = 42$이라고 합시다. 우리가 _운이 너무 좋아서,_ 다음 네 개의 relation을 찾았다고 합시다.\\\[\begin{aligned}\omega^{1} &= 42 &= 2^{1} \cdot 3^{1} \cdot 5^{0} \cdot 7^{1}\\\\ \omega^{499302} &= 23040 &= 2^{9} \cdot 3^{2} \cdot 5^{1} \cdot 7^{0}\\\\ \omega^{2505717} &= 94539375 &= 2^{0} \cdot 3^{2} \cdot 5^{4} \cdot 7^{5} \\\\ \omega^{2824498} &= 2470629 &= 2^{0} \cdot 3^{1} \cdot 5^{0} \cdot 7^{7}\end{aligned}\\\]이때 $\log_{\omega} 2$, $\log_{\omega} 3$, $\log_{\omega} 5$와 $\log_{\omega} 7$를 찾기 위해서는 오른쪽의 "행렬"이 항등행렬이 되도록 각 relation을 적절히 짜맞추면 됩니다. 즉, 선형대수를 조금 사용하면,\\\[\begin{bmatrix}1 & 1 & 0 & 1\\\\9&2&1&0\\\\0&2&4&5\\\\0&1&0&7\end{bmatrix}\cdot\begin{bmatrix}-47&24&-6&11\\\\252&-28&7&-41\\\\-81&9&40&-17\\\\-36&4&-1&30
\end{bmatrix}=169I\\\]이므로,\\\[169^{-1} \cdot \begin{bmatrix}-47&24&-6&11\\\\252&-28&7&-41\\\\-81&9&40&-17\\\\-36&4&-1&30
\end{bmatrix}\cdot\begin{bmatrix}1\\\\499302\\\\2505717\\\\2824498\end{bmatrix} \equiv \begin{bmatrix}47183297\\\\275563689\\\\17967103\\\\670497867\end{bmatrix} \mod \phi(N)\\\]입니다. 따라서\\\[\begin{aligned}\omega^{47183297} &= 2\\\\ \omega^{275563689} &= 3 \\\\ \omega^{17967103} &= 5 \\\\ \omega^{670497867} &= 7 \end{aligned}\\\]입니다!

결론적으로 위와 같이 factor base에 속하는 수들로 잘 나누어지는 relation들을 찾을 수만 있으면, 간단한 선형대수를 활용하여 factor base의 log 값을 찾을 수 있습니다.

## factor base를 어떻게 두어야 성능상 이득을 최대로 볼 수 있는가?

그런데 우리가 이 relation을 찾은 것은 _정말로 운이 좋았던_ 것일까요? 위 예시에서 특히 $\omega^{670497867} = 7$을 얻기 위해서 실제로는 $\omega$를 $670497867$의 $\frac{1}{100}$번도 곱하지 않았다는 사실에 주목합시다. 생각보다 $2$, $3$, $5$, $7$만으로 완전히 나누어떨어지는 수는 많습니다.

```cpp
#include <cstdio>
#include <bitset>
using std::bitset;

const int N = 993244853;
bitset<N> set;

int main() {
    set.set(1);
    for (int i=2; i<N; ++i) {
        for (const auto &p: {2, 3, 5, 7}) {
            if (i % p == 0 && set.test(i / p)) {
                set.set(i);
                break;
            }
        }
    }
    printf("%lu\n", set.count());
    return 0;
}
```

```
$ g++ test.cpp -o test -O2 -std=gnu++17
$ time ./test
5190

real    0m5.967s
user    0m5.927s
sys     0m0.040s
$
```

**일반적으로** 수가 작을수록 그 수가 나누어떨어뜨리는 수가 많기 때문에, factor base에 속하는 수는 작을수록 좋습니다. 하지만 작다고 모두 좋은 것은 아닙니다: 합성수의 경우, 더 작은 소수가 반드시 존재하기 때문에 그 소수를 삼는 것이 이득입니다. (물론 어떤 합성수를 취하거나 어떤 큰 소수를 취하는 것이 relation을 빨리 찾는 데 도움을 줄 수 있겠지만, 그렇게 되면 $N$과 $\omega$가 주어졌을 때 그것을 알아내는 방법도 알아야 하고, 논의가 복잡해지므로 생각하지 않겠습니다.) 그래서 우리는 factor base를 **$B$ 이하의 모든 소수**로 선택하겠습니다.

$N$ 이하의 양의 정수 중 $B$ 이하의 모든 소수로 나누어떨어지는 것이 얼마나 많을까요? 이 *굉장히 어려운* 질문의 답을 찾기 위해 [Dickman-de Bruijn function](https://en.wikipedia.org/wiki/Dickman_function)을 활용합니다. 여기에서는 증명 없이 결과만을 알아봅니다; 관련해 더 알아보고 싶으신 분은 해당 wikipedia 문서의 수많은 reference를 참고하시면 됩니다. $\Psi(x, y)$를 $x$ 이하의 양의 정수 중 가장 큰 소인수가 $y$ 이하인 수의 개수라고 할 때,\\\[\Psi(x, y) \approx xu^{-u} \quad\mathrm{where}\quad u := \log_{y}x\\\]라고 합니다. 즉 하나를 찾는 데 평균적으로 $u^{u}$번이 소요된다는 것입니다. 우리는 이것을 $B$번 정도 수행해야 하므로,\\\[T_{n} (B) := B \cdot \left(\frac{\log N}{\log B}\right)^{\log N / \log B}\\\]를 최소화해야 합니다. 그런데\\\[\begin{aligned}
    \log T_{n} (B) &= \log B + \frac{\log N}{\log B} (\log \log N - \log \log B)\\\\{}
    \frac{T_{n}' (B)}{T_{n} (B)} &= \frac{1}{B} - \frac{\log N}{B \log^{2} B}(\log \log N - \log \log B) - \frac{\log N}{\log B} \cdot \frac{1}{B \log B}\\\\{}
    \therefore T_{n}' (B) &= \frac{T_{n}(B) \log N}{B \log^{2} B}\left(\frac{\log^{2} B}{\log N} - \log \log N + \log \log B - 1 \right)
\end{aligned}\\\]
이므로, 우리는 $\log^{2} B + \log N \log \log B = \log N (\log \log N + 1)$이 $0$이 되는 $B$를 찾으면 됩니다. $\log^{2} B =: X$로 두면,\\\[\begin{aligned}
    X + \frac{\log N}{2} \log X &= \log N (\log \log N + 1)\\\\{}
    \frac{\log N}{2} \cdot \frac{2X}{\log N} + \frac{\log N}{2} \log \left(\frac{2X}{\log N}\right) &= \log N (\log \log N + 1) - \frac{\log N}{2} \log \left(\frac{\log N}{2}\right)\\\\{}
    \frac{2X}{\log N} + \log \left(\frac{2X}{\log N}\right) &= 2 (\log \log N + 1) - \log \left(\frac{\log N}{2}\right)\\\\{}
    &= \log \log N + \log 2 + 2
\end{aligned}\\\]
입니다. $Y := 2X / \log N$으로 치환하면, $Y + \log Y = \mathrm{const}$ 꼴의 식이 나왔습니다! 이는 [Lambert $W$ function](https://en.wikipedia.org/wiki/Lambert_W_function)을 이용하여 풀 수 있습니다.\\\[\begin{aligned}
    Y \exp Y &= \exp (\log \log N + \log 2 + 2)\\\\{}
    &= 2e^{2} \log N\\\\{}
    \therefore Y &= W(2e^{2} \log N)\\\\{}
    X &= \frac{1}{2} \log N \cdot W(2e^{2} \log N)\\\\{}
    B &= \exp \sqrt{\frac{1}{2} \log N \cdot W(2e^{2} \log N)}
\end{aligned}\\\]

$B$의 크기를 대강 추정해 보면, $W(x) \approx \log x - \log \log x$를 활용하여 $\log B \approx 1/\sqrt{2} \cdot \sqrt{\log N} \sqrt{\log \log N}$임을 알 수 있습니다. 따라서 $u = (\sqrt{2} + o(1)) \log N / \sqrt{\log N \log \log N} = (\sqrt{2} + o(1))\sqrt{\log N / \log \log N}$이고\\\[\begin{aligned}
    u^{u} &= \left((\sqrt{2} + o(1))\sqrt{\frac{\log N}{\log \log N}}\right)^{(\sqrt{2} + o(1))\sqrt{\log N / \log \log N}}\\\\{}
    &= \exp \left((\sqrt{2} + o(1))\frac{1}{2}\sqrt{\frac{\log N}{\log \log N}}(\log \log N - \log \log \log N)\right)\\\\{}
    &\approx \exp \left(\left(\frac{1}{\sqrt{2}} + o(1)\right) (\log N)^{1/2} (\log \log N)^{1/2}\right)
\end{aligned}\\\]
이며 따라서 전체 프로그램의 수행 시간인 $Bu^{u}$는, [L notation](https://en.wikipedia.org/wiki/L-notation)을 사용하면, $L_{n}\left[\frac{1}{2}; \sqrt{2} \right]$입니다. 이 값은 $B$와 비교하면 $B^{2}$ 정도이기 때문에, 만일 역행렬을 구할 때 $B^{3}$ 알고리즘을 사용한다면 알고리즘의 시간 복잡도가
$L_{n} \left\[\frac{1}{2};\frac{3}{\sqrt{2}}\right\]$ 정도로 늘어날 수 있습니다. 이는 (지수에 들어가기 때문에) 복잡도 상 큰 차이입니다. 속도를 더욱 빠르게 하려면 얻어진 matrix가 sparse하다는 것을 이용해서 $B^{2} \mathcal{P}(\log B)$ 정도에 돌아가는 inverse 알고리즘을 사용할 필요가 있습니다.

## 연습 문제: 이산로그가 장난이냐?

factor base를 활용한 문제가 BOJ에 내기에는 수행 시간이 너무 오래 걸린다고 생각했는데, [이산로그가 장난이냐?](https://www.acmicpc.net/problem/21864)라는 문제가 나와서 BOJ에서도 factor base speedup을 연습해 볼 수 있게 되었습니다. 아이디어는 2,000,000부터 거꾸로 계산하는 것은 쉽다는 것입니다: 2,000,000일 때의 답을 계산해서 코드에 넣고 주어진 값을 2,000,000에서 빼서 거꾸로 계산하면 됩니다. 이산로그를 실제로 구하는 것은 위에서 논의한 내용을 적용하면 됩니다. 사용된 $N = 10^{18} + 31$이 충분히 작기 때문에 $B^{3}$ inverse 알고리즘으로도 빠른 시간에 돌아갑니다.

[expand 167줄의 코드 보기]

```cpp
#ifdef BOJ
#pragma GCC optimize ("Ofast")
#pragma GCC target ("avx,avx2")
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <random>
#include <vector>

#ifdef BOJ
#include <immintrin.h>
#include <smmintrin.h>
#endif

using namespace std;

typedef pair<int, int> pii;
typedef unsigned long long ulli;

const ulli p = 1'000'000'000'000'000'031ULL;
const ulli q = p - 1;
const ulli w = 42ULL;

const int B = 29237;
vector<int> primes;

void init() {
    bitset<B+1> not_prime;
    for (int i=2; i<=B; ++i) {
        if (!not_prime.test(i)) {
            primes.push_back(i);
            for (int j=i*2; j<=B; j+=i) {
                not_prime.set(j);
            }
        }
    }
}

ulli modinv(ulli x, ulli n) {
    ulli a1 = n, b1 = 0ULL;
    ulli a2 = x % n, b2 = 1ULL;
    if (a2 == 0) return 0ULL;
    while (ulli a3 = a1 % a2) {
        ulli b3 = (b1 + (__uint128_t)(n - b2) * (a1 / a2)) % n;
        a1 = a2; b1 = b2;
        a2 = a3; b2 = b3;
    }
    if (a2 == 1) {
        return b2;
    }
    return 0ULL;
}

int main() {
    init();
    ulli x = 1ULL;
    ulli k = 0;
    vector<pair<ulli, vector<pair<int, ulli> > > > relation;
    while (relation.size() < primes.size()) {
        if ((k & 1023) == 1023) {
            printf("\r%lu / %lu", relation.size(), primes.size());
            fflush(stdout);
        }
        x = (__uint128_t)x * w % p;
        ++k;
        vector<ulli> y;
        y.resize(B+1);
        ulli z = x;
        for (const int &p: primes) {
            int c = 0;
            while (z % p == 0) {
                z /= p;
                ++c;
            }
            y[p] = c;
        }
        if (z == 1) {
            ulli now_k = k;
            for (const auto &[their_k, u]: relation) {
                const ulli multiplier = y[u[0].first];
                for (const auto &[p, c] : u) {
                    y[p] = (y[p] + (__int128_t)(::q - c) * multiplier) % ::q;
                }
                now_k = (now_k + (__uint128_t)(::q - their_k) * multiplier) % ::q;
            }
            int selected_p = -1;
            ulli selected_cnt = 0;
            for (const int &p: primes) {
                ulli inv = modinv(y[p], ::q);
                if (inv) {
                    selected_p = p;
                    selected_cnt = inv;
                    break;
                }
            }
            if (selected_p >= 0) {
                vector<pair<int, ulli> > to_relation;
                to_relation.emplace_back(selected_p, 1ULL);
                for (const int &p: primes) {
                    if (p == selected_p) continue;
                    if (y[p]) {
                        to_relation.emplace_back(p, (y[p] = (__uint128_t)y[p] * selected_cnt % ::q));
                    }
                }
                now_k = (__uint128_t)now_k * selected_cnt % ::q;
                relation.emplace_back(now_k, to_relation);
            }
        }
    }
    puts("");
    vector<ulli> ws;
    {
        ws.resize(B + 1);
        while (!relation.empty()) {
            const auto &[this_k, this_u] = relation.back();
            ulli k = this_k;
            for (int i=1; i<this_u.size(); ++i) {
                const auto &[p, c] = this_u[i];
                k = (k + (__uint128_t)(::q - c) * ws[p]) % ::q;
            }
            ws[this_u[0].first] = k;
            relation.pop_back();
        }
    }
    x = 300ULL;
    for (int i=0; i<1000000; ++i) {
        if ((i & 255) == 255) {
            printf("\r%6.2lf%%", i / 10000.);
            fflush(stdout);
        }
        k = 0;
        while (1) {
            ulli uk = ::q - k;
            ulli z = x;
            for (const int &p: primes) {
                while (z % p == 0) {
                    z /= p;
                    uk += ws[p];
                    if (uk >= ::q) {
                        uk -= ::q;
                    }
                }
            }
            if (z == 1) {
                x = uk;
                if (!x) {
                    x = ::q;
                }
                break;
            }
            x = (__uint128_t)x * w % p;
            ++k;
        }
    }
    printf("\r%6.2lf%%\n", 100.);
    printf("%llu\n", x);
    return 0;
}
```

[/expand]

```
3177 / 3178
100.00%
******************

real    19m10.651s
user    19m10.391s
sys     0m0.260s
```

# 소인수분해

이제 소인수분해를 논의해 봅시다. 소인수분해의 중요성은 두 번 말할 필요가 없을 것입니다. 문제는, 위에서 논의한 곱셈을 한 결과를 바로 알 수 있다는 성질은, 소인수분해에 대해서는 아무런 쓸모가 없다는 사실일 것입니다. 약수를 알기 전까지는 적용하기 힘든 사실이니까요. 조금 다른 방법을 생각할 필요가 있습니다.

## 제곱 합동식

해결책은 생각보다 단순합니다. $x^{2} \equiv y^{2} \mod N$인 (비자명한; i.e. $x \not\equiv \pm y \mod N$인) 두 자연수 $x$와 $y$를 구하면 됩니다. 그러면 $(x + y)(x - y)$가 $N$의 배수인데, $x \pm y$가 $N$으로 나누어떨어지지는 않으므로 $\gcd(x + y, N)$과 $\gcd(x - y, N)$이 $N$의 약수가 됩니다!

예를 들어 $N = 10001$이라고 합시다. 우리가 _운이 좋아_ $3711^{2} \equiv 144 \equiv 12^{2} \mod N$을 찾았다고 합시다. 그러면 $\gcd(3711 - 12, 10001) = 137$과 $\gcd(3711 + 12, 10001) = 73$으로 $N$의 두 약수가 주어집니다.

## Dixon's factorization algorithm

이 방법에서 간단하게 떠올릴 수 있는 factor base speedup이 있습니다. 좌변은 그대로 아무 수나 제곱하게 하고, 우변을 (제곱수일 필요는 없는) factor base로 확장한 뒤 전처럼 선형대수로 제곱이 되도록 맞추면 될 것 같습니다. 여전히 예로 $N = 10001$을 사용하겠습니다. 이젠 이 관계식을 찾기 위해 우리의 운이 너무 좋을 필요는 없다는 사실을 알고 계실 것입니다.

\\\[\begin{aligned}
    2872^{2} \equiv 7560 &\equiv 2^{3} \cdot 3^{3} \cdot 5^{1} \cdot 7^{1} \mod N\\\\{}
    4670^{2} \equiv 6720 &\equiv 2^{6} \cdot 3^{1} \cdot 5^{1} \cdot 7^{1}\mod N\\\\{}
    6061^{2} \equiv 2048 &\equiv 2^{11} \cdot 3^{0} \cdot 5^{0} \cdot 7^{0}\mod N
\end{aligned}\\\]

양 변을 모두 곱하면, $(2872 \cdot 4670 \cdot 6061)^{2} \equiv 8295^{2} \equiv 2528^{2} \equiv (2^{10} \cdot 3^{2} \cdot 5^{1} \cdot 7^{1})^{2} \mod N$이라는 관계식을 얻게 됩니다. 따라서 $\gcd(8295 - 2528, 10001) = 73$과 $\gcd(8295 + 2528, 10001) = 137$으로 $N$의 두 약수가 주어집니다.

## Quadratic Sieve

Quadratic Sieve(QS)는 여기에서 기민한 최적화 몇 개를 시행한 것입니다. 먼저, $N$이 충분히 큰 경우를 다룬다고 해 봅시다. 단순히 아무 수나 해 보는 것보다, $i = \left\lceil \sqrt{N} \right\rceil$ 근처부터 해 보는 게 좋지 않을까요? 왜냐하면 $\sqrt{N} \leq i \leq \sqrt{N} + 1$이므로, $N \leq i^{2} \leq N + 2\sqrt{N} + 1$이 되어 $i^{2} \mod N = i^{2} - N$이 $\mathcal{O}(\sqrt{N})$ 수준이기 때문입니다. 즉, $\sqrt{N}$보다는 크지만 충분히 가까운 수들만 보아도 원하는 값을 얻을 수 있을 것 같습니다. 수 자체가 작기 때문에 smooth한 수를 얻을 확률도 올라갑니다.

그런데, $x^{2} \mod N$이 $x^{2} - 2N$이 되려면 적어도 $x \geq \sqrt{2N}$이어야 합니다. $\sqrt{N}$부터 $\sqrt{2N}$까지는 대강 $(\sqrt{2} - 1)\sqrt{N} = \mathcal{O}(\sqrt{N})$개의 수가 있고, 이걸 다 볼 바에야 trial division을 하면 되지 않을까요? 따라서 우리는 지금부터 $\sqrt{N} \leq x \leq \sqrt{2N}$인 정수에 대해서만 고려하겠습니다.

그러면 나눗셈을 할 필요 없이 $x^{2} - N$을 계산하여 모든 소수로 나누어보면 됩니다. 그런데, 이 값이 소수 $p$로 나누어떨어지려면 $x^{2} \equiv N \mod p$가 되어야 합니다. 이는 [Tonelli-Shanks](https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm) 등을 통해 굉장히 빨리 계산할 수 있으며 한 번만 하면 됩니다. 이것의 해가 존재하지 않는다면 $x^{2} - N$이 어떤 수에 대해서도 나누어떨어지지 않는다는 것입니다. 즉, $x$의 범위를 이런 식으로 제한하면 **$B$를 줄이지 않고도** 소수의 개수를 유의미하게 줄일 수 있으며, 에라토스테네스의 체처럼 하여 **적어도 한 번 이상 나누어떨어지는 소수만으로 나눌 수 있습니다!**

예를 들어 봅시다. $N = 10001$로 두면, 우리는 $101 \leq x \leq 141$의 범위에서만 신경쓰면 됩니다. 아래의 표는 $101 \leq x \leq 109$에 대해서만 표시하겠습니다. $x^{2} - N$이 $p = 2$로 나누어떨어지려면, $x \equiv 1 \mod 2$여야 합니다. 이러한 $x$들에 $2$로 나누어떨어진다는 정보를 줍니다. (?는 아직 계산되지 않은 부분입니다.)

|$x$|$x^{2} - N$|possible $p$'s|
|:-:|:---------:|--------------|
|$101$|?|$2$, ?|
|$102$|?|?|
|$103$|?|$2$, ?|
|$104$|?|?|
|$105$|?|$2$, ?|
|$106$|?|?|
|$107$|?|$2$, ?|
|$108$|?|?|
|$109$|?|$2$, ?|

마찬가지로 $x^{2} - N$이 $p = 3$으로 나누어떨어지려면, $x^{2} \equiv 2 \mod 3$이 되어야 하는데 $2$는 $p = 3$에서의 **quadratic** residue가 아니므로 $3$으로 나누어떨어지는 것은 하나도 없습니다. QS에서는 Dixon에서 이런 부분이 최적화되는 것입니다! $x^{2} - N$이 $p = 5$로 나누어떨어지려면, $x \equiv \pm 1 \mod 5$가 되어야 하므로, $5$로 나누었을 때 나머지가 $1$ 혹은 $4$인 것들에 $5$를 추가해 줍니다.

|$x$|$x^{2} - N$|possible $p$'s|
|:-:|:---------:|--------------|
|$101$|?|$2$, $5$, ?|
|$102$|?|?|
|$103$|?|$2$, ?|
|$104$|?|$5$, ?|
|$105$|?|$2$, ?|
|$106$|?|$5$, ?|
|$107$|?|$2$, ?|
|$108$|?|?|
|$109$|?|$2$, $5$, ?|

마찬가지 방법으로 $p \leq 20 =: B$인 소수 $p$에 대해 모두 해 줍시다. $p = 7$, $p = 11$, $p = 17$의 경우는 계산을 해 보면 이차잉여가 아니게 됩니다. 따라서 실제로 추가되는 것은 $p = 13$일 때 $x \equiv \pm 2 \mod 13$과 $p = 19$일 때 $x \equiv \pm 8 \mod 19$뿐입니다.

|$x$|$x^{2} - N$|possible $p$'s|
|:-:|:---------:|--------------|
|$101$|?|$2$, $5$|
|$102$|?|$13$|
|$103$|?|$2$, $19$|
|$104$|?|$5$|
|$105$|?|$2$|
|$106$|?|$5$, $13$, $19$|
|$107$|?|$2$|
|$108$|?|*없음*|
|$109$|?|$2$, $5$|

지금은 $N = 10001$이라서 전부 계산할 수 있지만, $N$이 충분히 커지면 전체 구간에서 앞 $X$개씩만 먼저 처리해야 할 것입니다. 이 과정이 이차잉여에 대해 에라토스테네스의 체를 계산하는 과정과 유사하다고 해서 *quadratic sieve*라는 이름이 붙었습니다.

이 과정에서 중요한 것은 $x^{2} - N$의 계산 **없이** 각각이 $B$ 이하의 소수 중 무엇으로 나누어떨어질지를 모두 알았다는 것입니다. 이렇게 possible $p$를 계산하는 과정에서, 모든 수를 돌면서 나머지를 확인하면 **안 됩니다!** 에라토스테네스의 체처럼, $p$의 배수에 계산된 제곱근을 더하고 빼서 그 수에만 직접 더해야 합니다. 그렇게 하면 이 과정을 하는데 $X \sum_{p \leq B} \frac{1}{p} \sim X \log \log B$ 시간이 소요됩니다.

이제 실제로 $x^{2} - N$을 계산하면 되는데, 만일 $x^{2} - N$을 알고 있을 경우 $(x + 1)^{2} - N = (x^{2} - N) + (2x + 1)$로 표현할 수 있으므로, 곱셈 없이 자릿수에 대한 선형 시간에 다음 수를 알 수 있습니다. 수가 크면 곱셈보다는 덧셈이 빠르므로 의미있는 최적화입니다. 각 수를 구한 다음, 예전처럼 $B$ 이하의 소수로 전부 나누어 볼 필요 없이 possible $p$'s에 대해서만 나누어 보면 됩니다.

|$x$|$x^{2} - N$|possible $p$'s|
|:-:|:---------:|--------------|
|$101$|$200 = 2^{3} \cdot 5^{2} \cdot \color{blue}{1}$|$2$, $5$|
|$102$|$403 = 13 \cdot \color{red}{31}$|$13$|
|$103$|$608 = 2^{5} \cdot 19 \cdot \color{blue}{1}$|$2$, $19$|
|$104$|$815 = 5 \cdot \color{red}{163}$|$5$|
|$105$|$1024 = 2^{10} \cdot \color{blue}{1}$|$2$|
|$106$|$1235 = 5 \cdot 13 \cdot 19 \cdot \color{blue}{1}$|$5$, $13$, $19$|
|$107$|$1448 = 2^{3} \cdot \color{red}{181}$|$2$|
|$108$|$1663 = \color{red}{1663}$|*없음*|
|$109$|$1880 = 2^{3} \cdot 5 \cdot \color{red}{47}$|$2$, $5$|

이후의 과정은 Dixon's factorization algorithm과 동일합니다. 충분히 많은 relation을 계산한 다음, 선형대수를 이용해서 제곱을 맞춥니다.

이렇게 하면 나누는 횟수가 많이 줄어서, 놀랍게도\\\[B = \exp \left(\frac{1}{2}\cdot \sqrt{\log N \cdot W(2e^{2} \log N)}\right)\\\] 정도의 $B$가 가장 빠르고, 알고리즘의 전체 수행 시간은 $L_{n}\left[\frac{1}{2}; 1\right]$이 된다고 합니다.

## cutoff

위에서 "선형대수를 이용해 제곱을 맞춘다"라고 설명한 부분은, 사실 여기서 다루고자 하는 범위와는 많이 벗어나는 선형대수 내용이기 때문에 일부러 조금 얼버무린 감이 있습니다. 정확하게 알아보는 것은 선형대수와 현대대수에 맡기고, 선형대수의 힘을 빌리기 전에 우리가 할 수 있는 간단한 것들을 알아봅시다.

### count 1 cutoff

어떤 소수 $p_{i}$의 홀수 제곱이 relation에서 **단 한 번만** 나타나면, 그 relation은 쓰이지 않기 때문에 제거할 수 있습니다. 왜냐하면 이 relation이 쓰이면, 다른 relation에는 $p_{i}$의 홀수 제곱이 전혀 나타나지 않기 때문에 우변의 모든 지수를 짝수로 맞출 수 없기 때문입니다. 상대적으로 간단한 최적화이고, $\frac{1}{10}$ ~ $\frac{1}{100}$ 정도의 relation을 줄여 주기 때문에 애용됩니다.

### count 2 cutoff

어떤 소수 $p_{i}$의 홀수 제곱이 relation에서 **단 두 번** 나타나면, 그 relation들은 둘 다 쓰이거나 둘 다 쓰이지 않습니다. 따라서 두 relation을 합칠 수 있습니다. 이유는 count 1 cutoff와 비슷합니다. 간단한 최적화이지만, sparsity를 해치기 때문에 그렇게 자주 사용되지는 않습니다.

## Quadratic Sieve 코드

아래는 quadratic sieve를 C++로 구현한 코드입니다. mod $p$에서 제곱근을 구하는 함수는 Tonelli-Shanks, 행렬을 적용했을 때 $0$이 되는 $0$이 아닌 vector를 구하는 알고리즘은 [Wiedemann algorithm](https://en.wikipedia.org/wiki/Block_Wiedemann_algorithm)을 이용했습니다. 이 알고리즘 대신 Block Wiedemann algorithm을 사용하거나, [Block Lanczos algorithm](https://en.wikipedia.org/wiki/Block_Lanczos_algorithm)을 사용하면 속도 향상을 더욱 꾀할 수 있을 것입니다. 이 글의 주제는 factor base이기 때문에 이 부분을 성능에 신경써서 작성했고, 선형대수를 사용하는 부분은 구현만 간단하게 해 놓았습니다.

[expand 632줄의 코드 보기]

```cpp
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
using namespace std;

int modpow(int a, int b, int p) {
    int r;
    for (r=1; b; b>>=1) {
        if (b & 1) r = 1LL * r * a % p;
        a = 1LL * a * a % p;
    }
    return r;
}

int berlecamp_massey(const vector<int> &inst, vector<int> &result, int p) {
    int L = 0, m = 1;
    const int N = (int)inst.size();
    int b = 1;

    vector<int> B, C;
    B.push_back(1);
    C.push_back(1);

    for (int n = 0; n < N; ++n) {
        long long int d = inst[n];
        for (int i = 1; i <= L; ++i) {
            d += 1LL * C[i] * inst[n - i] % p;
        }
        d %= p;
        if (d == 0) {
            ++m;
        } else if (2 * L <= n) {
            vector<int> T(C);
            const int coeff = 1LL * (p - d) * modpow(b, p-2, p) % p;
            const int min_C_size = (int)B.size() + m;
            if (C.size() < min_C_size) {
                C.resize(min_C_size);
            }
            for (int i=0; i<B.size(); ++i) {
                auto &now = C[i + m];
                now = (now + 1LL * coeff * B[i]) % p;
            }
            L = n + 1 - L;
            B.resize(T.size());
            copy(T.begin(), T.end(), B.begin());
            b = d;
            m = 1;
        } else {
            const int coeff = 1LL * (p - d) * modpow(b, p-2, p) % p;
            const int min_C_size = (int)B.size() + m;
            if (C.size() < min_C_size) {
                C.resize(min_C_size);
            }
            for (int i=0; i<B.size(); ++i) {
                auto &now = C[i + m];
                now = (now + 1LL * coeff * B[i]) % p;
            }
            ++m;
        }
    }
    result.resize(C.size());
    copy(C.begin(), C.end(), result.begin());
    return L;
}

void mat_apply(const vector<pair<int, int> > &spmat, bool *x, bool *y) {
    for (const auto &[cy, cx]: spmat) {
        if (x[cx]) {
            y[cy] = !y[cy];
        }
    }
}

mt19937 mt((unsigned long long)chrono::high_resolution_clock::now().time_since_epoch().count());
bool wiedemann(int sz, const vector<pair<int, int> > &spmat, bool *res) {
    uniform_int_distribution bit(0, 1);
    bool *xbase = new bool[sz];
    bool *now = new bool[sz];
    bool *y = new bool[sz];
    for (int i=0; i<sz; ++i) {
        xbase[i] = bit(mt) ? true : false;
        y[i] = bit(mt) ? true : false;
    }
    memset(now, 0, sizeof(bool) * sz);
    mat_apply(spmat, xbase, now);
    bool *temp = new bool[sz];
    vector<int> bm_inst;
    for (int i=0; i<3*sz+3; ++i) {
        int bmv = 0;
        for (int j=0; j<sz; ++j) {
            if (now[j] && y[j]) {
                ++bmv;
            }
        }
        bmv &= 1;
        bm_inst.push_back(bmv);
        memset(temp, 0, sizeof(bool) * sz);
        mat_apply(spmat, now, temp);
        memcpy(now, temp, sizeof(bool) * sz);
    }
    memcpy(now, xbase, sizeof(bool) * sz);
    vector<int> rel;
    berlecamp_massey(bm_inst, rel, 2);
    reverse(rel.begin(), rel.end());
    for (const auto &coeff: rel) {
        if (coeff) {
            for (int i=0; i<sz; ++i) {
                if (now[i]) {
                    res[i] = !res[i];
                }
            }
        }
        memset(temp, 0, sizeof(bool) * sz);
        mat_apply(spmat, now, temp);
        memcpy(now, temp, sizeof(bool) * sz);
    }
    delete[] xbase;
    delete[] now;
    delete[] y;
    {
        int i;
        for (i=0; i<sz; ++i) {
            if (res[i]) {
                break;
            }
        }
        if (i >= sz) {
            delete[] temp;
            return false;
        }
    }    
    memset(temp, 0, sizeof(bool) * sz);
    mat_apply(spmat, res, temp);
    for (int i=0; i<sz; ++i) {
        if (temp[i]) {
            delete[] temp;
            return false;
        }
    }
    delete[] temp;
    return true;
}

class bigint {
public:
    vector<unsigned int> v;
    bigint(unsigned long long x = 0) {
        if (x) {
            v.push_back(x & 0x7FFFFFFFU);
            v.push_back((x >> 31) & 0x7FFFFFFFU);
            v.push_back((x >> 62) & 0x7FFFFFFFU);
            while (!this->v.empty() && !this->v.back()) this->v.pop_back();
        }
    }
    bigint(initializer_list<unsigned int> il) {
        v = il;
        while (!this->v.empty() && !this->v.back()) this->v.pop_back();
    }
    bigint(const bigint &x) {
        *this = x;
    }
    unsigned int size() const {
        return this->v.size();
    }
    bigint &operator=(const bigint &other) {
        if (this == &other) return *this;
        this->v.resize(other.size());
        copy(other.v.begin(), other.v.end(), this->v.begin());
        return *this;
    }
    bigint &operator+=(const bigint &other) {
        if (other.size() > this->size()) {
            this->v.resize(other.size());
        }
        unsigned int carry = 0U, i;
        for (i=0; i<other.size(); ++i) {
            carry = (this->v[i] += other.v[i] + carry) >> 31;
            this->v[i] &= 0x7fff'ffffU;
        }
        for (; carry && i<this->size(); ++i) {
            carry = (this->v[i] += carry) >> 31;
            this->v[i] &= 0x7fff'ffffU;
        }
        if (carry) {
            this->v.push_back(carry);
        }
        return *this;
    }
    bigint &operator-=(const bigint &other) {
        if (other.size() > this->size()) {
            this->v.resize(other.size());
        }
        unsigned int carry = 0U, i;
        for (i=0; i<other.size(); ++i) {
            carry = (this->v[i] -= other.v[i] + carry) >> 31;
            this->v[i] &= 0x7fff'ffffU;
        }
        for (; carry && i<this->size(); ++i) {
            carry = (this->v[i] -= carry) >> 31;
            this->v[i] &= 0x7fff'ffffU;
        }
        while (!this->v.empty() && !this->v.back()) this->v.pop_back();
        return *this;
    }
    bigint &operator*=(const bigint &other) {
        vector<unsigned long long> res;
        res.resize(this->size() + other.size());
        for (int i=0; i<this->size(); ++i) {
            for (int j=0; j<other.size(); ++j) {
                unsigned long long x = (unsigned long long)this->v[i] * other.v[j];
                res[i+j] += x & 0x7fff'ffffU;
                res[i+j+1] += x >> 31;
            }
        }
        this->v.resize(this->size() + other.size());
        unsigned long long carry = 0;
        for (int i=0; i<res.size(); ++i) {
            this->v[i] = (res[i] += carry) & 0x7fff'ffff;
            carry = res[i] >> 31;
        }
        while (!this->v.empty() && !this->v.back()) this->v.pop_back();
        return *this;
    }
    bigint &operator<<=(unsigned int x) {
        if (x) {
            this->v.insert(this->v.begin(), x, 0U);
        }
        return *this;
    }
    bigint &operator>>=(unsigned int x) {
        if (x) {
            if (this->size() <= x) {
                this->v.clear();
            } else {
                this->v.erase(this->v.begin(), this->v.begin() + x);
            }
        }
        return *this;
    }
    bigint &operator/=(const unsigned int x) {
        unsigned long long now = 0;
        for (auto it = this->v.rbegin(); it != this->v.rend(); ++it) {
            auto &v = *it;
            (now <<= 31) |= v;
            v = now / x;
            now %= x;
        }
        while (!this->v.empty() && !this->v.back()) this->v.pop_back();
        return *this;
    }
    unsigned int operator%(unsigned int x) const {
        unsigned now = 0;
        for (auto it = this->v.rbegin(); it != this->v.rend(); ++it) {
            now = (((unsigned long long)now << 31) | *it) % x;
        }
        return now;
    }
    bool operator==(const bigint &other) const {
        return this->v == other.v;
    }
    void invert(const unsigned int u) {
        if (this->size() > u) {
            this->v.clear();
            return;
        }
        unsigned int fr = min(0x8000'0000U / this->v.back() + 1, 0x7fff'ffffU);
        bigint res(fr);
        res <<= u - this->size();
        int ed = 2 * (int)log2(u) + 5;
        for (int i=0; i<ed; ++i) {
            bigint temp(res);
            res *= 2U;
            ((temp *= temp) *=* this) >>= u;
            res -= temp;
        }
        *this = res;
    }
    double log() const { // rough log
        if (!this->size()) {
            return -INFINITY;
        }
        if (this->size() <= 2) {
            unsigned long long res = this->v.back();
            if (this->size() == 2) {
                (res <<= 31) |= this->v.front();
            }
            return (double)logl(res);
        }
        const auto rev_it = this->v.rbegin();
        long double now = rev_it[2];
        (now /= 0x100'000'000ULL) += rev_it[1];
        (now /= 0x100'000'000ULL) += rev_it[0];
        return (double)(logl(now) + logl(2ULL) * (this->size() - 1) * 31);
    }
    void input() {
        this->v.clear();
        char c;
        while ('0' <= (c = getchar()) && c <= '9') {
            ((*this) *= 10U) += c & 15;
        }
    }
    void print() const {
        if (this->v.empty()) {
            printf("0");
            return;
        }
        char *wf = new char[this->size() * 10 + 5];
        char *ptr = wf + (this->size() * 10 + 3);
        *ptr = 0;
        bigint temp(*this);
        while (!temp.v.empty()) {
            unsigned int r = temp % 1'000'000'000U;
            char last = *ptr;
            sprintf(ptr - 9, "%09u", r);
            *ptr = last;
            ptr -= 9;
            temp /= 1'000'000'000U;
        }
        while (*ptr == '0') ++ptr;
        printf("%s", ptr);
        delete[] wf;
    }
};

void modmul(const bigint &a, const bigint &b, const bigint &n, const bigint &ninv, const int u, bigint &res) {
    (res = a) *= b;
    bigint temp(res);
    res -= (((temp *= ninv) >>= u) *= n);
}

void modpow(const bigint &a, int b, const bigint &n, const bigint &ninv, const int u, bigint &res) {
    bigint temp(a);
    for (res=1U; b; b>>=1) {
        if (b & 1) modmul(res, temp, n, ninv, u, res);
        modmul(temp, temp, n, ninv, u, temp);
    }
}

void mod(const bigint &a, const bigint &b, bigint &res) {
    bigint binv(b);
    int u = max(a.size(), b.size()) * 3 + 2;
    binv.invert(u);
    bigint temp(a);
    (res = a) -= (((temp *= binv) >>= u) *= b);
}

void gcd(const bigint &a, const bigint &b, bigint &c) {
    int u = max(a.size(), b.size()) * 5 + 4;
    bigint A(a), B(b);
    bigint C;
    mod(A, B, C);
    while (!C.v.empty()) {
        A = B;
        B = C;
        mod(A, B, C);
    }
    c = B;
}

void isqrt(const bigint &c, bigint &x) {
    x.v.clear();
    unsigned long long est = c.v.back();
    if (!(c.size() & 1)) {
        (est <<= 31) |= c.v[c.size() - 2];
    }
    x.v.push_back((unsigned int)sqrtl(est) + 1);
    x <<= (c.size() - 1) >> 1;
    int ed = 2 * (int)log2(c.size()) + 5;
    for (int i=0; i<ed; ++i) {
        bigint u(x);
        u *= 2U;
        const unsigned int invert_val = max(c.size(), x.size() * 2) + 3;
        u.invert(invert_val);
        (x *= x) += c;
        (x *= u) >>= invert_val;
    }
}

int tonelli_shanks(int x, int p) {
    if (p <= 1) {
        return -1;
    }
    if (p == 2) {
        return x & 1;
    }
    if (modpow(x, p >> 1, p) != 1) {
        return -1;
    }
    int z;
    for (z=2; z<p-1; ++z) {
        if (modpow(z, p >> 1, p) != 1) {
            break;
        }
    }
    int Q = p - 1, S = 0;
    while (!(Q & 1)) {
        ++S;
        Q >>= 1;
    }
    int R = modpow(x, (Q + 1) >> 1, p), t = modpow(x, Q, p);
    const int zQ = modpow(z, Q, p);
    int w = zQ;
    for (int M = S - 2; M >= 0; --M) {
        if (modpow(t, 1 << M, p) != 1) {
            t = 1LL * t * (1LL * w * w % p) % p;
            R = 1LL * R * w % p;
        }
        w = 1LL * w * w % p;
    }
    return min(R, p - R);
}

void init(int B, const bigint &n, vector<pair<int, int> > &res) {
    bool *is_not_prime = new bool[B + 1];
    memset(is_not_prime, 0, sizeof(bool) * (B + 1));
    for (int i=2; i<=B; ++i) {
        if (!is_not_prime[i]) {
            for (int j=2*i; j<=B; j+=i) {
                is_not_prime[j] = true;
            }
            int x = tonelli_shanks(n % i, i);
            if (x >= 0) {
                res.emplace_back(i, x);
            }
        }
    }
    delete[] is_not_prime;
}

const int BULK = 3e6;

int guess_B(const bigint &n) {
    double x = n.log();
    double Wx = 2 * x * exp(2);
    double Wy = log(Wx) - log(log(Wx));
    return (int)exp(sqrt(Wy * x) / 2.) + 20;
}

int main() {
    bigint x;
    x.input();

    bigint now;
    isqrt(x, now);
    {
        bigint temp(now);
        temp *= temp;
        if (temp == x) {
            now.print();
            return 0;
        }
    }
    now += 1U;
    bigint now_value(now);
    (now_value *= now_value) -= x;
    vector<pair<int, int> > base;
    int B = guess_B(x);
    init(B, x, base);
    for (const auto &[p, x]: base) {
        if (x == 0) {
            bigint(p).print();
            return 0;
        }
    }
    printf("B = %d, prime # = %lu\n", B, base.size());
    bigint started(now);

    // step 1: find relations
    vector<int> *possible_factors = new vector<int>[BULK];
    vector<pair<unsigned long long, vector<pair<int, unsigned int> > * > > relation_pointers;
    int cnt = 0;
    const int object = max((int)(base.size() * 1.15), 100);
    unsigned long long dist_from_start = 0;
    while (cnt <= object) {
        for (int i=0; i<BULK; ++i) {
            possible_factors[i].clear();
        }
        for (const auto &[p, x]: base) {
            int start = now % p;
            for (int i = x >= start ? x - start : p + x - start; i <= BULK; i += p) {
                possible_factors[i].push_back(p);
            }
            if (2 * x != p) {
                for (int i = (p - x) >= start ? p - x - start : 2 * p - x - start; i <= BULK; i += p) {
                    possible_factors[i].push_back(p);
                }
            }
        }
        vector<pair<int, unsigned int> > *factorization = new vector<pair<int, unsigned int> >;
        bool changed = false;
        for (int i=0; i<BULK; ++i) {
            bigint temp(now_value);
            for (const auto &p: possible_factors[i]) {
                unsigned int c = 0;
                while (temp % p == 0) {
                    temp /= p;
                    ++c;
                }
                factorization->emplace_back(p, c);
            }
            if (temp == 1U) {
                relation_pointers.emplace_back(dist_from_start, factorization);
                factorization = new vector<pair<int, unsigned int> >;
                ++cnt;
                changed = true;
            } else {
                factorization->clear();
            }
            ((now_value += now) += now) += 1U;
            now += 1U;
            ++dist_from_start;
        }
        if (changed) {
            printf("\r%6.2lf%%, found %d", min(cnt, object) * 100ULL / (double)object, cnt);
            fflush(stdout);
        }
        delete factorization;
    }
    delete[] possible_factors;
    printf("\n");
    fflush(stdout);
    // step 2-a: trivial cutoff
    size_t last_count;
    do {
        last_count = relation_pointers.size();
        map<int, unsigned int> count_map;
        for (int i=0; i<cnt; ++i) {
            const auto &now_vector = *relation_pointers[i].second;
            for (const auto &[p, c]: now_vector) {
                if (c & 1) {
                    ++count_map[p];
                }
            }
        }
        const auto lowest = [] (const vector<pair<int, unsigned int> > &relation, const map<int, unsigned int> &counts) {
            unsigned int res = -1U;
            for (const auto &[p, c]: relation) {
                if (c & 1) {
                    res = min(res, counts.find(p)->second);
                }
            }
            return res;
        };
        sort(relation_pointers.begin(), relation_pointers.end(), [=] (const auto &u, const auto &v) {
            const unsigned int uval = lowest(*u.second, count_map);
            const unsigned int vval = lowest(*v.second, count_map);
            return uval > vval;
        });
        while (!relation_pointers.empty() && lowest(*relation_pointers.back().second, count_map) == 1) {
            relation_pointers.pop_back();
        }
    } while (last_count != relation_pointers.size());
    printf("trivial cutoff: %lu\n", relation_pointers.size());
    // step 2-b: matrix construction and wiedemann
    vector<int> reduced_primes;
    for (int i=0; i<cnt; ++i) {
        const auto &now_vector = *(relation_pointers[i].second);
        for (const auto &[p, c]: now_vector) {
            if (c & 1) {
                reduced_primes.push_back(p);
            }
        }
    }
    sort(reduced_primes.begin(), reduced_primes.end());
    reduced_primes.erase(unique(reduced_primes.begin(), reduced_primes.end()), reduced_primes.end());
    printf("# of reduced primes: %lu\n", reduced_primes.size());
    if (reduced_primes.size() >= relation_pointers.size()) {
        fprintf(stderr, "ERROR: # of reduced primes is bigger than or equal to # of relation\n");
        fprintf(stderr, "try again with more relations\n");
        return -1;
    }
    while (true) {
        vector<pair<int, int> > matrix;
        for (int i=0; i<reduced_primes.size(); ++i) {
            const auto &v = *relation_pointers[i].second;
            for (const auto &[p, c]: v) {
                if (c & 1) {
                    const int j = distance(reduced_primes.begin(), lower_bound(reduced_primes.begin(), reduced_primes.end(), p));
                    matrix.emplace_back(j, i);
                }
            }
        }
        sort(matrix.begin(), matrix.end()); // for cache hit
        bool *result = new bool[reduced_primes.size()];
        memset(result, 0, sizeof(bool) * reduced_primes.size());
        for (int j=0; j<10; ++j) {
            bigint xinv(x);
            int u = (int)x.size() * 5 + 4;
            xinv.invert(u);
            if (wiedemann((int)reduced_primes.size(), matrix, result)) {
                bigint left_side(1), right_side(1);
                map<int, unsigned int> counts;
                for (int i=0; i<reduced_primes.size(); ++i) {
                    if (result[i]) {
                        bigint temp(relation_pointers[i].first);
                        temp += started;
                        modmul(left_side, temp, x, xinv, u, left_side);
                        for (const auto &[p, c]: *relation_pointers[i].second) {
                            counts[p] += c;
                        }
                    }
                }
                for (const auto &[p, tc]: counts) {
                    bigint temp;
                    modpow(p, tc >> 1, x, xinv, u, temp);
                    modmul(right_side, temp, x, xinv, u, right_side);
                }
                bigint temp(left_side);
                temp += right_side;
                if (!(temp == x) && !(left_side == right_side)) {
                    bigint temp2;
                    gcd(temp, x, temp2);
                    temp2.print();
                    puts("");
                    return 0;
                }
            }
        }
        delete[] result;
        printf("attempt failed, shuffling...\n");
        shuffle(relation_pointers.begin(), relation_pointers.end(), mt);
    }
    return 0;
}
```

[/expand]

### 테스트

몇 가지 테스트 결과를 봅시다.

```
$ time echo 340282366920938463463374607431768211457 | ./main
B = 46530, prime # = 2394
100.00%, found 2771
trivial cutoff: 2539
# of reduced primes: 2313
59649589127497217

real    0m36.290s
user    0m36.232s
sys     0m0.051s
```

입력으로 주어진 수는 일곱 번째 페르마 수 $2^{2^{7}} + 1$이고, 실제로\\\[
    2^{2^{7}} + 1 = \color{blue}{59649589127497217} \cdot 5704689200685129054721
\\\]이기 때문에 약수를 잘 구했다고 할 수 있습니다. 전체 시간 중 대부분의 시간을 relation을 구하는 데 사용했습니다.

```
$ time echo 10315820593624901285660301591780405139431637 | ./main
B = 94691, prime # = 4522
100.00%, found 5221
trivial cutoff: 4785
# of reduced primes: 4314
attempt failed, shuffling...
attempt failed, shuffling...
4587948617830910535641

real    2m56.709s
user    2m56.578s
sys     0m0.130s
```

입력으로 주어진 $N = 10315820593624901285660301591780405139431637$은 제가 만들어낸 수이고, 2019년 Number Theoretic Algorithms 스터디를 진행할 때 만들었습니다. 원문이 궁금하신 분들은 [이곳](https://site.thekipa.com/nta-2019/week6.pdf)을 보시면 됩니다. 만들 당시만 해도 WolframAlpha가 소인수분해를 굉장히 힘겹게 했었는데, 이제 캐시되었는지 성능이 좋아졌는지 곧바로 하더라구요. 무서운 놈들...

원문의 문제는 이것으로 RSA 암호를 푸는 것이었지만, 여기에서는 소인수분해 결과를 위주로 살펴봅시다. 일단 결과는\\\[
    N = 2248460358412211896157 \cdot \color{blue}{4587948617830910535641}
\\\]이기 때문에 맞습니다.

전체 시간 중 절반 정도를 relation 찾는 데에, 절반 정도를 실제로 푸는 데에 사용했는데, Wiedemann algorithm이 random algorithm이다 보니 좋은 해를 곧바로 찾으면 빨리 끝나고 그렇지 않으면 벡터들을 섞으면서 다른 kernel vector를 찾을 수 있도록 코드를 작성했습니다. Block Wiedemann algorithm을 사용하면 같은 시간에 더 많은 결과를 볼 수 있기 때문에, 더욱 안정적이고 빠르게 찾을 수 있을 것으로 기대됩니다.

```
$ time echo 157513841666999107978961658317028523253878748139938874167 | ./main
B = 638663, prime # = 25970
100.00%, found 29866 
trivial cutoff: 27053
# of reduced primes: 24817
29601658021629044173527313547

real    85m53.319s
user    85m53.032s
sys     0m0.271s
```

입력으로 주어진 $N = 157513841666999107978961658317028523253878748139938874167$은 [이 사이트](https://asecuritysite.com/encryption/random3?val=96)에서 96-bit 소수 두 개를 곱해서 만들었습니다. 결과는\\\[
    N = 5321115511567239427157507461 \cdot \color{blue}{29601658021629044173527313547}
\\\]이기 때문에 맞습니다. 특히 이 경우는 [Pollard's rho](https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm)가 사이클 위를 1초에 $10^{9}$번 돌아다닌다고 해도 소요 시간의 중앙값이 14시간 2분 42초이기 때문에, QS의 강력함을 잘 보여줍니다.

![WolframAlpha는 계산하지 못합니다.](/assets/images/speedup-using-factor-base/cantcompute.png)

80분 정도 관계를 찾았고, Wiedemann algorithm은 5분 정도 돌았습니다. 이 글의 주제와는 크게 관련이 없지만, 둘 모두 병렬화할 수 있기에 병렬화하면 또 얼마나 빨라질지도 기대됩니다.

# 마치며

이상으로 factor base를 활용한 속도 향상에 대해 살펴보았습니다. 문제가 곱셈적일 경우 factor base를 구성하는 것은 시간 복잡도 상으로도, 그리고 실제로도 속도 향상에 큰 도움이 됩니다.
