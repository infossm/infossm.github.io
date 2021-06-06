---
layout: post
title: Factor Base를 이용한 "어려운 문제"의 속도 개선
date: 2021-06-03 04:00
author: kipa00
tags: [algorithm]
---

asdf

# "어려운 문제"?

이 글에서의 **어려운 문제**는 크게 소인수분해와 이산 로그를 일컫습니다.

소인수분해 문제는 잘 알고 계실 것입니다.
> 수 $N$이 주어지면, $N$을 (서로 다를 필요는 없는) 소수 $p_{1}$, $p_{2}$, $\cdots$, $p_{n}$의 곱 $N = p_{1} p_{2} \cdots p_{n}$으로 분해하라.

예를 들어 $108 = 2 \cdot 2 \cdot 3 \cdot 3 \cdot 3 = 2^{2} \cdot 3^{3}$과 같은 식으로 분해할 수 있습니다. 조금 더 복잡한 예제로, $10001 = 73 \cdot 137$과 같이 분해됩니다. $n = 1$이어서 "분해할 수 없는" 경우도 있습니다. 소인수분해는 (순서를 무시하면) 유일하다는 것도 다들 잘 알고 계실 것입니다.

이산로그 문제는 다음과 같습니다.
> 고정된 수 $N$에 대해 $\mathbb{Z} _ {N}^{\times}$의 생성자인 $g$와 $x \in \mathbb{Z} _ {N}^{\times}$가 주어지면, $g^{n} = x$인 $n \in \mathbb{Z}/\phi(N)\mathbb{Z}$을 찾아라.

정의를 차근차근 이해해 봅시다. 먼저 $\mathbb{Z} _ {N}^{\times}$는 $N$과 서로소인 $N$ 미만의 음이 아닌 정수들을 모은 [군](https://en.wikipedia.org/wiki/Group_(mathematics))입니다. 왜 서로소만을 모았냐면, 위에 있는 작은 기호가 시사하듯, 연산으로 곱하기 $\times$를 사용할 것이기 때문입니다. 다만 모든 원소가 $N$ 미만이기 때문에, $x \times y := x \cdot_{\mathbb{Z}} y \mod N$이 되도록 정의합니다. ($\cdot_{\mathbb{Z}}$는 정수 곱하기입니다.) 그러면 이 집합이 실제로 군이 됨을 보일 수 있습니다. 즉 **모든 원소에 대해 역원이 존재합니다**.

$g$는 [생성자](https://en.wikipedia.org/wiki/Generating_set_of_a_group)로 정의되었는데, 군 $G$에 대해 생성자는 $\\{g^{i}: i \in \mathbb{Z}\\} = G$인 $g \in G$를 의미합니다. 즉, 앉아서 $g$만 곱하고 있어도 온 세상 $G$의 원소들을 다 만날 수 있다는 겁니다. 그렇다면 $g^{\phi(N)} = 1$이고, 모든 $0 \leq i \le \phi(N)$에 대해 $g^{i}$들이 전부 다를 수밖에 없습니다. 여기서 [group isomorphism](https://en.wikipedia.org/wiki/Group_isomorphism) $f: \mathbb{Z}/\phi(N)\mathbb{Z} \rightarrow \mathbb{Z} _ {n}^{\times}$를 $f(i) = g^{i}$와 같이 정의합니다. $f$는 isomorphism이니 역함수가 존재합니다. 이 문제는 바로 이 $f$의 **역함수**의 특정 지점에서의 값, 즉 $f^{-1}(x)$을 찾으라는 것입니다.

조금 많이 수학적으로 접근한 것 같은데, 수학 하시는 분들은 이렇게 두 군을 같은 것으로 보고 (같은 것이니) 더 이상 건드리지 않습니다. 하지만 우리는 이 값을 실제로 구하려고 했고, 그러다 보니 $f$를 계산하는 경우와는 달리 어려움에 마주하게 된 것입니다. 위의 말이 하나도 이해가 안 되시는 분들을 위해 쉬운 버전을 준비했습니다.

> 고정된 수 $N$에 대해 $N$ 미만의 음이 아닌 정수 $g$가, 임의의 $N$과 서로소인 $N$ 미만의 음이 아닌 정수 $x$에 대해 $g^{i}$를 $N$으로 나눈 나머지가 $x$인 $i \in \mathbb{Z}$가 존재한다고 한다. $x$가 주어지면, $\phi(N)$ 미만의 음이 아닌 정수 $i$를 실제로 구하라.

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

**일반적으로** 어떤 수로 나누어떨어지는 수가 작을수록 많기 때문에, factor base에 속하는 수는 작을수록 좋습니다. 하지만 작다고 모두 좋은 것은 아닙니다: 합성수의 경우, 더 작은 소수가 반드시 존재하기 때문에 그 소수를 삼는 것이 이득입니다. (물론 어떤 합성수를 취하거나 어떤 큰 소수를 취하는 것이 relation을 빨리 찾는 데 도움을 줄 수 있겠지만, 그렇게 되면 $N$과 $\omega$가 주어졌을 때 그것을 알아내는 방법도 알아야 하고, 논의가 복잡해지므로 생각하지 않겠습니다.) 그래서 우리는 factor base를 **$B$ 이하의 모든 소수**로 선택하겠습니다.

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

```
3177 / 3178
100.00%
******************

real    19m10.651s
user    19m10.391s
sys     0m0.260s
```

# Dixon's factorization algorithm