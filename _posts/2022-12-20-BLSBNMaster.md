---
layout: post

title: "Optimal Ate Pairings, BLS12-381, BN254 알아보기"

date: 2022-12-5

author: rkm0959

tags: [cryptography]
---

[이전 글](https://infossm.github.io/blog/2022/11/22/PairingMaster/)에서 이어서 가겠습니다. 

# Optimal Ate Pairings

[[Ver09]](https://eprint.iacr.org/2008/096.pdf)의 내용입니다. 다시 Tate Pairing으로 돌아가서, 

$$(f_{s, P}) = s(P) - ([s]P) - (s-1) \mathcal{O}$$

인 함수 $f_{s, P}$를 생각하면, reduced Tate Pairing 

$$t_r(P, Q) = f_{r, P}(Q)^{(q^k - 1) / r}$$

를 얻을 수 있었습니다. 이제 이를 최적화하는 Ate Pairing을 알아봅시다. 

**Lemma**

$$f_{ab, Q} = f_{a, Q}^b \cdot f_{b, [a]Q}$$

증명: 단순 계산. 어렵지 않습니다. 

그러면 특히 $[r]Q = \mathcal{O}$인 경우에는 

$$f_{mr, Q} = f_{r, Q}^m$$

가 성립하고, 그러니 $m$이 $r$의 배수가 아니면 

$$t_r(Q, P)^m = f_{mr, Q}(P)^{(q^k - 1) / r}$$

도 non-degenerate pairing이 됩니다. 

여기서 $\lambda = q \bmod r$이라고 합시다. 그러면 $\lambda^k \equiv 1 \bmod{r}$이므로 $m = (\lambda^k - 1) / r$이라 둘 수 있고, 

$$t_r(Q, P)^m = f_{\lambda^k - 1, Q}(P)^{(q^k - 1) / r}$$

이며, 여기서 $[\lambda^k - 1]Q = \mathcal{O}$임을 감안하고 계산하면 

$$t_r(Q, P)^m = f_{\lambda^k, Q}(P)^{(q^k - 1) / r}$$

임을 얻습니다. 

여기서 Lemma 및 수학적 귀납법을 사용해서 

$$f_{\lambda^k, Q}(P) = \prod_{i=0}^{k-1} f_{\lambda, [\lambda^i] Q}^{\lambda^{k-1-i}}(P)$$

임을 얻고, $[r]Q = \mathcal{O}$이니 

$$[\lambda^i]Q = [q^i]Q$$

임을 사용하면 

$$f_{\lambda^k, Q}(P) = \prod_{i=0}^{k-1} f_{\lambda, [q^i] Q}^{\lambda^{k-1-i}}(P)$$

인데, 여기서 

- $P$에 Frobenius Endomorphism을 써도 그대로 
- $Q$에 Frobenius Endomorphism을 쓰면 $q$배 

가 됨을 가지고 천천히 생각을 해보면 결국 전부 Frobenius Endomorphism으로 덮을 수 있음을 알 수 있고, 결국 

$$f_{\lambda^k, Q}(P) = \prod_{i=0}^{k-1} f_{\lambda, [q^i] Q}^{\lambda^{k-1-i}}(P) = f_{\lambda, Q}^{\sum_{i=0}^{k-1} \lambda^{k-1-i} q^i}(P)$$

이러면 지수가 $r$의 배수가 아닌 경우, 

$$a_\lambda (Q, P) = f_{\lambda, Q}(P)^{(q^k - 1) / r}$$

도 non-degenerate pairing임을 의미합니다. 

여기서 좋은 점은 $r$ 대신 $\lambda$를 쓸 수 있다는 것입니다. 이를 ate pairing이라고 부릅니다. 

한편, 여기서 $\lambda$의 크기를 생각해보면, $k$가 $q^k \equiv 1 \pmod{r}$을 만족하는 최소의 $k$이므로 $\Phi_k(q) \equiv 0 \pmod{r}$이 성립하고, 이는 $\lvert \Phi_k(\lambda) \rvert \ge r$을 의미합니다. 이러면 대강 

$$\log \lambda  \ge  \frac{1}{\phi(k)} \cdot \log r$$

정도를 예상할 수 있고, Miller Loop의 길이가 

$$\frac{1}{\phi(k)} \cdot \log r$$

정도면 훌륭함을 알 수 있습니다. 이 threshold가 "optimal"의 기준입니다. 즉, 기존 Miller Loop의 길이보다 $1/\phi(k)$의 상수 커팅이 이루어지면, optimal pairing이라고 부르게 됩니다. 

하지만 ate pairing만으로는 이 값에 도달하기 어렵습니다. 조금 더 노력해야합니다. 

이제 Optimal Ate Pairing을 알아봅시다. 

역시, 이번에도 핵심은 Frobenius Endomorphism을 최대한 활용하는 것입니다. 이를 위해서, 이번에는 $r$의 배수가 아닌 $m$을 잡고 $\lambda = mr = \sum_{i=0}^l c_i q^i$라고 썼을 때 $c_i$들이 작기를 바랍니다. 이 경우, 

$$a_{[c_0, \cdots, c_l]}(Q, P) = \left( \prod_{i=0}^l f_{c_i, Q}^{q^i}(P) \cdot \prod_{i=0}^{l-1} \frac{l_{[s_{i+1}]Q, [c_iq^i]Q}(P)}{v_{[s_i]Q}(P)} \right)^{(q^k - 1) / r}$$

는 bilinear pairing입니다. 단, $s_i = \sum_{j=i}^l c_jq^j$. 특히, 

$$mkq^{k-1} \not\equiv ((q^k - 1) / r) \cdot \sum_{i=0}^l i c_i q^{i-1} \pmod{r}$$

인 경우, 이 pairing은 non-degenerate pairing입니다. 

기본적으로, 증명은 

$$f_{a+b, Q} = f_{a, Q} \cdot f_{b, Q} \cdot \frac{l_{[a]Q, [b]Q}}{v_{[a+b]Q}}$$

를 사용한 다음, 

$$f_{c_iq^i, Q}(P) = f^{c_i}_{q^i, Q}(P) \cdot f_{c_i, [q^i]Q}(P) = f^{c_i}_{q^i, Q}(P) \cdot f_{c_i, Q}^{q^i}(P)$$

임을 사용하여 얻습니다. 여기서 Frobenius Endomorphism을 썼습니다.

이러면 

$$t(Q, P)^m = \left( \prod_{i=0}^{l-1} f^{c_i}_{q^i, Q}(P) \right)^{(q^k - 1) / r} \cdot a_{[c_0, \cdots, c_l]}(Q, P)$$

를 얻고, $f_{q^i, Q}(P)$가 모두 pairing임은 앞서 증명한 것과 동일한 방식으로 보일 수 있습니다. 

그러니 $a_{[c_0, \cdots, c_l]}$ 역시 pairing 입니다. 이때, non-degenerate 여부는 양쪽을 reduced ate pairing의 거듭제곱으로 표현하여 얻을 수 있습니다. 좋은 연습문제로 남기겠습니다. 여기서 $\lambda = \phi_k(q)$가 되도록 $m$을 잡는 방법을 고려해볼 수 있는데, 이 경우 degenerate pairing이 나오게 됩니다. 이 역시 어렵지 않으므로, 연습문제로 남기겠습니다. 식 형태가 다항식의 미분이어서 생각나는 그대로 가면 증명이 됩니다.

그래서 실제로는 바로 $\lambda = \phi_k(q)$로 갈 수는 없고, 격자를 활용하여 $\lambda$를 잡을 생각을 해야합니다. 

격자의 디자인 자체는 꽤 직관적인 편이며, Minkowski's Theorem을 생각하면 여기서 $l^\infty$-norm이 "optimal"한 instance가 있음은 확인할 수 있습니다. 그런데 사실 $l^\infty$-norm에 대한 optimality는 의미가 없을 수 있는 게, 사실 여기서 생각하는 연산의 횟수는 $l^1$-norm이 중요하기 때문입니다. $l^\infty$-norm에서 생각하고 싶다면, parallel computation이 가능하다는 가정이 깔려있어야 합니다. 

이 문제를 해결하기 위해서 사용되는 방법은 실제로 우리가 사용하는 많은 타원곡선에 대해서는 $q, r$등 여러 값이 parameter에 대한 다항식 형태로 나오기 때문에, 이에 대해 직접 $c_i$들을 계산할 수 있다는 점을 활용하는 것입니다. 이는 BN254 타원곡선에 대한 Optimal Ate Pairing에서 등장하게 되므로, 거기서 따로 소개하도록 하겠습니다. 

한편, 위와 같은 격자에서 $l^2$, $l^\infty$ norm이 최소인 벡터는 Minkowski's Theorem에서 얻어지는 bound를 크게 outperform 할 수 없음을 적당한 대수학으로 증명할 수 있습니다. 이에 대해서는 [[Ver09]](https://eprint.iacr.org/2008/096.pdf) 참고.

# BLS12-381

[BLS12-381](https://hackmd.io/@benjaminion/bls12-381#BLS12-381-For-The-Rest-Of-Us) 타원곡선은 

$$E: y^2 = x^3 + 4$$

로 표현되며, $x$가 `-0xd201000000010000`일 때 

$$q = \frac{1}{3} (x-1)^2 (x^4 - x^2 + 1) + x$$

$$r = x^4 - x^2 + 1$$

입니다. 여기서 바로 알 수 있는 좋은 점은 

- $x$가 low hamming weight를 가집니다. Miller Loop에서 최적화가 됩니다.
- 128 bit security를 가집니다. $r$이 대강 256 bit 소수라서 그렇습니다. 
- FFT를 가능하게 하기 위해서 $r - 1$이 가지는 $2$의 개수가 큽니다. 

또한, 이 타원곡선에서는 $k = 12$입니다. 참고로 여기서 

$$q \bmod r = x = \lambda, \quad \Phi_{12}(\lambda) = r$$

이므로 ate pairing만 사용해도 바로 optimality를 얻음을 알 수 있습니다. 

## Sextic Twist

여전히 $G_1$은 

$$E(\mathbb{F}_q)$$ 

안에 있어서 문제가 없는데, $G_2$가 문제입니다. 

$$\mathbb{F}_{q^{12}}$$ 

위에서 계산하는 게 문제인데, 다행히 이 형태의 타원곡선에서는 sextic twist를 활용해 최적화를 할 수 있습니다.

$$u^6 = (1 + i)^{-1}$$

인 $u$를 잡고, $E$에 속하는 $(x, y)$를 

$$(x, y) \rightarrow (x', y') = (x / u^2, y / u^3)$$

라는 변환을 적용시키면 

$$E' : y'^2 = x'^3 + 4(1+i)$$

라는 타원곡선으로 가게 됩니다. 여기서 재밌는 점은 이게 homomorphism이라는 것입니다. 

이는 단순 계산으로 증명할 수 있습니다. 더 좋은 점은 trace zero subgroup $G_2$에 이 mapping을 적용하면 

$$E'(\mathbb{F}_{q^2})$$

에서 계산을 해도 됨을 알 수 있다는 것입니다. 즉, 모든 연산을 $\mathbb{F}_{q^2}$에서 한 다음, 다시 mapping을 역산해서 원래 타원곡선으로 돌아가면 됩니다. 이렇게 해서 연산량을 크게 줄일 수 있습니다.

# BN254

[BN254](https://hackmd.io/@jpw/bn254#BN254-For-The-Rest-Of-Us) 타원곡선은 

$$E: y^2 = x^3 + 3$$

이며, $x$가 `4965661367192848881`일 때 

$$q = 36x^4 + 36x^3 + 24x^2 + 6x + 1$$

$$r = 36x^4 + 36x^3 + 18x^2 + 6x + 1$$

가 성립하는 타원곡선입니다. 역시 $k = 12$입니다.

## Sextic Twist

BLS12-381과 마찬가지로, 

$$u^6 = 9 + i$$

인 $u$를 잡고 homomorphism

$$(x, y) \rightarrow (x', y') = (x / u^2, y / u^3)$$

을 적용하면 마찬가지로 연산을 최적화할 수 있습니다. 

## Optimal Ate Pairing

여기서 Optimal Ate Pairing을 유도하려면 

$$(r, 0, 0, 0), \quad (-q, 1, 0, 0), \quad (-q^2, 0, 1, 0), \quad (-q^3, 0, 0, 1)$$

을 가지고 격자를 만든 다음에 짧은 벡터를 찾으면 됩니다. 여러 방법이 있으나 

$$(6x + 2, 1, -1, 1)$$

이 여기에 속함을 사용하여 Optimal Ate Pairing을 구축합니다. 

이제 $\mathbb{F}_q$에 있는 타원곡선을 $E_1$, extension field에 있는 (sextic twist 적용) 타원곡선을 $E_2$라 하고,
- Hashing to Curve
- Cofactor Clearing
- Subgroup Check

에 대한 이야기를 다루어보도록 하겠습니다. 두 타원곡선의 경우에서 모두 우리의 목표는 결국 

$$E_1(\mathbb{F}_q), \quad E_2(\mathbb{F}_{q^2})$$

에 hash를 한 다음 (Hashing to Curve), 추가적인 처리로 subgroup으로 mapping 하는 것 (Cofactor Clearing)이 목표입니다. 그 후, 필요하면 우리가 제대로 subgroup의 원소를 얻었는지 확인을 (Subgroup Check) 합니다. 

# Hashing to Curve

Elliptic Curve로 hashing을 하는 것은 그 자체로 매우 중요한 문제고 여러 연구가 진행된 문제입니다. 가장 직관적인 방식은 $H(m)$을 $x$로 갖는 타원곡선의 점으로 hash 하는 것이지만, 모든 $x$ 값이 valid 한 타원곡선의 점이 아니기 때문에 말처럼 쉽지가 않습니다. 이 경우, 타원곡선의 점이 나올 때까지 "다시 시도할 방법"을 하나 고안해서 이를 반복해야 합니다. 이러면 hash가 가능하기는 한데, constant time이 아니므로 여러모로 아쉬움이 남게 됩니다. 

이를 해결하기 위해 나온 방법이 SWU map 입니다. 대강의 흐름을 설명하자면 

- SW map이라는 임의의 Weierstrass Curve에 대해 가능한 방법이 있음
- Simplified SWU map이라는 $ab \neq 0$인 Weierstrass Curve에 대해 가능한 방법이 있음 
- Simplified SWU map을 modify 해서 $ab = 0$일 때도 사용할 수 있음

[IETF](https://datatracker.ietf.org/doc/id/draft-irtf-cfrg-hash-to-curve-06.html)에도 관련 내용이 있습니다. 참고하세요. 

BLS12-381이나 BN254나 전부 j-invariant가 $0$인, $a = 0$에 해당하는 타원곡선이기 때문에 둘 다 변형된 Simplified SWU map을 이용할 수 있습니다. 해당 논문은 [[WB19]](https://eprint.iacr.org/2019/403.pdf)입니다. 이 논문이 BLS12-381에 대한 설명을 하므로, BLS12-381에 대해서 설명하겠습니다. BN254에서도 큰 차이는 없을 것으로 생각됩니다.

## The Shallue–van de Woestijne map

목표로 하는 타원곡선이 

$$y^2 = f(x) = x^3 + ax + b$$

라고 합시다. 기본적인 아이디어는 

$$V = \{(x_1, x_2, x_3, x_4) : f(x_1)f(x_2)f(x_3) = x_4^2 \}$$

라고 하면, $V$ 위의 점에 대해서 $x_1, x_2, x_3$ 중 하나는 valid 한 타원곡선의 $x$ 좌표라는 점입니다. 이제 $V$의 원소 하나로 가면 됩니다. 이를 위해서, 새로운 surface 

$$S = \{(u, v, y) : y^2(u^2 + uv + v^2 + a) = -f(u) \}$$

를 준비합니다. 여기서 다음 mapping

$$\phi_1: (u, v, y) \rightarrow \left(v, -u-v, u+y^2, f(u+y^2) \cdot \frac{u^2 + uv + v^2 + a}{y} \right)$$

이 valid 한 $S \rightarrow V$ mapping임을 계산을 열심히 해서 증명할 수 있습니다. 또한, $\phi_1(u, v, y)$의 값을 가지고 $u, v, y$를 역으로 계산할 수 있음도 쉽게 알 수 있습니다. 그러니 $S$의 원소 하나로 가면 됩니다. 

이제 $u = u_0$를 하나 고정하되, $f(u_0) \neq 0$이고 $3u_0^2 + 4a \neq 0$이도록 하면 $S$를 

$$\left( y \left( v + \frac{1}{2} u_0 \right) \right)^2 + \left( \frac{3u_0^2 + 4a}{4} \right) y^2 = -f(u)$$

로 쓸 수 있고, $w = y(v + u_0/2)$라 하면 

$$w^2 + \left( \frac{3u_0^2 + 4a}{4} \right) y^2 = -f(u)$$

가 됩니다. 이는 유리수 점이 있음을 쉽게 증명할 수 있고 (e.g. Cauchy-Davenport) 이를 기반으로 $(w, y)$를 변수 $t$에 대한 유리식으로 parameterize 할 수 있고, 다시 이를 $(v, y)$로 바꾸면 이 역시 $t$에 대한 유리식으로 parameterize 할 수 있습니다. 즉, 종합하면 우리는 

$$\mathbb{A}^1 \rightarrow S \rightarrow V : \quad t \rightarrow (u, v, y) \rightarrow \left(v, -u-v, u+y^2, f(u+y^2) \cdot \frac{u^2 + uv + v^2 + a}{y} \right)$$

를 거쳐서 $V$의 원소를 얻고, $V$의 원소 $(x_1, x_2, x_3, x_4)$에서 $x_i$가 valid 한 타원곡선의 $x$ 좌표가 되는 최소의 index $i$를 잡은 다음 $x_i$를 선택하는 방식으로 hash를 할 수 있게 됩니다. 

BLS12-381에 이를 직접 적용하려면 특수 케이스 처리를 해야하는데, 이는 여기서는 생략하겠습니다. 

## The simplified Shallue–van de Woestijne–Ulas map 

여기서 사용되는 아이디어는 $u$를 non-square로 잡고 

$$f(ux) = u^3 f(x)$$

가 성립하는 $x$를 찾는 것입니다. 이를 풀면 

$$x = -\frac{b}{a} \left(1 + \frac{1}{u^2 + u} \right)$$

가 나옵니다. 여기서 $ab \neq 0$이어야 하는 이유를 알 수 있습니다.

특히, $q \equiv 3 \pmod{4}$를 가정하면 $u = -t^2$을 잡을 수 있고 

$$x = -\frac{b}{a} \left(1 + \frac{1}{t^4 - t^2} \right)$$

를 얻습니다. 이렇게 되면 $x$ 또는 $ux = -t^2x$가 valid한 타원곡선의 $x$ 좌표가 됨을 알 수 있습니다. 

물론 $t = -1, 0, 1$이면 그냥 $\infty$로 보내면 됩니다.

## An optimized SWU map for BLS12-381

해결해야 하는 문제는 
- Field Size가 $3 \pmod{4}$여야 한다는 제약조건 
- $ab \neq 0$이라는 제약조건

입니다. 그런데 사실 Field Size가 $3 \pmod{4}$라는 조건은 $-1$이 non-square라는 것에서만 사용했으니, 단순히 non-square $\xi$를 하나 잡기만 하면 됩니다. 문제는 $ab \neq 0$이라는 조건을 피하는 것입니다. 이를 해결하기 위해서 다른 j-invariant를 갖는 타원곡선을 하나 잡고 isogeny를 사용합니다. 

BLS12-381의 경우, $E_1$을 위해서는 11-isogenous curve를 잡은 다음 Velu의 공식을 이용하여 $E_1$으로 hash 할 수 있고, $E_2$의 경우에는 3-isogenous curve를 잡을 수 있습니다. 

여기에 추가적인 최적화를 몇 개 할 수 있는데, 이는 계산적인 면이므로 생략하겠습니다. 

논문에서 나온 예시를 간단한 것 하나만 들어보면,  

$$(U/V)^{(p+1)/4} = UV(UV^3)^{(p-3)/4}$$

를 사용해서 나눗셈 + 제곱근 계산을 거듭제곱 하나로 대체할 수 있습니다.

## Indifferentiable Hashing

참고문헌으로 [[BCI+10]](https://eprint.iacr.org/2009/340.pdf)가 있습니다. [[FFS+13]](https://eprint.iacr.org/2010/539.pdf), [[KT15]](https://hal.inria.fr/hal-01275711/document)도 있습니다. 

자세한 이론적인 내용은 추후에 Indifferentiable Hashing에 대한 논문을 깊이있게 읽고 작성해보도록 하겠습니다. 

대충 핵심만 정리해보면, 위 hash function은 나름 괜찮지만 "진짜" random, 즉 random oracle과는 어렵지 않게 distinguish 할 수 있습니다. hash function 결과가 uniform 하지도 않으며, 특히 $E$의 모든 점으로 가는 것도 아닙니다. 이 문제를 해결하기 위해서 hash를 "indifferentiable from random oracle"하게 만드는 방식이 필요합니다. 

생각해보면 지금 저희의 방식은 

$$\{0, 1\}^\star \rightarrow \mathbb{F}_p \rightarrow E: \quad m \rightarrow h(m) \rightarrow H(m) = f(h(m))$$

라고 볼 수 있는데, [[BCI+10]](https://eprint.iacr.org/2009/340.pdf)의 결론은 

$$H(m) = f(h_1(m)) + f(h_2(m))$$

$$H(m) = f(h_1(m)) + h_2(m) G$$

등의 construction을 사용하면 indifferentiable hashing을 얻을 수 있다는 것입니다.

추가로 Icart의 Hashing to Elliptic Curve를 다루는 2009년 논문도 유명합니다. [[Ica09]](https://eprint.iacr.org/2009/226.pdf)

이제 남은 것은 $E_1, E_2$에 온 점들을 $G_1, G_2$로 보내는 것입니다. 즉, 타원곡선 전체와 subgroup 그 자체에 대한 이야기입니다. 이야기를 시작하기 전에, BN254의 경우 $E_1 = G_1$이 성립하여 cofactor clearing이나 subgroup check을 할 필요가 없다는 것을 짚고 넘어갑시다. 

# Cofactor Clearing

Cofactor Clearing은 기본적으로 scalar multiplication으로 가능합니다. 사실 cofactor를 곱하기만 하면 되는데, 이것을 빠르게 하기 위해서 노력하는 게 이 주제의 큰 틀이라고 볼 수 있습니다. 

## BLS12-381, $G_1$

다시 BLS12-381을 보면 $x$가 `-0xd201000000010000`일 때 

$$q = \frac{1}{3}(x-1)^2(x^4 - x^2 + 1) + x$$

$$r = x^4 - x^2 + 1$$

인데, 여기서 $n = (1 - x) / 3$라고 하면 $E(\mathbb{F}_q)$의 구조는 

$$\mathbb{Z}_{3rn} \oplus \mathbb{Z}_{n}$$

임을 알 수 있습니다. 그래서 사실 cofactor를 곱할 때 

$$3n^2 = \frac{1}{3}(x-1)^2$$

를 곱할 필요가 없고, 그냥 $1-x$를 곱하기만 하면 됩니다. 이는 64bit 정수이고 심지어 hamming weight까지 작기 때문에, 그냥 곱해도 충분히 빠릅니다. [[WB19]](https://eprint.iacr.org/2019/403.pdf) 역시 이 방식을 사용하고 있습니다.

필요하면 j-invariant가 0인 타원곡선에서 나오는 mapping도 활용합니다. (아래 GLV 참고) 
- 이 mapping은 뒤에도 사용이 되므로 그때 다루겠습니다.

## BLS12-381, $G_2$

이 경우에 대한 논문은 [[BP17]](https://eprint.iacr.org/2017/419.pdf)입니다. 설명에 앞서 scalar multiplication 최적화의 paradigm 중 하나인 [GLV](https://www.iacr.org/archive/crypto2001/21390189.pdf) 에 대해서 알아봐야합니다. Optimal Ate Pairing의 세팅과 매우 비슷한데, 결국 endomorphism $[\lambda]P$를 계산하는 것이 쉬운 경우, 이를 활용하기 위해서 $[r]P$를 계산할 때 $r$을 $1, \lambda, \lambda^2, \cdots$의 작은 선형결합으로 나타내기를 원하며, 이를 위해서 LLL 등 lattice reduction을 활용하는 것입니다. cofactor clearing에서도 쓸 수 있겠죠?

그렇다면 여기서 사용되는 endomorphism은 무엇일까요? 다시 Frobenius Endomorphism입니다. 정확하게는, 현재 우리가 sextic twist 위에서 작업하고 있으므로, untwist-Frobenius-twist입니다. 이를 $\phi$라 하면 여전히 

$$\phi^2 - [t] \circ \phi + [q] = 0$$

이 성립합니다. 이 식이 항상 성립함을 가지고 cofactor clearing을 하려고 하면, 결국 

$$P \rightarrow (x^2 - x - 1)P + (x - 1) \phi(P) + \phi^2(2P)$$

를 거치면 $G_2$의 원소가 됨을 보일 수 있습니다. $E_2$의 abelian group 구조가 꽤 중요해 보이는데, 아직 확인을 못 했습니다. 다만 이 map이 실제로 잘 작동함은 SageMath로 쉽게 확인할 수 있습니다.

## BN254, $G_2$ 

BLS12-381처럼 untwist-Frobenius-twist를 활용하는 방향이 최선입니다.

특히, 여기서는 cofactor가 $q + t - 1$임을 계산으로 알 수 있기 때문에, 당장 

$$[t](\phi(P) + P) - \phi^2(P) - P$$

으로 계산해도 꽤 최적화가 됩니다. 이를 소개하는 것은 [[GS08]](https://eprint.iacr.org/2008/117.pdf).

# Subgroup Checks

여기에 대해서는 많은 논문이 있으나 제가 알기로는 [[DLZ+22]](https://eprint.iacr.org/2022/348.pdf)가 SOTA입니다.

추가로 참고할 논문은 [[HGP22]](https://eprint.iacr.org/2022/352.pdf), [[Sco21]](https://eprint.iacr.org/2021/1130.pdf) 정도가 있습니다. 옛날 논문 중에는 오류가 있는 것도 있다네요.

## BLS12-381, $G_1$

다시 BLS12-381의 parameter를 보면 

$$q = \frac{1}{3} (x-1)^2 (x^4 - x^2 + 1) + x$$

$$r = x^4 - x^2 + 1$$

입니다. 여기서 $q \equiv 1 \pmod{3}$임을 생각하면 

$$\psi: E \rightarrow E, \quad (x, y) \rightarrow (\beta x, y)$$

를 생각할 수 있습니다. 여기서 $\beta^3 \equiv 1 \pmod{q}$.

이 mapping이 subgroup $G_1$에서 

$$\psi(P) = \lambda P$$

라고 생각하면 $\lambda^2 + \lambda + 1 \equiv 0 \pmod{r}$이고 결국 $\lambda = -x^2$을 얻습니다. 즉, $P \in G_1$이면 

$$\psi(P) = -x^2 P$$

가 성립합니다. 재밌는 점은 역도 성립한다는 것입니다. 위 식을 가정하면 

$$\psi^3(P) = -x^6 P = P$$

이므로 $x^6+1 \equiv 0 \pmod{\text{ord}(P)}$입니다. $P \in E_1$이므로 $\text{ord}(P)$는 

$$\frac{1}{3}(x-1)^2 r$$

의 약수입니다. 그런데 만약 $P$가 $G_1$에 속하지 않는다면 

$$\text{ord}(P) = cr, \quad c \neq 1, \quad (x-1)^2 \equiv 0 \pmod{c}$$

를 얻습니다. 그런데 

$$x^6 + 1 \equiv 0 \pmod{c}$$

도 성립하므로 정리하면 $c = 1$을 얻을 수 있습니다. $x$의 값도 상수니까요. 그러니

$$\psi(P) = -x^2 P$$

만을 확인하면 subgroup membership check를 할 수 있습니다. 

## BLS12-381, $G_2$

여기서는 $E_1, E_2$ 각각에 대한 cofactor를 $h_1, h_2$라 하면 

$$\gcd(h_1, h_2) = 1$$

임을 이용합니다. 이미 $Q$가 $E_2$ 위에 있음을 알고 있으니 $\text{ord}(Q)$가 $h_2r$의 약수고,

$$h_1 r = (q - x) \equiv 0 \pmod{\text{ord}(Q)}$$

만 확인하면 충분합니다. 즉, 

$$qQ = xQ$$

가 성립하는지 확인하면 됩니다. 후자는 계산하기 쉬우니, 전자만 계산하면 됩니다.

이를 위해서 다시 untwist-Frobenius-twist mapping $\phi$를 가져오면 

$$\phi^2 - [t] \circ \phi + [q] = 0$$

이고 이를 정리하면 

$$qQ = \phi(tQ) - \phi^2(Q) = \phi(xQ) + \phi(Q) - \phi^2(Q)$$

를 얻습니다. 즉, 

$$\phi(xQ) + \phi(Q) - \phi^2(Q) = xQ$$

를 확인하는 것이 목표이며, $Q$가 $\phi - [1]$의 kernel에 속하지는 않으므로 이는 

$$\phi(Q) = xQ$$

를 확인하는 것과 동치입니다. $xQ$ 자체는 optimal ate pairing을 계산하면서 자연스럽게 계산됩니다.

## BN254, $G_2$ 

[[DLZ+22]](https://eprint.iacr.org/2022/348.pdf)의 더욱 최적화된 $G_2$ membership check를 소개하겠습니다.  

다시 BN254의 parameter를 소개하면 $x$가 `4965661367192848881`이며

$$q = 36x^4 + 36x^3 + 24x^2 + 6x + 1$$

$$r = 36x^4 + 36x^3 + 18x^2 + 6x + 1$$

입니다. BLS12-381에서 사용한 방식을 그대로 적용하면 

$$\phi(Q) = 6x^2 Q$$

가 성립하면 $Q \in G_2$임을 알 수 있습니다. 

여기서는 더욱 최적화를 하기 위해서 다음 정리를 증명합니다.

**Theorem 1, (Simplified) [DLZ+22]**: $\phi$가 untwist-Frobenius-twist라 하고 

$$\eta = \sum_{i=0}^s c_i q^i \equiv 0 \pmod{r}$$

$$f(\phi) = \sum_{i=0}^s c_i \phi^i, \quad g(\phi) = \phi^2 - t \cdot \phi + q$$

$$b_0 + b_1 \phi = f(\phi) \pmod{g(\phi)}$$

$$\gcd(b_0^2 + b_0 b_1 t + b_1^2 q, h_2 r) = r$$

이고 $Q \in E_2$면, $Q \in G_2$는 $f(\phi)(Q) = \mathcal{O}$와 동치다. 

**Proof of Theorem 1**. 

일단 $Q \in G_2$면 $\phi(Q) = qQ$니까 

$$f(\phi)(Q) = \eta Q = \mathcal{O}$$

가 성립합니다. 역으로, $f(\phi)(Q) = \mathcal{O}$면, 

$$b_0 Q + b_1 \phi(Q) = \mathcal{O}$$

$$\phi^2 (Q) - t \phi(Q) + qQ = \mathcal{O}$$

가 성립하므로 

$$(b_0^2 + b_0 b_1 t + b_1^2 q) Q = b_1^2 (\phi^2(Q) - t \phi(Q) + qQ) = \mathcal{O}$$

가 됩니다. 그러니 GCD 조건에서 $Q$의 order는 $r$. 

이를 가지고 $c_i$들의 값을 적당히 취하고 GCD 조건을 확인하면, BN254 타원곡선에서는 

$$(x+1)Q + \phi(xQ) + \phi^2(xQ) = \phi^3(2xQ)$$

를 확인하면 충분함을 알 수 있습니다. 실제로는 

$$x \not\equiv 4 \pmod{13}, \quad x \not\equiv 92 \pmod{97}$$

만을 확인하면 이를 사용할 수 있고, BN254는 이를 모두 만족합니다. 사실상 $xQ$만 계산하면 됩니다. 

# Conclusion 

지금까지 

- Ate Pairing, Optimal Ate Pairing
- BLS12-381, BN254의 parameter 및 기본 스펙 
- BLS12-381, BN254에 Hash-to-Curve
- BLS12-381, BN254에 Cofactor Clearing
- BLS12-381, BN254에 Subgroup Check

에 대해서 모두 알아봤습니다. 이 타원곡선에 대한 좋은 자료가 되었으면 좋겠습니다. 

[BLS12-381 참고자료](https://hackmd.io/@benjaminion/bls12-381#BLS12-381-For-The-Rest-Of-Us), [BN254 참고자료](https://hackmd.io/@jpw/bn254#BN254-For-The-Rest-Of-Us)도 참고해주세요. 감사합니다.