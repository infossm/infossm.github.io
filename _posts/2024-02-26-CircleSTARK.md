---
layout: post

title: "Circle STARK Part 1: Circle FFT"

date: 2024-02-26

author: rkm0959

tags: [cryptography, blockchain]
---

## ZKP를 위한 Underlying Field $\mathbb{F}_p$

ZKP의 패러다임을 다루거나, ZKP를 본격적으로 분류를 하게 되면 그에 대한 가능한 기준으로는 여러가지가 있습니다. 하지만 요즘 대부분이 생각하는 방식은 (수학적으로 엄밀하거나 완전하지는 않아도) $\mathbb{F}_p$ 위에서 ZKP가 작동을 할 때 $p$가 어떤 값을 가지냐를 가지고 많이 논의를 합니다. 

- $p$가 256 bit 정도를 가지는 경우
- $p$가 64 bit 정도거나 32 bit 정도로 작은 경우 
- $p$가 소수가 아니라 256 bit 정도의 2의 거듭제곱인 경우

여기에 추가적으로 $p$가 NTT를 지원한다던가, 타원곡선의 크기여야 한다던가 하는 등의 조건들이 더 붙기도 합니다. 작은 $p$의 경우, 대부분 FRI를 기반으로 한 STARK입니다. 엄밀하게 논문 상 정의로는 FRI를 기반으로 한다고 STARK는 아닙니다만 대부분 이렇게 부르고 있습니다.

FRI를 쓰는 $p$의 경우에서는 작은 $p$를 사용해도 되지만, 기본적으로는 FFT를 기반으로 하고 있기 때문에 $p - 1$이 2를 많이 가지고 있어야 합니다. ECFFT라는 기법을 통해서 (https://rkm0959.tistory.com/252) $p$에 대한 조건을 없앨 수 있으나, 일반적으로는 FFT를 사용했습니다. 대표적으로 사용되는 $p$는 Goldilocks $2^{64} - 2^{32} + 1$이나 BabyBear $2^{32} - 2^{27} + 1$ 정도였습니다. 

그런데 최근부터 더욱 빠른 $\mathbb{F}_p$ 위에서의 연산을 위해서, $p = 2^{31} - 1$을 사용하는 방법이 연구되기 시작했습니다. Monolith (https://infossm.github.io/blog/2023/07/14/Monolith/, https://eprint.iacr.org/2023/1025.pdf)와 같은 hash function을 위해서도 적합하며, 기본적으로 $\mathbb{F}_p$ 연산이 빠를 수 밖에 없는데다가 circle group을 사용하면, 그 group의 크기가 $p + 1 = 2^{31}$이 되어 여기서 FFT를 쓸 수 있음이 Polygon 팀에서 제안되어 (https://eprint.iacr.org/2023/824.pdf) 많은 관심을 받았습니다. 

최근 본격적으로 이 $p = 2^{31} - 1$과 circle group을 이용해서 STARK를 하는 방법이 논문으로 나왔고, 이 방식이 BabyBear prime을 사용하는 것보다 더 속도가 빨랐다는 결과가 나왔습니다. 이 글에서는 해당 논문인 https://eprint.iacr.org/2024/278.pdf 를 정리하도록 하겠습니다. 자세한 증명은 생략하고, 핵심 아이디어만 짚으면서 정리하도록 하겠습니다. 

## Circle Curve

$p + 1$이 2를 많이 가지고 있다고 합시다. 그러면 기본적으로 $p \equiv 3 \pmod{4}$. 

이제 curve $C(\mathbb{F}_p)$를 

$$x^2 + y^2 = 1$$

로 정의하면, 이 $C$와 projective line $P^1(\mathbb{F}_p)$ 사이에는 isomorphism이 있습니다. 

이는, 꽤 잘 알려져 있는 공식인 

$$t = \frac{y}{x + 1}, \quad (x, y) = \left( \frac{1 - t^2}{1 + t^2}, \frac{2t}{1 + t^2} \right)$$

을 사용하면 얻을 수 있습니다. 이를 기반으로, $C(\mathbb{F}_p)$의 크기가 $p + 1$임을 알 수 있습니다.

또한, 이미 잘 알려져 있는 공식인 

$$(x^2 + y^2)(z^2 + w^2) = (xz - yw)^2 + (xw + yz)^2$$

을 이용하면, group operation 

$$(x_0, y_0) \cdot (x_1, y_1) = (x_0 x_1 - y_0 y_1, x_0 y_1 + y_0 x_1)$$

을 생각할 수 있습니다. 이러면 identity는 $(1, 0)$. 또한, 

$$T_P(x, y) = P \cdot (x, y) = (P_x x - P_y y, P_x y + P_y x)$$

$$\pi (x, y) = (x^2 - y^2, 2xy)$$

$$J(x, y) = -(x, y) = (x, -y)$$

를 정의합시다. 또한, $(x, y) \rightarrow x + yi$를 생각하면 $\mathbb{F}_p(i)$의 multiplicative subgroup으로 가는 group homomorphism 이므로 $C$는 cyclic group입니다. 그러므로, 각 $N = 2^n$에 대해서 $C$의 size $N$ subgroup을 유일하게 잡을 수 있고 이를 $G_n$이라고 부를 수 있습니다. 

이제 coset에 대한 이야기를 할 수 있습니다. 점 $Q$에 대해서, $D = Q \cdot G_{n-1} \cup Q^{-1} G_{n-1}$이 disjoint union이면 이를 twin-coset of size $2^n$이라고 부릅니다. 만약 $D$가 $G_n$의 coset이 된다면, 이를 standard position coset이라고 부릅니다. 

standard position coset이 되려면 $Q$의 order가 정확히 $2^{n+1}$이면 됩니다. 이 경우, 

$$D = Q \cdot G_n = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$$

이 됨을 보일 수 있습니다. 또한, 각 $D \subset C(\mathbb{F}_p) \setminus G_m$이 $G_{m-1}$와 $J$에 대해서 invariant이라면, 임의의 $n \le m$에 대하여 $N = 2^n$ 크기의 twin-coset으로 decompose 될 수 있음을 증명할 수 있습니다. 특히, $D$가 standard position coset of size $M = 2^m$이면, 

$$D = Q \cdot G_m = \cup_{k=0}^{M/N-1} \left( Q^{4k+1} \cdot G_{n-1} \cup Q^{-4k-1} \cdot G_{n-1} \right)$$

임을 보일 수 있습니다. 뭔가 많지만 사실 다 크기 $2^n$ cyclic group 이야기라 어렵지 않습니다. 

또한, $D$가 twin-coset of size $2^n$이면, $\pi(D)$ 역시 twin-coset임을 알 수 있으며, $D$가 standard position coset이면 $\pi(D)$ 역시 standard position coset임을 증명할 수 있습니다. 

## Space of Polynomials

Circle STARK에서 다루는 다항식은 

$$\mathcal{L}_N(\mathbb{F}) = \{ p(x, y) \in \mathbb{F}[x, y]/(x^2 + y^2 - 1) : \deg p \le N/2 \}$$

입니다. 여기서 $\deg p$는 $p(x, y) + (x^2 + y^2 - 1)$에서 나올 수 있는 최소한의 total degree를 의미합니다. 여기서 다음 사실을 보일 수 있습니다. 

- $\mathcal{L}_N(\mathbb{F})$는 invariant under rotation 
- $\mathcal{L}_N(\mathbb{F})$의 dimension은 $N + 1$이며, 이의 non-trivial $f$는 $C(\mathbb{F})$에서 최대 $N$개의 근을 가짐

전자는 rotation이 linear mapping 이므로 자명합니다. 후자의 증명은 어렵지 않은데, 결국 앞서 언급한 $P^1(\mathbb{F}_p)$와의 isomorphism을 생각하면 결국 $\mathcal{L}_N(\mathbb{F})$는 결국 

$$\mathcal{L}_N(\mathbb{F}) = \left\{ \frac{p(t)}{(1+t^2)^{N/2}} : \deg p(t) \le N \right\}$$

와 같음을 증명할 수 있기 때문입니다. 또한, 기본적으로 $p(x, y)$는 

$$p(x, y) = p_0(x) + y p_1(x)$$

로 쓸 수 있는데, 이는 $y^2 = 1 - x^2$을 사용하면 얻을 수 있습니다. 이를 기반으로, 

$$1, x, \cdots, x^{N/2}, y, yx, \cdots yx^{N/2-1}$$

을 $\mathcal{L}_N(\mathbb{F})$의 monomial basis라고 볼 수 있습니다. 

이제 이 다항식들의 space를 기반으로, Reed-Solomon code의 대응인 Circle Code를 생각합니다. 

$$\mathcal{C}_N(F, D) = \{f(P)\vert_{P \in D} : f \in \mathcal{L}_N(\mathbb{F}_p) \}$$

이 경우, 이 code의 minimum distance는 최소 $\lvert D \rvert - N$ 이상입니다. 

단, 여기서 $D$는 $C(\mathbb{F}_p)$의 proper subset이 되도록 합니다.

또한, 이 code와 기본적인 Reed-Solomon code 사이에는 쉽게 계산 가능한 isomorphism이 있습니다. $Q \cdot D$에 $(-1, 0)$이 없도록 하면, 앞서 나온 $C$와 $P^1(\mathbb{F}_p)$ 사이의 isomorphism을 사용할 수 있습니다. $\phi$를 $C^1(\mathbb{F}_p)$에서 $\mathbb{F}_p$로 보내는 map 이라고 하면 $\phi(Q \cdot D)$를 생각할 수 있습니다. 즉, $Q \cdot D$에 $(-1, 0)$이 없으니 $\phi$가 infinity point를 거치지 않고 바로 $\mathbb{F}_p$로 가게 되고, 여기서 $S = \phi(Q \cdot D)$를 생각할 수 있습니다. 그러면, $w$가 $\mathcal{C}_N$의 codeword라면, $S$의 원소 $t$를 $(\phi \cdot T_Q)^{-1}$를 적용하여 $D$로 보내고, 여기서 $w$를 적용하여 $f$를 연산합니다. 그 다음, $(1+t^2)^{N/2}$를 곱하면 $\mathcal{L}_N(\mathbb{F})$의 representation에 의해서 그 결과는 결국 $t$에 degree $N$ 이하 다항식을 연산한 결과와 같습니다. 결국, $\mathcal{C}_N$은 $S$를 evaluation domain으로 하는 Reed-Solomon code와 isomorphic 합니다. 

이제 Vanishing Polynomial에 대해서 생각합시다. $D$가 $C(\mathbb{F}_p)$의 subset으로, 그 크기 $N$이 짝수라고 합시다. 단, $2 \le N < p + 1$. 이때, $D$의 vanishing polynomial을 잡기 위해서, $D$를 pair $N/2$개 $\{P_k, Q_k\}$로 분할하고, vanishing polynomial을 

$$v_D(x, y) = \prod_{k=1}^{N/2} \left( (x - P_{k, x})(Q_{k, y} - P_{k, y}) - (y - P_{k, y})(Q_{k, x} - P_{k, x}) \right)$$

라고 하면 이는 $D$에서 전부 $0$이 됩니다. 또한, 이는 $\mathcal{L}_N(\mathbb{F})$에 속하는데, $N$개의 근을 가지므로, 결국 $v$는 $D$ 밖에서는 non-zero 값을 가지며 모든 $D$에서 vanishing 하는 $\mathcal{L}_N(\mathbb{F})$의 원소는 $v_D$의 배수임을 알 수 있습니다. 

가장 중요한 것은 standard position coset이나 twin-coset의 vanishing polynomial입니다. $D = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$이라 하면, $\pi^{n-1}$을 적용하면 결국 $\pi^{n-1}(D) = \{(x_D, \pm y_D)\}$임을 알 수 있습니다. 그러므로, 간단하게 생각하면 

$$v_D(x, y) = \pi_x \circ \pi^{n-1}(x, y) - x_D$$

임을 알 수 있습니다. 특히, standard position coset에서는 $x_D = 0$임을 알 수 있으며, $v_D$가 사실 $y$에 depend 하지 않음을 알 수 있습니다. 이를 기반으로, standard position coset에서는 $v_D$가 $G_n$의 group action에서 alternate 한다는 것을 증명할 수 있습니다. 즉, $P$가 $G_n$의 generator인 경우, $D = Q \cdot G_n$이 standard position coset이라면 $v_D \circ T_P = -v_D$가 성립함을 알 수 있습니다. $\pi^{n-1}$을 $G_n$에 적용하면 $(\pm 1, 0)$으로 가므로, 

$$v_D \circ T_P = \pi_x \circ \pi^{n-1} \circ T_P = \pi_x \circ T_{(-1, 0)} \circ \pi^{n-1} = - \pi_x \circ \pi^{n-1} = -v_D$$

를 얻습니다. 또한, 예상할 수 있듯이 $D$가 크기가 짝수 $N$인 집합이고 $M$이 짝수이며 $2 \le N \le M < p+1$이 성립한다 했을 때, $f \in \mathcal{L}_M(\mathbb{F})$가 $D$에서 vanish하면 $f = q \cdot v$로 표현할 수 있으며, 이때 $v = v_D$임을 보일 수 있습니다. 

또한, DEEP의 technique을 그대로 쓸 수 있는데, linear functional 

$$v_P(x, y) = 1 - (P_x \cdot x + P_y \cdot y) - (-P_y \cdot x + P_x \cdot y) i$$

를 생각하면, 기본적으로 $(f - f(P)) / v_P$가 $\mathcal{L}_N(\mathbb{F}(i))$에 있음을 증명할 수 있습니다. 

## Circle FFT 

$v_k(x)$를 $2^k$ 크기의 standard position coset에 대한 vanishing polynomial이라고 합시다. 목표는 $D = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$에 대한 FFT를 진행하는 것입니다. 단, $Q \notin G_n$. 

각 $0 \le j < 2^n$에 대해서 $(j_0, \cdots, j_{n-1})$이 그 bit representation이라고 하면, FFT-basis of order $n$을 

$$b_j^{(n)}(x, y) = y^{j_0} \cdot v_1(x)^{j_1} \cdots v_{n-1}^{j_{n-1}}(x)$$

라고 둘 수 있습니다. $v_k(x)$의 degree가 최대 $2^{k-1}$이므로, $b_j$가 전부 $\mathcal{L}_N(\mathbb{F})$의 원소임을 알 수 있습니다. 물론 $N = 2^n$.

이 basis $\mathcal{B}_n$이 있을 때, $\mathcal{B}_n$에 대한 coefficient가 주어졌을 때 이 다항식을 $D$에서 evaluate 하는 것이 $\mathcal{O}(N \cdot n)$에 가능하며, 반대로 $D$ 위에서의 evaluation이 주어졌을 때 $\mathcal{B}_n$에 대한 coefficient를 구하는 것도 같은 시간 안에 가능합니다. 이 과정이 곧 Circle FFT입니다. 

그런데, $\mathcal{B}_n$은 $N$개의 원소로 이루어져 있지만, 정작 $\mathcal{L}_N$의 dimension은 $N+1$입니다. 이 차이는 나중에 따로 handle 됩니다. 

기본적인 아이디어는 $D_n = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$이 있을 때, $D_j = \pi(D_{j+1})$과 함께 $S_j = \pi_x(D_j)$를 생각하는 것입니다. 이러면 $S_{j-1} = \pi (S_j)$는 $x \rightarrow 2x^2 - 1$을 하는 것과 동일합니다. 

Circle FFT는 기본적으로 $D = D_n$에서의 evaluation 문제를 

$$D_n \rightarrow S_n \rightarrow S_{n-1} \rightarrow \cdots \rightarrow S_1$$

으로 reduce 하는 방식을 이용합니다. 우선 $D_n \rightarrow S_n$으로 가기 위해, 

$$f_0(x) = \frac{f(x, y) + f(x, -y)}{2}, \quad f_1(x) = \frac{f(x, y) - f(x, -y)}{2y}$$

를 생각해서 $f(x, y) = f_0(x) + y f_1(x)$이도록 합니다. 이를 기반으로 $f_0, f_1$을 얻습니다. 

그 이후부터는, 비슷하게 

$$f_0(\pi(x)) = \frac{f(x) + f(-x)}{2}, \quad f_1(\pi(x)) = \frac{f(x) - f(-x)}{2x}$$

이도록 해서 

$$f(x) = f_0(\pi(x)) + x \cdot f_1(\pi(x))$$

이도록 합니다. 단 $\pi(x) = 2x^2 - 1$. 이를 재귀적으로 반복하면 결국 계수들이 $2^n$개가 나오고, 이들이 실제로 $\mathcal{B}_n$에 대응되는 계수들임을 증명할 수 있습니다. inverse Circle FFT도 비슷하게 생각해서 얻을 수 있습니다. 

이제 $\mathcal{L}_N'(\mathbb{F})$를 $\mathcal{B}_n$의 span이라고 하면, 그 dimension이 $N$임을 알 수 있습니다. $\mathcal{B}_n$의 monomial들을 보면, 결국 

$$\mathcal{L}_N'(\mathbb{F}) = \{ p_0(x) + y p_1(x) : \deg p_i < N/2 \}$$

임을 알 수 있습니다. 즉, $\mathcal{L}_N(\mathbb{F})$와의 차이점은 $x^{N/2}$를 허용하는지 여부입니다. 특히, $v_n$의 degree가 $N/2$이므로, 

$$\mathcal{L}_N(\mathbb{F}) = \mathcal{L}_N'(\mathbb{F}) + \langle v_n \rangle$$

임을 얻습니다. 

다음 글부터 FFT space의 본격적인 성질과, 이를 STARK에 활용하는 방법에 대해서 알아보겠습니다.
