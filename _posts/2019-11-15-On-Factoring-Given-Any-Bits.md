---
layout: post
title:  "On Factoring Given Any Bits"
date:   2019-11-15 23:50
author: RBTree
tags: [cryptography, number-theory]
---

# 서론

이번에 [Belluminar](https://www.facebook.com/Belluminar/) 라고 하는 대회에 참가했습니다. 해당 대회는 각 팀마다 문제를 두 개 씩 출제하고, 대회 시간 동안 문제를 풀면서 점수를 겨루는 방식으로 구성되어 있습니다. 저는 이번에 공부했던 테크닉을 문제로 내기로 결심해서 작성을 했는데, 생각보다 만족스러운 퀄리티로 문제가 나오지 않아 아쉬웠습니다.

본래라면 공부한 내용을 포스트로 바로 쓰겠지만, 제 이름이나 아이디를 검색해 그대로 솔버에 사용하는 것을 원치 않았기 때문에 이러한 소셜 해킹을 방지하기 위해서 이제서야 포스팅을 하게 되었습니다.

이번 포스트는 제가 냈던 문제의 근간이 되는 Mathias Herrmann과 Alexander May의 "Solving Linear Equations Modulo Divisors: On Factoring Given Any Bits" 라는 논문에 대해서 간략하게 다룹니다.

# Factorization with random bits

또 RSA 이야기를 하게 되었습니다. RSA에 대한 설명은 [제가 과거에 올렸던 포스트](http://www.secmem.org/blog/2019/10/20/Smooth-number-and-Factorization/)를 참고해주세요.

RSA가 성립하는 중요한 요인 두 개 중 하나는 $N$이 소인수분해하기 어렵다는 것이고, 이를 공격하기 위한 많은 방법이 나와있습니다. 그 중 Coppersmith method라고 하는 공격 방법이 있는데, 이에 기반을 두고 있는 것이 위에서 언급한 논문입니다.

논문의 내용에 따르면, $N$을 이루는 두 소인수 중 하나인 $n$-bit $p$에 대해서, $p$의 임의의 위치의 비트가 최소 $n \ln 2$개 유출된다면, $p$를 복구하고 곧 $N$을 소인수분해 할 수 있습니다. 어떻게 이런 마법과 같은 방법이 가능한 것일까요? 이를 알기 위해서는 Coppersmith method의 기초가 되는 Howgrave-Graham Theorem와 LLL Algorithm에 대해서 이해해야 합니다.

## Solving Modular Equations

잠시 RSA와 소인수분해에 대해서는 제쳐두고, 다음 equation을 살펴봅시다.

$x^3 \equiv a (\mathbb{mod}\ N)$

해당 식을 푸는 것은 많이 어렵습니다. $x$가 충분히 커지면 $a$를 예측할 수 없게 되고, 결국 어떤 수학적 정리에 의존해서 풀 수 밖에 없겠죠. 그런데, 만약 다음과 같은 제한 조건이 있다고 합시다.

$x < N^{1/3}$

이렇게 된다면, $x^3$은 무조건 $N$보다 작을 것이고, 곧 정수 집합에서 $a^{1/3}$을 구하면 그것이 $x$일 것입니다. 즉, $x^3 \equiv a (\mathbb{mod}\ N)$을 $x^3 = a$라는 식으로 치환해서 풀 수 있다는 것이죠. Modular equation의 근을 구하는 것은 어려운 일이지만, 그냥 equation의 근을 구하는 것은 상대적으로 쉬운 일입니다.

다시 소인수분해 이야기로 돌아와봅시다. 우리는 $p$의 임의의 위치의 비트를 알고 있으면 소인수분해 할 수 있다고 합니다. 이를 단순화시켜서, $n$-bit 소인수 $p$의 중간 비트 몇개를 알고 있고, most significant한 bit $k_1$개와 least significant한 bit $k_2$개를 모른다고 합시다. 우리가 알고 있는 $p$의 정보를 $a$라고 합시다. Most significant한 청크의 값을 $x_1$, least significant한 청크의 값을 $x_2$라고 한다면 다음과 같이 식을 쓸 수 있습니다.

$2^{n-k_1} x_1 + 2^{k_2} a + x_2 = p$

우리가 $x_1, x_2$ 값을 알아낸다면 $p$를 복구할 수 있고, 곧 $N$을 소인수분해 할 수 있을 것입니다. 그런데, 이 식을 다음과 같이 써봅시다.

$2^{n-k_1} x_1 + 2^{k_2} a + x_2 \equiv 0 (\mathbb{mod}\ p)$

만약 $2^{n-k_1}$이나 $2^{k_2}$ 값이 충분히 작다면, 혹은 $x_1$이나 $x_2$가 어떤 값보다 작다는 제한 조건이 걸린다면, 앞서 얘기했던 것처럼 $2^{n-k_1} x_1 + 2^{k_2} a + x_2 \equiv 0 (\mathbb{mod}\ p)$ 를 풀지 않고 $2^{n-k_1} x_1 + 2^{k_2} a + x_2 = 0$을 푸는 거로 바꿀 수 있지 않을까요? 이것이 Coppersmith method의 기초입니다.

## Howgrave-Graham Theorem

어떤 다항식 $f(x)$가 주어졌을 때, $f(x)$의 coefficient vector를 $f(x)$의 계수들을 모아서 vector로 만든 것이라고 합시다. 예를 들어, $x^2 + 3x + 2$는 $[1, 3, 2]^T$로 표현하는 식입니다. 그리고 이 vector의 euclidean norm을 $f(x)$의 norm이라고 하고, $\lvert \lvert f(x) \rvert \rvert$로 표기합시다. 예를 들어서, $\lvert \lvert x^2+3x+2 \rvert \rvert = \sqrt{1^2+3^2+2^2} = \sqrt{14}$라고 할 수 있습니다.

Howgrave-Graham Theorem은 $f(x) \equiv 0 (\mathbb{mod}\ p^m)$이 주어져 있을 때, 어떤 조건이 성립해야지 $f(x) \equiv 0 (\mathbb{mod}\ p^m)$의 근들을 $f(x) = 0$을 풂으로써 얻을 수 있는 지를 알려줍니다.

---

**Howgrave-Graham Theorem**

Let $g(x_1,\ldots,x_n) \in \mathbb{Z}[x_1,\ldots,x_n]$ be an integer polynomial with at most $\omega$ monomials. Suppose that

1. $g(y_1, \ldots, y_n) \equiv 0 (\mathbb{mod}\ p^m)$ for $\lvert y_1 \rvert \leq X_1,\ldots, \lvert y_n \rvert \leq X_n$
2. $\lvert \lvert g(x_1X_1, \ldots, x_nX_n) \rvert \rvert  < \frac{p^m}{\sqrt{\omega}}$

Then $g(y_1,\ldots, y_n) = 0$ holds over the integers.

---

이를 정리해보자면, 우리가 구하려고 하는 근 $y_1, y_2, \ldots, y_n$이 있고, 이 근들이 어떤 값보다 작은지 ($\lvert y_1 \rvert \leq X_1,\ldots, \lvert y_n \rvert \leq X_n$)를 알고 있으며, $g(x_1X_1, \ldots, x_nX_n)$의 norm이 충분히 작다면 $g(x_1,\ldots, x_n) = 0$을 풀어서 $y_1, y_2, \ldots, y_n$을 구할 수 있다는 것입니다.

하지만 이것으로는 뭔가 부족해보입니다. 우리가 구했던 식 $2^{n-k_1} x_1 + 2^{k_2} a + x_2 \equiv 0 (\mathbb{mod}\ p)$ 이 Howgrave-Graham Theorem을 만족하지 않으면 의미가 없기 때문입니다. 이를 해결하기 위해서 LLL Algorithm이 사용됩니다.

## Lattice and LLL Algorithm

Lattice라고 하는 개념이 있습니다. Lattice의 basis vector $v_1, \ldots, v_n$이 주어져 있을 때, 이 vector들로 표현되는 lattice의 집합은 다음과 같이 정의됩니다.

$L = \{v \in \mathbb{Z}^m \mid v = \sum^{n}_{i=1} a_iv_i\ \mathbb{with}\ a_i \in \mathbb{Z} \}$

즉, 주어진 vector들의 정수배의 합으로만 이루어진 것을 lattice라고 하는 것입니다. 격자라는 이름의 뜻에서 살펴볼 수 있듯이 매우 쉬운 개념입니다. Basis vector의 개수는 lattice의 rank이고, basis vector의 size가 $n$이고 basis vector가 총 $n$개일 때 full-rank lattice라고 합니다. Matrix와 개념이 똑같죠? 심지어 lattice의 determinant도 정의할 수 있는데, full-rank lattice의 경우 basis vector로 구성된 matrix의 determinant와 동일합니다.

LLL Algorithm은 input으로 basis vector $b_1, \ldots, b_n$이 주어져 있을 때, 똑같은 lattice를 구성하면서 더 사이즈가 작은 basis vector $v_1, \ldots, v_n$을 출력합니다. 즉, $b_1, \ldots, b_n$을 통해 구성되는 Lattice가 $v_1, \ldots, v_n$을 통해 구성되는 Lattice가 같다는 뜻입니다.

---

**Lenstra–Lenstra–Lovász Algorithm**

Let $L \in \mathbb{Z}^n$ be a lattice spanned by $B = \{b_1, \ldots, b_n\}$. The LLL-algorithm outputs a reducecd lattice basis $\{v_1, \ldots, v_n\}$ with

$\lvert \lvert v_i \rvert \rvert \leq 2^{\frac{n(n-1)}{4(n-i+1)}} \det(L)^{\frac{1}{n-i+1}}$

in time polynomial in $n$ and in the bit-size of the entries of the basis matrix $B$.

---

살펴보면 $v_1$이 가장 작은 상한을 가지며 그 값이 $2^{\frac{n-1}{4}} \det(L)^{\frac{1}{n}}$임을 알 수 있습니다. Coppersmith method에서는 사실 더 작은 basis vector를 구하는 것에 관심이 있는 것이 아니라 이런 작은 $v_1$을 구할 수 있다는 데에 초점을 두고 있습니다.

## Using LLL Algorithm with Equations

LLL Algorithm을 응용하기 위해서는 새로운 equation들을 구해야 합니다.

우리가 구하려던 식을 $f(x_1, x_2) = 2^{n-k_1} x_1 + 2^{k_2} a + x_2$ 라고 합시다. 그리고 어떤 정수 $m$에 대해서 다음과 같은 $g_{k, i}(x_1, x_2)$를 생각해봅시다.

$g_{k,i}(x_1, x_2) = x_2^i f^k(x_1,x_2) N^{\max\{t-k, 0\}}$

이 식의 특징은 우리가 구하고 싶은 근 $y_1, y_2$에 대해서 $g_{k, i}(y_1, y_2) \equiv 0 (\mathbb{mod}\ p^m)$이라는 것입니다. 여러 $k$와 $i$ 값에 대해서 다양한 식을 만들 수 있겠죠? 또 여기서 특징 하나는, $k$나 $i$가 하나씩 늘어날 때마다, 새로운 항이 하나씩만 생긴다는 점입니다.

여기서 하고 싶은 일은 바로 이 $g_{k,i}(x_1, x_2)$들의 정수배의 합으로 이루어진 어떤 식 $h(x_1, x_2)$를 구해서, 이 $h(x)$가 Howgrave-Graham Theorem을 만족하게 해서 $h(x_1, x_2) = 0$을 풀어서 근 $y_1, y_2$를 구하는 것입니다.

$h(x)$를 구하기 위해서 LLL Algorithm을 사용해봅시다. 이 $g_{k, i}(x_1, x_2)$들의 coefficient vector로 구성되는 Lattice를 생각해봅시다. 그러면, 앞서 말했다시피 $g_{k,i}(x_1, x_2)$는 $k, i$가 하나씩 늘어날 때마다 새로운 항이 **하나씩만** 추가됩니다. Coefficient vector들로 나타내지는 matrix를 그림으로 나타내면 다음과 같습니다.

<img src="/assets/images/rbtree/herrman_may_08_figure_1.png">

우리는 LLL Algorithm의 output $v_1, \ldots, v_n$ 중 $v_1$이 $2^{\frac{n-1}{4}} \det(L)^{\frac{1}{n}}$보다 작다는 것을 알고 있습니다. 그런데 위의 matrix는 매우 determinant를 구하기 쉽습니다. 바로 대각선 값들만 곱하면 되죠. 이를 통해 $2^{\frac{n-1}{4}} \det(L)^{\frac{1}{n}}$의 값을 구한 뒤, 이 값이 Howgrave-Graham Theorem의 2번 조건을 만족하는 지를 확인하면 됩니다.

논문에서는 이를 바탕으로 LLL Algorithm을 통해서 구한 $v_1$을 coefficient vector로 하는 식 $h(x_1, x_2)$를 복구한 뒤 $h(x_1, x_2) = 0$을 풀어서 근을 구할 수 있음을 보이며, 이 때의 조건이 $p$의 약 41.4%임을 보입니다. 그리고 더 나아가, $x_1, x_2$가 아닌 임의의 개수의 $x$에 대해서도 풀 수 있으며, $p$의 $\ln(2)$만큼의 정보가 있으면 항상 $h(x)$를 구할 수 있음을 증명합니다.

# 마무리

논문에 대한 PoC 코드는 [링크](https://gist.github.com/jhs7jhs/0c26e83bb37866f5c7c6b8918a854333) 에서 확인할 수 있습니다. 너무 코드가 길기 때문에 포스트에 포함시키는 것이 부적절할 것 같아서 따로 gist에 올렸으며, Sage를 통해서 작성되어 있습니다. Belluminar에서도 해당 코드가 공유되었습니다(만 다른 사람이 읽긴 했는지 잘 모르겠습니다).

혹시 잘 이해가 가지 않았는데 자세히 이해하고 싶으시다면, Reference의 2번에서 공유하고 있는 Alexander May의 글을 꼭 한 번 읽어보시는 것을 추천합니다. 또한 동아리 동기가 [작성한 글](https://eyebrowmoon.github.io/hacking/crypto/rsa/2019/05/23/RSA_Attack_Using_LLL.html?fbclid=IwAR3twlXvYhjEjV7hGigA9NcB3KnyEXQaYor4fvXZmkER9rz67-sQLh5mW0w)도 추천합니다.

$h(x)$를 구한 뒤 $h(x) = 0$을 구하는 방법에 대해서 궁금한 분들도 계실텐데, 논문에서는 [Gröbner basis reduction](https://en.wikipedia.org/wiki/Gr%C3%B6bner_basis)를 통해서 구하거나, 다변수 상황에서의 Newton's method를 쓰는 것을 권장합니다. 저는 [이 글](https://arxiv.org/pdf/1208.399.pdf)을 따라서 Sage에서 Newton's method를 구현했습니다.

## Reference

1. [Solving Linear Equations Modulo Divisors: On Factoring Given Any Bits](https://link.springer.com/chapter/10.1007/978-3-540-89255-7_25)
2. [New RSA Vulnerabilities Using Lattice Reduction Methods](https://www.math.uni-frankfurt.de/~dmst/teaching/WS2015/Vorlesung/Alex.May.pdf)