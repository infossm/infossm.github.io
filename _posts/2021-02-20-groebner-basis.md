---
layout: post
title:  "Gröbner basis"
date:   2021-02-20 16:00
author: RBTree
tags: [cryptography]
---

# 서론

최근 다변수 Coppersmith Method와 관련해 살펴보면서 Gröbner basis, 한국어로는 그뢰브너 기저에 대해서 접하게 되었습니다.

다변수 Coppersmith Method의 과정을 간략하게 설명하자면 다음과 같습니다.

1. 변수 $x_1, x_2, \dots, x_k$에 대한 어떤 modular equation $f(x_1, x_2, \dots, x_k) \equiv 0 \pmod n$을 풀고 싶다. 단, $x_1, x_2, \dots, x_k$는 $n$에 비해서 매우 작다.
2. Howgrave-Graham Theorem과 LLL algorithm을 적용해 $g_1(x_1, x_2, \dots, x_k) = 0, g_2(x_1, x_2, \dots, x_k) = 0, \cdots, g_l(x_1, x_2, \dots, x_k) = 0$ 이라는 $l$ 개의 equation으로 변환한다. (Modular equation에서 일반적인 equation들로 변환!)
3. Modular equation을 푸는 것보다 그냥 equation 여러 개를 푸는 것이 더 쉬우므로, 하여튼 더 쉬운 문제로 변환하는데 성공했다. 푸는 것은 여러분에게 맡긴다.

문제는 3번입니다. 다변수 equation 여러 개를 얻은 뒤에 어떻게 하면 풀 수 있는 지에 대해서는 보통 두 가지 방법을 논문들에서 언급합니다.

- Resultant
- Gröbner basis

이 때 그뢰브너 기저를 정말 많은 논문에서 언급하고 있는데, 아직도 전혀 그뢰브너 기저에 대해서 이해하지 못했기 때문에 이번 기회에 한 번 공부해보고 도대체 어떻게 풀 수 있다는 것인지 알아보려고 합니다.

# 본론

## Ideal and basis

어떤 field $K$에 대해서, $K[X] = K[x_1, x_2, \dots, x_n]$은 $x_1, x_2, \dots, x_n$으로 구성되며 모든 계수가 $K$에 속하는 다항식들의 집합을 나타냅니다. 예를 들어서, $3x^3y^2 + 7x^2y^3 -4xy +8y^3 -17 \in \mathbb{Q}[x, y]$ 라고 할 수 있겠습니다.

이 때, 어떤 집합 $I \subseteq K[X]$을 다음과 같은 조건을 만족하면 ideal이라고 부릅니다.

1. $u, v \in I \implies u+v \in I$
2. $u \in I, v \in K[X] \implies uv \in I$

이 때 $I$가 어떤 $p_1, \dots, p_k$를 포함하는 가장 작은 ideal일 때, $I =\langle p_1, \dots, p_k \rangle$으로 표기하며, $\{p-1, \dots, p_k\}$를 $I$의 basis라고 합니다. 이 때 한 ideal에 대해서는 여러 개의 basis가 존재할 수 있습니다.

## Gröbner basis

그렇다면 Gröbner basis는 어떤 basis를 말하는 것일까요?

이를 정의하기 위해서는 monomial들의 순서를 정할 필요가 있습니다.

예를 들어서, 우리는 보통 식을 쓸 때 monomial이 큰 것부터 작은 것으로 순서대로 쓰곤 합니다. $x^3+3x^2+4x+1$ 과 같은 식을 보면, $x^3$부터 상수항까지 차례대로 적어나가죠.

마찬가지로 다변수인 경우에서도 순서를 정해서 적어내려가곤 할 것입니다. $3x^3y^2 + 7x^2y^3 - 4xy + 8y^3 - 17$ 과 같은 식을 보면, $x$에 대해서 내림차순으로 정렬하고 그 뒤 $y$에 대해서 내림차순으로 정렬한 것이란 것을 관측할 수 있죠.

이렇게 monomial의 순서를 정할 때, divisibility를 만족하면서 정한 순서를 term order라고 합니다. 여기서 divisibility는, 어떤 두 monomial이 있을 때 어느 한 쪽이 다른 한 쪽을 나눌 수 있다고 하면 나눌 수 있는 쪽이 더 뒤에 있어야 한다는 것입니다. $x^2y$와 $xy$는 $xy$가 $x^2y$를 나눌 수 있으므로 $xy$가 더 뒤에 위치해야 하고, $x^2y$와 $xy^2$의 경우 둘 다 서로를 나누지 못하기 때문에 서로 어느 순서로 놓여도 상관 없습니다.

이렇게 term order를 정하게 되면, 어떤 polynomial이든 term order에 따라서 가장 앞에 놓이게 되는 monomial이 있을 것입니다. 이를 head 혹은 leading term이라고 부릅니다. 이 때 Gröbner basis는, ideal의 basis들 중에서도 leading term들이 가능한 한 작은 (term order에 따라서) basis입니다. 하지만 이렇게만 말하면 엄밀하지 못합니다. 이를 좀 더 엄밀하게 표현해봅시다.

Ideal $I$의 어떤 basis가 head $h$를 가지는 polynomial을 포함하고 있다고 합시다. 그러면 $h$에 어떤 monomial을 곱해서 나오는 monomial들은 모두 $I$의 어떤 element의 head라고 할 수 있을 것입니다. 예를 들어, $K[x, y]$의 어떤 ideal의 basis가 두 개의 polynomial로 구성되고, 각 polynomial의 head를 $xy^3$과 $x^4y$라고 합시다. 그러면 이 두 head에 의해서 나타내지는 head들을 다음과 같이 좌표상에 그려볼 수 있을 것입니다.

![](/assets/images/rbtree/groebner_1.png)

이 때, 이 좌표상에 표현된 영역이 과연 ideal의 모든 element의 head를 나타낸다고 할 수 있을까요? 답은 '아니오'입니다.

![](/assets/images/rbtree/groebner_2.png)

경우에 따라서, 어떤 element는 $xy^3$이나 $x^4y$의 multiple로 표현되지 않는 head를 가지고 있을 수 있습니다.

Gröbner basis는 **이러한 경우가 없는** basis를 의미합니다. 즉, 모든 ideal의 element의 head는 Gröbner basis의 한 polynomial의 head의 multiple입니다. 이를 다시 표현하면 다음과 같습니다.

$\{g_1, \dots, g_k\}$가 Gröbner basis $\iff \forall p \in \langle g_1, \dots, g_k \rangle \setminus \{0\}\ \exist i \in \{1, \dots, k\}: \text{Head}(g_i) \vert \text{Head}(p)$

이 때 Gröbner basis에 대해서 다음과 같은 사실들이 있습니다.

- 모든 ideal $I \subseteq K[X]$는 유한한 Gröbner basis를 가진다.
- Gröbner basis는 unique하다.
- 어떤 임의의 basis가 주어졌을 때 이로부터 Gröbner basis를 계산할 수 있다.
- Gröbner basis를 계산하는 것은 hard problem이다.

## Gröbner basis를 통해 solution 찾기

이제 Gröbner basis를 어떻게 응용하면 solution을 찾을 수 있는지 sage를 사용한 예시를 통해 차근차근 살펴봅시다.

우선 $f_1 = xy - 2y, f_2 = 2y^2 - x^2$를 sage를 통해 풀어봅시다.

```
sage: R.<x, y> = PolynomialRing(QQ, order='lex')
sage: I = ideal(x*y - 2*y, 2*y^2 - x^2)
sage: B = I.groebner_basis()
sage: B
[y^3 - 2*y, x^2 - 2*y^2, x*y - 2*y]
```

Leading term이 작은 polynomial을 찾은 결과 $y^3 - 2y$라는 식이 나왔습니다. 이를 sage를 통해 푸는 것 또한 가능합니다.

```
sage: t = var('t')
sage: solve(t^3 - 2*t == 0, t)
[t == -sqrt(2), t == sqrt(2), t == 0]
```

이제 이렇게 구한 $\pm \sqrt{2}, 0$을 대입해보면서 $x$를 구하면 문제를 풀 수 있을 것입니다.

더 복잡한 예시로 가봅시다. $f_1 = xy - 2yz - z, f_2 = y^2 - x^2z + xz, f_3 = z^2 - y^2x + x$를 풀어봅시다.

```
sage: R.<x, y, z> = PolynomialRing(QQ, order='lex')
sage: I = ideal(x*y - 2*y*z - z, y^2 - x^2*z + x*z, z^2 - y^2*x + x)
sage: B = I.groebner_basis()
sage: B
[x + 3442816/43083*z^11 - 3106688/43083*z^10 + 2269472/43083*z^9 - 672120/4787*z^8 + 354746/43083*z^7 - 2475608/43083*z^6 + 1202033/43083*z^5 + 409939/43083*z^4 + 402334/43083*z^3 + 69484/43083*z^2 - 118717/43083*z, y^2 - 1686848/43083*z^11 + 2672704/43083*z^10 - 2592304/43083*z^9 + 1537316/14361*z^8 - 3008257/43083*z^7 + 2466820/43083*z^6 - 2288350/43083*z^5 + 663460/43083*z^4 - 583085/43083*z^3 + 267025/43083*z^2 - 11821/43083*z, y*z + 163456/4787*z^11 - 150784/14361*z^10 - 46432/4787*z^9 - 520424/14361*z^8 - 472018/14361*z^7 - 90346/14361*z^6 + 3695/4787*z^5 + 226657/14361*z^4 + 26380/14361*z^3 + 16681/14361*z^2 - 22001/14361*z, z^12 - z^11 + 3/4*z^10 - 31/16*z^9 + 29/64*z^8 - 15/16*z^7 + 45/64*z^6 - 3/64*z^5 + 17/64*z^4 - 1/16*z^3 - 1/64*z]
```

살펴보면 $z^{12} - z^{11} + 3/4z^{10} - 31/16z^9 + 29/64z^8 - 15/16z^7 + 45/64z^6 - 3/64z^5 + 17/64z^4 - 1/16z^3 - 1/64z$ 라는 일차식을 가지고 있는 것을 확인할 수 있습니다. 이 식의 경우 풀 수 없기 때문에 근사를 통해서 풀어야 하는데, 이 블로그 포스트의 범위를 벗어나기 때문에 생략하도록 하겠습니다.

## 다변수 Coppersmith에 Gröbner basis 사용해보기

제가 현재 3달째 방치해둔 https://github.com/jhs7jhs/lll라는 repository가 있습니다. 이 중 SSS method가 이변수 다항식을 사용하는데, 현재 resultant를 통해서 해를 구하고 있습니다. 이것을 Gröbner basis를 사용하게끔 바꿀 수 있지 않을까? 하고 생각해서, 뒷부분을 한 번 수정해보기로 했습니다.

```python
    for pol1_idx in range(nn - 1):
        for pol2_idx in range(pol1_idx + 1, nn):
            # for i and j, create the two polynomials
            PR.<a, b> = PolynomialRing(ZZ)
            pol1 = pol2 = 0
            for jj in range(nn):
                pol1 += monomials[jj](a,b) * BB[pol1_idx, jj] / monomials[jj](X, Y)
                pol2 += monomials[jj](a,b) * BB[pol2_idx, jj] / monomials[jj](X, Y)

            # resultant
            PR.<q> = PolynomialRing(ZZ)
            rr = pol1.resultant(pol2)

            # are these good polynomials?
            if rr.is_zero() or rr.monomials() == [1]:
                continue
            else:
                print("found them, using vectors", pol1_idx, "and", pol2_idx)
                found_polynomials = True
                break
        if found_polynomials:
            break

    if not found_polynomials:
        print("no independant vectors could be found. This should very rarely happen...")
        return 0, 0
    
    rr = rr(q, q)

    # solutions
    soly = rr.roots()

    if len(soly) == 0:
        print("Your prediction (delta) is too small")
        return 0, 0

    soly = soly[0][0]
    ss = pol1(q, soly)
    solx = ss.roots()[0][0]

    return solx, soly
```

이것이 현재 Resultant를 통해서 해를 구하는 코드입니다. 앞서 서론에서 설명한 것과 같이 $g$들이 $g_1$부터 $g_n$까지 나오게 되는데, 이 중 2개를 골라 resultant를 통해서 해를 구하는 과정입니다.

여기서 대신 Gröbner basis를 사용하는 코드를 작성해보기로 했습니다.

```python
    pols = []
    PR.<a, b> = PolynomialRing(ZZ, order='lex')
    for pol_idx in range(nn // 2):
        pol = 0
        for jj in range(nn):
            pol += monomials[jj](a,b) * BB[pol_idx, jj] / monomials[jj](X, Y)
        pols.append(pol)

    I = ideal(pols)
    B = I.groebner_basis()

    print(B)

    T.<t> = PolynomialRing(ZZ)
    solx = B[0](t, 0).roots()[0][0]
    soly = B[-1](0, t).roots()[0][0]

    return solx, soly
```

일단 코드를 다음과 같이 작성하니 정상적으로 해를 구하는 것을 확인할 수 있었지만, 여러가지 의문점을 남기고 있습니다.

- `for pol_idx in range(nn // 2)`의 경우, 어림짐작으로 조정한 범위입니다. `for pol_idx in range(nn)`을 사용할 경우 Gröbner basis가 `[1]`이 나와서, 전혀 해를 구할 수 없었습니다.
  이 경우 한 가지 추측을 해볼 수 있습니다. $g_1$부터 $g_n$까지의 모든 식이 항상 valid하지는 않습니다. 일반적인 경우 $g_1$부터 $g_n$ 중 어떤 $k$에 대해 $g_k$부터 $g_n$까지가 valid하지 않을 수가 있는데, 이러한 경우 valid하지 않은 식을 basis에 포함시킴으로써 정상적인 Gröbner basis를 계산하지 못한다고 가정해볼 수 있습니다.
  이를 피하기 위해서는 $g_1$부터 $g_n$까지를 모두 포함시켜 Gröbner basis를 계산해본 후, 정상적이지 않을 경우 맨 뒤 $g$부터 하나씩 제외시키면서 다시 Gröbner basis를 계산해보는 것을 생각해볼 수 있겠습니다.
- 우연찮게도 구한 Gröbner basis의 맨 첫 번째 식이 $a$로만 구성되어있고, 마지막 식이 $b$로만 구성되어 있어 맨 아래 세 줄과 같이 `roots()` 함수를 통해 손쉽게 해를 구할 수 있었습니다. 하지만 모든 다변수 Coppersmith method에서 다음과 같이 나오지는 않을 것입니다. 그러므로 일반화된 solver를 작성할 방법을 찾아볼 필요성이 있겠습니다.

이 중 첫 번째 부분을 다시 반영해 다음과 같이 코드를 작성해보았습니다.

```python
    pols = []
    PR.<a, b> = PolynomialRing(ZZ, order='lex')
    for pol_idx in range(nn):
        pol = 0
        for jj in range(nn):
            pol += monomials[jj](a,b) * BB[pol_idx, jj] / monomials[jj](X, Y)
        pols.append(pol)

    st, ed = 1, nn
    while st < ed:
        print(st, ed)
        md = (st + ed + 1) // 2
        I = ideal(pols[:md])
        B = I.groebner_basis()

        if len(B) == 1:
            ed = md - 1
        else:
            st = md

    I = ideal(pols[:st])
    B = I.groebner_basis()

    print(B)

    T.<t> = PolynomialRing(ZZ)
    solx = B[0](t, 0).roots()[0][0]
    soly = B[-1](0, t).roots()[0][0]

    return solx, soly
```

결과적으로 Resultant를 사용하는 방법과 동일하게 해를 잘 구하는 것을 확인할 수 있었습니다. 하지만 Resultant를 사용하는 방법에 비해서 다소 시간이 걸리는 문제가 있습니다. 경우에 따라서 Resultant가 해를 구하지 못하는 경우가 있기 때문에, 시간이 걸리더라도 더 엄밀하게 구할 수 있지 않을까? 하는 생각이 있습니다.

# 결론

이번 기회에 Gröbner basis를 어떻게 하면 다변수의 equation을 푸는데 사용할 수 있는지 확인해볼 수 있었습니다. 실제로 사용해보니 강력한 method인 것은 확실하지만, 여전히 의문점이 여러 개 남습니다.

- 만약 Gröbner basis를 통해서 다변수 equation들로부터 일변수 equation을 이끌어내지 못한다면, 현실적으로 풀지 못하는 것일까?
- 어떤 조건이면 항상 Gröbner basis를 통해서 일변수 equation을 도출해낼 수 있을까?
- 어떤 조건이면 Resultant로는 해를 구할 수 없지만 Gröbner basis를 통해서는 해를 구해낼 수 있을까?

이에 대한 궁금증을 해소하기 위해서는 좀 더 많은 공부가 필요할 것으로 생각됩니다. 우선 다변수 Coppersmith method에서 어떻게 $g$들이 유도가 되고, 이 $g$들이 어떤 특수한 관계성을 가지기에 Gröbner basis를 사용할 수 있는지 엄밀하게 확인할 수 있어야하는데, 어느 부분이든 쉽지 않기 때문에 이 부분은 훗날의 과제로 남겨두도록 하겠습니다.

# 참고 문헌

1. Gröbner bases and applications (by Manuel Kauers) http://www.algebra.uni-linz.ac.at/people/mkauers/publications/kauers18h.pdf
2. Gröbner bases and applications (by Christopher Hillar) https://www.msri.org/people/members/chillar/files/gbapplfinal.pdf
3. Gröbner Bases: A Short Introduction for Systems Theorists http://people.reed.edu/~davidp/pcmi/buchberger.pdf