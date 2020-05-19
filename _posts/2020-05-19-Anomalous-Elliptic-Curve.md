---
layout: post
title:  "Anomalous Elliptic Curves"
date:   2020-05-19 23:55
author: RBTree
tags: [cryptography, elliptic curve, ecc]
---

# 서론

저번 글에 이어서 이번에는 데프콘 CTF의 예선을 진행하면서 anomalous elliptic curve에 대해서 공부하게 되었습니다. 본래 CryptoHack[1] 에서 anomalous elliptic curve와 그 취약점에 공부한 바가 있지만, 예선에 나온 문제는 CryptoHack에서 공부했던 공격에 **취약하지 않은** anomalous elliptic curve를 구하는 문제였습니다.

어떤 문제가 나왔었는지 하나씩 살펴봅시다.

# 본론

## Anomalous Elliptic Curve

Anomalous elliptic curve는 곡선이 $F_p$ 위에서 정의될 때 order가 $p$ 인 곡선을 의미합니다. 타원 곡선을 정의를 하게 되면 그 곡선 상에 존재할 수 있는 점의 개수가 정해지는데, 이를 order라고 한다 이해하면 됩니다.

이런 특수한 성질은 매우 중요한 특징을 가집니다. 만약 곡선의 order가 합성수 $n$이고 소인수분해가 가능하고 그 결과가 $n = p_1p_2 \ldots p_k$ 라고 합시다. 그러면 전체 점의 group을 각 소인수 $p_i$에 대해 $p_i$ 개의 점들로 이루어진 subgroup들로 쪼갤 수 있게 됩니다. 이 subgroup의 크기는 $n$에 비해 작기 때문에 공격이 더 쉽게 이뤄지고, 각 subgroup에 대한 결과는 중국인의 나머지 정리처럼 하나로 합칠 수 있습니다. 그렇기에, 보통 소수 $p$의 $F_p$ 에 대해서 정의한 곡선이 order $p$ 를 가진다면 똑같이 소수이니 공격에 대해서 상대적으로 덜 취약하다고 볼 수 있습니다.

하지만 order가 $p$이게 되면 매우 취약한 부분이 생기게 됩니다.

## Smart's Attack

Smart의 공격은 anomalous elliptic curve 상의 점 $P, Q$에 대해서 $P = kQ$인 $k$를 쉽게 구하는 공격입니다. 이는 다음과 같은 성질에 기반을 합니다.

1. $F_p$ 상의 점들을 p진수에 대한 집합 $Q_p$로 옮길 수가 있더라!
2. 그런데 $Q_p$ 위로 옮기면 $E(Q_p)$ 상의 점 $X$를 $pZ_p$에 매칭되는 어떤 값 $x$로 homomorphic하게 매핑할 수가 있더라!

1번의 경우를 생각해봅시다. $f(x) = 0 (\text{mod}\ p)$가 있을 때, 분명 $f(x') = 0 (\text{mod}\ p^2)$ 이면서 $x = x' (\text{mod}\ p)$ 인 $x'$ 가 있을 수 있지 않을까 생각해볼 수 있습니다. 더 나아가서, $f(x') = 0(\text{mod}\ p^k)$ 에 대해서도 할 수 있을지 모릅니다. 이러한 경우에 대해서 Hensel이 다음을 증명했습니다 (Hensel's lemma [2]):

$f(x) = 0 (\text{mod}\ p^k)$ 이고 $f'(x) = 0(\text{mod}\ p)$ 일 때 $f(x') = 0 (\text{mod}\ p^{k+m})$이고 $x' = x (\text{mod}\ p^k)$ 인 $x'$가 존재한다.

또한 $x'$를 구하는 것은 Hensel's lifting으로, 알고리즘이 존재합니다.

2번의 경우 이 블로그에서 온전히 설명하는 것이 어려우므로 [3]의 논문을 참고해주세요. 간략하게 설명하자면,  $E(Q_p)$와 $pZ_p$ 사이에 다음과 같은 매칭이 성립합니다:

$\phi_p(X) = - \frac{x(X)}{y(X)}$

### Attack code

공격 방법에 대해서는 찾아보면 많이 나오지만, [4]의 코드가 제일 명확하므로 이를 참고해 sage 코드를 작성해봅시다.

```python
def HenselLift(P,p,prec):
    E = P.curve()
    Eq = E.change_ring(QQ)
    Ep = Eq.change_ring(Qp(p,prec))
    x_P, y_P = P.xy()
    x_lift = ZZ(x_P)
    y_lift = ZZ(y_P)
    x, y, a1, a2, a3, a4, a6 = var('x,y,a1,a2,a3,a4,a6')
    f(a1, a2, a3, a4, a6, x, y) = y^2 + a1*x*y + a3*y - x^3 - a2*x^2 - a4*x - a6
    g(y) = f(ZZ(Eq.a1()), ZZ(Eq.a2()), ZZ(Eq.a3()), ZZ(Eq.a4()), ZZ(Eq.a6()), ZZ(x_P), y)
    gDiff = g.diff()
    for i in range(1, prec):
        uInv = ZZ(gDiff(y=y_lift))
        u = uInv.inverse_mod(p^i)
        y_lift = y_lift - u * g(y_lift)
        y_lift = ZZ(Mod(y_lift,p^(i+1)))
    y_lift = y_lift + O(p^prec)
    return Ep([x_lift, y_lift])

def SmartAttack(P, Q, p, prec):
    E = P.curve()
    Eqq = E.change_ring(QQ)
    Eqp = Eqq.change_ring(Qp(p, prec))

    P_Qp = HenselLift(P, p, prec)    
    Q_Qp = HenselLift(Q, p, prec)    

    p_times_P = p * P_Qp
    p_times_Q = p * Q_Qp
    x_P,y_P = p_times_P.xy()
    x_Q,y_Q = p_times_Q.xy()

    phi_P = -(x_P / y_P)
    phi_Q = -(x_Q / y_Q)    
    k = phi_Q / phi_P
    k = Mod(k, p)
    return k
```

사용할 때는 다음과 같이 사용하면 됩니다.

```python
E = EllipticCurve(GF(43), [0, -4, 0, -128, -432])
print(E.order())
P = E([0, 16])
Q = 39 * P
print(SmartAttack(P, Q, 43, 8))
```

## Avoiding Smart's Attack?

이렇게 보면 하여튼 anomalous elliptic curve는 매우 위험한 것으로 보입니다. 데프콘 CTF에서는 다음과 같은 문제가 나왔습니다. [5]

```python
#!/usr/bin/env sage
from sage.all import *
from threshold import set_threshold
import random

FLAG = open("/flag", "r").read()


def launch_attack(P, Q, p):
    E = P.curve()
    Eqp = EllipticCurve(Qp(p, 8), [ZZ(t) for t in E.a_invariants()])

    P_Qps = Eqp.lift_x(ZZ(P.xy()[0]), all=True)
    for P_Qp in P_Qps:
        if GF(p)(P_Qp.xy()[1]) == P.xy()[1]:
            break

    Q_Qps = Eqp.lift_x(ZZ(Q.xy()[0]), all=True)
    for Q_Qp in Q_Qps:
        if GF(p)(Q_Qp.xy()[1]) == Q.xy()[1]:
            break

    p_times_P = p * P_Qp
    p_times_Q = p * Q_Qp

    x_P, y_P = p_times_P.xy()
    x_Q, y_Q = p_times_Q.xy()

    phi_P = -(x_P / y_P)
    phi_Q = -(x_Q / y_Q)
    k = phi_Q / phi_P

    return ZZ(k) % p


def attack(E, P, Q):
    private_key = launch_attack(P, Q, E.order())
    return private_key * P == Q


def input_int(msg):
    s = input(msg)
    return int(s)


def curve_agreement(threshold):
    print("Give me the coefficients of your curve in the form of y^2 = x^3 + ax + b mod p with p greater than %d:" % threshold)
    a = input_int("\ta = ")
    b = input_int("\tb = ")
    p = input_int("\tp = ")
    try:
        E = EllipticCurve(GF(p), [a, b])
        if p >= threshold and E.order() == p:
            P = random.choice(E.gens())
            print("Deal! Here is the generator: (%s, %s)" % (P.xy()[0], P.xy()[1]))
            return E, P
        else:
            raise ValueError
    except Exception:
        print("I don't like your curve. See you next time!")
        exit()


def receive_publickey(E):
    print("Send me your public key in the form of (x, y):")
    x = input_int("\tx = ")
    y = input_int("\ty = ")
    try:
        Q = E(x, y)
        return Q
    except TypeError:
        print("Your public key is invalid.")
        exit()


def banner():
    with open("/banner", "r") as f:
        print(f.read())


def main():
    banner()
    threshold = set_threshold()
    E, P = curve_agreement(threshold)
    Q = receive_publickey(E)
    if attack(E, P, Q):
        print("I know your private key. It's not safe. No answer :-)")
    else:
        print("Here is the answer: %s" % FLAG)


if __name__ == "__main__":
    main()
```

문제의 `launch_attack()`을 보면 위에서 소개한 함수에서 `prec` 을 8로 준 경우임을 확인할 수 있습니다. 문제에서는 이 attack이 통하지 않는 경우를 달라고 합니다. 어떻게 하면 attack을 피할 수 있을까요?

이와 관련해서 검색하다보니 흥미로운 글 [6]을 발견할 수 있었습니다. 이 글의 요지는 다음과 같은 경우 Smart's attack이 정상적으로 동작하지 않는다는 것이었습니다.

```python
sage: p=235322474717419
sage: a=0
sage: b=8856682
sage: E = EllipticCurve(GF(p), [a, b])
sage: P = E(200673830421813, 57025307876612)
sage: Q = E(40345734829479, 211738132651297)
sage: P.order() == p
True
```

그리고 답변을 보면, 해당 curve를 $Q_p$ 위로 옮긴 $E(Q_p)$가 원래 curve의 canonical lift이기 때문이라고 합니다. 이제 canonical lift가 뭔지 알아봅시다.

## Canonical Lift

본래 Smart's attack을 설명하는 원문[7] 에서도 canonical lift이면 해당 공격이 성립하지 않으나, 이러한 경우는 $F_p$에 대해서 $1/p$ 이기 때문에 현실적으로 볼 일이 없다고 언급합니다.

Canonical lift에 대해서는 그렇게 많은 문서가 존재하지 않지만, 다음 책[8]에서 Canonical lift를 다음과 같이 정의하고 있습니다.

![](/assets/images/rbtree/canonicallift.png)

이를 이해하기 위해서는 타원 곡선의 Endomorphism에 대해서 자세히 알아봐야하지만, 생략하고 다음 PPT [9]에서 소개하는 canonical lift의 필요충분조건을 살펴봅시다.

![](/assets/images/rbtree/canonicallift2.png)

밑에 적혀있는 식은 대부분의 경우 쉽게 성립하지 않지만, $j(\mathcal{E})$, 즉 curve의 j-variant가 0이라면 해당 식은 쉽게 성립합니다. j-variant가 0이라는 것은 바로, 곡선 $y^2 = x^3 + ax + b$ 에서 $a$가 0인 경우를 의미합니다. 이런 경우를 어떻게 하면 만들 수 있을까요?

## Bachet Anomalous Primes

$a$가 0인 경우는 일반적으로 Bachet's equation이라고 하며, 16세기 수학자 Claude Gaspar Bachet de Méziriac가 연구했던 equation이라 이렇게 이름이 붙여졌습니다.

Bachet anomalous prime이란, $E(F_p)$의 order가 $p$인 Bachet's equation 꼴의 타원곡선 $E$가 존재하는 소수 $p$를 의미합니다. 해당 PPT [10]는 Bachet anomalous prime의 흥미로운 성질에 대해서 다루는데, 그 중 하나는 바로 $p$가 어떤 정수 $n$ 에 대해서 $3n^2 + 3n + 1$ 꼴이여야 한다는 것입니다.

자, 이제 여기까지 왔으니 한 번 시도를 해봅시다.

우선 $p = 3n^2 + 3n + 1$ 꼴인 $p$를 탐색해봅시다.

```python
sage: while True:
....:     p = random_prime(2^150)
....:     q = 3 * p^2 + 3 * p + 1
....:     if q.is_prime():
....:         break
....:     
sage: p
597483864116511460384629581609997370604419007
sage: q
1070960903638793793346073212977144745230649115077006408609822474051879875814028659881855169
```

(굳이 소수인 $n$을 고른 것은 혹시 모를 경우에 대비해서입니다)

그리고 brute forcing을 통해 anomalous curve를 찾습니다.

```python
sage: i = 1
sage: while True:
....:     E = EllipticCurve(GF(q), [0, i])
....:     if E.order() == q:
....:         break
....:     i += 1
....: 
sage: E
Elliptic Curve defined by y^2 = x^3 + 19 over Finite Field of size 1070960903638793793346073212977144745230649115077006408609822474051879875814028659881855169
sage: E.order() == q
True
```

이 부분이 의외였는데, anomalous curve의 특징이 따로 더이상 없어 brute force 시도를 해봤더니 생각보다 금방 나왔습니다. 해당 `q` 에 대해서만 그런 성질이 성립하나 살펴봤는데, 적어도 시도해본 모든 경우에 있어서 빠르게 anomalous curve를 찾는 것을 확인할 수 있었습니다.

이제 문제에서 원하는 x, y를 구해봅시다. 이는 generator에 아무 값이나 곱해서 임의의 점을 구하면 됩니다.

```
sage: E.gens()
[(850194424131363838588909772639181716366575918001556629491986206564277588835368712774900915 : 749509706400667976882772182663506383952119723848300900481860146956631278026417920626334886 : 1)]
sage: E.gens()[0] * 12345234
(972878821713991831221798207121380773130152849767498373127180632818420916082932921892221544 : 827468985474282678088106409404180331501935386385174577466967374111148487410173918465191175 : 1)
```

이제 구한 `p`, `a=0`, `b=i`, `(x, y)` 값을 보내면 플래그를 얻을 수 있습니다.

# 결론

그래서 Smart's attack이 Bachet anomalous prime을 사용하는 anomalous curve에서 항상 성립하지 않느냐고 하면, 그것은 절대 아닙니다. 앞서 언급했던 글[6]에서 이를 회피하기 위해서는 $E(Q_p)$ 로 온전히 옮기지 말고 원래 커브의 coefficient에 $p$의 배수에 해당하는 난수를 더해 이 문제를 회피하라고 합니다. 어쨌든 $\text{mod}\ p$에 대해서는 똑같은 곡선이기 때문에, 이를 바탕으로 난수 $r, s$를 통해 $y^2 = x^3 + b$와  $F_p$ 상에서 동일한 곡선 $y^2 = x^3 + (p \cdot r) x + (p \cdot s) + b$ 를 만들어 공격을 수행하면, 해당 곡선은 j-invariant가 0이 아니기 때문에 쉽게 canonical lift의 조건이 성립하지 않습니다. 글을 바탕으로 위 문제에서 주어진 `launch_attack()` 을 수정해, 우리가 구한 anomalous curve에 대해서 동작하는 attack 코드를 작성할 수 있습니다.

```python
def launch_attack_fixed(P, Q, p):
    E = P.curve()
    Eqp = EllipticCurve(Qp(p, 8), [ZZ(t) + randint(0, p) * p for t in E.a_invariants()])
    P_Qps = Eqp.lift_x(ZZ(P.xy()[0]), all=True)
    for P_Qp in P_Qps:
        if GF(p)(P_Qp.xy()[1]) == P.xy()[1]:
            break
    Q_Qps = Eqp.lift_x(ZZ(Q.xy()[0]), all=True)
    for Q_Qp in Q_Qps:
        if GF(p)(Q_Qp.xy()[1]) == Q.xy()[1]:
            break
    p_times_P = p * P_Qp
    p_times_Q = p * Q_Qp
    x_P, y_P = p_times_P.xy()
    x_Q, y_Q = p_times_Q.xy()
    phi_P = -(x_P / y_P)
    phi_Q = -(x_Q / y_Q)
    k = phi_Q / phi_P
    return ZZ(k) % p
```

```python
sage: E
Elliptic Curve defined by y^2 = x^3 + 19 over Finite Field of size 1070960903638793793346073212977144745230649115077006408609822474051879875814028659881855169
sage: P = E.gens()[0]
sage: Q = E.gens()[0] * 12345234
sage: launch_attack_orig(P, Q, E.order())
944945964184953405932024859686264605682348261127842528926298926612915067879947366010037886
sage: launch_attack_fixed(P, Q, E.order())
12345234
```

저는 지금까지 아는 지식 상으로는 anomalous curve일 경우 무조건 Smart's attack이 성립한다고 알고 있었습니다. 그렇기에 이런 문제를 푸는 것은 개인적으로 신선한 충격이었고, 많은 것을 배울 수 있었습니다. 지금 이렇게 블로그 글을 정리하면서 제가 갖고 있는 지식을 어떻게 하면 쉽게 풀어낼 수 있을까 고민한 것만으로도 유익한 경험이라고 느낍니다.

하지만 anomalous curve가 왜 이렇게 쉽게 구해지는 지에 대해서는 의문이 있습니다. 이에 대해서는 문제 출제자의 풀이[5]를 봐도 저와 동일하게 $b$를 늘려가며 brute force하기 때문에 더 만족스러운 답을 찾지 못했습니다. 이번에 공부하면서 알게 된 지식을 정리하면서 더 살펴볼 필요성이 있겠습니다.

# 참고 문헌

1. CryptoHack [Link](https://cryptohack.org/)
2. Hensel's lemma [Link](https://en.wikipedia.org/wiki/Hensel's_lemma)
3. Generating Anomalous Elliptic Curves https://lasecwww.epfl.ch/pub/lasec/doc/LMVV05.pdf 
4. Weak Curves In Elliptic Curve Cryptography https://wstein.org/edu/2010/414/projects/novotney.pdf
5. https://github.com/o-o-overflow/dc2020q-notbefoooled-public
6. https://crypto.stackexchange.com/questions/70454/why-smarts-attack-doesnt-work-on-this-ecdlp
7. The Discrete Logarithm Problemon Elliptic Curves of Trace One https://link.springer.com/content/pdf/10.1007/s001459900052.pdf
8. Mathematical Foundations of Elliptic Curve Cryptography https://dmg.tuwien.ac.at/drmota/koppensteinerdiplomarbeit.pdf
9. Canonical Lift Methods https://homes.esat.kuleuven.ac.be/~fvercaut/talks/Satoh.pdf
10. Anomalous Primes and Elliptic Carmichael Numbers https://math.boisestate.edu/reu/publications/AnomalousPrimesAndEllipticCarmichaelNumbers.pdf