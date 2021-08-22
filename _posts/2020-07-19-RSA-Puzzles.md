---
layout: post
title:  "RSA Puzzles"
date:   2020-07-19 19:40
author: RBTree
tags: [cryptography, RSA, CTF]
---

# 서론

RSA는 공개키 암호의 일종으로, 공개키를 통해 평문을 암호화할 수 있으며 비밀키를 통해 암호문을 복호화할 수 있습니다. 이 때 비밀키로부터 공개키를 구할 수는 있지만, 공개키로부터 비밀키를 구하기는 어렵다는 비대칭적 특징을 지닙니다.

일반적으로 RSA에 대한 공격이라고 한다면 공개키만을 갖고 있는 상황에서 비밀키를 구할 수 있는 방법을 의미하곤 합니다. 예를 들어 RSA는 공개키 $(e, N)$과 비밀키 $(d, N)$으로 구성되는데, $N$를 소인수분해해 $N$을 이루는 두 소수 $p, q\ (N = pq)$를 구하거나 암호문으로부터 비밀키의 값을 모르는 상태로 평문 $m$을 구하는 것이 일반적인 공격 방법이라고 할 수 있겠습니다.

하지만 이러한 공격은 어떤 상황에서 성립하는지가 명확하기 때문에, CTF와 같은 대회에서 평범하게 공개키만을 주게 된다면 현존하는 공격 방식을 모두 트라이해보는 단순한 문제로 전락하게 됩니다. 그렇기 때문에 CTF에서 출제되는 문제는 일반적인 RSA에서는 존재해서는 안 되는 **수 사이의 관계성**이나, 공개키와 함께 원래는 **알면 안 되는 정보**를 제공하곤 합니다.

전자의 경우는 현실에서도 종종 발생하곤 합니다. 가장 대표적인 예는 2017년의 ROCA Attack([Link](https://en.wikipedia.org/wiki/ROCA_vulnerability))를 들 수 있을 것입니다. ROCA Attack의 경우 $N$을 이루는 소수 $p$, $q$의 생성 과정의 취약점을 공격해 $N$으로부터 $p$와 $q$를 복구합니다.

하지만 후자의 경우는 공개키 이외의 값을 굳이 공개할 이유가 없기에 현실에서 보기 매우 어렵습니다. 그렇기에 이러한 문제들의 풀이 방식은 RSA Attack이라고 하기 보다는 공개키 이외의 정보를 주고 비밀키를 풀어보라고 시키는 **RSA Puzzle** Solving이라고 부르는 것이 더 합당할 것입니다.

개인적으로 RSA Puzzle을 푸는 것을 통해서 다음과 같은 것들을 얻을 수 있다고 생각합니다.

- RSA에 대한 피상적인 이해를 넘어서기
- 정수론에 대한 감각 터득하기
- 나아가 실제 RSA 공격에 대한 이해도의 증가

이 글에서는 RSA Puzzle에 해당하는 세 개의 일련의 문제를 살펴보고자 합니다.

# 본론

## PlaidCTF 2019 - R u SAd?

[문제 관련 링크](https://ctftime.org/task/8209)

문제에서 핵심이 되는 코드를 추려보자면 다음과 같습니다.

```python
class Key:
	PRIVATE_INFO = ['P', 'Q', 'D', 'DmP1', 'DmQ1']
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		assert self.bits % 8 == 0

	def ispub(self):
		return all(not hasattr(self, key) for key in self.PRIVATE_INFO)

	def ispriv(self):
		return all(hasattr(self, key) for key in self.PRIVATE_INFO)

	def pub(self):
		p = deepcopy(self)
		for key in self.PRIVATE_INFO:
			if hasattr(p, key):
				delattr(p, key)
		return p

	def priv(self):
		raise NotImplementedError()

def genkey(bits):
	assert bits % 2 == 0
	while True:
		p = genprime(bits // 2)
		q = genprime(bits // 2)
		e = 65537
		d, _, g = egcd(e, (p-1) * (q-1))
		if g != 1: continue
		iQmP, iPmQ, _ = egcd(q, p)
		return Key(
			N=p*q, P=p, Q=q, E=e, D=d%((p-1)*(q-1)), DmP1=d%(p-1), DmQ1=d%(q-1),
			iQmP=iQmP%p, iPmQ=iPmQ%q, bits=bits,
		)
```

Key를 정의하면서 `N, P, Q, E, D, DmP1, DmQ1, iQmP, iPmQ, bits` 정보가 들어가지만, `Key`의 `pub` 메소드에서는 이 중 `P, Q, D, DmP1, DmQ1`만을 지워 `iQmP, iPmQ` 라는 정보를 남깁니다.

`iPmQ`는 $p^{-1}\mod q$, `iQmP`는 $q^{-1}\mod p$를 의미합니다. 그런데 보시다시피, `iPmQ, iQmP`의 유도 과정을 보면 q와 p에 대한 extended euclidean algorithm으로 구하는 것을 살펴볼 수 있습니다. 즉 마지막에 `iQmP%p, iPmQ%q`를 취하기 전 `iPmQ, iQmP`에 대해서 $iPmQ\cdot p + iQmP\cdot q = 1$이란 식이 성립함을 알 수 있습니다. 이 식이 성립하면 왜 위에 설명한 역수의 성질을 만족할까요?

식 $iPmQ\cdot p + iQmP\cdot q$를 생각해봅시다. 이 식에 $\text{mod}\ p$를 귀하면 $iPmQ\cdot 0 + iQmP\cdot q = 1 (\text{mod}\ p)$가 될 것입니다. 즉 $iQmP = q^{-1} (\text{mod}\ p)$가 됩니다. $iPmQ$에 대해서도 동일하게 생각할 수 있겠습니다.

그런데 더 나아가서, $iPmQ\cdot p + iQmP\cdot q$는 $\text{mod}\ p$에 대해서도 1이고 $\text{mod}\ q$에 대해서도 1이므로 중국인의 나머지 정리에 따라서 $\text{mod}\ pq$, 즉 $\text{mod}\ N$에 대해서도 1임을 알 수 있습니다. $p$와 $q$를 모르지만 $iPmQ, iQmP, N$에 대해서 $iPmQ\cdot p + iQmP\cdot q = 1 (\text{mod}\ N)$이 성립한다는 사실을 알게 되었습니다.

이를 어떤 정수 $k$ 에 대해서 $iPmQ \cdot p + iQmP \cdot q = k\cdot N + 1$ 이라고 적어봅시다. 그러면 우리가 갖고 있는 $iPmQ$와 $iQmP$ 값은 각각 $0 \leq iPmQ < q$, $0 \leq iQmP < p$가 성립하기 때문에 $0 \leq k\cdot N + 1 = iPmQ \cdot p + iQmP \cdot q < 2pq = 2N$ 이라는 식이 성립해야합니다. 곧 $k$가 1이여야 하며, $iPmQ \cdot p + iQmP \cdot q = N + 1$ 라는 식이 성립함을 알 수 있습니다.

이제 위에서 구한 식으로부터 $p$와 $q$를 구해봅시다. $iPmQ \cdot p + iQmP \cdot q = N + 1$의 양변에 $p$를 곱하면 $iPmQ \cdot p^2 + iQmP \cdot N = N \cdot p + p$가 됩니다. 정리하면 $iPmQ \cdot p^2 - (N + 1) p -+ iQmP \cdot N = 0$이라는 2차 방정식을 얻게 되므로, 이 식을 풀게 되면 $p$를 알 수 있게 됩니다. 이 때 판별식의 square root를 구해야하는 과정에서 Python의 경우 큰 정수에 대한 square root를 구하는 함수를 제공하지 않습니다. 이 경우 gmpy2의 `isqrt`나 `iroot`를 쓰면 쉽게 값을 구할 수 있습니다.

2차 방정식을 풀어 $p$를 얻으면 자연스럽게 $q$도 얻을 수 있게 되고, 곧 $d$의 값도 얻어낼 수 있게 됩니다. 이를 통해 문제를 풀 수 있습니다.

### 방법론

여기에서 얻을 수 있는 중요한 RSA Puzzle의 방법론이 있습니다.

- $iPmQ, iQmP$ 등의 값이 주어질 때, $iPmQ\cdot p + iQmP\cdot q$를 생각한다. ([Bézout's identity](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity))

## TokyoWesterns CTF 2019 - Happy!

[문제 링크](https://ctftime.org/task/9135)

아니나 다를까, 위의 R u SAd? 에서 영감을 받은 Happy! 라는 문제가 TWCTF 2019에서 나옵니다. 이 문제는 ruby로 작성되어있는데, 핵심이 되는 코드를 추려보자면 다음과 같습니다.

```ruby
class Key
    def initialize(attr)
        @attr = attr
    end

    def self.generate_key(bits, k)
        while true
            p = OpenSSL::BN::generate_prime(bits, true).to_i
            q = OpenSSL::BN::generate_prime(bits, true).to_i
            e = 65537
            next if e.gcd((p - 1) * (q - 1) * q ** (k - 1)) > 1
            d1 = e.pow((p - 1) / 2 - 2, (p - 1))
            fail unless d1 * e % (p - 1) == 1 
            d2 = e.pow(((q - 1) / 2 - 1) * (q - 1) * (k > 1 ? q ** (k - 2) : 1) - 1, q ** (k - 1) * (q - 1))
            fail unless d2 * e % (q ** (k - 1) * (q - 1)) == 1 
            cf = p.pow(q ** (k - 1) * (q - 1) - 1, q ** k)
            return Key.new({
                n: p * q ** k,
                e: e,
                p: p,
                q: q ** k,
                d1: d1,
                d2: d2,
                cf: cf,
            })
            break
        end
    end

    def private?
        @attr.key?(:d1) && @attr.key?(:d2) && @attr.key?(:p) && @attr.key?(:q)
    end

    def public?
        @attr.key?(:n) && @attr.key?(:e)
    end

    def public_key
        Key.new(@attr.reject{|k, v| [:p, :q, :d1, :d2, :ce].include?(k)})
    end
end
```

우선 `generate_key`를 보면 특이하게도 `n = p * q ** k`라는 방법으로 `n`을 생성합니다. 즉, $pq$ 가 아닌 어떤 $k$에 대한 $pq^k$로 정의하는 것이죠.

그리고 `generate_key`는 분명 `n, e, p, q, d1, d2, cf` 값을 설정합니다. 하지만 `public_key` method는 이 중 `p, q, d1, d2`, 그리고 **`ce`**를 삭제한 나머지를 export합니다. 결과적으로 `n, e, cf`가 남게 되어, `cf`라는 값이 남게 됩니다.

`cf`의 생성 과정을 살펴보면 $p^{q^{k-1}(q-1) - 1}\mod q^k$ 라는 값을 설정합니다. $\text{mod}\ q^k$에서 $\phi(q^k) = q^{k-1} (q-1)$이기 때문에, 오일러 정리에 따라 $q$와 서로소인 임의의 정수 $a$에 대해서 $a^{q^{k-1} (q-1)} = 1 (\text{mod}\ q^k)$임을 알고 있습니다. 곧, `cf`는 $p^{-1}\mod q^k$를 의미함을 알 수 있습니다.

이를 통해 다음과 같은 식을 세울 수 있습니다: $p - cf^{-1} = 0 (\text{mod}\ q^k)$. 그런데 이 식은 Coppersmith method라는 특수한 방식에 의해서 풀리는 식입니다. Coppersmith method에 대해서는 뒤에서 설명하고, 성립하는 조건을 살펴보면 다음과 같습니다. ([참고 링크](http://blog.rb-tree.xyz/2020/03/10/coppersmiths-method/))

> 어떻게 소인수분해가 되는지 모르는 $N$이 있고, $N$의 약수 $b \geq N^\beta$가 있다. $f_b(x)$는 monic인 일변수 다항식이고, 차수 $\delta$를 가진다.
>
> 우리는 다음 조건을 만족하는 $f_b(x) = 0 (\text{mod}\ b)$의 해 $x_0$을 $(\log N, \delta, \frac{1}{\epsilon})$에 대한 다항 시간 안에 구할 수 있다.
>
> $\lvert x_0 \rvert \leq \frac{1}{2}N^{\beta^2 / \delta-\epsilon}$

(monic이라는 것은 최고차항의 계수가 1인 경우를 의미합니다.)

식 $p - cf^{-1} = 0 (\text{mod}\ q^k)$에 대해서, $f_b(x) = x - cf^{-1}$ 로 정의한다고 생각합시다. (이 때 $cf^{-1}$은 $t = cf^{-1}\mod N$을 구하게 되면 $t = cf^{-1} (\text{mod}\ q^k)$가 성립하므로 해당 $t$ 값을 사용하면 됩니다.) $q^k$의 경우, $p$와 $q$의 bit-size가 동일하고 $k \geq 2$이기 때문에 대략적으로 $q^k \geq N^{2/3}$ 임을 알 수 있습니다. $\beta = \frac{2}{3}$으로 두고 $x_0$의 제한 조건을 정리해보면 $\frac{1}{2}N^{\beta^2 / \delta-\epsilon} = \frac{1}{2}N^{4/9 - \epsilon}$인데, $p$의 값은 대략적으로 $p \leq N^{1/3}$이므로 제한 조건 안에 들어간다는 것을 알 수 있습니다.

Coppersmith method는 sage에서 `small_root` 라는 메소드를 통해 구현이 되어있습니다. 다음과 같이 코드를 세워  $p$를 구할 수 있습니다.

```python
e = 65537
N = 5452318773620154613572502669913080727339917760196646730652258556145398937256752632887555812737783373177353194432136071770417979324393263857781686277601413222025718171529583036919918011865659343346014570936822522629937049429335236497295742667600448744568785484756006127827416640477334307947919462834229613581880109765730148235236895292544500644206990455843770003104212381715712438639535055758354549980537386992998458659247267900481624843632733660905364361623292713318244751154245275273626636275353542053068704371642619745495065026372136566314951936609049754720223393857083115230045986813313700617859091898623345607326632849260775745046701800076472162843326078037832455202509171395600120638911
cf = 25895436290109491245101531425889639027975222438101136560069483392652360882638128551753089068088836092997653443539010850513513345731351755050869585867372758989503310550889044437562615852831901962404615732967948739458458871809980240507942550191679140865230350818204637158480970417486015745968144497190368319745738055768539323638032585508830680271618024843807412695197298088154193030964621282487334463994562290990124211491040392961841681386221639304429670174693151
    
P.<x> = PolynomialRing(Zmod(N))
f = cf * x - 1
f = f.monic()
# 앞서 설명에서는 p - cf^-1으로 설명했지만, 이렇게 monic 메소드를 사용하면 더 쉽게 구할 수 있습니다.

roots = f.small_roots(beta=0.66)
# 더 엄밀하게 하기 위해서는 위의 제한 조건 식으로부터 epsilon의 최대값을 구하는 것이 좋습니다.
# sage는 일반적으로 beta^2 / 8 를 epsilon으로 사용합니다.

print(roots)
```

### 방법론

여기에서 얻을 수 있는 중요한 RSA Puzzle의 방법론이 있습니다.

- 어떤 식의 해를 구해야 하는 상황에서 해의 범위가 충분히 작다면, Coppersmith method를 의심해볼 수 있다.

## TSG CTF 2020 - Modulus Amittendus

[문제 링크](https://ctftime.org/writeup/22196)

또 위의 Happy! 로부터 영감을 받아 만들어진 Ruby로 작성된 문제입니다. 이 문제의 핵심 코드는 다음과 같습니다.

```ruby
class RSA
  def initialize
    @p = OpenSSL::BN::generate_prime(1024, true).to_i
    @q = OpenSSL::BN::generate_prime(1024, true).to_i
    @n = @p * @q
    @e = 65537
    @d = modinv(@e, (@p - 1) * (@q - 1))
    @exp1 = @d % (@p - 1)
    @exp2 = @d % (@q - 1)
    @cf = modinv(@q, @p)
  end

  def encrypt(m)
    m.pow(@e, @n)
  end

  def decrypt(c)
    m1 = c.pow(@exp1, @p)
    m2 = c.pow(@exp2, @q)
    (m2 + @cf * (m1 - m2) % @p * @q) % @n
  end

  def pubkey
    privkey.to_a[..2].to_h
  end

  def privkey
    {
      e: @e,
      n: @d,
      cf: @cf,
      p: @p,
      q: @q,
      exp1: @exp1,
      exp2: @exp2,
    }
  end
end
```

Export된 값은 `e, n, cf` 이지만 `privkey`를 보다시피 `n: @d,` 를 통해 `n`이 아닌 `d`를 export하고 있습니다. 문제는 오히려 `n`을 모르게 되었다는 점입니다.

이 문제의 경우, RSA Puzzle에서 자주 쓰이는 식인 $ed = k\phi(N) + 1$ 이라는 관계식을 사용할 수 있습니다. ($k$는 임의의 정수) $d$는 $e^{-1}\mod \phi(N)$이기 때문이죠. 중요한 것은 $0 \leq d < \phi(N)$이기 때문에, $0 \leq k\phi(N) + 1 = ed < e\phi(N)$ 이 성립해 $k$ 값이 $1$부터 $e - 1$ 사이라는 정보를 얻을 수 있습니다. 이 때 $e$가 65537로 작기 때문에, 가능한 $k$ 값을 모두 시도해볼 수 있습니다. 이를 통해 $\phi(N)$ 의 후보 리스트 (최대 65536개)를 구할 수 있습니다.

이제 문제는 $cf = q^{-1} (\text{mod}\ p)$를 어떻게 사용하느냐입니다. 일단 $\phi(N)$을 $p$로 나눠보면 $\phi(N) = (p - 1)(q - 1) = -q + 1 (\text{mod}\ p)$ 임을 알 수 있습니다. 살펴보니 $q$만 남아서 $cf$를 곱하기 좋은 꼴입니다. $(\phi(N) - 1)$ 에 $cf$를 곱하게 되면, $cf (\phi(N) - 1) = q^{-1} (-q) = -1 (\text{mod}\ p)$가 성립합니다. 즉, $cf(\phi(N) - 1) + 1$은 $p$의 배수입니다.

여기서 $p$에 대해서 일반적으로 성립하는 식을 생각해보면, 페르마의 소정리에 따라 $p$와 서로소인 $a$에 대해 $a^{p-1} = 1 (\text{mod}\ p)$라는 것을 떠올릴 수 있습니다. 그런데 여기에 $q-1$ 승을 취해도 값은 딱히 변하지 않습니다. 즉, $a^{\phi(N)} = a^{(p-1)(q-1)} = 1^{q-1} = 1 (\text{mod}\ p)$입니다. 곧, 적당히 고른 수 $a$에 대해서 $a^{\phi(N)} - 1$은 $p$의 배수입니다.

이렇게 되면 우리는 $p$의 배수인 수 $cf(\phi(N) - 1) + 1$와 $a^{\phi(N)} - 1$를 알게 되었습니다. 그리고 $\gcd(cf(\phi(N) - 1) + 1, a^{\phi(N)} - 1)$ 또한 $p$의 배수여야겠죠.  $a^{\phi(N)} - 1$ 끼리 gcd를 취하는 것은 $p$ 에 대한 어떤 정보도 주지 않지만, 또다른 $p$의 배수 $cf(\phi(N) - 1) + 1$와 임의의 $a$에 대한 $a^{\phi(N)} - 1$의 gcd를 계속 취하는 과정을 통해서 $p$를 구할 수 있습니다.

이를 통해 다음과 같이 솔버를 작성할 수 있습니다.

```python
from Crypto.Util.number import GCD, long_to_bytes

e = 65537
d = 27451162557471435115589774083548548295656504741540442329428952622804866596982747294930359990602468139076296433114830591568558281638895221175730257057177963017177029796952153436494826699802526267315286199047856818119832831065330607262567182123834935483241720327760312585050990828017966534872294866865933062292893033455722786996125448961180665396831710915882697366767203858387536850040283296013681157070419459208544201363726008380145444214578735817521392863391376821427153094146080055636026442795625833039248405951946367504865008639190248509000950429593990524808051779361516918410348680313371657111798761410501793645137
cf = 113350138578125471637271827037682321496361317426731366252238155037440385105997423113671392038498349668206564266165641194668802966439465128197299073392773586475372002967691512324151673246253769186679521811837698540632534357656221715752733588763108463093085549826122278822507051740839450621887847679420115044512
ct = 17320751473362084127402636657144071375427833219607663443601124449781249403644322557541872089652267070211212915903557690040206709235417332498271540915493529128300376560226137139676145984352993170584208658625255938806836396696141456961179529532070976247738546045494839964768476955634323305122778089058798906645471526156569091101098698045293624474978286797899191202843389249922173166570341752053592397746313995966365207638042347023262633148306194888008613632757146845037310325643855138147271259215908333877374609302786041209284422691820450450982123612630485471082506484250009427242444806889873164459216407213750735305784

for k in range(1, e + 1):
    if (e * d - 1) % k != 0:
        continue
    
    phi = (e * d - 1) // k
    if phi.bit_length() > 2048:
        continue
    
    kp = cf * (phi - 1) + 1

    gcd_result = kp
    a = 2

    # if phi % (gcd_result - 1) == 0, then it's phi
    while phi % (gcd_result - 1) != 0 and a <= 1000:
        # gcd(pow(a, phi), kp) is same as gcd(pow(a, phi) % kp, kp)
        # so we don't need to calculate full pow(a, phi)
        tmp = pow(a, phi, kp) - 1

        gcd_result = GCD(tmp, gcd_result)
        a += 1
        
    # Maybe this phi candidate is not a real phi of N
    if a > 1000:
        continue
    
    p = gcd_result
    q = phi // (p - 1) + 1

    n = p * q

    print(long_to_bytes(pow(ct, d, n)))
    break

```

### 방법론

여기에서 얻을 수 있는 중요한 RSA Puzzle의 두 방법론이 있습니다.

- $e$와 $d$에 대해서 $ed = k\phi(N) + 1$ 이라는 식이 성립하며, $k$의 범위는 $e$의 크기를 따라간다. 일반적으로 $e$가 작기 때문에 이를 통해 $k$를 순회하는 것 또한 가능하다.
- $p$의 배수를 둘 이상 구할 수 있는 경우, 둘의 최대공약수를 구하는 방법을 통해 $p$ 혹은 $p$의 작은 배수를 구하는 것이 가능하다.

#### 참고

이 문제가 R u SAd?와 Happy!로부터 영감을 받았다는 것을 알 수 있는 부분은 바로 문제의 플래그입니다:

```
TSGCTF{Okay_this_flag_will_be_quite_long_so_listen_carefully_Happiness_is_our_bodys_default_setting_Please_dont_feel_SAd_in_all_sense_Be_happy!_Anyway_this_challenge_is_simple_rewrite_of_HITCON_CTF_2019_Lost_Modulus_Again_so_Im_very_thankful_to_the_author}
```

# 결론

서로 연관된 이 세 문제는 모두 서로 다른 아이디어를 통해 풀어야 하는 문제입니다. 또한 세 문제에서 필요한 아이디어는 RSA Puzzle을 푸는 데에 있어 필요한 거의 모든 아이디어를 담고 있습니다.

그럼에도 저는 위의 세 문제 중 Happy!와 Modulus Amittendus를 시간 내로 풀지 못했습니다. 해당 문제의 키 아이디어 자체는 매우 기초적임에도 불구하고 다른 아이디어가 맞는 길이라고 생각하거나, 기초적인 식이나 수 사이의 관계를 떠올리지 못해 풀지 못한 경우입니다.

그런 점에서 이 글은 저의 일종의 반성문이기도 하고, RSA Puzzle이 잘 풀리지 않을 때 참고하면 좋은 글이기도 합니다. 맞는 아이디어 같은데 잘 풀리지 않거나, 지금 생각하는 아이디어가 틀린 아이디어인 것 같을 때, 이 글을 참고하면서 생각해보지 못한 수 사이의 관계를 돌아볼 수 있으면 좋겠습니다.

# 참고 문헌

1. ROCA vulnerability [https://en.wikipedia.org/wiki/ROCA_vulnerability](https://en.wikipedia.org/wiki/ROCA_vulnerability)
2. PlaidCTF 2019 - R u SAd? [https://ctftime.org/task/8209](https://ctftime.org/task/8209)
3. TokyoWesterns CTF 5th 2019 - Happy! [https://ctftime.org/task/9135](https://ctftime.org/task/9135)
4. TSG CTF 2020 - Modulus Amittendus [https://ctftime.org/writeup/22196](https://ctftime.org/writeup/22196)
5. Coppersmith's method [http://blog.rb-tree.xyz/2020/03/10/coppersmiths-method/](http://blog.rb-tree.xyz/2020/03/10/coppersmiths-method/)
