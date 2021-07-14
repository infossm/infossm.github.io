---

layout: post

title:  "The Attacks on GEA-1 and GEA-2"

date: 2021-06-15

author: rkm0959

tags: [cryptography]

---

# Introduction

최근 한 논문이 암호학계에서 크게 주목을 받고 있습니다. 최고의 컨퍼런스 중 하나인 EUROCRYPT 2021에 수록된 논문인 "Cryptanalysis of the GPRS Encryption Algorithms GEA-1 and GEA-2"라는 논문입니다. 

이 내용은 GEA-1, GEA-2라는 암호체계를 공격하는 알고리즘을 제시하는 논문입니다. 암호를 공격하는 논문은 많은데, 왜 하필 이 논문은 이렇게 주목을 많이 받고 있는 걸까요? 그 이유는 공격 과정에서 GEA-1에 백도어가 있다는 것이 사실상 확정적으로 밝혀졌기 때문입니다. 암호학적 백도어가 무엇인지는 barkingdong님의 http://www.secmem.org/blog/2020/09/20/Cryptographic-Backdoor/ 라는 글을 참고하시길 바랍니다. 이번 글에서는 GEA-1, GEA-2 중 백도어임이 사실상 밝혀진 GEA-1의 공격에 대하여 자세히 알아봅니다. GEA-1의 공격은 특히 암호학적인 지식이 적더라도 이해할 수 있을 정도인만큼, 이번 글을 읽어보는 것을 추천합니다. 이번 글의 목차는 순서대로 다음과 같습니다. 
- GEA-1이 어떻게 설계되었는지 알아봅니다. 
- GEA-1의 공격에 대하여 알아봅니다.
- GEA-1의 공격이 가능한 이유와, 이것이 백도어라고 보는 이유를 알아봅니다.
- 마지막으로, 이번 공격에 대한 여러 학자들의 의견을 알아봅니다.

# Construction of GEA-1

## Part 1 : The Three (Galois) LFSR

GEA-1은 Linear Feedback Shift Register, 즉 LFSR이라는 구조를 기반으로 설계된 암호체계입니다. LFSR에 대해서는 http://www.secmem.org/blog/2019/08/19/Linear-Feedback-Shift-Register/ 라는 글을 참고하시길 바랍니다. LFSR은 매우 빠르다는 장점이 있지만 단순히 LFSR을 하나 사용하면 암호학적인 가치가 없을 정도로 취약합니다. 예를 들어, LFSR를 하나만 사용하여 key stream을 생성할 경우, keystream을 보고 LFSR의 구조와 다음 bit를 예측하는 것은 Berlekamp-Massey 알고리즘을 통하여 할 수 있습니다. 이 알고리즘에 대해서는 https://www.secmem.org/blog/2019/05/17/BerlekampMassey/ 라는 글을 참고하시기 바랍니다. 

결국 LFSR을 이용하여 암호를 만들려면, LFSR을 단순히 사용해서는 안된다는 것입니다. 뭔가 더 복잡한 것을 추가해야한다는 것인데, 예를 들면 LFSR을 여러 개 사용하거나 LFSR의 형태를 변형시키는 방법을 사용합니다. 

이제 자세한 GEA-1의 구조를 설명해보겠습니다. 

GEA-1은 길이가 각각 31, 32, 33인 LFSR $A, B, C$를 갖습니다. 각각은 단순한 LFSR가 아니라 Galois Mode로 작동하는 LFSR입니다. 길이가 $n$인 Galois Mode LFSR은 고정된 $a_0, a_1, \cdots , a_{n-1}$에 대해서, 현재 상태가 $(l_0, l_1, \cdots, l_{n-1})$라 할 때 다음 상태는 $(a_0l_0 + l_1, a_1 l_0 + l_2, \cdots , a_{n-2} l_0 + l_{n-1}, a_{n-1} l_0)$가 됩니다. 

이는 $l_0$가 $0$이면 단순히 bitshift가 되고, $l_0=1$이면 bitshift 후 $a$가 상태에 XOR 되는 것입니다. 

이때, Galois Mode LFSR의 characteristic polynomial은 

$$g(X) = X^n + a_0 X^{n-1} + \cdots + a_{n-2} X + a_{n-1}$$

으로 정의가 됩니다. 

GEA-1의 각 LFSR A, B, C의 characteristic polynomial은 논문의 11 페이지에서 찾을 수 있습니다. 

## Part 2 : The Filter Function

Filter Function이란, LFSR을 더 분석하기 복잡하게 하기 위한 방법입니다. 기본적인 LFSR에서는 linear transformation으로 생성된 다음 상태 bit가 실제로 keystream의 bit가 됩니다. linear transformation의 결과 자체가 keystream bit가 되는 것이니, 분석하기가 상대적으로 쉬운 것입니다. 그렇다면, keystream bit를 생성하기 위해서 nonlinear transformation을 적용하면 해결되지 않을까요? 여기서 nonlinear transformation의 이름이 바로 filter function입니다. LFSR의 현재 상태에 있는 몇 개의 bit를 뽑아낸 다음, 이를 갖고 filter function의 값을 계산해, 이를 keystream의 bit로 사용하는 것입니다. 물론, filter function 역시 매우 잘 디자인해야 이를 통해 안전성을 얻을 수 있을 것입니다. 
GEA-1에서는 7개의 bit를 input으로 받는 degree 4 다변수 다항식을 filter function으로 사용합니다. 실제 함수의 형태는 논문의 6 페이지에서 찾을 수 있습니다. 

GEA-1의 output bit도, A, B, C 각각에 filter function을 적용한 bit 3개를 XOR하여 얻어집니다.

잘못 설계된 filter function은 여러 공격의 대상이 될 수 있으나, 이에 대한 내용은 추후로 미루도록 하겠습니다.

참고 : https://doc.sagemath.org/html/en/reference/cryptography/sage/crypto/boolean_function.html


## Part 3 : The Initialization

이제 GEA-1의 기본 작동 과정을 알았으니, GEA-1을 어떻게 초기화하는지를 알아봅시다. 우선 길이가 64인 NLFSR를 (nonlinear feedback shift register) 준비합니다. 이는 기본적인 LFSR처럼 현재 상태를 shift하고 새로운 bit를 적당히 계산해 상태의 맨 마지막 bit로 두는데, LFSR에서는 새로운 bit를 계산하기 위해서 linear transformation을 사용하고, NLFSR에서는 nonlinear transformation을 사용합니다. GEA-1에서는 이 nonlinear function을 앞서 설명한 filter function과 output bit를 XOR한 결과로 설정합니다. 이제 이 NLFSR을 초기화하는데, 이를 위해서 공개된 initialization vector 32bit $IV_0, IV_1, \cdots , IV_{31}$과 공개된 소통의 방향 1bit $dir$, 그리고 비밀키 64bit $K_0, K_1, \cdots , K_{63}$을 준비합니다. 지금 총 97bit가 있는데, 이를 위해서 전부 0으로 채워진 NLFSR에 이 97개의 bit를 input으로 주면서 clock을 합니다. bit를 input으로 주면서 clock을 한다의 의미는 
- 먼저 NLFSR을 정석적인 방식으로 update 합니다. 그 후, 새로 추가된 bit에 input bit를 XOR 합니다. 

97개의 bit를 input으로 주면서 clock을 한 뒤, 총 128번 NLFSR을 update 합니다. 총 225번의 NLFSR을 update 한 것입니다. 이 상태에서 NLFSR의 상태를 64bit string $s_0 s_1 \cdots s_{63}$이라 합시다. 이제 $A, B, C$를 봅시다.

이를 위해서, 앞서 NLFSR을 초기화한 것과 같이 진행합니다. 전부 0으로 채워진 $A, B, C$에 64bit $s_i$를 $A, B, C$에 input으로 주면서 clock을 합니다. 다만, $A$에서는 $s$를 그대로 순서대로 주고, $B$에서는 $s$를 $16$만큼 shift한 $s_{16}, s_{17}, \cdots , s_{63}, s_0, \cdots , s_{15}$을 주고, $C$에서는 $s$를 $32$만큼 shift한 $s_{32}, s_{33}, \cdots , s_{63}, s_0, \cdots , s_{32}$를 순서대로 줍니다. 만약 이 과정이 모두 끝났는데 $A, B, C$가 모두 $0$으로 채워져 있다면, 하나의 bit를 강제로 $1$로 바꿔줍니다.


# Attack on GEA-1

GEA-1에 대한 공격은, 궁극적으로 **초기화**와 **$A, B, C$ 세부사항**의 결과입니다. 즉, filter function은 GEA-1의 공격에 사용되지 않습니다. 논문의 2.3에서는 GEA-1에서 사용되는 filter function을 분석하는데, 그 결론은 "filter function의 설계는 잘 알려진 원리들을 사용하여 이루어졌다는 것 (generated following known and valid principles)"입니다. 본격적인 공격에 앞서, 간단한 사실들부터 확인해봅시다.

우선 최종 $s$의 결과를 안다면, key 64bit 역시 전부 얻어낼 수 있습니다. 이 과정은 단순히 update 과정을 역연산하는 것으로 어렵지 않게 알 수 있는 사실입니다. 마지막 128번의 update는 단순하게 역연산이 가능하고, 첫 33번의 update는 실제로 $IV$ 값과 $dir$이 공개된 값이므로 직접 할 수 있습니다. key bit가 input으로 주어지는 가운데의 64번의 update가 중요한데, 여기서 NLFSR의 상태를 잘 따라가보면 key를 복구할 수 있습니다. 예를 들어, 첫 33번의 update 후 다음 update를 위해 계산되는 bit의 값을 $b_0$라 한다면, 이 값은 우리가 바로 계산할 수 있는 값이고 실제로 NLFSR의 새로운 마지막 bit는 $K_0 \oplus b_0$가 됩니다. 이 bit는 64번 shift되어, 결국 마지막 128번의 update를 역연산한 상태의 첫 bit가 됩니다. 즉, $K_0 \oplus b_0$의 값 역시 계산할 수 있고 여기서 $K_0$를 얻습니다. 이와 같은 과정을 반복하면, 결국 $K_1, K_2, \cdots , K_{63}$까지 모두 얻을 수 있습니다. 그러니, 목표를 $s$를 찾는 것으로 둬도 충분합니다.

$s$가 주어졌을 때 $A, B, C$를 초기화하는 과정은 분명 linear 합니다. $A, B, C$ 각각을 update고 $s$를 input으로 주는 것은 각각이 전부 linear 하니, 이는 당연한 결과입니다. 특히, $s$가 전부 $0$인 경우 $A, B, C$ 역시 전부 $0$으로 채워지게 되니, 이 update 과정을 요약하면 결국 행렬 $M_A, M_B, M_C$가 있어 

$$\alpha = M_A s, \quad \beta = M_B s, \quad \gamma = M_C s$$

가 되고, $\alpha, \beta, \gamma$는 각각 $A, B, C$의 초기값이 됩니다. 물론, $M_A, M_B, M_C$는 각각 $31 \times 64$, $32 \times 64$, $33 \times 64$ 행렬로, base field는 $\mathbb{F}_2$입니다. $\alpha, \beta, \gamma$가 전부 $0$인 경우는 일어날 가능성이 거의 없으므로, 생각하지 않습니다.

$M_A, M_B, M_C$의 값은 초기화 과정과 $A, B, C$ 세부사항의 결과입니다. 그런데 다음이 성립합니다. 

$$\dim(\ker M_A \cap \ker M_C) = 24, \quad \dim(\ker M_B) = 32, \quad \ker(M_A) \cap \ker(M_B) \cap \ker(M_C) = \{0\}$$

**이게 의미하는 것은 $(\alpha, \gamma)$로 가능한 값의 개수가 $2^{40}$개라는 것**입니다. 이 사실이 **공격의 핵심**입니다.

$T_{AC} = \ker M_A \cap \ker M_C$라 하면, $T_{AC}$는 dimension 24, $\ker M_B$는 dimension 32고 이들은 교집합이 없으므로, 적당한 dimension 8 vector space $V$가 있어 핵심적인 결과인

$$\mathbb{F}_2^{64} =  \ker M_B \oplus T_{AC} \oplus V$$

가 성립합니다. 목표로 하고 있는 $s$가 있으면 유일한 $u \in \ker M_B$, $t \in T_{AC}$, $v \in V$가 있어 

$$s = u + t + v$$

이고, 이를 $\alpha, \beta, \gamma$에 대한 식에 대입하면

$$\beta = M_B(t + v), \quad \alpha = M_A(u + v), \quad \gamma = M_C(u+v)$$

를 얻습니다. 이제 저 공격의 핵심을 이용해, Meet In The Middle 접근을 해봅시다. 

keystream의 연속한 $l$개의 bit를 알고 있다고 가정합시다. 이를 $z_0, z_1, \cdots , z_{l-1}$이라 합시다. 가능한 $(u, v)$ $2^{40}$개를 모두 고려하여, $A, C$ 각각을 $(u, v)$로 초기화, $l$번의 update를 하면서 output bit를 모두 계산하여 

$$a_{u, v}^{(0)}, \cdots , a_{u, v}^{(l-1)}, \quad c_{u, v}^{(0)}, \cdots , c_{u, v}^{(l-1)}$$

을 얻습니다. 여기서 $a_{u, v}^{(k)}$는 $(u, v)$로 $A$를 초기화한 뒤 $A$에서 얻은 $k$번째 output bit를 말하고, $c_{u, v}^{(k)}$는 $C$에 대하여 비슷하게 정의된 값입니다. 
각 key bit는 $A, B, C$에서 나온 output bit들의 XOR로 정의가 됩니다. 그러므로, $\beta_{t, v} = M_B(t + v)$로 초기된 $B$에서 나오는 output bit $l$개는 차례대로 

$$a_{u, v}^{(0)} \oplus c_{u, v}^{(0)} \oplus z_0, \cdots, a_{u, v}^{(l-1)} \oplus c_{u, v}^{(l-1)} \oplus z_{l-1}$$

가 되어야 합니다. 

이 정도면 Meet In The Middle 공격이 가능함은 자명합니다. $A, C$에 대한 brute force 전부, $B$에 대한 brute force 전부를 하고 양쪽에서 match가 되는 값들을 찾아주면 됩니다. 특히, $v$가 $A, C$에 대한 계산과 $B$에 대한 계산에서 동시에 등장하므로, $Tab[v]$에 각 $t$에 대하여 $\beta_{t, v} = M_B (t + v)$로 초기화한 $B$에서 얻어진 길이 $l$의 output bitstream을 저장하면, MITM 계산을 더욱 효율적으로 할 수 있습니다. 즉, 

1. 각 $t, v$에 대하여 $\beta_{t, v}$로 초기화된 $B$에서 얻어진 길이 $l$의 output bitstream을 $Tab[v]$라는 table에 추가한다.
2. 각 $u, v$에 대하여 $(\alpha_{u, v}, \gamma_{u, v})$로 초기화된 $A, C$로 얻어지는 길이 $l$의 output bitstream을 계산하고, 이를 통해서 우리가 알고 있는 keystream을 복원하기 위해 $B$에서 나와야 하는 output bitstream을 계산한다. 이 값이 $Tab[v]$에 있는지 확인하고, 있다면 이를 통해서 $u, t, v$를 복원하고 $s = u + t + v$도 얻는다. 

만약 keystream의 연속한 $l$개의 bit를 알고 있다면, 여기서 가능한 $s$의 개수는 대략 $2^{64-l}$개라고 생각할 수 있습니다. $l$이 대략 $24$ 이상이면, 가능한 $s$의 개수가 $2^{40}$ 수준으로 줄어들어, 이들을 전부 시도해도 시간복잡도에 큰 차이가 없게 됩니다. 하지만 $s$가 정말 우리가 원하는 값인지를 확인하려면, $64$개의 bit를 알고 있어야 합니다. 다만, 이 $64$개의 bit는 연속한 keystream의 bit일 필요는 없고, 흩어져있어도 괜찮습니다. 

결론을 내리자면, 이 공격을 성공시키기 위해서 필요한 사전정보는 
- 전체 65개의 keystream bit를 알고 있어야 함, 그 중 24개 이상은 연속적인 keystream bit이어야 함


# The Backdoor

앞서 보았듯이, 공격의 핵심은 $\dim T_{AC} = 24$입니다. 이게 운이 없어서 일어날만한 일일까요? 이를 확인하기 위해, 논문의 저자들은 랜덤하게 LFSR $A, C$를 생성하고 $M_A, M_C$를 계산, $\dim T_{AC}$의 값을 확인하는 과정을 거쳤습니다. 그 결과는, $10^6$번의 계산을 했을 때 $\dim T_{AC} < 5$인 경우가 99.6% 이상이고, 실제로 얻었던 최대 $\dim T_{AC}$의 값이 $11$이었습니다. 즉, 단순히 운이 없어서 $\dim T_{AC} = 24$가 성립하게 되었다는 설명은 설득력이 매우 낮고, 결국 이는 고의로 암호학적 설계를 잘못했음을 암시합니다. 
실험 코드는 논문의 Appendix에 있습니다.

# The Implications

## Part 1 : The Requirements

앞서 설명한 것과 같이, GEA-1로 암호화되어 전송된 데이터를 복호화하려면,
- 우선 당연히 전송된 암호화된 데이터 자체를 얻어내야 합니다. 이는 이미 알려진 공격이 있었습니다.
- keystream의 65bit를 알아야하고, 그 중 24bit는 연속되어야 합니다. 암호화된 데이터를 알고 있다고 하면, 이는 결국 데이터 안에서 평문 자체를 알고 있는 부분이 있어야 한다는 것인데, 이는 header 등 format을 알고 있으니 문제가 없습니다. 정확하게 설명하면, GEA-1은 GPRS라는 2.5G 이동통신 표준에서 사용되는 암호체계고, 이 GPRS에서 사용되는 header들이 알려져 있으니 이를 이미 알고있는 평문으로 삼으면 됩니다. 
 
## Part 2 : The Severity

GEA-1의 공격은 key 자체를 복원하는 강력한 공격입니다. 그러니, 공격에 성공한다면 전송된 데이터 전체를 복호화할 수 있습니다. 즉, 해당 key를 사용하여 암호화되고 전송된 모든 트래픽을 복호화하여 얻을 수 있습니다. 

특히 GEA-1은 2.5G 표준에서 사용되는 체계지만, 지금 사용되는 스마트폰에서도 지원되는 암호체계입니다. ETSI가 2013년부터 GEA-1을 지원하는 것을 specification 상으로 금지했지만, 실제로는 Samsung Galaxy S9, Apple iPhone 8 등을 포함한 주요 스마트폰이 이를 지원하고 있습니다. 만약 이러한 기기에서 GEA-1을 사용하여 암호화된 데이터가 나온다면, 이를 복호화하는 것도 크게 어렵지 않을 것이라고 예상할 수 있습니다. 

# Conclusion

이 사태에 대한 암호학자 Matthew Green의 반응은 아래 트위터 thread에서 확인할 수 있습니다.
- https://twitter.com/matthew_d_green/status/1405169181880893447

2030년에는 지금 사용하고 있는 암호체계에 대해서 이런 논문이 나올수도 있다는 말이 인상적입니다. 

암호체계에 대한 백도어는 http://www.secmem.org/blog/2020/09/20/Cryptographic-Backdoor/ 에서 나온 것처럼 학문적인 문제임과 동시에 사회적, 정치적인 문제인만큼 많은 사람들의 관심이 필요한 주제입니다.

더 이상 이러한 일이 없기를 바라면서 글을 마치겠습니다. 