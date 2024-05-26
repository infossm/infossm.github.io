---
layout: post

title: "Minimizing Foreign Arithmetic in ZKP Circuits"

date: 2024-05-26

author: rkm0959

tags: [cryptography, blockchain]
---

https://eprint.iacr.org/2024/265.pdf 논문에 대해 다룹니다. 

# 소개 

ZKP를 사용하는 과정에서 가장 핵심적인 부분은 결국 $\mathbb{F}_p$ 위의 기본적인 연산들을 통해서 프로젝트 스펙에서 필요로 하는 constraint들을 표현하는 것에 있습니다. 하지만 $\mathbb{F}_p$ 위에서 모든 것을 표현하는 것이 쉽지만은 않습니다. 대표적인 사례는 바로 foreign field arithmetic인데, 이는 $\mathbb{F}_p$를 표현할 수 있는 ZKP system 위에서 $\mathbb{F}_q$의 연산을 증명하는 것을 의미합니다. 예를 들어, BN254 위의 PLONK를 하고 있다고 치면 BN254의 scalar field가 ZKP 상으로 native 하게 지원되지만, 여기서 SECP256K1 ECDSA verification을 하고 싶다면 SECP256K1의 base field에 대한 논의를 해야합니다. 

실제로 foreign field arithmetic은 overhead가 굉장히 심하고, 이를 피하기 위해서 native field가 프로젝트 스펙 상 원하는 field와 동일하도록 하는 트릭이 많이 연구되기도 했습니다. 이 논문은 추가적인 트릭을 사용해서 foreign group, foreign field를 피하는 방법에 대해서 다룹니다. 

# Part 1: Foreign Group

## Equal Discrete Logarithm: Protocol Description

Group $\mathbb{G}_p$의 크기가 소수 $p$라고 하고, 그 사이의 discrete logarithm을 아무도 알지 못하는 nothing-up-my-sleeve generator $G_p, H_p$를 잡읍시다. 비슷하게, $(\mathbb{G}_q, G_q, H_q)$ 역시 잡읍시다. 이때, 두 Pedersen commitment $X_p, X_q$가 같은 값을 commit 한 것임을 증명할 수 있습니다. 즉, 증명하고자 하는 relation은 

$$R_{dleq} = \{(X_p, X_q), (x, r_p, r_q) : X_p = xG_p + r_pH_p, X_q = xG_q + r_q H_q\}$$

입니다. 단, 이 프로토콜에는 파라미터 $b_x, b_c, b_f$가 있는데, 이때 

$$b_x + b_c + b_f < b_g = \lceil \log_2 (\min(p, q)) \rceil$$

입니다. 또한, $x$는 $[0, 2^{b_x})$에 존재해야 합니다. 이 프로토콜은 결국 $\Sigma$-protocol이고, Fiat-Shamir을 통해서 non-interactive 하게 만들 수 있습니다. 

이 protocol은 $\tau$번 반복되어야 하며, 이때 $\tau \cdot b_c \ge 128$이 성립해야 128-bit security를 얻을 수 있습니다. 이때, computational cost는 $2\tau$ multi-scalar multiplication이고, proof size는 대략 $\tau (b_c + b_f + \log_2 p + \log_2 q)$ 정도입니다. 

프로토콜은 다음 순서로 진행됩니다. 

Step 1
- Prover가 $k \in [0, 2^{b_x + b_c + b_f})$를 하난 sample 합니다.
- Prover가 $t_p \in [0, p), t_q \in [0, q)$를 sample 합니다. 
- $K_p = kG_p + t_pH_p$, $K_q = kG_q + t_q H_q$를 계산합니다. 
- Prover가 $K_p, K_q$를 verifier에게 보냅니다. 

Step 2
- Verifier가 $c \in [0, 2^{b_c})$를 sample 하고 prover에게 보냅니다.

Step 3
- $c \in [0, 2^{b_c})$가 아니면 abort
- $z = k + cx$를 $\mathbb{Z}$ 위에서 계산합니다. 
- $s_p = t_p + cr_p \pmod{p}$, $s_q = t_q + cr_q \pmod{q}$를 계산합니다.
- $z \in [2^{b_x + b_c}, 2^{b_x + b_c + b_f})$가 아니라면 abort 합니다. 
- $z, s_p, s_q$를 verifier에게 보냅니다. 

Step 4
- $zG_p + s_p H_p = K_p + cX_p$를 확인합니다.
- $zG_q + s_qH_q = K_q + cX_q$를 확인합니다. 
- $z \in [2^{b_x + b_c}, 2^{b_x + b_c + b_f})$가 아니라면 abort 합니다. 

Step 5
- 그 후, 다음 relation $R_{rp}$를 증명하는 range proof를 같이 합니다. 

$$R_{rp} = \{(x, r), X_p, b_x: X_p = xG_p + rH_p, 0 \le x < 2^{b_x} \}$$

Step 5의 range proof는 필요한 경우, Bulletproof 계열의 프로토콜로 가능합니다. 

## Proof of Completeness 

Honest execution만 보면 됩니다. Prover 단에서 abort 나지만 않으면 verifier는 성공함을 보이는 것은 쉬우니, prover 단에서 abort 나는지만 봅시다. 다행히, $c$를 고정하면 $z$의 범위는 $[cx, cx + 2^{b_x + b_c + b_f})$ 위에서 uniform 하고, 목표 위치는 $[2^{b_x + b_c}, 2^{b_x+b_c+b_f})$ 이므로 성공 확률은 

$$\frac{2^{b_x + b_c + b_f} - 2^{b_x + b_c}}{2^{b_x + b_c + b_f}} = 1 - 2^{-b_f}$$

입니다. 목표 위치가 $z$의 범위에 완전히 포함됨을 인지하면 자명합니다. 그러므로, prover의 abort 확률은 $2^{-b_f}$입니다. 또한, abort 하지 않았다면 $z$의 분포는 $[2^{b_x + b_c}, 2^{b_x + b_c + b_f})$ 위에서 uniform 함을 알 수 있습니다. 그러므로, 이 protocol은 $2^{-b_f}$ complete.

## Proof of Soundness 

Special soundness 증명을 위해, knowledge error가 최대 $2^{-b_c + 1} + \epsilon_{rp} + \epsilon_{dl}$임을 보일 수 있습니다. 여기서 $\epsilon_{rp}$는 range proof의 knowledge error, $\epsilon_{dl}$은 discrete logarithm 문제 해결의 advantage를 나타냅니다. 이를 증명하기 위해서, extractor를 준비합시다. 

간략하게만 설명하면, special soundness를 증명하는 것이니 동일한 $(K_p, K_q)$에 대해서 accepting transcript $(c, z, s_p, s_q, \pi_{rp})$를 두 개 준비합니다. 두 transcript의 $c$가 다르다고 가정하고, $\pi_{rp}$에서 knowledge extraction을 했다고 가정하고, discrete logarithm을 풀 수 없다고 가정합시다. 일단 

$$K_p + cX_p = zG_p + s_pH_p, \quad K_p + c'X_p = z'G_p + s_p'H_p$$

이므로 $X_p = x_pG + r_p H_p$인 $x_p, r_p$를 추출할 수 있고, $X_q = x_q G + r_q H_q$인 $x_q, r_q$도 추출할 수 있습니다. 또한, 이 추출 과정은 $\pi_{rp}$에서 추출한 것과 동일해야 합니다. 동일하지 않으면, discrete logarithm이 풀리기 때문입니다. 그러니, 예를 들어, $z - cx_p = z' - c'x_p$도 $\mathbb{F}_p$에서 성립하게 됩니다. 또한, $0 \le x_p < 2^{b_x}$도 얻어갈 수 있습니다. 

결론적으로 보면, 적당한 정수 $k,k',a,a',b,b'$가 있어 

$$z = k + cx_p + ap, \quad z' = k + c'x_p + a'p$$

$$z = k' + cx_q + bq, \quad z' = k' + c'x_q + b'q$$

가 성립함을 알 수 있습니다. 동시에, $z, z' \in [2^{b_x + b_c}, 2^{b_x + b_c + b_f})$를 알고 있습니다. 여기서 $x_p = x_q$가 $\mathbb{Z}$ 위에서 성립함을 보이면 충분합니다. 

우선 

$$\lvert a - a' \rvert p \le \lvert z - z' \rvert + x_p \lvert c - c' \rvert < 2^{b_x + b_c + b_f} $$

에서 $a = a'$을 알 수 있습니다. 그러므로, 

$$z - z' = (c-c')x_p = (c-c')x_q + (b-b')q$$

가 성립하여 $(c-c')(x_p-x_q)$가 $q$의 배수가 됩니다. 그런데 $c \neq c'$이므로 $x_p - x_q$가 $q$의 배수가 됩니다. 이미 $x_p$는 $2^{b_x}$ 미만이므로 $\mathbb{F}_q$에서 canonical 하고, 여기서 $x_p = x_q$를 얻습니다. 그러므로 extraction에 성공하고 special soundness의 증명이 끝납니다. 

## Proof of Zero-Knowledge 

Simulator만 만들면 됩니다. $z$를 $[2^{b_x + b_c}, 2^{b_x + b_c + b_f})$에서 random sample 하고, $s_p, s_q$ 각각을 $\mathbb{F}_p, \mathbb{F}_q$에서 random sample 합시다. 이제, $K_p = (zG_p + s_pH_p) - cX_p$로 역산하고 $K_q$도 비슷하게 합시다. 이제 simulator는 $2^{-b_f}$ 확률로 abort 하고, 그 외에는 이 $(K_p, K_q, c, (z, s_p, s_q))$를 전달합니다. 이 분포가 실제 transcript의 분포와 동일함을 증명할 수 있습니다. 여기서 Pedersen commitment의 perfectly hiding 성질이 사용됩니다. 

## Additional Comments 

일반적인 $x \in [0, \min(p, q))$에 대한 equal discrete logarithm을 증명하고 싶다면, 단순하게 $2^{b_x}$ 진법으로 $x$를 전개한 후, 각 chunk에 대한 증명을 각각 진행하면 됩니다. 

abort가 가능하다는 것은 constant time implementation이 아닐 수 있다는 것을 의미하는데, 이에 따른 timing attack을 고려해볼 수 있습니다. 그런데, 애초에 abort 확률이 $2^{-b_f}$로 input과 관계없이 동일하므로, timing attack으로 얻을 수 있는 추가 정보가 없습니다. 

# Part 2: Foreign Field

## Trading Group Operations for Hash Evaluations

Linear morphism $M \in \mathbb{G}^{m \times n}$을 생각합시다. 예를 들어, $M = [G, H]$면 $Mx = x_0 G + x_1 H$, 즉 Pedersen commitment가 됩니다. 여기서, $M \vec{x} = M \vec{x'}$인 $\vec{x} \neq \vec{x'}$을 찾기가 어렵다고 가정합시다. 이제, 증명하고자 하는 relation은 

$$R_{dlhash} = \{(\vec{x}, x_h, \vec{X}) : \vec{X} = M \vec{x}, x_h = H(\vec{x}) \}$$

입니다. 여기서 $H$는 적당한 해시함수면 됩니다. 

또한, 이 프로토콜은 다음을 증명하는 서브 프로토콜을 필요로 합니다. 

$$R_{crh} = \{((\vec{x}, \vec{k}), x_h, k_h, c, \vec{z}) : x_h = H(\vec{x}), k_h = H(\vec{k}), \vec{z} = \vec{k} + c \vec{x} \}$$

이는 단순히 $H$를 가지고 일반 SNARK로 덮는 등의 방법으로 해결할 수 있습니다. group operation이 필요하지 않으며, 해시함수만으로 (특히, vector들이 native field에 있다면 ZKP-friendly hash를 사용해서 native field arithmetic으로 가능) 증명을 할 수 있다는 것이 핵심입니다. 

프로토콜은 다음과 같이 진행됩니다.

Step 1
- Prover가 $\vec{k} \in \mathbb{F}_p^n$를 하나 뽑고, $k_h = H(\vec{k})$와 $\vec{K} = M\vec{k}$를 구합니다. 
- $k_h, \vec{K}$를 verifier에게 보냅니다. 

Step 2
- Verifier가 $\mathbb{F}_p$위의 $c$를 하나 sample 해서 prover에게 보냅니다. 

Step 3
- $\vec{z} = \vec{k} + c\vec{x}$를 계산하고, $\vec{z}$를 verifier에게 보냅니다. 

Step 4
- Verifier는 $M\vec{z} = \vec{K} + c\vec{X}$를 확인합니다. 

또한, 이때 $R_{crh}$에 대한 증명을 같이 진행합니다. 

Completeness는 자명하니, soundness와 zero-knowledge를 증명합시다. 

## Proof of Soundness 

$k_h, \vec{K}$에서 시작해서 서로 다른 두 $c$ 값에 대응되는 accepting transcript를 생각합시다. $(c, \vec{z}), (c', \vec{z}')$을 생각하고 각각에 대한 $\pi_{crh}$를 생각을 하면, $x_h, k_h$가 동일하니 추출되는 $\vec{x}, \vec{k}$도 동일할 것이므로 얻는 결론은 

$$\vec{z} = \vec{k} + c \vec{x}, \quad \vec{z}' = \vec{k} + c' \vec{x}$$

$$M\vec{z} = \vec{K} + c \vec{X}, \quad M\vec{z}' = \vec{K} + c' \vec{X}$$

입니다. 그러니 $(c - c') M \vec{x} = (c - c') \vec{X}$를 얻어 $\vec{X} = M \vec{x}$도 증명이 됩니다. 

## Proof of Zero-Knowledge

Hiding-compatible이라는 강력한 가정을 사용하면 zero-knowledge를 증명할 수 있습니다. 단순히 $c, \vec{z}, \vec{k}$를 뽑고, $\vec{R} = M\vec{z} - c\vec{X}$라고 둔 다음 $r_h = H(\vec{k})$로 두고 $\pi_{crh}$를 simulate 하면 됩니다. 이제 transcript를 $(\vec{R}, r_h, c, \vec{z}, \pi_{crh})$로 두면 됩니다. 

Hiding compatible의 정의는 사실상 위 과정이 zero-knowledge임을 보일 때 쓰는 논리 그 자체와 아예 동일합니다. $M$의 그 과정과 $H$의 그 과정이 너무나도 따로 놀아서, $(Mx, H(x))$와 $(Mx, H(y))$의 distribution이 computationally indistinguishable 하다는 것입니다. 이제 이 정의를 사용하면 위 방식으로 만든 transcript가 simulation을 잘 한다는 사실을 hybrid argument 등으로 보일 수 있습니다. 

이 외에도 논문에서는 foreign boolean arithmetic을 피하는 방법에 대해서 나오는데, AES의 구현에 대해서 설명하고 있습니다. 다만 이는 (Pedersen commitment를 활용한 점을 제외하면) lookup argument를 이용한 꽤 전형적인 방식이라고 생각되어, 이 글에서는 생략했습니다. 

어쨌든 가장 큰 bottleneck 중 하나인 foreign field arithmetic을 다양한 방법으로 최소화하는 것은 매우 중요한 최적화 기법이므로, 이를 더 잘 활용하는 circuit들이 나오면 좋을 것 같습니다. 또한, foreign field를 다루는 것 자체가 ZKP 상에서 많은 취약점의 근원이 되는 곳이기도 하니, 이를 안전하게 활용하는 방법에 대한 고민도 해야 할 것으로 보입니다. 