---
layout: post

title: "Isochronous Gaussian Sampling of [HPRR20]"

date: 2022-08-19

author: rkm0959

tags: [cryptography]
---

논문은 https://eprint.iacr.org/2019/1411.pdf 입니다.

# Post Quantum Cryptography and Lattices

지금 사용되고 있는 많은 암호화 체계, 예를 들면 블록체인에서 많이 사용되고 있는 ECDSA/EDDSA나 공개키 암호화를 위해 사용되는 RSA 등은 전부 충분히 강력한 양자컴퓨터가 나오면 더 이상 안전하지 않게 됩니다. 양자컴퓨터가 나오는 미래에 대비하기 위해 NIST는 2010년대 후반부터 양자컴퓨터가 나오더라도 안전한, Post Quantum Cryptography에 (이하 PQC) 대한 표준을 정하기 위한 과정을 밟기 시작했습니다. 여러 과정을 거쳐서 매우 최근 Round 4 발표 및 첫 표준이 등장하게 되었는데, PQC 암호체계를 구축하기 위한 접근은 다양하지만 굵직한 것들을 소개하면 다음과 같습니다.

- 격자 기반 암호학
- 코드 기반 암호학
- 다변수 기반 암호학
- 해시 기반 암호학
- Isogeny 기반 암호학

그 중 최근 다변수 기반 암호학 중 Round 3 후보군이었던 Rainbow가 깨지고 (by Ward Beullens) Round 4 후보군인 SIKE도 깨지고 (by Castryck, Decru) 그 와중에 많은 격자 기반 암호학들이 (CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON 등) NIST 표준으로 등재되면서 격자 기반 암호학이 강세를 보이고 있는 상황입니다. HEAAN 등 homomorphic encryption도 격자 기반이다보니 격자에 대한 관심은 더 커질 것 같습니다.

격자 기반 암호학에 대한 여러 글은 이미 이 사이트에도 있으니, 자세한 설명은 그 글들에 맡기겠습니다.

- https://www.secmem.org/blog/2020/11/22/Post-Quantum-Cryptography/
- https://www.secmem.org/blog/2020/12/20/SABER/
- https://www.secmem.org/blog/2020/10/23/SVP-and-CVP/

# FALCON, Timing Attacks and Isochronous Algorithms

격자 기반 서명 체계인 FALCON은 NTRU를 발전시킨 알고리즘입니다. FALCON에 대한 내부 체계는 추후 글에서 설명하도록 하고, 일단 이 글에서 가장 중요한 점은 이 과정에서 Discrete Gaussian 분포에서 Sampling을 해야한다는 점입니다.

즉,

$$\rho_{\sigma, \mu}(x) = \exp \left( - \frac{(x-\mu)^2}{2\sigma^2} \right)$$

가 있을 때, Discrete Gaussian 분포

$$D_{S, \sigma, \mu}(x) = \frac{\rho_{\sigma, \mu}(x)}{\sum_{z \in S} \rho_{\sigma, \mu}(z)}$$

에 맞게 $S$의 원소를 뽑는 과정이 필요합니다. 보통 $S = \mathbb{Z}, \mathbb{Z}^+$입니다.

그런데 이 과정에서 어떤 값이 뽑혔는지와 뽑는데 걸린 시간 사이의 상관관계가 있다면, 시간을 측정해서 선택된 값을 유추할 수 있게 됩니다. 이러한 공격을 Timing Attack이라고 부르기도 합니다.

최근 Timing Attack에 대한 세미나를 진행했습니다.

- https://github.com/rkm0959/rkm0959_presents/blob/main/TimeBasedCryptography.pdf

이러한 공격을 막으려면, 계산 시간과 결과가 독립적인 Isochronous 알고리즘을 개발해야 합니다. 위 발표자료를 보시면 알 수 있지만, 이 점은 FALCON의 보완해야 하는 점이었습니다. 하지만 [HPRR20] 논문이 등장하면서 괜찮아졌습니다.

# Isochronous Gaussian Sampling of [HPRR20]

먼저 FALCON에서는, $\sigma$의 범위를 $\sigma \in [\sigma_{\min}, \sigma_{\max}]$로 제한할 수 있습니다. 특히 $\sigma_{\max} \approx 1.8205$.

[HPRR20]의 접근은, 다음과 같습니다.

- Isochronous 하게 Sample 할 수 있는 BaseSampler를 하나 구축
- BaseSampler + Isochronous Rejection Sampling을 기반으로 Sampler 구축

실제로는 BaseSampler를 $D_{\mathbb{Z}^+, \sigma_{\max}, 0}$에 대해서 구축하고, 이를 기반으로 Linear Transform 및 Rejection Sampling을 거쳐 실제 원하는 분포인 $D_{\mathbb{Z}, \sigma, \mu}$에 도달하게 됩니다. 먼저 어떻게 변환을 하는지를 알아보도록 합시다.

## Transformation and Rejection Sampling

BaseSampler에서 $z_0$를 뽑았다고 합시다. 먼저 random bit $b$를 뽑고 $z = (2b-1)z_0 + b$라 합시다. 이제 $z$를 선택하기 위해서 Rejection Sampling을 하는데,

$$BG_{\sigma_{\max}}(z) = \frac{1}{2} \begin{cases} D_{\mathbb{Z}^+, \sigma_{\max}}(-z) & z \le 0 \\ D_{\mathbb{Z}^+, \sigma_{\max}}(z-1) & z \ge 1 \end{cases} $$

가 $z$의 분포이니 Rejection Sampling을 하려면

$$ \frac{D_{\mathbb{Z}, \sigma, \mu}(z)}{BG_{\sigma_{\max}}(z)} = \exp \left( \frac{z_0^2}{2\sigma^2_{\max}} - \frac{(z - \mu)^2}{2 \sigma^2} \right) $$

에 비례하는 확률로 Accept를 해야 합니다. 이를 위해서

$$ x = \frac{z_0^2}{2\sigma^2_{\max}} - \frac{(z - \mu)^2}{2 \sigma^2} $$

라고 하면, Isochronous 하게 $\exp(x)$를 계산하고 이에 비례하는 확률로 Accept를 하게 됩니다.

이때, Accept 확률은 시도 횟수와 연관되고 이는 다시 실행 시간과 직결되므로, 이 확률을 출력 결과와 독립적이어야 합니다. 이를 위해서 실제로는 성공 확률을

$$ \frac{\sigma_{\min}}{\sigma} \cdot \exp(x)$$

으로 두게 됩니다. 이후 Isochronous 하게 Bernoulli Sampling을 하면 (exercise for reader) 끝입니다.

이러면 이제 해결해야 하는 문제는 다음과 같습니다.

- BaseSampler는 어떻게 구축할 것인가
- $\exp$를 어떻게 계산할 것인가
- 왜 이 알고리즘은 Isochronous인가

이에 대한 설명은 간단하게 하겠습니다.

## Construction of BaseSampler

$\sigma_{\max}$가 작아서, 여기서는 직접 구축이 가능합니다. 약 $18$까지의 값에 대해서 Cummultative Distribution Table을 구축해놓으면, 단순 uniform random sample만 사용해도 빠르고 Isochronous하게 BaseSampler를 사용할 수 있게 됩니다. 무시 가능한 수준의 오차가 발생하게 되는데, 이에 대한 처리는 Renyi Divergence를 사용하여 진행하게 됩니다. 이는 Exponential 계산에서 발생하는 실수 오차에 대해서도 비슷한데, 자세한 부분은 이 글에서는 생략하도록 하겠습니다.

## Calculation of Exponential

계산 범위를 $[0, \ln 2]$로 축소시킨 다음에 다항식 근사를 적용시킵니다. 대충 10차 다항식으로 근사하면 $2^{-47}$의 relative error를 가진 근사식을 유도할 수 있습니다. 다항식의 계산은 물론 Isochronous하게 할 수 있습니다.

## Why is the Algorithm Isochronous

BaseSampler, Exponential이 모두 Isochronous 하므로, Rejection Sampling의 통과 확률이 거의 상수면 Sampling 과정 전체가 Isochronous 함을 증명하는 것과 마찬가지라고 볼 수 있습니다.

성공 확률을 열심히 계산해보면

$$ \frac{ \sigma_{\min} \cdot \sum_{z \in \mathbb{Z}} \rho_{\sigma, \mu}(z)}{2 \cdot \sigma \cdot \sum_{z \in \mathbb{Z}^+} \rho_{\sigma_{\max}, 0}(z)}$$

가 됨을 알 수 있습니다. 여기서

- $\sigma_{\min}$의 lower bound (생략했으나 실제로는 smoothing parameter가 들어가게 됩니다)
- $\rho$의 합에 대해서 사용할 수 있는 Poisson Summation Formula
- $\rho_{\sigma, \mu}$와 $\rho_{\sigma, 0}$ 사이에 대한 여러 알려진 bound 등

를 합치면 결론적으로 위 확률이

$$ \frac{\sigma_{\min} \cdot \sqrt{2 \pi}}{2 \cdot \sum_{z \in \mathbb{Z}^+} \rho_{\sigma_{\max}, 0}} \cdot \left[ 1, 1 + \frac{(1 + 2^{-80}) \epsilon}{n} \right]$$

에 속하게 됨을 알 수 있습니다. 여기서 $\epsilon, n$에 대해서는 설명하지 않았으나 FALCON의 context에서는 적당합니다.

위 식은 결국 $\sigma, \mu$에 대해서 독립이니, Isochronous임을 알 수 있습니다.

# Conclusion

지금까지 PQC에 대한 간단한 역사와 격자에 기반한 암호체계, 그리고 FALCON이 무엇인지 간략하게 알아보았습니다. FALCON 및 다른 여러 격자 기반 체계들은 Discrete Gaussian에서 Sampling을 해야했는데, 이 과정이 timing attack에 노출될 수 있기 때문에 Isochronous 한 알고리즘이 필요함을 알아보았습니다. 이를 해결한 논문이 [HPRR20]이며, 저자들이 문제를 어떻게 해결했는지도 알아보았습니다. 앞으로 소멤에 Post Quantum Cryptography에 대해서 자주 쓸 계획을 하고 있는데, 봐주시면 감사하겠습니다. 질문은 항상 제 개인 블로그 rkm0959.tistory.com 에서 받습니다. 감사합니다.
