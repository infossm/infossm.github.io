---
layout: post

title: "ZKP and Block Hash Oracles"

date: 2023-03-19

author: rkm0959

tags: [cryptography, blockchain]
---

이 내용은 ZK-School에서 멘토링한 [자료](https://github.com/rkm0959/rkm0959_presents/blob/main/ZKApplications.pdf)에 기반합니다. 


# ZKP의 목표

영지식 증명, Zero Knowledge Proof는 기본적으로 일련의 계산이 제대로 되었음을 간단하게 증명하는 것을 목표로 하고 있습니다. Zero Knowledge라는 부분도 중요하지만, 가장 핵심은 보통 Proof에 있습니다. 일단 Proof가 되어야지 Zero Knowledge를 이야기를 하겠죠? 그래서 대강 압축을 하자면, Alice가 $f(x)$라는 값을 알고 싶은데 이를 직접 계산하기에는 컴퓨팅 파워가 부족하다면, 그 컴퓨팅 파워가 있는 Bob이 $f(x)$를 직접 계산해서 $y = f(x)$라는 사실을 알아내고, Alice에게 단순히 $y$의 값을 넘기는 것이 아니라, 이 사실을 증명해내는 증명 $\pi$를 전달하는 겁니다. 이러면 Alice는 $\pi$를 검증하기만 하면 되며, 목표는 $\pi$의 검증이 $f$의 계산보다 간단하다는 것입니다. 

Privacy를 위해서 Zero Knowledge 성질을 활용하는 경우도 다수 있지만, 오늘의 이야기 주제는 기본적으로 Proof가 된다는 성질을 더 사용합니다. 이 경우, 어떤 연산을 $f$로 보고 무엇을 증명할 것인지를 파악하는 것이 중요하다고 볼 수 있습니다. 

블록체인에서는 EVM State에서 한 Transaction이 발생했을 때 다른 EVM State로 넘어가는 과정을 아예 $f$로 보고 이에 대한 증명을 제공하는 기술이 거의 최종판에 가깝다고 볼 수 있겠습니다. 이를 zkEVM이라고 하며, Polygon, Scroll 등 다양한 곳에서 이를 위해서 많은 개발과 연구를 하고 있습니다. 오늘은 이에 대한 이야기보다는 조금 더 작은 규모지만 여전히 중요한 문제를 다루도록 하겠습니다. 

# Block Hash Oracle 구축하기 

Block Hash는 기본적으로 대강 설명하자면 Block Header의 해시입니다. 이 Block Header에는 이전 블록의 해시, 블록의 timestamp, 당시 블록체인 상태에 대한 여러 값들이 (Merkle-Patricia Tree의 Root) 들어있는 등 매우 중요한 정보인데, 문제는 이 값을 on-chain에서 trustless 하게 저장하는 것이 쉬운 일이 아니라는 것입니다. 

가장 쉬운 방법으로는 blockhash opcode를 사용하는 것이지만, 과거 최근 256 블록에 대한 값 밖에 얻을 수 없어 그 한계점이 명확합니다. 매우 예전 블록에 대해서는 값을 얻을 수 없고, 지금 당장 사용하고 싶어도 무조건 256 블록에 한 번은 직접 컨트랙트를 호출해야 하기 때문입니다. 

이 문제를 해결하기 위해서, ZKP를 사용할 수 있습니다. 먼저 $N$번째 blockhash를 이미 알고 있다고 가정하고, $N-1$번째 blockhash를 증명하는 방법에 대해서 생각해봅시다. 기본적으로 $N$번째 blockhash는 그 block header의 해시이고, 그 block header 안에는 $N-1$번째 blockhash가 포함되어 있습니다. 그러므로,
- $N-1$ 번째 blockhash는 $h_{N-1}$이고,
- 적당한 다른 값들이 있어서 같이 잘 hash 하면 $h_N$이 나온다

라는 것을 증명하면 $h_{N-1}$에 대한 증명이 끝나겠습니다. 여기서 같이 잘 hash 한다는 것은, blockhash를 계산하는 공식에 맞게 데이터를 인코딩하고 hash 한다는 의미입니다. 이를 ZKP로 증명할 수 있으며, 이를 계속하면 증명이 완료가 됩니다. 

다만, 이 경우 수천만개에 달하는 많은 증명이 필요하고, 수천만개의 block hash를 전부 on-chain에 올려한다는 문제점이 있습니다. 이를 해결하기 위해서는, block hash들을 merkle tree에 넣어서 그 root를 on-chain에 올리는 방법을 생각해볼 수 있습니다. 이러면 사용자들은 merkle proof를 제공하면 되며, on-chain에서 저장해야 할 값의 개수를 줄일 수 있습니다. 

이를 위해서, 다음과 같은 그림을 그릴 수 있습니다. 세그먼트 트리마냥, 다음과 같은 재귀적인 각을 그려봅니다. 

$[s, e)$에 있는 범위의 blockhash들을 증명하고 그에 대한 merkle root를 증명된 상태로 얻고 싶다고 합시다. 
- $[s, m)$, $[m, e)$ 각각에 대해서 재귀적으로 증명합니다.
- 두 증명이 모두 verify 되는지 여부를 "증명할 것의 목록"에 추가합니다.
- 양쪽 merkle root를 가지고 새 merkle root를 계산합니다.
- $[s, m)$의 "마지막 blockhash"와 $[m, e)$의 "첫 번째 parenthash"가 같은지 확인합니다. 
- 위 세 개의 체크를 하나의 ZKP로 묶어서, 그 증명을 반환합니다.

위 재귀 과정에서 반환해야 하는 것은 "첫 parenthash", "마지막 blockhash", "merkle root", 그리고 "ZKP 증명"입니다. 물론 base case인 $e = s + 2$는 자명합니다. 

위 과정을 거치면 결국 증명 하나로 많은 양의 blockhash를 증명할 수 있습니다. 이러한 과정으로 blockhash oracle을 구축할 수 있고, 이런 것을 개발하는 팀에는 Axiom, Herodotus, Relic 등이 있습니다. 

한편, 이처럼 증명을 검증하는 과정 그 자체를 증명할 것으로 넣는 것은 ZKP에서 많이 사용되는 테크닉 중 하나입니다. 증명에서 필요한 체크의 일부를 뒤로 미루고, 마지막에 한꺼번에 확인하는 테크닉은 상당히 많이 사용되며, IVC/PCD 계열로 가도 기본적으로 많이 보이는 내용이라고 볼 수 있겠습니다. 

# Storage Proofs

Blockhash 안에 있는 값 중 중요한 값 하나는 바로 state root입니다. 이 값에는 각 주소의 nonce, balance, storage root, code hash가 들어있습니다. 이 storage root에는 다시 slot에 대응되는 value가 들어있습니다. 모든 구조체는 Merkle Patricia Tree로 구성되어 있어, 다음과 같은 그림을 그리는 게 가능해집니다. 
- Blockhash로부터 State Root 증명 
- State Root로부터 Storage Root 증명
- Storage Root로 부터 Storage Value 증명 

즉, 이를 위 내용과 종합하면 "특정 블록에서, 특정 주소의, 특정 슬롯에서" 어떤 값이 있었는지를 trustless 하게 얻을 수 있습니다. 이는 매우 강력한 결과로, 다양한 활용이 기대되는 결과입니다. 

많은 적용이 가능할 것으로 보이지만, 일단 가장 대표적인 사례를 설명하자면 바로 Uniswap V2의 TWAP이 있습니다. TWAP을 위해서 UniswapV2의 Pair 컨트랙트는 cumulative sum을 저장해두고 있는데, TWAP을 계산하려면 지금의 부분합도 필요하지만 과거의 부분합도 필요하다는 점이 문제입니다. 과거의 부분합을 가져오기 위한 방식으로, ZKP를 기반한 방식을 채택할 수 있겠습니다. 이러면 TWAP의 값도 trustless 하게 사용할 수 있습니다. 물론 TWAP 자체는 Multi-Block MEV 등의 문제로 여러모로 안전하다고 보기는 어렵기는 하지만요...


# On-Chain Randomness via VDFs

On-chain에서 랜덤 값을 생성하는 것은 매우 중요한 문제입니다. 보통 여기서 바라는 것은
- 예측이 불가능하고
- 편향되어 있지 않으며 
- 제대로 생성되었음을 검증할 수 있고
- 항상 랜덤 값을 받아올 수 있어야 함

입니다. 이를 기반으로 여러 솔루션을 비교해봅시다.

Option 1. 직접 생성해서 온체인으로 올린다.
- 일단 생성하는 사람을 단순히 믿어야 하는 점에서 실격입니다. 

Option 2. blockhash 사용
- block proposer에 의해서 어느 정도 편향이 가능합니다. 

Option 3. RANDAO 사용
- EIP4399를 기반으로 하여 사용이 가능합니다. 
- blockhash 보다는 괜찮지만 편향이 불가능하지는 않습니다. 

Option 4. Chainlink VRF
- 돈이 들지만 상당히 괜찮은 옵션입니다. 
- operator가 정상 작동을 해야 liveness가 보장됩니다.
- request censorship을 통한 bias가 가능합니다. 

위 네 옵션이 가진 크고 작은 문제를 하기 위해서, RANDAO 값에다가 VDF를 적용하는 방법을 생각할 수 있습니다. 

우선 RANDAO 값은 여러 block proposer에 의해서 섞이는 나름 랜덤한 값으로, blockhash 보다는 조작이 더 어려운 편입니다. 이 값 역시 block header 안에 들어있기 때문에, trustless 하게 복구하는 것이 가능합니다. 

VDF란, 다음 세 알고리즘으로 구성된 시스템입니다.
- Setup: Difficulty $t$와 Security Parameter $\lambda$를 가지고 key를 구축
- Eval: input $x$를 받고 output $y = f(x)$와 증명 $\pi$ 전달
- Verify: $x, y, \pi$를 받고 그 증명을 검증. 

기본적으로 ZKP와 비슷한 세팅인데, 여기서 중요한 점은 Sequentiality입니다. 즉, $(p, \sigma)$-sequential 하다는 것은 Eval을 $p(t)$개의 processor를 가지고 $\sigma(t)$ 시간 이내에 높은 확률로 제대로 계산할 수는 없다는 것입니다. 다르게 말하면, 일정 수준 위의 병렬화 계산이 어렵다는 것입니다. 즉, **무슨 수를 쓰더라도 무조건** $\sigma(t)$의 시간이 걸린다는 것입니다. 

이를 구축하기 위해서는 Hidden Order Group을 기반으로 한 여러 구축도 있지만, 여기서 사람들이 사용하는 것은 Incrementally Verifiable Computation 입니다. 이에 대해서 더 제대로 이해하기 위해, sequential function을 정의할 필요가 있습니다.

$(t, \epsilon)$-sequential function이란, $f(x)$를 $\text{poly}(\log t, \lambda)$개의 processor로 시간 $t$ 이내에 계산할 수 있지만, $\text{poly}(t, \lambda)$개의 processor로는 $(1 - \epsilon)t$ 시간내로 계산할 수 없는 함수를 말합니다. 정확하게는 $(1 - \epsilon)t$ 시간 내에 $f(x)$를 성공적으로 계산할 수 있을 확률이 무시할 수 있을 정도로 작습니다. 

이를 기반으로, iterative sequential function을 정의할 수 있습니다. $(t, \epsilon)$-sequential function을 $k$번 compose 한 함수가 $(kt, \epsilon)$-sequential function이라면, 이를 iterative sequential function이라고 합니다. 즉, $k$번 정직하게 반복 계산하는 것 외에는 더 좋은 방법이 없다면, iterative sequential function이라고 부릅니다. 

이제 $g$가 $(t, \epsilon)$-sequential function이라고 합시다. 그리고 $f$가 이를 $k$번 반복해서 나온 $(kt, \epsilon$)-sequential function이라고 합시다. 또한, 일련의 계산이 있을 때 그 ZKP 증명을 구축하는데 $\alpha$ 배의 시간이 걸린다고 합시다. 이 경우, 병렬화를 사용하여 $f$를 계산하면서 동시에 그 계산에 대한 ZKP를 구축할 수 있습니다.
- $kt/(\alpha + 1)$ 시간만큼 $f$를 계산하고 있습니다. 
- 그 후, 다른 processor를 가져와 지금까지 뽑은 진도에 대한 ZKP를 계산하게 합니다. 
- 이렇게 되면, 새 processor는 $kt$의 시간이 지났을 때 ZKP 계산이 끝납니다.
- 이제 남은 것은 $\alpha / (\alpha + 1)$ 비율의 계산입니다. 이를 재귀적으로 반복.

이렇게 되면 ZKP를 사용했으니 제대로 계산을 했다는 검증도 할 수 있고, $f$가 $(kt, \epsilon)$-sequential function이니 $f$를 계산하려면 무조건 $kt$의 시간을 태워야 합니다. 즉, 이를 기반으로 **시간을 충분히 태웠는지**를 검증할 수 있게 됩니다. 물론 검증하는 입장에서는 ZKP 몇 개만 검증하면 됩니다. 

이제 여기서 해야하는 것은 sequential function $g$를 구축하는 것입니다. 이에 대해서는 많은 연구가 이루어지고 있는데, 대표적인 예시인 MinRoot를 소개하자면 

$$(x_{i+1}, y_{i+1}) = ((x_i + y_i)^{(2p - 1) / 5}, x_i + i)$$

를 사용합니다. 모든 계산은 $\mathbb{F}_p$에서 이루어집니다. 이 경우, 보통 시작 상태 $S$와 끝 상태 $S'$이 있을 때, low-degree polynomial relation $G(S, S') = 0$이 성립하도록 구축합니다. 그 이유는, 이래야 SNARK의 계산이 간단해지기 때문입니다. 

이제 RANDAO의 값에 VDF를 적용한 값을 random 값으로 설정하면, RANDAO 값에 약간의 bias를 주려고 고민을 하더라도 그 결과가 VDF에 어떤 영향을 줄 지 시간 안에 계산을 할 수 없으므로, 그 random 값은 사실상 조작이 아예 불가능한 완전한 random 값이 됩니다. 게다가 RANDAO 값에 대한 증명이나 VDF에 대한 증명은 아무나 할 수 있으니, 특별한 권한 설정이나 중앙화가 필요없습니다. 



