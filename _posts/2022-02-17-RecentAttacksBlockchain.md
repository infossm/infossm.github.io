---

layout: post

title:  "Recents Large Attacks on Decentralized Applications"

date: 2022-2-17

author: rkm0959

tags: [blockchain]

---

# 서론 

이 글에서는 rekt.news에 등재된 블록체인 해킹 사건 중 가장 규모가 큰 10개를 아주 간략하게 돌아봅니다. 각각의 기초적인 원리를 살펴보고, 이 10가지의 사건을 통해서 제가 블록체인 보안에 대해서 느낀 점을 정리했습니다.

# Lineup

https://rekt.news/leaderboard/ 에 있습니다. 여기에도 옮기면,

- Poly Network : $611M
- Wormhole : $326M
- BitMart : $196M
- Compound : $147M
- Vulcan Forged : $140M
- Cream Finance : $130M
- Badger : $120M
- Qubit Finance : $80M
- Ascendex : $77M
- EasyFi : $59M

합치면 대략 2조 3천억 정도 되는 거대한 돈입니다.

# Case 1 : Compromised Private Key

해당 케이스 : BitMart, Vulcan Forged, Ascendex, EasyFi

추가적인 사례 : Dego Finance, LCX, Crypto.com, Kucoin
- BitMart : https://rekt.news/bitmart-rekt/
- Vulcan Forged : https://rekt.news/vulcan-forged-rekt/
- Ascendex : https://rekt.news/ascendex-rekt/
- EasyFi : https://rekt.news/easyfi-rekt/ 
- Dego Finance : https://rekt.news/dego-finance-rekt/
- LCX : https://rekt.news/lcx-rekt/
- Crypto.com : https://rekt.news/cryptocom-rekt/
- Kucoin : https://rekt.news/epic-hack-homie/

블록체인 위 금융에서 가장 중요한 객체는 바로 비밀키입니다. 전통금융시장에서는 개인정보가 하나 정도 털려도 모든 자산을 잃는 것은 어느 정도 막을 수 있습니다. 그러나 블록체인에서는, 한 주소에 대응되는 비밀키를 털린다면 그 주소를 가지고 아무 트랜잭션이나 보낼 수 있고, 이는 비밀키를 탈취한 공격자가 모든 자산을 공격자의 주소로 탈취할 수 있음을 의미합니다. 이렇게 중요한 객체인만큼 비밀키는 사용자의 각별한 관리가 필요한 대상입니다.

Compromised Private Key는 공격당한 비밀키를 의미하며, 즉 위에 나열된 공격들은 모두 단순히 비밀키가 털려서 발생한 사건들입니다. 

많은 경우, Centralized Exchange에서 (중앙화된 거래소, 예를 들면 Binance, Upbit 등 마켓) 이러한 일이 발생합니다. 위 링크들 중에서도 다수가 이러한 중앙화 거래소에서 발생한 사건들입니다. 중앙화 거래소에서 암호화폐를 거래할 때는 실제 블록체인에서 암호화폐가 움직이지는 않습니다. 이러한 거래들은 모두 중앙화 거래소의 중앙화된 서버에서 이루어지고, 실제로 사용자가 암호화폐를 거래소에 입출금할때만 화폐가 블록체인 위에서 움직입니다. 이때, 중앙화 거래소는 보통 암호화폐를 관리하기 위해서 다음과 같은 지갑(=블록체인 위 주소)를 준비합니다.
- Hot Wallet : 실제로 암호화폐를 자주 입출금하는 지갑입니다. 고객이 돈을 입출금하려고 하면 여기로 입출금이 됩니다.
- Cold Wallet : 거대한 양의 암호화폐를 보관하는 지갑입니다. 필요하면 Hot Wallet으로 입출금을 하는데, executive들의 승인이 필요합니다. 

Hot Wallet에는 전체 돈의 일정량 미만의 돈이 있도록 하여, 안전성을 높이는 것이 일반적인 것으로 알고 있습니다. 이런 곳의 Hot Wallet의 비밀키가 털리면 어떤 일이 발생할지는 뻔하죠? 당연히 수많은 고객들의 돈이 날라가게 됩니다. 비밀키가 털리게 되는 경로는 다양할 것이며, 이를 모두 분석하여 소개하는 것은 이 글의 목적에 벗어납니다. 여기서는 이렇게 중요한 비밀키를 조금 더 안전하게 관리하는 기술에 대해서 간략하게 소개하겠습니다.

## Secret Sharing, Threshold ECDSA, Multi-Signature

Secret Sharing은, 말 그대로 비밀키를 여러 사람이서 공유하는 기술입니다. 즉, 비밀키를 "쪼개서" 가질 수 있도록 합니다. 가장 기본적인 Secret Sharing의 예시로는 Shamir's Secret Sharing이 있습니다. 이 기술을 사용하면, 비밀키를 $N$개로 쪼갠 후, 그 중 아무거나 $T$개 이상이 모이면 제대로 된 비밀키를 복구할 수 있고, $T-1$개 이하가 모이면 비밀키에 대한 아무 정보도 얻을 수 없도록 할 수 있습니다. Lagrange Interpolation을 이용하는데, 여기서는 이 정도로만 소개하고 자세한 기술적인 내용은 미래의 글로 넘기도록 하겠습니다. 나중에 이 분야에 대한 자세한 글을 쓸 생각입니다 :)

대부분의 블록체인은, 트랜잭션을 보낼 때 그 트랜잭션을 보낸 사람이 해당 주소의 비밀키를 소유한 사람임을 인증하기 위하여 암호학적인 서명을 추가하게 합니다. 이는 대부분 타원곡선 암호학에 기초한 서명으로, ECDSA를 이용합니다. Secret Sharing을 이용하면 마찬가지로 비밀키를 $N$개로 쪼개고, 각자의 서명 $T$개 이상이 모이면 제대로 된 서명을 얻을 수 있고, $T-1$개 이하로는 서명을 얻을 수 없습니다. 물론, 이 과정에서 서로가 가지고 있는 비밀키의 share는 비밀인 상태로 유지가 됩니다. 이렇게 한 메시지를 여러 명이서 서명하는 과정을 Multi-Signature라 부르고, 특히 충분한 사람이 모이면 서명이 가능한 구조를 Threshold Signature Scheme이라 부릅니다. 이러한 기술을 사용하면 어떤 점이 좋을까요?

비밀키를 분할해서 관리한다면, 그 중 일부가 털려도 안전하기 때문에 큰 문제가 없습니다. 또한, 비밀키를 쪼개지 않는다면, 비밀키를 소유한 대상이 나쁜 마음을 먹고 이상한 행동을 할 수 있기 때문에 이를 막기 위한 조치로도 강력한 효과를 발휘할 수 있습니다. 실제로 Cold Wallet이나 Decentralized Application의 Owner 급 권한에 대응되는 비밀키의 경우, Multi-Signature 기술을 이용하여 관리를 하는 것이 어느 정도 기본적인 관례입니다. 

비밀키를 안전하게 관리하는 것은 이미 거대한 산업 분야입니다. Fireblocks가 대표적인데, 개인적으로 매우 respect 하는 회사 중 하나입니다.

# Case 2 : Bridges

해당 케이스 : Poly Network, Wormhole, Qubit Finance 

추가적인 사례 : Anyswap(Multichain), Meter
- Poly Network : https://rekt.news/polynetwork-rekt/
- Wormhole : https://rekt.news/wormhole-rekt/
- Qubit Finance : https://rekt.news/qubit-rekt/
- Anyswap : https://rekt.news/anyswap-rekt/
- Anyswap : https://twitter.com/TalBeerySec/status/1485265215314804737
- Meter : https://rekt.news/meter-rekt/

## Bridge란 무엇인가 

Bridge란, 한 체인에 있는 자산을 다른 체인으로 보내는 프로토콜입니다. 구현하는 방법은 다양하지만, 기본적인 구조는
- 한 체인에서 일정 자산을 lock 또는 burn 하는 트랜잭션을 보냅니다.
- 이를 지켜보고 있는 주소에서 다른 체인에서 해당 자산만큼을 발행하는 트랜잭션을 보냅니다 

즉, 제가 3 ETH를 이더리움 체인에서 솔라나 체인으로 보내고 싶다면, 
- 3 ETH를 이더리움에서 lock 또는 burn 하는 트랜잭션을 보내고, 이것이 채굴됨
- 이 상황을 지켜보고 있는 주소에서, 솔라나에서 ETH 발행하는 트랜잭션을 보냄

이 이루어집니다. 이런 구조에서 생길 수 있는 자연스러운 질문은

- 솔라나에서 ETH 발행을 어떻게 하는가?
  - 이는 진짜 ETH를 발행한다기보단, ETH와 1:1 대응이 되는 새로운 토큰을 발행한다고 보시면 됩니다. 
  - 예를 들어, Wormhole의 경우 ETH를 보내면 whETH (wormhole ETH) 가 발행됩니다. 
- 누가 트랜잭션을 지켜보고 있는가?
  - 매우 중요한 문제입니다. 기본적으로는 이를 지켜보고 있는 사람들 역시 하나의 네트워크를 이룹니다.
  - 여러 사람들이 양쪽 체인들의 상황을 지켜보고 Consensus에 도달합니다.
  - 많은 사람들이 합의에 성공하면, Threshold ECDSA 등을 사용하여 목적지 체인에서 트랜잭션을 보냅니다.
  
즉, 조금 더 제대로 과정을 설명해보면
- 이더리움 체인에서 사용자가 3 ETH를 X라는 주소로 옮김
- X라는 주소는 여러 사람들의 네트워크에서 Threshold ECDSA로 관리되고 있음
- 이 네트워크에서는 X에서 일어난 트랜잭션들을 관찰하고 이에 대한 Consensus에 도달함
- 네트워크에서 X에 3 ETH가 들어온 것을 확인함
- 솔라나 체인에서 X가 사용자 주소에 3 ETH에 대응되는 토큰을 발행해서 보냄 

물론, 솔라나에서 이 3 ETH에 대응되는 토큰을 지불해서 다시 이더리움 체인에서 3 ETH를 돌려받는 것도 가능합니다. 자세한 구현은 Wormhole, Multichain을 참고하세요.


## Vitalik Buterin on Multi-chain vs Cross-chain

https://old.reddit.com/r/ethereum/comments/rwojtk/ama_we_are_the_efs_research_team_pt_7_07_january/hrngyk8/

이더리움의 창시자 Vitalik Buterin은 최근 1월 브릿지에 대한 다음과 같은 입장을 펼쳤습니다. 

- 브릿지는 근본적인 보안 한계가 있다
- 여러 체인이 공존하는 환경에 (multi-chain) 대해서는 긍정적으로 보지만, 여러 체인이 서로 상호작용하는 환경에 (cross-chain) 대해서는 부정적이다

기본적인 근거는, 51% 공격이 성공해질 때 발생하는 위험이 브릿지가 있으면 폭발적으로 증가한다는 것입니다. 51% 공격, 즉 computing power의 절반을 공격자가 가져갔을 때 발생하는 공격을 생각해봅시다. 이 경우, 브릿지가 없을 때 공격자는
- transaction을 censor 할 수 있으나, consistent한 상태를 유지함

즉, 남의 이더리움을 뺏는 행위는 불가능합니다. 

브릿지를 쓰는 경우는 어떨까요? 앞선 3 ETH를 이더리움 체인에서 솔라나 체인으로 보내는 과정을 생각해봅시다. 만약 솔라나가 51% 공격을 당해서 X가 사용자 주소에 3 ETH에 대응되는 토큰을 발행하지 않았다면, 사용자는 3 ETH를 이더리움 체인에서 지불하고 아무것도 받지 못합니다. 결국 51% 공격으로 사람에게 금전적인 피해를 입힐 수 있고, 체인 개수가 많아지고 서로 복잡하게 얽히기 시작하면 더욱 상황이 복잡해질 것입니다. 

## Bridge여서 가능했던 공격 vs Bridge임과 무관하게 발생한 공격

Qubit Finance Bridge의 공격, Meter Bridge의 공격, Multichain의 최근 공격은 모두 Smart Contract 상에서 로직 오류 및 취약점이 있어 공격당한 경우로, 근본적으로 Bridge여서 발생한 문제는 아닙니다. 즉, Case 3에 소속되는 공격이라고도 볼 수 있습니다. 다만, 최근 Bridge에서 발생한 공격의 양이 급증하고 있고, Vitalik의 Bridge에 대한 경고가 각광받는 상황이라 그런지 이러한 공격도 더 조명받고 있습니다. 세 공격 모두 다 기술적으로 어려운 공격은 아니었으니, 위 rekt news 링크를 통해 충분히 원인과 공격 과정을 공부할 수 있을 것이라고 생각됩니다. 

Poly Network, Wormhole, Multichain의 작년 7월 공격은 Bridge여서 가능했던 공격이라고 볼 수 있습니다. Bridge의 핵심 컴포넌트 중 하나는 양쪽 체인의 상태를 관찰하는 사람들의 네트워크입니다. 이를 MPC Network라고 자주 부르는데, 이 컴포넌트의 목표는 결국 시작 체인의 트랜잭션을 잘 관찰한 뒤 도착 체인에 적절한 Signature를 Threshold ECDSA를 통해 제공하는 것입니다. 결국 앞선 예시에서 주소 X에 대응되는 Private Key가 가장 중요한 것이고, 이를 쪼개서 관리하는 네트워크가 MPC Network가 되는 거죠. 이 문단에서 설명하는 세 공격은 전부 이 Private Key에 대한 공격입니다.

작년 7월의 Multichain 공격은 ECDSA에서 동일한 nonce가 2번 사용되면서 private key recovery attack이 가능하게 되어 터졌습니다. 이는 Playstation 3 해킹에도 사용된 전형적인 공격입니다.

Poly Network와 Wormhole, 합쳐서 1조 가까운 돈이 사라진 두 공격에서는 "Signature를 검증하기 위해 사용해야 하는 Public Key를 다른 것으로 바꿔쳐서" Private Key에 대한 공격을 진행했습니다. 즉, Private Key를 X가 아닌 다른 공격자가 이미 알고 있는 Private Key로 바꿔친 것이죠. 이렇게 되면 MPC Network의 Signature가 필요하지 않고, 단순히 공격자가 Signature를 생성해서 보내면 됩니다. 두 공격의 디테일은 어느 정도 난이도가 있기 때문에, 이 정도로만 이해해도 충분한 것 같습니다. Poly Network의 경우에는 4byte sighash에 대해 알고 있어야 하며, Wormhole의 경우에는 Solana의 세부 구조를 잘 이해하고 있어야 합니다. 첨언하자면, Poly Network의 경우에는 해커가 돈을 전부 다 돌려줘 화이트해커로 역사에 남았고, Wormhole의 경우에는 해커가 돈을 가져갔고 Wormhole에 투자한 (정확하게는 개발사와 연관이 깊은) Jump Crypto에서 돈을 배상해준 것으로 알려져 있습니다. 

# Case 3 : Smart Contract Vulnerabilities

해당 케이스 : Compound, Cream Finance

추가적인 사례 : Kleva, Cream Finance, 기타 정말 많음
- Compound : https://rekt.news/compound-rekt/
- Compound : https://rekt.news/overcompensated/
- Cream Finance : https://rekt.news/cream-rekt-2/
- Kleva : https://docs.google.com/document/d/1X1Mvs6bg5cuTfCnKp-cDDwGOB-snhq4Z3-PozRXTUNs/
- Cream Finance : https://rekt.news/cream-rekt/
  
전형적인 Smart Contract의 Vulnerability를 다 모아놓은 것이 3번입니다. 특히, Cream Finance 공격의 경우, 이전 소멤 글에서 언급한 (https://www.secmem.org/blog/2021/11/20/SolidityVuln/) 공격으로 적은 담보로 큰 돈을 빌려서 이익을 보는 형태의 공격입니다. 위 글에서 Smart Contract의 대표적인 취약점들 일부를 소개했으니 참고하시면 좋을 것 같습니다. 별개로 앞으로 소멤 블로그에 블록체인 보안과 관련된 글을 최대한 자주 쓸 예정이니 참고해주시면 감사하겠습니다. 여기서는 Compound와 최근 Klaytn 체인에서 발생한 Kleva 사건에 대해서 매우 간략히 설명하겠습니다.

두 사건 모두 "공격자"는 없고, 코드의 오류로 단순 사용자들이 자연스럽게 코드를 공격하게 된 사건이라는 공통점이 있습니다. 

## Compound Finance

Compound Finance는 사용자들에게 COMP라는 Governance Token을 보상으로 분배합니다. 이때, 이를 분배하는 로직을 Governance Proposal 62 (https://compound.finance/governance/proposals/62)에서 수정했는데, 이 코드의 로직에 버그가 있어 문제가 발생했습니다. 문제는 단순한 부등호 실수로, >= 여야 하는 부분을 >로 작성하여 문제가 발생했습니다. 이로 인해 COMP가 지나치게 과하게 분배되었고, 잘못된 수량의 COMP를 분배받은 이용자들이 이를 시장에 팔면서 (소위 "던지면서") 문제가 커졌습니다. 사용자들의 돈이 날라간 사례는 아니지만, COMP를 소유하고 있던 사람들은 COMP의 가격이 폭락하면서 피해를 보기 때문입니다. 사건 발생 이후 Compound 직원이 잘못 받은 COMP를 되돌려주지 않으면 신상을 확보하겠다는 식의 발언을 해서 큰 논란이 되기도 했는데, 자세한 이야기와 분석은 위 rekt news의 글을 읽어보시는 것을 정말 추천드립니다. 

## Kleva Protocol

Kleva Protocol은 사실상 Alpaca Finance의 Klaytn 체인 위의 fork라고 볼 수 있습니다. 한국 회사에서 만든 것이기 때문에 말을 하기가 조금 조심스러운데, Theori에서 작성한 위 구글 독스 다큐멘트가 크게 도움이 됩니다. (저도 약간은 기여했습니다 ㅎㅎㅎ) 결론을 말하자면, Lending Market과 유사한 구조가 있는데, 여기서 예치한 사람에게 이자가 제대로 지급되지 않는 버그가 있었습니다. 이를 수정하기 위해서 개발진이 코드 업데이트를 했는데 (Proxy Pattern 사용) 이때 개발진에서 이자율을 10^12배 증가시켜버리는 어마어마한 실수를 저질러버려서 순간적으로 예치한 돈을 뺀 사람들이 어마어마한 양의 돈을 받게 되었습니다. 이때 실제로 빠진 돈의 양이 대강 $50M 정도 됩니다. 다행히도 이 사건은 잘 해결된 것으로 알고 있습니다. 


# Case 4 : Attack on Web2, not Web3

해당 케이스 : Badger

추가적인 사례 : Klayswap
- Badger : https://rekt.news/badger-rekt/
- Klayswap : https://medium.com/s2wblog/post-mortem-of-klayswap-incident-through-bgp-hijacking-898f26727d66 

제가 전통적인 웹, Web2에 대한 보안에 대해서는 잘 모르기 때문에 자세한 설명은 불가능하지만, 간략하게 설명하겠습니다.

이 형태의 공격에서는 블록체인 위의 대상을 공격하는 것이 아니라, 사용자 UI를 공격하는 형태로 진행됩니다. 
개발자가 아닌 일반적인 DeFi 사용자들이 인터넷 상의 UI를 통해서 거래를 하는 경우, 보통 다음과 같은 순서를 거칩니다.
- Metamask 등 지갑을 서비스에 연결
- 특정 행동을 (자금 예치, 회수, 거래 등) 하기 위한 데이터 입력 및 버튼 조작
- Metamask가 실제로 보낼 트랜잭션을 사용자에게 보여주면서 확인 요청
- 사용자가 이를 확인함
- Metamask가 서명 및 블록체인에 보내는 과정을 전부 완료하면서 끝 

이때, 웹사이트 자체를 해킹해서 웹사이트가 Metamask에게 이상한 트랜잭션을 보낼 것을 요구하도록 설계하면 어떻게 될까요? 즉, 정상적인 트랜잭션이 X고, 공격자에게 자산을 바치는 나쁜 트랜잭션을 Y라고 합시다. 정상적인 상황에서는 웹사이트가 사용자의 입력을 받고 Metamask에게 X를 보낼 것을 요구하게 됩니다. Metamask는 X를 사용자에게 보여주면서 승인을 요청하고, 승인이 되면 이를 블록체인으로 보냅니다. 그런데 공격이 되면 웹사이트 자체가 Metamask에게 Y를 보낼 것을 요구하고, Metamask는 Y를 사용자에게 보여줍니다. 만약 사용자가 별 제대로 된 확인 과정 없이 승인을 누른다면, 이 Y는 서명이 되어 블록체인으로 옮겨지고 실행되어 공격이 완료가 됩니다. 실제로 Badger, Klayswap 모두 이러한 형태의 공격입니다.

이 문제를 해결하는 방법은 항상 Metamask의 트랜잭션 설명을 읽는 것이고, 기술력이 된다면 최대한 코드를 써서 블록체인과 interact 하는 것입니다.

# 결론 및 개인적인 생각

지금까지 rekt에 등재된 역사상 최대 규모의 블록체인 해킹 Top 10을 알아보았습니다. 참고로 Bitfinex 해킹도 나름 유명한데, 최근 미국 정부에서 공격자를 잡아서 수 조원에 달하는 비트코인을 복구한 사건도 있으니 알아보시면 좋을 것 같습니다. 이 글을 쓰면서 든 개인적인 생각은

- Private Key 관리와 이에 대한 기술이 정말 중요하다는 점
- Bridge는 항상 위험하다는 점.
  - 기본적으로 Bridge는 서로 연결할 생각이 없었던 체인 두 개를 연결하는 거라 위험성을 동반할 수 밖에 없습니다. Cosmos의 IBC 같은 경우는 애초에 연결할 생각이 있었으니 다른 Bridge에 비해서는 안전하다고 볼 수 있겠죠. 쉽지 않습니다...
  - MPC Network의 또 다른 문제점 중 하나가, 노드의 수가 굉장히 적다는 점입니다. Multichain의 MPC Network 노드 개수는 30~40개, Wormhole의 Guardian (MPC Network 노드 개수) 수는 대략 20개 정도입니다. 조금 더 노드 개수가 많아져야 탈중앙화된 Bridge가 될 수 있겠죠? MPC에 대한 더욱 많은 연구와 투자가 필요해보입니다.
- 생각보다 전형적인 Smart Contract Vulnerability는 해킹 규모가 크지 않다는 점
  - 물론, 사례의 수 자체는 여기가 제일 많으니 무시할 수 없습니다.
  - Solidity 보안은 여전히 중요하고, 다른 블록체인에서의 보안도 뜨지 않을까 예상.
- Web2 등 다른 전통적인 보안도 점점 더 중요해진다는 점
  - 돈이 어마어마하게 모이고, 공격자도 똑똑해지는 만큼 정말 다양한 attack vector를 고려해야하는 시점이 온 것 같습니다. 보안도 Smart Contract의 범위를 넘어서서 더 넓게 하는 게 맞다고 봅니다. 돈이 직접 왔다갔다하는 곳인 만큼 더 조심해야겠죠.

이것으로 글을 마치겠습니다. 