---
layout: post

title: "Anatomy of a zkVM: Looking at SP1"

date: 2024-06-11

author: rkm0959

tags: [cryptography, blockchain, zero-knowledge]
---

# 소개

최근 Succinct Labs의 zkVM인 SP1을 자세히 볼 기회가 생겼습니다. zkVM은 ZKP를 통해서 하나의 프로그램 자체의 execution을 통째로 증명하는 것을 목표로 하고 있습니다. SP1의 경우에서는 RISC-V 프로그램을 대상으로 하고 있어, Rust을 구현한 후 컴파일하면 SP1이 바로 증명을 할 수 있게 됩니다. 이를 통해서, 복잡한 ZKP 서킷을 직접 작성하는 대신 Rust로 코드를 짜고, 이를 SP1으로 덮어서 ZKP를 사용할 수 있게 됩니다. 즉, zkEVM을 구현하기 위해서 EVM의 Rust 구현체를 가져다 쓰기만 하면 됩니다. 

이 글에서는 SP1의 구조에 대해서 매우 간략하게만 설명합니다. 

# Basic Setup

$p = 2^{31} - 2^{27} + 1$ BabyBear prime을 사용하는 PLONKish + FRI 세팅을 사용합니다. 

여러 증명을 하나로 묶기 위해서 전형적인 이진트리 방식의 recursion을 사용하고, 최종 증명은 크기를 줄이기 위해 Groth16으로 Wrap 합니다. 이러한 PLONKish + FRI + 이진트리 재귀 + Groth16 Wrapping은 현재 많은 팀들이 사용하는 인기있는 기술스택입니다. 작은 $p$를 사용하는 것 역시 인기있습니다. 

SP1의 서킷 작성은 Plonky3를 기반으로 되어있습니다. 특히, keccak256의 서킷은 Plonky3를 거의 그대로 사용합니다. 

# Proving Arbitrarily Large Programs

우선 임의의 크기를 갖는 프로그램을 증명하려면, 테이블 하나로는 부족합니다. 이를 위해서는 프로그램을 여러 개의 shard로 쪼개고, 각각의 shard를 증명한 다음, 이에 대한 증명을 하나로 합치는 과정이 필요합니다. 

이를 하려면 Fiat-Shamir 과정을 살짝 재조정할 필요가 있습니다. Permutation Argument를 위한 challenge를 sample 할 때, 각각의 shard가 모두 동일한 challenge를 사용해야 하기 때문입니다. 하지만 다행히도 이는 직관적인 풀이가 있는데, 단순히 모든 shard의 main trace commitment를 받은 후에 Fiat-Shamir로 challenge를 생성하면 되기 때문입니다. 즉, 각 shard의 증명을 검증하는 과정을 challeneger가 main trace 전부를 받은 상태에서 시작한다고 생각하면 됩니다. 이러면 permutation argument는 다 원하는대로 동일한 challenge로 진행되고, 나머지 STARK verification 과정은 각각의 shard에 알맞게 진행됩니다. 

또한, 각 shard 사이의 정보가 consistent 한지 확인하기 위해서, 각 shard 증명의 public value로 해당 shard의 시작하는 program counter와 끝나는 program counter를 제공합니다. 그러면 shard들의 program counter가 연결되어 있다는 것을 확인해서, shard 사이의 consistency를 확인할 수 있습니다. 

# Multi-Table Architecture 

SP1의 핵심 아이디어 중 하나는 각 테이블마다 연산 하나를 담당하게 하는 것입니다. 예를 들어, SHA256의 증명이 ZKP의 bottleneck이라면, SHA256만을 위해 잘 깎인 ZKP circuit 하나를 두고 이를 사용하게 합니다. bottleneck이 되는 부분에서 최적화를 하면 전체적으로 최적화가 잘 된다는 아이디어를 활용한 것이라고 볼 수 있겠습니다. 하지만 이를 사용하려면 문제 하나가 있는데, 바로 **테이블 간의 소통**이 필요하다는 것입니다. 메인 테이블이 SHA256 증명을 필요로 한다면, 그 사실을 SHA256를 위한 테이블에게 전달해야 합니다. 

이를 위해서, permutation argument를 사용합니다. 한 테이블에서 interaction을 **보내고**, 다른 쪽에서 interaction을 **받는** 형태의 구조를 생각하면, 결국 이 interaction들이 permutation을 이룸을 확인하면 됩니다. SP1은 memory, byte, program, alu, range, syscall 등 다양한 용도를 위해서 interaction을 사용하고, 이들이 permutation을 이룸을 확인하기 위해서 random linear combination trick과 logarithmic derivative를 이용합니다. 즉, 모든 interaction을 다 합쳐서 (보내고 받는 것의 Sign을 고려한 후)

$$\sum_{(\text{mult}, \text{value}) \in \text{Interaction}} \frac{\text{mult}}{\alpha + \text{RLC}(\text{value}, \beta)} = 0$$

가 성립함을 확인합니다. 이 $\alpha, \beta$가 앞서 언급했던 Fiat-Shamir로 뽑히는 값으로, $\mathbb{F}_{p^4}$의 원소입니다. 

$p$가 작고 $\mathbb{F}_p$의 characteristic이 작아, $p$개의 같은 interaction을 보내기만 할 수 있음에 주의해야 합니다. 이러한 low-characteristic 환경에서의 logarithmic derivative는 중요한 주제인 것으로 보입니다.  

이러한 특수한 table을 통해서 증명하는 연산들을 precompile이라 부르는데, SHA256, keccak, Elliptic Curve 연산 등 다양한 것들이 있습니다. 이러한 precompile은 언제든지 더 추가될 수 있습니다. 

# Memory and Program

Memory 구조는 유명한 memory-in-the-head 알고리즘으로 진행됩니다. Interaction 모델로 자연스럽게 구현할 수 있습니다. 결국, memory-in-the-head는 `PV = (address, prev_value, prev_time)`을 읽고 `CUR = (address, value, time)`을 쓸 때, 전체 permutation argument에 

$$- \frac{1}{\alpha + \text{RLC}(\text{PV}, \beta)} + \frac{1}{\alpha + \text{RLC}(\text{CUR}, \beta)}$$

를 추가하면 됩니다. 여기서 `time > prev_time`을 추가로 확인해야 합니다. 또한, initialization은 

$$\frac{1}{\alpha + \text{RLC}(\text{START}, \beta)}$$

로 추가되고, finalization은 

$$-\frac{1}{\alpha + \text{RLC}(\text{FINAL}, \beta)}$$

로 추가됩니다. 여기서 중요한 것은 각 memory address가 정확히 한 번 initialize 되고 finalize 되어야 한다는 것입니다. 이 조건이 만족하지 않는다면, memory-in-the-head 알고리즘의 기본 가정이 깨지게 됩니다. 

Memory는 여러 precompile과 CPU의 행동을 ZKP로 emulate 하는 과정에서 사용됩니다. 

memory가 read인 경우에는, `prev_value`와 `value`가 동일함을 확인해야 합니다. 

Program의 연산 과정을 증명할 때는, 항상 고정된 모두가 아는 값이 바로 program counter와 그에 해당하는 instruction, opcode selector가 되겠습니다. 즉, 특정 program counter에 어떤 instruction이 있고, immediate value가 무엇이며 다뤄야 하는 register는 어느 것인지 등이 되겠습니다. 이들은 고정된 공개값이므로 preprocessed column으로 두고 (즉, verifier도 이미 알고 있음) 이들을 메인 CPU로 옮깁니다. 다만, 경우에 따라 특정 program counter는 실행되지 않을 수 있으므로, 이 옮기는 과정은 main trace의 multiplicity를 갖고 이루어집니다. 이렇게 옮겨진 program counter와 instruction 정보들은 CPU 과정의 증명을 담당하는 ZKP 서킷으로 가게 되고, 여기서 다시 필요에 따라 각종 precompile들이나 memory argument를 호출하여 증명이 진행되는 것입니다. CPU 과정에서는 program counter, clock, memory load/store, branch, jump 등이 핸들링됩니다. 자세한 과정은 CpuChip을 보면 나옵니다. 

지금까지 간략하게 SP1의 구조를 살펴보았습니다. 