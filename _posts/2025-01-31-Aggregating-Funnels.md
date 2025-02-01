---
layout: post
title: "Aggregated Funnels"
date: 2025-01-31
author: yhunroh
tags: [concurrent, parallel]
---

## Intro

저번 글에서 Software Combining이 무엇인지, 이를 이용해 fetch-and-add를 어떻게 구현할 수 있는지를 살펴보았다. 이번에는 fetch-and-add 연산을 더 빠르게 만들어주는 Aggregating Funnels 알고리즘을 알아보자.

Aggregating Funnels는 [PPoPP 2025](https://ppopp25.sigplan.org/)에서 발표될 [논문](https://arxiv.org/abs/2411.14420)에서 소개된 알고리즘이다. 본인은 알고리즘 설계 및 실험 수행에 주로 참여하였다. 실험 코드 또한 https://github.com/Diuven/aggregating-funnels 에 공개되어 있다.

Combining Funnels와 목표하는 바는 비슷하고, 둘 다 Software Combining의 접근을 취하지만, 작동 방식 및 성능은 상당히 다르다. 이 글에서는 알고리즘의 개요와 성능에 대해서 설명한다.

### Key Points & Brainstorming

[이전 포스팅](https://infossm.github.io/blog/2024/07/29/Software-Combining/)에 fetch-and-add의 atomic함 및 software combining의 기본 아이디어에 대해서는 설명하였으니, 익숙하지 않다면 확인해보자. 이 글에서 Aggregating Funnels / Combining Funnels는 모두 fetch-and-add를 구현하는 자료구조로 생각하며 기술하며, '연산' (operation) 이라고 지칭하는 것은 각 스레드에서 처리하는 x.fetch-and-add(v) 함수 콜 한번이라고 생각하면 된다.

이 논문에서 가장 흥미로운 점 중 하나는, 단순히 CPU에 내장된 fetch-and-add(xadd) instruction을 사용하는 것 보다 더 빠르게 fetch-and-add를 알고리즘으로 구동할 수 있다는 점이다. 필연적으로 각 연산들은 xadd instruction을 사용하고, 거기에 앞뒤로 (로컬) 연산들을 더 처리해야 하지만, 그럼에도 성능이 빠를 수 있다는 점이 다소 비직관적이다.

예를 들어, 이전에 살펴본 Combining Funnels의 경우에, 한 연산은 각 funnels의 레벨을 통과해야 하므로 대략 logP (혹은 설정한 레이어 개수)의 CAS연산을 해야 하고, (본인이 승자일 경우에) 마지막 연산을 하고 나서도 다른 스레드들의 연산 결과를 차례로 분배해줘야 한다. 그러니 단순히 하드웨어(CPU)에 구현된 xadd를 부르는 것 보다는 훨씬 오래 걸릴 것이라고 직관적으로 생각할 수 있다.

하지만, 만약 스레드의 수가 10000개라고 생각해보자. 우리는 여전히 변수 하나에 fetch-and-add를 할 것이다. 그렇다면 HW에 구현된 xadd를 사용하면 10000개의 스레드가 차례로, 하나씩 해당 메모리 위치에 접근하며 처리될 것이다. 한편 Combining Funnel을 사용하면, 많은 수의 스레드들이 Funnel 어딘가에서 겹쳐서, 결과적으로는 소수의 승자 스레드만 해당 메모리 영역에 접근하고, 많은 수의 스레드들은 자신의 메모리 영역에서 대기할 것이다. 즉, HW의 경우에는 병목이 하나의 메모리 영역에 연산을 할 수 있는 횟수였다면, CombFunnels의 경우에는 병목(혹은 시간복잡도)가 funnel의 구조에 따르게 되어, 복잡도가 변하게 된다.

결국 'software combining에 필요한 overhead' 와 '하나의 변수에 sequential하게 연산을 할 수 있는 속도', 그리고 총 스레드 수에 따라서 어느 쪽이 좋은지가 결정되게 된다. 즉, 어떤 CPU를 사용하느냐, combining 알고리즘이 얼마나 효율적이냐에 따라 성능이 달라지게 된다. 이전에는 로컬 (즉, shared memory에 적용되는 연산이 아니더라도) 단일 instruction이 너무 느려 추가 overhead가 있으면 절대 빨라질 수 없었을 수도 있지만, 최근의 CPU들에서는 로컬 instruction들의 속도는 빨라진 한편 shared memory instruction의 속도는 여전히 느려 software combining 접근이 더 빠를 수 있다.

이 논문에서는 fetch-and-add를 위한 효율적인 software combining 알고리즘를 고안하고, 결과적으로 많은 스레드 환경 하에서 HW 및 Combining Funnel보다 나은 성능을 달성한다.

## Aggregating Funnels 알고리즘

이 글에서는 편의상 가장 단순한 형태의 알고리즘만을 설명한다. 이 버전에서는 항상 양의 값만이 operand이며, 오버플로우는 일어나지 않고, aggregator의 수는 상수이다. 그렇지 않은 경우에 대해서는 논문을 참고하자.

AggFunnel 알고리즘은 overhead를 최소화하기 위해, 각 연산에서는 fetch-and-add는 최대 두번 사용하며, O(1/P)번 연산마다 한번 shared memory 변수에 write한다.

4개의 'Aggregator'를 만들고, 각 스레드들을 균일하게 분배하여 하나의 Aggregator는 P/4개의 스레드를 담당하도록 한다. 각 연산은 해당 aggregator에서 시작하고, 동시에 연산을 진행하는 스레드들과 함께 하나의 batch를 이룬다.

하나의 aggregator에서는 각 batch마다 하나의 스레드가 'delegate' 스레드 (대표 스레드) 로 선정되어, 그 스레드만 main 변수에 접근할 수 있다. 대표 스레드는 main 변수에 자신이 속한 batch의 연산들을 모두 모아 한번에 적용시키고, 그 결과를 batch의 다른 스레드들이 볼 수 있도록 shared memory에 저장한다. 대표 스레드가 아닌 다른 스레드들은 해당 변수를 보며 기다리고 있다가, 자신의 batch가 처리된 것이 확인되면 자신의 fetch-and-add 값을 재구성하여 반환한다.

아래는 의사코드이다. df가 음수인 경우도 고려하여 적혀 있지만, 양수일 때만 보아도 무관하다.

![](/assets/images/yhunroh/aggfunnel/2025-0201-01.png)

앞서 설명한 알고리즘이 fetch-and-add에 대해 효율적으로 작동하는 가장 주요한 이유는, fetch-and-add(x)는 여러 개의 연산이 하나의 값으로 표현되도록 합쳐질 수 있고, 중간 결과를 분해하여 저장할 수 있으며, 결과가 강증가하기 때문이다.

line 20에 있는 대표 스레드가 이때까지 쌓인 연산들을 한번에 모아 main에 적용시키는 로직에서, 만약 여러 개의 연산이 짧게 합쳐질 수 없는 queue.enqueue라거나, CAS같은 연산이었다고 한다면, main을 접근하는 횟수가 크게 차이나지 않았을 것이다.

line 16에 있는 aggregator의 value에 자신의 연산을 먼저 적용하는 로직에서, 만약 분해하여 저장할 수 없는 분기 함수, 혹은 commutative하지 않는 함수의 경우에는 중간 결과를 모아 main에 적용할 수 없었을 것이다.

line 17에 있는 자신의 batch가 처리되었는지 확인하는 루프에서, 만약 aggregator의 값이 강증가하지 않는 average 혹은 max함수였다고 한다면, 단순 비교로 진행 상태를 확인할 수 없어 언제 어떻게 자신의 값을 가져갈 수 있을지를 정할 수 없을 것이다.

반대로, 이러한 성질을 만족하는 다른 함수들에 대해서도 Aggregated Funnel 알고리즘을 적용할 수 있다.

## 실험 결과

실험 결과는 하드웨어 및 벤치마크에 따라 다소 차이가 있을 수 있으나, Aggregated Funnel이 기존의 HW fetch-and-add보다 최대 4배 이상 빠르게 처리할 수 있음을 볼 수 있다.

흥미로운 점은, read가 많은 벤치마크의 경우에는 Combining Funnel도 HW보다 더 나은 성능을 보였다는 점이다. 20년 이상 되고 오래동안 revise되지 않은 아이디어인 만큼, SW combining으로 HW보다 더 나은 성능을 달성할 수 있다는 점이 흥미롭다.

![](/assets/images/yhunroh/aggfunnel/2025-0201-02.png)

fetch-and-add의 주요한 사용례 중 하나는, DB 혹은 load balancing 시스템에서 사용하는 concurrent queue이다. 현재 sota concurrent queue는 LCRQ (https://dl.acm.org/doi/10.1145/2442516.2442527)로 알려져 있는데, 이러한 큐 내부에서는 사용할 메모리 위치를 정하기 위해 fetch-and-add 연산을 사용하며, 현재 성능의 병목은 fetch-and-add이다.

따라서, 이 HW fetch-and-add를 AggFunnel로 교체하면 더 빠른 성능을 얻을 수 있다. 본 논문에서 확인하기로는 200스레드 정도에서 최대 4-5배정도 빨라지는 것을 확인했다.

## Closing

본 논문에서 발견한 SW combining의 potential과 제안한 새로운 fetch-and-add 알고리즘의 아이디어는 흥미로우나, 실제 프로젝트에 사용되기 위해서는 더 다양한 하드웨어에서의 성능을 검증해야 하고, 하드웨어 및 workload에 따라 구조를 adatptive하게 바꿀 수 있는지에 대한 연구가 필요하다. 아무쪼록 첫 논문인 만큼 유의미한 영향을 만들어주길 기대해본다.
