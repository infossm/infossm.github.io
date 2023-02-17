---
layout: post
title: "Introduction of Threat Modeling"
author: antemrdm
date: 2023-01-17
tags: [security]
---

# Introduction

보안공학은 자연적이지 않은, 공격자에 의해 의도된 결함들을 다루는 학문이라는 점에서 다른 공학과 구분되며, 이러한 특징은 보안공학을 더욱 체계적이고 formal 하게 만드는 것 같습니다. 보안공학을 공부하면 다른 분야에도 적용하면 좋을 만한 체계적인 개념들이 많이 등장해서 흥미로운 학문이라고 생각합니다. 이번 글을 시작으로 보안공학에 대한 개요를 정리하고자 합니다. 이번 글은 개인적으로 보안공학하면 대표적으로 떠오르는 개념은 Threat Modeling에 대해 간략하게 설명하겠습니다.

# Overview

Threat Modeling을 Threat과 Modeling으로 나누어서 살펴보겠습니다.

먼저 Threat은 자산에 가해질 수 있는 잠재적인 위험을 의미합니다. 여기서 자산은 여러 종류가 있지만, 이 글에서는 하나의 시스템이라고 생각하겠습니다.

Modeling은 어떤 시스템을 단순화, 추상화한 것을 의미합니다. 추상화의 목적은 대상의 불필요한 정보를 날리고 필요한 정보만 남겨 단순화시키면서 대상에 더 집중할 수 있고 대상을 이해할 수 있도록 하는 것입니다.

합쳐보면, Threat Modeling은 시스템에 가해질 수 있는 잠재적인 위험을 더 쉽게 분석하고 이해할 수 있도록 시스템을 추상화한 것을 의미합니다.

<p align="center">
    <img src="/assets/images/antemrdm/threat_modeling/Untitled.png" width="500"/>
    <br>
    https://www.microsoft.com/en-us/securityengineering/sdl/threatmodeling
</p>


# Soundness

추상화를 하는 과정에서 너무 많은 정보를 날리면 abstraction gap이 발생할 수 있습니다. 그렇게 되면 실제로는 존재할 수 있는 위험이 추상화된 모델에서 발견되지 않을 수 있습니다. 이러한 경우 Threat Modeling을 하는 의미가 없어집니다. 따라서 추상화된 모델에서 임의의 취약점 A가 발견되지 않았으면, 실제 대상에서도 취약점 A가 발견되지 않아야 합니다. 이러한 성질을 soundness라고 하며, soundness는 Threat Modeling에서 매우 중요한 요소입니다.

# Threat Modeling이 필요한 이유

여기서 이러한 의문이 생길 수 있습니다. soundness를 해칠 가능성이 있음에도 불구하고 굳이 Threat을 분석하기 위해서 Modeling을 해야하냐는 것입니다. 보안은 99를 막아도 1이 뚫리면 안되는 만큼, Modeling되지 않은 실제 시스템 그대로를 대상으로 설정하여 Threat을 분석해야 할 것 같은데 왜 Threat Modeling을 해야 할까요?

그 이유는 보안공학의 속성과 관련이 있습니다. 보안공학이 일반적인 공학과 구분되는 가장 큰 특징은 자연적이지 (랜덤하지) 않은 의도적인 fault를 다룬다는 것입니다. 예를 들어 컴퓨터 공학에서는 주로 자연적으로 발생하는 서버의 결함이나 의도하지 않은 데이터 결함에 다루지만, 보안공학에서는 주로 악의적인 유저가 의도적으로 수행하는 공격에 대해 다룹니다. 따라서 보안공학에서는 공격자의 입장에서 공격자처럼 생각하는 것이 중요합니다.

하지만 공격자의 입장에서 시스템에 대한 공격을 모의로 수행해보기란 쉽지 않습니다. 체계적으로 수행하지 않는 한 고려하지 못하는 요소들이 다수 발생할 것이고, 범위가 너무 넓어 체계적으로 관리 및 수행하기가 어렵습니다.

이러한 문제를 해결하기 위해 나온 방법론이 Threat Modeling입니다. 정리하자면 체계적인 방법론으로 공격자처럼 생각해서 시스템에 가해질 수 있는 Threat을 파악하기 위해 Threat Modeling을 사용합니다.

# Requirement

Threat Modeling을 통해 시스템에 가해질 수 있는 Threat을 분석했다고 가정합시다. 시스템에 대한 가능한 Threat들을 분석만 한다면 의미가 전혀 없습니다. Threat을 분석하는 이유는 해당 Threat을 제거하여 시스템을 안전하게 설계 및 개발하기 위함이거나 시스템이 안전하다는 것을 증명하기 위함일 것입니다. 따라서 Threat을 분석한 후에는 당연히 해당 Threat을 완화 혹은 제거해야 할 것입니다. 이때 그를 위해 필요한 요구사항을 security requirement 라고 합니다. 즉, Threat Modeling에서는 시스템을 추상화하고, 추상화된 Model에 대한 Threat을 체계적으로 분석하는 것에서 그치지 않고 그를 완화 또는 제거하기 위핸 security requiremnet들을 도출해내는 것까지 수행합니다.

# Good Requirement

requirement는 실질적으로 Threat을 제거하는 수단이므로 Threat Modeling에서 상당히 중요한 요소입니다. 잘못된 requirement를 도출하면 Threat이 제대로 제거가 안될 것이고, 여러 Threat에 대해서 중복되는 requirement들이 존재할 수도 있습니다. 이러한 문제점을 방지하는 좋은 requirement가 가지는 여러 성질들에 대해 알아보겠습니다.

1. completeness
    - 완전성
    - 필요한 requirement가 모두 도출되는 것을 의미합니다.
    - 만약 현재의 requirement들만으로 제거하지 못하는 임의의 Threat이 있다면 이 requirement들은 completeness를 만족하지 않습니다.
2. correctness
    - 정확성
    - requirement가 그 기능을 정확하게 명시하는 것을 의미합니다.
    - 예를 들어서 “충분하게 안전한 비밀번호”라는 requirement는 정확하지 않습니다. 충분하게 안전하다는 것이 모호한 개념이기 때문입니다.
    - 따라서 “숫자, 영문자, 특수문자를 하나 이상씩 포함하는 10자리 이상 비밀번호”라고 requirement를 도출한다면 훨씬 correctness를 가질 수 있습니다.
3. feasibility
    - requirement가 시스템의 자원 하에 구현할 수 있어야 함을 의미합니다.
    - 만약 도출한 requirement를 대상 시스템에 구현할 수 없거나 시스템의 성능에 큰 영향을 미친다면 결코 좋은 requirement가 아닐 것입니다.
    - 이 경우 Threat을 제거하기 위한 다른 requirement를 도출해야 합니다.
4. unambiguity
    - requirement가 모호하지 않게 명시되어야 함을 의미합니다.
    - 이를 위해 일반적으로는 formal method를 사용합니다.
    - 예를 들어, “비밀번호를 암호화해야 한다.“는 requirement는 모호합니다.
    - 따라서 이는 “비밀번호를 AES-256으로 암호화해야 한다. “ 등의 더욱 명시적인 requirement로 변경되어야 합니다.
5. prioritization
    - requirement 간의 우선순위가 존재해야 함을 의미합니다.
    - 모든 requirement들을 시스템에 구현할 수 없거나 구현하기 힘든 상황에서 우선으로 적용해야 할 requirement를 선택하기 위해서 requirement들 간의 우선순위가 존재해야 합니다.
6. verifiability
    - 해당 requirement가 제대로 구현되었는지 검증할 수 있어야 함을 의미합니다.
    - 만약 requirement를 구현함으로써 Threat이 제거되었는지를 검증할 수 없다면 시간과 비용을 들여 requirement를 구현하기는 힘들 것입니다.
    - 또한 이론적으로는 완벽한 requirement라고 하더라고 실제로 구현상의 문제로 Threat이 제거가 안될 수 있기 때문에 verifiability를 만족하는 것이 중요합니다.

# Structured Threat Assessment Process

Threat Modeling이 현실에서 어떻게 쓰이는지 한 가지 예시인 Threat Assessment Process를 간략히 살펴보겠습니다. Threat Assessment는 Threat Modeling과 결합된 모의 해킹입니다. 모의 해킹에 Threat Modeling이 필요한 이유는 보안 검사를 체계적으로 진행하기 위함입니다. 실제로 보안팀은 시스템에 대한 보안 검사를 진행할 때 중요한 부분 또는 본인이 잘하는 부분 위주로 검사를 진행합니다. 그래서 같은 시스템에 대해서라도 서로 다른 보안팀이 분석을 한다면 그 결과가 서로 다를 수 있습니다.

이러면 한 보안팀에 보안 검사를 맡겼을 때 검사 결과가 좋다고 하더라도, 다른 보안팀에게 보안 검사를 맡겼을 때 검사 결과가 다를 수도 있기 때문에 검사 결과에 대한 신뢰도가 줄어듭니다. 그러므로 시스템이 안전함을 확인하기 위해서 여러 보안팀에서 여러 번의 보안 검사를 수행해야 되고, 이는 시간적으로 또 비용적으로 비효율적입니다. 더 나아가 몇 번이나 보안 검사를 수행해야 시스템이 안전하다고 안심할 수 있는지를 알 수 없습니다.

따라서 어떤 보안팀이 검사를 진행하더라도 같은 결과가 나올 수 있도록 체계적인 보안 검사가 필요합니다. 다시말해 누가하더라도 재현이 가능할 정도로 보안 검사를 체계적으로 수행해야 합니다. 이를 위해서 Threat Modeling이 모의 해킹에 결합되는 것입니다.

이렇게 보안 검사가 체계적이라면 다양한 장점이 존재합니다. 먼저 프로세스가 체계적이므로 대량 분석이 가능합니다. 또한 큰 시스템의 경우 여러 보안팀에서 공동으로 검사를 진행할 수 있어서 검사 가능한 시스템의 크기도 커지고 시간적인 효율도 높아집니다.

# Threat Modeling의 장점

지금까지 Threat Modeling이 무엇인지, 왜 필요한지, 실제로 어떻게 적용이 되는지에 대해 알아봤습니다. 그 과정에서 Threat Modeling으로 달성할 수 있는 여러 장점들을 직접 혹은 간접적으로 언급했지만 다시 정리해보겠습니다.

1. security requirement를 쉽게 도출할 수 있다.
    - 시스템을 추상화했기 때문에 분석할 대상의 크기와 복잡도가 낮아져서 Threat에 대한 조치를 쉽게 도출할 수 있습니다.
2. security requirement를 쉽게 이해할 수 있다.
    - 시스템을 추상화했기 때문에 시스템에 대해 자세히 몰라도 관계자들이 쉽게 requirement를 이해할 수 있습니다.
    - 실제로 보안을 잘 모르는 개발팀에게 시스템에 어떤 Threat이 있으며, 이 Threat을 제거하기 위해 어떤 requirement가 필요한지를 설명해야 합니다.
    - 이때 일반적으로 개발팀은 보안 지식이 충분하지 않을 수 있기 때문에 이 과정에서 불필요하게 많은 시간이 소모될 수 있습니다.
    - Threat Modeling을 사용하면 비교적 개발팀 뿐만 아니라 다른 관계자들에게 어떤 Threat이 있으며, 이 requirement가 얼마나 중요하고, 왜 필요한지 등을 쉽게 설명하고 전달할 수 있습니다.
3. 구현상 버그를 줄일 수 있다.
    - 체계적인 프로세스가 존재하며, 개발 이전 설계 단계에서부터 requirement를 분석하기 때문에 구현상에서 버그가 줄어듭니다.
4. 사람의 능력에 의존하지 않고 일정 수준 이상을 결과를 도출할 수 있다.
5. engineering approach를 사용하여 아래 3가지를 만족한다.
    1. predictable : 예측가능
    2. reliable : 신뢰가능
    3. scalable : 시스템의 크기가 커지더라도 적용할 수 있다.

# Threat Modeling의 특징

본격적으로 Threat Modeling에 대해 알아보기 전에 마지막으로 Threat Modeling이 가지는 특징들에 대해 살펴보겠습니다.

1. team sport이다.
    - Threat Modeling은 여러 사람들의 협업을 통해서 진행됩니다.
    - 만약 대상 시스템이 작아서 혼자의 힘으로 분석할 수 있는 경우에는 Threat Modeling이 큰 힘을 발휘하지 못합니다.
    - 하지만 Threat Modeling이 사용되는 경우는 주로 대상 시스템의 크기가 매우 커서 추상화가 필요하고 분석에 많은 사람들이 필요한 경우입니다.
    - 그래서 Threat Modeling은 주로 협업으로 이루어집니다.
    - 그렇기에 프로세스가 체계화되어 있어야 하며, 긴 의사소통 없이도 빠르게 의견 전달이 이루어질 수 있도록 다이어그램이 많이 활용됩니다.
2. living document이다.
    - 한번 Threat Modeling을 통해 도출된 문서가 고정되지 않습니다.
    - 해당 문서는 시스템 환경의 변화나 기술의 발전에 따라 지속적으로 업데이트된다.
    - 예를 들어 시스템의 os가 업데이트되었다던지, 새로운 기술의 도입으로 시스템이 변경되는 경우, 그에 따라 변화한 시스템에 대해서 Threat Modeling 결과도 지속적으로 업데이트되어야 합니다.
    - 이를 remediation 과정이라고 부르기도 합니다.

# Conclusion

이번 글에서는 Threat Modeling이 어떤 개념이고 왜 필요한지, 어떠한 장점과 특징이 있는지를 살펴보았습니다. Threat Modeling의 세부적인 과정이나 다양한 방법론들에 대해서는 다음 글에서 자세히 다루어보겠습니다.