---
layout: post
title: "DDD Aggregate Pattern"
date: 2020-02-19 23:59
author: taeguk
tags: [DDD, Aggregate, Actor, Event Sourcing, CQRS, Microservice, Design Pattern, Software Design]
---

오늘은 제가 가장 좋아하는 소프트웨어 설계 기법인 **Aggregate Pattern** 에 대해서 소개해드리겠습니다!

## Aggregate Pattern 이란?

Aggregate Pattern 은 [Eric Evans 의 Domain-Driven Design](https://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215) 에서 소개된 설계 패턴으로써 아주 강력하고 scalable 한 설계 지침을 제공합니다. \
Aggregate 를 제대로 설명하기 위해서는 Entity 등과 같은 DDD 의 다른 개념들도 같이 설명이 필요한데요. 
이 글의 목적이 DDD 가 아니고 DDD 및 Aggregate 에 대한 개념적인 설명들은 인터넷에 많이 있으므로 이 글에서는 Aggregate Pattern 에 대한 통찰들을 전달하는데 집중하겠습니다. \
Aggregate 를 쉽게 말하자면 그냥 이름 그대로 연관 객체들의 묶음입니다. 
즉, Aggregate 를 설계하는 것은 서로 연관된 객체들을 하나의 Aggregate 안으로 묶는다는 것이죠. 
여기까지 말하면 사실 Aggregate 는 별로 의미가 없습니다. Aggregate 는 **몇 가지 특징 및 제약사항들**을 가짐으로써 힘을 발휘하게 됩니다.
- Aggregate 는 내부의 도메인 객체들을 완전히 독점 소유합니다. 이게 무슨 말이냐면, 객체들은 오직 하나의 Aggregate 에만 속하게 되고 객체들의 lifetime 은 속해있는 Aggregate 의 lifetime 에 제한된다는 것입니다. (Aggregate 가 삭제되면 내부의 객체들도 같이 삭제됩니다.)
- Aggregate 는 항상 내부 객체들의 불변성을 강제해야만 합니다. 무슨 말인고 하니 Aggregate 내부는 항상 비지니스 룰에 부합하는 상태를 유지해야한다는 뜻입니다.
- **Aggregate 각각이 transaction boundary 가 됩니다.** 즉 하나의 트랜잭션에서는 하나의 Aggregate 만 변경할 수 있습니다. 여러 개의 Aggregate 들을 한 트랜잭션으로 변경하는 것은 불가능합니다.

Aggregate 를 설계할 때는 위의 제약사항을 만족하도록 설계해야하고 이러한 제약사항이 Aggregate 를 매우 강력하고 확장성있게 만들어줍니다.

## Aggregate Pattern 이 강력한 이유

그러면 위에서 설명한 Aggregate 의 특징이자 제약사항들이 어떻게 Aggregate 를 위대하게 만드는지 설명해보겠습니다. 
위의 제약사항들을 한마디로 말하면 **"Aggregate 각각을 독립된 데이터 처리 단위로 만들어라"** 입니다.
- Aggregate 각각이 내부의 객체들을 완전히 독점 소유하고 있으므로, 어떤 Aggregate 내의 객체들이 수정되더라도 다른 Aggregate 들은 전혀 영향을 받지 않습니다.
- Aggregate 각각이 트랜잭션 경계가 되고 독립적인 불변성을 가지기때문에, Aggregate 각각이 독립적으로 연산을 처리할 수 있습니다.

소프트웨어 설계의 핵심은 복잡도를 관리하는 것이라고 생각하는데요. 
그런 측면에서 Aggregate 각각이 서로에 대해 완전히 분리되어 독립적인 데이터 처리 단위가 된다는 것은 **복잡도를 매우 낮춰주는 효과**를 가져옵니다.

또다른 중요한 포인트는 **소프트웨어의 동시성이 한눈에 보인다**는 점입니다. \
Aggregate 각각이 독립적으로 연산을 수행하고 각자의 상태를 변경시킬 수 있으므로, Aggregate 들은 모두 병렬적으로 연산을 수행할 수 있습니다. 
즉, **Aggregate 가 곧 병렬 처리의 단위**가 되기 때문에 Aggregate 를 설계한다는 것은 곧 소프트웨어의 병렬 처리 단위를 설계하는 것과 같은 의미가 됩니다. 
Aggregate 를 잘게 쪼개서 잘 설계하는 것은 곧 동시성의 극대화를 가져오고, Aggregate 를 너무 크게 설계하거나 잘못 설계하면 이는 곧 성능상의 문제를 가져오게 됩니다. 
즉, **애초에 설계단계에서 소프트웨어 성능을 같이 고려**하게 됩니다. \
참고로, 기본적으로 Aggregate 는 내부 상태에 대한 불변성 검증 및 유지를 위해서 Aggregate 하나는 1개의 명령만 동시에 처리할 수 있습니다. 
하지만 경우에따라 optimistic locking 을 사용할 수도 있고 크리티컬하지 않은 명령들에 대해서는 그냥 lock 을 사용하지 않는 식으로도 구현할 수 있기 때문에 필요에 따라 동시성을 좀 더 끌어올릴 수도 있습니다.

전통적인 stateless 서버 어플리케이션같은 경우 그냥 요청이 들어오는데로 DB 에 트랜잭션을 퍼붓게 되고 DB 가 알아서 트랜잭션을 수행하고 동시성을 컨트롤하게 됩니다. 
서버 코드 설계상으로 동시성이 명확하게 드러나지 않아서 DB 스키마, SQL 문을 보고서 병목과 동시성을 파악해야하는데 쉽지 않습니다. (DB 가 어떤 방식으로 locking 을 하는 지를 알아야합니다.) 
결국 성능 테스트를 해보거나 서비스 운영을 하면서 병목이 되는 부분을 찾아내 DB 구조와 SQL 을 최적화한다던지 하는 방식이 될 가능성이 높은거죠. \
반면에 Aggregate Pattern 을 사용하면 Aggregate 구조를 보면 동시성이 한눈에 보이고 어디가 병목이 될지 쉽게 파악할 수 있습니다. 
즉, Aggregate Pattern 을 활용하면 암시적으로 표현되어있고 복잡하게 꼬여있는 DB 상의 동시성을 명시적이고 직관적인 형태 (Aggregate) 로 끌어올린다고 말할 수 있겠습니다. \
물론 Aggregate 를 실제로 DB 에 persistence 하는 로직에 따라서 DB contention 이 발생해 예상치않은 동시성 저하가 있을 수도 있지만, 만약 persistence 에 있어서 이벤트소싱을 사용한다면 이러한 DB contention 또한 회피할 수 있겠죠? ㅎㅎ

그리고 Aggregate 가 서로에 대해 독립적이라는 점이 **상당한 유연성**을 가져오게 되는데 이는 **소프트웨어 scalability 및 분산 환경에서 있어서 상당한 이점**이 됩니다. 
무궁무진한 가능성들을 생각해볼 수 있는데 그중에 몇 가지를 예시로 들면 다음과 같습니다.
- Aggregate 마다 서로 다른 DB 를 사용할 수 있습니다. 물론 Aggregate persistence 로직에 따라 불가능할 수도 있지만, 각 aggregate 마다 각자 다른 최적의 DB 를 선택하여 사용할 수 있습니다. 이로 인해 DB 분산도 용이해지겠죠.
- 동일한 타입의 Aggregate 들을 분산 환경에 샤딩할 수 있습니다.
- 마이크로서비스 아키텍쳐에 적합합니다. (현실적으로 말은 안되지만) 극단적으로 한 개의 Aggregate 를 한 개의 마이크로서비스에 매핑할 수도 있습니다.

## Aggregate 설계는 어렵다

여기까지 읽으셨다면 아마도 Aggregate 의 매력에 깊은 감명을 받으셨을 것입니다. (제발 그랬기를..) \
Aggregate 가 현대 소프트웨어 복잡도를 해결하고 병렬/분산화를 쉽게 해주는 silver bullet 이면 좋겠지만 아쉽게도 단점은 있습니다. \
Aggregate Pattern 의 단점은 바로 설계가 너무 어렵다는 것입니다. \
위에서도 설명했듯이 Aggregate Pattern 이 매우 강력한 이유는 Aggregate 각각이 독립적인 데이터 처리 단위가 되기 때문입니다. 
하지만 문제 역시 거기에서 나옵니다. "Aggregate 각각을 독립적인 데이터 처리 단위로 만드는 것" 이 많은 고민이 필요하고 어려운 작업입니다. \
특정 요구사항을 Aggregate 로 올바르게 모델링할 수 만 있다면 Aggregate Pattern 은 아주 큰 이점을 가져옵니다. \
하지만 Aggregate 로 모델링하기 쉽지 않은 경우들이 있고 만약 잘못된 모델링을 하게 될 경우 오히려 안 좋은 결과를 가져올 수 있습니다.

예시를 통해 설명해보도록 하겠습니다. 
간단한 SNS 서비스를 Aggregate Pattern 을 바탕으로 만든다고 합시다. \
일단 유저 1명을 User 라는 Aggregate 1개로 모델링했다고 합시다. 
고객 1명이 여러 개의 디바이스를 이용해 동시다발적으로 명령을 요청하진 않을 것이므로 고객 1명에 대한 요청을 Aggregate 하나가 처리하더라도 동시성에는 문제가 없습니다. 
(하지만 단순히 예시일뿐 현실적으로 유저 1명에 대한 모든 정보를 Aggregate 1개에 몰아넣는 것은 좋지 않습니다. 동시성을 떠나서 DB 로 부터 Aggregate 를 로딩하는데 걸리는 시간이 오래걸리는 등의 문제가 있기 때문입니다.) 

1) 이 상황에서 쉽게 구현할 수 있는 요구사항으로는 유저가 자기의 프로필 사진을 수정하려는 경우가 있습니다. 그냥 User Aggregate 에 프로필 사진 수정을 요청하므로써 쉽게 구현할 수 있습니다.

2) 이것보다 약간 까다로운 요구사항은 다른 유저의 프로필 사진을 조회하는 기능입니다. 
A 유저가 B 유저의 프로필사진을 조회하기 위해서는 B 유저의 Aggregate 에 조회 요청을 보내야합니다. 
하지만 이렇게 할 경우 동시성에 문제가 생길 수 있습니다. 
B 유저가 인기가 많아서 많은 유저들이 B 유저의 프로필 사진을 조회하려고 한다면 B 유저의 Aggregate 에 부하가 심해지게 됩니다. \
따라서 Aggregate Pattern 을 사용할 경우 보통 Read Model 을 구축하는 식으로 해당 문제를 해결하게 됩니다. 
상태를 변경시키는 요청만 Aggregate 가 처리하고 상태를 조회하는 요청은 별도의 읽기 전용 Read Model 을 통해 처리하게 됩니다. 
Aggregate 가 가지고 있는 데이터는 eventual consistent 하게 Read Model 에 반영되게 됩니다. 
이런 방법을 **CQRS** 라고 부릅니다.

3) SNS 니까 팔로우 기능이 있어야겠죠? 
팔로우 기능같은 경우 좀 더 까다롭습니다. 
유저마다 팔로워 목록과 팔로윙 목록이 존재할 것입니다. 
A 유저가 B 유저를 팔로우한다고 하면, A 유저의 팔로윙 목록에 B 를 추가하고 B 유저의 팔로워 목록에 A 를 추가해야 합니다. 
이럴 경우 한번에 2개의 Aggregate 를 동시에 업데이트해야하는데 Aggregate 는 각각이 transaction boundary 를 가지므로 한번의 트랜잭션으로 업데이트하는 것이 불가능합니다. \
그렇다고 해서 팔로우 기능 자체를 하나의 Aggregate 로 분리하게 될 경우 동시성에 문제가 생깁니다. 
모든 유저들의 팔로우 요청을 하나의 Aggregate 가 처리하는 것은 말이 안되죠. \
이럴 경우 "A 유저의 팔로윙 목록에 B 를 추가하는 것"과 "B 유저의 팔로우 목록에 A 를 추가하는 것" 을 별도의 트랜잭션으로 따로 따로 수행합니다. 
이럴 경우, eventual consistency 의 특성때문에 A 유저의 팔로윙 목록에는 B 가 있는데 B 유저의 팔로우 목록에 A 가 없는 상황이 일시적으로 발생할 수 있습니다. 
또한, 팔로우 요청을 2개의 트랜잭션으로 분리했기 때문에 만약 하나의 트랜잭션만 실패한다면 성공한 다른 하나의 트랜잭션에 대한 롤백 로직도 수행을 해야만 합니다. 이러한 방법을 **SAGA 패턴**이라고 부릅니다. \
이렇듯 여러 개의 Aggregate 에 대해서 하나의 트랜잭션을 수행해야하는 경우 SAGA 패턴이나 Read Model 을 활용해서 해결하게 됩니다. 하지만 SAGA 패턴이 신경쓸 점도 많고 구현이 약간 까다롭기도 하고 eventual consistency 라는 점때문에 부담스러울 때가 있습니다. 이것과 관련해서 하고 싶은 말이 많은데 나중에 기회가 된다면 별도의 포스팅으로 다뤄보도록 하겠습니다.

4) 회원가입시 닉네임의 중복 체크 (유일성 체크) 를 구현하는 것도 까다롭습니다. 
Read Model 을 통해서 1차적으로 중복 체크를 수행할 수 있지만 Read Model 은 eventual consistency 이기 때문에 중복을 허용할 가능성도 존재합니다. 
따라서 추가적인 방지책이 필요한데 다양한 방법들을 생각해볼 수 있습니다. \
일단, 닉네임을 관리하는 Aggregate 를 만드는 것을 생각해볼 수 있습니다. 
회원가입시 일단 Read Model 을 통해 1차적으로 중복 체크를 한뒤 통과하면 해당 Aggregate 를 통해 닉네임을 예약하는 것입니다.
하지만 이 방법은 회원가입이 동시다발적으로 일어날 경우 동시성이 떨어질 수 있습니다.
이것이 문제가 된다면 규칙에 따라 여러 개의 Aggregate 를 만드는 것을 생각해볼 수 있습니다. 
닉네임이 영문자 소문자로만 이루졌다고 했을 때 닉네임이 시작하는 글자에 따라 Aggregate 를 분리하면 총 26개의 Aggregate 로 부하를 분산할 수 있습니다.
처음 2글자를 가지고 분리한다면 총 676개의 Aggregate 가 만들어지겠죠. \
두번째로, 일단 Read Model 에서의 중복 체크만 통과하면 회원가입을 허용해주고 뒤늦게 닉네임 충돌이 발견됐을 경우 계정을 정지하는 방법입니다. 
만약 시스템의 eventual consistency 속도가 매우 빠르다고 한다면 Read Model 이 굉장히 신뢰도가 높을 것이고 만약에 중복된 닉네임이 허용되더라도 충돌이 굉장히 빨리 발견될 것이기 때문에 이것도 하나의 방법이 될 수도 있습니다. \
마지막으로, 그냥 Aggregate 내의 트랜잭션에 중복 체크에 대한 부분을 추가하는 것입니다. 
좀 더 구체적으로 예시를 들면 DB 에 닉네임 목록을 저장하는 테이블을 하나 만듭니다. 
그리고 Aggregate 를 DB 에 추가할때 해당 테이블을 이용해서 중복체크를 수행하고 닉네임을 테이블에 추가하는 SQL 을 같은 트랜잭션에 묶는 것입니다. 이렇게 할 경우 서로 다른 Aggregate 들끼리 DB contention 이 발생하여 동시성이 떨어질수 있습니다. 그리고 그 사실이 Aggregate 설계상으로 잘 드러나지 않고 persistence 로직을 봐야 알 수 있는 암시적인 부분이라는 점에서 Aggregate 의 핵심 가치와는 상충되는 부분이라고 생각합니다. 또한 이렇게 되면 Aggregate 들이 서로 다른 DB 를 사용할 수 있는 등의 유연성을 포기해야 합니다. 하지만 구현이 되게 간단해지기 때문에 고려해볼만한 방법입니다.

5) 유저가 자신의 타임라인에 글을 올리는 경우를 생각해봅시다.
User Aggregate 가 해당 유저에 대한 모든 글 목록을 가지고 있고 모든 글에 대한 추가/삭제/수정 요청을 처리한다고 하더라도 동시성에는 문제가 없습니다. (한 유저가 여러개의 디바이스를 통해 동시다발적으로 글을 작성하진 않을 것이므로)
하지만 이렇게 되면 User Aggregate 를 로딩할 때마다 모든 글 목록을 같이 불러와야해서 매우 비효율적이고 메모리도 많이 소모하게 됩니다.
따라서 이건 좋은 Aggregate 설계가 아닙니다. \
해결책은 유저의 글 각각을 별도의 Aggregate 로 분리하는 것입니다. 
이렇게 되면 Aggregate 하나하나가 작은 크기를 유지하기 때문에 Aggregate 로딩이 효율적이고 메모리도 적게 소모하게 됩니다. \
Aggregate 를 쪼개지 않고서 할 수 있는 다른 방법은 Lazy Loading 을 활용하는 것입니다. 
User Aggregate 가 로딩될 때 모든 글 목록을 같이 로딩 (eager loading) 하는게 아니라 주어진 요청을 처리하는데 필요한 글들만 로딩하는 것입니다. \
보통 Aggregate 를 분리하는게 DDD 정론(?) 에 부합하는 방법이긴 하지만, Aggregate 를 분리하면 여러가지 제약사항들 (트랜잭션 경계, eventual consistency 등) 이 생기고 개발이 더 복잡해질 수 있기 때문에 경우에 따라 선택은 달라질 수 있다고 생각합니다.

Aggregate 설계시 맞닥뜨리는 대표적인 문제점들과 해결방안에 대해서 다뤄봤습니다. \
이렇듯 도메인 요구사항들을 Aggregate Pattern 에 맞게 모델링하는게 쉽지 않은 경우가 많습니다. 
그러나 비록 공부도 많이 필요하고 경험도 많이 필요하지만 모델링만 제대로 된다면 정말 든든한 설계가 되지 않을까 싶습니다.

## 다른 기술들과의 연관성

Aggregate Pattern 는 단순히 DDD 만에서의 이야기만은 아닙니다. 
다른 기술들에서도 Aggregate Pattern 과 밀접한 부분들이 많이 있습니다.

액터 모델같은 경우 액터를 Aggregate Root 에 연관지어 생각할 수 있습니다. (Aggregate Root 는 Aggregate 내의 객체들중 외부에 노출되는 단 하나의 객체를 의미합니다. Aggregate Root 는 다른 객체들을 모두 캡슐화하며 외부로 부터 요청이 왔을 때 다른 객체들과 함께 해당 요청을 수행합니다.) \
Aggregate 들은 서로 독립되어 있고 각자 병렬적으로 요청을 수행하고 자신의 상태를 변경시킵니다.
또한 Aggregate 는 불변성을 유지하기 위해 들어오는 요청들에 대한 동시성 컨트롤을 할 수 있어야 합니다.
놀랍게도 이는 정확하게 액터의 개념과 일치합니다. 
**따라서 Aggregate Pattern 을 실제로 구현하기 위한 구현 레벨의 기법으로서 액터 모델을 사용할 수 있습니다.**

이벤트 소싱은 persistence 기법으로서 기존의 전통적인 CRUD 방식과 대비됩니다. 
전통적으로는 DB 스키마를 설계하고 DB 에 CRUD (Create, Read, Update, Delete) 성격의 명령들을 이용해 상태를 persistence 하게 됩니다. 
하지만 이벤트 소싱은 상태가 아닌 발생한 이벤트들만을 쌓아나가고 이벤트들을 replay 함으로써 최종 상태를 얻게 됩니다. \
이벤트들은 순서가 존재하기 때문에 이벤트를 쌓아나갈때 번호 (sequence number 혹은 version) 도 같이 붙여나가게 됩니다.
만약 이벤트를 DB 에 쌓을때 이미 똑같은 번호의 이벤트가 존재하면 실패하게됩니다. 
따라서 이벤트가 동시다발적으로 발생할 경우 번호 중복이 빈번하게 발생해 문제가 됩니다. 
이를 해결하려면 동시성 컨트롤이 필요합니다. 
오직 하나의 요청만 동시에 처리가능하도록 한다면 즉 들어오는 요청들을 차례차례 처리하도록 한다면 이런 문제는 해결됩니다. 
단 동시성이 매우 낮아지는 문제가 발생합니다. 
따라서 이벤트를 전역적인 스코프에서 발생시키는 것이 아니라 특정 객체별로 독립적으로 이벤트를 발생시키도록 하고 이벤트를 쌓을때 객체 id 와 그 객체내에서의 이벤트 번호를 같이 붙여나가도록 합니다. 
이러한 방법은 Aggregate 의 개념과 정확히 일치합니다. 
"특정 객체별로" -> "Aggregate 별로", "객체 id" -> "Aggregate Id" 로 그대로 매핑됩니다. 
**즉, 이벤트 소싱을 사용할 때 필연적으로 Aggregate Pattern 을 같이 사용하게 됩니다.** \
Aggregate Pattern 을 사용할 때 이벤트 소싱은 필수가 아닙니다.
**하지만 이벤트 소싱을 같이 사용할 경우 Aggregate 들이 DB 레벨에서까지 독립적인 것이 보장되므로 더 많은 유연성과 장점을 얻을 수가 있습니다.**

마이크로서비스 아키텍쳐에서의 각 서비스들과 Aggregate 를 연관지어 생각할 수 있습니다.
사실 마이크로서비스와 Aggregate 는 서로 다른 레벨에서의 패턴입니다.
마이크로서비스는 좀 더 큰 규모의 아키텍쳐 레벨에서의 패턴이고 Aggregate 는 코드 레벨에서의 패턴이라고 할 수 있습니다.
하지만 서로 특성이 매우 비슷하고 Aggregate 가 분산 환경과 연관이 깊다보니, 마이크로서비스와 Aggregate 에서 비슷한 이슈들이 많습니다. 
예컨데, 마이크로서비스에서 분산 트랜잭션같은 경우 여러 개의 Aggregate 사이에서 트랜잭션을 수행해야하는 경우와 유사합니다.
따라서 해결책 또한 비슷합니다. (SAGA 패턴)

## 마무리

저는 어떤 문제를 해결하는데 있어서 "구조적인 해결책" 을 좋아합니다. 
그때그때 잔머리를 써서 해결하는게 아니라 애초에 문제가 발생하기 힘든 구조를 만드는 것이죠.
예를 들면 강력한 타입시스템은 프로그래머가 정해진 타입 규칙을 만족하게 코딩하도록 강제함으로써 소프트웨어 결함을 컴파일타임에 방지합니다. 
Aggregate Pattern 도 비슷한 맥락인 것 같습니다. 
일련의 설계 규칙들을 지키도록 강제함으로써 소프트웨어 복잡도를 효과적으로 분리하고 분산 환경으로의 확장성을 제공하고 트랜잭션 경계 및 동시성을 명확히 할 수 있도록 도와줍니다. \
또한 정적타이핑 시스템은 때로 강력한 타입시스템 규칙으로 인해 특정 구현이 힘들거나 번거로운 경우가 있습니다. (동적 타이핑에서는 쉽게 가능한데 말이죠.) 
마찬가지로 Aggregate Pattern 에서도 지켜야하는 규칙들로 인해 요구사항을 Aggregate 로 모델링하기 성가신 경우들이 있습니다. \
한마디로 정리하자면, **Aggregate Pattern 은 많은 문제들에 대해 설계레벨에서 구조적인 해결책을 제공합니다. 
하지만 정해진 규칙들을 만족하도록 Aggregate 를 모델링하는 것은 많은 노력이 필요합니다.**

지금까지 Aggregate Pattern 에 대해서 포스팅을 해봤습니다. 
Aggregate Pattern 은 정말 최근 저의 최애 원픽 설계 기법인데요. 
사실 보편적으로 알려져 있지 않아서 너무 안타까운 마음이 큽니다. 
많은 사람들이 관심을 가지고 연구도 많이 되어서 실무에도 많이 적용되었으면 좋겠습니다~

## Further Reading

* [Domain-Driven Design: Tackling Complexity in the Heart of Software](https://www.amazon.com/gp/product/0321125215)
* [https://cqrs.nu/Faq](https://cqrs.nu/Faq)
* [Martin Fowler - tagged by Domain Driven Design](https://martinfowler.com/tags/domain%20driven%20design.html)
* [What's the point of the Aggregate pattern?](https://medium.com/@philsarin/whats-the-point-of-the-aggregate-pattern-741a3132da5c)
* [Aggregates & Entities in Domain-Driven Design](http://thepaulrayner.com/blog/aggregates-and-entities-in-domain-driven-design/)
* [Effective Aggregate Design](https://dddcommunity.org/library/vernon_2011/)
* [Saga Pattern](https://microservices.io/patterns/data/saga.html)

**[이 포스팅을 taeguk 블로그에서 보기](https://taeguk2.blogspot.com/2020/02/ddd-aggregate-pattern.html)**
