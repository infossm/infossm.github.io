---
layout: post
title: "Akka typed 소개"
date: 2019-09-18 23:55
author: taeguk
tags: [Akka, Akka-Typed, Scala, Actor, strongly-typing, functional-programming]
---
안녕하세요~ 오늘은 akka 의 새로운 인터페이스 (API) 인 akka typed 에 대해서 소개시켜드리는 시간을 가져볼까 합니다!<br/>
Akka 및 액터 모델에 대한 기본적인 내용은 이미 알고 있다고 가정하며 scala 를 기반으로 설명하겠습니다.

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/09/akka-typed_18.html)**

## 기존 akka 에서의 문제점
akka 는 훌륭한 프레임워크이지만 인터페이스 (API) 상으로 아쉬운 부분들이 있습니다.
```scala
class EchoStringActor extends Actor {
  def receive = {
    case str: String => sender() ! str
    case _           => sender() ! "Unexpected message"
  }
}
```
1. 메세지의 타입이 `Any` 입니다.
	* Java/scala 는 정적 타이핑 언어로서 컴파일타임에 타입시스템을 이용해 많은 오류를 사전에 차단시켜줍니다. 그러나 akka 의 인터페이스에서는 액터가 받는 메세지의 타입이 `Any` 입니다. 따라서 정적 타이핑의 이점을 전혀 누릴수가 없습니다.
	* 만약에 개발자가 실수로 특정 타입의 메세지를 처리하는 것을 깜박하더라도 컴파일 에러가 나지 않습니다.
	* 액터들이 어떤 메세지를 서로 주고받는지가 명확하게 드러나지 않아서 저는 전체 액터 시스템을 분석하고 유지보수할 때 상당히 복잡하고 머리속에 정리가 잘 안된다는 느낌을 많이 받았습니다.
2. 함수형 스타일이 아닙니다.
	* Actor 를 정의하기위해서는 Actor 클래스를 상속받아서 새로운 클래스를 만들어줘야 합니다.
	* 메세지 처리 함수를 변경하기 위해서 `become()` 을 사용해야 하고 액터 내부의 상태를 mutable (`var`) 로서 관리해야 합니다. 저는 이로 인해 액터 코드가 상당히 복잡하고 스파게티처럼 꼬여있는 것 같다는 느낌을 많이 받았고 코드를 분석하고 유지보수하는게 까다롭다고 느꼈습니다.

그러면 akka typed 은 이러한 문제들을 어떠한 방향으로 해결했는지 살펴보겠습니다.

## akka typed 의 등장
akka typed 은 위에서 살펴본 기존 API 의 문제점들을 해결하기 위해 새롭게 개발된 API (인터페이스) 입니다. 여기서 주목할 점은 akka typed 은 단순히 새로운 인터페이스를 의미할 뿐이고 실제로 내부 동작은 기존과 똑같다는 것입니다.
```scala
object EchoString {
  case class Message(str: String, replyTo: ActorRef[String])
  
  def behavior: Behavior[String] =
    Behaviors.receiveMessage { message =>
      message.replyTo ! message.str
      Behaviors.same
    }
}
```
akka typed 은 강력한 타입시스템을 통해 개발자의 실수를 방지하고, 액터들간에 주고 받는 메세지가 명시적으로 드러남으로써 전체 액터시스템을 더 쉽게 이해할 수 있도록 해주고, 함수형 스타일로 액터를 개발할 수 있도록 합니다.<br/>
akka typed 가 구체적으로 어떻게 생겼는지 어떻게 사용하는지에 대한 내용들은 [공식 문서](https://doc.akka.io/docs/akka/current/typed/index.html)에 아주 잘 설명되어 있습니다. 제가 일일히 설명드리는건 굳이 불필요한거 같으니 공식 문서를 참고하시면 되겠습니다. 그보다 akka typed 의 특징과 장점들에 대해 정리해보겠습니다.

1. 강타입과 정적타이핑의 이점을 극대화
	* 액터가 받을 수 있는 메세지의 타입이 `ActorRef[Foo]` 와 같이 명시적으로 드러납니다. `ActorRef[Foo]` 타입의 액터에게는 오직 `Foo` 타입 및 서브타입의 메세지만 보낼 수 있고 그 외 타입의 메세지를 보내려고하면 컴파일 에러가 뜹니다.
	* 뿐만 아니라 액터들간에 주고 받는 메세지의 타입이 명시적으로 드러남으로서 전체 액터 시스템을 이해하는게 훨씬 수월해집니다.
2. 함수형 스타일
	* akka typed 에서는 액터를 정의하는 것이 아니라 "행위" (`Behavior`) 를 정의하게 됩니다. 액터 생성은 특정 Behavior 을 spawn 한다는 개념으로서 하게됩니다.
	* 메세지 처리 함수는 항상 메세지를 처리한 후 전이할 `Behavior` 를 반환하게 됩니다.
		* 기존에는 메세지 처리 함수를 변경하려면 `become()` 을 사용해야 했는데, akka typed 에서는 그냥 새로운 `Behavior` 를 반환하면 됩니다.
		* 기존에는 액터 내부의 상태를 mutable 하게 선언하고 메세지를 처리하면서 상태 객체를 수정하는 방식인데, akka typed 에서는 그냥 새로운 상태를 가지는 `Behavior` 를 만들어서 반환하면 되므로 상태를 immutable 하게 관리할 수 있습니다.
	* 좋고나쁨을 떠나서 다소 불편해진 점들이 있습니다.
		* `sender()` 가 없기때문에 메세지를 보내는 측에서 항상 응답을 받을 `ActorRef` 를 메세지에 같이 담아야 합니다. (위 `EchoString` 예시 참고)
		* 서로 다른 클래스 계통의 메세지를 받아야할 경우를 처리하기 까다롭습니다.
			* scala 는 sum type 을 지원하지 않기 때문에 특정 `Behavior` 가 서로 다른 클래스 계통의 메세지를 모두 받을 수 있어야 하는 경우 이를 구현하기가 까다롭습니다.
			* akka typed 에서는 [Message Adapter](https://doc.akka.io/docs/akka/current/typed/interaction-patterns.html#adapted-response) 라는 것을 통해 이에 대한 해결책을 제공합니다.
3. 기존 akka API 와 혼용해서 사용가능
	* 기존 akka API 의 액터와 akka typed 의 액터들도 서로 메세지를 주고 받을 수 있는 등 기존 akka API 와 혼용해서 사용이 가능합니다.
	* 따라서 점진적으로 akka typed 으로 교체하는 것도 가능합니다.

akka typed 의 특징들을 정리해봤는데요. akka typed 으로 한번 리펙토링을 쫙 수행해보시면 기존 akka 코드는 다신 보고 싶지 않으실 껍니다. 특히 강타입과 함수형 프로그래밍을 사랑하시는 분이라면 더더욱이요.

## 추가적인 차이점
기존 akka API 와 akka typed API 의 추가적인 차이점들에 대해서 몇 개만 더 설명해볼까 합니다.

1. Actor Persistence
	* Actor Persistence 에 대해서도 큰 차이점이 있습니다.
	* 기존에는 actor 를 persistence 하기 위해선 `PersistentActor` 를 상속받고 `receiveRecover` (actor 를 복구하기 위한 용도) 와 `receiveCommand` (command 를 처리하기 위한 용도) 를 정의해야 했습니다.
		* 이벤트를 persist 하기 위해선 사용자가 직접 `persist()` 를 호출해줘야 합니다.
		* 이벤트가 발생했을 때 상태를 사용자 코드에서 직접 변경해야 합니다.
	* akka typed 에서는 `EventSourcedBehavior` 를 생성하게 되고 생성자로 `commandHandler` (command 를 처리하고 Effect 를 발생) 와 `eventHandler` (이벤트가 발생했을 때 변경될 상태를 반환) 를 전달하게 됩니다.
		* akka typed 에서는 기존에 비해 많은 것들이 API 내부에서 수행됩니다. 이벤트 persistence 와 상태 변경이 API 내부에서 일어나게 됩니다.
		* `EventSourcedBehavior`, `commandHandler`, `eventHandler` 모두 함수형 스타일로 설계되어 있습니다 :)
		* actor 복구와 command 처리 모두 `commandHandler` 와 `receiveHandler` 를 이용해서 수행되게 됩니다.
	* 기존 인터페이스에서는 "actor 복구" 와 "command 처리" 를 사용자가 정의하도록 되어 있었는데 이는 너무 추상화가 덜된 것이였다고 생각합니다. akka typed 에서는 반면에 "command 처리시 어떤 행동을 할 것인지 (이벤트 발생 등)" 과 "이벤트가 발생했을 때 상태를 어떻게 변경할지" 를 사용자가 정의하게 합니다. 이게 좀 더 높은 수준의 추상화라고 생각합니다.
2. ALO (At-least-once) 지원
    * 기존에는 `PersistentActor` 에 `AtLeastOnceDelivery` 를 믹스인함으로써 ALO 를 사용할 수 있었습니다.
    * 그러나 akka typed 에서는 아직 ALO 를 지원하지 않습니다. 자세한 내용은 [해당 이슈](https://github.com/akka/akka/issues/20984)를 참고하시기 바랍니다.
3. FSM (Finite State Machine) 지원
    * 기존에는 FSM 을 위한 별도의 DSL 이 제공되었습니다.
    * 그러나 akka typed 에는 FSM 을 위한 별도의 기능이 존재하지 않습니다. 왜냐하면 `Behavior` 자체가 FSM 을 모델링하기에 적합하기 때문입니다. 자세한 내용은 [공식 문서](https://doc.akka.io/docs/akka/current/typed/fsm.html#behaviors-as-finite-state-machines)를 참고하시기 바랍니다.

## 마무리
[라이트밴드 블로그 포스팅](https://www.lightbend.com/blog/akka-2-5-22-brings-akka-typed-to-production-ready) 및 공식 문서에 따르면 akka typed 은 현재 production 에서 사용가능한 수준이라고 합니다. 그러나 API 는 [may change](https://doc.akka.io/docs/akka/current/common/may-change.html) 로 마킹되어 있어서 앞으로 하위호환성을 깨는 방향으로 수정될 가능성도 있습니다.<br/>
그러나 akka typed 가 몇 년이 넘는 오랜기간동안 개발되며 안정화가 된 만큼 API 가 수정되더라도 큰 변화는 없을 것이라고 생각합니다.<br/>
기존 untyped API 와 typed API 를 모두 써봤고 untyped 에서 typed 로 리펙토링도 해봤었는데요. 확실히 akka typed 이 압도적으로 우월하다고 느꼈습니다. Akka typed 를 처음 들었을때는 단순히 강타입 이라고만 생각했지만 실제로 공부하고 사용해보니 API 가 상당히 함수형 스타일로 설계되있고 이를 통해 얻을 수 있는 이점들이 많았습니다. 한번 맛보고나니까 앞으로 기존 untyped API 로는 개발 못할 거 같네요 ㅋㅋㅋ<br/>
아무튼 혹시 akka 를 사용하시는 다른 분께서 이 글을 보시면 akka typed 를 도입하는 걸 한번 고려해보시길 추천드립니다~
