---
layout: post
title: "Purely Functional Business Logic In Scala"
date: 2019-08-18 01:25
author: taeguk
tags: [Fuctional Programming, scala, cats, rwst, monad, DDD, Software Design]
---
안녕하세요~ 오늘은 스칼라에서 비지니스 로직을 purely functional 하게 설계하는 방법에 대해서 소개해볼까 합니다. <br/>
함수형 프로그래밍, 스칼라, cats, DDD 에 대해서 알고 계시면 이해가 수월하시겠지만 모르시는 분들을 위해서 기본적인 설명도 같이 첨부해두었습니다.

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/08/purely-functional-business-logic-in.html)**

## 다루는 내용
* cats 의 `RWST (ReaderWriterStateT)` 를 활용해서 핵심 비지니스 로직을 purely functional 하게 작성하는 방법
* 그렇게 작성된 비지니스 로직을 `DDD (Domain Driven Development)` 설계에서 사용하는 방법

## 비지니스 로직에 대한 고찰
소프트웨어 설계와 개발에 있어서 가장 중요한 건 요구사항 분석이라고 생각합니다. 그리고 그러한 요구사항을 처리하기 위한 로직 (즉, 비지니스 로직) 을 잘 구현하고 테스트하는 것이 개발에 있어서 가장 우선순위라고 생각합니다. <br/>
`DDD (Domain Driven Development)` 에서는 이러한 비지니스 로직들을 도메인 레이어에 위치시키고 핵심 도메인 엔티티와 규칙을 나타내는 코드는 세부 구현 (어떤 DB 를 사용하는지, 어떤 라이브러리를 사용하는지, 외부와 어떻게 통신하는지 등등) 에 의존하지 않도록 설계합니다. <br/>
자, 그렇다면 비지니스 로직을 도메인 레이어에 따로 분리시키고 세부 구현사항들과의 의존성을 제거하면 실질적으로 장점들이 있을까요? 인터넷에 검색해보시면 많은 이야기들을 보실 수 있는데요. 제가 경험한 장점들을 살짝 나열하자면 다음과 같습니다. <br/>
* 비지니스 로직과 요구사항들이 한눈에 들어오고 관리가 용이해집니다.
* 세부구현과의 의존성이 적으므로 테스트가 용이해집니다.
* 추후에 프레임워크나 기반 인프라를 변경하는 것이 용이해집니다. (보통 프레임워크나 인프라 위에서 비지니스 로직을 구현하는 것이 자연스럽다고 생각하실 수도 있지만 `DDD` 에서는 비지니스 로직이 가장 안쪽에 레이어에 위치하고 프레임워크나 인프라는 바깥 레이어에 위치하도록 하기때문에 이런 장점을 얻을 수 있게 됩니다.)

자, 그러면 이러한 비지니스 로직들을 purely functional 하게 작성하면 어떤 장점들이 있을까요? 일반적으로 purely functional 하게 코드를 작성했을 때의 장점들 (인터넷에 찾아보세요) 과 더불어 추가적으로 아래와 같은 장점들이 있습니다.
* 비지니스 로직을 나타내는 코드들은 side effect 가 없고 비지니스 로직들을 다 수행한 뒤에 최종적으로 한 곳에서 로직 수행 결과를 바탕으로 side effect 를 발생시킵니다. 따라서 side effect 코드가 최소화되고 한 곳으로 응집되기 때문에 얻는 장점들이 많습니다. 예를 들면, 로직을 수행하다가 중간에 실패하는 경우에도 손쉽게 [strong exception safety](https://en.wikipedia.org/wiki/Exception_safety) 를 보장할 수 있습니다. (트랙잭션 처리가 수월해집니다.)
* 비지니스 로직의 가독성이 상당히 향상됩니다. 저는 개인적으로 마치 요구사항 문서를 보고 있는 듯한 느낌을 받기도 합니다.

자세한 내용들은 코드를 통해서 살펴보시면 되겠습니다.

## cats 의 RWST
저는 이 포스팅을 위한 PoC 코드에서 비지니스 로직들을 purely functional 하게 작성할 때 cats 라이브러리의 RWST 를 사용하였습니다. 이 포스팅에서는 RWST 에 대해서 자세히 설명하진 않고 개념적으로 어떤 기능을 하는 지에 대해서만 간략하게 설명하겠습니다. <br/>
`RWST (ReaderWriteStateT)` 는 Reader 모나드, Writer 모나드, State 모나드를 합친 녀석입니다. (모나드에 대해 모르셔도 괜찮습니다. 그냥 특정 문맥 안에서 연산들을 순차적으로 수행하는데 도움을 주는 타입이라고 생각하시면 됩니다.) <br/>
일단 저 모나드 각각에 대해서 알아보도록 하겠습니다.
* `Reader 모나드` : 연산을 수행할 때 필요한 환경설정을 해주기 위한 용도라고 생각하시면 됩니다. 즉, Reader 모나드를 쓰면 필요한 의존성과 환경이 세팅된 "문맥" 하에서 연산들을 정의하고 체인처럼 순차적으로 엮을 수 있습니다. 나중에 Reader 모나드를 실행 (`run`) 시킬 때 구체적인 의존성을 주입해야 하고 그때서야 실제로 정의했던 연산들이 "실행" 되게 됩니다. Dependency Injection 을 위해서 사용되기도 합니다.
* `Writer 모나드` : 연산을 수행할 때 로깅을 할 수 있게끔 해줍니다. 다른말로 하면 "로깅을 할 수 있는 문맥" 을 제공합니다.
* `State 모나드` : 연산과정에서 상태를 얻고 변경할 수 있는 문맥을 제공합니다.

물론 위에 나열된 모나드를 쓰지 않고 함수의 인풋과 아웃풋으로도 환경 세팅 (의존성 주입), 로깅, 상태 변경이 가능합니다. 하지만 이렇게 할 경우 함수를 체인처럼 순차적으로 연결시킬 때, 연산 수행에 필요한 의존 객체, 로그를 나타내는 객체, 상태를 나타내는 객체를 모든 함수의 인풋으로 넣어야합니다. 또한 함수를 체인처럼 엮어야하므로 함수의 아웃풋으로도 항상 같이 반환해줘야 합니다. 하지만 이렇게 되면 함수 시그니처가 너무 복잡해지는 등의 고통을 받게 됩니다.. <br/>
따라서 그러한 객체들이 마치 전역 변수로 있는 것과 비슷한 느낌의 효과를 얻기 위해서 "특정 문맥" 안에서 함수를 체인처럼 엮는 기능이 필요한 것이고 이러한 기능을 `Reader`, `Writer`, `State` 모나드가 제공하는 것입니다. (사실 전역 변수라는 표현은 잘못된 표현이긴하지만 이해를 돕기위해 언급하였습니다.) <br/>
아무튼 `RWST` 는 이러한 세 가지 모나드를 합쳐놓은 녀석입니다. 일단은 이정도로만 설명을 마무리하도록 하겠습니다.

## PoC (Proof Of Concept)
이제 나머지 사항들은 코드를 통해 확인하시는게 좋을 것 같습니다. 함수형이나 스칼라에 익숙하지 않으시면 코드 분석이 힘들 수 있겠지만 전반적으로 코드가 주는 느낌은 느끼실 수(?) 있지 않을까 싶습니다. (참고 : [전체 소스코드 on github gist](https://gist.github.com/taeguk/6153695aad263777f4ba56ce202d2438))
```scala
/*
scalaVersion := "2.12.8"
scalacOptions ++= Seq(
  "-Xfatal-warnings",
  "-Ypartial-unification"
)
libraryDependencies += "org.typelevel" %% "cats-core" % "1.0.0"
libraryDependencies += "org.typelevel" %% "cats-effect" % "1.3.1"
libraryDependencies += "org.typelevel" %% "cats-effect-laws" % "1.3.1" % "test"
 */

import scala.language.higherKinds

import cats._
import cats.data._
import cats.effect._
import cats.implicits._

// 도메인 레이어
// 핵심 도메인 엔티티/로직/룰들이 위치하고 purely functional 하게 작성됩니다.
object DomainLayer {
  case class User(name: String, money: Int)

  sealed trait Menu
  case class Drink(name: String, alcoholicity: Double) extends Menu
  case class Food(name: String) extends Menu

  case class MenuOrder(menu: Menu, price: Int)

  case class Pub(
    name: String,
    menuBoard: Map[String, MenuOrder],  // key: name of menu
    dartGamePrice: Int
  )

  // RWST 를 좀 더 편하게 사용하기 위한 용도입니다.
  // 이렇게 제네릭 trait 으로 감싸지 않고 직접 `RWST[~~~~].xxx` 와 같이 사용하게 되면, 사용할 때마다 매번 제너릭 파라미터를
  // 넣어줘야해서 코드가 상당히 지저분해지게 됩니다.
  trait LogicHelper[F[_], E, L, S] {
    def ask(implicit F: Applicative[F], L: Monoid[L]): RWST[F, E, L, S, E] =
      RWST.ask

    def tell(l: L)(implicit F: Applicative[F]): RWST[F, E, L, S, Unit] =
      RWST.tell(l)

    def get(implicit F: Applicative[F], L: Monoid[L]): RWST[F, E, L, S, S] =
      RWST.get

    def modify(f: S => S)(implicit F: Applicative[F], L: Monoid[L]): RWST[F, E, L, S, Unit] =
      RWST.modify(f)

    def pure[A](a: A)(implicit F: Applicative[F], L: Monoid[L]): RWST[F, E, L, S, A] =
      RWST.pure(a)

    def raiseError[A](t: Throwable)(implicit F: MonadError[F, Throwable], L: Monoid[L]): RWST[F, E, L, S, A] =
      RWST.liftF(F.raiseError(t))
  }

  // 비지니스 로직이 표현력있고 간결하게 작성됩니다. 마치 요구사항 명세서를 읽는 것 같은 느낌을 주기도 합니다.
  trait PubLogics[F[_]] {
    type PubLogic[A] = RWST[F, Pub, Chain[String], User, A]
    object PubLogicHelper extends LogicHelper[F, Pub, Chain[String], User]
    import PubLogicHelper._

    def playDartGame(implicit F: MonadError[F, Throwable]): PubLogic[Unit] =
      for {
        _             <- tell(Chain.one("Play dart game"))
        dartGamePrice <- ask.map(_.dartGamePrice)
        currentMoney  <- get.map(_.money)
        _             <-
        if (currentMoney >= dartGamePrice)
          modify { user => user.copy(money = user.money - dartGamePrice) }
        else
          raiseError(new Exception(s"Money is not enough to play dart game. Money: $currentMoney, dart game price: $dartGamePrice"))
      } yield ()

    def orderMenu(menuName: String)(implicit F: MonadError[F, Throwable]): PubLogic[Menu] =
      for {
        _                 <- tell(Chain.one(s"Order the menu: $menuName"))
        menuBoard         <- ask.map(_.menuBoard)
        menuOrder         <-
          menuBoard.get(menuName) match {
            case Some(_menuOrder) => pure(_menuOrder)
            case None => raiseError(new Exception(s"Unknown menu: $menuName"))
          }
        (menu, menuPrice) =  (menuOrder.menu, menuOrder.price)
        currentMoney      <- get.map(_.money)
        _                 <-
          if (currentMoney >= menuPrice)
            modify { user => user.copy(money = user.money - menuPrice) }
          else
            raiseError(new Exception(s"Money is not enough to order. Money: $currentMoney, menu price: $menuPrice"))
      } yield menu

    // 실제 프로젝트에서는 "어떻게 놀 것인지" 를 파라미터로 받아서 그 것을 바탕으로 로직을 구성해야 하지만,
    // 이 코드는 어차피 개념증명을 위한 것이므로 다음과 같이 하드코딩하였습니다.
    def playInPub(implicit F: MonadError[F, Throwable]): PubLogic[Chain[Menu]] =
      for {
        nacho <- orderMenu("nacho")
        beer  <- orderMenu("beer")
        _     <- playDartGame
      } yield Chain(nacho, beer)
  }
}

// 어플리케이션 레이어
// 실질적으로 요청을 수행하며 그 과정에서 외부 통신, DB 접근, 서버 상태 변경등의 side effect 를 일으키게 됩니다.
object ApplicationLayer {
  import DomainLayer._

  case object PubLogicsWithIO extends PubLogics[IO]

  case class PlayInPubRequest(/* 생략 */)

  def playInPub(request: PlayInPubRequest, user: User, pub: Pub): Unit =
    // 실제 프로젝트에서는 `request` 를 바탕으로 `playInPub` 에 들어갈 파라미터를 구성해야 하지만,
    // 여기에서는 그냥 코드를 간단히 하기 위해 이 과정을 생략하였습니다.
    // 또한 여기에서는 편의를 위해 그냥 `user` 과 `pub` 을 파라미터로 받도록 하였지만,
    // 실제 프로젝트에서는 DB 를 통해서 읽어오는 등의 형태가 될 것 입니다.
    PubLogicsWithIO.playInPub
      .run(pub, user)
      .map { case (logs, updatedUser, orderedMenus) =>
        // 로직이 성공적으로 실행된 경우 그 결과를 바탕으로 각종 side effect 를 수행하면 됩니다.
        // 이처럼 실제 핵심 비지니스 로직은 모두 purely functional 하게 작성되게 되고,
        // side effect 를 발생시키는 코드는 최소화되고 응집되게 됩니다.
        println(s"Put the logs to logging system: $logs")
        println(s"Save the updated user state to database: $updatedUser")
        println(s"Send the ordered menus to client: $orderedMenus")
      }
      .handleError { cause =>
        // 로직을 수행하다가 중간에 실패하더라도 프로그램의 상태가 변하지 않습니다.
        // 따라서 transaction 처리가 매우 용이합니다.
        println(s"Failed to perform a logic: $cause")
      }
      .unsafeRunSync()
}

object BlogPosting extends App {
  import ApplicationLayer._
  import DomainLayer._

  val cheapPub = Pub(
    name = "Cheap Pub",
    menuBoard = Map(
      "nacho" -> MenuOrder(
        menu = Food("nacho"),
        price = 4000
      ),
      "beer" -> MenuOrder(
        menu = Drink("beer", 5.1),
        price = 3500
      )
    ),
    dartGamePrice = 2000
  )

  val premiumPub = Pub(
    name = "Premium Pub",
    menuBoard = Map(
      "nacho" -> MenuOrder(
        menu = Food("nacho"),
        price = 13000
      ),
      "beer" -> MenuOrder(
        menu = Drink("beer", 5.1),
        price = 10000
      )
    ),
    dartGamePrice = 4000
  )

  val user = User(name = "taeguk", money = 25000)

  println("-------------------------------------------")
  playInPub(PlayInPubRequest(), user, cheapPub)
  println("-------------------------------------------")
  playInPub(PlayInPubRequest(), user, premiumPub)
  println("-------------------------------------------")

  /* 실행결과는 다음과 같습니다.
-------------------------------------------
Put the logs to logging system: Chain(Order the menu: nacho, Order the menu: beer, Play dart game)
Save the updated user state to database: User(taeguk,15500)
Send the ordered menus to client: Chain(Food(nacho), Drink(beer,5.1))
-------------------------------------------
Failed to perform a logic: java.lang.Exception: Money is not enough to play dart game. Money: 2000, dart game price: 4000
-------------------------------------------
   */
}
```

## 마무리
오늘은 스칼라에서 비지니스 로직을 purely functional 하게 작성하고 DDD 설계시에 활용하는 방법에 대해 간단하게 살펴보았습니다. <br/>
앞으로는 스칼라나 함수형쪽으로 포스팅을 자주 하게 될 것 같네요. <br/>
다음에 또 만나요~
