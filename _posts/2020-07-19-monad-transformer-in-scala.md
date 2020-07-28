---
layout: post
title: "Monad Transformer In Scala"
date: 2020-07-19 13:00
author: taeguk
tags: [Scala, Monad Transformer, Monad, MTL, Functional Programming, RWST, ZIO]
---

안녕하세요~
오늘은 **Monad Transformer** 에 대해서 포스팅해보려고 합니다 ㅎㅎ

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2020/07/monad-transformer-in-scala_19.html)**

## Monad Transformer 란?

자, 다음과 같이 간단한 코드가 있습니다.

```scala
case class User(id: String, bestFriendId: String)

def getUser(userId: String): Future[Option[User]] = ???

def getBestFriendOfUser(userId: String): Future[Option[User]] =
  for {
    userOpt        <- getUser(userId)
    bestFriendIdOpt = userOpt.map(_.bestFriendId)
    bestFriendOpt  <- bestFriendIdOpt match {
                        case Some(bestFriendId) =>
                          getUser(bestFriendId)
                        case None               =>
                          Future.successful(None)
                      }
  } yield bestFriendOpt
```

`getBestFriendOfUser` 함수를 보면 `Future` 와 `Option` 이 함께 쓰이고 있는 것을 볼 수 있습니다.
`User` 가 `Option` 과 `Future` 에 의해 2중으로 감싸져있기 때문에 실제 `User` 값에 대해 뭔가 연산을 수행하려면 상당히 성가시고 가독성도 떨어지게 됩니다. <br/>
결국 문제가 되는 부분은 "2중으로 감싸져있다" 는 점입니다. 
즉, `Future` 와 `Option` 을 하나로 합친 타입을 만들어 사용한다면 코드가 깔끔해지지 않을까? 라는 생각을 해볼 수 있습니다.

```scala
case class FutureOption[+A](value: Future[Option[A]]) {
  def map[B](f: A => B): FutureOption[B] =
    FutureOption(value.map(_.map(f)))
  
  def flatMap[B](f: A => FutureOption[B]): FutureOption[B] =
    FutureOption(
      value.flatMap { (aOpt: Option[A]) => 
        val fOpt: Option[Future[Option[B]]] = aOpt.map(f(_).value)
        fOpt.getOrElse(Future.successful(None))
      }
    )
}

def getBestFriendOfUser(userId: String): Future[Option[User]] =
  (for {
    user       <- FutureOption(getUser(userId))
    bestFriend <- FutureOption(getUser(user.bestFriendId))
  } yield bestFriend).value
```

위와 같이 `Future` 와 `Option` 을 하나로 묶어서 사용할 수 있도록하는 `FutureOption` 헬퍼 타입을 만들어 사용한 결과 코드가 훨씬 깔끔해진 것을 확인할 수 있습니다. 
`flatMap` 의 경우 구현이 약간 복잡하지만 차분하게 읽으시면 이해가 되실 겁니다.

정리해보자면, 여러 개의 모나드를 중첩해서 사용할 경우 코드가 상당히 더러워지게 되고, 이를 해결하기 위해 여러 개의 모나드를 하나의 모나드인 것처럼 사용할 수 있게 만드는 헬퍼 타입을 만들 수 있습니다. <br/>
자, 근데 여기서 문제점이 있습니다. 
세상에는 수많은 종류의 모나드가 있는데 이러한 모나드들의 모든 조합을 위해 일일히 헬퍼 타입을 만든다면 너무 많은 헬퍼 타입이 필요해질 것입니다. 
따라서 제너릭을 활용한 방법이 필요합니다.

```scala
import scala.language.higherKinds

trait Monad[F[_]] {
  def pure[A](a: A): F[A]
  def map[A, B](value: F[A])(f: A => B): F[B]
  def flatMap[A, B](value: F[A])(f: A => F[B]): F[B]
}

implicit val futureMonad = new Monad[Future] {
  def pure[A](x: A): Future[A]                             = Future(x)
  def map[A, B](value: Future[A])(f: A => B): Future[B]    = value.map(f)
  def flatMap[A, B](value: Future[A])(f: (A) => Future[B]) = value.flatMap(f)
}

case class OptionT[F[_], A](value: F[Option[A]]) {
  def map[B](f: A => B)(implicit F: Monad[F]): OptionT[F, B] =
    OptionT(
      F.map(value) { (aOpt: Option[A]) =>
        aOpt.map(f)
      }
    )

  def flatMap[B](f: A => OptionT[F, B])(implicit F: Monad[F]): OptionT[F, B] =
    OptionT(
      F.flatMap(value) { (aOpt: Option[A]) =>
        val fOpt: Option[F[Option[B]]] = aOpt.map { (a: A) => f(a).value }
        fOpt.getOrElse(F.pure(None))
      }
    )
}

def getBestFriendOfUser(userId: String): Future[Option[User]] =
  (for {
    user       <- OptionT(getUser(userId))
    bestFriend <- OptionT(getUser(user.bestFriendId))
  } yield bestFriend).value
```

위 코드에서는 `FutureOption` 대신에 `OptionT` 를 활용한 것을 볼 수 있습니다. 
이렇게 되면 임의의 모나드 타입 `F` 에 대해서 `F[Option[A]]` 꼴로 중첩된 모나드를 다룰 수 있게 됩니다. <br/>
즉, 따라서 필요한 헬퍼 타입 (`OptionT`) 의 수가 매우 줄어들게 됩니다. 
만약 세상에 모나드가 10개가 있다면, 헬퍼 타입도 10개만 만들면 됩니다.

그리고 `OptionT` 를 무조건 저렇게 연산을 쉽게 하기위한 임시 객체로만 사용해야하는건 아닙니다. 
그냥 아래와 같이 사용하면 코드가 더 깔끔해지는 것을 확인할 수 있습니다.

```scala
def getUser(userId: String): OptionT[Future, User] = ???

def getBestFriendOfUser(userId: String): OptionT[Future, User] =
  for {
    user       <- getUser(userId)
    bestFriend <- getUser(user.bestFriendId)
  } yield bestFriend
```

자 그래서 지금까지 여러 개의 모나드를 조합해서 사용할 때 발생하는 문제점과 그것을 해결하기 위한 방법 (`OptionT`) 에 대해서 알아봤는데요~
`OptionT` 가 바로 Monad Transformer 입니다!! <br/>
즉, monad transformer 는 이렇게 여러 모나드들을 조합해서 사용하는데 도움을 주고, 대표적인 예시로서 위에서 저희가 직접 구현한 `OptionT` 가 있습니다.

>  Monad transformers allow modular composition of separate functional effect types into a single function effect with the ability to locally introduce and eliminate effect types. - John A. De Goes

Monad transformer 를 단순히 모나드들을 조합해서 사용할 때 발생하는 boilerplate code 를 해결하기 위한 용도로 보는 것은 사실 정확한 통찰은 아닙니다.
제가 인용한 문장에서 알 수 있듯이, **근본적으로 monad transformer 가 하는 역할은 여러 개의 모나드를 조합해서 하나의 모나드처럼 사용할 수 있게 하되, 한번에 오직 하나의 모나드 기능만 사용할 수 있도록 하는 것입니다.** <br/>
예를 들면, `OptionT` 는 임의의 모나드들 (`F`) 과 `Option` 모나드를 합쳐서 하나의 모나드처럼 사용할 수 있게 해줍니다. 
그리고 조합된 모나드들중에 하나의 모나드 기능을 선택해서 사용할 수 있게 해줍니다. (좀 있다 아래에서 예시를 볼 수 있습니다.)

자, 지금까지 monad transformer 에 대해서 알아봤는데요.
위에서는 설명을 위해 우리가 직접 구현을 해봤지만, 실제로는 cats 나 scalaz 같은 함수형 프로그래밍 라이브러리들에 이미 구현되어 있기 때문에 그걸 그냥 사용하시면 됩니다 :)

```scala
import cats.data._
import cats.implicits._

def getUser(userId: String): OptionT[Future, User] = ???

def getBestFriendOfUser(userId: String): OptionT[Future, User] =
  for {
    user       <- getUser(userId)
    bestFriend <- getUser(user.bestFriendId)
  } yield bestFriend
```

## MTL (Monad Transformer Library)

지금까지 monad transformer 를 사용해서 모나드들을 조합하는 방법을 다뤘는데요. 
사실 여기에는 여러가지 단점들이 있습니다. <br/>
바로 코드를 보시죠.

```scala
// https://github.com/typelevel/kind-projector 플러그인 필요
import cats.data._
import cats.implicits._

case class State(str: String)

def repeat(num: Int): EitherT[StateT[Future, State, *], Exception, String] =
  for {
    _     <- if (num < 0)
               EitherT.leftT[StateT[Future, State, *], Exception](new Exception("num should be equal or greater than 0."))
             else
               EitherT.rightT[StateT[Future, State, *], Exception]("dummy")
    state <- EitherT.liftF(StateT.get[Future, State])
  } yield state.str * num
```

긴 말 할 필요없이 위 코드만 봐도 monad transformer 의 끔찍함을 알 수 있습니다.

> 근본적으로 monad transformer 가 하는 역할은 여러 개의 모나드를 조합해서 하나의 모나드처럼 사용할 수 있게 하되, 한번에 오직 하나의 모나드 기능만 사용할 수 있도록 하는 것입니다.

제가 위에서 이런 말을 했었는데요.
위 코드에서 설명을 해보자면 `EitherT.leftT` 와 `EitherT.rightT` 는 `Either` 모나드의 기능을 사용하는 것이고, `EitherT.liftF(StateT.get[Future, State])` 는 `State` 모나드의 기능을 사용하는 것입니다. <br/>
여기서 문제점은 조합된 모나드의 기능을 사용하는게 편의성이 상당히 떨어지고 코드가 너저분해진다는 것입니다. <br/>
특히 monad transformer 가 중첩된 순서에 따라 사용방법이 달라집니다.
예를 들어, 위 코드에서 `Either` 모나드의 기능을 사용하는건 `EitherT` 만 쓰면 되는 반면, `State` 모나드의 기능을 사용하기 위해서는 `StateT` 뿐만 아니라 `EitherT.liftF` 도 써야합니다. <br/>

또한 monad transformer 를 사용할때는 중첩된 순서가 아주 중요해집니다. <br/>
```EitherT[StateT[Future, State, *], Exception, String]``` <br/>
```StateT[EitherT[Future, Exception, *], State, String]``` <br/>
이 두 타입은 서로 의미상으로는 동일하지만 실제로 타입은 달라서 서로 호환이 되지 않는 문제가 발생합니다.

이렇듯 monad transformer 는 많은 문제점들이 있습니다. 
여기서 우리는 한가지 아이디어를 생각해볼 수 있습니다. <br/>
모나드 기능 각각을 타입클래스로 만들면 어떨까요?
그리고 구체적인 모나드 타입이 아닌 그러한 타입클래스들에만 의존해서 코드를 작성하면 어떨까요? <br/>
이렇게 하면 모나드를 조합하는 순서는 더 이상 중요하지 않게 됩니다.
**이렇듯 구체적인 타입 (`EitherT`, `StateT` 등) 이 아닌 타입클래스 기반의 모나드 조합을 가능하게 해주고, 구체적인 타입 (monad transformer) 로의 구체화를 위한 타입 클래스 인스턴스들을 제공해주는 라이브러리를 MTL 라이브러리라고 부릅니다.** <br/>

```scala
import cats.MonadError
import cats.implicits._
import cats.mtl.implicits._ // monad transformer 로의 구체화를 위한 타입 클래스 인스턴스들
import cats.mtl.MonadState

// 타입 클래스 기반의 모나드 기능 조합/사용
def repeat[F[_]](num: Int)(implicit S: MonadState[F, State], E: MonadError[F, Exception]): F[String] =
  for {
    _     <- E.raiseError(new Exception("num should be equal or greater than 0.")).whenA(num < 0)
    state <- S.get
  } yield state.str * num

// 실제 monad transformer 로의 구체화.
val materializedProgram = repeat[EitherT[StateT[Future, State, *], Exception, *]](1)
```

위는 [cats-mtl](https://typelevel.org/cats-mtl/) 을 사용한 코드입니다. 
코드가 훨씬 간결하고 가독성이 있는 것을 알 수 있습니다. <br/>
뿐만 아니라 구체적인 타입이 아닌 타입 클래스 기반으로 추상화된 코드를 작성했기 때문에 추후에 concrete type 을 얼마던지 바꿀 수 있습니다. <br/>
사실 MTL 라이브러리를 쓴다고 해서 concrete type 으로 무조건 monad transformer 를 써야하는 것은 아닙니다.
단순히 타입 클래스 기반으로 모나드 기능들을 조합하고 사용하고 싶은 목적으로 MTL 라이브러리를 쓰고, concrete type 으로는 충분히 강력한 하나의 모나드 타입 (ex, ZIO) 만 사용해도 됩니다. <br/>

## 그 외 Monad Transformer 의 단점 및 대체재
사실 monad transformer 는 얼핏 보기에는 매우 좋아보입니다. 
자신이 원하는 모나드들을 마음대로 유연하게 조합해서 사용할 수 있기 때문이죠. 
MTL 라이브러리를 활용하면 단점들도 많이 커버가 됩니다. <br/>
하지만 그럼에도 불구하고 여전히 단점들이 남아있고 그중 가장 대표적인 단점은 성능문제 입니다. <br/>
위의 `OptionT` 구현을 보면 알 수 있듯이 monad transformer 기능을 위해 함수 호출과 객체 생성 오버헤드가 추가되게 됩니다. 
Monad transformer 가 중첩되면 될수록 이러한 오버헤드는 계속해서 커지게 됩니다. <br/>
또한 monad transformer 들은 stack safe 하지 않은 것들도 있어서 사용에 주의해야 합니다.
(이렇게 stack unsafe 한 transformer 들을 stack safe 하기 만들기 위해 성능을 더 포기하기도 합니다. [관련 벤치마크](https://github.com/iravid/transformer-benchmarks))

이러한 문제들을 해결하기 위한 실용적인 방법은 그냥 여러가지 기능들을 제공하는 강력한 모나드를 사용하는 것입니다.
예를 들면, cats 의 `RWS (ReaderWriteState)` 가 있습니다. 
이 모나드는 `Reader`, `Writer`, `State` 모나드의 기능을 모두 제공합니다.
원래였다면 monad transformer 를 사용해 3개의 모나드를 조합해서 써야하지만, `RWS` 를 사용하면 하나의 모나드로도 가능해지기 때문에 위에서 말한 성능 문제같은 것들이 해결되게 됩니다. <br/>
하지만, 여전히 `RWS` 가 지원하지 않는 기능들 (예외처리, 동시성 등) 이 많습니다. 
따라서 `RWS` 의 monad transformer 버전인 `RWST` 도 존재합니다. <br/>
그렇다면 monad transformer 가 아예 필요없을 정도로 아주 강력한 단 하나의 모나드는 없을까요?? 이를테면 God Monad?! <br/>
**있습니다!! 바로 [ZIO](https://zio.dev/) 입니다!!** <br/>
제가 인상깊게 본 발표자료 하나를 공유하겠습니다. 
꼭 보시길 추천드립니다!! ([https://www.slideshare.net/jdegoes/one-monad-to-rule-them-all](https://www.slideshare.net/jdegoes/one-monad-to-rule-them-all)) <br/>
나중에 기회가 되면 요즘 저의 최애 라이브러리인 ZIO 에 대해서도 포스팅해보도록 하겠습니다 ㅎㅎ

오랜만에 Monad Transformer 에 대해서 다뤄봤습니다 ㅎㅎ <br/>
빨리 함수형 프로그래밍이 대중화되는 세상이 오기를 바라면서.. 글을 마치겠습니다!
