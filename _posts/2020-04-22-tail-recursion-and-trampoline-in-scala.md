---
layout: post
title: "꼬리 재귀와 Trampoline In Scala"
date: 2020-04-22 23:59
author: taeguk
tags: [Scala, tail-recursion, Trampoline, functional-programming, TailRec, coroutine]
---

안녕하세요! 오늘은 스칼라 초심자를 대상으로 **Tail Recursion (꼬리 재귀)** 와 **Trampoline** 에 대해 포스팅하려고 합니다. <br/>
함수형 프로그래밍이나 모나드를 몰라도 이해할 수 있도록 노력해봤습니다~~

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2020/04/trampoline-in-scala.html)**

간단하게 1부터 n 까지 더해주는 함수를 아래와 같이 작성한 뒤 실행 해봅시다. 스택오버플로우 에러가 뜨는 것을 확인할 수 있습니다.

```scala
def unsafeSum(n: Int): Int =  
  if (n == 1) 1  
  else n + unsafeSum(n - 1)  
  
println(s"sum = ${unsafeSum(100000)}")

// 실행결과
// Exception in thread "main" java.lang.StackOverflowError
//     at dogs.MyApp$.unsafeSum(Main.scala:8)
//     at dogs.MyApp$.unsafeSum(Main.scala:8)
//     at dogs.MyApp$.unsafeSum(Main.scala:8)
//     at dogs.MyApp$.unsafeSum(Main.scala:8)
```

이번에는 아래와 같이 함수를 수정한 뒤에 실행 해봅시다. 스택오버플로우 없이 잘 동작하는 것을 확인할 수 있습니다.

```scala
def safeSum(n: Int, acc: Int): Int =  
  if (n == 0) acc  
  else safeSum(n - 1, n + acc)  
  
println(s"sum = ${safeSum(100000, 0)}")

// 실행결과
// sum = 705082704
```

대체 무슨 차이점 때문일까요?

`unsafeSum(5)` 가 값을 반환하려면 일단 `unsafeSum(4)` 를 호출해서 결과를 얻은다음에 그 결과에 `5` 를 더해야합니다. <br/>
즉, `unsafeSum(4)` 를 호출하기 전에 `5` 를 어딘가에 저장해두고, `unsafeSum(4)` 가 반환되면 그 반환값에 아까 저장해둔 `5` 를 더해야합니다. <br/>
어렵게 표현했지만 이것이 일반적인 함수의 실행과정이고 그렇기때문에 함수가 실행될때마다 콜스택이 점점 쌓여가는 것입니다. <br/>
예를 들어, 처음에 `unsafeSum(5)` 가 호출되어 `unsafeSum(1)` 실행까지 왔다면 스택에는 5, 4, 3, 2 가 차례대로 저장되어 있을 것입니다. <br/>
이렇게 계산을 위해 스택이 계속 쌓여나가는 구조이기 때문에 `unsafeSum` 는 n 이 클 경우 스택을 많이 소모하게돼 스택오버플로우가 발생하는 것입니다.

반면에 `safeSum(5, 0)` 이 값을 반환하려면 `safeSum(4, 5)` 를 호출하면 끝입니다. 
즉, `unsafeSum` 와는 달리 스택에 뭔가를 저장하고 있을 필요가 없습니다. 
따라서 이런 경우에 대해서 스칼라 컴파일러는 최적화를 수행하여 마치 다음과 같은 코드를 생성합니다.

```scala
def safeSum_optimized(initial_n: Int, initial_acc: Int): Int = {  
  var n = initial_n  
  var acc = initial_acc  
  
  while (true) {  
    if (n == 0)  
      return acc  
    else {  
      val next_n = n - 1  
      val next_acc = n + acc  
      n = next_n  
      acc = next_acc  
    }  
  }  
  
  ??? // Not reachable  
}
```

즉, `safeSum` 는 함수 실행의 마지막이 자기자신을 호출하는 것이기 때문에, 스택공간을 그대로 재활용해서 다음 함수 실행에 사용할 수 있는 것입니다. <br/>
이렇듯 함수 실행의 마지막 (꼬리) 가 자기자신을 호출 (재귀호출) 하는 것이기 때문에 이러한 재귀를 **꼬리 재귀 (Tail Recursion)** 라고 부릅니다.
그리고 스칼라 컴파일러는 이러한 꼬리 재귀 형태가 나올 경우 스택 공간을 재활용하도록 위와 같은 최적화를 수행해줍니다.

자신이 만든 함수가 꼬리 재귀인지 여부를 확인하려면 `@tailrec` 어노테이션을 활용하시면 됩니다.

```scala
import scala.annotation.tailrec  
  
@tailrec // 꼬리재귀가 아니므로 컴파일에러  
def unsafeSum(n: Int): Int =  
  if (n == 1) 1  
  else n + unsafeSum(n - 1)  
  
@tailrec
def safeSum(n: Int, acc: Int): Int =  
  if (n == 0) acc  
  else safeSum(n - 1, n + acc)
```

이렇듯 함수를 꼬리재귀 형태로 바꾸는건 함수 시그니쳐와 생각의 방식을 바꿔야만 가능합니다. <br/>
만약 복잡한 기능을 하는 재귀함수를 꼬리재귀 형태로 바꾸려고 한다면 쉽지 않을 것입니다. <br/>
그렇다면 임의의 함수를 꼬리재귀 형태로 바꿀 수 있는 정형화된 패턴같은 건 없을까요?

다음과 같이 의식의 흐름으로 생각해볼 수 있을 것 같습니다.
* `sum(5)` = `sum(4)` + 5 이다.
* `sum(5)` 의 계산은 다음과 같은 순서로 이루어진다.
	1. `sum(4)` 를 계산한다. (*추후 `LazyCall` 에 해당하는 부분*)
	2. 1번의 결과에 5 를 더한다. (*추후 `Logic` 에 해당하는 부분*)
	3. 2번의 결과를 반환한다. (*추후 `Return` 에 해당하는 부분*)
* `sum(4)` 의 계산을 별도의 스레드 (당연히 별도의 스택을 가지게 됨) 에서 수행하고 현재 스레드에서는 그 계산이 끝나기를 기다린다. 그 계산이 끝나면 거기에 5를 더해서 반환한다. 이렇게 하면 당연히 stack safe 하게 된다.
* 하지만 함수 호출을 할 때마다 다른 스레드에 일을 시키는건 현실적으로 말이 안된다. 스레드 한개에서 비슷하게 할 순 없을까? (코루틴 처럼)
* `sum(5)` 의 리턴값으로서 결과를 직접 반환하는게 아니라 다음과 같은 의미를 지니는 객체를 반환하면 어떨까? 그 객체는 `sum(4)` 의 결과를 인풋으로 받고 거기에 5를 더한 값을 아웃풋으로 하는 함수를 가진다. (즉, `result of sum(4)` => `result of sum(5)` 로의 함수)
* 그리고 외부에서 `sum(5)` 부터 `sum(1)` 을 그냥 독립적으로 실행한다음에 각 객체들의 함수를 이어 붙여 실행하면 어떨까? (`0` => `result of sum(1)` => ... => `result of sum(5)`)

위 생각들과 정확하게 일치하는 것은 아니지만 위에서 얻은 통찰들을 바탕으로 구체적인 패턴을 아래와 같이 만들어낼 수 있습니다.

```scala
// 함수를 추상화한 것입니다.
trait Function

// 값을 리턴함을 의미합니다.
case class Return(value: Int) extends Function
// 다른 함수의 호출을 의미합니다. 호출할 함수는 lazy 하게 얻어집니다.
case class LazyCall(getFunc: () => Function) extends Function
// 실제 로직을 의미합니다. 주어진 func 를 호출후 그 결과에 대해서 postFunc 를 실행하는 것을 의미합니다.
case class Logic(func: Function, postFunc: Int => Function) extends Function

// 함수를 해석/실행하여 값을 반환하는 역할을 합니다. (꼬리재귀를 만족함을 알 수 있습니다.)
@tailrec
def run(function: Function): Int =
  function match {
    case Return(value) => value
    case LazyCall(getFunc) => run(getFunc())
    case Logic(func, postFunc) =>
      // 논리적으로 func -> postFunc 순으로 실행되야합니다.
      // run(postFunc(run(func)) 처럼 하면 꼬리재귀를 만족시킬 수 없기 때문에 아래와 같이 해야합니다.
      func match {
        // Logic 안의 func 가 Return 인 경우는 리턴값에 대하여 postFunc 를 바로 호출해주면 됩니다.
        case Return(value) => run(postFunc(value))
        // Logic 안의 func 가 LazyCall 인 경우는 getFunc 를 호출하므로써 laziness 만 없애줍니다.
        case LazyCall(getFunc) => run(Logic(getFunc(), postFunc))
        // Logic 안에 Logic 이 있는 경우에는 두 개의 Logic 을 이어붙여주면 됩니다.
        case Logic(_func, _postFunc) =>
          // 논리적으로 _func -> _postFunc -> postFunc 순으로 실행되야합니다.
          run(Logic(
            _func,
            result => Logic(_postFunc(result), postFunc)
          ))
      }
  }

// 결과를 반환하는 것이 아니라 함수를 추상화한 객체를 반환합니다.
def sum(n: Int): Function =
  if (n == 1)
    Return(1)
  else
    Logic(
      // LazyCall 이 () => Function 을 인자로 받기 때문에 여기서 재귀호출이 발생하지 않습니다.
      LazyCall(() => sum(n - 1)),
      result => Return(n + result)
    )

def factorial(n: Int): Function =
  if (n == 1)
    Return(1)
  else
    Logic(
      LazyCall(() => factorial(n - 1)),
      result => Return(n * result)
    )

def fibo(n: Int): Function =
  if (n < 2)
    Return(n)
  else
    Logic(
      // 1. fibo(n - 1) 의 값을 먼저 구한다음, 
      LazyCall(() => fibo(n - 1)),
      firstResult =>
        Logic(
          // 2. 그다음으로 fibo(n - 2) 의 값을 구하고,
          LazyCall(() => fibo(n - 2)),
          // 3. 1번과 2번의 결과를 더한 값을 반환한다.
          secondResult => Return(firstResult + secondResult)
        )
    )
```

이렇듯 non-tail-recursive 한 함수를 stack safe 하게 만드는 일반화된 패턴이 있고 그 패턴을 **Trampoline** 이라고 부릅니다. <br/>
그러면 Trampoline 패턴을 쓰면 꼬리 재귀가 아닌 것이 꼬리 재귀가 되는 것일까요? `run` 함수가 꼬리 재귀 형태이긴 하지만 논리적으로는 느낌이 약간 다릅니다. <br/>
무슨 뜻인고하니, 처음에 소개한 `safeSum` 처럼 Trampoline 패턴을 쓰지않고 꼬리 재귀형태로 함수를 작성할 경우 컴파일러 최적화에 의해 스택을 계속 재활용하게 됩니다. 즉, 메모리 소모가 줄어든다는 것이죠. <br/>
반면에 Trampoline 패턴을 사용하게 되면 메모리 소모가 줄어드는 것은 아닙니다. 물론 `run` 함수가 꼬리 재귀이므로 스택은 똑같은걸 재활용하게 됩니다. 단, 대신에 계속해서 힙 메모리를 할당해 사용하게 됩니다. <br/>
**즉, Trampoline 패턴은 콜스택을 스택에 계속 쌓아올라가는 대신에 힙을 사용한다고 보시면 됩니다. 힙에 잘게 쪼개진 함수들이 여러 개 분포돼 있고 그 함수들을 (마치 트램펄린을 타듯이) 점프~ 점프~ 하면서 실행하는 것이죠!**

사실 Trampoline 패턴은 그리고 더 강력한 기능하고 넓은 의미를 가지고 있습니다.
A함수와 B함수가 번갈아가면서 호출되는 재귀 호출을 포함하여 임의의 함수호출에 대해 stack safety 를 보장해줍니다.
즉, 타입만 일치한다면 스택을 사용하지 않고 힙을 바탕으로 함수들이 서로를 호출할 수 있게 되는 것이죠. 
그리고 trampoline 을 coroutine 으로 생각할 수도 있습니다.

그리고 이러한 Trampoline 패턴은 이미 스칼라 표준 라이브러리에 구현되어있습니다. 
실제 내부 구현을 보면 핵심은 위에서 직접 구현한 것과 똑같습니다. 
다만 일반화가 잘되어 있고 네이밍이 좀 더 감성있고 monadic interface 를 통해 편의성이 좋습니다.
아래를 참고하시면 되겠습니다~

```scala
import scala.util.control.TailCalls._

def sum(n: Int): TailRec[Int] =
  if (n == 1) done(1)
  else tailcall(sum(n - 1).map(_ + n))

def factorial(n: Int): TailRec[Int] =
  if (n == 1) done(1)
  else tailcall(factorial(n - 1).map(_ * n))

def fibo(n: Int): TailRec[Int] =
  if (n < 2) done(n)
  else tailcall(
    for {
      firstResult  <- factorial(n - 1)
      secondResult <- factorial(n - 2)
    } yield firstResult + secondResult
  )

println(sum(100000).result)
```

Trampoline 에 대해 더 관심이 있으신 분은 [Stackless Scala With Free Monads](http://blog.higher-order.com/assets/trampolines.pdf) 를 읽어보시면 좋을 것 같습니다.

앞으로는 오늘처럼 그나마 좀 덜 고인 포스팅들을 자주 써보도록 하겠습니다. (고인물 포스팅은 써도 아무도 안봐서..ㅜㅜ)
