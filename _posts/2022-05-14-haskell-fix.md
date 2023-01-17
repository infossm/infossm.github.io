---
layout: post
title: Haskell fix 함수
author: buttercrab
date: 2022-05-14
tags: [haskell]
---

# 서론

하스켈에는 다음과 같은 함수가 있습니다.

```haskell
fix :: (a -> a) -> a
fix f = let x = f x in x
```

`Data.Function` 에 위치한 `fix` 함수는 자세하게 들여다 보면 신기한 특성을 가지고 있습니다.
함수를 조금씩 풀어봅시다.

```haskell
fix f -> let x = f x in x
      -> let x = f x in f x
      -> let x = f x in f (f x)
      -> let x = f x in f (f (f x))
      -> let x = f x in f (f (f (f x)))
      -> ...
```

처음 보았을 때는 무한히 재귀적으로 실행이 되는 것처럼 보입니다.
또한 초기값은 무엇일까요?
하스켈은 lazy evaluation을 하기 때문에 이러한 문제들이 발생하지 않습니다.

# Lazy evaluation

[Lazy evaluation](https://en.wikipedia.org/wiki/Lazy_evaluation)은 하스켈의 큰 장점 중 하나입니다.
값을 항상 계산하는 것이 아니라 값이 필요로 할 때에 그 값을 계산하게 됩니다.
다음 예제를 봅시다.

```haskell
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
```

위 코드는 무한한 피보나치 수열을 담고 있는 리스트입니다.
이것은 lazy evaluation 덕분에 가능합니다.
하스켈은 위 코드가 실행 되었을 때 실제로 실행을 시키지 않습니다.
아직은 `fibs`의 값을 몰라도 다른 결과에는 변함이 없기 때문이죠.

이제 다음과 같이 `fibs`의 값을 사용하려고 할 때 저 식을 계산하게 됩니다.

```haskell
print $ fibs !! 4
```

`fibs !! 4`은 리스트에서 4번째 원소를 가져오는 코드입니다.
코드가 어떻게 실행되는지 한단계씩 따라가 보겠습니다.

```haskell
-- zipWith definition
zipWith :: (a->b->c) -> [a]->[b]->[c]
zipWith f = go
  where
    go [] _ = []
    go _ [] = []
    go (x:xs) (y:ys) = f x y : go xs ys

-- calculation by step
fibs !! 4 -> (0 : 1 : zipWith (+) fibs (tail fibs)) !! 4
          -> (1 : zipWith (+) fibs (tail fibs)) !! 3
          -> (zipWith (+) fibs (tail fibs)) !! 2
          -> (zipWith (+) 
              (0 : 1 : zipWith (+) fibs (tail fibs)) 
              (1 : zipWith (+) fibs (tail fibs))) !! 2
          -> ((+) 0 1 : zipWith (+) 
              (1 : zipWith (+) fibs (tail fibs)) 
              (zipWith (+) fibs (tail fibs))) !! 2
          -> (zipWith (+) 
              (1 : zipWith (+) fibs (tail fibs)) 
              (zipWith (+) fibs (tail fibs))) !! 1
          -> ((+) 1 1 : zipWith (+) 
              (zipWith (+) fibs (tail fibs))
              (zipWith (+) 
                (1 : zipWith (+) fibs (tail fibs)) 
                (zipWith (+) fibs (tail fibs)))) !! 1
          -> (zipWith (+) 
              (zipWith (+) fibs (tail fibs))
              (zipWith (+) 
                (1 : zipWith (+) fibs (tail fibs)) 
                (zipWith (+) fibs (tail fibs)))) !! 0
          -> ((+) 1 2 : ...) !! 0
          -> 3
```

이렇게 계산을 진행해나갈 때, 값을 필요한 부분만 계산을 해나갑니다.
위처럼 lazy evaluation은 다른 프로그래밍 언어에서는 쉽게 볼 수 없는 기능이고 이 기능은 하스켈을 강력하게 만들어줍니다.

# `fix`의 간단한 예시

그럼 `fix` 함수의 간단한 예시를 들어볼까요?
Lazy evaluation을 통해서 실제로 무한 재귀를 하지 않는 예시들을 살펴봅시다.

```haskell
-- definition of const
const x _ = x

fix $ const "hello" -> "hello"

-- calculation by step
fix $ const "hello" -> let x = const "hello" x in x
                    -> let x = const "hello" x in const "hello" x
                    -> "hello"
```

실제 예시를 보면 `fix`가 유용하게 쓰일 수 있을까라는 의심이 사라지게 될 것입니다.
그럼 이제 더 자세하게 `fix` 함수를 알아봅시다.

# Bottom 값

하스켈에는 [bottom](https://wiki.haskell.org/Bottom)이라는 값이 존재합니다.
이 값은 어떤 식이 성공적으로 끝나지 않았을 때의 값을 의미합니다.
에러가 발생하거나 무한히 실행되어 값이 나오지 않는 경우, 프로그램이 종료되는 경우 등의 경우가 있습니다.
Bottom 값은 수학적으로 $\bot$이라고 표현합니다.

Bottom 값은 성공적으로 끝나지 않았기 때문에 일반적인 함수들에 대해서 다음과 같이 작용합니다.

예를 들어

$$f(x) = x + 1$$

이면어

$$f(\bot) = \bot$$

이 됩니다.

# fixpoint와 `fix` 함수

`fix` 함수는 왜 이름이 fix 일까요?
그 이유는 lazy evaluation에 의해서 `x = f x`, 즉 $x = f(x)$ 이면 재귀가 종료됩니다.

수학적으로 그 의미는 $y = f(x)$ 와 $y = x$의 교차점, 즉 fixpoint를 의미합니다.
함수 이름인 `fix`는 fixpoint에서 나왔습니다.

그러면 첫 예제부터 살펴볼까요?

```haskell
(const "hello") x == x
```

인 x는 무엇일까요?

```haskell
(const "hello") "hello" == "hello"
```

`"hello"`가 fixpoint가 됩니다.
즉, `fix $ const "hello"`의 값은 `const "hello"`의 fixpoint인 `"hello"`가 된 것입니다.

다음 함수에서의 fixpoint도 찾아봅시다.

```haskell
f x = x + 1
```

일반적인 수라면 없을 것 같습니다.
하지만 위에서 나온 bottom값을 생각해볼까요?

$$f(\bot) = \bot$$

즉, $\bot$이 fixpoint가 됩니다.
한 번 계산을 직접 해볼까요?

```haskell
fix (+1) = let x = (+1) x in x
         = let x = (+1) x in (1 + x)
         = let x = (+1) x in (1 + (1 + x))
         = ...
```

무한히 재귀를 돌게 되는 것을 알 수 있습니다.
무한히 돌게 되므로 결과적으로 함수의 값은 bottom 값이 됩니다.

이처럼 $f(\bot) = \bot$이면 `fix f`의 값은 $\bot$이 됨을 알 수 있습니다.

# `fix` 함수로 만드는 재귀함수

사실 지금까지는 `fix` 함수가 무엇인지를 알아보았지만 유용하게 쓰일 것 같지는 않습니다.
하지만 `fix` 함수로 인해서 익명의 재귀함수를 만들 수 있게 해줍니다.

다음 함수를 봅시다.

```haskell
fix (\rec n -> if n == 0 then 1 else n * rec (n - 1))
```

fix는 인자로 들어온 익명 함수의 첫 번째 인자로 자기 자신을 넣어주므로 익명 함수의 첫 번째 인자는 자기 자신이 됩니다.
즉, 재귀 함수처럼 구현이 가능한 것입니다.
이 사실을 알고 위 함수를 다시 보면 간단하게 팩토리얼을 구하는 함수임을 알 수 있습니다.

기존에는 다음과 같은 방법으로 정의를 할 수 있었습니다.

```haskell
fact n = if n == 0 then 1 else n * fact (n - 1)
```

하지만 `fix`를 사용함으로써 익명함수에서도 재귀함수를 사용할 수 있게 되었습니다.

몇 가지 예시를 더 볼까요?

```haskell
fix (\rec n -> if n <= 1 then n else rec (n - 1) + rec (n - 2))
```

# fixpoint 관점에서의 재귀함수

그러면 앞서 이야기해본 fixpoint 관점으로 재귀함수는 어떤 의미를 가지고 있을까요?
다음 예시를 생각해봅시다.

```haskell
fix (\rec n -> if n == 0 then 1 else n * rec (n - 1))
```

먼저 haskell은 curry를 통해 익명 함수의 인자 2개의 함수를 인자 1개 함수에 반환값이 인자가 1개인 함수로 바꾸어 줍니다.
즉, 반환값이 함수이므로 함수의 입력과 출력이 둘 다 팩토리얼 함수가 됩니다.
팩토리얼 함수로 같아 함수가 fixpoint가 된다고 말할 수 있습니다.

# 더 읽으면 좋은 글들

- [Haskell/Fix and recursion](https://en.wikibooks.org/wiki/Haskell/Fix_and_recursion)
- [Fixpoints in Haskell](https://cdsmithus.medium.com/fixpoints-in-haskell-294096a9fc10)
- [Grokking Fix](https://www.parsonsmatt.org/2016/10/26/grokking_fix.html)
