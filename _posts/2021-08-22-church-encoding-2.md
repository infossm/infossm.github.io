---
layout: post
title: "람다 표현과 처치 인코딩(2)"
date: 2021-08-22 22:00:00
author: psb0623
tags: [mathematics]
---

[이전 글](http://www.secmem.org/blog/2021/06/21/church-encoding/)에 이어서, 자연수와 그 연산들을 어떻게 함수로 인코딩하는지 알아보도록 하겠습니다. 

## 자연수 인코딩(Church Numerals)

처치 인코딩에서는 다음과 같이 자연수를 정의합니다.

$$ 0 = \lambda f. \lambda x. x $$

$$ 1 = \lambda f. \lambda x. f \, x $$

$$ 2 = \lambda f. \lambda x. f \, (f \, x) $$

$$ 3 = \lambda f. \lambda x. f \, (f \, (f \, x)) $$

$$ \vdots $$

$$ n = \lambda f. \lambda x. f^{n} \, x $$

간단히 말하자면, 처치 인코딩에서 자연수 $n$은 함수 $f$와 $x$를 받아서 $x$에 $f$를 $n$번 적용한 값을 내놓는 함수입니다.

그저 함수의 합성과 적용을 대신 해줄 뿐인 함수가 자연수라니, 상당히 와닿지 않는 느낌입니다.
도대체 왜 저게 '자연수'인지에 대한 의문은, 곧 이들 사이의 연산을 정의하면서 자연스럽게 해소될 것입니다.

사실, 우리가 어떤 것을 자연수라고 부르는 데에 있어서, 각 원소의 직접적인 형태보다는 그 원소들이 이루는 대수적 구조가 더 중요합니다.
우리에게 친숙한 $0$, $1$, $2$, $\cdots$와 같은 숫자들 역시 덧셈, 뺄셈 등의 연산이 같이 정의되지 않는다면, 그저 의미없는 구부러진 기호에 불과한 것처럼요.

그러므로, 위의 새로운 자연수 개념은 일단 정의로서 받아들인다고 생각하고, 이 때 연산들은 어떻게 정의되는지에 초점을 맞춰보도록 합시다. 

## 다음 수(Successor)

가장 처음으로 정의해볼 연산은 어떤 수의 다음 수를 구하는 연산입니다.

그냥 보기엔 정말 시시해 보이는 연산입니다. 하지만 자연수의 정의가 0(혹은 1)부터 시작하여 '하나씩 더하여' 얻는 수들의 집합이라는 점에서,
어떤 수의 다음 수를 얻는 연산이야말로 자연수의 근본 성질을 꿰뚫는 근본 연산이라고 말할 수 있습니다.

그렇다면, 이 기본적인 연산을 함수로는 어떻게 인코딩할 수 있을까요?

먼저, $n$과 $n+1$을 써놓고 둘 사이의 관계를 찾아보도록 합시다.

$$ n = \lambda f. \lambda x. f^{n} \, x $$

$$ n+1 = \lambda f. \lambda x. f^{n+1} \, x $$

$$ = \lambda f. \lambda x. f \, (f^{n} \, x) $$

여기서,

$$ n \, f \, x = (\lambda f. \lambda x. f^{n} \, x) \, f \, x = f^{n} \, x $$

이므로 $f^{n} \, x$를 $ n \, f \, x $로 바꿔 쓸 수 있습니다. 따라서,

$$ n+1 = \lambda f. \lambda x. f \, (n \, f \, x) $$

이제, 우리는 $n$을 이용해서 $n+1$을 나타내는 법을 알았습니다.

이 사실을 활용해서 다음 수를 구하는 연산을 인코딩할 수 있습니다.
$n$을 받아서 $n+1$을 내놓는 함수($ = \lambda n.(n+1)$)의 형태로 말이죠. 

이 함수의 이름을 $\text{succ}$이라고 했을 때,

$$ \text{succ} = \lambda n.(n+1) $$

$$ = \lambda n.( \lambda f. \lambda x. f \, (n \, f \, x) ) $$

$$ = \lambda n. \lambda f. \lambda x. f \, (n \, f \, x) $$

이것이 정말 제대로 작동하는지 예시를 통해 확인해 봅시다. 

$$ \text{succ} \, 2 = (\lambda n. \lambda f. \lambda x. f \, (n \, f \, x)) \, 2 $$

$$ = \lambda f. \lambda x. f \, (2 \, f \, x) $$

$$ = \lambda f. \lambda x. f \, (f^{2} \, x) $$

$$ = \lambda f. \lambda x. f^{3} \, x = 3 $$

성공적으로 $\text{succ} \, 2$가 $2$의 다음 수인 $3$이 되는 것을 확인할 수 있습니다.

## 덧셈(Addition)

다음 수를 구하는 연습도 해 보았으니, 이제 우리에게 가장 친숙한 덧셈을 인코딩해볼 차례입니다. 

덧셈은 이항 연산이므로, 덧셈의 함수 인코딩은 두 수 $n$과 $m$을 받아 $n+m$을 내놓을 수 있어야 합니다.
우선 $n$, $m$과 $n+m$을 써놓고 이들 사이의 관계를 찾아봅시다.

$$ n = \lambda f. \lambda x. f^{n} \, x $$

$$ m = \lambda f. \lambda x. f^{m} \, x $$

$$ n+m = \lambda f. \lambda x. f^{n+m} \, x $$

여기서, 우리는 $ f^{n} \, ( f^{m} \, x ) = f^{n+m} \, x $라는 관계를 활용할 수 있습니다.
$x$에 $f$를 $m$번 적용한 후 또 다시 $f$를 $n$번 적용하는 것은 결국 $x$에 $f$를 $m+n$번 적용하는 것과 마찬가지기 때문이죠.

따라서, 

$$ n+m = \lambda f. \lambda x. f^{n+m} \, x  $$

$$ = \lambda f. \lambda x. f^{n} \, ( f^{m} \, x ) $$

$$ = \lambda f. \lambda x. n \, f \, ( f^{m} \, x ) $$

$$ = \lambda f. \lambda x. n \, f \, ( m \, f \, x ) $$

이제, 우리는 $n$과 $m$을 이용해서 $n+m$을 표현할 수 있게 됐습니다.
아까와 비슷하게, $n$과 $m$을 받아서 $n+m$을 내놓는 $\text{plus}$ 함수를 작성할 수 있습니다.

$$\text{plus} = \lambda n. \lambda m.(n+m) $$

$$ = \lambda n. \lambda m. \lambda f. \lambda x. n \, f \, ( m \, f \, x ) $$

### 다른 방법

위에서 인코딩한 방식이 유일한 방법은 아닙니다.
생각해보면, $n$에 $m$을 더한다는 것은 $n$에 $1$을 $m$번 더한다는 것과 같습니다. $n$의 다음 수를 $m$번 구하는 연산과 동일한 것이죠.
따라서, 아까 구했던 $\text{succ}$ 함수를 재활용해서 $n+m$을 아래와 같이 표현할 수 있습니다. 

$$ n+m = \text{succ}^{m} \, n = m \, \text{succ} \, n $$

이 때, $\text{plus}$ 함수는 아래와 같이 써지게 됩니다. 훨씬 간결해진 모습이네요.

$$ \text{plus} = \lambda n. \lambda m.(n+m) = \lambda n. \lambda m. m \, \text{succ} \, n $$ 

## 곱셈(Multiplication)

곱셈도 덧셈과 비슷한 과정을 통해 인코딩할 수 있습니다.

$n$, $m$, $n \cdot m$의 관계를 보면, 

$$ n = \lambda f. \lambda x. f^{n} \, x $$

$$ m = \lambda f. \lambda x. f^{m} \, x $$

$$ n \cdot m = \lambda f. \lambda x. f^{n \cdot m} \, x $$

여기서, 우리는 $ (f^{n})^{m} \, x = f^{n \cdot m} \, x $라는 관계를 활용할 수 있습니다.
$f$를 $n$번 적용하는 과정을 $m$번 반복하면 $f$가 총 $n \cdot m$번 적용되는 것이나 마찬가지이기 때문이죠.

$$ n \cdot m = \lambda f. \lambda x. f^{n \cdot m} \, x $$

$$ = \lambda f. \lambda x.(f^{n})^{m} \, x $$

$$ = \lambda f. \lambda x. m \, f^{n} \, x $$

여기서, $ n \, f = (\lambda f. \lambda x. f^{n} \, x) \, f = \lambda x. f^{n} \, x$ 이고,
$\lambda x. f^{n} \, x$는 $f^{n}$이나 다를 바 없기 때문에($x$를 받아서 $f(x)$를 내놓는 함수는 $f$ 그 자체이기 때문입니다.)
$f^{n}$을 $n \, f$로 바꿔 쓸 수 있습니다.

따라서, 

$$ n \cdot m = \lambda f. \lambda x. m \, (n \, f) \, x $$

이제, 우리는 $n$과 $m$을 이용해서 $n \cdot m$을 표현할 수 있게 됐습니다.
$n$과 $m$을 받아서 곱셈을 수행하는 $\text{mul}$ 함수는 아래와 같이 인코딩됩니다.

$$ \text{mul} = \lambda n. \lambda m.(n \cdot m) $$

$$ = \lambda n. \lambda m. \lambda f. \lambda x.  m \, (n \, f) \, x $$

### 다른 방법

덧셈을 다른 방법으로 구하는 방법과 비슷합니다. $n$과 $m$을 곱한다는 것은 0에 $n$을 $m$번 더하는 것과 동일하므로, 아래와 같이 쓸 수 있습니다.

$$ \text{mul} = \lambda n. \lambda m. (\text{plus} \, n)^{m} \, 0 = \lambda n. \lambda m.m \, (\text{plus} \, n) \, ( \lambda f. \lambda x. x ) $$

## 이전 수(Predecessor)

이제, 이 글에서 제일 복잡한 부분인 이전 수를 구하는 연산입니다.
언뜻 보기에는 다음 수를 구하는 것과 비슷하게 쉬울 것 같아 보이지만, 직접 해보면 정말 만만치 않다는 것을 알 수 있습니다.
일반적인 정수의 경우에는 덧셈과 뺄셈은 사실상 동일한 연산이지만, 자연수에서는 음수의 부재로 인해 덧셈과 뺄셈이 완전히 다른 연산이라 그렇습니다.
작은 수에서 큰 수를 뺄 수 없는 것은 덤입니다.

그럼, 이전 수를 구하는 연산을 만드는 방법을 생각해보도록 합시다.
이 때 $0$의 이전 수는 원래 정의되지 않지만, 편의상 $0$의 이전 수도 $0$이라고 합시다.

$n$과 $n-1$을 써놓고 봤을 때,

$$ n = \lambda f. \lambda x. f^{n} \, x $$

$$ n-1 = \lambda f. \lambda x. f^{n-1} \, x $$

$n$을 이용해서 $f^{n-1}$을 이끌어내기란 쉽지 않아 보입니다.
$f$의 역함수가 존재한다면 간단히 해결될 문제이지만, 모든 함수가 역함수를 가지는 것이 아니라서 불가능하다는 것을 알 수 있습니다.

접근을 바꿔보면, $n-1$은 $-1$에 $1$을 $n$번 더한 것, 즉 $-1$의 다음 수를 $n$번 구하는 과정과 동일합니다.
하지만 우리가 정의한 자연수에는 $-1$이 존재하지 않습니다.
그렇다면, 혹시 $-1$은 아니지만 $-1$의 역할을 하는 무언가를 찾을 수는 없을까요?

만약 $\text{succ} \, (-1^{\ast}) = 0$을 만족하는 임의의 원소 $-1^{\ast}$가 존재한다면,
비록 $-1^{\ast}$를 $-1$이라고 부를 수는 없지만

$$ n-1 = \text{succ}^{n-1} \, 0 = \text{succ}^{n} \, (-1^{\ast}) $$

처럼 $n$을 이용해서 $n-1$을 표현할 수 있게 됩니다.
그렇다면, 이제 남은 것은 아래의 등식을 만족하는 $n$, 즉 $-1^{\ast}$을 찾는 것입니다.

$$ \text{succ} \, n = \lambda f. \lambda x. f \, (n \, f \, x) = 0 = \lambda f. \lambda x. x $$

하지만, $n$에다가 어떤 것을 집어넣더라도 위 등식을 성립하게 할 수 없습니다. 그렇다면 방법은 없는 걸까요?

기존의 자연수 정의 아래에서는 불가능할 것이 분명하므로, 아래와 같은 새로운 함수들을 생각해봅니다. ($f$와 $x$는 이미 주어졌다고 가정합니다.)

$$ 0' = \lambda h.h \, x $$

$$ 1' = \lambda h.h \, ( f \, x ) $$

$$ 2' = \lambda h.h \, ( f^{2} \, x ) $$

$$ 3' = \lambda h.h \, ( f^{3} \, x ) $$

$$ \vdots $$

$$ n' = \lambda h. h \, ( f^{n} \, x ) $$

이들은 처치 인코딩의 자연수들과 비슷한 특징을 가지기는 하지만, 엄연히 다르며 자연수로 분류되지도 않습니다.
따라서 이들을 그저 $n$과 유사하다는 의미로 $n'$처럼 표기하도록 하겠습니다.

$n'$과 $n$의 관계는 아래에서 보듯이 간단합니다.

$I = \lambda x.x$인 항등함수라고 할 때,

$$ \lambda f. \lambda x. (n' \, I) = \lambda f. \lambda x.(\lambda h. h \, ( f^{n} \, x ) \, I ) $$

$$ = \lambda f. \lambda x. I ( f^{n} \, x )  = \lambda f. \lambda x. f^{n} \, x = n $$

또한, $n$에서 했던 것처럼 $n'$에서도 다음 수를 구하는 $\text{succ}'$ 함수를 생각할 수 있습니다.

$$ n' \, f =  \lambda h. h \, ( f^{n} \, x ) \, f = f \, (f^{n} \, x) = f^{n+1} \, x $$

이므로, 

$$ (n+1)' = \lambda h. h \, ( f^{n+1} \, x ) = \lambda h. h \, (n' \, f) $$

처럼 표현됩니다. 따라서 $n'$을 받아서 $(n+1)'$을 내놓는 $\text{succ}'$ 함수는 아래와 같이 작성됩니다.

$$ \text{succ}' = \lambda n' .(\lambda h .h \, (n' \, f )) $$

이렇게 정의된 $\text{succ}'$에서는 $\text{succ}' \, (-1^{\ast}) = 0'$인 $-1^{\ast}$을 찾을 수 있습니다.

$$ \text{succ}' (\lambda u.x) = \lambda n' .(\lambda h .h \, (n' \, f )) \, (\lambda u.x) $$

$$ = \lambda h .h \, (\lambda u.x \, f ) = \lambda h. h \, x = 0' $$

이 되기 때문에, $-1^{\ast} = \lambda u.x$이라고 할 수 있습니다. 여기서 $\lambda u.x$는 무엇을 받든 $x$를 내놓는 상수함수의 의미를 가집니다.

$-1^{\ast}$이 존재하므로, 우리는 아래와 같이 $(n-1)'$을 나타낼 수 있습니다.

$$ (n-1)' = (\text{succ}')^{n} \, (-1^{\ast}) = n \, \text{succ}' \, (\lambda u. x) $$

여기서 쓰인 $n$은 위에서 임시로 도입한 것이 아닌, 함수를 $n$번 합성해주는 기존의 자연수라는 점에 유의하시기 바랍니다.
마지막으로 위에서 구한 $n$과 $n'$의 관계를 이용하면,

$$ n-1 = \lambda f. \lambda x. ((n-1)' \, I) = \lambda f. \lambda x. ( n \, \text{succ}' \, (\lambda u. x) ) \, I $$

$$ = \lambda f. \lambda x. ( n \, (\lambda n' .(\lambda h .h \, (n' \, f ))) \, (\lambda u. x) ) \, (\lambda x.x ) $$

좀 더 깔끔히 써본다면,

$$ n-1 = \lambda f. \lambda x. n \, (\lambda g .\lambda h .h \, (g \, f )) \, (\lambda u. x) \, (\lambda x.x ) $$

처럼 될 것입니다. 긴 과정을 통해, 드디어 $n-1$을 $n$을 이용해서 나타낼 수 있게 되었습니다.

따라서 이전 수를 구하는 $\text{pred}$ 함수는 아래와 같이 써집니다.

$$ \text{pred} \, n = \lambda n. (n-1) = \lambda n.\lambda f. \lambda x. n \, (\lambda g .\lambda h .h \, (g \, f )) \, (\lambda u. x) \, (\lambda x.x ) $$

위의 과정에 따르면, $\text{pred}$ 함수는 $0$이 아닌 모든 자연수들에 대해 정상적으로 이전 수를 구하게 될 것입니다.
그러나 이 함수에 $0$이 들어갔을 때가 문제인데, 값을 구해보면

$$ \text{pred} \, 0  = \lambda f. \lambda x. ( -1^{\ast} \, I)  $$

$$ = \lambda f. \lambda x. (\lambda u. x) \, (\lambda x. x) = \lambda f. \lambda x. x = 0$$

으로, 위에서 $0$의 이전 수를 $0$으로 간주하기로 한 것과 정확히 일치합니다.
따라서, 우리는 우리가 원했던 이전 수를 구하는 함수를 완벽히 찾아냈습니다.

## 뺄셈(Subtraction)

위의 $\text{pred}$ 함수를 이용해서 뺄셈 역시 만들어낼 수 있습니다.

$n$에서 $m$을 빼는 것은 $n$의 이전 수를 $m$번 구하는 것과 동일합니다. 따라서 빼기 함수 $\text{minus}$는 아래와 같이 써집니다.

$$ \text{minus} = \lambda n. \lambda m. \text{pred}^{m} \, n =  \lambda n. \lambda m. m \, \text{pred} \, n $$

이 때, $n$보다 $m$이 더 크다면, $0$의 이전 수를 $0$으로 생각하는 $\text{pred}$ 함수의 특성 상 $0$이 나오게 될 것입니다.
그 외의 경우에는 $n-m$의 값을 정확히 구해줄 것입니다.

## 마치며

이번 글에서는 처치 인코딩의 자연수들을 소개하고 덧셈, 뺄셈 등 이들 사이의 기본적인 연산들을 유도해 보았습니다.
정말 복잡한 과정이었지만, 우리가 알던 자연수와 연산들을 함수만 이용해서 표현할 수 있다는 것이 흥미롭지 않나요?

이 주제에 대해서는 자세히 설명된 한글 자료가 부족한 것 같아서 제 나름의 언어와 방식으로 열심히 설명해 보았는데, 이해하기 쉽게 잘 전달되었을지 모르겠네요.
이 내용에 대해 더 자세히 알고 싶으시거나, 처치 인코딩이 자연수를 넘어서 얼마나 더 확장될 수 있는지 궁금하시다면 아래 링크에 쓰인 내용을 참고하시면 좋을 것 같습니다.

마지막으로, 긴 글 읽어주셔서 감사합니다!

## References

[https://en.wikipedia.org/wiki/Church_encoding](https://en.wikipedia.org/wiki/Church_encoding)
