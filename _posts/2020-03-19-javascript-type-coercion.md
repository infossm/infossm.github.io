---
layout:     post
title:      "JavaScript의 형변환"
date:       2020-03-19 17:05
author:     evenharder
image:      /assets/images/evenharder-post/js-type/pexels-pixabay-glass-302743.jpg
tags:
  - Javascript
  - cheatsheet
---

![](/assets/images/evenharder-post/js-type/js.png)

JavaScript만큼 프로그래머들이 농담 따먹기를 하는 프로그래밍 언어가 PHP 말고 있을까요?

JavaScript의 매력이자 악명 높은 점이 타입의 유연성입니다. 변수를 선언할 때 타입을 지정할 필요가 없으며, 서로 타입이 다른 변수들끼리 연산을 해야 할 때 최대한 에러를 내지 않는 방향으로 진행이 됩니다. 이런 규칙을 통해 `![]+-*`만 이용해서 0부터 1000까지 만들라는 [이런 문제](https://ipsc.ksp.sk/2015/real/problems/m.html)도 있습니다. 하지만 일반적으로 이런 변환 과정은 갸우뚱할 때가 많으며, 프로그래머 개그의 단골 소재이기도 합니다.

이 포스트에서는 형변환(coercion)에 깔려있는 법칙들을 설명하고자 합니다. [Alexey Samoshkin님의 이 포스트](https://www.freecodecamp.org/news/js-type-coercion-explained-27ba3d9a2839/)에서 지대한 영향을 받았으며, 기본적으로 해당 포스트의 흐름을 따라 번역하되 내용과 예제를 보완하며 작성하였습니다.

그럼 시작 전에 다음 표현식이 어떻게 될지 생각해보시기 바랍니다.

```javascript
4 + 10 + "string"
"string" + 4 + 10
"true" == true
undefined == ''
8 * null
0 == "\n"
!![]+!!{}+!!"false"
~undefined
[2] > "1"
"hello" > 3
"hello" < 3
"-1" > "+1"
"-1" > +1
"b" + "a" + + "a" + "a"
[] + undefined + 1
[2,3,5] == [2,3,5]
{}+[]+{}+[1]
!+[]+[]+![]
!+[]+![]+[]
[1] + [2,3]
```

## 기본 이론

형변환은 명시적일 수도 있고, 암시적일 수도 있습니다. **명시적 형변환(explicit coercion)**은 `Number("123")` 처럼 프로그래머의 코드에서 암시적으로 자료형을 정해서 변환하는 과정입니다. **암시적 형변환(implicit coercion)**은 연산자 사용으로 인해 자연적으로 일어나는 형변환입니다. 대표적인 예시로 `==`, `+`, `>` 등 연산자의 사용이 있습니다. 예외적으로 `===`는 형변환을 야기하지 않습니다. 이 암시적 형변환을 잘 이용하면 더욱 가독성 있는 코드를 작성할 수 있지만 잘못 생각하면 프로그램의 버그가 될 수 있습니다.



JavaScript에서의 형변환은 세 가지가 있습니다.

+ `String`으로 형변환
+ `Number`로 형변환
+ `Boolean`으로 형변환

또, 원시 타입과 객체(object)에 대해 형변환이 다르게 적용됩니다. 각자 알아보려고 합니다.

## JavaScript의 원시 타입 형변환

### String conversion

명시적 형변환은 `String()` 함수를 쓰면 됩니다. 암시적 형변환은  `+` 연산자를 사용할 때 피연산자에 `String`이 있을 때 일어납니다.

String으로의 변환은 자연스럽습니다. 출력되는 형태 그대로 변환되기 때문입니다.

```javascript
String(12345)                   // "12345"
String(-3.14)                   // "-3.14"
String(true)                    // "true"
String(false)                   // "false"
String(undefined)               // "undefined"
String(null)                    // "null"
String(BigInt(42))              // "42"
```

`Symbol`은 암시적 형변환이 되지 않기 때문에 명시적 형변환을 해야 합니다.

```javascript
String(Symbol("Explicit"))      // "Explicit"
"and..." + Symbol("implicit")   // TypeError
```

### Boolean conversion

명시적 형변환을 하려면 `Boolean()`을 호출하면 됩니다. 암시적으로는 <code class="highlighter-rouge">&#124;&#124;</code>, `&&`, `!`에 의해 일어납니다. <code class="highlighter-rouge">&#124;&#124;</code>와 `&&`는 조건에 맞는 실제 피연산자를 반환하지만, 내부적으로는 형변환이 일어납니다.

`Boolean`형에는 `true`와 `false`밖에 없기 때문에, 거짓값(falsy value)를 열거하는 게 낫습니다. `''`, `0`, `NaN`, `null`, `undefined`, `false`, `BigInt(0)`가 전부입니다. 나머지(객체, `Date`, 리스트, 함수 등등) 는 전부 `true`로 변환됩니다.

### Number conversion

명시적 형변환을 하려면 `Number()`를 호출하면 됩니다. 암시적으로는 좀 많이 불립니다.

+ 비교 연산자 (`>`, `<`, `<=`, `>=`, `!=`, `==`) (단, 두 피연산자가 모두 `String`일 때는 제외)
+ 비트 연산자 (<code class="highlighter-rouge">&#124;</code>, `&`, `^`, `~`)
+ 산술 연산자 (`-`, `+`, `*`, `/`, `%`) (단, `+`의 연산자에 `String`이 있을 때는 제외)
+ 단항 연산자 (`+`)

변환하는 과정은 조금 복잡합니다.

+ `String`의 경우, 앞뒤 whitespace를 제외하고 빈 문자열이면 `0`으로, `Number`로 변환될 수 있으면 해당 `Number`로 (`Infinity`, `1e9` 등), 아니면 `NaN`으로 변환됩니다.
+ `null`은 `0`으로, `undefined`는 `NaN`으로 변환됩니다.
+ `Symbol`은 명시적으로도 암시적으로도 변환될 수 없으며 `TypeError`를 야기합니다.
+ `null`이나 `undefined`는 `==`에서 형변환이 일어나지 않으며, `null`과 `undefined`가 `==` 연산자에서 `true`가 되는 경우는 이 두 가지 밖에 없습니다.
+ `NaN`은 `!==` 연산자로도 `false`가 나옵니다.
+ `BigInt`는 명시적으로밖에 변환하지 못하며, 암시적 변환은 `TypeError`를 야기한다.
+ 변환은 아니지만, 비트 연산에서 `Infinity`, `-Infinity`, `NaN`은 `0`으로 취급됩니다.

## JavaScript의 object 형변환

그럼 `[1] + [2,3]` 같은 건 어떻게 적용되는 걸까요?

우선 JavaScript 엔진은 객제를 원시 타입으로 바꾸려는 시도를 합니다. 그리고 가능한 변환은 `String`, `Number`, `Boolean`밖에 없습니다. `Boolean`의 경우 앞서 말했듯이 무조건 `true`로 변환됩니다. 그 외로는 `[[ToPrimitive]]` 메서드를 이용해  변환되는데, 과정이 대략 다음과 같습니다.

+ `[[ToPrimitive]]` 메서드에 `preferredType`을 넘겨서 변환하고자 하는 형(`Number`나 `String`)을 명시할 수 있습니다 (필수는 아님).
+ `Number` 로 변환하든 `String`으로 변환하든 `Object.prototype`의 `valueOf`랑 `toString`을 사용하며, 임의의 object에 존재합니다.
+ 원시 타입이 입력으로 들어오면 그 입력을 그대로 반환합니다.
+ 두 경우 모두 `valueOf`와 `toString`을 기본적으로 호출하고, 그 결과가 원시 타입이면 이 값을 반환합니다.
  + `Number`로 변환하고자 하면 `valueOf`를 `toString`에 앞서, `String`으로 변환하고자 하면 반대로 `toString`을 `valueOf`에 앞서 호출합니다.
+ 이러고도 원시 타입이 나오지 않으면 `TypeError`를 반환합니다.

많은 내장 객체들이 `valueOf`가 정의되어 있지 않거나 (원시 타입이 아닌) `this`를 반환하는 경우가 많기 때문에, 어느 형변환을 하든 결과적으로 `toString`을 호출하게 됩니다.

각 연산자마다 `preferredType`을 지정해서 호출하지만, `==`와 `+`는 `preferredType`에 `default`를 넘깁니다. 이 경우 `Date`를 제외한 타입은 `Number`로 변환됩니다.

## 예제

```javascript
4 + 10 + "string"               // "14string"
"string" + 4 + 10               // "string410"
"true" == true                  // false
undefined == ''                 // false
8 * null                        // 0
0 == "\n"                       // true
!![]+!!{}+!!"false"             // 3
~undefined                      // -1
[2] > "1"                       // true
"hello" > 3                     // false
"hello" < 3                     // false
"-1" > "+1"                     // true
"-1" > +1                       // false
"b" + "a" + + "a" + "a"         // "baNaNa"
[] + undefined + 1              // undefined1
[2,3,5] == [2,3,5]              // false
{}+[]+{}+[1]                    // "0[object Object]1"
!+[]+[]+![]                     // "truefalse"
!+[]+![]+[]                     // "1"
[1] + [2,3]                     // "12,3"
```

하나하나 분석해보도록 하겠습니다.

`4 + 10 + "string"`에선 `4 + 10`이 먼저 계산되어 `14`가 되고, 이후 `14 + "string"`이 되어 `"14string"`이 됩니다.

`"string" + 4 + 10`에선 `"string" + 4`가 먼저 계산되어 `"string4"`가 되고, 이후 `"string4" + 10`이 계산되어 `"string410"`이 됩니다. `+`에 `String`이 들어가면 계속 `String`이라 보시면 됩니다.

`"true" == true`에서,  `==` 에 의해 numeric conversion이 일어나 `"true"`가 `NaN`이 되고 `true`는 `1`이 됩니다. 때문에 전체 식은 `false`가 됩니다.

`undefined == ''`에선, `==`에 `undefined`가 있기 때문에 numeric conversion이 일어나지 않습니다. 전체 식은 `false`가 됩니다.

`8 * null`에선 `null`이 `0`으로 변환되어 전체 식이 `0`이 됩니다.

`0 == "\n"`에선 numeric conversion이 일어나 `"\n"`이 `0`으로 변환됩니다. 때문에 전체 식은 `true`가 됩니다.

`!![]+!!{}+!!"false"`에선 `!!~sth~`의 `~sth~`가 `Boolean`으로 `true`로 변환되므로, 두 번 complement를 해 `true + true + true`가 됩니다. 이후는 numeric conversion이 일어나 3이 됩니다.

`~undefined`는 `undefined`가 `NaN`으로 형변환되고, 비트 연산에서 `NaN`이 0으로 간주되기 때문 `~NaN`은 `-1`로 계산됩니다.

`[2] > "1"`의 경우, numeric conversion이 일어나 `[2]`는 `valueOf` 메서드에 의해 `2`로, `"1"`은 `1`로 변환되기 때문에 `true`가 됩니다.

`"hello" > 3`과 `"hello" < 3`에서 `"hello"`는 `NaN`으로 변환되기 때문에 비교 결과도 둘 다 `NaN`이 됩니다.

`"-1" > "+1"`는 conversion이 일어나지 않습니다. `-`의 ASCII 코드(45)가 `+`보다 크므로(43), `true`가 됩니다.

`"-1" > +1`는 타입이 일치하지 않으므로 numeric conversion이 일어납니다. 결과는 `-1 > 1`이 되어 `false`입니다.

`"b" + "a" + + "a" + "a"`는 서두의 그림에 있던 예시입니다. 흐름을 표현하면 다음과 같습니다.

```javascript
>> "b" + "a" + + "a" + "a"
 - ("b" + "a") + + "a" + "a"
 - ("ba" + (+ "a")) + "a"
 - ("ba" + NaN) + "a"
 - "baNaN" + "a"
 - "baNaNa"
```

`[] + undefined + 1`에서 우선 `[] + undefined`이 계산됩니다. numeric conversion에 의해`[].valueOf()`가 호출되는데, 이는 자기 자신이므로 원시 타입이 아니어서 numeric conversion이 실패합니다. `Number([])`이 `0`임에도 불구하고 `[]`가 `object`이기 때문에 그렇습니다. 때문에 string conversion이 일어나고 이 땐 `"" + "undefined"`가 되어 `"undefined"`가 됩니다. 이후 결과는 당연히 `"undefined1"`이 됩니다.

`[2,3,5] == [2,3,5]`는 타입이 같아서 형변환이 일어나지 않고, 둘이 같은 객체가 아니므로 `false`가 됩니다. 이와 달리 `[2,3,5] == "2,3,5"`는 string conversion이 일어나므로 `true`가 됩니다.

`{}+[]+{}+[1]`는 원 포스트 최상단에 있는 예제인데, 약간의 낚시가 들어가 있습니다. 우선, 첫 중괄호 (`{}`)는 scope로 인식되어 연산에 아무런 영향이 없습니다. 실제 연산은 `+[]`부터 시작합니다.

```javascript
>> +[]+{}+[1]
 - +''+{}+[1]       // numeric conversion에서 []이 toString에 의해 ''로 변환
 - +0+{}+[1]        // 이후 ''이 (계속된 numeric conversion에 의해) 0으로 변환
 - 0+{}+[1]         // {}이 toString에 의해 "[object Object]"로 변환
 - "0[object Object]" + [1]
 - "0[object Object]1"
```

`!+[]+[]+![]`도 위랑 비슷합니다. `!+[]`는 `!0`이 되므로 `true`가 되며, `true+[]`에서 `[]`는 `''`으로 변환되어 `"true"`가 됩니다. 이후 나머지도 `String`이 되기에 결과적으로 `"truefalse"`가 됩니다.

`!+[]+![]+[]`는 비슷하지만 약간 다릅니다. `!+[]`가 `true`, `![]`는 `false`이므로 둘이 더해서 1이 되며, 이후 `[]`가 `''`으로 변환되어 결과적으로 `"1"`이 됩니다.

`[1] + [2,3]`에선 `valueOf`가 원시 타입을 반환하지 않으므로 `toString`이 사용되어 `"1" + "2,3"`이 되기에 `"12,3"`이 됩니다.

## 참고 자료

+ [JavaScript type coercion explained](https://www.freecodecamp.org/news/js-type-coercion-explained-27ba3d9a2839/) : 이 글의 기반이 된 포스트입니다.
+ [JavaScript Equality Table](https://dorey.github.io/JavaScript-Equality-Table/) : `==`에 대한 진리표입니다.
+ [wtfjs.com](https://wtfjs.com/) : JS의 고약한 예제들이 있습니다.
+ [Eloquent JavaScript](https://eloquentjavascript.net/) : 좋은 JS 입문서입니다. 웹 위주가 아닌 언어 위주로 설명되어 있습니다.

