---
layout: post
title:  "Kotlin Basic"
date:   2019-01-10 21:00:00
author: klight1994
tags: [android, kotlin]

---

 이번 포스팅에서는 기존에 자바로 개발하던 안드로이드 프로젝트를 2017년부터 정식으로 구글에서 안드로이드 언어로 채택된 코틀린을 사용하여 개발해보고자 기본 문법을 공부한 내용에 대해서 다루고 있습니다.  목차는 다음과 같습니다.

## [ 목차 ]

- ##### Kotlin이란?

- ##### Kotlin의 장점

- ##### Kotlin 기본 문법



## Kotlin이란?

 Kotlin은 2011년에 IntelliJ를 만든 JetBrain이 공개한 프로그래밍 언어입니다. 안드로이드 개발 툴인 Android Studio는 IntelliJ를 기반으로 만들어진 프로그램입니다. 따라서 JetBrain이 만든 Kotlin을 완벽하게 지원합니다. 또한 2017년에 구글에서 안드로이드의 공식 언어로 채택되었습니다.



## Kotlin의 장점

- 간결한 문법

- 변수를 Nullable과 NotNull로 나눔 (Null Safety)

- Java 6에 호환됨

- Java와의 상호 운용이 100%지원되어 기존의 모든 안드로이드 라이브러리 사용가능 



## Kotlin 기본 문법

#### 변수와 상수

변수는 var 키워드로 선언하며, 상수의 경우 val로 선언합니다.

자료형을 지정하지 않아도 추론해주는 형추론을 지원하여 자료형을 생략할 수 있습니다. 
```kotlin
var a: Int = 10
val b: Int = 20
```

변수 선언 방법은 var 변수명: 자료형으로 선언하면 변수가 생성됩니다. 상수는 같은 방법으로 키워드만 val로 수정해주면 됩니다.

#### 함수

함수의 선언방법은 fun 함수명(인자명1 : 자료형, 인자명2:자료형  ) : 반환자료형으로 선언합니다. 예를 들면

``` kotlin
fun hello(userName: String) : Unit{ 
	println("hello "+userName)
}
```

위와 같이 선언할 경우 hello("ssm")을 사용하면 Hello ssm이 출력됩니다. 여기서 Unit형은 자바와 c++의 void을 의미합니다. 반환자료형을 생략할 경우 Unit형이 사용됩니다.

#### 자료형

| 키워드 | 자료형            |
| ------ | ----------------- |
| Double | 64비트 부동소수점 |
| Float  | 32비트 부동소수점 |
| Long   | 64비트 정수       |
| Int    | 32비트 정수       |
| Short  | 16비트 정수       |
| Byte   | 8비트 정수        |
| String | 문자열            |
| Char   | 하나의 문자       |

#### 문자열

문자열 비교는 ==을 사용하며 이는 Java의 equals와 같습니다.

Kotlin에서는 문자열 템플릿 기능을 제공하며, 복잡한 문자열을 표현할 때 편리합니다. 자바와 같이 + 기호로 문자열을 연결할 수도 있으며, 추가적으로 $ 기호를 사용하면 문자열 리터럴 내부에 변수를 쉽게 표현할 수 있습니다. 예를 들면 다음과 같습니다.

``` kotlin
fun hello(userName: String) : Unit{ 
	println("hello $userName") 
}
```

또한 여러 줄의 문자열을 표현할 때 큰 따옴표 3개를 리터럴로 사용하면 여러 줄에 걸친 문자열을 표현할 수 있습니다다. 예를 들면 다음과 같습니다.

```kotlin
var str = """동해물과 백두산이 마르고
닳도록 하느님이 보우하사
우리나라 만세
"""
```


#### 배열

배열은 Array라는 타입으로 표현합니다. 초기화는 arrayOf( ) 메서드를 사용하여 배열의 생성과 초기화를 함께 수행합니다. 

예를 들면 다음과 같습니다.
``` kotlin
var numbers : Array<Int> = arrayOf(1, 2, 3, 4, 5)
numbers[0]=6
```


#### 제어문

제어문에는 if, when, for, while문이 있으며, when을 제외하고는 자바와 c++ 문법과 비슷합니다.

##### if

if문은 if(조건문)과 같이 사용하며,  else if, else문도 Java, C++과 같습니다.

##### for

for문은 Java의 foreach문과 비슷하여 배열을 순회하면서 모든 요소를 볼 수 있습니다. 예를 들면
``` kotlin
val numbers = arrayOf(1,2,3,4,5) 
for(num in numbers){
	println(num)
}
```
과 같이 사용할 수 있으며, 증감문으로 사용할 수도 있습니다. 방법은 다음과 같습니다.
``` kotlin
for(i in 1..3){
	println(i)
}
```
#####  while

while문은 주어진 조건이 참일 때 반복되며, while(조건문)과 같이 사용하며, do-while문도 Java, C++과 동일합니다.

##### when

when문은 자바의 switch문과 비슷하지만, 값이 하나인 경우는 물론 콤마나 in 연산자를 이용하여 값의 범위를 자유롭게 지정하는 것이 특징입니다. 예를 들면 다음과 같습니다.

``` kotlin
val x = 1
when(x){ 
	1-> println("x == 1")
	2, 3 -> println("x == 2 or x ==3 ") 
	in 4..7 -> println("x>=4 && x<=7")
	!in 8..10 -> println("x<8 or x>10")
	else -> { 
        println("x>=8 && x<=10") 
    }
}
```


#### 클래스

Kotlin에서의 클래스는 자바와 역할은 같지만 더 간결하게 표현됩니다.

##### 1. 클래스 선언

``` kotlin
// 기본 생성자
class Person{
	
}
val person = Person() // 인스턴스 생성
```

##### 2. 생성자

``` kotlin
class Person{
    constructor(name: String){
        println(name)
    }
}
```

##### 3. 멤버 변수

멤버 변수는 아래와 같이 작성할 수 있으며, 접근 제한자는 public(생략 가능),  private, protected, internal이 있습니다.

``` kotlin
class Person{
    var name // public
    private var password
    protected var gene
    internal var idx
}
```

각 접근 제한자의 변수, 함수 공개 범위는 아래 표와 같습니다.

| 키워드    | 공개 범위                     |
| --------- | ----------------------------- |
| public    | 전체 공개                     |
| private   | 현재 파일 내부에서만 사용가능 |
| internal  | 같은 모듈 내에서만 사용가능   |
| protected | 상속받은 클래스에서 사용가능  |

##### 4. 상속

Kotlin에서 클래스는 기본적으로 상속이 금지되어 있습니다. 따라서 상속을 사용하기 위해서는 꼭 open 키워드를 클래스 선언 앞에 추가해야 합니다. 예를 들면 다음과 같습니다.

```kotlin
open class Animal{

}

class Dog: Animal(){
    
}

```

##### 5. 내부 클래스

내부 클래스는 외부 클래스에 대한 참조를 갖고 있는 클래스를 말합니다. 내부 클래스 선언에는 inner 키워드를 써줍니다. 예를 들면 다음과 같이 사용합니다. inner 키워드가 없다면 oxygen 값을 감소하는 등의 외부 클래스 변수를 변경할 수 없습니다.

```kotlin
class Animal{
    var oxygen
    inner class Foot{
        fun run(){
            if(oxygen>0)
            oxygen-=1
        }
    }
}
```

##### 6. 추상 클래스

추상 클래스는 미구현 멧드가 포함된 클래스를 말합니다. 추상 클래스 선언에는 abstract를 써줍니다.  추상 클래스는 직접 인스턴스화 할 수 없고, 다른 클래스가 이를 상속하여 미구현 메서드를 구현해야 합니다. 이는 자바와 동일합니다. 예를 들면 다음과 같이 사용합니다.

```kotlin
abstract class Champion{
    abstract fun skillQ()
}

class Teemo: Champion(){
    override fun skillQ(){
        println("실명다트!")
    }
}

// val user = Champion() => Error
val user = Teemo()

```



#### 인터페이스

##### 1. 인터페이스 선언

인터페이스는 미구현 메서드를 포함하며 이를 상속하는 클래스에서 구현됩니다. 추상 클래스와 비슷하지만 추상 클래스는 단일 상속만 되고, 인터페이스는 다중 상속이 가능합니다. 또한 미구현 메서드 뿐만 아니라 구현된 메서드를 포함할 수 있습니다. 이는 자바 8의 default 메서드와 같습니다.  예를 들면 다음과 같습니다.

```kotlin
interface Dog{
    fun run()
    fun bark() = println("멍멍")
}

```

##### 2. 인터페이스 구현

```kotlin
class WelshCorgi : Dog{
    override fun run(){
        println("아장아장")
    }
}
```

##### 3. 상속과 인터페이스 동시 구현

1에서 설명한대로 상속할 때 클래스는 단일 상속, 인터페이스는 다중 상속이 가능합니다. 예를 들면 다음과 같습니다.

```kotlin
open class Dog{
    
}
interface Eatable{
    fun eat()
}
interface Sleepable{
    fun sleep()
}

class WelshCorgi: Dog(), Eatable, Sleepable{
    override fun eat(){
        println("냠냠")
    }
    
    override fun sleep(){
        println("Zzz")
    }
    
}

val mini = WelshCorgi()
mini.run()
mini.sleep()
```



#### null 가능성

kotlin에서는 기본적으로 객체에서 null값을 허용하지 않습니다. null값을 허용하려면 별도의 연산자를 사용해야 하고, null을 허용한 자료형을 사용할 때도 별도의 연산자로 호출하여 Null에 대하여 안전합니다.

##### 1. null 허용 연산자 [ ? ] 

kotlin에서 null값을 허용해주기 위해서는 자료형 오른쪽에 ? 기호를 추가해주면 됩니다. 예를 들면 다음과 같습니다.

```kotlin
val a : String // 값이 설정되지 않아 에러가 발생함
val b : String = null // Kotlin에서는 기본적으로 null이 허용되지 않음
val c : String? = null // ?연산자를 사용하여 nullable함 => 에러 발생하지 않음
```

##### 2. lateinit 키워드로 늦은 초기화

초기화를 나중에 하게 되는 경우가 발생할 경우, lateinit 키워드를 변수 선언 앞에 사용하면 1의 a의 경우처럼 에러가 발생하지 않습니다. 하지만 초기화를 하지 않을 경우 null 예외로 프로그램이 종료될 수 있으니 주의해야 합니다.  예를 들면 다음과 같습니다.

```kotlin
lateinit var a : String // 가능

a = "hello"
println(a)
```

lateinit을 사용하려면 상수가 아니여야 합니다. 또한 null값으로 초기화할 수 없고, 초기화 전에는 변수를 사용할 수 없습니다.

##### 3. lazy 키워드로 늦은 초기화

2에서 var를 늦은 초기화를 했다면 lazy는 값을 변경할 수 없는 val를 늦은 초기화하는 키워드 입니다. val 선언 뒤에는 by lazy 블록에 초기화에 필요한 코드를 작성합니다. 마지막 줄에는 초기화 할 값을 작성합니다. 이렇게 작성할 경우 상수가 처음 호출될 때 초기화 블록의 코드가 실행되며 그 이후에는 초기화된 값만 호출됩니다. 예를 들면 다음과 같습니다.

```kotlin
val str : String by lazy {
    println("한 번만 실행되는 부분")
    "hello world"
}

println(str) // 한 번만 실행되는 부분; hello world
println(str) // hello world
```

##### 4. null값이 아님을 보증해주는 연산자 [ !! ]

변수 뒤에 !!를 추가하면 null값이 아님을 보증하게 됩니다. 따라서 nullable하지 않은 변수의 초기화에 사용합니다. 예를 들면 다음과 같습니다.

```kotlin
val name : String? = "홍길동"

val me: String = name // nullable하지 않은 변수에 nullable 변수를 넣었으므로 에러
val you : String? = name // 가능

val we : String = name!! // null값이 아님을 보증했으므로 가능
```

##### 5. 안전한 호출 연산자 [ ?. ]

메서드 호출 시에 . 연산자 대신에 ?. 연산자를 사용하면 null값이 아닌 경우에만 메서드가 호출됩니다. null값이면 null을 반환합니다.  따라서 null예외를 방지할 수 있습니다. 예를 들면 다음과 같습니다. 

```kotlin
val str: String? = null
var upperCase = if(str != null) str else null // upperCase = null
upperCase = str?.toUpperCase // upperCase = null
```

위 코드가 Java로 구현되었다면 str.toUpperCase를 호출하는 부분에서 NullException이 발생했을 것입니다.

##### 6. 엘비스 연산자 [ ?: ]

안전한 호출 시에 null값이 아닌 다른 값을 반환하고 싶다면 엘비스 연산자를 함께 사용합니다. 예를 들면 다음과 같습니다.

```kotlin
val str: String? = null
var upperCase = if(str != null) str else null // upperCase = null
upperCase = str?.toUpperCase ?: "초기화하세요" // upperCase = "초기화하세요"
```



#### 컬렉션

컬렉션은 자료구조 리스트와 맵과 집합을 말합니다. 알고리즘 문제 풀이와 개발 시에 매우 유용하게 사용됩니다.
( C++에서  STL vector, map, set에 대응됩니다)

##### 1. 리스트

리스트는 배열처럼 같은 자료형의 데이터를 순서대로 가지고 있는 자료구조입니다. 중복된 아이템을 가질 수 있으며, 추가, 삭제, 교체 등이 있습니다. C++ STL vector와 비슷합니다.

요소를 변경할 수 없는 읽기 전용 리스트는 listOf() 메서드로 작성합니다. 예를 들면 다음과 같습니다.

```kotlin
val foods: List<String> = listOf("볶음밥", "제육볶음", "김치찌개")
```

요소를 변경할 수 있는 리스트는 mutableListOf() 메서드를 사용하여 작성합니다. Java와 다른 점은 특정 요소에 접근할 때 배열과 같이 []연산자로 접근할 수 있다는 점입니다.  예를 들면 다음과 같습니다.

```kotlin
val foods = mutableListOf("볶음밥", "제육볶음", "김치찌개")

foods.add("햄")
foods.removeAt(0)
foods[1]="된장찌개"

println(foods)  // [제육볶음, 된장찌개, 햄]
println(foods[0]) // 제육볶음
```

##### 2. 맵

맵은 key와 value의 쌍으로 이루어진 키가 중복될 수 없는 자료구조입니다. 리스트와 마찬가지로 mapOf() 메서드 읽기 전용 맵을 만들 수 있고, mutableMapOf() 메서드로 수정이 가능한 맵을 만들 수있습니다. C++ STL map과 비슷하고, 예를 들면 다음과 같습니다.

```kotlin
val map = mapOf("a" to 1, "b" to 2, "c" to 3) //  읽기 전용

val citiesMap = mutableMapOf("한국" to "서울", "일본" to "동경", "중국" to "북경")

citiesMap["한국"] = "서울특별시" // value 수정
citiesMap["미국"] = "워싱턴" // 추가
```

##### 3. 집합

집합은 중복되지 않는 요소들로 구성된 자료구조입니다. 위의 두 자료구조와 마찬가지로 setOf() 메서드로 읽기 전용, mutableSetOf() 메서드로 수정가능한 집합을 생성합니다. C++ STL set과 비슷하고 예를 들면 다음과 같습니다. 

```kotlin
val citySet = setOf("서울", "인천", "경기") // 읽기 전용

val citySet2 = mutableSetOf("서울", "인천", "경기")
citySet2.add("대전")
citySet2.remove("인천")

println(citySet2.size) // 3
println(citySet2.contains("서울")) // true
```



### 마무리

 이번 과제에서는 Kotlin에 대한 간단한 소개와 문법에 대해서 알아보았습니다. Kotlin언어를 사용하여 클린 코드와 디자인 패턴에서 배운 내용을 기존에 개발했던 코드에 적용하거나 새로운 프로젝트를 작성하여 간결한 코드를 작성할 수 있는 발판이 되었습니다. 기회가 된다면 클린 코드와 디자인 패턴을 적용한 코드들을 다루면서 디자인 패턴과 클린 코드에 대해서 간단하게 소개해보도록 하겠습니다. 감사합니다.