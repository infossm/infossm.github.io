---
layout: post
title: "Kotlin 변성"
date: 2024-06-21 20:00:00
author: knon0501
tags: [Kotlin]
---

이 포스트에서는 코틀린 제네릭의 주요 개념 중 하나인 변성에 대해 알아보겠습니다. 변성이란 List\<String\>와 List\<Any\>와 같이 기저 타입이 같고 타입 인자가 다른 여러 타입이 서로 어떤 관계가 있는지 설명하는 개념입니다. List\<Any\> 타입의 파라미터를 받는 함수에 List\<String\>을 넘기면 안전한지에 대한 질문을 생각해봅시다. 결론부터 말하면 안전합니다. 예를 들어 다음과 같은 코드는 정상적으로 컴파일됩니다,

```Kotlin
fun print(list: List<Any>){
    println(list.joinToString())
}
print(listOf("a","b","c"))
```


그러나 MutbleList의 경우는 그렇지 않습니다.

```Kotlin
fun add(list: MutableList<Any>) {
    list.add(42)
}
val strings = mutableListOf("a","b","c")
add(strings)
println(strings.maxBy{it.length}) 
```
만약 이 코드가 컴파일 된다면 실행시점에 예외가 발생합니다. 따라서 코틀린 컴파일러는 이러한 함수 호출을 금지합니다.
왜 List는 안전하지만 MutableList는 안전하지 않을까요? List는 원소를 변경,삭제,추가할 수 없지만 MutableList는 가능하기 때문입니다. 

## 공변성

List\<T\>를 생각해봅시다. A가 B의 하위 타입일 때 List\<A\>가 List\<B\>의 하위 타입이면 List는 공변적입니다. 코틀린에서 제네릭 클래스가 타입 파라미터에 대해 공변적임을 표시하려면 타입 파라미터 이름 앞에 out을 넣어야 합니다.

```Kotlin
interface List<out T> : Collection<T> {

}
```

공변성을 만족하려면 앞서 언급했듯이 원소를 변경,삭제,추가하면 안 되며 이를 위해 타입 파라미터 T는 메서드의 아웃 위치에서만 사용되어야 합니다. 
T가 함수의 반환 타입에 쓰인다면 T는 아웃(out) 위치에 있으며 T가 함수의 파라미터 타입에 쓰인다면 T는 인(in) 위치에 있습니다.
다음과 같은 함수의 반환 타입은 아웃 위치입니다.

```Kotlin
interface List<out T> : Collection<T> {
    operator fun get(index: Int): T
}
```

MutableList\<T\>는 T가 인과 아웃 위치에 둘 다 쓰이기 때문에 공변적인 클래스로 선언할 수 없습니다.

생성자 파라미터는 인스턴스를 생성한 뒤에 호출될 일이 없기 때문에 인이나 아웃 어느쪽도 아닙니다.

## 반공변성

반공변성은 공변성의 반대입니다. 예를 들어 Comparator\<T\>를 생각해 봅시다. Comparator\<Number\>로 Int를 비교할 수 있지만 Comparator\<Int\>로는 Number를 비교할 수 없습니다. 즉 A가 B의 하위타입이라면 Comparator\<B\>는 Comparator\<A\>의 하위타입이며 Comparator\<T\>는 반공변성을 띈다고 합니다. 

만약 타입 파라미터 T가 메서드의 인 위치에만 쓰인다면 그 클래스는 반공변성을 띄게 됩니다. 실제로 Comparator\<T\>의 메서드는 T를 반환하는 위치에서 사용하지 않습니다.

## 사용 지점 변성

클래스에서 공변/반공변적임을 표시하는 것을 선언 지점 변성이라 합니다. 선언 지점 변성이 없더라도 특정 메서드에서 공변성을 활용하고 싶을 수 있습니다. 예를 들어 다음과 같은 메서드를 생각해봅시다.
```Kotlin
fun<T> copy(input: MutableList<T>,target: MutableList<T>  ){
    input.forEach{
        target.add(it)
    }
} 

```
MutableList\<T\>는 무공변이기 때문에 input과 target의 타입이 정확히 같아야 합니다. 
그러나 이 메서드에서 input은 T를 아웃위치에서만 사용하고 target은 T를 인 위치에서만 사용하기 때문에 input과 target의 타입이 정확이 일치하지 않더라도 input이 target의 하위타입이면 실행이 가능한 것을 기대할 수 있습니다. 예를 들어 MutableList\<String\>의 원소를 MutableList\<Any\>로 복사해도 아무 문제가 없습니다. 다음과 같이 파라미터 앞에 in 혹은 out을 붙이면 이를 달성할 수 있습니다.
```Kotlin
fun<T> copy(input: MutableList<out T>,target: MutableList<T>  ){
    input.forEach{
        target.add(it)
    }
} 
\\혹은
fun<T> copy(input: MutableList<T>,target: MutableList<in T>  ){
    input.forEach{
        target.add(it)
    }
} 
```

## 스타 프로젝션 
제네릭 타입 인자 정보가 없음을 표시하기 위해 스타 프로젝션(star projection)을 사용할 수 있습니다.
중요한 것은 '*'와 'Any?'가 같지 않다는 것입니다. 예를 들어 다음과 같은 코드를 생각해 봅시다.

```Kotlin
fun getComparable(): Comparable<Any?>{
    return 1
}

```
이 코드는 컴파일 에러를 발생시킵니다. Int는 Comparable\<Any?\>를 구현하지 않기 때문입니다.
하지만 다음 코드는 컴파일 에러가 발생하지 않습니다.

```Kotlin
fun getComparable(): Comparable<*>{
    return 1
}
```
이제 '*'와 'Any?'의 차이를 이해하셨으리라 생각합니다.

## References
Kotlin In Action - Dmitry Jemerov, Sveltlana Isakova
