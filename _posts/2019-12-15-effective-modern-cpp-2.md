---
layout: post
title: "Effective Modern C++ (2)"
date: 2019-12-15 22:38
author: taeguk
tags: [C++, Modern-C++, Effective-Modern-C++]
---

저번시간에 이어서 오늘은 예전에 Effective Modern C++ 을 공부하며 정리했던 내용들을 포스팅해볼까 합니다~\
C++11/14 에서의 best practice 에 관한 내용으로서 최근 C++20 이 나오는 시점에서 이 또한 최신 내용은 아니긴 하지만 여전히 많은 부분들이 유효합니다.

## Chapter 4. 똑똑한 포인터 (Smart Pointer)

### 항목 18. 소유권 독점 자원의 관리에는 std::unique_ptr를 사용하라.

보통 smart pointer를 처음 접한 사람들은 std::shared_ptr만을 남용(?)하는 경향이 있다.그러나 std::shared_ptr이 매우 강력한 존재이긴 하지만 크게 2가지 측면에서 단점이 있다.\
첫 번째는 overhead이다. 참조 계수를 관리하기 위해 어쩔 수 없이 overhead가 존재한다.\
두 번째는 돌이킬 수 없다는 점이다. 한번 std::shared_ptr에 pointer를 물리고 나면 다시는 일반 pointer로 복귀할 수 없다. (단순히 .get()을 쓰면 raw pointer를 받을 수 있을 뿐이다. 여전히 프로그램 어딘가에서 포인터를 참조할 수 있을 수 있다.)

이러한 단점들 때문에 나는 기본적으로 std::unique_ptr을 사용한다. std::unique_ptr은 덜 강력하지만 위 2가지 단점이 없다. 추가적인 메모리를 요구하는 커스텀 삭제자를 지정하지 않는다면 performance는 raw pointer와 사실상 동일하고, std::unique_ptr를 사용하다가 적합하지 않은 상황이 오면 언제든지 정책을 raw pointer나 std::shared_ptr등으로 바꿀 수 있다.\
참고) raw pointer, std::unique_ptr, std::shared_ptr의 '생성 및 삭제' 성능 비교 : http://www.modernescpp.com/index.php/memory-and-performance-overhead-of-smart-pointer

특히 어떻게 쓰일 지 모르는 포인터를 반환해야 하는 경우는 std::unique_ptr를 쓰는 것이 좋다. std::unique_ptr은 어떤 형태로든지 변환이 가능하기 때문이다. 특별한 목적이 있는 경우가 아니라면, 팩토리 함수등에서 std::shared_ptr 을 반환하는 것은 좋지 않은 습관이다.\
하지만 std::shared_ptr로 쓰일 것이 확실한 경우라면 std::shared_ptr을 반환하는 것이 좋다. 왜냐하면 std::unique_ptr로 반환된 뒤 이것을 std::shared_ptr로 변환할 경우에는 actual object와 control block이 따로 관리되기 때문이다. 반면에 바로 std::shared_ptr로 반환할 경우에는 애초에 std::make_shared를 사용할 수 있기 때문에 더 효율적이다.
```cpp
#include <memory>
#include <iostream>

class Example {};

void deletor_func(Example *ptr) { delete ptr; }

struct DeletorClass {
    void operator() (Example *ptr) const { delete ptr; }
};

struct DeletorClass2 {
    DeletorClass2(std::int32_t num)
    {
        m_arr[0] = m_arr[1] = num;
    }
    void operator() (Example *ptr) const { delete ptr; }
    std::int32_t m_arr[2];
};

int main()
{
    auto a = std::make_unique<Example>();

    auto deletor = [](Example *ptr) { delete ptr; };
    auto b = std::unique_ptr<Example, decltype(deletor)>(new Example, deletor);
    /* In C++17,
        auto b = std::unique_ptr(new Example, [](Example *ptr) { delete ptr; });
    */

    int num = 13;
    auto deletor2 = [num](Example *ptr) { delete ptr; };
    auto c = std::unique_ptr<Example, decltype(deletor2)>(new Example, deletor2);

    auto d = std::unique_ptr<Example, void(*)(Example *)>(new Example, [](Example *ptr) { delete ptr; });
    auto e = std::unique_ptr<Example, DeletorClass>(new Example, DeletorClass());
    auto f = std::unique_ptr<Example, DeletorClass2>(new Example, DeletorClass2(1));

    // https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Empty_Base_Optimization
    // Result in x86 : 4 4 8 8 4 12
    std::cout << sizeof(a) << " " << sizeof(b) << " " << sizeof(c) << " " << 
        sizeof(d) << " " << sizeof(e) << " " << sizeof(f) << std::endl;
}
```

위는 custom deletor를 사용하는 경우에 객체의 size가 얼마나 늘어나는 지에 대한 코드이다.\
주목할 점은 std::unique_ptr은 custom deletor가 type의 일부가 된다는 것이다. 이로 인해 유연함은 떨어지지만, 매우 효율적이다.\
위 결과 중 b,c,e의 경우 어떻게 custom deletor의 역할을 할 수 있는 functor object를 넘겼는데도 불구하고, size가 4일 수 있는지 궁금할 것이다.\
이것은 내가 실험한 visual studio 2015에서 std::unique_ptr 내부적으로 empty base optimization을 활용하기 때문이다. 이것이 std::unique_ptr 표준에 명시된 내용인지는 모르겠지만, 대부분의 표준 라이브러리 구현의 경우 EBO를 활용해서 이런 식으로 최적화를 할 것이다. (참고 : https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Empty_Base_Optimization)

**[기억해 둘 사항들]**
- std::unique_ptr는 독점 소유권 의미론을 가진 자원의 관리를 위한, 작고 빠른 이동 전용 똑똑한 포인터이다.
- 기본적으로 자원 파괴는 delete를 통해 일어나나, 커스텀 삭제자를 지정할 수 도 있다. 상태 있는 삭제자나 함수 포인터를 사용하면 std::unique_ptr 객체의 크기가 커진다.
- std::unique_ptr를 std::shared_ptr로 손쉽게 변환할 수 있다.

### 항목 19. 소유권 공유 자원의 관리에는 std::shared_ptr를 사용하라.

std::shared_ptr은 std::unique_ptr과는 다르게 커스텀 삭제자가 타입의 일부가 아니다. 좀 더 유연하게 사용이 가능하지만, 서로 다른 타입의 커스텀 삭제자를 포함하기 위해 포인터 한 개를 더 가져야 하는 memory overhead와 최적화 방해 요소가 생기게 된다.\
또한 std::shared_ptr은 참조 횟수 관리가 필요한 순간에 overhead가 있다. (생성, 소멸, 복사 등..)\
그러면 dereference 일 때는 어떨까? 이 경우는 단순하게 생각하면 overhead가 없어 보인다. 그러나 실제로 std::shared_ptr은 raw pointer에 비해 2배의 메모리를 사용하므로, cache 측면에서 불리하다.\
즉, std::shared_ptr의 배열과 raw pointer의 배열이 있을 때 dereferencing performance를 비교하면 std::shared_ptr이 더 느리다. 즉, 상황에 따라 다르긴 하지만 std::shared_ptr은 dereference에 있어서도 overhead가 있다는 것에 유의해야 한다.

**[참조 횟수 관리가 성능에 끼치는 영향]**
- std::shared_ptr의 크기가 생 포인터의 두 배이다.
- 참조 횟수를 담을 메모리를 반드시 동적으로 할당해야 한다.
- 참조 횟수의 증가와 감소가 반드시 원자적 연산이어야 한다.

**[기억해 둘 사항들]**
- std:;shared_ptr는 임의의 공유 자원의 수명을 편리하게(쓰레기 수거에 맡길 때 만큼이나) 관리할 수 있는 수단을 제공한다.
- 대체로 std::shared_ptr 객체는 그 크기가 std::unique_ptr 객체의 두 배이며, 제어 블록에 관련된 추가 부담을 유발하며, 원자적 참조 횟수 조작을 요구한다.
- 자원은 기본적으로 delete를 통해 파괴되나, 커스텀 삭제자도 지원된다. 삭제자의 형식은 std::shared_ptr의 형식에 아무런 영향도 미치지 않는다.
- 생 포인터 형식의 변수로부터 std::shared_ptr를 생성하는 일은 피해야 한다.

### 항목 20. std::shared_ptr처럼 작동하되 대상을 잃을 수도 있는 포인터가 필요하면 std::weak_ptr를 사용하라.

예전에 Microsoft/CNTK 의 dangling pointer 문제를 std::weak_ptr을 이용해서 해결해 PR을 날리고 merge된 경험이 있다. std::weak_ptr이 쓰이는 실제 사례가 알고 싶은 분은 다음 링크를 참고해보면 좋을 것 같다.\
https://github.com/Microsoft/CNTK/pull/1441

**[기억해 둘 사항들]**
- std::shared_ptr처럼 작동하되 대상을 잃을 수도 있는 포인터가 필요하면 std::weak_ptr를 사용하라.
- std::weak_ptr의 잠재적인 용도로는 캐싱, 관찰자 목록, 그리고 std::shared_ptr 순환 고리 방지가 있다.

### 항목 21. new를 직접 사용하는 것보다 std::make_unique와 std::make_shared를 선호하라.

```cpp
#include <iostream>
#include <memory>

class Example
{
public:
    Example(int, int)
    { std::cout << "Example(int, int) called!" << std::endl;}

    /* http://stackoverflow.com/questions/17803475/why-is-stdinitializer-list-often-passed-by-value */
    Example(std::initializer_list<int>)
    { std::cout << "Example(std::initializer_list<int>) called!" << std::endl;}
};

int main()
{
    auto a = std::make_shared<Example>(1, 2); /* Example(int, int) called! */
    auto b = std::make_shared<Example>(std::initializer_list<int>{1, 2}); /* Example(std::initializer_list<int>) called! */
    auto tmp = { 1, 2 };
    auto c = std::make_shared<Example>(tmp); /* Example(std::initializer_list<int>) called! */

    /* https://akrzemi1.wordpress.com/2016/07/07/the-cost-of-stdinitializer_list/ */
    int arr[4] = { 1,3,5,7 };
    auto qq = { arr[0],2,arr[1],4,arr[2],6,arr[3],8 };
    std::cout << sizeof(qq) << std::endl; // 8 in x86, 16 in x64
}
```

중괄호 초기치를 perfect forwarding 할 수 없는 한계점 때문에 make_* 의 사용이 불가능하다면 위 코드에 나와있는 방법으로 해결할 수 있다.

**[기억해 둘 사항들]**
- new의 직접 사용에 비해, make 함수를 사용하면 소스 코드 중복의 여지가 없어지고, 예외 안전성이 향상되고, std::make_shared와 std::allocate_shared의 경우 더 작고 빠른 코드가 산출된다.
- make 함수의 사용이 불가능 또는 부적합한 경우로는 커스텀 삭제자를 지정해야 하는 경우와 중괄호 초기치를 전달해야 하는 경우가 있다.
- std::shared_ptr에 대해서는 make 함수가 부적합한 경우가 더 있는데, 두 가지 예를 들자면 (1) 커스텀 메모리 관리 기능을 가진 클래스를 다루는 경우와 (2) 메모리가 넉넉하지 않은 시스템에서 큰 객체를 자주 다루어야 하고 std::weak_ptr들이 해당 std::shared_ptr들보다 더 오래 살아남는 경우이다.

### 항목 22. Pimpl 관용구를 사용할 때에는 특수 멤버 함수들을 구현 파일에서 정의하라.

std::unique_ptr 형식의 pImpl 포인터를 사용할 때 '거지같은 문제'들이 발생하는 이유는 바로 std::default_delete 때문이다.\
std::unique_ptr은 deletor type이 template 인자로서 type에 포함된다. 그리고 그 template 인자의 기본 값이 std::default_delete이다.\
그리고 std::default_delete는 compile-time에 definition이 알려지고, 이 definition안에는 delete 구문이 있다.\
이 delete 구문은 객체의 소멸자를 호출할 것이다. 따라서 incomplete type은 std::unique_ptr과 함께 쓸 수 없다.\
따라서 이를 해결하기 위해 스콧 마이어스가 effective modern C++를 통해 제안한 것이 특수 멤버 함수들을 클래스 헤더에 선언하고, 구현파일에서 구현하는 방법이다.\
미안하지만 참 별로다.. 문제의 근원은 바로 std::default_delete이다. 따라서 이것을 사용하지 않으면 된다.\
결론은 std::unique_ptr<Impl, void (*)(Impl *)> 와 같이 function pointer를 deletor type으로 지정해주고, 생성자에서만 deletor를 넣어주는 것이다.\
물론 이렇게 할 경우, function pointer를 담아야 하므로 std::unique_ptr의 크기가 2배 커지는 단점이 있다. 이를 해결하기 위해서는 function pointer가 아닌 Functor를 활용하면 된다.\
struct Deletor { void operator() (Example *ptr) const { delete ptr; } }; 를 만든 뒤 std::unique_ptr<Impl, Deletor> 하는 식으로 하면 std::unique_ptr의 크기는 그대로 유지하면서 스콧마이어스가 제안한 방법보다 더 깔끔해진다.\
관련 내용에 더 관심이 있는 사람들은 아래 링크들을 보길 바란다.\
https://howardhinnant.github.io/incomplete.html \
http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html (강추)

**[기억해 둘 사항들]**
- Pimpl 관용구는 클래스 구현과 클래스 클라이언트 사이으이 컴파일 의존성을 줄임으로써 빌드 시간을 감소한다.
- std::unique_ptr 형식의 pImpl 포인터를 사용할 때에는 특수 멤버 함수들을 클래스 헤더에 선언하고 구현 파일에서 구현해야 한다. 컴파일러가 기본으로 작성하는 함수 구현들이 사용하기에 적합한 경우에도 그렇게 해야 한다.
- 위의 조언은 std::unique_ptr에 적용될 뿐, std:;shared_ptr에는 적용되지 않는다.

## Chapter 5. 오른값 참조, 이동 의미론, 완벽 전달

### 항목 23. std::move와 std::forward를 숙지하라.

```cpp
#include <iostream>

class Example
{
public:
    Example() = default;
    Example(const Example &) { std::cout << "copy constructor" << std::endl; }
    Example(Example &&) { std::cout << "move constructor" << std::endl; }
    Example(const Example &&) { std::cout << "const move constructor" << std::endl; }
};

/* std::forward is meaningful only for universal reference in template. */

void test1(Example ex)
{
    std::cout << "Test 1" << std::endl;
    Example a(std::forward<Example>(ex));    // move constructor
    Example b(std::forward<Example&>(ex));    // copy constructor
    Example c(std::forward<Example&&>(ex));    // move constructor
}

void test2(Example& ex)
{
    std::cout << "Test 2" << std::endl;
    Example a(std::forward<Example>(ex));    // move constructor
    Example b(std::forward<Example&>(ex));    // copy constructor
    Example c(std::forward<Example&&>(ex));    // move constructor
}

void test3(Example&& ex)
{
    std::cout << "Test 3" << std::endl;
    Example a(std::forward<Example>(ex));    // move constructor
    Example b(std::forward<Example&>(ex));    // copy constructor
    Example c(std::forward<Example&&>(ex));    // move constructor
}

template <typename T>
void test4(T&& a) /* universal reference(=forwarding reference) */
{
    std::cout << "Test 4" << std::endl;
    Example aa(std::forward<T>(a));
}

template <typename T>
void test5(const T&& a) /* Cannot be universal reference */
{
    std::cout << "Test 5" << std::endl;
    Example aa(std::forward<T>(a));  // Compile Error 
                                     // because cannot remove "const" through static_cast in std::forward.

    // Compile OK.
    // Example aa(std::forward<T>(const_cast<T>(a)));
}

int main()
{
    test1(Example());
    test2(Example());
    test3(Example());

    test4(Example());
    // test5(Example());

    const Example ex;
    test4(std::move(ex));   // const move constructor
    Example(std::move(ex));    // const move constructor
}
```
초심자가 착각할 수 있을 만한 점을 지적하고자 한다. std::forward는 템플릿과 함께 쓰여야만 한다.\
그냥 비템플릿 함수에서는 타입이 &&인지 아닌지를 알 수 있기 때문에 그냥 std::move를 쓰던가 말던가 하면된다. 괜히 std::forward를 사용하는 것은 관례도 아닐뿐더러, "동작도 제대로 하지 않는다"(std::forward는 단순히 T&& 로의 형변환만을 수행할 뿐이다. 여기서 reference collapsing이 일어난다.)\
그리고 T&& 만이 universal reference가 될 수 있다. test5같은 템플릿함수에서는 std::forward를 사용하지 못한다. 왜냐하면 std::forward내부에서 static_cast를 사용하는데, 여기에서 const를 제거할 수는 없기 때문이다. 대부분 const && 타입을 본 적이 거의 없을 텐데, 이 것도 단순히 그냥 우리가 아는 &&와 const가 합쳐진 것이다. 예외 사항 같은 것은 없다.

**[기억해 둘 사항들]**
 - std::move는 오른값으로서의 무조건 캐스팅을 수행한다. std::move 자체는 아무 것도 이동하지 않는다.
 - std::forward는 주어진 인수가 오른값에 묶인 경우에만 그것을 오른값으로 캐스팅한다.
 - std::move와 std::forward 둘 다, 실행시점에서는 아무 일도 하지 않는다.

### 항목 24. 보편 참조와 오른값 참조를 구별하라.

universal reference 라는 이름이 내포할 수 있는 오류 때문에 이름을 forwarding reference로 바꾸자는 제안이 제출되었었다. 관심있는 사람은 아래 링크를 참고.\
http://stackoverflow.com/questions/33904462/whats-the-standard-official-name-for-universal-references

**[기억해 둘 사항들]**
 - 함수 템플릿 매개변수의 형식이 T&& 형태이고 T가 연역된다면, 또는 객체를 auto&&로 선언한다면, 그 매개변수나 객체는 보편 참조이다.
 - 형식 선언의 형태가 정확히 형식&&가 아니면, 또는 형식 연역이 일어나지 않으면, 형식&&는 오른값 참조를 뜻한다.
 - 오른값으로 초기화되는 보편 참조는 오른값 참조에 해당한다. 왼값으로 초기화되는 보편 참조는 왼값 참조에 해당한다.

### 항목 25. 오른값 참조에는 std::move를, 보편 참조에는 std::forward를 사용하라.

**[기억해 둘 사항들]**
 - 오른값 참조나 보편 참조가 마지막으로 쓰이는 지점에서, 오른값 참조에는 std::move를, 보편 참조에는 std::forward를 적용하라.
 - 결과를 값 전달 방식으로 돌려주는 함수가 오른값 참조나 보편 참조를 돌려줄 때에도 각각 std::move나 std::forward를 적용하라.
 - 반환값 최적화의 대상이 될 수 있는 지역 객체에는 절대로 std::move나 std::forward를 적용하지 말아야 한다.

### 항목 26. 보편 참조에 대한 중복적재를 피하라.

아래의 중복 적재 해소 규칙에 따라 보편 참조에 대한 중복적재를 사용할 경우, 여러가지 문제들이 생길 수 있다. 중복 적재 해소 규칙을 따져보지 않은 채로 보편 참조에 대해 중복적재를 시도하면 의도치 않은 문제들이 생길 것이다. 그리고 만약 중복 적재 해소 규칙을 따져본다면 보편 참조에 대해 중복적재를 시도하지 않을 것이다. 즉, 결론은 보편 참조에 대해서는 중복적재(overloading)을 하지말라.
(번외로.. 어느새 책에서 사용하는 중복 적재라는 단어가 익숙해져 버렸다... Effective Modern C++ 책의 번역에 대해 말이 많다... 나도 처음에는 처음 보는 번역들이 어색했지만... 이제는 말할 수 있다.. 좋은 번역 같다고.)

**[중복 적재 해소 규칙]**
- 정확한 부합이 암시적 타입 캐스팅을 통한 부합보다 우선 시 된다.
- 어떤 함수 호출이 템플릿 인스턴스와 비템플릿 함수에 똑같이 부합한다면 비템플릿 함수를 우선시 한다.

**[기억해 둘 사항들]**
 - 보편 참조에 대한 중복적재는 거의 항상 보편 참조 중복적재 버전이 예상보다 자주 호출되는 상황으로 이어진다.
 - 완벽 전달 생성자들은 특히나 문제가 많다. 그런 생성자는 대체로 비const 왼값에 대한 복사 생성자보다 더 나은 부합이며, 기반 클래스 복사 및 이동 생성자들에 대한 파생 클래스의 호출들을 가로챌 수 있기 때문이다.

### 항목 27. 보편 참조에 대한 중복적재 대신 사용할 수 있는 기법들을 알아 두라.

보편 참조에 대한 중복적재로 인한 문제가 생겼을 때 할 수 있는 대처는 아래 5가지가 있다.
1. 중복적재를 포기한다.
   - 생성자에는 적용할 수 없다.
   - 해결책이라기 보단 '회피책'이다.
2. const T& 매개변수를 사용한다.
   - 보편 참조만큼 성능 측면에서 효율적이지 않다.
   - 그러나, 그렇게 효율적일 필요가 없는 경우에는 가장 깔끔한 해결책이다.
3. 값 전달 방식의 매개변수를 사용한다.
   - 이 논의에서는 const T& 를 사용하는 것과 근본적으로 같은 아이디어이다. 단지, 추가로 항목 41의 조언을 채택했을 뿐이다.
4. 꼬리표 배분(tag dispatch)를 사용한다.
   - 결국 문제가 생긴 원인은 보편 참조에 대한 중복적재 해소 규칙 때문이다. 즉, 중복적재 해소가 명확하게 일어날 수 있게 하는 것이 아이디어이다.
   - std::is_integral<T>, std::true_type 등을 활용해 tag dispatching을 해서 해결한다.
   - tag dispatching은 http://www.boost.org/community/generic_programming.html#tag_dispatching 을 참고.
5. 보편 참조를 받는 템플릿을 제한한다. (std::enable_if)
   - 위와 비슷한 아이디어로서, 중복적재 해소가 명확하게 일어날 수 있게끔 보편 참조를 받는 템플릿을 제한한다.
   - std::enable_if를 활용한다.
   - SFINAE (https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/SFINAE)
   - enable-if (https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/enable-if)

자, 크게 세 가지 그룹으로 나눌 수 있다. 1 / 2,3 / 4,5\
1번은 중복적재를 포기함으로서, 즉, 함수이름을 변경함으로서 해결을 한다. 하지만 이것은 사실상 생성자에는 적용할 수 없고, 이 것은 사실상 회피책이기 때문에 별로 의미가 없다.\
2,3번은 '보편 참조'를 포기하는 방법이다. 보편 참조를 포기하므로서 성능측면에서 효율성은 떨어지지만, 간단하게 문제를 해결할 수 있다. 그리고 또한 사용자 측면에서 훨씬 편리한 장점이 있다.\
4,5번은 '보편 참조'를 유지하되 중복적재 해소 과정에서 보편 참조로 deduction되는 것에 제한을 두는 것이다. 보편 참조를 유지하므로 성능측면에서 효율적이나, template meta programming을 이용해야하므로 문제를 해결하는 과정이 복잡하다. 그리고 인자가 정확하지 않을 경우 무지막지한 에러메세지들을 뱉어내고 파라미터로서 가능한 값들이 명확히 한눈에 들어오지 않는 보편 참조의 문제점으로 인해 사용자 편리성이 떨어지는 단점이 있다. 만약 사용한다면 4,5번 중에서는 개인적으로 5번을 사용하는 것이 가독성과 구조 측면에서 더 좋다고 생각한다.\
결론적으로 아마 2,3 과 4,5중에 해결책을 도모해야 할텐데, 일단 기본적으로는 2,3 (그중에서도 2번) 을  사용하고, 보편 참조를 사용해야 할만한 타당한 이유가 생겼을 때 바꾸는 것이 좋다고 생각된다. 물론 인터페이스 변경에 민감한 부분을 설계할 때는 초기에 충분한 고민이 선행되어야 할 것이다.

**[기억해 둘 사항들]**
 - 보편 참조와 중복적재의 조합에 대한 대안으로는 구별되는 함수 이름 사용, 매개변수를 const에 대한 왼값 참조로 전달, 매개변수를 값으로 전달, 꼬리표 배분 사용 등이 있다.
 - std::enable_if를 이용해서 템플릿의 인스턴스화를 제한함으로써 보편 참조와 중복적재를 함께 사용할 수 있다. std::enable_if는 컴파일러가 보편 참조 중복적재를 사용하는 조건을 프로그래머가 직접 제어하는 용도로 쓰인다.
 - 보편 참조 매개변수는 효율성 면에서 장점인 경우가 많지만, 대체로 사용성 면에서는 단점이 된다.

### 항목 28. 참조 축약을 숙지하라.

**[기억해 둘 사항들]**
 - 참조 축약은 템플릿 인스턴스화, auto 형식 연역, typedef와 별칭 선언의 지정 및 사용, decltype의 지정 및 사용이라는 네 가지 문맥에서 일어난다.
 - 컴파일러가 참조 축약 문맥에서 참조에 대한 참조를 만들어 내면, 그 결과는 하나의 참조가 된다. 원래의 두 참조 중 하나라도 왼값 참조이면 결과는 왼값 참조이고, 그렇지 않으면 오른값 참조이다.
 - 형식 연역이 왼값과 오른값을 구분하는 문맥과 참조 축약이 일어나는 문맥에서 보편 참조는 오른값 참조이다.

### 항목 29. 이동 연산이 존재하지 않고, 저렴하지 않고, 적용되지 않는다고 가정하라.

나는 이 항목의 이름에 대해 비판적인 시각을 가지고 있다. 책에서 말하는 move semantics가 도움이 되지 않는 시나리오는 다음과 같다.
1. 이동 연산이 없다.
2. 이동이 더 빠르지 않다.
3. 이동을 사용할 수 없다.

너무나 당연한 이야기이다.\
근데 코드를 짤 때 "이동 연산이 존재하지 않고, 저렴하지 않고, 적용되지 않는다고 가정하라" 는 것은 잘 이해가 안된다.\
일단 이동 연산이 없는 경우에는 자동으로 복사연산이 선택된다. 그리고 이동연산이 더 빠르진 않을 순 있어도 이동이 복사보단 느릴 경우는 (적어도 내가 알기론) 없다.\
또, 예외 안정성 때문에 이동을 사용할 수 없는 경우에 대해서는 std::move_if_noexcept 를 이용해서 조건부 move를 할 수 있다.\
즉, 나는 저렇게 가정을 해야 할 구체적인 사례를 알 지 못한다. 그리고 책에서도 그러한 구체적인 사례를 예시를 들지는 않고 있다.\
엄밀하게 따졌을 때, "이동 연산이 존재하지 않거나 저렴하지 않거나 적용되지 않을 수 있다는 것을 알아두라" 가 항목 이름으로서 더 적합하다고 생각한다.

**[기억해 둘 사항들]**
 - 이동 연산들이 존재하지 않고, 저렴하지 않고, 적용되지 않을 것이라고 가정하라.
 - 형식들과 이동 의미론 지원 여부를 미리 알 수 있는 경우에는 그런 가정을 둘 필요가 없다.

### 항목 30. 완벽 전달이 실패하는 경우들을 잘 알아두라.

**[완벽 전달이 실패한다는 것의 정의]**\
두 가지중에 하나라도 만족하면 완벽 전달이 실패한 것이다.
1. 컴파일러가 형식을 연역하지 못한다. (즉, template type deduction이 불가능 경우)
2. 컴파일러가 형식을 잘못 연역한다. (의도와는 다르게 type deduction이 일어난 경우를 말한다. 운이 좋으면 컴파일에러지만, 심각할 경우 다른 중복적재 버전이 선택되어 런타임에 잘못된 동작을 할 수도 있다.)

**[완벽 전달이 실패하는 경우들]**
1. 중괄호 초기치
    - 원인 : 항목 2에서 다루듯이, template type deduction에서는 중괄호 초기치가 실패한다. (반면, auto의 type deduction에서는 성공한다는 점을 다시 한번 기억하자.)
    - 해결책 : https://github.com/taeguk/Effective-Cpp-Series/blob/master/EffectiveModernCpp/ch4/21.cpp#L18-L20
2. 널 포인터를 뜻하는 0 또는 NULL
    - 원인 : 0이나 NULL은 정수 형식으로서 잘못 연역된다.
    - 해결책 : nullptr을 사용한다.
3. 선언만 된 정수 static const 및 constexpr 자료 멤버
    - 원인 : 완벽전달은 기본적으로 '참조'이므로 정의가 되있어야 한다. (그러나, 사실상 완벽전달에서도 const propagation이 가능하면 좋으므로 컴파일러에서 요령껏 처리해줄 수 도 있다.)
    - 해결책 : 정의도 제공한다.
4. 중복적재된 함수 이름과 템플릿 이름
    - 원인 : 완벽전달 함수 입장에서 구체적인 타입을 연역하지 못한다.
    - 해결책 : static_cast를 활용하는 등의 방법을 써서 구체적인 타입을 연역할 수 있게 한다.
5. 비 const 비트필드
    - 원인 : 비트에 대해 '참조'할 수 있는 방법(자료형)이 없기 때문이다. (비트필드에 대한 const 참조는 임시변수가 bitfield의 값을 잡고있게 하고 그 임시변수를 const 참조함으로서 가능하다.)
    - 해결책 : 비트필드를 담을 수 있는 값으로 전달함으로서 해결한다. (아래 참고)
		```cpp
		#include <iostream>

		struct test
		{
		    std::uint8_t a : 4,
		                 b : 4;
		};

		template <typename T>
		void foo(T& a) {}
		template <typename T>
		void bar(const T& a) {}

		template <typename ...T>
		void fwd_foo(T&&... args)
		{
		    foo(std::forward<T...>(args...));
		}
		template <typename ...T>
		void fwd_bar(T&&... args)
		{
		    bar(std::forward<T...>(args...));
		}

		int main()
		{
		    struct test t = {};
		    const struct test ct = {};

		    // foo(t.b);  // Compile Error
		    foo(ct.b);
		    bar(t.b);
		    bar(ct.b);

		    // fwd_foo(t.b);  // Compile Error
		    fwd_foo(ct.b);
		    // fwd_bar(t.b);  // Compile Error
		    fwd_bar(static_cast<std::uint8_t>(t.b));
		    fwd_bar(ct.b);
		}
		```

**[기억해 둘 사항들]**
 - 완벽 전달은 템플릿 형식 연역이 실패하거나 틀린 형식을 연역했을 때 실패한다.
 - 인수가 중괄호 초기치이거나 0 또는 NULL로 표현된 널 포인터, 선언만 된 정수 static const 및 constexpr 자료 멤버, 템플릿 및 중복적재된 함수 이름, 비트필드이면 완벽 전달이 실패한다.
