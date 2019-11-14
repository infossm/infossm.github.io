---
layout: post
title: "Effective Modern C++"
date: 2019-11-14 23:58
author: taeguk
tags: [C++, Modern C++, Effective Modern C++]
---

오늘은 예전에 Effective Modern C++ 을 공부하며 정리했던 내용들을 포스팅해볼까 한다.\
C++11/14 에서의 best practice 에 관한 내용으로서 최근 C++20 이 나오는 시점에서 이 또한 최신 내용은 아니긴 하지만 여전히 많은 부분들이 유효한 내용들이라서 큰 도움이 된다고 생각한다.

## Chapter 1. 형식 연역 (Type Deduction)

### 항목 1. 템플릿 형식 연역 규칙을 숙지하라.
아래에 적어놓은 코드를 보면 C++11/14에서의 Template Type Deduction 규칙을 파악할 수 있을 것이다.\
대부분 직관과 거의 잘 맞아떨어진다.\
그래도 주의 할 점 몇가지를 살펴보면, 일단 첫번째는 함수와 배열의 decay 부분이다.\
일반적으로 C에서 배열은 포인터로 붕괴되고, 함수도 포인터로 붕괴된다.\
C++에서도 이는 마찬가지인데, 주의할 점이 참조(&) 가 사용될 때는 붕괴가 되지 않는다.\
따라서 template type deduction에 있어서도 ParamType이 &일 경우에는 배열과 함수가 decay되지 않는다.\
그리고 두번째로는 보편 참조(universal reference) 이다.\
구체적인 것들은 아래 코드를 열심히 보면 이해가 될 것이다.
```cpp
#include <utility>

template <typename T>
void func_ref(/*ParamType*/ T &) {}

template <typename T>
void func_ptr(/*ParamType*/ T *) {}

template <typename T>
void func_unv_ref(/*ParamType*/ T &&) {}

template <typename T>
void func_val(/*ParamType*/ T) {}

template <typename T, std::size_t N>
void func_arr_ref(/*ParamType*/ T (&)[N]) {}

int main()
{
    /**********************************************/
    const volatile int a = 3;

    func_ref(a);  // T -> const volatile int
                  // ParamType -> const volatile int &

    func_ptr(&a);  // T -> const volatile int
                   // ParamType -> const volatile int *

    func_unv_ref(a);  // T -> const volatile int &
                      // ParamType -> const volatile int &

    func_unv_ref(std::move(a));  // T -> const volatile int
                                 // ParamType -> const volatile int &&

    func_val(a);  // T -> int
                  // ParamType -> int

    /**********************************************/
    const char * const str = "hello";

    func_ref(str);  // T -> const char * const
                    // ParamType -> const char * const &

    func_ptr(str);  // T -> const char
                    // ParamType -> const char *

    func_val(str);  // T -> const char *
                    // ParamType -> const char *

    /**********************************************/
    const int arr[10] = {};

    func_ref(arr);  // T -> const int [10]
                    // ParamType -> const int (&)[10]
                    // *** An array doesn't decay ***

    func_ptr(arr);  // T -> const int
                    // ParamType -> const int *
                    // An array decays to pointer.

    func_val(arr);  // T -> const int *
                    // ParamType -> const int *

    func_arr_ref(arr);  // T -> const int
                        // ParamType -> const int (&)[10]

    /**********************************************/
    void junk();

    func_ref(junk);   // T -> void ()
                      // ParamType -> void (&)()
                      // *** A function doesn't decay ***

    //func_ref(&junk); // COMPILE ERROR!!
                       // The reason is that a function doesn't decay in this case.

    func_ptr(junk);   // T -> void ()
                      // ParamType -> void (*)()
                      // A function decays to a pointer.

    func_ptr(&junk);  // Same to above.

    func_val(&junk);  // T -> void (*)()
                      // ParamType -> void (*)()
}

void junk() {}
```

**[기억해 둘 사항들]**
 - 템플릿 형식 연역 도중에 참조 형식의 인수들은 비참조로 취급된다. 즉, 참조성이 무시된다.
 - 보편 참조 매개변수에 대한 형식 연역 과정에서 왼값 인수들은 특별하게 취급된다.
 - 값 전달 방식의 매개변수에 대한 형식 연역 과정에서 const 또는 volatile(또는 그 둘 다인) 인수는 비 const, 비 volatile 인수로 취급된다.
 - 템플릿 형식 연역 과정에서 배열이나 함수 이름에 해당하는 인수는 포인터로 붕괴한다. 단, 그런 인수가 참조를 초기화하는 데 쓰이는 경우에는 포인터로 붕괴하지 않는다.

### 항목 2. auto의 형식 연역 규칙을 숙지하라.
C++11부터 추가된 auto의 Type Deduction Rule은 기본적으로 template의 type deduction rule과 같다.\
const auto & var = ...;\
가 있을 때, auto를 템플릿의 T로 const auto &를 템플릿의 ParamType으로 생각하면 기본적으로 대부분 맞다.\
그러나, 몇 가지 예외사항들과 주의해야할 점들이 있다.\
아래 코드에서 그 점들을 다루고 있다.\
코드를 보면 되지만, 직접 설명을 좀 하자면,,\
일단, std::initializer_list<auto> 같은 것은 안된다. 기본적으로 auto가 template과 type deduction rule이 거의 똑같다고 해도, 저런식으로 auto를 사용하는 것은 표준에 없다.\
그리고 중괄호 초기치에 대한 auto의 특별한 형식 연역 규칙을 주의해야 한다. (auto a = {1,2,3} 같은...)\
이런 규칙은 template에서는 허용되지 않는데 특별하게 auto에서 허용되는 규칙이다.\
그리고 또, auto x(1); 은 int로, auto x = {1} 와 auto x{1}은 std::initializer_list<int>로 연역되는 점을 주의해야 한다. 그러나 직접 초기화 구문을 이용한 중괄호 초기치에 대해 관련된 특별규칙을 없애자는 제안 N3922가 2014년 11월에 받아들여졌고, C++17에 최종적으로 반영되었다. (참고 : http://stackoverflow.com/questions/25612262/why-does-auto-x3-deduce-an-initializer-list)\
어쨌든 C++11에서 이런 예외사항이 있는데, C++14에 추가된 "람다의 매개변수 선언에 사용되는 auto"와 "함수의 반환 형식에 사용되는 auto"의 경우에는 이러한 예외사항이 적용되지 않는다. (즉, 이 경우들에서는 auto가 template의 형식 연역 규칙을 따른다고 봐야한다.)\
어쨌든, 정리하자면 중괄호 초기치와 관련된 부분을 제외하면 auto의 형식 연역 규칙은 template의 것과 완전히 같다.
```cpp
#include <initializer_list>

template <typename T>
void func_val(/*ParamType*/ T) {}

template <typename T>
void func_initlist(/*ParamType*/ std::initializer_list<T>) {}

int main()
{
    /***************** Exceptional Case in C++11 *****************/

    // func_val({ 1,2,3 });  // COMPILE ERROR!
    
    func_initlist({ 1,2,3 });  // T -> int
                               // ParamType -> std::initializer_list<int>

    auto a = { 1,2,3 };  // auto -> std::initializer_list<int>
                         // Type of Variable -> std::initializer_list<int>

    // std::initializer_list<auto> a = { 1,2,3 };  // There is no standard for it.

    /***************** Exceptional Case in C++14 *****************/

    auto lambda_func = [](auto) {};
    // lambda_func({ 1,2,3 });  // COMPILE ERROR!
}

auto exceptional_case_in_cpp14()
{
    // return { 1,2,3 };  // COMPILE ERROR!
}
```

**[기억해 둘 사항들]**
 - auto 형식 연역은 대체로 템플릿 형식 연역과 같지만, auto 형식 연역은 중괄호 초기치가 std::initializer_list를 나타낸다고 가정하는 반면 템플릿 형식 연역은 그렇지 않다는 차이가 있다.
 - 함수의 반환 형식이나 람다 매개변수에 쓰인 auto에 대해서는 auto 형식 연역이 아니라 템플릿 형식 연역이 적용된다.

### 항목 3. decltype의 작동 방식을 숙지하라.
decltype은 거의 항상 변수나 표현식의 형식을 아무 수정 없이 보고한다.\
그러나 아래와 같이 약간 주의해야 할 점들도 있다.
```cpp
// int func_1();
decltype(auto) func_1() { int a = 1; return a; }

// int& func_2();
decltype(auto) func_2() { int a = 1; return (a); }

template <typename T>
class Example
{
public:
    Example(const T& param) : m_var(param) {}
private:
    T m_var;
};

int main()
{
    /* Usage of "decltype" */

    auto var_1 = func_1();

    decltype(var_1) var_2;

    //Example ex(var_2);  // COMPILE ERROR!
                          // Maybe it is possible since C++17.
    Example<decltype(var_2)> ex(var_2);
}
```

decltype은 위 코드의 경우와 같이 종종 쓰이게 된다.\
특히 위 코드에서 Example 클래스의 객체를 생성할 때 같은 경우 C++14에서는 class constructor에 대해 template type deduction이 적용안되기 때문에 위와 같이 많이 사용하게 된다. 그러나 C++17에서 아마 이 부분이 가능해 질 것으로 보이므로 더 이상 이런 용법의 사용은 안하게 될 것 같다. (http://en.cppreference.com/w/cpp/language/class_template_deduction)

**[기억해 둘 사항들]**
 - decltype은 항상 변수나 표현식의 형식을 아무 수정 없이 보고한다.
 - decltype은 형식이 T이고 이름이 아닌 왼값 표현식에 대해서는 항상 T& 형식을 보고한다.
 - C++14는 declytype(auto)를 지원한다. decltype(auto)는 auto처럼 초기치로부터 형식을 연역하지만, 그 형식 연역 과정에서 decltype의 규칙들을 적용한다.

### 항목 4. 연역된 형식을 파악하는 방법을 알아두라.
내 경험 상 (Visual Studio 2015기준) 보통의 경우에는 IDE 편집기에서 마우스 커서를 갖다 대면, 연역된 형식을 파악할 수 있다. 그러나 표현식이나 형식이 좀 복잡해지면, 그리고 연역이 탬플릿 내에서 되는 경우 IDE가 형식을 알려주지 못하는 경우가 많다.

예를 들면, 이런식으로 형식을 불완전(?)하게 띄어준다.. 그나마 이 경우는 눈으로 파악이 가능한 경우,,, 실제로는 이 보다 더 복잡한 경우도 많다.\
그러나 컴파일러는 정확하게 형식을 알려주기 때문에 나는 이럴 때 일부러 연역된 형식을 알고 싶은 부분에 컴파일 에러를 만들어서 연역된 형식을 컴파일러가 띄워주도록 유도하기도 한다.\
아니면 책에서 소개해준대로 Boost.TypeIndex 라이브러리를 사용하는 것도 방법이 될 수 있다.\
하지만, 나는 지금껏 이렇게까지 해서 연역된 형식을 파악할 필요성을 느낀 적은 없다. 기본적인 C++ type deduction rule의 숙지 & IDE의 지원 정도면 충분한 것 같다.

**[기억해 둘 사항들]**
 - 컴파일러가 연역하는 형식을 IDE 편집기나 컴파일러 오류 메시지, Boost TypeIndex 라이브러리를 이용해서 파악할 수 있는 경우가 많다.
 - 일부 도구의 결과는 유용하지도 않고 정확하지도 않을 수 있으므로, C++의 형식 연역 규칙들을 제대로 이해하는 것은 여전히 필요한 일이다.

## Chapter 2. auto

### 항목 5. 명시적 형식 선언보다는 auto를 선호하라.
진짜 auto는 상당히 좋다. 그 근거를 대자면 여러가지가 있다.\
일단, 거의 반드시 auto를 사용해야만 하는 경우도 있다. (클로저의 형식같이 컴파일러만 알고있는 타입의 경우)\
그리고 형식 불일치로 인한 효율적, 이식성문제가 발생하지 않는다.\
std::size_t getSize() { ... }\
int size = getSize();\
간단한 예시로서 위 코드 같은 경우, 32bit system에서 컴파일했을 때는 문제가 없지만 64bit system에서 컴파일했을 때는 문제가 있을 수 있다. (예를 들어, windows 64bit의 경우 LLP64를 채택하기 때문에 문제가 생긴다.)\
그 외에도 책의 다른 예제들에서 볼 수 있듯이 명시적 형식 선언은 이식성과 효율성 측면에서 문제가 생길 수 있는 여지가 많다.\
그러나 auto를 사용하면 이런 문제들이 발생하지 않는다.\
또 심미적(?)인 측면에서도 auto를 쓰는 게 기다랗고 복잡한 타입이름을 늘어놓는 것 보다 훨씬 좋다.\
결론적으로, 굳이 명시적 형식 선언을 해야 할 필요성을 찾지 못한다면 auto를 사용하는 것이 옳다.\
그러나 가독성 측면에서 auto를 사용하면 타입이 정확하게 드러나지 않기 때문에 코드를 읽기 힘들다는 반론도 존재한다. 하지만 내 생각에는 어차피 거의 항상 IDE의 도움을 받게 되기 때문에 큰 문제는 없다고 본다. 오히려 장황하게 늘어져 있는 명시적 타입이 괜시리 눈과 머리를 피로하게 만들어 가독성이 떨어진다.

**[기억해 둘 사항들]**
 - auto 변수는 반드시 초기화해야 하며, 이식성 또는 효율성 문제를 유발할 수 있는 형식 불일치가 발생하는 경우가 거의 없으며, 대체로 변수의 형식을 명시적으로 지정할 때보다 타자량도 더 적다.
 - auto로 형식을 지정한 변수는 항목 2와 항목 6에서 설명한 문제점들을 겪을 수 있다.

### 항목 6. auto가 원치 않은 형식으로 연역될 때에는 명시적 형식의 초기치를 사용하라.
자, 변수를 선언할 때 개발자의 의도를 살펴보자.\
int a = (표현식);\
과 같은 코드가 있을 때, 변수의 타입을 int로 선언한 의도는 다음 둘 중 하나일 것이다.\
1. (표현식)의 결과가 int여서 a를 int로 선언했다.
2. (표현식)의 결과는 다른 타입이지만 이를 int형으로 암시적 캐스팅하기 위해 a를 int로 선언했다.

유감스럽게도, 코드를 읽는 사람은 개발자가 위 둘 중에 어떤 의도를 가지고 코드를 작성했는 지 판단하기 힘들다.\
그러나, auto를 사용한다면 위 2가지 의도를 분명하게 코드에 반영할 수 있다.\
1번 의도에 해당하는 경우,\
auto a = (표현식);\
과 같이 변수를 선언하면 된다.\
2번 의도에 해당하는 경우,\
auto a = static_cast<int>(표현식);\
과 같이 변수를 선언하면 된다.\
auto를 사용하면 개발자의 의도를 코드에 그대로 담을 수 있기 때문에 훨씬 좋다.\
그리고 1번 의도의 경우, 향후 (표현식)의 결과의 타입이 바뀌더라도 그에 따라 파생되는 변수들의 타입을 변경하는 추가적인 리펙토링이 필요 없기 때문에 훨씬 편하고, 무엇보다 그에 따른 실수들을 할 여지가 없어지기 때문에 훨씬 좋다.\
그러나 auto의 한 가지 단점은, 보이지 않는 대리자 타입 (예를 들면, std::vector<bool>::reference) 같은 것 때문에 실수를 범할 수 있다는 것이다. 다행히 이런 경우는 드물게 발생하긴 하지만, 여전히 꺼림칙한 부분임에는 틀림없다. 이 것은 마땅한 대비책이 없는 것 같고,,,, 단지 만약 어떤 문제의 원인이 보이지 않는 대리자 타입 때문임을 알게 된다면 static_cast를 이용해서 명시적으로 원하는 type으로 casting해서 사용하면 된다.

**[기억해 둘 사항들]**
 - "보이지 않는" 대리자 형식 때문에  auto가 초기화 표현식의 형식을 "잘못" 연역할 수 있다.
 - 형식 명시 초기치 관용구는 auto가 원하는 형식을 연역하도록 강제한다.

## Chapter 3. 현대적 C++에 적응하기

### 항목 7. 객체 생성 시 괄호(())와 중괄호({})를 구분하라.
내 경험과 개인적인 의견에 따르면,,, 객체 생성 시 왠만하면 괄호를 사용하고, 클래스 내에서 멤버 변수의 기본 값을 설정할 때와 std::initializer_list 를 매개변수로 받는 생성자를 호출 할 때에만 중괄호 초기치를 사용하는 것이 좋다. 이러면 별 문제가 없다.\
중괄호 초기치의 장점은 아래와 같다.
1. 가장 광범위하게 적용할 수 있는 초기화 구문이다.
2. 좁히기 변환을 방지할 수 있다.
3. C++의 most vexing parse에서 부터 자유롭다. ("선언으로 해석 할 수 있는 것은 항상 선언으로 해석해야 한다.")

일단 1번 경우와 2번의 장점으로 인해 클래스 내에서 멤버 변수의 기본 값을 설정할 때에는 중괄호 초기치를 사용하는 것이 좋다.\
그리고 2번 장점의 경우, 대부분 auto의 올바른 사용으로 인해 장점이 무색해진다. 어차피 auto 를 사용하면 암시적인 좁히기 변환이 불가능하기 때문이다. 오히려 auto와 중괄호 초기치의 결합은 type deduction에 있어서 혼란을 가져올 수 있기 때문에 좋지 않다. 따라서 chapter 2의 교훈에 따라 auto를 적극적으로 그리고 잘 활용한다면, 일반적으로 중괄호 초기치보다 괄호를 쓰는 것이 훨씬 좋다.\
그러면 3번의 장점을 봐보자. 3번 부분은 ReturnValue obj(); 같은 구문의 경우 이 것이 함수 선언으로 해석되는 것인데, 중괄호 초기치를 사용하면 그럴 일이 없기 때문에 좋다는 것이다... 글쌔... 그냥 ReturnValue obj;와 같이 사용하는 게 낫지 않을 까 싶다.\
이제 한번 템플릿 안에서 객체를 생성할 때는 괄호와 중괄호 중 어떤 걸 사용할 지 생각해보자.\
내 생각에는 템플릿에서는 더더욱 괄호를 쓰는 게 옳다. 템플릿 매개변수로 어떤 타입이 올지 모르는데 만약 그 타입의 생성자 중에 매개변수로 std::initializer_list를 받는 것이 있다면, 중괄호를 사용하는 것은 끔찍한 결정이 된다. 괄호를 사용하는 게 당연하다고 생각하고, std::make_shared등도 괄호를 채택했다.
```cpp
#include <iostream>
#include <string>

using namespace std;

class Example
{
public:
    Example(int a, int b) { cout << "normal" << endl; }
    Example(std::initializer_list<string> i) { cout << "initializer_list" << endl; }
};

int main()
{
    Example ex1({ 1, 2 }); // print "normal" (THERE IS NO "COMPILE ERROR")
    Example ex2{ 1, 2 };  // print "normal"
}
```
위는 마지막으로 착각할 수 있을 만한 부분을 보여준다. ex1 같이 괄호 안에 중괄호를 사용하면, 마치 '명시적으로' std::initializer_list를 매개변수로 받는 Example의 생성자를 호출할 것같은 "착각"을 할 수 있는데, ({})는 대체로 {}와 같다. 따라서 normal이 출력되게 한다. 착각하지 말도록 하자.

**[기억해 둘 사항들]**
- 중괄호 초기화는 가장 광범위하게 적용할 수 있는 초기화 구문이며, 좁히기 변환을 방지하며, C++의 가장 성가신 구문 해석 (most vexing parse)에서 자유롭다.
- 생성자 중복적재 해소 과정에서 중괄호 초기화는 가능한 한 std::initailizer_list 매개변수가 있는 생성자와 부합한다. (심지어 겉보기에 그보다 인수들에 더 잘 부합하는 생성자들이 있어도).
- 괄호와 중괄호의 선택이 의미 있는 차이를 만드는 예는 인수 두 개로 std::vector<수치 형식>을 생성하는 것이다.
- 템플릿 안에서 객체를 생성할 때 괄호를 사용할 것인지 중괄호를 사용할 것인지 선택하기가 어려울 수 있다.

### 항목 8. 0과 NULL보다 nullptr를 선호하라.

**[기억해 둘 사항들]**
- 0과 NULL보다 nullptr을 선호하라.
- 정수 형식과 포인터 형식에 대한 중복적재를 피하라.

### 항목 9. typedef보다 별칭 선언을 선호하라.

**[기억해 둘 사항들]**
- typedef는 템플릿화를 지원하지 않지만, 별칭 선언은 지원한다.
- 별칭 템플릿에서는 "::type" 접미어를 붙일 필요가 없다. 템플릿 안에서 typedef를 지칭할 때에는 "typename" 접두사를 붙여야 하는 경우가 많다.
- C++14는 C++11의 모든 형식 특질 변환에 대한 별칭 템플릿들을 제공한다.

### 항목 10. 범위 없는 enum보다 범위 있는 enum을 선호하라.
확실히 범위 있는 enum을 선호하는 게 맞긴 한데 범위 있는 enum은 underlying type으로 implicit type conversion이 안되서 좀 불편할 때가 많다. 보통 enum의 값을 배열 혹은 컨테이너의 index와 연관지을 때가 많은데, 이럴 때 일일히 static_cast를 해줘야 하는 점이 좀 불편하다. namespace와 범위 없는 enum을 조합하면, 범위는 있되, implicit type conversion은 가능한 enum을 만들 수 있다.
#include <type_traits>
```cpp

enum class Enum_A { A_1, A_2, A_3, };  // SCOPED, and implicit type conversion is not admitted.
namespace Enum_B { enum { B_1, B_2, B_3, }; }  // SCOPED, and implicit type conversion to underlying type.

int main()
{
    int arr[] = { 1,2,3,4 };

    // arr[Enum_A::A_1];  // COMPILER ERROR!
    arr[static_cast<std::underlying_type_t<Enum_A>>(Enum_A::A_1)];

    arr[Enum_B::B_1];
}
```

**[기억해 둘 사항들]**
- C++98 스타일의 enum을 이제는 범위 없는 enum이라고 부른다.
- 범위 있는 enum의 열거자들은 그 안에서만 보인다. 이 열거자들은 오직 캐스팅을 통해서만 다른 형식으로 변환된다.
- 범위 있는 enum과 범위 없는 enum 모두 바탕 형식 지정을 지원한다. 범위 있는 enum의 기본 바탕 형식은 int이다. 범위 없는 enum에는 기본 바탕 형식이 없다.
- 범위 있는 enum은 항상 전방 선언이 가능하다. 범위 없는 enum은 해당 선언에 바탕 형식을 지정하는 경우에만 전방 선언이 가능하다.

### 항목 11. 정의되지 않은 비공개 함수보다 삭제된 함수를 선호하라.
따로 하고 싶은 말은 삭제된 함수는 public으로 두는 것이 일반적인 관례이다. (public으로 둬야 컴파일러 메세지가 좀 더 정확하다.)

**[기억해 둘 사항들]**
- 정의되지 않은 비공개 함수보다 삭제된 함수를 선호하라.
- 비멤버 함수와 템플릿 인스턴스를 비롯한 그 어떤 함수도 삭제할 수 있다.

### 항목 12. 재정의 함수들을 override로 선언하라.
이 항목에서 final관련해서 하고 싶은 말이 있다.\
http://blog.naver.com/likeme96/220719204817 의 항목 36에서도 말했다시피, 재정의 하기 싫은 함수들 (즉, 가상 함수가 아닌 일반 함수들)에 virtual과 final keyword를 활용해주면 훨씬 좋지 않나라는 생각이 든다.
```cpp
class Base
{
    virtual void normal_function() final {};
    virtual void virtual_function() {};
};

class Derived : public Base
{
    // void normal_function() {};  // COMPILE ERROR
    virtual void virtual_function() override {};
};
```
즉, 위 코드와 같은 식인데, 원래 가상 함수가 아닌 일반 함수는 재정의 하지 않는 것이 관례인데, 이 것을 final을 활용하면, 컴파일단에서 제약을 가할 수 있다. 단, 꺼림칙한 점은 일반 함수인데도 불구하고, final keyword를 사용하기 위해 virtual keyword를 사용해야 하는 것이다. 물론 disassemble을 해서 확인한 결과 overhead는 없음을 확인했지만,, 표준위원회에서 final keyword를 virtual function에만 사용 가능하게 한 것에는 이유가 있지 않을 까라는 생각에 일단 실제 실무에는 사용을 자제하고 있다. 그러나 일반 함수에도 억지로 virtual keyword를 붙여서 final을 활용하는 게 더 좋은 습관이 아닌가 라는 생각이 든다. 그러나 이 부분은 stackoverflow에 물어보던지 해서 다른 고수분들의 자문을 좀 받아봐야 할 것 같다.

**[기억해 둘 사항들]**
- 재정의 함수는 override로 선언하라.
- 멤버 함수 참조 한정사를 이용하면 멤버 함수가 호출되는 객체(*this)의 왼값 버전과 오른값 버전을 다른 방식으로 처리할 수 있다.

### 항목 13. iterator보다 const_iterator를 선호하라.
뭐 워낙 기본적인 것이라 딱히 할 말은 없다. 다만 하고 싶은 말이 하나 있다.\
begin 이나 end등을 사용할 때 비멤버 버전을 사용하는 경우, 내장 배열과 std::begin등이 존재하지 않는 외부 라이브러리 혹은 내부적으로 작성한 container 클래스들도 지원할 수 있게 된다. 만약, 비멤버 버전을 사용하지 않고 지원하려면, 해당 container 클래스를 public상속하여 begin등을 추가로 구현해줘야 한다. 그러나 이 것은 상속을 사용하게 되면서 괜히 설계가 복잡해지고 굳이 subtyping을 하게 되기 때문에 추후 어떤 다른 문제들의 근원이 될 수 도 있다. 따라서 std::begin<> 등에 대해 해당 container 클래스에 대한 완전 특수화를 작성하고, std::begin<> 같은 비멤버 버전을 사용하는 게 옳은 설계이다.\
여기서 위와 같은 설계에 대한 교훈을 얻을 수 있다. C++을 이용한 소프트웨어를 설계할 때, 비슷한 클래스 군들이 같은 연산을 지원한다면 그 연산에 대한 비멤버 버전을 만드는 것을 고려해봄직하다.

**[기억해 둘 사항들]**
- iterator보다 const_iterator를 선호하라.
- 최대한 일반적인 코드에서는 begin, end, rbegin 등의 비멤버 버전들을 해당 멤버 함수들보다 선호하라.

### 항목 14. 예외를 방출하지 않을 함수는 noexcept로 선언하라.
넓은 계약 (wide constract) : 전제조건이 없는 함수들을 말한다.\
좁은 계약 (narrow contract) : 전제조건이 있는 함수들을 말한다.
* 라이브러리 개발자들은 넓은 계약을 가진 함수들에 대해서만 noexcept를 사용하는 경향이 있다.
* C++11부터 모든 메모리 해제 함수와 모든 소멸자는 암묵적으로 noexcept이다.

**noexcept로 인한 성능 향상**
1. 호출 스택이 풀릴 수도 아닐 수도 있게 되면서 컴파일러의 코드 작성이 더 효율적이게 된다.
2. std::vector::push_back, std::swap 등에서 이동 연산들의 noexcept 여부에 따라 라이브러리 코드의 동작이 더 효율적으로 바뀔 수 있다.

책을 읽으면서 인상 깊은 구절을 하나 소개한다.\
> 즉, 의미 있는 것은 함수가 예외를 하나라도 던질 수 있는지 아니면 절대로 던지지 않는지라는 이분법적 정보뿐이다.

**[기억해 둘 사항들]**
- noexcept는 함수의 인터페이스의 일부이다. 이는 호출자가 noexcept 여부에 의존할 수 있음을 뜻한다.
- noexcept 함수는 비except 함수보다 최적화의 여지가 크다.
- noexcept는 이동 연산들과 swap, 메모리 해제 함수들, 그리고 소멸자들에 특히나 유용하다.
- 대부분의 함수는 noexcept가 아니라 예외에 중립적이다.

### 항목 15. 가능하면 항상 constexpr을 사용하라.
C++11부터 추가된 constexpr로 인해 이제 더 이상 compile-time에 계산을 하기 위해 template meta programming을 할 필요가 없어졌다. C++11의 constexpr는 약간 미완성이지만, C++14에서 부족한 점들이 채워졌다. 다만 아쉬운 점은 VS2015에서 아직 C++14의 constexpr 기능을 완전히 구현하지 않은 것이다. 하지만 그래도 충분히 쓸만하다.
```cpp
#include <array>
#include <iostream>

class Example
{
public:
    constexpr Example(int num)
        : m_num(num)
    {}

    constexpr int Num() const noexcept { return m_num; }
    //constexpr Example& operator+=(const Example& ex) noexcept { m_num += ex.m_num; return *this; } // Not compiled in VS2015. But it is okay in C++14.
    //constexpr void SetNum(int num) noexcept { m_num = num; } // Not compiled in VS2015. But it is okay in C++14.

private:
    int m_num;
};

int main()
{
    constexpr Example a(3), b(5);
    //a += b;
    std::array<int, a.Num()> arr;
    std::cout << arr.size();
}
```

**[기억해 둘 사항들]**
- constexpr 객체는 const이며, 컴파일 도중에 알려지는 값들로 초기화된다.
- constexpr 함수는 그 값이 컴파일 도중에 알려지는 인수들로 호출하는 경우에는 컴파일 시점 결과를 산출한다.
- constexpr 객체나 함수는 비constexpr 객체나 함수보다 광범위한 문맥에서 사용할 수 있다.
- constexpr은 객체나 함수의 인터페이스의 일부이다.

### 항목 16.  const 멤버 함수를 스레드에 안전하게 작성하라.
멀티쓰레드 환경이 친숙하고 빈번한 시대가 되었기 때문에 클래스를 thread-safe하게 작성하는 것은 중요하다. 결론적으로는 const 멤버 함수뿐 아니라 모든 함수를 thread-safe하게 작성해야 한다. 그러나 유독 항목에서 const 멤버 함수를 강조하는 이유는 const 멤버 함수는 '읽기 전용 함수' 이기 때문이다. 그래서 동기화에 대한 고려를 안 하기 쉬운데 mutable 멤버 변수를 접근하는 경우 혹은 전역적인 변수들에 접근하는 경우 thread unsafe할 수 있기 때문에 const 멤버 함수의 경우에도 thread-safe를 항상 고려해야 한다. 물론, mutable 멤버 변수나 전역적인 변수들에 접근하지 않더라도 동기화를 고려해야하는 경우가 많다.
```cpp
#include <iostream>
#include <mutex>
#include <atomic>

class Rect_Unsafe
{
public:
    void Set(std::int32_t width, std::int32_t height)
    {
        m_width = width;
        m_height = height;
    }

    std::int32_t Area() const
    {
        return m_width * m_height;
    }

private:
    std::int32_t m_width;
    std::int32_t m_height;
};

class Rect_Safe_1
{
public:
    void Set(std::int32_t width, std::int32_t height)
    {
        std::lock_guard<decltype(m_mutex)> lk(m_mutex);
        m_width = width;
        m_height = height;
    }

    std::int32_t Area() const
    {
        std::lock_guard<decltype(m_mutex)> lk(m_mutex);
        return m_width * m_height;
    }

private:
    mutable std::mutex m_mutex;
    std::int32_t m_width;
    std::int32_t m_height;
};

class Rect_Safe_2
{
public:
    bool CheckLockFree()
    {
        return m_items.is_lock_free();  // return true;
    }

    void Set(std::int32_t width, std::int32_t height)
    {
        struct Items items;
        items.width = width;
        items.height = height;
        m_items = items;
    }

    std::int32_t Area() const
    {
        struct Items items = m_items;
        return items.width * items.height;
    }

private:
    #pragma pack(push, 1)
    struct Items 
    {
        std::int32_t width;
        std::int32_t height;
    };
    #pragma pack(pop)

    std::atomic<struct Items> m_items;
};

class Example
{
    void Work()
    {
        std::lock_guard<decltype(m_mutex)> lk(m_mutex);
        // Do some work...
        m_status = 1;
        // Do some work...
        m_status = 2;
        // Do some work...
        m_status = 3;
    }

    int Status() const { return m_status; }

private:
    std::mutex m_mutex;
    std::atomic<int> m_status;  // "volatile int" is "incorrect".
};

int main()
{
    Rect_Safe_2 rect_safe_2;

    std::cout << (rect_safe_2.CheckLockFree() ? "Lock Free" : "Need Lock") << std::endl;  // Lock Free
    rect_safe_2.Set(12, 11);
    rect_safe_2.Area();
}
```
위 코드에서 Rect_Unsafe 클래스의 Area 멤버 함수의 경우, 그 어떤 변수도 수정하지 않지만 thread-safe하지 않다. 간단하고 범용적인 해결책은 mutex를 이용해 동기화하는 것이다. 그러나 만약 이 경우 같이 두 개의 variable을 한 개의 atomic variable로 묶을 수 있다면 좀 더 효율적으로 해결이 가능하다.\
그리고 위 코드의 Example 클래스를 보자. 실제로 thread-safe하게 클래스를 설계하다보면 getter와 관련해서 위 같은 상황에 맞닥뜨릴 경우가 많다. 이 경우, std::atomic을 사용하면 된다. 주의할 점은 volatile로는 해결이 안된다는 것이다. 왜냐하면 "Do some work"와 m_status = ? 는 서로 연관이 없을테므로 compile 최적화에 의해 순서가 뒤바뀔 수 도 있기 때문이다. 따라서 std::atomic을 사용해야만 이러한 순차적 일관성을 보장할 수 있다.

**[기억해 둘 사항들]**
- 동시적 문맥에서 쓰이지 않을 것이 '확실한' 경우가 아니라면, const 멤버 함수는 스레드에 안전하게 작성하라.
- std::atomic 변수는 뮤텍스에 비해 성능상의 이점이 있지만, 하나의 변수 또는 메모리 장소를 다룰 대에만 적합하다.

### 항목 17. 특수 멤버 함수들의 자동 작성 조건을 숙지하라.
Rule of 5에 따르면 소멸자, 복사 생성자, 복사 대입 연산자, 이동 생성자, 이동 대입 연산자 중 한 개라도 명시적으로 선언을 해야 한다면, 나머지도 직접 선언을 해야만 한다.\
C++11 제정 시에는 기존의 rule of 3 가 충분한 공감대를 얻었기 때문에 소멸자나 복사 연산들 중에 한 개라도 명시적으로 선언이 되어 있으면 이동 연산들은 자동 작성되지 않는 것으로 표준이 제정되었다.\
그러나 C++98 제정 시에는 이와 같은 것들이 충분한 공감대를 얻지 못했고, C++11에 이르어서도 하위 호환성을 유지해야 하기 때문에 여전히 복사 생성자와 복사 대입 연산자는 이동연산들과 자기자신만 선언돼있지 않으면 자동 작성된다. 그러나, 우리는 rule of 5에 따라 소멸자나 복사연산자들이나 이동연산들 중 하나라도 명시적으로 선언돼있다면 복사 연산들도 직접 명시적으로 선언을 하는 것이 좋다. ( = default; 를 사용해도 이것은 명시적 선언이라는 점에 유의하자.)\
소멸자는 항상 기본적으로 작성되며 암시적으로 noexcept이다.

**[기억해 둘 사항들]**
- 컴파일러가 스스로 작성할 수 있는 멤버 함수들, 즉 기본 생성자와 소멸자, 복사 연산들, 이동 연산들을 가리켜 특수 멤버 함수라고 부른다.
- 이동 연산들은 이동 연산들이나 복사 연산들, 소멸자가 명시적으로 선언되어 있지 않은 클래스에 대해서만 자동으로 작성된다.
- 복사 생성자는 복사 생성자가 명시적으로 선언되어 있지 않은 클래스에 대해서만 자동으로 작성되며, 만일 이동 연산이 하나라도 선언되어 있으면 삭제된다. 복사 배정 연산자는 복사 배정 연산자가 명시적으로 선언되어 있지 않은 클래스에 대해서만 자동으로 작성되며, 만일 이동 연산이 하나라도 선언되어 있으면 삭제된다. 소멸자가 명시적으로 선언된 클래스에서 복사 연산들이 자동 작성되는 기능은 비권장이다.
- 멤버 함수 템플릿 때문에 특수 멤버 함수의 자동 작성이 금지되는 경우는 전혀 없다.
