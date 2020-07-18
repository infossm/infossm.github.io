---
layout: post
title: "Boost.Exception 소개"
date: 2019-06-16 23:30
author: taeguk
tags: [C++, Boost, Boost.Exception, Exception, Error Handling]
---

안녕하세요~ 오늘은 Boost.Exception 에 대해서 간단하게 소개해볼까 합니다.

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/06/boostexception.html)**

## Boost.Exception 이란?
[Boost.Exception](https://www.boost.org/doc/libs/1_70_0/libs/exception/doc/boost-exception.html) 은 예외 계층을 설계하고 예외 핸들링을 수행할 때 도움을 주는 라이브러리입니다.
제가 Boost.Exception 을 사용하면서 얻을 수 있었던 이점은 다음과 같습니다.
* 예외가 발생한 시점의 소스파일 이름, 라인 넘버, 함수명을 예외 객체에 쉽고 편하게 담을 수 있습니다.
	* `throw std::exception()` 대신에 `BOOST_THROW_EXCEPTION(std::exception())` 을 사용하면 std::exception 을 wrapping 하는 예외 객체가 만들어지고 그 안에 소스파일 이름, 라인 넘버, 함수명이 담깁니다.
	* 나중에 `diagnostic_information()` 혹은 `current_exception_diagnostic_information()` 을 통해 예외 정보를 출력하면 소스파일 이름, 라인 넘버, 함수명 정보가 같이 출력됩니다.
	* 이러한 정보는 나중에 문제의 원인을 파악할 때 매우 큰 도움이 됩니다.
* 예외 객체의 타입과 예외 객체에 담기는 정보의 종류가 서로 decoupling 됩니다.
	* 즉, 임의 타입의 예외 객체에 임의 타입의 정보를 담을 수 있습니다.
	* 예외 클래스 계통은 어떤 종류의 예외인지를 구분하는 tag 로서 설계하고, 예외 객체를 throw 하는 시점에 throw 측에서 담을 수 있는 임의의 정보들을 임의 타입의 예외 객체에 집어넣을 수 있습니다.
	* [공식문서의 Motivation 문서](https://www.boost.org/doc/libs/1_70_0/libs/exception/doc/motivation.html)를 읽어보시면 좋을 것 같습니다.

## BOOST_THROW_EXCEPTION()
`BOOST_THROW_EXCEPTION(~~~)` 를 사용하면 `~~~` 을 wrapping 하는 예외 객체를 만든후 그 안에 소스파일 이름, 라인 넘버, 함수명을 담아서 throw 하게 됩니다.
```cpp
#include <iostream>
#include <boost/exception/all.hpp>

void testFunc()
{
    BOOST_THROW_EXCEPTION(std::logic_error("test!!!"));
}

int main()
{
    try {
        testFunc();
    }
    catch (...) {
        std::cout << boost::current_exception_diagnostic_information() << std::endl;
    }
}
```
위 프로그램을 실행한 결과는 다음과 같습니다.
```
c:\users\xornr\documents\visual studio 2015\projects\devexam\memposting\main.cpp(6): Throw in function void __cdecl testFunc(void)
Dynamic exception type: struct boost::wrapexcept<class std::logic_error>
std::exception::what: test!!!
```
보시다시피 예외가 발생한 시점의 소스파일 이름, 라인 넘버, 함수명이 출력되는 것을 확인할 수 있습니다. 뿐만 아니라 `std::exception::what()` 의 호출결과도 같이 출력되는 것을 확인하실 수 있습니다. <br/>
여기서 주목하실점은 `Dynamic exception type: struct boost::wrapexcept<class std::logic_error>` 입니다. 예외 객체의 타입이 `std::logic_error` 가 아니라 `boost::wrapexcept<class std::logic_error>` 인 것을 알 수 있습니다. <br/>
원래 `std::logic_error` 객체에는 소스파일 이름등의 정보를 담을 수가 없습니다. 따라서, `BOOST_THROW_EXCEPTION()` 은 그러한 정보들을 담을 수 있는 타입인 `boost::wrapexcept<T>` 를 이용해서 `std::logic_error` 객체를 wrapping 하게 되는 것입니다.

## 예외 객체에 임의의 데이터를 담아서 throw 하기
Boost.Exception 라이브러리를 활용하면 예외 객체에 임의 타입의 데이터를 담아서 throw 할 수가 있습니다.

```cpp
#include <iostream>
#include <boost/exception/all.hpp>

typedef boost::error_info<struct tag_my_errinfo, std::string> my_errinfo;

void testFunc()
{
    throw boost::enable_error_info(std::logic_error("test!!!")) <<
          boost::errinfo_errno(1) <<
          my_errinfo("taeguk.github.io");
}

int main()
{
    try {
        testFunc();
    }
    catch (boost::exception const& x) {
        if (std::string const* e = boost::get_error_info<my_errinfo>(x)) {
            std::cout << "*** My Error Info : " << *e << " ***\n" << std::endl;
        }
        std::cout << boost::diagnostic_information(x) << std::endl;
    }
}
```
위 프로그램의 실행결과는 다음과 같습니다.
```
*** My Error Info : taeguk.github.io ***

Throw location unknown (consider using BOOST_THROW_EXCEPTION)
Dynamic exception type: struct boost::exception_detail::error_info_injector<class std::logic_error>
std::exception::what: test!!!
[struct boost::errinfo_errno_ *] = 1, "Operation not permitted"
[struct tag_my_errinfo *] = taeguk.github.io
```
먼저 예외를 throw 코드를 살펴봅시다. `std::logic_error` 에는 임의의 데이터를 같이 담을수가 없습니다. 따라서 `boost::enable_error_info()` 를 사용해서 `std::logic_error` 를 wrapping 하는 타입의 객체를 만들어야합니다. 실행결과를 보면 `boost::enable_error_info()` 는 `boost::exception_detail::error_info_injector<std::logic_error>` 객체를 반환하는 것을 알 수 있는데요. 이 타입은 `std::logic_error` 와 `boost::exception` 을 다중상속하는 형태로 구현되어 있습니다. `boost::exception` 가 객체에 임의의 데이터를 담을 수 있는 기능을 제공합니다.

`boost::diagnostic_information(x)` 결과를 보면 임의의 데이터 (`boost::errinfo_errno_` 와 `tag_my_errinfo`) 가 같이 출력되는 것을 확인할 수 있습니다. <br/>
또한, `boost::get_error_info<T>()` 를 이용하면 예외 객체에서 특정 타입의 데이터를 얻어낼 수 있습니다. 이 경우 예외 객체에 해당 타입의 데이터가 존재하지 않을 수 있으므로 반드시 null 체크를 해야만 합니다.

## boost::exception 를 상속하는 예외 클래스
위에서 언급했다시피 `boost::exception` 이 임의의 데이터를 삽입하고 얻을 수 있는 기능을 제공합니다. 따라서 이러한 기능을 사용하려면 예외 클래스가 반드시 `boost::exception` 를 상속받아야 합니다. (물론, `boost::enable_error_info()` 를 사용하면 자동으로 `boost::exception` 을 상속받는 예외 클래스를 만들어주긴 합니다.) <br/>
아래는 예시 소스코드와 실행결과입니다.
```cpp
#include <iostream>
#include <boost/exception/all.hpp>

typedef boost::error_info<struct tag_my_errinfo, std::string> my_errinfo;

class MyError : virtual public std::exception,
                virtual public boost::exception
{
public:
    using std::exception::exception;
};

void testFunc()
{
    throw MyError("test!!!") <<
          boost::errinfo_errno(1) <<
          my_errinfo("taeguk.github.io");
}

int main()
{
    try {
        testFunc();
    }
    catch (MyError const& x) {
        if (std::string const* e = boost::get_error_info<my_errinfo>(x)) {
            std::cout << "*** My Error Info : " << *e << " ***\n" << std::endl;
        }
        std::cout << boost::diagnostic_information(x) << std::endl;
    }
}
```
```
*** My Error Info : taeguk.github.io ***

Throw location unknown (consider using BOOST_THROW_EXCEPTION)
Dynamic exception type: class MyError
std::exception::what: test!!!
[struct boost::errinfo_errno_ *] = 1, "Operation not permitted"
[struct tag_my_errinfo *] = taeguk.github.io
```
이제 더이상 `boost::enable_error_info()` 를 사용하지 않아도 되는 것을 알 수 있습니다.

## 마무리
오늘은 Boost.Exception 에 대해서 간단하게 포스팅해봤습니다! <br/>
다음에 또 만나요~~
