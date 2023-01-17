---
layout: post
title: "Effective Modern C++ (3)"
date: 2020-01-18 18:07
author: taeguk
tags: [C++, Modern-C++, Effective-Modern-C++]
---

저번시간에 이어서 오늘은 예전에 Effective Modern C++ 을 공부하며 정리했던 내용들을 포스팅해볼까 합니다~\
C++11/14 에서의 best practice 에 관한 내용으로서 최근 C++20 이 나오는 시점에서 이 또한 최신 내용은 아니긴 하지만 여전히 많은 부분들이 유효합니다.

## Chapter 6. 람다 표현식

### 항목 31. 기본 갈무리 모드를 피하라.

참고로, 갈무리는 'capture'를 의미한다. (첨엔 어색했으나 나도 이제 완전히 이 번역에 익숙해져버렸다.) \
기본 갈무리 모드 (default capture mode)는 '참조 갈무리 모드'와 '값 갈무리 모드'가 있다. 기본 갈무리 모드를 피해야 하는 이유는 크게 dangling reference 위험과 명시적이지 않다는 점(그래서 착각/실수 할 수 있다는 점) 때문이다.  \
일단, 참조 갈무리 모드에서 dangling reference 문제가 발생할 수 있다는 것은 너무 당연하고, 값 갈무리 모드에서도 pointer가 capture되는 경우에 그러한 문제가 발생 할 수 있다는 것은 당연하다.  \
사실 결정적인 문제는 명시적이지 않다는 점이다. 이로 인해 문제가 발생 할 수 있을 만한 상황을 정리하면 다음과 같다.  
1. this pointer의 캡처  
    - 숨겨져있던 this pointer가 캡처됨으로 인해 dangling pointer 등의 문제가 발생할 수 있다.  
2. 람다표현식을 복붙(Copy&Paste) 할 경우, 실수할 가능성이 커진다.  
    - 유용한 람다표현식을 그대로 다른 곳에서 쓸 경우가 있을 수 있다. 이 경우, capture list가 명시적이지 않아서 문제들이 발생할 소지가 있다.  
3. 전역 변수나 static 같이 global scope를 가진 변수들 (즉, non-automatic variable들)의 값이 복사로서 capture될 것이라는 착각을 할 수 있다.  
    - 오직 automatic variable들(즉, 지역변수들) 만 capture될 수 있는데, 기본 값 갈무리 모드를 사용하면, non-automatic variable들도 값 복사로서 capture될 것이라는 착각을 할 여지가 있다.  
  
즉, 기본 갈무리 모드의 문제는 결국 '명시적이지 않다'는 것이다. 필요한 것들만 명시적으로 capture해주는 게 더 좋은 코드라고 할 수 있겠다.  
  
**[기억해 둘 사항들]**  
- 기본 참조 갈무리는 참조가 대상을 잃을 위험이 있다.  
- 기본 값 갈무리는 포인터(특히 this)가 대상을 잃을 수 있으며, 람다가 자기 완결적이라는 오해를 부를 수 있다.  
  

### 항목 32. 객체를 클로저 안으로 이동하려면 초기화 갈무리를 사용하라.

std::unique_ptr, std::future, std::future과 같이 이동 전용인 경우나 효율성 측면에서 이동이 필요할 경우, 클로저안으로 객체를 이동할 수 있어야 한다.  \
일단 C++14에서는 초기화 갈무리 (init capture)를 사용하면 되므로 너무 쉽다.  \
그러나 C++11에서는 초기화 갈무리가 없으므로 문제다. (C++11 얘기하기 너무 싫다... 맘편히 C++14이상만 생각하고 싶다!!!)  \
이에 대한 해결책은 다음과 같다.  
1. Functor를 만들어서 해결한다.  
2. std::bind를 활용한다.  
    - std::bind를 통해 람다의 매개변수에 왼값 참조로서 이동된 객체를 묶는다. (아래 코드 참고)  
```cpp
#include <memory>
#include <functional>
#include <iostream>

class Example {};

int main()
{
    std::unique_ptr<Example> p;

    auto func_in_cpp_11 = std::bind(
        [](const std::unique_ptr<Example>& p) {
            std::cout << "I'm an example in C++11 :(" << std::endl;
        },
        std::move(p)
    );

    auto func_in_cpp_14 = [p = std::move(p)]() {
        std::cout << "I'm an example in C++14 :)" << std::endl;
    };

    func_in_cpp_11();
    func_in_cpp_14();
}

void lambda_mutable_test()
{
    class Lambda
    {
        int& a;
        int b;
    public:
        Lambda(int &a, int b) : a(a), b(b) {}
        void operator()() const {
            a = 1;
            b = 2;  // Compiler Error
        }
    };

    int a, b;
    [&a, b]() {
        a = 1;
        b = 2;  // Compiler Error
    };
    Lambda lambda(a, b); lambda();

    ////////////////////////////////////////////
    // Think "int &" is similar to "const int *".

    class MutableLambda
    {
        int & a;
        int b;
    public:
        MutableLambda(int &a, int b) : a(a), b(b) {}
        void operator()() {
            a = 1;
            b = 2;
        }
    };

    int a, b;
    [&a, b]() mutable {
        a = 1;
        b = 2;
    };
    MutableLambda mutableLambda(a, b); mutableLambda();
}

```

이번 항목에 직접적인 연관은 없지만, Lambda mutable에 관한 간단한 실험 코드도 같이 첨부하였다.  
  
**[기억해 둘 사항들]**  
- 객체를 클로저 안으로 이동할 때에는 C++14의 초기화 갈무리를 사용하라.  
- C++11에서는 직접 작성한 클래스나 std::bind로 초기화 갈무리를 흉내 낼 수 있다.  
  

### 항목 33. std::forward를 통해서 전달할 auto&& 매개변수에는 decltype을 사용하라.

이번 항목은 C++14에만 해당하는 내용이다. C++14에 generic lambdas가 추가되면서, 매개변수 명세에 auto를 사용할 수 있게 되었다. 만약 lambda 내에서 perfect forwarding을 하고 싶을 경우, std::forward<decltype(...)>(...) 처럼 하면 된다.  
  
**[기억해 둘 사항들]**  
- std::forward를 통해서 전달할 auto&& 매개변수에는 decltype을 사용하라.  
  

### 항목 34. std::bind보다 람다를 선호하라.

std::bind보다 람다를 선호해야 할 이유는 '가독성'과 '성능'이다.\
일단 가독성 측면에서, std::bind로 하려면 난해하고 복잡한 것을 람다로는 간단하고 명료하게 할 수 있다.  \
그리고, 최적화 가능성이 람다가 더 많기 때문에 성능 측면에서도 std::bind보다 람다가 우월하다. (자세한 건 아래 참조)

```cpp
/*
    https://godbolt.org/g/25uYMS
*/

#include <vector>
#include <functional>

class Functor {
private:
    int a;
    int b;
public:
    Functor(int a, int b) : a(a), b(b) {}
    bool operator()(int n) const { return a < n && n < b; }
};

bool comp(int a, int b, int n) { return a < n && n < b; }

bool test_bind_function_pointer(int a, int b, int c)
{
    auto bind_func = std::bind(comp, a, b, std::placeholders::_1);
    std::vector<decltype(bind_func)> vec;
    vec.emplace_back(bind_func);
    return vec.back()(c);
}

bool test_bind_functor(int a, int b, int c)
{
    auto bind_func = std::bind(Functor(a, b), std::placeholders::_1);
    std::vector<decltype(bind_func)> vec;
    vec.emplace_back(bind_func);
    return vec.back()(c);
}

bool test_functor(int a, int b, int c)
{
    std::vector<Functor> vec;
    vec.emplace_back(a, b);
    return vec.back()(c);
}

bool test_lambda(int a, int b, int c)
{
    auto lambda = [a, b](int n) { return a < n && n < b; };
    std::vector<decltype(lambda)> vec;
    vec.emplace_back(lambda);
    return vec.back()(c);
}

int main() 
{
    test_bind_function_pointer(1, 2, 3);
    test_bind_functor(1, 2, 3);
    test_functor(1, 2, 3);
    test_lambda(1, 2, 3);
}

```

위 코드는 std::bind와 lambda의 최적화를 비교하기 위해 작성한 코드로서, [https://godbolt.org/g/25uYMS](https://godbolt.org/g/25uYMS) 에서 assemble된 결과를 볼 수 있다. gcc 6.3 -O3 -std=c++14 에서의 결과를 분석해보겠다.  \
일단 test_bind_function_pointer의 경우를 보면, std::bind 에 function pointer를 넘겨주기 때문에 어셈블리 상에서 call [QWORD PTR [rax-16]] 와 같이 동작하는 것을 볼 수 있다. 이러한 동작은 std::bind의 반환형이 std::_Bind<bool (*(int, int, std::_Placeholder<1>))(int, int, int)> 라는 것에 기인한다. callable object 부분의 type이 function pointer이므로 이와 같이 동작할 수 밖에 없는 것이다. 그러나 사실 그냥 comp를 호출하거나 comp를 inlining 하도록 최적화되기를 바랄 것이다. 실제로 똑같은 코드를 clang으로 컴파일할 경우, 이러한 최적화를 수행해줌을 확인할 수 있다. 하지만, 이러한 최적화를 범용적으로 기대하는 것은 힘들다.  \
두 번째로, test_bind_functor 같은 경우는 std::bind가 반환하는 타입이 std::_Bind<Functor (std::_Placeholder<1>) 이다. 즉, 호출될 functor object의 타입이 class로서 명확히 드러나므로, Functor::operator() 를 직접적으로 호출하거나 이 것이 inlining 될 것임을 기대할 수 있다. test_functor 도 마찬가지 이유에서 최적화를 기대할 수 있다.  \
마지막으로, test_lambda 이다. 이 경우, 가장 많은 최적화가 된 것을 볼 수 있다. lambda는 내부적으로 functor 로서 구현된다. 그런데 유독 람다의 경우 최적화가 더 많이 되는 이유는 무엇일까? 그 것은 바로 람다의 경우는 컴파일러가 자동적으로 생성하는 functor 를 사용하기 때문이다. 즉, test_bind_functor 나 test_functor 에서는 사용자가 정의한 functor 를 사용하므로 컴파일러가 사용자 타입에 대해 자세히 알지 못하지만, 람다의 경우 functor 를 컴파일러가 만들기 때문에 더 많은 정보를 가질 수 있고, 이로 인해 더 많은 최적화를 수행할 수 있다.  
  
즉, 결론적으로 이러한 이유에서 std::bind보다 람다를 선호해야한다.  \
그러나 C++11에서는 어쩔 수 없이 std::bind를 활용해야 하는 경우가 다음과 같이 2가지 있다.  
1. move semantics의 활용 (항목 32에서 다룸)  
2. polymorphic function object  
그러나, C++14에서는 각각 초기화 갈무리와 generic lambdas에 의해 해결되었으므로, C++14부터는 람다를 적극 사용해야 한다.  
  
**[기억해 둘 사항들]**  
- std::bind를 사용하는 것보다 람다가 더 읽기 쉽고 표현력이 좋다. 그리고 더 효율적일 수 있다.  
- C++14가 아닌 C++11에서는 이동 갈무리를 구현하거나 객체를 템플릿화된 함수 호출 연산자에 묶으려 할 때 std::bind가 유용할 수 있다.

## Chapter 7. 동시성 API

### 항목 35. 스레드 기반 프로그래밍보다 과제 기반 프로그래밍을 선호하라.

확실히 과제 기반 프로그래밍은 스레드 기반 프로그래밍보다 더 쉽고 가독성이나 설계 측면에서 더 우월하다. 하지만, 과하게 마구잡이로 사용하는 것을 주의해야 한다고 느꼈다. 스레드를 쓰면 간단하게 되는 것을 과제 기반으로 하려고 억지로 짜 맞추는 것은 좋지 않다.  \
그리고 과제 기반 프로그래밍은 스레드 기반 보다 더 높은 추상화로서, 내부적으로 스레드 고갈, over subscription, load balancing등의 문제를 해결해줄 수 도 있다. 실제로 현재에는 std::async가 visual studio에서는 PPL을 바탕으로 위와 같은 것을 어느 정도 해결해주고 있다. 그러나, libc++ (llvm의 std 구현)과 libstdc++ (gcc의 std구현)의 경우 비효율적인 방법으로 구현되어 있다. (관련 내용 : [http://rpgmakerxp.tistory.com/63](http://rpgmakerxp.tistory.com/63))  \
정말 비동기 작업을 효율적으로 진행해야 하고, 더 많은 부분들(thread priority, affinity, thread pool등)을 직접 컨트롤 해야하는 경우에는 std::async가 적합하지 않다. 이런 경우는 std::thread 혹은 Windows API / pthreads 등을 사용해야 한다.  \
그리고, 아직 C++11/14의 과제 기반 프로그래밍은 빈약한 편이므로, 좀 더 본격적인 과제 기반 프로그래밍이 필요한 경우에는 TBB, PPL, HPX등을 사용하는 것이 좋아 보인다. (그러나, C++의 향후 표준들에서 점차 강력해질 것이다.)  
  
**[기억해 둘 사항들]**  
- std::thread API에서는 비동기적으로 실행된 함수의 반환값을 직접 얻을 수 없으며, 만일 그런 함수가 예외를 던지면 프로그램이 종료된다.  
- 스레드 기반 프로그래밍에서는 스레드 고갈, 과다구독, 부하 균형화, 새 플랫폼으로의 적응을 독자가 직접 처리해야 한다.  
- std::async와 기본 시동 방침을 이용한 과제 기반 프로그래밍은 그런 대부분의 문제를 알아서 처리해준다.  
  

### 항목 36. 비동기성이 필수일 때에는 std::launch::async를 지정하라.
  
**[기억해 둘 사항들]**  
- std::async의 기본 시동 방침은 과제의 비동기적 실행과 동기적 실행을 모두 허용한다.  
- 그러나 이러한 유연성 때문에 thread_local 접근의 불확실성이 발생하고, 과제가 절대로 실행되지 않을 수도 있고, 시간 만료 기반 wait 호출에 대한 프로그램 논리에도 영향이 미친다.  
- 과제를 반드시 비동기적으로 실행해야 한다면 std::launch::async를 지정하라.  
  

### 항목 37. std::thread들을 모든 경로에서 합류 불가능하게 만들어라.

이 것은 되게 중요한 문제이다. 핵심은 단순하다. std::thread들을 모든 경로에서 합류 불가능 (unjoinable)하게 만들어야 한다. 만약 joinable한 상태에서 std::thread 객체의 소멸자가 호출되면 프로그램은 그냥 종료 되어버린다. 표준 위원회가 이런 선택을 한 것은 이유가 있다. 만약 그렇게 하지 않으려면, thread가 암묵적으로 join이 되거나 detach가 되는 방법을 채택해야 하는데, 이 것은 더 심각한 문제들을 가져올 수 있기 때문이다. 따라서 std::thread 객체는 항상 unjoinable한 상태에서 소멸되도록 표준을 제정한 것이 옳다.  

```cpp
#include <thread>
#include <condition_variable>
#include <iostream>
#include <atomic>
#include <vector>
#include <chrono>
#include <numeric>

class BackgroundWorkerPool
{
public:
    BackgroundWorkerPool()
    {
        std::cout << "[Main Thread] BackgroundWorkerPool() \n";
        m_worker[0] = std::thread(&BackgroundWorkerPool::BackgroundWorkerMain, this, 1);
        m_worker[1] = std::thread(&BackgroundWorkerPool::BackgroundWorkerMain, this, 2);
    }

    ~BackgroundWorkerPool()
    {
        std::cout << "[Main Thread] ~BackgroundWorkerPool() \n";
        m_workerTerminated = true;
        m_cvData.notify_all();
        m_worker[0].join();
        m_worker[1].join();
        std::cout << "[Main Thread] After joining worker threads. \n";
    }

    void FeedData(const std::vector<int>& data)
    {
        {
            std::lock_guard<decltype(m_mutex)> lk(m_mutex);
            std::cout << "[Main Thread] Feed data. \n";
            m_data.insert(std::end(m_data), std::begin(data), std::end(data));
            m_dataExist = true;
        }
        m_cvData.notify_one();
    }

private:
    void BackgroundWorkerMain(int id)
    {
        while (true)
        {
            std::unique_lock<decltype(m_mutex)> lk(m_mutex);
            m_cvData.wait_for(lk, std::chrono::milliseconds(3000), [this]() {
                return m_dataExist || m_workerTerminated;
            });

            std::cout << "[Worker Thread " << id << "] Wake up! \n";

            if (m_workerTerminated)
                break;

            std::cout << "[Worker Thread " << id << "] WORK! \n";
            auto result = std::accumulate(std::begin(m_data), std::end(m_data), 0);
            std::cout << "[Worker Thread " << id << "] Result = " << result << " \n";
            m_data.clear();
            m_dataExist = false;
        }

        std::cout << "[Worker Thread " << id << "] I'm terminated. \n";
    }

    std::mutex m_mutex;
    std::atomic<bool> m_workerTerminated{ false };
    std::vector<int> m_data;
    bool m_dataExist{ false };
    std::condition_variable m_cvData;
    std::thread m_worker[2];
};

int main()
{
    BackgroundWorkerPool workerPool;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    workerPool.FeedData({ 1,2,3,4,5,6,7,8 });
    workerPool.FeedData({ 1 });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    workerPool.FeedData({ 1,2,3,4,5 });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    workerPool.FeedData({ 1,2 });
    workerPool.FeedData({ 1,2,3,4,5,6,7,8,9 });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    workerPool.FeedData({ 1,2,3,4,5,6,7,8,9,10,11,12 });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    /* Execution Result
        [Main Thread] BackgroundWorkerPool()
        [Main Thread] Feed data.
        [Main Thread] Feed data.
        [Worker Thread 2] Wake up!
        [Worker Thread 2] WORK!
        [Worker Thread 2] Result = 37
        [Main Thread] Feed data.
        [Worker Thread 2] Wake up!
        [Worker Thread 2] WORK!
        [Worker Thread 2] Result = 15
        [Main Thread] Feed data.
        [Main Thread] Feed data.
        [Worker Thread 1] Wake up!
        [Worker Thread 1] WORK!
        [Worker Thread 1] Result = 48
        [Main Thread] Feed data.
        [Worker Thread 1] Wake up!
        [Worker Thread 1] WORK!
        [Worker Thread 1] Result = 78
        [Main Thread] ~BackgroundWorkerPool()
        [Worker Thread 2] Wake up!
        [Worker Thread 2] I'm terminated.
        [Worker Thread 1] Wake up!
        [Worker Thread 1] I'm terminated.
        [Main Thread] After joining worker threads.
    */
}

```
  
Thread를 합류 불가능하게 만들려면, thread를 detach하거나 join해야 한다. 근데, detach를 하는 것은 깔끔하지 못한 일종의 꼼수책이다. 따라서 join을 해서 끝내는 것이 깔끔하고 더 일반적인 방법인데 이 방법을 사용하려면, thread를 외부에서 종료 시킬 수 있어야 한다. Windows API의 TerminatedThread 같은 것을 이용해서 강제적으로 종료 시키면 각종 많은 문제들이 발생하기 때문에 graceful stop을 구현해야만 한다. 나는 보통 위 코드와 같이 구현해서 사용한다. 이 때 blocking I/O같이 infinite하게 blocking되는 함수 호출이나, 오랜 시간이 걸리는 연산을 수행하는 코드를 주의해야 한다. 즉, 수시로 termination 여부를 polling 할 수 있어야 하고, 이러한 polling의 간격이 곧 thread termination 요청을 하고 나서 실제 join이 될 때 까지의 응답 시간이 된다. 위 코드의 BackgroundWorkerPool의 소멸자에서 m_cvData.notify_all(); 를 한 것과 같이 thread 내의 blocking operation을 직접 깨울 수 있는 방법이 있을 때는 그러한 응답 시간을 더 줄일 수 있다.  \
아무튼 나는 위 코드와 같은 방법을 사용하는 데, 더 좋은 방법이 있는 지는 잘 모르겠다. 혹시 이 글을 보는 분들 중 더 좋은 방법을 아는 분이 계시면 알려주시면 감사하겠다. (C++ Concurrency in action 책에 interruptible thread 라는 개념이 등장한다던데 결국 근본적으로는 내가 쓰는 방법과 같은 것으로 보인다.)  
  
**[기억해 둘 사항들]**  
- 모든 경로에서 std::thread를 합류 불가능으로 만들어라.  
- 소멸 시 join 방식은 디버깅하기 어려운 성능 이상으로 이어질 수 있다.  
- 소멸 시 detach 방식은 디버깅하기 어려운 미정의 행동으로 이어질 수 있다.  
- 자료 멤버 목록에서 std::thread 객체를 마지막에 선언하라.  
  

### 항목 38. 스레드 핸들 소멸자들의 다양한 행동 방식을 주의하라.

std::thread와 비지연 과제에 대한 미래 객체는 시스템 스레드에 대응된다는 점에서 모두 '시스템 스레드 핸들' 이라고 말할 수 있다. 이러한 스레드 핸들의 소말자에 행동 방식에 대해 다루는 것이 이 항목이다. 일단 std::thread의 소멸자에 대한 부분은 항목 37에서 다루고 있다. 그러므로 이 항목에서는 미래 객체 소멸자의 행동 방식들에 대해 다룬다.  \
행동 방식은 제각각인 것처럼 보일 수 있지만, 사실 핵심은 간단하다.  \
**"시스템 스레드과 대응되는 유일한 미래객체일 경우에만, 시스템 스레드에 대해 암묵적 join을 수행한다. 그 외의 경우에는 그냥 바로 객체가 소멸된다."**  \
즉, 어떤 미래 객체가 소멸자에서 암묵적 join을 수행할 지, 아니면 그냥 바로 객체가 소멸되고 끝날 지 판단하는 것은 이 미래 객체가 시스템 스레드와 대응되는 유일한 미래 객체인지만 생각해보면 된다.  \
몇 가지 케이스를 통해 살펴보자.  
1. 지연된 과제에 대한 미래 객체  
    - 대응되는 시스템 스레드가 없으므로 바로 객체가 소멸된다.  
2. std::async (with std::launch::async) 호출에 의해 생성된 공유 상태를 참조하는 std::future  
    - 비동기적으로 실행된 과제의 공유 상태에 유일하게 대응되는 미래 객체이므로 암묵적 join이 수행된 뒤 객체가 소멸된다.  
3. std::async (with std::launch::async) 호출에 의해 생성된 공유 상태를 참조하는 std::shared_future  
    - 여러 개의 std::shared_future가 공유 상태를 참조하고 있을 것이다.  
    - 따라서 공유 상태를 참조하고 있는 마지막 std::shared_future 만 암묵적 join이 수행된다.  
4. std::packaged_task가 std::thread에 의해 실행되고 있을 때, std::packaged_task으로 부터 얻어진 std::future  
    - 시스템 스레드가 std::thread에 대응되므로, std::future는 시스템 스레드에 대한 책임이 없다. 따라서 바로 객체가 소멸된다.  
  
즉, 결과적으로만 보면 "std::async를 통해서 시동된 비지연(지연되지 않은) 과제에 대한 공유 상태를 참조하는 마지막 미래 객체의 소멸자"의 경우에만 과제가 완료될 때까지 차단 (즉, 암묵적 join) 되는 것이다.  \
하지만, 이 것의 근본적인 이유는 위에서 말한 '핵심' 때문이다.  \
그리고 더 나아가면 사실 std::thread 소멸자의 행동 방식과도 일맥상통하는 원리를 얻을 수 있다.  \
**결국에는 시스템 스레드 (바탕 스레드)에 대응되는 마지막 핸들이 시스템 스레드를 책임져야하는 것인데, 시스템 스레드가 종료되기 전에 핸들(객체)가 소멸될 경우, 바로 객체가 소멸되지 않고 특별한 행동이 일어나는 것이다.**  \
그리고 그 특별한 행동은 std::thread의 경우에는 프로그램 종료, 미래 객체의 경우에는 암묵적 join인 것이다.  \
다만, 표준위원회가 왜 std::thread와 미래 객체에 대해 특별한 행동으로서 서로 다른 것을 선택했는지는 잘 모르겠다.  \
아무튼, 나는 그래서 std::thread는 항상 unjoinable한 상태에서만 소멸할 수 있도록 코드를 짜고, 미래 객체가 소멸하면서 암시적 join이 일어날 수 있는 곳에는 주석으로서 그 사실을 명시한다.  
  
**[기억해 둘 사항들]**  
- 미래 객체의 소멸자는 그냥 미래 객체의 자료 멤버들을 파괴할 뿐이다.  
- std::async를 통해 시동된 비지연 과제에 대한 공유 상태를 참조하는 마지막 미래 객체의 소멸자는 그 과제가 완료될 때까지 차단된다.  
  

### 항목 39. 단발성 사건 통신에는 void 미래 객체를 고려하라.

일반적으로 thread 사이에서 어떤 사건(이벤트)를 통지하거나 흐름을 통제하고 싶을 경우, condition variable과 flag 변수가 조합되어 사용된다. 그러나 단발성 사건 통신의 경우에는 void 미래 객체를 활용하면 훨씬 간단하고 깔끔하게 설계가 가능하다.  
  
**[기억해 둘 사항들]**  
- 간단한 사건 통신을 수행할 때, 조건 변수 기반 설계에는 여분의 뮤텍스가 필요하고, 검출 과제와 반응 과제의 진행 순서에 제약이 있으며, 사건이 실제로 발생했는지를 반응 과제가 다시 확인해야 한다.  
- 플래그 기반 설계를 사용하면 그런 단점들이 없지만, 대신 차단이 아니라 폴링이 일어난다는 단점이 있다.  
- 조건 변수와 플래그를 조합할 수 도 있으나, 그런 조합을 이용한 통신 메커니즘은 필요 이상으로 복잡하다.  
- std::promise와 미래 객체를 사용하면 이러한 문제점들을 피할 수 있지만, 그런 접근방식은 공유 상태에 힙 메모리를 사용하며, 단발성 통신만 가능하다.  
  

### 항목 40. 동시성에는 std::atomic을 사용하고, volatile은 특별한 메모리에 사용하라.

동시성에 있어서 그냥 일반적인 변수를 사용할 때 데이터 레이스등의 문제가 발생하는 이유는 다음과 같다.  
1. 변수가 레지스터에 할당(캐시) 될 수 있다.  
2. Compiler instruction reordering에 의해 memory access의 순서가 바뀔 수 있다.  
    - A; B; 와 같은 코드가 compiler에 의해 B; A; 와 같이 순서가 바뀔 수 있다.  
    - [http://preshing.com/20120625/memory-ordering-at-compile-time/](http://preshing.com/20120625/memory-ordering-at-compile-time/)  
3. Memory visibility가 보장되지 않을 수 있다.  
    - 프로세서의 out of order execution등에 의해 memory visibility가 보장되지 않을 수 있다.  
    - 프로세서마다 memory consistency model이 다르다.  
    - [http://preshing.com/20120930/weak-vs-strong-memory-models/](http://preshing.com/20120930/weak-vs-strong-memory-models/)  
    - [http://www.kandroid.org/board/board.php?board=AndroidBeginner&command=body&no=102](http://www.kandroid.org/board/board.php?board=AndroidBeginner&command=body&no=102)  
    - [http://egloos.zum.com/studyfoss/v/5141402](http://egloos.zum.com/studyfoss/v/5141402)  
    - [http://stackoverflow.com/questions/7346893/out-of-order-execution-and-memory-fences](http://stackoverflow.com/questions/7346893/out-of-order-execution-and-memory-fences) (x86)  
  
이 3가지 이유중에 '표준' volatile'은 오직 1번 밖에 해결하지 못한다. 그러나 몇몇 컴파일러들을 volatile keyword에 대해 추가적으로 다른 문제점들을 해결해주기도 한다. 하지만, 표준적으로는 volatile을 동시성에 사용하는 것은 매우 위험한 행위이고, 위의 3가지 문제를 모두 해결하려면 C++11이후의 memory fense, std::atomic 등의 기능을 사용해야만 한다.  \
그렇다고 std::atomic이 volatile의 기능을 포함하고 있다는 식으로 착각하면 안된다. 예를 들면, x = 1; x = 2; 와 같은 코드의 경우, x가 volatile로 선언되있을 경우에는 최적화가 발생하지 않지만, x가 std::atomic인 경우 x = 2; 와 같은 식으로 최적화가 진행 될 수 있다. 따라서, volatile과 std::atomic은 포함 관계가 아닌 아예 서로 다른 역할과 기능을 가지고 있는 것으로 인식 하는 것이 옳다.  \
또 특별히 우리가 많이 사용하는 x86 의 경우에는 강력한 memory consistency model을 가지고 있다. 따라서 read-after-write의 경우를 제외하면 sequentially consistency를 가지고 있다고 할 수 있다. (자세한 건 위 링크들 참조) 따라서, 그냥 동시성을 위해 volatile을 써도 평소에 큰 문제가 생기지 않았을 가능성이 크고, 이에 따라 많은 사람들이 동시성에 volatile을 사용하는 실수와 착각을 하고 있다고 생각한다. (나도 과거에 그런 착각을 하던 시절이 있었다.)  
  
**[기억해 둘 사항들]**  
- std::atomic은 뮤텍스 보호 없이 여러 스레드가 접근하는 자료를 위한 것으로, 동시적 소프트웨어의 작성을 위한 도구이다.  
- volatile은 읽기와 기록을 최적화로 제거하지 말아야 하는 메모리를 위한 것으로, 특별한 메모리를 다룰 때 필요한 도구이다.

## Chapter 8. 다듬기

### 항목 41. 이동이 저렴하고 항상 복사되는 복사 가능 매개변수에 대해서는 값 전달을 고려하라.

왼값 참조와 오른값 참조 버전 2개를 모두 만드는 것은 코드 중복, 유지보수 측면에서 단점이고, 보편 참조 전달 버전은 항목 26/27/30 에서 말했던 것들과 같은 문제들이 발생할 수 있다. 따라서 약간의 효율성을 포기하면 이러한 단점들을 피할 수 있는데, 그것이 바로 값 전달을 활용하는 것이다. (효율성 : 보편 참조 버전 >= 왼값 참조 버전 + 오른값 참조 버전 >= 값 전달 버전)  \
그러나, 값 전달의 경우에도 주의해야 할 점들이 있다.  
1. 잘림 문제 (slicing problem)  
2. 값 전달 함수들이 꼬리를 물고 호출되면, 전체적인 성능이 급격히 하락할 수 있다.  
3. "값 전달 (복사 생성) -> 이동 배정" 의 경우 "참조 전달 -> 복사 배정" 보다 훨씬 비쌀 가능성이 있다. (예를 들면, std::string이나 std::vector등의 memory allocation 때문에)  
  
흠, 나는 값 전달 버전이 얼마나 유용할 지 의문이 든다. 일단 이동 연산 하나가 불필요하게 낭비된다. 사실 이동이 저렴한 경우에는 유지보수, 코드중복 해결의 장점을 봤을 때 이러한 사소한 낭비를 무시할 수 있다. 그러나 사실 큰 문제는 이동이 저렴하다고 생각했는데 저렴하지 않을 수 도 있다는 것이다. std::string이나 std::vector같은 경우, memory allocation 때문에 값 전달 후 이동 배정을 하는 것이 상당히 느려질 수 있는데, 코드를 짤 때 이러한 점을 간과하거나 실수 할 가능성이 크다. 또한, 잘림 문제도 주의해야 한다. 즉, 값 전달을 사용하는 것은 참조 버전보다 실수할 가능성이 더 크기 때문에 일단 기본적으로는 피해야 한다고 생각한다. 내 생각에는 일단 우선적으로 왼값/오른값 참조 버전을 사용하다가, 성능을 더 강화할 필요가 있을 경우에는 보편 참조 버전으로 바꾸는 것이 옳고, 그리고 왼값/오른값 참조 버전들 끼리의 코드 중복이 심해질 경우에 값 전달 버전 혹은 보편 참조 버전을 고려하는 것이 옳다고 생각한다.  
  
**[기억해 둘 사항들]**  
- 이동이 저렴하고 항상 복사되는 복사 가능 매개변수에 대해서는 값 전달이 참조 전달만큼이나 효율적이고, 구현하기가 더 쉽고, 산출되는 목적 코드의 크기도 더 작다.  
- 왼값 인수의 경우 값 전달(즉, 복사 생성) 다음의 이동 배정은 참조 전달 다음의 복사 배정보다 훨씬 비쌀 가능성이 있다.  
- 값 전달에서는 잘림 문제가 발생할 수 있으므로, 일반적으로 기반 클래스 매개변수 형식에 대해서는 값 전달이 적합하지 않다.  
  

### 항목 42. 삽입 대신 생성 삽입을 고려하라.

얼핏 생각하면 무조건 생성 삽입을 사용하는 것이 옳다고 생각할 수 있다. 나도 그래서 옛날에는 무조건 생성 삽입만을 사용하였다. 그러나 실제적으로 그냥 삽입이 생성 삽입보다 빠를 수도 있고, 삽입과 생성 삽입의 본질적인 차이를 생각해보면 '무조건'은 아니라는 생각이 들 것이다.  

```cpp
#include <set>
#include <iostream>

class Example
{
public:
    explicit Example(int a, int b)
        : m_a(a), m_b(b)
    {
        std::cout << "Example(int, int) called. \n";
    }
    ~Example()
    {
        std::cout << "~Example() called. \n";
    }
    Example(const Example& other)
        : m_a(other.m_a), m_b(other.m_b)
    {
        std::cout << "Example(const Example&) called. \n";
    }
    Example(Example&& other)
        : m_a(other.m_a), m_b(other.m_b)
    {
        std::cout << "Example(Example&&) called. \n";
    }
    Example& operator=(const Example& other)
    {
        std::cout << "operator=(const Example&) called. \n";
        m_a = other.m_a;
        m_b = other.m_b;
        return *this;
    }
    Example& operator=(Example&& other)
    {
        std::cout << "operator=(Example&&) called. \n";
        m_a = other.m_a;
        m_b = other.m_b;
        return *this;
    }

    bool operator<(const Example& other) const { return m_a + m_b < other.m_a + m_b; }

private:
    int m_a, m_b;
};

int main()
{
    std::set<Example> s;
    Example ex(11, 22);
    s.insert(ex);

    Example ex1(33, 44), ex2(55, 66);
    std::cout << "\n";
    std::cout << "-- insert when not duplicated --\n";
    s.insert(ex1);
    std::cout << "--------------------------------\n\n";
    std::cout << "-- emplace when not duplicated --\n";
    s.emplace(ex2);
    std::cout << "---------------------------------\n\n";

    std::cout << "-- insert when duplicated --\n";
    s.insert(ex);
    std::cout << "----------------------------\n\n";
    std::cout << "-- emplace when duplicated --\n";
    s.emplace(ex);
    std::cout << "-----------------------------\n\n";

    /* Execution Result
        Example(int, int) called.
        Example(const Example&) called.
        Example(int, int) called.
        Example(int, int) called.

        -- insert when not duplicated --
        Example(const Example&) called.
        --------------------------------

        -- emplace when not duplicated --
        Example(const Example&) called.
        ---------------------------------

        -- insert when duplicated --
        ----------------------------

        -- emplace when duplicated --
        Example(const Example&) called.
        ~Example() called.
        -----------------------------

        ~Example() called.
        ~Example() called.
        ~Example() called.
        ~Example() called.
        ~Example() called.
        ~Example() called.
    */
}

```
  
위 코드를 보면 알 수 있듯이, std::set과 같이 값의 중복이 금지된 컨테이너의 경우, 생성 삽입이 그냥 삽입보다 더 느릴 수도 있다. 왜냐하면, 생성 삽입의 경우 내부적으로 중복 체크를 위해서 임시 객체를 생성하는 반면에, 그냥 삽입은 참조로서 넘어온 객체를 중복 체크를 위해 사용하기 때문이다. 따라서, 내부적으로 중복 체크를 하는 컨테이너의 경우, 무작정 생성 삽입을 사용하다가는 성능이 더 하락할 수 있음을 명심해야 한다.  \
그리고 그냥 삽입과 생성 삽입 사이의 근본적인 차이에서 오는 유의점들이 있다. 첫째로, 그냥 삽입은 복사 초기화를 사용하므로 explicit 생성자를 사용 불가능한 반면, 생성 삽입은 내부적으로 직접 초기화를 사용해서 explicit 생성자를 호출할 수 있다. 따라서 그냥 삽입에서는 compiler error가 나는 것이, 생성 삽입에서는 정상적으로 컴파일될 수 있다. 두 번째로, 생성 삽입은 객체의 생성이 컨테이너 내부의 메모리까지 지연되므로, 예외 안정성 측면에서 문제가 생길 수 있다. 그냥 삽입의 경우, 객체가 바깥에서 생성이 된 뒤 컨테이너 내부에 복사되므로, 컨테이너 내부에 복사하는 과정에서 메모리 부족등의 예외가 발생해도 객체는 정상적으로 소멸된다. 그러나, 생성 삽입의 경우, 객체의 생성이 지연되므로, 컨테이너 내부에서 객체 생성 전에 예외가 발생하면, 객체 생성자의 인자들에 대한 참조를 잃게 된다. 즉, 만약 전달된 인자들 중에 new int 와 같은 것들이 있었다면, memory leak이 발생하는 것이다. 이러한 예외 안정성의 차이는 그냥 삽입은 컨테이너 외부에서 객체가 완전히 생성된 채로 들어오고, 생성 삽입은 컨테이너 내부에서 객체가 생성된다는 근본적인 차이에서 비롯되는 것이다. 따라서, 생성 삽입을 사용할 때는 예외 안정성에 대한 측면도 점검해야 할 필요가 있다. (물론, 사실 new int 와 같은 것들을 직접적으로 생성자의 인자로서 직접적으로 전달하려고 한 것 자체가 잘못이긴 하다.)  \
이렇듯, '생성 삽입' 이 '삽입' 보다 항상 우월한 것은 아니다. 그러나, 위에서 언급한 몇 가지 경우만 제외하면 '생성 삽입'이 '삽입'보다 나쁠 이유는 없다. 따라서 일단 기본적으로 '생성 삽입'을 사용하는 것을 원칙으로 하되, 내부적으로 중복 체크를 수행하는 컨테이너들에 대해서는 일반 '삽입'을 사용하는 것이 옳다. 그리고 explicit 생성자 관련 부분과 예외 안정성 부분의 문제는 근본적으로 생성 삽입보다 다른 곳에 근본적인 문제점이 존재하는 것이라고 생각한다. 따라서 생성 삽입을 사용하면서 다른 방법으로 해결 가능하기 때문에 이런 것들은 그냥 조심해서 사용하면 된다고 생각한다. 
  
**[기억해 둘 사항들]**  
- 이론적으로, 생성 삽입 함수들은 종종 해당 삽입 버전보다 더 효율적이어야 하며, 덜 효율적인 경우는 절대로 없어야 한다.  
- 실질적으로, 만일 (1) 추가하는 값이 컨테이너로 배정되는 것이 아니라 생성되고, (2) 인수 형식(들)이 컨테이너가 담는 형식과 다르고, (3) 그 값이 중복된 값이어도 컨테이너가 거부하지 않는다면, 생성 삽입 함수가 삽입 함수보다 빠를 가능성이 아주 크다.  
- 생성 삽입 함수는 삽입 함수라면 거부당했을 형식 변환들을 수행할 수 있다.
