---
layout: post
title: "Google Test와 gcov 소개"
date: 2019-07-21 21:00
author: zych1751
tags: [C++, test-coverage]
---



이 글에서는 Google Test를 이용하여 간단한 c++ 유닛 테스트 코드를 작성해보고 이후 gcov를 이용하여 코드 커버리지를 측정하는 방법에 대해서 다뤄보겠습니다.  

  

## 유닛 테스트란?

유닛 테스트는 특정 모듈(함수 or 특정 코드)이 내가 원하는 바 대로 정확히 작동하는지 검증하는 절차입니다.  

즉, 모든 함수와 메소드에 대한 테스트 케이스를 작성하는 절차를 말합니다.  

우리는 이를 자동화하고 반복할 수 있게 하여 코드가 수정되어 문제가 발생하였을 경우에 이를 빠르게 찾고 수정할 수 있도록 해주고, 유닛 테스트를 통과했다면 해당 모듈이 잘 작동하고 있음을 확신하고 코드를 작성할 수 있게 해줍니다.  

  

## Google Test

우리는 유닛 테스트를 Google Test 프레임워크를 사용하여 구현해 볼 것입니다.  

Google Test는 이 **[링크](<https://github.com/google/googletest>)**에서 다운받을 수 있습니다.  

그리고 Google Test를 설치하려면 CMake가 필요하므로 각자 운영체제에 맞춰 설치해줍니다.  

CMake의 설치가 완료 되었다면 아래와 같이 입력해주면 빌드가 완료됩니다. (git이 없다면 git clone 대신에 링크로 직접 들어가 다운로드 받은 후 googletest 폴더안에서 cmake 명령어부터 시작)  

```
git clone https://github.com/google/googletest.git
cd googletest
cmake CMakeLists.txt
make
```

위와 같이 실행하고 나면 lib 폴더안에 ligbmock.a, libgmock_main.a, libgtest.a, libgtest_main.a 파일이 생겼음을 확인할 수 있습니다.  

해당 파일을 컴파일러 라이브러리 기본 경로로 옮겨주면 설정은 끝입니다.  

```
sudo cp ./lib/*.a /usr/lib
```

이제 c++파일에서 gtest를 include를 할 수 있게 되었습니다.  



## 테스트 코드 작성

구글 테스트에서는 단정문을 만들어 해당 조건이 참인지 거짓인지 확인을 합니다.  

단정문의 경우 크게 ASSERT\_\* 형식과 EXPECT\_\*형식이 있는데 ASSERT의 경우 해당 테스트케이스에서 조건을 만족하지 못하면 뒤의 테스트를 실행하지 않고 현재 함수를 바로 중단합니다. 그와 반대로 EXPECT의 경우는 조건이 맞지 않아도 뒤의 테스트를 계속 실행합니다.  

단정문은 아래와 같은 종류들이 있습니다. (ASSERT와 EXPECT가 동일하므로 ASSERT만 표기)  

| 단정문                    | 내용                           |
| ------------------------- | ------------------------------ |
| ASSERT_TRUE(val);         | val == true                    |
| ASSERT_FALSE(val);        | val == false                   |
| ASSERT_EQ(val1, val2);    | val1 == val2                   |
| ASSERT_NE(val1, val2);    | val1 != val2                   |
| ASSERT_LT(val1, val2);    | val1 < val2                    |
| ASSERT_LE(val1, val2);    | val1 <= val2                   |
| ASSERT_GT(val1, val2);    | val1 > val2                    |
| ASSERT_GE(val1, val2);    | val1 >= val2                   |
| ASSERT_STREQ(val1, val2); | val1과 val2의 문자열이 같다.   |
| ASSERT_STRNE(val1, val2); | val1과 val2의 문자열이 다르다. |

그리고 각 테스트 케이스들은 아래와 같이 만들어집니다.  

```
TEST('테스트 그룹 이름', '그룹 내 하위 테스트 이름') {
    // 단정문...
}
```



이를 이용하여 간단하게 테스트 할 함수와 테스트 코드를 작성해 봅시다.  

```cpp
// main.cpp
int multiply(int a, int b) {
    return a * b;
}

int myMax(int a, int b, int c) {
    if(a >= b && a >= c)
        return a;
    else if(b >= a && b >= c)
        return b;
    return c;
}
```

```cpp
// test.cpp
#include <gtest/gtest.h>
#include "main.cpp"

TEST(Multiply, TwoMultiplyThree) {
	EXPECT_EQ(multiply(2, 3), 6);
}

TEST(Multiply, MultiplyZero) {
	EXPECT_EQ(multiply(2, 0), 0);
	EXPECT_EQ(multiply(-1, 0), 0);
	EXPECT_EQ(multiply(0, 0), 0);
}

TEST(Max, PositiveMax) {
	ASSERT_EQ(myMax(1, 3, 2), 3);
	ASSERT_EQ(myMax(2, 3, 2), 3);
	ASSERT_EQ(myMax(5, 5, 5), 5);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

```

이와 같이 작성하고 아래와 같이 실행하면 결과를 확인할 수 있습니다.

```
g++ -o main test.cpp -lgtest
./main
```

![](/assets/images/test-coverage/gtest_result.png)

Multiply와 Max 2개의 테스트 케이스로 나눠지고 각각의 테스트 3개가 모두 통과했음을 확인할 수 있습니다.  



## gcov

gcov는 프로그램에 대한 코드 커버리지 테스트를 수행할 수 있는 도구입니다.  

코드 커버리지란 테스트가 얼마나 충분한지에 대한 지표중 하나이며 우리는 gcov를 이용하여 각 라인별로 얼마나 실행되었는지를 측정할 것입니다.  

  

우선 각 운영체제에 맞게 gcov를 다운 받아줍니다.  

커버리지를 측정할때는 컴파일을 할때 g++에 -fprofile-arcs, -ftest-coverage 두개의 플래그를 설정하여 컴파일을 해주어야 합니다. 그리고 최적화 옵션을 사용하지 않고 컴파일을 해주어야 합니다. 왜냐하면 최적화 때문에 실제로는 실행되는 부분이지만 최적화에 의해 실행하지 않고 넘어갈 수 있게되어 실제 커버리지 값이랑 다르게 나올 수 있기 때문입니다.  

위의 작성한 코드들을 그대로 이용하여 아래와 같이 실행해 봅시다.  

```
g++ -o main -fprofile-arcs -ftest-coverage -lgtest test.cpp
./main
gcov test.cpp
```

실행을 하면 각 파일별로 전체 라인당 몇 %의 코드가 실행되었는지의 결과를 확인할 수 있습니다.  

![](/assets/images/test-coverage/gcov_test.png)

위 스크린샷에는 나오지 않았지만 main.cpp에서 커버리지가 87.5%가 나오고 있는데 어디가 실행이 안되었는지 궁금할 수 있습니다. 이는 gcov 실행결과로 생긴 .gcov파일을 확인하면 각 라인이 몇번 실행되었는지, 어떤 라인이 실행되지 않았는지를 알 수 있습니다.  

![](/assets/images/test-coverage/main_gcov.png)

이를 통해 10라인이 실행되지 않았음을 확인되었고, myMax에서 3번째 인자가 가장 큰 경우를 테스트하지 않았단 것을 확인하여 좀 더 테스트를 보강할 수 있게 됩니다.  

  

## 마무리

google test와 gcov를 통해 간단한 유닛 테스트 실행 및 커버리지 측정을 해보았습니다.  

저는 현재 gcc9와 lcov의 충돌이슈([링크](<https://github.com/linux-test-project/lcov/issues/58>)) 때문에 포스팅하지 못하였지만, lcov와 genhtml을 추가로 이용하면 아래와 같이 gui로도 전체적인 커버리지 결과를 확인할 수 있습니다.  

궁금하신 분들은 각자 한번 해보시기 바랍니다.  

![](/assets/images/test-coverage/lcov_report.jpg)
