---
layout: post
title: "C++ Boost 라이브러리를 Windows XP 에서 동작하도록 빌드하기"
date: 2019-01-09 22:00
author: taeguk
tags: [C++, Boost, Windows, XP]
---

안녕하세요~ 반갑습니다! <br/>
C++ 개발자 여러분~ boost 많이 쓰시죠?? 저도 boost 참 좋아합니다 ㅎㅎ <br/>
이렇게 좋은 boost 를 Windows XP/2003 환경에서 사용할 때는 그냥 일반적인 방법으로는 사용이 불가능한데요. 그래서 오늘은 Windows XP 환경에서 동작하도록 boost 를 빌드하는 방법에 대해서 알아보도록 하겠습니다!

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/02/c-boost-windows-xp_69.html)**

**참고로 제가 설명할 방법은 100% 완벽하게 검증된 방법은 아니며, 제가 조사/연구를 통해 알아낸 방법입니다. 따라서 문제가 있을 수도 있으므로 참고자료정도로만 사용해주시면 감사하겠습니다. 그리고 이 포스팅에서는 Visual C++ 2015 Update 3 를 사용한다고 가정하고 설명하겠습니다. 또한, 이 포스팅에서 말하는 Windows XP 는 정확히 말하면 XP Service Pack 3 를 뜻합니다.**

# 1단계) 사용하려는 라이브러리가 XP 를 지원하는 지 확인한다.

일단 첫 번째 단계는 사용하려는 라이브러리가 Windows XP 를 지원하는 지 확인하는 과정입니다. 다행히도 Boost 의 대부분 라이브러리는 XP 환경에서 잘 돌아갈텐데, 경우에 따라서 XP에서는 기능이 제한되거나 오동작을 일으킬 수 도 있으니, 라이브러리가 XP 를 지원하는 지 여부를 검토할 필요성이 있습니다.  <br/>
근데 아쉽게도 라이브러리들이 XP 를 지원하는 지 여부가 명백하게 정리된 글같은 건 없어 보입니다 ㅜㅜ 그리고, 각 라이브러리의 레퍼런스를 봐도 잘 안나오는 경우가 많죠. 결국, 부스트 커뮤니티 혹은 스택오버플로우의 글들을 뒤지거나 직접 질문을 해서 도움을 얻어야 합니다. 그리고 직접 라이브러리 내부 소스코드를 까서 XP 지원 여부를 파악해야 할 수도 있습니다. 그리고, 무엇보다 실제로 라이브러리를 사용하는 프로그램을 XP에서 동작시켜서 잘 동작하는지 직접 확인하는 과정도 꼭 필요합니다. <br/>
[이 글](https://groups.google.com/forum/#!topic/boost-developers-archive/BH-unGadG7E)은 2015년도에 부스트 커뮤니티에 올라왔던 Windows XP 지원 관련 설문인데 한번 읽어보셔도 좋을 것 같습니다.

## XP 호환 여부를 파악할 때 신경쓸 점
그러면 이제 부터 XP 지원 여부를 파악할 때 신경쓸 점들에 대해 알아보겠습니다.

1. XP 에서 지원하는 windows API 만을 사용해야만 합니다.
2. 암시적 TLS 사용 금지
    * XP 에서는 암시적 TLS에 관련된 known bug 가 있습니다. 따라서, 암시적 TLS 에 관련된 기능을 사용하면 안되고, `__declspec(thread)`, C++11 `thread_local` 또한 사용하면 안됩니다.
    * 좀 더 자세한 내용을 알고 싶으신 분은 [이 글](http://www.nynaeve.net/?p=187)을 참고하시면 좋을 것 같습니다.
3. C++11 Magic Statics 에 의존하면 안됩니다.
    * VC++2015 의 Magic Statics 구현은 내부적으로 암시적 TLS 를 사용합니다. 따라서, Magic Statics 에 의존하는 코드가 존재해서는 안됩니다.
    * 좀 더 자세한 내용을 알고 싶으신 분은 [이 글](http://www.jiniya.net/ng/2016/11/magic-statics/)을 읽어보시는 걸 추천드립니다.
4. `std::shared_mutex` 를 사용하면 안됩니다.
    * VC++2015 표준라이브러리의 `std::shared_mutex` 구현은 `SRWLOCK` 에 관련된 WinAPI 를 사용합니다. 근데, `SRWLOCK` 관련 WinAPI 는 Vista 이상에서 추가되었습니다. 따라서, XP 에서는 `std::shared_mutex` 가 동작할 수 가 없습니다. 관련된 내용은 [이 글](https://blogs.msdn.microsoft.com/vcblog/2016/01/22/vs-2015-update-2s-stl-is-c17-so-far-feature-complete/)을 참고하면 좋을 것 같습니다.
    * 뒤쪽에서 다룰 내용인데, 라이브러리를 빌드할 때, `_USING_V110_SDK71_` 매크로를 정의하게 됩니다.  이 매크로를 정의하면 C++ 표준라이브러리에서 `std::shared_mutex` 가 애초에 정의되지 않습니다. <br/>![](/assets/images/how-to-build-boost-for-windows-xp/shared-mutex.png)

# 2단계) Boost 가 XP 를 지원하도록 빌드하자.

일단, 이 포스팅에서는 빌드를 할 때 VC++2015 Update 3 를 사용할 것이므로, 이 것을 기반으로 설명을 드리도록 하겠습니다. 또한 기본적인 Boost 빌드 방법을 알고있다는 가정하에 설명을 진행하겠습니다. <br/>
VC++2015 Update 3 에서는 플랫폼 도구 집합으로 `v140` (default), `v140_xp` 가 있습니다. XP 를 지원하는 소프트웨어를 개발할 때는 `v140_xp` 를 사용해야만 합니다. <br/>
![](/assets/images/how-to-build-boost-for-windows-xp/v140_xp.png)
Boost 를 빌드할 때는 `b2.exe` 를 사용하여 빌드를 수행하게 되는데 이때 toolset 으로 비주얼스튜디오의 버전을 넣어주게 됩니다. 예를 들어, VC++2015 의 경우 `b2.exe --toolset=msvc-14.0 ...` 와 같은 식이 되는데, 문제는 이것은 비주얼 스튜디오에서의 플랫폼 도구 집합으로 치면 `v140` 을 의미하고, `v140_xp` 를 의미하도록 명시하는 방법이 `b2.exe` 에는 없다는 것입니다. <br/>
따라서, `v140` 과 `v140_xp` 의 차이점을 일일히 알고서 `b2.exe` 의 다른 파라미터를 통해서 그 차이점을 적용시켜야만 합니다. `v140` 과 `v140_xp` 의 차이점에 대해서는 [제가 과거에 작성했던 글](https://taeguk2.blogspot.com/2018/02/vc2015-vc140xp.html)을 참고하시길 바랍니다.

또한, Boost 가 XP 를 지원하도록 빌드하기 위해서는 여러가지 매크로들과 C++ Flag 를 설정해야만 합니다. 이제부터 그 매크로와 C++ Flag 를 하나씩 알아보겠습니다.

## `_WIN32_WINNT=0X0501`, `_NTDDI_VERSION=0X05010300` 매크로 정의
Windows 헤더 파일 (`windows.h` 등) 은 내부적으로 타겟 윈도우 버전을 의미하는 `_WIN32_WINNT`, `_NTDDI_VERSION` 등 매크로들의 값을 바탕으로 조건부 선언을 하고있습니다.  예를 들면, Vista 에서 추가된 `InitializeSRWLock` WinAPI 의 경우 `_WIN32_WINNT >= 0x0600` 일 경우에만 선언되도록 되어있습니다. <br/> ![](/assets/images/how-to-build-boost-for-windows-xp/srwlock.png) <br/>
타겟 윈도우 버전을 의미하는 매크로들이 어떤 게 있고 어떤 값들이 가능한 지는 [이 링크](https://docs.microsoft.com/en-us/windows/desktop/winprog/using-the-windows-headers#macros-for-conditional-declarations)를 보면 잘 나와있습니다. <br/>
우리는 XP SP3 를 타겟으로 해야하므로, `_WIN32_WINNT` 는 0x0501, `_NTDDI_VERSION` 는 0x05010300 으로 정의해야합니다.

## `BOOST_USE_WINAPI_VERSION=0X0501`, `BOOST_USE_NTDDI_VERSION=0X05010300` 매크로 정의
이 두 매크로가 어떤 역할을 하는 지에 대해서는 [이 링크](https://www.boost.org/doc/libs/1_69_0/libs/winapi/doc/html/winapi/config.html)를 참고하시길 바랍니다. 이 링크에 설명된 내용에 따르면, `BOOST_USE_WINAPI_VERSION`, `BOOST_USE_NTDDI_VERSION` 가 정의되어 있지 않으면 기본적으로 `_WIN32_WINNT`, `_NTDDI_VERSION` 의 값을 따라간다고 되어있긴합니다. 하지만 저는 그냥 명시적으로 정의를 해주었습니다.

## `BOOST_NO_CXX11_THREAD_LOCAL` 매크로 정의
포스팅의 1단계에서 말했듯이 XP 에서는 C++11 thread_local 키워드를 사용하면 안되므로, thread_local 키워드를 쓰는걸 막기위해 이 매크로를 정의하였습니다.

## `USING_V110_SDK71` 매크로 정의
플랫폼 도구 집합 `v140` 과 `v140_xp` 의 차이점들중에 하나가 이 매크로의 유무입니다. `v140_xp` 에서 이 매크로가 정의되므로 똑같이 정의해주도록 합니다.

## `/Zc:threadSafeInit-` 컴파일 플래그 설정
포스팅의 1단계에서 말했듯이 VC++2015 의 Magic Statics 구현은 XP 환경에서 문제가 있습니다. 따라서 Magic Statics 기능을 disable 해야만 하는데, `/Zc:threadSafeInit-` 플래그가 그러한 역할을 수행합니다.

그러면, 이제부터 XP 를 지원하도록 Boost.FileSystem 를 빌드해보도록 하겠습니다. (부스트 1.69.0 기준으로 테스트되었습니다.)

1. [부스트 소스코드](https://www.boost.org/users/download/)를 다운받아서 로컬에 압축을 풉니다.
2. `bootstrap.bat` 을 실행시킵니다. 이러면 `b2.exe` 가 생성됩니다.
3. 아래 명령어를 통해 빌드를 수행하면 됩니다.

## 32bit / 64bit 빌드 시, 공통으로 먼저 수행해야하는 명령어
```batch
set ORG_PATH=%PATH%

set INCLUDE=^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\include;^
C:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt;^
C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\include

set BUILD_CMD_COMMON= ^
b2 -a -j4 --stagedir=./stage/boost-xp ^
--with-filesystem ^
--reconfigure ^
toolset=msvc-14.0 variant=debug,release threading=multi runtime-link=static,shared link=static,shared ^
define="_WIN32_WINNT=0x0501" ^
define="NTDDI_VERSION=0x05010300" ^
define="BOOST_USE_WINAPI_VERSION=0x0501" ^
define="BOOST_USE_NTDDI_VERSION=0x05010300" ^
define="BOOST_NO_CXX11_THREAD_LOCAL" ^
define="USING_V110_SDK71" ^
cxxflags="/Zc:threadSafeInit- "
```

## 32bit 빌드 시 수행해야 하는 명령어
```batch
set PATH=^
C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\bin;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin;^
%ORG_PATH%

set LIBPATH=^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\lib;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib

set LIB=^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\lib;^
C:\Program Files (x86)\Windows Kits\10\lib\10.0.10240.0\ucrt\x86;^
C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\lib

%BUILD_CMD_COMMON% ^
address-model=32 ^
linkflags="/SUBSYSTEM:WINDOWS,5.01"
```

## 64bit 빌드 시 수행해야 하는 명령어
```batch
set PATH=^
C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\bin;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin;^
%ORG_PATH%

set LIBPATH=^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\lib\amd64;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64

set LIB=^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64;^
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\lib\amd64;^
C:\Program Files (x86)\Windows Kits\10\lib\10.0.10240.0\ucrt\x64;^
C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\lib\x64

%BUILD_CMD_COMMON% ^
address-model=64 ^
linkflags="/SUBSYSTEM:WINDOWS,5.02"
```

# 마무으리

Boost 를 XP 에서 사용하려면 상당히 까다롭죠?? ㅠㅠ 저도 작년 초에 XP 환경에서 boost 를 사용해야하는 상황에 처해서 많이 삽질을 했던 기억이 납니다. 이 포스팅은 그러한 저의 삽질/연구를 바탕으로 한 포스팅인데요. 완벽하게 검증된 방법은 아닙니다. (여러분이 보기에도 뭔가 꼼수같은 느낌이 많이 들죠??) <br/>
따라서 이 방법대로만 하면 XP 환경에서 아무런 문제없이 boost 를 사용할 수 있다! 라고 제가 말씀드리기는 어려울 것 같고, 참고정도는 가능하지 않을까 싶습니다. <br/>
잘못된 점 혹은 개선할 점 혹은 더 좋은 방법이 설명되어있는 자료링크 같은 것은 언제나 환영입니다! <br/>
지금까지 읽어주셔서 감사합니다~
