---
layout: post
title: "LD_PRELOAD 를 이용한 후킹"
date: 2019-04-10 22:54
author: taeguk
tags: [Linux, hooking, LD_PRELOAD]
---

안녕하세요. 오늘은 리눅스 환경에서 LD_PRELOAD 환경변수를 이용해서 후킹을 하는 방법에 대해 간략히 포스팅해볼까 합니다~

## 후킹이란?

> 후킹(영어: hooking)은 소프트웨어 공학 용어로, 운영 체제나 응용 소프트웨어 등의 각종 컴퓨터 프로그램에서 소프트웨어 구성 요소 간에 발생하는 함수 호출, 메시지, 이벤트 등을 중간에서 바꾸거나 가로채는 명령, 방법, 기술이나 행위를 말한다. 이때 이러한 간섭된 함수 호출, 이벤트 또는 메시지를 처리하는 코드를 후크(영어: hook)라고 한다. <br/>
크래킹(불법적인 해킹)을 할 때 크래킹 대상 컴퓨터의 메모리 정보, 키보드 입력 정보 등을 빼돌리기 위해서 사용되기도 한다. <br/>
**예를 들어 특정한 API를 후킹하게 되면 해당 API의 리턴값을 조작하는 등의 동작을 수행할 수 있다.** <br/>
(출처 : https://ko.wikipedia.org/wiki/%ED%9B%84%ED%82%B9)

후킹에 대해서는 위키백과에 설명이 잘 나와있어 해당 설명을 인용해봤습니다.

## printf 함수를 후킹해보자!
![](https://lh3.googleusercontent.com/OMP4Bd4h_4xjOO8nZda6Gs0TqyRJvha96R2SRvlAAztDmub29ZxaLuvF17f8lq2Vxdafv79pN77U) <br/>
위 사진에서 보이듯이 printf 함수를 후킹해서 이상한 문자열이 출력되도록 해보겠습니다.

![](https://lh3.googleusercontent.com/X8gv35qpugF9RZ2Bw3_WW24Td8ZPsWJeQDq7zREmKZxpJPL15n7rlHz1npslH75mLMaj9R5wLjxG) <br/>
printf 가 호출되는 원리를 그려보자면 위와 같습니다. 프로세스가 생성되고 실행되면서 libc.so 공유 라이브러리가 자동으로 로드되게 됩니다. 그리고 우리가 만든 프로그램에서 printf 를 호출하면 libc.so 안에 존재하는 printf 가 실제로 호출되게 됩니다.

![](https://lh3.googleusercontent.com/hDOUICNSiaQJrdqrjZDEho-kTRxPKN8fVb66xKAAiAjWYqTAV9ans1zgVl-eScyv-CNSEd2u1EJQ) <br/>
printf 를 후킹하기 위한 전반적인 전략은 다음과 같습니다.
* ./a.out 이 실행될 때 우리가 만든 hook_1.so 공유라이브러리가 자동으로 로드되도록 합니다.
* ./a.out 내에서 printf 함수를 호출하게 되면, libc.so 가 아닌 hook_1.so 내의 가짜 printf 가 호출되도록 합니다.
* hook_1.so 내에서는 libc.so 의 진짜 printf 를 이용해서 이상한 문자열(HOOKED!) 을 출력합니다.

이를 실제로 수행하기 위한 여러가지 방법들이 있을 수 있지만, 이 포스팅에서는 LD_PRELOAD 를 이용해서 매우 쉽고 간단하게 목표를 이뤄보도록 하겠습니다.

### LD_PRELOAD 란?
* 유닉스/리눅스 계열에서 사용되는 환경 변수입니다.
* 프로세스가 실행될 때, 이 환경 변수에 지정된 공유 라이브러리가 먼저 로드됩니다.
* 따라서 다른 라이브러리의 함수와 LD_PRELOAD 에 지정된 라이브러리의 함수가 서로 이름이 똑같을 경우 후자가 실행되게 됩니다. (라이브러리 적재 순서 때문에)

### Step 1. 가짜 printf 를 가지고 있는 공유 라이브러리를 만든다.
```c
/***** hook_1.c *****/
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

// 진자 printf가 존재하는 메모리 주소를 저장하고 있다.
int (*printf_real)(const char *, ...) = NULL;

// 라이브러리가 로드될 때 실행된다.
void __attribute__((constructor)) init_hooking()
{
    // 진짜 printf가 존재하는 메모리 주소를 가져온다.
    printf_real = dlsym(RTLD_NEXT, "printf");
    fprintf (stderr, "real printf is at %p\n", printf_real);
}

// 우리가 만든 가짜 printf
int printf(const char *format, ...)
{
    // 위에서 얻은 진짜 printf 를 이용해서 HOOKED! 를 화면에 출력한다.
    return printf_real("HOOKED!");
}
```
```bash
$ gcc -o hook_1.so hook_1.c -shared -fPIC -ldl
```

### Step 2. 가짜 printf 를 가지고 있는 공유 라이브러리를 LD_PRELOAD 환경변수에 지정한 후에 ./a.out 를 실행한다.
```
$ export LD_PRELOAD=./hook_1.so
$ ./a.out
real printf is at 0x7efffa463f60
HOOKED!
HOOKED!
HOOKED!
HOOKED!
HOOKED!
```
printf 함수 후킹에 성공했습니다!

## 응용 : free checker (Memory Leak Finder) 만들기

C언어 프로그램을 짤 때는 malloc 을 통해 할당한 메모리를 반드시 free 해줘야합니다. 만약 free 해주지 않으면 memory leak 이 발생하게 되는데, 이러한 memory leak 를 탐지해주는 툴을 후킹을 이용해 간단하게 만들어보도록 하겠습니다. ~~(사실 그냥 Valgrind 를 쓰면 되는데..)~~

### 전략
![](https://lh3.googleusercontent.com/heBJ9CVJRwtWGOGCEyTmUKb4Gfs-1D9Z25WoaFxo7s65UMSiO-lZJir1BMY3RhqZ8f9FpM9rn2Tc) <br/>
기본적인 전략은 printf 를 후킹할 때와 똑같습니다. malloc, calloc, free 를 모두 후킹해서 malloc/calloc 을 호출했는데 free 하지 않은 메모리가 있는지를 검사합니다.

### fc.c
가짜 malloc, calloc, free 가 정의되어 있는 공유 라이브러리 fc.so 의 소스코드입니다.
```c
// -ldl
#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdio.h>
#include <stdlib.h>

long long mallocCallNum = 0;
long long callocCallNum = 0;
long long freeCallNum = 0;

void* (*malloc_real)(size_t) = NULL;
void* (*calloc_real)(size_t,size_t) = NULL;
void (*free_real)(void*) = NULL;

void __attribute__((constructor)) init_hooking()
{
	fprintf(stderr, "init_hooking() call start!!!\n");
	malloc_real = dlsym(RTLD_NEXT, "malloc");
	fprintf(stderr, "malloc_real at %p \n", malloc_real);
	calloc_real = dlsym(RTLD_NEXT, "calloc");
	fprintf(stderr, "calloc_real at %p \n", calloc_real);
	free_real = dlsym(RTLD_NEXT, "free");
	fprintf(stderr, "free_real at %p \n", free_real);
	fprintf(stderr, "init_hooking() call finish!!!\n");
}

void __attribute__((destructor)) finish_hooking()
{
	fprintf(stderr, "-----free check result----- \n");
	fprintf(stderr, "malloc was called #%lld \n", mallocCallNum);
	fprintf(stderr, "calloc was called #%lld \n", callocCallNum);
	fprintf(stderr, "free was called #%lld \n", freeCallNum);
	fprintf(stderr, "malloc + calloc #%lld \n", mallocCallNum + callocCallNum);
	fprintf(stderr, "not free memory #%lld \n", mallocCallNum + callocCallNum - freeCallNum);
	fprintf(stderr, "--------------------------- \n");
}

void* malloc(size_t size)
{
	void* ret = malloc_real(size);
	fprintf(stderr, "[malloc Call #%lld (%u)] %p ", ++mallocCallNum, (unsigned int) size, ret);
	fprintf(stderr, "(free memory : %lld / %lld) \n", freeCallNum, mallocCallNum + callocCallNum);
	return ret;
}

void* calloc(size_t num, size_t size)
{
	void* ret = calloc_real(num,size);
	fprintf(stderr, "[calloc Call #%lld (%u, %u)] %p ", ++callocCallNum, (unsigned int) num, (unsigned int) size, ret);
	fprintf(stderr, "(free memory : %lld / %lld) \n", freeCallNum, mallocCallNum + callocCallNum);
	return ret;
}

void free(void *ptr)
{
	if(ptr == NULL)return;
	free_real(ptr);
	fprintf(stderr, "[free Call #%lld] %p ", ++freeCallNum, ptr);
	fprintf(stderr, "(free memory : %lld / %lld) \n", freeCallNum, mallocCallNum + callocCallNum);
}
```

### freeChecker.sh
fc.c 를 컴파일하고 그 결과로서 생성되는 fc.so 를 LD_PRELOAD 로 설정한다음에 타켓 프로그램을 실행시키는 과정을 간단하게 해주는 쉘 스크립트입니다.
```bash
#!/bin/bash

c_file=$(dirname $0)"/fc.c"
so_file=$(dirname $0)"/fc.so"
#if [ ! -f "$so_file" ];then
	echo "Making fc.so..."
	if [ ! -f "$c_file" ];then
		echo "$c_file 가 존재하지 않습니다.."
		exit
	fi
	gcc -fPIC -shared -o $so_file $c_file -ldl
	echo "Make Complete!"
	echo
#fi

echo "----------free check start----------"

if [ "$#" -eq "0" ]; then
	echo "[Usage] freeChecker.sh target_program target_program_parameters"
else
	cmd="LD_PRELOAD=$so_file $@"
	eval $cmd
fi
```

### 사용 예시
```bash
$ freeChecker.sh ./a.out
```
![](https://lh3.googleusercontent.com/yF08E1yCwZj5bNHGdOh5GmXxxHw-Mwj5kzQ4g5Ahu04ne4pOmp5S9QW4FB-8WVtArKlWtMYFQ3Zu)

## 결론
대학교 1학년때 LD_PRELOAD 를 이용한 free checker 를 만들어서 저를 비롯해 주변 친구들이 C언어 수업 과제할 때 잘 써먹었던 기억이 나네요 ㅎㅎ (리포지토리 : https://github.com/taeguk/free_checker) <br/>
아무튼 그 때 작성했던 소스코드를 가지고 간단하게 포스팅해봤습니다. 다음에 또 만나요~
