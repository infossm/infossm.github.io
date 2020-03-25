---
layout: post
title: "프로그램의 메모리 사용량 측정 방법"
date: 2020-03-21 18:00:00
author: jeonggyun
---

안녕하세요?

오늘은 프로그램의 메모리 사용량을 측정할 수 있는 방법에 대해 알아보았습니다.

아래 코드들은 최적화 방지를 위해 -O0 옵션으로 컴파일 후 실행되었습니다.

## Indroduction

보통 온라인 저지 시스템에서의 채점 결과 중에는, MLT(Memory Limit Exceeded)가 포함되어 있습니다. 문제에서 주어진 메모리 제한을 초과할 경우 이러한 메시지가 뜨게 됩니다.

사실 대부분의 문제에서는 메모리를 꽤 넉넉하게 주기 때문에, 메모리 제한 초과는 특수한 상황이 아니라면 그리 자주 발생하는 편은 아닙니다. 재귀함수가 계속해서 호출된다거나, 큐가 계속 커진다거나 하는 상황에서 그나마 자주 발생하는 편이고, 그마저도 MLE보다는 TLE를 먼저 만나게 되는 경우가 많습니다.

하지만, 프로그램이 정확히 얼만큼의 메모리를 쓰는지 미리 확인해볼 수 있다면 좋을 것입니다. 또, 간혹 어떠한 저지 시스템에서는 스택 사이즈 1MB 등의 넉넉하지 않은 용량을 두는 경우도 있는데, 이런 경우 조금 더 큰 도움이 될 수 있습니다.

프로그램의 실행시간은 time()이나 clock() API를 통해 쉽게 측정할 수 있지만, 메모리 사용량의 측정은 그리 쉽지 않습니다. 어떻게 하면 이를 확인할 수 있을까요?

## Physical Memory와 Virtual Memory

먼저 시작하기에 앞서, Physical Memory와 Virtual Memory 사이에 사용량의 차이가 있다는 점을 짚고 넘어가야 합니다.

Memory allocator는 최적화가 굉장히 잘 되어있어, 메모리를 할당하더라도 해당 크기의 메모리를 실제 메모리에 바로 할당하지 않고, 메모리에 실제 접근이 일어날 때 할당을 해줍니다.

때문에, 다음과 같은, 약 총 64GB의 메모리를 요구하는 코드도 4GB 크기의 램을 가진 제 노트북에서 무리없이 돌아갑니다.

```
#include <stdio.h>
#include <stdlib.h>
#define N (1 << 18)

int* arr[N];

int main() {
	for (int i = 0; i < N; ++i) arr[i] = malloc(N);
	for (int i = 0; i < N; ++i) free(arr[i]);
}
```

따라서 프로그래머가 작성한 코드가, 실제 얼마만큼의 메모리를 사용하는지를 확인하려면 Virtual Memory의 사용량을 확인해야 합니다.

## 메모리 사용량이 표시되는 곳

어떠한 프로세스의 메모리 사용량은, /proc/pid/status에서 확인이 가능합니다.

이 곳에는 pid번 프로세스의 많은 정보들이 표시되어 있는데, 이 중 VmPeak, VmSize, VmData, VmStk 항목이 중요합니다.

VmPeak는 프로세스가 가장 많은 메모리를 사용할 때 사용한 가상 메모리의 양입니다. VmSize는 현재 프로세스가 사용하고 있는 가상 메모리의 양입니다.

다음으로 VmData 항목에는 data 영역, bss 영역, heap 영역의 메모리 사용량 합이 표시되며, VmStk 항목에서는 stack 영역에서의 메모리 사용량 합이 표시됩니다. 단위는 모두 KB입니다.

위에 있는 64GB의 메모리를 사용하는 프로그램을 예시로 한 번 살펴보겠습니다.

```
VmPeak:	67380632 kB
VmSize:	67380632 kB
...
VmData:	67376304 kB
VmStk:	     132 kB
```
메모리가 할당된 직후의 상황입니다. free까지 진행이 완료된 직후의 상황은 다음과 같습니다.

```
VmPeak:	67380632 kB
VmSize:	    6552 kB
...
VmData:	    2224 kB
VmStk:	     132 kB
```

따라서, 프로세스의 종료 직전 VmPeak를 확인하면 프로그램이 얼마나 많은 메모리를 사용하였는지를 확인할 수 있습니다.

다음과 같은 코드를 이용할 경우 간편합니다.

```
int pid = getpid();
char target[30], buf[4096];

sprintf(target, "/proc/%d/status", pid);
FILE* f = fopen(target, "r");
fread(buf, 1, 4095, f);
buf[4095] = '\0';
fclose(f);

int mem;
char* ptr = strstr(buf, "VmPeak:");
sscanf(ptr, "%*s %d", &mem);
printf("Use %dMB\n", mem / (1 << 10));
```

## 프로그램의 메모리 사용량에 제한을 두는 방법

또다른 방법으로, 아예 스택이나 힙 사이즈에 제한을 두는 방법이 있습니다.

첫 번째 방법으로, 컴파일 시 옵션을 설정해주는 방법이 있습니다.

gcc의 경우 -Wl,--stack,1048576 또는 -Wl,--heap,1048576 옵션을 줌으로서 컴파일 시 스택이나 힙의 크기를 원하는 대로 조절할 수 있습니다.

하지만 이 방법은 특정 환경에서는 잘 적용되지 않습니다. 아마 Windows 환경에서 MinGW를 사용할 때만 잘 작동하는 것으로 보입니다.

두 번째 방법으로, rlimit api를 이용하는 방법이 있습니다.

rlimit는 특정 프로세스가 사용할 수 있는 자원을 한정해줍니다.

```
#include <sys/time.h>
#include <sys/resource.h>
```
를 include한 뒤 getrlimit, setrlimit 함수를 사용할 수 있습니다.

먼저 rlimit structure에는 다음과 같은 정보들이 있습니다.

```
struct rlimit {
	rlim_t rlim_cur;  /* Soft limit */
    rlim_t rlim_max;  /* Hard limit (ceiling for rlim_cur) */
};
```
Soft limit은 해당 프로세서에 한정된 자원의 한계이고, Hard limit은 Soft limit의 상한입니다. 저희가 살펴볼 것은 Soft limit입니다.

limit으로 설정할 수 있는 것들의 목록은 man 페이지에서 확인할 수 있습니다. 대표적으로, RLIMIT_CPU, RLIMIT_DATA, RLIMIT_STACK 등이 있습니다.

사용 예시는와 실행 결과는 아래와 같습니다.

```
int main() {
	const rlim_t StackSize = 1 * 1024 * 1024;
	const rlim_t DataSize = 10 * 1024 * 1024;

    struct rlimit rlim;

    // Stack
    getrlimit(RLIMIT_STACK, &rlim);
    printf("Stack Max: %lu\n", rlim.rlim_cur);
	rlim.rlim_cur = StackSize;
	setrlimit(RLIMIT_STACK, &rlim);
    getrlimit(RLIMIT_STACK, &rlim);
    printf("Stack Max: %lu\n", rlim.rlim_cur);

    // Data
    getrlimit(RLIMIT_DATA, &rlim);
    printf("Data MAX: %lu\n", rlim.rlim_cur);
	rlim.rlim_cur = DataSize;
	setrlimit(RLIMIT_DATA, &rlim);
    getrlimit(RLIMIT_DATA, &rlim);
    printf("Data Max: %lu\n", rlim.rlim_cur);
}

Stack Max: 8388608
Stack Max: 1048576
Data MAX: 18446744073709551615
Data Max: 10485760
```

초기에 Stack 영역의 크기는 8MB, Data 영역의 크기는 무제한으로 설정되어 있습니다. 이를 setrlimit 함수를 이용해 바꾸어준 뒤 다시 getrlimit 함수를 사용해 확인해보면, 성공적으로 바뀐 것을 확인할 수 있습니다.

Soft Limit을 초과할 경우 특정 시그널이 발생하는데, 자세한 사항은 역시 man 페이지를 통해 확인하실 수 있습니다.

RLIMIT_CPU를 초과할 경우 SIGXCPU 시그널이, RLIMIT_STACK를 초과할 경우 SEGSEGV 시그널이 발생하고 RLIMIT_DATA를 초과할 경우 더 이상 malloc이 이루어지지 않습니다.

```
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

int rec(int n) {
	int arr[1024];
	return rec(n - 1);
}

int main() {
	const rlim_t StackSize = 1 * 1024 * 1024;
	const rlim_t DataSize = 1 * 1024 * 1024;

    struct rlimit rlim;

	rlim.rlim_cur = StackSize;
	setrlimit(RLIMIT_STACK, &rlim);

	rlim.rlim_cur = DataSize;
	setrlimit(RLIMIT_DATA, &rlim);

	char* arr = malloc(2 << 20);
	if (arr == NULL) {
		printf("malloc fail\n");
	}

	rec(1000);
}

실행결과
malloc fail
1000
999
...
751
Segmentation fault (core dumped)
```

간단한 프로그램을 통해 잘 작동하는지 확인해보도록 하겠습니다.

Data 영역의 크기를 1MB로 설정하고, 2MB 크기의 메모리를 할당하려고 할 경우 malloc 오류가 발생합니다.

또, 함수 내부에서 4KB 만큼의 메모리를 사용하는 재귀함수의 경우, 1MB / 4KB = 250번 정도의 재귀가 반복되면 Segmentation fault가 발생하는 것을 확인할 수 있었습니다.

## Conclusion

메모리 사용량을 확인하는 방법과, linux api를 이용하여 이를 조절할 수 있다는 사실이 흥미로웠습니다.

실제 온라인 저지에서는 메모리가 초과될 경우 그 즉시 프로그램을 종료한다거나 하는 등의 여러 방법이 추가적으로 적용되어 있을 것이기 때문에, 정확히 어떤 방법을 통해 메모리를 측정하는지는 잘 모르겠습니다.

잘 모르던 내용을 혼자 찾아보며 공부했기 때문에, 혹시라도 틀린 내용이 있을 수 있습니다. 틀린 내용이 있다면 sslktong@dgist.ac.kr로 연락 부탁드리겠습니다.

아래는 글을 쓰며 참고한 자료들입니다.

[Hackerearth: Technical diving into memory used by a program in online judges](https://www.hackerearth.com/practice/notes/vivekprakash/technical-diving-into-memory-used-by-a-program-in-online-judges/)

[Stackoverflow: Change stack size for a C++ application in Linux during compilation with GNU compiler](https://stackoverflow.com/questions/2275550/change-stack-size-for-a-c-application-in-linux-during-compilation-with-gnu-com)