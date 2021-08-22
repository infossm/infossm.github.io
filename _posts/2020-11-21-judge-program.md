---
layout: post
title: 알고리즘 문제 judge program 만들기
date: 2020-11-21 22:00:00
author: jeonggyun
tags:
---

안녕하세요?

저는 이번에 알고리즘 문제의 judge program을 만들어 보았습니다. 소스 코드와 input data, 그리고 output data가 있을 때 각각의 코드마다 어떤 테스트 케이스에서 어떤 결과가 났는지를 출력해주는 프로그램을 목표로 제작하였고, 이 글에서는 제가 만든 프로그램을 어떻게 구현했는지에 대한 간단한 소개를 진행해보도록 하겠습니다.

많은 online judge 사이트, 대표적으로 백준 온라인 저지나 codeforces, sw expert academy 등에서는 많은 문제를 제공하고, 채점 결과와 실행 시간, 메모리 사용량 등도 알 수 있지만 직접 만든 문제나 테스트케이스 등에 대해 확인해보는 데에는 한계가 있습니다.

codeforces polygon, domjudge 등의 툴을 사용하는 것도 가능하지만, 이것보다 조금 더 가볍고 간편하게 사용할 수 있는 프로그램을 목표로 하였습니다.

사실 프로그램 제작 배경으로, 저는 이번 학기에 컴퓨터 알고리즘 수업의 조교를 맡고 있는데, 과제로 학생들에게 알고리즘 문제를 출제하였고 이를 어떻게 해야 쉽게 잘 채점할 수 있을까 고민하며 프로그램을 만들어보았습니다. 저와 같은 상황에 처한 사람들, 혹은 테스트 케이스를 다운로드 받아서 자신의 코드를 돌려보려는 사람들이 유용하게 사용할 수 있기를 바랍니다.


## 기본 기능 구현

채점 프로그램의 역할은 소스 코드, 그리고 테스트 데이터가 주어질 때 각 테스트 데이터에 대한 채점 결과를 출력할 수 있어야 합니다.

C++ 코드를 기준으로, 컴파일을 한 뒤, 각 테스트 케이스에 대해 input을 받아 출력을 내고, 이를 diff 사용해 output과 비교를 하거나, 특정 special judge를 사용하면 간단합니다.

사실 이 정도는 bash script로 충분하지만, 문제는 예외 상황이 발생할 때입니다.

케이스에 따라서 누군가가 영원히 끝나지 않는 프로그램을 작성할 수도 있고, 코드가 메모리를 얼마나 사용할지도 모릅니다. 때문에 일정 시간이 지나거나, 특정 메모리 이상을 사용한다면 프로그램을 종료해줄 필요성이 있습니다. 이를 리눅스에서 posix api를 사용해 구현해보겠습니다.

목표는 c++과 python3를 채점해주는 프로그램으로 하겠습니다.

프로그램의 수행 시간과 메모리 사용량을 읽으려면 프로세스의 정보를 읽어야 합니다.

실행한 프로세스의 pid를 받아오기 위해, fork()를 통해 프로세스를 만든 후 parent process에서 처리를 진행하도록 하겠습니다.

```c
int pid;
if ((pid = fork()) == 0) {
	// child
	if (strcmp(argv[3][0], "py") == 0) {
		execlp("python3", "python3", code_name, NULL);
	}
	else if (strcmp(argv[3][0], "cp") == 0) {
		execlp(argv[1], argv[1], NULL);
	}
}
else {
	// parent process
}
```

이제 parent process에서 pid를 이용해 여러 처리를 할 수 있게 되었습니다.

child process 부분을 조금만 더 살펴보도록 하겠습니다.

redirect 기능을 사용할 경우 간편한데, 현재 bash를 사용하는 것이 아니기 때문에 redirect 기능은 단순히 "<", ">" 등을 인자로 넣어주는 것으로는 불가능합니다.

dup2 함수를 이용해 redirect를 구현해야 하며, child process에서 exec을 하기 전에 아래와 같은 코드를 추가하면 됩니다.

```c
int in = open(input_file, O_RDONLY);
int out = open(output_file, O_CREAT | O_WRONLY | O_TRUNC, 0666);
int err = open("err", O_CREAT | O_WRONLY | O_TRUNC, 0666);
dup2(in, STDIN_FILENO);
dup2(out, STDOUT_FILENO);
dup2(err, STDERR_FILENO);
close(in);
close(out);
close(err);
```

이제 input_file 이름을 가진 파일을 stdin으로, stdout을 output_file 이름을 가진 파일로, stderr을 "err" 파일로 기록할 수 있게 되었습니다.


## 예외 상황 처리

리눅스 커널을 기준으로, 프로세스의 수행 시간은 /proc/[pid]/stat에 기록되어 있습니다.

[man 페이지](https://man7.org/linux/man-pages/man5/proc.5.html)에서 살펴보면, 시스템의 수행 시간을 나타내는 인자는 14번째 인자인 utime임을 알 수 있습니다.

utime의 단위는 \_SC\_CLK\_TCK라고 합니다. 보통 100의 값을 가지는데, 100Hz를 의미합니다. 아쉽게도 시간의 최소 정확도는 10ms가 되겠네요.

다음으로 프로세스의 메모리 사용량은, /proc/[pid]/status에 기록되어 있습니다.

마찬가지로 man page에서 살펴보면, VmPeak 항목을 살펴볼 수 있습니다.

따라서, 프로세스의 pid를 알면 아래와 같은 코드를 통해 프로세스의 수행 시간과 최대 메모리 사용량을 알 수 있습니다.

```c
char buf[4096];

char time_info[30];
char memory_info[30];
sprintf(time_info, "/proc/%d/stat", pid);
sprintf(memory_info, "/proc/%d/status", pid);

unsigned int time_used = 0, memory_used = 0;

FILE* f = fopen(time_info, "r");
if (f) {
	unsigned int utime;

	int read = fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %u", &utime);
	fclose(f);
	if (read == 1) {
		time_used = utime;
	}
}

f = fopen(memory_info, "r");
if (f) {
	int read = fread(buf, 1, 4095, f);
	buf[read] = '\0';
	fclose(f);

	char* ptr = strstr(buf, "VmPeak:");
	int mem;
	if (ptr) {
		read = sscanf(ptr, "%*s %d", &mem);
		if (read == 1) {
			memory_used = mem;
		}
	}
}

```

다만 여기에는 치명적인 단점이 있습니다. 프로세스가 종료되는 순간 /proc/[pid]/stat, /proc/[pid]/status 파일이 사라진다는 점입니다.

우리는 프로그램이 딱 종료되는 그 순간의 vmpeak, utime 정보를 알아야 하는데 파일이 사라져버리니 이 정보를 읽어올 수 없게 되는 것입니다. 어쩔 수 없이, 약간의 오차를 감수하고 while문을 통해 1ms마다 읽는 정도로 타협을 하겠습니다. 사실 이 1ms 사이에 메모리가 엄청 많이 할당된다거나 하는 일이 발생할 때 감지하지 못한다는 점은 큰 단점입니다.

이제 예외 상황을 처리해봅시다.

제한 시간 또는 제한 메모리를 초과할 경우, 프로세스를 종료해주어야 합니다. 이는 프로세스에 SIGKILL을 보내서 구현 가능하겠네요. 아래와 같은 코드를 추가합니다.

```c
if (time_used > MAX_TIME) {
	kill(pid, SIGKILL);
	ret = waitpid(pid, &status, 0);
	break;
}

if (memory_used > MAX_MEMORY) {
	kill(pid, SIGKILL);
	ret = waitpid(pid, &status, 0);
	break;
}
```

마지막으로 런타임 에러를 처리해봅시다.

```c
ret = waitpid(pid, &status, WNOHANG);
```
를 while 문에 추가하면 프로세스가 실행중일 때는 그냥 넘어가고, 프로세스가 종료될 경우 status 변수에 정보, ret에는 pid를 담아주는 역할을 수행합니다.

status 변수에는 return value와 프로세스 종료의 시그널 정보가 담기는데, 하위 8비트에는 시그널에 대한 정보, 그 다음 8비트에는 return value가 담기게 됩니다.

보통 런타임 에러가 발생할 때는 WIFSIGNALED(status)가 true가 되기 때문에 이를 쉽게 판별 가능하지만, python에서 런타임에러가 발생할 경우 signal 표시가 되지 않습니다. 다시 말해, WIFSIGNALED(status)가 false이고 대신 return value가 1입니다.

따라서 정확하게 판단하려면 status가 0인지 아닌지를 판별하면 됩니다.

대신 이 경우에는 return 1, exit(1) 등으로 정상적으로(?) 종료가 되어도 런타임에러로 처리가 된다는 단점이 있습니다. 보통 return 0을 통해 종료되는 것이 정상적인 프로그램이므로 이 부분에 대해 크게 신경쓰지 않는다면 상관없습니다.

## 그 외 전체적인 인터페이스

전체적인 인터페이스는 python을 이용해 구현하였습니다.

```python
import os
os.system("g++ -o judge judge.c")
source_files = os.listdir('source')
os.mkdir('result/{}'.format(name))
```
등의 명령어를 사용해 여러 쉘 명령어들을 간단히 호출하는 것이 가능하기 때문입니다.

```python
input_files = os.listdir('input')
output_files = os.listdir('output')

for i in input_files:
    for j in output_files:
        if i[:i.find('.')] == j[:j.find('.')]:
            pair.append((i, j))
```
등의 명령어를 사용하면 input과 output file을 하나의 pair로 유지 가능합니다.

때에 따라서는 judge.c, judge.py 파일만 있으면, source 폴더에 채점하려는 코드들을, input 폴더에 입력 데이터를, output 폴더에 출력 데이터를 넣은 후

> python3 judge.py

명령어를 실행하면 실행 결과 파일이 출력됩니다.

경우에 따라서는 spj.cpp 파일이 필요하고, judge.c 파일의 MAX_TIME, MAX_MEMORY 변수를 조정해주면 됩니다.

## 마무리

이렇게 posix API를 이용해 간단한 채점 프로그램을 만들어보았습니다. 전체적인 소스 코드는 [여기](https://github.com/jeonggyunkim/judge)에서 확인하실 수 있습니다.

사실 Domjudge와 같은 제대로 만들어진 오픈 소스 저지 사이트에서는 cgroups를 이용해 CPU 시간, 메모리 등의 자원을 정확히 할당하고 모니터링할 수 있다고 합니다. while문을 돌려서 오차가 발생하는 제 프로그램보다는 훨씬 더 정확합니다.
