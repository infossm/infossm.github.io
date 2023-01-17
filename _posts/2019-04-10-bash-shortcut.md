---
layout: post
title:  "bash 단축키 뜯어보기"
date:   2019-04-10 22:00
author: evenharder
tags:   unix Linux bash
---

![Don't do this.](/assets/images/evenharder-post/bash-shortcut/peek-20190410-rm-rf.gif)

bash같은 쉘은 정말 강력한 기능을 지니고 있고, 이러한 터미널 및 쉘이 *nix 계열의 심장이라 해도 과언이 아닙니다. 유명한 `sudo`나 `rm -rf /`, `apt-get`, `git clone`, `pip install`, `gcc -O2 -Wall -o test test.c`, `echo Hello World!` 등등 수많은 커맨드들이 오늘도 전 세계에 컴퓨터에서 돌아가고 있습니다. GUI에서 다양한 클릭과 스크롤을 통해서 진행되는 일들이 CLI에서는 한 줄의 명령어로 된다는 점이 매력이라 할 수 있겠습니다.

쉘은 커맨드 측면에서도 다양한 편의성을 제공하지만, 간단한 단축키를 통해서도 수많은 '방향키-지우기-다시 쓰기'를 한 번에 해결해낼 수 있습니다. Vi와 Emacs에 '우린 마우스를 쓰지 않겠다'라는 강렬한 의지가 돋보이는 단축키들과 명령어들이 도처에 깔려 있는데, 하물며 쉘은 안 그럴까요? 오늘은 bash shell (과 아마 이와 비슷한 zsh, tcsh 등의 쉘에서도 적용될)에 숨어있는 다양한 단축키들을 알아보도록 하겠습니다.

## GNU Readline
bash에 아주아주 긴 명령어를 입력하고 있었는데 세상에, 두 번째 인자에 오타가 있습니다. 언제 방향키를 통해 다시 두 번째 인자로 돌아갈까요? 이런 불편함을 해소하기 위해 bash 같은 쉘은 GNU Readline이라는 라이브러리를 이용합니다. 덕분에 단어 단위로 커서 이동, kill ring 운영, tab control, 이전에 실행한 명령어를 빨리 실행할 수 있는 편리한 기능들을 두세번의 타이핑만 필요한 단축키만으로 이용할 수 있습니다. 이 글에서 소개하는 대부분의 단축키들은 거진 다 GNU Readline에서 제공하는 것입니다.

### vi vs emacs
앞으로 수많은 단축키들이 열거될 텐데, 이 단축키들은 유명한 텍스트 편집기인 emacs에 그 기원을 듭니다 (Emacs mode). 당연하게도 Vi mode도 있습니다. Vi mode로 바꾸려면
```
$ set -o vi
```
를 bash에 입력하시면 됩니다. 만약 Emacs mode를 쓰고 싶으시면
```
$ set -o emacs
```
를 입력하시면 됩니다.

## 입력
기본적으로 어떤 문자를 입력하고 싶으면 그 문자에 해당하는 자판을 누르면 됩니다. a를 입력하고 싶으면 `a`를, `@`을 입력하고 싶으면 `Shift + 2`를 누르는 것처럼 말입니다. 기본적으로 무슨 문자들이 있는지를 살펴보려면 쉘에
```
$ man ascii
```
를 입력하면 보다 상세한 설명을 볼 수 있습니다.

### 컨트롤 문자 입력
그중 Control(Ctrl) 키와 다른 키를 조합하면 전혀 다른 일이 일어난다는 점은 대부분의 프로그래머들이 알고 있습니다. 예를 들어 `Ctrl + c`는 `SIGINT` 인터럽트를 프로그램에 보내고, `Ctrl + d`는 EOF를 입력합니다. Ctrl 키가 무슨 역할을 하는 걸까요?

Ctrl키의 역할은 단순합니다. Ctrl을 누르고 다른 키를 누르면 해당 키의 아스키 코드에서 7번째 비트와 6번째 비트를 없앱니다. 즉, `0x1f`와 bitwise and를 수행한 값을 입력합니다. 공교롭게도, 아스키 대문자는 7번째 비트가, 아스키 소문자는 6번째와 7번째 비트가 1입니다. 때문에 Ctrl을 누르고 아스키 문자를 입력하면 비트가 지워져 같은 아래 5비트를 가지는 **컨트롤 문자**가 전송됩니다. `C`와 `v`에 대한 예시는 다음과 같습니다.

![Ctrl키의 역할](/assets/images/evenharder-post/bash-shortcut/ctrl-key-pressed-white.png)

일반적으로 컨트롤 문자(아스키 코드가 31, 즉 `0x1f` 이하인 문자들)가 전송될 경우 bash는 설정에 따라 다양한 동작을 진행합니다. 예를 들어, 소문자 `j`는 아스키 코드 `0x6a`입니다. Ctrl을 눌러 비트를 없애면, 이 값은 `0x0a`로 변경되고, 여기에 해당하는 아스키 문자는 `\n`(newline) 문자입니다. `\n` 문자의 기능은 줄바꿈입니다. 실제로 `Ctrl + j`를 누르면 줄바꿈이 이루어짐을 알 수 있습니다. 마찬가지로 `Ctrl + J`(`Ctrl + Shift + j`)도 마찬가지입니다.

그럼 `Ctrl + C`를 누르면 무슨 일이 일어나는 걸까요? 기본적으로는 아스키 코드 `0x03`에 해당하는 문자(end-of-text)가 입력되어야 하나, 입력에 앞서서 미리 정의되어 있는 단축키에 의해 이미 말했던 `SIGINT` 인터럽트가 불립니다. 이런 다른 기능을 하는 조합이 '단축키'가 됩니다.

만약 컨트롤 문자 그 자체를 입력하고 싶으면, `Ctrl + v`를 입력한 다음 해당하는 키를 입력하면 됩니다. 예시로 `Ctrl + v Ctrl + c`를 누르면 아스키 코드가 3인 문자 `ETX`가 쉘에 온전히 입력됩니다. 엔터를 누르면 해당 문자가 일반적인 문자로 처리됨을 알 수 있습니다. 이 경우 대응되는 알파벳에 `^`(caret)이 앞에 붙어 컨트롤 문자임을 나타냅니다. (일반적으로 caret 뒤에는 알파벳 대문자가 옵니다)
![sleep 명령어를 통해 확인해보는 ^C, ^V](/assets/images/evenharder-post/bash-shortcut/peek-20190410-ctrl-v.gif)

### 메타 키
Meta key는 Ctrl처럼 modifier key로, 그 자체로만은 아무 역할이 없지만 다른 키랑 조합될 경우 다른 행동을 야기하는 키입니다. 현대 키보드에는 이 Meta Key 자체는 사라졌으나 `Alt`, `Esc`, Windows 운영체제의 경우 `Windows` 키가 그 역할을 대체합니다. 이후 조합에서 `Meta`가 보일 경우 이 키들을 대신 누르시면 됩니다.

## 커서 이동 및 삭제
방향키 사용을 최소화하기 위해 일반적으로 커서 이동에 화살표 키를 사용하는 것과는 달리 이미 다른 단축키에 할당이 되어 있습니다.

`Ctrl + b` : 커서를 한 칸 앞으로 움직입니다 (backward).

`Ctrl + f` : 커서를 한 칸 뒤로 움직입니다 (forward).

`Ctrl + h` : 커서 앞에 있는 글자를 지웁니다 (backspace).

`Ctrl + d` : 커서에 있는 글자를 지웁니다 (delete).

`Ctrl + a` : 커서를 줄의 맨 앞으로 보냅니다.

`Ctrl + e` : 커서를 줄의 맨 뒤로 보냅니다.

`Meta + b` : 커서를 현재 단어의 시작으로 보냅니다. 이미 단어의 시작에 있을 경우 이전 단어의 시작으로 보냅니다.

`Meta + f` : 커서를 현재 단어의 끝으로 보냅니다. 이미 단어의 끝에 있을 경우 다음 단어의 끝으로 보냅니다.

![단어 사이로 커서 이동하기. 단어가 길어지면 더 시간을 아낄 수 있습니다.](/assets/images/evenharder-post/bash-shortcut/peek-20190410-lorem-ipsum.gif)

`Ctrl + n` : 다음 명령어를 입력창에 놓습니다 (next).

`Ctrl + p` : 이전 명령어를 입력창에 놓습니다 (previous).

## 잘라내기, 붙여넣기 (kill ring)
GNU Readline은 잘라낸 단어들을 kill ring이라는 곳에 deque의 형태로 저장해서, 돌려가며 쓸 수 있게 합니다.

`Ctrl + w` : 커서 앞의 단어를 잘라냅니다. 직전에 사용한 명령어가 `Ctrl + w`였을 경우 이전에 자른 단어까지 합쳐서 kill ring에 저장합니다.

`Meta + d` : 커서 뒤의 단어를 잘라냅니다.

`Ctrl + u` : 줄의 처음부터 커서 앞까지를 잘라냅니다.

`Ctrl + k` : 커서 뒤부터 줄의 끝까지를 잘라냅니다.

`Ctrl + y` : kill ring의 맨 앞에 있는 문자열을 붙여넣습니다 (yank).

`Meta + y` : kill ring의 맨 앞에 있는 문자열을 맨 뒤로 보냅니다. 직전에 사용한 명령어가 `Ctrl + y`이거나 `Meta + y`일 때만 사용할 수 있습니다.

![간단한 kill ring 운영 방법](/assets/images/evenharder-post/bash-shortcut/peek-20190410-kill-ring.gif)

gif를 보면 잘라낸 문자열이 deque 같은 자료구조에 저장되는 것을 알 수 있습니다. front에 가까울수록 최근에 잘린 문자열입니다. 가장 최근에 잘라낸 문자열을 이후에 사용할 가능성이 높은 점을 감안하면 단순하면서도 강력한 clipboard 시스템이라고 할 수 있습니다.

## 교체
일부 단어들의 순서를 바꿀 때도 용이합니다.

`Ctrl + t` : 커서 바로 앞의 글자와 커서 밑의 글자를 바꾸며 커서를 한 칸 전진시킵니다.

`Meta + t` : 커서 앞의 있는 두 단어를 바꾸고 커서를 두 단어의 뒤에 위치시킵니다.

![단어 바꾸기](/assets/images/evenharder-post/bash-shortcut/peek-20190410-swap.gif)

## 수정
`Meta + u` : 현재 커서부터 현재 단어의 끝까지의 모든 알파벳 소문자를 대문자로 바꾼 뒤, 커서를 단어의 뒤로 위치시킵니다 (upper).

`Meta + l` : 현재 커서부터 현재 단어의 끝까지의 모든 알파벳 대문자를 소문자로 바꾼 뒤, 커서를 단어의 뒤로 위치시킵니다 (upper). 

`Meta + c` : 현재 커서부터 현재 단어의 끝까지의 알파벳 문자 중 첫 번째는 대문자로, 나머지는 소문자로 바꾼 뒤, 커서를 단어의 뒤로 위치시킵니다 (capitalize). 

![알파벳 대소문자 바꾸기](/assets/images/evenharder-post/bash-shortcut/peek-20190410-alphabet-case.gif)
 
## 이전 명령어 기록 불러오기
`!!` : 직전의 명령을 반복합니다.

`!x` : `x`로 시작하는 가장 직전에 실행했던 명령을 그 때의 인자와 함께 실행합니다. `x`는 여러 글자일 수 있습니다.

`!n` : `n`이 양수일 경우, `history` 명령을 통해 나오는 명령어 기준으로 `n`번째에 실행된 명령을 실행합니다. 음수일 경우, `-n`번 전에 실행했던 명령어를 실행합니다.

![이전 명령 실행하기. 자주 쓰는 명령은 !x 꼴로 부르면 편할 것 같습니다.](/assets/images/evenharder-post/bash-shortcut/peek-20190410-history.gif)


## 프로세스 조절
`Ctrl + c` : `SIGINT` 인터럽트를 프로세스에 보냅니다 (일반적으로 프로세스가 종료됩니다)

`Ctrl + l` : 현재 커서가 있는 줄을 맨 위로 위치시킵니다 (`clear` 명령어와 비슷합니다).

`Ctrl + s` : 쉘에 출력하는 것을 중지시킵니다. 이 상태에서 스크롤을 할 수도 있습니다.[예전에 텔레타이프가 현역 시절일 때부터 있던 유서 깊은 기능이라고 합니다](https://unix.stackexchange.com/questions/137842/what-is-the-point-of-ctrl-s#comment221445_137846).

`Ctrl + q` : `Ctrl + s`를 통해 중지한 출력을 재개합니다.

이런 특이한 동작이 일어나는 이유는 `Ctrl + q`나 `Ctrl + s`로 전송되는 컨트롤 문자가 각자 `DC1`(device control 1), `DC3`(device control 3)이기 때문입니다.[^1]

`Ctrl + d` : EOF(end-of-file) 마커를 보냅니다.[^2] 쉘에 입력될 경우 쉘을 종료합니다.

`Ctrl + z` : `SIGTSTP` 인터럽트를 보내 백그라운드로 돌립니다. 백그라운드의 프로세스는 `bg`, `fg` 등으로 관리할 수 있습니다.

## 기타
`Tab`, `Ctrl + i` : 자동완성을 진행해줍니다.

## 마치며
*nix 계열의 산물들은 정말 편의성을 극대화하기 위한 장치들이 곳곳에 숨어있습니다. 쉘의 단축키도 그 결실 중 하나라고 생각됩니다. CLI에서 수많은 명령을 돌려야 한다면, 이러한 단축키들이 '이동-지우기-고쳐쓰기'나 '일일히 다 치기' 과정을 조금 편하게 해주지 않을까 싶습니다. 백문이 불여일견이라고, 직접 단축키를 입력해보면 손에 익을 것이라고 생각됩니다. 모쪼록 편한 프로그래밍 라이프를 기원하겠습니다.

[^1]: Software flow control을 검색해보시길 바랍니다.
[^2]: Windows 운영체제에서는 `Ctrl + z`입니다.

## 출처
* [http://web.mit.edu/gnu/doc/html/features_7.html](http://web.mit.edu/gnu/doc/html/features_7.html)
* [https://ss64.com/bash/syntax-keyboard.html](https://ss64.com/bash/syntax-keyboard.html)
* [https://catern.com/posts/terminal_quirks.html](https://catern.com/posts/terminal_quirks.html)

<br>

