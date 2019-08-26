---
layout: post
title:  "Shellcoding and Bitflip Conjecture"
date:   2019-08-19 23:59
author: RBTree
tags: [shellcoding, ctf]
---

# 서론

이번에 DEF CON CTF 2019에 팀 SeoulPlusBadass로 다녀왔습니다. [결과 링크](https://www.oooverflow.io/dc-ctf-2019-finals/)

대회의 대부분의 문제가 프로그램의 취약점을 찾아서 익스플로잇하는 Pwnable 문제였는데, 그렇지 않은 문제는 ropship, DOOOM, bitflip conjecture 세 문제였습니다.

그 중에서도 Bitflip Conjecture는 흥미로운 Shellcoding 문제여서 이번에 공유하고자 합니다.

# Bitflip Conjecture

Bitflip Conjecture는 X64 Assembly로 작성된 Binary를 전송하는 문제입니다. 자세한 것은 문제에 접속하면 나오는 메시지를 살펴봅시다:

```
===============================================================================

Definition:
  A snippet of assembly code is `N-Flip Resistant` if its output remains
  constant (i.e., it produces the same output and exits with the same
  return value) even if ANY combination of N bits are flipped. 

One-flip Conjecture:
  The x86 architecture is such that it is possible to write any arbitrary
  program (of any length) in a way that is 1-flip resistant. 
                                       - Balzaroth (Vegas 2019)

===============================================================================

It is now your turn to provide a proof for this Conjecture, which has puzzled 
hackers and security researchers for hundreds of years.
Provide a shellcode (max 200 bytes) that prints 'I am Invincible!' and
then terminates with exit code zero.

Points are assigned based on how close you are from a complete proof
(i.e., based on how many bit flip your code was able to withstand)

-------------------------------------------------------------------------------

But first, how do you want the registers initialized before executing the code?
 1. I like all my registers set to zero
 2. I want them pointing to the middle of a 64KB R/W region of memory)
 3. Dont bother. Leave them as they are
```

즉, 주어진 코드의 임의의 n-bit를 flip해도 원래 의도대로 돌아가는 코드를 **n-flip resistant**하다고 정의합니다. 그리구 문제에서는 `'I am Invincible!'`을 출력하는 1-flip resistant한 코드를 작성하는 것이 목표입니다.

실제 문제에서는 최대 200 bytes를 보냈을 때 모든 1 bit를 한 번 씩 flip해보고, 그 중 올바르게 작동하지 않은 경우의 수를 카운트해서 1000점에서 빼서 점수를 부여했습니다. 예를 들어, 1, 2, 3, 4번째 bit를 flip했더니 올바르게 작동하지 않았다면 1000 - 4 = 996점이 됩니다.

또한 바이너리를 읽어보면, seccomp를 통해 write, exit을 제외한 syscall을 막는 것을 알 수 있습니다.

```c
  LODWORD(v5) = seccomp_init(0LL);
  v6 = v5;
  seccomp_rule_add(v5, 2147418112LL, 231LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 60LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 1LL, 0LL);
  seccomp_load(v6);
  v14 = atoi(argv[2])
```

X64 Assembly 코딩을 하기 위해서는 어셈블리에 대해서 잘 알고 있어야겠죠? 그리고 어셈블리로 쉽게 코딩할 수 있는 방법이 필요할 겁니다. 안타깝게도 어셈블리에 대해서 설명하는 것은 매우 시간이 오래 걸리기 때문에, 어셈블리 자체보다는 어떻게 하면 어셈블리를 CTF level에서 쉽게 코딩해보고, 어떻게 문제를 풀었는지에 초점을 맞추려고 합니다.

#Shellcoding

Shellcoding을 편하게 하는 방법은 여러 가지가 있을 수 있지만, 저는 Python의 pwntools 라이브러리를 사용하는 편입니다. pwntools에는 Shellcraft와 같은 Shellcoding 편의 기능이 있기 때문에, 익혀두면 좋을 것입니다.

우선, `'I am Invincible!'`을 출력하는 어셈 코드를 작성해봅시다.

```
from pwn import *
context.arch = 'amd64'

base_code = """
    /* push 'I am Invincible!' */
    mov rax, 0x21656c6269636e69
    push rax
    mov rax, 0x766e49206d612049
    push rax
    /* write(fd=1, buf='rsp', n=16) */
    push 1
    pop rdi
    push 0x10
    pop rdx
    mov rsi, rsp
    /* call write() */
    push SYS_write /* 1 */
    pop rax
    syscall
    /* exit(status=0) */
    xor edi, edi /* 0 */
    /* call exit() */
    push SYS_exit /* 0x3c */
    pop rax
    syscall
"""

asm_code = asm(base_code)
print(disasm(asm_code))

with open('shellcode', 'wb') as f:
    f.write(asm_code)
```

Stack에 `'I am Invincible!'` 이라는 string을 push하고, write syscall을 통해서 이를 출력시킵니다. 그 뒤 exit syscall을 통해 프로그램을 종료시킵니다.

이를 주어진 바이너리에 넣고 실행해보면, `'I am Invincible!'`을 정상적으로 출력하고 끝납니다.

##1-Flip Resistant 하게 만들기 위한 아이디어

이제 이 코드를 1-flip resistant하게 만들어야 합니다. 위의 메시지를 다시 살펴보시면, 문제에서 세 가지 옵션을 줬습니다.

```
But first, how do you want the registers initialized before executing the code?
 1. I like all my registers set to zero
 2. I want them pointing to the middle of a 64KB R/W region of memory)
 3. Dont bother. Leave them as they are
```

1번은 모든 레지스터를 0으로 세팅하고, 2번은 모든 레지스터를 mmap을 통해 할당한 R/W 영역의 위치로 세팅합니다. 3번은 우리가 전송한 shellcode를 실행하기 직전의 레지스터 상태를 그대로 들고 갑니다.

그런데 shellcode를 실행하기 직전의 상태를 보면 다음과 같습니다.

```
───────────────────────────────────────────────────────[ registers ]────
$rax   : 0x00007ffff7ff6000  →  0x9090909090909090
$rbx   : 0x0000000000000000
$rcx   : 0x0000000000000037
$rdx   : 0x00007ffff7ff6037  →  0xb848240cfe016a03
$rsp   : 0x00007fffffffe4f0  →  0x00007ffff7ffe168  →  0x0000555555554000  →   jg 0x555555554047
$rbp   : 0x00007fffffffe540  →  0x0000555555554db0  →  <__libc_csu_init+0> push r15
$rsi   : 0x0000000040000000
$rdi   : 0x00007fffffffe89d  →  0x2f3d4c4c45485300
$rip   : 0x0000555555554d99  →  <main+655> call rax
$r8    : 0x0000000000000000
$r9    : 0x1999999999999999
$r10   : 0x0000000000000000
$r11   : 0x00007ffff76345e0  →  0x0002000200020002
$r12   : 0x0000555555554a00  →  <_start+0> xor ebp, ebp
$r13   : 0x00007fffffffe620  →  0x0000000000000005
$r14   : 0x0000000000000000
$r15   : 0x0000000000000000
$eflags: [carry PARITY adjust zero sign trap INTERRUPT direction overflow resume virtualx86 identification]
$ds: 0x0000  $ss: 0x002b  $es: 0x0000  $fs: 0x0000  $cs: 0x0033  $gs: 0x0000
```

각 레지스터 값을 실제로 살펴보면, `rax`는 쉘코드의 시작 위치를 나타냅니다. 그리고 `rcx`는 몇 번째 바이트의 한 비트를 뒤집을지를 나타내고, stack memory를 살펴보거나 `xmm0` 레지스터를 살펴보면 그 중 몇 번째 비트를 flip하는 지도 저장되어 있습니다. `rdx`는 `rax+rcx` 값을 가리키고 있습니다.

즉, 우리의 shellcode가 실행될 때 레지스터와 메모리 값을 통해서 어느 비트가 flip 되었는 지를 알아낼 수 있고, 곧 3번 옵션을 선택하는 것이 매우 좋은 방법인 것 같습니다. 그렇다면 어떻게 이 문제를 풀어볼 수 있을까요?

### 1. Flip된 bit를 다시 flip하기

어디가 flip되었는지를 알고 있기 때문에, 다시 flip할 수 있습니다.

```
cvttsd2si eax, xmm0
xor [rdx], al
```

위의 assembly code를 사용하면, 다시 flip을 할 수 있게 됩니다. 하지만 해당 instruction은 6byte입니다. 만약 이 instruction 부분이 flip되면 정상적으로 돌려놓을 수 없을 것입니다. 실제로 이를 실행해봅시다. 실행 코드는 다음과 같습니다.

```python
import os

res = os.popen('./test shellcode 3').read()
if res != 'I am Invincible!':
    print("BASIC VERIFICATION FAIL")

cnt = 0
for byt in range(200):
    for bit in range(8):
        res = os.popen('./test shellcode 3 {} {} 2>&1'.format(byt, bit)).read()
        if res == 'I am Invincible!':
            cnt += 1
        else:
            print(byt, bit)
            print(res)

print('Result: {} / {}'.format(cnt, 200 * 8))
```

우선 Basic verification을 통과하지 못합니다. 아무 bit도 flip하지 않은 경우에 대해서도 동작해야하는데 동작하지 않습니다. 그리고 `Result: 1574 / 1600` 라는 결과가 나왔습니다. 26개의 bit에서 resistant하지 않았습니다. 이것으로는 만점과는 거리가 있어보입니다.

### 2. Flip된 위치에 따라서 Branch code를 사용하기

정상적인 코드를 2개를 붙여놓아봅시다.

```
─────────────────────────────────────────────────
Printing 'I am Invincible!'
─────────────────────────────────────────────────
Printing 'I am Invincible!'
─────────────────────────────────────────────────
```

한 bit만 flip되기 때문에, 앞 코드가 비정상이 되거나 뒷 코드가 비정상이 되거나 둘 중 하나입니다. 만약 두 코드 중  정상적인 코드를 하나 선택해서 그 쪽으로 jump할 수 있다면 좋을 것입니다.

```
─────────────────────────────────────────────────
If flip pos is A:
    Jump to B
else:
    continue (Jump to A)
─────────────────────────────────────────────────
[A] Printing 'I am Invincible!'
─────────────────────────────────────────────────
[B] Printing 'I am Invincible!'
─────────────────────────────────────────────────
```

이를 작성해보면 다음과 같습니다.

```python
from pwn import *

context.arch = 'amd64'
base_code = """
    /* push 'I am Invincible!\x00' */
    mov rax, 0x21656c6269636e69
    push rax
    mov rax, 0x766e49206d612049
    push rax
    /* write(fd=1, buf='rsp', n=16) */
    push 1
    pop rdi
    push 0x10
    pop rdx
    mov rsi, rsp
    /* call write() */
    push SYS_write /* 1 */
    pop rax
    syscall
    /* exit(status=0) */
    xor edi, edi /* 0 */
    /* call exit() */
    push SYS_exit /* 0x3c */
    pop rax
    syscall
"""

len_base_code = len(asm(base_code))

branch_code = """
    cmp cl, {}
    jb LABEL
""".format(len_base_code + 5 + 50)
# 5: len(branch_code)
# 50: offset

code = branch_code + base_code
code += "LABEL:"
code += base_code

code = asm(code)

with open('shellcode', 'wb') as f:
    f.write(code)
```

이를 테스트해보면 `Result: 1579 / 1600`가 나옵니다. 이 쪽이 좀 더 안정성이 있고, jump하는 코드 (`branch_code`)를 잘 고치기만 한다면 점수를 더 높일 수 있을 것 같습니다. 실제로 대회에서는 이 방법으로 계속 진행했습니다.

### 3. 1-flip Resistant한 코드 찾기

만약 A라는 코드가 있을 때, A의 임의의 1-bit를 flip했을 때 Invalid Instruction이나 Segmentation Fault와 같은 오류 없이 정상적으로 실행이 되기는 한다고 해봅시다. 그러면 이런 코드를 2개 연달아 붙여놓으면 둘 중 하나는 정상적으로 실행이 되어서 원래 의도대로 실행이 될 것입니다.

이런 아이디어를 사용한다면 문제에서 준 3개의 옵션 중 `2번: 모든 레지스터를 mmap을 통해 할당한 R/W 영역의 위치로 세팅한다.` 를 응용할 방법이 있을 지도 모릅니다.

하지만 대회가 끝날 때까지 쉽게 생각하지 못했습니다. 이에 대해서 참고할만한 흥미로운 [17년도 BlackHat PPT](https://www.blackhat.com/docs/us-17/thursday/us-17-Domas-Breaking-The-x86-ISA.pdf)가 있습니다. 나중에 시간날 때 한 번 체크해보세요.

## 1-Flip Resistant한 코드 작성하기

앞에서 살펴본 코드를 다시 봅시다.

```
    cmp cl, offset
    jb LABEL
```

`cmp cl, offset`은 `offset`이라는 immediate value와 `cl` register를 비교하는 instruction으로 4 byte를 차지합니다. 1-bit flip 된다고 하면 `offset` 값이 크게 변하거나 하면서 취약해질 수도 있고, 경우에 따라서 `[register + offset]` memory 위치를 참조하는 코드로 바뀌면서 segmentation fault로 변할 가능성도 있습니다.

그리고 X64 Assembly의 opcode를 볼 수 있는 [사이트](ref.x86asm.net/coder64.html)도 참조를 해봅시다. `jb` instruction은 opcode가 `72`인데, 당장 1-bit flip하면 나올 수 있는 `62` opcode가 invalid opcode입니다. 즉 더 좋은 jump instruction을 사용할 필요성이 있습니다.

해당 사이트를 참조해서 살펴보면, `78`부터 `7D` 까지의 opcode를 사용할 수 있는 것으로 보입니다. 적어도, 1-bit flip 했을 때 invalid opcode가 되지는 않습니다. 그런데 `jp/jnp` instruction은 parity flag에 따라 jump하는 instruction입니다. Parity flag는 instruction을 수행하고 나온 결과가 짝수냐 홀수냐에 따라서 set되는 flag이기 때문에 사용하는 것은 어려워보입니다. 곧 `js/jns/jl/jnl`이 현실적으로 사용 가능한 instruction입니다.

이러한 결론에 따라서 다음과 같이 코드를 작성해볼 수 있습니다.

```python
from pwn import *
context.arch = 'amd64'

base_code = """
    /* push 'I am Invincible!' */
    mov rax, 0x21656c6269636e69
    push rax
    mov rax, 0x766e49206d612049
    push rax
    /* write(fd=1, buf='rsp', n=16) */
    push 1
    pop rdi
    push 0x10
    pop rdx
    mov rsi, rsp
    /* call write() */
    push SYS_write /* 1 */
    pop rax
    syscall
    /* exit(status=0) */
    xor edi, edi /* 0 */
    /* call exit() */
    push SYS_exit /* 0x3c */
    pop rax
    syscall
"""

len_base_code = len(asm(base_code))

branch_code = """
branch_start:
    mov bl, branch_end - branch_start + 50
    sub cl, bl
    jge branch_start + 0x80
branch_end:
"""

code = branch_code
code += 'nop\n' * 10
code += base_code
code += 'nop\n' * 98
code += base_code
asm_code = asm(code)

with open('shellcode', 'wb') as f:
    f.write(asm_code)
```

이제 이를 통해서 `Result: 1594 / 1600`까지 줄일 수 있었습니다.

그리고 여기에서 계속 하나씩 깎고, 깎고, 깎다보면, 다음과 같은 code까지 올 수 있습니다.

```python
from pwn import *
context.arch = 'amd64'

base_code = """
    /* push 'I am Invincible!' */
    mov rax, 0x21656c6269636e69
    push rax
    mov rax, 0x766e49206d612049
    push rax
    /* write(fd=1, buf='rsp', n=16) */
    push 1
    pop rdi
    push 0x10
    pop rdx
    mov rsi, rsp
    /* call write() */
    push SYS_write /* 1 */
    pop rax
    syscall
    /* exit(status=0) */
    xor edi, edi /* 0 */
    /* call exit() */
    push SYS_exit /* 0x3c */
    pop rax
    syscall
"""

len_base_code = len(asm(base_code))

branch_code = """
branch_start:
    xor al, branch_end - branch_start + 119
    cmp al, cl
    jns branch_end + 119
branch_end:
"""

code = branch_code
code += 'push rax\n'
code += 'nop\n' * 9
code += base_code
code += 'nop\n' * 98
code += base_code
asm_code = asm(code)
print(len(asm_code))
print(asm_code.encode('hex'))
print(disasm(asm_code))

with open('shellcode', 'wb') as f:
    f.write(asm_code)
```

이는 경험적으로 나온 부분인데, 어떻게 코드가 나왔는지를 요약하면 다음과 같습니다.

- `js/jns`를 사용하면 `jl/jnl`을 사용할 때보다 jump instruction의 bit flip이 일어날 때 좀 더 잘 버팁니다.
- `cmp`의 opcode가 `80`이여서 처음에는 사용하지 못한다고 생각했는데, register끼리 비교하는 cmp의 경우 `38~3D` 로 사용이 가능합니다. 이 중에서도 `cmp bl, cl`을 쓸 지, `cmp cl, bl`을 쓸지 등등 다양하게 try해보면 그 중에서도 `cmp al, cl`이 안정적이라는 것을 알 수 있었습니다.
- 또한 `mov`의 경우 bitflip을 하다보면 illegal instruction이 무조건 한 번 나오는데, `al`이 0이기 때문에 `xor`을 통해서 초기화가 가능한데 이 경우 illegal instruction을 아예 없앨 수 있습니다.

# 마무리
결과적으로 한 bit에 대해서만 동작하지 않는 코드가 완성되어서 999점을 받을 수 있었습니다. 대회가 끝난 뒤 찾아보니, 한 팀이 1000점을 받는데 성공했고 저희를 포함해 3~4팀이 999점을 받은 것으로 보입니다.

해당 점수를 따라서 등수를 나누고, 등수에 따라서 **5등까지** 5분마다 점수가 들어오는 방식이였기 때문에 999점 코드를 만들고 당분간 점수를 받을 수 있었습니다. 이를 통해 얻은 점수가 대략적으로 KoTH 장르 점수에서 300점이었는데, KoTH 장르는 전체 점수의 20%로 합산이 되기 때문에 대략 60점 정도를 팀에 기여할 수 있었습니다. 999점 미만의 점수를 받았다면 5등 이내에 들지 못해 점수를 받지 못했을 것이고, [결과](https://www.oooverflow.io/dc-ctf-2019-finals/)를 보다시피 이 60점이 없었으면 12등까지 내려가는 참사가 벌어졌기 때문에 정말 다행이라고 생각하고 있습니다.

Shellcoding을 한지 몹시 오래 되었는데 이번 기회에 다시 연습해볼 수 있어서 좋았다고 생각합니다.