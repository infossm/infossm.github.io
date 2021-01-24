---
layout: post
title:  "SMT Solver in CTF - z3 vs Boolector"
date:   2020-06-19 23:50
author: RBTree
tags: [SMT, CTF]
---

# 서론

SMT (Satisfiability modulo theories) Solver는 CTF에서 빠질 수 없는 존재입니다. SMT Solver가 무엇인지 구체적으로 설명하기 보다는, 다음 z3py 예시를 보는 것이 더 직관적일 것입니다. ([Link](https://ericpony.github.io/z3py-tutorial/guide-examples.htm))

```python
x = Int('x')
y = Int('y')

s = Solver()
print s

s.add(x > 10, y == x + 2)
print s
print "Solving constraints in the solver s ..."
print s.check()

print "Create a new scope..."
s.push()
s.add(y < 11)
print s
print "Solving updated set of constraints..."
print s.check()

print "Restoring state..."
s.pop()
print s
print "Solving restored set of constraints..."
print s.check()
```

보다시피 solver에 `x > 10`, `y == x + 2`, `y < 11` 과 같은 condition을 주었을 때 이 값을 만족하는 `x`, `y` 값을 찾게 됩니다.

SMT Solver는 CTF에서 대체로 1. 로직 자체는 파악을 다 했으나 역연산이 불가능할 때 2. 단순한 백트래킹 이상으로 더 똑똑한 알고리즘이 없어보일 때 시도하게 되는 최후의 보루라고 할 수 있습니다. 큰 문제일수록 SMT Solver의 수행 시간은 길어지게 되지만, CTF 시간은 제한적이기 때문에 빠른 SMT Solver를 고르는 것은 중요한 문제라고 할 수 있습니다.

그러나 지금까지 CTF에서는 z3가 가장 많이 사용되는 모습을 보여주었습니다. 아마 처음 쓰일 당시에는 z3가 제일 빨랐을 가능성도 있고, 다른 SMT Solver에 비해서 z3가 z3py라는 매우 쓰기 편한 Python API를 제공하는 탓도 클 것이라고 생각합니다. 하지만 지금은 다양한 SMT solver가 나왔고, 매년 SMT-COMP([Link](https://smt-comp.github.io/introduction.html))라는 SMT Solver 관련 대회 및 워크샵이 개최되고 있습니다.

그러면 해당 대회에서 가장 성능이 좋아보이는 SMT Solver를 쓰면 되지 않느냐 생각할 수 있지만, 언제까지나 'CTF'에서 쓰기 편하고 성능이 좋은 SMT Solver를 구하는 것이 목적입니다. SMT-COMP 2019의 결과를 살펴보면 다양한 SMT Solver가 있으나, 이 중에서 Python API를 지원하는 것은 Boolector([Link]()) 뿐이어서 Boolector와 z3를 비교하는 방향으로 진행하게 되었습니다. 진행 과정에서는 실제 CTF에서 사용했던 Solver의 실행 시간을 위주로 비교하려고 합니다.

# 본론

## Boolector 설치

목표도 잘 잡았고, 이제 Boolector를 사용해보기만 하면 될 것 같아보입니다.

z3의 경우 Installation이 어렵지 않았습니다. PIP를 통해 한 번에 install이 가능하기 때문입니다. `pip install z3-solver` 커맨드를 사용하면 손쉽게 설치가 가능합니다.

Boolector는 PyPI에 올라가 있지 않아 직접 빌드해주어야 합니다.

```shell
# Download and build Boolector
git clone https://github.com/boolector/boolector
cd boolector

# Download and build Lingeling
./contrib/setup-lingeling.sh

# Download and build BTOR2Tools
./contrib/setup-btor2tools.sh

# Build Boolector
./configure.sh && cd build && make
```

이 때 우리는 Python API도 사용할 것이기 때문에 `./configure.sh` 에 `--python` flag를 주어야 합니다. 만약 Python3 유저라면 덤으로 `--py3` 플래그를 붙여 `./configure.sh --python --py3` 으로 실행해야 합니다.

이렇게 설치하게 되면 `build/lib` 안에 `pyboolector.so`와 기타 파일이 생깁니다. 이 `build/lib`을 `PYTHONPATH` 환경 변수로 export하고 실행하면 됩니다. 저는 홈 폴더 위에 git repository를 클론 받아 빌드했기 때문에 다음과 같이 실행했습니다.

```shell
PYTHONPATH="~/boolector/build/lib" python3 boolector_sol.py
```

## Boolector의 문제

이제 기존에 CTF에서 사용했던 z3 solver를 Boolector로 옮겨보기로 했습니다. 이 때까지만 해도 Boolector Python API가 겉보기에 z3py와 비슷해 간단하게 z3py의 일부 함수나 구문들을 Boolector에서 쓰이는 함수 이름으로 바꾸면 될 것이라고 생각했는데, 매우 큰 오산이었습니다.

다음 코드는 PlaidCTF 2020의 A Plaid Puzzle을 푸는 과정에서 사용된 코드입니다. 순서 없이 섞여있는 원소 256개 집합 A, B, C에 대해 f: A x B -> C가 주어져 있을 때, f가 만약 [GF(256)](https://en.wikipedia.org/wiki/Finite_field) 위에 정의된 곱셈 연산이라면 A, B, C를 GF(256) 와 일대일 대응시킬 수 있는 지를 찾는 코드입니다.

```python
from z3 import *

var_list = [ [] for _ in range(4) ]
var_dict = dict()

for i in range(64):
    ch_name = 'char%02d' % i
    a_name = 'var_A_%02d' % i
    b_name = 'var_B_%02d' % i
    c_name = 'var_C_%02d' % i

    var_char = BitVec(ch_name, 6)
    var_a = BitVec(a_name, 6)
    var_b = BitVec(b_name, 6)
    var_c = BitVec(c_name, 6)

    var_dict[ch_name] = var_char
    var_dict[a_name] = var_a
    var_dict[b_name] = var_b
    var_dict[c_name] = var_c

    var_list[0].append(var_char)
    var_list[1].append(var_a)
    var_list[2].append(var_b)
    var_list[3].append(var_c)

    
with open('ab_rules', 'r') as f:
    ab_rules = f.readlines()

with open('bc_rules', 'r') as f:
    bc_rules = f.readlines()

s = Solver()

# modulus = 0b1000011
# 0b1001, 0b10111, 0b110101, 0b11011, 0b11, 0b100001, 0b101101, 0b100111, 0b110011
modulus = 0b1001

def gf_mult(a, b):
    res = BitVecVal(0, 6)
    for i in range(6):
        res = If(res & 32 == 0, res << 1, (res << 1) ^ modulus)
        res = If(LShR(b, 5 - i) & 1 == 1, res ^ a, res)
    return res

for line in ab_rules:
    splitted = line.split()
    a = var_dict[splitted[1]]
    ch = var_dict[splitted[3]]
    b = var_dict[splitted[7]]

    s.add(gf_mult(a, ch) == b)    

for i in range(4096):
    line = bc_rules[i]

    a = var_dict[line[8:16]]
    b = var_dict[line[19:27]]
    c = var_dict[line[35:43]]

    s.add(a ^ b == c)

for i in range(64):
    for j in range(i + 1, 64):
        for k in range(4):
            s.add(var_list[k][i] != var_list[k][j])

if s.check() == unsat:
    print("NOOOO")
    exit(0)

m = s.model()
for k in range(4):
    for var in var_list[k]:
        print(m[var])
```

그리고 이를 Boolector Python API로 옮긴 것이 다음 코드입니다.

```python
import pyboolector
from pyboolector import Boolector

btor = Boolector()
btor.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)
btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 1)

var_list = [ [] for _ in range(4) ]
var_dict = dict()

for i in range(64):
    ch_name = 'char%02d' % i
    a_name = 'var_A_%02d' % i
    b_name = 'var_B_%02d' % i
    c_name = 'var_C_%02d' % i

    var_char = btor.Var(btor.BitVecSort(6), ch_name)
    var_a = btor.Var(btor.BitVecSort(6), a_name)
    var_b = btor.Var(btor.BitVecSort(6), b_name)
    var_c = btor.Var(btor.BitVecSort(6), c_name)

    var_dict[ch_name] = var_char
    var_dict[a_name] = var_a
    var_dict[b_name] = var_b
    var_dict[c_name] = var_c

    var_list[0].append(var_char)
    var_list[1].append(var_a)
    var_list[2].append(var_b)
    var_list[3].append(var_c)

    
with open('ab_rules', 'r') as f:
    ab_rules = f.readlines()

with open('bc_rules', 'r') as f:
    bc_rules = f.readlines()

# modulus = 0b1000011
# 0b1001, 0b10111, 0b110101, 0b11011, 0b11, 0b100001, 0b101101, 0b100111, 0b110011
modulus = 0b1001

def gf_mult(a, b):
    a = btor.Uext(a, 2)
    b = btor.Uext(b, 2)
    res = btor.Const(0, 8)
    for i in range(6):
        res = btor.Cond(res & 32 == 0, res << 1, (res << 1) ^ modulus)
        res = btor.Cond(btor.Srl(b, 5 - i) & 1 == 1, res ^ a, res)
    return res[5:0]

for line in ab_rules:
    splitted = line.split()
    a = var_dict[splitted[1]]
    ch = var_dict[splitted[3]]
    b = var_dict[splitted[7]]

    btor.Assert(gf_mult(a, ch) == b)    

for i in range(4096):
    line = bc_rules[i]

    a = var_dict[line[8:16]]
    b = var_dict[line[19:27]]
    c = var_dict[line[35:43]]

    btor.Assert(a ^ b == c)

for i in range(64):
    for j in range(i + 1, 64):
        for k in range(4):
            btor.Assert(var_list[k][i] != var_list[k][j])

if btor.Sat() == btor.UNSAT:
    print("NOOOO")
    exit(0)

m = btor.Print_model()
for k in range(4):
    for var in var_list[k]:
        print(var.symbol, var.assignment)
```

작성하는 과정에서 다음과 같은 문제점을 찾았습니다.

- z3py의 shift operator는 어떠한 제약도 없는데, Boolector에서는 무조건 vector의 bit-width가 무조건 $2^n$ 꼴이어야 합니다. 그래서 `gf_mult` 의 정의 과정에서 어쩔 수 없이 6-bit vector를 8-bit로 늘리는 것을 볼 수 있습니다.
- Bit vector의 slice operator가 `[upper:lower]`로 일반적으로 Python에서 쓰이는 형태와 다릅니다. `gf_mult`의 마지막 줄을 보시면 `res[5:0]`를 반환하는 것을 볼 수 있습니다. 이렇듯 순서가 반대여서 주의하고 작성해야합니다.
- 마지막에 model을 바탕으로 답을 출력할 때 `var.assignment`는 0과 1로 이루어진 string입니다. 별도로 int로 변환해주는 과정이 필요합니다.
- 이 코드에서는 찾아볼 수 없는 문제인데, 한 번 특정 model을 푼 뒤 다시 `Assert`를 사용해 constraint를 추가해주려고 하면 오류가 생깁니다. 이 경우 `b.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)` 와 같이 별도로 옵션을 설정해줘야지 model의 재사용이 가능합니다.

위와 같은 문제로 z3py로 작성된 Python 코드를 string replacement만으로 Boolector Python API로 바꾸겠다는 계획은 실패했습니다. 그리고 성능 비교를 위해서 일일이 코드를 바꿔줘야 할 뿐만 아니라, Boolector Python API 자체가 더 복잡하고 불편하기 때문에 CTF에서 사용하기에는 귀찮은 부분이 많아보였습니다.

그렇다면 정녕 Boolector를 쉽게 사용할 방법은 없는 걸까요?

## SMT-LIBv2 포맷으로 추출하기

우연찮게 Boolector, Yices, z3의 성능을 비교하고자 한 Stack overflow 글([Link](https://stackoverflow.com/questions/42371139/how-to-analyse-z3-performance-issues))과 이에 첨부된 결과([Link](http://scratch.clifford.at/compact_smt2_enc_r1102.html))를 찾게 되었습니다. 보면서 생각해보니 SMT-COMP에 있는 SMT Solver들도 서로 비교하기 위한 동일한 입력 방식이 있었을 것인데, 계속 놓치고 있었습니다.

해당 글에서는 SMT-LIBv2([Link](www.smtlib.org)) 포맷으로 작성된 파일들을 사용해 벤치마크를 작성했고, SMT-COMP도 동일한 방법을 사용하는 것을 확인할 수 있었습니다. 

마침 Boolector Python API를 보면서 model을 SMT-LIBv2 형태로 export할 수 있다는 것을 찾아냈습니다. 그렇다면 z3에도 동일한 기능이 있지 않을까 생각해서 검색해본 결과 version 4.8.7에서 추가가 된 것을 확인할 수 있었습니다. 다행히도 pip를 통해 설치한 버전은 4.8.8.0으로, 문제 없이 사용할 수 있는 기능이었습니다.

```
Version 4.8.7
=============
- New features
  - setting parameter on solver over the API by
    solver.smtlib2_log=<filename>
    enables tracing calls into the solver as SMTLIB2 commands.
    It traces, assert, push, pop, check_sat, get_consequences.
- Notes
  - various bug fixes
  - remove model_compress. Use model.compact
  - print weights with quantifiers when weight is != 1
```

문제는 릴리즈 노트대로 `solver.smtlib2_log="filename.smt2"`를 사용하면 절대 돌아가지 않는다는 점입니다. z3py와 관련된 이슈([Link](https://github.com/Z3Prover/z3/issues/1451))에서도 똑같이 쓰라고 했기 때문에 되어야 할텐데 어떤 문제인가 살펴봤더니, 다음 Stack overflow 글([Link](https://stackoverflow.com/questions/60698770/smt2-format-logfile))에서 해결법을 찾을 수 있었습니다.

1. 모든 solver에 대해 SMT-LIBv2 형태로 로그를 남기고 싶다면: `set_param("solver.smtlib2_log", "log.smt2")`
2. 특정 solver에 대해서만 SMT-LIBv2 형태로 로그를 남기고 싶다면: `solver.set("smtlib2_log", "log.smt2")`

이제 export를 하려면 다음과 같은 과정을 거치면 됩니다.

1. `Solver()`를 사용하는 코드 다음 줄에 `solver.set("smtlib2_log", "log.smt2")`를 남긴다.
2. `check()` 이후의 코드를 모두 주석처리한다.
   `check()` 자체가 실행이 되어버리면 z3 solver를 실행하기 시작하기 때문에, `check()` 직전까지 실행한 후의 로그에 check하는 코드를 더해주는 것이 좋습니다.
3. 주어진 SMT-LIBv2 로그를 다음과 같이 수정한다.

```
(set-option :produce-models true)
# 기존 log file 내용
(check-sat)
(get-model)
```

이제 model을 출력하는 SMT-LIBv2 input이 완성되었습니다.

## 추출한 SMT-LIBv2 파일로 실행하기

이를 바탕으로 boolector를 실행하면 다음과 같이 결과를 얻을 수 있습니다.

```shell
~/boolector/build/bin/boolector log.smt2
sat
(model
  (define-fun var_B_00 () (_ BitVec 6) #b110100)
  (define-fun var_A_00 () (_ BitVec 6) #b111111)
  (define-fun char56 () (_ BitVec 6) #b101111)
  (define-fun var_B_01 () (_ BitVec 6) #b011110)
  (define-fun var_A_01 () (_ BitVec 6) #b001101)
  (define-fun char05 () (_ BitVec 6) #b011100)
  (define-fun var_B_02 () (_ BitVec 6) #b110001)
  (define-fun var_A_02 () (_ BitVec 6) #b011100)
  (define-fun char23 () (_ BitVec 6) #b110100)
  (define-fun var_B_03 () (_ BitVec 6) #b100110)
  (define-fun var_A_03 () (_ BitVec 6) #b010101)
  (define-fun char57 () (_ BitVec 6) #b100001)
  (define-fun var_B_04 () (_ BitVec 6) #b000110)
  (define-fun var_A_04 () (_ BitVec 6) #b100000)
  (define-fun char34 () (_ BitVec 6) #b101001)
  (define-fun var_B_05 () (_ BitVec 6) #b101001)
  (define-fun var_A_05 () (_ BitVec 6) #b101011)
  ...
```

`-o` 옵션을 사용하면 output을 file로 빼낼 수 있습니다. `#010101`와 같이 binary string으로 출력되는 부분에 대해서 파서는 어쩔 수 없이 작성해야할 것으로 보입니다. 우선은 실행 시간을 비교하는 것이 목표이므로, 파서에 대해서는 고려하지 않았습니다.

z3의 경우 z3py만 설치하면 executable 없이 libz3만 설치되기 때문에 Python 코드를 작성해 SMT-LIBv2 파일을 실행했습니다.

```python
from z3 import *
s = Solver()

with open('log.smt2', 'r') as f:
    s.from_string(f.read())

print(s.check())
m = s.model()
print([(d, m[d]) for d in m])
```

## z3 vs Boolector

제가 이전에 작성한 z3 파일 몇개를 바탕으로 실행해보았습니다. 그런데 생각보다 z3로 작성한 솔버가 많지 않았고, 그 중에서도 작성한 솔버들이 잘 돌아가지 않거나, 실행 시간이 지나치게 길거나, 당시의 의도를 파악할 수 없는 부분이 다소 있어서 정리하는 데에는 시간이 많이 걸릴 것으로 생각됩니다.

현재 실행할 수 있는 3개의 솔버에 대해서 돌려보았습니다.

| Test case                        | z3 (sec) | Boolector (sec) | 비고                                                         |
| -------------------------------- | -------- | --------------- | ------------------------------------------------------------ |
| PlaidCTF 2020 - A Plaid Puzzle   | 24.12s   | 11.48s          | 중간에 GF(256) 추출용으로 사용된 코드                        |
| Hack.lu CTF 2019 - VsiMple       | 0.20s    | 0.02s           |                                                              |
| Google CTF 2019 Quals - minetest | 6.70s    | 11.72s          | Mincraft-like한 오픈소스 게임에서 구현된 논리 회로를 푸는 문제 |

의외로 Boolector가 꼭 빠르지는 않다는 것을 알 수 있었습니다. minetest의 경우 마인크래프트 논리회로를 푸는 문제인데, 단순한 형태일 경우 z3가 좋은 퍼포먼스를 보여주는 것일까요? 확실하지가 않습니다 :(

# 결론

의도도 좋았고 잘 진행했는데, 마지막의 마지막에 생각보다 갖고 있는 z3py 코드가 없어서 비교하는 데 어려움이 있었던 것이 아쉬운 점입니다. 가능하다면 github에 public repository로 만들고 다른 사람의 solver까지 들고 와서 비교해보는 과정을 거친다면 더 좋은 결과를 얻을 수 있을 것으로 생각됩니다.

또한 SMT-LIBv2 형태로 추출할 수 있게 되었기 때문에, SMT-COMP에서 찾아볼 수 있는 다른 SMT Solver에 대해서도 비교하는 과정을 거친다면 더 좋은 결과물을 얻을 수 있을 것으로 생각됩니다.

이와 별개로, 앞서 올렸던 Boolector / Yices / z3 벤치마크를 참고해본다면 거의 모든 경우에 있어서 Boolector가 좋은 성능을 보일 것으로 생각했는데, minetest solver의 경우 정반대의 결과가 나와서 다소 당황하기도 했습니다. CTF에서 가능하면 z3와 Boolector 양쪽에 대해서 실행하는 방향으로 가면 좋겠지만, 시간과 자원이 한정적이기 때문에 문제에 따라 작은 subproblem으로 나눌 수 있는 경우 subproblem에 대해서 시간을 비교해본다던지, z3가 너무 오래 걸려서 답이 나오지 않는 것이 거의 확실할 때 Boolector를 써본다던지 하는 방향으로 가면 좋을 것으로 생각됩니다.

마지막으로, 어떻게 하면 다른 SMT Solver에 대해서 유연하게 사용할 수 있을지 알아볼 수 있어서 좋은 경험이었다고 생각합니다. CTF에 자주 참여하는 다른 분들도 이를 참고해서 진행할 수 있으면 좋겠습니다.

# 참고 문헌

1. z3py guide https://ericpony.github.io/z3py-tutorial/guide-examples.htm
2. z3 https://github.com/Z3Prover/z3
3. SMT-COMP https://smt-comp.github.io/introduction.html
4. Boolector https://boolector.github.io/
5. smt - How to analyse z3 performance issues? https://stackoverflow.com/questions/42371139/how-to-analyse-z3-performance-issues
6. http://scratch.clifford.at/compact_smt2_enc_r1102.html
7. SMT-LIB www.smtlib.org
8. z3 - SMT2 format logfile https://stackoverflow.com/questions/60698770/smt2-format-logfile