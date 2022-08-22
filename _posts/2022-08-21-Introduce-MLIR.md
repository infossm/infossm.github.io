---
layout: post
title:  "MLIR 소개"
date:   2022-08-21-18:00
author: cs71107
tags: [compiler, MLIR]
---

## 들어가기 전에 ##

MLIR, 즉 Multi-Level Intermediate Representation은 하드웨어에 따른 연산들을 지원하는, 중간 언어(Intermediate Representation)에 대한 library라고 보면 됩니다. 아마 꽤 생소할 것 같습니다. 이 글에서는 MLIR에 대한 배경, 구성 요소 등에 대해 간단히 소개하고자 합니다.

## BackGround ##

본격적으로 MLIR에 대해 설명을 하기 전에, 이 글을 이해하는 데 있어서 알고 있으면 편한 개념들에 대해서 먼저 설명하도록 하겠습니다.

### IR ###

위에서도 언급한 IR은 Intermidate Representation의 약자로, 중간언어 집합을 말합니다. 어떤 source code가 있다고 할 때, compiler는 그것을 target machine에 맞는 target code로 전환시켜주게 됩니다. 이때 compiler는 source code가 어떤 코드인지 이해하고 그것을 저장해놓습니다. (Front-end) 그리고 그 저장된 것을 해석하여, target code로 전환시킵니다. (Back-end)이때, source code와 target code 사이에서 source code가 어떤 의미인지 저장해 놓은 것이 IR입니다.
IR은 몇 가지 성질을 가집니다. 우선 source language들과 target machine에 대해 독립적입니다. source language로 작성된 code의 기능을 정확히 저장하고 있어야 하므로, semantic analysis를 할 때 편리해야 합니다. 나중에 target machine에 대한 code로 다시 변환해야 하므로, assembly처럼 target machine code로 변환이 쉬워야 합니다.

얼핏 생각했을 때는 중간언어를 따로 저장해 두는 것이 비효율적으로 보일 수도 있습니다. 실제로, IR을 저장할 필요 없이 바로 target machine에 대한 code를 생성할 수도 있습니다. 그럼 왜 굳이, IR code를 생성하는 것일까요?
다음과 같은 상황을 생각해봅시다. $n$개의 source language가 있고, $m$개의 target machine이 있다고 합시다. 만약 IR이라는 중간 단계 없이 source에서 바로 target machine에 대한 code로 변환시킨다고 하면, 총 $n \times m$개의 compiler가 필요할 것입니다. 그리고 각각에 필요한 최적화 역시 전부 다 다를 것입니다.
하지만 IR을 도입한다면 상황이 달라집니다. 이제 각 $n$개의 language에 대해 IR로 변환시키는 compiler가 $n$개 필요할 것이고, $m$개의 target machine에 대해서, IR을 target machine에 맞게 최적화하는 compiler가 $m$개 필요합니다. source language와 target machine에 대해서 적당히 결합해서 사용하면 되므로, 총 $n+m$개의 compiler만으로 변환이 가능해집니다.
또, IR의 경우 source language, target machine과 모두 독립적으로 구성됩니다. 다음과 같은 상황을 생각해 봅시다. IR을 특정 machine A에 대한 machine language로 compile하는 compiler를 생각합시다. 만약 그 compiler에 대한 성능 개선을 해서, 1%의 성능 개선을 해냈다고 하면, 어떤 source language를 사용하든지, target machine이 A라고 하면 그 1%의 성능을 개선한 혜택을 모두 누릴 수 있을 것입니다. IR을 사용하지 않는다면, 특정 source language에서만 혜택을 누릴 수 있을 것입니다.

위와 같은 장점 때문에, IR은 다양한 언어에서 채택되어 사용되고 있습니다. 몇 가지 예시를 들면 다음과 같습니다.
먼저 java에서 쓰는 JVM이 대표적입니다. JVM에서 쓰는 Java bitecode가 IR의 일종입니다. 사실 JVM처럼 virtual machine이나 p-code machine을 target으로 하는 언어들은 IR을 사용하고 있다고 봐도 무방합니다.
그리고 다음에 설명할 LLVM이 있습니다. LLVM은 C/C++ code들을 LLVM IR로 변환시킨 후, 최적화를 거쳐 machine launguage로 변환시킵니다. 그외에도 Cython 등 다양한 언어에서 IR을 사용합니다.

### LLVM ###

다음으로 소개할 것은 LLVM입니다. 역시 생소하게 느끼실 분들이 많을 것 같습니다. 하지만, PS에서 많은 분들이 사용하시는 clang부터 llvm과 관련돼있는 project 중 하나입니다.

LLVM이란 compiler와 toolchain들의 집합이라고 생각하시면 됩니다. C++로 작성되었으며, compile time, link time, run time에서 최적화를 할 수 있게 해줍니다. LLVM은 위에서 설명한 IR이나 machine code를 생성할 수 있습니다.
LLVM은 IR을 생성하고, 그 IR을 최적화하는 LLVM optimizer를 가지고 있습니다. 그래서 LLVM에서는 source language를 target machine code로 만들 때 Front-end, Middle-end, Back-end이 세 단계를 거칩니다. Front-end와 Back-end는 IR에서 설명한 것과 같고, Middle-end는 주어진 IR을 LLVM에 있는 각종 pass들을 사용하여 최적화시킵니다.

LLVM에서는 LLVM-IR이라는 IR을 사용합니다. LLVM-IR은 IR을 설명할 때 설명한 IR의 특성들을 가집니다. 거기에 더해서, 모든 변수들이 SSA form이라는 특징을 가집니다. SSA form (static-single assignment form)이란, 각 변수의 값이 정확히 한번만 assign 된 형태를 말합니다. 다시 말해, 초기값이 한번 정해지면 앞으로 바뀌지 않습니다.
LLVM에서 LLVM-IR을 생성한 후, LLVM에서는 pass들을 통해 최적화를 적용할 수 있습니다. pass들의 목록은 [링크](https://llvm.org/docs/Passes.html)에서 확인 가능합니다. LLVM에서 pass는 IR code를 최적화할 때 사용합니다. 예를 들어, -dce option의 경우, 코드는 있지만 실제로 사용되지는 않는 code들을 제거합니다. pass는 최적화 외에 IR code들을 분석하는 경우에도 사용할 수 있습니다. -print-dom-info option의 경우 dominator에 관련된 정보를 출력합니다. 이 외에도 다양한 option이 있지만, 넘어가도록 하겠습니다.

다양한 언어에서 LLVM을 사용하고 있는데, PS를 하는 분들에게도 익숙하실 clang의 경우 Back-end가 LLVM입니다. 이 외에도 다양한 C++ code들은 물론이고, CUDA나 Rust 등 다양한 언어에 대해서 적용 가능합니다.

## What is MLIR? ##

이제 IR과 LLVM에 대해서 설명을 했으니, 본격적으로 MLIR에 대한 설명을 시작하도록 하겠습니다.

### MLIR 탄생 배경 ###

MLIR의 탄생 배경부터 설명을 해보겠습니다. 하나의 machine learning framework의 경우 다양한 compiler들을 가지며, 이 compiler들은 모두 다른 domain에서 작업을 하게 됩니다. 또, 각 언어마다 고유의 high-level IR을 사용하고 있다보니 개발에 있어서 많은 불편함을 만들었습니다. 이런 software fragmentation을 해결하기 위해, 여러 개의 domain에 대해서 공통적으로 적용가능한 IR을 만든 것이 MLIR입니다.

### MLIR의 요소 ###

MLIR은 다음과 같은 요소들로 구성됩니다.

- Operation
    - 흔히 생각하는 instruction, function, module이 모두 operation에 해당합니다.
    - operation들은 $0$개 이상의 Value를 결과로 생성할 수 있습니다. 그리고 operand와 attribute가 존재합니다.
    - 모두 SSA form을 만족해야합니다.
- Attribute
    - attribute는 key-value로 구성된 dictionary 형태로 존재합니다.
    - 각 attribute의 value는 string, integer 등의 type을 가질 수 있습니다.
- Location information
    - operation의 위치에 대한 정보를 저장합니다.
- Regions and Blocks
    - region과 Block은 코드의 구조에 대한 정보와 관련이 있습니다.
    - 하나의 region은 여러 개의 Block으로 구성됩니다.
    - 그리고 하나의 Block은 operation의 list로 구성됩니다. (다른 region을 포함하는 것도 가능합니다.)
    - 각 block은 terminator operation을 가집니다. (return 등)
- Value dominance and visibility
    - 각 Operation은 visible한 Value들만을 사용할 수 있습니다. 즉, control flow 상에서 반드시 이전에 정의돼있는 값만 사용할 수 있습니다.
    - 위의 정보는 dominance와 관련이 있으며, 어떤 value의 dominance 정보는 control flow에 대한 domiate tree를 통해 얻을 수 있습니다.
- Symbols and symbol table
    - C/C++ 처럼 symbol table을 따로 줄 수 있습니다.
    - symbol table의 symbol은 반드시 SSA form일 필요가 없습니다.
    - global variable들이나, 재귀함수 같이 SSA form으로 정의하기 힘든 것들을 symbol table을 사용해 symbol을 사용합니다.
- Dialects
    - Dialects는 Operation들의 group입니다.
    - 모든 Operation들을 한번에 하나의 Dialect에 넣는 것도 가능하지만, 실제로 그럴 경우 management 등에서 문제가 생길 수 있어 실제로는 하나의 Operation은 하나의 Dialect에 속하게 됩니다.
    - MLIR의 Dialect들은 한 code에 여러 개가 공존할 수 있습니다.
    - MLIR의 Dialect의 종류의 경우 나중에 자세히 설명하도록 하겠습니다.
- Type System
    - MLIR의 각 Value들은 type을 가집니다.
    - MLIR의 type system의 경우 user-extensible하며, llvm::Type이나 clang::Type같이 정의돼있는 걸 가져올 수도 있습니다.
    - MLIR에서는 strict하게 type checking을 하며, conversion rule을 가지고 있지 않습니다.
- Standard type
    - MLIR에서는 정수, 실수, tuple, tensor, multi-dimensional vector 등을 standard type으로 제공합니다.
- Functions and modules
    - MLIR은 function, 그리고 module로 구성됩니다.
    - function과 module 모두 하나의 region을 가지지만, function의 경우 terminator op로 return을 사용하고, module의 경우에는 control flow에 영향을 주지 않는 dummy operation을 사용합니다.

## MLIR Dialects ##

MLIR에서 Dialect들은 Operation들의 부분집합입니다. 한 Dialect내의 Operation들은 유사한 기능과 성질을 가집니다. 그렇기 때문에 각 Dialect들의 성질을 이해하면 MLIR을 보다 더 잘 이해할 수 있습니다. MLIR에서는 다양한 Dialect들이 존재하고, Dialect를 직접 정의하고 해당 연산을 구현하는 것도 가능합니다. (이 경우 compile 과정에서 좀 더 lower한 영역의 Dialect들로 구성된 연산으로 Conversion이 일어나게 됩니다.)

여기에서는 MLIR에서 자주 사용하는 Dialect들 중 몇개에 대해 간략히 설명하도록 하겠습니다. 더 많은 Dialect들의 목록은 [링크](https://mlir.llvm.org/docs/Dialects/)에 있습니다.

### Arith Dialect ###

이름 처럼 arithmetic한 연산들을 다룹니다. 정수, 실수들의 덧셈, 곱셈, 그리고 각종 비트 연산들을 포함해 여러 가지 연산들을 다룹니다. scalar 값뿐만 아니라 tensor와 vector도 operand가 될 수 있습니다. 그리고 MLIR에서는 index가 다른 일반적인 정수형과 혼용되는 것이 아니라, index라는 type으로 따로 존재합니다. 그렇기 때문에, Arith Dialect를 다룰 때에는 index와 다른 integer type을 혼용하고 있지 않은지 주의해야 합니다.

### SCF Dialect ###

if-else, for, while 처럼 flow에 관련된 Dialect입니다. SCF는 Structured Control Flow의 줄임말입니다. scf.if, scf.for, scf.while등 C/C++에서 if-else, for, while에 대응되는 연산들이 존재하며, scf.yield라는 terminator도 존재합니다.

### Memref Dialect ###

memory에 관련된 Dialect입니다. Memref Dialect에서는 memref type의 변수를 다룹니다. memref type의 변수의 경우 C/C++에서 사용하는 pointer 처럼 정보들이 저장된 위치를 저장하나, C/C++에서의 pointer와 달리 type casting에 있어서 더 까다로운 제한이 있습니다. Memref Dialect에서는 memref.alloc, memref.memcpy 등 C/C++에서도 볼 수 있는 memref type 변수에 관한 연산을 지원합니다.

### 기타 ###

위에서 언급한 것 외에도, gpu와 관련된 연산을 지원하는 GPU Dialect, tensor들과 관련된 Tensor Dialect, 수학 관련 함수들을 지원하는 Math Dialect 등이 존재합니다.

## Pass of MLIR ##

앞서 언급한 것처럼, MLIR에서 사용하는 LLVM의 경우 Middle-end가 존재합니다. 따라서, MLIR에서도 IR 단계에서 최적화가 가능할 것이라 생각할 수 있습니다.
실제로 MLIR에서도 LLVM에서처럼 Pass들이 존재하여 Lowering 및 최적화를 지원합니다. Lowering이란, 한 연산을 High-level의 Dialect에서 동일한 기능을 하는 Lower-level Dialect에 속하는 연산들로 옮기는 것을 의미합니다. 만약에 Dialect를 새로 만들었다면, 이런 Lowering 과정이 사실상 필수적일 것입니다.

MLIR에서 제공하는 Pass들의 목록은 [링크](https://mlir.llvm.org/docs/Passes/)에서 확인 가능합니다. canonicalize pass의 경우 기존 code를 canonicalize 해줍니다. (dead code elimination 등) 이 외에도 다양한 Pass들을 제공하니 직접 확인해보시기 바랍니다.
실제로는 MLIR에서 제공하는 Pass들 중에 원하는 기능이 없는 경우가 많기 때문에, 직접 Pass를 생성해서 사용하는 경우가 다수입니다. 이 방법까지 쓰려면 너무 길어지기 때문에, 여기서는 넘어가도록 하겠습니다.

## 마치며 ##

지금까지 MLIR에 대해 간단한 소개를 했습니다. MLIR은 compiler를 하시지 않는 분이라면 생소한 개념일 것입니다. 사실 MLIR은 훨씬 더 설명할 내용이 많지만, 여기에서는 제가 익숙하고, 중요하다고 생각하는 내용만 간략하게 담았습니다.
MLIR에 대한 더 많은 정보가 궁금하시다면 [링크](https://mlir.llvm.org/)를 참고하시기 바랍니다. 감사합니다!

## Reference ##

1. Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas, Vasilache, Oleksandr Zinenko, MLIR: A Compiler Infrastructure for the End of Moore’s Law

2. Princeton COS 320: Compilers, 2003년 2월 4일 수정, 2022년 8월 21일 접속, https://www.cs.princeton.edu/courses/archive/spr03/cs320/notes/IR-trans1.pdf

3. Wikipedia, 2022년 7월 24일 수정, 2022년 8월 21일 접속, https://en.wikipedia.org/wiki/LLVM
