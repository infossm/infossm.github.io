---
layout: post
title:  "MLIR 소개"
date:   2022-08-21-18:00
author: cs71107
tags: [optimization, MLIR]
---

## 들어가기 전에 ##

MLIR, 즉 Multi-Level Intermediate Representation은 하드웨어에 따른 연산들을 지원하는, 중간 언어(Intermediate Representation)에 대한 library라고 보면 됩니다. 아마 꽤 생소할 것 같습니다. 이 글에서는 MLIR에 대한 배경, 구성 요소 등에 대해 소개하고자 합니다.

## BackGround ##

본격적으로 MLIR에 대해 설명을 하기 전에, 이 글을 이해하는 데 있어서 알고 있으면 편한 개념들에 대해서 먼저 설명하도록 하겠습니다.

### IR ###

위에서도 언급한 IR은 Intermidate Representation의 약자로, 중간언어 집합을 말합니다. 어떤 source code가 있다고 할 때, compiler는 그것을 target machine에 맞는 target code로 전환시켜주게 됩니다. 이때 compiler는 source code가 어떤 코드인지 이해하고 그것을 저장해놓습니다. 그리고 그 저장된 것을 해석하여, target code로 전환시킵니다. 이때, source code와 target code 사이에서 source code가 어떤 의미인지 저장해 놓은 것이 IR입니다.
IR은 몇 가지 성질을 가집니다. 우선 source language들과 target machine에 대해 독립적입니다. source language로 작성된 code의 기능을 정확히 저장하고 있어야 하므로, semantic analysis를 할 때 편리해야 합니다. 나중에 target machine에 대한 code로 다시 translate해야 하므로, assembly처럼 target machine code로 translate이 쉬워야 합니다.
위의 특징과 더불어, IR은 machine language와 비교해서 몇 가지 차이점을 보입니다. 첫째로 하나의 instruction이 정확히 하나의 fundamental한 연산을 나타냅니다. 둘째로 instruction set에 control flow 관련 정보가 포함되지 않을 수 있습니다. 마지막으로, register의 개수가 제한이 없습니다. 실제 machin들은 register개수에 제한이 있지만, IR에서는 일단 그런 제한을 생각하지 않습니다.

얼핏 생각했을 때는 중간언어를 따로 저장해 두는 것이 비효율적으로 보일 수도 있습니다. 실제로, IR을 저장할 필요 없이 바로 target machine에 대한 code를 생성할 수도 있습니다. 그럼 왜 굳이, IR code를 생성하는 것일까요?
다음과 같은 상황을 생각해봅시다. $n$개의 source language가 있고, $m$개의 target machine이 있다고 합시다. 만약 IR이라는 중간 단계 없이 source에서 바로 target machine에 대한 code로 변환시킨다고 하면, 총 $n \times m$개의 compiler가 필요할 것입니다. 그리고 각각에 필요한 최적화 역시 전부 다 다를 것입니다.
하지만 IR을 도입한다면 상황이 달라집니다. 이제 각 $n$개의 language에 대해 IR로 변환시키는 compiler가 $n$개 필요할 것이고, $m$개의 target machine에 대해서, IR을 target machine에 맞게 최적화하는 compiler가 $m$개 필요합니다. source language와 target machine에 대해서 적당히 결합해서 사용하면 되므로, 총 $n+m$개의 compiler만으로 변환이 가능해집니다.

이렇게 source language, IR, target machine code를 분리했을 때 얻는 이점은 위에 서술한 것이 전부가 아닙니다. 앞서 말했듯 IR의 경우 source language, target machine과 모두 독립적으로 구성됩니다. 다음과 같은 상황을 생각해 봅시다. IR을 특정 machine A에 대한 machine language로 compile하는 compiler를 생각합시다. 만약 그 compiler에 대한 성능 개선을 해서, 1%의 성능 개선을 해냈다고 하면, 어떤 source language를 사용하든지, target machine이 A라고 하면 그 1%의 성능을 개선한 혜택을 모두 누릴 수 있을 것입니다. IR을 사용하지 않는다면, 특정 source language에서만 혜택을 누릴 수 있을 것입니다.

위와 같은 장점 때문에, IR은 다양한 언어에서 채택되어 사용되고 있습니다. 몇 가지 예시를 들면 다음과 같습니다.
먼저 java에서 쓰는 JVM이 대표적입니다. JVM에서 쓰는 Java bitecode가 IR의 일종입니다. 사실 JVM처럼 virtual machine이나 p-code machine을 target으로 하는 언어들은 IR을 사용하고 있다고 봐도 무방합니다.
그리고 다음에 설명할 LLVM이 있습니다. LLVM은 C/C++ code들을 LLVM IR로 변환시킨 후, 최적화를 거쳐 machine launguage로 변환시킵니다. 그외에도 Cython 등 다양한 언어에서 IR을 사용합니다.

### LLVM ###



## What is MLIR? ##


## MLIR Dialects ##



## Pass of MLIR ##



## Use of MLIR ##


## Reference ##

