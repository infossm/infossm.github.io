---
layout: post
title: "Barren Plateaus"
date: 2023-09-26
author: red1108
tags: [quantum, quantum-machine-learning]
---


## 서론

앞으로 당분간의 양자컴퓨터 시대를 **NISQ** area라고 부르는데, 이 뜻은 적당한 큐빗 수(~1000개)이면서, 에러를 제거할 수 없는 양자컴퓨터를 말한다. 큐빗 수에 제한을 둔 이유는 큐빗 수가 엄청나게 많다면 이들을 사용해서 에러가 없는 큐빗을 만들 수 있기 때문이다. 결국 NISQ를 한 문장으로 요약하자면, 지금처럼 에러가 무시할 수 없는 수준인 적당한 큐빗을 가진 양자컴퓨터라고 할 수 있다. 

IBM사는 433큐빗 양자컴퓨터를 개발하였고, IONQ사는 32큐빗 양자컴퓨터를 공개하였다. 두 회사의 큐빗 수가 10배 이상 차이나지만, 현재의 양자컴퓨터는 에러율이 중요하기 때문에 단순 비교는 불가능하다. 이 두 컴퓨터 모두 NISQ 시대의 대표적인 컴퓨터이다.

NISQ area에서 가장 유망한 분야는 **Quantum Machine Learning(QML)** 분야이다. QML하면 대표적인 것은 고전적으로 파라미터를 저장하고, 이 파라미터를 사용해 양자 회로를 조작하는 **Parameterized Quantum Circuit**이다. 보통 이 개념을 **Quantum Neural Network(QNN)** 이라고 부르기도 하는데, 고전적인 NN의 그것과 구조는 다르지만, 뭐 대충 목적과 방식이 비슷하니 Neural Network라고 부르는 듯 하다.

그런데 QNN에는 **Barren Plateaus** 라는 큰 한계점이 존재한다. 한국어로 번역하면 "불모의 고원" 이다. 이 개념은 양자 인공신경망에서 쌓을 수 있는 레이어와 사용할 수 있는 큐빗 개수에 한계가 존재한다는걸 시사한다. 큐빗수도 늘리고, 레이어도 많이 쌓아야 양자컴 쓰는 이유가 있는데, 그걸 못 한다니.. 아주 암울한 이야기이다. 지금부턴 Barren Plateaus가 무엇이며 왜 발생하는지를 소개하고자 한다.

## Barren Plateaus란?

Barren Plateaus는 양자 인공신경망에서 파라미터의 gradient가 지수적으로 감소해서 종국엔 학습이 불가능해지는 현상이다.

Barren Plateaus는 인공신경망에서의 **Gradient Vanishing**과 유사한 문제이다. Gradient vanishing은 활성화 함수를 ReLU로 바꾸거나, Skip connection을 사용하거나, 기타 등등 많은 해결법이 나오면서 요즘에는 크게 문제가 되지 않는 수준이다.

하지만 Barren Plateaus는 **일반적인** 양자 인공신경망이라면 피해갈 수 없는 것처럼 보인다. 그 이유는 양자 정보에서 사용되는 개념인 **측도 집약화** 현상 때문이다.

### 측도 집약화란?



### 참고문헌

[1] McClean, Jarrod R., et al. "Barren plateaus in quantum neural network training landscapes." Nature communications 9.1 (2018): 4812.
[2] https://www.tensorflow.org/quantum/tutorials/barren_plateaus?hl=ko