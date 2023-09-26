---
layout: post
title: "Barren Plateaus"
date: 2023-09-26
author: red1108
tags: [quantum, quantum-machine-learning]
---

이 글에서는 현재 양자 인공신경망이 직면한 가장 큰 문제인 **Barren Plateaus**에 대해 다룬다. 이 현상은 큐비트가 늘어나고, 회로의 깊이가 깊어질수록 gradient가 사라져서 학습이 불가능해지는 현상을 말한다.

본 글에서는 이 현상을 소개하고, 수학적으로 기술하고자 한다. 이때 논문 [2]의 내용을 참고하였다. 글의 말미에는 코드를 통해 이 현상이 실재함을 확인한다.

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

![측도 집약화](/assets/images/barren-plateaus/concentration.png)
<center><b>그림 1. 양자 상태 관측값의 측도 집약화 현상</b></center><br/>

위 그림 1은 측도 집약화를 간략하게 보여주고 있다. 임의의 양자 상태는 행렬로 표현되고, 고윳값이 상태별로 확률값을 가지기 때문에 행렬의 trace값이 1을 만족한다. 따라서 양자 상태는 Hypersphere에 표현된다.

랜덤하게 양자 상태를 생성하면 그 값은 Hypersphere상의 적도에 존재할 확률이 아주 아주 크다(Levy's lemma). 따라서 Hypersphere상에서 관측값을 구하면, 대부분의 값이 적도에 몰리는 현상이 발생하는데, 이 현상이 측도 집약화이다. 

큐비트가 늘어나면서 hypersphere의 차원이 커지면 적도로부터 특정 각도 이상 차이날 확률이 지수적으로 작아진다. 결국 모든 양자 인공신경망은 관측값(Observable) 기반으로 gradient를 계산하는데, 관측값이 적도에서 유의미하게 멀어질 확률이 사라지면 gradient의 크기도 줄어든다. 앞으로 Barren Plateaus에 대해서 증명할 모든 내용은 사실상 이 직관으로부터 이해 가능하다.

## 실습

간단한 파이썬 코드를 통해 Barren Plateau가 실재함을 확인해 보자.

```python
print("hello world")
```

### 참고문헌

[1] McClean, Jarrod R., et al. "Barren plateaus in quantum neural network training landscapes." Nature communications 9.1 (2018): 4812.
[2] https://www.tensorflow.org/quantum/tutorials/barren_plateaus?hl=ko