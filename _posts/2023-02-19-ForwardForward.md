---
layout: post
title:  "Forward Forward algorithm"
date:   2023-02-19
author: red1108
tags: [AI, Deep Learning, Machine Learning]
---
## 서론

인공지능의 대부 Hinton교수가 새로운 신경망 학습 방법을 고안했다. 2022년 12월 27일에 아카이브에 올라온 논문이라 글 작성 시점에서 아직 따끈따끈한 논문이다. 인공지능의 대부가 쓴 논문인 만큼 벌써 조금씩 인용이 되고 있다. 과연 Hinton교수가 이번에 발명한 학습방법은 어떤 것인지 살펴보도록 하자.

> 본 글은 Hinton교수님의 Forward Forward algorithm 논문[1] 중에서도 Forward Forward의 작동 원리만을 이해하는 것에 초점을 두고 정리하였다. 논문 전체 내용을 정리하는 목적이 아니므로 빠진 내용들이 있다. FF에 대해 더 궁금하다면 논문 원문을 살펴보는 것을 강력하게 추천한다. FF에 대한 Hinton교수님의 다양한 인사이트를 엿볼 수 있을 것이다.

## 왜 새로운 학습 방법을 찾아야 하는가

그동안 안정적으로 사용되어진 딥 러닝의 주된 학습 방식은 오류 역전파 알고리즘(Back Propagation, BP)방식이다. 하지만 BP방식도 잘 알려진 어려 문제점을 가지고 있다

1. 기울기 소실 문제

2. 과적합 문제

3. 로컬 미니멈 문제

4. 계산 비용 문제

상술한 문제들은 많은 연구들을 통해 해결법이 잘 제시되어 있는 편이지만, 좀 더 근본적인 문제점이 남아있다. 인공지능의 궁극적인 목표는 인간의 뇌와 유사한, 인간 수준 또는 그 이상의 문제해결 능력을 모방하는 것이다. 그런데 현재까지 알려진 바로는 우리 뇌는 오류 역전파 알고리즘으로 학습하지 않는 것 처럼 보인다. 왜냐하면 오류 역전파 알고리즘을 구현하기 위해서는 각 계층에서 일어나는 모든 계산과정의 gradient를 기록해야 하는데 우리 뇌에서는 그 정보를 저장하는 기관이 관찰되지 않았기 때문이다.

사실 현재의 딥러닝 방법론이 실제 뇌와 다른 점은 너무 많아서 나열하기도 힘든 수준이다. 뇌는 loss function을 정의하지도 않고 각각의 뉴런에서 계산이 분산되어 자연스러운 상호작용만으로 학습을 이뤄낸다. 그런데도 희소한 데이터만을 가지고 뛰어난 성능을 낸다. 에너지 소모율로 따진 효율성도 컴퓨터보다 훨씬 앞선다.

> 물론 인간 수준의 인지능력과 문제해결 능력을 갖춘 인공지능을 만들기 위해서 꼭 뇌의 기능을 완벽하게 규명할 필요는 없다. 이는 마치 새와 비행기의 관계와 같다. 비행기의 형태는 새의 날개를 모방하여 만들어졌지만, 날개의 퍼덕임이 아니라 엔진의 추력으로 비행한다. 원하는 목표를 달성하기 위해 어느 정도의 모방은 좋지만, 대상에 대한 완벽한 규명이 필수는 아니라는 점을 짚고 넘어가고자 한다.

이러한 문제점들로 인해 BP를 대체할만한 다양한 방법들이 연구되었지만, 아직까지 BP만한 방법은 발견되지 않았다고 보아도 무방하다. 그리고 이번에 나온 Hinton교수의 Forward Forward논문도 같은 결의 연구이지만, Forward Forward 역시 Backpropagation보다 나을지는 아직 모르는 일이다. 다만 Hinton교수의 연구라 관심을 많이 받는 만큼 발전시켜보려는 시도 또한 많이 이뤄질 것이고 쓸모 있는지 없는지 비교적 빨리 평가될 것이라 생각한다.

서론이 길었으니 이제 Forward Forward Algorithm에 대해 소개해 보겠다.

# Forward Forward Algorithm

## 간략한 소개

Forward Forward Algorithm(FF)는 Hinton교수가 제안한 다층 신경망을 학습하는 새로운 방법이다. 퍼셉트론을 기반으로 한 다층 퍼셉트론의 구조는 우리가 아는 그대로이다. 학습 방법만 다른 것이다.

오차 역전파 알고리즘(Backpropagation, BP)는 신경망을 순방향으로 훑으며 계산과정을 기록해나가고 뒤에서부터 시작해서 미분계수를 구하는 두 가지 Pass로 구성되어 있다. 이를 Forward pass & Backward pass로 부르자.

새롭게 제안된 방식인 Forward Forward알고리즘은 두 번의 Forward pass로 구성되어 있다. 그래서 알고리즘 이름이 Forward Forward인 것이다. 첫 번째 Forward pass는 positive data에서의 신경망의 **활성도**를 키우고 두 번째 Forward pass는 negative data에서 신경망의 **활성도**를 낮추는 역할이다.

> 우리 뇌에서 뉴런의 흥분은 전기적인 신호로 표현된다. 뉴런이 항상 흥분하는 것은 아니며 가지돌기로부터 받은 신호로부터 흥분의 여부가 결정된다. 어떨 때에 얼마나 흥분할지 결정하는것이 중요한 것이다.

여기서 말하는 신경망의 **활성도**는 논문에서 후술할 goodness function과 관련된다.

MNIST의 경우로 예시를 들어보자. 숫자 5가 5라고 제시하는 data는 positive data이다. 그리고 이 경우에는 신경망의 활성도가 높아지는 방향으로 학습이 진행된다. 숫자 7이 2라고 제시하는 data는 negative data이다. 이 경우엔 신경망의 활성도가 낮아지는 방향으로 학습이 진행된다. 모든 학습을 거친 다음 숫자 8사진을 이용해 inference를 진행해 보자. 만약 학습이 올바르게 진행되었다면 숫자 8이 0이라고 하는 경우부터 숫자 8이 9라고 하는 총 10가지의 경우들 중에서 제일 신경망의 활성도가 높아지는 경우는 숫자 8이 8이라고 말하는 경우일 것이다. 따라서 FF방식으로 학습된 모델은 숫자 8사진을 보고 이 숫자가 8임을 알아낼 수 있다.

아직은 간단한 문제에만 적용 가능하며, 논문에서도 MNIST와 CIFAR-10의 경우만 다루었다. 그리고 error rate는 CNN기반 오차 역전파 알고리즘과 비교해 보았을때 비슷하거나 살짝 높았다. 처음 제시된 개념인데도 거의 비슷한 정도의 error rate를 보인다는 점에서 발전 가능성이 높다고 판단된다.

이제 FF를 좀 더 자세히 알아보도록 하자.

## Positive & Negative Data

FF에서 사용하는 데이터는 두 가지 종류로 구분된다.

Positive data는 *올바른 데이터*라고 생각하는게 이해하는데 편하다. MNIST의 경우엔 사진과 라벨이 일치하는 데이터 쌍에 해당한다.

Negative data는 위와 대조적으로 *잘못된 데이터*라고 생각하면 된다. MNIST에서는 사진과 라벨이 일치하지 않는 데이터 쌍에 해당한다.

Positive data는 쉽게 제작할 수 있지만 Negative data를 만드는 것은 고민해보아야 할 문제이다. 논문에서는 MNIST의 경우에 Unsupervised FF와 Supervised FF 각각의 경우에 대하여 Negative data를 만드는 방법을 소개한다. 

### Negative Data in Unsupervised FF

첫 번째 방식은 원래 이미지로부터 변형해서 만드는 방식이다.

아무런 연관 없는 노이즈 이미지를 사용하면 Negative learning의 효과가 없다. 논문에서는 이를 "very different long range correlations but very similar short range correlations" 라고 표현한다. 

<br/>

![Hybrid image](https://user-images.githubusercontent.com/17401630/219955858-be190046-ac32-437d-a98b-5c457e5a3a22.png)
<center><b>그림 1: Negative data에 사용된 하이브리드 이미지</b></center>

<br/>

손글씨 모양으로는 보이면서(short range correlations) 실제 숫자는 아니어야(different long range correlations) 좋은 Negative data라고 볼 수 있다. 이는 Fig 1에서 만든 hybrid 이미지에서 확인할 수 있다. 논문에서는 두 손글씨 사진을 적당한 가중치를 두고 마스킹하여 더하는 기법으로 해당 데이터를 만들었다.

Contrastive Learning기법에서는 먼저 인공신경망을 이용해 입력 데이터로부터 representation vector을 생성한다. 이후 변환된 벡터를  linear classifier에 입력하여 각 라벨별로 확률분포를 얻어낸다.

Unsupervised FF의 경우, representation vector을 뽑아내는 인공신경망을 Forward Forward로 학습할 수 있다. 그리고 Contrastive Learning에 필요한 Negative data를 Fig1과 같은 방식으로 만든다.

하지만 조사 과정에서 위의 방법을 사용하는 구현된 코드를 찾기가 어려웠다. 대부분의 공개된 구현체들은 아래에 설명할 Supervised FF버전이다. 개인적으로는, Supervised FF가 훨씬 이해하기 쉽고 직관적이라고 생각한다.

### Negative Data in Supervised FF

인터넷에 검색했을 때 나오는 거의 대부분의 구현체는 이 방식으로 구현되었다. MNIST의 케이스를 가지고 설명을 이어가자.

Supervised FF에서는 이미지 안에 라벨 정보를 집어넣는다. MNIST는 손글씨를 제외하고는 나머지 부분은 전부 값이 0이기 때문에 라벨 정보를 one-hot으로 집어넣기가 쉽다. 맨 왼쪽 위의 연속한 10개 픽셀 중에서 라벨 위치에 해당하는 곳을 1로 색칠한다.

<br/>

![one-hot image](https://user-images.githubusercontent.com/17401630/219958020-c3bb1975-431b-415c-8e55-b3886dc14fd2.png)
<center><b>그림 2: 라벨 정보가 삽입된 Negative image</b></center>

<br/>

위 사진을 보면 바로 이해될 것이다. 0부터 9까지의 숫자를 사용하므로 숫자 5인 경우엔 6번째 위치에 표시되어있고, 숫자 8인 경우엔 9번째 위치에 표시되어 있다.

그리고 실제 숫자와 라벨이 동일하게 표시된 경우엔 Positive data로 사용하고, 숫자와 라벨이 다르면 Negative data로 사용한다.

## Goodness Function

Forward Forward Algorithm의 핵심은 Positive data에서는 goodness function의 값을 키우고, Negative data에서는 goodness function의 값을 작게 만드는 것이다.

goodness function를 어떻게 번역할지는 아직 합의되지 않았지만, '적합도', '활성도', '흥분도' 정도로 이해하면 될 것 같다. 앞으로는 그냥 goodness라고 적겠다.

goodness는 각 레이어의 출력 벡터(activity vector)의 **제곱합**으로 정의한다. activity vector의 norm값의 제곱과 동일하다. 그리고 이렇게 계산한 goodness값이 positive data에서는 커지고, negative data에서는 작아지도록 학습한다.

### layer normalization

레이어의 activity vector의 제곱합의 합이 goodness function이라면 그대로 사용했을때 문제가 발생한다. 레이어 A -> 레이어 B로 연결되는 과정을 생각해 보자. 

아무런 변경이 없다면 positive data의 경우 레이어 A의 activity vector의 크기가 커질 것이다. negative data의 경우에는 그 반대일 것이다. 그렇다면 이후의 레이어 B는 앞선 레이어의 activity vector의 크기를 그대로 반영하는 방향으로 무의미한 학습이 진행될 것이다.

따라서 각 레이어가 좀 더 유용한 feature를 학습하게 강제하기 위해서 **다음 레이어의 입력으로 넣어주기 전에 normalization을 진행한다.**

### loss function

FF는 goodness 값이 positive data에서는 threshold $\theta$ 보다 커지고 negative data에서는 $\theta$보다 작아지도록 학습한다. 이를 바탕으로 유도한 loss function은 다음과 같다.

$$
loss = f(\sum_{j}^{}{y_{pos, j}^2 - \theta}) + f(\sum_{j}^{}{\theta-y_{neg, j}^2})
$$

$y_j$는 **layer normalization전의** 레이어의 j번째 퍼셉트론의 활성도이다. f는 logistic function이다. 중요한 점은 loss function이 여러 레이어의 영향을 받아 계산되는게 아니라 각 레이어로만 계산된다는 점이다. 따라서 미분할때도 chain rule을 사용할 필요가 없고 **각 레이어별로 독립적인 학습이 진행된다**. 학습이 레이어별로 독립적으로 진행된다는 점은 정말 큰 장점이다. 

## 학습 진행

제일 단순한 Supervised FF를 기준으로 설명하겠다. 먼저 one-hot 방식으로 라벨 정보가 임베딩 된 데이터셋을 준비한다.

데이터셋은 (Positive data, Negative data)의 쌍으로 구성된다. Positive data는 사진 정보와 임베딩된 라벨의 정보가 일치하고, Negative data는 일치하지 않는다. 이제 Positive data에서는 goodness값이 커지도록 학습하고 Negative data에서는 작아지도록 학습한다. 이를 사진으로 표현하면 아래와 같다.

<br/>

![learning picture](https://user-images.githubusercontent.com/17401630/219960817-f556e2da-67fb-46f5-9022-3c66f4aca94b.png)
<center><b>그림 3: 간단하게 표현한 Supervised FF</b></center>
<center>goodness function의 값을 뇌의 활성에 비유하였다</center>

<br/>

Positive pass와 Negative pass를 진행하면서 각 레이어마다 loss function을 계산하고 학습을 진행하면 된다.

## Sleep

논문에서 정말 흥미로운 관점을 제공해서 이것도 소개하고자 한다.

만약 Positive learning과 Negative learning을 분리해서 학습하는 것이 가능하다고 생각해 보자. 그렇다면 우리 뇌는 낮에는 Positive learning을 진행하고 밤에는 낮의 학습을 기반으로 자는 동안 Negative learning을 진행하는 것은 아닐까?

<br/>

![day and night](https://user-images.githubusercontent.com/17401630/219962701-c60cf19f-10a7-4a58-a223-8dfa16c5470e.png)
<center><b>그림 4: 낮과 밤에서의 Positive & Negative learning</b></center>

<br/>

만약 잠을 자지 않으면 Positive learning만 계속 진행하는 것이다. Negative learning이 결여되므로 불면증이 주는 심각한 영향도 이 관점에서 생각할 수 있다.

물론 이것은 어디까지나 가설이다. 우리 뇌에서 Forward Forward Algorithm과 유사한 방식으로 학습이 일어나는지는 아무도 모르는 일이다.

아무래도 Hinton교수님이 인지심리학자이기도 하여서 이런 흥미로운 관점에서도 많이 생각하신 듯하다. 하지만 논문에서는 Forward pass와 Negative pass를 따로 분리하여 진행하는 경우엔 학습이 제대로 이뤄지지 않았다고 한다. 당연히 다른 방식으로는 가능할 수도 있으며 이것 또한 흥미로운 연구 주제이다.

## GAN과의 관계

Generative Adversarial Networks(GAN)을 어느 정도 아는 사람이라면 FF가 이와 유사함을 느꼈을 것이다. GAN에서 판별자는 fake/real 이미지를 구별하도록 학습한다. FF에서도 positive/negative data를 구별하도록 학습한다.

GAN은 모델 두개를 사용하는 데 반해 FF는 신경망 하나만 사용한다. GAN은 적대하는 신경망 두개를 동시에 학습해야 하기 때문에 학습이 힘들지만, FF는 그러한 문제점에서 자유롭다. 대표적인 예시로 판별자/생성자 중 둘 중 하나가 먼저 수렴해버리는 문제점으로부터 자유롭다.

하지만 FF와 GAN은 완전히 다른 분야이므로 비교할 수 없다. GAN은 생성 AI로써의 기능이 무척 중요하기 때문이다.

## Inference

이제 마지막으로 FF방식으로 학습된 인공신경망이 어떻게 MNIST에 활용되어 숫자를 예측하는지 설명하겠다.

이미지 데이터가 준비되었을 때, 라벨이 0이라 가정하고 one-hot으로 라벨을 삽입한다. 그리고 goodness를 계산한다. 이 방식을 나머지 1~9에도 반복한다. 즉 라벨이 0부터 9인 경우를 하나씩 다 돌려본다. 이 중에서 goodness가 가장 높은 경우를 이미지의 숫자로 추론한다.

논문에서는 이처럼 10번 반복하는 방식 외에도 10개 픽셀 값을 0.1로 채운 다음에 한번만 계산하여 추론이 가능한 방법도 제안한다. 이 방식이 궁금하다면 직접 논문을 읽어 보길 권한다. 다만 이 방식은 좀 더 오차가 커서 현재로써는 10번 반복하는 방식이 유용하다.

평균을 대입하여 추론하는 방식이 오차가 큰 이유를 추정해 보자면 학습에는 one-hot만 사용하는데 반해 추론에는 평균값을 넣었기 때문에 학습/테스트 데이터에 괴리감이 생기기 때문이라 추정된다.

## FF의 장점

Forward Forward Algorithm이라고 해서 Backpropagation알고리즘이 가진 문제점을 다 해결한 것은 아니다. 그래도 의미 있는 해결점이 몇가지 있다.

1. **BP에서 chain rule을 적용하기 위해 계산 과정에서 저장하는 많은 값들이 필요하지 않다.**
    FF는 각 레이어마다 학습을 진행하기 때문에 자연스럽게 방대한 계수를 저장하고 있을 필요가 없다.

2. **모델 중간에 Blackbox가 끼어있어도 여전히 학습이 가능하다.**
    BP는 모든 계산과정을 기록해야 하므로 Blackbox가 끼어있으면 학습이 불가능하다. 그러나 FF는 각 레이어의 학습에는 오로지 그 레이어만 영향을 주므로 Blackbox가 끼어있어도 학습이 가능하다.

## Discussion

개인적으로 FF를 좀 더 복잡한 문제에도 적용하기 위해서는 Negative data를 만드는 **일반적**이고 **확장 가능한**방식이 필요하다. 하지만 이 논문에서는 아직 좋은 방식이 제안되지 않았다. 이미지에 라벨을 삽입하여 정보로 제공하는 방식은 inference과정에서 라벨을 하나씩 다 바꾸어가며 대입해봐야 한다. 좋은 방식이 등장한다면 FF가 획기적으로 다른 복잡한 문제에도 적용될 수 있으리라 생각한다.

그리고 논문에서는 activation vector의 크기로 goodness를 계산했지만, 다른 방식이 좋을 수도 있다.

FF는 각각의 레이어만 학습에 사용하기 때문에 BP와 비교하였을 때 훨씬 사용하는 메모리가 적다. 실제로 둘을 비교한 웹페이지 [링크](https://medium.com/mlearning-ai/pytorch-implementation-of-forward-forward-algorithm-by-geoffrey-hinton-and-analysis-of-performance-7e4f1a26d70f)를 참고하면 좋을 것 같다. 신경망 깊이가 얕을 때는 FF가 메모리가 더 많이 들지만, 신경망이 깊어질수록 FF가 훨씬 더 적은 메모리가 필요하다.

기존에 BP를 대체할 새로운 알고리즘이 많이 제시되었지만 FF는 그나마 BP와 비슷한 정도의 오차율을 보였다. 아직까지는 간단한 문제에만 적용되었지만, 향후 연구를 통해 발전 가능성이 무궁무진한 아이디어라고 생각한다.

## 글을 마치며

16페이지짜리 논문에서 다루지 않는 내용이 꽤 있다. 기억나는 몇가지를 언급해 보자면

1. Recurrent FF

2. Negative data를 신경망으로부터 생성하는 방법

정도가 있다. 이외에도 자잘하게 생략한 내용들이 꽤 된다.

원문이 되는 (Hinton, 2022)[1] 논문에서는 인공지능 대부의 insight를 엿볼 수 있고, 어떤 관점에서 FF를 적용시켜보려고 했는지 훨씬 많은 내용이 담겨 있다. 이 글을 읽고 FF에 흥미가 생겼다면 위 논문을 직접 읽어보길 권한다.

## Reference

[1] Hinton, Geoffrey. "The forward-forward algorithm: Some preliminary investigations." arXiv preprint arXiv:2212.13345 (2022).
