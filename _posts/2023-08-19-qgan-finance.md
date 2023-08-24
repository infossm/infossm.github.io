---
layout: post
title: "QGAN in Finance"
date: 2023-08-19
author: red1108
tags: [quantum, quantum-machine-learning, gan, generative, ai, qgan, qml]
---

> Keywords: `GAN`, `QGAN`, `copula`, `finance data``

[아이온큐 공식 블로그](https://ionq.com/resources/generative-quantum-machine-learning-for-finance)에 소개되어 있기도 하고, 2022년 11월에 실린 논문인데 아직 한국에 소개하는 글이 없어서 블로그 글로 작성한다. ~~arxiv엔 2021년에 올라왔는데 왜이리 늦게 실렸는지 모르겠다. PRX저널에 실으려고 꽤 오래 존버했나..?~~

## 서론

이 논문은 특정한 결합확률 분포를 양자 생성기 (QGAN)으로 모델링하는것이 주제입니다. 꽤 흥미로운 접근법을 많이 보여주고 있어서 소개하고자 하였습니다. 이 논문에서 얻어갈만한 아이디어는 아래와 같습니다.

- 생성 문제를 probability integral transform을 통해 uniform distribution으로 바꾸고, 이걸 생성한 뒤 역변환 하는 법
- 양자 생성기(QGAN)에서 uniform distribution 분포를 생성하는 법
- 실제 양자컴퓨터를 사용한 학습에서 적은 shot으로도 학습하는 법

그럼 논문의 내용을 소개하도록 하겠습니다.

## copula

이 논문에서 generator가 생성하는 분포와 discriminator가 판단하는 분포는 원래 데이터의 분포가 아니라 `copula space` 라는 변환된 공간입니다. 이 변환을 Probability integral transform이라고 부릅니다. 주로 여러 분포의 correlation이 어떻게 되어 있는지를 확인하는데 사용한다고 합니다.

이 논문에서 알아야 할 점은 Probability integral transform을 거치고 나면 값의 범위가 [0, 1]에서 uniform하게 변한다는 것입니다. 값의 범위가 다루기 쉽고, uniformity가 보장된다는 점은 데이터를 모델링할 때, 굉장히 유용한 특성입니다. 또한 역변환을 거치면 다시 본래의 data space의 값을 복원할 수 있습니다. 복원하는 과정에서 원래 분포를 gaussian distributioin이라고 가정하기 때문에 probability integral transform을 한 뒤, 다시 역변환 하면 원래 값과 조금은 차이가 있습니다.

이제 수식으로 설명을 해보자면, n개의 random variables $(X_1, X_2, ..., X_n) 개가 있을 때, 누적확률밀도함수 $F_i(x)=P(X_i\leq x)$ 를 사용하여 변환을 진행합니다. 이 때문에 이름이 probability **integral** function이라 붙는 것이죠. 이제 각 확률변수 $X_i$ 에 대해서 $U_i = F_i(X_i)$ 를 정의하자. 정의에 의해 $0 \leq U_i \leq 1$ 을 만족한다. 이제 copula function은 아래와 같이 정의된다.

$$C(u_1, u_2, ..., u_n) = P(U_1 \leq u_1, ... , U_n \leq u_n)$$

Reverse probability integral transform 을 사용하면 copula space에서 본래의 확률분포로 변환할 수 있다.

$$(X_1, ..., X_n) = (F_1^{-1}(U_1), ... , F_n^{-1}(U_n))$$

확률변수의 값 -> 누적확률의 과정을 거꾸로 진행하여 누적확률 -> 확률변수 값으로 매핑하는 것이다. $F$의 정의로부터 $F^{-1}$은 자연스럽게 정의되는데, 확률변수의 분포를 normal distribution으로 가정하면 쉽게 계산 가능하다. 확률변수의 본래 분포를 어떻게 가정하냐에 따라 copula도 gaussian copula, t copula 등등 종류가 많아진다.

중요한 점은 각 $u_i \sim U_i$ 들은 [0, 1]에서 uniform distribution이라는 것이다.

## 결론



### 참고문헌

[1] Zhu, Elton Yechao, et al. "Generative quantum learning of joint probability distribution functions." Physical Review Research 4.4 (2022): 043092.