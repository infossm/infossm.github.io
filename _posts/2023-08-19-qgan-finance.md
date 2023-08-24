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

이 논문에서 generator가 생성하는 분포와 discriminator가 판단하는 분포는 원래 데이터의 분포가 아니라 `copula space` 라는 변환된 공간입니다. 이 변환을 Probability integral transform이라고 부릅니다.

copula라는 개념이 경제학에서도 많이 쓰이는데, 결합확률 분포를 모델링하는데 

## 결론



### 참고문헌

[1] Zhu, Elton Yechao, et al. "Generative quantum learning of joint probability distribution functions." Physical Review Research 4.4 (2022): 043092.