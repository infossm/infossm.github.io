---
layout: post
title: "QGAN in Finance"
date: 2023-08-19
author: red1108
tags: [quantum, quantum-machine-learning, gan, generative, ai, qgan, qml]
---

> Keywords: `GAN`, `QGAN`, `copula`, `finance data`

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

## 금융 데이터 준비

이 논문에서 준비한 데이터는 애플, 마이크로소프트의 일일 수익률이다. 2010년~2018년 까지의 데이터이다. 대략적으로 생각해 보면 두 회사는 같은 IT섹터에 포함되어 있으므로 당연히 주가 변동률이 어느정도 양의 correlation이 있을 것이다.

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/4ff93a26-6ea0-466a-a10c-2d9260f13c0a"></p>
<center><b>그림 1. (a)시작 금액을 $10,000로 했을때 수집 데이터 기간에서 자산의 변동. (b) 일일 수익률의 scatter plot </b></center>

scatter plot을 보면 오른쪽 위로 기울어져 있으므로 확실히 양의 상관관계가 보인다. 

QGAN을 사용하여 최종적으로 모사할 분포는 위 그림 1에서 (b)이다. 하지만 바로 모사하는것은 생성할 점의 분포가 정규화된 형태가 아니기 때문에 어렵다. 또한 논문의 표현에 따르면 joint probability에서의 두 정보가 얽힌 정도를 copula space로 옮기면 양자상태의 얽힘에 대응시킬 수 있다는 표현이 있는데, 이 문장의 의미는 아직 이해하지 못하였다.

결론적으로, 논문에서 사용한 프로세스는 다음과 같다.

1. 주어진 애플/마이크로소프트의 일일 수익률을 copula space로 변환한다.

2. 모델 (GAN, QGAN, QCBM)을 사용하여 copula space상의 분포를 생성하도록 학습한다.

3. 생성한 copula데이터를 역변환으로 original space로 옮긴다.

4. 변환된 데이터와 원래 데이터를 비교한다.

## 모델 소개

이 논문에선 GAN, QGAN, QBCM(quantum cirquit born mahcine)을 비교하였다. 결과론적으론 QGAN, QBCM에서 양자 우위를 확인하였다는 것이 논문의 골자이지만, 여기서는 QGAN, 그 중에서도 quantum generator에 대해서만 소개하도록 하겠다. Discriminator은 고전 신경망으로 이루어져 있고, 모델 구조도 단순히 fc layer 한두개 연결한 거라 예상 가능한 방식으로 학습을 진행하였다.

이 논문에서 사용한 QGAN의 신기했던 점은 다음과 같다.
1. quantum generator는 입력 노이즈를 받지 않는다.
2. quantum generator의 출력값을 uniform distribution으로 만들기 위한 방법론이 신기했다. 또한 이러한 상황에서도 학습이 된다는 점도 신선했다.

### Quantum generator 구조

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/79a6f239-fa6d-4dba-8c4c-6a1fed67dcf9"></p>
<center><b>그림 2. QGAN에서 generator circuit의 구조 </b></center>

위 그림은 논문에서 사용한 generator circuit의 구조이다. 생성할 값은 x, y좌표로 두개이므로 숫자 하나당 3큐빗을 사용하였다. 또한 3큐빗 단위로 모델의 레이어가 붙는데, 이는 그림 2의 (b)에서 확인할 수 있다. (b)가 레이어 한 층이고, 논문에서는 레이어 하나만 사용하였다. 총 파라미터 수는 2 * (9+3) = 24개이다. 파라미터 24개만으로 의미 있는 결과를 얻었다는게 대단한 것 같다.

위 그림에서 알아야 할 점은 두 가지이다. 좌표 하나 뽑는데 큐빗 3개를 썼다는 점. 그리고 큐빗 3개 단위로 레이어가 붙었다는 점이다.

### Output generating

QGAN을 코딩해 본 사람은 알겠지만, 단순히 Observable이나 0/1 확률을 출력으로 바로 쓰게 되면 모델의 출력이 중심부로 극도로 몰려 있는 현상이 생긴다. 이는 Bloch sphere에서 적도부분에 대부분의 확률이 몰리는 현상 때문이다.

하지만 Quantum generator가 만들어야 할 분포는 copula space이기 때문에 x, y좌표가 모두 uniform해야 한다. 이를 위해 굉장히 신기한 방법을 쓴다. shot을 여러번 보내서 평균낸 값을 사용하는 것이 아니라 shot 한번마다 데이터를 만드는 것이다. 0/1 basis로 측정한 결과가 101이라면 결과값은 이진법으로 0.101이 되는것이다. 물론 이러면 정밀도가 떨어지기 때문에 classical하게 랜덤 20비트 bitstring을 생성하여 뒤에(LSB위치에)붙인다. 0.10100101101.. 이렇게 되는것이다.

Observable들을 모델의 출력으로 사용하려면 반복횟수를 늘려서 값의 정밀도를 높여야 하는데, 이렇게 shot 한번당 모델 출력으로 1:1로 대응시키는 것은 참신한 접근이라 느껴졌다.

## Model training

모델의 파라미터는 24개이므로 parameter shift rule을 사용하여 epoch하나를 돌리려면 48번의 학습이 필요하다. 논문에선 이게 많다고 느꼈는지, epoch당 10번의 shot으로 모델을 학습하는 방식을 설명한다. 이미 heuristic optimization분야에선 많이 쓰이고 있는 spsa를 가져오는 것이다. 양자컴퓨팅 분야에서도 많이 쓰이는 방법인 것 같다. gradient추정치를 계산하는 방식은 아래와 같다.

$$(\hat{g}_k(\theta_k))_i=\frac{C(\theta_k+c_k\Delta_k)-C(\theta_k-c_k\Delta_k)}{2c_k(\Delta_k)_i}$$

$\hat{g}_k$는 k번째 iteration에서 gradient의 근사치이다. spsa algorithm은 이러한 iteration을 k=1,2,3,4,5 총 5번 반복한다. gradient 근사치를 구하고, $\theta_{k+1}=\theta_k-a_k\hat{g}_k(\theta_k)$ 이렇게 gradient를 적용하는 과정 또한 5번 반복하는 것이다.

k, i notation이 헷갈릴 수도 있는데 k는 현재 몇 번째 iteration인지를 나타내는 거고, i는 몇번째 원소에 대한 gradient인지를 의미하는 것이다. 아래 파이썬 코드를 참고하면 훨씬 직관적으로 이해할 수 있다.

```python
class SPSA_opt():
    def __init__(self, a=0.008, c=0.01, n_iter = 5, gamma=0.101):
        self.epoch = 1
        self.a = a
        self.c = c
        self.n_iter = n_iter
        self.gamma = gamma

    def step(self):
        self.epoch += 1
    
    def get_params(self):
        # (a, c) tuple return
        return self.a/self.epoch, self.c/(self.epoch**self.gamma)

def spsa_step(params):
    # 한 iter당 n_iter * 2 만큼 shot을 보냄
    spsa_opt = SPSA_opt()
    for i in range(spsa_opt.n_iter):
        a, c = spsa_opt.get_params()

        loss_plus = generator_cost_fn(params + c * delta).detach().numpy()
        loss_minus = generator_cost_fn(params - c * delta).detach().numpy()
        g = (loss_plus - loss_minus)/(2*c*delta)

        params -= a*g
        spsa_opt.step()
    
    return params
```

## Train result

<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/919a5706-c1d2-4472-b25d-046d001a6125"></p>
<center><b>그림 3. (a)noisy simulator (b) 실제 양자컴퓨터 상에서 학습한 결과 </b></center>

위 그림 3을 보면 생성자와 판별자가 어떻게 학습되었는지를 살펴볼 수 있다. 초반엔 discriminator의 학습이 먼저 진행되어 생성자가 만드는 이미지를 모두 가짜 이미지로 판단해버리다가 simulator에서는 300epoch, 실제 양자컴에서는 400epoch쯤에서 판별자가 무언가를 깨닫고 생성을 잘 해내기 시작했다. 논문 원문에서는 이를 Aha! moment라고 하던데 연구자들도 굉장히 신기하게 보았던 것 같다. 물론 이것도 신기하지만, 생각보다 양자컴퓨터에서도 별 무리 없이 학습이 잘 진행된다는 게 놀랍다.

논문에서는 생성된 분포가 실제 분포와 동일한지를 2-dimensional 2-sample KolmogorovSmirnov (KS) test 로 검정하였는데, 이를 지표를 통해 GAN, QGAN, QCBM의 생성 성능을 비교할 수 있다.

<center><b>표 1. 모델별 KS test결과 </b></center>
<p align="center"><img src="https://github.com/infossm/infossm.github.io/assets/17401630/d4fbe5b0-1cf7-4c2f-a0be-68e48054dfb4"></p>


$D_{ks}$는 낮을수록 좋고 p-value는 높을수록 좋다. 만약 p-value가 0.05보다 낮다면 "생성된 분포는 원래 분포와 같다"라는 귀무가설을 기각해야 해서 생성한 분포가 다르다는 것을 의미한다. 모든 모델의 파라미터 수를 24개로 고정하여서 GAN은 몇몇 경우에는 생성된 분포의 p-value가 threshold 미만인 것을 확인할 수 있다. QGAN은 시뮬레이션에서도, 실제 양자컴에서도 굉장히 좋은 성능을 보였다.

또한 학습 횟수는 GAN은 20000, QGAN은 2000, QCBM은 200번으로 고전적인 신경망보다 더 적은 epoch로도 학습이 가능하였다.

## 결론

- QGAN으로 GAN보다 더 좋은 성능을 내었다.
    - 양자 컴퓨팅에서의 파라미터 24개와 고전적인 신경망에서 파라미터 24개는 다르긴 하지만, 그래도 의미 있는 결과이다.
- 필요한 학습 횟수 또한 적었다.
    - 이것 또한 고전 신경망의 epoch 1번과 양자 컴퓨터의 epoch 1번을 동등하게 비교할 수 있느냐? 는 질문이 들긴 한다.

흥미로운 점은, 자체적으로 언제쯤 이 논문에서 사용한 QGAN기술이 상용화될지 결론 부분에서 예측한 게 있는데, 일반적인 구조화된 데이터셋이 10~100차원 정도라고 가정했을때 대략 현재의 에러율을 유지하면서 50큐빗 쯤 되면 제품성이 생길 것이라 예측했다.

개인적으로 이 논문은 양자 회로의 출력을 어떻게 후처리할지, 학습은 어떻게 해야 할지에 관해서 참고할 만한게 많은 논문이라 생각한다.

### 참고문헌

[1] Zhu, Elton Yechao, et al. "Generative quantum learning of joint probability distribution functions." Physical Review Research 4.4 (2022): 043092.