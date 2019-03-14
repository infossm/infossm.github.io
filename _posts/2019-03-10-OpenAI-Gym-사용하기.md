---
layout: post
title:  " OpenAI Gym 사용하기"
author: ho94949
date: 2019-03-09 15:00
tags: [OpenAI]
---

# 서론

[OpenAI Gym](https://gym.openai.com/)은 강화학습을 도와주고, 좀 더 일반적인 상황에서 강화학습을 할 수 있게 해주는 라이브러리 입니다. 이 게시글에서는 OpenAI Gym을 사용하는 법을 알아보고, 샘플 프로젝트인 CartPole-v1에서 동작하는 신경망을 만들어봅니다.

# OpenAI Gym의 설치

OpenAI Gym은 python3.5 이상에서 작동합니다. gym은 간단하게 pip로 설치할 수 있습니다. 그리고 이 샘플 프로젝트를 도와주는 numpy와 keras를 설치해야합니다. 기본적으로 이는 Python에 추가적인 지원을 해주는 [Anaconda](https://www.anaconda.com/distribution/)가 해줄 수 있으며, gym설치 및 numpy 업그레이드를 진행해야합니다.

```
pip install gym
pip install numpy --upgrade
```

# CartPole-v1


![CartPole](/assets/images/openaigym/cartpole.jpg)

우리가 실행시켜볼 프로젝트는 CartPole-v1 입니다. 이는 손 위에 막대를 세워놓고, 막대가 쓰러지지 않도록 하는 것을 시뮤레이션 해 놓은 것입니다. 일단 다음 코드를 실행해 봅시다.

```
import gym
import numpy as np
import random

env = gym.make('CartPole-v1')
goal_steps = 500

while True:
  obs = env.reset()
  for i in range(goal_steps):
    obs, reward, done, info = env.step(random.randrange(0, 2))
    if done: break
    env.render()
```

코드를 실행하면, 선 위에, 검정색 직사각형이 있고, 그 위에 갈색 직사각형이 있는것을 볼 수 있습니다. 이 검정색 직사각형이 막대이며, 0, 1에 따라 왼쪽 오른쪽으로 움직일 수 있습니다. 갈색 직사각형은 막대입니다. 그럼 이제부터 이 코드를 하나하나씩 따라가봅시다.

- 5번째 줄: `CartPole-v1`이라는 gym 환경을 만들어서 `env`에 저장합니다. 이 gym 환경은 예제 gym 환경이고, 자기가 원하는 환경을 만들수도 있습니다.
- 6번째 줄: `goal_steps` 는 이 환경에서, 최대 몇번까지로 동작을 제한할것인가를 설정합니다.
- 9번째 줄: 현재 환경을 초기화합니다.
- 10번째 줄: 매 프레임 마다 반복하게 됩니다. 최대 `goal_steps`까지로 제한했습니다.
- 11번째 줄: `random.randrange(0, 2)`로 환경에서의 동작을 합니다. 즉 0이나 1중 랜덤으로 하나를 주게 됩니다. 이 `env.step`의 반환값은 `obs, reward, done, info` 입니다. 
  - `obs`: 현재 상황을 의미합니다. 이 프로젝트에서는 [카트 위치, 카트 속도, 막대의 각도, 끝에서의 막대의 속도]를 의미합니다.
  - `reward`: 보상을 의미합니다. 이 프로젝트에서는 오래 버티는게 목표이며, 보상은 밑에서 설명할, 막대가 떨어지지 않는 이상 한 턴을 버텼으므로 1이 주어집니다.
  - `done`: 막대가 떨어졌는지를 의미합니다. 이 프로젝트에서 막대가 떨어졌다고 말하는 조건은, 막대의 각도가 12도 이상 기울었거나, 카트가 2.4칸 이상 움직인 경우(화면에서 나간 경우) 떨어졌다고 판단을 합니다.
- 12번째 줄: 막대가 떨어졌을 경우, 끝냅니다.
- 13번째 줄: 디스플레이를 합니다.

# 학습데이터 생성

우리가 만들어야 할 것은 오랜 시간을 버티는  (즉 reward의 합을 최대화 하는) 신경망을 제작해야합니다.

코드는 다음과 같습니다.

```
def data_preparation(N, K, f, render=False):
  game_data = []
  for i in range(N):
    score = 0
    game_steps = []
    obs = env.reset()
    for step in range(goal_steps):
      if render: env.render()
      action = f(obs)
      game_steps.append((obs, action))
      obs, reward, done, info = env.step(action)
      score += reward
      if done:
        break
    game_data.append((score, game_steps))
  
  game_data.sort(key=lambda s:-s[0])
  
  training_set = []
  for i in range(K):
    for step in game_data[i][1]:
      if step[1] == 0:
        training_set.append((step[0], [1, 0]))
      else:
        training_set.append((step[0], [0, 1]))

  print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
  if render:
    for i in game_data:
      print("Score: {0}".format(i[0]))
  
  return training_set
```

이 코드는 인자로, `N`, `K`, `f` 를 받습니다. 이 인자는 다음과 같은 의미를 가집니다.
- `N`: Cartpole을 실행해 볼 횟수
- `K`: 그 중에 뽑을 데이터의 갯수
- `f`: Cartpole을 어떤식으로 동작시킬 지 결정하는 함수.

그래서 우리는 2-14번째 줄에서 N번 동안 실행 해 보고, 현재 상황과 그 때 한 판단들을 다 저장해 놓습니다. 그리고 N번동안 실행한것을 모두 모은다음에, 점수가 상위 K개인것만 뽑은 후에 training set을 만들었습니다. training set의 구조는 관측을 담은 길이 4의 배열과, 행동을 담은 길이 2의 배열로 구성되어 있습니다. 이 길이 2의 배열의 뜻은, `[p, q]`가 p의 확률로 0, q의 확률로 1을 실행했다는 것이고, 여기서는 랜덤한 요소가 없기 때문에 `[1, 0]` 혹은 `[0, 1]` 입니다.

일단 처음에는 아무 정보가 없기 때문에 랜덤하게 실행해야 합니다. 즉, 다음과 같이 실행하면 우리는 랜덤한 게임 중 상위 K개의 데이터를 모을 수 있을 것입니다.

```
training_data = data_preparation(1000, 50, lambda s: random.randrange(0, 2))
```

실행 결과는 다음과 같은 형태일 것입니다: `50/1000th score: 46.0` 랜덤을 사용했기 때문에 환경마다 다르게 나올 수 있습니다.

# 데이터 학습 

우리는 이제 이 데이터를 학습해야합니다. 학습을 위해서는 신경망을 구축해야합니다. 우리가 쓸 신경망은, input layer의 차원이 4, 첫번째 hidden layer의 차원이 128, 두번째 hidden layer의 차원이 52, 그다음에 마지막 output layer의 차원이 2인 신경망을 사용할 것이고, ReLU activator를 사용할 것며, 마지막 hidden layer와 output layer 사이에는 softmax를 사용할 것입니다.

모든 layer는 dense하게 구성이 되어있습니다. 그리고 optimizer는 Adam을 쓸 것입니다. layer와 optimizer는 일반적으로 쓰이는 구성입니다.

우리는 `keras`패키지에 있는 모델을 그대로 사용할 것이며, 다음 `build_model`함수를 이용하여 미리 만들어진 신경망을 가지고 올 수 있습니다.

```
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam  
def build_model():
  model = Sequential()
  model.add(Dense(128, input_dim=4, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='mse', optimizer=Adam())
  return model
```

그리고 모델을 학습시키는 것은 꽤나 간단한 일입니다. 차원을 바꿔준 이후에 `model.fit` 함수를 이용하면 됩니다.

```
def train_model(model, training_set):
  X = np.array([i[0] for i in training_set]).reshape(-1, 4)
  y = np.array([i[1] for i in training_set]).reshape(-1, 2)
  model.fit(X, y, epochs=10)
```

그럼 이렇게 학습시킨 모델로 환경에 따라 예측 하는 방법은 `model.predict` 함수를 사용하면 되고, 다음과 같은 코드를 작성할 수 있습니다.

```
if __name__ == '__main__':
  N = 1000
  K = 50
  model = build_model()
  training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
  train_model(model, training_data)
  
  def predictor(s):
    return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0] )
    
  data_preparation(100, 100, predictor, True)
```  

# self-play

이와 같이 실행을 하면, 아예 랜덤을 쓴 경우보다는 좀 더 오래 막대기가 버티지만, 아직도 버티지 못합니다. 그래서 우리는 이렇게 만든 데이터 중 다시 상위 데이터만 고르는 방식으로 성능을 개선 해 보려고 합니다.

```
if __name__ == '__main__':
  N = 1000
  K = 50
  self_play_count = 10
  model = build_model()
  training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
  train_model(model, training_data)
  
  def predictor(s):
    return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0] )

  for i in range(self_play_count):
    K = (N//9 + K)//2
    training_data = data_preparation(N, K, predictor)
    train_model(model, training_data)
  
  data_preparation(100, 100, predictor, True)
```  

여기서 우리는 적당히 상위 K개를 고를 때의 K를 늘리고, 학습된 데이터로 다시 플레이를 해보며 상위 K개의 데이터를 가져오는 것을 반복합니다. 

이렇게 학습을 진행하면, 플레이가 매우 좋은 데이터만 선정하여 다시 학습을 진행하게 되고, 마지막에는 매우 균형맞추기를 잘하는 신경망을 볼 수 있습니다.


# 결론

이제 우리는 이 cartpole-v1 을 강화학습을 통해서 학습시켰습니다. 다양한 문제에 대해서, gym을 만들고 강화학습을 할 수 있게 만들어 주는 이 OpenAI Gym을 이용하면, 이 균형 맞추는 문제 뿐만 아니라 다양한 문제에서의 인공지능을 쉽게 만들 수 있을 것입니다.

