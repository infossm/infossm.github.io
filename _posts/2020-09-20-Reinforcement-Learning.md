---
layout: post
title: AI tutorial - Reinforcement Learning
date: 2020-09-20 20:55
author: cjmp1
tags: deep learning
---

# AI tutorial - 2 . Reinforcement Learning

## Contents

1. 들어가며
2. 강화학습이란?
3. CarRacing-v0를 이용한 강화학습
4. 결론
5. 참고

## 들어가며

  앞서 AI tutorial 글로 Image Classification 에 대해 다루었습니다. 그 후에 AI 분야에 관련해서 입문하기 쉽고 재미있는 부분을 고르는데에 있어서 고민이 많았습니다. 이미지 분류기에서 파생되는 객체탐지(Object Detection), 감정분석(Sentiment Analysis) 같은 분야에서 다루는 자연어처리 기법, 시계열 예측, 적대적 신경망, 음성 변조 같은 곳에 유용하게 사용되는 VAE(Variational AutoEncoder) 등 여러가지가 존재하고 있습니다. 

 그러나 대부분의 AI분야가 이론과 실습 분야는 손쉽게 접근할 수 있지만, 실제 자기가 직접 모델을 만들고, 데이터를 찾아서 다운받고, 커스텀화해서 만들어내는 재미가 매우 떨어지고 있다고 생각했습니다. 따라서 실제로 간단한 가이드라인이 제공된다면, 공부하는 사람이 직접 처음부터 코드를 작성하고, 학습시켜서 결과를 비교하는 과정을 체험해볼 수 있는 분야가 어디일지 고민해보았습니다.

 여러가지 분야 중 이에 가장 걸맞는 분야는 강화학습이라고 생각했습니다. 강화학습은 무엇보다 가장 손이가기 쉽고, 재미있고 그리고 결과를 만들어내는 재미가 있습니다. 특정 환경(Environment)를 설정하고, AI에게 여러가지 선택지를 주고 이에 따른 결과를 이용해 AI를 학습시키는 과정을 통해서, 정말 이 인공지능이 점점 학습을 통해 진화하고 있구나를 가장 직접적으로 느낄 수 있는 분야이기 때문입니다.

 따라서 이번에는 강화학습에 대해 한 번 설명해보도록 하겠습니다.

## 강화학습이란?

 강화학습은 어떤 환경(Environment) 속에서 agent(주체)가 특정 행위(action)를 하는 과정 속에서 그  행위에 대한 환경의 변화(결과)를 바탕으로 목표(reward)를 이룰 수 있도록 학습해 나가는 과정을 말합니다. 즉 다시말해 어떠한 상황(state) 속에서 보상(reward)를 최대화 할 수 있는 행위(action)을 학습해 나가는 과정이라고 생각하시면 되겠습니다.

 강아지를 예로 들어볼까요? 저희는 강아지에게 "손", "앉아", "엎드려" 등 여러가지의 문장을 학습시키려고 노력합니다. 이 때 이런 학습시키려는 사람과, 강아지 및 여러 요소들을 모두 합쳐 환경(Environment)라고 하고, 강아지는 행위를 하게되는 주체(agent)가 되며, 사람이 문장을 말했을 때, 강아지의 선택이 행위(action)이 되게 됩니다. 이에 따른 간식이 바로 보상(reward)가 되는 것이지요. 강아지는 이 보상을 최대화 하기 위해 처음에는 할 수 없었던 손, 앉아 그리고 엎드려 같은 행위들을 학습해나가게 되는 것 입니다.

 이제 이 강화학습을 실제로 수행하기 위해서, 수학적인 부분으로 접근해보도록 하겠습니다. 먼저 Policy(정책)라는 것이 존재합니다. policy는 현재의 상태와 행할 action이 mapping 되어 있는 것으로, 보통 초기에 랜덤 또는 확률에 의거하는 방식을 채택합니다. 

 그리고 Value Function이 존재합니다. 가치함수는 t 시점(시간)에서 어떤 action을 했을 때, 보상에 대한 기댓값으로 정의합니다. 처음에는 저희는 이 기댓값에 대한 함수 정보를 전혀 모르는 상태로 시작합니다. 이 가치함수 Q의 경우 시점에 따라 페널티를 추가로 받게 되는데, bellman equation에 따라 우리는 다음 시점의 Q함수가 모든 action에 대해 정의되어 있다면, 이를 Q(현재시점) = reward + penalty * Q'(다음시점) 과 같이 유기적으로 표현할 수 있습니다. (자세한 수학 정보는 아래 참고 논문을 참고하면 좋습니다) 이 뜻은 즉 Qi 가 Qi+1로 정의가 가능하다는 것이고, i 가 커지면 Q함수가 정의된다고 표현할 수도 있습니다. 이를 이용해 우리는 Loss Function 을 매 iteration i마다 정의 할 수 있게 됩니다.

 이 방법을 이용해서, 가치함수를 최대화시키는 policy를 배워 나가는 것이 강화학습 중에서도 DQN(deep q-learning)입니다. 이는 Playing Atari with Deep Reinforcement Learning 이라는 논문에서 처음 소개되었습니다. Atari 게임은 매우 고전적인 게임으로, 여러가지 종류의 게임을 포함하고 있는데, 이 게임 화면을 input으로 입력받아 픽셀값을 신경망에 집어넣어 학습을 하게 되는 것입니다.

 그렇다면 이제, 직접 수행해본 CarRacing-v0에서의 강화학습을 보여드리겠습니다.

## CarRacing-v0를 이용한 강화학습

CarRacing-v0 환경은 openaiGym에 이미 구현이 되어있는 상태입니다.

https://gym.openai.com/envs/CarRacing-v0/

위 링크를 들어가시면 github 코드 주소가 나오고 이를 사용하시면 됩니다. CarRacing의 경우 경주 자동차 한 대와, 트랙 그리고 트랙 주변에는 초록색 배경으로 이루어져있습니다. 이 때 자동차는 주체(agent)가 되고 행위(가속, 브레이크, 핸들 left, 핸들 right) (action) 들이 사용 가능합니다.

CarRacing 외에 자기가 직접만든 게임환경을 통해서 실습해보고 싶은 경우 필요한 함수들을 세팅해주면 됩니다. 저는 CarRacing 이전에 매우 간단한 20*20 환경에서 Snake 게임을 python으로 제작해서 돌려보았었습니다. 필요한 함수로는 init(초기화 함수), step(행동과 현재 상태를 입력으로 받아서 달라진 state, reward 등을 return해주는 함수), reset(환경을 reset해주는 함수), render(만들어진 게임을 window창으로 띄워주는 함수) 가 기본적으로 필요합니다. 이에 추가적으로 게임에 필요한 함수들이 있다면 추가를 해주고 나서 gym-push 패키지를 이용해 push해주고 import 하면 사용 가능합니다. 

우선 매우 간단한 코드를 살펴보겠습니다.

```python
import gym

env = gym.make('CarRacing-v0')
obs = env.reset()

finished = False
total_reward = 0
while not finished:
    obs, reward, finished, info = env.step(env.action_space.sample()) # 랜덤 무브
    total_reward = total_reward + reward
    env.render() # 화면 표시 및 env 변경
print('Total Reward: ', total_reward)
```

보시다 시피 env 라는 환경클래스 안에는 여러가지가 게임에 대한 전반적인 요소들이 모두 들어가 있어야합니다. 이 곳에서, action_space(가속, 브레이크, 핸들 left, 핸들 right)를 정의해주고, 학습 과정에서 어떻게 policy에 따라서 action을 선택할지를 모두 정의해주어야 합니다. 

```python
def get_action(q_value, train=False, step=None, params=None, device=None):
    if train:
        epsilon = params.epsilon_final + (params.epsilon_start - params.epsilon_final) * \
            math.exp(-1 * step / params.epsilon_step)
        if random.random() <= epsilon:
            action_index = random.randrange(get_action_space())
            action = ACTIONS[action_index]
            return torch.tensor([action_index], device=device)[0], action
    action_index = q_value.max(1)[1]
    action = ACTIONS[action_index[0]]
    return action_index[0], action
```

 위 방법은 epsilon-greedy라는 기법을 이용해서 action을 선택해주는 방법입니다. 난수 값이 epsilon 보다 작다면 무작위 행동을 취하게 되고, epsilon 보다 크거나 같다면, 현재까지 만들어진 Q함수를 통해서 action을 선택해주게 됩니다. 이 epsilon값은 시간이 가면 갈수록 특정 parameter값을 지정해서 점점 작아지게 만들어줍니다. 이렇게 하면, 초기에는 거의 대부분의 action이 random으로 결정되게 되지만, 시간이 지날수록 Q함수를 사용하는 비율이 높아지게 되는 것이지요. 

 이제 학습의 순서는 다음과 같습니다

- 현재 신경망으로 state에 대한 Q-value를 받아옵니다.
- get_action함수를 이용해 action을 선택하고, next_state를 구합니다
- next_state를 이용해 신경망을 업데이트 하게 됩니다.

신경망 업데이트는 아래와 같이 진행되게 됩니다.

- q_value = 현재 q_net 의 모든 기댓값의 합
- next_q_value = target q_net의 max값
- expected_q_value = rewards + penalty_parameter * next_q_value * finishflag

로 이루어집니다.

이를 코드로 구현하게 되면 아래와 같습니다.

```python
def run(self):
        state = torch.tensor(self.environment.reset(),
                             device=self.device,
                             dtype=torch.float32)
        self._update_target_q_net()
        for step in range(int(self.params.num_of_steps)):
            q_value = self.current_q_net(torch.stack([state]))
            action_index, action = get_action(q_value,
                                              train=True,
                                              step=step,
                                              params=self.params,
                                              device=self.device)
            next_state, reward, done = self.environment.step(action)
            next_state = torch.tensor(next_state,
                                      device=self.device,
                                      dtype=torch.float32)
            self.replay_memory.add(state, action_index, reward, next_state, done)
            state = next_state
            if done:
                state = torch.tensor(self.environment.reset(),
                                     device=self.device,
                                     dtype=torch.float32)
            if len(self.replay_memory.memory) > self.params.batch_size:
                loss = self._update_current_q_net()
                print('Update: {}. Loss: {}'.format(step, loss))
            if step % self.params.target_update_freq == 0:
                self._update_target_q_net()
        torch.save(self.target_q_net.state_dict(), self.model_path)

    def _update_current_q_net(self):
        batch = self.replay_memory.sample(self.params.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.current_q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0]

        expected_q_values = rewards + self.params.discount_factor * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())
```

위 dqn training 부분은 https://github.com/novicasarenac 에서 받아왔습니다.

이 외에도 강화학습 알고리즘은 매우 여러가지가 존재합니다. a2c, a3c 라는 actor-critic 이라는 알고리즘도 존재합니다.

강화학습에 유전 알고리즘 즉 Genetic Algorithm을 사용해볼 수도 있습니다. 

https://www.youtube.com/watch?v=Aut32pR5PQA 유트브에서 아주 재밌는 영상을 보게 되었고, 이 영상을 바탕으로 CarRacing-v0에 한 번 사용해보았습니다.

우선 유전 알고리즘에 대해 간단히 짚고 넘어가보겠습니다.

부모세대를 통해 자식세대를 만들어내는 방식을 통해서 점점 유전자가 강화하는 방법입니다. 유전자별로 적합도함수(가치함수)라는 것을 이용해서 그 유전자(해)가 얼마나 적합한지를 판단할수 있어야합니다. 그 후, 부모세대에서 주요 연산을 통해 자식세대를 생성하게 됩니다.

- 선택 : selection 기법은 다음세대로 전해질 유전자를 선택하는 기법을 말합니다. 이에는 룰렛휠, 랭킹 기법 등이 존재하는데, 룰렛 휠이란, 앞서 적합도 함수를 이용해서, 각 유전자별로 선택될 확률을 지정한 뒤 선택하게 되는 기법을 의미하며, 랭킹 기법은 가장 좋은 유전자를 순서대로 고르게 된다는 것을 의미합니다.
- 교차 : crossover 기법 즉 교차는 세대 내에서의 위치(순서) 변경을 통해 이루어지며 선택된 해들의 순서를 뒤바꾸는 방식으로 이루어지게 됩니다.
- 변이 : mutation은 선택된 유전자들 중에서 특정 유전자를 random하게 아예 변이시켜버리는 기법으로, 최적의 해에 더 빠르게 갈 수 있는 지름길역할을 해줍니다. 물론 변이 확률을 너무 크게 설정할 경우 오히려 학습에 악영향을 미칠 수 있습니다.

위 알고리즘을 코드로 구현하면 아래와 같습니다.

```python
def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, p: float):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases
    
    
def mutation(parent_weights_biases: np.array, p: float):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-10, 10)
    return child_weight_biases


def ranking_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[:2]
    return parent1, parent2


def roulette_wheel_selection(population: List[Individual]):
    total_fitness = np.sum([individual.fitness for individual in population])
    selection_probabilities = [individual.fitness / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]
```

다음은 main 함수입니다.

```python
def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        print('generating',i)
        # Selection
        # parent1 = roulette_wheel_selection(old_population)
        # parent2 = roulette_wheel_selection(old_population)
        parent1, parent2 = ranking_selection(old_population)
        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)
        # If children fitness is greater thant parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    POPULATION_SIZE = 100
    MAX_GENERATION = 2
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    p = Population(ConvNetTorchIndividal(None, None, None), POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, 0)
    p.run(env, generation, verbose=False, output_folder='')

    env.close()

```

 일반 그냥 합성신경망을 사용했으며, 인구수를 100으로 설정했습니다. 즉 한세대에 100개의 agent가 생성되며, 이들은 차례대로 selection -> crossover -> mutation 함수를 거친 후, 적합도 함수를 계산하게 되어 새로운 세대에 들어가게 됩니다. 

 아래 함수는 main함수로, 환경과 변이, 교차 확률 파라미터들을 설정해준 뒤, 학습을 실행하는 모습입니다. 

 실제 run 에서는 매 세대별 100개의 인구수의 최소 적합도와 최대 적합도를 출력해 학습 경과를 살펴보도록 해주었으며, old_population 을 바탕으로 generation 함수를 실행하게 됩니다. 아래는 그 경과를 출력하는 코드 부분입니다.

```python
max_fit = -1.0
min_fit = 10000
for p in self.old_population:
	p.calculate_fitness(env)
	print(p.fitness)
	if min_fit > p.fitness:
		min_fit = p.fitness
	if max_fit < p.fitness:
		max_fit = p.fitness
	print(c)
	c += 1
print(min_fit, max_fit)
minfit_.append(min_fit)
maxfit_.append(max_fit)
print(i, 'start')
```

 위에서 계속 설명되고 있는 적합도 함수는 그 유전자의 성능 즉 가치를 잘 매길 수 있는 함수를 만들어주면 됩니다. 저같은 경우 reward들의 총합으로 적합도 함수를 계산했습니다. 아래는 적합도 함수를 계산해주는 run_single 함수입니다.

```python
def run_single(self, env, n_episodes=100, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            env.render()
            obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
            obs = obs.reshape((-1, 3, 96, 96))
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()
```

즉 run 안에서 세대를 돌아가게 되며 각 세대별로, 100개의 인구를 결과적으로 만들어냅니다. 이 각 인구들은 run_single을 통해 적합도를 계산하게 되고, 매 세대가 끝날 때마다 신경망이 업데이트 되게 됩니다.

## 결론

 결과 dqn의 경우 일반적인 회귀함수와 비슷한 loss 감소율을 보여주었습니다. 하지만 학습시간이 매우 오래걸린다는 단점과 공간복잡도가 매우 크게 소요된다는 단점이 존재했습니다. 저로서는 한 번에 다 돌리기 힘든 양의 메모리여서, epoch을 나눠서 학습했습니다. 

 반면 Genetic Algorithm의 경우에 매우 간소하고 학습시간도 빠르지만, 결과가 썩 좋지 않았습니다. 특정 경우에는 mutation이 올바른 방향으로 이루어지고, 잘 선택되어서 몇세대 안으로 바로 좋은 결과물을 보여주기도 하는 반면, 특정 경우에는 몇십세대를 거치고 나서도 제자리에서 빙빙 도는 등, optimal solution을 찾지 못하는 경우도 존재했습니다.  

이번에는 강화학습이라는 분야를 다루어 보았습니다. 강화학습 또한 딥러닝이 나오게 되고 빠르게 발전한 분야중 하나입니다. 앞으로도 인공지능이 매우 전반적으로 크게 활용될 전망을 보여주고 있습니다. 따라서 여러가지 분야의 ai 기법들을 알아두는 것은 큰 도움이 될 거라고 생각합니다.

  강화학습이라는 재미있는 분야에 대해 접근하는데 조금이나마 도움이 됬으면 좋겠습니다.

## 참고

https://gym.openai.com/envs/CarRacing-v0/

https://github.com/novicasarenac
