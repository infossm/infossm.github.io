---
layout: post
title: UCB1 알고리즘의 pseudo-regret 분석
date: 2020-03-08 14:30
author: choyi0521
tags: [machine-learning, multi-armed-bandit, upper-confidence-bound]
---


# 소개

&nbsp;&nbsp;&nbsp;&nbsp;멀티 암드 밴딧(Multi-armed bandit) 문제는 순차적 의사결정 문제(sequential decision problems)의 일종으로써 충분한 정보가 주어지지 않은 상황에서 탐색(exploration)과 이용(exploitation)의 균형을 찾는 것을 목표로 합니다. 멀티 암드 밴딧 문제에는 다양한 변종이 있는데 이번 글에서는 확률론적 멀티 암드 밴딧(Stochastic Multi-armed Bandit)과 성능 지표인 후회값(regret)의 정의를 알아보겠습니다. 또한, 이 문제를 해결할 수 있는 간단한 알고리즘 중 하나인 UCB1의 유사 후회(pseudo-regret)의 상한이 라운드 수에 대한 로그 스케일 이하임을 증명해보겠습니다.

# Stochastic Multi-armed Bandit

&nbsp;&nbsp;&nbsp;&nbsp;확률론적 멀티 암드 밴딧(Stochastic Multi-armed Bandit)은 각 팔(arm)에 대한 보상(reward)이 알려지지 않은 분포에 따라 독립적으로 주어진다고 가정한 문제입니다. $i=1, 2, ..., K$번 팔에 대한 보상의 확률 분포 $\nu_i$는 $[0, 1]$에서 정의되어 있고 에이전트(agent)에게 알려지지 않았습니다. 매 시점 $t=1, 2, ..., n(>=K)$마다 에이전트는 $I_t$번 팔을 선택해서 보상 $X_{i,I_t} \sim \nu_i$을 얻습니다. 이때 $X_{i,t}$들은 모든 시점, 팔 번호에 대해 독립적으로 주어집니다.

&nbsp;&nbsp;&nbsp;&nbsp;에이전트의 목표는 다음과 같이 정의되는 유사 후회를 최소화하는 것입니다.

$$\overline{R}_n=\max_{i=1,...,K}\mathbb{E}\left[\sum_{t=1}^n X_{i,t} - \sum_{t=1}^n X_{I_t,t} \right]$$

&nbsp;&nbsp;&nbsp;&nbsp;$i=1,2,...,K$에 대해 $\nu_i$의 평균을 $\mu_i$라고 하고 $\mu^{\ast}=\max_{i=1,...,K}\mu_i$라고 하면 유사 후회를 다음과 같이 표현할 수 있습니다.

$$\overline{R}_n=n\mu^{\ast}-\sum_{t=1}^n \mathbb{E}\left[\mu_{I_t}\right]$$

&nbsp;&nbsp;&nbsp;&nbsp;또한, $\Delta_i=\mu^\ast-\mu_i$로 놓고 시점 $1,2,...,t$에서 $i$번 팔을 선택한 총 횟수를 $T_i(t)$라고 하면 유사 후회를 $\overline R_n=\sum_{i: \mu_i < \mu^\ast} \Delta_i \mathbb{E}\left[T_i(n) \right]$으로 표현할 수 있습니다. $\overline{R}_n$은 항상 0 이상의 값을 가지며, 에이전트가 최적의 팔을 많이 선택할수록 유사 후회의 값이 작아집니다. 따라서 이 값을 줄이기 위해서는 에이전트가 최적의 팔을 빠르게 찾을 수 있는 방법을 찾아야 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;앞으로 사용할 몇 가지의 표기를 추가로 정리하겠습니다. $\hat{\mu}_{i, j}$를 $i$번 팔이 처음부터 $j$번 선택되면서 얻은 보상의 표본 평균이라고 정의하겠습니다. 또한, 최적의 팔에 관한 변수에 $\ast$를 붙여서 표기할 것입니다. 예를 들어, $\hat{\mu}_j^\ast$는 최적의 팔이 처음부터 $j$번 선택되면서 얻은 보상의 표본 평균이고 $T^\ast(t)$는 최적의 팔이 시점 $1,2,...,t$에서 선택된 총 횟수입니다.

## UCB1 알고리즘

&nbsp;&nbsp;&nbsp;&nbsp;유사 후회를 줄이기 위해서는 보상 분포의 평균이 큰 팔을 선택하면 되지만 보상 분포가 정확히 어떠한지 에이전트에게 알려져 있지 않습니다. 이때 평균의 추정량으로 간단히 표본 평균(sample mean)을 사용하는 것을 고려할 수 있습니다. 하지만 표본 평균은 충분한 표본이 없을 때 실제 평균과의 차이가 클 수 있기 때문에 처음부터 표본 평균이 큰 팔만 선택하다보면 최적의 팔을 영영 선택하지 않게 될 수 있습니다. UCB(upper confidence bound) 계열의 알고리즘은 이러한 문제를 해결하기 위해 표본 평균에 불확실성과 관련된 항을 더한 값이 큰 팔을 선택합니다.

&nbsp;&nbsp;&nbsp;&nbsp;UCB1 알고리즘은 보상 평균에 대한 단측 신뢰 구간(one-sided confidence interval)의 상한을 비교해서 팔을 선택합니다. 뒤에서 증명할 Hoeffding’s Inequality에 따르면 $\mu_i$가 $\hat{\mu}_{i, T_i(t-1)}$보다 $\epsilon = \sqrt{\alpha \ln{t} \over {2T_i(t-1)}}$이상 차이가 날 확률이 $t^{-\alpha}$이하라는 것을 알 수 있습니다. 여기서 UCB1 알고리즘은 $\alpha=4$로 놓을 때 얻어지는 단측 신뢰 구간의 상한을 사용해서 다음과 같은 정책을 따릅니다.

>$t \le K$인 경우, $t$번 팔을 선택합니다.  
$t > K$인 경우, $\hat{\mu}_{i, T_i(t-1)} + \sqrt{2 \ln t \over {T_i(t-1)}}$가 최대가 되는 $i$번 팔을 선택합니다.



&nbsp;&nbsp;&nbsp;&nbsp;$\sqrt{2 \ln{t} \over {T_i(t-1)}}$ 값은 $i$번 팔을 많이 선택할수록 작아지기 때문에 처음에는 이 항에 큰 영향을 받다가 표본이 많이 쌓이면 표본 평균의 영향이 커짐을 알 수 있습니다. 따라서 초기에는 에이전트가 다양한 팔을 선택하다가 표본이 충분히 쌓인 뒤에는 표본 평균이 큰 팔을 주로 선택하게 됩니다.

# 보조 정리

&nbsp;&nbsp;&nbsp;&nbsp;이제 UCB1의 유사 후회의 상한을 분석하기 위해서 Hoeffding’s Inequality를 포함한 여러 보조 정리를 증명해보겠습니다.

>(Markov’s Inequality) 음이 아닌 확률 변수 $X$에 대해 $\mathbb{P}(X>t) \le {\mathbb{E}\left[ X \right] \over t}$이 성립한다.

&nbsp;&nbsp;&nbsp;&nbsp;다음과 같이 지시 함수(indicator function)을 사용해서 부등식을 세워봅시다.

$$X \ge t \mathsf{1} \left\{X \ge t\right\}$$

&nbsp;&nbsp;&nbsp;&nbsp;이 식은 $X \ge t$인 경우에 지시 함수 값이 $1$이 되고 $ X < t $인 경우에 우항이 $0$이 되기 때문에 실제로 성립함을 알 수 있습니다. 이제 식의 양변에 기댓값을 취해봅시다.

$$\mathbb{E}\left[ X \right] \ge t \mathbb{E}\left[ \mathsf{1} \left\{X \ge t\right\} \right] = t\mathbb{P}(X\ge t)$$

&nbsp;&nbsp;&nbsp;&nbsp;그러면 Markov’s Inequality인 ${\mathbb{E}\left[ X \right] \over t} \ge \mathbb{P}(X\ge t)$을 얻을 수 있습니다. $\blacksquare$

>(Chernoff Bounds) $X$를 확률 변수로 놓자. $\epsilon > 0$에 대해 $\mathbb{P}(X \ge \epsilon) \le e^{-\lambda \epsilon} \mathbb{E}\left[e^{\lambda X} \right]$이 성립한다.

&nbsp;&nbsp;&nbsp;&nbsp;Markov’s Inequality를 이용해 바로 증명이 가능합니다.

$$
\begin{aligned}
\mathbb{P}(X \ge \epsilon) &= \mathbb{P}(e^{\lambda X} \ge e^{\lambda \epsilon})\\
&\le e^{-\lambda \epsilon} \mathbb{E} \left[ e^{\lambda X} \right]\\
\end{aligned}
$$

$\blacksquare$

>(Hoeffding’s Lemma) 확률 변수 $X$가 $[a, b]$에서 거의 확실하게(almost surely) 발생하고 $\mathbb{E}\left[X\right]=0$이면 임의의 실수 $s$에 대해 $\mathbb{E}\left[e^{sX} \right] \le e^{s^2(b-a)^2\over 8}$이 성립한다.

&nbsp;&nbsp;&nbsp;&nbsp;$e^{sx}$는 $x$에 관한 볼록 함수이므로

$$e^{sX} \le {b-X \over b-a}e^{sa}+{X-a \over b-a}e^{sb}$$

&nbsp;&nbsp;&nbsp;&nbsp;양변에 기댓값을 취하고 $\alpha={-a\over b-a}, t=(b-a)s$로 놓아 스케일링 해줍니다.

$$
\begin{aligned}
\mathbb{E}\left[e^{sX}\right] &\le \mathbb{E}\left[{b-X \over b-a}e^{sa}+{X-a \over b-a}e^{sb}\right]\\
&={b-\mathbb{E}\left[X\right] \over b-a}e^{sa}+{\mathbb{E}\left[X\right]-a \over b-a}e^{sb}\\
&={b \over b-a}e^{sa}-{a \over b-a}e^{sb}\\
&=(1-\alpha)e^{-t\alpha}+\alpha e^{t(1-\alpha)}\\
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;부등식의 우변에 로그를 취한 값을 $f(t)$로 놓습니다. Taylor's theorem에 의해 임의의 실수 $u$에 대해 다음 식을 만족하는 $0$과 $u$ 사이의 값 $v$가 존재합니다.

$$f(u)=f(0)+uf'(0)+{1\over 2}u^2 f''(v)$$

&nbsp;&nbsp;&nbsp;&nbsp;$f(0), f'(0), f''(x)$는 다음과 같습니다. 

$$
\begin{aligned}
f(0)&=-\alpha x+\ln(1-\alpha+\alpha e^x)\Big|_{x=0}\\
&=0\\
f'(0)&=-\alpha+{\alpha e^x \over 1-\alpha+\alpha e^x}\Big|_{x=0}\\
&=0\\
f''(x)&={\alpha e^x(1-\alpha+\alpha e^x)-(\alpha e^x)^2 \over (1-\alpha+\alpha e^x)^2}\\
&={\alpha e^x \over 1-\alpha+\alpha e^x}\left(1-{\alpha e^x \over 1-\alpha+\alpha e^x}\right)\\
&\le {1\over 4}\\
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;따라서

$$
\begin{aligned}
\mathbb{E}\left[e^{sX}\right] &\le e^{f(t)}\\
&\le e^{ {1\over 8}t^2}\\
&= e^{s^2(b-a)^2\over 8}\\
\end{aligned}
$$

$\blacksquare$

>(Hoeffding’s Inequality) 독립된 확률 변수 $X_i, X_2, ..., X_m$가 주어졌고 각 $X_i$가 $[a_i, b_i]$에서 거의 확실하게(almost surely) 발생한다고 하자. 그러면 $\epsilon>0$에 대해  
$$
\begin{aligned}
\mathbb{P}\left({1\over m}\sum_{i=1}^m X_i - {1\over m}\sum_{i=1}^m\mathbb{E}[X_i] \ge \epsilon\right)\le \exp\left({-2\epsilon^2 m^2 \over \sum_{i=1}^m (b_i-a_i)^2}\right)\\
\mathbb{P}\left({1\over m}\sum_{i=1}^m X_i - {1\over m}\sum_{i=1}^m\mathbb{E}[X_i] \le -\epsilon\right)\le \exp\left({-2\epsilon^2 m^2 \over \sum_{i=1}^m (b_i-a_i)^2}\right)\\
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;$Y_i=X_i-\mathbb{E}\left[X_i\right]$로 놓읍시다. 그러면 $Y_i$는 $[a_i-\mathbb{E}\left[X_i\right], b_i-\mathbb{E}\left[X_i\right]]$에서 거의 확실하게 발생하고 $\mathbb{E}\left[Y_i\right]=0$을 만족합니다. 본 정리의 부등식을 증명하기 위해 Chernoff Bounds와 Hoeffding’s Lemma를 사용합니다.

$$
\begin{aligned}
\mathbb{P}\left({1\over m}\sum_{i=1}^m X_i - {1\over m}\sum_{i=1}^m \mathbb{E}[X_i] \ge \epsilon\right) &= \mathbb{P}\left(\sum_{i=1}^m Y_i \ge \epsilon m\right)\\
&\le e^{-\lambda\epsilon m}\mathbb{E}\left[e^{\lambda\sum_{i=1}^m Y_i}\right] &&\text{Chernoff Bounds}\\
&= e^{-\lambda\epsilon m}\prod_{i=1}^m\mathbb{E}\left[e^{\lambda Y_i}\right] &&\text{Independence of the }Y_i's\\
&\le e^{-\lambda\epsilon m}\prod_{i=1}^m \exp({\lambda^2(b_i-a_i)^2\over 8}) &&\text{Hoeffding’s Lemma}\\
&= \exp\left(\lambda^2{\sum_{i=1}^m (b_i-a_i)^2\over 8} - \lambda \epsilon m \right)\\
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;우리는 식을 세우면서 $\lambda$에 어떠한 제약을 하지 않았으므로 임의의 실수 값으로 설정할 수 있습니다. 가장 타이트한 부등식을 만들기 위해 마지막 항이 최솟값이 되도록 $\lambda={4\epsilon m \over \sum_{i=1}^m (b_i-a_i)^2}$를 대입합니다. 그러면 정리의 첫 번째 부등식인 $\mathbb{P}\left({1\over m}\sum_{i=1}^m X_i - {1\over m}\mathbb{E}[X_i] \ge \epsilon\right) \le \exp\left({-2\epsilon^2 m^2 \over \sum_{i=1}^m (b_i-a_i)^2}\right)$을 확인할 수 있습니다. 정리의 두 번째 부등식은 $-X_1, -X_2, ..., -X_m$에 첫 번째 부등식을 적용하여 증명합니다. $\blacksquare$

# 주요 정리

&nbsp;&nbsp;&nbsp;&nbsp;이제 본 글의 메인 주제인 UCB1의 유사 후회와 관련된 정리를 증명해보겠습니다.

> (P Auer 2002) $K>1$개의 팔이 있고 각 팔의 보상 분포가 $[0, 1]$에서 정의되어 있습니다. UCB1 정책의 유사 후회 $\overline{R}_n$은 다음을 만족합니다.  
$$\overline R_n \le 8\sum_{i:\mu_i<\mu^\ast}{\ln n \over \Delta_i} + \left(1+{\pi^2 \over 3}\right)\sum_{i=1}^K \Delta_i$$

&nbsp;&nbsp;&nbsp;&nbsp;유사 후회를 $\overline R_n = \sum_{\mu_i<\mu^\ast} \Delta_i\mathbb{E}\left[T_i(n) \right]$로 나타낼 수 있다는 점을 상기합시다. 시점 $t$에 최적의 팔이 아닌 $i$번 팔을 선택하는 경우는 $\hat \mu_{T^\ast(t-1)}^\ast + \sqrt{2 \ln{t} \over {T^\ast(t-1)}} \le \hat{\mu}_{i, T_i(t-1)} + \sqrt{2 \ln{t} \over {T_i(t-1)}}$입니다. 이 경우가 발생하는 횟수의 기댓값의 상한을 가지고 유사 후회의 상한을 구할 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;편의를 위해 $c_{t,s}={\sqrt{2 \ln{t} \over {s}}}$라고 하겠습니다. $\hat \mu_s^\ast + c_{t,s} \le \hat \mu_{i, s_i} + c_{t,s_i}$를 만족하면 다음 세 가지의 부등식 중 적어도 하나는 만족해야 합니다.

$$
\hat{\mu}_s^\ast - \mu^\ast \le -c_{t,s}\tag{1}
$$

$$
\hat{\mu}_{i, s_i}-\mu_i \ge c_{t,s_i}\tag{2}
$$

$$
\mu^\ast-\mu_i < 2c_{t,s_i}\tag{3}
$$

&nbsp;&nbsp;&nbsp;&nbsp;그렇지 않으면 $\hat \mu_s^\ast + c_{t,s} > \mu^\ast \ge \mu_i + 2c_{t,s_i} > \hat \mu_{i, s_i} + c_{t,s_i}$이므로 모순이기 때문입니다. 이렇게 세 개의 부등식으로 나누면 $s_i$를 크게 만들어 (3)을 거짓으로 만들고 (1) 혹은 (2)가 발생할 확률을 Hoeffding’s Inequality를 통해 제한 할 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;Hoeffding’s Inequality에 의해

$$\mathbb{P}\left(\hat{\mu}_s^\ast - \mu^\ast \le -c_{t,s}\right) \le t^{-4}$$

$$\mathbb{P}\left(\hat{\mu}_{i, s_i}-\mu_i \ge c_{t,s}\right) \le t^{-4}$$

&nbsp;&nbsp;&nbsp;&nbsp;(3)이 거짓일 $s_i$의 조건은

$$\Delta_i = \mu^\ast-\mu_i \ge 2c_{t,s_i}$$

$$s_i \ge {8\ln t \over \Delta_i^2}$$

&nbsp;&nbsp;&nbsp;&nbsp;이제 $\mathbb{E}[T_i(n)]$의 상한을 구해봅시다.

$$
\begin{aligned}
\mathbb{E}\left[T_i(n) \right]&=\mathbb{E}\left[\sum_{t=1}^n\mathsf{1}\{I_t=i\} \right]\\
&= 1 + \sum_{t=K+1}^n\mathbb{P}\left(I_t=i\right)\\
&\le \lceil{8\ln n \over \Delta_i^2}\rceil + \sum_{t=K+1}^n\mathbb{P}\left(I_t=i, T_i(t-1)\ge\lceil{8\ln n \over \Delta_i^2}\rceil\right)\\
&= \lceil{8\ln n \over \Delta_i^2}\rceil + \sum_{t=K+1}^n\mathbb{P}\left(\hat{\mu}_{T^\ast(t-1)}^\ast + c_{t, T^\ast(t-1)} \le \hat{\mu}_{i, T_i(t-1)} + c_{t, T_i(t-1)}, T_i(t-1)\ge\lceil{8\ln n \over \Delta_i^2}\rceil\right)\\
&\le \lceil{8\ln n \over \Delta_i^2}\rceil + \sum_{t=K+1}^n \sum_{s=1}^{t-1} \sum_{s_i=\lceil{8\ln n \over \Delta_i^2}\rceil}^{t-1} \mathbb{P}\left(\hat{\mu}_s^\ast + c_{t, s} \le \hat{\mu}_{i, s_i} + c_{t, s_i}\right)\\
&\le \lceil{8\ln n \over \Delta_i^2}\rceil + \sum_{t=K+1}^n \sum_{s=1}^{t-1} \sum_{s_i=\lceil{8\ln n \over \Delta_i^2}\rceil}^{t-1} \{\mathbb{P}\left(\hat{\mu}_s^\ast - \mu^\ast \le -c_{t,s}\right)+\mathbb{P}\left(\hat{\mu}_{i, s_i}-\mu_i \ge c_{t,s}\right)\}\\
&\le \lceil{8\ln n \over \Delta_i^2}\rceil + \sum_{t=1}^\infty \sum_{s=1}^{t} \sum_{s_i=1}^{t} 2t^{-4}\\
&\le {8\ln n \over \Delta_i^2} + 1 + {\pi^2 \over 3}
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;따라서 유사 후회는 다음과 같은 부등식을 만족합니다.

$$
\begin{aligned}
\overline{R}_n &= \sum_{i:\mu_i<\mu^\ast} \Delta_i\mathbb{E}\left[T_i(n) \right]\\
&\le 8\sum_{i:\mu_i<\mu^\ast}{\ln n \over \Delta_i} + \left(1+{\pi^2 \over 3}\right)\sum_{i=1}^K \Delta_i\\
\end{aligned}
$$

$\blacksquare$

# 마무리

&nbsp;&nbsp;&nbsp;&nbsp;지금까지 UCB1의 유사 후회의 상한을 분석하기 위해 여러가지 정리를 증명해보았습니다. UCB1 알고리즘을 공부할 때 통계학의 기초 정리를 증명하면서 좀 더 깊게 이해할 수 있었던 것 같습니다. 본 글이 멀티 암드 밴딧의 기본적인 알고리즘을 분석하는데 도움되었길 바랍니다.

# 참고문헌

* [<span style="color:blue">Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems</span>](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SurveyBCB12.pdf)
* [<span style="color:blue">Kevin Jamieson's lecture: Stochastic Multi-Armed Bandits, Regret Minimization</span>](https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture3/lecture3.pdf)
* [<span style="color:blue">Finite-time Analysis of the Multiarmed Bandit Problem</span>](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
