---
layout: post
title:  "Multi-Armed Bandit and UCB Algorithm"
author: harinboy
date: 2023-04-23 21:00
tags: [multi-armed-bandit, reinforcement-learning]
---

# 소개

&nbsp;&nbsp;&nbsp;&nbsp;이 글에서는 불확실성 속에서 좋은 선택을 찾아내야 하는 문제인 Multi-Armed Bandit 문제가 무엇인지 소개하고, 이를 해결하는 알고리즘 중 하나인 UCB 알고리즘을 최대한 쉽고 직관적인 방식으로 유도하고 증명할 것입니다. 기초적인 확률론과 통계학 (ex. 확률분포, 기댓값, 모평균 표본평균)에 대한 지식이 필요하지만 그 외의 배경지식이 없는 사람도 이해할 수 있게 작성했습니다.

# 문제 설명

### 돈을 주는 기계?
 
&nbsp;&nbsp;&nbsp;&nbsp;당신 앞에 2개의 버튼이 달린 기계가 있습니다. 놀랍게도, 이 기계는 버튼을 누르면 돈이 나옵니다! 그리고 운 좋은 당신은 어느 쪽이든 버튼을 누를 기회가 100번이나 있습니다!!  
&nbsp;&nbsp;&nbsp;&nbsp;각 버튼이 얼마를 주는지 모르는 당신은 일단 왼쪽 버튼을 몇 번 눌러봅니다. 100원, 10000원, 100원 ... 오른쪽 버튼도 몇 번 눌러봅니다. 5010원, 5020원, 4980원... 엥? 문득 당신은 남은 기회는 조금 더 전략적으로 써야겠다는 생각이 듭니다. 아무래도 두 버튼이 주는 액수가, 특히 주는 액수의 평균이 다를 것 같으니, 가능하면 평균이 더 큰 버튼을 더 많이 누르고 싶습니다. 그런데 어느 버튼이 "좋은 버튼"인지 알 수가 없으니 문제입니다. 1원이라도 더 벌려면 도대체 어떻게 해야할까요?


### Multi-Armed Bandit 문제의 수학적 정의
 
&nbsp;&nbsp;&nbsp;&nbsp;Multi-Armed Bandit 또는 $K$-armed Bandit라고 불리는 문제는 위에서 버튼으로 표현한 $K$개의 선택지(actions)들의 집합 $A=\lbrace1, 2, ..., K\rbrace$와 각 선택지가 주는 보상(reward)의 확률분포 $r(a)$ $(a\in A)$, 그리고 선택을 할 수 있는 횟수 $T$로 구성됩니다. $a$를 선택하면 확률분포 $r(a)$에서 하나의 랜덤하고 독립적인 표본(sample)이 보상으로 주어지고 T번의 선택 동안 얻는 보상을 최대화해야하는 문제입니다. 여기서 중요한 점은 $r(a)$에 대한 정보는 선택하는 쪽에서는 알 수가 없다는 것입니다. Multi-Armed라는 말은 원래는 버튼이 아니라 여러 슬롯 머신의 손잡이(arm)를 당기는 문제로 표현되었기 때문입니다.

&nbsp;&nbsp;&nbsp;&nbsp;일반성을 잃지 않고 선택을 할 때마다 0과 1 사이의 실수값을 보상으로 받는다고 하겠습니다. 즉, $r(a)\in[0, 1]$입니다.

&nbsp;&nbsp;&nbsp;&nbsp;만약 $r(a)$들에 대한 정보가 있다면 기댓값이 최대인 선택지를 $T$번 모두 선택하는 것이 가장 좋은 전략입니다. 이를 편하게 표현하기 위해 각 선택지의 보상의 기댓값과 기댓값이 가장 큰 선택지에 대한 기호를 정의합시다.

$\mu_a=\mathbb{E}[r(a)]$ : 각 선택지$(a\in A)$의 보상의 기댓값입니다.  
$a^* =\text{argmax}_ {a\in A}(\mu_a)$ : $\mu_a$가 최대가 되는 $a\in A$, 즉 기댓값 상 가장 좋은 선택지입니다. 편의상 유일하다고 하겠습니다.  
$\mu^* =\max_{a\in A}(\mu_a)$ : 가장 좋은 선택지가 주는 보상의 기댓값입니다.

&nbsp;&nbsp;&nbsp;&nbsp;보상의 확률분포 $r$에 대한 정보가 있다면 항상 $a^* $를 선택함으로써 
$T$번의 선택 후에는 평균적으로 $T\mu^* $의 보상을 얻을 수 있을 것입니다. 하지만 $r$에 대한 정보가 없으니 어떻게 해야할까요? 실제로 선택지들을 골라서 $r(a)$의 sample을 확인하여 $r$에 대한 정보를 조금씩 모을 수 밖에 없습니다. 결국 손해를 조금 보면서 여러 선택지들을 시도해보고, 얻은 정보를 바탕으로 가장 좋아보이는 선택지를 최대한 골라야 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;애초에 $\mu^* $가 작다면 얻을 수 있는 최대 보상도 작기 때문에 일정량의 보상을 반드시 얻어내는 알고리즘을 만들 수는 없습니다. 그래서 특정 알고리즘을 분석할 때에는 얻을 수 있는 보상보다는 최선의 결과인 $T\mu^* $에 비해 얼마나 손해를 보게 되는지에 주로 초점을 맞춥니다. 이를 표현하기 위해 regret이라는 개념을 정의하겠습니다. 만약 당신이 순서대로 $a_1, a_2, ..., a_T$의 선택지들을 순서대로 골랐다고 한다면 regret $R$은

$$R=T\mu^* -\sum_{t=1}^{T}\mu_{a_t}$$

로 정의됩니다. 이 때 한 번의 선택이 regret에 기여하는 양은 $\mu^* -\mu_{a_t}$입니다. $\text{argmax}_ {a\in A}(\mu_a)$가 유일하다고 했으니 $a^* $가 아닌 선택지를 고를때마다 regret은 늘어나게 됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;선택지 $a$를 골랐을 때에 발생하는 regret을 $\Delta_a=\mu^* -\mu_a$로 정의하고 $T$번의 선택이 모두 끝난 후 선택지를 $a$를 고른 횟수를 $N_{a, T}$이라고 한다면 regret을 다음과 같이 표현할 수도 있습니다.

$$R=\sum_{t=1}^{T}\Delta_{a_t}=\sum_{a=1}^{K}\Delta_aN_{a, T}$$

&nbsp;&nbsp;&nbsp;&nbsp;만약 앞에 나온 기계의 두 버튼을 공평하게 50번씩 누르고 나서 뒤늦게 한 쪽 버튼은 평균 4천원을, 다른 쪽 버튼은 평균 6천원을 준다는 사실을 알게된다면 당신은 잘못된 선택 한 번 당 2000원씩 50번, 총 10만원의 손해를 본 기분이 들 것입니다. 이 경우 10만원의 손해가 regret이 되는 겁니다. 우리의 목표는 regret이 작은 알고리즘을 만드는 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;$r(a)\in[0, 1]$이라고 가정했으므로 한 번의 선택으로 얻는 regret은 최대 1, 전체 regret은 최대 T입니다. 이것을 얼마나 더 줄일 수 있을까요? 단순히 몇 배 줄이는 것이 아니라 아예 복잡도 상으로 더 작은 regret을 얻을 수 있을까요? 놀랍게도 UCB 알고리즘은 regret의 기댓값이 $O(\sqrt{KT \log T })$가 됨을 보장합니다. 물론 Bandit 문제는 확률에 의해 결과가 결정되므로 아무리 좋은 알고리즘도 매우 운이 나쁘다면 큰 regret을 얻을 수도 있습니다. 하지만 UCB 알고리즘은 실패가 허용되는 확률 $\delta>0$가 임의로로 설정되었을 때 $1-\delta$ 이상의 확률로 $O\left(\sqrt{KT\log (KT/\delta)}\right)$의 regret을 얻을 수 있다는 것도 보일 수 있습니다. 반대로 얘기하면 $\sqrt{T}$ 스케일보다 큰 regret이 생길 가능성을 원하는 만큼 작게 줄일 수 있다는 뜻입니다.  

Side-note :  보통 "action"이라는 용어를 번역할 때에는 "행동"이라는 용어로 번역하는데, 이 글 안에서는 "선택지"라는 말이 더 이해하기 쉽다고 생각해 다르게 번역했습니다. 또 일부 문헌은 위에서 regret으로 정의한 $R$값을 pseudo-regret이라고 정의하기도 합니다. 그런 경우 진짜 regret은 기댓값인 $\mu_{a_t}$가 아닌 진짜로 얻은 보상을 기준으로 합니다. 하지만 regret의 기댓값이 pseudo-regret이고, 그런 문헌들도 분석하는 것은 pseudo-regret이기 때문에 이 글에서는 그냥 regret으로 지칭하겠습니다.


# 가장 좋은 "것 같은" 선택지

&nbsp;&nbsp;&nbsp;&nbsp;일단 당신은 버튼들을 이미 몇 번 눌렀습니다. 이제 첫 번째로 생각할 수 있는 전략은 지금까지 관찰한 결과를 바탕으로 남은 기회 동안 가장 좋은 것 같은 선택지를 계속 고르는 전략입니다. 이를 표현하기 위해 간단하게 시점 $t$까지 특정한 선택지를 고른 횟수와 특정 선택지를 골랐을 때에 얻은 보상들의 평균을 나타내는 기호를 정의하겠습니다.

$N_{a, t} = \sum_{i=1}^{t}\mathbb{1}(a_i = a)$ : $t$번째 선택 까지 $a$가 선택된 횟수입니다.  
$\hat{\mu}_ {a, t}=\frac{\sum_{i=1}^{t}r_i\mathbb{1}(a_i=a)}{N_{a, t}}$ : $t$번째 선택까지 $a$를 선택했을 때에 얻은 보상들의 평균입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이 때 $\mathbb{1}(a_i = a)$는 $i$번째로 선택한 선택지인 $a_i$가 $a$인 경우에 1, 그 외의 경우에 0인 함수이고, $r_i$는 $i$번째 선택 때에 얻은 reward입니다. $N_{a, t}$과 $\hat{\mu}_ {a, t}$은 $t$번의 선택 후 $t+1$번째 선택을 할 때에 고려할 수 있는 각 $a$에 대한 정보들 중 일부입니다. $\hat{\mu}_ {a, t}$가 정의되지 않는 상황을 피하기 위해서 가장 첫 $K$번의 선택은 $1, 2, ..., K$번 선택지를 한 번씩 시도한 후 $t\geq K$일 때부터 $\hat{\mu}_ {a, t}$를 정의하는 것으로 하겠습니다. 

&nbsp;&nbsp;&nbsp;&nbsp;지금까지의 관찰한 보상들을 바탕으로 좋은 선택지를 계속 고르는 방법들 중 하나는 지금까지 각 선택지가 준 보상의 평균이 가장 높은 선택지를 고르는 방법이 있습니다. 위의 기호로 표현하면 $\text{argmax}_ {a\in A} \hat{\mu}_ {a, t}$를 믿고 계속 고른다는 뜻입니다.

&nbsp;&nbsp;&nbsp;&nbsp;하지만 조금 걱정이 됩니다. 그렇게 고른 선택지가 가장 좋은 선택지가 아닌데 괜히 믿었다가는 큰 손해가 나는 것이 아닐까요? 그러면 다른 선택지도 조금씩은 선택해봐야하는 것일까요? 하지만 반대로 가장 좋은 선택지를 골라냈다면, 시험 삼아 다른 선택지를 선택해볼때마다 regret이 증가하게 됩니다. $\hat{\mu}_ {a, t}$ 값을 무작정 믿었다가는 잘못된 믿음으로 regret을 늘려버릴 수도 있고, $\hat{\mu}_ {a, t}$ 값을 너무 믿지 않았다가는 $\hat{\mu}_ {a, t}$이 주는 정보가 버려질 수도 것입니다. 그렇다면 $\hat{\mu}_ {a, t}$는 도대체 얼마나 믿어야하는 것일까요?

# 신뢰 구간 (Confidence Interval)

&nbsp;&nbsp;&nbsp;&nbsp;지금까지 관찰한 값을 통해 $\hat{\mu}_ {a, t}$를 구성할 수 있지만, 이 값이 얼마나 믿을 만한 값인지 생각해봐야할 때가 왔습니다. 값이 믿을 만하다는 것을 수학적으로 어떻게 정의할 수 있을까요? 통계학에서 이에 대응되는 개념이 몇 가지 있습니다만, 이 글에서 살펴볼 것은 신뢰 구간의 개념입니다. 

&nbsp;&nbsp;&nbsp;&nbsp;신뢰 구간은 알고 싶은 값, 이 경우에는 $\mu_a$,가 높은 확률로 존재할 구간입니다. 설문 조사 등에서 95% 신뢰 수준에서 오차가 몇 %p라고 표기하는 것이 바로 신뢰 구간의 개념입니다. 어떤 정해진 확률 $\delta$에 대해, $1-\delta$ 이상의 확률로 $\mu_a$와 $\hat{\mu}_ {a, t}$의 절대 오차가 수학적으로 $\beta_{a, t}(1-\delta)$ 이하라면, 신뢰 구간은

$$\mu_ a\in [\hat{\mu}_ {a, t}-\beta_ {a, t}(1-\delta), \hat{\mu}_ {a, t}+\beta_ {a, t}(1-\delta)]$$

이 됩니다. 오차의 크기이자 신뢰 구간의 크기를 나타내는 $\beta_{a, t}(1-\delta)$가 작을수록 우리는 $\hat{\mu}_ {a, t}$이 "믿을 만하다"고 생각하고, $\beta_{a, t}(1-\delta)$가 클수록 $\hat{\mu}_ {a, t}$이 "믿을 만하지 않다"고 생각할 수 있습니다. 여기서 $\delta$는 $\mu_a$가 신뢰 구간 밖에 있을 최대 확률로, 실패 확률이라고 부릅니다. 또 직관적으로 관찰한 표본의 수가 많을 수록 $\beta_{a, t}(1-\delta)$은 작아질 것이라고 기대할 수 있습니다. 우리는 위 신뢰 구간을 단순히 $[L_{a, t}(\delta), U_{a, t}(\delta)]$로 적겠습니다.

&nbsp;&nbsp;&nbsp;&nbsp;정리하자면, 각 선택지 $a$에 대해서 $t$번째까지의 선택의 결과를 관찰한 후 얻은 보상들을 바탕으로 $\mu_a$가 있을 법한 신뢰 구간 $[L_{a, t}(\delta), U_{a, t}(\delta)]$를 잡았고, 이론적으로 $\mu_a$가 이 구간 안에 있을 확률이 $1-\delta$ 이상, 밖에 있을 확률은 $\delta$ 미만입니다. 선택지 $a$를 많이 고를수록 구간 $[L_{a, t}(\delta), U_{a, t}(\delta)]$는 $\mu_a$를 향해 점점 좁아질 것이고, 고르지 않는다면 $[L_{a, t}(\delta), U_{a, t}(\delta)]$는 변화가 없을 것이라는 것을 직관적으로 이해하고 넘어가면 좋을 것 같습니다.

&nbsp;&nbsp;&nbsp;&nbsp;신뢰 구간을 어떻게 구했다고 치고, 그것을 사용해 알고리즘을 만들고 분석하는 부분을 자세히 보도록 하겠습니다. 신뢰 구간을 구하는 것 자체는 reward의 구조가 다른 bandit 문제에서는 달라지는 부분이지만, 신뢰 구간을 이용하는 아이디어는 똑같이 사용할 수 있기 때문입니다. 신뢰 구간을 구하는 방법은 잠시 후에 증명을 생략하고 보여드리도록 하겠습니다.

# Upper Confidence Bound의 아이디어

&nbsp;&nbsp;&nbsp;&nbsp; $t-1$번의 선택 후 시점 $t$ 때에 모든 신뢰 구간이 유효하다면, 다시 말해 모든 $a\in A$에 대해 $\mu_a \in [L_{a, t-1}, U_{a, t-1}]$인 경우에, $a_t$를 선택했을 때에 생기는 regret $\mu^* -\mu_{a_t}$의 범위를 알 수 있습니다. ($\delta$는 다음 섹션까지 잠시 생략하도록 하겠습니다.) $\mu^* $는 정확히 어느 선택지의 구간을 써야할지는 모르지만, 적어도 모든 구간 중에 가장 큰 값, 즉 $\max_{a\in A} U_{a, t-1}$보다는 작다는 것을 알 수 있습니다. 또 선택지 $a_t$를 고르면 $-\mu_{a_t}\leq-L_{a_t, t}$임도 알 수 있습니다. 두 가지를 결합하면, 

$$\mu^* -\mu_{a_t}\leq\max_{a\in A} U_{a, t-1}-L_{a_t, t-1}$$

이 성립합니다. 언뜻 보기에는 $\max_{a\in A} U_{a, t-1}$는 고정된 값이기 때문에 $L_{a_t, t-1}$를 최대화하는 선택지를 고르는 것이 좋아보입니다. 하지만 이것은 좋지 않습니다. 왼쪽과 오른쪽 버튼 두 선택지가 있을 때, 왼쪽이 조금 더 좋은 선택지라고 가정해봅시다 ($\mu_1>\mu_2$). 그런데 왼쪽과 오른쪽 버튼을 한 번씩 눌렀을 때 오른쪽 버튼이 우연히 더 높은 보상을 주었다고 합시다. 그래서 $L_ {1, t}<L_ {2, t}$가 되어 오른쪽 버튼을 계속 누른다면, $L_ {1, t}$ 값은 그대로인 반면 $L_{2, t}$값은 $\mu_2$를 향해 계속 증가하여 끝까지 틀린 선택지를 고르는 결과를 낳게 됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;분명 $\max_{a\in A} U_{a, t-1}-L_{a_t, t-1}$을 최소화하는 선택을 했지만, 좋은 알고리즘을 얻지는 못 했습니다. 이 알고리즘이 실패한 이유는 너무 greedy했기 때문입니다. 우리는 단 한순간의 $\max U_{a, t-1}-L_{a_t, t-1}$를 줄이는 것이 아니라, $\sum_{t=K+1}^{T}\max U_{a, t-1}-L_{a_t, t-1}$의 합을 줄여야 합니다. 그러기 위해서는 $\max U_{a, t-1}$항에도 관심을 기울여야 합니다. 분명 단 한순간의 선택에서는 이 항의 값은 고정이지만 다음 순간의 $\max U_{a, t-1}$를 줄이는 방법이 존재합니다. $U_{a, t-1}$가 가장 큰 바로 그 선택지를 선택하는 것입니다. $r(a)$의 더 많은 표본을 확인하면 점점 $U_{a, t-1}$가 감소하고 더 먼 시야에서 $\sum_{t=K+1}^{T}\max U_{a, t-1}-L_{a_t, t-1}$를 줄이는 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;그러면 두 가지 선택지로 좁혀졌습니다. $L_{a, t-1}$가 가장 큰 선택지를 골라 단기적인 regret을 줄이거나, $U_{a, t-1}$가 가장 큰 선택지를 골라 장기적으로 regret을 줄이는 것입니다. 두 가지를 적절히 섞어서, 가령 처음 몇 번은 $U_{a, t-1}$가 가장 큰 선택지를 고른 후 마지막 몇 번은 $L_{a, t-1}$가 가장 큰 선택지를 고르는 전략을 만드는 것이 좋아보입니다. 이 때 $U_{a, -1}$가 가장 큰 선택지를 끝까지 골라도 충분히 좋은 알고리즘을 얻을 수 있다는 것을 증명할 수 있습니다. 잠시 $U_{a, t-1}$가 가장 큰 선택지를 골랐을 때에 위에서 설정한 bound가 어떻게 되는지 보겠습니다. 

$$
\begin{aligned}
\mu^* -\mu_{a_t}&\leq\max U_{a, t-1}-L_{a_t, t-1}\\
&=U_{a_t, t-1}-L_{a_t, t-1}\\
&=2\beta_{a_t, t-1}
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;즉, $U_{a, t-1}$가 가장 큰 선택지를 고르면 커봐야 $2\beta_{a_t, t-1}$의 regret이 더해지게 됩니다. $\beta_{a, t-1}$들은 선택지 $a$가 많이 선택될수록 작아지는 것을 생각해보았을 때 충분히 긴 시간 후에는 $U_{a_t, t-1}$가 큰 선택지를 계속 고르는 것도 매우 작은 regret을 가져온다는 것을 감각적으로 알 수 있습니다. 결론적으로는 $K$개의 선택지를 모두 한 번씩 선택한 후 나머지 $T-K$번의 선택을 $U_{a, t-1}$가 가장 큰 것을 계속 고른다면, 전체 regret은 다음과 같이 됩니다.

$$
\begin{aligned}
R&\leq K + \sum_{t=K+1}^{T} 2\beta_{a_t, t-1}\\
&=K + \sum_{t=K+1}^{T} 2\beta_{a_t, t-1}(1-\delta)
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;가장 큰 $U_{a, t-1}$를 고른 알고리즘을 일반적으로 upper-confidence bound algorithm, 줄여서 UCB-알고리즘이라고 부릅니다. 또 바로 위에서 보인 부등식을 보면 더 tight한 $\beta_{a_t, t}(1-\delta)$를 구할수록 더 작은 이론적 regret bound를 구할 수 있다는 것을 알 수 있습니다.


# 신뢰 구간 구하기 -  Hoeffding's Inequality

&nbsp;&nbsp;&nbsp;&nbsp;지금까지는 보상의 구조, 즉 $r$이 별개의 $K$개의 $[0, 1]$ 사이의 실수값을 가지는 분포라는 사실을 사용하지 않았습니다. 그래서 다른 종류의 bandit 문제에서도 여기까지의 결론, 즉 신뢰구간 크기의 합, 혹은 그에 대응되는 값으로 regret을 표현할 수 있다는 식의 비슷한 결론을 얻을 수 있습니다. 이제 이 $K$-armed bandit에 대해 이론적으로 작은 신뢰 구간을 결정할수록 증명할 수 있는 regret bound도 작아지고, 사실 이 모든 증명의 가장 어려운 부분이기도 합니다. 이 경우에는 가장 작은 regret bound를 증명하기 위해 알려진 정리를 사용하겠습니다.

>## Hoeffding's Inequality
>
>$[a, b]$ 사이의 값을 가지고 $\mathbb{E}[X]=0$인 확률변수 X의 독립적인 sample $n$개 $X_1, X_2, ..., X_n$이 있을 때, $\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i$이라고 하자. 임의의 $\epsilon>0$에 대해,
>
>$$
\begin{aligned}
\mathbb{P}(|\bar{X}|\geq\epsilon)\leq 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)
\end{aligned}
>$$
>
>가 성립한다.

&nbsp;&nbsp;&nbsp;&nbsp;정리가 의미하는 것을 잠시 생각해보자면, 확률변수의 값이 bounded라면, 모평균과 표본평균이 $\epsilon$ 이상 차이날 확률이 표본 수 $n$에 지수적으로, 오차 범위 $\epsilon$에는 $\epsilon^2$에 지수적으로 감소한다는 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;우리가 정의한 Multi-Armed Bandit 문제에서는 $b-a=1$임을 이용하고 우변의 확률값을 하나의 변수 $\delta=2\exp\left(-2n\epsilon^2\right)$로 정의해 변형한다면 다음과 같이 됩니다.

>## Hoeffding's Inequality for K-armed Bandit
>어떤 선택지 $a\in A$에 대해 $r(a)$의 sample을 $N_{a, t}$개 관찰했다면, 
>
>$$
\begin{aligned}
\mathbb{P}\left(|\mu_{a}-\hat{\mu}_ {a, t}|\geq\sqrt{\frac{\log{\frac{2}{\delta}}}{2N_{a, t}}}\right)\leq \delta
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;즉, $\beta_{a, t}(1-\delta)=\sqrt{\frac{\log{\frac{2}{\delta}}}{2N_{a, t}}}$로 두면 $1-\delta$ 이상의 확률로 $\|\mu_{a}-\hat{\mu}_ {a, t}\|\leq\beta_{a, t}(1-\delta)$가 성립합니다.


# UCB Algorithm (Upper Confidence Bound Algorithm)과 분석

&nbsp;&nbsp;&nbsp;&nbsp;필요한 블록들은 모두 모였기 때문에, 최종적으로 알고리즘을 기술하고 regret bound를 증명하겠습니다.

>### UCB Algorithm
>
>$K, T, \delta$를 입력으로 받는다.  
>$1, 2, ..., K$번 선택지를 고른다.  
>$t=K+1, ..., T$ 동안
    $\hat{\mu}_ {a, t-1}+\sqrt{\frac{\log{\frac{2}{\delta}}}{2N_{a, t-1}}}$이 가장 큰 선택지를 고르고 보상 $r_i$을 확인한다.  
   
&nbsp;&nbsp;&nbsp;&nbsp;모든 신뢰 구간이 유효한 경우에, 즉 모든 $a\in A, t \in \lbrace1, 2, ..., T\rbrace$에 대해 $\mu_{a, t-1}\in [L_{a, t-1}(\delta), U_{a, t-1}(\delta)]$가 성립한 경우에 regret은 다음과 같이 bound됨은 위에서 이미 확인했습니다.

$$
\begin{aligned}
R&\leq K + \sum_{t=K+1}^{T} 2\beta_{a_t, t-1}(1-\delta)\\
&=K+\sum_{t=K+1}^{T}\sqrt{\frac{2\log{\frac{2}{\delta}}}{N_{a_t, t-1}}}
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;이제 위 값이 아무리 커봐야 $O\left(\sqrt{KT\log{\frac{1}{\delta}}}\right)$임만 보이면 됩니다. 수학적으로 엄밀한 증명보다 직관적인 설명을 먼저 하겠습니다. 각 선택지를 한 번씩 고른 후 UCB 알고리즘에 의해 어떤 선택지 $a$를 $t_1$ 시점에 처음으로 고른다면, $N_{a, t_1-1}=1$이고, 위 식에 의하면 그로 인해 발생하는 regret은 커봐야 $\sqrt{\frac{2\log{\frac{2}{\delta}}}{1}}$이 됩니다. 그 다음으로 고를 때에는 $N_{a, t_2-1}=2$이고 많아야 $\sqrt{\frac{2\log{\frac{2}{\delta}}}{2}}$의 regret이 발생합니다. 이런 식으로 특정한 선택지가 계속 선택된다면 그로 인한 regret의 크기 제한은 $\sqrt{\frac{2\log{\frac{2}{\delta}}}{3}}, \sqrt{\frac{2\log{\frac{2}{\delta}}}{4}}...$와 같이 점점 작아질 것입니다. 그러므로 위 합이 최대한 커지기 위해서는 $T$번의 선택이 $K$개의 선택지에 고르게 분포되는 경우에 최대가 될 것으로 직관적으로 생각할 수 있습니다. 그렇다면 대략

$$
\begin{aligned}
\sum_ {t=K+1}^T\sqrt{\frac{2\log{\frac{2}{\delta}}}{N_{a_t, t-1}}}&\leq K\sum_ {n=1}^{\frac{T}{K}}\sqrt{\frac{2\log{\frac{2}{\delta}}}{n}}\\
&\leq K\sqrt{2\log{\frac{2}{\delta}}}\left(1+\int_1^\frac{T}{K}\frac{1}{\sqrt{x}}dx\right)\\
&\leq K\sqrt{2\log{\frac{2}{\delta}}}\times2\sqrt{\frac{T}{K}}\\
&=2\sqrt{2KT\log{\frac{2}{\delta}}}?
\end{aligned}
$$

와 같이 되기를 기대할 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;아래는 수학적으로 그것이 사실임을 보이는 증명입니다. 위에서는 직관적으로 T개의 선택이 K개의 선택지에 고르게 분포되어야 한다는 가정은 부분은 코시 슈바르츠 부등식에 의해 정당화 될 수 있습니다. 세 번째 줄에서 네 번째 줄로 넘어갈 때에 코시 슈바르츠 부등식이 사용되었습니다.

$$
\begin{aligned}
R&=K+\sum_{t=K+1}^{T}\sqrt{\frac{2\log{\frac{2}{\delta}}}{N_{a_t, t-1}}}\\
&\leq K+\sum_{a=1}^{K}\sum_{n=1}^{N_{a, T}-1} \sqrt{\frac{2\log{\frac{2}{\delta}}}{n}}\\
&\leq K+\sum_{a=1}^{K}\sqrt{2\log{\frac{2}{\delta}}}\times2\sqrt{N_{a, T+1}}\\
&\leq K+2\sqrt{\log{\frac{2}{\delta}}}\times\sqrt{2K\sum_{a=1}^{K}N_{a, T+1}}\\
&=K+2\sqrt{\log{\frac{2}{\delta}}}\times\sqrt{2K(T-K)}\\
&\leq K+2\sqrt{2KT\log{\frac{2}{\delta}}}
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;증명이야 어찌됐든, 가장 중요한 것은 마지막 줄입니다. 이로써 당신은 이제 아주 똑똑한 방식으로 버튼을 누른다면, 많아봐야 $O\left(\sqrt{KT\log{\frac{1}{\delta}}} + K\right)$의 손해 밖에 얻을 수 없다는 것을 알았습니다! 그렇지만 한 가지 불안 요소가 있습니다. 위 부등식은 사실 앞에서 언급했다시피 "모든 신뢰 구간이 유효한 경우에"만 성립한다는 것입니다! 만약 "모든 신뢰 구간이 유효한 경우"가 일어날 확률이 꽤 작다면 우리가 겨우 얻은 부등식은 높은 확률로 쓸모가 없다는 뜻입니다!

&nbsp;&nbsp;&nbsp;&nbsp;특정한 신뢰 구간 하나가 실패할 사건의 확률은 $\delta$이고 그런 신뢰 구간이 $TK$개 있으므로 신뢰 구간들이 하나라도 실패할 사건의 확률은 다행히도 아무리 커봐야 $TK\delta$라는 것을 알 수 있습니다. 이 간단한 원리는 union bound, 혹은 Boole's inequality라는 거창한 이름을 가지고 있습니다. 여기서 주의해야할 점은 각 신뢰 구간이 성립/실패할 가능성은 서로 독립이 아니기 때문에 단순히 확률을 곱해서 구할 수 없다는 것입니다. 

&nbsp;&nbsp;&nbsp;&nbsp;이제 우리는 위 부등식이 $1-TK\delta$의 확률로 성립한다는 것까지도 알았습니다. 만약 기호적으로 $1-\delta$로 위 부등식이 성립할 확률을 표현하고자 한다면 성립할 가능성이 $1-\delta/KT$ 이상인 신뢰 구간들, 즉 $[L(\delta/KT), U(\delta/KT)]$를 사용해야 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;결과적으로 위에서 $\delta$로 썼던 값들을 $\delta/KT$로 바꿔준다면 이 알고리즘은 $1-\delta$ 이상의 확률로 다음과 같은 regret bound를 만족합니다.

$$
\begin{aligned}
R\leq 2\sqrt{2KT\log{\frac{2KT}{\delta}}}+K
\end{aligned}
$$

&nbsp;&nbsp;&nbsp;&nbsp;기댓값이 작은 알고리즘을 얻기 바란다면 $\delta=K/T$로 설정해봅시다. 그러면

$$
\begin{aligned}
\mathbb{E}[R]&\leq (1-\delta)(K+2\sqrt{2KT\log{\frac{2KT}{\delta}}}) + \delta T\\
&\leq2\sqrt{2KT\log{2T}}+2K
\end{aligned}
$$

로, $\mathbb{E}[R]=O(\sqrt{KT\log{T}}+K)$를 얻을 수 있습니다. 당신은 안심하며 UCB 알고리즘이 가르쳐주는대로 버튼을 눌렀고, 큰 돈을 벌 수 있었습니다!!

# 마치며

&nbsp;&nbsp;&nbsp;&nbsp;Bandit 문제와 이를 해결하는 알고리즘은 강화 학습의 기본이 되는 알고리즘이며, 그 자체로도 상품 추천 시스템 등에서 활발하게 적용되고 있습니다. 실제로 바둑으로 최초로 인간을 이긴 인공지능 알파고에서도 어느 위치에 돌을 놓을지를 선택을 할 때에 UCB 알고리즘의 한 변형이 사용됩니다. 비록 버튼을 누르면 돈을 주는 기계는 상상 속의 이야기지만, 불확실성을 극복하는 문제는 기계 학습 뿐만 아니라 인생에서도 항상 마주하는 문제입니다. 이 글을 통해 불확실성을 극복하는 전략에 대해 공부가 되셨으면 좋겠습니다.

# Reference

[1] Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine
Learning Research, 3:397–422, 2002.