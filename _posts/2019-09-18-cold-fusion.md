---
layout: post
title: "Cold Fusion"
author: jihoon
date: 2019-09-18 13:59
tags: [sequence-to-sequence, fusion, natural-language-processing, machine-learning]
---


# Cold Fusion

Attention을 사용한 Seq2Seq 모델은 기계 번역이나 이미지 캡션, 그리고 음성 인식 등의 여러 자연어 처리 태스크에서 사용되고 있습니다. [Cold Fusion](https://arxiv.org/pdf/1708.06426.pdf)은 Seq2Seq 기반의 모델에서 성능 향상을 위해서 제안된 방법 중 하나로, 사전에 학습된 언어 모델을 함께 사용하여 성능을 높이는 것을 목적으로 합니다. 이 글에서는 먼저 Attention을 사용한 Seq2Seq 모델에 대해서 간략히 알아보고, 이 모델의 성능을 높이기 위한 semi-supervised learning 기법에 대해서 알아볼 것입니다. 마지막으로, Cold Fusion 이전에 제안되었던 Langauge model을 사용하는 방법들인 Shallow Fusion과 Deep Fusion에 대해서 알아본 후에 Cold Fusion에 대해서 다루도록 하겠습니다.  

# Sequence-to-Sequence model with Attention

들어가기에 앞서, 이 글에서는 seq2seq나 Attention을 직접적으로 설명하는 것에 초점을 두지 않고 있기 때문에 이 두 아이디어에 대해서는 간략히 설명하도록 하겠습니다.

[Seq2Seq](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)는 크게 인코더와 디코더로 이루어져 있고, 이 두 구조는 LSTM이나 GRU 등 RNN 계열 구조로 이루어져 있습니다. 먼저 인코더에서 입력을 순서대로 받고, RNN 계열의 구조를 통해 하나의 context vector를 출력합니다. 다음에 디코더가 context vector를 입력으로 받아서 sequence를 출력하는 구조입니다. 하지만, 인코더의 입력이 고정된 크기의 context vector로 압축되어, 디코더는 인코더로부터 context vector 정보만 받습니다. 이러한 병목 현상 때문에, vanishing gradient 문제를 해결한 LSTM 등의 구조를 사용하더라도, 입력 문장의 길이가 길어질수록 모델의 성능이 떨어지는 문제점을 가지고 있습니다.

![](\assets\images\cold-fusion\attention.png)

그래서 해결법으로 제안된 방법이 [Attention](https://arxiv.org/pdf/1409.0473.pdf)을 이용한 방법입니다. 디코더에서 출력을 결정하기 전에 인코더의 context vector 뿐만 아니라 모든 스텝의 hidden state를 함께 사용하여 attention vector를 만들고, 디코더에서의 hidden state와 함께 고려하여 출력할 sequence를 결정합니다. 


## Semi-supervised Learning

이 문단에서는 semi-supervised learning을 사용하여 Seq2seq 모델의 성능을 높이는 방법들인 Backtranslation과 Unsupervised pre-training에 대해서 알아보도록 하겠습니다. 

### Backtranslation

대부분의 Seq2Seq 모델은 적지 않은 양의 parallel corpus를 사용하여 학습을 진행합니다. 예를 들어 한영 번역이나 영한 번역을 하기 위한 Seq2Seq 모델을 학습하고 싶다면, 한국어 문장과 그에 대응되는 영어 문장으로 이루어진 데이터가 필요합니다. 그리고 이러한 쌍이 수십만 개는 되어야 제대로 된 번역 모델을 얻을 수 있습니다. 하지만, 모든 태스크에서 많은 양의 parallel corpus를 모을 수 있는 것은 아닙니다. 한국어와 다른 언어 간의 번역에 대해서 생각해보면, 한국어와 영어 간의 데이터에 비해 한국어와 다른 언어 간의 parallel corpus는 상대적으로 적을 수 밖에 없고 자연스럽게 번역 모델의 성능도 좋지 않을 수 밖에 없습니다.

이러한 문제를 해결하기 위한 방법 중 하나로 Backtranslation이 제안되었습니다. 다시 번역의 예시를 이용해서 설명하자면, A 언어와 B 언어 사이의 parallel corpus와 B 언어로만 이루어진 C라는 monolingual corpus가 있을 때, C를 B 언어에서 A 언어로 모델을 이용하여 번역한 다음 (이를 A'이라고 합시다) A'와 C로 이루어진 parallel corpus도 함께 학습에 사용하는 방식입니다.

이 방법을 적용하면 실제로 기계 번역에서는 성능(BLEU score)가 높아지지만, 이미지의 캡션을 구하거나 음성 인식 태스크에서는 큰 개선효과를 얻기 힘들다고 알려져 있습니다.



### Unsupervised pre-training

이 방법은 일종의 warm start를 사용하는 방법입니다. source domain의 언어 모델을 사용하여 인코더의 파라미터, target domain의 언어 모델을 사용하여 디코더의 파라미터를 초기에 설정하고 Seq2seq 모델의 학습을 시작하는 방식입니다. 이 방법을 사용하면 사전 지식을 알고 있는 상태에서 모델의 학습을 시작하므로 실제로도 성능이 좋아진다고 알려져 있습니다.  




## Fusion

![](\assets\images\cold-fusion\examples.png)

위의 사진은 실제 Seq2seq를 사용하는 음성 인식 태스크에서의 예시를 보여줍니다. 위의 예시에서 Plain Seq2seq를 사용한 모델은 정확하게 예측하지는 못하고 발음 상으로 비슷한 결과를 내고 있습니다. 이러한 점에서 볼 때 만약 언어 모델을 함께 사용할 수 있다면 더 좋은 성능을 보일 것이라고 기대할 수 있습니다. 또한 언어 모델은 Seq2seq 학습에 사용하는 parallel corpus에 비해서 더 많은 양의 데이터를 통해 학습을 진행할 수 있다는 장점도 가지고 있습니다. 그래서 언어 모델을 함께 사용하여 Seq2seq 모델의 성능을 높이는 연구들이 진행되었습니다.



## Shallow Fusion

![](\assets\images\cold-fusion\shallow.png)

Shallow Fusion은 언어 모델을 사용하는 방법 중 가장 간단한 방법입니다. 먼저 언어 모델과 Seq2seq 모델을 따로 학습시킵니다. 그 후 디코딩을 할 때 언어 모델에서의 probability와 Seq2seq 모델에서의 probability를 서로 선형 결합하여 결과적으로 probability가 가장 높은 것을 선택합니다.



## Deep Fusion

![](\assets\images\cold-fusion\deep.png)

Deep Fusion은 Shallow Fusion의 성능을 높인 방법입니다. Shallow Fusion과 마찬가지로 언어 모델과 Seq2seq 모델을 따로 학습시키지만, 단순히 선형 결합을 하지는 않고 조금 더 복잡한, gate를 사용한 수식을 통해서 Fusion을 진행합니다.



## Cold Fusion

Shallow Fusion과 Deep Fusion을 사용하면 모델의 성능을 높일 수 있지만, 여전히 두 가지의 단점이 존재합니다.

- 언어 모델을 사용하지만, 여전히 Seq2seq를 학습할 때에는 언어 모델을 사용하지 않으므로 Seq2seq 내부에서 implicit한 언어 모델의 학습이 필요합니다. 당연히 이 implicit한 언어 모델로 인해서 디코더의 capacity의 일부가 사용되므로, 태스크 자체를 학습하는 capacity는 줄어들게 됩니다.
- 이러한 implicit 언어 모델은 학습에 사용한 데이터에 대해서 편향되어서 학습이 이루어집니다. 그러므로 다른 도메인에서 디코딩이 이루어진다면, overfitting으로 인해서 좋지 않은 성능을 보일 것입니다.

![](\assets\images\cold-fusion\cold.png)

이러한 문제점을 해결하기 위해서, cold fusion 방법은 언어 모델과 Seq2seq 모델을 따로 학습시키지 않습니다. 대신 Seq2seq 모델을 학습할 때 사전에 학습된 언어 모델을 함께 사용합니다. 그래서 입력이 specific하거나 noisy한 경우에 언어 모델을 좀 더 참조하도록 할 수 있게 되어 Seq2seq 모델의 디코더는 task를 해결하는 데에 좀 더 집중할 수 있도록 합니다. 즉, Cold fusion 방법은 언어 모델을 사용하는 방법을 학습한다고 볼 수도 있습니다. 또한 [fine-grained gating mechanism](https://arxiv.org/pdf/1611.01724.pdf)을 사용하여 언어 모델을 함께 사용할 때에 flexibility를 높였다고 서술하고 있습니다. 

![](\assets\images\cold-fusion\result1.png)

위의 실험 결과에서도 알 수 있듯이 Cold Fusion을 사용했을 때, 기본 Attention을 사용한 모델이나 Deep Fusion을 사용한 모델에 비해서 더 좋은 결과를 보이고 있습니다. 또한 실험에서 ReLU layer를 softmax 전에 사용했을 때, 성능이 좋아졌다고 서술하고 있습니다. 



# 마무리

지금까지 Seq2seq 모델과 모델의 성능을 향상시킬 수 있는 방법으로 Attention, Semi-supervised Learning, 그리고 언어 모델과의 Fusion을 사용하는 방법에 대해서 알아보았습니다. 수학이 필요한 내용들에 대해서는 최대한 자세한 설명을 하지는 않았는데, 혹시 더 궁금한 내용이 있다면 Reference에 논문 링크를 적어놓았으니 참고하시면 될 것 같습니다.



# Reference

- [http://ezeerway.blogspot.com/2019/06/component-fusion-learning-replaceable.html](http://ezeerway.blogspot.com/2019/06/component-fusion-learning-replaceable.html)
- [Sutskever, Ilya, Vinyals, Oriol, and Le, Quoc V. Sequence to sequence learning with neural networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- [Bahdanau, Dzmitry, Cho, Kyunghyun, and Bengio, Yoshua. Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Yang, Zhilin, Dhingra, Bhuwan, Yuan, Ye, Hu, Junjie, Cohen, William W, and Salakhutdinov, Ruslan. Words or characters? fine-grained gating for reading comprehension](https://arxiv.org/abs/1611.01724)
- [Anuroop Sriram, Heewoo Jun, Sanjeev Satheesh, and Adam Coates. Cold Fusion: Training Seq2Seq Models Together with Language Models](https://arxiv.org/pdf/1708.06426.pdf)
- [Changhao Shan et al. Component Fusion: Learning Replaceable Language Model Component for End-to-end Speech Recognition System](http://lxie.nwpu-aslp.org/papers/2019ICASSP-ChanghaoShan-LM.pdf)
