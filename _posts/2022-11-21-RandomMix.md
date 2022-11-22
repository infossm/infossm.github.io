---
layout: post
title:  "RandomMix: A mixed sample data augmentation method with multiple mixed modes"
date:   2022-11-21 08:00:00
author: VennTum
tags: [AI, deep-learning]
---

# [RandomMix: A mixed sample data augmentation method with multiple mixed modes](https://arxiv.org/abs/2205.08728)

RandomMix는 2022년도 5월 난징대에서 연구하여 arxiv에 공개된 data augmentation 논문입니다. 꽤나 최근에 나온 논문으로, 논문 자체의 내용이 크게 어렵지 않으면서도 지금까지 발표된 여러가지 mixed sample data augmentation들에 비해 높은 성능을 보여 SOTA를 달성했습니다. 뿐만 아니라 이미지의 robustness, diversity, cost의 관점에서도 좋은 향상을 보여주어 살펴볼 필요가 있는 논문입니다.

들어가기에 앞서, 본 글은 해당 논문을 그대로 번역하는 것이 아닌 관련된 다른 논문들의 설명을 추가하며 RandomMix 및 여러 data augmentation과의 비교 및 동향등을 함께 정리한다는 점을 말씀드립니다.

# Abstract

Data augmentation 기법은 뉴럴 네트워크의 generalization을 높여주어 학습 과정에서 학습 데이터 세트에 overfitting하는 경우를 막아주는 역할을 한다는 것이 실험적으로 보여져왔습니다. 특히, Mixup의 등장으로 두 가지 이상의 이미지를 섞어내는 Mixed Sample Data Augmentation(MSDA)가 화두가 되었으며, 굉장히 높은 성능을 보여 주목을 끌었습니다. 이 과정에서 두 이미지를 라벨 비율대로 섞는 Mixup(2017), source image에 target image를 섞는 Cutmix(2019) 등이 굉장히 큰 주목을 받아왔습니다(Mixup, Cutmix에 대한 간단한 설명은 이전에 포스팅한 SaliencyMix에서 확인하실 수 있습니다).

이러한 MSDA 기법의 성능을 향상하기 위해서 최근 논문들에서는 이미지의 saliency region을 확인하고, 해당 영역들을 mixed image에 최대한 남기려는 형태의 노력이 많이 이루어졌습니다. 이전에 제가 소개했던 SaliencyMix도 같은 이유로 saliency area를 남기는 형태로 mixing하고, cutmix의 label mix 방식을 따르는 방식이었죠.
최근에는 Saliency Map을 이용하는 것 뿐만 아니라, Class Activation Map(CAM)을 활용하는 형태의 논문이 나오기도 했습니다(SnapMix, 2021). 결국에 이러한 시도들도 모두 기존에 제시되었던 MSDA 기법들의 label mixing 및 image mixing에서 납득되지 않는 mixing 결과를 개선하기 위한 형태로 적용된 것으로 볼 수 있습니다.

그러나 이 논문은 MSDA의 최근 동향인 saliency information을 이용하는 형태의 기법들의 cost에 주목합니다.
기존의 Mixup의 경우는 두 이미지를 얼마의 label 비율로 섞을 것인지에 대한 lambda를 beta distribution을 통해 구하기만 하면 바로 mixing이 가능해서 이미지 크기만큼의 시간으로 바로 구할 수 있어 전체 학습 시간에 영향을 거의 미치지 않았습니다.
마찬가지로, CutMix의 경우도 주어진 target image를 어느 영역을 사용할 것인지, source image도 어느 영역을 사용할 것인지에 대한 random한 좌표만 정해지고나면 mixing에서는 큰 시간이 들지 않아서, 적용하더라도 학습 시간이 달라지지는 않는 장점이 있었죠.

그러나 최근에 등장한 기법들은 달랐습니다. 기본적으로 Saliency information을 구하기 위해서 pre-trained된 모델을 사용하여 주어진 이미지의 saliency map을 구하고, 이 중 어떤 영역이 saliency한지 saliency area를 찾아내야 했습니다. 이전에 다루었던 SaliencyMix 또한 이를 구해야하는 점으로 인해 기존의 방법들보다 time efficiency가 조금 떨어진다는 점이 논문에 나와있었습니다.

![](/assets/images/VennTum/data_augmentation/randommix_1.png)
<center>SaliencyMix Table 5 - 학습에 걸린 시간 비교</center>

그러나 SaliencyMix는 Saliency Map을 구하는 과정에만 시간을 조금 사용하고, 이후에는 주어진 Saliency Map에서 가장 높은 값을 가지는 위치를 argmax를 통해 얻어내어, 해당 좌표를 이용하는 CutMix와 동일하게 진행되기 때문에, 다른 시간은 많이 필요하지 않아서 꽤나 빠른 편에 속합니다.

더욱 높은 성능을 보여주면서 Saliency information을 사용하는 논문들로는 PuzzleMix(ICML 2020)와 Co-Mixup(ICLR 2021 Oral) 등의 논문이 있습니다. 위 논문들은 그냥 Saliency Map을 구하여 해당 정보를 사용하는 것 뿐만 아닌, image를 mixing하는 과정에서 saliency area의 손실, 영역의 겹침 등을 고려하여 최대한 saliency area를 살리면서 새로운 mixed image를 생성합니다.
또한 Co-Mixup의 경우, 기존의 방법들처럼 두 개의 이미지를 섞는 방식이 아닌, 3개 이상의 이미지를 섞을 때에도 saliency information을 고려하면서 섞을 수 있는 알고리즘을 고안하였고, 그 결과 앞선 여러 MSDA 기법들에 비해서 뚜렷한 성능 향상을 보여주며 해당 시점의 SOTA를 찍어 ICLR 2021의 Oral presentation으로 선정되는 결과를 보여주기도 하였습니다.
(두 논문의 자세한 동작 원리 및 contribution은 더 있으나, 지금은 Saliency information을 활용한 기법이라는 점에 초점을 맞추도록 하겠습니다. 자세한 내용은 논문을 살펴보시면 좋습니다)

그러나 해당 논문들은 이미지를 섞는 과정에서 정보의 손실을 최대한 줄이기 위해 많은 time cost가 사용된다는 단점이 있습니다(실제로 해당 기법들을 사용해보았을 때, PuzzleMix는 Mixup 대비 약 2배 정도, Co-mixup은 2~3배 정도의 시간이 소요되었습니다). 물론 Co-Mixup 등의 논문에서도 time efficiency가 떨어진다는 점을 인지하고, 해당 알고리즘을 brute-force하게 구현하는 것보다는 훨씬 더 빠르도록 optimize를 거쳐 소요 시간을 줄였으나 단순한 형태의 mixing보다는 느릴 수 밖에 없었습니다.

이러한 관점들에서 최근에는 cost efficiency를 추구하는 형태의 여러 논문들이 게시되기도 하였으나, 꽤나 많은 수의 논문들이 기존의 mixup, cutmix 혹은 단순한 형태의 enhanced MSDA보다는 성능이 높으면서 빠르게 동작하는 것을 보여주고, 자신들이 가진 논문의 contribution으로 성능의 SOTA보다는 이외의 것들에 초점을 맞추어 설명하는 경우들이 있었습니다. 즉, Co-Mixup보다도 더 성능이 좋으면서 빠른 논문들을 찾아보기가 쉽지는 않았습니다.

이러한 상황 속에서, 저자들은 자신들이 기존의 Saliency information을 사용하는 논문들에 비해 굉장히 빠르게 동작하면서도, accuracy, robustness, diversity 등 여러 방면에서 기존보다 효율적인 RandomMix를 제안합니다. 심지어 논문에서 말하는 table만 확인하면, 기존의 PuzzleMix, 심지어 Co-Mixup보다도 더 성능이 뛰어나다고 이야기합니다.
더욱이 RandomMix는 이미지를 mixing하는 파이프라인이 굉장히 간단하여 어떤 모델들에서도 매우 쉽게 적용할 수 있다는 강점을 가지고 있음을 말합니다.

# Method

기존의 Mixup의 경우, 두 개의 source image와 target image를 준비하여, beta distribution에서 얻은 lambda 값에 비례하여 각각의 이미지를 pixel-wise하게 비율대로 섞는 방식으로 구현됩니다.

그리고 CutMix의 경우, 이러한 Mixup의 방식과 특정 영역을 모두 black으로 바꿔버리는 Cutout을 합친 방식으로, target image의 영역을 source image의 랜덤한 위치에 삽입하는 방식으로 새로운 이미지를 만들어냅니다.

그리고 여기서 CutMix가 가진 단점을 보완하는 논문인 ResizeMix(2020), FMix(2021)에 대해서도 알아야합니다.

사실 RandomMix가 가지는 가장 큰 contribution을 이해하는 것에는 위 논문들을 정확하게 이해해야 하는 것과는 거리가 있기 때문에, 간단히 설명하고 넘어가도록 하겠습니다.

## [ResizeMix](https://arxiv.org/abs/2012.11101)

ResizeMix는 CutMix가 가진 가장 큰 단점을 해결하려는 노력에서부터 아이디어가 시작합니다. 이전 SaliencyMix의 경우, CutMix를 개선할 때에 기여한 가장 큰 contibution은 다음과 같습니다.

- 기존의 CutMix의 경우, target image에서 엉뚱한 영역을 잘라서 source image에 붙일 경우, 의미없는 이미지 mixing이 일어남과 동시에 label ratio도 망가진다.

SaliencyMix는 이 점을 target image에서 Saliency한 영역을 찾아서 이미지를 잘라내는 방식으로 CutMix의 단점을 개선하였습니다.

그러나 ResizeMix는 CutMix를 진행하는 과정에서, target image에서 잘못 이미지를 잘라내는 경우를 방지하기 위해서, source image의 Cutout된 영역 자체에 target image 자체를 해당 크기대로 resize하여 붙여넣는 방식을 제안합니다.

이 과정에서 ResizeMix는 기존의 MSDA 기법들이 중시하던 Saliency information을 활용한다는 것이 과연 중요한 것일지부터 확인하면서 시작합니다. 기존의 문제가 되었던 부분들을 실제로 다음과 같은 영역들로 구분하면서 실험을 진행합니다.

source image/target image의 patch를 선정하는 과정에서

- Non-saliency region
- Saliency region
- Random region

이렇게 3가지 경우로 나누어서 각각의 경우들을 매칭하면서 결과를 확인합니다. 이를 통해 얻은 결과는 우리의 예상과는 다르게도, random한 영역의 target patch를 random한 영역의 source patch에 붙여넣을 때가 가장 성능이 좋다는 것을 보여주면서, saliency information을 활용하는 것이 크게 의미없다는 것을 실험적으로 보여줍니다.

![](/assets/images/VennTum/data_augmentation/randommix_2.png)
<center>ResizeMix Table 1 - CIFAR-100에서 Salient region의 중요성 실험 결과</center>

이를 통해, ResizeMix의 저자들은 Saliency information을 활용하려는 시도들이 MSDA에서 크게 중요하지 않다는 것을 주장합니다.
그러나 실험 결과에서, Non-saliency region을 사용하게 되는 경우는 label의 misallocation을 유발하게 되어, 결과적으로 no labeled object에 label을 부여하는 경우가 발생할 수 있어서 결과에 악영향을 줄 수 있다는 점을 이야기합니다.

그 결과, CutMix를 진행하되, 정보가 없는 영역이 mixing되는 것을 예방하면서 굳이 saliency area를 선정하지 않아도 된다는 점을 착안해, target image를 그대로 resize하여 모든 정보를 source image patch에 넣어주는 방식인 ResizeMix를 제안합니다.

실제 실험 결과에서 ResizeMix는 SaliencyMix, PuzzleMix보다도 더 높은 Top-1 accuracy를 보이면서 성능이 향상되었다는 점을 저자들은 주장합니다.

![](/assets/images/VennTum/data_augmentation/randommix_3.png)
<center>ResizeMix Table 3,4 - CIFAR-100/ImageNet에서 ResizeMix Top-1 accuracy</center>

## [FMix](https://arxiv.org/abs/2002.12047)

FMix는 2020년도에 arxiv에 등록된 논문입니다.
사실 FMix는 그 자체를 모두 이야기하기에는 다루어야하는 내용이 꽤나 많은 논문입니다. 저자들의 기존의 MSDA에 대한 문제 인식 및 동기, 그리고 contribution에 대해서 이야기할 거리가 많습니다.
그러나 본 포스트에서는 RandomMix에서 사용하는 기법 중 하나라는 것 이외에는 FMix에 대해서 크게 다루지는 않을 예정입니다. 그 이유는 제가 처음 FMix를 읽을 때에도 의심되는 부분이었으며, 결과적으로 FMix가 ICLR 2021 blind submission에서 reject 되기도 한 이유인 성능의 개선 및 결과 분석의 신빙성 때문입니다. 물론 FMix가 가지는 contribution이 굉장히 중요하고 큰 의미를 가질 수도 있습니다. 그러나 많은 리뷰어들이 성능의 개선이 크지 않다는 점, 방법론과 결과 분석의 논리적 연결이 약하다는 점을 의심했고, 결과적으로 분석 과정의 근거 및 설득력에 의구심을 가져 reject를 받았습니다. 물론 아닐 수도 있으나, 검증이 크게 되지 않고 성능 또한 크게 개선되지 않은 논문을 다루는 것이 큰 의미가 있을지에 대해 확신할 수 없기 때문에, FMix의 방법론 정도만 다룬 후 넘어가겠습니다.

FMix 논문은 결과적으로, CutMix의 경우 cutting할 image patch를 결정할 때 정사각형 혹은 직사각형의 모양으로 잘라서 붙여넣게 됩니다. 그러나 FMix의 저자들은 CutMix의 기존 방식처럼 사각형 모양으로 자르는 것이 아닌 임의의 mask를 만들어서 해당 mask대로 source와 target image를 mixing하는 방법을 제안합니다.

이 때, 만들게 되는 임의의 마스크는 Fourier 공간에서 샘플링된 low frequency image에 임계값을 적용하여 얻는 임의의 이진 마스크를 사용합니다.
이러한 형태의 마스크는 결과적으로 다양한 형태로 만들어지고 데이터의 차원에 상관없이 적용 가능하다는 장점과 해당 마스크를 구하는 과정에서 특별한 time cost가 발생하지 않는다는 장점이 있어, 매 훈련마다 다양한 형태의 이진 마스크를 만들어서 여러가지 adversarial attack에 대해 강건성을 가질 수 있다고 이야기합니다.

이러한 접근은 기존의 네모난 형태의 mask를smooth하게 만들어서 모델의 robustness를 증가시키려는 노력을 한 SmoothMix과 비슷하게 느껴질 수도 있습니다.

결과적으로 FMix의 형태가 어떤 식으로 합성되는지 이미지를 보여드리는 것으로 넘어가겠습니다.

![](/assets/images/VennTum/data_augmentation/randommix_4.png)
<center>FMix - examples</center>

## RandomMix

이제 대방의 RandomMix가 적용되는 방법론에 대한 이야기를 할 차례입니다. 사실 어떻게 보면, 제가 확인한 바에 의하면 RandomMix의 경우도 conference 등에 제출되거나 peer review를 받은 상태가 아닌 것으로 알고 있습니다. 그 이야기는 곧, 과연 이 논문에서 다루는 내용의 증거 및 신빙성과도 연결될 수 있다고 볼 수 있겠습니다.

그러나 해당 논문에서 기여하는 알고리즘 자체가 굉장히 단순하면서도 결과적으로 지금까지 보여준 여러 높은 성능의 논문들보다 훨씬 좋은 결과를 보여주고 있습니다. 비단 accuracy 뿐만 아니라, robustness, time cost, diversity 등 다양한 관점에서 굉장히 좋은 성능을 보여주면서 기존의 MSDA 논문들의 연구 방향성을 다시 고려해볼 수 있도록 해주는 논문이기에 해당 내용을 알려드릴 필요성이 있다고 생각합니다.

RandomMix의 목적은 모델이 adversarial attack들에 대한 robustness를 가지고, 동시에 training 과정에서 우리가 앞서서 이야기했던 Mixup, CutMix, ResizeMix, FMix 총 4가지의 방법들을 통해 만들어진 이미지 데이터 셋을 사용하는 걸 통해 훈련 데이터의 diversity를 증가시키겠다는 것입니다.

그 과정은 다음과 같습니다.

- RandomMix를 하는 과정에서, Mixup, CutMix, ResizeMix, FMix를 확률적으로 선택하고 선택된 data augmentation을 적용한다.

굉장히 심플하고도 간단합니다. 물론 이 과정에서 각각의 augmentation을 선택할 확률을 어떻게 주냐는 문제를 고민할 수 있겠으나, RandomMix 논문에서는 각각의 weight를 모두 1로 주었습니다. 이를 통해 random하게 선택된 augmentation을 적용하면서 학습 과정에서 이미지들에 서로 다른 data augmentation을 확률적으로 적용하겠다는 것이 RandomMix의 목표입니다.

이를 통해 더욱 다양한 형태의 train image들이 만들어지면서 diversity가 증가하게 되는 결과를 낳고, 이를 통해 robustness와 accuracy의 향상도 기대하면서 RandomMix의 실험이 진행됩니다. 또한 저자들이 선택한 4개의 방법론 모두 time cost가 거의 없기 때문에, RandomMix의 경우도 굉장히 빠르게 동작한다는 것을 알 수 있고, 특별히 계산하거나 적용해야하는 제한 조건이 없기 때문에, 어떠한 모델과 파이프라인에도 적용할 수 있다는 장점이 있습니다.

이제 이러한 기법을 적용한 RandomMix의 결과를 살펴보겠습니다.


![](/assets/images/VennTum/data_augmentation/randommix_5.png)
<center>RandomMix Figure1 - RandomMix examples</center>

## Mixup

Mixup의 경우, 랜덤으로 선택된 두 개의 이미지를 beta distribution에 따라 선택된 비율 $\lambda$ 에 따라 두 이미지를 $\lambda$ 와 $(1-\lambda)$ 만큼 pixel-wise하게 섞어 새로운 이미지를 만들어 냅니다. 이렇게 만들어진 이미지의 라벨은 기존의 하나의 이미지의 라벨을 갖는 것이 아닌, 사용된 두 개의 이미지의 라벨을 해당 비율만큼 가지게 됩니다.
이렇게 만들어진 이미지는 모델이 training data들 사이의 linear behavior를 가지게 해주어 overfitting과 adversarial data에 대한 robustness를 가지게 된다는 장점이 있습니다.

![](/assets/images/VennTum/data_augmentation/mixup_1.png)

<center>고양이와 배 이미지 Mixup</center>

## Cutout

Cutout의 경우는 두 개의 이미지를 사용하는 기법은 아닙니다. 많은 기존의 data augmentation 기법들처럼 기존에 존재하는 training data에 변형을 주는 방식으로 새로운 data를 만들어내게 됩니다. 그 아이디어는 상당히 간단한데, 바로 기존에 존재하는 이미지의 특정 영역을 아예 제거해버리는 방법으로 새로운 이미지를 만들어내게 됩니다. 주어진 영역 중 어떤 영역을 선택하고 어떤 모양으로 선택하여 제거할 것인가는 논문에 여러 다양한 방법이 나와 있어 확인해보실 수 있습니다(결과적으로는 '영역의 크기' 이외에는 결과에 큰 차이가 없기는 합니다).
Cutout이 가지는 의미는 이를 사용하여 학습을 진행한 모델의 activation 영역이 커진다는 점입니다. 즉, 주어진 이미지에서 라벨을 분류하기 위해 이미지에서 더 많은 feature들을 찾아내고 이를 사용하게 된다는 것입니다. 이는 cutout을 해석하는 과정에서 가장 크게 영향을 미치는 점입니다.
mixup을 포함하여 최근에 많이 연구되는 data augmentation들이 많이 개선을 목표로 했던 점은 바로 모델이 이미지의 특정 영역, 즉, 가장 영향을 크게 미치는 feature에 집중하여 학습을 진행한다는 점이었습니다. 학습하는 과정에서 중요한 feature들에 집중하여 이를 학습하게 되는 경향이 있는데, 이는 조금 덜 중요한 feature들의 경우 덜 집중되거나 무시되는 경향을 만들어내고 결국 주어진 training data에 overfit하게 되며 robustness가 떨어지는 결과를 낳게 됩니다.

이 과정에서 cutout은 주어진 이미지의 특정 영역을 제거하여, 해당 영역이 가지고 있던 feature를 이미지 상에서 제거하는 역할을 하게 됩니다. 즉, 이 과정에서 모델이 집중하던 feature가 이미지에서 제거되었다면, 나머지 영역들에 있는 feature들에 집중하는 역할을 하게 되며, 이 과정에서 기존 이미지에서는 집중하지 않았던 feature들을 찾아 모델이 학습할 수 있는 결과를 낳게 되어 성능의 향상을 만들어 냅니다.

![](/assets/images/VennTum/data_augmentation/cutout_1.png)

<center>고양이의 특정 영역 Cutout</center>

## CutMix

CutMix는 Cutout을 기반으로 두고 있습니다. 기존의 Cutout의 경우, 특별한 이미지 영역을 선택하고 제거하기 때문에, 그 과정에서 해당 영역은 아예 아무런 정보가 없게 되어 정보의 손실을 일으키게 됩니다. 이를 해결하기 위해 cutout으로 제거한 영역을 training data의 또 다른 이미지 하나를 선택하고, 해당 영역만큼 다른 이미지의 영역으로 대체하여 2개의 이미지를 섞으려는 시도가 바로 CutMix 기법입니다. 이렇게 만들어진 새로운 이미지의 경우, 합쳐진 이미지에서 각각의 원본의 이미지가 차지하고 있는 영역의 크기의 비에 해당하는 값을 라벨로 가지게 됩니다. 즉, 7:3의 비율로 a와 b 이미지를 cutmix를 통해 섞게 되었다면, 해당 이미지는 0.7a + 0.3b 만큼의 각각의 이미지 라벨을 가지게 됩니다.

기존의 mixup의 경우, 두 이미지를 $\lambda$ 비율로 섞는 과정에서 원본 이미지의 객체가 다른 이미지로 덮이면서 실제 이미지의 왜곡이 일어나는 경우가 발생하게 되는 문제점이 있었습니다. 그러나 CutMix의 경우, 각각의 이미지가 존재하는 영역은 다른 이미지의 간섭을 받지 않기 때문에 각 이미지의 객체가 가지는 정보를 그대로 유지할 수 있게 됩니다.

실제로 cutmix를 통해 생성된 이미지에서 각각의 이미지의 object를 잘 확인할 수 있는지 확인하기 위해 [CAM(Class Activation Map)](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)을 통해 확인할 때에, 섞인 이미지의 각각의 영역에서 object를 상당히 잘 구분해낸다는 것을 알 수 있습니다. 이 과정에서 모델은 각각의 object에 해당하는 feature에서 자기가 속한 이미지에서 제외된 영역만큼의 feature가 제거된 이미지가 되고, 이 과정은 해당 영역만큼 제거된 다른 영역들의 feature 집중할 수 있도록 하는 역할을 하게 됩니다.

![](/assets/images/VennTum/data_augmentation/cutmix_1.png)

<center>고양이와 배 이미지 CutMix</center>

Cutmix는 여러가지 benchmark dataset들에 대해서 기존의 mixup과 cutout보다 더 나은 결과를 보이며 좋은 성능을 내는 data augmentation 기법이 되었습니다. 

# [SaliencyMix (2020)](https://arxiv.org/abs/2006.01791)

SaliencyMix는 이러한 CutMix를 기반으로 하고 있습니다. Saliencymix는 Cutmix가 가지고 있는 단점을 saliency map을 사용해서 해결하려는 노력에서부터 시작합니다.

기존의 cutmix의 경우, 두 개의 이미지를 특정 영역을 선택하여 선택하여 섞는 과정을 통해 만들어지는 이미지에서, cutout된 영역은 다른 이미지에서 해당 cutout된 영역이 차지하고 있던 영역을 그대로 복사하여 가지고 오게 됩니다.
이러다 보니, 다음과 같은 문제가 발생하게 됩니다.

- 선택된 영역을 채우는 이미지에 객체가 존재하지 않는 경우에도, 새롭게 생성된 이미지의 라벨은 영역의 비 만큼의 각 이미지의 라벨을 가지게 된다.

앞선 CutMix에 사용된 이미지를 살펴보도록 하겠습니다.

![](/assets/images/VennTum/data_augmentation/cutmix_1.png)

해당 이미지는 고양이외 배를 섞은 이미지가 되지만, 실제 cutmix를 통해 합성되는 이미지에서 배가 차지하는 영역은 해당 영역의 크기보다 훨씬 더 작다는 것을 알 수 있습니다. 이보다 더한 경우, 만약 cutmix를 진행하는 과정에서 이미지 라벨에 해당하는 객체가 선택되지 않고 배경만 합성된다면, 이렇게 생성된 이미지가 올바른 라벨을 가지게 되는지에 대한 의문에서 부터 SaliencyMix는 시작하게 됩니다.

SaliencyMix는 cutmix를 적용하는 과정에서, source image에서 특정 영역을 복사하여 target image에 붙여넣는 과정에서, source image 내에 object가 존재할 가능성이 높은 영역을 선택하여 이를 target image에 붙여넣어 이러한 문제를 해결하려 시도합니다. 이에 만약 해당 과정이 잘 이루어지게 된다면, 아래와 같이 고양이와 배를 mix하는 과정에서 배 object 영역을 잘 찾아내어 고양이 이미지에 붙여넣을 수 있습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_1.png)

<center>고양이와 배 이미지 SaliencyMix</center>

## Saliency Map

앞서 saliencymix의 목표를 이루기 위해서, 우리는 특정 이미지 내에서 실제 label에 해당하는 object가 어디에 있는지 찾아내는 것이 중요합니다. 하지만 만약 우리가 특정 label에 해당하는 object가 어디에 있는지 알아낼 수 있다면 쉽겠지만 실제로는 그렇게 쉽지 않습니다. 왜냐하면 이 문제를 해결하는 것 자체가 바로 image segmentation이 되기 때문입니다. Image classfication을 위해 이보다 더 어려운 image segmentation을 사용하는 것은 쉽지 않은 일이 됩니다.

그래서 saliencymix에서는 이와 유사한 방법을 적용하기 위해 saliency map을 사용하게 됩니다.

saliency map이란 물체에서 시각적으로 중요한 영역, 객체에 해당하는 지점들을 찾아서 어떠한 영역들이 시각적으로 중요한 영향을 미치는지 확인하기 위해서 도입된 아이디어입니다.
이미지를 확인하는 과정에서, 사람은 모든 픽셀들을 확인하는 것이 아닌, 이미지에서 중요한 영역들을 찾아서 먼저 확인하게 됩니다. 이처럼, saliency map은 이미지에서 중요한 영역들이 무엇인지를 각각의 pixel들에 대해 score를 매겨 이를 visualize 하여 보여줍니다.

![](/assets/images/VennTum/data_augmentation/saliencymap_2.png)

<center>cv2를 통해 구한 배 이미지의 SaliencyMap</center>

이러한 Saliency Map을 구하는 방법은 크게 두 가지가 존재합니다. 바로 Bottom-up approach와 Top-down approch입니다.
각각에 대해 간단히 설명 하고 진행하도록 하겠습니다.

Bottom-up approch의 경우, saliency map을 구하기 위해 고전적으로 사용해오던 unsupervised한 approch입니다. 특정 이미지에서 어떠한 영역이 중요한 영역인지 판단하는 과정을 다른 영역에 비해 pixel 값이 급격하게 변화가 일어나는 위치(밝기, 색상의 변경, 대비의 발생 등)를 찾고 이들을 사용하여 mapping을 진행하게 됩니다. 즉, 이를 위해서 해당 이미지가 어떠한 이미지인지, 어떠한 object를 포함하고 있는지에 대한 정보가 필요없이, 고전적인 계산 기법들을 통해 각 이미지 pixel들의 saliency score를 계산하여 구하게 됩니다. 이로 인해 이미지의 종류에 상관없이 사용할 수 있으며 이미지의 크기 등에 대해서도 robust한 결과를 보입니다.

Top-down approach의 경우, 실제 label을 가진 training data를 통해 neural network를 학습시켜서 만들어내는 supervised-learning을 사용합니다. saliency detection에 대한 task를 해결하는 model을 만들어, 해당 NN을 이미 label이 분류되어 있는 dataset을 통해 학습을 진행합니다. 이렇게 만들어진 model의 경우, 특정 이미지에서 이미 학습을 진행했던 label에 대한 saliency detection을 model의 정확도만큼 잘 수행할 수 있게 됩니다. supervised-learning을 통해 학습 통해 pre-trained 된 model을 사용하기 때문에 기존의 bottom-up approach에 비해 성능이 더 좋은 경우가 많습니다.

Saliencymix에서는 이러한 두 가지 중 bottom-up approach를 사용합니다. Top-down approach의 경우 특정 label들에 대해 학습을 진행해야 하기 때문에, 학습하는 과정에서 사용한 dataset에 대해 더 biased 된 경향을 보일 수 있어, 해당 논문에서는 label에 대한 정보를 알 수 없는 unseen image들에 대해서도 보편적으로 generalize하게 사용할 수 있는 bottom-up approch를 사용합니다.

실제로 해당 논문에서는 OpenCV의 cv2에 내장되어있는 saliency map을 구하는 함수(cv2.saliency)를 통해 saliency map을 구합니다.

(물론 해당 논문에서 bottom-up approch를 사용한 것이지, top-down approch를 통해 saliency map을 구하는 경우도 있습니다. 이에 대한 예시로 [puzzlemix (2020)](https://arxiv.org/abs/2009.06962)가 있습니다)

## BBox(Bounding Box)

SaliencyMix는 이렇게 source image에 대한 saliency map을 구하고, 여기에서 bbox를 찾아내어 해당 영역을 target image에 붙여넣는 방식으로 mix를 진행합니다. 이 때, bbox를 구하는 방법은 saliency map에서 가장 큰 interest score를 가진 pixel의 위치를 구한 이후, 해당 위치를 중심으로 하는 특정 크기의 정사각형 영역을 선택하는 방식으로 구하게 됩니다.

bbox의 크기를 선택하는 과정은 beta distribution에 따르는 특정한 랜덤 값 $\lambda$ 에 대해 다음과 같이 구하게 됩니다.

- cut_W = sqrt(1 - $\lambda$) * W
- cut_H = sqrt(1 - $\lambda$) * H

beta distribution에 사용하는 beta는 사용자가 직접 넣는 hyper parameter에 의해 결정됩니다.

이렇게 구한 bbox가 실제 이미지의 크기 범위 밖으로 벗어나는 경우, 벗어난 영역만큼 반대로 bbox를 밀어서 이미지 영역을 선택하게 됩니다.
(해당 과정은 saliencymix code를 읽어보시면 수월하게 이해할 수 있습니다)

## Experiment Result

해당 논문에서는 이렇게 만들어진 saliency mix를 다양한 computer vision task에 적용하여 결과의 향상을 확인하였습니다.

### Image Classification

Image classification의 경우, Resnet, WideResNet에 해당하는 model들을 사용하여 학습을 진행하여 CIFAR-10, CIFAR-100, ImageNet 등의 image classfication을 위한 benchmark dataset들을 통해 top-1, top-5 error를 확인하였습니다.

본 포스팅에서는 대략적인 경향과 결과에 대한 설명을 진행할 예정이며, 실험에 사용된 자세한 셋팅 조건은 논문을 참고해주시면 되겠습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_table_1.PNG)

<center>CIFAR dataset에 대한 여러 data augmentation image classificatino result</center>

위 테이블에서 확인할 수 있듯, CIFAR image dataset에서 saliency mix는 실제로 기존의 cutout과 cutmix에 비해서 더욱 개선된 top-1 error를 보임을 확인할 수 있습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_table_2.PNG)

<center>ImageNet dataset에 대한 여러 data augmentation image classification result</center>

또한 ImageNet에서도 saliency mix는 cutout, mixup, cutmix보다도 더 개선된 top-1, top-5 error를 보인다는 것을 확인할 수 있습니다.

### Object Detection

Object detection의 경우, ResNet-50을 backbone으로 사용하는 Faster RCNN을 기반으로 실험을 진행하며, benchmark dataset으로 Pascal VOC 2007과 2012 dataset을 사용합니다. 

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_table_3.PNG)

<center>Object detection에 대한 여러 data augmentation result</center>

위 테이블에서 확인할 수 있듯, saliencymix는 mixup, cutout, cutmix보다 더 높은 성능을 보였습니다.

### Adversarial Attack

Adversarial Attack에 대해 간단히 설명하면, image에 의도적인 손상을 가해 model이 제대로 image classification을 진행하지 못하게 만드는 공격을 이야기합니다. 사람이 보았을 때에도 이미지에 변화가 생긴 것으로 인식하는 attack 뿐만 아니라, image를 사람이 육안으로 보았을 때에는 아무런 변화가 없음에도 불구하고 각각의 pixel에 해당하는 data에는 심각한 손상이 일어나서 model의 입장에서는 기존에 training data로 사용되던 image에 비해 엄청나게 큰 변동이 일어난 상태로 만드는 adversarial attack등도 존재합니다.

해당 실험의 경우, Resnet을 model로 사용하여 [FGSM (2014)](https://arxiv.org/abs/1412.6572) 을 사용하여 ImageNet에 adversarial attack을 적용한 validataion dataset을 통해 실험을 진행하였습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_table_4.PNG)

<center>Adversarial Attack에 대한 여러 data augmentation result</center>

실제 위 테이블에서 확인할 수 있듯, saliencymix는 cutout, mixup, cutmix에 비해 adversarial attack에 대한 robustness가 향상되었음을 확인할 수 있습니다.

### Training Time

모델이 train을 진행하는 과정에서, 그 소요시간이 얼마나 단축되는 가는 model training에서 굉장히 중요한 요소입니다. 이를 위해 training data를 사용해 새로운 image를 생성하는 data augmentation 과정 또한, 소요 시간이 train time에 영향을 주기에 얼마나 빠르게 동작하는지가 중요합니다.

이를 측정하기 위해 논문에서는 Resnet을 architecture로 사용하고 CIFAR-10을 학습하는데 걸리는 시간을 측정하여 saliencymix의 training time을 측정하였습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_table_5.PNG)

<center>여러 data augmentation의 trainig time result</center>

그 결과, saliencymix의 경우 source image에 대해 saliency map을 추출하는 과정이 추가되어 다른 data augmentation에 비해서는 소폭의 training time이 더 필요했다는 것을 확인할 수 있습니다. 

### CAM Analysis

결과의 향상을 visualization을 통해서 확인하기 위해, 논문에서는 saliency mix 및 여러가지 data augmentation 기법들에 대해 CAM을 통해서 모델이 주어진 이미지에서 특정 label에 해당하는 object를 찾기 위해 어떠한 feature들에 대한 activation이 높게 나타났는지 CAM을 통해 확인하였습니다.

![](/assets/images/VennTum/data_augmentation/saliencymix_paper_figure_4.PNG)

<center>여러 data augmentation을 적용한 model의 CAM result</center>

주어진 이미지의 왼쪽을 보면 각각의 label에 대해 어떠한 pixel들이 높은 activation을 나타냈는지 확인할 수 있습니다. 각각 tent, marmot, vizsla의 image들에 대해 saliencymix가 다른 data augmentation이나 baseline에 비해 실제 사람이 육안으로 확인하는 object에 더욱 가까운 영역에 높은 activation을 보였다는 것을 확인할 수 있습니다.

또한 오른쪽 이미지를 확인하면, mixup, cutout, cutmix에 비해 saliencymix의 경우 augmented image에서 각각의 label에 해당하는 object가 image 내에 잘 남아있어 해당 label의 feature들에 더 높고 정교하게 activation map이 나타난다는 것을 확인할 수 있습니다.

## 마치며

Mixup과 Cutmix는 image data augmentation에서 뜨거운 감자였던 획기적인 논문이었습니다. 그리고 Saliency Mix는 cutmix를 개선하기 위해 간단한 아이디어를 적용하여 성능의 향상을 이루어냈습니다.

물론 현재에는 [Co-mixup (2021)](https://openreview.net/forum?id=gvxJzw8kW4b)을 비롯한 다양한 data augmentation 기법들이 새롭게 등장하면서 saliency mix의 성능이 가장 좋다고 이야기할 수는 없지만, 사용방법이 간단하고 코드가 굉장히 짧다는 장점이 있어, data augmentation을 사용할 일이 있다면 사용해볼만한 논문입니다.
