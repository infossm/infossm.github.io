---
layout: post
title: "FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores"
date: 2024-01-29
author: billcho
tags: [parallel, cuda, fft]
---

# Introduction
이번 글에서는 작년 11월에 처음 arXiv에 게재되었고 ICLR 2024 poster로 발표될 예정인 [FlashFFTConv](https://arxiv.org/abs/2311.05908)에 대해 다루어 보고자 한다.

먼저 introduction에서는 논문과 조금 다르게 필자가 생각하는 motivation들을 적어 보고자 한다.

## Scaling Law of LLM
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)는 OpenAI에서 연구한 결과를 정리한 article로, 다음 [Figure 1]의 결과를 제시한다. [Figure 1]은 여러 metric들이 지수적으로 증가함에 따라, LLM의 성능이 개선되는 것을 잘 보여주고 있다. 이러한 transformer 기반 모델의 특성은 현재 LLM을 NLP의 중심으로 만들었을 뿐만 아니라 computer vision 등 다른 분야에서도 SOTA accuracy를 달성하는 데 큰 역할을 했다.

<p align="center"><img src="/assets/images/billcho/scaling-law-for-llm.png"></p>
<center><b>[Figure 1] Language Modeling Test Loss by Amount of Compute, Dataset Size, Parameters</b></center>

## Accelerating Transformers
모델의 구조를 바꾸지 않더라도 속도를 개선시킬 수 있다면 더 큰 모델을 돌릴 수 있기 때문에, scaling law에 따라 같은 compute 자원 상에서 정확도가 높아지게 된다. 따라서 transformer based model을 가속하는 것은 model architecture 자체를 개선하는 것과 함께 ML 및 NLP 연구의 중요한 방향으로 자리잡고 있다.  
Quantization, [FlashAttention](https://arxiv.org/abs/2205.14135), [2:4 Sparsity](https://arxiv.org/abs/2104.08378) 등이 대표적인 예시로써 현재 존재하는 GPU의 architecture을 효율적으로 활용하거나 일부 바꾸어 큰 성능 개선을 이끌어냈다. 이러한 방법들에 대해서는 추후 다른 글에서 더 자세히 다루어 볼 것이다.

## Long Sequence Modeling
하지만 transformer based model은 약 16K token 이상의 long sequence task에 대한 근본적인 약점이 있다. 책 단위의 summary나 고해상도 이미지의 computer vision task, 일반적으로 높은 sample rate를 가지고 있는 음성 파일들에 대한 lossless한 분석 등은 모두 매우 긴 sequence에 대한 처리를 요구하기에 long sequence task는 중요하지만 해결 방법이 아직 완전히 정립되지 않았다.  
이는 self-attention 연산이 sequence length에 quadratic한 cost를 가지기 때문으로, [Longformer](https://arxiv.org/abs/2004.05150), [Linformer](https://arxiv.org/abs/2006.04768) 등의 기법들이 대안으로써 제시되었다. 최근 출시된 [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) 등에서도 유사한 방법을 사용하고 있지만 long sequence에서 training 과정에서의 어려움과 정확도의 감소는 transformer 계열의 모델들이 여전히 해결하기 어려운 문제로 남아 있다.

## SSM Based Model
State Space Model(SSM)은 제어 이론 계열에서 주로 사용되던 시계열 모형으로써 입력 $u(t)$, 출력 $y(t)$에 대해 내부 상태 $x(t)$를 가지는 다음 [Equation 1]과 같은 모델이다. 이 때 네 가지 parameter $A, B, C, D$가 시간에 따라 변화하지 않는 상황을 Linear Time Invariant(LTI)라고 부르고 이산화된 LTI SSM은 convolution과 fft를 이용해 $O(NlogN)$ 시간 복잡도에 $N$개의 항을 계산할 수 있음이 [알려져 있다](https://arxiv.org/abs/2102.11417). 이는 길이 $N$의 두 sequence에 대한 fft를 계산하고 pointwise multiplication 후 inverse fft를 통해 convolution이 계산 가능하기 때문이다.

<div align="center"> $\begin{align} \dot{x}(t) = A(t)x(t)+B(t)u(t) \\ y(t) = C(t)x(t)+D(t)u(t) \end{align}$ </div>
<center><b>[Equation 1] State Space Model</b></center>

[S4](https://arxiv.org/abs/2111.00396), [S5](https://arxiv.org/abs/2208.04933), [H3](https://arxiv.org/abs/2212.14052) 등 SSM based model들은 sequence length $N$에 대해 $O(N^2)$의 시간복잡도를 가지던 기존의 transformer based model과 달리 효율적이고 빠른 long sequence에 대한 추론이 가능하여 많은 주목을 받고 있다. SSM based model들의 최신 동향에 대해서도 다른 글에서 가능하면 다루어 볼 것이다.

# FlashFFTConv
결론적으로, fft를 활용해 convolution을 빠르게 계산하는 것은 SSM based model의 발전과 함께 중요한 task로 떠오르고 있다. FlashFFTConv는 제목과 같이 order-$p$ Monarch decomposition, tensor core, kernel fusion 등 다양한 기법을 사용하여 연산을 최대 7.93배 가속시켰고, 모델 전체에서도 최대 4.4배 가속을 이루어 냈다.

## Monarch Decomposition

## Tensor Core

## Kernel Fusion

## Sparse Convolution
논문의 저자들은 근사를 통해 연산을 더욱 가속시키는 알고리즘으로 partial convolution, frequency-sparse convolution을 추가적으로 제시하였다. 해당 알고리즘에 대해서는 논문의 3.3과 부록을 참고하면 좋을 것 같다.

# Conclusion
최근 ML에서 long convolution을 사용하는 SSM(State Space Model) base의 model architecture이 큰 관심을 받고 있는데, FlashFFTConv는 트렌드에 맞추어 뻔한 주제가 될 수 있는 FFT를 바탕으로도 좋은 building block을 제시한 논문이라고 생각한다. 

이뿐만 아니라 Problem Solving(PS)에서도 다항식 곱셈으로 대표되는, long sequence의 convolution을 계산하는 것은 유의미한 task이고 계산과학에서도 fourier transform은 자주 사용되는 기법이다. 물론 해당 논문에서는 FP16/BF16만을 target으로 하기에 지금 당장 HPC(High Performance Computing) 관련 application들에 적용하기에는 precision 문제가 있겠지만, [ICLR 리뷰](https://openreview.net/forum?id=gPKTTAfYBp)를 보면 저자들도 해당 문제에 관해 인식하고 있는 것 같고 알고리즘이 FP16/BF16에 그렇게 dependent하지도 않아 확장될 여지도 많을 것이다.
