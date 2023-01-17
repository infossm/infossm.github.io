---
layout: post
title: Compressed Sensing
date: 2022-04-19 22:27:10 +0900
author: ainta
tags:
 - Signal processing
 - Optimization
 - MR Imaging
---
# Introduction
Compressed Sensing(CS)이란 어떠한 신호(signal)를 효율적으로 수집 및 복원하는 기법으로, 이미지 압축이나 복원, 카메라 센서 등에 사용되는 기술이다. 간단히 예를 들어 설명하자면, N개의 픽셀로 이루어진 사진을 얻기 위해서는 N번의 observation이 필요하다고 생각하는 것이 일반적이라면 CS를 이용하는 경우 훨씬 적은 측정으로도 이를 loss 없이, 또는 매우 적은 loss로 얻을 수 있다. 이 글에서는 CS 이론의 배경과 기반 및 실제 적용에 대해 간략히 살펴본다.
## History

**Nyquist-Shannon sampling theorem**

> $f(t)$ 가 $B$ 헤르츠 이상의 주파수를 가지지 않을 때, $1/2B$초 주기로 $f(t)$를 측정하면 $f$를 완벽하게 결정할 수 있다.

위 정리는 고전적인 signal processing의 기반이 되는 정리이다. $B$ 헤르츠 이상의 주파수를 가지지 않는다는 뜻은 $f(t)$의 Fourier transform $F(\omega)$가 $-2\pi B < \omega < 2\pi B$ 범위에서만 0이 아닌 값을 갖는다는 뜻이다. 그러나 이 정리는 $f$를 결정하기 위해서 sampling rate (1초에 sample을 몇번 얻어야 하는지)가 $2B$ 이상이어야 한다는 큰 제약 조건을 전제로 하고 있다.
적은 sampling rate로도 원래의 신호를 온전하게 얻어낼 수 있는 방법은 없을까?

## Sparsity

CS의 기반은 하나의 가정 위에 이루어져 있는데, 우리가 측정하고자 하는 신호들은 어떠한 domain에서는 sparse하다는 것이다. 즉, 필요한 값이 $N$개라면 $K \ll N$이 존재하여 데이터를 transform해서 0이 아닌 값이 $K$개 이하가 되도록 할 수 있다는 것이다.
예를 들어, 512x512 grayscale 이미지가 주어진 상황을 생각해 보자. 이 이미지는 0 이상 255 이하의 정수 262144개로 표현된다. 이미지를 transform을 통해 (주로 linear transform) 0이 아닌값이 1000개인 domain으로 보낼 수 있으며, 이 transform의 inverse transform이 존재한다고 가정하자. 이러한 경우 Compressed Sensing을 통해 적은 횟수의 sampling만으로 이미지를 reconstruct하는 것이 가능하다.
CS를 간단히 한 문장으로 설명하자면, 우리가 다루는 신호를 다른 space로 보내 sparse하게 만드는 transform을 찾은 후에, 이를 이용하여 적은 observation만으로 원래의 신호와 거의 동일하게 복원하는 방법이라고 볼 수 있다. 이러한 transform의 대표적인 예시로는 discrete fourier transform, wavelet transform 등이 있다.

## Wavelet Transform

Fourier transform이 Image domain을 Frequency domain으로 이동시키듯, Wavelet transform은 Image domain을 Wavelet domain으로 이동시키는 역할을 한다. 우리가 실제로 얻는 많은 신호들은 Wavelet transform을 거치면 sparse해진다. 즉 대부분의 값이 0 또는 0에 매우 가까운 값이 된다. 이미지가 그 대표적인 예라고 볼 수 있는데, 2D image에서 Wavelet Transform을 하면 어떻게 되는지 간단한 실습을 통해 알아보자.

```python
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from matplotlib import pyplot as plt
import requests
import pywt
url = "https://eeweb.engineering.nyu.edu/~yao/EL5123/image/lena_gray.bmp"
img = np.array(Image.open(requests.get(url, stream=True).raw))
res = pywt.dwtn(img, 'db1')
plt.figure()
f, axarr = plt.subplots(2, 2)
axarr[0][0].imshow(res['aa'], cmap= cm.gray)
axarr[0][1].imshow(res['ad'], cmap= cm.gray)
axarr[1][0].imshow(res['da'], cmap= cm.gray)
axarr[1][1].imshow(res['dd'], cmap= cm.gray)
plt.show()
```
아래는 예시 이미지로 자주 활용되는 512 x 512 grayscale Lena image를 2D wavelet transform 후 얻어지는 결과이다. 원본은 512x512 이미지인데, wavelet transform으로 얻은 결과는 원본과 유사하지만 크기는 가로 세로가 모두 절반인 256x256 이미지 하나와 sparse한 정보를 가진 (유의미한 pixel의 개수가 적은) 256x256 이미지 3개로 변환되었다. 이렇게 얻은 4개의 이미지로 다시 inverse wavelet transform을 하면 loss 없이 원래의 이미지를 얻을 수 있다.

![Figure 1. Wavelet Transform](/assets/images/compressed-sensing/Figure_3.png)

한편, wavelet transform으로 얻은 4개의 이미지 중 sparse하지 않은 하나에 대해서는 다시 wavelet transform을 하면 원본과 유사한 128x128 이미지 하나와 sparse한 128x128 이미지 3개가 나올 것이다. 이를 반복하여 적용하면 픽셀 수가 적은 작은 정사각 이미지 하나와 sparse한 나머지 이미지들을 얻을 것이다. 이를 wavelet decomposition이라 한다. 아래는 wavelet decomposition을 수행하여 그 결과를 확인하는 코드이다.

```python
coeff = pywt.coeffs_to_array(pywt.wavedecn(img, 'db1'))
plt.figure()
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img, cmap= cm.gray)
axarr[1].imshow(np.minimum(abs(coeff[0]),256), cmap= cm.gray)
plt.show()
```

![Figure 2. Wavelet Decomposition](/assets/images/compressed-sensing/Figure_1.png)
Wavelet Decomposition 후 이미지인 오른쪽 이미지는 밝기가 특정 값 이상인 픽셀의 수가 왼쪽 원본에 비해 훨씬 적다. Wavelet Decomposition에서 밝기가 10 이상인 픽셀들만 남기고 나머지를 0으로 바꾼 후, 다시 Inverse Wavelet Transform을 통해 복원해보자.

```python
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img, cmap= cm.gray)
axarr[1].imshow(pywt.waverecn( pywt.array_to_coeffs((abs(coeff[0])>10) * coeff[0], coeff[1]), 'db1'), cmap=cm.gray)
plt.show()
```

![Figure 3. Wavelet Reconstruction](/assets/images/compressed-sensing/Figure_2.png)

대부분의 픽셀 값을 0으로 바꾼 후 복원했음에도 왼쪽의 원본과 크게 차이나지 않는 이미지로 복원되었음을 확인할 수 있다. 실제로 JPEG-2000등의 이미지 압축 기술은 wavelet transform을 이용한다. 물론 JPEG-2000은 위 예처럼 단순하게 작동하는 것은 아니지만, wavelet transform이 이미지를 sparse하게 변환한다는 동일한 원리를 기반으로 한다.

# Compressed Sensing

많은 경우에, 우리가 측정하고자 하는 신호가 $x$라면 Sensing process를 통해 실제로 얻을 수 있는 값은 $y = \Phi x + \epsilon$ 형태이다. 여기서 $x \in R^n, y, \epsilon \in R^m$이며 Measurement Linear Operator $\Phi$는 $m$ by $n$ matrix로 생각할 수 있다. $\epsilon$은 측정 오차를 나타낸다.
여기에서 먼저 오차를 무시하고 생각해보자. $y$와 $\Phi$가 주어져있을 때, $m < n$이어서 $y = \Phi x$를 만족하는 $x$가 여러가지 있다면 그 중 어떤 것을 골라야 할까?

앞서 CS에서는 신호를 sparse하게 만드는 transform을 이용하여 복원한다고 하였다. 해당 transform을 $\Psi$라 하면 $\Psi x$는 sparse하다. 따라서, $y$로부터 가장 likely한 $x$를 construct하는 문제는 $y = \Phi x$를 만족하는 $x$ 중 $\Psi x$가 가장 sparse한 $x$를 찾는 최적화 문제를 푸는 것으로 생각할 수 있다.
이 때, $\Psi x$의 sparseness를 원래 정의대로 0이 아닌 element의 개수, 즉 L0 norm으로 정의한다면 각 원소가 $0$일 때만 튀는 불연속적인 함수가 되기 때문에 optimization 문제의 설정에 적합하지 않다. 이에 따라, L1 norm인 $ \lVert \Psi x \rVert _{1} $ 를 minimize하는 $x$를 구하는 것으로 대신한다.

즉, 아래의 최적화 문제의 해 $x$를 구하는 optimization problem이 된다.

> $min_{x} \lVert  \Psi x \rVert _{1} \hspace{10mm}$ subject to $\hspace{10mm} \Phi x = y$ 

이 문제를 basis pursuit이라 하며, 이는 Linear Programming 문제로 변형 가능하기 때문에 Simplex 등의 방법으로 최적해를 구할 수 있다. Linear Programming으로 변형하는 과정을 소개하자면, $1 \le i \le N$에 대해 변수 $s_i$를 정의하면  $-s_i \le (\Psi x)_i \le s_i$ 및 $0 \le s_i$, $y = \Phi x$ 를 만족한다고 놓은 상태에서 $s_i$의 합을 minimize하는 문제로 바뀐다.

그러나 실제로는 측정에 오차가 존재함에 따라 $\Phi x = y$ 상에서 올바르게 reconstruction이 되지 않는 경우도 존재한다. $\Phi x = y$ 상에서 문제를 해결하지 않고, $\Phi x - y$의 L2 Norm에 해당하는 term을 더함으로써 이를 최소화하는 아래 optimization problem으로 앞서 발생하는 문제점은 해결할 수 있다:

Minimize

$$ \lVert  \Phi x - y \rVert _{2}^{2} +  \lambda \lVert  \Psi x \rVert _{1}$$

위와 같은 꼴의 최적화 문제는 lasso optimizaion problem이라고 하며, Linear Programming으로는 해결할 수 없다. 보통 lasso optimizaion 문제는 iterative한 method를 사용하여 해를 구한다. 좋은 결과를 얻는 방법으로는 Alternating direction method of multipliers(ADMM)이 대표적이고, 그 외에 ISTA(iterative soft-thresholding algorithm) 와 이를 빠르게 수렴하도록 개선한 FISTA 등이 있는데 이들은 Proximal gradient descent에서 출발한 것이다. 이와 관련된 개념들은 [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)에서 찾아볼 수 있고, 한글 자료로는 [모두를 위한 컨벡스 최적화](https://convex-optimization-for-all.github.io/)에 잘 정리되어 있다.

## MR Image Reconstruction

의료 영상 기법 중 하나인 MRI (Magnetic Resonance Imaging)를 이용하면 조직의 단면 이미지를 얻을 수 있다. MR Imaging에서는 각 코일별로 데이터가 측정되는데, 이 데이터의 domain은 k-space이다. k-space란 image domain에서 fourier transform을 취하면 나오는 domain이다. 

discrete fourier transform은 linear operator이므로, 측정으로 얻는 k-space의 데이터를 $y$, 실제 얻고자 하는 신호(이미지)를 $x$ 라 하면 $y = FSx$의 관계가 성립한다. 여기서 $S$는 각 코일의 sensitivity map이고, $F$는 discrete fourier transform에 해당하는 linear operator이다.

MR Imaging에서는 이미지를 얻는 데 오랜 시간이 걸리기 때문에, 시간 단축을 위해 보다 적은 수의 k-space data로 이미지를 reconstruct하는 방법이 연구되고 있다. Compressed Sensing이 여기에 어떻게 이용될 수 있는지 알아보자.

먼저, k-space data를 얻을 때 가속을 해서 적은 수의 데이터를 얻었다고 하자. 그러면 얻은 데이터 $y = UFSx$ 로 볼 수 있다. 여기서 $U$는 얻지 못한 부분을 $0$으로 만드는 undersampling matrix이다. 또한, $x$는 image domain이므로 wavelet transform을 거치면 sparse한 신호가 되어야 한다. 따라서, Sensitivity map $S$가 주어져 있는 경우 아래 optimization problem을 해결함으로써 reconstruction이 가능하다.

Minimize

$$ \lVert UFS x - y \rVert _{2}^{2} +  \lambda \lVert  \Psi x \rVert _{1}$$

최근에는 위 optimization problem을 해결하는 ISTA나 ADMM등의 iterative method에서의 한 iteration을 Deep Neural Network를 이용하여 reconstruction의 성능을 높이는 시도가 많이 나타나고 있다. ([pISTA-SENSE-ResNet](https://arxiv.org/pdf/1910.00650.pdf), [Autotuning Plug-and-Play Algorithms for MRI](https://arxiv.org/pdf/2204.04771.pdf)) 이는 MR Imaging에 한정되지 않고 [Deep Compressed Sensing](https://arxiv.org/abs/1905.06723) 등  Compressed Sensing 전반을 DNN을 이용하여 해결하고자 하는 경우가 많아지는 추세이다.

## Conclusion

Compressed Sensing은 여기서 다룬 주제인 이미지 압축 및 복원뿐 아니라 카메라의 센서, 우주 망원경, 전자 현미경 등 다양한 산업과 과학 분야 전반에 걸쳐 이용되는 매우 활용도가 높은 분야이다. 이를 공부하고 이해함으로써 적은 측정으로 보다 많고 정확한 데이터를 얻는 방법을 알 수 있을 것이다.

## Further Study

자료를 찾아보면서 Compressed Sensing의 이론과 연관된 분야들 역시 흥미로운 분야가 많음을 알게 되었다. 먼저, Convex optimization에서는은Compressed Sensing에서 나오는 문제를 해결하는 방법을 다룬다. 그리고 비교적 친숙한 fourier transform 이외에 wavelet transform 등이 정확히 무엇인지 아는 것도 해석학에서 공부해볼만한 주제라고 생각한다. 처음에 잠깐 소개했던 Nyquist-Shannon sampling theorem 은 Information Theory 의 주제라고도 볼 수 있는데, information theory에 대한 내용도 재밌는 것이 많아서 추후에 다룰 계획이다.
