---
layout: post
title: "Bloom Filter"
date: 2019-05-17 22:48
author: taeguk
tags: [Bloom Filter, Scalable Bloom Filter, membership test, 자료구조]
---

안녕하세요, 오늘은 Bloom Filter 와 변종들중에 하나인 Scalable Bloom Filter 에 대해서 알아보는 시간을 가지려고 합니다. <br/>
너무 자세하거나 수학적인 설명은 배제하고 직관적인 이해를 기반으로 포스팅하겠습니다! ~~자세한건 그냥 논문을 보세요...~~

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/05/bloom-filter.html)**

## Bloom Filter 란?

[Bloom Filter]([https://en.wikipedia.org/wiki/Bloom_filter]) 는 집합내에 특정 원소가 존재하는지 확인하는데 사용되는 자료구조입니다. <br/>
이러한 "membership test" 용도로 사용되는 자료구조들은 Bloom Filter 외에도 다양합니다. 대표적이고 널리 알려진 것으로는 Balanced Binary Search Tree (AVL, red-black tree 등) 과 해시 테이블등이 있습니다. 이 자료구조들의 특징은 **100% 정확도로 membership test 를 수행할 수 있다는 것**입니다. <br/>
**Bloom Filter 는 이러한 정확도를 희생해서 메모리 사이즈를 최소화하는 것을 목표로 합니다.**

## Bloom Filter 구조

Bloom Filter 자료구조는 기다란 비트 배열로 구성됩니다. 그리고 "원소 추가" 와 "원소 검사" 연산을 수행하기 위해 서로 다른 해시 함수들이 사용됩니다. <br/>
비트 배열의 길이를 m, 해시 함수의 개수를 k 라고 하겠습니다. k개의 해시 함수들은 [0, m) 범위의 해시 값을 출력하게 됩니다.

그러면 Bloom Filter 의 2가지 연산에 대해서 알아보겠습니다.
1. 원소 추가
	* 추가하려는 원소에 대해서 k개의 해시 값을 계산한 다음, 각 해시 값을 비트 배열의 인덱스로 해서 대응하는 비트를 1로 세팅합니다.
2. 원소 검사 (membership test)
	* 검사하려는 원소에 대해서 k개의 해시 값을 계산한 다음, 각 해시 값을 비트 배열의 인덱스로 해서 대응하는 비트를 읽습니다. k개의 비트가 모두 1인 경우 원소는 집합에 속한다고 판단하고, 그렇지 않은경우 속하지 않는다고 판단합니다.

매우 간단하죠? [Bloom filter by example](https://llimllib.github.io/bloomfilter-tutorial/) 사이트에서 직접 눈으로 확인해보시면 이해에 더 도움이 되실 것 같습니다.

## Bloom Filter 특징

* 확률적 자료구조입니다.
	* 100% 정확도로 원소 존재 여부를 검사할 수 있는 해시 테이블등과는 달리, Bloom filter 는 정확도가 확률적입니다.
	* 예를 들면, 해시 테이블은 hash collision 을 극복하기 위해 원소의 전체 데이터를 비교하는 로직이 있습니다. 그리고 이를 위해서 원소의 전체 데이터를 자료구조 내부에 저장하게 됩니다.
	* 반면에, Bloom Filter 는 원소의 전체 데이터를 저장하지 않고, 해시 함수를 통해 원소의 특징 값들만 뽑아서 이걸 그냥 비트 배열에 반영시켜버립니다. 따라서 정확도는 떨어지게 되지만 메모리 사이즈를 매우 절약할 수 있게 되는 것입니다.
* False negative 는 존재하지 않습니다.
	* False negative (원소 검사시 존재하지 않는 원소를 존재한다고 판단할 가능성) 는 존재하지 않습니다. 위에서 "원소 추가" 와 "원소 검사" 연산이 어떻게 동작하는 지를 생각해보면 자명합니다.
* False positive 가 존재할 수 있습니다.
	* 원소 검사시 존재하지 않는 원소를 존재한다고 판단할 가능성이 있습니다.
	* False positive probability (이하 FPP) 는 [수학적으로 계산이 가능](https://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives)합니다. (이상적인 해시 함수의 개수 또한 계산이 가능합니다.)
* k개의 해시 계산이 병렬적으로 수행될 수 있다면 속도측면에서 큰 이점을 얻을 수 있습니다.

## Bloom Filter 활용

Bloom Filter 는 메모리 사이즈를 줄여야하는 경우에 유용하게 활용될 수 있습니다. 단, 문제가 되는 부분은 false positive 가 발생할 수 있다는 것입니다. <br/>
따라서, 약간의 false positive 가 발생하더라도 감안할 수 있는 경우가 아니라면 false positive 에 대비할 방법이 반드시 필요합니다. 해결책중에 하나는 Bloom Filter 외에도 해시테이블과 같이 100% 정확도를 제공하는 자료구조를 따로 구성해놓는 것입니다. 아래의 사진을 보시면 이해가 되실 것 같습니다. <br/>
![](https://lh3.googleusercontent.com/p30gyBHpZlbLGNhZeZF3sWBdXrCpQ96dYasZat1Ycu_rmUaxoSrMejcb1isCI6ueaVgXe4ab7bCj) (출처 : 위키백과) <br/>
Bloom Filter 는 메모리를 적게 사용하므로 RAM 과 같은 빠른 저장공간에 로드하고, 100% 정확도를 제공하는 자료구조는 하드디스크같은 느리지만 용량이 큰 저장공간에 위치시킵니다. Bloom Filter 를 이용해서 불필요한 접근을 먼저 필터링할 수 있고, FPP 를 막기 위한 용도로만 disk 에 접근하게 되므로 성능이 향상됩니다. <br/>
만약 원소 검사 요청이 대부분 존재하는 원소에 대한 검사 요청이라면, 이러한 구조는 별로 이점이 없을 것 입니다. 대부분의 경우에는 FPP 를 막기 위해 결국 disk 에 접근해야하기 때문입니다. <br/>
따라서, 이러한 구조는 원소 검사 요청이 대부분 존재하지 않는 원소에 대한 검사 요청일 경우에 활용할만한 구조입니다. <br/>
구글 크롬이 주어진 URL 인지 악성인지 여부를 체크할 때 Bloom Filter 를 활용하는 것도 이러한 맥락입니다. 대부분의 URL 은 정상이기 때문에 위와 같은 구조를 채택했을 때 성능상 큰 이점을 얻을 수 있는 것입니다.

## Bloom Filter Variants

Bloom Filter 는 간단하지만 매우 활용도가 높은 자료구조입니다.  <br/>
그러나 원소의 삭제가 불가능하고 원소의 개수가 많아질수록 false positive 의 확률이 높아지는 등의 문제점도 많습니다. 또한 특정 목적으로 사용하기 위해 개선할 수 있는 부분도 많이 있는데요. 따라서, 원조 Bloom Filter 에서 뻗어나온 다양한 변종들이 많이 존재합니다. <br/>
어떠한 변종들이 있는 지는 [위키백과](https://en.wikipedia.org/wiki/Bloom_filter#Extensions_and_applications)와 [이 블로그 글](http://matthias.vallentin.net/blog/2011/06/a-garden-variety-of-bloom-filters/)을 참고하시면 좋을 것 같습니다.  <br/>
저는 많은 변종들중에 하나인 "Scalable Bloom Filter" 에 대해서만 다뤄보겠습니다.

## Scalable Bloom Filter

Bloom Filter 의 대표적인 한계점중에 하나는 동적으로 원소를 추가하기에 효율적이지 않다는 것입니다. 원소의 개수가 고정되어있다면 Bloom Filter 를 구성하는 시점에, 수학적 이론을 통해서 원하는 FPP, 원하는 메모리 사이즈를 고려해서 최적의 hash 함수 개수, 메모리 사이즈를 결정할 수 있습니다. <br/>
그러나, 원소의 개수가 동적으로 계속 변경된다면 Bloom Filter 를 구성하는 시점에 최적의 hash 함수 개수, 메모리 사이즈를 결정할 수가 없게 됩니다. 또한 원소가 예상보다 훨씬 많아지게 된다면 FPP 가 너무 커져서 문제가 생길 수 있습니다.

이러한 문제를 해결하기 위해 나온 것이 [Scalable Bloom Filter](http://gsd.di.uminho.pt/members/cbm/ps/dbloom.pdf) 입니다. 대략적인 아이디어를 설명을 하자면 다음과 같습니다. <br/>
Scalable Bloom Filter 는 Bloom Filter 들의 나열로서 구성됩니다. 원소들이 계속 추가되다가 어느순간 일정 FPP 를 넘기게 되면 새로운 필터를 추가하게 됩니다. 그리고 그 다음부터 추가되는 원소들은 새롭게 추가된 필터에 추가되게 됩니다. 이렇게 필터가 추가될때마다 stage 가 1씩 증가한다고 말합니다. <br/>
membership test 는 필터들 각각에 대해서 membership test 를 수행함으로써 원소의 존재여부를 판단하게 됩니다.

Scalable Bloom Filter 를 구성할 때는 여러가지 상수를 결정해야만 하는데 이를 통해서 좀 더 보충설명을 해보도록 하겠습니다.
* Error probability tightening rate
	* Scalable Bloom Filter 는 각 stage 별로 최대 FPP 가 정해져 있습니다. 이러한 최대 FPP 는 첫 stage 에서는 높지만 stage 가 증가할수록 점점 작아지게 되어있습니다. 필터 전체의 FPP 는 각 stage 들의 최대 FPP 들을 통해서 결정되는데요. stage 가 무한대로 갈수록 전체적인 FPP 는 특정 값으로 수렴하게 됩니다.
	* stage 가 증가할 수록 최대 FPP 가 줄어드는 비율을 의미하는 상수가 바로 error probability tightening rate 입니다.
* Growth rate of the size
	* Scalable Bloom Filter 가 특정 stage 에서 최대 FPP 에 도달하게 되면 새로운 필터를 추가함으로써 다음 stage 로 넘어가게 됩니다. 이 때 추가할 필터의 크기를 결정하는 상수가 바로 growth rate of the size 입니다.
	* 예를 들어 이 값이 2 라면, stage 가 증가할수록 추가되는 필터의 크기는 10, 20, 40, 80, ... 와 같이 됩니다.

약간 복잡할 수 있는데 관심있으신 분은 [논문](http://gsd.di.uminho.pt/members/cbm/ps/dbloom.pdf)을 읽어보시는 걸 추천합니다. 논문에는 각 상수가 성능에 주는 영향과 최적의 상수 값을 찾는 방법등이 잘 나와있습니다.

## 결론

오늘은 Bloom Filter 에 대해서 한번 알아봤습니다 ㅎㅎ <br/>
간단하지만 정말 유용한 자료구조인 것 같네요. 알고있으면 적용할 만한 곳이 많이 있을 것 같습니다! (개인적으로 학부에서 필수로 가르쳐야할만한 자료구조인 것 같다는 생각이 듭니다.) <br/>
다음에 또 만나요~~
