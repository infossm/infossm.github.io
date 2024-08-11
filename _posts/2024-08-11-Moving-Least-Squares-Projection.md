---
layout: post
title: "Moving Least Squares Projection"
date: 2024-08-11
author: mhy908
tags: [Point-Set-Surfaces, algorithm]
---

## Point Based Graphics

Point Based Graphics는 컴퓨터 그래픽스의 한 분야로, 기존의 다각형 기반으로 형상을 모델링하는 기법인 Polygon Based Graphics와 달리 점들을 사용하여 3D 객체의 표면을 표현하는 기법입니다.

각 점은 위치 뿐만 아니라 그 지점의 색상과 텍스쳐 등 다양한 속성도 가질 수 있습니다. 여기서 객체를 표현하는데 사용된 점들의 집합을 Point Cloud라고 합니다. 이 Point Cloud가 조밀할수록 더욱 세밀한 형태를 표현할 수 있습니다.

![](/assets/images/Moving-Least-Squares-Projection/pic2.png)

Point Based Graphics는 복잡한 형상을 기존 기법보다 더 효율적이고 자세하게 표현할 수 있다는 장점이 있습니다. 또한 3D 스캐닝 등의 기법을 통해 물리적으로 존재하는 사물에 대한 정보는 Point Cloud로 표현되기에 이를 컴퓨터 화면 속으로 옮기는데 유리한 면 또한 존재합니다.

![](/assets/images/Moving-Least-Squares-Projection/pic1.png)

위 그림은 Point Based Graphics의 전체적인 pipeline입니다. 이 글에서는 그 중 두번째 단계인 Surface Reconstruction에 대해 다뤄보도록 하겠습니다.


