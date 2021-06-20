---
layout: post
title:  "Object Detection"
date:   2021-06-20 08:00:00
author: VennTum
tags: [AI, computer-vision, deep-learning]
---

# Object Detection이란

Computer Vision(컴퓨터 비전)이란 컴퓨터 공학의 관점에서, 인간의 시각 시스템이 할 수 있는 작업을 구현하고 이를 자동화하는 방법을 다루는 학문입니다. 이를 위해 이미지 및 비디오에 대한 수집, 처리, 분석을 진행하기 위해 필요한 여러가지 주제들에 대한 연구가 이루어지고 있습니다.

Object Detection(객체 감지)란 컴퓨터 비전의 하위 분야 중 하나로 전체 디지털 이미지 및 비디오 내에서 유의미한 특정 객체를 감지하는 작업을 합니다.
이러한 object detection은 Image retrieval(이미지 검색), Image annotation(이미지 주석), Face detection(얼굴 인식), Video Tracking(비디오 추적) 등 다양한 분야의 문제를 해결하기 위해 사용됩니다.

자율 주행 자동차에서, 차량 주변에 있는 비디오를 인식하여 자동으로 비디오 이미지 내에 차량이 있는지 혹은 사람이 있는지 등을 분석하여 주변 객체들에 대한 위치, 이동 정보를 주는 것(video tracking)이나, 휴대폰 보정 카메라 어플에서 얼굴을 감지하여 해당 부분의 이미지 보정을 적용하는 것(face detection), 포털 사이트의 이미지 검색을 이용할 때, 사진을 업로드하면 해당 사진 내에 존재하는 객체들을 자동으로 인식하여 이와 비슷한 유형의 이미지를 찾아주는 것(image annotation & retrieval) 등이 object detection을 통해 이루어지는 작업들입니다.

비슷한 개념으로는 Image Classification(이미지 분류)이 있습니다. object detection의 경우, 한 사진 내에 우리가 object로 분류한 객체들에 대해, 어느 위치에(localization) 어떤 종류의 object가 있는지(classification)에 대한 정보를 주는 것이며, image classification은 하나의 사진 자체를 특정 개체로 분류하는 일을 합니다.

사진

