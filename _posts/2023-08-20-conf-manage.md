---
layout: post
title: "Facebook의 설정 관리 시스템"
date: 2023-08-20
author: leejseo
tags: [introduction, software-design]
---

이 글에서는 [SOSP 2015 논문](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/008-tang.pdf)에 소개된 Facebook의 설정 관리 시스템을 살펴봅니다.

# Introduction

오늘날 대규모 웹서비스는 단순히 한대의 서버에서 돌아가지 않고, 수천대, 수만대 혹은 수십만대 이상의 서버로 이루어진 분산 환경에서 돌아갑니다.
이러한 상황에서 동일한 역할을 하는 서버들 사이에도 각 서버별로 다른 설정 값을 넣어줘야 하는 상황은 존재하며, 이를 "잘" 관리하는 것은 몹시 어려운 일입니다.

또한, 설정 값의 수정 또한 빈번한 경우에는 더더욱 어렵습니다.
예를 들어, 서버가 새로 시작하는 경우에만 설정 값을 읽어온다면, 새로운 설정을 적용하고 싶을 때 마다 서버를 재시작해야 합니다.
이는 몹시 비효율적입니다.

오늘 소개하는 논문은 이러한 상황에서 잘 활용할 수 있는 설정 관리 시스템을 제안합니다.

# Use Cases

저자들이 소개한 Facebook 내에서의 설정 값 사용예로는 다음이 있었습니다.
- Feature guarding
    - 새 기능을 개발한 후 설정 값을 통해 disable한 상태로 배포한다. 그 이후 일부 유저에게만 켜보면서 검증해보고, 이상이 있을 시 설정 값만 다시 바꾸어 롤백
- Experiments
    - Live traffic에 대한 A/B testing
- Traffic control
    - Application-level control과 load balancing 등
- Monitoring
- Updating ML models
- Controlling an application's internal behavior
    - 각 작업별로 할당된 메모리 용량 변경, 디스크에 배치로 write 하는 크기 단위 수정 등

# Goals

저자들은 설정 관리 시스템을 구성하면서 다음의 목표를 달성하고자 했습니다.

- 확장성
    - Facebook에는 프론트엔드 제품, 백엔드 서비스, 모바일 앱, 데이터 저장소를 포함하여 수많은 시스템이 있다. 이 시스템 전체로 확장될 수 있는 중앙화된 설정 관리 체계를 만들어 각 시스템 별로 별도의 설정 관리 체계를 갖추지 않아도 되도록 하자.
- 편하고 안전한 설정 관리
    - Facebook의 설정 파일 크기의 중간 값은 1KB이며, 큰 파일은 MB/GB 단위 크기를 가진다. 이렇게 큰 설정을 직접 수정하는 것은 불편할 뿐만 아니라 오류에 취약하다.
    - 이러한 문제를 해결하기 위해 설정을 일종의 하이레벨 (소스) "코드"로써 관리하고 (configuration-as-code) 이를 컴파일 하여 실제 설정 파일을 생성하자.
- 설정 값 오류로 부터 생기는 시스템 오류 방지
    - 설정 값에 대한 가정(예: 확률의 경우 0.0 이상 1.0 이하임을 검사할 수 있겠다.)을 검증하는 validator를 만들고, 자동으로 실행되게 하자.
    - 설정 값 수정 또한 코드 변경과 동일하게 취급하여 리뷰 및 테스트를 거치자.
    - 자동화된 [Canary test](https://en.wikipedia.org/wiki/Feature_toggle#Canary_release) 를 수행해 시스템 오류를 유발할 수 있는지 검증하고, 문제가 있는 경우엔 자동으로 롤백하자.
- 설정 값 디펜던시 지원
- 확장성 있고 안전한 설정 값 배포
    - 설정 값 관리 시스템에 문제가 생기더라도 다른 서버들의 가용성에 문제가 생기지 않아야 한다.

# Components Overview

저자들이 이를 위해 제시한 시스템은 다음의 컴포넌트로 구성됩니다.

- Configerator
    - 버전 관리, 설정 작성, 코드 리뷰, 자동화된 canary 등 가장 기본이 되는 기능을 제공
    - 다른 컴포넌트들은 이 위에서 작성됨
- Gatekeeper
    - 새 기능의 rollout을 관리
    - 예) 최적의 parameter를 찾기 위한 A/B testing
- Package Vessel
    - P2P file 전송을 통해 config 전달을 도움
    - 머신러닝 모델과 같은 큰 config을 consistency guarantee 희생 없이 전달하는데 쓰임
- Sitevars
    - Front-end (PHP) 제품에서 사용하기 위한 API를 제공
- Mobile Config
    - 모바일 앱의 설정을 관리
    - Configerator, Gatekeeper 등의 backend system과 Android/iOS 모바일 앱 사이를 이어주는 다리의 역할.

![](/assets/images/2023-08-20-conf/01.png)

위의 설명을 보면 알 수 있다 싶이, Configerator가 가장 근간을 구성하며, 그 위에서 Gatekeeper가 기능의 rollout을 관리합니다.
모바일 앱의 경우 Mobile Config을 통해 다른 기능들에 접근하며, frontend 제품은 Sitevars를 통해 접근합니다.

# Configerator

먼저, Configerator component에 대해 살펴봅시다.

## Configuration Authoring

저자들은 다음의 가정에 기반하여 Configerator를 설계했습니다.

1. 대부분의 엔지니어들은 설정을 직접 수정하기 보다 설정을 생성하는 코드 작성을 선호한다.
2. 대부분의 경우, 프로그램을 관리하는게 설정 값을 관리하는 것 보다 편리하다.

이를 바탕으로 엔지니어가

1. 설정 데이터의 schema 파일을 정의하고 (Thrift code)
2. 설정 파일(JSON)을 생성하는 Python script를 작성하고
3. (Optional)Python 기반의 validator를 작성

할 수 있도록 했습니다. 또한, 다음 예시에서와 같이 이 과정에서 다른 코드를 import 하는 등 다양한 기능을 지원하여 설정 작성 코드를 "모듈화" 할 수 있도록 했습니다.

![](/assets/images/2023-08-20-conf/02.png)

## Preventing Config Errors

저자들은 설정 오류로 인한 서비스 다운 등을 방지하기 위해 다음의 과정을 거치도록 하였습니다.

1. Validator를 통한 검증
2. 설정 생성 코드 및 생성된 설정에 대한 코드 리뷰
3. 자동화된 integration test (+ manual test)
4. 자동화된 canary test

Canary test는 수십대 정도의 적은 서버에 먼저 배포해보는 단계와 전체 클러스터에서 배포해보는 단계, 이렇게 두 개의 단계로 나누어 수행됩니다. Healthcheck metric, Click-through rate 등 여러 주요 지표에서 기존 설정을 쓰는 서버와 비교했을 때 문제가 생기지 않음을 검증합니다.

## Scalable and Reliable Config Distribution

저자들이 제안한 Configerator에서는 분산 코디네이션 서비스인 ZooKeeper를 개조하여 만든 Zeus를 통해 설정을 배포합니다.
여기에서는 leader, observer, proxy, 이렇게 세 level로 구성된 tree 형태의 구조를 사용합니다.

각 리더는 수백대의 observer를 자식으로 가지고 있고, 각 데이터 센터에는 여러 observer가 있습니다. 각 observer는 리더의 데이터의 read-only copy를 들고 있습니다. Leader의 데이터에 수정이 있을 때 마다 leader는 observer에게로 해당 변경 내용을 전송합니다.

각 서버는 Configerator proxy process를 돌리고 있습니다. Proxy는 데이터 센터 내의 observer 하나를 구독하고 있습니다. 구독할 observer는 무작위로 선택하며, 이에 실패하는 경우 다른 observer의 구독을 시도합니다.

Proxy는 전체 설정 값을 들고 있지는 않고, 해당 서버에서 실제로 사용할 설정 값만 들고 있습니다. 그리고 이를 들고 있을 뿐만 아니라 on-disk cache에도 저장해놓습니다. 또한, 구독하고 있는 observer의 데이터에 변경이 있는 경우 proxy는 이에 대한 알림을 수신하고, 자신이 들고 있는 데이터를 업데이트 합니다.

각 애플리케이션은 설정 값을 읽을 때 먼저 자신의 서버 내에 있는 proxy에 요청하며, 실패하는 경우 on-disk cache로 부터 읽어갑니다.

저자들은 이러한 설정 배포 과정을 통해 가용성을 높일 수 있었습니다. 그리고 다행히도, 앞서 언급했듯이 고용량 설정의 경우 Package Vessel을 통해 별도로 배포하기 때문에 이 구성에서 네트워크 사용량과 관련하여 큰 문제가 발생하지는 않는다고 합니다.

# Package Vessel

빠르고 효율적인 설정 배포를 위해 저자들은 대용량 설정 배포시 Package Vessel 이라는 별도의 컴포넌트를 이용하는 방법을 제안하였습니다. 대용량 설정의 경우 설정의 버전과 설정을 가져올 수 있는 서버 정보 등 간단한 메타 데이터만 Configerator에 올리고, 이를 기반으로 별도의 서버에서 P2P 통신을 통해 가져오도록 하였습니다.

# Interface

## Sitevars

Sitevars는 UI를 제공하며, (frontend 제품에서 주로 사용되는) 간단한 key/value 설정 관리를 위한 용도로 설계되었습니다. 목적이 목적인 만큼, Configerator 에서와 달리 Python/Thrift 코드를 작성하지 않고도 사용할 수 있습니다.
사용자가 원하는 경우 validation을 위한 간단한 script를 작성할 수도 있도록 하였습니다.

또한, 타입 오류 등을 방지하기 위해 기존의 값들을 기반으로 타입을 추론하고, 기존의 값들과 다른 타입을 입력할 경우 경고하는 기능 등도 삽입하였습니다.

## Gatekeeper

Gatekeeper는 feature guarding을 위한 인터페이스를 지원하며, 다음 pseudo code와 같은 간단한 syntax를 통해 이용 가능합니다.
```
if (gk.check(project_id, user_id)) {
    // show new
} else {
    // show default
}
```

Gatekeeper에서 check 하는 로직 내부에서 다음 pseudo code와 같이 feature 적용 조건들을 검사하고, 조건에 맞는 경우 설정된 비율에 따라 feature를 적용합니다.
```
bool check(project, user_id) {
    // ...
    if (project == "project_id") {
        if (constraint1(user_id)) {
            return rand(user_id) < prob_1
        }
        if (constraint2(user_id)) {
            return rand(user_id) < prob_2
        }
        return false
    }
    // ...
}
```

## Mobile Config

Mobile Config은 주기적으로 설정을 서버로 부터 pull 해오며, 버그 수정 등 긴급한 경우에는 수정된 설정을 서버가 push 할 수 있도록 구성하였습니다.

# Usage Results

저자들은 이 시스템을 사용하는 내부 직원들의 통계를 기반으로 다음의 결과를 제시했습니다. 개인적으로는 이 결론 또한 굉장히 재밌게 읽었습니다.

- 많은 엔지니어가 config 보다 config 생성 코드 작성을 선호하는가?
    - 그렇다: 75%의 config이 생성된 config 이었으며, 89%의 업데이트는 자동화된 도구를 통해서 이루어졌다.
    - 또한, 이러한 이유에 대한 가설로 raw config을 쓸 때에 비해 compiled config을 쓸 때에 더 용량이 큰 config을 많이 작성하게 되었음을 제시했다.
- Config 수정에 대한 (직관을 줄 수 있는) 재미있는 통계
    - 50%의 config 수정은 한 줄 혹은 두 줄 수준의 작은 수정이었다.
    - 생성된지 오래된 config와 최근에 생성된 config 모두 유의미한 비율로 업데이트 된다.
    - 오랫동안 업데이트 되지 않은 config 또한 유의미한 비율로 존재한다.
    - 70% 이상의 config은 2명 이내의 co-author를 가진다.
- Config와 관련한 퍼포먼스
    - Config이 production 서버에 도달하기 까지의 latency: 기본적으로 10초 대의 시간이 소요됨. 하지만, 로드가 많은 경우 30초 이상 소요되는 경우도 있음. Daily 및 weekly pattern이 있다.

