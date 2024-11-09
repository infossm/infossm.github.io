---
layout: post
title: "Github Action + AWS Code Deploy로 CI/CD 구축하기"
date: 2024-10-31
author: Rn
tags: [github-action, AWS, CI/CD]
---

# Intro

CI/CD를 구축하는 방법은 여러 가지가 있지만, 이번 글에서는 Github Action과 AWS Code Deploy를 이용하여 CI/CD를 구축하는 방법에 대해 알아보겠습니다.

AWS Code Deploy에서 on-premise 서버 등 AWS 서비스가 아닌 개인 머신에 배포하는 기능도 있지만, 이번 글에서는 EC2 인스턴스에 배포하는 방법에 대해 알아보겠습니다.

# CI/CD

CI/CD는 Continuous Integration/Continuous Deployment의 약자로, 지속적 통합/지속적 배포를 의미합니다.

직관적인 말로 풀이하면, 작성한 코드를 자동으로 원하는 머신에 배포(설치)하는 것을 말합니다.

CI/CD가 없다면, 다음과 같은 과정을 거쳐야 합니다.

1. 코드 작성
2. 머신 접속
3. 코드 업로드
4. 코드 빌드 및 실행
5. 결과 확인

위 다섯 가지 과정을 사람이 모두 반복적으로 수행해야 합니다.

하지만, CI/CD를 적용한다면 다음과 같이 간단해집니다.

1. 코드 작성
2. CI/CD 실행
3. 결과 확인

# Github Action

Github Action은 Github 저장소에 Push하거나, Pull Request를 Merge할 때 미리 정의해 둔 작업을 수행하도록 설정할 수 있는 도구입니다.

예를 들어, 코드를 Push하면 자동으로 테스트 코드를 실행해 주는 등 다양한 작업을 설정할 수 있습니다.

# AWS Code Deploy

AWS Code Deploy는 AWS 서비스 중 하나로, 코드를 배포하는 서비스입니다.

코드가 저장된 위치 (Github, S3 등)와 배포할 위치 (EC2, Lambda 등)를 설정하면, 코드를 자동으로 배포해 줍니다.

코드 배포는 다음과 같은 과정을 거쳐 배포됩니다.

1. 배포할 위치에서 코드를 다운로드합니다.
2. 다운로드한 코드에 미리 정의된 작업 (Shell Script)을 수행합니다.

따라서, 배포 시 필요한 스크립트 (Shell Script로 작성된 배포 스크립트)만 작성하면, 코드 배포 시 해당 작업을 수행해 줍니다.

이번 글에서는 Github Action을 통해 작성한 코드를 S3에 업로드하고, AWS Code Deploy를 통해 EC2 인스턴스에 배포하는 방법에 대해 알아보겠습니다.

---

# 0. Pre-Requisite

이번 글을 따라 하기 전 다음 세 가지는 이미 만들어져 있다고 가정합니다.

- Github 저장소
- AWS 계정
- EC2 인스턴스

# 1. AWS IAM User 설정

AWS IAM이란, AWS 리소스에 일부 권한만 갖게 하는 권한 인증 방식입니다. 루트 계정에서 IAM 계정을 생성할 수 있습니다.

먼저 github action이 AWS에 접근할 수 있도록 권한을 줘야 합니다.

물론 루트 계정 권한으로 접근하게 만들 수도 있지만, 보안상의 이유로 권한을 제한하는 것이 좋습니다. 만약, 루트 계정이 노출된다면, 모든 AWS 리소스에 접근할 수 있기 때문입니다.

따라서 IAM을 이용해 깃허브 액션을 위한 계정을 만들어야 합니다.

`IAM > 액세스 관리 > 사용자 > 사용자 추가` 를 통해 새로운 사용자를 만들 수 있습니다.

Github Action은 `S3`에 파일 업로드를 한 뒤 `CodeDeploy`를 실행해야 하므로 아래 두 권한이 필요합니다.

권한 추가 탭에서 `AmazonS3FullAccess`와 `AWSCodeDeployFullAccess`를 체크 해줍니다.

이후 계정을 생성할 수 있습니다.

계정을 생성한 뒤에, `IAM > 사용자` 탭에서 생성한 사용자를 클릭하고, `액세스 키 만들기`를 클릭해서 아이디, 패스워드에 해당하는 `Access Key`와 `Secret Access Key`를 발급받을 수 있습니다.

이는, 추후 Github Action에서 사용할 것이기 때문에, 외부에 공개되지 않는 안전한 곳에 기록해 두는 것이 좋습니다.

# 2. AWS IAM Role 설정

AWS IAM Role(역할)은 AWS IAM User와 유사한 방식입니다.

AWS IAM User는 새로운 유저를 생성해서 해당 유저가 권한을 가지고 있는 방식이고, AWS IAM Role은 이미 존재하는 무언가(어플리케이션 등)에게 권한을 부여하는 방식입니다.

권한을 관리한다는 사실은 똑같지만, AWS IAM Role은 로그인 등을 할 수 있는 Access Key가 발급되지 않습니다. 따라서 EC2 혹은 CodeDeploy와 같은 AWS 내부 서비스에 권한을 부여할 때 사용합니다.

AWS CodeDeploy는 배포해야 할 EC2 인스턴스에 접근해야 하므로 EC2에 접근할 수 있는 권한이 필요하고,
EC2에서 CodeDeploy 요청을 받기 위해선 CodeDeploy에 접근할 수 있는 권한과 코드를 내려받기 위해 코드가 저장된 S3에 접근할 수 있는 권한이 필요합니다.

따라서 CodeDeploy와 EC2를 위해 AWS IAM Role을 생성해야 합니다.

`IAM > 액세스 관리 > 역할 > 역할 만들기`를 통해 Role을 생성할 수 있습니다.

이름은 알아볼 수 있게 원하는 걸로 정하면 됩니다.

이후 권한을 사용할 수 있는 타겟을 `AWS 서비스`를 선택하고 사용 사례를 골라주면 됩니다.

EC2를 위한 역할과, CodeDeploy를 위한 역할 두 개를 만들어야 합니다.

`EC2`를 위해 생성하는 역할의 타겟은 `EC2`를 선택하면 되고, `CodeDeploy`를 위해 생성하는 역할의 타겟은 `CodeDeploy`를 선택하면 됩니다.

`CodeDeploy`를 선택하면 필요한 권한이 모두 채워져 있습니다. 그럴 일은 없지만, 채워져 있지 않다면, `AWSCodeDeployFullAccess`를 추가하면 됩니다.

`EC2`를 선택한 곳에선 `AWSCodeDeployFullAccess`, `AmazonS3FullAccess`를 추가해 줍니다.

이렇게 되면 `CodeDeploy`를 위한 역할과 `EC2`를 위한 역할이 생성되었습니다.

# 3. AWS EC2 설정

EC2는 Ubuntu를 기준으로 설명합니다.

인스턴스를 생성하거나 생성한 뒤 태그를 추가해야 합니다. 태그를 추가하는 이유는 CodeDeploy에서 배포할 인스턴스를 찾기 위해 사용됩니다. 적당히 필터링할 수 있게끔 key, value를 생성하면 됩니다. 기본적으로 instance-name이 Name: {instance-name}으로 생성되어 있습니다. 이를 이용해도 되고, 여러 인스턴스를 한 번에 배포하려면 태그를 추가해도 됩니다.

먼저 EC2에 위에서 생성한 IAM Role을 연결해 줍니다.

`EC2 > 인스턴스 > IAM Role을 설정할 인스턴스 체크 > 작업 > 보안 > IAM 역할 수정` 을 통해 IAM Role을 연결할 수 있습니다. 위에서 생성했던 Role 이름을 선택한 뒤 IAM 역할 업데이트 버튼을 누르면 됩니다. 또는 EC2 인스턴스 생성 시 IAM Role을 설정할 수 있습니다.

이후 EC2에 CodeDeploy 클라이언트를 설치해야 합니다. 다음 코드를 통해 설치할 수 있습니다.

[공식 설치 문서](https://docs.aws.amazon.com/ko_kr/codedeploy/latest/userguide/codedeploy-agent-operations-install-ubuntu.html)

```shell
 $ sudo apt update
 $ sudo apt install ruby-full
 $ sudo apt install wget
 $ cd ~ 
 # 현재는 ap-northeast-2를 기준으로 설치하고 있습니다.
 # 다른 리전을 사용하고 있다면, 아래 링크에서 리전 두 개를 바꿔서 사용하면 됩니다.
 # ex) wget https://aws-codedeploy-{OTHER_REGION}.s3.{OTHER_REGION}.amazonaws.com/latest/install
 $ wget https://aws-codedeploy-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/latest/install
 $ chmod +x ./install
 $ sudo ./install auto
 $ sudo service codedeploy-agent status # 설치 확인
```

이후 초록색 글씨로 `active(running)`이 표시되어 있다면 제대로 설치가 된 것입니다.

**\[정보\] IAM Role을 설치하기 이전 CodeDeploy Client 를 설치했다면, 배포할 때 오류가 발생할 수도 있습니다. 이때는 codedeploy-agent를 껐다 다시 실행해 주면 문제를 해결할 수 있습니다.**

# 4. AWS S3 설정

배포할 코드를 저장할 파일 스토리지입니다.

원하는 이름으로 버킷을 생성하면 됩니다. 이후 `.github/workflows/**.yml` (Github Action 파일, 추후 자세히 소개) 파일에 버킷 이름을 수정해 주면 됩니다. 현재 예제에는 `deploy_bucket` 이름으로 버킷을 사용 중입니다.

IAM권한이 있어 S3에 접근할 수 때문에 공개 범위는 비공개로 설정하면 됩니다.

# 5. AWS CodeDeploy

배포를 관리해 줄 서비스입니다.

`CodeDeploy > 애플리케이션 > 애플리케이션 생성` 을 통해 애플리케이션을 생성합니다.

이번 글에서는 EC2로 배포할 것이기 때문에 `EC2/온프레미스`를 선택합니다.

이후 배포 그룹을 선택해야 합니다. 배포 그룹이란, 해당 어플리케이션으로 배포할 EC2 인스턴스를 모두 선택한 그룹을 뜻합니다.

`CodeDeploy > 배포 그룹 > 배포 그룹 생성` 을 통해 배포 그룹을 생성합니다.

이때 필요한 내용은 다음과 같습니다.

서비스 역할은 위에서 생성한 IAM Role을 선택합니다.

배포 유형은 다음 두 가지가 존재합니다.

* `현재 위치`: 기존에 실행 중인 인스턴스에 배포합니다. 즉, 기존에 실행되던 프로세스가 중지되고 배포가 진행됩니다.
* `블루/그린`: 추가적인 인스턴스를 생성하고, 배포가 완료되면 기존 인스턴스를 제거합니다. 이 방식은 비용이 추가로 발생할 수 있습니다.

원하는 방식으로 선택하면 됩니다.

이후 환경 구성에서 EC2를 선택하고 EC2를 생성할 때 만들었던 태그를 검색해서 모두 넣어줍니다. 해당 태그에 해당하는 모든 인스턴스에 배포를 진행합니다.

배포 설정은 여러 가지가 있지만 이번 글에서는 세 가지만 소개합니다.

* `AllAtOnce`: 한 번에 모든 인스턴스에 배포합니다. 모든 인스턴스가 멈추는 시간이 발생할 수 있습니다.
* `HalfAtATime`: 절반의 인스턴스에 배포합니다. 즉, 배포하려는 인스턴스가 4개라면 2개씩 배포합니다. 2개 인스턴스에 먼저 배포가 끝났다면 이후 나머지 2개를 배포합니다. 이 방식은 서비스가 중단되는 시간은 없지만, 트래픽이 몰릴 경우 문제가 발생할 수 있습니다.
    * 주의할 점은 배포할 인스턴스 개수가 최소 두 개 이상이어야 합니다. 그렇지 않으면 배포에 오류가 발생합니다.
* `OneAtATime`: 한 번에 한 개의 인스턴스에 배포합니다. 이 방식은 서비스가 중단되는 시간도 없고, 트래픽이 몰릴 경우에도 문제 될 확률이 낮습니다. 다만, 배포 시간이 길어질 수 있습니다.

원하는 방식으로 설정하면 됩니다. 

이후 로드밸런서는 기존에 사용하고 있는 로드밸런서가 있다면 설정해 주면 됩니다. 배포하고 있는 인스턴스로 요청이 오지 않게끔 설정해 주는 옵셥입니다. 로드 밸런서를 사용하고 있지 않다면 체크를 해제하고 넘어가면 됩니다. 로드밸런서를 사용하고 있더라도, 배포 방식이 무중단 패치가 가능하다면 체크를 해제해도 무방합니다.

위와 같이 설정하면 배포 그룹이 생성됩니다.

# 6. appspec.yml 작성

위 내용까지만 하고 배포를 진행하면 S3에 업로드한 파일이 설치만 됩니다.

따라서 프로젝트를 빌드 후 실행하는 코드를 작성해야 합니다.

현재 레포지토리에 있는 `appspec.yml`을 템플릿으로 사용하고, `deploy.sh`에 빌드 및 실행하는 코드를 작성하면 됩니다.

이때 소스 코드를 설치할 폴더 이름이나 경로를 변경하고 싶다면, `files`에 있는 `destination`를 변경하면 됩니다.

현재는 유저 루트 폴더(`/home/ubuntu/`)에 `deploy_folder` 라는 이름으로 소스코드를 설치합니다.

또한, 배포의 시간제한을 설정할 수도 있습니다. `hooks`에 있는 `timeout`을 변경하면 됩니다. 현재는 30분(1800초)으로 설정되어 있습니다.

배포 스크립트를 변경하고 싶다면 `hooks`에 있는 `location`을 변경하면 됩니다.

```yaml
version: 0.0
os: linux
files:
  - source: /
    destination: /home/ubuntu/deploy_folder/
    overwrite: yes

permissions:
  - object: /
    pattern: "**"
    owner: ubuntu
    group: ubuntu

hooks:
  ApplicationStart:
    - location: deploy.sh
      timeout: 1800
      runas: ubuntu
```

다음 코드는 `deploy.sh`의 예제 파일입니다.

다음 코드는 예시 이므로, 실제로는 원하는 방식대로 작성하면 됩니다.

```shell
#!/bin/bash

cd /home/ubuntu/deploy_folder && ... # 빌드 및 실행 코드
```

# 7. Github Action 설정

깃허브 액션은 푸시, 풀 리퀘스트 등 이벤트가 발생했을 때 미리 작성한 yml 코드를 토대로 깃허브 서버 내 도커에서 코드를 실행해 주는 도구입니다.

깃허브 액션을 사용하기 위해 Git Repository의 최상위 폴더를 기준으로 `.github/workflows/**.yml` 파일을 작성해야 합니다.

주석을 읽으며 기입해야 하는 정보를 채워 넣거나 필요한 부분이 있다면 추가하면 됩니다. (컴파일 후 압축 등)

아래 예제는 `product` 브랜치에 merge됐을 경우 배포를 진행하는 코드입니다.

만약 `product`브랜치에 push됐을 경우 배포를 진행하고 싶다면, 아래 주석을 참고하여 수정하면 됩니다. (push 옵션 주석 해제 및, pull_request 옵션 주석 처리)

```yaml
{% raw %}
name: product_deploy

on:
  # 프로덕트 브랜치에 푸시할 때도 액션을 실행합니다.
  # push:
  #   branches: [ product ]

  # 풀 리퀘스트가 closed 즉 머지 됐을 때 실행합니다.
  pull_request:
    types: [ closed ]
    branches: [ product ]
  
  # 위 두 옵션을 같이 켜게 되면, 풀 리퀘스트가 머지됐을 때 액션 두 개가 동시에 실행됩니다.
  # 따라서 한 옵션만 켜야 합니다.

  # github 웹에서 수동으로 액션을 실행할 수 있도록 해주는 옵션입니다.
  workflow_dispatch:

jobs:
  # job 이름입니다.
  deploy:
    # job이 실행될 도커 이미지입니다. 다른 버전을 사용하고 싶다면, 버전을 변경하면 됩니다.
    runs-on: ubuntu-18.04
    # job의 workflow입니다. 순서대로 실행됩니다.
    steps:
    # checkout을 사용하면 $GITHUB_WORKSPACE 변수로 프로젝트가 깔려있는 위치를 알 수 있습니다.
    - name: checkout release
      uses: actions/checkout@v3

    # Github Action의 Secret을 설정하면 다음 명령어를 사용해서 접근할 수 있습니다.
    # ${{ secrets.설정한_SECRET_이름 }}
    # 아래 내용은 github secret에 등록해 놓은 .env파일을 프로젝트 내부에 설치하는 명령입니다.
    # 간단한 예시이므로 필요 없다면 지우거나 원하는 대로 변경하면 됩니다.
    - name: install .env
      run: |
        echo "${{ secrets.DOT_ENV_PRODUCT }}" > $GITHUB_WORKSPACE/config/.env.product.env

    # 설명 생략
    - name: test
      run: echo "No Tests"
    
    # 현재 프로젝트에 포함되는 모든 내용을 압축해서 저장하는 내용입니다.
    # 이후 s3에 업로드하기 위해 압축을 진행합니다.
    # 현재는 "deploy_server.tar.gz" 라는 이름으로 압축합니다.
    - name: archive deploy_server
      run: tar cvfz $GITHUB_WORKSPACE/deploy_server.tar.gz *
      
    # s3,code_deploy에 접근하기 위해 AWS cli 설정을 진행합니다.
    # 쉽게 이야기하면, AWS 서비스를 이용하기 위해 인증을 진행하는 과정입니다.
    - name: AWS configure credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_KEY }}
        aws-region: ap-northeast-2
        
    # S3에 업로드합니다.
    # 업로드할 리전과 버킷은 현재 "ap-northeast-2", "deploy_bucket"입니다.
    # 업로드할 파일 이름은 현재 "deploy_server.tar.gz" 입니다.
    - name: upload to S3
      run: aws s3 cp --region ap-northeast-2 $GITHUB_WORKSPACE/deploy_server.tar.gz s3://deploy_bucket
  
    # AWS CodeDeploy를 실행합니다.
    # 이후 appspec.yml 을 서버에서 실행합니다.
    - name: deploy with AWS codeDeploy
      run: aws deploy create-deployment
        --application-name github-action # [이 주석은 실제 실행 시 지워야 합니다. 지우지 않으면 에러 발생 ]code deploy 어플리케이션 이름입니다. 현재는 "github-action"으로 등록되어 있습니다.
        --deployment-group-name github-action-group # [이 주석은 실제 실행 시 지워야 합니다. 지우지 않으면 에러 발생 ]code deploy에서 설정한 deploy group 이름입니다. 해당 그룹에 포함되어 있는 모든 인스턴스에 배포가 진행됩니다. 현재는 "github-action-group"으로 등록되어 있습니다.
        --file-exists-behavior OVERWRITE # [이 주석은 실제 실행 시 지워야 합니다. 지우지 않으면 에러 발생 ]파일이 존재한다면 덮어씁니다. 만약 버전 관리를 한다면 이 구문을 빼고 버전 및 에러 핸들링을 해야 합니다.
        --s3-location bucket=deploy_bucket,bundleType=tgz,key=deploy_server.tar.gz # [이 주석은 실제 실행 시 지워야 합니다. 지우지 않으면 에러 발생 ]업로드할 S3 버킷 이름과, 키입니다. 키는 위에서 업로드한 S3 키를 넣어주면 됩니다. 현재는 "deploy_bucket", "deploy_server.tar.gz" 입니다.
        --region ap-northeast-2 # [이 주석은 실제 실행 시 지워야 합니다. 지우지 않으면 에러 발생 ]AWS Region입니다. AWS 인스턴스를 생성한 리전을 넣어주시면 됩니다.


  # 여러 잡을 생성 해서 needs를 설정하면 위상정렬된 순서로 실행이 보장됩니다.
  # needs를 설정하지 않는다면, 병렬로 실행됩니다.
  # after_deploy:
  #   needs: deploy
  #   runs-on: ubuntu-18.04
  #   steps:
  #   - name: checkout release
  #     uses: actions/checkout@v3

  #     #...
{% endraw %}
```

# 8. Github Secrets 설정

깃허브 시크릿은 깃허브 액션에서 사용할 비밀 데이터를 저장하기 위해 사용합니다.

예를 들어, DB 비밀번호, 혹은 이번 예제에서 사용할 AWS Keys를 저장하는 데 사용됩니다. 깃허브 시크릴을 이용하지 않으면, 코드에 직접 입력해야 하지만, 이렇게 되면 보안상 취약한 정보가 빠져나갈 수 있습니다.

깃허브 시크릿을 사용하면 보안상 이슈를 해결할 수 있습니다.

`Repository > Setting > Secrets & Variables > Actions`에서 시크릿을 추가할 수 있습니다. Repository의 종속성이 있는 시크릿은 Repository Secret으로 추가하면 되고, 만약 같은 Organization의 여러 Repository에서 사용한다면, Organization Secret으로 추가하면 됩니다.

이 프로젝트에서는 `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `DOT_ENV_PRODUCT` 파일을 시크릿으로 저장합니다.

`AWS_ACCESS_KEY`와 `AWS_SECRET_KEY`는 1단계에서 생성한 IAM User의 Access Key와 Secret Access Key를 넣어주면 됩니다.

`DOT_ENV_PRODUCT`는 프로젝트 내부에 있는 `.env` 파일을 시크릿으로 저장한 것입니다. 만약 필요 없다면, 설정하지 않아도 됩니다.

또, 추가적인 비밀 데이터가 존재한다면, 필요한 만큼 저장해 주면 됩니다.

# 9. 배포

위 과정을 모두 진행했다면, 이제 배포를 진행할 수 있습니다.

7단계에서 설정한 Github Action을 실행하면, 배포가 시작됩니다. `Repository > Actions` 탭에서 실행 결과를 확인할 수 있습니다.

Github Action이 오류 없이 실행됐다면, Code Deploy에서 배포가 진행됩니다.

`AWS Console > CodeDeploy > 배포` 탭에서 배포가 진행되는 것을 확인할 수 있습니다.

`{배포 ID} > view events`를 통해 배포 과정을 확인할 수 있습니다. 오류 로그도 확인할 수 있습니다.

# Conclusion

이번 글에서는 Github Action과 AWS Code Deploy를 이용하여 CI/CD를 구축하는 방법에 대해 알아보았습니다.

CI/CD를 한 번만 구축하면, 코드를 배포하는 과정을 자동화할 수 있습니다. 따라서, 코드를 배포하는 과정을 반복적으로 수행해야 하는 경우에 유용합니다.