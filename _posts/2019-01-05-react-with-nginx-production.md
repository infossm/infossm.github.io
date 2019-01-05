---
layout: post
title:  "리액트 배포 및 자동화"
date: 2019-01-05 20:00
author: 서남규
tags: [web, production, nginx, docker, travis]
---

리액트의 개발을 마치고 서버를 통해 배포를 할 때의 과정을 간단하게 정리해보았습니다. 리액트라고 적어 놓긴 했지만 SPA(Single Page Application) 라면 굳이 리액트가 아니어도 같은 방법으로 배포가 가능합니다.  
보통 aws, firbase 같은 CDN 서비스를 이용하여 배포, 관리를 많이 하기도 하지만 이 글에서는 서버에서 직접 nginx, docker을 이용하여 배포하는 과정을 다루며 추가로 github과 연동이 되는 travis를 이용하여 자동화를 합니다.  
  
이 글에서 사용되는 소스 코드는 [이 링크](https://github.com/zych1751/react-production-test)에서 확인할 수 있습니다. 해당 소스 코드는 [create-react-app](https://github.com/facebook/create-react-app)을 이용하여 만들어지는 기본 소스 코드에서 자동화, 배포할 때 필요한 소스 코드들만 추가하였습니다.  

# 개요

![](/assets/images/react-production/preview.png)  
전체적인 동작 과정은 위 그림과 같습니다.  
  
유저가 github에 푸시를 하면(혹은 머지를 하면) travis에서 github에서 코드가 바뀐 것을 확인하고 특정 코드(.travis.yml)를 실행합니다. 여기서 테스트와 빌드를 하게 되고 성공하면 빌드 결과물과 설정 파일들을 도커 이미지 위로 올린 후, 해당 이미지에서 웹 서버 역할을 할 nginx를 실행합니다. 그리고 만들어진 도커 이미지를 docker hub에 올립니다. 서버에서 위 이미지를 pull 받고 도커를 실행하게 되면 배포는 완료됩니다.  

위 자동화가 한번 구축되고 나면 실제로 하게 되는 일은 개발 완료 후 github에 push, 배포 서버에서 docker hub을 pull 받고 실행 이 2과정만 진행하면 배포가 완료가 됩니다. 즉, 매번 빌드를 할 필요가 없어지게 됩니다.  
  
# Github  
  

여기서 git과 github에 관한 자세한 내용은 다루지 않습니다.  
알아둬야 할 것은 travis의 실행은 github에 있는 파일들로만 실행되므로 아래에서 설명하는 필요한 파일(.travis.yml, Dockerfile ...)들은 모두 github에 올라와 있어야 합니다.  
  
# Travis  
  
travis-ci는 이름을 보면 알겠지만 CI(Continuous Integration)를 가능하게 해주는 서비스입니다. 이 서비스를 이용해 코드를 수정할 때마다 테스트와 빌드를 수행하여 문제점을 미리 찾아내 실제 배포를 할 때 검증 및 릴리즈 시간을 줄일 수 있습니다.  

travis를 작동시키기 위해서는 처음으로 [travis-ci.org](https://travis-ci.org)(만약에 레포지토리가 private이라면 [travis-ci.com](https://travis-ci.com))에 레포지토리를 등록해야 합니다.  
  
travis가 작동할 때 수행되어야 할 목표는 아래와 같습니다.  
  
 * react 소스 코드를 빌드 하기
 * 빌드 한 결과를 이용하여 도커 위에 웹 서버 구축하기(nginx 이용)
 * 만들어진 도커 이미지를 docker hub에 업로드하기
  
travis는 깃헙 코드가 수정되거나 빌드 요청을 받았을 때, github 레포지토리를 복사하여 .travis.yml 코드를 읽고 실행하게 됩니다. 위 작업에 해당하는 .travis.yml은 아래와 같이 쓸 수 있습니다.  
```yaml
language: node.js
node_js:
    - "node"

sudo: required

services:
    - docker

install:
    - npm install

script:
    - npm run build

after_success:
    - docker login -u "$DOCKER_USERNAME" -p "$DOCKER_PASSWORD"
    - docker build -t zych1751/react-test .
    - docker push zych1751/react-test
```

npm을 이용하여 리액트의 필요한 라이브러리를 설치하고 빌드 하는 부분까지는 따로 설명하지 않겠습니다.  
여기서 테스트 코드가 따로 있다면 script에 추가하면 됩니다. 테스트, 빌드가 모두 성공하였다면 after_success를 실행하게 되는데 첫번째 명령어를 보면 $DOCKER_USERNAME, $DOCKER_PASSWORD가 있는데 이것은 travis에서 환경 변수로 등록해주어야 합니다.

![](/assets/images/react-production/environment.png)

환경 변수를 추가했다면 문제 없이 로그인 될 것이며 다음 명령어인 docker build를 실행하게 됩니다. docker은 Dockerfile을 바탕으로 도커 이미지를 만들기 때문에 깃헙 레포에 아래 코드를 Dockerfile로 추가해 줍니다.
```docker
FROM nginx:1.14.2-alpine

COPY ./build /var/www/test.zychspace.com
COPY ./nginx.conf /etc/nginx/conf.d/test.zychspace.com.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

위 코드는 docker 안으로 빌드 결과물과 설정 파일을 가져와서 nginx를 실행하는 코드입니다.
여기서 test.zychspace.com은 임의의 이름으로 설정하면 됩니다.  
도커에서 nginx 설정 파일인 nginx.conf를 추가로 해주고 있는데 이 파일 또한 깃헙 레포에 추가해 줍니다.
```nginx
server {
    root /var/www/test.zychspace.com;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    listen 8080 default_server;
}
```

react와 같은 SPA에서는 html 파일 자체는 1개만 사용하기 때문에 try_files를 하지 않게 되면 http://test.zychspace.com 와 같은 요청과 그 안에서 상대주소(/path1/path2)를 통해 이동하는 것은 문제가 없겠지만 도메인이 붙어버린 절대주소(http://test.zychspace.com/path1/path2)와 같은 이동에서는 웹 서버에서 root/path1/path2 파일을 찾게 되는데 해당 파일을 찾을 수 없어 404응답을 내려버립니다. 이를 해결하기 위해 try_files를 추가한 nginx.conf 파일을 추가해 절대주소가 어떻게 와도 index.html을 렌더하고 그 뒤에 내부 리액트 라우터에서 처리하게 합니다. 해당 파일을 도커이미지 위에 추가해준 뒤 nginx를 실행시키면 웹 서버의 이미지가 완성됩니다.  
  
마지막으로 해당 이미지를 travis가 docker hub으로 push함으로써 travis는 종료하게 됩니다.  
  
# Docker Hub  
  
github이 소스 코드의 버전 관리를 해주는 호스팅 서비스라면 docker hub은 docker 이미지의 버전관리를 해주는 호스팅 서비스입니다.  
docker hub에서 할 일은 travis에서 docker push를 해주는 레포지토리(위 코드에선 react-test)를 만들어야 합니다.

위 작업을 모두 완료하고 나면 깃헙에 새로 push를 하거나 travis에 빌드를 요청하면 docker hub에 자동으로 업데이트되는 것을 확인할 수 있습니다.  
![](/assets/images/react-production/travis-result.png)
![](/assets/images/react-production/docker-hub.png)

# 배포

배포할 서버에 접속한 후 아래 명령어를 입력해줍니다.
```
docker pull zych1751/react-test
docker run -p 7000:8080 --name react-test zych1751/react-test:latest
```
해당 서버에서 7000포트로 요청이 들어오면 도커 안의 8080포트로 요청이 들어가고 react의 결과물이 나오게 됩니다.

이를 이용해 서버에서 proxy를 이용하여 특정 도메인으로 요청이 들어왔을 때 7000포트로 redirect 해주면 완성됩니다.
아래 코드는 서버에서 proxy 요청을 보내는 코드 예시입니다.
```nginx
# /etc/nginx/conf.d/zychspace.com.conf
server {
    server_name test.zychspace.com www.test.zychspace.com;

    location / {
        proxy_pass http://127.0.0.1:7000;
    }
}
```

위와 같이 하고 원하는 도메인 설정까지 완료되면 아래 처럼 배포가 완성됨을 확인할 수 있습니다.
![](/assets/images/react-production/result.png)

