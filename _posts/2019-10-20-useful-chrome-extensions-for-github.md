---
layout: post
title: "유용한 Github 관련 크롬 익스텐션 소개"
date: 2019-10-20 19:03
author: taeguk
tags: [Github, Chrome Extension, Open Source]
---

안녕하세요! 제 이름은 [taeguk](https://taeguk.github.io)입니다~ (~~갑자기?~~) <br/>
오늘은 가벼운 주제로 포스팅해보려고 합니다. 다들 github 많이 사용하실텐데요~ <br/>
오늘은 제가 사용하는 크롬 확장 프로그램들중에서 github 를 사용할 때 아주 유용한 것들을 소개해드리는 시간을 가져보려고 합니다^^ <br/>

## 1. Refined Github

이거는 진짜 필수입니다!! <br/>
이 확장 프로그램은 github 웹페이지 자체를 아주 화끈하게(?) 바꿔버립니다! 정말 UI 가 더 예쁘고 사용성있게 변경되는데요. 그 외에도 다양한 기능들을 추가로 제공해줍니다. <br/>
예를 들면 다음과 같습니다. 아래 사진은 익스텐션 적용 전 & 후 인데요. 적용 후에 UI 가 더 정돈되는걸 볼 수 있습니다. 뿐만아니라 `OpenAll` 과 같은 버튼을 추가적으로 지원하는 것을 볼 수 있습니다. <br/>

![](https://lh3.googleusercontent.com/1V5EJACyN-SniIkGVZW9twc56UztvAAHdXEpwdbWmT-7eGIIrqMIvCIquXLwr9I7hWrDvnukAEHY) <br/>
![](https://lh3.googleusercontent.com/wlp-5jc9ILh7Nlg0hlGlPvB_0JbnX8wlA_cqWRZQAHf7bRz6Wh69QeXzknG-xt6qOzLig7Ee5jRd) <br/>
그 외에도 제공해주는 기능이 무척 많은데요! 자세한 것은 [깃허브](https://github.com/sindresorhus/refined-github)를 참고하시기 바랍니다. <br/>
참고로 오픈소스계의 유명인사(?) [Sindre Sorhus](https://github.com/sindresorhus) 님이 개발하셨습니다! 짱짱bb <br/>


## 2. Octotree

이건 github 상에서 코드베이스를 분석할 때 매우 유용한 확장 프로그램입니다. <br/>
보통 github 에서 소스파일을 왔다갔다 하면서 코드를 보기가 매우 불편한데요. Octotree 를 사용하면 마치 IDE 를 사용하는 것처럼 편하게 여러 개의 소스파일들을 드나들면서 코드 분석이 가능합니다. <br/>

![](https://lh3.googleusercontent.com/jBhIp5cOgZBZyWf3b_mxKCy0LaHZ6hjRzU0Da0WOMs1gT00F-mT0iAI0SfSexAFnx9tCnusLg_Q2) <br/>
위 사진에서 볼수있듯이 브라우저의 왼편에 navigator 가 생겨서 그걸 이용해서 편하게 원하는 소스파일을 찾아갈 수 있습니다. <br/>

![](https://lh3.googleusercontent.com/vl9J8jfqAbiFs8Ku7xoFuVsRWQVpj1NKoZdaeVdo_f-bxOQTnQywj_IeRan6cbqM0wDEQdBR-Mvc) <br/>
또 추가적으로 특히나 좋은 것은 PR 의 diff 를 확인할때 유용하게 활용이 가능하다는 것입니다. 이 경우에는 변경된 파일들에 대해서만 트리가 구성되서 코드리뷰시 아주 유용하게 활용이 가능합니다 :) <br/>

![](https://lh3.googleusercontent.com/x4_CnIfebhy-Q_tYwwOgN5wVbSOp0PIlskcAduNSTSkVY8dXpRxHTboKd9R_YqwuFfeQZwFtG_B9) <br/>
근데 octotree 를 사용하시다보면 위와 같은 에러를 볼 때가 있을텐데요. 이는 octotree 가 내부적으로 사용하는 Github API 에서 API 횟수 제한을 두고 있기 때문에 발생합니다.  <br/>
[Github API v3 문서](https://developer.github.com/v3/#rate-limiting)에 따르면 횟수 제한은 인증이 되지 않은 경우에 대해서는 1시간에 60번, 인증이 된 경우에는 1시간에 5000번이라고 합니다. <br/>
따라서 개인 계정에 대한 Github access token 을 생성하셔서 octotree 에 등록하시면 이러한 제한이 훨씬 넉넉해지므로 큰 문제없이 사용하실 수 있습니다. 또한 private repository 에 대해서도 octotree 를 사용하실 수 있게 됩니다. <br/>

![](https://lh3.googleusercontent.com/fZBvZhDGOkjxWSQcgShliK-RW-2lkfSfWUafzCH4y5dltdQhkzMJkJKAVOn6hL8Q0TGvg01WgoEA) <br/>
만약 보안상의 이유로 private repository 에 대한 접근을 허용하기 싫으시면 위와 같이 체크박스를 다 해제하시면 됩니다~ <br/>

이러한 API 횟수 제한은 앞으로 설명할 확장 프로그램들에서도 공통적으로 존재하는 문제이므로 참고하시기 바랍니다. <br/>


## 3. Git History Browser Extension

이건 [githistory.xyz](https://githistory.xyz/) 라는 서비스를 github 상에서 편하게 사용할 수 있게 하는 확장 프로그램입니다. <br/>
![](https://lh3.googleusercontent.com/_yurC7GzvNgbehUayCjASz0B-CUGB_Ce0fM40pAdXC_n_e3YQLVL1XYl8Rn-pAVzM-Ox80my02p-) <br/>
확장 프로그램을 설치하게 되면 github 상에서 소스코드를 볼 때 위와 같이 `Open in Git History` 라는 버튼이 추가로 생기게 되는데요. <br/>
이 버튼을 클릭하면 githistory.xyz 사이트로 이동이 되게 됩니다. <br/>

![](https://lh3.googleusercontent.com/S4_h5ZfPM4akC1IroI4ENWm3xJHAQfqsqqzvrrr1vWC8ki50W8BoNanbD9m-M0AF9IqXkcgFQdhU) <br/>
그러면 위와 같이 아주 멋있는 UI/UX 로 git history 를 볼 수 있게 됩니다. 마우스 혹은 키보드 화살표를 이용해서 커밋들을 이동할 수 있고 이동할때마다 변경사항들이 임팩트있게 보여지게 됩니다. <br/> 
궁금하신 분들은 [이 링크](https://github.githistory.xyz/taeguk/System-Programming-SICXE-Project/blob/master/opcode.c)를 클릭하셔서 한번 확인해보세요~ <br/>


## 4. Github Gloc

이 확장 프로그램은 리포지토리의 라인수를 보여줍니다! <br/>

![](https://lh3.googleusercontent.com/-3uaPtuBbTrzkdBVUhLKYWLgbMdQC1WnDq8fJP5CDdiYJ1S_mW6ABWkGQPp2pvsj6RpF6j3_wlDk) <br/>
위와 같이 리포지토리를 볼 때 전체 라인 수가 나오게 됩니다. <br/>
Github 를 돌아다니면서 여러 오픈소스들을 구경할 때 전체 코드베이스의 규모가 어느정돈지 궁금할 때가 있는데요. 이럴때 이 확장 프로그램을 이용해 라인 수를 알고나면 규모를 파악하는데 도움이 됩니다~ <br/>

![](https://lh3.googleusercontent.com/ZiXyPH4NgD-u-aEOa80tgIr8N_ejHnlKJKnxcrY7AQdHjNQ9LN8i8qh-4siJluzIzwoiR-nzMl8m) <br/>
위와 같이 개인 혹은 단체의 리포지토리 목록을 확인할 때에도 라인 수를 확인할 수 있습니다. <br/>


## 5. ColorGit

이건 제 친구가 개발한 확장 프로그램인데요! Github 의 컬러 스킴을 바꿔줍니다~ <br/>

![](https://lh3.googleusercontent.com/qd1iAk1CFvCZd0j_7OYuG9UbOxJdeUg-jAQY-2oBrz_9Llf-VK-r8F4WJPUWVCf8RXTWrRcV2SfT) <br/>
![](https://lh3.googleusercontent.com/DjPSeWRkvt2J8yFP8tJqGZM8cg4XtKN3QhnZN0jCzCV1aVdDMJ4ZZ9VhlnhTEuYgmwb4jITsFipl) <br/>
위와 같이 컬러 스킴을 선택할 수 있고 그에 맞게 github 사이트의 색깔들이 바뀌게 됩니다~ <br/>


### 마무리

오늘은 제가 사용하는 Github 크롬 확장 프로그램들을 소개해드리는 시간을 가졌습니다! <br/>
여러분들도 이러한 확장 프로그램들을 이용해서 더 즐거운 Github Life 가 되시길 바랍니다~ <br/>

아래는 오늘 소개한 확장 프로그램들의 github 리포지토리 링크입니다~ <br/>
* [Refined Github](https://github.com/sindresorhus/refined-github)
* [Octotree](https://github.com/ovity/octotree)
* [Git History Browser Extension](https://github.com/LuisReinoso/git-history-browser-extension)
* [Github Gloc](https://github.com/artem-solovev/gloc)
* [ColorGit](https://github.com/ddyokim/colorGit)

**[이 포스팅을 taeguk 블로그에서 보기](https://taeguk2.blogspot.com/2019/10/github_20.html)**
