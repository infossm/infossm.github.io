---
layout: post
title:  "알고 나면 유용한 TeX 팁들"
date:   2019-03-10 20:30
author: evenharder
tags:   LaTeX tips
---

TeX은 Donald Knuth가 만든 조판 언어이며, 수많은 분야에서 논문, 책자, 강의 자료 등을 만드는 데 사용됩니다. TeX에 대한 찬양을 하기에는 여백이 너무 부족하므로 생략하도록 하겠습니다. 이 포스트는 TeX 설치법이나 입문 또한 다루지 않습니다. 해당 내용은 Overleaf 같은 온라인 TeX 편집기 사이트에서 찾아보시길 바랍니다. 대신, 이 포스트는 어느 정도 TeX을 쓸 수 있는 분들에게 유용할 (혹은 이미 알고 있을) 팁을 다룹니다.

## 그림 폴더 경로 설정
일반적으로 TeX에 이미지를 넣을 때는 `\includegraphics`를 사용합니다 (`graphicx` 패키지 필요). 이 때 파일의 경로를 지정해야 하는데 이미지가 TeX 파일과 같이 있으면 난잡하기 때문에 `img` 폴더를 별도로 만드는 경우가 많습니다. 이 안에 `fig01.png`가 있으면 `\includegraphics{./img/fig01.png}`를 사용하게 됩니다.

어쩌면 `img` 폴더에서 불러올 이미지가 많을 수도 있습니다. 이 때 유용한 명령어는 다음과 같습니다.
{% raw %}
```tex
\usepackage{graphicx}
\graphicspath{{./img/}} % maybe graphicx?
```
{% endraw %}
저 코드를 추가하면 `\includegraphics{fig01.png}`만 써도 됩니다. 여러 개의 폴더를
{% raw %}
```tex
\usepackage{graphicx}
\graphicspath{{./img01/}{./img02/}{./img03/}}
```
{% endraw %}
등으로 추가할 수 있습니다.

## 제목 여백 조정

기본적으로 여백은 `geometry` 패키지로 관리하며, `\usepackage[top=3cm, left=3cm, right=3cm, bottom=2cm]{geometry}`의 형태로 호출합니다. 다만 기본적으로 첫 장에
`\maketitle`로 제목을 만들고 나면 예상보다 제목과 상단 사이의 여백 큰 경우가 많습니다. 이를 줄이는 좋은 방법 중 하나는 다음과 같습니다.

```tex
\usepackage{titling}
\setlength{\droptitle}{-2.5cm}
```
다른 설정을 건들 필요도 없고 `\vskip` 등의 비기도 사용할 필요가 없어서 편합니다.

## 단원 이름 등 보조 서식 변경
`kotex` 패키지를 사용하면 한글 문서를 다루기가 대단히 편해집니다. 절 이름도 `Section`에서 `절`로 바뀌며, `Figure`도 `그림`으로 바뀌는 등 많은 작업이 밑바탕에서 이루어집니다. 그러나 때로는 영어 명칭을 그대로 유지하거나, 한글 명칭 구조를 변경해야 할 때가 있습니다. 이 때 사용하는 명령어가 `\kscntformat`입니다.

[한국어 텍 v2.0 사용 설명서](http://ctan.math.washington.edu/tex-archive/language/korean/kotex-utf/doc/kotexdoc.pdf)의 '5.3.2 우리말 이름'부터 해당 명령어의 사용 방법이 나와 있습니다.

```tex
\kscntformat{단원 이름}{앞}{뒤}
```

예시로, `제 {절 번호} 절` 꼴로 나오는 `\section`을 `{절 번호}.`로 바꾸고 싶다면

```tex
\kscntformat{section}{}{.}
% 기존엔 \kscntformat{section}{\ksTHE}{\sectionname}으로
% `제 {절 번호} 절`로 표현됨
```
을 사용하면 되며, `제`나 `절` 등의 글씨를 바꾸고 싶으면 `\renewcommand`를 통해 바꾸면 됩니다. 해당 명령어의 사용 예시는 다음과 같습니다.
```tex
\renewcommand*{\proofname}{풀이}
\renewcommand*{\tablename}{Table}
```

## XeTeX을 통해 글꼴 바꾸기
XeTeX은 유니코드, 오픈타입 폰트 등을 지원하는 TeX 엔진입니다. 당연히 한글 글꼴도 적용할 수 있으며, 맑은 고딕 폰트를 사용하여 거의 한글 문서와 비슷하게 만들 수도 있습니다. 한글 글꼴은 알파벳 글꼴과는 달리 기울임을 직접적으로 지원하지 않기 때문에 fake slanting을 이용하여 표현해야 합니다. 이는 다음의 코드로 문서에 적용할 수 있습니다.

```tex
\usepackage[hangul]{xetexko} % compile with XeTeX
\setmainfont{맑은 고딕}
\setmainhangulfont[ItalicFont={*},ItalicFeatures={FakeSlant=.167}]{맑은 고딕}
```
## 줄바꿈이 가능한 셀 생성
TeX에서 생성되는 표는 기본적으로 줄바꿈이 일어나지 않습니다. 때문에 줄바꿈을 사용하고 싶으면 `\multirow` 등의 명령을 이용해야 하고 조금 사용법이 복잡합니다. 그러나 `\makecell` 명령을 사용하면 (`\makecell` 패키지 필요) 쉽게 만들 수 있습니다.

사용 예시는 다음과 같습니다.
```tex
\documentclass{article}
\usepackage{makecell}
\begin{document}
    \begin{table}[h]
        \centering
        \begin{tabular}{|c|l|c|}
            \hline
            Lorem & \makecell{Ipsum \\ (Dolor)} & Sit Amet \\ \hline
        \end{tabular}
    \end{table}
\end{document}
```
기본적으로 `\makecell`을 통해서 만들어지는 셀엔 가운데 정렬(`c`)이 적용됩니다. 다른 옵션을 사용하고 싶으면 `\makecell[r]{}` 등으로 옵션을 넣으면 됩니다.

## 유니코드 책갈피 생성
TeX으로 pdf 파일을 컴파일할 때 `\section` 등의 명령어로 책갈피가 만들어지는 경우가 있습니다. TeX의 기본 설정으로는 유니코드로 구성된 책갈피는 깨져보입니다. 이를 방지하기 위해 

```tex
\usepackage[unicode,bookmarks=true]{hyperref}
```
로 옵션을 전달해주면 유니코드로 구성된 책갈피도 깨지지 않고 잘 표현됩니다.

## SI 단위계 사용

SI 단위계를 쓸 때 어떻게 쓰시나요? 직접 알파벳으로 쓰시나요? 아니면 해당 유니코드를 찾아서 쓰시나요? TeX으로는 `siunitx` 패키지를 통해 SI 단위계 사용을 읽는 것만큼 쉽게 할 수 있습니다. 간단한 예제는 다음과 같습니다.
```tex
\documentclass{article}
\usepackage{siunitx}
\begin{document}
    \begin{itemize}
        \item 9.8 \si{\metre\per\second\square}
        \item \SI{3386}{\newton}
        \item \SI{261.3}{\joule\per\sec}
        \item 9.8 \si[per-mode=symbol]{\metre\per\second\square}
        \item \SI{761}{\mmHg}
        \item \SI{530}{\kilo\cal}
    \end{itemize}
\end{document}
```
슬래시를 써서 사용할지, 첨자로 `-1`제곱을 표현할지도 조절할 수 있습니다.

## enumerate 환경의 글머리 기호
`enumerate` 환경은 ordered list를 만들며, 기본적으로 1. / 2. / 3. / ... 꼴로 글머리 기호가 붙습니다. 알파벳이나 로마자를 순서대로 쓰고 싶을 경우 다음과 같이 옵션을 추가하면 됩니다.

```tex
\documentclass{article}
\usepackage{enumitem}
\begin{document}
    \begin{enumerate}[label=\Alph*.] % A. / B. / C.
        \item A new world
        \item Let it be
        \item Under the sea
    \end{enumerate}
    \begin{enumerate}[label=(\roman*)] % (i) / (ii) / (iii) / (iv) / (v) / (vi) / (vii)
        \item Whatever goes upon two legs is an enemy.
        \item Whatever goes upon four legs, or has wings, is a friend.
        \item No animal shall wear clothes.
        \item No animal shall sleep in a bed.
        \item No animal shall drink alcohol.
        \item No animal shall kill any other animal.
        \item All animals are equal.
    \end{enumerate}
\end{document}
```

## 더미 텍스트 대입
Lorem ipsum dolor sit amet이라는 유명한 구절을 알고 계신지 모르겠습니다. 더미 텍스트의 대명사인 Lorem ipsum은 글씨 배치가 어떻게 될지 테스트하기 위해 흔히 사용됩니다. 사용 예시는 다음과 같습니다.

```tex
\documentclass{article}
\usepackage{lipsum
\title{Title here}
\author{\texttt{@evenharder}}
\begin{document}
    \maketitle
    \lipsum[1]    
\end{document}
```

## 정리하며
TeX은 정말 방대한 프로그램이며, '이걸 어떻게 하면 나타낼 수 있을까' 등의 문제는 이미 [TeX Stack Exchange](https://tex.stackexchange.com/)에 나와있는 경우가 많습니다. 가끔씩은 이미 방문했던 문서에 다시 들어가는 슬픈 일이 일어나기도 하는데, 본 포스트처럼 본인이 알고 있는 팁들을 하나씩 쌓아놓는 것도 좋은 습관일 듯 합니다. 이후에도 종종 TeX 포스팅으로 찾아뵙겠습니다.
