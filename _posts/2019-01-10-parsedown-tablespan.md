---
layout: post
title: Parsedown Tablespan
date: 2019-01-10 23:00
author: KENNYSOFT
tags: [PHP, Parsedown, Composer]
---

## 들어가며

필자의 블로그는 이 글을 쓰는 지금 현재 Markdown을 PHP의 Parsedown을 통해 서비스하고 있다. 그런데 이 사용을 확대하려고 보니 표에서 rowspan, colspan 등으로 일컬어지는 셀 병합 기능을 나타낼 수 있는 문법이 없었다. 어느 서비스에서 개별적으로 사용하려고 대충 정해 놓은 것은 있지만 어쨌든 공식적으로는 존재하지 않는다.

MultiMarkdown에서 제공하는 'Long Cell' 문법<sup>[#](http://fletcher.github.io/MultiMarkdown-6/syntax/tables.html)</sup>이 있으나 이를 빌드하기가 쉽지 않았다. 이에 나도 개인적으로 사용하기 위해 문법을 하나 만들어 보고 이에 맞추어 직접 Parsedown의 extension을 개발해 보기로 했다.

## 문법

원체 Markdown의 표 문법이 단순해서 특별한 장치를 만들기는 어려워 보였다. 이에 단지 셀이 병합될 방향을 나타내면 어떨까 했다. 예를 들면 다음과 같다.

```markdown
| Column A | Column B | Column C |
| :------: | :------: | :------: |
| 1x2 Cell |    <     | 2x1 Cell |
|    v     | 1x1 Cell |    ^     |
| 2x1 Cell |    >     | 1x2 Cell |
```

이 표가 아래와 같이 렌더링 되었으면 하는 것이다.

<table>
<thead>
<tr>
<th style="text-align: center;">Column A</th>
<th style="text-align: center;">Column B</th>
<th style="text-align: center;">Column C</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;" colspan="2">1x2 Cell</td>
<td style="text-align: center;" rowspan="2">2x1 Cell</td>
</tr>
<tr>
<td style="text-align: center;" rowspan="2">2x1 Cell</td>
<td style="text-align: center;">1x1 Cell</td>
</tr>
<tr>
<td style="text-align: center;" colspan="2">1x2 Cell</td>
</tr>
</tbody>
</table>

다만 고려한 것 중 `v`의 경우 단순 알파벳으로 일반적으로도 쓰일 수 있어서 제외하기로 했다. 물론 `\`로 이스케이프 할 수야 있겠지만 일반 사용에 불편할 수 있기 때문이다.

`<`와 `>`의 경우 일단은 굳이 양방향이 필요한가 해서 일단은 `>`만을 사용하도록 하였다.

## PHP

Parsedown이 extension에 대해 확장성 있게 설계되었기에 ParsedownExtra를 기반으로 한 클래스를 생성했다.

`blockTableComplete()` 함수는 표를 모두 인식했을 때 후처리해야 할 것을 정의하는 함수로, 상위 클래스에서 정의된 적조차 없으므로 super를 쓸 필요가 없다.

소스코드<sup>[#](http://github.com/KENNYSOFT/parsedown-tablespan/blob/master/ParsedownTablespan.php)</sup>는 병합 대상 문자가 있는 경우 속성과 함께 잘 합치는 것인데, 일일이 설명하는 것은 별 의미가 없어 보여서 간단한 팁을 모아 보았다.

* 일반적인 비교에 `===`를 쓰자.
* C++처럼 Reference 자료형이 있어서 이를 받아오기 위해 `=&`를 사용했다. 그 자체가 연산자는 아닌 것 같은데 C++에서 일반적으로 쓰는 공백과는 다르게 사용한다.
* `foreach`는 `foreach ($Array as $Index => $Item)` 형태로 사용할 수 있다.
* 한편 위에서 `$Item`은 새로운 변수가 생성되는 것이므로 원래 `$Array[$Index]`는 변하지 않는다. 직접 다시 반영시켜주기보다는 `foreach ($Array as $Index => &$Item)`을 쓰면 좋다.
* 빈 배열은 `array()`로 만든다.
* 배열의 길이는 `count($Array)`로 구한다.
* 배열의 마지막에 값을 삽입하는 것은 `[]=`이다. 역시 그 자체가 연산자는 아닌 것 같다.
* `if (!isset($A)) return $A; else return null;` 같은 처리를 한 문장 안에서 해야 할 때가 있다. 그럴 때는 `@$A ?: null`로 간편하게 사용할 수 있다. `@`를 붙이는 것은 그냥 사용 시 발생하는 경고를 방지하기 위해서이다.

## 릴리즈<sup>[#](http://github.com/KENNYSOFT/parsedown-tablespan/releases)</sup>

```php
$ParsedownTablespan = new ParsedownTablespan();

echo $ParsedownTablespan->text('
| >     | >           |   Colspan    | >           | for thead |
| ----- | :---------: | -----------: | ----------- | --------- |
| Lorem | ipsum       |    dolor     | sit         | amet      |
| ^     | -           |      >       | right align | .         |
| ,     | >           | center align | >           | 2x2 cell  |
| >     | another 2x2 |      +       | >           | ^         |
| >     | ^           |              |             | !         |
');
```

<table>
<thead>
<tr>
<th style="text-align: center;" colspan="3">Colspan</th>
<th colspan="2">for thead</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Lorem</td>
<td style="text-align: center;">ipsum</td>
<td style="text-align: right;">dolor</td>
<td>sit</td>
<td>amet</td>
</tr>
<tr>
<td style="text-align: center;">-</td>
<td style="text-align: right;" colspan="2">right align</td>
<td>.</td>
</tr>
<tr>
<td>,</td>
<td style="text-align: center;" colspan="2">center align</td>
<td colspan="2" rowspan="2">2x2 cell</td>
</tr>
<tr>
<td style="text-align: center;" colspan="2" rowspan="2">another 2x2</td>
<td style="text-align: right;">+</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td></td>
<td>!</td>
</tr>
</tbody>
</table>

위와 같은 예제를 만들고 README를 작성하여 릴리즈했다.

Parsedown repository의 Wiki 중 'Extensions and Related Library' 페이지<sup>[#](http://github.com/erusev/parsedown/wiki/Extensions-and-Related-Libraries)</sup>에 링크도 추가했다.

## Composer

언어마다 하나쯤 있는 패키지 모음 및 의존성 해결 라이브러리의 PHP 버전 정도로 이해했다.

등록하는 것은 어렵지 않다. GitHub에 릴리즈한 repository 주소를 제출하면 자동 검사 후 양식에 맞는 경우 바로 등록된다. 등록된 페이지<sup>[#](http://packagist.org/packages/kennysoft/parsedown-tablespan)</sup>에는 dependency 등의 간단한 정보와 README가 표시된다.

이제 설치하기 위해서는 다음 명령어를 실행하면 된다.

```shell
composer require kennysoft/parsedown-tablespan
```

다만 정작 나는 Composer를 PHP 개발을 하면서 써본 적이 없기는 하다.

## HTML5

정렬의 경우 `<td>`에 `style` 속성으로 `text-align` 값을 설정하면 되고 이미 Parsedown이 그렇게 하고 있다. 한편 GitHub에 README를 작성할 때 각 셀에는 해당 속성이 적용되지 않는다.

이에 HTML5에서 지원이 제거되기는 했지만, `align` 속성을 주면 Chrome 71 기준으로 여전히 작동하기는 한다. 그래서 README만을 위해 해당 방법으로 표를 만들었다. 한편 이렇게 등록하니 Composer에서 README를 긁어갔을 때 정렬이 또 깨진다. 두 가지를 동시에 적는 것이 좋겠다.

## 개선 계획

단지 내가 쓰려고 만든 것이고 별 계획 없이 즉흥적으로 만든 것이라 체계성이 부족하다. 심지어 동아리 스터디 발표에 쓰려고 일찍 릴리즈를 한 감도 없잖아 있어서 이미 생각해 놓은 개선 계획이 몇 가지 있다.

### 좌우 병합 시 정렬 모드 문제

문법에서 `<`와 `>` 중 `>`만 사용할 수 있도록 하였다. 그러나 이 경우 정렬 모드를 어느 칸에 맞춰야 하는지가 확실치 않다. 현재는 우측으로 병합하는 도중 처음으로 설정된 정렬 모드를 계속 따라가게 되어 있는데, 이 경우 의도치 않은 정렬이 될 수 있어 이를 해결하기 위해 두 방향 모두를 지원하여 실제 내용이 들어있는 칸의 정렬 모드로 적용하도록 개선하려고 한다.

### 상하좌우 병합 시 고정 문법 문제

상하좌우 방향으로 모두 병합할 경우 현재는 고정된 순서를 지켜야 한다. 반드시 좌우 병합 후 상하 병합을 해야 하는데 이는 다음과 같다.

```markdown
| Column A | Column B |
| :------: | :------: |
|    >     | 2x2 Cell |
|    >     |    ^     |
```

이에 다음과 같이 상하 병합 후 좌우 병합을 해도 되도록 개선하고자 한다.

```markdown
| Column A | Column B |
| :------: | :------: |
|    >     | 2x2 Cell |
|    ^     |    ^     |
```

### MultiMarkdown 문법 지원

앞서 언급한 MultiMarkdown의 경우 `||` 처럼 내부에 공백조차 없는 셀을 병합하는 것으로 처리한다. Parsedown에서는 저러면 셀이 아닌 것으로 처리해서 해당 행의 열 개수가 부족해지는 현상이 발생한다. 이를 해결하고 싶은데, 다만 `blockTableComplete()`가 아닌 파싱 도중의 동작인 것 같아서 추가 분석이 필요하다.