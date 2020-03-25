---
layout: post
title: Android Studio를 이용한 앱 UI
date: 2020-01-16 10:10
author: cjmp1
tags: app
---

# Android Studio를 이용한 앱 UI

## Contents

- 앱 공부를 시작하며
- 앱 개발의 구성
- Android Studio 구조
- 레이아웃
- 마치며
- Reference

### 앱 공부를 시작하며

앱 공부를 시작하게 된 계기는 백지상태로 참여한 학교에서 주최한 해커톤 대회 때문이었다. 예전에 잠깐 만져본 것 이외에는 아예 처음 접해보는 분야여서 많은 어려움도 있었지만 나름의 매력이 있고 재미있는 분야라는 생각이 들었다. 따라서 대회가 끝나고 난 후에도 조금 더 공부를 해보았고 처음 도전해보는 사람들도 쉽게 접할 수 있고 이해할 수 있는 분야라는 생각이 들어서 글을 작성해보게 되었다.

### 앱 개발의 구성

먼저 앱 개발은 어떻게 구성되어 있는지부터 살펴보자

앱 개발을 하는 과정은 어떤 앱을 만들던지 간에 동일하게 적용되고, 일반적인 응용프로그램 제작과 동일한 절차를 밟는다.

기획 - 디자인 - 개발 - 검토 - 출시/배포 - (업데이트)

1. ##### 기획

   기획 단계에서는 앱의 전반적인 목표를 설정한다고 생각하면 된다. 사용자에게 어떤 도움을 주는 앱을 만들 것인지, 주 목표로 하는 사용자는 누구인지, 수익 구조등에 대해서도 정리가 되어야 한다. 이 과정을 완벽하게 할수록 후에 개발과정에 있어서 문제가 없고 순조롭게 진행될 수 있다. 

   앞부분의 정리가 완료되면 이제 화면 밑그림을 그리는데 전념해야 한다. 화면 밑그림은 각 화면에 담을 정보들과, 각종 기능들 그리고 화면 간의 이동을 정리한 그래프를 그려주는 것을 말한다. 기획자들의 생각들이 모두 일치 될 수 있고 사용자에게 잘 전해질 수 있는 화면을 구성하는 것이 매우 중요하다. 나는 Adobe XD를 이용해서 화면 그래프를 그려보았고 매우 유용했다.

   화면 그래프가 완성되었다면 이제 각종 기술스택을 정해야 한다. 우선 서버의 유무와, 서버 프로그래밍 언어를 정한다. 그리고 OS (안드로이드 iOS) 별로 앱을 따로 만드는 네이티브 앱을 만들 것인지 ,함께 제작할 것인지 아니면 둘 중 하나만 지원하는 앱을 만들 것인지를 결정해야 한다. 이 결정에 따라 프론트엔드 기술 스택이 조정된다. 이 과정이 끝나면 이제 실제 화면 디자인으로 넘어가게 된다.

2. ##### 디자인

   디자인은 OS 별로 차이가 있고 여러가지 예시들을 살펴보면서 구성하는 방법이 좋다. 검색을 통해서 여러가지 테마들을 찾아볼 수 있으며, 가져온 후에 구상한 그림에 맞게 커스터마이징 하는 부분이 중요하다. 디자인 부분에서 경험이라고는 포토샵 밖에없던 나로서 많이 걱정을 했었는데, 디자인의 경우에 기본에 충실하고 의외로 재능의 요소가 그렇게 중요하지 않다는 것을 배울 수 있었다. 디자인 검토가 끝나면 이제 개발 단계로 넘어가게 된다.

3. ##### 개발

   개발은 크게 웹에서 사용하는 언어와 동일하게 프런트와 백으로 나뉜다. 프런트는 앞서 디자인에서 만들어진 UI를 토대로 앱의 얼굴을 만들어내고, 백의 경우네는 API 서버개발, DB설계를 도맡아 진행하게 됩니다. API 서버의 경우내는 서버와 앱이 서로 주고받을 정보를 구성하는 작업으로 앱에서 사용자가 로그인을 하거나, 서버에 데이터를 요구할 때, 이를 전송해주는 역할을 합니다. 또한 DB가 필요한 앱의 경우 그 DB를 관리해주게 됩니다. 저는 서버에는 node.js, DB의 경우에 mongoDB를 사용했고, 프런트는 곧 설명할 android studio Java를 사용했습니다

4. ##### 검토

   앱 개발이 끝나면 바로 출시를 하는 것이 아닌 검토 작업이 필요합니다. 우리가 원하는대로 만들어졌는지를 확인하는 작업도 중요하지만, 실제 사용자가 앱을 사용할 때, 오류가 있거나 생각지 못한 문제가 발생할 수 있습니다. 편의성 문제도 포함이되구요. 따라서 검토작업은 단계에 걸쳐서 할 수도 있고 버전을 나누어 진행을 할 수도 있습니다.

5. ##### 출시

   검토가 끝났다면 이제 우리가 잘 알고있는 마켓에다가 등록을 하게 됩니다. 이제는 마케팅과의 싸움이 시작됩니다.

이렇게 앱 제작의 순서에 대해 알아보았습니다. 그렇다면 이제 제가 주력으로 했던 Android Studio (Java)로 앱 프런트를 구성하는 방법에 대해 알아보겠습니다.

### Android Studio 구조

우선 실제 코딩에 앞서 Android Studio 툴에 대해 간략한 설명이 필요할 것 같습니다. Android Studio 다운로드는 아래 링크를 통해서 자신의 컴퓨터 환경에 맞게 설치를 해주면 됩니다.

[안드로이드 스튜디오 다운로드](https://developer.android.com/studio) 

안드로이드 스튜디오 설치가 완료되었다면 실행후에 첫 프로젝트를 생성해보겠습니다. 

Create New Project를 누른 후에 Empty Activity 를 누르고 Next를 눌러줍니다. Name에는 내 앱의 이름이 들어가게 됩니다. 자동으로 생성되는 저장 위치가 보이고 그 아래에 이제 모든 파일들이 들어가게 됩니다. Language는 Java를 선택해주고, Minimum API level은 내 앱이 지원할 API의 최소 레벨을 의미합니다. 즉 이 값이 낮을 수록 많은 종류의 기기를 지원해주게 됩니다. (예전 device들도 지원이 되는 것입니다.) 검색을 해보고 고르거나, 나중에 바꿀 수 있으니 낮은 버전으로 선택을 해주고 Finish를 누릅니다. 시간이 흐른 후에 프로젝트 화면이 나오게 됩니다. 아래와 같이 나오면 맞게 된 것입니다.

![이미지1](./assets/images/AS_UI/1.png)

다른 개발 툴과 같은 모양을 하고 있습니다. 왼쪽에 프로젝트 뷰, 오른쪽에 코딩을 하는 곳 맨 아래에 빌드가 되고 로그나 결과를 확인 할 수 있는 창이 구성되어 있습니다.

우선 왼쪽 창에 대해 간단히 알아보겠습니다. 크게 app 부분과 Gradle Scripts 란 부분으로 나누어져 있습니다. 이는 안드로이드 스튜디오라는 코드 편집기와 빌드 시스템이 서로 독립적이기 때문입니다. 즉 아래 Gradle 부분은 빌드 절차나 설정에 대한 정보가 작성되어 수정 할 수 있고 실제 앱 모듈 부분은 모두 위에서 작성하게 됩니다. 또한 라이브러리들이 많아짐에 따라 이를 일일히 추가하지 않고 관리해주는 역할도 하게 됩니다. **build.gradle** 파일을 간단하게 살펴보면 여러가지 정보가 저장되있음을 확인 할 수 있습니다. android 단락에 빌드 설정이 들어가게 됩니다. 보시면 Sdk 버전,  buildTool 버전이 작성되어 있고, 앞서 말씀드린 Minimum API level을 수정할 수 있는 minSdkVersion도 있는것을 확인할 수 있습니다. 그리고 맨 아래에 dependencies 부분에서는 추가하는 라이브러리들이 들어가게 됩니다.

이제 위에 있는 app 부분을 살펴보겠습니다. app 부분은 세가지 파트로 구성되어 있습니다. manifest, java, res 입니다. res는 쉽게 연상될 수 있듯이 resource의 약자로, 어플에서 사용될 상수값들이나, 아이콘, 이미지, 문자열 그리고 화면의 바탕을 그려주는 XML layout이 포함됩니다. java 파트에서는 말 그대로 java 소스 코드 파일들을 포함하게 됩니다. manifest는 안드로이드에서 앱 코드를 실행하기 위해 확인하는 파일이라고 생각할 수 있습니다. 내용을 살펴보면, 각 기기에 나타날 앱의 아이콘도 지정이 되어있는 것을 확인할 수 있고,

```java
 <activity android:name = ".MainActivity">
```

이처럼 액티비티들이 정의되어있습니다. activity를 추가하게 되면 이 파일에도 자동으로 수정이 이루어집니다. 안드로이드 액티비티에 대해서는 후에 더 자세히 다뤄보도록 하겠습니다. MainActivity 안에 내용으로 intent-filter 라는 항목이 들어가 있는 것은 수행할 행동을 지정해 주는 것인데요, 처음 초기 메인 화면으로 MainActivity를 실행하겠다는 뜻을 말합니다. 그럼 이 MainActivity 파일은 어디있을까요?

바로 java폴더안에 들어있으며, 처음 상태는 아래와 같이 이루어져있습니다.

```java
import ...

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}
```

그냥 간단하게 R(root) 폴더의 layout 폴더의 activity_main 파일을 View 로 보여주겠다는 의미입니다. 즉 activty_main 파일을 바꿔주면 처음 실행했을 때 화면이 바뀌게 되는 것이죠. 앞서 말했듯이 activity_main(레이아웃 파일) 은 /res/layout 에 들어있으며, xml파일로 이루어져 있습니다. 이제부터 본격적으로 레이아웃을 작성하는 방법에 대해 설명해보겠습니다.

### 레이아웃

/res/layout 폴더에 들어가 activity_main.xml 파일을 열어보겠습니다. 

![이미지2](./assets/images/AS_UI/2.png)

화면을 보면 빨간색으로 된 부분에 **design, text** 라고 쓰여진 부분이 있습니다. design 탭에서는 드래그 앤 드롭으로 원하는 layout 또는 widget들을 추가해줄 수 있습니다. 간단한 화면이거나, 숙달된 경우 클릭으로 쉽게 layout을 짤 수 있고 매우 빠른 작업이 가능하게 됩니다. 나주에 사용할 수 있겠지만 처음에는 모두 코딩을 하시는 것을 추천드립니다. text파트에 들어가면 왼쪽에 수정할 수 있는 코드, 오른쪽에 preview로 화면에 나오게 될 모습이 보여지게 됩니다.

우선 기본으로 주어지는 코드를 살펴보겠습니다.

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout           
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintLeft_toLeftOf="parent"
        app:layout_constraintRight_toRightOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

html과 비슷한 형식으로 </> 로 item들이 지정됩니다. 즉 textview 라는 item은 constraintlayout에 상속이 되는 것이죠. 그렇다면 constraintlayout이 무엇일까요? 이를 알기위해서는 layout의 종류에 대해 알 필요가 있습니다. 

1. constraint layout

   가장 먼저 등장한 constraint layout은 안에 속하는 item들을 상대적으로 배치하는 것에 초점이 맞추어져 있습니다. 그렇다 보니 화면 크기가 달라져도 어느정도 유연하게 대처가 가능한 장접이 있습니다. 그리고 앞에서 보여드린 design탭에서 거의 코딩이 완료가 가능합니다. 세부적인 부분만 수정해주면 되죠. 상대적으로 배치한다는 의미에 대해서 좀더 자세히 설명을 하겠습니다.

   백그라운드 화면 A, 그 안에 두가지 직사각형 모양의 widget 2개 B, C를 배치해보겠습니다. A, B, C는 모두 top, bottom, left 그리고 right 변이 있습니다. 이 변을 이용해서 상대적인 위치를 배치하게 됩니다. 

   안쪽 부분의 TextView를 지우고 아래의 코드를 넣어보겠습니다.

   ```xml
   <Button
       android:id="@+id/B"
       android:layout_width="wrap_content"
       android:layout_height="wrap_content"
       app:layout_constraintBottom_toBottomOf="parent"
       app:layout_constraintStart_toStartOf="parent"
       app:layout_constraintTop_toTopOf="parent" />
   
   <Button
       android:id="@+id/C"
       android:layout_width="wrap_content"
       android:layout_height="wrap_content"
       app:layout_constraintBottom_toBottomOf="parent"
       app:layout_constraintEnd_toEndOf="parent"
       app:layout_constraintTop_toTopOf="parent" />
   ```

   두개의 버튼이 추가되었습니다.  layout_constraint**X**_to**Y**of = "" 라는 문구들이 각 3개씩 추가 되었습니다. 이는 자신의 X속성을 "" 의 Y속성에 상대적으로 배치하겠다는 뜻입니다. 즉 Button의 bottom 은 parent 의 bottom 에 맞추어지는 것이지요 코드에서 보시면 bottom은 bottom 에 top은 top에 맞추어져 있으므로 상대적으로 중간에 위치하게 됩니다. 이는 어떤 화면을 상대하더라도 같은 결과를 내는 장점이 있습니다. 만약 수평으로 여러개의 item 들을 동일한 간격으로 두고 싶다면 chain을 이용할 수 있습니다.

   ```xml
   <Button
       android:id="@+id/B"
       android:layout_width="wrap_content"
       android:layout_height="wrap_content"
       app:layout_constraintBottom_toBottomOf="parent"
       app:layout_constraintEnd_toStartOf="@id/C"
       app:layout_constraintHorizontal_chainStyle="spread"
       app:layout_constraintStart_toStartOf="parent"
       app:layout_constraintTop_toTopOf="parent" />
   
   <Button
       android:id="@+id/C"
       android:layout_width="wrap_content"
       android:layout_height="wrap_content"
       app:layout_constraintBottom_toBottomOf="parent"
       app:layout_constraintEnd_toEndOf="parent"
       app:layout_constraintHorizontal_chainStyle="spread"
       app:layout_constraintStart_toEndOf="@+id/B"
       app:layout_constraintTop_toTopOf="parent" />
   ```

   이렇게 각 아이템에 chainstyle을 넣어주고 start와 end를 지정해주면 가능하게 됩니다. 

   그렇다면 아직 살펴보지 못한 것으로  android:id 부분과 android:layout_width(height) 부분이 있습니다. android:id 는 변수명이라고 생각하시면 됩니다. 위와 같이 @+id/(변수명) 으로 지정해주면 이 item은 그 변수명으로 여러가지 상호작용을 관리해줄 수 가 있습니다. 가장 큰 예로 onClick(사용자가 눌렀을 때 react하는 것) 이 있습니다. layout_width(height) 는 가로 세로 크기라고 쉽게 생각할 수 있습니다. 그 내용으로 "wrap_content" 가 들어있는데 이는 내용의 크기에 맞추겠다는 뜻입니다. 이외에는 "match_parent" 가 있씁니다. 이는 parent 의 height 와 width 에 맞추겠다는 뜻이 됩니다. 아니면 우리가 실제로 값을 집어 넣어 크기를 조절해줄수도 있습니다.

2. linear layout

   linear layout에 대해 설명하기 위해선 앞서 코드를 보겠습니다.

   ```xml
   <LinearLayout
       android:layout_width="match_parent"
       android:layout_height="match_parent"
       android:weightSum="3"
       android:orientation="vertical">
   
       <LinearLayout
           android:layout_width="match_parent"
           android:layout_height="0dp"
           android:layout_weight="1"
           android:orientation="vertical">
       </LinearLayout>
   
       <LinearLayout
           android:layout_width="match_parent"
           android:layout_height="0dp"
           android:layout_weight="1"
           android:weightSum="2"
           android:orientation="horizontal">
   
           <LinearLayout
               android:layout_width="0dp"
               android:layout_height="match_parent"
               android:layout_weight="1"
               android:orientation="vertical">
           </LinearLayout>
   
           <LinearLayout
               android:layout_width="0dp"
               android:layout_height="match_parent"
               android:layout_weight="1"
               android:orientation="vertical">
           </LinearLayout>
   
       </LinearLayout>
   
       <LinearLayout
           android:layout_width="match_parent"
           android:layout_height="0dp"
           android:layout_weight="1"
           android:orientation="vertical">
       </LinearLayout>
   
   </LinearLayout>
   ```

   보지 못한 요소로는 layout_weight, weight_sum, orientation 이 있습니다.

   orientation은 linear layout의 contents들이 적재될 방향을 의미합니다. 즉 vertical 이면 수직으로 배치되게 되고 horizontal 이면 수평으로 배치되게 됩니다. 

   linear layout 에서는 weight_sum 이라는 요소와 layout_weight 라는 요소가 존재하는데요, 이는 상대적인 크기를 이용해 layout의 크기를 조정해줄 수 있게 됩니다. 어떤 A layout 에 포함되는 항목들을 A의 orientation 방향으로 각 layout_weight 만큼 가중치를 두어 크기를 지정하게 됩니다. 즉 A의 orientation이 horizontal 일 때, layout_weight="1" 인 B 와 layout_weight="2" C layout이 있다면 C layout의 width 는 B layout width 의 2배가 되고 이 두 width를 합치면 A의 width가 되는 것이지요. 이 때 weight_sum 을 지정해 안에 포함되는 속성들의 weight들의 총합을 지정해줄 수 있습니다. 위의 코드에서도 볼 수 있듯이 이렇게 지정되게 되는 width 또는 height 값은 "0dp" 로 작성되어 있는 것을 보실 수 있습니다. 자동으로 지정이 되는 것이지요. 그렇다면 위 코드의 결과를 살펴보겠습니다.

   ![이미지3](./assets/images/AS_UI/3.png)

   앞서 설명한대로 가장 외곽이 weight_sum 이 3인 vertical layout입니다. 따라서 내부 layout들은 모두 layout_weight 이 "1" 이기 때문에 동일한 비율로 수직으로 배치되었습니다. 그리고 중간 layout은 horizontal로 설정이 되었고 weight_sum 이 2입니다. 따라서 내부 layout들 모두 layout_weight 이 "1" 이 때문에 동일한 비율로 수평으로 배치된 것을 확인할 수 있습니다. 모든 layout은 linear layout으로 작성이 가능합니다. 따라서 잘 알아두고 활용 가능하도록 만드는 것이 중요합니다.

3. frame layout

   frame layout 또한 매우 중요합니다. frame layout을 제외하면 모든 layout은 각자만의 방법을 통해 안에 들어있는 widget item들을 화면에 표시하게 됩니다. frame layout도 근본은 같지만, 목적이 다릅니다. Frame 이라는 단어는 "액자"를 의미합니다. 즉 이 액자안에 원하는 모습을 선택해서 보여줄 수 있게 하는 것이 Frame layout의 목표입니다. 나중에 좀 더 자세히 설명하겠습니다.

4. table layout

   table 모양으로 layout을 만들어 주는 것입니다. linear layout으로 커버가 가능하지만, weight를 따로 지정해주지 않아도 된다는 장점이 있습니다. <TableLayout 안에 <TableRow 속성을 넣어 view를 만들게 됩니다.

5. grid layout

   grid 모양으로 layout을 만들어 주는 것입니다. table 과 비슷한데요, table은 각 행별로 속성을 넣어주었다고 한다면 gridlayout에서는 미리 크기를 지정해놓고, 이에 맞추어 안에 item들을 원하는 대로 배치해주게 됩니다. colspan, rowspan 이라는 html에서도 자주 사용했던 속성들을 이용해 grid의 부분들을 연결해주기도 합니다.

6. relative layout

   linear layout이 모든 layout을 구사할 수는 있겠지만, 상대적으로 더 복잡해지는 layout이 있기 마련입니다. 이는 relative layout으로 커버 가능합니다. 처음 relative layout을 찾아 보게 된 계기는 linear layout에서 정 중앙이라는 개념이 orientation에 대해서만 적용되어, 아예  layout의 정중앙에 무언 가를 배치하는 것이 불가능해서였습니다. relative layout에서는 constraint layout 과 비슷하게 상대적으로 위치를 지정해줍니다. constraint 보다는 적은 뷰 속성을 가지고 상대적 위치를 지정하기 때문에, 코드는 간결하고 편하지만, constraint가 좀더 유연한 장점은 있습니다.

### 마치며

앞서 앱 프런트 개발의 가장 기초가 되는 Android Studio 를 통해 layout을 기초적으로 작성하는 방법을 설명했습니다. 다음에는 실제 화면을 가지고 layout이 어떻게 구성되어있는지를 분석해 보고, Activity라는 개념에 대해 자세히 설명해보겠습니다. 앱 웹 쪽 개발이 한 번쯤은 해볼만하다는 생각이 들어서 공부하게 되었고 나름 재미가 있어서 글까지 작성하게 되었습니다. 현재는 데이터 사이언스외에 백엔드 심화적인 부분을 공부하며 프런트는 flutter라는 언어를 공부중에 있습니다. flutter는 ios, android를 같이 코딩할 수 있다는 장점이 있습니다. 여력이 된다면 java를 이용한 포스팅이 끝나는 데로 flutter 밑 백엔드도 다뤄보고 싶은 생각입니다. 긴 글 읽어주셔서 감사합니다.

### Reference

[안드로이드 스튜디오 홈페이지 가이드](https://developer.android.com/studio/intro)

