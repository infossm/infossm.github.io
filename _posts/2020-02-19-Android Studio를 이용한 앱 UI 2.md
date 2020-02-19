# Android Studio를 이용한 앱 UI 2

## Contents

- 들어가며
- 만들 UI 화면 분석
- 코딩
- 다른 언어들
- 발전 방향

### 들어가며

이전 글에서는 앱 개발에 대한 기본적인 순서와 각 순서별로 간략하게 중요한 것들, 구성되어야 하는 것들에 대해 설명했고, Android Studio tool의 구성에 대해 다루었고, 마지막 부분에 layout에 대한 설명을 간단하게 해주었었습니다. 

이번에는 실제 유명한 화면 몇개를 분석하고 직접 코딩하는 부분을 실제로 다루어 보고 UI 제작에 유의미한 여러가지 tool들을 소개하며, 또 이 분야의 발전 방향성에 대해 간단하게 소개하는 것으로 마무리 지어보겠습니다.

### 만들 UI 화면 분석

저번에 배운 layout 만으로도 우리는 실로 많이 사용하는 어플들의 화면을 직접 만들어 볼 수 있습니다. 그 layout들을 모두 작성하기에 앞서서 그 레이아웃에서 신경써주어야 할 부분들과 어떤 특성들이 있는지, 그리고 전체적인 설계를 미리 그려놓고 시작해야합니다. 

우선 이번에 만들어볼 화면은 바로 잘 알려진 SNS 앱인 인스타그램 어플의 UI입니다. 실제 인스타그램은 Android Studio가 아닌 React-Native 를 이용해서 만들어졌습니다. 리액트 네이티브 툴의 경우 안드로이드/ios를 동시에 지원하는 장점이 있습니다. 하지만 화면 자체는 Android Studio로도 구성이 가능하기에 진행하려고 합니다. 아래는 인스타그램의 화면입니다.

![](C:\Users\USER-PC\Desktop\work\x.jpg)

맨 위에 상태바가 그대로 존재하며, (상태바란 KT 5:06 이 표시된 모바일 기기 위에 뜨는 기본 창을 말한다) 그 아래로 instagram 메인 수평 바가 존재한다. 사진을 바로 찍을 수 있는 카메라 기능과, 라이브, 포스팅 기능으로 연결되는 아이콘들이 존재한다. 그 아래로는 수평인데 넘길 수 scroll이 가능한 바가 존재하고 내가 팔로우 한 사람들로 연결되는 메인 이미지들로 구성되어 있다. 그 밑으로는 팔로우한 사람들의 포스트들을 확인 할 수 있는 수직 scroll 바가 존재한다. 그리고 맨 아래에 고정되어 있는 하단 바가 존재한다.

수평 scroll 바의 경우에 수직 scroll  바 안에 속하기 때문에 가장 크게 layout을 짤 때 그리게 되는 그림은 다음과 같다. 

1. 상태 바
2. main 바
3. 수직 scroll 바
4. bottom navi 바

그렇다면 차근 차근 이 레이아웃들을 구성해보도록 하자.

### 코딩

우선 우리는 bottom navi 바를 만들 것이다. bottom navi 바의 경우에는 하단에 계속 고정이 되어 있어야 한다는 특징이 있다.

먼저 아래에 사용하게될 5개의 아이콘들을 미리 저장해 놓고 사용하기 위해, res폴더 안에 새로운 resource type 이 menu 인 resource directory를 만들어준다. 그 안에 xml 파일을 만들고 아래와 같이 각 아이콘들과 id를 지정해줍니다.

```xml
<?xml version="1.0" encoding="utf-8"?>
<menu xmlns:android="http://schemas.android.com/apk/res/android">

    <item
        android:id="@+id/bottom_home"
        android:enabled="true"
        android:icon="@drawable/ic_home_24px"
        android:title="home"/>

    <item
        android:id="@+id/bottom_search"
        android:enabled="true"
        android:icon="@drawable/ic_search_24px"
        android:title="search"/>

    <item
        android:id="@+id/bottom_post"
        android:enabled="true"
        android:icon="@drawable/ic_add_24px"
        android:title="post"/>

    <item
        android:id="@+id/bottom_follow"
        android:enabled="true"
        android:icon="@drawable/ic_favorite_border_24px"
        android:title="follow"/>

    <item
        android:id="@+id/bottom_mypage"
        android:enabled="true"
        android:icon="@drawable/ic_perm_identity_24px"
        android:title="my page"/>
</menu>
```

이 때 @drawable/ic_~~" 형태로 아이콘을 지정해주게 되는데 이 아이콘들은 우리가 직접 벡터이미지를 넣어주어야 합니다. 벡터 이미지 파일을 구했다면,

File -> New -> Vector Asset 으로 들어가서 벡터 이미지 파일과 그 이름을 지정해주면 위와 같이 모든 코드에서 사용 가능합니다. 화면을 보면 Clip Art 부분에서 내장되어 있는 icon 벡터이미지파일을 사용할 수 도 있습니다. 하지만 필요한 icon이 없는 경우에는 직접 만들거나 구해와야 합니다.

이제 실제 main 화면에 BottomNavigationView를 추가할 차례입니다. BottomNavigationView가 추가 되면 우리는 고른 항목에 따른 화면을 Main 화면에 띄워주어야 합니다. 이는 FrameLayout 을 사용하면 가능합니다. FrameLayout 은 앞서 설명했듯이 액자의 개념이라고 생각을 해주면 쉬운데, 액자안에 우리가 원하는 화면을 띄울 수 있도록 만들어 주는 것 입니다. 우선 아래와 같이 

```xml
<com.google.android.material.bottomnavigation.BottomNavigationView
        android:id="@+id/bottomNavi"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:background="@color/colorWhite"
        app:itemIconTint="#000000"
        app:itemTextColor="#000000"
        app:layout_constraintBottom_toBottomOf="parent"
        app:menu="@menu/bottom_menu" />
```

BottomNavigationView 항목을 추가하고, 그 위에 FrameLayout을 추가해 줍니다.

```xml
<FrameLayout
    android:id="@+id/main_frame"
    android:layout_width="395dp"
    android:layout_height="659dp"
    app:layout_constraintBottom_toTopOf="@+id/bottomNavi"
    app:layout_constraintEnd_toEndOf="parent"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toTopOf="parent">
</FrameLayout>
```

매 layout에 id를 지정해 주어서 나중에 관리하기 쉽게 만들어 주는 것도 중요합니다. 이제 액자틀을 만들었으니, 그 액자에 들어갈 화면들을 또 만들어주어야 합니다. 총 아이콘이 5개이므로, 5개의 layout들을 생성해줍니다. 각 layout의 화면 코드는 다음과 같이 우선 간단하게 통일 시켜 놓겠습니다.

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="N번 화면"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

이 작업이 완성되면 이제 아래 NavigationView의 Icon을 클릭했을 때 원하는 화면을 액자에 띄울 수 있도록 코딩을 해주어야 합니다. 이는 JavaClass파일을 편집해서 만들어낼 수 있습니다. 각 Frag별로 class 파일을 만들어주고 아래 와 같이 코드를 통일 시켜 줍니다.

```java
public class Frag1 extends Fragment {
    private View view;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        view = inflater.inflate(R.layout.frag1, container, false);
        return view;
    }
}
```

public class Frag1, R.layout.frag1 에서 숫자부분은 각 Frag별로 바꾸어주면 됩니다. 코드에서 inflater란 우리가 원래 사용했던 Activity에서 setContentView와 동일개념이라고 생각해주면 됩니다. 하지만 Fragment 속성이기 때문에 위와 같이 편집해줍니다.

그 후 MainActivity.java 에서 이들을 관리해주는 것이 필요합니다.

```java
private BottomNavigationView bottomNavigationView;
private FragmentManager fm;
private FragmentTransaction ft;
private Frag1 frag1;
private Frag2 frag2;
private Frag3 frag3;
private Frag4 frag4;
private Frag5 frag5;
```

위와 같이 각 속성들을 선언해줍니다. ft는 실제 화면의 transaction을 실행해주고 fm이 이를 관리해주게 됩니다. 액티비티가 생성되는 onCreate 안에

```java
bottomNavigationView = (BottomNavigationView) findViewById(R.id.bottomNavi);
```

위 코드를 통해 bottonNavi 뷰를 불러오게 됩니다.

```java
bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_home:
                setFrag(0);
                break;
            case R.id.action_search:
                setFrag(1);
                break;
            case R.id.action_post:
                setFrag(2);
                break;
            case R.id.action_follow:
                setFrag(3);
                break;
            case R.id.action_mypage:
                setFrag(4);
                break;
        }
        return true;
    }
});
```

그리고 ItemSelectedListener를 통해서 각 아이템이 선택되면 화면을 바꿔주는 함수를 실행할 수 있도록 switch문을 이용해서 작성해줍니다. 

```java
frag1 = new Frag1();
frag2 = new Frag2();
frag3 = new Frag3();
frag4 = new Frag4();
frag5 = new Frag5();
setFrag(0);
```

그리고는 초기 화면들을 만들어주고 처음에는 home 화면을 보여주기 위해 setFrag(0)을 실행합니다.

그럼 이제 화면을 바꿔주는 setFrag 함수를 작성해야합니다.

```java
private void setFrag(int n){
    fm = getSupportFragmentManager();
    ft = fm.beginTransaction();
    switch(n){
        case 0:
            ft.replace(R.id.main_frame, frag1);
            ft.commit();
            break;
        case 1:
            ft.replace(R.id.main_frame, frag2);
            ft.commit();
            break;
        case 2:
            ft.replace(R.id.main_frame, frag3);
            ft.commit();
            break;
        case 3:
            ft.replace(R.id.main_frame, frag4);
            ft.commit();
            break;
        case 4:
            ft.replace(R.id.main_frame, frag5);
            ft.commit();
            break;
    }
}
```

setFrag 함수는 위와 같이 구성되어 있습니다. fm, ft를 선언한 후에, replace를 통해 화면을 바꾸고 commit을 해주어야합니다. 이제 실행해보면 아이콘을 선택함에 따라서 화면이 잘 변경되는 것을 확인할 수 있습니다. 이제 우리는 home 화면을 수정해주어야 합니다. home 화면이 현재 ConstraintLayout으로 되어 있으므로 삭제하고 LinearLayout으로 바꾸어 주어야 한다. 우선 layout 의 구조는 아래와 같이 구성을 해주어야한다.

- LinearLayout(vertical)
  - LinearLayout(horizontal) : camera icon / instagram 문구 / live icon / insta Direct icon
  - LinearLayout(vertical)
    - ScrollView(LinearLayout-vertical) : ScrollView 에 하나의 view만이 가능하기 때문에 LinearLayout을 추가해 그 안에 구성물들을 넣어주게 된다.
      - HorizontalScrollView(LinearLayout-horizontal) : follow들의 스토리로 갈 수 있는 image icon이 배치되고, 수평으로 스크롤 가능하다.
      - LinearLayout(vertical)
        - LinearLayout(horizontal) : 각 포스트 위의 상단바 image icon / 이름 / 포스트 추가기능 icon
        - ImageView : 포스트 이미지
      - LinearLayout(vertical) : 위와 같은 포스트 정보가 계속 추가되게 된다.

이렇게 간단하게 구성이 되어있다. 이를 코드로 구현해보면

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:orientation="vertical"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:layout_marginLeft="15dp"
    android:layout_marginTop="60dp"
    android:layout_marginRight="15dp"
    android:layout_marginBottom="15dp"
    android:weightSum="10">
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:weightSum="10"
        android:orientation="horizontal">
        <ImageView
            android:layout_width="0dp"
            android:layout_weight="1"
            android:layout_height="match_parent"
            android:scaleType="centerInside"
            android:src="@drawable/ic_photo_camera_black_24dp">
        </ImageView>
        <RelativeLayout
            android:layout_width="0dp"
            android:layout_height="match_parent"
            android:layout_weight="2">
            <TextView
                android:layout_width="match_parent"
                android:layout_height="match_parent"
                android:gravity="center"
                android:text="Instagram"
                android:textColor="#000000"/>
        </RelativeLayout>
        <LinearLayout
            android:layout_width="0dp"
            android:layout_height="match_parent"
            android:layout_weight="5">
        </LinearLayout>
        <ImageView
            android:layout_width="0dp"
            android:layout_weight="1"
            android:layout_height="match_parent"
            android:scaleType="centerInside"
            android:src="@drawable/ic_live_tv_black_24dp">
        </ImageView>
        <ImageView
            android:layout_width="0dp"
            android:layout_weight="1"
            android:scaleType="centerInside"
            android:layout_height="match_parent"
            android:src="@drawable/ic_send_black_24dp">
        </ImageView>
    </LinearLayout>
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="9"
        android:orientation="vertical">
        <ScrollView
            android:layout_width="match_parent"
            android:layout_height="wrap_content">
            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="vertical">

                <HorizontalScrollView
                    android:layout_width="match_parent"
                    android:layout_height="45dp">
                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="match_parent"
                        android:orientation="horizontal" >

                        <ImageView
                            android:layout_width="81dp"
                            android:layout_height="match_parent"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <ImageView
                            android:layout_width="81dp"
                            android:layout_height="match_parent"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <ImageView
                            android:layout_width="81dp"
                            android:layout_height="match_parent"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <ImageView
                            android:layout_width="81dp"
                            android:layout_height="match_parent"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <ImageView
                            android:layout_width="83dp"
                            android:layout_height="match_parent"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>

                    </LinearLayout>
                    
                </HorizontalScrollView>
                <LinearLayout
                    android:layout_width="match_parent"
                    android:layout_height="450dp"
                    android:orientation="vertical"
                    android:weightSum="10">
                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="0dp"
                        android:layout_weight="1"
                        android:orientation="horizontal"
                        android:weightSum="10">
                        <ImageView
                                android:layout_width="0dp"
                                android:layout_height="match_parent"
                                android:layout_weight="1"
                                android:layout_marginLeft="10dp"
                                android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <RelativeLayout
                                android:layout_width="0dp"
                                android:layout_height="match_parent"
                                android:layout_weight="2">
                                <TextView
                                    android:layout_width="match_parent"
                                    android:layout_height="match_parent"
                                    android:gravity="center"
                                    android:text="hello"/>
                        </RelativeLayout>
                        <LinearLayout
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="6"/>
                        <ImageView
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="1"
                            android:src ="@drawable/ic_more_vert_24px"/>
                    </LinearLayout>
                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="0dp"
                        android:layout_weight="9">

                        <ImageView
                            android:layout_width="match_parent"
                            android:layout_height="match_parent"
                            android:scaleType="fitXY"
                            android:src="@drawable/sample" />
                    </LinearLayout>

                </LinearLayout>
                <LinearLayout
                    android:layout_width="match_parent"
                    android:layout_height="450dp"
                    android:orientation="vertical"
                    android:weightSum="10">
                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="0dp"
                        android:layout_weight="1"
                        android:orientation="horizontal"
                        android:weightSum="10">
                        <ImageView
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="1"
                            android:layout_marginLeft="10dp"
                            android:src ="@drawable/ic_account_circle_black_24dp"/>
                        <RelativeLayout
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="2">
                            <TextView
                                android:layout_width="match_parent"
                                android:layout_height="match_parent"
                                android:gravity="center"
                                android:text="hello"/>
                        </RelativeLayout>
                        <LinearLayout
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="6"/>
                        <ImageView
                            android:layout_width="0dp"
                            android:layout_height="match_parent"
                            android:layout_weight="1"
                            android:src ="@drawable/ic_more_vert_24px"/>
                    </LinearLayout>
                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="0dp"
                        android:layout_weight="9">

                        <ImageView
                            android:layout_width="match_parent"
                            android:layout_height="match_parent"
                            android:scaleType="fitXY"
                            android:src="@drawable/sample" />
                    </LinearLayout>

                </LinearLayout>
            </LinearLayout>
        </ScrollView>
    </LinearLayout>
</LinearLayout>
```

매우 길게 코드가 나온다. 유의해야될 부분들은 아래와 같다.

1. text를 center에 align 시키기 위해서는 LinearLayout 배치에서 이루어질 수 없다. 따라서 weight를 가진 RelativeLayout을 만들어주고 그 안에 TextView를 만든 뒤 gravity="center"로 지정해주면 된다.
2. 이미지를 확대하는 방법은 android:scaleTyle 구문을 조정해주면 된다. fitXY 속성은 화면 꽉차게 만들어준다.
3. HorizontalScrollView / ScrollView 모두 각 하나의 View만을 가질 수 있다. 따라서 자식 View를 LinearLayout으로 만들어 준 후 크기를 match_parent로 지정한 후에, 그 안에 자식들을 만들어 주면 된다.

위는 간단하게 layout을 잡아주는 코드라고 생각해주면 된다. 이렇게 구성이 된 후에, margin들을 이쁘게 조절하고, icon / 글씨체를 수정하고, DB와 연동해서 post들을 불러와 view를 동적으로 추가할 수 있게 바꾸어 주어야 한다. ScrollView에 보여지는 post 수와 HorizontalScrollView에 보여지는 follow들의 수는 정해진 것이 아니기 때문이다.

![](C:\Users\USER-PC\Desktop\work\y.png)

그리고 bottom navigation view의 경우에 item들을 원칙으로 가지고 있으므로 우리가 실제 인스타그램 앱에서 볼 수 있는 bottom bar와는 조금 다르다. 실제 어플 처럼 하단 바를 만들고 싶다면, 그냥 모든 뷰마다 아래에 똑같이 만들 bar 모양의 layout을 박아주고 layout_gravity="bottom"으로 지정해주면 해결된다. 매우 매우 극심한 노가다를 필요로 한다. 

그리고 이 코딩은 화면의 해상도를 고려하지 않았다. 실제 UI개발에 있어서는 화면 해상도를 고려한 작업과, 그 가이드라인에 준수하는 방법을 택해주는 것이 매우 중요하다.

이 과정이 끝나면 이제 home layout을 완성해주었다고 할 수 있다. 이제 또 필요한 layout들을 만들어 주어야 한다. my page 화면 , posting화면 검색화면, live 화면 direct message 화면 등등 한 어플을 만들기 위해서 작성해야할 layout들은 매우 많다. layout을 다 만들었다면 이제 각 layout들을 연결해준다. activity들을 연결하기위해 flow chart를 구성해주고, 그에 맞게 activity를 언제 on 하고 언제 kill 할지를 잘 생각해주어야 한다.

그 후에는 이제 서버와 연결하는 작업을 진행한다. 내 정보를 화면이 바뀔 때마다, 서버에 request해서 받아올 수도 있고, 한 번 받아와서 내 Device에 저장하는 방식을 선택할 수 있다. 이 과정에서는 또 보안 문제가 중요해진다. 이렇게 받아온 정보들을 가지고, layout들에서 지정해놓은 각 id들을 이용해서 java파일을 수정해 알맞는 이미지와 텍스트들을 띄워주게 된다. 이렇게 프런트의 역할은 끝난다고 볼 수 있다.

### 다른 언어들

앞서 instagram은 실제로 react-native를 이용하여 작성되었다고 언급했다. react-native의 경우 android/ios를 동시에 코딩할 수 있다는 장점이 있다. 이는 최근 인기를 몰고 있는 flutter라는 언어도 동일하다. 나는 android-studio / flutter를 다루어 보았는데 개인적으로 flutter에 한 표를 주고 싶다. 좀 더 짜임새 있는 구성과 간결한 코딩이 가능하다는 점이 가장 큰 메리트를 주었던 것 같다. 아무래도 프런트 역할을 맡은 사람의 입장에서는 헷갈리지 않고, 간결한 코딩을 하는 것이 중요하다. 나중에 수정하기 쉽고, 알아보기도 쉽기 때문이다. 이 외에도, 여러가지 언어들이 존재한다.

### 발전방향

사실 앱 부분을 공부하면서 흥미로운 소식을 접했다. Sketch2Code 라고 실제 손으로 그려진 화면을 컴퓨터에게 입력으로 주어졌을 때, AI가 이를 분석하여 모델을 통과해, 진짜 html 화면을 구성하는 코드를 만들어 내는 것이다. 이 분야가 만약 발전이 잘 되어서, UI개발자가 원하는 그림을 잘 표현해 준다면 사실 위에 배운 코딩 방법은 우린 알 필요가 없다. AI에게 맡기면 되기 때문! 아니면 확실하게 반영하지 못하더라도 전체적인 윤곽을 잡아주기만 해도, 훨씬 편리한 코딩이 가능 할 것이라고 본다. 결국 현재 모든 분야에 있어서 가장 중요시 되는 것은 인공지능이라고 본다. 그래도 앱 UI를 공부하는 것은 실로 큰 도움이 되었다. 복잡한 구조를 최대한 간결하게 구성하고, flow chart를 그려보는 것은 코딩을 하는 사람의 입장에서 많은 스킬들을 익힐 수 있다.  여러분도 한 번쯤은 공부해보기를 권유해본다.

