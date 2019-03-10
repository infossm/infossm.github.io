---
layout: post
title: AMP(Accelerated Mobile Pages) 기여
date: 2019-03-10 23:00
author: KENNYSOFT
tags: [AMP, mathml]
---

## 들어가며

그동안 블로깅을 하면서 추가한 기능들을 AMP로 옮기려고 보니 가장 문제가 되는 것이 외부 스크립트를 쓰는 것들이었다. 크게 두 가지로 구문 강조와 수식이 있는데, 전자는 찾아보니 그냥 GitHub Gist를 쓰라는 이야기가 많아서 일단은 그냥 두기로 했다. 이제 수식을 쓰기 위한 내용이다.

일반 웹페이지에서는 다음과 같이 MathJax를 사용하면 된다.

![MathJax 데모](/assets/images/amphtml-contribution/mathjax-demo.png)

그러나 AMP의 정책상 내부에서 Javascript를 사용할 수 없기에, Extension을 하나 만들고 본 프로젝트에 인정을 받으면 된다. 이에 이미 만들어져 있던 것이 `<amp-mathml>` 이었고 형식은 다음과 같다.

![amp-mathml 문서](/assets/images/amphtml-contribution/amp-mathml-document.png)

## 오류 사항

이제 다음 내용을 렌더링해 보았더니 아래와 같이 되었다.

```html
  <amp-mathml
    layout="container"
    data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]">
  </amp-mathml>
  <p>MathJax <amp-mathml layout="container" inline data-formula="\(x = {-b \pm \sqrt{b^2-4ac} \over 2a}\)"></amp-mathml> MathML <amp-mathml layout="container" inline data-formula="\( \cos(θ+φ) \)"></amp-mathml> AMP</p>
```

![amp-mathml 렌더링](/assets/images/amphtml-contribution/amp-mathml-rendering.png)

Inline 수식이 뭔가 이상하다. 일반 MathJax로 다음 문단을 렌더링하면 이렇게 된다.

```html
  <p>\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]</p>
  <p>MathJax \(x = {-b \pm \sqrt{b^2-4ac} \over 2a}\) MathML \( \cos(θ+φ) \) AMP</p>
```

![MathJax 렌더링](/assets/images/amphtml-contribution/mathjax-rendering.png)

원인을 찾아보니 태그의 inline 속성이 MathJax에서 inline 형태로 만들어 주는 것에 대해 대응이 되지 않고 있었다. 단지 display형 수식을 넣을 것만 생각하고 만든 것으로 보였다. 그렇다면? 고쳐야지.

## 고치기

아무래도 대단위 프로젝트이다 보니 코드만 고치면 되는 것이 아니고 그에 해당하는 문서와 테스트도 같이 맞춰주어야 한다. 그러다 보니 커밋도 여러 개로 나뉘었다.

![커밋 로그](/assets/images/amphtml-contribution/commit-log.png)

```shell
git diff 5dfb0e08b56e77d627195789de4b59d0d656f3d1 373a2608d5f9e35bb8f310768f8d1762b764d467
```

위 명령으로 직접 비교해보면 바꾼 내용이 많지는 않다.

```diff
diff --git a/3p/mathml.js b/3p/mathml.js
index a4b76e072..bd7ed79fe 100644
--- a/3p/mathml.js
+++ b/3p/mathml.js
@@ -46,13 +46,13 @@ export function mathml(global, data) {
 
   getMathmlJs(
       global,
-      'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML',
+      'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML',
       mathjax => {
         // Dimensions are given by the parent frame.
         delete data.width;
         delete data.height;
         const div = document.createElement('div');
-        div.setAttribute('id','mathmlformula');
+        div.setAttribute('id', 'mathmlformula');
         div.textContent = data.formula;
         setStyle(div, 'visibility', 'hidden');
         global.document.body.appendChild(div);
@@ -62,15 +62,20 @@ export function mathml(global, data) {
         mathjax.Hub.Queue(function() {
           const rendered = document.getElementById('MathJax-Element-1-Frame');
           // Remove built in mathjax margins.
-          const display = document.getElementsByClassName('MJXc-display');
-          if (display[0]) {
-            display[0].setAttribute('style','margin-top:0;margin-bottom:0');
-            context.requestResize(
-                rendered./*OK*/offsetWidth,
-                rendered./*OK*/offsetHeight
-            );
-            setStyle(div, 'visibility', 'visible');
+          let display = document.getElementsByClassName('MJXc-display');
+          if (!display[0]) {
+            const span = document.createElement('span');
+            span.setAttribute('class', 'mjx-chtml MJXc-display');
+            span.appendChild(rendered);
+            div.appendChild(span);
+            display = document.getElementsByClassName('MJXc-display');
           }
+          display[0].setAttribute('style','margin-top:0;margin-bottom:0');
+          context.requestResize(
+              rendered./*OK*/offsetWidth,
+              rendered./*OK*/offsetHeight
+          );
+          setStyle(div, 'visibility', 'visible');
         });
       }
   );
diff --git a/examples/amp-mathml.amp.html b/examples/amp-mathml.amp.html
index 26a0b6e97..35b2e466f 100644
--- a/examples/amp-mathml.amp.html
+++ b/examples/amp-mathml.amp.html
@@ -23,9 +23,9 @@
   <h2>Double angle formula for Cosines</h2>
   <amp-mathml
     layout="container"
-    data-formula="\[ \cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ) \]">
+    data-formula="$$ \cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ) $$">
   </amp-mathml>
-  <h2>Inline formula.</h2>
-  This is an example of a formula placed  inline in the middle of a block of text. <amp-mathml layout="container" inline data-formula="\[ \cos(θ+φ) \]"></amp-mathml> This shows how the formula will fit inside a block of text and can be styled with CSS.
+  <h2>Inline formula</h2>
+  <p>This is an example of a formula of <amp-mathml layout="container" inline data-formula="`x`"></amp-mathml>, <amp-mathml layout="container" inline data-formula="\(x = {-b \pm \sqrt{b^2-4ac} \over 2a}\)"></amp-mathml> placed inline in the middle of a block of text. <amp-mathml layout="container" inline data-formula="\( \cos(θ+φ) \)"></amp-mathml> This shows how the formula will fit inside a block of text and can be styled with CSS.</p>
 </body>
 </html>
diff --git a/extensions/amp-mathml/0.1/amp-mathml.css b/extensions/amp-mathml/0.1/amp-mathml.css
index 4003be606..394a233c6 100644
--- a/extensions/amp-mathml/0.1/amp-mathml.css
+++ b/extensions/amp-mathml/0.1/amp-mathml.css
@@ -16,4 +16,5 @@
 
 amp-mathml[inline] {
   display: inline-block;
+  vertical-align: middle;
 }
diff --git a/extensions/amp-mathml/0.1/test/validator-amp-mathml.html b/extensions/amp-mathml/0.1/test/validator-amp-mathml.html
index c1cd42dd7..d373640e9 100644
--- a/extensions/amp-mathml/0.1/test/validator-amp-mathml.html
+++ b/extensions/amp-mathml/0.1/test/validator-amp-mathml.html
@@ -30,7 +30,7 @@
 <body>
   <!-- Valid -->
   <amp-mathml layout="container" data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
-  <amp-mathml layout="container" inline data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
+  <amp-mathml layout="container" inline data-formula="\(x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\)"></amp-mathml>
   <!-- Invalid: unsupported layout value -->
   <amp-mathml layout="responsive" width="10px" height="10px" data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
   <!-- Invalid: missing formula -->
diff --git a/extensions/amp-mathml/0.1/test/validator-amp-mathml.out b/extensions/amp-mathml/0.1/test/validator-amp-mathml.out
index d7accb56d..c27da4f37 100644
--- a/extensions/amp-mathml/0.1/test/validator-amp-mathml.out
+++ b/extensions/amp-mathml/0.1/test/validator-amp-mathml.out
@@ -31,7 +31,7 @@ FAIL
 |  <body>
 |    <!-- Valid -->
 |    <amp-mathml layout="container" data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
-|    <amp-mathml layout="container" inline data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
+|    <amp-mathml layout="container" inline data-formula="\(x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\)"></amp-mathml>
 |    <!-- Invalid: unsupported layout value -->
 |    <amp-mathml layout="responsive" width="10px" height="10px" data-formula="\[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]"></amp-mathml>
 >>   ^~~~~~~~~
diff --git a/extensions/amp-mathml/amp-mathml.md b/extensions/amp-mathml/amp-mathml.md
index 36012b992..3c8a73706 100644
--- a/extensions/amp-mathml/amp-mathml.md
+++ b/extensions/amp-mathml/amp-mathml.md
@@ -55,12 +55,12 @@ This extension creates an iframe and renders a MathML formula.
 #### Example: Double angle formula for Cosines
 
 ```html
-<amp-mathml layout="container" data-formula="\[ \cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ) \]">
+<amp-mathml layout="container" data-formula="$$ \cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ) $$">
 </amp-mathml>
 ```　
 #### Example: Inline formula

-This is an example of a formula placed  inline in the middle of a block of text. `<amp-mathml layout="container" inline data-formula="\[ \cos(θ+φ) \]"></amp-mathml>` This shows how the formula will fit inside a block of text and can be styled with CSS.
+This is an example of a formula of ``<amp-mathml layout="container" inline data-formula="`x`"></amp-mathml>``, `<amp-mathml layout="container" inline data-formula="\(x = {-b \pm \sqrt{b^2-4ac} \over 2a}\)"></amp-mathml>` placed inline in the middle of a block of text. `<amp-mathml layout="container" inline data-formula="\( \cos(θ+φ) \)"></amp-mathml>` This shows how the formula will fit inside a block of text and can be styled with CSS.

 ## Attributes

```

(주: diff 내에 Code fence 문법이 들어가 있는데 이게 escape도 제대로 되지 않아서 임의로 U+3000을 붙였다.)

중요한 내용만 보자면 3p/mathml.js:L65-78에서 `MJXc-display`가 있을 때만 적용되는 것이 아니라 없으면 직접 만들고 적용하도록 했고, extensions/amp-mathml/0.1/amp-mathml.css:L19에서 `vertical-align: middle;` 스타일을 추가했다. 이게 해결 전부고 나머지는 라이브러리 버전 bump, 코딩 스타일 맞추기, 문서/테스트 업데이트 정도다.

## Pull Request<sup>[#](/assets/images/amphtml-contribution/http://github.com/ampproject/amphtml/pull/19436)</sup>

사실 위 작업에 앞서 Pull Request를 염두에 두고 Issue를 먼저 만들었다.<sup>[#](/assets/images/amphtml-contribution/http://github.com/ampproject/amphtml/issues/19420)</sup>

![Issue](/assets/images/amphtml-contribution/issue.png)

열심히 영어로 작문하며 커뮤니케이션을 한 결과 Merge에 성공했다. 처음에는 이상하게 테스트 통과를 못 해서 한참 더 기다려야 했다.

![Pull Request](/assets/images/amphtml-contribution/pr.png)

## 나가며

Merge된 직후이니 가장 최근 커밋에 내가 있는 모습이다.

![amphtml 메인](/assets/images/amphtml-contribution/amphtml-main.png)

완료하고 ampproject의 멤버도 함께 되었다. 하려면 2FA(Two-factor authentication)를 활성화시켜야 하는데 약간 귀찮다.

![ampproject 멤버](/assets/images/amphtml-contribution/ampproject-member.png)

따로 설명은 안 했지만 빌드 환경 구축에도 사실 조금 시간이 걸렸는데, 기여한다고 생각하니 즐거운 시간이었다. 앞으로 구문 강조에 관해서도 기여해 보고 싶다.