---
layout: post
title: "Custom Select Box 만들기 (overflow 문제 해결) - (Nuxt3, Vue3)"
date: 2025-02-27
author: Rn
tags: [nuxt, vue, select]
---

# Intro

안녕하세요. 이번 글에서는 `Nuxt3`에서 `Custom Select Box`를 구현해보려고 합니다. 이때, `overflow`를 허용하는 Component 내부에 존재하는 `Custom Select Box`에 대해 생기는 스크롤 문제를 해결하는 것이 이 글의 목표입니다. 명확한 해결책을 제시해 주는 곳이 없어 이를 해결할 수 있는 방법을 공유하고자 합니다.

또한, `Select Box` 외에도 `Tooltip`과 같이 특정 컴포넌트가 `position: absolute`인 상태로 상대 위치를 계산해야 하는 상황에서도 동일한 방법으로 해결할 수 있습니다.

`Nuxt3`로 작성된 글이긴 하지만, `javascript` 기반으로 작동하는 여러 트릭이 많아서 굳이 `Nuxt3`를 몰라도, 같은 고민을 하고 계시다면 읽어보시는 것을 추천드립니다.

해당 프로젝트의 모든 예시 코드는 [이 프로젝트](https://github.com/thak1411/rn-select-nuxt3-example)에서 확인할 수 있습니다. 직접 실행 및 테스트해 보며 글을 읽는 것을 추천드립니다.

# Select Box

`select box`는 `html`에서 기본적으로 제공하는 `dropdown list`입니다.

```html
<select>
  <option>1</option>
  <option>2</option>
  <option>3</option>
  <option>4</option>
  <option>5</option>
</select>
```
<div>
  <span>example select box</span>

  <select>
    <option>1</option>
    <option>2</option>
    <option>3</option>
    <option>4</option>
    <option>5</option>
  </select>
</div>

하지만 이 `Select Box`의 버튼 디자인은 수정할 수 있지만, `Select Box`를 눌렀을 때 나오는 `Option List`의 디자인은 수정할 수 없습니다. 그 이유는 `Option List`는 browser 및 os에 의존성을 갖는 native element이기 때문입니다.

따라서, `html`에서 기본적으로 제공하는 `select`를 커스텀하여 디자인을 변경할 수는 없고, 별도로 `Select Box`를 구현해야 합니다.

> `color`, `background-color` 등 간단한 옵션은 수정할 수 있지만, 입맛에 맞도록 모든 디자인을 수정할 수는 없습니다. [링크 참조](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/option#styling_with_css)

# Custom Select Box

보통 `select`를 구현하기 위해서 다음 방법을 사용합니다. 아래 코드는 핵심 부분만 나타내며, 전체 코드는 [여기](https://github.com/thak1411/rn-select-nuxt3-example/blob/main/components/rn-select-normal.vue)에서 확인할 수 있습니다.

Custom Select Box 내부에 `dropdown`을 열었다 닫았다 할 `button`과, `dropdown`을 그릴 `Option List`를 만들어 줍니다.

이때, `Option List`가 자리를 차지할 경우 `dropdown`을 열 때마다 다른 컴포넌트의 디자인이 깨질 수 있기 때문에, 자리를 차지하지 않도록 `position: absolute`를 사용합니다. 이때, `absolute component(Option List)`의 상대 위치를 계산하기 위해 (버튼 아래 생겨야 하므로) `rn-select`에 `position: relative`를 사용합니다.

이렇게 구현하면, 기본적인 select box를 만들 수 있습니다.

```vue
<!-- components/rn-select-normal.vue -->
<template>
  <div ref="selectRef" class="rn-select">
    <button class="rn-select-btn" @click="isOpen = !isOpen">
      <slot name="label" :option="props.options[selected]"></slot>
      <i class="rn-select-arrow" :class="{ 'rotate': isOpen }"></i>
    </button>
    <div class="rn-select-option" v-if="isOpen">
      <ul>
        <li v-for="(option, key) in props.options" :key="option" @click="clickOption(key)">
          <slot name="option" :option="option"></slot>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.rn-select {
  position: relative;
}
.rn-select-option {
  left: 0;
  position: absolute;
  top: calc(100% + 4px);
}
</style>
```

```vue
<!-- app.vue -->
<template>
  <div>
    <h1>Rn-Select Test</h1>
    <div class="wrapper">
      <rn-select-normal v-model="selected" :options="items">
        <template v-if="selected == -1" #label>아이템을 선택하세요.</template>
        <template v-else #label="{ option }">{{ option.label }}</template>
        <template #option="{ option }">{{ option.label }}</template>
      </rn-select-normal>
    </div>
  </div>
</template>

<script setup>

const selected = ref(-1)
const items = ref([
  { label: 'Item 1', value: 1 },
  { label: 'Item 2', value: 2 },
  { label: 'Item 3', value: 3 },
])
</script>

<style>
.wrapper {
  width: 200px;
  height: 100px;
  padding: 10px;
  background-color: #ececec;
}
</style>
```

> 위 코드는 제가 주로 사용하는 방식으로, `rn-select-btn`과 `rn-select-option`의 내부는 원하는 대로 수정하셔도 무방합니다.

---

위 예제를 실행하면, 다음과 같은 결과를 확인할 수 있습니다.

![](/assets/images/2025-02-27-select-overflow/rn-select-normal-1.png)

![](/assets/images/2025-02-27-select-overflow/rn-select-normal-2.png)

# Overflow Issue

위 방식대로 사용해도, 대부분의 상황에서는 문제없이 동작합니다. 다만, `Custom Select Box`의 부모 컴포넌트가 `overflow` 혹은 `overflow-x`를 이용해서 가로 스크롤을 사용할 때 문제가 발생합니다.

다음 코드를 살펴보겠습니다. 기존과 동일한 예시이지만, 부모 컴포넌트에 `overflow-x: auto`를 추가했습니다.

```vue
<!-- app.vue -->
<template>
  <div>
    <h1>Rn-Select Overflow Test</h1>
    <div class="wrapper2">
      AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
      <rn-select-normal v-model="selected" :options="items">
        <template v-if="selected == -1" #label>아이템을 선택하세요.</template>
        <template v-else #label="{ option }">{{ option.label }}</template>
        <template #option="{ option }">{{ option.label }}</template>
      </rn-select-normal>
    </div>
  </div>
</template>

<script setup>

const selected = ref(-1)
const items = ref([
  { label: 'Item 1', value: 1 },
  { label: 'Item 2', value: 2 },
  { label: 'Item 3', value: 3 },
])
</script>

<style>
.wrapper2 {
  width: 200px;
  height: 100px;
  padding: 10px;
  overflow-x: auto;
  overflow-y: visible;
  background-color: #ececec;
}
</style>
```

결과물 사진을 보면 아시다시피 가로 스크롤만 설정했음에도 불구하고, `Option List`가 `wrapper box`에 가려지는 모습을 볼 수 있습니다.

![](/assets/images/2025-02-27-select-overflow/rn-select-normal-overflow-1.png)

![](/assets/images/2025-02-27-select-overflow/rn-select-normal-overflow-2.png)

![](/assets/images/2025-02-27-select-overflow/rn-select-normal-overflow-3.png)


> 이 문제가 발생하는 이유는 `position: absolute`로 설정된 컴포넌트는 부모 컴포넌트의 높이 계산에 포함시킬 수 없기 때문입니다. 만약, 높이 계산에 포함시킨다면, 부모 컴포넌트에서 가로 스크롤이 생기는 위치 혹은, 의도치 않은 부모 컴포넌트의 높이 변경이 발생할 수도 있기 때문입니다.

따라서, 이 문제는 `CSS`로 해결이 불가능합니다.

# Solution

`CSS`의 한계로 인해, `CSS` 옵션을 수정하는 것으로는 가로 스크롤을 포함한 `overflow` 문제를 해결할 수 없습니다. 따라서, `javascript`의 개입이 필수적입니다.

이를 해결하기 위한 방법으로 `popper.js`를 사용할 수도 있습니다. 이를 `vue`에서 사용하기 편하도록 만들어 놓은 `vue3-popper` 컴포넌트도 존재합니다.

하지만, 이 라이브러리는 `Nuxt3`을 위해 개발된 것이 아니기 때문에, `Nuxt3`의 `SSR` 방식을 사용하는 프로젝트에 해당 컴포넌트를 사용하면, `Hydration` 문제가 발생합니다. 이 이유는, 초기화 단계에서 `DOM API`를 사용하기 때문입니다. 이를 방지하고자 `<client-only>`를 사용하면, `Hydration` 문제는 해결되지만, 로딩이 되는 찰나의 순간에 깜빡거리는 현상이 발생하게 됩니다.

또한, 라이브러리에서 기본적으로 제공하는 포지셔닝 방식과 기본 디자인이 존재하므로 이를 수정해서 쓰기에는 불편함이 있습니다. 따라서, 라이브러리를 사용하지 않고, 직접 해결하는 방식이 더 효율적일 수 있습니다.

---

문제를 해결하기 위해선 다음과 같은 방법을 사용하면 됩니다.

1. `Option List`가 부모 컴포넌트의 `overflow` 속성에 영향받지 않도록 외부(`Body Tag` 등)에 그려줍니다. - (Portal)
2. `Option List`가 열리는 순간, `Select Box`로부터 상대 위치(혹은 원하는 위치)를 계산해 `Option List`를 해당 위치로 그려줍니다. - (Positioning)
3. `Option List`가 이동해야 할 때마다, 위치를 다시 계산해 이동시켜 줍니다. - (Resize, Scroll)

아래 코드들은 핵심 부분만 나타내며, 전체 코드는 [여기](https://github.com/thak1411/rn-select-nuxt3-example/blob/main/components/rn-select.vue)에서 확인할 수 있습니다.

## Solution Step 1 - Portal

`Portal`이란, 현재 컴포넌트 외부에 컴포넌트를 그려주는 방식입니다. `document.querySelector`처럼 동적으로 위치를 계산해 그려줄 수도 있지만, 이 역시 `Hydration` 문제가 발생할 수 있습니다. 이번 예시에서는 `Hydration`문제가 발생하지 않는, `Body Tag`에 컴포넌트를 그려주는 방식을 사용하겠습니다.

> 만약 `Portal`을 지원하지 않는 환경이라면, 이 역시 `javascript`로 직접 계산해 그려주는 방식을 사용하면 됩니다.

`Nuxt3`에는 `teleport`라는 이름의 컴포넌트로 `Portal` 기능을 사용할 수 있습니다. 다음과 같이 `rn-select-option`을 `teleport`로 `Body Tag`에 그려줍니다.

```vue
<!-- components/rn-select.vue -->
<template>
  <div ref="selectRef" class="rn-select">
    <button class="rn-select-btn" @click="isOpen = !isOpen">
      <slot name="label" :option="props.options[selected]"></slot>
      <i class="rn-select-arrow" :class="{ 'rotate': isOpen }"></i>
    </button>
    <teleport to="body">
      <div class="rn-select-option" v-if="isOpen" :style="optionStyle">
        <ul>
          <li v-for="(option, key) in props.options" :key="option" @click="clickOption(key)">
            <slot name="option" :option="option"></slot>
          </li>
        </ul>
      </div>
    </teleport>
  </div>
</template>
```

## Solution Step 2 - Positioning

위와 같이 `Portal`을 사용하면, `Option List`가 부모 컴포넌트의 `overflow` 속성에 영향받지 않습니다. 하지만, `rn-select-btn`로부터 상대 위치 계산도 불가능한 상태가 됩니다.

따라서, `rn-select-btn`의 위치를 계산해 `Option List`를 그려줘야 합니다.

`Nuxt3` 혹은 `vue3`에서는 `component ref`기능을 사용해서 `DOM`에 접근할 수 있습니다. 이를 이용해 `rn-select-btn`의 위치를 계산해 `Option List`를 그려주면 됩니다.

> 만약, `component ref` 기능을 사용할 수 없다면, `document.querySelector`를 사용해도 무방합니다.

```vue
<!-- components/rn-select.vue -->
<script setup>
const selectRef = ref(null)
const isOpen = ref(false)
const optionX = ref(0)
const optionY = ref(0)

const optionStyle = computed(() => {
  return {
    top: `${optionY.value}px`,
    left: `${optionX.value}px`,
  }
})

const getWindow = element => {
  return element.ownerDocument ? element.ownerDocument.defaultView : window
}

const update = () => {
  if (!isOpen.value) return
  if (!selectRef.value) return 
  
  const rect = selectRef.value.getBoundingClientRect()
  const wnd = getWindow(selectRef.value)

  const distanceFromBodyTop = rect.top + wnd.scrollY
  const distanceFromBodyLeft = rect.left + wnd.scrollX

  optionX.value = distanceFromBodyLeft
  optionY.value = distanceFromBodyTop + rect.height + 4
}
watch(isOpen, update)
</script>
```

> `window.~~`를 사용하지 않고 `getWindow(~~).~~` 함수를 사용하는 이유는, `iframe`이나 `shadow DOM`처럼 여러 문서가 결합된 환경에서도 문제없이 동작하게 만들기 위함입니다. `window`객체를 그대로 사용하게 되면, 해당 요소가 포함된 `window`가 아닌 문맥상 다른 `window`를 사용하게 될 수도 있습니다. 따라서, 올바른 `window`를 사용하기 위해 `ownerDocument.defaultView`를 사용합니다.

`update`함수는 다음과 같은 방식으로 작동합니다.

1. `rn-select`의 `DOM Element`를 통해 `rn-select`의 위치를 계산합니다.
    * `getBoundingClientRect` 함수를 사용하면, `DOM Element`의 화면상 위치와 크기를 계산할 수 있습니다.
    * `window.scrollY`, `window.scrollX`를 사용해 스크롤 위치를 계산합니다.
    * `rn-select`가 화면상 위에서 얼마나 떨어져 있는지 (`rect.top`) + 현재 스크롤이 얼마나 되어 있는지 (`window.scrollY`)를 계산해 `rn-select`의 세로 위치를 계산합니다. 같은 방식으로 가로 위치도 계산합니다.
2. `rn-select`의 위치로부터 `Option List`의 위치를 계산합니다.
    * 위 예제에서는 `rn-select-option`의 가로 위치는 `rn-select`의 왼쪽 끝에 그려줍니다.
    * 세로 위치는 `rn-select`의 아래쪽 끝에 그려주기 위해 `rn-select`의 높이(`rect.height`)를 더해줍니다. `+4`를 해주는 이유는 `rn-select`의 아래쪽에 `margin`을 주기 위함입니다.
3. `optionX`, `optionY`에 계산된 위치를 저장합니다.
4. 계산된 좌표를 `optionStyle`로 반환해 `Option List`의 위치를 결정합니다.
    * Step 1의 `:style="optionStyle"` 참고

> 만약 우측 정렬 혹은 다른 위치에 그려주고 싶다면, 업데이트 수식을 원하는 대로 수정하면 됩니다.

`watch`를 통해 `isOpen`이 변경될 때마다 `update` 함수가 실행되는 구조입니다. 즉, 사용자가 `Option List`를 열 때마다, `rn-select-btn`의 위치를 계산해 `Option List`를 그려주는 것입니다.

이 상태로 예제를 실행시켜 보면, `Option List`가 `Wrapper`의 `overflow` 영향을 받지 않고, `rn-select-btn`의 위치에 따라 `Option List`가 그려지는 것을 확인할 수 있습니다. 하지만, `Wrapper`의 가로 스크롤을 사용하거나, `Window`의 크기가 변경될 때(브라우저 화면 크기 변경으로 인해 주변 컴포넌트의 위치 및 크기가 바뀔 때), `Option List`의 위치가 변경되지 않는 것을 확인할 수 있습니다.

물론, 가로 스크롤을 진행한 이후 `Option List`를 닫았다가 다시 열면, 제대로 그려지긴 하지만, `UI/UX` 관점에서 좋지 않습니다.

![](/assets/images/2025-02-27-select-overflow/rn-select-error.png)

## Solution Step 3 - Resize, Scroll

위 문제를 해결하기 위해선, 동적으로 변하는 주변 컴포넌트 환경에 따라 `Option List`의 위치를 다시 계산해야 합니다.

`Option List`의 위치가 변할 수 있는 상황은 다음과 같습니다.

1. `Window`의 크기가 변경될 때
2. **모든 부모 컴포넌트**의 `Scroll`이 발생할 때

따라서, `Window`의 크기 변경 이벤트(`resize`)와 **모든 부모 컴포넌트**의 스크롤 발생 이벤트(`Scroll`)를 감지해 `Option List`의 위치를 다시 계산해야 합니다.

```vue
<!-- components/rn-select.vue -->
<script setup>
const addListenerToScrollParent = element => {
  if (!element) return
  const isBody = element.nodeName == 'BODY'
  const target = isBody ? getWindow(element) : element

  target.addEventListener('scroll', update)
  scrollParents.value.push(target)

  if (!isBody) {
    addListenerToScrollParent(element.parentNode || element.host)
  }
}
onMounted(() => {
  watch(isOpen, update)

  getWindow(selectRef.value).addEventListener('resize', update)
  addListenerToScrollParent(selectRef.value.parentNode || selectRef.value.host)
})
</script>
```

위 코드는 다음과 같이 동작합니다.

1. component가 mount되면, `isOpen`이 변경될 때마다 `update` 함수가 실행되도록 `watch`를 등록합니다.
2. `Window`의 크기 변경 이벤트(`resize`)를 감지해 `update` 함수가 실행되도록 `getWindow(..).addEventListener('resize', update)`를 사용합니다.
3. `모든 부모 컴포넌트`의 `Scroll` 이벤트를 감지해 `update` 함수가 실행되도록 `addListenerToScrollParent` 함수를 사용합니다. 재귀적인 구조를 가지며, `rn-select`의 부모 컴포넌트부터 루트 문서 객체까지 `scroll` 이벤트를 등록해 줍니다.
    * `Body Tag`가 아닐 경우 해당 컴포넌트에 `scroll` 이벤트를 감지해 `update` 함수가 실행되도록 합니다.
    * `Body Tag`일 경우, `Window`에 `scroll` 이벤트를 감지해 `update` 함수가 실행되도록 합니다. (`Body Tag`는 `DOM`의 최상위 요소이지만, `scroll` 이벤트는 보통 `window`에서 발생합니다.)
    * 부모 컴포넌트로 접근하기 위해서 `element.parentNode || element.host`를 사용하는 이유는 `shadow DOM` 환경에서 부모 컴포넌트에 접근하기 위함입니다. `element.parentNode`가 존재하지 않는 경우 (`shadow DOM` 환경) `element.host`를 사용하도록 하는 코드입니다.
    * `scrollParents` 배열에 감지한 부모 컴포넌트를 저장합니다. 저장하는 이유는 `component unmount` 시, 이벤트 리스너를 제거하기 위함입니다.

위와 같이 사용하면, `Window`의 크기 변경 혹은 `모든 부모 컴포넌트`의 `Scroll` 이벤트가 발생할 때마다 `Option List`의 위치가 다시 계산되어 제대로 그려지는 것을 확인할 수 있습니다.

![](/assets/images/2025-02-27-select-overflow/rn-select-1.png)

![](/assets/images/2025-02-27-select-overflow/rn-select-2.png)

## Solution Step 4 - Optimization

위 코드로 문제를 해결할 수 있지만, 환경에 따라 다음과 같은 퍼포먼스 이슈가 존재합니다.

1. 스크롤 시 스크롤이 늦게 반응하는 현상
2. `Option List`가 이동할 때 프레임 드랍(화면이 툭툭 끊기며 이동하는 것처럼 보이는 현상)이 일어나는 현상

### Optimization 1 - passive

이슈를 해결하기에 앞서 왜 스크롤 반응이 늦어지는 현상이 발생하는지 알아보겠습니다.

`javascript`는 단일 스레드 방식이라고 알려져 있지만, 사실은 숨겨진 스레드가 존재합니다. 해당 글에서는 `Main thread`와 `Compositor thread`에 대해서만 간략하게 소개하고 넘어가겠습니다.

`Main thread`는 우리가 일반적으로 인지하고 있는 `javascript`의 실행을 담당하는 담당하는 스레드입니다. `Compositor thread`는 화면을 구성하는데 필요한 정보를 계산하는 스레드입니다.

브라우저에 클릭, 사이즈 변경, 스크롤 등 **입력**을 하게 되면, `Compositor thread`가 `Main thread`에게 해당 이벤트를 전달합니다. `Main thread`는 해당 이벤트를 처리하고, `Compositor thread`에게 화면을 다시 그리라고 요청합니다.

`Compositor thread`는 `Main thread`로부터 정보를 받을 때까지 `lock`에 걸린 상태로 대기하게 됩니다. 즉, `Main thread`의 작업이 끝날 때까지 화면을 그리지 않고 대기하게 됩니다.

대기하는 이유는, `Main thread`에서 `preventDefault`를 사용해 이벤트를 중단시킬 수 있기 때문입니다. 따라서 `Main thread`가 응답하기 전까지, `Painting`작업을 해야 하는지, 혹은 하면 안 되는지 모르기 때문에 대기해야 합니다.

`Scroll`이벤트는 스크롤을 하는 매 순간마다 발생하는 이벤트로, 짧은 순간에 많은 양의 이벤트가 발생하게 됩니다. 따라서 `Main thread`가 해야 할 일이 한 번에 많이 쌓이게 되고, 응답이 점점 늦어지면서, 스크롤 반응이 늦어지는 현상이 발생하게 됩니다.

이를 막기 위해 `passive` 옵션을 사용할 수 있습니다. `passive` 옵션을 사용하면, `preventDefault()`(이벤트 기본 동작 취소)를 사용할 수 없게 되어, `Compositor thread`가 `Main thread`의 응답을 기다리지 않고 바로 화면을 그리게 됩니다. 즉, 대기하는 시간 없이 화면을 그려 반응이 느려지는 현상을 막을 수 있습니다.

다음과 같이 `addEventListener`의 세 번째 인자로 `{ passive: true }`를 사용하면 됩니다.

```javascript
element.addEventListener('scroll', update, { passive: true })
```

> `Scroll` 이벤트는 취소가 불가능한 이벤트입니다. 따라서, `Scroll` 이벤트에 대해 기본적으로 `passive`를 사용하도록 설정하는 브라우저가 있긴 합니다. 하지만, 모든 브라우저가 해당 옵션을 지원하는 것은 아니므로, `passive` 옵션을 명시적으로 사용하는 것이 좋습니다.

### Optimization 2 - rAF(requestAnimationFrame), Debounce

이슈를 해결하기에 앞서 왜 프레임 드랍 현상이 발생하는지 알아보겠습니다.

프레임 드랍을 이해하기 위해선 브라우저 프레임에 대해 먼저 알아야 합니다.

브라우저는 화면을 구성하는데 프레임 단위로 화면을 그려줍니다. 모니터 주사율에 따라 프레임이 달라지지만, 대부분의 모니터는 60Hz(1초에 60 프레임)입니다. 즉, 1초에 60번 화면을 그려줍니다. (만약, 144Hz 모니터라면 1초에 144번 화면을 그려줍니다.)

60Hz 모니터 기준으로 1 프레임은 약 16.67ms(1000ms / 60)입니다. 즉, 16.67ms마다 새로운 화면을 그려줍니다.

이때, 다음과 같이 코드가 작동했다고 가정해 보겠습니다. (단순 예시일 뿐이므로, 정확한 시점은 무시하고 흐름만 확인하면 됩니다.)

1. 0ms 시점에 새로운 화면을 그렸습니다.
2. 10ms 시점에 새로운 화면을 그리기 위해 정보를 업데이트하는 작업을 시작했습니다. (0ms에 시작되어야 할 이벤트 처리가 `javascript`의 `event task queue`가 다른 작업으로 쌓여있어 업데이트가 늦었다고 가정)
3. 16.67ms 시점에 두 번째 화면을 그렸습니다.
4. 20ms 시점에 정보 업데이트 작업이 끝났습니다.
5. 33.34ms 시점에 세 번째 화면을 그렸습니다.

다음과 같이 두 번째 프레임에 그려져야 할 정보가, 세 번째 프레임에 그려지며 1 프레임이 누락되는 현상이 발생하게 됩니다. 이를 프레임 드랍이라고 합니다. 또한, `javascript`의 `event task queue`가 충분히 많이 쌓여있다는 가정 하에는 1 프레임 이상이 누락될 수도 있습니다.

이를 해결하기 위해선, `rAF(requestAnimationFrame)`를 이용할 수 있습니다. `rAF`란 브라우저가 매 프레임 시작 시 지정한 함수가 우선적으로 실행되도록 보장해 주는 함수입니다. 즉, `rAF`를 사용하면, 위와 같이 `event task queue`가 쌓여있어도, 매 프레임마다 지정한 함수가 먼저 실행되도록 보장해 줍니다.

> 물론, `rAF`사용해도 실행 시간을 무조건적으로 보장할 수는 없습니다. 큐가 밀려 프레임 드랍이 발생하기도 합니다. 하지만, `rAF`를 적절히 사용하면, 프레임 드랍이 발생할 확률을 대폭 줄일 수 있습니다.

`rAF`가 우선 실행을 보장할 수 있는 이유는 `javascript`의 비동기 처리를 담당하는 `event queue`에 존재하는 우선순위 개념 때문입니다. 크게 세 가지 큐가 존재합니다.

1. `Microtask Queue` - `Promise`, `MutationObserver` 등이 사용하는 큐로, `Microtask`가 존재할 경우, `Microtask`가 가장 먼저 실행됩니다.
2. `Animation Queue` - `rAF`가 사용하는 큐로, `Microtask Queue`가 비어있고, `Animation`이 존재할 경우, `Animation`이 실행됩니다.
3. `Task Queue` - `setTimeout`, `setInterval` 등이 사용하는 큐로, `Microtask Queue`, `Animation Queue`가 비어있고, `Task`가 존재할 경우, `Task`가 실행됩니다.

> `Microtask Queue`와 `Animation Queue`는 한 번 작업을 시작하면, `Queue`가 빌 때까지 같은 `Queue`에서 작업을 계속합니다. 반면, `Task Queue`는 한 번 작업을 진행한 뒤, 다음 작업을 진행하기 전에 다른 `Queue`를 확인합니다.

> 즉, `Task Queue`에 작업이 10개가 밀려있고, 가장 최근 작업에서 `Microtask`를 생성했다면, 해당 작업이 끝난 후, `Microtask`가 먼저 실행됩니다. 그리고, `Microtask`가 끝난 후, `Task Queue`에 있는 나머지 작업이 실행됩니다.

> 반면, `Animation Queue`에 작업이 10개가 밀려있고, 가장 최근 작업에서 `Microtask`를 생성했다고 해도, 해당 작업이 끝난 후, `Animation Queue`에 있는 나머지 작업이 먼저 실행됩니다. 즉, `Animation Queue`에 작업이 남아있을 경우, `Microtask`가 생성되어도 `Animation Queue`에 있는 작업이 먼저 실행됩니다. 이후 `Animation Queue`가 비어있을 때, `Microtask`가 실행됩니다.

따라서, 다음과 같이 `rAF`를 사용하면 프레임 드랍을 방지하며 `Option List`의 위치를 설정하는 코드를 최적화할 수 있습니다.

```vue
<!-- components/rn-select.vue -->
<script setup>
const throttledUpdate = ref(null)
const addListenerToScrollParent = element => {
  if (!element) return
  const isBody = element.nodeName == 'BODY'
  const target = isBody ? getWindow(element) : element

  target.addEventListener('scroll', throttledUpdate.value, { passive: true })
  scrollParents.value.push(target)

  if (!isBody) {
    addListenerToScrollParent(element.parentNode || element.host)
  }
}
onMounted(() => {
  throttledUpdate.value = () => requestAnimationFrame(update)
  watch(isOpen, throttledUpdate.value)

  getWindow(selectRef.value).addEventListener('resize', throttledUpdate.value, { passive: true })
  addListenerToScrollParent(selectRef.value.parentNode || selectRef.value.host)
})
</script>
```

하지만, `rAF`만 사용하면 한 가지 문제가 더 발생합니다.

만약, 한 프레임 사이에 2번 이상 `scroll` 혹은 `resize` 이벤트가 발생했다고 가정해 보겠습니다. 이 경우, `Animation Queue`에 화면을 업데이트하는 작업이 2개 이상 쌓이게 됩니다. 그렇게 되면, 다음과 같은 상황이 발생합니다.

* `한 프레임 내에 모든 이벤트를 처리할 수 있는 경우`: 화면이 업데이트되기 전, 컴포넌트의 위치를 두 번 바꾸기 때문에 의미 없는 작업이 발생합니다.
* `한 프레임 내에 모든 이벤트를 처리할 수 없는 경우`: 앞에서 소개한 이유와 동일하게 프레임 드랍이 발생할 수 있습니다.

즉, 한 프레임 내에 이벤트가 여러 번 발생하게 된다면, 의미 없는 일(화면에 반영되지 않을 업데이트 작업)을 처리하다가 프레임 드랍이 발생할 확률이 높아집니다.

따라서, 한 프레임에 업데이트가 최대 한 번만 발생하도록, `Debounce`를 사용하면 됩니다. `Debounce`란, 일정 시간 동안 함수가 실행되는 횟수를 제한하는 기법입니다.

일반적으로는 100ms, 1000ms처럼 시간을 설정해 해당 시간 내에 최대 한 번만 실행하도록, `setTimeout` 등과 함께 사용합니다. 하지만, 이번 경우는 `rAF`와 함께 사용해야 하므로, 특수한 형태라고 할 수 있습니다.

```vue
<!-- components/rn-select.vue -->
<script setup>
const debounce = fn => {
  let lock = false
  return () => {
    if (lock) return
    lock = true

    if (window.Promise) {
      window.Promise.resolve().then(() => {
        lock = false
        fn()
      })
    } else {
      setTimeout(() => {
        lock = false
        fn()
      }, 1)
    }
  }
}
const addListenerToScrollParent = element => {
  if (!element) return
  const isBody = element.nodeName == 'BODY'
  const target = isBody ? getWindow(element) : element

  target.addEventListener('scroll', throttledUpdate.value, { passive: true })
  scrollParents.value.push(target)

  if (!isBody) {
    addListenerToScrollParent(element.parentNode || element.host)
  }
}
onMounted(() => {
  const debouncedUpdate = debounce(update)
  throttledUpdate.value = () => requestAnimationFrame(debouncedUpdate)
  watch(isOpen, throttledUpdate.value)

  getWindow(selectRef.value).addEventListener('resize', throttledUpdate.value, { passive: true })
  addListenerToScrollParent(selectRef.value.parentNode || selectRef.value.host)
})
</script>
```

`debounce` 함수는 다음과 같이 동작합니다.

1. `closure`를 사용해 `lock` 변수를 생성합니다.
2. `closure`로 반환되는 함수는 다음과 같이 동작합니다.  
    1. `lock`이 `true`일 경우 함수 실행을 무시하고 종료합니다.
    2. `lock`이 `false`일 경우 `lock`을 `true`로 변경합니다.
    3. `Microtask`를 지원한다면, `Microtask`에, 지원하지 않는다면, `Task`에 파라미터로 넘겨받은 함수를 등록합니다.
    4. 실제 `task`가 실행될 때 `lock`을 `false`로 변경합니다.

따라서, `debounce`를 적용한 함수를 두 번 이상 연달아 호출하게 된다면 다음과 같은 흐름으로 동작합니다.

1. 첫 번째 `debounce` 함수가 실행되며, `lock`을 `true`로 변경합니다.
2. 이후 `Queue`에 작업을 등록하고, 함수가 종료됩니다.
3. 두 번째 `debounce` 함수가 실행되며, 이미 `lock`이 `true`이므로 함수 실행을 무시하고 종료합니다.
4. 세 번째 이상 함수도 동일하게 `lock`이 `true`이므로 함수 실행을 무시하고 종료합니다.
5. `Task`가 끝났으므로 `Queue`에 등록된 업데이트 함수가 실행되며, `lock`을 `false`로 변경합니다.

이를 `rAF`와 함께 적용한다면 다음과 같이 작동하게 됩니다.

1. 한 프레임 사이에서 `scroll` 혹은 `resize` 이벤트가 여러 번 발생해, `Animation Queue`에 `debouncedUpdate` 함수가 여러 번 쌓이게 됩니다.
2. `Animation Update` 시점이 되어, `Animation Queue`에서 첫 번째 작업(`debouncedUpdate`)을 실행합니다.
3. `debouncedUpdate`에서 `lock`을 `true`로 변경하고, `Microtask`에 업데이트 함수를 등록합니다.
4. 이후 첫 번째 작업이 종료되고, 아직 `Animation Queue`에 작업이 남아있으므로, 두 번째 작업이(`debouncedUpdate`) 실행됩니다.
5. 두 번째 작업에서는 이미 `lock`이 `true`이므로 함수 실행을 무시하고 종료합니다.
6. 세 번째 이상 작업도 동일하게 `lock`이 `true`이므로 함수 실행을 무시하고 종료합니다.
7. `Animation Queue`에 작업물이 남아있지 않으므로, `Microtask Queue`에 등록된 업데이트 함수 하나가 실행되며, `lock`을 `false`로 변경합니다.

따라서, `rAF`와 `Debounce`를 사용하면, 한 프레임 내에 업데이트가 최대 한 번만 발생하도록 할 수 있습니다. (이 순서로 실행되는 이유는 `rAF` 챕터에서 소개했으므로 자세한 설명은 생략하겠습니다.)

> \[주의\] 앞서 소개했듯 `Event Queue`의 특성 때문에, `Debounce`를 사용할 때 `Microtask`로 작업을 등록하는 것이 프레임 드랍을 방지하는데 유리합니다. 하지만, `Microtask`를 지원하지 않는 브라우저도 존재하므로, 지원 여부를 검사해서 지원하지 않는 브라우저의 경우 `setTimeout`을 사용해 `Task Queue`에 등록할 수 있도록 구현해야 합니다.

> \[주의\] `setTimeout`을 사용할 때 `0ms`로 설정하면, 대부분 브라우저에서는 바로 `Task Queue`로 작업을 등록하지만, 일부 브라우저에서는 별도 비동기 `task`로 분리하지 않고 바로 실행하는 경우가 존재합니다. (대표적으로 `Edge`와 `Firefox`가 있습니다.) 따라서, 이를 구분해서 `setTimeout`을 `0ms` 혹은 `1ms`로 설정하거나 안전하게 처리하기 위해 무조건적으로 `1ms`를 사용하는 것을 권장합니다.

# Conclusion

위와 같이 여러 트릭을 활용해서 `Custom Select Box`를 구현하는 방법을 알아봤습니다. 굳이 `Select Box`가 아닌 다른 `Absolute Position Component`를 구현하거나 프레임 단위 작업을 해야 하는 경우, 위와 같은 방법을 사용하면 됩니다. 또한, 경우에 따라 필요한 방식대로 응용해서 적용해 보는 것을 추천합니다.