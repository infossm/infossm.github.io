---
layout: post
title:  "MobX 내부 살펴보기"
date:   2019-03-09 22:00:00
author: orange4glace
tags: [MobX, React]
---

React는 상태 관리(State management)를 위해 기본적으로 setState 함수를 사용합니다. React를 처음 접하기 시작했다면 몇 가지 컴포넌트들을 만들고, 활용해보면서 React가 props와 state를 변경하는 것 만으로도 인터렉티브한 웹 어플리케이션을 얼마나 쉽고 간단하게 만들 수 있는지에 감탄할겁니다. 

하지만 여러분들의 어플리케이션의 규모가 커짐에 따라 setState만으로는 상태 관리가 충분하지 않다는 것을 느끼게 됩니다. 관리해야 할 컴포넌트가 늘어나고, 자연스럽게 컴포넌트 간의 의존성이 생기게 됩니다. 여러분은 React의 State API를 대체할 무언가를 찾으러 나설 겁니다. 그 중 가장 대표적인 두 라이브러리가 Redux와 **MobX**이죠.
Redux와 MobX를 둘 다 써본 입장으로서, 두 라이브러리 중 승자는 단연 MobX라고 말할 수 있습니다. 까놓고 말해서, Redux는 React의 State API를 사용할 떄 보다 더 최악의 경험을 선사해 주었습니다. Redux의 철학인 Single Source of Truth는 여러 State들을 한 곳에 몰아박는 결과를 만들어냈고, 여러가지 상태 중 컴포넌트에 필요한 상태만을 얻기 위해 복잡하게 구성되어 있는 상태를 다시 컴포넌트 내에서 풀어내야 하고, React Component에서 Redux를 사용하기 위해 명시해줘야 하는 API들은 코드의 양을 늘리는 데에만 기여했습니다.

반면 MobX는 충격적인 경험이었습니다. 마치 눈 앞에서 흑마법을 보는 것 같은 기분이라고 할까요? 컴포넌트의 상태를 관리하기 위해서 해야 할 일은 단지 하나, 변수 앞에 `@observable` 하나만 붙여주면 값이 바뀔 때 마다 알아서 뷰(View)가 스스로 다시 렌더링 되는거죠, 아무런 함수도 호출하지 않았는데!

알고 계셨다고요? 그럼 더 좋습니다. 이 문서는 MobX를 알고 계시는 분들을 위한 문서니까요. 모르고 계셨다면 지금 당장 사용보세요. [이미 많은 문서들이 여러분들을 기다리고 있으니까요.](https://www.google.com/search?q=react+mobx)
이 문서는 MobX를 소개하고 기본적인 사용법을 말하고자 하는 문서가 아닙니다. 이미 충분히 MobX를 활용하고 있지만, 도대체 이 신기한 흑마법이 어떻게 일어나는 지 궁금하다면, 계속 스크롤을 내려주세요.

많은 React의 상태 관리 라이브러리가 그렇듯, MobX는 원래 React를 위해 태어난 라이브러리가 아닙니다. MobX가 React에 잘 녹아들 수 있는 이유는 [mobx-react](https://github.com/mobxjs/mobx-react)라는 third-party 라이브러리가 둘을 이어주기 때문일 뿐이지, (그리고 물론, 그럴 수 있도록 React라는 프레임워크가 잘 만들어졌기 때문이기도 하고요.) 이를 제외하면 두 라이브러리 간의 관계는 전혀 없습니다.

*MobX*의 가장 핵심적인 부분은 `observable`이죠. 이 함수 또는 데코레이션(`@`) 하나로, 여러분들은 특정 객체를 감시할 수 있게 됩니다.
```typescript
class MyObservable {
  @observable myValue = 5;
}
```
우선 `observable`앞에 붙는 *Decorator*, 즉 `@` 에 대해 먼저 알아볼 필요가 있습니다. 물론 observable을 사용하기 위해서 Decorator를 사용하지 않고 함수를 호출하듯이 사용 할 수도 있지만, *React*를 통해 *MobX*를 접했다면 자연스럽게 `@observable` 를 사용했을 것입니다. Decorator는 ES7에 정의되어 있습니다. 아직은 어떠한 브라우저에서도 지원하지 않고, 심지어 정식으로 채택된 Spec도 아니지만, 사실상 많은 곳에서 이미 polyfill을 거쳐 사용되고 있기 때문에 최종 채택 단계에서 누락될 일은 없어보입니다. 

Java를 사용해 보셨다면 적어도 한번은 (`@Override public void..`) 봤을 눈에 익은 패턴인 Decorator가 Javascript에 들어왔습니다. 물론 아직은 정식 스펙이 아니기 때문에 *Babel* 이나 *Typescript* 와 같은 Transpiler를 거쳐 사용해야 합니다. *React* 개발을 위한 환경을 구축했다면 필연적으로 위 두 Transpiler 중 하나를 사용하게 되기 때문에, 이에 대한 인지가 충분하지 않다면 Decorator가 Javascript 기능이라고 착각 할 수도 있지만, **아직은** 아니라는 걸 기억해두세요.

그럼 이 Decorator는 어떻게 사용할 수 있고, 어떤 일을 해낼 수 있을까요? 기본적인 예제를 통해 살펴보는 것이 가장 빠르겠죠. 여기서는 Typescript를 기준으로 설명합니다. Babel을 사용하더라도 같은 Spec을 구현하고 있기 때문에 큰 차이는 없습니다.

```typescript
function decorate(target: any, prop: string): any {
  Object.defineProperty(target, prop, {
    enumerable: true,
    configurable: false,
    get() {
      console.log('get', prop);
    },
    set(value: any) {
      console.log('set', prop, value);
    }
  })
}

class Car {
  @decorate color;
}

let car = new Car();
car.color = 'red';
car.color;

/* Result (console)
set color red
get color */
```

`Car class`에서 생성된 객체의 `color` property에 각각 `set`과 `get`으로 접근했을 때 행동을 재정의 한 것을 볼 수 있습니다. 간단하네요. Decorator가 클래스의 property에 적용되었을 때는 첫 번째 인자(`target`)로 해당 클래스의 `prototype`을, 두 번째 인자(`prop`)로 해당 property의 이름을 넘기게 됩니다. Decorator를 사용하지 않더라도 충분히 같은 내용을 구현 할 수 있습니다. 하지만 Decorator를 사용함으로써 해당 property가 어떤 행동을 하는지 직관적으로 표현할 수 있고, 따로 함수를 호출할 필요가 없게 됩니다. 길게 설명했지만, Decorator는 *Syntax Sugar*에 불과합니다.

Decorator보다 주목해야 할 부분은, `get()` 과 `set(value)` 입니다. `getter`와 `setter`는 ES2015에 포함된 Spec으로, 대부분의 브라우저가 이를 지원하고 있습니다. 클래스의 property에 접근할 때 그 값을 가져오는 대신 미리 정의된 `get()` 또는 `set(value)` 함수를 호출하도록 하는 방법이죠.
감이 오셨나요? `@observable`은 해당 property의 `setter`를 재정의하여 `setter`가 호출될 때 마다 해당 property를 사용하는 *reaction*들(`autorun`, `computed`, `when`)을 실행하게 만듭니다.

`getter, setter` 패턴에서 주의해야 할 점은 `getter setter`로 정의된 property를 **일반적인 property처럼 사용하기 위해선** 별도의 저장소가 필요하다는 점입니다.
```typescript
class Car {
  @decorate color;
}

function decorate(target: any, prop: string): any {
  Object.defineProperty(target, prop, {
    get() { return target[prop]; },
    set(value: any) { target[prop] = value; }
  })
}

let instance = new Car();
console.log(instance.color);
```

이 코드를 실행하면 어떻게 될까요?

```
Uncaught RangeError: Maximum call stack size exceeded
```

콜 스택이 터져버립니다. `instance.color`를 호출하면 `get() { return target[prop]; }`이 호출되고, 해당 함수는 다시 `get() { return target[prop]; }`을 호출합니다. 무한히 반복되겠죠. 따라서 다음과 같이 별도의 저장소에 접근하고자 하는 변수를 따로 기록해야 합니다.

```typescript
function decorate(target: any, prop: string): any {
  Object.defineProperty(target, prop, {
    get() { return target[store][prop]; },
    set(value: any) { target[store][prop] = value; }
  })
}
```

여기서 재미있는 기법이 하나 있습니다. 인스턴스에서 `target[store]`와 같은 임시 저장소에 변수를 저장하기 위해선 우선 `target[store]`를 생성할 필요가 있습니다. 즉 실제론 이렇게 구현되어야 하죠.

```typescript
function decorate(target: any, prop: string): any {
  Object.defineProperty(target, prop, {
    get() {
      if (target.hasOwnProperty('store'))
        Object.defineProperty(target, 'store', {
          value: new Set()
        })
      return target[store][prop];
    },
    set(value: any) {
      if (target.hasOwnProperty('store'))
        Object.defineProperty(target, 'store', {
          value: new Set()
        })
      target[store][prop] = value;
    }
  })
}
```

이렇게 되면 매번 Property에 접근할 때 마다 `target[store]`가 존재하는지 아닌지 체크할 필요가 있습니다. 빈번하게 호출이 일어나는 `getter setter`에서 이러한 `If` 분기는 때로는 치명적일 수 있습니다. 이러한 문제를 다음과 같이 해결 할 수 있습니다.

```typescript
function decorate(target: any, prop: string): any {

  function intiializeStore(target: any, prop: string) {
    if (target.hasOwnProperty('store'))
      Object.defineProperty(target, 'store', {
        value: new Set()
      })
    Object.defineProperty(target, prop, {
      get() { return target[store][prop]; }
      set(value) { target[store][prop] = value; }
    })
  }

  Object.defineProperty(target, prop, {
    get() {
      intiializeStore(target);
      return target[store][prop];
    },
    set(value: any) {
      intiializeStore(target);
      target[store][prop] = value;
    }
  })
}
```

위 코드는 Property에 대해 최초로 `getter`나 `setter`로 접근하게 되면, 해당 인스턴스의 `store`를 생성하고, Property를 다시 정의하게 됩니다. 즉 Property의 최초의 `get`이나 `set`의 호출에 대해서만 `store`가 존재하는지 체크하고, 이후의 접근부터는 `store`가 존재함이 확실하기 때문에 따로 체크해줄 필요가 없게 됩니다. 인스턴스가 생성될 때 마다 Property가 새로 정의된다는 단점이 있지만, Property에 접근할 때 마다 `If` 분기를 거치는 것 보다는 훨씬 나은 해결법이죠.

그렇다면 `autorun`, `computed` 같은 함수들, 즉 `reaction`은 어떻게 동작할까요? 이 함수들은 신기하게도 타겟 함수에 정의된 `observable`들을 자동으로 추적하여 해당 값들이 바뀔 때 마다 자동으로 실행하게 만들어줍니다.

```typescript
autorun(() => {
  // instance.color 값이 바뀔 때 마다 자동으로 호출됩니다
  console.log("Instance color changed to", instance.color);
})
```

앞서 `observable`을 설명할 때 `getter`와 `setter`를 이야기 했었는데요. `setter`는 해당 값의 변경을 알리기 위해 사용된다면, `getter`는 해당 값에 접근하는 `reaction`들을 대해 추적하는 역할을 합니다. 

```typescript
autorun(() => {
  console.log("changed to", instance.color /*여기서 color getter가 호출됩니다*/);
})
```

`reaction`의 대상 함수가 실행되는 와중에, `observable` 한 값에 접근을 하면, 해당 `observable`에 대한 `getter`가 실행되고, `getter`에 정의된 함수는 이후에 다시 해당 `observable`의 `setter`가 호출될 때 다시 해당 `reaction`이 호출되도록 설정합니다.

좋습니다. `ES6 Class`가 어떻게 Observable하게 변할 수 있는지 살펴봤습니다. 그런데 MobX는 Class뿐만 아니라 `Array`, `Set`, `Map` 에 대해서도 Observable 을 지원하고 있죠. 이는 클래스와 비슷하게 해당 클래스들의 프로토타입을 래핑함으로써 구현됩니다. 예를 들어서, `Array.splice`의 경우 `Object.defineProperty(Array.prototype, 'splice', ...)`를 통해 `splice()` 함수 자체를 재정의합니다.

하지만 일부 재정의 될 수 없는 액션들이 있습니다. `Array`의 subscript(`[]`)는 prototype이 아닌 operator이고, 자바스크립트는 _subscript overriding_ 을 지원하지 않기 때문에, MobX 4 까지는 `Observable Array`를 구현하기 위해 실제 `Array`가 아닌 `Array-like Object`를 별도로 생성하여 구현했습니다.

MobX 5.0 부터는 ES6를 지원하도록 변경되면서 앞서 언급한 `Array-like Object`를 따로 생성하지 않게 되었습니다. 대신, ES6에 새로 추가된 기능인 `Proxy`를 사용합니다. `Proxy`는 자바스크립트가 가지는 기본적인 동작(Access, Allocate, Enumerate, Subscript, Function call, ...)에 대해 새로운 동작을 정의할 수 있도록 해주는 기능입니다. ES6에 정식 채택되었고, 이미 모던 브라우저 대부분이 `Proxy`를 사용할 수 있습니다. MobX 5.0은 이 `Proxy`를 사용해 `Array`의 `[]` 동작을 앞서 Class에 적용된 `getter setter`와 비슷한 동작을 정의하여 `Observable Array`를 구현했습니다.

React와 MobX를 연결시켜주는 *mobx-react*의 동작은 간단합니다. `render()` 함수가 `autorun()`으로 감싸지는 것 뿐이니까요. `render()` 함수 내에서 사용된 `observable`이 업데이트 될 떄 마다, `React.Component.forceUpdate()` 함수를 호출하는 것이 전부입니다. 

지금까지 MobX의 내부 동작 과정에 대해 매우 간략하게 알아보았습니다. 좀 더 자세히 알아보고 싶다면 [소스코드](https://github.com/mobxjs/mobx)를 분석해보세요. 작지만 파워풀한 라이브러리가 얼마나 잘 짜여졌는지 볼 수 있을 것 입니다.