---
layout: post
title:  "Web Worker-Postable 라이브러리 작성기"
date:   2019-04-10 18:00:00
author: orange4glace
tags: [javascript, typescript, webworker]
---

필자는 최근 [Electron](https://electronjs.org/) 플랫폼을 기반으로 한 오픈소스 영상 편집 프로그램을 만들고 있습니다. 프리미어 프로의 저질 짝퉁 버전이라고나 할까요. 웹 생태계의 특성상 새로운 기술들이 빠르게 적용되긴 힘들지만, 동시에 웹에 대한 많은 관심이 수년간 이어지면서 이에 대한 논의와 도입이 활발하게 이루어지고 있는것은 굉장히 즐거운 일이라고 생각합니다. 저같이 뭣도 모르는 녀석도 영상 편집 프로그램 같은걸 만들 생각도 할 수 있게 해주니까요.

이 프로그램에서 핵심이 되는 부분 중 하나는 [OffscreenCanvas](https://developers.google.com/web/updates/2018/08/offscreen-canvas) 이라는 기술입니다. 영상 편집 프로그램은 당연히 영상을 렌더링해서 보여주는 화면이 필요하겠죠. 기본적으로 생각한다면 HTMLCanvasElement의 CanvasRenderingContext2D 또는 WebGLRenderingContext 를 통하여 원하는 그래픽을 렌더링할 수 있습니다. 그러나 문제는 영상 편집 프로그램의 특성 상 Canvas를 렌더링하는데 시간이 오래 걸릴 수 있다는 것입니다. HTMLCanvasElement는 웹의 UI를 담당하는 thread에 속해있기 때문에, Canvas 렌더링에 많은 시간이 소요된다면 그 시간만큼 전체 웹 페이지가 멈출것이고, 사용자는 결국 답답함을 느끼겠죠. 그래서 나온 기술이 앞서 언급한 OffscreenCanvas입니다. OffscreenCanvas는 Canvas를 Main thread에서 분리하여 WebWorker thread에서 별도로 작업할 수 있도록 만들어주는 기술입니다. (OffscreenCanvas에 대한 더 많은 정보는 [여기]((https://developers.google.com/web/updates/2018/08/offscreen-canvas))에서 확인해보세요.)

그런데 문제가 있네요. 렌더링에 필요한 모든 데이터가 Main thread(UI Thread)에 있다는 겁니다. Main thread와 Worker thread는 같은 프로세스 상에 존재하긴 하더라도, Object같은 데이터는 공유할 수 없는 구조를 가지고 있습니다. ([SharedArrayBuffer](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)는 예외) 물론 메모리가 공유가 된다고 하더라도 앞서 언급한 영상 편집 프로그램같은 경우는 데이터를 공유한다고 해서 해결책이 되는것도 아닙니다. 여기서 생각한 해결책은, Main thread에 존재하는 렌더링에 필요한 데이터들을 WebWorker thread에 복사하는 것입니다. Main thread에서 Object가 생성되면 WebWorker에도 메세지를 보내 해당 Object를 생성하고, 마찬가지로 Object의 property가 변경됐을 때도 메세지를 보내 해당 Object의 property를 변경하게 만드는 것이죠.

한눈에 봐도 엄청나게 귀찮은 작업이 될 것 같습니다. 일일이 복사하고자 하는 클래스의 constructor와 property setter에 WebWorker에 메세지를 보내는 코드를 넣어줘야 하니까요. 복사해야하는 클래스가 한두개도 아니고, 꽤 많다면, 엄청나게 귀찮을 뿐더러 하나 빠뜨리기라도 한다면 찾기도 쉽지 않게 되겠죠.
그래서 작성하게 된 라이브러리가 바로 *[worker-postable](https://github.com/orange4glace/worker-postable)* 입니다. 이 라이브러리는 클래스를 Postable하게 만들어, Instance가 생성되거나 Property가 업데이트될 때 마다 자동으로 WebWorker에 메세지를 보내 데이터를 업데이트 해 줍니다.

*worker-postable*은 이전 포스트에서 소개한 [Mobx](https://mobx.js.org/index.html)에 기반을 두고 있습니다. Mobx의 `observable`은 오브젝트의 업데이트를 손쉽게 추적 할 수 있도록 해주기 때문에, 마찬가지로 오브젝트의 업데이트를 추적해야 하는 위 라이브러리에 알맞춤이죠. 라이브러리 자체가 *Mobx*에 기반을 두고있기 때문에, 코드 또한 *Mobx*로 부터 많은 영감을 받았습니다.

*worker-postable*의 기본적인 사용법은 다음과 같습니다.


**Main thread**
```typescript
import { Postable, postable } from 'worker-postable'

@Postable
class Vector2 {
  name: string;
  @postable x: number;
  @postable y: number;
  @postable child: Vector2;

  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
}
```

**Worker thread**
```typescript
import { Posted } from 'worker-postable'

@Posted('Vector2')
class Vector2Posted {
  x: number;
  y: number;
  child: Vector2Posted;
}
```

decorator를 이용하는 부분에서 *Mobx*와 비슷한 부분을 볼 수 있습니다. Main thread code에서 Vector2 클래스의 Class decorator로 `@Postable`을 붙여줌으로써 `Vector2`는 *Postable*한 클래스가 되어 , 객체가 생성될 때 Worker thread로 복사되게 됩니다. 또한 `x`, `y` property 앞에 `@postable`을 붙여줌으로써 `x`, `y` 두 property는 `postable`한 property가 되어, 값이 변경될 때 마다 Worker thread로 복사되게 됩니다.

Worker thread code에서는 동일한 구조를 가지는 `Vector2Posted`라는 클래스를 작성하고, 앞에 `@Posted('Vector2')`를 붙여주었습니다. 이는 해당 클래스가 *Postable*하고, Main thread의 `Vector2` 클래스에 해당한다는 것을 알려줍니다.

**Main thread**
```typescript
import { ref } from 'worker-postable'

let instance = new Vector2(10, 20);
ref(instance);
```

이후 객체를 생성하고, `ref` 함수를 호출합니다. 모든 *postable* 객체는 내부에 *postable-ref count*라는 값을 가집니다. 최초 값은 0으로, `ref` 함수는 해당 객체의 *postable-ref count*를 1 증가시킵니다. `worker-postable`에서 실제로 객체가 복사되는 시점은 객체의 *postable-ref count*가 0에서 1이 되는 순간입니다. 다시 말해, `ref`가 호출되기 전 까지 `instance`는 Worker로 복사되지 않습니다. 이러한 과정이 존재하는 이유는 값을 계산하기 위해 임시적으로 생성되는 객체같이, 복사될 필요가 없는 객체까지 복사되는 불필요한 자원 낭비를 방지하기 위함입니다. 또한 Worker 측의 자원 관리를 위해서도 사용됩니다. *postable-ref count*가 0이 되면, 해당 객체는 Worker thread에서 완전히 deference되고, 이후 Garbage collect 과정을 거치게 됩니다.

객체의 postable-ref count가 변경되면, 필요에 따라 객체 내의 Postable property의 postable-ref count 또한 같이 변경됩니다. 즉 `instance`의 *postable-ref count*가 1 증가했으니, `instance`의 postable property인 `instance.child`의 *postable-ref count* 역시 1 증가하게 됩니다. 따라서 `instance` 를 한번 `ref` 시켰다면 특정 상황이 아닌 이상 `instance.child`를 따로 `ref` 시켜주지 않아도 `instance.child` 역시 업데이트 대상이 됩니다.


그럼 동작 과정을 하나하나 살펴볼까요.
우선 Class decorator인  `@Postable` 는 클래스에 Postable 하게 만듭니다. 

<a name='constructor-decorator'></a>
```typescript
function Postable(__constructor) {
  asPostablePrototype(__constructor.prototype);
  const handler/*:ProxyHandler<T>*/ = {
    construct: function(target, args, newTarget) {
      let instance = Reflect.construct(target, args, newTarget);
      asPostableObject(instance);
      return instance;
    }
  }
  return new Proxy(constructor, handler)
}
```

Class decorator는 해당 Class의 constructor를 인자로 받습니다. 인자를 넘겨받은 함수는 우선 `asPostablePrototype` 함수를 호출하여 해당 constructor의 prototype에 *Postable*과 관련된 정보들을 기록해 넣게 됩니다. 객체가 생성될 때 일어날 동작들을 정의하는 단계라고 보면 되겠죠. 즉 다음과 같은 과정들을 거칩니다.

1) `@postable` property들을 기록
2) *postable-ref count* 가 1이 됐을 때 WorkerThread로 생성 메세지 전송
3) *postable-ref count* 가 0이 됐을 때 WorkerThread로 파괴 메세지 전송

`@postable` property를 기록하는 1)의 과정은 다음과 같습니다. `@postable` property들을 담을 `Set` 을 생성하여 class constructor에 prototype으로 저장해줍니다.
```typescript
function asPostablePrototype(target: any) {
  let propSet;
  if (target.__proto__.hasOwnProperty(POSTABLE_PROPS)) propSet =
      new Set(target.__proto__[POSTABLE_PROPS]);
  else propSet = new Set();
  Object.defineProperty(target, POSTABLE_PROPS, {
    enumerable: false,
    writable: true,
    configurable: true,
    value: propSet
  })

  ... POSTABLE_FUNC_POST_CREATED 함수 정의 부분
```

*Postable* 클래스는 다른 *Postable*한 클래스로부터 상속될 가능성이 있습니다. 따라서 만약 부모 클래스가 *Postable* 클래스라면, 부모의 `Set`을 그대로 복사하여 `@postable` property들을 상속하게 됩니다.
`@postable` property들을 담을 컨테이너를 생성하긴 했지만 실제로 property들을 컨테이너에 저장하진 않았죠. 이 과정은 이후 `@postable` 가 호출될 때 저장하도록 정의됩니다.

다음은 객체가 *postable-ref* 되어 Worker로 데이터를 전송하는 부분입니다. 마찬가지로, 객체의 constructor의 prototype에 함수를 하나 정의해주게 됩니다. 이 함수는 나중에 객체가 `ref` 됐을 때 *worker-postable* 내부 동작에 의해 호출됩니다.

**POSTABLE_FUNC_POST_CREATED 함수 정의**
```typescript
...
Object.defineProperty(target, POSTABLE_FUNC_POST_CREATED, {
  enumerable: false,
  writable: false,
  configurable: false,
  value: function() {
    ... 내부 코드 부분
  }
})

...
```

**POSTABLE_FUNC_POST_CREATED 함수 내부 코드 1**
```typescript
let props: any[] = [];
(this[POSTABLE_PROPS] as Set<string>).forEach(prop => {
  let value = this[prop];
  if (isObject(value)) {
    asPostableObject(value);
    ref(value);
  }
  props.push([prop, serialize(value)]);
})
postMessage({
  type: MessageType.OBJECT_CREATED,
  constructor: this.constructor.name,
  id: this[POSTABLE_ADMINISTRATOR].id,
  props: props
});

...
```

`POSTABLE_FUNC_POST_CREATED` 함수가 호출되는 시점은, 해당 객체의 *postable-ref count*가 0에서 1이 됐을 때 입니다. 즉 Worker thread에 실제로 데이터가 전송되는 시점이죠. 해당 *postable object*가 property로 또다른 *postable object*를 가지고 있다면, Worker thread에서 생성되는 시점에 우선 해당 property에 해당하는 값이 Worker thread 쪽에도 있어야겠죠. 따라서 property 중 *postable object*를 가지고 있다면 해당 객체를 먼저 전송할 수 있게끔 `ref` 하는 과정을 거칩니다.

이후 `postMessage` 를 통해 Worker에 전달할 메세지를 생성합니다. 실제로 넘겨주는 것은 메세지의 타입, 클래스의 이름, *postable object*의 id, 그리고 초기화 property 값들입니다. 메세지를 받은 Worker thread쪽에서는 넘겨받은 클래스의 이름을 가지고 실제 constructor를 가져와, 객체를 생성하고, 해당 객체에 id를 설정한 후, property 값들을 초기화하게 됩니다.

**POSTABLE_FUNC_POST_CREATED 함수 내부 코드 2**
```typescript
... POSTABLE_FUNC_POST_CREATED 함수 내부 코드 1 부분

(this[POSTABLE_PROPS] as Set<string>).forEach(prop => {
  this[POSTABLE_ADMINISTRATOR].observeDisposers.add(observe(this, prop, change => {
    if (change.type == 'update') {
      let oldValue = change.oldValue as any
      if (isObject(oldValue)) unref(oldValue);

      let value = change.newValue as any;
      if (isObject(value)) {
        let postable = asPostableObject(value);
        ref(postable);
      }
      postMessage({
        type: MessageType.OBJECT_UPDTAED,
        object: this[POSTABLE_ADMINISTRATOR].id,
        property: prop,
        value: serialize(value)
      })
    }
  }))
})
```
마지막으로, *Mobx*의 `observe` api를 통해 `@postable` property들을 감시하게 됩니다. `observe` api가 `@observable`의 값이 업데이트될 때 마다 callback을 받을 수 있도록 해주는 덕분에, 해당 callback에 값의 업데이트를 Worker 측에 전달하도록 하면 됩니다.

여기까지 객체가 *postable*에 의해 referenced 되었을 때 동일한 객체가 Worker thread에서 생성되도록 하는 부분을 살펴봤습니다. 객체가 *postable*에 의해 deferenced되어 객체가 소멸되어야 하는 부분은 생성되는 동작과 반대로 동작합니다. `observe` 객체를 파괴해서, 더 이상 값을 업데이트 받지 않도록 해주면 됩니다.

이제 실제 *postable class*의 인스턴스가 생성되었을 때 인스턴스 각각에 대한 *postable* 관련 정보, 예를 들면 *postable id* 또는 *postable-ref count* 등을 기록하는 저장소를 생성하는 과정이 필요합니다. 이는 위에서 한번 살펴본 [`@Postable` constructor decorator](#constructor-decorator)에서, 생성자를 [proxy](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Proxy)로 만듦으로써, 생성자가 호출될 때 마다 해당 인스턴스에 대해 `asPostableObject` 함수를 호출하여 필요한 데이터를 준비하게 됩니다.

마지막으로 `@postable` property들을 이전에 생성했던 *postable class*의 `[POSTABLE_PROPS]`에 기록해야 합니다. 이는 property decorator function으로 아주 간단하게 가능하죠.

**@postable decorator function**
```typescript
function postable(target: any, prop: string) {
  if (typeof target[prop] == 'function') return;
  asPostablePrototype(target);
  // Define property to __proto__
  target[POSTABLE_PROPS].add(prop);
  return observable(target, prop);
}
```

단순히 `target[POSTABLE_PROPS]`(여기서 `target`은 constructor의 prototype)에 해당 property의 이름을 넣어주면 됩니다. 마지막으로 *Mobx*의 `observable` api를 호출함으로써 해당 property를 *observable*하게 만들어줍니다.

*Postable class*와 class에서 생성되는 *Postable object*의 생성과 소멸 과정을 전체적으로 살펴보면 다음과 같습니다.

1) 클래스가 `@Postable` 하게 됩니다. 즉 클래스의 객체가 `ref`, `unref` 될 때 Worker thread에 메세지를 보내도록 셋업되고, `@postable` property들을 저장할 보관소를 생성합니다. 또한 클래스의 객체가 `ref` 됐을 때 `@postable` property에 대한 업데이트 callback을 받도록 만듭니다.
2) 클래스의 property들이 `@postable` 하게 됩니다. 클래스의 *postable property* 저장소 (`[POSTABLE_PROPS]`)에 해당 property를 기록합니다.
3) 클래스의 인스턴스가 생성됩니다. 인스턴스는 `Postable` 관련 정보들을 담을 저장소를 생성합니다. *postable id*, *postable-ref count* 등이 설정됩니다.
4) 클래스의 인스턴스가 `ref` 됩니다. 1)에서의 설정에 의해 Worker에 객체 생성 메세지가 전송되고, `@postable` property들의 업데이트에 대한 callback을 받아 업데이트가 일어날 때 마다 업데이트 메세지를 Worker에 전송합니다.
5) 클래스 인스턴스가 `unref` 됩니다. 1)에서의 설정에 의해 Worker에 객체 파괴 메세지가 전송되고, Worker는 해당 객체를 완전히 *dereference*하고 garbage collect 되도록 합니다.

Javascript는 굉장히 유연한 script language라고 생각합니다. prototype chaining을 기반으로 두고있고, 사용자의 필요에 따라 prototype을 원하는대로 재정의할 수 있기 때문에, 다른 언어같았다면 클래스 멤버 자체를 다른 클래스를 통해 감싸는 등의 과정을 통해 해낼 수 있는 일을, Javascript에서는 객체 자체의 구조는 그대로 두면서도 여러가지 추가적인 기능을 넣을 수 있기 때문이죠.

*worker-postable* 이라는 라이브러리를 작성하긴 했지만, 솔직히 그렇게 대중적으로 사용될 일은 없어보입니다. 애초에 WebWorker 자체가 쓰이는 일이 많이 없으니까요. WebWorker에 많은 데이터를 복사할 일은 더더욱 없을지도 모릅니다. 하지만 최소한 제가 작업중인 프로젝트에서는 아주 유용하게 사용 중입니다. 아직 *worker-postable*의 설계면에서 부자연스러운 부분이 있긴 하지만, 덕분에 아주 작업이 편해졌으니까요. 이 보잘것 없는 라이브러리의 소스가 더 궁금하시다면 [여기](https://github.com/orange4glace/worker-postable)에서 확인해주세요. 이슈와 풀 리퀘스트는 언제나 환영입니다!