---
layout: post
title: "Rust Crossbeam의 Epoch 기반 메모리 관리"
author: buttercrab
date: 2023-05-20
tags: [rust]
---

## 서론

프로그래밍 언어는 다양한 방법을 통해 메모리를 관리합니다.
이러한 메모리 관리 방법은 크게 두 가지 방법으로 나눌 수 있습니다: GC(Garbage Collection)와 직접 메모리를 관리하는 방법입니다.
GC는 메모리를 관리하는 방법 중 가장 편리한 방법이지만, GC가 동작하는 동안에는 프로그램이 멈추는 단점이 있습니다.
반면 직접 메모리를 관리하는 방법은 GC가 동작하지 않기 때문에 프로그램이 멈추지 않습니다.
하지만 직접 메모리를 관리하는 방법은 GC보다 훨씬 어렵습니다.

Rust는 직접 메모리를 관리하는 방법을 사용합니다.
Rust에서의 메모리 관리 방법은 예전 글인 [Rust의 Borrow Checker](https://infossm.github.io/blog/2022/11/22/rust-borrow-checker/)을 읽으시면 좋을 것 같습니다.

하지만 borrow checker로는 모든 방법의 메모리 관리를 할 수 없습니다.
예를 들어, 다음과 같은 코드를 생각해봅시다.

```rust
let mut v = vec![1, 2, 3];

// in thread 1
let v1 = &mut v;

// in thread 2
let v2 = &mut v;
```

위 코드는 두 개의 스레드에서 하나의 벡터를 동시에 수정하려고 합니다.
이러한 경우 borrow checker는 동시에 하나의 벡터를 수정할 수 없다고 판단하여 컴파일을 실패시킵니다.
하지만 동시에 하나의 벡터를 수정할 수 있는 조건에서는 안전하게 수정이 가능합니다.
이를 위해 Rust는 `unsafe` 키워드를 제공합니다.
`unsafe` 키워드를 사용하면 borrow checker를 우회하여 메모리를 관리할 수 있습니다.

## Rust의 `std::sync`

그래서 Rust에서는 `unsafe` 키워드를 사용하여 메모리를 관리하는 라이브러리를 제공합니다.
이는 `std::sync` 라이브러리입니다.

다음은 `std::sync` 라이브러리를 사용하여 위 코드를 비슷하게 작성한 예시입니다.

```rust
use std::sync::{Arc, RwLock};

let mut v = vec![1, 2, 3];
let mut x = Arc::new(RwLock::new(v));

// in thread 1
let mut x1 = x.clone();
x1.write().unwrap().push(1);

// in thread 2
let mut x2 = x.clone();
x2.write().unwrap().push(2);
```

`Arc`는 `Atomic Reference Counting`의 약자로, 여러 스레드에서 하나의 변수를 참조할 수 있도록 합니다.
`RwLock`은 `Read-Write Lock`의 약자로, 여러 스레드에서 하나의 변수를 읽을 수 있고, 하나의 스레드에서만 변수를 쓸 수 있도록 합니다.

## Rust의 `crossbeam`

하지만 `std::sync` 라이브러리는 한정적인 기능을 가지고 있습니다.
매번 `Arc`와 `RwLock`을 사용하는 것은 매우 불편하며 매번 카운터를 변경하기 때문에 성능이 떨어집니다.
이러한 문제를 해결하기 위해 `crossbeam` 라이브러리가 만들어졌습니다.

`crossbeam` 라이브러리는 `std::sync` 라이브러리와 비슷한 기능을 가지고 있지만, 더욱 편리하고 성능이 좋습니다.
예를 들어 `crossbeam` 라이브러리를 사용하면 다음과 같이 코드를 작성할 수 있습니다.

```rust
use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

let mut x = Arc::new(AtomicCell::new(1));

// in thread 1
let mut x1 = x.clone();
x1.fetch_add(1);

// in thread 2
let mut x2 = x.clone();
x2.fetch_add(2);
```

`crossbeam`은 다양한 기능들이 있지만 이 글에서는 메모리 관리 방식에 대해서만 다루겠습니다.
`crossbeam`의 메모리 관리 방식인 `epoch`을 사용하면 다음과 같이 코드를 작성할 수 있습니다.

```rust
use crossbeam::epoch::{self, Atomic, Owned};
use std::sync::Arc;

let x = Arc::new(Atomic::new(1));

let guard = epoch::pin();

let y = x.load(Ordering::Relaxed, &guard);
*y += 1;

let z = x.load(Ordering::Relaxed, &guard);
*z += 1;
```

여기서 나오는 `epoch`이 무엇일까요?
`epoch`은 `crossbeam` 라이브러리에서 제공하는 메모리 관리 방법입니다.

## Epoch 기반 메모리 관리

Garbage Collector는 다양한 방법을 통해 메모리를 관리합니다.
대표적으로 Reference Counting과 Mark and Sweep이 있습니다.
Reference Counting은 메모리를 참조하는 개수를 세어서 메모리를 관리하는 방법입니다.
앞에서 살펴본 `Arc`는 Reference Counting을 사용합니다.
Mark and Sweep은 메모리를 참조하는 개수를 세지 않고, 메모리를 참조하는 개수를 세는 대신에 메모리를 사용하는 스레드를 추적합니다.

이러한 방법은 각자 장단점이 존재합니다.
Reference Counting은 메모리를 관리하는 방법 중 가장 간단하지만, 메모리를 관리하는 비용이 큽니다.
Mark and Sweep은 메모리를 관리하는 비용이 적지만, 프로그램이 잠시 정지됩니다.

epoch 기반 메모리 관리 방식은 시간을 epoch으로 쪼개서 관리를 합니다.
위의 방식은 메모리에 참조할 때마다 Garbage Collector의 코드가 실행됩니다.
하지만 epoch 기반 메모리 관리 방식은 중간 과정을 보지 않고 epoch이 변할 때만 상황 변화를 보는 방법입니다.

이러한 Garbage Collection은 다음과 같은 원리를 기반으로 합니다:

- 어떤 노드가 garbage가 되면, 이후에도 계속 garbage로 남아있습니다.
- 이로 인해 어떠한 시점에서 garbage라는 것이 확인된 노드는 삭제해도 안전합니다.

그렇기 때문에 특정 시점마다 snapshot을 찍어서 Garbage Collector가 동작하게 됩니다.


## Rust의 `crossbeam::epoch`

`crossbeam::epoch`은 epoch 기반 메모리 관리 방식을 사용합니다.
위에서 살펴본 `epoch::pin()` 함수를 통해 epoch을 관리합니다.

자세히 살펴보겠습니다.

```rust
let x = Arc::new(Atomic::new(1));

let guard: Guard = epoch::pin();

let y: Shared<i32> = x.load(Ordering::Relaxed, &guard);
```

여기서 `Guard`는 `epoch::pin()` 함수의 반환값입니다.
`Guard`는 epoch을 관리하는 구조체입니다.
`Guard`가 `drop`되면 epoch이 끝나게 됩니다.
`crossbeam`은 이 epoch을 통해 메모리를 관리합니다.

우선 `crossbeam`은 global epoch 값을 가지고 있습니다. 
이 값은 modular 3에 해당되는 값을 가지고 있습니다.
`crossbeam`은 global epoch 값이 변할 때마다 메모리를 관리합니다.
각 쓰레드는 local epoch 값을 가지고 있고 global epoch 값이 변할 때마다 local epoch 값을 global epoch 값으로 업데이트합니다.
그리고 각 epoch 마다 global garbage linked list를 관리합니다.
각 epoch 마다 global garbage linked list에 추가되는 노드는 global epoch 값이 변할 때마다 추가됩니다.
그리고 2 epoch 이상 전에 추가된 노드는 global epoch 값이 변할 때마다 삭제됩니다.
1 epoch 이전에 추가된 노드를 삭제하지 않는 이유는 concurrent 하게 garbage collector가 동작하기 때문입니다.
만약 1 epoch 이전에 추가된 노드를 삭제하게 되면, concurrent 하게 garbage collector가 동작하는 도중에 노드가 삭제되어서 문제가 발생할 수 있습니다.

## `crossbeam::epoch`의 API

`crossbeam::epoch`은 크게 다음과 같은 API를 제공합니다.

- [`Owned`](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/struct.Owned.html): 메모리를 소유하는 구조체입니다.
- [`Shared`](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/struct.Shared.html): 메모리를 참조하는 구조체입니다.
- [`Atomic`](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/struct.Atomic.html): `Atomic` 타입입니다.

Garbage Collector에서 가장 중요한 두 가지 연산인 load와 store에 대해서 알아보겠습니다.

### `load`

`load`는 다음과 같이 정의되어 있습니다.

```rust
impl<T> Atomic<T> {
    pub fn load<'a>(&self, order: Ordering, _: &'a Guard) -> Option<Shared<'a, T>>;
}
```

`load`는 `Atomic` 타입의 메모리를 참조하는 연산입니다.
`load`를 하게 되면 `Shared` 타입의 메모리를 반환합니다.
`Shared` 타입은 `Atomic` 타입의 메모리를 참조하는 타입입니다.

### `store`

`store`는 다음과 같이 정의되어 있습니다.

```rust
impl<T> Atomic<T> {
    pub fn store(&self, val: Option<Owned<T>>, ord: Ordering);
}
```

`store`는 `Atomic` 타입의 메모리를 업데이트하는 연산입니다.
`store`를 하게 되면 `Atomic` 타입의 메모리를 업데이트합니다.

## `crossbeam::epoch`의 사용 예시

[이 블로그](https://aturon.github.io/blog/2015/08/27/epoch/)에서 예시를 가져왔습니다.

```rust
use std::sync::atomic::Ordering::{Acquire, Release, Relaxed};
use std::ptr;

use crossbeam::mem::epoch::{self, Atomic, Owned};

pub struct TreiberStack<T> {
    head: Atomic<Node<T>>,
}

struct Node<T> {
    data: T,
    next: Atomic<Node<T>>,
}

impl<T> TreiberStack<T> {
    pub fn new() -> TreiberStack<T> {
        TreiberStack {
            head: Atomic::new()
        }
    }

    pub fn push(&self, t: T) {
        // allocate the node via Owned
        let mut n = Owned::new(Node {
            data: t,
            next: Atomic::new(),
        });

        // become active
        let guard = epoch::pin();

        loop {
            // snapshot current head
            let head = self.head.load(Relaxed, &guard);

            // update `next` pointer with snapshot
            n.next.store_shared(head, Relaxed);

            // if snapshot is still good, link in the new node
            match self.head.cas_and_ref(head, n, Release, &guard) {
                Ok(_) => return,
                Err(owned) => n = owned,
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        // become active
        let guard = epoch::pin();

        loop {
            // take a snapshot
            match self.head.load(Acquire, &guard) {
                // the stack is non-empty
                Some(head) => {
                    // read through the snapshot, *safely*!
                    let next = head.next.load(Relaxed, &guard);

                    // if snapshot is still good, update from `head` to `next`
                    if self.head.cas_shared(Some(head), next, Release) {
                        unsafe {
                            // mark the node as unlinked
                            guard.unlinked(head);

                            // extract out the data from the now-unlinked node
                            return Some(ptr::read(&(*head).data))
                        }
                    }
                }

                // we observed the stack empty
                None => return None
            }
        }
    }
}
```

코드를 보시면 `guard.unlink`를 통해 메모리를 해제하는 것을 볼 수 있습니다.

## 참고자료

- [crossbeam::epoch](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/)
- [crossbeam::epoch의 사용 예시](https://aturon.github.io/blog/2015/08/27/epoch/)

