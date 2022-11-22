---
layout: post
title: Rust의 Borrow Checker
author: buttercrab
date: 2022-11-22
tags: [Rust]
---

# Rust의 소유권 시스템

대부분의 프로그래밍 언어는 두 가지 중 하나의 방법으로 메모리 관리를 했습니다.
첫 번째 방법은 개발자에게 메모리 관리에 대한 모든 권한을 주고 알아서 관리하라고 하는 겁니다.
C, assembly 등의 언어에서 사용되고 있습니다.
또 다른 방법은 GC(Garbage Collector)를 이용하는 것입니다. 
여기서는 GC의 작동원리에 대해서는 다루지 않지만, 메모리를 관리하는 방법이라고 알아두시면 좋을 것 같습니다.

각 방법에는 장점들과 단점들이 있습니다. 
먼저 첫 번째 방법은 개발자가 작성한 프로그램의 성능을 개발자가 예측할 수 있고 개발자가 작성한 코드만 실행이 돼서 같은 코드를 구현하였을 때 성능이 비교적 뛰어납니다.
하지만 개발자가 모든 메모리를 관리해야 해서 실수할 가능성이 커지고 구현 난이도가 올라갑니다.
두 번째 방법은 개발자가 메모리를 관리할 필요가 없어서 개발자의 구현 난이도가 낮아집니다.
하지만 GC가 코드가 실행될 때 같이 돌아가거나, 몇몇 언어 같은 경우에는 GC가 돌아갈 때 프로그램이 모두 멈춰야 하는 경우도 있어서, 오버헤드가 높아지고 같은 코드를 구현하였을 때 성능이 비교적 느립니다.

예시로 Go 언어는 GC를 사용하는데, 위의 단점에 대한 [블로그 글](https://discord.com/blog/why-discord-is-switching-from-go-to-rust)이 있습니다. 

이러한 두 방법은 성능과 편리함의 싸움이라고 할 수 있습니다.
지금까지는 두 방법은 성능을 선택하냐, 편리함을 선택하냐의 차이였는데요,
그러면 이러한 질문을 할 수 있겠죠.
둘 다를 선택할 수는 없나요?

그래서 나온 새로운 방식이 소유권(Ownership)이라는 방식입니다.
Rust의 소유권 방식은 생소한 개념으로 많은 사람들이 언어의 진입장벽을 높이는 요소라고 하지만, 이해만 한다면 코드를 작성하는게 쉬워집니다.

그래서 소유권이란 무엇일까요?
소유권은 변수가 값을 소유한다는 것입니다.
타 언어처럼 값의 주소를 가지고 있는 것이 아니라 값을 가지고 있지만 그 값을 변수가 소유한 상태입니다.

이 때 소유권에는 규칙이 있습니다.

1. 한 값은 한 변수만 소유할 수 있다.
2. 어떤 변수가 스코프를 벗어나면 그 값은 해제(free, drop)된다.
3. 어떤 변수는 자신의 값을 빌려줄 수 있다.
4. immutable borrow/reference (읽기만 가능한 빌림)은 무한히 빌려줄 수 있다.
5. mutable borrow/reference (쓰기도 가능한 빌림)은 1번만 빌려줄 수 있다.
6. immutable borrow와 mutable borrow를 동시에 빌려줄 수 없다.

```rust
fn main() {
    let a = String::from("hello"); // a는 1을 소유
    {
        let b = a; // a의 값이 b로 이동
        {
            let c = &b; // b의 immutable borrow
            let d = &b; // 여러 번의 immutable borrow
        } // c, d는 해제되어 빌린 것이 반납됨
        {
            let e = &mut b; // b의 mutable borrow
            // immutable borrow가 반납되어 없었으므로 borrow 가능
        } // e는 해제되어 빌린 것이 반납됨
    } // b는 해제(free, drop) 된다
}
```

# 변수의 Liveness

어떤 위치에서 변수가 살아있다는 것을 변수의 liveness, 변수가 live하다고 합니다. 
어떤 변수가 스코프를 벗어나지 않아도 더 이상 쓰이지 않는다면 죽은 것과 마찬가지 입니다.

다음 예제를 볼까요?

```rust
fn main() {
    let a = 1;
    // a는 live
    print(a);
    // a는 live하지 않다.
    // b는 live하지 않다.
    let mut b = 1;
    // 1의 값이 쓰이지 않아서 b는 live하지 않다.
    b = 2;
    // b는 print문에서 쓰일 수 있어서 아직 live하다.
    if some_condition {
        // if문 안으로 들어온 순간 2의 값은 필요가 없어서
        // b는 live하지 않다.
        b = 3;
        // b는 live하다. 
    }
    // b는 live하다.
    print(b);
    // b는 live하지 않다.
}
```

이처럼 현재 값이 후에 쓰일 수 있으면 live한 것이고, 후에 쓰이지 않으면 live하지 않은 것입니다.

# Lifetime

Lifetime은 위처럼 어떤 변수가 살아있는 코드의 범위입니다.
Lifetime은 이름 앞에 `'`를 붙여 표현합니다.
다음 예제를 볼까요?

```rust
fn main() {
    let r;                // ---------+-- 'a
    {                     //          |
        let x = 5;        // -+-- 'b  |
    }                     // -+       |
}                         // ---------+
```

`r`의 lifetime은 `'a`이고 `x`의 lifetime은 `'b`입니다.
이렇게 변수의 lifetime은 변수가 해제되기까지의 범위를 의미합니다.

그렇다면 reference는 어떨까요?
reference의 lifetime은 reference가 사용된 코드의 범위를 나타냅니다.

```rust
fn main() {
    let a = 1;
    let b: &'a u32 = &'b a; // 실제 문법은 아니지만 이해를 돕기 위해 추가했습니다.
    print(b);
}
```

위 코드를 보면 `'b`가 `'a`보다 길게 살아야 한다는 것을 알 수 있습니다.
컴파일러는 이러한 부분을 보면서 lifetime을 추리하게 됩니다.

# NLL (Non-Lexical Lifetime)

Rust는 안전한 프로그램을 만들기 위해서 이러한 borrow checker가 존재합니다.
Halting Problem 등으로 인해 모든 프로그램이 안전한지 아닌지를 몰라서 Rust는 통과되는 프로그램은 안전하게 하자는 철학으로 몇몇 안전한 프로그램이 통과가 안되더라도 통과되는 프로그램은 안전하게끔 하였습니다.
그리고 안전하지만 통과가 안되는 프로그램들을 최대한 줄이는 방향으로 나아가고 있습니다.
그래서 다음 코드를 볼까요?

```rust
fn main() {
  	let mut a = 1;
  	let b = &a;
  	a += 1;
}
```

위 코드는 `b`가 뒤에서 안쓰여서 `a += 1`이 실행될 때 immutable reference가 free되어도 돼서 실제로 문제가 없는 코드입니다.
하지만 lifetime 관점에서 borrow checking을 하게 된다면 두 reference가 겹치는 것으로 판단되어 이 프로그램은 통과되지 않습니다.

그래서 NLL (Non-Lexical Lifetime) 개념이 나오게 되었습니다.
Lifetime을 실제로 뒤에서 쓰이지 않으면 끊어버리는 방법을 채택하여 Lifetime을 세분화하고 더욱 겹치지 않도록 하였습니다.
그래서 위와 같은 더 많은 프로그램을 통과 시킬 수 있게 되었습니다.

그래서 NLL은 다음과 같은 방법으로 프로그램을 변형시킵니다.

```rust
fn main() {
  	let a = 1;
    {
  		let b = &a;
    }
  	a += 1;
}
```

# Polonius

NLL을 이용하면 더 많은 프로그램을 통과시킬 수 있는 것은 맞지만 아직도 몇몇 프로그램은 통과가 안되고 있습니다.
다음 예제를 볼까요?

```rust
fn get_or_insert(
    map: &mut HashMap<u32, String>,
) -> &String {
    match map.get(&22) {
        Some(v) => v,
        None => {
            map.insert(22, String::from("hi"));
            &map[&22]
        }
    }
}
```

[Rust Playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=02dd605963fb98b3a035a655b9148221)에서 코드를 실행해보면 다음과 같은 에러가 뜨게 됩니다.

```rust
error[E0502]: cannot borrow `*map` as mutable because it is also borrowed as immutable
 --> src/main.rs:9:13
  |
4 |     map: &mut HashMap<u32, String>,
  |          - let's call the lifetime of this reference `'1`
5 | ) -> &String {
6 |     match map.get(&22) {
  |           ------------ immutable borrow occurs here
7 |         Some(v) => v,
  |                    - returning this value requires that `*map` is borrowed for `'1`
8 |         None => {
9 |             map.insert(22, String::from("hi"));
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here
```

하지만 생각을 조금만 하게 되면 위의 코드는 문제가 없는 코드임을 알 수 있습니다.
`None` branch로 들어가게 되면 `map.get`으로 빌려온 immutable reference는 live하지 않아서 `map`의 reference는 없게 되어 `map`을 mutable하게 빌릴 수 있게 됩니다.
하지만 NLL은 이를 통과시키지 않습니다.

그래서 이러한 컴파일러에 맞추어서 라이브러리가 [새롭게](https://doc.rust-lang.org/stable/std/collections/struct.HashMap.html#method.entry) 발전하기도 합니다.
하지만 이는 임시방편일 뿐이라 새로운 borrow checker인 [Polonius](https://github.com/rust-lang/polonius)를 고안해냅니다.
이는 위에서 다룬 내용을 새로운 측면으로 바라봅니다.
바로 변수 및 reference의 origin, 즉 원천이 어디인지를 찾는 것입니다.

아래 예제를 보겠습니다.

```rust
fn main() {
  	let mut a = 1;
  	let b = &a;
  	a += 1;
    print(b);
}
```

위 코드는 에러입니다.
`a += 1`을 실행할 때 `b`가 live 하므로 에러가 됩니다.
기존의 borrow checker는 두 reference의 lifetime이 겹쳐서 에러가 나는 것으로 판별합니다.
Polonius는 반대로 작동합니다.
`a += 1`을 실행할 때 `b`가 live 하고 `b`의 origin이 `a`이므로 겹치게 되어 에러가 나는 것으로 판별합니다.

이 방법을 위의 예제에 적용해 볼까요?
원래 에러가 나는 부분인 `map.insert` 부분을 봅시다.
`v`는 live 하지 않고 유일하게 live 한 변수는 `map`으로 겹치는 것이 존재하지 않아 에러가 나지 않습니다.

그럼 어떻게 반대로 작동한다고 해서 통과가 안되던 코드가 통과가 될까요?
반대로 작동을 하게 되면 lifetime이 가장 작은 단위로 쪼개지는 것과 같은 원리로 작동하기 때문입니다.

기존의 borrow checker에서 NLL로 넘어온 것도 lifetime을 더 작게 쪼개서 분석하였는데, Polonius는 이를 더 작은 단위로 쪼개서 분석을 해서 더 많은 코드를 통과시킬 수 있게 되었습니다.

# 더 읽어보면 좋을 내용

- [NLL RFC](https://rust-lang.github.io/rfcs/2094-nll.html)
- [Polonius Book](https://rust-lang.github.io/polonius/what_is_polonius.html)
- [Polonius Youtube Seminar](https://youtu.be/_agDeiWek8w)